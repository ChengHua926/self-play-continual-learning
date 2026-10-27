"""Many copies of the fixed world at once, on the GPU, with MuJoCo Warp.

The same scene as sim.Sim (MuJoCo 3.17 on the CPU) and the same controller (controller.py, in torch),
for thousands of copies, each with its own start and its own actions. In each of the 25 physics steps
of a control step, MuJoCo Warp first works out the positions, contacts, mass matrix and forces
(step1); the controller reads them and writes the motor commands; then MuJoCo Warp finishes the
physics step (step2). Warp and torch share the arrays, without copying, and work on one GPU stream.

For speed, each of these pieces is recorded once as a GPU graph and then replayed: without that,
launching Warp's many small programs one by one costs about 5 times the physics itself.

MuJoCo Warp computes in 32-bit floats, where MuJoCo on the CPU uses 64-bit; the controller here
works in 64-bit. test_gpu.py measures how far the copies land from sim.Sim, and gpu_speed.py how fast.
"""

import mujoco_warp as mjw
import torch
import warp as wp

import controller
from sim import PHYSICS_STEPS, Sim

CONTACTS_PER_COPY = 256         # room for contacts in each copy; MuJoCo Warp drops any beyond it (128 overflowed)
CONSTRAINTS_PER_COPY = 1024     # and for constraint rows (random starts reached 543)
PARKED = (10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0)     # where robosuite parks the unused round nut


class GpuWorld:
    def __init__(self, copies, device="cuda:0"):
        self.layout = Sim()             # one CPU copy, for the names and indices
        model, layout = self.layout.model, self.layout
        self.copies, self.device = copies, device
        self.m = mjw.put_model(model)
        self.d = mjw.put_data(model, layout.data, nworld=copies, nconmax=CONTACTS_PER_COPY, njmax=CONSTRAINTS_PER_COPY)
        self.stream = wp.get_stream(device)                 # Warp's stream, which torch uses too
        self.torch_stream = wp.stream_to_torch(self.stream)
        view = wp.to_torch
        self.qpos, self.qvel, self.ctrl, self.time = view(self.d.qpos), view(self.d.qvel), view(self.d.ctrl), view(self.d.time)
        self.hand_position = view(self.d.site_xpos)[:, layout.hand_site]
        self.hand_orientation = view(self.d.site_xmat)[:, layout.hand_site]
        # For the hand-written ways (gpu_ways.py). Read between control steps, these are one 2 ms physics
        # step old (positions are worked out at the start of each physics step).
        self.handle_position = view(self.d.site_xpos)[:, model.site("SquareNut_handle_site").id]
        self.nut_orientation = view(self.d.xmat)[:, model.body("SquareNut_main").id]
        # The arm's joints and speeds come first in the scene, so plain slices keep these live views of
        # Warp's arrays (indexing with a list would make one-off copies instead).
        assert layout.arm_joints == layout.arm_v == layout.arm_q == list(range(7))
        self.anchors, self.axes = view(self.d.xanchor)[:, :7], view(self.d.xaxis)[:, :7]
        self.holding_up, self.mass_entries = view(self.d.qfrc_bias)[:, :7], view(self.d.M)
        # Where each entry of the arm's 7 x 7 mass matrix sits in MuJoCo's compressed rows (lower half).
        rows, columns, places = [], [], []
        for i, row in enumerate(layout.arm_v):
            for k in range(model.M_rownnz[row]):
                column = model.M_colind[model.M_rowadr[row] + k]
                if column in layout.arm_v:
                    rows.append(i), columns.append(layout.arm_v.index(column)), places.append(model.M_rowadr[row] + k)
        self.mass_rows, self.mass_columns = torch.tensor(rows, device=device), torch.tensor(columns, device=device)
        self.mass_places = torch.tensor(places, device=device)
        self.round_nut_q = [model.joint("RoundNut_joint0").qposadr[0] + i for i in range(7)]
        double = dict(dtype=torch.float64, device=device)
        self.torque_limits = torch.tensor(layout.torque_limits, **double)
        self.finger_middle = torch.tensor(layout.finger_middle, **double)
        self.finger_half = torch.tensor(layout.finger_half, **double)
        # Buffers that the recorded graphs read and write in place.
        self.actions = torch.zeros(copies, 7, **double)
        self.finger_targets = torch.zeros(copies, 2, **double)
        self.goal_position = torch.zeros(copies, 3, **double)
        self.goal_orientation = torch.eye(3, **double).repeat(copies, 1, 1)
        self.graphs = None

    def mass(self):
        """[copies, 7, 7] the arm's mass matrix, unpacked from MuJoCo's compressed rows."""
        entries = self.mass_entries[:, self.mass_places].double()
        full = torch.zeros(self.copies, 7, 7, dtype=torch.float64, device=self.device)
        full[:, self.mass_rows, self.mass_columns] = entries
        full[:, self.mass_columns, self.mass_rows] = entries
        return full

    def set_goal(self):
        position, orientation = controller.goal(self.actions, self.hand_position.double(), self.hand_orientation.double())
        self.goal_position.copy_(position)
        self.goal_orientation.copy_(orientation)

    def control(self):
        torques = controller.torques(self.goal_position, self.goal_orientation, self.hand_position.double(),
                                     self.hand_orientation.double(), self.anchors.double(), self.axes.double(),
                                     self.mass(), self.holding_up.double(), self.qpos[:, :7].double(),
                                     self.qvel[:, :7].double(), self.torque_limits)
        self.finger_targets.copy_(controller.finger_targets(self.finger_targets, self.actions[:, 6]))
        self.ctrl[:, :7] = torques.float()
        self.ctrl[:, 7:9] = (self.finger_middle + self.finger_half * self.finger_targets).float()

    def record(self):
        """Run each piece once, then record it as a GPU graph to replay. The warm-up run moves the
        copies on by one physics step, so the caller sets the starts afterwards."""
        with wp.ScopedStream(self.stream), torch.cuda.stream(self.torch_stream):
            mjw.step1(self.m, self.d)
            self.set_goal()
            self.control()
            mjw.step2(self.m, self.d)
            with wp.ScopedCapture(stream=self.stream) as first_half:
                mjw.step1(self.m, self.d)
            with wp.ScopedCapture(stream=self.stream) as second_half:
                mjw.step2(self.m, self.d)
            goal, control = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(goal, stream=self.torch_stream):
                self.set_goal()
            with torch.cuda.graph(control, stream=self.torch_stream):
                self.control()
        self.graphs = (first_half.graph, second_half.graph, goal, control)

    def set_starts(self, arm, fingers, nut_position, nut_quaternion):
        """Each copy at its own start, at rest. Arguments are [copies, ...] tensors."""
        if self.graphs is None:
            self.record()
        layout = self.layout
        with wp.ScopedStream(self.stream), torch.cuda.stream(self.torch_stream):
            mjw.reset_data(self.m, self.d)
            self.qpos[:, layout.arm_q] = arm.float()
            self.qpos[:, layout.finger_q] = fingers.float()
            self.qpos[:, layout.nut_q] = torch.cat([nut_position, nut_quaternion], dim=1).float()
            self.qpos[:, self.round_nut_q] = torch.tensor(PARKED, device=self.device)
            self.qvel.zero_()
            self.finger_targets.copy_(((fingers.double() - self.finger_middle) / self.finger_half).clamp(-1, 1))

    def set_recorded(self, flat_states, finger_targets):
        """Each copy at a state as robosuite records it (time, all positions, all velocities)."""
        if self.graphs is None:
            self.record()
        nq = self.layout.model.nq
        with wp.ScopedStream(self.stream), torch.cuda.stream(self.torch_stream):
            mjw.reset_data(self.m, self.d)
            self.time[:] = flat_states[:, 0].float()
            self.qpos[:] = flat_states[:, 1:1 + nq].float()
            self.qvel[:] = flat_states[:, 1 + nq:].float()
            self.finger_targets.copy_(finger_targets.double())

    def step(self, actions):
        """One control step for every copy; actions is a [copies, 7] tensor."""
        first_half, second_half, goal, control = self.graphs
        with torch.cuda.stream(self.torch_stream):
            self.actions.copy_(actions.to(self.device, torch.float64))
            for i in range(PHYSICS_STEPS):
                wp.capture_launch(first_half, stream=self.stream)
                if i == 0:
                    goal.replay()
                control.replay()
                wp.capture_launch(second_half, stream=self.stream)

    def state(self):
        """[copies, 33] the state, as sim.Sim.state gives it."""
        layout = self.layout
        q, v = self.qpos, self.qvel
        return torch.cat([q[:, layout.arm_q], q[:, layout.finger_q], q[:, layout.nut_q], self.finger_targets.float(),
                          v[:, layout.arm_v], v[:, layout.finger_v], v[:, layout.nut_v]], dim=1)

    def recorded(self):
        return torch.cat([self.time[:, None], self.qpos, self.qvel], dim=1)

    def contacts_dropped(self):
        """Whether the copies filled all the room for contacts in the last physics step (the count
        stops at the room, so full means some were dropped)."""
        return int(self.d.nacon.numpy()[0]) >= self.copies * CONTACTS_PER_COPY
