"""The fixed world: robosuite's Square scene, set up as in robomimic's human demos.

A Panda arm stands at a table that holds two pegs and a square nut (robosuite's NutAssemblySquare;
robomimic calls it Square). The test task is to put the nut around the square peg. The scene also
holds a round nut, which Square never uses: robosuite parks it far away, and it stays there.

The arm takes 7 numbers at 20 Hz through robosuite's controller (OSC_POSE): move the hand (3, up to
5 cm a step), turn the hand (3, up to 0.5 rad a step), and the gripper (1: positive closes,
negative opens, 0 leaves it as it is). Physics runs 25 steps of 2 ms inside each control step.

The state has 33 numbers:
    arm joint angles (7), finger positions (2), nut position (3), nut orientation (4, a quaternion),
    finger targets (2): where the gripper drives its fingers; each gripper action moves them a little,
        so they carry what earlier actions did,
    then the velocities of the arm (7), the fingers (2) and the nut (6).

Two pieces of the controller matter beyond the physics. The finger targets are set at every start to
where the fingers are, so a start holds still. The arm's resting posture (which the controller uses
for the arm's seventh, spare degree of freedom) stays at the arm's home pose for every start: the
human demos were recorded that way, and they replay to 1e-7 a step only with it (test_robot.py).
"""

import mujoco
import numpy as np
import robosuite

# As recorded in the human demos' file (test_robot.py checks this).
CONTROLLER = {"type": "OSC_POSE", "input_max": 1, "input_min": -1,
              "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
              "kp": 150, "damping": 1, "impedance_mode": "fixed", "kp_limits": [0, 300], "damping_limits": [0, 10],
              "position_limits": None, "orientation_limits": None, "uncouple_pos_ori": True,
              "control_delta": True, "interpolation": None, "ramp_ratio": 0.2}
# Control steps in one experiment: 2.5 s. Any situation can be a start, so an experiment need not be
# as long as a human demo (median 150 steps), and short ones give more different starts for the compute.
STEPS = 50
STATE_SIZE = 33
TABLE_TOP = 0.82            # height of the table's surface, in meters


class World:
    """One copy of the scene. Each process makes its own."""

    def __init__(self):
        self.env = robosuite.make("NutAssemblySquare", robots=["Panda"], controller_configs=CONTROLLER,
                                  control_freq=20, has_renderer=False, has_offscreen_renderer=False,
                                  use_camera_obs=False, use_object_obs=False, ignore_done=True,
                                  reward_shaping=False, hard_reset=False)
        self.env.reset()
        self.model, self.data = self.env.sim.model._model, self.env.sim.data._data
        self.robot = self.env.robots[0]
        self.nut = next(nut for nut in self.env.nuts if "Square" in nut.name)
        address = lambda name: (self.model.joint(name).qposadr[0], self.model.joint(name).dofadr[0])
        arm = [address(f"robot0_joint{i}") for i in range(1, 8)]
        fingers = [address("gripper0_finger_joint1"), address("gripper0_finger_joint2")]
        nut, round_nut = address("SquareNut_joint0"), address("RoundNut_joint0")
        self.arm_q, self.arm_v = [q for q, _ in arm], [v for _, v in arm]
        self.finger_q, self.finger_v = [q for q, _ in fingers], [v for _, v in fingers]
        self.nut_q, self.nut_v = list(range(nut[0], nut[0] + 7)), list(range(nut[1], nut[1] + 6))
        self.round_nut_q = list(range(round_nut[0], round_nut[0] + 7))
        self.parked_round_nut = self.data.qpos[self.round_nut_q].copy()
        self.arm_range = self.model.jnt_range[[self.model.joint(f"robot0_joint{i}").id for i in range(1, 8)]].copy()
        self.finger_range = self.model.jnt_range[[self.model.joint(f"gripper0_finger_joint{i}").id for i in (1, 2)]].copy()
        self.square_peg = self.data.xpos[self.model.body("peg1").id][:2].copy()
        # The scene's usual start: the arm's home pose, the fingers as robosuite resets them, and the
        # nut in the middle of robosuite's own placement area, turned 0, at robosuite's placing height.
        self.default_start = {"arm": np.array(self.robot.init_qpos, dtype=float),
                              "fingers": self.data.qpos[self.finger_q].copy(),
                              "nut_position": np.array([-0.1125, 0.1675, self.data.qpos[self.nut_q[2]]]),
                              "nut_quaternion": np.array([1.0, 0.0, 0.0, 0.0])}

    def set_start(self, arm, fingers, nut_position, nut_quaternion):
        """Put everything where the start says, at rest, and reset the two controller pieces."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_q] = arm
        self.data.qpos[self.finger_q] = fingers
        self.data.qpos[self.nut_q] = np.concatenate([nut_position, nut_quaternion])
        self.data.qpos[self.round_nut_q] = self.parked_round_nut
        mujoco.mj_forward(self.model, self.data)
        self.set_controller_rest()

    def set_recorded(self, flat_state):
        """Set a state as robosuite records it (time, then all positions, then all velocities),
        as in the human demos' files. The controller pieces are left as they are."""
        self.env.sim.set_state_from_flattened(flat_state)
        self.env.sim.forward()

    def start_recorded(self, flat_state):
        """Start from a demo's first recorded state, with the finger targets at 0, where robosuite's
        reset leaves them."""
        self.set_recorded(flat_state)
        self.robot.gripper.current_action = np.zeros(2)
        self.robot.controller.update_initial_joints(self.default_start["arm"])
        self.robot.controller.reset_goal()

    def recorded(self):
        """The state as robosuite records it, to compare with the demos' files."""
        return self.env.sim.get_state().flatten()

    def set_controller_rest(self):
        """The finger targets go to where the fingers are; the resting posture is the home pose."""
        self.robot.gripper.current_action = self.finger_targets_at(self.data.qpos[self.finger_q])
        self.robot.controller.update_initial_joints(self.default_start["arm"])
        self.robot.controller.reset_goal()

    def finger_targets_at(self, fingers):
        """The gripper command (-1 to 1 per finger) that holds the fingers where they are."""
        low, high = self.finger_range[:, 0], self.finger_range[:, 1]
        return (fingers - (low + high) / 2) / ((high - low) / 2)

    def step(self, action):
        self.env.step(np.asarray(action, dtype=float))

    def state(self):
        q, v = self.data.qpos, self.data.qvel
        return np.concatenate([q[self.arm_q], q[self.finger_q], q[self.nut_q], self.robot.gripper.current_action,
                               v[self.arm_v], v[self.finger_v], v[self.nut_v]])

    def hand(self):
        return self.env.sim.data.site_xpos[self.robot.eef_site_id].copy()

    def hand_axes(self):
        """The hand's x, y and z axes, as the columns of a 3 x 3 matrix. The fingers close along y."""
        return self.env.sim.data.site_xmat[self.robot.eef_site_id].reshape(3, 3).copy()

    def nut_position(self):
        return self.data.qpos[self.nut_q[:3]].copy()

    def nut_axes(self):
        """The nut's x, y and z axes, as columns. Its handle sticks out along x and is 3.2 cm wide along y."""
        return self.data.xmat[self.model.body("SquareNut_main").id].reshape(3, 3).copy()

    def handle(self):
        """Where the nut's handle is, in the world."""
        return self.data.site_xpos[self.model.site(self.nut.important_sites["handle"]).id].copy()

    def deepest_overlap(self):
        """How far, in meters, the two things most inside each other overlap (0 if nothing does)."""
        depths = [self.data.contact[i].dist for i in range(self.data.ncon)]
        return max(0.0, -min(depths)) if depths else 0.0

    def touching(self):
        """Whether any finger touches the nut."""
        nut = set(self.model.geom(name).id for name in self.nut.contact_geoms)
        fingers = set(self.model.geom(name).id for name in self.robot.gripper.contact_geoms)
        contacts = [(self.data.contact[i].geom1, self.data.contact[i].geom2) for i in range(self.data.ncon)]
        return any((a in nut and b in fingers) or (b in nut and a in fingers) for a, b in contacts)

    def holding(self):
        """robosuite's own grasp check: both fingers' pads on the nut."""
        return bool(self.env._check_grasp(gripper=self.robot.gripper, object_geoms=self.nut.contact_geoms))

    def nut_on_peg(self):
        """robosuite's own success check: the nut around the square peg, and the gripper away from it."""
        return bool(self.env._check_success())
