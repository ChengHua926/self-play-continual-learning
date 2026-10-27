"""Our own copy of the fixed world, on the newer MuJoCo (3.17): the step toward running it on the GPU.

robosuite runs the world through its own Python controller, one copy at a time, on an older MuJoCo
(2.3.2), and the GPU simulators need the newer one. So this rebuilds the same world on the newer
MuJoCo: the scene exported from robosuite (export_scene.py), and robosuite's controller written out
here as plain array math, so that the next step can run thousands of copies at once.

The controller is robosuite's OSC_POSE with the human demos' settings. At the first of the 25
physics steps in each control step it sets a goal for the hand: the action's first 3 numbers move
it up to 5 cm, the next 3 turn it up to 0.5 rad (as an axis times an angle). At every physics step
it then pushes the hand toward that goal like a spring (stiffness 150, critically damped), working out
the joint torques through the arm's mass matrix, adds what holds the arm up against gravity, and
uses the arm's spare seventh freedom to drift toward the home pose. Torques are clipped to the
motors' limits. The gripper's finger targets move 0.01 per physics step toward closed or open, by
the sign of the action's last number.

Two differences on purpose:
    - On a turn of exactly 0, this holds the hand's orientation as it is, where robosuite goes back to
      its last goal orientation. That keeps the 33-number state complete (robosuite's last goal would
      be another hidden number), and the human demos never turn by exactly 0.
    - The unused round nut, parked 10 m away, floats there instead of falling to the floor, so it never
      touches anything (on the GPU, its resting contacts would cost every copy for nothing).

How it compares with robosuite: test_sim.py, against what reference.py recorded.
"""

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

SCENE = Path(__file__).resolve().parent.parent / "data" / "robot_scene" / "square.xml"
PHYSICS_STEPS = 25          # in each control step: 25 x 2 ms = 50 ms, so 20 control steps a second
STIFFNESS = 150.0
DAMPING = 2 * np.sqrt(STIFFNESS)            # critically damped
HOME_STIFFNESS = 10.0                       # for the spare freedom's drift toward home
HOME_DAMPING = 2 * np.sqrt(HOME_STIFFNESS)
LARGEST = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])     # meters and radians per control step
FINGER_SPEED = 0.01         # change of the finger targets per physics step
HOME = np.array([0, np.pi / 16, 0, -np.pi / 2 - np.pi / 3, 0, np.pi - 0.2, np.pi / 4])    # robosuite's Panda home pose


class Sim:
    """One copy of the world on MuJoCo 3.17, with the same state and controls as world.World."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.model.body_gravcomp[self.model.body("RoundNut_main").id] = 1.0     # the parked round nut floats
        self.data = mujoco.MjData(self.model)
        joint = self.model.joint
        self.arm_joints = [joint(f"robot0_joint{i}").id for i in range(1, 8)]
        self.arm_q = [joint(f"robot0_joint{i}").qposadr[0] for i in range(1, 8)]
        self.arm_v = [joint(f"robot0_joint{i}").dofadr[0] for i in range(1, 8)]
        self.finger_q = [joint(f"gripper0_finger_joint{i}").qposadr[0] for i in (1, 2)]
        self.finger_v = [joint(f"gripper0_finger_joint{i}").dofadr[0] for i in (1, 2)]
        nut_q, nut_v = joint("SquareNut_joint0").qposadr[0], joint("SquareNut_joint0").dofadr[0]
        self.nut_q, self.nut_v = list(range(nut_q, nut_q + 7)), list(range(nut_v, nut_v + 6))
        self.hand_site = self.model.site("gripper0_grip_site").id
        self.arm_motors, self.finger_motors = list(range(7)), [7, 8]
        self.torque_limits = self.model.actuator_ctrlrange[self.arm_motors]
        low, high = self.model.actuator_ctrlrange[self.finger_motors].T
        self.finger_middle, self.finger_half = (low + high) / 2, (high - low) / 2
        self.finger_targets = np.zeros(2)
        self.goal_position, self.goal_orientation = np.zeros(3), np.eye(3)

    def set_recorded(self, flat_state, finger_targets):
        """Set a state as robosuite records it (time, all positions, all velocities)."""
        nq = self.model.nq
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = flat_state[0]
        self.data.qpos[:], self.data.qvel[:] = flat_state[1:1 + nq], flat_state[1 + nq:]
        self.finger_targets = np.array(finger_targets, dtype=float)
        mujoco.mj_forward(self.model, self.data)

    def recorded(self):
        return np.concatenate([[self.data.time], self.data.qpos, self.data.qvel])

    def state(self):
        q, v = self.data.qpos, self.data.qvel
        return np.concatenate([q[self.arm_q], q[self.finger_q], q[self.nut_q], self.finger_targets,
                               v[self.arm_v], v[self.finger_v], v[self.nut_v]])

    def arm_now(self):
        """The hand's position and orientation, the site's Jacobians for the arm's 7 joints, the hand's
        velocities, the arm's mass matrix and the forces that hold it up (gravity and motion)."""
        model, data = self.model, self.data
        position = data.site_xpos[self.hand_site].copy()
        orientation = data.site_xmat[self.hand_site].reshape(3, 3).copy()
        jac_position, jac_turn = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac_position, jac_turn, self.hand_site)
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass)
        arm = self.arm_v
        return (position, orientation, jac_position[:, arm], jac_turn[:, arm],
                jac_position @ data.qvel, jac_turn @ data.qvel, mass[np.ix_(arm, arm)], data.qfrc_bias[arm].copy())

    def controller_inputs(self):
        """What the batched controller (controller.py) reads from the physics, for this one copy."""
        model, data = self.model, self.data
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass)
        return {"hand_position": data.site_xpos[self.hand_site].copy(),
                "hand_orientation": data.site_xmat[self.hand_site].reshape(3, 3).copy(),
                "anchors": data.xanchor[self.arm_joints].copy(), "axes": data.xaxis[self.arm_joints].copy(),
                "mass": mass[np.ix_(self.arm_v, self.arm_v)], "holding_up": data.qfrc_bias[self.arm_v].copy(),
                "q": data.qpos[self.arm_q].copy(), "v": data.qvel[self.arm_v].copy()}

    def set_goal(self, action):
        position, orientation = self.arm_now()[:2]
        delta = np.clip(action[:6], -1, 1) * LARGEST
        self.goal_position = position + delta[:3]
        self.goal_orientation = Rotation.from_rotvec(delta[3:]).as_matrix() @ orientation

    def torques(self):
        position, orientation, jac_position, jac_turn, velocity, turning, mass, holding_up = self.arm_now()
        # The orientation error: half the sum of crosses of the current and goal axes (robosuite's formula).
        turn_error = 0.5 * sum(np.cross(orientation[:, i], self.goal_orientation[:, i]) for i in range(3))
        force = STIFFNESS * (self.goal_position - position) - DAMPING * velocity
        twist = STIFFNESS * turn_error - DAMPING * turning
        jac = np.vstack([jac_position, jac_turn])
        mass_inverse = np.linalg.inv(mass)
        # How heavy the hand feels along each direction (robosuite's "lambda" matrices).
        heavy_position = np.linalg.pinv(jac_position @ mass_inverse @ jac_position.T)
        heavy_turn = np.linalg.pinv(jac_turn @ mass_inverse @ jac_turn.T)
        heavy = np.linalg.pinv(jac @ mass_inverse @ jac.T)
        torques = jac.T @ np.concatenate([heavy_position @ force, heavy_turn @ twist]) + holding_up
        # The spare freedom: drift toward home, kept out of the hand's motion (the "nullspace").
        nullspace = np.eye(7) - (mass_inverse @ jac.T @ heavy) @ jac
        q, v = self.data.qpos[self.arm_q], self.data.qvel[self.arm_v]
        torques += nullspace.T @ (mass @ (HOME_STIFFNESS * (HOME - q) - HOME_DAMPING * v))
        return np.clip(torques, self.torque_limits[:, 0], self.torque_limits[:, 1])

    def controls(self, action, first):
        """The 9 motor commands for one physics step (the state must be forwarded first)."""
        if first:
            self.set_goal(action)
        torques = self.torques()
        self.finger_targets = np.clip(self.finger_targets + np.array([-1.0, 1.0]) * FINGER_SPEED * np.sign(action[6]), -1, 1)
        return np.concatenate([torques, self.finger_middle + self.finger_half * self.finger_targets])

    def step(self, action):
        action = np.asarray(action, dtype=float)
        for i in range(PHYSICS_STEPS):
            mujoco.mj_forward(self.model, self.data)
            self.data.ctrl[:] = self.controls(action, first=(i == 0))
            mujoco.mj_step(self.model, self.data)
