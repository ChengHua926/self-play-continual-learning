"""Checks of our copy of the world (sim.py) against what robosuite did (reference.py's recording).

Two things are checked separately:
    the controller: from the same state, our first motor commands of a control step against
        robosuite's (torques reach 80, so 1e-3 is a millionth-scale difference);
    the physics: a whole control step against robosuite's. The newer MuJoCo resolves contacts a
        little differently from robosuite's older one: given robosuite's own motor commands, it
        lands a median 0.55 mm away on the demo steps, where the fingers touch the nut, and about
        0.01 mm in free motion. Our whole steps should differ by no more than that.

Run them with:   ../.venv/bin/python -m unittest test_sim -v      (from the robot folder)
"""

import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

import controller
from sim import HOME, Sim

REFERENCE = Path(__file__).resolve().parent.parent / "data" / "robot_scene" / "reference.npz"
DEMO_STEPS = 677            # the first steps in the recording come from 5 demos, the rest are random


@unittest.skipUnless(REFERENCE.exists(), "needs reference.py's recording")
class TestSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = np.load(REFERENCE)
        cls.sim = Sim()

    def start_at(self, i):
        self.sim.set_recorded(self.reference["before"][i], self.reference["targets"][i])
        self.sim.data.qacc_warmstart[:] = self.reference["warmstart"][i]
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def test_home_pose_is_robosuites(self):
        np.testing.assert_allclose(HOME, self.reference["home"], atol=1e-12)

    def test_motor_commands_match_robosuites(self):
        gaps = []
        for i in range(len(self.reference["action"])):
            self.start_at(i)
            ours = self.sim.controls(self.reference["action"][i], first=True)
            gaps.append(np.abs(ours - self.reference["controls"][i][0]).max())
        self.assertLess(np.median(gaps), 1e-4)
        self.assertLess(max(gaps), 1e-3)

    def test_whole_steps_differ_only_as_the_physics_does(self):
        gaps = []
        for i in range(len(self.reference["action"])):
            self.start_at(i)
            self.sim.step(self.reference["action"][i])
            gaps.append(np.abs(self.sim.recorded()[1:17] - self.reference["after"][i][1:17]).max())
        gaps = np.array(gaps)
        self.assertLess(np.median(gaps[:DEMO_STEPS]), 1e-3)
        self.assertLess(np.median(gaps[DEMO_STEPS:]), 1e-4)

    def test_the_batched_controller_gives_the_same_commands(self):
        rows = range(0, len(self.reference["action"]), 7)
        inputs, expected = [], []
        for i in rows:
            self.start_at(i)
            inputs.append(self.sim.controller_inputs())
            targets_before = self.sim.finger_targets.copy()
            expected.append((self.sim.controls(self.reference["action"][i], first=True)[:7], targets_before,
                             self.sim.finger_targets.copy()))
        batch = {name: torch.tensor(np.array([row[name] for row in inputs])) for name in inputs[0]}
        action = torch.tensor(self.reference["action"][list(rows)])
        goal = controller.goal(action, batch["hand_position"], batch["hand_orientation"])
        limits = torch.tensor(self.sim.torque_limits)
        ours = controller.torques(*goal, **batch, limits=limits)
        np.testing.assert_allclose(ours.numpy(), np.array([e[0] for e in expected]), atol=1e-8)
        targets = controller.finger_targets(torch.tensor(np.array([e[1] for e in expected])), action[:, 6])
        np.testing.assert_allclose(targets.numpy(), np.array([e[2] for e in expected]), atol=1e-12)

    def test_state_has_33_numbers_with_the_finger_targets_after_the_nut(self):
        self.start_at(0)
        state = self.sim.state()
        self.assertEqual(state.shape, (33,))
        np.testing.assert_array_equal(state[16:18], self.reference["targets"][0])


if __name__ == "__main__":
    unittest.main()
