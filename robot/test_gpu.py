"""Checks of the GPU world (gpu_world.py) against our CPU copy (sim.py), from robosuite's recorded steps.

Every 7th step of reference.py's recording starts its own copy, and each copy takes one control step.
MuJoCo Warp computes in 32-bit floats, so the GPU lands a little away from the 64-bit CPU copy; this
measures how far, and checks it stays small.

Run them with:   ../.venv/bin/python -m unittest test_gpu -v      (on a machine with an NVIDIA GPU)
"""

import unittest
from pathlib import Path

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from gpu_world import PARKED, GpuWorld
from sim import Sim

REFERENCE = Path(__file__).resolve().parent.parent / "data" / "robot_scene" / "reference.npz"
DEMO_STEPS = 677


@unittest.skipUnless(torch.cuda.is_available() and REFERENCE.exists(), "needs an NVIDIA GPU and reference.py's recording")
class TestGpuWorld(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = np.load(REFERENCE)
        cls.rows = list(range(0, len(cls.reference["action"]), 7))
        cls.world = GpuWorld(len(cls.rows))
        cls.cpu = Sim()
        cls.before, cls.targets, cls.actions = (torch.tensor(cls.reference[k][cls.rows], device="cuda")
                                                for k in ("before", "targets", "action"))

    def test_robosuite_parks_the_round_nut_where_we_do(self):
        np.testing.assert_allclose(self.reference["before"][0][1 + 16:1 + 23], PARKED)

    def test_arm_mass_matrix_matches_the_cpu(self):
        self.world.set_recorded(self.before, self.targets)
        with wp.ScopedStream(self.world.stream):
            mjw.forward(self.world.m, self.world.d)
        gpu = self.world.mass().cpu().numpy()
        for k, i in enumerate(self.rows[:20]):
            self.cpu.set_recorded(self.reference["before"][i], self.reference["targets"][i])
            np.testing.assert_allclose(gpu[k], self.cpu.controller_inputs()["mass"], atol=1e-4)

    def test_one_control_step_lands_near_the_cpu_copy(self):
        self.world.set_recorded(self.before, self.targets)
        self.world.step(self.actions)
        gpu = self.world.recorded().cpu().numpy()
        gaps = []
        for k, i in enumerate(self.rows):
            self.cpu.set_recorded(self.reference["before"][i], self.reference["targets"][i])
            self.cpu.step(self.reference["action"][i])
            gaps.append(np.abs(gpu[k, 1:17] - self.cpu.recorded()[1:17]).max())
        gaps, demo = np.array(gaps), np.array(self.rows) < DEMO_STEPS
        print(f"\n   one control step, GPU vs CPU copy: demo steps median {np.median(gaps[demo]):.1e}, "
              f"largest {gaps[demo].max():.1e}; random steps median {np.median(gaps[~demo]):.1e}, largest {gaps[~demo].max():.1e}")
        self.assertLess(np.median(gaps[demo]), 1e-3)
        self.assertLess(np.median(gaps[~demo]), 1e-3)
        self.assertFalse(self.world.contacts_dropped())


if __name__ == "__main__":
    unittest.main()
