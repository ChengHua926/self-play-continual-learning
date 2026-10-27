"""Checks with known answers for the fixed world and the machine.

Run them with:   ../.venv-robosuite/bin/python -m unittest test_robot -v      (from the robot folder)
"""

import json
import unittest
from pathlib import Path

import h5py
import numpy as np

import robot_machine as machine
from world import CONTROLLER, STEPS, World

DEMOS = Path(__file__).resolve().parent.parent / "data" / "robomimic" / "square_ph" / "low_dim_v141.hdf5"


class TestWorld(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = World()

    @unittest.skipUnless(DEMOS.exists(), "needs the Square demos")
    def test_settings_match_the_demos(self):
        settings = json.loads(h5py.File(DEMOS, "r")["data"].attrs["env_args"])
        self.assertEqual(settings["env_name"], "NutAssemblySquare")
        self.assertEqual(settings["env_kwargs"]["controller_configs"], CONTROLLER)
        self.assertEqual(settings["env_kwargs"]["control_freq"], 20)
        self.assertEqual(settings["env_kwargs"]["robots"], ["Panda"])

    @unittest.skipUnless(DEMOS.exists(), "needs the Square demos")
    def test_one_step_from_a_recorded_state_gives_the_recorded_next_state(self):
        demo = h5py.File(DEMOS, "r")["data"]["demo_1"]
        states, actions = demo["states"][()], demo["actions"][()]
        self.world.start_recorded(states[0])
        errors = []
        for t in range(len(actions) - 1):
            self.world.set_recorded(states[t])          # the finger targets carry on from the last step
            self.world.step(actions[t])
            errors.append(np.abs(self.world.recorded()[1:24] - states[t + 1][1:24]).max())
        # A few steps at the moments of contact jump by millimeters; the rest match to float precision.
        self.assertLess(np.median(errors), 1e-5)
        self.assertLess(np.percentile(errors, 90), 1e-4)

    def test_unwritten_bytes_mean_the_usual_start_and_holding_still(self):
        start, actions = machine.decode(b"", self.world)
        for name, value in self.world.default_start.items():
            np.testing.assert_allclose(start[name], value, atol=1e-9)
        self.assertEqual(actions.shape, (STEPS, 7))
        self.assertTrue((actions == 0).all())

    def test_bytes_span_each_range(self):
        start, _ = machine.decode(bytes([0] * 9), self.world)
        np.testing.assert_allclose(start["arm"], self.world.arm_range[:, 0])
        np.testing.assert_allclose(start["fingers"], self.world.finger_range[:, 0])
        start, actions = machine.decode(bytes([255] * 12 + [0, 0, 0] + [128, 0, 255]), self.world)
        np.testing.assert_allclose(start["arm"], self.world.arm_range[:, 1])
        np.testing.assert_allclose(start["nut_position"], machine.NUT_HIGH)
        np.testing.assert_allclose(actions[0, :3], [0, -1, 1])

    def test_a_start_is_where_it_was_written_and_at_rest(self):
        start, _ = machine.decode(bytes(range(40, 55)), self.world)
        self.world.set_start(**start)
        state = self.world.state()
        np.testing.assert_allclose(state[:7], start["arm"])
        np.testing.assert_allclose(state[7:9], start["fingers"])
        np.testing.assert_allclose(state[9:12], start["nut_position"])
        np.testing.assert_allclose(state[16:18], self.world.finger_targets_at(start["fingers"]))
        self.assertTrue((state[18:] == 0).all())

    def test_the_same_bytes_give_the_same_experiment(self):
        written = np.random.default_rng(0).integers(0, 256, machine.NEEDED, dtype=np.uint8).tobytes()
        first, _, _ = machine.run_bytes(self.world, written)
        second, _, _ = machine.run_bytes(self.world, written)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
