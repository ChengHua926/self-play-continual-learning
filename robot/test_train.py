"""Checks with known answers for the pieces self_play.py adds: decoding bytes on the GPU, the
quaternions it makes, and the paper's reward on the world model.

Run them with:   ../.venv/bin/python -m unittest test_train -v      (from the robot folder)
"""

import unittest

import numpy as np
import torch

import paths  # noqa: F401
import gpu_ways
import reward as paper_reward
import robot_machine
import world_model
from robot_machine import NEEDED
from sim import Sim


class TestDecoding(unittest.TestCase):
    def test_gpu_decoding_follows_the_machines_rules(self):
        sim = Sim()
        rng = np.random.default_rng(0)
        written = rng.integers(0, 256, (64, NEEDED), dtype=np.uint8)
        counts = np.concatenate([[0, 5, 15, 16, 100], rng.integers(0, NEEDED + 1, 59)])
        start, actions = gpu_ways.decode(torch.from_numpy(written), torch.from_numpy(counts), sim, "cpu")
        for i in range(64):
            expected_start, expected_actions = robot_machine.decode(written[i, :counts[i]].tobytes(), sim)
            for name in ("arm", "fingers", "nut_position"):
                np.testing.assert_allclose(start[name][i].numpy(), expected_start[name], atol=1e-12)
            q, expected_q = start["nut_quaternion"][i].numpy(), expected_start["nut_quaternion"]
            self.assertTrue(np.allclose(q, expected_q, atol=1e-9) or np.allclose(q, -expected_q, atol=1e-9))
            np.testing.assert_allclose(actions[i].numpy(), expected_actions, atol=1e-12)


class TestReward(unittest.TestCase):
    def test_reward_on_the_world_model_is_the_exact_derivative(self):
        torch.manual_seed(0)
        scales = {"state_mean": np.zeros(33), "state_spread": np.ones(33), "change_spread": np.full(33, 0.1)}
        model = world_model.WorldModel(scales).double()
        states = torch.randn(6, 11, 33, dtype=torch.float64)
        actions = torch.rand(6, 10, 7, dtype=torch.float64) * 2 - 1
        rows = world_model.pack(states[:, :-1], actions, states[:, 1:])
        along = {name: torch.randn_like(p) for name, p in model.named_parameters()}
        ours = paper_reward.alignment(model, rows, along, "cpu", losses_of=world_model.packed_losses)
        exact = []
        for i in range(len(rows)):
            loss = world_model.packed_losses(model, rows[i:i + 1]).sum()
            grads = torch.autograd.grad(loss, list(model.parameters()))
            exact.append(sum((g * along[name]).sum() for g, (name, _) in zip(grads, model.named_parameters())).abs().item())
        # The reward adds up in 32-bit arithmetic, as in the reproduction, so it agrees to about 1e-7 of
        # the size of the numbers it adds up (about the largest reward here).
        np.testing.assert_allclose(ours.numpy(), exact, atol=1e-6 * max(exact))


if __name__ == "__main__":
    unittest.main()
