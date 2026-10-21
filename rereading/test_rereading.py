"""Small checks that the rereading code does what it says.

Run them with:   ../.venv/bin/python -m unittest test_rereading -v      (from the rereading folder)
"""

import unittest

import numpy as np
import torch

import train
import updates
from model import CONTEXT, GPT


class WindowTests(unittest.TestCase):
    def test_targets_are_the_inputs_one_token_later(self):
        tokens = np.arange(3 * CONTEXT + 7)
        cut = train.windows(tokens)
        self.assertEqual(cut.shape, (3, CONTEXT + 1))
        self.assertTrue((cut[1, :-1] + 1 == cut[1, 1:]).all())
        self.assertEqual(cut[1, 0], CONTEXT)  # windows start every CONTEXT tokens, so none repeats a token as input

    def test_held_out_text_is_not_in_the_training_pool(self):
        held_out = np.load(train.HELD_OUT_FILE)
        training = np.load(train.TRAINING_FILE)
        stretch = 64
        seen = {training[i:i + stretch].tobytes() for i in range(0, len(training) - stretch, 997)}
        found = sum(held_out[i:i + stretch].tobytes() in seen for i in range(0, len(held_out) - stretch))
        self.assertEqual(found, 0)  # text.py leaves out held-out documents that share a stretch with the pool


class NeighbourTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = GPT("29M")
        text = np.load(train.TRAINING_FILE)[:6 * 65]
        self.tokens = torch.from_numpy(text.astype(np.int64)).view(6, 65)

    def true_gradients(self, centred):
        """Each sequence's own last-layer gradient, from autograd, as one row each."""
        rows = []
        for sequence in self.tokens:
            self.model.zero_grad()
            _, logits = self.model(sequence[None, :-1])
            torch.nn.functional.cross_entropy(logits[0], sequence[1:], reduction="sum").backward()
            rows.append(self.model.out.weight.grad.flatten().clone())
        rows = torch.stack(rows)
        return rows - rows.mean(dim=0) if centred else rows

    def test_the_cosines_match_the_true_last_layer_gradients(self):
        with torch.no_grad():
            features, logits = self.model(self.tokens[:, :-1])
        others = ~torch.eye(6, dtype=torch.bool)
        inner = updates.gradient_inner_products(features, logits, self.tokens[:, 1:])
        for centred in (False, True):
            ours = updates.cosines(inner, centred)
            exact = torch.nn.functional.normalize(self.true_gradients(centred), dim=1)
            exact = exact @ exact.T
            self.assertLess((ours - exact)[others].abs().max().item(), 0.01, f"centred={centred}")

    def test_a_sequence_repeated_in_the_batch_is_its_own_closest_neighbour(self):
        tokens = self.tokens.clone()
        tokens[5] = tokens[0]
        with torch.no_grad():
            features, logits = self.model(tokens[:, :-1])
        cosine = updates.cosines(updates.gradient_inner_products(features, logits, tokens[:, 1:]), centred=True)
        self.assertGreater(cosine[0, 5], 0.99)
        self.assertLess(cosine[0, 1:5].abs().max(), cosine[0, 5])

    def test_a_group_gets_weight_and_a_one_off_none(self):
        cosines = torch.full((8, 8), 0.0)
        cosines[:3, :3] = 0.9  # sequences 0, 1 and 2 need the same change
        cosines += torch.diag(torch.ones(8) - cosines.diagonal())
        cosines[5, 6] = cosines[6, 5] = -0.1  # small differences among the rest
        weights, fell_back = updates.neighbour_weights(cosines, spreads=2.0)
        self.assertFalse(fell_back)
        self.assertTrue((weights[:3] > 0).all())
        self.assertTrue((weights[3:] == 0).all())
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=5)

    def test_with_no_neighbours_anywhere_the_batch_is_learned_as_usual(self):
        cosines = torch.eye(5) + 0.01 * (1 - torch.eye(5))
        weights, fell_back = updates.neighbour_weights(cosines, spreads=2.0)
        self.assertTrue(fell_back)
        self.assertTrue(torch.equal(weights, torch.ones(5)))

    def test_batch_agreement_favours_the_gradients_that_point_with_the_average(self):
        gradients = torch.tensor([[1.0, 0.0], [1.0, 0.1], [1.0, -0.1], [0.0, 1.0], [-1.0, 0.0]])
        weights, fell_back = updates.batch_agreement_weights(gradients @ gradients.T)
        self.assertFalse(fell_back)
        self.assertGreater(weights[0], weights[3])   # along the average beats across it
        self.assertEqual(weights[4].item(), 0.0)     # against it gets nothing
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=5)


class TrainingTests(unittest.TestCase):
    def test_a_short_run_learns_and_records_every_check(self):
        run = train.Run(size="29M", unique_tokens=40 * CONTEXT, passes=3, sequences=8, learning_rate=3e-3)
        original = train.HELD_OUT_WINDOWS
        train.HELD_OUT_WINDOWS = 16
        try:
            record = train.train(run)
        finally:
            train.HELD_OUT_WINDOWS = original
        self.assertEqual(len(record["held_out"]), len(train.EARLY_CHECKS) + 3)
        self.assertLess(record["training"][-1], record["training"][0])
        self.assertEqual(record["best_held_out"], min(record["held_out"]))

    def test_every_method_runs(self):
        original = train.HELD_OUT_WINDOWS
        train.HELD_OUT_WINDOWS = 8
        try:
            for method in ("neighbours", "shuffled neighbours", "batch agreement"):
                run = train.Run(size="29M", unique_tokens=16 * CONTEXT, passes=1, sequences=8, method=method)
                record = train.train(run)
                self.assertTrue(np.isfinite(record["best_held_out"]))
        finally:
            train.HELD_OUT_WINDOWS = original


if __name__ == "__main__":
    unittest.main()
