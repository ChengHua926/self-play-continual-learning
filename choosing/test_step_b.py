"""Small checks that the pieces of step B do what they say.

Run them with:   python3 -m unittest test_step_b -v

The checks of report_step_b.py are tried on "oracle" signals, which know the answer, so
that a mistake in the checks themselves would show up here and not in the results.
"""

import unittest
from unittest.mock import patch

import numpy as np

from measures import ensemble_disagreement, feature_novelty, pinned_weight_uncertainty, random_network_novelty
from network import Network
from report_step_b import (
    above_average_rounds, narrow_outranks, new_object_wins, object_average_agreement, top_pick_share, top_picks,
)
from run_step2 import repeat_noise, run, train
from run_step_b import screen
from signals import (
    SIMILAR, last_layer_agreement_with_similar, similar_pushes, slope_agreement_repeated, slope_agreements,
    slope_consistency,
)
from world import OBJECTS, TABLE, TOY, World, narrowed


def some_rounds(rng, in_group=10, runs=3, rounds=70, pushes=64):
    """A made-up group of pushes, as [run, round, push]: in_group pushes of every round are in it."""
    keys = rng.random((runs, rounds, pushes))
    return keys < np.sort(keys, axis=-1)[..., in_group:in_group + 1]


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_the_top_picks_are_the_highest_quarter_of_each_round(self):
        values = self.rng.random((5, 64))
        top = top_picks(values, self.rng)
        np.testing.assert_array_equal(top.sum(axis=-1), 16)
        for row, picked in zip(values, top):
            self.assertGreater(row[picked].min(), row[~picked].max())

    def test_an_oracle_puts_its_whole_group_among_the_top_picks(self):
        group = some_rounds(self.rng)
        self.assertEqual(top_pick_share(group.astype(float), group, slice(None), self.rng), 1.0)
        self.assertEqual(top_pick_share(-group.astype(float), group, slice(None), self.rng), 0.0)

    def test_a_random_signal_puts_a_quarter_of_its_group_among_the_top_picks(self):
        group = some_rounds(self.rng)
        share = top_pick_share(self.rng.random(group.shape), group, slice(None), self.rng)
        self.assertAlmostEqual(share, 0.25, delta=0.02)

    def test_rounds_in_which_the_signal_was_not_read_do_not_count(self):
        group = some_rounds(self.rng)
        values = group.astype(float)
        values[:, :35] = np.nan  # not read in the first half of the rounds
        self.assertEqual(top_pick_share(values, group, slice(None), self.rng), 1.0)

    def test_narrow_parts_outrank_the_rest_always_for_an_oracle_and_half_the_time_by_chance(self):
        of_kind, narrow_part = some_rounds(self.rng, in_group=32), some_rounds(self.rng, in_group=12)
        self.assertEqual(narrow_outranks(narrow_part.astype(float), of_kind, narrow_part), 1.0)
        self.assertEqual(narrow_outranks(-narrow_part.astype(float), of_kind, narrow_part), 0.0)
        chance = narrow_outranks(self.rng.random(of_kind.shape), of_kind, narrow_part)
        self.assertAlmostEqual(chance, 0.5, delta=0.03)

    def test_a_new_object_wins_only_if_it_beats_every_old_one(self):
        # 70 rounds of 64 pushes on 6 objects. Objects 4 (a toy) and 5 arrive together in round 20.
        objects = np.tile(np.arange(64) % 6, (70, 1))
        new = np.zeros((70, 64), dtype=bool)
        new[20:22] = objects[20:22] >= 4
        new_learnable = new & (objects != 4)
        old_ones = 1.0 * (objects < 4)
        self.assertEqual(new_object_wins(-old_ones, objects, new_learnable, new), 1.0)
        self.assertEqual(new_object_wins(old_ones, objects, new_learnable, new), 0.0)
        # The new toy scoring highest of all neither helps nor hurts: only the new object against the old ones counts.
        toy_highest = 3.0 * (objects == 4) + 2.0 * (objects == 5) + 1.0 * (objects < 4)
        self.assertEqual(new_object_wins(toy_highest, objects, new_learnable, new), 1.0)
        # Beating the old objects' average is not enough: the new object must beat every one of them.
        one_old_higher = 2.0 * (objects == 5) + 1.0 * (objects < 3) + 3.0 * (objects == 3)
        self.assertEqual(new_object_wins(one_old_higher, objects, new_learnable, new), 0.0)

    def test_the_object_average_forecasts_a_signal_that_depends_only_on_the_object(self):
        objects = np.tile(np.arange(64) % 6, (70, 1))
        rng = np.random.default_rng(0)
        self.assertAlmostEqual(object_average_agreement(objects * 1.0, objects), 1.0)
        self.assertAlmostEqual(object_average_agreement(rng.random((70, 64)), objects), 0.0, delta=0.05)

    def test_an_oracle_scores_its_group_above_average_in_every_round(self):
        group = some_rounds(self.rng)
        self.assertEqual(above_average_rounds(group.astype(float), group, slice(None)), 1.0)
        self.assertEqual(above_average_rounds(-group.astype(float), group, slice(None)), 0.0)


def written_out(layer_inputs, layer_errors):
    """Every example's slope for every weight and bias, written out in full: one row per example."""
    rows = []
    for n in range(len(layer_inputs[0])):
        weights = [np.outer(layer_input[n], layer_error[n]).ravel() for layer_input, layer_error in zip(layer_inputs, layer_errors)]
        biases = [layer_error[n] for layer_error in layer_errors]
        rows.append(np.concatenate(weights + biases))
    return np.array(rows)


def full_slopes(network, inputs, results):
    """Every push's slope of the loss, for every weight and bias, written out in full."""
    _, layer_inputs, layer_errors = network.backward(inputs, results)
    return written_out(layer_inputs, layer_errors)


class WorldTests(unittest.TestCase):
    def test_a_narrow_world_acts_like_a_plain_one_exactly_in_the_narrow_parts(self):
        narrow = narrowed(TABLE)
        pushes = TABLE.random_pushes(5000, np.ones(6), np.random.default_rng(0))
        same = np.isclose(TABLE.expected_result(pushes), narrow.expected_result(pushes)).all(axis=1)
        not_toy = pushes.obj != TOY
        np.testing.assert_array_equal(same[not_toy], narrow.in_narrow_part(pushes)[not_toy])


class SlopeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.network = Network(rng, sizes=(12, 16, 16, 3))
        self.inputs, self.first, self.second = rng.normal(size=(20, 12)), rng.normal(size=(20, 3)), rng.normal(size=(20, 3))

    def test_slope_agreements_multiply_the_full_slopes(self):
        slopes = full_slopes(self.network, self.inputs, self.first)
        np.testing.assert_allclose(slope_agreements(self.network, self.inputs, self.first), slopes @ slopes.T)

    def test_the_same_push_twice_multiplies_the_slopes_from_its_two_results(self):
        expected = (full_slopes(self.network, self.inputs, self.first) * full_slopes(self.network, self.inputs, self.second)).sum(axis=1)
        np.testing.assert_allclose(slope_agreement_repeated(self.network, self.inputs, self.first, self.second), expected)

    def test_the_last_layer_version_uses_the_last_layers_slopes_only(self):
        similar = similar_pushes(self.network, self.inputs)
        features = self.network.features(self.inputs)
        error = 2 * (self.network.predict(self.inputs) - self.first)
        last_layer = np.array([np.concatenate([np.outer(f, e).ravel(), e]) for f, e in zip(features, error)])
        expected = np.take_along_axis(last_layer @ last_layer.T, similar, axis=1).mean(axis=1)
        np.testing.assert_allclose(last_layer_agreement_with_similar(self.network, self.inputs, self.first, similar), expected)

    def test_similar_pushes_never_include_the_push_itself(self):
        similar = similar_pushes(self.network, self.inputs)
        self.assertEqual(similar.shape, (20, SIMILAR))
        self.assertFalse((similar == np.arange(20)[:, None]).any())

    def test_consistency_is_highest_when_slopes_agree_and_about_one_when_they_are_unrelated(self):
        similar = np.array([[j for j in range(10) if j != i][:SIMILAR] for i in range(10)])
        np.testing.assert_allclose(slope_consistency(np.ones((10, 10)), similar), SIMILAR + 1)  # all the same slope
        np.testing.assert_allclose(slope_consistency(np.eye(10), similar), 1)                   # all at right angles


class NetworkTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.network = Network(rng, sizes=(4, 5, 5, 3))
        self.inputs, self.targets = rng.normal(size=(6, 4)), rng.normal(size=(6, 3))

    def nudged(self, measure, layer, row, column):
        """How much measure() changes when one weight is nudged, per unit of nudge."""
        before = measure()
        self.network.weights[layer][row, column] += 1e-6
        after = measure()
        self.network.weights[layer][row, column] -= 1e-6
        return (after - before) / 1e-6

    def test_output_slopes_say_how_an_output_moves_with_a_weight(self):
        layer_inputs, layer_errors = self.network.output_slopes(self.inputs, 1)
        for layer, row, column in ((0, 1, 2), (1, 3, 0), (2, 2, 1)):
            said = layer_inputs[layer][:, row] * layer_errors[layer][:, column]
            measured = self.nudged(lambda: self.network.predict(self.inputs)[:, 1], layer, row, column)
            np.testing.assert_allclose(said, measured, atol=1e-4)

    def test_with_noise_given_a_miss_costs_its_square_over_twice_the_noise(self):
        noise = np.random.default_rng(1).uniform(0.1, 2.0, size=(6, 3))
        _, layer_inputs, layer_errors = self.network.backward(self.inputs, self.targets, noise)
        loss = lambda: ((self.network.predict(self.inputs) - self.targets) ** 2 / (2 * noise)).sum(axis=1).mean()
        for layer, row, column in ((0, 1, 2), (2, 4, 1)):
            said = (layer_inputs[layer].T @ layer_errors[layer] / len(self.inputs))[row, column]
            self.assertAlmostEqual(said, self.nudged(loss, layer, row, column), places=4)


class MeasureTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.network = Network(self.rng)
        everything_but_the_spinner = np.ones(6)
        everything_but_the_spinner[OBJECTS.index("spinner")] = 0
        self.seen = TABLE.random_pushes(2000, everything_but_the_spinner, self.rng)
        self.fresh = TABLE.random_pushes(600, np.ones(6), self.rng)
        self.spinner = self.fresh.obj == OBJECTS.index("spinner")
        self.seen_results = TABLE.push(self.seen, self.rng)
        train(self.network, TABLE.as_inputs(self.seen), self.seen_results, 50, self.rng)

    def test_pinned_weights_add_up_squared_slopes_times_how_loose_each_weight_is(self):
        """The same number, the long way: every weight's slope and looseness written out in full."""
        rng = np.random.default_rng(1)
        network = Network(rng, sizes=(4, 5, 5, 3))
        memory, results, inputs = rng.normal(size=(30, 4)), rng.normal(size=(30, 3)), rng.normal(size=(6, 4))
        slopes = lambda x: [written_out(*network.output_slopes(x, output)) for output in range(3)]
        pinning = sum((slope**2).sum(axis=0) for slope in slopes(memory))
        noise = ((network.predict(memory) - results) ** 2).mean()
        prior = np.concatenate([np.full(w.size, w.shape[0] / 2) for w in network.weights]
                               + [np.full(b.size, w.shape[0] / 2) for w, b in zip(network.weights, network.biases)])
        looseness = 1 / (pinning / noise + prior)
        expected = sum(slope**2 @ looseness for slope in slopes(inputs))
        np.testing.assert_allclose(pinned_weight_uncertainty(network, inputs, memory, results, rng), expected)

    def test_pushes_on_an_object_never_seen_rest_on_looser_weights(self):
        uncertainty = pinned_weight_uncertainty(
            self.network, TABLE.as_inputs(self.fresh), TABLE.as_inputs(self.seen), self.seen_results, self.rng)
        self.assertGreater(uncertainty[self.spinner].mean(), 1.2 * uncertainty[~self.spinner].mean())

    def test_pushes_on_an_object_never_seen_are_far_from_everything_seen(self):
        novelty = feature_novelty(self.network, TABLE.as_inputs(self.fresh), TABLE.as_inputs(self.seen))
        self.assertGreater(novelty[self.spinner].mean(), novelty[~self.spinner].mean())

    def test_identical_networks_never_disagree(self):
        inputs = TABLE.as_inputs(self.fresh)
        np.testing.assert_array_equal(ensemble_disagreement([self.network, self.network.copy()], inputs), 0)

    def test_the_imitator_gets_better_where_it_has_trained(self):
        target, imitator = Network(self.rng, (12, 64, 64, 16)), Network(self.rng, (12, 64, 64, 16))
        seen = TABLE.as_inputs(self.seen)
        before = random_network_novelty(target, imitator, seen).mean()
        train(imitator, seen, target.predict(seen), 200, self.rng)
        self.assertLess(random_network_novelty(target, imitator, seen).mean(), before / 2)


class ScreenTests(unittest.TestCase):
    def test_the_measures_read_before_acting_do_not_see_the_rounds_results(self):
        """Change every result of the last round: the measures read before acting must not move, the labels must."""
        original = World.push

        def screen_with(change):
            calls = []

            def push(world, pushes, rng):
                calls.append(1)
                results = original(world, pushes, rng)
                return results + change if len(calls) > 2 * 5 else results  # every round pushes twice; round 5 is the last

            with patch("run_step_b.ROUNDS", 6), patch.object(World, "push", push):
                return screen(1, "6", 0)[0]

        unchanged, changed = screen_with(0.0), screen_with(1.0)
        for name in ("pinned weights", "prediction drift", "feature novelty", "ensemble disagreement",
                     "random-network novelty", "predicted size", "push strength", "noise from repeats", "predicts its noise"):
            np.testing.assert_array_equal(unchanged[name][5], changed[name][5])
        for name in ("raw error", "repeat agreement", "progress over time", "forward progress"):
            self.assertFalse(np.array_equal(unchanged[name][5], changed[name][5]))


class NoiseFromRepeatsTests(unittest.TestCase):
    def test_the_noise_of_a_push_that_never_changes_is_exactly_zero(self):
        rng = np.random.default_rng(0)
        pushes = TABLE.random_pushes(20000, np.ones(6), rng)
        noise = repeat_noise(TABLE.push(pushes, rng), TABLE.push(pushes, rng))
        toy = pushes.obj == TOY
        np.testing.assert_array_equal(noise[~toy], 0)
        # For the toy, on average, it is the variance of its results. Its turn is twist x (a number from -1 to 1).
        self.assertAlmostEqual(noise[toy, 2].mean(), TABLE.twist[TOY] ** 2 / 3, delta=0.03)

    def test_the_guard_does_every_push_twice_even_with_the_fair_budget(self):
        with patch("run_step2.ROUNDS", 2):
            _, pushes_made, _, _ = run("even pushing", 1, seed=0, guard="noise from repeats", budget="fair")
        np.testing.assert_array_equal(pushes_made.sum(axis=1), 128)


if __name__ == "__main__":
    unittest.main()
