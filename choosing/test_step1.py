"""Small checks that the pieces of step 1 do what they say.

Run them with:   python3 -m unittest test_step1 -v
"""

import unittest

import numpy as np

from answer_key import value_of_extra_pushes
from network import Network
from report_step1 import pair_score
from signals import forward_progress, progress_over_time, raw_error, repeat_agreement, slope_agreement
from world import OBJECTS, TOY, Pushes, as_inputs, expected_result, push, random_pushes


def one_push(name, direction=0.0, strength=1.0, hit_point=0.0, facing=0.0):
    return Pushes(
        obj=np.array([OBJECTS.index(name)]),
        direction=np.array([direction]),
        strength=np.array([strength]),
        hit_point=np.array([hit_point]),
        facing=np.array([facing]),
    )


class WorldTests(unittest.TestCase):
    def test_boxes_slide_straight_along_the_push(self):
        np.testing.assert_allclose(expected_result(one_push("wood")), [[1.0, 0.0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(expected_result(one_push("steel", direction=np.pi / 2)), [[0.0, 0.4, 0.0]], atol=1e-12)

    def test_roller_goes_far_the_way_it_faces_and_barely_sideways(self):
        along = expected_result(one_push("roller", direction=0.0, facing=0.0))
        sideways = expected_result(one_push("roller", direction=np.pi / 2, facing=0.0))
        np.testing.assert_allclose(along, [[1.5, 0.0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(sideways, [[0.0, 0.1, 0.0]], atol=1e-12)

    def test_spinner_turns_when_hit_off_centre(self):
        centred = expected_result(one_push("spinner", hit_point=0.0))
        at_the_edge = expected_result(one_push("spinner", hit_point=1.0))
        np.testing.assert_allclose(centred, [[0.8, 0.0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(at_the_edge, [[0.0, 0.0, 1.5]], atol=1e-12)

    def test_only_the_toy_gives_a_different_result_each_time(self):
        rng = np.random.default_rng(0)
        pushes = random_pushes(2000, np.ones(len(OBJECTS)), rng)
        first, second = push(pushes, rng), push(pushes, rng)
        toy = pushes.obj == TOY
        np.testing.assert_array_equal(first[~toy], second[~toy])
        self.assertTrue((first[toy] != second[toy]).all())

    def test_the_toy_averages_out_to_no_movement(self):
        rng = np.random.default_rng(0)
        pushes = random_pushes(20000, np.eye(len(OBJECTS))[TOY], rng)
        np.testing.assert_array_equal(expected_result(pushes), 0)
        np.testing.assert_allclose(push(pushes, rng).mean(axis=0), 0, atol=0.02)


class NetworkTests(unittest.TestCase):
    def test_backward_gives_the_true_slopes(self):
        """Nudge a weight by hand, see how the loss moves, and compare with what backward() says."""
        rng = np.random.default_rng(0)
        network = Network(rng, sizes=(4, 5, 5, 3))
        inputs, targets = rng.normal(size=(6, 4)), rng.normal(size=(6, 3))

        def loss():
            return ((network.predict(inputs) - targets) ** 2).sum(axis=1).mean()

        _, layer_inputs, layer_errors = network.backward(inputs, targets)
        for layer in range(3):
            said = (layer_inputs[layer].T @ layer_errors[layer] / len(inputs))[1, 2]
            before = loss()
            network.weights[layer][1, 2] += 1e-6
            measured = (loss() - before) / 1e-6
            network.weights[layer][1, 2] -= 1e-6
            self.assertAlmostEqual(said, measured, places=4)

    def test_training_makes_the_error_smaller(self):
        rng = np.random.default_rng(0)
        network = Network(rng)
        pushes = random_pushes(512, np.ones(len(OBJECTS)), rng)
        inputs, results = as_inputs(pushes), expected_result(pushes)
        before = raw_error(network.predict(inputs), results).mean()
        for _ in range(200):
            network.train_step(inputs, results)
        self.assertLess(raw_error(network.predict(inputs), results).mean(), before / 10)


class SignalTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.network = Network(self.rng)

    def test_repeat_agreement_is_the_whole_error_when_nothing_is_chance(self):
        predictions, results = self.rng.normal(size=(8, 3)), self.rng.normal(size=(8, 3))
        np.testing.assert_allclose(repeat_agreement(predictions, results, results), raw_error(predictions, results))

    def test_repeat_agreement_is_about_zero_when_the_miss_is_all_chance(self):
        predictions = np.zeros((50000, 3))
        first, second = self.rng.normal(size=(50000, 3)), self.rng.normal(size=(50000, 3))
        self.assertAlmostEqual(repeat_agreement(predictions, first, second).mean(), 0, places=1)

    def test_slope_agreement_is_positive_for_the_same_miss_and_negative_for_opposite_misses(self):
        inputs = as_inputs(one_push("wood")).repeat(2, axis=0)
        prediction = self.network.predict(inputs)
        same_miss = prediction + [[1, 0, 0], [1, 0, 0]]
        opposite_miss = prediction + [[1, 0, 0], [-1, 0, 0]]
        self.assertTrue((slope_agreement(self.network, inputs, same_miss) > 0).all())
        self.assertTrue((slope_agreement(self.network, inputs, opposite_miss) < 0).all())

    def test_a_network_makes_no_progress_against_itself(self):
        pushes = random_pushes(16, np.ones(len(OBJECTS)), self.rng)
        inputs, results = as_inputs(pushes), push(pushes, self.rng)
        np.testing.assert_array_equal(progress_over_time(self.network.copy(), self.network, inputs, results), 0)

    def test_forward_progress_is_zero_when_the_guesses_did_not_change(self):
        predictions, results = self.rng.normal(size=(8, 3)), self.rng.normal(size=(8, 3))
        np.testing.assert_array_equal(forward_progress(predictions, predictions, results), 0)

    def test_forward_progress_is_positive_when_the_guesses_moved_toward_the_results(self):
        results = self.rng.normal(size=(8, 3))
        self.assertTrue((forward_progress(results + 1.0, results + 0.5, results) > 0).all())


class AnswerKeyTests(unittest.TestCase):
    def test_without_extra_pushes_the_two_copies_stay_identical(self):
        """Both copies must see the very same batches and new pushes, so with nothing extra every gain is exactly 0."""
        rng = np.random.default_rng(0)
        everything = np.ones(len(OBJECTS))
        memory = random_pushes(512, everything, rng)
        gains = value_of_extra_pushes(
            Network(rng), as_inputs(memory), push(memory, rng), obj=0,
            test_pushes=random_pushes(300, everything, rng), on_table=everything.astype(bool), shares=everything,
            seed=1, rounds=2, extra_pushes=0, pushes_per_round=16, train_steps=5, batch_size=64,
        )
        self.assertEqual(gains, [0, 0])


class ScoreTests(unittest.TestCase):
    def test_pair_score_counts_the_gaps_it_orders_right(self):
        answer_key = np.array([[3.0, 2.0, 1.0]])
        self.assertEqual(pair_score(np.array([[30.0, 20.0, 10.0]]), answer_key), 1.0)
        self.assertEqual(pair_score(np.array([[10.0, 20.0, 30.0]]), answer_key), 0.0)
        # Wrong only about the second against the third object: a gap of 1, out of 1 + 2 + 1.
        self.assertEqual(pair_score(np.array([[30.0, 10.0, 20.0]]), answer_key), 0.75)


if __name__ == "__main__":
    unittest.main()
