"""Small checks that the pieces of step 2 do what they say.

Run them with:   python3 -m unittest test_step2 -v
"""

import unittest

import numpy as np

from critic import Critic
from network import Network
from run_step1 import PAST_SELF_AGE, PUSHES_PER_ROUND
from run_step2 import CHECK_EVERY, LEFT_TO_CHANCE, PATIENCE, choose_pushes, grade_before_training, train
from world import EDGE, OBJECTS, STRONG, TABLE, TOY, as_inputs, joined, make_world, narrowed, random_pushes


class WorldTests(unittest.TestCase):
    def test_taking_pushes_apart_and_joining_them_gives_the_same_pushes(self):
        pushes = random_pushes(10, np.ones(len(OBJECTS)), np.random.default_rng(0))
        first, rest = pushes.take(np.arange(3)), pushes.take(np.arange(3, 10))
        np.testing.assert_array_equal(as_inputs(joined(first, rest)), as_inputs(pushes))

    def test_a_bigger_world_has_the_same_mix_as_the_table(self):
        world = make_world(24, np.random.default_rng(0))
        kinds = list(world.kinds)
        self.assertEqual([kinds.count(k) for k in ("box", "roller", "spinner", "toy")], [12, 4, 4, 4])
        self.assertEqual((world.appears_at == 20).sum() + (world.appears_at == 40).sum(), 8)  # a third arrive late
        self.assertEqual(world.input_size, 24 + 6)

    def test_a_box_in_a_bigger_world_slides_its_own_distance(self):
        world = make_world(24, np.random.default_rng(0))
        box = world.kinds.index("box")
        pushes = world.random_pushes(1, np.eye(24)[box], np.random.default_rng(1))
        pushes.strength[:], pushes.direction[:] = 1.0, 0.0
        np.testing.assert_allclose(world.expected_result(pushes), [[world.reach[box], 0.0, 0.0]], atol=1e-12)

    def test_only_toys_give_a_different_result_each_time_in_a_bigger_world(self):
        world, rng = make_world(24, np.random.default_rng(0)), np.random.default_rng(1)
        pushes = world.random_pushes(2000, np.ones(24), rng)
        first, second = world.push(pushes, rng), world.push(pushes, rng)
        toy = np.asarray(world.kinds)[pushes.obj] == "toy"
        np.testing.assert_array_equal(first[~toy], second[~toy])
        self.assertTrue((first[toy] != second[toy]).all())


class CriticTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.network = Network(self.rng)
        self.everything = np.ones(len(OBJECTS))

    def toy_lover(self):
        """A critic taught that toy pushes are worth 1 and all other pushes 0."""
        critic = Critic()
        pushes = random_pushes(600, self.everything, self.rng)
        critic.learn(self.network, as_inputs(pushes), (pushes.obj == TOY).astype(float))
        return critic

    def test_scores_every_push_the_same_before_it_has_learned_anything(self):
        pushes = random_pushes(20, self.everything, self.rng)
        np.testing.assert_array_equal(Critic().score(self.network, as_inputs(pushes)), 0)

    def test_learns_which_object_its_grades_favour(self):
        critic = self.toy_lover()
        fresh = random_pushes(600, self.everything, self.rng)
        scores = critic.score(self.network, as_inputs(fresh))
        self.assertGreater(scores[fresh.obj == TOY].mean(), scores[fresh.obj != TOY].mean() + 0.5)

    def test_learning_never_changes_the_network(self):
        before = [weights.copy() for weights in self.network.weights]
        self.toy_lover()
        for old, new in zip(before, self.network.weights):
            np.testing.assert_array_equal(old, new)

    def test_the_critics_favourites_are_the_pushes_that_get_made(self):
        on_table = self.everything.astype(bool)
        chosen = choose_pushes(self.toy_lover(), self.network, TABLE, on_table, self.everything, self.rng)
        self.assertEqual(len(chosen), PUSHES_PER_ROUND)
        # Every push the critic picks should be a toy push. Only the random ones may be something else.
        self.assertGreaterEqual((chosen.obj == TOY).sum(), PUSHES_PER_ROUND - LEFT_TO_CHANCE)

    def test_progress_over_time_grades_nothing_until_there_is_a_past_self(self):
        pushes = random_pushes(8, self.everything, self.rng)
        inputs, results = as_inputs(pushes), TABLE.push(pushes, self.rng)
        predictions = self.network.predict(inputs)
        too_few = [self.network.copy()] * (PAST_SELF_AGE - 1)
        enough = [self.network.copy()] * PAST_SELF_AGE
        self.assertIsNone(grade_before_training("progress over time", self.network, too_few, predictions, inputs, results, results))
        self.assertEqual(len(grade_before_training("progress over time", self.network, enough, predictions, inputs, results, results)), 8)


class NarrowWorldTests(unittest.TestCase):
    def setUp(self):
        self.world = narrowed(TABLE)

    def one_push(self, name, strength=1.0, hit_point=0.0, direction=0.0, facing=0.0):
        pushes = self.world.random_pushes(1, np.eye(6)[OBJECTS.index(name)], np.random.default_rng(0))
        pushes.strength[:], pushes.hit_point[:], pushes.direction[:], pushes.facing[:] = strength, hit_point, direction, facing
        return pushes

    def test_a_box_moves_only_when_pushed_hard(self):
        weak = self.world.expected_result(self.one_push("wood", strength=STRONG - 0.1))
        hard = self.world.expected_result(self.one_push("wood", strength=1.0))
        np.testing.assert_allclose(weak, [[0.0, 0.0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(hard, [[1.0, 0.0, 0.0]], atol=1e-12)

    def test_a_spinner_turns_only_when_hit_near_its_edge(self):
        near_centre = self.world.expected_result(self.one_push("spinner", hit_point=EDGE - 0.1))
        at_the_edge = self.world.expected_result(self.one_push("spinner", hit_point=1.0))
        self.assertEqual(near_centre[0, 2], 0.0)
        self.assertAlmostEqual(at_the_edge[0, 2], 1.5)

    def test_a_roller_rolls_only_when_pushed_along_its_facing(self):
        along = self.world.expected_result(self.one_push("roller", direction=0.0))
        at_45_degrees = self.world.expected_result(self.one_push("roller", direction=np.pi / 4))
        np.testing.assert_allclose(along, [[1.5, 0.0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(at_45_degrees[0, :2], 0.1 * np.sin(np.pi / 4) * np.array([0.0, 1.0]), atol=1e-12)

    def test_about_half_the_test_pushes_land_in_the_narrow_parts(self):
        pushes = self.world.test_pushes(6000, np.random.default_rng(0))
        spinner = pushes.obj == OBJECTS.index("spinner")
        at_the_edge = np.abs(pushes.hit_point[spinner]) >= EDGE
        self.assertAlmostEqual(at_the_edge.mean(), 0.5 + 0.5 * (1 - EDGE), delta=0.05)

    def test_plain_worlds_draw_the_same_test_pushes_as_before(self):
        a = TABLE.test_pushes(50, np.random.default_rng(3))
        b = TABLE.random_pushes(50, np.ones(6), np.random.default_rng(3))
        np.testing.assert_array_equal(as_inputs(a), as_inputs(b))


class NoiseTests(unittest.TestCase):
    def test_backward_gives_the_true_slopes_when_the_network_predicts_its_noise(self):
        """Nudge a weight by hand, see how the loss moves, and compare with what backward() says."""
        rng = np.random.default_rng(0)
        network = Network(rng, sizes=(4, 5, 5, 3), predicts_noise=True)
        inputs, targets = rng.normal(size=(6, 4)), rng.normal(size=(6, 3))

        def loss():
            _, outputs = network._forward(inputs)
            log_variance, _ = network._log_variance(outputs)
            miss = outputs[:, :3] - targets
            return (miss**2 / (2 * np.exp(log_variance)) + log_variance / 2).sum(axis=1).mean()

        _, layer_inputs, layer_errors = network.backward(inputs, targets)
        for layer, row, column in ((0, 1, 2), (1, 1, 2), (2, 1, 2), (2, 1, 4)):  # (2, 1, 4) feeds a noise output
            said = (layer_inputs[layer].T @ layer_errors[layer] / len(inputs))[row, column]
            before = loss()
            network.weights[layer][row, column] += 1e-6
            measured = (loss() - before) / 1e-6
            network.weights[layer][row, column] -= 1e-6
            self.assertAlmostEqual(said, measured, places=4)

    def test_it_learns_that_the_toy_is_noisier_than_wood(self):
        rng = np.random.default_rng(0)
        network = Network(rng, predicts_noise=True)
        pushes = random_pushes(1000, np.eye(6)[OBJECTS.index("wood")] + np.eye(6)[TOY], rng)
        inputs, results = as_inputs(pushes), TABLE.push(pushes, rng)
        for _ in range(300):
            network.train_step(inputs, results)
        noise = network.predicted_noise(inputs).sum(axis=1)
        self.assertGreater(noise[pushes.obj == TOY].mean(), 10 * noise[pushes.obj != TOY].mean())


class GettingWorse:
    """A stand-in network that learns nothing, and whose error rises every time it predicts."""

    def __init__(self):
        self.predictions_made = 0

    def train_step(self, inputs, targets):
        pass

    def predict(self, inputs):
        self.predictions_made += 1
        return np.full((len(inputs), 3), float(self.predictions_made))


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.memory = (np.zeros((10, 12)), np.zeros((10, 3)))
        self.validation = (np.zeros((4, 12)), np.zeros((4, 3)))

    def test_without_the_guard_every_step_is_taken(self):
        self.assertEqual(train(GettingWorse(), *self.memory, 100, self.rng), 100)

    def test_the_guard_stops_once_the_validation_error_stops_falling(self):
        steps = train(GettingWorse(), *self.memory, 100, self.rng, self.validation)
        self.assertEqual(steps, CHECK_EVERY * PATIENCE)


if __name__ == "__main__":
    unittest.main()
