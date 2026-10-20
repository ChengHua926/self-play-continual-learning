"""Small checks that the learners of the game do what they say (learners.py).

Run them with:   .venv/bin/python -m unittest test_learners -v
"""

import unittest
from unittest.mock import patch

import numpy as np

import explore
from explore import Burst, press_randomly, start
from game import BUTTONS, MEMORY_SIZE
from learners import (
    LEARNERS, SNAPSHOTS, NewPlacesCritic, ProgressCritic, RawErrorCritic, disagreement, every_button, steepness,
)
from network import Network
from test_set import TestSet
from world_model import Replay, WorldModel


def a_small_world(seed=0):
    """A short random exploration: its places, a replay of its steps, and a world model trained a little on them."""
    rng = np.random.default_rng(seed)
    game, places = start(seed)
    before, buttons, after, _ = press_randomly(game, rng, places, presses=3_000, stop_at_lost_life=False)
    replay = Replay(rng)
    replay.add(before, buttons, after)
    model = WorldModel(rng)
    model.learn(replay, steps=100)
    return rng, places, replay, model, (before, buttons, after)


class ScoreTests(unittest.TestCase):
    def test_every_learner_scores_every_candidate_with_a_real_number(self):
        rng, places, replay, model, _ = a_small_world()
        candidates = rng.integers(len(places), size=40)
        for name, learner_class in LEARNERS.items():
            learner = learner_class(np.random.default_rng(1), model, replay)
            score = learner.score(places, candidates)
            self.assertEqual(score.shape, (40,), name)
            self.assertTrue(np.isfinite(score).all(), name)

    def test_steepness_is_zero_where_the_buttons_make_no_difference(self):
        rng, places, replay, model, _ = a_small_world()
        model.network.weights[0][MEMORY_SIZE:] = 0  # the network no longer reads the button switches
        memories = np.array(places.memory[:10])
        np.testing.assert_allclose(steepness(model, memories), 0, atol=1e-12)

    def test_identical_world_models_do_not_disagree(self):
        rng, places, replay, model, _ = a_small_world()
        x = every_button(np.array(places.memory[:5]))
        np.testing.assert_allclose(disagreement([model, model, model], x), 0, atol=1e-12)

    def test_random_directions_give_each_weights_summed_squared_slope_on_average(self):
        # The base of pinned weights: averaged over many random directions over the outputs, a direction's
        # squared slope with respect to a weight equals the sum over outputs of each output's squared slope.
        rng = np.random.default_rng(0)
        network = Network(rng, sizes=(6, 16, 16, 4))
        x = rng.normal(size=(3, 6))

        def first_layer_squared_slopes(slopes):  # each first-layer weight's squared slope, summed over the examples
            layer_inputs, layer_errors = slopes
            return (layer_inputs[0] ** 2).T @ layer_errors[0] ** 2

        exact = sum(first_layer_squared_slopes(network.output_slopes(x, output)) for output in range(4))
        estimate = np.mean([first_layer_squared_slopes(network.direction_slopes(x, direction))
                            for direction in rng.normal(size=(20_000, 4))], axis=0)
        np.testing.assert_allclose(estimate, exact, rtol=0.05, atol=1e-6)


class CriticTests(unittest.TestCase):
    def bursts(self, places, steps, count=4):
        before, buttons, after = steps
        size = len(buttons) // count
        return [Burst(place=i, before=before[i * size:(i + 1) * size], buttons=buttons[i * size:(i + 1) * size],
                      after=after[i * size:(i + 1) * size], met=set(), new_places=i) for i in range(count)]

    def test_a_critic_gets_one_grade_per_burst_and_learns_from_them(self):
        rng, places, replay, model, steps = a_small_world()
        bursts = self.bursts(places, steps)
        for critic_class in (RawErrorCritic, NewPlacesCritic):
            critic = critic_class(rng, model, replay)
            critic.see(places, bursts)
            self.assertEqual(critic.grades.shape, (4,))
            critic.learn(places, bursts, steps=5)
            self.assertTrue(critic.critic.learned)
        critic = NewPlacesCritic(rng, model, replay)
        critic.see(places, bursts)
        np.testing.assert_array_equal(critic.grades, [0, 1, 2, 3])  # the new places each burst found

    def test_progress_is_graded_only_once_a_copy_10_rounds_old_exists(self):
        rng, places, replay, model, steps = a_small_world()
        bursts = self.bursts(places, steps)
        critic = ProgressCritic(rng, model, replay)
        for round_number in range(SNAPSHOTS + 1):
            critic.see(places, bursts)
            if round_number < SNAPSHOTS:
                self.assertIsNone(critic.grades)
            critic.learn(places, bursts, steps=1)
            model.learn(replay, steps=5)
        self.assertEqual(critic.grades.shape, (4,))
        self.assertTrue(critic.critic.learned)


class RunTests(unittest.TestCase):
    def test_every_learner_runs_and_is_measured(self):
        _, _, _, _, (before, buttons, after) = a_small_world()
        small_test_set = TestSet(before[:500], buttons[:500].astype(np.uint8), after[:500])
        with patch("explore.MEASURE_EVERY", 15_000):
            for name in LEARNERS:
                lines = explore.run(name, 0, small_test_set, frames=30_000)
                self.assertEqual(len(lines["frames"]), 2, name)
                self.assertTrue(((lines["knowledge"] >= 0) & (lines["knowledge"] <= 1)).all(), name)


if __name__ == "__main__":
    unittest.main()
