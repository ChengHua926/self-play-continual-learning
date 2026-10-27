"""Small checks that the pieces of the toy comparison do what they say.

Run them with:   python3 -m unittest test_comparison -v
"""

import unittest
from unittest.mock import patch

import numpy as np

import run_comparison
from critic import Critic, NeuralCritic
from measures import steepness
from network import Network
from report_comparison import pushes_saved
from run_comparison import (
    COPIES, LEARNERS, NUDGE, SNAPSHOTS, Learner, combined, halves, label_before_training, measured,
)
from signals import teaching_value
from world import RESULT_SIZE, make_world, near_copies


def a_learner(world, rng):
    """A learner with everything every plan can need: memory, pushes seen, an ensemble, and old copies of itself."""
    sizes, small = (world.input_size, 128, 128, 128, RESULT_SIZE), (world.input_size, 64, 64, 16)
    me = Learner(Network(rng, sizes), settings_from=len(world.names), rng=rng)
    me.others = [Network(rng, sizes) for _ in range(4)]
    me.target, me.imitator = Network(rng, small), Network(rng, small)
    pushes = world.random_pushes(300, np.ones(len(world.names)), rng)
    me.memory_inputs, me.memory_results = world.as_inputs(pushes), world.push(pushes, rng)
    me.seen = me.memory_inputs
    me.past = [me.network.copy() for _ in range(SNAPSHOTS)]
    for _ in range(5):  # so that the network now differs from its old copies
        me.network.train_step(me.memory_inputs, me.memory_results)
    return me


class BudgetTests(unittest.TestCase):
    def test_every_learner_spends_256_actions_per_round(self):
        for name, plan in LEARNERS.items():
            a, b = halves(plan)
            actions = a + b + (b if plan.repeats else 0) + b * plan.copies
            self.assertEqual(actions, 256, name)

    def test_every_learner_adds_256_records_to_memory_each_round(self):
        for name, half_a in (("raw error", 128), ("repeat agreement", 128), ("near copies 2%", 96)):
            sizes = []

            def counting_train(network, memory_inputs, *rest):  # stands in for train: counts, trains nothing
                sizes.append(len(memory_inputs))
                return 0

            with patch("run_comparison.ROUNDS", 2), patch("run_comparison.train", counting_train):
                run_comparison.run(name, "6", 0)
            # Half A goes in before training, half B and its repeats or copies after it.
            self.assertEqual(sizes, [half_a, 256 + half_a], name)


class SteepnessTests(unittest.TestCase):
    def test_steepness_matches_nudging_each_setting_of_a_push_a_little(self):
        rng = np.random.default_rng(0)
        world = make_world(6, rng)
        network = Network(rng, sizes=(world.input_size, 32, 32, RESULT_SIZE))
        pushes = world.random_pushes(5, np.ones(6), rng)
        tiny = 1e-6
        by_nudging = np.zeros(5)
        for setting in ("direction", "strength", "hit_point", "facing"):
            up, down = pushes.take(np.arange(5)), pushes.take(np.arange(5))  # copies
            setattr(up, setting, getattr(pushes, setting) + tiny)
            setattr(down, setting, getattr(pushes, setting) - tiny)
            moved = network.predict(world.as_inputs(up)) - network.predict(world.as_inputs(down))
            by_nudging += ((moved / (2 * tiny)) ** 2).sum(axis=1)
        steep = steepness(network, world.as_inputs(pushes), settings_from=len(world.names))
        np.testing.assert_allclose(steep, by_nudging, rtol=1e-4)


class CombinedTests(unittest.TestCase):
    def test_a_push_scores_high_only_if_it_is_high_on_both_parts(self):
        score = np.array([10.0, 10.0, 0.0, 0.0])  # pushes: high on both, high on one, high on the other, low on both
        other = np.array([5.0, 0.0, 5.0, 0.0])
        both = combined(score, other)
        self.assertEqual(both.argmax(), 0)
        self.assertGreater(both[0], 2 * max(both[1], both[2]))
        # Only the order within each part counts, so stretching a part changes nothing.
        np.testing.assert_allclose(combined(score**3, other * 100), both)


class NeuralCriticTests(unittest.TestCase):
    def test_it_learns_which_object_its_grades_favour(self):
        rng = np.random.default_rng(0)
        world = make_world(6, rng)
        network = Network(rng, sizes=(world.input_size, 128, 128, 128, RESULT_SIZE))
        critic = NeuralCritic(np.random.default_rng(1))
        test = world.random_pushes(600, np.ones(6), rng)
        np.testing.assert_array_equal(critic.score(network, world.as_inputs(test)), 0)  # nothing learned yet
        for _ in range(5):
            pushes = world.random_pushes(128, np.ones(6), rng)
            critic.learn(network, world.as_inputs(pushes), (pushes.obj == 2).astype(float))
        forecasts = critic.score(network, world.as_inputs(test))
        self.assertGreater(forecasts[test.obj == 2].min(), forecasts[test.obj != 2].max())

    def test_it_keeps_only_the_last_rounds(self):
        rng = np.random.default_rng(0)
        network = Network(rng, sizes=(8, 128, 128, 128, RESULT_SIZE))
        critic = NeuralCritic(np.random.default_rng(1), rounds_remembered=3, steps=1)
        for round_number in range(5):
            critic.learn(network, rng.normal(size=(10, 8)), np.full(10, float(round_number)))
        self.assertEqual([grades[0] for _, grades in critic.recent], [2.0, 3.0, 4.0])


class NearCopyTests(unittest.TestCase):
    def test_near_copies_keep_the_object_and_stay_within_the_nudge(self):
        rng = np.random.default_rng(0)
        world = make_world(6, rng)
        pushes = world.random_pushes(50, np.ones(6), rng)
        copies = near_copies(pushes, NUDGE, rng, COPIES)
        original = pushes.take(np.repeat(np.arange(50), COPIES))  # each push, once for each of its copies
        self.assertEqual(len(copies), 50 * COPIES)
        np.testing.assert_array_equal(copies.obj, original.obj)
        for setting, span in (("direction", np.pi), ("strength", 0.8), ("hit_point", 2.0), ("facing", np.pi)):
            moved = np.abs(getattr(copies, setting) - getattr(original, setting))
            self.assertLessEqual(moved.max(), NUDGE * span + 1e-12, setting)
            self.assertGreater(moved.max(), 0, setting)
        self.assertTrue(((copies.strength >= 0.2) & (copies.strength <= 1.0)).all())
        self.assertTrue((np.abs(copies.hit_point) <= 1).all())


class TeachingValueTests(unittest.TestCase):
    def test_a_push_is_not_compared_with_itself(self):
        agreement = np.array([[100.0, 1, 2], [1, 100, 3], [2, 3, 100]])
        yardstick = np.array([True, True, False])
        # Pushes 0 and 1 are in the yardstick, so each is compared with the other only. Push 2 is compared with both.
        np.testing.assert_allclose(teaching_value(agreement, yardstick), [1.0, 1.0, 2.5])


class WiringTests(unittest.TestCase):
    """Every label and measure a plan names must be read: a misspelt name would leave a learner choosing at random."""

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.world = make_world(6, self.rng)
        self.me = a_learner(self.world, self.rng)

    def test_every_label_gives_one_finite_grade_per_push(self):
        world, me, rng = self.world, self.me, self.rng
        pushes = world.random_pushes(32, np.ones(6), rng)
        inputs, first, second = world.as_inputs(pushes), world.push(pushes, rng), world.push(pushes, rng)
        copies = near_copies(pushes, NUDGE, rng, COPIES)
        copy_inputs, copy_results = world.as_inputs(copies), world.push(copies, rng)
        yardstick = rng.random(32) < 0.25
        for name, plan in LEARNERS.items():
            if plan.label in (None, "forward progress"):  # forward progress is read after training
                continue
            grades = label_before_training(plan, me, me.network.predict(inputs), inputs, first, second, copy_inputs,
                                           copy_results, yardstick, pushes.obj)
            self.assertEqual(grades.shape, (32,), name)
            self.assertTrue(np.isfinite(grades).all(), name)

    def test_every_measure_gives_one_finite_score_per_candidate(self):
        inputs = self.world.as_inputs(self.world.random_pushes(512, np.ones(6), self.rng))
        for name, plan in LEARNERS.items():
            if plan.measure is None:
                continue
            scores = measured(plan.measure, self.me, inputs)
            self.assertEqual(scores.shape, (512,), name)
            self.assertTrue(np.isfinite(scores).all(), name)
            self.assertGreater(scores.std(), 0, name)


class RunTests(unittest.TestCase):
    def test_every_learner_runs_and_every_critic_learns(self):
        """12 rounds: enough for the label "progress over 10 rounds" to be read once."""
        neural, line = NeuralCritic.learn, Critic.learn
        for name, plan in LEARNERS.items():
            with (patch("run_comparison.ROUNDS", 12),
                  patch.object(NeuralCritic, "learn", autospec=True, side_effect=neural) as neural_learned,
                  patch.object(Critic, "learn", autospec=True, side_effect=line) as line_learned):
                lines = run_comparison.run(name, "narrow 6", 0)
            for line_name, values in lines.items():
                self.assertEqual(values.shape, (12,), (name, line_name))
            self.assertTrue(np.isfinite(lines["learnable"]).all(), name)
            learned = neural_learned.call_count + line_learned.call_count
            if plan.label is None:
                self.assertEqual(learned, 0, name)
            else:
                self.assertGreater(learned, 0, name)


class PushesSavedTests(unittest.TestCase):
    rounds = np.arange(70.0)
    reference = 1 / (1 + rounds)  # the error of not choosing falls round by round

    def test_sooner_saves_pushes_and_never_leaves_a_gap(self):
        faster, slower = 1 / (1 + 2 * self.rounds), 2 / (1 + self.rounds)
        lines = np.stack([self.reference, faster, slower])[:, None, None, :]  # [learner, world, run, round]
        share, at_cap = pushes_saved(lines, reference=0, first_window=0)
        self.assertEqual(share[0, 0], 0)
        self.assertGreater(share[1, 0], 0.4)
        self.assertTrue(np.isnan(share[2, 0]))
        self.assertFalse(at_cap.any())

    def test_a_low_error_before_the_last_arrival_does_not_count(self):
        early_dip = self.reference.copy()
        early_dip[:20] = 0  # as if the error were low only because the late objects were not there yet
        lines = np.stack([self.reference, early_dip])[:, None, None, :]
        share, at_cap = pushes_saved(lines, reference=0, first_window=40)
        self.assertEqual(share[1, 0], 0)
        self.assertFalse(at_cap[1, 0])
        share, at_cap = pushes_saved(lines, reference=0, first_window=0)
        self.assertGreater(share[1, 0], 0.5)  # without the rule, the dip would look like a big saving
        self.assertTrue(at_cap[1, 0])


if __name__ == "__main__":
    unittest.main()
