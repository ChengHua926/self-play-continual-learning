"""Small checks that the pieces of step 2 do what they say.

Run them with:   python3 -m unittest test_step2 -v
"""

import unittest

import numpy as np

from critic import Critic
from network import Network
from run_step1 import PUSHES_PER_ROUND
from run_step2 import LEFT_TO_CHANCE, choose_pushes
from world import OBJECTS, TOY, as_inputs, joined, random_pushes


class WorldTests(unittest.TestCase):
    def test_taking_pushes_apart_and_joining_them_gives_the_same_pushes(self):
        pushes = random_pushes(10, np.ones(len(OBJECTS)), np.random.default_rng(0))
        first, rest = pushes.take(np.arange(3)), pushes.take(np.arange(3, 10))
        np.testing.assert_array_equal(as_inputs(joined(first, rest)), as_inputs(pushes))


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
        chosen = choose_pushes(self.toy_lover(), self.network, self.everything.astype(bool), self.rng)
        self.assertEqual(len(chosen), PUSHES_PER_ROUND)
        # Every push the critic picks should be a toy push. Only the random ones may be something else.
        self.assertGreaterEqual((chosen.obj == TOY).sum(), PUSHES_PER_ROUND - LEFT_TO_CHANCE)


if __name__ == "__main__":
    unittest.main()
