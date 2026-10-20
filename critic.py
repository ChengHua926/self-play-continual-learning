"""The critic: a plain head that forecasts how good a push would be, before it is made.

It reads the learner's last hidden layer and draws a straight line from those 128
numbers to one forecast. It learns from the grades the learner's pushes got in the last
few rounds, by least squares. The learner's network is only read, never changed.

A straight-line head cannot stay hopeful about places it has never seen: whatever it
learns from the pushes it has seen, it applies everywhere. So step 2 does not rely on it
for that. A share of every round's pushes is left to chance instead.

NeuralCritic, added for the toy comparison, is the same idea with a small network of its own
in place of the straight line, and it keeps learning from round to round.

At the bottom are two tools to measure a critic (used by the critic check in run_step2.py):
forecast_agreement, and object_average, the simplest forecaster to compare it with.
"""

import numpy as np
from scipy.stats import spearmanr

from network import Network


class Critic:
    def __init__(self, rounds_remembered=3, penalty=1.0):
        self.rounds_remembered = rounds_remembered  # grades older than this are dropped: they go stale
        self.penalty = penalty                      # keeps the weights small when there are few grades
        self.recent = []                            # (inputs, grades), one entry per round
        self.weights = None                         # nothing learned yet: every push gets the same score

    def score(self, network, inputs):
        """The forecast for each push."""
        if self.weights is None:
            return np.zeros(len(inputs))
        return _with_one(network.features(inputs)) @ self.weights

    def learn(self, network, inputs, grades):
        """Add one round of graded pushes, and fit the head again to the recent rounds."""
        self.recent = (self.recent + [(inputs, grades)])[-self.rounds_remembered:]
        features = _with_one(network.features(np.concatenate([inputs for inputs, _ in self.recent])))
        targets = np.concatenate([grades for _, grades in self.recent])

        # Least squares with a small penalty on large weights (also called ridge regression).
        size = features.shape[1]
        self.weights = np.linalg.solve(features.T @ features + self.penalty * np.eye(size), features.T @ targets)


class NeuralCritic:
    """A small network that reads the learner's last hidden layer and forecasts a push's grade.

    It is a side branch: it reads the learner's features and never changes the learner's
    network. Unlike Critic it is not refit from scratch every round. It keeps its weights,
    holds the graded pushes of the last few rounds, and takes a number of small training
    steps on them every round. Grades are best given standardized (see run_comparison.py).
    """

    def __init__(self, rng, rounds_remembered=10, steps=100, batch_size=128, hidden=64, feature_size=128):
        self.rng = rng
        self.net = Network(rng, sizes=(feature_size, hidden, hidden, 1))
        self.rounds_remembered = rounds_remembered
        self.steps = steps
        self.batch_size = batch_size
        self.recent = []      # (inputs, grades), one entry per round
        self.learned = False  # nothing learned yet: every push gets the same score

    def score(self, network, inputs):
        """The forecast for each push."""
        if not self.learned:
            return np.zeros(len(inputs))
        return self.net.predict(network.features(inputs))[:, 0]

    def learn(self, network, inputs, grades):
        """Add one round of graded pushes, and take some training steps on the recent rounds."""
        self.recent = (self.recent + [(inputs, grades)])[-self.rounds_remembered:]
        features = network.features(np.concatenate([inputs for inputs, _ in self.recent]))
        targets = np.concatenate([grades for _, grades in self.recent])[:, None]
        for _ in range(self.steps):
            picked = self.rng.integers(0, len(features), self.batch_size)
            self.net.train_step(features[picked], targets[picked])
        self.learned = True


def forecast_agreement(forecasts, grades):
    """How well forecasts put pushes in the same order as the grades the pushes then got.

    1 means the same order, 0 means no relation, -1 means the opposite order. Only the
    order counts (this is the rank correlation, also called Spearman's), so the forecasts
    need not have the same scale as the grades. Pushes with equal forecasts share a rank.
    """
    if len(grades) < 3 or forecasts.std() == 0 or grades.std() == 0:
        return np.nan  # too few pushes, or no order to compare
    return spearmanr(forecasts, grades).statistic


def object_average(objects, past_objects, past_grades):
    """The simplest forecaster to compare the critic with: each push gets the average grade
    that its object got in the past. It knows which object a push is for, and nothing else.

    An object without past grades gets the average of all past grades.
    """
    forecasts = np.full(len(objects), past_grades.mean())
    for obj in np.unique(objects):
        graded = past_objects == obj
        if graded.any():
            forecasts[objects == obj] = past_grades[graded].mean()
    return forecasts


def _with_one(features):
    """Add a column of ones, so that the head can also learn a constant."""
    return np.column_stack([features, np.ones(len(features))])
