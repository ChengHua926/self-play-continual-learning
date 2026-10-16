"""The critic: a plain head that forecasts how good a push would be, before it is made.

It reads the learner's last hidden layer and draws a straight line from those 128
numbers to one forecast. It learns from the grades the learner's pushes got in the last
few rounds, by least squares. The learner's network is only read, never changed.

A straight-line head cannot stay hopeful about places it has never seen: whatever it
learns from the pushes it has seen, it applies everywhere. So step 2 does not rely on it
for that. A share of every round's pushes is left to chance instead.
"""

import numpy as np


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


def _with_one(features):
    """Add a column of ones, so that the head can also learn a constant."""
    return np.column_stack([features, np.ones(len(features))])
