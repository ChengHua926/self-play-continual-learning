"""The world model of the game: given the memory now and a button, guess how the memory changes.

It is the network of the toy (network.py), with 3 hidden layers of 256. It reads the 128 bytes
of memory (each divided by 255) and 18 on/off switches, one per button, with exactly one on.
It guesses the change of every byte, measured the short way round (game.change).

Every byte counts the same. A byte's change is learned in units of that byte's typical change:
the root mean square of its change over every step the learner has made (Replay.typical_change),
at least 1. Without this, two bytes that jump around like a random number generator (bytes 5
and 6, about 60% of all squared change in a long run) would dominate the training, and no
network can predict them.

It learns from a memory of the steps it has made (Replay). The game makes about two million
steps in a run, too many to keep, so the replay keeps an even random sample of all of them.
"""

import numpy as np

from game import BUTTONS, MEMORY_SIZE, change
from network import Network

HIDDEN = 256
BATCH_SIZE = 256
REPLAY_SIZE = 200_000


def inputs(memories, buttons):
    """What the world model reads: the memory, scaled to 0-1, and one switch per button."""
    switches = np.zeros((len(buttons), BUTTONS))
    switches[np.arange(len(buttons)), buttons] = 1
    return np.concatenate([memories / 255, switches], axis=1)


class WorldModel:
    def __init__(self, rng):
        self.network = Network(rng, sizes=(MEMORY_SIZE + BUTTONS, HIDDEN, HIDDEN, HIDDEN, MEMORY_SIZE))
        self.scale = np.ones(MEMORY_SIZE)  # each byte's typical change, in bytes (at least 1)

    def guess(self, memories, buttons):
        """The change of every byte the world model expects, in bytes."""
        return self.network.predict(inputs(memories, buttons)) * self.scale

    def learn(self, replay, steps):
        """Take some training steps on the replay, each byte's change in units of its typical change so far."""
        self.scale = replay.typical_change()
        for _ in range(steps):
            before, buttons, after = replay.batch(BATCH_SIZE)
            self.network.train_step(inputs(before, buttons), change(before, after) / self.scale)


class Replay:
    """An even random sample of every step made so far: memory before, button, memory after.

    Each new step replaces a random kept one with the right probability, so that every step
    ever made is equally likely to be kept (this is called reservoir sampling).
    """

    def __init__(self, rng, size=REPLAY_SIZE):
        self.rng = rng
        self.before = np.zeros((size, MEMORY_SIZE), dtype=np.uint8)
        self.button = np.zeros(size, dtype=np.uint8)
        self.after = np.zeros((size, MEMORY_SIZE), dtype=np.uint8)
        self.seen = 0  # steps offered so far
        self.kept = 0  # steps held now
        self.squared_change = np.zeros(MEMORY_SIZE)  # each byte's squared change, summed over every step offered

    def add(self, before, buttons, after):
        self.squared_change += (change(before, after).astype(float) ** 2).sum(axis=0)
        for b, button, a in zip(before, buttons, after):
            if self.kept < len(self.button):
                slot = self.kept
                self.kept += 1
            else:
                slot = self.rng.integers(self.seen + 1)
                if slot >= len(self.button):
                    self.seen += 1
                    continue
            self.before[slot], self.button[slot], self.after[slot] = b, button, a
            self.seen += 1

    def typical_change(self):
        """Each byte's root mean square change over every step offered so far, in bytes, at least 1."""
        return np.maximum(np.sqrt(self.squared_change / max(self.seen, 1)), 1)

    def batch(self, size):
        picked = self.rng.integers(self.kept, size=size)
        return self.before[picked], self.button[picked], self.after[picked]
