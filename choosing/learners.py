"""How each learner scores candidate places in the game, and what it learns from each round.

Every learner has the same things: the saved places, the world model, and its replay. Each
round (explore.py) it scores candidate places before playing (score). After playing, it sees
what happened from each place it played (see), before the world model trains on those steps,
and it learns after the world model has trained (learn). Most learners use only score.

    uniform choice          every place scores the same: "not choosing"
    cell counts             places visited least and jumped back to least score highest, with
                            Go-Explore's counts and weights for this game without game knowledge
                            (0.3 on visits, 0.1 on times chosen), so that a place jumped back to
                            again and again without paying off fades. Unlike Go-Explore, which
                            draws places in proportion to their scores, it takes the best of its
                            candidates, like every rule here.
    feature novelty         far from the states seen so far, in the world model's own view:
                            the average distance to the 5 nearest of 10,000 states sampled from
                            the replay (a fixed-size memory, so the cost does not grow)
    random-network novelty  a small network learns to imitate a fixed random network on the
                            states seen; it imitates badly on states unlike them
    ensemble disagreement   five world models trained on the same replay disagree here
    pinned weights          the guesses here rest on weights the data has barely pinned down
                            (the toy's measure, with 4 random directions in place of 128 outputs)
    steepness               the 18 buttons lead to very different guesses here: "do my choices
                            matter here?"
    ... x steepness         the place's rank on the first score times its rank on steepness
    critics                 a small network (critic.NeuralCritic) learns to forecast, from the
                            world model's view of a place, the grade exploring from it got:
        raw error               how wrong the world model was on the steps found there
        progress over 10 rounds how much better it got on them than 10 rounds before
        new places found        how many new places exploring from there turned up
        disagreement found      how much the five world models disagreed on the steps found
        do-nothing critic       a random grade: a control
Scores of measures that look at the world model use its outputs, which are each byte's change
in units of that byte's typical change, so that every byte counts the same.

CONTROLS are rules for checking the setup, not candidates:
    most visited            places visited most score highest: it keeps exploring where play
                            has been most, so it should do clearly worse than uniform choice
    room oracle             uses the room number, which no learner may: places in the rooms
                            visited least score highest. It shows how much choosing well can
                            help here at all.
"""

import numpy as np

from critic import NeuralCritic
from game import BUTTONS, MEMORY_SIZE, ROOM_BYTE, change
from measures import feature_novelty
from network import Network
from run_comparison import combined
from world_model import BATCH_SIZE, HIDDEN, WorldModel, inputs

NOTHING = 0               # the button "nothing", used where a place needs one button to be read
MEMORY_SAMPLE = 10_000    # feature novelty: states sampled from the replay to compare with
ENSEMBLE = 5              # world models in the ensemble, the learner's own included
PROJECTIONS = 4           # pinned weights: random directions over the 128 outputs
FISHER_SAMPLE = 1024      # pinned weights: steps of the replay used to measure how pinned each weight is
SNAPSHOTS = 10            # progress over 10 rounds: copies of the world model kept


def every_button(memories):
    """The world model's input for each memory with each of the 18 buttons: 18 rows per memory."""
    return inputs(np.repeat(memories, BUTTONS, axis=0), np.tile(np.arange(BUTTONS), len(memories)))


def place_view(memories):
    """The world model's input for each memory with the button "nothing": how a place is read."""
    return inputs(memories, np.full(len(memories), NOTHING))


def steepness(model, memories):
    """How different the 18 buttons' guesses are: their spread around their average, summed over the bytes."""
    guesses = model.network.predict(every_button(memories)).reshape(len(memories), BUTTONS, MEMORY_SIZE)
    return guesses.var(axis=1).sum(axis=1)


def disagreement(models, x):
    """How much the world models' guesses for inputs x spread around their average, summed over the bytes."""
    guesses = np.stack([model.network.predict(x) for model in models])
    return guesses.var(axis=0).sum(axis=1)


class UniformChoice:
    jumps_back = True  # every learner but random play jumps back to saved places

    def __init__(self, rng, model, replay):
        self.rng, self.model, self.replay = rng, model, replay

    def score(self, places, candidates):
        return np.zeros(len(candidates))

    def see(self, places, bursts):
        """What happened from each place played this round, before the world model trains on it."""

    def learn(self, places, bursts, steps):
        """After the world model has taken this round's training steps (steps of them)."""


class RandomPlay(UniformChoice):
    """Never jumps back: presses random buttons from the start of the game (explore.rounds)."""

    jumps_back = False


class CellCounts(UniformChoice):
    def score(self, places, candidates):
        visits, chosen = np.array(places.visits)[candidates], np.array(places.chosen)[candidates]
        return 0.3 / np.sqrt(visits + 0.001) + 0.1 / np.sqrt(chosen + 0.001)


class MostVisited(UniformChoice):
    def score(self, places, candidates):
        return np.array(places.visits, dtype=float)[candidates]


class RoomOracle(UniformChoice):
    def score(self, places, candidates):
        rooms = np.array(places.memory)[:, ROOM_BYTE]
        visits_per_room = np.bincount(rooms, weights=places.visits)
        return 1 / np.sqrt(visits_per_room[rooms[candidates]])


class FeatureNovelty(UniformChoice):
    def score(self, places, candidates):
        if self.replay.kept == 0:  # nothing seen yet, in the first round
            return np.zeros(len(candidates))
        seen, _, _ = self.replay.batch(MEMORY_SAMPLE)
        memories = np.array(places.memory)[candidates]
        return feature_novelty(self.model.network, place_view(memories), place_view(seen))


class RandomNetworkNovelty(UniformChoice):
    def __init__(self, rng, model, replay):
        super().__init__(rng, model, replay)
        sizes = (MEMORY_SIZE, 64, 64, 16)
        self.target, self.imitator = Network(rng, sizes), Network(rng, sizes)

    def score(self, places, candidates):
        x = np.array(places.memory)[candidates] / 255
        return ((self.imitator.predict(x) - self.target.predict(x)) ** 2).sum(axis=1)

    def learn(self, places, bursts, steps):
        for _ in range(steps):  # as many steps as the world model took
            seen, _, _ = self.replay.batch(BATCH_SIZE)
            self.imitator.train_step(seen / 255, self.target.predict(seen / 255))


class EnsembleDisagreement(UniformChoice):
    def __init__(self, rng, model, replay):
        super().__init__(rng, model, replay)
        self.models = [model] + [WorldModel(rng) for _ in range(ENSEMBLE - 1)]

    def score(self, places, candidates):
        memories = np.array(places.memory)[candidates]
        return disagreement(self.models, every_button(memories)).reshape(len(candidates), BUTTONS).mean(axis=1)

    def learn(self, places, bursts, steps):
        for other in self.models[1:]:  # the learner's own world model has trained already
            other.learn(self.replay, steps)


class PinnedWeights(UniformChoice):
    def score(self, places, candidates):
        if self.replay.kept == 0:  # nothing has pinned any weight yet, in the first round
            return np.zeros(len(candidates))
        memories = np.array(places.memory)[candidates]
        uncertainty = pinned_weights(self.model, every_button(memories), self.replay, self.rng)
        return uncertainty.reshape(len(candidates), BUTTONS).mean(axis=1)


def pinned_weights(model, x, replay, rng):
    """measures.pinned_weight_uncertainty for the world model, with random directions in place of outputs.

    The toy's measure sums, over the network's outputs, each output's squared slope with
    respect to every weight. With 128 outputs that is 128 backward passes. A random direction
    v over the outputs (each entry drawn from a standard normal) gives the same sum on average:
    the squared slope of v . outputs has expected value equal to the sum over outputs of their
    squared slopes. PROJECTIONS such directions estimate it with 4 backward passes.
    """
    network = model.network
    before, buttons, after = replay.batch(FISHER_SAMPLE)
    sample = inputs(before, buttons)
    noise = ((network.predict(sample) - change(before, after) / model.scale) ** 2).mean()  # the average squared miss
    directions = rng.normal(size=(PROJECTIONS, network.result_size))

    weight_pinning = [np.zeros_like(weights) for weights in network.weights]
    bias_pinning = [np.zeros_like(biases) for biases in network.biases]
    for direction in directions:
        layer_inputs, layer_errors = network.direction_slopes(sample, direction)
        for k, (layer_input, layer_error) in enumerate(zip(layer_inputs, layer_errors)):
            weight_pinning[k] += (layer_input**2).T @ layer_error**2 / PROJECTIONS
            bias_pinning[k] += (layer_error**2).sum(axis=0) / PROJECTIONS
    scale = replay.seen / len(sample) / noise
    prior = [weights.shape[0] / 2 for weights in network.weights]  # 1 / the variance each layer was drawn with
    weight_looseness = [1 / (scale * pinning + start) for pinning, start in zip(weight_pinning, prior)]
    bias_looseness = [1 / (scale * pinning + start) for pinning, start in zip(bias_pinning, prior)]

    uncertainty = np.zeros(len(x))
    for direction in directions:
        layer_inputs, layer_errors = network.direction_slopes(x, direction)
        for k, (layer_input, layer_error) in enumerate(zip(layer_inputs, layer_errors)):
            uncertainty += ((layer_input**2 @ weight_looseness[k]) * layer_error**2).sum(axis=1) / PROJECTIONS
            uncertainty += layer_error**2 @ bias_looseness[k] / PROJECTIONS
    return uncertainty


class Steepness(UniformChoice):
    def score(self, places, candidates):
        return steepness(self.model, np.array(places.memory)[candidates])


def times_steepness(learner_class):
    """The learner's place ranks times their ranks on steepness: high only where both are high."""

    class TimesSteepness(learner_class):
        def score(self, places, candidates):
            return combined(super().score(places, candidates), steepness(self.model, np.array(places.memory)[candidates]))

    return TimesSteepness


class Critic(UniformChoice):
    """A critic learns to forecast the grade of exploring from a place. The grade is set by label()."""

    def __init__(self, rng, model, replay):
        super().__init__(rng, model, replay)
        self.critic = NeuralCritic(np.random.default_rng(rng.integers(2**32)), feature_size=HIDDEN)
        self.grades = None

    def score(self, places, candidates):
        return self.critic.score(self.model.network, place_view(np.array(places.memory)[candidates]))

    def see(self, places, bursts):
        self.grades = np.array([self.label(burst) for burst in bursts])

    def learn(self, places, bursts, steps):
        if self.grades is None:  # nothing to learn from yet
            return
        played = place_view(np.array([places.memory[burst.place] for burst in bursts]))
        spread = self.grades.std()
        standardized = (self.grades - self.grades.mean()) / spread if spread > 0 else self.grades * 0
        self.critic.learn(self.model.network, played, standardized)

    def miss(self, model, burst):
        """How wrong a world model is on a burst's steps, each byte's miss in units of its typical change now."""
        guessed = model.guess(burst.before, burst.buttons)
        return (((guessed - change(burst.before, burst.after)) / self.model.scale) ** 2).sum(axis=1).mean()


class RawErrorCritic(Critic):
    def label(self, burst):
        return self.miss(self.model, burst)


class ProgressCritic(Critic):
    def __init__(self, rng, model, replay):
        super().__init__(rng, model, replay)
        self.past = []  # copies of the world model from earlier rounds, newest last

    def label(self, burst):
        return self.miss(self.past[0], burst) - self.miss(self.model, burst)

    def see(self, places, bursts):
        if len(self.past) == SNAPSHOTS:  # no grades before there is a copy of the world model 10 rounds old
            super().see(places, bursts)
        else:
            self.grades = None
        snapshot = WorldModel.__new__(WorldModel)  # a frozen copy of the world model as it is now
        snapshot.network, snapshot.scale = self.model.network.copy(), self.model.scale.copy()
        self.past = (self.past + [snapshot])[-SNAPSHOTS:]


class NewPlacesCritic(Critic):
    def label(self, burst):
        return burst.new_places


class DisagreementFoundCritic(Critic):
    def __init__(self, rng, model, replay):
        super().__init__(rng, model, replay)
        self.models = [model] + [WorldModel(rng) for _ in range(ENSEMBLE - 1)]

    def label(self, burst):
        return disagreement(self.models, inputs(burst.before, burst.buttons)).mean()

    def learn(self, places, bursts, steps):
        for other in self.models[1:]:
            other.learn(self.replay, steps)
        super().learn(places, bursts, steps)


class DoNothingCritic(Critic):
    def label(self, burst):
        return self.rng.normal()


LEARNERS = {
    "uniform choice": UniformChoice,
    "random play": RandomPlay,
    "do-nothing critic": DoNothingCritic,
    "cell counts": CellCounts,
    "feature novelty": FeatureNovelty,
    "feature novelty x steepness": times_steepness(FeatureNovelty),
    "random-network novelty": RandomNetworkNovelty,
    "random-network novelty x steepness": times_steepness(RandomNetworkNovelty),
    "ensemble disagreement": EnsembleDisagreement,
    "disagreement x steepness": times_steepness(EnsembleDisagreement),
    "pinned weights": PinnedWeights,
    "steepness": Steepness,
    "raw error critic": RawErrorCritic,
    "progress over 10 rounds critic": ProgressCritic,
    "new places found critic": NewPlacesCritic,
    "disagreement found critic": DisagreementFoundCritic,
}

CONTROLS = {"most visited": MostVisited, "room oracle": RoomOracle}
