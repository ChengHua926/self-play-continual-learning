"""The loop every learner runs in the game: choose saved places, jump back, explore, learn.

Each round:
    1. choose 32 saved places to jump back to: the learner's 24 best-scored of 8 x 24 random
       candidates, plus 8 picked at random (a quarter left to chance, as in the toy)
    2. from each, press up to 50 random buttons, holding the previous button with probability
       0.95, because random presses that change every frame barely move the man (Go-Explore
       explores the same way). A run of presses ends early when a life is lost.
    3. every screen code not met before is saved as a new place
    4. the steps go into the world model's replay, and the world model takes one training step
       for every 50 steps played, so that every learner gets the same training per frame
Every MEASURE_EVERY frames the world model's knowledge of the game is measured on the test set
(test_set.py), together with which of the test set's rooms the run has reached so far.

Random play is the one learner that never jumps back: it presses random buttons the same way
from the start of the game, and starts the game again when it is over.

Learners differ only in how they score candidate places (SCORES). For the positive control:
    uniform choice   every place scores the same, so the choice is random: "not choosing"
    cell counts      places visited least score highest (Go-Explore's idea)
"""

import numpy as np

from game import BUTTONS, ROOM_BYTE, Game
from world_model import Replay, WorldModel

FRAMES = 8_000_000
PLACES_PER_ROUND = 32
CANDIDATES_PER_PLACE = 8
LEFT_TO_CHANCE = 0.25
PRESSES = 50           # presses after each jump back
HOLD = 0.95            # chance of holding the previous button
STEPS_PER_TRAINING_STEP = 50  # the world model takes one training step per 50 steps played
MEASURE_EVERY = 100_000


class Places:
    """The saved places: one saved game per screen code, with its memory and how often it was visited."""

    def __init__(self):
        self.index = {}   # screen code -> place number
        self.saved, self.memory, self.visits = [], [], []

    def __len__(self):
        return len(self.saved)

    def visit(self, game):
        """Count a visit to the place the game is in now. Returns whether the place is new."""
        code = game.screen_code()
        if code in self.index:
            self.visits[self.index[code]] += 1
            return False
        self.index[code] = len(self.saved)
        self.saved.append(game.save())
        self.memory.append(game.memory())
        self.visits.append(1)
        return True


def uniform_choice(places, candidates):
    return np.zeros(len(candidates))


def cell_counts(places, candidates):
    return 1 / np.sqrt(np.array(places.visits)[candidates])


SCORES = {"uniform choice": uniform_choice, "cell counts": cell_counts}


def choose(rule, places, rng):
    """The place numbers to jump back to this round: the best-scored candidates, plus some left to chance."""
    chance = round(LEFT_TO_CHANCE * PLACES_PER_ROUND)
    best = PLACES_PER_ROUND - chance
    candidates = rng.integers(len(places), size=CANDIDATES_PER_PLACE * best)
    score = SCORES[rule](places, candidates)
    return np.concatenate([candidates[np.argsort(-score, kind="stable")[:best]], rng.integers(len(places), size=chance)])


def press_randomly(game, rng, places, presses, stop_at_lost_life):
    """Press random buttons, holding the previous one with probability HOLD. Returns the steps made."""
    before, buttons, after = [], [], []
    lives, button = game.lives(), rng.integers(BUTTONS)
    for _ in range(presses):
        if rng.random() > HOLD:
            button = rng.integers(BUTTONS)
        before.append(game.memory())
        game.press(button)
        after.append(game.memory())
        buttons.append(button)
        if game.over():
            if stop_at_lost_life:
                break
            game.start_again()
            lives = game.lives()
            continue
        if game.lives() < lives:
            if stop_at_lost_life:
                break
            lives = game.lives()
        places.visit(game)
    return np.array(before), np.array(buttons), np.array(after)


def rounds(rule, game, places, rng, frames):
    """Play rounds until the frame budget is spent. Yields the steps of each round."""
    while game.frames_played < frames:
        if rule == "random play":
            yield press_randomly(game, rng, places, PLACES_PER_ROUND * PRESSES, stop_at_lost_life=False)
            continue
        steps = []
        for place in choose(rule, places, rng):
            game.restore(places.saved[place])
            steps.append(press_randomly(game, rng, places, PRESSES, stop_at_lost_life=True))
        yield tuple(np.concatenate(part) for part in zip(*steps))


def start(seed):
    """A new game and its first saved place."""
    game = Game(seed)
    places = Places()
    places.visit(game)
    return game, places


def run(rule, seed, test_set, frames=FRAMES):
    """One learner, one run. Returns lines measured every MEASURE_EVERY frames."""
    rng = np.random.default_rng(seed)
    game, places = start(seed)
    model, replay = WorldModel(rng), Replay(rng)
    rooms = {int(game.memory()[ROOM_BYTE])}
    owed, trained = 0.0, 0  # training steps earned by the steps played but not yet taken; steps taken
    lines = {"frames": [], "knowledge": [], "room_errors": [], "room_squared_misses": [], "rooms_reached": [],
             "rooms": [], "places": [], "training_steps": []}
    for before, buttons, after in rounds(rule, game, places, rng, frames):
        replay.add(before, buttons, after)
        owed += len(buttons) / STEPS_PER_TRAINING_STEP
        model.learn(replay, int(owed))
        trained += int(owed)
        owed -= int(owed)
        rooms.update(int(room) for room in np.unique(after[:, ROOM_BYTE]))
        if game.frames_played >= MEASURE_EVERY * (len(lines["frames"]) + 1):
            knowledge, room_errors, squared_misses = test_set.knowledge(model)
            lines["frames"].append(game.frames_played)
            lines["knowledge"].append(knowledge)
            lines["room_errors"].append(room_errors)
            lines["room_squared_misses"].append(squared_misses.astype(np.float32))
            lines["rooms_reached"].append(np.isin(test_set.rooms, list(rooms)))  # which test rooms it has been in
            lines["rooms"].append(len(rooms))
            lines["places"].append(len(places))
            lines["training_steps"].append(trained)
    return {name: np.array(values) for name, values in lines.items()}
