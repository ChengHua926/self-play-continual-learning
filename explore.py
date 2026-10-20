"""The loop every learner runs in the game: choose saved places, jump back, explore, learn.

Each round:
    1. choose 32 saved places to jump back to: the learner's 24 best-scored of 8 x 24 random
       candidates, plus 8 picked at random (a quarter left to chance, as in the toy)
    2. from each, press up to 50 random buttons, holding the previous button with probability
       0.95, because random presses that change every frame barely move the man (Go-Explore
       explores the same way). A run of presses ends early when a life is lost.
    3. every screen code not met before is saved as a new place, except those met in the last 20
       presses before a lost life: the game counts a death up to 17 presses late, so in those
       places the man was already falling to his death (DOOMED)
    4. the steps go into the world model's replay, and the world model takes one training step
       for every 50 steps played, so that every learner gets the same training per frame
Every MEASURE_EVERY frames the world model's knowledge of the game is measured on the test set
(test_set.py), together with which of the test set's rooms the run has reached so far.

Random play is the one learner that never jumps back: it presses random buttons the same way
from the start of the game, and starts the game again when it is over.

Learners differ only in how they score candidate places, and in what some of them learn from
what each run of presses found (learners.py).
"""

from dataclasses import dataclass

import numpy as np

from game import BUTTONS, FRAME_SKIP, ROOM_BYTE, Game
from learners import CONTROLS, LEARNERS
from world_model import Replay, WorldModel

FRAMES = 8_000_000
PLACES_PER_ROUND = 32
CANDIDATES_PER_PLACE = 8
LEFT_TO_CHANCE = 0.25
PRESSES = 50           # presses after each jump back
HOLD = 0.95            # chance of holding the previous button
DOOMED = 20            # places met this many presses or fewer before a lost life are not saved
STEPS_PER_TRAINING_STEP = 50  # the world model takes one training step per 50 steps played
MEASURE_EVERY = 100_000


class Places:
    """The saved places: one saved game per screen code, with its memory, how often play passed
    through it (visits), and how often a learner jumped back to it (chosen)."""

    def __init__(self):
        self.index = {}   # screen code -> place number
        self.saved, self.memory, self.visits, self.chosen = [], [], [], []

    def __len__(self):
        return len(self.saved)

    def add(self, code, saved, memory, visits=1):
        self.index[code] = len(self.saved)
        self.saved.append(saved)
        self.memory.append(memory)
        self.visits.append(visits)
        self.chosen.append(0)


@dataclass
class Burst:
    """The steps of one run of presses: memory before, button, memory after, and the screen codes met.

    new_places is set at the end of the round: how many places saved this round it met.
    """

    place: int  # the place it started from (None for random play)
    before: np.ndarray
    buttons: np.ndarray
    after: np.ndarray
    met: set
    new_places: int = 0


def choose(learner, places, rng):
    """The place numbers to jump back to this round: the best-scored candidates, plus some left to chance."""
    chance = round(LEFT_TO_CHANCE * PLACES_PER_ROUND)
    best = PLACES_PER_ROUND - chance
    candidates = rng.integers(len(places), size=CANDIDATES_PER_PLACE * best)
    score = learner.score(places, candidates)
    assert np.isfinite(score).all(), "a score that is not a number would leave the choice to the candidates' order"
    return np.concatenate([candidates[np.argsort(-score, kind="stable")[:best]], rng.integers(len(places), size=chance)])


def press_randomly(game, rng, places, presses, stop_at_lost_life):
    """Press random buttons, holding the previous one with probability HOLD.

    A screen code not met before waits DOOMED presses before it is saved as a place; if a life is
    lost first, it is dropped. If the presses run out while codes are still waiting, pressing goes
    on until each has waited DOOMED presses or a life is lost, starting no new waiting codes:
    otherwise the screens of a fall that ends just after the last press would be saved. Those
    extra presses are played and counted like any other. Returns the steps made (memory before,
    button, memory after) and the screen codes met.
    """
    before, buttons, after, met = [], [], [], set()
    waiting = {}  # new screen codes not saved yet -> [press met, saved game, memory, visits]
    lives, button = game.lives(), rng.integers(BUTTONS)
    press = 0
    while press < presses or waiting:
        if rng.random() > HOLD:
            button = rng.integers(BUTTONS)
        before.append(game.memory())
        game.press(button)
        after.append(game.memory())
        buttons.append(button)
        if game.over() or game.lives() < lives:
            waiting.clear()  # all met in the last DOOMED presses: the man was already falling to his death
            if stop_at_lost_life:
                break
            if game.over():
                game.start_again()
            lives = game.lives()
            press += 1
            continue
        code = game.screen_code()
        met.add(code)
        if code in places.index:
            places.visits[places.index[code]] += 1
        elif code in waiting:
            waiting[code][3] += 1
        elif press < presses:  # once the presses have run out, only the waiting codes matter
            waiting[code] = [press, game.save(), game.memory(), 1]
        for ready in [waiting_code for waiting_code, (met_at, *_) in waiting.items() if met_at <= press - DOOMED]:
            places.add(ready, *waiting.pop(ready)[1:])
        press += 1
    return np.array(before), np.array(buttons), np.array(after), met


def rounds(learner, game, places, rng, frames):
    """Play rounds until the frame budget is spent. Yields each round's runs of presses (Burst)."""
    while game.frames_played < frames:
        known = set(places.index)  # the places saved before this round
        if not learner.jumps_back:  # random play
            bursts = [Burst(None, *press_randomly(game, rng, places, PLACES_PER_ROUND * PRESSES, stop_at_lost_life=False))]
        else:
            bursts = []
            for place in choose(learner, places, rng):
                places.chosen[place] += 1
                game.restore(places.saved[place])
                bursts.append(Burst(place, *press_randomly(game, rng, places, PRESSES, stop_at_lost_life=True)))
        # Each run of presses is credited with every place saved this round that it met, whichever met it first.
        saved_now = set(places.index) - known
        for burst in bursts:
            burst.new_places = len(burst.met & saved_now)
        yield bursts


def start(seed):
    """A new game and its first saved place."""
    game = Game(seed)
    places = Places()
    places.add(game.screen_code(), game.save(), game.memory())
    return game, places


def run(rule, seed, test_set, frames=FRAMES):
    """One learner, one run. Returns lines measured every MEASURE_EVERY frames."""
    rng = np.random.default_rng(seed)
    game, places = start(seed)
    model, replay = WorldModel(rng), Replay(rng)
    learner = (LEARNERS | CONTROLS)[rule](rng, model, replay)
    rooms = {int(game.memory()[ROOM_BYTE])}
    owed, trained = 0.0, 0  # training steps earned by the steps played but not yet taken; steps taken
    lines = {"frames": [], "knowledge": [], "room_errors": [], "room_squared_misses": [], "rooms_reached": [],
             "rooms": [], "places": [], "training_steps": [], "rounds": [], "presses_per_burst": []}
    rounds_played, bursts_played = 0, 0
    for bursts in rounds(learner, game, places, rng, frames):
        rounds_played += 1
        bursts_played += len(bursts)
        learner.see(places, bursts)  # what was found from each place, before the world model trains on it
        before, buttons, after = (np.concatenate([getattr(burst, part) for burst in bursts])
                                  for part in ("before", "buttons", "after"))
        replay.add(before, buttons, after)
        owed += len(buttons) / STEPS_PER_TRAINING_STEP
        steps = int(owed)
        model.learn(replay, steps)
        learner.learn(places, bursts, steps)
        trained += steps
        owed -= steps
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
            lines["rounds"].append(rounds_played)
            lines["presses_per_burst"].append(game.frames_played / FRAME_SKIP / bursts_played)
    return {name: np.array(values) for name, values in lines.items()}
