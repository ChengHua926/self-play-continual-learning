"""A check of the world model with an answer we know: given data from every room, it knows every room.

If a world model cannot learn a room even from steps taken in it, low knowledge later would mean
a weak model or a broken score, not poor exploring. So before any comparison of learners:

The twin of the test set is built by the same procedure as the test set (test_set.py: the same
harness and explorers), from eight other explorer runs (seeds 201 to 208), with up to 300 places
per room from each explorer instead of 100: the same kind of places, and apart from the start of
the game (in room 1, which is not judged) none of the test set's own. A fifth of the twin's
places, each with its 18 presses, is held out of training. Two world models are trained:
    twin           on the twin's other places
    first room     on the twin's other places in the first room (room 1) only
Each trains for up to 40,000 steps, as many as a learner takes in 8 million frames, and is
measured on its held-out places after 250, 500, 1,000, 2,000, 4,000, 8,000, 16,000, 24,000,
32,000 and 40,000 steps (CHECKS); it is judged on the test set at its best held-out point.
Trained over and over on a fixed set of steps, a network learns them by heart and gets worse at
new ones (an audit found this before the run); a learner, whose steps keep coming, is not
trained that way.

The answer we know: the twin model knows every test room; the first-room model knows the first
room and little else. The rule, fixed before the run: the check passes if, in every judged room,
the twin model's knowledge is at least 10 points above the first-room model's. Judged rooms:
every test room other than the first with at least 20 places in the test set and 25 in the twin
(so that about 20 are left to train on). If fewer than 5 rooms can be judged, the check cannot
be judged.

Run it with:   .venv/bin/python check_world_model.py      (about 30 minutes)
It writes results/montezuma_twin_set.npz and results/world_model_check.txt.
"""

from multiprocessing import Pool

import numpy as np

import test_set
from game import BUTTONS, ROOM_BYTE
from test_set import TestSet
from world_model import Replay, WorldModel

TWIN_FILE = "results/montezuma_twin_set.npz"
REPORT_FILE = "results/world_model_check.txt"
TWIN_EXPLORERS = tuple(("cell counts", seed) for seed in (201, 202, 203, 204)) + \
    tuple(("uniform choice", seed) for seed in (205, 206, 207, 208))
TWIN_PLACES_PER_ROOM = 300
HELD_OUT = 0.2
CHECKS = (250, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 24_000, 32_000, 40_000)  # training steps; the last is
#   what a learner takes in 8 million frames, at one step per 50 steps played
FIRST_ROOM = 1
MARGIN = 0.10
PLACES_NEEDED = 20       # a room is judged only with at least this many places in the test set,
TWIN_PLACES_NEEDED = 25  # and this many in the twin (a fifth of them are held out)
ROOMS_NEEDED = 5         # fewer rooms to judge than this, and the check cannot be judged


def build_twin():
    with Pool(len(TWIN_EXPLORERS)) as pool:
        parts = pool.starmap(test_set.explore_and_press,
                             [(rule, seed, TWIN_PLACES_PER_ROOM) for rule, seed in TWIN_EXPLORERS])
    before, button, after = (np.concatenate(part) for part in zip(*parts))
    np.savez_compressed(TWIN_FILE, before=before, button=button, after=after)
    return before, button, after


def trained_with_held_out(before, button, after, rng):
    """Train on four fifths of the places; keep the model that did best on the other fifth.

    Returns that model, its knowledge of the held-out places at every check, and which check it is.
    """
    place = np.arange(len(button)) // BUTTONS  # each place's 18 presses are stored one after another
    held = np.isin(place, rng.permutation(place.max() + 1)[:round(HELD_OUT * (place.max() + 1))])
    replay = Replay(rng, size=int((~held).sum()))  # big enough to keep every training step
    replay.add(before[~held], button[~held], after[~held])
    held_out = TestSet(before[held], button[held], after[held])
    model, best, best_knowledge, best_check, curve, trained = WorldModel(rng), None, -1.0, None, [], 0
    for check, steps in enumerate(CHECKS):
        model.learn(replay, steps - trained)
        trained = steps
        knowledge = held_out.knowledge(model)[0]
        curve.append(knowledge)
        if knowledge > best_knowledge:
            best_knowledge, best_check = knowledge, check
            best = WorldModel.__new__(WorldModel)  # a frozen copy of the model as it is now
            best.network, best.scale = model.network.copy(), model.scale.copy()
    assert best is not None, "every check gave a knowledge that is not a number: the network diverged"
    return best, curve, best_check


def main():
    before, button, after = build_twin()
    tests = test_set.load()
    rng = np.random.default_rng(0)
    first = before[:, ROOM_BYTE] == FIRST_ROOM
    twin_model, twin_curve, twin_best = trained_with_held_out(before, button, after, rng)
    first_model, first_curve, first_best = trained_with_held_out(before[first], button[first], after[first], rng)
    twin_per_room = np.maximum(1 - tests.knowledge(twin_model)[1] / tests.no_change, 0)
    first_per_room = np.maximum(1 - tests.knowledge(first_model)[1] / tests.no_change, 0)

    twin_places = {int(room): int((before[::BUTTONS, ROOM_BYTE] == room).sum()) for room in tests.rooms}
    test_places = {int(room): int((tests.room == room).sum() // BUTTONS) for room in tests.rooms}
    judged = [i for i, room in enumerate(tests.rooms) if room != FIRST_ROOM
              and test_places[int(room)] >= PLACES_NEEDED and twin_places[int(room)] >= TWIN_PLACES_NEEDED]

    lines = [f"The twin of the test set: {len(button) // BUTTONS:,} places ({len(button):,} steps), "
             f"{HELD_OUT:.0%} of them held out.",
             "Knowledge of the held-out places after "
             + ", ".join(f"{steps:,}" for steps in CHECKS) + " training steps:",
             "  twin       " + " ".join(f"{k:.0%}" for k in twin_curve) + f"   best at {CHECKS[twin_best]:,}",
             "  first room " + " ".join(f"{k:.0%}" for k in first_curve) + f"   best at {CHECKS[first_best]:,}",
             "", "Knowledge on the test set, room by room, at each model's best held-out point:", "",
             f"{'test room':>10s}{'test places':>13s}{'twin places':>13s}{'twin':>8s}{'first room':>12s}   judged"]
    for i, room in enumerate(tests.rooms):
        lines.append(f"{room:10d}{test_places[int(room)]:13d}{twin_places[int(room)]:13d}{twin_per_room[i]:8.0%}"
                     f"{first_per_room[i]:12.0%}   {'yes' if i in judged else 'no'}")
    lines.append(f"{'all':>10s}{'':26s}{twin_per_room.mean():8.0%}{first_per_room.mean():12.0%}")
    passed = all(twin_per_room[i] >= first_per_room[i] + MARGIN for i in judged)
    verdict = "CANNOT BE JUDGED (too few rooms)" if len(judged) < ROOMS_NEEDED else "PASSES" if passed else "FAILS"
    lines += ["", f"The rule: in every judged room ({len(judged)}), the twin model knows at least {MARGIN:.0%} more "
                  "than the first-room model.",
              f"Verdict: the world-model check {verdict}."]
    report = "\n".join(lines)
    print(report)
    with open(REPORT_FILE, "w") as file:
        file.write(report + "\n")


if __name__ == "__main__":
    main()
