"""A check of the world model with an answer we know: given data from every room, it knows every room.

If a world model cannot learn a room even from steps taken in it, low knowledge later would mean
a weak model or a broken score, not poor exploring. So before any comparison of learners:

The twin of the test set is built the same way as the test set (test_set.py), from four other
explorer runs (seeds 201 to 204), with up to 300 places per room instead of 100: the same kind
of places, none of them the test set's own. Two world models are trained, each for as many
training steps as a learner takes in a run of 8 million frames (TRAINING_STEPS):
    twin           on all of the twin's steps
    first room     on the twin's steps in the first room (room 1) only

The answer we know: the twin model knows every test room; the first-room model knows the first
room and little else. The rule, fixed before the run: the check passes if, in every test room
other than the first, the twin model's knowledge is at least 10 points above the first-room
model's. Test rooms the twin's explorers never reached are reported, and left out of the rule;
if fewer than 5 rooms are left to judge, the check cannot be judged.

Run it with:   .venv/bin/python check_world_model.py      (about 30 minutes)
It writes results/montezuma_twin_set.npz and results/world_model_check.txt.
"""

from multiprocessing import Pool

import numpy as np

import test_set
from game import ROOM_BYTE
from world_model import Replay, WorldModel

TWIN_FILE = "results/montezuma_twin_set.npz"
REPORT_FILE = "results/world_model_check.txt"
TWIN_EXPLORERS = (("cell counts", 201), ("cell counts", 202), ("uniform choice", 203), ("uniform choice", 204))
TWIN_PLACES_PER_ROOM = 300
TRAINING_STEPS = 40_000  # what a learner takes in 8 million frames, at one step per 50 steps played
FIRST_ROOM = 1
MARGIN = 0.10
ROOMS_NEEDED = 5  # fewer rooms to judge than this, and the check cannot be judged


def build_twin():
    with Pool(len(TWIN_EXPLORERS)) as pool:
        parts = pool.starmap(test_set.explore_and_press,
                             [(rule, seed, TWIN_PLACES_PER_ROOM) for rule, seed in TWIN_EXPLORERS])
    before, button, after = (np.concatenate(part) for part in zip(*parts))
    np.savez_compressed(TWIN_FILE, before=before, button=button, after=after)
    return before, button, after


def trained_on(before, button, after, seed):
    rng = np.random.default_rng(seed)
    replay = Replay(rng, size=len(button))  # big enough to keep every step
    replay.add(before, button, after)
    model = WorldModel(rng)
    model.learn(replay, TRAINING_STEPS)
    return model


def main():
    before, button, after = build_twin()
    tests = test_set.load()
    first = before[:, ROOM_BYTE] == FIRST_ROOM
    twin_rooms = set(np.unique(before[:, ROOM_BYTE]).tolist())
    twin = tests.knowledge(trained_on(before, button, after, seed=0))
    first_room = tests.knowledge(trained_on(before[first], button[first], after[first], seed=0))
    twin_per_room = np.maximum(1 - twin[1] / tests.no_change, 0)
    first_per_room = np.maximum(1 - first_room[1] / tests.no_change, 0)

    lines = [f"The twin of the test set: {len(button):,} steps in rooms {sorted(twin_rooms)}; "
             f"{int(first.sum()):,} of them in the first room.",
             f"Each model took {TRAINING_STEPS:,} training steps. Knowledge on the test set, room by room:", "",
             f"{'test room':>10s}{'twin':>8s}{'first room':>12s}   in the twin?"]
    for room, a, b in zip(tests.rooms, twin_per_room, first_per_room):
        lines.append(f"{room:10d}{a:8.0%}{b:12.0%}   {'yes' if room in twin_rooms else 'no'}")
    lines.append(f"{'all':>10s}{twin[0]:8.0%}{first_room[0]:12.0%}")
    judged = [i for i, room in enumerate(tests.rooms) if room != FIRST_ROOM and room in twin_rooms]
    passed = all(twin_per_room[i] >= first_per_room[i] + MARGIN for i in judged)
    verdict = "CANNOT BE JUDGED (too few rooms)" if len(judged) < ROOMS_NEEDED else "PASSES" if passed else "FAILS"
    lines += ["", f"The rule: in every test room other than the first that the twin reaches ({len(judged)} rooms), the twin "
                  f"model knows at least {MARGIN:.0%} more than the first-room model.",
              f"Verdict: the world-model check {verdict}."]
    report = "\n".join(lines)
    print(report)
    with open(REPORT_FILE, "w") as file:
        file.write(report + "\n")


if __name__ == "__main__":
    main()
