"""Step A: part of the map again, with a fair push budget, and a check of the critics.

In the first map, every learner made 128 pushes per round and did each one twice. Only
repeat agreement uses the second copy; for the other learners it was a wasted action.
Step A gives every learner the same 256 actions per round (the "fair" budget, see
run_step2.py) and asks two questions:

    1. Does the ranking of the labels hold once the budget is fair?
    2. Do the critics forecast their labels? (the critic check, see run_step2.check_critic)

It runs the part of the map where the answer matters most: random worlds of 6 and 24
objects, plain and narrow, 50 training steps per round, no guard, and 10 runs. Next to the
learners of the map runs one control, "random grades" (see run_step2.py). Every run
is done with both budgets, on the same worlds, so that the two can be compared run by
run. (With the doubled budget, runs 0 to 4 repeat the first map exactly.)

Run it with:   python3 run_step_a.py
It takes a few minutes, writes results/step_a.npz, and prints the report.
"""

from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import report_step_a
import run_map
from run_map import run_in_its_world
from run_step1 import ROUNDS

LEARNERS = (*run_map.LEARNERS, "random grades")
BUDGETS = ("doubled", "fair")
WORLDS = ("6", "24", "narrow 6", "narrow 24")
TRAIN_STEPS = 50
GUARD = "none"
SEEDS = tuple(range(10))


def run_one(budget, world, learner, seed):
    return run_in_its_world(learner, TRAIN_STEPS, world, GUARD, seed, budget=budget)


if __name__ == "__main__":
    cells = list(product(BUDGETS, WORLDS, LEARNERS, SEEDS))
    with Pool() as pool:  # the runs do not depend on each other, so do them side by side
        results = pool.starmap(run_one, cells)

    # Every line is saved as [budget, world, learner, run, round].
    shape = (len(BUDGETS), len(WORLDS), len(LEARNERS), len(SEEDS), ROUNDS)
    Path("results").mkdir(exist_ok=True)
    np.savez(
        report_step_a.RESULTS_FILE,
        **{line: np.array([result[line] for result in results]).reshape(shape) for line in results[0]},
        budgets=BUDGETS, worlds=WORLDS, learners=LEARNERS,
    )
    report_step_a.main()
