"""The positive control: does our measuring stick reward exploring?

Three learners, 10 runs each, 8 million frames per run (explore.py):
    uniform choice   jump back to a saved place picked at random: "not choosing"
    cell counts      jump back to the places visited least (Go-Explore's idea)
    random play      never jump back
Every learner trains the same world model on its own steps, one training step per 50 steps
played. Its knowledge of the game is measured on the test set (test_set.py): the share of the
game's movement it predicts, room by room, every room counting the same. Run k of every learner
uses seed k.

The rule, fixed before the run. The control passes if cell counts:
    - ends with more knowledge than uniform choice in at least 7 of the 10 runs, and
    - reaches more rooms than uniform choice, on average over the runs.
"Ends" means the average of the last 5 measurements, the last 500,000 frames.
The report adds checks that do not depend on the rule (report_positive_control.py), above all
whether knowledge goes with the rooms a run reached.

Run it with:   .venv/bin/python run_positive_control.py      (about an hour)
It writes results/positive_control.npz and prints the report (report_positive_control.py).
"""

from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import report_positive_control
import test_set
from explore import run
from game import BUTTONS

LEARNERS = ("uniform choice", "cell counts", "random play")
SEEDS = tuple(range(10))


def run_one(learner, seed):
    return run(learner, seed, test_set.load())


if __name__ == "__main__":
    cells = list(product(LEARNERS, SEEDS))
    with Pool() as pool:
        finished = dict(zip(cells, pool.starmap(run_one, cells, chunksize=1)))

    # Runs can differ by a measurement or two in length; keep the measurements all runs have.
    length = min(len(lines["frames"]) for lines in finished.values())

    def saved(line):  # [learner, run, measurement, ...]
        return np.array([[finished[(learner, seed)][line][:length] for seed in SEEDS] for learner in LEARNERS])

    Path("results").mkdir(exist_ok=True)
    tests = test_set.load()
    np.savez(report_positive_control.RESULTS_FILE, **{line: saved(line) for line in finished[cells[0]]},
             learners=LEARNERS, rooms_tested=tests.rooms, no_change=tests.no_change,
             no_change_misses=tests.no_change_misses,
             places_per_room=np.array([(tests.room == room).sum() // BUTTONS for room in tests.rooms]))
    report_positive_control.main()
