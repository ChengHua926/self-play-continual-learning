"""Checks of the setup with answers we know, before any comparison of learners.

Five rules, 10 runs each, 8 million frames per run, in the same loop as every learner
(explore.py, learners.py). Run k of every rule uses seed k.
    uniform choice   jump back to saved places at random: "not choosing"
    random play      never jump back
    most visited     jump back to the places visited most
    room oracle      spread the jumps over rooms, using room numbers no learner may use
    cell counts      Go-Explore's rule for this game: places visited least and chosen least
report_checks.py reads them with rules fixed before the run. The world model is checked on its
own, in check_world_model.py.

Run it with:   ../.venv/bin/python run_checks.py      (about an hour)
Each run is saved when it ends (runs.py); after a crash, running it again continues. At the end
it writes results/checks.npz and prints the report.
"""

import numpy as np

import report_checks
import test_set
from runs import gather, run_all

RULES = ("uniform choice", "random play", "most visited", "room oracle", "cell counts")
SEEDS = tuple(range(10))
RUNS_FOLDER = "results/checks_runs"


if __name__ == "__main__":
    run_all(RUNS_FOLDER, RULES, SEEDS)
    tests = test_set.load()
    np.savez(report_checks.RESULTS_FILE, **gather(RUNS_FOLDER, RULES, SEEDS), rules=RULES,
             rooms_tested=tests.rooms, no_change=tests.no_change)
    report_checks.main()
