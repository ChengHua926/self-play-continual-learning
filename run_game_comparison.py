"""The game comparison: every way of choosing where to explore, run end to end in Montezuma's Revenge.

Every learner runs the same loop (explore.py): the same jumping back, the same random presses,
the same world model trained one step per 50 steps played, 8 million frames. Learners differ
only in how they choose the places to jump back to (learners.py). Each is run 10 times; run k
of every learner uses seed k. Its world model's knowledge of the game is measured on the test
set (test_set.py): the share of the game's movement it predicts, room by room, every room
counting the same.

The reading rules, fixed before the run, with the same reading as the setup checks (run_checks.py):
    - each run is read by its knowledge averaged over the whole run, every 100,000 frames from
      the start to 8 million, so that a run that learns sooner counts for more; its knowledge
      at the end (the last 500,000 frames) is shown beside it
    - a learner works if its whole-run knowledge is above uniform choice's in at least 7 of the
      10 runs (run k against run k)
    - a learner is flagged if its average whole-run knowledge is more than 5% below uniform choice's
With no real difference, 7 or more of 10 runs happen 17% of the time, so among 15 learners
about 2 or 3 would "work" by luck alone; the report says so. Also shown, because they measure
exploring directly and do not depend on the world model: rooms reached, and when the first room
was left. The setup checks found that knowledge in the far rooms is noisy: the world model
learns the visits it saw better than the rooms, and the far test rooms rest on few explorer runs.

The runs of uniform choice, random play and cell counts are the setup checks' own: the same code
and seeds give the same runs, so they are copied from results/checks_runs, not run again.

Run it with:   .venv/bin/python run_game_comparison.py      (about 4 hours)
Each run is saved when it ends (runs.py); after a crash, running it again continues. At the end
it writes results/game_comparison.npz and prints the report (report_game_comparison.py).
"""


import shutil
from itertools import product
from pathlib import Path

import numpy as np

import report_game_comparison
import test_set
from learners import LEARNERS
from runs import gather, run_all, run_file

SEEDS = tuple(range(10))
RUNS_FOLDER = "results/game_comparison_runs"  # one file per run, so that a crash loses little
CHECKS_FOLDER = "results/checks_runs"
FROM_THE_CHECKS = ("uniform choice", "random play", "cell counts")


if __name__ == "__main__":
    Path(RUNS_FOLDER).mkdir(parents=True, exist_ok=True)
    for learner, seed in product(FROM_THE_CHECKS, SEEDS):
        if run_file(CHECKS_FOLDER, learner, seed).exists() and not run_file(RUNS_FOLDER, learner, seed).exists():
            shutil.copy(run_file(CHECKS_FOLDER, learner, seed), run_file(RUNS_FOLDER, learner, seed))
    run_all(RUNS_FOLDER, list(LEARNERS), SEEDS)
    tests = test_set.load()
    np.savez(report_game_comparison.RESULTS_FILE, **gather(RUNS_FOLDER, list(LEARNERS), SEEDS),
             learners=list(LEARNERS), rooms_tested=tests.rooms, no_change=tests.no_change,
             no_change_misses=tests.no_change_misses)
    report_game_comparison.main()
