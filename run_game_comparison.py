"""The game comparison: every way of choosing where to explore, run end to end in Montezuma's Revenge.

Every learner runs the same loop (explore.py): the same jumping back, the same random presses,
the same world model trained one step per 50 steps played, 8 million frames. Learners differ
only in how they choose the places to jump back to (learners.py). Each is run 10 times; run k
of every learner uses seed k. Its world model's knowledge of the game is measured on the test
set (test_set.py): the share of the game's movement it predicts, room by room, every room
counting the same.

The reading rules, fixed before the run:
    - a learner works if it ends with more knowledge than uniform choice in at least 7 of the
      10 runs ("ends" is the average of the last 5 measurements, the last 500,000 frames)
    - a learner is flagged if its average end knowledge is more than 5% below uniform choice's
With no real difference, 7 or more of 10 runs happen 17% of the time, so among 15 learners
about 2 or 3 would "work" by luck alone; the report says so.

Run it with:   .venv/bin/python run_game_comparison.py      (about 4 hours)
Each run is saved when it ends (runs.py); after a crash, running it again continues. At the end
it writes results/game_comparison.npz and prints the report (report_game_comparison.py).
"""


import numpy as np

import report_game_comparison
import test_set
from learners import LEARNERS
from runs import gather, run_all

SEEDS = tuple(range(10))
RUNS_FOLDER = "results/game_comparison_runs"  # one file per run, so that a crash loses little


if __name__ == "__main__":
    run_all(RUNS_FOLDER, list(LEARNERS), SEEDS)
    tests = test_set.load()
    np.savez(report_game_comparison.RESULTS_FILE, **gather(RUNS_FOLDER, list(LEARNERS), SEEDS),
             learners=list(LEARNERS), rooms_tested=tests.rooms, no_change=tests.no_change,
             no_change_misses=tests.no_change_misses)
    report_game_comparison.main()
