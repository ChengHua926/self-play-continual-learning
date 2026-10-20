"""Running many learner runs side by side, each saved as soon as it ends, so a crash loses little.

Each run (one learner, one seed) is saved to its own file in a folder, first under a temporary
name and then renamed, so a run killed while saving leaves no half-written file behind. Runs
already saved are skipped, so running again after a crash continues where it stopped. A run that
fails is reported and the others go on. Runs are handed out seed by seed, so slow and fast
learners are mixed and the cores stay busy.
"""

import os
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import test_set
from explore import run


def run_file(folder, learner, seed):
    return Path(folder) / f"{learner.replace(' ', '_')}__{seed}.npz"


def run_and_save(cell):
    """Run one learner with one seed and save its lines. Returns what went wrong, if anything."""
    folder, learner, seed = cell
    try:
        lines = run(learner, seed, test_set.load())
        unfinished = run_file(folder, learner, seed).with_suffix(".part.npz")
        np.savez(unfinished, **lines)
        os.replace(unfinished, run_file(folder, learner, seed))
        return learner, seed, None
    except Exception:  # one failing run must not take the others down with it
        return learner, seed, traceback.format_exc()


def run_all(folder, learners, seeds):
    """Run every learner with every seed that has no saved file yet, and save each run when it ends."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    cells = [(folder, learner, seed) for seed in seeds for learner in learners
             if not run_file(folder, learner, seed).exists()]
    failed = []
    with Pool() as pool:
        for done, (learner, seed, problem) in enumerate(pool.imap_unordered(run_and_save, cells), start=1):
            print(f"  {done}/{len(cells)} runs done: {learner}, run {seed}{' FAILED' if problem else ''}", flush=True)
            if problem:
                print(problem, flush=True)
                failed.append((learner, seed))
    if failed:
        raise RuntimeError(f"{len(failed)} runs failed: {failed}")


def gather(folder, learners, seeds):
    """Every saved line as [learner, run, measurement, ...], cut to the measurements every run has."""
    loaded = {(learner, seed): np.load(run_file(folder, learner, seed)) for learner in learners for seed in seeds}
    length = min(len(lines["frames"]) for lines in loaded.values())
    names = loaded[(learners[0], seeds[0])].files
    return {name: np.array([[loaded[(learner, seed)][name][:length] for seed in seeds] for learner in learners])
            for name in names}
