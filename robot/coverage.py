"""The coverage check, before any learning. For the ways that learn nothing (baselines.py): how
often do they reach what the human demos do, how often do their starts put things inside each other,
and how fast does the world run?

For each experiment it notes whether each of these happened at any step:
    touched     a finger touches the nut
    held        both fingers hold the nut (robosuite's own grasp check)
    lifted      held, with the nut at least 10 cm above where it rests on the table
    over peg    held, with the nut's center within 3 cm of the square peg's, seen from above
    on peg      robosuite's own success check: the nut around the square peg, the gripper away
and at the start, whether two things are inside each other by more than 5 mm.
The 200 human demos are counted the same way, from their recorded states, for scale.

Run it with:   ../.venv-robosuite/bin/python coverage.py [experiments per way, default 2000]
"""

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

from baselines import WAYS
from world import STEPS, World

DEMOS = Path(__file__).resolve().parent.parent / "data" / "robomimic" / "square_ph" / "low_dim_v141.hdf5"
EVENTS = ("touched", "held", "lifted", "over peg", "on peg")
NUT_REST = 0.83             # the nut's height resting on the table (measured), in meters
OVERLAP = 0.005             # meters
WORKERS = 10

world = None


def events_now(world):
    held = world.holding()
    nut = world.nut_position()
    return {"touched": world.touching(), "held": held, "lifted": held and nut[2] >= NUT_REST + 0.10,
            "over peg": held and np.linalg.norm(nut[:2] - world.square_peg) < 0.03, "on peg": world.nut_on_peg()}


def one_experiment(task):
    global world
    if world is None:
        world = World()
    way, seed = task
    seen = dict.fromkeys(EVENTS, False)

    def watch(world):
        for name, happened in events_now(world).items():
            seen[name] |= happened

    started = time.time()
    _, _, overlap = WAYS[way](world, np.random.default_rng(seed), watch)
    return way, seen, overlap > OVERLAP, time.time() - started


def demo_events():
    """Which events happen in each human demo, read off its recorded states."""
    found = []
    demos = h5py.File(DEMOS, "r")["data"]
    local = World()
    for name in demos:
        seen = dict.fromkeys(EVENTS, False)
        for state in demos[name]["states"][()]:
            local.set_recorded(state)
            for event, happened in events_now(local).items():
                seen[event] |= happened
        found.append(seen)
    return found


def main(count):
    tasks = [(way, seed) for way in WAYS for seed in range(count)]
    results = {way: [] for way in WAYS}
    with Pool(WORKERS) as pool:
        for way, seen, overlapping, seconds in pool.imap_unordered(one_experiment, tasks, chunksize=8):
            results[way].append((seen, overlapping, seconds))
    print(f"Share of experiments in which each happened at least once ({count} experiments per way)")
    print(f"   {'way':20s}" + "".join(f"{e:>10s}" for e in EVENTS) + f"{'overlap':>10s}{'s each':>9s}")
    for way, rows in results.items():
        shares = [np.mean([seen[e] for seen, _, _ in rows]) for e in EVENTS]
        print(f"   {way:20s}" + "".join(f"{s:10.2%}" for s in shares)
              + f"{np.mean([o for _, o, _ in rows]):10.2%}{np.mean([s for _, _, s in rows]):9.2f}")
    demos = demo_events()
    print(f"   {'human demos':20s}" + "".join(f"{np.mean([d[e] for d in demos]):10.2%}" for e in EVENTS)
          + f"   ({len(demos)} demos)")
    print(f"(seconds each: one experiment of {STEPS} steps on one core, with {WORKERS} running at once)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
