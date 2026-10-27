"""Test 1's data: the situations the human demos pass through, with what happens next simulated in
our own world.

Each of the 200 Square demos is a path of recorded states, with the humans' actions. Replaying a
whole demo drifts: at a few moments of contact a step jumps by millimeters, and everything after
differs. So this goes one step at a time: from each recorded state, with that step's action, our
world gives the next state. The world models never train on these; test 1 scores how well each one
predicts them.

Saves data/robomimic/square_ph/test_steps.npz with, for all demos one after another:
    seen      [steps, 33]  the states the humans passed through, as our world reads them
    actions   [steps, 7]   the humans' actions
    after     [steps, 33]  the next state our world gives
    lengths   [demos]      steps in each demo, to split them again
Run it with:   ../.venv-robosuite/bin/python test_data.py      (a few minutes)
"""

from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

from world import World

DEMOS = Path(__file__).resolve().parent.parent / "data" / "robomimic" / "square_ph" / "low_dim_v141.hdf5"
SAVED = DEMOS.with_name("test_steps.npz")
WORKERS = 10

world = None


def one_demo(name):
    global world
    if world is None:
        world = World()
    demo = h5py.File(DEMOS, "r")["data"][name]
    states, actions = demo["states"][()], demo["actions"][()]
    world.start_recorded(states[0])
    seen, after = [], []
    for t in range(len(actions)):
        world.set_recorded(states[t])       # the finger targets carry on from the step before
        seen.append(world.state())
        world.step(actions[t])
        after.append(world.state())
    return np.array(seen), actions, np.array(after)


def main():
    names = sorted(h5py.File(DEMOS, "r")["data"].keys(), key=lambda k: int(k.split("_")[1]))
    with Pool(WORKERS) as pool:
        demos = pool.map(one_demo, names)
    np.savez(SAVED, seen=np.concatenate([d[0] for d in demos]), actions=np.concatenate([d[1] for d in demos]),
             after=np.concatenate([d[2] for d in demos]), lengths=np.array([len(d[1]) for d in demos]))
    print(f"saved {sum(len(d[1]) for d in demos)} steps from {len(demos)} demos to {SAVED}")


if __name__ == "__main__":
    main()
