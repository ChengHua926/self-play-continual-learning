"""Record what robosuite does, step by step, so that our own copy of the world can be checked against it.

For each recorded control step it keeps:
    before       the state as robosuite records it (time, all positions, all velocities)
    warmstart    MuJoCo's solver warm start before the step (it nudges how contacts resolve)
    targets      the finger targets before the step
    action       the 7 action numbers
    controls     the 9 motor commands robosuite sent at each of the 25 physics steps [25, 9]
    after        the state after the step
The steps come from 5 human demos (one step at a time from their recorded states, as test 1 does)
and from 20 plain random experiments. Neither turns the hand by exactly 0 about all three axes (the
demos never do; random bytes do it once in 16.7 million steps), so the one way our copy differs on
purpose never comes up: on a turn of exactly 0, our copy holds the hand's orientation as it is,
where robosuite goes back to its last goal.
    data/robot_scene/reference.npz
Run it with:   ../.venv-robosuite/bin/python reference.py
"""

from pathlib import Path

import h5py
import numpy as np

import machine
from world import World

DEMOS = Path(__file__).resolve().parent.parent / "data" / "robomimic" / "square_ph" / "low_dim_v141.hdf5"
SAVED = Path(__file__).resolve().parent.parent / "data" / "robot_scene" / "reference.npz"


def main():
    world = World()
    controls, rows = [], []
    original_step = world.env.sim.step

    def step_and_keep():
        controls.append(world.data.ctrl.copy())
        original_step()

    world.env.sim.step = step_and_keep

    def record(action):
        before = (world.recorded(), world.data.qacc_warmstart.copy(), world.robot.gripper.current_action.copy())
        controls.clear()
        world.step(action)
        rows.append((*before, np.asarray(action, dtype=float), np.array(controls), world.recorded()))

    demos = h5py.File(DEMOS, "r")["data"]
    for name in ("demo_0", "demo_1", "demo_2", "demo_3", "demo_4"):
        states, actions = demos[name]["states"][()], demos[name]["actions"][()]
        world.start_recorded(states[0])
        for t in range(len(actions)):
            world.set_recorded(states[t])
            record(actions[t])

    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        start, actions = machine.decode(rng.integers(0, 256, machine.NEEDED, dtype=np.uint8).tobytes(), world)
        world.set_start(**start)
        for action in actions:
            record(action)

    names = ("before", "warmstart", "targets", "action", "controls", "after")
    np.savez(SAVED, home=world.default_start["arm"],
             **{name: np.array([row[i] for row in rows]) for i, name in enumerate(names)})
    print(f"recorded {len(rows)} steps to {SAVED}")


if __name__ == "__main__":
    main()
