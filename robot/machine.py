"""The machine: bytes in, an experiment out, as the paper's machine turns a program into its output.

A generator's program writes bytes (the paper's machine in ../selfplay runs it). Here those bytes
describe one experiment in the fixed world (world.py):
    the start, 15 bytes: 7 arm joint angles, 2 finger positions, the nut's position (3) and its
        orientation (3 angles). Each byte spans its number's whole range, from low (byte 0) to high
        (byte 255): each joint's own limits; for the nut, the table's 80 x 80 cm and up to 50 cm
        above it; a full turn for each angle. Nothing checks that a start makes sense: a nut written
        into a peg is a nut written into a peg, and the physics deals with it.
    then 7 bytes per control step, for up to 50 steps. Byte b gives (b - 128) / 127, clipped to
        [-1, 1], so 128 gives 0: the hand holds still and the gripper stays as it is.
Bytes the program does not write take the default: the scene's usual start, and 0 for actions.
Everything starts at rest. The world then runs all 50 steps, and every state is recorded.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from world import STATE_SIZE, STEPS, TABLE_TOP

START_BYTES = 15
ACTION_SIZE = 7
NEEDED = START_BYTES + ACTION_SIZE * STEPS      # 365 bytes describe a whole experiment
NUT_LOW = np.array([-0.4, -0.4, TABLE_TOP])
NUT_HIGH = np.array([0.4, 0.4, TABLE_TOP + 0.5])


def spread(byte, low, high):
    return low + (high - low) * byte / 255.0


def quaternion(angles):
    """MuJoCo's (w, x, y, z) quaternion for turning by these angles about x, then y, then z."""
    x, y, z, w = Rotation.from_euler("xyz", angles).as_quat()
    return np.array([w, x, y, z])


def decode(written, world):
    """The bytes a program wrote -> (start, actions [STEPS, 7])."""
    written = np.frombuffer(bytes(written[:NEEDED]), dtype=np.uint8).astype(float)
    start_bytes, action_bytes = written[:START_BYTES], written[START_BYTES:]
    lows = np.concatenate([world.arm_range[:, 0], world.finger_range[:, 0], NUT_LOW, [-np.pi] * 3])
    highs = np.concatenate([world.arm_range[:, 1], world.finger_range[:, 1], NUT_HIGH, [np.pi] * 3])
    default = world.default_start
    numbers = np.concatenate([default["arm"], default["fingers"], default["nut_position"],
                              Rotation.from_quat(np.roll(default["nut_quaternion"], -1)).as_euler("xyz")])
    numbers[:len(start_bytes)] = spread(start_bytes, lows[:len(start_bytes)], highs[:len(start_bytes)])
    start = {"arm": numbers[:7], "fingers": numbers[7:9], "nut_position": numbers[9:12],
             "nut_quaternion": quaternion(numbers[12:15])}
    actions = np.zeros(STEPS * ACTION_SIZE)
    actions[:len(action_bytes)] = np.clip((action_bytes - 128) / 127, -1, 1)
    return start, actions.reshape(STEPS, ACTION_SIZE)


def run(world, start, act, watch=None):
    """Run one experiment. act(step) gives the 7 action numbers for that step, and may look at the
    world (a hand-written policy does). watch(world), if given, is called at the start and after
    every step (the coverage check uses it to note what happens). Returns the states
    [STEPS + 1, 33], the actions [STEPS, 7] and how deep, in meters, the two things most inside
    each other overlap at the start."""
    world.set_start(**start)
    overlap = world.deepest_overlap()
    states, actions = np.zeros((STEPS + 1, STATE_SIZE)), np.zeros((STEPS, ACTION_SIZE))
    states[0] = world.state()
    if watch:
        watch(world)
    for step in range(STEPS):
        actions[step] = act(step)
        world.step(actions[step])
        states[step + 1] = world.state()
        if watch:
            watch(world)
    return states, actions, overlap


def run_bytes(world, written, watch=None):
    """Run the experiment that these bytes describe."""
    start, actions = decode(written, world)
    return run(world, start, lambda step: actions[step], watch)
