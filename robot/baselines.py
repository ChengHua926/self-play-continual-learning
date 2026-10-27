"""The ways of making experiments that learn nothing, for comparison with the generator.

Both people's ways share a start: the scene's own random start, widened. The arm starts near its
home pose, each joint off by a normal spread of 0.3 rad, where robosuite uses 0.02 (clipped to the
joint's limits). The gripper starts open or closed, by a coin flip. The nut rests flat anywhere on
the middle 60 x 60 cm of the table, turned any way about the vertical; robosuite's own area is
0.5 x 11.5 cm. The hand then moves through waypoints, a new one every 10 steps (0.5 s), going
straight toward each as fast as the controller allows.

peoples_way         random waypoints: each anywhere over that part of the table, from 1 to 28 cm
                    above it, with the gripper randomly open or closed. The hand keeps its
                    orientation.
peoples_way_at_nut  a scripted, randomized grasp, as hand-written generators of practice data do
                    (Instant Policy's, for example, puts its waypoints on the objects). The five
                    waypoints are:
                      1. 5 to 12 cm above the nut's handle, gripper open;
                      2. down at the handle, gripper open;
                      3. the same place, gripper closed;
                      4. a random point over the table, as in peoples_way, still closed;
                      5. the same place, gripper open.
                    Each aim at the handle is off by up to 1.5 cm sideways, and the hand turns so
                    its fingers close across the handle's width.
plain_random        random bytes for the whole experiment: a random start over the machine's full
                    ranges, then random actions, each step drawn on its own.
"""

import numpy as np

from machine import NEEDED, run, run_bytes
from world import TABLE_TOP

WAYPOINT_EVERY = 10
AREA = 0.3                  # half the side of the middle part of the table used, in meters
LARGEST_MOVE = 0.05         # meters a step, the controller's largest
LARGEST_TURN = 0.5          # radians a step


def widened_start(world, rng):
    default = world.default_start
    arm = np.clip(default["arm"] + rng.normal(0, 0.3, 7), world.arm_range[:, 0], world.arm_range[:, 1])
    fingers = world.finger_range[:, 1] if rng.random() < 0.5 else world.finger_range[:, 0]
    yaw = rng.uniform(0, 2 * np.pi)
    return {"arm": arm, "fingers": fingers,
            "nut_position": np.array([*rng.uniform(-AREA, AREA, 2), default["nut_position"][2]]),
            "nut_quaternion": np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])}


def random_point(rng):
    return np.array([*rng.uniform(-AREA, AREA, 2), rng.uniform(TABLE_TOP + 0.01, TABLE_TOP + 0.28)])


def toward(world, target, turn=0.0, gripper=0.0):
    """The action that moves the hand straight toward target, turns it by turn about the vertical,
    and sets the gripper."""
    move = np.clip((target - world.hand()) / LARGEST_MOVE, -1, 1)
    return np.concatenate([move, [0.0, 0.0, np.clip(turn / LARGEST_TURN, -1, 1)], [gripper]])


def turn_across_handle(world):
    """How far to turn the hand about the vertical so its fingers close along the handle's width:
    the angle from the hand's y axis to the nut's y axis, seen from above, folded into +-90 degrees
    (the two fingers are alike, so either way round is fine)."""
    hand, nut = world.hand_axes()[:2, 1], world.nut_axes()[:2, 1]
    angle = np.arctan2(hand[0] * nut[1] - hand[1] * nut[0], hand @ nut)
    return (angle + np.pi / 2) % np.pi - np.pi / 2


def peoples_way(world, rng, watch=None):
    plan = {}

    def act(step):
        if step % WAYPOINT_EVERY == 0:
            plan["target"], plan["gripper"] = random_point(rng), rng.choice([-1.0, 1.0])
        return toward(world, plan["target"], gripper=plan["gripper"])

    return run(world, widened_start(world, rng), act, watch)


def peoples_way_at_nut(world, rng, watch=None):
    plan = {}

    def act(step):
        waypoint = step // WAYPOINT_EVERY
        if step % WAYPOINT_EVERY == 0:
            if waypoint == 0:
                plan["aim"] = np.array([*rng.uniform(-0.015, 0.015, 2), 0.0])
                plan["target"] = world.handle() + plan["aim"] + [0, 0, rng.uniform(0.05, 0.12)]
            elif waypoint in (1, 2):
                plan["target"] = world.handle() + plan["aim"]
            elif waypoint == 3:
                plan["target"] = random_point(rng)
        gripper = 1.0 if waypoint in (2, 3) else -1.0
        turn = turn_across_handle(world) if waypoint <= 2 else 0.0
        return toward(world, plan["target"], turn, gripper)

    return run(world, widened_start(world, rng), act, watch)


def plain_random(world, rng, watch=None):
    return run_bytes(world, rng.integers(0, 256, NEEDED, dtype=np.uint8).tobytes(), watch)


WAYS = {"peoples_way": peoples_way, "peoples_way_at_nut": peoples_way_at_nut, "plain_random": plain_random}
