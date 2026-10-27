"""How each arm makes its experiments, for the GPU world: in batches, as torch tensors.

decode            bytes a program wrote -> a start and 50 steps of actions, by robot_machine.decode's
                  rules (test_train.py checks they agree)
plain_random      random bytes for the whole experiment
PeoplesWay        the two hand-written ways of baselines.py (the scene's random starts, widened, then
                  random waypoints, or a scripted randomized grasp of the nut), for one run's copies,
                  choosing each step's actions from where the hand and nut are now
"""

import math

import torch

from constants import STEPS, TABLE_TOP
from robot_machine import ACTION_SIZE, NEEDED, NUT_HIGH, NUT_LOW, START_BYTES
from sim import DEFAULT_FINGERS, DEFAULT_NUT, HOME

WAYPOINT_EVERY = 10
AREA = 0.3                  # half the side of the middle part of the table used, in meters
LARGEST_MOVE, LARGEST_TURN = 0.05, 0.5


def quaternion(angles):
    """[n, 4] MuJoCo (w, x, y, z) quaternions for turning by angles [n, 3] about x, then y, then z
    (scipy's "xyz", as robot_machine.quaternion uses)."""
    half = angles / 2
    c, s = half.cos(), half.sin()
    zero = torch.zeros_like(c[:, 0])
    turns = [torch.stack([c[:, k], *(s[:, k] if j == k else zero for j in range(3))], dim=1) for k in range(3)]
    result = turns[0]
    for turn in turns[1:]:          # each later turn is applied after (on the left of) the earlier ones
        w1, x1, y1, z1 = turn.unbind(1)
        w2, x2, y2, z2 = result.unbind(1)
        result = torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                              w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=1)
    return result


def decode(written, counts, layout, device):
    """written [n, 365] bytes (uint8 tensor), counts [n] how many of them the program wrote ->
    (start: dict of [n, ...] tensors, actions [n, STEPS, 7]). Unwritten bytes give the default start
    and 0 actions, as in robot_machine.decode."""
    double = dict(dtype=torch.float64, device=device)
    values = written.to(device).double()
    wrote = torch.arange(NEEDED, device=device)[None] < counts.to(device)[:, None]
    low = torch.cat([torch.tensor(layout.arm_range[:, 0]), torch.tensor(layout.finger_range[:, 0]),
                     torch.tensor(NUT_LOW), torch.full((3,), -math.pi, dtype=torch.float64)]).to(**double)
    high = torch.cat([torch.tensor(layout.arm_range[:, 1]), torch.tensor(layout.finger_range[:, 1]),
                      torch.tensor(NUT_HIGH), torch.full((3,), math.pi, dtype=torch.float64)]).to(**double)
    default = torch.tensor([*HOME, *DEFAULT_FINGERS, *DEFAULT_NUT, 0.0, 0.0, 0.0], **double)
    numbers = torch.where(wrote[:, :START_BYTES], low + (high - low) * values[:, :START_BYTES] / 255, default)
    start = {"arm": numbers[:, :7], "fingers": numbers[:, 7:9], "nut_position": numbers[:, 9:12],
             "nut_quaternion": quaternion(numbers[:, 12:15])}
    actions = torch.where(wrote[:, START_BYTES:], ((values[:, START_BYTES:] - 128) / 127).clamp(-1, 1), 0.0)
    return start, actions.view(-1, STEPS, ACTION_SIZE)


def plain_random(copies, generator, layout, device):
    written = torch.randint(0, 256, (copies, NEEDED), generator=generator, device=device, dtype=torch.uint8)
    return decode(written, torch.full((copies,), NEEDED, device=device), layout, device)


class PeoplesWay:
    """One run's hand-written way, for its copies: rows of the GPU world."""

    def __init__(self, world, rows, generator, aimed):
        self.world, self.rows, self.generator, self.aimed = world, rows, generator, aimed
        self.copies = rows.stop - rows.start

    def uniform(self, shape, low, high):
        return low + (high - low) * torch.rand(shape, generator=self.generator, device=self.world.device, dtype=torch.float64)

    def start(self):
        """The scene's own random start, widened (baselines.widened_start)."""
        layout, device, n = self.world.layout, self.world.device, self.copies
        arm_low, arm_high = (torch.tensor(layout.arm_range[:, k], device=device) for k in (0, 1))
        arm = (torch.tensor(HOME, device=device) + 0.3 * torch.randn(n, 7, generator=self.generator, device=device,
                                                                     dtype=torch.float64)).clamp(arm_low, arm_high)
        # The two fingers' joints run opposite ways: open is (high, low) of their ranges, closed (low, high).
        opened = torch.tensor(layout.finger_range[[0, 1], [1, 0]], device=device)
        closed = torch.tensor(layout.finger_range[[0, 1], [0, 1]], device=device)
        fingers = torch.where(torch.rand(n, 1, generator=self.generator, device=device) < 0.5, opened, closed)
        yaw = self.uniform((n,), 0.0, 2 * math.pi)
        nut = torch.stack([self.uniform((n,), -AREA, AREA), self.uniform((n,), -AREA, AREA),
                           torch.full((n,), DEFAULT_NUT[2], dtype=torch.float64, device=device)], dim=1)
        turn = torch.stack([(yaw / 2).cos(), torch.zeros_like(yaw), torch.zeros_like(yaw), (yaw / 2).sin()], dim=1)
        return {"arm": arm, "fingers": fingers, "nut_position": nut, "nut_quaternion": turn}

    def random_point(self):
        n = self.copies
        return torch.stack([self.uniform((n,), -AREA, AREA), self.uniform((n,), -AREA, AREA),
                            self.uniform((n,), TABLE_TOP + 0.01, TABLE_TOP + 0.28)], dim=1)

    def act(self, step):
        """[copies, 7]: this step's actions, toward the current waypoint."""
        world, rows, n = self.world, self.rows, self.copies
        waypoint = step // WAYPOINT_EVERY
        if step % WAYPOINT_EVERY == 0:
            if not self.aimed:
                self.target = self.random_point()
                self.gripper = torch.where(torch.rand(n, generator=self.generator, device=world.device) < 0.5, -1.0, 1.0).double()
            elif waypoint == 0:
                self.aim = torch.cat([self.uniform((n, 2), -0.015, 0.015), torch.zeros(n, 1, dtype=torch.float64, device=world.device)], dim=1)
                lift = torch.stack([torch.zeros(n, dtype=torch.float64, device=world.device)] * 2 + [self.uniform((n,), 0.05, 0.12)], dim=1)
                self.target = world.handle_position[rows].double() + self.aim + lift
            elif waypoint in (1, 2):
                self.target = world.handle_position[rows].double() + self.aim
            elif waypoint == 3:
                self.target = self.random_point()
        hand = world.hand_position[rows].double()
        move = ((self.target - hand) / LARGEST_MOVE).clamp(-1, 1)
        turn = torch.zeros(n, dtype=torch.float64, device=world.device)
        if self.aimed:
            gripper = torch.full((n,), 1.0 if waypoint in (2, 3) else -1.0, dtype=torch.float64, device=world.device)
            if waypoint <= 2:
                turn = (self.turn_across_handle() / LARGEST_TURN).clamp(-1, 1)
        else:
            gripper = self.gripper
        zero = torch.zeros(n, 2, dtype=torch.float64, device=world.device)
        return torch.cat([move, zero, turn[:, None], gripper[:, None]], dim=1)

    def turn_across_handle(self):
        """baselines.turn_across_handle: the angle from the hand's y axis to the nut's, seen from above,
        folded into +-90 degrees."""
        hand = self.world.hand_orientation[self.rows].double()[:, :2, 1]
        nut = self.world.nut_orientation[self.rows].double()[:, :2, 1]
        angle = torch.atan2(hand[:, 0] * nut[:, 1] - hand[:, 1] * nut[:, 0], (hand * nut).sum(dim=1))
        return torch.remainder(angle + math.pi / 2, math.pi) - math.pi / 2
