"""How many experiments a second the GPU world runs, with thousands of copies at once.

Each copy runs one 50-step experiment from a random start with random actions, as plain random
does (the most contacts, so the slowest case). Also reports the most constraint rows and contacts
any copy needed, against the room each copy has.

Run it with:   ../.venv/bin/python gpu_speed.py [copies ...]      (default 1024 4096 16384)
"""

import sys
import time

import torch
import warp as wp

from gpu_world import CONSTRAINTS_PER_COPY, CONTACTS_PER_COPY, GpuWorld
from constants import STEPS


def random_starts(world, generator):
    copies, device = world.copies, world.device
    uniform = lambda shape, low, high: low + (high - low) * torch.rand(shape, generator=generator, device=device, dtype=torch.float64)
    low, high = (torch.tensor(world.layout.model.jnt_range[:7].T[k], device=device) for k in (0, 1))
    fingers = torch.stack([uniform((copies,), 0.0, 0.04), uniform((copies,), -0.04, 0.0)], dim=1)
    nut = torch.stack([uniform((copies,), -0.4, 0.4), uniform((copies,), -0.4, 0.4), uniform((copies,), 0.82, 1.32)], dim=1)
    quaternion = torch.nn.functional.normalize(torch.randn(copies, 4, generator=generator, device=device, dtype=torch.float64), dim=1)
    return uniform((copies, 7), low, high), fingers, nut, quaternion


def measure(copies):
    world = GpuWorld(copies)
    generator = torch.Generator(device=world.device).manual_seed(0)
    for warmup in (True, False):
        world.set_starts(*random_starts(world, generator))
        torch.cuda.synchronize()
        started, most_rows, most_contacts = time.time(), 0, 0
        for _ in range(STEPS):
            world.step(torch.rand(copies, 7, generator=generator, device=world.device) * 2 - 1)
            most_rows = max(most_rows, int(wp.to_torch(world.d.nefc).max()))
            most_contacts = max(most_contacts, int(world.d.nacon.numpy()[0]))
        torch.cuda.synchronize()
        seconds = time.time() - started
    print(f"{copies:6d} copies: {seconds:6.1f} s for one {STEPS}-step experiment each = {copies / seconds:7.0f} experiments a second; "
          f"most constraint rows in a copy {most_rows} (room {CONSTRAINTS_PER_COPY}); most contacts at once {most_contacts / copies:.0f} a copy (room {CONTACTS_PER_COPY})", flush=True)


if __name__ == "__main__":
    for copies in [int(c) for c in sys.argv[1:]] or [1024, 4096, 16384]:
        measure(copies)
