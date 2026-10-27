"""Test 1's data again, with each next state from the GPU world (the world the models train in).

As test_data.py does with robosuite: from each state the human demos passed through, with that
step's action, the world gives the next state; the finger targets start at 0 in each demo and carry
on from step to step. Here all 200 demos go at once, one step at a time.
Needs data/robomimic/square_ph/demo_states.npz (the demos' recorded states and actions, taken out of
robomimic's file). Saves data/robomimic/square_ph/test_steps_gpu.npz, laid out as test_steps.npz.

Run it with:   ../.venv/bin/python gpu_test_data.py      (on a machine with an NVIDIA GPU)
"""

from pathlib import Path

import numpy as np
import torch

from gpu_world import GpuWorld

FOLDER = Path(__file__).resolve().parent.parent / "data" / "robomimic" / "square_ph"


def main():
    demos = np.load(FOLDER / "demo_states.npz")
    lengths = demos["lengths"]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    world = GpuWorld(len(lengths))
    seen, after = np.zeros((lengths.sum(), 33)), np.zeros((lengths.sum(), 33))
    targets = torch.zeros(len(lengths), 2, dtype=torch.float64, device=world.device)
    for t in range(lengths.max()):
        rows = starts + np.minimum(t, lengths - 1)          # finished demos repeat their last step, unused
        world.set_recorded(torch.tensor(demos["states"][rows], device=world.device), targets)
        state_before = world.state().cpu().numpy()
        world.step(torch.tensor(demos["actions"][rows], device=world.device))
        targets = world.finger_targets.clone()
        state_after = world.state().cpu().numpy()
        going = t < lengths
        seen[rows[going]], after[rows[going]] = state_before[going], state_after[going]
    np.savez(FOLDER / "test_steps_gpu.npz", seen=seen, actions=demos["actions"], after=after, lengths=lengths)
    print(f"saved {lengths.sum()} steps from {len(lengths)} demos")


if __name__ == "__main__":
    main()
