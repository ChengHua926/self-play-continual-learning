"""Test 1: how well a world model predicts what happens in the situations the human demos pass through.

The 200 Square demos' steps (gpu_test_data.py): the state a human was in, the human's action, and the
next state as the GPU world gives it. None of the world models train on them. The demos are cut into
pieces of up to 50 steps, as long as a training experiment; within each piece the model sees the
states so far and predicts the next, as in training. The score is the loss in nats per number (lower
is better), over all steps, and separately over the steps where the human holds the nut (about half)
and the rest.
"""

from pathlib import Path

import numpy as np
import torch

from constants import STEPS
from world_model import step_losses

TEST = Path(__file__).resolve().parent.parent / "data" / "robomimic" / "square_ph"


class TestOne:
    def __init__(self, device):
        steps = np.load(TEST / "test_steps_gpu.npz")
        holding = np.load(TEST / "demo_states.npz")["holding"]
        pieces = []                     # (first step, length) of each piece of up to STEPS steps
        first = 0
        for length in steps["lengths"]:
            pieces += [(first + k, min(STEPS, length - k)) for k in range(0, length, STEPS)]
            first += length
        rows = np.array([[start + min(t, length - 1) for t in range(STEPS)] for start, length in pieces])
        self.used = torch.tensor(np.arange(STEPS)[None] < np.array([length for _, length in pieces])[:, None], device=device)
        take = lambda name: torch.tensor(steps[name][rows], dtype=torch.float32, device=device)
        self.seen, self.actions, self.after = take("seen"), take("actions"), take("after")
        self.holding = torch.tensor(holding[rows], device=device) & self.used

    @torch.no_grad()
    def score(self, model):
        losses = torch.cat([step_losses(model, *(x[i:i + 256] for x in (self.seen, self.actions, self.after)))
                            for i in range(0, len(self.seen), 256)])
        mean = lambda mask: (losses[mask].mean().item())
        return {"all": mean(self.used), "holding": mean(self.holding), "not holding": mean(self.used & ~self.holding)}
