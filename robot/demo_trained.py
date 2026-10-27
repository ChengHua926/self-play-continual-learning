"""Test 1's known-answer check: a world model trained on the human demos' own steps.

It trains the runs' world model (same size, scales and learning rate) on 150 of the 200 demos, cut
into pieces of up to 50 steps as test 1 cuts them, and scores it every 250 training steps on the other
50 demos. A world model whose experience covered the humans' situations should do about this well,
and it should beat an untrained one by a wide margin; otherwise test 1 cannot see what a world model
has learned. The runs' final learners can then be scored on the same 50 demos.

Run it with:   ../.venv/bin/python demo_trained.py      (minutes, on the CPU)
"""

from pathlib import Path

import numpy as np
import torch

from scores import TestOne
from world_model import WorldModel, step_losses

SCALES = Path(__file__).resolve().parent / "results" / "scales.npz"
STEPS, BATCH, LEARNING_RATE, WARMUP = 2000, 64, 3e-3, 50
HELD_OUT = range(150, 200)


def main():
    torch.manual_seed(0)
    training, held_out = TestOne("cpu", demos=range(150)), TestOne("cpu", demos=HELD_OUT)
    model = WorldModel(dict(np.load(SCALES)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0.0)
    print(f"{len(training.seen)} training pieces, {len(held_out.seen)} held-out pieces (demos 150 to 199)")
    print("training step: held-out test 1, nats per number: all / holding / not holding")
    for step in range(STEPS + 1):
        if step % 250 == 0:
            scores = held_out.score(model)
            print(f"   {step:5d}: {scores['all']:.3f} / {scores['holding']:.3f} / {scores['not holding']:.3f}", flush=True)
        if step == STEPS:
            break
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * min(1.0, (step + 1) / WARMUP)
        batch = torch.randint(0, len(training.seen), (BATCH,))
        losses = step_losses(model, training.seen[batch], training.actions[batch], training.after[batch])
        used = training.used[batch]
        loss = (losses * used).sum() / used.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.save(model.state_dict(), Path(__file__).resolve().parent / "results" / "demo_trained.pt")


if __name__ == "__main__":
    main()
