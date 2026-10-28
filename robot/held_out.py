"""Each run's world model on the 50 demos that the demo-trained model never saw (demos 150 to 199).

Test 1 during the runs scores all 200 demos. This scores each run's checkpoint at one round on the
held-out 50 only, so that it can sit next to the model trained on the other 150 (demo_trained.py),
and splits the loss by part of the state, to show what each model gets wrong: a part where a model
is sure and wrong costs many nats. It also gives each model's typical error with its spreads left
out: the median of |real change - predicted change| over all numbers and steps, in the units the
model predicts in (each number's change divided by its spread over random experiments).

Run it with:   ../.venv/bin/python held_out.py <runs folder> <round, e.g. 1024>
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

from demo_trained import HELD_OUT, SCALES
from scores import TestOne
from world_model import WorldModel

PARTS = {"arm": range(0, 7), "fingers": range(7, 9), "nut place": range(9, 12), "nut turn": range(12, 16),
         "finger goal": range(16, 18), "arm speed": range(18, 25), "finger speed": range(25, 27),
         "nut speed": range(27, 33)}


@torch.no_grad()
def loss_by_part(model, test):
    """{part of the state: its loss, nats per number}, over the steps used, and the typical error."""
    mean, log_spread = model(test.seen, test.actions)
    change = (test.after - test.seen) / model.change_spread
    nll = 0.5 * ((change - mean) / log_spread.exp()) ** 2 + log_spread + 0.5 * math.log(2 * math.pi)
    typical = (change - mean)[test.used].abs().median().item()
    return {part: nll[test.used][:, list(numbers)].mean().item() for part, numbers in PARTS.items()}, typical


def main(folder, round_):
    test = TestOne("cpu", demos=HELD_OUT)
    paths = sorted(Path(folder).glob(f"learner_*_{int(round_):04d}.pt"))
    paths.append(Path(__file__).resolve().parent / "results" / "demo_trained.pt")
    print("Test 1 on the held-out demos 150 to 199, nats per number: all / holding / not holding;")
    print("then the typical error; then the loss on each part of the state: " + ", ".join(PARTS))
    for path in paths:
        model = WorldModel(dict(np.load(SCALES)))
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        scores = test.score(model)
        by_part, typical = loss_by_part(model, test)
        print(f"   {path.stem:34s} {scores['all']:7.3f} / {scores['holding']:7.3f} / {scores['not holding']:7.3f}"
              f"   {typical:.3f}   " + " ".join(f"{loss:6.2f}" for loss in by_part.values()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
