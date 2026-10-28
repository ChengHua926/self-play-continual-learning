"""Each run's last world model on the 50 demos that the demo-trained model never saw (demos 150 to 199).

Test 1 during the runs scores all 200 demos. This scores each run's last checkpoint on the held-out
50 only, so that it can sit next to the model trained on the other 150 (demo_trained.py). It also
says how sure each model is: the median size of its errors measured in its own predicted spreads.
A model whose spreads are honest has a median near 0.67; far above 1 means it is too sure.

Run it with:   ../.venv/bin/python held_out.py <runs folder> <round, e.g. 1024>
"""

import sys
from pathlib import Path

import numpy as np
import torch

from demo_trained import HELD_OUT, SCALES
from scores import TestOne
from world_model import WorldModel


@torch.no_grad()
def errors_in_spreads(model, test):
    """The median of |real change - predicted change| / predicted spread, over the steps used."""
    mean, log_spread = model(test.seen, test.actions)
    change = (test.after - test.seen) / model.change_spread
    return ((change - mean) / log_spread.exp()).abs()[test.used].median().item()


def main(folder, round_):
    test = TestOne("cpu", demos=HELD_OUT)
    paths = sorted(Path(folder).glob(f"learner_*_{int(round_):04d}.pt"))
    paths.append(Path(__file__).resolve().parent / "results" / "demo_trained.pt")
    print("Test 1 on the held-out demos 150 to 199, nats per number: all / holding / not holding;"
          " then the median error in predicted spreads")
    for path in paths:
        model = WorldModel(dict(np.load(SCALES)))
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        scores = test.score(model)
        print(f"   {path.stem:34s} {scores['all']:7.3f} / {scores['holding']:7.3f} / {scores['not holding']:7.3f}"
              f"   {errors_in_spreads(model, test):7.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
