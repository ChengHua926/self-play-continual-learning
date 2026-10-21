"""The known-answer check, before any comparison: does our small setup show what Kim et al. found?

Kim et al. ("Pre-training under infinite compute", ICLR 2026) trained models that reread a fixed
set of text many times, tuning the number of passes, and scored each run by its held-out loss at
the end. At their smallest size (154M parameters on 210M unique tokens, learning rate 3e-3):
    - with standard weight decay (0.1), the best number of passes was 8; with more, the end
      loss got worse: it overfits;
    - with weight decay 0.8, the best was 16 passes, and its end loss was about 0.09 lower.
If our setup does not show both, it is too small or too short to be in their regime, and a
comparison in it would say nothing about that regime.

Our setup: the SIZE model on the first UNIQUE_TOKENS of the training pool, with their learning
rate, schedule and weight-decay rule (train.py), one run for each weight decay in WEIGHT_DECAYS
and each number of passes in PASSES, on SEED, a seed the comparison will not judge on. A run's
score is its held-out loss at the end, on all the held-out text. The SIZE model has 29.0M
parameters, 3.2M of them outside the embedding and output layer; on 20M unique tokens that is
1.45 parameters per token (theirs 0.73) and 6 tokens per parameter outside those two layers
(theirs 9). The check takes about 2 hours on one A100.

The rules, fixed before the run:
    1. it overfits: with weight decay 0.1, the 32-pass run ends at least RISE above the best
       of the 0.1 runs
    2. strong weight decay helps: the best 0.8 run ends at least GAIN below the best 0.1 run
Also reported, not a rule: the best number of passes for each weight decay (Kim et al.: 8, then 16).
Every run saves its weights at the end; the 0.8 run with 32 passes also saves them after passes
1, 4, 16 and 32, for look_at_neighbours.py.

Run it with:   python run_known_answer.py      (from the rereading folder)
Only the report, from the saved runs:   python run_known_answer.py --report
Each run is saved to results/known_answer/ as it ends; running it again skips the runs already
saved, unless their settings or the training code changed since (see fingerprint). The code has
changed since this check ran (the neighbour parts), so run it again only with --report, or it
retrains all 8 runs.
"""

import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import train
from train import Run

SIZE = "29M"
UNIQUE_TOKENS = 20_000_000
WEIGHT_DECAYS = (0.1, 0.8)
PASSES = (4, 8, 16, 32)
SEED = 100
RISE = 0.05   # nats per token
GAIN = 0.02   # nats per token
FOLDER = Path("results/known_answer")
CHECKPOINTED = (0.8, 32)  # weight decay, passes
CHECKPOINTS = (1, 4, 16, 32)


def run_file(run):
    return FOLDER / f"{run.method}_wd{run.weight_decay}_passes{run.passes}_seed{run.seed}.json"


# The training code's version, taken once when the program starts: the code it trains with.
CODE = hashlib.sha256(b"".join(Path(name).read_bytes() for name in ("train.py", "model.py", "updates.py"))).hexdigest()[:16]


def fingerprint(run):
    """The run's settings, the held-out check size and the training code: a saved run counts only if all match."""
    return {"run": asdict(run), "held_out_windows": train.HELD_OUT_WINDOWS, "code": CODE}


def save(record, path):
    part = path.with_suffix(".part")
    part.write_text(json.dumps(record))
    os.replace(part, path)


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    base = Run(size=SIZE, unique_tokens=UNIQUE_TOKENS, seed=SEED)
    records = {}
    for decay, passes in product(WEIGHT_DECAYS, PASSES):
        run = replace(base, weight_decay=decay, passes=passes)
        path = run_file(run)
        settings = fingerprint(run)  # taken before training, so it describes the code the run used
        if not path.exists() or json.loads(path.read_text()).get("fingerprint") != settings:
            checkpoints = CHECKPOINTS if (decay, passes) == CHECKPOINTED else ()
            record = train.train(run, checkpoint_passes=checkpoints,
                                 checkpoint_file=str(FOLDER / "wd0.8_after_{passes}_passes.pt"),
                                 final_weights_file=str(path.with_suffix(".pt")))
            record["fingerprint"] = settings
            save(record, path)
            print(f"weight decay {decay}, {passes} passes: held-out loss at the end {record['final_held_out']:.3f}, "
                  f"{record['seconds'] / 60:.0f} minutes", flush=True)
        records[decay, passes] = json.loads(path.read_text())
    report(records)


def report(records):
    print("\nHeld-out loss at the end of each run (nats per token):")
    print(f"{'weight decay':>14s}" + "".join(f"{p:>8d}" for p in PASSES) + "   passes")
    for decay in WEIGHT_DECAYS:
        print(f"{decay:>14}" + "".join(f"{records[decay, p]['final_held_out']:8.3f}" for p in PASSES))
    best = {decay: min(records[decay, p]["final_held_out"] for p in PASSES) for decay in WEIGHT_DECAYS}
    rise = records[0.1, max(PASSES)]["final_held_out"] - best[0.1]
    gain = best[0.1] - best[0.8]
    print(f"\n1. With weight decay 0.1, {max(PASSES)} passes end {rise:.3f} above the best 0.1 run "
          f"(rule: at least {RISE}): {'PASSES' if rise >= RISE else 'FAILS'}")
    print(f"2. The best 0.8 run ends {gain:.3f} below the best 0.1 run (rule: at least {GAIN}): "
          f"{'PASSES' if gain >= GAIN else 'FAILS'}")
    best_passes = {decay: min(PASSES, key=lambda p: records[decay, p]["final_held_out"]) for decay in WEIGHT_DECAYS}
    print(f"Not a rule: the best number of passes is {best_passes[0.1]} with weight decay 0.1 and "
          f"{best_passes[0.8]} with 0.8 (Kim et al.: 8 and 16).")


def report_saved():
    """The report from the saved runs, without training; says if any was trained with other code or settings."""
    base = Run(size=SIZE, unique_tokens=UNIQUE_TOKENS, seed=SEED)
    records = {}
    for decay, passes in product(WEIGHT_DECAYS, PASSES):
        run = replace(base, weight_decay=decay, passes=passes)
        record = json.loads(run_file(run).read_text())
        if record["fingerprint"]["run"] != asdict(run):
            print(f"Note: the saved run for weight decay {decay}, {passes} passes has other settings: {record['fingerprint']['run']}")
        records[decay, passes] = record
    codes = {r["fingerprint"]["code"] for r in records.values()}
    print(f"Training code version(s) of the saved runs: {', '.join(sorted(codes))}; now: {fingerprint(base)['code']}")
    report(records)


if __name__ == "__main__":
    report_saved() if "--report" in sys.argv else main()
