"""How the novelty reward was fooled. A look after the runs, not part of the pass rule.

Novelty paid an output for being far, in the learner's features, from the outputs seen before
(reward.novelty). Most of novelty-0's final programs use the random-byte instruction, so each run
of the same program writes different bytes. This runs 64 of its programs 8 times each, with
different random tapes, and asks, with novelty-0's learner at round 2048 and 10,000 outputs
drawn from its memory:
    - the novelty the reward measured (mean distance to the 5 nearest outputs in memory);
    - the same for 512 other outputs from memory, for scale;
    - how far each run lands from the nearest other run of the same program;
    - the novelty if the other 7 runs were in memory too;
    - the learner's loss on these outputs.

Run it with:   ../.venv/bin/python look_at_novelty.py      (a minute or two on the Mac)
"""

import json
from pathlib import Path

import numpy as np
import torch

import machine
import reward
import score
from generator import sample
from model import ByteModel
from train import Run, new_model

RESULTS = Path(__file__).resolve().parent / "results"
PROGRAMS, RUNS = 64, 8
EXACT = "donot_use_mm_for_euclid_dist"     # as reward.novelty computes distances
reward.CHUNK = 16                           # smaller passes than on the GPU, for the Mac's memory

where = score.device()
state = torch.load(RESULTS / "novelty-0" / "state.pt", map_location="cpu", weights_only=False)
assert state["round"] == 2048
run = Run(**state["record"]["run"])
learner = new_model(run, where)
learner.load_state_dict(state["learner"])
memory_rows, _, kept = state["memory"]
drawn = memory_rows[np.random.default_rng(0).choice(kept, size=10_000 + 512, replace=False)]
seen = torch.from_numpy(drawn[:10_000].astype(np.int64))
others = torch.from_numpy(drawn[10_000:].astype(np.int64))

shape = json.loads((RESULTS / "novelty-0" / "config.json").read_text())["generator"]
generator = ByteModel(**shape)
generator.load_state_dict(torch.load(RESULTS / "novelty-0" / "generator_2048.pth", map_location="cpu",
                                     weights_only=True)["generator_state_dict"])
torch.manual_seed(0)
programs = sample(generator, PROGRAMS, torch.device("cpu"))
each_run = [p for p in programs for _ in range(RUNS)]      # program 0 eight times, then program 1, ...
max_steps, tape_length = machine.limits(run.width, run.layers)
outputs, _, _ = machine.run(each_run, np.arange(len(each_run)), max_steps, tape_length)
rows = torch.from_numpy(outputs.astype(np.int64))
changes = np.mean([len({outputs[i * RUNS + j].tobytes() for j in range(RUNS)}) > 1 for i in range(PROGRAMS)])

new, old, other = (reward.features(learner, r, where) for r in (rows, seen, others))
to_memory = torch.cdist(new, old, compute_mode=EXACT)
novelty = to_memory.topk(reward.NEAREST, dim=1, largest=False).values.mean(dim=1)
other_novelty = torch.cdist(other, old, compute_mode=EXACT).topk(reward.NEAREST, dim=1, largest=False).values.mean(dim=1)
by_program = new.view(PROGRAMS, RUNS, -1)
between_runs = torch.cdist(by_program, by_program, compute_mode=EXACT)     # [program, run, run]
between_runs[:, torch.arange(RUNS), torch.arange(RUNS)] = float("inf")     # a run is not compared with itself
with_other_runs = torch.cat([to_memory.view(PROGRAMS, RUNS, -1), between_runs], dim=2)
novelty_with_other_runs = with_other_runs.topk(reward.NEAREST, dim=2, largest=False).values.mean(dim=2)
loss = reward.sequence_losses(learner, rows, where)

print(f"{PROGRAMS} of novelty-0's final programs, each run {RUNS} times: {changes:.0%} write different bytes from run to run")
print(f"   novelty, as the reward measured it:                 median {novelty.median():.2f}")
print(f"   the same for 512 other outputs from memory:         median {other_novelty.median():.2f}")
print(f"   distance to the nearest other run of the program:   median {between_runs.min(dim=2).values.median():.2f}")
print(f"   novelty if the other 7 runs were in memory too:     median {novelty_with_other_runs.median():.2f}")
print(f"   the learner's loss on these outputs:                median {loss.median():.3f} nats per byte")
