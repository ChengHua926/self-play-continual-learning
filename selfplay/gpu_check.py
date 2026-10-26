"""The first thing to run on a new GPU: is the reward still exact there, and how long does a round take?

1. The reward (reward.alignment) against the exact derivative from autograd, on 16 outputs after
   a few learner steps, and its time for a full pool.
2. The seconds per round of each arm at full size (a pool of 1,536 programs). The first rounds
   have no mutants or replay yet, so rounds 3 to 5 are timed. The self-play run also scores its
   learner, so scoring is tried in the GPU's deterministic mode too.

Run it once before launching runs: it also compiles the machine (numba) and stores the result,
so the runs do not all compile it at the same time.

Run it with:   ../.venv/bin/python gpu_check.py
"""

import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

import generator
import machine
import reward
import train

where = torch.device("cuda")
steps, tape = machine.limits(64, 1)

torch.manual_seed(0)
learner = train.new_model(train.Run(), where)
optimizer = torch.optim.AdamW(learner.parameters(), lr=1e-2, betas=(0.9, train.Run().adam_beta2), weight_decay=0.0)
rng = np.random.default_rng(0)
start = reward.snapshot(learner)
printing = [b"SX[.+]F", b"S,[.,]F", b"S++[.>+<]F", b"SX[>+.<-]F"]
for _ in range(6):
    programs = generator.uniform_programs(12, rng) + printing
    rows = torch.from_numpy(machine.run(programs, rng.integers(0, 2 ** 63, size=16), steps, tape)[0].astype(np.int64))
    train.learner_step(learner, optimizer, rows, where)
along = reward.direction(learner, optimizer, start)
exact = []
for row in rows.to(where):
    learner.zero_grad()
    F.cross_entropy(learner(row[None, :-1])[0].float(), row[1:]).backward()
    exact.append(abs(sum((p.grad * along[name]).sum() for name, p in learner.named_parameters()).item()))
exact = torch.tensor(exact)
ours = reward.alignment(learner, rows, along, where).cpu()
torch.cuda.synchronize()
started = time.time()
reward.alignment(learner, rows.repeat(96, 1), along, where)    # a pool of 1,536 rows
torch.cuda.synchronize()
print(f"reward: largest error / largest value {((ours - exact).abs() / exact.max()).max().item():.1e}, "
      f"{time.time() - started:.2f} s for a pool of 1,536", flush=True)

for arm in train.ARMS:
    with tempfile.TemporaryDirectory() as folder:
        record = train.train(train.Run(arm=arm, rounds=6), folder, where=where, scores=arm == "self-play")
    print(f"{arm}: {np.mean(record['rounds']['seconds'][3:]):.2f} s per round", flush=True)
print(f"most GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
