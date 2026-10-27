"""Is the paper's reward just progress over time? A look after the runs, not part of the pass rule.

Self-play-0's learner at round 2048, loaded from its saved state, scores the same outputs five ways:
    paper       |gradient now . P x (weights at round 1024 - weights now)|, as the run computed it (reward.py)
    no P        the same without Adam's step sizes P
    one round   the same, looking back to round 2047 instead of 1024 (the paper's last_step)
    progress    |loss with the weights of round 1024 - loss now|: progress over time, measured exactly
    loss        the loss now (our prediction_error reward)
Without P, and to first order, "paper" is "progress": the slope of the loss along the weight change,
times the change, is about the change in the loss.

The outputs are 256 fresh programs from each of the final generators of self-play-0, novelty-0 and
prediction_error-0, and 256 uniform programs from g0, all run on the machine. The prediction_error
generator's programs mostly print random bytes, so they show how each score treats noise.

Run it with:   ../.venv/bin/python look_at_reward.py      (a few minutes on the Mac)
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

import machine
import reward
import score
from generator import sample, uniform_programs
from model import ByteModel
from train import Run, new_model

RESULTS = Path(__file__).resolve().parent / "results"
COUNT = 256
reward.CHUNK = 16     # smaller passes than on the GPU, for the Mac's memory

where = score.device()
state = torch.load(RESULTS / "self-play-0" / "state.pt", map_location="cpu", weights_only=False)
assert state["round"] == 2048
run = Run(**state["record"]["run"])
learner = new_model(run, where)
learner.load_state_dict(state["learner"])
optimizer = torch.optim.AdamW(learner.parameters(), lr=run.learning_rate, betas=(0.9, run.adam_beta2), weight_decay=0.0)
optimizer.load_state_dict(state["learner_optimizer"])
then = {r: {name: w.to(where) for name, w in state["then"][r].items()} for r in (1024, 2047, 2048)}
now = reward.snapshot(learner)
assert all(torch.equal(now[name], then[2048][name]) for name in now), "then[2048] should be the weights now"

sources = {}
for arm in ("self-play-0", "novelty-0", "prediction_error-0"):
    shape = json.loads((RESULTS / arm / "config.json").read_text())["generator"]
    generator = ByteModel(**shape)
    generator.load_state_dict(torch.load(RESULTS / arm / "generator_2048.pth", map_location="cpu",
                                         weights_only=True)["generator_state_dict"])
    torch.manual_seed(0)
    sources[arm.rsplit("-", 1)[0]] = sample(generator, COUNT, torch.device("cpu"))
sources["uniform"] = uniform_programs(COUNT, np.random.default_rng(0))
max_steps, tape_length = machine.limits(run.width, run.layers)
names, rows = [], []
for name, programs in sources.items():
    outputs, _, _ = machine.run(programs, np.arange(COUNT), max_steps, tape_length)
    rows.append(outputs)
    names += [name] * COUNT
rows, names = torch.from_numpy(np.concatenate(rows).astype(np.int64)), np.array(names)

scores = {
    "paper": reward.alignment(learner, rows, reward.direction(learner, optimizer, then[1024]), where),
    "no P": reward.alignment(learner, rows, {name: then[1024][name] - now[name] for name in now}, where),
    "one round": reward.alignment(learner, rows, reward.direction(learner, optimizer, then[2047]), where),
}
old = copy.deepcopy(learner)
with torch.no_grad():
    for name, p in old.named_parameters():
        p.copy_(then[1024][name])
loss_now = reward.sequence_losses(learner, rows, where)
scores["progress"] = (reward.sequence_losses(old, rows, where) - loss_now).abs()
scores["loss"] = loss_now
scores = {k: v.cpu().numpy() for k, v in scores.items()}

print("Each score by where the output came from: median, mean and 90th percentile")
for k, values in scores.items():
    print(f"   {k}")
    for source in sources:
        x = values[names == source]
        print(f"      {source:18s} median {np.median(x):9.3g}   mean {x.mean():9.3g}   90% {np.percentile(x, 90):9.3g}")

print("Noise in self-play's own pool: the median random-byte output's score, standardized like the")
print("generator's advantage (minus the mean, over the spread, of self-play's outputs)")
own, noise = names == "self-play", names == "prediction_error"
for k in ("paper", "no P", "progress"):
    z = (np.median(scores[k][noise]) - scores[k][own].mean()) / scores[k][own].std()
    print(f"   {k:10s} {z:+.2f}")

print("Rank agreement (Spearman) between scores: all 1,024 outputs / self-play's own 256")
order = list(scores)
for i, a in enumerate(order):
    for b in order[i + 1:]:
        print(f"   {a:10s} vs {b:10s} {spearmanr(scores[a], scores[b]).statistic:5.2f} / "
              f"{spearmanr(scores[a][own], scores[b][own]).statistic:5.2f}")

print("Where each score's top 10% (102 outputs) came from")
top = len(names) // 10
for k, values in scores.items():
    best = names[np.argsort(-values)[:top]]
    print(f"   {k:10s} " + ", ".join(f"{source} {np.mean(best == source):.0%}" for source in sources))
