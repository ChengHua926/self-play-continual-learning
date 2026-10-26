"""What a generator writes, in a few numbers, to compare our runs with the authors' released ones.

For each generator checkpoint, sample PROGRAMS programs, run them on the machine, and report:
    symbols   mean program length (symbols between S and F)
    kl        mean of log g(x) - log g0(x) over the generator's own programs: how far it has moved
              from the prior, in nats (an estimate of KL(g || g0))
    wrote     share of programs that write at least one byte
    bytes     mean number of bytes written
    depth     share of programs at loop depth 0, 1, 2 and 3 or more

movement() compares two checkpoints of one model, say rounds 0 and 256: the median change of a
weight in each group (embedding, attention, feed-forward, norms, output). Unlike the numbers
above, it depends almost only on the learning rate, so it tests the learning rates that train.py
inferred from the released weights.

Run it with:   ../.venv/bin/python fingerprint.py <generator_XXXX.pth> [...]
          or:  ../.venv/bin/python fingerprint.py --movement <earlier.pth> <later.pth>
(each checkpoint's config.json is read from its folder).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

import machine
from archive import body
from generator import log_probs, prior_log_probs, sample
from model import ByteModel

PROGRAMS = 1024


def fingerprint(checkpoint, where, seed=0):
    shape = json.loads((Path(checkpoint).parent / "config.json").read_text())["generator"]
    model = ByteModel(**shape).to(where)
    model.load_state_dict(torch.load(checkpoint, map_location=where, weights_only=True)["generator_state_dict"])
    torch.manual_seed(seed)
    programs = sample(model, PROGRAMS, where)
    with torch.no_grad():
        kl = (log_probs(model, programs, where).cpu() - prior_log_probs(programs)).mean().item()
    max_steps, tape_length = machine.limits(shape["d_model"], shape["n_layers"])
    _, written, depths = machine.run(programs, np.arange(PROGRAMS), max_steps, tape_length)
    return {"symbols": float(np.mean([len(body(p)) for p in programs])), "kl": kl,
            "wrote": float((written > 0).mean()), "bytes": float(written.mean()),
            "depth": [float((np.minimum(depths, 3) == d).mean()) for d in range(4)]}


GROUPS = {"embedding": ("wte",), "attention": ("attn",), "feed-forward": ("mlp",),
          "norms": ("layernorm", "ln_f"), "output": ("lm_head",)}


def movement(earlier, later):
    """{group: median |change| of a weight} between two checkpoints of the same model."""
    a, b = (next(v for k, v in torch.load(f, map_location="cpu", weights_only=True).items() if k.endswith("state_dict"))
            for f in (earlier, later))
    result = {}
    for group, marks in GROUPS.items():
        names = [n for n in a if a[n].is_floating_point() and any(m in n for m in marks)]
        result[group] = torch.cat([(b[n] - a[n]).abs().flatten() for n in names]).median().item()
    return result


if __name__ == "__main__":
    where = torch.device("cpu")
    if sys.argv[1] == "--movement":
        print(", ".join(f"{group} {change:.4f}" for group, change in movement(sys.argv[2], sys.argv[3]).items()))
        sys.exit()
    for checkpoint in sys.argv[1:]:
        f = fingerprint(checkpoint, where)
        print(f"{checkpoint}: {f['symbols']:.1f} symbols, KL {f['kl']:.1f} nats, {f['wrote']:.0%} write, "
              f"{f['bytes']:.0f} bytes, depth 0/1/2/3+ " + "/".join(f"{d:.0%}" for d in f["depth"]), flush=True)
