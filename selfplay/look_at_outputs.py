"""What a final generator's outputs look like: how many of its programs use the random-byte
instruction (,), how much zlib can shrink the bytes they write, and two example programs.
A look after the runs, not part of the pass rule (compare.py).

zlib keeps close to 100% of random bytes and close to 0% of a constant, so the share it keeps
says how random an output is. Only outputs of at least 64 bytes are counted.

Run it with:   ../.venv/bin/python look_at_outputs.py results/*/generator_2048.pth
"""

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

import machine
from archive import body
from generator import sample
from model import ByteModel

COUNT = 256

for checkpoint in sys.argv[1:]:
    shape = json.loads((Path(checkpoint).parent / "config.json").read_text())["generator"]
    model = ByteModel(**shape)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["generator_state_dict"])
    torch.manual_seed(0)
    programs = sample(model, COUNT, torch.device("cpu"))
    max_steps, tape_length = machine.limits(shape["d_model"], shape["n_layers"])
    outputs, written, _ = machine.run(programs, np.arange(COUNT), max_steps, tape_length)
    random = np.mean([b"," in machine.expand(p) for p in programs])
    long = [i for i in range(COUNT) if written[i] >= 64]
    shrink = np.mean([len(zlib.compress(outputs[i, 1:1 + written[i]].tobytes(), 9)) / written[i] for i in long])
    print(f"{Path(checkpoint).parent.name}: {random:.0%} use ',', zlib keeps {shrink:.0%} of the bytes written")
    for i in long[:2]:
        print(f"    program {bytes(body(programs[i])).decode('ascii', 'replace')!r} -> {written[i]} bytes, "
              f"first 16: {outputs[i, 1:17].tolist()}")
