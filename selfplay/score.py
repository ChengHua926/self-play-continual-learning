"""Bits per byte of a learner on held-out real data, as Cowsik et al. score it.

Each evaluation sequence is 4,095 bytes of real data (text, DNA, audio...), which the learner
never trained on. The learner reads the tag byte O and then the data, and at every position
guesses the next byte. The score is the average over positions and sequences of -log2 of the
probability it gave the byte that came: fewer bits per byte means better prediction. Their
scorer takes the first 256 sequences of each corpus; so does this one.

The four corpora shipped with their code are used here (data/selfplay/eval/):
    dclm_ranked   web text, each byte replaced by its frequency rank
    dna           DNA, 8 symbols
    audio_8bit    speech audio, 8-bit samples
    audio_16bit   speech audio, 16-bit samples as byte pairs

Run it with:   .venv/bin/python selfplay/score.py <checkpoint.pth> [<config.json>]
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import OUTPUT_TAG, ByteModel

EVAL_FOLDER = Path(__file__).resolve().parent.parent / "data" / "selfplay" / "eval"
CORPORA = ("dclm_ranked", "dna", "audio_8bit", "audio_16bit")
SEQUENCES = 256
BATCH = 8


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def corpus(name):
    """[sequence, 4096]: the tag byte O, then 4,095 bytes of data."""
    rows = []
    with open(EVAL_FOLDER / f"{name}.jsonl") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line)["sequence"])
            if len(rows) == SEQUENCES:
                break
    data = torch.tensor(np.array(rows, dtype=np.int64))
    return torch.cat([torch.full((len(data), 1), OUTPUT_TAG), data], dim=1)


def load(checkpoint, config, where):
    """A learner from one of the released files (or one saved the same way)."""
    model = ByteModel(**json.loads(Path(config).read_text())["learner"]).to(where)
    state = torch.load(checkpoint, map_location=where, weights_only=True)["learner_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and set(missing) <= {"program_vocab_mask"}, (missing, unexpected)
    return model.eval()


@torch.no_grad()
def bits_per_byte(model, sequences, where):
    total = 0.0
    for batch in sequences.split(BATCH):
        batch = batch.to(where)
        logits = model(batch[:, :-1]).float()
        losses = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten(), reduction="none")
        total += losses.view(len(batch), -1).mean(dim=1).sum().item()
    return total / len(sequences) / math.log(2)


if __name__ == "__main__":
    where = device()
    checkpoint = sys.argv[1]
    config = sys.argv[2] if len(sys.argv) > 2 else str(Path(checkpoint).parent / "config.json")
    model = load(checkpoint, config, where)
    for name in CORPORA:
        print(f"{name:12s} {bits_per_byte(model, corpus(name), where):.4f} bits per byte", flush=True)
