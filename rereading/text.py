"""The text for the rereading runs: one DCLM file, tokenized once, split by document.

DCLM-baseline, the corpus Kim et al. used, comes as compressed files of web documents, one JSON
object per line with the document in "text". This reads one file in order and tokenizes each
document with GPT-2's tokenizer, ending it with GPT-2's end-of-text token.
    - The first documents, HELD_OUT tokens' worth, become the held-out text. No run trains on them.
    - The documents after them become the training pool, TRAINING tokens in all.
A run takes its fixed training set from the start of the pool, so a smaller set is the start of
a bigger one.

Web text repeats itself: an archive page can hold posts that also appear on other pages. A
held-out document that shares even one stretch of COPIED tokens with the training pool would
reward memorizing, which is what the runs test, so every such document is left out.

Run it with:   .venv/bin/python -I rereading/text.py data/dclm/shard_00000000_processed.jsonl.zst data/tokens
(-I because it reads a downloaded file.) It writes held_out.npy and training.npy, as 16-bit token numbers.
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import tiktoken
import zstandard

HELD_OUT = 2_000_000
TRAINING = 40_000_000
COPIED = 50  # tokens


def stretches(tokens):
    """A 64-bit fingerprint of every stretch of COPIED tokens, one per starting position."""
    tokens = np.asarray(tokens, dtype=np.uint64)
    fingerprint = np.zeros(len(tokens) - COPIED + 1, dtype=np.uint64)
    for k in range(COPIED):  # multiply-and-add, wrapping around at 2^64
        fingerprint = fingerprint * np.uint64(1_000_003) + tokens[k:len(tokens) - COPIED + 1 + k]
    return fingerprint


def not_copied(held_out, training):
    """The held-out documents that share no stretch of COPIED tokens with the training pool."""
    seen = np.sort(stretches(np.concatenate(training)))
    kept = []
    for document in held_out:
        if len(document) < COPIED:
            kept.append(document)
            continue
        mine = stretches(document)
        where = np.minimum(np.searchsorted(seen, mine), len(seen) - 1)
        if not (seen[where] == mine).any():
            kept.append(document)
    return kept


def documents(path):
    with open(path, "rb") as file:
        lines = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(file), encoding="utf-8")
        for line in lines:
            yield json.loads(line)["text"]


def main(source, folder):
    encoding = tiktoken.get_encoding("gpt2")
    held_out, training = [], []
    held_out_size = training_size = 0
    for text in documents(source):
        tokens = encoding.encode_ordinary(text) + [encoding.eot_token]
        if held_out_size < HELD_OUT:
            held_out.append(tokens)
            held_out_size += len(tokens)
        else:
            training.append(tokens)
            training_size += len(tokens)
            if training_size >= TRAINING:
                break
    kept = not_copied(held_out, training)
    print(f"held out: left out {len(held_out) - len(kept)} of {len(held_out):,} documents for sharing "
          f"a {COPIED}-token stretch with the training pool")
    held_out = kept
    Path(folder).mkdir(parents=True, exist_ok=True)
    for name, parts in (("held_out", held_out), ("training", training)):
        tokens = np.concatenate([np.array(part, dtype=np.uint16) for part in parts])
        np.save(Path(folder) / f"{name}.npy", tokens)
        print(f"{name}: {len(parts):,} documents, {len(tokens):,} tokens")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
