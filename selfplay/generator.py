"""Sampling programs from the generator, and the log-probabilities its training needs.

The generator is a ByteModel (model.py) that writes the tag S, then up to MAX_PROGRAM bytes from
the 19-symbol alphabet, ending with F. It samples at temperature 1, as in Cowsik et al.

The prior it is pulled toward, g0, gives a program of n symbols probability 19^-n: every
symbol equally likely at every position, so short programs are much likelier than long ones.
uniform_programs() draws programs from g0 itself, for the arm that has no generator.

A program that reaches MAX_PROGRAM symbols without an F is cut off there and has no F; it runs
all its symbols, as in the authors' sampling (their sample_programs.py).
"""

import math

import numpy as np
import torch

from model import END, INSTRUCTIONS, PROGRAM_TAG, allowed_logits

MAX_PROGRAM = 255
SYMBOLS = 19          # 18 instructions and F
ALPHABET = list(INSTRUCTIONS) + [END]
COLUMN = torch.zeros(256, dtype=torch.long)   # each symbol's place in ALPHABET
COLUMN[ALPHABET] = torch.arange(SYMBOLS)


@torch.no_grad()
def sample(generator, count, where):
    """A list of count programs, each as bytes: S, the symbols, then F (none if cut off)."""
    tokens = torch.full((count, 1), PROGRAM_TAG, dtype=torch.long, device=where)
    finished = torch.zeros(count, dtype=torch.bool, device=where)
    for _ in range(MAX_PROGRAM):
        logits = allowed_logits(generator(tokens), tokens)[:, -1].float()
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
        nxt = torch.where(finished, torch.full_like(nxt, END), nxt)
        tokens = torch.cat([tokens, nxt[:, None]], dim=1)
        finished |= nxt == END
        if finished.all():
            break
    programs = []
    for row in tokens.tolist():
        body = row[1:row.index(END) + 1] if END in row else row[1:]
        programs.append(bytes([PROGRAM_TAG] + body))
    return programs


def uniform_programs(count, rng):
    """count programs from g0: each symbol drawn uniformly from the 19 until F, or until cut off."""
    alphabet = np.frombuffer(INSTRUCTIONS + bytes([END]), dtype=np.uint8)
    programs = []
    for row in alphabet[rng.integers(0, SYMBOLS, size=(count, MAX_PROGRAM))]:
        body = row.tobytes()
        end = body.find(bytes([END]))
        programs.append(bytes([PROGRAM_TAG]) + (body[:end + 1] if end >= 0 else body))
    return programs


def as_batch(programs, where):
    """[program, longest + 1] tokens, padded with F after each program's end."""
    longest = max(len(p) for p in programs)
    batch = torch.full((len(programs), longest), END, dtype=torch.long)
    for i, p in enumerate(programs):
        batch[i, :len(p)] = torch.tensor(list(p))
    return batch.to(where)


def log_probs(generator, programs, where):
    """[program]: log-probability the generator gives each program, its F included.

    Only the logits of the 19 symbols are kept. At every position that counts, allowed_logits
    would rule out all other bytes anyway, so this gives the same numbers, and the pool of 1,536
    programs needs about a thirteenth of the memory for them.
    """
    batch = as_batch(programs, where)
    logits = generator(batch[:, :-1])[..., ALPHABET].float()
    per_symbol = torch.log_softmax(logits, dim=-1).gather(-1, COLUMN.to(where)[batch[:, 1:], None]).squeeze(-1)
    lengths = torch.tensor([len(p) - 1 for p in programs], device=where)   # symbols after S, F included
    counted = torch.arange(batch.shape[1] - 1, device=where)[None] < lengths[:, None]
    return (per_symbol * counted).sum(dim=1)


def prior_log_probs(programs):
    """[program]: log g0 = -(number of symbols, F included) x log 19."""
    return torch.tensor([-(len(p) - 1) * math.log(SYMBOLS) for p in programs])
