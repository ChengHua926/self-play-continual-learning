"""One rereading run: train the model on a fixed set of text, read many times, and record its held-out loss.

A run (Run, below) names everything that can differ between runs. Its fixed set is the first
unique_tokens of the training pool (text.py), cut into windows of CONTEXT tokens. Each pass
reads every window once, in a new random order. The model's average next-token loss on the
first HELD_OUT_WINDOWS held-out windows is recorded at a few points in the first pass and then
after every pass.

The run's score is its held-out loss at the end, measured on all the held-out windows (3,699,
from 1,595 documents), not just the HELD_OUT_WINDOWS used for the curve along the way. The best
loss along the way is recorded too, and so is the loss at the end on the first TRAINING_SAMPLE
training windows: the gap between the two shows how much of what was learned was memorized.

Training follows Kim et al. ("Pre-training under infinite compute", ICLR 2026, appendix A and
their released code): AdamW with betas (0.9, 0.95), the learning rate rising over the first 1% of
steps and then falling along a cosine to 0 at the end of the run, gradients clipped to length 1,
weight decay (multiplied by the learning rate, as in PyTorch) on the weight matrices but not on
the embeddings, biases or norms, 16-bit bfloat16 arithmetic for the forward pass. Like them, a
run is scored by its held-out loss at the end, where the learning rate has reached 0; the number
of passes is a setting to tune, like the weight decay. The best held-out loss along the way is
recorded too.
"""

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

import updates
from model import CONTEXT, GPT

TRAINING_FILE = "../data/tokens/training.npy"
HELD_OUT_FILE = "../data/tokens/held_out.npy"
HELD_OUT_WINDOWS = 256       # 131,072 tokens
TRAINING_SAMPLE = 256        # windows
WARMUP = 0.01                # share of the run's steps
BETAS = (0.9, 0.95)
EARLY_CHECKS = (0.25, 0.5)   # passes, besides the end of every pass


@dataclass(frozen=True)
class Run:
    size: str = "29M"             # a size in model.SIZES
    unique_tokens: int = 2_000_000
    passes: int = 32
    method: str = "plain"         # plain, neighbours, shuffled neighbours or batch agreement (updates.py)
    weight_decay: float = 0.1
    learning_rate: float = 3e-3   # Kim et al.'s best for their smallest model, with or without strong weight decay
    sequences: int = 64           # per batch
    seed: int = 0
    spreads: float = 2.0          # neighbours: how many standard deviations above the batch's average cosine
    centred: bool = True          # neighbours: subtract the batch's average gradient first
    exact: bool = False           # neighbours: 32-bit instead of bfloat16 for the gradients' inner products


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def windows(tokens):
    """[window, CONTEXT + 1]: the tokens cut into windows; each window's inputs are its first CONTEXT
    tokens and its targets the last CONTEXT, one token later."""
    count = (len(tokens) - 1) // CONTEXT
    starts = np.arange(count) * CONTEXT
    return np.stack([tokens[start:start + CONTEXT + 1] for start in starts]).astype(np.int64)


def learning_rate(step, total, peak):
    warmup = max(round(WARMUP * total), 1)
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step + 1 - warmup) / max(total - warmup, 1)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def optimizer_for(model, run):
    """Weight decay on the weight matrices, the output layer's included; not on the embeddings, biases or norms."""
    embeddings = {id(model.tokens.weight), id(model.positions.weight)}
    decayed = [p for p in model.parameters() if p.dim() >= 2 and id(p) not in embeddings]
    not_decayed = [p for p in model.parameters() if p.dim() < 2 or id(p) in embeddings]
    return torch.optim.AdamW([{"params": decayed, "weight_decay": run.weight_decay},
                              {"params": not_decayed, "weight_decay": 0.0}],
                             lr=run.learning_rate, betas=BETAS)


def low_precision(where):
    return torch.autocast(where.type, dtype=torch.bfloat16)


@torch.no_grad()
def held_out_loss(model, held_out, where):
    model.eval()
    total = 0.0
    for batch in held_out.split(32):
        with low_precision(where):
            _, logits = model(batch[:, :-1])
        total += updates.sequence_losses(logits, batch[:, 1:]).sum().item()
    model.train()
    return total / len(held_out)


def batch_weights(run, features, logits, targets, shuffler, recent):
    """[sequence]: the weight of each sequence's loss for this method, or None for plain training.

    Notes in recent how degenerate the weights are: the share at 0, the effective batch size
    (sum of weights, squared, over the sum of squared weights: 64 when all are equal), the
    largest weight, and whether the batch fell back to plain weights.
    """
    if run.method == "plain":
        return None
    inner = updates.gradient_inner_products(features, logits, targets, exact=run.exact)
    if run.method == "batch agreement":
        weights, fell_back = updates.batch_agreement_weights(inner)
    else:
        weights, fell_back = updates.neighbour_weights(updates.cosines(inner, run.centred), run.spreads)
        if run.method == "shuffled neighbours":
            weights = weights[torch.randperm(len(weights), generator=shuffler).to(weights.device)]
        else:
            assert run.method == "neighbours", f"unknown method {run.method}"
    recent["zero"].append((weights == 0).float().mean().item())
    recent["effective"].append((weights.sum() ** 2 / (weights ** 2).sum()).item())
    recent["largest"].append(weights.max().item())
    recent["fallback"].append(fell_back)
    return weights


def train(run, where=None, checkpoint_passes=(), checkpoint_file=None, final_weights_file=None):
    """Train one run. Returns its record: held-out loss at each check, at the end, and its best.

    checkpoint_passes: after these passes, save the model's weights to checkpoint_file.format(passes=...).
    final_weights_file: save the weights at the end there, so the end can be measured again without retraining.
    """
    where = where or device()
    torch.manual_seed(run.seed)
    order = np.random.default_rng(run.seed)
    training = torch.from_numpy(windows(np.load(TRAINING_FILE)[:run.unique_tokens + 1])).to(where)
    all_held_out = torch.from_numpy(windows(np.load(HELD_OUT_FILE))).to(where)
    held_out = all_held_out[:HELD_OUT_WINDOWS]
    model = GPT(run.size).to(where)
    optimizer = optimizer_for(model, run)
    steps_per_pass = len(training) // run.sequences
    total = steps_per_pass * run.passes
    checks = {round(share * steps_per_pass) for share in EARLY_CHECKS} | \
             {steps_per_pass * (p + 1) for p in range(run.passes)}

    shuffler = torch.Generator().manual_seed(run.seed)
    record = {"run": asdict(run), "steps_per_pass": steps_per_pass, "check_steps": [], "held_out": [],
              "training": [], "zero_weights": [], "effective_batch": [], "largest_weight": [], "fallbacks": []}
    recent, started, step = {"loss": [], "zero": [], "effective": [], "largest": [], "fallback": []}, time.time(), 0
    for _ in range(run.passes):
        for batch in training[torch.from_numpy(order.permutation(len(training)))[:steps_per_pass * run.sequences]] \
                .split(run.sequences):
            inputs, targets = batch[:, :-1], batch[:, 1:]
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, total, run.learning_rate)
            optimizer.zero_grad(set_to_none=True)
            with low_precision(where):
                features, logits = model(inputs)
            losses = updates.sequence_losses(logits, targets)
            weights = batch_weights(run, features, logits, targets, shuffler, recent)
            loss = losses.mean() if weights is None else (weights * losses).mean()
            loss.backward()
            recent["loss"].append(losses.mean().item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step in checks:
                record["check_steps"].append(step)
                record["held_out"].append(held_out_loss(model, held_out, where))
                record["training"].append(float(np.mean(recent["loss"])))
                if weights is not None:
                    record["zero_weights"].append(float(np.mean(recent["zero"])))
                    record["effective_batch"].append(float(np.mean(recent["effective"])))
                    record["largest_weight"].append(float(np.mean(recent["largest"])))
                    record["fallbacks"].append(int(np.sum(recent["fallback"])))
                recent = {key: [] for key in recent}
                passes_done = step / steps_per_pass
                if round(passes_done, 6) in checkpoint_passes and checkpoint_file:
                    torch.save(model.state_dict(), checkpoint_file.format(passes=round(passes_done)))
    record["final_held_out"] = held_out_loss(model, all_held_out, where)
    record["final_training_sample"] = held_out_loss(model, training[:TRAINING_SAMPLE], where)
    record["best_held_out"] = min(record["held_out"])
    if final_weights_file:
        torch.save(model.state_dict(), final_weights_file)
    record["seconds"] = time.time() - started
    return record
