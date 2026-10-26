"""The generator's reward for a program (Cowsik et al., equation 2).

    r = | < gradient of the output's loss at the learner's weights now,  P x (weights then - weights now) > |

"Weights now" are the learner's weights after this round's step, and "weights then" its weights
after the step of round e // 2, half-way back to the start, so the window grows as training
goes on. P is AdamW's own step size for each weight, lr / (sqrt(v) + eps), where v is the
optimizer's running average of squared gradients. An output the learner has mastered has almost
no gradient. An output whose structure the learner has not been learning has a gradient that
points elsewhere. Both get little reward. The reward is largest for outputs that would teach the
learner a lot, in the direction it has been learning.

The inner product is the derivative of the output's loss along the direction P x (then - now),
computed exactly for all outputs from one ordinary forward and backward pass. Each weight enters
the model through one layer, so the derivative is a sum over layers of
    < gradient of the loss at the layer's output,  how the output moves when the weight moves along the direction >.
For a matrix W used as y = x W^T, the output moves by x U^T, where U is the direction's part for
W. For a norm's weights (y = w * normalized x) it moves by U * normalized x, and for the
embedding (y = W[byte]) by U[byte]. Hooks compute the second factor in the forward pass and
take the inner product when the backward pass reaches the layer. No output's own gradient is
ever formed, and the attention keeps its fused kernel.

The pass runs in 32-bit arithmetic. In 16-bit bfloat16, as the learner trains, it is 2.6 times
faster, but once the learner has trained for 20 steps the rewards came out up to 16% of the
largest reward off (1.6% for the median one), from the coarse 16-bit pass itself.

Two other ways to get it were exact but slower on the GPU, and one was not exact:
    - forward-mode differentiation, as the authors do, through attention written out by hand: 4 s a round;
    - one gradient per output (vmap of grad): 42 s a round, as the fused attention has no batched backward;
    - a central difference (the loss a small step along the direction, minus a small step back) is
      off by 7% to over 100%, because P spans a factor of about 1,000 across weights.

loss_delta, one of the authors' comparison rewards, uses the realized change instead: the
output's loss before the learner's step minus its loss after, as an absolute value (train.py).

Two rewards of ours, for comparison with the paper's (train.py):
    novelty           how far an output is from the outputs seen before, in the learner's own
                      features (novelty below), as in our game comparison, where it was the winner
    prediction_error  the learner's loss on the output after its step (sequence_losses)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RMSNorm

CHUNK = 64          # rows per pass


def per_row_losses(logits, rows):
    """[row]: each output's mean cross-entropy over its 4,095 bytes. (The logits are flattened
    first, the layout rereading/updates.py used on the GPU in deterministic mode.)"""
    losses = F.cross_entropy(logits.flatten(0, 1).float(), rows[:, 1:].flatten(), reduction="none")
    return losses.view(len(rows), -1).mean(dim=1)


@torch.no_grad()
def sequence_losses(model, rows, where):
    """[row]: each output's loss, in 32-bit arithmetic."""
    return torch.cat([per_row_losses(model(part[:, :-1]), part) for part in rows.to(where).split(CHUNK)])


def snapshot(model):
    """A copy of the model's weights, by name."""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


@torch.no_grad()
def direction(learner, optimizer, then):
    """P x (then - now), by parameter name. Zero for weights the optimizer has not stepped yet."""
    group = optimizer.param_groups[0]
    lr, eps, (_, beta2) = group["lr"], group["eps"], group["betas"]
    result = {}
    for name, p in learner.named_parameters():
        state = optimizer.state.get(p, {})
        if not state:
            result[name] = torch.zeros_like(p)
            continue
        v = state["exp_avg_sq"] / (1 - beta2 ** float(state["step"]))     # with Adam's bias correction
        result[name] = lr / (v.sqrt() + eps) * (then[name] - p)
    return result


def alignment(learner, rows, along, where):
    """[row]: |derivative of each output's loss along the direction `along`|, exactly, in 32-bit arithmetic."""
    layers = {module: name for name, module in learner.named_modules()
              if isinstance(module, (nn.Linear, nn.Embedding, RMSNorm))}
    assert {name + ".weight" for name in layers.values()} == set(along), "every weight must be in one hooked layer"
    slopes = []
    for part in rows.to(where).split(CHUNK):
        slope = torch.zeros(len(part), device=where)

        def record(layer, inputs, output):
            x, u = inputs[0].detach(), along[layers[layer] + ".weight"]
            if isinstance(layer, nn.Linear):
                moved = x @ u.T
            elif isinstance(layer, nn.Embedding):
                moved = u[x]
            else:
                moved = u * layer.normalize(x)

            def add(gradient):      # returns nothing, so the backward pass itself is unchanged
                slope.add_((gradient * moved).flatten(1).sum(dim=1))
            output.register_hook(add)

        hooks = [layer.register_forward_hook(record) for layer in layers]
        try:
            losses = per_row_losses(learner(part[:, :-1]), part)
            torch.autograd.grad(losses.sum(), list(learner.parameters()))
        finally:
            for hook in hooks:
                hook.remove()
        slopes.append(slope.abs())
    return torch.cat(slopes)


# Feature novelty. NEAREST, SEEN and a memory that is an even sample of everything seen are as in
# our game comparison (measures.py, learners.py, world_model.Replay). Unlike the game, an output is
# a whole row of 4,095 bytes, mostly zero padding when a program writes little, and its feature is
# the average over all of them, as the learner's loss and the paper's reward also average over all.
NEAREST = 5           # the reward is the average distance to this many nearest outputs
SEEN = 10_000         # earlier outputs drawn each round to compare with
MEMORY = 65_536       # outputs kept: an even random sample of every output trained on so far


@torch.no_grad()
def features(model, rows, where):
    """[row, width]: the model's last hidden layer (what its final norm puts out), averaged over
    each row's positions, in 32-bit arithmetic."""
    if len(rows) == 0:
        return torch.zeros(0, model.width, device=where)
    found = []
    hook = model.ln_f.register_forward_hook(lambda layer, inputs, output: found.append(output.mean(dim=1)))
    try:
        for part in rows.to(where).split(CHUNK):
            model(part[:, :-1])
    finally:
        hook.remove()
    return torch.cat(found)


def novelty(learner, rows, seen, where):
    """[row]: the average distance, in the learner's features as they are now, from each output to
    its NEAREST closest outputs among `seen` (earlier ones) and the rest of this round's."""
    new, old = features(learner, rows, where), features(learner, seen, where)
    # Exact differences, not cdist's faster |a|^2 + |b|^2 - 2ab: the features are about 8 long and,
    # early in training, closer together than that form resolves in 32 bits (off by 2e-3 at 1e-2).
    distances = torch.cdist(new, torch.cat([old, new]), compute_mode="donot_use_mm_for_euclid_dist")
    itself = torch.arange(len(new), device=where)
    distances[itself, len(old) + itself] = float("inf")       # an output is not compared with itself
    return distances.topk(NEAREST, dim=1, largest=False).values.mean(dim=1)


class SeenOutputs:
    """An even random sample of every output the learner has trained on, up to `size`.

    Each new output replaces a random kept one with the right probability, so that every output
    ever seen is equally likely to be kept (reservoir sampling, as world_model.Replay does). A
    window of recent outputs instead would count a program the replay bank brings back from long
    ago as new again.
    """

    def __init__(self, length, size=MEMORY):
        self.rows = np.zeros((size, length), dtype=np.uint8)
        self.seen = 0   # outputs offered so far
        self.kept = 0   # outputs held now

    def add(self, outputs, rng):
        for row in outputs:
            if self.kept < len(self.rows):
                slot = self.kept
                self.kept += 1
            else:
                slot = rng.integers(self.seen + 1)
            if slot < len(self.rows):
                self.rows[slot] = row
            self.seen += 1

    def draw(self, count, rng):
        """Up to `count` of the kept outputs, drawn without repeats, as learner rows."""
        if self.kept == 0:
            return torch.zeros(0, self.rows.shape[1], dtype=torch.long)
        chosen = rng.choice(self.kept, size=min(count, self.kept), replace=False)
        return torch.from_numpy(self.rows[chosen].astype(np.int64))
