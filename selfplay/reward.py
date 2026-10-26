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

The pass runs in 16-bit bfloat16 arithmetic, like the learner's own training pass (and, most
likely, the authors' fused kernel). On the GPU that is 2.6 times faster than 32-bit, and the
rewards stay within 0.5% of the largest one, in the same order (rank correlation 0.998 or more).
The tests check the 32-bit version against autograd as well.

Two other ways to get it were exact but slower on the GPU, and one was not exact:
    - forward-mode differentiation, as the authors do, through attention written out by hand: 4 s a round;
    - one gradient per output (vmap of grad): 42 s a round, as the fused attention has no batched backward;
    - a central difference (the loss a small step along the direction, minus a small step back) is
      off by 7% to over 100%, because P spans a factor of about 1,000 across weights.

loss_delta, one of the authors' comparison rewards, uses the realized change instead: the
output's loss before the learner's step minus its loss after, as an absolute value (train.py).
"""

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


def alignment(learner, rows, along, where, precision=torch.bfloat16):
    """[row]: |derivative of each output's loss along the direction `along`| (precision: bfloat16 or float32)."""
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
                slope.add_((gradient.float() * moved.float()).flatten(1).sum(dim=1))
            output.register_hook(add)

        hooks = [layer.register_forward_hook(record) for layer in layers]
        try:
            with torch.autocast(where.type, dtype=torch.bfloat16, enabled=precision == torch.bfloat16):
                logits = learner(part[:, :-1])
            torch.autograd.grad(per_row_losses(logits, part).sum(), list(learner.parameters()))
        finally:
            for hook in hooks:
                hook.remove()
        slopes.append(slope.abs())
    return torch.cat(slopes)
