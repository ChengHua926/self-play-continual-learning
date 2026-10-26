"""The generator's reward for a program (Cowsik et al., equation 2).

    r = | < gradient of the output's loss at the learner's weights now,  P x (weights then - weights now) > |

"Weights now" are the learner's weights after this round's step, and "weights then" its weights
after the step of round e // 2, half-way back to the start, so the window grows as training goes on. P is AdamW's own step size for each weight,
lr / (sqrt(v) + eps), where v is the optimizer's running average of squared gradients. An output
the learner has mastered has almost no gradient. An output whose structure the learner has not
been learning has a gradient that points elsewhere. Both get little reward. The reward is
largest for outputs that would teach the learner a lot, in the direction it has been learning.

The inner product is the derivative of the output's loss along the direction P x (then - now).
As in the paper, it is computed exactly by forward-mode differentiation, one pass for all the
outputs, without forming any output's gradient. The attention is computed with plain matrix
products for this (model.py), since the fused kernel does not support it on every device.

A central difference (the loss a small step along the direction, minus the loss a small step
back) does not work here. The direction's entries range over a factor of about 1,000, because
P is large wherever v is small, so no single step size suits all the weights. On a test row it
was off from the exact derivative by 7% at best, and by more than 100% at small steps.

loss_delta, one of the authors' comparison rewards, uses the realized change instead: the
output's loss before the learner's step minus its loss after, as an absolute value (train.py).
"""

import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

CHUNK = 64          # rows per pass
EXACT_CHUNK = 4     # rows per forward-mode pass, which holds each row's full attention matrix (64 MB each)


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
    """[row]: |derivative of each output's loss along the direction `along`|, exactly."""
    weights = {name: p.detach() for name, p in learner.named_parameters()}
    slopes = []
    for part in rows.to(where).split(EXACT_CHUNK):
        def losses(w):
            return per_row_losses(functional_call(learner, w, (part[:, :-1],), {"plain": True}), part)
        slopes.append(jvp(losses, (weights,), (along,))[1].abs())
    return torch.cat(slopes)
