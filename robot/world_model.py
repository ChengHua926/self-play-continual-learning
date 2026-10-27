"""The learner: a world model that predicts each next state of an experiment from what came before.

A small transformer built from the paper's own blocks (selfplay/model.py: Llama-style, rotary
positions, no biases), run over time steps instead of bytes. Step t's input is the state at t
(33 numbers) and the action taken at t (7), the state scaled to about unit size. From steps 0 to t it
predicts the state at t + 1, as the change from the state at t, with a mean and a spread for each
of the 33 numbers. The loss is the negative log-likelihood of the real next state under that
prediction, in nats per number (lower is better), averaged over an experiment's steps.

The scales come from scales.npz: each number's spread over random experiments, measured once
before the runs (train.py makes it), the same for every arm.

Its size and settings are guesses (train.py's Run): 2 blocks of width 128, about half a million weights.
"""

import math

import torch
import torch.nn as nn

import paths  # noqa: F401  (makes ../selfplay importable)
from model import Block, RMSNorm

STATE, ACTION = 33, 7


class WorldModel(nn.Module):
    def __init__(self, scales, width=128, heads=4, layers=2):
        super().__init__()
        self.width = width
        self.read = nn.Linear(STATE + ACTION, width, bias=False)       # in place of the paper's byte embedding
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.ln_f = RMSNorm(width)
        self.predict = nn.Linear(width, 2 * STATE, bias=False)         # a mean and a log-spread per number
        head_width = width // heads
        self.register_buffer("inv_freq", 1.0 / 10000 ** (torch.arange(0, head_width, 2).float() / head_width),
                             persistent=False)
        for name in ("state_mean", "state_spread", "change_spread"):
            self.register_buffer(name, torch.as_tensor(scales[name], dtype=torch.float32))

    def forward(self, states, actions):
        """states [batch, steps, 33] and actions [batch, steps, 7] -> the predicted change of each
        state by the next step, as a mean and a log-spread [batch, steps, 33] each, in scaled units."""
        angles = torch.outer(torch.arange(states.shape[1], device=states.device).float(), self.inv_freq)
        angles = torch.cat((angles, angles), dim=-1)[None, None]
        x = self.read(torch.cat([(states - self.state_mean) / self.state_spread, actions], dim=-1))
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        mean, log_spread = self.predict(self.ln_f(x)).chunk(2, dim=-1)
        return mean, log_spread.clamp(-8, 4)


def step_losses(model, states, actions, next_states):
    """[batch, steps]: the loss at each step, nats per number.
    states and next_states are [batch, steps, 33]: next_states[:, t] follows states[:, t]."""
    mean, log_spread = model(states, actions)
    change = (next_states - states) / model.change_spread
    nll = 0.5 * ((change - mean) / log_spread.exp()) ** 2 + log_spread + 0.5 * math.log(2 * math.pi)
    return nll.mean(dim=2)


def losses(model, states, actions, next_states):
    """[batch]: each experiment's loss, nats per number, averaged over its steps."""
    return step_losses(model, states, actions, next_states).mean(dim=1)


def pack(states, actions, next_states):
    """One tensor per experiment, [batch, steps, 73], so the paper's reward code can split it in chunks."""
    return torch.cat([states, actions, next_states], dim=-1)


def packed_losses(model, part):
    """losses() for packed rows: what reward.alignment calls."""
    return losses(model, part[..., :STATE], part[..., STATE:STATE + ACTION], part[..., STATE + ACTION:])
