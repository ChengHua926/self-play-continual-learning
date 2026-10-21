"""The language model for the rereading runs: a small GPT-2-style transformer.

It reads CONTEXT tokens and guesses each next one. GPT-2's vocabulary (50,257 tokens) is padded
to 50,304 so the last layer's size divides evenly on a GPU. The output layer has its own weights,
not shared with the token embedding, as in Kim et al.'s models, so that weight decay can reach it
without reaching the embedding. Sizes are named by their total parameter count; the embedding and
the output layer are VOCABULARY x width each, so small models are mostly these two (see SIZES).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCABULARY = 50_304
CONTEXT = 512
SIZES = {               # layers, width, heads   (total parameters; without the embedding and output layer)
    "29M": (4, 256, 4),    # 29.0M; 3.2M
    "50M": (6, 384, 6),    # 49.5M; 10.6M
    "77M": (8, 512, 8),    # 77.0M; 25.2M
    "163M": (12, 768, 12),  # 162.7M; 85.1M
}


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attend = nn.Linear(width, 3 * width)
        self.mix = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.attend(self.norm1(x)).split(width, dim=2)
        q, k, v = (t.view(batch, length, self.heads, -1).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.mix(y.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, size):
        super().__init__()
        layers, width, heads = SIZES[size]
        self.width = width
        self.tokens = nn.Embedding(VOCABULARY, width)
        self.positions = nn.Embedding(CONTEXT, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, VOCABULARY, bias=False)
        self.apply(self.start)

    @staticmethod
    def start(module):
        """GPT-2's starting weights: small random numbers, zero biases."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        """Returns the features fed to the last layer, [batch, length, width], and the guesses, [batch, length, vocabulary]."""
        x = self.tokens(tokens) + self.positions(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = block(x)
        features = self.norm(x)
        return features, self.out(features)
