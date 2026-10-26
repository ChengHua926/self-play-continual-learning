"""The byte-level transformer used for both the learner and the generator.

The same design as in Cowsik et al., "Self-Play Pretraining with Zero Data" (arXiv 2609.30063):
a Llama-style decoder over bytes (vocabulary 256) with a 4,096-byte context. Its parts are the
usual Llama ones:
    - RMSNorm before attention and before the feed-forward layer
    - rotary position embeddings (base 10,000)
    - attention with one key/value head per four query heads, scaled by 1/sqrt(head width)
    - a SwiGLU feed-forward layer, its width 8/3 of the model's, rounded up to a multiple of 256
    - no biases, and an output layer separate from the input embedding
The parameter names match the authors' released checkpoints, so their weights load directly
(huggingface.co/nourya-cohen/solomonoff-paper).

A sequence starts with one tag byte: S for a program, O for a program's output. Program rows
may only use the machine's 18 instructions and F, and nothing after the first F counts; output
rows may use any byte (see allowed_logits).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

INSTRUCTIONS = b"<>+-[].,ZRLNCGHWVX"   # 8 Brainfuck instructions, then 10 macros (machine.py)
END = ord("F")                         # ends a program
PROGRAM_TAG, OUTPUT_TAG = ord("S"), ord("O")
VOCABULARY = 256
CONTEXT = 4096


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def normalize(self, x):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)).to(x.dtype)

    def forward(self, x):
        return self.weight * self.normalize(x)


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    turned = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos + turned * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.kv_heads = heads, max(1, heads // 4)
        self.head_width = width // heads
        self.q_proj = nn.Linear(width, heads * self.head_width, bias=False)
        self.k_proj = nn.Linear(width, self.kv_heads * self.head_width, bias=False)
        self.v_proj = nn.Linear(width, self.kv_heads * self.head_width, bias=False)
        self.o_proj = nn.Linear(heads * self.head_width, width, bias=False)

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q = self.q_proj(x).view(batch, length, self.heads, self.head_width).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.kv_heads, self.head_width).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_width).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        groups = self.heads // self.kv_heads
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.head_width ** -0.5)
        return self.o_proj(y.transpose(1, 2).reshape(batch, length, width))


class FeedForward(nn.Module):
    def __init__(self, width):
        super().__init__()
        hidden = 256 * math.ceil(int(8 * width / 3) / 256)
        self.gate_proj = nn.Linear(width, hidden, bias=False)
        self.up_proj = nn.Linear(width, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(width), RMSNorm(width)
        self.attn = Attention(width, heads)
        self.mlp = FeedForward(width)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class ByteModel(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, max_len=CONTEXT, vocab_size=VOCABULARY, **unused):
        super().__init__()
        self.width, self.heads, self.layers = d_model, n_heads, n_layers
        self.wte = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        head_width = d_model // n_heads
        self.register_buffer("inv_freq", 1.0 / 10000 ** (torch.arange(0, head_width, 2).float() / head_width),
                             persistent=False)
        allowed = torch.zeros(vocab_size, dtype=torch.bool)
        allowed[list(INSTRUCTIONS) + [END]] = True
        self.register_buffer("program_vocab_mask", allowed)  # the name the released weights use

    def forward(self, tokens, last=False):
        """[batch, length, 256]: next-byte logits, before allowed_logits. last: only the last
        position's ([batch, 1, 256]), which is all sampling needs."""
        angles = torch.outer(torch.arange(tokens.shape[1], device=tokens.device).float(), self.inv_freq)
        angles = torch.cat((angles, angles), dim=-1)[None, None]
        x = self.wte(tokens)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.ln_f(x[:, -1:] if last else x))


def allowed_logits(logits, tokens):
    """Program rows: only the 18 instructions and F, and nothing after the first F.
    Output rows (tag O): every byte allowed."""
    output_row = tokens[:, :1] == OUTPUT_TAG                                   # [batch, 1]
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    first_end = torch.where(tokens == END, positions, tokens.shape[1]).amin(dim=1, keepdim=True)
    before_end = positions < first_end                                         # [batch, length]
    instruction = torch.zeros(VOCABULARY, dtype=torch.bool, device=tokens.device)
    instruction[list(INSTRUCTIONS) + [END]] = True
    keep = output_row[..., None] | (before_end[..., None] & instruction)
    keep = keep | (torch.arange(VOCABULARY, device=tokens.device) == END)    # F stays possible
    return torch.where(keep, logits, float("-inf"))
