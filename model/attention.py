import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPT2Config


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask. (B, T, C) -> (B, T, C)"""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.head_dim = config.head_dim

        # project x to Q, K, V in one shot
        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # upper-triangular mask: position i must not see j > i
        T = config.block_size
        self.register_buffer("causal_mask", torch.ones(T, T, dtype=torch.bool).triu(diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)

        # (B, T, C) -> (B, n_head, T, head_dim)
        def to_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # scaled dot-product attention
        scale  = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn   = self.attn_drop(F.softmax(scores, dim=-1))

        out = torch.matmul(attn, v)                                  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)         # merge heads
        return self.resid_drop(self.c_proj(out))
