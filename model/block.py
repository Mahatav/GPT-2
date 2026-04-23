import torch
import torch.nn as nn

from config import GPT2Config
from model.attention import CausalSelfAttention
from model.mlp import MLP


class TransformerBlock(nn.Module):
    """
    One transformer block: pre-norm attention + pre-norm MLP, both with residuals.

    Pre-norm (ln before the sublayer, not after) makes training more stable
    at init when the sublayer outputs tend to be noisy.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
