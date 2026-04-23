import torch
import torch.nn as nn

from config import GPT2Config


class MLP(nn.Module):
    """Feed-forward block: expand to 4x, GELU, contract back. Applied per-position."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc   = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.drop   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(self.gelu(self.c_fc(x))))
