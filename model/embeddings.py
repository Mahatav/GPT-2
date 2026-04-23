import torch
import torch.nn as nn

from config import GPT2Config


class GPT2Embeddings(nn.Module):
    """
    Learned token + position embeddings summed together.

    wte is tied to the lm_head weight in GPT2 — same matrix handles both
    input lookup and output scoring, which cuts the param count and works
    surprisingly well in practice.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.wte  = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe  = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device)
        return self.drop(self.wte(idx) + self.wpe(pos))
