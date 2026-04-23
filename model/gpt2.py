import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPT2Config
from model.embeddings import GPT2Embeddings
from model.block import TransformerBlock


class GPT2(nn.Module):
    """
    GPT-2 language model.

    forward(idx, targets) -> (logits, loss)   during training
    forward(idx)          -> (logits, None)   during inference (last position only)
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.embeddings = GPT2Embeddings(config)
        self.blocks     = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f       = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head    = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: same matrix for input embedding and output projection
        self.lm_head.weight = self.embeddings.wte.weight

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters()) - self.embeddings.wte.weight.numel()
        print(
            f"GPT-2 | layers={config.n_layer}  heads={config.n_head}  "
            f"embd={config.n_embd}  vocab={config.vocab_size}  "
            f"ctx={config.block_size}  params={n_params/1e6:.2f}M"
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            # scale down residual projections so the residual stream variance
            # stays ~1 regardless of depth (each block adds two projections)
            if hasattr(module, "_is_residual_proj"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        for block in self.blocks:
            block.attn.c_proj._is_residual_proj = True
            block.mlp.c_proj._is_residual_proj  = True

    def forward(
        self,
        idx:     torch.Tensor,
        targets: torch.Tensor = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embeddings(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])   # only last position at inference
            loss   = None

        return logits, loss

    def configure_optimizer(
        self,
        lr:           float,
        weight_decay: float = 0.1,
        betas:        tuple = (0.9, 0.95),
        device_type:  str   = "cpu",
    ) -> torch.optim.Optimizer:
        # weight decay on matrices only — not biases or norms
        decay    = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        groups   = [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = (p for p in self.parameters() if not trainable_only or p.requires_grad)
        return sum(p.numel() for p in params)
