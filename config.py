from dataclasses import dataclass
from typing import Optional


@dataclass
class GPT2Config:
    vocab_size: int   = 50257
    block_size: int   = 1024   # context window length
    n_layer:    int   = 12
    n_head:     int   = 12
    n_embd:     int   = 768
    dropout:    float = 0.1
    bias:       bool  = True

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_params_estimate(self) -> int:
        embed   = self.vocab_size * self.n_embd + self.block_size * self.n_embd
        per_blk = (
            3 * self.n_embd * self.n_embd +
            self.n_embd * self.n_embd +
            self.n_embd * 4 * self.n_embd +
            4 * self.n_embd * self.n_embd +
            4 * self.n_embd
        )
        return embed + self.n_layer * per_blk + self.n_embd


@dataclass
class TrainConfig:
    # batch
    batch_size:                  int   = 16
    gradient_accumulation_steps: int   = 1

    # optimizer
    max_iters:     int   = 5000
    learning_rate: float = 3e-4
    weight_decay:  float = 0.1
    beta1:         float = 0.9
    beta2:         float = 0.95
    grad_clip:     float = 1.0

    # lr schedule
    warmup_iters:   int   = 100
    lr_decay_iters: int   = 5000
    min_lr:         float = 6e-5

    # eval & logging
    eval_interval:  int = 200
    eval_iters:     int = 50
    log_interval:   int = 10

    # checkpointing
    checkpoint_dir:      str           = "checkpoints"
    checkpoint_interval: int           = 500
    resume_from:         Optional[str] = None


# ── size presets ──────────────────────────────────────────────────────────────

GPT2_NANO = GPT2Config(          # ~0.8M params — fits on CPU, trains in seconds
    vocab_size=256,
    block_size=128,
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.0,
)

GPT2_MICRO = GPT2Config(         # ~10M params — good for char-level experiments
    vocab_size=256,
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
)

GPT2_SMALL = GPT2Config(         # 117M — original GPT-2 small
    vocab_size=50257,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.1,
)

GPT2_MEDIUM = GPT2Config(        # 345M
    vocab_size=50257,
    block_size=1024,
    n_layer=24,
    n_head=16,
    n_embd=1024,
    dropout=0.1,
)

GPT2_LARGE = GPT2Config(         # 762M
    vocab_size=50257,
    block_size=1024,
    n_layer=36,
    n_head=20,
    n_embd=1280,
    dropout=0.1,
)

GPT2_XL = GPT2Config(            # 1.5B
    vocab_size=50257,
    block_size=1024,
    n_layer=48,
    n_head=25,
    n_embd=1600,
    dropout=0.1,
)

MODEL_PRESETS: dict[str, GPT2Config] = {
    "nano":   GPT2_NANO,
    "micro":  GPT2_MICRO,
    "small":  GPT2_SMALL,
    "medium": GPT2_MEDIUM,
    "large":  GPT2_LARGE,
    "xl":     GPT2_XL,
}
