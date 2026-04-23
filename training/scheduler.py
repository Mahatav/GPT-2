"""
Linear warmup then cosine decay — same schedule used in GPT-2/3 training.

lr(t) = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * decay_ratio))
"""

import math


class CosineWarmupScheduler:
    """
    Manually sets lr on the optimizer each step — no hidden pytorch state.
    """

    def __init__(
        self,
        optimizer:    "torch.optim.Optimizer",
        warmup_iters: int,
        max_iters:    int,
        max_lr:       float,
        min_lr:       float = 6e-5,
    ):
        if warmup_iters >= max_iters:
            raise ValueError("warmup_iters must be < max_iters")

        self.optimizer    = optimizer
        self.warmup_iters = warmup_iters
        self.max_iters    = max_iters
        self.max_lr       = max_lr
        self.min_lr       = min_lr
        self._step_count  = 0
        self._set_lr(0.0)

    def step(self) -> float:
        self._step_count += 1
        lr = self.get_lr(self._step_count)
        self._set_lr(lr)
        return lr

    def get_lr(self, it: int) -> float:
        if it <= self.warmup_iters:
            return self.max_lr * it / self.warmup_iters
        if it > self.max_iters:
            return self.min_lr
        ratio = (it - self.warmup_iters) / (self.max_iters - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    @property
    def step_count(self) -> int:
        return self._step_count

    def state_dict(self) -> dict:
        return {"step_count": self._step_count}

    def load_state_dict(self, state: dict) -> None:
        self._step_count = state["step_count"]
        self._set_lr(self.get_lr(self._step_count))

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr
