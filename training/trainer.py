"""
Training loop for GPT-2.

The Trainer class handles:
  - Sampling random batches from train/val datasets
  - The forward + backward + optimizer step loop
  - Gradient accumulation (simulate larger batches without more memory)
  - Gradient norm clipping (prevent exploding gradients)
  - Periodic evaluation on the validation set
  - Periodic checkpoint saving
  - Logging through the Logger interface
"""

import time
import os
import numpy as np
import torch

from config import GPT2Config, TrainConfig
from model.gpt2 import GPT2
from training.dataset import TextDataset
from training.scheduler import CosineWarmupScheduler
from utils.logger import Logger
from utils.checkpoint import save_checkpoint, load_checkpoint, latest_checkpoint


class Trainer:
    """
    Manages the GPT-2 training loop.

    Parameters
    ----------
    model      : the GPT2 instance to train
    train_data : TextDataset for training batches
    val_data   : TextDataset for evaluation batches
    config     : TrainConfig with all loop hyperparameters
    device     : torch.device to run on
    logger     : Logger instance (one will be created if not provided)
    """

    def __init__(
        self,
        model:      GPT2,
        train_data: TextDataset,
        val_data:   TextDataset,
        config:     TrainConfig,
        device:     torch.device,
        logger:     Logger = None,
    ):
        self.model      = model
        self.train_data = train_data
        self.val_data   = val_data
        self.config     = config
        self.device     = device
        self.log        = logger or Logger()

        self.model_config = model.config   # GPT2Config (for saving)

        # ── Optimizer + scheduler ──────────────────────────────────────────
        self.optimizer = model.configure_optimizer(
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            device_type=device.type,
        )
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_iters=config.warmup_iters,
            max_iters=config.lr_decay_iters,
            max_lr=config.learning_rate,
            min_lr=config.min_lr,
        )

        # ── Training state ─────────────────────────────────────────────────
        self.iteration  = 0
        self.best_val   = float("inf")
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []

        # ── Resume from checkpoint if requested ────────────────────────────
        if config.resume_from:
            self._resume(config.resume_from)

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns a history dict with 'train_losses' and 'val_losses' lists.
        """
        cfg = self.config
        self.log.section("Training Loop")
        self.log.info(f"Starting from iteration {self.iteration}")
        self.log.info(f"Total iterations: {cfg.max_iters}")
        self.log.info(f"Batch size: {cfg.batch_size} "
                      f"× {cfg.gradient_accumulation_steps} accumulation steps")

        t_start = time.time()
        t_iter  = time.time()

        self.model.train()

        while self.iteration < cfg.max_iters:
            # ── Gradient accumulation ──────────────────────────────────────
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(cfg.gradient_accumulation_steps):
                x, y = self._random_batch(self.train_data)
                _, loss = self.model(x, y)
                loss = loss / cfg.gradient_accumulation_steps
                loss.backward()
                accum_loss += loss.item()

            # ── Gradient clipping ──────────────────────────────────────────
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip
                )

            # ── Optimizer + LR scheduler step ─────────────────────────────
            lr = self.scheduler.step()
            self.optimizer.step()

            self.iteration += 1
            self.train_losses.append(accum_loss)

            # ── Per-step logging ───────────────────────────────────────────
            if self.iteration % cfg.log_interval == 0:
                dt_ms = 1000 * (time.time() - t_iter) / cfg.log_interval
                t_iter = time.time()
                self.log.training_step(
                    it=self.iteration,
                    total=cfg.max_iters,
                    metrics={"loss": accum_loss, "lr": lr},
                    dt_ms=dt_ms,
                )

            # ── Evaluation ────────────────────────────────────────────────
            if self.iteration % cfg.eval_interval == 0 or self.iteration == cfg.max_iters:
                train_loss = self._estimate_loss(self.train_data)
                val_loss   = self._estimate_loss(self.val_data)
                self.val_losses.append(val_loss)

                saved = False
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self._save(tag="best")
                    saved = True

                self.log.eval_checkpoint(
                    it=self.iteration,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    saved=saved,
                )
                self.model.train()

            # ── Periodic checkpoint ────────────────────────────────────────
            if cfg.checkpoint_interval > 0 and self.iteration % cfg.checkpoint_interval == 0:
                self._save(tag=f"iter{self.iteration:06d}")

        # ── End of training ────────────────────────────────────────────────
        elapsed = time.time() - t_start
        print()   # clear the last inline progress line
        self.log.success(
            f"Training complete in {elapsed/60:.1f} min  "
            f"| best val loss: {self.best_val:.4f}"
        )

        return {
            "train_losses": self.train_losses,
            "val_losses":   self.val_losses,
            "best_val":     self.best_val,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _random_batch(
        self, dataset: TextDataset
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (x, y) pairs from the dataset."""
        B   = self.config.batch_size
        T   = self.model_config.block_size
        n   = len(dataset.tokens)
        ix  = np.random.randint(0, n - T, size=(B,))
        x = torch.stack([
            torch.from_numpy(dataset.tokens[i : i + T].astype(np.int64)) for i in ix
        ]).to(self.device)
        y = torch.stack([
            torch.from_numpy(dataset.tokens[i + 1 : i + T + 1].astype(np.int64)) for i in ix
        ]).to(self.device)
        return x, y

    @torch.no_grad()
    def _estimate_loss(self, dataset: TextDataset) -> float:
        """Average loss over eval_iters random batches (no gradient)."""
        self.model.eval()
        losses = []
        for _ in range(self.config.eval_iters):
            x, y = self._random_batch(dataset)
            _, loss = self.model(x, y)
            losses.append(loss.item())
        self.model.train()
        return float(np.mean(losses))

    def _save(self, tag: str) -> None:
        """Save a labelled checkpoint to the checkpoint directory."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, f"ckpt_{tag}.pt")
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            config=self.model_config,
            iteration=self.iteration,
            val_loss=self.best_val,
        )

    def _resume(self, path: str) -> None:
        """Load a checkpoint and restore training state."""
        self.log.info(f"Resuming from {path}")
        ckpt = load_checkpoint(path, device=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.iteration = ckpt["iteration"]
        self.best_val  = ckpt["val_loss"]
        self.scheduler._step_count = self.iteration
        self.log.success(f"Resumed at iteration {self.iteration}, val_loss={self.best_val:.4f}")
