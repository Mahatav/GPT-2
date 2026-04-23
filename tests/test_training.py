"""Tests for training/ — TextDataset, CosineWarmupScheduler, Trainer."""

import math
import unittest
import numpy as np
import torch

from config import GPT2Config, TrainConfig
from model.gpt2 import GPT2
from tokenizer.char_tokenizer import CharTokenizer
from training.dataset import TextDataset, make_loader
from training.scheduler import CosineWarmupScheduler


TINY = GPT2Config(
    vocab_size=16,
    block_size=8,
    n_layer=2,
    n_head=2,
    n_embd=16,
    dropout=0.0,
)

# Enough tokens to form several training examples
TOKENS = np.arange(200, dtype=np.int64) % TINY.vocab_size
BLOCK  = TINY.block_size


# ── TextDataset ───────────────────────────────────────────────────────────────

class TestTextDataset(unittest.TestCase):

    def setUp(self):
        self.ds = TextDataset(TOKENS, block_size=BLOCK)

    def test_length(self):
        # Each start position up to len(tokens) - block_size is valid
        self.assertEqual(len(self.ds), len(TOKENS) - BLOCK)

    def test_item_shapes(self):
        x, y = self.ds[0]
        self.assertEqual(x.shape, (BLOCK,))
        self.assertEqual(y.shape, (BLOCK,))

    def test_target_is_shifted_by_one(self):
        x, y = self.ds[0]
        # y should be x shifted left by 1, then the next token
        self.assertTrue(torch.equal(x[1:], y[:-1]),
                        "y[:-1] should match x[1:] (target = input shifted by 1)")

    def test_item_dtype_is_int64(self):
        x, y = self.ds[0]
        self.assertEqual(x.dtype, torch.int64)
        self.assertEqual(y.dtype, torch.int64)

    def test_from_text_factory(self):
        text = "hello world this is a test corpus " * 10
        tok  = CharTokenizer(text)
        ds   = TextDataset.from_text(text, tok, block_size=8)
        self.assertIsInstance(ds, TextDataset)
        self.assertGreater(len(ds), 0)
        x, y = ds[0]
        self.assertEqual(x.shape, (8,))

    def test_train_val_split_sizes(self):
        train_ds, val_ds = self.ds.train_val_split(val_fraction=0.1)
        total = len(train_ds.tokens) + len(val_ds.tokens)
        self.assertEqual(total, len(TOKENS))
        self.assertGreater(len(train_ds.tokens), len(val_ds.tokens))

    def test_too_small_dataset_raises(self):
        tiny_tokens = np.arange(5, dtype=np.int64)
        with self.assertRaises(ValueError):
            TextDataset(tiny_tokens, block_size=8)

    def test_repr_contains_useful_info(self):
        r = repr(self.ds)
        self.assertIn("TextDataset", r)
        self.assertIn("block_size", r)

    def test_make_loader(self):
        loader = make_loader(self.ds, batch_size=4, shuffle=False)
        batch  = next(iter(loader))
        x, y   = batch
        self.assertEqual(x.shape[0], 4)   # batch dimension
        self.assertEqual(x.shape[1], BLOCK)


# ── CosineWarmupScheduler ─────────────────────────────────────────────────────

class TestCosineWarmupScheduler(unittest.TestCase):

    MAX_LR  = 3e-4
    MIN_LR  = 6e-5
    WARMUP  = 10
    MAX_IT  = 100

    def _make_scheduler(self):
        model = GPT2(TINY)
        opt   = model.configure_optimizer(lr=self.MAX_LR)
        sched = CosineWarmupScheduler(
            optimizer=opt,
            warmup_iters=self.WARMUP,
            max_iters=self.MAX_IT,
            max_lr=self.MAX_LR,
            min_lr=self.MIN_LR,
        )
        return sched, opt

    def test_lr_increases_during_warmup(self):
        sched, _ = self._make_scheduler()
        lrs = [sched.step() for _ in range(self.WARMUP)]
        for i in range(len(lrs) - 1):
            self.assertLessEqual(lrs[i], lrs[i + 1],
                                 "LR should monotonically increase during warmup")

    def test_lr_at_peak_is_max_lr(self):
        sched, _ = self._make_scheduler()
        for _ in range(self.WARMUP):
            lr = sched.step()
        self.assertAlmostEqual(lr, self.MAX_LR, places=6)

    def test_lr_decreases_after_warmup(self):
        sched, _ = self._make_scheduler()
        for _ in range(self.WARMUP):
            sched.step()
        lrs = [sched.step() for _ in range(self.MAX_IT - self.WARMUP)]
        for i in range(len(lrs) - 1):
            self.assertGreaterEqual(lrs[i], lrs[i + 1],
                                    "LR should monotonically decrease after warmup")

    def test_lr_floor_is_min_lr(self):
        sched, _ = self._make_scheduler()
        for _ in range(self.MAX_IT + 10):   # go past max_iters
            lr = sched.step()
        self.assertAlmostEqual(lr, self.MIN_LR, places=6)

    def test_state_dict_roundtrip(self):
        sched, _ = self._make_scheduler()
        for _ in range(30):
            sched.step()
        state = sched.state_dict()
        self.assertEqual(state["step_count"], 30)

        # Restore into a fresh scheduler and check current lr matches
        sched2, _ = self._make_scheduler()
        sched2.load_state_dict(state)
        self.assertAlmostEqual(sched2.current_lr, sched.current_lr, places=6)

    def test_warmup_iters_must_be_less_than_max(self):
        model = GPT2(TINY)
        opt   = model.configure_optimizer(lr=1e-3)
        with self.assertRaises(ValueError):
            CosineWarmupScheduler(opt, warmup_iters=100, max_iters=50,
                                  max_lr=1e-3, min_lr=1e-5)

    def test_get_lr_formula_warmup(self):
        sched, _ = self._make_scheduler()
        for t in range(1, self.WARMUP + 1):
            expected = self.MAX_LR * t / self.WARMUP
            actual   = sched.get_lr(t)
            self.assertAlmostEqual(actual, expected, places=6)

    def test_get_lr_formula_cosine(self):
        sched, _ = self._make_scheduler()
        # At the midpoint of decay
        mid = (self.WARMUP + self.MAX_IT) // 2
        ratio = (mid - self.WARMUP) / (self.MAX_IT - self.WARMUP)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        expected = self.MIN_LR + coeff * (self.MAX_LR - self.MIN_LR)
        actual   = sched.get_lr(mid)
        self.assertAlmostEqual(actual, expected, places=6)


# ── Trainer integration test ──────────────────────────────────────────────────

class TestTrainerIntegration(unittest.TestCase):
    """Very short training run — just verifies loss goes down, not a full run."""

    def test_loss_decreases_over_training(self):
        from training.trainer import Trainer

        text = ("abcdefghijklmnop" * 200)   # 16-char vocab, plenty of tokens
        tok  = CharTokenizer(text)

        tiny_cfg = GPT2Config(
            vocab_size=tok.vocab_size,
            block_size=8,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
        train_cfg = TrainConfig(
            batch_size=8,
            max_iters=50,
            learning_rate=1e-3,
            warmup_iters=5,
            lr_decay_iters=50,
            eval_interval=50,
            eval_iters=10,
            log_interval=100,
            checkpoint_interval=0,
        )

        device    = torch.device("cpu")
        model     = GPT2(tiny_cfg).to(device)
        tokens    = np.array(tok.encode(text), dtype=np.int64)
        train_ds  = TextDataset(tokens[:int(0.9 * len(tokens))], block_size=8)
        val_ds    = TextDataset(tokens[int(0.9 * len(tokens)):], block_size=8)

        trainer = Trainer(model, train_ds, val_ds, train_cfg, device)
        history = trainer.train()

        # After training the val loss list should be non-empty
        self.assertGreater(len(history["val_losses"]), 0)
        # Final val loss should be less than random init ≈ log(vocab_size)
        expected_random = math.log(tok.vocab_size)
        self.assertLess(history["best_val"], expected_random)


if __name__ == "__main__":
    unittest.main()
