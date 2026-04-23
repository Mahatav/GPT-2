"""Tests for utils/checkpoint.py — save, load, and resume."""

import os
import tempfile
import unittest
import torch

from config import GPT2Config, TrainConfig
from model.gpt2 import GPT2
from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    resume_model,
    list_checkpoints,
    latest_checkpoint,
)


TINY = GPT2Config(
    vocab_size=16,
    block_size=8,
    n_layer=2,
    n_head=2,
    n_embd=16,
    dropout=0.0,
)


class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        self.model = GPT2(TINY)
        self.opt   = self.model.configure_optimizer(lr=1e-3)
        self.tmpdir = tempfile.mkdtemp()
        self.ckpt_path = os.path.join(self.tmpdir, "test_ckpt.pt")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ── save_checkpoint ────────────────────────────────────────────────────

    def test_save_creates_file(self):
        save_checkpoint(
            path=self.ckpt_path,
            model=self.model,
            optimizer=self.opt,
            config=TINY,
            iteration=42,
            val_loss=1.234,
        )
        self.assertTrue(os.path.exists(self.ckpt_path))

    def test_saved_file_is_nonzero(self):
        save_checkpoint(self.ckpt_path, self.model, self.opt, TINY, 10, 2.0)
        self.assertGreater(os.path.getsize(self.ckpt_path), 0)

    # ── load_checkpoint ────────────────────────────────────────────────────

    def test_load_returns_expected_keys(self):
        save_checkpoint(self.ckpt_path, self.model, self.opt, TINY, 10, 1.5)
        ckpt = load_checkpoint(self.ckpt_path)
        for key in ("iteration", "val_loss", "model_state", "optim_state", "config"):
            self.assertIn(key, ckpt, f"Key '{key}' missing from checkpoint")

    def test_load_iteration_matches_saved(self):
        save_checkpoint(self.ckpt_path, self.model, self.opt, TINY, 99, 0.5)
        ckpt = load_checkpoint(self.ckpt_path)
        self.assertEqual(ckpt["iteration"], 99)

    def test_load_val_loss_matches_saved(self):
        save_checkpoint(self.ckpt_path, self.model, self.opt, TINY, 10, 0.987)
        ckpt = load_checkpoint(self.ckpt_path)
        self.assertAlmostEqual(ckpt["val_loss"], 0.987, places=4)

    def test_load_config_matches_saved(self):
        save_checkpoint(self.ckpt_path, self.model, self.opt, TINY, 10, 1.0)
        ckpt = load_checkpoint(self.ckpt_path)
        self.assertEqual(ckpt["config"]["n_layer"], TINY.n_layer)
        self.assertEqual(ckpt["config"]["vocab_size"], TINY.vocab_size)

    def test_load_without_optimizer(self):
        save_checkpoint(self.ckpt_path, self.model, self.opt, TINY, 5, 1.0)
        ckpt = load_checkpoint(self.ckpt_path, load_optimizer=False)
        self.assertNotIn("optim_state", ckpt)

    def test_load_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_checkpoint("/nonexistent/path/ckpt.pt")

    # ── resume_model ───────────────────────────────────────────────────────

    def test_resume_model_reconstructs_architecture(self):
        save_checkpoint(self.ckpt_path, self.model, self.opt, TINY, 77, 0.9)
        restored, cfg, it, loss = resume_model(self.ckpt_path)
        self.assertIsInstance(restored, GPT2)
        self.assertEqual(cfg.n_layer, TINY.n_layer)
        self.assertEqual(it, 77)
        self.assertAlmostEqual(loss, 0.9, places=4)

    def test_resume_model_weights_match(self):
        save_checkpoint(self.ckpt_path, self.model, self.opt, TINY, 1, 1.0)
        restored, _, _, _ = resume_model(self.ckpt_path)

        orig_params    = {n: p for n, p in self.model.named_parameters()}
        restore_params = {n: p for n, p in restored.named_parameters()}

        for name in orig_params:
            self.assertTrue(
                torch.allclose(orig_params[name], restore_params[name]),
                f"Parameter {name} differs after resume",
            )

    # ── list_checkpoints / latest_checkpoint ──────────────────────────────

    def test_list_checkpoints_empty_dir(self):
        ckpts = list_checkpoints(self.tmpdir)
        self.assertEqual(ckpts, [])

    def test_list_checkpoints_finds_pt_files(self):
        for i in range(3):
            save_checkpoint(
                os.path.join(self.tmpdir, f"ckpt_{i}.pt"),
                self.model, self.opt, TINY, i, float(i),
            )
        ckpts = list_checkpoints(self.tmpdir)
        self.assertEqual(len(ckpts), 3)

    def test_latest_checkpoint_returns_newest(self):
        import time
        for i in range(3):
            path = os.path.join(self.tmpdir, f"ckpt_{i}.pt")
            save_checkpoint(path, self.model, self.opt, TINY, i, float(i))
            time.sleep(0.01)   # ensure distinct modification times
        latest = latest_checkpoint(self.tmpdir)
        self.assertIn("ckpt_2.pt", latest)

    def test_latest_checkpoint_nonexistent_dir_returns_none(self):
        result = latest_checkpoint("/this/does/not/exist")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
