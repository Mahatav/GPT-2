"""Tests for config.py — GPT2Config and TrainConfig."""

import unittest
from config import (
    GPT2Config, TrainConfig,
    GPT2_NANO, GPT2_MICRO, GPT2_SMALL, GPT2_MEDIUM, GPT2_LARGE, GPT2_XL,
    MODEL_PRESETS,
)


class TestGPT2Config(unittest.TestCase):

    def test_default_construction(self):
        cfg = GPT2Config()
        self.assertEqual(cfg.vocab_size, 50257)
        self.assertEqual(cfg.block_size, 1024)
        self.assertEqual(cfg.n_layer, 12)
        self.assertEqual(cfg.n_head, 12)
        self.assertEqual(cfg.n_embd, 768)
        self.assertAlmostEqual(cfg.dropout, 0.1)
        self.assertTrue(cfg.bias)

    def test_head_dim_property(self):
        cfg = GPT2Config(n_embd=768, n_head=12)
        self.assertEqual(cfg.head_dim, 64)

        cfg2 = GPT2Config(n_embd=384, n_head=6)
        self.assertEqual(cfg2.head_dim, 64)

    def test_invalid_embd_head_ratio_raises(self):
        with self.assertRaises(ValueError):
            GPT2Config(n_embd=100, n_head=7)  # 100 is not divisible by 7

    def test_n_params_estimate_positive(self):
        cfg = GPT2Config()
        self.assertGreater(cfg.n_params_estimate, 0)

    def test_custom_config(self):
        cfg = GPT2Config(vocab_size=256, block_size=128, n_layer=4, n_head=4, n_embd=128)
        self.assertEqual(cfg.vocab_size, 256)
        self.assertEqual(cfg.head_dim, 32)


class TestModelPresets(unittest.TestCase):

    def test_all_presets_valid(self):
        for name, preset in MODEL_PRESETS.items():
            with self.subTest(preset=name):
                self.assertIsInstance(preset, GPT2Config)
                self.assertEqual(preset.n_embd % preset.n_head, 0,
                                 f"{name}: n_embd must divide evenly by n_head")

    def test_preset_param_ordering(self):
        # Larger presets should have more parameters
        self.assertLess(GPT2_NANO.n_params_estimate, GPT2_MICRO.n_params_estimate)
        self.assertLess(GPT2_SMALL.n_params_estimate, GPT2_MEDIUM.n_params_estimate)
        self.assertLess(GPT2_MEDIUM.n_params_estimate, GPT2_LARGE.n_params_estimate)
        self.assertLess(GPT2_LARGE.n_params_estimate, GPT2_XL.n_params_estimate)

    def test_model_presets_dict_complete(self):
        expected = {"nano", "micro", "small", "medium", "large", "xl"}
        self.assertEqual(set(MODEL_PRESETS.keys()), expected)


class TestTrainConfig(unittest.TestCase):

    def test_default_construction(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.batch_size, 16)
        self.assertGreater(cfg.max_iters, 0)
        self.assertGreater(cfg.learning_rate, 0)
        self.assertGreater(cfg.warmup_iters, 0)
        self.assertLess(cfg.min_lr, cfg.learning_rate)
        self.assertIsNone(cfg.resume_from)

    def test_custom_values(self):
        cfg = TrainConfig(batch_size=32, max_iters=1000, learning_rate=1e-3)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.max_iters, 1000)
        self.assertAlmostEqual(cfg.learning_rate, 1e-3)


if __name__ == "__main__":
    unittest.main()
