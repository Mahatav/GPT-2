"""
Tests for model/ — all components from embeddings up to GPT2.

All tests use a tiny config so they run in milliseconds on CPU.
"""

import math
import unittest
import torch
import torch.nn.functional as F

from config import GPT2Config
from model.embeddings import GPT2Embeddings
from model.attention import CausalSelfAttention
from model.mlp import MLP
from model.block import TransformerBlock
from model.gpt2 import GPT2


# ── Tiny config shared across all tests ───────────────────────────────────────
TINY = GPT2Config(
    vocab_size=16,
    block_size=8,
    n_layer=2,
    n_head=2,
    n_embd=16,
    dropout=0.0,
    bias=True,
)

B, T = 2, 6   # batch size, sequence length for tests


class TestGPT2Embeddings(unittest.TestCase):

    def setUp(self):
        self.embed = GPT2Embeddings(TINY)
        self.idx = torch.randint(0, TINY.vocab_size, (B, T))

    def test_output_shape(self):
        out = self.embed(self.idx)
        self.assertEqual(out.shape, (B, T, TINY.n_embd))

    def test_output_dtype_float(self):
        out = self.embed(self.idx)
        self.assertEqual(out.dtype, torch.float32)

    def test_sequence_too_long_raises(self):
        long_idx = torch.randint(0, TINY.vocab_size, (B, TINY.block_size + 1))
        with self.assertRaises(AssertionError):
            self.embed(long_idx)

    def test_token_and_position_differ(self):
        # Different positions should produce different embeddings
        idx1 = torch.zeros(1, 2, dtype=torch.long)
        out = self.embed(idx1)
        # pos 0 and pos 1 embeddings (same token) should differ due to pos embed
        self.assertFalse(torch.allclose(out[0, 0], out[0, 1]))


class TestCausalSelfAttention(unittest.TestCase):

    def setUp(self):
        self.attn = CausalSelfAttention(TINY)
        self.attn.eval()
        self.x = torch.randn(B, T, TINY.n_embd)

    def test_output_shape(self):
        out = self.attn(self.x)
        self.assertEqual(out.shape, (B, T, TINY.n_embd))

    def test_causal_mask_is_upper_triangular(self):
        # The buffer should be a boolean upper-triangular matrix
        mask = self.attn.causal_mask
        self.assertEqual(mask.dtype, torch.bool)
        for i in range(mask.size(0)):
            for j in range(mask.size(1)):
                if j > i:
                    self.assertTrue(mask[i, j].item(), f"mask[{i},{j}] should be True (masked)")
                else:
                    self.assertFalse(mask[i, j].item(), f"mask[{i},{j}] should be False (visible)")

    def test_causal_mask_prevents_future_info(self):
        """
        Zeroing out a future token's input should NOT change earlier positions' output.
        If the mask works correctly, position i never sees tokens j > i.
        """
        x1 = torch.randn(1, T, TINY.n_embd)
        x2 = x1.clone()
        x2[0, -1] = 0.0   # zero out the last token's input

        with torch.no_grad():
            out1 = self.attn(x1)
            out2 = self.attn(x2)

        # All positions except the last should be identical
        self.assertTrue(
            torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-5),
            "Earlier positions must not be affected by changes to future tokens",
        )

    def test_output_is_finite(self):
        out = self.attn(self.x)
        self.assertTrue(torch.isfinite(out).all())


class TestMLP(unittest.TestCase):

    def setUp(self):
        self.mlp = MLP(TINY)
        self.mlp.eval()
        self.x = torch.randn(B, T, TINY.n_embd)

    def test_output_shape(self):
        out = self.mlp(self.x)
        self.assertEqual(out.shape, (B, T, TINY.n_embd))

    def test_output_is_finite(self):
        out = self.mlp(self.x)
        self.assertTrue(torch.isfinite(out).all())

    def test_hidden_expansion_is_4x(self):
        # c_fc should expand to 4 * n_embd
        self.assertEqual(self.mlp.c_fc.out_features, 4 * TINY.n_embd)
        self.assertEqual(self.mlp.c_proj.in_features, 4 * TINY.n_embd)


class TestTransformerBlock(unittest.TestCase):

    def setUp(self):
        self.block = TransformerBlock(TINY)
        self.block.eval()
        self.x = torch.randn(B, T, TINY.n_embd)

    def test_output_shape(self):
        out = self.block(self.x)
        self.assertEqual(out.shape, (B, T, TINY.n_embd))

    def test_residual_connection_contributes(self):
        # Output should not equal the sub-layer alone (residual adds x)
        out = self.block(self.x)
        self.assertFalse(torch.allclose(out, self.x, atol=1e-4),
                         "Block output should differ from input due to sub-layers")

    def test_output_is_finite(self):
        out = self.block(self.x)
        self.assertTrue(torch.isfinite(out).all())


class TestGPT2(unittest.TestCase):

    def setUp(self):
        self.model = GPT2(TINY)
        self.model.eval()
        self.idx     = torch.randint(0, TINY.vocab_size, (B, T))
        self.targets = torch.randint(0, TINY.vocab_size, (B, T))

    # ── Shape tests ───────────────────────────────────────────────────────

    def test_forward_with_targets_loss_is_scalar(self):
        _, loss = self.model(self.idx, self.targets)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_forward_without_targets_returns_none_loss(self):
        logits, loss = self.model(self.idx)
        self.assertIsNone(loss)

    def test_logits_shape_with_targets(self):
        logits, _ = self.model(self.idx, self.targets)
        self.assertEqual(logits.shape, (B, T, TINY.vocab_size))

    def test_logits_shape_inference_last_token_only(self):
        # Without targets, only the last position is computed for efficiency
        logits, _ = self.model(self.idx)
        self.assertEqual(logits.shape, (B, 1, TINY.vocab_size))

    # ── Value tests ────────────────────────────────────────────────────────

    def test_loss_is_positive_finite(self):
        _, loss = self.model(self.idx, self.targets)
        self.assertTrue(loss.item() > 0)
        self.assertTrue(math.isfinite(loss.item()))

    def test_untrained_loss_near_log_vocab(self):
        # Random init → loss ≈ -log(1/vocab_size) = log(vocab_size)
        # Allow a generous tolerance since init can vary
        _, loss = self.model(self.idx, self.targets)
        expected = math.log(TINY.vocab_size)
        self.assertAlmostEqual(loss.item(), expected, delta=1.0)

    def test_logits_finite(self):
        logits, _ = self.model(self.idx, self.targets)
        self.assertTrue(torch.isfinite(logits).all())

    # ── Architecture tests ─────────────────────────────────────────────────

    def test_weight_tying(self):
        # lm_head.weight and embeddings.wte.weight must be the same tensor
        self.assertIs(
            self.model.lm_head.weight,
            self.model.embeddings.wte.weight,
            "Weight tying: lm_head.weight must be the same object as wte.weight",
        )

    def test_correct_number_of_blocks(self):
        self.assertEqual(len(self.model.blocks), TINY.n_layer)

    def test_num_parameters_positive(self):
        self.assertGreater(self.model.num_parameters(), 0)

    def test_sequence_length_too_long_raises(self):
        long_idx = torch.randint(0, TINY.vocab_size, (1, TINY.block_size + 1))
        with self.assertRaises(AssertionError):
            self.model(long_idx)

    # ── Gradient tests ─────────────────────────────────────────────────────

    def test_loss_is_differentiable(self):
        self.model.train()
        idx     = torch.randint(0, TINY.vocab_size, (B, T))
        targets = torch.randint(0, TINY.vocab_size, (B, T))
        _, loss = self.model(idx, targets)
        loss.backward()
        # Check that all parameters received gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")

    # ── Optimizer tests ────────────────────────────────────────────────────

    def test_configure_optimizer_returns_adamw(self):
        opt = self.model.configure_optimizer(lr=1e-3)
        self.assertIsInstance(opt, torch.optim.AdamW)

    def test_optimizer_has_two_param_groups(self):
        opt = self.model.configure_optimizer(lr=1e-3)
        self.assertEqual(len(opt.param_groups), 2)

    def test_optimizer_decay_group_has_weight_decay(self):
        opt = self.model.configure_optimizer(lr=1e-3, weight_decay=0.1)
        # First group has weight decay, second does not
        self.assertGreater(opt.param_groups[0]["weight_decay"], 0)
        self.assertEqual(opt.param_groups[1]["weight_decay"], 0.0)


if __name__ == "__main__":
    unittest.main()
