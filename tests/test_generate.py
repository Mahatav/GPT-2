"""Tests for generate.py — Generator sampling strategies."""

import unittest
import torch
import torch.nn.functional as F

from config import GPT2Config
from model.gpt2 import GPT2
from tokenizer.char_tokenizer import CharTokenizer
from generate import Generator


TINY = GPT2Config(
    vocab_size=16,
    block_size=8,
    n_layer=2,
    n_head=2,
    n_embd=16,
    dropout=0.0,
)

CORPUS = "abcdefghijklmnop" * 50   # exactly 16 unique chars → vocab_size=16


class TestGenerator(unittest.TestCase):

    def setUp(self):
        self.tok    = CharTokenizer(CORPUS)
        self.device = torch.device("cpu")

        cfg = GPT2Config(
            vocab_size=self.tok.vocab_size,
            block_size=TINY.block_size,
            n_layer=TINY.n_layer,
            n_head=TINY.n_head,
            n_embd=TINY.n_embd,
            dropout=0.0,
        )
        self.model = GPT2(cfg).to(self.device)
        self.gen   = Generator(self.model, self.tok, self.device)
        self.prompt = "abcd"

    # ── Return type tests ──────────────────────────────────────────────────

    def test_greedy_returns_string(self):
        out = self.gen.greedy(self.prompt, max_new_tokens=5)
        self.assertIsInstance(out, str)

    def test_top_k_returns_string(self):
        out = self.gen.top_k(self.prompt, k=5, max_new_tokens=5)
        self.assertIsInstance(out, str)

    def test_top_p_returns_string(self):
        out = self.gen.top_p(self.prompt, p=0.9, max_new_tokens=5)
        self.assertIsInstance(out, str)

    def test_sample_returns_string(self):
        out = self.gen.sample(self.prompt, max_new_tokens=5, temperature=1.0)
        self.assertIsInstance(out, str)

    # ── Length tests ───────────────────────────────────────────────────────

    def test_greedy_output_longer_than_prompt(self):
        out = self.gen.greedy(self.prompt, max_new_tokens=10)
        self.assertGreater(len(out), len(self.prompt))

    def test_output_length_matches_new_tokens(self):
        n = 8
        out = self.gen.greedy(self.prompt, max_new_tokens=n)
        # char-level: output length = prompt length + n new tokens
        self.assertEqual(len(out), len(self.prompt) + n)

    def test_prompt_preserved_at_start(self):
        out = self.gen.greedy(self.prompt, max_new_tokens=5)
        self.assertTrue(out.startswith(self.prompt),
                        f"Output {repr(out)} should start with prompt {repr(self.prompt)}")

    # ── Determinism tests ─────────────────────────────────────────────────

    def test_greedy_is_deterministic(self):
        out1 = self.gen.greedy(self.prompt, max_new_tokens=10)
        out2 = self.gen.greedy(self.prompt, max_new_tokens=10)
        self.assertEqual(out1, out2, "Greedy should be fully deterministic")

    # ── Filtering function tests ──────────────────────────────────────────

    def test_top_k_filter_keeps_k_tokens(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        k = 3
        filtered = Generator._apply_top_k(logits, k)
        # Count non-inf values
        non_inf = (filtered != float("-inf")).sum().item()
        self.assertEqual(non_inf, k)

    def test_top_k_filter_keeps_highest_values(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        filtered = Generator._apply_top_k(logits, k=3)
        # Top 3 are indices 2, 3, 4 (values 3, 4, 5)
        self.assertTrue(torch.isfinite(filtered[0, 2]))
        self.assertTrue(torch.isfinite(filtered[0, 3]))
        self.assertTrue(torch.isfinite(filtered[0, 4]))
        self.assertEqual(filtered[0, 0].item(), float("-inf"))
        self.assertEqual(filtered[0, 1].item(), float("-inf"))

    def test_top_p_filter_probs_sum_geq_p(self):
        logits = torch.randn(1, 20)
        p = 0.8
        filtered = Generator._apply_top_p(logits, p)
        probs = F.softmax(filtered, dim=-1)
        total = probs[probs > 0].sum().item()
        self.assertGreaterEqual(total, p - 1e-5,
                                "Remaining probability mass must be at least p")

    def test_top_p_with_p_1_keeps_all_tokens(self):
        logits = torch.randn(1, 10)
        filtered = Generator._apply_top_p(logits, p=1.0)
        # With p=1.0 everything passes — no -inf values
        self.assertTrue(torch.isfinite(filtered).all())

    def test_top_k_with_k_equals_vocab_keeps_all(self):
        logits = torch.randn(1, 16)
        filtered = Generator._apply_top_k(logits, k=16)
        self.assertTrue(torch.isfinite(filtered).all())

    # ── Context cropping test ─────────────────────────────────────────────

    def test_long_context_is_cropped(self):
        # Feed a prompt longer than block_size — should not crash
        long_prompt = self.prompt * 10   # way longer than block_size=8
        out = self.gen.greedy(long_prompt, max_new_tokens=3)
        self.assertIsInstance(out, str)


if __name__ == "__main__":
    unittest.main()
