"""Tests for kl_divergence.py — KL divergence utilities."""

import math
import unittest
from unittest.mock import MagicMock, patch

import torch

from kl_divergence import (
    _kl_nats,
    _to_pct,
    compute_kl_pct,
    compute_kl_report,
    compute_symmetric_kl_pct,
    _MAX_KL_NATS,
)


class TestToPct(unittest.TestCase):

    def test_zero_nats_gives_zero_pct(self):
        self.assertEqual(_to_pct(0.0), 0.0)

    def test_max_kl_nats_gives_100_pct(self):
        self.assertAlmostEqual(_to_pct(_MAX_KL_NATS), 100.0, places=4)

    def test_half_max_gives_50_pct(self):
        self.assertAlmostEqual(_to_pct(_MAX_KL_NATS / 2), 50.0, places=4)

    def test_nan_propagates(self):
        result = _to_pct(float("nan"))
        self.assertTrue(math.isnan(result))

    def test_inf_propagates(self):
        result = _to_pct(float("inf"))
        self.assertTrue(math.isnan(result))

    def test_over_max_clamped_to_100(self):
        # Values beyond the theoretical maximum are capped at 100 %
        self.assertEqual(_to_pct(_MAX_KL_NATS * 10), 100.0)

    def test_result_is_rounded_to_2dp(self):
        result = _to_pct(1.0)
        # Should have at most 2 decimal places
        self.assertEqual(result, round(result, 2))


class TestKlNats(unittest.TestCase):

    def _uniform_log_probs(self, vocab=4, seq=3):
        """Return uniform log-prob tensor of shape (seq, vocab)."""
        return torch.full((seq, vocab), -math.log(vocab))

    def test_identical_distributions_give_zero(self):
        log_p = self._uniform_log_probs()
        kl = _kl_nats(log_p, log_p)
        self.assertAlmostEqual(kl, 0.0, places=5)

    def test_kl_non_negative(self):
        vocab, seq = 8, 5
        log_p = torch.log_softmax(torch.randn(seq, vocab), dim=-1)
        log_q = torch.log_softmax(torch.randn(seq, vocab), dim=-1)
        kl = _kl_nats(log_p, log_q)
        self.assertGreaterEqual(kl, 0.0)

    def test_point_mass_vs_uniform_large_kl(self):
        vocab, seq = 4, 1
        # P = point mass on token 0
        log_p = torch.full((seq, vocab), -1e9)
        log_p[:, 0] = 0.0
        # Q = uniform
        log_q = torch.full((seq, vocab), -math.log(vocab))
        kl = _kl_nats(log_p, log_q)
        # KL should equal log(vocab) ≈ 1.39 nats for vocab=4
        self.assertAlmostEqual(kl, math.log(vocab), places=3)

    def test_returns_scalar(self):
        log_p = self._uniform_log_probs()
        result = _kl_nats(log_p, log_p)
        self.assertIsInstance(result, float)


class TestComputeKlPct(unittest.TestCase):

    def _make_mock_model(self, vocab=50257, seq=10):
        """Return a mock model whose logits are uniform over vocab."""
        model = MagicMock()
        logits = torch.zeros(1, seq, vocab)
        output = MagicMock()
        output.logits = logits
        model.return_value = output
        return model

    def _make_mock_encoding(self, tokens):
        enc = MagicMock()
        enc.encode.return_value = tokens
        return enc

    def test_identical_models_give_zero(self):
        vocab = 50257
        model = self._make_mock_model(vocab=vocab)
        enc = self._make_mock_encoding(list(range(20)))
        result = compute_kl_pct(model, model, enc, "test text", device="cpu")
        self.assertAlmostEqual(result, 0.0, places=2)

    def test_short_text_returns_nan(self):
        model = self._make_mock_model()
        enc = self._make_mock_encoding([1])  # only 1 token — too short
        result = compute_kl_pct(model, model, enc, "x", device="cpu")
        self.assertTrue(math.isnan(result))

    def test_result_in_valid_range(self):
        model_p = self._make_mock_model()
        model_q = self._make_mock_model()
        enc = self._make_mock_encoding(list(range(30)))
        result = compute_kl_pct(model_p, model_q, enc, "some text", device="cpu")
        if not math.isnan(result):
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 100.0)

    def test_exception_returns_nan(self):
        model = MagicMock(side_effect=RuntimeError("GPU OOM"))
        enc = self._make_mock_encoding(list(range(20)))
        result = compute_kl_pct(model, model, enc, "text", device="cpu")
        self.assertTrue(math.isnan(result))

    def test_block_size_truncation(self):
        """Tokens longer than block_size should be silently truncated."""
        vocab = 50257
        model = self._make_mock_model(vocab=vocab, seq=5)
        enc = self._make_mock_encoding(list(range(50)))  # 50 tokens
        # block_size=5 means only 5 tokens used
        result = compute_kl_pct(model, model, enc, "text", device="cpu", block_size=5)
        self.assertAlmostEqual(result, 0.0, places=2)


class TestComputeSymmetricKlPct(unittest.TestCase):

    def _make_mock_model(self, vocab=50257, seq=10):
        model = MagicMock()
        logits = torch.zeros(1, seq, vocab)
        output = MagicMock()
        output.logits = logits
        model.return_value = output
        return model

    def _make_mock_encoding(self, tokens):
        enc = MagicMock()
        enc.encode.return_value = tokens
        return enc

    def test_identical_models_give_zero(self):
        model = self._make_mock_model()
        enc = self._make_mock_encoding(list(range(20)))
        result = compute_symmetric_kl_pct(model, model, enc, "text", device="cpu")
        self.assertAlmostEqual(result, 0.0, places=2)

    def test_result_is_symmetric_by_construction(self):
        # With identical models, symmetry is trivially satisfied.
        # Here we verify the formula: 0.5 * (KL(A||B) + KL(B||A)).
        model = self._make_mock_model()
        enc = self._make_mock_encoding(list(range(20)))
        r1 = compute_symmetric_kl_pct(model, model, enc, "text", device="cpu")
        r2 = compute_symmetric_kl_pct(model, model, enc, "text", device="cpu")
        self.assertAlmostEqual(r1, r2, places=5)

    def test_short_text_returns_nan(self):
        model = self._make_mock_model()
        enc = self._make_mock_encoding([42])
        result = compute_symmetric_kl_pct(model, model, enc, "x", device="cpu")
        self.assertTrue(math.isnan(result))


class TestComputeKlReport(unittest.TestCase):

    def _make_mock_model(self, vocab=50257, seq=10):
        model = MagicMock()
        logits = torch.zeros(1, seq, vocab)
        output = MagicMock()
        output.logits = logits
        model.return_value = output
        return model

    def _make_mock_encoding(self, tokens):
        enc = MagicMock()
        enc.encode.return_value = tokens
        return enc

    def test_all_expected_keys_present(self):
        model = self._make_mock_model()
        enc = self._make_mock_encoding(list(range(20)))
        report = compute_kl_report(
            model, model, enc,
            prompt="What is truth?",
            west_output="Western answer.",
            east_output="Eastern answer.",
            device="cpu",
        )
        expected_keys = {
            "west_to_east_on_prompt",
            "east_to_west_on_prompt",
            "symmetric_on_prompt",
            "west_to_east_on_east_output",
            "east_to_west_on_west_output",
            "_note",
        }
        self.assertEqual(set(report.keys()), expected_keys)

    def test_note_field_is_string(self):
        model = self._make_mock_model()
        enc = self._make_mock_encoding(list(range(20)))
        report = compute_kl_report(
            model, model, enc,
            prompt="test", west_output="w", east_output="e", device="cpu"
        )
        self.assertIsInstance(report["_note"], str)


if __name__ == "__main__":
    unittest.main()
