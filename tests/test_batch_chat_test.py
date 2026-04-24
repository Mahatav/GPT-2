"""Tests for batch_chat_test.py — batch prompt evaluation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from batch_chat_test import PHILOSOPHICAL_PROMPTS, run_batch_chat_test


def _make_mock_model():
    """Return a minimal mock model compatible with inference_utils.generate_with_history."""
    model = MagicMock()
    param = MagicMock()
    import torch
    param.device = torch.device("cpu")
    model.parameters.return_value = iter([param])
    return model


def _make_mock_encoding():
    """Return a mock tiktoken encoding."""
    import torch
    enc = MagicMock()
    enc.encode.return_value = list(range(20))
    enc.decode.return_value = "Generated philosophical answer."
    enc.eot_token = 50256
    return enc


class TestPhilosophicalPrompts(unittest.TestCase):

    def test_prompts_non_empty(self):
        self.assertGreater(len(PHILOSOPHICAL_PROMPTS), 0)

    def test_all_prompts_are_strings(self):
        for p in PHILOSOPHICAL_PROMPTS:
            self.assertIsInstance(p, str)

    def test_all_prompts_end_with_question_mark(self):
        for p in PHILOSOPHICAL_PROMPTS:
            self.assertTrue(p.strip().endswith("?"), f"Not a question: {p!r}")

    def test_no_duplicate_prompts(self):
        self.assertEqual(len(PHILOSOPHICAL_PROMPTS), len(set(PHILOSOPHICAL_PROMPTS)))


class TestRunBatchChatTest(unittest.TestCase):

    @patch("batch_chat_test.load_model")
    @patch("batch_chat_test.get_tokenizer")
    @patch("batch_chat_test.generate_with_history")
    def test_creates_output_file(self, mock_gen, mock_tok, mock_load):
        mock_load.return_value = _make_mock_model()
        mock_tok.return_value = _make_mock_encoding()
        mock_gen.return_value = ("A wise answer.", "accumulated history")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "results.json")
            run_batch_chat_test(
                checkpoint_path="/fake/checkpoint",
                output_path=output_path,
            )
            self.assertTrue(Path(output_path).exists())

    @patch("batch_chat_test.load_model")
    @patch("batch_chat_test.get_tokenizer")
    @patch("batch_chat_test.generate_with_history")
    def test_output_json_structure(self, mock_gen, mock_tok, mock_load):
        mock_load.return_value = _make_mock_model()
        mock_tok.return_value = _make_mock_encoding()
        mock_gen.return_value = ("Wisdom.", "history")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "results.json")
            results = run_batch_chat_test("/fake/ckpt", output_path)

        for key in ("timestamp", "checkpoint", "device", "config", "responses"):
            self.assertIn(key, results)

    @patch("batch_chat_test.load_model")
    @patch("batch_chat_test.get_tokenizer")
    @patch("batch_chat_test.generate_with_history")
    def test_response_count_matches_prompts(self, mock_gen, mock_tok, mock_load):
        mock_load.return_value = _make_mock_model()
        mock_tok.return_value = _make_mock_encoding()
        mock_gen.return_value = ("Answer.", "history text")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "results.json")
            results = run_batch_chat_test("/fake/ckpt", output_path)

        self.assertEqual(len(results["responses"]), len(PHILOSOPHICAL_PROMPTS))

    @patch("batch_chat_test.load_model")
    @patch("batch_chat_test.get_tokenizer")
    @patch("batch_chat_test.generate_with_history")
    def test_each_response_has_required_keys(self, mock_gen, mock_tok, mock_load):
        mock_load.return_value = _make_mock_model()
        mock_tok.return_value = _make_mock_encoding()
        mock_gen.return_value = ("Deep thought.", "history")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "results.json")
            results = run_batch_chat_test("/fake/ckpt", output_path)

        for resp in results["responses"]:
            self.assertIn("prompt", resp)
            self.assertIn("response", resp)
            self.assertIn("response_length", resp)

    @patch("batch_chat_test.load_model")
    @patch("batch_chat_test.get_tokenizer")
    @patch("batch_chat_test.generate_with_history")
    def test_response_length_matches_actual_response(self, mock_gen, mock_tok, mock_load):
        mock_load.return_value = _make_mock_model()
        mock_tok.return_value = _make_mock_encoding()
        reply = "This is exactly twenty-three chars."
        mock_gen.return_value = (reply, "history")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "r.json")
            results = run_batch_chat_test("/fake/ckpt", output_path)

        for resp in results["responses"]:
            self.assertEqual(resp["response_length"], len(resp["response"]))

    @patch("batch_chat_test.load_model")
    @patch("batch_chat_test.get_tokenizer")
    @patch("batch_chat_test.generate_with_history")
    def test_generation_error_recorded_not_raised(self, mock_gen, mock_tok, mock_load):
        mock_load.return_value = _make_mock_model()
        mock_tok.return_value = _make_mock_encoding()
        mock_gen.side_effect = RuntimeError("GPU crashed")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "r.json")
            # Should not raise
            results = run_batch_chat_test("/fake/ckpt", output_path)

        error_responses = [r for r in results["responses"] if "[ERROR:" in r["response"]]
        self.assertEqual(len(error_responses), len(PHILOSOPHICAL_PROMPTS))

    @patch("batch_chat_test.load_model")
    @patch("batch_chat_test.get_tokenizer")
    @patch("batch_chat_test.generate_with_history")
    def test_output_file_is_valid_json(self, mock_gen, mock_tok, mock_load):
        mock_load.return_value = _make_mock_model()
        mock_tok.return_value = _make_mock_encoding()
        mock_gen.return_value = ("Answer.", "history")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "r.json")
            run_batch_chat_test("/fake/ckpt", output_path)
            with open(output_path) as f:
                data = json.load(f)
            self.assertIn("responses", data)

    @patch("batch_chat_test.load_model")
    @patch("batch_chat_test.get_tokenizer")
    def test_model_load_failure_raises(self, mock_tok, mock_load):
        mock_load.side_effect = FileNotFoundError("Checkpoint not found")
        mock_tok.return_value = _make_mock_encoding()

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                run_batch_chat_test("/bad/path", str(Path(tmpdir) / "r.json"))


if __name__ == "__main__":
    unittest.main()
