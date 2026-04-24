"""Tests for inference_utils.py — model loading, generation, history management."""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import torch

from inference_utils import generate, generate_with_history


def _make_mock_model(output_token_ids):
    """Return a mock GPT2LMHeadModel that generates the given output token IDs."""
    model = MagicMock()
    # model.generate returns shape (1, prompt_len + new_tokens)
    model.generate.return_value = torch.tensor([output_token_ids])
    # next(model.parameters()).device should return cpu
    param = MagicMock()
    param.device = torch.device("cpu")
    model.parameters.return_value = iter([param])
    return model


def _make_mock_encoding(encode_result, decode_result="decoded text"):
    """Return a mock tiktoken encoding."""
    enc = MagicMock()
    enc.encode.return_value = encode_result
    enc.decode.return_value = decode_result
    enc.eot_token = 50256
    return enc


class TestGenerate(unittest.TestCase):

    def test_returns_string(self):
        prompt_ids = [1, 2, 3]
        output_ids = prompt_ids + [4, 5]
        model = _make_mock_model(output_ids)
        enc = _make_mock_encoding(prompt_ids, decode_result="hello")
        result = generate(model, enc, "test prompt", device="cpu")
        self.assertIsInstance(result, str)

    def test_decodes_only_new_tokens(self):
        prompt_ids = [10, 20, 30]
        new_ids = [40, 50]
        output_ids = prompt_ids + new_ids
        model = _make_mock_model(output_ids)
        enc = _make_mock_encoding(prompt_ids, decode_result="new content")
        generate(model, enc, "prompt", device="cpu")
        # decode should be called with only the new tokens
        enc.decode.assert_called_once_with(new_ids)

    def test_passes_generation_kwargs_to_model(self):
        prompt_ids = [1, 2]
        model = _make_mock_model(prompt_ids + [3])
        enc = _make_mock_encoding(prompt_ids)
        generate(
            model, enc, "prompt",
            max_new_tokens=42,
            temperature=0.7,
            top_p=0.8,
            device="cpu",
        )
        call_kwargs = model.generate.call_args[1]
        self.assertEqual(call_kwargs["max_new_tokens"], 42)
        self.assertAlmostEqual(call_kwargs["temperature"], 0.7)
        self.assertAlmostEqual(call_kwargs["top_p"], 0.8)

    def test_uses_provided_device(self):
        prompt_ids = [1, 2, 3]
        model = _make_mock_model(prompt_ids + [4])
        enc = _make_mock_encoding(prompt_ids)
        generate(model, enc, "text", device="cpu")
        # Check that the input tensor was sent to cpu
        call_args = model.generate.call_args[0][0]
        self.assertEqual(call_args.device.type, "cpu")

    def test_no_repetition_penalty_by_default(self):
        """Default repetition_penalty should be 1.0 (no penalty)."""
        prompt_ids = [1, 2]
        model = _make_mock_model(prompt_ids + [3])
        enc = _make_mock_encoding(prompt_ids)
        generate(model, enc, "prompt", device="cpu")
        kwargs = model.generate.call_args[1]
        self.assertAlmostEqual(kwargs.get("repetition_penalty", 1.0), 1.0)


class TestGenerateWithHistory(unittest.TestCase):

    def _mock_setup(self, prompt_reply="The answer is wisdom."):
        prompt_ids = list(range(10))
        # Output includes extra tokens; decode returns the reply
        model = _make_mock_model(prompt_ids + [99, 100])
        enc = _make_mock_encoding(prompt_ids, decode_result=prompt_reply)
        return model, enc

    def test_returns_tuple_of_two_strings(self):
        model, enc = self._mock_setup()
        result = generate_with_history(model, enc, "What is wisdom?", device="cpu")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        reply, history = result
        self.assertIsInstance(reply, str)
        self.assertIsInstance(history, str)

    def test_history_contains_prompt(self):
        model, enc = self._mock_setup(prompt_reply="insight")
        _, history = generate_with_history(
            model, enc, "What is truth?", history="", device="cpu"
        )
        self.assertIn("What is truth?", history)

    def test_history_contains_reply(self):
        model, enc = self._mock_setup(prompt_reply="insight")
        _, history = generate_with_history(
            model, enc, "Question?", history="", device="cpu"
        )
        # The reply (before any "Human:" split) should appear in the history
        self.assertIn("insight", history)

    def test_prior_history_preserved_in_new_history(self):
        model, enc = self._mock_setup(prompt_reply="reply text")
        prior = "Human: prior question\nAssistant: prior answer\n"
        _, history = generate_with_history(
            model, enc, "New question?", history=prior, device="cpu"
        )
        self.assertIn("prior question", history)

    def test_no_history_formats_prompt_correctly(self):
        model, enc = self._mock_setup()
        generate_with_history(model, enc, "Who are you?", history="", device="cpu")
        call_arg = enc.encode.call_args[0][0]
        self.assertIn("Human: Who are you?", call_arg)
        self.assertIn("Assistant:", call_arg)

    def test_with_history_prepends_history_to_prompt(self):
        model, enc = self._mock_setup()
        prior = "Human: Hello\nAssistant: Hi\n"
        generate_with_history(model, enc, "Follow-up?", history=prior, device="cpu")
        call_arg = enc.encode.call_args[0][0]
        self.assertTrue(call_arg.startswith(prior))

    def test_reply_strips_human_continuation(self):
        # If the model echoes "Human:" in its output, only the part before it is kept
        model, enc = self._mock_setup(prompt_reply="deep answer\nHuman: next question")
        reply, _ = generate_with_history(model, enc, "prompt", device="cpu")
        self.assertEqual(reply, "deep answer")
        self.assertNotIn("Human:", reply)


class TestLoadModel(unittest.TestCase):

    def test_load_model_calls_from_pretrained(self):
        with patch("inference_utils.GPT2LMHeadModel") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.from_pretrained.return_value = mock_instance
            from inference_utils import load_model
            load_model("/fake/path", device="cpu")
            mock_cls.from_pretrained.assert_called_once_with("/fake/path")
            mock_instance.eval.assert_called_once()
            mock_instance.to.assert_called_once_with("cpu")

    def test_load_model_defaults_to_cpu_when_no_cuda(self):
        with patch("inference_utils.GPT2LMHeadModel") as mock_cls:
            mock_cls.from_pretrained.return_value = MagicMock()
            with patch("torch.cuda.is_available", return_value=False):
                from inference_utils import load_model
                load_model("/fake/path")
                mock_cls.from_pretrained.return_value.to.assert_called_once_with("cpu")


class TestGetTokenizer(unittest.TestCase):

    def test_get_tokenizer_calls_tiktoken(self):
        with patch("inference_utils.tiktoken") as mock_tiktoken:
            mock_tiktoken.get_encoding.return_value = MagicMock()
            from inference_utils import get_tokenizer
            get_tokenizer()
            mock_tiktoken.get_encoding.assert_called_once_with("gpt2")


if __name__ == "__main__":
    unittest.main()
