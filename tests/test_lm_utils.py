"""Tests for lm_utils.py — dataset, collator, trainer builder, and text loading."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from lm_utils import (
    LMDataset,
    LMExample,
    SimpleLMDataCollator,
    StreamingLMDataset,
    build_trainer,
    load_texts_from_data_dir,
    make_blocks,
)


# ---------------------------------------------------------------------------
# make_blocks
# ---------------------------------------------------------------------------

class TestMakeBlocks(unittest.TestCase):

    def test_exact_multiple(self):
        ids = list(range(12))
        blocks = make_blocks(ids, block_size=4)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0], [0, 1, 2, 3])
        self.assertEqual(blocks[2], [8, 9, 10, 11])

    def test_partial_tail_dropped(self):
        # 10 tokens with block_size=4 → 2 full blocks (8 tokens), 2 leftover dropped
        ids = list(range(10))
        blocks = make_blocks(ids, block_size=4)
        self.assertEqual(len(blocks), 2)

    def test_shorter_than_block_returns_empty(self):
        blocks = make_blocks([1, 2, 3], block_size=8)
        self.assertEqual(blocks, [])

    def test_single_block(self):
        ids = list(range(5))
        blocks = make_blocks(ids, block_size=5)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0], ids)

    def test_block_size_one(self):
        ids = [10, 20, 30]
        blocks = make_blocks(ids, block_size=1)
        self.assertEqual(len(blocks), 3)

    def test_all_blocks_are_correct_length(self):
        ids = list(range(100))
        block_size = 7
        blocks = make_blocks(ids, block_size)
        for b in blocks:
            self.assertEqual(len(b), block_size)


# ---------------------------------------------------------------------------
# LMDataset
# ---------------------------------------------------------------------------

class TestLMDataset(unittest.TestCase):

    def setUp(self):
        self.blocks = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.ds = LMDataset(self.blocks)

    def test_len(self):
        self.assertEqual(len(self.ds), 3)

    def test_getitem_returns_lm_example(self):
        item = self.ds[0]
        self.assertIsInstance(item, LMExample)

    def test_getitem_input_ids_match(self):
        self.assertEqual(self.ds[1].input_ids, [4, 5, 6])

    def test_empty_dataset(self):
        ds = LMDataset([])
        self.assertEqual(len(ds), 0)


# ---------------------------------------------------------------------------
# SimpleLMDataCollator
# ---------------------------------------------------------------------------

class TestSimpleLMDataCollator(unittest.TestCase):

    PAD_ID = 0

    def setUp(self):
        self.collator = SimpleLMDataCollator(pad_id=self.PAD_ID)

    def test_collate_lm_examples_returns_dict(self):
        features = [LMExample([1, 2, 3]), LMExample([4, 5, 6])]
        batch = self.collator(features)
        self.assertIsInstance(batch, dict)

    def test_collate_lm_examples_has_required_keys(self):
        features = [LMExample([1, 2]), LMExample([3, 4])]
        batch = self.collator(features)
        for key in ("input_ids", "labels", "attention_mask"):
            self.assertIn(key, batch)

    def test_collate_pads_shorter_sequences(self):
        features = [LMExample([1, 2, 3]), LMExample([4, 5])]
        batch = self.collator(features)
        # All rows should have the same length (max length = 3)
        self.assertEqual(batch["input_ids"].shape[1], 3)
        # Padded position should be PAD_ID
        self.assertEqual(batch["input_ids"][1, 2].item(), self.PAD_ID)

    def test_collate_attention_mask_zeros_on_pad(self):
        features = [LMExample([1, 2, 3]), LMExample([4, 5])]
        batch = self.collator(features)
        # The padded position should have mask=0
        self.assertEqual(batch["attention_mask"][1, 2].item(), 0)
        # Non-padded positions should have mask=1
        self.assertEqual(batch["attention_mask"][1, 0].item(), 1)

    def test_collate_tensors_directly(self):
        tensors = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        batch = self.collator(tensors)
        self.assertEqual(batch["input_ids"].shape, (2, 3))

    def test_collate_labels_equal_input_ids(self):
        features = [LMExample([1, 2, 3]), LMExample([4, 5, 6])]
        batch = self.collator(features)
        self.assertTrue(torch.equal(batch["input_ids"], batch["labels"]))

    def test_unknown_type_raises(self):
        with self.assertRaises(TypeError):
            self.collator(["not_a_valid_type"])


# ---------------------------------------------------------------------------
# load_texts_from_data_dir
# ---------------------------------------------------------------------------

class TestLoadTextsFromDataDir(unittest.TestCase):

    def test_loads_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "a.txt").write_text("hello world", encoding="utf-8")
            (p / "b.txt").write_text("foo bar", encoding="utf-8")
            texts = list(load_texts_from_data_dir(tmpdir))
            self.assertEqual(len(texts), 2)
            self.assertIn("hello world", texts)
            self.assertIn("foo bar", texts)

    def test_ignores_non_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "data.txt").write_text("valid", encoding="utf-8")
            (p / "skip.json").write_text("{}", encoding="utf-8")
            texts = list(load_texts_from_data_dir(tmpdir))
            self.assertEqual(len(texts), 1)

    def test_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            list(load_texts_from_data_dir("/nonexistent/path"))

    def test_no_txt_files_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                list(load_texts_from_data_dir(tmpdir))

    def test_recurses_into_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            sub = p / "sub"
            sub.mkdir()
            (sub / "nested.txt").write_text("nested content", encoding="utf-8")
            texts = list(load_texts_from_data_dir(tmpdir))
            self.assertEqual(len(texts), 1)
            self.assertEqual(texts[0], "nested content")


# ---------------------------------------------------------------------------
# StreamingLMDataset
# ---------------------------------------------------------------------------

class TestStreamingLMDataset(unittest.TestCase):

    def _make_encoding(self, ids):
        enc = MagicMock()
        enc.encode.return_value = ids
        return enc

    def test_raises_on_missing_dir(self):
        enc = self._make_encoding([1, 2, 3])
        with self.assertRaises(FileNotFoundError):
            StreamingLMDataset("/nonexistent/dir", enc, eos_id=0, block_size=4)

    def test_raises_on_invalid_shuffle_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            enc = self._make_encoding([1, 2, 3])
            with self.assertRaises(ValueError):
                StreamingLMDataset(tmpdir, enc, eos_id=0, block_size=4, shuffle_buffer=0)

    def test_raises_on_invalid_block_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            enc = self._make_encoding([1, 2, 3])
            with self.assertRaises(ValueError):
                StreamingLMDataset(tmpdir, enc, eos_id=0, block_size=0)

    def test_iterates_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            # Write a file with enough text for 3 blocks of size 4
            (p / "data.txt").write_text("dummy", encoding="utf-8")
            # Return 13 tokens from encode (→ 3 full blocks of size 4)
            enc = MagicMock()
            enc.encode.return_value = list(range(13))
            ds = StreamingLMDataset(tmpdir, enc, eos_id=99, block_size=4, shuffle_buffer=5)
            blocks = list(ds)
            self.assertGreater(len(blocks), 0)
            for block in blocks:
                self.assertIsInstance(block, torch.Tensor)
                self.assertEqual(len(block), 4)

    def test_raises_when_no_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            enc = self._make_encoding(list(range(20)))
            ds = StreamingLMDataset(tmpdir, enc, eos_id=0, block_size=4)
            with self.assertRaises(FileNotFoundError):
                list(ds)


# ---------------------------------------------------------------------------
# build_trainer (smoke test — just checks it returns a Trainer)
# ---------------------------------------------------------------------------

class TestBuildTrainer(unittest.TestCase):

    def test_returns_trainer_object(self):
        from transformers import GPT2Config, GPT2LMHeadModel, Trainer
        # Use a real (tiny) HuggingFace model so Trainer can infer the framework.
        cfg = GPT2Config(
            vocab_size=100, n_positions=16, n_embd=16, n_layer=2, n_head=2
        )
        model = GPT2LMHeadModel(cfg)
        dataset = LMDataset([[1, 2, 3, 4]] * 4)
        collator = SimpleLMDataCollator(pad_id=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                model, dataset, collator,
                output_dir=tmpdir,
                max_steps=2,
                save_steps=1,
                logging_steps=1,
            )
            self.assertIsInstance(trainer, Trainer)


if __name__ == "__main__":
    unittest.main()
