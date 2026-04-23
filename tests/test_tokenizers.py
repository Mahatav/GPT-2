"""Tests for tokenizer/ — CharTokenizer and BPETokenizer."""

import os
import tempfile
import unittest

from tokenizer.char_tokenizer import CharTokenizer
from tokenizer.bpe_tokenizer import BPETokenizer


CORPUS = (
    "Hello world! This is a test corpus.\n"
    "It has multiple lines and punctuation: ,;:!?\n"
    "Numbers too: 123 456.\n"
) * 5


class TestCharTokenizer(unittest.TestCase):

    def setUp(self):
        self.tok = CharTokenizer(CORPUS)

    def test_vocab_size_equals_unique_chars(self):
        expected = len(set(CORPUS))
        self.assertEqual(self.tok.vocab_size, expected)

    def test_encode_returns_ints(self):
        ids = self.tok.encode("Hello")
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_decode_returns_string(self):
        ids = self.tok.encode("Hello")
        text = self.tok.decode(ids)
        self.assertIsInstance(text, str)

    def test_roundtrip_exact(self):
        sample = "Hello world!"
        self.assertEqual(self.tok.decode(self.tok.encode(sample)), sample)

    def test_encode_full_corpus_roundtrip(self):
        self.assertTrue(self.tok.roundtrip(CORPUS))

    def test_unknown_chars_silently_skipped(self):
        # Chinese character not in CORPUS — should not raise
        ids = self.tok.encode("Hello 你好")
        text = self.tok.decode(ids)
        self.assertEqual(text, "Hello ")   # unknown chars dropped

    def test_encode_length_matches_known_chars(self):
        sample = "abc"
        ids = self.tok.encode(sample)
        self.assertEqual(len(ids), 3)

    def test_vocab_ids_are_contiguous(self):
        # Every id from 0 to vocab_size-1 should be representable
        vocab = self.tok.vocab_list()
        self.assertEqual(len(vocab), self.tok.vocab_size)

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self.tok.save(path)
            loaded = CharTokenizer.load(path)
            self.assertEqual(loaded.vocab_size, self.tok.vocab_size)
            self.assertEqual(loaded.encode("Hello"), self.tok.encode("Hello"))
            self.assertTrue(loaded.roundtrip("Hello world!"))
        finally:
            os.unlink(path)


class TestBPETokenizer(unittest.TestCase):

    BPE_VOCAB = 60   # small enough to train quickly

    def setUp(self):
        self.tok = BPETokenizer()
        self.tok.train(CORPUS, vocab_size=self.BPE_VOCAB, verbose=False)

    def test_vocab_size_matches_target(self):
        self.assertEqual(self.tok.vocab_size, self.BPE_VOCAB)

    def test_encode_returns_ints(self):
        ids = self.tok.encode("Hello")
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_decode_returns_string(self):
        ids = self.tok.encode("Hello world!")
        self.assertIsInstance(self.tok.decode(ids), str)

    def test_roundtrip(self):
        sample = "Hello world!"
        self.assertTrue(self.tok.roundtrip(sample))

    def test_full_corpus_roundtrip(self):
        self.assertTrue(self.tok.roundtrip(CORPUS))

    def test_bpe_compresses_sequences(self):
        # BPE sequences should be shorter than char-level sequences
        char_len = len(CORPUS)
        bpe_len  = len(self.tok.encode(CORPUS))
        self.assertLess(bpe_len, char_len,
                        "BPE should produce shorter sequences than character-level")

    def test_unknown_chars_skipped(self):
        ids = self.tok.encode("Hello 你好")  # CJK chars not in training set
        text = self.tok.decode(ids)
        self.assertIn("H", text)  # known chars should appear

    def test_train_requires_vocab_larger_than_chars(self):
        tok = BPETokenizer()
        n_unique = len(set(CORPUS))
        with self.assertRaises(ValueError):
            tok.train(CORPUS, vocab_size=n_unique - 1, verbose=False)

    def test_encode_before_train_raises(self):
        tok = BPETokenizer()
        with self.assertRaises(RuntimeError):
            tok.encode("hello")

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self.tok.save(path)
            loaded = BPETokenizer.load(path)
            self.assertEqual(loaded.vocab_size, self.tok.vocab_size)
            self.assertEqual(loaded.encode("Hello"), self.tok.encode("Hello"))
        finally:
            os.unlink(path)

    def test_vocab_table_returns_string(self):
        table = self.tok.vocab_table(max_rows=10)
        self.assertIsInstance(table, str)
        self.assertIn("ID", table)


if __name__ == "__main__":
    unittest.main()
