"""Tests for vocab_bubbles.py — tokenization, frequency counting, and circle packing."""

import math
import unittest
from collections import Counter

import numpy as np

from vocab_bubbles import (
    FILTER,
    _blend,
    _scale_radii,
    pack_circles,
    raw_bigrams,
    raw_counts,
    tokenize,
    top_counts,
)


class TestTokenize(unittest.TestCase):

    def test_extracts_alpha_words(self):
        result = tokenize("Hello, world!")
        self.assertIn("hello", result)
        self.assertIn("world", result)

    def test_lowercases(self):
        result = tokenize("UPPER Mixed lower")
        self.assertEqual(result, ["upper", "mixed", "lower"])

    def test_filters_single_char_words(self):
        result = tokenize("I am a good person")
        self.assertNotIn("i", result)
        self.assertNotIn("a", result)
        self.assertIn("am", result)

    def test_strips_numbers(self):
        result = tokenize("year 2024 AD")
        self.assertNotIn("2024", result)
        self.assertIn("year", result)
        self.assertIn("ad", result)

    def test_empty_string(self):
        self.assertEqual(tokenize(""), [])

    def test_only_punctuation(self):
        self.assertEqual(tokenize("... --- !!!"), [])


class TestRawCounts(unittest.TestCase):

    def test_counts_content_words(self):
        counts = raw_counts(["dharma karma dharma"])
        self.assertEqual(counts["dharma"], 2)
        self.assertEqual(counts["karma"], 1)

    def test_filters_stopwords(self):
        # "the", "and", "is" are all in FILTER
        counts = raw_counts(["the cat and the dog is here"])
        self.assertNotIn("the", counts)
        self.assertNotIn("and", counts)
        self.assertNotIn("is", counts)
        self.assertIn("cat", counts)
        self.assertIn("dog", counts)

    def test_accumulates_across_texts(self):
        counts = raw_counts(["apple banana", "apple cherry"])
        self.assertEqual(counts["apple"], 2)
        self.assertEqual(counts["banana"], 1)
        self.assertEqual(counts["cherry"], 1)

    def test_empty_texts_list(self):
        counts = raw_counts([])
        self.assertEqual(len(counts), 0)

    def test_all_stopwords_gives_empty(self):
        counts = raw_counts(["the and is are a"])
        self.assertEqual(len(counts), 0)


class TestRawBigrams(unittest.TestCase):

    def test_counts_content_bigrams(self):
        bigrams = raw_bigrams(["dharma karma nirvana"])
        self.assertIn(("dharma", "karma"), bigrams)
        self.assertIn(("karma", "nirvana"), bigrams)

    def test_skips_stopword_adjacent_bigrams(self):
        # "the dog" → "the" is a stopword, "dog" is content
        # Only bigrams where BOTH tokens are content words are kept
        bigrams = raw_bigrams(["the dog runs fast"])
        # "the dog" should be skipped ("the" is a stopword)
        self.assertNotIn(("the", "dog"), bigrams)
        # "dog runs" — "dog" and "runs" are both content
        self.assertIn(("dog", "runs"), bigrams)

    def test_accumulates_across_texts(self):
        bigrams = raw_bigrams(["cat dog", "cat dog"])
        self.assertEqual(bigrams[("cat", "dog")], 2)

    def test_empty_returns_empty(self):
        bigrams = raw_bigrams([])
        self.assertEqual(len(bigrams), 0)

    def test_single_word_produces_no_bigrams(self):
        bigrams = raw_bigrams(["dharma"])
        self.assertEqual(len(bigrams), 0)


class TestTopCounts(unittest.TestCase):

    def test_returns_top_n(self):
        counter = Counter({"a": 10, "b": 5, "c": 3, "d": 1})
        top = top_counts(counter, 2)
        self.assertEqual(len(top), 2)
        self.assertIn("a", top)
        self.assertIn("b", top)
        self.assertNotIn("d", top)

    def test_top_n_larger_than_counter(self):
        counter = Counter({"x": 2, "y": 1})
        top = top_counts(counter, 100)
        self.assertEqual(len(top), 2)

    def test_returns_counter_type(self):
        counter = Counter({"a": 5})
        top = top_counts(counter, 1)
        self.assertIsInstance(top, Counter)


class TestBlend(unittest.TestCase):

    def test_frac_zero_gives_red(self):
        """frac=0 → pure red end of the spectrum."""
        r, g, b = _blend(0.0)
        self.assertGreater(r, g)
        self.assertGreater(r, b)

    def test_frac_one_gives_blue(self):
        """frac=1 → pure blue end of the spectrum."""
        r, g, b = _blend(1.0)
        self.assertGreater(b, r)

    def test_frac_half_gives_purple(self):
        """frac=0.5 → purple midpoint."""
        color = _blend(0.5)
        self.assertEqual(len(color), 3)
        # All components in [0, 1]
        for c in color:
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)

    def test_all_components_in_range(self):
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            color = _blend(frac)
            for c in color:
                self.assertGreaterEqual(c, 0.0)
                self.assertLessEqual(c, 1.0)

    def test_returns_tuple(self):
        result = _blend(0.5)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)


class TestScaleRadii(unittest.TestCase):

    def test_largest_freq_gets_max_radius(self):
        freqs = np.array([1.0, 5.0, 10.0])
        radii = _scale_radii(freqs, max_r=2.0, min_r=0.2)
        self.assertAlmostEqual(radii[2], 2.0, places=5)

    def test_smallest_freq_gets_min_radius(self):
        freqs = np.array([1.0, 5.0, 10.0])
        radii = _scale_radii(freqs, max_r=2.0, min_r=0.2)
        self.assertAlmostEqual(radii[0], 0.2, places=5)

    def test_all_same_freq_gives_midpoint(self):
        freqs = np.array([3.0, 3.0, 3.0])
        radii = _scale_radii(freqs, max_r=2.0, min_r=0.2)
        expected = (2.0 + 0.2) / 2
        for r in radii:
            self.assertAlmostEqual(r, expected, places=5)

    def test_output_length_matches_input(self):
        freqs = np.array([1.0, 2.0, 3.0, 4.0])
        radii = _scale_radii(freqs)
        self.assertEqual(len(radii), 4)


class TestPackCircles(unittest.TestCase):

    def test_returns_correct_number_of_positions(self):
        radii = [1.0, 0.8, 0.5, 0.3]
        positions = pack_circles(radii)
        self.assertEqual(len(positions), len(radii))

    def test_positions_are_2d_tuples(self):
        radii = [1.0, 0.5]
        positions = pack_circles(radii)
        for p in positions:
            self.assertEqual(len(p), 2)

    def test_first_circle_at_origin(self):
        positions = pack_circles([1.0, 0.5])
        self.assertAlmostEqual(positions[0][0], 0.0, places=5)
        self.assertAlmostEqual(positions[0][1], 0.0, places=5)

    def test_no_circles_overlap(self):
        radii = [0.8, 0.6, 0.5, 0.4, 0.3]
        positions = pack_circles(radii, seed=0)
        gap = 0.04  # slight tolerance smaller than the pack_circles gap param
        for i in range(len(radii)):
            for j in range(i + 1, len(radii)):
                xi, yi = positions[i]
                xj, yj = positions[j]
                dist = math.hypot(xi - xj, yi - yj)
                min_dist = radii[i] + radii[j]
                self.assertGreaterEqual(
                    dist + gap, min_dist,
                    f"Circles {i} and {j} overlap: dist={dist:.3f}, min={min_dist:.3f}",
                )

    def test_single_circle(self):
        positions = pack_circles([1.0])
        self.assertEqual(len(positions), 1)

    def test_deterministic_with_same_seed(self):
        radii = [1.0, 0.7, 0.4]
        p1 = pack_circles(radii, seed=42)
        p2 = pack_circles(radii, seed=42)
        for (x1, y1), (x2, y2) in zip(p1, p2):
            self.assertAlmostEqual(x1, x2, places=5)
            self.assertAlmostEqual(y1, y2, places=5)


class TestFilterSet(unittest.TestCase):

    def test_common_stopwords_in_filter(self):
        for word in ("the", "and", "is", "of", "to", "in", "a"):
            self.assertIn(word, FILTER, f"Expected '{word}' in FILTER")

    def test_pronouns_in_filter(self):
        for word in ("i", "you", "he", "she", "they", "we"):
            self.assertIn(word, FILTER)

    def test_content_words_not_in_filter(self):
        for word in ("philosophy", "dharma", "consciousness", "wisdom"):
            self.assertNotIn(word, FILTER)


if __name__ == "__main__":
    unittest.main()
