"""Tests for stats_analysis.py — Bhattacharyya coefficient and overlap metrics."""

import math
import unittest

from stats_analysis import (
    analyze_category,
    bhattacharyya_coefficient,
    bhattacharyya_distance,
    compute_overlap_metrics,
    get_word_distribution,
    tokenize,
)


class TestTokenize(unittest.TestCase):

    def test_extracts_words(self):
        result = tokenize("Hello world")
        self.assertIn("hello", result)
        self.assertIn("world", result)

    def test_lowercases(self):
        result = tokenize("UPPER lower Mixed")
        self.assertEqual(result, ["upper", "lower", "mixed"])

    def test_filters_single_char(self):
        result = tokenize("I am a big dog")
        # "i", "a" are length 1 — should be filtered
        self.assertNotIn("i", result)
        self.assertNotIn("a", result)
        self.assertIn("am", result)
        self.assertIn("big", result)
        self.assertIn("dog", result)

    def test_strips_punctuation(self):
        result = tokenize("hello, world! it's great.")
        self.assertIn("hello", result)
        self.assertIn("world", result)
        self.assertIn("great", result)
        # Apostrophes split tokens — "it" and "great" still included
        self.assertNotIn("hello,", result)

    def test_empty_string(self):
        self.assertEqual(tokenize(""), [])

    def test_numbers_excluded(self):
        result = tokenize("year 2024 philosophy")
        self.assertNotIn("2024", result)
        self.assertIn("year", result)


class TestGetWordDistribution(unittest.TestCase):

    def test_sums_to_one(self):
        dist = get_word_distribution(["apple orange apple"])
        total = sum(dist.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_more_frequent_word_has_higher_prob(self):
        dist = get_word_distribution(["apple apple apple orange"])
        self.assertGreater(dist["apple"], dist["orange"])

    def test_empty_texts_returns_empty(self):
        dist = get_word_distribution([""])
        self.assertEqual(dist, {})

    def test_all_texts_included(self):
        dist = get_word_distribution(["cat dog", "fish bird"])
        for word in ("cat", "dog", "fish", "bird"):
            self.assertIn(word, dist)

    def test_values_are_non_negative(self):
        dist = get_word_distribution(["hello world"])
        for v in dist.values():
            self.assertGreaterEqual(v, 0.0)


class TestBhattacharyyaCoefficient(unittest.TestCase):

    def test_identical_distributions_give_one(self):
        p = {"a": 0.5, "b": 0.5}
        bc = bhattacharyya_coefficient(p, p)
        self.assertAlmostEqual(bc, 1.0, places=6)

    def test_disjoint_distributions_give_zero(self):
        p = {"a": 1.0}
        q = {"b": 1.0}
        bc = bhattacharyya_coefficient(p, q)
        self.assertAlmostEqual(bc, 0.0, places=6)

    def test_result_between_zero_and_one(self):
        p = {"a": 0.6, "b": 0.4}
        q = {"a": 0.2, "b": 0.8}
        bc = bhattacharyya_coefficient(p, q)
        self.assertGreaterEqual(bc, 0.0)
        self.assertLessEqual(bc, 1.0)

    def test_symmetry(self):
        p = {"a": 0.7, "b": 0.3}
        q = {"a": 0.4, "b": 0.6}
        self.assertAlmostEqual(
            bhattacharyya_coefficient(p, q),
            bhattacharyya_coefficient(q, p),
            places=6,
        )

    def test_partial_overlap(self):
        p = {"a": 0.5, "b": 0.5}
        q = {"b": 0.5, "c": 0.5}
        bc = bhattacharyya_coefficient(p, q)
        # Only word "b" is shared → BC = sqrt(0.5 * 0.5) = 0.5
        self.assertAlmostEqual(bc, 0.5, places=6)


class TestBhattacharyyaDistance(unittest.TestCase):

    def test_identical_distributions_give_zero(self):
        p = {"a": 0.5, "b": 0.5}
        bd = bhattacharyya_distance(p, p)
        self.assertAlmostEqual(bd, 0.0, places=6)

    def test_disjoint_distributions_give_inf(self):
        p = {"a": 1.0}
        q = {"b": 1.0}
        bd = bhattacharyya_distance(p, q)
        self.assertEqual(bd, float("inf"))

    def test_non_negative(self):
        p = {"a": 0.6, "b": 0.4}
        q = {"a": 0.2, "b": 0.8}
        bd = bhattacharyya_distance(p, q)
        self.assertGreaterEqual(bd, 0.0)

    def test_distance_is_minus_log_bc(self):
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 0.4, "b": 0.6}
        bc = bhattacharyya_coefficient(p, q)
        bd = bhattacharyya_distance(p, q)
        self.assertAlmostEqual(bd, -math.log(bc), places=6)


class TestComputeOverlapMetrics(unittest.TestCase):

    def test_returns_expected_keys(self):
        result = compute_overlap_metrics(["hello world"], ["hello world"])
        expected = {
            "bhattacharyya_coefficient",
            "bhattacharyya_distance",
            "bhattacharyya_interpretation",
            "unique_western_words",
            "unique_eastern_words",
            "common_words",
        }
        self.assertEqual(set(result.keys()), expected)

    def test_identical_texts_high_bc(self):
        text = ["the quick brown fox jumps over the lazy dog"]
        result = compute_overlap_metrics(text, text)
        self.assertGreater(result["bhattacharyya_coefficient"], 0.99)

    def test_disjoint_vocabularies_give_zero_bc(self):
        result = compute_overlap_metrics(["alpha beta gamma"], ["delta epsilon zeta"])
        self.assertAlmostEqual(result["bhattacharyya_coefficient"], 0.0, places=4)

    def test_common_words_count(self):
        result = compute_overlap_metrics(["apple banana"], ["banana cherry"])
        # "banana" is common — but single-char words filtered; these are all 2+ chars
        self.assertEqual(result["common_words"], 1)

    def test_bc_is_rounded(self):
        result = compute_overlap_metrics(["test text"], ["test text"])
        bc = result["bhattacharyya_coefficient"]
        self.assertEqual(bc, round(bc, 4))


class TestAnalyzeCategory(unittest.TestCase):

    def _make_eval(self, west, east):
        return {"western_output": west, "eastern_output": east}

    def test_returns_category_name(self):
        evals = [self._make_eval("soul reason virtue", "dharma karma nirvana")]
        result = analyze_category("ethics", evals)
        self.assertEqual(result["category"], "ethics")

    def test_returns_num_prompts(self):
        evals = [
            self._make_eval("text one", "text two"),
            self._make_eval("text three", "text four"),
        ]
        result = analyze_category("self", evals)
        self.assertEqual(result["num_prompts"], 2)

    def test_includes_bc_and_bd(self):
        evals = [self._make_eval("hello world", "hello world")]
        result = analyze_category("test", evals)
        self.assertIn("bhattacharyya_coefficient", result)
        self.assertIn("bhattacharyya_distance", result)

    def test_empty_evaluations(self):
        # Should handle gracefully (empty distribution → BC = 0)
        result = analyze_category("empty", [])
        self.assertEqual(result["num_prompts"], 0)


if __name__ == "__main__":
    unittest.main()
