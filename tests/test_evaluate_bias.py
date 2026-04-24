"""Tests for evaluate_bias.py — output analysis helpers."""

import unittest

from evaluate_bias import (
    EASTERN_MARKERS,
    WESTERN_MARKERS,
    analyze_single_output,
    compute_concept_frequencies,
    compute_repetition_score,
    compute_type_token_ratio,
)


class TestComputeRepetitionScore(unittest.TestCase):

    def test_no_repetition_gives_zero(self):
        # Unique 4-grams throughout
        text = "the quick brown fox jumps over the lazy dog and runs away"
        score = compute_repetition_score(text)
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_full_repetition_gives_high_score(self):
        # Same 4-gram repeated many times
        text = "a b c d " * 20
        score = compute_repetition_score(text)
        self.assertGreater(score, 0.8)

    def test_partial_repetition_between_zero_and_one(self):
        # "a b c d" repeats → 4-gram ("a","b","c","d") appears more than once
        text = "a b c d a b c d a b c d e f g h"
        score = compute_repetition_score(text)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_short_text_below_ngram_size_gives_zero(self):
        text = "short"
        score = compute_repetition_score(text, ngram_size=4)
        self.assertEqual(score, 0.0)

    def test_empty_string_gives_zero(self):
        self.assertEqual(compute_repetition_score(""), 0.0)

    def test_custom_ngram_size(self):
        # "a b" repeated: every bigram is the same
        text = "a b a b a b a b a b"
        score_2 = compute_repetition_score(text, ngram_size=2)
        score_4 = compute_repetition_score(text, ngram_size=4)
        # Bigger ngram window → more unique sequences expected, but here both loop
        self.assertGreater(score_2, 0.0)
        self.assertGreater(score_4, 0.0)


class TestComputeTypeTokenRatio(unittest.TestCase):

    def test_all_unique_gives_one(self):
        text = "apple banana cherry date elderberry"
        ratio = compute_type_token_ratio(text)
        self.assertAlmostEqual(ratio, 1.0, places=4)

    def test_all_same_gives_low_ratio(self):
        text = "the the the the the"
        ratio = compute_type_token_ratio(text)
        self.assertAlmostEqual(ratio, 1 / 5, places=4)

    def test_empty_string_gives_zero(self):
        self.assertEqual(compute_type_token_ratio(""), 0.0)

    def test_result_between_zero_and_one(self):
        text = "philosophy is the love of wisdom and wisdom is rare"
        ratio = compute_type_token_ratio(text)
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)

    def test_case_insensitive(self):
        # "The" and "the" are the same type
        text_lower = "the the the"
        text_mixed = "The THE the"
        self.assertAlmostEqual(
            compute_type_token_ratio(text_lower),
            compute_type_token_ratio(text_mixed),
            places=4,
        )


class TestComputeConceptFrequencies(unittest.TestCase):

    def test_returns_all_expected_keys(self):
        result = compute_concept_frequencies("some text")
        expected = {
            "eastern_marker_count",
            "western_marker_count",
            "eastern_ratio",
            "western_ratio",
        }
        self.assertEqual(set(result.keys()), expected)

    def test_eastern_text_has_higher_eastern_ratio(self):
        eastern_text = " ".join(EASTERN_MARKERS[:5] * 3)
        result = compute_concept_frequencies(eastern_text)
        self.assertGreater(result["eastern_ratio"], result["western_ratio"])

    def test_western_text_has_higher_western_ratio(self):
        western_text = " ".join(WESTERN_MARKERS[:5] * 3)
        result = compute_concept_frequencies(western_text)
        self.assertGreater(result["western_ratio"], result["eastern_ratio"])

    def test_no_markers_gives_zero_counts(self):
        result = compute_concept_frequencies("the sky is blue and grass is green")
        self.assertEqual(result["eastern_marker_count"], 0)
        self.assertEqual(result["western_marker_count"], 0)

    def test_ratios_sum_to_one_when_both_present(self):
        text = "dharma logos"  # one eastern, one western
        result = compute_concept_frequencies(text)
        if result["eastern_marker_count"] + result["western_marker_count"] > 0:
            total = result["eastern_ratio"] + result["western_ratio"]
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_ratios_zero_when_no_markers(self):
        # Use text that contains no marker substrings at all.
        # Avoid words like "flying" (contains "yin") or "reason" (western marker).
        result = compute_concept_frequencies("spectacular colorful parrots")
        self.assertEqual(result["eastern_ratio"], 0.0)
        self.assertEqual(result["western_ratio"], 0.0)

    def test_case_insensitive_counting(self):
        text_lower = "dharma karma nirvana"
        text_upper = "DHARMA KARMA NIRVANA"
        r_lower = compute_concept_frequencies(text_lower)
        r_upper = compute_concept_frequencies(text_upper)
        self.assertEqual(r_lower["eastern_marker_count"], r_upper["eastern_marker_count"])


class TestAnalyzeSingleOutput(unittest.TestCase):

    def test_returns_all_expected_keys(self):
        result = analyze_single_output("some philosophical text here")
        expected = {
            "length_chars",
            "length_words",
            "repetition_score",
            "type_token_ratio",
            "concept_frequencies",
        }
        self.assertEqual(set(result.keys()), expected)

    def test_length_chars_correct(self):
        text = "hello world"
        result = analyze_single_output(text)
        self.assertEqual(result["length_chars"], len(text))

    def test_length_words_correct(self):
        text = "one two three four five"
        result = analyze_single_output(text)
        self.assertEqual(result["length_words"], 5)

    def test_repetition_score_in_range(self):
        result = analyze_single_output("the quick brown fox")
        self.assertGreaterEqual(result["repetition_score"], 0.0)
        self.assertLessEqual(result["repetition_score"], 1.0)

    def test_concept_frequencies_is_dict(self):
        result = analyze_single_output("dharma logos")
        self.assertIsInstance(result["concept_frequencies"], dict)

    def test_empty_string(self):
        result = analyze_single_output("")
        self.assertEqual(result["length_chars"], 0)
        self.assertEqual(result["length_words"], 0)


class TestMarkerLists(unittest.TestCase):

    def test_western_markers_non_empty(self):
        self.assertGreater(len(WESTERN_MARKERS), 0)

    def test_eastern_markers_non_empty(self):
        self.assertGreater(len(EASTERN_MARKERS), 0)

    def test_markers_are_lowercase(self):
        for m in WESTERN_MARKERS:
            self.assertEqual(m, m.lower(), f"Western marker not lowercase: {m!r}")
        for m in EASTERN_MARKERS:
            self.assertEqual(m, m.lower(), f"Eastern marker not lowercase: {m!r}")

    def test_no_overlap_between_lists(self):
        overlap = set(WESTERN_MARKERS) & set(EASTERN_MARKERS)
        self.assertEqual(overlap, set(), f"Overlapping markers: {overlap}")


if __name__ == "__main__":
    unittest.main()
