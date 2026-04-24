"""Tests for progressive_pipeline.py — step computation and data-loading helpers."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Stub out optional heavy dependencies that aren't needed for the tested functions.
for _mod in ("gpt2_pretrain", "tiktoken", "transformers",
             "transformers.GPT2Config", "transformers.GPT2LMHeadModel"):
    sys.modules.setdefault(_mod, MagicMock())

from progressive_pipeline import (
    TIME_PERIODS,
    compute_epoch_capped_steps,
    compute_max_steps,
    compute_save_steps,
    get_period_directories,
    load_cumulative_texts,
)


class TestComputeMaxSteps(unittest.TestCase):

    def test_scales_with_texts(self):
        steps = compute_max_steps(num_texts=50, steps_per_text=20)
        self.assertEqual(steps, 1000)

    def test_respects_min_steps(self):
        # 1 text * 20 steps = 20, but min is 300
        steps = compute_max_steps(num_texts=1, min_steps=300)
        self.assertEqual(steps, 300)

    def test_respects_max_steps_cap(self):
        steps = compute_max_steps(num_texts=10000, max_steps_cap=5000)
        self.assertEqual(steps, 5000)

    def test_override_returns_exactly_override(self):
        steps = compute_max_steps(num_texts=50, override=999)
        self.assertEqual(steps, 999)

    def test_override_bypasses_cap(self):
        # Even if override > max_steps_cap, override wins
        steps = compute_max_steps(num_texts=1, max_steps_cap=100, override=9999)
        self.assertEqual(steps, 9999)

    def test_override_bypasses_min(self):
        steps = compute_max_steps(num_texts=1, min_steps=300, override=5)
        self.assertEqual(steps, 5)

    def test_zero_texts_gives_min_steps(self):
        steps = compute_max_steps(num_texts=0, min_steps=300)
        self.assertEqual(steps, 300)

    def test_formula_correctness(self):
        # steps = clamp(40 * 20, 300, 5000) = clamp(800, 300, 5000) = 800
        steps = compute_max_steps(
            num_texts=40,
            steps_per_text=20,
            min_steps=300,
            max_steps_cap=5000,
        )
        self.assertEqual(steps, 800)


class TestComputeEpochCappedSteps(unittest.TestCase):

    def test_cap_applied_when_steps_exceed_epoch_budget(self):
        # dataset_size=10, batch_size=2 → 5 steps/epoch
        # max_epochs=3 → epoch_cap = 15
        # max_steps=1000 → should be capped to 15
        result = compute_epoch_capped_steps(
            max_steps=1000,
            dataset_size=10,
            batch_size=2,
            max_epochs=3,
            min_steps=1,
        )
        self.assertEqual(result, 15)

    def test_no_cap_when_steps_within_epoch_budget(self):
        # epoch_cap = 500, max_steps = 200 → no reduction
        result = compute_epoch_capped_steps(
            max_steps=200,
            dataset_size=1000,
            batch_size=2,
            max_epochs=1,
            min_steps=1,
        )
        self.assertEqual(result, 200)

    def test_min_steps_floor_preserved(self):
        # epoch_cap = 1 (tiny dataset), max_steps = 1, min_steps = 300
        result = compute_epoch_capped_steps(
            max_steps=1,
            dataset_size=1,
            batch_size=100,
            max_epochs=1,
            min_steps=300,
        )
        self.assertEqual(result, 300)

    def test_zero_dataset_safe(self):
        # dataset_size=0 → steps_per_epoch = max(1, 0) = 1
        result = compute_epoch_capped_steps(
            max_steps=500,
            dataset_size=0,
            batch_size=8,
            max_epochs=3,
            min_steps=1,
        )
        # epoch_cap = 1 * 3 = 3
        self.assertEqual(result, 3)

    def test_result_never_exceeds_max_steps(self):
        result = compute_epoch_capped_steps(
            max_steps=50,
            dataset_size=10000,
            batch_size=2,
            max_epochs=100,
            min_steps=1,
        )
        self.assertLessEqual(result, 50)


class TestComputeSaveSteps(unittest.TestCase):

    def test_roughly_ten_percent(self):
        self.assertEqual(compute_save_steps(1000), 100)
        self.assertEqual(compute_save_steps(5000), 500)

    def test_minimum_100(self):
        # For very small max_steps the floor is 100
        self.assertEqual(compute_save_steps(50), 100)
        self.assertEqual(compute_save_steps(100), 100)

    def test_large_value(self):
        self.assertEqual(compute_save_steps(10000), 1000)


class TestGetPeriodDirectories(unittest.TestCase):

    def _make_region_dirs(self, tmpdir, region, periods):
        """Create fake period directories and return base data dir path."""
        base = Path(tmpdir) / region
        for p in periods:
            (base / p).mkdir(parents=True, exist_ok=True)
        return tmpdir

    def test_invalid_period_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                get_period_directories("east", "9999", data_base_dir=tmpdir)

    def test_returns_only_existing_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Only create dirs for "older (BC)" and "100"
            self._make_region_dirs(tmpdir, "east", ["older (BC)", "100"])
            # Ask for dirs up to "200" — "200" doesn't exist so should be skipped
            dirs = get_period_directories("east", "200", data_base_dir=tmpdir)
            names = [d.name for d in dirs]
            self.assertIn("older (BC)", names)
            self.assertIn("100", names)
            self.assertNotIn("200", names)

    def test_cumulative_up_to_period(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            periods_to_create = ["older (BC)", "100", "200", "300"]
            self._make_region_dirs(tmpdir, "west", periods_to_create)
            # Ask up to "200" → should include "older (BC)", "100", "200"
            dirs = get_period_directories("west", "200", data_base_dir=tmpdir)
            self.assertEqual(len(dirs), 3)

    def test_first_period_returns_one_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_region_dirs(tmpdir, "east", ["older (BC)"])
            dirs = get_period_directories("east", "older (BC)", data_base_dir=tmpdir)
            self.assertEqual(len(dirs), 1)

    def test_missing_region_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # No region directory at all
            dirs = get_period_directories("east", "100", data_base_dir=tmpdir)
            self.assertEqual(dirs, [])


class TestLoadCumulativeTexts(unittest.TestCase):

    def _setup_data(self, tmpdir, region, period_files):
        """
        period_files: dict mapping period_name → list of (filename, content)
        """
        base = Path(tmpdir) / region
        for period, files in period_files.items():
            period_dir = base / period
            period_dir.mkdir(parents=True, exist_ok=True)
            for fname, content in files:
                (period_dir / fname).write_text(content, encoding="utf-8")

    def test_loads_all_texts_up_to_period(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_data(tmpdir, "east", {
                "older (BC)": [("a.txt", "ancient text")],
                "100": [("b.txt", "early CE text"), ("c.txt", "more CE text")],
            })
            texts = load_cumulative_texts("east", "100", data_base_dir=tmpdir)
            self.assertEqual(len(texts), 3)

    def test_does_not_include_future_period(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_data(tmpdir, "east", {
                "older (BC)": [("a.txt", "old text")],
                "200": [("b.txt", "future text")],
            })
            # Ask only up to "older (BC)" — "200" should not be included
            texts = load_cumulative_texts("east", "older (BC)", data_base_dir=tmpdir)
            self.assertEqual(len(texts), 1)
            self.assertIn("old text", texts)

    def test_empty_when_no_dirs_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            texts = load_cumulative_texts("east", "100", data_base_dir=tmpdir)
            self.assertEqual(texts, [])


class TestTimePeriods(unittest.TestCase):

    def test_first_period_is_bc(self):
        self.assertEqual(TIME_PERIODS[0], "older (BC)")

    def test_last_period_is_2000(self):
        self.assertEqual(TIME_PERIODS[-1], "2000")

    def test_contains_21_periods(self):
        self.assertEqual(len(TIME_PERIODS), 21)

    def test_ce_periods_are_string_centuries(self):
        # All CE periods should be parseable as integers
        for p in TIME_PERIODS[1:]:
            self.assertTrue(p.isdigit(), f"Period '{p}' is not a digit string")


if __name__ == "__main__":
    unittest.main()
