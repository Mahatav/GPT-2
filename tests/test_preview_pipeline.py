"""Tests for preview_pipeline.py — training plan preview (no actual training)."""

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Stub out optional heavy dependencies pulled in transitively through
# preview_pipeline → progressive_pipeline.
for _mod in ("gpt2_pretrain", "tiktoken",
             "transformers", "transformers.GPT2Config", "transformers.GPT2LMHeadModel"):
    sys.modules.setdefault(_mod, MagicMock())

from preview_pipeline import preview_training_plan


def _capture(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), capturing stdout, and return (result, printed_text)."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        result = fn(*args, **kwargs)
    finally:
        sys.stdout = old
    return result, buf.getvalue()


class TestPreviewTrainingPlan(unittest.TestCase):

    def _make_data(self, tmpdir, region, periods):
        base = Path(tmpdir) / region
        for p in periods:
            period_dir = base / p
            period_dir.mkdir(parents=True, exist_ok=True)
            (period_dir / "sample.txt").write_text("text", encoding="utf-8")

    def test_prints_period_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, "east", ["older (BC)", "100"])
            self._make_data(tmpdir, "west", ["older (BC)", "100"])
            _, output = _capture(
                preview_training_plan,
                periods=["older (BC)", "100"],
                regions=["east", "west"],
                data_dir=tmpdir,
            )
            self.assertIn("older (BC)", output)
            self.assertIn("100", output)

    def test_prints_region_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, "east", ["older (BC)"])
            self._make_data(tmpdir, "west", ["older (BC)"])
            _, output = _capture(
                preview_training_plan,
                periods=["older (BC)"],
                regions=["east", "west"],
                data_dir=tmpdir,
            )
            self.assertIn("EAST", output)
            self.assertIn("WEST", output)

    def test_shows_total_models_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, "east", ["older (BC)", "100"])
            self._make_data(tmpdir, "west", ["older (BC)", "100"])
            _, output = _capture(
                preview_training_plan,
                periods=["older (BC)", "100"],
                regions=["east", "west"],
                data_dir=tmpdir,
            )
            # 2 periods × 2 regions = 4 models
            self.assertIn("4", output)

    def test_shows_output_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, "east", ["older (BC)"])
            _, output = _capture(
                preview_training_plan,
                periods=["older (BC)"],
                regions=["east"],
                data_dir=tmpdir,
            )
            self.assertIn("outputs/", output)
            self.assertIn("progressive_east", output)

    def test_shows_run_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, "east", ["older (BC)"])
            _, output = _capture(
                preview_training_plan,
                periods=["older (BC)"],
                regions=["east"],
                data_dir=tmpdir,
            )
            self.assertIn("progressive_pipeline.py", output)

    def test_handles_missing_period_directories(self):
        """If a period dir doesn't exist, preview should still run (no crash)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't create any dirs — preview should log warnings but not raise
            _, output = _capture(
                preview_training_plan,
                periods=["older (BC)"],
                regions=["east"],
                data_dir=tmpdir,
            )
            # Should still print the header
            self.assertIn("PROGRESSIVE TRAINING PIPELINE PREVIEW", output)

    def test_single_period_single_region(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, "east", ["older (BC)"])
            _, output = _capture(
                preview_training_plan,
                periods=["older (BC)"],
                regions=["east"],
                data_dir=tmpdir,
            )
            self.assertIn("[1/1]", output)

    def test_prints_text_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, "east", ["older (BC)"])
            self._make_data(tmpdir, "west", ["older (BC)"])
            # Each period has 1 txt file, so cumulative count should be 1
            _, output = _capture(
                preview_training_plan,
                periods=["older (BC)"],
                regions=["east", "west"],
                data_dir=tmpdir,
            )
            self.assertIn("1", output)  # 1 text file loaded


class TestPreviewPipelineMain(unittest.TestCase):

    def test_invalid_period_returns_nonzero(self):
        from preview_pipeline import main
        with patch("sys.argv", ["preview_pipeline.py", "--periods", "INVALID_PERIOD"]):
            result = main()
        self.assertNotEqual(result, 0)

    def test_valid_periods_return_zero(self):
        from preview_pipeline import main
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.argv", [
                "preview_pipeline.py",
                "--periods", "older (BC)",
                "--data-dir", tmpdir,
            ]):
                with patch("builtins.print"):
                    result = main()
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
