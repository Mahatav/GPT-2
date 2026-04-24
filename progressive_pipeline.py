"""Progressive philosophical training pipeline across time periods.

Trains models progressively by accumulating data from sequential time periods,
analyzing how philosophical ideas evolve across BC and CE eras.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from gpt2_pretrain import build_gpt2_from_scratch
from lm_utils import build_trainer, load_texts_from_data_dir, SimpleLMDataCollator, LMDataset, make_blocks
import tiktoken


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Time period sequence (in chronological order)
TIME_PERIODS = [
    "older (BC)",
    "100",
    "200",
    "300",
    "400",
    "500",
    "600",
    "700",
    "800",
    "900",
    "1000",
    "1100",
    "1200",
    "1300",
    "1400",
    "1500",
    "1600",
    "1700",
    "1800",
    "1900",
    "2000",
]

# Training hyperparameters
DEFAULT_CONFIG = {
    "block_size": 1024,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "max_steps": None,
    "learning_rate": 5e-4,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "save_steps": None,
    "logging_steps": 50,
    # Dynamic-step tuning knobs
    "steps_per_text": 20,
    "min_steps": 300,
    "max_steps_cap": 5000,
    # Epoch cap — hard ceiling regardless of steps_per_text
    "max_epochs": 3,
}


# ---------------------------------------------------------------------------
# Dynamic step computation
# ---------------------------------------------------------------------------

def compute_max_steps(
    num_texts: int,
    steps_per_text: int = DEFAULT_CONFIG["steps_per_text"],
    min_steps: int = DEFAULT_CONFIG["min_steps"],
    max_steps_cap: int = DEFAULT_CONFIG["max_steps_cap"],
    override: int | None = None,
) -> int:
    """Return training steps scaled to corpus size.

    If *override* is provided (via --max-steps CLI flag) that value is
    returned as-is, allowing manual control when needed.

    Formula:
        steps = clamp(num_texts * steps_per_text, min_steps, max_steps_cap)

    Examples (current defaults, steps_per_text=20):
        20 texts  ->   400 steps
        50 texts  ->  1000 steps
        100 texts ->  2000 steps
        250 texts ->  5000 steps (capped)

    Note: compute_epoch_capped_steps() applies a further hard epoch ceiling
    after the dataset is built, so the value returned here may be reduced.
    """
    if override is not None:
        return override
    steps = num_texts * steps_per_text
    return max(min_steps, min(steps, max_steps_cap))


def compute_epoch_capped_steps(
    max_steps: int,
    dataset_size: int,
    batch_size: int,
    max_epochs: int = DEFAULT_CONFIG["max_epochs"],
    min_steps: int = DEFAULT_CONFIG["min_steps"],
) -> int:
    """Apply a hard epoch ceiling to prevent memorization on small corpora.

    Without this cap, a small dataset (e.g. 8 texts → ~80 blocks) combined
    with a large max_steps value causes the trainer to cycle through the data
    dozens or hundreds of times, driving loss below 0.5 and producing
    verbatim text regurgitation at inference time.

    Args:
        max_steps:    Steps from compute_max_steps() (already clipped to cap).
        dataset_size: Number of blocks in the LMDataset.
        batch_size:   Effective batch size (per_device * grad_accum).
        max_epochs:   Hard ceiling on training epochs (default: 3).
        min_steps:    Never go below this regardless of epoch cap.

    Returns:
        Steps capped so training never exceeds max_epochs full passes.
    """
    steps_per_epoch = max(1, dataset_size // batch_size)
    epoch_cap = steps_per_epoch * max_epochs
    capped = min(max_steps, epoch_cap)
    return max(capped, min_steps)


def compute_save_steps(max_steps: int) -> int:
    """Save a checkpoint roughly every 10 % of training."""
    return max(100, max_steps // 10)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_period_directories(region: str, up_to_period: str, data_base_dir: str = "data") -> List[Path]:
    """Get list of directories to include for cumulative training up to given period."""
    if up_to_period not in TIME_PERIODS:
        raise ValueError(f"Period '{up_to_period}' not in TIME_PERIODS")

    period_idx = TIME_PERIODS.index(up_to_period)
    base_path = Path(data_base_dir) / region

    dirs = []
    for period in TIME_PERIODS[:period_idx + 1]:
        period_dir = base_path / period
        if not period_dir.exists():
            logger.warning(f"Period directory not found: {period_dir}")
            continue
        dirs.append(period_dir)

    return dirs


def load_cumulative_texts(region: str, up_to_period: str, data_base_dir: str = "data") -> List[str]:
    """Load all texts for cumulative training up to given period."""
    period_dirs = get_period_directories(region, up_to_period, data_base_dir)
    texts = []

    logger.info(f"Loading texts for {region} up to period {up_to_period}")
    for period_dir in period_dirs:
        logger.info(f"  Loading from: {period_dir}")
        try:
            period_texts = list(load_texts_from_data_dir(period_dir))
            texts.extend(period_texts)
            logger.info(f"    Loaded {len(period_texts)} texts")
        except FileNotFoundError as e:
            logger.warning(f"    Error loading texts: {e}")

    logger.info(f"Total texts loaded: {len(texts)}")
    return texts


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_period_model(
    region: str,
    period: str,
    output_base_dir: str,
    config: dict,
    data_base_dir: str = "data",
    max_steps_override: int | None = None,
) -> Path:
    """Train a model for a given region and time period."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {region.upper()} model for period: {period}")
    logger.info(f"{'='*60}")

    # Load cumulative texts
    texts = load_cumulative_texts(region, period, data_base_dir)
    if not texts:
        raise ValueError(f"No texts loaded for {region} region up to period {period}")

    # Build model, dataset, collator first so we know dataset_size
    start_time = time.time()
    model, dataset, collator = build_gpt2_from_scratch(
        texts=texts,
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        use_streaming=False,
    )

    # ── Dynamic step computation with epoch cap ───────────────────────────
    max_steps = compute_max_steps(
        num_texts=len(texts),
        steps_per_text=config.get("steps_per_text", DEFAULT_CONFIG["steps_per_text"]),
        min_steps=config.get("min_steps", DEFAULT_CONFIG["min_steps"]),
        max_steps_cap=config.get("max_steps_cap", DEFAULT_CONFIG["max_steps_cap"]),
        override=max_steps_override,
    )

    # Apply epoch cap (skip if user passed an explicit --max-steps override)
    if max_steps_override is None:
        effective_batch = (
            config.get("per_device_train_batch_size", DEFAULT_CONFIG["per_device_train_batch_size"])
            * config.get("gradient_accumulation_steps", DEFAULT_CONFIG["gradient_accumulation_steps"])
        )
        max_steps = compute_epoch_capped_steps(
            max_steps=max_steps,
            dataset_size=len(dataset),
            batch_size=effective_batch,
            max_epochs=config.get("max_epochs", DEFAULT_CONFIG["max_epochs"]),
            min_steps=config.get("min_steps", DEFAULT_CONFIG["min_steps"]),
        )

    save_steps = compute_save_steps(max_steps)
    logger.info(
        f"Corpus size: {len(texts)} texts | "
        f"Dataset: {len(dataset)} blocks | "
        f"max_steps={max_steps} | save_steps={save_steps}"
    )

    # Set up output directory
    output_dir = Path(output_base_dir) / f"progressive_{region}" / f"period_{period}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Output directory: {output_dir}")

    # Build trainer
    trainer = build_trainer(
        model,
        dataset,
        collator,
        output_dir=str(output_dir),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=config["logging_steps"],
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Find best checkpoint
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {output_dir}")

    best_checkpoint = checkpoints[-1]
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.1f} seconds")
    logger.info(f"Best checkpoint: {best_checkpoint}")

    return best_checkpoint


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def run_batch_chat_test(
    model_checkpoint: Path,
    region: str,
    period: str,
    output_base_dir: str,
) -> Path:
    """Run batch chat test for a trained model."""
    logger.info(f"Running batch chat test for {region} period {period}...")

    output_dir = Path(output_base_dir) / "progressive_evaluations" / f"period_{period}_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"chat_responses_{region}.json"

    try:
        result = subprocess.run(
            [
                sys.executable,
                "batch_chat_test.py",
                "--checkpoint",
                str(model_checkpoint),
                "--output",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.error(f"Chat test failed: {result.stderr}")
        else:
            logger.info(f"Chat test results saved to: {output_file}")
    except subprocess.TimeoutExpired:
        logger.error("Chat test timed out")
    except Exception as e:
        logger.error(f"Error running chat test: {e}")

    return output_file


def run_evaluate_bias(
    east_checkpoint: Path,
    west_checkpoint: Path,
    period: str,
    output_base_dir: str,
) -> Path:
    """Run bias evaluation comparing east and west models."""
    logger.info(f"Running bias evaluation for period {period}...")

    output_dir = Path(output_base_dir) / "progressive_evaluations" / f"period_{period}_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "evaluate_bias.py",
                "--western-path",
                str(west_checkpoint),
                "--eastern-path",
                str(east_checkpoint),
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            logger.error(f"Bias evaluation failed: {result.stderr}")
            raise RuntimeError(
                f"Bias evaluation failed for period {period} with exit code {result.returncode}"
            )
        else:
            logger.info(f"Bias evaluation completed")
    except subprocess.TimeoutExpired:
        logger.error("Bias evaluation timed out")
    except Exception as e:
        logger.error(f"Error running bias evaluation: {e}")

    return output_dir


def resolve_latest_checkpoint(period_dir: Path) -> Path:
    """Return the most recent checkpoint directory for a period output folder."""
    checkpoints = sorted(period_dir.glob("checkpoint-*"))
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {period_dir}")
    return checkpoints[-1]


def save_training_manifest(
    manifest_path: Path,
    period: str,
    east_checkpoint: Path,
    west_checkpoint: Path,
    training_config: dict,
):
    """Save training manifest for tracking completed trainings."""
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    manifest[period] = {
        "timestamp": datetime.now().isoformat(),
        "east_checkpoint": str(east_checkpoint),
        "west_checkpoint": str(west_checkpoint),
        "training_config": training_config,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Training manifest updated: {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Progressive philosophical training pipeline"
    )
    parser.add_argument(
        "--periods",
        type=str,
        nargs="+",
        default=TIME_PERIODS,
        help="Time periods to train (default: all)",
    )
    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=["east", "west"],
        help="Regions to train (default: east and west)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Base output directory",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Base data directory",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Fixed training steps per model (bypasses dynamic scheduling "
            "and epoch cap). If omitted, steps are computed dynamically "
            f"({DEFAULT_CONFIG['steps_per_text']} steps/text, "
            f"min={DEFAULT_CONFIG['min_steps']}, "
            f"cap={DEFAULT_CONFIG['max_steps_cap']}, "
            f"max_epochs={DEFAULT_CONFIG['max_epochs']})."
        ),
    )
    parser.add_argument(
        "--steps-per-text",
        type=int,
        default=DEFAULT_CONFIG["steps_per_text"],
        help=f"Steps per text file for dynamic scheduling (default: {DEFAULT_CONFIG['steps_per_text']})",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=DEFAULT_CONFIG["min_steps"],
        help=f"Minimum steps regardless of corpus size (default: {DEFAULT_CONFIG['min_steps']})",
    )
    parser.add_argument(
        "--max-steps-cap",
        type=int,
        default=DEFAULT_CONFIG["max_steps_cap"],
        help=f"Maximum steps cap regardless of corpus size (default: {DEFAULT_CONFIG['max_steps_cap']})",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_CONFIG["max_epochs"],
        help=(
            f"Hard epoch ceiling applied after dataset is built (default: {DEFAULT_CONFIG['max_epochs']}). "
            "Prevents memorization on small corpora. Ignored if --max-steps is set."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
        help="Learning rate",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation steps",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from specific period",
    )

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["learning_rate"] = args.learning_rate
    config["steps_per_text"] = args.steps_per_text
    config["min_steps"] = args.min_steps
    config["max_steps_cap"] = args.max_steps_cap
    config["max_epochs"] = args.max_epochs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    manifest_path = output_dir / "training_manifest.json"

    start_idx = 0
    if args.resume_from:
        if args.resume_from not in args.periods:
            logger.error(f"Period {args.resume_from} not in training list")
            sys.exit(1)
        start_idx = args.periods.index(args.resume_from)

    periods_to_train = args.periods[start_idx:]

    logger.info("Starting progressive training pipeline")
    logger.info(f"Periods: {periods_to_train}")
    logger.info(f"Regions: {args.regions}")
    logger.info(f"Output directory: {output_dir}")
    if args.max_steps is not None:
        logger.info(f"Step mode: FIXED ({args.max_steps} steps — epoch cap disabled)")
    else:
        logger.info(
            f"Step mode: DYNAMIC "
            f"({args.steps_per_text} steps/text, "
            f"min={args.min_steps}, cap={args.max_steps_cap}, "
            f"max_epochs={args.max_epochs})"
        )
    logger.info(f"Config: {config}")

    for period in periods_to_train:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# PERIOD: {period}")
        logger.info(f"{'#'*60}")

        checkpoints = {}

        for region in args.regions:
            try:
                checkpoint = train_period_model(
                    region=region,
                    period=period,
                    output_base_dir=str(output_dir),
                    config=config,
                    data_base_dir=str(data_dir),
                    max_steps_override=args.max_steps,
                )
                checkpoints[region] = checkpoint
            except Exception as e:
                logger.error(f"Training failed for {region} period {period}: {e}")
                continue

        # Always resolve to the latest checkpoint on disk after training so
        # evaluation runs against the most recent saved model state.
        resolved_checkpoints = {}
        for region in args.regions:
            period_dir = output_dir / f"progressive_{region}" / f"period_{period}"
            try:
                resolved_checkpoints[region] = resolve_latest_checkpoint(period_dir)
            except Exception as e:
                logger.warning(
                    f"Could not resolve latest checkpoint for {region} period {period}: {e}"
                )

        if resolved_checkpoints:
            checkpoints = resolved_checkpoints

        if not checkpoints:
            logger.error(f"No models trained for period {period}, skipping evaluation")
            continue

        if "east" in checkpoints and "west" in checkpoints:
            save_training_manifest(
                manifest_path,
                period,
                checkpoints["east"],
                checkpoints["west"],
                config,
            )

        if not args.skip_evaluation and "east" in checkpoints and "west" in checkpoints:
            logger.info(f"\nRunning evaluations for period {period}...")

            run_evaluate_bias(
                checkpoints["east"],
                checkpoints["west"],
                period,
                str(output_dir),
            )

            for region, checkpoint in checkpoints.items():
                run_batch_chat_test(
                    checkpoint,
                    region,
                    period,
                    str(output_dir),
                )
        elif not args.skip_evaluation:
            logger.warning(
                f"Skipping evaluations for period {period}: both east and west checkpoints are required. "
                f"Available checkpoints: {sorted(checkpoints.keys())}"
            )

    logger.info(f"\n{'='*60}")
    logger.info("Progressive training pipeline completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
