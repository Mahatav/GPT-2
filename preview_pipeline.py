"""Preview script for progressive training pipeline.

Shows what will be trained without actually running training.
"""

import argparse
from pathlib import Path

from progressive_pipeline import (
    TIME_PERIODS,
    get_period_directories,
    load_cumulative_texts,
)


def preview_training_plan(
    periods: list,
    regions: list,
    data_dir: str = "data",
):
    """Show preview of training plan without running training."""
    print("\n" + "="*70)
    print("PROGRESSIVE TRAINING PIPELINE PREVIEW")
    print("="*70)

    print(f"\nTraining Configuration:")
    print(f"  Periods: {periods}")
    print(f"  Regions: {regions}")
    print(f"  Data Directory: {data_dir}")

    total_trainings = len(periods) * len(regions)
    print(f"\nTotal Models to Train: {total_trainings} ({len(periods)} periods × {len(regions)} regions)")

    print("\nTraining Schedule:")
    print("-" * 70)

    for i, period in enumerate(periods, 1):
        print(f"\n[{i}/{len(periods)}] Period: {period}")

        # Get directories for this period
        try:
            dirs = get_period_directories(regions[0] if regions else "east", period, data_dir)
            total_dirs = len(dirs)
            print(f"  Cumulative time periods: {total_dirs}")
            print(f"  Periods included: {' → '.join([d.name for d in dirs])}")
        except Exception as e:
            print(f"  Error: {e}")
            continue

        # Show text counts per region
        for region in regions:
            try:
                texts = load_cumulative_texts(region, period, data_dir)
                print(f"  {region.upper():5} - {len(texts):3} text files loaded")
            except Exception as e:
                print(f"  {region.upper():5} - Error: {e}")

    print("\nOutput Structure:")
    print("-" * 70)
    print("""
outputs/
├── training_manifest.json
├── progressive_east/
│   └── period_{period}/checkpoint-500/  (for each period)
├── progressive_west/
│   └── period_{period}/checkpoint-500/  (for each period)
└── progressive_evaluations/
    └── period_{period}_evaluation/
        ├── bias_evaluation_*.json
        ├── chat_responses_east.json
        └── chat_responses_west.json
""")

    print("\nEstimated Training Time (per model):")
    print("-" * 70)
    period_times = {
        "older (BC)": "5-10 min",
        "100": "10-15 min",
        "200": "15-20 min",
        "300": "20-25 min",
        "500": "30-40 min",
        "1000": "40-50 min",
        "1500": "50-70 min",
        "2000": "70-90 min",
    }

    for period in periods:
        est_time = period_times.get(period, "variable")
        print(f"  {period:10} - {est_time}")

    total_periods_count = len(periods)
    region_count = len(regions)
    print(f"\nEstimate for full pipeline ({total_periods_count} periods × {region_count} regions):")
    print(f"  Minimum: ~{total_periods_count * region_count * 5} minutes")
    print(f"  Maximum: ~{total_periods_count * region_count * 90} minutes")

    print("\n" + "="*70)
    print("To start training, run:")
    print(f"  python progressive_pipeline.py --periods {' '.join(periods)}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preview progressive training pipeline without running it"
    )
    parser.add_argument(
        "--periods",
        type=str,
        nargs="+",
        default=TIME_PERIODS,
        help="Time periods to preview (default: all)",
    )
    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=["east", "west"],
        help="Regions to preview (default: east and west)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Base data directory",
    )

    args = parser.parse_args()

    # Validate periods
    for period in args.periods:
        if period not in TIME_PERIODS:
            print(f"Error: Period '{period}' not found in TIME_PERIODS")
            return 1

    preview_training_plan(args.periods, args.regions, args.data_dir)
    return 0


if __name__ == "__main__":
    exit(main())
