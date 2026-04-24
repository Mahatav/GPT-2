"""Statistical analysis for model output distributions.

Computes the Bhattacharyya coefficient and distance to measure overlap
between Western and Eastern model output distributions [0=no overlap, 1=complete overlap].

Usage:
    python3 stats_analysis.py --results-file outputs/bias_evaluation_5000/bias_evaluation_*.json
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


def tokenize(text: str) -> list[str]:
    """Tokenize text into words."""
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1]


def get_word_distribution(texts: list[str]) -> dict[str, float]:
    """Compute normalized word frequency distribution."""
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    total = sum(counter.values())
    if total == 0:
        return {}

    return {word: count / total for word, count in counter.items()}


def bhattacharyya_coefficient(p: dict[str, float], q: dict[str, float]) -> float:
    """Compute Bhattacharyya coefficient BC(P, Q) = Σ sqrt(p_i * q_i).

    Measures overlap between two discrete probability distributions.
    Value in [0, 1]: 0 = no overlap, 1 = complete overlap.
    Assumes p and q are already normalized (sum to 1).
    """
    all_words = set(p.keys()) | set(q.keys())
    return sum(math.sqrt(p.get(w, 0.0) * q.get(w, 0.0)) for w in all_words)


def bhattacharyya_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Compute Bhattacharyya distance D_B = -ln(BC(P, Q)).

    Value in [0, inf): 0 = identical distributions, inf = no overlap.
    """
    bc = bhattacharyya_coefficient(p, q)
    return -math.log(bc) if bc > 0 else float("inf")


def _interpret_bc(bc: float) -> str:
    if bc >= 0.9: return "very high overlap"
    if bc >= 0.7: return "high overlap"
    if bc >= 0.5: return "moderate overlap"
    if bc >= 0.3: return "low overlap"
    return "very low overlap"


def compute_overlap_metrics(western_texts: list[str], eastern_texts: list[str]) -> dict:
    """Compute all statistical metrics between Western and Eastern model outputs."""

    west_dist = get_word_distribution(western_texts)
    east_dist = get_word_distribution(eastern_texts)

    bc = bhattacharyya_coefficient(west_dist, east_dist)
    metrics = {
        "bhattacharyya_coefficient": round(bc, 4),
        "bhattacharyya_distance": round(bhattacharyya_distance(west_dist, east_dist), 4),
        "bhattacharyya_interpretation": _interpret_bc(bc),
        "unique_western_words": len(west_dist),
        "unique_eastern_words": len(east_dist),
        "common_words": len(set(west_dist.keys()) & set(east_dist.keys())),
    }

    return metrics


def analyze_category(category_name: str, evaluations: list[dict]) -> dict:
    """Analyze a single category."""
    west_texts = [e["western_output"] for e in evaluations if "western_output" in e]
    east_texts = [e["eastern_output"] for e in evaluations if "eastern_output" in e]

    return {
        "category": category_name,
        "num_prompts": len(west_texts),
        **compute_overlap_metrics(west_texts, east_texts)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis of model output distributions"
    )
    parser.add_argument(
        "--results-file",
        required=True,
        help="Path to bias_evaluation JSON file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file for results (default: print to console)",
    )
    args = parser.parse_args()

    with open(args.results_file) as f:
        results = json.load(f)

    evaluations = results.get("evaluations", [])

    print("=" * 70)
    print("STATISTICAL ANALYSIS OF MODEL OUTPUT DISTRIBUTIONS")
    print("=" * 70)

    category_results = {}
    for category in set(e.get("category", "unknown") for e in evaluations):
        cat_evals = [e for e in evaluations if e.get("category") == category]
        category_results[category] = analyze_category(category, cat_evals)

    print(f"\n{'Category':<30} {'BC':>8} {'BD':>8}  {'Overlap'}")
    print("-" * 60)

    for cat, res in sorted(category_results.items()):
        print(f"{cat.replace('_', ' ').title():<30} "
              f"{res['bhattacharyya_coefficient']:>8.4f} {res['bhattacharyya_distance']:>8.4f}  "
              f"{res['bhattacharyya_interpretation']}")

    print("\n" + "=" * 70)
    print("OVERALL METRICS (All Categories Combined)")
    print("=" * 70)

    west_all = [e["western_output"] for e in evaluations if "western_output" in e]
    east_all = [e["eastern_output"] for e in evaluations if "eastern_output" in e]
    overall = compute_overlap_metrics(west_all, east_all)

    print(f"\nBhattacharyya Coefficient:    {overall['bhattacharyya_coefficient']:.4f}  ({overall['bhattacharyya_interpretation']})")
    print(f"Bhattacharyya Distance:       {overall['bhattacharyya_distance']:.4f}")
    print(f"\nUnique Western Words: {overall['unique_western_words']}")
    print(f"Unique Eastern Words: {overall['unique_eastern_words']}")
    print(f"Common Words:          {overall['common_words']}")

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
Bhattacharyya Coefficient (BC = Σ sqrt(p_i * q_i)):
  - 0 = no overlap, 1 = complete overlap
  - Higher = more similar vocabularies

Bhattacharyya Distance (BD = -ln(BC)):
  - 0 = identical distributions, inf = no overlap
  - Proper metric-space distance derived from BC
""")

    output_data = {
        "results_file": args.results_file,
        "overall_metrics": overall,
        "category_metrics": category_results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
