"""Vocabulary bubble visualizer for bias evaluation outputs.

Produces three charts from the JSON written by evaluate_bias.py:

  vocab_bubbles_western.png  — Western model core vocabulary (blue)
  vocab_bubbles_eastern.png  — Eastern model core vocabulary (red)
  vocab_bubbles_combined.png — Both models merged; bubble colour shows which
                               model favours the word (blue=western, red=eastern,
                               purple=shared equally)

Circle size always reflects word frequency.
Pronouns, prepositions, and common stop words are excluded.
"""

from __future__ import annotations

import json
import math
import re
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Hardcoded paths
# ---------------------------------------------------------------------------

RESULTS_DIR = "/home/marora15/outputs_full/progressive_evaluations"
OUTPUT_DIR  = "/home/marora15/outputs_full/progressive_evaluations/vocab_bubbles"

# ---------------------------------------------------------------------------
# Filter lists
# ---------------------------------------------------------------------------

PRONOUNS = {
    "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "which", "what",
    "this", "that", "these", "those",
}

PREPOSITIONS = {
    "a", "about", "above", "across", "after", "against", "along", "among",
    "around", "as", "at", "before", "behind", "below", "beneath", "beside",
    "between", "beyond", "by", "despite", "down", "during", "except",
    "for", "from", "in", "including", "inside", "into", "near", "of",
    "off", "on", "onto", "out", "outside", "over", "per", "since",
    "through", "throughout", "to", "toward", "towards", "under", "until",
    "unto", "up", "upon", "via", "with", "within", "without",
}

AUXILIARY_AND_STOPWORDS = {
    "am", "an", "and", "any", "are", "also", "be", "been", "being",
    "both", "but", "can", "cannot", "could", "did", "do", "does",
    "done", "don", "each", "either", "else", "even", "every",
    "few", "get", "got", "had", "has", "have", "here", "how",
    "if", "is", "just", "may", "might", "more", "most", "must",
    "neither", "no", "nor", "not", "now", "or", "other", "rather",
    "really", "s", "shall", "should", "so", "some", "such", "t",
    "than", "the", "then", "there", "though", "thus", "too",
    "was", "were", "when", "where", "whether", "while", "will",
    "would", "very", "yet", "one", "two", "three", "all", "none",
    "hence", "therefore", "however", "although", "because",
}

FILTER = PRONOUNS | PREPOSITIONS | AUXILIARY_AND_STOPWORDS

# Colour anchors
_BLUE   = np.array([0x25 / 255, 0x63 / 255, 0xEB / 255])
_PURPLE = np.array([0x7C / 255, 0x3A / 255, 0xED / 255])
_RED    = np.array([0xDC / 255, 0x26 / 255, 0x26 / 255])

BG_COLOUR = "#0F172A"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1]


def raw_counts(texts: list[str]) -> Counter:
    counter: Counter = Counter()
    for t in texts:
        counter.update(w for w in tokenize(t) if w not in FILTER)
    return counter


def raw_bigrams(texts: list[str]) -> Counter:
    counter: Counter = Counter()
    for t in texts:
        tokens = tokenize(t)
        pairs = (
            (tokens[i], tokens[i + 1])
            for i in range(len(tokens) - 1)
            if tokens[i] not in FILTER and tokens[i + 1] not in FILTER
        )
        counter.update(pairs)
    return counter


def top_counts(counter: Counter, top_n: int) -> Counter:
    return Counter(dict(counter.most_common(top_n)))


def _blend(frac: float) -> tuple:
    if frac >= 0.5:
        t = (frac - 0.5) * 2.0
        rgb = _PURPLE * (1 - t) + _BLUE * t
    else:
        t = frac * 2.0
        rgb = _RED * (1 - t) + _PURPLE * t
    return tuple(float(c) for c in rgb)


# ---------------------------------------------------------------------------
# Circle packing
# ---------------------------------------------------------------------------

def _overlaps(nx, ny, nr, positions, radii, gap=0.05):
    for (px, py), pr in zip(positions, radii):
        if math.hypot(nx - px, ny - py) < nr + pr + gap:
            return True
    return False


def pack_circles(radii: list[float], seed: int = 42) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    positions: list = []
    placed: list = []

    for r in radii:
        if not positions:
            positions.append((0.0, 0.0))
            placed.append(r)
            continue

        best = None
        best_dist = math.inf

        for _ in range(600):
            ref_idx = rng.integers(len(positions))
            cx, cy = positions[ref_idx]
            cr = placed[ref_idx]
            angle = rng.uniform(0, 2 * math.pi)
            dist = cr + r + 0.05
            nx = cx + dist * math.cos(angle)
            ny = cy + dist * math.sin(angle)

            if not _overlaps(nx, ny, r, positions, placed):
                d = math.hypot(nx, ny)
                if d < best_dist:
                    best_dist = d
                    best = (nx, ny)
                if d < r * 0.5:
                    break

        if best is None:
            best = (max(p[0] for p in positions) + r * 2.5 + 0.1, 0.0)

        positions.append(best)
        placed.append(r)

    return positions


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _scale_radii(freqs: np.ndarray, max_r=2.0, min_r=0.18) -> np.ndarray:
    f_min, f_max = freqs.min(), freqs.max()
    if f_max == f_min:
        return np.full(len(freqs), (max_r + min_r) / 2)
    return min_r + (freqs - f_min) / (f_max - f_min) * (max_r - min_r)


def _render_bubbles(ax, words, radii, colours, title,
                    title_colour="white", max_r=2.0, font_scale=7.0):
    positions = pack_circles(radii.tolist())

    for (x, y), r, word, colour in zip(positions, radii, words, colours):
        ax.add_patch(plt.Circle((x, y), r, color=colour, alpha=0.60, linewidth=0))
        fs = max(5, min(14, r * font_scale))
        ax.text(x, y, word, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white",
                multialignment="center", clip_on=True)

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    pad = max_r * 1.3
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG_COLOUR)
    ax.set_title(title, fontsize=15, fontweight="bold", color=title_colour, pad=10)


def draw_single_model(ax, counts, colour, title):
    words = list(counts.keys())
    freqs = np.array([counts[w] for w in words], dtype=float)
    radii = _scale_radii(freqs)
    order = np.argsort(-radii)
    words   = [words[i] for i in order]
    radii   = radii[order]
    colours = [colour] * len(words)
    _render_bubbles(ax, words, radii, colours, title)


def draw_combined(ax, west_raw, east_raw, top_n):
    all_words = set(west_raw.keys()) | set(east_raw.keys())
    word_data = {}
    for w in all_words:
        wf = west_raw.get(w, 0)
        ef = east_raw.get(w, 0)
        total = wf + ef
        word_data[w] = (total, wf / total)

    sorted_words = sorted(word_data, key=lambda w: word_data[w][0], reverse=True)[:top_n]
    words   = sorted_words
    freqs   = np.array([word_data[w][0] for w in words], dtype=float)
    fracs   = [word_data[w][1] for w in words]
    radii   = _scale_radii(freqs)
    order   = np.argsort(-radii)
    words   = [words[i] for i in order]
    radii   = radii[order]
    fracs   = [fracs[i] for i in order]
    colours = [_blend(f) for f in fracs]

    legend = [
        mpatches.Patch(color=tuple(_BLUE),   label="Western dominant"),
        mpatches.Patch(color=tuple(_PURPLE), label="Shared equally"),
        mpatches.Patch(color=tuple(_RED),    label="Eastern dominant"),
    ]
    _render_bubbles(ax, words, radii, colours,
                    "Combined Vocabulary — colour shows model dominance")
    ax.legend(handles=legend, loc="lower right",
              framealpha=0.3, labelcolor="white",
              facecolor=BG_COLOUR, edgecolor="none", fontsize=10)


# ---------------------------------------------------------------------------
# Bigram charts
# ---------------------------------------------------------------------------

def _prep_bigram_display(bigram_counter, top_n):
    top    = bigram_counter.most_common(top_n)
    labels = [f"{a}\n{b}" for (a, b), _ in top]
    freqs  = np.array([f for _, f in top], dtype=float)
    return labels, freqs


def draw_single_bigrams(ax, bigram_counter, colour, title, top_n):
    labels, freqs = _prep_bigram_display(bigram_counter, top_n)
    radii   = _scale_radii(freqs, max_r=2.4, min_r=0.3)
    order   = np.argsort(-radii)
    labels  = [labels[i] for i in order]
    radii   = radii[order]
    colours = [colour] * len(labels)
    _render_bubbles(ax, labels, radii, colours, title, max_r=2.4, font_scale=5.5)


def draw_combined_bigrams(ax, west_bigrams, east_bigrams, top_n):
    all_bigrams = set(west_bigrams.keys()) | set(east_bigrams.keys())
    bigram_data = {}
    for bg in all_bigrams:
        wf = west_bigrams.get(bg, 0)
        ef = east_bigrams.get(bg, 0)
        total = wf + ef
        bigram_data[bg] = (total, wf / total)

    sorted_bgs = sorted(bigram_data, key=lambda b: bigram_data[b][0], reverse=True)[:top_n]
    labels  = [f"{a}\n{b}" for a, b in sorted_bgs]
    freqs   = np.array([bigram_data[bg][0] for bg in sorted_bgs], dtype=float)
    fracs   = [bigram_data[bg][1] for bg in sorted_bgs]
    radii   = _scale_radii(freqs, max_r=2.4, min_r=0.3)
    order   = np.argsort(-radii)
    labels  = [labels[i] for i in order]
    radii   = radii[order]
    colours = [_blend(fracs[i]) for i in order]

    legend = [
        mpatches.Patch(color=tuple(_BLUE),   label="Western dominant"),
        mpatches.Patch(color=tuple(_PURPLE), label="Shared equally"),
        mpatches.Patch(color=tuple(_RED),    label="Eastern dominant"),
    ]
    _render_bubbles(ax, labels, radii, colours,
                    "Combined Bigrams — colour shows model dominance",
                    max_r=2.4, font_scale=5.5)
    ax.legend(handles=legend, loc="lower right",
              framealpha=0.3, labelcolor="white",
              facecolor=BG_COLOUR, edgecolor="none", fontsize=10)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load_latest_results(output_dir: str) -> dict:
    files = sorted(Path(output_dir).glob("bias_evaluation_*.json"))
    if not files:
        raise FileNotFoundError(f"No bias_evaluation_*.json found in {output_dir}")
    path = files[-1]
    print(f"Loading: {path}")
    with open(path) as f:
        return json.load(f)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved:  {path}")


def build_charts(results, save_dir, top_n=60, top_n_bigrams=40):
    evaluations = results.get("evaluations", [])
    if not evaluations:
        raise ValueError("No evaluations found in the results file.")

    western_texts = [e["western_output"] for e in evaluations if "western_output" in e]
    eastern_texts = [e["eastern_output"] for e in evaluations if "eastern_output" in e]

    west_raw     = raw_counts(western_texts)
    east_raw     = raw_counts(eastern_texts)
    west_top     = top_counts(west_raw, top_n)
    east_top     = top_counts(east_raw, top_n)
    west_bigrams = raw_bigrams(western_texts)
    east_bigrams = raw_bigrams(eastern_texts)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    word_sub   = f"(size = frequency · top {top_n} words · pronouns & prepositions removed)"
    bigram_sub = f"(size = frequency · top {top_n_bigrams} bigrams · both words must be content words)"

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(BG_COLOUR)
    draw_single_model(ax, west_top, "#2563EB", "Western Model — Core Vocabulary")
    fig.suptitle(word_sub, fontsize=11, color="#94A3B8", y=0.02)
    _save(fig, Path(save_dir) / "vocab_bubbles_western.png")

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(BG_COLOUR)
    draw_single_model(ax, east_top, "#DC2626", "Eastern Model — Core Vocabulary")
    fig.suptitle(word_sub, fontsize=11, color="#94A3B8", y=0.02)
    _save(fig, Path(save_dir) / "vocab_bubbles_eastern.png")

    fig, ax = plt.subplots(figsize=(16, 14))
    fig.patch.set_facecolor(BG_COLOUR)
    draw_combined(ax, west_raw, east_raw, top_n)
    fig.suptitle(word_sub, fontsize=11, color="#94A3B8", y=0.02)
    _save(fig, Path(save_dir) / "vocab_bubbles_combined.png")

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(BG_COLOUR)
    draw_single_bigrams(ax, west_bigrams, "#2563EB",
                        "Western Model — Top Bigrams", top_n_bigrams)
    fig.suptitle(bigram_sub, fontsize=11, color="#94A3B8", y=0.02)
    _save(fig, Path(save_dir) / "bigram_bubbles_western.png")

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(BG_COLOUR)
    draw_single_bigrams(ax, east_bigrams, "#DC2626",
                        "Eastern Model — Top Bigrams", top_n_bigrams)
    fig.suptitle(bigram_sub, fontsize=11, color="#94A3B8", y=0.02)
    _save(fig, Path(save_dir) / "bigram_bubbles_eastern.png")

    fig, ax = plt.subplots(figsize=(16, 14))
    fig.patch.set_facecolor(BG_COLOUR)
    draw_combined_bigrams(ax, west_bigrams, east_bigrams, top_n_bigrams)
    fig.suptitle(bigram_sub, fontsize=11, color="#94A3B8", y=0.02)
    _save(fig, Path(save_dir) / "bigram_bubbles_combined.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = load_latest_results(RESULTS_DIR)
    build_charts(results, OUTPUT_DIR)
