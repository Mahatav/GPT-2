# vocab_bubbles.md

Generates bubble-chart visualizations of the core vocabulary used by western and eastern models. Circle size represents word frequency; in the combined chart, colour represents which model favours each word (blue = western, red = eastern, purple = shared).

## Output files

| File | Description |
|------|-------------|
| `vocab_bubbles_western.png` | Top words from the western model (blue) |
| `vocab_bubbles_eastern.png` | Top words from the eastern model (red) |
| `vocab_bubbles_combined.png` | Both models merged with colour-coded dominance |
| `bigram_bubbles_western.png` | Top bigrams from the western model (blue) |
| `bigram_bubbles_eastern.png` | Top bigrams from the eastern model (red) |
| `bigram_bubbles_combined.png` | Both models' bigrams merged |

## Stop-word filtering

Three sets of filtered words are combined into `FILTER`:

- **`PRONOUNS`** — personal and relative pronouns (I, you, he, …)
- **`PREPOSITIONS`** — spatial and logical prepositions (in, of, through, …)
- **`AUXILIARY_AND_STOPWORDS`** — auxiliaries, conjunctions, common function words

Only content words survive and appear in the charts.

## Public API

### `tokenize(text) → list[str]`

Extracts lowercase alphabetic tokens of length ≥ 2. Identical to `stats_analysis.tokenize`.

### `raw_counts(texts) → Counter`

Counts content-word frequencies across a list of texts, excluding all words in `FILTER`.

### `raw_bigrams(texts) → Counter`

Counts bigram frequencies where **both** words are content words (not in `FILTER`).

### `top_counts(counter, top_n) → Counter`

Returns a `Counter` containing only the `top_n` most common entries.

### `_blend(frac) → tuple[float, float, float]`

Maps a western-dominance fraction in `[0, 1]` to an RGB colour:

| `frac` | Colour |
|--------|--------|
| 0.0 | Red (eastern dominant) |
| 0.5 | Purple (shared equally) |
| 1.0 | Blue (western dominant) |

### `pack_circles(radii, seed=42) → list[tuple[float, float]]`

Greedy circle packing algorithm. Places circles one at a time, choosing positions adjacent to already-placed circles that minimise the distance from the origin. Deterministic given the same `seed`.

```python
positions = pack_circles([2.0, 1.5, 1.0, 0.5])
# → [(0.0, 0.0), (x1, y1), (x2, y2), (x3, y3)]
```

No two circles overlap (gap ≥ 0.05 units).

### `_scale_radii(freqs, max_r=2.0, min_r=0.18) → np.ndarray`

Linear scaling of frequency values to radius values:

```
radius = min_r + (freq - freq_min) / (freq_max - freq_min) × (max_r - min_r)
```

All-equal frequencies produce the midpoint radius `(max_r + min_r) / 2`.

### `build_charts(results, save_dir, top_n=60, top_n_bigrams=40)`

Main entry point — reads evaluations from a `results` dict (as produced by `evaluate_bias.py`) and writes all six PNG files to `save_dir`.

```python
from vocab_bubbles import load_latest_results, build_charts

results = load_latest_results("outputs/progressive_evaluations")
build_charts(results, save_dir="outputs/progressive_evaluations/vocab_bubbles")
```

### `load_latest_results(output_dir) → dict`

Finds and loads the most recently created `bias_evaluation_*.json` in `output_dir`.

## Visual design

- Background: `#0F172A` (dark navy)
- Word labels: white, bold, font size scaled to circle radius
- Combined chart legend in the lower-right corner

## Running directly

```bash
# Uses hardcoded RESULTS_DIR and OUTPUT_DIR (server paths)
python vocab_bubbles.py
```

To use with different paths, call `build_charts` programmatically:

```python
import json
from pathlib import Path
from vocab_bubbles import build_charts

with open("my_results.json") as f:
    results = json.load(f)

build_charts(results, save_dir="my_charts", top_n=80, top_n_bigrams=50)
```

## Dependencies

| Package | Role |
|---------|------|
| `matplotlib` | Figure rendering (backend: `Agg` for headless servers) |
| `numpy` | Array operations for radius scaling and colour blending |
| `collections.Counter` | Word and bigram frequency counting |
