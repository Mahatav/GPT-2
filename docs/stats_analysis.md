# stats_analysis.md

Measures vocabulary overlap between western and eastern model outputs using the **Bhattacharyya coefficient** — a proper probability-theoretic measure of distribution similarity.

## Metrics

### Bhattacharyya Coefficient (BC)

```
BC(P, Q) = Σ sqrt(p_i × q_i)
```

- Range: **[0, 1]**
- **1** = identical word distributions
- **0** = completely disjoint vocabularies

| BC range | Interpretation |
|----------|---------------|
| ≥ 0.9 | Very high overlap |
| 0.7–0.9 | High overlap |
| 0.5–0.7 | Moderate overlap |
| 0.3–0.5 | Low overlap |
| < 0.3 | Very low overlap |

### Bhattacharyya Distance (BD)

```
BD(P, Q) = -ln(BC(P, Q))
```

- Range: **[0, ∞)**
- **0** = identical distributions
- **∞** = no overlap (disjoint vocabularies)
- A proper metric-space distance derived from BC.

## Public API

### `tokenize(text) → list[str]`

Extracts lowercase alphabetic words of length ≥ 2. Numbers and punctuation are stripped.

```python
tokenize("Plato's Republic, 380 BC")
# → ["plato", "republic", "bc"]
```

### `get_word_distribution(texts) → dict[str, float]`

Builds a normalized word frequency distribution over a list of texts.

```python
dist = get_word_distribution(["dharma is truth", "truth is freedom"])
# → {"dharma": 0.2, "is": 0.4, "truth": 0.2, "freedom": 0.2}
```

Returns `{}` if all texts are empty or contain no extractable words.

### `bhattacharyya_coefficient(p, q) → float`

Computes BC from two `dict[str, float]` normalized distributions.

```python
p = {"soul": 0.6, "logos": 0.4}
q = {"dharma": 0.5, "soul": 0.5}
bc = bhattacharyya_coefficient(p, q)  # → sqrt(0.6 × 0.5) ≈ 0.548
```

### `bhattacharyya_distance(p, q) → float`

Returns `-log(BC)`. Returns `float("inf")` when BC = 0 (disjoint vocabularies).

### `compute_overlap_metrics(western_texts, eastern_texts) → dict`

Compute all metrics in a single call.

```python
from stats_analysis import compute_overlap_metrics

result = compute_overlap_metrics(
    western_texts=["soul virtue reason logos"],
    eastern_texts=["dharma karma nirvana"],
)
```

**Returned keys:**

| Key | Type | Description |
|-----|------|-------------|
| `bhattacharyya_coefficient` | float | BC(W, E), rounded to 4 dp |
| `bhattacharyya_distance` | float | BD(W, E), rounded to 4 dp |
| `bhattacharyya_interpretation` | str | Human-readable overlap label |
| `unique_western_words` | int | Vocabulary size of western texts |
| `unique_eastern_words` | int | Vocabulary size of eastern texts |
| `common_words` | int | Words appearing in both distributions |

### `analyze_category(category_name, evaluations) → dict`

Convenience wrapper that extracts outputs from a list of evaluation dicts and calls `compute_overlap_metrics`.

```python
evals = [
    {"western_output": "logos reason virtue", "eastern_output": "dharma karma"},
    ...
]
result = analyze_category("ethics_morality", evals)
# → {"category": "ethics_morality", "num_prompts": 1, "bhattacharyya_coefficient": ..., ...}
```

## CLI

```
python stats_analysis.py --results-file <path> [--output <path>]
```

Reads a `bias_evaluation_*.json` file produced by `evaluate_bias.py` and prints a per-category table plus overall metrics.

```bash
python stats_analysis.py \
  --results-file outputs/progressive_evaluations/bias_evaluation_20250424_143200.json \
  --output analysis_summary.json
```

**Output example:**

```
Category                       BC         BD    Overlap
-------------------------------------------------------
Body Mind                  0.8821     0.1255    high overlap
Death Immortality          0.7634     0.2703    high overlap
Ethics Morality            0.6109     0.4927    moderate overlap
...

OVERALL METRICS
Bhattacharyya Coefficient:    0.7821  (high overlap)
Bhattacharyya Distance:       0.2457
Unique Western Words:  1234
Unique Eastern Words:  1198
Common Words:           892
```

## Dependencies

- `math` — `sqrt`, `log`
- `collections.Counter` — word frequency counting
- `re` — word extraction
