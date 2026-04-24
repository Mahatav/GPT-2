# evaluate_bias.md

Compares a western-trained and an eastern-trained GPT-2 model across 14 philosophical categories (280+ prompts total), producing a JSON report with per-response text metrics, cross-perplexity, and KL divergence.

## Philosophical categories

| Category | Prompts |
|----------|---------|
| `self_identity` | 20 |
| `purpose_meaning` | 20 |
| `ethics_morality` | 20 |
| `reality_existence` | 20 |
| `knowledge_truth` | 20 |
| `death_immortality` | 21 |
| `nature_universe` | 23 |
| `enlightenment_liberation` | 20 |
| `free_will_fate` | 20 |
| `good_evil` | 20 |
| `society_justice` | 20 |
| `death_meaning` | 20 |
| `body_mind` | 20 |
| `wisdom_truth` | 20 |

## Generation config

`GENERATION_CONFIG` (module constant) — tuned to suppress looping on small models:

```python
GENERATION_CONFIG = {
    "temperature": 0.4,
    "repetition_penalty": 1.3,
    "no_repeat_ngram_size": 4,
    "top_p": 0.9,
    "do_sample": True,
}
```

## Marker vocabulary

`WESTERN_MARKERS` and `EASTERN_MARKERS` — curated lists of tradition-specific terms used to measure conceptual bias in model outputs (e.g. `"logos"`, `"kant"` vs `"dharma"`, `"nirvana"`). No term appears in both lists.

## Text analysis helpers

### `compute_repetition_score(text, ngram_size=4) → float`

Returns the fraction of N-grams that are repeated. 0 = no repetition, approaching 1 = near-complete looping.

### `compute_type_token_ratio(text) → float`

`unique_tokens / total_tokens`. Higher = more lexically diverse output.

### `compute_concept_frequencies(text) → dict`

Counts how many eastern/western marker words appear and computes their ratio.

```python
{
    "eastern_marker_count": 3,
    "western_marker_count": 1,
    "eastern_ratio": 0.75,
    "western_ratio": 0.25,
}
```

### `compute_perplexity(model, encoding, text, device, block_size=1024) → float`

Per-token cross-entropy perplexity of `model` on `text`. Returns `float("inf")` on error or if the text is too short.

### `analyze_single_output(text) → dict`

Combines all metrics into a single dict:

```python
{
    "length_chars": 412,
    "length_words": 73,
    "repetition_score": 0.04,
    "type_token_ratio": 0.68,
    "concept_frequencies": { ... },
}
```

## `evaluate_models(western_path, eastern_path, output_dir, ...) → dict`

Main evaluation loop. For each prompt in every category:

1. Generates a response from each model.
2. Computes cross-perplexity (how surprised is each model by the other's output?).
3. Calls `compute_kl_report` (from `kl_divergence`) for KL metrics.
4. Calls `analyze_single_output` for text quality metrics.

Results are written to `{output_dir}/bias_evaluation_{timestamp}.json`.

## `analyze_bias(results) → None`

Prints a summary table to stdout with per-category averages:

```
Category                  W-Rep  E-Rep  W-TTR  E-TTR  W→E PPL  E→W PPL  W East%  E East%  KL W→E%  KL E→W%  KL Sym%
```

Column guide:
- **W-Rep / E-Rep** — repetition score (lower is better)
- **W-TTR / E-TTR** — type-token ratio (higher is better)
- **W→E PPL / E→W PPL** — cross-perplexity (higher = more surprised by the other's text)
- **W East% / E East%** — fraction of philosophical markers that are eastern-tradition
- **KL W→E% / KL E→W% / KL Sym%** — KL divergence percentages from `kl_divergence.py`

## `run_stats_analysis(results) → None`

Prints Bhattacharyya coefficient and distance per category and overall (delegates to `stats_analysis.py`).

## Output JSON structure

```json
{
  "timestamp": "...",
  "western_model": "...",
  "eastern_model": "...",
  "config": { "max_tokens": 150, "temperature": 0.4, ... },
  "evaluations": [
    {
      "category": "self_identity",
      "prompt": "What is the true nature of the self?",
      "western_output": "...",
      "eastern_output": "...",
      "western_metrics": { "length_chars": 312, "repetition_score": 0.02, ... },
      "eastern_metrics": { ... },
      "cross_perplexity": {
        "western_model_on_eastern_text": 45.2,
        "eastern_model_on_western_text": 38.7
      },
      "kl_divergence": {
        "west_to_east_on_prompt": 18.4,
        "east_to_west_on_prompt": 21.1,
        "symmetric_on_prompt": 19.7,
        "west_to_east_on_east_output": 24.3,
        "east_to_west_on_west_output": 19.8,
        "_note": "all values are percentages ..."
      }
    },
    ...
  ]
}
```

## CLI

```
python evaluate_bias.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--western-path` | auto-detected | Path to western model checkpoint |
| `--eastern-path` | auto-detected | Path to eastern model checkpoint |
| `--output-dir` | auto-detected | Directory for result JSON |
| `--max-tokens` | 150 | Max generated tokens per response |
| `--temperature` | 0.4 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--top-k` | 0 | Top-k sampling (0 = disabled) |
| `--repetition-penalty` | 1.3 | Repetition penalty |
| `--analyze-only` | off | Skip generation; analyze the most recent existing results file |

### Examples

```bash
# Full evaluation with auto-detected checkpoints
python evaluate_bias.py

# Analyze existing results without re-running generation
python evaluate_bias.py --analyze-only

# Custom checkpoints
python evaluate_bias.py \
  --western-path outputs/progressive_west/period_1000/checkpoint-800 \
  --eastern-path outputs/progressive_east/period_1000/checkpoint-650 \
  --output-dir outputs/evaluations/1000
```

## Path auto-detection

The script searches for `outputs_full` in the following order:
1. `~/outputs_full` (server)
2. `./outputs` (local)
3. `/home/marora15/outputs_full` (absolute server path)

## Dependencies

| Module | Role |
|--------|------|
| `inference_utils` | `load_model`, `get_tokenizer`, `generate` |
| `kl_divergence` | `compute_kl_report` |
| `stats_analysis` | `compute_overlap_metrics`, `analyze_category` |
