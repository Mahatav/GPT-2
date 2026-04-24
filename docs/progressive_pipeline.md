# progressive_pipeline.py

Trains GPT-2 models progressively by accumulating philosophical texts from sequential time periods (BC → 2000 CE). For each period, separate east and west models are trained on the cumulative corpus, then evaluated side-by-side.

## Data layout

```
data/
├── east/
│   ├── older (BC)/   ← texts from ancient period
│   ├── 100/
│   ├── 200/
│   └── ...
└── west/
    ├── older (BC)/
    ├── 100/
    └── ...
```

Each subdirectory contains `.txt` files. Training for period `N` uses all texts from `older (BC)` through `N` (cumulative).

## Time periods

`TIME_PERIODS` (module constant) — ordered list of 21 period labels:

```python
["older (BC)", "100", "200", ..., "2000"]
```

## Step scheduling

Training steps are computed dynamically from corpus size, with two layers of protection against over-fitting on small corpora:

### `compute_max_steps(num_texts, steps_per_text, min_steps, max_steps_cap, override)`

```
steps = clamp(num_texts × steps_per_text, min_steps, max_steps_cap)
```

| num_texts | steps (defaults) |
|-----------|-----------------|
| 20        | 400             |
| 50        | 1 000           |
| 100       | 2 000           |
| 250       | 5 000 (capped)  |

`override` bypasses the formula entirely (useful for `--max-steps` CLI flag).

### `compute_epoch_capped_steps(max_steps, dataset_size, batch_size, max_epochs, min_steps)`

Applies a hard ceiling so the trainer never loops through the dataset more than `max_epochs` times. Prevents verbatim memorization on very small corpora (< ~10 texts).

```
steps_per_epoch = max(1, dataset_size // batch_size)
epoch_cap       = steps_per_epoch × max_epochs
result          = max(min(max_steps, epoch_cap), min_steps)
```

### `compute_save_steps(max_steps) → int`

Returns `max(100, max_steps // 10)` — saves a checkpoint roughly every 10 % of training.

## Data helpers

### `get_period_directories(region, up_to_period, data_base_dir) → List[Path]`

Returns the list of existing period directories for `region` from the beginning of `TIME_PERIODS` through `up_to_period` (inclusive). Missing directories are silently skipped with a warning.

### `load_cumulative_texts(region, up_to_period, data_base_dir) → List[str]`

Loads all `.txt` files from the directories returned by `get_period_directories`. Returns a flat list of text strings.

## Training

### `train_period_model(region, period, output_base_dir, config, data_base_dir, max_steps_override) → Path`

Full training run for one (region, period) pair:

1. Loads cumulative texts.
2. Calls `build_gpt2_from_scratch` (from `gpt2_pretrain`) to get model, dataset, collator.
3. Computes dynamic step schedule.
4. Calls `build_trainer` and runs `.train()`.
5. Returns path to the latest checkpoint.

Output is written to `{output_base_dir}/progressive_{region}/period_{period}/`.

## Evaluation helpers

### `run_batch_chat_test(model_checkpoint, region, period, output_base_dir) → Path`

Invokes `batch_chat_test.py` as a subprocess with the given checkpoint. Results are saved to `{output_base_dir}/progressive_evaluations/period_{period}_evaluation/chat_responses_{region}.json`.

### `run_evaluate_bias(east_checkpoint, west_checkpoint, period, output_base_dir) → Path`

Invokes `evaluate_bias.py` as a subprocess comparing the east and west checkpoints. Raises `RuntimeError` if the subprocess exits with a non-zero code.

### `resolve_latest_checkpoint(period_dir) → Path`

Returns the highest-numbered `checkpoint-*` subdirectory inside `period_dir`. Raises `RuntimeError` if none exist.

### `save_training_manifest(manifest_path, period, east_checkpoint, west_checkpoint, training_config)`

Appends an entry to `training_manifest.json` recording the timestamp, checkpoint paths, and config for each completed period. Safe to call incrementally — loads and merges with existing content.

## CLI

```
python progressive_pipeline.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--periods` | all 21 | Space-separated period labels to train |
| `--regions` | `east west` | Regions to train |
| `--output-dir` | `./outputs` | Root output directory |
| `--data-dir` | `./data` | Root data directory |
| `--max-steps` | dynamic | Fixed steps per model (bypasses dynamic scheduling) |
| `--steps-per-text` | 20 | Steps per text file (dynamic mode) |
| `--min-steps` | 300 | Minimum steps (dynamic mode) |
| `--max-steps-cap` | 5000 | Maximum steps cap (dynamic mode) |
| `--max-epochs` | 3 | Hard epoch ceiling (dynamic mode) |
| `--learning-rate` | 5e-4 | AdamW learning rate |
| `--skip-evaluation` | off | Skip `evaluate_bias` and `batch_chat_test` |
| `--resume-from` | — | Resume pipeline from the specified period |

### Examples

```bash
# Full pipeline, all periods
python progressive_pipeline.py

# Train only 100–300 CE, skip evaluation
python progressive_pipeline.py --periods 100 200 300 --skip-evaluation

# Fixed 500-step budget per model
python progressive_pipeline.py --max-steps 500

# Resume after a crash at "500"
python progressive_pipeline.py --resume-from 500
```

## Dependencies

| Module | Role |
|--------|------|
| `gpt2_pretrain` | `build_gpt2_from_scratch` — constructs model, dataset, collator |
| `lm_utils` | `build_trainer`, `load_texts_from_data_dir` |
| `transformers` | `GPT2Config`, `GPT2LMHeadModel` (imported but unused directly) |
| `tiktoken` | GPT-2 tokenizer (used inside `gpt2_pretrain`) |

## Outputs

```
outputs/
├── training_manifest.json          ← per-period checkpoint registry
├── progressive_east/
│   └── period_{period}/
│       └── checkpoint-{N}/         ← HuggingFace model checkpoint
├── progressive_west/
│   └── period_{period}/
│       └── checkpoint-{N}/
└── progressive_evaluations/
    └── period_{period}_evaluation/
        ├── bias_evaluation_*.json
        ├── chat_responses_east.json
        └── chat_responses_west.json
```
