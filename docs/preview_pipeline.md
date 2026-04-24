# preview_pipeline.md

Dry-run preview of the progressive training pipeline. Prints what would be trained (periods, regions, text counts, output structure, estimated times) without loading any models or running any training.

## `preview_training_plan(periods, regions, data_dir="data")`

Prints a full human-readable plan to stdout.

```python
from preview_pipeline import preview_training_plan
from progressive_pipeline import TIME_PERIODS

preview_training_plan(
    periods=TIME_PERIODS,
    regions=["east", "west"],
    data_dir="./data",
)
```

**Output sections:**

1. **Training Configuration** — periods, regions, data directory.
2. **Total Models to Train** — `len(periods) × len(regions)`.
3. **Training Schedule** — for each period: which prior periods are included (cumulative), and how many `.txt` files each region would load.
4. **Output Structure** — directory tree showing where checkpoints and evaluations will be saved.
5. **Estimated Training Time** — rough per-model time estimates by period.
6. **Run Command** — the exact `python progressive_pipeline.py` invocation to launch training.

## CLI

```
python preview_pipeline.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--periods` | all 21 | Space-separated period labels to include |
| `--regions` | `east west` | Regions to include |
| `--data-dir` | `./data` | Root data directory |

Returns exit code `0` on success, `1` if any period label is not in `TIME_PERIODS`.

### Examples

```bash
# Preview the full pipeline
python preview_pipeline.py

# Preview only ancient and early CE periods
python preview_pipeline.py --periods "older (BC)" 100 200 300

# Preview a single region
python preview_pipeline.py --regions east
```

## Dependencies

| Module | Role |
|--------|------|
| `progressive_pipeline` | `TIME_PERIODS`, `get_period_directories`, `load_cumulative_texts` |
