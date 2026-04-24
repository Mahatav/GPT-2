# batch_chat_test.md

Runs a fixed set of 24 philosophical prompts through a trained model in a single non-interactive session and saves all responses to a JSON file. Used by `progressive_pipeline.py` as part of post-training evaluation.

## Prompts

`PHILOSOPHICAL_PROMPTS` — a module-level list of 24 philosophical questions covering:

- Self and identity
- Meaning and purpose
- Ethics
- Nature of reality and knowledge
- Death and mortality
- Humanity's relationship with the cosmos
- Enlightenment and liberation

Prompts are asked sequentially in a single conversation thread, so each model response is conditioned on the accumulated prior history.

## `run_batch_chat_test(checkpoint_path, output_path, max_new_tokens=150, temperature=0.8) → dict`

Loads a model from `checkpoint_path`, runs all 24 prompts through `generate_with_history`, and writes results to `output_path`.

```python
from batch_chat_test import run_batch_chat_test

results = run_batch_chat_test(
    checkpoint_path="outputs/progressive_east/period_1000/checkpoint-1200",
    output_path="evaluations/east_1000.json",
    max_new_tokens=200,
    temperature=0.7,
)
```

**Returns** a dict matching the JSON structure written to disk.

If generation fails for any individual prompt, the error is caught and recorded as `"[ERROR: ...]"` in `response` — the remaining prompts still run.

## Output JSON structure

```json
{
  "timestamp": "2025-04-24T14:32:00.000000",
  "checkpoint": "outputs/progressive_east/period_1000/checkpoint-1200",
  "device": "cuda",
  "config": {
    "max_new_tokens": 150,
    "temperature": 0.8
  },
  "responses": [
    {
      "prompt": "What is the true nature of the self?",
      "response": "The self is ...",
      "response_length": 312
    },
    ...
  ]
}
```

## CLI

```
python batch_chat_test.py --checkpoint <path> --output <path> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Path to HuggingFace model checkpoint directory |
| `--output` | required | Path for output JSON file |
| `--max-new-tokens` | 150 | Maximum generated tokens per response |
| `--temperature` | 0.8 | Sampling temperature |

### Example

```bash
python batch_chat_test.py \
  --checkpoint outputs/progressive_east/period_2000/checkpoint-1299 \
  --output evaluations/east_2000_chat.json \
  --max-new-tokens 200 \
  --temperature 0.7
```

## Dependencies

| Module | Role |
|--------|------|
| `inference_utils` | `load_model`, `get_tokenizer`, `generate_with_history` |
| `torch` | Device detection |
