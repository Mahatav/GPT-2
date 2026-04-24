# kl_divergence.md

Utilities for measuring how differently two GPT-2 models distribute probability over the next token. All public functions return a **percentage** (0–100 %) so results are directly comparable regardless of vocabulary size.

## Normalization

Raw KL divergence is in nats. We normalize by the theoretical maximum for a GPT-2 vocabulary:

```
pct = raw_kl_nats / log(vocab_size) × 100
    = raw_kl_nats / log(50 257)     × 100
    ≈ raw_kl_nats / 10.83           × 100
```

| Value | Meaning |
|-------|---------|
| 0 % | Identical next-token distributions at every position |
| 50 % | Moderately divergent |
| 100 % | Maximally divergent (theoretical ceiling) |

## Public API

### `compute_kl_pct(model_p, model_q, encoding, text, device, block_size=1024) → float`

**KL(P ‖ Q)** — measures how much `model_q` would be surprised by `model_p`'s predictions. Asymmetric: swapping P and Q gives a different result.

```python
from kl_divergence import compute_kl_pct
import tiktoken

enc = tiktoken.get_encoding("gpt2")
pct = compute_kl_pct(model_p, model_q, enc, "What is justice?", device="cuda")
# e.g. 23.4  → models disagree by ~23% on this text
```

Returns `float("nan")` if:
- The text tokenizes to fewer than 2 tokens.
- Any exception occurs during the forward pass.

### `compute_symmetric_kl_pct(model_a, model_b, encoding, text, device, block_size=1024) → float`

**0.5 × (KL(A‖B) + KL(B‖A))** — a symmetric measure useful when neither model is the clear reference.

```python
from kl_divergence import compute_symmetric_kl_pct

sym = compute_symmetric_kl_pct(model_east, model_west, enc, text, device="cuda")
```

### `compute_kl_report(model_west, model_east, encoding, prompt, west_output, east_output, device, block_size=1024) → dict`

Computes all five KL metrics for a single prompt/response pair and returns them in a dict ready to embed in an evaluation JSON.

```python
report = compute_kl_report(
    model_west, model_east, enc,
    prompt="What is truth?",
    west_output="Truth is correspondence to reality...",
    east_output="Truth is the cessation of craving...",
    device="cuda",
)
```

**Returned keys:**

| Key | Description |
|-----|-------------|
| `west_to_east_on_prompt` | KL(W‖E) evaluated on the shared prompt |
| `east_to_west_on_prompt` | KL(E‖W) evaluated on the shared prompt |
| `symmetric_on_prompt` | Symmetric KL on the shared prompt |
| `west_to_east_on_east_output` | KL(W‖E) on the eastern model's generated text |
| `east_to_west_on_west_output` | KL(E‖W) on the western model's generated text |
| `_note` | Human-readable reminder that all values are percentages |

## Internal helpers

These are not part of the public API but are tested independently:

### `_kl_nats(log_p, log_q) → float`

Mean per-token KL(P ‖ Q) in nats across the sequence.

- `log_p`, `log_q`: `torch.Tensor` of shape `(seq_len, vocab)` — log-softmax outputs.
- Returns a scalar `float`.

### `_to_pct(nats) → float`

Converts raw nats to a percentage, clamped to `[0, 100]`. Returns `nan` for `nan` or `inf` inputs.

### `_get_log_probs(model, input_ids) → torch.Tensor`

Runs a no-grad forward pass and returns `log_softmax(logits)` of shape `(seq_len, vocab)`.

## Dependencies

- `torch` — tensor operations and no-grad forward passes
- `math` — `log`, `isnan`, `isinf`
