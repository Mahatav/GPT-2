# inference_utils.md

Shared utilities for loading trained GPT-2 checkpoints and generating text, used by both `batch_chat_test.py` and `evaluate_bias.py`.

## `load_model(checkpoint_path, device=None) → GPT2LMHeadModel`

Loads a HuggingFace GPT-2 model from a checkpoint directory and places it in eval mode on the specified device.

```python
from inference_utils import load_model

model = load_model("outputs/progressive_west/period_2000/checkpoint-2118", device="cuda")
```

If `device` is `None`, CUDA is used when available; otherwise CPU.

## `get_tokenizer() → tiktoken.Encoding`

Returns the GPT-2 BPE encoding via tiktoken. All models in this project use the same tokenizer.

```python
from inference_utils import get_tokenizer

enc = get_tokenizer()
tokens = enc.encode("What is justice?")
```

## `generate(model, encoding, prompt, ...) → str`

Generates a continuation of `prompt` using HuggingFace's `.generate()`. Returns only the **newly generated tokens** (the prompt is stripped).

```python
from inference_utils import generate

text = generate(
    model, enc, "What is the meaning of life?",
    max_new_tokens=150,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.3,
    no_repeat_ngram_size=4,
    device="cuda",
)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 200 | Maximum tokens to generate beyond the prompt |
| `temperature` | 0.8 | Sampling temperature (lower = more focused) |
| `repetition_penalty` | 1.0 | > 1.0 penalises already-seen tokens |
| `no_repeat_ngram_size` | 0 | Hard block on repeating any N-gram of this size (0 = disabled) |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `top_k` | 0 | Top-k sampling (0 = disabled) |
| `do_sample` | `True` | Stochastic sampling; set `False` for greedy |
| `device` | inferred | Device string; inferred from model parameters if `None` |

Both `eos_token_id` and `pad_token_id` are set to the GPT-2 end-of-text token (`50256`).

## `generate_with_history(model, encoding, prompt, history="", ...) → tuple[str, str]`

Generates a reply in a multi-turn conversation format, returning `(reply, updated_history)`.

```python
from inference_utils import generate_with_history

reply, history = generate_with_history(model, enc, "What is justice?")
reply2, history = generate_with_history(model, enc, "Can it be measured?", history=history)
```

**Prompt format:**

```
{history}Human: {prompt}
Assistant:
```

The reply is the generated text up to (but not including) the next `Human:` token, stripped of whitespace.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `history` | `""` | Accumulated conversation so far |
| `max_new_tokens` | 200 | Maximum new tokens per turn |
| `temperature` | 0.8 | Sampling temperature |
| `top_p` | 0.9 | Nucleus sampling |
| `device` | inferred | Device string |

## Dependencies

- `tiktoken` — GPT-2 BPE tokenizer
- `torch` — tensor construction and device management
- `transformers` — `GPT2LMHeadModel`
