# lm_utils.md

Shared building blocks for causal language-model pretraining: dataset classes, a data collator, a streaming dataset, a trainer factory, and a text loader.

## `make_blocks(token_ids, block_size) → List[List[int]]`

Splits a flat list of token IDs into fixed-length chunks. The last partial chunk is silently dropped.

```python
from lm_utils import make_blocks

blocks = make_blocks(list(range(25)), block_size=8)
# → [[0..7], [8..15], [16..23]]  (24 tokens; last token dropped)
```

## `LMExample`

Dataclass holding a single training example.

```python
@dataclass
class LMExample:
    input_ids: List[int]
```

## `LMDataset`

A standard `torch.utils.data.Dataset` that wraps a list of fixed-length token blocks.

```python
from lm_utils import LMDataset

dataset = LMDataset(blocks)   # blocks: List[List[int]]
len(dataset)                  # number of blocks
dataset[0]                    # LMExample(input_ids=[...])
```

## `SimpleLMDataCollator`

Collates a batch of `LMExample` objects (or raw tensors from `StreamingLMDataset`) into the dict that `Trainer` expects.

```python
from lm_utils import SimpleLMDataCollator

collator = SimpleLMDataCollator(pad_id=50256)  # GPT-2 EOS token
batch = collator([example1, example2, ...])
# batch keys: "input_ids", "labels", "attention_mask"
# labels == input_ids (standard causal LM objective)
# attention_mask = 1 for real tokens, 0 for padding
```

Accepts two input formats:
- `List[LMExample]` — from `LMDataset`
- `List[torch.Tensor]` — from `StreamingLMDataset`

## `StreamingLMDataset`

A memory-efficient `IterableDataset` that reads `.txt` files on the fly. Suitable for large corpora that don't fit in RAM.

```python
from lm_utils import StreamingLMDataset
import tiktoken

enc = tiktoken.get_encoding("gpt2")
ds = StreamingLMDataset(
    data_dir="data/east",
    tokenizer=enc,
    eos_id=enc.eot_token,
    block_size=1024,
    shuffle_buffer=10_000,
)
```

**Constructor parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | — | Path to a directory containing `.txt` files (searched recursively) |
| `tokenizer` | — | Any tokenizer with an `.encode(text)` method |
| `eos_id` | — | Token ID appended between documents |
| `block_size` | 1024 | Fixed context window size in tokens |
| `shuffle_buffer` | 10 000 | Number of blocks buffered for shuffling |
| `subdir` | `None` | Optional subdirectory appended to `data_dir` |

Blocks are yielded as `torch.Tensor` of dtype `long` and shape `(block_size,)`.

## `build_trainer(model, dataset, collator, output_dir, ...) → transformers.Trainer`

Constructs a HuggingFace `Trainer` configured for RTX 6000 Ada GPUs with sensible defaults.

```python
from lm_utils import build_trainer

trainer = build_trainer(
    model,
    dataset,
    collator,
    output_dir="./outputs/run_1",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,   # effective batch = 32
    learning_rate=5e-4,
    max_steps=2000,
    save_steps=200,
    logging_steps=50,
)
trainer.train()
```

**Key defaults:**

| Setting | Value | Rationale |
|---------|-------|-----------|
| `bf16` | auto-detected | Preferred on Ada (bfloat16 avoids overflow vs fp16) |
| `fp16` | fallback | Used when bf16 unsupported but CUDA available |
| `optim` | `adamw_torch_fused` | Faster fused kernel on Ada |
| `warmup_steps` | 100 | Short warmup for fine-tuning-scale runs |
| `weight_decay` | 0.1 | Standard GPT-style regularization |
| `max_grad_norm` | 1.0 | Gradient clipping for stability |
| `gradient_checkpointing` | `False` | Disabled by default; enable if OOM |
| `ddp_find_unused_parameters` | `False` | Required for multi-GPU efficiency |
| `dataloader_num_workers` | 4 | Parallel data loading |

TensorBoard logging is enabled only if `tensorboard` or `tensorboardX` is importable.

**Full parameter list:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `per_device_train_batch_size` | 8 | Batch size per GPU |
| `gradient_accumulation_steps` | 4 | Steps before optimizer update |
| `learning_rate` | 5e-4 | Peak LR for AdamW |
| `max_steps` | 100 000 | Total training steps |
| `save_steps` | 1 000 | Checkpoint frequency |
| `logging_steps` | 10 | Log frequency |
| `save_total_limit` | 3 | Maximum checkpoints to keep |

## `load_texts_from_data_dir(data_dir) → Iterable[str]`

Yields the contents of every `.txt` file under `data_dir` (recursive).

```python
from lm_utils import load_texts_from_data_dir

for text in load_texts_from_data_dir("data/west/older (BC)"):
    print(len(text), "chars")
```

Raises `FileNotFoundError` if the directory doesn't exist or contains no `.txt` files.

## Dependencies

- `torch` — tensor operations and `IterableDataset`
- `transformers` — `Trainer`, `TrainingArguments`
