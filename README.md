# GPT-2 from scratch

Built this to actually understand the architecture — no HuggingFace, no pretrained weights, just PyTorch and NumPy. Every piece is implemented explicitly: BPE tokenizer, causal attention, MLP, training loop, LR schedule, checkpointing, and generation.

---

## Run it

No text file needed — a built-in Shakespeare corpus is used by default.

```bash
# quick demo, trains in ~10 seconds on CPU
python main.py --preset nano --max-iters 500

# better output, needs a few minutes on MPS/GPU
python main.py --preset micro --max-iters 3000

# train on your own text file
python main.py --text-file mybook.txt --preset micro --max-iters 5000

# BPE tokenizer instead of character-level
python main.py --tokenizer bpe --bpe-vocab 800 --max-iters 3000

# just print the architecture, skip training
python main.py --no-train --preset small
```

---

## Project layout

```
├── main.py               entry point — wires everything together
├── config.py             GPT2Config, TrainConfig, and 6 size presets
├── generate.py           Generator: greedy / top-k / top-p sampling
│
├── tokenizer/
│   ├── base.py           abstract BaseTokenizer
│   ├── char_tokenizer.py character-level (save/load supported)
│   └── bpe_tokenizer.py  full BPE from scratch (train/encode/decode/save/load)
│
├── model/
│   ├── embeddings.py     token + positional embeddings
│   ├── attention.py      causal multi-head self-attention
│   ├── mlp.py            feed-forward block (4x expand, GELU, contract)
│   ├── block.py          transformer block (pre-norm, residuals)
│   └── gpt2.py           full model + optimizer setup
│
├── training/
│   ├── dataset.py        TextDataset — sliding window over tokens
│   ├── scheduler.py      cosine warmup, written from scratch
│   └── trainer.py        training loop with grad accumulation + checkpointing
│
├── utils/
│   ├── logger.py         coloured terminal output + inline progress bar
│   └── checkpoint.py     save / load / resume
│
└── tests/                105 tests covering every component
```

---

## Model presets

| preset | layers | heads | embd | params | notes |
|--------|--------|-------|------|--------|-------|
| nano   | 4  | 4  | 128  | ~0.8M  | CPU, seconds |
| micro  | 6  | 6  | 384  | ~10M   | good for char-level |
| small  | 12 | 12 | 768  | 117M   | original GPT-2 small |
| medium | 24 | 16 | 1024 | 345M   | |
| large  | 36 | 20 | 1280 | 762M   | |
| xl     | 48 | 25 | 1600 | 1.5B   | |

---

## Architecture

Standard GPT-2 decoder stack. Nothing exotic.

```
token ids  (B, T)
    │
    ▼
token embedding + position embedding    ← both learned, summed together
    │
    ▼
transformer block × N
    ├─ LayerNorm
    ├─ causal self-attention  +  residual
    ├─ LayerNorm
    └─ MLP (4x expand → GELU → contract)  +  residual
    │
    ▼
final LayerNorm
    │
    ▼
linear head  →  logits / cross-entropy loss
```

A few design decisions worth noting:

**Pre-norm** — LayerNorm goes before each sublayer (not after). Keeps gradients healthier early in training when the sublayer outputs are noisy.

**Weight tying** — the token embedding matrix and the output linear head share weights. Cuts the parameter count in half for the largest tensor in the model, and tends to improve perplexity.

**Residual projection scaling** — the `c_proj` layers in each attention and MLP block get their init std scaled by `1 / sqrt(2 * n_layer)`. Without this, adding N blocks worth of residuals makes the variance grow with depth.

**AdamW selective decay** — weight decay only on 2D weight matrices, not on biases or LayerNorm parameters.

---

## CLI flags

| flag | default | what it does |
|------|---------|--------------|
| `--preset` | `micro` | model size |
| `--tokenizer` | `char` | `char` or `bpe` |
| `--bpe-vocab` | `512` | BPE vocab size |
| `--max-iters` | `2000` | training iterations |
| `--batch-size` | `16` | batch size |
| `--lr` | `3e-4` | peak learning rate |
| `--text-file` | built-in Shakespeare | path to a `.txt` corpus (optional) |
| `--checkpoint-dir` | `checkpoints` | where to save `.pt` files |
| `--no-train` | `False` | print architecture and exit |

---

## Generation

Three sampling modes, all temperature-controllable:

- **greedy** — always picks the highest-prob token. Deterministic, often repetitive. Good for sanity checks.
- **top-k** — keeps the k best tokens, samples from those. Lower k = more focused.
- **top-p** (nucleus) — keeps the smallest set of tokens whose cumulative probability hits p. Adapts the pool size to how confident the model is at each step.

```python
from generate import Generator

gen = Generator(model, tokenizer)
print(gen.top_k("To be or not", k=40, temperature=0.8, max_new_tokens=200))
print(gen.top_p("To be or not", p=0.9, temperature=0.9, max_new_tokens=200))
```

---

## Checkpoints

```python
from utils.checkpoint import resume_model

model, config, iteration, val_loss = resume_model("checkpoints/ckpt_best.pt")
```

To resume training, pass the checkpoint path via `TrainConfig.resume_from` in code — there is no CLI flag for this yet.

---

## Tests

```bash
python -m pytest tests/ -v
```

105 tests, runs in ~2 seconds. Covers: config validation, tokenizer roundtrips, attention causal masking, tensor shapes at every layer, weight tying, gradient flow, scheduler math, dataset splits, and checkpoint save/load/resume.
