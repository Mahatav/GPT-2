"""
main.py — GPT-2 from scratch: end-to-end demo.

Wires together every component of the project:
  1. Tokenization   (CharTokenizer or BPETokenizer)
  2. Dataset        (TextDataset with train/val split)
  3. Model          (GPT2 with all sub-modules)
  4. Training       (Trainer with cosine-warmup schedule)
  5. Generation     (Generator with greedy / top-k / top-p sampling)

Run:
  python main.py                  # full demo on sample text
  python main.py --preset nano    # tiny model, CPU, seconds
  python main.py --preset micro   # ~10M params, char-level
"""

import argparse
import os
import torch

# ── Project modules ────────────────────────────────────────────────────────────
from config import GPT2Config, TrainConfig, MODEL_PRESETS
from tokenizer import CharTokenizer, BPETokenizer
from model import GPT2
from training import TextDataset, Trainer
from generate import Generator
from utils import Logger


# ── Demo corpus ───────────────────────────────────────────────────────────────

SAMPLE_TEXT = """\
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.

All the world's a stage, and all the men and women merely players;
They have their exits and their entrances, and one man in his time
plays many parts. At first, the infant, mewling and puking in the nurse's arms.
Then the whining schoolboy, with his satchel and shining morning face,
creeping like snail unwillingly to school.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones.

Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day
To the last syllable of recorded time,
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player
That struts and frets his hour upon the stage
And then is heard no more.
""" * 60   # repeat so the model has enough tokens to learn from


# ── Device selection ──────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def build_tokenizer(text: str, use_bpe: bool = False, bpe_vocab_size: int = 512):
    """Train and return a tokenizer on the given corpus."""
    log = Logger()
    if use_bpe:
        log.section("BPE Tokenizer")
        tok = BPETokenizer()
        tok.train(text, vocab_size=bpe_vocab_size, verbose=True)
        log.success(f"BPE vocab size: {tok.vocab_size}")
        # Quick sanity check
        sample = text[:80]
        assert tok.roundtrip(sample), "BPE encode→decode roundtrip failed!"
        log.success("Roundtrip encode→decode check passed")
        return tok
    else:
        log.section("Character Tokenizer")
        tok = CharTokenizer(text)
        log.info(f"Unique characters: {tok.vocab_size}")
        log.info(f"Vocab: {' '.join(repr(c) for c in tok.vocab_list()[:20])} ...")
        return tok


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(
    vocab_size: int,
    preset:     str = "micro",
    device:     torch.device = None,
) -> GPT2:
    """Instantiate GPT2 with a named preset, overriding vocab_size."""
    device = device or get_device()

    base_cfg = MODEL_PRESETS.get(preset)
    if base_cfg is None:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(MODEL_PRESETS)}")

    # Override vocab_size to match the actual tokenizer
    cfg = GPT2Config(
        vocab_size=vocab_size,
        block_size=base_cfg.block_size,
        n_layer=base_cfg.n_layer,
        n_head=base_cfg.n_head,
        n_embd=base_cfg.n_embd,
        dropout=base_cfg.dropout,
        bias=base_cfg.bias,
    )
    return GPT2(cfg).to(device)


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(
    model:      GPT2,
    tokenizer,
    text:       str,
    train_cfg:  TrainConfig,
    device:     torch.device,
    log:        Logger,
) -> Trainer:
    """Build datasets and run the Trainer."""

    log.section("Dataset")
    dataset = TextDataset.from_text(text, tokenizer, model.config.block_size)
    train_ds, val_ds = dataset.train_val_split(val_fraction=0.1)
    log.info(f"Total tokens  : {dataset.token_count():,}")
    log.info(f"Train examples: {len(train_ds):,}")
    log.info(f"Val   examples: {len(val_ds):,}")

    trainer = Trainer(
        model=model,
        train_data=train_ds,
        val_data=val_ds,
        config=train_cfg,
        device=device,
        logger=log,
    )
    history = trainer.train()
    return trainer


# ── Generation ────────────────────────────────────────────────────────────────

def run_generation(model: GPT2, tokenizer, device: torch.device, log: Logger):
    """Run a few generation examples with different strategies."""
    gen = Generator(model, tokenizer, device)

    log.section("Text Generation")

    prompts = [
        "To be",
        "All the world",
        "Tomorrow",
    ]

    strategies = [
        ("Greedy",      lambda p: gen.greedy(p, max_new_tokens=120)),
        ("Top-k (k=40, τ=0.8)", lambda p: gen.top_k(p, k=40, temperature=0.8, max_new_tokens=120)),
        ("Top-p (p=0.9, τ=0.9)", lambda p: gen.top_p(p, p=0.9, temperature=0.9, max_new_tokens=120)),
    ]

    for prompt in prompts:
        for strategy_name, strategy_fn in strategies:
            log.info(f"[{strategy_name}]  prompt: '{prompt}'")
            output = strategy_fn(prompt)
            log.generated_text(output, prompt=prompt)


# ── CLI + Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GPT-2 from scratch")
    p.add_argument("--preset",     default="micro",  choices=list(MODEL_PRESETS),
                   help="Model size preset (default: micro)")
    p.add_argument("--tokenizer",  default="char",   choices=["char", "bpe"],
                   help="Tokenizer type (default: char)")
    p.add_argument("--bpe-vocab",  type=int, default=512,
                   help="BPE vocab size (only used when --tokenizer=bpe)")
    p.add_argument("--max-iters",  type=int, default=2000,
                   help="Training iterations (default: 2000)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size (default: 16)")
    p.add_argument("--lr",         type=float, default=3e-4,
                   help="Peak learning rate (default: 3e-4)")
    p.add_argument("--no-train",   action="store_true",
                   help="Skip training, only show architecture info")
    p.add_argument("--text-file",  type=str, default=None,
                   help="Path to a text file to train on (overrides built-in sample)")
    p.add_argument("--checkpoint-dir", default="checkpoints",
                   help="Directory for saving checkpoints")
    return p.parse_args()


def print_architecture(model: GPT2, log: Logger):
    """Print a detailed breakdown of the model architecture."""
    cfg = model.config
    log.section("Architecture")
    log.kv("Preset",             f"GPT-2 {cfg.n_layer}L / {cfg.n_head}H / {cfg.n_embd}d")
    log.kv("Vocab size",         f"{cfg.vocab_size:,}")
    log.kv("Context window",     f"{cfg.block_size} tokens")
    log.kv("Transformer blocks", cfg.n_layer)
    log.kv("Attention heads",    cfg.n_head)
    log.kv("Head dimension",     f"{cfg.head_dim}  (n_embd / n_head)")
    log.kv("Embedding dim",      cfg.n_embd)
    log.kv("MLP hidden dim",     f"{4 * cfg.n_embd}  (4 × n_embd)")
    log.kv("Dropout",            cfg.dropout)
    log.kv("Parameters",         f"{model.num_parameters() / 1e6:.2f} M")
    log.kv("Weight tying",       "wte ↔ lm_head  (embedding = output)")

    log.section("Component shapes  (B=batch, T=seq, C=n_embd, H=n_head, D=head_dim)")
    rows = [
        ("Input token ids",         f"(B, T)"),
        ("After token + pos embed", f"(B, T, {cfg.n_embd})"),
        ("Q / K / V per head",      f"(B, {cfg.n_head}, T, {cfg.head_dim})"),
        ("Attention scores",         f"(B, {cfg.n_head}, T, T)"),
        ("After MLP",               f"(B, T, {cfg.n_embd})"),
        ("Final logits",            f"(B, T, {cfg.vocab_size})"),
    ]
    for label, shape in rows:
        log.kv(label, shape, width=30)


def main():
    args = parse_args()
    log  = Logger("GPT-2")
    log.header("GPT-2 from Scratch  |  PyTorch + NumPy")

    # ── Device ────────────────────────────────────────────────────────────
    device = get_device()
    log.info(f"Device: {device}")

    # ── Corpus ────────────────────────────────────────────────────────────
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()
        log.info(f"Loaded text from {args.text_file}  ({len(text):,} chars)")
    else:
        text = SAMPLE_TEXT
        log.info(f"Using built-in Shakespeare sample  ({len(text):,} chars)")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    use_bpe = (args.tokenizer == "bpe")
    tokenizer = build_tokenizer(text, use_bpe=use_bpe, bpe_vocab_size=args.bpe_vocab)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(
        vocab_size=tokenizer.vocab_size,
        preset=args.preset,
        device=device,
    )
    print_architecture(model, log)

    if args.no_train:
        log.info("--no-train set. Exiting after architecture display.")
        return

    # ── Training config ───────────────────────────────────────────────────
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        learning_rate=args.lr,
        warmup_iters=max(50, args.max_iters // 20),
        lr_decay_iters=args.max_iters,
        eval_interval=max(100, args.max_iters // 10),
        eval_iters=40,
        log_interval=10,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=max(500, args.max_iters // 4),
    )
    log.section("Training Config")
    log.kv("Iterations",   train_cfg.max_iters)
    log.kv("Batch size",   train_cfg.batch_size)
    log.kv("Peak LR",      train_cfg.learning_rate)
    log.kv("Warmup iters", train_cfg.warmup_iters)
    log.kv("Min LR",       train_cfg.min_lr)
    log.kv("Grad clip",    train_cfg.grad_clip)

    # ── Train ─────────────────────────────────────────────────────────────
    run_training(model, tokenizer, text, train_cfg, device, log)

    # ── Generate ──────────────────────────────────────────────────────────
    run_generation(model, tokenizer, device, log)


if __name__ == "__main__":
    main()
