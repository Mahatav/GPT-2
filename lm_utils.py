"""Shared utilities for causal LM pretraining."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import Trainer, TrainingArguments


@dataclass
class LMExample:
    input_ids: List[int]


class LMDataset(Dataset):
    def __init__(self, input_id_blocks: List[List[int]]):
        self._data = input_id_blocks

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> LMExample:
        return LMExample(input_ids=self._data[idx])


class SimpleLMDataCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features) -> dict:
        # Handle both LMExample objects (from LMDataset) and raw tensors (from StreamingLMDataset)
        if isinstance(features[0], torch.Tensor):
            # StreamingLMDataset yields tensors directly
            input_tensor = torch.stack(features)
        elif isinstance(features[0], LMExample):
            # LMDataset yields LMExample objects
            max_len = max(len(f.input_ids) for f in features)
            input_ids = [f.input_ids + [self.pad_id] * (max_len - len(f.input_ids)) for f in features]
            input_tensor = torch.tensor(input_ids, dtype=torch.long)
        else:
            raise TypeError(f"Unexpected feature type: {type(features[0])}")

        return {
            "input_ids": input_tensor,
            "labels": input_tensor.clone(),
            "attention_mask": (input_tensor != self.pad_id).long(),
        }


class StreamingLMDataset(IterableDataset):
    """Memory-efficient streaming dataset for large corpora.

    Reads text files on-the-fly instead of loading everything into memory.
    Supports optional buffer shuffling for better randomization.
    """

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer,
        eos_id: int,
        block_size: int = 1024,
        shuffle_buffer: int = 10000,
        subdir: str = None,
    ):
        if shuffle_buffer < 1:
            raise ValueError(f"shuffle_buffer must be positive, got {shuffle_buffer}")
        if block_size < 1:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.data_dir = Path(data_dir)
        if subdir:
            self.data_dir = self.data_dir / subdir
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self.block_size = block_size
        self.shuffle_buffer = shuffle_buffer

        if not self.data_dir.exists() or not self.data_dir.is_dir():
            raise FileNotFoundError(f"Expected a data directory at: {self.data_dir}")

    def _encode_text(self, text: str) -> List[int]:
        """Tokenize text without forcing tokenizer-specific kwargs."""
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            return self.tokenizer.encode(text)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # Get list of files (sorted for reproducibility)
        txt_files = sorted(self.data_dir.rglob("*.txt"))
        if not txt_files:
            raise FileNotFoundError("No .txt files found in data directory.")

        # If in a worker process, split files across workers
        if worker_info is not None:
            txt_files = txt_files[worker_info.id :: worker_info.num_workers]

        # Shuffle file order for variety
        txt_files = list(txt_files)
        random.shuffle(txt_files)

        buffer: List[torch.Tensor] = []

        for file_path in txt_files:
            text = file_path.read_text(encoding="utf-8")
            ids = self._encode_text(text) + [self.eos_id]

            # Yield blocks from this file
            for i in range(0, len(ids) - self.block_size + 1, self.block_size):
                block = torch.tensor(ids[i : i + self.block_size], dtype=torch.long)

                if len(buffer) < self.shuffle_buffer:
                    buffer.append(block)
                else:
                    # Randomly replace from buffer to maintain ~shuffle_buffer size
                    idx = random.randint(0, self.shuffle_buffer - 1)
                    yield buffer[idx]
                    buffer[idx] = block

        # Drain remaining buffer (shuffled)
        random.shuffle(buffer)
        for block in buffer:
            yield block


def make_blocks(token_ids: List[int], block_size: int) -> List[List[int]]:
    blocks = []
    for i in range(0, len(token_ids) - block_size + 1, block_size):
        blocks.append(token_ids[i : i + block_size])
    return blocks


def build_trainer(
    model: torch.nn.Module,
    dataset: Dataset,
    collator: SimpleLMDataCollator,
    output_dir: str,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-4,
    max_steps: int = 100000,
    save_steps: int = 1000,
    logging_steps: int = 10,
    save_total_limit: int = 3,
) -> Trainer:
    """Build trainer optimized for RTX 6000 Ada GPUs.

    Default settings:
    - bf16 preferred on Ada architecture (falls back to fp16 on older GPUs)
    - batch_size=8 * gradient_accumulation=4 = effective batch size of 32 per GPU
    - With 2 GPUs: total effective batch size of 64
    - Uses ~45GB VRAM per GPU (safe for 49GB available)
    """
    _cuda = torch.cuda.is_available()
    _bf16 = _cuda and torch.cuda.is_bf16_supported()

    # Only enable TensorBoard reporting when backend is available.
    report_to = []
    try:
        import tensorboard  # noqa: F401
        report_to = ["tensorboard"]
    except Exception:
        try:
            import tensorboardX  # noqa: F401
            report_to = ["tensorboard"]
        except Exception:
            report_to = []

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_strategy="steps",
        save_total_limit=save_total_limit,
        warmup_steps=100,
        weight_decay=0.1,
        # Precision: prefer bf16 on Ada, fall back to fp16, cpu otherwise
        bf16=_bf16,
        bf16_full_eval=_bf16,
        fp16=_cuda and not _bf16,
        use_cpu=not _cuda,
        # Multi-GPU optimizations
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Stability
        max_grad_norm=1.0,
        # Logging
        report_to=report_to,
        run_name="llm_training",
        logging_first_step=True,
        # Memory
        gradient_checkpointing=False,  # Enable if running out of memory
        optim="adamw_torch_fused",     # Faster optimizer for Ada GPUs
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )


def load_texts_from_data_dir(data_dir: str | Path) -> Iterable[str]:
    path = Path(data_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Expected a data directory at: {path}")

    txt_files = sorted(path.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError("No .txt files found in data/ directory.")

    for file_path in txt_files:
        yield file_path.read_text(encoding="utf-8")
