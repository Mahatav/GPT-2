"""
PyTorch Dataset for language model training.

A language model is trained to predict the next token at every position.
So given a sequence of tokens [t0, t1, t2, ..., tN], the dataset produces:
  x = [t0, t1, t2, ..., t_{N-1}]   (input)
  y = [t1, t2, t3, ..., t_N]       (target — shifted by one)

The model sees x and must predict y, learning to complete sequences.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer.base import BaseTokenizer


class TextDataset(Dataset):
    """
    Sliding-window dataset over a flat array of token ids.

    Each item is a (block_size,) input + (block_size,) target pair.
    Items overlap by block_size - 1 tokens; this maximises use of data.

    Parameters
    ----------
    tokens     : 1-D numpy array of integer token ids
    block_size : number of tokens per training example (= context window)
    """

    def __init__(self, tokens: np.ndarray, block_size: int):
        if len(tokens) <= block_size:
            raise ValueError(
                f"Dataset has {len(tokens)} tokens but block_size={block_size}. "
                f"Need at least block_size + 1 tokens."
            )
        self.tokens     = tokens
        self.block_size = block_size

    def __len__(self) -> int:
        # Every starting position produces one valid (x, y) pair
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Slice a chunk of block_size + 1 consecutive tokens
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))  # input
        y = torch.from_numpy(chunk[1: ].astype(np.int64))  # target (shifted by 1)
        return x, y

    # ── Factories ──────────────────────────────────────────────────────────

    @classmethod
    def from_text(
        cls,
        text:       str,
        tokenizer:  BaseTokenizer,
        block_size: int,
    ) -> "TextDataset":
        """Encode raw text with a tokenizer and wrap in a dataset."""
        tokens = np.array(tokenizer.encode(text), dtype=np.int64)
        return cls(tokens, block_size)

    def train_val_split(
        self,
        val_fraction: float = 0.1,
    ) -> tuple["TextDataset", "TextDataset"]:
        """
        Split token array into train / val datasets.

        The split is positional (first 90% = train), NOT shuffled,
        to prevent data leakage between train and val in sequential text.
        """
        split = int((1.0 - val_fraction) * len(self.tokens))
        return (
            TextDataset(self.tokens[:split],  self.block_size),
            TextDataset(self.tokens[split:],  self.block_size),
        )

    # ── Info ───────────────────────────────────────────────────────────────

    def token_count(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        return (
            f"TextDataset(tokens={len(self.tokens):,}  "
            f"block_size={self.block_size}  examples={len(self):,})"
        )


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_loader(
    dataset:    TextDataset,
    batch_size: int,
    shuffle:    bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Thin wrapper so callers don't need to import DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )
