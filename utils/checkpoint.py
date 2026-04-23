"""
Checkpoint saving and loading.

A checkpoint stores everything needed to resume training or run inference:
  - model weights (state_dict)
  - optimizer state (momentum buffers, adaptive learning rates)
  - iteration number + best validation loss
  - the GPT2Config as a plain dict (so we can reconstruct the model)
  - tokenizer class name + its save file path

Checkpoints are saved as .pt files via torch.save / torch.load.
"""

import os
import json
from typing import Optional
import torch

from config import GPT2Config
from model.gpt2 import GPT2


# ── Saving ────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path:       str,
    model:      GPT2,
    optimizer:  torch.optim.Optimizer,
    config:     GPT2Config,
    iteration:  int,
    val_loss:   float,
    extra:      dict = None,
) -> None:
    """
    Persist a full training snapshot to disk.

    Parameters
    ----------
    path      : file path to write (should end in .pt)
    model     : the GPT2 instance
    optimizer : the AdamW optimizer
    config    : GPT2Config (architecture metadata)
    iteration : current training step
    val_loss  : most recent validation loss
    extra     : optional additional items to store (e.g. tokenizer info)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    payload = {
        "iteration":   iteration,
        "val_loss":    val_loss,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        # Store config as a plain dict so we can load without importing GPT2Config
        "config": {
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_layer":    config.n_layer,
            "n_head":     config.n_head,
            "n_embd":     config.n_embd,
            "dropout":    config.dropout,
            "bias":       config.bias,
        },
    }
    if extra:
        payload.update(extra)

    torch.save(payload, path)


# ── Loading ───────────────────────────────────────────────────────────────────

def load_checkpoint(
    path:           str,
    device:         torch.device = None,
    load_optimizer: bool = True,
) -> dict:
    """
    Load a checkpoint and return it as a dict.

    The caller is responsible for actually calling model.load_state_dict()
    and optimizer.load_state_dict() on the returned values.  This keeps the
    function dependency-free — no model or optimizer instance required.

    Returns
    -------
    dict with keys: iteration, val_loss, model_state, optim_state, config
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device if device is not None else "cpu"
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if not load_optimizer:
        ckpt.pop("optim_state", None)

    return ckpt


def resume_model(
    path:   str,
    device: torch.device = None,
) -> tuple[GPT2, GPT2Config, int, float]:
    """
    Convenience: rebuild GPT2 from a checkpoint file without any prior config.

    Returns
    -------
    (model, config, iteration, val_loss)
    """
    ckpt   = load_checkpoint(path, device=device, load_optimizer=False)
    config = GPT2Config(**ckpt["config"])
    model  = GPT2(config)
    model.load_state_dict(ckpt["model_state"])
    if device:
        model = model.to(device)
    return model, config, ckpt["iteration"], ckpt["val_loss"]


# ── Utilities ─────────────────────────────────────────────────────────────────

def list_checkpoints(directory: str) -> list[str]:
    """Return all .pt files in a directory, sorted newest last."""
    if not os.path.isdir(directory):
        return []
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".pt")
    ]
    return sorted(files, key=os.path.getmtime)


def latest_checkpoint(directory: str) -> Optional[str]:
    """Return the path of the most recently modified checkpoint, or None."""
    ckpts = list_checkpoints(directory)
    return ckpts[-1] if ckpts else None
