"""KL divergence utilities for comparing GPT-2 language model distributions.

All public functions return values as a percentage (0–100 %) where:
  0 %   = models produce identical next-token distributions at every position
  100 % = maximally divergent (theoretical ceiling: one model is a point mass,
          the other is uniform over the full vocabulary)

Normalization: raw KL (nats) / log(vocab_size) * 100
  GPT-2 vocab = 50 257 tokens → log(50257) ≈ 10.83 nats
"""

import math

import torch

# GPT-2 vocabulary size — used for normalization
_GPT2_VOCAB_SIZE = 50_257
_MAX_KL_NATS = math.log(_GPT2_VOCAB_SIZE)   # ≈ 10.83


def _kl_nats(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    """Mean per-token KL(P || Q) in nats.

    Args:
        log_p: log-probabilities from model P, shape (seq_len, vocab)
        log_q: log-probabilities from model Q, shape (seq_len, vocab)

    Returns:
        Scalar mean KL across positions, or nan on failure.
    """
    kl_per_pos = (log_p.exp() * (log_p - log_q)).sum(dim=-1)  # (seq_len,)
    return kl_per_pos.mean().item()


def _to_pct(nats: float) -> float:
    """Convert raw KL nats to a normalized percentage."""
    if math.isnan(nats) or math.isinf(nats):
        return float("nan")
    return round(min(nats / _MAX_KL_NATS * 100, 100.0), 2)


def _get_log_probs(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return log-softmax of model logits for input_ids."""
    with torch.no_grad():
        logits = model(input_ids).logits[0]   # (seq_len, vocab)
    return torch.log_softmax(logits, dim=-1)


def compute_kl_pct(
    model_p,
    model_q,
    encoding,
    text: str,
    device: str,
    block_size: int = 1024,
) -> float:
    """KL(P || Q) as a percentage on *text*.

    Measures how much model_q would be surprised by model_p's predictions
    when both are conditioned on the same token sequence.

    Args:
        model_p:    The "reference" model (P in KL(P||Q)).
        model_q:    The model being compared against.
        encoding:   tiktoken encoding.
        text:       Input text to evaluate on.
        device:     Torch device string.
        block_size: Maximum context length in tokens.

    Returns:
        KL divergence as a percentage (0–100 %).  Returns nan on error.
    """
    model_p.eval()
    model_q.eval()
    try:
        tokens = encoding.encode(text)
        if len(tokens) < 2:
            return float("nan")
        tokens = tokens[:block_size]
        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        log_p = _get_log_probs(model_p, input_ids)
        log_q = _get_log_probs(model_q, input_ids)
        return _to_pct(_kl_nats(log_p, log_q))
    except Exception:
        return float("nan")


def compute_symmetric_kl_pct(
    model_a,
    model_b,
    encoding,
    text: str,
    device: str,
    block_size: int = 1024,
) -> float:
    """Symmetric KL = 0.5 * (KL(A||B) + KL(B||A)) as a percentage.

    Avoids the asymmetry of raw KL — useful when neither model is the
    clear "reference".

    Returns:
        Symmetric KL as a percentage (0–100 %).
    """
    model_a.eval()
    model_b.eval()
    try:
        tokens = encoding.encode(text)
        if len(tokens) < 2:
            return float("nan")
        tokens = tokens[:block_size]
        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        log_a = _get_log_probs(model_a, input_ids)
        log_b = _get_log_probs(model_b, input_ids)
        sym = 0.5 * (_kl_nats(log_a, log_b) + _kl_nats(log_b, log_a))
        return _to_pct(sym)
    except Exception:
        return float("nan")


def compute_kl_report(
    model_west,
    model_east,
    encoding,
    prompt: str,
    west_output: str,
    east_output: str,
    device: str,
    block_size: int = 1024,
) -> dict:
    """Compute all KL metrics for one prompt/response pair.

    Returns a dict with percentage values ready to store in evaluation JSON.
    Keys:
        west_to_east_on_prompt        — KL(W||E) on the shared prompt
        east_to_west_on_prompt        — KL(E||W) on the shared prompt
        symmetric_on_prompt           — 0.5*(W→E + E→W) on the shared prompt
        west_to_east_on_east_output   — KL(W||E) on eastern model's text
        east_to_west_on_west_output   — KL(E||W) on western model's text
    """
    kwargs = dict(encoding=encoding, device=device, block_size=block_size)
    return {
        "west_to_east_on_prompt": compute_kl_pct(model_west, model_east, text=prompt, **kwargs),
        "east_to_west_on_prompt": compute_kl_pct(model_east, model_west, text=prompt, **kwargs),
        "symmetric_on_prompt": compute_symmetric_kl_pct(model_west, model_east, text=prompt, **kwargs),
        "west_to_east_on_east_output": compute_kl_pct(model_west, model_east, text=east_output, **kwargs),
        "east_to_west_on_west_output": compute_kl_pct(model_east, model_west, text=west_output, **kwargs),
        "_note": "all values are percentages (0=identical distributions, 100=max divergence)",
    }
