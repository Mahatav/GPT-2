"""Shared inference utilities for GPT-2 models."""

from __future__ import annotations

import tiktoken
import torch
from transformers import GPT2LMHeadModel


def load_model(checkpoint_path: str, device: str = None) -> GPT2LMHeadModel:
    """Load a trained GPT-2 model from checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    model.eval()
    model.to(device)
    return model


def get_tokenizer():
    """Get the GPT-2 tokenizer."""
    return tiktoken.get_encoding("gpt2")


def generate(
    model: GPT2LMHeadModel,
    encoding: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    top_p: float = 0.9,
    top_k: int = 0,
    do_sample: bool = True,
    device: str = None,
) -> str:
    """Generate text from a model given a prompt."""
    if device is None:
        device = next(model.parameters()).device

    input_ids = encoding.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=encoding.eot_token,
            eos_token_id=encoding.eot_token,
        )

    new_tokens = output[0][len(input_ids):]
    return encoding.decode(new_tokens.tolist())


def generate_with_history(
    model: GPT2LMHeadModel,
    encoding: tiktoken.Encoding,
    prompt: str,
    history: str = "",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = None,
) -> tuple[str, str]:
    """Generate text with conversation history.

    Returns:
        tuple: (response, updated_history)
    """
    if device is None:
        device = next(model.parameters()).device

    full_prompt = f"{history}Human: {prompt}\nAssistant:" if history else f"Human: {prompt}\nAssistant:"

    response = generate(
        model, encoding, full_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    reply = response.split("Human:")[0].strip()
    updated_history = f"{history}Human: {prompt}\nAssistant: {reply}\n"

    return reply, updated_history
