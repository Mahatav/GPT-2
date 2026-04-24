"""Batch chat test for non-interactive model evaluation.

Runs predefined philosophical prompts against a trained model
and saves responses to a JSON file for analysis.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch

from inference_utils import generate_with_history, get_tokenizer, load_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Philosophical prompts for testing
PHILOSOPHICAL_PROMPTS = [
    "What is the true nature of the self?",
    "Who am I in the grand scheme of existence?",
    "What defines a person's identity?",
    "What is the meaning of life?",
    "Why do we exist?",
    "What gives life purpose?",
    "What makes an action morally right?",
    "How should one live a good life?",
    "What is the foundation of ethics?",
    "What is the nature of reality?",
    "Is the physical world the ultimate reality?",
    "What is the relationship between mind and matter?",
    "What is true knowledge?",
    "How can we know what is real?",
    "What is the nature of truth?",
    "What happens after death?",
    "Is death the end of existence?",
    "How should we face mortality?",
    "What is the relationship between humans and nature?",
    "What is the cosmic order of the universe?",
    "Are humans separate from or connected to the cosmos?",
    "How can one achieve enlightenment?",
    "What is liberation from suffering?",
    "What is the ultimate goal of spiritual practice?",
]


def run_batch_chat_test(
    checkpoint_path: str,
    output_path: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
) -> dict:
    """Run chat test on predefined prompts and save results.

    Args:
        checkpoint_path: Path to model checkpoint
        output_path: Path to save JSON results
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        Results dictionary
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model from {checkpoint_path} on {device}")

    try:
        model = load_model(checkpoint_path, device)
        encoding = get_tokenizer()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_path,
        "device": device,
        "config": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        "responses": [],
    }

    logger.info(f"Running {len(PHILOSOPHICAL_PROMPTS)} prompts...")
    history = ""

    for i, prompt in enumerate(PHILOSOPHICAL_PROMPTS):
        logger.info(f"  [{i+1}/{len(PHILOSOPHICAL_PROMPTS)}] {prompt[:50]}...")

        try:
            reply, history = generate_with_history(
                model,
                encoding,
                prompt,
                history=history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )

            results["responses"].append({
                "prompt": prompt,
                "response": reply,
                "response_length": len(reply),
            })
        except Exception as e:
            logger.error(f"    Error generating response: {e}")
            results["responses"].append({
                "prompt": prompt,
                "response": f"[ERROR: {str(e)}]",
                "response_length": 0,
            })

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    total_responses = len(results["responses"])
    avg_length = sum(r["response_length"] for r in results["responses"]) / total_responses if total_responses > 0 else 0
    logger.info(f"Total responses: {total_responses}")
    logger.info(f"Average response length: {avg_length:.1f} characters")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run batch chat test on philosophical prompts"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save results JSON file",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=150,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )

    args = parser.parse_args()

    run_batch_chat_test(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
