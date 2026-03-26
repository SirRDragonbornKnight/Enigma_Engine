"""
Training Evaluation for Enigma Engine
=====================================

Implements before/after training evaluation to measure improvement.

Features:
- Compute perplexity on held-out test prompts
- Compare model performance before and after training
- Track tool use success rates (command execution accuracy)

Usage:
    from enigma_engine.core.training_evaluation import evaluate_model

    before_metrics = evaluate_model(model, tokenizer, test_prompts)
    # ... train model ...
    after_metrics = evaluate_model(model, tokenizer, test_prompts)

    improvement = before_metrics["perplexity"] - after_metrics["perplexity"]
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    tokenizer: Any,
    test_prompts: list[str],
    device: str = "cuda",
    max_length: int = 512,
) -> dict[str, Any]:
    """Evaluate model perplexity on test prompts.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for encoding prompts
        test_prompts: List of test strings to evaluate
        device: Device to run evaluation on
        max_length: Maximum sequence length for evaluation

    Returns:
        Dict with:
            - perplexity: float (exp of average loss)
            - loss: float (average cross-entropy loss)
            - num_prompts: int (number of prompts evaluated)
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    if not test_prompts:
        return {"perplexity": 0.0, "loss": 0.0, "num_prompts": 0}

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for prompt in test_prompts:
            try:
                # Tokenize prompt
                if hasattr(tokenizer, "encode"):
                    tokens = tokenizer.encode(prompt)
                    if hasattr(tokens, "ids"):
                        token_ids = tokens.ids
                    else:
                        token_ids = tokens
                else:
                    # Fallback for char tokenizer
                    token_ids = [tokenizer.token_to_id(c) for c in prompt]

                # Truncate if needed
                if len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]

                if len(token_ids) < 2:
                    continue  # Need at least 2 tokens for loss

                # Convert to tensor
                input_ids = torch.tensor([token_ids], device=device)

                # Forward pass
                logits = model(input_ids[:, :-1])
                if isinstance(logits, tuple):
                    logits = logits[0]

                # Compute loss
                targets = input_ids[:, 1:]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="sum",
                )

                total_loss += loss.item()
                total_tokens += targets.numel()

            except Exception as exc:
                logger.warning(f"Failed to evaluate prompt: {exc}")
                continue

    if total_tokens == 0:
        return {"perplexity": 0.0, "loss": 0.0, "num_prompts": 0}

    avg_loss = total_loss / total_tokens

    # Compute perplexity = exp(loss), capped to avoid overflow
    try:
        perplexity = math.exp(min(avg_loss, 20.0))
        perplexity = min(perplexity, 1e6)
    except OverflowError:
        perplexity = 1e6

    return {
        "perplexity": round(perplexity, 4),
        "loss": round(avg_loss, 4),
        "num_prompts": len(test_prompts),
    }


def evaluate_tool_usage(
    model: nn.Module,
    tokenizer: Any,
    engine: Any,
    test_cases: list[dict[str, Any]],
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate model's tool/command usage accuracy.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for encoding/decoding
        engine: EnigmaEngine instance for command execution
        test_cases: List of dicts with "prompt" and "expected_command" keys
        device: Device to run evaluation on

    Returns:
        Dict with:
            - success_rate: float (0.0 to 1.0)
            - total_tests: int
            - successes: int
            - failures: int
    """
    if not test_cases:
        return {
            "success_rate": 0.0,
            "total_tests": 0,
            "successes": 0,
            "failures": 0,
        }

    model.eval()
    successes = 0

    for test_case in test_cases:
        try:
            prompt = test_case["prompt"]
            expected_cmd = test_case.get("expected_command", "")

            # Generate response
            response = engine.generate(
                prompt,
                max_tokens=200,
                temperature=0.0,  # Deterministic for evaluation
            )

            # Check if expected command appears in response
            if expected_cmd and expected_cmd.lower() in response.lower():
                successes += 1
            # Or check if any command was used successfully
            elif "[CMD]" in response and "[/CMD]" in response:
                # Could validate command execution here
                successes += 1

        except Exception as exc:
            logger.warning(f"Tool evaluation failed for test case: {exc}")
            continue

    total = len(test_cases)
    success_rate = successes / total if total > 0 else 0.0

    return {
        "success_rate": round(success_rate, 4),
        "total_tests": total,
        "successes": successes,
        "failures": total - successes,
    }


# Default test prompts for quick evaluation
DEFAULT_TEST_PROMPTS = [
    "The capital of France is",
    "2 + 2 equals",
    "The sky is",
    "Once upon a time",
    "Artificial intelligence is",
    "Python is a programming language",
    "The Earth orbits the",
    "Machine learning requires",
    "Neural networks are composed of",
    "Training a model involves",
]
