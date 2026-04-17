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

from pathlib import Path

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
        return {"perplexity": float("inf"), "loss": float("inf"),
                "num_prompts": 0}

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    evaluated = 0

    try:
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
                    logits_flat = (logits if logits.dim() == 2
                                   else logits.reshape(-1, logits.size(-1)))
                    loss = F.cross_entropy(
                        logits_flat,
                        targets.reshape(-1),
                        reduction="sum",
                    )

                    total_loss += loss.item()
                    total_tokens += targets.numel()
                    evaluated += 1

                except Exception as exc:
                    logger.warning(f"Failed to evaluate prompt: {exc}")
                    continue

        if total_tokens == 0:
            return {"perplexity": float("inf"), "loss": float("inf"),
                    "num_prompts": evaluated}

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
            "num_prompts": evaluated,
        }
    finally:
        if was_training:
            model.train()


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

    was_training = model.training
    model.eval()
    successes = 0

    try:
        for test_case in test_cases:
            try:
                prompt = test_case.get("prompt")
                if not prompt:
                    continue
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
    finally:
        if was_training:
            model.train()


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


# Default tool/command test cases for evaluating command usage accuracy
DEFAULT_TOOL_TEST_CASES: list[dict[str, str]] = [
    {
        "prompt": "Search the web for the latest Python release",
        "expected_command": "search.web",
    },
    {
        "prompt": "Read the file data/training.txt",
        "expected_command": "file.read",
    },
    {
        "prompt": "Write 'hello world' to outputs/test.txt",
        "expected_command": "file.write",
    },
    {
        "prompt": "Search for images of neural network diagrams",
        "expected_command": "search.images",
    },
    {
        "prompt": "List all files in the data directory",
        "expected_command": "file.list",
    },
]


def run_golden_eval(
    model: nn.Module,
    tokenizer: Any,
    golden_path: str | Path,
    device: str = "cuda",
    max_gen: int = 200,
) -> dict[str, Any]:
    """Run golden prompt regression evaluation.

    Reads a JSON file containing test cases with prompts and expected
    keywords.  Generates a response for each prompt and checks that
    the expected keywords appear.  Used to detect regressions after
    training — if the model used to answer correctly and now doesn't,
    it's a regression.

    JSON format::

        [
          {"prompt": "What is 2+2?", "expected": ["4"]},
          {"prompt": "Capital of France?", "expected": ["Paris"]}
        ]

    Args:
        model: The model to evaluate.
        tokenizer: Tokenizer for encoding/decoding.
        golden_path: Path to JSON file with test cases.
        device: Device to run on.
        max_gen: Maximum tokens to generate per prompt.

    Returns:
        Dict with pass_rate, passed, total, and per-case results.
    """
    import json
    import torch

    golden_path = Path(golden_path)
    if not golden_path.exists():
        logger.warning("Golden eval file not found: %s", golden_path)
        return {"pass_rate": 0.0, "passed": 0, "total": 0, "results": []}

    try:
        cases = json.loads(golden_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load golden eval: %s", exc)
        return {"pass_rate": 0.0, "passed": 0, "total": 0, "results": []}

    if not cases:
        return {"pass_rate": 0.0, "passed": 0, "total": 0, "results": []}

    was_training = model.training
    model.eval()
    results: list[dict[str, Any]] = []
    passed = 0

    try:
        with torch.no_grad():
            for case in cases:
                prompt = case.get("prompt", "")
                expected = case.get("expected", [])
                if not prompt or not expected:
                    continue

                try:
                    # Tokenize prompt
                    if hasattr(tokenizer, "encode"):
                        tokens = tokenizer.encode(prompt)
                        if hasattr(tokens, "ids"):
                            token_ids = tokens.ids
                        else:
                            token_ids = tokens
                    else:
                        token_ids = [
                            tokenizer.token_to_id(c) for c in prompt]

                    # Greedy generate
                    generated = list(token_ids)
                    for _ in range(max_gen):
                        inp = torch.tensor(
                            [generated], device=device)
                        logits = model(inp)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        next_id = logits[0, -1].argmax().item()
                        # Stop on EOS
                        eos_id = getattr(tokenizer, "eos_token_id", None)
                        if eos_id is not None and next_id == eos_id:
                            break
                        generated.append(next_id)

                    # Decode response (only the generated part)
                    gen_ids = generated[len(token_ids):]
                    if hasattr(tokenizer, "decode"):
                        response = tokenizer.decode(gen_ids)
                    else:
                        response = "".join(
                            tokenizer.id_to_token(i) for i in gen_ids)

                    # Check expected keywords
                    resp_lower = response.lower()
                    found = [
                        kw for kw in expected
                        if kw.lower() in resp_lower]
                    case_passed = len(found) == len(expected)

                except Exception as exc:
                    logger.debug("Golden eval case failed: %s", exc)
                    response = ""
                    found = []
                    case_passed = False

                if case_passed:
                    passed += 1

                results.append({
                    "prompt": prompt,
                    "expected": expected,
                    "found": found,
                    "passed": case_passed,
                    "response": response[:200],
                })

        total = len(results)
        return {
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "passed": passed,
            "total": total,
            "results": results,
        }
    finally:
        if was_training:
            model.train()
