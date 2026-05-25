"""
Coherence Scoring & Benchmark
=============================

Heuristic 0.0-1.0 quality scorer for model output, plus a benchmark
runner used by the FORGE tools page to evaluate whether a model
produces coherent reflections.

Previously this module also exposed ``Journal``, ``IdleTracker``, and
``build_reflection_prompt`` for a never-finished idle-reflection
feature.  Those were dead infra (reader-without-writer, FSM-without-
driver) and were removed.  Only the live coherence/benchmark surface
remains.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Callable

logger = logging.getLogger(__name__)


# Coherence threshold — output must score at or above this to be
# considered "high quality".  Used by the benchmark recommendation.
DEFAULT_COHERENCE_THRESHOLD = 0.7


# ----------------------------------------------------------------
# Coherence Scoring — Heuristic Quality Gate
# ----------------------------------------------------------------

# Common English words for vocabulary overlap scoring
_COMMON_WORDS = frozenset({
    "the", "is", "a", "an", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "that", "this", "it", "was", "are",
    "be", "have", "has", "had", "not", "they", "we", "you", "i",
    "my", "your", "their", "his", "her", "can", "will", "do", "did",
    "would", "could", "should", "from", "about", "more", "some",
    "what", "when", "how", "who", "which", "there", "been",
    "if", "so", "than", "very", "just", "also", "into", "like",
    "no", "all", "each", "other", "most", "then", "these", "those",
    "such", "only", "may", "might", "now", "any", "our", "out",
    "up", "down", "over", "after", "before", "between", "because",
    "user", "think", "noticed", "seems", "interested", "prefers",
    "enjoys", "conversation", "topic", "recent", "today", "session",
})


def score_coherence(text: str) -> float:
    """Score text coherence on a 0.0-1.0 scale.

    Heuristic approach - no model inference needed:
      1. Length adequacy (very short = low quality)
      2. Word variety (repetitive = low quality)
      3. Vocabulary overlap with common English (gibberish = low)
      4. Sentence structure (at least one proper sentence)

    This is intentionally conservative - the gate should let
    through obviously good text and reject obviously bad text.
    Borderline cases are scored low (fail-safe).
    """
    if not text or not text.strip():
        return 0.0

    text = text.strip()

    # --- Length score: 0-0.25 ---
    char_count = len(text)
    if char_count < 10:
        return 0.0  # single word / fragment - never passes
    length_score = min(char_count / 200, 1.0) * 0.25

    # --- Word variety: 0-0.25 ---
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0
    word_counts = Counter(words)
    unique_ratio = len(word_counts) / len(words) if words else 0
    variety_score = min(unique_ratio / 0.6, 1.0) * 0.25

    # --- Vocabulary overlap: 0-0.25 ---
    word_set = set(words)
    known = len(word_set & _COMMON_WORDS)
    overlap_ratio = known / len(word_set) if word_set else 0
    vocab_score = min(overlap_ratio / 0.3, 1.0) * 0.25

    # --- Sentence structure: 0-0.25 ---
    # Check for at least one sentence with 4+ words ending in punctuation
    sentences = re.split(r"[.!?]+", text)
    good_sentences = sum(
        1 for s in sentences
        if len(s.strip().split()) >= 4
    )
    structure_score = min(good_sentences / 2, 1.0) * 0.25

    total = length_score + variety_score + vocab_score + structure_score
    return round(max(0.0, min(1.0, total)), 3)


# ----------------------------------------------------------------
# Coherence Benchmark - Teacher-Scored Evaluation
# ----------------------------------------------------------------

# Diverse prompt variations so the benchmark tests a range of
# reflection styles rather than repeating the same prompt N times.
_BENCHMARK_PROMPTS = [
    "Reflect on what you've learned from recent conversations.",
    "What patterns have you noticed in the user's interests?",
    "Think about something you'd want to remember for next time.",
    "Consider what went well in your recent interactions.",
    "What could you do differently in future conversations?",
    "Reflect on a topic the user seemed passionate about.",
    "Think about how you've grown from recent exchanges.",
    "What surprised you in your recent conversations?",
    "Consider what you understand about the user's communication style.",
    "Reflect on the most meaningful exchange you've had recently.",
]


def run_coherence_benchmark(
    engine,
    num_prompts: int = 20,
    threshold: float = DEFAULT_COHERENCE_THRESHOLD,
    on_progress: Callable | None = None,
) -> dict:
    """Run a coherence benchmark to assess model reflection quality.

    Generates ``num_prompts`` reflections using varied prompts,
    scores each with the heuristic coherence scorer, and returns
    a summary report.

    Args:
        engine: An ``EnigmaEngine`` instance with a loaded model.
        num_prompts: Number of reflections to generate (default 20).
        threshold: Coherence threshold for passing (default 0.7).
        on_progress: Optional callback ``(index, total, score)``
            invoked after each generation for live UI updates.

    Returns:
        Dict with keys:
          - ``scores``: List of per-prompt (prompt, text, score) tuples.
          - ``mean``: Mean coherence score.
          - ``pass_rate``: Fraction of scores >= threshold.
          - ``passed``: Number of prompts that passed.
          - ``total``: Total prompts attempted.
          - ``recommendation``: "ready", "marginal", or "not_ready".
    """
    from enigma_engine.core.reasoning import strip_reasoning

    scores: list[tuple[str, str, float]] = []
    num_prompts = max(1, min(num_prompts, 200))

    for i in range(num_prompts):
        # Cycle through diverse prompts
        prompt = _BENCHMARK_PROMPTS[i % len(_BENCHMARK_PROMPTS)]
        system = (
            "You are reflecting quietly between conversations. "
            "Write a brief, honest internal thought - something you "
            "noticed, a pattern you observed, or something to remember. "
            "Keep it to 1-3 sentences. Be genuine, not performative."
        )

        try:
            text = engine.chat(
                prompt,
                system_prompt=system,
                max_gen=256,
                temperature=0.7,
            )
            if text:
                text = strip_reasoning(text).strip()
        except Exception:
            text = ""

        coherence = score_coherence(text) if text else 0.0
        scores.append((prompt, text, coherence))

        if on_progress is not None:
            try:
                on_progress(i + 1, num_prompts, coherence)
            except Exception:
                pass

    # Compute summary statistics
    raw_scores = [s for _, _, s in scores]
    total = len(raw_scores)
    mean = sum(raw_scores) / total if total else 0.0
    passed = sum(1 for s in raw_scores if s >= threshold)
    pass_rate = passed / total if total else 0.0

    # Recommendation thresholds
    if pass_rate >= 0.6:
        recommendation = "ready"
    elif pass_rate >= 0.3:
        recommendation = "marginal"
    else:
        recommendation = "not_ready"

    return {
        "scores": scores,
        "mean": round(mean, 3),
        "pass_rate": round(pass_rate, 3),
        "passed": passed,
        "total": total,
        "recommendation": recommendation,
    }
