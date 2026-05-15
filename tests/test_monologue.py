"""
Tests for the coherence scorer and benchmark.

The Journal / IdleTracker / build_reflection_prompt / monologue_mode
trio was removed (dead infra: reader-without-writer, FSM-without-driver,
signal-without-consumer).  These tests cover only the live surface:
score_coherence and run_coherence_benchmark.

Run with: python -m pytest tests/test_monologue.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ====================================================================
# Coherence Scoring
# ====================================================================


class TestCoherenceScoring:
    """Tests for the heuristic coherence scorer."""

    def test_empty_text(self):
        """Empty text scores 0."""
        from enigma_engine.core.monologue import score_coherence
        assert score_coherence("") == 0.0

    def test_gibberish(self):
        """Random characters score low."""
        from enigma_engine.core.monologue import score_coherence
        score = score_coherence("asdf jkl; qwer uiop zxcv")
        assert score < 0.5

    def test_coherent_sentence(self):
        """A well-formed sentence scores higher than gibberish."""
        from enigma_engine.core.monologue import score_coherence
        good = score_coherence(
            "I noticed the user is interested in machine learning "
            "and prefers concise explanations.")
        bad = score_coherence("xxx yyy zzz aaa bbb ccc")
        assert good > bad

    def test_score_range(self):
        """Score is always in [0.0, 1.0]."""
        from enigma_engine.core.monologue import score_coherence
        for text in ["", "hello", "A" * 1000, "The quick brown fox."]:
            s = score_coherence(text)
            assert 0.0 <= s <= 1.0

    def test_very_short_text(self):
        """Very short text (< 10 chars) scores low."""
        from enigma_engine.core.monologue import score_coherence
        assert score_coherence("Hi") < 0.5

    def test_repetitive_text(self):
        """Highly repetitive text scores lower."""
        from enigma_engine.core.monologue import score_coherence
        repetitive = "the the the the the the the the the the"
        varied = "The user enjoys reading books about history and science."
        assert score_coherence(varied) > score_coherence(repetitive)

    def test_coherence_threshold_default(self):
        """Default coherence threshold is 0.7."""
        from enigma_engine.core.monologue import DEFAULT_COHERENCE_THRESHOLD
        assert DEFAULT_COHERENCE_THRESHOLD == 0.7


# ====================================================================
# Coherence Benchmark
# ====================================================================


class TestCoherenceBenchmark:
    """Tests for the coherence benchmark function."""

    def test_benchmark_prompts_exist(self):
        """Benchmark has diverse prompts."""
        from enigma_engine.core.monologue import _BENCHMARK_PROMPTS
        assert len(_BENCHMARK_PROMPTS) >= 10

    def test_benchmark_prompts_unique(self):
        """Benchmark prompts are all unique."""
        from enigma_engine.core.monologue import _BENCHMARK_PROMPTS
        assert len(set(_BENCHMARK_PROMPTS)) == len(_BENCHMARK_PROMPTS)

    def test_benchmark_with_mock_engine(self):
        """Benchmark runs with a mock engine and returns valid results."""
        from unittest.mock import MagicMock
        from enigma_engine.core.monologue import run_coherence_benchmark

        engine = MagicMock()
        engine.chat.return_value = (
            "I noticed the user is interested in learning about "
            "machine learning and prefers detailed explanations "
            "with practical examples.")

        result = run_coherence_benchmark(engine, num_prompts=5)

        assert "scores" in result
        assert "mean" in result
        assert "pass_rate" in result
        assert "passed" in result
        assert "total" in result
        assert "recommendation" in result
        assert result["total"] == 5
        assert 0.0 <= result["mean"] <= 1.0
        assert 0.0 <= result["pass_rate"] <= 1.0

    def test_benchmark_empty_responses(self):
        """Benchmark handles empty model responses gracefully."""
        from unittest.mock import MagicMock
        from enigma_engine.core.monologue import run_coherence_benchmark

        engine = MagicMock()
        engine.chat.return_value = ""

        result = run_coherence_benchmark(engine, num_prompts=3)

        assert result["total"] == 3
        assert result["mean"] == 0.0
        assert result["passed"] == 0

    def test_benchmark_recommendation_not_ready(self):
        """Low-quality responses yield 'not_ready' recommendation."""
        from unittest.mock import MagicMock
        from enigma_engine.core.monologue import run_coherence_benchmark

        engine = MagicMock()
        engine.chat.return_value = "uh ok"

        result = run_coherence_benchmark(engine, num_prompts=5)
        assert result["recommendation"] == "not_ready"

    def test_benchmark_recommendation_ready(self):
        """High-quality responses yield 'ready' recommendation."""
        from unittest.mock import MagicMock
        from enigma_engine.core.monologue import run_coherence_benchmark

        engine = MagicMock()
        engine.chat.return_value = (
            "The user seems very interested in physics and astronomy. "
            "They ask thoughtful questions about orbital mechanics and "
            "seem to appreciate when I provide mathematical detail.")

        result = run_coherence_benchmark(engine, num_prompts=5)
        assert result["recommendation"] == "ready"

    def test_benchmark_progress_callback(self):
        """Progress callback is invoked for each prompt."""
        from unittest.mock import MagicMock
        from enigma_engine.core.monologue import run_coherence_benchmark

        engine = MagicMock()
        engine.chat.return_value = "A short but somewhat decent thought."
        progress_calls = []

        def on_progress(idx, total, score):
            progress_calls.append((idx, total, score))

        run_coherence_benchmark(
            engine, num_prompts=4, on_progress=on_progress)

        assert len(progress_calls) == 4
        assert progress_calls[0][0] == 1
        assert progress_calls[-1][0] == 4

    def test_benchmark_exception_in_chat(self):
        """Benchmark handles engine.chat() exceptions gracefully."""
        from unittest.mock import MagicMock
        from enigma_engine.core.monologue import run_coherence_benchmark

        engine = MagicMock()
        engine.chat.side_effect = RuntimeError("model error")

        result = run_coherence_benchmark(engine, num_prompts=3)
        assert result["total"] == 3
        assert result["mean"] == 0.0

    def test_benchmark_clamps_num_prompts(self):
        """num_prompts is clamped between 1 and 200."""
        from unittest.mock import MagicMock
        from enigma_engine.core.monologue import run_coherence_benchmark

        engine = MagicMock()
        engine.chat.return_value = "A thought."

        result = run_coherence_benchmark(engine, num_prompts=0)
        assert result["total"] == 1
