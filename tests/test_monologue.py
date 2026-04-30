"""
Tests for the monologue/journal system (Phase 5: Self-Initiated Behavior).

Covers:
- Journal storage (read/write/trim)
- Coherence scoring (quality gate)
- Reflection prompt building
- Idle detection logic

Run with: python -m pytest tests/test_monologue.py -v
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ════════════════════════════════════════════════════════════════════
# Journal Storage
# ════════════════════════════════════════════════════════════════════


class TestJournal:
    """Tests for the per-model journal file."""

    def test_empty_journal(self, tmp_path):
        """New journal has no entries."""
        from enigma_engine.core.monologue import Journal
        j = Journal(journal_dir=tmp_path)
        assert j.entries == []
        assert j.count == 0

    def test_add_entry(self, tmp_path):
        """Can add an entry and read it back."""
        from enigma_engine.core.monologue import Journal
        j = Journal(journal_dir=tmp_path)
        j.add("I noticed the user seems interested in physics.")
        assert j.count == 1
        assert "physics" in j.entries[0]["text"]

    def test_add_multiple_entries(self, tmp_path):
        """Multiple entries are ordered chronologically."""
        from enigma_engine.core.monologue import Journal
        j = Journal(journal_dir=tmp_path)
        j.add("First thought.")
        j.add("Second thought.")
        assert j.count == 2
        assert j.entries[0]["text"] == "First thought."
        assert j.entries[1]["text"] == "Second thought."

    def test_persistence(self, tmp_path):
        """Journal persists across instances."""
        from enigma_engine.core.monologue import Journal
        j1 = Journal(journal_dir=tmp_path)
        j1.add("Persistent thought.")

        j2 = Journal(journal_dir=tmp_path)
        assert j2.count == 1
        assert "Persistent" in j2.entries[0]["text"]

    def test_trim_oldest(self, tmp_path):
        """Journal trims oldest entries when over capacity."""
        from enigma_engine.core.monologue import Journal
        j = Journal(journal_dir=tmp_path, max_entries=3)
        j.add("A")
        j.add("B")
        j.add("C")
        j.add("D")
        assert j.count == 3
        texts = [e["text"] for e in j.entries]
        assert "A" not in texts
        assert "D" in texts

    def test_latest(self, tmp_path):
        """latest() returns the most recent entry."""
        from enigma_engine.core.monologue import Journal
        j = Journal(journal_dir=tmp_path)
        j.add("Old thought.")
        j.add("New thought.")
        latest = j.latest()
        assert latest is not None
        assert latest["text"] == "New thought."

    def test_latest_empty(self, tmp_path):
        """latest() returns None on empty journal."""
        from enigma_engine.core.monologue import Journal
        j = Journal(journal_dir=tmp_path)
        assert j.latest() is None

    def test_entry_has_timestamp(self, tmp_path):
        """Each entry has a timestamp field."""
        from enigma_engine.core.monologue import Journal
        j = Journal(journal_dir=tmp_path)
        j.add("Thought.")
        entry = j.entries[0]
        assert "timestamp" in entry
        assert isinstance(entry["timestamp"], str)

    def test_empty_text_rejected(self, tmp_path):
        """Empty or whitespace-only text is not added."""
        from enigma_engine.core.monologue import Journal
        j = Journal(journal_dir=tmp_path)
        j.add("")
        j.add("   ")
        assert j.count == 0

    def test_build_context(self, tmp_path):
        """build_context() returns formatted journal entries."""
        from enigma_engine.core.monologue import Journal
        j = Journal(journal_dir=tmp_path)
        j.add("The user enjoys math puzzles.")
        j.add("They prefer detailed explanations.")
        ctx = j.build_context(max_entries=5)
        assert "math puzzles" in ctx
        assert "detailed explanations" in ctx


# ════════════════════════════════════════════════════════════════════
# Coherence Scoring (Quality Gate)
# ════════════════════════════════════════════════════════════════════


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


# ════════════════════════════════════════════════════════════════════
# Reflection Prompt Building
# ════════════════════════════════════════════════════════════════════


class TestReflectionPrompt:
    """Tests for building reflection prompts from emotional state."""

    def test_returns_string(self):
        """Always returns a non-empty string."""
        from enigma_engine.core.monologue import build_reflection_prompt
        prompt = build_reflection_prompt(
            emotional_state={"valence": 0.3, "engagement": 0.7},
            recent_topics=["Python", "machine learning"],
            memory_facts=["User likes cats"],
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 10

    def test_includes_emotional_context(self):
        """Prompt incorporates emotional state when non-baseline."""
        from enigma_engine.core.monologue import build_reflection_prompt
        prompt = build_reflection_prompt(
            emotional_state={"valence": 0.8, "engagement": 0.9},
            recent_topics=[],
            memory_facts=[],
        )
        # Should mention the positive mood or engagement
        assert "positive" in prompt.lower() or "engaged" in prompt.lower() \
            or "good" in prompt.lower() or "conversation" in prompt.lower()

    def test_includes_topics(self):
        """Prompt mentions recent conversation topics."""
        from enigma_engine.core.monologue import build_reflection_prompt
        prompt = build_reflection_prompt(
            emotional_state={},
            recent_topics=["quantum physics", "cooking"],
            memory_facts=[],
        )
        assert "quantum physics" in prompt or "cooking" in prompt

    def test_empty_context(self):
        """Works with all-empty context."""
        from enigma_engine.core.monologue import build_reflection_prompt
        prompt = build_reflection_prompt(
            emotional_state={},
            recent_topics=[],
            memory_facts=[],
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ════════════════════════════════════════════════════════════════════
# Idle Timer Logic
# ════════════════════════════════════════════════════════════════════


class TestIdleDetection:
    """Tests for the idle detection helper."""

    def test_initial_not_idle(self):
        """Newly created tracker is not idle."""
        from enigma_engine.core.monologue import IdleTracker
        tracker = IdleTracker(idle_threshold_seconds=300)
        assert not tracker.is_idle()

    def test_activity_resets(self):
        """Recording activity resets the idle state."""
        from enigma_engine.core.monologue import IdleTracker
        tracker = IdleTracker(idle_threshold_seconds=0.01)
        tracker.record_activity()
        # Immediately after activity, should not be idle
        assert not tracker.is_idle()

    def test_becomes_idle(self):
        """Tracker reports idle after threshold passes."""
        from enigma_engine.core.monologue import IdleTracker
        tracker = IdleTracker(idle_threshold_seconds=0.01)
        tracker.record_activity()
        time.sleep(0.05)
        assert tracker.is_idle()

    def test_has_reflected_prevents_double_trigger(self):
        """After marking reflection done, doesn't trigger again until activity."""
        from enigma_engine.core.monologue import IdleTracker
        tracker = IdleTracker(idle_threshold_seconds=0.01)
        tracker.record_activity()
        time.sleep(0.05)
        assert tracker.is_idle()
        tracker.mark_reflected()
        # Now should not trigger idle again (already reflected)
        assert not tracker.is_idle()
        # Until new activity + idle again
        tracker.record_activity()
        time.sleep(0.05)
        assert tracker.is_idle()


# ════════════════════════════════════════════════════════════════════
# Monologue Mode Config
# ════════════════════════════════════════════════════════════════════


class TestMonologueMode:
    """Tests for the monologue mode setting."""

    def test_default_mode(self):
        """Default monologue mode is 'disabled'."""
        from enigma_engine.core.monologue import DEFAULT_MONOLOGUE_MODE
        assert DEFAULT_MONOLOGUE_MODE == "disabled"

    def test_valid_modes(self):
        """Valid monologue modes are defined."""
        from enigma_engine.core.monologue import MONOLOGUE_MODES
        assert "disabled" in MONOLOGUE_MODES
        assert "journal_only" in MONOLOGUE_MODES
        assert "automatic" in MONOLOGUE_MODES

    def test_coherence_threshold_default(self):
        """Default coherence threshold is 0.7."""
        from enigma_engine.core.monologue import DEFAULT_COHERENCE_THRESHOLD
        assert DEFAULT_COHERENCE_THRESHOLD == 0.7


# ════════════════════════════════════════════════════════════════════
# Greeting on App Open (Journal Entry Display)
# ════════════════════════════════════════════════════════════════════


class TestJournalGreeting:
    """Tests for greeting display from journal entries on model load."""

    def test_latest_entry_with_coherence(self, tmp_path):
        """Journal entry stores and returns coherence score."""
        from enigma_engine.core.monologue import Journal
        j = Journal(journal_dir=tmp_path)
        j.add("High quality reflection about the user.", coherence=0.85)
        entry = j.latest()
        assert entry is not None
        assert entry["coherence"] == 0.85

    def test_greeting_blocked_by_low_coherence(self, tmp_path):
        """Entry with low coherence would not pass the quality gate."""
        from enigma_engine.core.monologue import (
            Journal, DEFAULT_COHERENCE_THRESHOLD,
        )
        j = Journal(journal_dir=tmp_path)
        j.add("bad text", coherence=0.3)
        entry = j.latest()
        assert entry["coherence"] < DEFAULT_COHERENCE_THRESHOLD

    def test_greeting_allowed_by_high_coherence(self, tmp_path):
        """Entry with high coherence passes the quality gate."""
        from enigma_engine.core.monologue import (
            Journal, DEFAULT_COHERENCE_THRESHOLD,
        )
        j = Journal(journal_dir=tmp_path)
        j.add(
            "I noticed the user enjoys discussing physics and prefers "
            "detailed explanations with examples.",
            coherence=0.85,
        )
        entry = j.latest()
        assert entry["coherence"] >= DEFAULT_COHERENCE_THRESHOLD

    def test_greeting_empty_journal_returns_none(self, tmp_path):
        """latest() returns None on empty journal — no greeting."""
        from enigma_engine.core.monologue import Journal
        j = Journal(journal_dir=tmp_path)
        assert j.latest() is None


# ════════════════════════════════════════════════════════════════════
# Config Persistence (monologue_mode in forge_config)
# ════════════════════════════════════════════════════════════════════


class TestMonologueModeConfig:
    """Tests for monologue_mode persistence in CONFIG defaults."""

    def test_monologue_mode_in_defaults(self):
        """monologue_mode is present in CONFIG with default 'disabled'."""
        from enigma_engine.config import CONFIG
        assert CONFIG.get("monologue_mode") == "disabled"

    def test_monologue_mode_valid_values(self):
        """monologue_mode default matches valid monologue modes."""
        from enigma_engine.config import CONFIG
        from enigma_engine.core.monologue import MONOLOGUE_MODES
        assert CONFIG.get("monologue_mode") in MONOLOGUE_MODES


# ════════════════════════════════════════════════════════════════════
# Structural Tests (Wiring Verification)
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# Coherence Benchmark
# ════════════════════════════════════════════════════════════════════


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

        # Mock engine that returns coherent-ish text
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
