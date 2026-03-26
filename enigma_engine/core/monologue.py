"""
Inner Monologue & Journal System — Phase 5
==========================================

Per-model journal storage, coherence quality gate, reflection prompt
builder, and idle detection.  The monologue system lets the AI reflect
during idle periods, storing journal entries that persist with the model.

Architecture:
  - ``Journal`` — per-model markdown journal (read/write/trim)
  - ``score_coherence()`` — heuristic quality gate (0.0–1.0)
  - ``build_reflection_prompt()`` — assembles context for reflection
  - ``IdleTracker`` — detects when the user has been inactive

The quality gate (default threshold 0.7) prevents low-quality
reflections from reaching the user.  At current student model scale
(125M), most output will be journal-only.  Teacher models or larger
students can pass the gate for user-facing greetings.

Directory layout::

    data/model_contexts/<model_key>/journal.json

Usage::

    journal = Journal(journal_dir=model_context_dir)
    journal.add("The user seems interested in physics.")
    latest = journal.latest()  # most recent entry or None
    context = journal.build_context(max_entries=5)
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------

# Valid monologue modes:
#   disabled     — no background reflection at all
#   journal_only — AI reflects, stores journal, but never initiates
#   automatic    — AI reflects, stores journal, and can initiate
#                  greetings/thoughts if quality gate passes
MONOLOGUE_MODES = ("disabled", "journal_only", "automatic")
DEFAULT_MONOLOGUE_MODE = "disabled"

# Coherence threshold — model output must score above this
# to be shown to the user.  Below it → journal-only storage.
DEFAULT_COHERENCE_THRESHOLD = 0.7

# Maximum journal entries per model
_MAX_JOURNAL_ENTRIES = 200

# Idle threshold before triggering reflection (seconds)
DEFAULT_IDLE_SECONDS = 300  # 5 minutes


# ----------------------------------------------------------------
# Journal — Per-Model Persistent Storage
# ----------------------------------------------------------------

class Journal:
    """Per-model journal stored as ``journal.json``.

    Each entry has:
      - ``text``: the reflection content
      - ``timestamp``: ISO 8601 creation time
      - ``coherence``: float score at time of writing

    Thread-safe via file-level atomicity (atomic writes).
    """

    def __init__(
        self,
        journal_dir: Path,
        max_entries: int = _MAX_JOURNAL_ENTRIES,
    ) -> None:
        self._path = Path(journal_dir) / "journal.json"
        self._max_entries = max_entries
        self._entries: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Read journal from disk."""
        if not self._path.exists():
            self._entries = []
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            entries = data.get("entries", [])
            # Validate structure
            valid = []
            for e in entries:
                if isinstance(e, dict) and "text" in e:
                    valid.append(e)
            self._entries = valid
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load journal: %s", exc)
            self._entries = []

    def _save(self) -> None:
        """Write journal to disk atomically."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "entry_count": len(self._entries),
                "entries": self._entries,
            }
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(self._path, data)
        except OSError as exc:
            logger.error("Failed to save journal: %s", exc)

    def add(self, text: str, coherence: float = 0.0) -> None:
        """Add a journal entry. Rejects empty text."""
        text = text.strip()
        if not text:
            return
        entry = {
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "coherence": round(coherence, 3),
        }
        self._entries.append(entry)
        # Trim oldest if over capacity
        while len(self._entries) > self._max_entries:
            self._entries.pop(0)
        self._save()

    @property
    def entries(self) -> list[dict]:
        """Return a copy of all journal entries."""
        return list(self._entries)

    @property
    def count(self) -> int:
        """Number of journal entries."""
        return len(self._entries)

    def latest(self) -> dict | None:
        """Return the most recent entry, or None if empty."""
        if not self._entries:
            return None
        return self._entries[-1]

    def build_context(self, max_entries: int = 5) -> str:
        """Format recent journal entries for injection into prompts.

        Returns a string suitable for appending to the system prompt,
        or empty string if no entries.
        """
        if not self._entries:
            return ""
        recent = self._entries[-max_entries:]
        lines = ["[JOURNAL — Recent reflections]"]
        for entry in recent:
            ts = entry.get("timestamp", "")
            text = entry.get("text", "")
            lines.append(f"- ({ts}) {text}")
        return "\n".join(lines)


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
    """Score text coherence on a 0.0–1.0 scale.

    Heuristic approach — no model inference needed:
      1. Length adequacy (very short = low quality)
      2. Word variety (repetitive = low quality)
      3. Vocabulary overlap with common English (gibberish = low)
      4. Sentence structure (at least one proper sentence)

    This is intentionally conservative — the gate should let
    through obviously good text and reject obviously bad text.
    Borderline cases are scored low (fail-safe).
    """
    if not text or not text.strip():
        return 0.0

    text = text.strip()

    # --- Length score: 0–0.25 ---
    char_count = len(text)
    if char_count < 10:
        return 0.0  # single word / fragment — never passes
    length_score = min(char_count / 200, 1.0) * 0.25

    # --- Word variety: 0–0.25 ---
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0
    word_counts = Counter(words)
    unique_ratio = len(word_counts) / len(words) if words else 0
    variety_score = min(unique_ratio / 0.6, 1.0) * 0.25

    # --- Vocabulary overlap: 0–0.25 ---
    word_set = set(words)
    known = len(word_set & _COMMON_WORDS)
    overlap_ratio = known / len(word_set) if word_set else 0
    vocab_score = min(overlap_ratio / 0.3, 1.0) * 0.25

    # --- Sentence structure: 0–0.25 ---
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
# Reflection Prompt Builder
# ----------------------------------------------------------------

def build_reflection_prompt(
    emotional_state: dict[str, float] | None = None,
    recent_topics: list[str] | None = None,
    memory_facts: list[str] | None = None,
) -> str:
    """Build a prompt that asks the model to reflect.

    Assembles available context (emotional state, recent conversation
    topics, memory facts) into a prompt that invites introspection.
    The model's response becomes a journal entry.

    Args:
        emotional_state: Current emotional dimensions (may be empty).
        recent_topics: Keywords from recent conversations.
        memory_facts: Known facts about the user from PersistentMemory.

    Returns:
        A system-prompt-style string for generating a reflection.
    """
    parts = [
        "You are reflecting quietly between conversations. "
        "Write a brief, honest internal thought — something you "
        "noticed about recent interactions, a pattern you observed, "
        "or something you want to remember for next time. "
        "Keep it to 1-3 sentences. Be genuine, not performative."
    ]

    # Add emotional context
    if emotional_state:
        from enigma_engine.core.model_context import _EMOTIONAL_BASELINE
        significant = []
        for key, val in emotional_state.items():
            baseline = _EMOTIONAL_BASELINE.get(key, 0.0)
            if abs(val - baseline) >= 0.2:
                if key == "valence":
                    mood = "positive" if val > 0 else "negative"
                    significant.append(f"mood is {mood}")
                elif key == "engagement":
                    level = "high" if val > baseline else "low"
                    significant.append(f"engagement is {level}")
                elif key == "frustration" and val > 0.3:
                    significant.append("some frustration detected")
                elif key == "trust":
                    level = "high" if val > baseline else "low"
                    significant.append(f"trust level is {level}")
        if significant:
            parts.append(
                f"\nEmotional context: {', '.join(significant)}."
            )

    # Add recent topics
    if recent_topics:
        topics_str = ", ".join(recent_topics[:5])
        parts.append(f"\nRecent conversation topics: {topics_str}.")

    # Add memory facts
    if memory_facts:
        facts_str = "; ".join(memory_facts[:5])
        parts.append(f"\nKnown about the user: {facts_str}.")

    return "\n".join(parts)


# ----------------------------------------------------------------
# Idle Tracker
# ----------------------------------------------------------------

class IdleTracker:
    """Tracks user activity to detect idle periods.

    Usage:
        tracker = IdleTracker(idle_threshold_seconds=300)
        tracker.record_activity()  # call on each user interaction
        if tracker.is_idle():
            # trigger reflection
            tracker.mark_reflected()
    """

    def __init__(self, idle_threshold_seconds: float = DEFAULT_IDLE_SECONDS):
        self._threshold = idle_threshold_seconds
        self._last_activity: float = time.monotonic()
        self._reflected_since_activity: bool = False

    def record_activity(self) -> None:
        """Reset idle timer — call on user interaction."""
        self._last_activity = time.monotonic()
        self._reflected_since_activity = False

    def is_idle(self) -> bool:
        """Return True if idle threshold has passed and not yet reflected."""
        if self._reflected_since_activity:
            return False
        elapsed = time.monotonic() - self._last_activity
        return elapsed >= self._threshold

    def mark_reflected(self) -> None:
        """Mark that a reflection has been done for this idle period."""
        self._reflected_since_activity = True


# ----------------------------------------------------------------
# Coherence Benchmark — Teacher-Scored Evaluation
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
    on_progress: "callable | None" = None,
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
            "Write a brief, honest internal thought — something you "
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
