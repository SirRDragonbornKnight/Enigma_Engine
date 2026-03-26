"""
Per-Model Context Storage
============================

Each AI model gets its own directory with persistent history and prompt.
When a model is loaded, its context auto-loads. When switching models,
the current model's context auto-saves.

Every model also has an identity — display name, personality, avatar,
stats (message/session counts), training history, tags, and notes.
This makes each model feel like a distinct AI with its own profile.

Directory layout:
    data/model_contexts/<model_stem>/
        context.json   — system prompt, config overrides, identity, metadata
        history.json   — chat messages array

Usage:
    ctx = ModelContext("enigma_small")
    ctx.load()
    ctx.history.append({"role": "user", "content": "hello"})
    ctx.system_prompt = "You are helpful."
    ctx.display_name = "Enigma"
    ctx.save()
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Base directory for all model contexts
_CONTEXTS_DIR = Path(__file__).parent.parent.parent / "data" / "model_contexts"
_PROMPTS_DIR = Path(__file__).parent.parent.parent / "data" / "prompts"

# Maximum number of messages to persist in history.json.
# Older messages beyond this cap are dropped on save.
# This prevents unbounded history growth on disk while keeping
# full history available in-memory for the current session.
MAX_CONTEXT_HISTORY = 500

# Emotional state baseline — neutral starting point.
# Valence: 0.0 (neutral), Arousal: 0.2 (calm), Engagement: 0.5 (moderate),
# Trust: 0.5 (neutral), Frustration: 0.0 (patient).
_EMOTIONAL_BASELINE: dict[str, float] = {
    "valence": 0.0,
    "arousal": 0.2,
    "engagement": 0.5,
    "trust": 0.5,
    "frustration": 0.0,
}

# How much each sentiment update shifts the state (0-1 blend factor).
# Low value = slow, gradual change. High = reactive.
_EMOTIONAL_LERP = 0.3

# How much state decays toward baseline per call (idle drift).
_EMOTIONAL_DECAY = 0.1

# Hard ranges per dimension.
_EMOTIONAL_RANGES: dict[str, tuple[float, float]] = {
    "valence": (-1.0, 1.0),
    "arousal": (0.0, 1.0),
    "engagement": (0.0, 1.0),
    "trust": (0.0, 1.0),
    "frustration": (0.0, 1.0),
}


def set_max_context_history(value: int) -> None:
    """Update the history persistence cap at runtime."""
    global MAX_CONTEXT_HISTORY
    MAX_CONTEXT_HISTORY = max(1, int(value))


def get_contexts_dir() -> Path:
    """Return the base directory for model context storage."""
    return _CONTEXTS_DIR


class ModelContext:
    """Persistent per-model storage for history, prompt, and identity.

    Attributes:
        model_key:        Unique key derived from model filename stem.
        system_prompt:    The system prompt for this model.
        history:          List of chat messages (role/content dicts).
        config:           Config overrides (temperature, etc.).
        last_used:        Timestamp of last save.

        Identity fields:
        display_name:     Human-friendly name for this AI.
        personality:      Short personality description.
        avatar:           Path to avatar image file.
        created_at:       ISO timestamp of first creation.
        total_messages:   Lifetime message count.
        total_sessions:   Lifetime session count.
        training_history: List of training run records.
        tags:             User-defined tags for organization.
        notes:            Freeform notes about this model.
    """

    def __init__(self, model_key: str) -> None:
        self.model_key = model_key
        self.system_prompt: str = self._default_prompt()
        self.history: list[dict[str, str]] = []
        self.config: dict[str, Any] = {}
        self.last_used: float = 0.0

        # Active session file path — persisted so reload resumes
        # the same session instead of creating a duplicate.
        self.session_path: str = ""

        # Identity fields
        self.display_name: str = ""
        self.personality: str = ""
        self.avatar: str = ""
        self.created_at: str = datetime.now(
            timezone.utc).isoformat(timespec="seconds")
        self.total_messages: int = 0
        self.total_sessions: int = 0
        self.training_history: list[dict[str, Any]] = []
        self.tags: list[str] = []
        self.notes: str = ""

        # Emotional state — persistent per-model internal state
        self.emotional_state: dict[str, float] = dict(_EMOTIONAL_BASELINE)
        self._emotional_lock = threading.Lock()

        # Journal — lazily loaded on first access
        self._journal = None

    @staticmethod
    def _default_prompt() -> str:
        """Load default prompt from data/prompts/chat.md, fallback to builtin."""
        for suffix in (".md", ".txt"):
            path = _PROMPTS_DIR / f"chat{suffix}"
            if path.exists():
                try:
                    return path.read_text(encoding="utf-8").strip()
                except OSError:
                    continue
        return "You are a helpful AI assistant."

    # ----------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------

    @property
    def context_dir(self) -> Path:
        """Directory for this model's context files."""
        return _CONTEXTS_DIR / self.model_key

    @property
    def context_path(self) -> Path:
        """Path to context.json (prompt, config, metadata)."""
        return self.context_dir / "context.json"

    @property
    def history_path(self) -> Path:
        """Path to history.json (chat messages)."""
        return self.context_dir / "history.json"

    @property
    def journal(self):
        """Per-model journal (lazy-loaded on first access)."""
        if self._journal is None:
            from enigma_engine.core.monologue import Journal
            self.context_dir.mkdir(parents=True, exist_ok=True)
            self._journal = Journal(journal_dir=self.context_dir)
        return self._journal

    # ----------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------

    def load(self) -> None:
        """Load context and history from disk. No-op if files missing."""
        self._load_context()
        self._load_history()

    def _load_context(self) -> None:
        """Read context.json into attributes (supports old + new format)."""
        if not self.context_path.exists():
            return
        try:
            data = json.loads(
                self.context_path.read_text(encoding="utf-8"))
            self.system_prompt = data.get(
                "system_prompt", self.system_prompt)
            self.config = data.get("config", self.config)
            self.last_used = data.get("last_used", 0.0)

            # Identity fields — graceful defaults for old context.json
            self.session_path = data.get("session_path", "")
            self.display_name = data.get("display_name", "")
            self.personality = data.get("personality", "")
            self.avatar = data.get("avatar", "")
            self.created_at = data.get(
                "created_at", self.created_at)
            self.total_messages = data.get("total_messages", 0)
            self.total_sessions = data.get("total_sessions", 0)
            self.training_history = data.get(
                "training_history", [])
            self.tags = data.get("tags", [])
            self.notes = data.get("notes", "")

            # Emotional state — load saved or keep baseline
            saved_emo = data.get("emotional_state")
            if isinstance(saved_emo, dict):
                for key in _EMOTIONAL_BASELINE:
                    if key in saved_emo:
                        lo, hi = _EMOTIONAL_RANGES[key]
                        self.emotional_state[key] = max(
                            lo, min(hi, float(saved_emo[key])))

            logger.info(
                "Loaded context for model: %s", self.model_key)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to load context for %s: %s",
                self.model_key, exc)

    def _load_history(self) -> None:
        """Read history.json into self.history."""
        if not self.history_path.exists():
            return
        try:
            data = json.loads(
                self.history_path.read_text(encoding="utf-8"))
            messages = data.get("messages", [])
            # Validate each message has role and content
            valid = []
            for msg in messages:
                if ("role" in msg and "content" in msg
                        and isinstance(msg["content"], str)):
                    valid.append({
                        "role": msg["role"],
                        "content": msg["content"],
                    })
            self.history = valid
            logger.info(
                "Loaded %d messages for model: %s",
                len(valid), self.model_key)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to load history for %s: %s",
                self.model_key, exc)

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------

    def save(self) -> None:
        """Persist context and history to disk."""
        self.context_dir.mkdir(parents=True, exist_ok=True)
        self.last_used = time.time()
        self._save_context()
        self._save_history()

    def _save_context(self) -> None:
        """Write context.json from current attributes (includes identity)."""
        data = {
            "model_key": self.model_key,
            "system_prompt": self.system_prompt,
            "config": self.config,
            "last_used": self.last_used,
            "session_path": self.session_path,
            # Identity fields
            "display_name": self.display_name,
            "personality": self.personality,
            "avatar": self.avatar,
            "created_at": self.created_at,
            "total_messages": self.total_messages,
            "total_sessions": self.total_sessions,
            "training_history": self.training_history,
            "tags": self.tags,
            "notes": self.notes,
            "emotional_state": dict(self.emotional_state),
        }
        try:
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(self.context_path, data)
        except OSError as exc:
            logger.error(
                "Failed to save context for %s: %s",
                self.model_key, exc)

    def _save_history(self) -> None:
        """Write history.json from self.history.

        Caps saved messages at MAX_CONTEXT_HISTORY — keeps the most
        recent messages and drops oldest.  In-memory history is not
        modified so the current session retains full context.
        """
        messages = self.history
        if len(messages) > MAX_CONTEXT_HISTORY:
            messages = messages[-MAX_CONTEXT_HISTORY:]
        data = {
            "model_key": self.model_key,
            "message_count": len(messages),
            "saved_at": self.last_used,
            "messages": messages,
        }
        try:
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(self.history_path, data)
        except OSError as exc:
            logger.error(
                "Failed to save history for %s: %s",
                self.model_key, exc)

    # ----------------------------------------------------------------
    # Identity helpers
    # ----------------------------------------------------------------

    def increment_messages(self, count: int = 1) -> None:
        """Increment the lifetime message counter."""
        self.total_messages += count

    def increment_sessions(self) -> None:
        """Increment the lifetime session counter."""
        self.total_sessions += 1

    def record_training_run(
        self, *, mode: str, epochs: int, best_loss: float,
        before_perplexity: float | None = None,
        after_perplexity: float | None = None,
    ) -> None:
        """Append a training run record to training_history."""
        entry: dict = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "mode": mode,
            "epochs": epochs,
            "best_loss": round(best_loss, 4),
        }
        if before_perplexity is not None:
            entry["before_perplexity"] = round(before_perplexity, 4)
        if after_perplexity is not None:
            entry["after_perplexity"] = round(after_perplexity, 4)
        self.training_history.append(entry)

    @property
    def memory_fact_count(self) -> int:
        """Return the number of facts in PersistentMemory (0 if unavailable)."""
        try:
            from enigma_engine.core.memory import PersistentMemory
            mem = PersistentMemory()
            return len(mem.facts)
        except Exception:
            return 0

    def export_identity(self) -> dict[str, Any]:
        """Export identity fields as a standalone dict for sharing.

        Returns a dict with all identity fields plus model_key.
        Can be written to a JSON file for sharing or backup.
        """
        return {
            "model_key": self.model_key,
            "display_name": self.display_name,
            "personality": self.personality,
            "avatar": self.avatar,
            "created_at": self.created_at,
            "total_messages": self.total_messages,
            "total_sessions": self.total_sessions,
            "training_history": list(self.training_history),
            "tags": list(self.tags),
            "notes": self.notes,
            "memory_facts": self.memory_fact_count,
            "emotional_state": dict(self.emotional_state),
        }

    # ----------------------------------------------------------------
    # Emotional state
    # ----------------------------------------------------------------

    def update_emotional_state(self, user_message: str) -> None:
        """Update emotional state based on user message sentiment.

        Blends current state toward the detected sentiment using
        a lerp factor.  Values are clamped to their defined ranges.

        Args:
            user_message: The raw text the user sent.
        """
        from enigma_engine.core.sentiment import analyze_sentiment
        signals = analyze_sentiment(user_message)
        with self._emotional_lock:
            for key, signal in signals.items():
                current = self.emotional_state.get(key, _EMOTIONAL_BASELINE[key])
                # Lerp toward signal
                new_val = current + _EMOTIONAL_LERP * (signal - current)
                lo, hi = _EMOTIONAL_RANGES[key]
                self.emotional_state[key] = round(
                    max(lo, min(hi, new_val)), 3)

    def decay_emotional_state(self) -> None:
        """Drift emotional state toward baseline.

        Called when idle or between sessions to prevent
        permanent extremes.
        """
        with self._emotional_lock:
            for key, baseline in _EMOTIONAL_BASELINE.items():
                current = self.emotional_state.get(key, baseline)
                new_val = current + _EMOTIONAL_DECAY * (baseline - current)
                lo, hi = _EMOTIONAL_RANGES[key]
                self.emotional_state[key] = round(
                    max(lo, min(hi, new_val)), 3)

    def reset_emotional_state(self) -> None:
        """Reset emotional state to neutral baseline.

        Called on major retraining or by user request.
        """
        with self._emotional_lock:
            self.emotional_state = dict(_EMOTIONAL_BASELINE)

    # ----------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------

    def clear_history(self) -> None:
        """Clear in-memory history (does not delete file)."""
        self.history.clear()

    def delete(self) -> None:
        """Remove this model's context directory from disk."""
        import shutil
        if self.context_dir.exists():
            shutil.rmtree(self.context_dir, ignore_errors=True)
            logger.info(
                "Deleted context for model: %s", self.model_key)


# ================================================================
# Helper functions
# ================================================================

def model_key_from_path(model_path: str) -> str:
    """Derive a context key from a model file path.

    Uses the stem of the filename, lowercased, with spaces
    replaced by underscores.  For example:
        models/Enigma_Small.pth  ->  enigma_small
        models/qwen2.5-32b-instruct/model.safetensors
            -> qwen2.5-32b-instruct
    """
    p = Path(model_path)
    stem = p.stem.lower().replace(" ", "_")
    # If stem is generic (model, weights, etc.), use parent dir name
    generic_stems = {"model", "weights", "pytorch_model",
                     "model-00001-of-00001"}
    if stem in generic_stems and p.parent.name:
        stem = p.parent.name.lower().replace(" ", "_")
    return stem


def load_model_context(model_path: str) -> ModelContext:
    """Create and load a ModelContext for the given model path."""
    key = model_key_from_path(model_path)
    ctx = ModelContext(key)
    ctx.load()
    return ctx


def list_model_contexts() -> list[dict[str, Any]]:
    """List all saved model contexts.

    Returns:
        List of dicts with model_key, last_used, message_count,
        system_prompt preview, identity fields, and path.
    """
    results: list[dict[str, Any]] = []
    if not _CONTEXTS_DIR.exists():
        return results
    for d in sorted(_CONTEXTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        context_file = d / "context.json"
        history_file = d / "history.json"
        entry: dict[str, Any] = {
            "model_key": d.name,
            "path": str(d),
            "last_used": 0.0,
            "message_count": 0,
            "system_prompt": "",
            # Identity fields
            "display_name": "",
            "tags": [],
            "total_messages": 0,
            "total_sessions": 0,
        }
        if context_file.exists():
            try:
                data = json.loads(
                    context_file.read_text(encoding="utf-8"))
                entry["last_used"] = data.get("last_used", 0.0)
                entry["system_prompt"] = data.get(
                    "system_prompt", "")[:80]
                entry["display_name"] = data.get(
                    "display_name", "")
                entry["tags"] = data.get("tags", [])
                entry["total_messages"] = data.get(
                    "total_messages", 0)
                entry["total_sessions"] = data.get(
                    "total_sessions", 0)
            except (json.JSONDecodeError, OSError):
                pass
        if history_file.exists():
            try:
                data = json.loads(
                    history_file.read_text(encoding="utf-8"))
                entry["message_count"] = data.get(
                    "message_count", 0)
            except (json.JSONDecodeError, OSError):
                pass
        results.append(entry)
    return results
