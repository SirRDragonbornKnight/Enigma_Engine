"""
Per-Model Context Storage
============================

Each AI model gets its own directory with persistent history and prompt.
When a model is loaded, its context auto-loads. When switching models,
the current model's context auto-saves.

Directory layout:
    data/model_contexts/<model_stem>/
        context.json   — system prompt, config overrides, metadata
        history.json   — chat messages array

Usage:
    ctx = ModelContext("enigma_small")
    ctx.load()
    ctx.history.append({"role": "user", "content": "hello"})
    ctx.system_prompt = "You are helpful."
    ctx.save()
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Base directory for all model contexts
_CONTEXTS_DIR = Path(__file__).parent.parent.parent / "data" / "model_contexts"


def get_contexts_dir() -> Path:
    """Return the base directory for model context storage."""
    return _CONTEXTS_DIR


class ModelContext:
    """Persistent per-model storage for history and prompt.

    Attributes:
        model_key:     Unique key derived from model filename stem.
        system_prompt: The system prompt for this model.
        history:       List of chat messages (role/content dicts).
        config:        Config overrides (temperature, etc.).
        last_used:     Timestamp of last save.
        profile_id:    ID of the active profile, if any.
    """

    def __init__(self, model_key: str) -> None:
        self.model_key = model_key
        self.system_prompt: str = "You are a helpful AI assistant."
        self.history: list[dict[str, str]] = []
        self.config: dict[str, Any] = {}
        self.last_used: float = 0.0
        self.profile_id: str = ""

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

    # ----------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------

    def load(self) -> None:
        """Load context and history from disk. No-op if files missing."""
        self._load_context()
        self._load_history()

    def _load_context(self) -> None:
        """Read context.json into attributes."""
        if not self.context_path.exists():
            return
        try:
            data = json.loads(
                self.context_path.read_text(encoding="utf-8"))
            self.system_prompt = data.get(
                "system_prompt", self.system_prompt)
            self.config = data.get("config", self.config)
            self.last_used = data.get("last_used", 0.0)
            self.profile_id = data.get("profile_id", "")
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
        """Write context.json from current attributes."""
        data = {
            "model_key": self.model_key,
            "system_prompt": self.system_prompt,
            "config": self.config,
            "last_used": self.last_used,
            "profile_id": self.profile_id,
        }
        try:
            self.context_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error(
                "Failed to save context for %s: %s",
                self.model_key, exc)

    def _save_history(self) -> None:
        """Write history.json from self.history."""
        data = {
            "model_key": self.model_key,
            "message_count": len(self.history),
            "saved_at": self.last_used,
            "messages": self.history,
        }
        try:
            self.history_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error(
                "Failed to save history for %s: %s",
                self.model_key, exc)

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
        system_prompt preview, and path.
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
        }
        if context_file.exists():
            try:
                data = json.loads(
                    context_file.read_text(encoding="utf-8"))
                entry["last_used"] = data.get("last_used", 0.0)
                entry["system_prompt"] = data.get(
                    "system_prompt", "")[:80]
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
