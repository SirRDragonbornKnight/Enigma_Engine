"""
Enigma AI Engine - Local API Server
====================================

FastAPI server that exposes the engine over HTTP for local use.
Connect from any device on your local network (PC, phone, tablet).

Usage:
    python run.py --serve              # Start on port 8080
    python run.py --serve --port 9000  # Custom port

Endpoints:
    GET  /api/health        Health check
    GET  /api/system        System/hardware info
    GET  /api/models        List available models
    GET  /api/models/status Current model status
    POST /api/models/load   Load a model
    POST /api/models/unload Unload current model
    POST /api/chat          Send a chat message
    POST /api/chat/stream   Stream chat via SSE
    POST /api/batch         Batch inference (multiple prompts)
    GET  /api/profiles      List AI profiles
    GET  /api/profiles/{id} Get profile details
    POST /api/profiles/{id}/activate  Activate a profile
    GET  /api/config        Get generation config
    POST /api/config        Update generation config
    GET  /api/history       Legacy alias -- returns active-conversation history
    DELETE /api/history     Clear ALL conversations + KV cache (legacy alias)
    GET  /api/training/status  Training progress
    POST /api/train         Start training

Conversation endpoints (MC-1):
    POST   /api/conversations            Create a new conversation thread
    GET    /api/conversations            List all conversation IDs + metadata
    DELETE /api/conversations/{id}       Delete a conversation
    GET    /api/conversations/{id}/history  Per-conversation history
"""
from __future__ import annotations

import hmac
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from enigma_engine import __version__
from enigma_engine.core.json_schema_mask import validate_json_schema_shape
logger = logging.getLogger(__name__)

# Maximum chat history entries (user + assistant pairs) per conversation.
# Oldest entries evicted when exceeded.  Prevents unbounded memory growth
# on long-running server sessions.
MAX_HISTORY = 10_000

# Maximum simultaneous conversations the daemon will keep in memory
# (MC-1). LRU eviction kicks in past this number -- the least-recently
# touched conversation is dropped when a new one is created.  Bounds
# memory at MAX_CONVERSATIONS * MAX_HISTORY entries.
# D2 fix (Pass 156z9dc): floor clamped to 2.  A value of 1 is a soft
# brick -- _evict_locked skips the active conversation so the count
# permanently sits at 2 (active + one pending) until one is deleted.
_MAX_CONVERSATIONS_RAW = 100
MAX_CONVERSATIONS = max(_MAX_CONVERSATIONS_RAW, 2)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
API_DIR = Path(__file__).parent
PROJECT_ROOT = API_DIR.parent.parent
PROFILES_DIR = PROJECT_ROOT / "profiles"
MODELS_DIR = PROJECT_ROOT / "models"
# MC-2: on-disk conversation store.  Each conversation is one JSONL file
# named <conversation_id>.jsonl.  Created on first write; absent means
# the conversation has no history yet.
CONVERSATIONS_DIR = PROJECT_ROOT / "data" / "conversations"

# MC-2b: persisted active-conversation pointer.  One JSON file alongside
# the conversations directory so the active id survives daemon restarts.
# Skipped on load if the saved id is no longer in ``_histories`` (the
# user evicted/deleted it between sessions).
ACTIVE_CONV_FILE = CONVERSATIONS_DIR / "_active.json"

# ---------------------------------------------------------------------------
# App State (module-level singleton — one engine per process)
# ---------------------------------------------------------------------------

class AppState:
    """Holds the loaded engine, config overrides, and per-conversation chat history.

    All mutable state is guarded by ``_lock``.  Read-only accessors
    return *snapshots* (copy-on-write) so callers never hold a
    reference to the live internal list/dict.

    MC-1: history is scoped per ``conversation_id`` so multiple
    clients (terminal REPL, desktop GUI, future browser viewer) don't
    cross-contaminate threads.  The engine's KV cache is invalidated
    on every conversation switch — leaving it primed with another
    conversation's tokens would produce garbage continuations.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.engine: Any = None
        self.model_path: str | None = None
        self._model_info: dict[str, Any] = {}
        # MC-1: per-conversation history map.  Keys are conversation
        # IDs (UUID4 strings by default), values are lists of
        # {"role", "content"} dicts.  Insertion order in
        # ``_conv_order`` doubles as the LRU ordering for eviction:
        # touched conversations move to the end.
        self._histories: dict[str, list[dict[str, str]]] = {}
        self._conv_order: list[str] = []
        self._active_conv_id: str | None = None
        self.active_profile: str | None = None
        self.config_overrides: dict[str, Any] = {}
        self.start_time: float = time.time()

    # -- Copy-on-write snapshots (Suggestion #8D) ----------------------

    def history_snapshot(
        self, conversation_id: str | None = None
    ) -> list[dict[str, str]]:
        """Return a shallow copy of a conversation's history.

        ``conversation_id=None`` returns the currently-active
        conversation, or ``[]`` if none has been started yet.
        Raises ``KeyError`` if a specific ID was supplied and is unknown.
        """
        with self._lock:
            if conversation_id is None:
                if self._active_conv_id is None:
                    return []
                return list(self._histories.get(self._active_conv_id, []))
            if conversation_id not in self._histories:
                raise KeyError(conversation_id)
            return list(self._histories[conversation_id])

    @property
    def model_info(self) -> dict[str, Any]:
        """Direct access for internal mutations."""
        return self._model_info

    @model_info.setter
    def model_info(self, value: dict[str, Any]) -> None:
        with self._lock:
            self._model_info = value

    def model_info_snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of model info (safe for readers)."""
        with self._lock:
            return dict(self._model_info)

    # -- Conversation management (MC-1) -------------------------------

    def create_conversation(self) -> str:
        """Allocate a fresh conversation ID and return it.

        Triggers LRU eviction if ``MAX_CONVERSATIONS`` would be exceeded.
        """
        cid = uuid.uuid4().hex
        with self._lock:
            self._histories[cid] = []
            self._conv_order.append(cid)
            evicted = self._evict_locked()
        # MC-2: delete evicted conversation files outside the lock.
        for eid in evicted:
            self._delete_disk_conversation(eid)
        return cid

    def list_conversations(self) -> list[dict[str, Any]]:
        """List known conversations with message counts (LRU order)."""
        with self._lock:
            return [
                {"id": cid,
                 "messages": len(self._histories[cid]),
                 "active": cid == self._active_conv_id}
                for cid in self._conv_order
                if cid in self._histories
            ]

    def delete_conversation(self, conversation_id: str) -> None:
        """Drop a conversation.  Clears engine state if it was active.

        Raises ``KeyError`` if the ID is unknown.
        """
        with self._lock:
            if conversation_id not in self._histories:
                raise KeyError(conversation_id)
            was_active = conversation_id == self._active_conv_id
            del self._histories[conversation_id]
            try:
                self._conv_order.remove(conversation_id)
            except ValueError:
                pass
            if was_active:
                self._active_conv_id = None
        if was_active:
            self._invalidate_engine_state()
        # MC-2: delete the on-disk file regardless of whether it was active.
        self._delete_disk_conversation(conversation_id)
        # MC-2b: persist updated active pointer if it changed.
        if was_active:
            self._persist_active_conv_id()

    def _evict_locked(self) -> list[str]:
        """LRU-evict oldest conversations until under MAX_CONVERSATIONS.

        Must be called while ``self._lock`` is held.  Never evicts
        the currently-active conversation.  Returns a list of evicted
        IDs so the caller can clean up on-disk files outside the lock.
        """
        evicted: list[str] = []
        while len(self._conv_order) > MAX_CONVERSATIONS:
            for victim in list(self._conv_order):
                if victim == self._active_conv_id:
                    continue
                self._conv_order.remove(victim)
                self._histories.pop(victim, None)
                evicted.append(victim)
                break
            else:
                # Every remaining conversation is the active one.  Bail
                # to avoid an infinite loop; the operator picked an
                # impossibly small MAX_CONVERSATIONS.
                break
        return evicted

    def _touch_locked(self, conversation_id: str) -> None:
        """Move a conversation to the end of the LRU queue.

        Must be called while ``self._lock`` is held. MC-1a B3: refuses
        to re-add a conversation that is no longer in ``_histories`` —
        otherwise a TOCTOU race between resolve/activate and DELETE can
        silently resurrect a deleted conv in the LRU queue.
        """
        if conversation_id not in self._histories:
            try:
                self._conv_order.remove(conversation_id)
            except ValueError:
                pass
            return
        try:
            self._conv_order.remove(conversation_id)
        except ValueError:
            pass
        self._conv_order.append(conversation_id)

    def _invalidate_engine_state(self) -> None:
        """Clear engine KV cache + history-summary cache.

        Called on conversation switch and on full history clear.
        Both engine methods are best-effort: legacy engines may not
        implement them.
        """
        engine = self.engine
        if engine is None:
            return
        if hasattr(engine, "clear_kv_cache"):
            try:
                engine.clear_kv_cache()
            except Exception as exc:
                logger.warning("Engine clear_kv_cache raised: %s", exc)
        if hasattr(engine, "clear_history"):
            try:
                engine.clear_history()
            except Exception as exc:
                logger.warning("Engine clear_history raised: %s", exc)

    def _resolve_and_activate(
            self, conversation_id: str | None) -> tuple[str, bool]:
        """Resolve + activate atomically under a single lock acquisition.

        MC-1a B3 fix: separate ``_resolve_conversation`` + ``_activate``
        calls create a TOCTOU window — a concurrent ``DELETE
        /api/conversations/{cid}`` between the two unlocked sections
        could leave ``_active_conv_id`` pointing at a deleted id and
        cause downstream ``setdefault`` calls to resurrect it.

        Auto-creation (``conversation_id is None``) is delegated to
        ``create_conversation`` outside this method, which is itself
        atomic; the returned id then enters this method to be activated.
        Raises ``KeyError`` if a specific id is supplied and unknown.
        Returns ``(conv_id, switched)``.
        """
        if conversation_id is None:
            conversation_id = self.create_conversation()
        with self._lock:
            if conversation_id not in self._histories:
                raise KeyError(conversation_id)
            switched = self._active_conv_id != conversation_id
            self._active_conv_id = conversation_id
            self._touch_locked(conversation_id)
        # MC-2b: persist active pointer when it changes so the next
        # daemon boot resumes on the same conversation.
        if switched:
            self._persist_active_conv_id()
        return conversation_id, switched

    def _append_turn_if_alive_locked(
            self, conversation_id: str,
            user_msg: str, assistant_msg: str) -> bool:
        """Append a user/assistant pair only if the conv still exists.

        MC-1a B3 fix: replaces ``_histories.setdefault(cid, [])`` in the
        post-generation history-tracking blocks. If the conversation was
        deleted while the engine was generating, the response is still
        returned to the caller (already produced) but the history is NOT
        resurrected. Returns ``True`` if the turn was stored.
        Must be called while ``self._lock`` is held.
        """
        conv = self._histories.get(conversation_id)
        if conv is None:
            return False
        conv.append({"role": "user", "content": user_msg})
        conv.append({"role": "assistant", "content": assistant_msg})
        self._trim_history_locked(conversation_id)
        return True

    # -- Engine management --------------------------------------------------

    def load_model(self, model_path: str) -> dict[str, Any]:
        """Load a model into the engine."""
        from enigma_engine.core import EnigmaEngine

        # Unload previous
        self.unload_model()

        try:
            engine = EnigmaEngine(model_path=model_path)
        except Exception:
            # Cleanup any partially loaded state
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            raise

        # Gather info
        param_count = 0
        if hasattr(engine, "model") and engine.model is not None:
            param_count = sum(p.numel() for p in engine.model.parameters())

        device = "cpu"
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass

        info = {
            "path": model_path,
            "parameters": param_count,
            "device": device,
            "loaded": True,
        }

        with self._lock:
            self.engine = engine
            self.model_path = model_path
            self._model_info = info
        return dict(info)

    def unload_model(self):
        """Unload the current model and free memory."""
        with self._lock:
            if self.engine is not None:
                del self.engine
                self.engine = None
                self.model_path = None
                self._model_info = {}
                self._histories.clear()
                self._conv_order.clear()
                self._active_conv_id = None

            # Free GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        # MC-2b: clear the active pointer on disk after unload.
        self._persist_active_conv_id()

    def chat(self, message: str, temperature: float | None = None,
             max_tokens: int | None = None,
             top_p: float | None = None,
             top_k: int | None = None,
             repetition_penalty: float | None = None,
             json_schema: dict[str, Any] | None = None,
             system_prompt: str | None = None,
             conversation_id: str | None = None) -> tuple[str, str]:
        """Send a message to the engine and get a response.

        MC-1: ``conversation_id`` scopes the history.  ``None``
        auto-creates a new conversation.  Returns ``(response, conv_id)``.
        On conversation switch the engine KV cache is cleared and the
        full conversation history is passed in via the ``history``
        kwarg so the engine prefills against the right context.

        N-15b: ``json_schema`` is forwarded through engine.chat() into
        engine.generate(), where ``JsonSchemaConstraint`` masks logits
        per token so the response is structurally valid JSON. GGUF
        models raise ``NotImplementedError`` (their sampler is
        in-process C++ and never sees our mask).

        Pass 156z9g: ``system_prompt`` lets callers (the auto-research
        wiring, primarily) inject per-request context without mutating
        the engine's persistent ``system_prompt`` attribute. Forwarded
        when the underlying ``engine.chat`` accepts it; falls through
        the existing TypeError fallback otherwise.
        """
        if self.engine is None:
            raise RuntimeError("No model loaded")

        # Resolve / activate the conversation.  KeyError bubbles out
        # so the route handler can return 404. MC-1a B3: single locked
        # critical section so a concurrent DELETE cannot squeeze in.
        conv_id, switched = self._resolve_and_activate(conversation_id)
        if switched:
            self._invalidate_engine_state()

        # Build kwargs from config overrides + per-request overrides
        kwargs: dict[str, Any] = {}
        if self.config_overrides:
            kwargs.update(self.config_overrides)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = repetition_penalty
        if json_schema is not None:
            kwargs["json_schema"] = json_schema
        if system_prompt is not None:
            kwargs["system_prompt"] = system_prompt

        # MC-1: pass the per-conversation history explicitly so the
        # engine prefills against the right context regardless of its
        # internal state.
        history = self.history_snapshot(conv_id)
        try:
            response = self.engine.chat(message, history=history, **kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" in msg or "got an unexpected" in msg:
                logger.debug(
                    "Engine.chat() does not accept kwargs, retrying without")
                response = self.engine.chat(message)
            else:
                raise

        # Track history under lock. MC-1a B3: skip the append if the
        # conv was deleted mid-generation rather than resurrecting it.
        with self._lock:
            stored = self._append_turn_if_alive_locked(
                conv_id, message, response)

        # MC-2: persist after every exchange (outside the lock to keep I/O off the hot path).
        if stored:
            self._persist_conversation(conv_id)
        return response, conv_id

    def _trim_history_locked(self, conversation_id: str) -> None:
        """Evict oldest entries when a conversation exceeds MAX_HISTORY.

        Must be called while self._lock is held.
        """
        conv = self._histories.get(conversation_id)
        if conv is None:
            return
        if len(conv) > MAX_HISTORY:
            excess = len(conv) - MAX_HISTORY
            del conv[:excess]

    def clear_all_conversations(self) -> None:
        """Clear all conversations and invalidate engine state.

        Used by ``DELETE /api/history`` (legacy nuke-everything route).
        """
        with self._lock:
            all_ids = list(self._histories.keys())
            self._histories.clear()
            self._conv_order.clear()
            self._active_conv_id = None
        self._invalidate_engine_state()
        # MC-2: purge all on-disk files.
        for cid in all_ids:
            self._delete_disk_conversation(cid)
        # MC-2b: clear the active pointer on disk too.
        self._persist_active_conv_id()

    def rollback_last_turn(self, conversation_id: str) -> bool:
        """Drop the last user+assistant pair from a conversation.

        Returns ``True`` if a pair was removed, ``False`` otherwise.
        Used by AutoResearch-2 retry on ``/api/chat`` so the retry call
        sees clean history instead of the low-confidence reply it is
        about to replace.  Raises ``KeyError`` if the conversation is
        unknown.
        """
        with self._lock:
            if conversation_id not in self._histories:
                raise KeyError(conversation_id)
            conv = self._histories[conversation_id]
            if len(conv) < 2:
                return False
            # Only roll back when the tail looks like a complete
            # exchange (user followed by assistant).  Anything else
            # means a caller has been mutating the list out-of-band;
            # leave it alone.
            if (conv[-2].get("role") != "user"
                    or conv[-1].get("role") != "assistant"):
                return False
            del conv[-2:]
        # MC-2: persist rollback result.
        self._persist_conversation(conversation_id)
        return True

    # -- MC-2: disk persistence -----------------------------------------

    def _persist_conversation(self, conversation_id: str) -> None:
        """Atomically write a conversation's history to disk.

        Writes ``CONVERSATIONS_DIR/<conversation_id>.jsonl``.
        Each line is a JSON object ``{"role": ..., "content": ...}``.
        Skipped silently when ``CONVERSATIONS_DIR`` is not writable so
        that unit tests without the real filesystem path don't break.

        MC-2 B3 sibling: if the conversation was deleted between the
        post-generation history append and this persist call, skip the
        write. Without this gate ``_histories.get(cid, [])`` returns the
        empty default and we'd write a phantom empty ``.jsonl`` that
        survives daemon restart as a zero-message ghost conversation
        occupying a ``MAX_CONVERSATIONS`` slot.
        """
        from enigma_engine.core.safe_save import atomic_write_text

        with self._lock:
            if conversation_id not in self._histories:
                return
            messages = list(self._histories[conversation_id])

        try:
            CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
            text = "\n".join(json.dumps(m, ensure_ascii=False) for m in messages)
            atomic_write_text(CONVERSATIONS_DIR / f"{conversation_id}.jsonl", text)
        except Exception as exc:
            logger.warning("Could not persist conversation %s: %s",
                           conversation_id, exc)

    def _delete_disk_conversation(self, conversation_id: str) -> None:
        """Remove the on-disk file(s) for a conversation if they exist.

        MC-2c: ``atomic_write_text`` writes a ``<path>.bak`` revision-back
        backup on every successful save (see ``core/safe_save.py``). When
        a conversation is deleted, its ``.bak`` becomes a permanent
        orphan — ``glob("*.jsonl")`` doesn't see it on boot so it never
        resurrects the deleted conv, but it accumulates forever as a
        silent storage leak. Unlink both files in one shot.
        """
        path = CONVERSATIONS_DIR / f"{conversation_id}.jsonl"
        bak_path = CONVERSATIONS_DIR / f"{conversation_id}.jsonl.bak"
        for p in (path, bak_path):
            try:
                p.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Could not delete on-disk conversation file %s: %s",
                               p.name, exc)

    def _persist_active_conv_id(self) -> None:
        """Atomically write the active-conversation pointer to disk.

        MC-2b: the active id survives daemon restart so a user who
        ``Ctrl+C``'d the server mid-session resumes on the same
        conversation. Reads ``_active_conv_id`` under the lock; writes
        outside the lock to keep I/O off the hot path. Best-effort —
        write failures log a WARNING but never raise (a stale active
        pointer is recoverable; a crashed boot is not).
        """
        from enigma_engine.core.safe_save import atomic_write_text

        with self._lock:
            active = self._active_conv_id
        try:
            CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                ACTIVE_CONV_FILE,
                json.dumps({"active_conv_id": active}, ensure_ascii=False),
            )
        except Exception as exc:
            logger.warning("MC-2b: could not persist active conv id: %s", exc)

    def _load_active_conv_id_from_disk(self) -> None:
        """Restore the active-conversation pointer from disk on boot.

        MC-2b: only restores if the saved id is still present in
        ``_histories`` after ``load_conversations_from_disk`` has run
        — otherwise the user evicted/deleted it between sessions and
        the pointer is stale. Caller must invoke AFTER the histories
        have been loaded.
        """
        if not ACTIVE_CONV_FILE.exists():
            return
        try:
            payload = json.loads(ACTIVE_CONV_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "MC-2b: could not parse active conv pointer (%s); ignoring",
                exc)
            return
        saved = payload.get("active_conv_id") if isinstance(payload, dict) else None
        if not isinstance(saved, str):
            return
        with self._lock:
            if saved in self._histories:
                self._active_conv_id = saved
                logger.info("MC-2b: restored active conversation %s", saved)
                stale = False
            else:
                logger.info(
                    "MC-2b: saved active conv %s no longer in histories; "
                    "starting with no active conversation", saved)
                stale = True
        # Self-heal: rewrite disk to ``null`` so the same stale id is not
        # rediscovered (and re-logged) on every subsequent boot.
        if stale:
            self._persist_active_conv_id()

    def load_conversations_from_disk(self) -> int:
        """Load persisted conversations from ``CONVERSATIONS_DIR`` on startup.

        Skips corrupt files with a WARNING.  Ignores conversations whose
        IDs are not valid UUID4 hexstrings (defence against stray files).
        Returns the count of conversations successfully loaded.

        If more than ``MAX_CONVERSATIONS`` files are found, only the
        ``MAX_CONVERSATIONS`` most-recently-modified ones are loaded
        and the older excess files are deleted from disk (MC-2a) so a
        previously-higher cap cannot leave the conversations directory
        growing unbounded.
        """
        if not CONVERSATIONS_DIR.exists():
            return 0

        all_files = sorted(
            CONVERSATIONS_DIR.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        # MC-2a follow-up: partition valid-shape (UUID4 hex stem) from
        # stray files BEFORE the cap slice. Otherwise a stray ``*.jsonl``
        # with a newer mtime than a real conversation would occupy a
        # kept-slot and push a valid conv into the excess slice, causing
        # silent data loss. Stray files are skipped (not counted toward
        # cap, not deleted) — they don't belong to us and we shouldn't
        # destroy operator backups / hand-edits.
        files: list[Path] = []
        for path in all_files:
            cid = path.stem
            if len(cid) == 32 and all(c in "0123456789abcdef" for c in cid):
                files.append(path)
            else:
                logger.warning(
                    "Skipping unexpected file in conversations dir: %s",
                    path.name)
        # Cap to the allowed maximum so we don't bloat memory on first
        # boot after lowering the limit.  MC-2a: also unlink the excess
        # files on disk — keeping them around is a silent storage leak
        # because every future boot will see them, sort them, and drop
        # them again forever.
        excess = files[MAX_CONVERSATIONS:]
        files = files[:MAX_CONVERSATIONS]
        for path in excess:
            bak_path = path.with_suffix(path.suffix + ".bak")
            try:
                path.unlink(missing_ok=True)
                # MC-2c: also unlink the .bak sibling so it doesn't
                # become an orphan after the parent is evicted.
                bak_path.unlink(missing_ok=True)
                logger.warning(
                    "MC-2a: deleted excess conversation file %s "
                    "(disk count exceeded MAX_CONVERSATIONS=%d)",
                    path.name, MAX_CONVERSATIONS,
                )
            except Exception as exc:
                logger.warning(
                    "MC-2a: could not delete excess conversation file %s: %s",
                    path.name, exc,
                )

        # MC-2c: sweep orphan ``.bak`` files whose parent ``.jsonl`` no
        # longer exists. Backups are created by ``atomic_write_text`` on
        # every save and become permanent garbage once the live file is
        # deleted (by user delete, LRU eviction, or an older code path
        # that pre-dated this sweep). Cleanup is best-effort and silent
        # on misses — orphans don't affect correctness, only disk usage.
        kept_stems = {p.stem for p in files}
        for bak in CONVERSATIONS_DIR.glob("*.jsonl.bak"):
            # bak.name = "<cid>.jsonl.bak" -> parent stem is bak.name[:-len(".jsonl.bak")]
            parent_stem = bak.name[:-len(".jsonl.bak")]
            if parent_stem in kept_stems:
                continue
            live_parent = CONVERSATIONS_DIR / f"{parent_stem}.jsonl"
            if live_parent.exists():
                continue
            try:
                bak.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning(
                    "MC-2c: could not delete orphan backup %s: %s",
                    bak.name, exc,
                )

        loaded = 0
        for path in reversed(files):  # oldest first -> LRU order
            cid = path.stem
            try:
                messages: list[dict[str, str]] = []
                for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    entry = json.loads(raw)
                    if not isinstance(entry, dict):
                        raise ValueError(f"line {lineno}: expected object, got {type(entry).__name__}")
                    messages.append(entry)
            except Exception as exc:
                logger.warning(
                    "Skipping corrupt conversation file %s: %s", path.name, exc)
                continue

            with self._lock:
                self._histories[cid] = messages
                if cid not in self._conv_order:
                    self._conv_order.append(cid)
            loaded += 1
            logger.debug("Loaded conversation %s (%d messages)", cid, len(messages))

        if loaded:
            logger.info("MC-2: restored %d conversation(s) from disk", loaded)
        return loaded


state = AppState()

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Enigma AI Engine",
    version=__version__,
    docs_url="/api/docs",
    redoc_url=None,
)

# ---------------------------------------------------------------------------
# CORS — opt-in only via --cors-origins  (Suggestion #6)
# No middleware = no attack surface.  Call enable_cors() at runtime to add it.
# ---------------------------------------------------------------------------

def enable_cors(origins: list[str]) -> None:
    """Add CORS middleware with the given origin list.

    Only call this *before* uvicorn starts (inside run_server).
    """
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS enabled for origins: %s", origins)




# ---------------------------------------------------------------------------
# Input Limits  (Suggestion #7)
# ---------------------------------------------------------------------------
MAX_MESSAGE_LENGTH = 32_768   # 32K chars (~8K tokens)
MAX_BATCH_PROMPTS = 50
MAX_PATH_LENGTH = 256

# ---------------------------------------------------------------------------
# Concurrency Lock  (Suggestion #7)
# ---------------------------------------------------------------------------
_inference_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Auto-research helper (Pass 156z9g — AutoResearch-2 API parity)
# ---------------------------------------------------------------------------

def _maybe_research_context(message: str, web_access: bool) -> str:
    """Return ``[WEB RESEARCH]`` context for ``message`` if appropriate.

    Mirrors the GUI wiring in ``gui_logic_chat.py`` so /api/chat and
    /api/chat/stream callers get the same Stage-A pre-gen behaviour as
    the desktop UI. Off by default (``web_access=False``) for privacy;
    callers must opt in per request.

    Returns the empty string on every off-path (web_access=False, query
    too short / trivial, web_utils unavailable, fetch failure, no
    results) so the caller can unconditionally concatenate with
    ``system_prompt`` without a None check.
    """
    if not web_access:
        return ""
    try:
        from enigma_engine.core.auto_research import (
            auto_research, should_auto_research,
        )
    except ImportError:
        logger.debug("auto_research module unavailable; skipping")
        return ""
    try:
        if not should_auto_research(message):
            return ""
        return auto_research(message, max_results=3) or ""
    except Exception as exc:
        logger.debug("Auto-research pre-gen failed: %s", exc)
        return ""

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    # MC-1: per-client conversation scoping.  ``None`` auto-creates a
    # fresh conversation; the response carries the assigned ID so the
    # client can pin subsequent turns to the same thread.  Unknown IDs
    # return 404.
    conversation_id: str | None = Field(default=None, max_length=128)
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    # N-15b/c: optional JSON schema. When set, the engine masks logits
    # per token so the response is structurally valid JSON. Supported on
    # both /api/chat (Pass 156z4) and /api/chat/stream (Pass 156z6, N-15c)
    # — the constraint FSM advances per yielded token in streaming mode.
    # Native PyTorch only — GGUF backend rejects loud (llama.cpp uses
    # its own sampler).
    json_schema: dict[str, Any] | None = None
    # Pass 156z9g AutoResearch-2 API parity: opt-in web research per
    # request. Default False (privacy-safe). When True, the handler
    # runs ``should_auto_research`` on the message; if it triggers,
    # ``auto_research`` fetches DuckDuckGo results + fetches page
    # text and prepends a ``[WEB RESEARCH]`` block to the system
    # prompt before calling the engine. /api/chat additionally runs
    # the post-gen ``should_retry_with_research`` gate (Stage A); the
    # streaming endpoint runs pre-gen only because post-gen retry
    # would require a second SSE stream the existing
    # ``_inference_lock`` design doesn't support.
    web_access: bool = False

class ChatResponse(BaseModel):
    message: str
    tokens_used: int = 0

class ModelLoadRequest(BaseModel):
    path: str = Field(..., max_length=MAX_PATH_LENGTH)

class EngineFlagsUpdate(BaseModel):
    inline_search_enabled: bool | None = None
    inline_search_splice_enabled: bool | None = None


class ConfigUpdate(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None

    def validated(self) -> dict[str, Any]:
        """Return only non-None fields, clamped to valid ranges."""
        # max_tokens upper bound scaled to VRAM (S807)
        try:
            from enigma_engine.core.hardware_detection import InferenceMemoryBudget
            max_tok_cap = InferenceMemoryBudget().api_max_tokens
        except Exception:
            max_tok_cap = 4096
        limits = {
            "temperature": (0.0, 2.0),
            "top_p": (0.0, 1.0),
            "top_k": (1, 200),
            "max_tokens": (16, max_tok_cap),
            "repetition_penalty": (1.0, 2.0),
        }
        result: dict[str, Any] = {}
        for key, (lo, hi) in limits.items():
            val = getattr(self, key)
            if val is not None:
                clamped = max(lo, min(hi, val))
                result[key] = type(lo)(clamped)  # preserve int/float type
        return result


# ---------------------------------------------------------------------------
# Health & System
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "version": __version__,
        "uptime": round(time.time() - state.start_time, 1),
        "model_loaded": state.engine is not None,
    }


@app.get("/api/system")
async def system_info():
    """Return hardware and system information."""
    import platform

    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "device": "cpu",
        "torch_available": False,
        "cuda_available": False,
        "gpu_name": None,
        "vram_gb": None,
    }

    try:
        import torch
        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info["vram_gb"] = round(vram, 1)
    except ImportError:
        pass

    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        info["cpu_count"] = psutil.cpu_count(logical=True)
    except ImportError:
        pass

    return info


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/api/models")
async def list_models():
    """List available model files."""
    models = []
    if MODELS_DIR.exists():
        for ext in ("*.pth", "*.pt", "*.gguf", "*.bin", "*.safetensors"):
            for p in MODELS_DIR.glob(ext):
                size_mb = p.stat().st_size / (1024 * 1024)
                models.append({
                    "name": p.stem,
                    "filename": p.name,
                    "path": str(p.relative_to(MODELS_DIR)),
                    "size_mb": round(size_mb, 1),
                    "format": p.suffix.lstrip("."),
                })
        # Also check subdirectories one level deep
        for subdir in MODELS_DIR.iterdir():
            if subdir.is_dir():
                for ext in ("*.pth", "*.pt", "*.gguf", "*.bin", "*.safetensors"):
                    for p in subdir.glob(ext):
                        size_mb = p.stat().st_size / (1024 * 1024)
                        models.append({
                            "name": f"{subdir.name}/{p.stem}",
                            "filename": p.name,
                            "path": str(p.relative_to(MODELS_DIR)),
                            "size_mb": round(size_mb, 1),
                            "format": p.suffix.lstrip("."),
                        })

    return {"models": models}


@app.get("/api/models/status")
async def model_status():
    """Get the status of the currently loaded model."""
    if state.engine is None:
        return {"loaded": False, "model": None}
    return {"loaded": True, "model": state.model_info_snapshot()}


@app.post("/api/models/load")
async def load_model(req: ModelLoadRequest):
    """Load a model by path."""
    path = Path(req.path)
    if not path.is_absolute():
        path = (MODELS_DIR / path).resolve()
    else:
        path = path.resolve()
    # Prevent path traversal — model must be inside MODELS_DIR
    try:
        path.relative_to(MODELS_DIR.resolve())
    except ValueError:
        raise HTTPException(403, "Path must be inside the models directory") from None
    if not path.exists():
        raise HTTPException(404, f"Model not found: {req.path}")
    try:
        info = state.load_model(str(path))
        return {"status": "ok", "model": info}
    except Exception as exc:
        # Ensure partial load doesn't leave stale state
        try:
            state.unload_model()
        except Exception:
            pass
        logger.exception("Failed to load model")
        raise HTTPException(500, f"Failed to load model: {exc}") from exc


@app.post("/api/models/unload")
async def unload_model():
    """Unload the current model and free memory."""
    state.unload_model()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Send a chat message and get a response."""
    if state.engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "No model loaded. Load a model first via /api/models/load."},
        )
    # Pass 156z9ac: validate json_schema shape at the boundary so a
    # malformed schema returns HTTP 400 with the validator message,
    # not HTTP 500 wrapping a deep ValueError from inside generation.
    # Lock is acquired AFTER validation — a bad-schema request must
    # not block other clients waiting on the inference lock.
    if req.json_schema is not None:
        try:
            validate_json_schema_shape(req.json_schema)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid json_schema: {exc}"},
            )
    if not _inference_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=429,
            content={"error": "Engine busy — another request is in progress."},
        )
    try:
        kw: dict[str, Any] = {}
        if req.temperature is not None:
            kw["temperature"] = req.temperature
        if req.max_tokens is not None:
            kw["max_tokens"] = req.max_tokens
        if req.top_p is not None:
            kw["top_p"] = req.top_p
        if req.top_k is not None:
            kw["top_k"] = req.top_k
        if req.repetition_penalty is not None:
            kw["repetition_penalty"] = req.repetition_penalty
        if req.json_schema is not None:
            kw["json_schema"] = req.json_schema

        # Pass 156z9g: AutoResearch-2 pre-gen wiring. Empty string when
        # off / not triggered / fetch failed — caller-side concat is
        # always safe.
        web_ctx = _maybe_research_context(req.message, req.web_access)
        if web_ctx:
            kw["system_prompt"] = web_ctx

        try:
            response, conv_id = state.chat(
                req.message,
                conversation_id=req.conversation_id,
                **kw,
            )
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={"error": f"Unknown conversation_id: {req.conversation_id}"},
            )

        # Stage A post-gen retry. Mirrors gui_logic_chat.py: if web
        # access is on, no pre-gen research ran, and the visible reply
        # scores >= threshold uncertain, retry once with research
        # context. Skipped on the streaming path (would need a second
        # SSE stream the inference lock model doesn't support).
        if (req.web_access and not web_ctx
                and isinstance(response, str)):
            try:
                from enigma_engine.core.auto_research import (
                    auto_research as _ar_fetch,
                    should_retry_with_research,
                )
                if should_retry_with_research(req.message, response):
                    retry_ctx = _ar_fetch(req.message, max_results=3)
                    if retry_ctx:
                        retry_kw = dict(kw)
                        retry_kw["system_prompt"] = retry_ctx
                        logger.info(
                            "AutoResearch-2: low-confidence reply on "
                            "/api/chat, retrying with research context")
                        # MC-1a B2: drop the failed user+assistant pair
                        # before retry so the retry call sees clean
                        # history instead of the low-confidence reply
                        # it is about to replace.
                        try:
                            state.rollback_last_turn(conv_id)
                        except KeyError:
                            pass
                        response, conv_id = state.chat(
                            req.message,
                            conversation_id=conv_id,
                            **retry_kw,
                        )
            except Exception as exc:
                logger.debug(
                    "Auto-research post-gen retry failed: %s", exc)
        # Cap response length to prevent memory exhaustion
        truncated = False
        if isinstance(response, str) and len(response) > 500_000:
            original_len = len(response)
            response = response[:500_000]
            truncated = True
            logger.warning("Response truncated from %d to 500000 chars", original_len)
        result: dict[str, Any] = {"message": response, "conversation_id": conv_id}
        if truncated:
            result["truncated"] = True
        return result
    except Exception as exc:
        logger.exception("Chat error")
        return JSONResponse(
            status_code=500,
            content={"error": f"Generation failed: {exc}"},
        )
    finally:
        _inference_lock.release()


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream chat via Server-Sent Events.

    Returns an SSE stream where each event contains a token.
    Events:
        event: start   — generation begins
        event: token   — a single token (data.content)
        event: end     — generation complete (data.content = full text)
        event: error   — an error occurred
    """
    if state.engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "No model loaded. Load a model first via /api/models/load."},
        )
    # Pass 156z9ac: same boundary-validation as /api/chat (sibling
    # boundary).  Validate BEFORE acquiring the inference lock so a
    # bad-schema request returns 400 immediately without queueing
    # behind a real generation.
    if req.json_schema is not None:
        try:
            validate_json_schema_shape(req.json_schema)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid json_schema: {exc}"},
            )

    # MC-1: fast 404 for explicit-but-unknown IDs before acquiring the
    # inference lock (no orphan risk — we don't create anything here).
    # Auto-creation (conversation_id=None) is deferred until AFTER we
    # hold the lock so a 429 response can't leave an empty orphan
    # conversation in _histories.  B1 fix (Pass 156z9dc).
    if req.conversation_id is not None:
        with state._lock:
            if req.conversation_id not in state._histories:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Unknown conversation_id: {req.conversation_id}"},
                )

    if not _inference_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=429,
            content={"error": "Engine busy — another request is in progress."},
        )

    # Auto-create or re-validate + activate atomically. MC-1a B3:
    # single locked critical section so a concurrent DELETE cannot
    # race between resolve and activate.
    try:
        conv_id, switched = state._resolve_and_activate(req.conversation_id)
    except KeyError:
        _inference_lock.release()
        return JSONResponse(
            status_code=404,
            content={"error": f"Unknown conversation_id: {req.conversation_id}"},
        )

    # MC-1: invalidate KV cache on switch.
    if switched:
        state._invalidate_engine_state()

    # Build kwargs from config overrides + per-request overrides
    kwargs: dict[str, Any] = {}
    if state.config_overrides:
        kwargs.update(state.config_overrides)
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature
    if req.max_tokens is not None:
        kwargs["max_tokens"] = req.max_tokens
    if req.top_p is not None:
        kwargs["top_p"] = req.top_p
    if req.top_k is not None:
        kwargs["top_k"] = req.top_k
    if req.repetition_penalty is not None:
        kwargs["repetition_penalty"] = req.repetition_penalty
    # N-15c: json_schema reaches stream_generate via stream_chat's **kwargs
    # forwarding. Omit-when-None to stay compatible with legacy engines
    # that lack the parameter on stream_generate (same discipline as the
    # non-streaming /api/chat handler).
    if req.json_schema is not None:
        kwargs["json_schema"] = req.json_schema

    # Pass 156z9g: AutoResearch-2 pre-gen wiring (streaming). Post-gen
    # retry is intentionally NOT wired here — it would require ending
    # the first stream and starting a second one inside the same
    # request, which the SSE generator + ``_inference_lock`` model
    # doesn't support. /api/chat (non-streaming) has the full Stage A
    # gate; streaming callers only get pre-gen context.
    web_ctx = _maybe_research_context(req.message, req.web_access)
    if web_ctx:
        kwargs["system_prompt"] = web_ctx

    def _sse_generator():
        """Yield SSE-formatted events from engine.stream_chat."""
        from enigma_engine.core.streaming import StreamChunk, StreamEvent

        try:
            # Start event — MC-1: include conversation_id so clients
            # auto-creating a thread learn the assigned ID before any
            # tokens arrive.
            start_chunk = StreamChunk(
                content="", event=StreamEvent.START,
                metadata={"message": req.message,
                          "conversation_id": conv_id})
            yield start_chunk.to_sse()

            full_response = []
            try:
                gen = state.engine.stream_chat(
                    req.message,
                    history=state.history_snapshot(conv_id),
                    **kwargs)
                for token in gen:
                    full_response.append(token)
                    token_chunk = StreamChunk(
                        content=token, event=StreamEvent.TOKEN)
                    yield token_chunk.to_sse()

                # End event with full response
                combined = "".join(full_response)
                end_chunk = StreamChunk(
                    content=combined, event=StreamEvent.END,
                    metadata={"conversation_id": conv_id})
                yield end_chunk.to_sse()

                # Track history under lock (per-conv). MC-1a B3: skip
                # if the conv was deleted mid-stream rather than
                # resurrecting it.
                with state._lock:
                    stored = state._append_turn_if_alive_locked(
                        conv_id, req.message, combined)
                # MC-2: persist after stream completes.
                if stored:
                    state._persist_conversation(conv_id)

            except Exception as exc:
                logger.exception("Stream chat error")
                err_msg = str(exc)
                err_chunk = StreamChunk(
                    content=err_msg, event=StreamEvent.ERROR)
                yield err_chunk.to_sse()
        finally:
            _inference_lock.release()

    try:
        return FastAPIStreamingResponse(
            _sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception:
        _inference_lock.release()
        raise


# ---------------------------------------------------------------------------
# Batch Inference
# ---------------------------------------------------------------------------

class BatchRequest(BaseModel):
    prompts: list[str] = Field(..., max_length=MAX_BATCH_PROMPTS)
    max_tokens: int = 100
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None

    @field_validator("prompts")
    @classmethod
    def validate_prompt_lengths(cls, v: list[str]) -> list[str]:
        for i, p in enumerate(v):
            if len(p) > MAX_MESSAGE_LENGTH:
                raise ValueError(
                    f"prompts[{i}] exceeds {MAX_MESSAGE_LENGTH} characters")
        return v


@app.post("/api/batch")
async def batch_inference(req: BatchRequest):
    """Process multiple prompts in a single request.

    Uses batch_generate if available on the engine, otherwise
    falls back to sequential generation per prompt.
    """
    if state.engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "No model loaded. Load a model first via /api/models/load."},
        )
    if not req.prompts:
        return JSONResponse(
            status_code=400,
            content={"error": "Prompts list cannot be empty."},
        )
    if not _inference_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=429,
            content={"error": "Engine busy \u2014 another request is in progress."},
        )

    kwargs: dict[str, Any] = {}
    if state.config_overrides:
        kwargs.update(state.config_overrides)
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature
    if req.top_k is not None:
        kwargs["top_k"] = req.top_k
    if req.top_p is not None:
        kwargs["top_p"] = req.top_p

    try:
        # Prefer batch_generate for efficiency
        if hasattr(state.engine, "batch_generate"):
            responses = state.engine.batch_generate(
                req.prompts, max_gen=req.max_tokens, **kwargs)
        else:
            # Sequential fallback
            responses = []
            for prompt in req.prompts:
                try:
                    resp = state.engine.chat(prompt, **kwargs)
                    responses.append(resp)
                except TypeError:
                    logger.debug("Engine.chat() does not accept kwargs, retrying without")
                    resp = state.engine.chat(prompt)
                    responses.append(resp)

        return {
            "responses": responses,
            "count": len(responses),
        }
    except Exception as exc:
        logger.exception("Batch inference error")
        return JSONResponse(
            status_code=500,
            content={"error": f"Batch generation failed: {exc}"},
        )
    finally:
        _inference_lock.release()


# ---------------------------------------------------------------------------
# Chat History
# ---------------------------------------------------------------------------

@app.get("/api/history")
async def get_history():
    """Get chat history for the currently-active conversation (legacy route).

    MC-1: with per-conversation scoping this returns the most recent
    conversation's transcript, or an empty list if none has been
    started yet.  New clients should prefer
    ``GET /api/conversations/{id}/history``.
    """
    return {"history": state.history_snapshot()}


@app.delete("/api/history")
async def clear_history():
    """Clear all conversations and invalidate engine state (legacy nuke route).

    MC-1: matches the prior behaviour of wiping everything in one
    call.  Clients that want per-conversation deletion should use
    ``DELETE /api/conversations/{id}`` instead.
    """
    state.clear_all_conversations()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Conversations (MC-1)
# ---------------------------------------------------------------------------

@app.post("/api/conversations")
async def create_conversation():
    """Allocate a fresh conversation and return its ID.

    Triggers LRU eviction if ``MAX_CONVERSATIONS`` would be exceeded.
    """
    cid = state.create_conversation()
    return {"id": cid}


@app.get("/api/conversations")
async def list_conversations():
    """List known conversations (LRU order, oldest first)."""
    return {"conversations": state.list_conversations()}


@app.get("/api/conversations/{conv_id}/history")
async def get_conversation_history(conv_id: str):
    """Return the message history for a specific conversation."""
    try:
        history = state.history_snapshot(conv_id)
    except KeyError:
        raise HTTPException(404, f"Unknown conversation_id: {conv_id}") from None
    return {"history": history}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Drop a conversation.  Clears engine KV cache if it was active."""
    try:
        state.delete_conversation(conv_id)
    except KeyError:
        raise HTTPException(404, f"Unknown conversation_id: {conv_id}") from None
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

@app.get("/api/profiles")
async def list_profiles():
    """List available AI profiles."""
    profiles = []
    if PROFILES_DIR.exists():
        for p in sorted(PROFILES_DIR.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                profiles.append({
                    "id": p.stem,
                    "name": data.get("name", p.stem),
                    "description": data.get("description", ""),
                })
            except (json.JSONDecodeError, OSError):
                continue
    return {"profiles": profiles, "active": state.active_profile}


def _safe_profile_path(profile_id: str) -> Path:
    """Resolve profile path and reject traversal attempts."""
    # Strip path separators to prevent directory traversal
    safe_id = Path(profile_id).name
    path = (PROFILES_DIR / f"{safe_id}.json").resolve()
    try:
        path.relative_to(PROFILES_DIR.resolve())
    except ValueError:
        raise HTTPException(403, "Invalid profile ID") from None
    return path


@app.get("/api/profiles/{profile_id}")
async def get_profile(profile_id: str):
    """Get full profile details."""
    path = _safe_profile_path(profile_id)
    if not path.exists():
        raise HTTPException(404, f"Profile not found: {profile_id}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        raise HTTPException(500, f"Corrupt profile: {profile_id}") from None
    return data


@app.post("/api/profiles/{profile_id}/activate")
async def activate_profile(profile_id: str):
    """Activate an AI profile (applies its generation settings)."""
    path = _safe_profile_path(profile_id)
    if not path.exists():
        raise HTTPException(404, f"Profile not found: {profile_id}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        raise HTTPException(500, f"Corrupt profile: {profile_id}") from None

    # Validate generation settings BEFORE acquiring lock (no I/O under lock).
    gen = data.get("generation", {})
    validated: dict = {}
    if gen:
        validated = ConfigUpdate(**{
            k: v for k, v in gen.items()
            if k in ConfigUpdate.__annotations__
        }).validated()

    # Atomically update shared state under the AppState lock so concurrent
    # chat calls always observe a consistent (active_profile, config_overrides)
    # pair.  Capture the engine reference inside the lock; apply the profile
    # to it outside the lock (heavy engine op must not block chat callers).
    with state._lock:
        state.active_profile = profile_id
        if validated:
            state.config_overrides.update(validated)
        engine = state.engine

    # Apply remaining profile fields (system_prompt, adapter,
    # generation knobs) to the live engine when one is loaded. Pass
    # 156z2 audit fix: closes the dead-infra-to-dead-infra gap caught
    # in self-audit on Pass 156z. Pre-fix, the API endpoint dropped
    # ``system_prompt`` and ``adapter`` on the floor — only
    # ``generation`` survived the trip from disk to engine.
    # ``apply_profile_to_engine`` is the canonical applier and uses
    # ``hasattr`` guards throughout + catches adapter failures
    # internally, so calling it on a partially-initialised engine is
    # safe. Engine-not-loaded path is a no-op (profile id still saved
    # on state), preserving the existing UX where users set the active
    # profile before loading a model.
    if engine is not None:
        from enigma_engine.core.ai_profile import (
            AIProfile, apply_profile_to_engine,
        )
        apply_profile_to_engine(AIProfile.from_dict(data), engine)

    return {"status": "ok", "active": profile_id, "settings": gen}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@app.get("/api/config")
async def get_config():
    """Get current generation config."""
    from enigma_engine.config import CONFIG

    # Merge engine defaults with overrides
    config = {
        "temperature": CONFIG.get("temperature", 0.8),
        "top_p": CONFIG.get("top_p", 0.9),
        "top_k": CONFIG.get("top_k", 50),
        "max_tokens": CONFIG.get("max_gen", 100),
        "repetition_penalty": CONFIG.get("repetition_penalty", 1.1),
    }
    config.update(state.config_overrides)
    return config


@app.post("/api/config")
async def update_config(req: ConfigUpdate):
    """Update generation config (values clamped to valid ranges)."""
    updates = req.validated()
    with state._lock:
        state.config_overrides.update(updates)
    return {"status": "ok", "config": {**await get_config()}}


@app.post("/api/config/engine-flags")
async def update_engine_flags(req: EngineFlagsUpdate):
    """Push engine-level boolean flags to the live engine.

    Returns ``{"status": "no-engine"}`` when no model is loaded (flags are
    silently dropped — caller must re-push after model load).
    """
    with state._lock:
        engine = state.engine
    if engine is None:
        return {"status": "no-engine", "applied": {}}
    applied: dict[str, bool] = {}
    for flag, val in [
        ("inline_search_enabled", req.inline_search_enabled),
        ("inline_search_splice_enabled", req.inline_search_splice_enabled),
    ]:
        if val is not None:
            try:
                setattr(engine, flag, bool(val))
                applied[flag] = bool(val)
            except Exception:
                pass
    return {"status": "ok", "applied": applied}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# Module-level training state for the API
_training_lock = threading.Lock()
_active_training_trainer: Any = None
_training_state: dict[str, Any] = {
    "active": False,
    "progress": 0,
    "message": "",
    "epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "best_loss": None,
    "step": 0,
    "total_steps": 0,
    "lr": 0.0,
    "tok_s": 0,
    "output_path": "",
    "abort_reason": "",
}


class TrainRequest(BaseModel):
    # Legacy SFT-only fields (kept for backward compatibility).
    data_file: str | None = Field(default=None, max_length=MAX_PATH_LENGTH)
    epochs: int = Field(default=5, ge=1, le=1000)
    learning_rate: float = Field(default=0.00005, gt=0.0, le=1.0)
    batch_size: int = Field(default=4, ge=1, le=256)

    # Dispatcher fields (forwarded verbatim to TrainingJobConfig).
    mode: str | None = None
    data: Any = None
    training: dict[str, Any] | None = None
    dpo: dict[str, Any] | None = None
    grpo: dict[str, Any] | None = None
    lora: dict[str, Any] | None = None
    simpo: dict[str, Any] | None = None
    kto: dict[str, Any] | None = None
    orpo: dict[str, Any] | None = None
    rest: dict[str, Any] | None = None
    reward_model: dict[str, Any] | None = None
    vision: dict[str, Any] | None = None
    audio: dict[str, Any] | None = None
    self_play: dict[str, Any] | None = None
    rlhf: dict[str, Any] | None = None
    remax: dict[str, Any] | None = None
    adaptive: dict[str, Any] | None = None
    allow_experimental: bool = False
    resume_from: str | None = None


@app.get("/api/training/status")
async def training_status():
    """Get the current training status."""
    with _training_lock:
        return dict(_training_state)


@app.delete("/api/training/cancel")
async def cancel_training():
    """Request cancellation for the active training run."""
    trainer = None
    with _training_lock:
        if not _training_state["active"]:
            return {"status": "idle", "message": "No training in progress."}
        trainer = _active_training_trainer
        _training_state["message"] = "Cancel requested"
        _training_state["abort_reason"] = "cancel_requested"

    if trainer is not None and hasattr(trainer, "request_stop"):
        trainer.request_stop()

    return {"status": "cancelling"}


@app.post("/api/train")
async def start_training(req: TrainRequest):
    """Start a training run in the background.

    Requires a model to be loaded. Two request shapes are supported:

    1. Legacy SFT: pass `data_file` (resolved under data/) plus optional
       epochs/learning_rate/batch_size. Routes to dispatcher mode 'sft'.
    2. Config-body: pass `mode` plus dispatcher fields (data, training, dpo,
       grpo, lora, ...). Forwarded verbatim to TrainingJobConfig.
    """
    if state.engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "No model loaded. Load a model first via /api/models/load."},
        )
    with _training_lock:
        if _training_state["active"]:
            return JSONResponse(
                status_code=409,
                content={"error": "Training already in progress."},
            )

    has_mode = req.mode is not None
    has_data_file = req.data_file is not None
    if has_mode == has_data_file:
        raise HTTPException(
            422,
            "Provide exactly one of 'mode' or 'data_file'",
        )

    # Build dispatcher payload.
    is_legacy = not has_mode

    if is_legacy:
        # Legacy SFT path: data_file required, resolved under data/.
        if not req.data_file:
            raise HTTPException(422, "data_file is required when mode is not set")
        data_dir = PROJECT_ROOT / "data"
        data_path = (data_dir / req.data_file).resolve()
        try:
            data_path.relative_to(data_dir.resolve())
        except ValueError:
            raise HTTPException(403, "Data file must be inside the data directory") from None
        if not data_path.exists():
            raise HTTPException(404, f"Data file not found: {req.data_file}")
        dispatch_dict: dict[str, Any] = {
            "mode": "sft",
            "data": data_path.read_text(encoding="utf-8"),
            "training": {
                "epochs": req.epochs,
                "learning_rate": req.learning_rate,
                "batch_size": req.batch_size,
            },
        }
    else:
        # Config-body path: forward dispatcher fields directly.
        dispatch_dict = {"mode": req.mode}
        if req.data is not None:
            dispatch_dict["data"] = req.data
        for field in (
            "training", "dpo", "grpo", "lora", "simpo", "kto", "orpo",
            "rest", "reward_model", "vision", "audio", "self_play",
            "rlhf", "remax", "adaptive", "resume_from",
        ):
            value = getattr(req, field, None)
            if value is not None:
                dispatch_dict[field] = value
        if req.allow_experimental:
            dispatch_dict["allow_experimental"] = True

    from enigma_engine.training.schema import TrainingJobConfig, materialize_dispatch_payload

    validated_payload = dict(dispatch_dict)
    validated_payload["data"] = materialize_dispatch_payload(
        validated_payload.get("data"),
        validated_payload.get("mode"),
    )
    try:
        job = TrainingJobConfig.model_validate(validated_payload)
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc

    import threading

    def _run_training():
        """Background training thread."""
        global _active_training_trainer
        try:
            with _training_lock:
                _training_state.update({
                    "active": True,
                    "progress": 0,
                    "message": "Initializing...",
                    "epoch": 0,
                    "total_epochs": job.training.epochs,
                    "loss": 0.0,
                    "best_loss": None,
                    "step": 0,
                    "total_steps": 0,
                    "lr": 0.0,
                    "tok_s": 0,
                    "output_path": "",
                    "abort_reason": "",
                })

            from enigma_engine.training import (
                build_dispatch_context,
                run_training,
            )

            _trainer_ref: list = []
            _throughput_tokens = [0]
            _throughput_time = [0.0]

            def on_progress(pct: int, msg: str):
                with _training_lock:
                    _training_state["progress"] = pct
                    _training_state["message"] = msg

            def on_epoch_complete(epoch: int, loss: float):
                with _training_lock:
                    _training_state["epoch"] = epoch + 1
                    _training_state["loss"] = loss

            def on_loss(loss: float):
                t = _trainer_ref[0] if _trainer_ref else None
                step = getattr(getattr(t, "state", None), "step", 0) if t else 0
                total = getattr(t, "_total_training_steps", 0) if t else 0
                lr_val = 0.0
                if t and hasattr(t, "optimizer") and t.optimizer.param_groups:
                    lr_val = float(t.optimizer.param_groups[0]["lr"])
                tok_s = 0
                if _throughput_time[0] > 0:
                    tok_s = int(_throughput_tokens[0]
                                / max(0.001, _throughput_time[0]))
                _throughput_tokens[0] = 0
                _throughput_time[0] = 0.0
                with _training_lock:
                    _training_state["loss"] = loss
                    _training_state["step"] = step
                    _training_state["total_steps"] = total
                    _training_state["lr"] = lr_val
                    _training_state["tok_s"] = tok_s

            def on_throughput(tokens: int, step_time: float):
                _throughput_tokens[0] += tokens
                _throughput_time[0] += step_time

            def on_trainer_ready(t) -> None:
                global _active_training_trainer
                _trainer_ref.append(t)
                with _training_lock:
                    _active_training_trainer = t
                    _training_state["total_steps"] = getattr(
                        t, "_total_training_steps", 0)

            ctx = build_dispatch_context(
                engine=state.engine,
                on_progress=on_progress,
                on_epoch_complete=on_epoch_complete,
                on_loss=on_loss,
                on_throughput=on_throughput,
                on_trainer_ready=on_trainer_ready,
            )

            result = run_training(job, ctx)

            # Persist trained weights back to the loaded model file.
            # Only .pth checkpoints can be re-saved this way; GGUF files
            # are read-only from Python so we skip them.
            # Gate: do not save if training was cancelled or failed.
            saved_path = ""
            model_path = state.model_path or ""
            abort_reason = ""
            with _training_lock:
                abort_reason = _training_state.get("abort_reason", "")
            if (not abort_reason  # Only save on success (no abort_reason)
                    and model_path.endswith(".pth")
                    and state.engine is not None
                    and hasattr(state.engine, "model")
                    and state.engine.model is not None):
                from enigma_engine.core.safe_save import atomic_torch_save
                import dataclasses
                import math
                m = state.engine.model
                cfg = getattr(m, "config", None)
                cfg_dict = (
                    dataclasses.asdict(cfg)
                    if cfg is not None and dataclasses.is_dataclass(cfg)
                    else {}
                )
                atomic_torch_save(
                    {
                        "model_state_dict": m.state_dict(),
                        "model_config": cfg_dict,
                        "training_state": {
                            "epochs": getattr(result, "epoch", 0),
                            "best_loss": getattr(result, "best_loss",
                                                 float("inf")),
                        },
                    },
                    model_path,
                )
                saved_path = Path(model_path).name  # Use basename only (M1 fix)
                logger.info("Training complete — model saved to %s", saved_path)

            best = getattr(result, "best_loss", float("inf"))
            import math
            with _training_lock:
                _training_state.update({
                    "active": False,
                    "progress": 100,
                    "message": "Training complete",
                    "best_loss": None if (best is None or math.isnan(best) or math.isinf(best)) else best,
                    "output_path": saved_path,
                    "abort_reason": "",
                })
        except Exception as exc:
            logger.exception("Training error")
            with _training_lock:
                _training_state.update({
                    "active": False,
                    "progress": 0,
                    "message": f"Training failed: {exc}",
                    "abort_reason": str(exc),
                })
        finally:
            with _training_lock:
                _active_training_trainer = None

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()

    return {
        "status": "started",
        "mode": job.mode,
        "data_file": req.data_file,
        "epochs": job.training.epochs,
    }


# ---------------------------------------------------------------------------
# Server runner (called from run.py --serve)
# ---------------------------------------------------------------------------

def run_server(host: str = "127.0.0.1", port: int = 8080, model_path: str | None = None,
               api_key: str | None = None, cors_origins: list[str] | None = None):
    """Start the local API server.

    Args:
        host: Bind address. Defaults to 127.0.0.1 (localhost only).
              Use '0.0.0.0' to expose to the local network.
        port: Port number.
        model_path: Optional model to pre-load.
        api_key: Optional API key. When set, all /api/* requests must
                 include an ``Authorization: Bearer <key>`` header.
        cors_origins: Optional list of allowed CORS origins.
                      CORS middleware is only added when this is provided.
    """
    import uvicorn

    # Install CORS middleware only when explicitly requested
    if cors_origins:
        enable_cors(cors_origins)
    else:
        logger.info("CORS disabled (pass --cors-origins to enable)")

    # Install API-key middleware when a key is provided
    if api_key:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import JSONResponse as _JSONResponse

        _key = api_key  # capture in closure

        class _APIKeyMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                # Skip key check for CORS preflight OPTIONS requests
                if request.method == "OPTIONS":
                    return await call_next(request)
                if request.url.path.startswith("/api/"):
                    auth = request.headers.get("authorization", "")
                    if not hmac.compare_digest(auth, f"Bearer {_key}"):
                        return _JSONResponse(
                            {"error": "Invalid or missing API key"},
                            status_code=401,
                        )
                return await call_next(request)

        app.add_middleware(_APIKeyMiddleware)

    # MC-2: restore persisted conversations from disk.
    state.load_conversations_from_disk()
    # MC-2b: restore the active-conversation pointer (if the saved id
    # is still in the just-loaded histories).
    state._load_active_conv_id_from_disk()

    # Pre-load a model if specified
    if model_path:
        logger.info("Pre-loading model: %s", model_path)
        try:
            state.load_model(model_path)
            params = f"{state.model_info_snapshot().get('parameters', 0):,}"
            logger.info("Model loaded: %s params", params)
        except Exception as exc:
            logger.warning("Could not pre-load model: %s", exc)
            logger.info(
                "Server starting without a model. "
                "Load one via POST /api/models/load")

    logger.info("Enigma Engine API Server")
    logger.info("http://localhost:%s", port)
    logger.info("API docs: http://localhost:%s/api/docs", port)
    logger.info("Press Ctrl+C to stop")

    uvicorn.run(app, host=host, port=port, log_level="info")
