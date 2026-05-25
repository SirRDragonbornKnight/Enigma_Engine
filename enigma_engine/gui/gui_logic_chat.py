"""
Enigma Engine - GUI Chat Logic
=================================

Mixin providing chat messaging, session management,
thinking indicator, typewriter effect, file attachment,
history management, and system prompt handling.
Split from gui_logic.py to keep files under 800 lines.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog
from typing import Any

from enigma_engine.gui.widgets import C_ACCENT
from enigma_engine.gui.scanners import (
    CONFIG_LIMITS, DATA_DIR, MEMORY_DIR,
    clamp_config, scan_sessions,
)
from enigma_engine.core.model_context import load_model_context

logger = logging.getLogger(__name__)


class LogicChatMixin:
    """Mixin providing chat, session, and history logic.

    Expects the host class to have chat_display, history_list,
    prompt_editor, history, engine, config_overrides, etc.
    """

    # ================================================================
    # Logic - Chat input history (Up/Down arrow recall)
    # ================================================================

    _INPUT_HISTORY_MAX = 50

    def _init_input_history(self):
        """Initialize input history state. Called once during setup."""
        self._input_history: list[str] = []
        self._input_hist_idx: int = -1
        self._input_hist_draft: str = ""

    def _on_input_up(self, event):
        """Up arrow: recall previous input (only from first line)."""
        tb = self.chat_input._textbox
        # Only activate when cursor is on the first line
        cursor_line = int(tb.index("insert").split(".")[0])
        if cursor_line != 1:
            return  # let default behavior handle it
        if not self._input_history:
            return "break"
        if self._input_hist_idx == -1:
            # Starting to browse — save current draft
            self._input_hist_draft = self.chat_input.get(
                "1.0", "end").strip()
            self._input_hist_idx = len(self._input_history) - 1
        elif self._input_hist_idx > 0:
            self._input_hist_idx -= 1
        else:
            return "break"  # already at oldest
        self._set_input_text(self._input_history[self._input_hist_idx])
        return "break"

    def _on_input_down(self, event):
        """Down arrow: go forward in history or back to draft."""
        tb = self.chat_input._textbox
        # Only activate when cursor is on the last line
        cursor_line = int(tb.index("insert").split(".")[0])
        last_line = int(tb.index("end-1c").split(".")[0])
        if cursor_line != last_line:
            return  # let default behavior handle it
        if self._input_hist_idx == -1:
            return "break"  # not browsing
        if self._input_hist_idx < len(self._input_history) - 1:
            self._input_hist_idx += 1
            self._set_input_text(
                self._input_history[self._input_hist_idx])
        else:
            # Past the newest — restore draft
            self._input_hist_idx = -1
            self._set_input_text(self._input_hist_draft)
        return "break"

    def _set_input_text(self, text: str):
        """Replace chat input content and resize."""
        self.chat_input.delete("1.0", "end")
        self.chat_input.insert("1.0", text)
        self._auto_resize_input()

    # ================================================================
    # Logic - Chat
    # ================================================================

    def _on_input_enter(self, event):
        """Enter sends message. Shift+Enter adds newline."""
        if event.state & 0x1:
            return
        # Prevent double-send crash while AI is generating
        if getattr(self, '_is_generating', False):
            return "break"
        self._send_message()
        return "break"

    _INPUT_MIN_H = 56
    _INPUT_MAX_H = 200

    def _auto_resize_input(self, _event=None):
        """Grow or shrink the chat input box to fit its content."""
        tb = self.chat_input._textbox
        num_lines = int(tb.index("end-1c").split(".")[0])
        # ~28 px per line, clamped between min and max
        desired = max(self._INPUT_MIN_H,
                      min(num_lines * 28, self._INPUT_MAX_H))
        self.chat_input.configure(height=desired)

    def _build_api_chat_client(self):
        """Create an EnigmaClient for GUI chat-over-API mode."""
        from enigma_engine import EnigmaClient

        base_url = str(
            getattr(self, "api_base_url", "http://127.0.0.1:8080")
            or "http://127.0.0.1:8080"
        ).strip()
        return EnigmaClient(base_url)

    def _get_api_chat_client(self):
        """Return a cached EnigmaClient instance."""
        client = getattr(self, "_api_chat_client", None)
        if client is None:
            client = self._build_api_chat_client()
            self._api_chat_client = client
        return client

    def _build_api_chat_payload(
            self,
            message: str,
            kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Build API-facing message and kwargs from GUI chat kwargs."""
        api_kwargs: dict[str, Any] = {}
        for key in (
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "repetition_penalty",
            "json_schema",
        ):
            if key in kwargs and kwargs[key] is not None:
                api_kwargs[key] = kwargs[key]

        system_prompt = str(kwargs.get("system_prompt", "") or "").strip()
        api_message = message
        if system_prompt:
            api_message = (
                "[SYSTEM PROMPT]\n"
                f"{system_prompt}\n"
                "[/SYSTEM PROMPT]\n\n"
                f"{message}"
            )

        return api_message, api_kwargs

    def _chat_request_stream(
            self,
            message: str,
            kwargs: dict[str, Any]):
        """Return an API stream iterator when API mode is enabled.

        Returns ``None`` when API mode is disabled or stream setup fails.
        """
        use_api_chat = getattr(self, "use_api_chat", False) is True
        if not use_api_chat:
            return None

        client = self._get_api_chat_client()
        api_message, api_kwargs = LogicChatMixin._build_api_chat_payload(
            self, message, kwargs)

        try:
            return client.chat_stream(
                api_message,
                web_access=False,
                **api_kwargs,
            )
        except Exception as exc:
            logger.info(
                "API stream chat failed; falling back to non-stream chat: %s",
                exc,
            )
            return None

    def _chat_request(
            self,
            message: str,
            kwargs: dict[str, Any],
            *,
            prefer_stream: bool = True) -> str:
        """Route one chat request via API client or local engine.

        API mode is opt-in via ``self.use_api_chat``. On API failure,
        we fall back to local engine mode when available.
        """
        use_api_chat = getattr(self, "use_api_chat", False) is True

        if use_api_chat:
            try:
                client = self._get_api_chat_client()
                api_message, api_kwargs = LogicChatMixin._build_api_chat_payload(
                    self, message, kwargs)

                if prefer_stream:
                    try:
                        chunks = list(client.chat_stream(
                            api_message,
                            web_access=False,
                            **api_kwargs,
                        ))
                        if chunks:
                            return "".join(chunks)
                    except Exception as stream_exc:
                        logger.info(
                            "API stream chat failed; trying non-stream chat: %s",
                            stream_exc,
                        )

                return client.chat(
                    api_message,
                    web_access=False,
                    **api_kwargs,
                )
            except Exception as exc:
                logger.warning(
                    "API chat failed; falling back to local engine: %s",
                    exc,
                )

        if self.engine is None:
            raise RuntimeError(
                "No local model loaded and API chat unavailable. "
                "Load a model or disable API chat mode."
            )
        try:
            return self.engine.chat(message, **kwargs)
        except TypeError:
            return self.engine.chat(message)

    def _postprocess_response_text(self, response: str) -> tuple[str, str]:
        """Normalize response text for downstream processing.

        Returns ``(thinking_text, clean_response)`` where incomplete
        reasoning tags are removed and complete ``<think>`` blocks are
        extracted from the answer text.
        """
        from enigma_engine.core.reasoning import (
            extract_reasoning,
            has_reasoning,
            strip_incomplete_think,
        )

        if has_reasoning(response):
            return extract_reasoning(response)
        return "", strip_incomplete_think(response)

    def _send_message(self):
        # Guard against double-send while generation is running
        if getattr(self, '_is_generating', False):
            return
        msg = self.chat_input.get("1.0", "end").strip()
        if not msg:
            return
        # Record in input history for Up/Down recall
        hist = getattr(self, "_input_history", None)
        if hist is None:
            self._init_input_history()
            hist = self._input_history
        if not hist or hist[-1] != msg:
            hist.append(msg)
            if len(hist) > self._INPUT_HISTORY_MAX:
                hist.pop(0)
        self._input_hist_idx = -1
        use_api_chat = getattr(self, "use_api_chat", False) is True
        if self.engine is None and not use_api_chat:
            self._chat_system(
                "No model loaded. Go to ROUTER and load one first.")
            return

        # Ensure a session path exists for auto-save
        if not getattr(self, "_current_session_path", ""):
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            counter = getattr(self, "_session_counter", 0) + 1
            self._session_counter = counter
            ts = time.strftime("%Y%m%d_%H%M%S")
            name = f"session_{ts}_{counter}"
            self._current_session_path = str(
                MEMORY_DIR / f"{name}.json")

        # Handle file attachment
        file_context = ""
        if self.attached_file:
            try:
                fpath = Path(self.attached_file)
                fsize = fpath.stat().st_size
                # Detect binary files by reading a sample
                try:
                    sample = fpath.read_bytes()[:8192]
                    if b'\x00' in sample:
                        content = (
                            f"[Binary file: {fpath.name}, "
                            f"{fsize:,} bytes]")
                        logger.info(
                            "Attachment %s is binary (%s bytes)",
                            fpath.name, fsize)
                    elif fsize > 1_000_000:  # 1MB safety cap
                        with open(fpath, encoding="utf-8",
                                  errors="replace") as f:
                            content = f.read(4000)
                        content += "\n[...truncated — file too large]"
                        logger.warning(
                            "Attachment %s truncated (%s bytes)",
                            fpath.name, fsize)
                    else:
                        content = fpath.read_text(
                            encoding="utf-8", errors="replace")
                        if len(content) > 4000:
                            content = content[:4000] + "\n[...truncated]"
                except OSError:
                    content = f"[Could not read: {fpath.name}]"
                file_context = (
                    f"\n[Attached file: "
                    f"{Path(self.attached_file).name}]"
                    f"\n{content}\n")
            except OSError:
                file_context = (
                    f"\n[Failed to read: {self.attached_file}]")
            self._clear_attachment()

        self.chat_input.delete("1.0", "end")
        self._auto_resize_input()  # Reset input height after clearing
        timestamp = time.strftime("%H:%M")
        self._chat_append(
            "timestamp", f"\n  {timestamp} ",
            "user_prefix", f"{self.user_name}  ",
            "user", msg + "\n")
        # Render any media/links the user posted
        self._process_media_in_text(msg, "user")
        if file_context:
            self._chat_append("file_tag", "  [file attached]\n")
        self._is_generating = True
        self._stop_requested = False
        self.send_btn.configure(state="disabled")
        # Show STOP button, hide SEND button
        try:
            self.stop_btn.grid(row=1, column=1, sticky="s")
            self.send_btn.grid_forget()
        except Exception as exc:
            logger.debug("Button swap to STOP failed: %s", exc)
        self._show_thinking()

        full_msg = msg + file_context

        # Build base prompt on main thread (fast: reads GUI widgets)
        # Auto-research (network I/O) is deferred to background thread
        try:
            user_prompt = self.prompt_editor.get("1.0", "end").strip()
        except Exception:
            user_prompt = ""
        gui_ctx = self._build_gui_context()
        _msg_for_research = msg
        _web_access_on = getattr(self, "web_access", False)
        _pending_exchange_image_path = str(
            getattr(self, "_pending_correction_image_path", "") or ""
        ).strip()

        def _gen():
            try:
                # Build combined prompt — auto-research runs here
                # (off main thread) so network I/O doesn't freeze GUI
                web_research_ctx = ""
                if _web_access_on:
                    try:
                        from enigma_engine.core.auto_research import (
                            auto_research, should_auto_research)
                        if should_auto_research(_msg_for_research):
                            web_research_ctx = auto_research(
                                _msg_for_research, max_results=3)
                    except Exception as exc:
                        logger.debug("Auto-research failed: %s", exc)
                combined_prompt = (
                    f"{user_prompt}\n\n{gui_ctx}" if user_prompt
                    else gui_ctx)
                if web_research_ctx:
                    combined_prompt += f"\n\n{web_research_ctx}"

                kwargs: dict[str, Any] = {}
                kwargs.update(self.config_overrides)
                kwargs["system_prompt"] = combined_prompt
                kwargs["history"] = list(self.history)

                # Enable reasoning if toggle is on
                if getattr(self, 'reasoning_enabled', False):
                    kwargs["reasoning"] = True

                # N-15b (Pass 156z9aa): forward GUI-staged JSON
                # schema constraint to engine.chat.  None = no
                # constraint (the common case); non-None means the
                # user pasted+Applied a schema on the CONFIG page.
                # Engine validates the schema dict; GGUF backend
                # raises NotImplementedError (constraint never
                # reaches llama.cpp's sampler — Pass 156z3 contract).
                _gui_json_schema = getattr(self, "json_schema", None)
                if _gui_json_schema is not None:
                    kwargs["json_schema"] = _gui_json_schema

                # Log to CMD activity
                ai = self._active_ai_name()
                self._cmd_activity(
                    "info", f"[{ai}] Processing: {msg[:80]}...\n"
                    if len(msg) > 80
                    else f"[{ai}] Processing: {msg}\n")

                try:
                    stream_iter = self._chat_request_stream(full_msg, kwargs)
                    resp = ""
                    streamed_live = False
                    if stream_iter is not None:
                        stream_chunks: list[str] = []
                        ts = time.strftime("%H:%M")
                        for chunk in stream_iter:
                            if getattr(self, '_stop_requested', False):
                                break
                            if not chunk:
                                continue
                            if not streamed_live:
                                streamed_live = True

                                def _show_stream_header(
                                        t: str = ts):
                                    self._hide_thinking()
                                    ai_name = self._active_ai_name()
                                    self._chat_append(
                                        "timestamp", f"\n  {t} ",
                                        "assistant_prefix",
                                        f"{ai_name}  ")

                                self.after(0, _show_stream_header)

                            stream_chunks.append(chunk)
                            self.after(
                                0,
                                lambda c=chunk: self._append_stream_chunk(c),
                            )

                        resp = "".join(stream_chunks)

                    if not resp:
                        # Either no stream path or stream produced no tokens.
                        # Fall back to one-shot chat without retrying stream.
                        resp = self._chat_request(
                            full_msg, kwargs, prefer_stream=False)
                except ValueError as exc:
                    # N-15b (Pass 156z9ab) — JsonSchemaConstraint
                    # raises ValueError on unsupported schema shapes
                    # (non-object root, missing properties, malformed
                    # spec).  Surface a friendly status message and
                    # ABORT — do NOT silently retry without the
                    # constraint, that would generate unconstrained
                    # output the user explicitly opted out of.
                    schema_err = str(exc)
                    logger.warning(
                        "JSON schema rejected by engine: %s",
                        schema_err)

                    def _show_schema_err(
                            m: str = schema_err) -> None:
                        self._hide_thinking()
                        self._chat_system(
                            f"JSON schema rejected: {m}. "
                            "Edit or Clear the schema on the CONFIG "
                            "page, then resend."
                        )
                        if hasattr(self, "status_bar"):
                            self.status_bar.set_left(
                                f"[!] JSON schema rejected: {m[:80]}"
                            )
                    self.after(0, _show_schema_err)
                    return

                # Check if user hit STOP during generation
                if getattr(self, '_stop_requested', False):
                    def _stopped():
                        self._hide_thinking()
                        self._chat_system("Generation stopped.")
                    self.after(0, _stopped)
                    return

                # Normalize assistant text for command parsing/history/TTS.
                # In stream mode the visible transcript already contains the
                # raw chunk text; this post-process path only affects internal
                # handling (stored history, command parsing, TTS payload).
                thinking_text, resp = self._postprocess_response_text(resp)

                if not streamed_live:
                    # AutoResearch-2 Stage A wiring (Pass 154):
                    # Post-generation uncertainty gate. If web access is on,
                    # no pre-gen research ran, and the visible reply scores
                    # >= threshold uncertain, retry once with research context.
                    # Streamed path intentionally skips this: once chunks are
                    # already visible in the transcript, we do not replace the
                    # answer with a second hidden retry response.
                    if (_web_access_on
                            and not web_research_ctx
                            and not getattr(self, '_stop_requested', False)):
                        try:
                            from enigma_engine.core.auto_research import (
                                auto_research as _ar_fetch,
                                should_retry_with_research)
                            if should_retry_with_research(
                                    _msg_for_research, resp):
                                retry_ctx = _ar_fetch(
                                    _msg_for_research, max_results=3)
                                if retry_ctx:
                                    retry_kwargs = dict(kwargs)
                                    retry_kwargs["system_prompt"] = (
                                        f"{combined_prompt}\n\n{retry_ctx}")
                                    logger.info(
                                        "AutoResearch-2: low-confidence "
                                        "reply, retrying with research "
                                        "context")
                                    resp = self._chat_request(
                                        full_msg,
                                        retry_kwargs,
                                        prefer_stream=False)
                                    thinking_text, resp = (
                                        self._postprocess_response_text(resp))
                        except Exception as exc:
                            logger.debug(
                                "AutoResearch-2 retry failed: %s", exc)

                # Parse and execute [CMD] blocks from response
                from enigma_engine.core.commands import (
                    parse_commands)
                clean_text, commands = parse_commands(resp)
                cmd_output = ""
                if commands:
                    cmd_output = self._cmd_execute_ai_commands(
                        commands)

                self.history.append(
                    {"role": "user", "content": msg})
                self.history.append(
                    {"role": "assistant", "content": clean_text})
                self._last_exchange_prompt = msg
                self._last_exchange_wrong_response = clean_text
                self._last_exchange_image_path = _pending_exchange_image_path
                self._pending_correction_image_path = ""

                # Trim history to prevent unbounded RAM growth
                self._trim_chat_history()

                # Track message stats in identity card
                ctx = getattr(self, "model_context", None)
                if ctx is not None:
                    ctx.increment_messages(2)

                # Extract memorable facts from user message
                # Only auto-extract in "automatic" mode;
                # "manual" relies on explicit commands,
                # "disabled" blocks add() at the core level.
                try:
                    mem_mode = self._get_memory_mode()
                    if mem_mode == "automatic":
                        from enigma_engine.core.memory import get_memory
                        mem = get_memory()
                        new_facts = mem.extract_facts(msg)
                        if new_facts:
                            logger.info(
                                "Auto-remembered: %s",
                                ", ".join(new_facts))
                except Exception as exc:
                    logger.debug("Auto-remember failed: %s", exc)

                # Update emotional state from user message
                try:
                    ctx = getattr(self, "model_context", None)
                    if ctx is not None:
                        ctx.update_emotional_state(msg)
                        self.after(
                            0, self._update_emotional_display)
                except Exception as exc:
                    logger.debug(
                        "Emotional state update failed: %s", exc)

                # Auto-save per-model context after each exchange
                self._save_model_context()
                # Auto-save session to memory/
                self._auto_save_session()
                # Generate AI session title after the first exchange
                if len(self.history) == 2:
                    self._generate_session_title(msg, clean_text)
                # Feed exchange to BackgroundTrainer if enabled
                self._feed_background_trainer(msg, clean_text)
                # Collect image paths generated by commands
                cmd_images = list(
                    getattr(self, "_cmd_image_paths", []))
                self._cmd_image_paths = []
                cmd_files = list(
                    getattr(self, "_cmd_file_paths", []))
                self._cmd_file_paths = []
                ts = time.strftime("%H:%M")
                if streamed_live:
                    def _finish_streamed(r=clean_text, co=cmd_output,
                                         imgs=cmd_images,
                                         files=cmd_files):
                        ai = self._active_ai_name()
                        self._hide_thinking()
                        self._chat_append("assistant", "\n")
                        if co:
                            self._chat_system(co)
                        for img_path in imgs:
                            self._insert_media(img_path, "file")
                        for file_path in files:
                            self._insert_file_link(file_path)
                        self._cmd_activity(
                            "info",
                            f"[{ai}] Response delivered "
                            f"({len(r)} chars)\n")
                        self._cmd_activity("divider", "\n")
                        self._update_token_counter()
                        self._tts_speak(r)

                    self.after(0, _finish_streamed)
                else:
                    def _show(r=clean_text, t=ts, co=cmd_output,
                              think=thinking_text,
                              imgs=cmd_images,
                              files=cmd_files):
                        self._hide_thinking()
                        ai = self._active_ai_name()
                        self._chat_append(
                            "timestamp", f"\n  {t} ",
                            "assistant_prefix", f"{ai}  ")
                        # Show reasoning section if present
                        if think:
                            self._chat_append(
                                "reasoning_label",
                                "\n  \U0001f9e0 Reasoning:\n",
                                "reasoning",
                                think + "\n")
                        # Store deferred output to show after typewriter finishes
                        self._deferred_cmd_output = co
                        self._deferred_cmd_images = imgs
                        self._deferred_cmd_files = files
                        self._deferred_ai_name = ai
                        self._typewriter("assistant", r + "\n")
                        # Speak only the answer (not the reasoning)
                        self._tts_speak(r)
                        # Command output and images are now deferred until
                        # after typewriter finishes
                    self.after(0, _show)
            except Exception as exc:
                tb = traceback.format_exc()
                logger.error("Chat generation failed:\n%s", tb)
                def _err(e=str(exc)):
                    self._hide_thinking()
                    self._chat_error(e)
                    # Clear deferred output so stale data doesn't
                    # leak into the next successful response
                    self._deferred_cmd_output = ""
                    self._deferred_cmd_images = []
                    self._deferred_cmd_files = []
                    self._deferred_ai_name = ""
                self.after(0, _err)
            finally:
                def _restore_send():
                    self._is_generating = False
                    self._stop_requested = False
                    self.send_btn.configure(state="normal")
                    # Restore SEND button, hide STOP button
                    try:
                        self.send_btn.grid(row=1, column=1,
                                           sticky="s")
                        self.stop_btn.grid_forget()
                    except Exception as exc:
                        logger.debug(
                            "Button swap to SEND failed: %s", exc)
                self.after(0, _restore_send)

        threading.Thread(target=_gen, daemon=True).start()

    def _stop_generation(self):
        """Stop the current AI generation and typewriter animation."""
        self._stop_requested = True
        # Pass 156z9fg: propagate stop to the engine so the daemon
        # generation thread breaks out of its token loop instead of
        # burning GPU producing tokens the UI has already discarded.
        # Gated on _generation_lock.locked() so a stale True flag
        # from an idle stop cannot eat the next generation (mirrors
        # the gate in builtin_commands.stop_cmd from Pass 156z9ff).
        engine = getattr(self, "engine", None)
        if engine is not None:
            lock = getattr(engine, "_generation_lock", None)
            if lock is not None and lock.locked():
                engine._cancel_generation = True
        self._hide_thinking()
        # Stop TTS if voice is speaking
        self._tts_stop()
        # Re-enable send controls immediately
        self._is_generating = False
        try:
            self.send_btn.configure(state="normal")
            self.send_btn.grid(row=1, column=1, sticky="s")
            self.stop_btn.grid_forget()
        except Exception as exc:
            logger.debug("Stop-generation UI restore failed: %s", exc)
        self._chat_system("Generation stopped by user.")

    def _update_token_counter(self):
        """Update the token counter label with current history size."""
        counter = getattr(self, "_token_counter", None)
        if counter is None:
            return
        history = getattr(self, "history", [])
        # Estimate tokens: ~4 chars per token (simple, fast heuristic)
        total_chars = sum(
            len(m.get("content", "")) for m in history)
        token_est = total_chars // 4
        try:
            counter.configure(text=f"{token_est} tokens")
        except Exception:
            logger.debug("Token counter update failed")

    def _edit_last_message(self):
        """Remove the last user+assistant exchange and put the
        user message back in the input box for editing.

        Guards against editing while generating or with empty history.
        """
        # Cannot edit while AI is generating (also check _stop_requested
        # because _stop_generation() eagerly clears _is_generating while
        # the daemon thread may still be appending to self.history)
        if getattr(self, '_is_generating', False) \
                or getattr(self, '_stop_requested', False):
            self._chat_system("Cannot edit while AI is generating.")
            return
        if not self.history:
            self._chat_system("No messages to edit.")
            return
        # Find the last user message in history
        last_user_msg = ""
        # Remove last assistant + user pair (they come in pairs)
        if (len(self.history) >= 2
                and self.history[-1].get("role") == "assistant"
                and self.history[-2].get("role") == "user"):
            last_user_msg = self.history[-2].get("content", "")
            self.history.pop()  # Remove assistant
            self.history.pop()  # Remove user
        elif self.history[-1].get("role") == "user":
            # Only a user message (no response yet)
            last_user_msg = self.history[-1].get("content", "")
            self.history.pop()
        else:
            self._chat_system("No user message to edit.")
            return
        # Redisplay remaining history
        self._restore_history_display()
        # Put the message back in the input
        self.chat_input.delete("1.0", "end")
        self.chat_input.insert("1.0", last_user_msg)
        self.chat_input.focus()
        # Auto-save updated history
        self._save_model_context()
        self._auto_save_session()

    def _append_correction_jsonl(
            self,
            row: dict[str, Any],
            target_path: Path | None = None) -> None:
        """Append one correction row to corrections.jsonl.

        Uses bounded I/O (tail-byte check + append) so repeated saves
        stay O(1) per row instead of rewriting the full file each time.
        """
        path = target_path or (DATA_DIR / "corrections.jsonl")
        line = json.dumps(row, ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab+") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                if size > 0:
                    handle.seek(-1, os.SEEK_END)
                    if handle.read(1) != b"\n":
                        handle.write(b"\n")
                handle.write(encoded)
        except OSError as exc:
            raise OSError(f"failed writing correction store: {exc}") from exc

    def _record_correction_for_last_exchange(self, right_response: str) -> bool:
        """Store a correction for the most recent user/assistant exchange."""
        corrected = (right_response or "").strip()
        if not corrected:
            self._chat_system("Enter a corrected answer first.")
            return False
        if len(self.history) < 2:
            self._chat_system("No assistant reply available to correct yet.")
            return False
        if not (
            self.history[-1].get("role") == "assistant"
            and self.history[-2].get("role") == "user"
        ):
            self._chat_system("Last exchange is not a user/assistant pair.")
            return False

        prompt = str(self.history[-2].get("content", "")).strip()
        wrong_response = str(self.history[-1].get("content", "")).strip()
        if not prompt or not wrong_response:
            self._chat_system("Cannot save correction for an empty exchange.")
            return False

        exchange_image_path = ""
        if (
            getattr(self, "_last_exchange_prompt", "") == prompt
            and getattr(self, "_last_exchange_wrong_response", "") == wrong_response
        ):
            exchange_image_path = str(
                getattr(self, "_last_exchange_image_path", "") or ""
            ).strip()
        modality = "vision" if exchange_image_path else "text"

        row = {
            "prompt": prompt,
            "wrong_response": wrong_response,
            "right_response": corrected,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "modality": modality,
            "model_path": getattr(self, "model_path", ""),
            "session_path": getattr(self, "_current_session_path", ""),
        }
        if exchange_image_path:
            row["image_path"] = exchange_image_path
        try:
            self._append_correction_jsonl(row)
        except OSError as exc:
            self._chat_error(str(exc))
            return False

        self._chat_system("Correction saved to data/corrections.jsonl")
        return True

    def _save_last_correction_from_input(self):
        """Capture correction text from chat input for the last assistant reply."""
        try:
            corrected = self.chat_input.get("1.0", "end").strip()
        except Exception:
            corrected = ""
        if not self._record_correction_for_last_exchange(corrected):
            return
        self.chat_input.delete("1.0", "end")
        self._auto_resize_input()

    def _chat_append(self, *tag_text_pairs):
        """Append multiple (tag, text) pairs to chat display."""
        if len(tag_text_pairs) % 2 != 0:
            logger.error(
                "_chat_append called with odd arg count (%d) "
                "— last tag has no text, skipping it.",
                len(tag_text_pairs))
            tag_text_pairs = tag_text_pairs[:-1]
        tb = self.chat_display._textbox
        for i in range(0, len(tag_text_pairs), 2):
            tag = tag_text_pairs[i]
            text = tag_text_pairs[i + 1]
            tb.insert("end", text, tag)
        self._auto_resize_chat()
        self._scroll_chat_to_bottom()

    def _chat_system(self, text: str):
        """Show a SYSTEM message as a visible third speaker."""
        ts = time.strftime("%H:%M")
        self._chat_append(
            "timestamp", f"\n  {ts} ",
            "system_prefix", "System  ",
            "system_msg", text + "\n")

    def _chat_error(self, text: str):
        """Show a SYSTEM error as a visible third speaker."""
        ts = time.strftime("%H:%M")
        self._chat_append(
            "timestamp", f"\n  {ts} ",
            "system_prefix", "System  ",
            "error", text + "\n")

    def _chat_session_marker(self, text: str):
        """Render a divider line marking a session-state change.

        Pass 156v Step 1 (Session-1 unification): single source of
        truth for visually distinct chat-log markers when runtime
        state changes (LoRA adapter swap / clear / stack apply).
        Future incremental adoption: model swap, profile swap,
        system-prompt edit, RAG corpus change.

        The divider uses the dedicated ``session_marker`` text-tag
        (configured on the chat display) so it is visually separate
        from regular ``_chat_system`` messages — the user can scan
        the chat log and find the seam where weights changed if
        quality regresses afterwards.

        Use this for SUCCESS state changes only. Errors (e.g.
        engine raised, file missing) and load-first hints continue
        to use ``_chat_error`` and ``_chat_system`` respectively.

        Args:
            text: Short human-readable description of the change
                (e.g. ``"LoRA adapter: foo_lora"``,
                ``"LoRA stack: foo@0.70, bar@0.30"``,
                ``"using base weights"``). The helper wraps it in
                horizontal bars for the divider look.
        """
        self._chat_append(
            "session_marker", f"\n  ─── {text} ───\n")

    def _reset_display(self):
        """Clear the chat display widget only."""
        # Stop all GIF animations
        for anim in getattr(self, "_chat_gif_animations", []):
            anim["active"] = False
        self._chat_gif_animations = []
        self._chat_images = []
        self._chat_media_refs = {}
        self._chat_file_refs = {}
        self._link_urls = {}
        self.chat_display.clear()
        self._auto_resize_chat()

    def _auto_resize_chat(self):
        """No-op — native CTkTextbox scrollbar handles scrolling.

        Previously this tried to expand the text widget height to
        match content inside a CTkScrollableFrame wrapper.  That
        estimation was unreliable and caused hidden text.  Now the
        built-in tk.Text scrollbar does all the work.
        """

    def _sync_chat_display_height(self):
        """No-op — native CTkTextbox scrollbar handles scrolling."""

    def _scroll_chat_to_bottom(self):
        """Scroll chat display to show the latest content.

        Uses the native tk.Text see() method which is lightweight
        and reliable — no geometry hacks needed.
        """
        try:
            self.chat_display._textbox.see("end")
        except Exception:
            pass

    def _trim_chat_history(self):
        """Drop oldest messages when history exceeds the cap.

        Prevents unbounded RAM growth during long conversations.
        Full history is still saved to disk in session files.
        Only the in-memory list is capped.
        """
        from enigma_engine.gui.media import MAX_CHAT_HISTORY
        if len(self.history) > MAX_CHAT_HISTORY:
            # Drop oldest messages, keeping the most recent
            trim_count = len(self.history) - MAX_CHAT_HISTORY
            del self.history[:trim_count]
            logger.info(
                f"Trimmed {trim_count} old messages from RAM "
                f"(kept {MAX_CHAT_HISTORY} most recent)"
            )

    def _trim_chat_images(self):
        """Drop oldest PhotoImage refs when the list exceeds the cap.

        Images embedded via image_create() keep a Python reference
        to prevent garbage collection.  Without a cap the list grows
        forever, leaking memory during long conversations.

        Also stops old GIF animations and releases their frame refs
        to prevent unbounded memory growth from animated GIFs.
        """
        from enigma_engine.gui.media import MAX_CHAT_IMAGES
        imgs = getattr(self, "_chat_images", [])
        if len(imgs) > MAX_CHAT_IMAGES:
            # Drop the oldest refs — tk will blank those images but
            # the chat text and captions remain readable.
            del imgs[:len(imgs) - MAX_CHAT_IMAGES]

        # Cap active GIF animations (each holds N frames of PhotoImages)
        max_gifs = 5
        anims = getattr(self, "_chat_gif_animations", [])
        while len(anims) > max_gifs:
            old = anims.pop(0)
            old["active"] = False   # stop the after() loop
            old["frames"] = []      # release frame PhotoImage refs

    def _new_chat(self):
        """Start a fresh conversation - clear chat and reset AI state.

        Creates a new session file so the previous chat persists
        in the sidebar history.  No confirmation needed — the
        current chat is auto-saved anyway.
        """
        # S618: Stop generation before clearing history to prevent
        # the daemon thread from appending to a cleared list.
        if getattr(self, '_is_generating', False):
            self._stop_generation()
        self._reset_display()
        self.history.clear()
        self._pending_correction_image_path = ""
        self._last_exchange_image_path = ""
        self._last_exchange_prompt = ""
        self._last_exchange_wrong_response = ""
        self._update_token_counter()
        # Create a new session path with counter for uniqueness
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        counter = getattr(self, "_session_counter", 0) + 1
        self._session_counter = counter
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = f"session_{ts}_{counter}"
        self._current_session_path = str(
            MEMORY_DIR / f"{name}.json")
        # Track session count in identity card
        ctx = getattr(self, "model_context", None)
        if ctx is not None:
            ctx.increment_sessions()
        # Save cleared history to model context
        self._save_model_context()
        use_api_chat = getattr(self, "use_api_chat", False) is True
        if use_api_chat:
            get_client = getattr(self, "_get_api_chat_client", None)
            if callable(get_client):
                try:
                    get_client().clear_history()
                except Exception as exc:
                    logger.warning("API clear history failed: %s", exc)
        if self.engine:
            if hasattr(self.engine, "clear_history"):
                self.engine.clear_history()
            if hasattr(self.engine, "clear_kv_cache"):
                self.engine.clear_kv_cache()
        self._chat_system("New conversation started.")
        self._refresh_history_list()

    # ================================================================
    # Logic - Per-model context
    # ================================================================

    def _save_model_context(self):
        """Save current history and prompt to the loaded model's context."""
        ctx = getattr(self, "model_context", None)
        if ctx is None:
            return
        ctx.history = list(self.history)
        # Track which session file this history lives in
        ctx.session_path = getattr(
            self, "_current_session_path", "")
        # Capture current system prompt
        try:
            prompt = self.prompt_editor.get("1.0", "end").strip()
            if prompt:
                ctx.system_prompt = prompt
        except Exception as exc:
            logger.debug("Save model context prompt failed: %s", exc)
        # Capture config overrides
        ctx.config = dict(self.config_overrides)
        try:
            ctx.save()
        except Exception as exc:
            logger.error("Failed to save model context: %s", exc)

    def _load_model_context(self, model_path: str):
        """Load per-model context when a model is loaded."""
        ctx = load_model_context(model_path)
        self.model_context = ctx
        # Restore history
        self.history = list(ctx.history)
        # Resume saved session file instead of creating a duplicate
        if ctx.session_path and Path(ctx.session_path).exists():
            self._current_session_path = ctx.session_path
        # Restore system prompt into editor
        if ctx.system_prompt:
            try:
                self.prompt_editor.delete("1.0", "end")
                self.prompt_editor.insert("1.0", ctx.system_prompt)
                if (self.engine
                        and hasattr(self.engine, "system_prompt")):
                    self.engine.system_prompt = ctx.system_prompt
            except Exception as exc:
                logger.debug("Load model context prompt failed: %s", exc)
        # Restore config overrides
        if ctx.config:
            for key, val in ctx.config.items():
                if key in CONFIG_LIMITS:
                    clamped = clamp_config(key, val)
                    self.config_overrides[key] = clamped
                    entry = self.config_entries.get(key)
                    if entry:
                        lo, hi, step = CONFIG_LIMITS[key]
                        entry.delete(0, "end")
                        if step == int(step) and lo == int(lo):
                            entry.insert(0, str(int(clamped)))
                        else:
                            entry.insert(0, str(round(clamped, 2)))
        # Display loaded history in chat
        self._restore_history_display()

    def _restore_history_display(self):
        """Replay loaded history into the chat display widget."""
        self._reset_display()
        if not self.history:
            return
        ai = self._active_ai_name()
        for msg in self.history:
            role = msg.get("role", "system")
            content = msg.get("content", "")
            if role == "user":
                self._chat_append(
                    "user_prefix",
                    f"\n  {self.user_name}  ",
                    "user", content + "\n")
                self._process_media_in_text(content, "user")
            elif role == "assistant":
                self._chat_append(
                    "assistant_prefix",
                    f"\n  {ai}  ",
                    "assistant", content + "\n")
                self._process_media_in_text(content, "assistant")
            else:
                self._chat_system(content)
        count = len(self.history)
        self._chat_system(
            f"Restored {count} messages from model context.")

    # ================================================================
    # Logic - Thinking indicator
    # ================================================================

    def _show_thinking(self):
        """Show animated processing indicator."""
        # Cancel any pending animation callback to prevent stacking
        pending = getattr(self, "_think_after_id", None)
        if pending is not None:
            try:
                self.after_cancel(pending)
            except (ValueError, Exception):
                pass
        self._thinking_active = True
        self._think_n = 0
        self._think_after_id = None
        self._do_think()

    def _do_think(self):
        if not self._thinking_active:
            self._think_after_id = None
            return
        self._think_n += 1
        # Fixed-width animation: always 3 chars (dots + spaces)
        n = self._think_n % 4
        if n == 0:
            dots = "   "  # All spaces
        else:
            dots = "." * n + " " * (3 - n)  # Dots + trailing spaces
        try:
            self.thinking_label.configure(
                text=f"  PROCESSING{dots}")
        except Exception:
            self._think_after_id = None
            return
        self._think_after_id = self.after(
            getattr(self, "_thinking_tick_ms", 350),
            self._do_think)

    def _hide_thinking(self):
        self._thinking_active = False
        pending = getattr(self, "_think_after_id", None)
        if pending is not None:
            try:
                self.after_cancel(pending)
            except (ValueError, Exception):
                pass
            self._think_after_id = None
        try:
            self.thinking_label.configure(text="")
        except Exception:
            pass

    # ================================================================
    # Logic - Typewriter effect
    # ================================================================

    def _append_stream_chunk(self, chunk: str):
        """Append one streamed text chunk to the assistant transcript."""
        if not chunk:
            return
        try:
            self.chat_display._textbox.insert("end", chunk, "assistant")
        except Exception:
            return
        if "\n" in chunk or len(chunk) >= 6:
            self._auto_resize_chat()
            self._scroll_chat_to_bottom()

    def _typewriter(self, tag: str, text: str, idx: int = 0):
        """Insert text into chat with inline media rendering.

        Processes text character-by-character, then renders any
        media references (images, GIFs, videos) and clickable
        links that are found in the text.
        Stops early if _stop_requested is set.
        """
        # Bail out if the window is being destroyed
        if getattr(self, '_shutting_down', False):
            return
        # Stop early if user cancelled
        if getattr(self, '_stop_requested', False):
            # Insert what we have so far, then stop
            self._process_media_in_text(text[:idx], tag)
            self._auto_resize_chat()
            self._scroll_chat_to_bottom()
            return
        if idx >= len(text):
            # After typewriter finishes, process media in full text
            self._process_media_in_text(text, tag)
            self._auto_resize_chat()
            self._scroll_chat_to_bottom()
            
            # Now show deferred command output and images
            # (so they appear AFTER the AI response, not during typing)
            co = getattr(self, "_deferred_cmd_output", "")
            imgs = getattr(self, "_deferred_cmd_images", [])
            files = getattr(self, "_deferred_cmd_files", [])
            ai = getattr(self, "_deferred_ai_name", "ENIGMA")
            
            if co:
                self._chat_system(co)
            for img_path in imgs:
                self._insert_media(img_path, "file")
            for file_path in files:
                self._insert_file_link(file_path)
            
            self._cmd_activity(
                "info",
                f"[{ai}] Response delivered "
                f"({len(text)} chars)\n")
            self._cmd_activity("divider", "\n")
            self._update_token_counter()
            
            # Clear deferred output for next response
            self._deferred_cmd_output = ""
            self._deferred_cmd_images = []
            self._deferred_cmd_files = []
            self._deferred_ai_name = ""
            return
        end = min(idx + 3, len(text))
        try:
            self.chat_display._textbox.insert(
                "end", text[idx:end], tag)
        except Exception:
            # Widget destroyed before typewriter finished — stop silently
            return
        # Throttle resize: only update every ~50 chars or on newlines
        # to avoid performance issues during rapid typewriter ticks
        chunk = text[idx:end]
        if '\n' in chunk or idx % 48 < 3:
            self._auto_resize_chat()
            # Scroll to bottom only when we resize (amortises see() calls)
            self._scroll_chat_to_bottom()
        self.after(
            getattr(self, "_typewriter_tick_ms", 8),
            lambda: self._typewriter(tag, text, end))
    # ================================================================
    # Logic - File attachment
    # ================================================================

    def _attach_file(self):
        from enigma_engine.gui.media import (
            IMAGE_EXTENSIONS, GIF_EXTENSIONS, VIDEO_EXTENSIONS,
        )
        path = filedialog.askopenfilename(
            title="Attach File",
            filetypes=[
                ("Text files", "*.txt *.md *.py *.json *.csv *.log"),
                ("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff"),
                ("GIFs", "*.gif"),
                ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*"),
            ])
        if not path:
            return
        ext = Path(path).suffix.lower()
        name = Path(path).name
        # Media files render inline immediately
        if ext in IMAGE_EXTENSIONS or ext in GIF_EXTENSIONS or \
                ext in VIDEO_EXTENSIONS:
            self._attach_image(path)
            return
        # Text files attach for next SEND
        self.attached_file = path
        self.file_indicator.configure(
            text=f"\U0001f4ce {name}  [click SEND to include]")

    def _clear_attachment(self):
        self.attached_file = None
        self.file_indicator.configure(text="")
    # ================================================================
    # Logic - History / Sessions (unified)
    # ================================================================

    def _refresh_history_list(self):
        """Rebuild the sidebar history list from memory/ files.

        The active session gets a filled circle ● marker,
        inactive sessions get an empty circle ○.
        """
        sessions = scan_sessions()
        # Store session data for click-to-load and delete
        self._sessions_data = sessions
        current = getattr(self, "_current_session_path", "")
        self.history_list.configure(state="normal")
        self.history_list.delete("1.0", "end")
        if not sessions:
            self.history_list.insert("1.0", "// No saved sessions\n")
        else:
            for i, s in enumerate(sessions):
                ts = ""
                if s["saved_at"]:
                    ts = time.strftime(
                        " %m/%d %H:%M",
                        time.localtime(s["saved_at"]))
                # Active session gets filled marker
                is_active = (
                    os.path.normpath(s["path"])
                    == os.path.normpath(current)
                    if current else False)
                marker = "\u25cf" if is_active else "\u25cb"
                line = (
                    f"{marker} {s['name']}"
                    f" ({s['messages']} msgs){ts}\n")
                self.history_list.insert("end", line)
                # Tag each line for click detection
                line_num = i + 1
                tag = f"session_{i}"
                # Active session uses accent color
                fg = C_ACCENT if is_active else "#b0b0b0"
                self.history_list._textbox.tag_add(
                    tag,
                    f"{line_num}.0",
                    f"{line_num}.end")
                self.history_list._textbox.tag_configure(
                    tag, foreground=fg)
                # Hover effect
                self.history_list._textbox.tag_bind(
                    tag, "<Enter>",
                    lambda e, t=tag: (
                        self.history_list._textbox.tag_configure(
                            t, foreground="#e8e8e8",
                            underline=True)))
                self.history_list._textbox.tag_bind(
                    tag, "<Leave>",
                    lambda e, t=tag, c=fg: (
                        self.history_list._textbox.tag_configure(
                            t, foreground=c,
                            underline=False)))
        self.history_list.configure(state="disabled")

    def _on_history_click(self, event):
        """Handle click on a session in the history list."""
        sessions = getattr(self, "_sessions_data", [])
        if not sessions:
            return
        # Get the line number that was clicked
        tb = self.history_list._textbox
        index = tb.index(f"@{event.x},{event.y}")
        line = int(index.split(".")[0]) - 1
        if 0 <= line < len(sessions):
            self._selected_session_index = line
            self._load_session_by_path(sessions[line]["path"])

    def _load_session_by_path(self, path: str):
        """Load a session file directly and make it the active session."""
        try:
            data = json.loads(
                Path(path).read_text(encoding="utf-8"))
            messages = data.get("messages", [])
            self.history = messages
            # Track this as the active session
            self._current_session_path = str(path)
            self._reset_display()
            ai = self._active_ai_name()
            for msg in messages:
                role = msg.get("role", "system")
                content = msg.get("content", "")
                if role == "user":
                    self._chat_append(
                        "user_prefix",
                        f"\n  {self.user_name}  ",
                        "user", content + "\n")
                    # Make links/media clickable in loaded messages
                    self._process_media_in_text(content, "user")
                elif role == "assistant":
                    self._chat_append(
                        "assistant_prefix",
                        f"\n  {ai}  ",
                        "assistant", content + "\n")
                    self._process_media_in_text(
                        content, "assistant")
                else:
                    self._chat_system(content)
            name = data.get("name", Path(path).stem)
            self._chat_system(
                f"Loaded session: {name} "
                f"({len(messages)} messages)")
            # Sync model context to point at this session
            self._save_model_context()
            self._refresh_history_list()
        except (json.JSONDecodeError, OSError) as exc:
            self._chat_error(f"Failed to load: {exc}")

    def _delete_session(self):
        """Show inline delete confirmation for the selected session.

        If the active session is deleted, start a new chat.
        """
        sessions = getattr(self, "_sessions_data", [])
        if not sessions:
            self._chat_system("No sessions to delete.")
            return

        # Use tracked selection from _on_history_click
        line = getattr(self, "_selected_session_index", -1)

        if line < 0 or line >= len(sessions):
            # Fall back to active session
            current = getattr(self, "_current_session_path", "")
            for i, s in enumerate(sessions):
                if (current and os.path.normpath(s["path"])
                        == os.path.normpath(current)):
                    line = i
                    break

        if line < 0 or line >= len(sessions):
            self._chat_system("No session selected to delete.")
            return

        session = sessions[line]
        # Store pending delete info and show inline bar
        self._pending_delete_session = session
        row = getattr(self, "_delete_confirm_row", None)
        if row is not None:
            row.grid(
                row=3, column=0, sticky="ew", padx=4, pady=(0, 4))

    def _confirm_delete_session(self):
        """Actually delete the session after inline confirmation."""
        session = getattr(self, "_pending_delete_session", None)
        self._cancel_delete_session()
        if session is None:
            return

        path = Path(session["path"])
        name = session["name"]

        try:
            from enigma_engine.core.safe_save import unlink_with_backup
            unlink_with_backup(path)
            self._chat_system(f"Deleted session: {name}")
        except OSError as exc:
            self._chat_error(f"Failed to delete: {exc}")
            return

        # If we just deleted the active session, start fresh
        current = getattr(self, "_current_session_path", "")
        if (current
                and os.path.normpath(str(path))
                == os.path.normpath(current)):
            self._new_chat()
        else:
            self._refresh_history_list()

    def _cancel_delete_session(self):
        """Hide the inline delete confirmation bar."""
        self._pending_delete_session = None
        row = getattr(self, "_delete_confirm_row", None)
        if row is not None:
            row.grid_forget()

    def _rename_session(self):
        """Show inline rename entry for the current active session."""
        current = getattr(self, "_current_session_path", "")
        if not current or not Path(current).exists():
            self._chat_system("No active session to rename.")
            return

        # Pre-fill with the current session name
        name = Path(current).stem
        try:
            data = json.loads(
                Path(current).read_text(encoding="utf-8"))
            name = data.get("name", name)
        except (json.JSONDecodeError, OSError):
            pass

        self._rename_entry.delete(0, "end")
        self._rename_entry.insert(0, name)
        self._rename_row.grid(
            row=2, column=0, sticky="ew", padx=4, pady=(0, 4))
        self._rename_entry.focus_set()
        self._rename_entry.select_range(0, "end")

    def _confirm_rename(self):
        """Apply the inline rename and hide the entry row."""
        new_name = self._rename_entry.get().strip()
        self._cancel_rename()
        if not new_name:
            return

        current = getattr(self, "_current_session_path", "")
        if not current or not Path(current).exists():
            return

        try:
            from enigma_engine.core.safe_save import atomic_write_json
            path = Path(self._current_session_path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["name"] = new_name
            atomic_write_json(path, data)
            self._chat_system(f"Session renamed to: {new_name}")
            self._refresh_history_list()
        except (json.JSONDecodeError, OSError) as exc:
            self._chat_error(f"Failed to rename: {exc}")

    def _cancel_rename(self):
        """Hide the inline rename row."""
        try:
            self._rename_row.grid_forget()
        except Exception:
            pass

    def _generate_session_title(
        self, user_msg: str, ai_response: str,
    ):
        """Ask the AI to name this session after the first exchange.

        Runs in a background thread so chatting is not blocked.
        The generated title is written to the session file and the
        sidebar refreshes to show it.
        """
        engine = getattr(self, "engine", None)
        if engine is None or not hasattr(engine, "chat"):
            return
        session_path = getattr(self, "_current_session_path", "")
        if not session_path:
            return

        def _title_gen(eng=engine, sp=session_path,
                       um=user_msg, ar=ai_response):
            try:
                prompt = (
                    "Generate a short title (3-6 words, no quotes, "
                    "no punctuation) that summarizes this conversation. "
                    "Reply ONLY with the title, nothing else.\n\n"
                    f"User: {um[:200]}\n"
                    f"Assistant: {ar[:200]}")
                title = eng.chat(
                    prompt,
                    system_prompt=(
                        "You are a title generator. Reply with ONLY "
                        "a short title. No quotes, no explanation."),
                    max_gen=30,
                    history=[],
                )
                # Clean up — strip quotes, newlines, limit length
                title = title.strip().strip('"\'').strip()
                title = title.split("\n")[0].strip()
                if not title or len(title) > 60:
                    return
                # Write to session file
                path = Path(sp)
                if not path.exists():
                    return
                data = json.loads(
                    path.read_text(encoding="utf-8"))
                data["name"] = title
                from enigma_engine.core.safe_save import (
                    atomic_write_json)
                atomic_write_json(path, data)
                self.after(0, self._refresh_history_list)
            except Exception as exc:
                logger.debug(
                    "Session title generation failed: %s", exc)

        threading.Thread(
            target=_title_gen, daemon=True).start()

    def _auto_save_session(self):
        """Auto-save current history to the active session file.

        Writes to _current_session_path so the same file is
        updated on every exchange — no duplicate files.
        """
        if not self.history:
            return
        current = getattr(self, "_current_session_path", "")
        if not current:
            return
        try:
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            path = Path(self._current_session_path)
            # Preserve existing name if file already exists
            name = Path(current).stem
            if path.exists():
                try:
                    existing = json.loads(
                        path.read_text(encoding="utf-8"))
                    name = existing.get("name", name)
                except (json.JSONDecodeError, OSError):
                    pass
            data = {
                "name": name,
                "saved_at": time.time(),
                "message_count": len(self.history),
                "messages": self.history,
            }
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(path, data)
            # Refresh history sidebar so changes appear live
            self.after(0, self._refresh_history_list)
        except OSError as exc:
            logger.debug("Auto-save session failed: %s", exc)

    def _get_memory_mode(self) -> str:
        """Return the current memory mode from gui_settings.

        Returns one of: "automatic", "manual", "disabled".
        Defaults to "automatic" if not configured.
        """
        try:
            settings_path = DATA_DIR / "gui_settings.json"
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
                mode = settings.get("memory_mode", "automatic")
                if mode in ("automatic", "manual", "disabled"):
                    return mode
        except (json.JSONDecodeError, OSError):
            pass
        return "automatic"

    def _feed_background_trainer(self, prompt: str, response: str):
        """Feed a chat exchange to the BackgroundTrainer if enabled.

        Checks the ``learn_while_chatting`` setting in gui_settings.json.
        If True and a ModRouter with a trainer is available, sends the
        prompt/response pair as a training example with engagement score.
        """
        try:
            enabled = getattr(self, "_chat_learning_enabled", None)
            if enabled is None:
                settings_path = DATA_DIR / "gui_settings.json"
                if not settings_path.exists():
                    return
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
                enabled = settings.get("learn_while_chatting", False)
            if not enabled:
                return
            router = getattr(self, "_router", None)
            if router is None:
                return

            # Compute engagement score from current emotional state
            engagement = 1.0
            try:
                from enigma_engine.core.sentiment import (
                    compute_engagement_score)
                emotional_state = getattr(
                    self, "_emotional_state", None)
                if emotional_state:
                    engagement = compute_engagement_score(
                        emotional_state)
            except Exception:
                pass

            router.add_training_example(
                prompt, response, weight=engagement)
            logger.debug(
                "Fed chat exchange to BackgroundTrainer "
                "(%d chars prompt, %d chars response, "
                "engagement=%.2f)",
                len(prompt), len(response), engagement)
        except Exception as exc:
            logger.debug(
                "Could not feed trainer: %s", exc)

    def _export_chat(self):
        """Export current chat history as Markdown, JSON, plain text, HTML, or PDF."""
        if not self.history:
            self._chat_system("Nothing to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Chat",
            defaultextension=".md",
            filetypes=[
                ("Markdown", "*.md"),
                ("HTML", "*.html"),
                ("PDF", "*.pdf"),
                ("JSON", "*.json"),
                ("Text files", "*.txt"),
            ])
        if not path:
            return

        ext = Path(path).suffix.lower()
        ai_name = self._active_ai_name()
        user_name = getattr(self, "user_name", "You")

        if ext == ".html":
            from enigma_engine.core.chat_export import export_html
            export_html(
                self.history, path,
                title="Chat Export", ai_name=ai_name, user_name=user_name,
            )
        elif ext == ".pdf":
            from enigma_engine.core.chat_export import export_pdf
            try:
                export_pdf(
                    self.history, path,
                    title="Chat Export", ai_name=ai_name, user_name=user_name,
                )
            except ImportError as exc:
                self._chat_system(str(exc))
                return
        elif ext == ".json":
            # Structured JSON export
            import json as _json
            data = {
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "message_count": len(self.history),
                "messages": self.history,
            }
            Path(path).write_text(
                _json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8")
        elif ext == ".md":
            # Markdown export with role headers
            lines: list[str] = []
            lines.append("# Chat Export")
            lines.append(f"*Exported {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
            for msg in self.history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    lines.append(f"### {user_name}\n{content}\n")
                elif role == "assistant":
                    lines.append(f"### {ai_name}\n{content}\n")
                elif role == "system":
                    lines.append(f"### System\n*{content}*\n")
                else:
                    lines.append(f"### {role.title()}\n{content}\n")
            Path(path).write_text(
                "\n".join(lines), encoding="utf-8")
        else:
            # Plain text fallback
            lines_txt: list[str] = []
            for msg in self.history:
                role = msg.get("role", "?").upper()
                content = msg.get("content", "")
                lines_txt.append(f"[{role}]\n{content}\n")
            Path(path).write_text(
                "\n".join(lines_txt), encoding="utf-8")

        self._chat_system(f"Exported to {Path(path).name}")

    # ================================================================
    # Logic - System prompt
    # ================================================================

    def _load_system_prompt(self) -> str:
        prompts_path = DATA_DIR / "prompts.json"
        if prompts_path.exists():
            try:
                data = json.loads(
                    prompts_path.read_text(encoding="utf-8"))
                return data.get("current", {}).get(
                    "system_prompt",
                    "You are a helpful AI assistant.")
            except (json.JSONDecodeError, OSError):
                pass
        return "You are a helpful AI assistant."

    def _apply_system_prompt(self):
        prompt = self.prompt_editor.get("1.0", "end").strip()
        if not prompt:
            self._chat_system("Prompt cannot be empty.")
            return
        if self.engine and hasattr(self.engine, "system_prompt"):
            self.engine.system_prompt = prompt
        # Persist to prompts.json
        prompts_path = DATA_DIR / "prompts.json"
        try:
            if prompts_path.exists():
                data = json.loads(
                    prompts_path.read_text(encoding="utf-8"))
            else:
                data = {"current": {}, "templates": {},
                        "prompts_by_purpose": {}}
            data["current"]["system_prompt"] = prompt
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(prompts_path, data)
        except (json.JSONDecodeError, OSError):
            pass
        # Also save to per-model context
        self._save_model_context()
        self._chat_system("System prompt updated.")

    def _reset_system_prompt(self):
        default = self._load_system_prompt()
        self.prompt_editor.delete("1.0", "end")
        self.prompt_editor.insert("1.0", default)
        self._chat_system("System prompt reset to default.")
