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
        if self.engine is None:
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
                content = Path(self.attached_file).read_text(
                    encoding="utf-8", errors="replace")
                if len(content) > 4000:
                    content = content[:4000] + "\n[...truncated]"
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

        # Build full system prompt: user prompt + GUI context
        try:
            user_prompt = self.prompt_editor.get("1.0", "end").strip()
        except Exception:
            user_prompt = ""
        gui_ctx = self._build_gui_context()

        # Proactive web research: auto-search if web access is ON
        # and the query looks like it would benefit from research
        web_research_ctx = ""
        if getattr(self, "web_access", False):
            try:
                from enigma_engine.core.auto_research import (
                    auto_research, should_auto_research)
                if should_auto_research(msg):
                    web_research_ctx = auto_research(
                        msg, max_results=3)
            except Exception as exc:
                logger.debug("Auto-research failed: %s", exc)

        combined_prompt = (
            f"{user_prompt}\n\n{gui_ctx}" if user_prompt
            else gui_ctx)
        if web_research_ctx:
            combined_prompt += f"\n\n{web_research_ctx}"

        def _gen():
            try:
                kwargs: dict[str, Any] = {}
                kwargs.update(self.config_overrides)
                kwargs["system_prompt"] = combined_prompt
                kwargs["history"] = list(self.history)

                # Enable reasoning if toggle is on
                if getattr(self, 'reasoning_enabled', False):
                    kwargs["reasoning"] = True

                # Log to CMD activity
                ai = self._active_ai_name()
                self._cmd_activity(
                    "info", f"[{ai}] Processing: {msg[:80]}...\n"
                    if len(msg) > 80
                    else f"[{ai}] Processing: {msg}\n")

                try:
                    resp = self.engine.chat(full_msg, **kwargs)
                except TypeError:
                    resp = self.engine.chat(full_msg)

                # Check if user hit STOP during generation
                if getattr(self, '_stop_requested', False):
                    def _stopped():
                        self._hide_thinking()
                        self._chat_system("Generation stopped.")
                    self.after(0, _stopped)
                    return

                # Extract reasoning if present (Qwen3 outputs <think>
                # blocks by default even without the reasoning toggle)
                from enigma_engine.core.reasoning import (
                    extract_reasoning, has_reasoning,
                    strip_incomplete_think)
                thinking_text = ""
                if has_reasoning(resp):
                    thinking_text, resp = extract_reasoning(resp)
                else:
                    # Strip truncated <think> from token-limited output
                    resp = strip_incomplete_think(resp)

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

                # Track message stats in identity card
                ctx = getattr(self, "model_context", None)
                if ctx is not None:
                    ctx.increment_messages(2)

                # Extract memorable facts from user message
                try:
                    from enigma_engine.core.memory import get_memory
                    mem = get_memory()
                    new_facts = mem.extract_facts(msg)
                    if new_facts:
                        logger.info(
                            "Auto-remembered: %s",
                            ", ".join(new_facts))
                except Exception as exc:
                    logger.debug("Auto-remember failed: %s", exc)

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
                ts = time.strftime("%H:%M")
                def _show(r=clean_text, t=ts, co=cmd_output,
                          think=thinking_text,
                          imgs=cmd_images):
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
                    self._typewriter("assistant", r + "\n")
                    # Speak only the answer (not the reasoning)
                    self._tts_speak(r)
                    # Show command results inline as SYSTEM
                    if co:
                        self._chat_system(co)
                    # Render generated images inline in chat
                    for img_path in imgs:
                        self._insert_media(img_path, "file")
                    self._cmd_activity(
                        "info",
                        f"[{ai}] Response delivered "
                        f"({len(r)} chars)\n")
                    self._cmd_activity("divider", "\n")
                    self._update_token_counter()
                self.after(0, _show)
            except Exception as exc:
                def _err(e=str(exc)):
                    self._hide_thinking()
                    self._chat_error(e)
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
        # Cannot edit while AI is generating
        if getattr(self, '_is_generating', False):
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

    def _chat_append(self, *tag_text_pairs):
        """Append multiple (tag, text) pairs to chat display."""
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

    def _reset_display(self):
        """Clear the chat display widget only."""
        # Stop all GIF animations
        for anim in getattr(self, "_chat_gif_animations", []):
            anim["active"] = False
        self._chat_gif_animations = []
        self._chat_images = []
        self._chat_media_refs = {}
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
        self._reset_display()
        self.history.clear()
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
        ctx.save()

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
        self._thinking_active = True
        self._think_n = 0
        self._do_think()

    def _do_think(self):
        if not self._thinking_active:
            return
        self._think_n += 1
        dots = "." * (self._think_n % 4)
        try:
            self.thinking_label.configure(
                text=f"  PROCESSING{dots}")
        except Exception:
            return
        self.after(350, self._do_think)

    def _hide_thinking(self):
        self._thinking_active = False
        try:
            self.thinking_label.configure(text="")
        except Exception:
            pass

    # ================================================================
    # Logic - Typewriter effect
    # ================================================================

    def _typewriter(self, tag: str, text: str, idx: int = 0):
        """Insert text into chat with inline media rendering.

        Processes text character-by-character, then renders any
        media references (images, GIFs, videos) and clickable
        links that are found in the text.
        Stops early if _stop_requested is set.
        """
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
            return
        end = min(idx + 3, len(text))
        self.chat_display._textbox.insert(
            "end", text[idx:end], tag)
        # Throttle resize: only update every ~50 chars or on newlines
        # to avoid performance issues during rapid typewriter ticks
        chunk = text[idx:end]
        if '\n' in chunk or idx % 48 < 3:
            self._auto_resize_chat()
        # Always scroll to bottom for smooth, constant scrolling
        self._scroll_chat_to_bottom()
        self.after(8, lambda: self._typewriter(tag, text, end))
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
            path.unlink()
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
            path = Path(self._current_session_path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["name"] = new_name
            path.write_text(
                json.dumps(data, indent=2), encoding="utf-8")
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
                path.write_text(
                    json.dumps(data, indent=2),
                    encoding="utf-8")
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
            path.write_text(
                json.dumps(data, indent=2), encoding="utf-8")
            # Refresh history sidebar so changes appear live
            self.after(0, self._refresh_history_list)
        except OSError:
            pass  # Silent failure — auto-save is best-effort

    def _feed_background_trainer(self, prompt: str, response: str):
        """Feed a chat exchange to the BackgroundTrainer if enabled.

        Checks the ``learn_while_chatting`` setting in gui_settings.json.
        If True and a ModRouter with a trainer is available, sends the
        prompt/response pair as a training example.
        """
        try:
            settings_path = DATA_DIR / "gui_settings.json"
            if not settings_path.exists():
                return
            settings = json.loads(
                settings_path.read_text(encoding="utf-8"))
            if not settings.get("learn_while_chatting", False):
                return
            router = getattr(self, "_router", None)
            if router is None:
                return
            router.add_training_example(prompt, response)
            logger.debug(
                "Fed chat exchange to BackgroundTrainer "
                "(%d chars prompt, %d chars response)",
                len(prompt), len(response))
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
            prompts_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8")
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
