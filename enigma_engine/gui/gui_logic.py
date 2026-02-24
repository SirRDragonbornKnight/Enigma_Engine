"""
Enigma Engine - GUI Logic Methods
===================================

Mixin providing business logic for EnigmaGUI.
Handles config, model loading, chat,
history, profiles, and route assignments.

Training / model management: see gui_forge.py (ForgeMixin)
Brick process management:    see gui_bricks.py (BrickMixin)
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from tkinter import filedialog
from typing import Any

import customtkinter as ctk

from enigma_engine.gui.widgets import (
    C_ACCENT, C_ACCENT_DIM, C_GREEN, C_ORANGE,
    C_PURPLE, C_RED, C_SURFACE, C_TEXT_DIM,
)
from enigma_engine.gui.scanners import (
    CONFIG_LIMITS, DATA_DIR, MEMORY_DIR, PROFILES_DIR,
    ROUTE_KEYS, PATH_SETTINGS,
    clamp_config, scan_models, scan_sessions,
    load_path_settings, save_path_settings,
)
from enigma_engine.core.model_context import (
    ModelContext, model_key_from_path, load_model_context,
)
# Re-export so existing imports keep working
from enigma_engine.gui.gui_forge import ForgeMixin  # noqa: F401
from enigma_engine.gui.gui_bricks import BrickMixin  # noqa: F401


class LogicMixin:
    """Mixin providing logic methods for EnigmaGUI.

    Expects the host class to have:
    - engine, model_path, active_profile, config_overrides
    - history, brick_processes, training_active
    - voice_enabled, attached_file
    - route_assignments: dict mapping route keys to model paths
    - model_context: ModelContext | None for per-model storage
    - Widget references from page builders
    """

    # ================================================================
    # Logic - Config
    # ================================================================

    def _load_config_defaults(self):
        try:
            from enigma_engine.config import CONFIG
            defaults = {
                "temperature": CONFIG.get("temperature", 0.8),
                "top_p": CONFIG.get("top_p", 0.9),
                "top_k": CONFIG.get("top_k", 50),
                "max_tokens": CONFIG.get("max_gen", 100),
                "repetition_penalty": CONFIG.get(
                    "repetition_penalty", 1.1),
            }
        except ImportError:
            defaults = {
                "temperature": 0.8, "top_p": 0.9, "top_k": 50,
                "max_tokens": 100, "repetition_penalty": 1.1,
            }
        for name, val in defaults.items():
            entry = self.config_entries.get(name)
            if entry:
                entry.delete(0, "end")
                entry.insert(0, str(val))

    def _validate_config(self, name: str):
        entry = self.config_entries.get(name)
        if not entry:
            return
        text = entry.get().strip()
        try:
            val = float(text)
        except ValueError:
            val = self.config_overrides.get(name, 0.8)
        clamped = clamp_config(name, val)
        lo, hi, step = CONFIG_LIMITS[name]
        if step == int(step) and lo == int(lo):
            display = str(int(clamped))
        else:
            display = str(round(clamped, 2))
        entry.delete(0, "end")
        entry.insert(0, display)
        self.config_overrides[name] = clamped

    # ================================================================
    # Logic - Model loading
    # ================================================================

    def _load_model(self, path: str):
        if self.engine is not None:
            self._unload_model()

        self._set_header_status("LOADING...", C_ORANGE)
        self.header_dot.set_color(C_ORANGE)
        self._chat_system(f"Loading {Path(path).name}...")
        self.send_btn.configure(state="disabled")

        def _load():
            try:
                from enigma_engine.core import EnigmaEngine
                self.engine = EnigmaEngine(model_path=path)
                self.model_path = path
                pc = 0
                if (hasattr(self.engine, "model")
                        and self.engine.model is not None):
                    pc = sum(
                        p.numel()
                        for p in self.engine.model.parameters())
                self.after(0, lambda: self._on_model_loaded(path, pc))
            except Exception as exc:
                self.after(0, lambda: self._on_model_error(str(exc)))

        threading.Thread(target=_load, daemon=True).start()

    def _on_model_loaded(self, path: str, param_count: int):
        device = "CPU"
        try:
            import torch
            device = "CUDA" if torch.cuda.is_available() else "CPU"
        except ImportError:
            pass

        name = Path(path).stem
        self._set_header_status(
            f"{name.upper()} // {device}", C_GREEN)
        self.header_dot.set_color(C_GREEN)
        self.send_btn.configure(state="normal")
        self.unload_btn.configure(state="normal")
        self.status_bar.set_left(f"\u26a1 {name.upper()} LOADED")

        # Load AI display name from model folder
        self._load_model_display_name(path)

        # Load per-model context (history + prompt)
        self._load_model_context(path)
        self._chat_system(
            f"Model online: {param_count:,} params on {device}")

        # Track the loaded model in chat route assignment
        self.route_assignments["chat"] = path

        # Update the chat route dropdown to show the loaded model
        route_menus = getattr(self, "_route_menus", {})
        chat_menu = route_menus.get("chat")
        if chat_menu:
            # Find the display name matching this path
            display = "None"
            for m in self.models_data:
                if m["path"] == path:
                    display = m["name"]
                    break
            chat_menu.set(display)

        self._update_route_status()

    def _on_model_error(self, error: str):
        self._set_header_status("LOAD FAILED", C_RED)
        self.header_dot.set_color(C_RED)
        self._chat_error(f"Failed to load model: {error}")
        self.send_btn.configure(state="normal")

    def _load_model_display_name(self, path: str):
        """Read AI display name from model_info.json in the model folder.

        Looks for a model_info.json file in the same directory as the
        model file. If found and it has a "display_name" field, that
        name is used for the AI in chat. Otherwise falls back to
        self.ai_name (default "ENIGMA").
        """
        self._model_display_name = None
        model_dir = Path(path).parent
        info_path = model_dir / "model_info.json"
        if info_path.exists():
            try:
                data = json.loads(
                    info_path.read_text(encoding="utf-8"))
                name = data.get("display_name", "").strip()
                if name:
                    self._model_display_name = name
            except (json.JSONDecodeError, OSError):
                pass

    def _active_ai_name(self) -> str:
        """Return the current AI display name for chat messages.

        Priority: model_info.json display_name > self.ai_name
        """
        model_name = getattr(self, "_model_display_name", None)
        if model_name:
            return model_name
        return getattr(self, "ai_name", "ENIGMA")

    def _unload_model(self):
        # Save per-model context before unloading
        self._save_model_context()
        self._model_display_name = None
        if self.engine is not None:
            del self.engine
            self.engine = None
            self.model_path = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        self._set_header_status("NO MODEL", C_TEXT_DIM)
        self.header_dot.set_color(C_TEXT_DIM)
        self.unload_btn.configure(state="disabled")
        self._chat_system("Model unloaded.")
        self.status_bar.set_left("\u26a1 READY")
        # Clear chat route assignment
        self.route_assignments.pop("chat", None)
        route_menus = getattr(self, "_route_menus", {})
        chat_menu = route_menus.get("chat")
        if chat_menu:
            chat_menu.set("None")
        self._update_route_status()

    # ================================================================
    # Logic - Chat
    # ================================================================

    def _on_input_enter(self, event):
        """Enter sends message. Shift+Enter adds newline."""
        if event.state & 0x1:
            return
        self._send_message()
        return "break"

    def _send_message(self):
        msg = self.chat_input.get("1.0", "end").strip()
        if not msg:
            return
        if self.engine is None:
            self._chat_system(
                "No model loaded. Go to ROUTER and load one first.")
            return

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
        timestamp = time.strftime("%H:%M")
        self._chat_append(
            "timestamp", f"\n  {timestamp} ",
            "user_prefix", f"{self.user_name}  ",
            "user", msg + "\n")
        if file_context:
            self._chat_append("file_tag", "  [file attached]\n")
        self.send_btn.configure(state="disabled")
        self._show_thinking()

        full_msg = msg + file_context

        def _gen():
            try:
                kwargs: dict[str, Any] = {}
                kwargs.update(self.config_overrides)
                try:
                    resp = self.engine.chat(full_msg, **kwargs)
                except TypeError:
                    resp = self.engine.chat(full_msg)
                self.history.append(
                    {"role": "user", "content": msg})
                self.history.append(
                    {"role": "assistant", "content": resp})
                # Auto-save per-model context after each exchange
                self._save_model_context()
                ts = time.strftime("%H:%M")
                def _show(r=resp, t=ts):
                    self._hide_thinking()
                    ai = self._active_ai_name()
                    self._chat_append(
                        "timestamp", f"\n  {t} ",
                        "assistant_prefix", f"{ai}  ")
                    self._typewriter("assistant", r + "\n")
                self.after(0, _show)
            except Exception as exc:
                def _err(e=str(exc)):
                    self._hide_thinking()
                    self._chat_error(e)
                self.after(0, _err)
            finally:
                self.after(0, lambda: self.send_btn.configure(
                    state="normal"))

        threading.Thread(target=_gen, daemon=True).start()

    def _chat_append(self, *tag_text_pairs):
        """Append multiple (tag, text) pairs to chat display."""
        self.chat_display.configure(state="normal")
        tb = self.chat_display._textbox
        for i in range(0, len(tag_text_pairs), 2):
            tag = tag_text_pairs[i]
            text = tag_text_pairs[i + 1]
            tb.insert("end", text, tag)
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def _chat_system(self, text: str):
        self._chat_append("system", f"\n  // {text}\n")

    def _chat_error(self, text: str):
        self._chat_append("error", f"\n  [!] {text}\n")

    def _reset_display(self):
        """Clear the chat display widget only."""
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")

    def _new_chat(self):
        """Start a fresh conversation - clear chat and reset AI state."""
        self._reset_display()
        self.history.clear()
        # Save cleared history to model context
        self._save_model_context()
        if self.engine:
            if hasattr(self.engine, "clear_history"):
                self.engine.clear_history()
            if hasattr(self.engine, "clear_kv_cache"):
                self.engine.clear_kv_cache()
        self._chat_system("New conversation started.")

    # ================================================================
    # Logic - Per-model context
    # ================================================================

    def _save_model_context(self):
        """Save current history and prompt to the loaded model's context."""
        ctx = getattr(self, "model_context", None)
        if ctx is None:
            return
        ctx.history = list(self.history)
        # Capture current system prompt
        try:
            prompt = self.prompt_editor.get("1.0", "end").strip()
            if prompt:
                ctx.system_prompt = prompt
        except Exception:
            pass
        # Capture config overrides
        ctx.config = dict(self.config_overrides)
        ctx.profile_id = self.active_profile or ""
        ctx.save()

    def _load_model_context(self, model_path: str):
        """Load per-model context when a model is loaded."""
        ctx = load_model_context(model_path)
        self.model_context = ctx
        # Restore history
        self.history = list(ctx.history)
        # Restore system prompt into editor
        if ctx.system_prompt:
            try:
                self.prompt_editor.delete("1.0", "end")
                self.prompt_editor.insert("1.0", ctx.system_prompt)
                if (self.engine
                        and hasattr(self.engine, "system_prompt")):
                    self.engine.system_prompt = ctx.system_prompt
            except Exception:
                pass
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
        # Restore profile
        if ctx.profile_id:
            self.active_profile = ctx.profile_id
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
                    "user_prefix", f"\n  {self.user_name}  ",
                    "user", content + "\n")
            elif role == "assistant":
                self._chat_append(
                    "assistant_prefix", f"\n  {ai}  ",
                    "assistant", content + "\n")
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
        """Insert text into chat character-by-character."""
        if idx >= len(text):
            return
        end = min(idx + 3, len(text))
        self.chat_display.configure(state="normal")
        self.chat_display._textbox.insert(
            "end", text[idx:end], tag)
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")
        self.after(8, lambda: self._typewriter(tag, text, end))

    # ================================================================
    # Logic - Voice toggle
    # ================================================================

    def _on_voice_toggle(self, is_on: bool):
        self.voice_enabled = is_on
        state = "ON" if is_on else "OFF"
        self._chat_system(f"Voice output {state}")

    # ================================================================
    # Logic - Voice input (speech-to-text)
    # ================================================================

    def _toggle_voice_input(self):
        """Start or stop voice input recording.

        Uses listen_in_background so the user can click again
        to immediately stop recording instead of waiting for
        a timeout.
        """
        # If already recording, stop it
        if getattr(self, "_voice_recording", False):
            self._voice_recording = False
            stopper = getattr(self, "_voice_stopper", None)
            if stopper:
                stopper(wait_for_stop=False)
                self._voice_stopper = None
            self._voice_input_done()
            self._chat_system("Recording stopped.")
            return

        try:
            import speech_recognition as sr
        except ImportError:
            self._chat_error(
                "speech_recognition not installed. "
                "Run: pip install SpeechRecognition")
            return

        self._voice_recording = True
        self._voice_got_audio = False
        self.mic_btn.configure(
            fg_color=C_RED, text_color="#ffffff")
        self._chat_system("Listening... click mic again to stop.")

        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True

        def _on_audio(rec, audio):
            """Called in background when speech is detected."""
            if not self._voice_recording or self._voice_got_audio:
                return
            self._voice_got_audio = True
            # Stop the background listener
            stopper = getattr(self, "_voice_stopper", None)
            if stopper:
                stopper(wait_for_stop=False)
                self._voice_stopper = None
            try:
                text = rec.recognize_google(audio)
                self.after(0, lambda t=text: self._on_voice_text(t))
            except Exception as exc:
                err = str(exc) if str(exc) else type(exc).__name__
                self.after(0, lambda e=err: self._chat_error(
                    f"Voice input failed: {e}"))
            finally:
                self._voice_recording = False
                self.after(0, self._voice_input_done)

        try:
            mic = sr.Microphone()
            # Store the stopper so clicking mic again can cancel
            self._voice_stopper = recognizer.listen_in_background(
                mic, _on_audio, phrase_time_limit=30)
        except Exception as exc:
            self._voice_recording = False
            self._voice_input_done()
            err = str(exc) if str(exc) else type(exc).__name__
            self._chat_error(f"Microphone error: {err}")

    def _on_voice_text(self, text: str):
        """Insert transcribed text into the chat input."""
        if not text:
            return
        current = self.chat_input.get("1.0", "end").strip()
        if current:
            self.chat_input.insert("end", " " + text)
        else:
            self.chat_input.delete("1.0", "end")
            self.chat_input.insert("1.0", text)
        self._chat_system(f"Voice: \"{text}\"")

    def _voice_input_done(self):
        """Reset mic button after recording ends."""
        try:
            self.mic_btn.configure(
                fg_color=C_SURFACE, text_color=C_TEXT_DIM)
        except Exception:
            pass

    # ================================================================
    # Logic - File attachment
    # ================================================================

    def _attach_file(self):
        path = filedialog.askopenfilename(
            title="Attach File",
            filetypes=[
                ("Text files", "*.txt *.md *.py *.json *.csv *.log"),
                ("All files", "*.*"),
            ])
        if path:
            self.attached_file = path
            name = Path(path).name
            self.file_indicator.configure(
                text=f"\U0001f4ce {name}  [click SEND to include]")

    def _clear_attachment(self):
        self.attached_file = None
        self.file_indicator.configure(text="")

    # ================================================================
    # Logic - History / Sessions
    # ================================================================

    def _refresh_history_list(self):
        sessions = scan_sessions()
        self.history_list.configure(state="normal")
        self.history_list.delete("1.0", "end")
        if not sessions:
            self.history_list.insert("1.0", "// No saved sessions\n")
        else:
            for s in sessions:
                ts = ""
                if s["saved_at"]:
                    ts = time.strftime(
                        " %m/%d %H:%M",
                        time.localtime(s["saved_at"]))
                self.history_list.insert(
                    "end",
                    f"\u25cb {s['name']}"
                    f" ({s['messages']} msgs){ts}\n")
        self.history_list.configure(state="disabled")

    def _save_session(self):
        if not self.history:
            self._chat_system("Nothing to save (chat is empty).")
            return
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        name = time.strftime("session_%Y%m%d_%H%M%S")
        path = MEMORY_DIR / f"{name}.json"
        data = {
            "name": name,
            "saved_at": time.time(),
            "message_count": len(self.history),
            "messages": self.history,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        # Also persist to per-model context
        self._save_model_context()
        self._chat_system(f"Session saved: {name}")
        self._refresh_history_list()

    def _load_session(self):
        path = filedialog.askopenfilename(
            title="Load Session",
            initialdir=str(MEMORY_DIR),
            filetypes=[("JSON files", "*.json")])
        if not path:
            return
        try:
            data = json.loads(
                Path(path).read_text(encoding="utf-8"))
            messages = data.get("messages", [])
            self.history = messages
            self._reset_display()
            for msg in messages:
                role = msg.get("role", "system")
                content = msg.get("content", "")
                if role == "user":
                    self._chat_append(
                        "user_prefix", "\n  YOU  ",
                        "user", content + "\n")
                elif role == "assistant":
                    self._chat_append(
                        "assistant_prefix", "\n  ENIGMA  ",
                        "assistant", content + "\n")
                else:
                    self._chat_system(content)
            name = data.get("name", Path(path).stem)
            self._chat_system(
                f"Loaded session: {name} "
                f"({len(messages)} messages)")
        except (json.JSONDecodeError, OSError) as exc:
            self._chat_error(f"Failed to load: {exc}")

    def _export_chat(self):
        if not self.history:
            self._chat_system("Nothing to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Chat",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")])
        if not path:
            return
        lines = []
        for msg in self.history:
            role = msg.get("role", "?").upper()
            content = msg.get("content", "")
            lines.append(f"[{role}]\n{content}\n")
        Path(path).write_text(
            "\n".join(lines), encoding="utf-8")
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

    # ================================================================
    # Logic - Profiles
    # ================================================================

    def _on_profile_selected(self, choice: str):
        for p in self.profiles_data:
            if p["name"] == choice:
                self._activate_profile(p)
                return

    def _activate_profile(self, profile: dict):
        profile_id = profile["id"]
        path = PROFILES_DIR / f"{profile_id}.json"
        if not path.exists():
            self._chat_error(f"Profile missing: {profile_id}")
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        self.active_profile = profile_id

        gen = data.get("generation", {})
        for key, val in gen.items():
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

        sys_prompt = data.get("system_prompt", "")
        if sys_prompt:
            self.prompt_editor.delete("1.0", "end")
            self.prompt_editor.insert("1.0", sys_prompt)
            if self.engine and hasattr(self.engine, "system_prompt"):
                self.engine.system_prompt = sys_prompt

        name = data.get("name", profile_id)
        self.active_profile_label.configure(
            text=name, text_color=C_PURPLE)
        # Update CORE page profile indicator
        if hasattr(self, "_core_profile_label"):
            self._core_profile_label.configure(
                text=f"PROFILE: {name}")
        self._chat_system(f"Profile activated: {name}")

    # ================================================================
    # Logic - Route assignments
    # ================================================================

    def _assign_model_to_route(self, route_key: str, choice: str):
        """Assign a model to a specific route.

        Args:
            route_key: Route identifier (e.g. 'chat', 'trainer', brick name).
            choice: Model name from dropdown, or 'None' to clear.
        """
        if choice == "None":
            self._unassign_route(route_key)
            return

        # Find model path from name
        model_path = None
        for m in self.models_data:
            if m["name"] == choice:
                model_path = m["path"]
                break
        if not model_path:
            self._chat_system(f"Model '{choice}' not found.")
            return

        self.route_assignments[route_key] = model_path

        # If assigning to chat, load the model into the engine
        if route_key == "chat":
            self._load_model(model_path)
        else:
            self._chat_system(
                f"Route {route_key.upper()} assigned: {choice}")

        self._update_route_status()

    def _unassign_route(self, route_key: str):
        """Remove model assignment from a route."""
        self.route_assignments.pop(route_key, None)

        if route_key == "chat" and self.engine is not None:
            self._unload_model()
        else:
            self._chat_system(
                f"Route {route_key.upper()} cleared.")

        self._update_route_status()

    def _update_route_status(self):
        """Update route connection labels with current model state."""
        route_labels = getattr(self, "_route_labels", {})
        if not route_labels:
            return

        # Update built-in routes (chat, trainer) based on assignments
        for key in ROUTE_KEYS:
            ref = route_labels.get(key)
            if not ref:
                continue
            dot, lbl = ref
            assigned = self.route_assignments.get(key)
            if assigned:
                name = Path(assigned).stem
                dot.set_color(C_GREEN)
                lbl.configure(text=name, text_color=C_GREEN)
            else:
                dot.set_color(C_TEXT_DIM)
                lbl.configure(
                    text="No model", text_color=C_TEXT_DIM)

        # Update brick routes
        for brick in self.bricks_data:
            brick_key = brick["name"].lower()
            ref = route_labels.get(brick_key)
            if not ref:
                continue
            dot, lbl = ref
            assigned = self.route_assignments.get(brick_key)
            running = brick.get("_running", False)
            if assigned:
                name = Path(assigned).stem
                status = f"{name}"
                if running:
                    status += " (running)"
                dot.set_color(C_GREEN if running else C_ORANGE)
                lbl.configure(text=status,
                              text_color=C_GREEN if running
                              else C_ORANGE)
            elif running:
                dot.set_color(C_GREEN)
                lbl.configure(
                    text="Running", text_color=C_GREEN)
            else:
                dot.set_color(C_TEXT_DIM)
                lbl.configure(
                    text="Stopped", text_color=C_TEXT_DIM)

    # ================================================================
    # Helpers
    # ================================================================

    def _set_header_status(self, text: str, color: str):
        self.header_status.configure(text=text, text_color=color)

    # ================================================================
    # Logic - Directory path settings
    # ================================================================

    def _browse_path(self, key: str):
        """Open a directory picker for a path setting."""
        from tkinter import filedialog as fd
        entry = self.path_entries.get(key)
        if not entry:
            return
        current = entry.get().strip()
        chosen = fd.askdirectory(
            title=f"Select {PATH_SETTINGS[key][0]}",
            initialdir=current if current else None)
        if chosen:
            entry.delete(0, "end")
            entry.insert(0, chosen)

    def _save_paths(self):
        """Save directory path overrides from the CONFIG page."""
        paths = {}
        for key, entry in self.path_entries.items():
            val = entry.get().strip()
            if val:
                paths[key] = val
        save_path_settings(paths)
        self._chat_system(
            "Paths saved. Changes take effect on next launch.")

    def _reset_paths(self):
        """Reset all path entries to their defaults."""
        for key, entry in self.path_entries.items():
            _, default = PATH_SETTINGS.get(key, ("", ""))
            entry.delete(0, "end")
            entry.insert(0, str(default))
        self._chat_system("Paths reset to defaults.")

    def _load_path_settings(self):
        """Load saved path overrides into the path entries."""
        overrides = load_path_settings()
        for key, val in overrides.items():
            entry = self.path_entries.get(key)
            if entry and val:
                entry.delete(0, "end")
                entry.insert(0, val)

    # ================================================================
    # Logic - Display names
    # ================================================================

    def _save_display_names(self):
        """Save user and AI display names from CONFIG entries."""
        user = self._user_name_entry.get().strip()
        ai = self._ai_name_entry.get().strip()
        if user:
            self.user_name = user
        if ai:
            self.ai_name = ai
        # Persist to gui_settings.json
        settings_path = DATA_DIR / "gui_settings.json"
        data = {}
        if settings_path.exists():
            try:
                data = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        data["user_display_name"] = self.user_name
        data["ai_display_name"] = self.ai_name
        try:
            settings_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass
        self._chat_system(
            f"Names saved: {self.user_name} / {self.ai_name}")

    def _reset_display_names(self):
        """Reset display names to defaults."""
        self.user_name = "YOU"
        self.ai_name = "ENIGMA"
        self._user_name_entry.delete(0, "end")
        self._user_name_entry.insert(0, "YOU")
        self._ai_name_entry.delete(0, "end")
        self._ai_name_entry.insert(0, "ENIGMA")
        self._chat_system("Names reset to defaults.")

    def _load_display_names(self):
        """Load saved display names from gui_settings.json."""
        settings_path = DATA_DIR / "gui_settings.json"
        if not settings_path.exists():
            return
        try:
            data = json.loads(
                settings_path.read_text(encoding="utf-8"))
            user = data.get("user_display_name", "").strip()
            ai = data.get("ai_display_name", "").strip()
            if user:
                self.user_name = user
            if ai:
                self.ai_name = ai
        except (json.JSONDecodeError, OSError):
            pass

    # ================================================================
    # Logic - Web search
    # ================================================================

    def _web_search_dialog(self):
        """Open a dialog to search the web and insert results into chat."""
        dialog = ctk.CTkInputDialog(
            text="Enter your search query:",
            title="Web Search")
        query = dialog.get_input()
        if not query or not query.strip():
            return
        query = query.strip()
        self._chat_system(f"Searching the web for: {query}")

        def _search():
            try:
                from enigma_engine.core.commands import get_registry
                registry = get_registry()
                result = registry.execute(f"search.web {query}")
                def _show():
                    if result.success:
                        self._chat_append(
                            "system",
                            f"\n  // Web results:\n")
                        self._chat_append(
                            "assistant", result.output + "\n")
                    else:
                        self._chat_error(result.output)
                self.after(0, _show)
            except Exception as exc:
                self.after(
                    0, lambda: self._chat_error(str(exc)))

        import threading
        threading.Thread(target=_search, daemon=True).start()

    def destroy(self):
        self.training_active = False
        for proc in self.brick_processes.values():
            if proc.poll() is None:
                proc.terminate()
        super().destroy()
