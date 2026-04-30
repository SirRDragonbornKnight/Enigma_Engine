"""
Enigma Engine - Desktop Interface
===================================

Desktop app built with CustomTkinter.
Minimal processing, clean layout.

Pages:
  CORE   - Chat with history sidebar, prompts, voice, file attach
  ROUTER - Browse models, assign to chat/trainer/mods
  FORGE  - Train models and tokenizers
  CONFIG - Generation params, customization

Usage:
    python run.py --gui
    python run.py --gui --model models/enigma_small.pth
"""
from __future__ import annotations

import ctypes
import logging
import os
import platform
import subprocess
import threading
import time
from typing import Any

import customtkinter as ctk

from enigma_engine.gui.widgets import (
    C_ACCENT, C_BG, C_BORDER, C_BORDER_ACCENT,
    C_PANEL, C_SURFACE, C_TEXT, C_TEXT_BRIGHT,
    C_TEXT_DIM, VERSION,
    FONT_SECTION, FONT_SMALL, FONT_TINY, FONT_TITLE,
    NavButton, SelectableLabel, StatusBar, StatusDot, Tooltip,
    themed_button,
)
from enigma_engine.gui.gui_pages import PagesMixin
from enigma_engine.gui.gui_logic import LogicMixin
from enigma_engine.gui.gui_forge import ForgeMixin
from enigma_engine.gui.gui_mods import ModMixin
from enigma_engine.gui.gui_mod_page import ModPageMixin
from enigma_engine.gui.gui_cmd_page import CMDPageMixin
from enigma_engine.gui.gui_docs_page import DocsPageMixin

# Re-exports (tests import these from enigma_engine.gui.desktop)
from enigma_engine.gui.scanners import (  # noqa: F401
    CONFIG_DESCRIPTIONS, CONFIG_LIMITS, ROUTE_KEYS,
    MODS_DIR, DATA_DIR, INFO_DIR, MEMORY_DIR, MODELS_DIR,
    PROJECT_ROOT, SESSIONS_DIR,
    clamp_config,
    scan_mods, scan_docs, scan_models,
    scan_sessions, scan_training_data,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Main GUI
# -------------------------------------------------------------------

class EnigmaGUI(
        DocsPageMixin, ForgeMixin, ModMixin, ModPageMixin,
        CMDPageMixin, LogicMixin, PagesMixin, ctk.CTk):
    """Enigma Engine desktop interface.

    Combines mixins via multiple inheritance:
    - PagesMixin:      UI page construction (CORE, MODELS, ROUTER, FORGE, CONFIG)
    - DocsPageMixin:   Documentation browser, file management
    - ModPageMixin:    Per-mod page construction
    - CMDPageMixin:    Command terminal page
    - LogicMixin:      Chat, sessions, routes
    - ForgeMixin:      Training, model create/delete
    - ModMixin:        Mod subprocess management
    All mixins access shared state through self.* attributes set here.
    """

    def __init__(self, model_path: str | None = None):
        super().__init__()

        self.title("ENIGMA ENGINE")
        self.geometry("1440x900")
        self.minsize(800, 500)
        self._restore_window_geometry()
        self.configure(fg_color=C_BG)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Engine state
        self.engine = None
        self.model_path = model_path
        self.config_overrides: dict[str, Any] = {}
        self.history: list[dict[str, str]] = []
        self.mod_processes: dict[str, subprocess.Popen] = {}
        self._mod_lock = threading.Lock()
        self.training_active = False
        self.voice_enabled = False
        self.web_access = False
        self.reasoning_enabled = False
        self.attached_file: str | None = None
        self._boot_time = time.time()
        self._thinking_active = False
        self._model_loading = False
        self._chat_fullscreen = False
        self._pinned = False
        self._is_generating = False
        self._stop_requested = False
        # Deferred output: show command results AFTER response finishes typing
        self._deferred_cmd_output = ""
        self._deferred_cmd_images: list = []
        self._deferred_cmd_files: list = []
        self._deferred_ai_name = ""
        self._chat_file_refs: dict[str, str] = {}
        self._chat_images: list = []
        self._chat_gif_animations: list = []
        self._tts_engine_ref = None
        self._tts_queue = None
        self._tts_alive = False
        self._tts_stop_event = threading.Event()
        self._voice_recording = False
        self._voice_stopper = None
        self._shutting_down = False
        self._resize_timer = None
        self._resizing = False
        self._model_suspended_by_minimize = False
        self._suspended_model_path: str | None = None

        # Active session file path (unified history)
        self._current_session_path: str = ""
        # Create initial session so first chat is auto-saved
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self._session_counter = 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._current_session_path = str(
            MEMORY_DIR / f"session_{ts}_1.json")

        # Display names (loaded from gui_settings.json)
        self.user_name = "YOU"
        self.ai_name = "ENIGMA"

        # Per-model context (history + prompt persistence)
        self.model_context = None

        # Per-route model assignments: route_key -> model path
        self.route_assignments: dict[str, str | None] = {}

        # Scan filesystem
        self.mods_data = scan_mods()
        self.models_data = scan_models()
        self.training_files = scan_training_data()

        # Performance defaults loaded from gui_settings.json
        self._auto_load_chat_model = self._read_gui_bool_setting(
            "auto_load_chat_model", True)
        self._auto_start_mods = self._read_gui_bool_setting(
            "auto_start_mods", True)
        self._auto_unload_on_minimize = self._read_gui_bool_setting(
            "auto_unload_on_minimize", False)
        self._chat_learning_enabled = self._read_gui_bool_setting(
            "learn_while_chatting", False)
        self._refresh_performance_mode()

        # Build UI
        self._pages: dict[str, ctk.CTkFrame] = {}
        self._nav_buttons: dict[str, NavButton] = {}
        self._current_page = ""
        self._build_shell()
        self._build_page_core()
        self._build_page_models()
        self._build_page_router()
        self._build_page_forge()
        self._build_page_config()
        self._build_page_docs()
        self._build_page_cmd()
        for mod in self.mods_data:
            mod["_running"] = False
            self._build_page_mod(mod)
        self._switch_page("CORE")
        self._load_config_defaults()
        self._load_display_names()
        self._load_route_assignments()
        self._start_status_ticker()

        # Start mod router for mod communication
        self._router = None
        try:
            from enigma_engine.router import ModRouter
            # Continuous-3b (Pass 156o): forward saved anchor override
            # to the router so the BackgroundTrainer rehearsal layer
            # uses the user's chosen file (or repo default when blank).
            from enigma_engine.gui.scanners import _resolve_anchor_path
            _saved_anchor = self._read_gui_str_setting(
                "anchor_data_path", "")
            _anchor_path = _resolve_anchor_path(_saved_anchor)
            self._router = ModRouter(
                enable_training=self._chat_learning_enabled,
                anchor_data_path=_anchor_path)
            self._router.set_inference_idle_check(
                lambda: not bool(getattr(self, "_is_generating", False))
            )
            self._router.start()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Router startup failed (optional): {e}")

        # Auto-start mods only when enabled in settings.
        if self._auto_start_mods:
            for mod in self.mods_data:
                self._auto_start_mod(mod)

        # Register mod management and tool commands
        self._register_gui_commands()

        # Count model params in background (deferred from scan)
        self.after(200, self._count_model_params_background)

        # Auto-load model if given
        if self.model_path:
            self.after(300, lambda: self._load_model(self.model_path))

        # Make all label text copyable via right-click
        self.after(500, self._enable_label_copy)

        # Escape key stops active generation
        self._bind_escape_stop()

        # Page navigation shortcuts Ctrl+1..7
        self._bind_page_nav_shortcuts()

        # Cleanup on window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Bind resize event handler to reduce lag
        self.bind("<Configure>", self._on_window_resize)
        self.bind("<Unmap>", self._on_window_unmap, add="+")
        self.bind("<Map>", self._on_window_map, add="+")

        # Watchdog: exit if parent process (terminal) dies
        self._start_parent_watchdog()

    def _read_gui_bool_setting(self, key: str, default: bool) -> bool:
        """Read a boolean setting from gui_settings.json."""
        import json

        settings_path = DATA_DIR / "gui_settings.json"
        if not settings_path.exists():
            return default
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            value = data.get(key, default)
            if isinstance(value, bool):
                return value
        except Exception as exc:
            logger.debug("Could not read setting %s: %s", key, exc)
        return default

    def _read_gui_str_setting(self, key: str, default: str) -> str:
        """Read a string setting from gui_settings.json."""
        import json

        settings_path = DATA_DIR / "gui_settings.json"
        if not settings_path.exists():
            return default
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            value = data.get(key, default)
            if isinstance(value, str):
                return value
        except Exception as exc:
            logger.debug("Could not read setting %s: %s", key, exc)
        return default

    def _refresh_performance_mode(self):
        """Derive live UI throttles from the low-memory gaming preset."""
        self._gaming_mode_active = (
            not self._auto_load_chat_model
            and not self._auto_start_mods
            and self._auto_unload_on_minimize
            and not self._chat_learning_enabled)
        if self._gaming_mode_active:
            self._status_tick_ms = 5000
            self._thinking_tick_ms = 700
            self._typewriter_tick_ms = 16
        else:
            self._status_tick_ms = 1000
            self._thinking_tick_ms = 350
            self._typewriter_tick_ms = 8

    def _sync_router_training_state(self):
        """Keep router background training aligned with the current setting."""
        router = getattr(self, "_router", None)
        if router is None:
            return
        try:
            router.set_training_enabled(self._chat_learning_enabled)
        except Exception as exc:
            logger.debug("Could not sync router training state: %s", exc)

    # ================================================================
    # Window lifecycle
    # ================================================================

    def _bind_escape_stop(self):
        """Bind Escape key to stop generation. Reusable after fullscreen."""
        self.bind("<Escape>", lambda e: self._stop_generation()
                  if getattr(self, '_is_generating', False) else None)

    def _bind_page_nav_shortcuts(self):
        """Bind Ctrl+1..7 to switch between pages."""
        pages = ["CORE", "CMD", "DOCS", "MODELS", "ROUTER", "FORGE", "CONFIG"]
        for i, page in enumerate(pages, start=1):
            self.bind(
                f"<Control-Key-{i}>",
                lambda e, p=page: self._switch_page(p))
        # Ctrl+N for new chat
        self.bind("<Control-n>", lambda e: self._new_chat())

    def _on_window_resize(self, event):
        """Handle window resize events with debouncing to reduce lag."""
        # Only process resize events for the main window, not child widgets
        if event.widget != self:
            return
        
        # Cancel any pending resize completion
        if self._resize_timer:
            self.after_cancel(self._resize_timer)
        
        # Mark as resizing and schedule completion check
        self._resizing = True
        self._resize_timer = self.after(150, self._on_resize_complete)
    
    def _on_resize_complete(self):
        """Called after resize has stopped for 150ms."""
        self._resizing = False
        self._resize_timer = None

    def _on_window_unmap(self, _event=None):
        """Suspend the loaded chat model when the window is minimized."""
        if not getattr(self, "_auto_unload_on_minimize", False):
            return
        if self.state() != "iconic":
            return
        self._suspend_model_memory(silent=True)

    def _on_window_map(self, _event=None):
        """Resume a model that was auto-suspended due to minimize."""
        if not getattr(self, "_model_suspended_by_minimize", False):
            return
        if self.state() == "iconic":
            return
        self._resume_suspended_model()

    def _on_close(self):
        """Clean up resources when the window is closed."""
        if getattr(self, '_shutting_down', False):
            return  # Already shutting down, don't re-enter
        self._shutting_down = True

        # Stop any active training so background threads release GPU
        if self.training_active:
            self.training_active = False

        # Stop all GIF animations so their after() loops stop
        for anim in getattr(self, '_chat_gif_animations', []):
            anim['active'] = False

        # Flush any pending log entries before shutdown
        flush = getattr(self, '_flush_log', None)
        if flush:
            try:
                flush()
            except (RuntimeError, AttributeError):
                pass  # Widget may be partially destroyed

        # Close the session log file
        close_log = getattr(self, '_close_log_file', None)
        if close_log:
            try:
                close_log()
            except (RuntimeError, AttributeError):
                pass

        # Hide the window immediately so it feels instant
        self.withdraw()

        # Stop loaded model backend (including GGUF llama-server) first.
        if getattr(self, 'engine', None) is not None:
            try:
                self._save_model_context()
                self._release_loaded_engine()
            except Exception as exc:
                logger.debug("Engine release failed during shutdown: %s", exc)

        # Stop all mod subprocesses
        with self._mod_lock:
            procs = list(self.mod_processes.items())
        for mod_id, proc in procs:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
        with self._mod_lock:
            self.mod_processes.clear()

        # Kill any running CMD page subprocess
        cmd_proc = getattr(self, '_cmd_proc', None)
        if cmd_proc and cmd_proc.poll() is None:
            try:
                cmd_proc.terminate()
                cmd_proc.wait(timeout=3)
            except Exception as exc:
                try:
                    cmd_proc.kill()
                except Exception as kill_exc:
                    logger.debug("CMD subprocess kill failed: %s", kill_exc)
                logger.debug("CMD subprocess terminate/wait failed: %s", exc)

        # Stop the mod router (with timeout so it can't block shutdown)
        if self._router:
            import threading as _thr
            def _stop_router():
                try:
                    self._router.stop()
                except Exception as exc:
                    logger.debug("Router stop failed during shutdown: %s", exc)
            t = _thr.Thread(target=_stop_router, daemon=True)
            t.start()
            t.join(timeout=3)

        # Stop voice input / TTS
        self._voice_recording = False
        stopper = getattr(self, '_voice_stopper', None)
        if stopper:
            try:
                stopper(wait_for_stop=False)
            except Exception as exc:
                logger.debug("Voice stopper failed during shutdown: %s", exc)
        self._tts_shutdown()

        # Save Forge training settings before widgets are destroyed
        save_brief = getattr(self, '_save_training_brief', None)
        if save_brief:
            try:
                save_brief()
            except Exception as exc:
                logger.debug("Training brief save failed during shutdown: %s", exc)

        self._save_window_geometry()
        self._save_config_overrides()

        # Cancel pending auto-save timer
        auto_save_id = getattr(self, '_docs_auto_save_id', None)
        if auto_save_id:
            try:
                self.after_cancel(auto_save_id)
            except ValueError:
                pass  # timer already fired

        # Force GC + CUDA cleanup so GPU memory is freed
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Release single-instance lock so re-opening works immediately
        _release_instance_lock()

        self.destroy()

    def _start_parent_watchdog(self):
        """Monitor the parent process and exit if it dies.

        When the VS Code terminal is closed, PowerShell terminates
        but the child python process stays orphaned.  This watchdog
        detects that and triggers a clean shutdown.
        """
        ppid = os.getppid()
        if not ppid or ppid <= 0:
            return

        def _watch():
            while not getattr(self, '_shutting_down', False):
                time.sleep(2)
                if not self._is_process_alive(ppid):
                    logger.info(
                        "Parent process %d gone — shutting down", ppid)
                    self.after(0, self._on_close)
                    break

        t = threading.Thread(target=_watch, daemon=True)
        t.start()

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        """Check whether a process is still running (Windows)."""
        try:
            kernel32 = ctypes.windll.kernel32
            # SYNCHRONIZE (0x00100000) — minimal access right
            handle = kernel32.OpenProcess(0x00100000, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            # If the check itself fails, assume still alive
            return True

    def _restart_gui(self):
        """Restart the GUI application (e.g. for font size changes).

        Launches a new process with the same arguments, then
        cleans up and exits the current one.
        """
        import subprocess
        import sys

        # Launch new instance before we tear down
        subprocess.Popen([sys.executable] + sys.argv)

        # Clean up and exit
        self._on_close()
        sys.exit(0)

    # ================================================================
    # Live theme switching
    # ================================================================

    def _apply_theme_live(self, name: str):
        """Apply a new colour theme live without restarting.

        Updates all C_* constants, then walks the entire widget tree
        and remaps old theme colours to the new ones.
        """
        from enigma_engine.gui.widgets import reload_theme
        from enigma_engine.gui.themes import save_theme_preference

        color_map = reload_theme(name)
        save_theme_preference(name)

        if not color_map:
            self.status_bar.set_left(f"Theme '{name}' — already active")
            return

        # Update main window background
        from enigma_engine.gui.widgets import C_BG
        self.configure(fg_color=C_BG)

        # Walk all widgets and remap old theme colours to new
        self._retheme_tree(self, color_map)

        self.status_bar.set_left(f"Theme '{name}' applied")

    def _retheme_tree(self, widget, color_map: dict[str, str]):
        """Recursively walk widget tree and remap theme colours."""
        self._retheme_one(widget, color_map)
        try:
            for child in widget.winfo_children():
                self._retheme_tree(child, color_map)
        except Exception:
            pass

    def _retheme_one(self, widget, color_map: dict[str, str]):
        """Remap theme colours on a single widget."""
        import tkinter as tk

        # --- CustomTkinter widgets ---
        ctk_props = (
            "fg_color", "border_color", "text_color",
            "hover_color", "button_color", "button_hover_color",
            "placeholder_text_color",
            "scrollbar_button_color", "scrollbar_button_hover_color",
            "selected_color", "selected_hover_color",
            "unselected_color", "unselected_hover_color",
            "progress_color",
        )
        if isinstance(widget, ctk.CTkBaseClass):
            for prop in ctk_props:
                try:
                    val = widget.cget(prop)
                    if isinstance(val, (list, tuple)):
                        check = str(val[1]).lower()
                    else:
                        check = str(val).lower()
                    if check in color_map:
                        widget.configure(**{prop: color_map[check]})
                except (ValueError, AttributeError, tk.TclError):
                    continue

        # --- Plain tk.Entry (inside SelectableLabel) ---
        if isinstance(widget, tk.Entry) and not isinstance(
                widget, tk.Text):
            for prop in ("fg", "bg", "readonlybackground",
                         "selectbackground", "selectforeground"):
                try:
                    val = str(widget.cget(prop)).lower()
                    if val in color_map:
                        widget.configure(**{prop: color_map[val]})
                except (tk.TclError, AttributeError):
                    continue

        # --- Plain tk.Text (inside CTkTextbox) ---
        if isinstance(widget, tk.Text):
            for prop in ("fg", "bg", "selectbackground",
                         "selectforeground", "insertbackground"):
                try:
                    val = str(widget.cget(prop)).lower()
                    if val in color_map:
                        widget.configure(**{prop: color_map[val]})
                except (tk.TclError, AttributeError):
                    continue
            # Remap tag foreground/background colours
            for tag in widget.tag_names():
                for tag_prop in ("foreground", "background",
                                 "selectforeground", "selectbackground"):
                    try:
                        val = widget.tag_cget(tag, tag_prop)
                        if val and str(val).lower() in color_map:
                            widget.tag_configure(
                                tag,
                                **{tag_prop: color_map[str(val).lower()]})
                    except (tk.TclError, AttributeError):
                        continue

        # --- CTkTextbox: also update inner _textbox widget ---
        if isinstance(widget, ctk.CTkTextbox):
            inner = getattr(widget, "_textbox", None)
            if inner and inner is not widget:
                self._retheme_one(inner, color_map)

    # ================================================================
    # Window geometry persistence
    # ================================================================

    def _restore_window_geometry(self):
        """Restore window position and size from gui_settings.json."""
        import json
        settings_path = DATA_DIR / "gui_settings.json"
        if not settings_path.exists():
            return
        try:
            data = json.loads(
                settings_path.read_text(encoding="utf-8"))
            w = data.get("window_width", 0)
            h = data.get("window_height", 0)
            x = data.get("window_x")
            y = data.get("window_y")
            if w >= 800 and h >= 500 and x is not None and y is not None:
                # Use virtual screen bounds (all monitors) for clamping
                vx, vy, vw, vh = self._get_virtual_screen_bounds()
                x = max(vx - w + 100, min(vx + vw - 100, x))
                y = max(vy - h + 50, min(vy + vh - 50, y))
                self.geometry(f"{w}x{h}+{x}+{y}")
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    @staticmethod
    def _get_virtual_screen_bounds() -> tuple[int, int, int, int]:
        """Return (x, y, width, height) of the virtual screen spanning all monitors."""
        import sys
        if sys.platform == "win32":
            try:
                import ctypes
                user32 = ctypes.windll.user32
                vx = user32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
                vy = user32.GetSystemMetrics(77)   # SM_YVIRTUALSCREEN
                vw = user32.GetSystemMetrics(78)   # SM_CXVIRTUALSCREEN
                vh = user32.GetSystemMetrics(79)   # SM_CYVIRTUALSCREEN
                if vw > 0 and vh > 0:
                    return (vx, vy, vw, vh)
            except Exception:
                pass
        return (0, 0, 1920, 1080)

    def _save_window_geometry(self):
        """Save current window position and size to gui_settings.json."""
        import json
        from enigma_engine.core.safe_save import atomic_write_json
        settings_path = DATA_DIR / "gui_settings.json"
        data = {}
        if settings_path.exists():
            try:
                data = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        try:
            data["window_width"] = self.winfo_width()
            data["window_height"] = self.winfo_height()
            data["window_x"] = self.winfo_x()
            data["window_y"] = self.winfo_y()
            atomic_write_json(settings_path, data)
        except (OSError, RuntimeError):
            pass

    # ================================================================
    # Label copy - right-click to copy label text
    # ================================================================

    def _enable_label_copy(self):
        """Walk all widgets and bind right-click copy on CTkLabels.

        This makes every label in the GUI copyable so users can
        select and copy any visible text. SelectableLabel widgets
        already have built-in copy menus, so they are skipped.
        """

        def _bind_label(widget):
            # SelectableLabel handles its own copy menu
            if isinstance(widget, SelectableLabel):
                pass
            elif isinstance(widget, ctk.CTkLabel):
                widget.bind("<Button-3>", lambda e, w=widget: (
                    self._show_label_copy_menu(e, w)), add="+")
            try:
                for child in widget.winfo_children():
                    _bind_label(child)
            except Exception:
                pass

        _bind_label(self)

    def _show_label_copy_menu(self, event, widget):
        """Show a right-click menu to copy label text."""
        import tkinter as tk
        text = ""
        try:
            text = widget.cget("text")
        except Exception:
            return
        if not text:
            return
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(
            label="Copy",
            command=lambda: self._copy_to_clipboard(text))
        menu.tk_popup(event.x_root, event.y_root)

    def _copy_to_clipboard(self, text: str):
        """Copy text to the system clipboard."""
        self.clipboard_clear()
        self.clipboard_append(text)

    # ================================================================
    # Shell - header + nav + content + status bar
    # ================================================================

    def _build_shell(self):
        # Header bar
        header = ctk.CTkFrame(
            self, height=56, fg_color=C_PANEL, corner_radius=0,
            border_width=0)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        self._header = header

        # Bottom border on header
        ctk.CTkFrame(
            header, height=1, fg_color=C_BORDER,
            corner_radius=0).pack(fill="x", side="bottom")

        # Title
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left", padx=(16, 0))
        SelectableLabel(
            title_frame, text="ENIGMA", font=FONT_TITLE,
            text_color=C_TEXT_BRIGHT).pack(side="left")
        SelectableLabel(
            title_frame, text=" ENGINE", font=FONT_TITLE,
            text_color=C_ACCENT).pack(side="left")
        SelectableLabel(
            title_frame, text=f"  {VERSION}", font=FONT_TINY,
            text_color=C_TEXT_DIM).pack(side="left", pady=(6, 0))

        # Nav toggle button in header (arrow icon)
        self._nav_toggle = themed_button(
            header, "\u25c0", style="icon",
            width=38, height=38,
            font=FONT_SECTION, text_color=C_ACCENT,
            command=self._toggle_nav)
        self._nav_toggle.pack(side="left", padx=(4, 0))
        Tooltip(self._nav_toggle, "Collapse navigation")

        # Pin / always-on-top toggle
        self._pin_btn = themed_button(
            header, "\U0001F4CC", style="icon",
            width=38, height=38,
            font=FONT_SECTION,
            command=self._toggle_pin)
        self._pin_btn.pack(side="left", padx=(2, 0))
        Tooltip(self._pin_btn, "Pin window on top")

        # Keyboard shortcuts help button
        self._shortcuts_btn = themed_button(
            header, "?", style="icon",
            width=38, height=38,
            font=FONT_SECTION,
            command=self._show_shortcuts_overlay)
        self._shortcuts_btn.pack(side="left", padx=(2, 0))
        Tooltip(self._shortcuts_btn, "Keyboard shortcuts")

        # Right side status
        status_frame = ctk.CTkFrame(header, fg_color="transparent")
        status_frame.pack(side="right", padx=16)

        self.header_dot = StatusDot(status_frame, color=C_TEXT_DIM)
        self.header_dot.pack(side="left", padx=(0, 6))

        self.header_status = SelectableLabel(
            status_frame, text="NO MODEL", font=FONT_SMALL,
            text_color=C_TEXT_DIM)
        self.header_status.pack(side="left")
        Tooltip(status_frame,
                "Model status: gray=none, orange=loading, "
                "green=ready, red=failed")

        # Body (nav + content)
        body = ctk.CTkFrame(self, fg_color=C_BG, corner_radius=0)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # Nav rail
        self._nav_expanded = True
        self._nav_width = 170
        self._nav_collapsed_width = 46
        # Control nav width via the grid column, not frame width
        body.grid_columnconfigure(0, minsize=self._nav_width)
        nav = ctk.CTkFrame(
            body, fg_color=C_PANEL,
            corner_radius=0, border_width=0)
        nav.grid(row=0, column=0, sticky="nsew")
        self._nav_frame = nav
        self._body = body

        # Right border on nav
        nav_border = ctk.CTkFrame(
            nav, width=1, fg_color=C_BORDER, corner_radius=0)
        nav_border.pack(side="right", fill="y")

        nav_items = ["CORE", "CMD", "DOCS", "MODELS", "ROUTER", "FORGE", "CONFIG"]
        for label in nav_items:
            btn = NavButton(
                nav, label,
                lambda l=label: self._switch_page(l))
            btn.pack(fill="x", pady=0)
            self._nav_buttons[label] = btn

        # Mod section in nav
        self._mod_sep = ctk.CTkFrame(
            nav, height=1, fg_color=C_BORDER,
            corner_radius=0)
        self._mod_sep.pack(fill="x", padx=8, pady=(12, 8))

        self._mod_label = SelectableLabel(
            nav, text="  MODS",
            font=FONT_TINY, text_color=C_TEXT_DIM, anchor="w")
        self._mod_label.pack(fill="x", padx=4, pady=(0, 4))

        if self.mods_data:
            for mod in self.mods_data:
                page_name = f"MOD_{mod['id']}"
                btn = NavButton(
                    nav, mod["name"],
                    lambda pn=page_name: self._switch_page(pn))
                btn.pack(fill="x", pady=0)
                self._nav_buttons[page_name] = btn
        else:
            SelectableLabel(
                nav, text="  (none)", font=FONT_TINY,
                text_color=C_TEXT_DIM).pack(anchor="w", padx=10)

        # Content area
        self.content = ctk.CTkFrame(body, fg_color=C_BG, corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew", padx=(1, 0))
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        # Status bar
        self.status_bar = StatusBar(self)
        self.status_bar.pack(fill="x", side="bottom")
        self.status_bar.set_left("READY")

    def _toggle_nav(self):
        """Collapse or expand the nav rail.

        When collapsed, the nav is fully hidden (0px) and the
        content area expands to use all available space.
        When expanded, full 170px with labels.
        """
        if self._nav_expanded:
            # Collapse: hide nav entirely so content expands
            self._nav_expanded = False
            self._nav_frame.grid_remove()
            self._body.grid_columnconfigure(0, minsize=0)
            self._nav_toggle.configure(
                text="\u25b6", text_color=C_TEXT_DIM)
        else:
            # Expand: show nav, restore column width
            self._nav_expanded = True
            self._nav_frame.grid()
            self._body.grid_columnconfigure(
                0, minsize=self._nav_width)
            self._nav_toggle.configure(
                text="\u25c0", text_color=C_ACCENT)

    def _toggle_pin(self):
        """Toggle always-on-top window pinning."""
        self._pinned = not self._pinned
        self.attributes("-topmost", self._pinned)
        if self._pinned:
            self._pin_btn.configure(
                text_color=C_ACCENT, fg_color=C_SURFACE)
        else:
            self._pin_btn.configure(
                text_color=C_TEXT_DIM, fg_color="transparent")

    def _show_shortcuts_overlay(self):
        """Toggle an inline dropdown listing all keyboard shortcuts.

        Uses place() inside the main window so it is never hidden
        behind an always-on-top window (unlike CTkToplevel).
        """
        existing = getattr(self, '_shortcuts_dropdown', None)
        if existing:
            try:
                existing.destroy()
            except Exception:
                pass
            self._shortcuts_dropdown = None
            return

        dropdown = ctk.CTkFrame(
            self, fg_color=C_PANEL, corner_radius=6,
            border_width=1, border_color=C_BORDER_ACCENT)
        self._shortcuts_dropdown = dropdown

        ctk.CTkLabel(
            dropdown, text="Keyboard Shortcuts",
            font=FONT_SECTION, text_color=C_TEXT_BRIGHT
        ).pack(padx=12, pady=(10, 6))

        shortcuts = [
            ("Ctrl + 1\u20137", "Switch pages (CORE \u2192 CONFIG)"),
            ("Ctrl + N", "New chat session"),
            ("Escape", "Stop generation"),
            ("Shift + Return", "Newline in chat input"),
            ("Ctrl + Z", "Undo (docs editor)"),
            ("Ctrl + Y", "Redo (docs editor)"),
            ("Ctrl + S", "Save (docs editor)"),
            ("Ctrl + F", "Find (docs editor)"),
            ("Up / Down", "Command history (CMD page)"),
        ]

        for key_combo, description in shortcuts:
            row = ctk.CTkFrame(dropdown, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=2)
            ctk.CTkLabel(
                row, text=key_combo, font=FONT_TINY,
                text_color=C_ACCENT, width=130, anchor="w"
            ).pack(side="left")
            ctk.CTkLabel(
                row, text=description, font=FONT_TINY,
                text_color=C_TEXT, anchor="w"
            ).pack(side="left", fill="x", expand=True)

        # Pad the bottom
        ctk.CTkFrame(
            dropdown, fg_color="transparent", height=8
        ).pack()

        # Position below the ? button, anchored to its right edge
        self.update_idletasks()
        btn = self._shortcuts_btn
        btn_rx = btn.winfo_rootx() - self.winfo_rootx()
        btn_y = (btn.winfo_rooty() - self.winfo_rooty()
                 + btn.winfo_height() + 4)
        dropdown.update_idletasks()
        dd_w = dropdown.winfo_reqwidth()
        x = max(4, btn_rx + btn.winfo_width() - dd_w)
        dropdown.place(x=x, y=btn_y)
        dropdown.lift()

        # Dismiss on Escape
        def _dismiss(event=None):
            dd = getattr(self, '_shortcuts_dropdown', None)
            if dd:
                try:
                    dd.destroy()
                except Exception:
                    pass
                self._shortcuts_dropdown = None

        self.bind("<Escape>", lambda e: _dismiss(), add="+")

    def _switch_page(self, name: str):
        if name == self._current_page:
            return
        for key, btn in self._nav_buttons.items():
            btn.set_active(key == name)
        for key, page in self._pages.items():
            if key == name:
                page.grid(row=0, column=0, sticky="nsew")
            else:
                page.grid_forget()
        self._current_page = name

        # Safety: re-enable SEND button when switching to CORE
        # if no generation or model loading is actively running
        if (name == "CORE" and not self._thinking_active
                and not self._model_loading
                and not self._is_generating):
            try:
                self.send_btn.configure(state="normal")
            except Exception:
                pass

    def _make_page(self, name: str) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content, fg_color=C_BG, corner_radius=0)
        self._pages[name] = page
        return page

    def _count_model_params_background(self):
        """Count native model params in a background thread.

        Deferred from scan_models() to avoid loading multi-GB
        models synchronously during boot.
        """
        if getattr(self, "_gaming_mode_active", False):
            return
        from enigma_engine.gui.scanners import _count_params_native
        native = [m for m in self.models_data
                  if m["format"] in ("pth", "pt") and not m["params"]]
        if not native:
            return

        def _count():
            from pathlib import Path
            for m in native:
                try:
                    p = _count_params_native(Path(m["path"]))
                    if p:
                        m["params"] = p
                except Exception:
                    pass
            # Refresh MODELS page UI if it exists
            self.after(0, self._refresh_model_cards_if_visible)

        threading.Thread(target=_count, daemon=True).start()

    def _refresh_model_cards_if_visible(self):
        """Refresh model cards after background param counting."""
        frame = getattr(self, 'model_cards_frame', None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        if self.models_data:
            self._populate_model_cards(frame)

    # ================================================================
    # Monologue / Journal greeting
    # ================================================================

    def _get_monologue_mode(self) -> str:
        """Return the current monologue mode.

        Reads from gui_settings.json first, falls back to
        get_config for the default.
        """
        try:
            settings_path = DATA_DIR / "gui_settings.json"
            if settings_path.exists():
                import json
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
                mode = settings.get("monologue_mode", "")
                if mode:
                    return mode
        except (json.JSONDecodeError, OSError):
            pass
        from enigma_engine.config.defaults import get_config
        return get_config("monologue_mode", "disabled")

    def _show_journal_greeting(self):
        """Show a journal-based greeting if monologue is enabled.

        Reads the most recent journal entry that passes the
        coherence quality gate and displays it as a system message.
        """
        try:
            mode = self._get_monologue_mode()
            if mode == "disabled":
                return

            from enigma_engine.core.monologue import (
                Journal, DEFAULT_COHERENCE_THRESHOLD)

            model_path = getattr(self, "model_path", None)
            if not model_path:
                return

            from pathlib import Path as _Path
            journal_dir = _Path(model_path).parent
            journal = Journal(journal_dir)
            entry = journal.latest()
            if entry is None:
                return

            coherence = entry.get("coherence", 0.0)
            if coherence < DEFAULT_COHERENCE_THRESHOLD:
                return

            text = entry.get("text", "").strip()
            if text:
                self._chat_system(f"[Journal] {text}")
        except Exception as exc:
            logger.debug("Journal greeting failed: %s", exc)

    def _start_status_ticker(self):
        """Update status bar with uptime every second."""
        # Detect CPU and GPU names in background
        # to avoid blocking the UI on first tick
        self._hw_device_label = (
            platform.processor()
            or platform.machine()
            or "CPU")
        tick_ms = getattr(self, "_status_tick_ms", 1000)

        def _detect_hw():
            cpu_name = self._hw_device_label
            try:
                import cpuinfo
                info = cpuinfo.get_cpu_info()
                cpu_name = info.get("brand_raw", cpu_name)
            except Exception:
                pass
            device = cpu_name
            try:
                import torch
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    mem_gb = props.total_memory / (1024 ** 3)
                    device = (
                        f"{cpu_name} // {props.name} {mem_gb:.0f}GB")
            except ImportError:
                pass
            self._hw_device_label = device

        if not getattr(self, "_gaming_mode_active", False):
            threading.Thread(target=_detect_hw, daemon=True).start()

        def _tick():
            if getattr(self, '_shutting_down', False):
                return
            # Skip updates when window is minimized — saves CPU
            if self.state() != 'iconic':
                elapsed = int(time.time() - self._boot_time)
                h = elapsed // 3600
                m = (elapsed % 3600) // 60
                s = elapsed % 60
                self.status_bar.set_right(
                    f"UPTIME {h:02d}:{m:02d}:{s:02d}")
                self.status_bar.set_center(self._hw_device_label)
            self.after(tick_ms, _tick)
        self.after(100, _tick)


# -------------------------------------------------------------------
# Single-instance guard
# -------------------------------------------------------------------

_lock_handle = None  # kept alive for process lifetime


def _release_instance_lock():
    """Release the single-instance lock so re-opening works immediately."""
    global _lock_handle
    if _lock_handle is None:
        return
    if os.name == "nt":
        import ctypes as _ct
        _ct.windll.kernel32.ReleaseMutex(_lock_handle)
        _ct.windll.kernel32.CloseHandle(_lock_handle)
    else:
        try:
            import fcntl
            fcntl.flock(_lock_handle, fcntl.LOCK_UN)
            _lock_handle.close()
        except Exception:
            pass
    _lock_handle = None


def _acquire_instance_lock() -> bool:
    """Prevent multiple GUI instances from running simultaneously.

    Uses a Windows kernel mutex (or a lockfile on other OSes).
    Returns True if this is the only instance, False if another
    GUI is already running.
    """
    global _lock_handle
    if os.name == "nt":
        # Windows: named mutex is the most reliable approach
        import ctypes as _ct
        _lock_handle = _ct.windll.kernel32.CreateMutexW(
            None, True, "EnigmaEngineGUI_SingleInstance")
        # ERROR_ALREADY_EXISTS = 183
        if _ct.windll.kernel32.GetLastError() == 183:
            _ct.windll.kernel32.CloseHandle(_lock_handle)
            _lock_handle = None
            return False
        return True
    else:
        # Unix: use a lockfile with fcntl
        import fcntl
        from pathlib import Path as _LockPath
        lock_path = _LockPath(
            os.environ.get("XDG_RUNTIME_DIR",
                           "/tmp")) / ".enigma_gui.lock"  # noqa: S108
        _lock_handle = open(lock_path, "w")  # noqa: SIM115
        try:
            fcntl.flock(_lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            _lock_handle.close()
            _lock_handle = None
            return False


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

def run_gui(model_path: str | None = None):
    """Launch the desktop GUI."""
    if not _acquire_instance_lock():
        logger.warning(
            "Another Enigma GUI is already running — exiting.")
        print("\n  Another Enigma GUI is already running.\n"
              "  Close it first, or use Task Manager to end "
              "leftover python.exe processes.\n")
        return

    app = EnigmaGUI(model_path=model_path)

    # On Windows, register a console event handler so closing the cmd
    # window triggers a clean shutdown instead of leaving a zombie.
    if os.name == "nt":
        import ctypes as _ct

        @_ct.WINFUNCTYPE(_ct.c_bool, _ct.c_uint)
        def _console_handler(event):
            # CTRL_CLOSE_EVENT=2, CTRL_LOGOFF_EVENT=5, CTRL_SHUTDOWN_EVENT=6
            if event in (2, 5, 6):
                try:
                    app._on_close()
                except Exception:
                    pass
                # Return True so Python doesn't raise SystemExit
                # while we're cleaning up
                return True
            return False

        _ct.windll.kernel32.SetConsoleCtrlHandler(_console_handler, True)

    try:
        app.mainloop()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # Ensure resources are freed even on abnormal exit
        if not getattr(app, '_shutting_down', False):
            try:
                app._on_close()
            except Exception:
                pass
        _release_instance_lock()


if __name__ == "__main__":
    run_gui()
