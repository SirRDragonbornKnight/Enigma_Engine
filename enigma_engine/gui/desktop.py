"""
Enigma Engine - Desktop Interface
===================================

Desktop app built with CustomTkinter.
Minimal processing, clean layout.

Pages:
  CORE   - Chat with history sidebar, prompts, voice, file attach
  ROUTER - Browse models, assign to chat/trainer/bricks
  FORGE  - Train models and tokenizers
  CONFIG - Generation params, customization

Usage:
    python run.py --gui
    python run.py --gui --model models/enigma_small.pth
"""
from __future__ import annotations

import platform
import subprocess
import time
from typing import Any

import customtkinter as ctk

from enigma_engine.gui.widgets import (
    C_ACCENT, C_ACCENT_DIM, C_BG, C_BORDER, C_BORDER_ACCENT,
    C_CYAN, C_GREEN, C_PANEL, C_SURFACE, C_TEXT, C_TEXT_BRIGHT,
    C_TEXT_DIM,
    FONT_SECTION, FONT_SMALL, FONT_TINY, FONT_TITLE,
    NavButton, StatusBar, StatusDot, Tooltip,
)
from enigma_engine.gui.gui_pages import PagesMixin
from enigma_engine.gui.gui_logic import LogicMixin
from enigma_engine.gui.gui_forge import ForgeMixin
from enigma_engine.gui.gui_bricks import BrickMixin
from enigma_engine.gui.gui_brick_page import BrickPageMixin
from enigma_engine.gui.gui_cmd_page import CMDPageMixin
from enigma_engine.gui.gui_docs_page import DocsPageMixin

# Backward-compatible re-exports (tests and external code
# import these from enigma_engine.gui.desktop)
from enigma_engine.gui.scanners import (  # noqa: F401
    CONFIG_DESCRIPTIONS, CONFIG_LIMITS, ROUTE_KEYS,
    BRICKS_DIR, DATA_DIR, INFO_DIR, MEMORY_DIR, MODELS_DIR,
    PROFILES_DIR, PROJECT_ROOT, SESSIONS_DIR,
    clamp_config,
    scan_bricks, scan_docs, scan_models, scan_profiles,
    scan_sessions, scan_training_data,
)


# -------------------------------------------------------------------
# Main GUI
# -------------------------------------------------------------------

class EnigmaGUI(
        DocsPageMixin, ForgeMixin, BrickMixin, BrickPageMixin,
        CMDPageMixin, LogicMixin, PagesMixin, ctk.CTk):
    """Enigma Engine desktop interface.

    Combines mixins via multiple inheritance:
    - PagesMixin:      UI page construction (CORE, MODELS, ROUTER, FORGE, CONFIG)
    - DocsPageMixin:   Documentation browser, profiles, file management
    - BrickPageMixin:  Per-brick page construction
    - CMDPageMixin:    Command terminal page
    - LogicMixin:      Chat, sessions, profiles, routes
    - ForgeMixin:      Training, model create/delete
    - BrickMixin:      Brick subprocess management
    All mixins access shared state through self.* attributes set here.
    """

    def __init__(self, model_path: str | None = None):
        super().__init__()

        self.title("ENIGMA ENGINE")
        self.geometry("1440x900")
        self.minsize(800, 500)
        self.configure(fg_color=C_BG)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Engine state
        self.engine = None
        self.model_path = model_path
        self.active_profile: str | None = None
        self.config_overrides: dict[str, Any] = {}
        self.history: list[dict[str, str]] = []
        self.brick_processes: dict[str, subprocess.Popen] = {}
        self.training_active = False
        self.voice_enabled = False
        self.attached_file: str | None = None
        self._boot_time = time.time()
        self._thinking_active = False
        self._chat_fullscreen = False

        # Display names (loaded from gui_settings.json)
        self.user_name = "YOU"
        self.ai_name = "ENIGMA"

        # Per-model context (history + prompt persistence)
        self.model_context = None

        # Per-route model assignments: route_key -> model path
        self.route_assignments: dict[str, str | None] = {}

        # Scan filesystem
        self.bricks_data = scan_bricks()
        self.models_data = scan_models()
        self.profiles_data = scan_profiles()
        self.training_files = scan_training_data()

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
        for brick in self.bricks_data:
            brick["_running"] = False
            self._build_page_brick(brick)
        self._switch_page("CORE")
        self._load_config_defaults()
        self._load_display_names()
        self._start_status_ticker()

        # Auto-start all bricks
        for brick in self.bricks_data:
            self._auto_start_brick(brick)

        # Auto-load model if given
        if self.model_path:
            self.after(300, lambda: self._load_model(self.model_path))

        # Make all label text copyable via right-click
        self.after(500, self._enable_label_copy)

    # ================================================================
    # Label copy - right-click to copy label text
    # ================================================================

    def _enable_label_copy(self):
        """Walk all widgets and bind right-click copy on CTkLabels.

        This makes every label in the GUI copyable so users can
        select and copy any visible text.
        """
        import tkinter as tk

        def _bind_label(widget):
            if isinstance(widget, ctk.CTkLabel):
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
        ctk.CTkLabel(
            title_frame, text="ENIGMA", font=FONT_TITLE,
            text_color=C_TEXT_BRIGHT).pack(side="left")
        ctk.CTkLabel(
            title_frame, text=" ENGINE", font=FONT_TITLE,
            text_color=C_ACCENT).pack(side="left")
        ctk.CTkLabel(
            title_frame, text="  1.1.0", font=FONT_TINY,
            text_color=C_TEXT_DIM).pack(side="left", pady=(6, 0))

        # Nav toggle button in header (arrow icon)
        self._nav_toggle = ctk.CTkButton(
            header, text="\u25c0", width=38, height=38,
            font=FONT_SECTION, corner_radius=2,
            fg_color="transparent", hover_color=C_SURFACE,
            text_color=C_ACCENT, command=self._toggle_nav)
        self._nav_toggle.pack(side="left", padx=(4, 0))
        Tooltip(self._nav_toggle, "Collapse navigation")

        # Right side status
        status_frame = ctk.CTkFrame(header, fg_color="transparent")
        status_frame.pack(side="right", padx=16)

        self.header_dot = StatusDot(status_frame, color=C_TEXT_DIM)
        self.header_dot.pack(side="left", padx=(0, 6))

        self.header_status = ctk.CTkLabel(
            status_frame, text="NO MODEL", font=FONT_SMALL,
            text_color=C_TEXT_DIM)
        self.header_status.pack(side="left")

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

        # Brick section in nav
        self._brick_sep = ctk.CTkFrame(
            nav, height=1, fg_color=C_BORDER,
            corner_radius=0)
        self._brick_sep.pack(fill="x", padx=8, pady=(12, 8))

        self._brick_label = ctk.CTkLabel(
            nav, text="  BRICKS",
            font=FONT_TINY, text_color=C_TEXT_DIM, anchor="w")
        self._brick_label.pack(fill="x", padx=4, pady=(0, 4))

        if self.bricks_data:
            for brick in self.bricks_data:
                page_name = f"BRICK_{brick['id']}"
                btn = NavButton(
                    nav, brick["name"],
                    lambda pn=page_name: self._switch_page(pn))
                btn.pack(fill="x", pady=0)
                self._nav_buttons[page_name] = btn
        else:
            ctk.CTkLabel(
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

    def _make_page(self, name: str) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content, fg_color=C_BG, corner_radius=0)
        self._pages[name] = page
        return page

    def _start_status_ticker(self):
        """Update status bar with uptime every second."""
        # Detect CPU name once
        cpu_name = platform.processor() or platform.machine() or "Unknown CPU"
        # Shorten common prefixes for display
        for prefix in ("Intel64 Family", "Intel(R) Core(TM)", "AMD"):
            if cpu_name.startswith(prefix):
                break
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            cpu_name = info.get("brand_raw", cpu_name)
        except Exception:
            pass

        def _tick():
            elapsed = int(time.time() - self._boot_time)
            h = elapsed // 3600
            m = (elapsed % 3600) // 60
            s = elapsed % 60
            self.status_bar.set_right(
                f"UPTIME {h:02d}:{m:02d}:{s:02d}")

            device = cpu_name
            try:
                import torch
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    mem_gb = props.total_memory / (1024 ** 3)
                    device = f"{cpu_name} // {props.name} {mem_gb:.0f}GB"
            except ImportError:
                pass
            self.status_bar.set_center(device)

            self.after(1000, _tick)
        self.after(100, _tick)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

def run_gui(model_path: str | None = None):
    """Launch the desktop GUI."""
    app = EnigmaGUI(model_path=model_path)
    app.mainloop()


if __name__ == "__main__":
    run_gui()
