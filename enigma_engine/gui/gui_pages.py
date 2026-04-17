"""
Enigma Engine - GUI Page Builders
===================================

Mixin providing page construction methods for EnigmaGUI.
Each page is a separate method building its widget tree.

Pages: CORE, MODELS, ROUTER (here)
FORGE page: gui_pages_forge.py
CONFIG page: gui_pages_config.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import customtkinter as ctk

from enigma_engine.gui.widgets import (
    C_ACCENT, C_ACCENT_DIM, C_BG, C_BORDER,
    C_CYAN, C_GREEN, C_INPUT,
    C_ORANGE, C_PANEL, C_PURPLE,
    C_RED, C_RED_DIM, C_RED_HOVER, C_SURFACE, C_TEXT, C_TEXT_BRIGHT,
    C_TEXT_DIM,
    CollapsiblePanel,
    FONT_BODY, FONT_CHAT, FONT_INPUT,
    FONT_SECTION, FONT_SMALL, FONT_TINY,
    HUDFrame, SectionLabel, SelectableLabel, SelectableTextbox,
    StatusDot, ToggleButton, Tooltip,
    themed_button, themed_entry, themed_numeric_entry, themed_dropdown,
    themed_scroll,
    wire_hotkeys,
)
# Re-export so existing imports keep working
from enigma_engine.gui.gui_mod_page import ModPageMixin  # noqa: F401
from enigma_engine.gui.gui_pages_forge import ForgePageMixin  # noqa: F401
from enigma_engine.gui.gui_pages_config import ConfigPageMixin  # noqa: F401

logger = logging.getLogger(__name__)


class PagesMixin(ForgePageMixin, ConfigPageMixin):
    """Mixin providing page builder methods for EnigmaGUI.

    Expects the host class to have attributes set in __init__:
    - profiles_data, models_data, training_files, mods_data
    - config_overrides, config_entries
    - Various widget references (set during page construction)
    """

    # ================================================================
    # PAGE: CORE - Chat + History Sidebar
    # ================================================================

    def _build_page_core(self):
        page = self._make_page("CORE")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        # Top bar: label
        top = ctk.CTkFrame(page, fg_color="transparent", height=48)
        top.grid(row=0, column=0, sticky="ew",
                 padx=10, pady=(8, 2))

        SectionLabel(top, "Neural Interface").pack(
            side="left", fill="x", expand=True)

        # Fullscreen toggle button
        self._fullscreen_btn = themed_button(
            top, "\u26f6", style="icon",
            width=38, height=34,
            font=FONT_SECTION, command=self._toggle_chat_fullscreen)
        self._fullscreen_btn.pack(side="right", padx=(4, 0))
        Tooltip(self._fullscreen_btn, "Fullscreen chat")

        # Sidebar toggle button
        self._sidebar_toggle_btn = themed_button(
            top, "\u25e8", style="icon",
            width=38, height=34,
            font=FONT_SECTION,
            text_color=C_ACCENT, command=self._toggle_sidebar)
        self._sidebar_toggle_btn.pack(side="right", padx=(4, 0))

        # Resizable paned layout: chat | sidebar
        import tkinter as tk
        self._core_pane = tk.PanedWindow(
            page, orient="horizontal", sashwidth=6,
            bg=C_BG, borderwidth=0, sashpad=0,
            opaqueresize=True)
        self._core_pane.grid(row=1, column=0, sticky="nsew",
                             padx=0, pady=(2, 8))
        # Style the sash cursor
        self._core_pane.configure(sashcursor="sb_h_double_arrow")

        # Left column: chat + input
        chat_col = ctk.CTkFrame(self._core_pane, fg_color="transparent")
        chat_col.grid_columnconfigure(0, weight=1)
        chat_col.grid_rowconfigure(0, weight=1)

        # Chat display — native CTkTextbox scrollbar handles scrolling.
        # Previous design wrapped the textbox in CTkScrollableFrame and
        # tried to auto-expand the widget height, but the height
        # estimation was unreliable, causing text to be hidden.
        # The tk.Text widget handles its own scrolling natively, so we
        # just let the built-in scrollbar do the job.
        self.chat_display = SelectableTextbox(
            chat_col, wrap="word",
            font=FONT_CHAT, fg_color=C_PANEL, text_color=C_TEXT,
            border_width=1, border_color=C_BORDER,
            corner_radius=2, height=100,
            scrollbar_button_color=C_ACCENT_DIM,
            scrollbar_button_hover_color=C_ACCENT)
        self.chat_display.grid(
            row=0, column=0, sticky="nsew", padx=0, pady=0)
        # Backward compat — _chat_scroll no longer needed
        self._chat_scroll = None

        # Color tags for chat
        tb = self.chat_display._textbox
        tb.tag_configure("user", foreground=C_PURPLE,
                         lmargin1=12, lmargin2=12, rmargin=12,
                         spacing1=2)
        tb.tag_configure("assistant", foreground=C_ACCENT,
                         lmargin1=12, lmargin2=12, rmargin=12,
                         spacing1=2)
        tb.tag_configure("system", foreground=C_TEXT_DIM,
                         lmargin1=12, lmargin2=12)
        tb.tag_configure("error", foreground=C_RED,
                         lmargin1=12, lmargin2=12)
        tb.tag_configure("system_prefix", foreground=C_ORANGE,
                         font=("Consolas", 16, "bold"),
                         lmargin1=12)
        tb.tag_configure("system_msg", foreground=C_ORANGE,
                         lmargin1=12, lmargin2=12, rmargin=12,
                         spacing1=2)
        tb.tag_configure("timestamp", foreground=C_TEXT_DIM,
                         font=("Consolas", 12))
        tb.tag_configure("user_prefix", foreground=C_PURPLE,
                         font=("Consolas", 16, "bold"),
                         lmargin1=12)
        tb.tag_configure("assistant_prefix", foreground=C_ACCENT,
                         font=("Consolas", 16, "bold"),
                         lmargin1=12)
        tb.tag_configure("file_tag", foreground=C_CYAN,
                         font=("Consolas", 13),
                         lmargin1=12)
        # Clickable link tag (cyan, underlined, hand cursor)
        tb.tag_configure("link", foreground=C_CYAN,
                         underline=True,
                         lmargin1=12, lmargin2=12, rmargin=12)
        tb.tag_bind("link", "<Button-1>", self._on_link_click)
        # Media caption (dim, small, under images)
        tb.tag_configure("media_caption", foreground=C_TEXT_DIM,
                         font=("Consolas", 12),
                         lmargin1=12, lmargin2=12)
        # Video label tag (cyan, clickable)
        tb.tag_configure("video_link", foreground=C_CYAN,
                         font=("Consolas", 14, "bold"),
                         underline=True,
                         lmargin1=12, lmargin2=12)
        tb.tag_bind("video_link", "<Button-1>",
                    self._on_video_click)
        # File link tag (cyan, clickable)
        tb.tag_configure("file_link", foreground=C_CYAN,
                 font=("Consolas", 13, "bold"),
                 underline=True,
                 lmargin1=12, lmargin2=12)
        tb.tag_bind("file_link", "<Button-1>",
                self._on_file_click)
        # Single Motion/Leave handler for cursor — replaces per-tag
        # Enter/Leave which could miss events and leave cursor stuck.
        _CLICKABLE_TAGS = frozenset(("link", "video_link", "file_link"))

        def _chat_cursor_motion(event):
            try:
                idx = tb.index(f"@{event.x},{event.y}")
                tags = tb.tag_names(idx)
            except Exception:
                tags = ()
            if _CLICKABLE_TAGS.intersection(tags):
                tb.configure(cursor="hand2")
            else:
                tb.configure(cursor="")

        tb.bind("<Motion>", _chat_cursor_motion)
        tb.bind("<Leave>", lambda e: tb.configure(cursor=""))
        # Reasoning / chain-of-thought tag (dim italic)
        tb.tag_configure("reasoning", foreground=C_TEXT_DIM,
                         font=("Consolas", 13, "italic"),
                         lmargin1=24, lmargin2=24, rmargin=12,
                         spacing1=2)
        tb.tag_configure("reasoning_label", foreground=C_TEXT_DIM,
                         font=("Consolas", 12, "bold"),
                         lmargin1=12)
        # Store media references for click handling
        self._chat_media_refs = {}
        self._chat_gif_animations = []
        # Per-URL tag-to-URL mapping for precise link clicks
        self._link_urls = {}

        # Input area
        input_area = ctk.CTkFrame(chat_col, fg_color="transparent")
        input_area.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        input_area.grid_columnconfigure(0, weight=1)
        # Keep control column width fixed so PROCESSING text
        # never shifts the chat input width.
        input_area.grid_columnconfigure(1, minsize=140)
        # Lock indicator row height so it never shifts the chat
        input_area.grid_rowconfigure(0, minsize=24)

        # File attachment indicator (blue)
        self.file_indicator = SelectableLabel(
            input_area, text="", font=FONT_TINY,
            text_color=C_ACCENT, height=20)
        self.file_indicator.grid(
            row=0, column=0, sticky="w", padx=2)

        # Thinking indicator (fixed width so animation
        # does not shift the layout as dots change)
        self.thinking_label = SelectableLabel(
            input_area, text="", font=FONT_TINY,
            text_color=C_ACCENT_DIM, height=20,
            width=140, anchor="e")
        self.thinking_label.grid(
            row=0, column=1, sticky="e", padx=2)

        # Multi-line input + SEND button side by side
        self.chat_input = ctk.CTkTextbox(
            input_area, height=56, font=FONT_INPUT, wrap="word",
            fg_color=C_INPUT, border_color=C_ACCENT_DIM,
            border_width=1, text_color=C_TEXT_BRIGHT, corner_radius=2)
        self.chat_input.grid(row=1, column=0, sticky="nsew",
                             padx=(0, 4))
        wire_hotkeys(self.chat_input)
        self.chat_input.bind("<Return>", self._on_input_enter)
        self.chat_input.bind("<Shift-Return>", lambda e: None)
        self.chat_input.bind("<KeyRelease>", self._auto_resize_input)
        self.chat_input.bind("<Button-3>", self._chat_input_context_menu)
        # Up/Down arrow input history recall
        self._init_input_history()
        self.chat_input.bind("<Up>", self._on_input_up)
        self.chat_input.bind("<Down>", self._on_input_down)

        # SEND button to the right of the input
        self.send_btn = themed_button(
            input_area, "SEND", style="primary",
            width=80, height=56,
            font=FONT_SECTION, command=self._send_message)
        self.send_btn.grid(row=1, column=1, sticky="s")

        # STOP button (same slot as SEND, shown during generation)
        self.stop_btn = themed_button(
            input_area, "STOP", style="danger",
            width=80, height=56,
            font=FONT_SECTION, command=self._stop_generation)
        # Hidden until generation starts (SEND grid slot is reused)

        # Utility toolbar below the input
        toolbar = ctk.CTkFrame(input_area, fg_color="transparent")
        toolbar.grid(row=2, column=0, columnspan=2, sticky="ew",
                     pady=(4, 0))

        # Left side: attach, new, web access
        self._attach_btn = themed_button(
            toolbar, "\U0001f4ce", style="icon",
            width=38, height=32,
            font=FONT_SMALL, command=self._attach_file)
        self._attach_btn.pack(side="left", padx=(0, 4))

        self._new_btn = themed_button(
            toolbar, "NEW", style="secondary",
            width=60, height=32,
            font=FONT_SMALL, command=self._new_chat)
        self._new_btn.pack(side="left", padx=(0, 4))

        self._web_btn = ToggleButton(
            toolbar, text_on="\U0001f310", text_off="\U0001f310",
            on_toggle=self._on_web_access_toggle, start_on=False,
            width=38, height=32)
        self._web_btn.pack(side="left")

        # Reasoning toggle (chain-of-thought)
        self._reasoning_btn = ToggleButton(
            toolbar, text_on="\U0001f9e0", text_off="\U0001f9e0",
            on_toggle=self._on_reasoning_toggle, start_on=False,
            width=38, height=32)
        self._reasoning_btn.pack(side="left", padx=(4, 0))

        # RAG toggle (document Q&A)
        self._rag_btn = ToggleButton(
            toolbar, text_on="\U0001f4da", text_off="\U0001f4da",
            on_toggle=self._on_rag_toggle, start_on=False,
            width=38, height=32)
        self._rag_btn.pack(side="left", padx=(4, 0))

        # Edit last message button
        self._edit_btn = themed_button(
            toolbar, "\u270E", style="icon",
            width=38, height=32,
            font=FONT_SMALL,
            command=self._edit_last_message)
        self._edit_btn.pack(side="left", padx=(4, 0))

        # Right side: voice output + mic input
        self.mic_btn = themed_button(
            toolbar, "\U0001f3a4", style="icon",
            width=38, height=32,
            font=FONT_SMALL, command=self._toggle_voice_input)
        self.mic_btn.pack(side="right", padx=(4, 0))

        self.voice_btn = ToggleButton(
            toolbar, text_on="\U0001f50a", text_off="\U0001f507",
            on_toggle=self._on_voice_toggle, start_on=False,
            width=38, height=32)
        self.voice_btn.pack(side="right")

        # Token counter label (right side of toolbar)
        self._token_counter = SelectableLabel(
            toolbar, text="0 tokens", font=FONT_TINY,
            text_color=C_TEXT_DIM, height=20, anchor="e")
        self._token_counter.pack(side="right", padx=(8, 8))

        # Tooltips for CORE page buttons
        Tooltip(self.send_btn, "Send message (Enter)")
        Tooltip(self.stop_btn, "Stop AI generation")
        Tooltip(self.voice_btn, "Voice output on/off")
        Tooltip(self.mic_btn, "Voice input (mic)")
        Tooltip(self._attach_btn, "Attach file or image")
        Tooltip(self._new_btn, "Start new conversation")
        Tooltip(self._web_btn, "AI web access on/off")
        Tooltip(self._reasoning_btn, "Chain-of-thought reasoning on/off")
        Tooltip(self._rag_btn, "Document Q&A — index files for context")
        Tooltip(self._edit_btn, "Edit last message")
        Tooltip(self._sidebar_toggle_btn, "Toggle sidebar")

        # Exit fullscreen button (hidden by default)
        self._exit_fullscreen_btn = themed_button(
            top, "\u2716 EXIT", style="danger",
            width=90, height=34,
            font=FONT_SMALL,
            fg_color=C_RED, text_color=C_TEXT_BRIGHT,
            command=self._exit_chat_fullscreen)
        # Not packed until fullscreen is entered

        # Add chat_col to paned window
        self._core_pane.add(chat_col, stretch="always",
                            minsize=300)

        # Right sidebar: history + system prompt + emotional state
        sidebar = ctk.CTkFrame(self._core_pane, fg_color="transparent")
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(0, weight=1)
        sidebar.grid_rowconfigure(1, weight=1)
        sidebar.grid_rowconfigure(2, weight=0)
        self._core_pane.add(sidebar, stretch="never",
                            minsize=180, width=320)

        # --- Collapsible HISTORY panel ---
        self._hist_panel = CollapsiblePanel(
            sidebar, "History", color=C_PURPLE,
            start_expanded=True,
            on_toggle=lambda _: self._rebalance_sidebar())
        self._hist_panel.grid(row=0, column=0, sticky="nsew",
                              pady=(0, 2))

        hist_frame = HUDFrame(
            self._hist_panel.content, glow_color=C_BORDER)
        hist_frame.pack(fill="both", expand=True)
        hist_frame.grid_columnconfigure(0, weight=1)
        hist_frame.grid_rowconfigure(0, weight=1)

        self.history_list = SelectableTextbox(
            hist_frame, font=FONT_SMALL, fg_color=C_PANEL,
            text_color=C_TEXT,
            border_width=0, corner_radius=2, wrap="word")
        self.history_list.grid(
            row=0, column=0, sticky="nsew", padx=4, pady=4)
        # Bind click to load session
        self.history_list._textbox.bind(
            "<Button-1>", self._on_history_click)

        hist_btns = ctk.CTkFrame(hist_frame, fg_color="transparent")
        hist_btns.grid(row=1, column=0, sticky="ew", padx=4, pady=4)

        _hist_tooltips = {
            "RENAME": "Rename selected session",
            "DELETE": "Delete selected session",
            "EXPORT": "Export chat (Markdown, JSON, or text)",
        }
        for text, color, cmd in [
                ("RENAME", C_TEXT_BRIGHT, self._rename_session),
                ("DELETE", C_RED, self._delete_session),
                ("EXPORT", C_TEXT_DIM, self._export_chat)]:
            btn = themed_button(
                hist_btns, text, style="secondary",
                width=60, height=30,
                font=FONT_TINY,
                text_color=color, command=cmd
            )
            btn.pack(side="left", padx=(0, 3))
            Tooltip(btn, _hist_tooltips[text])

        # Inline rename row (hidden until RENAME is clicked)
        self._rename_row = ctk.CTkFrame(
            hist_frame, fg_color="transparent")
        # Not gridded until _rename_session shows it
        self._rename_entry = ctk.CTkEntry(
            self._rename_row, height=28, font=FONT_TINY,
            fg_color=C_INPUT, border_color=C_ACCENT_DIM,
            border_width=1, text_color=C_TEXT_BRIGHT,
            corner_radius=2, placeholder_text="New name")
        wire_hotkeys(self._rename_entry)
        self._rename_entry.pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        _ok_rename = themed_button(
            self._rename_row, "OK", style="primary",
            width=40, height=28,
            font=FONT_TINY, command=self._confirm_rename)
        _ok_rename.pack(side="left", padx=(0, 2))
        _cancel_rename = themed_button(
            self._rename_row, "X", style="secondary",
            width=30, height=28,
            font=FONT_TINY, command=self._cancel_rename)
        _cancel_rename.pack(side="left")
        self._rename_entry.bind(
            "<Return>", lambda e: self._confirm_rename())
        self._rename_entry.bind(
            "<Escape>", lambda e: self._cancel_rename())

        # Inline delete confirmation row (hidden until DELETE is clicked)
        self._delete_confirm_row = ctk.CTkFrame(
            hist_frame, fg_color="transparent")
        # Not gridded until _delete_session shows it
        _del_label = SelectableLabel(
            self._delete_confirm_row, text="Delete?",
            font=FONT_TINY, text_color=C_RED, anchor="w")
        _del_label.pack(side="left", padx=(4, 4))
        _del_yes = themed_button(
            self._delete_confirm_row, "YES", style="danger",
            width=40, height=28,
            font=FONT_TINY, command=self._confirm_delete_session)
        _del_yes.pack(side="left", padx=(0, 2))
        _del_no = themed_button(
            self._delete_confirm_row, "NO", style="secondary",
            width=40, height=28,
            font=FONT_TINY, command=self._cancel_delete_session)
        _del_no.pack(side="left")

        # --- Collapsible SYSTEM PROMPT panel ---
        self._prompt_panel = CollapsiblePanel(
            sidebar, "System Prompt", color=C_ACCENT,
            start_expanded=True,
            on_toggle=lambda _: self._rebalance_sidebar())
        self._prompt_panel.grid(row=1, column=0, sticky="nsew",
                                pady=(2, 0))

        prompt_frame = HUDFrame(
            self._prompt_panel.content, glow_color=C_BORDER)
        prompt_frame.pack(fill="both", expand=True)
        prompt_frame.grid_columnconfigure(0, weight=1)
        prompt_frame.grid_rowconfigure(0, weight=1)

        self.prompt_editor = ctk.CTkTextbox(
            prompt_frame, font=FONT_SMALL, fg_color=C_PANEL,
            text_color=C_TEXT, border_width=0, corner_radius=2,
            wrap="word")
        self.prompt_editor.grid(
            row=0, column=0, sticky="nsew", padx=4, pady=4)
        wire_hotkeys(self.prompt_editor)

        default_prompt = self._load_system_prompt()
        self.prompt_editor.insert("1.0", default_prompt)

        prompt_btns = ctk.CTkFrame(prompt_frame, fg_color="transparent")
        prompt_btns.grid(row=1, column=0, sticky="ew", padx=4, pady=4)

        apply_btn = themed_button(
            prompt_btns, "APPLY", style="action",
            width=70, height=30,
            font=FONT_TINY, command=self._apply_system_prompt
        )
        apply_btn.pack(side="left", padx=(0, 4))
        Tooltip(apply_btn, "Apply prompt to current session")

        reset_btn = themed_button(
            prompt_btns, "RESET", style="secondary",
            width=70, height=30,
            font=FONT_TINY, command=self._reset_system_prompt
        )
        reset_btn.pack(side="left")
        Tooltip(reset_btn, "Reset prompt to default")

        # --- Collapsible EMOTIONAL STATE panel ---
        self._emo_panel = CollapsiblePanel(
            sidebar, "Emotional State", color=C_ORANGE,
            start_expanded=False,
            on_toggle=lambda _: self._rebalance_sidebar())
        self._emo_panel.grid(row=2, column=0, sticky="nsew",
                             pady=(2, 0))

        emo_frame = HUDFrame(
            self._emo_panel.content, glow_color=C_BORDER)
        emo_frame.pack(fill="x")
        emo_frame.grid_columnconfigure(0, weight=1)

        # Build one bar per emotional dimension
        self._emo_bars: dict[str, tuple[ctk.CTkLabel, ctk.CTkProgressBar]] = {}
        _emo_labels = {
            "valence": ("Valence", "Negative \u2190\u2192 Positive"),
            "arousal": ("Energy", "Calm \u2190\u2192 Energized"),
            "engagement": ("Engagement", "Detached \u2190\u2192 Engaged"),
            "trust": ("Trust", "Guarded \u2190\u2192 Open"),
            "frustration": ("Frustration", "Patient \u2190\u2192 Frustrated"),
        }
        for idx, (key, (label, tooltip_text)) in enumerate(
                _emo_labels.items()):
            row_frame = ctk.CTkFrame(emo_frame, fg_color="transparent")
            row_frame.grid(row=idx, column=0, sticky="ew",
                           padx=6, pady=(3, 0))
            row_frame.grid_columnconfigure(0, minsize=110)
            row_frame.grid_columnconfigure(1, weight=1)

            name_lbl = SelectableLabel(
                row_frame, text=label, font=FONT_TINY,
                text_color=C_TEXT_DIM, anchor="w")
            name_lbl.grid(row=0, column=0, sticky="w")

            bar = ctk.CTkProgressBar(
                row_frame, height=10, corner_radius=2,
                fg_color=C_SURFACE,
                progress_color=C_ORANGE,
                border_width=0)
            bar.grid(row=0, column=1, sticky="ew", padx=(4, 0))
            bar.set(0.0)

            val_lbl = ctk.CTkLabel(
                row_frame, text="0.0", font=("Consolas", 11),
                text_color=C_TEXT_DIM, width=36, anchor="e")
            val_lbl.grid(row=0, column=2, sticky="e", padx=(4, 0))
            Tooltip(row_frame, tooltip_text)

            self._emo_bars[key] = (val_lbl, bar)

        emo_btns = ctk.CTkFrame(emo_frame, fg_color="transparent")
        emo_btns.grid(row=len(_emo_labels), column=0,
                      sticky="ew", padx=6, pady=(4, 4))

        emo_reset_btn = themed_button(
            emo_btns, "RESET", style="secondary",
            width=60, height=28,
            font=FONT_TINY, command=self._reset_emotional_state)
        emo_reset_btn.pack(side="left")
        Tooltip(emo_reset_btn, "Reset emotional state to baseline")

        # Hide panel if user toggled it off in CONFIG
        if not self._read_gui_bool_setting(
                "show_emotional_state", True):
            self._emo_panel.grid_remove()

        # --- Collapsible JOURNAL panel ---
        self._journal_panel = CollapsiblePanel(
            sidebar, "Journal", color=C_PURPLE,
            start_expanded=False,
            on_toggle=lambda _: self._rebalance_sidebar())
        self._journal_panel.grid(row=3, column=0, sticky="nsew",
                                 pady=(2, 0))

        journal_frame = HUDFrame(
            self._journal_panel.content, glow_color=C_BORDER)
        journal_frame.pack(fill="both", expand=True)

        self._journal_display = SelectableTextbox(
            journal_frame, height=120, font=FONT_TINY,
            fg_color=C_SURFACE, text_color=C_TEXT_DIM,
            wrap="word")
        self._journal_display.pack(fill="both", expand=True,
                                   padx=4, pady=4)
        self._journal_display.configure(state="disabled")
        Tooltip(self._journal_panel.content,
                "AI reflections from idle periods.\n"
                "Enable in CONFIG > Monologue Mode.")

        # Store sidebar ref for rebalancing
        self._sidebar = sidebar
        self._sidebar_visible = True
        self._core_page = page
        self._rebalance_sidebar()

        # Populate journal display if model already loaded
        self._refresh_journal_display()

        self._refresh_history_list()
        self._chat_system(
            "Load a model from ROUTER, then start chatting.")

    def _toggle_sidebar(self):
        """Show or hide the CORE page sidebar.

        When hidden, the chat column expands to full width.
        When shown, the sidebar reappears with its panels.
        Uses PanedWindow to allow drag-to-resize.
        """
        if self._sidebar_visible:
            # Remove sidebar pane from the PanedWindow
            self._core_pane.forget(self._sidebar)
            self._sidebar_visible = False
            self._sidebar_toggle_btn.configure(text_color=C_TEXT_DIM)
        else:
            # Re-add sidebar pane to the PanedWindow
            self._core_pane.add(self._sidebar, stretch="never",
                                minsize=180, width=320)
            self._sidebar_visible = True
            self._sidebar_toggle_btn.configure(text_color=C_ACCENT)
            self._rebalance_sidebar()

    def _rebalance_sidebar(self):
        """Adjust sidebar row weights based on which panels are expanded.

        When both are expanded they share the space equally.
        When only one is expanded it takes all the space.
        When both are collapsed neither row stretches.
        """
        hist_open = self._hist_panel.is_expanded
        prompt_open = self._prompt_panel.is_expanded
        journal_open = getattr(self, "_journal_panel", None) and self._journal_panel.is_expanded
        self._sidebar.grid_rowconfigure(0, weight=1 if hist_open else 0)
        self._sidebar.grid_rowconfigure(1, weight=1 if prompt_open else 0)
        self._sidebar.grid_rowconfigure(2, weight=0)
        self._sidebar.grid_rowconfigure(3, weight=1 if journal_open else 0)

    def _update_emotional_display(self):
        """Refresh the emotional state bars from model_context."""
        bars = getattr(self, "_emo_bars", None)
        if not bars:
            return
        ctx = getattr(self, "model_context", None)
        if ctx is None:
            return
        from enigma_engine.core.model_context import _EMOTIONAL_RANGES
        state = ctx.emotional_state
        for key, (val_lbl, bar) in bars.items():
            raw = state.get(key, 0.0)
            lo, hi = _EMOTIONAL_RANGES.get(key, (0.0, 1.0))
            # Normalize to 0-1 for the progress bar
            span = hi - lo
            normalized = (raw - lo) / span if span > 0 else 0.0
            normalized = max(0.0, min(1.0, normalized))
            bar.set(normalized)
            val_lbl.configure(text=f"{raw:+.2f}" if lo < 0 else f"{raw:.2f}")

    def _reset_emotional_state(self):
        """Reset the model's emotional state and refresh the display."""
        ctx = getattr(self, "model_context", None)
        if ctx is None:
            return
        ctx.reset_emotional_state()
        self._update_emotional_display()

    def _set_emo_panel_visible(self, visible: bool):
        """Show or hide the emotional state sidebar panel."""
        panel = getattr(self, "_emo_panel", None)
        if panel is None:
            return
        if visible:
            panel.grid()
        else:
            panel.grid_remove()
        self._rebalance_sidebar()

    def _refresh_journal_display(self):
        """Populate the journal panel with recent entries."""
        display = getattr(self, "_journal_display", None)
        if display is None:
            return
        ctx = getattr(self, "model_context", None)
        if ctx is None:
            text = "(no model loaded)"
        else:
            try:
                journal = ctx.journal
                if journal.count == 0:
                    text = "(no journal entries yet)"
                else:
                    lines = []
                    for entry in journal.entries[-5:]:
                        ts = entry.get("timestamp", "")
                        # Show just the date portion
                        short_ts = ts[:10] if len(ts) >= 10 else ts
                        lines.append(
                            f"[{short_ts}] {entry.get('text', '')}")
                    text = "\n\n".join(lines)
            except Exception:
                text = "(journal unavailable)"
        display.configure(state="normal")
        display.delete("1.0", "end")
        display.insert("1.0", text)
        display.configure(state="disabled")

    def _toggle_chat_fullscreen(self):
        """Enter fullscreen chat mode — hides header, nav, status bar.

        The CORE page expands to cover the entire GUI window.
        Press Escape or click EXIT to return to normal layout.
        """
        if self._chat_fullscreen:
            self._exit_chat_fullscreen()
            return

        self._chat_fullscreen = True

        # Remember state to restore later
        self._fs_was_nav_expanded = self._nav_expanded
        self._fs_was_sidebar_visible = self._sidebar_visible

        # Hide shell elements
        self._header.pack_forget()
        self.status_bar.pack_forget()

        # Hide nav rail
        if self._nav_expanded:
            self._nav_frame.grid_remove()
            self._body.grid_columnconfigure(0, minsize=0)

        # Hide sidebar to maximize chat space
        if self._sidebar_visible:
            self._core_pane.forget(self._sidebar)
            self._sidebar_visible = False

        # Switch to CORE page if not already
        if self._current_page != "CORE":
            self._switch_page("CORE")

        # Show exit button and hide fullscreen button
        self._fullscreen_btn.pack_forget()
        self._sidebar_toggle_btn.pack_forget()
        self._exit_fullscreen_btn.pack(side="right", padx=(4, 0))
        self._fullscreen_btn.configure(text_color=C_ACCENT)

        # Bind Escape key to exit fullscreen
        self.bind("<Escape>", self._on_escape_fullscreen)

    def _exit_chat_fullscreen(self, _event=None):
        """Exit fullscreen chat mode — restores header, nav, status bar."""
        if not self._chat_fullscreen:
            return
        self._chat_fullscreen = False

        # Rebind Escape to stop-generation
        self._bind_escape_stop()

        # Hide exit button, restore fullscreen + sidebar toggle buttons
        # Pack order must match initial setup: fullscreen first (rightmost),
        # then sidebar toggle (to its left).
        self._exit_fullscreen_btn.pack_forget()
        self._fullscreen_btn.pack(side="right", padx=(4, 0))
        self._sidebar_toggle_btn.pack(side="right", padx=(4, 0))
        self._fullscreen_btn.configure(text_color=C_TEXT_DIM)

        # Restore header (must pack before body for correct order)
        self._header.pack(fill="x", side="top", before=self._body)

        # Restore status bar
        self.status_bar.pack(fill="x", side="bottom")

        # Restore nav rail
        if self._fs_was_nav_expanded:
            self._nav_frame.grid()
            self._body.grid_columnconfigure(
                0, minsize=self._nav_width)
            self._nav_expanded = True
            self._nav_toggle.configure(
                text="\u25c0", text_color=C_ACCENT)
        else:
            self._nav_expanded = False

        # Restore sidebar
        if self._fs_was_sidebar_visible:
            self._core_pane.add(self._sidebar, stretch="never",
                                minsize=180, width=320)
            self._sidebar_visible = True
            self._sidebar_toggle_btn.configure(text_color=C_ACCENT)
            self._rebalance_sidebar()

    def _on_escape_fullscreen(self, event):
        """Handle Escape key in fullscreen.

        If AI is generating, stop generation first.
        If not generating, exit fullscreen.
        """
        if getattr(self, '_is_generating', False):
            self._stop_generation()
            return
        self._exit_chat_fullscreen()

    # ================================================================
    # CORE - Chat input context menu
    # ================================================================

    def _chat_input_context_menu(self, event):
        """Show right-click context menu for multi-line chat input."""
        import tkinter as tk
        menu = tk.Menu(self, tearoff=0)
        tb = self.chat_input._textbox
        has_sel = bool(tb.tag_ranges("sel"))
        menu.add_command(
            label="Cut",
            state="normal" if has_sel else "disabled",
            command=lambda: (
                self.clipboard_clear(),
                self.clipboard_append(tb.get("sel.first", "sel.last")),
                tb.delete("sel.first", "sel.last"),
            ))
        menu.add_command(
            label="Copy",
            state="normal" if has_sel else "disabled",
            command=lambda: (
                self.clipboard_clear(),
                self.clipboard_append(tb.get("sel.first", "sel.last")),
            ))
        menu.add_command(
            label="Paste",
            command=self._paste_into_chat_input)
        menu.add_separator()
        menu.add_command(
            label="Select All",
            command=lambda: (
                tb.tag_add("sel", "1.0", "end-1c"),
                tb.mark_set("insert", "end-1c"),
            ))
        menu.tk_popup(event.x_root, event.y_root)

    def _paste_into_chat_input(self):
        """Paste clipboard text into chat input, replacing selection."""
        try:
            text = self.clipboard_get()
        except Exception:
            return
        tb = self.chat_input._textbox
        if tb.tag_ranges("sel"):
            tb.delete("sel.first", "sel.last")
        tb.insert("insert", text)

    # ================================================================
    # PAGE: MODELS - Model management (create / delete)
    # ================================================================

    def _build_page_models(self):
        page = self._make_page("MODELS")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(page, fg_color="transparent", height=48)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        SectionLabel(top, "Models").pack(
            side="left", fill="x", expand=True)

        content = HUDFrame(page, glow_color=C_BORDER)
        content.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)

        # Create new model form
        create_frame = ctk.CTkFrame(content, fg_color="transparent")
        create_frame.pack(fill="x", padx=8, pady=(8, 4))

        SectionLabel(create_frame, "New Model",
                     color=C_TEXT).pack(anchor="w", pady=(0, 4))

        form_row = ctk.CTkFrame(create_frame, fg_color="transparent")
        form_row.pack(fill="x")

        SelectableLabel(
            form_row, text="Name:", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 4))
        self.new_model_name = themed_entry(
            form_row, width=200, height=30, font=FONT_SMALL)
        self.new_model_name.pack(side="left", padx=(0, 8))

        SelectableLabel(
            form_row, text="Memory (GB):", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 4))
        # Auto-detect available memory (GPU VRAM or system RAM)
        _default_mem = "8"
        try:
            from enigma_engine.core.hardware_detection import (
                detect_hardware)
            _hw = detect_hardware()
            if _hw.gpu_available and _hw.gpu_vram_gb > 0:
                _default_mem = str(int(_hw.gpu_vram_gb))
            elif _hw.ram_gb > 0:
                _default_mem = str(int(_hw.ram_gb))
        except Exception:
            pass
        self.model_vram_var = ctk.StringVar(value=_default_mem)
        _mem_entry = themed_numeric_entry(
            form_row, mode="int",
            textvariable=self.model_vram_var,
            width=60, height=30, font=FONT_SMALL)
        _mem_entry.pack(side="left", padx=(0, 8))
        Tooltip(_mem_entry,
                "Available memory for training (GB).\n"
                "GPU VRAM if available, otherwise system RAM.\n"
                "The largest model that fits will be selected.")

        _create_btn = themed_button(
            form_row, "CREATE", style="action",
            width=90, height=30,
            font=FONT_SMALL, command=self._create_new_model)
        _create_btn.pack(side="left")
        Tooltip(_create_btn, "Create a new empty model")

        _import_btn = themed_button(
            form_row, "IMPORT", style="action",
            width=90, height=30,
            font=FONT_SMALL, command=self._import_model)
        _import_btn.pack(side="left", padx=(8, 0))
        Tooltip(_import_btn, "Import a model file from disk")

        _download_btn = themed_button(
            form_row, "DOWNLOAD", style="action",
            width=100, height=30,
            font=FONT_SMALL, command=self._download_huggingface)
        _download_btn.pack(side="left", padx=(8, 0))
        Tooltip(_download_btn, "Download a model from HuggingFace")

        # HuggingFace download row (inline entry for repo ID)
        hf_row = ctk.CTkFrame(create_frame, fg_color="transparent")
        hf_row.pack(fill="x", pady=(4, 0))
        SelectableLabel(
            hf_row, text="HuggingFace:", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 4))
        self._hf_repo_entry = themed_entry(
            hf_row, width=260, height=30, font=FONT_SMALL)
        self._hf_repo_entry.pack(side="left", padx=(0, 4))
        self._hf_repo_entry.configure(
            placeholder_text="e.g. gpt2 or username/model-name")
        self._hf_repo_entry.bind(
            "<Return>", lambda e: self._download_huggingface())

        # Status label for create/delete feedback (visible on this page)
        self._models_status = SelectableLabel(
            create_frame, text="", font=FONT_TINY,
            text_color=C_TEXT_DIM, anchor="w")
        self._models_status.pack(anchor="w", pady=(4, 0))

        # Separator
        ctk.CTkFrame(
            content, height=1, fg_color=C_ACCENT_DIM,
            corner_radius=0).pack(fill="x", padx=8, pady=4)

        # Model list (scrollable)
        scroll = themed_scroll(content)
        scroll.pack(fill="both", expand=True, padx=4, pady=4)
        scroll.grid_columnconfigure(0, weight=1)
        self.model_cards_frame = scroll

        if not self.models_data:
            SelectableLabel(
                scroll, text="No model files in models/",
                font=FONT_BODY, text_color=C_TEXT_DIM
            ).pack(anchor="w", padx=8, pady=20)
        else:
            self._populate_model_cards(scroll)

    # ================================================================
    # PAGE: ROUTER - Route assignments
    # ================================================================

    def _build_page_router(self):
        page = self._make_page("ROUTER")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(page, fg_color="transparent", height=48)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        SectionLabel(top, "Router").pack(
            side="left", fill="x", expand=True)

        self.unload_btn = ctk.CTkButton(
            top, text="UNLOAD", width=100, height=34,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_RED,
            text_color=C_TEXT_DIM, command=self._unload_model,
            state="disabled")
        self.unload_btn.pack(side="right")

        self.suspend_btn = ctk.CTkButton(
            top, text="SUSPEND", width=110, height=34,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT_DIM, command=self._suspend_model_memory,
            state="disabled")
        self.suspend_btn.pack(side="right", padx=(0, 6))

        # Route connections
        route_section = HUDFrame(page, glow_color=C_BORDER)
        route_section.grid(row=1, column=0, sticky="nsew",
                           padx=10, pady=4)

        SectionLabel(route_section, "Connections").pack(
            anchor="w", padx=8, pady=(8, 4))

        route_scroll = themed_scroll(route_section)
        route_scroll.pack(fill="both", expand=True, padx=4, pady=4)

        # Route labels stored for live updates
        self._route_labels: dict[str, tuple] = {}
        # Route dropdowns stored for reading selections
        self._route_menus: dict[str, ctk.CTkOptionMenu] = {}

        # Build model name list for route dropdowns
        model_names = ["None"] + [m["name"] for m in self.models_data]

        # Chat route - model for conversations
        self._build_route_card(
            route_scroll, "CHAT",
            "AI model for conversations",
            model_names)

        # Trainer route - teacher model in the Forge
        self._build_route_card(
            route_scroll, "TRAINER",
            "Teacher model that generates curriculum and tests the student",
            model_names)

        # Student route - the AI being trained
        self._build_route_card(
            route_scroll, "STUDENT",
            "AI model being trained and evaluated",
            model_names)

        # Per-mod routes
        for mod in self.mods_data:
            desc = mod.get("description", "")
            if not desc:
                desc = f"Mod module (port {mod.get('port', '?')})"
            # Use mod ID as route key (matches CMD page and route_assignments)
            self._build_route_card(
                route_scroll, mod["id"].upper(), desc,
                model_names)

    def _build_route_card(self, parent, name: str, desc: str,
                          model_names: list[str] | None = None):
        """Build a single route connection card with model selector."""
        card = HUDFrame(parent, glow_color=C_BORDER)
        card.pack(fill="x", padx=4, pady=3)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=10, pady=8)
        inner.grid_columnconfigure(1, weight=1)

        dot = StatusDot(inner, color=C_TEXT_DIM)
        dot.grid(row=0, column=0, rowspan=3, padx=(0, 8))

        SelectableLabel(
            inner, text=name, font=FONT_SECTION,
            text_color=C_TEXT_BRIGHT, anchor="w"
        ).grid(row=0, column=1, sticky="w")

        SelectableLabel(
            inner, text=desc, font=FONT_TINY,
            text_color=C_TEXT_DIM, anchor="w"
        ).grid(row=1, column=1, sticky="w")

        status_lbl = SelectableLabel(
            inner, text="No model", font=FONT_SMALL,
            text_color=C_TEXT_DIM)
        status_lbl.grid(row=0, column=2, rowspan=2, padx=(8, 0))

        # Store for status updates
        route_key = name.lower()
        self._route_labels[route_key] = (dot, status_lbl)

        # Model assignment dropdown
        if model_names and len(model_names) > 1:
            assign_row = ctk.CTkFrame(card, fg_color="transparent")
            assign_row.pack(fill="x", padx=10, pady=(0, 6))

            SelectableLabel(
                assign_row, text="Model:", font=FONT_TINY,
                text_color=C_TEXT_DIM
            ).pack(side="left", padx=(0, 4))

            menu = themed_dropdown(
                assign_row, model_names, width=220, height=30,
                command=lambda choice, rk=route_key: (
                    self._assign_model_to_route(rk, choice)))
            menu.set("None")
            menu.pack(side="left")
            self._route_menus[route_key] = menu

    def _populate_model_cards(self, parent):
        """Build a card for each model with identity info and actions."""
        from enigma_engine.core.model_context import (
            ModelContext, model_key_from_path,
        )
        # Native formats that support resize/training
        native_formats = {"pth", "pt"}

        for model in self.models_data:
            fmt = model["format"]
            is_native = fmt in native_formats

            # Load identity context for this model
            key = model_key_from_path(model["path"])
            ctx = ModelContext(key)
            ctx.load()

            card = HUDFrame(parent, glow_color=C_BORDER)
            card.pack(fill="x", padx=4, pady=3)

            inner = ctk.CTkFrame(card, fg_color="transparent")
            inner.pack(fill="x", padx=10, pady=8)
            inner.grid_columnconfigure(0, weight=1)

            # Row 0 — Model name (editable inline) + action buttons
            identity_name = ctx.display_name
            file_name = model["name"]
            param_str = model.get("params")

            if identity_name:
                title_text = identity_name
            else:
                title_text = file_name

            # Inline-editable name entry (looks like a label
            # until EDIT is clicked)
            name_entry = ctk.CTkEntry(
                inner, font=FONT_SECTION, height=32,
                fg_color="transparent", border_width=0,
                text_color=C_TEXT_BRIGHT, corner_radius=2,
                state="disabled")
            name_entry.grid(row=0, column=0, sticky="ew")
            # Insert text while enabled, then disable
            name_entry.configure(state="normal")
            name_entry.insert(0, title_text)
            name_entry.configure(state="disabled")

            # Param count and RAM shown in the info row below,
            # not next to the name

            btn_frame = ctk.CTkFrame(inner, fg_color="transparent")
            btn_frame.grid(row=0, column=1, padx=(8, 0), sticky="e")

            # Inline save/cancel row (hidden until EDIT clicked)
            edit_row = ctk.CTkFrame(inner, fg_color="transparent")
            _save_btn = themed_button(
                edit_row, "SAVE", style="primary",
                width=50, height=24,
                font=FONT_TINY,
                command=lambda e=name_entry, m=model,
                er=edit_row: self._save_inline_name(e, m, er))
            _save_btn.pack(side="left", padx=(0, 4))
            _cancel_btn = themed_button(
                edit_row, "CANCEL", style="secondary",
                width=60, height=24,
                font=FONT_TINY,
                command=lambda e=name_entry, m=model,
                er=edit_row: self._cancel_inline_name(e, m, er))
            _cancel_btn.pack(side="left")
            # Bind Enter/Escape while editing
            name_entry.bind(
                "<Return>",
                lambda ev, e=name_entry, m=model,
                er=edit_row: self._save_inline_name(e, m, er))
            name_entry.bind(
                "<Escape>",
                lambda ev, e=name_entry, m=model,
                er=edit_row: self._cancel_inline_name(e, m, er))

            _edit = themed_button(
                btn_frame, "EDIT", style="action",
                width=65, height=28,
                font=FONT_TINY,
                command=lambda e=name_entry,
                er=edit_row: self._start_inline_edit(e, er))
            _edit.pack(side="left", padx=(0, 4))
            Tooltip(_edit, "Edit display name")

            _export = themed_button(
                btn_frame, "EXPORT CARD", style="secondary",
                width=95, height=28,
                font=FONT_TINY,
                command=lambda m=model: self._export_identity(m))
            _export.pack(side="left", padx=(0, 4))
            Tooltip(_export, "Export identity card to JSON")

            _copy = themed_button(
                btn_frame, "COPY", style="secondary",
                width=70, height=28,
                font=FONT_TINY,
                command=lambda m=model: self._copy_model(m))
            _copy.pack(side="left", padx=(0, 4))
            Tooltip(_copy, "Duplicate this model")

            if is_native:
                _grow = themed_button(
                    btn_frame, "GROW", style="primary",
                    width=70, height=28,
                    font=FONT_TINY,
                    command=lambda m=model: self._grow_model(m))
                _grow.pack(side="left", padx=(0, 4))
                Tooltip(_grow,
                        "Expand this model to a larger size "
                        "(progressive growing)")

            _delete = themed_button(
                btn_frame, "DELETE", style="secondary",
                width=75, height=28,
                font=FONT_TINY,
                hover_color=C_RED,
                command=lambda m=model: self._delete_model(m))
            _delete.pack(side="left")
            Tooltip(_delete, "Permanently delete this model")

            # Right-click context menu on the card
            def _show_card_menu(event, m=model, e=name_entry,
                                er=edit_row):
                import tkinter as tk
                menu = tk.Menu(self, tearoff=0)
                menu.add_command(
                    label="Rename file",
                    command=lambda: self._start_file_rename(e, m, er))
                menu.add_command(
                    label="Delete",
                    command=lambda: self._delete_model(m))
                menu.tk_popup(event.x_root, event.y_root)

            card.bind("<Button-3>", _show_card_menu)
            inner.bind("<Button-3>", _show_card_menu)
            name_entry.bind(
                "<Button-3>", _show_card_menu)

            # Row 3 — Checkpoint info (count + clean button)
            from enigma_engine.gui.scanners import MODELS_DIR
            model_stem = Path(model["path"]).stem
            ckpt_dir = MODELS_DIR / "checkpoints" / model_stem
            ckpt_count = 0
            protected_count = 0
            if ckpt_dir.is_dir():
                for cf in ckpt_dir.iterdir():
                    if cf.suffix == ".pt":
                        ckpt_count += 1
                        keep_f = cf.parent / (cf.name + ".keep")
                        if keep_f.exists():
                            protected_count += 1

            if ckpt_count > 0:
                ckpt_frame = ctk.CTkFrame(
                    inner, fg_color="transparent")
                ckpt_frame.grid(
                    row=3, column=0, columnspan=2,
                    sticky="w", pady=(4, 0))

                prot_info = (
                    f"  ({protected_count} protected)"
                    if protected_count else "")
                SelectableLabel(
                    ckpt_frame,
                    text=f"Checkpoints: {ckpt_count}{prot_info}",
                    font=FONT_TINY, text_color=C_TEXT_DIM,
                    anchor="w"
                ).pack(side="left", padx=(0, 8))

                themed_button(
                    ckpt_frame, "CLEAN", style="secondary",
                    width=60, height=22, font=FONT_TINY,
                    command=lambda d=ckpt_dir: self._clean_checkpoints(d)
                ).pack(side="left", padx=(0, 4))

            # Row 4 — Inline delete confirmation (hidden)
            delete_row = ctk.CTkFrame(inner, fg_color="transparent")
            _del_msg = SelectableLabel(
                delete_row, text=f"Delete {model['name']}?",
                font=FONT_TINY, text_color=C_RED, anchor="w")
            _del_msg.pack(side="left", padx=(0, 6))
            _del_yes = ctk.CTkButton(
                delete_row, text="YES", width=45, height=24,
                font=FONT_TINY, corner_radius=2,
                fg_color=C_RED_DIM, hover_color=C_RED_HOVER,
                text_color=C_RED,
                command=lambda m=model,
                dr=delete_row: self._confirm_delete_model(m, dr))
            _del_yes.pack(side="left", padx=(0, 4))
            _del_no = ctk.CTkButton(
                delete_row, text="NO", width=45, height=24,
                font=FONT_TINY, corner_radius=2,
                fg_color=C_SURFACE, hover_color=C_BORDER,
                text_color=C_TEXT_DIM,
                command=lambda dr=delete_row: dr.grid_forget())
            _del_no.pack(side="left")
            # Store references so _delete_model can find the right card
            card._delete_row = delete_row
            delete_row._model = model

            # Row 1 — Tag + format info + file name when identity name differs
            tag = "NATIVE" if is_native else "EXTERNAL"
            tag_color = C_GREEN if is_native else C_ORANGE
            # Estimate RAM usage from file size
            size_mb = model['size_mb']
            if size_mb >= 1024:
                ram_str = f"~{size_mb / 1024:.1f} GB RAM"
            else:
                ram_str = f"~{size_mb:.0f} MB RAM"
            if param_str:
                from enigma_engine.gui.scanners import (
                    _format_param_count)
                formatted = _format_param_count(param_str)
                info = (f"{fmt.upper()}  //  {formatted} params"
                        f"  //  {ram_str}")
            else:
                info = f"{fmt.upper()}  //  {ram_str}"

            # Append preset name if saved in context
            if ctx.preset_name:
                info += f"  //  {ctx.preset_name}"

            info_frame = ctk.CTkFrame(inner, fg_color="transparent")
            info_frame.grid(row=1, column=0, columnspan=2,
                            sticky="w", pady=(2, 0))

            SelectableLabel(
                info_frame, text=tag, font=FONT_TINY,
                text_color=tag_color, anchor="w"
            ).pack(side="left", padx=(0, 6))

            SelectableLabel(
                info_frame, text=info, font=FONT_TINY,
                text_color=C_TEXT_DIM, anchor="w"
            ).pack(side="left")

            # Show file name as subtitle if identity name is set
            if identity_name:
                SelectableLabel(
                    info_frame, text=f"  ({file_name})",
                    font=FONT_TINY, text_color=C_TEXT_DIM, anchor="w"
                ).pack(side="left")

            # Row 2 — Identity info (personality, stats, tags)
            has_identity = (
                ctx.personality or ctx.total_messages > 0
                or ctx.tags or ctx.training_history
            )
            if has_identity:
                id_frame = ctk.CTkFrame(inner, fg_color="transparent")
                id_frame.grid(row=2, column=0, columnspan=2,
                              sticky="w", pady=(4, 0))

                if ctx.personality:
                    SelectableLabel(
                        id_frame, text=ctx.personality,
                        font=FONT_TINY, text_color=C_TEXT,
                        anchor="w"
                    ).pack(anchor="w")

                # Stats line: messages, sessions, training runs
                stats_parts = []
                if ctx.total_messages > 0:
                    stats_parts.append(
                        f"{ctx.total_messages} messages")
                if ctx.total_sessions > 0:
                    stats_parts.append(
                        f"{ctx.total_sessions} sessions")
                if ctx.training_history:
                    stats_parts.append(
                        f"{len(ctx.training_history)} training runs")
                if stats_parts:
                    SelectableLabel(
                        id_frame, text="  //  ".join(stats_parts),
                        font=FONT_TINY, text_color=C_TEXT_DIM,
                        anchor="w"
                    ).pack(anchor="w")

                # Tags row
                if ctx.tags:
                    tags_text = "  ".join(
                        f"[{t}]" for t in ctx.tags)
                    SelectableLabel(
                        id_frame, text=tags_text,
                        font=FONT_TINY, text_color=C_ACCENT,
                        anchor="w"
                    ).pack(anchor="w")

    # ================================================================
    # Identity editing — inline name field
    # ================================================================

    def _start_inline_edit(self, entry, edit_row):
        """Activate inline name editing on a model card."""
        entry.configure(
            state="normal",
            fg_color=C_INPUT,
            border_width=1,
            border_color=C_ACCENT)
        entry.focus_set()
        entry.select_range(0, "end")
        edit_row.grid(
            row=3, column=0, columnspan=2,
            sticky="w", pady=(4, 0))

    def _start_file_rename(self, entry, model, edit_row):
        """Activate inline file rename on a model card.

        Similar to display-name editing but saves to the filesystem
        via _rename_model() instead of ModelContext.
        """
        src = Path(model["path"])
        # Show the actual filename stem for file rename
        entry.configure(state="normal")
        entry.delete(0, "end")
        entry.insert(0, src.stem)
        entry.configure(
            fg_color=C_INPUT,
            border_width=1,
            border_color=C_ORANGE)
        entry.focus_set()
        entry.select_range(0, "end")
        # Tag the entry so save knows this is a file rename
        entry._file_rename_model = model
        edit_row.grid(
            row=3, column=0, columnspan=2,
            sticky="w", pady=(4, 0))

    def _save_inline_name(self, entry, model, edit_row):
        """Save the edited name — display name or file rename."""
        # Check if this is a file rename (set by _start_file_rename)
        file_rename_model = getattr(entry, "_file_rename_model", None)
        if file_rename_model is not None:
            new_name = entry.get().strip()
            entry._file_rename_model = None
            # Return to read-only appearance first
            entry.configure(
                state="disabled",
                fg_color=C_PANEL,
                border_width=0,
                border_color=C_BORDER)
            edit_row.grid_forget()
            # Delegate to the actual file rename logic
            self._rename_model(file_rename_model, new_name=new_name)
            return

        # Normal display-name edit flow
        from enigma_engine.core.model_context import (
            ModelContext, model_key_from_path,
        )
        new_name = entry.get().strip()
        key = model_key_from_path(model["path"])
        ctx = ModelContext(key)
        ctx.load()

        # If the name matches the filename, clear display_name
        file_stem = Path(model["path"]).stem
        if new_name == file_stem:
            ctx.display_name = ""
        else:
            ctx.display_name = new_name
        ctx.save()

        # Return to read-only appearance
        entry.configure(
            state="disabled",
            fg_color=C_PANEL,
            border_width=0)
        edit_row.grid_forget()
        self.status_bar.set_left(
            f"Name updated for {key}")

    def _cancel_inline_name(self, entry, model, edit_row):
        """Cancel editing and restore original display name."""
        # Clear file rename tag if set
        entry._file_rename_model = None
        from enigma_engine.core.model_context import (
            ModelContext, model_key_from_path,
        )
        key = model_key_from_path(model["path"])
        ctx = ModelContext(key)
        ctx.load()

        original = ctx.display_name or model["name"]
        entry.configure(state="normal")
        entry.delete(0, "end")
        entry.insert(0, original)
        entry.configure(
            state="disabled",
            fg_color=C_PANEL,
            border_width=0,
            border_color=C_BORDER)
        edit_row.grid_forget()

    def _export_identity(self, model: dict):
        """Export a model's identity card to a JSON file."""
        from tkinter import filedialog
        from enigma_engine.core.model_context import (
            ModelContext, model_key_from_path,
        )
        import json

        key = model_key_from_path(model["path"])
        ctx = ModelContext(key)
        ctx.load()

        data = ctx.export_identity()
        default_name = f"{key}_identity.json"

        path = filedialog.asksaveasfilename(
            title="Export Identity Card",
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            parent=self)
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.status_bar.set_left(
                f"\u2728 Identity exported: {Path(path).name}")
        except OSError as exc:
            err_msg = str(exc)
            self.status_bar.set_left(f"Export failed: {err_msg}")

