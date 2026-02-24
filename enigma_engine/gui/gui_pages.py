"""
Enigma Engine - GUI Page Builders
===================================

Mixin providing page construction methods for EnigmaGUI.
Each page is a separate method building its widget tree.

Pages: CORE, MODELS, ROUTER, FORGE, CONFIG
"""
from __future__ import annotations

import customtkinter as ctk

from enigma_engine.gui.widgets import (
    C_ACCENT, C_ACCENT_DIM, C_ACCENT_MUTED, C_BG, C_BORDER,
    C_BORDER_ACCENT, C_CYAN, C_GREEN, C_GREEN_DIM, C_INPUT,
    C_PANEL, C_PURPLE, C_RED, C_SURFACE, C_TEXT, C_TEXT_BRIGHT,
    C_TEXT_DIM,
    FONT_BODY, FONT_CHAT, FONT_INPUT, FONT_MONO,
    FONT_SECTION, FONT_SMALL, FONT_TINY,
    CollapsiblePanel, HUDFrame, SectionLabel, SelectableTextbox,
    StatusDot, ToggleButton, Tooltip,
    themed_entry, themed_dropdown, themed_scroll,
)
from enigma_engine.gui.scanners import (
    CONFIG_DESCRIPTIONS, CONFIG_DISPLAY_NAMES, CONFIG_LIMITS,
    PATH_SETTINGS,
)
# Re-export so existing imports keep working
from enigma_engine.gui.gui_brick_page import BrickPageMixin  # noqa: F401


class PagesMixin:
    """Mixin providing page builder methods for EnigmaGUI.

    Expects the host class to have attributes set in __init__:
    - profiles_data, models_data, training_files, bricks_data
    - config_overrides, config_entries
    - Various widget references (set during page construction)
    """

    # ================================================================
    # PAGE: CORE - Chat + History Sidebar
    # ================================================================

    def _build_page_core(self):
        page = self._make_page("CORE")
        page.grid_columnconfigure(0, weight=3)
        page.grid_columnconfigure(1, weight=1, minsize=200)
        page.grid_rowconfigure(1, weight=1)

        # Top bar: label + profile
        top = ctk.CTkFrame(page, fg_color="transparent", height=48)
        top.grid(row=0, column=0, columnspan=2, sticky="ew",
                 padx=10, pady=(8, 2))

        SectionLabel(top, "Neural Interface").pack(
            side="left", fill="x", expand=True)

        # Fullscreen toggle button
        self._fullscreen_btn = ctk.CTkButton(
            top, text="\u26f6", width=38, height=34,
            font=FONT_SECTION, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT_DIM, command=self._toggle_chat_fullscreen)
        self._fullscreen_btn.pack(side="right", padx=(4, 0))
        Tooltip(self._fullscreen_btn, "Fullscreen chat")

        # Sidebar toggle button
        self._sidebar_toggle_btn = ctk.CTkButton(
            top, text="\u25e8", width=38, height=34,
            font=FONT_SECTION, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_ACCENT, command=self._toggle_sidebar)
        self._sidebar_toggle_btn.pack(side="right", padx=(4, 0))

        # Active profile indicator (managed from DOCS page)
        self._core_profile_label = ctk.CTkLabel(
            top, text="", font=FONT_TINY,
            text_color=C_PURPLE)
        self._core_profile_label.pack(side="right", padx=(0, 8))

        # Left column: chat + input
        chat_col = ctk.CTkFrame(page, fg_color="transparent")
        chat_col.grid(row=1, column=0, sticky="nsew",
                      padx=(10, 4), pady=(2, 8))
        chat_col.grid_columnconfigure(0, weight=1)
        chat_col.grid_rowconfigure(0, weight=1)

        # Chat display
        chat_frame = HUDFrame(chat_col, glow_color=C_BORDER)
        chat_frame.grid(row=0, column=0, sticky="nsew")
        chat_frame.grid_columnconfigure(0, weight=1)
        chat_frame.grid_rowconfigure(0, weight=1)

        self.chat_display = SelectableTextbox(
            chat_frame, wrap="word",
            font=FONT_CHAT, fg_color=C_PANEL, text_color=C_TEXT,
            border_width=0, corner_radius=2)
        self.chat_display.grid(
            row=0, column=0, sticky="nsew", padx=4, pady=4)

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

        # Input area
        input_area = ctk.CTkFrame(chat_col, fg_color="transparent")
        input_area.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        input_area.grid_columnconfigure(0, weight=1)

        # File attachment indicator (blue)
        self.file_indicator = ctk.CTkLabel(
            input_area, text="", font=FONT_TINY,
            text_color=C_ACCENT, height=20)
        self.file_indicator.grid(
            row=0, column=0, sticky="w", padx=2)

        # Thinking indicator (fixed width so animation
        # does not shift the layout as dots change)
        self.thinking_label = ctk.CTkLabel(
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
        self.chat_input.bind("<Return>", self._on_input_enter)
        self.chat_input.bind("<Shift-Return>", lambda e: None)

        # SEND button to the right of the input
        self.send_btn = ctk.CTkButton(
            input_area, text="SEND", width=80, height=56,
            font=FONT_SECTION, corner_radius=2,
            fg_color=C_GREEN_DIM, hover_color="#1a5a2a",
            text_color=C_GREEN, command=self._send_message)
        self.send_btn.grid(row=1, column=1, sticky="ns")

        # Utility toolbar below the input (voice, mic, attach, new)
        toolbar = ctk.CTkFrame(input_area, fg_color="transparent")
        toolbar.grid(row=2, column=0, columnspan=2, sticky="w",
                     pady=(4, 0))

        self.voice_btn = ToggleButton(
            toolbar, text_on="\U0001f50a", text_off="\U0001f507",
            on_toggle=self._on_voice_toggle, start_on=False,
            width=38, height=32)
        self.voice_btn.pack(side="left", padx=(0, 4))

        self.mic_btn = ctk.CTkButton(
            toolbar, text="\U0001f3a4", width=38, height=32,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT_DIM, command=self._toggle_voice_input)
        self.mic_btn.pack(side="left", padx=(0, 4))

        self._attach_btn = ctk.CTkButton(
            toolbar, text="\U0001f4ce", width=38, height=32,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_TEXT_DIM, command=self._attach_file)
        self._attach_btn.pack(side="left", padx=(0, 4))

        self._new_btn = ctk.CTkButton(
            toolbar, text="NEW", width=60, height=32,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_BORDER,
            text_color=C_TEXT_DIM, command=self._new_chat)
        self._new_btn.pack(side="left", padx=(0, 4))

        self._web_btn = ctk.CTkButton(
            toolbar, text="\U0001f310", width=38, height=32,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            text_color=C_CYAN, command=self._web_search_dialog)
        self._web_btn.pack(side="left")

        # Tooltips for CORE page buttons
        Tooltip(self.send_btn, "Send message (Enter)")
        Tooltip(self.voice_btn, "Voice output on/off")
        Tooltip(self.mic_btn, "Voice input (mic)")
        Tooltip(self._attach_btn, "Attach file")
        Tooltip(self._new_btn, "Start new conversation")
        Tooltip(self._web_btn, "Web search")
        Tooltip(self._sidebar_toggle_btn, "Toggle sidebar")

        # Exit fullscreen button (hidden by default)
        self._exit_fullscreen_btn = ctk.CTkButton(
            top, text="\u2716 EXIT", width=90, height=34,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_RED, hover_color="#b91c1c",
            text_color=C_TEXT_BRIGHT,
            command=self._exit_chat_fullscreen)
        # Not packed until fullscreen is entered

        # Right sidebar: history + system prompt
        sidebar = ctk.CTkFrame(page, fg_color="transparent")
        sidebar.grid(row=1, column=1, sticky="nsew",
                     padx=(4, 10), pady=(2, 8))
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(0, weight=1)
        sidebar.grid_rowconfigure(1, weight=1)

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

        hist_btns = ctk.CTkFrame(hist_frame, fg_color="transparent")
        hist_btns.grid(row=1, column=0, sticky="ew", padx=4, pady=4)

        for text, color, cmd in [
                ("SAVE", C_TEXT_BRIGHT, self._save_session),
                ("LOAD", C_TEXT_DIM, self._load_session),
                ("EXPORT", C_TEXT_DIM, self._export_chat)]:
            ctk.CTkButton(
                hist_btns, text=text, width=70, height=30,
                font=FONT_TINY, corner_radius=2,
                fg_color=C_SURFACE, hover_color=C_BORDER,
                text_color=color, command=cmd
            ).pack(side="left", padx=(0, 4))

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

        default_prompt = self._load_system_prompt()
        self.prompt_editor.insert("1.0", default_prompt)

        prompt_btns = ctk.CTkFrame(prompt_frame, fg_color="transparent")
        prompt_btns.grid(row=1, column=0, sticky="ew", padx=4, pady=4)

        ctk.CTkButton(
            prompt_btns, text="APPLY", width=70, height=30,
            font=FONT_TINY, corner_radius=2,
            fg_color=C_ACCENT_DIM, hover_color=C_ACCENT_MUTED,
            text_color=C_ACCENT, command=self._apply_system_prompt
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            prompt_btns, text="RESET", width=70, height=30,
            font=FONT_TINY, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_BORDER,
            text_color=C_TEXT_DIM, command=self._reset_system_prompt
        ).pack(side="left")

        # Store sidebar ref for rebalancing
        self._sidebar = sidebar
        self._sidebar_visible = True
        self._core_page = page
        self._rebalance_sidebar()

        self._refresh_history_list()
        self._chat_system(
            "Load a model from ROUTER, then start chatting.")

    def _toggle_sidebar(self):
        """Show or hide the CORE page sidebar.

        When hidden, the chat column expands to full width.
        When shown, the sidebar reappears with its panels.
        """
        if self._sidebar_visible:
            self._sidebar.grid_forget()
            self._core_page.grid_columnconfigure(1, weight=0, minsize=0)
            self._sidebar_visible = False
            self._sidebar_toggle_btn.configure(text_color=C_TEXT_DIM)
        else:
            self._sidebar.grid(
                row=1, column=1, sticky="nsew",
                padx=(4, 10), pady=(2, 8))
            self._core_page.grid_columnconfigure(1, weight=1, minsize=200)
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
        self._sidebar.grid_rowconfigure(0, weight=1 if hist_open else 0)
        self._sidebar.grid_rowconfigure(1, weight=1 if prompt_open else 0)

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
            self._sidebar.grid_forget()
            self._core_page.grid_columnconfigure(
                1, weight=0, minsize=0)
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

        # Unbind Escape
        self.unbind("<Escape>")

        # Hide exit button, restore fullscreen + sidebar toggle buttons
        self._exit_fullscreen_btn.pack_forget()
        self._sidebar_toggle_btn.pack(side="right", padx=(4, 0))
        self._fullscreen_btn.pack(side="right", padx=(4, 0))
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
            self._sidebar.grid(
                row=1, column=1, sticky="nsew",
                padx=(4, 10), pady=(2, 8))
            self._core_page.grid_columnconfigure(
                1, weight=1, minsize=200)
            self._sidebar_visible = True
            self._sidebar_toggle_btn.configure(text_color=C_ACCENT)
            self._rebalance_sidebar()

    def _on_escape_fullscreen(self, event):
        """Handle Escape key to exit fullscreen."""
        self._exit_chat_fullscreen()

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

        ctk.CTkLabel(
            form_row, text="Name:", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 4))
        self.new_model_name = themed_entry(
            form_row, width=160, height=30, font=FONT_SMALL)
        self.new_model_name.pack(side="left", padx=(0, 8))

        ctk.CTkLabel(
            form_row, text="Size:", font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(side="left", padx=(0, 4))
        self.new_model_size = themed_dropdown(
            form_row,
            ["pi_zero", "nano", "tiny", "small",
             "medium", "large"],
            width=120, height=30)
        self.new_model_size.set("small")
        self.new_model_size.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            form_row, text="CREATE", width=90, height=30,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_ACCENT_DIM, hover_color=C_ACCENT_MUTED,
            text_color=C_ACCENT, command=self._create_new_model
        ).pack(side="left")

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
            ctk.CTkLabel(
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

        # Trainer route - model to be trained in the Forge
        self._build_route_card(
            route_scroll, "TRAINER",
            "Target model for training in the Forge",
            model_names)

        # Per-brick routes
        for brick in self.bricks_data:
            desc = brick.get("description", "")
            if not desc:
                desc = f"Brick module (port {brick.get('port', '?')})"
            self._build_route_card(
                route_scroll, brick["name"].upper(), desc,
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

        ctk.CTkLabel(
            inner, text=name, font=FONT_SECTION,
            text_color=C_TEXT_BRIGHT, anchor="w"
        ).grid(row=0, column=1, sticky="w")

        ctk.CTkLabel(
            inner, text=desc, font=FONT_TINY,
            text_color=C_TEXT_DIM, anchor="w"
        ).grid(row=1, column=1, sticky="w")

        status_lbl = ctk.CTkLabel(
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

            ctk.CTkLabel(
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
        for model in self.models_data:
            card = HUDFrame(parent, glow_color=C_BORDER)
            card.pack(fill="x", padx=4, pady=3)

            inner = ctk.CTkFrame(card, fg_color="transparent")
            inner.pack(fill="x", padx=10, pady=8)
            inner.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                inner, text=model["name"], font=FONT_SECTION,
                text_color=C_TEXT_BRIGHT, anchor="w"
            ).grid(row=0, column=0, sticky="w")

            info = f"{model['format'].upper()}  //  {model['size_mb']} MB"
            ctk.CTkLabel(
                inner, text=info, font=FONT_TINY,
                text_color=C_TEXT_DIM, anchor="w"
            ).grid(row=1, column=0, sticky="w")

            ctk.CTkButton(
                inner, text="DELETE", width=80, height=34,
                font=FONT_SMALL, corner_radius=2,
                fg_color=C_SURFACE, hover_color=C_RED,
                text_color=C_TEXT_DIM,
                command=lambda m=model: self._delete_model(m)
            ).grid(row=0, column=1, rowspan=2, padx=(8, 0),
                   sticky="e")

    # ================================================================
    # PAGE: FORGE - Training
    # ================================================================

    def _build_page_forge(self):
        page = self._make_page("FORGE")
        page.grid_columnconfigure(0, weight=1)
        page.grid_columnconfigure(1, weight=1)
        page.grid_columnconfigure(2, weight=1)
        page.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(page, fg_color="transparent", height=48)
        top.grid(row=0, column=0, columnspan=3, sticky="ew",
                 padx=10, pady=(8, 2))
        SectionLabel(top, "The Forge").pack(
            side="left", fill="x", expand=True)

        # Left: controls
        controls = HUDFrame(page, glow_color=C_BORDER)
        controls.grid(row=1, column=0, sticky="nsew",
                      padx=(10, 4), pady=4)
        ctrl_scroll = themed_scroll(controls)
        ctrl_scroll.pack(fill="both", expand=True, padx=4, pady=4)

        self._forge_heading(ctrl_scroll, "TRAIN MODEL")

        self._forge_label(ctrl_scroll, "Data source")
        self.train_data_var = ctk.StringVar(
            value=self.training_files[0]["path"]
            if self.training_files else "")
        data_opts = [
            f"{f['name']} ({f['size_kb']} KB)"
            for f in self.training_files]
        if data_opts:
            self.train_data_menu = themed_dropdown(
                ctrl_scroll, data_opts, width=280,
                command=self._on_data_selected)
            self.train_data_menu.pack(anchor="w", padx=10, pady=(0, 6))
        else:
            ctk.CTkLabel(
                ctrl_scroll, text="No data files in data/",
                font=FONT_TINY, text_color=C_TEXT_DIM
            ).pack(anchor="w", padx=10, pady=(0, 6))

        self._forge_label(ctrl_scroll, "Model size")
        self.model_size_var = ctk.StringVar(value="small")
        themed_dropdown(
            ctrl_scroll,
            ["pi_zero", "nano", "tiny", "small",
             "medium", "large"],
            variable=self.model_size_var
        ).pack(anchor="w", padx=10, pady=(0, 6))

        self._forge_label(ctrl_scroll, "Epochs")
        self.epochs_entry = self._forge_entry(ctrl_scroll, "10")

        self._forge_label(ctrl_scroll, "Batch size")
        self.batch_entry = self._forge_entry(ctrl_scroll, "4")

        self._forge_label(ctrl_scroll, "Learning rate")
        self.lr_entry = self._forge_entry(ctrl_scroll, "0.0001")

        btn_row = ctk.CTkFrame(ctrl_scroll, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(8, 4))

        self.train_model_btn = ctk.CTkButton(
            btn_row, text="START TRAINING", width=170, height=38,
            font=FONT_SECTION, corner_radius=2,
            fg_color=C_ACCENT_DIM, hover_color=C_ACCENT_MUTED,
            text_color=C_ACCENT, command=self._start_model_training)
        self.train_model_btn.pack(side="left")

        self.stop_train_btn = ctk.CTkButton(
            btn_row, text="STOP", width=80, height=38,
            font=FONT_SECTION, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_RED,
            text_color=C_TEXT_DIM, command=self._stop_training,
            state="disabled")
        self.stop_train_btn.pack(side="left", padx=(6, 0))

        self._forge_heading(ctrl_scroll, "TRAIN TOKENIZER")
        self._forge_label(ctrl_scroll, "Vocabulary size")
        self.vocab_entry = self._forge_entry(ctrl_scroll, "8000")

        self.train_tok_btn = ctk.CTkButton(
            ctrl_scroll, text="TRAIN TOKENIZER", width=180, height=38,
            font=FONT_SECTION, corner_radius=2,
            fg_color=C_ACCENT_DIM, hover_color=C_ACCENT_MUTED,
            text_color=C_ACCENT, command=self._start_tokenizer_training)
        self.train_tok_btn.pack(anchor="w", padx=10, pady=(6, 10))

        # Center: data editor
        editor_panel = HUDFrame(page, glow_color=C_BORDER)
        editor_panel.grid(row=1, column=1, sticky="nsew",
                          padx=4, pady=4)

        editor_top = ctk.CTkFrame(
            editor_panel, fg_color="transparent")
        editor_top.pack(fill="x", padx=8, pady=(8, 2))
        SectionLabel(editor_top, "Data Editor",
                     color=C_CYAN).pack(
            side="left", fill="x", expand=True)

        self.data_file_label = ctk.CTkLabel(
            editor_panel, text="No file selected",
            font=FONT_TINY, text_color=C_TEXT_DIM)
        self.data_file_label.pack(anchor="w", padx=10,
                                  pady=(0, 2))

        self.data_editor = ctk.CTkTextbox(
            editor_panel, font=FONT_MONO, fg_color=C_INPUT,
            text_color=C_TEXT, border_width=0,
            corner_radius=2, wrap="word")
        self.data_editor.pack(fill="both", expand=True,
                              padx=6, pady=(2, 4))

        editor_btns = ctk.CTkFrame(
            editor_panel, fg_color="transparent")
        editor_btns.pack(fill="x", padx=8, pady=(0, 6))

        ctk.CTkButton(
            editor_btns, text="SAVE", width=80, height=30,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_GREEN_DIM, hover_color="#1a5a2a",
            text_color=C_GREEN, command=self._save_data_file
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            editor_btns, text="NEW FILE", width=100, height=30,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_ACCENT_DIM, hover_color=C_ACCENT_MUTED,
            text_color=C_ACCENT, command=self._new_data_file
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            editor_btns, text="REFRESH", width=90, height=30,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_BORDER,
            text_color=C_TEXT_DIM,
            command=self._refresh_data_files
        ).pack(side="left")

        # Right: log output
        log_panel = HUDFrame(page, glow_color=C_BORDER)
        log_panel.grid(row=1, column=2, sticky="nsew",
                       padx=(4, 10), pady=4)

        SectionLabel(log_panel, "Output Log", color=C_GREEN).pack(
            anchor="w", padx=8, pady=(8, 4))

        self.train_log = SelectableTextbox(
            log_panel, font=FONT_MONO,
            fg_color=C_PANEL, text_color=C_GREEN,
            border_width=0, corner_radius=2)
        self.train_log.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Load initial data file into editor
        self._editing_data_path = None
        if self.training_files:
            self._load_data_into_editor(self.training_files[0]["path"])

    def _forge_heading(self, parent, text: str):
        ctk.CTkLabel(
            parent, text=text, font=FONT_SECTION,
            text_color=C_TEXT_BRIGHT
        ).pack(anchor="w", padx=10, pady=(12, 2))
        ctk.CTkFrame(
            parent, height=1, fg_color=C_ACCENT_DIM, corner_radius=0
        ).pack(fill="x", padx=10, pady=(0, 6))

    def _forge_label(self, parent, text: str):
        ctk.CTkLabel(
            parent, text=text, font=FONT_TINY, text_color=C_TEXT_DIM
        ).pack(anchor="w", padx=10, pady=(2, 0))

    def _forge_entry(self, parent, default: str) -> ctk.CTkEntry:
        entry = themed_entry(parent)
        entry.insert(0, default)
        entry.pack(anchor="w", padx=10, pady=(0, 6))
        return entry

    # ================================================================
    # PAGE: CONFIG
    # ================================================================

    def _build_page_config(self):
        page = self._make_page("CONFIG")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(page, fg_color="transparent", height=48)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        SectionLabel(top, "Settings").pack(
            side="left", fill="x", expand=True)

        scroll = themed_scroll(page, fg_color=C_BG, corner_radius=0)
        scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        scroll.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            scroll,
            text="These settings control how the AI generates text.",
            font=FONT_SMALL, text_color=C_TEXT_DIM
        ).pack(anchor="w", padx=8, pady=(4, 8))

        self.config_entries: dict[str, ctk.CTkEntry] = {}

        for name, (lo, hi, step) in CONFIG_LIMITS.items():
            card = HUDFrame(scroll, glow_color=C_BORDER)
            card.pack(fill="x", padx=4, pady=3)

            inner = ctk.CTkFrame(card, fg_color="transparent")
            inner.pack(fill="x", padx=10, pady=6)
            inner.grid_columnconfigure(0, weight=1)

            display_name = CONFIG_DISPLAY_NAMES.get(
                name, name.replace("_", " ").upper())
            ctk.CTkLabel(
                inner, text=display_name, font=FONT_SECTION,
                text_color=C_TEXT_BRIGHT, anchor="w"
            ).grid(row=0, column=0, sticky="w")

            desc = CONFIG_DESCRIPTIONS.get(name, "")
            ctk.CTkLabel(
                inner, text=desc, font=FONT_TINY,
                text_color=C_TEXT_DIM, anchor="w", wraplength=500
            ).grid(row=1, column=0, sticky="w")

            ctk.CTkLabel(
                inner, text=f"Range: {lo} to {hi}", font=FONT_TINY,
                text_color=C_ACCENT_DIM, anchor="w"
            ).grid(row=2, column=0, sticky="w", pady=(2, 0))

            entry = themed_entry(inner, width=120, height=36)
            entry.grid(
                row=0, column=1, rowspan=3, padx=(8, 0), sticky="e")
            entry.bind(
                "<FocusOut>",
                lambda e, n=name: self._validate_config(n))
            entry.bind(
                "<Return>",
                lambda e, n=name: self._validate_config(n))
            self.config_entries[name] = entry

        # --- Display names section ---
        names_card = HUDFrame(scroll, glow_color=C_ACCENT_DIM)
        names_card.pack(fill="x", padx=4, pady=(10, 4))
        names_inner = ctk.CTkFrame(
            names_card, fg_color="transparent")
        names_inner.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(
            names_inner, text="DISPLAY NAMES",
            font=FONT_SECTION, text_color=C_TEXT_BRIGHT
        ).pack(anchor="w")
        ctk.CTkLabel(
            names_inner,
            text="Customize how you and the AI appear in chat.",
            font=FONT_TINY, text_color=C_TEXT_DIM, wraplength=500
        ).pack(anchor="w", pady=(2, 6))

        user_row = ctk.CTkFrame(
            names_inner, fg_color="transparent")
        user_row.pack(fill="x", pady=2)
        user_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            user_row, text="Your Name",
            font=FONT_SMALL, text_color=C_TEXT,
            width=140, anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=(0, 4))
        self._user_name_entry = themed_entry(
            user_row, width=200, height=30, font=FONT_SMALL)
        self._user_name_entry.grid(
            row=0, column=1, sticky="w", padx=(0, 4))
        self._user_name_entry.insert(0, self.user_name)

        ai_row = ctk.CTkFrame(
            names_inner, fg_color="transparent")
        ai_row.pack(fill="x", pady=2)
        ai_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            ai_row, text="AI Name",
            font=FONT_SMALL, text_color=C_TEXT,
            width=140, anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=(0, 4))
        self._ai_name_entry = themed_entry(
            ai_row, width=200, height=30, font=FONT_SMALL)
        self._ai_name_entry.grid(
            row=0, column=1, sticky="w", padx=(0, 4))
        self._ai_name_entry.insert(0, self.ai_name)

        name_btns = ctk.CTkFrame(
            names_inner, fg_color="transparent")
        name_btns.pack(fill="x", pady=(6, 0))
        ctk.CTkButton(
            name_btns, text="SAVE NAMES", width=120, height=30,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_ACCENT_DIM, hover_color=C_ACCENT_MUTED,
            text_color=C_ACCENT, command=self._save_display_names
        ).pack(side="left", padx=(0, 4))
        ctk.CTkButton(
            name_btns, text="RESET", width=80, height=30,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_BORDER,
            text_color=C_TEXT_DIM,
            command=self._reset_display_names
        ).pack(side="left")

        # Active profile display
        prof_card = HUDFrame(scroll, glow_color=C_PURPLE)
        prof_card.pack(fill="x", padx=4, pady=(10, 4))
        prof_inner = ctk.CTkFrame(prof_card, fg_color="transparent")
        prof_inner.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(
            prof_inner, text="ACTIVE PROFILE",
            font=FONT_SECTION, text_color=C_PURPLE
        ).pack(anchor="w")

        self.active_profile_label = ctk.CTkLabel(
            prof_inner, text="None selected", font=FONT_BODY,
            text_color=C_TEXT_DIM)
        self.active_profile_label.pack(anchor="w", pady=(2, 0))

        ctk.CTkLabel(
            prof_inner,
            text="Switch profiles from the CORE page dropdown. "
                 "Profiles automatically set generation parameters.",
            font=FONT_TINY, text_color=C_TEXT_DIM, wraplength=500
        ).pack(anchor="w", pady=(4, 0))

        # Brick info section
        brick_card = HUDFrame(scroll, glow_color=C_BORDER)
        brick_card.pack(fill="x", padx=4, pady=(10, 4))
        brick_inner = ctk.CTkFrame(brick_card, fg_color="transparent")
        brick_inner.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(
            brick_inner, text="BRICK MODULES",
            font=FONT_SECTION, text_color=C_TEXT_BRIGHT
        ).pack(anchor="w")
        ctk.CTkLabel(
            brick_inner,
            text="Bricks are plugin programs that connect to the "
                 "engine. They auto-start when the app launches.",
            font=FONT_TINY, text_color=C_TEXT_DIM, wraplength=500
        ).pack(anchor="w", pady=(2, 0))

        for brick in self.bricks_data:
            row = ctk.CTkFrame(brick_inner, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(
                row,
                text=f"{brick['name']} v{brick['version']}",
                font=FONT_SMALL, text_color=C_TEXT
            ).pack(side="left")
            if brick.get("description"):
                ctk.CTkLabel(
                    row,
                    text=f"  {brick['description'][:60]}",
                    font=FONT_TINY, text_color=C_TEXT_DIM
                ).pack(side="left", padx=(8, 0))

        # --- Paths section ---
        paths_card = HUDFrame(scroll, glow_color=C_BORDER)
        paths_card.pack(fill="x", padx=4, pady=(10, 4))
        paths_inner = ctk.CTkFrame(
            paths_card, fg_color="transparent")
        paths_inner.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(
            paths_inner, text="DIRECTORY PATHS",
            font=FONT_SECTION, text_color=C_TEXT_BRIGHT
        ).pack(anchor="w")
        ctk.CTkLabel(
            paths_inner,
            text="Set where the engine reads and writes files. "
                 "Changes take effect on next launch.",
            font=FONT_TINY, text_color=C_TEXT_DIM, wraplength=500
        ).pack(anchor="w", pady=(2, 6))

        self.path_entries: dict[str, ctk.CTkEntry] = {}

        for key, (display_name, default) in PATH_SETTINGS.items():
            path_row = ctk.CTkFrame(
                paths_inner, fg_color="transparent")
            path_row.pack(fill="x", pady=2)
            path_row.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(
                path_row, text=display_name,
                font=FONT_SMALL, text_color=C_TEXT,
                width=140, anchor="w"
            ).grid(row=0, column=0, sticky="w", padx=(0, 4))

            entry = themed_entry(
                path_row, width=300, height=30,
                font=FONT_SMALL)
            entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))
            entry.insert(0, str(default))
            self.path_entries[key] = entry

            ctk.CTkButton(
                path_row, text="...", width=36, height=30,
                font=FONT_SMALL, corner_radius=2,
                fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
                text_color=C_TEXT_DIM,
                command=lambda k=key: self._browse_path(k)
            ).grid(row=0, column=2)

        path_btns = ctk.CTkFrame(
            paths_inner, fg_color="transparent")
        path_btns.pack(fill="x", pady=(6, 0))

        ctk.CTkButton(
            path_btns, text="SAVE PATHS", width=120, height=30,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_ACCENT_DIM, hover_color=C_ACCENT_MUTED,
            text_color=C_ACCENT, command=self._save_paths
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            path_btns, text="RESET", width=80, height=30,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_BORDER,
            text_color=C_TEXT_DIM, command=self._reset_paths
        ).pack(side="left")

        # Load saved path overrides into entries
        self._load_path_settings()
