"""
Enigma Engine - GUI Config Page Builder
=========================================

Mixin providing the CONFIG page construction for EnigmaGUI.
Split from gui_pages.py to keep files under 800 lines.
"""
from __future__ import annotations

import logging

import customtkinter as ctk

from enigma_engine.gui.widgets import (
    C_ACCENT_DIM, C_BG, C_BORDER,
    C_SURFACE, C_TEXT, C_TEXT_BRIGHT, C_TEXT_DIM,
    FONT_SECTION, FONT_SMALL, FONT_TINY,
    HUDFrame, SectionLabel, SelectableLabel, Tooltip,
    themed_button, themed_dropdown, themed_entry, themed_scroll,
)
from enigma_engine.gui.scanners import (
    CONFIG_DESCRIPTIONS, CONFIG_DISPLAY_NAMES, CONFIG_LIMITS,
    DATA_DIR, PATH_SETTINGS,
)

logger = logging.getLogger(__name__)


class ConfigPageMixin:
    """Mixin providing the CONFIG page builder for EnigmaGUI.

    Expects the host class to have:
    - _make_page, user_name, ai_name, mods_data
    - config_entries, path_entries (set during construction)
    - _validate_config, _save_display_names, _reset_display_names
    - _save_paths, _reset_paths, _browse_path, _load_path_settings
    - _restart_gui, status_bar
    """

    # ================================================================
    # PAGE: CONFIG
    # ================================================================

    def _build_page_config(self):
        page = self._make_page("CONFIG")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        # Load gui_settings.json once — all widgets read from this dict
        _cached_settings: dict = {}
        try:
            import json as _json
            _settings_path = DATA_DIR / "gui_settings.json"
            if _settings_path.exists():
                _cached_settings = _json.loads(
                    _settings_path.read_text(encoding="utf-8"))
        except Exception:
            pass

        top = ctk.CTkFrame(page, fg_color="transparent", height=48)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        SectionLabel(top, "Settings").pack(
            side="left", fill="x", expand=True)

        scroll = themed_scroll(page, fg_color=C_BG, corner_radius=0)
        scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        scroll.grid_columnconfigure(0, weight=1)

        SelectableLabel(
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
            SelectableLabel(
                inner, text=display_name, font=FONT_SECTION,
                text_color=C_TEXT_BRIGHT, anchor="w"
            ).grid(row=0, column=0, sticky="w")

            desc = CONFIG_DESCRIPTIONS.get(name, "")
            ctk.CTkLabel(
                inner, text=desc, font=FONT_TINY,
                text_color=C_TEXT_DIM, anchor="w", wraplength=500
            ).grid(row=1, column=0, sticky="w")

            SelectableLabel(
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

        # --- Generation behavior section ---
        # Stage B-2c (Pass 156z9w): expose the engine-level
        # `inline_search_enabled` flag so users probing the
        # `<search>` syntax can suppress the post-gen scan + WARNING
        # without editing code.  Default True preserves Pass 156z9d
        # always-on observability.
        gen_card = HUDFrame(scroll, glow_color=C_BORDER)
        gen_card.pack(fill="x", padx=4, pady=(10, 4))
        gen_inner = ctk.CTkFrame(gen_card, fg_color="transparent")
        gen_inner.pack(fill="x", padx=10, pady=8)

        SelectableLabel(
            gen_inner, text="GENERATION BEHAVIOR",
            font=FONT_SECTION, text_color=C_TEXT_BRIGHT
        ).pack(anchor="w")
        ctk.CTkLabel(
            gen_inner,
            text="Engine-level toggles applied after each model load.",
            font=FONT_TINY, text_color=C_TEXT_DIM, wraplength=500
        ).pack(anchor="w", pady=(2, 6))

        _api_chat_enabled = bool(
            _cached_settings.get("use_api_chat", False))
        self._use_api_chat_var = ctk.BooleanVar(value=_api_chat_enabled)
        self._use_api_chat_cb = ctk.CTkCheckBox(
            gen_inner,
            text="Use API client for CORE chat send path",
            variable=self._use_api_chat_var,
            font=FONT_SMALL, text_color=C_TEXT,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2,
            command=self._toggle_use_api_chat)
        self._use_api_chat_cb.pack(anchor="w", pady=(4, 0))
        Tooltip(self._use_api_chat_cb,
                "When enabled, CORE chat requests route through\n"
                "EnigmaClient over HTTP instead of direct local\n"
                "engine.chat calls. If API request fails, GUI\n"
                "falls back to local engine when available.")

        api_url_row = ctk.CTkFrame(gen_inner, fg_color="transparent")
        api_url_row.pack(fill="x", pady=(6, 0))
        SelectableLabel(
            api_url_row,
            text="API URL:",
            font=FONT_TINY,
            text_color=C_TEXT_DIM,
        ).pack(side="left", padx=(0, 6))
        _api_url = str(
            _cached_settings.get("api_base_url", "http://127.0.0.1:8080")
            or "http://127.0.0.1:8080"
        )
        self._api_base_url_entry = themed_entry(
            api_url_row, width=280, height=30, font=FONT_SMALL)
        self._api_base_url_entry.pack(side="left", padx=(0, 6))
        self._api_base_url_entry.insert(0, _api_url)
        self._api_base_url_save_btn = themed_button(
            api_url_row,
            text="Save URL",
            command=self._save_api_base_url,
            width=90,
            style="secondary",
        )
        self._api_base_url_save_btn.pack(side="left")
        Tooltip(self._api_base_url_entry,
                "Base URL for API chat mode, e.g.\n"
                "http://127.0.0.1:8080")

        _inline_search = bool(
            _cached_settings.get("inline_search_enabled", True))
        self._inline_search_enabled_var = ctk.BooleanVar(
            value=_inline_search)
        self._inline_search_enabled_cb = ctk.CTkCheckBox(
            gen_inner,
            text="Record inline <search> emissions",
            variable=self._inline_search_enabled_var,
            font=FONT_SMALL, text_color=C_TEXT,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2,
            command=self._toggle_inline_search_enabled)
        self._inline_search_enabled_cb.pack(anchor="w", pady=(4, 0))
        Tooltip(self._inline_search_enabled_cb,
                "When enabled, the engine scans every reply for\n"
                "<search>...</search> tags, records the queries on\n"
                "engine.last_search_queries, and logs a WARNING for\n"
                "each emission. Disable to silence the scan when\n"
                "asking the AI about the <search> syntax itself.")

        # B-3a: opt-in splice / auto-stop.  Default OFF so existing
        # users see no change.  When ON, the native non-GGUF text
        # generation path appends ``</search>`` to ``stop_strings``
        # so the model halts cleanly at the closing tag instead of
        # rambling past it.  Sibling generation paths (stream /
        # vision / specul / medusa / lookahead / batch / GGUF) emit
        # a one-shot WARNING when the flag is ON and queries land in
        # ``last_search_queries`` — they don't yet honour the
        # auto-stop or splice (B-3b/c/d will add real splice).
        _inline_splice = bool(
            _cached_settings.get("inline_search_splice_enabled", False))
        self._inline_search_splice_enabled_var = ctk.BooleanVar(
            value=_inline_splice)
        self._inline_search_splice_enabled_cb = ctk.CTkCheckBox(
            gen_inner,
            text="Inline <search> auto-stop (B-3a, opt-in)",
            variable=self._inline_search_splice_enabled_var,
            font=FONT_SMALL, text_color=C_TEXT,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2,
            command=self._toggle_inline_search_splice_enabled)
        self._inline_search_splice_enabled_cb.pack(anchor="w", pady=(4, 0))
        Tooltip(self._inline_search_splice_enabled_cb,
                "When enabled, the native text generation path stops\n"
                "as soon as the model emits </search>, instead of\n"
                "rambling past it. Required for the upcoming RAG\n"
                "splice feature (B-3b/c). Streaming, vision, GGUF,\n"
                "and the speculative/medusa/lookahead paths log a\n"
                "WARNING instead — auto-stop applies to native text\n"
                "only in B-3a.")

        # N-14 (Pass 156z9ds): GUI surface for `rag_backend`
        # selection.  The factory `make_rag_index()` reads
        # CONFIG["rag_backend"] each call; this dropdown writes the
        # user choice to forge_config.json so the next RAG index
        # build picks it up.  Live indexes are not rebuilt — the
        # toggle takes effect when the user re-runs the
        # "Build RAG index" action (e.g. via the FORGE NOTES page).
        SelectableLabel(
            gen_inner, text="RAG BACKEND (retrieval index)",
            font=FONT_TINY, text_color=C_TEXT_BRIGHT
        ).pack(anchor="w", pady=(10, 0))
        ctk.CTkLabel(
            gen_inner,
            text=(
                "bm25 = keyword retrieval (default, zero extra "
                "deps).  dense = semantic retrieval via "
                "sentence-transformers + faiss-cpu (better recall "
                "on paraphrased queries; falls back to bm25 with "
                "a WARNING if the deps are missing)."
            ),
            font=FONT_TINY, text_color=C_TEXT_DIM, wraplength=500,
            justify="left"
        ).pack(anchor="w", pady=(2, 4))
        rag_row = ctk.CTkFrame(gen_inner, fg_color="transparent")
        rag_row.pack(fill="x", pady=(0, 4))
        from enigma_engine.config import CONFIG as _CONFIG
        _rag_backend_value = str(
            _CONFIG.get("rag_backend", "bm25") or "bm25")
        if _rag_backend_value not in ("bm25", "dense"):
            _rag_backend_value = "bm25"
        self._rag_backend_var = ctk.StringVar(value=_rag_backend_value)
        self._rag_backend_menu = themed_dropdown(
            rag_row, ["bm25", "dense"],
            variable=self._rag_backend_var,
            width=120, height=30,
            command=lambda _v: self._set_rag_backend())
        self._rag_backend_menu.pack(side="left", padx=(0, 6))
        Tooltip(self._rag_backend_menu,
                "Selecting a backend writes to forge_config.json.\n"
                "Takes effect on the NEXT RAG index build — live\n"
                "indexes already loaded keep their current backend\n"
                "until rebuilt.")

        # N-15b (Pass 156z9aa): GUI surface for `json_schema`
        # constrained decoding.  The library + API endpoint already
        # accept the field (Pass 156z3 + Pass 156z6); this is the
        # last surface that prevented users from reaching the
        # feature without writing Python.  Persist the RAW text so
        # the user's editing state (incl. invalid drafts) survives
        # restarts; parse on Apply, persist parsed dict to
        # self.json_schema, log loud on parse failure.
        SelectableLabel(
            gen_inner, text="JSON SCHEMA (constrained decoding)",
            font=FONT_TINY, text_color=C_TEXT_BRIGHT
        ).pack(anchor="w", pady=(10, 0))
        ctk.CTkLabel(
            gen_inner,
            text=(
                "When non-empty and valid, the next chat reply is "
                "constrained to JSON matching this schema.  Empty "
                "= unconstrained.  GGUF backends raise an error "
                "(constraint never reaches llama.cpp's sampler)."
            ),
            font=FONT_TINY, text_color=C_TEXT_DIM, wraplength=500,
            justify="left"
        ).pack(anchor="w", pady=(2, 4))
        self._json_schema_textbox = ctk.CTkTextbox(
            gen_inner,
            height=100,
            font=FONT_SMALL,
            fg_color=C_SURFACE,
            text_color=C_TEXT,
            border_color=C_ACCENT_DIM,
            border_width=1,
            corner_radius=2,
        )
        # Pre-fill from persisted raw text (saved on prior Apply).
        _persisted_schema_text = str(
            _cached_settings.get("json_schema_text", ""))
        if _persisted_schema_text:
            self._json_schema_textbox.insert("1.0", _persisted_schema_text)
        self._json_schema_textbox.pack(fill="x", pady=(0, 4))
        Tooltip(self._json_schema_textbox,
                "Paste a JSON Schema dict.  Apply parses it and\n"
                "stores the result on self.json_schema; the next\n"
                "chat send forwards it to engine.chat(json_schema=...).\n"
                "Empty + Apply clears the constraint.")
        btn_row = ctk.CTkFrame(gen_inner, fg_color="transparent")
        btn_row.pack(fill="x", pady=(0, 4))
        self._json_schema_apply_btn = themed_button(
            btn_row, text="Apply",
            command=self._apply_json_schema,
            width=80,
        )
        self._json_schema_apply_btn.pack(side="left", padx=(0, 6))
        self._json_schema_clear_btn = themed_button(
            btn_row, text="Clear",
            command=self._clear_json_schema,
            width=80,
            style="secondary",
        )
        self._json_schema_clear_btn.pack(side="left")

        # --- Display names section ---
        names_card = HUDFrame(scroll, glow_color=C_ACCENT_DIM)
        names_card.pack(fill="x", padx=4, pady=(10, 4))
        names_inner = ctk.CTkFrame(
            names_card, fg_color="transparent")
        names_inner.pack(fill="x", padx=10, pady=8)

        SelectableLabel(
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
        SelectableLabel(
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
        SelectableLabel(
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
        themed_button(
            name_btns, "SAVE NAMES", style="action",
            width=120, height=30,
            font=FONT_SMALL,
            command=self._save_display_names
        ).pack(side="left", padx=(0, 4))
        themed_button(
            name_btns, "RESET", style="secondary",
            width=80, height=30,
            font=FONT_SMALL,
            command=self._reset_display_names
        ).pack(side="left")

        # --- Theme section ---
        theme_card = HUDFrame(scroll, glow_color=C_ACCENT_DIM)
        theme_card.pack(fill="x", padx=4, pady=(10, 4))
        theme_inner = ctk.CTkFrame(
            theme_card, fg_color="transparent")
        theme_inner.pack(fill="x", padx=10, pady=8)

        SelectableLabel(
            theme_inner, text="THEME",
            font=FONT_SECTION, text_color=C_TEXT_BRIGHT
        ).pack(anchor="w")
        ctk.CTkLabel(
            theme_inner,
            text="Change the color theme.",
            font=FONT_TINY, text_color=C_TEXT_DIM, wraplength=500
        ).pack(anchor="w", pady=(2, 6))

        theme_row = ctk.CTkFrame(
            theme_inner, fg_color="transparent")
        theme_row.pack(fill="x", pady=2)
        theme_row.grid_columnconfigure(1, weight=1)

        SelectableLabel(
            theme_row, text="Color Theme",
            font=FONT_SMALL, text_color=C_TEXT,
            width=140, anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=(0, 4))

        from enigma_engine.gui.themes import (
            get_theme_names, load_active_theme)
        active_theme = load_active_theme()
        self.theme_var = ctk.StringVar(value=active_theme.name)
        self.theme_dd = themed_dropdown(
            theme_row, width=200,
            values=get_theme_names(),
            variable=self.theme_var,
            command=self._apply_theme)
        self.theme_dd.grid(
            row=0, column=1, sticky="w", padx=(0, 4))

        # --- Font size section ---
        font_card = HUDFrame(scroll, glow_color=C_ACCENT_DIM)
        font_card.pack(fill="x", padx=4, pady=(10, 4))
        font_inner = ctk.CTkFrame(
            font_card, fg_color="transparent")
        font_inner.pack(fill="x", padx=10, pady=8)

        SelectableLabel(
            font_inner, text="FONT SIZE",
            font=FONT_SECTION, text_color=C_TEXT_BRIGHT
        ).pack(anchor="w")
        ctk.CTkLabel(
            font_inner,
            text="Adjust font size across the entire GUI. "
                 "Takes effect on restart.",
            font=FONT_TINY, text_color=C_TEXT_DIM, wraplength=500
        ).pack(anchor="w", pady=(2, 6))

        font_row = ctk.CTkFrame(
            font_inner, fg_color="transparent")
        font_row.pack(fill="x", pady=2)
        font_row.grid_columnconfigure(1, weight=1)

        SelectableLabel(
            font_row, text="Size Offset",
            font=FONT_SMALL, text_color=C_TEXT,
            width=140, anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=(0, 4))

        from enigma_engine.gui.widgets import get_font_size_offset
        self._font_size_entry = themed_entry(
            font_row, width=80, height=30, font=FONT_SMALL)
        self._font_size_entry.grid(
            row=0, column=1, sticky="w", padx=(0, 4))
        self._font_size_entry.insert(
            0, str(get_font_size_offset()))

        SelectableLabel(
            font_row, text="Range: -4 to 8",
            font=FONT_TINY, text_color=C_ACCENT_DIM, anchor="w"
        ).grid(row=0, column=2, sticky="w", padx=(8, 0))

        themed_button(
            font_inner, "APPLY", style="action",
            width=90, height=30,
            font=FONT_SMALL,
            command=self._apply_font_size
        ).pack(anchor="w", pady=(6, 0))
        Tooltip(self._font_size_entry,
                "Positive values increase text size,\n"
                "negative values decrease it.\n"
                "0 = default size.")

        # --- Memory section ---
        mem_card = HUDFrame(scroll, glow_color=C_ACCENT_DIM)
        mem_card.pack(fill="x", padx=4, pady=(10, 4))
        mem_inner = ctk.CTkFrame(
            mem_card, fg_color="transparent")
        mem_inner.pack(fill="x", padx=10, pady=8)

        SelectableLabel(
            mem_inner, text="MEMORY",
            font=FONT_SECTION, text_color=C_TEXT_BRIGHT
        ).pack(anchor="w")
        ctk.CTkLabel(
            mem_inner,
            text="Controls how the AI remembers facts about you "
                 "across conversations.",
            font=FONT_TINY, text_color=C_TEXT_DIM, wraplength=500
        ).pack(anchor="w", pady=(2, 6))

        mem_mode_row = ctk.CTkFrame(
            mem_inner, fg_color="transparent")
        mem_mode_row.pack(fill="x", pady=2)
        mem_mode_row.grid_columnconfigure(1, weight=1)

        SelectableLabel(
            mem_mode_row, text="Memory Mode",
            font=FONT_SMALL, text_color=C_TEXT,
            width=140, anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=(0, 4))

        # Load current memory mode from cached settings
        _mem_mode = _cached_settings.get("memory_mode", "automatic")

        _mem_options = [
            "automatic - AI extracts facts from chat",
            "manual - only remembers when told",
            "disabled - no memory storage",
        ]
        _mem_display = next(
            (v for v in _mem_options if v.startswith(_mem_mode)),
            _mem_options[0])
        self._memory_mode_var = ctk.StringVar(value=_mem_display)
        self._memory_mode_dd = themed_dropdown(
            mem_mode_row, width=280,
            values=_mem_options,
            variable=self._memory_mode_var,
            command=self._change_memory_mode)
        self._memory_mode_dd.grid(
            row=0, column=1, sticky="w", padx=(0, 4))
        Tooltip(self._memory_mode_dd,
                "automatic — AI auto-extracts facts from chat + "
                "remembers via commands\n"
                "manual — AI only remembers when you say "
                "'remember that ...' or via commands\n"
                "disabled — no memory storage at all")

        # Chat history cap
        hist_row = ctk.CTkFrame(
            mem_inner, fg_color="transparent")
        hist_row.pack(fill="x", pady=2)
        hist_row.grid_columnconfigure(1, weight=1)

        SelectableLabel(
            hist_row, text="History cap",
            font=FONT_SMALL, text_color=C_TEXT,
            width=140, anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=(0, 4))

        # Load current history cap from cached settings
        try:
            _hist_cap = int(_cached_settings.get("history_cap", 500))
        except (ValueError, TypeError):
            _hist_cap = 500

        self._history_cap_var = ctk.StringVar(value=str(_hist_cap))
        hist_entry = themed_entry(
            hist_row, textvariable=self._history_cap_var, width=80)
        hist_entry.grid(row=0, column=1, sticky="w", padx=(0, 4))
        hist_entry.bind(
            "<FocusOut>",
            lambda e: self._change_history_cap())
        hist_entry.bind(
            "<Return>",
            lambda e: self._change_history_cap())
        Tooltip(hist_entry,
                "Max messages saved to disk per session.\n"
                "In-memory history is unlimited.\n"
                "Range: 10–10000. Default: 500")

        # Apply saved history cap on startup
        try:
            from enigma_engine.core.model_context import set_max_context_history
            set_max_context_history(_hist_cap)
        except Exception:
            pass


        # Show emotional state panel toggle
        _show_emo = bool(
            _cached_settings.get("show_emotional_state", True))
        self._show_emo_var = ctk.BooleanVar(value=_show_emo)
        self._show_emo_cb = ctk.CTkCheckBox(
            mem_inner,
            text="Show emotional state panel",
            variable=self._show_emo_var,
            font=FONT_SMALL, text_color=C_TEXT,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2,
            command=self._toggle_show_emotional_state)
        self._show_emo_cb.pack(anchor="w", pady=(6, 0))
        Tooltip(self._show_emo_cb,
                "Show or hide the emotional state bars on\n"
                "the CORE page sidebar. The AI still uses\n"
                "emotional awareness internally either way.")

        # --- Training section ---
        train_card = HUDFrame(scroll, glow_color=C_ACCENT_DIM)
        train_card.pack(fill="x", padx=4, pady=(10, 4))
        train_inner = ctk.CTkFrame(
            train_card, fg_color="transparent")
        train_inner.pack(fill="x", padx=10, pady=8)

        SelectableLabel(
            train_inner, text="TRAINING",
            font=FONT_SECTION, text_color=C_TEXT_BRIGHT
        ).pack(anchor="w")
        ctk.CTkLabel(
            train_inner,
            text="Background training options for chat sessions.",
            font=FONT_TINY, text_color=C_TEXT_DIM, wraplength=500
        ).pack(anchor="w", pady=(2, 6))

        lwc_row = ctk.CTkFrame(
            train_inner, fg_color="transparent")
        lwc_row.pack(fill="x", pady=2)
        lwc_row.grid_columnconfigure(1, weight=1)

        # Load current setting from cached settings
        _lwc_val = _cached_settings.get("learn_while_chatting", False)

        self._learn_while_chatting_var = ctk.BooleanVar(
            value=_lwc_val)
        self._learn_while_chatting_cb = ctk.CTkCheckBox(
            lwc_row,
            text="Learn while chatting",
            variable=self._learn_while_chatting_var,
            font=FONT_SMALL, text_color=C_TEXT,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2,
            command=self._toggle_learn_while_chatting)
        self._learn_while_chatting_cb.grid(
            row=0, column=0, sticky="w")
        Tooltip(self._learn_while_chatting_cb,
                "When enabled, chat exchanges are fed to the\n"
                "background trainer so the AI improves over time.\n"
                "Requires TRAINER route to be assigned.")

        # Continuous-3b (Pass 156o): anchor file widget — display
        # current path + row count, browse to a custom JSONL, reset
        # to the repo default. Anchors are rehearsed every replay
        # pass alongside the recent slice to mitigate forgetting of
        # skills absent from current chat.
        from enigma_engine.gui.scanners import _resolve_anchor_path
        _anchor_saved = _cached_settings.get("anchor_data_path", "")
        _anchor_resolved = _resolve_anchor_path(_anchor_saved)
        self._anchor_path_var = ctk.StringVar(value=_anchor_saved)

        anchor_label_row = ctk.CTkFrame(
            train_inner, fg_color="transparent")
        anchor_label_row.pack(fill="x", pady=(8, 2))
        SelectableLabel(
            anchor_label_row, text="Anchor rehearsal file",
            font=FONT_SMALL, text_color=C_TEXT, anchor="w"
        ).pack(side="left")

        self._anchor_status_label = ctk.CTkLabel(
            train_inner,
            text=self._format_anchor_status(_anchor_resolved),
            font=FONT_TINY, text_color=C_TEXT_DIM,
            wraplength=500, anchor="w", justify="left",
        )
        self._anchor_status_label.pack(anchor="w", pady=(0, 4))

        anchor_btn_row = ctk.CTkFrame(
            train_inner, fg_color="transparent")
        anchor_btn_row.pack(fill="x", pady=(0, 4))
        themed_button(
            anchor_btn_row, "BROWSE...", style="action",
            width=110, height=28, font=FONT_SMALL,
            command=self._browse_anchor_file
        ).pack(side="left", padx=(0, 6))
        themed_button(
            anchor_btn_row, "USE DEFAULT", style="action",
            width=130, height=28, font=FONT_SMALL,
            command=self._reset_anchor_file
        ).pack(side="left")
        Tooltip(self._anchor_status_label,
                "JSONL with rows like\n"
                '{"prompt": "...", "response": "..."}\n'
                "Rehearsed on every replay pass so general skills\n"
                "stay sharp even when recent chat is narrow.")

        # --- Performance section ---
        perf_card = HUDFrame(scroll, glow_color=C_ACCENT_DIM)
        perf_card.pack(fill="x", padx=4, pady=(10, 4))
        perf_inner = ctk.CTkFrame(
            perf_card, fg_color="transparent")
        perf_inner.pack(fill="x", padx=10, pady=8)

        SelectableLabel(
            perf_inner, text="PERFORMANCE",
            font=FONT_SECTION, text_color=C_TEXT_BRIGHT
        ).pack(anchor="w")
        ctk.CTkLabel(
            perf_inner,
            text=("Reduce background memory usage while keeping the app open. "
                  "Launch settings apply on next start."),
            font=FONT_TINY, text_color=C_TEXT_DIM, wraplength=500
        ).pack(anchor="w", pady=(2, 6))

        _auto_load_chat = bool(
            _cached_settings.get("auto_load_chat_model", True))
        _auto_start_mods = bool(
            _cached_settings.get("auto_start_mods", True))
        _auto_unload_on_minimize = bool(
            _cached_settings.get("auto_unload_on_minimize", False))

        self._auto_load_chat_model_var = ctk.BooleanVar(
            value=_auto_load_chat)
        self._auto_start_mods_var = ctk.BooleanVar(
            value=_auto_start_mods)
        self._auto_unload_on_minimize_var = ctk.BooleanVar(
            value=_auto_unload_on_minimize)

        self._auto_load_chat_model_cb = ctk.CTkCheckBox(
            perf_inner,
            text="Auto-load chat model on launch",
            variable=self._auto_load_chat_model_var,
            font=FONT_SMALL, text_color=C_TEXT,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2,
            command=self._toggle_auto_load_chat_model)
        self._auto_load_chat_model_cb.pack(anchor="w", pady=(0, 2))

        self._auto_start_mods_cb = ctk.CTkCheckBox(
            perf_inner,
            text="Auto-start mods on launch",
            variable=self._auto_start_mods_var,
            font=FONT_SMALL, text_color=C_TEXT,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2,
            command=self._toggle_auto_start_mods)
        self._auto_start_mods_cb.pack(anchor="w", pady=(0, 2))

        self._auto_unload_on_minimize_cb = ctk.CTkCheckBox(
            perf_inner,
            text="Unload chat model when minimized",
            variable=self._auto_unload_on_minimize_var,
            font=FONT_SMALL, text_color=C_TEXT,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2,
            command=self._toggle_auto_unload_on_minimize)
        self._auto_unload_on_minimize_cb.pack(anchor="w")
        Tooltip(self._auto_unload_on_minimize_cb,
                "When enabled, minimizing the window releases model memory\n"
                "and reloads the model when you restore the window.")

        _confirm_file_ops = bool(
            _cached_settings.get("confirm_file_operations", True))
        self._confirm_file_operations_var = ctk.BooleanVar(
            value=_confirm_file_ops)
        self._confirm_file_operations_cb = ctk.CTkCheckBox(
            perf_inner,
            text="Confirm AI file operations",
            variable=self._confirm_file_operations_var,
            font=FONT_SMALL, text_color=C_TEXT,
            fg_color=C_SURFACE, hover_color=C_ACCENT_DIM,
            border_color=C_ACCENT_DIM, corner_radius=2,
            command=self._toggle_confirm_file_operations)
        self._confirm_file_operations_cb.pack(anchor="w", pady=(4, 0))
        Tooltip(self._confirm_file_operations_cb,
                "When enabled, you must approve each file write/append\n"
                "the AI requests. When disabled, file operations are\n"
                "auto-approved.")

        themed_button(
            perf_inner, "APPLY GAMING MODE", style="action",
            width=180, height=30,
            font=FONT_SMALL,
            command=self._apply_gaming_mode_preset
        ).pack(anchor="w", pady=(8, 0))
        ctk.CTkLabel(
            perf_inner,
            text=("Sets launch to low-memory mode: no model autoload, "
                  "no mod autostart, unload model on minimize."),
            font=FONT_TINY, text_color=C_TEXT_DIM, wraplength=500
        ).pack(anchor="w", pady=(2, 0))

        # Mod info section
        mod_card = HUDFrame(scroll, glow_color=C_BORDER)
        mod_card.pack(fill="x", padx=4, pady=(10, 4))
        mod_inner = ctk.CTkFrame(mod_card, fg_color="transparent")
        mod_inner.pack(fill="x", padx=10, pady=8)

        SelectableLabel(
            mod_inner, text="MOD MODULES",
            font=FONT_SECTION, text_color=C_TEXT_BRIGHT
        ).pack(anchor="w")
        ctk.CTkLabel(
            mod_inner,
            text="Mods are plugin programs that connect to the "
                 "engine. Auto-start can be changed in PERFORMANCE.",
            font=FONT_TINY, text_color=C_TEXT_DIM, wraplength=500
        ).pack(anchor="w", pady=(2, 0))

        for mod in self.mods_data:
            row = ctk.CTkFrame(mod_inner, fg_color="transparent")
            row.pack(fill="x", pady=2)
            SelectableLabel(
                row,
                text=f"{mod['name']} v{mod['version']}",
                font=FONT_SMALL, text_color=C_TEXT
            ).pack(side="left")
            if mod.get("description"):
                SelectableLabel(
                    row,
                    text=f"  {mod['description'][:60]}",
                    font=FONT_TINY, text_color=C_TEXT_DIM
                ).pack(side="left", padx=(8, 0))

        # --- Paths section ---
        paths_card = HUDFrame(scroll, glow_color=C_BORDER)
        paths_card.pack(fill="x", padx=4, pady=(10, 4))
        paths_inner = ctk.CTkFrame(
            paths_card, fg_color="transparent")
        paths_inner.pack(fill="x", padx=10, pady=8)

        SelectableLabel(
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

            SelectableLabel(
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

            themed_button(
                path_row, "...", style="secondary",
                width=36, height=30,
                font=FONT_SMALL,
                command=lambda k=key: self._browse_path(k)
            ).grid(row=0, column=2)

        path_btns = ctk.CTkFrame(
            paths_inner, fg_color="transparent")
        path_btns.pack(fill="x", pady=(6, 0))

        themed_button(
            path_btns, "SAVE PATHS", style="action",
            width=120, height=30,
            font=FONT_SMALL,
            command=self._save_paths
        ).pack(side="left", padx=(0, 4))

        themed_button(
            path_btns, "RESET", style="secondary",
            width=80, height=30,
            font=FONT_SMALL,
            command=self._reset_paths
        ).pack(side="left")

        # --- Backup / Restore section ---
        backup_card = HUDFrame(scroll, glow_color=C_BORDER)
        backup_card.pack(fill="x", padx=4, pady=(10, 4))
        backup_inner = ctk.CTkFrame(
            backup_card, fg_color="transparent")
        backup_inner.pack(fill="x", padx=10, pady=8)

        SelectableLabel(
            backup_inner, text="BACKUP / RESTORE",
            font=FONT_SECTION, text_color=C_TEXT_BRIGHT
        ).pack(anchor="w")
        ctk.CTkLabel(
            backup_inner,
            text="Export settings, notes, and prompts to a ZIP file "
                 "or restore from a previous backup.",
            font=FONT_TINY, text_color=C_TEXT_DIM, wraplength=500
        ).pack(anchor="w", pady=(2, 6))

        backup_btns = ctk.CTkFrame(
            backup_inner, fg_color="transparent")
        backup_btns.pack(fill="x", pady=(2, 0))

        themed_button(
            backup_btns, "EXPORT BACKUP", style="action",
            width=140, height=30,
            font=FONT_SMALL,
            command=self._export_backup
        ).pack(side="left", padx=(0, 8))

        themed_button(
            backup_btns, "IMPORT BACKUP", style="secondary",
            width=140, height=30,
            font=FONT_SMALL,
            command=self._import_backup
        ).pack(side="left")

        # Inline import confirmation bar (hidden until needed)
        self._import_confirm_bar = ctk.CTkFrame(
            backup_inner, fg_color="transparent")
        SelectableLabel(
            self._import_confirm_bar,
            text="Overwrite settings with backup?",
            font=FONT_TINY, text_color="#f97316", anchor="w"
        ).pack(side="left", padx=(0, 6))
        themed_button(
            self._import_confirm_bar, "YES", style="warning",
            width=40, height=24,
            font=FONT_TINY,
            command=self._confirm_import_backup
        ).pack(side="left", padx=(0, 4))
        themed_button(
            self._import_confirm_bar, "NO", style="secondary",
            width=40, height=24,
            font=FONT_TINY,
            command=self._import_confirm_bar.pack_forget
        ).pack(side="left")

        # S643: Dynamic wraplength — labels adapt to window width
        _wrap_labels: list = []
        _wrap_resize_timer = [None]  # mutable container for debounce
        _wrap_last_w = [0]  # skip no-op reconfigures

        def _collect_wrap(widget):
            for child in widget.winfo_children():
                if isinstance(child, ctk.CTkLabel):
                    try:
                        wl = child.cget("wraplength")
                        if wl and int(wl) > 0:
                            _wrap_labels.append(child)
                    except Exception:
                        pass
                _collect_wrap(child)

        _collect_wrap(scroll)

        def _apply_config_wraplength(w):
            _wrap_resize_timer[0] = None
            _wrap_last_w[0] = w
            for lbl in _wrap_labels:
                lbl.configure(wraplength=w)

        def _on_config_resize(event):
            w = event.width - 60
            if w < 100 or w == _wrap_last_w[0]:
                return
            # Debounce: cancel pending, schedule after 100ms
            if _wrap_resize_timer[0] is not None:
                try:
                    self.after_cancel(_wrap_resize_timer[0])
                except (ValueError, Exception):
                    pass
            _wrap_resize_timer[0] = self.after(
                100, lambda: _apply_config_wraplength(w))

        scroll.bind("<Configure>", _on_config_resize)

        # Load saved path overrides into entries
        self._load_path_settings()

    # ------------------------------------------------------------------
    # Theme picker
    # ------------------------------------------------------------------

    def _apply_theme(self, name: str):
        """Apply theme live without restart."""
        self._apply_theme_live(name)

    # ------------------------------------------------------------------
    # Memory mode
    # ------------------------------------------------------------------

    def _change_memory_mode(self, mode: str):
        """Save memory_mode setting to gui_settings.json."""
        mode = mode.split(" - ", 1)[0]
        import json
        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["memory_mode"] = mode
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)
            # Apply live: set disabled flag on memory singleton
            try:
                from enigma_engine.core.memory import get_memory
                mem = get_memory()
                mem.disabled = (mode == "disabled")
            except Exception:
                pass
            self.status_bar.set_left(
                f"\u26a1 Memory mode set to: {mode}")
        except Exception as exc:
            logger.debug("Could not save memory_mode: %s", exc)

    # ------------------------------------------------------------------
    # Emotional state panel visibility
    # ------------------------------------------------------------------

    def _toggle_show_emotional_state(self):
        """Save show_emotional_state setting and update panel visibility."""
        import json
        visible = self._show_emo_var.get()
        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["show_emotional_state"] = visible
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)
        except Exception as exc:
            logger.debug("Could not save show_emotional_state: %s", exc)
        # Toggle panel on the CORE page sidebar
        if hasattr(self, "_set_emo_panel_visible"):
            self._set_emo_panel_visible(visible)
        state = "visible" if visible else "hidden"
        self.status_bar.set_left(
            f"\u26a1 Emotional state panel {state}")

    # ------------------------------------------------------------------
    # API chat mode controls (ARCH-1c bridge)
    # ------------------------------------------------------------------

    def _toggle_use_api_chat(self):
        """Persist ``use_api_chat`` and update live in-memory state."""
        import json

        enabled = bool(self._use_api_chat_var.get())
        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["use_api_chat"] = enabled
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)
        except Exception as exc:
            logger.debug("Could not save use_api_chat: %s", exc)

        self.use_api_chat = enabled
        if not enabled:
            self._api_chat_client = None

        state = "enabled" if enabled else "disabled"
        self.status_bar.set_left(f"[API] CORE chat routing {state}")

    def _save_api_base_url(self):
        """Persist API base URL and reset cached client instance."""
        import json

        raw = ""
        try:
            raw = self._api_base_url_entry.get().strip()
        except Exception:
            raw = ""
        base_url = raw or "http://127.0.0.1:8080"

        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["api_base_url"] = base_url
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)
        except Exception as exc:
            logger.debug("Could not save api_base_url: %s", exc)

        self.api_base_url = base_url
        self._api_chat_client = None
        self.status_bar.set_left(f"[API] URL saved: {base_url}")

    # ------------------------------------------------------------------
    # Inline <search> emissions toggle (Stage B-2c, Pass 156z9w)
    # ------------------------------------------------------------------

    def _toggle_inline_search_enabled(self):
        """Persist `inline_search_enabled` to gui_settings.json AND
        apply it to the live engine if one is loaded.

        Mirrors the engine attribute set in `_init_common`.  If the
        engine is loaded mid-session, the toggle takes effect on the
        NEXT generation call (the helper reads the attribute via
        `getattr(self, "inline_search_enabled", True)` on every call).
        """
        import json
        enabled = self._inline_search_enabled_var.get()
        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["inline_search_enabled"] = enabled
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)
        except Exception as exc:
            logger.debug(
                "Could not save inline_search_enabled: %s", exc)
        # Update in-memory mirror used by `_on_model_loaded` to
        # re-apply on subsequent loads.
        self.inline_search_enabled = enabled
        # Apply to live engine if loaded.
        eng = getattr(self, "engine", None)
        if eng is not None:
            eng.inline_search_enabled = enabled
        state = "enabled" if enabled else "disabled"
        self.status_bar.set_left(
            f"\u26a1 Inline <search> recording {state}")

    # ------------------------------------------------------------------
    # Inline <search> splice / auto-stop toggle (B-3a)
    # ------------------------------------------------------------------

    def _toggle_inline_search_splice_enabled(self):
        """Persist `inline_search_splice_enabled` to gui_settings.json
        AND apply it to the live engine if one is loaded.

        Mirrors the engine attribute set in ``_init_common``.  When
        True the native non-GGUF ``_generate_text`` path appends
        ``</search>`` to ``stop_strings``; sibling paths log a
        WARNING.  Default False — opt-in feature.
        """
        import json
        enabled = self._inline_search_splice_enabled_var.get()
        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["inline_search_splice_enabled"] = enabled
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)
        except Exception as exc:
            logger.debug(
                "Could not save inline_search_splice_enabled: %s",
                exc)
        self.inline_search_splice_enabled = enabled
        eng = getattr(self, "engine", None)
        if eng is not None:
            eng.inline_search_splice_enabled = enabled
        state = "enabled" if enabled else "disabled"
        self.status_bar.set_left(
            f"\u26a1 Inline <search> auto-stop {state}")

    # ------------------------------------------------------------------
    # RAG backend dropdown (N-14, Pass 156z9ds)
    # ------------------------------------------------------------------

    def _set_rag_backend(self):
        """Persist `rag_backend` to forge_config.json so the next
        `make_rag_index()` call honours the choice.

        Live RAG indexes are NOT rebuilt — the toggle only affects
        future builds (e.g. the next "Build RAG index" action from
        the FORGE NOTES page).  If the dropdown value is invalid the
        method falls back to ``"bm25"`` and rewrites the StringVar
        so the visible state matches what was actually persisted.
        """
        try:
            value = str(self._rag_backend_var.get() or "bm25")
        except Exception as exc:
            logger.warning("Could not read rag_backend var: %s", exc)
            return
        if value not in ("bm25", "dense"):
            logger.warning(
                "Unknown rag_backend %r, falling back to bm25", value)
            value = "bm25"
            try:
                self._rag_backend_var.set("bm25")
            except Exception:
                pass
        try:
            from enigma_engine.config import save_config, update_config
            update_config({"rag_backend": value})
            save_config()
        except Exception as exc:
            logger.warning("Could not persist rag_backend: %s", exc)
            self.status_bar.set_left(
                f"[!] RAG backend save failed: {exc}")
            return
        self.status_bar.set_left(
            f"\u26a1 RAG backend set to {value} "
            f"(applies on next index build)")

    # ------------------------------------------------------------------
    # JSON schema constrained decoding (N-15b, Pass 156z9aa)
    # ------------------------------------------------------------------

    def _apply_json_schema(self):
        """Parse the textbox content as JSON and stage it for the
        next chat call.  Behaviour matrix:
        - Empty/whitespace text → clear ``self.json_schema = None``,
          persist empty string.  Status: "JSON schema cleared".
        - Valid JSON dict → set ``self.json_schema`` to the parsed
          dict, persist raw text.  Status: "JSON schema applied
          (N keys)".
        - Valid JSON but not a dict (list, str, number, bool, null)
          → leave ``self.json_schema`` unchanged, persist nothing,
          status reports the type mismatch.  json_schema MUST be a
          dict per the engine contract.
        - Invalid JSON → leave ``self.json_schema`` unchanged,
          persist nothing, status reports the parse error.

        Persisting only on success means a user typing an invalid
        draft, restarting, and re-typing won't be silently corrupted
        by their last bad attempt — the textbox shows the LAST
        SUCCESSFULLY APPLIED text on next boot.
        """
        import json
        try:
            raw = self._json_schema_textbox.get("1.0", "end").strip()
        except Exception as exc:
            logger.warning(
                "Could not read json_schema textbox: %s", exc)
            self.status_bar.set_left(
                "[!] JSON schema: could not read textbox")
            return

        if not raw:
            self.json_schema = None
            ok = self._persist_json_schema_text("")
            if ok:
                self.status_bar.set_left("\u26a1 JSON schema cleared")
            else:
                self.status_bar.set_left(
                    "[!] JSON schema cleared in memory but disk "
                    "save failed (see log)")
            return

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.status_bar.set_left(
                f"[!] JSON schema parse error: {exc.msg} (line "
                f"{exc.lineno}, col {exc.colno})")
            return

        if not isinstance(parsed, dict):
            self.status_bar.set_left(
                f"[!] JSON schema must be a dict, got "
                f"{type(parsed).__name__}")
            return

        # Pass 156z9ad: structural shape validation at the boundary
        # closest to the user.  Without this, a schema like
        # {"type": "array"} or a malformed properties block passes
        # Apply with "schema applied (N keys)" and only fails at
        # send time deep inside JsonSchemaConstraint.__init__ —
        # surfacing as a chat-system ValueError caught by Pass
        # 156z9ab Finding 5.  Same don't-clobber semantics as the
        # parse-error and non-dict branches above: attr and disk
        # both stay at their last successfully-applied state.
        from enigma_engine.core.json_schema_mask import (
            validate_json_schema_shape,
        )
        try:
            validate_json_schema_shape(parsed)
        except ValueError as exc:
            self.status_bar.set_left(
                f"[!] JSON schema invalid: {exc}")
            return

        self.json_schema = parsed
        ok = self._persist_json_schema_text(raw)
        if ok:
            self.status_bar.set_left(
                f"\u26a1 JSON schema applied ({len(parsed)} keys)")
        else:
            self.status_bar.set_left(
                f"[!] JSON schema applied in memory ({len(parsed)} "
                "keys) but disk save failed (see log) — will not "
                "survive restart")

    def _clear_json_schema(self):
        """Clear the textbox + persisted state + live attribute.
        Equivalent to selecting all text, deleting, and clicking
        Apply — but explicit so the user sees a dedicated control.
        """
        try:
            self._json_schema_textbox.delete("1.0", "end")
        except Exception as exc:
            logger.debug(
                "Could not clear json_schema textbox: %s", exc)
        self.json_schema = None
        ok = self._persist_json_schema_text("")
        if ok:
            self.status_bar.set_left("\u26a1 JSON schema cleared")
        else:
            self.status_bar.set_left(
                "[!] JSON schema cleared in memory but disk save "
                "failed (see log)")

    def _persist_json_schema_text(self, text: str) -> bool:
        """Atomic-write the raw textbox content to gui_settings.json
        under key ``json_schema_text``.  Stored as raw text (NOT
        the parsed dict) so the user's exact formatting / comments-
        inside-strings / whitespace round-trip on restart.  Boot-
        load re-parses on next start.

        Returns True on success, False on disk failure (full disk,
        permission denied, JSON corruption on prior settings file).
        Caller is responsible for surfacing the failure — silently
        swallowing the IOError would let ``_apply_json_schema``
        post a misleading "schema applied" status while the schema
        is gone on next boot.  Pass 156z9ab fix.
        """
        import json
        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["json_schema_text"] = text
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)
            return True
        except Exception as exc:
            logger.warning(
                "Could not save json_schema_text to %s: %s",
                settings_path, exc)
            return False

    # ------------------------------------------------------------------
    # History cap
    # ------------------------------------------------------------------

    def _change_history_cap(self):
        """Save and apply history_cap setting."""
        import json
        try:
            val = int(self._history_cap_var.get())
        except (ValueError, TypeError):
            self.status_bar.set_left(
                "[!] History cap must be a number (10-10000)")
            return
        val = max(10, min(val, 10000))
        self._history_cap_var.set(str(val))
        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["history_cap"] = val
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)
            from enigma_engine.core.model_context import set_max_context_history
            set_max_context_history(val)
            self.status_bar.set_left(
                f"\u26a1 History cap set to: {val}")
        except Exception as exc:
            logger.debug("Could not save history_cap: %s", exc)

    # ------------------------------------------------------------------
    # Learn while chatting toggle
    # ------------------------------------------------------------------

    def _toggle_learn_while_chatting(self):
        """Save learn_while_chatting setting to gui_settings.json."""
        import json
        enabled = self._learn_while_chatting_var.get()
        # Warn if enabling but no router/trainer is available
        if enabled:
            router = getattr(self, "_router", None)
            has_trainer = router is not None and getattr(router, "trainer", None) is not None
            if not has_trainer:
                self.status_bar.set_left(
                    "\u26a0 No TRAINER route assigned — "
                    "chat exchanges will not be used for learning")
        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["learn_while_chatting"] = enabled
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)
            self._chat_learning_enabled = enabled
            if hasattr(self, "_refresh_performance_mode"):
                self._refresh_performance_mode()
            if hasattr(self, "_sync_router_training_state"):
                self._sync_router_training_state()
            state = "enabled" if enabled else "disabled"
            self.status_bar.set_left(
                f"\u26a1 Learn while chatting {state}")
        except Exception as exc:
            logger.debug("Could not save learn_while_chatting: %s", exc)

    # ------------------------------------------------------------------
    # Anchor file (Continuous-3b, Pass 156o)
    # ------------------------------------------------------------------

    def _format_anchor_status(self, resolved_path) -> str:
        """Human-readable line for the anchor status label.

        Three branches:
        - resolved=None             -> "no anchor file (recent-only)"
        - resolved exists           -> "<path>  (N rows)"
        - resolved set but missing  -> "<path>  (file missing)"
        """
        from pathlib import Path as _Path
        if resolved_path is None:
            return "(none — replay rehearses recent chat only)"
        p = _Path(resolved_path)
        if not p.exists():
            return f"{p}  (file missing)"
        try:
            with p.open("r", encoding="utf-8") as fh:
                count = sum(1 for line in fh if line.strip())
        except Exception:
            count = 0
        return f"{p}  ({count} row{'s' if count != 1 else ''})"

    def _save_anchor_data_path(self, value: str) -> None:
        """Persist `anchor_data_path` to gui_settings.json."""
        import json
        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["anchor_data_path"] = value
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)
        except Exception as exc:
            logger.debug("Could not save anchor_data_path: %s", exc)

    def _refresh_anchor_status(self) -> None:
        """Re-render the anchor status label from current saved path."""
        from enigma_engine.gui.scanners import _resolve_anchor_path
        saved = self._anchor_path_var.get()
        resolved = _resolve_anchor_path(saved)
        if hasattr(self, "_anchor_status_label"):
            self._anchor_status_label.configure(
                text=self._format_anchor_status(resolved))

    def _browse_anchor_file(self) -> None:
        """Pick a custom JSONL anchor file."""
        from tkinter import filedialog
        chosen = filedialog.askopenfilename(
            title="Select anchor rehearsal file",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
        )
        if not chosen:
            return
        self._anchor_path_var.set(chosen)
        self._save_anchor_data_path(chosen)
        self._refresh_anchor_status()
        self.status_bar.set_left(
            "\u26a1 Anchor file updated — restart to apply")

    def _reset_anchor_file(self) -> None:
        """Clear the saved override; fall back to repo default."""
        self._anchor_path_var.set("")
        self._save_anchor_data_path("")
        self._refresh_anchor_status()
        self.status_bar.set_left(
            "\u26a1 Anchor file reset to default — restart to apply")

    def _toggle_auto_load_chat_model(self):
        """Save auto_load_chat_model setting to gui_settings.json."""
        import json

        enabled = self._auto_load_chat_model_var.get()
        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["auto_load_chat_model"] = enabled
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)
            self._auto_load_chat_model = enabled
            if hasattr(self, "_refresh_performance_mode"):
                self._refresh_performance_mode()
            self.status_bar.set_left(
                "\u26a1 Auto-load chat model "
                f"{'enabled' if enabled else 'disabled'} (next launch)")
        except Exception as exc:
            logger.debug("Could not save auto_load_chat_model: %s", exc)

    def _toggle_auto_start_mods(self):
        """Save auto_start_mods setting to gui_settings.json."""
        import json

        enabled = self._auto_start_mods_var.get()
        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["auto_start_mods"] = enabled
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)
            self._auto_start_mods = enabled
            if hasattr(self, "_refresh_performance_mode"):
                self._refresh_performance_mode()
            self.status_bar.set_left(
                "\u26a1 Auto-start mods "
                f"{'enabled' if enabled else 'disabled'} (next launch)")
        except Exception as exc:
            logger.debug("Could not save auto_start_mods: %s", exc)

    def _toggle_auto_unload_on_minimize(self):
        """Save auto_unload_on_minimize and apply immediately."""
        import json

        enabled = self._auto_unload_on_minimize_var.get()
        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["auto_unload_on_minimize"] = enabled
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)
            self._auto_unload_on_minimize = enabled
            if hasattr(self, "_refresh_performance_mode"):
                self._refresh_performance_mode()
            self.status_bar.set_left(
                "\u26a1 Minimize unload "
                f"{'enabled' if enabled else 'disabled'}")
        except Exception as exc:
            logger.debug("Could not save auto_unload_on_minimize: %s", exc)

    def _toggle_confirm_file_operations(self):
        """Save confirm_file_operations setting to gui_settings.json."""
        import json

        enabled = self._confirm_file_operations_var.get()
        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["confirm_file_operations"] = enabled
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)
            self._confirm_file_operations = enabled
            state = "enabled" if enabled else "disabled (auto-approve)"
            self.status_bar.set_left(
                f"\u26a1 File operation confirmation {state}")
        except Exception as exc:
            logger.debug(
                "Could not save confirm_file_operations: %s", exc)

    def _apply_gaming_mode_preset(self):
        """Apply practical low-memory defaults for gaming sessions."""
        import json

        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))

            settings["auto_load_chat_model"] = False
            settings["auto_start_mods"] = False
            settings["auto_unload_on_minimize"] = True
            settings["learn_while_chatting"] = False
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)

            self._auto_load_chat_model = False
            self._auto_start_mods = False
            self._auto_unload_on_minimize = True
            self._chat_learning_enabled = False

            if hasattr(self, "_auto_load_chat_model_var"):
                self._auto_load_chat_model_var.set(False)
            if hasattr(self, "_auto_start_mods_var"):
                self._auto_start_mods_var.set(False)
            if hasattr(self, "_auto_unload_on_minimize_var"):
                self._auto_unload_on_minimize_var.set(True)
            if hasattr(self, "_learn_while_chatting_var"):
                self._learn_while_chatting_var.set(False)
            if hasattr(self, "_refresh_performance_mode"):
                self._refresh_performance_mode()
            if hasattr(self, "_sync_router_training_state"):
                self._sync_router_training_state()

            self.status_bar.set_left(
                "\u26a1 Gaming mode applied (low-overhead mode active)")
        except Exception as exc:
            logger.debug("Could not apply gaming mode preset: %s", exc)

    # ------------------------------------------------------------------
    # Font size adjustment
    # ------------------------------------------------------------------

    def _apply_font_size(self):
        """Save font_size_offset to gui_settings.json and restart."""
        import json
        raw = self._font_size_entry.get().strip()
        try:
            offset = int(raw)
        except ValueError:
            self.status_bar.set_left(
                "Invalid font size offset — must be an integer")
            return
        # Clamp to valid range
        offset = max(-4, min(8, offset))

        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["font_size_offset"] = offset
            from enigma_engine.core.safe_save import atomic_write_json
            atomic_write_json(settings_path, settings)
            self.status_bar.set_left(
                f"\u2728 Font size offset {offset} saved — restarting...")
            self.after(800, self._restart_gui)
        except Exception as exc:
            logger.debug("Could not save font_size_offset: %s", exc)

    # ------------------------------------------------------------------
    # Backup / Restore
    # ------------------------------------------------------------------

    def _export_backup(self):
        """Export settings, notes, and prompts to a ZIP file."""
        import zipfile
        from pathlib import Path
        from tkinter import filedialog

        dest = filedialog.asksaveasfilename(
            title="Export Backup",
            defaultextension=".zip",
            filetypes=[("ZIP archive", "*.zip")],
            initialfile="enigma_backup.zip")
        if not dest:
            return

        try:
            data_dir = DATA_DIR
            with zipfile.ZipFile(dest, "w",
                                 zipfile.ZIP_DEFLATED) as zf:
                # gui_settings.json
                settings_path = data_dir / "gui_settings.json"
                if settings_path.exists():
                    zf.write(str(settings_path),
                             "gui_settings.json")

                # route_assignments.json
                routes_path = data_dir / "route_assignments.json"
                if routes_path.exists():
                    zf.write(str(routes_path),
                             "route_assignments.json")

                # prompts.json
                prompts_path = data_dir / "prompts.json"
                if prompts_path.exists():
                    zf.write(str(prompts_path), "prompts.json")

                # notes/ directory
                notes_dir = data_dir / "notes"
                if notes_dir.is_dir():
                    for f in notes_dir.rglob("*"):
                        if f.is_file():
                            arc = Path("notes") / f.relative_to(
                                notes_dir)
                            zf.write(str(f), str(arc))

                # prompts/ directory
                prompts_dir = data_dir / "prompts"
                if prompts_dir.is_dir():
                    for f in prompts_dir.rglob("*"):
                        if f.is_file():
                            arc = Path("prompts") / f.relative_to(
                                prompts_dir)
                            zf.write(str(f), str(arc))

                # profiles/ directory (at project root)
                profiles_dir = Path(
                    data_dir).parent / "profiles"
                if profiles_dir.is_dir():
                    for f in profiles_dir.rglob("*"):
                        if f.is_file():
                            arc = Path("profiles") / f.relative_to(
                                profiles_dir)
                            zf.write(str(f), str(arc))

            self.status_bar.set_left(
                f"\u2705 Backup exported to {Path(dest).name}")
        except Exception as exc:
            logger.error("Backup export failed: %s", exc)
            self.status_bar.set_left(
                f"Backup failed: {exc}")

    def _import_backup(self):
        """Open file picker and show inline confirmation bar."""
        import zipfile
        from tkinter import filedialog

        src = filedialog.askopenfilename(
            title="Import Backup",
            filetypes=[("ZIP archive", "*.zip")])
        if not src:
            return

        # Verify it looks like a valid backup
        try:
            with zipfile.ZipFile(src, "r") as zf:
                names = zf.namelist()
        except Exception as exc:
            self.status_bar.set_left(f"Invalid ZIP: {exc}")
            return

        if not any(n == "gui_settings.json" for n in names):
            self.status_bar.set_left(
                "Not a valid backup — gui_settings.json missing.")
            return

        # Store path and show inline confirmation bar
        self._pending_import_path = src
        self._import_confirm_bar.pack(fill="x", pady=(4, 0))

    def _confirm_import_backup(self):
        """Actually import the backup after inline confirmation."""
        self._import_confirm_bar.pack_forget()
        src = getattr(self, "_pending_import_path", None)
        if not src:
            return
        self._pending_import_path = None

        import zipfile
        from pathlib import Path

        try:
            data_dir = DATA_DIR
            project_root = Path(data_dir).parent.resolve()
            with zipfile.ZipFile(src, "r") as zf:
                for entry in zf.namelist():
                    # Security: reject absolute or traversal paths
                    if entry.startswith("/") or ".." in entry:
                        continue
                    target = data_dir / entry
                    # Profiles go to project-level profiles/
                    if entry.startswith("profiles/"):
                        target = (
                            Path(data_dir).parent / entry)
                    # Final resolve check: must stay under project root
                    if not target.resolve().is_relative_to(project_root):
                        continue
                    # Directories for notes/ and prompts/
                    if entry.endswith("/"):
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(entry))

            self.status_bar.set_left(
                "\u2705 Backup imported — restart to apply.")
        except Exception as exc:
            logger.error("Backup import failed: %s", exc)
            self.status_bar.set_left(f"Import failed: {exc}")
