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
    C_ACCENT, C_ACCENT_DIM, C_ACCENT_MUTED, C_BG, C_BORDER,
    C_SURFACE, C_TEXT, C_TEXT_BRIGHT, C_TEXT_DIM,
    FONT_SECTION, FONT_SMALL, FONT_TINY,
    HUDFrame, SectionLabel, SelectableLabel, Tooltip,
    themed_dropdown, themed_entry, themed_scroll,
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

        ctk.CTkButton(
            font_inner, text="APPLY", width=90, height=30,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_ACCENT_DIM, hover_color=C_ACCENT_MUTED,
            text_color=C_ACCENT,
            command=self._apply_font_size
        ).pack(anchor="w", pady=(6, 0))
        Tooltip(self._font_size_entry,
                "Positive values increase text size,\n"
                "negative values decrease it.\n"
                "0 = default size.")

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

        # Load current setting
        _lwc_val = False
        try:
            _lwc_path = DATA_DIR / "gui_settings.json"
            if _lwc_path.exists():
                import json as _json
                _lwc_settings = _json.loads(
                    _lwc_path.read_text(encoding="utf-8"))
                _lwc_val = _lwc_settings.get(
                    "learn_while_chatting", False)
        except Exception:
            pass

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

        _auto_load_chat = True
        _auto_start_mods = True
        _auto_unload_on_minimize = False
        try:
            _perf_path = DATA_DIR / "gui_settings.json"
            if _perf_path.exists():
                import json as _json
                _perf_settings = _json.loads(
                    _perf_path.read_text(encoding="utf-8"))
                _auto_load_chat = bool(
                    _perf_settings.get("auto_load_chat_model", True))
                _auto_start_mods = bool(
                    _perf_settings.get("auto_start_mods", True))
                _auto_unload_on_minimize = bool(
                    _perf_settings.get("auto_unload_on_minimize", False))
        except Exception:
            pass

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

        ctk.CTkButton(
            perf_inner, text="APPLY GAMING MODE", width=180, height=30,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_ACCENT_DIM, hover_color=C_ACCENT_MUTED,
            text_color=C_ACCENT,
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

        ctk.CTkButton(
            backup_btns, text="EXPORT BACKUP", width=140, height=30,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_ACCENT_DIM, hover_color=C_ACCENT_MUTED,
            text_color=C_ACCENT, command=self._export_backup
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            backup_btns, text="IMPORT BACKUP", width=140, height=30,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_BORDER,
            text_color=C_TEXT_DIM, command=self._import_backup
        ).pack(side="left")

        # Inline import confirmation bar (hidden until needed)
        self._import_confirm_bar = ctk.CTkFrame(
            backup_inner, fg_color="transparent")
        SelectableLabel(
            self._import_confirm_bar,
            text="Overwrite settings with backup?",
            font=FONT_TINY, text_color="#f97316", anchor="w"
        ).pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            self._import_confirm_bar, text="YES", width=40, height=24,
            font=FONT_TINY, corner_radius=2,
            fg_color="#3a2a11", hover_color="#5a3a1a",
            text_color="#f97316",
            command=self._confirm_import_backup
        ).pack(side="left", padx=(0, 4))
        ctk.CTkButton(
            self._import_confirm_bar, text="NO", width=40, height=24,
            font=FONT_TINY, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_BORDER,
            text_color=C_TEXT_DIM,
            command=self._import_confirm_bar.pack_forget
        ).pack(side="left")

        # Load saved path overrides into entries
        self._load_path_settings()

    # ------------------------------------------------------------------
    # Theme picker
    # ------------------------------------------------------------------

    def _apply_theme(self, name: str):
        """Apply theme live without restart."""
        self._apply_theme_live(name)

    # ------------------------------------------------------------------
    # Learn while chatting toggle
    # ------------------------------------------------------------------

    def _toggle_learn_while_chatting(self):
        """Save learn_while_chatting setting to gui_settings.json."""
        import json
        enabled = self._learn_while_chatting_var.get()
        settings_path = DATA_DIR / "gui_settings.json"
        try:
            settings: dict = {}
            if settings_path.exists():
                settings = json.loads(
                    settings_path.read_text(encoding="utf-8"))
            settings["learn_while_chatting"] = enabled
            settings_path.write_text(
                json.dumps(settings, indent=2),
                encoding="utf-8")
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
            settings_path.write_text(
                json.dumps(settings, indent=2),
                encoding="utf-8")
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
            settings_path.write_text(
                json.dumps(settings, indent=2),
                encoding="utf-8")
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
            settings_path.write_text(
                json.dumps(settings, indent=2),
                encoding="utf-8")
            self._auto_unload_on_minimize = enabled
            if hasattr(self, "_refresh_performance_mode"):
                self._refresh_performance_mode()
            self.status_bar.set_left(
                "\u26a1 Minimize unload "
                f"{'enabled' if enabled else 'disabled'}")
        except Exception as exc:
            logger.debug("Could not save auto_unload_on_minimize: %s", exc)

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
            settings_path.write_text(
                json.dumps(settings, indent=2),
                encoding="utf-8")

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
            settings_path.write_text(
                json.dumps(settings, indent=2),
                encoding="utf-8")
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
            with zipfile.ZipFile(src, "r") as zf:
                for entry in zf.namelist():
                    # Security: reject absolute or traversal paths
                    if entry.startswith("/") or ".." in entry:
                        continue
                    target = data_dir / entry
                    # Directories for notes/ and prompts/
                    if entry.endswith("/"):
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    # Profiles go to project-level profiles/
                    if entry.startswith("profiles/"):
                        target = (
                            Path(data_dir).parent / entry)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(entry))

            self.status_bar.set_left(
                "\u2705 Backup imported — restart to apply.")
        except Exception as exc:
            logger.error("Backup import failed: %s", exc)
            self.status_bar.set_left(f"Import failed: {exc}")
