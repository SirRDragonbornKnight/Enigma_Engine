"""
Enigma Engine - Mod Page Builder
=====================================

Mixin providing per-mod page construction for EnigmaGUI.
Extracted from gui_pages.py to keep files under 800 lines.
"""
from __future__ import annotations

import customtkinter as ctk

from enigma_engine.gui.widgets import (
    C_ACCENT, C_ACCENT_DIM, C_ACCENT_MUTED, C_BORDER,
    C_GREEN, C_INPUT, C_PANEL,
    C_TEXT, C_TEXT_BRIGHT, C_TEXT_DIM,
    FONT_MONO, FONT_SECTION, FONT_SMALL, FONT_TINY,
    HUDFrame, SectionLabel, SelectableLabel, SelectableTextbox,
    StatusDot, Tooltip,
    themed_button, themed_entry, themed_dropdown,
)


class ModPageMixin:
    """Mixin providing mod page builder for EnigmaGUI.

    Expects the host class to have:
    - _make_page, _start_mod, _stop_mod, _send_mod_command
    """

    # ================================================================
    # PAGE: MOD - Per-mod pages built from mod.json
    # ================================================================

    def _build_page_mod(self, mod: dict):
        """Build a dedicated page for a mod module.

        Layout:
          Top bar: mod name + START / STOP buttons
          Left:    info card, commands list, UI widgets
          Right:   output log
        """
        page_name = f"MOD_{mod['id']}"
        page = self._make_page(page_name)
        page.grid_columnconfigure(0, weight=1)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(1, weight=1)

        # --- Top bar ---
        top = ctk.CTkFrame(page, fg_color="transparent", height=48)
        top.grid(row=0, column=0, columnspan=2, sticky="ew",
                 padx=10, pady=(8, 2))

        SectionLabel(top, mod["name"]).pack(
            side="left", fill="x", expand=True)

        # Status dot (shows green when mod is running)
        mod["_page_dot"] = StatusDot(top, color=C_TEXT_DIM)
        mod["_page_dot"].pack(side="right", padx=(4, 0))

        mod["_page_status"] = SelectableLabel(
            top, text="READY", font=FONT_TINY,
            text_color=C_TEXT_DIM)
        mod["_page_status"].pack(side="right", padx=(0, 4))

        # No START/STOP buttons — mods auto-launch when commands sent

        # --- Left column: info + commands + widgets ---
        left_col = ctk.CTkFrame(page, fg_color="transparent")
        left_col.grid(row=1, column=0, sticky="nsew",
                      padx=(10, 4), pady=4)
        left_col.grid_columnconfigure(0, weight=1)
        left_col.grid_rowconfigure(2, weight=1)

        # Info card
        info_card = HUDFrame(left_col, glow_color=C_BORDER)
        info_card.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        info_inner = ctk.CTkFrame(info_card, fg_color="transparent")
        info_inner.pack(fill="x", padx=10, pady=8)

        SelectableLabel(
            info_inner, text=mod["name"], font=FONT_SECTION,
            text_color=C_TEXT_BRIGHT
        ).pack(anchor="w")

        desc = mod.get("description", "")
        if desc:
            ctk.CTkLabel(
                info_inner, text=desc, font=FONT_SMALL,
                text_color=C_TEXT, wraplength=500, anchor="w",
                justify="left"
            ).pack(anchor="w", pady=(2, 0))

        detail_line = (
            f"v{mod.get('version', '?')}  //  "
            f"Port {mod.get('port', '?')}")
        SelectableLabel(
            info_inner, text=detail_line, font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(anchor="w", pady=(4, 0))

        # Dependencies
        deps = mod.get("dependencies", [])
        if deps:
            SelectableLabel(
                info_inner,
                text=f"Requires: {', '.join(deps)}",
                font=FONT_TINY, text_color=C_TEXT_DIM
            ).pack(anchor="w", pady=(2, 0))

        # Settings summary
        settings = mod.get("settings", {})
        if settings:
            settings_keys = list(settings.keys())
            if len(settings_keys) > 4:
                summary = ", ".join(settings_keys[:4]) + "..."
            else:
                summary = ", ".join(settings_keys)
            SelectableLabel(
                info_inner,
                text=f"Settings: {summary}",
                font=FONT_TINY, text_color=C_TEXT_DIM
            ).pack(anchor="w", pady=(2, 0))

        # AI prompt hint (tells the AI how to use this mod)
        prompt = mod.get("prompt", "")
        if prompt:
            ctk.CTkLabel(
                info_inner,
                text=f"AI usage: {prompt}",
                font=FONT_TINY, text_color=C_ACCENT_DIM,
                wraplength=450, anchor="w", justify="left"
            ).pack(anchor="w", pady=(4, 0))

        # Rules (optional): concise constraints for this mod.
        rules = mod.get("rules", [])
        if isinstance(rules, str):
            rules = [rules]
        if rules:
            ctk.CTkLabel(
                info_inner,
                text="Rules:",
                font=FONT_TINY,
                text_color=C_ACCENT,
                anchor="w",
                justify="left",
            ).pack(anchor="w", pady=(4, 0))
            for rule in rules[:6]:
                ctk.CTkLabel(
                    info_inner,
                    text=f"- {str(rule)}",
                    font=FONT_TINY,
                    text_color=C_TEXT_DIM,
                    wraplength=450,
                    anchor="w",
                    justify="left",
                ).pack(anchor="w", pady=(1, 0))

        # Commands list
        commands_full = mod.get("commands_full", [])
        if commands_full:
            cmd_card = HUDFrame(left_col, glow_color=C_BORDER)
            cmd_card.grid(row=1, column=0, sticky="ew", pady=(0, 4))

            cmd_inner = ctk.CTkFrame(cmd_card, fg_color="transparent")
            cmd_inner.pack(fill="x", padx=10, pady=8)

            SelectableLabel(
                cmd_inner, text="COMMANDS", font=FONT_SECTION,
                text_color=C_TEXT_BRIGHT
            ).pack(anchor="w", pady=(0, 4))

            for cmd in commands_full:
                cmd_name = cmd.get("name", "")
                cmd_desc = cmd.get("description", "")
                cmd_args = cmd.get("args", {})
                row = ctk.CTkFrame(cmd_inner, fg_color="transparent")
                row.pack(fill="x", pady=1)
                SelectableLabel(
                    row, text=cmd_name, font=FONT_SMALL,
                    text_color=C_ACCENT, width=120, anchor="w"
                ).pack(side="left")
                if cmd_desc:
                    SelectableLabel(
                        row, text=cmd_desc, font=FONT_TINY,
                        text_color=C_TEXT_DIM, anchor="w"
                    ).pack(side="left", padx=(8, 0))

                # Show args for commands that have them
                if cmd_args:
                    for arg_name, arg_info in cmd_args.items():
                        if isinstance(arg_info, dict):
                            arg_type = arg_info.get("type", "")
                            required = arg_info.get("required", False)
                            arg_desc = arg_info.get("description", "")
                            req_tag = "*" if required else ""
                            arg_text = f"  {arg_name}{req_tag}"
                            if arg_type:
                                arg_text += f" ({arg_type})"
                            if arg_desc:
                                arg_text += f" - {arg_desc}"
                        else:
                            arg_text = f"  {arg_name}"
                        arg_row = ctk.CTkFrame(
                            cmd_inner, fg_color="transparent")
                        arg_row.pack(fill="x", pady=0)
                        SelectableLabel(
                            arg_row, text=arg_text,
                            font=FONT_TINY,
                            text_color=C_TEXT_DIM, anchor="w"
                        ).pack(side="left", padx=(16, 0))

        # UI widgets from mod.json
        ui_spec = mod.get("ui", {})
        widgets_spec = ui_spec.get("widgets", [])
        if widgets_spec:
            ui_card = HUDFrame(left_col, glow_color=C_ACCENT_DIM)
            ui_card.grid(row=2, column=0, sticky="nsew")

            ui_inner = ctk.CTkFrame(ui_card, fg_color="transparent")
            ui_inner.pack(fill="both", expand=True, padx=10, pady=8)

            SelectableLabel(
                ui_inner, text="INTERFACE", font=FONT_SECTION,
                text_color=C_TEXT_BRIGHT
            ).pack(anchor="w", pady=(0, 6))

            # Store widget references for command sending
            mod["_ui_widgets"] = {}

            for widget in widgets_spec:
                w_type = widget.get("type", "")
                w_id = widget.get("id", "")
                w_label = widget.get("label", w_id)

                if w_type == "text_input":
                    SelectableLabel(
                        ui_inner, text=w_label, font=FONT_TINY,
                        text_color=C_TEXT_DIM
                    ).pack(anchor="w", pady=(4, 0))
                    entry = themed_entry(ui_inner)
                    entry.pack(fill="x", pady=(0, 4))
                    mod["_ui_widgets"][w_id] = entry

                elif w_type == "text_area":
                    SelectableLabel(
                        ui_inner, text=w_label, font=FONT_TINY,
                        text_color=C_TEXT_DIM
                    ).pack(anchor="w", pady=(4, 0))
                    rows = widget.get("rows", 3)
                    tb = ctk.CTkTextbox(
                        ui_inner, height=rows * 28, font=FONT_MONO,
                        fg_color=C_INPUT,
                        border_color=C_ACCENT_DIM,
                        border_width=1,
                        text_color=C_TEXT_BRIGHT,
                        corner_radius=2, wrap="word")
                    tb.pack(fill="x", pady=(0, 4))
                    # Enable undo/redo for text areas
                    tb._textbox.configure(undo=True, maxundo=-1)
                    mod["_ui_widgets"][w_id] = tb

                elif w_type == "number":
                    num_row = ctk.CTkFrame(
                        ui_inner, fg_color="transparent")
                    num_row.pack(fill="x", pady=(4, 4))
                    SelectableLabel(
                        num_row, text=w_label, font=FONT_TINY,
                        text_color=C_TEXT_DIM
                    ).pack(side="left")
                    default = str(widget.get("default", ""))
                    entry = themed_entry(num_row, width=100)
                    entry.insert(0, default)
                    entry.pack(side="right")
                    mod["_ui_widgets"][w_id] = entry

                elif w_type == "button":
                    cmd_name = widget.get("command", "")
                    themed_button(
                        ui_inner, w_label.upper(),
                        style="action",
                        width=140, height=36,
                        font=FONT_SECTION,
                        command=lambda m=mod, c=cmd_name: (
                            self._send_mod_command(m, c))
                    ).pack(anchor="w", pady=(4, 4))

                elif w_type == "dropdown":
                    SelectableLabel(
                        ui_inner, text=w_label, font=FONT_TINY,
                        text_color=C_TEXT_DIM
                    ).pack(anchor="w", pady=(4, 0))
                    options = widget.get("options", ["Default"])
                    dropdown = themed_dropdown(
                        ui_inner, options, width=200, height=34)
                    default = widget.get("default", "")
                    if default and default in options:
                        dropdown.set(default)
                    dropdown.pack(anchor="w", pady=(0, 4))
                    mod["_ui_widgets"][w_id] = dropdown

                elif w_type == "checkbox":
                    var = ctk.BooleanVar(
                        value=widget.get("default", False))
                    cb = ctk.CTkCheckBox(
                        ui_inner, text=w_label,
                        font=FONT_SMALL,
                        text_color=C_TEXT,
                        fg_color=C_ACCENT_DIM,
                        hover_color=C_ACCENT_MUTED,
                        checkmark_color=C_ACCENT,
                        variable=var)
                    cb.pack(anchor="w", pady=(4, 4))
                    # Store the var so we can read it later
                    cb._mod_var = var
                    mod["_ui_widgets"][w_id] = cb

        # --- Right column: output log ---
        log_panel = HUDFrame(page, glow_color=C_BORDER)
        log_panel.grid(row=1, column=1, sticky="nsew",
                       padx=(4, 10), pady=4)

        log_header = ctk.CTkFrame(log_panel, fg_color="transparent")
        log_header.pack(fill="x", padx=8, pady=(8, 4))

        SectionLabel(log_header, "Output", color=C_GREEN).pack(
            side="left")

        clear_log_btn = themed_button(
            log_header, "CLEAR", style="secondary",
            width=60, height=24,
            font=FONT_TINY,
            command=lambda m=mod: m["_log"].clear())
        clear_log_btn.pack(side="right")
        Tooltip(clear_log_btn, "Clear output log")

        mod["_log"] = SelectableTextbox(
            log_panel, font=FONT_MONO,
            fg_color=C_PANEL, text_color=C_GREEN,
            border_width=0, corner_radius=2)
        mod["_log"].pack(
            fill="both", expand=True, padx=6, pady=(0, 6))

        # S643: Dynamic wraplength — labels adapt to window width
        _mod_wrap_labels: list = []
        _mod_resize_timer = [None]  # mutable container for debounce
        _mod_last_w = [0]  # skip no-op reconfigures

        def _collect_mod_wrap(widget):
            for child in widget.winfo_children():
                if isinstance(child, ctk.CTkLabel):
                    try:
                        wl = child.cget("wraplength")
                        if wl and int(wl) > 0:
                            _mod_wrap_labels.append(child)
                    except Exception:
                        pass
                _collect_mod_wrap(child)

        _collect_mod_wrap(left_col)

        def _apply_mod_wraplength(w):
            _mod_resize_timer[0] = None
            _mod_last_w[0] = w
            for lbl in _mod_wrap_labels:
                lbl.configure(wraplength=w)

        def _on_mod_resize(event):
            w = event.width - 40
            if w < 100 or w == _mod_last_w[0]:
                return
            # Debounce: cancel pending, schedule after 100ms
            if _mod_resize_timer[0] is not None:
                try:
                    self.after_cancel(_mod_resize_timer[0])
                except (ValueError, Exception):
                    pass
            _mod_resize_timer[0] = self.after(
                100, lambda: _apply_mod_wraplength(w))

        left_col.bind("<Configure>", _on_mod_resize)
