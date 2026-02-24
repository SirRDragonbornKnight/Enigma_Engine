"""
Enigma Engine - Brick Page Builder
=====================================

Mixin providing per-brick page construction for EnigmaGUI.
Extracted from gui_pages.py to keep files under 800 lines.
"""
from __future__ import annotations

import customtkinter as ctk

from enigma_engine.gui.widgets import (
    C_ACCENT, C_ACCENT_DIM, C_ACCENT_MUTED, C_BORDER,
    C_BORDER_ACCENT, C_GREEN, C_INPUT, C_PANEL, C_RED,
    C_SURFACE, C_TEXT, C_TEXT_BRIGHT, C_TEXT_DIM,
    FONT_MONO, FONT_SECTION, FONT_SMALL, FONT_TINY,
    HUDFrame, SectionLabel, SelectableTextbox, StatusDot,
    themed_entry, themed_dropdown,
)


class BrickPageMixin:
    """Mixin providing brick page builder for EnigmaGUI.

    Expects the host class to have:
    - _make_page, _start_brick, _stop_brick, _send_brick_command
    """

    # ================================================================
    # PAGE: BRICK - Per-brick pages built from brick.json
    # ================================================================

    def _build_page_brick(self, brick: dict):
        """Build a dedicated page for a brick module.

        Layout:
          Top bar: brick name + START / STOP buttons
          Left:    info card, commands list, UI widgets
          Right:   output log
        """
        page_name = f"BRICK_{brick['id']}"
        page = self._make_page(page_name)
        page.grid_columnconfigure(0, weight=1)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(1, weight=1)

        # --- Top bar ---
        top = ctk.CTkFrame(page, fg_color="transparent", height=48)
        top.grid(row=0, column=0, columnspan=2, sticky="ew",
                 padx=10, pady=(8, 2))

        SectionLabel(top, brick["name"]).pack(
            side="left", fill="x", expand=True)

        # Status dot + label
        brick["_page_dot"] = StatusDot(top, color=C_TEXT_DIM)
        brick["_page_dot"].pack(side="right", padx=(4, 0))

        brick["_page_status"] = ctk.CTkLabel(
            top, text="STOPPED", font=FONT_TINY,
            text_color=C_TEXT_DIM)
        brick["_page_status"].pack(side="right", padx=(0, 4))

        brick["_stop_btn"] = ctk.CTkButton(
            top, text="STOP", width=80, height=34,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_SURFACE, hover_color=C_RED,
            text_color=C_TEXT_DIM,
            command=lambda b=brick: self._stop_brick(b),
            state="disabled")
        brick["_stop_btn"].pack(side="right", padx=(4, 8))

        brick["_start_btn"] = ctk.CTkButton(
            top, text="START", width=80, height=34,
            font=FONT_SMALL, corner_radius=2,
            fg_color=C_ACCENT_DIM, hover_color=C_ACCENT_MUTED,
            text_color=C_ACCENT,
            command=lambda b=brick: self._start_brick(b))
        brick["_start_btn"].pack(side="right")

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

        ctk.CTkLabel(
            info_inner, text=brick["name"], font=FONT_SECTION,
            text_color=C_TEXT_BRIGHT
        ).pack(anchor="w")

        desc = brick.get("description", "")
        if desc:
            ctk.CTkLabel(
                info_inner, text=desc, font=FONT_SMALL,
                text_color=C_TEXT, wraplength=500, anchor="w",
                justify="left"
            ).pack(anchor="w", pady=(2, 0))

        detail_line = (
            f"v{brick.get('version', '?')}  //  "
            f"Port {brick.get('port', '?')}")
        ctk.CTkLabel(
            info_inner, text=detail_line, font=FONT_TINY,
            text_color=C_TEXT_DIM
        ).pack(anchor="w", pady=(4, 0))

        # Dependencies
        deps = brick.get("dependencies", [])
        if deps:
            ctk.CTkLabel(
                info_inner,
                text=f"Requires: {', '.join(deps)}",
                font=FONT_TINY, text_color=C_TEXT_DIM
            ).pack(anchor="w", pady=(2, 0))

        # Settings summary
        settings = brick.get("settings", {})
        if settings:
            settings_keys = list(settings.keys())
            if len(settings_keys) > 4:
                summary = ", ".join(settings_keys[:4]) + "..."
            else:
                summary = ", ".join(settings_keys)
            ctk.CTkLabel(
                info_inner,
                text=f"Settings: {summary}",
                font=FONT_TINY, text_color=C_TEXT_DIM
            ).pack(anchor="w", pady=(2, 0))

        # AI prompt hint (tells the AI how to use this brick)
        prompt = brick.get("prompt", "")
        if prompt:
            ctk.CTkLabel(
                info_inner,
                text=f"AI usage: {prompt}",
                font=FONT_TINY, text_color=C_ACCENT_DIM,
                wraplength=450, anchor="w", justify="left"
            ).pack(anchor="w", pady=(4, 0))

        # Commands list
        commands_full = brick.get("commands_full", [])
        if commands_full:
            cmd_card = HUDFrame(left_col, glow_color=C_BORDER)
            cmd_card.grid(row=1, column=0, sticky="ew", pady=(0, 4))

            cmd_inner = ctk.CTkFrame(cmd_card, fg_color="transparent")
            cmd_inner.pack(fill="x", padx=10, pady=8)

            ctk.CTkLabel(
                cmd_inner, text="COMMANDS", font=FONT_SECTION,
                text_color=C_TEXT_BRIGHT
            ).pack(anchor="w", pady=(0, 4))

            for cmd in commands_full:
                cmd_name = cmd.get("name", "")
                cmd_desc = cmd.get("description", "")
                cmd_args = cmd.get("args", {})
                row = ctk.CTkFrame(cmd_inner, fg_color="transparent")
                row.pack(fill="x", pady=1)
                ctk.CTkLabel(
                    row, text=cmd_name, font=FONT_SMALL,
                    text_color=C_ACCENT, width=120, anchor="w"
                ).pack(side="left")
                if cmd_desc:
                    ctk.CTkLabel(
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
                        ctk.CTkLabel(
                            arg_row, text=arg_text,
                            font=FONT_TINY,
                            text_color=C_TEXT_DIM, anchor="w"
                        ).pack(side="left", padx=(16, 0))

        # UI widgets from brick.json
        ui_spec = brick.get("ui", {})
        widgets_spec = ui_spec.get("widgets", [])
        if widgets_spec:
            ui_card = HUDFrame(left_col, glow_color=C_ACCENT_DIM)
            ui_card.grid(row=2, column=0, sticky="nsew")

            ui_inner = ctk.CTkFrame(ui_card, fg_color="transparent")
            ui_inner.pack(fill="both", expand=True, padx=10, pady=8)

            ctk.CTkLabel(
                ui_inner, text="INTERFACE", font=FONT_SECTION,
                text_color=C_TEXT_BRIGHT
            ).pack(anchor="w", pady=(0, 6))

            # Store widget references for command sending
            brick["_ui_widgets"] = {}

            for widget in widgets_spec:
                w_type = widget.get("type", "")
                w_id = widget.get("id", "")
                w_label = widget.get("label", w_id)

                if w_type == "text_input":
                    ctk.CTkLabel(
                        ui_inner, text=w_label, font=FONT_TINY,
                        text_color=C_TEXT_DIM
                    ).pack(anchor="w", pady=(4, 0))
                    entry = themed_entry(ui_inner)
                    entry.pack(fill="x", pady=(0, 4))
                    brick["_ui_widgets"][w_id] = entry

                elif w_type == "text_area":
                    ctk.CTkLabel(
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
                    brick["_ui_widgets"][w_id] = tb

                elif w_type == "number":
                    num_row = ctk.CTkFrame(
                        ui_inner, fg_color="transparent")
                    num_row.pack(fill="x", pady=(4, 4))
                    ctk.CTkLabel(
                        num_row, text=w_label, font=FONT_TINY,
                        text_color=C_TEXT_DIM
                    ).pack(side="left")
                    default = str(widget.get("default", ""))
                    entry = themed_entry(num_row, width=100)
                    entry.insert(0, default)
                    entry.pack(side="right")
                    brick["_ui_widgets"][w_id] = entry

                elif w_type == "button":
                    cmd_name = widget.get("command", "")
                    ctk.CTkButton(
                        ui_inner, text=w_label.upper(),
                        width=140, height=36,
                        font=FONT_SECTION, corner_radius=2,
                        fg_color=C_ACCENT_DIM,
                        hover_color=C_ACCENT_MUTED,
                        text_color=C_ACCENT,
                        command=lambda b=brick, c=cmd_name: (
                            self._send_brick_command(b, c))
                    ).pack(anchor="w", pady=(4, 4))

                elif w_type == "dropdown":
                    ctk.CTkLabel(
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
                    brick["_ui_widgets"][w_id] = dropdown

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
                    cb._brick_var = var
                    brick["_ui_widgets"][w_id] = cb

        # --- Right column: output log ---
        log_panel = HUDFrame(page, glow_color=C_BORDER)
        log_panel.grid(row=1, column=1, sticky="nsew",
                       padx=(4, 10), pady=4)

        SectionLabel(log_panel, "Output", color=C_GREEN).pack(
            anchor="w", padx=8, pady=(8, 4))

        brick["_log"] = SelectableTextbox(
            log_panel, font=FONT_MONO,
            fg_color=C_PANEL, text_color=C_GREEN,
            border_width=0, corner_radius=2)
        brick["_log"].pack(
            fill="both", expand=True, padx=6, pady=(0, 6))
