"""
Enigma Engine - Widget Library
========================================

Reusable components for the desktop GUI.
"""
from __future__ import annotations

import customtkinter as ctk


# -------------------------------------------------------------------
# Color Palette - Dark theme with silver/gray accent
# -------------------------------------------------------------------
C_BG = "#080808"
C_PANEL = "#0e0e0e"
C_SURFACE = "#181818"
C_INPUT = "#1c1c1c"
C_ACCENT = "#8B95A5"
C_ACCENT_DIM = "#2a2a2a"
C_ACCENT_MUTED = "#3d3d3d"
C_PURPLE = "#a855f7"
C_CYAN = "#22d3ee"
C_TEXT = "#b0b0b0"
C_TEXT_DIM = "#555555"
C_TEXT_BRIGHT = "#e8e8e8"
C_GREEN = "#22c55e"
C_GREEN_DIM = "#0e3a1e"
C_RED = "#ef4444"
C_ORANGE = "#f97316"
C_BORDER = "#1f1f1f"
C_BORDER_ACCENT = "#2e2e2e"

# -------------------------------------------------------------------
# Fonts - monospace, all sizes +5 from base
# -------------------------------------------------------------------
FONT_TITLE = ("Consolas", 26, "bold")
FONT_SECTION = ("Consolas", 17, "bold")
FONT_BODY = ("Consolas", 16)
FONT_SMALL = ("Consolas", 15)
FONT_TINY = ("Consolas", 14)
FONT_CHAT = ("Consolas", 16)
FONT_INPUT = ("Consolas", 17)
FONT_MONO = ("Consolas", 16)
FONT_CMD = ("Consolas", 15)


# -------------------------------------------------------------------
# Widget factories - reduce per-call boilerplate
# -------------------------------------------------------------------

def themed_entry(parent, width=140, height=34, **kw):
    """CTkEntry with standard dark theme applied."""
    kw.setdefault("font", FONT_MONO)
    kw.setdefault("fg_color", C_INPUT)
    kw.setdefault("border_color", C_ACCENT_DIM)
    kw.setdefault("border_width", 1)
    kw.setdefault("text_color", C_TEXT_BRIGHT)
    kw.setdefault("corner_radius", 2)
    return ctk.CTkEntry(parent, width=width, height=height, **kw)


def themed_dropdown(parent, values, width=180, height=32, **kw):
    """CTkOptionMenu with standard dark theme applied."""
    kw.setdefault("font", FONT_SMALL)
    kw.setdefault("fg_color", C_INPUT)
    kw.setdefault("button_color", C_ACCENT_DIM)
    kw.setdefault("button_hover_color", C_ACCENT_MUTED)
    return ctk.CTkOptionMenu(
        parent, values=values, width=width, height=height, **kw)


def themed_scroll(parent, **kw):
    """CTkScrollableFrame with standard dark theme applied."""
    kw.setdefault("fg_color", C_PANEL)
    kw.setdefault("corner_radius", 2)
    kw.setdefault("scrollbar_button_color", C_ACCENT_DIM)
    kw.setdefault("scrollbar_button_hover_color", C_ACCENT_MUTED)
    return ctk.CTkScrollableFrame(parent, **kw)


# -------------------------------------------------------------------
# HUDFrame - bordered panel
# -------------------------------------------------------------------

class HUDFrame(ctk.CTkFrame):
    """Panel with accent border."""

    def __init__(self, master, glow_color=C_ACCENT_DIM, **kwargs):
        kwargs.setdefault("fg_color", C_PANEL)
        kwargs.setdefault("border_color", glow_color)
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("corner_radius", 2)
        super().__init__(master, **kwargs)


# Backward compat alias
GlowFrame = HUDFrame


# -------------------------------------------------------------------
# StatusDot - color indicator
# -------------------------------------------------------------------

class StatusDot(ctk.CTkLabel):
    """Colored dot indicator."""

    def __init__(self, master, color=C_TEXT_DIM, **kwargs):
        kwargs.setdefault("text", "\u25cf")
        kwargs.setdefault("font", ("Consolas", 15))
        kwargs.setdefault("text_color", color)
        kwargs.setdefault("width", 20)
        super().__init__(master, **kwargs)

    def set_color(self, color: str):
        self.configure(text_color=color)


# -------------------------------------------------------------------
# NavButton - side rail navigation button
# -------------------------------------------------------------------

class NavButton(ctk.CTkFrame):
    """Navigation button with left-edge active indicator."""

    def __init__(self, master, label: str, on_click, **kwargs):
        super().__init__(master, fg_color="transparent", height=44,
                         corner_radius=0)
        self.pack_propagate(False)

        # Left accent bar (visible when active)
        self._bar = ctk.CTkFrame(
            self, width=3, fg_color="transparent", corner_radius=0)
        self._bar.pack(side="left", fill="y")

        self._btn = ctk.CTkButton(
            self, text=f"  {label}", height=44,
            font=FONT_SECTION, anchor="w",
            fg_color="transparent", text_color=C_TEXT_DIM,
            hover_color=C_SURFACE, corner_radius=0,
            command=on_click)
        self._btn.pack(side="left", fill="both", expand=True)

    def set_active(self, active: bool):
        if active:
            self._bar.configure(fg_color=C_ACCENT)
            self._btn.configure(
                fg_color=C_SURFACE, text_color=C_TEXT_BRIGHT)
        else:
            self._bar.configure(fg_color="transparent")
            self._btn.configure(
                fg_color="transparent", text_color=C_TEXT_DIM)


# -------------------------------------------------------------------
# SectionLabel - header with accent line
# -------------------------------------------------------------------

class SectionLabel(ctk.CTkFrame):
    """Header: TITLE --------------------------------"""

    def __init__(self, master, text: str, color=C_ACCENT, **kwargs):
        kwargs.setdefault("fg_color", "transparent")
        kwargs.setdefault("height", 34)
        super().__init__(master, **kwargs)

        ctk.CTkLabel(
            self, text=text.upper(), font=FONT_SECTION,
            text_color=color
        ).pack(side="left")

        line = ctk.CTkFrame(
            self, height=1, fg_color=C_BORDER_ACCENT, corner_radius=0)
        line.pack(side="left", fill="x", expand=True, padx=(12, 0))


# -------------------------------------------------------------------
# ToggleButton - icon toggle with on/off state
# -------------------------------------------------------------------

class ToggleButton(ctk.CTkButton):
    """Square button that toggles between two states."""

    def __init__(self, master, text_on: str, text_off: str,
                 on_toggle=None, start_on: bool = False, **kwargs):
        self._on = start_on
        self._text_on = text_on
        self._text_off = text_off
        self._on_toggle = on_toggle

        kwargs.setdefault("width", 46)
        kwargs.setdefault("height", 42)
        kwargs.setdefault("font", FONT_BODY)
        kwargs.setdefault("corner_radius", 2)
        super().__init__(
            master, text=text_on if start_on else text_off,
            command=self._toggle, **kwargs)
        self._apply_style()

    @property
    def is_on(self) -> bool:
        return self._on

    def _toggle(self):
        self._on = not self._on
        self._apply_style()
        if self._on_toggle:
            self._on_toggle(self._on)

    def _apply_style(self):
        if self._on:
            self.configure(
                text=self._text_on,
                fg_color=C_GREEN_DIM, text_color=C_GREEN)
        else:
            self.configure(
                text=self._text_off,
                fg_color=C_SURFACE, text_color=C_TEXT_DIM)


# -------------------------------------------------------------------
# StatusBar - bottom bar with system readouts
# -------------------------------------------------------------------

class CollapsiblePanel(ctk.CTkFrame):
    """Panel with a clickable header that toggles content visibility.

    The header shows the title and a chevron indicator.
    Click the header to expand or collapse the content area.
    When collapsed, only the thin header row is visible.

    Usage:
        panel = CollapsiblePanel(parent, "HISTORY", color=C_PURPLE)
        panel.pack(fill="both", expand=True)
        # Add widgets into panel.content
        my_widget = ctk.CTkTextbox(panel.content, ...)
        my_widget.pack(fill="both", expand=True)
    """

    def __init__(
        self, master, title: str, color=C_ACCENT,
        start_expanded: bool = True, on_toggle=None, **kwargs,
    ):
        kwargs.setdefault("fg_color", "transparent")
        kwargs.setdefault("corner_radius", 0)
        super().__init__(master, **kwargs)

        self._expanded = start_expanded
        self._on_toggle = on_toggle
        self._color = color

        # Header row (always visible)
        self._header = ctk.CTkFrame(
            self, fg_color=C_PANEL, height=32, corner_radius=2,
            border_width=1, border_color=C_BORDER)
        self._header.pack(fill="x")
        self._header.pack_propagate(False)

        self._chevron = ctk.CTkLabel(
            self._header, text="\u25bc" if start_expanded else "\u25b6",
            font=("Consolas", 12), text_color=C_TEXT_DIM, width=20)
        self._chevron.pack(side="left", padx=(8, 0))

        self._title_label = ctk.CTkLabel(
            self._header, text=title.upper(),
            font=FONT_SMALL, text_color=color)
        self._title_label.pack(side="left", padx=(4, 0))

        # Make header clickable
        for widget in (self._header, self._chevron, self._title_label):
            widget.bind("<Button-1>", lambda e: self.toggle())

        # Content area (shown/hidden)
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        if start_expanded:
            self.content.pack(fill="both", expand=True, pady=(2, 0))

    @property
    def is_expanded(self) -> bool:
        """Whether the content area is currently visible."""
        return self._expanded

    def toggle(self):
        """Toggle between expanded and collapsed."""
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    def expand(self):
        """Show the content area."""
        if self._expanded:
            return
        self._expanded = True
        self._chevron.configure(text="\u25bc")
        self.content.pack(fill="both", expand=True, pady=(2, 0))
        if self._on_toggle:
            self._on_toggle(True)

    def collapse(self):
        """Hide the content area."""
        if not self._expanded:
            return
        self._expanded = False
        self._chevron.configure(text="\u25b6")
        self.content.pack_forget()
        if self._on_toggle:
            self._on_toggle(False)


# -------------------------------------------------------------------
# SelectableTextbox - read-only but allows text selection and copy
# -------------------------------------------------------------------

class SelectableTextbox(ctk.CTkTextbox):
    """Read-only CTkTextbox that allows text selection and copy.

    Unlike state='disabled', this widget lets users click, drag to
    select text, and copy via Ctrl+C. All editing keys are blocked
    so content cannot be modified by the user.

    Existing code that toggles state='normal'/state='disabled' around
    inserts continues to work because configure() silently ignores
    state changes — the widget stays 'normal' internally.

    Convenience methods write() and clear() are also provided.
    """

    def __init__(self, master, **kwargs):
        # Strip state param — always stay "normal" for selection
        kwargs.pop("state", None)
        super().__init__(master, **kwargs)
        # Block editing keys on the underlying tkinter Text widget
        self._textbox.bind("<Key>", self._on_key)
        # Right-click copy menu
        self._textbox.bind("<Button-3>", self._show_copy_menu)

    def _on_key(self, event):
        """Allow selection and copy shortcuts, block editing."""
        # Allow Ctrl+C (copy) and Ctrl+A (select all)
        if event.state & 4:  # Ctrl key held
            if event.keysym.lower() in ("c", "a"):
                return
        # Allow navigation keys (with or without Shift for selection)
        if event.keysym in (
                "Left", "Right", "Up", "Down",
                "Home", "End", "Prior", "Next",
                "Shift_L", "Shift_R",
                "Control_L", "Control_R"):
            return
        # Block all other keys (typing, delete, backspace, etc.)
        return "break"

    def _show_copy_menu(self, event):
        """Show a right-click context menu with Copy and Select All."""
        import tkinter as tk
        menu = tk.Menu(self, tearoff=0)
        try:
            sel = self._textbox.get("sel.first", "sel.last")
            if sel:
                menu.add_command(
                    label="Copy",
                    command=self._copy_selected)
        except Exception:
            pass
        menu.add_command(
            label="Select All",
            command=self._select_all)
        menu.tk_popup(event.x_root, event.y_root)

    def _copy_selected(self):
        """Copy selected text to clipboard."""
        try:
            sel = self._textbox.get("sel.first", "sel.last")
            self.clipboard_clear()
            self.clipboard_append(sel)
        except Exception:
            pass

    def _select_all(self):
        """Select all text in the widget."""
        self._textbox.tag_add("sel", "1.0", "end")

    def configure(self, **kwargs):
        """Override to ignore state changes — always stay normal."""
        kwargs.pop("state", None)
        super().configure(**kwargs)

    def cget(self, attribute):
        if attribute == "state":
            return "normal"
        return super().cget(attribute)

    def write(self, text: str, tag: str | None = None):
        """Insert text at the end. Convenience method."""
        if tag:
            self._textbox.insert("end", text, tag)
        else:
            self._textbox.insert("end", text)
        self.see("end")

    def clear(self):
        """Remove all text. Convenience method."""
        self._textbox.delete("1.0", "end")


class StatusBar(ctk.CTkFrame):
    """Bottom status bar showing system info."""

    def __init__(self, master, **kwargs):
        kwargs.setdefault("height", 30)
        kwargs.setdefault("fg_color", C_PANEL)
        kwargs.setdefault("corner_radius", 0)
        super().__init__(master, **kwargs)
        self.pack_propagate(False)

        ctk.CTkFrame(
            self, height=1, fg_color=C_BORDER,
            corner_radius=0).pack(fill="x", side="top")

        self._left = ctk.CTkLabel(
            self, text="", font=FONT_TINY, text_color=C_TEXT_DIM)
        self._left.pack(side="left", padx=12)

        self._right = ctk.CTkLabel(
            self, text="", font=FONT_TINY, text_color=C_TEXT_DIM)
        self._right.pack(side="right", padx=12)

        self._center = ctk.CTkLabel(
            self, text="", font=FONT_TINY, text_color=C_TEXT_DIM)
        self._center.pack(expand=True)

    def set_left(self, text: str):
        self._left.configure(text=text)

    def set_right(self, text: str):
        self._right.configure(text=text)

    def set_center(self, text: str):
        self._center.configure(text=text)


# -------------------------------------------------------------------
# Tooltip - hover description for any widget
# -------------------------------------------------------------------

class Tooltip:
    """Displays a small tooltip near the mouse when hovering a widget.

    Usage:
        Tooltip(my_button, "Send your message")
    """

    def __init__(self, widget, text: str, delay: int = 400):
        self._widget = widget
        self._text = text
        self._delay = delay
        self._tip_window = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._cancel, add="+")
        widget.bind("<ButtonPress>", self._cancel, add="+")

    def _schedule(self, event=None):
        self._cancel()
        self._after_id = self._widget.after(
            self._delay, self._show)

    def _cancel(self, event=None):
        if self._after_id:
            self._widget.after_cancel(self._after_id)
            self._after_id = None
        self._hide()

    def _show(self):
        if self._tip_window:
            return
        import tkinter as tk
        x = self._widget.winfo_rootx() + 10
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip_window = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        label = tk.Label(
            tw, text=self._text, justify="left",
            background="#1c1c1c", foreground="#b0b0b0",
            relief="solid", borderwidth=1,
            font=("Consolas", 12), padx=6, pady=3)
        label.pack()

    def _hide(self):
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None
