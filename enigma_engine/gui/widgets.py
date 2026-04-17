"""
Enigma Engine - Widget Library
========================================

Reusable components for the desktop GUI.
"""
from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from enigma_engine.gui.themes import load_active_theme

# -------------------------------------------------------------------
# Color Palette - loaded from the active theme at import time
# -------------------------------------------------------------------
_theme = load_active_theme()

C_BG = _theme.bg
C_PANEL = _theme.panel
C_SURFACE = _theme.surface
C_INPUT = _theme.input
C_ACCENT = _theme.accent
C_ACCENT_DIM = _theme.accent_dim
C_ACCENT_MUTED = _theme.accent_muted
C_PURPLE = _theme.purple
C_PURPLE_DIM = _theme.purple_dim
C_PURPLE_MUTED = _theme.purple_muted
C_CYAN = _theme.cyan
C_TEXT = _theme.text
C_TEXT_DIM = _theme.text_dim
C_TEXT_BRIGHT = _theme.text_bright
C_GREEN = _theme.green
C_GREEN_DIM = _theme.green_dim
C_GREEN_HOVER = _theme.green_hover
C_RED = _theme.red
C_RED_DIM = _theme.red_dim
C_RED_HOVER = _theme.red_hover
C_ORANGE = _theme.orange
C_ORANGE_DIM = _theme.orange_dim
C_ORANGE_HOVER = _theme.orange_hover
C_CYAN_DIM = _theme.cyan_dim
C_CYAN_HOVER = _theme.cyan_hover
C_BORDER = _theme.border
C_BORDER_ACCENT = _theme.border_accent

# App version — single source in enigma_engine/__init__.py
from enigma_engine import __version__ as VERSION  # noqa: F401 (re-exported)

# -------------------------------------------------------------------
# Fonts - monospace, all sizes +5 from base
# -------------------------------------------------------------------
FONT_FAMILY = "Consolas"

# Base sizes before any offset
_FONT_BASE_SIZES = {
    "FONT_TITLE": 26,
    "FONT_SECTION": 17,
    "FONT_BODY": 16,
    "FONT_SMALL": 15,
    "FONT_TINY": 14,
    "FONT_CHAT": 16,
    "FONT_INPUT": 17,
    "FONT_MONO": 16,
    "FONT_CMD": 15,
}
_font_size_offset: int = 0

FONT_TITLE = (FONT_FAMILY, 26, "bold")
FONT_SECTION = (FONT_FAMILY, 17, "bold")
FONT_BODY = (FONT_FAMILY, 16)
FONT_SMALL = (FONT_FAMILY, 15)
FONT_TINY = (FONT_FAMILY, 14)
FONT_CHAT = (FONT_FAMILY, 16)
FONT_INPUT = (FONT_FAMILY, 17)
FONT_MONO = (FONT_FAMILY, 16)
FONT_CMD = (FONT_FAMILY, 15)


def get_font_size_offset() -> int:
    """Return the current font size offset (0 = default)."""
    return _font_size_offset


def set_font_size_offset(offset: int) -> None:
    """Apply a font size offset to all FONT_* tuples.

    Offset is added to each base size. For example, offset=2
    makes FONT_BODY go from size 16 to 18. Offset can be negative.
    Clamped to [-4, 8] range for safety.
    """
    global _font_size_offset  # noqa: PLW0603
    global FONT_TITLE, FONT_SECTION, FONT_BODY, FONT_SMALL  # noqa: PLW0603
    global FONT_TINY, FONT_CHAT, FONT_INPUT, FONT_MONO, FONT_CMD  # noqa: PLW0603

    offset = max(-4, min(8, offset))
    _font_size_offset = offset

    FONT_TITLE = (FONT_FAMILY, _FONT_BASE_SIZES["FONT_TITLE"] + offset, "bold")
    FONT_SECTION = (FONT_FAMILY, _FONT_BASE_SIZES["FONT_SECTION"] + offset, "bold")
    FONT_BODY = (FONT_FAMILY, _FONT_BASE_SIZES["FONT_BODY"] + offset)
    FONT_SMALL = (FONT_FAMILY, _FONT_BASE_SIZES["FONT_SMALL"] + offset)
    FONT_TINY = (FONT_FAMILY, _FONT_BASE_SIZES["FONT_TINY"] + offset)
    FONT_CHAT = (FONT_FAMILY, _FONT_BASE_SIZES["FONT_CHAT"] + offset)
    FONT_INPUT = (FONT_FAMILY, _FONT_BASE_SIZES["FONT_INPUT"] + offset)
    FONT_MONO = (FONT_FAMILY, _FONT_BASE_SIZES["FONT_MONO"] + offset)
    FONT_CMD = (FONT_FAMILY, _FONT_BASE_SIZES["FONT_CMD"] + offset)


# Load font size offset from settings at import time
def _load_font_size_offset() -> None:
    """Load persisted font_size_offset from gui_settings.json."""
    import json as _json
    from pathlib import Path as _Path
    settings_path = (
        _Path(__file__).parent.parent.parent / "data" / "gui_settings.json")
    try:
        if settings_path.exists():
            data = _json.loads(
                settings_path.read_text(encoding="utf-8"))
            offset = data.get("font_size_offset", 0)
            if isinstance(offset, int) and offset != 0:
                set_font_size_offset(offset)
    except Exception:
        pass


_load_font_size_offset()


def reload_theme(name: str) -> dict[str, str]:
    """Switch to a new theme and update all C_* module constants.

    Returns a mapping of old hex colours (lowercase) to new ones
    so callers can walk existing widgets and remap their colours.
    Also propagates updated constants to all enigma_engine.gui.*
    modules that imported them.
    """
    import sys
    from dataclasses import fields as dc_fields
    from enigma_engine.gui.themes import Theme, get_theme

    global _theme  # noqa: PLW0603
    global C_BG, C_PANEL, C_SURFACE, C_INPUT  # noqa: PLW0603
    global C_ACCENT, C_ACCENT_DIM, C_ACCENT_MUTED  # noqa: PLW0603
    global C_PURPLE, C_PURPLE_DIM, C_PURPLE_MUTED  # noqa: PLW0603
    global C_CYAN, C_TEXT, C_TEXT_DIM, C_TEXT_BRIGHT  # noqa: PLW0603
    global C_GREEN, C_GREEN_DIM, C_GREEN_HOVER  # noqa: PLW0603
    global C_RED, C_RED_DIM, C_RED_HOVER  # noqa: PLW0603
    global C_ORANGE, C_ORANGE_DIM, C_ORANGE_HOVER  # noqa: PLW0603
    global C_CYAN_DIM, C_CYAN_HOVER  # noqa: PLW0603
    global C_BORDER, C_BORDER_ACCENT  # noqa: PLW0603

    old = _theme
    new = get_theme(name)

    # Build old→new colour mapping (lowercase for case-insensitive match)
    color_map: dict[str, str] = {}
    for f in dc_fields(Theme):
        if f.name == "name":
            continue
        old_val = getattr(old, f.name)
        new_val = getattr(new, f.name)
        if old_val != new_val:
            color_map[old_val.lower()] = new_val

    # Update module-level constants
    _theme = new
    C_BG = new.bg
    C_PANEL = new.panel
    C_SURFACE = new.surface
    C_INPUT = new.input
    C_ACCENT = new.accent
    C_ACCENT_DIM = new.accent_dim
    C_ACCENT_MUTED = new.accent_muted
    C_PURPLE = new.purple
    C_PURPLE_DIM = new.purple_dim
    C_PURPLE_MUTED = new.purple_muted
    C_CYAN = new.cyan
    C_TEXT = new.text
    C_TEXT_DIM = new.text_dim
    C_TEXT_BRIGHT = new.text_bright
    C_GREEN = new.green
    C_GREEN_DIM = new.green_dim
    C_GREEN_HOVER = new.green_hover
    C_RED = new.red
    C_RED_DIM = new.red_dim
    C_RED_HOVER = new.red_hover
    C_ORANGE = new.orange
    C_ORANGE_DIM = new.orange_dim
    C_ORANGE_HOVER = new.orange_hover
    C_CYAN_DIM = new.cyan_dim
    C_CYAN_HOVER = new.cyan_hover
    C_BORDER = new.border
    C_BORDER_ACCENT = new.border_accent

    # Rebuild button styles with updated colours
    global BUTTON_STYLES  # noqa: PLW0603
    BUTTON_STYLES = _build_button_styles()

    # Propagate to all GUI modules that imported C_* constants
    this_mod = sys.modules[__name__]
    c_names = [k for k in vars(this_mod) if k.startswith("C_")]
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod is this_mod:
            continue
        if not mod_name.startswith("enigma_engine.gui."):
            continue
        for c_name in c_names:
            if hasattr(mod, c_name):
                setattr(mod, c_name, getattr(this_mod, c_name))

    return color_map


# -------------------------------------------------------------------
# Hotkey helpers — Ctrl+Z / Ctrl+Y / Ctrl+A for all input widgets
# -------------------------------------------------------------------

class _EntryUndoStack:
    """Lightweight undo/redo stack for tk.Entry (which has no native undo)."""

    __slots__ = ("_stack", "_redo", "_lock")
    _MAX_DEPTH = 200

    def __init__(self) -> None:
        self._stack: list[str] = []
        self._redo: list[str] = []
        self._lock = False

    def push(self, text: str) -> None:
        if self._lock:
            return
        if self._stack and self._stack[-1] == text:
            return
        self._stack.append(text)
        if len(self._stack) > self._MAX_DEPTH:
            self._stack = self._stack[-self._MAX_DEPTH:]
        self._redo.clear()

    def undo(self, current: str) -> str | None:
        if not self._stack:
            return None
        # Save current for redo
        self._redo.append(current)
        return self._stack.pop()

    def redo(self) -> str | None:
        if not self._redo:
            return None
        return self._redo.pop()

    @property
    def lock(self) -> bool:
        return self._lock

    @lock.setter
    def lock(self, value: bool) -> None:
        self._lock = value


def wire_hotkeys(widget) -> None:
    """Bind Ctrl+Z (undo), Ctrl+Y (redo), and Ctrl+A (select all).

    Works for both ``CTkTextbox`` (tk.Text-backed) and ``CTkEntry``
    (tk.Entry-backed) widgets.
    """
    if isinstance(widget, ctk.CTkTextbox):
        _wire_textbox_hotkeys(widget)
    elif isinstance(widget, ctk.CTkEntry):
        _wire_entry_hotkeys(widget)


def _wire_textbox_hotkeys(widget: ctk.CTkTextbox) -> None:
    inner = widget._textbox
    inner.configure(undo=True, maxundo=-1, autoseparators=True)

    def _undo(event=None):
        try:
            inner.edit_undo()
        except tk.TclError:
            pass
        return "break"

    def _redo(event=None):
        try:
            inner.edit_redo()
        except tk.TclError:
            pass
        return "break"

    def _select_all(event=None):
        inner.tag_add("sel", "1.0", "end-1c")
        return "break"

    widget.bind("<Control-z>", _undo)
    widget.bind("<Control-Z>", _undo)
    widget.bind("<Control-y>", _redo)
    widget.bind("<Control-Y>", _redo)
    widget.bind("<Control-a>", _select_all)
    widget.bind("<Control-A>", _select_all)


def _wire_entry_hotkeys(widget: ctk.CTkEntry) -> None:
    inner = widget._entry
    stack = _EntryUndoStack()

    # Seed with initial content
    stack.push(inner.get())

    # Record every keystroke change
    def _on_change(*_args):
        stack.push(inner.get())

    sv = widget._textvariable if hasattr(widget, "_textvariable") else None
    if sv is not None:
        sv.trace_add("write", _on_change)
    else:
        inner.bind("<KeyRelease>", _on_change, add=True)

    def _undo(event=None):
        current = inner.get()
        prev = stack.undo(current)
        if prev is not None:
            stack.lock = True
            inner.delete(0, "end")
            inner.insert(0, prev)
            stack.lock = False
        return "break"

    def _redo(event=None):
        val = stack.redo()
        if val is not None:
            stack.lock = True
            inner.delete(0, "end")
            inner.insert(0, val)
            stack.lock = False
        return "break"

    def _select_all(event=None):
        inner.select_range(0, "end")
        inner.icursor("end")
        return "break"

    widget.bind("<Control-z>", _undo)
    widget.bind("<Control-Z>", _undo)
    widget.bind("<Control-y>", _redo)
    widget.bind("<Control-Y>", _redo)
    widget.bind("<Control-a>", _select_all)
    widget.bind("<Control-A>", _select_all)


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
    entry = ctk.CTkEntry(parent, width=width, height=height, **kw)
    wire_hotkeys(entry)
    # Select all text on focus so user can immediately type a new value
    entry.bind("<FocusIn>", lambda e: e.widget.after(
        10, lambda: e.widget.select_range(0, "end")))
    return entry


def themed_numeric_entry(
    parent, *, mode: str = "int",
    allow_auto: bool = False,
    width: int = 80, height: int = 34, **kw,
) -> ctk.CTkEntry:
    """CTkEntry that only accepts numeric input.

    Args:
        mode: ``"int"`` for integers only, ``"float"`` for decimals
              and scientific notation (e.g. ``5e-5``).
        allow_auto: When True the literal text ``auto`` is also
                    accepted (useful for batch-size fields).
    """
    entry = themed_entry(parent, width=width, height=height, **kw)

    # Keystroke filter — runs on every insertion/deletion
    def _validate_key(event):
        # Allow control keys (backspace, delete, arrows, select-all)
        if event.keysym in (
            "BackSpace", "Delete", "Left", "Right",
            "Home", "End", "Tab", "Return",
        ):
            return
        # Allow Ctrl+A/C/V/X shortcuts
        if event.state & 0x4:  # Control key held
            return

        inner = entry._entry  # type: ignore[attr-defined]
        after_text = inner.get()

        if allow_auto and after_text.lower() in (
            "a", "au", "aut", "auto",
        ):
            return

        if mode == "int":
            if not after_text.lstrip("-").isdigit() and after_text != "":
                # Reject — restore previous value
                inner.delete(0, "end")
                inner.insert(0, after_text[:-1] if len(after_text) > 1
                             else "")
        else:
            # Float mode: digits, dot, e/E, minus
            stripped = after_text
            if not stripped:
                return
            try:
                # Allow partial typing like "0.", "1e", "1e-"
                if stripped in (".", "-", "e", "E") or stripped.endswith(
                    ("e", "E", "e-", "E-", ".")
                ):
                    return
                float(stripped)
            except ValueError:
                inner.delete(0, "end")
                inner.insert(0, after_text[:-1] if len(after_text) > 1
                             else "")

    entry.bind("<KeyRelease>", _validate_key)
    return entry


def themed_dropdown(parent, values, width=180, height=32, **kw):
    """CTkOptionMenu with standard dark theme applied."""
    kw.setdefault("font", FONT_SMALL)
    kw.setdefault("fg_color", C_INPUT)
    kw.setdefault("button_color", C_ACCENT_DIM)
    kw.setdefault("button_hover_color", C_ACCENT_MUTED)
    return ctk.CTkOptionMenu(
        parent, values=values, width=width, height=height, **kw)


def themed_scroll(parent, **kw):
    """CTkScrollableFrame with standard dark theme applied.

    Sets ``yscrollincrement`` on the underlying canvas so that
    scroll speed scales naturally with the OS mouse-wheel setting
    (no custom multiplier needed).
    """
    kw.setdefault("fg_color", C_PANEL)
    kw.setdefault("corner_radius", 2)
    kw.setdefault("scrollbar_button_color", C_ACCENT_DIM)
    kw.setdefault("scrollbar_button_hover_color", C_ACCENT_MUTED)
    frame = ctk.CTkScrollableFrame(parent, **kw)

    # CTk's _mouse_wheel_all does `scroll(-int(delta/6), "units")`.
    # On Windows delta=120 per notch → 20 units.  Without
    # yscrollincrement the canvas treats 1 unit = 1/10 of the view,
    # which is unpredictable.  Setting yscrollincrement=5 makes each
    # unit = 5 px, so a single notch scrolls 100 px — and the OS
    # delta naturally scales with the user's mouse speed setting.
    frame._parent_canvas.configure(yscrollincrement=5)

    return frame


# -------------------------------------------------------------------
# Button styles — single source of truth for all button colours
# -------------------------------------------------------------------

def _build_button_styles() -> dict[str, dict[str, str]]:
    """Build the style dict from current C_* globals.

    Called at module load time and after reload_theme().
    """
    return {
        "primary": {
            "fg_color": C_GREEN_DIM,
            "hover_color": C_GREEN_HOVER,
            "text_color": C_GREEN,
        },
        "danger": {
            "fg_color": C_RED_DIM,
            "hover_color": C_RED_HOVER,
            "text_color": C_RED,
        },
        "action": {
            "fg_color": C_ACCENT_DIM,
            "hover_color": C_ACCENT_MUTED,
            "text_color": C_ACCENT,
        },
        "tool": {
            "fg_color": C_CYAN_DIM,
            "hover_color": C_CYAN_HOVER,
            "text_color": C_CYAN,
        },
        "secondary": {
            "fg_color": C_SURFACE,
            "hover_color": C_BORDER,
            "text_color": C_TEXT_DIM,
        },
        "warning": {
            "fg_color": C_ORANGE_DIM,
            "hover_color": C_ORANGE_HOVER,
            "text_color": C_ORANGE,
        },
        "icon": {
            "fg_color": "transparent",
            "hover_color": C_SURFACE,
            "text_color": C_TEXT_DIM,
        },
    }


BUTTON_STYLES: dict[str, dict[str, str]] = _build_button_styles()


def themed_button(parent, text: str, *, style: str = "action",
                  width: int = 100, height: int = 30, **kw):
    """CTkButton with consistent theme-aware colours.

    Styles: primary, danger, action, tool, secondary, warning, icon.
    Caller can override any kwarg (fg_color, hover_color, etc.).
    """
    colors = BUTTON_STYLES.get(style, BUTTON_STYLES["action"])
    kw.setdefault("fg_color", colors["fg_color"])
    kw.setdefault("hover_color", colors["hover_color"])
    kw.setdefault("text_color", colors["text_color"])
    kw.setdefault("font", FONT_SMALL)
    kw.setdefault("corner_radius", 2)
    return ctk.CTkButton(parent, text=text, width=width,
                         height=height, **kw)


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
# SelectableLabel - label with click-drag text selection
# -------------------------------------------------------------------

def _resolve_parent_bg(widget, fg_color):
    """Walk up the parent chain to find the actual background color.

    For widgets with fg_color='transparent', this finds the first
    ancestor with a real background color. Falls back to C_BG.
    """
    if fg_color and fg_color != "transparent":
        if isinstance(fg_color, (list, tuple)):
            return fg_color[1]  # dark mode value
        return fg_color
    parent = widget
    depth = 0
    while parent and depth < 50:
        depth += 1
        try:
            fg = parent.cget("fg_color")
            if fg and fg != "transparent":
                if isinstance(fg, (list, tuple)):
                    return fg[1]
                return fg
        except Exception:
            try:
                bg = parent.cget("bg")
                if bg:
                    return bg
            except Exception:
                pass
        parent = getattr(parent, "master", None)
    return C_BG


class SelectableLabel(ctk.CTkFrame):
    """Label that supports click-drag text selection and copy.

    Uses a tkinter Entry widget in readonly state so users can
    click-drag to select text, Ctrl+C to copy, and right-click
    for a copy menu. No blinking cursor (insertwidth=0).

    Drop-in replacement for CTkLabel in display/read-only contexts.
    Does NOT support wraplength (use CTkLabel for wrapped text).
    """

    def __init__(self, master, text="", font=FONT_BODY,
                 text_color=C_TEXT, fg_color="transparent",
                 anchor="w", width=0, height=0, **kwargs):
        # Build frame container
        frame_kw: dict = {
            "fg_color": fg_color, "corner_radius": 0}
        if height:
            frame_kw["height"] = height
        if width:
            frame_kw["width"] = width
        # When width is pinned but height is not, compute height from
        # the font to avoid CTkFrame's 200px default leaking through
        # when pack/grid propagation is disabled below.
        if width and not height:
            try:
                import tkinter.font as tkfont
                f = tkfont.Font(font=font)
                frame_kw["height"] = f.metrics("linespace") + 6
            except Exception:
                frame_kw["height"] = 26
        # Pop CTkLabel-specific kwargs that don't apply to CTkFrame
        kwargs.pop("wraplength", None)
        kwargs.pop("justify", None)
        kwargs.pop("image", None)
        kwargs.pop("compound", None)
        kwargs.pop("cursor", None)
        super().__init__(master, **frame_kw, **kwargs)
        # When explicit dimensions are provided, keep this widget's
        # requested size stable even as text changes.
        self._fixed_width = bool(width)
        self._fixed_height = bool(height)
        if self._fixed_width or self._fixed_height:
            self.pack_propagate(False)
            self.grid_propagate(False)

        self._text_val = str(text)
        self._text_color = text_color
        self._font_val = font

        # Map anchor to Entry justify
        justify_map = {"w": "left", "center": "center", "e": "right"}
        justify = justify_map.get(anchor, "left")

        # Resolve actual background color for seamless look
        bg = _resolve_parent_bg(master, fg_color)

        # Readonly Entry — allows selection, blocks editing
        self._var = tk.StringVar(value=self._text_val)
        self._entry = tk.Entry(
            self, textvariable=self._var,
            font=font, fg=text_color,
            readonlybackground=bg,
            borderwidth=0, highlightthickness=0,
            relief="flat", insertwidth=0,
            state="readonly",
            selectbackground=C_ACCENT_DIM,
            selectforeground=C_TEXT_BRIGHT,
            justify=justify,
            cursor="arrow",
            width=max(len(self._text_val), 1))
        self._entry.pack(fill="both", expand=True)

        # Right-click copy menu
        self._entry.bind("<Button-3>", self._show_copy_menu)

    def configure(self, **kwargs):
        """Support text, text_color, and font updates."""
        text = kwargs.pop("text", None)
        text_color = kwargs.pop("text_color", None)
        font = kwargs.pop("font", None)
        # Ignore CTkLabel-only kwargs silently
        kwargs.pop("wraplength", None)
        kwargs.pop("justify", None)
        kwargs.pop("image", None)
        kwargs.pop("compound", None)
        kwargs.pop("cursor", None)

        if text is not None:
            self._text_val = str(text)
            self._var.set(self._text_val)
            # Keep requested size stable for fixed-size labels.
            if not self._fixed_width:
                self._entry.configure(
                    width=max(len(self._text_val), 1))
        if text_color is not None:
            self._text_color = text_color
            self._entry.configure(fg=text_color)
        if font is not None:
            self._font_val = font
            self._entry.configure(font=font)

        # Pass remaining to CTkFrame
        if kwargs:
            super().configure(**kwargs)

    def cget(self, attribute):
        """Read widget attributes."""
        if attribute == "text":
            return self._text_val
        if attribute == "text_color":
            return self._text_color
        if attribute == "font":
            return self._font_val
        return super().cget(attribute)

    def _show_copy_menu(self, event):
        """Right-click context menu with Copy and Select All."""
        menu = tk.Menu(self, tearoff=0)
        try:
            sel = self._entry.selection_get()
            if sel:
                menu.add_command(
                    label="Copy",
                    command=lambda: self._copy(sel))
        except Exception:
            pass
        text = self._text_val
        if text:
            menu.add_command(
                label="Copy All",
                command=lambda: self._copy(text))
            menu.add_command(
                label="Select All",
                command=self._select_all)
        menu.tk_popup(event.x_root, event.y_root)
        menu.bind("<Unmap>", lambda _e: menu.destroy())

    def _copy(self, text):
        """Copy text to system clipboard."""
        self.clipboard_clear()
        self.clipboard_append(text)

    def _select_all(self):
        """Select all text in the entry."""
        self._entry.selection_range(0, "end")


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

        SelectableLabel(
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

    def set_state(self, on: bool):
        """Programmatically set toggle state without triggering callback."""
        self._on = on
        self._apply_style()

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

        self._title_label = SelectableLabel(
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
        # Hide the blinking insertion cursor — read-only, not editable
        self._textbox.configure(insertwidth=0)
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

        self._left = SelectableLabel(
            self, text="", font=FONT_TINY, text_color=C_TEXT_DIM)
        self._left.pack(side="left", padx=12)

        self._right = SelectableLabel(
            self, text="", font=FONT_TINY, text_color=C_TEXT_DIM)
        self._right.pack(side="right", padx=12)

        self._center = SelectableLabel(
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

    Stays visible as long as the mouse is over the widget or the tooltip
    itself.  A short grace period (150 ms) lets the mouse travel between
    the widget and the tooltip without flickering.

    Usage:
        Tooltip(my_button, "Send your message")
    """

    _active: "Tooltip | None" = None  # class-level: only one tooltip at a time
    _focus_bound_roots: set = set()  # root windows with FocusOut binding

    def __init__(self, widget, text: str, delay: int = 1000):
        self._widget = widget
        self._text = text
        self._delay = delay
        self._tip_window = None
        self._after_id = None
        self._hide_timer_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._on_widget_leave, add="+")
        widget.bind("<ButtonPress>", self._dismiss, add="+")
        widget.bind("<Destroy>", self._on_destroy, add="+")
        # Dismiss tooltips when app loses focus (once per root window)
        root = widget.winfo_toplevel()
        root_id = id(root)
        if root_id not in Tooltip._focus_bound_roots:
            root.bind("<FocusOut>", Tooltip._on_app_focus_out, add="+")
            Tooltip._focus_bound_roots.add(root_id)

    @staticmethod
    def _on_app_focus_out(event=None):
        """Dismiss active tooltip when app loses focus."""
        # Only react when the root window itself loses focus
        if event and event.widget is event.widget.winfo_toplevel():
            if Tooltip._active is not None:
                Tooltip._active._hide()

    def _on_destroy(self, event=None):
        """Clean up when widget is destroyed."""
        if self._after_id:
            try:
                self._widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self._cancel_hide_timer()
        self._hide()

    def _schedule(self, event=None):
        self._cancel_hide_timer()
        if self._tip_window:
            return  # already showing
        if self._after_id:
            self._widget.after_cancel(self._after_id)
        self._after_id = self._widget.after(
            self._delay, self._show)

    def _on_widget_leave(self, event=None):
        # On Leave events, ignore child-boundary transitions:
        # tkinter fires <Leave> on the parent when the cursor enters
        # a child widget.  Check if the pointer is still within this
        # widget's bounds before actually cancelling.
        if event:
            try:
                ex = str(event.type)
                if ex in ('8', 'Leave'):
                    mx = self._widget.winfo_pointerx()
                    my = self._widget.winfo_pointery()
                    wx = self._widget.winfo_rootx()
                    wy = self._widget.winfo_rooty()
                    if (wx <= mx < wx + self._widget.winfo_width()
                            and wy <= my < wy + self._widget.winfo_height()):
                        return  # pointer still inside — child transition
            except Exception:
                pass
        if self._after_id:
            self._widget.after_cancel(self._after_id)
            self._after_id = None
        # Don't hide immediately — allow mouse to travel to the tooltip
        if self._tip_window:
            self._schedule_hide_timer()

    def _dismiss(self, event=None):
        """Immediate hide on click."""
        if self._after_id:
            self._widget.after_cancel(self._after_id)
            self._after_id = None
        self._cancel_hide_timer()
        self._hide()

    def _cancel_hide_timer(self):
        if self._hide_timer_id:
            try:
                self._widget.after_cancel(self._hide_timer_id)
            except Exception:
                pass
            self._hide_timer_id = None

    def _schedule_hide_timer(self):
        self._cancel_hide_timer()
        self._hide_timer_id = self._widget.after(150, self._hide)

    def _on_tip_enter(self, event=None):
        self._cancel_hide_timer()

    def _on_tip_leave(self, event=None):
        self._schedule_hide_timer()

    def _show(self):
        if self._tip_window:
            return
        # Dismiss any other active tooltip first
        if Tooltip._active is not None and Tooltip._active is not self:
            Tooltip._active._hide()
        Tooltip._active = self
        import tkinter as tk
        x = self._widget.winfo_rootx() + 10
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip_window = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_transient(self._widget.winfo_toplevel())
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self._text, justify="left",
            background=C_INPUT, foreground=C_TEXT,
            relief="solid", borderwidth=1,
            font=FONT_TINY, padx=6, pady=3)
        label.pack()
        # Let mouse hover over the tooltip without it disappearing
        tw.bind("<Enter>", self._on_tip_enter)
        tw.bind("<Leave>", self._on_tip_leave)
        # Watchdog: auto-dismiss if mouse drifts away and Leave was missed
        self._start_watchdog()

    def _hide(self):
        self._cancel_hide_timer()
        self._cancel_watchdog()
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None
        if Tooltip._active is self:
            Tooltip._active = None

    # -- Watchdog: catches stuck tooltips when Leave events are missed --

    _WATCHDOG_INTERVAL = 1000  # ms between checks

    def _start_watchdog(self):
        """Schedule periodic check that tooltip is still warranted."""
        self._watchdog_id = self._widget.after(
            self._WATCHDOG_INTERVAL, self._watchdog_check)

    def _cancel_watchdog(self):
        wid = getattr(self, "_watchdog_id", None)
        if wid:
            try:
                self._widget.after_cancel(wid)
            except Exception:
                pass
            self._watchdog_id = None

    def _watchdog_check(self):
        """Auto-dismiss if mouse is not over the widget or tooltip."""
        self._watchdog_id = None
        if not self._tip_window:
            return
        try:
            mx = self._widget.winfo_pointerx()
            my = self._widget.winfo_pointery()
            # Check widget bounds
            wx = self._widget.winfo_rootx()
            wy = self._widget.winfo_rooty()
            in_widget = (wx <= mx < wx + self._widget.winfo_width()
                         and wy <= my < wy + self._widget.winfo_height())
            # Check tooltip bounds
            in_tip = False
            tw = self._tip_window
            if tw and tw.winfo_exists():
                tx = tw.winfo_rootx()
                ty = tw.winfo_rooty()
                in_tip = (tx <= mx < tx + tw.winfo_width()
                          and ty <= my < ty + tw.winfo_height())
            if not in_widget and not in_tip:
                self._hide()
                return
        except Exception:
            # Widget destroyed or other error — clean up
            self._hide()
            return
        # Still valid — reschedule
        self._watchdog_id = self._widget.after(
            self._WATCHDOG_INTERVAL, self._watchdog_check)
