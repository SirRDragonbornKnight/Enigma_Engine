"""
Enigma Engine — GUI Theme System
====================================

Defines colour themes for the desktop GUI.  Each theme is a ``Theme``
dataclass mapping logical colour roles to hex values.

How it works:
    1. ``widgets.py`` calls ``load_active_theme()`` at module-load time.
    2. The returned ``Theme`` populates the ``C_*`` constants other modules
       import.
    3. The user's preference is read from ``data/gui_settings.json["theme"]``.
    4. Changing the theme requires a restart (constants are bound once).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to gui_settings.json (mirrors DATA_DIR in scanners.py)
_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "gui_settings.json"


# =========================================================================
# Theme dataclass
# =========================================================================

@dataclass(frozen=True)
class Theme:
    """A complete colour theme for the Enigma GUI."""
    name: str

    # Backgrounds
    bg: str              # deepest background
    panel: str           # side-panel / nav-rail background
    surface: str         # card / elevated surface
    input: str           # text-entry background

    # Accent
    accent: str          # primary accent (silver-gray by default)
    accent_dim: str      # muted version of accent
    accent_muted: str    # hover / subtle accent

    # Semantic colours
    purple: str          # user messages
    purple_dim: str
    purple_muted: str
    cyan: str            # CMD highlights
    green: str           # success / running
    green_dim: str
    green_hover: str     # green button hover
    red: str             # error / stopped
    red_dim: str         # destructive button background
    red_hover: str       # red button hover
    orange: str          # warnings / system messages
    orange_dim: str      # warning button background
    orange_hover: str    # warning button hover
    cyan_dim: str        # tool button background
    cyan_hover: str      # tool button hover

    # Text
    text: str            # normal text
    text_dim: str        # secondary / placeholder
    text_bright: str     # headings / emphasis

    # Borders
    border: str          # default border
    border_accent: str   # accented border

    def to_dict(self) -> dict:
        """Serialise to dict (excludes ``name``)."""
        d = asdict(self)
        d.pop("name", None)
        return d


# =========================================================================
# Preset themes
# =========================================================================

THEMES: dict[str, Theme] = {
    "dark": Theme(
        name="dark",
        bg="#080808", panel="#0e0e0e", surface="#181818", input="#1c1c1c",
        accent="#8B95A5", accent_dim="#2a2a2a", accent_muted="#3d3d3d",
        purple="#a855f7", purple_dim="#2a1a3e", purple_muted="#3d2a55",
        cyan="#22d3ee",
        green="#22c55e", green_dim="#0e3a1e", green_hover="#1a5a2a",
        red="#ef4444", red_dim="#3b1111", red_hover="#5a1a1a",
        orange="#f97316", orange_dim="#3a2a11", orange_hover="#5a3a1a",
        cyan_dim="#0e2a3a", cyan_hover="#1a3a4e",
        text="#b0b0b0", text_dim="#555555", text_bright="#e8e8e8",
        border="#1f1f1f", border_accent="#2e2e2e",
    ),
    "midnight": Theme(
        name="midnight",
        bg="#0a0e1a", panel="#0f1428", surface="#161c32", input="#1a2038",
        accent="#7b8fbb", accent_dim="#1e2744", accent_muted="#2e3a5a",
        purple="#9b6dff", purple_dim="#261640", purple_muted="#3a2460",
        cyan="#38bdf8",
        green="#34d399", green_dim="#0c2e1c", green_hover="#165a30",
        red="#f87171", red_dim="#3a1118", red_hover="#5a1a22",
        orange="#fb923c", orange_dim="#3a2a14", orange_hover="#5a3a1e",
        cyan_dim="#0e2438", cyan_hover="#1a3448",
        text="#b4bed0", text_dim="#4a5578", text_bright="#e2e8f0",
        border="#1a2340", border_accent="#283654",
    ),
    "carbon": Theme(
        name="carbon",
        bg="#121212", panel="#1a1a1a", surface="#222222", input="#282828",
        accent="#a0a0a0", accent_dim="#333333", accent_muted="#444444",
        purple="#bb86fc", purple_dim="#2d1b4e", purple_muted="#3e2a62",
        cyan="#03dac6",
        green="#4caf50", green_dim="#1b3a1d", green_hover="#245a26",
        red="#cf6679", red_dim="#3a1420", red_hover="#5a1e2e",
        orange="#ffab40", orange_dim="#3a2e14", orange_hover="#5a4020",
        cyan_dim="#0e2e2a", cyan_hover="#1a3e38",
        text="#c0c0c0", text_dim="#666666", text_bright="#f0f0f0",
        border="#2a2a2a", border_accent="#3a3a3a",
    ),
    "solarized": Theme(
        name="solarized",
        bg="#002b36", panel="#073642", surface="#0a3f4c", input="#0d4a58",
        accent="#839496", accent_dim="#094050", accent_muted="#0e5263",
        purple="#d33682", purple_dim="#2a0e1e", purple_muted="#3d1530",
        cyan="#2aa198",
        green="#859900", green_dim="#1a2000", green_hover="#2a3a10",
        red="#dc322f", red_dim="#2a0e0e", red_hover="#3e1616",
        orange="#cb4b16", orange_dim="#2a1e0a", orange_hover="#3e2e14",
        cyan_dim="#0a2e2c", cyan_hover="#143e3a",
        text="#93a1a1", text_dim="#586e75", text_bright="#eee8d5",
        border="#094050", border_accent="#0e5263",
    ),
}

DEFAULT_THEME = "dark"


# =========================================================================
# Public API
# =========================================================================

def get_theme_names() -> list[str]:
    """Return available theme names."""
    return list(THEMES.keys())


def get_theme(name: str) -> Theme:
    """Return theme by name, falling back to ``dark`` if unknown."""
    return THEMES.get(name, THEMES[DEFAULT_THEME])


def load_active_theme() -> Theme:
    """Read the user's theme preference from gui_settings.json.

    Returns the matching ``Theme`` (or ``dark`` if not set / invalid).
    """
    theme_name = DEFAULT_THEME
    try:
        if _SETTINGS_PATH.exists():
            data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
            theme_name = data.get("theme", DEFAULT_THEME)
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Could not read theme from settings: %s", exc)
    return get_theme(theme_name)


def save_theme_preference(name: str) -> None:
    """Persist the theme name to gui_settings.json."""
    data: dict = {}
    try:
        if _SETTINGS_PATH.exists():
            data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    data["theme"] = name
    try:
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        from enigma_engine.core.safe_save import atomic_write_json
        atomic_write_json(_SETTINGS_PATH, data)
    except OSError as exc:
        logger.warning("Could not save theme preference: %s", exc)
