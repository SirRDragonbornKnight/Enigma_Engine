"""Tests for GUI theme system and color validation."""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestGUIThemes:
    """Verify the theme system works correctly."""

    def test_theme_dataclass_fields(self):
        """Theme should have all required colour fields."""
        from enigma_engine.gui.themes import Theme
        import dataclasses
        names = {f.name for f in dataclasses.fields(Theme)}
        required = {
            "name", "bg", "panel", "surface", "input",
            "accent", "accent_dim", "accent_muted",
            "purple", "purple_dim", "purple_muted", "cyan",
            "green", "green_dim", "red", "orange",
            "text", "text_dim", "text_bright",
            "border", "border_accent",
        }
        assert required.issubset(names)

    def test_default_theme_is_dark(self):
        """Default theme should be 'dark'."""
        from enigma_engine.gui.themes import DEFAULT_THEME
        assert DEFAULT_THEME == "dark"

    def test_at_least_four_themes(self):
        """Should have at least 4 preset themes."""
        from enigma_engine.gui.themes import THEMES
        assert len(THEMES) >= 4

    def test_get_theme_names(self):
        """get_theme_names returns list of strings."""
        from enigma_engine.gui.themes import get_theme_names
        names = get_theme_names()
        assert isinstance(names, list)
        assert "dark" in names
        assert all(isinstance(n, str) for n in names)

    def test_get_theme_valid(self):
        """get_theme returns the correct Theme object."""
        from enigma_engine.gui.themes import get_theme
        theme = get_theme("dark")
        assert theme.name == "dark"
        assert theme.bg == "#080808"

    def test_get_theme_unknown_falls_back(self):
        """Unknown theme name falls back to dark."""
        from enigma_engine.gui.themes import get_theme
        theme = get_theme("nonexistent_theme_xyz")
        assert theme.name == "dark"

    def test_load_active_theme_returns_theme(self):
        """load_active_theme returns a Theme instance."""
        from enigma_engine.gui.themes import Theme, load_active_theme
        theme = load_active_theme()
        assert isinstance(theme, Theme)

    def test_all_themes_have_hex_colors(self):
        """All colour values in all themes should be valid #hex strings."""
        import dataclasses
        from enigma_engine.gui.themes import THEMES, Theme
        colour_fields = [
            f.name for f in dataclasses.fields(Theme) if f.name != "name"]
        for theme_name, theme in THEMES.items():
            for field_name in colour_fields:
                val = getattr(theme, field_name)
                assert val.startswith("#"), (
                    f"{theme_name}.{field_name} = {val!r} is not a hex colour")
                # Check valid hex length (4 or 7 including #)
                assert len(val) in (4, 7), (
                    f"{theme_name}.{field_name} = {val!r} bad hex length")

    def test_theme_to_dict(self):
        """to_dict should return a dict without the name key."""
        from enigma_engine.gui.themes import get_theme
        d = get_theme("dark").to_dict()
        assert isinstance(d, dict)
        assert "name" not in d
        assert "bg" in d

    def test_widgets_use_theme_colors(self):
        """widgets.py C_* constants should match the active theme."""
        from enigma_engine.gui.themes import load_active_theme
        from enigma_engine.gui.widgets import C_BG, C_PANEL, C_TEXT
        theme = load_active_theme()
        assert C_BG == theme.bg
        assert C_PANEL == theme.panel
        assert C_TEXT == theme.text

    def test_save_theme_preference(self, tmp_path):
        """save_theme_preference writes to settings json."""
        import enigma_engine.gui.themes as themes_mod
        fake_settings = tmp_path / "gui_settings.json"
        original = themes_mod._SETTINGS_PATH
        try:
            themes_mod._SETTINGS_PATH = fake_settings
            themes_mod.save_theme_preference("midnight")
            import json
            data = json.loads(fake_settings.read_text(encoding="utf-8"))
            assert data["theme"] == "midnight"
        finally:
            themes_mod._SETTINGS_PATH = original

    def test_midnight_theme_values(self):
        """Midnight theme should have distinct blue-tinted colours."""
        from enigma_engine.gui.themes import get_theme
        theme = get_theme("midnight")
        assert theme.name == "midnight"
        # Midnight bg should be different from dark bg
        assert theme.bg != "#080808"


# =========================================================================
# Item 20 — Plugin API for Custom Commands
# =========================================================================

