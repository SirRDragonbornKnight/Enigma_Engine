"""
Tests for the AI Overlay system.

Verifies:
- Overlay modes and configuration
- Theme system
- Position management
- Settings persistence
- Chat bridge functionality
"""

import unittest
import sys
import json
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import only non-Qt components for testing
# The full overlay window requires PyQt5 which may not be available in CI
from forge_ai.gui.overlay.overlay_modes import (
    OverlayMode, OverlayPosition, OverlaySettings,
    MinimalOverlay, CompactOverlay, FullOverlay
)
from forge_ai.gui.overlay.overlay_themes import (
    OverlayTheme, OVERLAY_THEMES, get_theme
)
from forge_ai.gui.overlay.overlay_compatibility import OverlayCompatibility


class TestOverlayModes(unittest.TestCase):
    """Test overlay mode configurations."""
    
    def test_overlay_mode_enum(self):
        """Test that all overlay modes are defined."""
        self.assertEqual(OverlayMode.HIDDEN.value, "hidden")
        self.assertEqual(OverlayMode.MINIMAL.value, "minimal")
        self.assertEqual(OverlayMode.COMPACT.value, "compact")
        self.assertEqual(OverlayMode.FULL.value, "full")
    
    def test_overlay_position_enum(self):
        """Test that all position presets are defined."""
        positions = [
            OverlayPosition.TOP_LEFT,
            OverlayPosition.TOP_RIGHT,
            OverlayPosition.BOTTOM_LEFT,
            OverlayPosition.BOTTOM_RIGHT,
            OverlayPosition.CENTER,
            OverlayPosition.CUSTOM,
            OverlayPosition.FOLLOW_CURSOR,
        ]
        self.assertEqual(len(positions), 7)
    
    def test_minimal_overlay_config(self):
        """Test minimal overlay configuration."""
        config = MinimalOverlay()
        self.assertTrue(config.show_avatar)
        self.assertFalse(config.show_name)
        self.assertEqual(config.max_response_lines, 1)
        self.assertEqual(config.width, 300)
        self.assertEqual(config.height, 60)
    
    def test_compact_overlay_config(self):
        """Test compact overlay configuration."""
        config = CompactOverlay()
        self.assertTrue(config.show_avatar)
        self.assertTrue(config.show_name)
        self.assertTrue(config.show_input)
        self.assertEqual(config.max_response_lines, 3)
        self.assertEqual(config.width, 350)
        self.assertEqual(config.height, 150)
    
    def test_full_overlay_config(self):
        """Test full overlay configuration."""
        config = FullOverlay()
        self.assertTrue(config.show_avatar)
        self.assertTrue(config.show_name)
        self.assertTrue(config.show_input)
        self.assertTrue(config.show_history)
        self.assertTrue(config.show_controls)
        self.assertEqual(config.width, 450)
        self.assertEqual(config.height, 400)
    
    def test_overlay_settings_defaults(self):
        """Test default overlay settings."""
        settings = OverlaySettings()
        self.assertEqual(settings.mode, OverlayMode.COMPACT)
        self.assertEqual(settings.position, OverlayPosition.TOP_RIGHT)
        self.assertEqual(settings.opacity, 0.9)
        self.assertFalse(settings.click_through)
        self.assertTrue(settings.always_on_top)
        self.assertEqual(settings.theme_name, "gaming")
        self.assertTrue(settings.remember_position)
        self.assertFalse(settings.show_on_startup)


class TestOverlayThemes(unittest.TestCase):
    """Test overlay theme system."""
    
    def test_theme_presets_exist(self):
        """Test that all preset themes are available."""
        expected_themes = ["dark", "light", "gaming", "minimal", "cyberpunk", "stealth"]
        for theme_name in expected_themes:
            self.assertIn(theme_name, OVERLAY_THEMES)
    
    def test_get_theme(self):
        """Test theme retrieval."""
        dark_theme = get_theme("dark")
        self.assertIsInstance(dark_theme, OverlayTheme)
        self.assertEqual(dark_theme.name, "dark")
    
    def test_get_theme_fallback(self):
        """Test theme fallback for invalid name."""
        theme = get_theme("nonexistent")
        self.assertEqual(theme.name, "dark")  # Should fallback to dark
    
    def test_theme_stylesheet_generation(self):
        """Test that themes can generate Qt stylesheets."""
        theme = OVERLAY_THEMES["gaming"]
        stylesheet = theme.to_stylesheet()
        self.assertIsInstance(stylesheet, str)
        self.assertIn("background-color", stylesheet)
        self.assertIn("color", stylesheet)
        self.assertIn(theme.accent_color, stylesheet)
    
    def test_gaming_theme_properties(self):
        """Test gaming theme has appropriate properties."""
        theme = OVERLAY_THEMES["gaming"]
        self.assertEqual(theme.text_color, "#00ff00")  # Green text
        self.assertEqual(theme.accent_color, "#00ff00")
        self.assertEqual(theme.font_family, "Consolas")
        # Gaming theme should be semi-transparent
        self.assertIn("0.5", theme.background_color)
    
    def test_light_theme_properties(self):
        """Test light theme has appropriate properties."""
        theme = OVERLAY_THEMES["light"]
        self.assertIn("255, 255, 255", theme.background_color)  # White bg
        self.assertEqual(theme.text_color, "#000000")  # Black text
    
    def test_theme_customization(self):
        """Test creating custom theme."""
        custom = OverlayTheme(
            name="custom",
            background_color="rgba(100, 100, 100, 0.8)",
            text_color="#ffffff",
            accent_color="#ff0000",
        )
        self.assertEqual(custom.name, "custom")
        self.assertEqual(custom.accent_color, "#ff0000")


class TestOverlayCompatibility(unittest.TestCase):
    """Test overlay compatibility detection."""
    
    def setUp(self):
        """Create compatibility checker."""
        self.compat = OverlayCompatibility()
    
    def test_detect_game_mode(self):
        """Test game mode detection returns valid value."""
        mode = self.compat.detect_game_mode()
        valid_modes = ["fullscreen", "borderless", "windowed", "unknown"]
        self.assertIn(mode, valid_modes)
    
    def test_adjust_for_fullscreen(self):
        """Test adjustments for fullscreen mode."""
        adjustments = self.compat.adjust_for_game("fullscreen")
        self.assertIn("window_level", adjustments)
        self.assertIn("use_transparency", adjustments)
        self.assertIn("click_through_recommended", adjustments)
        # Fullscreen should recommend click-through
        self.assertTrue(adjustments["click_through_recommended"])
    
    def test_adjust_for_windowed(self):
        """Test adjustments for windowed mode."""
        adjustments = self.compat.adjust_for_game("windowed")
        self.assertIn("window_level", adjustments)
        self.assertTrue(adjustments["use_transparency"])
    
    def test_check_overlay_support(self):
        """Test overlay support checking."""
        result = self.compat.check_game_overlay_support(None)
        self.assertIn("supported", result)
        self.assertIn("notes", result)
        self.assertIn("anti_cheat", result)
        # No game exe should be supported
        self.assertTrue(result["supported"])
        self.assertFalse(result["anti_cheat"])
    
    def test_detect_anti_cheat(self):
        """Test anti-cheat detection."""
        # Test with known problematic game
        result = self.compat.check_game_overlay_support("game_battleye.exe")
        self.assertFalse(result["supported"])
        self.assertTrue(result["anti_cheat"])
        self.assertIn("battleye", result["notes"].lower())
    
    def test_get_recommended_settings(self):
        """Test recommended settings generation."""
        settings = self.compat.get_recommended_settings()
        self.assertIn("mode", settings)
        self.assertIn("adjustments", settings)
        self.assertIn("platform", settings)


class TestOverlayConfiguration(unittest.TestCase):
    """Test overlay configuration and persistence."""
    
    def test_settings_serialization(self):
        """Test that settings can be saved to JSON."""
        settings = OverlaySettings(
            mode=OverlayMode.FULL,
            position=OverlayPosition.CENTER,
            opacity=0.8,
            theme_name="cyberpunk"
        )
        
        # Convert to dict for JSON
        settings_dict = {
            "mode": settings.mode.value,
            "position": settings.position.value,
            "opacity": settings.opacity,
            "theme_name": settings.theme_name,
        }
        
        # Should be JSON serializable
        json_str = json.dumps(settings_dict)
        self.assertIsInstance(json_str, str)
        
        # Should be able to deserialize
        loaded = json.loads(json_str)
        self.assertEqual(loaded["mode"], "full")
        self.assertEqual(loaded["position"], "center")
        self.assertEqual(loaded["opacity"], 0.8)
    
    def test_opacity_bounds(self):
        """Test that opacity is bounded correctly."""
        settings = OverlaySettings()
        
        # Valid range
        settings.opacity = 0.5
        self.assertEqual(settings.opacity, 0.5)
        
        # Should clamp to valid range in actual implementation
        # (This would be tested in the window class)
    
    def test_custom_position(self):
        """Test custom position coordinates."""
        settings = OverlaySettings(
            position=OverlayPosition.CUSTOM,
            custom_x=100,
            custom_y=200
        )
        self.assertEqual(settings.position, OverlayPosition.CUSTOM)
        self.assertEqual(settings.custom_x, 100)
        self.assertEqual(settings.custom_y, 200)


def run_tests():
    """Run all overlay tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestOverlayModes))
    suite.addTests(loader.loadTestsFromTestCase(TestOverlayThemes))
    suite.addTests(loader.loadTestsFromTestCase(TestOverlayCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestOverlayConfiguration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
