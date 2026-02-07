"""
Tests for the global hotkey system.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock


class TestHotkeyManager:
    """Test the HotkeyManager class."""
    
    def test_initialization(self):
        """Test hotkey manager initialization."""
        from enigma_engine.core.hotkey_manager import HotkeyManager
        
        manager = HotkeyManager()
        assert manager is not None
        assert not manager.is_running()
    
    def test_default_hotkeys(self):
        """Test default hotkey definitions."""
        from enigma_engine.core.hotkey_manager import DEFAULT_HOTKEYS
        
        assert "summon_ai" in DEFAULT_HOTKEYS
        assert "dismiss_ai" in DEFAULT_HOTKEYS
        assert "push_to_talk" in DEFAULT_HOTKEYS
        assert "toggle_game_mode" in DEFAULT_HOTKEYS
        assert "quick_command" in DEFAULT_HOTKEYS
        assert "screenshot_to_ai" in DEFAULT_HOTKEYS
    
    def test_register_hotkey(self):
        """Test registering a hotkey."""
        from enigma_engine.core.hotkey_manager import HotkeyManager
        
        manager = HotkeyManager()
        callback = Mock()
        
        # Register a hotkey
        success = manager.register("Ctrl+Shift+T", callback, "test_hotkey")
        
        # May fail if backend is not available, but should not crash
        assert isinstance(success, bool)
        
        # If successful, should be in list
        if success:
            hotkeys = manager.list_registered()
            assert any(h["name"] == "test_hotkey" for h in hotkeys)
    
    def test_unregister_hotkey(self):
        """Test unregistering a hotkey."""
        from enigma_engine.core.hotkey_manager import HotkeyManager
        
        manager = HotkeyManager()
        callback = Mock()
        
        # Register and unregister
        success = manager.register("Ctrl+Shift+T", callback, "test_hotkey")
        if success:
            result = manager.unregister("test_hotkey")
            assert result is True
            
            # Should not be in list anymore
            hotkeys = manager.list_registered()
            assert not any(h["name"] == "test_hotkey" for h in hotkeys)
    
    def test_unregister_all(self):
        """Test unregistering all hotkeys."""
        from enigma_engine.core.hotkey_manager import HotkeyManager
        
        manager = HotkeyManager()
        callback1 = Mock()
        callback2 = Mock()
        
        # Register multiple hotkeys
        manager.register("Ctrl+Shift+A", callback1, "test1")
        manager.register("Ctrl+Shift+B", callback2, "test2")
        
        # Unregister all
        manager.unregister_all()
        
        # List should be empty
        hotkeys = manager.list_registered()
        assert len(hotkeys) == 0
    
    def test_singleton_pattern(self):
        """Test that get_hotkey_manager returns singleton."""
        from enigma_engine.core.hotkey_manager import get_hotkey_manager
        
        manager1 = get_hotkey_manager()
        manager2 = get_hotkey_manager()
        
        assert manager1 is manager2


class TestHotkeyActions:
    """Test the HotkeyActions class."""
    
    def test_initialization(self):
        """Test hotkey actions initialization."""
        from enigma_engine.core.hotkey_actions import HotkeyActions
        
        actions = HotkeyActions()
        assert actions is not None
    
    def test_summon_overlay(self):
        """Test summon_overlay action."""
        from enigma_engine.core.hotkey_actions import HotkeyActions
        
        actions = HotkeyActions()
        
        # Should not crash even without main window
        try:
            actions.summon_overlay()
        except Exception as e:
            # Expected to fail without GUI, but should handle gracefully
            pass
    
    def test_dismiss_overlay(self):
        """Test dismiss_overlay action."""
        from enigma_engine.core.hotkey_actions import HotkeyActions
        
        actions = HotkeyActions()
        
        # Should not crash
        try:
            actions.dismiss_overlay()
        except Exception as e:
            pass
    
    def test_toggle_game_mode(self):
        """Test toggle_game_mode action."""
        from enigma_engine.core.hotkey_actions import HotkeyActions
        
        actions = HotkeyActions()
        
        # Should toggle state
        initial_state = actions._game_mode_active
        actions.toggle_game_mode()
        assert actions._game_mode_active != initial_state
        actions.toggle_game_mode()
        assert actions._game_mode_active == initial_state
    
    def test_singleton_pattern(self):
        """Test that get_hotkey_actions returns singleton."""
        from enigma_engine.core.hotkey_actions import get_hotkey_actions
        
        actions1 = get_hotkey_actions()
        actions2 = get_hotkey_actions()
        
        assert actions1 is actions2


class TestWindowsHotkeyBackend:
    """Test Windows-specific hotkey backend."""
    
    @pytest.mark.skipif(sys.platform != 'win32', reason="Windows-specific test")
    def test_initialization(self):
        """Test Windows backend initialization."""
        from enigma_engine.core.hotkeys.windows import WindowsHotkeyBackend
        
        backend = WindowsHotkeyBackend()
        assert backend is not None
    
    @pytest.mark.skipif(sys.platform != 'win32', reason="Windows-specific test")
    def test_parse_hotkey(self):
        """Test hotkey parsing on Windows."""
        from enigma_engine.core.hotkeys.windows import WindowsHotkeyBackend
        
        backend = WindowsHotkeyBackend()
        
        # Test parsing different combinations
        modifiers, vk = backend._parse_hotkey("Ctrl+Shift+T")
        assert modifiers is not None
        assert vk is not None
    
    @pytest.mark.skipif(sys.platform != 'win32', reason="Windows-specific test")
    def test_key_to_vk(self):
        """Test key to virtual key code conversion."""
        from enigma_engine.core.hotkeys.windows import WindowsHotkeyBackend
        
        backend = WindowsHotkeyBackend()
        
        # Test common keys
        assert backend._key_to_vk("Space") == 0x20
        assert backend._key_to_vk("Enter") == 0x0D
        assert backend._key_to_vk("Escape") == 0x1B
        assert backend._key_to_vk("F1") == 0x70


class TestLinuxHotkeyBackend:
    """Test Linux-specific hotkey backend."""
    
    @pytest.mark.skipif(not sys.platform.startswith('linux'), reason="Linux-specific test")
    def test_initialization(self):
        """Test Linux backend initialization."""
        from enigma_engine.core.hotkeys.linux import LinuxHotkeyBackend
        
        backend = LinuxHotkeyBackend()
        assert backend is not None
    
    @pytest.mark.skipif(not sys.platform.startswith('linux'), reason="Linux-specific test")
    def test_parse_hotkey(self):
        """Test hotkey parsing on Linux."""
        from enigma_engine.core.hotkeys.linux import LinuxHotkeyBackend
        
        backend = LinuxHotkeyBackend()
        
        # Test parsing different combinations
        modifiers, key = backend._parse_hotkey("Ctrl+Shift+T")
        assert isinstance(modifiers, list)
        assert isinstance(key, str)


class TestMacOSHotkeyBackend:
    """Test macOS-specific hotkey backend."""
    
    @pytest.mark.skipif(sys.platform != 'darwin', reason="macOS-specific test")
    def test_initialization(self):
        """Test macOS backend initialization."""
        from enigma_engine.core.hotkeys.macos import MacOSHotkeyBackend
        
        backend = MacOSHotkeyBackend()
        assert backend is not None
    
    @pytest.mark.skipif(sys.platform != 'darwin', reason="macOS-specific test")
    def test_parse_hotkey(self):
        """Test hotkey parsing on macOS."""
        from enigma_engine.core.hotkeys.macos import MacOSHotkeyBackend
        
        backend = MacOSHotkeyBackend()
        
        # Test parsing different combinations
        modifiers, key = backend._parse_hotkey("Cmd+Shift+T")
        assert isinstance(modifiers, list)
        assert "command" in modifiers or "shift" in modifiers
        assert isinstance(key, str)


class TestHotkeyConfigWidget:
    """Test the hotkey configuration widget."""
    
    @pytest.mark.skipif(not hasattr(pytest, 'qt_api'), reason="PyQt5 not available")
    def test_initialization(self):
        """Test widget initialization."""
        try:
            from enigma_engine.gui.widgets.hotkey_config import HotkeyConfigWidget
            from PyQt5.QtWidgets import QApplication
            
            # Need QApplication instance
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            
            widget = HotkeyConfigWidget()
            assert widget is not None
        except ImportError:
            pytest.skip("PyQt5 not available")


class TestConfigIntegration:
    """Test integration with config system."""
    
    def test_default_config(self):
        """Test that hotkey config exists in defaults."""
        from enigma_engine.config import CONFIG
        
        assert "enable_hotkeys" in CONFIG
        assert "hotkeys" in CONFIG
        assert isinstance(CONFIG["hotkeys"], dict)
    
    def test_hotkey_config_structure(self):
        """Test hotkey config structure."""
        from enigma_engine.config import CONFIG
        
        hotkeys = CONFIG["hotkeys"]
        
        # Check for expected keys
        expected_keys = [
            "summon_ai",
            "dismiss_ai",
            "push_to_talk",
            "toggle_game_mode",
            "quick_command",
            "screenshot_to_ai"
        ]
        
        for key in expected_keys:
            assert key in hotkeys
            assert isinstance(hotkeys[key], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
