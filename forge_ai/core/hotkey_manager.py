"""
================================================================================
HOTKEY MANAGER - Global Hotkey System
================================================================================

Global hotkey registration and handling system that works across all platforms,
even when ForgeAI is not the focused window.

FILE: forge_ai/core/hotkey_manager.py
TYPE: Core Utility
MAIN CLASS: HotkeyManager

FEATURES:
    - Global hotkey registration (works in fullscreen games)
    - Platform-specific backends (Windows, Linux, macOS)
    - Conflict detection
    - Easy rebinding
    - Default hotkey presets

USAGE:
    from forge_ai.core.hotkey_manager import HotkeyManager, DEFAULT_HOTKEYS
    
    manager = HotkeyManager()
    manager.register(
        hotkey=DEFAULT_HOTKEYS["summon_ai"],
        callback=show_overlay,
        name="summon_ai"
    )
    
    manager.start()
    # ... hotkeys are now active ...
    manager.stop()
"""

import sys
import logging
import threading
from typing import Callable, List, Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default hotkey bindings
DEFAULT_HOTKEYS = {
    "summon_ai": "Ctrl+Shift+Space",        # Open AI overlay
    "dismiss_ai": "Escape",                  # Close AI overlay (when overlay has focus)
    "push_to_talk": "Ctrl+Shift+T",          # Hold to speak
    "toggle_game_mode": "Ctrl+Shift+G",      # Toggle game mode
    "quick_command": "Ctrl+Shift+C",         # Quick command input
    "screenshot_to_ai": "Ctrl+Shift+S",      # Screenshot and ask AI about it
}


@dataclass
class HotkeyInfo:
    """Information about a registered hotkey."""
    name: str
    hotkey: str
    callback: Callable
    enabled: bool = True


class HotkeyManager:
    """
    Global hotkey registration and handling.
    
    Works even when:
    - Game is fullscreen
    - Another app is focused
    - Multiple monitors
    """
    
    def __init__(self):
        """Initialize the hotkey manager."""
        self._hotkeys: Dict[str, HotkeyInfo] = {}
        self._backend: Optional[Any] = None
        self._running = False
        self._lock = threading.Lock()
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the platform-specific backend."""
        try:
            if sys.platform == 'win32':
                from .hotkeys.windows import WindowsHotkeyBackend
                self._backend = WindowsHotkeyBackend()
            elif sys.platform == 'darwin':
                from .hotkeys.macos import MacOSHotkeyBackend
                self._backend = MacOSHotkeyBackend()
            else:  # Linux and other Unix-like systems
                from .hotkeys.linux import LinuxHotkeyBackend
                self._backend = LinuxHotkeyBackend()
            
            logger.info(f"Initialized hotkey backend: {self._backend.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize hotkey backend: {e}")
            self._backend = None
    
    def register(self, hotkey: str, callback: Callable, name: str) -> bool:
        """
        Register a global hotkey.
        
        Args:
            hotkey: Key combination ("Ctrl+Shift+Space", "F12", etc.)
            callback: Function to call when pressed
            name: Human-readable name for this hotkey
            
        Returns:
            True if registration succeeded, False otherwise
        """
        if not self._backend:
            logger.warning("No hotkey backend available")
            return False
        
        with self._lock:
            # Check if name already registered
            if name in self._hotkeys:
                logger.warning(f"Hotkey '{name}' already registered, unregistering first")
                self.unregister(name)
            
            try:
                # Register with backend
                success = self._backend.register(hotkey, callback, name)
                
                if success:
                    # Store hotkey info
                    self._hotkeys[name] = HotkeyInfo(
                        name=name,
                        hotkey=hotkey,
                        callback=callback,
                        enabled=True
                    )
                    logger.info(f"Registered hotkey '{name}': {hotkey}")
                    return True
                else:
                    logger.warning(f"Failed to register hotkey '{name}': {hotkey}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error registering hotkey '{name}': {e}")
                return False
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a hotkey by name.
        
        Args:
            name: Name of the hotkey to unregister
            
        Returns:
            True if unregistration succeeded, False otherwise
        """
        if not self._backend:
            return False
        
        with self._lock:
            if name not in self._hotkeys:
                logger.warning(f"Hotkey '{name}' not registered")
                return False
            
            try:
                # Unregister from backend
                success = self._backend.unregister(name)
                
                if success:
                    # Remove from our records
                    del self._hotkeys[name]
                    logger.info(f"Unregistered hotkey '{name}'")
                    return True
                else:
                    logger.warning(f"Failed to unregister hotkey '{name}'")
                    return False
                    
            except Exception as e:
                logger.error(f"Error unregistering hotkey '{name}': {e}")
                return False
    
    def unregister_all(self):
        """Unregister all hotkeys."""
        if not self._backend:
            return
        
        with self._lock:
            names = list(self._hotkeys.keys())
            for name in names:
                self.unregister(name)
    
    def list_registered(self) -> List[Dict[str, Any]]:
        """
        List all registered hotkeys.
        
        Returns:
            List of dictionaries containing hotkey information
        """
        with self._lock:
            return [
                {
                    "name": info.name,
                    "hotkey": info.hotkey,
                    "enabled": info.enabled,
                }
                for info in self._hotkeys.values()
            ]
    
    def is_available(self, hotkey: str) -> bool:
        """
        Check if hotkey is available (not used by system/other apps).
        
        Args:
            hotkey: Key combination to check
            
        Returns:
            True if available, False if potentially in use
        """
        if not self._backend:
            return False
        
        try:
            return self._backend.is_available(hotkey)
        except Exception as e:
            logger.error(f"Error checking hotkey availability: {e}")
            return False
    
    def start(self):
        """Start listening for hotkeys."""
        if not self._backend:
            logger.warning("No hotkey backend available")
            return
        
        if self._running:
            logger.warning("Hotkey manager already running")
            return
        
        try:
            self._backend.start()
            self._running = True
            logger.info("Hotkey manager started")
        except Exception as e:
            logger.error(f"Error starting hotkey manager: {e}")
    
    def stop(self):
        """Stop listening for hotkeys."""
        if not self._backend:
            return
        
        if not self._running:
            return
        
        try:
            self._backend.stop()
            self._running = False
            logger.info("Hotkey manager stopped")
        except Exception as e:
            logger.error(f"Error stopping hotkey manager: {e}")
    
    def is_running(self) -> bool:
        """Check if the hotkey manager is running."""
        return self._running
    
    def __del__(self):
        """Cleanup when manager is destroyed."""
        try:
            self.stop()
            self.unregister_all()
        except Exception:
            pass  # Ignore cleanup errors during shutdown


# Singleton instance
_manager: Optional[HotkeyManager] = None


def get_hotkey_manager() -> HotkeyManager:
    """
    Get the global hotkey manager instance.
    
    Returns:
        The singleton HotkeyManager instance
    """
    global _manager
    if _manager is None:
        _manager = HotkeyManager()
    return _manager
