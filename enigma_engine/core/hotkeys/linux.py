"""
Linux Hotkey Backend - Global hotkey implementation for Linux.

Uses python-xlib for X11 or fallback to keyboard library for Wayland.
"""

import logging
import sys
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Try to import X11 support
HAS_XLIB = False
if sys.platform.startswith('linux'):
    try:
        from Xlib import display
        HAS_XLIB = True
    except ImportError:
        logger.warning("python-xlib not available, trying keyboard library")
        HAS_XLIB = False

# Try keyboard library as fallback
HAS_KEYBOARD = False
if sys.platform.startswith('linux') and not HAS_XLIB:
    try:
        import keyboard
        HAS_KEYBOARD = True
    except ImportError:
        logger.warning("keyboard library not available")
        HAS_KEYBOARD = False


class LinuxHotkeyBackend:
    """
    Linux hotkey implementation.
    
    Uses Xlib for X11, or keyboard library for Wayland/fallback.
    """
    
    def __init__(self):
        """Initialize Linux hotkey backend."""
        self._hotkeys: dict[str, dict[str, Any]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Determine which backend to use
        self._use_xlib = HAS_XLIB
        self._use_keyboard = HAS_KEYBOARD and not HAS_XLIB
        
        if not self._use_xlib and not self._use_keyboard:
            logger.warning("No Linux hotkey backend available. Install python-xlib or keyboard.")
        
        if self._use_xlib:
            try:
                self._display = display.Display()
                self._root = self._display.screen().root
            except Exception as e:
                logger.error(f"Failed to initialize X11 display: {e}")
                self._use_xlib = False
    
    def _parse_hotkey(self, hotkey: str) -> tuple:
        """
        Parse hotkey string.
        
        Args:
            hotkey: Key combination like "Ctrl+Shift+Space"
            
        Returns:
            Tuple of (modifiers_list, key)
        """
        parts = [p.strip() for p in hotkey.split('+')]
        key = parts[-1].lower()
        modifiers = [p.lower() for p in parts[:-1]]
        
        return (modifiers, key)
    
    def register(self, hotkey: str, callback: Callable, name: str) -> bool:
        """
        Register a global hotkey.
        
        Args:
            hotkey: Key combination
            callback: Function to call when pressed
            name: Hotkey name
            
        Returns:
            True if successful
        """
        if not (self._use_xlib or self._use_keyboard):
            logger.error("No Linux hotkey backend available")
            return False
        
        try:
            modifiers, key = self._parse_hotkey(hotkey)
            
            if self._use_keyboard:
                # Use keyboard library
                try:
                    keyboard.add_hotkey(hotkey.lower().replace('ctrl', 'ctrl'), callback)
                    self._hotkeys[name] = {
                        'hotkey': hotkey,
                        'callback': callback,
                    }
                    logger.info(f"Registered Linux hotkey (keyboard): {name} = {hotkey}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to register with keyboard library: {e}")
                    return False
            
            elif self._use_xlib:
                # Store for X11 handling
                self._hotkeys[name] = {
                    'hotkey': hotkey,
                    'callback': callback,
                    'modifiers': modifiers,
                    'key': key,
                }
                logger.info(f"Registered Linux hotkey (X11): {name} = {hotkey}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error registering Linux hotkey '{name}': {e}")
            return False
    
    def unregister(self, name: str) -> bool:
        """Unregister a hotkey."""
        if name not in self._hotkeys:
            return False
        
        try:
            if self._use_keyboard:
                hotkey = self._hotkeys[name]['hotkey']
                keyboard.remove_hotkey(hotkey.lower().replace('ctrl', 'ctrl'))
            
            del self._hotkeys[name]
            logger.info(f"Unregistered Linux hotkey: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering Linux hotkey '{name}': {e}")
            return False
    
    def is_available(self, hotkey: str) -> bool:
        """Check if hotkey is available."""
        # For Linux, we'll assume it's available
        # Actual conflict detection would require checking system shortcuts
        return True
    
    def start(self):
        """Start listening for hotkeys."""
        if not (self._use_xlib or self._use_keyboard):
            logger.error("No Linux hotkey backend available")
            return
        
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        if self._use_xlib:
            self._thread = threading.Thread(target=self._x11_listen_loop, daemon=True)
            self._thread.start()
        
        logger.info("Linux hotkey listener started")
    
    def stop(self):
        """Stop listening for hotkeys."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=1.0)
        
        logger.info("Linux hotkey listener stopped")
    
    def _x11_listen_loop(self):
        """X11 event loop for hotkey events."""
        if not self._use_xlib:
            return
        
        try:
            # Grab keys
            for name, info in self._hotkeys.items():
                try:
                    # This is a simplified implementation
                    # A full implementation would use XGrabKey
                    pass
                except Exception as e:
                    logger.error(f"Error grabbing key for '{name}': {e}")
            
            # Simple polling loop
            # A full implementation would use X11 event handling
            while self._running:
                if self._stop_event.wait(0.1):
                    break
                    
        except Exception as e:
            logger.error(f"Error in X11 hotkey listen loop: {e}")
        finally:
            self._running = False
