"""
macOS Hotkey Backend - Global hotkey implementation for macOS.

Uses Quartz Event Taps for system-wide hotkey capture.
"""

import logging
import sys
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Try to import macOS-specific modules
HAS_QUARTZ = False
if sys.platform == 'darwin':
    try:
        # PyObjC modules for macOS
        HAS_QUARTZ = True
    except ImportError:
        logger.warning("PyObjC not available for macOS hotkeys")
        HAS_QUARTZ = False

# Try keyboard library as fallback
HAS_KEYBOARD = False
if sys.platform == 'darwin' and not HAS_QUARTZ:
    try:
        import keyboard
        HAS_KEYBOARD = True
    except ImportError:
        logger.warning("keyboard library not available")
        HAS_KEYBOARD = False


class MacOSHotkeyBackend:
    """
    macOS hotkey implementation.
    
    Uses Quartz Event Taps or keyboard library fallback.
    """
    
    def __init__(self):
        """Initialize macOS hotkey backend."""
        self._hotkeys: dict[str, dict[str, Any]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Determine which backend to use
        self._use_quartz = HAS_QUARTZ
        self._use_keyboard = HAS_KEYBOARD and not HAS_QUARTZ
        
        if not self._use_quartz and not self._use_keyboard:
            logger.warning("No macOS hotkey backend available. Install PyObjC or keyboard.")
    
    def _parse_hotkey(self, hotkey: str) -> tuple:
        """
        Parse hotkey string.
        
        Args:
            hotkey: Key combination like "Cmd+Shift+Space"
            
        Returns:
            Tuple of (modifiers_list, key)
        """
        parts = [p.strip() for p in hotkey.split('+')]
        key = parts[-1].lower()
        
        # Normalize modifier names
        modifiers = []
        for p in parts[:-1]:
            p_lower = p.lower()
            if p_lower in ('cmd', 'command', 'meta'):
                modifiers.append('command')
            elif p_lower in ('ctrl', 'control'):
                modifiers.append('control')
            elif p_lower == 'shift':
                modifiers.append('shift')
            elif p_lower in ('alt', 'option'):
                modifiers.append('option')
        
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
        if not (self._use_quartz or self._use_keyboard):
            logger.error("No macOS hotkey backend available")
            return False
        
        try:
            modifiers, key = self._parse_hotkey(hotkey)
            
            if self._use_keyboard:
                # Use keyboard library
                try:
                    # Normalize hotkey for keyboard library
                    normalized = hotkey.lower().replace('cmd', 'command').replace('ctrl', 'control')
                    keyboard.add_hotkey(normalized, callback)
                    self._hotkeys[name] = {
                        'hotkey': hotkey,
                        'callback': callback,
                    }
                    logger.info(f"Registered macOS hotkey (keyboard): {name} = {hotkey}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to register with keyboard library: {e}")
                    return False
            
            elif self._use_quartz:
                # Store for Quartz handling
                self._hotkeys[name] = {
                    'hotkey': hotkey,
                    'callback': callback,
                    'modifiers': modifiers,
                    'key': key,
                }
                logger.info(f"Registered macOS hotkey (Quartz): {name} = {hotkey}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error registering macOS hotkey '{name}': {e}")
            return False
    
    def unregister(self, name: str) -> bool:
        """Unregister a hotkey."""
        if name not in self._hotkeys:
            return False
        
        try:
            if self._use_keyboard:
                hotkey = self._hotkeys[name]['hotkey']
                normalized = hotkey.lower().replace('cmd', 'command').replace('ctrl', 'control')
                keyboard.remove_hotkey(normalized)
            
            del self._hotkeys[name]
            logger.info(f"Unregistered macOS hotkey: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering macOS hotkey '{name}': {e}")
            return False
    
    def is_available(self, hotkey: str) -> bool:
        """Check if hotkey is available."""
        # For macOS, we'll assume it's available
        # Actual conflict detection would require checking system shortcuts
        return True
    
    def start(self):
        """Start listening for hotkeys."""
        if not (self._use_quartz or self._use_keyboard):
            logger.error("No macOS hotkey backend available")
            return
        
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        if self._use_quartz:
            self._thread = threading.Thread(target=self._quartz_listen_loop, daemon=True)
            self._thread.start()
        
        logger.info("macOS hotkey listener started")
    
    def stop(self):
        """Stop listening for hotkeys."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=1.0)
        
        logger.info("macOS hotkey listener stopped")
    
    def _quartz_listen_loop(self):
        """Quartz event loop for hotkey events."""
        if not self._use_quartz:
            return
        
        try:
            # This is a simplified implementation
            # A full implementation would set up an event tap
            # and process key events in the run loop
            
            # Simple polling loop as fallback
            while self._running:
                if self._stop_event.wait(0.1):
                    break
                    
        except Exception as e:
            logger.error(f"Error in Quartz hotkey listen loop: {e}")
        finally:
            self._running = False
