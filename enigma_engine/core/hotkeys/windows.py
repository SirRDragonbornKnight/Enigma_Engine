"""
Windows Hotkey Backend - Global hotkey implementation for Windows.

Uses the Windows API (via ctypes) to register global hotkeys using RegisterHotKey.
"""

import logging
import sys
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Only import Windows-specific modules on Windows
if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes

    # Windows API constants
    MOD_ALT = 0x0001
    MOD_CONTROL = 0x0002
    MOD_SHIFT = 0x0004
    MOD_WIN = 0x0008
    WM_HOTKEY = 0x0312
    
    # Windows API functions
    user32 = ctypes.windll.user32
    RegisterHotKey = user32.RegisterHotKey
    UnregisterHotKey = user32.UnregisterHotKey


class WindowsHotkeyBackend:
    """
    Windows hotkey implementation using ctypes/win32api.
    
    Uses RegisterHotKey for global hotkeys.
    """
    
    def __init__(self):
        """Initialize Windows hotkey backend."""
        self._hotkeys: dict[str, dict[str, Any]] = {}
        self._next_id = 1
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def _parse_hotkey(self, hotkey: str) -> tuple:
        """
        Parse hotkey string into modifiers and virtual key code.
        
        Args:
            hotkey: Key combination like "Ctrl+Shift+Space"
            
        Returns:
            Tuple of (modifiers, vk_code)
        """
        if sys.platform != 'win32':
            raise RuntimeError("Windows backend only works on Windows")
        
        parts = [p.strip() for p in hotkey.split('+')]
        
        modifiers = 0
        key = parts[-1]  # Last part is the actual key
        
        # Parse modifiers
        for part in parts[:-1]:
            part_lower = part.lower()
            if part_lower in ('ctrl', 'control'):
                modifiers |= MOD_CONTROL
            elif part_lower == 'shift':
                modifiers |= MOD_SHIFT
            elif part_lower == 'alt':
                modifiers |= MOD_ALT
            elif part_lower in ('win', 'super', 'meta'):
                modifiers |= MOD_WIN
        
        # Convert key to virtual key code
        vk_code = self._key_to_vk(key)
        
        return (modifiers, vk_code)
    
    def _key_to_vk(self, key: str) -> int:
        """
        Convert key name to Windows virtual key code.
        
        Args:
            key: Key name (e.g., "Space", "F12", "A")
            
        Returns:
            Virtual key code
        """
        if sys.platform != 'win32':
            return 0
        
        key_upper = key.upper()
        
        # Common special keys
        special_keys = {
            'SPACE': 0x20,
            'ENTER': 0x0D,
            'RETURN': 0x0D,
            'TAB': 0x09,
            'BACKSPACE': 0x08,
            'DELETE': 0x2E,
            'INSERT': 0x2D,
            'HOME': 0x24,
            'END': 0x23,
            'PAGEUP': 0x21,
            'PAGEDOWN': 0x22,
            'UP': 0x26,
            'DOWN': 0x28,
            'LEFT': 0x25,
            'RIGHT': 0x27,
            'ESCAPE': 0x1B,
            'ESC': 0x1B,
        }
        
        # Function keys F1-F12
        if key_upper.startswith('F') and len(key_upper) <= 3:
            try:
                num = int(key_upper[1:])
                if 1 <= num <= 12:
                    return 0x70 + num - 1
            except ValueError:
                pass  # Intentionally silent
        
        # Check special keys
        if key_upper in special_keys:
            return special_keys[key_upper]
        
        # Single character keys (A-Z, 0-9)
        if len(key) == 1:
            if key.isalpha():
                return ord(key_upper)
            elif key.isdigit():
                return ord(key)
        
        # Default fallback
        logger.warning(f"Unknown key: {key}, using Space as fallback")
        return 0x20  # Space
    
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
        if sys.platform != 'win32':
            logger.error("Windows backend only works on Windows")
            return False
        
        try:
            modifiers, vk_code = self._parse_hotkey(hotkey)
            hotkey_id = self._next_id
            self._next_id += 1
            
            # Register with Windows
            result = RegisterHotKey(None, hotkey_id, modifiers, vk_code)
            
            if result:
                self._hotkeys[name] = {
                    'id': hotkey_id,
                    'hotkey': hotkey,
                    'callback': callback,
                    'modifiers': modifiers,
                    'vk_code': vk_code,
                }
                logger.info(f"Registered Windows hotkey: {name} = {hotkey}")
                return True
            else:
                logger.error(f"Failed to register Windows hotkey: {name} = {hotkey}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering Windows hotkey '{name}': {e}")
            return False
    
    def unregister(self, name: str) -> bool:
        """Unregister a hotkey."""
        if sys.platform != 'win32':
            return False
        
        if name not in self._hotkeys:
            return False
        
        try:
            hotkey_id = self._hotkeys[name]['id']
            result = UnregisterHotKey(None, hotkey_id)
            
            if result:
                del self._hotkeys[name]
                logger.info(f"Unregistered Windows hotkey: {name}")
                return True
            else:
                logger.error(f"Failed to unregister Windows hotkey: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Error unregistering Windows hotkey '{name}': {e}")
            return False
    
    def is_available(self, hotkey: str) -> bool:
        """Check if hotkey is available."""
        if sys.platform != 'win32':
            return False
        
        try:
            modifiers, vk_code = self._parse_hotkey(hotkey)
            # Try to register temporarily
            test_id = 9999
            result = RegisterHotKey(None, test_id, modifiers, vk_code)
            if result:
                UnregisterHotKey(None, test_id)
                return True
            return False
        except (OSError, ctypes.WinError, Exception):
            return False  # Hotkey registration failed
    
    def start(self):
        """Start listening for hotkeys."""
        if sys.platform != 'win32':
            logger.error("Windows backend only works on Windows")
            return
        
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("Windows hotkey listener started")
    
    def stop(self):
        """Stop listening for hotkeys."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=1.0)
        
        logger.info("Windows hotkey listener stopped")
    
    def _listen_loop(self):
        """Message loop for hotkey events."""
        if sys.platform != 'win32':
            return
        
        try:
            msg = wintypes.MSG()
            while self._running:
                # Check for messages with a timeout
                result = user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1)
                
                if result:
                    if msg.message == WM_HOTKEY:
                        # Find and call the callback
                        hotkey_id = msg.wParam
                        for name, info in self._hotkeys.items():
                            if info['id'] == hotkey_id:
                                try:
                                    info['callback']()
                                except Exception as e:
                                    logger.error(f"Error in hotkey callback '{name}': {e}")
                                break
                
                # Small sleep to avoid busy-waiting
                if self._stop_event.wait(0.01):
                    break
                    
        except Exception as e:
            logger.error(f"Error in Windows hotkey listen loop: {e}")
        finally:
            self._running = False
