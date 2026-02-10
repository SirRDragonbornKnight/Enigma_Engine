"""
Global Hotkeys for Enigma AI Engine

System-wide keyboard shortcuts for quick access to features:
- Toggle AI assistant
- Push-to-talk
- Quick commands
- Game mode triggers
- Screen capture

Works across platforms (Windows, Linux, macOS) with fallbacks.

Usage:
    from enigma_engine.utils.hotkeys import HotkeyManager, get_hotkey_manager
    
    manager = get_hotkey_manager()
    manager.register("ctrl+shift+a", lambda: print("AI Activated!"))
    manager.start()
"""

from __future__ import annotations

import logging
import platform
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)

# Platform detection
PLATFORM = platform.system().lower()


class HotkeyState(Enum):
    """Hotkey manager states."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class HotkeyBinding:
    """A registered hotkey binding."""
    id: str
    keys: str  # e.g., "ctrl+shift+a"
    callback: Callable
    description: str = ""
    enabled: bool = True
    category: str = "general"


# Key code mappings
KEY_CODES = {
    # Modifiers
    'ctrl': 0x11, 'control': 0x11,
    'alt': 0x12,
    'shift': 0x10,
    'win': 0x5B, 'super': 0x5B, 'meta': 0x5B,
    
    # Letters
    'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45,
    'f': 0x46, 'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A,
    'k': 0x4B, 'l': 0x4C, 'm': 0x4D, 'n': 0x4E, 'o': 0x4F,
    'p': 0x50, 'q': 0x51, 'r': 0x52, 's': 0x53, 't': 0x54,
    'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58, 'y': 0x59,
    'z': 0x5A,
    
    # Numbers
    '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
    '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
    
    # Function keys
    'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73,
    'f5': 0x74, 'f6': 0x75, 'f7': 0x76, 'f8': 0x77,
    'f9': 0x78, 'f10': 0x79, 'f11': 0x7A, 'f12': 0x7B,
    
    # Special keys
    'space': 0x20, 'enter': 0x0D, 'return': 0x0D,
    'tab': 0x09, 'escape': 0x1B, 'esc': 0x1B,
    'backspace': 0x08, 'delete': 0x2E,
    'insert': 0x2D, 'home': 0x24, 'end': 0x23,
    'pageup': 0x21, 'pagedown': 0x22,
    'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
    
    # Numpad
    'numpad0': 0x60, 'numpad1': 0x61, 'numpad2': 0x62,
    'numpad3': 0x63, 'numpad4': 0x64, 'numpad5': 0x65,
    'numpad6': 0x66, 'numpad7': 0x67, 'numpad8': 0x68,
    'numpad9': 0x69, 'numpad+': 0x6B, 'numpad-': 0x6D,
    'numpad*': 0x6A, 'numpad/': 0x6F, 'numpad.': 0x6E,
}


def parse_hotkey(hotkey_str: str) -> Tuple[Set[str], str]:
    """
    Parse a hotkey string into modifiers and key.
    
    Args:
        hotkey_str: e.g., "ctrl+shift+a"
        
    Returns:
        (modifiers, main_key) tuple
    """
    parts = hotkey_str.lower().replace(' ', '').split('+')
    modifiers = set()
    main_key = None
    
    for part in parts:
        if part in ('ctrl', 'control', 'alt', 'shift', 'win', 'super', 'meta'):
            modifiers.add(part if part not in ('control', 'super', 'meta') 
                         else 'ctrl' if part == 'control' else 'win')
        else:
            main_key = part
    
    return modifiers, main_key


class WindowsHotkeyListener:
    """Windows-specific hotkey listener using Win32 API."""
    
    def __init__(self):
        self.hotkeys: Dict[int, HotkeyBinding] = {}
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._next_id = 1
        
        # Try to load Win32 modules
        self._ctypes = None
        self._user32 = None
        try:
            import ctypes
            self._ctypes = ctypes
            self._user32 = ctypes.windll.user32
        except Exception as e:
            logger.warning(f"Win32 API not available: {e}")
    
    def register(self, binding: HotkeyBinding) -> bool:
        """Register a hotkey."""
        if self._user32 is None:
            return False
        
        modifiers, key = parse_hotkey(binding.keys)
        
        # Build modifier flags
        mod_flags = 0
        if 'alt' in modifiers:
            mod_flags |= 0x0001  # MOD_ALT
        if 'ctrl' in modifiers:
            mod_flags |= 0x0002  # MOD_CONTROL
        if 'shift' in modifiers:
            mod_flags |= 0x0004  # MOD_SHIFT
        if 'win' in modifiers:
            mod_flags |= 0x0008  # MOD_WIN
        
        # Add NO_REPEAT flag
        mod_flags |= 0x4000  # MOD_NOREPEAT
        
        # Get virtual key code
        vk_code = KEY_CODES.get(key, 0)
        if vk_code == 0:
            logger.error(f"Unknown key: {key}")
            return False
        
        # Register the hotkey
        hotkey_id = self._next_id
        self._next_id += 1
        
        result = self._user32.RegisterHotKey(None, hotkey_id, mod_flags, vk_code)
        
        if result:
            self.hotkeys[hotkey_id] = binding
            logger.info(f"Registered hotkey: {binding.keys} -> {binding.description}")
            return True
        else:
            logger.error(f"Failed to register hotkey: {binding.keys}")
            return False
    
    def unregister(self, binding: HotkeyBinding) -> bool:
        """Unregister a hotkey."""
        if self._user32 is None:
            return False
        
        # Find hotkey ID
        hotkey_id = None
        for hid, hb in self.hotkeys.items():
            if hb.id == binding.id:
                hotkey_id = hid
                break
        
        if hotkey_id is not None:
            self._user32.UnregisterHotKey(None, hotkey_id)
            del self.hotkeys[hotkey_id]
            return True
        
        return False
    
    def start(self):
        """Start listening for hotkeys."""
        if self._user32 is None or self.running:
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._message_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop listening for hotkeys."""
        self.running = False
        
        # Post quit message to exit message loop
        if self._user32 is not None:
            self._user32.PostThreadMessageW(
                threading.current_thread().ident, 0x0012, 0, 0  # WM_QUIT
            )
    
    def _message_loop(self):
        """Windows message loop for hotkey events."""
        from ctypes import byref, wintypes
        
        class MSG(self._ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("message", wintypes.UINT),
                ("wParam", wintypes.WPARAM),
                ("lParam", wintypes.LPARAM),
                ("time", wintypes.DWORD),
                ("pt", wintypes.POINT),
            ]
        
        msg = MSG()
        WM_HOTKEY = 0x0312
        
        while self.running:
            result = self._user32.GetMessageW(byref(msg), None, 0, 0)
            
            if result == -1 or result == 0:
                break
            
            if msg.message == WM_HOTKEY:
                hotkey_id = msg.wParam
                if hotkey_id in self.hotkeys:
                    binding = self.hotkeys[hotkey_id]
                    if binding.enabled:
                        try:
                            binding.callback()
                        except Exception as e:
                            logger.error(f"Hotkey callback error: {e}")


class PynputHotkeyListener:
    """Cross-platform hotkey listener using pynput library."""
    
    def __init__(self):
        self.hotkeys: Dict[str, HotkeyBinding] = {}
        self.running = False
        self._listener = None
        self._current_keys: Set[str] = set()
        
        # Try to import pynput
        self._pynput = None
        try:
            from pynput import keyboard
            self._pynput = keyboard
        except ImportError:
            logger.warning("pynput not available, hotkeys disabled")
    
    def register(self, binding: HotkeyBinding) -> bool:
        """Register a hotkey."""
        self.hotkeys[binding.keys.lower()] = binding
        logger.info(f"Registered hotkey: {binding.keys} -> {binding.description}")
        return True
    
    def unregister(self, binding: HotkeyBinding) -> bool:
        """Unregister a hotkey."""
        key = binding.keys.lower()
        if key in self.hotkeys:
            del self.hotkeys[key]
            return True
        return False
    
    def start(self):
        """Start listening for hotkeys."""
        if self._pynput is None or self.running:
            return
        
        self.running = True
        
        self._listener = self._pynput.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.start()
    
    def stop(self):
        """Stop listening for hotkeys."""
        self.running = False
        if self._listener:
            self._listener.stop()
            self._listener = None
    
    def _key_to_name(self, key) -> Optional[str]:
        """Convert pynput key to string name."""
        if hasattr(key, 'char') and key.char:
            return key.char.lower()
        elif hasattr(key, 'name'):
            return key.name.lower()
        return None
    
    def _on_press(self, key):
        """Handle key press."""
        key_name = self._key_to_name(key)
        if key_name:
            self._current_keys.add(key_name)
            self._check_hotkeys()
    
    def _on_release(self, key):
        """Handle key release."""
        key_name = self._key_to_name(key)
        if key_name and key_name in self._current_keys:
            self._current_keys.discard(key_name)
    
    def _check_hotkeys(self):
        """Check if any hotkey combination is pressed."""
        for hotkey_str, binding in self.hotkeys.items():
            if not binding.enabled:
                continue
            
            modifiers, main_key = parse_hotkey(hotkey_str)
            
            # Check if all modifiers are pressed
            modifiers_pressed = True
            for mod in modifiers:
                # Map modifier names
                mod_keys = {
                    'ctrl': {'ctrl', 'ctrl_l', 'ctrl_r'},
                    'alt': {'alt', 'alt_l', 'alt_r', 'alt_gr'},
                    'shift': {'shift', 'shift_l', 'shift_r'},
                    'win': {'cmd', 'cmd_l', 'cmd_r', 'super', 'super_l', 'super_r'},
                }
                if not any(k in self._current_keys for k in mod_keys.get(mod, {mod})):
                    modifiers_pressed = False
                    break
            
            # Check if main key is pressed
            if modifiers_pressed and main_key in self._current_keys:
                try:
                    binding.callback()
                except Exception as e:
                    logger.error(f"Hotkey callback error: {e}")


class KeyboardHotkeyListener:
    """Simple fallback using keyboard library."""
    
    def __init__(self):
        self.hotkeys: Dict[str, HotkeyBinding] = {}
        self.running = False
        self._keyboard = None
        
        try:
            import keyboard
            self._keyboard = keyboard
        except ImportError:
            logger.warning("keyboard library not available")
    
    def register(self, binding: HotkeyBinding) -> bool:
        """Register a hotkey."""
        if self._keyboard is None:
            return False
        
        try:
            self._keyboard.add_hotkey(
                binding.keys,
                binding.callback,
                suppress=False
            )
            self.hotkeys[binding.keys] = binding
            logger.info(f"Registered hotkey: {binding.keys}")
            return True
        except Exception as e:
            logger.error(f"Failed to register hotkey: {e}")
            return False
    
    def unregister(self, binding: HotkeyBinding) -> bool:
        """Unregister a hotkey."""
        if self._keyboard is None:
            return False
        
        try:
            self._keyboard.remove_hotkey(binding.keys)
            if binding.keys in self.hotkeys:
                del self.hotkeys[binding.keys]
            return True
        except Exception:
            return False
    
    def start(self):
        """Start listening (keyboard library is always listening)."""
        self.running = True
    
    def stop(self):
        """Stop listening."""
        self.running = False
        if self._keyboard:
            for hotkey in list(self.hotkeys.keys()):
                try:
                    self._keyboard.remove_hotkey(hotkey)
                except Exception:
                    pass
            self.hotkeys.clear()


class HotkeyManager:
    """
    Cross-platform hotkey manager.
    
    Automatically selects the best available backend:
    1. Windows: Native Win32 API
    2. Cross-platform: pynput
    3. Fallback: keyboard library
    """
    
    def __init__(self):
        self.bindings: Dict[str, HotkeyBinding] = {}
        self.state = HotkeyState.STOPPED
        self._listener = None
        self._lock = threading.Lock()
        
        # Select backend
        self._init_backend()
    
    def _init_backend(self):
        """Initialize the appropriate backend."""
        if PLATFORM == 'windows':
            listener = WindowsHotkeyListener()
            if listener._user32 is not None:
                self._listener = listener
                logger.info("Using Windows native hotkey backend")
                return
        
        # Try pynput
        listener = PynputHotkeyListener()
        if listener._pynput is not None:
            self._listener = listener
            logger.info("Using pynput hotkey backend")
            return
        
        # Fallback to keyboard
        listener = KeyboardHotkeyListener()
        if listener._keyboard is not None:
            self._listener = listener
            logger.info("Using keyboard library backend")
            return
        
        logger.warning("No hotkey backend available")
    
    def register(
        self,
        keys: str,
        callback: Callable,
        description: str = "",
        category: str = "general",
        enabled: bool = True
    ) -> str:
        """
        Register a global hotkey.
        
        Args:
            keys: Hotkey combination (e.g., "ctrl+shift+a")
            callback: Function to call when hotkey pressed
            description: Human-readable description
            category: Category for organization
            enabled: Whether hotkey is initially enabled
            
        Returns:
            Binding ID
        """
        import uuid
        
        with self._lock:
            binding_id = str(uuid.uuid4())[:8]
            
            binding = HotkeyBinding(
                id=binding_id,
                keys=keys,
                callback=callback,
                description=description,
                enabled=enabled,
                category=category
            )
            
            self.bindings[binding_id] = binding
            
            # Register with backend if running
            if self._listener and self.state == HotkeyState.RUNNING:
                self._listener.register(binding)
            
            return binding_id
    
    def unregister(self, binding_id: str) -> bool:
        """Unregister a hotkey by ID."""
        with self._lock:
            if binding_id not in self.bindings:
                return False
            
            binding = self.bindings[binding_id]
            
            if self._listener:
                self._listener.unregister(binding)
            
            del self.bindings[binding_id]
            return True
    
    def enable(self, binding_id: str) -> bool:
        """Enable a hotkey."""
        if binding_id in self.bindings:
            self.bindings[binding_id].enabled = True
            return True
        return False
    
    def disable(self, binding_id: str) -> bool:
        """Disable a hotkey."""
        if binding_id in self.bindings:
            self.bindings[binding_id].enabled = False
            return True
        return False
    
    def start(self):
        """Start listening for hotkeys."""
        if self._listener is None:
            logger.error("No hotkey backend available")
            return
        
        with self._lock:
            if self.state == HotkeyState.RUNNING:
                return
            
            # Register all bindings
            for binding in self.bindings.values():
                self._listener.register(binding)
            
            self._listener.start()
            self.state = HotkeyState.RUNNING
            logger.info("Hotkey manager started")
    
    def stop(self):
        """Stop listening for hotkeys."""
        if self._listener is None:
            return
        
        with self._lock:
            if self.state == HotkeyState.STOPPED:
                return
            
            self._listener.stop()
            self.state = HotkeyState.STOPPED
            logger.info("Hotkey manager stopped")
    
    def pause(self):
        """Pause hotkey listening."""
        self.state = HotkeyState.PAUSED
        for binding in self.bindings.values():
            binding.enabled = False
    
    def resume(self):
        """Resume hotkey listening."""
        self.state = HotkeyState.RUNNING
        for binding in self.bindings.values():
            binding.enabled = True
    
    def get_bindings(self, category: Optional[str] = None) -> List[HotkeyBinding]:
        """Get all bindings, optionally filtered by category."""
        if category:
            return [b for b in self.bindings.values() if b.category == category]
        return list(self.bindings.values())
    
    def get_binding(self, binding_id: str) -> Optional[HotkeyBinding]:
        """Get a specific binding by ID."""
        return self.bindings.get(binding_id)


# Global instance
_hotkey_manager: Optional[HotkeyManager] = None
_manager_lock = threading.Lock()


def get_hotkey_manager() -> HotkeyManager:
    """Get the global hotkey manager instance."""
    global _hotkey_manager
    
    with _manager_lock:
        if _hotkey_manager is None:
            _hotkey_manager = HotkeyManager()
        return _hotkey_manager


# Default hotkey definitions for Enigma AI Engine
DEFAULT_HOTKEYS = {
    "toggle_ai": {
        "keys": "ctrl+shift+a",
        "description": "Toggle AI assistant window",
        "category": "assistant"
    },
    "push_to_talk": {
        "keys": "ctrl+shift+space",
        "description": "Push to talk (hold)",
        "category": "voice"
    },
    "quick_command": {
        "keys": "ctrl+shift+c",
        "description": "Quick command input",
        "category": "assistant"
    },
    "screenshot_analyze": {
        "keys": "ctrl+shift+s",
        "description": "Screenshot and analyze",
        "category": "vision"
    },
    "game_mode_toggle": {
        "keys": "ctrl+shift+g",
        "description": "Toggle game mode",
        "category": "game"
    },
    "emergency_stop": {
        "keys": "ctrl+shift+escape",
        "description": "Emergency stop all AI actions",
        "category": "safety"
    }
}


def setup_default_hotkeys(callbacks: Optional[Dict[str, Callable]] = None):
    """
    Set up default Enigma AI Engine hotkeys.
    
    Args:
        callbacks: Dict mapping hotkey name to callback function
    """
    manager = get_hotkey_manager()
    callbacks = callbacks or {}
    
    for name, config in DEFAULT_HOTKEYS.items():
        callback = callbacks.get(name, lambda n=name: logger.info(f"Hotkey triggered: {n}"))
        
        manager.register(
            keys=config["keys"],
            callback=callback,
            description=config["description"],
            category=config["category"]
        )
    
    logger.info(f"Registered {len(DEFAULT_HOTKEYS)} default hotkeys")
