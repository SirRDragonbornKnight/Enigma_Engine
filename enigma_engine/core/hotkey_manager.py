"""
================================================================================
HOTKEY MANAGER - Global Hotkey System
================================================================================

Global hotkey registration and handling system that works across all platforms,
even when Enigma AI Engine is not the focused window.

FILE: enigma_engine/core/hotkey_manager.py
TYPE: Core Utility
MAIN CLASS: HotkeyManager

FEATURES:
    - Global hotkey registration (works in fullscreen games)
    - Platform-specific backends (Windows, Linux, macOS)
    - Conflict detection
    - Easy rebinding
    - Default hotkey presets
    - Per-application hotkey profiles

USAGE:
    from enigma_engine.core.hotkey_manager import HotkeyManager, DEFAULT_HOTKEYS
    
    manager = HotkeyManager()
    manager.register(
        hotkey=DEFAULT_HOTKEYS["summon_ai"],
        callback=show_overlay,
        name="summon_ai"
    )
    
    manager.start()
    # ... hotkeys are now active ...
    manager.stop()
    
    # Per-app profiles
    manager.add_app_profile("Discord", {"summon_ai": "Ctrl+D"})
    manager.start_app_detection()  # Auto-switches profiles based on foreground app
"""

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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


@dataclass
class HotkeyProfile:
    """
    A profile of hotkey bindings for a specific application context.
    
    Allows different hotkeys for different applications (e.g., games vs chat apps).
    """
    name: str                                   # Profile name (e.g., "Discord", "Minecraft")
    app_patterns: List[str] = field(default_factory=list)  # Window title/process patterns to match
    hotkey_overrides: Dict[str, str] = field(default_factory=dict)  # name -> hotkey mapping
    enabled: bool = True
    
    def matches_app(self, window_title: str, process_name: str) -> bool:
        """Check if this profile matches the given app."""
        check_str = f"{window_title.lower()} {process_name.lower()}"
        return any(pattern.lower() in check_str for pattern in self.app_patterns)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "app_patterns": self.app_patterns,
            "hotkey_overrides": self.hotkey_overrides,
            "enabled": self.enabled,
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'HotkeyProfile':
        return HotkeyProfile(
            name=data.get("name", "Unknown"),
            app_patterns=data.get("app_patterns", []),
            hotkey_overrides=data.get("hotkey_overrides", {}),
            enabled=data.get("enabled", True),
        )


class ChordSequence:
    """
    Multi-step hotkey sequence (chord) that triggers only after
    pressing a sequence of keys in order within a time window.
    
    Example:
        # Ctrl+K followed by Ctrl+C within 1 second
        chord = ChordSequence(
            name="copy_special",
            sequence=["Ctrl+K", "Ctrl+C"],
            callback=do_special_copy,
            timeout=1.0
        )
    """
    
    def __init__(
        self,
        name: str,
        sequence: List[str],
        callback: Callable,
        timeout: float = 1.0
    ):
        """
        Initialize a chord sequence.
        
        Args:
            name: Unique name for this chord
            sequence: List of key combos in order (e.g., ["Ctrl+K", "Ctrl+C"])
            callback: Function to call when sequence completes
            timeout: Max seconds between steps (resets if exceeded)
        """
        self.name = name
        self.sequence = sequence
        self.callback = callback
        self.timeout = timeout
        
        self._current_index = 0
        self._last_press_time: Optional[float] = None
    
    def on_key(self, hotkey: str) -> bool:
        """
        Handle a key press. Call this when any key in the sequence is pressed.
        
        Args:
            hotkey: The hotkey string that was pressed
            
        Returns:
            True if the sequence completed and callback was triggered
        """
        current_time = time.time()
        
        # Check timeout
        if self._last_press_time and (current_time - self._last_press_time) > self.timeout:
            self._current_index = 0  # Reset
        
        # Check if this is the next expected key
        if self._current_index < len(self.sequence):
            expected = self.sequence[self._current_index]
            if hotkey.lower() == expected.lower():
                self._current_index += 1
                self._last_press_time = current_time
                
                # Check if complete
                if self._current_index >= len(self.sequence):
                    self._current_index = 0
                    self._last_press_time = None
                    try:
                        self.callback()
                    except Exception as e:
                        logger.error(f"Chord callback error: {e}")
                    return True
        
        return False
    
    def reset(self) -> None:
        """Reset the sequence progress."""
        self._current_index = 0
        self._last_press_time = None
    
    @property
    def progress(self) -> int:
        """Current step in the sequence (0-indexed)."""
        return self._current_index
    
    @property
    def is_in_progress(self) -> bool:
        """Whether a sequence has been started but not completed."""
        return self._current_index > 0


class ChordManager:
    """
    Manages multiple chord sequences.
    
    Works alongside the regular HotkeyManager to support multi-step sequences.
    
    Usage:
        chords = ChordManager()
        chords.add_chord("copy_special", ["Ctrl+K", "Ctrl+C"], my_callback)
        
        # In your hotkey callback:
        def on_any_key(hotkey):
            chords.on_key(hotkey)
    """
    
    def __init__(self) -> None:
        self._chords: Dict[str, ChordSequence] = {}
        self._lock = threading.Lock()
    
    def add_chord(
        self,
        name: str,
        sequence: List[str],
        callback: Callable,
        timeout: float = 1.0
    ) -> bool:
        """Add a new chord sequence."""
        with self._lock:
            self._chords[name] = ChordSequence(
                name=name,
                sequence=sequence,
                callback=callback,
                timeout=timeout
            )
            logger.info(f"Added chord: {name} = {' -> '.join(sequence)}")
            return True
    
    def remove_chord(self, name: str) -> bool:
        """Remove a chord sequence."""
        with self._lock:
            if name in self._chords:
                del self._chords[name]
                logger.info(f"Removed chord: {name}")
                return True
            return False
    
    def on_key(self, hotkey: str) -> List[str]:
        """
        Process a key press across all chords.
        
        Args:
            hotkey: The hotkey string that was pressed
            
        Returns:
            List of chord names that completed
        """
        completed = []
        with self._lock:
            for chord in self._chords.values():
                if chord.on_key(hotkey):
                    completed.append(chord.name)
        return completed
    
    def reset_all(self) -> None:
        """Reset all chord sequences."""
        with self._lock:
            for chord in self._chords.values():
                chord.reset()
    
    def list_chords(self) -> List[Dict[str, Any]]:
        """List all registered chords."""
        with self._lock:
            return [
                {
                    "name": c.name,
                    "sequence": c.sequence,
                    "timeout": c.timeout,
                    "progress": c.progress,
                    "in_progress": c.is_in_progress,
                }
                for c in self._chords.values()
            ]


class HotkeyManager:
    """
    Global hotkey registration and handling.
    
    Works even when:
    - Game is fullscreen
    - Another app is focused
    - Multiple monitors
    
    Supports per-application hotkey profiles that auto-switch
    based on the foreground window.
    """
    
    def __init__(self, profiles_path: Optional[str] = None) -> None:
        """Initialize the hotkey manager."""
        self._hotkeys: dict[str, HotkeyInfo] = {}
        self._backend: Optional[Any] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Per-app profile support
        self._profiles: Dict[str, HotkeyProfile] = {}
        self._active_profile: Optional[str] = None
        self._base_bindings: Dict[str, str] = {}  # Original hotkey bindings
        self._app_detecting = False
        self._detection_thread: Optional[threading.Thread] = None
        self._on_profile_changed: List[Callable[[Optional[str], Optional[str]], None]] = []
        
        # Profile persistence path
        if profiles_path:
            self._profiles_path = Path(profiles_path)
        else:
            from enigma_engine.config import CONFIG
            data_dir = CONFIG.get('data_dir', 'data') if isinstance(CONFIG, dict) else getattr(CONFIG, 'data_dir', 'data')
            self._profiles_path = Path(data_dir) / "hotkey_profiles.json"
        
        self._initialize_backend()
        self._load_profiles()
    
    def _initialize_backend(self) -> None:
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
    
    def unregister_all(self) -> None:
        """Unregister all hotkeys."""
        if not self._backend:
            return
        
        with self._lock:
            names = list(self._hotkeys.keys())
            for name in names:
                self.unregister(name)
    
    def list_registered(self) -> list[dict[str, Any]]:
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
    
    def start(self) -> None:
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
    
    def stop(self) -> None:
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
    
    # ===== Per-Application Profile Support =====
    
    def add_profile(
        self,
        name: str,
        app_patterns: List[str],
        hotkey_overrides: Dict[str, str],
        enabled: bool = True
    ) -> bool:
        """
        Add a per-application hotkey profile.
        
        Args:
            name: Profile name (e.g., "Discord", "Minecraft")
            app_patterns: Window title/process patterns to match
            hotkey_overrides: Dict of hotkey name -> new binding
            enabled: Whether profile is active
            
        Returns:
            True if added successfully
            
        Example:
            manager.add_profile(
                "Discord",
                app_patterns=["discord", "Discord"],
                hotkey_overrides={"summon_ai": "Ctrl+D", "push_to_talk": "Ctrl+T"}
            )
        """
        with self._lock:
            self._profiles[name] = HotkeyProfile(
                name=name,
                app_patterns=app_patterns,
                hotkey_overrides=hotkey_overrides,
                enabled=enabled
            )
        
        self._save_profiles()
        logger.info(f"Added hotkey profile: {name}")
        return True
    
    def remove_profile(self, name: str) -> bool:
        """Remove a hotkey profile."""
        with self._lock:
            if name in self._profiles:
                del self._profiles[name]
                self._save_profiles()
                logger.info(f"Removed hotkey profile: {name}")
                return True
        return False
    
    def get_profile(self, name: str) -> Optional[HotkeyProfile]:
        """Get a profile by name."""
        return self._profiles.get(name)
    
    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all profiles."""
        return [p.to_dict() for p in self._profiles.values()]
    
    def get_active_profile(self) -> Optional[str]:
        """Get the currently active profile name."""
        return self._active_profile
    
    def activate_profile(self, name: Optional[str] = None) -> None:
        """
        Activate a specific profile or return to default.
        
        Args:
            name: Profile name, or None for default bindings
        """
        old_profile = self._active_profile
        
        if name is None:
            # Revert to base bindings
            self._apply_bindings(self._base_bindings)
            self._active_profile = None
            logger.info("Reverted to default hotkey bindings")
        else:
            profile = self._profiles.get(name)
            if not profile or not profile.enabled:
                logger.warning(f"Profile not found or disabled: {name}")
                return
            
            # Apply profile overrides
            bindings = dict(self._base_bindings)
            bindings.update(profile.hotkey_overrides)
            self._apply_bindings(bindings)
            self._active_profile = name
            logger.info(f"Activated hotkey profile: {name}")
        
        # Notify callbacks
        if old_profile != self._active_profile:
            for cb in self._on_profile_changed:
                try:
                    cb(old_profile, self._active_profile)
                except Exception as e:
                    logger.debug(f"Profile change callback error: {e}")
    
    def _apply_bindings(self, bindings: Dict[str, str]) -> None:
        """Apply a set of hotkey bindings, re-registering as needed."""
        for name, info in list(self._hotkeys.items()):
            new_hotkey = bindings.get(name)
            if new_hotkey and new_hotkey != info.hotkey:
                # Re-register with new binding
                callback = info.callback
                self.unregister(name)
                self.register(new_hotkey, callback, name)
    
    def on_profile_changed(self, callback: Callable[[Optional[str], Optional[str]], None]) -> None:
        """Register callback when active profile changes. Args: (old_profile, new_profile)"""
        self._on_profile_changed.append(callback)
    
    # ===== Auto-detection =====
    
    def start_app_detection(self, interval: float = 2.0) -> None:
        """
        Start automatic profile switching based on foreground app.
        
        Args:
            interval: Seconds between checks
        """
        if self._app_detecting:
            return
        
        # Store current bindings as base
        self._base_bindings = {
            name: info.hotkey for name, info in self._hotkeys.items()
        }
        
        self._app_detecting = True
        self._detection_thread = threading.Thread(
            target=self._detection_loop,
            args=(interval,),
            daemon=True
        )
        self._detection_thread.start()
        logger.info("Started app detection for hotkey profiles")
    
    def stop_app_detection(self) -> None:
        """Stop automatic profile switching."""
        self._app_detecting = False
        if self._detection_thread:
            self._detection_thread.join(timeout=5.0)
            self._detection_thread = None
        logger.info("Stopped app detection for hotkey profiles")
    
    def _detection_loop(self, interval: float) -> None:
        """Detection loop for auto-switching profiles."""
        while self._app_detecting:
            try:
                window_title, process_name = self._get_foreground_app()
                
                # Check profiles
                matched_profile = None
                for profile in self._profiles.values():
                    if profile.enabled and profile.matches_app(window_title, process_name):
                        matched_profile = profile.name
                        break
                
                # Switch if needed
                if matched_profile != self._active_profile:
                    self.activate_profile(matched_profile)
            
            except Exception as e:
                logger.debug(f"App detection error: {e}")
            
            time.sleep(interval)
    
    def _get_foreground_app(self) -> tuple[str, str]:
        """Get the foreground window title and process name."""
        if sys.platform == 'win32':
            try:
                import ctypes
                from ctypes import wintypes
                
                user32 = ctypes.windll.user32
                psapi = ctypes.windll.psapi
                kernel32 = ctypes.windll.kernel32
                
                # Get foreground window
                hwnd = user32.GetForegroundWindow()
                
                # Get window title
                length = user32.GetWindowTextLengthW(hwnd)
                title = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, title, length + 1)
                window_title = title.value
                
                # Get process name
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                
                PROCESS_QUERY_INFORMATION = 0x0400
                PROCESS_VM_READ = 0x0010
                handle = kernel32.OpenProcess(
                    PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
                    False,
                    pid.value
                )
                
                process_name = ""
                if handle:
                    try:
                        name_buffer = ctypes.create_unicode_buffer(260)
                        psapi.GetModuleBaseNameW(handle, None, name_buffer, 260)
                        process_name = name_buffer.value
                    finally:
                        kernel32.CloseHandle(handle)
                
                return window_title, process_name
            except Exception as e:
                logger.debug(f"Windows foreground detection failed: {e}")
                return "", ""
        else:
            # Linux/macOS - basic implementation
            try:
                import subprocess
                result = subprocess.run(
                    ['xdotool', 'getactivewindow', 'getwindowname'],
                    capture_output=True,
                    text=True,
                    timeout=1.0
                )
                return result.stdout.strip(), ""
            except Exception:
                return "", ""
    
    # ===== Profile Persistence =====
    
    def _load_profiles(self) -> None:
        """Load profiles from disk."""
        if self._profiles_path.exists():
            try:
                with open(self._profiles_path) as f:
                    data = json.load(f)
                
                for profile_data in data.get("profiles", []):
                    profile = HotkeyProfile.from_dict(profile_data)
                    self._profiles[profile.name] = profile
                
                logger.debug(f"Loaded {len(self._profiles)} hotkey profiles")
            except Exception as e:
                logger.warning(f"Failed to load hotkey profiles: {e}")
    
    def _save_profiles(self) -> None:
        """Save profiles to disk."""
        try:
            self._profiles_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "profiles": [p.to_dict() for p in self._profiles.values()]
            }
            
            with open(self._profiles_path, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.warning(f"Failed to save hotkey profiles: {e}")

    def __del__(self) -> None:
        """Cleanup when manager is destroyed."""
        try:
            self.stop_app_detection()
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
