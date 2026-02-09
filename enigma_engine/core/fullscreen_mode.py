"""
================================================================================
Fullscreen Mode - Enhanced visibility control for fullscreen applications.
================================================================================

This module provides intelligent visibility management when fullscreen apps
are detected (games, video players, presentations, etc.).

FEATURES:
1. Per-monitor control - Show avatar on specific monitors only
2. Category toggles - Control avatar, spawned objects, effects, particles
3. Smooth fade transitions - Elegant fade in/out instead of instant hide
4. Global hotkey toggle - Instant visibility toggle via hotkey

USAGE:
    from enigma_engine.core.fullscreen_mode import get_fullscreen_controller
    
    controller = get_fullscreen_controller()
    
    # Configure which monitors show the avatar
    controller.set_allowed_monitors([1, 2])  # Only show on monitors 1 and 2
    
    # Configure which elements are visible
    controller.set_category_visible('avatar', True)
    controller.set_category_visible('spawned_objects', False)
    
    # Register elements for visibility control
    controller.register_element('avatar', avatar_window, category='avatar')
    controller.register_element('speech_bubble', bubble_window, category='spawned_objects')
    
    # Global hotkey toggle
    controller.set_toggle_hotkey('ctrl+shift+h')
"""

import logging
import platform
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import json

logger = logging.getLogger(__name__)


class ElementCategory(Enum):
    """Categories of visual elements that can be toggled."""
    AVATAR = "avatar"
    SPAWNED_OBJECTS = "spawned_objects"
    EFFECTS = "effects"
    PARTICLES = "particles"


@dataclass
class VisibilitySettings:
    """Settings for visibility control."""
    # Per-category visibility
    category_visible: Dict[str, bool] = field(default_factory=lambda: {
        "avatar": True,
        "spawned_objects": True,
        "effects": True,
        "particles": True,
    })
    
    # Per-monitor settings (None = all monitors)
    allowed_monitors: Optional[List[int]] = None
    
    # Fade settings
    fade_enabled: bool = True
    fade_duration_ms: int = 300  # Milliseconds for fade transition
    
    # Fullscreen detection
    auto_hide_on_fullscreen: bool = True
    fullscreen_check_interval: float = 2.0  # Seconds
    
    # Hotkey
    toggle_hotkey: Optional[str] = None  # e.g., "ctrl+shift+h"


@dataclass
class RegisteredElement:
    """A visual element registered for visibility control."""
    element_id: str
    widget: Any  # QWidget or similar
    category: str
    monitor: Optional[int] = None  # Which monitor the element is on
    _original_opacity: float = 1.0
    _target_opacity: float = 1.0
    _current_opacity: float = 1.0


class FullscreenController:
    """
    Controls visibility of visual elements based on fullscreen detection
    and user preferences.
    """
    
    def __init__(self):
        self._settings = VisibilitySettings()
        self._elements: Dict[str, RegisteredElement] = {}
        
        # State
        self._enabled = False
        self._fullscreen_active = False
        self._manually_hidden = False  # User toggled via hotkey
        
        # Monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Fade animation state
        self._fade_timer = None
        self._fade_callbacks: List[Callable[[], None]] = []
        
        # Callbacks
        self._on_visibility_change: List[Callable[[bool], None]] = []
        self._on_fullscreen_change: List[Callable[[bool], None]] = []
        
        # Hotkey registration
        self._hotkey_registered = False
        self._hotkey_id = None
        
        # Monitor info cache
        self._monitor_cache: List[Dict] = []
        self._monitor_cache_time = 0
    
    @property
    def is_visible(self) -> bool:
        """Whether elements should currently be visible."""
        if self._manually_hidden:
            return False
        if self._fullscreen_active and self._settings.auto_hide_on_fullscreen:
            return False
        return True
    
    @property
    def fullscreen_detected(self) -> bool:
        """Whether a fullscreen app is currently detected."""
        return self._fullscreen_active
    
    # ========================================================================
    # Configuration Methods
    # ========================================================================
    
    def set_allowed_monitors(self, monitors: Optional[List[int]]):
        """
        Set which monitors can show elements.
        
        Args:
            monitors: List of monitor indices (0-based), or None for all
        """
        self._settings.allowed_monitors = monitors
        self._update_element_visibility()
    
    def set_category_visible(self, category: str, visible: bool):
        """
        Set visibility for an entire category.
        
        Args:
            category: Category name ('avatar', 'spawned_objects', 'effects', 'particles')
            visible: Whether elements in this category should be visible
        """
        self._settings.category_visible[category] = visible
        self._update_element_visibility()
    
    def get_category_visible(self, category: str) -> bool:
        """Get visibility setting for a category."""
        return self._settings.category_visible.get(category, True)
    
    def set_fade_enabled(self, enabled: bool, duration_ms: int = 300):
        """Enable or disable fade transitions."""
        self._settings.fade_enabled = enabled
        self._settings.fade_duration_ms = duration_ms
    
    def set_auto_hide_on_fullscreen(self, enabled: bool):
        """Enable or disable auto-hide when fullscreen app detected."""
        self._settings.auto_hide_on_fullscreen = enabled
        self._update_element_visibility()
    
    def set_toggle_hotkey(self, hotkey: Optional[str]):
        """
        Set the global hotkey for toggling visibility.
        
        Args:
            hotkey: Hotkey string like 'ctrl+shift+h', or None to disable
        """
        # Unregister old hotkey
        if self._hotkey_registered:
            self._unregister_hotkey()
        
        self._settings.toggle_hotkey = hotkey
        
        # Register new hotkey
        if hotkey:
            self._register_hotkey(hotkey)
    
    # ========================================================================
    # Element Registration
    # ========================================================================
    
    def register_element(
        self,
        element_id: str,
        widget: Any,
        category: str = "avatar",
    ):
        """
        Register a visual element for visibility control.
        
        Args:
            element_id: Unique identifier for the element
            widget: The Qt widget to control
            category: Element category for group toggling
        """
        element = RegisteredElement(
            element_id=element_id,
            widget=widget,
            category=category,
        )
        
        # Get current opacity if supported
        if hasattr(widget, 'windowOpacity'):
            element._original_opacity = widget.windowOpacity()
            element._current_opacity = element._original_opacity
        
        self._elements[element_id] = element
        
        # Apply current visibility state
        self._apply_visibility_to_element(element)
        
        logger.debug(f"Registered element: {element_id} (category: {category})")
    
    def unregister_element(self, element_id: str):
        """Unregister an element from visibility control."""
        if element_id in self._elements:
            # Restore original opacity
            element = self._elements[element_id]
            if hasattr(element.widget, 'setWindowOpacity'):
                element.widget.setWindowOpacity(element._original_opacity)
            del self._elements[element_id]
            logger.debug(f"Unregistered element: {element_id}")
    
    # ========================================================================
    # Lifecycle
    # ========================================================================
    
    def enable(self):
        """Enable fullscreen detection and visibility control."""
        if self._enabled:
            return
        
        self._enabled = True
        self._stop_event.clear()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="FullscreenMonitor",
        )
        self._monitor_thread.start()
        
        # Register hotkey if set
        if self._settings.toggle_hotkey:
            self._register_hotkey(self._settings.toggle_hotkey)
        
        logger.info("Fullscreen mode controller enabled")
    
    def disable(self):
        """Disable fullscreen detection."""
        if not self._enabled:
            return
        
        self._enabled = False
        self._stop_event.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        
        # Unregister hotkey
        self._unregister_hotkey()
        
        # Restore all elements
        self._manually_hidden = False
        self._fullscreen_active = False
        self._update_element_visibility()
        
        logger.info("Fullscreen mode controller disabled")
    
    def toggle_visibility(self):
        """Toggle visibility manually (hotkey action)."""
        self._manually_hidden = not self._manually_hidden
        self._update_element_visibility()
        
        logger.info(f"Visibility toggled: {'hidden' if self._manually_hidden else 'visible'}")
        
        # Notify callbacks
        for callback in self._on_visibility_change:
            try:
                callback(self.is_visible)
            except Exception as e:
                logger.error(f"Visibility change callback error: {e}")
    
    def show_all(self):
        """Show all elements (override fullscreen detection)."""
        self._manually_hidden = False
        self._update_element_visibility()
    
    def hide_all(self):
        """Hide all elements."""
        self._manually_hidden = True
        self._update_element_visibility()
    
    # ========================================================================
    # Callbacks
    # ========================================================================
    
    def on_visibility_change(self, callback: Callable[[bool], None]):
        """Register callback for visibility changes."""
        self._on_visibility_change.append(callback)
    
    def on_fullscreen_change(self, callback: Callable[[bool], None]):
        """Register callback for fullscreen detection changes."""
        self._on_fullscreen_change.append(callback)
    
    # ========================================================================
    # Monitoring
    # ========================================================================
    
    def _monitor_loop(self):
        """Main monitoring loop for fullscreen detection."""
        while not self._stop_event.is_set():
            try:
                was_fullscreen = self._fullscreen_active
                self._fullscreen_active = self._detect_fullscreen()
                
                if was_fullscreen != self._fullscreen_active:
                    logger.info(f"Fullscreen {'detected' if self._fullscreen_active else 'ended'}")
                    self._update_element_visibility()
                    
                    # Notify callbacks
                    for callback in self._on_fullscreen_change:
                        try:
                            callback(self._fullscreen_active)
                        except Exception as e:
                            logger.error(f"Fullscreen change callback error: {e}")
                
            except Exception as e:
                logger.error(f"Fullscreen monitor error: {e}")
            
            self._stop_event.wait(self._settings.fullscreen_check_interval)
    
    def _detect_fullscreen(self) -> bool:
        """Detect if a fullscreen application is active."""
        if platform.system() != "Windows":
            return self._detect_fullscreen_unix()
        
        return self._detect_fullscreen_windows()
    
    def _detect_fullscreen_windows(self) -> bool:
        """Detect fullscreen on Windows."""
        try:
            import ctypes
            from ctypes import wintypes
            
            user32 = ctypes.windll.user32
            
            # Get foreground window
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return False
            
            # Get window rect
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            
            # Get work area (screen size minus taskbar)
            # We check against full screen including taskbar for true fullscreen
            screen_width = user32.GetSystemMetrics(0)  # SM_CXSCREEN
            screen_height = user32.GetSystemMetrics(1)  # SM_CYSCREEN
            
            # Also check virtual screen for multi-monitor setups
            virtual_width = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
            virtual_height = user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
            
            window_width = rect.right - rect.left
            window_height = rect.bottom - rect.top
            
            # Check if window covers primary screen
            covers_primary = (
                window_width >= screen_width and 
                window_height >= screen_height
            )
            
            # Check if window is borderless fullscreen
            GWL_STYLE = -16
            style = user32.GetWindowLongW(hwnd, GWL_STYLE)
            WS_POPUP = 0x80000000
            WS_BORDER = 0x00800000
            WS_CAPTION = 0x00C00000
            
            is_borderless = (style & WS_POPUP) and not (style & WS_CAPTION)
            
            return covers_primary or is_borderless
            
        except Exception as e:
            logger.debug(f"Fullscreen detection error: {e}")
            return False
    
    def _detect_fullscreen_unix(self) -> bool:
        """Detect fullscreen on Linux/Mac."""
        # Basic implementation - can be enhanced
        try:
            import subprocess
            
            if platform.system() == "Darwin":  # macOS
                # Use AppleScript to check frontmost app
                script = '''
                tell application "System Events"
                    set frontApp to first application process whose frontmost is true
                    set windowCount to count of windows of frontApp
                    if windowCount > 0 then
                        set frontWindow to first window of frontApp
                        return "fullscreen:" & (value of attribute "AXFullScreen" of frontWindow as string)
                    end if
                end tell
                '''
                result = subprocess.check_output(['osascript', '-e', script], text=True, timeout=2)
                return 'fullscreen:true' in result.lower()
            else:  # Linux
                # Check for _NET_WM_STATE_FULLSCREEN
                result = subprocess.check_output(
                    ['xdotool', 'getactivewindow', 'getwindowname'],
                    text=True,
                    timeout=2
                )
                # Would need additional checks for actual fullscreen state
                return False
                
        except Exception:
            return False
    
    # ========================================================================
    # Visibility Control
    # ========================================================================
    
    def _update_element_visibility(self):
        """Update visibility of all registered elements."""
        target_visible = self.is_visible
        
        for element in self._elements.values():
            self._apply_visibility_to_element(element)
    
    def _apply_visibility_to_element(self, element: RegisteredElement):
        """Apply visibility settings to a single element."""
        should_show = self._should_element_be_visible(element)
        
        if self._settings.fade_enabled:
            self._fade_element(element, 1.0 if should_show else 0.0)
        else:
            self._set_element_visible(element, should_show)
    
    def _should_element_be_visible(self, element: RegisteredElement) -> bool:
        """Determine if an element should be visible based on all settings."""
        # Check manual hide
        if self._manually_hidden:
            return False
        
        # Check fullscreen auto-hide
        if self._fullscreen_active and self._settings.auto_hide_on_fullscreen:
            return False
        
        # Check category visibility
        if not self._settings.category_visible.get(element.category, True):
            return False
        
        # Check monitor restrictions
        if self._settings.allowed_monitors is not None:
            element_monitor = self._get_element_monitor(element)
            if element_monitor not in self._settings.allowed_monitors:
                return False
        
        return True
    
    def _get_element_monitor(self, element: RegisteredElement) -> int:
        """Get which monitor an element is on."""
        if element.monitor is not None:
            return element.monitor
        
        try:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtCore import QPoint
            
            # Get element center position
            widget = element.widget
            if hasattr(widget, 'pos') and hasattr(widget, 'width'):
                center = QPoint(
                    widget.pos().x() + widget.width() // 2,
                    widget.pos().y() + widget.height() // 2
                )
                
                # Find which screen contains this point
                app = QApplication.instance()
                if app:
                    desktop = app.desktop()
                    return desktop.screenNumber(center)
        except Exception:
            pass
        
        return 0  # Default to primary monitor
    
    def _set_element_visible(self, element: RegisteredElement, visible: bool):
        """Set element visibility instantly."""
        widget = element.widget
        
        try:
            if visible:
                if hasattr(widget, 'show'):
                    widget.show()
                if hasattr(widget, 'setWindowOpacity'):
                    widget.setWindowOpacity(element._original_opacity)
            else:
                if hasattr(widget, 'hide'):
                    widget.hide()
        except Exception as e:
            logger.error(f"Failed to set element visibility: {e}")
    
    def _fade_element(self, element: RegisteredElement, target_opacity: float):
        """Fade an element to target opacity."""
        element._target_opacity = target_opacity
        
        # Use Qt timer for smooth animation
        try:
            from PyQt5.QtCore import QTimer
            
            # If already at target, no animation needed
            if abs(element._current_opacity - target_opacity) < 0.01:
                return
            
            # Show widget if fading in
            if target_opacity > 0 and hasattr(element.widget, 'show'):
                element.widget.show()
            
            # Calculate step size for smooth fade
            steps = max(1, self._settings.fade_duration_ms // 16)  # ~60fps
            step_size = (target_opacity - element._current_opacity) / steps
            
            def fade_step():
                try:
                    element._current_opacity += step_size
                    
                    # Clamp to target
                    if step_size > 0:
                        element._current_opacity = min(element._current_opacity, target_opacity)
                    else:
                        element._current_opacity = max(element._current_opacity, target_opacity)
                    
                    # Apply opacity
                    if hasattr(element.widget, 'setWindowOpacity'):
                        element.widget.setWindowOpacity(element._current_opacity)
                    
                    # Check if done
                    if abs(element._current_opacity - target_opacity) < 0.01:
                        # Fade complete
                        if target_opacity == 0 and hasattr(element.widget, 'hide'):
                            element.widget.hide()
                        return  # Don't schedule next step
                    
                    # Schedule next step
                    QTimer.singleShot(16, fade_step)
                    
                except Exception as e:
                    logger.error(f"Fade step error: {e}")
            
            # Start fade
            fade_step()
            
        except ImportError:
            # Qt not available, fall back to instant
            self._set_element_visible(element, target_opacity > 0)
    
    # ========================================================================
    # Hotkey Support
    # ========================================================================
    
    def _register_hotkey(self, hotkey: str):
        """Register a global hotkey."""
        if platform.system() != "Windows":
            logger.warning("Global hotkeys only supported on Windows currently")
            return
        
        try:
            import ctypes
            from ctypes import wintypes
            
            # Parse hotkey string
            modifiers, vk = self._parse_hotkey(hotkey)
            if vk is None:
                logger.error(f"Invalid hotkey: {hotkey}")
                return
            
            # Register hotkey in a separate thread
            def hotkey_listener():
                try:
                    user32 = ctypes.windll.user32
                    
                    # Register hotkey
                    hotkey_id = 1  # Arbitrary ID
                    if not user32.RegisterHotKey(None, hotkey_id, modifiers, vk):
                        logger.error(f"Failed to register hotkey: {hotkey}")
                        return
                    
                    self._hotkey_id = hotkey_id
                    self._hotkey_registered = True
                    logger.info(f"Registered global hotkey: {hotkey}")
                    
                    # Message loop
                    msg = wintypes.MSG()
                    while not self._stop_event.is_set():
                        if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):  # PM_REMOVE
                            if msg.message == 0x0312:  # WM_HOTKEY
                                self.toggle_visibility()
                        time.sleep(0.05)
                    
                    # Unregister on exit
                    user32.UnregisterHotKey(None, hotkey_id)
                    
                except Exception as e:
                    logger.error(f"Hotkey listener error: {e}")
            
            # Start listener thread
            thread = threading.Thread(target=hotkey_listener, daemon=True, name="HotkeyListener")
            thread.start()
            
        except Exception as e:
            logger.error(f"Failed to register hotkey: {e}")
    
    def _unregister_hotkey(self):
        """Unregister the global hotkey."""
        self._hotkey_registered = False
        # The listener thread will clean up when stop_event is set
    
    def _parse_hotkey(self, hotkey: str) -> Tuple[int, Optional[int]]:
        """Parse a hotkey string into modifiers and virtual key code."""
        MOD_ALT = 0x0001
        MOD_CONTROL = 0x0002
        MOD_SHIFT = 0x0004
        MOD_WIN = 0x0008
        
        # Virtual key codes
        VK_MAP = {
            'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45,
            'f': 0x46, 'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A,
            'k': 0x4B, 'l': 0x4C, 'm': 0x4D, 'n': 0x4E, 'o': 0x4F,
            'p': 0x50, 'q': 0x51, 'r': 0x52, 's': 0x53, 't': 0x54,
            'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58, 'y': 0x59,
            'z': 0x5A,
            '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
            '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
            'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73,
            'f5': 0x74, 'f6': 0x75, 'f7': 0x76, 'f8': 0x77,
            'f9': 0x78, 'f10': 0x79, 'f11': 0x7A, 'f12': 0x7B,
            'space': 0x20, 'enter': 0x0D, 'tab': 0x09, 'escape': 0x1B,
        }
        
        modifiers = 0
        vk = None
        
        parts = hotkey.lower().replace(' ', '').split('+')
        
        for part in parts:
            if part in ('ctrl', 'control'):
                modifiers |= MOD_CONTROL
            elif part == 'alt':
                modifiers |= MOD_ALT
            elif part == 'shift':
                modifiers |= MOD_SHIFT
            elif part in ('win', 'windows', 'super'):
                modifiers |= MOD_WIN
            elif part in VK_MAP:
                vk = VK_MAP[part]
            else:
                logger.warning(f"Unknown hotkey part: {part}")
        
        return modifiers, vk
    
    # ========================================================================
    # Monitor Utilities
    # ========================================================================
    
    def get_monitors(self) -> List[Dict]:
        """Get list of available monitors with their info."""
        # Cache for 5 seconds
        if time.time() - self._monitor_cache_time < 5 and self._monitor_cache:
            return self._monitor_cache
        
        monitors = []
        
        try:
            from PyQt5.QtWidgets import QApplication
            
            app = QApplication.instance()
            if app:
                desktop = app.desktop()
                for i in range(desktop.screenCount()):
                    geo = desktop.screenGeometry(i)
                    monitors.append({
                        'index': i,
                        'name': f"Monitor {i + 1}",
                        'width': geo.width(),
                        'height': geo.height(),
                        'x': geo.x(),
                        'y': geo.y(),
                        'is_primary': i == desktop.primaryScreen(),
                    })
        except Exception as e:
            logger.error(f"Failed to get monitors: {e}")
            monitors = [{'index': 0, 'name': 'Primary', 'is_primary': True}]
        
        self._monitor_cache = monitors
        self._monitor_cache_time = time.time()
        
        return monitors
    
    # ========================================================================
    # Persistence
    # ========================================================================
    
    def save_settings(self, path: Optional[Path] = None):
        """Save settings to file."""
        if path is None:
            path = Path("data/fullscreen_settings.json")
        
        data = {
            "category_visible": self._settings.category_visible,
            "allowed_monitors": self._settings.allowed_monitors,
            "fade_enabled": self._settings.fade_enabled,
            "fade_duration_ms": self._settings.fade_duration_ms,
            "auto_hide_on_fullscreen": self._settings.auto_hide_on_fullscreen,
            "toggle_hotkey": self._settings.toggle_hotkey,
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved fullscreen settings to {path}")
    
    def load_settings(self, path: Optional[Path] = None):
        """Load settings from file."""
        if path is None:
            path = Path("data/fullscreen_settings.json")
        
        if not path.exists():
            return
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            if "category_visible" in data:
                self._settings.category_visible.update(data["category_visible"])
            if "allowed_monitors" in data:
                self._settings.allowed_monitors = data["allowed_monitors"]
            if "fade_enabled" in data:
                self._settings.fade_enabled = data["fade_enabled"]
            if "fade_duration_ms" in data:
                self._settings.fade_duration_ms = data["fade_duration_ms"]
            if "auto_hide_on_fullscreen" in data:
                self._settings.auto_hide_on_fullscreen = data["auto_hide_on_fullscreen"]
            if "toggle_hotkey" in data:
                self._settings.toggle_hotkey = data["toggle_hotkey"]
            
            logger.info(f"Loaded fullscreen settings from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load fullscreen settings: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "enabled": self._enabled,
            "visible": self.is_visible,
            "fullscreen_detected": self._fullscreen_active,
            "manually_hidden": self._manually_hidden,
            "category_visible": self._settings.category_visible.copy(),
            "allowed_monitors": self._settings.allowed_monitors,
            "fade_enabled": self._settings.fade_enabled,
            "toggle_hotkey": self._settings.toggle_hotkey,
            "registered_elements": list(self._elements.keys()),
        }


# Global instance
_fullscreen_controller: Optional[FullscreenController] = None


def get_fullscreen_controller() -> FullscreenController:
    """Get or create the global fullscreen controller instance."""
    global _fullscreen_controller
    if _fullscreen_controller is None:
        _fullscreen_controller = FullscreenController()
    return _fullscreen_controller


__all__ = [
    'FullscreenController',
    'ElementCategory',
    'VisibilitySettings',
    'get_fullscreen_controller',
]
