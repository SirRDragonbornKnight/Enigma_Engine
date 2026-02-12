"""
Advanced Gesture Detection System
==================================

Detects complex input gestures combining mouse and keyboard actions:
- Mouse click + modifier keys (e.g., Ctrl+Right-Click)
- Scroll wheel + modifiers (e.g., Shift+Scroll)
- Drag patterns (e.g., Alt+Drag-Up)
- Mouse button sequences (e.g., Double-Right-Click)
- Combined gestures (e.g., Ctrl+Shift+Middle-Click)

Usage:
    from enigma_engine.core.gesture_manager import GestureManager, Gesture
    
    manager = GestureManager()
    
    # Register a gesture
    manager.register(
        gesture=Gesture(
            modifiers=["Ctrl", "Shift"],
            mouse_action="right_click"
        ),
        callback=show_context_menu,
        name="context_menu"
    )
    
    # Register scroll gesture
    manager.register(
        gesture=Gesture(
            modifiers=["Alt"],
            mouse_action="scroll_up"
        ),
        callback=zoom_in,
        name="zoom_in"
    )
    
    manager.start()

Dependencies:
    pip install pynput  # Cross-platform mouse/keyboard hooks
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Check if pynput is available
try:
    from pynput import mouse, keyboard
    from pynput.keyboard import Key
    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False
    logger.warning("pynput not installed - gesture detection unavailable")


class MouseAction(Enum):
    """Mouse actions that can be part of a gesture."""
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    MIDDLE_CLICK = "middle_click"
    DOUBLE_CLICK = "double_click"
    DOUBLE_RIGHT = "double_right"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    DRAG_UP = "drag_up"
    DRAG_DOWN = "drag_down"
    DRAG_LEFT = "drag_left"
    DRAG_RIGHT = "drag_right"
    HOVER_CORNER_TL = "hover_corner_tl"  # Top-left screen corner
    HOVER_CORNER_TR = "hover_corner_tr"  # Top-right screen corner
    HOVER_CORNER_BL = "hover_corner_bl"  # Bottom-left screen corner
    HOVER_CORNER_BR = "hover_corner_br"  # Bottom-right screen corner


class Modifier(Enum):
    """Keyboard modifiers for gestures."""
    CTRL = "ctrl"
    ALT = "alt"
    SHIFT = "shift"
    WIN = "win"  # Windows key / Command key on Mac
    

@dataclass
class Gesture:
    """
    Defines a gesture pattern to match.
    
    A gesture consists of:
    - Zero or more keyboard modifiers (Ctrl, Shift, Alt, Win)
    - A mouse action (click, scroll, drag, hover)
    - Optional parameters (e.g., drag distance threshold)
    """
    mouse_action: str                           # MouseAction value
    modifiers: List[str] = field(default_factory=list)  # Modifier values
    drag_threshold: int = 50                    # Pixels for drag detection
    double_click_window: float = 0.5            # Seconds for double-click
    corner_size: int = 10                       # Pixels from corner for hover
    
    def __post_init__(self) -> None:
        # Normalize modifiers to lowercase
        self.modifiers = [m.lower() for m in self.modifiers]
        self.mouse_action = self.mouse_action.lower()
    
    def matches_modifiers(self, active_modifiers: Set[str]) -> bool:
        """Check if active modifiers match required modifiers."""
        required = set(self.modifiers)
        return required == active_modifiers
    
    @property
    def display_name(self) -> str:
        """Human-readable gesture name."""
        parts = [m.capitalize() for m in self.modifiers]
        
        action_names = {
            "left_click": "Click",
            "right_click": "Right-Click",
            "middle_click": "Middle-Click",
            "double_click": "Double-Click",
            "double_right": "Double-Right-Click",
            "scroll_up": "Scroll Up",
            "scroll_down": "Scroll Down",
            "drag_up": "Drag Up",
            "drag_down": "Drag Down",
            "drag_left": "Drag Left",
            "drag_right": "Drag Right",
            "hover_corner_tl": "Hover Top-Left",
            "hover_corner_tr": "Hover Top-Right",
            "hover_corner_bl": "Hover Bottom-Left",
            "hover_corner_br": "Hover Bottom-Right",
        }
        
        parts.append(action_names.get(self.mouse_action, self.mouse_action))
        return "+".join(parts)
    
    def to_dict(self) -> dict:
        return {
            "mouse_action": self.mouse_action,
            "modifiers": self.modifiers,
            "drag_threshold": self.drag_threshold,
            "double_click_window": self.double_click_window,
            "corner_size": self.corner_size,
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'Gesture':
        return Gesture(
            mouse_action=data["mouse_action"],
            modifiers=data.get("modifiers", []),
            drag_threshold=data.get("drag_threshold", 50),
            double_click_window=data.get("double_click_window", 0.5),
            corner_size=data.get("corner_size", 10),
        )


@dataclass
class GestureBinding:
    """A gesture bound to a callback."""
    name: str
    gesture: Gesture
    callback: Callable[[], None]
    enabled: bool = True


class GestureManager:
    """
    Manages gesture detection and callbacks.
    
    Listens for mouse and keyboard events globally and triggers callbacks
    when registered gesture patterns are matched.
    """
    
    def __init__(self) -> None:
        """Initialize the gesture manager."""
        self._bindings: Dict[str, GestureBinding] = {}
        self._running = False
        
        # Listeners
        self._mouse_listener: Optional[Any] = None
        self._keyboard_listener: Optional[Any] = None
        
        # State tracking
        self._active_modifiers: Set[str] = set()
        self._last_click_time: Dict[str, float] = {}  # button -> time
        self._last_click_pos: Optional[Tuple[int, int]] = None
        self._drag_start: Optional[Tuple[int, int]] = None
        self._is_dragging = False
        self._mouse_pos: Tuple[int, int] = (0, 0)
        self._screen_size: Tuple[int, int] = (1920, 1080)  # Default, updated on start
        
        # Callbacks
        self.on_gesture_detected: Optional[Callable[[str, Gesture], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
    
    def register(
        self,
        gesture: Gesture,
        callback: Callable[[], None],
        name: str
    ) -> bool:
        """
        Register a gesture with callback.
        
        Args:
            gesture: The gesture pattern to match
            callback: Function to call when gesture detected
            name: Unique name for this gesture binding
            
        Returns:
            True if registered successfully
        """
        if name in self._bindings:
            logger.warning(f"Gesture '{name}' already registered, replacing")
        
        self._bindings[name] = GestureBinding(
            name=name,
            gesture=gesture,
            callback=callback,
            enabled=True
        )
        
        logger.info(f"Registered gesture: {name} = {gesture.display_name}")
        return True
    
    def unregister(self, name: str) -> bool:
        """Unregister a gesture by name."""
        if name in self._bindings:
            del self._bindings[name]
            logger.info(f"Unregistered gesture: {name}")
            return True
        return False
    
    def enable(self, name: str) -> bool:
        """Enable a gesture binding."""
        if name in self._bindings:
            self._bindings[name].enabled = True
            return True
        return False
    
    def disable(self, name: str) -> bool:
        """Disable a gesture binding without unregistering."""
        if name in self._bindings:
            self._bindings[name].enabled = False
            return True
        return False
    
    def start(self) -> bool:
        """Start listening for gestures."""
        if not _PYNPUT_AVAILABLE:
            logger.error("pynput not available, cannot start gesture detection")
            if self.on_error:
                self.on_error("pynput library not installed")
            return False
        
        if self._running:
            return True
        
        self._running = True
        
        # Get screen size
        try:
            import ctypes
            user32 = ctypes.windll.user32
            self._screen_size = (
                user32.GetSystemMetrics(0),
                user32.GetSystemMetrics(1)
            )
        except Exception:
            pass  # Use default
        
        # Start mouse listener
        self._mouse_listener = mouse.Listener(
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll,
            on_move=self._on_mouse_move
        )
        self._mouse_listener.start()
        
        # Start keyboard listener
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self._keyboard_listener.start()
        
        logger.info("Gesture detection started")
        return True
    
    def stop(self) -> None:
        """Stop listening for gestures."""
        self._running = False
        
        if self._mouse_listener:
            self._mouse_listener.stop()
            self._mouse_listener = None
        
        if self._keyboard_listener:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        
        self._active_modifiers.clear()
        logger.info("Gesture detection stopped")
    
    def _on_key_press(self, key) -> None:
        """Handle key press - track modifiers."""
        if not _PYNPUT_AVAILABLE:
            return
        
        try:
            if key == Key.ctrl_l or key == Key.ctrl_r:
                self._active_modifiers.add("ctrl")
            elif key == Key.alt_l or key == Key.alt_r or key == Key.alt_gr:
                self._active_modifiers.add("alt")
            elif key == Key.shift_l or key == Key.shift_r or key == Key.shift:
                self._active_modifiers.add("shift")
            elif key == Key.cmd or key == Key.cmd_r:
                self._active_modifiers.add("win")
        except Exception as e:
            logger.debug(f"Key press handling error: {e}")
    
    def _on_key_release(self, key) -> None:
        """Handle key release - track modifiers."""
        if not _PYNPUT_AVAILABLE:
            return
        
        try:
            if key == Key.ctrl_l or key == Key.ctrl_r:
                self._active_modifiers.discard("ctrl")
            elif key == Key.alt_l or key == Key.alt_r or key == Key.alt_gr:
                self._active_modifiers.discard("alt")
            elif key == Key.shift_l or key == Key.shift_r or key == Key.shift:
                self._active_modifiers.discard("shift")
            elif key == Key.cmd or key == Key.cmd_r:
                self._active_modifiers.discard("win")
        except Exception as e:
            logger.debug(f"Key release handling error: {e}")
    
    def _on_mouse_click(self, x: int, y: int, button, pressed: bool) -> None:
        """Handle mouse click events."""
        if not pressed:
            # Button released - check for drag end
            if self._is_dragging:
                self._handle_drag_end(x, y)
            return
        
        # Button pressed
        self._mouse_pos = (x, y)
        
        # Track for drag detection
        self._drag_start = (x, y)
        self._is_dragging = False
        
        # Determine button name
        button_name = str(button).split(".")[-1].lower()  # "left", "right", "middle"
        
        # Check for double-click
        current_time = time.time()
        is_double = False
        
        if button_name in self._last_click_time:
            time_diff = current_time - self._last_click_time[button_name]
            if time_diff < 0.5:  # Double-click window
                is_double = True
        
        self._last_click_time[button_name] = current_time
        self._last_click_pos = (x, y)
        
        # Determine action
        if is_double:
            if button_name == "left":
                action = "double_click"
            elif button_name == "right":
                action = "double_right"
            else:
                action = f"double_{button_name}"
        else:
            if button_name == "left":
                action = "left_click"
            elif button_name == "right":
                action = "right_click"
            elif button_name == "middle":
                action = "middle_click"
            else:
                action = f"{button_name}_click"
        
        # Check for matching gestures
        self._check_gestures(action)
    
    def _on_mouse_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        """Handle mouse scroll events."""
        self._mouse_pos = (x, y)
        
        if dy > 0:
            action = "scroll_up"
        elif dy < 0:
            action = "scroll_down"
        else:
            return
        
        self._check_gestures(action)
    
    def _on_mouse_move(self, x: int, y: int) -> None:
        """Handle mouse move events."""
        prev_pos = self._mouse_pos
        self._mouse_pos = (x, y)
        
        # Check drag
        if self._drag_start and not self._is_dragging:
            dx = x - self._drag_start[0]
            dy = y - self._drag_start[1]
            
            # Check if moved enough to be a drag
            for binding in self._bindings.values():
                if not binding.enabled:
                    continue
                
                threshold = binding.gesture.drag_threshold
                
                if abs(dx) > threshold or abs(dy) > threshold:
                    self._is_dragging = True
                    break
        
        # Check corner hover
        for binding in self._bindings.values():
            if not binding.enabled:
                continue
            
            gesture = binding.gesture
            action = gesture.mouse_action
            
            if action.startswith("hover_corner_"):
                corner_size = gesture.corner_size
                screen_w, screen_h = self._screen_size
                
                in_corner = False
                if action == "hover_corner_tl":
                    in_corner = x < corner_size and y < corner_size
                elif action == "hover_corner_tr":
                    in_corner = x > screen_w - corner_size and y < corner_size
                elif action == "hover_corner_bl":
                    in_corner = x < corner_size and y > screen_h - corner_size
                elif action == "hover_corner_br":
                    in_corner = x > screen_w - corner_size and y > screen_h - corner_size
                
                if in_corner and gesture.matches_modifiers(self._active_modifiers):
                    self._trigger_gesture(binding)
    
    def _handle_drag_end(self, x: int, y: int) -> None:
        """Handle the end of a drag gesture."""
        if not self._drag_start:
            self._is_dragging = False
            return
        
        dx = x - self._drag_start[0]
        dy = y - self._drag_start[1]
        
        # Determine drag direction
        if abs(dx) > abs(dy):
            # Horizontal drag
            if dx > 0:
                action = "drag_right"
            else:
                action = "drag_left"
        else:
            # Vertical drag
            if dy > 0:
                action = "drag_down"
            else:
                action = "drag_up"
        
        self._check_gestures(action)
        
        self._drag_start = None
        self._is_dragging = False
    
    def _check_gestures(self, action: str) -> None:
        """Check if any gestures match the current action and modifiers."""
        for binding in self._bindings.values():
            if not binding.enabled:
                continue
            
            gesture = binding.gesture
            
            if gesture.mouse_action == action:
                if gesture.matches_modifiers(self._active_modifiers):
                    self._trigger_gesture(binding)
    
    def _trigger_gesture(self, binding: GestureBinding) -> None:
        """Trigger a gesture callback."""
        logger.debug(f"Gesture detected: {binding.name}")
        
        try:
            binding.callback()
        except Exception as e:
            logger.error(f"Gesture callback error for '{binding.name}': {e}")
        
        if self.on_gesture_detected:
            try:
                self.on_gesture_detected(binding.name, binding.gesture)
            except Exception as e:
                logger.error(f"on_gesture_detected callback error: {e}")
    
    @property
    def is_running(self) -> bool:
        """Check if gesture detection is active."""
        return self._running
    
    @property
    def registered_gestures(self) -> List[str]:
        """Get list of registered gesture names."""
        return list(self._bindings.keys())
    
    def get_gesture(self, name: str) -> Optional[Gesture]:
        """Get a gesture by name."""
        binding = self._bindings.get(name)
        return binding.gesture if binding else None
    
    def list_bindings(self) -> List[Dict[str, Any]]:
        """Get all bindings as dicts."""
        return [
            {
                "name": b.name,
                "gesture": b.gesture.display_name,
                "enabled": b.enabled,
            }
            for b in self._bindings.values()
        ]


# Default gestures for common actions
DEFAULT_GESTURES = {
    "quick_screenshot": Gesture(
        modifiers=["ctrl", "shift"],
        mouse_action="right_click"
    ),
    "context_help": Gesture(
        modifiers=["alt"],
        mouse_action="middle_click"
    ),
    "zoom_in": Gesture(
        modifiers=["ctrl"],
        mouse_action="scroll_up"
    ),
    "zoom_out": Gesture(
        modifiers=["ctrl"],
        mouse_action="scroll_down"
    ),
}


# Singleton instance
_gesture_manager: Optional[GestureManager] = None


def get_gesture_manager() -> GestureManager:
    """Get the global gesture manager instance."""
    global _gesture_manager
    if _gesture_manager is None:
        _gesture_manager = GestureManager()
    return _gesture_manager


def register_gesture(
    name: str,
    mouse_action: str,
    modifiers: List[str] = None,
    callback: Callable[[], None] = None
) -> bool:
    """
    Register a gesture with the global manager.
    
    Convenience function for quick gesture registration.
    
    Args:
        name: Unique name for the gesture
        mouse_action: Mouse action (left_click, right_click, scroll_up, etc.)
        modifiers: Keyboard modifiers (ctrl, shift, alt, win)
        callback: Function to call when gesture detected
        
    Returns:
        True if registered successfully
    """
    manager = get_gesture_manager()
    gesture = Gesture(
        mouse_action=mouse_action,
        modifiers=modifiers or []
    )
    return manager.register(gesture=gesture, callback=callback or (lambda: None), name=name)


def start_gesture_detection() -> bool:
    """Start the global gesture manager."""
    return get_gesture_manager().start()


def stop_gesture_detection() -> None:
    """Stop the global gesture manager."""
    get_gesture_manager().stop()


__all__ = [
    "GestureManager",
    "Gesture",
    "GestureBinding",
    "MouseAction",
    "Modifier",
    "DEFAULT_GESTURES",
    "get_gesture_manager",
    "register_gesture",
    "start_gesture_detection",
    "stop_gesture_detection",
]
