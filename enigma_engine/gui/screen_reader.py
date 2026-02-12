"""
Screen Reader Support for Enigma AI Engine

Full accessibility support for blind and low-vision users.

Features:
- Screen reader announcements
- ARIA-compatible output
- Keyboard navigation focus
- High contrast mode
- Audio cues

Usage:
    from enigma_engine.gui.screen_reader import ScreenReaderSupport
    
    sr = ScreenReaderSupport()
    sr.announce("New message received")
    
    # Enable for PyQt widget
    sr.make_accessible(widget)
"""

import logging
import platform
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AnnouncementPriority(Enum):
    """Priority levels for announcements."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AccessibleElement:
    """Represents an accessible UI element."""
    name: str
    role: str
    description: str = ""
    value: str = ""
    shortcut: str = ""
    parent: Optional[str] = None


class ScreenReaderBackend:
    """Base class for screen reader backends."""
    
    def announce(self, text: str, priority: AnnouncementPriority = AnnouncementPriority.NORMAL):
        """Announce text to screen reader."""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if backend is available."""
        return False


class WindowsNarrator(ScreenReaderBackend):
    """Windows Narrator/NVDA/JAWS support."""
    
    def __init__(self):
        self._tolk = None
        self._accessible = None
        
        # Try to initialize Tolk (screen reader library)
        try:
            import tolk
            tolk.load()
            if tolk.is_loaded():
                self._tolk = tolk
                logger.info(f"Tolk initialized with {tolk.detect_screen_reader()}")
        except ImportError:
            pass  # Intentionally silent
        
        # Try Windows accessibility API
        if self._tolk is None:
            try:
                import win32com.client
                self._accessible = win32com.client.Dispatch("SAPI.SpVoice")
            except Exception:
                pass  # Intentionally silent
    
    def announce(self, text: str, priority: AnnouncementPriority = AnnouncementPriority.NORMAL):
        if self._tolk:
            interrupt = priority in (AnnouncementPriority.HIGH, AnnouncementPriority.CRITICAL)
            self._tolk.speak(text, interrupt)
        elif self._accessible:
            self._accessible.Speak(text)
    
    def is_available(self) -> bool:
        return self._tolk is not None or self._accessible is not None


class MacVoiceOver(ScreenReaderBackend):
    """macOS VoiceOver support."""
    
    def __init__(self):
        self._available = platform.system() == "Darwin"
    
    def announce(self, text: str, priority: AnnouncementPriority = AnnouncementPriority.NORMAL):
        if not self._available:
            return
        
        import subprocess
        
        # Use osascript to announce via VoiceOver
        script = f'tell application \"VoiceOver\" to output \"{text}\"'
        
        try:
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5
            )
        except Exception:
            # Fallback to say command
            subprocess.run(["say", text], capture_output=True, timeout=30)
    
    def is_available(self) -> bool:
        return self._available


class LinuxOrca(ScreenReaderBackend):
    """Linux Orca/speechd support."""
    
    def __init__(self):
        self._spd = None
        
        try:
            import speechd
            self._spd = speechd.SSIPClient("enigma")
        except ImportError:
            pass  # Intentionally silent
    
    def announce(self, text: str, priority: AnnouncementPriority = AnnouncementPriority.NORMAL):
        if self._spd:
            priority_map = {
                AnnouncementPriority.LOW: 0,
                AnnouncementPriority.NORMAL: 50,
                AnnouncementPriority.HIGH: 75,
                AnnouncementPriority.CRITICAL: 100
            }
            self._spd.speak(text, priority=priority_map.get(priority, 50))
    
    def is_available(self) -> bool:
        return self._spd is not None


class FallbackTTS(ScreenReaderBackend):
    """Fallback TTS when no screen reader is available."""
    
    def __init__(self):
        self._engine = None
        
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
        except Exception:
            pass  # Intentionally silent
    
    def announce(self, text: str, priority: AnnouncementPriority = AnnouncementPriority.NORMAL):
        if self._engine:
            self._engine.say(text)
            self._engine.runAndWait()
    
    def is_available(self) -> bool:
        return self._engine is not None


class ScreenReaderSupport:
    """Main screen reader support class."""
    
    def __init__(self):
        """Initialize screen reader support."""
        self._backend = self._detect_backend()
        self._enabled = True
        
        # Announcement queue
        self._queue: List[tuple] = []
        self._last_announcement = ""
        self._last_time = 0
        
        # Element registry
        self._elements: Dict[str, AccessibleElement] = {}
        
        # Settings
        self._verbosity = "normal"  # low, normal, high
        self._announce_typing = False
        self._announce_focus = True
        
        logger.info(f"Screen reader support initialized (backend: {type(self._backend).__name__})")
    
    def _detect_backend(self) -> ScreenReaderBackend:
        """Detect and initialize appropriate backend."""
        system = platform.system()
        
        if system == "Windows":
            backend = WindowsNarrator()
            if backend.is_available():
                return backend
        elif system == "Darwin":
            backend = MacVoiceOver()
            if backend.is_available():
                return backend
        elif system == "Linux":
            backend = LinuxOrca()
            if backend.is_available():
                return backend
        
        # Fallback
        return FallbackTTS()
    
    @property
    def enabled(self) -> bool:
        """Check if screen reader support is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        """Enable/disable screen reader support."""
        self._enabled = value
    
    def is_available(self) -> bool:
        """Check if a screen reader is available."""
        return self._backend.is_available()
    
    def announce(
        self,
        text: str,
        priority: AnnouncementPriority = AnnouncementPriority.NORMAL,
        interrupt: bool = False
    ):
        """
        Announce text to screen reader.
        
        Args:
            text: Text to announce
            priority: Announcement priority
            interrupt: Whether to interrupt current speech
        """
        if not self._enabled:
            return
        
        # Avoid duplicate announcements
        current_time = time.time()
        if text == self._last_announcement and current_time - self._last_time < 1.0:
            return
        
        self._last_announcement = text
        self._last_time = current_time
        
        # If interrupt, use high priority
        if interrupt:
            priority = AnnouncementPriority.CRITICAL
        
        self._backend.announce(text, priority)
    
    def announce_message(self, sender: str, message: str):
        """Announce a chat message."""
        self.announce(f"{sender} says: {message}")
    
    def announce_status(self, status: str):
        """Announce status change."""
        self.announce(f"Status: {status}", AnnouncementPriority.LOW)
    
    def announce_error(self, error: str):
        """Announce an error."""
        self.announce(f"Error: {error}", AnnouncementPriority.HIGH)
    
    def announce_focus(self, element_name: str, element_type: str = ""):
        """
        Announce focus change.
        
        Args:
            element_name: Name of focused element
            element_type: Type of element (button, text field, etc.)
        """
        if not self._announce_focus:
            return
        
        if element_type:
            self.announce(f"{element_name}, {element_type}")
        else:
            self.announce(element_name)
    
    def register_element(self, element: AccessibleElement):
        """
        Register an accessible element.
        
        Args:
            element: Element information
        """
        self._elements[element.name] = element
    
    def get_element_description(self, name: str) -> str:
        """Get full description of an element."""
        element = self._elements.get(name)
        
        if not element:
            return name
        
        parts = [element.name]
        
        if element.role:
            parts.append(element.role)
        
        if element.description:
            parts.append(element.description)
        
        if element.shortcut:
            parts.append(f"Shortcut: {element.shortcut}")
        
        return ", ".join(parts)
    
    def make_accessible(
        self,
        widget: Any,
        name: str,
        role: str = "",
        description: str = ""
    ):
        """
        Make a PyQt/PySide widget accessible.
        
        Args:
            widget: Qt widget
            name: Accessible name
            role: Widget role
            description: Widget description
        """
        try:
            from PyQt5.QtWidgets import QWidget
            from PyQt5.QtCore import Qt
            
            if isinstance(widget, QWidget):
                widget.setAccessibleName(name)
                
                if description:
                    widget.setAccessibleDescription(description)
                
                # Make focusable
                widget.setFocusPolicy(Qt.StrongFocus)
                
                # Connect focus events
                widget.focusInEvent = lambda e, w=widget, n=name, r=role: self._on_focus(e, w, n, r)
                
                # Register
                self.register_element(AccessibleElement(
                    name=name,
                    role=role,
                    description=description
                ))
                
        except ImportError:
            pass  # Intentionally silent
    
    def _on_focus(self, event: Any, widget: Any, name: str, role: str):
        """Handle focus event."""
        self.announce_focus(name, role)
        
        # Call original handler
        if hasattr(widget.__class__.__bases__[0], 'focusInEvent'):
            widget.__class__.__bases__[0].focusInEvent(widget, event)
    
    def set_verbosity(self, level: str):
        """
        Set announcement verbosity.
        
        Args:
            level: "low", "normal", or "high"
        """
        self._verbosity = level
    
    def get_keyboard_shortcuts(self) -> Dict[str, str]:
        """Get list of keyboard shortcuts."""
        shortcuts = {
            "Ctrl+N": "New conversation",
            "Ctrl+S": "Save conversation",
            "Ctrl+O": "Open conversation",
            "Ctrl+Enter": "Send message",
            "Ctrl+L": "Clear conversation",
            "Ctrl+T": "Toggle settings",
            "F1": "Help",
            "F6": "Navigate between panels",
            "Tab": "Next element",
            "Shift+Tab": "Previous element",
            "Escape": "Cancel/Close dialog"
        }
        
        # Add registered shortcuts
        for element in self._elements.values():
            if element.shortcut:
                shortcuts[element.shortcut] = element.description or element.name
        
        return shortcuts
    
    def announce_shortcuts(self):
        """Announce available keyboard shortcuts."""
        shortcuts = self.get_keyboard_shortcuts()
        text = "Keyboard shortcuts: " + " ".join(f"{k}: {v}," for k, v in list(shortcuts.items())[:5])
        self.announce(text)


class AccessibleWindow:
    """Mixin for making windows accessible."""
    
    def __init__(self):
        self._screen_reader = get_screen_reader()
    
    def setup_accessibility(self):
        """Setup accessibility features for window."""
        # This would be called by the window class
    
    def announce(self, text: str):
        """Announce text."""
        self._screen_reader.announce(text)


# Global instance
_screen_reader: Optional[ScreenReaderSupport] = None


def get_screen_reader() -> ScreenReaderSupport:
    """Get or create global screen reader support."""
    global _screen_reader
    if _screen_reader is None:
        _screen_reader = ScreenReaderSupport()
    return _screen_reader


def announce(text: str, priority: str = "normal"):
    """Quick announce function."""
    sr = get_screen_reader()
    sr.announce(text, AnnouncementPriority(priority))
