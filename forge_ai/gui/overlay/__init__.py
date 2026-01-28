"""
Overlay System for ForgeAI - Transparent always-on-top AI interface.

This module provides a transparent overlay window that can be used for
AI interaction while gaming or using other applications.

Main Components:
- AIOverlay: Main overlay window with transparency and always-on-top
- OverlayMode: Different display modes (MINIMAL, COMPACT, FULL, HIDDEN)
- OverlayTheme: Visual customization themes
- OverlayChatBridge: Integration with chat system
"""

# Import base classes that don't require PyQt5
from .overlay_modes import OverlayMode, OverlayPosition
from .overlay_themes import OverlayTheme, OVERLAY_THEMES

# Try to import PyQt5-dependent classes
try:
    from .overlay_window import AIOverlay
    from .overlay_chat import OverlayChatBridge
    HAS_PYQT = True
except ImportError:
    AIOverlay = None
    OverlayChatBridge = None
    HAS_PYQT = False

__all__ = [
    'OverlayMode',
    'OverlayPosition',
    'OverlayTheme',
    'OVERLAY_THEMES',
    'HAS_PYQT',
]

# Only export PyQt5-dependent classes if available
if HAS_PYQT:
    __all__.extend(['AIOverlay', 'OverlayChatBridge'])
