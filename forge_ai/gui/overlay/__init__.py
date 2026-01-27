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

from .overlay_window import AIOverlay
from .overlay_modes import OverlayMode, OverlayPosition
from .overlay_themes import OverlayTheme, OVERLAY_THEMES
from .overlay_chat import OverlayChatBridge

__all__ = [
    'AIOverlay',
    'OverlayMode',
    'OverlayPosition',
    'OverlayTheme',
    'OVERLAY_THEMES',
    'OverlayChatBridge',
]
