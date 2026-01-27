"""
Overlay Modes - Different display modes for the AI overlay.

Defines the different modes the overlay can be in and the positions it can occupy.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class OverlayMode(Enum):
    """Display modes for the overlay."""
    HIDDEN = "hidden"      # Not visible
    MINIMAL = "minimal"    # Avatar + 1 line response
    COMPACT = "compact"    # Avatar + response + input
    FULL = "full"          # Complete interface


class OverlayPosition(Enum):
    """Position presets for the overlay."""
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"
    CUSTOM = "custom"           # User-defined x, y
    FOLLOW_CURSOR = "follow_cursor"


@dataclass
class MinimalOverlay:
    """
    Minimal mode layout - Just the essentials.
    
    Layout:
    ┌─────────────────────┐
    │ 🤖 AI response here │
    └─────────────────────┘
    """
    show_avatar: bool = True
    show_name: bool = False
    max_response_lines: int = 1
    width: int = 300
    height: int = 60


@dataclass
class CompactOverlay:
    """
    Compact mode layout - Response and input.
    
    Layout:
    ┌─────────────────────────┐
    │ 🤖 AI response here     │
    │ ┌─────────────────────┐ │
    │ │ Type here...        │ │
    │ └─────────────────────┘ │
    └─────────────────────────┘
    """
    show_avatar: bool = True
    show_name: bool = True
    show_input: bool = True
    max_response_lines: int = 3
    width: int = 350
    height: int = 150


@dataclass
class FullOverlay:
    """
    Full mode layout - Complete chat interface.
    
    Layout:
    ┌───────────────────────────┐
    │ 🤖 AI Name         [─][×] │
    ├───────────────────────────┤
    │ Chat history              │
    │ ...                       │
    │ ...                       │
    ├───────────────────────────┤
    │ [Type here...      ] [➤]  │
    └───────────────────────────┘
    """
    show_avatar: bool = True
    show_name: bool = True
    show_input: bool = True
    show_history: bool = True
    show_controls: bool = True
    width: int = 450
    height: int = 400


@dataclass
class OverlaySettings:
    """Settings for overlay behavior and appearance."""
    mode: OverlayMode = OverlayMode.COMPACT
    position: OverlayPosition = OverlayPosition.TOP_RIGHT
    custom_x: Optional[int] = None
    custom_y: Optional[int] = None
    opacity: float = 0.9
    click_through: bool = False
    always_on_top: bool = True
    theme_name: str = "gaming"
    hotkey: str = "Ctrl+Shift+A"
    remember_position: bool = True
    show_on_startup: bool = False
