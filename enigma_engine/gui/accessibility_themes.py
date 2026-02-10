"""
Accessibility Themes for Enigma AI Engine

High contrast and accessibility-focused themes for the GUI.

Features:
- High contrast themes (light and dark)
- Color blind friendly palettes
- Large text options
- Dyslexia-friendly fonts
- Reduced motion mode
- Screen reader support helpers

Usage:
    from enigma_engine.gui.accessibility_themes import (
        apply_high_contrast,
        apply_colorblind_theme,
        get_theme,
        set_font_scale
    )
    
    # Apply high contrast dark theme
    apply_high_contrast(app, mode='dark')
    
    # Apply colorblind-friendly theme
    apply_colorblind_theme(app, type='deuteranopia')
    
    # Scale fonts
    set_font_scale(app, scale=1.5)
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Try PyQt5 imports
try:
    from PyQt5.QtWidgets import QApplication
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False


class ThemeType(Enum):
    """Available theme types."""
    DEFAULT = auto()
    HIGH_CONTRAST_DARK = auto()
    HIGH_CONTRAST_LIGHT = auto()
    COLORBLIND_DEUTERANOPIA = auto()    # Red-green (most common)
    COLORBLIND_PROTANOPIA = auto()       # Red weakness
    COLORBLIND_TRITANOPIA = auto()       # Blue-yellow
    LARGE_TEXT = auto()
    DYSLEXIA_FRIENDLY = auto()


class MotionPreference(Enum):
    """Motion preferences."""
    FULL = auto()           # All animations
    REDUCED = auto()        # Minimal animations
    NONE = auto()           # No animations


@dataclass
class ThemeColors:
    """Color palette for a theme."""
    # Main colors
    background: str = "#1e1e1e"
    foreground: str = "#ffffff"
    
    # Accent colors
    primary: str = "#007acc"
    secondary: str = "#6c757d"
    
    # Status colors
    success: str = "#28a745"
    warning: str = "#ffc107"
    error: str = "#dc3545"
    info: str = "#17a2b8"
    
    # UI elements
    border: str = "#444444"
    selection: str = "#264f78"
    hover: str = "#333333"
    disabled: str = "#666666"
    
    # Text
    text_primary: str = "#ffffff"
    text_secondary: str = "#cccccc"
    text_disabled: str = "#888888"
    
    # Chat specific
    user_message_bg: str = "#2d4a2d"
    assistant_message_bg: str = "#2d2d4a"
    system_message_bg: str = "#4a2d2d"


@dataclass
class ThemeConfig:
    """Complete theme configuration."""
    name: str
    type: ThemeType
    colors: ThemeColors
    
    # Font settings
    font_family: str = "Segoe UI"
    font_size: int = 10
    font_weight: str = "normal"
    line_height: float = 1.4
    
    # Spacing
    padding: int = 8
    margin: int = 4
    border_radius: int = 4
    border_width: int = 1
    
    # Animation
    motion: MotionPreference = MotionPreference.FULL
    transition_duration: int = 200  # ms
    
    # Accessibility
    focus_indicator_width: int = 2
    min_touch_target: int = 44  # px


# Pre-defined themes
THEMES: Dict[ThemeType, ThemeConfig] = {
    ThemeType.DEFAULT: ThemeConfig(
        name="Default Dark",
        type=ThemeType.DEFAULT,
        colors=ThemeColors()
    ),
    
    ThemeType.HIGH_CONTRAST_DARK: ThemeConfig(
        name="High Contrast Dark",
        type=ThemeType.HIGH_CONTRAST_DARK,
        colors=ThemeColors(
            background="#000000",
            foreground="#ffffff",
            primary="#ffff00",          # Bright yellow
            secondary="#00ffff",        # Cyan
            success="#00ff00",          # Bright green
            warning="#ffff00",          # Yellow
            error="#ff0000",            # Bright red
            info="#00ffff",             # Cyan
            border="#ffffff",
            selection="#0000ff",        # Pure blue
            hover="#333333",
            disabled="#666666",
            text_primary="#ffffff",
            text_secondary="#ffff00",
            text_disabled="#888888",
            user_message_bg="#003300",
            assistant_message_bg="#000033",
            system_message_bg="#330000"
        ),
        border_width=2,
        focus_indicator_width=3
    ),
    
    ThemeType.HIGH_CONTRAST_LIGHT: ThemeConfig(
        name="High Contrast Light",
        type=ThemeType.HIGH_CONTRAST_LIGHT,
        colors=ThemeColors(
            background="#ffffff",
            foreground="#000000",
            primary="#0000cc",          # Dark blue
            secondary="#660066",        # Purple
            success="#006600",          # Dark green
            warning="#996600",          # Dark yellow/orange
            error="#cc0000",            # Dark red
            info="#006666",             # Dark cyan
            border="#000000",
            selection="#ffff00",        # Yellow highlight
            hover="#eeeeee",
            disabled="#999999",
            text_primary="#000000",
            text_secondary="#333333",
            text_disabled="#666666",
            user_message_bg="#e6ffe6",
            assistant_message_bg="#e6e6ff",
            system_message_bg="#ffe6e6"
        ),
        border_width=2,
        focus_indicator_width=3
    ),
    
    ThemeType.COLORBLIND_DEUTERANOPIA: ThemeConfig(
        name="Colorblind Friendly (Deuteranopia)",
        type=ThemeType.COLORBLIND_DEUTERANOPIA,
        colors=ThemeColors(
            background="#1e1e1e",
            foreground="#ffffff",
            primary="#0077bb",          # Blue (safe)
            secondary="#ee7733",        # Orange
            success="#009988",          # Teal (instead of green)
            warning="#ee7733",          # Orange
            error="#cc3311",            # Red-orange
            info="#33bbee",             # Light blue
            border="#444444",
            selection="#0077bb",
            hover="#333333",
            disabled="#666666",
            text_primary="#ffffff",
            text_secondary="#cccccc",
            text_disabled="#888888",
            user_message_bg="#002244",
            assistant_message_bg="#442200",
            system_message_bg="#440022"
        )
    ),
    
    ThemeType.COLORBLIND_PROTANOPIA: ThemeConfig(
        name="Colorblind Friendly (Protanopia)",
        type=ThemeType.COLORBLIND_PROTANOPIA,
        colors=ThemeColors(
            background="#1e1e1e",
            foreground="#ffffff",
            primary="#0077bb",
            secondary="#ddaa33",
            success="#009988",
            warning="#ddaa33",
            error="#bb5566",
            info="#33bbee",
            border="#444444",
            selection="#0077bb",
            hover="#333333",
            disabled="#666666",
            text_primary="#ffffff",
            text_secondary="#cccccc",
            text_disabled="#888888",
            user_message_bg="#002244",
            assistant_message_bg="#443300",
            system_message_bg="#442244"
        )
    ),
    
    ThemeType.COLORBLIND_TRITANOPIA: ThemeConfig(
        name="Colorblind Friendly (Tritanopia)",
        type=ThemeType.COLORBLIND_TRITANOPIA,
        colors=ThemeColors(
            background="#1e1e1e",
            foreground="#ffffff",
            primary="#ee3377",          # Pink
            secondary="#009988",        # Teal
            success="#009988",          # Teal
            warning="#ee3377",          # Pink
            error="#cc3311",            # Red
            info="#009988",             # Teal
            border="#444444",
            selection="#ee3377",
            hover="#333333",
            disabled="#666666",
            text_primary="#ffffff",
            text_secondary="#cccccc",
            text_disabled="#888888",
            user_message_bg="#002244",
            assistant_message_bg="#330033",
            system_message_bg="#440000"
        )
    ),
    
    ThemeType.LARGE_TEXT: ThemeConfig(
        name="Large Text",
        type=ThemeType.LARGE_TEXT,
        colors=ThemeColors(),
        font_size=14,
        padding=12,
        margin=8,
        min_touch_target=56
    ),
    
    ThemeType.DYSLEXIA_FRIENDLY: ThemeConfig(
        name="Dyslexia Friendly",
        type=ThemeType.DYSLEXIA_FRIENDLY,
        colors=ThemeColors(
            background="#faf8f5",        # Warm off-white
            foreground="#333333",
            primary="#0066cc",
            secondary="#666666",
            success="#228b22",
            warning="#cc8800",
            error="#cc3333",
            info="#336699",
            border="#cccccc",
            selection="#cce5ff",
            hover="#f0ede8",
            disabled="#999999",
            text_primary="#333333",
            text_secondary="#555555",
            text_disabled="#888888",
            user_message_bg="#e8f4e8",
            assistant_message_bg="#e8e8f4",
            system_message_bg="#f4e8e8"
        ),
        font_family="OpenDyslexic, Comic Sans MS, Arial",
        font_size=12,
        line_height=1.6,
        padding=12
    )
}


def get_theme(theme_type: ThemeType) -> ThemeConfig:
    """Get a theme configuration."""
    return THEMES.get(theme_type, THEMES[ThemeType.DEFAULT])


def generate_stylesheet(theme: ThemeConfig) -> str:
    """
    Generate Qt stylesheet from theme config.
    
    Args:
        theme: ThemeConfig to use
        
    Returns:
        Qt stylesheet string
    """
    c = theme.colors
    
    # Animation settings
    transition = f"{theme.transition_duration}ms" if theme.motion != MotionPreference.NONE else "0ms"
    
    stylesheet = f"""
        /* Global */
        QWidget {{
            background-color: {c.background};
            color: {c.text_primary};
            font-family: {theme.font_family};
            font-size: {theme.font_size}pt;
            border: none;
        }}
        
        /* Main Window */
        QMainWindow {{
            background-color: {c.background};
        }}
        
        /* Labels */
        QLabel {{
            color: {c.text_primary};
            padding: {theme.padding // 2}px;
        }}
        
        /* Buttons */
        QPushButton {{
            background-color: {c.primary};
            color: {c.foreground};
            border: {theme.border_width}px solid {c.border};
            border-radius: {theme.border_radius}px;
            padding: {theme.padding}px {theme.padding * 2}px;
            min-height: {theme.min_touch_target}px;
        }}
        
        QPushButton:hover {{
            background-color: {c.hover};
            border-color: {c.primary};
        }}
        
        QPushButton:pressed {{
            background-color: {c.selection};
        }}
        
        QPushButton:disabled {{
            background-color: {c.disabled};
            color: {c.text_disabled};
        }}
        
        QPushButton:focus {{
            border: {theme.focus_indicator_width}px solid {c.primary};
        }}
        
        /* Line Edit */
        QLineEdit {{
            background-color: {c.background};
            color: {c.text_primary};
            border: {theme.border_width}px solid {c.border};
            border-radius: {theme.border_radius}px;
            padding: {theme.padding}px;
            min-height: {theme.min_touch_target}px;
        }}
        
        QLineEdit:focus {{
            border: {theme.focus_indicator_width}px solid {c.primary};
        }}
        
        /* Text Edit */
        QTextEdit, QPlainTextEdit {{
            background-color: {c.background};
            color: {c.text_primary};
            border: {theme.border_width}px solid {c.border};
            border-radius: {theme.border_radius}px;
            padding: {theme.padding}px;
            line-height: {theme.line_height};
        }}
        
        QTextEdit:focus, QPlainTextEdit:focus {{
            border: {theme.focus_indicator_width}px solid {c.primary};
        }}
        
        /* Combo Box */
        QComboBox {{
            background-color: {c.background};
            color: {c.text_primary};
            border: {theme.border_width}px solid {c.border};
            border-radius: {theme.border_radius}px;
            padding: {theme.padding}px;
            min-height: {theme.min_touch_target}px;
        }}
        
        QComboBox:focus {{
            border: {theme.focus_indicator_width}px solid {c.primary};
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 30px;
        }}
        
        /* Lists and Tables */
        QListWidget, QTableWidget, QTreeWidget {{
            background-color: {c.background};
            color: {c.text_primary};
            border: {theme.border_width}px solid {c.border};
            border-radius: {theme.border_radius}px;
        }}
        
        QListWidget::item, QTableWidget::item, QTreeWidget::item {{
            padding: {theme.padding}px;
            min-height: {theme.min_touch_target}px;
        }}
        
        QListWidget::item:selected, QTableWidget::item:selected, QTreeWidget::item:selected {{
            background-color: {c.selection};
            color: {c.foreground};
        }}
        
        QListWidget::item:hover, QTableWidget::item:hover, QTreeWidget::item:hover {{
            background-color: {c.hover};
        }}
        
        /* Scroll Bars */
        QScrollBar:vertical {{
            background-color: {c.background};
            width: 16px;
            border: none;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {c.border};
            border-radius: 8px;
            min-height: 40px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {c.primary};
        }}
        
        QScrollBar:horizontal {{
            background-color: {c.background};
            height: 16px;
            border: none;
        }}
        
        QScrollBar::handle:horizontal {{
            background-color: {c.border};
            border-radius: 8px;
            min-width: 40px;
        }}
        
        /* Tabs */
        QTabWidget::pane {{
            border: {theme.border_width}px solid {c.border};
            border-radius: {theme.border_radius}px;
        }}
        
        QTabBar::tab {{
            background-color: {c.background};
            color: {c.text_secondary};
            border: {theme.border_width}px solid {c.border};
            padding: {theme.padding}px {theme.padding * 2}px;
            min-height: {theme.min_touch_target}px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {c.selection};
            color: {c.foreground};
            border-bottom: none;
        }}
        
        QTabBar::tab:hover {{
            background-color: {c.hover};
        }}
        
        /* Progress Bar */
        QProgressBar {{
            background-color: {c.background};
            border: {theme.border_width}px solid {c.border};
            border-radius: {theme.border_radius}px;
            text-align: center;
            color: {c.text_primary};
        }}
        
        QProgressBar::chunk {{
            background-color: {c.primary};
            border-radius: {theme.border_radius - 1}px;
        }}
        
        /* Sliders */
        QSlider::groove:horizontal {{
            background-color: {c.border};
            height: 8px;
            border-radius: 4px;
        }}
        
        QSlider::handle:horizontal {{
            background-color: {c.primary};
            width: {theme.min_touch_target}px;
            margin: -16px 0;
            border-radius: {theme.min_touch_target // 2}px;
        }}
        
        /* Check Box */
        QCheckBox {{
            spacing: {theme.padding}px;
        }}
        
        QCheckBox::indicator {{
            width: 24px;
            height: 24px;
            border: {theme.border_width}px solid {c.border};
            border-radius: 4px;
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {c.primary};
            border-color: {c.primary};
        }}
        
        QCheckBox::indicator:focus {{
            border: {theme.focus_indicator_width}px solid {c.primary};
        }}
        
        /* Message styling */
        .user-message {{
            background-color: {c.user_message_bg};
            border-radius: {theme.border_radius}px;
            padding: {theme.padding}px;
        }}
        
        .assistant-message {{
            background-color: {c.assistant_message_bg};
            border-radius: {theme.border_radius}px;
            padding: {theme.padding}px;
        }}
        
        .system-message {{
            background-color: {c.system_message_bg};
            border-radius: {theme.border_radius}px;
            padding: {theme.padding}px;
        }}
        
        /* Status colors */
        .success {{
            color: {c.success};
        }}
        
        .warning {{
            color: {c.warning};
        }}
        
        .error {{
            color: {c.error};
        }}
        
        .info {{
            color: {c.info};
        }}
    """
    
    return stylesheet


def apply_theme(
    app_or_widget,
    theme: ThemeConfig | ThemeType | str
):
    """
    Apply a theme to an application or widget.
    
    Args:
        app_or_widget: QApplication or QWidget
        theme: ThemeConfig, ThemeType, or theme name string
    """
    if not PYQT5_AVAILABLE:
        logger.warning("PyQt5 not available, cannot apply theme")
        return
    
    # Get theme config
    if isinstance(theme, str):
        theme = ThemeType[theme.upper().replace(' ', '_')]
    if isinstance(theme, ThemeType):
        theme = get_theme(theme)
    
    # Generate and apply stylesheet
    stylesheet = generate_stylesheet(theme)
    app_or_widget.setStyleSheet(stylesheet)
    
    logger.info(f"Applied theme: {theme.name}")


def apply_high_contrast(app_or_widget, mode: str = 'dark'):
    """
    Apply high contrast theme.
    
    Args:
        app_or_widget: QApplication or QWidget
        mode: 'dark' or 'light'
    """
    theme_type = (
        ThemeType.HIGH_CONTRAST_DARK if mode.lower() == 'dark'
        else ThemeType.HIGH_CONTRAST_LIGHT
    )
    apply_theme(app_or_widget, theme_type)


def apply_colorblind_theme(app_or_widget, type: str = 'deuteranopia'):
    """
    Apply colorblind-friendly theme.
    
    Args:
        app_or_widget: QApplication or QWidget
        type: 'deuteranopia', 'protanopia', or 'tritanopia'
    """
    type_map = {
        'deuteranopia': ThemeType.COLORBLIND_DEUTERANOPIA,
        'protanopia': ThemeType.COLORBLIND_PROTANOPIA,
        'tritanopia': ThemeType.COLORBLIND_TRITANOPIA
    }
    theme_type = type_map.get(type.lower(), ThemeType.COLORBLIND_DEUTERANOPIA)
    apply_theme(app_or_widget, theme_type)


def set_font_scale(app_or_widget, scale: float = 1.0):
    """
    Scale fonts throughout the application.
    
    Args:
        app_or_widget: QApplication or QWidget
        scale: Font scale factor (1.0 = normal, 1.5 = 50% larger)
    """
    if not PYQT5_AVAILABLE:
        return
    
    if isinstance(app_or_widget, QApplication):
        font = app_or_widget.font()
    else:
        font = app_or_widget.font()
    
    new_size = int(font.pointSize() * scale)
    font.setPointSize(new_size)
    
    if isinstance(app_or_widget, QApplication):
        app_or_widget.setFont(font)
    else:
        app_or_widget.setFont(font)
    
    logger.info(f"Font scale set to {scale}x (size: {new_size}pt)")


def set_motion_preference(preference: MotionPreference):
    """
    Set global motion preference.
    
    Args:
        preference: MotionPreference value
    """
    # Store in config or environment
    import os
    os.environ['ENIGMA_MOTION_PREFERENCE'] = preference.name
    logger.info(f"Motion preference set to: {preference.name}")


def get_motion_preference() -> MotionPreference:
    """Get current motion preference."""
    import os
    pref = os.environ.get('ENIGMA_MOTION_PREFERENCE', 'FULL')
    try:
        return MotionPreference[pref]
    except KeyError:
        return MotionPreference.FULL


def should_animate() -> bool:
    """Check if animations should be used."""
    return get_motion_preference() != MotionPreference.NONE


def list_themes() -> List[Dict[str, Any]]:
    """List all available themes."""
    return [
        {
            'type': t.name,
            'name': THEMES[t].name,
            'description': _get_theme_description(t)
        }
        for t in ThemeType
    ]


def _get_theme_description(theme_type: ThemeType) -> str:
    """Get description for a theme type."""
    descriptions = {
        ThemeType.DEFAULT: "Standard dark theme",
        ThemeType.HIGH_CONTRAST_DARK: "High contrast for low vision (dark background)",
        ThemeType.HIGH_CONTRAST_LIGHT: "High contrast for low vision (light background)",
        ThemeType.COLORBLIND_DEUTERANOPIA: "Optimized for red-green color blindness",
        ThemeType.COLORBLIND_PROTANOPIA: "Optimized for red weakness",
        ThemeType.COLORBLIND_TRITANOPIA: "Optimized for blue-yellow color blindness",
        ThemeType.LARGE_TEXT: "Larger text and touch targets",
        ThemeType.DYSLEXIA_FRIENDLY: "Font and colors optimized for dyslexia"
    }
    return descriptions.get(theme_type, "")
