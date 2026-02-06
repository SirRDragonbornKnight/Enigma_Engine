"""
================================================================================
Global UI Settings - Centralized font/scaling/theme management.
================================================================================

Controls:
- Font sizes (with user-adjustable scaling)
- Theme colors (including Cerulean variants)
- Consistent styling across all tabs
- Settings persistence

USAGE:
    from forge_ai.gui.ui_settings import UISettings, get_ui_settings
    
    settings = get_ui_settings()
    font_size = settings.get_font_size("normal")  # Returns scaled size
    settings.set_scale(1.2)  # 120% scaling
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FontSizes:
    """Font sizes for different UI elements (base values before scaling)."""
    tiny: int = 13
    small: int = 14
    normal: int = 16
    medium: int = 17
    large: int = 19
    xlarge: int = 21
    title: int = 24
    header: int = 26
    huge: int = 30


@dataclass
class ThemeColors:
    """Color palette for a theme."""
    name: str
    
    # Base colors
    background: str = "#1e1e2e"
    background_alt: str = "#181825"
    surface: str = "#313244"
    surface_hover: str = "#45475a"
    border: str = "#45475a"
    
    # Text colors
    text: str = "#cdd6f4"
    text_secondary: str = "#a6adc8"
    text_muted: str = "#6c7086"
    
    # Accent colors
    primary: str = "#89b4fa"      # Main accent (buttons, links)
    secondary: str = "#cba6f7"    # Secondary accent
    success: str = "#a6e3a1"
    warning: str = "#f9e2af"
    error: str = "#f38ba8"
    info: str = "#89dceb"
    
    # Special colors
    highlight: str = "#74c7ec"
    selection: str = "#45475a"


# Pre-defined themes
THEMES = {
    "dark": ThemeColors(
        name="Dark",
        background="#1e1e2e",
        background_alt="#181825",
        surface="#313244",
        surface_hover="#45475a",
        border="#45475a",
        text="#cdd6f4",
        text_secondary="#a6adc8",
        text_muted="#6c7086",
        primary="#89b4fa",
        secondary="#cba6f7",
        success="#a6e3a1",
        warning="#f9e2af",
        error="#f38ba8",
        info="#89dceb",
        highlight="#74c7ec",
        selection="#45475a",
    ),
    "cerulean": ThemeColors(
        name="Cerulean",
        background="#0a1929",       # Deep navy
        background_alt="#071421",   # Darker navy
        surface="#0d2137",          # Navy surface
        surface_hover="#12344f",    # Lighter navy
        border="#1a4a6e",           # Cerulean border
        text="#e3f2fd",             # Light blue-white
        text_secondary="#90caf9",   # Light cerulean
        text_muted="#5c8ab5",       # Muted cerulean
        primary="#007bb5",          # TRUE CERULEAN
        secondary="#4fc3f7",        # Light cerulean
        success="#4caf50",          # Green
        warning="#ffb74d",          # Amber
        error="#ef5350",            # Red
        info="#29b6f6",             # Cerulean info
        highlight="#00acc1",        # Cyan cerulean
        selection="#1565c0",        # Deep cerulean selection
    ),
    "cerulean_light": ThemeColors(
        name="Cerulean Light",
        background="#e1f5fe",       # Very light cerulean
        background_alt="#b3e5fc",   # Light cerulean
        surface="#ffffff",          # White
        surface_hover="#e3f2fd",    # Hover blue
        border="#4fc3f7",           # Cerulean border
        text="#01579b",             # Dark cerulean text
        text_secondary="#0277bd",   # Medium cerulean
        text_muted="#4fc3f7",       # Light cerulean
        primary="#0097a7",          # Cerulean primary
        secondary="#00838f",        # Teal cerulean
        success="#2e7d32",          # Green
        warning="#f57c00",          # Orange
        error="#c62828",            # Red
        info="#0288d1",             # Blue
        highlight="#007bb5",        # TRUE CERULEAN highlight
        selection="#81d4fa",        # Light selection
    ),
    "midnight": ThemeColors(
        name="Midnight",
        background="#0d0d1a",
        background_alt="#080810",
        surface="#1a1a2e",
        surface_hover="#25253d",
        border="#2e2e4d",
        text="#e6e6fa",
        text_secondary="#b8b8d1",
        text_muted="#6666aa",
        primary="#7c3aed",
        secondary="#a78bfa",
        success="#34d399",
        warning="#fbbf24",
        error="#f87171",
        info="#60a5fa",
        highlight="#8b5cf6",
        selection="#4c1d95",
    ),
    "shadow": ThemeColors(
        name="Shadow",
        background="#121212",
        background_alt="#0a0a0a",
        surface="#1e1e1e",
        surface_hover="#2d2d2d",
        border="#333333",
        text="#e0e0e0",
        text_secondary="#9e9e9e",
        text_muted="#616161",
        primary="#bb86fc",
        secondary="#03dac6",
        success="#00c853",
        warning="#ffab00",
        error="#cf6679",
        info="#2196f3",
        highlight="#03dac6",
        selection="#3700b3",
    ),
    "ocean_cerulean": ThemeColors(
        name="Ocean Cerulean",
        background="#002b36",       # Solarized dark base
        background_alt="#001f27",   # Darker
        surface="#073642",          # Surface
        surface_hover="#0a4a59",    # Hover
        border="#007bb5",           # CERULEAN border
        text="#93a1a1",             # Light text
        text_secondary="#839496",   # Secondary
        text_muted="#586e75",       # Muted
        primary="#007bb5",          # TRUE CERULEAN
        secondary="#2aa198",        # Cyan
        success="#859900",          # Green
        warning="#b58900",          # Yellow
        error="#dc322f",            # Red
        info="#268bd2",             # Blue
        highlight="#007bb5",        # CERULEAN
        selection="#007bb5",        # CERULEAN selection
    ),
}


class UISettings:
    """
    Global UI settings manager.
    
    Handles font scaling, themes, and persistent settings.
    """
    
    def __init__(self):
        self._font_sizes = FontSizes()
        self._scale = 1.0
        self._opacity = 0.95  # Dialog transparency (0.0 = invisible, 1.0 = solid)
        self._theme_name = "cerulean"  # Default to cerulean
        self._theme = THEMES["cerulean"]
        self._listeners: list[Callable[[], None]] = []
        
        # Load saved settings
        self._load_settings()
    
    def _load_settings(self):
        """Load settings from file."""
        try:
            from ..config import CONFIG
            settings_path = Path(CONFIG["data_dir"]) / "ui_settings.json"
            
            if settings_path.exists():
                with open(settings_path) as f:
                    data = json.load(f)
                
                self._scale = data.get("scale", 1.0)
                self._opacity = data.get("opacity", 0.95)
                theme_name = data.get("theme", "cerulean")
                if theme_name in THEMES:
                    self._theme_name = theme_name
                    self._theme = THEMES[theme_name]
                
                # Load custom font sizes if set
                if "font_sizes" in data:
                    for key, value in data["font_sizes"].items():
                        if hasattr(self._font_sizes, key):
                            setattr(self._font_sizes, key, value)
                
                logger.debug(f"Loaded UI settings: scale={self._scale}, theme={self._theme_name}")
        except Exception as e:
            logger.warning(f"Could not load UI settings: {e}")
    
    def save_settings(self):
        """Save settings to file."""
        try:
            from ..config import CONFIG
            settings_path = Path(CONFIG["data_dir"]) / "ui_settings.json"
            
            data = {
                "scale": self._scale,
                "opacity": self._opacity,
                "theme": self._theme_name,
                "font_sizes": {
                    "tiny": self._font_sizes.tiny,
                    "small": self._font_sizes.small,
                    "normal": self._font_sizes.normal,
                    "medium": self._font_sizes.medium,
                    "large": self._font_sizes.large,
                    "xlarge": self._font_sizes.xlarge,
                    "title": self._font_sizes.title,
                    "header": self._font_sizes.header,
                    "huge": self._font_sizes.huge,
                }
            }
            
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(settings_path, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Saved UI settings")
        except Exception as e:
            logger.warning(f"Could not save UI settings: {e}")
    
    # Font size methods
    
    def get_font_size(self, size_name: str) -> int:
        """Get scaled font size."""
        base_size = getattr(self._font_sizes, size_name, self._font_sizes.normal)
        return int(base_size * self._scale)
    
    def get_all_font_sizes(self) -> dict[str, int]:
        """Get all scaled font sizes."""
        return {
            "tiny": self.get_font_size("tiny"),
            "small": self.get_font_size("small"),
            "normal": self.get_font_size("normal"),
            "medium": self.get_font_size("medium"),
            "large": self.get_font_size("large"),
            "xlarge": self.get_font_size("xlarge"),
            "title": self.get_font_size("title"),
            "header": self.get_font_size("header"),
            "huge": self.get_font_size("huge"),
        }
    
    def set_scale(self, scale: float):
        """Set font scale (1.0 = 100%)."""
        self._scale = max(0.5, min(2.0, scale))  # Clamp between 50% and 200%
        self.save_settings()
        self._notify_listeners()
    
    def get_scale(self) -> float:
        """Get current scale."""
        return self._scale
    
    @property
    def scale(self) -> float:
        """Current font scale (1.0 = 100%)."""
        return self._scale
    
    # Opacity/transparency methods
    
    def set_opacity(self, opacity: float):
        """Set dialog opacity (0.0 = invisible, 1.0 = solid)."""
        self._opacity = max(0.5, min(1.0, opacity))  # Clamp between 50% and 100%
        self.save_settings()
        self._notify_listeners()
    
    def get_opacity(self) -> float:
        """Get current dialog opacity."""
        return self._opacity
    
    @property
    def opacity(self) -> float:
        """Current dialog opacity (0.0-1.0)."""
        return self._opacity
    
    @property
    def current_theme(self) -> str:
        """Current theme name."""
        return self._theme_name
    
    @property
    def theme(self) -> ThemeColors:
        """Current theme colors."""
        return self._theme
    
    @property
    def font_sizes(self) -> FontSizes:
        """Base font sizes (before scaling)."""
        return self._font_sizes
    
    # Theme methods
    
    def set_theme(self, theme_name: str):
        """Set the active theme."""
        if theme_name in THEMES:
            self._theme_name = theme_name
            self._theme = THEMES[theme_name]
            self.save_settings()
            self._notify_listeners()
    
    def get_theme(self) -> ThemeColors:
        """Get current theme colors."""
        return self._theme
    
    def get_theme_name(self) -> str:
        """Get current theme name."""
        return self._theme_name
    
    def get_available_themes(self) -> list[str]:
        """Get list of available theme names."""
        return list(THEMES.keys())
    
    # Stylesheet generation
    
    def get_global_stylesheet(self) -> str:
        """Generate global stylesheet with current settings."""
        t = self._theme
        fs = self.get_all_font_sizes()
        
        return f"""
            /* Global font and color settings */
            QWidget {{
                background-color: {t.background};
                color: {t.text};
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: {fs['normal']}px;
            }}
            
            QMainWindow {{
                background-color: {t.background};
            }}
            
            /* Labels */
            QLabel {{
                color: {t.text};
                font-size: {fs['normal']}px;
            }}
            
            /* Buttons */
            QPushButton {{
                background-color: {t.primary};
                color: {t.background};
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: {fs['normal']}px;
                font-weight: bold;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {t.highlight};
            }}
            QPushButton:pressed {{
                background-color: {t.secondary};
            }}
            QPushButton:disabled {{
                background-color: {t.surface};
                color: #f38ba8;
                border: 2px dashed #f38ba8;
            }}
            
            /* Input fields */
            QLineEdit, QTextEdit, QPlainTextEdit {{
                background-color: {t.surface};
                color: {t.text};
                border: 1px solid {t.border};
                border-radius: 4px;
                padding: 6px 10px;
                font-size: {fs['normal']}px;
            }}
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
                border-color: {t.primary};
            }}
            
            /* Combo boxes */
            QComboBox {{
                background-color: {t.surface};
                color: {t.text};
                border: 1px solid {t.border};
                border-radius: 4px;
                padding: 6px 10px;
                font-size: {fs['normal']}px;
                min-height: 20px;
            }}
            QComboBox:hover {{
                border-color: {t.primary};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {t.surface};
                color: {t.text};
                selection-background-color: {t.selection};
                border: 1px solid {t.border};
            }}
            
            /* Spin boxes */
            QSpinBox, QDoubleSpinBox {{
                background-color: {t.surface};
                color: {t.text};
                border: 1px solid {t.border};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: {fs['normal']}px;
            }}
            
            /* Checkboxes */
            QCheckBox {{
                color: {t.text};
                font-size: {fs['normal']}px;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid {t.border};
                background-color: {t.surface};
            }}
            QCheckBox::indicator:checked {{
                background-color: {t.primary};
                border-color: {t.primary};
            }}
            
            /* Group boxes */
            QGroupBox {{
                background-color: {t.surface};
                border: 1px solid {t.border};
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px;
                font-size: {fs['medium']}px;
                font-weight: bold;
                color: {t.text};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {t.primary};
            }}
            
            /* Scroll areas */
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background-color: {t.background_alt};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {t.surface_hover};
                border-radius: 6px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {t.primary};
            }}
            
            /* Lists */
            QListWidget {{
                background-color: {t.surface};
                color: {t.text};
                border: 1px solid {t.border};
                border-radius: 4px;
                font-size: {fs['normal']}px;
            }}
            QListWidget::item {{
                padding: 8px;
            }}
            QListWidget::item:selected {{
                background-color: {t.selection};
            }}
            QListWidget::item:hover {{
                background-color: {t.surface_hover};
            }}
            
            /* Tab widget */
            QTabWidget::pane {{
                border: 1px solid {t.border};
                background-color: {t.background};
            }}
            QTabBar::tab {{
                background-color: {t.surface};
                color: {t.text_secondary};
                padding: 8px 16px;
                font-size: {fs['normal']}px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {t.primary};
                color: {t.background};
            }}
            
            /* Progress bars */
            QProgressBar {{
                background-color: {t.surface};
                border: none;
                border-radius: 4px;
                height: 8px;
                text-align: center;
                font-size: {fs['small']}px;
            }}
            QProgressBar::chunk {{
                background-color: {t.primary};
                border-radius: 4px;
            }}
            
            /* Sliders */
            QSlider::groove:horizontal {{
                background-color: {t.surface};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background-color: {t.primary};
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background-color: {t.primary};
                border-radius: 3px;
            }}
            
            /* Tooltips */
            QToolTip {{
                background-color: {t.surface};
                color: {t.text};
                border: 1px solid {t.border};
                padding: 4px 8px;
                font-size: {fs['small']}px;
            }}
            
            /* Frame */
            QFrame {{
                background-color: transparent;
            }}
            
            /* Menus */
            QMenu {{
                background-color: {t.surface};
                color: {t.text};
                border: 1px solid {t.border};
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 20px;
            }}
            QMenu::item:selected {{
                background-color: {t.selection};
            }}
        """
    
    def get_title_style(self) -> str:
        """Get style for title labels."""
        fs = self.get_font_size("title")
        return f"font-size: {fs}px; font-weight: bold; color: {self._theme.primary};"
    
    def get_header_style(self) -> str:
        """Get style for header labels."""
        fs = self.get_font_size("large")
        return f"font-size: {fs}px; font-weight: bold;"
    
    def get_section_header_style(self) -> str:
        """Get style for section headers inside panels."""
        fs = self.get_font_size("medium")
        return f"font-size: {fs}px; font-weight: bold; color: {self._theme.text};"
    
    def get_label_style(self) -> str:
        """Get style for normal labels."""
        fs = self.get_font_size("normal")
        return f"font-size: {fs}px; color: {self._theme.text};"
    
    def get_subtitle_style(self) -> str:
        """Get style for subtitle/secondary text."""
        fs = self.get_font_size("normal")
        return f"font-size: {fs}px; color: {self._theme.text_secondary};"
    
    def get_muted_style(self) -> str:
        """Get style for muted/hint text."""
        fs = self.get_font_size("small")
        return f"font-size: {fs}px; color: {self._theme.text_muted};"
    
    def get_status_style(self, status: str = "normal") -> str:
        """Get style for status labels (success, warning, error, info)."""
        fs = self.get_font_size("normal")
        colors = {
            "success": self._theme.success,
            "warning": self._theme.warning,
            "error": self._theme.error,
            "info": self._theme.info,
            "normal": self._theme.text,
            "muted": self._theme.text_muted,
        }
        color = colors.get(status, self._theme.text)
        return f"font-size: {fs}px; color: {color};"
    
    def get_bold_label_style(self) -> str:
        """Get style for bold labels."""
        fs = self.get_font_size("normal")
        return f"font-size: {fs}px; font-weight: bold; color: {self._theme.text};"
    
    def get_card_style(self) -> str:
        """Get style for card/panel containers."""
        return f"""
            background-color: {self._theme.surface};
            border: 1px solid {self._theme.border};
            border-radius: 8px;
            padding: 12px;
        """
    
    def get_button_style(self, variant: str = "primary") -> str:
        """Get style for buttons (primary, secondary, danger)."""
        fs = self.get_font_size("normal")
        if variant == "secondary":
            return f"""
                background-color: {self._theme.surface};
                color: {self._theme.text};
                border: 1px solid {self._theme.border};
                border-radius: 6px;
                padding: 8px 16px;
                font-size: {fs}px;
            """
        elif variant == "danger":
            return f"""
                background-color: {self._theme.error};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: {fs}px;
                font-weight: bold;
            """
        else:  # primary
            return f"""
                background-color: {self._theme.primary};
                color: {self._theme.background};
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: {fs}px;
                font-weight: bold;
            """
    
    # Listener management
    
    def add_listener(self, callback: Callable[[], None]):
        """Add a listener for settings changes."""
        self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable[[], None]):
        """Remove a listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def _notify_listeners(self):
        """Notify all listeners of changes."""
        for listener in self._listeners:
            try:
                listener()
            except Exception as e:
                logger.error(f"Listener error: {e}")


# Global instance
_ui_settings: Optional[UISettings] = None


def get_ui_settings() -> UISettings:
    """Get the global UI settings instance."""
    global _ui_settings
    if _ui_settings is None:
        _ui_settings = UISettings()
    return _ui_settings


def apply_dialog_transparency(dialog):
    """
    Apply the configured transparency to a QDialog.
    
    Call this in your dialog's __init__ after setting up the UI.
    
    Args:
        dialog: QDialog instance to apply transparency to
    """
    try:
        settings = get_ui_settings()
        opacity = settings.get_opacity()
        if opacity < 1.0:
            dialog.setWindowOpacity(opacity)
    except Exception:
        pass  # Silently fail if PyQt5 not available


__all__ = [
    'UISettings',
    'FontSizes',
    'ThemeColors',
    'THEMES',
    'get_ui_settings',
    'apply_dialog_transparency',
]
