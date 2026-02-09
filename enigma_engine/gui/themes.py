"""
Theme System for Enigma AI Engine GUI

Provides dark/light theme support with customization.

Usage:
    from enigma_engine.gui.themes import ThemeManager, get_theme_manager
    
    theme = get_theme_manager()
    
    # Apply to application
    theme.apply_theme(app)
    
    # Toggle theme
    theme.toggle_theme()
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from PyQt5.QtWidgets import QApplication

logger = logging.getLogger(__name__)


class ThemeType(Enum):
    """Available theme types."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"  # Follow system


@dataclass
class ThemeColors:
    """Colors for a theme."""
    # Base colors
    background: str
    surface: str
    on_background: str
    on_surface: str
    
    # Primary colors
    primary: str
    primary_variant: str
    on_primary: str
    
    # Secondary colors
    secondary: str
    on_secondary: str
    
    # Status colors
    error: str
    warning: str
    success: str
    info: str
    
    # Component specific
    border: str
    disabled: str
    input_background: str
    tooltip_background: str
    
    # Selection
    selection: str
    selection_text: str


# Predefined themes
DARK_THEME = ThemeColors(
    background="#1e1e2e",
    surface="#2a2a3a",
    on_background="#cdd6f4",
    on_surface="#bac2de",
    
    primary="#89b4fa",
    primary_variant="#74c7ec",
    on_primary="#1e1e2e",
    
    secondary="#a6e3a1",
    on_secondary="#1e1e2e",
    
    error="#f38ba8",
    warning="#f9e2af",
    success="#a6e3a1",
    info="#89dceb",
    
    border="#45475a",
    disabled="#585b70",
    input_background="#313244",
    tooltip_background="#313244",
    
    selection="#45475a",
    selection_text="#cdd6f4"
)

LIGHT_THEME = ThemeColors(
    background="#eff1f5",
    surface="#ffffff",
    on_background="#4c4f69",
    on_surface="#5c5f77",
    
    primary="#1e66f5",
    primary_variant="#7287fd",
    on_primary="#ffffff",
    
    secondary="#40a02b",
    on_secondary="#ffffff",
    
    error="#d20f39",
    warning="#df8e1d",
    success="#40a02b",
    info="#209fb5",
    
    border="#ccd0da",
    disabled="#9ca0b0",
    input_background="#e6e9ef",
    tooltip_background="#5c5f77",
    
    selection="#bcc0cc",
    selection_text="#4c4f69"
)


class ThemeManager:
    """
    Manages application theming.
    
    Features:
    - Dark/light theme toggle
    - Persistent preference
    - Custom theme support
    - System theme detection
    """
    
    THEMES = {
        ThemeType.DARK: DARK_THEME,
        ThemeType.LIGHT: LIGHT_THEME
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        from ..config import CONFIG
        
        self.config_path = config_path or Path(
            CONFIG.get("data_dir", "data")
        ) / "theme_config.json"
        
        self.current_theme = ThemeType.DARK
        self._app: Optional[QApplication] = None
        
        self._load_config()
    
    def _load_config(self):
        """Load theme preference from config."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)
                theme_name = data.get("theme", "dark")
                self.current_theme = ThemeType(theme_name)
                logger.debug(f"Loaded theme preference: {self.current_theme.value}")
            except Exception as e:
                logger.warning(f"Could not load theme config: {e}")
    
    def _save_config(self):
        """Save theme preference to config."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump({"theme": self.current_theme.value}, f)
        except Exception as e:
            logger.error(f"Could not save theme config: {e}")
    
    def get_colors(self) -> ThemeColors:
        """Get current theme colors."""
        return self.THEMES.get(self.current_theme, DARK_THEME)
    
    def set_theme(self, theme: ThemeType):
        """Set the current theme."""
        self.current_theme = theme
        self._save_config()
        
        if self._app:
            self._apply_stylesheet(self._app)
        
        logger.info(f"Theme changed to {theme.value}")
    
    def toggle_theme(self):
        """Toggle between dark and light themes."""
        if self.current_theme == ThemeType.DARK:
            self.set_theme(ThemeType.LIGHT)
        else:
            self.set_theme(ThemeType.DARK)
    
    def apply_theme(self, app: QApplication):
        """
        Apply theme to application.
        
        Args:
            app: QApplication instance
        """
        self._app = app
        self._apply_stylesheet(app)
    
    def _apply_stylesheet(self, app: QApplication):
        """Generate and apply stylesheet."""
        colors = self.get_colors()
        stylesheet = self._generate_stylesheet(colors)
        app.setStyleSheet(stylesheet)
    
    def _generate_stylesheet(self, c: ThemeColors) -> str:
        """Generate complete stylesheet from theme colors."""
        return f"""
/* Main Window */
QMainWindow, QDialog {{
    background-color: {c.background};
    color: {c.on_background};
}}

/* Generic Widget */
QWidget {{
    background-color: transparent;
    color: {c.on_background};
    font-family: 'Segoe UI', 'Inter', sans-serif;
}}

/* Labels */
QLabel {{
    color: {c.on_background};
}}

/* Buttons */
QPushButton {{
    background-color: {c.primary};
    color: {c.on_primary};
    border: none;
    border-radius: 4px;
    padding: 6px 16px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {c.primary_variant};
}}

QPushButton:pressed {{
    background-color: {c.primary};
}}

QPushButton:disabled {{
    background-color: {c.disabled};
    color: {c.border};
}}

/* Secondary Buttons */
QPushButton[secondary="true"] {{
    background-color: {c.surface};
    color: {c.on_surface};
    border: 1px solid {c.border};
}}

QPushButton[secondary="true"]:hover {{
    background-color: {c.background};
}}

/* Line Edit */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {c.input_background};
    color: {c.on_surface};
    border: 1px solid {c.border};
    border-radius: 4px;
    padding: 6px;
    selection-background-color: {c.selection};
    selection-color: {c.selection_text};
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {c.primary};
}}

/* Combo Box */
QComboBox {{
    background-color: {c.input_background};
    color: {c.on_surface};
    border: 1px solid {c.border};
    border-radius: 4px;
    padding: 6px;
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox QAbstractItemView {{
    background-color: {c.surface};
    color: {c.on_surface};
    border: 1px solid {c.border};
    selection-background-color: {c.selection};
}}

/* Spin Box */
QSpinBox, QDoubleSpinBox {{
    background-color: {c.input_background};
    color: {c.on_surface};
    border: 1px solid {c.border};
    border-radius: 4px;
    padding: 4px;
}}

/* Slider */
QSlider::groove:horizontal {{
    background-color: {c.border};
    height: 4px;
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background-color: {c.primary};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}

/* Check Box */
QCheckBox {{
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {c.border};
    border-radius: 3px;
}}

QCheckBox::indicator:checked {{
    background-color: {c.primary};
    border-color: {c.primary};
}}

/* Radio Button */
QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {c.border};
    border-radius: 9px;
}}

QRadioButton::indicator:checked {{
    background-color: {c.primary};
    border-color: {c.primary};
}}

/* Tab Widget */
QTabWidget::pane {{
    border: 1px solid {c.border};
    border-radius: 4px;
    background-color: {c.surface};
}}

QTabBar::tab {{
    background-color: {c.background};
    color: {c.on_background};
    padding: 8px 16px;
    border-bottom: 2px solid transparent;
}}

QTabBar::tab:selected {{
    color: {c.primary};
    border-bottom: 2px solid {c.primary};
}}

QTabBar::tab:hover {{
    background-color: {c.surface};
}}

/* Scroll Bar */
QScrollBar:vertical {{
    background-color: {c.background};
    width: 12px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {c.border};
    border-radius: 6px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {c.disabled};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {c.background};
    height: 12px;
}}

QScrollBar::handle:horizontal {{
    background-color: {c.border};
    border-radius: 6px;
    min-width: 20px;
}}

/* Progress Bar */
QProgressBar {{
    background-color: {c.border};
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {c.primary};
    border-radius: 4px;
}}

/* List Widget */
QListWidget, QTreeWidget, QTableWidget {{
    background-color: {c.surface};
    color: {c.on_surface};
    border: 1px solid {c.border};
    border-radius: 4px;
    alternate-background-color: {c.background};
}}

QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {c.selection};
    color: {c.selection_text};
}}

QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {c.background};
}}

/* Table */
QHeaderView::section {{
    background-color: {c.surface};
    color: {c.on_surface};
    border: none;
    border-bottom: 1px solid {c.border};
    padding: 8px;
}}

/* Group Box */
QGroupBox {{
    border: 1px solid {c.border};
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
}}

QGroupBox::title {{
    color: {c.on_background};
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}}

/* Menu */
QMenuBar {{
    background-color: {c.background};
    color: {c.on_background};
}}

QMenuBar::item:selected {{
    background-color: {c.surface};
}}

QMenu {{
    background-color: {c.surface};
    color: {c.on_surface};
    border: 1px solid {c.border};
}}

QMenu::item:selected {{
    background-color: {c.selection};
}}

/* Tool Tip */
QToolTip {{
    background-color: {c.tooltip_background};
    color: {c.on_surface};
    border: 1px solid {c.border};
    border-radius: 4px;
    padding: 4px;
}}

/* Status Bar */
QStatusBar {{
    background-color: {c.surface};
    color: {c.on_surface};
    border-top: 1px solid {c.border};
}}

/* Splitter */
QSplitter::handle {{
    background-color: {c.border};
}}

/* Dock Widget */
QDockWidget {{
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background-color: {c.surface};
    color: {c.on_surface};
    padding: 8px;
}}
"""
    
    def get_stylesheet(self) -> str:
        """Get current theme stylesheet."""
        return self._generate_stylesheet(self.get_colors())


# Global instance
_theme_manager: Optional[ThemeManager] = None


def get_theme_manager() -> ThemeManager:
    """Get or create global theme manager."""
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager
