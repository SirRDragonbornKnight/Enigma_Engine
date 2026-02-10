"""
UI Density Options

Provides compact/comfortable/spacious layout density options for the GUI.
Controls padding, margins, spacing, button sizes, and widget heights.

FILE: enigma_engine/gui/density_options.py
TYPE: UI Configuration System
MAIN CLASSES: DensityMode, DensitySettings, DensityManager
"""

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QWidget

from ..config import CONFIG

logger = logging.getLogger(__name__)


class DensityMode(Enum):
    """Available density modes."""
    COMPACT = "compact"
    COMFORTABLE = "comfortable"  # Default
    SPACIOUS = "spacious"
    CUSTOM = "custom"


@dataclass
class DensitySettings:
    """Density configuration values."""
    # Spacing
    item_spacing: int = 8          # Space between items in layouts
    group_spacing: int = 16        # Space between groups/sections
    widget_margin: int = 8         # Margin inside widgets
    container_margin: int = 12     # Margin for containers/groups
    
    # Padding
    button_padding_h: int = 16     # Horizontal button padding
    button_padding_v: int = 8      # Vertical button padding
    input_padding: int = 8         # Input field padding
    list_item_padding: int = 8     # List item padding
    
    # Heights
    button_height: int = 32        # Minimum button height
    input_height: int = 32         # Input field height
    row_height: int = 36           # List/table row height
    tab_height: int = 36           # Tab bar height
    toolbar_height: int = 44       # Toolbar height
    
    # Border radius
    border_radius_small: int = 4   # Small elements
    border_radius_medium: int = 6  # Buttons, inputs
    border_radius_large: int = 8   # Cards, groups
    
    # Icon sizes
    icon_size_small: int = 16
    icon_size_medium: int = 20
    icon_size_large: int = 24
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DensitySettings':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Preset density configurations
DENSITY_PRESETS: dict[DensityMode, DensitySettings] = {
    DensityMode.COMPACT: DensitySettings(
        item_spacing=4,
        group_spacing=8,
        widget_margin=4,
        container_margin=6,
        button_padding_h=8,
        button_padding_v=4,
        input_padding=4,
        list_item_padding=4,
        button_height=24,
        input_height=24,
        row_height=28,
        tab_height=28,
        toolbar_height=32,
        border_radius_small=2,
        border_radius_medium=4,
        border_radius_large=6,
        icon_size_small=14,
        icon_size_medium=16,
        icon_size_large=20
    ),
    DensityMode.COMFORTABLE: DensitySettings(
        # All defaults
    ),
    DensityMode.SPACIOUS: DensitySettings(
        item_spacing=12,
        group_spacing=24,
        widget_margin=12,
        container_margin=16,
        button_padding_h=24,
        button_padding_v=12,
        input_padding=12,
        list_item_padding=12,
        button_height=44,
        input_height=40,
        row_height=48,
        tab_height=44,
        toolbar_height=56,
        border_radius_small=6,
        border_radius_medium=8,
        border_radius_large=12,
        icon_size_small=18,
        icon_size_medium=24,
        icon_size_large=32
    )
}


class DensityManager(QObject):
    """Manages UI density settings across the application."""
    
    density_changed = pyqtSignal(object)  # Emits DensitySettings
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self._mode = DensityMode.COMFORTABLE
        self._settings = DENSITY_PRESETS[DensityMode.COMFORTABLE]
        self._custom_settings: Optional[DensitySettings] = None
        self._callbacks: list[Callable[[DensitySettings], None]] = []
        self._settings_path = Path(CONFIG.get("data_dir", "data")) / "density_settings.json"
        self._load_settings()
        self._initialized = True
    
    def _load_settings(self):
        """Load density settings from file."""
        try:
            if self._settings_path.exists():
                with open(self._settings_path) as f:
                    data = json.load(f)
                    
                mode_str = data.get("mode", "comfortable")
                self._mode = DensityMode(mode_str)
                
                if self._mode == DensityMode.CUSTOM and "custom" in data:
                    self._custom_settings = DensitySettings.from_dict(data["custom"])
                    self._settings = self._custom_settings
                else:
                    self._settings = DENSITY_PRESETS.get(self._mode, DENSITY_PRESETS[DensityMode.COMFORTABLE])
                    
                logger.debug(f"Loaded density mode: {self._mode.value}")
        except Exception as e:
            logger.warning(f"Could not load density settings: {e}")
    
    def _save_settings(self):
        """Save density settings to file."""
        try:
            data = {
                "mode": self._mode.value
            }
            if self._custom_settings:
                data["custom"] = self._custom_settings.to_dict()
            
            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._settings_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save density settings: {e}")
    
    @property
    def mode(self) -> DensityMode:
        """Current density mode."""
        return self._mode
    
    @property
    def settings(self) -> DensitySettings:
        """Current density settings."""
        return self._settings
    
    def set_mode(self, mode: DensityMode):
        """Set density mode."""
        self._mode = mode
        if mode == DensityMode.CUSTOM and self._custom_settings:
            self._settings = self._custom_settings
        else:
            self._settings = DENSITY_PRESETS.get(mode, DENSITY_PRESETS[DensityMode.COMFORTABLE])
        
        self._save_settings()
        self._notify()
    
    def set_custom(self, settings: DensitySettings):
        """Set custom density settings."""
        self._mode = DensityMode.CUSTOM
        self._custom_settings = settings
        self._settings = settings
        self._save_settings()
        self._notify()
    
    def get_stylesheet(self) -> str:
        """Generate stylesheet additions for current density."""
        s = self._settings
        
        return f"""
            /* Density-specific styles */
            
            /* Buttons */
            QPushButton {{
                padding: {s.button_padding_v}px {s.button_padding_h}px;
                min-height: {s.button_height}px;
                border-radius: {s.border_radius_medium}px;
            }}
            
            /* Inputs */
            QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox {{
                padding: {s.input_padding}px;
                min-height: {s.input_height}px;
                border-radius: {s.border_radius_medium}px;
            }}
            
            /* Group boxes */
            QGroupBox {{
                margin-top: {s.container_margin}px;
                padding: {s.container_margin}px;
                border-radius: {s.border_radius_large}px;
            }}
            
            /* Lists */
            QListWidget::item {{
                padding: {s.list_item_padding}px;
                min-height: {s.row_height}px;
            }}
            
            /* Tabs */
            QTabBar::tab {{
                padding: {s.button_padding_v}px {s.button_padding_h}px;
                min-height: {s.tab_height}px;
            }}
            
            /* Scroll areas */
            QScrollArea > QWidget > QWidget {{
                margin: {s.widget_margin}px;
            }}
            
            /* Toolbars */
            QToolBar {{
                min-height: {s.toolbar_height}px;
                spacing: {s.item_spacing}px;
            }}
            
            /* Checkboxes and radio */
            QCheckBox, QRadioButton {{
                spacing: {s.item_spacing}px;
            }}
            
            /* Sliders */
            QSlider::handle {{
                width: {s.icon_size_medium}px;
                height: {s.icon_size_medium}px;
            }}
        """
    
    def get_spacing(self) -> int:
        """Get item spacing for layouts."""
        return self._settings.item_spacing
    
    def get_margin(self) -> int:
        """Get widget margin."""
        return self._settings.widget_margin
    
    def get_icon_size(self, size: str = "medium") -> int:
        """Get icon size.
        
        Args:
            size: "small", "medium", or "large"
        """
        sizes = {
            "small": self._settings.icon_size_small,
            "medium": self._settings.icon_size_medium,
            "large": self._settings.icon_size_large
        }
        return sizes.get(size, self._settings.icon_size_medium)
    
    def apply_to_layout(self, layout):
        """Apply density settings to a layout.
        
        Args:
            layout: QLayout to configure
        """
        layout.setSpacing(self._settings.item_spacing)
        layout.setContentsMargins(
            self._settings.widget_margin,
            self._settings.widget_margin,
            self._settings.widget_margin,
            self._settings.widget_margin
        )
    
    def on_change(self, callback: Callable[[DensitySettings], None]):
        """Register callback for density changes."""
        self._callbacks.append(callback)
    
    def _notify(self):
        """Notify all listeners of density change."""
        self.density_changed.emit(self._settings)
        for cb in self._callbacks:
            try:
                cb(self._settings)
            except Exception as e:
                logger.error(f"Density callback error: {e}")


def get_density_manager() -> DensityManager:
    """Get the density manager singleton."""
    return DensityManager()


def apply_density_to_widget(widget: QWidget, manager: Optional[DensityManager] = None):
    """Apply current density settings to a widget.
    
    Args:
        widget: Widget to configure
        manager: DensityManager instance (uses singleton if None)
    """
    if manager is None:
        manager = get_density_manager()
    
    # Apply stylesheet
    current = widget.styleSheet()
    density_css = manager.get_stylesheet()
    widget.setStyleSheet(current + density_css)
    
    # Apply to child layouts
    if hasattr(widget, 'layout') and widget.layout():
        manager.apply_to_layout(widget.layout())


__all__ = [
    'DensityMode',
    'DensitySettings',
    'DensityManager',
    'DENSITY_PRESETS',
    'get_density_manager',
    'apply_density_to_widget'
]
