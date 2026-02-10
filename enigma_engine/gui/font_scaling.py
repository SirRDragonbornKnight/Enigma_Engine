"""
Font Size Scaling for Enigma AI Engine

UI-wide zoom and font size scaling for accessibility.

Features:
- Global font scale (50% to 200%)
- Keyboard shortcuts (Ctrl++, Ctrl+-, Ctrl+0)
- Per-widget scaling
- DPI awareness
- Persistence across sessions

Usage:
    from enigma_engine.gui.font_scaling import FontScaler, get_scaler
    
    # Apply to QApplication
    scaler = FontScaler(app)
    scaler.set_scale(1.2)  # 120%
    
    # In a window
    scaler.apply_to_widget(main_window)
    
    # Quick zoom
    scaler.zoom_in()   # +10%
    scaler.zoom_out()  # -10%
    scaler.reset()     # 100%
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Qt imports
try:
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QMainWindow, QMenu, QAction
    )
    from PyQt5.QtGui import QFont, QKeySequence
    from PyQt5.QtCore import Qt, QObject, pyqtSignal
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


# Settings file
SETTINGS_DIR = Path.home() / ".enigma_engine"
SCALE_FILE = SETTINGS_DIR / "font_scale.json"


@dataclass
class ScaleSettings:
    """Font scaling settings."""
    # Global scale (1.0 = 100%)
    global_scale: float = 1.0
    
    # Per-element type scales (multiplied with global)
    element_scales: Dict[str, float] = None
    
    # Min/max bounds
    min_scale: float = 0.5
    max_scale: float = 2.5
    
    # Step size for zoom in/out
    step: float = 0.1
    
    # Base sizes
    base_font_size: int = 10
    
    def __post_init__(self):
        if self.element_scales is None:
            self.element_scales = {}
    
    def get_effective_scale(self, element_type: str = "") -> float:
        """Get effective scale for an element type."""
        type_scale = self.element_scales.get(element_type, 1.0)
        return self.global_scale * type_scale
    
    def get_font_size(self, element_type: str = "") -> int:
        """Get scaled font size."""
        scale = self.get_effective_scale(element_type)
        return max(6, int(self.base_font_size * scale))


if QT_AVAILABLE:
    class FontScaler(QObject):
        """
        UI-wide font scaling manager.
        """
        
        # Signals
        scale_changed = pyqtSignal(float)  # Emits new scale
        
        # Default element type scales
        DEFAULT_ELEMENT_SCALES = {
            "title": 1.6,
            "heading": 1.3,
            "subheading": 1.15,
            "body": 1.0,
            "small": 0.85,
            "caption": 0.75,
            "button": 1.0,
            "input": 1.0,
            "menu": 0.95,
            "status": 0.9,
            "tooltip": 0.85,
            "code": 0.95,
        }
        
        def __init__(
            self,
            app: Optional[QApplication] = None,
            settings: Optional[ScaleSettings] = None
        ):
            """
            Initialize font scaler.
            
            Args:
                app: QApplication instance
                settings: Custom scale settings
            """
            super().__init__()
            
            self._app = app or QApplication.instance()
            self._settings = settings or ScaleSettings()
            self._widgets: List[QWidget] = []
            self._original_fonts: Dict[int, QFont] = {}
            
            # Load saved settings
            self._load_settings()
            
            # Apply initial scale
            if self._app:
                self._apply_app_scale()
        
        @property
        def scale(self) -> float:
            """Current global scale."""
            return self._settings.global_scale
        
        @scale.setter
        def scale(self, value: float):
            """Set global scale."""
            self.set_scale(value)
        
        def set_scale(self, scale: float, save: bool = True):
            """
            Set the global font scale.
            
            Args:
                scale: Scale factor (1.0 = 100%)
                save: Save to settings file
            """
            # Clamp to bounds
            scale = max(self._settings.min_scale, min(self._settings.max_scale, scale))
            
            if scale == self._settings.global_scale:
                return
            
            self._settings.global_scale = scale
            
            # Apply to app
            if self._app:
                self._apply_app_scale()
            
            # Apply to tracked widgets
            for widget in self._widgets:
                if widget:
                    self._apply_widget_scale(widget)
            
            # Save
            if save:
                self._save_settings()
            
            # Emit signal
            self.scale_changed.emit(scale)
            
            logger.info(f"Font scale set to {scale:.0%}")
        
        def zoom_in(self):
            """Increase font size by step."""
            self.set_scale(self._settings.global_scale + self._settings.step)
        
        def zoom_out(self):
            """Decrease font size by step."""
            self.set_scale(self._settings.global_scale - self._settings.step)
        
        def reset(self):
            """Reset to 100% scale."""
            self.set_scale(1.0)
        
        def set_element_scale(self, element_type: str, scale: float):
            """
            Set scale for a specific element type.
            
            Args:
                element_type: Type name (title, heading, body, etc.)
                scale: Scale relative to global
            """
            self._settings.element_scales[element_type] = scale
            self._save_settings()
        
        def _apply_app_scale(self):
            """Apply scale to the application."""
            if not self._app:
                return
            
            # Get base font
            font = self._app.font()
            
            # Calculate new size
            new_size = self._settings.get_font_size()
            font.setPointSize(new_size)
            
            # Apply to application
            self._app.setFont(font)
            
            # Force all widgets to update
            for widget in self._app.allWidgets():
                widget.setFont(font)
        
        def apply_to_widget(self, widget: QWidget, element_type: str = ""):
            """
            Apply scaling to a specific widget.
            
            Args:
                widget: Widget to scale
                element_type: Optional type for different scaling
            """
            # Store original font
            widget_id = id(widget)
            if widget_id not in self._original_fonts:
                self._original_fonts[widget_id] = QFont(widget.font())
            
            # Track widget
            if widget not in self._widgets:
                self._widgets.append(widget)
            
            # Apply scale
            self._apply_widget_scale(widget, element_type)
        
        def _apply_widget_scale(self, widget: QWidget, element_type: str = ""):
            """Apply current scale to a widget."""
            widget_id = id(widget)
            
            # Get original or current font
            if widget_id in self._original_fonts:
                base_font = QFont(self._original_fonts[widget_id])
            else:
                base_font = QFont(widget.font())
            
            # Determine element type from widget class if not specified
            if not element_type:
                element_type = self._get_widget_type(widget)
            
            # Calculate size
            scale = self._settings.get_effective_scale(element_type)
            new_size = max(6, int(base_font.pointSize() * scale))
            
            # Apply
            base_font.setPointSize(new_size)
            widget.setFont(base_font)
        
        def _get_widget_type(self, widget: QWidget) -> str:
            """Determine element type from widget class."""
            widget_class = type(widget).__name__
            
            type_mapping = {
                'QLabel': 'body',
                'QPushButton': 'button',
                'QLineEdit': 'input',
                'QTextEdit': 'body',
                'QPlainTextEdit': 'code',
                'QComboBox': 'input',
                'QSpinBox': 'input',
                'QCheckBox': 'body',
                'QRadioButton': 'body',
                'QGroupBox': 'subheading',
                'QMenuBar': 'menu',
                'QMenu': 'menu',
                'QStatusBar': 'status',
                'QToolBar': 'button',
            }
            
            return type_mapping.get(widget_class, 'body')
        
        def apply_to_window(self, window: QMainWindow):
            """
            Apply scaling to all widgets in a window.
            
            Args:
                window: Window to scale
            """
            # Apply to window itself
            self.apply_to_widget(window)
            
            # Apply to all children recursively
            self._apply_recursive(window)
        
        def _apply_recursive(self, widget: QWidget):
            """Recursively apply to all children."""
            for child in widget.findChildren(QWidget):
                self.apply_to_widget(child)
        
        def install_shortcuts(self, window: QWidget):
            """
            Install keyboard shortcuts for zooming.
            
            Ctrl++ : Zoom in
            Ctrl+- : Zoom out
            Ctrl+0 : Reset
            """
            # Zoom in
            zoom_in_action = QAction("Zoom In", window)
            zoom_in_action.setShortcuts([
                QKeySequence.ZoomIn,
                QKeySequence(Qt.CTRL + Qt.Key_Equal),
                QKeySequence(Qt.CTRL + Qt.Key_Plus),
            ])
            zoom_in_action.triggered.connect(self.zoom_in)
            window.addAction(zoom_in_action)
            
            # Zoom out
            zoom_out_action = QAction("Zoom Out", window)
            zoom_out_action.setShortcuts([
                QKeySequence.ZoomOut,
                QKeySequence(Qt.CTRL + Qt.Key_Minus),
            ])
            zoom_out_action.triggered.connect(self.zoom_out)
            window.addAction(zoom_out_action)
            
            # Reset
            reset_action = QAction("Reset Zoom", window)
            reset_action.setShortcut(QKeySequence(Qt.CTRL + Qt.Key_0))
            reset_action.triggered.connect(self.reset)
            window.addAction(reset_action)
            
            logger.debug("Installed zoom shortcuts")
        
        def get_scaled_stylesheet(
            self,
            base_stylesheet: str,
            element_type: str = ""
        ) -> str:
            """
            Scale font sizes in a stylesheet.
            
            Args:
                base_stylesheet: Original stylesheet
                element_type: Element type for scaling
                
            Returns:
                Scaled stylesheet
            """
            import re
            
            scale = self._settings.get_effective_scale(element_type)
            
            def scale_font_size(match):
                size = int(match.group(1))
                new_size = max(6, int(size * scale))
                return f"font-size: {new_size}px"
            
            def scale_point_size(match):
                size = int(match.group(1))
                new_size = max(6, int(size * scale))
                return f"font-size: {new_size}pt"
            
            # Scale px font sizes
            scaled = re.sub(
                r'font-size:\s*(\d+)px',
                scale_font_size,
                base_stylesheet,
                flags=re.IGNORECASE
            )
            
            # Scale pt font sizes
            scaled = re.sub(
                r'font-size:\s*(\d+)pt',
                scale_point_size,
                scaled,
                flags=re.IGNORECASE
            )
            
            return scaled
        
        def create_zoom_menu(self, parent: QWidget = None) -> QMenu:
            """Create a zoom control menu."""
            menu = QMenu("Zoom", parent)
            
            # Zoom in
            zoom_in = menu.addAction("Zoom In\tCtrl++")
            zoom_in.triggered.connect(self.zoom_in)
            
            # Zoom out
            zoom_out = menu.addAction("Zoom Out\tCtrl+-")
            zoom_out.triggered.connect(self.zoom_out)
            
            menu.addSeparator()
            
            # Reset
            reset = menu.addAction("Reset\tCtrl+0")
            reset.triggered.connect(self.reset)
            
            menu.addSeparator()
            
            # Preset scales
            presets = [
                ("75%", 0.75),
                ("100%", 1.0),
                ("125%", 1.25),
                ("150%", 1.5),
                ("200%", 2.0),
            ]
            
            for label, scale in presets:
                action = menu.addAction(label)
                action.triggered.connect(lambda checked, s=scale: self.set_scale(s))
            
            return menu
        
        def _load_settings(self):
            """Load settings from file."""
            try:
                if SCALE_FILE.exists():
                    with open(SCALE_FILE) as f:
                        data = json.load(f)
                    self._settings.global_scale = data.get("global_scale", 1.0)
                    self._settings.element_scales = data.get("element_scales", {})
                    logger.debug(f"Loaded font scale: {self._settings.global_scale}")
            except Exception as e:
                logger.warning(f"Failed to load scale settings: {e}")
        
        def _save_settings(self):
            """Save settings to file."""
            try:
                SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
                with open(SCALE_FILE, 'w') as f:
                    json.dump({
                        "global_scale": self._settings.global_scale,
                        "element_scales": self._settings.element_scales,
                    }, f)
            except Exception as e:
                logger.warning(f"Failed to save scale settings: {e}")
        
        def get_dpi_scale(self) -> float:
            """Get system DPI scale factor."""
            if not self._app:
                return 1.0
            
            # Get primary screen
            screen = self._app.primaryScreen()
            if screen:
                return screen.devicePixelRatio()
            
            return 1.0
        
        def auto_scale_for_dpi(self):
            """Automatically adjust scale based on DPI."""
            dpi_scale = self.get_dpi_scale()
            
            # High DPI might need less scaling since system already scales
            if dpi_scale > 1.5:
                self.set_scale(1.0)
            elif dpi_scale > 1.0:
                self.set_scale(1.1)
            else:
                # Standard DPI - keep current or use 1.0
                pass


# Global instance
_scaler: Optional['FontScaler'] = None


def get_scaler(app: Optional[QApplication] = None) -> 'FontScaler':
    """Get or create the global font scaler."""
    global _scaler
    if _scaler is None:
        _scaler = FontScaler(app)
    return _scaler


def set_font_scale(scale: float):
    """Quick function to set font scale."""
    scaler = get_scaler()
    scaler.set_scale(scale)


def zoom_in():
    """Quick zoom in."""
    get_scaler().zoom_in()


def zoom_out():
    """Quick zoom out."""
    get_scaler().zoom_out()


def reset_zoom():
    """Quick reset zoom."""
    get_scaler().reset()


def install_zoom_shortcuts(window):
    """Install zoom shortcuts on a window."""
    get_scaler().install_shortcuts(window)
