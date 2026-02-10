"""
================================================================================
Unified Tab Patterns - Standard patterns for consistent GUI behavior.
================================================================================

This module provides:
- Device-aware styling (Pi to 4K monitor)
- Standard patterns for all generation tabs
- Memory-efficient preview handling
- Consistent color theming
- Unified worker base class

USAGE:
    from enigma_engine.gui.tabs.unified_patterns import (
        get_style_config, get_button_style, Colors,
        UnifiedWorker, create_styled_button
    )
    
    # Get device-aware button style
    btn = QPushButton("Generate")
    btn.setStyleSheet(get_button_style('primary'))

All generation tabs should use these patterns for consistency.
"""

import gc
import logging
import os
from pathlib import Path
from typing import Optional

try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QGroupBox,
        QPushButton,
        QSpinBox,
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

logger = logging.getLogger(__name__)


# =============================================================================
# Device-Aware Style Configuration
# =============================================================================

class DeviceUIClass:
    """Device UI classification for adaptive layouts."""
    EMBEDDED = "embedded"      # Pi, small screens (< 800px)
    MOBILE = "mobile"          # Phones, tablets (800-1200px)
    DESKTOP = "desktop"        # Standard monitors (1200-2560px)
    HIGHRES = "highres"        # 4K and above (> 2560px)


class StyleConfig:
    """
    Device-aware style configuration.
    
    Adjusts UI elements based on device capabilities:
    - Smaller fonts/buttons on low-res screens
    - Larger touch targets on mobile
    - Reduced animations on low-power devices
    - Memory-efficient preview sizes
    - Adaptive spacing and margins
    """
    
    def __init__(self):
        self._device_class = None
        self._ui_class = None
        self._screen_dpi = 96
        self._screen_width = 1920
        self._screen_height = 1080
        self._is_touch = False
        self._detect_device()
    
    def _detect_device(self):
        """Detect device class, screen properties, and UI class."""
        # Hardware detection
        try:
            from ...core.device_profiles import get_device_profiler
            profiler = get_device_profiler()
            self._device_class = profiler.classify()
        except ImportError:
            self._device_class = None
        
        # Screen detection
        if HAS_PYQT:
            try:
                app = QApplication.instance()
                if app:
                    screen = app.primaryScreen()
                    if screen:
                        self._screen_dpi = screen.logicalDotsPerInch()
                        size = screen.size()
                        self._screen_width = size.width()
                        self._screen_height = size.height()
            except Exception:
                pass
        
        # Determine UI class based on screen width
        if self._screen_width < 800:
            self._ui_class = DeviceUIClass.EMBEDDED
        elif self._screen_width < 1200:
            self._ui_class = DeviceUIClass.MOBILE
        elif self._screen_width < 2560:
            self._ui_class = DeviceUIClass.DESKTOP
        else:
            self._ui_class = DeviceUIClass.HIGHRES
        
        # Touch detection (basic heuristic)
        self._is_touch = self._ui_class in {DeviceUIClass.EMBEDDED, DeviceUIClass.MOBILE}
    
    @property
    def is_low_power(self) -> bool:
        """Check if running on low-power device."""
        if self._device_class is None:
            return False
        try:
            from ...core.device_profiles import DeviceClass
            return self._device_class in {
                DeviceClass.EMBEDDED,
                DeviceClass.MOBILE,
                DeviceClass.LAPTOP_LOW,
            }
        except ImportError:
            return False
    
    @property
    def is_embedded(self) -> bool:
        """Check if running on embedded device (Pi, etc.)."""
        if self._device_class is None:
            return False
        try:
            from ...core.device_profiles import DeviceClass
            return self._device_class == DeviceClass.EMBEDDED
        except ImportError:
            return False
    
    @property
    def ui_class(self) -> str:
        """Get UI classification for layout decisions."""
        return self._ui_class or DeviceUIClass.DESKTOP
    
    @property
    def screen_width(self) -> int:
        """Get screen width in pixels."""
        return self._screen_width
    
    @property
    def screen_height(self) -> int:
        """Get screen height in pixels."""
        return self._screen_height
    
    @property
    def is_touch(self) -> bool:
        """Check if device likely has touch input."""
        return self._is_touch
    
    @property
    def base_font_size(self) -> int:
        """Get appropriate base font size."""
        if self._ui_class == DeviceUIClass.EMBEDDED:
            return 9
        elif self._ui_class == DeviceUIClass.MOBILE:
            return 11
        elif self._ui_class == DeviceUIClass.HIGHRES:
            return 13
        return 11  # Desktop default
    
    @property
    def header_font_size(self) -> int:
        """Get header font size."""
        return self.base_font_size + 4
    
    @property
    def button_padding(self) -> str:
        """Get button padding CSS."""
        if self._is_touch:
            return "10px 16px"  # Larger for touch
        if self.is_low_power:
            return "4px 8px"
        return "8px 16px"
    
    @property
    def min_button_height(self) -> int:
        """Minimum button height for touch targets."""
        if self._is_touch:
            return 44  # Apple HIG recommendation
        return 30
    
    @property
    def preview_max_size(self) -> int:
        """Max size for image previews."""
        if self._ui_class == DeviceUIClass.EMBEDDED:
            return 200
        elif self._ui_class == DeviceUIClass.MOBILE:
            return 300
        elif self._ui_class == DeviceUIClass.HIGHRES:
            return 800
        return 512  # Desktop
    
    @property
    def enable_animations(self) -> bool:
        """Whether to enable UI animations."""
        return not self.is_low_power
    
    @property
    def spacing(self) -> int:
        """Standard spacing between elements."""
        if self._ui_class == DeviceUIClass.EMBEDDED:
            return 4
        elif self._ui_class == DeviceUIClass.MOBILE:
            return 6
        return 8
    
    @property
    def margins(self) -> int:
        """Standard margins for layouts."""
        if self._ui_class == DeviceUIClass.EMBEDDED:
            return 4
        elif self._ui_class == DeviceUIClass.MOBILE:
            return 6
        return 10
    
    @property
    def border_radius(self) -> int:
        """Standard border radius."""
        if self._ui_class == DeviceUIClass.EMBEDDED:
            return 3
        return 6
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for generation."""
        if self.is_embedded:
            return 1
        if self.is_low_power:
            return 2
        return 4


# Singleton style config
_style_config: Optional[StyleConfig] = None

def get_style_config() -> StyleConfig:
    """Get the global style configuration."""
    global _style_config
    if _style_config is None:
        _style_config = StyleConfig()
    return _style_config


# =============================================================================
# Unified Color Palette
# =============================================================================

class Colors:
    """
    Catppuccin-inspired color palette.
    
    These colors work well in both light and dark themes.
    """
    # Background colors
    BG_PRIMARY = "#1e1e2e"
    BG_SECONDARY = "#313244"
    BG_TERTIARY = "#45475a"
    
    # Text colors
    TEXT_PRIMARY = "#cdd6f4"
    TEXT_SECONDARY = "#a6adc8"
    TEXT_MUTED = "#6c7086"
    
    # Accent colors
    ACCENT_BLUE = "#89b4fa"
    ACCENT_GREEN = "#a6e3a1"
    ACCENT_RED = "#f38ba8"
    ACCENT_YELLOW = "#f9e2af"
    ACCENT_PURPLE = "#cba6f7"
    ACCENT_TEAL = "#94e2d5"
    
    # Semantic colors
    SUCCESS = ACCENT_GREEN
    ERROR = ACCENT_RED
    WARNING = ACCENT_YELLOW
    INFO = ACCENT_BLUE


# =============================================================================
# Unified Style Sheets
# =============================================================================

def get_button_style(style_type: str = "primary") -> str:
    """Get consistent button stylesheet."""
    config = get_style_config()
    padding = config.button_padding
    font_size = config.base_font_size
    
    styles = {
        "primary": f"""
            QPushButton {{
                background-color: {Colors.ACCENT_BLUE};
                color: {Colors.BG_PRIMARY};
                border: none;
                border-radius: 6px;
                padding: {padding};
                font-weight: bold;
                font-size: {font_size}px;
            }}
            QPushButton:hover {{
                background-color: #7aa2f7;
            }}
            QPushButton:pressed {{
                background-color: #5d87e0;
            }}
            QPushButton:disabled {{
                background-color: {Colors.BG_SECONDARY};
                color: {Colors.ERROR};
                border: 2px dashed {Colors.ERROR};
            }}
        """,
        "secondary": f"""
            QPushButton {{
                background-color: {Colors.ACCENT_BLUE};
                color: {Colors.BG_PRIMARY};
                border: none;
                border-radius: 6px;
                padding: {padding};
                font-weight: bold;
                font-size: {font_size}px;
            }}
            QPushButton:hover {{
                background-color: #b4befe;
            }}
            QPushButton:pressed {{
                background-color: #74c7ec;
            }}
            QPushButton:disabled {{
                background-color: {Colors.BG_SECONDARY};
                color: {Colors.ERROR};
                border: 2px dashed {Colors.ERROR};
            }}
        """,
        "success": f"""
            QPushButton {{
                background-color: {Colors.SUCCESS};
                color: {Colors.BG_PRIMARY};
                border: none;
                border-radius: 6px;
                padding: {padding};
                font-weight: bold;
                font-size: {font_size}px;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_TEAL};
            }}
            QPushButton:pressed {{
                background-color: #74c7a0;
            }}
            QPushButton:disabled {{
                background-color: {Colors.BG_SECONDARY};
                color: {Colors.ERROR};
                border: 2px dashed {Colors.ERROR};
            }}
        """,
        "danger": f"""
            QPushButton {{
                background-color: {Colors.ERROR};
                color: {Colors.BG_PRIMARY};
                border: none;
                border-radius: 6px;
                padding: {padding};
                font-weight: bold;
                font-size: {font_size}px;
            }}
            QPushButton:hover {{
                background-color: #f5c2e7;
            }}
            QPushButton:pressed {{
                background-color: #d06080;
            }}
            QPushButton:disabled {{
                background-color: {Colors.BG_SECONDARY};
                color: {Colors.ERROR};
                border: 2px dashed {Colors.ERROR};
            }}
        """,
    }
    
    return styles.get(style_type, styles["primary"])


def get_header_style() -> str:
    """Get header label stylesheet."""
    config = get_style_config()
    return f"""
        QLabel {{
            font-size: {config.base_font_size + 4}px;
            font-weight: bold;
            color: {Colors.TEXT_PRIMARY};
            padding: 4px 0;
        }}
    """


def get_status_style() -> str:
    """Get status label stylesheet."""
    config = get_style_config()
    return f"""
        QLabel {{
            color: {Colors.TEXT_SECONDARY};
            font-size: {config.base_font_size}px;
            padding: 2px 4px;
        }}
    """


def get_progress_style() -> str:
    """Get progress bar stylesheet."""
    return f"""
        QProgressBar {{
            border: 1px solid {Colors.BG_TERTIARY};
            border-radius: 4px;
            background-color: {Colors.BG_SECONDARY};
            height: 8px;
            text-align: center;
        }}
        QProgressBar::chunk {{
            background-color: {Colors.ACCENT_BLUE};
            border-radius: 3px;
        }}
    """


def get_preview_style() -> str:
    """Get preview area stylesheet."""
    return f"""
        QLabel {{
            background-color: {Colors.BG_SECONDARY};
            border: 1px solid {Colors.BG_TERTIARY};
            border-radius: 4px;
            padding: 8px;
        }}
    """


def get_group_style() -> str:
    """Get group box stylesheet."""
    config = get_style_config()
    return f"""
        QGroupBox {{
            font-weight: bold;
            font-size: {config.base_font_size}px;
            border: 1px solid {Colors.BG_TERTIARY};
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 8px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: {Colors.ACCENT_BLUE};
        }}
    """


def get_input_style() -> str:
    """Get text input stylesheet."""
    config = get_style_config()
    return f"""
        QTextEdit, QLineEdit {{
            background-color: {Colors.BG_SECONDARY};
            color: {Colors.TEXT_PRIMARY};
            border: 1px solid {Colors.BG_TERTIARY};
            border-radius: 4px;
            padding: 4px;
            font-size: {config.base_font_size}px;
        }}
        QTextEdit:focus, QLineEdit:focus {{
            border-color: {Colors.ACCENT_BLUE};
        }}
    """


# =============================================================================
# Memory-Efficient Preview Handling
# =============================================================================

class PreviewManager:
    """
    Memory-efficient preview image management.
    
    - Scales images to reasonable preview sizes
    - Caches pixmaps for quick redisplay
    - Clears cache on low memory
    """
    
    def __init__(self, max_cache_items: int = 5):
        self._cache: dict[str, QPixmap] = {}
        self._max_cache = max_cache_items
        self._access_order: list[str] = []
    
    def load_preview(
        self,
        path: str,
        max_width: int = None,
        max_height: int = None
    ) -> Optional['QPixmap']:
        """
        Load an image as a scaled preview.
        
        Args:
            path: Path to image file
            max_width: Maximum width (default: from style config)
            max_height: Maximum height (default: same as width)
            
        Returns:
            QPixmap scaled to fit within bounds
        """
        if not HAS_PYQT:
            return None
        
        config = get_style_config()
        max_width = max_width or config.preview_max_size
        max_height = max_height or max_width
        
        cache_key = f"{path}_{max_width}x{max_height}"
        
        # Check cache
        if cache_key in self._cache:
            # Move to end of access order
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            return self._cache[cache_key]
        
        # Load and scale
        try:
            pixmap = QPixmap(path)
            if pixmap.isNull():
                return None
            
            scaled = pixmap.scaled(
                max_width, max_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # Add to cache
            self._cache[cache_key] = scaled
            self._access_order.append(cache_key)
            
            # Prune cache if needed
            while len(self._cache) > self._max_cache:
                oldest = self._access_order.pop(0)
                if oldest in self._cache:
                    del self._cache[oldest]
            
            return scaled
            
        except Exception as e:
            logger.warning(f"Could not load preview {path}: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached previews."""
        self._cache.clear()
        self._access_order.clear()
        gc.collect()


# Singleton preview manager
_preview_manager: Optional[PreviewManager] = None

def get_preview_manager() -> PreviewManager:
    """Get the global preview manager."""
    global _preview_manager
    if _preview_manager is None:
        _preview_manager = PreviewManager()
    return _preview_manager


# =============================================================================
# Unified Worker Base
# =============================================================================

if HAS_PYQT:
    class UnifiedWorker(QThread):
        """
        Base worker class with proper cleanup and cancellation.
        
        All generation workers should inherit from this.
        """
        finished = pyqtSignal(dict)
        progress = pyqtSignal(int)
        status = pyqtSignal(str)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self._stop_requested = False
            self._start_time = 0.0
        
        def request_stop(self):
            """Request graceful stop."""
            self._stop_requested = True
        
        def is_stopped(self) -> bool:
            """Check if stop was requested."""
            return self._stop_requested
        
        def start_timer(self):
            """Start timing the operation."""
            import time
            self._start_time = time.time()
        
        def get_duration(self) -> float:
            """Get elapsed time since start_timer()."""
            import time
            return time.time() - self._start_time
        
        def emit_success(self, path: str = "", extra: dict = None):
            """Emit a successful result."""
            result = {
                "success": True,
                "path": path,
                "duration": self.get_duration(),
            }
            if extra:
                result.update(extra)
            self.finished.emit(result)
        
        def emit_error(self, error: str):
            """Emit an error result."""
            self.finished.emit({
                "success": False,
                "error": error,
                "duration": self.get_duration(),
            })
        
        def emit_cancelled(self):
            """Emit a cancelled result."""
            self.finished.emit({
                "success": False,
                "error": "Cancelled by user",
                "cancelled": True,
                "duration": self.get_duration(),
            })
        
        def run(self):
            """Override this in subclasses."""
            raise NotImplementedError("Subclasses must implement run()")
        
        def cleanup(self):
            """
            Override this for cleanup after run completes.
            Called automatically via finished signal if connected properly.
            """
            gc.collect()


# =============================================================================
# Utility Functions
# =============================================================================

def create_styled_button(
    text: str,
    style: str = "primary",
    tooltip: str = "",
    min_width: int = 80,
) -> 'QPushButton':
    """Create a consistently styled button."""
    if not HAS_PYQT:
        return None
    
    btn = QPushButton(text)
    btn.setStyleSheet(get_button_style(style))
    btn.setMinimumWidth(min_width)
    if tooltip:
        btn.setToolTip(tooltip)
    return btn


def create_styled_group(title: str) -> 'QGroupBox':
    """Create a consistently styled group box."""
    if not HAS_PYQT:
        return None
    
    group = QGroupBox(title)
    group.setStyleSheet(get_group_style())
    return group


def create_spinner(
    min_val: int,
    max_val: int,
    default: int,
    suffix: str = "",
    tooltip: str = "",
) -> 'QSpinBox':
    """Create a consistently styled spin box."""
    if not HAS_PYQT:
        return None
    
    spinner = QSpinBox()
    spinner.setRange(min_val, max_val)
    spinner.setValue(default)
    if suffix:
        spinner.setSuffix(suffix)
    if tooltip:
        spinner.setToolTip(tooltip)
    return spinner


def create_double_spinner(
    min_val: float,
    max_val: float,
    default: float,
    step: float = 0.1,
    decimals: int = 1,
    suffix: str = "",
    tooltip: str = "",
) -> 'QDoubleSpinBox':
    """Create a consistently styled double spin box."""
    if not HAS_PYQT:
        return None
    
    spinner = QDoubleSpinBox()
    spinner.setRange(min_val, max_val)
    spinner.setValue(default)
    spinner.setSingleStep(step)
    spinner.setDecimals(decimals)
    if suffix:
        spinner.setSuffix(suffix)
    if tooltip:
        spinner.setToolTip(tooltip)
    return spinner


def open_in_explorer(path: str):
    """Open file location in system file explorer."""
    import subprocess
    import sys
    
    path = str(path)
    
    if sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", path])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    else:
        subprocess.Popen(["xdg-open", str(Path(path).parent)])


def open_in_viewer(path: str):
    """Open file in default application."""
    import subprocess
    import sys
    
    path = str(path)
    
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def create_no_scroll_combo(
    items: list[str] = None,
    tooltip: str = "",
    min_width: int = 120,
) -> 'QComboBox':
    """
    Create a combo box that doesn't respond to scroll wheel.
    
    This prevents accidental value changes when scrolling the page.
    """
    if not HAS_PYQT:
        return None
    
    from .shared_components import NoScrollComboBox
    
    combo = NoScrollComboBox()
    if items:
        combo.addItems(items)
    if tooltip:
        combo.setToolTip(tooltip)
    combo.setMinimumWidth(min_width)
    return combo


def apply_device_aware_layout(layout, widget=None):
    """
    Apply device-aware spacing and margins to a layout.
    
    Args:
        layout: QLayout to configure
        widget: Optional widget to set margins on
    """
    config = get_style_config()
    
    layout.setSpacing(config.spacing)
    layout.setContentsMargins(
        config.margins, config.margins,
        config.margins, config.margins
    )
    
    if widget and HAS_PYQT:
        widget.setContentsMargins(
            config.margins, config.margins,
            config.margins, config.margins
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Device/UI classes
    'DeviceUIClass',
    
    # Style config
    'StyleConfig',
    'get_style_config',
    'Colors',
    
    # Style functions
    'get_button_style',
    'get_header_style',
    'get_status_style',
    'get_progress_style',
    'get_preview_style',
    'get_group_style',
    'get_input_style',
    
    # Preview management
    'PreviewManager',
    'get_preview_manager',
    
    # Worker base
    'UnifiedWorker',
    
    # Factory functions
    'create_styled_button',
    'create_styled_group',
    'create_spinner',
    'create_double_spinner',
    'create_no_scroll_combo',
    
    # Utilities
    'open_in_explorer',
    'open_in_viewer',
]
