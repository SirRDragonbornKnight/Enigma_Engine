"""
Shared UI Components - Reusable UI patterns from avatar tab.

Features:
- Preset selection with themes
- Color customization with picker
- Module state checking
- Overlay/popup windows
- Settings persistence
- NoScrollComboBox - combo that ignores scroll wheel
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QComboBox, QGroupBox, QColorDialog,
        QDialog, QSlider, QSpinBox, QCheckBox
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QEvent
    from PyQt5.QtGui import QColor, QCursor, QPixmap, QPainter
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


# =============================================================================
# NoScrollComboBox - Combo box that ignores scroll wheel events
# =============================================================================
class NoScrollComboBox(QComboBox):
    """A QComboBox that ignores scroll wheel events to prevent accidental changes.
    
    Use this instead of QComboBox when the dropdown shouldn't respond to scrolling,
    which is especially useful when combo boxes are in scrollable areas.
    """
    
    def wheelEvent(self, event):
        """Ignore scroll wheel events - user must click to change value."""
        event.ignore()


def disable_scroll_on_combos(parent_widget):
    """Install event filter on all QComboBox children to disable scroll wheel.
    
    Call this after creating a widget tree to disable scroll wheel on all combo boxes.
    
    Args:
        parent_widget: The parent widget containing combo boxes
    """
    if not HAS_PYQT:
        return
    
    class ScrollFilter(QWidget):
        def eventFilter(self, obj, event):
            if event.type() == QEvent.Wheel and isinstance(obj, QComboBox):
                event.ignore()
                return True
            return False
    
    scroll_filter = ScrollFilter(parent_widget)
    for combo in parent_widget.findChildren(QComboBox):
        combo.setFocusPolicy(Qt.StrongFocus)  # Only respond when focused
        combo.installEventFilter(scroll_filter)

from ...config import CONFIG

# Settings directory
SETTINGS_DIR = Path(CONFIG.get("information_dir", "information"))
SETTINGS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Preset System
# =============================================================================

# Built-in style presets for various generation types
STYLE_PRESETS = {
    "image": {
        "default": {"style": "balanced", "quality": "standard"},
        "cinematic": {"style": "cinematic", "quality": "high", "suffix": ", cinematic lighting, dramatic composition"},
        "anime": {"style": "anime", "quality": "standard", "suffix": ", anime style, vibrant colors"},
        "photorealistic": {"style": "photorealistic", "quality": "high", "suffix": ", photorealistic, 8k, detailed"},
        "artistic": {"style": "artistic", "quality": "standard", "suffix": ", artistic, painterly, impressionist"},
        "minimal": {"style": "minimal", "quality": "fast", "suffix": ", minimalist, clean"},
        "dark": {"style": "dark", "quality": "standard", "suffix": ", dark mood, moody lighting"},
        "vibrant": {"style": "vibrant", "quality": "standard", "suffix": ", vibrant colors, colorful"},
    },
    "video": {
        "default": {"fps": 8, "frames": 16, "motion": "normal"},
        "smooth": {"fps": 24, "frames": 48, "motion": "smooth"},
        "cinematic": {"fps": 24, "frames": 72, "motion": "slow"},
        "fast": {"fps": 15, "frames": 30, "motion": "fast"},
        "loop": {"fps": 12, "frames": 24, "motion": "loop"},
    },
    "audio": {
        "default": {"rate": 150, "volume": 1.0, "pitch": "normal"},
        "slow": {"rate": 100, "volume": 1.0, "pitch": "low"},
        "fast": {"rate": 200, "volume": 1.0, "pitch": "normal"},
        "whisper": {"rate": 120, "volume": 0.6, "pitch": "low"},
        "energetic": {"rate": 180, "volume": 1.0, "pitch": "high"},
        "calm": {"rate": 130, "volume": 0.8, "pitch": "low"},
    },
    "code": {
        "default": {"language": "python", "style": "clean"},
        "documented": {"language": "python", "style": "documented", "suffix": "# Add comprehensive docstrings"},
        "minimal": {"language": "python", "style": "minimal"},
        "typed": {"language": "python", "style": "typed", "suffix": "# Include type hints"},
        "tested": {"language": "python", "style": "tested", "suffix": "# Include unit tests"},
    },
}

# Color theme presets
COLOR_PRESETS = {
    "default": {"primary": "#6366f1", "secondary": "#8b5cf6", "accent": "#10b981"},
    "warm": {"primary": "#f59e0b", "secondary": "#ef4444", "accent": "#f97316"},
    "cool": {"primary": "#3b82f6", "secondary": "#06b6d4", "accent": "#8b5cf6"},
    "nature": {"primary": "#22c55e", "secondary": "#84cc16", "accent": "#14b8a6"},
    "sunset": {"primary": "#f97316", "secondary": "#ec4899", "accent": "#f59e0b"},
    "ocean": {"primary": "#0ea5e9", "secondary": "#06b6d4", "accent": "#3b82f6"},
    "fire": {"primary": "#ef4444", "secondary": "#f97316", "accent": "#f59e0b"},
    "dark": {"primary": "#6b7280", "secondary": "#4b5563", "accent": "#9ca3af"},
    "pastel": {"primary": "#c4b5fd", "secondary": "#fbcfe8", "accent": "#a5f3fc"},
    "forest": {"primary": "#166534", "secondary": "#15803d", "accent": "#4ade80"},
    "berry": {"primary": "#9333ea", "secondary": "#c026d3", "accent": "#ec4899"},
    "monochrome": {"primary": "#525252", "secondary": "#737373", "accent": "#a3a3a3"},
}


class PresetSelector(QWidget):
    """Dropdown for selecting style presets."""
    
    preset_changed = pyqtSignal(str, dict)  # preset_name, preset_data
    
    def __init__(self, category: str = "image", parent=None):
        super().__init__(parent)
        self.category = category
        self.presets = STYLE_PRESETS.get(category, {})
        self._custom_presets: Dict[str, dict] = {}
        
        self._setup_ui()
        self._load_custom_presets()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("Preset:"))
        
        self.combo = NoScrollComboBox()
        self.combo.setMinimumWidth(120)
        self.combo.setToolTip("Select a preset configuration")
        self._update_combo()
        self.combo.currentTextChanged.connect(self._on_selection)
        layout.addWidget(self.combo, stretch=1)
        
        # Save custom preset button
        self.save_btn = QPushButton("Save")
        self.save_btn.setFixedWidth(50)
        self.save_btn.setToolTip("Save current settings as preset")
        self.save_btn.clicked.connect(self._save_custom_preset)
        layout.addWidget(self.save_btn)
    
    def _update_combo(self):
        self.combo.clear()
        # Built-in presets
        for name in self.presets.keys():
            self.combo.addItem(name.title())
        # Custom presets
        if self._custom_presets:
            self.combo.insertSeparator(self.combo.count())
            for name in self._custom_presets.keys():
                self.combo.addItem(f"[Custom] {name}")
    
    def _on_selection(self, text: str):
        name = text.lower().replace("[custom] ", "")
        if name in self.presets:
            self.preset_changed.emit(name, self.presets[name])
        elif name in self._custom_presets:
            self.preset_changed.emit(name, self._custom_presets[name])
    
    def _save_custom_preset(self):
        from PyQt5.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if ok and name:
            # Emit signal so parent can provide current settings
            self.preset_changed.emit(f"__save__{name}", {})
    
    def add_custom_preset(self, name: str, data: dict):
        """Add a custom preset and persist it."""
        self._custom_presets[name] = data
        self._save_custom_presets()
        self._update_combo()
    
    def _load_custom_presets(self):
        path = SETTINGS_DIR / f"{self.category}_presets.json"
        if path.exists():
            try:
                with open(path) as f:
                    self._custom_presets = json.load(f)
            except Exception:
                pass
    
    def _save_custom_presets(self):
        path = SETTINGS_DIR / f"{self.category}_presets.json"
        try:
            with open(path, 'w') as f:
                json.dump(self._custom_presets, f, indent=2)
        except Exception:
            pass


class ColorCustomizer(QWidget):
    """Color customization panel with presets and individual pickers."""
    
    colors_changed = pyqtSignal(dict)  # {"primary": "#...", "secondary": "#...", "accent": "#..."}
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._colors = COLOR_PRESETS["default"].copy()
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Preset row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Theme:"))
        
        self.preset_combo = NoScrollComboBox()
        self.preset_combo.setToolTip("Select a color theme preset")
        self.preset_combo.addItems([p.title() for p in COLOR_PRESETS.keys()])
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_row.addWidget(self.preset_combo, stretch=1)
        layout.addLayout(preset_row)
        
        # Color buttons
        btn_row = QHBoxLayout()
        
        self.primary_btn = QPushButton("Primary")
        self.primary_btn.setStyleSheet(f"background: {self._colors['primary']}; color: white;")
        self.primary_btn.clicked.connect(lambda: self._pick_color("primary"))
        btn_row.addWidget(self.primary_btn)
        
        self.secondary_btn = QPushButton("Secondary")
        self.secondary_btn.setStyleSheet(f"background: {self._colors['secondary']}; color: white;")
        self.secondary_btn.clicked.connect(lambda: self._pick_color("secondary"))
        btn_row.addWidget(self.secondary_btn)
        
        self.accent_btn = QPushButton("Accent")
        self.accent_btn.setStyleSheet(f"background: {self._colors['accent']}; color: white;")
        self.accent_btn.clicked.connect(lambda: self._pick_color("accent"))
        btn_row.addWidget(self.accent_btn)
        
        layout.addLayout(btn_row)
    
    def _apply_preset(self, preset_name: str):
        name = preset_name.lower()
        if name in COLOR_PRESETS:
            self._colors = COLOR_PRESETS[name].copy()
            self._update_buttons()
            self.colors_changed.emit(self._colors)
    
    def _pick_color(self, which: str):
        current = QColor(self._colors.get(which, "#ffffff"))
        color = QColorDialog.getColor(current, self, f"Pick {which.title()} Color")
        if color.isValid():
            self._colors[which] = color.name()
            self._update_buttons()
            self.colors_changed.emit(self._colors)
    
    def _update_buttons(self):
        self.primary_btn.setStyleSheet(f"background: {self._colors['primary']}; color: white;")
        self.secondary_btn.setStyleSheet(f"background: {self._colors['secondary']}; color: white;")
        self.accent_btn.setStyleSheet(f"background: {self._colors['accent']}; color: white;")
    
    def get_colors(self) -> Dict[str, str]:
        return self._colors.copy()
    
    def set_colors(self, colors: Dict[str, str]):
        self._colors.update(colors)
        self._update_buttons()


class ModuleStateChecker:
    """Check if a module is enabled in ModuleManager.
    
    NOTE: For UI purposes, we typically want features enabled by default.
    Use is_actually_loaded() for operational checks where you need to know
    if a module is truly available.
    """
    
    @staticmethod
    def is_enabled(module_name: str) -> bool:
        """Check if module features should be enabled in UI.
        
        Returns True by default to allow tab functionality to work
        without requiring modules to be explicitly loaded.
        """
        # Default to True - tabs should work standalone
        return True
    
    @staticmethod
    def is_actually_loaded(module_name: str) -> bool:
        """Check if module is actually loaded (for operational checks)."""
        try:
            from ...modules import get_manager
            manager = get_manager()
            if manager:
                return manager.is_loaded(module_name)
        except Exception:
            pass
        return False
    
    @staticmethod
    def get_manager():
        """Get the ModuleManager instance."""
        try:
            from ...modules import get_manager
            return get_manager()
        except Exception:
            return None


class SettingsPersistence:
    """Save and load tab settings."""
    
    def __init__(self, tab_name: str):
        self.tab_name = tab_name
        self.path = SETTINGS_DIR / f"{tab_name}_settings.json"
    
    def save(self, settings: Dict[str, Any]):
        try:
            with open(self.path, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass
    
    def load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}


def create_settings_group(title: str, widgets: List[tuple]) -> QGroupBox:
    """Create a settings group box with label-widget pairs.
    
    Args:
        title: Group box title
        widgets: List of (label_text, widget) tuples
        
    Returns:
        QGroupBox with the widgets arranged
    """
    group = QGroupBox(title)
    layout = QVBoxLayout()
    
    for label_text, widget in widgets:
        row = QHBoxLayout()
        if label_text:
            row.addWidget(QLabel(label_text))
        row.addWidget(widget, stretch=1 if label_text else 0)
        layout.addLayout(row)
    
    group.setLayout(layout)
    return group


def create_action_button(text: str, callback: Callable, icon: str = "", 
                         tooltip: str = "", enabled: bool = True) -> QPushButton:
    """Create a styled action button."""
    btn = QPushButton(f"{icon} {text}".strip() if icon else text)
    btn.clicked.connect(callback)
    if tooltip:
        btn.setToolTip(tooltip)
    btn.setEnabled(enabled)
    return btn


# Floating overlay window (like avatar desktop pet)
if HAS_PYQT:
    # Cross-platform Qt flags
    Qt_FramelessWindowHint = getattr(Qt, 'FramelessWindowHint', 0x00000800)
    Qt_WindowStaysOnTopHint = getattr(Qt, 'WindowStaysOnTopHint', 0x00040000)
    Qt_Tool = getattr(Qt, 'Tool', 0x00000008)
    Qt_WA_TranslucentBackground = getattr(Qt, 'WA_TranslucentBackground', 32)
    Qt_LeftButton = getattr(Qt, 'LeftButton', 1)
    
    class FloatingOverlay(QWidget):
        """Transparent, draggable overlay window.
        
        Features:
        - Drag anywhere to move
        - Right-click menu
        - Scroll to resize
        - Always on top
        """
        
        closed = pyqtSignal()
        
        def __init__(self, size: int = 200):
            super().__init__(None)
            
            self.setWindowFlags(
                Qt_FramelessWindowHint |
                Qt_WindowStaysOnTopHint |
                Qt_Tool
            )
            self.setAttribute(Qt_WA_TranslucentBackground, True)
            
            self._size = size
            self.setFixedSize(self._size, self._size)
            self.move(100, 100)
            
            self._drag_pos = None
            self.content_widget = None
            
            self.setMouseTracking(True)
        
        def set_content(self, widget: QWidget):
            """Set the content widget."""
            if self.content_widget:
                self.content_widget.setParent(None)
            
            self.content_widget = widget
            widget.setParent(self)
            widget.setGeometry(10, 10, self._size - 20, self._size - 20)
        
        def mousePressEvent(self, event):
            if event.button() == Qt_LeftButton:
                self._drag_pos = event.globalPos() - self.pos()
                event.accept()
        
        def mouseMoveEvent(self, event):
            if self._drag_pos is not None and event.buttons() == Qt_LeftButton:
                self.move(event.globalPos() - self._drag_pos)
                event.accept()
        
        def mouseReleaseEvent(self, event):
            self._drag_pos = None
        
        def wheelEvent(self, event):
            delta = event.angleDelta().y()
            if delta > 0:
                self._size = min(800, self._size + 20)
            else:
                self._size = max(50, self._size - 20)
            
            self.setFixedSize(self._size, self._size)
            if self.content_widget:
                self.content_widget.setGeometry(10, 10, self._size - 20, self._size - 20)
            event.accept()
        
        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            
            # Semi-transparent background
            painter.setBrush(QColor(30, 30, 46, 200))
            painter.setPen(QColor(69, 71, 90))
            painter.drawRoundedRect(5, 5, self._size - 10, self._size - 10, 10, 10)


class DirectoryWatcher:
    """Watch a directory for file changes and trigger callbacks.
    
    Usage:
        watcher = DirectoryWatcher("/path/to/watch", [".png", ".jpg"])
        watcher.on_change(lambda: refresh_list())
        watcher.start()  # Starts checking every 3 seconds
        watcher.stop()   # Stop watching
    """
    
    def __init__(self, directory: Path, extensions: Optional[List[str]] = None, 
                 interval_ms: int = 3000, recursive: bool = True):
        from PyQt5.QtCore import QTimer
        
        self.directory = Path(directory)
        self.extensions = set(ext.lower() for ext in (extensions or []))
        self.interval = interval_ms
        self.recursive = recursive
        self._callbacks: List[Callable] = []
        self._last_state: Dict[str, float] = {}
        self._timer: Optional[QTimer] = None
    
    def on_change(self, callback: Callable):
        """Register a callback for when files change."""
        self._callbacks.append(callback)
    
    def start(self):
        """Start watching."""
        from PyQt5.QtCore import QTimer
        self._last_state = self._scan_files()
        self._timer = QTimer()
        self._timer.timeout.connect(self._check)
        self._timer.start(self.interval)
    
    def stop(self):
        """Stop watching."""
        if self._timer:
            self._timer.stop()
            self._timer = None
    
    def _scan_files(self) -> Dict[str, float]:
        """Scan directory and return file paths with modification times."""
        files = {}
        if not self.directory.exists():
            return files
        
        pattern = "**/*" if self.recursive else "*"
        for f in self.directory.glob(pattern):
            if f.is_file():
                if not self.extensions or f.suffix.lower() in self.extensions:
                    try:
                        files[str(f)] = f.stat().st_mtime
                    except OSError:
                        pass
        return files
    
    def _check(self):
        """Check for changes."""
        try:
            current = self._scan_files()
            if current != self._last_state:
                self._last_state = current
                for cb in self._callbacks:
                    try:
                        cb()
                    except Exception:
                        pass
        except Exception:
            pass


# Export all
__all__ = [
    'NoScrollComboBox',
    'disable_scroll_on_combos',
    'STYLE_PRESETS',
    'COLOR_PRESETS',
    'PresetSelector',
    'ColorCustomizer',
    'ModuleStateChecker',
    'SettingsPersistence',
    'create_settings_group',
    'create_action_button',
    'FloatingOverlay',
    'DirectoryWatcher',
]
