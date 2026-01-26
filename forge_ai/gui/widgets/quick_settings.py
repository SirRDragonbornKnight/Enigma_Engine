"""
================================================================================
Quick Settings Panel - Fast access to common settings.
================================================================================

A collapsible panel that provides quick access to:
- Gaming mode toggle
- Distributed mode settings  
- Resource usage limits
- Voice/Avatar toggles
- Performance presets

USAGE:
    from forge_ai.gui.widgets.quick_settings import QuickSettingsPanel
    
    panel = QuickSettingsPanel()
    layout.addWidget(panel)
    
    # Connect to changes
    panel.settings_changed.connect(on_settings_change)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QSlider, QCheckBox, QComboBox,
        QGroupBox, QFrame, QSizePolicy, QSpacerItem,
        QToolButton, QScrollArea
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve
    from PyQt5.QtGui import QFont, QIcon
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

logger = logging.getLogger(__name__)


# Style constants
PANEL_STYLE = """
    QGroupBox {
        background-color: #313244;
        border: 1px solid #45475a;
        border-radius: 8px;
        margin-top: 8px;
        padding: 8px;
        font-weight: bold;
        color: #cdd6f4;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }
"""

TOGGLE_STYLE = """
    QCheckBox {
        color: #cdd6f4;
        spacing: 8px;
    }
    QCheckBox::indicator {
        width: 36px;
        height: 20px;
        border-radius: 10px;
        background-color: #45475a;
    }
    QCheckBox::indicator:checked {
        background-color: #89b4fa;
    }
"""

SLIDER_STYLE = """
    QSlider::groove:horizontal {
        border: 1px solid #45475a;
        height: 6px;
        background: #313244;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #89b4fa;
        border: none;
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }
    QSlider::sub-page:horizontal {
        background: #89b4fa;
        border-radius: 3px;
    }
"""

PRESET_BUTTON_STYLE = """
    QPushButton {
        background-color: #45475a;
        color: #cdd6f4;
        border: none;
        border-radius: 4px;
        padding: 6px 12px;
        font-size: 11px;
    }
    QPushButton:hover {
        background-color: #585b70;
    }
    QPushButton:pressed {
        background-color: #313244;
    }
    QPushButton:checked {
        background-color: #89b4fa;
        color: #1e1e2e;
    }
"""


class QuickSettingsPanel(QWidget):
    """
    Quick settings panel with common toggles and sliders.
    """
    
    settings_changed = pyqtSignal(dict)  # Emitted when any setting changes
    
    def __init__(self, parent=None):
        if not HAS_PYQT:
            raise ImportError("PyQt5 required")
        
        super().__init__(parent)
        self._settings: Dict[str, Any] = {}
        self._collapsed = False
        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self):
        """Set up the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Header with collapse button
        header = QHBoxLayout()
        
        self.title_label = QLabel("Quick Settings")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #cdd6f4;")
        header.addWidget(self.title_label)
        
        header.addStretch()
        
        self.collapse_btn = QToolButton()
        self.collapse_btn.setText("-")
        self.collapse_btn.setStyleSheet("QToolButton { color: #cdd6f4; border: none; }")
        self.collapse_btn.clicked.connect(self._toggle_collapse)
        header.addWidget(self.collapse_btn)
        
        layout.addLayout(header)
        
        # Content area
        self.content = QWidget()
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        
        # Preset buttons
        content_layout.addWidget(self._create_presets_section())
        
        # Mode toggles
        content_layout.addWidget(self._create_toggles_section())
        
        # Resource sliders
        content_layout.addWidget(self._create_resources_section())
        
        layout.addWidget(self.content)
    
    def _create_presets_section(self) -> QWidget:
        """Create performance presets section."""
        group = QGroupBox("Presets")
        group.setStyleSheet(PANEL_STYLE)
        
        layout = QHBoxLayout(group)
        layout.setSpacing(4)
        
        presets = [
            ("Low Power", "embedded"),
            ("Balanced", "balanced"),
            ("Performance", "performance"),
            ("Gaming", "gaming"),
        ]
        
        self.preset_buttons = {}
        for name, preset_id in presets:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet(PRESET_BUTTON_STYLE)
            btn.clicked.connect(lambda checked, p=preset_id: self._apply_preset(p))
            self.preset_buttons[preset_id] = btn
            layout.addWidget(btn)
        
        return group
    
    def _create_toggles_section(self) -> QWidget:
        """Create mode toggles section."""
        group = QGroupBox("Modes")
        group.setStyleSheet(PANEL_STYLE)
        
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        
        # Gaming mode
        self.gaming_toggle = QCheckBox("Gaming Mode")
        self.gaming_toggle.setStyleSheet(TOGGLE_STYLE)
        self.gaming_toggle.setToolTip("Reduce AI resources when games are detected")
        self.gaming_toggle.toggled.connect(lambda v: self._on_setting_change("gaming_mode", v))
        layout.addWidget(self.gaming_toggle)
        
        # Distributed mode
        self.distributed_toggle = QCheckBox("Distributed Mode")
        self.distributed_toggle.setStyleSheet(TOGGLE_STYLE)
        self.distributed_toggle.setToolTip("Offload heavy tasks to remote servers")
        self.distributed_toggle.toggled.connect(lambda v: self._on_setting_change("distributed_mode", v))
        layout.addWidget(self.distributed_toggle)
        
        # Voice
        self.voice_toggle = QCheckBox("Voice Output")
        self.voice_toggle.setStyleSheet(TOGGLE_STYLE)
        self.voice_toggle.setToolTip("Enable text-to-speech")
        self.voice_toggle.toggled.connect(lambda v: self._on_setting_change("voice_enabled", v))
        layout.addWidget(self.voice_toggle)
        
        # Avatar
        self.avatar_toggle = QCheckBox("Avatar Display")
        self.avatar_toggle.setStyleSheet(TOGGLE_STYLE)
        self.avatar_toggle.setToolTip("Show desktop avatar")
        self.avatar_toggle.toggled.connect(lambda v: self._on_setting_change("avatar_enabled", v))
        layout.addWidget(self.avatar_toggle)
        
        return group
    
    def _create_resources_section(self) -> QWidget:
        """Create resource limits section."""
        group = QGroupBox("Resources")
        group.setStyleSheet(PANEL_STYLE)
        
        layout = QVBoxLayout(group)
        layout.setSpacing(12)
        
        # Max tokens
        tokens_layout = QHBoxLayout()
        tokens_layout.addWidget(QLabel("Max Tokens:"))
        self.tokens_label = QLabel("100")
        self.tokens_label.setMinimumWidth(40)
        self.tokens_label.setStyleSheet("color: #89b4fa;")
        tokens_layout.addWidget(self.tokens_label)
        
        self.tokens_slider = QSlider(Qt.Horizontal)
        self.tokens_slider.setStyleSheet(SLIDER_STYLE)
        self.tokens_slider.setRange(10, 500)
        self.tokens_slider.setValue(100)
        self.tokens_slider.valueChanged.connect(self._on_tokens_changed)
        tokens_layout.addWidget(self.tokens_slider)
        
        layout.addLayout(tokens_layout)
        
        # Temperature
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.temp_label = QLabel("0.8")
        self.temp_label.setMinimumWidth(40)
        self.temp_label.setStyleSheet("color: #89b4fa;")
        temp_layout.addWidget(self.temp_label)
        
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setStyleSheet(SLIDER_STYLE)
        self.temp_slider.setRange(0, 200)  # 0.0 to 2.0
        self.temp_slider.setValue(80)
        self.temp_slider.valueChanged.connect(self._on_temp_changed)
        temp_layout.addWidget(self.temp_slider)
        
        layout.addLayout(temp_layout)
        
        # CPU limit (for gaming mode)
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(QLabel("CPU Limit:"))
        self.cpu_label = QLabel("100%")
        self.cpu_label.setMinimumWidth(40)
        self.cpu_label.setStyleSheet("color: #89b4fa;")
        cpu_layout.addWidget(self.cpu_label)
        
        self.cpu_slider = QSlider(Qt.Horizontal)
        self.cpu_slider.setStyleSheet(SLIDER_STYLE)
        self.cpu_slider.setRange(10, 100)
        self.cpu_slider.setValue(100)
        self.cpu_slider.valueChanged.connect(self._on_cpu_changed)
        cpu_layout.addWidget(self.cpu_slider)
        
        layout.addLayout(cpu_layout)
        
        return group
    
    def _toggle_collapse(self):
        """Toggle panel collapse state."""
        self._collapsed = not self._collapsed
        self.content.setVisible(not self._collapsed)
        self.collapse_btn.setText("+" if self._collapsed else "-")
    
    def _apply_preset(self, preset_id: str):
        """Apply a performance preset."""
        # Uncheck other presets
        for pid, btn in self.preset_buttons.items():
            btn.setChecked(pid == preset_id)
        
        presets = {
            "embedded": {
                "gaming_mode": False,
                "distributed_mode": True,  # Offload to server
                "voice_enabled": True,
                "avatar_enabled": False,
                "max_tokens": 30,
                "temperature": 0.5,
                "cpu_limit": 50,
            },
            "balanced": {
                "gaming_mode": True,
                "distributed_mode": False,
                "voice_enabled": True,
                "avatar_enabled": True,
                "max_tokens": 100,
                "temperature": 0.8,
                "cpu_limit": 80,
            },
            "performance": {
                "gaming_mode": False,
                "distributed_mode": False,
                "voice_enabled": True,
                "avatar_enabled": True,
                "max_tokens": 200,
                "temperature": 0.8,
                "cpu_limit": 100,
            },
            "gaming": {
                "gaming_mode": True,
                "distributed_mode": True,  # Offload when gaming
                "voice_enabled": True,
                "avatar_enabled": False,  # Don't show overlay
                "max_tokens": 50,
                "temperature": 0.5,
                "cpu_limit": 30,
            },
        }
        
        if preset_id in presets:
            settings = presets[preset_id]
            self._apply_settings(settings)
            self._settings.update(settings)
            self._save_settings()
            self.settings_changed.emit(self._settings)
    
    def _apply_settings(self, settings: Dict[str, Any]):
        """Apply settings to UI controls."""
        if "gaming_mode" in settings:
            self.gaming_toggle.setChecked(settings["gaming_mode"])
        if "distributed_mode" in settings:
            self.distributed_toggle.setChecked(settings["distributed_mode"])
        if "voice_enabled" in settings:
            self.voice_toggle.setChecked(settings["voice_enabled"])
        if "avatar_enabled" in settings:
            self.avatar_toggle.setChecked(settings["avatar_enabled"])
        if "max_tokens" in settings:
            self.tokens_slider.setValue(settings["max_tokens"])
        if "temperature" in settings:
            self.temp_slider.setValue(int(settings["temperature"] * 100))
        if "cpu_limit" in settings:
            self.cpu_slider.setValue(settings["cpu_limit"])
    
    def _on_setting_change(self, key: str, value: Any):
        """Handle individual setting change."""
        self._settings[key] = value
        self._save_settings()
        self.settings_changed.emit(self._settings)
    
    def _on_tokens_changed(self, value: int):
        """Handle max tokens slider change."""
        self.tokens_label.setText(str(value))
        self._on_setting_change("max_tokens", value)
    
    def _on_temp_changed(self, value: int):
        """Handle temperature slider change."""
        temp = value / 100.0
        self.temp_label.setText(f"{temp:.1f}")
        self._on_setting_change("temperature", temp)
    
    def _on_cpu_changed(self, value: int):
        """Handle CPU limit slider change."""
        self.cpu_label.setText(f"{value}%")
        self._on_setting_change("cpu_limit", value)
    
    def _load_settings(self):
        """Load settings from file."""
        try:
            settings_file = Path.home() / ".forge_ai" / "quick_settings.json"
            if settings_file.exists():
                with open(settings_file) as f:
                    self._settings = json.load(f)
                    self._apply_settings(self._settings)
        except Exception as e:
            logger.warning(f"Could not load quick settings: {e}")
    
    def _save_settings(self):
        """Save settings to file."""
        try:
            settings_dir = Path.home() / ".forge_ai"
            settings_dir.mkdir(parents=True, exist_ok=True)
            
            with open(settings_dir / "quick_settings.json", "w") as f:
                json.dump(self._settings, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save quick settings: {e}")
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current settings."""
        return self._settings.copy()


class ResourceMonitor(QWidget):
    """
    Real-time resource usage monitor widget.
    
    Shows:
    - CPU usage
    - RAM usage
    - GPU/VRAM usage (if available)
    - Current AI mode
    """
    
    def __init__(self, parent=None):
        if not HAS_PYQT:
            raise ImportError("PyQt5 required")
        
        super().__init__(parent)
        self._setup_ui()
        self._start_monitoring()
    
    def _setup_ui(self):
        """Set up the UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)
        
        # CPU
        self.cpu_label = QLabel("CPU: --%")
        self.cpu_label.setStyleSheet("color: #a6e3a1; font-size: 10px;")
        layout.addWidget(self.cpu_label)
        
        # RAM
        self.ram_label = QLabel("RAM: --MB")
        self.ram_label.setStyleSheet("color: #89b4fa; font-size: 10px;")
        layout.addWidget(self.ram_label)
        
        # VRAM (if GPU)
        self.vram_label = QLabel("")
        self.vram_label.setStyleSheet("color: #cba6f7; font-size: 10px;")
        layout.addWidget(self.vram_label)
        
        # Mode indicator
        self.mode_label = QLabel("Mode: Full")
        self.mode_label.setStyleSheet("color: #f9e2af; font-size: 10px;")
        layout.addWidget(self.mode_label)
        
        layout.addStretch()
    
    def _start_monitoring(self):
        """Start the monitoring timer."""
        from PyQt5.QtCore import QTimer
        
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_stats)
        self._timer.start(2000)  # Update every 2 seconds
        
        # Initial update
        self._update_stats()
    
    def _update_stats(self):
        """Update resource statistics."""
        try:
            import psutil
            
            # CPU
            cpu = psutil.cpu_percent()
            cpu_color = "#a6e3a1" if cpu < 70 else "#f9e2af" if cpu < 90 else "#f38ba8"
            self.cpu_label.setText(f"CPU: {cpu:.0f}%")
            self.cpu_label.setStyleSheet(f"color: {cpu_color}; font-size: 10px;")
            
            # RAM
            mem = psutil.virtual_memory()
            ram_mb = mem.used / (1024 * 1024)
            ram_pct = mem.percent
            ram_color = "#89b4fa" if ram_pct < 70 else "#f9e2af" if ram_pct < 90 else "#f38ba8"
            self.ram_label.setText(f"RAM: {ram_mb:.0f}MB ({ram_pct:.0f}%)")
            self.ram_label.setStyleSheet(f"color: {ram_color}; font-size: 10px;")
            
        except ImportError:
            self.cpu_label.setText("CPU: N/A")
            self.ram_label.setText("RAM: N/A")
        
        # VRAM (try to get GPU info)
        try:
            import torch
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / (1024 * 1024)
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                vram_pct = (vram_used / vram_total) * 100 if vram_total > 0 else 0
                vram_color = "#cba6f7" if vram_pct < 70 else "#f9e2af" if vram_pct < 90 else "#f38ba8"
                self.vram_label.setText(f"VRAM: {vram_used:.0f}MB")
                self.vram_label.setStyleSheet(f"color: {vram_color}; font-size: 10px;")
            else:
                self.vram_label.setText("")
        except ImportError:
            self.vram_label.setText("")
        
        # Mode (check gaming mode)
        try:
            from ...core.gaming_mode import get_gaming_mode
            gm = get_gaming_mode()
            if gm.active_game:
                self.mode_label.setText(f"Mode: Gaming")
                self.mode_label.setStyleSheet("color: #f38ba8; font-size: 10px;")
            else:
                self.mode_label.setText("Mode: Full")
                self.mode_label.setStyleSheet("color: #a6e3a1; font-size: 10px;")
        except Exception:
            pass
    
    def set_mode(self, mode: str):
        """Manually set the mode display."""
        colors = {
            "full": "#a6e3a1",
            "gaming": "#f38ba8",
            "low_power": "#f9e2af",
            "distributed": "#89b4fa",
        }
        color = colors.get(mode.lower(), "#cdd6f4")
        self.mode_label.setText(f"Mode: {mode.title()}")
        self.mode_label.setStyleSheet(f"color: {color}; font-size: 10px;")


__all__ = [
    'QuickSettingsPanel',
    'ResourceMonitor',
]
