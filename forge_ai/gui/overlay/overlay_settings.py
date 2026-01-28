"""
Overlay Settings Widget - Configuration UI for overlay system.

Provides a settings panel for configuring overlay appearance and behavior.
"""

import logging
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QSlider, QComboBox, QCheckBox, QPushButton, QLineEdit,
    QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from .overlay_modes import OverlayMode, OverlayPosition
from .overlay_themes import OVERLAY_THEMES

logger = logging.getLogger(__name__)


class OverlaySettingsWidget(QWidget):
    """
    Settings widget for configuring the overlay.
    
    Signals:
        settings_changed: Emitted when any setting changes
    """
    
    settings_changed = pyqtSignal(dict)  # settings dict
    
    def __init__(self, overlay=None, parent=None):
        super().__init__(parent)
        self.overlay = overlay
        self.setup_ui()
        
        # Load current settings if overlay exists
        if self.overlay:
            self._load_from_overlay()
            
    def setup_ui(self):
        """Setup the settings UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Overlay Settings")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # Display Mode Group
        mode_group = QGroupBox("Display Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        mode_label = QLabel("Mode:")
        mode_layout.addWidget(mode_label)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Minimal - Just response",
            "Compact - Response + input",
            "Full - Complete chat",
            "Hidden - Not visible"
        ])
        self.mode_combo.currentIndexChanged.connect(self._on_settings_changed)
        mode_layout.addWidget(self.mode_combo)
        
        layout.addWidget(mode_group)
        
        # Position Group
        position_group = QGroupBox("Position")
        position_layout = QVBoxLayout(position_group)
        
        pos_label = QLabel("Position:")
        position_layout.addWidget(pos_label)
        
        self.position_combo = QComboBox()
        self.position_combo.addItems([
            "Top Left",
            "Top Right",
            "Bottom Left",
            "Bottom Right",
            "Center",
            "Custom"
        ])
        self.position_combo.currentIndexChanged.connect(self._on_settings_changed)
        position_layout.addWidget(self.position_combo)
        
        layout.addWidget(position_group)
        
        # Appearance Group
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QVBoxLayout(appearance_group)
        
        # Theme selector
        theme_label = QLabel("Theme:")
        appearance_layout.addWidget(theme_label)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(OVERLAY_THEMES.keys()))
        self.theme_combo.currentTextChanged.connect(self._on_settings_changed)
        appearance_layout.addWidget(self.theme_combo)
        
        # Opacity slider
        opacity_label = QLabel("Opacity:")
        appearance_layout.addWidget(opacity_label)
        
        opacity_layout = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(10)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(90)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        
        self.opacity_value = QLabel("90%")
        opacity_layout.addWidget(self.opacity_value)
        
        appearance_layout.addLayout(opacity_layout)
        
        layout.addWidget(appearance_group)
        
        # Behavior Group
        behavior_group = QGroupBox("Behavior")
        behavior_layout = QVBoxLayout(behavior_group)
        
        self.always_on_top_check = QCheckBox("Always on top")
        self.always_on_top_check.setChecked(True)
        self.always_on_top_check.stateChanged.connect(self._on_settings_changed)
        behavior_layout.addWidget(self.always_on_top_check)
        
        self.click_through_check = QCheckBox("Click-through (pass clicks to game)")
        self.click_through_check.setChecked(False)
        self.click_through_check.stateChanged.connect(self._on_settings_changed)
        behavior_layout.addWidget(self.click_through_check)
        
        self.remember_position_check = QCheckBox("Remember position")
        self.remember_position_check.setChecked(True)
        self.remember_position_check.stateChanged.connect(self._on_settings_changed)
        behavior_layout.addWidget(self.remember_position_check)
        
        self.show_on_startup_check = QCheckBox("Show on startup")
        self.show_on_startup_check.setChecked(False)
        self.show_on_startup_check.stateChanged.connect(self._on_settings_changed)
        behavior_layout.addWidget(self.show_on_startup_check)
        
        layout.addWidget(behavior_group)
        
        # Hotkey Group
        hotkey_group = QGroupBox("Hotkey")
        hotkey_layout = QHBoxLayout(hotkey_group)
        
        hotkey_label = QLabel("Show/Hide:")
        hotkey_layout.addWidget(hotkey_label)
        
        self.hotkey_input = QLineEdit()
        self.hotkey_input.setPlaceholderText("Ctrl+Shift+A")
        self.hotkey_input.setText("Ctrl+Shift+A")
        self.hotkey_input.textChanged.connect(self._on_settings_changed)
        hotkey_layout.addWidget(self.hotkey_input)
        
        layout.addWidget(hotkey_group)
        
        # Action Buttons
        button_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply_settings)
        button_layout.addWidget(self.apply_btn)
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(self.reset_btn)
        
        layout.addLayout(button_layout)
        
        layout.addStretch()
        
    def _load_from_overlay(self):
        """Load settings from overlay."""
        if not self.overlay:
            return
            
        settings = self.overlay.settings
        
        # Set mode
        mode_index = {
            OverlayMode.MINIMAL: 0,
            OverlayMode.COMPACT: 1,
            OverlayMode.FULL: 2,
            OverlayMode.HIDDEN: 3,
        }.get(settings.mode, 1)
        self.mode_combo.setCurrentIndex(mode_index)
        
        # Set position
        pos_index = {
            OverlayPosition.TOP_LEFT: 0,
            OverlayPosition.TOP_RIGHT: 1,
            OverlayPosition.BOTTOM_LEFT: 2,
            OverlayPosition.BOTTOM_RIGHT: 3,
            OverlayPosition.CENTER: 4,
            OverlayPosition.CUSTOM: 5,
        }.get(settings.position, 1)
        self.position_combo.setCurrentIndex(pos_index)
        
        # Set theme
        theme_index = list(OVERLAY_THEMES.keys()).index(settings.theme_name)
        self.theme_combo.setCurrentIndex(theme_index)
        
        # Set opacity
        opacity_percent = int(settings.opacity * 100)
        self.opacity_slider.setValue(opacity_percent)
        self.opacity_value.setText(f"{opacity_percent}%")
        
        # Set checkboxes
        self.always_on_top_check.setChecked(settings.always_on_top)
        self.click_through_check.setChecked(settings.click_through)
        self.remember_position_check.setChecked(settings.remember_position)
        self.show_on_startup_check.setChecked(settings.show_on_startup)
        
        # Set hotkey
        self.hotkey_input.setText(settings.hotkey)
        
    def _on_opacity_changed(self, value: int):
        """Handle opacity slider change."""
        self.opacity_value.setText(f"{value}%")
        self._on_settings_changed()
        
    def _on_settings_changed(self):
        """Handle any setting change."""
        settings = self._get_settings_dict()
        self.settings_changed.emit(settings)
        
    def _apply_settings(self):
        """Apply settings to overlay."""
        if not self.overlay:
            logger.warning("No overlay to apply settings to")
            return
            
        settings = self._get_settings_dict()
        
        # Apply mode
        mode_map = [
            OverlayMode.MINIMAL,
            OverlayMode.COMPACT,
            OverlayMode.FULL,
            OverlayMode.HIDDEN,
        ]
        self.overlay.set_mode(mode_map[settings['mode_index']])
        
        # Apply position
        pos_map = [
            OverlayPosition.TOP_LEFT,
            OverlayPosition.TOP_RIGHT,
            OverlayPosition.BOTTOM_LEFT,
            OverlayPosition.BOTTOM_RIGHT,
            OverlayPosition.CENTER,
            OverlayPosition.CUSTOM,
        ]
        self.overlay.set_position(pos_map[settings['position_index']])
        
        # Apply theme
        self.overlay.set_theme(settings['theme'])
        
        # Apply opacity
        self.overlay.set_opacity(settings['opacity'])
        
        # Apply click-through
        self.overlay.set_click_through(settings['click_through'])
        
        # Update overlay settings
        self.overlay.settings.always_on_top = settings['always_on_top']
        self.overlay.settings.remember_position = settings['remember_position']
        self.overlay.settings.show_on_startup = settings['show_on_startup']
        self.overlay.settings.hotkey = settings['hotkey']
        
        # Re-setup window if always_on_top changed
        if self.overlay.settings.always_on_top != settings['always_on_top']:
            self.overlay._setup_window()
            
        logger.info("Overlay settings applied")
        
    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.mode_combo.setCurrentIndex(1)  # Compact
        self.position_combo.setCurrentIndex(1)  # Top Right
        self.theme_combo.setCurrentIndex(2)  # Gaming
        self.opacity_slider.setValue(90)
        self.always_on_top_check.setChecked(True)
        self.click_through_check.setChecked(False)
        self.remember_position_check.setChecked(True)
        self.show_on_startup_check.setChecked(False)
        self.hotkey_input.setText("Ctrl+Shift+A")
        
    def _get_settings_dict(self) -> dict:
        """Get current settings as dictionary."""
        return {
            'mode_index': self.mode_combo.currentIndex(),
            'position_index': self.position_combo.currentIndex(),
            'theme': self.theme_combo.currentText(),
            'opacity': self.opacity_slider.value() / 100.0,
            'always_on_top': self.always_on_top_check.isChecked(),
            'click_through': self.click_through_check.isChecked(),
            'remember_position': self.remember_position_check.isChecked(),
            'show_on_startup': self.show_on_startup_check.isChecked(),
            'hotkey': self.hotkey_input.text().strip(),
        }
        
    def set_overlay(self, overlay):
        """Set the overlay instance."""
        self.overlay = overlay
        self._load_from_overlay()
