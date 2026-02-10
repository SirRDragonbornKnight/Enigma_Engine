"""
Expression Mapping UI

GUI for mapping avatar blend shapes to AI emotions.
Allows customization of how different emotions display.

FILE: enigma_engine/gui/dialogs/expression_mapping.py
TYPE: GUI Dialog
MAIN CLASSES: ExpressionMappingDialog, BlendShapeEditor, EmotionPreview
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

try:
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtWidgets import (
        QCheckBox,
        QDialog,
        QDoubleSpinBox,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSlider,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

logger = logging.getLogger(__name__)


# Default emotion to blend shape mappings
DEFAULT_EMOTION_MAPPINGS: dict[str, dict[str, float]] = {
    "happy": {
        "smile": 0.8,
        "brow_up": 0.2,
        "eye_happy": 0.5
    },
    "sad": {
        "frown": 0.7,
        "brow_down": 0.4,
        "eye_sad": 0.6
    },
    "angry": {
        "frown": 0.5,
        "brow_angry": 0.8,
        "eye_narrow": 0.4
    },
    "surprised": {
        "brow_up": 0.9,
        "eye_wide": 0.7,
        "mouth_open": 0.5
    },
    "fearful": {
        "brow_up": 0.6,
        "eye_wide": 0.8,
        "frown": 0.3
    },
    "disgusted": {
        "nose_wrinkle": 0.6,
        "brow_angry": 0.3,
        "frown": 0.4
    },
    "neutral": {
        "smile": 0.0,
        "frown": 0.0,
        "brow_up": 0.0
    },
    "thinking": {
        "brow_up_inner": 0.4,
        "eye_squint": 0.2,
        "mouth_slight": 0.1
    },
    "excited": {
        "smile": 0.9,
        "brow_up": 0.5,
        "eye_wide": 0.4
    },
    "confused": {
        "brow_up_inner": 0.6,
        "head_tilt": 0.3,
        "frown": 0.2
    }
}


# Standard VRM blend shapes
VRM_BLEND_SHAPES = [
    "A", "I", "U", "E", "O",  # Visemes
    "Blink", "Blink_L", "Blink_R",  # Eye
    "Joy", "Angry", "Sorrow", "Fun",  # Emotions (VRM standard)
    "LookUp", "LookDown", "LookLeft", "LookRight",  # Eye direction
    "Neutral"
]


@dataclass
class BlendShapeMapping:
    """Mapping of a blend shape for an emotion."""
    blend_shape: str
    weight: float = 0.5
    transition_time: float = 0.3


@dataclass
class EmotionMapping:
    """Complete mapping for an emotion."""
    emotion_name: str
    blend_shapes: dict[str, float] = field(default_factory=dict)
    transition_speed: float = 0.3
    can_blink: bool = True
    override_look_at: bool = False


class ExpressionMappingConfig:
    """Configuration for expression mappings."""
    
    def __init__(self):
        self._emotions: dict[str, EmotionMapping] = {}
        self._available_blend_shapes: list[str] = VRM_BLEND_SHAPES.copy()
        
        # Initialize defaults
        for emotion, shapes in DEFAULT_EMOTION_MAPPINGS.items():
            self._emotions[emotion] = EmotionMapping(
                emotion_name=emotion,
                blend_shapes=shapes.copy()
            )
    
    def get_mapping(self, emotion: str) -> Optional[EmotionMapping]:
        return self._emotions.get(emotion)
    
    def set_mapping(self, emotion: str, mapping: EmotionMapping):
        self._emotions[emotion] = mapping
    
    def get_blend_shape_weight(self, emotion: str, blend_shape: str) -> float:
        mapping = self._emotions.get(emotion)
        if mapping:
            return mapping.blend_shapes.get(blend_shape, 0.0)
        return 0.0
    
    def set_blend_shape_weight(self, emotion: str, blend_shape: str, weight: float):
        if emotion not in self._emotions:
            self._emotions[emotion] = EmotionMapping(emotion_name=emotion)
        self._emotions[emotion].blend_shapes[blend_shape] = weight
    
    def add_emotion(self, emotion: str):
        if emotion not in self._emotions:
            self._emotions[emotion] = EmotionMapping(emotion_name=emotion)
    
    def remove_emotion(self, emotion: str):
        if emotion in self._emotions:
            del self._emotions[emotion]
    
    def get_all_emotions(self) -> list[str]:
        return list(self._emotions.keys())
    
    def add_blend_shape(self, name: str):
        if name not in self._available_blend_shapes:
            self._available_blend_shapes.append(name)
    
    def get_all_blend_shapes(self) -> list[str]:
        return self._available_blend_shapes.copy()
    
    def to_dict(self) -> dict:
        return {
            "emotions": {
                name: {
                    "blend_shapes": m.blend_shapes,
                    "transition_speed": m.transition_speed,
                    "can_blink": m.can_blink,
                    "override_look_at": m.override_look_at
                }
                for name, m in self._emotions.items()
            },
            "blend_shapes": self._available_blend_shapes
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ExpressionMappingConfig':
        config = cls()
        config._emotions.clear()
        
        for name, mapping_data in data.get("emotions", {}).items():
            config._emotions[name] = EmotionMapping(
                emotion_name=name,
                blend_shapes=mapping_data.get("blend_shapes", {}),
                transition_speed=mapping_data.get("transition_speed", 0.3),
                can_blink=mapping_data.get("can_blink", True),
                override_look_at=mapping_data.get("override_look_at", False)
            )
        
        config._available_blend_shapes = data.get("blend_shapes", VRM_BLEND_SHAPES.copy())
        return config


if HAS_PYQT:
    class BlendShapeSlider(QFrame):
        """Custom slider for blend shape weight."""
        
        value_changed = pyqtSignal(str, float)
        
        def __init__(self, blend_shape: str, initial_value: float = 0.0, parent=None):
            super().__init__(parent)
            self._blend_shape = blend_shape
            self._setup_ui(initial_value)
        
        def _setup_ui(self, initial_value: float):
            layout = QHBoxLayout(self)
            layout.setContentsMargins(4, 2, 4, 2)
            
            # Name label
            self._name_label = QLabel(self._blend_shape)
            self._name_label.setFixedWidth(120)
            layout.addWidget(self._name_label)
            
            # Slider
            self._slider = QSlider(Qt.Horizontal)
            self._slider.setRange(0, 100)
            self._slider.setValue(int(initial_value * 100))
            self._slider.valueChanged.connect(self._on_slider_change)
            layout.addWidget(self._slider, 1)
            
            # Value label
            self._value_label = QLabel(f"{initial_value:.2f}")
            self._value_label.setFixedWidth(40)
            layout.addWidget(self._value_label)
        
        def _on_slider_change(self, value: int):
            weight = value / 100.0
            self._value_label.setText(f"{weight:.2f}")
            self.value_changed.emit(self._blend_shape, weight)
        
        def get_value(self) -> float:
            return self._slider.value() / 100.0
        
        def set_value(self, value: float):
            self._slider.setValue(int(value * 100))


    class ExpressionMappingDialog(QDialog):
        """Dialog for mapping expressions to blend shapes."""
        
        mapping_changed = pyqtSignal(str, str, float)  # emotion, blend_shape, weight
        
        def __init__(self, config: ExpressionMappingConfig = None, parent=None):
            super().__init__(parent)
            self._config = config or ExpressionMappingConfig()
            self._sliders: dict[str, BlendShapeSlider] = {}
            self._current_emotion: Optional[str] = None
            self._setup_ui()
            self._load_first_emotion()
            
            # Apply transparency
            try:
                from ..ui_settings import apply_dialog_transparency
                apply_dialog_transparency(self)
            except ImportError:
                pass
        
        def _setup_ui(self):
            self.setWindowTitle("Expression Mapping")
            self.setMinimumSize(600, 500)
            
            layout = QVBoxLayout(self)
            
            # Tabs
            tabs = QTabWidget()
            layout.addWidget(tabs)
            
            # Main mapping tab
            mapping_widget = QWidget()
            mapping_layout = QHBoxLayout(mapping_widget)
            tabs.addTab(mapping_widget, "Mappings")
            
            # Emotion list
            emotion_group = QGroupBox("Emotions")
            emotion_layout = QVBoxLayout(emotion_group)
            
            self._emotion_list = QListWidget()
            for emotion in self._config.get_all_emotions():
                self._emotion_list.addItem(emotion)
            self._emotion_list.currentTextChanged.connect(self._on_emotion_select)
            emotion_layout.addWidget(self._emotion_list)
            
            # Add/remove emotion buttons
            btn_layout = QHBoxLayout()
            add_btn = QPushButton("Add")
            add_btn.clicked.connect(self._add_emotion)
            remove_btn = QPushButton("Remove")
            remove_btn.clicked.connect(self._remove_emotion)
            btn_layout.addWidget(add_btn)
            btn_layout.addWidget(remove_btn)
            emotion_layout.addLayout(btn_layout)
            
            mapping_layout.addWidget(emotion_group)
            
            # Blend shape sliders
            sliders_group = QGroupBox("Blend Shapes")
            sliders_layout = QVBoxLayout(sliders_group)
            
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            
            self._sliders_container = QWidget()
            self._sliders_layout = QVBoxLayout(self._sliders_container)
            self._sliders_layout.addStretch()
            
            scroll.setWidget(self._sliders_container)
            sliders_layout.addWidget(scroll)
            
            mapping_layout.addWidget(sliders_group, 2)
            
            # Settings tab
            settings_widget = QWidget()
            settings_layout = QVBoxLayout(settings_widget)
            tabs.addTab(settings_widget, "Settings")
            
            # Transition speed
            speed_layout = QHBoxLayout()
            speed_layout.addWidget(QLabel("Transition Speed:"))
            self._speed_spin = QDoubleSpinBox()
            self._speed_spin.setRange(0.1, 2.0)
            self._speed_spin.setSingleStep(0.1)
            self._speed_spin.setValue(0.3)
            speed_layout.addWidget(self._speed_spin)
            speed_layout.addStretch()
            settings_layout.addLayout(speed_layout)
            
            # Can blink
            self._can_blink = QCheckBox("Allow blinking during expression")
            self._can_blink.setChecked(True)
            settings_layout.addWidget(self._can_blink)
            
            # Override look at
            self._override_look = QCheckBox("Override eye tracking")
            settings_layout.addWidget(self._override_look)
            
            settings_layout.addStretch()
            
            # Dialog buttons
            btn_box = QHBoxLayout()
            
            reset_btn = QPushButton("Reset to Defaults")
            reset_btn.clicked.connect(self._reset_defaults)
            btn_box.addWidget(reset_btn)
            
            btn_box.addStretch()
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(self.reject)
            btn_box.addWidget(cancel_btn)
            
            ok_btn = QPushButton("Save")
            ok_btn.clicked.connect(self.accept)
            btn_box.addWidget(ok_btn)
            
            layout.addLayout(btn_box)
        
        def _load_first_emotion(self):
            if self._emotion_list.count() > 0:
                self._emotion_list.setCurrentRow(0)
        
        def _on_emotion_select(self, emotion: str):
            if not emotion:
                return
                
            self._current_emotion = emotion
            
            # Save current settings if we have a previous emotion
            if self._current_emotion:
                mapping = self._config.get_mapping(self._current_emotion)
                if mapping:
                    mapping.transition_speed = self._speed_spin.value()
                    mapping.can_blink = self._can_blink.isChecked()
                    mapping.override_look_at = self._override_look.isChecked()
            
            # Clear existing sliders
            for slider in self._sliders.values():
                slider.deleteLater()
            self._sliders.clear()
            
            # Get mapping
            mapping = self._config.get_mapping(emotion)
            
            # Load settings
            if mapping:
                self._speed_spin.setValue(mapping.transition_speed)
                self._can_blink.setChecked(mapping.can_blink)
                self._override_look.setChecked(mapping.override_look_at)
            
            # Create sliders for all blend shapes
            for blend_shape in self._config.get_all_blend_shapes():
                weight = self._config.get_blend_shape_weight(emotion, blend_shape)
                slider = BlendShapeSlider(blend_shape, weight)
                slider.value_changed.connect(self._on_weight_change)
                
                # Insert before stretch
                self._sliders_layout.insertWidget(
                    self._sliders_layout.count() - 1,
                    slider
                )
                self._sliders[blend_shape] = slider
        
        def _on_weight_change(self, blend_shape: str, weight: float):
            if self._current_emotion:
                self._config.set_blend_shape_weight(
                    self._current_emotion,
                    blend_shape,
                    weight
                )
                self.mapping_changed.emit(self._current_emotion, blend_shape, weight)
        
        def _add_emotion(self):
            from PyQt5.QtWidgets import QInputDialog
            name, ok = QInputDialog.getText(self, "Add Emotion", "Emotion name:")
            if ok and name:
                name = name.lower().strip()
                if name not in self._config.get_all_emotions():
                    self._config.add_emotion(name)
                    self._emotion_list.addItem(name)
        
        def _remove_emotion(self):
            current = self._emotion_list.currentItem()
            if current:
                emotion = current.text()
                self._config.remove_emotion(emotion)
                self._emotion_list.takeItem(self._emotion_list.row(current))
        
        def _reset_defaults(self):
            reply = QMessageBox.question(
                self,
                "Reset Defaults",
                "Reset all mappings to defaults?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._config = ExpressionMappingConfig()
                self._emotion_list.clear()
                for emotion in self._config.get_all_emotions():
                    self._emotion_list.addItem(emotion)
                self._load_first_emotion()
        
        def get_config(self) -> ExpressionMappingConfig:
            """Get the edited configuration."""
            # Save current emotion settings
            if self._current_emotion:
                mapping = self._config.get_mapping(self._current_emotion)
                if mapping:
                    mapping.transition_speed = self._speed_spin.value()
                    mapping.can_blink = self._can_blink.isChecked()
                    mapping.override_look_at = self._override_look.isChecked()
            return self._config

else:
    # Stub when PyQt5 not available
    BlendShapeSlider = None
    ExpressionMappingDialog = None


# Singleton config
_expression_config: Optional[ExpressionMappingConfig] = None


def get_expression_config() -> ExpressionMappingConfig:
    """Get the expression mapping config singleton."""
    global _expression_config
    if _expression_config is None:
        _expression_config = ExpressionMappingConfig()
    return _expression_config


def set_expression_config(config: ExpressionMappingConfig):
    """Set the expression mapping config."""
    global _expression_config
    _expression_config = config


__all__ = [
    'ExpressionMappingDialog',
    'ExpressionMappingConfig',
    'EmotionMapping',
    'BlendShapeMapping',
    'BlendShapeSlider',
    'DEFAULT_EMOTION_MAPPINGS',
    'VRM_BLEND_SHAPES',
    'get_expression_config',
    'set_expression_config'
]
