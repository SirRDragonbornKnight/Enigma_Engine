"""
Personality Tab - Configure AI personality traits and presets.

Allows users to:
- View and adjust personality traits with sliders
- Apply personality presets
- Enable/disable personality evolution
- View personality description
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSlider, QCheckBox, QTextEdit,
    QMessageBox, QFrame
)
from PyQt5.QtCore import Qt

from .shared_components import NoScrollComboBox


def create_personality_tab(parent):
    """Create the personality configuration tab."""
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setSpacing(10)
    layout.setContentsMargins(12, 12, 12, 12)

    # === HEADER ===
    header = QLabel("Personality")
    header.setObjectName("header")
    layout.addWidget(header)

    # === COMPACT PRESETS ROW ===
    presets_row = QHBoxLayout()
    presets_row.addWidget(QLabel("Preset:"))

    parent.personality_preset_combo = NoScrollComboBox()
    parent.personality_preset_combo.setToolTip("Select a personality preset")
    parent.personality_preset_combo.addItem("-- Select --", "")
    parent.personality_preset_combo.addItem("Professional", "professional")
    parent.personality_preset_combo.addItem("Friendly", "friendly")
    parent.personality_preset_combo.addItem("Creative", "creative")
    parent.personality_preset_combo.addItem("Analytical", "analytical")
    parent.personality_preset_combo.addItem("Teacher", "teacher")
    parent.personality_preset_combo.addItem("Comedian", "comedian")
    parent.personality_preset_combo.addItem("Coach", "coach")
    presets_row.addWidget(parent.personality_preset_combo)

    apply_preset_btn = QPushButton("Apply")
    apply_preset_btn.setToolTip("Apply selected preset")
    apply_preset_btn.clicked.connect(lambda: _apply_preset(parent))
    presets_row.addWidget(apply_preset_btn)
    presets_row.addStretch()
    layout.addLayout(presets_row)

    # === PERSONALITY TRAITS (compact) ===
    traits_group = QGroupBox("Personality Traits")
    traits_layout = QVBoxLayout(traits_group)
    traits_layout.setSpacing(8)

    # Create compact sliders for each trait
    trait_info = [
        ('humor_level', 'Humor', 'Serious', 'Silly'),
        ('formality', 'Formality', 'Casual', 'Formal'),
        ('verbosity', 'Verbosity', 'Brief', 'Detailed'),
        ('curiosity', 'Curiosity', 'Direct', 'Curious'),
        ('empathy', 'Empathy', 'Logical', 'Emotional'),
        ('creativity', 'Creativity', 'Factual', 'Creative'),
        ('confidence', 'Confidence', 'Cautious', 'Assertive'),
        ('playfulness', 'Playfulness', 'Serious', 'Playful'),
    ]

    parent.trait_sliders = {}
    parent.trait_override_checks = {}
    parent.trait_spinboxes = {}

    for trait_key, trait_name, low_label, high_label in trait_info:
        # Single compact row per trait
        trait_row = QHBoxLayout()
        trait_row.setSpacing(6)
        
        # Trait name
        name_lbl = QLabel(trait_name)
        name_lbl.setFixedWidth(70)
        name_lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
        trait_row.addWidget(name_lbl)
        
        # Low label
        low_lbl = QLabel(low_label)
        low_lbl.setFixedWidth(55)
        low_lbl.setStyleSheet("color: #888; font-size: 10px;")
        low_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        trait_row.addWidget(low_lbl)

        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        parent.trait_sliders[trait_key] = slider
        trait_row.addWidget(slider, stretch=1)
        
        # High label
        high_lbl = QLabel(high_label)
        high_lbl.setFixedWidth(55)
        high_lbl.setStyleSheet("color: #888; font-size: 10px;")
        trait_row.addWidget(high_lbl)

        # Numeric spinbox
        from PyQt5.QtWidgets import QDoubleSpinBox
        value_spin = QDoubleSpinBox()
        value_spin.setRange(0.0, 1.0)
        value_spin.setValue(0.50)
        value_spin.setSingleStep(0.05)
        value_spin.setDecimals(2)
        value_spin.setFixedWidth(55)
        value_spin.valueChanged.connect(
            lambda val, key=trait_key: _update_trait_from_spin(parent, key, val)
        )
        parent.trait_spinboxes[trait_key] = value_spin
        trait_row.addWidget(value_spin)
        
        # Lock checkbox
        override_check = QCheckBox("Lock")
        override_check.setToolTip("Lock trait to prevent evolution")
        override_check.setStyleSheet("font-size: 10px;")
        override_check.stateChanged.connect(
            lambda state, key=trait_key: _toggle_trait_override(parent, key, state)
        )
        parent.trait_override_checks[trait_key] = override_check
        trait_row.addWidget(override_check)

        # Connect slider
        slider.valueChanged.connect(
            lambda val, spin=value_spin, key=trait_key: _update_trait_value(
                parent, key, val, spin
            )
        )

        traits_layout.addLayout(trait_row)

    layout.addWidget(traits_group)

    # === COMPACT EVOLUTION + BUTTONS ROW ===
    controls_row = QHBoxLayout()
    
    parent.allow_evolution_check = QCheckBox("Allow Evolution")
    parent.allow_evolution_check.setChecked(True)
    parent.allow_evolution_check.setToolTip("Allow traits to evolve from conversations")
    parent.allow_evolution_check.stateChanged.connect(
        lambda state: _toggle_evolution(parent, state)
    )
    controls_row.addWidget(parent.allow_evolution_check)
    controls_row.addStretch()

    refresh_btn = QPushButton("Refresh")
    refresh_btn.setToolTip("Reload current personality")
    refresh_btn.clicked.connect(lambda: _refresh_personality(parent))
    controls_row.addWidget(refresh_btn)

    save_btn = QPushButton("Save")
    save_btn.setToolTip("Save current personality settings")
    save_btn.clicked.connect(lambda: _save_personality(parent))
    controls_row.addWidget(save_btn)
    
    save_custom_btn = QPushButton("Save Custom")
    save_custom_btn.setToolTip("Save as a new custom preset")
    save_custom_btn.clicked.connect(lambda: _save_custom_preset(parent))
    controls_row.addWidget(save_custom_btn)

    reset_btn = QPushButton("Reset")
    reset_btn.setToolTip("Reset to default personality")
    reset_btn.clicked.connect(lambda: _reset_personality(parent))
    controls_row.addWidget(reset_btn)

    layout.addLayout(controls_row)

    # === COMPACT STATUS ===
    parent.personality_status = QTextEdit()
    parent.personality_status.setReadOnly(True)
    parent.personality_status.setMaximumHeight(80)
    parent.personality_status.setStyleSheet("font-family: Consolas, monospace; font-size: 10px;")
    parent.personality_status.setPlaceholderText("Personality status...")
    layout.addWidget(parent.personality_status)

    # Initialize personality
    _init_personality(parent)

    return tab


def _init_personality(parent):
    """Initialize personality system."""
    try:
        from ...core.personality import AIPersonality

        model_name = getattr(parent, 'current_model_name', 'forge_ai')
        parent.personality = AIPersonality(model_name)
        parent.personality.load()  # Load if exists, else use defaults

        # Update UI with loaded values
        _update_ui_from_personality(parent)
        _refresh_personality(parent)

    except ImportError:
        parent.personality = None
        parent.personality_status.setPlainText(
            "Personality system not available.\n"
            "Make sure forge_ai.core.personality module exists."
        )
    except Exception as e:
        parent.personality = None
        parent.personality_status.setPlainText(f"Error initializing personality: {e}")


def _update_ui_from_personality(parent):
    """Update UI elements to reflect personality state."""
    if not parent.personality:
        return

    p = parent.personality

    # Update trait sliders and spinboxes
    for trait_key, slider in parent.trait_sliders.items():
        value = p.get_effective_trait(trait_key)
        slider.blockSignals(True)
        slider.setValue(int(value * 100))
        slider.blockSignals(False)
        
        # Update spinbox
        spinbox = parent.trait_spinboxes.get(trait_key)
        if spinbox:
            spinbox.blockSignals(True)
            spinbox.setValue(value)
            spinbox.blockSignals(False)

        # Update override checkbox
        override_check = parent.trait_override_checks.get(trait_key)
        if override_check:
            override_check.blockSignals(True)
            override_check.setChecked(trait_key in p.user_overrides)
            override_check.blockSignals(False)

    # Update evolution checkbox
    parent.allow_evolution_check.blockSignals(True)
    parent.allow_evolution_check.setChecked(p.allow_evolution)
    parent.allow_evolution_check.blockSignals(False)


def _update_trait_value(parent, trait_key, slider_value, spinbox):
    """Handle trait slider value change."""
    value = slider_value / 100.0
    spinbox.blockSignals(True)
    spinbox.setValue(value)
    spinbox.blockSignals(False)

    if not parent.personality:
        return

    # If override is checked, set as user override
    override_check = parent.trait_override_checks.get(trait_key)
    if override_check and override_check.isChecked():
        parent.personality.set_user_override(trait_key, value)
    else:
        # Otherwise update base trait
        setattr(parent.personality.traits, trait_key, value)


def _update_trait_from_spin(parent, trait_key, value):
    """Handle trait spinbox value change."""
    # Update slider
    slider = parent.trait_sliders.get(trait_key)
    if slider:
        slider.blockSignals(True)
        slider.setValue(int(value * 100))
        slider.blockSignals(False)

    if not parent.personality:
        return

    # If override is checked, set as user override
    override_check = parent.trait_override_checks.get(trait_key)
    if override_check and override_check.isChecked():
        parent.personality.set_user_override(trait_key, value)
    else:
        # Otherwise update base trait
        setattr(parent.personality.traits, trait_key, value)


def _toggle_trait_override(parent, trait_key, state):
    """Toggle trait override."""
    if not parent.personality:
        return

    if state == Qt.Checked:
        # Set current value as override
        slider = parent.trait_sliders[trait_key]
        value = slider.value() / 100.0
        parent.personality.set_user_override(trait_key, value)
    else:
        # Clear override
        parent.personality.clear_user_override(trait_key)


def _toggle_evolution(parent, state):
    """Toggle personality evolution."""
    if not parent.personality:
        return

    parent.personality.allow_evolution = (state == Qt.Checked)


def _apply_preset(parent):
    """Apply selected personality preset."""
    if not parent.personality:
        QMessageBox.warning(parent, "Error", "Personality system not available")
        return

    preset = parent.personality_preset_combo.currentData()
    if not preset:
        QMessageBox.information(parent, "Info", "Please select a preset first")
        return

    try:
        parent.personality.set_preset(preset)
        _update_ui_from_personality(parent)
        _refresh_personality(parent)

        QMessageBox.information(
            parent, "Preset Applied",
            f"Applied '{preset}' personality preset.\n\n"
            "Traits have been set as overrides to prevent evolution."
        )
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Failed to apply preset: {e}")


def _refresh_personality(parent):
    """Refresh personality status display."""
    if not parent.personality:
        return

    try:
        description = parent.personality.get_personality_description()
        parent.personality_status.setPlainText(description)
    except Exception as e:
        parent.personality_status.setPlainText(f"Error getting status: {e}")


def _save_personality(parent):
    """Save current personality to file."""
    if not parent.personality:
        QMessageBox.warning(parent, "Error", "Personality system not available")
        return

    try:
        filepath = parent.personality.save()
        QMessageBox.information(
            parent, "Saved",
            f"Personality saved to:\n{filepath}"
        )
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Failed to save: {e}")


def _reset_personality(parent):
    """Reset personality to default values."""
    if not parent.personality:
        QMessageBox.warning(parent, "Error", "Personality system not available")
        return

    reply = QMessageBox.question(
        parent, "Reset Personality",
        "This will reset all traits to default values and clear all overrides.\n\n"
        "Are you sure?",
        QMessageBox.Yes | QMessageBox.No
    )

    if reply != QMessageBox.Yes:
        return

    try:
        # Reset all traits to default values (0.5)
        for trait_name in parent.personality.traits.__annotations__:
            setattr(parent.personality.traits, trait_name, 0.5)
        
        # Clear overrides and reset other state
        parent.personality.clear_all_overrides()
        parent.personality.allow_evolution = True
        parent.personality.interests.clear()
        parent.personality.dislikes.clear()
        parent.personality.opinions.clear()
        parent.personality.mood = "neutral"

        _update_ui_from_personality(parent)
        _refresh_personality(parent)

        QMessageBox.information(parent, "Reset", "Personality reset to defaults.")
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Failed to reset: {e}")


def _save_custom_preset(parent):
    """Save current personality as a custom preset."""
    if not parent.personality:
        QMessageBox.warning(parent, "Error", "Personality system not available")
        return

    from PyQt5.QtWidgets import QInputDialog
    
    name, ok = QInputDialog.getText(
        parent, "Save Custom Preset",
        "Enter a name for this preset:",
        text="my_personality"
    )
    
    if not ok or not name.strip():
        return
    
    try:
        from pathlib import Path
        from ...config import CONFIG
        import json
        
        # Get current trait values
        traits = {}
        for trait_key, slider in parent.trait_sliders.items():
            traits[trait_key] = slider.value() / 100.0
        
        # Save to presets folder
        presets_dir = Path(CONFIG.get("data_dir", "data")) / "personality_presets"
        presets_dir.mkdir(parents=True, exist_ok=True)
        
        preset_file = presets_dir / f"{name.strip()}.json"
        preset_file.write_text(json.dumps(traits, indent=2))
        
        QMessageBox.information(
            parent, "Saved",
            f"Custom preset '{name}' saved!\n\nYou can load it from the presets dropdown."
        )
        
        # Add to presets combo if not already there
        combo = parent.personality_preset_combo
        preset_exists = False
        for i in range(combo.count()):
            if combo.itemData(i) == f"custom:{name}":
                preset_exists = True
                break
        
        if not preset_exists:
            combo.addItem(f"Custom: {name}", f"custom:{name}")
            
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Failed to save preset: {e}")


def _load_default_personality(parent):
    """Load the default balanced personality."""
    if not parent.personality:
        QMessageBox.warning(parent, "Error", "Personality system not available")
        return
    
    try:
        # Set all traits to 0.5 (balanced)
        for trait_key, slider in parent.trait_sliders.items():
            slider.blockSignals(True)
            slider.setValue(50)
            slider.blockSignals(False)
            
            spinbox = parent.trait_spinboxes.get(trait_key)
            if spinbox:
                spinbox.blockSignals(True)
                spinbox.setValue(0.5)
                spinbox.blockSignals(False)
            
            if hasattr(parent.personality, 'traits'):
                setattr(parent.personality.traits, trait_key, 0.5)
        
        # Clear overrides
        for check in parent.trait_override_checks.values():
            check.blockSignals(True)
            check.setChecked(False)
            check.blockSignals(False)
        
        if hasattr(parent.personality, 'clear_all_overrides'):
            parent.personality.clear_all_overrides()
        
        _refresh_personality(parent)
        QMessageBox.information(parent, "Loaded", "Default balanced personality loaded.")
        
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Failed to load default: {e}")
