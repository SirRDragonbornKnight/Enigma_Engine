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
    QPushButton, QComboBox, QSlider, QCheckBox, QTextEdit,
    QMessageBox, QFrame
)
from PyQt5.QtCore import Qt


def create_personality_tab(parent):
    """Create the personality configuration tab."""
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setSpacing(15)

    # === PERSONALITY PRESETS ===
    presets_group = QGroupBox("🎭 Personality Presets")
    presets_layout = QVBoxLayout(presets_group)

    preset_desc = QLabel(
        "Choose a personality preset to quickly configure traits, "
        "or customize individual traits below."
    )
    preset_desc.setWordWrap(True)
    presets_layout.addWidget(preset_desc)

    preset_row = QHBoxLayout()
    preset_row.addWidget(QLabel("Preset:"))

    parent.personality_preset_combo = QComboBox()
    parent.personality_preset_combo.addItem("-- Select Preset --", "")
    parent.personality_preset_combo.addItem("💼 Professional", "professional")
    parent.personality_preset_combo.addItem("😊 Friendly", "friendly")
    parent.personality_preset_combo.addItem("Creative", "creative")
    parent.personality_preset_combo.addItem("🔬 Analytical", "analytical")
    parent.personality_preset_combo.addItem("👩‍🏫 Teacher", "teacher")
    parent.personality_preset_combo.addItem("😂 Comedian", "comedian")
    parent.personality_preset_combo.addItem("Coach", "coach")
    preset_row.addWidget(parent.personality_preset_combo)

    apply_preset_btn = QPushButton("Apply Preset")
    apply_preset_btn.clicked.connect(lambda: _apply_preset(parent))
    preset_row.addWidget(apply_preset_btn)

    preset_row.addStretch()
    presets_layout.addLayout(preset_row)

    layout.addWidget(presets_group)

    # === PERSONALITY TRAITS ===
    traits_group = QGroupBox("Personality Traits")
    traits_layout = QVBoxLayout(traits_group)

    traits_desc = QLabel(
        "Adjust individual traits using the sliders. "
        "Override traits to prevent them from auto-evolving."
    )
    traits_desc.setWordWrap(True)
    traits_layout.addWidget(traits_desc)

    # Create sliders for each trait
    trait_info = [
        ('humor_level', '😄 Humor', 'Serious ↔ Silly'),
        ('formality', '👔 Formality', 'Casual ↔ Formal'),
        ('verbosity', 'Verbosity', 'Brief ↔ Detailed'),
        ('curiosity', '🤔 Curiosity', 'Answers Only ↔ Asks Questions'),
        ('empathy', '💚 Empathy', 'Logical ↔ Emotional'),
        ('creativity', 'Creativity', 'Factual ↔ Imaginative'),
        ('confidence', '💪 Confidence', 'Hedging ↔ Assertive'),
        ('playfulness', '🎮 Playfulness', 'Professional ↔ Fun'),
    ]

    parent.trait_sliders = {}
    parent.trait_override_checks = {}

    for trait_key, trait_name, trait_desc in trait_info:
        trait_frame = QFrame()
        trait_frame.setFrameStyle(QFrame.StyledPanel)
        trait_layout = QVBoxLayout(trait_frame)
        trait_layout.setContentsMargins(8, 8, 8, 8)

        # Trait header row
        header_row = QHBoxLayout()
        trait_label = QLabel(trait_name)
        trait_label.setStyleSheet("font-weight: bold;")
        header_row.addWidget(trait_label)

        override_check = QCheckBox("Override")
        override_check.setToolTip("Lock this trait to prevent auto-evolution")
        override_check.stateChanged.connect(
            lambda state, key=trait_key: _toggle_trait_override(parent, key, state)
        )
        parent.trait_override_checks[trait_key] = override_check
        header_row.addWidget(override_check)

        header_row.addStretch()

        value_label = QLabel("0.50")
        value_label.setMinimumWidth(40)
        header_row.addWidget(value_label)

        trait_layout.addLayout(header_row)

        # Slider with description
        slider_row = QHBoxLayout()
        desc_label = QLabel(trait_desc)
        desc_label.setStyleSheet("color: #888; font-size: 11px;")
        slider_row.addWidget(desc_label)
        trait_layout.addLayout(slider_row)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(25)
        slider.valueChanged.connect(
            lambda val, lbl=value_label, key=trait_key: _update_trait_value(
                parent, key, val, lbl
            )
        )
        parent.trait_sliders[trait_key] = (slider, value_label)
        trait_layout.addWidget(slider)

        traits_layout.addWidget(trait_frame)

    layout.addWidget(traits_group)

    # === EVOLUTION SETTINGS ===
    evolution_group = QGroupBox("🧬 Personality Evolution")
    evolution_layout = QVBoxLayout(evolution_group)

    evolution_desc = QLabel(
        "When enabled, personality traits naturally evolve based on "
        "conversation feedback. Overridden traits are not affected."
    )
    evolution_desc.setWordWrap(True)
    evolution_layout.addWidget(evolution_desc)

    parent.allow_evolution_check = QCheckBox("Allow Personality Evolution")
    parent.allow_evolution_check.setChecked(True)
    parent.allow_evolution_check.stateChanged.connect(
        lambda state: _toggle_evolution(parent, state)
    )
    evolution_layout.addWidget(parent.allow_evolution_check)

    layout.addWidget(evolution_group)

    # === PERSONALITY STATUS ===
    status_group = QGroupBox("📋 Personality Status")
    status_layout = QVBoxLayout(status_group)

    parent.personality_status = QTextEdit()
    parent.personality_status.setReadOnly(True)
    parent.personality_status.setMaximumHeight(150)
    parent.personality_status.setStyleSheet("font-family: Consolas, monospace;")
    status_layout.addWidget(parent.personality_status)

    # Button row
    btn_row = QHBoxLayout()

    refresh_btn = QPushButton("Refresh")
    refresh_btn.clicked.connect(lambda: _refresh_personality(parent))
    btn_row.addWidget(refresh_btn)

    save_btn = QPushButton("Save Personality")
    save_btn.clicked.connect(lambda: _save_personality(parent))
    btn_row.addWidget(save_btn)

    reset_btn = QPushButton("🔁 Reset to Default")
    reset_btn.clicked.connect(lambda: _reset_personality(parent))
    btn_row.addWidget(reset_btn)

    btn_row.addStretch()
    status_layout.addLayout(btn_row)

    layout.addWidget(status_group)

    layout.addStretch()

    # Initialize personality
    _init_personality(parent)

    return tab


def _init_personality(parent):
    """Initialize personality system."""
    try:
        from ...core.personality import AIPersonality

        model_name = getattr(parent, 'current_model_name', 'enigma')
        parent.personality = AIPersonality(model_name)
        parent.personality.load()  # Load if exists, else use defaults

        # Update UI with loaded values
        _update_ui_from_personality(parent)
        _refresh_personality(parent)

    except ImportError:
        parent.personality = None
        parent.personality_status.setPlainText(
            "Personality system not available.\n"
            "Make sure enigma.core.personality module exists."
        )
    except Exception as e:
        parent.personality = None
        parent.personality_status.setPlainText(f"Error initializing personality: {e}")


def _update_ui_from_personality(parent):
    """Update UI elements to reflect personality state."""
    if not parent.personality:
        return

    p = parent.personality

    # Update trait sliders
    for trait_key, (slider, value_label) in parent.trait_sliders.items():
        value = p.get_effective_trait(trait_key)
        slider.blockSignals(True)
        slider.setValue(int(value * 100))
        slider.blockSignals(False)
        value_label.setText(f"{value:.2f}")

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


def _update_trait_value(parent, trait_key, slider_value, value_label):
    """Handle trait slider value change."""
    value = slider_value / 100.0
    value_label.setText(f"{value:.2f}")

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
        slider, _ = parent.trait_sliders[trait_key]
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
