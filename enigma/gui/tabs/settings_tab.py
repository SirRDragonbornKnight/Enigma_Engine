"""
Settings Tab - Resource management and application settings.

Allows users to control CPU/RAM usage so the AI doesn't hog resources
while gaming or doing other tasks.
"""

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QComboBox, QSpinBox, QSlider, QCheckBox,
    QTextEdit, QMessageBox, QLineEdit
)
from PyQt5.QtCore import Qt


def _get_env_key(key_name: str) -> str:
    """Get an environment variable value, return empty string if not set."""
    return os.environ.get(key_name, "")


def _save_api_keys(parent):
    """Save API keys to environment variables and .env file."""
    keys_to_save = {
        "HF_TOKEN": parent.hf_token_input.text().strip(),
        "OPENAI_API_KEY": parent.openai_key_input.text().strip(),
        "REPLICATE_API_TOKEN": parent.replicate_key_input.text().strip(),
        "ELEVENLABS_API_KEY": parent.elevenlabs_key_input.text().strip(),
    }
    
    # Set environment variables for current session
    saved_count = 0
    for key, value in keys_to_save.items():
        if value:
            os.environ[key] = value
            saved_count += 1
    
    # Try to save to .env file for persistence
    try:
        from ...config import CONFIG
        from pathlib import Path
        
        env_file = Path(CONFIG.get("project_root", ".")) / ".env"
        
        # Read existing .env content
        existing_lines = []
        if env_file.exists():
            existing_lines = env_file.read_text().splitlines()
        
        # Update or add keys
        new_lines = []
        keys_found = set()
        for line in existing_lines:
            if "=" in line and not line.strip().startswith("#"):
                key = line.split("=")[0].strip()
                if key in keys_to_save:
                    if keys_to_save[key]:  # Only write if value is not empty
                        new_lines.append(f"{key}={keys_to_save[key]}")
                    keys_found.add(key)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        # Add new keys that weren't in file
        for key, value in keys_to_save.items():
            if key not in keys_found and value:
                new_lines.append(f"{key}={value}")
        
        # Write back
        env_file.write_text("\n".join(new_lines) + "\n")
        
        parent.api_status_label.setText(f"✓ Saved {saved_count} key(s) to .env file")
        parent.api_status_label.setStyleSheet("color: #22c55e; font-style: italic;")
    except Exception as e:
        # Still saved to environment, just not persisted
        parent.api_status_label.setText(f"✓ Keys set for this session (couldn't save .env: {e})")
        parent.api_status_label.setStyleSheet("color: #f59e0b; font-style: italic;")


def _toggle_key_visibility(parent):
    """Toggle between showing and hiding API keys."""
    from PyQt5.QtWidgets import QLineEdit
    
    inputs = [
        parent.hf_token_input,
        parent.openai_key_input,
        parent.replicate_key_input,
        parent.elevenlabs_key_input,
    ]
    
    # Check current mode of first input
    if parent.hf_token_input.echoMode() == QLineEdit.Password:
        for inp in inputs:
            inp.setEchoMode(QLineEdit.Normal)
    else:
        for inp in inputs:
            inp.setEchoMode(QLineEdit.Password)


def _toggle_ai_lock(parent, state):
    """Toggle AI control lock - prevents user from changing settings."""
    is_locked = state == Qt.Checked
    
    # If trying to unlock and PIN is set, verify it
    if not is_locked and hasattr(parent, '_ai_lock_pin_set') and parent._ai_lock_pin_set:
        from PyQt5.QtWidgets import QInputDialog
        pin, ok = QInputDialog.getText(
            parent, "Unlock", "Enter PIN to unlock:",
            QLineEdit.Password
        )
        if not ok or pin != parent._ai_lock_pin_set:
            parent.ai_lock_checkbox.setChecked(True)  # Keep locked
            QMessageBox.warning(parent, "Incorrect PIN", "The PIN you entered is incorrect.")
            return
    
    # Save PIN if locking and PIN is set
    if is_locked:
        pin = parent.ai_lock_pin.text().strip()
        if pin:
            parent._ai_lock_pin_set = pin
            parent.ai_lock_pin.clear()
            parent.ai_lock_pin.setPlaceholderText("PIN set")
    
    # Store lock state
    parent._ai_control_locked = is_locked
    
    # Update status
    if is_locked:
        parent.ai_lock_status.setText("LOCKED - Only AI can change settings")
        parent.ai_lock_status.setStyleSheet("color: #ef4444; font-weight: bold;")
    else:
        parent.ai_lock_status.setText("Unlocked")
        parent.ai_lock_status.setStyleSheet("color: #22c55e;")
    
    # Get list of controls to lock/unlock
    lockable_widgets = _get_lockable_widgets(parent)
    
    for widget in lockable_widgets:
        widget.setEnabled(not is_locked)
    
    # Always keep the lock checkbox and PIN field enabled
    parent.ai_lock_checkbox.setEnabled(True)
    parent.ai_lock_pin.setEnabled(not is_locked)


def _get_lockable_widgets(parent):
    """Get list of widgets that should be locked when AI control is enabled."""
    widgets = []
    
    # Settings widgets
    lockable_attrs = [
        # Power mode
        'resource_mode_combo',
        # Theme
        'theme_combo',
        # Autonomous mode
        'autonomous_enabled_check',
        'autonomous_activity_spin',
        # API Keys
        'hf_token_input',
        'openai_key_input', 
        'replicate_key_input',
        'elevenlabs_key_input',
        # Training controls
        'epochs_spin',
        'batch_spin',
        'lr_spin',
        'train_file_combo',
        'btn_train',
        # Chat controls
        'chat_input',
        'send_btn',
        # Personality sliders
        'curiosity_slider',
        'friendliness_slider',
        'creativity_slider',
        'formality_slider',
        'humor_slider',
    ]
    
    for attr in lockable_attrs:
        if hasattr(parent, attr):
            widgets.append(getattr(parent, attr))
    
    return widgets


def create_settings_tab(parent):
    """Create the settings/resources tab."""
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setSpacing(15)
    
    # === AI CONTROL LOCK ===
    lock_group = QGroupBox("AI Control Lock")
    lock_layout = QVBoxLayout(lock_group)
    
    lock_desc = QLabel(
        "When locked, only the AI can change settings. "
        "This prevents accidental changes while the AI is working."
    )
    lock_desc.setWordWrap(True)
    lock_layout.addWidget(lock_desc)
    
    lock_row = QHBoxLayout()
    parent.ai_lock_checkbox = QCheckBox("Lock settings for AI control")
    parent.ai_lock_checkbox.stateChanged.connect(
        lambda state: _toggle_ai_lock(parent, state)
    )
    lock_row.addWidget(parent.ai_lock_checkbox)
    
    parent.ai_lock_status = QLabel("")
    parent.ai_lock_status.setStyleSheet("color: #888; font-style: italic;")
    lock_row.addWidget(parent.ai_lock_status)
    lock_row.addStretch()
    lock_layout.addLayout(lock_row)
    
    # Password protection for unlock
    pwd_row = QHBoxLayout()
    pwd_row.addWidget(QLabel("Unlock PIN (optional):"))
    parent.ai_lock_pin = QLineEdit()
    parent.ai_lock_pin.setPlaceholderText("Set a 4-digit PIN")
    parent.ai_lock_pin.setMaxLength(4)
    parent.ai_lock_pin.setMaximumWidth(100)
    parent.ai_lock_pin.setEchoMode(QLineEdit.Password)
    pwd_row.addWidget(parent.ai_lock_pin)
    pwd_row.addStretch()
    lock_layout.addLayout(pwd_row)
    
    layout.addWidget(lock_group)
    
    # === DEVICE INFO ===
    device_group = QGroupBox(" Hardware Detection")
    device_layout = QVBoxLayout(device_group)
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            device_info = f"GPU Available: {gpu_name} ({gpu_mem} MB)"
            device_style = "color: #22c55e; font-weight: bold;"
        else:
            device_info = "No GPU - Using CPU only"
            device_style = "color: #f59e0b; font-weight: bold;"
        
        cpu_count = torch.get_num_threads()
        cpu_info = f"CPU Threads: {cpu_count}"
    except Exception:
        device_info = "Warning: PyTorch not available"
        device_style = "color: #ef4444;"
        cpu_info = ""
    
    device_label = QLabel(device_info)
    device_label.setStyleSheet(device_style)
    device_layout.addWidget(device_label)
    
    if cpu_info:
        cpu_label = QLabel(cpu_info)
        device_layout.addWidget(cpu_label)
    
    layout.addWidget(device_group)
    
    # === POWER MODE ===
    power_group = QGroupBox(" Power Mode")
    power_layout = QVBoxLayout(power_group)
    
    # Mode description
    mode_desc = QLabel(
        "Control how much CPU/GPU the AI uses. Lower settings free up resources for gaming or other apps."
    )
    mode_desc.setWordWrap(True)
    power_layout.addWidget(mode_desc)
    
    # Mode selector
    mode_row = QHBoxLayout()
    mode_row.addWidget(QLabel("Mode:"))
    
    parent.resource_mode_combo = QComboBox()
    parent.resource_mode_combo.addItem("Minimal - Best for gaming", "minimal")
    parent.resource_mode_combo.addItem("Gaming - AI in background", "gaming")
    parent.resource_mode_combo.addItem("Balanced - Normal use (default)", "balanced")
    parent.resource_mode_combo.addItem("Performance - Faster AI responses", "performance")
    parent.resource_mode_combo.addItem("Maximum - Use all resources", "max")
    parent.resource_mode_combo.setCurrentIndex(2)  # Default to balanced
    parent.resource_mode_combo.currentIndexChanged.connect(
        lambda idx: _apply_resource_mode(parent)
    )
    mode_row.addWidget(parent.resource_mode_combo)
    mode_row.addStretch()
    power_layout.addLayout(mode_row)
    
    # Mode details
    parent.power_mode_details_label = QLabel(
        "Balanced: Moderate resource usage. Good for normal use."
    )
    parent.power_mode_details_label.setStyleSheet("color: #888; font-style: italic;")
    power_layout.addWidget(parent.power_mode_details_label)
    
    layout.addWidget(power_group)

    # === THEME SELECTOR ===
    theme_group = QGroupBox("Theme")
    theme_layout = QVBoxLayout(theme_group)

    theme_desc = QLabel(
        "Choose a visual theme for the application. "
        "Changes apply immediately."
    )
    theme_desc.setWordWrap(True)
    theme_layout.addWidget(theme_desc)

    theme_row = QHBoxLayout()
    theme_row.addWidget(QLabel("Theme:"))

    parent.theme_combo = QComboBox()
    parent.theme_combo.addItem("Dark (Default)", "dark")
    parent.theme_combo.addItem("Light", "light")
    parent.theme_combo.addItem("High Contrast", "high_contrast")
    parent.theme_combo.addItem("Midnight", "midnight")
    parent.theme_combo.addItem("Forest", "forest")
    parent.theme_combo.addItem("Sunset", "sunset")
    parent.theme_combo.currentIndexChanged.connect(
        lambda idx: _apply_theme(parent)
    )
    theme_row.addWidget(parent.theme_combo)

    theme_row.addStretch()
    theme_layout.addLayout(theme_row)

    parent.theme_description_label = QLabel(
        "Dark theme with soft colors (default)"
    )
    parent.theme_description_label.setStyleSheet("color: #888; font-style: italic;")
    theme_layout.addWidget(parent.theme_description_label)

    layout.addWidget(theme_group)

    # === AUTONOMOUS MODE ===
    autonomous_group = QGroupBox("Autonomous Mode")
    autonomous_layout = QVBoxLayout(autonomous_group)
    
    autonomous_desc = QLabel(
        "Allow AI to act on its own - explore curiosities, learn from the web, "
        "and evolve personality when you're not chatting. "
        "Can be turned off at any time."
    )
    autonomous_desc.setWordWrap(True)
    autonomous_layout.addWidget(autonomous_desc)
    
    parent.autonomous_enabled_check = QCheckBox("Enable Autonomous Mode")
    parent.autonomous_enabled_check.stateChanged.connect(
        lambda state: _toggle_autonomous(parent, state)
    )
    autonomous_layout.addWidget(parent.autonomous_enabled_check)
    
    # Autonomous settings
    autonomous_settings = QHBoxLayout()
    autonomous_settings.addWidget(QLabel("Activity Level:"))
    
    parent.autonomous_activity_spin = QSpinBox()
    parent.autonomous_activity_spin.setRange(1, 20)
    parent.autonomous_activity_spin.setValue(12)
    parent.autonomous_activity_spin.setSuffix(" actions/hour")
    parent.autonomous_activity_spin.setToolTip("How many autonomous actions per hour")
    parent.autonomous_activity_spin.setEnabled(False)  # Disabled until autonomous mode enabled
    autonomous_settings.addWidget(parent.autonomous_activity_spin)
    autonomous_settings.addStretch()
    autonomous_layout.addLayout(autonomous_settings)
    
    layout.addWidget(autonomous_group)
    
    # === API KEYS ===
    api_group = QGroupBox("🔑 API Keys")
    api_layout = QVBoxLayout(api_group)
    
    api_desc = QLabel(
        "Configure API keys for cloud services. Keys are stored in environment variables. "
        "Leave blank to use local models only."
    )
    api_desc.setWordWrap(True)
    api_layout.addWidget(api_desc)
    
    # HuggingFace Token
    hf_row = QHBoxLayout()
    hf_row.addWidget(QLabel("HuggingFace Token:"))
    parent.hf_token_input = QLineEdit()
    parent.hf_token_input.setPlaceholderText("hf_... (for gated models like Llama)")
    parent.hf_token_input.setEchoMode(QLineEdit.Password)
    parent.hf_token_input.setText(_get_env_key("HF_TOKEN"))
    hf_row.addWidget(parent.hf_token_input)
    api_layout.addLayout(hf_row)
    
    # OpenAI API Key
    openai_row = QHBoxLayout()
    openai_row.addWidget(QLabel("OpenAI API Key:"))
    parent.openai_key_input = QLineEdit()
    parent.openai_key_input.setPlaceholderText("sk-... (for DALL-E, GPT-4)")
    parent.openai_key_input.setEchoMode(QLineEdit.Password)
    parent.openai_key_input.setText(_get_env_key("OPENAI_API_KEY"))
    openai_row.addWidget(parent.openai_key_input)
    api_layout.addLayout(openai_row)
    
    # Replicate Token
    replicate_row = QHBoxLayout()
    replicate_row.addWidget(QLabel("Replicate Token:"))
    parent.replicate_key_input = QLineEdit()
    parent.replicate_key_input.setPlaceholderText("r8_... (for cloud video/audio/3D)")
    parent.replicate_key_input.setEchoMode(QLineEdit.Password)
    parent.replicate_key_input.setText(_get_env_key("REPLICATE_API_TOKEN"))
    replicate_row.addWidget(parent.replicate_key_input)
    api_layout.addLayout(replicate_row)
    
    # ElevenLabs Key
    eleven_row = QHBoxLayout()
    eleven_row.addWidget(QLabel("ElevenLabs Key:"))
    parent.elevenlabs_key_input = QLineEdit()
    parent.elevenlabs_key_input.setPlaceholderText("(for cloud TTS)")
    parent.elevenlabs_key_input.setEchoMode(QLineEdit.Password)
    parent.elevenlabs_key_input.setText(_get_env_key("ELEVENLABS_API_KEY"))
    eleven_row.addWidget(parent.elevenlabs_key_input)
    api_layout.addLayout(eleven_row)
    
    # Save keys button
    api_buttons = QHBoxLayout()
    save_keys_btn = QPushButton("Save API Keys")
    save_keys_btn.clicked.connect(lambda: _save_api_keys(parent))
    api_buttons.addWidget(save_keys_btn)
    
    show_keys_btn = QPushButton("Show/Hide")
    show_keys_btn.clicked.connect(lambda: _toggle_key_visibility(parent))
    api_buttons.addWidget(show_keys_btn)
    
    api_buttons.addStretch()
    api_layout.addLayout(api_buttons)
    
    parent.api_status_label = QLabel("")
    parent.api_status_label.setStyleSheet("color: #888; font-style: italic;")
    api_layout.addWidget(parent.api_status_label)
    
    layout.addWidget(api_group)
    
    # === CURRENT STATUS ===
    status_group = QGroupBox("Current Status")
    status_layout = QVBoxLayout(status_group)
    
    parent.power_status = QTextEdit()
    parent.power_status.setReadOnly(True)
    parent.power_status.setMaximumHeight(150)
    parent.power_status.setStyleSheet("font-family: Consolas, monospace;")
    status_layout.addWidget(parent.power_status)
    
    refresh_btn = QPushButton("Refresh Status")
    refresh_btn.clicked.connect(lambda: _refresh_power_status(parent))
    status_layout.addWidget(refresh_btn)
    
    layout.addWidget(status_group)
    
    layout.addStretch()
    
    # Initial status refresh
    _refresh_power_status(parent)
    
    return tab


def _load_saved_settings(parent):
    """Load saved settings from CONFIG into the UI."""
    from ...config import CONFIG
    
    # Load resource mode
    saved_mode = CONFIG.get("resource_mode", "balanced")
    mode_map = {"minimal": 0, "gaming": 1, "balanced": 2, "performance": 3, "max": 4}
    if saved_mode in mode_map:
        parent.resource_mode_combo.setCurrentIndex(mode_map[saved_mode])


def _apply_theme(parent):
    """Apply selected theme to the application."""
    theme_id = parent.theme_combo.currentData()

    # Theme descriptions
    descriptions = {
        "dark": "Dark theme with soft colors (default)",
        "light": "Light theme for bright environments",
        "high_contrast": "High contrast for accessibility",
        "midnight": "Deep blue midnight theme",
        "forest": "Nature-inspired green theme",
        "sunset": "Warm sunset colors"
    }
    parent.theme_description_label.setText(descriptions.get(theme_id, ""))

    try:
        from ..theme_system import ThemeManager

        # Get or create theme manager
        if not hasattr(parent, 'theme_manager'):
            parent.theme_manager = ThemeManager()

        # Apply theme
        if parent.theme_manager.set_theme(theme_id):
            stylesheet = parent.theme_manager.get_current_stylesheet()
            # Apply to main window
            main_window = parent.window()
            if main_window:
                main_window.setStyleSheet(stylesheet)
    except ImportError:
        QMessageBox.warning(
            parent, "Theme Error",
            "Theme system module (theme_system) not found.\n\n"
            "Please ensure the enigma.gui.theme_system module is properly installed."
        )
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Failed to apply theme: {e}")


def _apply_resource_mode(parent):
    """Apply selected resource mode."""
    mode = parent.resource_mode_combo.currentData()
    
    # Update description
    descriptions = {
        "minimal": "Minimal: Uses 1 CPU thread, low priority. Best while gaming!",
        "gaming": "Gaming: AI runs in background, prioritizes gaming performance.",
        "balanced": "Balanced: Uses moderate resources. Good for normal use.",
        "performance": "Performance: Uses more resources for faster AI responses.",
        "max": "Maximum: Uses all available resources. May slow other apps."
    }
    parent.power_mode_details_label.setText(descriptions.get(mode, ""))


def _update_gpu_label(parent, value):
    """Update GPU percentage label - only if gpu_label exists."""
    if hasattr(parent, 'gpu_label'):
        parent.gpu_label.setText(f"{value}%")


def _update_cpu_threads(parent, value):
    """Handle CPU thread change."""
    pass  # Applied when Apply button is clicked


def _update_priority(parent, state):
    """Handle priority checkbox change."""
    pass  # Applied when Apply button is clicked


def _apply_all_settings(parent):
    """Apply all resource settings."""
    try:
        from ...core.power_mode import get_power_manager, PowerLevel
        
        mode = parent.resource_mode_combo.currentData()
        power_mgr = get_power_manager()
        
        # Convert string to PowerLevel enum
        level = PowerLevel(mode)
        power_mgr.set_level(level)
        
        # Update description - match the mode values from resource_mode_combo
        descriptions = {
            "minimal": "Minimal: Uses 1 CPU thread, low priority. Best while gaming!",
            "gaming": "Gaming: AI runs in background, prioritizes gaming performance.",
            "balanced": "Balanced: Moderate resource usage. Good for normal use.",
            "performance": "Performance: Uses more resources for faster AI responses.",
            "max": "Maximum: Uses all available resources. May slow other apps."
        }
        parent.power_mode_details_label.setText(descriptions.get(mode, ""))
        
        # Refresh status
        _refresh_power_status(parent)
        
        QMessageBox.information(parent, "Power Mode Changed", 
            f"Power mode set to: {mode.upper()}\n\n"
            f"Batch size: {power_mgr.settings.max_batch_size}\n"
            f"Max tokens: {power_mgr.settings.max_tokens}\n"
            f"GPU: {'Enabled' if power_mgr.settings.use_gpu else 'Disabled'}"
        )
    except ImportError:
        QMessageBox.warning(parent, "Error", "Power mode manager not available")
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Failed to apply power mode: {e}")


def _toggle_autonomous(parent, state):
    """Toggle autonomous mode on/off."""
    try:
        from ...core.autonomous import AutonomousManager
        
        # Get current model name
        model_name = getattr(parent, 'current_model_name', 'enigma')
        autonomous = AutonomousManager.get(model_name)
        
        if state == Qt.Checked:
            # Set activity level
            max_actions = parent.autonomous_activity_spin.value()
            autonomous.max_actions_per_hour = max_actions
            
            # Start autonomous mode
            autonomous.start()
            parent.autonomous_activity_spin.setEnabled(True)
            
            QMessageBox.information(parent, "Autonomous Mode", 
                "Autonomous mode enabled!\n\n"
                "AI will explore topics, learn, and evolve on its own.\n"
                "You can disable this at any time."
            )
        else:
            # Stop autonomous mode
            autonomous.stop()
            parent.autonomous_activity_spin.setEnabled(False)
            
    except ImportError:
        QMessageBox.warning(parent, "Error", "Autonomous mode not available")
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Failed to toggle autonomous mode: {e}")


def _refresh_power_status(parent):
    """Refresh power status display."""
    try:
        from ...core.power_mode import get_power_manager
        import torch
        
        power_mgr = get_power_manager()
        
        status_text = f"""Power Mode: {power_mgr.level.value.upper()}

Settings:
  Max Batch Size: {power_mgr.settings.max_batch_size}
  Max Tokens: {power_mgr.settings.max_tokens}
  GPU Enabled: {'Yes' if power_mgr.settings.use_gpu else 'No'}
  Thread Count: {power_mgr.settings.thread_count if power_mgr.settings.thread_count > 0 else 'Auto'}
  Response Delay: {power_mgr.settings.response_delay}s
  Paused: {'Yes' if power_mgr.is_paused else 'No'}

System:
  PyTorch Threads: {torch.get_num_threads()}"""
        
        if torch.cuda.is_available():
            status_text += f"""
  GPU Available: Yes
  GPU Name: {torch.cuda.get_device_name(0)}"""
        else:
            status_text += """
  GPU Available: No"""
        
        parent.power_status.setPlainText(status_text)
        
    except ImportError:
        parent.power_status.setPlainText(
            "Power mode manager not available.\n"
            "Make sure enigma.core.power_mode module exists."
        )
    except Exception as e:
        parent.power_status.setPlainText(f"Error getting status: {e}")
