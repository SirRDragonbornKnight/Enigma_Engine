"""
Settings Tab - Resource management and application settings.

Allows users to control CPU/RAM usage so the AI doesn't hog resources
while gaming or doing other tasks.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QComboBox, QSpinBox, QSlider, QCheckBox,
    QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt


def create_settings_tab(parent):
    """Create the settings/resources tab."""
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setSpacing(15)
    
    # === DEVICE INFO ===
    device_group = QGroupBox("🖥️ Hardware Detection")
    device_layout = QVBoxLayout(device_group)
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            device_info = f"✅ GPU Available: {gpu_name} ({gpu_mem} MB)"
            device_style = "color: #22c55e; font-weight: bold;"
        else:
            device_info = "❌ No GPU - Using CPU only"
            device_style = "color: #f59e0b; font-weight: bold;"
        
        cpu_count = torch.get_num_threads()
        cpu_info = f"CPU Threads: {cpu_count}"
    except Exception:
        device_info = "⚠️ PyTorch not available"
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
    power_group = QGroupBox("⚡ Power Mode")
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
    parent.resource_mode_combo.addItem("🎮 Minimal - Best for gaming", "minimal")
    parent.resource_mode_combo.addItem("🕹️ Gaming - AI in background", "gaming")
    parent.resource_mode_combo.addItem("⚖️ Balanced - Normal use (default)", "balanced")
    parent.resource_mode_combo.addItem("🚀 Performance - Faster AI responses", "performance")
    parent.resource_mode_combo.addItem("💪 Maximum - Use all resources", "max")
    parent.resource_mode_combo.setCurrentIndex(2)  # Default to balanced
    parent.resource_mode_combo.currentIndexChanged.connect(
        lambda idx: _apply_resource_mode(parent)
    )
    mode_row.addWidget(parent.power_mode_combo)
    mode_row.addStretch()
    power_layout.addLayout(mode_row)
    
    # Mode details
    parent.power_mode_details_label = QLabel(
        "Balanced: Moderate resource usage. Good for normal use."
    )
    parent.power_mode_details_label.setStyleSheet("color: #888; font-style: italic;")
    power_layout.addWidget(parent.power_mode_details_label)
    
    layout.addWidget(power_group)
    
    # === AUTONOMOUS MODE ===
    autonomous_group = QGroupBox("🤖 Autonomous Mode")
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
    
    # === CURRENT STATUS ===
    status_group = QGroupBox("📊 Current Status")
    status_layout = QVBoxLayout(status_group)
    
    parent.power_status = QTextEdit()
    parent.power_status.setReadOnly(True)
    parent.power_status.setMaximumHeight(150)
    parent.power_status.setStyleSheet("font-family: Consolas, monospace;")
    status_layout.addWidget(parent.power_status)
    
    refresh_btn = QPushButton("🔄 Refresh Status")
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
    mode_map = {"minimal": 0, "balanced": 1, "performance": 2, "max": 3}
    if saved_mode in mode_map:
        parent.resource_mode_combo.setCurrentIndex(mode_map[saved_mode])
    
    # Load CPU threads
    saved_threads = CONFIG.get("cpu_threads", 0)
    parent.cpu_threads_spin.setValue(saved_threads)
    
    # Load GPU memory fraction
    saved_gpu = CONFIG.get("gpu_memory_fraction", 0.5)
    parent.gpu_slider.setValue(int(saved_gpu * 100))
    _update_gpu_label(parent, int(saved_gpu * 100))
    
    # Load low priority setting
    saved_priority = CONFIG.get("low_priority", False)
    parent.low_priority_check.setChecked(saved_priority)


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
    parent.mode_details_label.setText(descriptions.get(mode, ""))
    
    # Update advanced controls to match mode
    mode_settings = {
        "minimal": {"threads": 1, "gpu": 20, "priority": True},
        "gaming": {"threads": 2, "gpu": 30, "priority": True},
        "balanced": {"threads": 0, "gpu": 50, "priority": False},
        "performance": {"threads": 0, "gpu": 70, "priority": False},
        "max": {"threads": 0, "gpu": 90, "priority": False},
    }
    
    settings = mode_settings.get(mode, mode_settings["balanced"])
    parent.cpu_threads_spin.setValue(settings["threads"])
    parent.gpu_slider.setValue(settings["gpu"])
    parent.low_priority_check.setChecked(settings["priority"])


def _update_gpu_label(parent, value):
    """Update GPU percentage label."""
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
        
        mode = parent.power_mode_combo.currentData()
        power_mgr = get_power_manager()
        
        # Convert string to PowerLevel enum
        level = PowerLevel(mode)
        power_mgr.set_level(level)
        
        # Update description
        descriptions = {
            "full": "Full: Uses all available resources for maximum performance.",
            "balanced": "Balanced: Moderate resource usage. Good for normal use.",
            "low": "Low: Minimal resources, slower responses. CPU only.",
            "gaming": "Gaming: Pauses background tasks, minimal CPU/GPU. Best while gaming!",
            "background": "Background: Lowest priority, minimal resources.",
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
