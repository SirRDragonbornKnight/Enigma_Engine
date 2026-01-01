"""
Settings Tab - Resource management and application settings.

Allows users to control CPU/RAM usage so the AI doesn't hog resources
while gaming or doing other tasks.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QComboBox, QSpinBox, QSlider, QCheckBox,
    QTextEdit
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
    
    # === RESOURCE MODE ===
    resource_group = QGroupBox("⚡ Resource Mode (for Gaming/Multitasking)")
    resource_layout = QVBoxLayout(resource_group)
    
    # Mode description
    mode_desc = QLabel(
        "Choose how much CPU/RAM the AI uses. Lower settings let you game or run other apps smoothly."
    )
    mode_desc.setWordWrap(True)
    resource_layout.addWidget(mode_desc)
    
    # Mode selector
    mode_row = QHBoxLayout()
    mode_row.addWidget(QLabel("Mode:"))
    
    parent.resource_mode_combo = QComboBox()
    parent.resource_mode_combo.addItem("🎮 Minimal - Best for gaming", "minimal")
    parent.resource_mode_combo.addItem("⚖️ Balanced - Normal use (default)", "balanced")
    parent.resource_mode_combo.addItem("🚀 Performance - Faster AI responses", "performance")
    parent.resource_mode_combo.addItem("💪 Maximum - Use all resources", "max")
    parent.resource_mode_combo.setCurrentIndex(1)  # Default to balanced
    parent.resource_mode_combo.currentIndexChanged.connect(
        lambda idx: _apply_resource_mode(parent)
    )
    mode_row.addWidget(parent.resource_mode_combo)
    mode_row.addStretch()
    resource_layout.addLayout(mode_row)
    
    # Mode details
    parent.mode_details_label = QLabel(
        "Balanced: Uses moderate resources. Good for normal use."
    )
    parent.mode_details_label.setStyleSheet("color: #888; font-style: italic;")
    resource_layout.addWidget(parent.mode_details_label)
    
    layout.addWidget(resource_group)
    
    # === ADVANCED SETTINGS ===
    advanced_group = QGroupBox("🔧 Advanced Resource Settings")
    advanced_layout = QVBoxLayout(advanced_group)
    
    # CPU Threads
    cpu_row = QHBoxLayout()
    cpu_row.addWidget(QLabel("CPU Threads:"))
    
    parent.cpu_threads_spin = QSpinBox()
    parent.cpu_threads_spin.setRange(0, 32)
    parent.cpu_threads_spin.setValue(0)
    parent.cpu_threads_spin.setSpecialValueText("Auto")
    parent.cpu_threads_spin.setToolTip("0 = Auto, or set specific number of CPU threads")
    parent.cpu_threads_spin.valueChanged.connect(lambda v: _update_cpu_threads(parent, v))
    cpu_row.addWidget(parent.cpu_threads_spin)
    
    cpu_row.addWidget(QLabel("(0 = Auto, higher = more CPU usage)"))
    cpu_row.addStretch()
    advanced_layout.addLayout(cpu_row)
    
    # GPU Memory (if available)
    gpu_row = QHBoxLayout()
    gpu_row.addWidget(QLabel("GPU Memory:"))
    
    parent.gpu_slider = QSlider(Qt.Horizontal)
    parent.gpu_slider.setRange(10, 100)
    parent.gpu_slider.setValue(50)
    parent.gpu_slider.setTickInterval(10)
    parent.gpu_slider.setTickPosition(QSlider.TicksBelow)
    parent.gpu_slider.valueChanged.connect(lambda v: _update_gpu_label(parent, v))
    gpu_row.addWidget(parent.gpu_slider)
    
    parent.gpu_label = QLabel("50%")
    parent.gpu_label.setMinimumWidth(50)
    gpu_row.addWidget(parent.gpu_label)
    
    advanced_layout.addLayout(gpu_row)
    
    # Low Priority checkbox
    parent.low_priority_check = QCheckBox("Run at low priority (helps other apps run smoother)")
    parent.low_priority_check.stateChanged.connect(lambda s: _update_priority(parent, s))
    advanced_layout.addWidget(parent.low_priority_check)
    
    layout.addWidget(advanced_group)
    
    # === CURRENT STATUS ===
    status_group = QGroupBox("📊 Current Resource Status")
    status_layout = QVBoxLayout(status_group)
    
    parent.resource_status = QTextEdit()
    parent.resource_status.setReadOnly(True)
    parent.resource_status.setMaximumHeight(150)
    parent.resource_status.setStyleSheet("font-family: Consolas, monospace;")
    status_layout.addWidget(parent.resource_status)
    
    refresh_btn = QPushButton("🔄 Refresh Status")
    refresh_btn.clicked.connect(lambda: _refresh_status(parent))
    status_layout.addWidget(refresh_btn)
    
    layout.addWidget(status_group)
    
    # === APPLY BUTTON ===
    apply_row = QHBoxLayout()
    apply_row.addStretch()
    
    apply_btn = QPushButton("✅ Apply Settings")
    apply_btn.setMinimumWidth(150)
    apply_btn.clicked.connect(lambda: _apply_all_settings(parent))
    apply_row.addWidget(apply_btn)
    
    layout.addLayout(apply_row)
    
    layout.addStretch()
    
    # Load saved settings from CONFIG
    _load_saved_settings(parent)
    
    # Initial status refresh
    _refresh_status(parent)
    
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
        "balanced": "Balanced: Uses moderate resources. Good for normal use.",
        "performance": "Performance: Uses more resources for faster AI responses.",
        "max": "Maximum: Uses all available resources. May slow other apps."
    }
    parent.mode_details_label.setText(descriptions.get(mode, ""))
    
    # Update advanced controls to match mode
    mode_settings = {
        "minimal": {"threads": 1, "gpu": 20, "priority": True},
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
        from enigma.core.resources import (
            apply_resource_mode, set_cpu_threads, 
            set_gpu_memory_fraction, set_low_priority
        )
        
        # Apply mode first
        mode = parent.resource_mode_combo.currentData()
        apply_resource_mode(mode)
        
        # Then apply specific overrides
        threads = parent.cpu_threads_spin.value()
        if threads > 0:
            set_cpu_threads(threads)
        
        gpu_fraction = parent.gpu_slider.value() / 100.0
        set_gpu_memory_fraction(gpu_fraction)
        
        low_priority = parent.low_priority_check.isChecked()
        set_low_priority(low_priority)
        
        _refresh_status(parent)
        
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(parent, "Settings Applied", 
            f"Resource mode: {mode}\n"
            f"CPU threads: {'Auto' if threads == 0 else threads}\n"
            f"GPU memory: {parent.gpu_slider.value()}%\n"
            f"Low priority: {'Yes' if low_priority else 'No'}"
        )
        
    except ImportError:
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.warning(parent, "Error", "Could not load resource manager")
    except Exception as e:
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.warning(parent, "Error", f"Failed to apply settings: {e}")


def _refresh_status(parent):
    """Refresh resource status display."""
    try:
        from enigma.core.resources import get_resource_info
        info = get_resource_info()
        
        status_text = f"""Resource Mode: {info['mode'].upper()}
        
CPU Cores Available: {info['cpu_count']}
Torch Threads: {info.get('torch_threads', 'N/A')}
Low Priority: {'Yes' if info['low_priority'] else 'No'}

GPU Available: {'Yes' if info['gpu_available'] else 'No'}"""
        
        if info['gpu_available']:
            status_text += f"""
GPU Name: {info.get('gpu_name', 'Unknown')}
GPU Memory: {info.get('gpu_memory_total_mb', 0)} MB
GPU Limit: {int(info['gpu_memory_fraction'] * 100)}%"""
        
        parent.resource_status.setPlainText(status_text)
        
    except ImportError:
        parent.resource_status.setPlainText(
            "Resource manager not available.\n"
            "Make sure enigma.core.resources module exists."
        )
    except Exception as e:
        parent.resource_status.setPlainText(f"Error getting status: {e}")
