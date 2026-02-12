"""
Settings Tab Section Builders - Modular UI construction.

This module contains helper functions that build individual sections
of the Settings tab, keeping the main create_settings_tab() function clean.
"""

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .shared_components import NoScrollComboBox

# Checked constant
Checked = Qt.CheckState.Checked


def build_quick_settings_section(parent, sync_funcs: dict) -> QGroupBox:
    """Build the Quick Settings section with common toggles.
    
    Args:
        parent: Parent window with widget attributes
        sync_funcs: Dict of sync functions (_sync_autonomous_toggles, etc.)
    
    Returns:
        QGroupBox containing quick settings widgets
    """
    group = QGroupBox("Quick Settings")
    layout = QVBoxLayout(group)
    layout.setSpacing(8)
    
    desc = QLabel("Common settings you can toggle quickly.")
    desc.setWordWrap(True)
    desc.setStyleSheet("color: #a6adc8; margin-bottom: 6px;")
    layout.addWidget(desc)
    
    # Row of quick toggles
    row1 = QHBoxLayout()
    
    # Autonomous Mode
    parent.quick_autonomous_check = QCheckBox("Autonomous Mode")
    parent.quick_autonomous_check.setToolTip(
        "AI acts on its own when you're not chatting:\n"
        "- Explores curiosities\n"
        "- Learns from the web\n"
        "- Evolves personality"
    )
    parent.quick_autonomous_check.stateChanged.connect(
        lambda state: sync_funcs['autonomous'](parent, state)
    )
    row1.addWidget(parent.quick_autonomous_check)
    
    # Game Mode
    parent.quick_game_mode_check = QCheckBox("Game Mode")
    parent.quick_game_mode_check.setToolTip(
        "Reduce AI resource usage for gaming:\n"
        "- Lower CPU/GPU usage\n"
        "- AI stays responsive but minimal"
    )
    parent.quick_game_mode_check.stateChanged.connect(
        lambda state: sync_funcs['game_mode'](parent, state)
    )
    row1.addWidget(parent.quick_game_mode_check)
    
    # Always on top
    parent.quick_always_on_top = QCheckBox("Window on Top")
    parent.quick_always_on_top.setToolTip("Keep main window above others")
    parent.quick_always_on_top.stateChanged.connect(
        lambda state: sync_funcs['always_on_top'](parent, state)
    )
    row1.addWidget(parent.quick_always_on_top)
    
    row1.addStretch()
    layout.addLayout(row1)
    
    # Content Rating row
    content_row = QHBoxLayout()
    content_row.addWidget(QLabel("Content Rating:"))
    
    parent.content_rating_combo = NoScrollComboBox()
    parent.content_rating_combo.setToolTip(
        "Control content restrictions:\n"
        "SFW - Safe for work, no explicit content\n"
        "Mature - Violence, mild language allowed\n"
        "NSFW - Adult content (requires trained model)"
    )
    parent.content_rating_combo.addItem("SFW (Safe)", "sfw")
    parent.content_rating_combo.addItem("Mature", "mature")
    parent.content_rating_combo.addItem("NSFW (Adult)", "nsfw")
    parent.content_rating_combo.currentIndexChanged.connect(
        lambda idx: sync_funcs['content_rating'](parent)
    )
    content_row.addWidget(parent.content_rating_combo)
    
    parent.content_rating_status = QLabel("")
    parent.content_rating_status.setStyleSheet("color: #a6adc8; font-style: italic;")
    content_row.addWidget(parent.content_rating_status)
    
    content_row.addStretch()
    layout.addLayout(content_row)
    
    # Quick links row
    row2 = QHBoxLayout()
    
    go_persona_btn = QPushButton("Personality")
    go_persona_btn.setToolTip("Configure AI personality")
    go_persona_btn.clicked.connect(lambda: sync_funcs['go_to_tab'](parent, "Persona"))
    row2.addWidget(go_persona_btn)
    
    go_avatar_btn = QPushButton("Avatar")
    go_avatar_btn.setToolTip("Avatar display settings")
    go_avatar_btn.clicked.connect(lambda: sync_funcs['go_to_tab'](parent, "Avatar"))
    row2.addWidget(go_avatar_btn)
    
    go_modules_btn = QPushButton("Modules")
    go_modules_btn.setToolTip("Enable/disable AI capabilities")
    go_modules_btn.clicked.connect(lambda: sync_funcs['go_to_tab'](parent, "Modules"))
    row2.addWidget(go_modules_btn)
    
    row2.addStretch()
    layout.addLayout(row2)
    
    return group


def build_device_info_section(parent, apply_nn_backend_func, update_status_func) -> QGroupBox:
    """Build the Hardware device info section.
    
    Returns:
        QGroupBox containing hardware info and NN backend selector
    """
    group = QGroupBox("Hardware")
    layout = QVBoxLayout(group)
    
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
    layout.addWidget(device_label)
    
    if cpu_info:
        cpu_label = QLabel(cpu_info)
        layout.addWidget(cpu_label)
    
    # Neural Network Backend
    nn_backend_row = QHBoxLayout()
    nn_backend_row.addWidget(QLabel("NN Backend:"))
    
    parent.nn_backend_combo = NoScrollComboBox()
    parent.nn_backend_combo.setToolTip(
        "Neural network backend:\n"
        "Auto - Uses pure Python for nano/micro, PyTorch for larger\n"
        "Pure Python - Zero dependencies, works anywhere (slow)\n"
        "PyTorch - Fast, requires PyTorch installed"
    )
    parent.nn_backend_combo.addItem("Auto (recommended)", "auto")
    parent.nn_backend_combo.addItem("Pure Python (no dependencies)", "pure")
    parent.nn_backend_combo.addItem("PyTorch (fast)", "torch")
    
    # Set default based on config
    try:
        from ...config import CONFIG
        current_backend = CONFIG.get("nn_backend", "auto")
        idx_map = {"auto": 0, "pure": 1, "torch": 2}
        parent.nn_backend_combo.setCurrentIndex(idx_map.get(current_backend, 0))
    except Exception:
        parent.nn_backend_combo.setCurrentIndex(0)
    
    parent.nn_backend_combo.currentIndexChanged.connect(
        lambda idx: apply_nn_backend_func(parent)
    )
    nn_backend_row.addWidget(parent.nn_backend_combo)
    
    # Show current Python info
    try:
        from ...builtin.neural_network import get_python_info
        info = get_python_info()
        runtime_label = QLabel(f"{info['implementation']}")
        if info['is_pypy']:
            runtime_label.setStyleSheet("color: #22c55e; font-weight: bold;")
            runtime_label.setToolTip("PyPy detected - Pure Python will be faster!")
        else:
            runtime_label.setStyleSheet("color: #bac2de;")
        nn_backend_row.addWidget(runtime_label)
    except Exception:
        pass  # Intentionally silent
    
    nn_backend_row.addStretch()
    layout.addLayout(nn_backend_row)
    
    # Backend status
    parent.nn_backend_status = QLabel("")
    parent.nn_backend_status.setStyleSheet("color: #bac2de; font-style: italic; font-size: 11px;")
    layout.addWidget(parent.nn_backend_status)
    update_status_func(parent)
    
    return group


def build_device_profile_section(parent, apply_profile_func, auto_detect_func) -> QGroupBox:
    """Build the Device Profile section.
    
    Returns:
        QGroupBox containing profile selector and auto-detect
    """
    group = QGroupBox("Device Profile")
    layout = QVBoxLayout(group)
    
    desc = QLabel(
        "Select a profile for your device type. This configures resources, features, and tools appropriately."
    )
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Profile selector
    profile_row = QHBoxLayout()
    profile_row.addWidget(QLabel("Profile:"))
    
    parent.device_profile_combo = NoScrollComboBox()
    parent.device_profile_combo.setToolTip("Select a device profile for optimal configuration")
    
    # Add profiles
    parent.device_profile_combo.addItem("Raspberry Pi / Robot - Minimal resources", "raspberry_pi")
    parent.device_profile_combo.addItem("Phone / Tablet - Avatar display only", "phone")
    parent.device_profile_combo.addItem("PC Gaming Mode - AI in background", "pc_gaming")
    parent.device_profile_combo.addItem("Workstation / RTX - Full power", "workstation")
    parent.device_profile_combo.addItem("Balanced (Default)", "balanced")
    
    parent.device_profile_combo.setCurrentIndex(4)  # Default to balanced
    parent.device_profile_combo.currentIndexChanged.connect(
        lambda idx: apply_profile_func(parent)
    )
    profile_row.addWidget(parent.device_profile_combo)
    
    # Auto-detect button
    auto_detect_btn = QPushButton("Auto-detect")
    auto_detect_btn.setToolTip("Automatically detect the best profile for this hardware")
    auto_detect_btn.setFixedWidth(90)
    auto_detect_btn.clicked.connect(lambda: auto_detect_func(parent))
    profile_row.addWidget(auto_detect_btn)
    
    profile_row.addStretch()
    layout.addLayout(profile_row)
    
    # Profile details
    parent.profile_details_label = QLabel(
        "Balanced: Moderate resource usage. Good for normal desktop use."
    )
    parent.profile_details_label.setWordWrap(True)
    parent.profile_details_label.setStyleSheet("color: #bac2de; font-style: italic;")
    layout.addWidget(parent.profile_details_label)
    
    # Features enabled info
    parent.profile_features_label = QLabel("")
    parent.profile_features_label.setWordWrap(True)
    parent.profile_features_label.setStyleSheet("color: #bac2de; font-size: 12px;")
    layout.addWidget(parent.profile_features_label)
    
    # Profile status
    parent.profile_status_label = QLabel("")
    parent.profile_status_label.setStyleSheet("color: #22c55e; font-style: italic;")
    layout.addWidget(parent.profile_status_label)
    
    return group


def build_power_mode_section(parent, apply_mode_func) -> QGroupBox:
    """Build the Power Mode section.
    
    Returns:
        QGroupBox containing power mode controls
    """
    import os as os_module
    
    group = QGroupBox("Power Mode")
    layout = QVBoxLayout(group)
    
    # Mode description
    desc = QLabel(
        "Control how much CPU/GPU the AI uses. Lower settings free up resources for gaming or other apps."
    )
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Mode selector
    mode_row = QHBoxLayout()
    mode_row.addWidget(QLabel("Mode:"))
    
    parent.resource_mode_combo = NoScrollComboBox()
    parent.resource_mode_combo.setToolTip("Select resource usage mode")
    parent.resource_mode_combo.addItem("Minimal - Best for gaming", "minimal")
    parent.resource_mode_combo.addItem("Gaming - AI in background", "gaming")
    parent.resource_mode_combo.addItem("Balanced - Normal use (default)", "balanced")
    parent.resource_mode_combo.addItem("Performance - Faster AI responses", "performance")
    parent.resource_mode_combo.addItem("Maximum - Use all resources", "max")
    parent.resource_mode_combo.setCurrentIndex(2)  # Default to balanced
    parent.resource_mode_combo.currentIndexChanged.connect(
        lambda idx: apply_mode_func(parent)
    )
    mode_row.addWidget(parent.resource_mode_combo)
    mode_row.addStretch()
    layout.addLayout(mode_row)
    
    # Mode details
    parent.power_mode_details_label = QLabel(
        "Balanced: Moderate resource usage. Good for normal use."
    )
    parent.power_mode_details_label.setStyleSheet("color: #bac2de; font-style: italic;")
    layout.addWidget(parent.power_mode_details_label)
    
    # Custom resource settings
    custom_frame = QGroupBox("Custom Resource Limits")
    custom_layout = QVBoxLayout(custom_frame)
    
    # GPU usage spinbox
    gpu_row = QHBoxLayout()
    gpu_row.addWidget(QLabel("GPU Usage (%):"))
    
    parent.gpu_spinbox = QSpinBox()
    parent.gpu_spinbox.setRange(10, 95)
    parent.gpu_spinbox.setValue(85)
    parent.gpu_spinbox.setSuffix("%")
    parent.gpu_spinbox.setMinimumWidth(80)
    gpu_row.addWidget(parent.gpu_spinbox)
    gpu_row.addStretch()
    custom_layout.addLayout(gpu_row)
    
    # CPU threads spinbox
    cpu_row = QHBoxLayout()
    cpu_row.addWidget(QLabel("CPU Threads:"))
    
    max_threads = os_module.cpu_count() or 8
    parent.cpu_spinbox = QSpinBox()
    parent.cpu_spinbox.setRange(1, max_threads)
    parent.cpu_spinbox.setValue(max(1, max_threads // 2))
    parent.cpu_spinbox.setMinimumWidth(80)
    cpu_row.addWidget(parent.cpu_spinbox)
    
    cpu_info = QLabel(f"(max: {max_threads})")
    cpu_info.setStyleSheet("color: #bac2de;")
    cpu_row.addWidget(cpu_info)
    cpu_row.addStretch()
    custom_layout.addLayout(cpu_row)
    
    layout.addWidget(custom_frame)
    
    # Power status
    parent.power_status_label = QLabel("")
    parent.power_status_label.setStyleSheet("color: #bac2de; font-style: italic;")
    layout.addWidget(parent.power_status_label)
    
    return group


def build_api_keys_section(parent, save_keys_func, toggle_visibility_func, get_env_key_func) -> QGroupBox:
    """Build the API Keys section.
    
    Returns:
        QGroupBox containing API key inputs
    """
    group = QGroupBox("API Keys")
    layout = QVBoxLayout(group)
    
    desc = QLabel(
        "Configure API keys for cloud services. Keys are stored in environment variables. "
        "Leave blank to use local models only."
    )
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # HuggingFace Token
    hf_row = QHBoxLayout()
    hf_row.addWidget(QLabel("HuggingFace Token:"))
    parent.hf_token_input = QLineEdit()
    parent.hf_token_input.setPlaceholderText("hf_... (for gated models like Llama)")
    parent.hf_token_input.setEchoMode(QLineEdit.Password)
    parent.hf_token_input.setText(get_env_key_func("HF_TOKEN"))
    hf_row.addWidget(parent.hf_token_input)
    layout.addLayout(hf_row)
    
    # OpenAI API Key
    openai_row = QHBoxLayout()
    openai_row.addWidget(QLabel("OpenAI API Key:"))
    parent.openai_key_input = QLineEdit()
    parent.openai_key_input.setPlaceholderText("sk-... (for DALL-E, GPT-4)")
    parent.openai_key_input.setEchoMode(QLineEdit.Password)
    parent.openai_key_input.setText(get_env_key_func("OPENAI_API_KEY"))
    openai_row.addWidget(parent.openai_key_input)
    layout.addLayout(openai_row)
    
    # Anthropic/Claude API Key
    anthropic_row = QHBoxLayout()
    anthropic_row.addWidget(QLabel("Claude API Key:"))
    parent.anthropic_key_input = QLineEdit()
    parent.anthropic_key_input.setPlaceholderText("sk-ant-... (for Claude)")
    parent.anthropic_key_input.setEchoMode(QLineEdit.Password)
    parent.anthropic_key_input.setText(get_env_key_func("ANTHROPIC_API_KEY"))
    anthropic_row.addWidget(parent.anthropic_key_input)
    layout.addLayout(anthropic_row)
    
    # Replicate Token
    replicate_row = QHBoxLayout()
    replicate_row.addWidget(QLabel("Replicate Token:"))
    parent.replicate_key_input = QLineEdit()
    parent.replicate_key_input.setPlaceholderText("r8_... (for cloud video/audio/3D)")
    parent.replicate_key_input.setEchoMode(QLineEdit.Password)
    parent.replicate_key_input.setText(get_env_key_func("REPLICATE_API_TOKEN"))
    replicate_row.addWidget(parent.replicate_key_input)
    layout.addLayout(replicate_row)
    
    # ElevenLabs Key
    eleven_row = QHBoxLayout()
    eleven_row.addWidget(QLabel("ElevenLabs Key:"))
    parent.elevenlabs_key_input = QLineEdit()
    parent.elevenlabs_key_input.setPlaceholderText("(for cloud TTS)")
    parent.elevenlabs_key_input.setEchoMode(QLineEdit.Password)
    parent.elevenlabs_key_input.setText(get_env_key_func("ELEVENLABS_API_KEY"))
    eleven_row.addWidget(parent.elevenlabs_key_input)
    layout.addLayout(eleven_row)
    
    # Save keys button
    api_buttons = QHBoxLayout()
    save_keys_btn = QPushButton("Save API Keys")
    save_keys_btn.clicked.connect(lambda: save_keys_func(parent))
    api_buttons.addWidget(save_keys_btn)
    
    show_keys_btn = QPushButton("Show/Hide")
    show_keys_btn.clicked.connect(lambda: toggle_visibility_func(parent))
    api_buttons.addWidget(show_keys_btn)
    
    api_buttons.addStretch()
    layout.addLayout(api_buttons)
    
    parent.api_status_label = QLabel("")
    parent.api_status_label.setStyleSheet("color: #bac2de; font-style: italic;")
    layout.addWidget(parent.api_status_label)
    
    return group


def build_cache_management_section(parent, refresh_func, open_func, clear_func) -> QGroupBox:
    """Build the Cache Management section.
    
    Returns:
        QGroupBox containing cache management controls
    """
    group = QGroupBox("Cache Management")
    layout = QVBoxLayout(group)
    
    desc = QLabel(
        "Manage downloaded models and cached data. Clear cache to free disk space."
    )
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Cache path label
    parent.cache_path_label = QLabel("Location: Unknown")
    parent.cache_path_label.setStyleSheet("color: #bac2de; font-size: 11px;")
    layout.addWidget(parent.cache_path_label)
    
    # Cache size label  
    parent.cache_size_label = QLabel("Cache Size: Calculating...")
    parent.cache_size_label.setStyleSheet("font-weight: bold;")
    layout.addWidget(parent.cache_size_label)
    
    # Buttons
    btn_row = QHBoxLayout()
    
    refresh_btn = QPushButton("Refresh")
    refresh_btn.setToolTip("Refresh cache information")
    refresh_btn.clicked.connect(lambda: refresh_func(parent))
    btn_row.addWidget(refresh_btn)
    
    open_btn = QPushButton("Open Folder")
    open_btn.setToolTip("Open cache folder in file explorer")
    open_btn.clicked.connect(lambda: open_func(parent))
    btn_row.addWidget(open_btn)
    
    clear_btn = QPushButton("Clear Cache")
    clear_btn.setToolTip("Delete all downloaded models")
    clear_btn.setStyleSheet("color: #ef4444;")
    clear_btn.clicked.connect(lambda: clear_func(parent))
    btn_row.addWidget(clear_btn)
    
    btn_row.addStretch()
    layout.addLayout(btn_row)
    
    return group


def build_reset_settings_section(parent, reset_func, export_func, import_func) -> QGroupBox:
    """Build the Reset Settings section.
    
    Returns:
        QGroupBox containing reset and backup controls
    """
    group = QGroupBox("Reset Settings")
    layout = QVBoxLayout(group)
    
    desc = QLabel(
        "Reset all settings to defaults or backup/restore your configuration."
    )
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    btn_row = QHBoxLayout()
    
    reset_btn = QPushButton("Reset to Defaults")
    reset_btn.setToolTip("Reset all settings to factory defaults")
    reset_btn.setStyleSheet("color: #ef4444;")
    reset_btn.clicked.connect(lambda: reset_func(parent))
    btn_row.addWidget(reset_btn)
    
    export_btn = QPushButton("Export Settings")
    export_btn.setToolTip("Save settings to a file")
    export_btn.clicked.connect(lambda: export_func(parent))
    btn_row.addWidget(export_btn)
    
    import_btn = QPushButton("Import Settings")
    import_btn.setToolTip("Load settings from a file")
    import_btn.clicked.connect(lambda: import_func(parent))
    btn_row.addWidget(import_btn)
    
    btn_row.addStretch()
    layout.addLayout(btn_row)
    
    parent.reset_status_label = QLabel("")
    parent.reset_status_label.setStyleSheet("color: #bac2de; font-style: italic;")
    layout.addWidget(parent.reset_status_label)
    
    return group


def build_game_mode_section(parent, sync_func, toggle_aggressive_func, manual_toggle_func) -> QGroupBox:
    """Build the Game Mode section for zero-lag gaming.
    
    Returns:
        QGroupBox containing game mode controls
    """
    group = QGroupBox("Game Mode - Zero Lag Gaming")
    layout = QVBoxLayout(group)
    
    desc = QLabel(
        "Game Mode automatically detects when you're gaming and reduces AI resource usage "
        "to prevent frame drops. AI stays responsive but uses minimal CPU/GPU."
    )
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Enable game mode
    enable_row = QHBoxLayout()
    parent.game_mode_checkbox = QCheckBox("Enable Game Mode")
    parent.game_mode_checkbox.setToolTip("Auto-detect games and reduce AI resource usage")
    parent.game_mode_checkbox.stateChanged.connect(lambda state: sync_func(parent, state))
    enable_row.addWidget(parent.game_mode_checkbox)
    
    parent.game_mode_status_label = QLabel("Game Mode: Disabled")
    parent.game_mode_status_label.setStyleSheet("color: #bac2de; font-style: italic;")
    enable_row.addWidget(parent.game_mode_status_label)
    enable_row.addStretch()
    layout.addLayout(enable_row)
    
    # Aggressive mode option
    aggressive_row = QHBoxLayout()
    parent.game_mode_aggressive_checkbox = QCheckBox("Aggressive Mode (maximum performance)")
    parent.game_mode_aggressive_checkbox.setToolTip(
        "Maximum performance: AI uses absolute minimum resources.\n"
        "Balanced: AI can do light background tasks."
    )
    parent.game_mode_aggressive_checkbox.stateChanged.connect(
        lambda state: toggle_aggressive_func(parent, state)
    )
    aggressive_row.addWidget(parent.game_mode_aggressive_checkbox)
    aggressive_row.addStretch()
    layout.addLayout(aggressive_row)
    
    # Current limits display
    limits_frame = QGroupBox("Current Limits")
    limits_layout = QVBoxLayout(limits_frame)
    
    parent.game_mode_limits_label = QLabel(
        "CPU: <100%, GPU: Allowed, Background Tasks: Enabled"
    )
    parent.game_mode_limits_label.setStyleSheet("color: #bac2de; font-size: 12px;")
    limits_layout.addWidget(parent.game_mode_limits_label)
    
    layout.addWidget(limits_frame)
    
    # Manual toggle button
    manual_row = QHBoxLayout()
    parent.game_mode_manual_btn = QPushButton("Toggle Game Mode Manually")
    parent.game_mode_manual_btn.setToolTip("Manually enable/disable game mode without auto-detection")
    parent.game_mode_manual_btn.clicked.connect(lambda: manual_toggle_func(parent))
    manual_row.addWidget(parent.game_mode_manual_btn)
    manual_row.addStretch()
    layout.addLayout(manual_row)
    
    return group


def build_display_settings_section(parent, apply_font_size_func, apply_font_scale_func, 
                                   apply_theme_func, init_func) -> QGroupBox:
    """Build the Display Settings section.
    
    Returns:
        QGroupBox containing display/theme controls
    """
    group = QGroupBox("Display Settings")
    layout = QVBoxLayout(group)

    desc = QLabel(
        "Adjust font size, theme, and overall appearance. "
        "Changes apply immediately and are saved automatically."
    )
    desc.setWordWrap(True)
    layout.addWidget(desc)

    # Font size (in pixels)
    font_size_row = QHBoxLayout()
    font_size_row.addWidget(QLabel("Font Size (px):"))

    parent.font_size_spinbox = QSpinBox()
    parent.font_size_spinbox.setToolTip("Base font size in pixels (default: 13)")
    parent.font_size_spinbox.setRange(8, 32)
    parent.font_size_spinbox.setValue(13)
    parent.font_size_spinbox.setSuffix(" px")
    parent.font_size_spinbox.valueChanged.connect(lambda val: apply_font_size_func(parent, val))
    font_size_row.addWidget(parent.font_size_spinbox)
    font_size_row.addStretch()
    layout.addLayout(font_size_row)

    # Font scale (percentage multiplier)
    scale_row = QHBoxLayout()
    scale_row.addWidget(QLabel("Font Scale:"))

    parent.font_scale_combo = NoScrollComboBox()
    parent.font_scale_combo.setToolTip("Adjust text size throughout the application")
    parent.font_scale_combo.addItem("Tiny (75%)", 0.75)
    parent.font_scale_combo.addItem("Small (90%)", 0.9)
    parent.font_scale_combo.addItem("Normal (100%)", 1.0)
    parent.font_scale_combo.addItem("Large (120%)", 1.2)
    parent.font_scale_combo.addItem("Extra Large (150%)", 1.5)
    parent.font_scale_combo.addItem("Huge (200%)", 2.0)
    parent.font_scale_combo.setCurrentIndex(2)
    parent.font_scale_combo.currentIndexChanged.connect(lambda idx: apply_font_scale_func(parent))
    scale_row.addWidget(parent.font_scale_combo)
    scale_row.addStretch()
    layout.addLayout(scale_row)

    # Theme selector
    theme_row = QHBoxLayout()
    theme_row.addWidget(QLabel("Theme:"))

    parent.theme_combo = NoScrollComboBox()
    parent.theme_combo.setToolTip("Select application theme")
    parent.theme_combo.addItem("Dark", "dark")
    parent.theme_combo.addItem("Cerulean", "cerulean")
    parent.theme_combo.addItem("Cerulean Light", "cerulean_light")
    parent.theme_combo.addItem("Midnight", "midnight")
    parent.theme_combo.addItem("Shadow", "shadow")
    parent.theme_combo.addItem("Ocean Cerulean", "ocean_cerulean")
    parent.theme_combo.currentIndexChanged.connect(lambda idx: apply_theme_func(parent))
    theme_row.addWidget(parent.theme_combo)
    theme_row.addStretch()
    layout.addLayout(theme_row)

    # Theme description
    parent.theme_description_label = QLabel("Classic dark theme with soft grays")
    parent.theme_description_label.setStyleSheet("color: #bac2de; font-style: italic;")
    layout.addWidget(parent.theme_description_label)

    # Initialize from saved settings
    init_func(parent)
    
    return group


def build_cloud_ai_section(parent, toggle_func, update_models_func, go_to_tab_func) -> QGroupBox:
    """Build the Cloud/API AI Mode section.
    
    Returns:
        QGroupBox containing cloud AI provider controls
    """
    group = QGroupBox("Cloud/API AI Mode")
    layout = QVBoxLayout(group)
    
    desc = QLabel(
        "<b>FREE option:</b> Ollama runs AI locally (no API key needed!)<br>"
        "<b>Paid options:</b> GPT-4, Claude, Gemini (need API keys)<br>"
        "Perfect for Raspberry Pi - use powerful AI without training!"
    )
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Enable cloud mode
    parent.cloud_mode_check = QCheckBox("Enable Cloud/API AI for Chat")
    parent.cloud_mode_check.setToolTip(
        "Route chat to API instead of local trained model.\n"
        "Ollama is FREE! Others need API keys."
    )
    parent.cloud_mode_check.stateChanged.connect(lambda state: toggle_func(parent, state))
    layout.addWidget(parent.cloud_mode_check)
    
    # Provider selection
    provider_row = QHBoxLayout()
    provider_row.addWidget(QLabel("Provider:"))
    parent.cloud_provider_combo = NoScrollComboBox()
    parent.cloud_provider_combo.setToolTip("Select AI provider")
    parent.cloud_provider_combo.addItem("[FREE] Ollama (Local)", "ollama")
    parent.cloud_provider_combo.addItem("OpenAI (GPT-4)", "openai")
    parent.cloud_provider_combo.addItem("Anthropic (Claude)", "anthropic")
    parent.cloud_provider_combo.addItem("Google (Gemini - Free tier)", "google")
    parent.cloud_provider_combo.setEnabled(False)
    parent.cloud_provider_combo.currentIndexChanged.connect(lambda: update_models_func(parent))
    provider_row.addWidget(parent.cloud_provider_combo)
    provider_row.addStretch()
    layout.addLayout(provider_row)
    
    # Model selection
    model_row = QHBoxLayout()
    model_row.addWidget(QLabel("Model:"))
    parent.cloud_model_combo = NoScrollComboBox()
    parent.cloud_model_combo.setToolTip("Select AI model")
    parent.cloud_model_combo.addItem("llama3.2:1b (Fast, FREE)", "llama3.2:1b")
    parent.cloud_model_combo.addItem("llama3.2:3b (Better, FREE)", "llama3.2:3b")
    parent.cloud_model_combo.addItem("mistral:7b (Quality, FREE)", "mistral:7b")
    parent.cloud_model_combo.setEnabled(False)
    model_row.addWidget(parent.cloud_model_combo)
    model_row.addStretch()
    layout.addLayout(model_row)
    
    # Status
    parent.cloud_status_label = QLabel("For Ollama: Install from https://ollama.ai then run: ollama run llama3.2:1b")
    parent.cloud_status_label.setStyleSheet("color: #bac2de; font-style: italic;")
    parent.cloud_status_label.setWordWrap(True)
    layout.addWidget(parent.cloud_status_label)
    
    # Link to Modules tab
    link_row = QHBoxLayout()
    go_btn = QPushButton("Open Modules Tab")
    go_btn.setToolTip("Manage all AI modules and providers")
    go_btn.clicked.connect(lambda: go_to_tab_func(parent, "Modules"))
    link_row.addWidget(go_btn)
    link_row.addStretch()
    layout.addLayout(link_row)
    
    return group


def build_chat_names_section(parent, save_func, load_func) -> QGroupBox:
    """Build the Chat Names section.
    
    Returns:
        QGroupBox containing name customization
    """
    group = QGroupBox("Chat Names")
    layout = QVBoxLayout(group)
    layout.setSpacing(6)
    
    desc = QLabel("Customize how you and the AI appear in chat.")
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # User display name
    user_row = QHBoxLayout()
    user_row.addWidget(QLabel("Your Name:"))
    parent.user_display_name_input = QLineEdit()
    parent.user_display_name_input.setPlaceholderText("You")
    parent.user_display_name_input.setMaximumWidth(200)
    parent.user_display_name_input.textChanged.connect(lambda text: save_func(parent))
    user_row.addWidget(parent.user_display_name_input)
    user_row.addStretch()
    layout.addLayout(user_row)
    
    # Load saved name
    load_func(parent)
    
    note = QLabel("The AI's display name is automatically set to the loaded model name.")
    note.setStyleSheet("color: #bac2de; font-style: italic; font-size: 12px;")
    note.setWordWrap(True)
    layout.addWidget(note)
    
    return group


def build_context_window_section(parent, save_func, load_func) -> QGroupBox:
    """Build the Context Window Settings section.
    
    Returns:
        QGroupBox containing context/memory settings
    """
    group = QGroupBox("Context Window - Memory and Auto-Continue")
    layout = QVBoxLayout(group)
    layout.setSpacing(8)
    
    desc = QLabel(
        "Control how the AI handles its context memory. When context fills up, the AI may "
        "forget earlier messages, leading to inconsistent responses (hallucinations)."
    )
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Show token counter checkbox
    parent.show_tokens_check = QCheckBox("Show token counter in chat")
    parent.show_tokens_check.setToolTip("Display a token usage bar above the chat input")
    parent.show_tokens_check.stateChanged.connect(lambda: save_func(parent))
    layout.addWidget(parent.show_tokens_check)
    
    # Auto-continue checkbox
    parent.auto_continue_check = QCheckBox("Auto-continue when context is full")
    parent.auto_continue_check.setToolTip("Automatically start a new context when full")
    parent.auto_continue_check.stateChanged.connect(lambda: save_func(parent))
    layout.addWidget(parent.auto_continue_check)
    
    # Context threshold slider
    threshold_row = QHBoxLayout()
    threshold_row.addWidget(QLabel("Warning at:"))
    parent.context_threshold_slider = QSlider(Qt.Orientation.Horizontal)
    parent.context_threshold_slider.setRange(50, 95)
    parent.context_threshold_slider.setValue(80)
    parent.context_threshold_slider.setToolTip("Show warning when context reaches this percentage")
    parent.context_threshold_slider.valueChanged.connect(lambda: save_func(parent))
    threshold_row.addWidget(parent.context_threshold_slider)
    parent.context_threshold_label = QLabel("80%")
    threshold_row.addWidget(parent.context_threshold_label)
    layout.addLayout(threshold_row)
    
    # Status label
    parent.context_settings_status = QLabel("")
    parent.context_settings_status.setStyleSheet("color: #bac2de; font-style: italic;")
    layout.addWidget(parent.context_settings_status)
    
    # Load saved settings
    load_func(parent)
    
    return group


def build_window_options_section(parent, toggle_on_top_func, mini_chat_on_top_func,
                                  populate_monitors_func, move_to_monitor_func,
                                  save_startup_func, load_startup_func, update_info_func) -> QGroupBox:
    """Build the Window Options section.
    
    Returns:
        QGroupBox containing window behavior settings
    """
    group = QGroupBox("Window Options")
    layout = QVBoxLayout(group)
    
    desc = QLabel("Configure window behavior and multi-monitor options.")
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Always on top checkbox
    parent.always_on_top_check = QCheckBox("Always on Top (Main Window)")
    parent.always_on_top_check.setToolTip("Keep the main window above other windows")
    parent.always_on_top_check.stateChanged.connect(lambda state: toggle_on_top_func(parent, state))
    layout.addWidget(parent.always_on_top_check)
    
    # Quick Chat always on top
    parent.mini_chat_on_top_check = QCheckBox("Quick Chat Always on Top")
    parent.mini_chat_on_top_check.setToolTip("Keep Quick Chat above other windows (default: on)")
    parent.mini_chat_on_top_check.setChecked(True)
    parent.mini_chat_on_top_check.stateChanged.connect(lambda state: mini_chat_on_top_func(parent, state))
    layout.addWidget(parent.mini_chat_on_top_check)
    
    # Monitor selection
    monitor_row = QHBoxLayout()
    monitor_row.addWidget(QLabel("Display:"))
    
    parent.monitor_combo = NoScrollComboBox()
    parent.monitor_combo.setToolTip("Select display for Quick Chat window")
    populate_monitors_func(parent)
    parent.monitor_combo.currentIndexChanged.connect(lambda idx: move_to_monitor_func(parent, idx))
    monitor_row.addWidget(parent.monitor_combo)
    
    refresh_btn = QPushButton("Refresh")
    refresh_btn.setMaximumWidth(70)
    refresh_btn.clicked.connect(lambda: populate_monitors_func(parent, preserve_selection=True))
    monitor_row.addWidget(refresh_btn)
    monitor_row.addStretch()
    layout.addLayout(monitor_row)
    
    # Startup position
    startup_row = QHBoxLayout()
    startup_row.addWidget(QLabel("Startup Position:"))
    parent.startup_position_combo = NoScrollComboBox()
    parent.startup_position_combo.setToolTip("Where the main window appears on startup")
    parent.startup_position_combo.addItem("Center on Display", "center")
    parent.startup_position_combo.addItem("Remember Last Position", "remember")
    parent.startup_position_combo.currentIndexChanged.connect(lambda idx: save_startup_func(parent))
    startup_row.addWidget(parent.startup_position_combo)
    startup_row.addStretch()
    layout.addLayout(startup_row)
    
    load_startup_func(parent)
    
    # Display info
    parent.display_info_label = QLabel("")
    parent.display_info_label.setStyleSheet("color: #bac2de; font-style: italic;")
    update_info_func(parent)
    layout.addWidget(parent.display_info_label)
    
    return group


def build_audio_device_section(parent, refresh_func, test_mic_func, 
                               toggle_profanity_func, load_profanity_func, go_to_tab_func) -> QGroupBox:
    """Build the Audio Device Settings section.
    
    Returns:
        QGroupBox containing audio device controls
    """
    group = QGroupBox("Audio Devices")
    layout = QVBoxLayout(group)
    
    desc = QLabel("Select input (microphone) and output (speaker) devices for voice features.")
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Refresh button
    refresh_row = QHBoxLayout()
    refresh_btn = QPushButton("Refresh Devices")
    refresh_btn.setToolTip("Scan for available audio devices")
    refresh_btn.clicked.connect(lambda: refresh_func(parent))
    refresh_row.addWidget(refresh_btn)
    refresh_row.addStretch()
    layout.addLayout(refresh_row)
    
    # Input device
    input_row = QHBoxLayout()
    input_row.addWidget(QLabel("Microphone:"))
    parent.audio_input_combo = NoScrollComboBox()
    parent.audio_input_combo.setMinimumWidth(250)
    parent.audio_input_combo.setToolTip("Select microphone for voice input")
    input_row.addWidget(parent.audio_input_combo)
    input_row.addStretch()
    layout.addLayout(input_row)
    
    # Output device
    output_row = QHBoxLayout()
    output_row.addWidget(QLabel("Speaker:"))
    parent.audio_output_combo = NoScrollComboBox()
    parent.audio_output_combo.setMinimumWidth(250)
    parent.audio_output_combo.setToolTip("Select speaker for voice output")
    output_row.addWidget(parent.audio_output_combo)
    output_row.addStretch()
    layout.addLayout(output_row)
    
    # Mic test
    mic_row = QHBoxLayout()
    parent.mic_test_btn = QPushButton("Test Microphone")
    parent.mic_test_btn.clicked.connect(lambda: test_mic_func(parent))
    mic_row.addWidget(parent.mic_test_btn)
    
    parent.mic_level_bar = QSlider(Qt.Orientation.Horizontal)
    parent.mic_level_bar.setRange(0, 100)
    parent.mic_level_bar.setValue(0)
    parent.mic_level_bar.setEnabled(False)
    mic_row.addWidget(parent.mic_level_bar)
    
    parent.mic_status_label = QLabel("Not tested")
    parent.mic_status_label.setStyleSheet("color: #bac2de;")
    mic_row.addWidget(parent.mic_status_label)
    mic_row.addStretch()
    layout.addLayout(mic_row)
    
    # Profanity filter
    profanity_row = QHBoxLayout()
    parent.profanity_filter_checkbox = QCheckBox("Enable Profanity Filter")
    parent.profanity_filter_checkbox.setChecked(True)
    parent.profanity_filter_checkbox.setToolTip("Filter profanity from voice transcription")
    parent.profanity_filter_checkbox.stateChanged.connect(lambda state: toggle_profanity_func(parent, state))
    profanity_row.addWidget(parent.profanity_filter_checkbox)
    profanity_row.addStretch()
    layout.addLayout(profanity_row)
    
    load_profanity_func(parent)
    
    # Link to Audio tab
    link_row = QHBoxLayout()
    go_btn = QPushButton("Open Audio Tab")
    go_btn.setToolTip("Generate speech and audio in the Audio tab")
    go_btn.clicked.connect(lambda: go_to_tab_func(parent, "Audio"))
    link_row.addWidget(go_btn)
    link_row.addStretch()
    layout.addLayout(link_row)
    
    return group


def build_autonomous_mode_section(parent, sync_func) -> QGroupBox:
    """Build the Autonomous Mode Settings section.
    
    Returns:
        QGroupBox containing autonomous mode controls
    """
    group = QGroupBox("Autonomous Mode Settings")
    layout = QVBoxLayout(group)
    
    note = QLabel(
        "Toggle Autonomous Mode in Quick Settings above. "
        "Configure advanced settings here."
    )
    note.setWordWrap(True)
    note.setStyleSheet("color: #a6adc8; font-style: italic;")
    layout.addWidget(note)
    
    # Activity level
    settings_row = QHBoxLayout()
    settings_row.addWidget(QLabel("Activity Level:"))
    
    parent.autonomous_activity_spin = QSpinBox()
    parent.autonomous_activity_spin.setRange(1, 20)
    parent.autonomous_activity_spin.setValue(12)
    parent.autonomous_activity_spin.setSuffix(" actions/hour")
    parent.autonomous_activity_spin.setToolTip("How many autonomous actions per hour")
    settings_row.addWidget(parent.autonomous_activity_spin)
    settings_row.addStretch()
    layout.addLayout(settings_row)
    
    # Reference checkbox (synced with quick settings)
    parent.autonomous_enabled_check = QCheckBox("Enable Autonomous Mode")
    parent.autonomous_enabled_check.stateChanged.connect(lambda state: sync_func(parent, state))
    layout.addWidget(parent.autonomous_enabled_check)
    
    return group


def build_web_server_section(parent, toggle_func) -> QGroupBox:
    """Build the Web Server section.
    
    Returns:
        QGroupBox containing web server controls
    """
    group = QGroupBox("Web Interface - Remote Access")
    layout = QVBoxLayout(group)
    
    desc = QLabel(
        "Access Enigma AI Engine from any device on your local network. "
        "Works with phones, tablets, and other computers without cloud services."
    )
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Enable checkbox
    parent.web_server_checkbox = QCheckBox("Enable Web Server")
    parent.web_server_checkbox.setToolTip("Start web server for remote access")
    parent.web_server_checkbox.stateChanged.connect(lambda state: toggle_func(parent, state))
    layout.addWidget(parent.web_server_checkbox)
    
    # Port selection
    port_row = QHBoxLayout()
    port_row.addWidget(QLabel("Port:"))
    parent.web_port_spin = QSpinBox()
    parent.web_port_spin.setRange(1024, 65535)
    parent.web_port_spin.setValue(5000)
    parent.web_port_spin.setToolTip("Port for web server (default: 5000)")
    port_row.addWidget(parent.web_port_spin)
    port_row.addStretch()
    layout.addLayout(port_row)
    
    # Status label
    parent.web_status_label = QLabel("Web server not running")
    parent.web_status_label.setStyleSheet("color: #bac2de; font-style: italic;")
    layout.addWidget(parent.web_status_label)
    
    return group


def build_system_prompt_section(parent, apply_preset_func, save_func, save_as_func,
                                 delete_func, reset_func, load_presets_func, load_prompt_func) -> QGroupBox:
    """Build the System Prompt section.
    
    Returns:
        QGroupBox containing system prompt editor
    """
    group = QGroupBox("System Prompt")
    layout = QVBoxLayout(group)
    
    desc = QLabel(
        "Customize the system prompt that tells the AI how to behave. "
        "This affects both the main Chat and Quick Chat."
    )
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Preset selector
    preset_row = QHBoxLayout()
    preset_row.addWidget(QLabel("Preset:"))
    parent.system_prompt_preset = NoScrollComboBox()
    parent.system_prompt_preset.setToolTip("Select system prompt preset")
    parent.system_prompt_preset.addItem("Simple (recommended for small models)", "simple")
    parent.system_prompt_preset.addItem("Full (with tools, for larger models)", "full")
    parent.system_prompt_preset.addItem("Enigma Engine Complete (avatar, vision, tools)", "enigma_engine_full")
    parent.system_prompt_preset.addItem("Custom", "custom")
    load_presets_func(parent)
    parent.system_prompt_preset.currentIndexChanged.connect(lambda: apply_preset_func(parent))
    preset_row.addWidget(parent.system_prompt_preset)
    preset_row.addStretch()
    layout.addLayout(preset_row)
    
    # Custom prompt text area
    parent.custom_system_prompt = QTextEdit()
    parent.custom_system_prompt.setPlaceholderText(
        "Enter your custom system prompt here...\n\n"
        "Example: You are a helpful AI assistant. Be friendly and concise."
    )
    parent.custom_system_prompt.setMaximumHeight(120)
    parent.custom_system_prompt.setStyleSheet("""
        QTextEdit {
            background-color: #2d2d2d;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 8px;
            font-family: monospace;
        }
    """)
    layout.addWidget(parent.custom_system_prompt)
    
    # Buttons
    btn_row = QHBoxLayout()
    
    save_btn = QPushButton("Save")
    save_btn.setToolTip("Save current prompt to selected preset")
    save_btn.clicked.connect(lambda: save_func(parent))
    btn_row.addWidget(save_btn)
    
    save_as_btn = QPushButton("Save As New Preset")
    save_as_btn.setToolTip("Save current prompt as a new custom preset")
    save_as_btn.clicked.connect(lambda: save_as_func(parent))
    btn_row.addWidget(save_as_btn)
    
    delete_btn = QPushButton("Delete Preset")
    delete_btn.setToolTip("Delete the selected custom preset")
    delete_btn.clicked.connect(lambda: delete_func(parent))
    btn_row.addWidget(delete_btn)
    
    reset_btn = QPushButton("Reset")
    reset_btn.setToolTip("Reset to Simple preset")
    reset_btn.clicked.connect(lambda: reset_func(parent))
    btn_row.addWidget(reset_btn)
    
    btn_row.addStretch()
    layout.addLayout(btn_row)
    
    parent.prompt_status_label = QLabel("")
    parent.prompt_status_label.setStyleSheet("color: #bac2de; font-style: italic;")
    layout.addWidget(parent.prompt_status_label)
    
    load_prompt_func(parent)
    
    return group


def build_connection_status_section(parent, create_indicator_func, refresh_func) -> QGroupBox:
    """Build the AI Connection Status section.
    
    Returns:
        QGroupBox containing connection status indicators
    """
    group = QGroupBox("AI Connection Status")
    layout = QVBoxLayout(group)
    
    desc = QLabel("Shows what AI is currently active and which components are loaded.")
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Active AI display
    parent.active_ai_label = QLabel("Active AI: Not configured")
    parent.active_ai_label.setStyleSheet(
        "font-weight: bold; font-size: 12px; padding: 5px; "
        "background: #1e1e2e; border-radius: 4px;"
    )
    layout.addWidget(parent.active_ai_label)
    
    # Connection indicators
    parent.connection_indicators = QWidget()
    conn_grid = QHBoxLayout(parent.connection_indicators)
    conn_grid.setSpacing(12)
    
    parent.model_status = create_indicator_func("Model", "disconnected")
    parent.tokenizer_status = create_indicator_func("Tokenizer", "disconnected")
    parent.inference_status = create_indicator_func("Inference", "disconnected")
    parent.memory_status = create_indicator_func("Memory", "disconnected")
    
    conn_grid.addWidget(parent.model_status)
    conn_grid.addWidget(parent.tokenizer_status)
    conn_grid.addWidget(parent.inference_status)
    conn_grid.addWidget(parent.memory_status)
    conn_grid.addStretch()
    
    layout.addWidget(parent.connection_indicators)
    
    # Refresh button
    refresh_btn = QPushButton("Check Connections")
    refresh_btn.clicked.connect(lambda: refresh_func(parent))
    layout.addWidget(refresh_btn)
    
    return group


def build_current_status_section(parent, refresh_func) -> QGroupBox:
    """Build the Current Status section.
    
    Returns:
        QGroupBox containing power/resource status display
    """
    group = QGroupBox("Current Status")
    layout = QVBoxLayout(group)
    
    parent.power_status = QTextEdit()
    parent.power_status.setReadOnly(True)
    parent.power_status.setMaximumHeight(150)
    parent.power_status.setStyleSheet("font-family: Consolas, monospace;")
    layout.addWidget(parent.power_status)
    
    refresh_btn = QPushButton("Refresh Status")
    refresh_btn.clicked.connect(lambda: refresh_func(parent))
    layout.addWidget(refresh_btn)
    
    return group


def build_hotkey_config_section(parent, toggle_func) -> QGroupBox:
    """Build the Global Hotkeys section.
    
    Returns:
        QGroupBox containing hotkey configuration
    """
    group = QGroupBox("Global Hotkeys")
    layout = QVBoxLayout(group)
    
    desc = QLabel(
        "Configure global keyboard shortcuts that work even when Enigma AI Engine is not focused.\n"
        "These work in fullscreen games and other applications."
    )
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Enable/disable
    enable_row = QHBoxLayout()
    parent.hotkeys_enabled_check = QCheckBox("Enable global hotkeys")
    parent.hotkeys_enabled_check.setChecked(True)
    parent.hotkeys_enabled_check.stateChanged.connect(lambda state: toggle_func(parent, state))
    enable_row.addWidget(parent.hotkeys_enabled_check)
    
    parent.hotkeys_status_label = QLabel("")
    parent.hotkeys_status_label.setStyleSheet("color: #bac2de; font-style: italic;")
    enable_row.addWidget(parent.hotkeys_status_label)
    enable_row.addStretch()
    layout.addLayout(enable_row)
    
    # Try to add hotkey config widget
    try:
        from ..widgets.hotkey_config import HotkeyConfigWidget
        parent.hotkey_config_widget = HotkeyConfigWidget()
        layout.addWidget(parent.hotkey_config_widget)
    except ImportError:
        layout.addWidget(QLabel("Hotkey configuration widget not available"))
    
    return group


def build_overlay_settings_section(parent, toggle_func, save_funcs: dict) -> QGroupBox:
    """Build the AI Overlay Settings section.
    
    Returns:
        QGroupBox containing overlay configuration
    """
    group = QGroupBox("AI Overlay - Gaming & Multitasking Interface")
    layout = QVBoxLayout(group)
    
    desc = QLabel(
        "The AI overlay floats on top of games and other apps, providing quick AI access "
        "without leaving your current application."
    )
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Enable overlay
    parent.overlay_enabled_check = QCheckBox("Enable AI Overlay")
    parent.overlay_enabled_check.setToolTip("Show floating AI overlay on demand")
    parent.overlay_enabled_check.stateChanged.connect(lambda state: toggle_func(parent, state))
    layout.addWidget(parent.overlay_enabled_check)
    
    # Opacity slider
    opacity_row = QHBoxLayout()
    opacity_row.addWidget(QLabel("Opacity:"))
    parent.overlay_opacity_slider = QSlider(Qt.Orientation.Horizontal)
    parent.overlay_opacity_slider.setRange(20, 100)
    parent.overlay_opacity_slider.setValue(90)
    parent.overlay_opacity_slider.setToolTip("Overlay transparency (20-100%)")
    parent.overlay_opacity_slider.valueChanged.connect(lambda: save_funcs.get('save', lambda p: None)(parent))
    opacity_row.addWidget(parent.overlay_opacity_slider)
    parent.overlay_opacity_label = QLabel("90%")
    opacity_row.addWidget(parent.overlay_opacity_label)
    layout.addLayout(opacity_row)
    
    # Position selector
    pos_row = QHBoxLayout()
    pos_row.addWidget(QLabel("Position:"))
    parent.overlay_position_combo = NoScrollComboBox()
    parent.overlay_position_combo.addItem("Top Right", "top_right")
    parent.overlay_position_combo.addItem("Top Left", "top_left")
    parent.overlay_position_combo.addItem("Bottom Right", "bottom_right")
    parent.overlay_position_combo.addItem("Bottom Left", "bottom_left")
    parent.overlay_position_combo.addItem("Center", "center")
    parent.overlay_position_combo.currentIndexChanged.connect(
        lambda: save_funcs.get('save', lambda p: None)(parent)
    )
    pos_row.addWidget(parent.overlay_position_combo)
    pos_row.addStretch()
    layout.addLayout(pos_row)
    
    return group


def build_fullscreen_visibility_section(parent, save_func, load_func, toggle_func, refresh_func) -> QGroupBox:
    """Build the Fullscreen Visibility Settings section.
    
    Returns:
        QGroupBox containing fullscreen detection and visibility controls
    """
    group = QGroupBox("Fullscreen Visibility Settings")
    layout = QVBoxLayout(group)
    
    desc = QLabel(
        "Control what happens when a fullscreen application is detected. "
        "Hide or fade avatar and effects during fullscreen apps."
    )
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Auto-hide checkbox
    parent.fs_auto_hide_check = QCheckBox("Auto-hide when fullscreen detected")
    parent.fs_auto_hide_check.setToolTip("Automatically hide elements when a fullscreen app is detected")
    parent.fs_auto_hide_check.setChecked(True)
    parent.fs_auto_hide_check.stateChanged.connect(lambda: save_func(parent))
    layout.addWidget(parent.fs_auto_hide_check)
    
    # Fade transition
    parent.fs_fade_check = QCheckBox("Use fade transitions")
    parent.fs_fade_check.setToolTip("Smoothly fade elements instead of instant hide/show")
    parent.fs_fade_check.setChecked(True)
    parent.fs_fade_check.stateChanged.connect(lambda: save_func(parent))
    layout.addWidget(parent.fs_fade_check)
    
    # Category toggles
    cat_label = QLabel("Visible Categories:")
    cat_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
    layout.addWidget(cat_label)
    
    cat_row = QHBoxLayout()
    
    parent.fs_avatar_check = QCheckBox("Avatar")
    parent.fs_avatar_check.setChecked(True)
    parent.fs_avatar_check.stateChanged.connect(lambda: save_func(parent))
    cat_row.addWidget(parent.fs_avatar_check)
    
    parent.fs_objects_check = QCheckBox("Objects")
    parent.fs_objects_check.setChecked(True)
    parent.fs_objects_check.stateChanged.connect(lambda: save_func(parent))
    cat_row.addWidget(parent.fs_objects_check)
    
    parent.fs_effects_check = QCheckBox("Effects")
    parent.fs_effects_check.setChecked(True)
    parent.fs_effects_check.stateChanged.connect(lambda: save_func(parent))
    cat_row.addWidget(parent.fs_effects_check)
    
    parent.fs_particles_check = QCheckBox("Particles")
    parent.fs_particles_check.setChecked(True)
    parent.fs_particles_check.stateChanged.connect(lambda: save_func(parent))
    cat_row.addWidget(parent.fs_particles_check)
    
    cat_row.addStretch()
    layout.addLayout(cat_row)
    
    # Hotkey row
    hotkey_row = QHBoxLayout()
    hotkey_row.addWidget(QLabel("Toggle Hotkey:"))
    
    parent.fs_hotkey_input = QLineEdit()
    parent.fs_hotkey_input.setPlaceholderText("e.g., ctrl+shift+h")
    parent.fs_hotkey_input.setMaximumWidth(150)
    parent.fs_hotkey_input.editingFinished.connect(lambda: save_func(parent))
    hotkey_row.addWidget(parent.fs_hotkey_input)
    
    toggle_btn = QPushButton("Toggle Now")
    toggle_btn.clicked.connect(lambda: toggle_func(parent))
    hotkey_row.addWidget(toggle_btn)
    
    refresh_btn = QPushButton("Refresh")
    refresh_btn.clicked.connect(lambda: refresh_func(parent))
    hotkey_row.addWidget(refresh_btn)
    
    hotkey_row.addStretch()
    layout.addLayout(hotkey_row)
    
    # Status
    parent.fs_status_label = QLabel("Not initialized")
    parent.fs_status_label.setStyleSheet("color: #bac2de; font-style: italic;")
    layout.addWidget(parent.fs_status_label)
    
    load_func(parent)
    
    return group


def build_federated_learning_section(parent) -> Optional[QGroupBox]:
    """Build the Federated Learning section.
    
    Returns:
        QGroupBox containing federated learning controls, or None if unavailable
    """
    try:
        from ..widgets.federated_widget import FederatedLearningWidget
        
        group = QGroupBox("Federated Learning")
        layout = QVBoxLayout(group)
        
        desc = QLabel(
            "Learn from collective experience without sharing your data. "
            "Only model improvements are shared, never raw conversations."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #bac2de; font-style: italic;")
        layout.addWidget(desc)
        
        parent.federated_widget = FederatedLearningWidget(parent)
        layout.addWidget(parent.federated_widget)  # type: ignore[arg-type]
        
        return group
    except Exception:
        return None
