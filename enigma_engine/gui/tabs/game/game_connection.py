"""
Game Connection Module

Allows the AI to connect to and control games/applications.
The same AI being trained learns to interact with these games.

Supports loading connection configs from files instead of presets.
"""

import json
from pathlib import Path

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ....config import CONFIG
from ..shared_components import NoScrollComboBox


class CustomGameDialog(QDialog):
    """Dialog for configuring custom games for AI routing."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Custom Game")
        self.setMinimumWidth(450)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Form
        form = QFormLayout()
        
        # Basic info
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., My Game")
        form.addRow("Game Name:", self.name_input)
        
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("e.g., mygame (lowercase, no spaces)")
        form.addRow("Game ID:", self.id_input)
        
        # Game type
        self.type_combo = NoScrollComboBox()
        self.type_combo.addItems([
            "sandbox", "fps", "rpg", "strategy", "puzzle", 
            "sports", "simulation", "fighting", "platformer",
            "moba", "survival", "card", "other"
        ])
        self.type_combo.setToolTip("Select the game genre for AI behavior")
        form.addRow("Game Type:", self.type_combo)
        
        # Detection
        self.process_input = QLineEdit()
        self.process_input.setPlaceholderText("game.exe (optional)")
        self.process_input.setToolTip("Process name for auto-detection")
        form.addRow("Process Name:", self.process_input)
        
        self.window_input = QLineEdit()
        self.window_input.setPlaceholderText("Game Window Title (optional)")
        self.window_input.setToolTip("Window title for auto-detection")
        form.addRow("Window Title:", self.window_input)
        
        # AI Settings
        self.model_combo = NoScrollComboBox()
        self.model_combo.addItems(["nano", "micro", "tiny", "small", "medium", "large"])
        self.model_combo.setCurrentText("small")
        self.model_combo.setToolTip("AI model size to use for this game")
        form.addRow("Model Size:", self.model_combo)
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(
            "Enter a system prompt to customize AI behavior for this game...\n"
            "e.g., You are a helpful assistant for [Game]. You know about..."
        )
        self.prompt_input.setMaximumHeight(100)
        form.addRow("System Prompt:", self.prompt_input)
        
        # Options
        self.quick_responses = QCheckBox("Quick Responses (shorter replies)")
        self.quick_responses.setToolTip("Enable for fast-paced games")
        form.addRow("", self.quick_responses)
        
        self.voice_enabled = QCheckBox("Voice Enabled")
        self.voice_enabled.setChecked(True)
        self.voice_enabled.setToolTip("Enable voice interaction for this game")
        form.addRow("", self.voice_enabled)
        
        self.multiplayer_aware = QCheckBox("Multiplayer Aware")
        self.multiplayer_aware.setToolTip("AI considers multiplayer context")
        form.addRow("", self.multiplayer_aware)
        
        # Wiki URL for knowledge
        self.wiki_input = QLineEdit()
        self.wiki_input.setPlaceholderText("https://wiki.example.com (optional)")
        self.wiki_input.setToolTip("Wiki URL for web lookups")
        form.addRow("Wiki URL:", self.wiki_input)
        
        layout.addLayout(form)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _validate_and_accept(self):
        """Validate inputs before accepting."""
        name = self.name_input.text().strip()
        game_id = self.id_input.text().strip().lower().replace(" ", "_")
        
        if not name:
            QMessageBox.warning(self, "Validation Error", "Game name is required")
            return
        if not game_id:
            QMessageBox.warning(self, "Validation Error", "Game ID is required")
            return
        
        # Auto-generate ID if empty
        if not self.id_input.text().strip():
            self.id_input.setText(game_id)
        
        self.accept()
    
    def get_config(self) -> dict:
        """Get the game configuration from dialog inputs."""
        return {
            "name": self.name_input.text().strip(),
            "id": self.id_input.text().strip().lower().replace(" ", "_"),
            "type": self.type_combo.currentText(),
            "model": self.model_combo.currentText(),
            "system_prompt": self.prompt_input.toPlainText().strip(),
            "process_names": [self.process_input.text().strip()] if self.process_input.text().strip() else [],
            "window_titles": [self.window_input.text().strip()] if self.window_input.text().strip() else [],
            "quick_responses": self.quick_responses.isChecked(),
            "voice_enabled": self.voice_enabled.isChecked(),
            "multiplayer_aware": self.multiplayer_aware.isChecked(),
            "wiki_url": self.wiki_input.text().strip(),
        }


class GamingProfileDialog(QDialog):
    """Dialog for creating/editing gaming resource profiles."""
    
    def __init__(self, parent=None, profile=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Gaming Profile" if profile else "New Gaming Profile")
        self.setMinimumWidth(500)
        self.profile = profile
        self._setup_ui()
        if profile:
            self._load_profile(profile)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Form layout
        form = QFormLayout()
        
        # Basic info
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., My Favorite Game")
        form.addRow("Profile Name:", self.name_input)
        
        # Process names (comma-separated)
        self.process_input = QLineEdit()
        self.process_input.setPlaceholderText("game.exe, game2.exe (comma-separated)")
        self.process_input.setToolTip("Process names to detect - comma-separated")
        form.addRow("Process Names:", self.process_input)
        
        # Priority
        self.priority_combo = NoScrollComboBox()
        self.priority_combo.addItems(["BACKGROUND", "LOW", "MEDIUM", "HIGH", "FULL"])
        self.priority_combo.setCurrentText("HIGH")
        self.priority_combo.setToolTip(
            "BACKGROUND: AI uses minimal resources\n"
            "LOW: Reduced resources, defers heavy tasks\n"
            "MEDIUM: Balanced resources\n"
            "HIGH: More resources, most tasks allowed\n"
            "FULL: No restrictions"
        )
        form.addRow("Priority:", self.priority_combo)
        
        # Resource limits section
        resource_group = QGroupBox("Resource Limits")
        resource_layout = QFormLayout(resource_group)
        
        self.cpu_inference = QCheckBox("Force CPU inference")
        self.cpu_inference.setToolTip("Use CPU instead of GPU when this game is running")
        resource_layout.addRow("", self.cpu_inference)
        
        self.max_vram = QSpinBox()
        self.max_vram.setRange(0, 48000)
        self.max_vram.setValue(512)
        self.max_vram.setSuffix(" MB")
        self.max_vram.setToolTip("Max VRAM for AI (0 = unlimited)")
        resource_layout.addRow("Max VRAM:", self.max_vram)
        
        self.max_ram = QSpinBox()
        self.max_ram.setRange(0, 128000)
        self.max_ram.setValue(2048)
        self.max_ram.setSuffix(" MB")
        self.max_ram.setToolTip("Max RAM for AI (0 = unlimited)")
        resource_layout.addRow("Max RAM:", self.max_ram)
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(0, 64)
        self.batch_size.setValue(1)
        self.batch_size.setToolTip("Max batch size (0 = default)")
        resource_layout.addRow("Batch Size:", self.batch_size)
        
        form.addRow(resource_group)
        
        # Behavior options
        behavior_group = QGroupBox("Behavior")
        behavior_layout = QVBoxLayout(behavior_group)
        
        self.defer_heavy = QCheckBox("Defer heavy tasks until game ends")
        self.defer_heavy.setChecked(True)
        self.defer_heavy.setToolTip("Queue image/video generation until game closes")
        behavior_layout.addWidget(self.defer_heavy)
        
        self.voice_enabled = QCheckBox("Voice enabled")
        self.voice_enabled.setChecked(True)
        self.voice_enabled.setToolTip("Allow voice interaction while gaming")
        behavior_layout.addWidget(self.voice_enabled)
        
        self.avatar_enabled = QCheckBox("Avatar enabled")
        self.avatar_enabled.setChecked(True)
        self.avatar_enabled.setToolTip("Show avatar overlay while gaming")
        behavior_layout.addWidget(self.avatar_enabled)
        
        form.addRow(behavior_group)
        
        # Notes
        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText("Optional notes about this profile...")
        self.notes_input.setMaximumHeight(60)
        form.addRow("Notes:", self.notes_input)
        
        layout.addLayout(form)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _load_profile(self, profile):
        """Load existing profile into dialog."""
        self.name_input.setText(profile.name)
        self.process_input.setText(", ".join(profile.process_names))
        self.priority_combo.setCurrentText(profile.priority.name)
        self.cpu_inference.setChecked(profile.cpu_inference)
        self.max_vram.setValue(profile.max_vram_mb)
        self.max_ram.setValue(profile.max_ram_mb)
        self.batch_size.setValue(profile.batch_size)
        self.defer_heavy.setChecked(profile.defer_heavy_tasks)
        self.voice_enabled.setChecked(profile.voice_enabled)
        self.avatar_enabled.setChecked(profile.avatar_enabled)
        self.notes_input.setPlainText(profile.notes or "")
    
    def _validate_and_accept(self):
        """Validate inputs before accepting."""
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Validation Error", "Profile name is required")
            return
        if not self.process_input.text().strip():
            QMessageBox.warning(self, "Validation Error", "At least one process name is required")
            return
        self.accept()
    
    def get_profile(self):
        """Get the profile data from dialog inputs."""
        from enigma_engine.core.gaming_mode import GamingPriority, GamingProfile
        
        process_names = [
            p.strip() for p in self.process_input.text().split(",")
            if p.strip()
        ]
        
        return GamingProfile(
            name=self.name_input.text().strip(),
            process_names=process_names,
            priority=GamingPriority[self.priority_combo.currentText()],
            cpu_inference=self.cpu_inference.isChecked(),
            max_vram_mb=self.max_vram.value(),
            max_ram_mb=self.max_ram.value(),
            batch_size=self.batch_size.value(),
            defer_heavy_tasks=self.defer_heavy.isChecked(),
            voice_enabled=self.voice_enabled.isChecked(),
            avatar_enabled=self.avatar_enabled.isChecked(),
            notes=self.notes_input.toPlainText().strip(),
        )


# Game configs directory
GAME_CONFIG_DIR = Path(CONFIG["data_dir"]) / "game"


def create_game_subtab(parent):
    """
    Create the game connection sub-tab.
    
    The AI plays games WITH you using:
    - Input simulation (keyboard/mouse/controller)
    - Game-specific API connections
    - Vision to understand game state
    """
    widget = QWidget()
    layout = QVBoxLayout()
    layout.setSpacing(8)
    layout.setContentsMargins(10, 10, 10, 10)
    
    # === GAME AI ROUTING ===
    routing_group = QGroupBox("Game AI Routing")
    routing_layout = QVBoxLayout(routing_group)
    
    routing_desc = QLabel(
        "Different games use different AI behaviors. Auto-detect or select manually."
    )
    routing_desc.setWordWrap(True)
    routing_layout.addWidget(routing_desc)
    
    # Game detection toggle
    from PyQt5.QtWidgets import QCheckBox
    parent.auto_game_check = QCheckBox("Auto-Detect Running Game")
    parent.auto_game_check.setChecked(False)
    parent.auto_game_check.stateChanged.connect(
        lambda state: _toggle_game_detection(parent, state)
    )
    parent.auto_game_enabled = False
    routing_layout.addWidget(parent.auto_game_check)
    
    # Manual game selector
    game_select_row = QHBoxLayout()
    game_select_row.addWidget(QLabel("Active Game:"))
    parent.game_routing_combo = NoScrollComboBox()
    parent.game_routing_combo.setToolTip("Select the active game for AI routing")
    parent.game_routing_combo.addItem("(None)", "none")
    parent.game_routing_combo.addItem("Blender (Avatar)", "blender")
    parent.game_routing_combo.addItem("Minecraft", "minecraft")
    parent.game_routing_combo.addItem("Terraria", "terraria")
    parent.game_routing_combo.addItem("Valorant", "valorant")
    parent.game_routing_combo.addItem("League of Legends", "league")
    parent.game_routing_combo.addItem("Dark Souls", "darksouls")
    parent.game_routing_combo.addItem("Stardew Valley", "stardew")
    parent.game_routing_combo.addItem("Factorio", "factorio")
    parent.game_routing_combo.addItem("Custom...", "custom")
    # Use activated signal - only fires on explicit user selection, not when scrolling
    parent.game_routing_combo.activated.connect(
        lambda: _change_active_game(parent)
    )
    game_select_row.addWidget(parent.game_routing_combo)
    game_select_row.addStretch()
    routing_layout.addLayout(game_select_row)
    
    # Game routing status
    parent.game_routing_status = QLabel("No game active - using default AI")
    parent.game_routing_status.setStyleSheet("color: #bac2de; font-style: italic;")
    routing_layout.addWidget(parent.game_routing_status)
    
    layout.addWidget(routing_group)
    
    # === CO-PLAY SECTION ===
    coplay_group = QGroupBox("Co-Play (AI plays WITH you)")
    coplay_layout = QVBoxLayout(coplay_group)
    
    coplay_desc = QLabel(
        "AI acts as your teammate/companion using keyboard, mouse, or controller input."
    )
    coplay_desc.setWordWrap(True)
    coplay_layout.addWidget(coplay_desc)
    
    # Role selection
    role_row = QHBoxLayout()
    role_row.addWidget(QLabel("AI Role:"))
    parent.coplay_role_combo = NoScrollComboBox()
    parent.coplay_role_combo.addItems([
        "Companion (follows your lead)",
        "Teammate (equal partner)",
        "Support (heals/buffs)",
        "Defender (protects position)",
        "Explorer (scouts ahead)",
        "Coach (advice only)",
        "Opponent (friendly competition)",
    ])
    parent.coplay_role_combo.setToolTip("How the AI behaves when playing with you")
    role_row.addWidget(parent.coplay_role_combo)
    role_row.addStretch()
    coplay_layout.addLayout(role_row)
    
    # Input method
    input_row = QHBoxLayout()
    input_row.addWidget(QLabel("Input:"))
    parent.coplay_input_combo = NoScrollComboBox()
    parent.coplay_input_combo.addItems(["Keyboard/Mouse", "Controller", "Game API"])
    parent.coplay_input_combo.setToolTip("How the AI sends inputs to the game")
    input_row.addWidget(parent.coplay_input_combo)
    input_row.addStretch()
    coplay_layout.addLayout(input_row)
    
    # Co-play options
    options_row = QHBoxLayout()
    parent.coplay_announce = QCheckBox("Announce actions")
    parent.coplay_announce.setChecked(True)
    parent.coplay_announce.setToolTip("AI tells you what it's doing")
    options_row.addWidget(parent.coplay_announce)
    
    parent.coplay_ask_major = QCheckBox("Ask before big decisions")
    parent.coplay_ask_major.setChecked(True)
    parent.coplay_ask_major.setToolTip("AI asks before major actions like using items")
    options_row.addWidget(parent.coplay_ask_major)
    options_row.addStretch()
    coplay_layout.addLayout(options_row)
    
    # Co-play control buttons
    coplay_btn_row = QHBoxLayout()
    parent.btn_coplay_start = QPushButton("Start Co-Play")
    parent.btn_coplay_start.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: bold;")
    parent.btn_coplay_start.clicked.connect(lambda: _start_coplay(parent))
    coplay_btn_row.addWidget(parent.btn_coplay_start)
    
    parent.btn_coplay_pause = QPushButton("Pause")
    parent.btn_coplay_pause.setEnabled(False)
    parent.btn_coplay_pause.clicked.connect(lambda: _pause_coplay(parent))
    coplay_btn_row.addWidget(parent.btn_coplay_pause)
    
    parent.btn_coplay_stop = QPushButton("Stop")
    parent.btn_coplay_stop.setEnabled(False)
    parent.btn_coplay_stop.clicked.connect(lambda: _stop_coplay(parent))
    coplay_btn_row.addWidget(parent.btn_coplay_stop)
    coplay_btn_row.addStretch()
    coplay_layout.addLayout(coplay_btn_row)
    
    # Co-play status
    parent.coplay_status = QLabel("Co-Play: Inactive")
    parent.coplay_status.setStyleSheet("color: #bac2de; font-style: italic;")
    coplay_layout.addWidget(parent.coplay_status)
    
    layout.addWidget(coplay_group)
    
    # === GAMING PROFILES (Resource Management) ===
    profiles_group = QGroupBox("Gaming Profiles (Resource Limits)")
    profiles_layout = QVBoxLayout(profiles_group)
    
    profiles_desc = QLabel(
        "Configure AI resource limits per-game. Games are auto-detected by process name."
    )
    profiles_desc.setWordWrap(True)
    profiles_layout.addWidget(profiles_desc)
    
    # Gaming mode toggle
    parent.gaming_mode_check = QCheckBox("Enable Gaming Mode")
    parent.gaming_mode_check.setToolTip("Auto-detect games and apply resource limits")
    parent.gaming_mode_check.stateChanged.connect(
        lambda state: _toggle_gaming_mode(parent, state)
    )
    profiles_layout.addWidget(parent.gaming_mode_check)
    
    # Profile list with buttons
    profile_list_row = QHBoxLayout()
    
    parent.gaming_profile_list = QListWidget()
    parent.gaming_profile_list.setMaximumHeight(100)
    parent.gaming_profile_list.setToolTip("Double-click to edit a profile")
    parent.gaming_profile_list.itemDoubleClicked.connect(
        lambda item: _edit_gaming_profile(parent, item)
    )
    profile_list_row.addWidget(parent.gaming_profile_list)
    
    # Profile buttons
    profile_btn_col = QVBoxLayout()
    btn_add_profile = QPushButton("Add")
    btn_add_profile.setMaximumWidth(60)
    btn_add_profile.clicked.connect(lambda: _add_gaming_profile(parent))
    profile_btn_col.addWidget(btn_add_profile)
    
    btn_edit_profile = QPushButton("Edit")
    btn_edit_profile.setMaximumWidth(60)
    btn_edit_profile.clicked.connect(lambda: _edit_selected_profile(parent))
    profile_btn_col.addWidget(btn_edit_profile)
    
    btn_remove_profile = QPushButton("Remove")
    btn_remove_profile.setMaximumWidth(60)
    btn_remove_profile.clicked.connect(lambda: _remove_gaming_profile(parent))
    profile_btn_col.addWidget(btn_remove_profile)
    
    btn_steam_import = QPushButton("Steam")
    btn_steam_import.setMaximumWidth(60)
    btn_steam_import.setToolTip("Import games from Steam library")
    btn_steam_import.clicked.connect(lambda: _import_steam_games(parent))
    profile_btn_col.addWidget(btn_steam_import)
    
    profile_btn_col.addStretch()
    profile_list_row.addLayout(profile_btn_col)
    
    profiles_layout.addLayout(profile_list_row)
    
    # FPS monitoring options
    fps_row = QHBoxLayout()
    parent.fps_adaptive_check = QCheckBox("FPS Adaptive Scaling")
    parent.fps_adaptive_check.setChecked(True)
    parent.fps_adaptive_check.setToolTip("Automatically reduce AI resources when FPS drops")
    parent.fps_adaptive_check.stateChanged.connect(
        lambda state: _toggle_fps_adaptive(parent, state)
    )
    fps_row.addWidget(parent.fps_adaptive_check)
    
    fps_row.addWidget(QLabel("Target FPS:"))
    parent.target_fps_spin = QSpinBox()
    parent.target_fps_spin.setRange(30, 240)
    parent.target_fps_spin.setValue(60)
    parent.target_fps_spin.setMaximumWidth(70)
    parent.target_fps_spin.valueChanged.connect(
        lambda v: _set_target_fps(parent, v)
    )
    fps_row.addWidget(parent.target_fps_spin)
    fps_row.addStretch()
    profiles_layout.addLayout(fps_row)
    
    # FPS stats display
    parent.fps_stats_label = QLabel("FPS: -- | Scale: 100%")
    parent.fps_stats_label.setStyleSheet("color: #bac2de; font-family: monospace;")
    profiles_layout.addWidget(parent.fps_stats_label)
    
    # Current status
    parent.gaming_status_label = QLabel("Gaming Mode: Disabled")
    parent.gaming_status_label.setStyleSheet("color: #bac2de; font-style: italic;")
    profiles_layout.addWidget(parent.gaming_status_label)
    
    layout.addWidget(profiles_group)
    
    # === OUTPUT AT TOP ===
    # Log (main output area)
    parent.game_log = QTextEdit()
    parent.game_log.setReadOnly(True)
    parent.game_log.setPlaceholderText("Connection events will appear here...")
    parent.game_log.setStyleSheet("""
        QTextEdit {
            background-color: #1e1e2e;
            border: 1px solid #313244;
            border-radius: 4px;
            padding: 8px;
            font-family: 'Consolas', monospace;
        }
    """)
    layout.addWidget(parent.game_log, stretch=1)
    
    # Status
    parent.game_status_label = QLabel("Status: Not connected")
    parent.game_status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
    layout.addWidget(parent.game_status_label)
    
    # === CONTROLS ===
    # Config file selector row
    config_row = QHBoxLayout()
    config_row.addWidget(QLabel("Config:"))
    parent.game_config_combo = NoScrollComboBox()
    parent.game_config_combo.setToolTip("Select a game configuration file")
    parent.game_config_combo.currentIndexChanged.connect(
        lambda idx: _on_game_config_changed(parent, idx)
    )
    config_row.addWidget(parent.game_config_combo, stretch=1)
    btn_refresh = QPushButton("Refresh")
    btn_refresh.setMaximumWidth(60)
    btn_refresh.clicked.connect(lambda: _refresh_game_configs(parent, preserve_selection=True))
    config_row.addWidget(btn_refresh)
    btn_new = QPushButton("New")
    btn_new.setMaximumWidth(40)
    btn_new.clicked.connect(lambda: _create_new_game_config(parent))
    config_row.addWidget(btn_new)
    layout.addLayout(config_row)
    
    # Connection settings row
    conn_row = QHBoxLayout()
    conn_row.addWidget(QLabel("Protocol:"))
    parent.game_protocol_combo = NoScrollComboBox()
    parent.game_protocol_combo.setToolTip("Select the communication protocol")
    parent.game_protocol_combo.addItems(["websocket", "http", "tcp", "udp", "osc"])
    parent.game_protocol_combo.setMaximumWidth(100)
    conn_row.addWidget(parent.game_protocol_combo)
    conn_row.addWidget(QLabel("Host:"))
    parent.game_host_input = QLineEdit("localhost")
    parent.game_host_input.setMaximumWidth(120)
    conn_row.addWidget(parent.game_host_input)
    conn_row.addWidget(QLabel("Port:"))
    parent.game_port_spin = QSpinBox()
    parent.game_port_spin.setRange(1, 65535)
    parent.game_port_spin.setValue(8765)
    parent.game_port_spin.setMaximumWidth(70)
    conn_row.addWidget(parent.game_port_spin)
    conn_row.addStretch()
    layout.addLayout(conn_row)
    
    # Endpoint row
    endpoint_row = QHBoxLayout()
    endpoint_row.addWidget(QLabel("Endpoint:"))
    parent.game_endpoint_input = QLineEdit("/")
    endpoint_row.addWidget(parent.game_endpoint_input)
    layout.addLayout(endpoint_row)
    
    # Buttons row
    btn_row = QHBoxLayout()
    parent.btn_game_connect = QPushButton("Connect")
    parent.btn_game_connect.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: bold;")
    parent.btn_game_connect.clicked.connect(lambda: _connect_to_game(parent))
    btn_row.addWidget(parent.btn_game_connect)
    parent.btn_game_disconnect = QPushButton("Disconnect")
    parent.btn_game_disconnect.clicked.connect(lambda: _disconnect_from_game(parent))
    parent.btn_game_disconnect.setEnabled(False)
    btn_row.addWidget(parent.btn_game_disconnect)
    btn_save_config = QPushButton("Save Config")
    btn_save_config.clicked.connect(lambda: _save_current_game_config(parent))
    btn_row.addWidget(btn_save_config)
    btn_row.addStretch()
    layout.addLayout(btn_row)
    
    widget.setLayout(layout)
    
    # Initialize
    parent.game_connection = None
    GAME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _refresh_game_configs(parent)
    
    # Load custom games from previous sessions
    _load_custom_games(parent)
    
    # Initialize gaming profiles list
    _refresh_gaming_profiles(parent)
    
    return widget


# === GAMING PROFILE MANAGEMENT FUNCTIONS ===

def _toggle_gaming_mode(parent, state):
    """Toggle gaming mode on/off."""
    try:
        from enigma_engine.core.gaming_mode import get_gaming_mode
        gm = get_gaming_mode()
        
        if state:
            gm.enable()
            parent.gaming_status_label.setText("Gaming Mode: Enabled (monitoring)")
            parent.gaming_status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
            parent.game_log.append("[OK] Gaming mode enabled - monitoring for games")
            
            # Register status callback
            def on_game_start(game, profile):
                parent.gaming_status_label.setText(f"Gaming Mode: {profile.name}")
                parent.game_log.append(f"[G] Detected: {game} -> {profile.name}")
            
            def on_game_end(game):
                parent.gaming_status_label.setText("Gaming Mode: Enabled (monitoring)")
                parent.game_log.append(f"[G] Game ended: {game}")
            
            def on_fps_update(stats):
                # Use signal for thread-safe UI update
                try:
                    from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                    QMetaObject.invokeMethod(
                        parent.fps_stats_label,
                        "setText",
                        Qt.QueuedConnection,
                        Q_ARG(str, f"FPS: {stats.current_fps:.0f} | Scale: {int(gm.get_fps_scale() * 100)}%")
                    )
                except Exception:
                    pass  # Intentionally silent
            
            gm.on_game_start(on_game_start)
            gm.on_game_end(on_game_end)
            gm.on_fps_update(on_fps_update)
            
            # Apply current settings
            gm.set_fps_adaptive(parent.fps_adaptive_check.isChecked())
            gm.set_target_fps(float(parent.target_fps_spin.value()))
        else:
            gm.disable()
            parent.gaming_status_label.setText("Gaming Mode: Disabled")
            parent.gaming_status_label.setStyleSheet("color: #bac2de; font-style: italic;")
            parent.fps_stats_label.setText("FPS: -- | Scale: 100%")
            parent.fps_stats_label.setStyleSheet("color: #bac2de; font-family: monospace;")
            parent.game_log.append("[x] Gaming mode disabled")
            
    except Exception as e:
        parent.game_log.append(f"[X] Gaming mode error: {e}")


def _refresh_gaming_profiles(parent):
    """Refresh the gaming profiles list."""
    parent.gaming_profile_list.clear()
    
    try:
        from enigma_engine.core.gaming_mode import get_gaming_mode, DEFAULT_GAMING_PROFILES
        gm = get_gaming_mode()
        
        for key, profile in gm.profiles.items():
            is_default = key in DEFAULT_GAMING_PROFILES
            prefix = "[D] " if is_default else "[C] "
            processes = ", ".join(profile.process_names[:2])
            if len(profile.process_names) > 2:
                processes += "..."
            
            item = QListWidgetItem(f"{prefix}{profile.name} ({processes})")
            item.setData(256, key)  # Store key in user role
            item.setData(257, is_default)  # Store if default
            
            if is_default:
                item.setToolTip(f"Default profile: {profile.notes or 'No description'}")
            else:
                item.setToolTip(f"Custom profile: {profile.notes or 'No description'}")
            
            parent.gaming_profile_list.addItem(item)
            
    except Exception as e:
        parent.game_log.append(f"[X] Failed to load gaming profiles: {e}")


def _toggle_fps_adaptive(parent, state):
    """Toggle FPS adaptive scaling."""
    try:
        from enigma_engine.core.gaming_mode import get_gaming_mode
        gm = get_gaming_mode()
        gm.set_fps_adaptive(bool(state))
        if state:
            parent.game_log.append("[OK] FPS adaptive scaling enabled")
        else:
            parent.game_log.append("[x] FPS adaptive scaling disabled")
            parent.fps_stats_label.setText("FPS: -- | Scale: 100%")
    except Exception as e:
        parent.game_log.append(f"[X] FPS adaptive error: {e}")


def _set_target_fps(parent, fps):
    """Set target FPS for monitoring."""
    try:
        from enigma_engine.core.gaming_mode import get_gaming_mode
        gm = get_gaming_mode()
        gm.set_target_fps(float(fps))
    except Exception:
        pass  # Silent fail if gaming mode not available


def _update_fps_display(parent, stats):
    """Update FPS stats display from callback."""
    try:
        scale = 100
        try:
            from enigma_engine.core.gaming_mode import get_gaming_mode
            scale = int(get_gaming_mode().get_fps_scale() * 100)
        except Exception:
            pass  # Intentionally silent
        
        if stats.current_fps > 0:
            fps_text = f"FPS: {stats.current_fps:.0f} (avg: {stats.average_fps:.0f}) | Scale: {scale}%"
            
            # Color based on scale
            if scale >= 80:
                color = "#a6e3a1"  # Green
            elif scale >= 50:
                color = "#f59e0b"  # Orange
            else:
                color = "#f38ba8"  # Red
            
            parent.fps_stats_label.setText(fps_text)
            parent.fps_stats_label.setStyleSheet(f"color: {color}; font-family: monospace;")
        else:
            parent.fps_stats_label.setText("FPS: -- | Scale: 100%")
            parent.fps_stats_label.setStyleSheet("color: #bac2de; font-family: monospace;")
    except Exception:
        pass  # Intentionally silent


def _add_gaming_profile(parent):
    """Add a new gaming profile."""
    dialog = GamingProfileDialog(parent)
    if dialog.exec_() == QDialog.Accepted:
        profile = dialog.get_profile()
        
        try:
            from enigma_engine.core.gaming_mode import get_gaming_mode
            gm = get_gaming_mode()
            gm.add_game_profile(profile)
            
            _refresh_gaming_profiles(parent)
            parent.game_log.append(f"[OK] Added gaming profile: {profile.name}")
            
        except Exception as e:
            parent.game_log.append(f"[X] Failed to add profile: {e}")


def _import_steam_games(parent):
    """Import games from Steam library."""
    try:
        from enigma_engine.core.steam_integration import get_steam_integration
        steam = get_steam_integration()
        
        if not steam.is_available:
            QMessageBox.warning(
                parent,
                "Steam Not Found",
                "Could not find Steam installation. Make sure Steam is installed."
            )
            return
        
        # Get games
        games = steam.get_installed_games(refresh=True)
        
        if not games:
            QMessageBox.information(
                parent,
                "No Games Found",
                "No games found in your Steam library."
            )
            return
        
        # Register with gaming mode
        count = steam.register_with_gaming_mode()
        
        _refresh_gaming_profiles(parent)
        
        if count > 0:
            parent.game_log.append(f"[OK] Imported {count} games from Steam ({len(games)} total in library)")
            QMessageBox.information(
                parent,
                "Steam Import Complete",
                f"Imported {count} new games from Steam.\n\n"
                f"Total games in library: {len(games)}\n"
                f"Games already registered were skipped."
            )
        else:
            parent.game_log.append(f"[!] No new games to import (all {len(games)} already registered)")
            QMessageBox.information(
                parent,
                "Already Imported",
                f"All {len(games)} games from Steam are already registered."
            )
            
    except ImportError:
        parent.game_log.append("[X] Steam integration not available")
    except Exception as e:
        parent.game_log.append(f"[X] Steam import failed: {e}")
        QMessageBox.warning(
            parent,
            "Import Failed",
            f"Failed to import Steam games: {e}"
        )


def _edit_gaming_profile(parent, item):
    """Edit a gaming profile from list item."""
    key = item.data(256)
    is_default = item.data(257)
    
    if is_default:
        QMessageBox.information(
            parent,
            "Cannot Edit",
            "Default profiles cannot be edited. Create a custom profile instead."
        )
        return
    
    try:
        from enigma_engine.core.gaming_mode import get_gaming_mode
        gm = get_gaming_mode()
        profile = gm.profiles.get(key)
        
        if profile:
            dialog = GamingProfileDialog(parent, profile)
            if dialog.exec_() == QDialog.Accepted:
                new_profile = dialog.get_profile()
                
                # Remove old profile if name changed
                if key != new_profile.name.lower().replace(" ", "_"):
                    gm.remove_game_profile(profile.name, auto_save=False)
                
                gm.add_game_profile(new_profile)
                _refresh_gaming_profiles(parent)
                parent.game_log.append(f"[OK] Updated gaming profile: {new_profile.name}")
                
    except Exception as e:
        parent.game_log.append(f"[X] Failed to edit profile: {e}")


def _edit_selected_profile(parent):
    """Edit the currently selected profile."""
    item = parent.gaming_profile_list.currentItem()
    if item:
        _edit_gaming_profile(parent, item)
    else:
        QMessageBox.information(parent, "No Selection", "Please select a profile to edit.")


def _remove_gaming_profile(parent):
    """Remove the selected gaming profile."""
    item = parent.gaming_profile_list.currentItem()
    if not item:
        QMessageBox.information(parent, "No Selection", "Please select a profile to remove.")
        return
    
    key = item.data(256)
    is_default = item.data(257)
    
    if is_default:
        QMessageBox.information(
            parent,
            "Cannot Remove",
            "Default profiles cannot be removed."
        )
        return
    
    # Confirm
    reply = QMessageBox.question(
        parent,
        "Confirm Removal",
        f"Remove gaming profile '{key}'?",
        QMessageBox.Yes | QMessageBox.No
    )
    
    if reply == QMessageBox.Yes:
        try:
            from enigma_engine.core.gaming_mode import get_gaming_mode
            gm = get_gaming_mode()
            
            if gm.remove_game_profile(key):
                _refresh_gaming_profiles(parent)
                parent.game_log.append(f"[OK] Removed gaming profile: {key}")
            else:
                parent.game_log.append(f"[!] Profile not found: {key}")
                
        except Exception as e:
            parent.game_log.append(f"[X] Failed to remove profile: {e}")


def load_game_configs() -> list:
    """Get list of available game config files."""
    configs = []
    if GAME_CONFIG_DIR.exists():
        for f in GAME_CONFIG_DIR.glob("*.json"):
            configs.append(f)
    return configs


def _refresh_game_configs(parent, preserve_selection=False):
    """Refresh the list of available game configs."""
    # Remember current selection if preserving
    previous_data = parent.game_config_combo.currentData() if preserve_selection else None
    
    parent.game_config_combo.blockSignals(True)
    parent.game_config_combo.clear()
    parent.game_config_combo.addItem("-- Select Config File --", None)
    
    if GAME_CONFIG_DIR.exists():
        for config_file in sorted(GAME_CONFIG_DIR.glob("*.json")):
            parent.game_config_combo.addItem(config_file.stem, str(config_file))
    
    # Restore selection if found
    if previous_data:
        for i in range(parent.game_config_combo.count()):
            if parent.game_config_combo.itemData(i) == previous_data:
                parent.game_config_combo.setCurrentIndex(i)
                break
    
    parent.game_config_combo.blockSignals(False)


def _on_game_config_changed(parent, index):
    """Handle game config selection."""
    config_path = parent.game_config_combo.currentData()
    if not config_path:
        return
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        # Apply config to UI
        if "protocol" in config:
            idx = parent.game_protocol_combo.findText(config["protocol"])
            if idx >= 0:
                parent.game_protocol_combo.setCurrentIndex(idx)
        if "host" in config:
            parent.game_host_input.setText(config["host"])
        if "port" in config:
            parent.game_port_spin.setValue(config["port"])
        if "endpoint" in config:
            parent.game_endpoint_input.setText(config["endpoint"])
        
        parent.game_log.append(f"Loaded config: {Path(config_path).stem}")
        
    except Exception as e:
        parent.game_log.append(f"[X] Failed to load config: {e}")


def _browse_game_config(parent):
    """Browse for a game config file."""
    path, _ = QFileDialog.getOpenFileName(
        parent, "Select Game Config",
        str(GAME_CONFIG_DIR),
        "JSON Files (*.json);;All Files (*)"
    )
    if path:
        for i in range(parent.game_config_combo.count()):
            if parent.game_config_combo.itemData(i) == path:
                parent.game_config_combo.setCurrentIndex(i)
                return
        
        parent.game_config_combo.addItem(Path(path).stem, path)
        parent.game_config_combo.setCurrentIndex(parent.game_config_combo.count() - 1)


def _create_new_game_config(parent):
    """Create a new game config file."""
    path, _ = QFileDialog.getSaveFileName(
        parent, "Create Game Config",
        str(GAME_CONFIG_DIR / "new_game.json"),
        "JSON Files (*.json)"
    )
    if path:
        config = {
            "name": Path(path).stem,
            "protocol": "websocket",
            "host": "localhost",
            "port": 8765,
            "endpoint": "/",
            "description": "New game connection config"
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        _refresh_game_configs(parent)
        # Select the new config
        for i in range(parent.game_config_combo.count()):
            if parent.game_config_combo.itemData(i) == path:
                parent.game_config_combo.setCurrentIndex(i)
                break


def _save_current_game_config(parent):
    """Save current settings to the selected config file."""
    config_path = parent.game_config_combo.currentData()
    if not config_path:
        _create_new_game_config(parent)
        return
    
    config = {
        "name": Path(config_path).stem,
        "protocol": parent.game_protocol_combo.currentText(),
        "host": parent.game_host_input.text(),
        "port": parent.game_port_spin.value(),
        "endpoint": parent.game_endpoint_input.text()
    }
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        parent.game_log.append(f"[OK] Config saved: {Path(config_path).stem}")
    except Exception as e:
        parent.game_log.append(f"[X] Failed to save: {e}")


def _connect_to_game(parent):
    """Connect to game using current settings."""
    protocol = parent.game_protocol_combo.currentText()
    host = parent.game_host_input.text()
    port = parent.game_port_spin.value()
    endpoint = parent.game_endpoint_input.text()
    
    parent.game_log.append(f"Connecting via {protocol} to {host}:{port}{endpoint}...")
    
    try:
        if protocol == "websocket":
            parent.game_log.append("[>] WebSocket connection...")
            # Actual WebSocket connection
            try:
                import websocket
            except ImportError:
                parent.game_log.append("[!] websocket-client not installed. Install with: pip install websocket-client")
                parent.game_connection = None
            else:
                try:
                    parent.game_connection = websocket.create_connection(f"ws://{host}:{port}{endpoint}")
                    parent.game_log.append("[OK] WebSocket connection established")
                except (websocket.WebSocketException, OSError, ConnectionError) as e:
                    parent.game_log.append(f"[X] WebSocket connection failed: {e}")
                    parent.game_connection = None
            
        elif protocol == "http":
            parent.game_log.append("[>] HTTP API ready")
            # HTTP doesn't maintain persistent connection, store info for requests
            parent.game_connection = {"type": "http", "host": host, "port": port, "endpoint": endpoint}
            
        elif protocol == "osc":
            parent.game_log.append("[>] OSC client ready")
            # Actual OSC client
            try:
                from pythonosc import udp_client
                parent.game_connection = udp_client.SimpleUDPClient(host, port)
                parent.game_log.append("[OK] OSC client initialized")
            except ImportError:
                parent.game_log.append("[!] python-osc not installed. Install with: pip install python-osc")
                parent.game_connection = None
        
        # Mark as connected if we have a connection
        if parent.game_connection:
            parent.game_status_label.setText(f"Status: Connected ({protocol}://{host}:{port})")
            parent.game_status_label.setStyleSheet("color: #a6e3a1;")
            parent.btn_game_connect.setEnabled(False)
            parent.btn_game_disconnect.setEnabled(True)
            parent.game_log.append("[OK] Connection established")
        else:
            parent.game_status_label.setText("Status: Connection failed (missing library)")
            parent.game_status_label.setStyleSheet("color: #f38ba8;")
        
    except Exception as e:
        parent.game_log.append(f"[X] Connection failed: {e}")
        parent.game_status_label.setText("Status: Connection failed")


def _disconnect_from_game(parent):
    """Disconnect from game."""
    if parent.game_connection:
        try:
            parent.game_connection.close()
        except Exception:
            pass  # Intentionally silent
    parent.game_connection = None
    parent.game_status_label.setText("Status: Not connected")
    parent.game_status_label.setStyleSheet("color: #f38ba8;")
    parent.btn_game_connect.setEnabled(True)
    parent.btn_game_disconnect.setEnabled(False)
    parent.game_log.append("[x] Disconnected")


# === GAME AI ROUTING FUNCTIONS ===

def _toggle_game_detection(parent, state):
    """Toggle automatic game detection."""
    from PyQt5.QtWidgets import QMessageBox
    enabled = state == 2
    
    try:
        from enigma_engine.tools.game_router import get_game_router
        
        router = get_game_router()
        
        if enabled:
            router.start_detection(interval=5.0)
            router.on_game_detected(lambda game: _on_game_detected(parent, game))
            parent.game_routing_combo.setEnabled(False)
            parent.game_routing_status.setText("Watching for games...")
            parent.game_routing_status.setStyleSheet("color: #3b82f6;")
        else:
            router.stop_detection()
            parent.game_routing_combo.setEnabled(True)
            parent.game_routing_status.setText("Auto-detection disabled")
            parent.game_routing_status.setStyleSheet("color: #bac2de;")
    except ImportError:
        parent.auto_game_check.setChecked(False)
        QMessageBox.information(
            parent, "psutil Required",
            "Game detection requires psutil. Install with:\npip install psutil"
        )
    except Exception as e:
        parent.auto_game_check.setChecked(False)
        parent.game_routing_status.setText(f"Detection error: {e}")
        parent.game_routing_status.setStyleSheet("color: #ef4444;")


def _on_game_detected(parent, game_id: str):
    """Called when a game is auto-detected."""
    try:
        from enigma_engine.tools.game_router import get_game_router
        router = get_game_router()
        config = router.get_game(game_id)
        
        if config:
            parent.game_routing_status.setText(f"Detected: {config.name}")
            parent.game_routing_status.setStyleSheet("color: #22c55e; font-weight: bold;")
            
            # Update combo without triggering change
            parent.game_routing_combo.blockSignals(True)
            for i in range(parent.game_routing_combo.count()):
                if parent.game_routing_combo.itemData(i) == game_id:
                    parent.game_routing_combo.setCurrentIndex(i)
                    break
            parent.game_routing_combo.blockSignals(False)
    except Exception:
        pass  # Intentionally silent


def _change_active_game(parent):
    """Manually change active game."""
    game_id = parent.game_routing_combo.currentData()
    
    if game_id == "custom":
        dialog = CustomGameDialog(parent)
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.get_config()
            try:
                from enigma_engine.tools.game_router import (
                    GameConfig,
                    GameType,
                    get_game_router,
                )
                router = get_game_router()
                
                # Create GameConfig from dialog
                game_config = GameConfig(
                    name=config["name"],
                    type=GameType(config["type"]),
                    model=config["model"],
                    system_prompt=config["system_prompt"],
                    process_names=config["process_names"],
                    window_titles=config["window_titles"],
                    quick_responses=config["quick_responses"],
                    voice_enabled=config["voice_enabled"],
                    multiplayer_aware=config["multiplayer_aware"],
                    wiki_url=config["wiki_url"],
                )
                
                # Register the custom game
                router.register_game(config["id"], game_config)
                
                # Add to combo box
                parent.game_routing_combo.insertItem(
                    parent.game_routing_combo.count() - 1,  # Before "Custom..."
                    config["name"],
                    config["id"]
                )
                
                # Select it
                parent.game_routing_combo.setCurrentIndex(
                    parent.game_routing_combo.count() - 2
                )
                
                # Set as active
                router.set_active_game(config["id"])
                parent.game_routing_status.setText(f"Active: {config['name']}")
                parent.game_routing_status.setStyleSheet("color: #a6e3a1; font-weight: bold;")
                
                # Save to config file
                _save_custom_game(config)
                
                if hasattr(parent, 'game_log'):
                    parent.game_log.append(f"[OK] Custom game '{config['name']}' registered")
                    
            except Exception as e:
                QMessageBox.warning(
                    parent, "Error",
                    f"Failed to register custom game: {e}"
                )
        
        # Reset combo if dialog was cancelled
        if parent.game_routing_combo.currentData() == "custom":
            parent.game_routing_combo.setCurrentIndex(0)
        return
    
    try:
        from enigma_engine.tools.game_router import get_game_router
        router = get_game_router()
        
        if game_id == "none":
            router.set_active_game(None)
            parent.game_routing_status.setText("No game active - using default AI")
            parent.game_routing_status.setStyleSheet("color: #bac2de; font-style: italic;")
        else:
            router.set_active_game(game_id)
            config = router.get_game(game_id)
            if config:
                parent.game_routing_status.setText(f"Active: {config.name}")
                parent.game_routing_status.setStyleSheet("color: #a6e3a1; font-weight: bold;")
    except ImportError:
        parent.game_routing_status.setText("Game router not available")
        parent.game_routing_status.setStyleSheet("color: #f59e0b;")
    except Exception as e:
        parent.game_routing_status.setText(f"Error: {str(e)[:30]}")
        parent.game_routing_status.setStyleSheet("color: #ef4444;")


def _save_custom_game(config: dict):
    """Save custom game configuration to file."""
    custom_games_file = Path(CONFIG["data_dir"]) / "game" / "custom_games.json"
    custom_games_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing
    existing = []
    if custom_games_file.exists():
        try:
            with open(custom_games_file) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    
    # Add/update
    found = False
    for i, game in enumerate(existing):
        if game.get("id") == config["id"]:
            existing[i] = config
            found = True
            break
    
    if not found:
        existing.append(config)
    
    # Save
    with open(custom_games_file, 'w') as f:
        json.dump(existing, f, indent=2)


def _load_custom_games(parent):
    """Load custom games from config file and add to combo."""
    custom_games_file = Path(CONFIG["data_dir"]) / "game" / "custom_games.json"
    if not custom_games_file.exists():
        return
    
    try:
        with open(custom_games_file) as f:
            games = json.load(f)
        
        for game in games:
            # Check if not already in combo
            found = False
            for i in range(parent.game_routing_combo.count()):
                if parent.game_routing_combo.itemData(i) == game.get("id"):
                    found = True
                    break
            
            if not found:
                # Insert before "Custom..."
                parent.game_routing_combo.insertItem(
                    parent.game_routing_combo.count() - 1,
                    game.get("name", game.get("id")),
                    game.get("id")
                )
                
                # Also register with router
                try:
                    from enigma_engine.tools.game_router import (
                        GameConfig,
                        GameType,
                        get_game_router,
                    )
                    router = get_game_router()
                    game_config = GameConfig(
                        name=game.get("name", game.get("id")),
                        type=GameType(game.get("type", "other")),
                        model=game.get("model", "small"),
                        system_prompt=game.get("system_prompt", ""),
                        process_names=game.get("process_names", []),
                        window_titles=game.get("window_titles", []),
                        quick_responses=game.get("quick_responses", False),
                        voice_enabled=game.get("voice_enabled", True),
                        multiplayer_aware=game.get("multiplayer_aware", False),
                        wiki_url=game.get("wiki_url", ""),
                    )
                    router.register_game(game.get("id"), game_config)
                except Exception:
                    pass  # Router may not be available
                    
    except Exception:
        pass  # File may be corrupted


# === CO-PLAY CONTROL FUNCTIONS ===

def _start_coplay(parent):
    """Start co-play mode."""
    try:
        from enigma_engine.tools.game_coplay import get_coplayer
        
        coplayer = get_coplayer()
        
        # Set role from combo
        role_text = parent.coplay_role_combo.currentText().lower()
        if "companion" in role_text:
            coplayer.set_role("companion")
        elif "teammate" in role_text:
            coplayer.set_role("teammate")
        elif "support" in role_text:
            coplayer.set_role("support")
        elif "defender" in role_text:
            coplayer.set_role("defender")
        elif "explorer" in role_text:
            coplayer.set_role("explorer")
        elif "coach" in role_text:
            coplayer.set_role("coach")
        elif "opponent" in role_text:
            coplayer.set_role("opponent")
        
        # Set options
        coplayer.config.announce_actions = parent.coplay_announce.isChecked()
        coplayer.config.ask_before_major = parent.coplay_ask_major.isChecked()
        
        # Connect to selected game
        game_id = parent.game_routing_combo.currentData()
        if game_id and game_id != "none":
            coplayer.connect_game(game_id)
        
        # Register callbacks
        def on_action(action):
            parent.game_log.append(f"[AI] {action.reason}")
        
        def on_speak(message):
            parent.game_log.append(f"[AI says] {message}")
        
        coplayer.on_action(on_action)
        coplayer.on_speak(on_speak)
        
        # Start
        coplayer.start()
        
        # Update UI
        parent.btn_coplay_start.setEnabled(False)
        parent.btn_coplay_pause.setEnabled(True)
        parent.btn_coplay_stop.setEnabled(True)
        parent.coplay_status.setText("Co-Play: Active")
        parent.coplay_status.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        parent.game_log.append("[OK] Co-Play started - AI is now your " + coplayer.config.role.value)
        
        # Store reference
        parent._coplayer = coplayer
        
    except ImportError as e:
        parent.game_log.append(f"[!] Co-play requires pynput: pip install pynput")
    except Exception as e:
        parent.game_log.append(f"[X] Failed to start co-play: {e}")


def _pause_coplay(parent):
    """Pause co-play mode."""
    if hasattr(parent, '_coplayer') and parent._coplayer:
        if parent._coplayer._paused:
            parent._coplayer.resume()
            parent.btn_coplay_pause.setText("Pause")
            parent.coplay_status.setText("Co-Play: Active")
            parent.coplay_status.setStyleSheet("color: #a6e3a1; font-weight: bold;")
            parent.game_log.append("[>] Co-Play resumed")
        else:
            parent._coplayer.pause()
            parent.btn_coplay_pause.setText("Resume")
            parent.coplay_status.setText("Co-Play: Paused")
            parent.coplay_status.setStyleSheet("color: #f59e0b; font-weight: bold;")
            parent.game_log.append("[||] Co-Play paused")


def _stop_coplay(parent):
    """Stop co-play mode."""
    if hasattr(parent, '_coplayer') and parent._coplayer:
        parent._coplayer.stop()
        parent._coplayer = None
    
    parent.btn_coplay_start.setEnabled(True)
    parent.btn_coplay_pause.setEnabled(False)
    parent.btn_coplay_pause.setText("Pause")
    parent.btn_coplay_stop.setEnabled(False)
    parent.coplay_status.setText("Co-Play: Inactive")
    parent.coplay_status.setStyleSheet("color: #bac2de; font-style: italic;")
    parent.game_log.append("[x] Co-Play stopped")
