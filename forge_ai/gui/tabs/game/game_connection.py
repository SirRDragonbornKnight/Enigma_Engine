"""
Game Connection Module

Allows the AI to connect to and control games/applications.
The same AI being trained learns to interact with these games.

Supports loading connection configs from files instead of presets.
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QLineEdit, QSpinBox, QTextEdit, QFileDialog,
    QDialog, QDialogButtonBox, QCheckBox, QFormLayout, QMessageBox
)
import json

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


# Game configs directory
GAME_CONFIG_DIR = Path(CONFIG["data_dir"]) / "game"


def create_game_subtab(parent):
    """
    Create the game connection sub-tab.
    
    The AI you are training will learn to control these games.
    Connection settings are loaded from config files.
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
    
    return widget


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
        with open(config_path, 'r') as f:
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
            pass
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
        from forge_ai.tools.game_router import get_game_router
        
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
        from forge_ai.tools.game_router import get_game_router
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
        pass


def _change_active_game(parent):
    """Manually change active game."""
    game_id = parent.game_routing_combo.currentData()
    
    if game_id == "custom":
        dialog = CustomGameDialog(parent)
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.get_config()
            try:
                from forge_ai.tools.game_router import get_game_router, GameConfig, GameType
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
        from forge_ai.tools.game_router import get_game_router
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
            with open(custom_games_file, 'r') as f:
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
        with open(custom_games_file, 'r') as f:
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
                    from forge_ai.tools.game_router import get_game_router, GameConfig, GameType
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
