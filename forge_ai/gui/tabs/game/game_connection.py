"""
Game Connection Module

Allows the AI to connect to and control games/applications.
The same AI being trained learns to interact with these games.

Supports loading connection configs from files instead of presets.
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, 
    QGroupBox, QLineEdit, QSpinBox, QTextEdit, QFileDialog
)
import json

from ....config import CONFIG


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
    parent.game_routing_combo = QComboBox()
    parent.game_routing_combo.addItem("(None)", "none")
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
    parent.game_routing_status.setStyleSheet("color: #888; font-style: italic;")
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
    parent.game_config_combo = QComboBox()
    parent.game_config_combo.currentIndexChanged.connect(
        lambda idx: _on_game_config_changed(parent, idx)
    )
    config_row.addWidget(parent.game_config_combo, stretch=1)
    btn_refresh = QPushButton("Refresh")
    btn_refresh.setMaximumWidth(60)
    btn_refresh.clicked.connect(lambda: _refresh_game_configs(parent))
    config_row.addWidget(btn_refresh)
    btn_new = QPushButton("New")
    btn_new.setMaximumWidth(40)
    btn_new.clicked.connect(lambda: _create_new_game_config(parent))
    config_row.addWidget(btn_new)
    layout.addLayout(config_row)
    
    # Connection settings row
    conn_row = QHBoxLayout()
    conn_row.addWidget(QLabel("Protocol:"))
    parent.game_protocol_combo = QComboBox()
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
    
    return widget


def load_game_configs() -> list:
    """Get list of available game config files."""
    configs = []
    if GAME_CONFIG_DIR.exists():
        for f in GAME_CONFIG_DIR.glob("*.json"):
            configs.append(f)
    return configs


def _refresh_game_configs(parent):
    """Refresh the list of available game configs."""
    parent.game_config_combo.clear()
    parent.game_config_combo.addItem("-- Select Config File --", None)
    
    if GAME_CONFIG_DIR.exists():
        for config_file in sorted(GAME_CONFIG_DIR.glob("*.json")):
            parent.game_config_combo.addItem(config_file.stem, str(config_file))


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
            parent.game_routing_status.setStyleSheet("color: #888;")
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
            parent.game_routing_status.setText(f"🎮 Detected: {config.name}")
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
    from PyQt5.QtWidgets import QMessageBox
    game_id = parent.game_routing_combo.currentData()
    
    if game_id == "custom":
        QMessageBox.information(
            parent, "Custom Game",
            "Custom game configuration coming soon!\n"
            "For now, add games in forge_ai/tools/game_router.py"
        )
        parent.game_routing_combo.setCurrentIndex(0)
        return
    
    try:
        from forge_ai.tools.game_router import get_game_router
        router = get_game_router()
        
        if game_id == "none":
            router.set_active_game(None)
            parent.game_routing_status.setText("No game active - using default AI")
            parent.game_routing_status.setStyleSheet("color: #888; font-style: italic;")
        else:
            router.set_active_game(game_id)
            config = router.get_game(game_id)
            if config:
                parent.game_routing_status.setText(f"🎮 Active: {config.name}")
                parent.game_routing_status.setStyleSheet("color: #a6e3a1; font-weight: bold;")
    except ImportError:
        parent.game_routing_status.setText("Game router not available")
        parent.game_routing_status.setStyleSheet("color: #f59e0b;")
    except Exception as e:
        parent.game_routing_status.setText(f"Error: {str(e)[:30]}")
        parent.game_routing_status.setStyleSheet("color: #ef4444;")
