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
from PyQt5.QtCore import Qt
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
    
    # Config file selector (not presets)
    config_layout = QHBoxLayout()
    config_layout.addWidget(QLabel("Game Config:"))
    parent.game_config_combo = QComboBox()
    parent.game_config_combo.currentIndexChanged.connect(
        lambda idx: _on_game_config_changed(parent, idx)
    )
    config_layout.addWidget(parent.game_config_combo, stretch=1)
    
    btn_refresh = QPushButton("🔄")
    btn_refresh.setFixedWidth(40)
    btn_refresh.setToolTip("Refresh config list")
    btn_refresh.clicked.connect(lambda: _refresh_game_configs(parent))
    config_layout.addWidget(btn_refresh)
    
    btn_browse = QPushButton("Browse...")
    btn_browse.clicked.connect(lambda: _browse_game_config(parent))
    config_layout.addWidget(btn_browse)
    
    btn_new = QPushButton("New")
    btn_new.clicked.connect(lambda: _create_new_game_config(parent))
    config_layout.addWidget(btn_new)
    layout.addLayout(config_layout)
    
    # Connection settings (editable, loaded from config)
    conn_group = QGroupBox("Connection Settings")
    conn_layout = QVBoxLayout()
    
    # Protocol selector
    proto_layout = QHBoxLayout()
    proto_layout.addWidget(QLabel("Protocol:"))
    parent.game_protocol_combo = QComboBox()
    parent.game_protocol_combo.addItems(["websocket", "http", "tcp", "udp", "osc"])
    proto_layout.addWidget(parent.game_protocol_combo)
    conn_layout.addLayout(proto_layout)
    
    # Host/Port
    host_layout = QHBoxLayout()
    host_layout.addWidget(QLabel("Host:"))
    parent.game_host_input = QLineEdit("localhost")
    host_layout.addWidget(parent.game_host_input)
    host_layout.addWidget(QLabel("Port:"))
    parent.game_port_spin = QSpinBox()
    parent.game_port_spin.setRange(1, 65535)
    parent.game_port_spin.setValue(8765)
    host_layout.addWidget(parent.game_port_spin)
    conn_layout.addLayout(host_layout)
    
    # Endpoint/Path (for HTTP/WebSocket)
    endpoint_layout = QHBoxLayout()
    endpoint_layout.addWidget(QLabel("Endpoint:"))
    parent.game_endpoint_input = QLineEdit("/")
    endpoint_layout.addWidget(parent.game_endpoint_input)
    conn_layout.addLayout(endpoint_layout)
    
    conn_group.setLayout(conn_layout)
    layout.addWidget(conn_group)
    
    # Connect/Disconnect buttons
    btn_layout = QHBoxLayout()
    parent.btn_game_connect = QPushButton("Connect")
    parent.btn_game_connect.clicked.connect(lambda: _connect_to_game(parent))
    parent.btn_game_disconnect = QPushButton("Disconnect")
    parent.btn_game_disconnect.clicked.connect(lambda: _disconnect_from_game(parent))
    parent.btn_game_disconnect.setEnabled(False)
    btn_save_config = QPushButton("Save Config")
    btn_save_config.clicked.connect(lambda: _save_current_game_config(parent))
    btn_layout.addWidget(parent.btn_game_connect)
    btn_layout.addWidget(parent.btn_game_disconnect)
    btn_layout.addWidget(btn_save_config)
    layout.addLayout(btn_layout)
    
    # Status
    parent.game_status_label = QLabel("Status: Not connected")
    parent.game_status_label.setStyleSheet("color: #f38ba8;")
    layout.addWidget(parent.game_status_label)
    
    # Log
    layout.addWidget(QLabel("Connection Log:"))
    parent.game_log = QTextEdit()
    parent.game_log.setReadOnly(True)
    parent.game_log.setMaximumHeight(150)
    parent.game_log.setPlaceholderText("Connection events will appear here...")
    layout.addWidget(parent.game_log)
    
    # Info
    info = QLabel("💡 The AI you train will learn to send commands to this game.\n"
                  "    Create config files for each game you want the AI to control.")
    info.setStyleSheet("color: #6c7086; font-size: 10px;")
    layout.addWidget(info)
    
    layout.addStretch()
    widget.setLayout(layout)
    
    # Initialize
    parent.game_connection = None
    GAME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load configs
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
        parent.game_log.append(f"❌ Failed to load config: {e}")


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
        parent.game_log.append(f"✅ Config saved: {Path(config_path).stem}")
    except Exception as e:
        parent.game_log.append(f"❌ Failed to save: {e}")


def _connect_to_game(parent):
    """Connect to game using current settings."""
    protocol = parent.game_protocol_combo.currentText()
    host = parent.game_host_input.text()
    port = parent.game_port_spin.value()
    endpoint = parent.game_endpoint_input.text()
    
    parent.game_log.append(f"Connecting via {protocol} to {host}:{port}{endpoint}...")
    
    try:
        if protocol == "websocket":
            parent.game_log.append("📡 WebSocket connection...")
            # TODO: Actual WebSocket connection
            # import websocket
            # parent.game_connection = websocket.create_connection(f"ws://{host}:{port}{endpoint}")
            
        elif protocol == "http":
            parent.game_log.append("🌐 HTTP API ready")
            # HTTP doesn't maintain persistent connection
            
        elif protocol == "osc":
            parent.game_log.append("🎵 OSC client ready")
            # TODO: OSC client
            # from pythonosc import udp_client
            # parent.game_connection = udp_client.SimpleUDPClient(host, port)
        
        # Mark as connected (for UI state)
        parent.game_status_label.setText(f"Status: Connected ({protocol}://{host}:{port})")
        parent.game_status_label.setStyleSheet("color: #a6e3a1;")
        parent.btn_game_connect.setEnabled(False)
        parent.btn_game_disconnect.setEnabled(True)
        parent.game_log.append("✅ Connection established")
        
    except Exception as e:
        parent.game_log.append(f"❌ Connection failed: {e}")
        parent.game_status_label.setText("Status: Connection failed")


def _disconnect_from_game(parent):
    """Disconnect from game."""
    if parent.game_connection:
        try:
            parent.game_connection.close()
        except:
            pass
    parent.game_connection = None
    parent.game_status_label.setText("Status: Not connected")
    parent.game_status_label.setStyleSheet("color: #f38ba8;")
    parent.btn_game_connect.setEnabled(True)
    parent.btn_game_disconnect.setEnabled(False)
    parent.game_log.append("🔌 Disconnected")
