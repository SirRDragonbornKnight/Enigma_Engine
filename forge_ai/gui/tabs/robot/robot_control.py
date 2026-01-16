"""
Robot Control Module

Allows the AI to connect to and control physical robots.
The same AI being trained learns to control these robots.

Supports loading connection configs from files instead of presets.
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, 
    QGroupBox, QLineEdit, QTextEdit, QFileDialog
)
import json

from ....config import CONFIG


# Robot configs directory
ROBOT_CONFIG_DIR = Path(CONFIG["data_dir"]) / "robot"


def create_robot_subtab(parent):
    """
    Create the robot control sub-tab.
    
    The AI you are training will learn to control these robots.
    Connection settings are loaded from config files.
    """
    widget = QWidget()
    layout = QVBoxLayout()
    layout.setSpacing(8)
    layout.setContentsMargins(10, 10, 10, 10)
    
    # === OUTPUT AT TOP ===
    # Log (main output area)
    parent.robot_log = QTextEdit()
    parent.robot_log.setReadOnly(True)
    parent.robot_log.setPlaceholderText("Robot communication will appear here...")
    parent.robot_log.setStyleSheet("""
        QTextEdit {
            background-color: #1e1e2e;
            border: 1px solid #313244;
            border-radius: 4px;
            padding: 8px;
            font-family: 'Consolas', monospace;
        }
    """)
    layout.addWidget(parent.robot_log, stretch=1)
    
    # Status
    parent.robot_status_label = QLabel("Status: Not connected")
    parent.robot_status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
    layout.addWidget(parent.robot_status_label)
    
    # === CONTROLS ===
    # Config file selector row
    config_row = QHBoxLayout()
    config_row.addWidget(QLabel("Config:"))
    parent.robot_config_combo = QComboBox()
    parent.robot_config_combo.currentIndexChanged.connect(
        lambda idx: _on_robot_config_changed(parent, idx)
    )
    config_row.addWidget(parent.robot_config_combo, stretch=1)
    btn_refresh = QPushButton("Refresh")
    btn_refresh.setMaximumWidth(60)
    btn_refresh.clicked.connect(lambda: _refresh_robot_configs(parent))
    config_row.addWidget(btn_refresh)
    btn_new = QPushButton("New")
    btn_new.setMaximumWidth(40)
    btn_new.clicked.connect(lambda: _create_new_robot_config(parent))
    config_row.addWidget(btn_new)
    layout.addLayout(config_row)
    
    # Connection type row
    type_row = QHBoxLayout()
    type_row.addWidget(QLabel("Type:"))
    parent.robot_type_combo = QComboBox()
    parent.robot_type_combo.addItems(["serial", "http", "ros", "gpio", "mqtt"])
    parent.robot_type_combo.setMaximumWidth(80)
    parent.robot_type_combo.currentTextChanged.connect(
        lambda t: _on_robot_type_changed(parent, t)
    )
    type_row.addWidget(parent.robot_type_combo)
    type_row.addWidget(QLabel("Port:"))
    parent.robot_port_input = QLineEdit("/dev/ttyUSB0")
    parent.robot_port_input.setMaximumWidth(120)
    type_row.addWidget(parent.robot_port_input)
    type_row.addWidget(QLabel("Baud:"))
    parent.robot_baud_combo = QComboBox()
    parent.robot_baud_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
    parent.robot_baud_combo.setCurrentText("115200")
    parent.robot_baud_combo.setMaximumWidth(80)
    type_row.addWidget(parent.robot_baud_combo)
    type_row.addStretch()
    layout.addLayout(type_row)
    
    # Network settings row
    net_row = QHBoxLayout()
    net_row.addWidget(QLabel("Host:"))
    parent.robot_host_input = QLineEdit("localhost")
    parent.robot_host_input.setMaximumWidth(120)
    net_row.addWidget(parent.robot_host_input)
    net_row.addWidget(QLabel("Net Port:"))
    parent.robot_net_port_input = QLineEdit("11311")
    parent.robot_net_port_input.setMaximumWidth(70)
    net_row.addWidget(parent.robot_net_port_input)
    net_row.addStretch()
    layout.addLayout(net_row)
    
    # Command input row
    cmd_row = QHBoxLayout()
    cmd_row.addWidget(QLabel("Command:"))
    parent.robot_cmd_input = QLineEdit()
    parent.robot_cmd_input.setPlaceholderText("Enter command to send...")
    parent.robot_cmd_input.returnPressed.connect(lambda: _send_robot_command(parent))
    cmd_row.addWidget(parent.robot_cmd_input)
    btn_send = QPushButton("Send")
    btn_send.setMaximumWidth(50)
    btn_send.clicked.connect(lambda: _send_robot_command(parent))
    cmd_row.addWidget(btn_send)
    layout.addLayout(cmd_row)
    
    # Buttons row
    btn_row = QHBoxLayout()
    parent.btn_robot_connect = QPushButton("Connect")
    parent.btn_robot_connect.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: bold;")
    parent.btn_robot_connect.clicked.connect(lambda: _connect_robot(parent))
    btn_row.addWidget(parent.btn_robot_connect)
    parent.btn_robot_disconnect = QPushButton("Disconnect")
    parent.btn_robot_disconnect.clicked.connect(lambda: _disconnect_robot(parent))
    parent.btn_robot_disconnect.setEnabled(False)
    btn_row.addWidget(parent.btn_robot_disconnect)
    btn_save_config = QPushButton("Save Config")
    btn_save_config.clicked.connect(lambda: _save_current_robot_config(parent))
    btn_row.addWidget(btn_save_config)
    btn_row.addStretch()
    layout.addLayout(btn_row)
    
    widget.setLayout(layout)
    
    # Initialize
    parent.robot_connection = None
    ROBOT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _refresh_robot_configs(parent)
    
    return widget
    
    return widget


def load_robot_configs() -> list:
    """Get list of available robot config files."""
    configs = []
    if ROBOT_CONFIG_DIR.exists():
        for f in ROBOT_CONFIG_DIR.glob("*.json"):
            configs.append(f)
    return configs


def _refresh_robot_configs(parent):
    """Refresh the list of available robot configs."""
    parent.robot_config_combo.clear()
    parent.robot_config_combo.addItem("-- Select Config File --", None)
    
    if ROBOT_CONFIG_DIR.exists():
        for config_file in sorted(ROBOT_CONFIG_DIR.glob("*.json")):
            parent.robot_config_combo.addItem(config_file.stem, str(config_file))


def _on_robot_config_changed(parent, index):
    """Handle robot config selection."""
    config_path = parent.robot_config_combo.currentData()
    if not config_path:
        return
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Apply config to UI
        if "type" in config:
            idx = parent.robot_type_combo.findText(config["type"])
            if idx >= 0:
                parent.robot_type_combo.setCurrentIndex(idx)
        if "port" in config:
            parent.robot_port_input.setText(config["port"])
        if "baud" in config:
            parent.robot_baud_combo.setCurrentText(str(config["baud"]))
        if "host" in config:
            parent.robot_host_input.setText(config["host"])
        if "net_port" in config:
            parent.robot_net_port_input.setText(str(config["net_port"]))
        
        parent.robot_log.append(f"Loaded config: {Path(config_path).stem}")
        
    except Exception as e:
        parent.robot_log.append(f"[X] Failed to load config: {e}")


def _on_robot_type_changed(parent, robot_type):
    """Update UI based on robot type."""
    is_serial = robot_type == "serial"
    is_network = robot_type in ["http", "ros", "mqtt"]
    
    parent.robot_port_input.setEnabled(is_serial or robot_type == "gpio")
    parent.robot_baud_combo.setEnabled(is_serial)
    parent.robot_host_input.setEnabled(is_network)
    parent.robot_net_port_input.setEnabled(is_network)


def _browse_robot_config(parent):
    """Browse for a robot config file."""
    path, _ = QFileDialog.getOpenFileName(
        parent, "Select Robot Config",
        str(ROBOT_CONFIG_DIR),
        "JSON Files (*.json);;All Files (*)"
    )
    if path:
        for i in range(parent.robot_config_combo.count()):
            if parent.robot_config_combo.itemData(i) == path:
                parent.robot_config_combo.setCurrentIndex(i)
                return
        
        parent.robot_config_combo.addItem(Path(path).stem, path)
        parent.robot_config_combo.setCurrentIndex(parent.robot_config_combo.count() - 1)


def _create_new_robot_config(parent):
    """Create a new robot config file."""
    path, _ = QFileDialog.getSaveFileName(
        parent, "Create Robot Config",
        str(ROBOT_CONFIG_DIR / "new_robot.json"),
        "JSON Files (*.json)"
    )
    if path:
        config = {
            "name": Path(path).stem,
            "type": "serial",
            "port": "/dev/ttyUSB0",
            "baud": 115200,
            "description": "New robot connection config"
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        _refresh_robot_configs(parent)
        # Select the new config
        for i in range(parent.robot_config_combo.count()):
            if parent.robot_config_combo.itemData(i) == path:
                parent.robot_config_combo.setCurrentIndex(i)
                break


def _save_current_robot_config(parent):
    """Save current settings to the selected config file."""
    config_path = parent.robot_config_combo.currentData()
    if not config_path:
        _create_new_robot_config(parent)
        return
    
    config = {
        "name": Path(config_path).stem,
        "type": parent.robot_type_combo.currentText(),
        "port": parent.robot_port_input.text(),
        "baud": int(parent.robot_baud_combo.currentText()),
        "host": parent.robot_host_input.text(),
        "net_port": parent.robot_net_port_input.text()
    }
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        parent.robot_log.append(f"[OK] Config saved: {Path(config_path).stem}")
    except Exception as e:
        parent.robot_log.append(f"[X] Failed to save: {e}")


def _connect_robot(parent):
    """Connect to robot using current settings."""
    robot_type = parent.robot_type_combo.currentText()
    
    parent.robot_log.append(f"Connecting via {robot_type}...")
    
    try:
        if robot_type == "serial":
            port = parent.robot_port_input.text()
            baud = int(parent.robot_baud_combo.currentText())
            parent.robot_log.append(f"[>] Serial: {port} @ {baud} baud...")
            # Actual serial connection
            try:
                import serial  # type: ignore
            except ImportError:
                parent.robot_log.append("[!] pyserial not installed. Install with: pip install pyserial")
                parent.robot_connection = None
            else:
                try:
                    parent.robot_connection = serial.Serial(port, baud, timeout=1)
                    parent.robot_log.append("[OK] Serial connection established")
                except (serial.SerialException, OSError, ValueError) as e:
                    parent.robot_log.append(f"[X] Serial connection failed: {e}")
                    parent.robot_connection = None
            
        elif robot_type == "http":
            host = parent.robot_host_input.text()
            port = parent.robot_net_port_input.text()
            parent.robot_log.append(f"[>] HTTP: http://{host}:{port}")
            # Store connection info for HTTP requests
            parent.robot_connection = {"type": "http", "url": f"http://{host}:{port}"}
            
        elif robot_type == "ros":
            host = parent.robot_host_input.text()
            port = parent.robot_net_port_input.text()
            parent.robot_log.append(f"[*] ROS Master: {host}:{port}")
            # ROS connection
            try:
                import rospy  # type: ignore
                if not rospy.core.is_initialized():
                    rospy.init_node('forge_robot_client', anonymous=True)
                parent.robot_connection = {"type": "ros", "host": host, "port": port}
                parent.robot_log.append("[OK] ROS node initialized")
            except ImportError:
                parent.robot_log.append("[!] rospy not installed. ROS not available.")
                parent.robot_connection = None
            
        elif robot_type == "gpio":
            parent.robot_log.append("[>] GPIO initialized")
            # GPIO setup for Raspberry Pi
            try:
                import RPi.GPIO as GPIO  # type: ignore
                GPIO.setmode(GPIO.BCM)
                parent.robot_connection = {"type": "gpio", "GPIO": GPIO}
                parent.robot_log.append("[OK] GPIO configured (BCM mode)")
            except ImportError:
                parent.robot_log.append("[!] RPi.GPIO not installed. GPIO not available.")
                parent.robot_connection = None
            
        elif robot_type == "mqtt":
            host = parent.robot_host_input.text()
            port = parent.robot_net_port_input.text()
            parent.robot_log.append(f"[>] MQTT: {host}:{port}")
            # MQTT connection
            try:
                import paho.mqtt.client as mqtt  # type: ignore
                client = mqtt.Client()
                client.connect(host, int(port))
                client.loop_start()
                parent.robot_connection = {"type": "mqtt", "client": client}
                parent.robot_log.append("[OK] MQTT connection established")
            except ImportError:
                parent.robot_log.append("[!] paho-mqtt not installed. Install with: pip install paho-mqtt")
                parent.robot_connection = None
        
        # Mark as connected if we have a connection
        if parent.robot_connection:
            parent.robot_status_label.setText(f"Status: Connected ({robot_type})")
            parent.robot_status_label.setStyleSheet("color: #a6e3a1;")
            parent.btn_robot_connect.setEnabled(False)
            parent.btn_robot_disconnect.setEnabled(True)
            parent.robot_log.append("[OK] Connection established")
        else:
            parent.robot_status_label.setText("Status: Connection failed (missing library)")
            parent.robot_status_label.setStyleSheet("color: #f38ba8;")
        
    except Exception as e:
        parent.robot_log.append(f"[X] Connection failed: {e}")
        parent.robot_status_label.setText("Status: Connection failed")


def _disconnect_robot(parent):
    """Disconnect from robot."""
    if parent.robot_connection:
        try:
            parent.robot_connection.close()
        except Exception:
            pass
    parent.robot_connection = None
    parent.robot_status_label.setText("Status: Not connected")
    parent.robot_status_label.setStyleSheet("color: #f38ba8;")
    parent.btn_robot_connect.setEnabled(True)
    parent.btn_robot_disconnect.setEnabled(False)
    parent.robot_log.append("[x] Disconnected")


def _send_robot_command(parent):
    """Send command to robot."""
    cmd = parent.robot_cmd_input.text().strip()
    if not cmd:
        return
    
    parent.robot_log.append(f">> {cmd}")
    parent.robot_cmd_input.clear()
    
    if parent.robot_connection:
        try:
            parent.robot_connection.write(f"{cmd}\n".encode())
        except Exception as e:
            parent.robot_log.append(f"[X] Send failed: {e}")
    else:
        parent.robot_log.append("[!] Not connected - command logged only")
