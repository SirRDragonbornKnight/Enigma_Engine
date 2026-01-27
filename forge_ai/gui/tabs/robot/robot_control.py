"""
Robot Control Module

Allows the AI to connect to and control physical robots.
The same AI being trained learns to control these robots.

Supports loading connection configs from files instead of presets.
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QLineEdit, QTextEdit, QFileDialog, QCheckBox, QMessageBox
)
import json

from ....config import CONFIG
from ..shared_components import NoScrollComboBox


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
    
    # === SAFETY CONTROLS AT TOP (Easy Access) ===
    safety_group = QGroupBox("Safety Controls")
    safety_layout = QVBoxLayout(safety_group)
    safety_layout.setSpacing(8)
    
    # E-STOP button - BIG and prominent
    parent.estop_btn = QPushButton("EMERGENCY STOP")
    parent.estop_btn.setMinimumHeight(50)
    parent.estop_btn.setStyleSheet("""
        QPushButton {
            background-color: #dc2626;
            color: white;
            font-weight: bold;
            font-size: 18px;
            padding: 15px;
            border-radius: 8px;
            border: 3px solid #991b1b;
        }
        QPushButton:hover {
            background-color: #b91c1c;
        }
        QPushButton:pressed {
            background-color: #7f1d1d;
        }
    """)
    parent.estop_btn.clicked.connect(lambda: _robot_estop(parent))
    safety_layout.addWidget(parent.estop_btn)
    
    # Mode and Camera row
    mode_row = QHBoxLayout()
    mode_row.addWidget(QLabel("Mode:"))
    parent.robot_mode_combo = NoScrollComboBox()
    parent.robot_mode_combo.setToolTip("Select robot control mode")
    parent.robot_mode_combo.addItem("DISABLED", "disabled")
    parent.robot_mode_combo.addItem("MANUAL (User)", "manual")
    parent.robot_mode_combo.addItem("AUTO (AI)", "auto")
    parent.robot_mode_combo.addItem("SAFE (Limited)", "safe")
    parent.robot_mode_combo.currentIndexChanged.connect(
        lambda: _change_robot_mode(parent)
    )
    mode_row.addWidget(parent.robot_mode_combo)
    
    parent.robot_camera_check = QCheckBox("Camera Feed")
    parent.robot_camera_check.stateChanged.connect(
        lambda state: _toggle_robot_camera(parent, state)
    )
    mode_row.addWidget(parent.robot_camera_check)
    mode_row.addStretch()
    safety_layout.addLayout(mode_row)
    
    layout.addWidget(safety_group)
    
    # === OUTPUT ===
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
    parent.robot_config_combo = NoScrollComboBox()
    parent.robot_config_combo.setToolTip("Select a robot configuration file")
    parent.robot_config_combo.currentIndexChanged.connect(
        lambda idx: _on_robot_config_changed(parent, idx)
    )
    config_row.addWidget(parent.robot_config_combo, stretch=1)
    btn_refresh = QPushButton("Refresh")
    btn_refresh.setMaximumWidth(60)
    btn_refresh.clicked.connect(lambda: _refresh_robot_configs(parent, preserve_selection=True))
    config_row.addWidget(btn_refresh)
    btn_new = QPushButton("New")
    btn_new.setMaximumWidth(40)
    btn_new.clicked.connect(lambda: _create_new_robot_config(parent))
    config_row.addWidget(btn_new)
    layout.addLayout(config_row)
    
    # Connection type row
    type_row = QHBoxLayout()
    type_row.addWidget(QLabel("Type:"))
    parent.robot_type_combo = NoScrollComboBox()
    parent.robot_type_combo.setToolTip("Select the connection type for the robot")
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
    parent.robot_baud_combo = NoScrollComboBox()
    parent.robot_baud_combo.setToolTip("Select the baud rate for serial connection")
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


def _refresh_robot_configs(parent, preserve_selection=False):
    """Refresh the list of available robot configs."""
    # Remember current selection if preserving
    previous_data = parent.robot_config_combo.currentData() if preserve_selection else None
    
    parent.robot_config_combo.blockSignals(True)
    parent.robot_config_combo.clear()
    parent.robot_config_combo.addItem("-- Select Config File --", None)
    
    if ROBOT_CONFIG_DIR.exists():
        for config_file in sorted(ROBOT_CONFIG_DIR.glob("*.json")):
            parent.robot_config_combo.addItem(config_file.stem, str(config_file))
    
    # Restore selection if found
    if previous_data:
        for i in range(parent.robot_config_combo.count()):
            if parent.robot_config_combo.itemData(i) == previous_data:
                parent.robot_config_combo.setCurrentIndex(i)
                break
    
    parent.robot_config_combo.blockSignals(False)


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


# ===== SAFETY CONTROLS =====

def _change_robot_mode(parent):
    """Change robot control mode."""
    mode = parent.robot_mode_combo.currentData()
    
    try:
        from forge_ai.tools.robot_modes import get_mode_controller, RobotMode
        
        controller = get_mode_controller()
        
        mode_map = {
            "disabled": RobotMode.DISABLED,
            "manual": RobotMode.MANUAL,
            "auto": RobotMode.AUTO,
            "safe": RobotMode.SAFE,
        }
        
        robot_mode = mode_map.get(mode, RobotMode.DISABLED)
        success = controller.set_mode(robot_mode)
        
        if success:
            status_map = {
                "disabled": ("Status: Disabled", "#888"),
                "manual": ("Status: MANUAL - User control", "#22c55e"),
                "auto": ("Status: AUTO - AI control enabled", "#3b82f6"),
                "safe": ("Status: SAFE - Limited AI control", "#f59e0b"),
            }
            text, color = status_map.get(mode, ("Status: Unknown", "#888"))
            parent.robot_status_label.setText(text)
            parent.robot_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            parent.robot_log.append(f"[MODE] Changed to {mode.upper()}")
        else:
            parent.robot_status_label.setText("Status: Failed to change mode")
            parent.robot_status_label.setStyleSheet("color: #ef4444;")
    except ImportError:
        parent.robot_status_label.setText("Status: Mode controller not configured")
        parent.robot_status_label.setStyleSheet("color: #888;")
    except Exception as e:
        parent.robot_status_label.setText(f"Status: Error - {e}")
        parent.robot_status_label.setStyleSheet("color: #ef4444;")


def _robot_estop(parent):
    """Emergency stop the robot."""
    try:
        from forge_ai.tools.robot_modes import get_mode_controller
        
        controller = get_mode_controller()
        controller.emergency_stop("User pressed E-STOP button")
        
        parent.robot_status_label.setText("Status: E-STOP ACTIVE")
        parent.robot_status_label.setStyleSheet("color: #ef4444; font-weight: bold;")
        parent.robot_mode_combo.setCurrentIndex(0)  # Set to disabled
        parent.robot_log.append("[E-STOP] Emergency stop activated!")
        
        QMessageBox.critical(
            parent, "Emergency Stop",
            "Robot has been emergency stopped!\n\n"
            "To resume, set robot to DISABLED mode first, then re-enable."
        )
    except Exception as e:
        parent.robot_log.append(f"[E-STOP] Error: {e}")
        QMessageBox.warning(parent, "E-STOP Error", f"Could not E-STOP: {e}")


def _toggle_robot_camera(parent, state):
    """Toggle robot camera feed."""
    enabled = state == 2
    
    try:
        from forge_ai.tools.robot_modes import get_mode_controller, CameraConfig
        
        controller = get_mode_controller()
        
        if enabled:
            controller.setup_camera(CameraConfig(enabled=True, device_id=0))
            controller.start_camera()
            parent.robot_log.append("[CAMERA] Camera feed enabled")
        else:
            controller.stop_camera()
            parent.robot_log.append("[CAMERA] Camera feed disabled")
    except ImportError:
        parent.robot_camera_check.setChecked(False)
        QMessageBox.information(
            parent, "OpenCV Required",
            "Camera requires OpenCV. Install with:\npip install opencv-python"
        )
    except Exception as e:
        parent.robot_camera_check.setChecked(False)
        QMessageBox.warning(parent, "Camera Error", f"Could not enable camera: {e}")
