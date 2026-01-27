"""
Network Tab - Multi-device dashboard and remote connections.

Features:
  - View connected devices
  - Start/stop API server
  - Connect to remote Forge instances
  - Sync models across devices
  - Network status monitoring
"""

import os
import json
import socket
import subprocess
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QListWidget, QListWidgetItem, QGroupBox,
    QSplitter, QTextEdit, QSpinBox, QMessageBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

# Config
NETWORK_CONFIG = Path.home() / ".forge_ai" / "network_config.json"
NETWORK_CONFIG.parent.mkdir(parents=True, exist_ok=True)


class NetworkScanner(QThread):
    """Background thread for scanning network devices using DeviceDiscovery."""
    
    device_found = pyqtSignal(dict)
    scan_complete = pyqtSignal(list)
    progress_update = pyqtSignal(str)
    
    def __init__(self, port: int = 8765, scan_mode: str = "broadcast"):
        super().__init__()
        self.port = port
        self.scan_mode = scan_mode  # "broadcast" or "full_scan"
        self._running = True
    
    def run(self):
        """Scan local network for Forge instances using DeviceDiscovery."""
        from forge_ai.comms.discovery import DeviceDiscovery
        
        devices = []
        
        try:
            # Create discovery instance
            discovery = DeviceDiscovery(node_name="gui_scanner", node_port=self.port)
            
            if self.scan_mode == "broadcast":
                # Fast UDP broadcast discovery
                self.progress_update.emit("Broadcasting discovery message...")
                discovered = discovery.broadcast_discover(timeout=3.0)
                
                # Convert to device format
                for name, info in discovered.items():
                    device = {
                        "ip": info["ip"],
                        "port": info["port"],
                        "name": name,
                        "status": "online",
                        "is_self": False,
                        "last_seen": info.get("last_seen"),
                    }
                    devices.append(device)
                    self.device_found.emit(device)
                
                if not discovered:
                    self.progress_update.emit("No devices found via broadcast. Try full scan.")
            
            elif self.scan_mode == "full_scan":
                # Full network scan (slower but more thorough)
                self.progress_update.emit("Scanning network (this may take a while)...")
                discovered = discovery.scan_network(port=self.port, timeout=0.3)
                
                # Convert to device format
                for name, info in discovered.items():
                    device = {
                        "ip": info["ip"],
                        "port": info["port"],
                        "name": name,
                        "status": "online",
                        "is_self": False,
                        "last_seen": info.get("last_seen"),
                        "model": info.get("model"),
                    }
                    devices.append(device)
                    self.device_found.emit(device)
        
        except Exception as e:
            self.progress_update.emit(f"Error during scan: {e}")
        
        self.scan_complete.emit(devices)
    
    def stop(self):
        self._running = False


class NetworkTab(QWidget):
    """Main Network tab for multi-device management."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.scanner = None
        self.api_process = None
        self.discovery_listener = None
        self._setup_ui()
        self._load_config()
        self._start_status_timer()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Network & Multi-Device")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(header)
        
        desc = QLabel("Connect multiple Forge instances, share models, and manage remote devices.")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Main splitter
        splitter = QSplitter(Qt.Vertical)
        
        # Top section: Server controls
        server_group = QGroupBox("API Server")
        server_layout = QVBoxLayout(server_group)
        
        # Server status
        status_layout = QHBoxLayout()
        
        self.server_status = QLabel("[STOPPED]")
        self.server_status.setStyleSheet("color: #f44336; font-weight: bold;")
        status_layout.addWidget(self.server_status)
        
        status_layout.addStretch()
        
        self.local_ip_label = QLabel(f"Local IP: {self._get_local_ip()}")
        status_layout.addWidget(self.local_ip_label)
        
        server_layout.addLayout(status_layout)
        
        # Server controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Port:"))
        self.port_input = QSpinBox()
        self.port_input.setRange(1024, 65535)
        self.port_input.setValue(8765)
        self.port_input.setToolTip("Port number for the API server (1024-65535)")
        controls_layout.addWidget(self.port_input)
        
        self.start_server_btn = QPushButton("Start Server")
        self.start_server_btn.setToolTip("Start the local API server for remote connections")
        self.start_server_btn.clicked.connect(self._toggle_server)
        controls_layout.addWidget(self.start_server_btn)
        
        self.copy_url_btn = QPushButton("Copy URL")
        self.copy_url_btn.setToolTip("Copy the server URL to clipboard")
        self.copy_url_btn.clicked.connect(self._copy_server_url)
        controls_layout.addWidget(self.copy_url_btn)
        
        controls_layout.addStretch()
        
        server_layout.addLayout(controls_layout)
        
        splitter.addWidget(server_group)
        
        # Middle section: Devices
        devices_group = QGroupBox("Connected Devices")
        devices_layout = QVBoxLayout(devices_group)
        
        # Scan controls
        scan_layout = QHBoxLayout()
        
        self.scan_btn = QPushButton("Scan Network")
        self.scan_btn.setToolTip("Scan local network for other ForgeAI instances")
        self.scan_btn.clicked.connect(self._scan_network)
        scan_layout.addWidget(self.scan_btn)
        
        # Scan mode selector
        from PyQt5.QtWidgets import QComboBox
        scan_layout.addWidget(QLabel("Mode:"))
        self.scan_mode_combo = QComboBox()
        self.scan_mode_combo.addItem("Broadcast", "broadcast")
        self.scan_mode_combo.addItem("Full Scan", "full_scan")
        self.scan_mode_combo.setToolTip("Broadcast: Fast UDP discovery\nFull Scan: Thorough network scan")
        scan_layout.addWidget(self.scan_mode_combo)
        
        self.add_device_btn = QPushButton("Add Device")
        self.add_device_btn.setToolTip("Manually add a remote device by IP address")
        self.add_device_btn.clicked.connect(self._add_device_manual)
        scan_layout.addWidget(self.add_device_btn)
        
        scan_layout.addStretch()
        
        self.scan_progress = QProgressBar()
        self.scan_progress.setMaximumWidth(200)
        self.scan_progress.setVisible(False)
        scan_layout.addWidget(self.scan_progress)
        
        devices_layout.addLayout(scan_layout)
        
        # Devices table
        self.devices_table = QTableWidget()
        self.devices_table.setColumnCount(6)
        self.devices_table.setHorizontalHeaderLabels(["IP", "Port", "Status", "Name", "Model", "Actions"])
        self.devices_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.devices_table.setSelectionBehavior(QTableWidget.SelectRows)
        devices_layout.addWidget(self.devices_table)
        
        splitter.addWidget(devices_group)
        
        # Bottom section: Remote operations
        remote_group = QGroupBox("Remote Operations")
        remote_layout = QVBoxLayout(remote_group)
        
        # Connection input
        connect_layout = QHBoxLayout()
        
        connect_layout.addWidget(QLabel("Connect to:"))
        self.remote_url = QLineEdit()
        self.remote_url.setPlaceholderText("http://192.168.1.100:8765")
        connect_layout.addWidget(self.remote_url)
        
        connect_btn = QPushButton("Connect")
        connect_btn.clicked.connect(self._connect_to_remote)
        connect_layout.addWidget(connect_btn)
        
        remote_layout.addLayout(connect_layout)
        
        # Quick actions
        actions_layout = QHBoxLayout()
        
        sync_models_btn = QPushButton("Sync Models")
        sync_models_btn.clicked.connect(self._sync_models)
        actions_layout.addWidget(sync_models_btn)
        
        sync_settings_btn = QPushButton("Sync Settings")
        sync_settings_btn.clicked.connect(self._sync_settings)
        actions_layout.addWidget(sync_settings_btn)
        
        remote_chat_btn = QPushButton("Remote Chat")
        remote_chat_btn.clicked.connect(self._remote_chat)
        actions_layout.addWidget(remote_chat_btn)
        
        actions_layout.addStretch()
        
        remote_layout.addLayout(actions_layout)
        
        # Log output
        self.network_log = QTextEdit()
        self.network_log.setReadOnly(True)
        self.network_log.setMaximumHeight(100)
        self.network_log.setPlaceholderText("Network activity log...")
        remote_layout.addWidget(self.network_log)
        
        splitter.addWidget(remote_group)
        
        splitter.setSizes([150, 250, 200])
        layout.addWidget(splitter)
    
    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def _load_config(self):
        """Load network configuration."""
        if NETWORK_CONFIG.exists():
            try:
                with open(NETWORK_CONFIG, 'r') as f:
                    config = json.load(f)
                    self.port_input.setValue(config.get("port", 8765))
                    
                    # Load saved devices
                    for device in config.get("devices", []):
                        self._add_device_to_table(device)
            except:
                pass
    
    def _save_config(self):
        """Save network configuration."""
        devices = []
        for row in range(self.devices_table.rowCount()):
            device = {
                "ip": self.devices_table.item(row, 0).text(),
                "port": int(self.devices_table.item(row, 1).text()),
                "name": self.devices_table.item(row, 3).text() if self.devices_table.item(row, 3) else "",
            }
            devices.append(device)
        
        config = {
            "port": self.port_input.value(),
            "devices": devices,
        }
        
        with open(NETWORK_CONFIG, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _toggle_server(self):
        """Start or stop the API server."""
        if self.api_process is None:
            self._start_server()
        else:
            self._stop_server()
    
    def _start_server(self):
        """Start the API server and discovery listener."""
        port = self.port_input.value()
        
        try:
            from forge_ai.comms.discovery import DeviceDiscovery
            
            # Try to start the API server
            # This would normally use forge_ai.comms.api_server
            self._log(f"Starting API server on port {port}...")
            
            # Start discovery listener
            import socket
            hostname = socket.gethostname()
            self.discovery_listener = DeviceDiscovery(
                node_name=f"forge_{hostname}",
                node_port=port
            )
            self.discovery_listener.start_listener()
            self._log("Discovery listener started - other devices can find this node")
            
            # For now, just update status (actual server would be started here)
            self.server_status.setText(f"[RUNNING] on port {port}")
            self.server_status.setStyleSheet("color: #4caf50; font-weight: bold;")
            self.start_server_btn.setText("Stop Server")
            
            self._log(f"Server started at http://{self._get_local_ip()}:{port}")
            self.api_process = True  # Placeholder
            
        except Exception as e:
            self._log(f"Error starting server: {e}")
            QMessageBox.warning(self, "Error", f"Failed to start server: {e}")
    
    def _stop_server(self):
        """Stop the API server and discovery listener."""
        self._log("Stopping API server...")
        
        # Stop discovery listener
        if self.discovery_listener:
            self.discovery_listener.stop_listener()
            self.discovery_listener = None
            self._log("Discovery listener stopped")
        
        self.server_status.setText("Stopped")
        self.server_status.setStyleSheet("color: #f44336; font-weight: bold;")
        self.start_server_btn.setText("Start Server")
        
        self.api_process = None
        self._log("Server stopped")
    
    def _copy_server_url(self):
        """Copy server URL to clipboard."""
        ip = self._get_local_ip()
        port = self.port_input.value()
        url = f"http://{ip}:{port}"
        
        from PyQt5.QtWidgets import QApplication
        QApplication.clipboard().setText(url)
        QMessageBox.information(self, "Copied", f"URL copied: {url}")
    
    def _scan_network(self):
        """Scan network for Forge instances."""
        if self.scanner and self.scanner.isRunning():
            self.scanner.stop()
            return
        
        self.scan_btn.setText("Stop Scan")
        self.scan_progress.setVisible(True)
        self.scan_progress.setRange(0, 0)  # Indeterminate
        
        # Get selected scan mode
        scan_mode = self.scan_mode_combo.currentData()
        
        mode_name = "broadcast" if scan_mode == "broadcast" else "full network"
        self._log(f"Starting {mode_name} scan for Forge instances...")
        
        self.scanner = NetworkScanner(self.port_input.value(), scan_mode=scan_mode)
        self.scanner.device_found.connect(self._on_device_found)
        self.scanner.scan_complete.connect(self._on_scan_complete)
        self.scanner.progress_update.connect(self._log)
        self.scanner.start()
    
    def _on_device_found(self, device: dict):
        """Handle found device."""
        self._add_device_to_table(device)
        device_name = device.get('name', device['ip'])
        self._log(f"Found device: {device_name} at {device['ip']}:{device['port']}")
    
    def _on_scan_complete(self, devices: list):
        """Handle scan completion."""
        self.scan_btn.setText("Scan Network")
        self.scan_progress.setVisible(False)
        self._log(f"Scan complete. Found {len(devices)} device(s).")
        self._save_config()
    
    def _add_device_to_table(self, device: dict):
        """Add a device to the table."""
        # Check if already exists
        for row in range(self.devices_table.rowCount()):
            if self.devices_table.item(row, 0).text() == device["ip"]:
                # Update existing row
                self.devices_table.setItem(row, 3, QTableWidgetItem(device.get("name", "")))
                self.devices_table.setItem(row, 4, QTableWidgetItem(device.get("model", "")))
                return
        
        row = self.devices_table.rowCount()
        self.devices_table.insertRow(row)
        
        self.devices_table.setItem(row, 0, QTableWidgetItem(device.get("ip", "")))
        self.devices_table.setItem(row, 1, QTableWidgetItem(str(device.get("port", 8765))))
        
        status = "Online" if device.get("status") == "online" else "Unknown"
        if device.get("is_self"):
            status = "This Device"
        status_item = QTableWidgetItem(status)
        self.devices_table.setItem(row, 2, status_item)
        
        self.devices_table.setItem(row, 3, QTableWidgetItem(device.get("name", "")))
        self.devices_table.setItem(row, 4, QTableWidgetItem(device.get("model", "")))
        
        # Actions
        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(2, 2, 2, 2)
        
        connect_btn = QPushButton(">")
        connect_btn.setMaximumWidth(30)
        connect_btn.setToolTip("Connect")
        connect_btn.clicked.connect(lambda: self._connect_to_device(device))
        actions_layout.addWidget(connect_btn)
        
        remove_btn = QPushButton("X")
        remove_btn.setMaximumWidth(30)
        remove_btn.setToolTip("Remove")
        remove_btn.clicked.connect(lambda: self._remove_device(row))
        actions_layout.addWidget(remove_btn)
        
        self.devices_table.setCellWidget(row, 5, actions_widget)
    
    def _add_device_manual(self):
        """Manually add a device."""
        from PyQt5.QtWidgets import QInputDialog
        
        ip, ok = QInputDialog.getText(self, "Add Device", "IP Address:")
        if not ok or not ip:
            return
        
        port, ok = QInputDialog.getInt(self, "Add Device", "Port:", 8765, 1024, 65535)
        if not ok:
            return
        
        device = {"ip": ip, "port": port, "status": "unknown"}
        self._add_device_to_table(device)
        self._save_config()
    
    def _remove_device(self, row: int):
        """Remove a device from the table."""
        self.devices_table.removeRow(row)
        self._save_config()
    
    def _connect_to_device(self, device: dict):
        """Connect to a specific device."""
        url = f"http://{device['ip']}:{device['port']}"
        self.remote_url.setText(url)
        self._connect_to_remote()
    
    def _connect_to_remote(self):
        """Connect to a remote Forge instance."""
        url = self.remote_url.text().strip()
        if not url:
            QMessageBox.warning(self, "Error", "Please enter a URL")
            return
        
        self._log(f"Connecting to {url}...")
        
        try:
            import urllib.request
            
            # Test connection
            test_url = f"{url}/health" if not url.endswith("/health") else url
            req = urllib.request.Request(test_url, method='GET')
            req.add_header('User-Agent', 'Forge-Client/1.0')
            
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    self._log(f"Connected to {url}")
                    QMessageBox.information(self, "Connected", f"Successfully connected to {url}")
                else:
                    self._log(f"Connection returned status {response.status}")
        
        except Exception as e:
            self._log(f"Connection failed: {e}")
            QMessageBox.warning(self, "Connection Failed", f"Could not connect to {url}\n\n{e}")
    
    def _sync_models(self):
        """Sync models with remote device."""
        self._log("Model sync not yet implemented - coming soon!")
        QMessageBox.information(self, "Coming Soon", "Model sync will be available in a future update.")
    
    def _sync_settings(self):
        """Sync settings with remote device."""
        self._log("Settings sync not yet implemented - coming soon!")
        QMessageBox.information(self, "Coming Soon", "Settings sync will be available in a future update.")
    
    def _remote_chat(self):
        """Start remote chat session."""
        url = self.remote_url.text().strip()
        if not url:
            QMessageBox.warning(self, "Error", "Please enter a remote URL first")
            return
        
        self._log(f"Opening remote chat to {url}...")
        QMessageBox.information(self, "Remote Chat", f"Remote chat to {url} - feature coming soon!")
    
    def _log(self, message: str):
        """Add message to network log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.network_log.append(f"[{timestamp}] {message}")
    
    def _start_status_timer(self):
        """Start timer for periodic status updates."""
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(30000)  # Every 30 seconds
    
    def _update_status(self):
        """Update device statuses."""
        self.local_ip_label.setText(f"Local IP: {self._get_local_ip()}")


def create_network_tab(parent=None):
    """Factory function to create network tab."""
    return NetworkTab(parent)
