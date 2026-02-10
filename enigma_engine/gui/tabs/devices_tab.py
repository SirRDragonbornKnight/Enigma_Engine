"""
Devices Tab - Manage network devices and offloading

Shows connected devices, their capabilities, and allows configuration
of task offloading settings.
"""

import logging
from typing import Any, Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class DeviceItem(QWidget):
    """Widget representing a single device."""
    
    clicked = pyqtSignal(str)  # device_id
    
    def __init__(self, device_info: dict[str, Any], parent=None):
        super().__init__(parent)
        self.device_id = device_info.get("id", "unknown")
        self.device_info = device_info
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        
        # Status indicator
        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(12, 12)
        self._update_status_indicator()
        layout.addWidget(self.status_indicator)
        
        # Device info
        info_layout = QVBoxLayout()
        
        name = self.device_info.get("name", "Unknown Device")
        self.name_label = QLabel(name)
        self.name_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        info_layout.addWidget(self.name_label)
        
        details = []
        if self.device_info.get("gpu"):
            details.append("GPU")
        if self.device_info.get("ram_gb"):
            details.append(f"{self.device_info['ram_gb']}GB RAM")
        if self.device_info.get("role"):
            details.append(self.device_info["role"])
        
        self.details_label = QLabel(" | ".join(details) if details else "No details")
        self.details_label.setStyleSheet("color: #bac2de;")
        info_layout.addWidget(self.details_label)
        
        layout.addLayout(info_layout, 1)
        
        # Latency
        latency = self.device_info.get("latency_ms", 0)
        self.latency_label = QLabel(f"{latency:.0f}ms")
        self.latency_label.setStyleSheet("color: #bac2de; font-size: 9pt;")
        layout.addWidget(self.latency_label)
        
        self.setCursor(Qt.PointingHandCursor)
    
    def _update_status_indicator(self):
        health = self.device_info.get("health", "unknown")
        colors = {
            "healthy": "#4CAF50",
            "degraded": "#FF9800",
            "unhealthy": "#F44336",
            "unknown": "#888888",
        }
        color = colors.get(health, "#888888")
        self.status_indicator.setStyleSheet(
            f"background-color: {color}; border-radius: 6px;"
        )
    
    def mousePressEvent(self, event):
        self.clicked.emit(self.device_id)
        super().mousePressEvent(event)


class DevicesTab(QWidget):
    """
    Device management tab.
    
    Features:
    - List of discovered devices
    - Device capabilities display
    - Offload mode configuration
    - Manual device addition
    - Health monitoring
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._devices: dict[str, dict] = {}
        self._selected_device: Optional[str] = None
        self._setup_ui()
        self._setup_refresh_timer()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        
        # Left panel - Device list
        left_panel = QVBoxLayout()
        
        # Header
        header = QHBoxLayout()
        title = QLabel("Network Devices")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        header.addWidget(title)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        header.addWidget(self.refresh_btn)
        
        left_panel.addLayout(header)
        
        # Device list
        self.device_list = QListWidget()
        self.device_list.setMinimumWidth(280)
        self.device_list.currentItemChanged.connect(self._on_device_selected)
        left_panel.addWidget(self.device_list)
        
        # Add device button
        add_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add Device")
        self.add_btn.clicked.connect(self._add_device_dialog)
        add_layout.addWidget(self.add_btn)
        
        self.scan_btn = QPushButton("Scan Network")
        self.scan_btn.clicked.connect(self._scan_network)
        add_layout.addWidget(self.scan_btn)
        
        left_panel.addLayout(add_layout)
        
        layout.addLayout(left_panel)
        
        # Right panel - Device details and settings
        right_panel = QVBoxLayout()
        
        # Device details
        details_group = QGroupBox("Device Details")
        details_layout = QFormLayout()
        
        self.detail_name = QLabel("-")
        details_layout.addRow("Name:", self.detail_name)
        
        self.detail_address = QLabel("-")
        details_layout.addRow("Address:", self.detail_address)
        
        self.detail_status = QLabel("-")
        details_layout.addRow("Status:", self.detail_status)
        
        self.detail_gpu = QLabel("-")
        details_layout.addRow("GPU:", self.detail_gpu)
        
        self.detail_ram = QLabel("-")
        details_layout.addRow("RAM:", self.detail_ram)
        
        self.detail_models = QLabel("-")
        details_layout.addRow("Loaded Models:", self.detail_models)
        
        details_group.setLayout(details_layout)
        right_panel.addWidget(details_group)
        
        # Offload settings
        offload_group = QGroupBox("Offload Settings")
        offload_layout = QFormLayout()
        
        self.offload_mode = QComboBox()
        self.offload_mode.addItems([
            "Auto (Recommended)",
            "Local Only",
            "Remote Only",
            "Prefer Local",
            "Prefer Remote"
        ])
        self.offload_mode.currentIndexChanged.connect(self._on_offload_mode_changed)
        offload_layout.addRow("Mode:", self.offload_mode)
        
        self.auto_failover = QCheckBox("Enable automatic failover")
        self.auto_failover.setChecked(True)
        offload_layout.addRow("", self.auto_failover)
        
        self.primary_device = QComboBox()
        self.primary_device.addItem("(Auto-select)")
        offload_layout.addRow("Primary Device:", self.primary_device)
        
        offload_group.setLayout(offload_layout)
        right_panel.addWidget(offload_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout()
        
        stats_layout.addWidget(QLabel("Local Requests:"), 0, 0)
        self.stat_local = QLabel("0")
        stats_layout.addWidget(self.stat_local, 0, 1)
        
        stats_layout.addWidget(QLabel("Remote Requests:"), 1, 0)
        self.stat_remote = QLabel("0")
        stats_layout.addWidget(self.stat_remote, 1, 1)
        
        stats_layout.addWidget(QLabel("Failovers:"), 2, 0)
        self.stat_failover = QLabel("0")
        stats_layout.addWidget(self.stat_failover, 2, 1)
        
        stats_layout.addWidget(QLabel("Errors:"), 3, 0)
        self.stat_errors = QLabel("0")
        stats_layout.addWidget(self.stat_errors, 3, 1)
        
        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)
        
        right_panel.addStretch()
        
        # Action buttons
        actions_layout = QHBoxLayout()
        
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self._test_selected_device)
        self.test_btn.setEnabled(False)
        actions_layout.addWidget(self.test_btn)
        
        self.remove_btn = QPushButton("Remove Device")
        self.remove_btn.clicked.connect(self._remove_selected_device)
        self.remove_btn.setEnabled(False)
        actions_layout.addWidget(self.remove_btn)
        
        right_panel.addLayout(actions_layout)
        
        layout.addLayout(right_panel, 1)
    
    def _setup_refresh_timer(self):
        """Setup timer for periodic refresh."""
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_devices)
        self.refresh_timer.start(10000)  # Refresh every 10 seconds
    
    def _refresh_devices(self):
        """Refresh device list."""
        try:
            from ..comms.discovery import DeviceDiscovery
            from ..network import get_inference_gateway

            # Get discovered devices
            discovery = DeviceDiscovery("Enigma AI Engine", 5000)
            devices = discovery.discovered
            
            # Also get devices from inference gateway
            gateway = get_inference_gateway()
            stats = gateway.get_stats()
            
            # Update statistics
            self.stat_local.setText(str(stats.get("local_requests", 0)))
            self.stat_remote.setText(str(stats.get("remote_requests", 0)))
            self.stat_failover.setText(str(stats.get("fallbacks", 0)))
            self.stat_errors.setText(str(stats.get("errors", 0)))
            
            # Update device list
            self._update_device_list(devices)
            
        except ImportError as e:
            logger.debug(f"Discovery not available: {e}")
        except Exception as e:
            logger.error(f"Failed to refresh devices: {e}")
    
    def _update_device_list(self, devices: dict[str, dict]):
        """Update the device list widget."""
        self._devices = devices
        current = self.device_list.currentItem()
        current_id = current.data(Qt.UserRole) if current else None
        
        self.device_list.clear()
        self.primary_device.clear()
        self.primary_device.addItem("(Auto-select)")
        
        for device_id, info in devices.items():
            item = QListWidgetItem()
            item.setData(Qt.UserRole, device_id)
            
            widget = DeviceItem({"id": device_id, **info})
            widget.clicked.connect(lambda d: self._select_device(d))
            
            item.setSizeHint(widget.sizeHint())
            self.device_list.addItem(item)
            self.device_list.setItemWidget(item, widget)
            
            # Add to primary device dropdown
            name = info.get("name", device_id)
            self.primary_device.addItem(name, device_id)
        
        # Restore selection
        if current_id:
            for i in range(self.device_list.count()):
                item = self.device_list.item(i)
                if item.data(Qt.UserRole) == current_id:
                    self.device_list.setCurrentItem(item)
                    break
    
    def _on_device_selected(self, current, previous):
        """Handle device selection."""
        if current is None:
            self._clear_details()
            return
        
        device_id = current.data(Qt.UserRole)
        self._select_device(device_id)
    
    def _select_device(self, device_id: str):
        """Select and show details for a device."""
        self._selected_device = device_id
        info = self._devices.get(device_id, {})
        
        # Update details
        self.detail_name.setText(info.get("name", device_id))
        self.detail_address.setText(f"{info.get('ip', '?')}:{info.get('port', '?')}")
        
        health = info.get("health", "unknown")
        self.detail_status.setText(health.capitalize())
        
        gpu = info.get("gpu", False)
        self.detail_gpu.setText("Yes" if gpu else "No")
        
        ram = info.get("ram_gb", 0)
        self.detail_ram.setText(f"{ram} GB" if ram else "Unknown")
        
        models = info.get("models", [])
        self.detail_models.setText(", ".join(models) if models else "None")
        
        # Enable buttons
        self.test_btn.setEnabled(True)
        self.remove_btn.setEnabled(True)
    
    def _clear_details(self):
        """Clear device details."""
        self._selected_device = None
        self.detail_name.setText("-")
        self.detail_address.setText("-")
        self.detail_status.setText("-")
        self.detail_gpu.setText("-")
        self.detail_ram.setText("-")
        self.detail_models.setText("-")
        self.test_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)
    
    def _on_offload_mode_changed(self, index: int):
        """Handle offload mode change."""
        modes = ["auto", "local", "remote", "prefer_local", "prefer_remote"]
        mode = modes[index] if index < len(modes) else "auto"
        
        try:
            from ..network import InferenceMode, get_inference_gateway
            gateway = get_inference_gateway()
            gateway.mode = InferenceMode(mode)
            logger.info(f"Offload mode changed to: {mode}")
        except Exception as e:
            logger.error(f"Failed to change offload mode: {e}")
    
    def _add_device_dialog(self):
        """Show dialog to add device manually."""
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Device")
        dialog.setMinimumWidth(300)
        
        layout = QFormLayout(dialog)
        
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("My PC")
        layout.addRow("Name:", name_edit)
        
        address_edit = QLineEdit()
        address_edit.setPlaceholderText("192.168.1.100")
        layout.addRow("IP Address:", address_edit)
        
        port_spin = QSpinBox()
        port_spin.setRange(1, 65535)
        port_spin.setValue(5000)
        layout.addRow("Port:", port_spin)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            name = name_edit.text() or "Unknown"
            address = address_edit.text()
            port = port_spin.value()
            
            if address:
                self._add_device(name, address, port)
    
    def _add_device(self, name: str, address: str, port: int):
        """Add a device to the pool."""
        try:
            from ..network import get_inference_gateway
            
            gateway = get_inference_gateway()
            gateway.add_server(address, port)
            
            # Add to local list
            device_id = f"{address}:{port}"
            self._devices[device_id] = {
                "name": name,
                "ip": address,
                "port": port,
                "health": "unknown",
            }
            self._update_device_list(self._devices)
            
            logger.info(f"Added device: {name} at {address}:{port}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to add device: {e}")
    
    def _scan_network(self):
        """Scan network for devices."""
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning...")
        
        try:
            from ..comms.discovery import DeviceDiscovery
            
            discovery = DeviceDiscovery("Enigma AI Engine", 5000)
            discovery.broadcast_discovery()
            
            # Wait a bit for responses
            QTimer.singleShot(3000, self._on_scan_complete)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to scan network: {e}")
            self._on_scan_complete()
    
    def _on_scan_complete(self):
        """Handle scan completion."""
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Scan Network")
        self._refresh_devices()
    
    def _test_selected_device(self):
        """Test connection to selected device (runs in background)."""
        if not self._selected_device:
            return
        
        info = self._devices.get(self._selected_device, {})
        address = info.get("ip")
        port = info.get("port", 5000)
        
        if not address:
            QMessageBox.warning(self, "Error", "No address for device")
            return
        
        self.test_btn.setEnabled(False)
        self.test_btn.setText("Testing...")
        
        # Run network test in background thread
        import threading
        def do_test():
            try:
                from ..comms.remote_client import RemoteClient
                
                client = RemoteClient(address, port)
                result = client.health_check()
                
                from PyQt5.QtCore import QTimer
                def show_result():
                    self.test_btn.setEnabled(True)
                    self.test_btn.setText("Test Connection")
                    if result:
                        QMessageBox.information(
                            self, "Success",
                            f"Connection to {address}:{port} successful!"
                        )
                    else:
                        QMessageBox.warning(
                            self, "Failed",
                            f"Connection to {address}:{port} failed"
                        )
                QTimer.singleShot(0, show_result)
                    
            except Exception as e:
                from PyQt5.QtCore import QTimer
                def show_error():
                    self.test_btn.setEnabled(True)
                    self.test_btn.setText("Test Connection")
                    QMessageBox.warning(self, "Error", f"Test failed: {e}")
                QTimer.singleShot(0, show_error)
        
        thread = threading.Thread(target=do_test, daemon=True)
        thread.start()
    
    def _remove_selected_device(self):
        """Remove selected device."""
        if not self._selected_device:
            return
        
        info = self._devices.get(self._selected_device, {})
        address = info.get("ip")
        port = info.get("port", 5000)
        
        try:
            from ..network import get_inference_gateway
            
            gateway = get_inference_gateway()
            gateway.remove_server(address, port)
            
            del self._devices[self._selected_device]
            self._update_device_list(self._devices)
            self._clear_details()
            
            logger.info(f"Removed device: {self._selected_device}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to remove device: {e}")
