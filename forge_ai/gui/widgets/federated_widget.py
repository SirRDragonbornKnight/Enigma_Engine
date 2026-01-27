"""
Federated Learning Widget - GUI Controls for Federated Learning Settings

Provides UI for configuring privacy-preserving distributed learning.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QCheckBox, QComboBox, QSpinBox, QSlider,
    QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer


class FederatedWidget(QWidget):
    """Widget for federated learning settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Federated Learning Settings")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Enable privacy-preserving distributed learning where devices "
            "share model improvements without sharing raw data."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Enable/Disable
        self.enable_check = QCheckBox("Enable Federated Learning")
        self.enable_check.stateChanged.connect(self._on_enable_changed)
        layout.addWidget(self.enable_check)
        
        # Settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        
        # Participation mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Participation:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Opt-In (Must Enable)", "Opt-Out (Default On)", "Disabled"])
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        settings_layout.addLayout(mode_layout)
        
        # Privacy level
        privacy_layout = QHBoxLayout()
        privacy_layout.addWidget(QLabel("Privacy Level:"))
        self.privacy_combo = QComboBox()
        self.privacy_combo.addItems(["None", "Low", "Medium", "High", "Maximum"])
        self.privacy_combo.setCurrentText("High")
        privacy_layout.addWidget(self.privacy_combo)
        privacy_layout.addStretch()
        settings_layout.addLayout(privacy_layout)
        
        # Minimum devices
        min_devices_layout = QHBoxLayout()
        min_devices_layout.addWidget(QLabel("Min Devices:"))
        self.min_devices_spin = QSpinBox()
        self.min_devices_spin.setMinimum(2)
        self.min_devices_spin.setMaximum(100)
        self.min_devices_spin.setValue(2)
        min_devices_layout.addWidget(self.min_devices_spin)
        min_devices_layout.addStretch()
        settings_layout.addLayout(min_devices_layout)
        
        # Round duration
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Round Duration (min):"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setMinimum(1)
        self.duration_spin.setMaximum(60)
        self.duration_spin.setValue(5)
        duration_layout.addWidget(self.duration_spin)
        duration_layout.addStretch()
        settings_layout.addLayout(duration_layout)
        
        # Data filtering
        self.filter_pii_check = QCheckBox("Remove Personally Identifiable Information")
        self.filter_pii_check.setChecked(True)
        settings_layout.addWidget(self.filter_pii_check)
        
        self.filter_inappropriate_check = QCheckBox("Filter Inappropriate Content")
        self.filter_inappropriate_check.setChecked(True)
        settings_layout.addWidget(self.filter_inappropriate_check)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Status group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Federated learning is disabled")
        self.status_label.setStyleSheet("color: #888;")
        status_layout.addWidget(self.status_label)
        
        self.peers_label = QLabel("Connected peers: 0")
        status_layout.addWidget(self.peers_label)
        
        self.round_label = QLabel("Current round: N/A")
        status_layout.addWidget(self.round_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.discover_btn = QPushButton("Discover Peers")
        self.discover_btn.clicked.connect(self._on_discover_peers)
        self.discover_btn.setEnabled(False)
        button_layout.addWidget(self.discover_btn)
        
        self.apply_btn = QPushButton("Apply Settings")
        self.apply_btn.clicked.connect(self._on_apply_settings)
        button_layout.addWidget(self.apply_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Disable settings initially
        self._toggle_settings(False)
    
    def _on_enable_changed(self, state):
        """Handle enable checkbox change."""
        enabled = state == Qt.CheckState.Checked.value
        self._toggle_settings(enabled)
        
        if enabled:
            self.status_label.setText("Federated learning is enabled")
            self.status_label.setStyleSheet("color: #22c55e;")
        else:
            self.status_label.setText("Federated learning is disabled")
            self.status_label.setStyleSheet("color: #888;")
    
    def _toggle_settings(self, enabled: bool):
        """Enable/disable settings based on enabled state."""
        self.mode_combo.setEnabled(enabled)
        self.privacy_combo.setEnabled(enabled)
        self.min_devices_spin.setEnabled(enabled)
        self.duration_spin.setEnabled(enabled)
        self.filter_pii_check.setEnabled(enabled)
        self.filter_inappropriate_check.setEnabled(enabled)
        self.discover_btn.setEnabled(enabled)
    
    def _on_discover_peers(self):
        """Discover federated learning peers."""
        try:
            from ...comms.discovery import discover_federated_learning_peers
            
            self.status_label.setText("Discovering peers...")
            self.status_label.setStyleSheet("color: #3b82f6;")
            
            # Discover peers
            peers = discover_federated_learning_peers(timeout=3.0)
            
            if peers:
                self.peers_label.setText(f"Connected peers: {len(peers)}")
                self.status_label.setText(f"Found {len(peers)} peer(s)")
                self.status_label.setStyleSheet("color: #22c55e;")
            else:
                self.peers_label.setText("Connected peers: 0")
                self.status_label.setText("No peers found")
                self.status_label.setStyleSheet("color: #f59e0b;")
                
        except Exception as e:
            self.status_label.setText(f"Discovery failed: {str(e)}")
            self.status_label.setStyleSheet("color: #ef4444;")
    
    def _on_apply_settings(self):
        """Apply federated learning settings."""
        try:
            from ...config import CONFIG, update_config
            
            # Map UI values to config
            mode_map = {
                "Opt-In (Must Enable)": "opt_in",
                "Opt-Out (Default On)": "opt_out",
                "Disabled": "disabled",
            }
            
            privacy_map = {
                "None": "none",
                "Low": "low",
                "Medium": "medium",
                "High": "high",
                "Maximum": "maximum",
            }
            
            # Update config
            federated_config = {
                "mode": mode_map[self.mode_combo.currentText()],
                "privacy_level": privacy_map[self.privacy_combo.currentText()],
                "min_devices": self.min_devices_spin.value(),
                "round_duration": self.duration_spin.value() * 60,  # Convert to seconds
                "remove_pii": self.filter_pii_check.isChecked(),
                "remove_inappropriate": self.filter_inappropriate_check.isChecked(),
            }
            
            update_config({"federated": {**CONFIG.get("federated", {}), **federated_config}})
            
            self.status_label.setText("Settings applied successfully")
            self.status_label.setStyleSheet("color: #22c55e;")
            
            # Reset after 3 seconds
            QTimer.singleShot(3000, lambda: self._reset_status_label())
            
        except Exception as e:
            self.status_label.setText(f"Failed to apply: {str(e)}")
            self.status_label.setStyleSheet("color: #ef4444;")
    
    def _reset_status_label(self):
        """Reset status label to default."""
        if self.enable_check.isChecked():
            self.status_label.setText("Federated learning is enabled")
            self.status_label.setStyleSheet("color: #22c55e;")
        else:
            self.status_label.setText("Federated learning is disabled")
            self.status_label.setStyleSheet("color: #888;")
    
    def get_settings(self):
        """Get current settings."""
        return {
            "enabled": self.enable_check.isChecked(),
            "mode": self.mode_combo.currentText(),
            "privacy_level": self.privacy_combo.currentText(),
            "min_devices": self.min_devices_spin.value(),
            "round_duration": self.duration_spin.value(),
            "filter_pii": self.filter_pii_check.isChecked(),
            "filter_inappropriate": self.filter_inappropriate_check.isChecked(),
        }
