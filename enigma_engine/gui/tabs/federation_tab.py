"""
================================================================================
FEDERATION TAB - GUI FOR FEDERATED LEARNING
================================================================================

GUI interface for managing federated learning.

FILE: enigma_engine/gui/tabs/federation_tab.py
TYPE: GUI Tab
MAIN CLASS: FederationTab

FEATURES:
    - Create/join federations
    - View federation stats
    - Configure privacy settings
    - Monitor training rounds
    - Manage participants

USAGE:
    tab = FederationTab(parent)
"""

import logging
from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...config import CONFIG
from ...federated import (
    FederatedLearning,
    FederationDiscovery,
    FederationMode,
    FederationRole,
)

logger = logging.getLogger(__name__)


class FederationTab(QWidget):
    """
    GUI for federated learning.
    
    Shows:
    - Current federations (joined)
    - Available federations (can join)
    - Create federation button
    - Federation stats (rounds, participants, improvements)
    - Privacy settings
    """
    
    def __init__(self, parent=None):
        """
        Initialize federation tab.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.parent_window = parent
        
        # Initialize federated learning
        self.federated_learning: Optional[FederatedLearning] = None
        self.discovery: Optional[FederationDiscovery] = None
        
        # Setup UI
        self._setup_ui()
        
        # Check if federated learning is enabled
        if CONFIG.get("federated_learning", {}).get("enabled", False):
            self._initialize_federated_learning()
        
        # Setup auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh_stats)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Header
        header = QLabel("Federated Learning")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(header)
        
        # Description
        desc = QLabel(
            "Share model improvements without sharing data. "
            "Devices collaborate by sharing only weight updates."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a6adc8; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Enable/Disable section
        self._create_enable_section(layout)
        
        # Role selection
        self._create_role_section(layout)
        
        # My Federations section
        self._create_my_federations_section(layout)
        
        # Available Federations section
        self._create_available_federations_section(layout)
        
        # Federation Stats section
        self._create_stats_section(layout)
        
        # Privacy Settings section
        self._create_privacy_section(layout)
        
        layout.addStretch()
    
    def _create_enable_section(self, layout):
        """Create enable/disable section."""
        group = QGroupBox("Status")
        group_layout = QHBoxLayout()
        
        self.status_label = QLabel("Disabled")
        self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
        group_layout.addWidget(QLabel("Federated Learning:"))
        group_layout.addWidget(self.status_label)
        group_layout.addStretch()
        
        self.enable_btn = QPushButton("Enable")
        self.enable_btn.clicked.connect(self._toggle_federated_learning)
        group_layout.addWidget(self.enable_btn)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def _create_role_section(self, layout):
        """Create role selection section."""
        group = QGroupBox("Role")
        group_layout = QHBoxLayout()
        
        group_layout.addWidget(QLabel("I want to:"))
        
        self.role_combo = QComboBox()
        self.role_combo.addItems([
            "Participate (contribute updates)",
            "Coordinate (manage federation)",
            "Observe (receive only)"
        ])
        self.role_combo.currentIndexChanged.connect(self._on_role_changed)
        group_layout.addWidget(self.role_combo, stretch=1)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def _create_my_federations_section(self, layout):
        """Create section for federations user has joined."""
        group = QGroupBox("My Federations")
        group_layout = QVBoxLayout()
        
        # List of joined federations
        self.my_federations_list = QListWidget()
        self.my_federations_list.itemDoubleClicked.connect(self._on_federation_selected)
        group_layout.addWidget(self.my_federations_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        create_btn = QPushButton("Create New")
        create_btn.clicked.connect(self._create_federation)
        btn_layout.addWidget(create_btn)
        
        leave_btn = QPushButton("Leave Selected")
        leave_btn.clicked.connect(self._leave_federation)
        btn_layout.addWidget(leave_btn)
        
        btn_layout.addStretch()
        group_layout.addLayout(btn_layout)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def _create_available_federations_section(self, layout):
        """Create section for available federations."""
        group = QGroupBox("Available Federations")
        group_layout = QVBoxLayout()
        
        # List of available federations
        self.available_federations_list = QListWidget()
        self.available_federations_list.itemDoubleClicked.connect(self._join_selected_federation)
        group_layout.addWidget(self.available_federations_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        discover_btn = QPushButton("Discover")
        discover_btn.clicked.connect(self._discover_federations)
        btn_layout.addWidget(discover_btn)
        
        join_btn = QPushButton("Join Selected")
        join_btn.clicked.connect(self._join_selected_federation)
        btn_layout.addWidget(join_btn)
        
        btn_layout.addStretch()
        group_layout.addLayout(btn_layout)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def _create_stats_section(self, layout):
        """Create federation statistics section."""
        group = QGroupBox("Federation Statistics")
        group_layout = QGridLayout()
        
        # Stats labels
        self.rounds_label = QLabel("0")
        self.participants_label = QLabel("0")
        self.updates_label = QLabel("0")
        self.avg_loss_label = QLabel("N/A")
        
        group_layout.addWidget(QLabel("Completed Rounds:"), 0, 0)
        group_layout.addWidget(self.rounds_label, 0, 1)
        
        group_layout.addWidget(QLabel("Active Participants:"), 1, 0)
        group_layout.addWidget(self.participants_label, 1, 1)
        
        group_layout.addWidget(QLabel("Total Updates:"), 2, 0)
        group_layout.addWidget(self.updates_label, 2, 1)
        
        group_layout.addWidget(QLabel("Avg Loss:"), 3, 0)
        group_layout.addWidget(self.avg_loss_label, 3, 1)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def _create_privacy_section(self, layout):
        """Create privacy settings section."""
        group = QGroupBox("Privacy Settings")
        group_layout = QVBoxLayout()
        
        # Differential privacy
        self.dp_checkbox = QCheckBox("Enable Differential Privacy")
        self.dp_checkbox.setChecked(True)
        self.dp_checkbox.stateChanged.connect(self._on_privacy_changed)
        group_layout.addWidget(self.dp_checkbox)
        
        epsilon_layout = QHBoxLayout()
        epsilon_layout.addWidget(QLabel("Privacy Budget (ε):"))
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.1, 10.0)
        self.epsilon_spin.setValue(1.0)
        self.epsilon_spin.setSingleStep(0.1)
        self.epsilon_spin.valueChanged.connect(self._on_privacy_changed)
        epsilon_layout.addWidget(self.epsilon_spin)
        epsilon_layout.addWidget(QLabel("(lower = more private)"))
        epsilon_layout.addStretch()
        group_layout.addLayout(epsilon_layout)
        
        # Compression
        self.compression_checkbox = QCheckBox("Enable Update Compression")
        self.compression_checkbox.setChecked(True)
        group_layout.addWidget(self.compression_checkbox)
        
        # Quantization
        quant_layout = QHBoxLayout()
        quant_layout.addWidget(QLabel("Quantization Bits:"))
        self.quant_combo = QComboBox()
        self.quant_combo.addItems(["8", "16", "32"])
        self.quant_combo.setCurrentText("8")
        quant_layout.addWidget(self.quant_combo)
        quant_layout.addStretch()
        group_layout.addLayout(quant_layout)
        
        # Sparsity
        sparsity_layout = QHBoxLayout()
        sparsity_layout.addWidget(QLabel("Sparsity (top %):"))
        self.sparsity_spin = QSpinBox()
        self.sparsity_spin.setRange(1, 100)
        self.sparsity_spin.setValue(10)
        sparsity_layout.addWidget(self.sparsity_spin)
        sparsity_layout.addStretch()
        group_layout.addLayout(sparsity_layout)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def _initialize_federated_learning(self):
        """Initialize federated learning system."""
        try:
            # Get role from combo box
            role_text = self.role_combo.currentText()
            if "Coordinate" in role_text:
                role = FederationRole.COORDINATOR
            elif "Observe" in role_text:
                role = FederationRole.OBSERVER
            else:
                role = FederationRole.PARTICIPANT
            
            # Initialize federated learning
            self.federated_learning = FederatedLearning(role=role)
            
            # Initialize discovery
            self.discovery = FederationDiscovery()
            self.discovery.start()
            
            # Update UI
            self.status_label.setText("Enabled")
            self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
            self.enable_btn.setText("Disable")
            
            logger.info("Federated learning initialized")
            
        except Exception as e:
            logger.error(f"Error initializing federated learning: {e}")
            QMessageBox.critical(self, "Error", f"Failed to initialize: {str(e)}")
    
    def _toggle_federated_learning(self):
        """Toggle federated learning on/off."""
        if self.federated_learning:
            # Disable
            if self.discovery:
                self.discovery.stop()
            self.federated_learning = None
            self.discovery = None
            
            self.status_label.setText("Disabled")
            self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
            self.enable_btn.setText("Enable")
            
            # Clear lists
            self.my_federations_list.clear()
            self.available_federations_list.clear()
            
            logger.info("Federated learning disabled")
        else:
            # Enable
            self._initialize_federated_learning()
    
    def _on_role_changed(self):
        """Handle role change."""
        if self.federated_learning:
            # Need to reinitialize with new role
            self._reinitialize_with_new_role()
    
    def _reinitialize_with_new_role(self):
        """Reinitialize federated learning with new role."""
        # Disable current instance
        if self.discovery:
            self.discovery.stop()
        self.federated_learning = None
        self.discovery = None
        
        # Re-enable with new role
        self._initialize_federated_learning()
    
    def _on_privacy_changed(self):
        """Handle privacy setting change."""
        # Update config
        enabled = self.dp_checkbox.isChecked()
        epsilon = self.epsilon_spin.value()
        
        logger.debug(f"Privacy settings changed: enabled={enabled}, epsilon={epsilon}")
    
    def _create_federation(self):
        """Create a new federation."""
        if not self.federated_learning:
            QMessageBox.warning(self, "Not Enabled", "Please enable federated learning first.")
            return
        
        # Simple dialog for federation name
        from PyQt5.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(
            self,
            "Create Federation",
            "Federation Name:"
        )
        
        if not ok or not name:
            return
        
        try:
            # Create federation
            federation_id = self.federated_learning.create_federation(
                name=name,
                mode=FederationMode.PRIVATE,
                description=f"Created from GUI"
            )
            
            # Refresh list
            self._refresh_my_federations()
            
            QMessageBox.information(
                self,
                "Success",
                f"Federation '{name}' created!\nID: {federation_id[:8]}..."
            )
            
        except Exception as e:
            logger.error(f"Error creating federation: {e}")
            QMessageBox.critical(self, "Error", f"Failed to create federation: {str(e)}")
    
    def _leave_federation(self):
        """Leave selected federation."""
        current = self.my_federations_list.currentItem()
        if not current:
            QMessageBox.warning(self, "No Selection", "Please select a federation to leave.")
            return
        
        # Confirm
        reply = QMessageBox.question(
            self,
            "Confirm",
            "Are you sure you want to leave this federation?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Leave federation (placeholder)
            self._refresh_my_federations()
    
    def _discover_federations(self):
        """Discover available federations."""
        if not self.federated_learning or not self.discovery:
            QMessageBox.warning(self, "Not Enabled", "Please enable federated learning first.")
            return
        
        try:
            # Discover federations
            federations = self.discovery.discover_federations(timeout=3.0)
            
            # Update list
            self.available_federations_list.clear()
            for federation in federations:
                item = QListWidgetItem(
                    f"{federation.name} ({federation.participants} participants)"
                )
                item.setData(Qt.UserRole, federation)
                self.available_federations_list.addItem(item)
            
            logger.info(f"Discovered {len(federations)} federations")
            
        except Exception as e:
            logger.error(f"Error discovering federations: {e}")
            QMessageBox.critical(self, "Error", f"Discovery failed: {str(e)}")
    
    def _join_selected_federation(self):
        """Join selected federation."""
        current = self.available_federations_list.currentItem()
        if not current:
            QMessageBox.warning(self, "No Selection", "Please select a federation to join.")
            return
        
        federation = current.data(Qt.UserRole)
        
        try:
            # Join federation
            self.federated_learning.join_federation(federation.id)
            
            # Refresh lists
            self._refresh_my_federations()
            
            QMessageBox.information(
                self,
                "Success",
                f"Joined federation '{federation.name}'"
            )
            
        except Exception as e:
            logger.error(f"Error joining federation: {e}")
            QMessageBox.critical(self, "Error", f"Failed to join: {str(e)}")
    
    def _on_federation_selected(self, item):
        """Handle federation selection."""
        # Show federation details
    
    def _refresh_my_federations(self):
        """Refresh list of joined federations."""
        if not self.federated_learning:
            return
        
        self.my_federations_list.clear()
        
        # Get federations
        federations = self.federated_learning.list_federations()
        
        for federation in federations:
            item = QListWidgetItem(
                f"{federation.name} - Round {federation.current_round}"
            )
            item.setData(Qt.UserRole, federation)
            self.my_federations_list.addItem(item)
    
    def _refresh_stats(self):
        """Refresh federation statistics."""
        if not self.federated_learning:
            return
        
        try:
            # Get coordinator stats if coordinator
            if hasattr(self.federated_learning, 'coordinator'):
                stats = self.federated_learning.coordinator.get_stats()
                
                self.rounds_label.setText(str(stats.get('current_round', 0)))
                self.participants_label.setText(str(stats.get('participants', 0)))
                
                agg_stats = stats.get('aggregator_stats', {})
                self.updates_label.setText(str(agg_stats.get('total_updates', 0)))
        
        except Exception as e:
            logger.debug(f"Error refreshing stats: {e}")


def create_federation_tab(parent):
    """
    Create federation tab.
    
    Args:
        parent: Parent widget
    
    Returns:
        FederationTab instance
    """
    return FederationTab(parent)
