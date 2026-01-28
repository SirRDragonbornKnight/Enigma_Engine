"""
Federated Learning Widget for GUI

Provides user interface for federated learning configuration and monitoring.
Shows participation status, privacy settings, contribution stats, and network info.
"""

import logging
from typing import Optional, Any

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
        QPushButton, QCheckBox, QComboBox, QSpinBox, QTextEdit,
        QFormLayout, QScrollArea, QFrame, QLineEdit, QListWidget,
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QFont
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

logger = logging.getLogger(__name__)


if HAS_PYQT:
    class FederatedLearningWidget(QWidget):
        """
        GUI widget for federated learning settings and monitoring.
        
        Features:
        - Participation control (opt-in/out)
        - Privacy level selection
        - Data filtering configuration
        - Contribution statistics
        - Network status
        """
        
        # Signals
        settings_changed = pyqtSignal(dict)
        
        def __init__(self, parent=None):
            """Initialize the widget."""
            super().__init__(parent)
            
            # Import here to avoid circular dependencies
            from ...learning import (
                FederatedLearning,
                FederatedMode,
                PrivacyLevel,
                FederatedDataFilter,
            )
            
            self.federated_learning: Optional[FederatedLearning] = None
            self.data_filter = FederatedDataFilter()
            
            self._init_ui()
            self._load_settings()
        
        def _init_ui(self):
            """Initialize the user interface."""
            layout = QVBoxLayout()
            layout.setSpacing(10)
            
            # Title
            title = QLabel("Federated Learning")
            title_font = QFont()
            title_font.setPointSize(14)
            title.setFont(title_font)
            layout.addWidget(title)
            
            # Description
            desc = QLabel(
                "Share model improvements without sharing your data. "
                "Only weight updates are shared, never raw conversations."
            )
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #888; font-style: italic;")
            layout.addWidget(desc)
            
            # Participation section
            layout.addWidget(self._create_participation_group())
            
            # Privacy section
            layout.addWidget(self._create_privacy_group())
            
            # Data filtering section
            layout.addWidget(self._create_filtering_group())
            
            # Statistics section
            layout.addWidget(self._create_stats_group())
            
            # Network section
            layout.addWidget(self._create_network_group())
            
            # Stretch at bottom
            layout.addStretch()
            
            # Scroll area
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll_widget = QWidget()
            scroll_widget.setLayout(layout)
            scroll.setWidget(scroll_widget)
            
            main_layout = QVBoxLayout()
            main_layout.addWidget(scroll)
            self.setLayout(main_layout)
        
        def _create_participation_group(self) -> QGroupBox:
            """Create participation control group."""
            group = QGroupBox("Participation")
            layout = QVBoxLayout()
            
            # Enable/disable checkbox
            self.enabled_checkbox = QCheckBox("Enable Federated Learning")
            self.enabled_checkbox.setToolTip(
                "Allow this device to participate in federated learning"
            )
            self.enabled_checkbox.stateChanged.connect(self._on_enabled_changed)
            layout.addWidget(self.enabled_checkbox)
            
            # Status label
            self.status_label = QLabel("Status: Disabled")
            self.status_label.setStyleSheet("font-style: italic; color: #888;")
            layout.addWidget(self.status_label)
            
            # Mode selection
            mode_layout = QHBoxLayout()
            mode_layout.addWidget(QLabel("Mode:"))
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Opt-In", "Opt-Out", "Disabled"])
            self.mode_combo.setToolTip(
                "Opt-In: Must explicitly enable\n"
                "Opt-Out: Enabled by default\n"
                "Disabled: No federated learning"
            )
            self.mode_combo.currentTextChanged.connect(self._on_settings_changed)
            mode_layout.addWidget(self.mode_combo)
            mode_layout.addStretch()
            layout.addLayout(mode_layout)
            
            group.setLayout(layout)
            return group
        
        def _create_privacy_group(self) -> QGroupBox:
            """Create privacy settings group."""
            group = QGroupBox("Privacy Settings")
            layout = QVBoxLayout()
            
            # Privacy level
            level_layout = QHBoxLayout()
            level_layout.addWidget(QLabel("Privacy Level:"))
            self.privacy_combo = QComboBox()
            self.privacy_combo.addItems(["None", "Low", "Medium", "High", "Maximum"])
            self.privacy_combo.setToolTip(
                "None: Share everything\n"
                "Low: Anonymize device ID\n"
                "Medium: + Differential privacy\n"
                "High: + Secure aggregation (recommended)\n"
                "Maximum: + Homomorphic encryption"
            )
            self.privacy_combo.currentTextChanged.connect(self._on_settings_changed)
            level_layout.addWidget(self.privacy_combo)
            level_layout.addStretch()
            layout.addLayout(level_layout)
            
            # Privacy explanation
            self.privacy_explain = QLabel()
            self.privacy_explain.setWordWrap(True)
            self.privacy_explain.setStyleSheet("color: #666; font-size: 10pt;")
            layout.addWidget(self.privacy_explain)
            self._update_privacy_explanation()
            
            # Differential privacy settings
            dp_group = QGroupBox("Differential Privacy")
            dp_layout = QFormLayout()
            
            self.epsilon_spin = QSpinBox()
            self.epsilon_spin.setRange(1, 100)
            self.epsilon_spin.setValue(10)
            self.epsilon_spin.setSuffix(" (×0.1)")
            self.epsilon_spin.setToolTip("Privacy budget (lower = more privacy)")
            dp_layout.addRow("Epsilon:", self.epsilon_spin)
            
            dp_group.setLayout(dp_layout)
            layout.addWidget(dp_group)
            
            group.setLayout(layout)
            return group
        
        def _create_filtering_group(self) -> QGroupBox:
            """Create data filtering group."""
            group = QGroupBox("Data Filtering")
            layout = QVBoxLayout()
            
            # Exclude private chats
            self.exclude_private_check = QCheckBox("Exclude private conversations")
            self.exclude_private_check.setChecked(True)
            self.exclude_private_check.setToolTip("Never share data from private chats")
            layout.addWidget(self.exclude_private_check)
            
            # Sanitize PII
            self.sanitize_pii_check = QCheckBox("Sanitize personal information")
            self.sanitize_pii_check.setChecked(True)
            self.sanitize_pii_check.setToolTip(
                "Remove emails, phone numbers, credit cards, etc."
            )
            layout.addWidget(self.sanitize_pii_check)
            
            # Excluded keywords
            keywords_label = QLabel("Excluded Keywords:")
            layout.addWidget(keywords_label)
            
            self.keywords_list = QListWidget()
            self.keywords_list.setMaximumHeight(100)
            # Add default keywords
            default_keywords = ["password", "credit card", "ssn", "api key"]
            self.keywords_list.addItems(default_keywords)
            layout.addWidget(self.keywords_list)
            
            # Add keyword button
            add_keyword_layout = QHBoxLayout()
            self.keyword_input = QLineEdit()
            self.keyword_input.setPlaceholderText("Enter keyword to exclude...")
            add_keyword_layout.addWidget(self.keyword_input)
            add_keyword_btn = QPushButton("Add")
            add_keyword_btn.clicked.connect(self._add_keyword)
            add_keyword_layout.addWidget(add_keyword_btn)
            layout.addLayout(add_keyword_layout)
            
            group.setLayout(layout)
            return group
        
        def _create_stats_group(self) -> QGroupBox:
            """Create statistics group."""
            group = QGroupBox("Contribution Statistics")
            layout = QFormLayout()
            
            self.updates_sent_label = QLabel("0")
            layout.addRow("Updates Sent:", self.updates_sent_label)
            
            self.updates_received_label = QLabel("0")
            layout.addRow("Updates Received:", self.updates_received_label)
            
            self.samples_contributed_label = QLabel("0")
            layout.addRow("Samples Contributed:", self.samples_contributed_label)
            
            self.rounds_participated_label = QLabel("0")
            layout.addRow("Rounds Participated:", self.rounds_participated_label)
            
            group.setLayout(layout)
            return group
        
        def _create_network_group(self) -> QGroupBox:
            """Create network status group."""
            group = QGroupBox("Network Status")
            layout = QFormLayout()
            
            self.network_mode_label = QLabel("Peer-to-Peer")
            layout.addRow("Mode:", self.network_mode_label)
            
            self.peers_label = QLabel("0")
            layout.addRow("Connected Peers:", self.peers_label)
            
            self.coordinator_label = QLabel("N/A")
            layout.addRow("Coordinator:", self.coordinator_label)
            
            self.current_round_label = QLabel("0")
            layout.addRow("Current Round:", self.current_round_label)
            
            group.setLayout(layout)
            return group
        
        def _on_enabled_changed(self, state):
            """Handle enable/disable change."""
            enabled = state == Qt.Checked
            
            if enabled:
                self.status_label.setText("Status: Enabled")
                self.status_label.setStyleSheet("font-style: italic; color: #22c55e;")
            else:
                self.status_label.setText("Status: Disabled")
                self.status_label.setStyleSheet("font-style: italic; color: #888;")
            
            self._on_settings_changed()
        
        def _on_settings_changed(self):
            """Handle settings change."""
            settings = self.get_settings()
            self.settings_changed.emit(settings)
            self._update_privacy_explanation()
        
        def _update_privacy_explanation(self):
            """Update privacy level explanation."""
            level = self.privacy_combo.currentText().lower()
            
            explanations = {
                "none": "No privacy protection. All data shared as-is.",
                "low": "Device ID is anonymized. Others can't identify your device.",
                "medium": "Device ID anonymized + differential privacy adds noise to protect individual contributions.",
                "high": "Medium privacy + secure aggregation prevents seeing individual updates (recommended).",
                "maximum": "High privacy + homomorphic encryption for maximum protection.",
            }
            
            self.privacy_explain.setText(explanations.get(level, ""))
        
        def _add_keyword(self):
            """Add excluded keyword."""
            keyword = self.keyword_input.text().strip()
            if keyword:
                self.keywords_list.addItem(keyword)
                self.keyword_input.clear()
                self.data_filter.add_excluded_keyword(keyword)
        
        def _load_settings(self):
            """Load settings from config."""
            try:
                from ...config import get_config
                
                fl_config = get_config("federated_learning", {})
                
                # Load participation
                enabled = fl_config.get("enabled", False)
                self.enabled_checkbox.setChecked(enabled)
                
                # Load privacy level
                privacy_level = fl_config.get("privacy_level", "high")
                index = self.privacy_combo.findText(privacy_level.title())
                if index >= 0:
                    self.privacy_combo.setCurrentIndex(index)
                
            except Exception as e:
                logger.warning(f"Could not load federated learning settings: {e}")
        
        def get_settings(self) -> dict:
            """Get current settings as dictionary."""
            return {
                "enabled": self.enabled_checkbox.isChecked(),
                "mode": self.mode_combo.currentText().lower().replace("-", "_"),
                "privacy_level": self.privacy_combo.currentText().lower(),
                "epsilon": self.epsilon_spin.value() / 10.0,
                "exclude_private": self.exclude_private_check.isChecked(),
                "sanitize_pii": self.sanitize_pii_check.isChecked(),
                "excluded_keywords": [
                    self.keywords_list.item(i).text()
                    for i in range(self.keywords_list.count())
                ],
            }
        
        def update_stats(self, stats: dict):
            """Update statistics display."""
            self.updates_sent_label.setText(str(stats.get("updates_sent", 0)))
            self.updates_received_label.setText(str(stats.get("updates_received", 0)))
            self.samples_contributed_label.setText(str(stats.get("total_samples_contributed", 0)))
            self.rounds_participated_label.setText(str(stats.get("rounds_participated", 0)))
        
        def update_network_status(self, status: dict):
            """Update network status display."""
            self.network_mode_label.setText(status.get("mode", "N/A"))
            self.peers_label.setText(str(status.get("peers", 0)))
            self.coordinator_label.setText(status.get("coordinator", "N/A"))
            self.current_round_label.setText(str(status.get("current_round", 0)))

else:
    # Fallback if PyQt5 not available
    class FederatedLearningWidget:
        """Stub class when PyQt5 is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt5 is required for FederatedLearningWidget")
