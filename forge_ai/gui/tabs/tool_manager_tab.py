"""
Tool Manager Tab - GUI for enabling/disabling tools.

Allows users to:
  - Enable/disable individual tools
  - Apply presets (minimal, standard, full, camera_only, etc.)
  - See tool dependencies
  - Save/load custom profiles
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QGroupBox,
    QMessageBox, QInputDialog, QSplitter, QTextEdit,
    QCheckBox, QProgressBar, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor

from .shared_components import NoScrollComboBox
from forge_ai.tools.tool_manager import (
    ToolManager, get_tool_manager, TOOL_CATEGORIES, 
    TOOL_DEPENDENCIES, PRESETS
)


class ToolManagerTab(QWidget):
    """GUI for managing which tools are enabled/disabled."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = get_tool_manager()
        self._setup_ui()
        self._load_tools()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        # Header
        header = QLabel("Tool Manager")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header)
        
        desc = QLabel("Enable/disable AI tools to customize your Forge installation. "
                     "Disable unused tools to save memory and speed up responses.")
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; color: #bac2de;")
        layout.addWidget(desc)
        
        # Stats bar
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("color: #bac2de; font-style: italic; font-size: 12px;")
        layout.addWidget(self.stats_label)
        
        # Preset selector - more compact
        preset_group = QGroupBox("Quick Presets")
        preset_layout = QHBoxLayout(preset_group)
        preset_layout.setContentsMargins(6, 6, 6, 6)
        
        self.preset_combo = NoScrollComboBox()
        self.preset_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.preset_combo.setToolTip("Select a preset configuration for quick setup")
        for name, info in PRESETS.items():
            self.preset_combo.addItem(f"{name} - {info['description'][:30]}", name)
        preset_layout.addWidget(self.preset_combo)
        
        apply_preset_btn = QPushButton("Apply")
        apply_preset_btn.setFixedWidth(60)
        apply_preset_btn.clicked.connect(self._apply_preset)
        preset_layout.addWidget(apply_preset_btn)
        
        layout.addWidget(preset_group)
        
        # Main splitter - allows user to resize panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Tool tree
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        
        tree_label = QLabel("Tools by Category:")
        tree_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        left_layout.addWidget(tree_label)
        
        self.tool_tree = QTreeWidget()
        self.tool_tree.setHeaderLabels(["Tool", "Status"])
        self.tool_tree.setColumnWidth(0, 180)
        self.tool_tree.header().setStretchLastSection(True)
        self.tool_tree.setMinimumWidth(200)
        self.tool_tree.itemChanged.connect(self._on_item_changed)
        self.tool_tree.currentItemChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.tool_tree)
        
        # Bulk actions
        bulk_layout = QHBoxLayout()
        bulk_layout.setSpacing(4)
        
        enable_all_btn = QPushButton("Enable All")
        enable_all_btn.setFixedHeight(26)
        enable_all_btn.setToolTip("Enable all tools (may need extra packages)")
        enable_all_btn.clicked.connect(self._enable_all)
        bulk_layout.addWidget(enable_all_btn)
        
        disable_all_btn = QPushButton("Disable All")
        disable_all_btn.setFixedHeight(26)
        disable_all_btn.setToolTip("Disable all tools (training-only mode)")
        disable_all_btn.clicked.connect(self._disable_all)
        bulk_layout.addWidget(disable_all_btn)
        
        left_layout.addLayout(bulk_layout)
        
        splitter.addWidget(left_widget)
        
        # Right: Tool info - collapsible
        right_widget = QWidget()
        right_widget.setMinimumWidth(150)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(4)
        
        info_label = QLabel("Tool Information:")
        info_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        right_layout.addWidget(info_label)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(80)
        self.info_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.info_text.setStyleSheet("font-size: 12px;")
        right_layout.addWidget(self.info_text, 1)  # stretch factor 1
        
        # Dependencies section
        deps_label = QLabel("Required Packages:")
        deps_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        right_layout.addWidget(deps_label)
        
        self.deps_text = QTextEdit()
        self.deps_text.setReadOnly(True)
        self.deps_text.setStyleSheet("font-size: 12px;")
        self.deps_text.setMinimumHeight(60)
        self.deps_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.deps_text, 1)  # stretch factor 1
        
        # Profile management - compact
        profile_group = QGroupBox("Profiles")
        profile_layout = QHBoxLayout(profile_group)
        profile_layout.setContentsMargins(6, 6, 6, 6)
        
        save_profile_btn = QPushButton("Save")
        save_profile_btn.clicked.connect(self._save_profile)
        profile_layout.addWidget(save_profile_btn)
        
        load_profile_btn = QPushButton("Load")
        load_profile_btn.clicked.connect(self._load_profile)
        profile_layout.addWidget(load_profile_btn)
        
        right_layout.addWidget(profile_group)
        
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)  # Left panel (tools) gets more space
        splitter.setStretchFactor(1, 2)  # Right panel (info) gets less
        splitter.setSizes([350, 250])
        
        # Add splitter with stretch factor so it expands to fill available space
        layout.addWidget(splitter, 1)
        
        # Bottom buttons - compact
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(6)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_tools)
        bottom_layout.addWidget(refresh_btn)
        
        bottom_layout.addStretch()
        
        install_deps_btn = QPushButton("Install Dependencies")
        install_deps_btn.clicked.connect(self._install_dependencies)
        bottom_layout.addWidget(install_deps_btn)
        
        layout.addLayout(bottom_layout)
    
    def _load_tools(self):
        """Load tools into the tree."""
        self.tool_tree.blockSignals(True)
        self.tool_tree.clear()
        
        enabled_count = 0
        total_count = 0
        
        for category, tools in TOOL_CATEGORIES.items():
            # Create category item
            cat_item = QTreeWidgetItem([category.upper(), ""])
            cat_item.setFlags(cat_item.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
            
            cat_enabled = 0
            for tool_name in tools:
                total_count += 1
                tool_item = QTreeWidgetItem([tool_name, ""])
                tool_item.setFlags(tool_item.flags() | Qt.ItemIsUserCheckable)
                tool_item.setData(0, Qt.UserRole, tool_name)
                
                if self.manager.is_enabled(tool_name):
                    tool_item.setCheckState(0, Qt.Checked)
                    tool_item.setText(1, "Enabled")
                    cat_enabled += 1
                    enabled_count += 1
                else:
                    tool_item.setCheckState(0, Qt.Unchecked)
                    tool_item.setText(1, "Disabled")
                
                # Color based on dependencies
                deps = TOOL_DEPENDENCIES.get(tool_name, [])
                if deps:
                    tool_item.setToolTip(0, f"Requires: {', '.join(deps)}")
                
                cat_item.addChild(tool_item)
            
            # Update category status
            cat_item.setText(1, f"{cat_enabled}/{len(tools)} enabled")
            self.tool_tree.addTopLevelItem(cat_item)
        
        self.tool_tree.expandAll()
        self.tool_tree.blockSignals(False)
        
        # Update stats
        self.stats_label.setText(
            f"Profile: {self.manager.current_profile} | "
            f"Enabled: {enabled_count}/{total_count} tools"
        )
        
        # Update dependencies display
        self._update_dependencies()
    
    def _on_item_changed(self, item, column):
        """Handle checkbox changes."""
        tool_name = item.data(0, Qt.UserRole)
        if not tool_name:
            return  # Category item
        
        if item.checkState(0) == Qt.Checked:
            self.manager.enable_tool(tool_name)
            item.setText(1, "Enabled")
        else:
            self.manager.disable_tool(tool_name)
            item.setText(1, "Disabled")
        
        self._update_stats()
        self._update_dependencies()
    
    def _on_selection_changed(self, current, previous):
        """Show info for selected tool."""
        if not current:
            return
        
        tool_name = current.data(0, Qt.UserRole)
        if not tool_name:
            # Category selected
            self.info_text.setHtml("<i>Select a tool to see details</i>")
            return
        
        info = self.manager.get_tool_info(tool_name)
        deps = TOOL_DEPENDENCIES.get(tool_name, [])
        
        html = f"""
        <h3>{tool_name}</h3>
        <p><b>Category:</b> {info.get('category', 'Unknown')}</p>
        <p><b>Status:</b> {'Enabled' if info.get('enabled') else 'Disabled'}</p>
        <p><b>Dependencies:</b> {', '.join(deps) if deps else 'None (built-in)'}</p>
        """
        
        self.info_text.setHtml(html)
    
    def _update_stats(self):
        """Update the stats label."""
        stats = self.manager.get_stats()
        self.stats_label.setText(
            f"Profile: {stats['profile']} | "
            f"Enabled: {stats['enabled']}/{stats['total_tools']} tools"
        )
    
    def _update_dependencies(self):
        """Update the dependencies display."""
        deps_result = self.manager.get_dependencies()
        
        if deps_result['total_packages'] == 0:
            self.deps_text.setPlainText("No additional packages required.")
        else:
            text = f"Required packages ({deps_result['total_packages']}):\n\n"
            text += "pip install " + " ".join(deps_result['packages'])
            self.deps_text.setPlainText(text)
    
    def _apply_preset(self):
        """Apply selected preset."""
        preset_name = self.preset_combo.currentData()
        
        reply = QMessageBox.question(
            self, "Apply Preset",
            f"Apply the '{preset_name}' preset?\n\n"
            f"{PRESETS[preset_name]['description']}\n\n"
            "This will change which tools are enabled.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            result = self.manager.apply_preset(preset_name)
            if result['success']:
                self._load_tools()
                QMessageBox.information(
                    self, "Success",
                    f"Applied '{preset_name}' preset.\n"
                    f"Enabled: {result['enabled_count']} tools\n"
                    f"Disabled: {result['disabled_count']} tools"
                )
    
    def _enable_all(self):
        """Enable all tools."""
        reply = QMessageBox.question(
            self, "Enable All",
            "Enable ALL tools?\n\nThis may require additional packages.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.manager.apply_preset("full")
            self._load_tools()
    
    def _disable_all(self):
        """Disable all tools."""
        reply = QMessageBox.question(
            self, "Disable All",
            "Disable ALL tools?\n\nThis will leave only the training mode available.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.manager.apply_preset("training_only")
            self._load_tools()
    
    def _save_profile(self):
        """Save current config as a profile."""
        name, ok = QInputDialog.getText(
            self, "Save Profile",
            "Enter profile name:"
        )
        
        if ok and name:
            result = self.manager.save_profile(name)
            if result['success']:
                QMessageBox.information(
                    self, "Saved",
                    f"Profile '{name}' saved with {result['tool_count']} tools."
                )
    
    def _load_profile(self):
        """Load a saved profile."""
        profiles = self.manager.list_profiles()
        
        all_profiles = profiles['builtin_presets'] + [
            p['name'] for p in profiles['custom_profiles']
        ]
        
        name, ok = QInputDialog.getItem(
            self, "Load Profile",
            "Select profile:",
            all_profiles,
            0, False
        )
        
        if ok and name:
            result = self.manager.load_profile(name)
            if result['success']:
                self._load_tools()
                QMessageBox.information(
                    self, "Loaded",
                    f"Loaded '{name}' profile with {result['enabled_count']} tools."
                )
    
    def _install_dependencies(self):
        """Show pip install command for dependencies."""
        deps_result = self.manager.get_dependencies()
        
        if deps_result['total_packages'] == 0:
            QMessageBox.information(
                self, "Dependencies",
                "No additional packages required for enabled tools."
            )
            return
        
        packages = deps_result['packages']
        cmd = f"pip install {' '.join(packages)}"
        
        reply = QMessageBox.question(
            self, "Install Dependencies",
            f"Install {len(packages)} required packages?\n\n"
            f"Command:\n{cmd}\n\n"
            "This will run in a terminal.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Copy to clipboard
            try:
                from PyQt5.QtWidgets import QApplication
                QApplication.clipboard().setText(cmd)
                QMessageBox.information(
                    self, "Copied",
                    "Install command copied to clipboard!\n\n"
                    "Paste and run in your terminal."
                )
            except:
                QMessageBox.information(
                    self, "Command",
                    f"Run this command in your terminal:\n\n{cmd}"
                )
