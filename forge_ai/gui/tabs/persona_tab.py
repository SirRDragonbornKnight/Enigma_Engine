"""
Persona Management Tab - Create, Copy, Export, Import AI Personas

This tab allows users to:
- View and manage their AI personas
- Copy personas to create variants
- Export personas to share with others
- Import personas from files
- Edit persona details
- Switch between personas

Usage:
    from forge_ai.gui.tabs.persona_tab import create_persona_tab
    
    persona_widget = create_persona_tab(parent_window)
    tabs.addTab(persona_widget, "Personas")
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget,
    QTextEdit, QLineEdit, QMessageBox, QFileDialog, QGroupBox, QFormLayout,
    QComboBox, QListWidgetItem, QDialog, QDialogButtonBox, QCheckBox,
    QSplitter, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
from pathlib import Path

from ...core.persona import PersonaManager, AIPersona, get_persona_manager


# =============================================================================
# STYLE CONSTANTS
# =============================================================================
STYLE_PRIMARY_BTN = """
    QPushButton {
        background-color: #a6e3a1;
        color: #1e1e2e;
        font-weight: bold;
        padding: 8px 12px;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #94e2d5;
    }
    QPushButton:disabled {
        background-color: #45475a;
        color: #6c7086;
    }
"""

STYLE_SECONDARY_BTN = """
    QPushButton {
        background-color: #45475a;
        padding: 8px 12px;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #585b70;
    }
"""

STYLE_DANGER_BTN = """
    QPushButton {
        background-color: #f38ba8;
        color: #1e1e2e;
        font-weight: bold;
        padding: 8px 12px;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #eba0ac;
    }
"""

STYLE_LIST_WIDGET = """
    QListWidget {
        background-color: #1e1e2e;
        border: 1px solid #45475a;
        border-radius: 4px;
        padding: 4px;
    }
    QListWidget::item {
        padding: 8px;
        border-radius: 4px;
    }
    QListWidget::item:selected {
        background-color: #89b4fa;
        color: #1e1e2e;
    }
    QListWidget::item:hover {
        background-color: #313244;
    }
"""

STYLE_GROUP_BOX = """
    QGroupBox {
        border: 1px solid #45475a;
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 8px;
        font-weight: bold;
    }
    QGroupBox::title {
        color: #89b4fa;
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
    }
"""


class PersonaTab(QWidget):
    """
    Persona management tab widget.
    
    Signals:
        persona_changed: Emitted when the current persona changes
    """
    persona_changed = pyqtSignal(str)  # persona_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = get_persona_manager()
        self.current_persona = None
        self.setup_ui()
        self.load_personas()
    
    def setup_ui(self):
        """Create the UI layout."""
        layout = QVBoxLayout()
        layout.setSpacing(12)
        
        # Header
        header = QLabel("AI Persona Management")
        header.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #cdd6f4;
                padding: 8px;
            }
        """)
        layout.addWidget(header)
        
        # Info text
        info = QLabel("Create, customize, and manage your AI personas. Copy your AI to create variants or import personas from others.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #a6adc8; padding: 4px 8px;")
        layout.addWidget(info)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Persona list
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Persona details
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([300, 500])
        layout.addWidget(splitter)
        
        self.setLayout(layout)
    
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with persona list."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Persona list
        list_label = QLabel("Your Personas:")
        list_label.setStyleSheet("font-weight: bold; color: #cdd6f4;")
        layout.addWidget(list_label)
        
        self.persona_list = QListWidget()
        self.persona_list.setStyleSheet(STYLE_LIST_WIDGET)
        self.persona_list.itemClicked.connect(self.on_persona_selected)
        layout.addWidget(self.persona_list)
        
        # Action buttons
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(8)
        
        self.btn_activate = QPushButton("Set as Current")
        self.btn_activate.setStyleSheet(STYLE_PRIMARY_BTN)
        self.btn_activate.clicked.connect(self.activate_persona)
        btn_layout.addWidget(self.btn_activate)
        
        self.btn_copy = QPushButton("Copy Persona")
        self.btn_copy.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_copy.clicked.connect(self.copy_persona)
        btn_layout.addWidget(self.btn_copy)
        
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setStyleSheet(STYLE_DANGER_BTN)
        self.btn_delete.clicked.connect(self.delete_persona)
        btn_layout.addWidget(self.btn_delete)
        
        layout.addLayout(btn_layout)
        
        # Import/Export section
        io_group = QGroupBox("Import/Export")
        io_group.setStyleSheet(STYLE_GROUP_BOX)
        io_layout = QVBoxLayout()
        
        btn_import = QPushButton("Import from File")
        btn_import.setStyleSheet(STYLE_SECONDARY_BTN)
        btn_import.clicked.connect(self.import_persona)
        io_layout.addWidget(btn_import)
        
        btn_export = QPushButton("Export to File")
        btn_export.setStyleSheet(STYLE_SECONDARY_BTN)
        btn_export.clicked.connect(self.export_persona)
        io_layout.addWidget(btn_export)
        
        btn_templates = QPushButton("Load Template")
        btn_templates.setStyleSheet(STYLE_SECONDARY_BTN)
        btn_templates.clicked.connect(self.load_template)
        io_layout.addWidget(btn_templates)
        
        io_group.setLayout(io_layout)
        layout.addWidget(io_group)
        
        panel.setLayout(layout)
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with persona details."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Details group
        details_group = QGroupBox("Persona Details")
        details_group.setStyleSheet(STYLE_GROUP_BOX)
        details_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self.on_details_changed)
        details_layout.addRow("Name:", self.name_edit)
        
        self.style_combo = QComboBox()
        self.style_combo.addItems(["balanced", "concise", "detailed", "casual"])
        self.style_combo.currentTextChanged.connect(self.on_details_changed)
        details_layout.addRow("Response Style:", self.style_combo)
        
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(["default"])  # TODO: Load from voice profiles
        self.voice_combo.currentTextChanged.connect(self.on_details_changed)
        details_layout.addRow("Voice Profile:", self.voice_combo)
        
        self.avatar_combo = QComboBox()
        self.avatar_combo.addItems(["default"])  # TODO: Load from avatar presets
        self.avatar_combo.currentTextChanged.connect(self.on_details_changed)
        details_layout.addRow("Avatar Preset:", self.avatar_combo)
        
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        # System prompt
        prompt_group = QGroupBox("System Prompt")
        prompt_group.setStyleSheet(STYLE_GROUP_BOX)
        prompt_layout = QVBoxLayout()
        
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setMaximumHeight(100)
        self.prompt_edit.textChanged.connect(self.on_details_changed)
        prompt_layout.addWidget(self.prompt_edit)
        
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # Description
        desc_group = QGroupBox("Description")
        desc_group.setStyleSheet(STYLE_GROUP_BOX)
        desc_layout = QVBoxLayout()
        
        self.desc_edit = QTextEdit()
        self.desc_edit.setMaximumHeight(80)
        self.desc_edit.textChanged.connect(self.on_details_changed)
        desc_layout.addWidget(self.desc_edit)
        
        desc_group.setLayout(desc_layout)
        layout.addWidget(desc_group)
        
        # Save button
        self.btn_save = QPushButton("Save Changes")
        self.btn_save.setStyleSheet(STYLE_PRIMARY_BTN)
        self.btn_save.clicked.connect(self.save_persona)
        self.btn_save.setEnabled(False)
        layout.addWidget(self.btn_save)
        
        # Add stretch to push everything up
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def load_personas(self):
        """Load personas into the list."""
        self.persona_list.clear()
        personas = self.manager.list_personas()
        
        current_id = self.manager.current_persona_id
        
        for persona_info in personas:
            item = QListWidgetItem(persona_info['name'])
            item.setData(Qt.UserRole, persona_info['id'])
            
            # Mark current persona
            if persona_info['id'] == current_id:
                item.setText(f"{persona_info['name']} (Current)")
                item.setForeground(Qt.green)
            
            self.persona_list.addItem(item)
    
    def on_persona_selected(self, item):
        """Handle persona selection."""
        persona_id = item.data(Qt.UserRole)
        self.current_persona = self.manager.load_persona(persona_id)
        
        if self.current_persona:
            self.display_persona(self.current_persona)
    
    def display_persona(self, persona: AIPersona):
        """Display persona details in the form."""
        self.name_edit.setText(persona.name)
        self.prompt_edit.setPlainText(persona.system_prompt)
        self.desc_edit.setPlainText(persona.description)
        
        # Set response style
        index = self.style_combo.findText(persona.response_style)
        if index >= 0:
            self.style_combo.setCurrentIndex(index)
        
        # Set voice profile
        index = self.voice_combo.findText(persona.voice_profile_id)
        if index >= 0:
            self.voice_combo.setCurrentIndex(index)
        
        # Set avatar preset
        index = self.avatar_combo.findText(persona.avatar_preset_id)
        if index >= 0:
            self.avatar_combo.setCurrentIndex(index)
        
        self.btn_save.setEnabled(False)
    
    def on_details_changed(self):
        """Enable save button when details change."""
        if self.current_persona:
            self.btn_save.setEnabled(True)
    
    def save_persona(self):
        """Save changes to current persona."""
        if not self.current_persona:
            return
        
        # Update persona with form values
        self.current_persona.name = self.name_edit.text()
        self.current_persona.system_prompt = self.prompt_edit.toPlainText()
        self.current_persona.description = self.desc_edit.toPlainText()
        self.current_persona.response_style = self.style_combo.currentText()
        self.current_persona.voice_profile_id = self.voice_combo.currentText()
        self.current_persona.avatar_preset_id = self.avatar_combo.currentText()
        
        # Save to disk
        self.manager.save_persona(self.current_persona)
        
        # Reload list
        self.load_personas()
        
        self.btn_save.setEnabled(False)
        QMessageBox.information(self, "Saved", f"Persona '{self.current_persona.name}' saved successfully!")
    
    def activate_persona(self):
        """Set selected persona as current."""
        if not self.current_persona:
            QMessageBox.warning(self, "No Selection", "Please select a persona first.")
            return
        
        self.manager.set_current_persona(self.current_persona.id)
        self.load_personas()
        self.persona_changed.emit(self.current_persona.id)
        
        QMessageBox.information(self, "Activated", f"'{self.current_persona.name}' is now your current persona!")
    
    def copy_persona(self):
        """Copy the selected persona."""
        if not self.current_persona:
            QMessageBox.warning(self, "No Selection", "Please select a persona to copy.")
            return
        
        # Dialog to get new name
        dialog = CopyPersonaDialog(self.current_persona.name, self)
        if dialog.exec_() == QDialog.Accepted:
            new_name = dialog.name_edit.text()
            copy_learning = dialog.learning_check.isChecked()
            
            # Copy persona
            new_persona = self.manager.copy_persona(
                self.current_persona.id,
                new_name,
                copy_learning_data=copy_learning
            )
            
            if new_persona:
                self.load_personas()
                QMessageBox.information(self, "Success", f"Created copy: '{new_name}'")
            else:
                QMessageBox.critical(self, "Error", "Failed to copy persona.")
    
    def delete_persona(self):
        """Delete the selected persona."""
        if not self.current_persona:
            QMessageBox.warning(self, "No Selection", "Please select a persona to delete.")
            return
        
        if self.current_persona.id == "default":
            QMessageBox.warning(self, "Cannot Delete", "Cannot delete the default persona.")
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete '{self.current_persona.name}'? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.manager.delete_persona(self.current_persona.id):
                self.current_persona = None
                self.load_personas()
                QMessageBox.information(self, "Deleted", "Persona deleted successfully.")
            else:
                QMessageBox.critical(self, "Error", "Failed to delete persona.")
    
    def export_persona(self):
        """Export the selected persona to file."""
        if not self.current_persona:
            QMessageBox.warning(self, "No Selection", "Please select a persona to export.")
            return
        
        # File dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Persona",
            f"{self.current_persona.name}.forge-ai",
            "ForgeAI Persona (*.forge-ai);;JSON Files (*.json)"
        )
        
        if file_path:
            result = self.manager.export_persona(self.current_persona.id, Path(file_path))
            if result:
                QMessageBox.information(self, "Success", f"Exported to: {file_path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to export persona.")
    
    def import_persona(self):
        """Import a persona from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Persona",
            "",
            "ForgeAI Persona (*.forge-ai);;JSON Files (*.json)"
        )
        
        if file_path:
            # Ask for optional name override
            from PyQt5.QtWidgets import QInputDialog
            name, ok = QInputDialog.getText(
                self,
                "Import Persona",
                "Enter a new name (or leave empty to keep original):"
            )
            
            persona = self.manager.import_persona(Path(file_path), name if ok and name else None)
            
            if persona:
                self.load_personas()
                QMessageBox.information(self, "Success", f"Imported: '{persona.name}'")
            else:
                QMessageBox.critical(self, "Error", "Failed to import persona.")
    
    def load_template(self):
        """Load a template persona."""
        templates_dir = self.manager.templates_dir
        
        # Find available templates
        templates = list(templates_dir.glob("*.forge-ai"))
        
        if not templates:
            QMessageBox.information(self, "No Templates", "No template personas found.")
            return
        
        # Dialog to choose template
        template_names = [t.stem.replace('_', ' ').title() for t in templates]
        
        from PyQt5.QtWidgets import QInputDialog
        choice, ok = QInputDialog.getItem(
            self,
            "Load Template",
            "Choose a template persona:",
            template_names,
            0,
            False
        )
        
        if ok and choice:
            # Find the file
            template_file = templates[template_names.index(choice)]
            
            # Import it
            persona = self.manager.import_persona(template_file)
            
            if persona:
                self.load_personas()
                QMessageBox.information(self, "Success", f"Loaded template: '{persona.name}'")
            else:
                QMessageBox.critical(self, "Error", "Failed to load template.")


class CopyPersonaDialog(QDialog):
    """Dialog for copying a persona."""
    
    def __init__(self, original_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Copy Persona")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # Name input
        form_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        self.name_edit.setText(f"{original_name} (Copy)")
        form_layout.addRow("New Name:", self.name_edit)
        
        layout.addLayout(form_layout)
        
        # Options
        self.learning_check = QCheckBox("Copy learning data")
        self.learning_check.setToolTip("Include training data from the original persona")
        layout.addWidget(self.learning_check)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)


def create_persona_tab(parent=None) -> PersonaTab:
    """
    Create the persona management tab.
    
    Args:
        parent: Parent widget
        
    Returns:
        PersonaTab widget
    """
    return PersonaTab(parent)
