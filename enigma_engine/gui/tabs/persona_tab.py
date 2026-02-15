"""
Prompt Management Tab - Create, Copy, Export, Import AI Prompts

This tab allows users to:
- View and manage their AI prompts/personas
- Copy prompts to create variants
- Export prompts to share with others
- Import prompts from files
- Edit prompt details
- Switch between prompts

Usage:
    from enigma_engine.gui.tabs.persona_tab import create_persona_tab
    
    prompt_widget = create_persona_tab(parent_window)
    tabs.addTab(prompt_widget, "Prompt")
"""

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...core.persona import AIPersona, get_persona_manager

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
        border: none;
    }
    QPushButton:hover {
        background-color: #94e2d5;
    }
    QPushButton:pressed {
        background-color: #74c7ec;
    }
    QPushButton:disabled {
        background-color: #313244;
        color: #f38ba8;
        border: 2px dashed #f38ba8;
    }
"""

STYLE_SECONDARY_BTN = """
    QPushButton {
        background-color: #89b4fa;
        color: #1e1e2e;
        font-weight: bold;
        padding: 8px 12px;
        border-radius: 4px;
        border: none;
    }
    QPushButton:hover {
        background-color: #b4befe;
    }
    QPushButton:pressed {
        background-color: #74c7ec;
    }
    QPushButton:disabled {
        background-color: #313244;
        color: #f38ba8;
        border: 2px dashed #f38ba8;
    }
"""

STYLE_DANGER_BTN = """
    QPushButton {
        background-color: #f38ba8;
        color: #1e1e2e;
        font-weight: bold;
        padding: 8px 12px;
        border-radius: 4px;
        border: none;
    }
    QPushButton:hover {
        background-color: #eba0ac;
    }
    QPushButton:pressed {
        background-color: #fab387;
    }
    QPushButton:disabled {
        background-color: #313244;
        color: #f38ba8;
        border: 2px dashed #f38ba8;
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
    Prompt management tab widget.
    
    Signals:
        persona_changed: Emitted when the current prompt changes
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
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Header - fixed height, no expansion
        header = QLabel("Prompt Settings")
        header.setStyleSheet("font-weight: bold; color: #cdd6f4; font-size: 12px;")
        header.setFixedHeight(20)
        header.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(header)
        
        # Info text - fixed height, no expansion
        info = QLabel("Configure your AI's system prompt and personality.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #a6adc8; font-size: 10px;")
        info.setFixedHeight(18)
        info.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
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
        """Create the left panel with prompt list."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Prompt list
        list_label = QLabel("Saved Prompts:")
        list_label.setStyleSheet("font-weight: bold; color: #cdd6f4;")
        layout.addWidget(list_label)
        
        self.persona_list = QListWidget()
        self.persona_list.setStyleSheet(STYLE_LIST_WIDGET)
        self.persona_list.itemClicked.connect(self.on_persona_selected)
        layout.addWidget(self.persona_list)
        
        # Action buttons
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(8)
        
        btn_new = QPushButton("New")
        btn_new.setStyleSheet(STYLE_PRIMARY_BTN)
        btn_new.clicked.connect(self.prepare_new_prompt)
        btn_layout.addWidget(btn_new)
        
        self.btn_activate = QPushButton("Set as Current")
        self.btn_activate.setStyleSheet(STYLE_PRIMARY_BTN)
        self.btn_activate.clicked.connect(self.activate_persona)
        btn_layout.addWidget(self.btn_activate)
        
        self.btn_copy = QPushButton("Copy")
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
        """Create the right panel with prompt details."""
        from PyQt5.QtWidgets import QScrollArea
        
        # Scroll area wrapper
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Details group
        details_group = QGroupBox("Prompt Details")
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
        self.voice_combo.addItems(["default"])
        # Load voice profiles from data directory
        self._load_voice_profiles()
        self.voice_combo.currentTextChanged.connect(self.on_details_changed)
        details_layout.addRow("Voice Profile:", self.voice_combo)
        
        self.avatar_combo = QComboBox()
        self.avatar_combo.addItems(["default"])
        # Load avatar presets from data directory
        self._load_avatar_presets()
        self.avatar_combo.currentTextChanged.connect(self.on_details_changed)
        details_layout.addRow("Avatar Preset:", self.avatar_combo)
        
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        # System prompt - THE MOST IMPORTANT SETTING
        prompt_group = QGroupBox("System Prompt (What your AI should be)")
        prompt_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #89b4fa;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                background-color: rgba(137, 180, 250, 0.1);
            }
            QGroupBox::title {
                color: #89b4fa;
                subcontrol-origin: margin;
                left: 10px;
            }
        """)
        prompt_layout = QVBoxLayout()
        
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setMinimumHeight(150)
        self.prompt_edit.setPlaceholderText(
            "Tell your AI who it should be and how it should behave.\\n\\n"
            "Example: You are a friendly coding assistant. You write clean, "
            "well-documented code and explain concepts clearly. You ask "
            "clarifying questions when needed.\\n\\n"
            "Be specific! This is the most important setting for your AI."
        )
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
        scroll.setWidget(panel)
        return scroll
    
    def load_personas(self):
        """Load prompts into the list."""
        self.persona_list.clear()
        personas = self.manager.list_personas()
        
        current_id = self.manager.current_persona_id
        
        for persona_info in personas:
            item = QListWidgetItem(persona_info['name'])
            item.setData(Qt.UserRole, persona_info['id'])
            
            # Mark current prompt
            if persona_info['id'] == current_id:
                item.setText(f"{persona_info['name']} (Current)")
                item.setForeground(Qt.green)
            
            self.persona_list.addItem(item)
    
    def _load_voice_profiles(self):
        """Load voice profiles from data directory (async for system voices)."""
        from pathlib import Path

        # Voice profiles directory (fast - local file check)
        voice_dir = Path("data/voice_profiles")
        if voice_dir.exists():
            for profile_file in voice_dir.glob("*.json"):
                profile_name = profile_file.stem
                if profile_name not in [self.voice_combo.itemText(i) for i in range(self.voice_combo.count())]:
                    self.voice_combo.addItem(profile_name)
        
        # Load system voices in background thread (pyttsx3 init can be slow)
        import threading
        def load_system_voices():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                voice_names = []
                for voice in voices[:10]:  # Limit to first 10
                    voice_name = voice.name.split()[-1] if voice.name else voice.id
                    voice_names.append(voice_name)
                
                # Update combo on main thread using QTimer
                from PyQt5.QtCore import QTimer
                def add_voices():
                    for voice_name in voice_names:
                        if voice_name not in [self.voice_combo.itemText(i) for i in range(self.voice_combo.count())]:
                            self.voice_combo.addItem(voice_name)
                
                QTimer.singleShot(0, add_voices)
            except Exception:
                pass  # Intentionally silent
        
        thread = threading.Thread(target=load_system_voices, daemon=True)
        thread.start()
    
    def _load_avatar_presets(self):
        """Load avatar presets from data directory."""
        from pathlib import Path

        # Avatar presets directory
        avatar_dir = Path("data/avatar")
        if avatar_dir.exists():
            # Load .png, .jpg images as presets
            for img_file in avatar_dir.glob("*.png"):
                preset_name = img_file.stem
                if preset_name not in [self.avatar_combo.itemText(i) for i in range(self.avatar_combo.count())]:
                    self.avatar_combo.addItem(preset_name)
            for img_file in avatar_dir.glob("*.jpg"):
                preset_name = img_file.stem
                if preset_name not in [self.avatar_combo.itemText(i) for i in range(self.avatar_combo.count())]:
                    self.avatar_combo.addItem(preset_name)
        
        # Also load VRM models if any
        vrm_dir = Path("data/avatar/vrm")
        if vrm_dir.exists():
            for vrm_file in vrm_dir.glob("*.vrm"):
                preset_name = f"vrm:{vrm_file.stem}"
                if preset_name not in [self.avatar_combo.itemText(i) for i in range(self.avatar_combo.count())]:
                    self.avatar_combo.addItem(preset_name)
    
    def on_persona_selected(self, item):
        """Handle prompt selection."""
        self._creating_new = False  # Cancel any new prompt creation
        persona_id = item.data(Qt.UserRole)
        self.current_persona = self.manager.load_persona(persona_id)
        
        if self.current_persona:
            self.display_persona(self.current_persona)
    
    def display_persona(self, persona: AIPersona):
        """Display prompt details in the form."""
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
        if self.current_persona or getattr(self, '_creating_new', False):
            self.btn_save.setEnabled(True)
    
    def prepare_new_prompt(self):
        """Prepare the right panel for creating a new prompt."""
        self._creating_new = True
        self.current_persona = None
        self.persona_list.clearSelection()
        
        # Clear all fields
        self.name_edit.setText("")
        self.name_edit.setPlaceholderText("My Custom AI")
        self.prompt_edit.setPlainText("")
        self.desc_edit.setPlainText("")
        self.style_combo.setCurrentIndex(0)
        self.voice_combo.setCurrentIndex(0)
        self.avatar_combo.setCurrentIndex(0)
        
        # Enable save button
        self.btn_save.setEnabled(True)
        self.btn_save.setText("Create Prompt")
        
        # Focus on name field
        self.name_edit.setFocus()
    
    def save_persona(self):
        """Save changes to current prompt or create new one."""
        # If creating new prompt
        if getattr(self, '_creating_new', False):
            name = self.name_edit.text().strip()
            if not name:
                QMessageBox.warning(self, "Invalid Name", "Please enter a name for the prompt.")
                return
            
            # Create persona ID from name
            import re
            from datetime import datetime
            persona_id = re.sub(r'[^a-z0-9_]', '_', name.lower()).strip('_')
            
            # Check if ID already exists
            if self.manager.persona_exists(persona_id):
                persona_id = f"{persona_id}_{int(datetime.now().timestamp())}"
            
            # Create new persona object
            from ...core.persona import AIPersona
            new_persona = AIPersona(
                id=persona_id,
                name=name,
                created_at=datetime.now().isoformat(),
                personality_traits={},
                description=self.desc_edit.toPlainText() or f"Custom prompt: {name}",
                system_prompt=self.prompt_edit.toPlainText() or "You are a helpful AI assistant.",
                response_style=self.style_combo.currentText(),
                voice_profile_id=self.voice_combo.currentText(),
                avatar_preset_id=self.avatar_combo.currentText()
            )
            
            # Save it
            try:
                self.manager.save_persona(new_persona)
                self._creating_new = False
                self.btn_save.setText("Save Changes")
                self.load_personas()
                # Select the new prompt
                for i in range(self.persona_list.count()):
                    item = self.persona_list.item(i)
                    if item.data(Qt.UserRole) == new_persona.id:
                        self.persona_list.setCurrentItem(item)
                        self.on_persona_selected(item)
                        break
                QMessageBox.information(self, "Created", f"Prompt '{name}' created successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create prompt: {e}")
            return
        
        # Otherwise, update existing prompt
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
        QMessageBox.information(self, "Saved", f"Prompt '{self.current_persona.name}' saved successfully!")
    
    def activate_persona(self):
        """Set selected prompt as current."""
        if not self.current_persona:
            QMessageBox.warning(self, "No Selection", "Please select a prompt first.")
            return
        
        self.manager.set_current_persona(self.current_persona.id)
        self.load_personas()
        self.persona_changed.emit(self.current_persona.id)
        # No popup - the list already shows (Current) next to active prompt
    
    def copy_persona(self):
        """Copy the selected prompt."""
        if not self.current_persona:
            QMessageBox.warning(self, "No Selection", "Please select a prompt to copy.")
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
            "Enigma AI Engine Persona (*.forge-ai);;JSON Files (*.json)"
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
            "Enigma AI Engine Persona (*.forge-ai);;JSON Files (*.json)"
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
    """Dialog for copying a prompt."""
    
    def __init__(self, original_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Copy Prompt")
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
        self.learning_check.setToolTip("Include training data from the original prompt")
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
    Create the prompt management tab.
    
    Args:
        parent: Parent widget
        
    Returns:
        PersonaTab widget
    """
    return PersonaTab(parent)
