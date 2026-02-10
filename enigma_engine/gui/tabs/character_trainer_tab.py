"""
Character Trainer Tab - GUI for extracting and training AI on character personalities.

This tab allows users to:
- Scan training data for characters
- Extract character dialogue patterns and traits
- Preview character personality before training
- Generate character-specific training datasets
- Generate task-based training (image prompts, avatar control, tools)
- Train specialized models for characters
"""

import logging
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...config import CONFIG
from ...tools.data_trainer import (
    CharacterProfile, 
    CharacterTrainer, 
    get_character_trainer,
    get_task_trainer
)

logger = logging.getLogger(__name__)


class ScanWorker(QThread):
    """Background worker for scanning characters."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, trainer: CharacterTrainer, data_path: str, min_count: int):
        super().__init__()
        self.trainer = trainer
        self.data_path = data_path
        self.min_count = min_count
    
    def run(self):
        try:
            characters = self.trainer.scan_for_characters(
                self.data_path, 
                min_dialogue_count=self.min_count
            )
            self.finished.emit(characters)
        except Exception as e:
            self.error.emit(str(e))


class ExtractWorker(QThread):
    """Background worker for extracting character data."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, trainer: CharacterTrainer, character_name: str, data_path: str, aliases: list):
        super().__init__()
        self.trainer = trainer
        self.character_name = character_name
        self.data_path = data_path
        self.aliases = aliases
    
    def run(self):
        try:
            self.progress.emit(f"Extracting '{self.character_name}'...")
            result = self.trainer.extract_character(
                self.character_name,
                self.data_path,
                aliases=self.aliases
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class CharacterTrainerTab(QWidget):
    """
    Tab for training AI on specific character personalities and task capabilities.
    
    Features:
    - Scan training data for character names
    - Extract character dialogue and traits
    - Preview personality profile
    - Generate training datasets
    - Train specialized character models
    - Generate task training (image prompts, avatar control, tool usage)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.trainer = get_character_trainer()
        self.task_trainer = get_task_trainer()
        self.current_profile: Optional[CharacterProfile] = None
        self.data_path = str(CONFIG.get("training_data_dir", "data/training"))
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)
        
        # Header
        header = QLabel("AI Trainer")
        header.setStyleSheet("""
            font-size: 12px;
            font-weight: bold;
            color: #f5c2e7;
            padding: 8px;
        """)
        layout.addWidget(header)
        
        description = QLabel(
            "Train your AI on character personalities and task capabilities (image generation, avatar control, tools)."
        )
        description.setStyleSheet("color: #a6adc8; padding: 4px 8px;")
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Tab widget for Character vs Task training
        self.training_tabs = QTabWidget()
        self.training_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #45475a;
                border-radius: 6px;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #45475a;
                color: #f5c2e7;
            }
        """)
        
        # Character Training Tab
        char_widget = self._create_character_training_widget()
        self.training_tabs.addTab(char_widget, "Character Personality")
        
        # Task Training Tab
        task_widget = self._create_task_training_widget()
        self.training_tabs.addTab(task_widget, "Task Training")
        
        layout.addWidget(self.training_tabs, 1)
        
        self.setLayout(layout)
    
    def _create_character_training_widget(self) -> QWidget:
        """Create the character training widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 8, 4, 4)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Character discovery
        left_panel = self._create_discovery_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Character profile
        right_panel = self._create_profile_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([400, 600])
        layout.addWidget(splitter, 1)
        
        # Bottom panel - Actions
        actions_panel = self._create_actions_panel()
        layout.addWidget(actions_panel)
        
        return widget
    
    def _create_task_training_widget(self) -> QWidget:
        """Create the task training widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 8, 4, 4)
        
        # Description
        desc = QLabel(
            "Generate training data to teach your AI specific capabilities. "
            "These datasets train the AI HOW to do things, not just how to speak."
        )
        desc.setStyleSheet("color: #a6adc8; padding: 8px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(12)
        
        # Task categories
        task_groups = [
            ("Image Prompting", "image", "#89b4fa", 
             "Teach AI to write detailed, effective image generation prompts from simple requests."),
            ("Avatar Control", "avatar", "#f5c2e7", 
             "Teach AI to use avatar commands ([emotion:X], [gesture:X]) for expressive interactions."),
            ("Tool Usage", "tools", "#a6e3a1", 
             "Teach AI to use web search, file operations, and other tools appropriately."),
            ("Code Generation", "code", "#f9e2af", 
             "Teach AI to write clean, documented, well-structured code."),
        ]
        
        self.task_checkboxes = {}
        
        for name, key, color, description in task_groups:
            group = QGroupBox(name)
            group.setStyleSheet(f"""
                QGroupBox {{
                    font-weight: bold;
                    border: 1px solid #45475a;
                    border-radius: 6px;
                    margin-top: 8px;
                    padding-top: 8px;
                }}
                QGroupBox::title {{
                    color: {color};
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 4px;
                }}
            """)
            group_layout = QVBoxLayout(group)
            
            # Description
            desc_label = QLabel(description)
            desc_label.setStyleSheet("color: #a6adc8;")
            desc_label.setWordWrap(True)
            group_layout.addWidget(desc_label)
            
            # Example count
            count = self.task_trainer.get_example_count(key)
            count_label = QLabel(f"Examples: {count} training pairs")
            count_label.setStyleSheet("color: #6c7086; font-size: 10px;")
            group_layout.addWidget(count_label)
            
            # Checkbox row
            checkbox_row = QHBoxLayout()
            checkbox = QCheckBox(f"Include {name.lower()} training")
            checkbox.setChecked(True)
            self.task_checkboxes[key] = checkbox
            checkbox_row.addWidget(checkbox)
            
            # Preview button
            btn_preview = QPushButton("Preview")
            btn_preview.setMaximumWidth(80)
            btn_preview.clicked.connect(lambda checked, k=key: self._preview_task_examples(k))
            checkbox_row.addWidget(btn_preview)
            checkbox_row.addStretch()
            
            group_layout.addLayout(checkbox_row)
            scroll_layout.addWidget(group)
        
        # Custom examples section
        custom_group = QGroupBox("Custom Examples")
        custom_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                color: #cba6f7;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
        """)
        custom_layout = QVBoxLayout(custom_group)
        
        custom_desc = QLabel("Load additional training examples from a JSON file:")
        custom_desc.setStyleSheet("color: #a6adc8;")
        custom_layout.addWidget(custom_desc)
        
        load_row = QHBoxLayout()
        self.custom_examples_path = QLineEdit()
        self.custom_examples_path.setPlaceholderText("Path to custom examples JSON...")
        load_row.addWidget(self.custom_examples_path)
        
        btn_browse_custom = QPushButton("Browse")
        btn_browse_custom.clicked.connect(self._browse_custom_examples)
        load_row.addWidget(btn_browse_custom)
        
        btn_load_custom = QPushButton("Load")
        btn_load_custom.clicked.connect(self._load_custom_examples)
        load_row.addWidget(btn_load_custom)
        custom_layout.addLayout(load_row)
        
        scroll_layout.addWidget(custom_group)
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)
        
        # Generation options
        options_frame = QFrame()
        options_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 8px;
                background: #1e1e2e;
            }
        """)
        options_layout = QHBoxLayout(options_frame)
        
        options_layout.addWidget(QLabel("Format:"))
        self.task_format_qa = QCheckBox("Q&A")
        self.task_format_qa.setChecked(True)
        options_layout.addWidget(self.task_format_qa)
        
        self.task_format_chat = QCheckBox("Chat")
        options_layout.addWidget(self.task_format_chat)
        
        self.task_include_system = QCheckBox("Include system prompts")
        self.task_include_system.setChecked(True)
        options_layout.addWidget(self.task_include_system)
        
        options_layout.addStretch()
        
        # Action buttons
        btn_generate_selected = QPushButton("Generate Selected")
        btn_generate_selected.setStyleSheet("background: #a6e3a1; color: #1e1e2e; font-weight: bold;")
        btn_generate_selected.clicked.connect(self._generate_selected_tasks)
        options_layout.addWidget(btn_generate_selected)
        
        btn_generate_all = QPushButton("Generate All Tasks")
        btn_generate_all.setStyleSheet("background: #f5c2e7; color: #1e1e2e; font-weight: bold;")
        btn_generate_all.clicked.connect(self._generate_all_tasks)
        options_layout.addWidget(btn_generate_all)
        
        layout.addWidget(options_frame)
        
        return widget
    
    def _preview_task_examples(self, task_type: str):
        """Show preview of task examples."""
        examples = self.task_trainer.examples.get(task_type, [])
        if not examples:
            QMessageBox.information(self, "No Examples", f"No examples found for {task_type}")
            return
        
        # Create preview dialog
        dialog = QMessageBox(self)
        dialog.setWindowTitle(f"{task_type.title()} Training Examples")
        dialog.setIcon(QMessageBox.Information)
        
        # Format preview text
        preview = []
        for i, ex in enumerate(examples[:3], 1):
            preview.append(f"--- Example {i} ({ex.difficulty}) ---")
            preview.append(f"Input: {ex.input_prompt}")
            preview.append(f"Output: {ex.expected_output[:200]}...")
            preview.append("")
        
        if len(examples) > 3:
            preview.append(f"... and {len(examples) - 3} more examples")
        
        dialog.setText("\n".join(preview))
        dialog.exec_()
    
    def _browse_custom_examples(self):
        """Browse for custom examples file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Custom Examples", "", "JSON Files (*.json)"
        )
        if path:
            self.custom_examples_path.setText(path)
    
    def _load_custom_examples(self):
        """Load custom examples from file."""
        path = self.custom_examples_path.text()
        if not path:
            QMessageBox.warning(self, "Error", "Please select a file first.")
            return
        
        count = self.task_trainer.load_examples_from_file(path)
        if count > 0:
            QMessageBox.information(self, "Loaded", f"Loaded {count} custom examples.")
        else:
            QMessageBox.warning(self, "Error", "Failed to load examples or file is empty.")
    
    def _generate_selected_tasks(self):
        """Generate training data for selected tasks."""
        # Get selected tasks
        selected = [key for key, cb in self.task_checkboxes.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Error", "Please select at least one task type.")
            return
        
        # Get output path
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Task Training Data", "task_training.txt", "Text Files (*.txt)"
        )
        if not path:
            return
        
        format_type = "qa" if self.task_format_qa.isChecked() else "chat"
        include_system = self.task_include_system.isChecked()
        
        # Generate each selected task
        output_base = Path(path)
        generated = []
        
        for task_type in selected:
            output_path = str(output_base.parent / f"{output_base.stem}_{task_type}{output_base.suffix}")
            if task_type == "image":
                success = self.task_trainer.generate_image_training(output_path, format_type, include_system)
            elif task_type == "avatar":
                success = self.task_trainer.generate_avatar_training(output_path, format_type, include_system)
            elif task_type == "tools":
                success = self.task_trainer.generate_tool_training(output_path, format_type, include_system)
            elif task_type == "code":
                success = self.task_trainer.generate_code_training(output_path, format_type, include_system)
            else:
                continue
            
            if success:
                generated.append(f"{task_type}: {output_path}")
        
        if generated:
            QMessageBox.information(
                self, "Generated", 
                f"Generated training datasets:\n\n" + "\n".join(generated)
            )
        else:
            QMessageBox.warning(self, "Error", "Failed to generate training data.")
    
    def _generate_all_tasks(self):
        """Generate combined training data for all tasks."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Combined Training Data", "all_tasks_training.txt", "Text Files (*.txt)"
        )
        if not path:
            return
        
        format_type = "qa" if self.task_format_qa.isChecked() else "chat"
        
        if self.task_trainer.generate_all_tasks(path, format_type):
            QMessageBox.information(self, "Generated", f"Combined training dataset saved to:\n{path}")
        else:
            QMessageBox.warning(self, "Error", "Failed to generate combined training data.")
    
    def _create_discovery_panel(self) -> QWidget:
        """Create the character discovery panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 6, 0)
        
        # Data source group
        source_group = QGroupBox("Data Source")
        source_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                color: #89b4fa;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
        """)
        source_layout = QVBoxLayout(source_group)
        
        # Path input
        path_row = QHBoxLayout()
        self.path_input = QLineEdit(self.data_path)
        self.path_input.setPlaceholderText("Path to training data...")
        path_row.addWidget(self.path_input)
        
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._browse_data_path)
        path_row.addWidget(btn_browse)
        source_layout.addLayout(path_row)
        
        # Scan button
        scan_row = QHBoxLayout()
        self.btn_scan = QPushButton("Scan for Characters")
        self.btn_scan.setStyleSheet("""
            background: #89b4fa; 
            color: #1e1e2e; 
            font-weight: bold;
            padding: 8px 16px;
        """)
        self.btn_scan.clicked.connect(self._scan_characters)
        scan_row.addWidget(self.btn_scan)
        
        self.scan_status = QLabel("")
        self.scan_status.setStyleSheet("color: #a6adc8;")
        scan_row.addWidget(self.scan_status, 1)
        source_layout.addLayout(scan_row)
        
        layout.addWidget(source_group)
        
        # Characters list group
        chars_group = QGroupBox("Discovered Characters")
        chars_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                color: #a6e3a1;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
        """)
        chars_layout = QVBoxLayout(chars_group)
        
        # Filter
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Type to filter...")
        self.filter_input.textChanged.connect(self._filter_characters)
        filter_row.addWidget(self.filter_input)
        chars_layout.addLayout(filter_row)
        
        # Characters table
        self.characters_table = QTableWidget()
        self.characters_table.setColumnCount(2)
        self.characters_table.setHorizontalHeaderLabels(["Character", "Dialogues"])
        self.characters_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.characters_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.characters_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.characters_table.setSelectionMode(QTableWidget.SingleSelection)
        self.characters_table.itemSelectionChanged.connect(self._on_character_selected)
        chars_layout.addWidget(self.characters_table)
        
        # Manual entry
        manual_row = QHBoxLayout()
        self.manual_name = QLineEdit()
        self.manual_name.setPlaceholderText("Or enter character name manually...")
        manual_row.addWidget(self.manual_name)
        
        btn_extract_manual = QPushButton("Extract")
        btn_extract_manual.clicked.connect(self._extract_manual_character)
        manual_row.addWidget(btn_extract_manual)
        chars_layout.addLayout(manual_row)
        
        layout.addWidget(chars_group, 1)
        
        return panel
    
    def _create_profile_panel(self) -> QWidget:
        """Create the character profile panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 0, 0, 0)
        
        # Profile group
        profile_group = QGroupBox("Character Profile")
        profile_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                color: #f5c2e7;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
        """)
        profile_layout = QVBoxLayout(profile_group)
        
        # Character name and stats
        header_row = QHBoxLayout()
        self.profile_name = QLabel("No character selected")
        self.profile_name.setStyleSheet("font-size: 14px; font-weight: bold; color: #cdd6f4;")
        header_row.addWidget(self.profile_name)
        
        self.profile_stats = QLabel("")
        self.profile_stats.setStyleSheet("color: #a6adc8;")
        header_row.addWidget(self.profile_stats, 1)
        profile_layout.addLayout(header_row)
        
        # Traits display
        traits_row = QHBoxLayout()
        
        # Personality traits
        traits_col = QVBoxLayout()
        traits_col.addWidget(QLabel("Personality Traits:"))
        self.traits_list = QListWidget()
        self.traits_list.setMaximumHeight(120)
        traits_col.addWidget(self.traits_list)
        traits_row.addLayout(traits_col)
        
        # Speech patterns
        speech_col = QVBoxLayout()
        speech_col.addWidget(QLabel("Speech Patterns:"))
        self.speech_list = QListWidget()
        self.speech_list.setMaximumHeight(120)
        speech_col.addWidget(self.speech_list)
        traits_row.addLayout(speech_col)
        
        profile_layout.addLayout(traits_row)
        
        # Catchphrases
        profile_layout.addWidget(QLabel("Catchphrases:"))
        self.catchphrases_text = QTextEdit()
        self.catchphrases_text.setMaximumHeight(60)
        self.catchphrases_text.setReadOnly(True)
        profile_layout.addWidget(self.catchphrases_text)
        
        # Sample dialogues
        profile_layout.addWidget(QLabel("Sample Dialogues:"))
        self.dialogues_text = QTextEdit()
        self.dialogues_text.setReadOnly(True)
        profile_layout.addWidget(self.dialogues_text, 1)
        
        layout.addWidget(profile_group, 1)
        
        # System prompt preview
        prompt_group = QGroupBox("Generated System Prompt")
        prompt_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                color: #f9e2af;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
        """)
        prompt_layout = QVBoxLayout(prompt_group)
        
        self.system_prompt_text = QTextEdit()
        self.system_prompt_text.setMaximumHeight(100)
        self.system_prompt_text.setPlaceholderText("System prompt will be generated from character profile...")
        prompt_layout.addWidget(self.system_prompt_text)
        
        layout.addWidget(prompt_group)
        
        return panel
    
    def _create_actions_panel(self) -> QWidget:
        """Create the actions panel."""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 8px;
                background: #1e1e2e;
            }
        """)
        layout = QHBoxLayout(panel)
        
        # Export options
        self.export_qa = QCheckBox("Export Q&A format")
        self.export_qa.setChecked(True)
        layout.addWidget(self.export_qa)
        
        self.export_chat = QCheckBox("Export chat format")
        layout.addWidget(self.export_chat)
        
        layout.addStretch()
        
        # Action buttons
        btn_save_profile = QPushButton("Save Profile")
        btn_save_profile.setToolTip("Save character profile to JSON file")
        btn_save_profile.clicked.connect(self._save_profile)
        layout.addWidget(btn_save_profile)
        
        btn_generate = QPushButton("Generate Dataset")
        btn_generate.setToolTip("Generate training dataset from character")
        btn_generate.setStyleSheet("background: #a6e3a1; color: #1e1e2e; font-weight: bold;")
        btn_generate.clicked.connect(self._generate_dataset)
        layout.addWidget(btn_generate)
        
        btn_train = QPushButton("Train Model")
        btn_train.setToolTip("Train a specialized model for this character")
        btn_train.setStyleSheet("background: #f5c2e7; color: #1e1e2e; font-weight: bold;")
        btn_train.clicked.connect(self._train_character_model)
        layout.addWidget(btn_train)
        
        return panel
    
    def _browse_data_path(self):
        """Browse for data directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Training Data Directory", self.data_path
        )
        if path:
            self.path_input.setText(path)
            self.data_path = path
    
    def _scan_characters(self):
        """Scan for characters in training data."""
        self.data_path = self.path_input.text()
        if not self.data_path:
            QMessageBox.warning(self, "Error", "Please select a data path first.")
            return
        
        self.btn_scan.setEnabled(False)
        self.scan_status.setText("Scanning...")
        
        self.scan_worker = ScanWorker(self.trainer, self.data_path, min_count=3)
        self.scan_worker.finished.connect(self._on_scan_complete)
        self.scan_worker.error.connect(self._on_scan_error)
        self.scan_worker.start()
    
    def _on_scan_complete(self, characters: dict):
        """Handle scan completion."""
        self.btn_scan.setEnabled(True)
        self.characters_table.setRowCount(0)
        
        # Sort by dialogue count
        sorted_chars = sorted(characters.items(), key=lambda x: x[1], reverse=True)
        
        for name, count in sorted_chars:
            row = self.characters_table.rowCount()
            self.characters_table.insertRow(row)
            self.characters_table.setItem(row, 0, QTableWidgetItem(name))
            self.characters_table.setItem(row, 1, QTableWidgetItem(str(count)))
        
        self.scan_status.setText(f"Found {len(characters)} characters")
    
    def _on_scan_error(self, error: str):
        """Handle scan error."""
        self.btn_scan.setEnabled(True)
        self.scan_status.setText("Scan failed")
        QMessageBox.warning(self, "Scan Error", error)
    
    def _filter_characters(self, text: str):
        """Filter characters table by text."""
        text = text.lower()
        for row in range(self.characters_table.rowCount()):
            item = self.characters_table.item(row, 0)
            if item:
                visible = text in item.text().lower()
                self.characters_table.setRowHidden(row, not visible)
    
    def _on_character_selected(self):
        """Handle character selection."""
        selected = self.characters_table.selectedItems()
        if selected:
            character_name = selected[0].text()
            self._extract_character(character_name)
    
    def _extract_manual_character(self):
        """Extract manually entered character."""
        name = self.manual_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a character name.")
            return
        self._extract_character(name)
    
    def _extract_character(self, character_name: str):
        """Extract character data."""
        self.profile_name.setText(f"Extracting {character_name}...")
        self.profile_stats.setText("")
        
        self.extract_worker = ExtractWorker(
            self.trainer, 
            character_name, 
            self.data_path,
            aliases=[]
        )
        self.extract_worker.finished.connect(self._on_extract_complete)
        self.extract_worker.error.connect(self._on_extract_error)
        self.extract_worker.start()
    
    def _on_extract_complete(self, result):
        """Handle extraction completion."""
        if not result.success:
            self.profile_name.setText("Extraction failed")
            QMessageBox.warning(self, "Error", result.error)
            return
        
        self.current_profile = result.character
        self._display_profile(result.character)
    
    def _on_extract_error(self, error: str):
        """Handle extraction error."""
        self.profile_name.setText("Error")
        QMessageBox.warning(self, "Extraction Error", error)
    
    def _display_profile(self, profile: CharacterProfile):
        """Display character profile in UI."""
        self.profile_name.setText(profile.name)
        self.profile_stats.setText(
            f"{profile.dialogue_count} dialogues | {len(profile.source_files)} files"
        )
        
        # Personality traits
        self.traits_list.clear()
        for trait, score in sorted(
            profile.personality_traits.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            if score > 0.1:
                item = QListWidgetItem(f"{trait}: {score:.0%}")
                self.traits_list.addItem(item)
        
        # Speech patterns
        self.speech_list.clear()
        for pattern, value in profile.speech_patterns.items():
            if isinstance(value, float):
                if value > 0.1:
                    item = QListWidgetItem(f"{pattern}: {value:.0%}")
                    self.speech_list.addItem(item)
            else:
                item = QListWidgetItem(f"{pattern}: {value}")
                self.speech_list.addItem(item)
        
        # Catchphrases
        self.catchphrases_text.clear()
        if profile.catchphrases:
            self.catchphrases_text.setPlainText("\n".join(profile.catchphrases[:5]))
        else:
            self.catchphrases_text.setPlainText("(No distinctive catchphrases found)")
        
        # Sample dialogues
        self.dialogues_text.clear()
        for dialogue in profile.sample_dialogues[:10]:
            self.dialogues_text.append(f"- {dialogue}\n")
        
        # Generate system prompt
        system_prompt = self.trainer._generate_system_prompt(profile)
        self.system_prompt_text.setPlainText(system_prompt)
    
    def _save_profile(self):
        """Save character profile to file."""
        if not self.current_profile:
            QMessageBox.warning(self, "Error", "No character profile to save.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Character Profile",
            f"{self.current_profile.name}_profile.json",
            "JSON Files (*.json)"
        )
        if path:
            if self.trainer.save_profile(self.current_profile.name, path):
                QMessageBox.information(self, "Saved", f"Profile saved to {path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to save profile.")
    
    def _generate_dataset(self):
        """Generate training dataset from character."""
        if not self.current_profile:
            QMessageBox.warning(self, "Error", "No character profile selected.")
            return
        
        # Determine format
        formats = []
        if self.export_qa.isChecked():
            formats.append("qa")
        if self.export_chat.isChecked():
            formats.append("chat")
        
        if not formats:
            formats = ["qa"]
        
        # Get save path
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Training Dataset",
            f"{self.current_profile.name}_training.txt",
            "Text Files (*.txt)"
        )
        
        if not path:
            return
        
        # Generate dataset
        system_prompt = self.system_prompt_text.toPlainText()
        
        for fmt in formats:
            output_path = path if fmt == formats[0] else path.replace(".txt", f"_{fmt}.txt")
            success = self.trainer.generate_training_dataset(
                self.current_profile.name,
                self.data_path,
                output_path,
                format=fmt,
                system_prompt=system_prompt
            )
            
            if success:
                logger.info(f"Generated {fmt} dataset: {output_path}")
        
        QMessageBox.information(
            self, 
            "Dataset Generated", 
            f"Training dataset saved to:\n{path}"
        )
    
    def _train_character_model(self):
        """Train a model on the character data."""
        if not self.current_profile:
            QMessageBox.warning(self, "Error", "No character profile selected.")
            return
        
        # Generate dataset first
        output_dir = Path(CONFIG.get("training_data_dir", "data/training"))
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = output_dir / f"{self.current_profile.name}_training.txt"
        
        system_prompt = self.system_prompt_text.toPlainText()
        
        if not self.trainer.generate_training_dataset(
            self.current_profile.name,
            self.data_path,
            str(dataset_path),
            format="qa",
            system_prompt=system_prompt
        ):
            QMessageBox.warning(self, "Error", "Failed to generate training dataset.")
            return
        
        # Try to trigger training via parent window
        if hasattr(self, 'parent_window') and self.parent_window:
            # Check if training tab exists
            if hasattr(self.parent_window, 'training_data_path'):
                self.parent_window.training_data_path = str(dataset_path)
                QMessageBox.information(
                    self,
                    "Ready to Train",
                    f"Training dataset generated at:\n{dataset_path}\n\n"
                    "Switch to the Training tab to start training."
                )
            else:
                QMessageBox.information(
                    self,
                    "Dataset Ready",
                    f"Training dataset generated at:\n{dataset_path}\n\n"
                    "Open the Training tab and load this file to train."
                )
        else:
            QMessageBox.information(
                self,
                "Dataset Ready",
                f"Training dataset generated at:\n{dataset_path}"
            )


def create_character_trainer_tab(parent=None):
    """Create the character trainer tab."""
    return CharacterTrainerTab(parent)
