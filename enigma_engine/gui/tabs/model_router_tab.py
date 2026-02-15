"""
Model Router Tab - Assign models to tools.

Simple interface for configuring which AI models handle which tasks.
"""

import json
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Tool categories and styling
TOOL_CATEGORIES = {
    "Generation": {
        "color": "#e91e63",
        "tools": ["chat", "image", "code", "video", "audio", "3d", "gif"]
    },
    "Code (Other)": {
        "color": "#f39c12",
        "tools": ["code_other"]
    },
    "Perception": {
        "color": "#3498db",
        "tools": ["vision", "camera"]
    },
    "Memory": {
        "color": "#9b59b6",
        "tools": ["embeddings", "memory"]
    },
    "Output": {
        "color": "#2ecc71",
        "tools": ["avatar", "web"]
    },
    "Training": {
        "color": "#00bcd4",
        "tools": ["teach"]
    }
}

TOOL_INFO = {
    "chat": {"name": "Chat", "desc": "Text conversation and reasoning"},
    "image": {"name": "Image Gen", "desc": "Generate images from text"},
    "code": {"name": "Code Gen", "desc": "Generate and edit code"},
    "code_other": {"name": "Other Languages", "desc": "Fallback for other programming languages"},
    "video": {"name": "Video Gen", "desc": "Generate video clips"},
    "audio": {"name": "Audio/TTS", "desc": "Text-to-speech and audio"},
    "3d": {"name": "3D Gen", "desc": "Generate 3D models"},
    "gif": {"name": "GIF Gen", "desc": "Create animated GIFs"},
    "vision": {"name": "Vision", "desc": "Analyze images and screens"},
    "camera": {"name": "Camera", "desc": "Webcam capture and analysis"},
    "embeddings": {"name": "Embeddings", "desc": "Semantic text vectors"},
    "memory": {"name": "Memory", "desc": "Conversation storage"},
    "avatar": {"name": "Avatar", "desc": "Visual AI representation"},
    "web": {"name": "Web Tools", "desc": "Web search and fetch"},
    "teach": {"name": "Teacher AI", "desc": "Train and evaluate student models"},
}


class ModelSelectDialog(QDialog):
    """Simple dialog to select a model for a tool."""
    
    def __init__(self, tool_name: str, current_model: str = "", parent=None):
        super().__init__(parent)
        self.selected_model = current_model
        self.setWindowTitle(f"Select Model for {tool_name}")
        self.setMinimumSize(400, 500)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Instructions
        label = QLabel("Select a model to handle this task:")
        label.setStyleSheet("font-size: 12px; margin-bottom: 8px;")
        layout.addWidget(label)
        
        # Model list
        self.model_list = QListWidget()
        self.model_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #444;
                border-radius: 4px;
                background: #1a1a2e;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #333;
            }
            QListWidget::item:selected {
                background: #3498db;
            }
            QListWidget::item:hover {
                background: #2d2d44;
            }
        """)
        self.model_list.itemDoubleClicked.connect(self._select_and_close)
        layout.addWidget(self.model_list)
        
        # Populate models
        self._populate_models()
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear Assignment")
        clear_btn.setStyleSheet("""
            QPushButton {
                background: #e74c3c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
            }
            QPushButton:hover { background: #c0392b; }
        """)
        clear_btn.clicked.connect(self._clear_assignment)
        btn_layout.addWidget(clear_btn)
        
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        select_btn = QPushButton("Select")
        select_btn.setStyleSheet("""
            QPushButton {
                background: #2ecc71;
                color: #1e1e2e;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background: #27ae60; }
        """)
        select_btn.clicked.connect(self._select_model)
        btn_layout.addWidget(select_btn)
        
        layout.addLayout(btn_layout)
        
    def _populate_models(self):
        """Populate the model list from registry."""
        self.model_list.clear()
        
        # Get models from registry
        try:
            from enigma_engine.core.model_registry import ModelRegistry
            registry = ModelRegistry()
            models = registry.list_models()
            
            if models:
                for model_name, model_info in models.items():
                    item = QListWidgetItem(model_name)
                    item.setData(Qt.ItemDataRole.UserRole, model_name)
                    
                    # Add size info if available
                    size = model_info.get("size", "")
                    if size:
                        item.setText(f"{model_name}  ({size})")
                    
                    # Highlight current selection
                    if model_name == self.selected_model:
                        item.setSelected(True)
                        self.model_list.setCurrentItem(item)
                    
                    self.model_list.addItem(item)
        except Exception as e:
            item = QListWidgetItem(f"Error loading models: {e}")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.model_list.addItem(item)
            
        # Add "None" option
        none_item = QListWidgetItem("(No model - use default)")
        none_item.setData(Qt.ItemDataRole.UserRole, "")
        none_item.setForeground(QColor("#888"))
        self.model_list.insertItem(0, none_item)
        
        if not self.selected_model:
            self.model_list.setCurrentRow(0)
            
    def _select_model(self):
        """Select the highlighted model."""
        current = self.model_list.currentItem()
        if current:
            self.selected_model = current.data(Qt.ItemDataRole.UserRole) or ""
            self.accept()
            
    def _select_and_close(self, item):
        """Double-click to select and close."""
        self.selected_model = item.data(Qt.ItemDataRole.UserRole) or ""
        self.accept()
        
    def _clear_assignment(self):
        """Clear the assignment."""
        self.selected_model = ""
        self.accept()


class PromptSelectDialog(QDialog):
    """Dialog to select a prompt/persona for a model."""
    
    def __init__(self, tool_name: str, current_prompt: str = "", parent=None):
        super().__init__(parent)
        self.selected_prompt = current_prompt
        self.setWindowTitle(f"Select Prompt for {tool_name}")
        self.setMinimumSize(400, 400)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        label = QLabel("Select a prompt/persona for this model:")
        label.setStyleSheet("font-size: 12px; margin-bottom: 8px;")
        layout.addWidget(label)
        
        self.prompt_list = QListWidget()
        self.prompt_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #444;
                border-radius: 4px;
                background: #1a1a2e;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #333;
            }
            QListWidget::item:selected { background: #9b59b6; }
            QListWidget::item:hover { background: #2d2d44; }
        """)
        self.prompt_list.itemDoubleClicked.connect(self._select_and_close)
        layout.addWidget(self.prompt_list)
        
        self._populate_prompts()
        
        btn_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear Prompt")
        clear_btn.setStyleSheet("background: #e74c3c; color: white; border: none; border-radius: 4px; padding: 10px 20px;")
        clear_btn.clicked.connect(self._clear_prompt)
        btn_layout.addWidget(clear_btn)
        
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        select_btn = QPushButton("Select")
        select_btn.setStyleSheet("background: #9b59b6; color: white; border: none; border-radius: 4px; padding: 10px 20px; font-weight: bold;")
        select_btn.clicked.connect(self._select_prompt)
        btn_layout.addWidget(select_btn)
        
        layout.addLayout(btn_layout)
        
    def _populate_prompts(self):
        """Populate prompts from personas folder."""
        self.prompt_list.clear()
        
        # Add "None" option
        none_item = QListWidgetItem("(No prompt - use default)")
        none_item.setData(Qt.ItemDataRole.UserRole, "")
        none_item.setForeground(QColor("#888"))
        self.prompt_list.addItem(none_item)
        
        try:
            from enigma_engine.config import CONFIG
            personas_dir = Path(CONFIG.get("data_dir", "data")) / "personas"
            
            if personas_dir.exists():
                for p in sorted(personas_dir.iterdir()):
                    if p.is_dir():
                        persona_file = p / "persona.json"
                        if persona_file.exists():
                            try:
                                with open(persona_file, encoding='utf-8') as f:
                                    data = json.load(f)
                                name = data.get("name", p.name)
                                item = QListWidgetItem(name)
                                item.setData(Qt.ItemDataRole.UserRole, p.name)
                                
                                if p.name == self.selected_prompt:
                                    item.setSelected(True)
                                    self.prompt_list.setCurrentItem(item)
                                    
                                self.prompt_list.addItem(item)
                            except (json.JSONDecodeError, OSError, KeyError):
                                pass
        except Exception as e:
            item = QListWidgetItem(f"Error: {e}")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.prompt_list.addItem(item)
            
        if not self.selected_prompt:
            self.prompt_list.setCurrentRow(0)
            
    def _select_prompt(self):
        current = self.prompt_list.currentItem()
        if current:
            self.selected_prompt = current.data(Qt.ItemDataRole.UserRole) or ""
            self.accept()
            
    def _select_and_close(self, item):
        self.selected_prompt = item.data(Qt.ItemDataRole.UserRole) or ""
        self.accept()
        
    def _clear_prompt(self):
        self.selected_prompt = ""
        self.accept()


class ModelRouterTab(QWidget):
    """Simple model router configuration tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.assignments: dict[str, str] = {}  # tool -> model_name
        self.prompts: dict[str, str] = {}  # tool -> prompt_id
        self._setup_ui()
        self._load_config()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("Model Router")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        header.addWidget(title)
        
        header.addStretch()
        
        # Help text
        help_label = QLabel("Click a tool to assign a model")
        help_label.setStyleSheet("color: #888; font-size: 11px;")
        header.addWidget(help_label)
        
        layout.addLayout(header)
        
        # Description
        desc = QLabel("Assign specialized models to different tasks. If no model is assigned, the default model is used.")
        desc.setStyleSheet("color: #bac2de; font-size: 11px; margin-bottom: 8px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Tool table - 3 columns now
        self.tool_table = QTableWidget()
        self.tool_table.setColumnCount(3)
        self.tool_table.setHorizontalHeaderLabels(["Task", "Model", "Prompt"])
        self.tool_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tool_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tool_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tool_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tool_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tool_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tool_table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.tool_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #444;
                border-radius: 6px;
                background: #1a1a2e;
                gridline-color: #333;
            }
            QTableWidget::item {
                padding: 12px;
            }
            QTableWidget::item:selected {
                background: #3d5a80;
            }
            QHeaderView::section {
                background: #2d2d44;
                padding: 10px;
                border: none;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.tool_table)
        
        # Populate table
        self._populate_tool_table()
        
        # Bottom buttons
        bottom = QHBoxLayout()
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #bac2de; font-style: italic;")
        bottom.addWidget(self.status_label)
        
        bottom.addStretch()
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_config)
        bottom.addWidget(refresh_btn)
        
        reset_btn = QPushButton("Clear All")
        reset_btn.setStyleSheet("""
            QPushButton {
                background: #2d2d2d;
                color: #bac2de;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover { background: #3d3d3d; }
        """)
        reset_btn.clicked.connect(self._reset_defaults)
        bottom.addWidget(reset_btn)
        
        save_btn = QPushButton("Save")
        save_btn.setStyleSheet("""
            QPushButton {
                background: #2ecc71;
                color: #1e1e2e;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background: #27ae60; }
        """)
        save_btn.clicked.connect(self._save_config)
        bottom.addWidget(save_btn)
        
        layout.addLayout(bottom)
        
    def _populate_tool_table(self):
        """Populate the tools table."""
        self.tool_table.setRowCount(len(TOOL_INFO))
        
        row = 0
        for tool_id, info in TOOL_INFO.items():
            # Find category color
            color = "#bac2de"
            for cat, cat_info in TOOL_CATEGORIES.items():
                if tool_id in cat_info["tools"]:
                    color = cat_info["color"]
                    break
            
            # Tool name
            name_item = QTableWidgetItem(info["name"])
            name_item.setData(Qt.ItemDataRole.UserRole, tool_id)
            name_item.setForeground(QColor(color))
            name_item.setToolTip(info["desc"])
            self.tool_table.setItem(row, 0, name_item)
            
            # Model (placeholder)
            model_item = QTableWidgetItem("(default)")
            model_item.setForeground(QColor("#666"))
            self.tool_table.setItem(row, 1, model_item)
            
            # Prompt (placeholder)
            prompt_item = QTableWidgetItem("(default)")
            prompt_item.setForeground(QColor("#666"))
            self.tool_table.setItem(row, 2, prompt_item)
            
            row += 1
            
    def _on_cell_double_clicked(self, row, column):
        """Handle double-click - column 1 for model, column 2 for prompt."""
        tool_item = self.tool_table.item(row, 0)
        if not tool_item:
            return
            
        tool_id = tool_item.data(Qt.ItemDataRole.UserRole)
        tool_name = TOOL_INFO.get(tool_id, {}).get("name", tool_id)
        
        if column == 1:
            # Model selection
            current_model = self.assignments.get(tool_id, "")
            dialog = ModelSelectDialog(tool_name, current_model, self)
            if dialog.exec_() == QDialog.DialogCode.Accepted:
                self.assignments[tool_id] = dialog.selected_model
                self._update_tool_row(tool_id, row)
                self.status_label.setText(f"Changed {tool_name} model - save!")
                self.status_label.setStyleSheet("color: #f39c12; font-style: italic;")
        elif column == 2:
            # Prompt selection
            current_prompt = self.prompts.get(tool_id, "")
            dialog = PromptSelectDialog(tool_name, current_prompt, self)
            if dialog.exec_() == QDialog.DialogCode.Accepted:
                self.prompts[tool_id] = dialog.selected_prompt
                self._update_tool_row(tool_id, row)
                self.status_label.setText(f"Changed {tool_name} prompt - save!")
                self.status_label.setStyleSheet("color: #f39c12; font-style: italic;")
            
    def _update_tool_row(self, tool_id: str, row: int = None):
        """Update a single row in the table."""
        if row is None:
            # Find the row
            for r in range(self.tool_table.rowCount()):
                item = self.tool_table.item(r, 0)
                if item and item.data(Qt.ItemDataRole.UserRole) == tool_id:
                    row = r
                    break
            if row is None:
                return
                
        # Update model column
        model = self.assignments.get(tool_id, "")
        model_item = self.tool_table.item(row, 1)
        if model:
            model_item.setText(model)
            model_item.setForeground(QColor("#2ecc71"))
        else:
            model_item.setText("(default)")
            model_item.setForeground(QColor("#666"))
            
        # Update prompt column
        prompt = self.prompts.get(tool_id, "")
        prompt_item = self.tool_table.item(row, 2)
        if prompt:
            prompt_item.setText(prompt)
            prompt_item.setForeground(QColor("#9b59b6"))
        else:
            prompt_item.setText("(default)")
            prompt_item.setForeground(QColor("#666"))
            
    def _load_config(self):
        """Load routing configuration from file."""
        self.assignments.clear()
        self.prompts.clear()
        
        try:
            from enigma_engine.config import CONFIG
            config_path = Path(CONFIG.get("data_dir", "data")) / "tool_routing.json"
            
            if config_path.exists():
                with open(config_path) as f:
                    data = json.load(f)
                    self.assignments = data.get("assignments", {})
                    self.prompts = data.get("prompts", {})
                    
            # Update all rows
            for row in range(self.tool_table.rowCount()):
                item = self.tool_table.item(row, 0)
                if item:
                    tool_id = item.data(Qt.ItemDataRole.UserRole)
                    self._update_tool_row(tool_id, row)
                    
            self.status_label.setText("Configuration loaded")
            self.status_label.setStyleSheet("color: #bac2de; font-style: italic;")
            
        except Exception as e:
            self.status_label.setText(f"Error loading: {e}")
            self.status_label.setStyleSheet("color: #e74c3c; font-style: italic;")
            
    def _save_config(self):
        """Save routing configuration to file."""
        try:
            from enigma_engine.config import CONFIG
            config_path = Path(CONFIG.get("data_dir", "data")) / "tool_routing.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, "w") as f:
                json.dump({
                    "assignments": self.assignments,
                    "prompts": self.prompts
                }, f, indent=2)
                
            self.status_label.setText("Saved!")
            self.status_label.setStyleSheet("color: #2ecc71; font-style: italic;")
            
        except Exception as e:
            self.status_label.setText(f"Error saving: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")
            
    def _reset_defaults(self):
        """Clear all assignments."""
        reply = QMessageBox.question(
            self, "Clear All?",
            "Clear all model and prompt assignments?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.assignments.clear()
            self.prompts.clear()
            for row in range(self.tool_table.rowCount()):
                item = self.tool_table.item(row, 0)
                if item:
                    tool_id = item.data(Qt.ItemDataRole.UserRole)
                    self._update_tool_row(tool_id, row)
            self.status_label.setText("Cleared - remember to save!")
            self.status_label.setStyleSheet("color: #f39c12; font-style: italic;")
            
    def refresh_models(self):
        """Refresh (called from parent when models change)."""
        pass  # No dropdowns to refresh anymore
