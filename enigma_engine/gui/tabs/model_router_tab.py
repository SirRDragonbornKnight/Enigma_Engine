"""
Model Router Tab - Assign models to tools.

Clean, modern interface for configuring which AI models handle which tasks.
"""

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .shared_components import NoScrollComboBox

# Tool categories and styling
TOOL_CATEGORIES = {
    "Generation": {
        "color": "#e91e63",
        "tools": ["chat", "image", "code", "video", "audio", "3d", "gif"]
    },
    "Code (Specialized)": {
        "color": "#f39c12",
        "tools": ["code_python", "code_javascript", "code_rust", "code_cpp", "code_java"]
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
    }
}

TOOL_INFO = {
    "chat": {"name": "Chat", "desc": "Text conversation and reasoning"},
    "image": {"name": "Image Gen", "desc": "Generate images from text"},
    "code": {"name": "Code Gen", "desc": "Generate and edit code (general)"},
    "code_python": {"name": "Python", "desc": "Specialized Python code generation"},
    "code_javascript": {"name": "JavaScript", "desc": "Specialized JS/TS code generation"},
    "code_rust": {"name": "Rust", "desc": "Specialized Rust code generation"},
    "code_cpp": {"name": "C/C++", "desc": "Specialized C/C++ code generation"},
    "code_java": {"name": "Java", "desc": "Specialized Java/Kotlin code generation"},
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
}

# Model presets by category
MODEL_PRESETS = {
    "Local Modules": [
        ("local:stable-diffusion", "Stable Diffusion (Image)"),
        ("local:code", "Local Code Generator"),
        ("local:tts", "Local TTS"),
        ("local:video", "AnimateDiff (Video)"),
        ("local:3d", "Shap-E (3D)"),
    ],
    "Chat/LLM Models": [
        ("huggingface:Qwen/Qwen2.5-3B-Instruct", "Qwen 2.5 3B"),
        ("huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B"),
        ("huggingface:microsoft/phi-2", "Phi-2"),
        ("huggingface:mistralai/Mistral-7B-Instruct-v0.2", "Mistral 7B"),
    ],
    "Code Models - General": [
        ("huggingface:Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen Coder 1.5B"),
        ("huggingface:bigcode/starcoder2-3b", "StarCoder2 3B"),
        ("huggingface:Salesforce/codegen-350M-mono", "CodeGen 350M Mono"),
        ("huggingface:WizardLMTeam/WizardCoder-15B-V1.0", "WizardCoder 15B"),
    ],
    "Code Models - Python": [
        ("huggingface:Salesforce/codegen-350M-mono", "CodeGen 350M Python"),
        ("huggingface:bigcode/starcoder2-3b", "StarCoder2 3B"),
        ("huggingface:replit/replit-code-v1-3b", "Replit Code 3B"),
    ],
    "Code Models - JavaScript": [
        ("huggingface:Salesforce/codegen-350M-multi", "CodeGen 350M Multi"),
        ("huggingface:bigcode/starcoder2-3b", "StarCoder2 3B"),
    ],
    "Code Models - Rust": [
        ("huggingface:bigcode/starcoder2-3b", "StarCoder2 3B"),
        ("huggingface:Salesforce/codegen-350M-multi", "CodeGen 350M Multi"),
    ],
    "Image Models": [
        ("huggingface:nota-ai/bk-sdm-small", "BK-SDM Small (Fast)"),
        ("huggingface:stabilityai/sd-turbo", "SD Turbo"),
        ("huggingface:stabilityai/stable-diffusion-xl-base-1.0", "SDXL Base"),
    ],
    "Audio Models": [
        ("huggingface:hexgrad/Kokoro-82M", "Kokoro TTS 82M"),
        ("huggingface:suno/bark-small", "Bark Small"),
        ("huggingface:myshell-ai/MeloTTS-English", "MeloTTS"),
    ],
    "Vision Models": [
        ("huggingface:microsoft/Florence-2-base", "Florence-2 Base"),
        ("huggingface:Salesforce/blip2-opt-2.7b", "BLIP-2"),
    ],
    "Embedding Models": [
        ("huggingface:sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6"),
        ("huggingface:BAAI/bge-small-en-v1.5", "BGE Small"),
    ],
    "API Providers": [
        ("api:openai", "OpenAI API"),
        ("api:replicate", "Replicate API"),
        ("api:elevenlabs", "ElevenLabs API"),
    ],
}


class ModelRouterTab(QWidget):
    """Modern model router configuration tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.assignments: dict[str, list[dict]] = {}  # tool -> [{model_id, priority}]
        self._setup_ui()
        self._load_config()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("Model Router")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        header.addWidget(title)
        
        header.addStretch()
        
        # Quick assign dropdown
        header.addWidget(QLabel("Quick Assign:"))
        self.quick_model = NoScrollComboBox()
        self.quick_model.setMinimumWidth(200)
        self._populate_model_dropdown(self.quick_model)
        header.addWidget(self.quick_model)
        
        self.quick_tool = NoScrollComboBox()
        self.quick_tool.setMinimumWidth(120)
        for tool_id, info in TOOL_INFO.items():
            self.quick_tool.addItem(info["name"], tool_id)
        header.addWidget(self.quick_tool)
        
        quick_add_btn = QPushButton("Assign")
        quick_add_btn.setStyleSheet("""
            QPushButton {
                background: #2ecc71;
                color: #1e1e2e;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background: #27ae60; }
        """)
        quick_add_btn.clicked.connect(self._quick_assign)
        header.addWidget(quick_add_btn)
        
        layout.addLayout(header)
        
        # Main content with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Tool list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        left_layout.addWidget(QLabel("Tools"))
        
        self.tool_table = QTableWidget()
        self.tool_table.setColumnCount(3)
        self.tool_table.setHorizontalHeaderLabels(["Tool", "Model", "Status"])
        self.tool_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tool_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tool_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tool_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tool_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tool_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tool_table.itemSelectionChanged.connect(self._on_tool_selected)
        self.tool_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #444;
                border-radius: 4px;
                background: #1a1a2e;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::item:selected {
                background: #3d5a80;
            }
            QHeaderView::section {
                background: #2d2d44;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
        left_layout.addWidget(self.tool_table)
        
        splitter.addWidget(left_panel)
        
        # Right side - Assignment details
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 0, 0, 0)
        
        self.detail_title = QLabel("Select a tool")
        self.detail_title.setStyleSheet("font-size: 13px; font-weight: bold;")
        right_layout.addWidget(self.detail_title)
        
        self.detail_desc = QLabel("")
        self.detail_desc.setStyleSheet("color: #bac2de; font-size: 11px;")
        self.detail_desc.setWordWrap(True)
        right_layout.addWidget(self.detail_desc)
        
        # Assigned models list
        right_layout.addWidget(QLabel("Assigned Models (higher priority tried first):"))
        
        self.assigned_list = QListWidget()
        self.assigned_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #444;
                border-radius: 4px;
                background: #1a1a2e;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333;
            }
            QListWidget::item:selected {
                background: #3d5a80;
            }
        """)
        self.assigned_list.setMinimumHeight(150)
        right_layout.addWidget(self.assigned_list)
        
        # Add model controls
        add_frame = QFrame()
        add_frame.setStyleSheet("QFrame { background: #2d2d44; border-radius: 4px; padding: 6px; }")
        add_layout = QVBoxLayout(add_frame)
        add_layout.setSpacing(6)
        
        add_layout.addWidget(QLabel("Add Model:"))
        
        model_row = QHBoxLayout()
        self.model_combo = NoScrollComboBox()
        self._populate_model_dropdown(self.model_combo)
        self.model_combo.setMinimumWidth(200)
        model_row.addWidget(self.model_combo, stretch=1)
        
        model_row.addWidget(QLabel("Priority:"))
        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(1, 100)
        self.priority_spin.setValue(50)
        self.priority_spin.setToolTip("Higher = tried first (100=primary, 50=backup)")
        self.priority_spin.setFixedWidth(60)
        model_row.addWidget(self.priority_spin)
        
        add_layout.addLayout(model_row)
        
        btn_row = QHBoxLayout()
        
        add_btn = QPushButton("Add Model")
        add_btn.setStyleSheet("""
            QPushButton {
                background: #2ecc71;
                color: #1e1e2e;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background: #27ae60; }
        """)
        add_btn.clicked.connect(self._add_model)
        btn_row.addWidget(add_btn)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.setStyleSheet("""
            QPushButton {
                background: #e74c3c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background: #c0392b; }
        """)
        remove_btn.clicked.connect(self._remove_model)
        btn_row.addWidget(remove_btn)
        
        btn_row.addStretch()
        add_layout.addLayout(btn_row)
        
        right_layout.addWidget(add_frame)
        right_layout.addStretch()
        
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 400])
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: #444;
                width: 4px;
            }
            QSplitter::handle:hover {
                background: #666;
            }
        """)
        
        layout.addWidget(splitter)
        
        # Bottom buttons
        bottom = QHBoxLayout()
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #bac2de; font-style: italic;")
        bottom.addWidget(self.status_label)
        
        bottom.addStretch()
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_config)
        bottom.addWidget(refresh_btn)
        
        reset_btn = QPushButton("Reset Defaults")
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
        
        save_btn = QPushButton("Save Configuration")
        save_btn.setStyleSheet("""
            QPushButton {
                background: #007bb5;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background: #0095d9; }
        """)
        save_btn.clicked.connect(self._save_config)
        bottom.addWidget(save_btn)
        
        layout.addLayout(bottom)
        
        # Populate tool table
        self._populate_tool_table()
        
    def _populate_model_dropdown(self, combo: NoScrollComboBox):
        """Populate a model dropdown with presets."""
        combo.clear()
        combo.addItem("Select a model...", "")
        
        for category, models in MODEL_PRESETS.items():
            combo.addItem(f"-- {category} --", "")
            for model_id, display_name in models:
                combo.addItem(f"  {display_name}", model_id)
        
        # Add Forge models
        combo.addItem("-- Forge Models --", "")
        try:
            from enigma_engine.core.model_registry import ModelRegistry
            registry = ModelRegistry()
            for model in registry.list_models():
                model_name = model.get("name", model) if isinstance(model, dict) else str(model)
                combo.addItem(f"  {model_name}", f"forge:{model_name}")
        except Exception:
            combo.addItem("  default", "forge:default")
            
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
            self.tool_table.setItem(row, 0, name_item)
            
            # Model (placeholder)
            model_item = QTableWidgetItem("No model")
            model_item.setForeground(QColor("#666"))
            self.tool_table.setItem(row, 1, model_item)
            
            # Status
            status_item = QTableWidgetItem("--")
            status_item.setForeground(QColor("#666"))
            self.tool_table.setItem(row, 2, status_item)
            
            row += 1
            
    def _update_tool_row(self, tool_id: str):
        """Update a single tool row in the table."""
        for row in range(self.tool_table.rowCount()):
            item = self.tool_table.item(row, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == tool_id:
                assigns = self.assignments.get(tool_id, [])
                
                if assigns:
                    # Show primary model
                    sorted_assigns = sorted(assigns, key=lambda x: -x.get("priority", 0))
                    primary = sorted_assigns[0]["model_id"]
                    # Shorten display
                    display = primary.split(":")[-1]
                    if len(display) > 20:
                        display = display[:18] + "..."
                    
                    model_item = QTableWidgetItem(display)
                    model_item.setForeground(QColor("#cdd6f4"))
                    model_item.setToolTip(f"Primary: {primary}\nTotal: {len(assigns)} model(s)")
                    self.tool_table.setItem(row, 1, model_item)
                    
                    status_item = QTableWidgetItem("Ready")
                    status_item.setForeground(QColor("#2ecc71"))
                    self.tool_table.setItem(row, 2, status_item)
                else:
                    model_item = QTableWidgetItem("No model")
                    model_item.setForeground(QColor("#666"))
                    self.tool_table.setItem(row, 1, model_item)
                    
                    status_item = QTableWidgetItem("--")
                    status_item.setForeground(QColor("#666"))
                    self.tool_table.setItem(row, 2, status_item)
                break
                
    def _on_tool_selected(self):
        """Handle tool selection."""
        items = self.tool_table.selectedItems()
        if not items:
            return
            
        tool_id = items[0].data(Qt.ItemDataRole.UserRole)
        if not tool_id:
            return
            
        info = TOOL_INFO.get(tool_id, {"name": tool_id, "desc": ""})
        
        self.detail_title.setText(info["name"])
        self.detail_desc.setText(info["desc"])
        
        # Update assigned list
        self._refresh_assigned_list(tool_id)
        
    def _refresh_assigned_list(self, tool_id: str):
        """Refresh the assigned models list for a tool."""
        self.assigned_list.clear()
        
        assigns = self.assignments.get(tool_id, [])
        sorted_assigns = sorted(assigns, key=lambda x: -x.get("priority", 0))
        
        for assign in sorted_assigns:
            model_id = assign["model_id"]
            priority = assign.get("priority", 50)
            
            # Determine icon/color
            if model_id.startswith("forge:"):
                prefix = "[Forge]"
                color = "#3498db"
            elif model_id.startswith("huggingface:"):
                prefix = "[HF]"
                color = "#f39c12"
            elif model_id.startswith("local:"):
                prefix = "[Local]"
                color = "#2ecc71"
            elif model_id.startswith("api:"):
                prefix = "[API]"
                color = "#e91e63"
            else:
                prefix = "[?]"
                color = "#bac2de"
                
            display = model_id.split(":")[-1]
            item = QListWidgetItem(f"{prefix} {display}  [Priority: {priority}]")
            item.setData(Qt.ItemDataRole.UserRole, model_id)
            item.setForeground(QColor(color))
            item.setToolTip(f"Full ID: {model_id}\nPriority: {priority}")
            self.assigned_list.addItem(item)
            
    def _get_selected_tool(self) -> Optional[str]:
        """Get currently selected tool ID."""
        items = self.tool_table.selectedItems()
        if items:
            return items[0].data(Qt.ItemDataRole.UserRole)
        return None
        
    def _add_model(self):
        """Add a model to the selected tool."""
        tool_id = self._get_selected_tool()
        if not tool_id:
            QMessageBox.warning(self, "No Tool Selected", "Please select a tool first.")
            return
            
        model_id = self.model_combo.currentData()
        if not model_id:
            QMessageBox.warning(self, "No Model Selected", "Please select a model to add.")
            return
            
        priority = self.priority_spin.value()
        
        # Check if already assigned
        if tool_id not in self.assignments:
            self.assignments[tool_id] = []
            
        for assign in self.assignments[tool_id]:
            if assign["model_id"] == model_id:
                QMessageBox.warning(self, "Already Assigned", 
                    f"This model is already assigned to {tool_id}.")
                return
                
        self.assignments[tool_id].append({
            "model_id": model_id,
            "priority": priority
        })
        
        self._refresh_assigned_list(tool_id)
        self._update_tool_row(tool_id)
        self.status_label.setText(f"Added model to {tool_id} - remember to save!")
        self.status_label.setStyleSheet("color: #f39c12; font-style: italic;")
        
    def _remove_model(self):
        """Remove selected model from the tool."""
        tool_id = self._get_selected_tool()
        if not tool_id:
            return
            
        current = self.assigned_list.currentItem()
        if not current:
            QMessageBox.warning(self, "No Model Selected", "Please select a model to remove.")
            return
            
        model_id = current.data(Qt.ItemDataRole.UserRole)
        
        if tool_id in self.assignments:
            self.assignments[tool_id] = [
                a for a in self.assignments[tool_id] 
                if a["model_id"] != model_id
            ]
            
        self._refresh_assigned_list(tool_id)
        self._update_tool_row(tool_id)
        self.status_label.setText(f"Removed model from {tool_id} - remember to save!")
        self.status_label.setStyleSheet("color: #f39c12; font-style: italic;")
        
    def _quick_assign(self):
        """Quick assign model to tool."""
        model_id = self.quick_model.currentData()
        tool_id = self.quick_tool.currentData()
        
        if not model_id or not tool_id:
            return
            
        if tool_id not in self.assignments:
            self.assignments[tool_id] = []
            
        # Check if already assigned
        for assign in self.assignments[tool_id]:
            if assign["model_id"] == model_id:
                QMessageBox.information(self, "Already Assigned", 
                    f"This model is already assigned to {TOOL_INFO[tool_id]['name']}.")
                return
                
        self.assignments[tool_id].append({
            "model_id": model_id,
            "priority": 50
        })
        
        self._update_tool_row(tool_id)
        
        # Select the tool to show details
        for row in range(self.tool_table.rowCount()):
            item = self.tool_table.item(row, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == tool_id:
                self.tool_table.selectRow(row)
                break
                
        self.status_label.setText(f"Assigned to {TOOL_INFO[tool_id]['name']} - remember to save!")
        self.status_label.setStyleSheet("color: #2ecc71; font-style: italic;")
        
    def _load_config(self):
        """Load routing configuration."""
        self.assignments.clear()
        
        try:
            from enigma_engine.core.tool_router import get_router
            router = get_router()
            
            for tool_id in TOOL_INFO:
                try:
                    assigns = router.get_assignments(tool_id)
                    self.assignments[tool_id] = [
                        {"model_id": a.model_id, "priority": a.priority}
                        for a in assigns
                    ]
                except Exception:
                    self.assignments[tool_id] = []
                    
            # Update all rows
            for tool_id in TOOL_INFO:
                self._update_tool_row(tool_id)
                
            self.status_label.setText("Configuration loaded")
            self.status_label.setStyleSheet("color: #bac2de; font-style: italic;")
            
        except Exception as e:
            self.status_label.setText(f"Error loading config: {e}")
            self.status_label.setStyleSheet("color: #e74c3c; font-style: italic;")
            
    def _save_config(self):
        """Save routing configuration."""
        try:
            from enigma_engine.core.tool_router import get_router
            router = get_router()
            
            for tool_id, assigns in self.assignments.items():
                # Clear existing
                try:
                    for assign in router.get_assignments(tool_id):
                        router.unassign_model(tool_id, assign.model_id)
                except Exception:
                    pass  # Intentionally silent
                    
                # Add new
                for assign in assigns:
                    try:
                        router.assign_model(
                            tool_id, 
                            assign["model_id"],
                            priority=assign.get("priority", 50)
                        )
                    except Exception:
                        pass  # Intentionally silent
                        
            self.status_label.setText("Configuration saved!")
            self.status_label.setStyleSheet("color: #2ecc71; font-style: italic;")
            
            QMessageBox.information(self, "Saved", "Model routing configuration saved successfully!")
            
        except Exception as e:
            self.status_label.setText(f"Error saving: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {e}")
            
    def _reset_defaults(self):
        """Reset to default configuration."""
        reply = QMessageBox.question(
            self, "Reset?",
            "Reset all tool assignments to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from enigma_engine.core.tool_router import get_router
                router = get_router()
                router._set_defaults()
                self._load_config()
                self.status_label.setText("Reset to defaults")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to reset: {e}")
                
    def refresh_models(self):
        """Refresh model dropdowns."""
        self._populate_model_dropdown(self.quick_model)
        self._populate_model_dropdown(self.model_combo)
        self.status_label.setText("Model list refreshed")
