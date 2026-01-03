"""
Model Router Tab - Assign models to tools.

This tab lets you:
  - See all available tools (chat, image, code, etc.)
  - Assign models to each tool (Enigma, HuggingFace, local, API)
  - Set priorities for fallback
  - Add HuggingFace models by ID
"""

from typing import Dict, List, Optional, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QGroupBox, QScrollArea, QFrame,
    QListWidget, QListWidgetItem, QSpinBox, QMessageBox,
    QGridLayout, QSizePolicy, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


# Tool icons/colors
TOOL_STYLES = {
    "chat": {"color": "#3498db", "icon": "💬"},
    "image": {"color": "#e91e63", "icon": "🎨"},
    "code": {"color": "#9b59b6", "icon": "💻"},
    "video": {"color": "#e74c3c", "icon": "🎬"},
    "audio": {"color": "#f39c12", "icon": "🔊"},
    "3d": {"color": "#1abc9c", "icon": "🎲"},
    "web": {"color": "#2ecc71", "icon": "🌐"},
    "memory": {"color": "#34495e", "icon": "🧠"},
}


class ToolAssignmentWidget(QFrame):
    """Widget showing assignments for one tool."""
    
    assignment_changed = pyqtSignal(str)  # tool_name
    
    def __init__(self, tool_name: str, tool_def, parent=None):
        super().__init__(parent)
        self.tool_name = tool_name
        self.tool_def = tool_def
        self.assignments: List[Dict] = []
        self._setup_ui()
        
    def _setup_ui(self):
        style = TOOL_STYLES.get(self.tool_name, {"color": "#888", "icon": "⚙"})
        
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet(f"""
            ToolAssignmentWidget {{
                border: 2px solid {style['color']};
                border-radius: 8px;
                background: rgba(0,0,0,0.2);
                margin: 5px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Header
        header = QHBoxLayout()
        
        icon_label = QLabel(style['icon'])
        icon_label.setStyleSheet("font-size: 20px;")
        header.addWidget(icon_label)
        
        name_label = QLabel(self.tool_name.upper())
        name_label.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {style['color']};")
        header.addWidget(name_label)
        
        header.addStretch()
        
        # Status indicator
        self.status_label = QLabel("No model")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        header.addWidget(self.status_label)
        
        layout.addLayout(header)
        
        # Description
        desc_label = QLabel(self.tool_def.description)
        desc_label.setStyleSheet("color: #aaa; font-size: 10px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Assigned models list
        self.model_list = QListWidget()
        self.model_list.setMaximumHeight(80)
        self.model_list.setStyleSheet("""
            QListWidget {
                background: rgba(0,0,0,0.3);
                border: 1px solid #333;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:selected {
                background: rgba(255,255,255,0.1);
            }
        """)
        layout.addWidget(self.model_list)
        
        # Add model controls
        add_layout = QHBoxLayout()
        
        self.model_input = QComboBox()
        self.model_input.setEditable(True)
        self.model_input.setPlaceholderText("Select or enter model...")
        self.model_input.setMinimumWidth(200)
        self._populate_model_options()
        add_layout.addWidget(self.model_input)
        
        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(1, 100)
        self.priority_spin.setValue(10)
        self.priority_spin.setPrefix("Priority: ")
        self.priority_spin.setFixedWidth(100)
        add_layout.addWidget(self.priority_spin)
        
        add_btn = QPushButton("+")
        add_btn.setFixedWidth(30)
        add_btn.clicked.connect(self._add_model)
        add_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("-")
        remove_btn.setFixedWidth(30)
        remove_btn.clicked.connect(self._remove_model)
        add_layout.addWidget(remove_btn)
        
        layout.addLayout(add_layout)
        
    def _populate_model_options(self):
        """Populate model dropdown with available options."""
        self.model_input.clear()
        
        # Add categories
        self.model_input.addItem("── Enigma Models ──")
        
        # Get Enigma models
        try:
            from enigma.core.model_registry import ModelRegistry
            registry = ModelRegistry()
            for model in registry.list_models():
                self.model_input.addItem(f"enigma:{model['name']}")
        except Exception:
            self.model_input.addItem("enigma:default")
            
        self.model_input.addItem("── HuggingFace Models ──")
        # Popular HuggingFace models for text
        hf_models = [
            "huggingface:mistralai/Mistral-7B-Instruct-v0.2",
            "huggingface:microsoft/phi-2",
            "huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "huggingface:google/gemma-2b-it",
        ]
        for model in hf_models:
            self.model_input.addItem(model)
            
        self.model_input.addItem("── Local Modules ──")
        local_modules = [
            "local:stable-diffusion",
            "local:animatediff", 
            "local:tts",
            "local:shap-e",
            "local:web-tools",
            "local:memory",
        ]
        for module in local_modules:
            self.model_input.addItem(module)
            
        self.model_input.addItem("── API Providers ──")
        api_providers = [
            "api:openai",
            "api:replicate",
            "api:elevenlabs",
        ]
        for provider in api_providers:
            self.model_input.addItem(provider)
            
        # Reset to empty
        self.model_input.setCurrentText("")
        
    def _add_model(self):
        """Add a model assignment."""
        model_id = self.model_input.currentText().strip()
        if not model_id or model_id.startswith("──"):
            return
            
        priority = self.priority_spin.value()
        
        # Check if already assigned
        for assign in self.assignments:
            if assign["model_id"] == model_id:
                QMessageBox.warning(self, "Already Assigned", 
                    f"Model {model_id} is already assigned to this tool.")
                return
                
        self.assignments.append({
            "model_id": model_id,
            "priority": priority
        })
        
        self._refresh_list()
        self.assignment_changed.emit(self.tool_name)
        
    def _remove_model(self):
        """Remove selected model assignment."""
        current = self.model_list.currentItem()
        if not current:
            return
            
        model_id = current.data(Qt.UserRole)
        self.assignments = [a for a in self.assignments if a["model_id"] != model_id]
        
        self._refresh_list()
        self.assignment_changed.emit(self.tool_name)
        
    def _refresh_list(self):
        """Refresh the assignments list display."""
        self.model_list.clear()
        
        # Sort by priority
        sorted_assigns = sorted(self.assignments, key=lambda x: -x.get("priority", 0))
        
        for assign in sorted_assigns:
            model_id = assign["model_id"]
            priority = assign.get("priority", 10)
            
            # Determine type and color
            if model_id.startswith("enigma:"):
                color = "#3498db"
                icon = "🔷"
            elif model_id.startswith("huggingface:"):
                color = "#f39c12"
                icon = "🤗"
            elif model_id.startswith("local:"):
                color = "#2ecc71"
                icon = "💻"
            elif model_id.startswith("api:"):
                color = "#e91e63"
                icon = "☁️"
            else:
                color = "#888"
                icon = "?"
                
            item = QListWidgetItem(f"{icon} {model_id} (P:{priority})")
            item.setData(Qt.UserRole, model_id)
            item.setForeground(Qt.white)
            self.model_list.addItem(item)
            
        # Update status
        if self.assignments:
            self.status_label.setText(f"{len(self.assignments)} model(s)")
            self.status_label.setStyleSheet("color: #2ecc71; font-size: 11px;")
        else:
            self.status_label.setText("No model")
            self.status_label.setStyleSheet("color: #e74c3c; font-size: 11px;")
            
    def set_assignments(self, assignments: List[Dict]):
        """Set assignments from loaded config."""
        self.assignments = assignments
        self._refresh_list()
        
    def get_assignments(self) -> List[Dict]:
        """Get current assignments."""
        return self.assignments


class ModelRouterTab(QWidget):
    """Tab for configuring model-to-tool routing."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tool_widgets: Dict[str, ToolAssignmentWidget] = {}
        self._setup_ui()
        self._load_config()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("Model Router")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        header.addWidget(title)
        
        header.addStretch()
        
        # Save button
        save_btn = QPushButton("Save Configuration")
        save_btn.clicked.connect(self._save_config)
        header.addWidget(save_btn)
        
        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        header.addWidget(reset_btn)
        
        layout.addLayout(header)
        
        # Description
        desc = QLabel(
            "Assign AI models to each tool. Higher priority models are tried first. "
            "You can assign multiple models for fallback support."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Add HuggingFace model section
        hf_group = QGroupBox("Add HuggingFace Model")
        hf_layout = QHBoxLayout(hf_group)
        
        hf_layout.addWidget(QLabel("Model ID:"))
        self.hf_input = QLineEdit()
        self.hf_input.setPlaceholderText("e.g., mistralai/Mistral-7B-Instruct-v0.2")
        hf_layout.addWidget(self.hf_input)
        
        hf_add_btn = QPushButton("Add to List")
        hf_add_btn.clicked.connect(self._add_hf_model)
        hf_layout.addWidget(hf_add_btn)
        
        layout.addWidget(hf_group)
        
        # Scroll area for tools
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
        
        tools_container = QWidget()
        tools_layout = QGridLayout(tools_container)
        tools_layout.setSpacing(10)
        
        # Create tool widgets
        try:
            from enigma.core.tool_router import TOOL_DEFINITIONS
            tools = TOOL_DEFINITIONS
        except ImportError:
            tools = {}
            
        row, col = 0, 0
        for tool_name, tool_def in tools.items():
            widget = ToolAssignmentWidget(tool_name, tool_def)
            widget.assignment_changed.connect(self._on_assignment_changed)
            self.tool_widgets[tool_name] = widget
            tools_layout.addWidget(widget, row, col)
            
            col += 1
            if col >= 2:  # 2 columns
                col = 0
                row += 1
                
        tools_layout.setRowStretch(row + 1, 1)
        
        scroll.setWidget(tools_container)
        layout.addWidget(scroll)
        
        # Status bar
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self.status_label)
        
    def _load_config(self):
        """Load routing configuration."""
        try:
            from enigma.core.tool_router import get_router
            router = get_router()
            
            for tool_name, widget in self.tool_widgets.items():
                assignments = router.get_assignments(tool_name)
                widget.set_assignments([
                    {"model_id": a.model_id, "priority": a.priority}
                    for a in assignments
                ])
                
            self.status_label.setText("Configuration loaded")
        except Exception as e:
            self.status_label.setText(f"Error loading config: {e}")
            
    def _save_config(self):
        """Save routing configuration."""
        try:
            from enigma.core.tool_router import get_router
            router = get_router()
            
            for tool_name, widget in self.tool_widgets.items():
                # Clear existing
                for assign in router.get_assignments(tool_name):
                    router.unassign_model(tool_name, assign.model_id)
                    
                # Add new
                for assign in widget.get_assignments():
                    router.assign_model(
                        tool_name, 
                        assign["model_id"],
                        priority=assign.get("priority", 10)
                    )
                    
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
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                from enigma.core.tool_router import get_router
                router = get_router()
                router._set_defaults()
                self._load_config()
                self.status_label.setText("Reset to defaults")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to reset: {e}")
                
    def _add_hf_model(self):
        """Add a HuggingFace model to all dropdowns."""
        model_id = self.hf_input.text().strip()
        if not model_id:
            return
            
        # Format as huggingface:repo/model
        if not model_id.startswith("huggingface:"):
            model_id = f"huggingface:{model_id}"
            
        # Add to all tool widgets
        for widget in self.tool_widgets.values():
            widget.model_input.addItem(model_id)
            
        self.hf_input.clear()
        self.status_label.setText(f"Added {model_id} to model list")
        
    def _on_assignment_changed(self, tool_name: str):
        """Handle assignment change in a tool widget."""
        self.status_label.setText(f"Modified {tool_name} - remember to save!")
        self.status_label.setStyleSheet("color: #f39c12; font-style: italic;")
