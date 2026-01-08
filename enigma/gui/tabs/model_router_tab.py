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
    "chat": {"color": "#3498db", "icon": "[C]"},
    "image": {"color": "#e91e63", "icon": "[I]"},
    "code": {"color": "#9b59b6", "icon": "[<>]"},
    "video": {"color": "#e74c3c", "icon": "[V]"},
    "audio": {"color": "#f39c12", "icon": "[A]"},
    "3d": {"color": "#1abc9c", "icon": "[3D]"},
    "web": {"color": "#2ecc71", "icon": "[W]"},
    "memory": {"color": "#34495e", "icon": "[M]"},
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
        style = TOOL_STYLES.get(self.tool_name, {"color": "#888", "icon": "[?]"})
        
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
        self.priority_spin.setMinimumWidth(100)
        add_layout.addWidget(self.priority_spin)
        
        add_btn = QPushButton("Add")
        add_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        add_btn.setToolTip("Add selected model to this tool")
        add_btn.clicked.connect(self._add_model)
        add_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("Remove")
        remove_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        remove_btn.setToolTip("Remove selected model from this tool")
        remove_btn.clicked.connect(self._remove_model)
        add_layout.addWidget(remove_btn)
        
        layout.addLayout(add_layout)
        
    def _populate_model_options(self):
        """Populate model dropdown with available options."""
        self.model_input.clear()
        
        # Add categories
        self.model_input.addItem("-- Enigma Models --")
        
        # Get Enigma models
        try:
            from enigma.core.model_registry import ModelRegistry
            registry = ModelRegistry()
            for model in registry.list_models():
                model_name = model.get("name", model) if isinstance(model, dict) else str(model)
                self.model_input.addItem(f"enigma:{model_name}")
        except Exception:
            self.model_input.addItem("enigma:default")
            
        self.model_input.addItem("-- HuggingFace Models --")
        # Get HuggingFace models from registry with size info
        try:
            from enigma.core.model_registry import ModelRegistry
            registry = ModelRegistry()
            for model_name, model_info in registry.registry.get("models", {}).items():
                if model_info.get("source") == "huggingface":
                    hf_id = model_info.get("huggingface_id", model_name)
                    size_str = model_info.get("size", "HF")
                    # Show size if available: "huggingface:model/id (HF-124M)"
                    if size_str and size_str != "huggingface":
                        self.model_input.addItem(f"huggingface:{hf_id} ({size_str})")
                    else:
                        self.model_input.addItem(f"huggingface:{hf_id}")
        except Exception:
            pass
        
        # Common HuggingFace presets with sizes
        self.model_input.addItem("huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B)")
        self.model_input.addItem("huggingface:Qwen/Qwen2-1.5B-Instruct (1.5B)")
        self.model_input.addItem("huggingface:Salesforce/codegen-350M-mono (350M)")
            
        self.model_input.addItem("-- API Providers --")
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
        if not model_id or model_id.startswith("--"):
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
            
        model_id = current.data(Qt.ItemDataRole.UserRole)
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
                icon = "[E]"
            elif model_id.startswith("huggingface:"):
                color = "#f39c12"
                icon = "[HF]"
            elif model_id.startswith("local:"):
                color = "#2ecc71"
                icon = "[L]"
            elif model_id.startswith("api:"):
                color = "#e91e63"
                icon = "[API]"
            else:
                color = "#888"
                icon = "[?]"
                
            item = QListWidgetItem(f"{icon} {model_id} (P:{priority})")
            item.setData(Qt.ItemDataRole.UserRole, model_id)
            item.setForeground(Qt.GlobalColor.white)
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
        
        # Quick assign section - Apply model to all tools
        quick_group = QGroupBox("Quick Assign - Apply to All Tools")
        quick_layout = QHBoxLayout(quick_group)
        
        quick_layout.addWidget(QLabel("Model:"))
        self.quick_model_combo = QComboBox()
        self.quick_model_combo.setMinimumWidth(250)
        self._populate_quick_combo()
        quick_layout.addWidget(self.quick_model_combo)
        
        apply_all_btn = QPushButton("⚡ Apply to ALL Tools")
        apply_all_btn.setStyleSheet("background-color: #f9e2af; color: #1e1e2e; font-weight: bold;")
        apply_all_btn.setToolTip("Assign this model to every tool at once")
        apply_all_btn.clicked.connect(self._apply_to_all_tools)
        quick_layout.addWidget(apply_all_btn)
        
        quick_layout.addStretch()
        layout.addWidget(quick_group)
        
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
    
    def refresh_models(self):
        """Refresh model dropdowns in all tool widgets."""
        for tool_name, widget in self.tool_widgets.items():
            widget._populate_model_options()
        # Also refresh the quick assign combo
        self._populate_quick_combo()
        self.status_label.setText("Model list refreshed")
            
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
        
        # Try to get model size info
        size_suffix = ""
        try:
            from enigma.core.huggingface_loader import get_huggingface_model_info
            # Extract just the model ID part (after "huggingface:")
            hf_id = model_id.replace("huggingface:", "")
            info = get_huggingface_model_info(hf_id)
            if not info.get("error"):
                size_suffix = f" ({info['size_str']})"
        except Exception:
            pass
        
        display_text = f"{model_id}{size_suffix}"
            
        # Add to all tool widgets
        for widget in self.tool_widgets.values():
            widget.model_input.addItem(display_text)
            
        self.hf_input.clear()
        self.status_label.setText(f"Added {display_text} to model list")
        
    def _on_assignment_changed(self, tool_name: str):
        """Handle assignment change in a tool widget."""
        self.status_label.setText(f"Modified {tool_name} - remember to save!")
        self.status_label.setStyleSheet("color: #f39c12; font-style: italic;")

    def _populate_quick_combo(self):
        """Populate the quick assign model dropdown."""
        self.quick_model_combo.clear()
        self.quick_model_combo.addItem("Select a model...")
        
        # Add Enigma models from registry
        try:
            from enigma.core.model_registry import ModelRegistry
            registry = ModelRegistry()
            for name, info in registry.registry.get("models", {}).items():
                size = info.get("size", "?")
                source = info.get("source", "enigma")
                if source == "huggingface":
                    hf_id = info.get("huggingface_id", name)
                    self.quick_model_combo.addItem(f"huggingface:{hf_id} [{size}]", f"huggingface:{hf_id}")
                else:
                    self.quick_model_combo.addItem(f"{name} [{size}]", name)
        except Exception as e:
            print(f"Error loading models for quick combo: {e}")
            
    def _apply_to_all_tools(self):
        """Apply selected model to all tools at once."""
        if self.quick_model_combo.currentIndex() == 0:
            QMessageBox.warning(self, "No Model", "Please select a model first")
            return
            
        model_id = self.quick_model_combo.currentData()
        if not model_id:
            model_id = self.quick_model_combo.currentText().split(" [")[0]
            
        reply = QMessageBox.question(
            self, "Apply to All?",
            f"Assign '{model_id}' to ALL tools?\n\n"
            "This will replace all existing assignments.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        try:
            from enigma.core.tool_router import get_router
            router = get_router()
            
            # Apply to all tools
            count = 0
            for tool_name, widget in self.tool_widgets.items():
                # Clear existing and set new
                router.assign_model(tool_name, model_id, priority=100)
                count += 1
                
            # Reload UI
            self._load_config()
            
            # Save configuration
            router._save_config()
            
            self.status_label.setText(f"Applied '{model_id}' to {count} tools")
            self.status_label.setStyleSheet("color: #2ecc71; font-style: italic;")
            
            QMessageBox.information(
                self, "Applied", 
                f"'{model_id}' has been assigned to all {count} tools and saved."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply: {e}")
