"""
Model Router Tab - Assign models to tools.

This tab lets you:
  - See all available tools (chat, image, code, etc.)
  - Assign models to each tool (Forge, HuggingFace, local, API)
  - Set priorities for fallback ordering
  
Priority: Higher numbers are tried first. Use 100 for primary, 50 for fallback.
"""

from typing import Dict, List, Optional, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QScrollArea, QFrame,
    QListWidget, QListWidgetItem, QSpinBox, QMessageBox,
    QGridLayout, QSizePolicy, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from .shared_components import NoScrollComboBox


# Tool icons/colors - Plain text for professional appearance
TOOL_STYLES = {
    "chat": {"color": "#3498db", "icon": "[Chat]"},
    "image": {"color": "#e91e63", "icon": "[Image]"},
    "code": {"color": "#9b59b6", "icon": "[Code]"},
    "video": {"color": "#e74c3c", "icon": "[Video]"},
    "audio": {"color": "#f39c12", "icon": "[Audio]"},
    "3d": {"color": "#1abc9c", "icon": "[3D]"},
    "gif": {"color": "#ff6b6b", "icon": "[GIF]"},
    "web": {"color": "#2ecc71", "icon": "[Web]"},
    "memory": {"color": "#34495e", "icon": "[Memory]"},
    "embeddings": {"color": "#9b59b6", "icon": "[Embed]"},
    "camera": {"color": "#00bcd4", "icon": "[Camera]"},
    "vision": {"color": "#ff9800", "icon": "[Vision]"},
    "avatar": {"color": "#e91e63", "icon": "[Avatar]"},
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
                margin: 2px;
            }}
        """)
        # Allow widget to shrink
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumWidth(200)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Header
        header = QHBoxLayout()
        header.setSpacing(4)
        
        icon_label = QLabel(style['icon'])
        icon_label.setStyleSheet("font-size: 16px;")
        header.addWidget(icon_label)
        
        name_label = QLabel(self.tool_name.upper())
        name_label.setStyleSheet(f"font-weight: bold; font-size: 12px; color: {style['color']};")
        header.addWidget(name_label)
        
        header.addStretch()
        
        # Status indicator
        self.status_label = QLabel("No model")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        header.addWidget(self.status_label)
        
        layout.addLayout(header)
        
        # Description (hidden when small to save space)
        desc_label = QLabel(self.tool_def.description[:60] + "..." if len(self.tool_def.description) > 60 else self.tool_def.description)
        desc_label.setStyleSheet("color: #aaa; font-size: 9px;")
        desc_label.setWordWrap(True)
        desc_label.setMaximumHeight(30)
        layout.addWidget(desc_label)
        
        # Assigned models list
        self.model_list = QListWidget()
        self.model_list.setMaximumHeight(60)
        self.model_list.setMinimumHeight(40)
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
        
        # Add model controls - more compact layout
        add_layout = QHBoxLayout()
        add_layout.setSpacing(4)
        
        self.model_input = NoScrollComboBox()
        self.model_input.setEditable(True)
        self.model_input.setPlaceholderText("Select model...")
        self.model_input.setMinimumWidth(120)
        self.model_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.model_input.setToolTip("Select or enter a model for this tool")
        self._populate_model_options()
        add_layout.addWidget(self.model_input)
        
        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(1, 100)
        self.priority_spin.setValue(50)
        self.priority_spin.setToolTip("Priority: Higher = tried first (100=primary, 50=backup)")
        self.priority_spin.setFixedWidth(55)
        add_layout.addWidget(self.priority_spin)
        
        add_btn = QPushButton("+")
        add_btn.setFixedWidth(28)
        add_btn.setToolTip("Add model")
        add_btn.clicked.connect(self._add_model)
        add_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("-")
        remove_btn.setFixedWidth(28)
        remove_btn.setToolTip("Remove model")
        remove_btn.clicked.connect(self._remove_model)
        add_layout.addWidget(remove_btn)
        
        layout.addLayout(add_layout)
        
    def _populate_model_options(self):
        """Populate model dropdown with available options."""
        self.model_input.clear()
        
        # Add local modules first - these are the direct implementations
        self.model_input.addItem("-- Local Modules --")
        local_modules = [
            "local:stable-diffusion",
            "local:code",
            "local:tts",
            "local:video",
            "local:3d",
        ]
        for module_id in local_modules:
            self.model_input.addItem(module_id)
        
        # Recommended HuggingFace models by task
        self.model_input.addItem("-- HF: Chat/LLM --")
        self.model_input.addItem("huggingface:Qwen/Qwen2.5-3B-Instruct")
        self.model_input.addItem("huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.model_input.addItem("huggingface:microsoft/phi-2")
        
        self.model_input.addItem("-- HF: Code --")
        self.model_input.addItem("huggingface:Salesforce/codegen-350M-mono")
        self.model_input.addItem("huggingface:bigcode/starcoder2-3b")
        self.model_input.addItem("huggingface:Qwen/Qwen2.5-Coder-1.5B-Instruct")
        
        self.model_input.addItem("-- HF: Image --")
        self.model_input.addItem("huggingface:stabilityai/sd-turbo")
        self.model_input.addItem("huggingface:nota-ai/bk-sdm-small")
        self.model_input.addItem("huggingface:stabilityai/stable-diffusion-xl-base-1.0")
        
        self.model_input.addItem("-- HF: Audio/TTS --")
        self.model_input.addItem("huggingface:hexgrad/Kokoro-82M")
        self.model_input.addItem("huggingface:myshell-ai/MeloTTS-English")
        self.model_input.addItem("huggingface:suno/bark-small")
        
        self.model_input.addItem("-- HF: Video --")
        self.model_input.addItem("huggingface:ali-vilab/text-to-video-ms-1.7b")
        self.model_input.addItem("huggingface:damo-vilab/text-to-video-ms-1.7b")
        
        self.model_input.addItem("-- HF: 3D --")
        self.model_input.addItem("huggingface:openai/shap-e")
        self.model_input.addItem("huggingface:openai/point-e")
        
        self.model_input.addItem("-- HF: Vision --")
        self.model_input.addItem("huggingface:Salesforce/blip2-opt-2.7b")
        self.model_input.addItem("huggingface:llava-hf/llava-1.5-7b-hf")
        self.model_input.addItem("huggingface:microsoft/Florence-2-base")
        
        self.model_input.addItem("-- HF: Embeddings --")
        self.model_input.addItem("huggingface:sentence-transformers/all-MiniLM-L6-v2")
        self.model_input.addItem("huggingface:BAAI/bge-small-en-v1.5")
        self.model_input.addItem("huggingface:thenlper/gte-small")
        
        # Add Forge models
        self.model_input.addItem("-- Forge Models --")
        try:
            from forge_ai.core.model_registry import ModelRegistry
            registry = ModelRegistry()
            for model in registry.list_models():
                model_name = model.get("name", model) if isinstance(model, dict) else str(model)
                self.model_input.addItem(f"forge:{model_name}")
        except Exception:
            self.model_input.addItem("forge:default")
            
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
            if model_id.startswith("forge:"):
                color = "#3498db"
                icon = "[F]"
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
                
            item = QListWidgetItem(f"{icon} {model_id.split(':')[-1]} [{priority}]")
            item.setData(Qt.ItemDataRole.UserRole, model_id)
            item.setToolTip(f"Full ID: {model_id}\nPriority: {priority} (higher = tried first)")
            item.setForeground(Qt.GlobalColor.white)
            self.model_list.addItem(item)
            
        # Update status
        if self.assignments:
            # Check actual readiness
            ready = self._check_tool_ready()
            if ready:
                self.status_label.setText(f"[OK] Ready ({len(self.assignments)})")
                self.status_label.setStyleSheet("color: #2ecc71; font-size: 11px; font-weight: bold;")
            else:
                self.status_label.setText(f"[!] Not Loaded ({len(self.assignments)})")
                self.status_label.setStyleSheet("color: #f39c12; font-size: 11px;")
        else:
            self.status_label.setText("[X] No model")
            self.status_label.setStyleSheet("color: #e74c3c; font-size: 11px;")
    
    def _check_tool_ready(self) -> bool:
        """Check if this tool's model is loaded and ready."""
        if not self.assignments:
            return False
        
        # Check based on tool type
        if self.tool_name == "image":
            try:
                from ..tabs.image_tab import get_provider
                provider = get_provider('local')
                return provider is not None and provider.is_loaded
            except Exception:
                return False
        
        elif self.tool_name == "code":
            try:
                from ..tabs.code_tab import get_provider
                provider = get_provider('local')
                return provider is not None and provider.is_loaded
            except Exception:
                return False
        
        elif self.tool_name == "audio":
            try:
                from ..tabs.audio_tab import get_provider
                provider = get_provider('local')
                return provider is not None and provider.is_loaded
            except Exception:
                return False
        
        elif self.tool_name == "video":
            try:
                from ..tabs.video_tab import get_provider
                provider = get_provider('local')
                return provider is not None and provider.is_loaded
            except Exception:
                return False
        
        elif self.tool_name == "3d":
            try:
                from ..tabs.threed_tab import get_provider
                provider = get_provider('local')
                return provider is not None and provider.is_loaded
            except Exception:
                return False
        
        elif self.tool_name == "chat":
            # Chat is ready if main engine is loaded
            try:
                return True  # Assume ready if model assigned
            except Exception:
                return False
        
        # For other tools, assume ready if model assigned
        return len(self.assignments) > 0
            
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
            "Assign AI models to each tool. Higher priority numbers are tried first. "
            "Select a model from the dropdown and click + to add, - to remove."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Scroll area for tools
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
        
        tools_container = QWidget()
        tools_layout = QVBoxLayout(tools_container)
        tools_layout.setSpacing(6)
        tools_layout.setContentsMargins(4, 4, 4, 4)
        
        # Create tool widgets in a single column (responsive)
        try:
            from forge_ai.core.tool_router import TOOL_DEFINITIONS
            tools = TOOL_DEFINITIONS
        except ImportError:
            tools = {}
            
        for tool_name, tool_def in tools.items():
            widget = ToolAssignmentWidget(tool_name, tool_def)
            widget.assignment_changed.connect(self._on_assignment_changed)
            self.tool_widgets[tool_name] = widget
            tools_layout.addWidget(widget)
                
        tools_layout.addStretch()
        
        scroll.setWidget(tools_container)
        layout.addWidget(scroll)
        
        # Status bar
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self.status_label)
        
    def _load_config(self):
        """Load routing configuration."""
        try:
            from forge_ai.core.tool_router import get_router
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
            from forge_ai.core.tool_router import get_router
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
                from forge_ai.core.tool_router import get_router
                router = get_router()
                router._set_defaults()
                self._load_config()
                self.status_label.setText("Reset to defaults")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to reset: {e}")
                
    def _on_assignment_changed(self, tool_name: str):
        """Handle assignment change in a tool widget."""
        self.status_label.setText(f"Modified {tool_name} - remember to save!")
        self.status_label.setStyleSheet("color: #f39c12; font-style: italic;")
    
    def refresh_models(self):
        """Refresh model dropdowns in all tool widgets."""
        for tool_name, widget in self.tool_widgets.items():
            widget._populate_model_options()
        self.status_label.setText("Model list refreshed")
