"""
Model Selector Widget - Shared dropdown for selecting AI models in generation tabs.

This widget syncs with the Model Router tab so changes in one place reflect everywhere.
"""

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QPushButton, QWidget

from .shared_components import NoScrollComboBox

# Tool -> List of model options (local + free HuggingFace options)
# Format: (model_id, display_name, model_type, is_free)
TOOL_MODEL_OPTIONS = {
    "image": [
        ("local:stable-diffusion", "Local: Stable Diffusion", "local", True),
        ("huggingface:stabilityai/stable-diffusion-xl-base-1.0", "HF FREE: SDXL", "huggingface", True),
        ("huggingface:stabilityai/sd-turbo", "HF FREE: SD Turbo (fast)", "huggingface", True),
        ("huggingface:black-forest-labs/FLUX.1-schnell", "HF FREE: FLUX.1 Schnell", "huggingface", True),
        ("api:openai", "API: DALL-E 3 (paid)", "api", False),
        ("api:replicate", "API: Replicate (paid)", "api", False),
    ],
    "code": [
        ("enigma:default", "Enigma: Default Model", "enigma_engine", True),
        ("huggingface:Qwen/Qwen2.5-Coder-7B-Instruct", "HF FREE: Qwen2.5 Coder 7B", "huggingface", True),
        ("huggingface:deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "HF FREE: DeepSeek Coder V2", "huggingface", True),
        ("huggingface:bigcode/starcoder2-3b", "HF FREE: StarCoder2 3B", "huggingface", True),
        ("api:openai", "API: GPT-4 (paid)", "api", False),
    ],
    "audio": [
        ("local:tts", "Local: pyttsx3 (offline)", "local", True),
        ("huggingface:hexgrad/Kokoro-82M", "HF FREE: Kokoro 82M", "huggingface", True),
        ("huggingface:suno/bark-small", "HF FREE: Bark Small", "huggingface", True),
        ("huggingface:facebook/mms-tts-eng", "HF FREE: MMS TTS English", "huggingface", True),
        ("api:elevenlabs", "API: ElevenLabs (paid)", "api", False),
    ],
    "video": [
        ("local:animatediff", "Local: AnimateDiff", "local", True),
        ("huggingface:ali-vilab/text-to-video-ms-1.7b", "HF FREE: ModelScope Video", "huggingface", True),
        ("api:replicate", "API: Replicate (paid)", "api", False),
    ],
    "3d": [
        ("local:shap-e", "Local: Shap-E", "local", True),
        ("huggingface:openai/shap-e", "HF FREE: Shap-E", "huggingface", True),
        ("huggingface:openai/point-e", "HF FREE: Point-E", "huggingface", True),
        ("api:replicate", "API: Replicate (paid)", "api", False),
    ],
    "embeddings": [
        ("local:embeddings", "Local: Sentence Transformers", "local", True),
        ("huggingface:sentence-transformers/all-MiniLM-L6-v2", "HF FREE: MiniLM-L6", "huggingface", True),
        ("huggingface:BAAI/bge-small-en-v1.5", "HF FREE: BGE Small", "huggingface", True),
        ("huggingface:thenlper/gte-small", "HF FREE: GTE Small", "huggingface", True),
        ("api:openai", "API: OpenAI Embeddings (paid)", "api", False),
    ],
    "vision": [
        ("local:vision", "Local: Vision Analysis", "local", True),
        ("huggingface:microsoft/Florence-2-base", "HF FREE: Florence 2", "huggingface", True),
        ("huggingface:Salesforce/blip2-opt-2.7b", "HF FREE: BLIP2", "huggingface", True),
        ("huggingface:nlpconnect/vit-gpt2-image-captioning", "HF FREE: ViT-GPT2 Caption", "huggingface", True),
    ],
    "chat": [
        ("enigma:default", "Enigma: Default Model", "enigma_engine", True),
        ("huggingface:deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "HF FREE: DeepSeek-R1 32B", "huggingface", True),
        ("huggingface:meta-llama/Llama-3.3-70B-Instruct", "HF FREE: Llama 3.3 70B", "huggingface", True),
        ("huggingface:Qwen/Qwen2.5-7B-Instruct", "HF FREE: Qwen2.5 7B", "huggingface", True),
        ("huggingface:microsoft/Phi-3.5-mini-instruct", "HF FREE: Phi-3.5 Mini 3.8B", "huggingface", True),
        ("huggingface:mistralai/Mistral-7B-Instruct-v0.3", "HF FREE: Mistral 7B v0.3", "huggingface", True),
        ("huggingface:google/gemma-2-9b-it", "HF FREE: Gemma 2 9B", "huggingface", True),
        ("api:openai", "API: GPT-4 (paid)", "api", False),
    ],
}


class ModelSelector(QWidget):
    """
    Dropdown widget for selecting which AI model to use for a tool.
    
    Syncs with the tool router so model selection is consistent across:
    - The individual generation tabs (Image, Code, Audio, etc.)
    - The Model Router tab
    """
    
    model_changed = pyqtSignal(str, str)  # tool_name, model_id
    
    def __init__(self, tool_name: str, parent=None, show_label: bool = True):
        super().__init__(parent)
        self.tool_name = tool_name
        self._setup_ui(show_label)
        self._load_from_router()
    
    def _setup_ui(self, show_label: bool):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        if show_label:
            label = QLabel("Model:")
            label.setStyleSheet("font-weight: bold;")
            layout.addWidget(label)
        
        # Model dropdown
        self.model_combo = NoScrollComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.setToolTip(
            "Select which AI model to use for this task.\n"
            "• Local: Runs on your device (free, may need dependencies)\n"
            "• HF: HuggingFace models (free, downloads model)\n"
            "• API: Cloud services (requires API key & payment)"
        )
        self._populate_options()
        self.model_combo.currentIndexChanged.connect(self._on_selection_changed)
        layout.addWidget(self.model_combo)
        
        # Status indicator
        self.status_label = QLabel("[*]")
        self.status_label.setToolTip("Model status")
        self.status_label.setStyleSheet("color: #bac2de; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        # Apply button
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setToolTip("Apply this model selection to the router")
        self.apply_btn.clicked.connect(self._apply_selection)
        self.apply_btn.setFixedWidth(60)
        layout.addWidget(self.apply_btn)
    
    def _populate_options(self):
        """Populate dropdown with model options for this tool."""
        self.model_combo.clear()
        
        options = TOOL_MODEL_OPTIONS.get(self.tool_name, [])
        
        # Add header
        self.model_combo.addItem(f"-- Select {self.tool_name.title()} Model --", "")
        
        # Group by free vs paid (handle both 3-tuple and 4-tuple formats)
        free_options = []
        paid_options = []
        for opt in options:
            if len(opt) > 3:
                if opt[3]:  # is_free flag
                    free_options.append(opt)
                else:
                    paid_options.append(opt)
            else:
                # Old 3-tuple format - assume free
                free_options.append(opt + (True,))
        
        if free_options:
            self.model_combo.addItem("== FREE OPTIONS ==", "__separator__")
            
            current_type = None
            for opt in free_options:
                model_id = opt[0]
                display_name = opt[1]
                model_type = opt[2]
                
                # Add separator for different types
                if model_type != current_type:
                    if model_type == "local":
                        self.model_combo.addItem("-- Local (No API needed) --", "__separator__")
                    elif model_type == "huggingface":
                        self.model_combo.addItem("-- HuggingFace (Free API) --", "__separator__")
                    elif model_type == "enigma_engine":
                        self.model_combo.addItem("-- Your Forge Models --", "__separator__")
                    current_type = model_type
                
                self.model_combo.addItem(display_name, model_id)
        
        if paid_options:
            self.model_combo.addItem("== PAID OPTIONS ==", "__separator__")
            for opt in paid_options:
                model_id = opt[0]
                display_name = opt[1]
                self.model_combo.addItem(display_name, model_id)
        
        # Also add any Forge models from registry
        try:
            from ...core.model_registry import ModelRegistry
            registry = ModelRegistry()
            models = registry.list_models()
            if models:
                self.model_combo.addItem("-- Your Trained Models --", "__separator__")
                for model in models:
                    name = model.get("name", str(model)) if isinstance(model, dict) else str(model)
                    if not name.startswith("enigma:"):
                        self.model_combo.addItem(f"Forge: {name}", f"enigma:{name}")
        except Exception:
            pass  # Intentionally silent
    
    def _load_from_router(self):
        """Load current model assignment from the router."""
        try:
            from ...core.tool_router import get_router
            router = get_router()
            assignments = router.get_assignments(self.tool_name)
            
            if assignments:
                # Get highest priority assignment
                best = max(assignments, key=lambda x: x.priority)
                model_id = best.model_id
                
                # Find and select in combo
                for i in range(self.model_combo.count()):
                    if self.model_combo.itemData(i) == model_id:
                        self.model_combo.setCurrentIndex(i)
                        break
                else:
                    # Model not in dropdown - add it
                    self.model_combo.addItem(f"Custom: {model_id}", model_id)
                    self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
                
                self._update_status(True)
            else:
                self._update_status(False)
                
        except Exception as e:
            print(f"Error loading router config: {e}")
            self._update_status(False)
    
    def _on_selection_changed(self, index: int):
        """Handle dropdown selection change."""
        model_id = self.model_combo.itemData(index)
        
        # Ignore separator items
        if not model_id or model_id == "__separator__":
            return
        
        # Visual feedback - show pending
        self.status_label.setText("[...]")
        self.status_label.setStyleSheet("color: #f39c12; font-size: 12px;")
        self.status_label.setToolTip("Selection pending - click Apply")
    
    def _apply_selection(self):
        """Apply the selected model to the router."""
        model_id = self.model_combo.currentData()
        
        if not model_id or model_id == "__separator__":
            QMessageBox.warning(self, "No Selection", "Please select a valid model.")
            return
        
        try:
            from ...core.tool_router import get_router
            router = get_router()
            
            # Clear existing assignments for this tool
            existing = router.get_assignments(self.tool_name)
            for assign in existing:
                router.unassign_model(self.tool_name, assign.model_id)
            
            # Add new assignment with high priority
            router.assign_model(self.tool_name, model_id, priority=100)
            
            # Update status
            self._update_status(True)
            
            # Emit signal for any listeners
            self.model_changed.emit(self.tool_name, model_id)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply model: {e}")
            self._update_status(False)
    
    def _update_status(self, is_set: bool):
        """Update the status indicator."""
        if is_set:
            self.status_label.setText("[OK]")
            self.status_label.setStyleSheet("color: #2ecc71; font-size: 12px;")
            self.status_label.setToolTip("Model assigned")
        else:
            self.status_label.setText("[*]")
            self.status_label.setStyleSheet("color: #bac2de; font-size: 12px;")
            self.status_label.setToolTip("No model assigned")
    
    def get_current_model(self) -> Optional[str]:
        """Get currently selected model ID."""
        return self.model_combo.currentData()
    
    def set_model(self, model_id: str):
        """Set the selected model (used when router changes externally)."""
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == model_id:
                self.model_combo.setCurrentIndex(i)
                self._update_status(True)
                return
        
        # Model not found - add it
        self.model_combo.addItem(f"Custom: {model_id}", model_id)
        self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
        self._update_status(True)
    
    def refresh(self):
        """Refresh from router (call when router changes)."""
        self._load_from_router()


def get_model_options(tool_name: str) -> list[tuple]:
    """Get available model options for a tool."""
    return TOOL_MODEL_OPTIONS.get(tool_name, [])
