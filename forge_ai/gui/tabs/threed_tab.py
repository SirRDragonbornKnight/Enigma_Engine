"""
3D Generation Tab - Generate 3D models from text prompts.

Providers:
  - LOCAL: Shap-E (or built-in OBJ fallback)
  - REPLICATE: Cloud 3D generation (requires replicate, API key)
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QTextEdit, QProgressBar,
        QMessageBox, QGroupBox, QSpinBox, QDoubleSpinBox,
        QFileDialog, QLineEdit
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from ...config import CONFIG
from .shared_components import NoScrollComboBox, disable_scroll_on_combos

# Output directory
OUTPUT_DIR = Path(CONFIG.get("outputs_dir", "outputs")) / "3d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 3D Generation Implementations
# =============================================================================

class Local3DGen:
    """Local 3D generation using Shap-E with built-in fallback."""
    
    def __init__(self, model: str = "shap-e"):
        self.model_type = model
        self.pipe = None
        self.is_loaded = False
        self._using_builtin = False
        self._builtin_3d = None
    
    def load(self) -> bool:
        # Try Shap-E first
        try:
            import torch
            from diffusers import ShapEPipeline
            
            self.pipe = ShapEPipeline.from_pretrained(
                "openai/shap-e",
                torch_dtype=torch.float16
            )
            
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
            
            self.is_loaded = True
            self._using_builtin = False
            return True
        except ImportError:
            pass
        except Exception as e:
            print(f"Shap-E not available: {e}")
        
        # Fall back to built-in 3D generator
        try:
            from ...builtin import Builtin3DGen
            self._builtin_3d = Builtin3DGen()
            if self._builtin_3d.load():
                self.is_loaded = True
                self._using_builtin = True
                print("Using built-in 3D generator (geometric primitives)")
                return True
        except Exception as e:
            print(f"Built-in 3D gen failed: {e}")
        
        return False
    
    def unload(self):
        if self.pipe:
            del self.pipe
            self.pipe = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        if self._builtin_3d:
            self._builtin_3d.unload()
            self._builtin_3d = None
        self.is_loaded = False
        self._using_builtin = False
    
    def generate(self, prompt: str, guidance_scale: float = 15.0,
                 num_inference_steps: int = 64, **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}
        
        # Use built-in if available
        if self._using_builtin:
            timestamp = int(time.time())
            filepath = str(OUTPUT_DIR / f"3d_{timestamp}.obj")
            result = self._builtin_3d.generate(prompt)
            if result.get("success") and result.get("obj_data"):
                with open(filepath, 'w') as f:
                    f.write(result["obj_data"])
                result["path"] = filepath
            return result
        
        try:
            start = time.time()
            
            # Ensure prompt is a string (CLIP tokenizer requires str type)
            if prompt is not None:
                prompt = str(prompt).strip()
            if not prompt:
                return {"success": False, "error": "Prompt cannot be empty"}
            
            output = self.pipe(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            
            # Save as PLY mesh
            timestamp = int(time.time())
            filename = f"3d_{timestamp}.ply"
            filepath = OUTPUT_DIR / filename
            
            # Export mesh
            mesh = output.images[0]
            
            # Try to save as PLY
            try:
                import trimesh
                if hasattr(mesh, 'export'):
                    mesh.export(str(filepath))
                else:
                    # Convert vertices/faces if needed
                    tmesh = trimesh.Trimesh(
                        vertices=mesh.verts.cpu().numpy() if hasattr(mesh, 'verts') else mesh,
                        faces=mesh.faces.cpu().numpy() if hasattr(mesh, 'faces') else None
                    )
                    tmesh.export(str(filepath))
            except ImportError:
                # Fallback - save raw data
                import pickle
                filepath = OUTPUT_DIR / f"3d_{timestamp}.pkl"
                with open(filepath, 'wb') as f:
                    pickle.dump(mesh, f)
            
            return {
                "success": True,
                "path": str(filepath),
                "duration": time.time() - start
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class Cloud3DGen:
    """Cloud 3D generation via Replicate."""
    
    def __init__(self, api_key: Optional[str] = None,
                 service: str = "replicate"):
        self.api_key = api_key or os.environ.get("REPLICATE_API_TOKEN")
        self.service = service
        self.client = None
        self.is_loaded = False
    
    def load(self) -> bool:
        try:
            import replicate
            os.environ["REPLICATE_API_TOKEN"] = self.api_key or ""
            self.client = replicate
            self.is_loaded = bool(self.api_key)
            return self.is_loaded
        except ImportError:
            print("Install: pip install replicate")
            return False
    
    def unload(self):
        self.client = None
        self.is_loaded = False
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded or missing API key"}
        
        try:
            import requests
            start = time.time()
            
            # Use Shap-E on Replicate
            output = self.client.run(
                "cjwbw/shap-e:cf86502e5ffeb7f4c8f68cdf57f3bea50c18a5e3e5f42e37be4e5f3a16dcd62e",
                input={
                    "prompt": prompt,
                    "guidance_scale": kwargs.get('guidance_scale', 15.0),
                }
            )
            
            # Download the result
            result_url = output if isinstance(output, str) else output[0]
            resp = requests.get(result_url)
            
            timestamp = int(time.time())
            # Determine extension from URL or default to .glb
            ext = ".glb"
            if ".ply" in result_url:
                ext = ".ply"
            elif ".obj" in result_url:
                ext = ".obj"
            
            filename = f"cloud_3d_{timestamp}{ext}"
            filepath = OUTPUT_DIR / filename
            filepath.write_bytes(resp.content)
            
            return {
                "success": True,
                "path": str(filepath),
                "duration": time.time() - start
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# GUI Components
# =============================================================================

_providers = {
    'local': None,
    'replicate': None,
}


def get_provider(name: str):
    global _providers
    
    if name == 'local' and _providers['local'] is None:
        _providers['local'] = Local3DGen()
    elif name == 'replicate' and _providers['replicate'] is None:
        _providers['replicate'] = Cloud3DGen()
    
    return _providers.get(name)


class ThreeDGenerationWorker(QThread):
    """Background worker for 3D generation."""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    
    def __init__(self, prompt, guidance_scale, steps, provider_name, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.provider_name = provider_name
    
    def run(self):
        try:
            self.progress.emit(10)
            
            # Check if router has 3d assignments - use router if configured
            try:
                from ...core.tool_router import get_router
                router = get_router()
                assignments = router.get_assignments("3d")
                
                if assignments:
                    self.progress.emit(30)
                    params = {
                        "prompt": str(self.prompt).strip() if self.prompt else "",
                        "guidance_scale": float(self.guidance_scale),
                        "num_inference_steps": int(self.steps)
                    }
                    result = router.execute_tool("3d", params)
                    self.progress.emit(100)
                    self.finished.emit(result)
                    return
            except Exception as router_error:
                print(f"Router fallback: {router_error}")
            
            # Direct provider fallback
            provider = get_provider(self.provider_name)
            if provider is None:
                self.finished.emit({"success": False, "error": "Unknown provider"})
                return
            
            if not provider.is_loaded:
                self.progress.emit(20)
                if not provider.load():
                    self.finished.emit({"success": False, "error": "Failed to load provider"})
                    return
            
            self.progress.emit(40)
            
            result = provider.generate(
                self.prompt,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.steps
            )
            
            self.progress.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)})


class ThreeDTab(QWidget):
    """Tab for 3D model generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.worker = None
        self.last_3d_path = None
        self.setup_ui()
        
        # Register references on parent window for chat integration
        if parent:
            parent.threed_prompt = self.prompt_input
            parent.threed_tab = self
            parent._generate_3d = self._generate_3d
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("3D Generation")
        header.setObjectName("header")
        layout.addWidget(header)
        
        # 3D Preview at TOP
        self.preview_label = QLabel("Generated 3D model info will appear here")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(180)
        self.preview_label.setStyleSheet("background-color: #2d2d2d; border-radius: 4px;")
        layout.addWidget(self.preview_label, stretch=1)
        
        # Progress and Status
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Provider and Options in one row
        settings_layout = QHBoxLayout()
        
        settings_layout.addWidget(QLabel("Provider:"))
        self.provider_combo = NoScrollComboBox()
        self.provider_combo.addItems(['Local (Shap-E)', 'Replicate (Cloud)'])
        settings_layout.addWidget(self.provider_combo)
        
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self._load_provider)
        settings_layout.addWidget(self.load_btn)
        
        settings_layout.addWidget(QLabel("Guidance:"))
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 20.0)
        self.guidance_spin.setValue(15.0)
        settings_layout.addWidget(self.guidance_spin)
        
        settings_layout.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 100)
        self.steps_spin.setValue(64)
        settings_layout.addWidget(self.steps_spin)
        
        settings_layout.addStretch()
        layout.addLayout(settings_layout)
        
        # Prompt - compact
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("Prompt:"))
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Describe the 3D object you want to generate...")
        prompt_layout.addWidget(self.prompt_input)
        layout.addLayout(prompt_layout)
        
        # Reference - compact
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("Reference:"))
        self.ref_input_path = QLineEdit()
        self.ref_input_path.setPlaceholderText("Optional image for image-to-3D")
        self.ref_input_path.setReadOnly(True)
        ref_layout.addWidget(self.ref_input_path)
        
        browse_ref_btn = QPushButton("Browse")
        browse_ref_btn.clicked.connect(self._browse_reference)
        ref_layout.addWidget(browse_ref_btn)
        
        clear_ref_btn = QPushButton("Clear")
        clear_ref_btn.clicked.connect(self._clear_reference)
        ref_layout.addWidget(clear_ref_btn)
        layout.addLayout(ref_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Generate 3D")
        self.generate_btn.setStyleSheet("background-color: #e67e22; font-weight: bold; padding: 8px;")
        self.generate_btn.clicked.connect(self._generate_3d)
        btn_layout.addWidget(self.generate_btn)
        
        self.open_btn = QPushButton("Open")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_3d)
        btn_layout.addWidget(self.open_btn)
        
        self.save_btn = QPushButton("Save As")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_3d)
        btn_layout.addWidget(self.save_btn)
        
        self.open_folder_btn = QPushButton("Output Folder")
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        btn_layout.addWidget(self.open_folder_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
    
    def _get_provider_name(self) -> str:
        text = self.provider_combo.currentText()
        if 'Local' in text:
            return 'local'
        elif 'Replicate' in text:
            return 'replicate'
        return 'local'
    
    def _load_provider(self):
        provider_name = self._get_provider_name()
        provider = get_provider(provider_name)
        
        if provider and not provider.is_loaded:
            self.status_label.setText(f"Loading {provider_name}...")
            self.load_btn.setEnabled(False)
            
            from PyQt5.QtCore import QTimer
            def do_load():
                success = provider.load()
                if success:
                    self.status_label.setText(f"{provider_name} loaded!")
                else:
                    self.status_label.setText(f"Failed to load {provider_name}")
                self.load_btn.setEnabled(True)
            
            QTimer.singleShot(100, do_load)
    
    def _generate_3d(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "No Prompt", "Please enter a prompt")
            return
        
        provider_name = self._get_provider_name()
        
        self.generate_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Generating 3D model (this may take a while)...")
        
        self.worker = ThreeDGenerationWorker(
            prompt,
            self.guidance_spin.value(),
            self.steps_spin.value(),
            provider_name
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_generation_complete)
        self.worker.start()
    
    def _on_generation_complete(self, result: dict):
        self.generate_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        if result.get("success"):
            path = result.get("path", "")
            duration = result.get("duration", 0)
            
            self.last_3d_path = path
            self.open_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.status_label.setText(f"Generated in {duration:.1f}s - Saved to: {path}")
        else:
            error = result.get("error", "Unknown error")
            self.status_label.setText(f"Error: {error}")
    
    def _open_3d(self):
        if self.last_3d_path and Path(self.last_3d_path).exists():
            from .output_helpers import open_in_default_viewer
            open_in_default_viewer(self.last_3d_path)
    
    def _save_3d(self):
        """Save the generated 3D model to a custom location."""
        if not self.last_3d_path:
            return
        
        ext = Path(self.last_3d_path).suffix
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save 3D Model",
            str(Path.home() / f"generated_model{ext}"),
            f"3D Files (*{ext});;PLY Files (*.ply);;GLB Files (*.glb);;OBJ Files (*.obj);;All Files (*.*)"
        )
        if path:
            import shutil
            shutil.copy(self.last_3d_path, path)
            QMessageBox.information(self, "Saved", f"3D model saved to:\n{path}")
    
    def _open_output_folder(self):
        from .output_helpers import open_folder
        open_folder(OUTPUT_DIR)
    
    def _browse_reference(self):
        """Browse for a reference image for image-to-3D generation."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Image",
            str(Path.home()),
            "Image Files (*.png *.jpg *.jpeg *.webp *.bmp);;All Files (*.*)"
        )
        if path:
            self.ref_input_path.setText(path)
    
    def _clear_reference(self):
        """Clear the reference input."""
        self.ref_input_path.clear()


def create_threed_tab(parent) -> QWidget:
    """Factory function for creating the 3D tab."""
    if not HAS_PYQT:
        raise ImportError("PyQt5 is required for the 3D Tab")
    return ThreeDTab(parent)
