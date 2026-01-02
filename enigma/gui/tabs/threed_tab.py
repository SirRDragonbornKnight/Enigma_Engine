"""
3D Generation Tab - Generate 3D models from text prompts.

Providers:
  - LOCAL: Shap-E or Point-E (requires diffusers with shap-e support)
  - REPLICATE: Cloud 3D generation (requires replicate, API key)
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QComboBox, QTextEdit, QProgressBar,
        QMessageBox, QGroupBox, QSpinBox, QDoubleSpinBox,
        QFileDialog
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from ...config import CONFIG

# Output directory
OUTPUT_DIR = Path(CONFIG.get("outputs_dir", "outputs")) / "3d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 3D Generation Implementations
# =============================================================================

class Local3DGen:
    """Local 3D generation using Shap-E."""
    
    def __init__(self, model: str = "shap-e"):
        self.model_type = model
        self.pipe = None
        self.is_loaded = False
    
    def load(self) -> bool:
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
            return True
        except ImportError as e:
            print(f"Install: pip install diffusers[torch] transformers accelerate")
            print(f"Missing: {e}")
            return False
        except Exception as e:
            print(f"Failed to load 3D model: {e}")
            return False
    
    def unload(self):
        if self.pipe:
            del self.pipe
            self.pipe = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        self.is_loaded = False
    
    def generate(self, prompt: str, guidance_scale: float = 15.0,
                 num_inference_steps: int = 64, **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}
        
        try:
            start = time.time()
            
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
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("3D Generation")
        header.setFont(QFont('Arial', 14, QFont.Bold))
        header.setStyleSheet("color: #e67e22;")
        layout.addWidget(header)
        
        # Provider selection
        provider_group = QGroupBox("Provider")
        provider_layout = QHBoxLayout()
        
        self.provider_combo = QComboBox()
        self.provider_combo.addItems([
            'Local (Shap-E)',
            'Replicate (Cloud)'
        ])
        provider_layout.addWidget(self.provider_combo)
        
        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self._load_provider)
        provider_layout.addWidget(self.load_btn)
        
        provider_layout.addStretch()
        provider_group.setLayout(provider_layout)
        layout.addWidget(provider_group)
        
        # Prompt
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout()
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setMaximumHeight(80)
        self.prompt_input.setPlaceholderText("Describe the 3D object you want to generate...")
        prompt_layout.addWidget(self.prompt_input)
        
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout()
        
        options_layout.addWidget(QLabel("Guidance Scale:"))
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 20.0)
        self.guidance_spin.setValue(15.0)
        self.guidance_spin.setSingleStep(0.5)
        options_layout.addWidget(self.guidance_spin)
        
        options_layout.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 100)
        self.steps_spin.setValue(64)
        options_layout.addWidget(self.steps_spin)
        
        options_layout.addStretch()
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Generate 3D Model")
        self.generate_btn.setStyleSheet("background-color: #e67e22; font-weight: bold; padding: 10px;")
        self.generate_btn.clicked.connect(self._generate_3d)
        btn_layout.addWidget(self.generate_btn)
        
        self.open_btn = QPushButton("Open 3D File")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_3d)
        btn_layout.addWidget(self.open_btn)
        
        self.open_folder_btn = QPushButton("Open Output Folder")
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        btn_layout.addWidget(self.open_folder_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Status
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Info
        info_label = QLabel(
            "⚠️ 3D generation requires significant GPU memory (4GB+ VRAM).\n"
            "Output formats: PLY, GLB, OBJ depending on provider."
        )
        info_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(info_label)
        
        layout.addStretch()
    
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
        prompt = self.prompt_input.toPlainText().strip()
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
            self.status_label.setText(f"Generated in {duration:.1f}s - Saved to: {path}")
        else:
            error = result.get("error", "Unknown error")
            self.status_label.setText(f"Error: {error}")
    
    def _open_3d(self):
        if self.last_3d_path and Path(self.last_3d_path).exists():
            import subprocess
            import sys
            
            if sys.platform == 'darwin':
                subprocess.run(['open', self.last_3d_path])
            elif sys.platform == 'win32':
                os.startfile(self.last_3d_path)
            else:
                subprocess.run(['xdg-open', self.last_3d_path])
    
    def _open_output_folder(self):
        import subprocess
        import sys
        
        if sys.platform == 'darwin':
            subprocess.run(['open', str(OUTPUT_DIR)])
        elif sys.platform == 'win32':
            subprocess.run(['explorer', str(OUTPUT_DIR)])
        else:
            subprocess.run(['xdg-open', str(OUTPUT_DIR)])


def create_threed_tab(parent) -> QWidget:
    """Factory function for creating the 3D tab."""
    if not HAS_PYQT:
        raise ImportError("PyQt5 is required for the 3D Tab")
    return ThreeDTab(parent)
