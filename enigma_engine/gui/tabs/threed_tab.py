"""
3D Generation Tab - Generate 3D models from text prompts.

Providers:
  - LOCAL: Shap-E (or built-in OBJ fallback)
  - REPLICATE: Cloud 3D generation (requires replicate, API key)
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtWidgets import (
        QDoubleSpinBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from ...config import CONFIG
from .shared_components import NoScrollComboBox

# Output directory
OUTPUT_DIR = Path(CONFIG.get("outputs_dir", "outputs")) / "3d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 3D Generation Implementations
# =============================================================================

class Local3DGen:
    """Local 3D generation using Shap-E with built-in fallback."""
    
    def __init__(self, model: str = "shap-e", use_cpu: bool = False):
        self.model_type = model
        self.pipe = None
        self.is_loaded = False
        self._using_builtin = False
        self._builtin_3d = None
        self.use_cpu = use_cpu  # Force CPU mode to save GPU memory
        self._device = None
    
    def load(self) -> bool:
        # Try Shap-E first
        try:
            import torch
            from diffusers import ShapEPipeline

            # Check GPU memory before loading (Shap-E needs ~4GB)
            if torch.cuda.is_available() and not self.use_cpu:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
                if free_mem < 3.5:  # Need at least 3.5GB free
                    logger.warning(f"Only {free_mem:.1f}GB GPU memory free. Shap-E needs ~4GB. Using CPU instead.")
                    self.use_cpu = True
            
            # Use float32 on CPU, float16 on GPU
            dtype = torch.float32 if self.use_cpu else torch.float16
            
            logger.info(f"Loading Shap-E 3D generator ({'CPU' if self.use_cpu else 'GPU'})...")
            self.pipe = ShapEPipeline.from_pretrained(
                "openai/shap-e",
                torch_dtype=dtype,
                low_cpu_mem_usage=True  # Reduce memory during loading
            )
            
            if torch.cuda.is_available() and not self.use_cpu:
                self._device = "cuda"
                self.pipe = self.pipe.to("cuda")
                # Enable memory efficient attention if available
                try:
                    self.pipe.enable_attention_slicing(1)
                except Exception as e:
                    logger.debug(f"Attention slicing not available: {e}")
            else:
                self._device = "cpu"
            
            self.is_loaded = True
            self._using_builtin = False
            logger.info(f"Shap-E loaded successfully on {self._device}")
            return True
        except ImportError:
            logger.info("Shap-E not available - diffusers not installed")
        except Exception as e:
            logger.warning(f"Shap-E not available: {e}")
        
        # Fall back to built-in 3D generator
        try:
            from ...builtin import Builtin3DGen
            self._builtin_3d = Builtin3DGen()
            if self._builtin_3d.load():
                self.is_loaded = True
                self._using_builtin = True
                logger.info("Using built-in 3D generator (geometric primitives)")
                return True
        except Exception as e:
            logger.warning(f"Built-in 3D gen failed: {e}")
        
        return False
    
    def unload(self):
        if self.pipe:
            del self.pipe
            self.pipe = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            except ImportError:
                pass  # Intentionally silent
        if self._builtin_3d:
            self._builtin_3d.unload()
            self._builtin_3d = None
        self.is_loaded = False
        self._using_builtin = False
        self._device = None
    
    def generate(self, prompt: str, guidance_scale: float = 15.0,
                 num_inference_steps: int = 64, **kwargs) -> dict[str, Any]:
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
            logger.warning("Replicate not available. Install: pip install replicate")
            return False
    
    def unload(self):
        self.client = None
        self.is_loaded = False
    
    def generate(self, prompt: str, **kwargs) -> dict[str, Any]:
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
            resp = requests.get(result_url, timeout=120)
            
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
    status = pyqtSignal(str)  # For estimated time updates
    
    def __init__(self, prompt, guidance_scale, steps, provider_name, width=256, height=256, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.provider_name = provider_name
        self.width = width
        self.height = height
        self._stop_requested = False
        self._start_time = None
    
    def request_stop(self):
        """Request the worker to stop."""
        self._stop_requested = True
    
    def run(self):
        import time
        self._start_time = time.time()
        
        try:
            if self._stop_requested:
                self.finished.emit({"success": False, "error": "Cancelled by user"})
                return
                
            self.progress.emit(10)
            self.status.emit("Estimated: 30-120 seconds...")
            
            # Check if router has 3d assignments - use router if configured
            try:
                from ...core.tool_router import get_router
                router = get_router()
                assignments = router.get_assignments("3d")
                
                if assignments:
                    if self._stop_requested:
                        self.finished.emit({"success": False, "error": "Cancelled by user"})
                        return
                    self.progress.emit(30)
                    self.status.emit("Generating via router...")
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
                logger.debug(f"Router fallback: {router_error}")
            
            if self._stop_requested:
                self.finished.emit({"success": False, "error": "Cancelled by user"})
                return
            
            # Direct provider fallback
            provider = get_provider(self.provider_name)
            if provider is None:
                self.finished.emit({"success": False, "error": "Unknown provider"})
                return
            
            if not provider.is_loaded:
                self.progress.emit(20)
                self.status.emit("Loading 3D model (this may take a minute)...")
                if not provider.load():
                    self.finished.emit({"success": False, "error": "Failed to load provider"})
                    return
            
            if self._stop_requested:
                self.finished.emit({"success": False, "error": "Cancelled by user"})
                return
            
            self.progress.emit(40)
            self.status.emit("Generating 3D model...")
            
            result = provider.generate(
                self.prompt,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.steps
            )
            
            import time
            elapsed = time.time() - self._start_time
            self.status.emit(f"Completed in {elapsed:.1f}s")
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
        self.load_btn.setToolTip("Load the 3D model (uses ~4GB GPU memory)")
        settings_layout.addWidget(self.load_btn)
        
        self.unload_btn = QPushButton("Unload")
        self.unload_btn.clicked.connect(self._unload_provider)
        self.unload_btn.setToolTip("Unload to free GPU memory")
        self.unload_btn.setEnabled(False)
        settings_layout.addWidget(self.unload_btn)
        
        # CPU mode checkbox
        from PyQt5.QtWidgets import QCheckBox
        self.cpu_mode_check = QCheckBox("CPU Mode")
        self.cpu_mode_check.setToolTip("Use CPU instead of GPU (slower but saves GPU memory)")
        self.cpu_mode_check.setChecked(False)
        settings_layout.addWidget(self.cpu_mode_check)
        
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
        
        # Size options row
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Output Size:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 512)
        self.width_spin.setValue(256)
        self.width_spin.setSingleStep(64)
        self.width_spin.setToolTip("3D preview/render width")
        size_layout.addWidget(self.width_spin)
        
        size_layout.addWidget(QLabel("x"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 512)
        self.height_spin.setValue(256)
        self.height_spin.setSingleStep(64)
        self.height_spin.setToolTip("3D preview/render height")
        size_layout.addWidget(self.height_spin)
        
        size_layout.addStretch()
        layout.addLayout(size_layout)
        
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
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("background-color: #e74c3c; font-weight: bold; padding: 8px;")
        self.stop_btn.clicked.connect(self._stop_generation)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Stop the current generation")
        btn_layout.addWidget(self.stop_btn)
        
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
        
        # For local provider, recreate with CPU mode setting
        if provider_name == 'local':
            use_cpu = self.cpu_mode_check.isChecked()
            global _providers
            # Unload existing if any
            if _providers.get('local') and _providers['local'].is_loaded:
                _providers['local'].unload()
            _providers['local'] = Local3DGen(use_cpu=use_cpu)
        
        provider = get_provider(provider_name)
        
        if provider and not provider.is_loaded:
            self.status_label.setText(f"Loading {provider_name}...")
            self.load_btn.setEnabled(False)
            self.cpu_mode_check.setEnabled(False)
            
            # Load in background thread to prevent UI freeze
            import threading
            def do_load():
                success = provider.load()
                from PyQt5.QtCore import Q_ARG, QMetaObject, Qt
                if success:
                    device = getattr(provider, '_device', 'unknown')
                    QMetaObject.invokeMethod(
                        self.status_label, "setText",
                        Qt.QueuedConnection,
                        Q_ARG(str, f"{provider_name} loaded on {device}!")
                    )
                    # Enable unload button
                    from PyQt5.QtCore import QTimer
                    def enable_unload():
                        self.unload_btn.setEnabled(True)
                    QTimer.singleShot(0, enable_unload)
                else:
                    QMetaObject.invokeMethod(
                        self.status_label, "setText",
                        Qt.QueuedConnection,
                        Q_ARG(str, f"Failed to load {provider_name}")
                    )
                QMetaObject.invokeMethod(
                    self.load_btn, "setEnabled",
                    Qt.QueuedConnection,
                    Q_ARG(bool, True)
                )
                from PyQt5.QtCore import QTimer
                def enable_cpu_check():
                    self.cpu_mode_check.setEnabled(True)
                QTimer.singleShot(0, enable_cpu_check)
            
            thread = threading.Thread(target=do_load, daemon=True)
            thread.start()
        elif provider and provider.is_loaded:
            device = getattr(provider, '_device', 'already loaded')
            self.status_label.setText(f"{provider_name} already loaded on {device}")
    
    def _unload_provider(self):
        """Unload the 3D model to free GPU memory."""
        provider_name = self._get_provider_name()
        provider = get_provider(provider_name)
        
        if provider and provider.is_loaded:
            provider.unload()
            self.status_label.setText(f"{provider_name} unloaded - GPU memory freed")
            self.unload_btn.setEnabled(False)
    
    def _generate_3d(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "No Prompt", "Please enter a prompt")
            return
        
        provider_name = self._get_provider_name()
        
        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Generating 3D model (this may take a while)...")
        
        self.worker = ThreeDGenerationWorker(
            prompt,
            self.guidance_spin.value(),
            self.steps_spin.value(),
            provider_name,
            self.width_spin.value(),
            self.height_spin.value()
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self._on_generation_complete)
        self.worker.start()
    
    def _stop_generation(self):
        """Stop the current generation."""
        if self.worker and self.worker.isRunning():
            self.status_label.setText("Stopping generation...")
            self.worker.request_stop()
            self.stop_btn.setEnabled(False)
            
            # Give thread time to stop gracefully after 2 seconds
            from PyQt5.QtCore import QTimer
            def force_stop():
                if self.worker and self.worker.isRunning():
                    # Wait for graceful completion instead of terminate
                    if not self.worker.wait(1000):
                        # Thread still running - it will complete in background
                        pass
                    self._on_generation_complete({"success": False, "error": "Stopped by user"})
            QTimer.singleShot(2000, force_stop)
    
    def _on_generation_complete(self, result: dict):
        self.generate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
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
    
    def closeEvent(self, event):
        """Clean up worker thread when tab is closed."""
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            if not self.worker.wait(2000):  # Wait up to 2 seconds
                # Thread didn't finish in time, but don't force terminate
                # It will complete naturally in the background
                pass
        super().closeEvent(event)


def create_threed_tab(parent) -> QWidget:
    """Factory function for creating the 3D tab."""
    if not HAS_PYQT:
        raise ImportError("PyQt5 is required for the 3D Tab")
    return ThreeDTab(parent)
