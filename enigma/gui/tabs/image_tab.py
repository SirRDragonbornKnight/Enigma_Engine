"""
Image Generation Tab - Generate images using local or cloud models.

Providers:
  - LOCAL: Stable Diffusion (requires diffusers, torch)
  - OPENAI: DALL-E 3 (requires openai, API key)
  - REPLICATE: SDXL/Flux (requires replicate, API key)
"""

import os
import io
import time
import base64
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QComboBox, QTextEdit, QProgressBar,
        QMessageBox, QFileDialog, QSpinBox, QGroupBox,
        QDoubleSpinBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QPixmap
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from ...config import CONFIG

# Output directory
OUTPUT_DIR = Path(CONFIG.get("outputs_dir", "outputs")) / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Image Generation Implementations
# =============================================================================

class StableDiffusionLocal:
    """Local Stable Diffusion image generation."""
    
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-1"):
        self.model_id = model_id
        self.pipe = None
        self.is_loaded = False
    
    def load(self) -> bool:
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype
            ).to(device)
            
            self.is_loaded = True
            return True
        except ImportError:
            print("Install: pip install diffusers transformers accelerate torch")
            return False
        except Exception as e:
            print(f"Failed to load Stable Diffusion: {e}")
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
    
    def generate(self, prompt: str, width: int = 512, height: int = 512,
                 steps: int = 30, guidance: float = 7.5, 
                 negative_prompt: str = "", **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}
        
        try:
            start = time.time()
            
            result = self.pipe(
                prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            
            image = result.images[0]
            
            # Save to file
            timestamp = int(time.time())
            filename = f"sd_{timestamp}.png"
            filepath = OUTPUT_DIR / filename
            image.save(str(filepath))
            
            return {
                "success": True,
                "path": str(filepath),
                "duration": time.time() - start
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class OpenAIImage:
    """OpenAI DALL-E image generation (CLOUD - requires API key)."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "dall-e-3"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = None
        self.is_loaded = False
    
    def load(self) -> bool:
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.is_loaded = bool(self.api_key)
            return self.is_loaded
        except ImportError:
            print("Install: pip install openai")
            return False
    
    def unload(self):
        self.client = None
        self.is_loaded = False
    
    def generate(self, prompt: str, width: int = 1024, height: int = 1024,
                 **kwargs) -> Dict[str, Any]:
        if not self.is_loaded or not self.client:
            return {"success": False, "error": "Not loaded or missing API key"}
        
        try:
            start = time.time()
            
            # DALL-E 3 only supports certain sizes
            size = f"{width}x{height}"
            if size not in ["1024x1024", "1792x1024", "1024x1792"]:
                size = "1024x1024"
            
            response = self.client.images.generate(
                model=self.model,
                prompt=prompt,
                size=size,
                n=1,
                response_format="b64_json",
            )
            
            image_data = base64.b64decode(response.data[0].b64_json)
            
            # Save to file
            timestamp = int(time.time())
            filename = f"dalle_{timestamp}.png"
            filepath = OUTPUT_DIR / filename
            filepath.write_bytes(image_data)
            
            return {
                "success": True,
                "path": str(filepath),
                "duration": time.time() - start
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class ReplicateImage:
    """Replicate image generation (CLOUD - requires API key)."""
    
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "stability-ai/sdxl:latest"):
        self.api_key = api_key or os.environ.get("REPLICATE_API_TOKEN")
        self.model = model
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
    
    def generate(self, prompt: str, width: int = 1024, height: int = 1024,
                 **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded or missing API key"}
        
        try:
            import requests
            start = time.time()
            
            output = self.client.run(
                self.model,
                input={
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                }
            )
            
            # Download image
            image_url = output[0] if isinstance(output, list) else output
            resp = requests.get(image_url)
            
            # Save to file
            timestamp = int(time.time())
            filename = f"replicate_{timestamp}.png"
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

# Global instances (lazy loaded)
_providers = {
    'local': None,
    'openai': None,
    'replicate': None,
}


def get_provider(name: str):
    """Get or create a provider instance."""
    global _providers
    
    if name == 'local' and _providers['local'] is None:
        _providers['local'] = StableDiffusionLocal()
    elif name == 'openai' and _providers['openai'] is None:
        _providers['openai'] = OpenAIImage()
    elif name == 'replicate' and _providers['replicate'] is None:
        _providers['replicate'] = ReplicateImage()
    
    return _providers.get(name)


class ImageGenerationWorker(QThread):
    """Background worker for image generation."""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    
    def __init__(self, prompt, width, height, steps, guidance, 
                 negative_prompt, provider_name, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.negative_prompt = negative_prompt
        self.provider_name = provider_name
    
    def run(self):
        try:
            self.progress.emit(10)
            
            provider = get_provider(self.provider_name)
            if provider is None:
                self.finished.emit({"success": False, "error": "Unknown provider"})
                return
            
            # Load if needed
            if not provider.is_loaded:
                self.progress.emit(20)
                if not provider.load():
                    self.finished.emit({"success": False, "error": "Failed to load provider"})
                    return
            
            self.progress.emit(40)
            
            # Generate
            result = provider.generate(
                self.prompt,
                width=self.width,
                height=self.height,
                steps=self.steps,
                guidance=self.guidance,
                negative_prompt=self.negative_prompt,
            )
            
            self.progress.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)})


class ImageTab(QWidget):
    """Tab for image generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.worker = None
        self.last_image_path = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Image Generation")
        header.setFont(QFont('Arial', 14, QFont.Bold))
        header.setStyleSheet("color: #e74c3c;")
        layout.addWidget(header)
        
        # Provider selection
        provider_group = QGroupBox("Provider")
        provider_layout = QHBoxLayout()
        
        self.provider_combo = QComboBox()
        self.provider_combo.addItems([
            'Local (Stable Diffusion)',
            'OpenAI (DALL-E 3) - Cloud',
            'Replicate (SDXL) - Cloud'
        ])
        provider_layout.addWidget(self.provider_combo)
        
        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self._load_provider)
        provider_layout.addWidget(self.load_btn)
        
        provider_layout.addStretch()
        provider_group.setLayout(provider_layout)
        layout.addWidget(provider_group)
        
        # Prompt input
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout()
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setMaximumHeight(80)
        self.prompt_input.setPlaceholderText("Describe the image you want to generate...")
        prompt_layout.addWidget(self.prompt_input)
        
        self.neg_prompt_input = QTextEdit()
        self.neg_prompt_input.setMaximumHeight(50)
        self.neg_prompt_input.setPlaceholderText("Negative prompt (what to avoid)...")
        prompt_layout.addWidget(self.neg_prompt_input)
        
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout()
        
        options_layout.addWidget(QLabel("Width:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(256, 2048)
        self.width_spin.setValue(512)
        self.width_spin.setSingleStep(64)
        options_layout.addWidget(self.width_spin)
        
        options_layout.addWidget(QLabel("Height:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(256, 2048)
        self.height_spin.setValue(512)
        self.height_spin.setSingleStep(64)
        options_layout.addWidget(self.height_spin)
        
        options_layout.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 150)
        self.steps_spin.setValue(30)
        options_layout.addWidget(self.steps_spin)
        
        options_layout.addWidget(QLabel("Guidance:"))
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 20.0)
        self.guidance_spin.setValue(7.5)
        self.guidance_spin.setSingleStep(0.5)
        options_layout.addWidget(self.guidance_spin)
        
        options_layout.addStretch()
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Generate button
        btn_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Generate Image")
        self.generate_btn.setStyleSheet("background-color: #e74c3c; font-weight: bold; padding: 10px;")
        self.generate_btn.clicked.connect(self._generate_image)
        btn_layout.addWidget(self.generate_btn)
        
        self.save_btn = QPushButton("Save As...")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_image)
        btn_layout.addWidget(self.save_btn)
        
        self.open_folder_btn = QPushButton("Open Output Folder")
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        btn_layout.addWidget(self.open_folder_btn)
        
        layout.addLayout(btn_layout)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Status
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Result area
        self.result_label = QLabel("Generated image will appear here")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumHeight(300)
        self.result_label.setStyleSheet("background-color: #2d2d2d; border-radius: 4px;")
        layout.addWidget(self.result_label, stretch=1)
    
    def _get_provider_name(self) -> str:
        """Get provider name from combo box."""
        text = self.provider_combo.currentText()
        if 'Local' in text:
            return 'local'
        elif 'OpenAI' in text:
            return 'openai'
        elif 'Replicate' in text:
            return 'replicate'
        return 'local'
    
    def _load_provider(self):
        """Pre-load the selected provider."""
        provider_name = self._get_provider_name()
        provider = get_provider(provider_name)
        
        if provider and not provider.is_loaded:
            self.status_label.setText(f"Loading {provider_name}...")
            self.load_btn.setEnabled(False)
            
            # Load in thread to not block UI
            from PyQt5.QtCore import QTimer
            def do_load():
                success = provider.load()
                if success:
                    self.status_label.setText(f"{provider_name} loaded successfully!")
                else:
                    self.status_label.setText(f"Failed to load {provider_name}")
                self.load_btn.setEnabled(True)
            
            QTimer.singleShot(100, do_load)
        else:
            self.status_label.setText(f"{provider_name} already loaded")
    
    def _generate_image(self):
        """Generate an image."""
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "No Prompt", "Please enter a prompt")
            return
        
        provider_name = self._get_provider_name()
        
        self.generate_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Generating...")
        
        self.worker = ImageGenerationWorker(
            prompt,
            self.width_spin.value(),
            self.height_spin.value(),
            self.steps_spin.value(),
            self.guidance_spin.value(),
            self.neg_prompt_input.toPlainText().strip(),
            provider_name
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_generation_complete)
        self.worker.start()
    
    def _on_generation_complete(self, result: dict):
        """Handle generation completion."""
        self.generate_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        if result.get("success"):
            path = result.get("path", "")
            duration = result.get("duration", 0)
            
            if path and Path(path).exists():
                self.last_image_path = path
                pixmap = QPixmap(path)
                scaled = pixmap.scaled(
                    self.result_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.result_label.setPixmap(scaled)
                self.save_btn.setEnabled(True)
                self.status_label.setText(f"Generated in {duration:.1f}s - Saved to: {path}")
            else:
                self.status_label.setText("Generation complete (no image path)")
        else:
            error = result.get("error", "Unknown error")
            self.status_label.setText(f"Error: {error}")
            self.result_label.setText(f"Generation failed:\n{error}")
    
    def _save_image(self):
        """Save the generated image to a custom location."""
        if not self.last_image_path:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            str(Path.home() / "generated_image.png"),
            "PNG Images (*.png);;JPEG Images (*.jpg)"
        )
        if path:
            import shutil
            shutil.copy(self.last_image_path, path)
            QMessageBox.information(self, "Saved", f"Image saved to:\n{path}")
    
    def _open_output_folder(self):
        """Open the output folder in file manager."""
        import subprocess
        import sys
        
        if sys.platform == 'darwin':
            subprocess.run(['open', str(OUTPUT_DIR)])
        elif sys.platform == 'win32':
            subprocess.run(['explorer', str(OUTPUT_DIR)])
        else:
            subprocess.run(['xdg-open', str(OUTPUT_DIR)])


def create_image_tab(parent) -> QWidget:
    """Factory function for creating the image tab."""
    if not HAS_PYQT:
        raise ImportError("PyQt5 is required for the Image Tab")
    return ImageTab(parent)
