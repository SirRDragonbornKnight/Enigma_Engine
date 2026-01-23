"""
================================================================================
🎨 IMAGE GENERATION TAB - CREATE VISUAL ART
================================================================================

Generate images using local or cloud AI models! From simple procedural art
to Stable Diffusion to DALL-E 3.

📍 FILE: forge_ai/gui/tabs/image_tab.py
🏷️ TYPE: GUI Tab + Image Generators
🎯 MAIN CLASSES: ImageTab, PlaceholderImage, StableDiffusionLocal, OpenAIImage

┌─────────────────────────────────────────────────────────────────────────────┐
│  AVAILABLE PROVIDERS:                                                       │
│                                                                             │
│  🟢 PLACEHOLDER  - Built-in procedural art (NO dependencies!)              │
│  🟡 LOCAL        - Stable Diffusion (requires diffusers, torch)           │
│  🟠 OPENAI       - DALL-E 3 (requires openai, API key)                    │
│  🔴 REPLICATE    - SDXL/Flux (requires replicate, API key)               │
└─────────────────────────────────────────────────────────────────────────────┘

📁 OUTPUT LOCATION: outputs/images/

🔗 CONNECTED FILES:
    → USES:      forge_ai/builtin/ (BuiltinImageGen fallback)
    → USES:      forge_ai/config/ (CONFIG paths)
    ← USED BY:   forge_ai/gui/enhanced_window.py (loaded as tab)
    ← USED BY:   forge_ai/modules/registry.py (ImageGenLocalModule)

📖 PROVIDER CLASSES:
    • PlaceholderImage       - No dependencies, procedural art
    • StableDiffusionLocal   - Local SD with diffusers library
    • OpenAIImage            - DALL-E 3 via OpenAI API
    • ReplicateImage         - SDXL via Replicate API

📖 SEE ALSO:
    • forge_ai/gui/tabs/code_tab.py   - Code generation tab
    • forge_ai/gui/tabs/video_tab.py  - Video generation tab
    • forge_ai/gui/tabs/audio_tab.py  - Audio generation tab
    • forge_ai/core/tool_router.py    - Routes "image" requests here
"""

import os
import io
import time
import base64
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QComboBox, QTextEdit, QProgressBar,
        QMessageBox, QFileDialog, QSpinBox, QGroupBox,
        QDoubleSpinBox, QLineEdit, QCheckBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QPixmap, QImage, QPainter, QColor
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

class PlaceholderImage:
    """
    Built-in image generator - creates procedural images with no external dependencies.
    Uses the forge_ai.builtin.BuiltinImageGen for actual generation.
    """
    
    def __init__(self):
        self.is_loaded = False
        self._builtin_gen = None
    
    def load(self) -> bool:
        # Try to use our built-in generator
        try:
            from ...builtin import BuiltinImageGen
            self._builtin_gen = BuiltinImageGen()
            self._builtin_gen.load()
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Built-in image gen failed: {e}")
            # Still return True - we have fallback
            self.is_loaded = True
            return True
    
    def unload(self):
        if self._builtin_gen:
            self._builtin_gen.unload()
            self._builtin_gen = None
        self.is_loaded = False
    
    def generate(self, prompt: str, width: int = 512, height: int = 512,
                 **kwargs) -> Dict[str, Any]:
        """Generate a procedural image based on the prompt."""
        try:
            start = time.time()
            timestamp = int(time.time())
            filename = f"generated_{timestamp}.png"
            filepath = OUTPUT_DIR / filename
            
            # Try built-in generator first
            if self._builtin_gen:
                result = self._builtin_gen.generate(prompt, width=width, height=height)
                if result.get("success") and result.get("image_data"):
                    with open(filepath, 'wb') as f:
                        f.write(result["image_data"])
                    return {
                        "success": True,
                        "path": str(filepath),
                        "duration": time.time() - start,
                        "style": result.get("style", "procedural"),
                        "is_builtin": True
                    }
            
            # Fallback: create using Qt or PIL
            try:
                from PIL import Image, ImageDraw, ImageFont
                
                # Create gradient background
                img = Image.new('RGB', (width, height))
                for y in range(height):
                    r = int(40 + (y / height) * 60)
                    g = int(60 + (y / height) * 40)  
                    b = int(100 + (y / height) * 80)
                    for x in range(width):
                        img.putpixel((x, y), (r, g, b))
                
                draw = ImageDraw.Draw(img)
                
                # Add prompt text
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                # Wrap text
                words = prompt.split()
                lines = []
                current = ""
                for word in words:
                    test = current + " " + word if current else word
                    if len(test) < 40:
                        current = test
                    else:
                        lines.append(current)
                        current = word
                if current:
                    lines.append(current)
                
                y_pos = height // 2 - len(lines) * 15
                for line in lines[:5]:  # Max 5 lines
                    bbox = draw.textbbox((0, 0), line, font=font)
                    text_width = bbox[2] - bbox[0]
                    x_pos = (width - text_width) // 2
                    draw.text((x_pos, y_pos), line, fill=(255, 255, 255), font=font)
                    y_pos += 30
                
                # Add "PLACEHOLDER" watermark
                draw.text((10, height - 30), "PLACEHOLDER - Install diffusers for real images", 
                          fill=(150, 150, 150), font=font)
                
                img.save(str(filepath))
                
            except ImportError:
                # Fallback: create using Qt
                img = QImage(width, height, QImage.Format_RGB32)
                painter = QPainter(img)
                
                # Gradient background
                for y in range(height):
                    color = QColor(40 + int((y/height)*60), 
                                   60 + int((y/height)*40),
                                   100 + int((y/height)*80))
                    painter.setPen(color)
                    painter.drawLine(0, y, width, y)
                
                # Add text
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(img.rect(), Qt.AlignCenter | Qt.TextWordWrap, prompt[:200])
                
                painter.setPen(QColor(150, 150, 150))
                painter.drawText(10, height - 10, "PLACEHOLDER - Install diffusers for real images")
                
                painter.end()
                img.save(str(filepath))
            
            return {
                "success": True,
                "path": str(filepath),
                "duration": time.time() - start,
                "is_placeholder": True
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

class StableDiffusionLocal:
    """Local Stable Diffusion image generation."""
    
    def __init__(self, model_id: str = "nota-ai/bk-sdm-small"):
        # nota-ai/bk-sdm-small is a small (~500MB), fast SD model
        # Works well on limited GPU memory
        self.model_id = model_id
        self.pipe = None
        self.is_loaded = False
    
    def load(self) -> bool:
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            # Clear GPU cache first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Check available GPU memory
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            if device == "cuda":
                try:
                    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    free_gb = free_mem / (1024**3)
                    print(f"GPU free memory: {free_gb:.1f} GB")
                    # Need at least 2GB for SD small, fall back to CPU
                    if free_gb < 2.0:
                        print("Not enough GPU memory, using CPU instead")
                        device = "cpu"
                        dtype = torch.float32
                except Exception:
                    pass
            
            print(f"Loading Stable Diffusion from {self.model_id}...")
            print(f"Device: {device}, dtype: {dtype}")
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None,  # Disable safety checker for speed
                requires_safety_checker=False,
            )
            
            self.pipe = self.pipe.to(device)
            
            # Enable memory optimizations
            if device == "cuda":
                try:
                    self.pipe.enable_attention_slicing()
                except Exception:
                    pass
            
            self.is_loaded = True
            print("Stable Diffusion loaded successfully!")
            return True
        except ImportError as e:
            print(f"Install required: pip install diffusers transformers accelerate")
            print(f"Import error: {e}")
            return False
        except Exception as e:
            error_str = str(e).lower()
            print(f"Failed to load Stable Diffusion: {e}")
            
            # If CUDA OOM, try CPU
            if "cuda" in error_str and ("memory" in error_str or "oom" in error_str):
                print("GPU out of memory, trying CPU...")
                try:
                    from diffusers import StableDiffusionPipeline
                    import torch
                    torch.cuda.empty_cache()
                    
                    self.pipe = StableDiffusionPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                    ).to("cpu")
                    
                    self.is_loaded = True
                    print("Stable Diffusion loaded on CPU!")
                    return True
                except Exception as cpu_e:
                    print(f"CPU fallback also failed: {cpu_e}")
            
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
        self.is_loaded = False
    
    def generate(self, prompt: str, width: int = 512, height: int = 512,
                 steps: int = 30, guidance: float = 7.5, 
                 negative_prompt: str = "", **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}
        
        try:
            start = time.time()
            
            # Ensure prompt is a string (CLIP tokenizer requires str type)
            if prompt is None:
                return {"success": False, "error": "Prompt cannot be None"}
            prompt = str(prompt).strip()
            if not prompt:
                return {"success": False, "error": "Prompt cannot be empty"}
            
            # Handle negative prompt - must be str or None
            neg_prompt = str(negative_prompt).strip() if negative_prompt else None
            if neg_prompt == "":
                neg_prompt = None
            
            result = self.pipe(
                prompt,
                negative_prompt=neg_prompt,
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
    'placeholder': None,
    'local': None,
    'openai': None,
    'replicate': None,
}

# Track load errors for better messages
_load_errors = {}


def get_provider(name: str):
    """Get or create a provider instance."""
    global _providers
    
    if name == 'placeholder' and _providers['placeholder'] is None:
        _providers['placeholder'] = PlaceholderImage()
    elif name == 'local' and _providers['local'] is None:
        _providers['local'] = StableDiffusionLocal()
    elif name == 'openai' and _providers['openai'] is None:
        _providers['openai'] = OpenAIImage()
    elif name == 'replicate' and _providers['replicate'] is None:
        _providers['replicate'] = ReplicateImage()
    
    return _providers.get(name)


def preload_local_provider(callback=None):
    """
    Preload the local Stable Diffusion provider in a background thread.
    
    Args:
        callback: Optional function to call with (success: bool, message: str) when done
    
    Returns:
        The thread that's doing the loading (can be joined if needed)
    """
    import threading
    
    def _load():
        try:
            provider = get_provider('local')
            if provider and not provider.is_loaded:
                success = provider.load()
                if callback:
                    msg = "Stable Diffusion loaded successfully" if success else "Failed to load Stable Diffusion"
                    callback(success, msg)
            elif provider and provider.is_loaded:
                if callback:
                    callback(True, "Stable Diffusion already loaded")
        except Exception as e:
            if callback:
                callback(False, f"Error loading Stable Diffusion: {e}")
    
    thread = threading.Thread(target=_load, daemon=True, name="SDPreloader")
    thread.start()
    return thread


def is_local_provider_loaded() -> bool:
    """Check if the local Stable Diffusion provider is loaded."""
    provider = _providers.get('local')
    return provider is not None and provider.is_loaded


def get_load_error(name: str) -> str:
    """Get the last load error for a provider."""
    return _load_errors.get(name, "")


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
        self._stop_requested = False
    
    def request_stop(self):
        """Request the worker to stop."""
        self._stop_requested = True
    
    def run(self):
        global _load_errors
        try:
            if self._stop_requested:
                self.finished.emit({"success": False, "error": "Cancelled by user"})
                return
                
            self.progress.emit(10)
            
            # Check if router has image assignments - use router if configured
            try:
                from ...core.tool_router import get_router
                router = get_router()
                assignments = router.get_assignments("image")
                
                if assignments:
                    if self._stop_requested:
                        self.finished.emit({"success": False, "error": "Cancelled by user"})
                        return
                    # Use router to execute with assigned model
                    self.progress.emit(30)
                    params = {
                        "prompt": str(self.prompt).strip() if self.prompt else "",
                        "width": int(self.width),
                        "height": int(self.height),
                        "steps": int(self.steps),
                        "guidance": float(self.guidance),
                        "negative_prompt": str(self.negative_prompt).strip() if self.negative_prompt else ""
                    }
                    result = router.execute_tool("image", params)
                    self.progress.emit(100)
                    self.finished.emit(result)
                    return
            except Exception as router_error:
                # Router not available or failed, fall back to direct provider
                print(f"Router fallback: {router_error}")
            
            if self._stop_requested:
                self.finished.emit({"success": False, "error": "Cancelled by user"})
                return
                
            # Direct provider fallback
            provider = get_provider(self.provider_name)
            if provider is None:
                self.finished.emit({"success": False, "error": f"Unknown provider: {self.provider_name}"})
                return
            
            # Load if needed
            if not provider.is_loaded:
                self.progress.emit(20)
                if not provider.load():
                    # Build helpful error message
                    if self.provider_name == 'local':
                        error_msg = (
                            "Failed to load Stable Diffusion.\n\n"
                            "To fix, install: pip install diffusers transformers accelerate\n\n"
                            "Or try 'Placeholder' provider to test without dependencies."
                        )
                    elif self.provider_name == 'openai':
                        error_msg = (
                            "Failed to load OpenAI DALL-E.\n\n"
                            "Make sure you have:\n"
                            "1. pip install openai\n"
                            "2. Set OPENAI_API_KEY environment variable"
                        )
                    elif self.provider_name == 'replicate':
                        error_msg = (
                            "Failed to load Replicate.\n\n"
                            "Make sure you have:\n"
                            "1. pip install replicate\n"
                            "2. Set REPLICATE_API_TOKEN environment variable"
                        )
                    else:
                        error_msg = f"Failed to load provider: {self.provider_name}"
                    
                    _load_errors[self.provider_name] = error_msg
                    self.finished.emit({"success": False, "error": error_msg})
                    return
            
            if self._stop_requested:
                self.finished.emit({"success": False, "error": "Cancelled by user"})
                return
                
            self.progress.emit(40)
            
            # Generate - ensure prompt is definitely a string
            prompt_str = str(self.prompt).strip() if self.prompt else ""
            if not prompt_str:
                self.finished.emit({"success": False, "error": "Prompt cannot be empty"})
                return
            
            neg_prompt_str = str(self.negative_prompt).strip() if self.negative_prompt else ""
            
            result = provider.generate(
                prompt_str,
                width=int(self.width),
                height=int(self.height),
                steps=int(self.steps),
                guidance=float(self.guidance),
                negative_prompt=neg_prompt_str,
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
        
        # Register references on parent window for chat integration
        if parent:
            parent.image_prompt = self.prompt_input
            parent.image_tab = self
            parent._generate_image = self._generate_image
        
        # Check ready status on init and periodically
        self._check_ready_status()
    
    def _check_ready_status(self):
        """Check if image generation is ready and update indicator."""
        try:
            provider = get_provider('local')
            if provider and provider.is_loaded:
                self.ready_indicator.setText("[OK] Ready")
                self.ready_indicator.setStyleSheet("color: #2ecc71; font-size: 11px;")
            else:
                self.ready_indicator.setText("[!] Not Loaded")
                self.ready_indicator.setStyleSheet("color: #f39c12; font-size: 11px;")
        except Exception:
            self.ready_indicator.setText("[X] Not Ready")
            self.ready_indicator.setStyleSheet("color: #e74c3c; font-size: 11px;")
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header with status
        header_layout = QHBoxLayout()
        header = QLabel("Image Generation")
        header.setObjectName("header")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        # Ready indicator
        self.ready_indicator = QLabel("[X] Not Ready")
        self.ready_indicator.setStyleSheet("color: #e74c3c; font-size: 11px;")
        header_layout.addWidget(self.ready_indicator)
        
        layout.addLayout(header_layout)
        
        # Result area at TOP
        self.result_label = QLabel("Generated image will appear here")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumHeight(200)
        self.result_label.setStyleSheet("background-color: #2d2d2d; border-radius: 4px;")
        layout.addWidget(self.result_label, stretch=1)
        
        # Progress and Status
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Prompt input
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout()
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setMaximumHeight(60)
        self.prompt_input.setPlaceholderText("Describe the image you want to generate...")
        prompt_layout.addWidget(self.prompt_input)
        
        self.neg_prompt_input = QTextEdit()
        self.neg_prompt_input.setMaximumHeight(40)
        self.neg_prompt_input.setPlaceholderText("Negative prompt (what to avoid)...")
        prompt_layout.addWidget(self.neg_prompt_input)
        
        # Reference image input
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("Reference:"))
        self.ref_image_path = QLineEdit()
        self.ref_image_path.setPlaceholderText("Optional reference image for img2img")
        self.ref_image_path.setReadOnly(True)
        ref_layout.addWidget(self.ref_image_path)
        
        self.browse_ref_btn = QPushButton("Browse")
        self.browse_ref_btn.clicked.connect(self._browse_reference_image)
        ref_layout.addWidget(self.browse_ref_btn)
        
        self.clear_ref_btn = QPushButton("Clear")
        self.clear_ref_btn.clicked.connect(self._clear_reference_image)
        ref_layout.addWidget(self.clear_ref_btn)
        
        prompt_layout.addLayout(ref_layout)
        
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # Style Presets (from shared components)
        try:
            from .shared_components import PresetSelector, STYLE_PRESETS
            
            preset_row = QHBoxLayout()
            self.style_preset = PresetSelector("image", self)
            self.style_preset.preset_changed.connect(self._apply_style_preset)
            preset_row.addWidget(self.style_preset)
            preset_row.addStretch()
            layout.addLayout(preset_row)
            
            self._style_suffix = ""  # Will be appended to prompt
        except ImportError:
            self._style_suffix = ""
        
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
        
        # Auto-open options
        auto_layout = QHBoxLayout()
        
        self.auto_open_file_cb = QCheckBox("Auto-open file in explorer")
        self.auto_open_file_cb.setChecked(False)  # Don't auto-open folder by default
        self.auto_open_file_cb.setToolTip("Open the generated file in your file explorer when done")
        auto_layout.addWidget(self.auto_open_file_cb)
        
        self.auto_open_image_cb = QCheckBox("Auto-open image viewer")
        self.auto_open_image_cb.setChecked(False)
        self.auto_open_image_cb.setToolTip("Open the image in your default image viewer")
        auto_layout.addWidget(self.auto_open_image_cb)
        
        auto_layout.addStretch()
        layout.addLayout(auto_layout)
        
        # Generate button
        btn_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Generate Image")
        self.generate_btn.setStyleSheet("background-color: #e74c3c; font-weight: bold; padding: 10px;")
        self.generate_btn.clicked.connect(self._generate_image)
        btn_layout.addWidget(self.generate_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("background-color: #7f8c8d; font-weight: bold; padding: 10px;")
        self.stop_btn.clicked.connect(self._stop_generation)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Stop the current generation")
        btn_layout.addWidget(self.stop_btn)
        
        self.save_btn = QPushButton("Save As...")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_image)
        btn_layout.addWidget(self.save_btn)
        
        self.open_folder_btn = QPushButton("Open Output Folder")
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        btn_layout.addWidget(self.open_folder_btn)
        
        layout.addLayout(btn_layout)
    
    def _load_model(self):
        """Load the image generation model."""
        self.status_label.setText("Loading Stable Diffusion...")
        self.generate_btn.setEnabled(False)
        
        import threading
        
        def do_load():
            try:
                provider = get_provider('local')
                if provider:
                    success = provider.load()
                    from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                    if success:
                        QMetaObject.invokeMethod(
                            self.status_label, "setText",
                            Qt.QueuedConnection, 
                            Q_ARG(str, "[OK] Model loaded and ready!")
                        )
                        QMetaObject.invokeMethod(
                            self.ready_indicator, "setText",
                            Qt.QueuedConnection,
                            Q_ARG(str, "[OK] Ready")
                        )
                        QMetaObject.invokeMethod(
                            self.ready_indicator, "setStyleSheet",
                            Qt.QueuedConnection,
                            Q_ARG(str, "color: #2ecc71; font-size: 11px;")
                        )
                    else:
                        QMetaObject.invokeMethod(
                            self.status_label, "setText",
                            Qt.QueuedConnection,
                            Q_ARG(str, "[X] Failed to load model")
                        )
                    QMetaObject.invokeMethod(
                        self.generate_btn, "setEnabled",
                        Qt.QueuedConnection,
                        Q_ARG(bool, True)
                    )
            except Exception as e:
                from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(
                    self.status_label, "setText",
                    Qt.QueuedConnection,
                    Q_ARG(str, f"[X] Error: {e}")
                )
                QMetaObject.invokeMethod(
                    self.generate_btn, "setEnabled",
                    Qt.QueuedConnection,
                    Q_ARG(bool, True)
                )
        
        thread = threading.Thread(target=do_load, daemon=True)
        thread.start()
    
    def _browse_reference_image(self):
        """Browse for a reference image."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if path:
            self.ref_image_path.setText(path)
            self.status_label.setText(f"Reference image loaded: {Path(path).name}")
    
    def _clear_reference_image(self):
        """Clear the reference image."""
        self.ref_image_path.clear()
        self.status_label.setText("Reference image cleared")
    
    def _apply_style_preset(self, name: str, preset: dict):
        """Apply a style preset."""
        if name.startswith("__save__"):
            # User wants to save current settings as preset
            save_name = name.replace("__save__", "")
            current = {
                "width": self.width_spin.value(),
                "height": self.height_spin.value(),
                "steps": self.steps_spin.value(),
                "guidance": self.guidance_spin.value(),
                "suffix": self._style_suffix
            }
            if hasattr(self, 'style_preset'):
                self.style_preset.add_custom_preset(save_name, current)
            self.status_label.setText(f"Saved preset: {save_name}")
            return
            
        # Apply preset settings
        if "suffix" in preset:
            self._style_suffix = preset["suffix"]
        else:
            self._style_suffix = ""
        
        # Apply dimensions based on quality
        quality = preset.get("quality", "standard")
        if quality == "fast":
            self.width_spin.setValue(384)
            self.height_spin.setValue(384)
            self.steps_spin.setValue(15)
        elif quality == "high":
            self.width_spin.setValue(768)
            self.height_spin.setValue(768)
            self.steps_spin.setValue(50)
        
        self.status_label.setText(f"Applied style: {name}")
    
    def _generate_image(self):
        """Generate an image using available providers."""
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "No Prompt", "Please enter a prompt")
            return
        
        # Apply style suffix if set
        if hasattr(self, '_style_suffix') and self._style_suffix:
            prompt = prompt + self._style_suffix
        
        # Prefer LOCAL providers - only fall back to cloud if local unavailable
        # Local is private, free, and works offline
        provider_name = 'local'  # Default to local Stable Diffusion
        
        # Check if local provider can be used
        provider = get_provider('local')
        if provider and provider.is_loaded:
            provider_name = 'local'
        elif not provider:
            # Local not available, try placeholder as fallback
            provider_name = 'placeholder'
        
        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
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
    
    def _stop_generation(self):
        """Stop the current generation."""
        if self.worker and self.worker.isRunning():
            self.status_label.setText("Stopping generation...")
            self.worker.request_stop()
            self.stop_btn.setEnabled(False)
            
            # Force terminate if still running after 2 seconds
            from PyQt5.QtCore import QTimer
            def force_stop():
                if self.worker and self.worker.isRunning():
                    self.worker.terminate()
                    self.worker.wait(1000)
                    self._on_generation_complete({"success": False, "error": "Force stopped by user"})
            QTimer.singleShot(2000, force_stop)
    
    def _on_generation_complete(self, result: dict):
        """Handle generation completion."""
        self.generate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setVisible(False)
        
        if result.get("success"):
            path = result.get("path", "")
            duration = result.get("duration", 0)
            is_placeholder = result.get("is_placeholder", False)
            
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
                
                status = f"Generated in {duration:.1f}s - Saved to: {path}"
                if is_placeholder:
                    status += " (placeholder)"
                self.status_label.setText(status)
                
                # Auto-open file in explorer (select the file)
                if self.auto_open_file_cb.isChecked():
                    self._open_file_in_explorer(path)
                
                # Auto-open in image viewer
                if self.auto_open_image_cb.isChecked():
                    self._open_in_default_viewer(path)
                
                # Show popup preview (from main window if available)
                self._show_popup_preview(path)
                
                # Update ready status
                self._check_ready_status()
            else:
                self.status_label.setText("Generation complete (no image path)")
        else:
            error = result.get("error", "Unknown error")
            self.status_label.setText(f"Error: {error}")
            self.result_label.setText(f"Generation failed:\n{error}\n\nTo fix, install: pip install diffusers transformers accelerate")
            self._check_ready_status()
    
    def _show_popup_preview(self, path: str):
        """Show a popup preview of the generated image."""
        try:
            # Try to get the main window's popup function
            main_window = self.window()
            if hasattr(main_window, '_show_generation_popup'):
                main_window._show_generation_popup(path, 'image')
            else:
                # Fallback: create popup directly
                from ..enhanced_window import GenerationPreviewPopup
                popup = GenerationPreviewPopup(
                    parent=self,
                    result_path=path,
                    result_type='image'
                )
                popup.show()
        except Exception as e:
            print(f"Could not show preview popup: {e}")
    
    def _open_file_in_explorer(self, path: str):
        """Open file explorer with the file selected."""
        from .output_helpers import open_file_in_explorer
        open_file_in_explorer(path)
    
    def _open_in_default_viewer(self, path: str):
        """Open file in the default application."""
        from .output_helpers import open_in_default_viewer
        open_in_default_viewer(path)
    
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
        from .output_helpers import open_folder
        open_folder(OUTPUT_DIR)


def create_image_tab(parent) -> QWidget:
    """Factory function for creating the image tab."""
    if not HAS_PYQT:
        raise ImportError("PyQt5 is required for the Image Tab")
    return ImageTab(parent)
