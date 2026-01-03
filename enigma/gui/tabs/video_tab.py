"""
Video Generation Tab - Generate videos using local or cloud models.

Providers:
  - LOCAL: AnimateDiff (requires diffusers with video support)
  - REPLICATE: Cloud video generation (requires replicate, API key)
"""

import os
import time
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
    from PyQt5.QtGui import QFont
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from ...config import CONFIG

# Output directory
OUTPUT_DIR = Path(CONFIG.get("outputs_dir", "outputs")) / "videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Video Generation Implementations
# =============================================================================

class LocalVideo:
    """Local video generation using AnimateDiff."""
    
    def __init__(self, model_id: str = "guoyww/animatediff-motion-adapter-v1-5-2"):
        self.model_id = model_id
        self.pipe = None
        self.is_loaded = False
    
    def load(self) -> bool:
        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
            import torch
            
            adapter = MotionAdapter.from_pretrained(self.model_id)
            model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
            
            self.pipe = AnimateDiffPipeline.from_pretrained(
                model_id,
                motion_adapter=adapter,
            )
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config,
                beta_schedule="linear"
            )
            
            import torch
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
            
            self.is_loaded = True
            return True
        except ImportError:
            print("Install: pip install diffusers[torch] transformers accelerate")
            return False
        except Exception as e:
            print(f"Failed to load video model: {e}")
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
    
    def generate(self, prompt: str, duration: float = 2.0, fps: int = 8,
                 **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}
        
        try:
            start = time.time()
            
            num_frames = int(duration * fps)
            
            output = self.pipe(
                prompt,
                num_frames=num_frames,
                guidance_scale=7.5,
            )
            
            frames = output.frames[0]
            
            # Save as GIF
            timestamp = int(time.time())
            filename = f"video_{timestamp}.gif"
            filepath = OUTPUT_DIR / filename
            
            frames[0].save(
                str(filepath),
                format='GIF',
                save_all=True,
                append_images=frames[1:],
                duration=1000 // fps,
                loop=0
            )
            
            return {
                "success": True,
                "path": str(filepath),
                "duration": time.time() - start,
                "frames": len(frames)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class ReplicateVideo:
    """Replicate video generation (CLOUD - requires API key)."""
    
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "anotherjesse/zeroscope-v2-xl:latest"):
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
    
    def generate(self, prompt: str, duration: float = 4.0, fps: int = 24,
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
                    "num_frames": int(duration * fps),
                    "fps": fps,
                }
            )
            
            # Download video
            video_url = output if isinstance(output, str) else output[0]
            resp = requests.get(video_url)
            
            # Save to file
            timestamp = int(time.time())
            filename = f"replicate_video_{timestamp}.mp4"
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
        _providers['local'] = LocalVideo()
    elif name == 'replicate' and _providers['replicate'] is None:
        _providers['replicate'] = ReplicateVideo()
    
    return _providers.get(name)


class VideoGenerationWorker(QThread):
    """Background worker for video generation."""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    
    def __init__(self, prompt, duration, fps, provider_name, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.duration = duration
        self.fps = fps
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
                duration=self.duration,
                fps=self.fps
            )
            
            self.progress.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)})


class VideoTab(QWidget):
    """Tab for video generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.worker = None
        self.last_video_path = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Video Generation")
        header.setFont(QFont('Arial', 14, QFont.Bold))
        header.setStyleSheet("color: #9b59b6;")
        layout.addWidget(header)
        
        # Provider selection
        provider_group = QGroupBox("Provider")
        provider_layout = QHBoxLayout()
        
        self.provider_combo = QComboBox()
        self.provider_combo.addItems([
            'Local (AnimateDiff)',
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
        self.prompt_input.setPlaceholderText("Describe the video you want to generate...")
        prompt_layout.addWidget(self.prompt_input)
        
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout()
        
        options_layout.addWidget(QLabel("Duration (sec):"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.5, 10.0)
        self.duration_spin.setValue(2.0)
        self.duration_spin.setSingleStep(0.5)
        options_layout.addWidget(self.duration_spin)
        
        options_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(4, 30)
        self.fps_spin.setValue(8)
        options_layout.addWidget(self.fps_spin)
        
        options_layout.addStretch()
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Generate Video")
        self.generate_btn.setStyleSheet("background-color: #9b59b6; font-weight: bold; padding: 10px;")
        self.generate_btn.clicked.connect(self._generate_video)
        btn_layout.addWidget(self.generate_btn)
        
        self.open_btn = QPushButton("Open Video")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_video)
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
            "Note: Video generation requires significant GPU memory.\n"
            "Local generation may take several minutes on CPU."
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
    
    def _generate_video(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "No Prompt", "Please enter a prompt")
            return
        
        provider_name = self._get_provider_name()
        
        self.generate_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Generating video (this may take a while)...")
        
        self.worker = VideoGenerationWorker(
            prompt,
            self.duration_spin.value(),
            self.fps_spin.value(),
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
            
            self.last_video_path = path
            self.open_btn.setEnabled(True)
            self.status_label.setText(f"Generated in {duration:.1f}s - Saved to: {path}")
        else:
            error = result.get("error", "Unknown error")
            self.status_label.setText(f"Error: {error}")
    
    def _open_video(self):
        if self.last_video_path and Path(self.last_video_path).exists():
            import subprocess
            import sys
            
            if sys.platform == 'darwin':
                subprocess.run(['open', self.last_video_path])
            elif sys.platform == 'win32':
                os.startfile(self.last_video_path)
            else:
                subprocess.run(['xdg-open', self.last_video_path])
    
    def _open_output_folder(self):
        import subprocess
        import sys
        
        if sys.platform == 'darwin':
            subprocess.run(['open', str(OUTPUT_DIR)])
        elif sys.platform == 'win32':
            subprocess.run(['explorer', str(OUTPUT_DIR)])
        else:
            subprocess.run(['xdg-open', str(OUTPUT_DIR)])


def create_video_tab(parent) -> QWidget:
    """Factory function for creating the video tab."""
    if not HAS_PYQT:
        raise ImportError("PyQt5 is required for the Video Tab")
    return VideoTab(parent)
