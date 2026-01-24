"""
Video Generation Tab - Generate videos using local or cloud models.

Providers:
  - LOCAL: AnimateDiff (or built-in GIF fallback)
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
        QDoubleSpinBox, QLineEdit, QCheckBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from ...config import CONFIG
from .output_helpers import open_file_in_explorer, open_in_default_viewer, open_folder

# Output directory
OUTPUT_DIR = Path(CONFIG.get("outputs_dir", "outputs")) / "videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Video Generation Implementations
# =============================================================================

class LocalVideo:
    """Local video generation using AnimateDiff with built-in fallback."""
    
    def __init__(self, model_id: str = "guoyww/animatediff-motion-adapter-v1-5-2"):
        self.model_id = model_id
        self.pipe = None
        self.is_loaded = False
        self._using_builtin = False
        self._builtin_video = None
    
    def load(self) -> bool:
        # Try AnimateDiff first
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
            self._using_builtin = False
            return True
        except ImportError:
            pass
        except Exception as e:
            print(f"AnimateDiff not available: {e}")
        
        # Fall back to built-in video generator
        try:
            from ...builtin import BuiltinVideoGen
            self._builtin_video = BuiltinVideoGen()
            if self._builtin_video.load():
                self.is_loaded = True
                self._using_builtin = True
                print("Using built-in video generator (animated GIF)")
                return True
        except Exception as e:
            print(f"Built-in video gen failed: {e}")
        
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
        if self._builtin_video:
            self._builtin_video.unload()
            self._builtin_video = None
        self.is_loaded = False
        self._using_builtin = False
    
    def generate(self, prompt: str, duration: float = 2.0, fps: int = 8,
                 **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}
        
        # Use built-in if available
        if self._using_builtin:
            timestamp = int(time.time())
            filepath = str(OUTPUT_DIR / f"video_{timestamp}.gif")
            result = self._builtin_video.generate(prompt, frames=int(duration * fps), duration=duration)
            if result.get("success") and result.get("video_data"):
                with open(filepath, 'wb') as f:
                    f.write(result["video_data"])
                result["path"] = filepath
            return result
        
        try:
            start = time.time()
            
            # Ensure prompt is a string (CLIP tokenizer requires str type)
            if prompt is not None:
                prompt = str(prompt).strip()
            if not prompt:
                return {"success": False, "error": "Prompt cannot be empty"}
            
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
        self._stop_requested = False
    
    def request_stop(self):
        """Request the worker to stop."""
        self._stop_requested = True
    
    def run(self):
        try:
            if self._stop_requested:
                self.finished.emit({"success": False, "error": "Cancelled by user"})
                return
                
            self.progress.emit(10)
            
            # Check if router has video assignments - use router if configured
            try:
                from ...core.tool_router import get_router
                router = get_router()
                assignments = router.get_assignments("video")
                
                if assignments:
                    if self._stop_requested:
                        self.finished.emit({"success": False, "error": "Cancelled by user"})
                        return
                    self.progress.emit(30)
                    params = {
                        "prompt": str(self.prompt).strip() if self.prompt else "",
                        "duration": float(self.duration),
                        "fps": int(self.fps)
                    }
                    result = router.execute_tool("video", params)
                    self.progress.emit(100)
                    self.finished.emit(result)
                    return
            except Exception as router_error:
                print(f"Router fallback: {router_error}")
            
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
                if not provider.load():
                    self.finished.emit({"success": False, "error": "Failed to load provider"})
                    return
            
            if self._stop_requested:
                self.finished.emit({"success": False, "error": "Cancelled by user"})
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
        
        # Register references on parent window for chat integration
        if parent:
            parent.video_prompt = self.prompt_input
            parent.video_tab = self
            parent._generate_video = self._generate_video
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Video Generation")
        header.setObjectName("header")
        layout.addWidget(header)
        
        # Output preview at TOP
        self.preview_label = QLabel("Generated video will appear here")
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
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(['Local (AnimateDiff)', 'Replicate (Cloud)'])
        self.provider_combo.setToolTip("Local: Uses AnimateDiff (requires GPU)\nCloud: Uses Replicate API (requires API key)")
        settings_layout.addWidget(self.provider_combo)
        
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self._load_provider)
        self.load_btn.setToolTip("Load the selected video generation provider")
        settings_layout.addWidget(self.load_btn)
        
        settings_layout.addWidget(QLabel("Duration:"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.5, 10.0)
        self.duration_spin.setValue(1.0)  # Lower default for GPU memory
        self.duration_spin.setSuffix("s")
        self.duration_spin.setToolTip("Lower values use less GPU memory (1-2s recommended for 8GB VRAM)")
        settings_layout.addWidget(self.duration_spin)
        
        settings_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(4, 30)
        self.fps_spin.setValue(6)  # Lower default for GPU memory
        self.fps_spin.setToolTip("Lower FPS = fewer frames = less GPU memory (6-8 recommended for 8GB VRAM)")
        settings_layout.addWidget(self.fps_spin)
        
        settings_layout.addStretch()
        layout.addLayout(settings_layout)
        
        # Prompt - compact
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("Prompt:"))
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Describe the video you want to generate...")
        self.prompt_input.setToolTip("Describe the video scene, motion, and style.\nExample: 'A cat walking through a forest, cinematic'")
        prompt_layout.addWidget(self.prompt_input)
        layout.addLayout(prompt_layout)
        
        # Reference - compact
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("Reference:"))
        self.ref_input_path = QLineEdit()
        self.ref_input_path.setPlaceholderText("Optional reference video/image")
        self.ref_input_path.setReadOnly(True)
        self.ref_input_path.setToolTip("Optional: Select a reference video or image as a starting point")
        ref_layout.addWidget(self.ref_input_path)
        
        browse_ref_btn = QPushButton("Browse")
        browse_ref_btn.clicked.connect(self._browse_reference)
        browse_ref_btn.setToolTip("Browse for a reference video or image")
        ref_layout.addWidget(browse_ref_btn)
        
        clear_ref_btn = QPushButton("Clear")
        clear_ref_btn.clicked.connect(self._clear_reference)
        clear_ref_btn.setToolTip("Clear the reference file")
        ref_layout.addWidget(clear_ref_btn)
        layout.addLayout(ref_layout)
        
        # Auto-open options
        auto_layout = QHBoxLayout()
        self.auto_open_file_cb = QCheckBox("Auto-open file in explorer")
        self.auto_open_file_cb.setChecked(True)
        self.auto_open_file_cb.setToolTip("Automatically open the saved file location")
        auto_layout.addWidget(self.auto_open_file_cb)
        self.auto_open_viewer_cb = QCheckBox("Auto-open in default app")
        self.auto_open_viewer_cb.setChecked(False)
        self.auto_open_viewer_cb.setToolTip("Automatically open the video in your default media player")
        auto_layout.addWidget(self.auto_open_viewer_cb)
        auto_layout.addStretch()
        layout.addLayout(auto_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Generate Video")
        self.generate_btn.setStyleSheet("background-color: #9b59b6; font-weight: bold; padding: 8px;")
        self.generate_btn.clicked.connect(self._generate_video)
        self.generate_btn.setToolTip("Start generating a video from your prompt")
        btn_layout.addWidget(self.generate_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("background-color: #e74c3c; font-weight: bold; padding: 8px;")
        self.stop_btn.clicked.connect(self._stop_generation)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Stop the current generation (may take a moment)")
        btn_layout.addWidget(self.stop_btn)
        
        self.open_btn = QPushButton("Open")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_video)
        self.open_btn.setToolTip("Open the generated video in your default player")
        btn_layout.addWidget(self.open_btn)
        
        self.save_btn = QPushButton("Save As")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_video)
        self.save_btn.setToolTip("Save the video to a custom location")
        btn_layout.addWidget(self.save_btn)
        
        self.open_folder_btn = QPushButton("Output Folder")
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        self.open_folder_btn.setToolTip("Open the folder where videos are saved")
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
    
    def _generate_video(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "No Prompt", "Please enter a prompt")
            return
        
        provider_name = self._get_provider_name()
        
        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
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
        self.generate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setVisible(False)
        
        if result.get("success"):
            path = result.get("path", "")
            duration = result.get("duration", 0)
            
            self.last_video_path = path
            self.open_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.status_label.setText(f"Generated in {duration:.1f}s - Saved to: {path}")
            
            # Auto-open features
            if self.auto_open_file_cb.isChecked():
                open_file_in_explorer(path)
            if self.auto_open_viewer_cb.isChecked():
                open_in_default_viewer(path)
        else:
            error = result.get("error", "Unknown error")
            self.status_label.setText(f"Error: {error}")
    
    def _open_video(self):
        if self.last_video_path and Path(self.last_video_path).exists():
            open_in_default_viewer(self.last_video_path)
    
    def _save_video(self):
        """Save the generated video to a custom location."""
        if not self.last_video_path:
            return
        
        ext = Path(self.last_video_path).suffix
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Video",
            str(Path.home() / f"generated_video{ext}"),
            f"Video Files (*{ext});;All Files (*.*)"
        )
        if path:
            import shutil
            shutil.copy(self.last_video_path, path)
            QMessageBox.information(self, "Saved", f"Video saved to:\n{path}")
    
    def _open_output_folder(self):
        open_folder(OUTPUT_DIR)
    
    def _browse_reference(self):
        """Browse for a reference video or image."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Video/Image",
            str(Path.home()),
            "Media Files (*.mp4 *.gif *.avi *.mov *.webm *.png *.jpg *.jpeg);;All Files (*.*)"
        )
        if path:
            self.ref_input_path.setText(path)
    
    def _clear_reference(self):
        """Clear the reference input."""
        self.ref_input_path.clear()


def create_video_tab(parent) -> QWidget:
    """Factory function for creating the video tab."""
    if not HAS_PYQT:
        raise ImportError("PyQt5 is required for the Video Tab")
    return VideoTab(parent)
