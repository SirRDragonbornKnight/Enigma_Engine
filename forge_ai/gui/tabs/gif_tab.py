"""
GIF Generation Tab - Create animated GIFs from images or prompts.

Provides easy GIF creation from:
  - Text prompts (generates multiple frames)
  - Existing images (animate sequence)
  - Image variations (morph between prompts)
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QProgressBar,
    QMessageBox, QFileDialog, QSpinBox, QGroupBox,
    QDoubleSpinBox, QListWidget, QListWidgetItem, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap

from .shared_components import NoScrollComboBox

HAS_PYQT = True

from ...config import CONFIG

# Output directory
OUTPUT_DIR = Path(CONFIG.get("outputs_dir", "outputs")) / "gifs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class GIFGenerationWorker(QThread):
    """Background worker for GIF generation."""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int, str)
    
    def __init__(self, prompts: List[str], frames_per_prompt: int,
                 width: int, height: int, fps: int, loop: bool, parent=None):
        super().__init__(parent)
        self.prompts = prompts
        self.frames_per_prompt = frames_per_prompt
        self.width = width
        self.height = height
        self.fps = fps
        self.loop = loop
    
    def run(self):
        try:
            from PIL import Image
            import io
            
            self.progress.emit(5, "Starting GIF generation...")
            
            frames = []
            total_frames = len(self.prompts) * self.frames_per_prompt
            
            # Try to use local image generation
            try:
                from .image_tab import get_provider
                provider = get_provider('local')
                
                if provider is None:
                    raise ImportError("No provider available")
                
                if not provider.is_loaded:
                    self.progress.emit(10, "Loading image model...")
                    if not provider.load():
                        self.finished.emit({
                            "success": False,
                            "error": "Failed to load image generation model"
                        })
                        return
                
                frame_idx = 0
                for prompt_idx, prompt in enumerate(self.prompts):
                    for i in range(self.frames_per_prompt):
                        progress_pct = int(20 + (frame_idx / total_frames) * 70)
                        self.progress.emit(progress_pct, f"Generating frame {frame_idx + 1}/{total_frames}...")
                        
                        # Generate image
                        result = provider.generate(
                            prompt,
                            width=self.width,
                            height=self.height,
                            steps=20,  # Fewer steps for speed
                            guidance=7.0,
                        )
                        
                        if result.get("success") and result.get("path"):
                            img = Image.open(result["path"])
                            frames.append(img.copy())
                        
                        frame_idx += 1
                        
            except ImportError:
                # Fallback: create placeholder frames
                self.progress.emit(30, "Image model not available, creating placeholder GIF...")
                
                for i in range(len(self.prompts) * self.frames_per_prompt):
                    # Create a simple colored frame
                    img = Image.new('RGB', (self.width, self.height), 
                                    color=(50 + i * 10, 100, 150))
                    frames.append(img)
            
            if not frames:
                self.finished.emit({
                    "success": False,
                    "error": "No frames generated"
                })
                return
            
            self.progress.emit(90, "Saving GIF...")
            
            # Save as GIF
            timestamp = int(time.time())
            filename = f"gif_{timestamp}.gif"
            filepath = OUTPUT_DIR / filename
            
            frames[0].save(
                str(filepath),
                format='GIF',
                save_all=True,
                append_images=frames[1:],
                duration=1000 // self.fps,
                loop=0 if self.loop else 1
            )
            
            self.progress.emit(100, "Done!")
            
            self.finished.emit({
                "success": True,
                "path": str(filepath),
                "frame_count": len(frames)
            })
            
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)})


class GIFTab(QWidget):
    """Tab for GIF generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.worker = None
        self.last_gif_path = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("GIF Generation")
        header.setObjectName("header")
        layout.addWidget(header)
        
        # Result area at TOP
        self.result_label = QLabel("Generated GIF will appear here")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setMinimumHeight(200)
        self.result_label.setStyleSheet("background-color: #2d2d2d; border-radius: 4px;")
        layout.addWidget(self.result_label, stretch=1)
        
        # Progress and Status
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Options row
        options_layout = QHBoxLayout()
        
        options_layout.addWidget(QLabel("Size:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 1024)
        self.width_spin.setValue(256)
        options_layout.addWidget(self.width_spin)
        
        options_layout.addWidget(QLabel("x"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 1024)
        self.height_spin.setValue(256)
        options_layout.addWidget(self.height_spin)
        
        options_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(8)
        options_layout.addWidget(self.fps_spin)
        
        options_layout.addWidget(QLabel("Frames/prompt:"))
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(1, 10)
        self.frames_spin.setValue(1)
        options_layout.addWidget(self.frames_spin)
        
        options_layout.addStretch()
        layout.addLayout(options_layout)
        
        # Prompts input - compact
        prompts_layout = QVBoxLayout()
        prompts_layout.addWidget(QLabel("Animation Prompts (one per line):"))
        
        self.prompts_input = QTextEdit()
        self.prompts_input.setMaximumHeight(80)
        self.prompts_input.setPlaceholderText("a cat sitting\\na cat standing\\na cat jumping")
        prompts_layout.addWidget(self.prompts_input)
        layout.addLayout(prompts_layout)
        
        # Import images - compact row
        import_layout = QHBoxLayout()
        import_layout.addWidget(QLabel("Or Import:"))
        
        self.import_btn = QPushButton("Import Images")
        self.import_btn.clicked.connect(self._import_images)
        import_layout.addWidget(self.import_btn)
        
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(50)
        import_layout.addWidget(self.image_list)
        
        self.clear_images_btn = QPushButton("Clear")
        self.clear_images_btn.clicked.connect(lambda: self.image_list.clear())
        import_layout.addWidget(self.clear_images_btn)
        layout.addLayout(import_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Generate GIF")
        self.generate_btn.setStyleSheet("background-color: #f39c12; font-weight: bold; padding: 8px;")
        self.generate_btn.clicked.connect(self._generate_gif)
        btn_layout.addWidget(self.generate_btn)
        
        self.gif_from_images_btn = QPushButton("GIF from Images")
        self.gif_from_images_btn.clicked.connect(self._gif_from_imported)
        btn_layout.addWidget(self.gif_from_images_btn)
        
        self.save_btn = QPushButton("Save As")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_gif)
        btn_layout.addWidget(self.save_btn)
        
        self.open_folder_btn = QPushButton("Output Folder")
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        btn_layout.addWidget(self.open_folder_btn)
        
        layout.addLayout(btn_layout)
        
        # Store imported image paths
        self.imported_images = []
    
    def _import_images(self):
        """Import images for GIF creation."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.gif *.bmp)"
        )
        if paths:
            self.imported_images.extend(paths)
            for path in paths:
                self.image_list.addItem(Path(path).name)
    
    def _generate_gif(self):
        """Generate GIF from prompts."""
        text = self.prompts_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Prompts", "Please enter at least one prompt")
            return
        
        prompts = [p.strip() for p in text.split('\n') if p.strip()]
        if not prompts:
            QMessageBox.warning(self, "No Prompts", "Please enter valid prompts")
            return
        
        self.generate_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Starting generation...")
        
        self.worker = GIFGenerationWorker(
            prompts,
            self.frames_spin.value(),
            self.width_spin.value(),
            self.height_spin.value(),
            self.fps_spin.value(),
            loop=True
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_generation_complete)
        self.worker.start()
    
    def _gif_from_imported(self):
        """Create GIF from imported images."""
        if not self.imported_images:
            QMessageBox.warning(self, "No Images", "Import images first")
            return
        
        try:
            from PIL import Image
            
            self.status_label.setText("Creating GIF from images...")
            
            frames = []
            for path in self.imported_images:
                img = Image.open(path)
                # Resize to match target size
                img = img.resize((self.width_spin.value(), self.height_spin.value()))
                frames.append(img)
            
            if frames:
                timestamp = int(time.time())
                filename = f"gif_{timestamp}.gif"
                filepath = OUTPUT_DIR / filename
                
                frames[0].save(
                    str(filepath),
                    format='GIF',
                    save_all=True,
                    append_images=frames[1:],
                    duration=1000 // self.fps_spin.value(),
                    loop=0
                )
                
                self.last_gif_path = str(filepath)
                self._display_gif(filepath)
                self.status_label.setText(f"GIF created: {filepath}")
                self.save_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create GIF: {e}")
    
    def _on_progress(self, value: int, message: str):
        """Handle progress updates."""
        self.progress.setValue(value)
        self.status_label.setText(message)
    
    def _on_generation_complete(self, result: dict):
        """Handle generation completion."""
        self.generate_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        if result.get("success"):
            path = result.get("path", "")
            frame_count = result.get("frame_count", 0)
            
            if path and Path(path).exists():
                self.last_gif_path = path
                self._display_gif(Path(path))
                self.save_btn.setEnabled(True)
                self.status_label.setText(f"Generated {frame_count} frames - Saved to: {path}")
            else:
                self.status_label.setText("Generation complete (no output path)")
        else:
            error = result.get("error", "Unknown error")
            self.status_label.setText(f"Error: {error}")
            self.result_label.setText(f"Generation failed:\n{error}")
    
    def _display_gif(self, filepath: Path):
        """Display the generated GIF."""
        try:
            from PyQt5.QtGui import QMovie
            
            movie = QMovie(str(filepath))
            self.result_label.setMovie(movie)
            movie.start()
        except Exception:
            # Fallback to static image
            pixmap = QPixmap(str(filepath))
            scaled = pixmap.scaled(
                self.result_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.result_label.setPixmap(scaled)
    
    def _save_gif(self):
        """Save the generated GIF to a custom location."""
        if not self.last_gif_path:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save GIF",
            str(Path.home() / "animation.gif"),
            "GIF Images (*.gif)"
        )
        if path:
            import shutil
            shutil.copy(self.last_gif_path, path)
            QMessageBox.information(self, "Saved", f"GIF saved to:\n{path}")
    
    def _open_output_folder(self):
        """Open the output folder in file manager."""
        from .output_helpers import open_folder
        open_folder(OUTPUT_DIR)


def create_gif_tab(parent) -> QWidget:
    """Factory function for creating the GIF tab."""
    if not HAS_PYQT:
        raise ImportError("PyQt5 is required for the GIF Tab")
    return GIFTab(parent)
