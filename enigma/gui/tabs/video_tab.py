"""
Video Generation Tab - Generate videos using local or cloud models.
"""

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
        QPushButton, QFrame, QComboBox, QTextEdit, QProgressBar,
        QMessageBox, QFileDialog, QDoubleSpinBox, QSpinBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from pathlib import Path
from ...config import CONFIG


# Provider colors for UI badges
PROVIDER_COLORS = {
    'LOCAL': '#27ae60',
    'REPLICATE': '#000000',
    'RUNWAY': '#7c3aed',
}


class VideoGenerationWorker(QThread):
    """Background worker for video generation."""
    finished = pyqtSignal(str)  # Path to generated video
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, prompt, duration, provider, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.duration = duration
        self.provider = provider
    
    def run(self):
        try:
            self.progress.emit(10)
            
            # Try to use module manager if available
            try:
                from ...modules.manager import ModuleManager
                manager = ModuleManager()
                
                if self.provider == 'LOCAL':
                    if manager.is_loaded('video_gen_local'):
                        module = manager.get_module('video_gen_local')
                        self.progress.emit(30)
                        result = module.generate(
                            self.prompt,
                            duration=self.duration
                        )
                        self.progress.emit(100)
                        self.finished.emit(result.get('path', ''))
                        return
                else:
                    if manager.is_loaded('video_gen_api'):
                        module = manager.get_module('video_gen_api')
                        self.progress.emit(30)
                        result = module.generate(
                            self.prompt,
                            duration=self.duration
                        )
                        self.progress.emit(100)
                        self.finished.emit(result.get('path', ''))
                        return
            except ImportError:
                pass
            
            # Mock generation for demo
            self.progress.emit(50)
            import time
            time.sleep(1)
            self.progress.emit(100)
            self.finished.emit('')
            
        except Exception as e:
            self.error.emit(str(e))


class ProviderCard(QFrame):
    """Card displaying a single video provider."""
    
    def __init__(self, name: str, info: dict, parent=None):
        super().__init__(parent)
        self.provider_name = name
        self.provider_info = info
        self.setup_ui()
    
    def setup_ui(self):
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(1)
        self.setMaximumHeight(100)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        
        # Header
        header = QHBoxLayout()
        
        name_label = QLabel(self.provider_info.get('name', self.provider_name))
        name_label.setFont(QFont('Arial', 10, QFont.Bold))
        header.addWidget(name_label)
        
        header.addStretch()
        
        # Provider badge
        provider = self.provider_info.get('provider', 'UNKNOWN')
        color = PROVIDER_COLORS.get(provider, '#666')
        provider_label = QLabel(provider)
        provider_label.setStyleSheet(
            f"background-color: {color}; color: white; "
            f"padding: 2px 6px; border-radius: 3px; font-size: 9px;"
        )
        header.addWidget(provider_label)
        
        layout.addLayout(header)
        
        # Description
        desc = self.provider_info.get('description', 'No description')
        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #888; font-size: 9px;")
        layout.addWidget(desc_label)


class VideoTab(QWidget):
    """Tab for video generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.worker = None
        self.last_video_path = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        
        # Left: Provider list
        left = QWidget()
        left.setMaximumWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        type_label = QLabel("Video Providers")
        type_label.setFont(QFont('Arial', 12, QFont.Bold))
        type_label.setStyleSheet("color: #9b59b6;")
        left_layout.addWidget(type_label)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        cards_widget = QWidget()
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setSpacing(8)
        
        # Available providers
        providers = {
            'local_video': {
                'name': 'AnimateDiff (Local)',
                'description': 'Generate videos locally. Requires good GPU.',
                'requirements': ['torch', 'diffusers'],
                'provider': 'LOCAL',
            },
            'replicate_video': {
                'name': 'Replicate Video',
                'description': 'Cloud video generation (Zeroscope, AnimateDiff).',
                'requirements': ['replicate'],
                'provider': 'REPLICATE',
                'needs_api_key': True,
            },
            'runway_video': {
                'name': 'Runway Gen-2',
                'description': 'High quality video generation via Runway.',
                'requirements': ['runway-python'],
                'provider': 'RUNWAY',
                'needs_api_key': True,
            },
        }
        
        for name, info in providers.items():
            card = ProviderCard(name, info)
            cards_layout.addWidget(card)
        
        cards_layout.addStretch()
        scroll.setWidget(cards_widget)
        left_layout.addWidget(scroll)
        
        layout.addWidget(left)
        
        # Right: Generation panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        # Provider selection
        provider_row = QHBoxLayout()
        provider_row.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(['AnimateDiff (Local)', 'Replicate', 'Runway Gen-2'])
        provider_row.addWidget(self.provider_combo)
        provider_row.addStretch()
        right_layout.addLayout(provider_row)
        
        # Prompt input
        prompt_label = QLabel("Video Description:")
        right_layout.addWidget(prompt_label)
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setMaximumHeight(80)
        self.prompt_input.setPlaceholderText(
            "Describe the video you want to generate...\n"
            "Example: A cat walking through a field of flowers, cinematic lighting"
        )
        right_layout.addWidget(self.prompt_input)
        
        # Options
        options = QHBoxLayout()
        options.addWidget(QLabel("Duration:"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1, 30)
        self.duration_spin.setValue(4)
        self.duration_spin.setSuffix(" sec")
        options.addWidget(self.duration_spin)
        
        options.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(8, 60)
        self.fps_spin.setValue(24)
        options.addWidget(self.fps_spin)
        
        options.addStretch()
        right_layout.addLayout(options)
        
        # Resolution
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(['512x512', '768x432', '1024x576', '1280x720'])
        self.resolution_combo.setCurrentIndex(1)
        res_row.addWidget(self.resolution_combo)
        res_row.addStretch()
        right_layout.addLayout(res_row)
        
        # Generate button
        btn_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Video")
        self.generate_btn.setStyleSheet("background-color: #9b59b6; font-weight: bold;")
        self.generate_btn.clicked.connect(self._generate_video)
        btn_layout.addWidget(self.generate_btn)
        
        self.play_btn = QPushButton("Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._play_video)
        btn_layout.addWidget(self.play_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_video)
        btn_layout.addWidget(self.save_btn)
        
        right_layout.addLayout(btn_layout)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        right_layout.addWidget(self.progress)
        
        # Status/Preview area
        self.status_label = QLabel("Video preview will appear here")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(200)
        self.status_label.setStyleSheet("background-color: #2d2d2d; border-radius: 4px;")
        right_layout.addWidget(self.status_label, stretch=1)
        
        # Info text
        info_label = QLabel(
            "Note: Video generation is resource-intensive.\n"
            "Local generation requires a powerful GPU (8GB+ VRAM recommended)."
        )
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        right_layout.addWidget(info_label)
        
        layout.addWidget(right, stretch=1)
    
    def _generate_video(self):
        """Generate a video."""
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "No Prompt", "Please describe the video")
            return
        
        # Determine provider
        provider_text = self.provider_combo.currentText()
        if 'Local' in provider_text:
            provider = 'LOCAL'
        else:
            provider = 'API'
        
        self.generate_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Generating video...\nThis may take several minutes.")
        
        self.worker = VideoGenerationWorker(
            prompt,
            self.duration_spin.value(),
            provider
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_generation_complete)
        self.worker.error.connect(self._on_generation_error)
        self.worker.start()
    
    def _on_generation_complete(self, path: str):
        """Handle generation completion."""
        self.generate_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        if path and Path(path).exists():
            self.last_video_path = path
            self.status_label.setText(f"Video generated!\n{path}")
            self.play_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
        else:
            self.status_label.setText(
                "Video generation completed.\n\n"
                "To use this feature, load 'video_gen_local' or 'video_gen_api'\n"
                "module from the Modules tab."
            )
            QMessageBox.information(
                self, "Generation Complete",
                "Video generation request completed.\n\n"
                "To enable full functionality, load the appropriate video module."
            )
    
    def _on_generation_error(self, error: str):
        """Handle generation error."""
        self.generate_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText(f"Error: {error}")
        QMessageBox.warning(self, "Generation Failed", f"Error: {error}")
    
    def _play_video(self):
        """Play the generated video."""
        if not self.last_video_path or not Path(self.last_video_path).exists():
            return
        
        import subprocess
        import sys
        
        try:
            if sys.platform == 'linux':
                subprocess.Popen(['xdg-open', self.last_video_path])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', self.last_video_path])
            else:
                subprocess.Popen(['start', '', self.last_video_path], shell=True)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open video player: {e}")
    
    def _save_video(self):
        """Save the generated video."""
        if not self.last_video_path:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Video",
            str(Path.home() / "generated_video.mp4"),
            "MP4 Videos (*.mp4);;All Files (*)"
        )
        if path:
            import shutil
            shutil.copy(self.last_video_path, path)
            QMessageBox.information(self, "Saved", f"Video saved to:\n{path}")


def create_video_tab(parent) -> QWidget:
    """Factory function for creating the video tab."""
    return VideoTab(parent)


if not HAS_PYQT:
    class VideoTab:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt5 is required for the Video Tab")
    
    def create_video_tab(parent):
        raise ImportError("PyQt5 is required for the Video Tab")
