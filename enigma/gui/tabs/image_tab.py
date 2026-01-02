"""
Image Generation Tab - Generate images using local or cloud models.
"""

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
        QPushButton, QFrame, QComboBox, QTextEdit, QProgressBar,
        QMessageBox, QFileDialog, QSpinBox, QGroupBox, QCheckBox,
        QDoubleSpinBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QPixmap
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from pathlib import Path
from ...config import CONFIG


# Provider colors for UI badges
PROVIDER_COLORS = {
    'LOCAL': '#27ae60',
    'OPENAI': '#10a37f',
    'STABILITY': '#7c3aed',
    'REPLICATE': '#000000',
    'HUGGINGFACE': '#ffcc00',
}


class ImageGenerationWorker(QThread):
    """Background worker for image generation."""
    finished = pyqtSignal(str)  # Path to generated image
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, prompt, width, height, provider, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.width = width
        self.height = height
        self.provider = provider
    
    def run(self):
        try:
            self.progress.emit(10)
            
            # Try to use module manager if available
            try:
                from ...modules.manager import ModuleManager
                manager = ModuleManager()
                
                if self.provider == 'LOCAL':
                    if manager.is_loaded('image_gen_local'):
                        module = manager.get_module('image_gen_local')
                        self.progress.emit(50)
                        result = module.generate(
                            self.prompt,
                            width=self.width,
                            height=self.height
                        )
                        self.progress.emit(100)
                        self.finished.emit(result.get('path', ''))
                        return
                else:
                    if manager.is_loaded('image_gen_api'):
                        module = manager.get_module('image_gen_api')
                        self.progress.emit(50)
                        result = module.generate(
                            self.prompt,
                            width=self.width,
                            height=self.height
                        )
                        self.progress.emit(100)
                        self.finished.emit(result.get('path', ''))
                        return
            except ImportError:
                pass
            
            # Mock generation for testing
            self.progress.emit(50)
            import time
            time.sleep(1)  # Simulate work
            self.progress.emit(100)
            self.finished.emit('')  # No image, just demo
            
        except Exception as e:
            self.error.emit(str(e))


class ProviderCard(QFrame):
    """Card displaying a single image provider."""
    
    def __init__(self, name: str, info: dict, parent=None):
        super().__init__(parent)
        self.provider_name = name
        self.provider_info = info
        self.setup_ui()
    
    def setup_ui(self):
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(1)
        self.setMaximumHeight(120)
        
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
        
        # Requirements
        reqs = self.provider_info.get('requirements', [])
        if reqs:
            req_label = QLabel(f"Requires: {', '.join(reqs)}")
            req_label.setStyleSheet("color: #666; font-size: 8px; font-style: italic;")
            layout.addWidget(req_label)


class ImageTab(QWidget):
    """Tab for image generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.worker = None
        self.last_image_path = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        
        # Left: Provider list
        left = QWidget()
        left.setMaximumWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        type_label = QLabel("Image Providers")
        type_label.setFont(QFont('Arial', 12, QFont.Bold))
        type_label.setStyleSheet("color: #e74c3c;")
        left_layout.addWidget(type_label)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        cards_widget = QWidget()
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setSpacing(8)
        
        # Available providers
        providers = {
            'stable_diffusion_local': {
                'name': 'Stable Diffusion (Local)',
                'description': 'Run SD locally on your GPU. Best for privacy and unlimited generations.',
                'requirements': ['torch', 'diffusers', 'transformers'],
                'provider': 'LOCAL',
            },
            'openai_dalle': {
                'name': 'DALL-E 3',
                'description': 'OpenAI\'s latest image model. High quality, costs per image.',
                'requirements': ['openai'],
                'provider': 'OPENAI',
                'needs_api_key': True,
            },
            'replicate_image': {
                'name': 'Replicate (SDXL/Flux)',
                'description': 'Cloud image generation. Pay per generation.',
                'requirements': ['replicate'],
                'provider': 'REPLICATE',
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
        self.provider_combo.addItems(['Local (Stable Diffusion)', 'OpenAI (DALL-E)', 'Replicate'])
        provider_row.addWidget(self.provider_combo)
        provider_row.addStretch()
        right_layout.addLayout(provider_row)
        
        # Prompt input
        prompt_label = QLabel("Prompt:")
        right_layout.addWidget(prompt_label)
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setMaximumHeight(80)
        self.prompt_input.setPlaceholderText("Describe the image you want to generate...")
        right_layout.addWidget(self.prompt_input)
        
        # Negative prompt
        neg_label = QLabel("Negative Prompt (optional):")
        right_layout.addWidget(neg_label)
        
        self.neg_prompt_input = QTextEdit()
        self.neg_prompt_input.setMaximumHeight(50)
        self.neg_prompt_input.setPlaceholderText("What to avoid in the image...")
        right_layout.addWidget(self.neg_prompt_input)
        
        # Options
        options = QHBoxLayout()
        options.addWidget(QLabel("Width:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(256, 2048)
        self.width_spin.setValue(512)
        self.width_spin.setSingleStep(64)
        options.addWidget(self.width_spin)
        
        options.addWidget(QLabel("Height:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(256, 2048)
        self.height_spin.setValue(512)
        self.height_spin.setSingleStep(64)
        options.addWidget(self.height_spin)
        
        options.addStretch()
        right_layout.addLayout(options)
        
        # Steps and guidance
        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 150)
        self.steps_spin.setValue(30)
        steps_row.addWidget(self.steps_spin)
        
        steps_row.addWidget(QLabel("Guidance:"))
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 20.0)
        self.guidance_spin.setValue(7.5)
        self.guidance_spin.setSingleStep(0.5)
        steps_row.addWidget(self.guidance_spin)
        
        steps_row.addStretch()
        right_layout.addLayout(steps_row)
        
        # Generate button
        btn_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Image")
        self.generate_btn.setStyleSheet("background-color: #e74c3c; font-weight: bold;")
        self.generate_btn.clicked.connect(self._generate_image)
        btn_layout.addWidget(self.generate_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_image)
        btn_layout.addWidget(self.save_btn)
        
        right_layout.addLayout(btn_layout)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        right_layout.addWidget(self.progress)
        
        # Result area
        self.result_label = QLabel("Generated image will appear here")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumHeight(200)
        self.result_label.setStyleSheet("background-color: #2d2d2d; border-radius: 4px;")
        self.result_label.setScaledContents(False)
        right_layout.addWidget(self.result_label, stretch=1)
        
        layout.addWidget(right, stretch=1)
    
    def _generate_image(self):
        """Generate an image."""
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "No Prompt", "Please enter a prompt")
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
        
        self.worker = ImageGenerationWorker(
            prompt,
            self.width_spin.value(),
            self.height_spin.value(),
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
            self.last_image_path = path
            pixmap = QPixmap(path)
            scaled = pixmap.scaled(
                self.result_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.result_label.setPixmap(scaled)
            self.save_btn.setEnabled(True)
        else:
            self.result_label.setText("Image generated (no display available)\nModule may not be loaded.")
            QMessageBox.information(
                self, "Generation Complete",
                "Image generation completed.\n\n"
                "To use this feature, load 'image_gen_local' or 'image_gen_api' "
                "module from the Modules tab."
            )
    
    def _on_generation_error(self, error: str):
        """Handle generation error."""
        self.generate_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.result_label.setText(f"Error: {error}")
        QMessageBox.warning(self, "Generation Failed", f"Error: {error}")
    
    def _save_image(self):
        """Save the generated image."""
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


def create_image_tab(parent) -> QWidget:
    """Factory function for creating the image tab."""
    return ImageTab(parent)


if not HAS_PYQT:
    class ImageTab:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt5 is required for the Image Tab")
    
    def create_image_tab(parent):
        raise ImportError("PyQt5 is required for the Image Tab")
