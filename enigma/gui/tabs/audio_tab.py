"""
Audio Generation Tab - Generate audio/speech using local or cloud models.
"""

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
        QPushButton, QFrame, QComboBox, QTextEdit, QProgressBar,
        QMessageBox, QFileDialog, QDoubleSpinBox, QSlider, QGroupBox
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
    'ELEVENLABS': '#1a1a2e',
    'REPLICATE': '#000000',
    'OPENAI': '#10a37f',
}


class AudioGenerationWorker(QThread):
    """Background worker for audio generation."""
    finished = pyqtSignal(str)  # Path to generated audio
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, text, voice, provider, parent=None):
        super().__init__(parent)
        self.text = text
        self.voice = voice
        self.provider = provider
    
    def run(self):
        try:
            self.progress.emit(10)
            
            # Try to use module manager if available
            try:
                from ...modules.manager import ModuleManager
                manager = ModuleManager()
                
                if self.provider == 'LOCAL':
                    if manager.is_loaded('audio_gen_local'):
                        module = manager.get_module('audio_gen_local')
                        self.progress.emit(50)
                        result = module.speak(self.text, voice=self.voice)
                        self.progress.emit(100)
                        self.finished.emit(result.get('path', ''))
                        return
                else:
                    if manager.is_loaded('audio_gen_api'):
                        module = manager.get_module('audio_gen_api')
                        self.progress.emit(50)
                        result = module.speak(self.text, voice=self.voice)
                        self.progress.emit(100)
                        self.finished.emit(result.get('path', ''))
                        return
            except ImportError:
                pass
            
            # Try basic pyttsx3 fallback
            try:
                import pyttsx3
                engine = pyttsx3.init()
                
                # Save to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                
                self.progress.emit(50)
                engine.save_to_file(self.text, temp_path)
                engine.runAndWait()
                self.progress.emit(100)
                self.finished.emit(temp_path)
                return
            except ImportError:
                pass
            
            # Mock for demo
            self.progress.emit(100)
            self.finished.emit('')
            
        except Exception as e:
            self.error.emit(str(e))


class ProviderCard(QFrame):
    """Card displaying a single audio provider."""
    
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


class AudioTab(QWidget):
    """Tab for audio/speech generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.worker = None
        self.last_audio_path = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        
        # Left: Provider list
        left = QWidget()
        left.setMaximumWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        type_label = QLabel("Audio Providers")
        type_label.setFont(QFont('Arial', 12, QFont.Bold))
        type_label.setStyleSheet("color: #2ecc71;")
        left_layout.addWidget(type_label)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        cards_widget = QWidget()
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setSpacing(8)
        
        # Available providers
        providers = {
            'local_tts': {
                'name': 'Local TTS (pyttsx3)',
                'description': 'Offline text-to-speech. Basic quality, free.',
                'requirements': ['pyttsx3'],
                'provider': 'LOCAL',
            },
            'elevenlabs_tts': {
                'name': 'ElevenLabs',
                'description': 'Premium voice synthesis. Very realistic.',
                'requirements': ['elevenlabs'],
                'provider': 'ELEVENLABS',
                'needs_api_key': True,
            },
            'openai_tts': {
                'name': 'OpenAI TTS',
                'description': 'OpenAI text-to-speech voices.',
                'requirements': ['openai'],
                'provider': 'OPENAI',
                'needs_api_key': True,
            },
            'replicate_music': {
                'name': 'MusicGen',
                'description': 'Generate music from text prompts.',
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
        
        # Mode selection
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Text-to-Speech', 'Music Generation'])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        right_layout.addLayout(mode_row)
        
        # Provider selection
        provider_row = QHBoxLayout()
        provider_row.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(['Local (pyttsx3)', 'ElevenLabs', 'OpenAI TTS'])
        provider_row.addWidget(self.provider_combo)
        provider_row.addStretch()
        right_layout.addLayout(provider_row)
        
        # Text input
        text_label = QLabel("Text to Speak:")
        right_layout.addWidget(text_label)
        
        self.text_input = QTextEdit()
        self.text_input.setMaximumHeight(100)
        self.text_input.setPlaceholderText(
            "Enter the text you want to convert to speech..."
        )
        right_layout.addWidget(self.text_input)
        
        # Voice settings group
        voice_group = QGroupBox("Voice Settings")
        voice_layout = QVBoxLayout(voice_group)
        
        # Voice selection
        voice_row = QHBoxLayout()
        voice_row.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(['Default', 'Male', 'Female'])
        voice_row.addWidget(self.voice_combo)
        voice_row.addStretch()
        voice_layout.addLayout(voice_row)
        
        # Speed control
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(100)
        speed_row.addWidget(self.speed_slider)
        self.speed_label = QLabel("100%")
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_label.setText(f"{v}%")
        )
        speed_row.addWidget(self.speed_label)
        voice_layout.addLayout(speed_row)
        
        right_layout.addWidget(voice_group)
        
        # Generate button
        btn_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Audio")
        self.generate_btn.setStyleSheet("background-color: #2ecc71; font-weight: bold;")
        self.generate_btn.clicked.connect(self._generate_audio)
        btn_layout.addWidget(self.generate_btn)
        
        self.play_btn = QPushButton("Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._play_audio)
        btn_layout.addWidget(self.play_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_audio)
        btn_layout.addWidget(self.save_btn)
        
        right_layout.addLayout(btn_layout)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        right_layout.addWidget(self.progress)
        
        # Status area
        self.status_label = QLabel("Audio will be generated here")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(100)
        self.status_label.setStyleSheet("background-color: #2d2d2d; border-radius: 4px;")
        right_layout.addWidget(self.status_label, stretch=1)
        
        layout.addWidget(right, stretch=1)
    
    def _on_mode_changed(self, mode: str):
        """Update UI when mode changes."""
        if mode == 'Music Generation':
            self.provider_combo.clear()
            self.provider_combo.addItems(['MusicGen (Replicate)'])
            self.text_input.setPlaceholderText(
                "Describe the music you want to generate...\n"
                "Example: Upbeat electronic music with synths"
            )
        else:
            self.provider_combo.clear()
            self.provider_combo.addItems(['Local (pyttsx3)', 'ElevenLabs', 'OpenAI TTS'])
            self.text_input.setPlaceholderText(
                "Enter the text you want to convert to speech..."
            )
    
    def _generate_audio(self):
        """Generate audio."""
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Text", "Please enter some text")
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
        self.status_label.setText("Generating audio...")
        
        self.worker = AudioGenerationWorker(
            text,
            self.voice_combo.currentText().lower(),
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
            self.last_audio_path = path
            self.status_label.setText(f"Audio generated!\n{Path(path).name}")
            self.play_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
        else:
            self.status_label.setText(
                "Audio generation completed.\n\n"
                "Install pyttsx3 for local TTS:\n"
                "pip install pyttsx3"
            )
    
    def _on_generation_error(self, error: str):
        """Handle generation error."""
        self.generate_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText(f"Error: {error}")
        QMessageBox.warning(self, "Generation Failed", f"Error: {error}")
    
    def _play_audio(self):
        """Play the generated audio."""
        if not self.last_audio_path or not Path(self.last_audio_path).exists():
            return
        
        import subprocess
        import sys
        
        try:
            if sys.platform == 'linux':
                subprocess.Popen(['xdg-open', self.last_audio_path])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', self.last_audio_path])
            else:
                subprocess.Popen(['start', '', self.last_audio_path], shell=True)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open audio player: {e}")
    
    def _save_audio(self):
        """Save the generated audio."""
        if not self.last_audio_path:
            return
        
        # Get extension from source
        ext = Path(self.last_audio_path).suffix or '.wav'
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Audio",
            str(Path.home() / f"generated_audio{ext}"),
            f"Audio Files (*{ext});;All Files (*)"
        )
        if path:
            import shutil
            shutil.copy(self.last_audio_path, path)
            QMessageBox.information(self, "Saved", f"Audio saved to:\n{path}")


def create_audio_tab(parent) -> QWidget:
    """Factory function for creating the audio tab."""
    return AudioTab(parent)


if not HAS_PYQT:
    class AudioTab:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt5 is required for the Audio Tab")
    
    def create_audio_tab(parent):
        raise ImportError("PyQt5 is required for the Audio Tab")
