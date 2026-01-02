"""
Audio Generation Tab - Text-to-speech and audio generation.

Providers:
  - LOCAL: pyttsx3 offline TTS
  - ELEVENLABS: High-quality cloud TTS (requires API key)
  - REPLICATE: Cloud audio generation (requires API key)
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QComboBox, QTextEdit, QProgressBar,
        QMessageBox, QGroupBox, QSlider, QFileDialog
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from ...config import CONFIG

# Output directory
OUTPUT_DIR = Path(CONFIG.get("outputs_dir", "outputs")) / "audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Audio/TTS Implementations
# =============================================================================

class LocalTTS:
    """Local text-to-speech using pyttsx3."""
    
    def __init__(self):
        self.engine = None
        self.is_loaded = False
        self.voices = []
        self.current_voice = 0
    
    def load(self) -> bool:
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.voices = self.engine.getProperty('voices')
            self.is_loaded = True
            return True
        except ImportError:
            print("Install: pip install pyttsx3")
            return False
        except Exception as e:
            print(f"Failed to load TTS: {e}")
            return False
    
    def unload(self):
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
            self.engine = None
        self.is_loaded = False
    
    def get_voices(self) -> list:
        if not self.is_loaded:
            return []
        return [v.name for v in self.voices]
    
    def set_voice(self, index: int):
        if self.is_loaded and 0 <= index < len(self.voices):
            self.engine.setProperty('voice', self.voices[index].id)
            self.current_voice = index
    
    def set_rate(self, rate: int):
        if self.is_loaded:
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float):
        if self.is_loaded:
            self.engine.setProperty('volume', volume)
    
    def speak(self, text: str) -> Dict[str, Any]:
        """Speak text directly (no file)."""
        if not self.is_loaded:
            return {"success": False, "error": "TTS not loaded"}
        
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate audio file from text."""
        if not self.is_loaded:
            return {"success": False, "error": "TTS not loaded"}
        
        try:
            start = time.time()
            
            timestamp = int(time.time())
            filename = f"tts_{timestamp}.wav"
            filepath = OUTPUT_DIR / filename
            
            self.engine.save_to_file(text, str(filepath))
            self.engine.runAndWait()
            
            return {
                "success": True,
                "path": str(filepath),
                "duration": time.time() - start
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class ElevenLabsTTS:
    """ElevenLabs high-quality TTS (CLOUD - requires API key)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.client = None
        self.is_loaded = False
        self.voices = []
        self.current_voice_id = None
    
    def load(self) -> bool:
        if not self.api_key:
            print("ElevenLabs requires ELEVENLABS_API_KEY")
            return False
        
        try:
            from elevenlabs import ElevenLabs
            self.client = ElevenLabs(api_key=self.api_key)
            
            # Fetch available voices
            voices_response = self.client.voices.get_all()
            self.voices = [(v.voice_id, v.name) for v in voices_response.voices]
            
            if self.voices:
                self.current_voice_id = self.voices[0][0]
            
            self.is_loaded = True
            return True
        except ImportError:
            print("Install: pip install elevenlabs")
            return False
        except Exception as e:
            print(f"Failed to load ElevenLabs: {e}")
            return False
    
    def unload(self):
        self.client = None
        self.is_loaded = False
    
    def get_voices(self) -> list:
        return [v[1] for v in self.voices]
    
    def set_voice(self, index: int):
        if 0 <= index < len(self.voices):
            self.current_voice_id = self.voices[index][0]
    
    def generate(self, text: str, **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "ElevenLabs not loaded"}
        
        try:
            start = time.time()
            
            audio = self.client.text_to_speech.convert(
                voice_id=self.current_voice_id or "21m00Tcm4TlvDq8ikWAM",
                text=text,
                model_id="eleven_multilingual_v2"
            )
            
            # Audio is a generator, collect it
            audio_bytes = b"".join(audio)
            
            timestamp = int(time.time())
            filename = f"elevenlabs_{timestamp}.mp3"
            filepath = OUTPUT_DIR / filename
            filepath.write_bytes(audio_bytes)
            
            return {
                "success": True,
                "path": str(filepath),
                "duration": time.time() - start
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class ReplicateAudio:
    """Replicate audio generation (CLOUD - requires API key)."""
    
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "suno-ai/bark:latest"):
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
    
    def generate(self, text: str, **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded or missing API key"}
        
        try:
            import requests
            start = time.time()
            
            output = self.client.run(
                self.model,
                input={"prompt": text}
            )
            
            # Download audio
            audio_url = output if isinstance(output, str) else output.get("audio_out")
            if not audio_url:
                return {"success": False, "error": "No audio URL returned"}
            
            resp = requests.get(audio_url)
            
            timestamp = int(time.time())
            filename = f"replicate_audio_{timestamp}.wav"
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
    'elevenlabs': None,
    'replicate': None,
}


def get_provider(name: str):
    global _providers
    
    if name == 'local' and _providers['local'] is None:
        _providers['local'] = LocalTTS()
    elif name == 'elevenlabs' and _providers['elevenlabs'] is None:
        _providers['elevenlabs'] = ElevenLabsTTS()
    elif name == 'replicate' and _providers['replicate'] is None:
        _providers['replicate'] = ReplicateAudio()
    
    return _providers.get(name)


class AudioGenerationWorker(QThread):
    """Background worker for audio generation."""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    
    def __init__(self, text, provider_name, save_file=True, parent=None):
        super().__init__(parent)
        self.text = text
        self.provider_name = provider_name
        self.save_file = save_file
    
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
            
            self.progress.emit(50)
            
            if self.save_file:
                result = provider.generate(self.text)
            else:
                # Only LocalTTS supports speak()
                if hasattr(provider, 'speak'):
                    result = provider.speak(self.text)
                else:
                    result = provider.generate(self.text)
            
            self.progress.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)})


class AudioTab(QWidget):
    """Tab for audio generation / text-to-speech."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.worker = None
        self.last_audio_path = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Audio Generation / Text-to-Speech")
        header.setFont(QFont('Arial', 14, QFont.Bold))
        header.setStyleSheet("color: #e74c3c;")
        layout.addWidget(header)
        
        # Provider selection
        provider_group = QGroupBox("Provider")
        provider_layout = QHBoxLayout()
        
        self.provider_combo = QComboBox()
        self.provider_combo.addItems([
            'Local (pyttsx3)',
            'ElevenLabs (Cloud)',
            'Replicate (Cloud)'
        ])
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        provider_layout.addWidget(self.provider_combo)
        
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self._load_provider)
        provider_layout.addWidget(self.load_btn)
        
        provider_layout.addStretch()
        provider_group.setLayout(provider_layout)
        layout.addWidget(provider_group)
        
        # Voice selection (for local/elevenlabs)
        voice_group = QGroupBox("Voice Settings")
        voice_layout = QVBoxLayout()
        
        voice_row = QHBoxLayout()
        voice_row.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.currentIndexChanged.connect(self._on_voice_changed)
        voice_row.addWidget(self.voice_combo, 1)
        voice_layout.addLayout(voice_row)
        
        # Rate slider (local only)
        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("Rate:"))
        self.rate_slider = QSlider(Qt.Horizontal)
        self.rate_slider.setRange(50, 300)
        self.rate_slider.setValue(150)
        self.rate_slider.valueChanged.connect(self._on_rate_changed)
        rate_row.addWidget(self.rate_slider)
        self.rate_label = QLabel("150")
        rate_row.addWidget(self.rate_label)
        voice_layout.addLayout(rate_row)
        
        # Volume slider (local only)
        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Volume:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        vol_row.addWidget(self.volume_slider)
        self.volume_label = QLabel("100%")
        vol_row.addWidget(self.volume_label)
        voice_layout.addLayout(vol_row)
        
        voice_group.setLayout(voice_layout)
        layout.addWidget(voice_group)
        
        # Text input
        text_group = QGroupBox("Text to Speak")
        text_layout = QVBoxLayout()
        
        self.text_input = QTextEdit()
        self.text_input.setMaximumHeight(100)
        self.text_input.setPlaceholderText("Enter text to convert to speech...")
        text_layout.addWidget(self.text_input)
        
        text_group.setLayout(text_layout)
        layout.addWidget(text_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.speak_btn = QPushButton("🔊 Speak")
        self.speak_btn.setStyleSheet("background-color: #e74c3c; font-weight: bold; padding: 10px;")
        self.speak_btn.clicked.connect(self._speak_text)
        btn_layout.addWidget(self.speak_btn)
        
        self.save_btn = QPushButton("💾 Save to File")
        self.save_btn.clicked.connect(self._save_to_file)
        btn_layout.addWidget(self.save_btn)
        
        self.play_btn = QPushButton("▶ Play Last")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._play_last)
        btn_layout.addWidget(self.play_btn)
        
        self.open_folder_btn = QPushButton("📁 Output Folder")
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
            "💡 Local TTS works offline. Cloud services require API keys.\n"
            "ElevenLabs: Set ELEVENLABS_API_KEY | Replicate: Set REPLICATE_API_TOKEN"
        )
        info_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(info_label)
        
        layout.addStretch()
    
    def _get_provider_name(self) -> str:
        text = self.provider_combo.currentText()
        if 'Local' in text:
            return 'local'
        elif 'ElevenLabs' in text:
            return 'elevenlabs'
        elif 'Replicate' in text:
            return 'replicate'
        return 'local'
    
    def _on_provider_changed(self, index):
        provider_name = self._get_provider_name()
        
        # Show/hide voice settings based on provider
        is_local = provider_name == 'local'
        self.rate_slider.setEnabled(is_local)
        self.volume_slider.setEnabled(is_local)
        
        # Update voices if provider is loaded
        provider = get_provider(provider_name)
        if provider and provider.is_loaded:
            self._populate_voices(provider)
    
    def _populate_voices(self, provider):
        self.voice_combo.clear()
        if hasattr(provider, 'get_voices'):
            voices = provider.get_voices()
            self.voice_combo.addItems(voices)
    
    def _on_voice_changed(self, index):
        if index >= 0:
            provider_name = self._get_provider_name()
            provider = get_provider(provider_name)
            if provider and provider.is_loaded:
                provider.set_voice(index)
    
    def _on_rate_changed(self, value):
        self.rate_label.setText(str(value))
        provider = get_provider('local')
        if provider and provider.is_loaded:
            provider.set_rate(value)
    
    def _on_volume_changed(self, value):
        self.volume_label.setText(f"{value}%")
        provider = get_provider('local')
        if provider and provider.is_loaded:
            provider.set_volume(value / 100.0)
    
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
                    self._populate_voices(provider)
                else:
                    self.status_label.setText(f"Failed to load {provider_name}")
                self.load_btn.setEnabled(True)
            
            QTimer.singleShot(100, do_load)
    
    def _speak_text(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Text", "Please enter some text to speak")
            return
        
        provider_name = self._get_provider_name()
        
        # Only local supports direct speak
        if provider_name != 'local':
            self._save_to_file()
            return
        
        self.speak_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Speaking...")
        
        self.worker = AudioGenerationWorker(text, provider_name, save_file=False)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_speak_complete)
        self.worker.start()
    
    def _on_speak_complete(self, result: dict):
        self.speak_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        if result.get("success"):
            self.status_label.setText("Done speaking!")
        else:
            error = result.get("error", "Unknown error")
            self.status_label.setText(f"Error: {error}")
    
    def _save_to_file(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Text", "Please enter some text")
            return
        
        provider_name = self._get_provider_name()
        
        self.save_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Generating audio file...")
        
        self.worker = AudioGenerationWorker(text, provider_name, save_file=True)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_save_complete)
        self.worker.start()
    
    def _on_save_complete(self, result: dict):
        self.save_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        if result.get("success"):
            path = result.get("path", "")
            duration = result.get("duration", 0)
            
            self.last_audio_path = path
            self.play_btn.setEnabled(True)
            self.status_label.setText(f"Saved in {duration:.1f}s - {path}")
        else:
            error = result.get("error", "Unknown error")
            self.status_label.setText(f"Error: {error}")
    
    def _play_last(self):
        if self.last_audio_path and Path(self.last_audio_path).exists():
            import subprocess
            import sys
            
            if sys.platform == 'darwin':
                subprocess.run(['afplay', self.last_audio_path])
            elif sys.platform == 'win32':
                os.startfile(self.last_audio_path)
            else:
                # Try common Linux players
                for player in ['aplay', 'paplay', 'mpv', 'ffplay']:
                    try:
                        subprocess.run([player, self.last_audio_path])
                        break
                    except FileNotFoundError:
                        continue
    
    def _open_output_folder(self):
        import subprocess
        import sys
        
        if sys.platform == 'darwin':
            subprocess.run(['open', str(OUTPUT_DIR)])
        elif sys.platform == 'win32':
            subprocess.run(['explorer', str(OUTPUT_DIR)])
        else:
            subprocess.run(['xdg-open', str(OUTPUT_DIR)])


def create_audio_tab(parent) -> QWidget:
    """Factory function for creating the audio tab."""
    if not HAS_PYQT:
        raise ImportError("PyQt5 is required for the Audio Tab")
    return AudioTab(parent)
