"""
Voice Cloning Tab - Create custom voices from audio samples.

Features:
- Upload voice clips to clone a voice
- AI-generated personality voices (AI decides the voice)
- Voice profile management
- Real-time preview
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QTextEdit, QProgressBar,
        QMessageBox, QGroupBox, QSlider, QFileDialog, QLineEdit, 
        QCheckBox, QListWidget, QListWidgetItem, QTabWidget,
        QFrame, QSpinBox, QDoubleSpinBox, QGridLayout, QScrollArea
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QIcon
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from .shared_components import NoScrollComboBox

from ...config import CONFIG
from ...voice.voice_profile import VoiceProfile, VoiceEngine, PROFILES_DIR, get_engine
from ...voice.voice_generator import AIVoiceGenerator, create_voice_from_samples
from ...voice.audio_analyzer import AudioAnalyzer

# Voice samples directory
SAMPLES_DIR = Path(CONFIG.get("data_dir", "data")) / "voice_profiles" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


class VoiceGenerationWorker(QThread):
    """Background worker for voice generation."""
    
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, task: str, **kwargs):
        super().__init__()
        self.task = task
        self.kwargs = kwargs
    
    def run(self):
        try:
            if self.task == "clone_from_samples":
                self._clone_from_samples()
            elif self.task == "generate_from_personality":
                self._generate_from_personality()
            elif self.task == "analyze_sample":
                self._analyze_sample()
        except Exception as e:
            self.error.emit(str(e))
    
    def _clone_from_samples(self):
        audio_files = self.kwargs.get("audio_files", [])
        name = self.kwargs.get("name", "cloned_voice")
        
        if not audio_files:
            self.error.emit("No audio files provided")
            return
        
        self.progress.emit("Analyzing audio samples...")
        
        # Create voice from samples
        profile = create_voice_from_samples(audio_files, name)
        
        self.progress.emit("Voice profile created!")
        self.finished.emit({
            "success": True,
            "profile": profile,
            "name": name
        })
    
    def _generate_from_personality(self):
        personality_desc = self.kwargs.get("personality_desc", "")
        name = self.kwargs.get("name", "ai_voice")
        base_voice = self.kwargs.get("base_voice", "default")
        
        self.progress.emit("AI is analyzing personality...")
        
        # Create a mock personality from description using AI-driven mapping
        profile = self._ai_generate_voice(personality_desc, name, base_voice)
        
        self.progress.emit("AI voice generated!")
        self.finished.emit({
            "success": True,
            "profile": profile,
            "name": name
        })
    
    def _ai_generate_voice(self, description: str, name: str, base_voice: str) -> VoiceProfile:
        """AI-driven voice generation from personality description."""
        desc_lower = description.lower()
        
        # AI analyzes personality traits from description
        traits = {
            "confident": any(w in desc_lower for w in ["confident", "bold", "assertive", "leader", "powerful"]),
            "playful": any(w in desc_lower for w in ["playful", "fun", "cheerful", "happy", "energetic", "bubbly"]),
            "calm": any(w in desc_lower for w in ["calm", "peaceful", "serene", "gentle", "soft", "soothing"]),
            "mysterious": any(w in desc_lower for w in ["mysterious", "dark", "enigmatic", "cryptic", "shadowy"]),
            "wise": any(w in desc_lower for w in ["wise", "old", "ancient", "mentor", "sage", "knowledgeable"]),
            "robotic": any(w in desc_lower for w in ["robot", "ai", "machine", "synthetic", "digital", "electronic"]),
            "sarcastic": any(w in desc_lower for w in ["sarcastic", "witty", "sardonic", "dry", "ironic"]),
            "nervous": any(w in desc_lower for w in ["nervous", "anxious", "worried", "timid", "shy"]),
            "deep": any(w in desc_lower for w in ["deep", "bass", "low", "baritone"]),
            "high": any(w in desc_lower for w in ["high", "squeaky", "shrill", "soprano"]),
            "fast": any(w in desc_lower for w in ["fast", "quick", "rapid", "speedy"]),
            "slow": any(w in desc_lower for w in ["slow", "deliberate", "measured", "thoughtful"]),
        }
        
        # Calculate voice parameters based on traits
        pitch = 1.0
        speed = 1.0
        volume = 0.85
        effects = []
        
        # Pitch adjustments
        if traits["deep"]:
            pitch -= 0.25
        if traits["high"]:
            pitch += 0.25
        if traits["confident"]:
            pitch -= 0.1
        if traits["nervous"]:
            pitch += 0.15
        if traits["wise"]:
            pitch -= 0.15
        if traits["playful"]:
            pitch += 0.1
        
        # Speed adjustments
        if traits["fast"]:
            speed += 0.3
        if traits["slow"]:
            speed -= 0.2
        if traits["nervous"]:
            speed += 0.15
        if traits["calm"]:
            speed -= 0.1
        if traits["wise"]:
            speed -= 0.15
        
        # Volume adjustments
        if traits["confident"]:
            volume = 0.95
        if traits["calm"]:
            volume = 0.75
        if traits["mysterious"]:
            volume = 0.7
        
        # Effects
        if traits["robotic"]:
            effects.append("robotic")
        if traits["mysterious"]:
            effects.append("echo")
        
        # Clamp values
        pitch = max(0.5, min(1.5, pitch))
        speed = max(0.6, min(1.5, speed))
        volume = max(0.5, min(1.0, volume))
        
        # Create profile
        profile = VoiceProfile(
            name=name,
            pitch=pitch,
            speed=speed,
            volume=volume,
            voice=base_voice,
            effects=effects,
            description=f"AI-generated voice: {description[:100]}"
        )
        profile.save()
        
        return profile
    
    def _analyze_sample(self):
        audio_path = self.kwargs.get("audio_path", "")
        
        if not audio_path or not Path(audio_path).exists():
            self.error.emit("Audio file not found")
            return
        
        self.progress.emit("Analyzing audio...")
        
        analyzer = AudioAnalyzer()
        is_valid, issues = analyzer.validate_audio_quality(audio_path)
        features = analyzer.analyze_audio(audio_path)
        
        self.finished.emit({
            "success": True,
            "valid": is_valid,
            "issues": issues,
            "features": {
                "duration": features.duration,
                "pitch": features.average_pitch,
                "energy": features.energy,
                "sample_rate": features.sample_rate
            }
        })


class VoiceCloneTab(QWidget):
    """Voice Cloning Tab for creating custom avatar voices."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.audio_samples: List[str] = []
        self.current_profile: Optional[VoiceProfile] = None
        self.worker: Optional[VoiceGenerationWorker] = None
        
        self._init_ui()
        self._load_existing_profiles()
    
    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Voice Cloning & AI Voice Generation")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet("color: #89b4fa; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Tab widget for different modes
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #45475a; border-radius: 8px; }
            QTabBar::tab { padding: 8px 16px; margin: 2px; }
            QTabBar::tab:selected { background: #89b4fa; color: #1e1e2e; border-radius: 4px; }
            QTabBar::tab:!selected { background: #313244; color: #cdd6f4; border-radius: 4px; }
        """)
        
        # Tab 1: Clone from Audio
        clone_tab = self._create_clone_tab()
        tabs.addTab(clone_tab, "Clone from Audio")
        
        # Tab 2: AI Generate Voice
        ai_tab = self._create_ai_generation_tab()
        tabs.addTab(ai_tab, "AI Generate Voice")
        
        # Tab 3: Voice Profiles
        profiles_tab = self._create_profiles_tab()
        tabs.addTab(profiles_tab, "Voice Profiles")
        
        layout.addWidget(tabs)
        
        # Progress and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #a6adc8; font-style: italic;")
        layout.addWidget(self.status_label)
    
    def _create_clone_tab(self) -> QWidget:
        """Create the voice cloning from audio tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Instructions
        info = QLabel(
            "Upload 3-10 audio clips of a voice to clone. Each clip should be:\n"
            "• 3-30 seconds long\n"
            "• Clear speech (no background music/noise)\n"
            "• Single speaker only"
        )
        info.setStyleSheet("color: #bac2de; background: #313244; padding: 10px; border-radius: 8px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Voice name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Voice Name:"))
        self.clone_name_input = QLineEdit()
        self.clone_name_input.setPlaceholderText("e.g., MyCharacter, GladosVoice")
        name_layout.addWidget(self.clone_name_input)
        layout.addLayout(name_layout)
        
        # Audio samples list
        samples_group = QGroupBox("Audio Samples")
        samples_layout = QVBoxLayout(samples_group)
        
        self.samples_list = QListWidget()
        self.samples_list.setMaximumHeight(150)
        samples_layout.addWidget(self.samples_list)
        
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("+ Add Audio Files")
        add_btn.setToolTip("Add audio samples of the voice to clone")
        add_btn.clicked.connect(self._on_add_samples)
        btn_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.setToolTip("Remove selected audio sample from the list")
        remove_btn.clicked.connect(self._on_remove_sample)
        btn_layout.addWidget(remove_btn)
        
        analyze_btn = QPushButton("Analyze")
        analyze_btn.setToolTip("Analyze the selected audio sample")
        analyze_btn.clicked.connect(self._on_analyze_sample)
        btn_layout.addWidget(analyze_btn)
        
        samples_layout.addLayout(btn_layout)
        layout.addWidget(samples_group)
        
        # Analysis results
        self.analysis_label = QLabel("")
        self.analysis_label.setStyleSheet("color: #a6adc8; padding: 5px;")
        self.analysis_label.setWordWrap(True)
        layout.addWidget(self.analysis_label)
        
        # Clone button
        clone_btn = QPushButton("Create Cloned Voice")
        clone_btn.setToolTip("Create a new voice profile based on the audio samples")
        clone_btn.setStyleSheet("""
            QPushButton { 
                background: #a6e3a1; color: #1e1e2e; 
                padding: 12px; font-weight: bold; 
                border-radius: 8px; font-size: 12px;
            }
            QPushButton:hover { background: #94e2d5; }
        """)
        clone_btn.clicked.connect(self._on_clone_voice)
        layout.addWidget(clone_btn)
        
        layout.addStretch()
        return widget
    
    def _create_ai_generation_tab(self) -> QWidget:
        """Create the AI voice generation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Instructions
        info = QLabel(
            "Let the AI create a voice based on a personality description.\n"
            "The AI will analyze the traits and generate matching voice parameters.\n"
            "Just describe the character - the AI does the rest!"
        )
        info.setStyleSheet("color: #bac2de; background: #313244; padding: 10px; border-radius: 8px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Voice name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Voice Name:"))
        self.ai_name_input = QLineEdit()
        self.ai_name_input.setPlaceholderText("e.g., WiseOldMentor, EnergeticAssistant")
        name_layout.addWidget(self.ai_name_input)
        layout.addLayout(name_layout)
        
        # Personality description
        desc_group = QGroupBox("Character/Personality Description")
        desc_layout = QVBoxLayout(desc_group)
        
        self.personality_input = QTextEdit()
        self.personality_input.setPlaceholderText(
            "Describe the character's personality, voice qualities, and traits.\n\n"
            "Examples:\n"
            "• \"A confident, deep-voiced leader who speaks deliberately and calmly\"\n"
            "• \"A nervous, high-pitched AI assistant who talks fast and stutters\"\n"
            "• \"A wise, ancient sage who speaks slowly with gravitas\"\n"
            "• \"A playful, energetic character who sounds cheerful and bubbly\"\n"
            "• \"A mysterious, robotic AI with an echoing, synthetic voice\""
        )
        self.personality_input.setMaximumHeight(150)
        desc_layout.addWidget(self.personality_input)
        layout.addWidget(desc_group)
        
        # Base voice selection
        base_layout = QHBoxLayout()
        base_layout.addWidget(QLabel("Base Voice Type:"))
        self.base_voice_combo = NoScrollComboBox()
        self.base_voice_combo.addItems(["default", "male", "female"])
        self.base_voice_combo.setToolTip("Select base voice type for AI voice generation")
        base_layout.addWidget(self.base_voice_combo)
        base_layout.addStretch()
        layout.addLayout(base_layout)
        
        # Generate button
        gen_btn = QPushButton("Generate AI Voice")
        gen_btn.setStyleSheet("""
            QPushButton { 
                background: #89b4fa; color: #1e1e2e; 
                padding: 12px; font-weight: bold; 
                border-radius: 8px; font-size: 12px;
            }
            QPushButton:hover { background: #74c7ec; }
        """)
        gen_btn.clicked.connect(self._on_generate_ai_voice)
        layout.addWidget(gen_btn)
        
        # Quick presets
        presets_group = QGroupBox("Quick Personality Presets")
        presets_layout = QGridLayout(presets_group)
        
        presets = [
            ("Cold AI", "A cold, calculating AI with a robotic, monotone voice. Speaks deliberately."),
            ("Friendly", "A warm, friendly assistant who speaks cheerfully with an upbeat tone."),
            ("Wise Sage", "An ancient wise mentor who speaks slowly and thoughtfully with gravitas."),
            ("Energetic", "A hyperactive, fast-talking character full of energy and excitement."),
            ("Mysterious", "A dark, mysterious entity with a deep, echoing voice."),
            ("Nervous", "A timid, nervous character who speaks quickly and with uncertainty."),
        ]
        
        for i, (label, desc) in enumerate(presets):
            btn = QPushButton(label)
            btn.setStyleSheet("padding: 8px;")
            btn.clicked.connect(lambda checked, d=desc: self.personality_input.setPlainText(d))
            presets_layout.addWidget(btn, i // 3, i % 3)
        
        layout.addWidget(presets_group)
        
        layout.addStretch()
        return widget
    
    def _create_profiles_tab(self) -> QWidget:
        """Create the voice profiles management tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Profile list
        list_group = QGroupBox("Saved Voice Profiles")
        list_layout = QVBoxLayout(list_group)
        
        self.profiles_list = QListWidget()
        self.profiles_list.itemClicked.connect(self._on_profile_selected)
        list_layout.addWidget(self.profiles_list)
        
        btn_layout = QHBoxLayout()
        
        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self._on_preview_voice)
        btn_layout.addWidget(preview_btn)
        
        apply_btn = QPushButton("Use for Avatar")
        apply_btn.clicked.connect(self._on_apply_to_avatar)
        btn_layout.addWidget(apply_btn)
        
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._on_delete_profile)
        btn_layout.addWidget(delete_btn)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_existing_profiles)
        btn_layout.addWidget(refresh_btn)
        
        list_layout.addLayout(btn_layout)
        layout.addWidget(list_group)
        
        # Profile details
        details_group = QGroupBox("Profile Details")
        details_layout = QVBoxLayout(details_group)
        
        self.profile_details = QLabel("Select a profile to view details")
        self.profile_details.setStyleSheet("color: #a6adc8; padding: 10px;")
        self.profile_details.setWordWrap(True)
        details_layout.addWidget(self.profile_details)
        
        # Manual adjustment sliders
        adjust_layout = QGridLayout()
        
        adjust_layout.addWidget(QLabel("Pitch:"), 0, 0)
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setRange(50, 150)
        self.pitch_slider.setValue(100)
        self.pitch_slider.valueChanged.connect(self._on_slider_changed)
        adjust_layout.addWidget(self.pitch_slider, 0, 1)
        self.pitch_label = QLabel("1.0")
        adjust_layout.addWidget(self.pitch_label, 0, 2)
        
        adjust_layout.addWidget(QLabel("Speed:"), 1, 0)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 150)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self._on_slider_changed)
        adjust_layout.addWidget(self.speed_slider, 1, 1)
        self.speed_label = QLabel("1.0")
        adjust_layout.addWidget(self.speed_label, 1, 2)
        
        adjust_layout.addWidget(QLabel("Volume:"), 2, 0)
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(30, 100)
        self.volume_slider.setValue(85)
        self.volume_slider.valueChanged.connect(self._on_slider_changed)
        adjust_layout.addWidget(self.volume_slider, 2, 1)
        self.volume_label = QLabel("0.85")
        adjust_layout.addWidget(self.volume_label, 2, 2)
        
        details_layout.addLayout(adjust_layout)
        
        save_changes_btn = QPushButton("Save Changes")
        save_changes_btn.clicked.connect(self._on_save_profile_changes)
        details_layout.addWidget(save_changes_btn)
        
        layout.addWidget(details_group)
        
        # Preview text
        preview_group = QGroupBox("Preview Text")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_text_input = QLineEdit()
        self.preview_text_input.setText("Hello! This is a voice preview test.")
        preview_layout.addWidget(self.preview_text_input)
        
        layout.addWidget(preview_group)
        
        layout.addStretch()
        return widget
    
    def _on_add_samples(self):
        """Add audio sample files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Files",
            "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac *.m4a);;All Files (*)"
        )
        
        if files:
            for f in files:
                if f not in self.audio_samples:
                    self.audio_samples.append(f)
                    item = QListWidgetItem(Path(f).name)
                    item.setData(Qt.UserRole, f)
                    self.samples_list.addItem(item)
    
    def _on_remove_sample(self):
        """Remove selected sample."""
        current = self.samples_list.currentItem()
        if current:
            path = current.data(Qt.UserRole)
            if path in self.audio_samples:
                self.audio_samples.remove(path)
            self.samples_list.takeItem(self.samples_list.row(current))
    
    def _on_analyze_sample(self):
        """Analyze selected audio sample."""
        current = self.samples_list.currentItem()
        if not current:
            QMessageBox.warning(self, "No Selection", "Select an audio file to analyze")
            return
        
        audio_path = current.data(Qt.UserRole)
        
        self.worker = VoiceGenerationWorker("analyze_sample", audio_path=audio_path)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_analysis_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
    
    def _on_analysis_done(self, result):
        """Handle analysis completion."""
        self.progress_bar.setVisible(False)
        
        if result.get("success"):
            features = result.get("features", {})
            issues = result.get("issues", [])
            
            analysis = f"Analysis:\n"
            analysis += f"  Duration: {features.get('duration', 0):.1f}s\n"
            analysis += f"  Pitch: {features.get('pitch', 1.0):.2f}\n"
            analysis += f"  Energy: {features.get('energy', 0.5):.2f}\n"
            analysis += f"  Sample Rate: {features.get('sample_rate', 0)} Hz\n"
            
            if issues:
                analysis += f"\nIssues:\n"
                for issue in issues:
                    analysis += f"  - {issue}\n"
            else:
                analysis += "\nAudio quality looks good!"
            
            self.analysis_label.setText(analysis)
    
    def _on_clone_voice(self):
        """Start voice cloning process."""
        name = self.clone_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please enter a name for the voice")
            return
        
        if len(self.audio_samples) < 1:
            QMessageBox.warning(self, "No Samples", "Please add at least one audio sample")
            return
        
        self.worker = VoiceGenerationWorker(
            "clone_from_samples",
            audio_files=self.audio_samples,
            name=name
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_clone_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Creating cloned voice...")
    
    def _on_clone_done(self, result):
        """Handle cloning completion."""
        self.progress_bar.setVisible(False)
        
        if result.get("success"):
            profile = result.get("profile")
            self.current_profile = profile
            self.status_label.setText(f"Voice '{result.get('name')}' created successfully!")
            self._load_existing_profiles()
            QMessageBox.information(self, "Success", f"Voice profile '{result.get('name')}' created!")
    
    def _on_generate_ai_voice(self):
        """Generate voice from AI personality analysis."""
        name = self.ai_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please enter a name for the voice")
            return
        
        personality = self.personality_input.toPlainText().strip()
        if not personality:
            QMessageBox.warning(self, "Missing Description", "Please describe the personality/character")
            return
        
        base_voice = self.base_voice_combo.currentText()
        
        self.worker = VoiceGenerationWorker(
            "generate_from_personality",
            personality_desc=personality,
            name=name,
            base_voice=base_voice
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_ai_gen_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("AI is generating voice...")
    
    def _on_ai_gen_done(self, result):
        """Handle AI generation completion."""
        self.progress_bar.setVisible(False)
        
        if result.get("success"):
            profile = result.get("profile")
            self.current_profile = profile
            self.status_label.setText(f"AI voice '{result.get('name')}' generated!")
            self._load_existing_profiles()
            QMessageBox.information(
                self, "Success", 
                f"AI-generated voice '{result.get('name')}' created!\n\n"
                f"Parameters:\n"
                f"  Pitch: {profile.pitch:.2f}\n"
                f"  Speed: {profile.speed:.2f}\n"
                f"  Volume: {profile.volume:.2f}\n"
                f"  Effects: {', '.join(profile.effects) if profile.effects else 'None'}"
            )
    
    def _load_existing_profiles(self):
        """Load existing voice profiles."""
        self.profiles_list.clear()
        
        # Load from profiles directory
        if PROFILES_DIR.exists():
            for file in PROFILES_DIR.glob("*.json"):
                try:
                    profile = VoiceProfile.load(file.stem)
                    item = QListWidgetItem(f"{profile.name}")
                    item.setData(Qt.UserRole, file.stem)
                    self.profiles_list.addItem(item)
                except Exception:
                    pass
    
    def _on_profile_selected(self, item):
        """Handle profile selection."""
        profile_name = item.data(Qt.UserRole)
        try:
            profile = VoiceProfile.load(profile_name)
            self.current_profile = profile
            
            # Update details
            details = f"Name: {profile.name}\n"
            details += f"Voice Type: {profile.voice}\n"
            details += f"Pitch: {profile.pitch:.2f}\n"
            details += f"Speed: {profile.speed:.2f}\n"
            details += f"Volume: {profile.volume:.2f}\n"
            details += f"Effects: {', '.join(profile.effects) if profile.effects else 'None'}\n"
            details += f"Description: {profile.description or 'No description'}"
            
            self.profile_details.setText(details)
            
            # Update sliders
            self.pitch_slider.setValue(int(profile.pitch * 100))
            self.speed_slider.setValue(int(profile.speed * 100))
            self.volume_slider.setValue(int(profile.volume * 100))
            
        except Exception as e:
            self.profile_details.setText(f"Error loading profile: {e}")
    
    def _on_slider_changed(self):
        """Handle slider value changes."""
        pitch = self.pitch_slider.value() / 100
        speed = self.speed_slider.value() / 100
        volume = self.volume_slider.value() / 100
        
        self.pitch_label.setText(f"{pitch:.2f}")
        self.speed_label.setText(f"{speed:.2f}")
        self.volume_label.setText(f"{volume:.2f}")
    
    def _on_save_profile_changes(self):
        """Save changes to current profile."""
        if not self.current_profile:
            QMessageBox.warning(self, "No Profile", "Select a profile first")
            return
        
        self.current_profile.pitch = self.pitch_slider.value() / 100
        self.current_profile.speed = self.speed_slider.value() / 100
        self.current_profile.volume = self.volume_slider.value() / 100
        
        self.current_profile.save()
        self.status_label.setText(f"Profile '{self.current_profile.name}' saved!")
    
    def _on_preview_voice(self):
        """Preview the selected voice."""
        if not self.current_profile:
            QMessageBox.warning(self, "No Profile", "Select a profile first")
            return
        
        text = self.preview_text_input.text().strip()
        if not text:
            text = "Hello! This is a voice preview test."
        
        try:
            engine = get_engine()
            engine.set_profile(self.current_profile)
            engine.speak(text)
        except Exception as e:
            QMessageBox.warning(self, "TTS Error", f"Could not preview voice: {e}")
    
    def _on_apply_to_avatar(self):
        """Apply voice to current avatar."""
        if not self.current_profile:
            QMessageBox.warning(self, "No Profile", "Select a profile first")
            return
        
        # Try to set on main window's avatar
        if self.main_window and hasattr(self.main_window, 'avatar'):
            avatar = self.main_window.avatar
            if avatar:
                avatar.voice_profile = self.current_profile
                self.status_label.setText(f"Applied '{self.current_profile.name}' to avatar!")
                QMessageBox.information(self, "Applied", f"Voice '{self.current_profile.name}' applied to avatar!")
                return
        
        # Store in settings for next avatar load
        if self.main_window and hasattr(self.main_window, 'settings'):
            self.main_window.settings['avatar_voice'] = self.current_profile.name
        
        self.status_label.setText(f"Voice '{self.current_profile.name}' will be used for avatar.")
    
    def _on_delete_profile(self):
        """Delete selected profile."""
        current = self.profiles_list.currentItem()
        if not current:
            return
        
        profile_name = current.data(Qt.UserRole)
        
        confirm = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete voice profile '{profile_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                profile_path = PROFILES_DIR / f"{profile_name}.json"
                if profile_path.exists():
                    profile_path.unlink()
                self._load_existing_profiles()
                self.status_label.setText(f"Deleted '{profile_name}'")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not delete: {e}")
    
    def _on_progress(self, msg):
        """Handle progress updates."""
        self.status_label.setText(msg)
    
    def _on_error(self, error):
        """Handle errors."""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error}")
        QMessageBox.warning(self, "Error", error)


# For standalone testing
if __name__ == "__main__" and HAS_PYQT:
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    tab = VoiceCloneTab()
    tab.resize(800, 600)
    tab.show()
    sys.exit(app.exec_())
