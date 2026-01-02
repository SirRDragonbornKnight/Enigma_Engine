"""
Addons Tab - Manage AI Capabilities
====================================

Control image generation, code generation, video, audio, and more.
Connect to local models or cloud APIs.
"""


try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
        QPushButton, QFrame, QComboBox, QTextEdit, QTabWidget,
        QProgressBar, QMessageBox, QFileDialog, QSpinBox,
        QDoubleSpinBox, QLineEdit, QGroupBox, QCheckBox
    )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


# Type colors
TYPE_COLORS = {
    'IMAGE': '#e74c3c',
    'CODE': '#3498db',
    'VIDEO': '#9b59b6',
    'AUDIO': '#2ecc71',
    'EMBEDDING': '#f39c12',
    'SPEECH': '#1abc9c',
    'VISION': '#e67e22',
}

# Provider colors
PROVIDER_COLORS = {
    'LOCAL': '#27ae60',
    'OPENAI': '#10a37f',
    'ANTHROPIC': '#d4a27f',
    'STABILITY': '#7c3aed',
    'REPLICATE': '#000000',
    'ELEVENLABS': '#1a1a2e',
    'HUGGINGFACE': '#ffcc00',
    'MOCK': '#666666',
}


class AddonCard(QFrame):
    """Card displaying a single addon."""
    
    def __init__(self, addon_name: str, addon_info: dict, parent=None):
        super().__init__(parent)
        self.addon_name = addon_name
        self.addon_info = addon_info
        self.setup_ui()
    
    def setup_ui(self):
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(1)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        
        # Header
        header = QHBoxLayout()
        
        name = self.addon_info.get('name', self.addon_name)
        name_label = QLabel(name)
        name_label.setFont(QFont('Arial', 10, QFont.Bold))
        header.addWidget(name_label)
        
        header.addStretch()
        
        # Provider badge
        provider = self.addon_info.get('provider', 'UNKNOWN')
        color = PROVIDER_COLORS.get(provider, '#666')
        provider_label = QLabel(provider)
        provider_label.setStyleSheet(f"background-color: {color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 9px;")
        header.addWidget(provider_label)
        
        layout.addLayout(header)
        
        # Description
        desc = self.addon_info.get('description', 'No description')
        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #888; font-size: 9px;")
        layout.addWidget(desc_label)
        
        # Requirements
        reqs = self.addon_info.get('requirements', [])
        if reqs:
            req_label = QLabel(f"Requires: {', '.join(reqs)}")
            req_label.setStyleSheet("color: #666; font-size: 8px; font-style: italic;")
            layout.addWidget(req_label)
        
        # API key indicator
        if self.addon_info.get('needs_api_key'):
            key_label = QLabel("🔑 Requires API key")
            key_label.setStyleSheet("color: #f39c12; font-size: 8px;")
            layout.addWidget(key_label)
        
        # Actions
        actions = QHBoxLayout()
        
        self.enable_btn = QPushButton("Enable")
        self.enable_btn.setMaximumWidth(80)
        actions.addWidget(self.enable_btn)
        
        self.config_btn = QPushButton("Configure")
        self.config_btn.setMaximumWidth(80)
        actions.addWidget(self.config_btn)
        
        actions.addStretch()
        
        layout.addLayout(actions)


class GenerationPanel(QWidget):
    """Panel for testing generation with an addon."""
    
    def __init__(self, addon_type: str, parent=None):
        super().__init__(parent)
        self.addon_type = addon_type
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Prompt input
        prompt_label = QLabel("Prompt:")
        layout.addWidget(prompt_label)
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setMaximumHeight(100)
        self.prompt_input.setPlaceholderText(f"Enter your {self.addon_type.lower()} prompt...")
        layout.addWidget(self.prompt_input)
        
        # Type-specific options
        if self.addon_type == 'IMAGE':
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
            layout.addLayout(options)
            
        elif self.addon_type == 'CODE':
            options = QHBoxLayout()
            options.addWidget(QLabel("Language:"))
            self.lang_combo = QComboBox()
            self.lang_combo.addItems(['python', 'javascript', 'typescript', 'rust', 'go', 'java', 'c++', 'c#'])
            options.addWidget(self.lang_combo)
            options.addStretch()
            layout.addLayout(options)
            
        elif self.addon_type == 'VIDEO':
            options = QHBoxLayout()
            options.addWidget(QLabel("Duration:"))
            self.duration_spin = QDoubleSpinBox()
            self.duration_spin.setRange(1, 30)
            self.duration_spin.setValue(4)
            self.duration_spin.setSuffix(" sec")
            options.addWidget(self.duration_spin)
            options.addStretch()
            layout.addLayout(options)
            
        elif self.addon_type == 'AUDIO':
            options = QHBoxLayout()
            options.addWidget(QLabel("Duration:"))
            self.duration_spin = QDoubleSpinBox()
            self.duration_spin.setRange(1, 60)
            self.duration_spin.setValue(10)
            self.duration_spin.setSuffix(" sec")
            options.addWidget(self.duration_spin)
            options.addStretch()
            layout.addLayout(options)
        
        # Generate button
        btn_layout = QHBoxLayout()
        self.generate_btn = QPushButton(f"Generate {self.addon_type.title()}")
        self.generate_btn.setStyleSheet("background-color: #3498db; font-weight: bold;")
        btn_layout.addWidget(self.generate_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)
        
        layout.addLayout(btn_layout)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Result area
        self.result_label = QLabel("Result will appear here")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumHeight(200)
        self.result_label.setStyleSheet("background-color: #2d2d2d; border-radius: 4px;")
        layout.addWidget(self.result_label)


class AddonsTab(QWidget):
    """Main tab for managing and using addons."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("AI Addons")
        title.setFont(QFont('Arial', 16, QFont.Bold))
        header.addWidget(title)
        
        header.addStretch()
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_addons)
        header.addWidget(refresh_btn)
        
        # Install custom addon
        install_btn = QPushButton("Install Addon...")
        install_btn.clicked.connect(self.install_addon)
        header.addWidget(install_btn)
        
        layout.addLayout(header)
        
        # Description
        desc = QLabel("Extend Enigma with image generation, code generation, video, audio, and more.")
        desc.setStyleSheet("color: #888;")
        layout.addWidget(desc)
        
        # Main content - tabs by type
        tabs = QTabWidget()
        
        # Image tab
        image_tab = self._create_type_tab('IMAGE', {
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
        })
        tabs.addTab(image_tab, "🖼️ Image")
        
        # Code tab
        code_tab = self._create_type_tab('CODE', {
            'enigma_code': {
                'name': 'Enigma Code',
                'description': 'Use your trained Enigma model for code generation.',
                'requirements': [],
                'provider': 'LOCAL',
            },
            'openai_code': {
                'name': 'GPT-4 Code',
                'description': 'OpenAI GPT-4 for complex code tasks.',
                'requirements': ['openai'],
                'provider': 'OPENAI',
                'needs_api_key': True,
            },
        })
        tabs.addTab(code_tab, "💻 Code")
        
        # Video tab
        video_tab = self._create_type_tab('VIDEO', {
            'replicate_video': {
                'name': 'Replicate Video',
                'description': 'Cloud video generation (Zeroscope, AnimateDiff).',
                'requirements': ['replicate'],
                'provider': 'REPLICATE',
                'needs_api_key': True,
            },
            'local_video': {
                'name': 'AnimateDiff (Local)',
                'description': 'Generate videos locally. Requires good GPU.',
                'requirements': ['torch', 'diffusers'],
                'provider': 'LOCAL',
            },
        })
        tabs.addTab(video_tab, "🎬 Video")
        
        # Audio tab
        audio_tab = self._create_type_tab('AUDIO', {
            'local_tts': {
                'name': 'Local TTS',
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
            'replicate_audio': {
                'name': 'MusicGen',
                'description': 'Generate music from text prompts.',
                'requirements': ['replicate'],
                'provider': 'REPLICATE',
                'needs_api_key': True,
            },
        })
        tabs.addTab(audio_tab, "🎵 Audio")
        
        # Embedding tab
        embed_tab = self._create_type_tab('EMBEDDING', {
            'local_embedding': {
                'name': 'Sentence Transformers',
                'description': 'Local embeddings for semantic search.',
                'requirements': ['sentence-transformers'],
                'provider': 'LOCAL',
            },
            'openai_embedding': {
                'name': 'OpenAI Embeddings',
                'description': 'High quality embeddings via API.',
                'requirements': ['openai'],
                'provider': 'OPENAI',
                'needs_api_key': True,
            },
        })
        tabs.addTab(embed_tab, "🔍 Search")
        
        # HuggingFace Settings tab
        hf_tab = self._create_huggingface_tab()
        tabs.addTab(hf_tab, "🤗 HuggingFace")
        
        layout.addWidget(tabs)
    
    def _create_huggingface_tab(self) -> QWidget:
        """Create the HuggingFace configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Header
        header_layout = QHBoxLayout()
        hf_icon = QLabel("🤗")
        hf_icon.setFont(QFont('Arial', 24))
        header_layout.addWidget(hf_icon)
        
        title = QLabel("HuggingFace Integration")
        title.setFont(QFont('Arial', 14, QFont.Bold))
        title.setStyleSheet("color: #ffcc00;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        desc = QLabel("Connect to HuggingFace for models, embeddings, and inference API.")
        desc.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # API Key Configuration
        api_group = QGroupBox("🔑 API Configuration")
        api_layout = QVBoxLayout(api_group)
        
        key_row = QHBoxLayout()
        key_label = QLabel("HuggingFace Token:")
        key_row.addWidget(key_label)
        
        self.hf_token_input = QLineEdit()
        self.hf_token_input.setPlaceholderText("hf_xxxxxxxxxxxx")
        self.hf_token_input.setEchoMode(QLineEdit.Password)
        key_row.addWidget(self.hf_token_input, stretch=1)
        
        show_btn = QPushButton("👁")
        show_btn.setMaximumWidth(30)
        show_btn.setCheckable(True)
        show_btn.toggled.connect(lambda checked: self.hf_token_input.setEchoMode(
            QLineEdit.Normal if checked else QLineEdit.Password))
        key_row.addWidget(show_btn)
        
        api_layout.addLayout(key_row)
        
        # Save/Test buttons
        btn_row = QHBoxLayout()
        save_key_btn = QPushButton("💾 Save Token")
        save_key_btn.clicked.connect(self._save_hf_token)
        btn_row.addWidget(save_key_btn)
        
        test_btn = QPushButton("🧪 Test Connection")
        test_btn.clicked.connect(self._test_hf_connection)
        btn_row.addWidget(test_btn)
        
        btn_row.addStretch()
        api_layout.addLayout(btn_row)
        
        # How to get token
        help_label = QLabel(
            "<p>Get your token from: "
            "<a href='https://huggingface.co/settings/tokens' style='color: #ffcc00;'>"
            "huggingface.co/settings/tokens</a></p>"
            "<p><i>Token is stored in: ~/.enigma/hf_token</i></p>"
        )
        help_label.setOpenExternalLinks(True)
        help_label.setStyleSheet("color: #666; font-size: 10px;")
        api_layout.addWidget(help_label)
        
        layout.addWidget(api_group)
        
        # Model Selection
        model_group = QGroupBox("📦 Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        # Text generation
        text_row = QHBoxLayout()
        text_row.addWidget(QLabel("Text Generation:"))
        self.hf_text_model = QComboBox()
        self.hf_text_model.addItems([
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "microsoft/DialoGPT-medium",
            "bigscience/bloom-560m",
        ])
        self.hf_text_model.setEditable(True)
        text_row.addWidget(self.hf_text_model, stretch=1)
        model_layout.addLayout(text_row)
        
        # Image generation
        img_row = QHBoxLayout()
        img_row.addWidget(QLabel("Image Generation:"))
        self.hf_image_model = QComboBox()
        self.hf_image_model.addItems([
            "stabilityai/stable-diffusion-2-1",
            "runwayml/stable-diffusion-v1-5",
            "CompVis/stable-diffusion-v1-4",
        ])
        self.hf_image_model.setEditable(True)
        img_row.addWidget(self.hf_image_model, stretch=1)
        model_layout.addLayout(img_row)
        
        # Embeddings
        embed_row = QHBoxLayout()
        embed_row.addWidget(QLabel("Embeddings:"))
        self.hf_embed_model = QComboBox()
        self.hf_embed_model.addItems([
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ])
        self.hf_embed_model.setEditable(True)
        embed_row.addWidget(self.hf_embed_model, stretch=1)
        model_layout.addLayout(embed_row)
        
        # Use local checkbox
        self.hf_use_local = QCheckBox("Download and run models locally (requires disk space)")
        model_layout.addWidget(self.hf_use_local)
        
        layout.addWidget(model_group)
        
        # Status
        status_group = QGroupBox("📊 Status")
        status_layout = QVBoxLayout(status_group)
        
        self.hf_status_label = QLabel("Not connected")
        self.hf_status_label.setStyleSheet("color: #f39c12;")
        status_layout.addWidget(self.hf_status_label)
        
        layout.addWidget(status_group)
        
        layout.addStretch()
        
        # Load saved token
        self._load_hf_token()
        
        return widget
    
    def _load_hf_token(self):
        """Load saved HuggingFace token."""
        import os
        from pathlib import Path
        
        # Try environment variable first
        token = os.environ.get('HUGGINGFACE_TOKEN', '')
        
        # Try saved file
        if not token:
            token_file = Path.home() / ".enigma" / "hf_token"
            if token_file.exists():
                try:
                    token = token_file.read_text().strip()
                except Exception:
                    pass
        
        if token:
            self.hf_token_input.setText(token)
            self.hf_status_label.setText("✓ Token loaded")
            self.hf_status_label.setStyleSheet("color: #2ecc71;")
    
    def _save_hf_token(self):
        """Save HuggingFace token."""
        from pathlib import Path
        
        token = self.hf_token_input.text().strip()
        if not token:
            QMessageBox.warning(self, "No Token", "Please enter a HuggingFace token")
            return
        
        # Save to file
        try:
            enigma_dir = Path.home() / ".enigma"
            enigma_dir.mkdir(exist_ok=True)
            token_file = enigma_dir / "hf_token"
            token_file.write_text(token)
            
            self.hf_status_label.setText("✓ Token saved")
            self.hf_status_label.setStyleSheet("color: #2ecc71;")
            QMessageBox.information(self, "Saved", "Token saved to ~/.enigma/hf_token")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save token: {e}")
    
    def _test_hf_connection(self):
        """Test HuggingFace API connection."""
        token = self.hf_token_input.text().strip()
        if not token:
            QMessageBox.warning(self, "No Token", "Please enter a HuggingFace token first")
            return
        
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            user = api.whoami()
            username = user.get('name', 'Unknown')
            
            self.hf_status_label.setText(f"✓ Connected as: {username}")
            self.hf_status_label.setStyleSheet("color: #2ecc71;")
            QMessageBox.information(self, "Success", f"Connected to HuggingFace as: {username}")
        except ImportError:
            QMessageBox.warning(
                self, "Missing Package",
                "huggingface-hub is not installed.\n\n"
                "Install with: pip install huggingface-hub"
            )
        except Exception as e:
            self.hf_status_label.setText(f"✗ Connection failed")
            self.hf_status_label.setStyleSheet("color: #e74c3c;")
            QMessageBox.warning(self, "Connection Failed", f"Could not connect: {e}")
    
    def _create_type_tab(self, addon_type: str, addons: dict) -> QWidget:
        """Create a tab for a specific addon type."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Left: Addon list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        type_label = QLabel(f"{addon_type} Addons")
        type_label.setFont(QFont('Arial', 12, QFont.Bold))
        color = TYPE_COLORS.get(addon_type, '#666')
        type_label.setStyleSheet(f"color: {color};")
        left_layout.addWidget(type_label)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        cards_widget = QWidget()
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setSpacing(10)
        
        for name, info in addons.items():
            card = AddonCard(name, info)
            cards_layout.addWidget(card)
        
        cards_layout.addStretch()
        scroll.setWidget(cards_widget)
        left_layout.addWidget(scroll)
        
        layout.addWidget(left, stretch=1)
        
        # Right: Generation panel
        right = GenerationPanel(addon_type)
        layout.addWidget(right, stretch=1)
        
        return widget
    
    def refresh_addons(self):
        """Refresh addon list."""
        QMessageBox.information(self, "Refresh", "Addon list refreshed!")
    
    def install_addon(self):
        """Install a custom addon from file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Install Addon",
            "",
            "Python Files (*.py)"
        )
        if path:
            QMessageBox.information(
                self,
                "Install Addon",
                f"Would install addon from:\n{path}\n\n(Not yet implemented)"
            )


def create_addons_tab(window) -> QWidget:
    """Factory function for creating the addons tab."""
    return AddonsTab(window)


if not HAS_PYQT:
    class AddonsTab:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt5 is required for the Addons Tab")
