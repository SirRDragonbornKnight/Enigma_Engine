"""
Module Manager Tab - Control all Enigma capabilities
====================================================

Unified control panel for all modules and AI capabilities.
Everything toggleable, organized by category.
"""
from typing import Dict, List

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea,
        QLabel, QPushButton, QFrame, QGroupBox, QCheckBox,
        QLineEdit, QProgressBar, QMessageBox, QSplitter,
        QTextEdit
    )
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


# Helper function to make labels selectable
def make_selectable_label(text: str, **kwargs) -> QLabel:
    """Create a QLabel with selectable text."""
    label = QLabel(text)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    for key, value in kwargs.items():
        if hasattr(label, f'set{key.capitalize()}'):
            getattr(label, f'set{key.capitalize()}')(value)
    return label


# Category colors and icons
CATEGORY_STYLES = {
    'core': {'color': '#e74c3c', 'icon': '⚙️', 'name': 'Core'},
    'memory': {'color': '#3498db', 'icon': '🧠', 'name': 'Memory'},
    'interface': {'color': '#2ecc71', 'icon': '🖥️', 'name': 'Interface'},
    'perception': {'color': '#9b59b6', 'icon': '👁️', 'name': 'Perception'},
    'output': {'color': '#f39c12', 'icon': '📤', 'name': 'Output'},
    'generation': {'color': '#e91e63', 'icon': '✨', 'name': 'AI Generation'},
    'tools': {'color': '#1abc9c', 'icon': '🔧', 'name': 'Tools'},
    'network': {'color': '#e67e22', 'icon': '🌐', 'name': 'Network'},
    'extension': {'color': '#95a5a6', 'icon': '🔌', 'name': 'Extension'},
}


class ModuleCard(QFrame):
    """Visual card for a single module."""
    
    def __init__(self, module_id: str, module_info: dict, parent=None):
        super().__init__(parent)
        self.module_id = module_id
        self.module_info = module_info
        self.is_loaded = False
        self.setup_ui()
        
    def setup_ui(self):
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(1)
        self.setMinimumWidth(280)
        self.setMaximumWidth(350)
        
        category = self.module_info.get('category', 'extension').lower()
        style = CATEGORY_STYLES.get(category, CATEGORY_STYLES['extension'])
        color = style['color']
        
        self.setStyleSheet(f"""
            ModuleCard {{
                border: 2px solid {color};
                border-radius: 8px;
                background-color: rgba(0,0,0,0.2);
            }}
            ModuleCard:hover {{
                background-color: rgba(255,255,255,0.05);
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header row
        header = QHBoxLayout()
        
        # Icon + Name
        icon = style['icon']
        name = self.module_info.get('name', self.module_id)
        name_label = make_selectable_label(f"{icon} {name}")
        name_label.setFont(QFont('Arial', 10, QFont.Bold))
        header.addWidget(name_label)
        
        header.addStretch()
        
        # Toggle switch
        self.toggle = QCheckBox()
        self.toggle.setStyleSheet(f"""
            QCheckBox::indicator {{
                width: 40px;
                height: 20px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {color};
                border-radius: 10px;
            }}
            QCheckBox::indicator:unchecked {{
                background-color: #444;
                border-radius: 10px;
            }}
        """)
        header.addWidget(self.toggle)
        
        layout.addLayout(header)
        
        # Description
        desc = self.module_info.get('description', 'No description')
        desc_label = make_selectable_label(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #aaa; font-size: 9px;")
        desc_label.setMaximumHeight(40)
        layout.addWidget(desc_label)
        
        # Requirements/info row
        info_parts = []
        
        reqs = self.module_info.get('requirements', [])
        if reqs:
            info_parts.append(f"Needs: {', '.join(reqs[:2])}")
        
        provides = self.module_info.get('provides', [])
        if provides:
            info_parts.append(f"Adds: {', '.join(provides[:2])}")
        
        if self.module_info.get('needs_api_key'):
            info_parts.append("🔑 API key")
        
        if self.module_info.get('needs_gpu'):
            info_parts.append("🎮 GPU")
        
        if info_parts:
            info_label = make_selectable_label(" • ".join(info_parts))
            info_label.setStyleSheet("color: #666; font-size: 8px;")
            layout.addWidget(info_label)
        
        # Status + Configure button
        bottom = QHBoxLayout()
        
        self.status_label = make_selectable_label("○ Off")
        self.status_label.setStyleSheet("color: #666; font-size: 9px;")
        bottom.addWidget(self.status_label)
        
        bottom.addStretch()
        
        self.config_btn = QPushButton("⚙")
        self.config_btn.setMaximumWidth(30)
        self.config_btn.setToolTip("Configure")
        self.config_btn.setEnabled(self.module_info.get('has_config', True))
        bottom.addWidget(self.config_btn)
        
        layout.addLayout(bottom)
    
    def set_loaded(self, loaded: bool):
        self.is_loaded = loaded
        self.toggle.setChecked(loaded)
        if loaded:
            self.status_label.setText("● On")
            self.status_label.setStyleSheet("color: #2ecc71; font-size: 9px;")
        else:
            self.status_label.setText("○ Off")
            self.status_label.setStyleSheet("color: #666; font-size: 9px;")


class CategorySection(QWidget):
    """A collapsible section for a category of modules."""
    
    def __init__(self, category: str, parent=None):
        super().__init__(parent)
        self.category = category
        self.cards: List[ModuleCard] = []
        self.is_collapsed = False
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 10)
        
        style = CATEGORY_STYLES.get(self.category.lower(), CATEGORY_STYLES['extension'])
        
        # Header
        header = QHBoxLayout()
        
        self.collapse_btn = QPushButton("▼")
        self.collapse_btn.setMaximumWidth(25)
        self.collapse_btn.setFlat(True)
        self.collapse_btn.clicked.connect(self.toggle_collapse)
        header.addWidget(self.collapse_btn)
        
        title = QLabel(f"{style['icon']} {style['name']}")
        title.setFont(QFont('Arial', 12, QFont.Bold))
        title.setStyleSheet(f"color: {style['color']};")
        header.addWidget(title)
        
        header.addStretch()
        
        self.count_label = QLabel("0 modules")
        self.count_label.setStyleSheet("color: #666;")
        header.addWidget(self.count_label)
        
        # Enable all / Disable all
        self.enable_all_btn = QPushButton("All On")
        self.enable_all_btn.setMaximumWidth(60)
        self.enable_all_btn.clicked.connect(self.enable_all)
        header.addWidget(self.enable_all_btn)
        
        self.disable_all_btn = QPushButton("All Off")
        self.disable_all_btn.setMaximumWidth(60)
        self.disable_all_btn.clicked.connect(self.disable_all)
        header.addWidget(self.disable_all_btn)
        
        layout.addLayout(header)
        
        # Cards container
        self.cards_widget = QWidget()
        self.cards_layout = QGridLayout(self.cards_widget)
        self.cards_layout.setSpacing(10)
        layout.addWidget(self.cards_widget)
    
    def add_card(self, card: ModuleCard):
        self.cards.append(card)
        row = (len(self.cards) - 1) // 3
        col = (len(self.cards) - 1) % 3
        self.cards_layout.addWidget(card, row, col)
        self.count_label.setText(f"{len(self.cards)} modules")
    
    def toggle_collapse(self):
        self.is_collapsed = not self.is_collapsed
        self.cards_widget.setVisible(not self.is_collapsed)
        self.collapse_btn.setText("▶" if self.is_collapsed else "▼")
    
    def enable_all(self):
        for card in self.cards:
            card.toggle.setChecked(True)
    
    def disable_all(self):
        for card in self.cards:
            card.toggle.setChecked(False)


class ModulesTab(QWidget):
    """Tab for managing all Enigma modules."""
    
    def __init__(self, parent=None, module_manager=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.categories: Dict[str, CategorySection] = {}
        self.all_cards: Dict[str, ModuleCard] = {}
        self.setup_ui()
        
        # Refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_status)
        self.refresh_timer.start(5000)
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title and controls
        header = QHBoxLayout()
        
        title = QLabel("🎛️ Module Manager")
        title.setFont(QFont('Arial', 16, QFont.Bold))
        header.addWidget(title)
        
        subtitle = QLabel("Enable/disable any capability")
        subtitle.setStyleSheet("color: #888;")
        header.addWidget(subtitle)
        
        header.addStretch()
        
        # Search
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search modules...")
        self.search_input.setMaximumWidth(200)
        self.search_input.textChanged.connect(self.filter_modules)
        header.addWidget(self.search_input)
        
        self.refresh_btn = QPushButton("↻ Refresh")
        self.refresh_btn.clicked.connect(self.refresh_status)
        header.addWidget(self.refresh_btn)
        
        layout.addLayout(header)
        
        # Main content - splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Modules organized by category
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.modules_widget = QWidget()
        self.modules_layout = QVBoxLayout(self.modules_widget)
        self.modules_layout.setSpacing(15)
        
        scroll.setWidget(self.modules_widget)
        left_layout.addWidget(scroll)
        
        splitter.addWidget(left_widget)
        
        # Right: Status panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Quick stats
        stats_group = QGroupBox("Status")
        stats_layout = QVBoxLayout(stats_group)
        
        self.loaded_label = QLabel("Loaded: 0 / 0")
        stats_layout.addWidget(self.loaded_label)
        
        self.cpu_bar = self._create_resource_bar("CPU")
        stats_layout.addWidget(self.cpu_bar)
        
        self.mem_bar = self._create_resource_bar("Memory")
        stats_layout.addWidget(self.mem_bar)
        
        self.gpu_bar = self._create_resource_bar("GPU")
        stats_layout.addWidget(self.gpu_bar)
        
        self.vram_bar = self._create_resource_bar("VRAM")
        stats_layout.addWidget(self.vram_bar)
        
        right_layout.addWidget(stats_group)
        
        # Activity log
        log_group = QGroupBox("Activity")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont('Consolas', 9))
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        right_layout.addWidget(log_group)
        
        right_layout.addStretch()
        
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 250])
        
        layout.addWidget(splitter)
        
        # Populate
        self.populate_modules()
        self.log("Module Manager initialized")
        
    def _create_resource_bar(self, name: str) -> QProgressBar:
        bar = QProgressBar()
        bar.setMaximum(100)
        bar.setValue(0)
        bar.setTextVisible(True)
        bar.setFormat(f"{name}: %p%")
        return bar
    
    def populate_modules(self):
        """Populate with all modules organized by category."""
        modules = self._get_all_modules()
        
        # Group by category
        by_category = {}
        for mod_id, info in modules.items():
            cat = info.get('category', 'extension').lower()
            if cat not in by_category:
                by_category[cat] = {}
            by_category[cat][mod_id] = info
        
        # Create sections in order
        category_order = ['core', 'generation', 'memory', 'perception', 'output', 'tools', 'network', 'interface', 'extension']
        
        for cat in category_order:
            if cat in by_category:
                section = CategorySection(cat)
                self.categories[cat] = section
                
                for mod_id, info in by_category[cat].items():
                    card = ModuleCard(mod_id, info)
                    card.toggle.stateChanged.connect(lambda state, mid=mod_id: self._on_toggle(mid, state))
                    card.config_btn.clicked.connect(lambda _, mid=mod_id: self._on_configure(mid))
                    section.add_card(card)
                    self.all_cards[mod_id] = card
                
                self.modules_layout.addWidget(section)
        
        self.modules_layout.addStretch()
        
        # Sync with actually loaded modules from ModuleManager
        self._sync_loaded_modules()
        
        # Auto-enable essential modules if ModuleManager is available
        self._auto_enable_essential_modules()
        
        self._update_stats()
    
    def _get_all_modules(self) -> dict:
        """Get all modules from registry."""
        # Try to get from actual registry
        try:
            from enigma.modules.registry import MODULE_REGISTRY
            modules = {}
            for mod_id, mod_class in MODULE_REGISTRY.items():
                info = mod_class.INFO
                modules[mod_id] = {
                    'name': info.name,
                    'description': info.description,
                    'category': info.category.value,
                    'requirements': info.requires,
                    'provides': info.provides,
                    'has_config': bool(info.config_schema),
                    'needs_gpu': info.requires_gpu or info.min_vram_mb > 0,
                    'needs_api_key': 'api_key' in str(info.config_schema),
                }
            return modules
        except Exception as e:
            print(f"Could not load from registry: {e}")
        
        # Fallback to hardcoded list
        return {
            # Core
            'model': {'name': 'Enigma Model', 'category': 'core', 'description': 'Core transformer model (nano to omega)', 'requirements': ['torch'], 'provides': ['text_generation'], 'has_config': True, 'needs_gpu': True},
            'tokenizer': {'name': 'Tokenizer', 'category': 'core', 'description': 'BPE tokenizer for text encoding', 'requirements': [], 'provides': ['tokenization'], 'has_config': True},
            'training': {'name': 'Training', 'category': 'core', 'description': 'Model training with AMP', 'requirements': ['model', 'tokenizer'], 'provides': ['model_training'], 'has_config': True, 'needs_gpu': True},
            'inference': {'name': 'Inference', 'category': 'core', 'description': 'Text generation engine', 'requirements': ['model', 'tokenizer'], 'provides': ['inference'], 'has_config': True},
            
            # Generation (AI Capabilities)
            'image_gen_local': {'name': 'Image Gen (Local)', 'category': 'generation', 'description': 'Stable Diffusion on your GPU', 'requirements': ['torch', 'diffusers'], 'provides': ['image_generation'], 'has_config': True, 'needs_gpu': True},
            'image_gen_api': {'name': 'Image Gen (Cloud)', 'category': 'generation', 'description': 'DALL-E / Replicate images', 'requirements': [], 'provides': ['image_generation'], 'has_config': True, 'needs_api_key': True},
            'code_gen_local': {'name': 'Code Gen (Local)', 'category': 'generation', 'description': 'Code using Enigma model', 'requirements': ['inference'], 'provides': ['code_generation'], 'has_config': True},
            'code_gen_api': {'name': 'Code Gen (Cloud)', 'category': 'generation', 'description': 'GPT-4 code generation', 'requirements': [], 'provides': ['code_generation'], 'has_config': True, 'needs_api_key': True},
            'video_gen_local': {'name': 'Video Gen (Local)', 'category': 'generation', 'description': 'AnimateDiff videos', 'requirements': ['torch', 'diffusers'], 'provides': ['video_generation'], 'has_config': True, 'needs_gpu': True},
            'video_gen_api': {'name': 'Video Gen (Cloud)', 'category': 'generation', 'description': 'Replicate video gen', 'requirements': [], 'provides': ['video_generation'], 'has_config': True, 'needs_api_key': True},
            'audio_gen_local': {'name': 'Audio/TTS (Local)', 'category': 'generation', 'description': 'Local text-to-speech', 'requirements': ['pyttsx3'], 'provides': ['tts'], 'has_config': True},
            'audio_gen_api': {'name': 'Audio/TTS (Cloud)', 'category': 'generation', 'description': 'ElevenLabs / MusicGen', 'requirements': [], 'provides': ['tts', 'music'], 'has_config': True, 'needs_api_key': True},
            
            # Memory
            'memory': {'name': 'Memory Manager', 'category': 'memory', 'description': 'Conversation storage', 'requirements': [], 'provides': ['memory'], 'has_config': True},
            'embedding_local': {'name': 'Embeddings (Local)', 'category': 'memory', 'description': 'Semantic search vectors', 'requirements': ['sentence-transformers'], 'provides': ['embeddings'], 'has_config': True},
            'embedding_api': {'name': 'Embeddings (Cloud)', 'category': 'memory', 'description': 'OpenAI embeddings', 'requirements': [], 'provides': ['embeddings'], 'has_config': True, 'needs_api_key': True},
            
            # Perception
            'voice_input': {'name': 'Voice Input', 'category': 'perception', 'description': 'Speech-to-text', 'requirements': [], 'provides': ['stt'], 'has_config': True},
            'vision': {'name': 'Vision', 'category': 'perception', 'description': 'Camera and image analysis', 'requirements': ['opencv'], 'provides': ['vision'], 'has_config': True},
            
            # Output
            'voice_output': {'name': 'Voice Output', 'category': 'output', 'description': 'Text-to-speech', 'requirements': [], 'provides': ['tts'], 'has_config': True},
            'avatar': {'name': 'Avatar', 'category': 'output', 'description': 'Visual AI representation', 'requirements': [], 'provides': ['avatar'], 'has_config': True},
            
            # Tools
            'web_tools': {'name': 'Web Tools', 'category': 'tools', 'description': 'Web search and fetch', 'requirements': ['requests'], 'provides': ['web'], 'has_config': True},
            'file_tools': {'name': 'File Tools', 'category': 'tools', 'description': 'File operations', 'requirements': [], 'provides': ['files'], 'has_config': True},
            
            # Network
            'api_server': {'name': 'API Server', 'category': 'network', 'description': 'REST API endpoint', 'requirements': ['flask'], 'provides': ['api'], 'has_config': True},
            'network': {'name': 'Multi-Device', 'category': 'network', 'description': 'Distributed inference', 'requirements': [], 'provides': ['distributed'], 'has_config': True},
            
            # Interface
            'gui': {'name': 'GUI', 'category': 'interface', 'description': 'Graphical interface', 'requirements': ['PyQt5'], 'provides': ['gui'], 'has_config': True},
        }
    
    def _on_toggle(self, module_id: str, state: int):
        """Handle module toggle."""
        enabled = state == Qt.Checked
        action = "Loading" if enabled else "Unloading"
        self.log(f"{action} {module_id}...")
        
        # Actually load/unload the module if ModuleManager is available
        if self.module_manager:
            try:
                if enabled:
                    success = self.module_manager.load(module_id)
                    if not success:
                        self.log(f"✗ Failed to load {module_id}")
                        # Revert toggle
                        if module_id in self.all_cards:
                            self.all_cards[module_id].toggle.blockSignals(True)
                            self.all_cards[module_id].toggle.setChecked(False)
                            self.all_cards[module_id].toggle.blockSignals(False)
                        return
                else:
                    success = self.module_manager.unload(module_id)
                    if not success:
                        self.log(f"✗ Failed to unload {module_id}")
                        return
            except Exception as e:
                self.log(f"✗ Error: {str(e)}")
                return
        
        # Update card
        if module_id in self.all_cards:
            self.all_cards[module_id].set_loaded(enabled)
        
        self.log(f"✓ {module_id} {'enabled' if enabled else 'disabled'}")
        self._update_stats()
    
    def _sync_loaded_modules(self):
        """Sync UI state with actually loaded modules."""
        if not self.module_manager:
            return
        
        try:
            loaded_modules = self.module_manager.list_loaded()
            for mod_id in loaded_modules:
                if mod_id in self.all_cards:
                    self.all_cards[mod_id].toggle.blockSignals(True)
                    self.all_cards[mod_id].set_loaded(True)
                    self.all_cards[mod_id].toggle.blockSignals(False)
            self.log(f"Synced {len(loaded_modules)} loaded modules")
        except Exception as e:
            self.log(f"Could not sync modules: {e}")
    
    def _auto_enable_essential_modules(self):
        """Auto-enable essential modules that should be on by default."""
        if not self.module_manager:
            self.log("⚠ No module manager - modules won't auto-load")
            return
        
        # Essential modules that should be loaded by default
        essential_modules = [
            'model',      # Core model
            'tokenizer',  # Tokenizer
            'inference',  # Inference engine
        ]
        
        # Check which ones aren't loaded yet
        try:
            loaded = self.module_manager.list_loaded()
            for mod_id in essential_modules:
                if mod_id not in loaded and mod_id in self.all_cards:
                    # Try to load it
                    self.log(f"Auto-loading essential module: {mod_id}")
                    try:
                        success = self.module_manager.load(mod_id)
                        if success:
                            self.all_cards[mod_id].toggle.blockSignals(True)
                            self.all_cards[mod_id].set_loaded(True)
                            self.all_cards[mod_id].toggle.blockSignals(False)
                            self.log(f"✓ {mod_id} loaded automatically")
                        else:
                            self.log(f"⚠ Could not auto-load {mod_id}")
                    except Exception as e:
                        self.log(f"⚠ Error loading {mod_id}: {e}")
        except Exception as e:
            self.log(f"Error during auto-enable: {e}")
    
    def _on_configure(self, module_id: str):
        """Show configuration for a module."""
        QMessageBox.information(
            self,
            f"Configure: {module_id}",
            f"Configuration for {module_id}\n\n(Full config dialog coming soon)"
        )
    
    def filter_modules(self, text: str):
        """Filter modules by search text."""
        text = text.lower()
        for mod_id, card in self.all_cards.items():
            visible = (not text or 
                       text in mod_id.lower() or 
                       text in card.module_info.get('name', '').lower() or
                       text in card.module_info.get('description', '').lower())
            card.setVisible(visible)
    
    def _update_stats(self):
        """Update loaded/total stats."""
        loaded = sum(1 for c in self.all_cards.values() if c.is_loaded)
        total = len(self.all_cards)
        self.loaded_label.setText(f"Loaded: {loaded} / {total}")
    
    def refresh_status(self):
        """Refresh system status and module states."""
        # Sync module states with ModuleManager
        if self.module_manager:
            try:
                loaded = self.module_manager.list_loaded()
                for mod_id, card in self.all_cards.items():
                    is_loaded = mod_id in loaded
                    if card.is_loaded != is_loaded:
                        card.toggle.blockSignals(True)
                        card.set_loaded(is_loaded)
                        card.toggle.blockSignals(False)
                self._update_stats()
            except Exception:
                pass
        
        # Update resource bars
        try:
            import psutil
            self.cpu_bar.setValue(int(psutil.cpu_percent()))
            self.mem_bar.setValue(int(psutil.virtual_memory().percent))
        except ImportError:
            pass
        
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                self.vram_bar.setValue(int(allocated / total * 100))
                self.gpu_bar.setValue(50)  # Would need pynvml for actual GPU usage
        except Exception:
            pass
    
    def log(self, message: str):
        """Add message to log."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")


if not HAS_PYQT:
    class ModulesTab:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt5 is required for the Module Manager")
