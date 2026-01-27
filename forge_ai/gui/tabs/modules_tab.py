"""
Module Manager Tab - Control all Forge capabilities
====================================================

Clean, functional interface for managing modules.
"""
from typing import Dict, List, TYPE_CHECKING, Any, cast

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea,
    QLabel, QPushButton, QFrame, QGroupBox, QCheckBox,
    QLineEdit, QProgressBar, QMessageBox, QSplitter,
    QTextEdit, QSizePolicy, QListWidget, QListWidgetItem,
    QStackedWidget
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor

from .shared_components import NoScrollComboBox

# Qt enum constants
AlignCenter = Qt.AlignmentFlag.AlignCenter
ScrollBarAlwaysOff = Qt.ScrollBarPolicy.ScrollBarAlwaysOff
Checked = Qt.CheckState.Checked


# Category definitions (no emojis, just colors)
CATEGORIES = {
    'core': {'color': '#e74c3c', 'name': 'Core'},
    'generation': {'color': '#e91e63', 'name': 'AI Generation'},
    'memory': {'color': '#3498db', 'name': 'Memory'},
    'perception': {'color': '#9b59b6', 'name': 'Perception'},
    'output': {'color': '#f39c12', 'name': 'Output'},
    'tools': {'color': '#1abc9c', 'name': 'Tools'},
    'network': {'color': '#e67e22', 'name': 'Network'},
    'interface': {'color': '#2ecc71', 'name': 'Interface'},
    'extension': {'color': '#95a5a6', 'name': 'Extension'},
}


class ModuleListItem(QFrame):
    """A single module row in the list."""
    
    def __init__(self, module_id: str, module_info: dict, parent=None):
        super().__init__(parent)
        self.module_id = module_id
        self.module_info = module_info
        self.is_loaded = False
        self.is_processing = False  # Flag to prevent double-clicks
        self._setup_ui()
        
    def _setup_ui(self):
        self.setMinimumHeight(50)
        self.setMaximumHeight(60)
        self.setFrameStyle(QFrame.NoFrame)
        
        category = self.module_info.get('category', 'extension').lower()
        color = CATEGORIES.get(category, CATEGORIES['extension'])['color']
        
        self.setStyleSheet(f"""
            ModuleListItem {{
                background: transparent;
                border-left: 4px solid {color};
                padding-left: 8px;
            }}
            ModuleListItem:hover {{
                background: rgba(255,255,255,0.08);
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(12)
        
        # Toggle checkbox - larger for easier clicking
        self.toggle = QCheckBox()
        self.toggle.setFixedSize(24, 24)
        self.toggle.setStyleSheet("""
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
            QCheckBox::indicator:checked {
                background-color: #2ecc71;
                border: 2px solid #27ae60;
                border-radius: 4px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #333;
                border: 2px solid #555;
                border-radius: 4px;
            }
        """)
        self.toggle.setToolTip("Enable or disable this module")
        layout.addWidget(self.toggle)
        
        # Name and description
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        name = self.module_info.get('name', self.module_id)
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        info_layout.addWidget(self.name_label)
        
        desc = self.module_info.get('description', '')
        # Truncate long descriptions
        if len(desc) > 60:
            desc = desc[:60] + "..."
        self.desc_label = QLabel(desc)
        self.desc_label.setStyleSheet("color: #888; font-size: 13px;")
        info_layout.addWidget(self.desc_label)
        
        layout.addLayout(info_layout, stretch=1)
        
        # Status indicator
        self.status_label = QLabel("OFF")
        self.status_label.setFixedWidth(40)
        self.status_label.setAlignment(AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 13px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.status_label)
        
        # Requirements indicator (compact)
        needs = []
        if self.module_info.get('needs_gpu'):
            needs.append("GPU")
        if self.module_info.get('needs_api_key'):
            needs.append("API")
        
        if needs:
            req_label = QLabel("|".join(needs))
            req_label.setStyleSheet("color: #f39c12; font-size: 12px;")
            req_label.setFixedWidth(45)
            layout.addWidget(req_label)
    
    def set_loaded(self, loaded: bool):
        self.is_loaded = loaded
        self.toggle.setChecked(loaded)
        if loaded:
            self.status_label.setText("ON")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #2ecc71;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
        else:
            self.status_label.setText("OFF")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #666;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
        self.is_processing = False
    
    def set_processing(self, processing: bool):
        """Set processing state - disables toggle during load/unload."""
        self.is_processing = processing
        self.toggle.setEnabled(not processing)
        if processing:
            self.status_label.setText("...")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #f39c12;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)


class ModulesTab(QWidget):
    """Clean module management interface."""
    
    def __init__(self, parent=None, module_manager=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.module_items: Dict[str, ModuleListItem] = {}
        self._setup_ui()
        
        # Refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh_status)
        self.refresh_timer.start(5000)
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("Module Manager")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Search
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search...")
        self.search_input.setFixedWidth(220)
        self.search_input.setMinimumHeight(32)
        self.search_input.textChanged.connect(self._filter_modules)
        self.search_input.setToolTip("Search modules by name or description")
        self.search_input.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 14px;
            }
        """)
        header_layout.addWidget(self.search_input)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setMinimumWidth(90)
        refresh_btn.setMinimumHeight(32)
        refresh_btn.clicked.connect(self._refresh_status)
        refresh_btn.setToolTip("Refresh module status and resource usage")
        refresh_btn.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                font-size: 13px;
            }
        """)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Main content
        content_layout = QHBoxLayout()
        
        # Left side - Category filter + Module list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        
        # Category filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Category:"))
        
        self.category_combo = NoScrollComboBox()
        self.category_combo.addItem("All Categories", "all")
        for cat_id, cat_info in CATEGORIES.items():
            self.category_combo.addItem(cat_info['name'], cat_id)
        self.category_combo.currentIndexChanged.connect(self._filter_modules)
        self.category_combo.setMinimumWidth(120)
        self.category_combo.setMinimumHeight(30)
        self.category_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.category_combo.setToolTip("Filter modules by category")
        filter_layout.addWidget(self.category_combo)
        
        filter_layout.addStretch()
        
        # Quick actions - improved button styling
        self.enable_all_btn = QPushButton("Enable All")
        self.enable_all_btn.setMinimumWidth(90)
        self.enable_all_btn.setMinimumHeight(32)
        self.enable_all_btn.clicked.connect(self._enable_all_visible)
        self.enable_all_btn.setToolTip("Enable all visible modules in the current filter")
        self.enable_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: #1e1e2e;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        filter_layout.addWidget(self.enable_all_btn)
        
        self.disable_all_btn = QPushButton("Disable All")
        self.disable_all_btn.setMinimumWidth(90)
        self.disable_all_btn.setMinimumHeight(32)
        self.disable_all_btn.clicked.connect(self._disable_all_visible)
        self.disable_all_btn.setToolTip("Disable all visible modules in the current filter")
        self.disable_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        filter_layout.addWidget(self.disable_all_btn)
        
        left_layout.addLayout(filter_layout)
        
        # Module list in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #444;
                border-radius: 6px;
                background: transparent;
            }
        """)
        
        self.modules_container = QWidget()
        self.modules_layout = QVBoxLayout(self.modules_container)
        self.modules_layout.setSpacing(4)
        self.modules_layout.setContentsMargins(8, 8, 8, 8)
        
        scroll.setWidget(self.modules_container)
        left_layout.addWidget(scroll)
        
        content_layout.addWidget(left_panel, stretch=3)
        
        # Right side - Status panel (collapsible on small screens)
        right_panel = QWidget()
        right_panel.setMinimumWidth(200)
        right_panel.setMaximumWidth(320)
        right_panel.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)
        right_layout.setSpacing(12)
        
        # Stats box
        stats_box = QGroupBox("Status")
        stats_layout = QVBoxLayout(stats_box)
        stats_layout.setSpacing(10)
        
        self.loaded_label = QLabel("Loaded: 0 / 0")
        self.loaded_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        stats_layout.addWidget(self.loaded_label)
        
        # AI Connection indicator
        conn_box = QFrame()
        conn_box.setStyleSheet("""
            QFrame {
                background: #1a1a1a;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        conn_layout = QVBoxLayout(conn_box)
        conn_layout.setContentsMargins(8, 8, 8, 8)
        conn_layout.setSpacing(4)
        
        conn_title = QLabel("AI Status")
        conn_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        conn_layout.addWidget(conn_title)
        
        self.ai_status_indicator = QLabel("[OFF] Disconnected")
        self.ai_status_indicator.setStyleSheet("color: #ef4444; font-size: 14px;")
        conn_layout.addWidget(self.ai_status_indicator)
        
        self.ai_status_detail = QLabel("No modules loaded")
        self.ai_status_detail.setStyleSheet("color: #888; font-size: 13px;")
        self.ai_status_detail.setWordWrap(True)
        conn_layout.addWidget(self.ai_status_detail)
        
        stats_layout.addWidget(conn_box)
        
        # Resource bars
        stats_layout.addWidget(QLabel("CPU Usage:"))
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setMaximum(100)
        self.cpu_bar.setTextVisible(True)
        self.cpu_bar.setFixedHeight(20)
        stats_layout.addWidget(self.cpu_bar)
        
        stats_layout.addWidget(QLabel("Memory:"))
        self.mem_bar = QProgressBar()
        self.mem_bar.setMaximum(100)
        self.mem_bar.setTextVisible(True)
        self.mem_bar.setFixedHeight(20)
        stats_layout.addWidget(self.mem_bar)
        
        stats_layout.addWidget(QLabel("GPU VRAM:"))
        self.vram_bar = QProgressBar()
        self.vram_bar.setMaximum(100)
        self.vram_bar.setTextVisible(True)
        self.vram_bar.setFixedHeight(20)
        stats_layout.addWidget(self.vram_bar)
        
        right_layout.addWidget(stats_box)
        
        # Activity log
        log_box = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_box)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont('Consolas', 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background: #1a1a1a;
                border: none;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(lambda: self.log_text.clear())
        log_layout.addWidget(clear_btn)
        
        right_layout.addWidget(log_box)
        
        right_layout.addStretch()
        
        content_layout.addWidget(right_panel, stretch=1)
        
        layout.addLayout(content_layout)
        
        # Populate modules
        self._populate_modules()
        self._log("Module Manager ready")
        
    def _populate_modules(self):
        """Populate the module list."""
        modules = self._get_all_modules()
        
        # Sort by category then name
        sorted_modules = sorted(
            modules.items(), 
            key=lambda x: (x[1].get('category', 'z'), x[1].get('name', x[0]))
        )
        
        current_category = None
        
        for mod_id, info in sorted_modules:
            category = info.get('category', 'extension').lower()
            
            # Add category header if changed
            if category != current_category:
                current_category = category
                cat_info = CATEGORIES.get(category, CATEGORIES['extension'])
                
                header = QLabel(cat_info['name'].upper())
                header.setStyleSheet(f"""
                    QLabel {{
                        color: {cat_info['color']};
                        font-size: 11px;
                        font-weight: bold;
                        padding: 10px 5px 5px 5px;
                        border-bottom: 1px solid {cat_info['color']};
                    }}
                """)
                self.modules_layout.addWidget(header)
            
            # Add module item
            item = ModuleListItem(mod_id, info)
            item.toggle.stateChanged.connect(
                lambda state, mid=mod_id: self._on_toggle(mid, state)
            )
            self.modules_layout.addWidget(item)
            self.module_items[mod_id] = item
        
        self.modules_layout.addStretch()
        
        # Sync with loaded modules
        self._sync_loaded_modules()
        self._update_stats()
        
    def _get_all_modules(self) -> dict:
        """Get all available modules."""
        try:
            from forge_ai.modules.registry import MODULE_REGISTRY
            modules = {}
            for mod_id, mod_class in MODULE_REGISTRY.items():
                info = mod_class.INFO
                modules[mod_id] = {
                    'name': info.name,
                    'description': info.description,
                    'category': info.category.value,
                    'requirements': info.requires,
                    'provides': info.provides,
                    'needs_gpu': info.requires_gpu or info.min_vram_mb > 0,
                    'needs_api_key': 'api_key' in str(info.config_schema),
                }
            return modules
        except Exception:
            pass
        
        # Fallback list
        return {
            'model': {'name': 'Forge Model', 'category': 'core', 'description': 'Core transformer model', 'needs_gpu': True},
            'tokenizer': {'name': 'Tokenizer', 'category': 'core', 'description': 'Text tokenization'},
            'training': {'name': 'Training', 'category': 'core', 'description': 'Model training', 'needs_gpu': True},
            'inference': {'name': 'Inference (Local)', 'category': 'core', 'description': 'Local text generation'},
            'chat_api': {'name': 'Chat (Cloud)', 'category': 'core', 'description': 'GPT-4/Claude chat - No GPU needed!', 'needs_api_key': True},
            'personality': {'name': 'Personality', 'category': 'core', 'description': 'Customize AI personality and tone'},
            'instructions': {'name': 'Instructions', 'category': 'core', 'description': 'System prompts for AI behavior'},
            'model_router': {'name': 'Model Router', 'category': 'core', 'description': 'Route to specialized models'},
            'scaling': {'name': 'Model Scaling', 'category': 'core', 'description': 'Grow/shrink model sizes'},
            'huggingface': {'name': 'HuggingFace', 'category': 'core', 'description': 'HuggingFace Hub integration'},
            'image_gen_local': {'name': 'Image Gen (Local)', 'category': 'generation', 'description': 'Stable Diffusion', 'needs_gpu': True},
            'image_gen_api': {'name': 'Image Gen (Cloud)', 'category': 'generation', 'description': 'DALL-E / Replicate', 'needs_api_key': True},
            'code_gen_local': {'name': 'Code Gen (Local)', 'category': 'generation', 'description': 'Local code generation'},
            'code_gen_api': {'name': 'Code Gen (Cloud)', 'category': 'generation', 'description': 'GPT-4 code', 'needs_api_key': True},
            'video_gen_local': {'name': 'Video Gen (Local)', 'category': 'generation', 'description': 'AnimateDiff', 'needs_gpu': True},
            'video_gen_api': {'name': 'Video Gen (Cloud)', 'category': 'generation', 'description': 'Replicate video', 'needs_api_key': True},
            'audio_gen_local': {'name': 'Audio (Local)', 'category': 'generation', 'description': 'Local TTS'},
            'audio_gen_api': {'name': 'Audio (Cloud)', 'category': 'generation', 'description': 'ElevenLabs', 'needs_api_key': True},
            'threed_gen_local': {'name': '3D Gen (Local)', 'category': 'generation', 'description': 'Shap-E / Point-E', 'needs_gpu': True},
            'threed_gen_api': {'name': '3D Gen (Cloud)', 'category': 'generation', 'description': 'Replicate 3D', 'needs_api_key': True},
            'gif_gen': {'name': 'GIF Generation', 'category': 'generation', 'description': 'Create animated GIFs'},
            'memory': {'name': 'Memory', 'category': 'memory', 'description': 'Conversation storage'},
            'embedding_local': {'name': 'Embeddings (Local)', 'category': 'memory', 'description': 'Semantic vectors'},
            'embedding_api': {'name': 'Embeddings (Cloud)', 'category': 'memory', 'description': 'OpenAI embeddings', 'needs_api_key': True},
            'notes': {'name': 'Notes', 'category': 'memory', 'description': 'Persistent notes storage'},
            'sessions': {'name': 'Sessions', 'category': 'memory', 'description': 'Conversation session management'},
            'voice_input': {'name': 'Voice Input', 'category': 'perception', 'description': 'Speech-to-text'},
            'vision': {'name': 'Vision', 'category': 'perception', 'description': 'Image analysis'},
            'camera': {'name': 'Camera', 'category': 'perception', 'description': 'Webcam capture and analysis'},
            'motion_tracking': {'name': 'Motion Tracking', 'category': 'perception', 'description': 'Body/hand/face tracking'},
            'voice_output': {'name': 'Voice Output', 'category': 'output', 'description': 'Text-to-speech'},
            'avatar': {'name': 'Avatar', 'category': 'output', 'description': 'Visual representation'},
            'voice_clone': {'name': 'Voice Cloning', 'category': 'output', 'description': 'Clone voices from samples'},
            'web_tools': {'name': 'Web Tools', 'category': 'tools', 'description': 'Web search/fetch'},
            'file_tools': {'name': 'File Tools', 'category': 'tools', 'description': 'File operations'},
            'scheduler': {'name': 'Scheduler', 'category': 'tools', 'description': 'Timed tasks and reminders'},
            'terminal': {'name': 'Terminal', 'category': 'tools', 'description': 'Command execution'},
            'game_ai': {'name': 'Game AI', 'category': 'tools', 'description': 'Gaming AI assistant'},
            'robot_control': {'name': 'Robot Control', 'category': 'tools', 'description': 'Robot/hardware control'},
            'api_server': {'name': 'API Server', 'category': 'network', 'description': 'REST API'},
            'network': {'name': 'Multi-Device', 'category': 'network', 'description': 'Distributed inference'},
            'gui': {'name': 'GUI', 'category': 'interface', 'description': 'Graphical interface'},
            'analytics': {'name': 'Analytics', 'category': 'interface', 'description': 'Usage stats and insights'},
            'dashboard': {'name': 'Dashboard', 'category': 'interface', 'description': 'System overview'},
            'examples': {'name': 'Examples', 'category': 'interface', 'description': 'Prompt templates'},
            'logs': {'name': 'Logs', 'category': 'interface', 'description': 'System log viewer'},
            'workspace': {'name': 'Workspace', 'category': 'interface', 'description': 'Project management'},
        }
    
    def _on_toggle(self, module_id: str, state: int):
        """Handle module toggle."""
        # Check if item exists and is not already processing
        if module_id in self.module_items:
            item = self.module_items[module_id]
            if item.is_processing:
                # Already processing - ignore this click
                return
            # Set processing state to prevent double-clicks
            item.set_processing(True)
        
        enabled = state == Checked
        action = "Loading" if enabled else "Unloading"
        self._log(f"{action} {module_id}...")
        
        if self.module_manager:
            try:
                # Debug: check can_load first
                if enabled:
                    can_load, reason = self.module_manager.can_load(module_id)
                    if not can_load:
                        self._log(f"Cannot load {module_id}: {reason}")
                    
                if enabled:
                    success = self.module_manager.load(module_id)
                else:
                    success = self.module_manager.unload(module_id)
                
                if not success:
                    self._log(f"FAILED: Could not {'load' if enabled else 'unload'} {module_id}")
                    # Revert
                    if module_id in self.module_items:
                        item = self.module_items[module_id]
                        item.toggle.blockSignals(True)
                        item.toggle.setChecked(not enabled)
                        item.toggle.blockSignals(False)
                        item.set_processing(False)
                    return
                
                self._sync_options_menu(module_id, enabled)
                self._sync_tab_visibility(module_id, enabled)
                
                # Save config to persist changes
                try:
                    self.module_manager.save_config()
                    self._log(f"Config saved")
                except Exception as save_err:
                    self._log(f"Warning: Could not save config: {save_err}")
                
            except Exception as e:
                self._log(f"ERROR: {str(e)}")
                return
        
        if module_id in self.module_items:
            self.module_items[module_id].set_loaded(enabled)
        
        self._log(f"OK: {module_id} {'enabled' if enabled else 'disabled'}")
        self._update_stats()
    
    def _sync_tab_visibility(self, module_id: str, enabled: bool):
        """Sync tab visibility with module state in main window."""
        try:
            main_window: Any = self.parent()
            while main_window and not hasattr(main_window, 'on_module_toggled'):
                main_window = main_window.parent()
            
            if main_window and hasattr(main_window, 'on_module_toggled'):
                main_window.on_module_toggled(module_id, enabled)
        except Exception:
            pass
    
    def _sync_options_menu(self, module_id: str, enabled: bool):
        """Sync with main window Options menu."""
        try:
            main_window: Any = self.parent()
            while main_window and not hasattr(main_window, 'avatar_action'):
                main_window = main_window.parent()
            
            if not main_window:
                return
            
            if module_id == 'avatar' and hasattr(main_window, 'avatar_action'):
                main_window.avatar_action.blockSignals(True)
                main_window.avatar_action.setChecked(enabled)
                main_window.avatar_action.setText(f"Avatar ({'ON' if enabled else 'OFF'})")
                main_window.avatar_action.blockSignals(False)
            
            elif module_id == 'voice_output' and hasattr(main_window, 'auto_speak_action'):
                main_window.auto_speak_action.blockSignals(True)
                main_window.auto_speak_action.setChecked(enabled)
                main_window.auto_speak_action.setText(f"AI Auto-Speak ({'ON' if enabled else 'OFF'})")
                main_window.auto_speak = enabled
                main_window.auto_speak_action.blockSignals(False)
            
            elif module_id == 'voice_input' and hasattr(main_window, 'microphone_action'):
                main_window.microphone_action.blockSignals(True)
                main_window.microphone_action.setChecked(enabled)
                main_window.microphone_action.setText(f"Microphone ({'ON' if enabled else 'OFF'})")
                main_window.microphone_enabled = enabled
                main_window.microphone_action.blockSignals(False)
                
        except Exception:
            pass
    
    def _sync_loaded_modules(self):
        """Sync UI with actually loaded modules."""
        if not self.module_manager:
            return
        
        try:
            loaded = self.module_manager.list_loaded()
            for mod_id in loaded:
                if mod_id in self.module_items:
                    item = self.module_items[mod_id]
                    item.toggle.blockSignals(True)
                    item.set_loaded(True)
                    item.toggle.blockSignals(False)
            self._log(f"Synced {len(loaded)} loaded modules")
        except Exception as e:
            self._log(f"Sync error: {e}")
    
    def _filter_modules(self):
        """Filter modules by search and category."""
        search_text = self.search_input.text().lower()
        selected_cat = self.category_combo.currentData()
        
        for mod_id, item in self.module_items.items():
            info = item.module_info
            
            # Category match
            cat_match = (selected_cat == "all" or 
                        info.get('category', '').lower() == selected_cat)
            
            # Search match
            search_match = (not search_text or
                          search_text in mod_id.lower() or
                          search_text in info.get('name', '').lower() or
                          search_text in info.get('description', '').lower())
            
            item.setVisible(cat_match and search_match)
    
    def _enable_all_visible(self):
        """Enable all visible modules with progress feedback and failure handling."""
        # Collect modules to enable
        to_enable = []
        for mod_id, item in self.module_items.items():
            if item.isVisible() and not item.is_loaded:
                to_enable.append(mod_id)
        
        if not to_enable:
            self._log("No modules to enable")
            return
        
        # Warn user this may take a while
        reply = QMessageBox.question(
            self, "Enable All Modules",
            f"This will attempt to load {len(to_enable)} modules.\n"
            "Some modules may fail if dependencies are missing.\n"
            "The UI may freeze during loading.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        self._log(f"Enabling {len(to_enable)} modules...")
        
        # Track failures for summary
        self._enable_failures = []
        self._enable_successes = []
        
        # Use a list to track progress through modules
        self._enable_queue = to_enable.copy()
        self._enable_next_module()
    
    def _enable_next_module(self):
        """Enable the next module in the queue with error handling."""
        if not hasattr(self, '_enable_queue') or not self._enable_queue:
            # Done - show summary
            successes = len(getattr(self, '_enable_successes', []))
            failures = getattr(self, '_enable_failures', [])
            
            if failures:
                self._log(f"Completed: {successes} enabled, {len(failures)} failed")
                for mod_id, reason in failures:
                    self._log(f"  FAILED: {mod_id} - {reason}")
            else:
                self._log(f"All {successes} modules enabled successfully")
            return
        
        mod_id = self._enable_queue.pop(0)
        if mod_id in self.module_items:
            item = self.module_items[mod_id]
            if not item.is_loaded:
                self._log(f"Enabling {mod_id}... ({len(self._enable_queue)} remaining)")
                # Process events so the log updates
                from PyQt5.QtWidgets import QApplication
                QApplication.processEvents()
                
                # Check if module can be loaded first
                if self.module_manager:
                    can_load, reason = self.module_manager.can_load(mod_id)
                    if not can_load:
                        self._log(f"  Skipping {mod_id}: {reason}")
                        self._enable_failures.append((mod_id, reason))
                    else:
                        # Try to enable
                        item.toggle.setChecked(True)
                        # Check if it actually loaded
                        if item.is_loaded:
                            self._enable_successes.append(mod_id)
                        else:
                            self._enable_failures.append((mod_id, "Load failed"))
                else:
                    item.toggle.setChecked(True)
                    self._enable_successes.append(mod_id)
        
        # Schedule next module with a small delay to allow UI to update
        if self._enable_queue:
            QTimer.singleShot(100, self._enable_next_module)
        else:
            # Final summary after last module
            QTimer.singleShot(200, self._enable_next_module)
    
    def _disable_all_visible(self):
        """Disable all visible modules."""
        for item in self.module_items.values():
            if item.isVisible() and item.is_loaded:
                item.toggle.setChecked(False)
    
    def _update_stats(self):
        """Update statistics display."""
        loaded = sum(1 for i in self.module_items.values() if i.is_loaded)
        total = len(self.module_items)
        self.loaded_label.setText(f"Loaded: {loaded} / {total}")
        
        # Update AI connection status
        loaded_ids = [mid for mid, item in self.module_items.items() if item.is_loaded]
        
        core_loaded = any(m in loaded_ids for m in ['model', 'tokenizer', 'inference'])
        gen_loaded = any('gen' in m for m in loaded_ids)
        
        if core_loaded and gen_loaded:
            self.ai_status_indicator.setText("[ON] Connected (Full)")
            self.ai_status_indicator.setStyleSheet("color: #22c55e; font-size: 14px; font-weight: bold;")
            self.ai_status_detail.setText("Core AI + generation ready")
        elif core_loaded:
            self.ai_status_indicator.setText("[ON] Connected (Core)")
            self.ai_status_indicator.setStyleSheet("color: #22c55e; font-size: 14px; font-weight: bold;")
            self.ai_status_detail.setText("Chat available, enable gen modules for more")
        elif loaded > 0:
            self.ai_status_indicator.setText("[...] Partial")
            self.ai_status_indicator.setStyleSheet("color: #f59e0b; font-size: 14px; font-weight: bold;")
            self.ai_status_detail.setText(f"{loaded} modules loaded, enable core for chat")
        else:
            self.ai_status_indicator.setText("[OFF] Disconnected")
            self.ai_status_indicator.setStyleSheet("color: #ef4444; font-size: 14px;")
            self.ai_status_detail.setText("Enable modules to start")
    
    def _refresh_status(self):
        """Refresh status indicators."""
        # Sync module states
        if self.module_manager:
            try:
                loaded = self.module_manager.list_loaded()
                for mod_id, item in self.module_items.items():
                    is_loaded = mod_id in loaded
                    if item.is_loaded != is_loaded:
                        item.toggle.blockSignals(True)
                        item.set_loaded(is_loaded)
                        item.toggle.blockSignals(False)
                self._update_stats()
            except Exception:
                pass
        
        # Update resource bars
        try:
            import psutil
            self.cpu_bar.setValue(int(psutil.cpu_percent()))
            self.mem_bar.setValue(int(psutil.virtual_memory().percent))
        except ImportError:
            self.cpu_bar.setValue(0)
            self.mem_bar.setValue(0)
        
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                self.vram_bar.setValue(int(allocated / total * 100))
            else:
                self.vram_bar.setValue(0)
        except Exception:
            self.vram_bar.setValue(0)
    
    def _log(self, message: str):
        """Add to activity log."""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {message}")
