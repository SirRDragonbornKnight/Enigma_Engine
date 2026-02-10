"""
Module Manager Tab - Control all Forge capabilities
====================================================

Clean, functional interface for managing modules.
"""
from typing import Any

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

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
        self.is_processing = False
        self._setup_ui()
        
    def _setup_ui(self):
        self.setMinimumHeight(50)
        self.setMaximumHeight(60)
        self.setFrameStyle(QFrame.NoFrame)
        self.setCursor(Qt.CursorShape.PointingHandCursor)  # Show hand cursor to indicate clickable
        
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
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)
        
        # Toggle button - styled as ON/OFF switch
        self.toggle_btn = QPushButton("OFF")
        self.toggle_btn.setFixedSize(50, 28)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: #bac2de;
                border: 2px solid #555;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:checked {
                background-color: #2ecc71;
                color: #1e1e2e;
                border: 2px solid #27ae60;
            }
            QPushButton:hover {
                border-color: #888;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        self.toggle_btn.setToolTip("Click to enable or disable this module")
        layout.addWidget(self.toggle_btn)
        
        # Keep toggle as alias for compatibility
        self.toggle = self.toggle_btn
        # Name and description
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        name = self.module_info.get('name', self.module_id)
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.name_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        info_layout.addWidget(self.name_label)
        
        desc = self.module_info.get('description', '')
        # Truncate long descriptions
        if len(desc) > 60:
            desc = desc[:60] + "..."
        self.desc_label = QLabel(desc)
        self.desc_label.setStyleSheet("color: #bac2de; font-size: 11px;")
        self.desc_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        info_layout.addWidget(self.desc_label)
        
        layout.addLayout(info_layout, stretch=1)
        
        # Requirements indicator (compact)
        needs = []
        if self.module_info.get('needs_gpu'):
            needs.append("GPU")
        if self.module_info.get('needs_api_key'):
            needs.append("API")
        
        if needs:
            req_label = QLabel("|".join(needs))
            req_label.setStyleSheet("color: #f39c12; font-size: 12px;")
            req_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
            req_label.setFixedWidth(45)
            layout.addWidget(req_label)
    
    def set_loaded(self, loaded: bool):
        self.is_loaded = loaded
        self.toggle_btn.blockSignals(True)
        self.toggle_btn.setChecked(loaded)
        self.toggle_btn.setText("ON" if loaded else "OFF")
        self.toggle_btn.blockSignals(False)
        self.is_processing = False
        self.toggle_btn.setEnabled(True)
    
    def set_processing(self, processing: bool):
        """Set processing state - disables toggle during load/unload."""
        self.is_processing = processing
        self.toggle_btn.setEnabled(not processing)
        if processing:
            self.toggle_btn.setText("...")


class ModulesTab(QWidget):
    """Clean module management interface."""
    
    def __init__(self, parent=None, module_manager=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.module_items: dict[str, ModuleListItem] = {}
        self._setup_ui()
        
        # Refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh_status)
        self.refresh_timer.start(5000)
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("Module Manager")
        title.setStyleSheet("font-size: 12px; font-weight: bold;")
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
                font-size: 12px;
            }
        """)
        header_layout.addWidget(self.search_input)
        
        # Dependencies button
        deps_btn = QPushButton("Dependencies")
        deps_btn.setMinimumWidth(90)
        deps_btn.setMinimumHeight(32)
        deps_btn.clicked.connect(self._show_dependency_graph)
        deps_btn.setToolTip("Show module dependency graph")
        deps_btn.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                font-size: 12px;
            }
        """)
        header_layout.addWidget(deps_btn)
        
        # Profiles menu
        profiles_btn = QPushButton("Profiles")
        profiles_btn.setMinimumWidth(90)
        profiles_btn.setMinimumHeight(32)
        profiles_btn.setToolTip("Save or load module profiles")
        profiles_btn.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                font-size: 12px;
            }
        """)
        profiles_btn.clicked.connect(self._show_profiles_menu)
        header_layout.addWidget(profiles_btn)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setMinimumWidth(90)
        refresh_btn.setMinimumHeight(32)
        refresh_btn.clicked.connect(self._refresh_status)
        refresh_btn.setToolTip("Refresh module status and resource usage")
        refresh_btn.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                font-size: 12px;
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
        left_layout.setSpacing(6)
        
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
                font-size: 12px;
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
                font-size: 12px;
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
        
        # Use splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        
        # Right side - Status panel (resizable)
        right_panel = QWidget()
        right_panel.setMinimumWidth(250)
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(8)
        
        # Stats box
        stats_box = QGroupBox("Status")
        stats_layout = QVBoxLayout(stats_box)
        stats_layout.setSpacing(10)
        
        self.loaded_label = QLabel("Loaded: 0 / 0")
        self.loaded_label.setStyleSheet("font-size: 12px; font-weight: bold;")
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
        conn_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        conn_layout.addWidget(conn_title)
        
        self.ai_status_indicator = QLabel("[OFF] Disconnected")
        self.ai_status_indicator.setStyleSheet("color: #ef4444; font-size: 12px;")
        conn_layout.addWidget(self.ai_status_indicator)
        
        self.ai_status_detail = QLabel("No modules loaded")
        self.ai_status_detail.setStyleSheet("color: #bac2de; font-size: 11px;")
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
        
        right_layout.addWidget(log_box, stretch=1)  # Let log box expand
        
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (50/50 split)
        splitter.setSizes([500, 500])
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: #444;
                width: 4px;
            }
            QSplitter::handle:hover {
                background: #666;
            }
        """)
        
        content_layout.addWidget(splitter)
        
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
            
            # Simple direct connection - clicked signal only fires on user clicks
            item.toggle_btn.clicked.connect(
                lambda checked, m=mod_id: self._on_toggle_bool(m, checked)
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
            from enigma_engine.modules.registry import MODULE_REGISTRY
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
        except Exception as e:
            logger.debug(f"Could not load registry modules: {e}")
        
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
    
    def _on_toggle_bool(self, module_id: str, enabled: bool):
        """Handle module toggle (bool version)."""
        # Check if item exists
        if module_id not in self.module_items:
            return
            
        item = self.module_items[module_id]
        
        # Prevent rapid repeated clicks
        if item.is_processing:
            return
        
        # Set processing state
        item.set_processing(True)
        
        action = "Loading" if enabled else "Unloading"
        self._log(f"{action} {module_id}...")
        
        success = False
        if self.module_manager:
            try:
                if enabled:
                    can_load, reason = self.module_manager.can_load(module_id)
                    if not can_load:
                        # Show conflict warning dialog for certain errors
                        if "conflict" in reason.lower() or "provided by" in reason.lower():
                            reply = QMessageBox.warning(
                                self, "Module Conflict",
                                f"Cannot load '{module_id}':\n\n{reason}\n\n"
                                "Would you like to unload the conflicting module first?",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No
                            )
                            if reply == QMessageBox.Yes:
                                # Try to unload the conflicting module
                                conflict_mod = self._extract_conflict_module(reason)
                                if conflict_mod:
                                    self.module_manager.unload(conflict_mod)
                                    self._refresh_status()
                                    can_load, reason = self.module_manager.can_load(module_id)
                        
                        if not can_load:
                            # Use friendly error messages
                            try:
                                from ...modules.error_messages import (
                                    format_error_for_terminal,
                                    get_friendly_error,
                                )
                                error_info = get_friendly_error(module_id, reason)
                                self._log(format_error_for_terminal(error_info).strip())
                            except ImportError:
                                self._log(f"Cannot load {module_id}: {reason}")
                    
                    if can_load:
                        success = self.module_manager.load(module_id)
                else:
                    success = self.module_manager.unload(module_id)
                
                if success:
                    self._sync_options_menu(module_id, enabled)
                    self._sync_tab_visibility(module_id, enabled)
                    try:
                        self.module_manager.save_config()
                    except Exception as e:
                        logger.warning(f"Could not save module config: {e}")
                    self._log(f"OK: {module_id} {'enabled' if enabled else 'disabled'}")
                else:
                    self._log(f"FAILED: Could not {'load' if enabled else 'unload'} {module_id}")
                
            except Exception as e:
                self._log(f"ERROR: {str(e)}")
        
        # Update UI state based on result
        item.toggle_btn.blockSignals(True)
        if success:
            item.set_loaded(enabled)
        else:
            # Revert button to previous state
            item.toggle_btn.setChecked(not enabled)
            item.toggle_btn.setText("ON" if not enabled else "OFF")
            item.set_processing(False)
        item.toggle_btn.blockSignals(False)
        
        self._update_stats()
    
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
                        # Use friendly error messages
                        try:
                            from ...modules.error_messages import (
                                format_error_for_terminal,
                                get_friendly_error,
                            )
                            error_info = get_friendly_error(module_id, reason)
                            self._log(format_error_for_terminal(error_info).strip())
                        except ImportError:
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
        except Exception as e:
            logger.debug(f"Could not sync tab visibility for {module_id}: {e}")
    
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
                
        except Exception as e:
            logger.debug(f"Could not sync options menu for {module_id}: {e}")
    
    def _sync_loaded_modules(self):
        """Sync UI with actually loaded modules."""
        if not self.module_manager:
            return
        
        try:
            loaded = self.module_manager.list_loaded()
            for mod_id in loaded:
                if mod_id in self.module_items:
                    item = self.module_items[mod_id]
                    item.toggle_btn.blockSignals(True)
                    item.set_loaded(True)
                    item.toggle_btn.blockSignals(False)
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
            self.ai_status_indicator.setStyleSheet("color: #22c55e; font-size: 12px; font-weight: bold;")
            self.ai_status_detail.setText("Core AI + generation ready")
        elif core_loaded:
            self.ai_status_indicator.setText("[ON] Connected (Core)")
            self.ai_status_indicator.setStyleSheet("color: #22c55e; font-size: 12px; font-weight: bold;")
            self.ai_status_detail.setText("Chat available, enable gen modules for more")
        elif loaded > 0:
            self.ai_status_indicator.setText("[...] Partial")
            self.ai_status_indicator.setStyleSheet("color: #f59e0b; font-size: 12px; font-weight: bold;")
            self.ai_status_detail.setText(f"{loaded} modules loaded, enable core for chat")
        else:
            self.ai_status_indicator.setText("[OFF] Disconnected")
            self.ai_status_indicator.setStyleSheet("color: #ef4444; font-size: 12px;")
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
            except Exception as e:
                logger.debug(f"Error updating module status: {e}")
        
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
        except Exception as e:
            logger.debug(f"Could not get VRAM usage: {e}")
            self.vram_bar.setValue(0)
    
    def _log(self, message: str):
        """Add to activity log."""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {message}")
    # =========================================================================
    # DEPENDENCY VISUALIZATION
    # =========================================================================
    
    def _show_dependency_graph(self):
        """Show module dependency graph in a dialog."""
        from PyQt5.QtWidgets import QDialog, QTextEdit, QVBoxLayout, QPushButton
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Module Dependencies")
        dialog.setMinimumSize(600, 500)
        dialog.setStyleSheet("""
            QDialog {
                background: #1e1e2e;
                color: #cdd6f4;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        
        text = QTextEdit()
        text.setReadOnly(True)
        text.setFont(QFont('Consolas', 10))
        text.setStyleSheet("""
            QTextEdit {
                background: #1a1a1a;
                color: #cdd6f4;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        
        # Build dependency tree
        dep_text = self._build_dependency_text()
        text.setHtml(dep_text)
        
        layout.addWidget(text)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def _build_dependency_text(self) -> str:
        """Build HTML text showing module dependencies."""
        lines = ["<h3 style='color: #89b4fa;'>Module Dependency Graph</h3>"]
        lines.append("<p style='color: #a6adc8;'>Arrows show dependencies (A -> B means A requires B)</p>")
        lines.append("<hr style='border-color: #444;'/>")
        
        modules = self._get_all_modules()
        
        # Get actual module info from registry if available
        try:
            from ...modules.registry import MODULE_REGISTRY
            
            # Group by category
            categories = {}
            for mod_id, mod_class in MODULE_REGISTRY.items():
                info = mod_class.get_info()
                cat = info.category.value if hasattr(info.category, 'value') else str(info.category)
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append((mod_id, info))
            
            for cat, mods in sorted(categories.items()):
                cat_color = CATEGORIES.get(cat, {'color': '#95a5a6'})['color']
                cat_name = CATEGORIES.get(cat, {'name': cat.title()})['name']
                lines.append(f"<h4 style='color: {cat_color};'>{cat_name}</h4>")
                
                for mod_id, info in sorted(mods, key=lambda x: x[0]):
                    requires = info.requires if info.requires else []
                    conflicts = info.conflicts if info.conflicts else []
                    provides = info.provides if info.provides else []
                    
                    loaded_marker = ""
                    if self.module_manager:
                        try:
                            if mod_id in self.module_manager.list_loaded():
                                loaded_marker = " <span style='color: #22c55e;'>[LOADED]</span>"
                        except Exception:
                            pass
                    
                    lines.append(f"<p style='margin-left: 20px;'><b style='color: #cdd6f4;'>{mod_id}</b>{loaded_marker}</p>")
                    
                    if requires:
                        lines.append(f"<p style='margin-left: 40px; color: #89b4fa;'>Requires: {', '.join(requires)}</p>")
                    if conflicts:
                        lines.append(f"<p style='margin-left: 40px; color: #f38ba8;'>Conflicts: {', '.join(conflicts)}</p>")
                    if provides:
                        lines.append(f"<p style='margin-left: 40px; color: #a6e3a1;'>Provides: {', '.join(provides)}</p>")
                    
        except ImportError:
            lines.append("<p style='color: #fab387;'>Module registry not available. Showing basic module list.</p>")
            for mod_id, mod_info in modules.items():
                lines.append(f"<p style='margin-left: 20px;'>{mod_id}: {mod_info.get('description', '')}</p>")
        
        return "\n".join(lines)

    # =========================================================================
    # MODULE PROFILES SYSTEM
    # =========================================================================
    
    def _show_profiles_menu(self):
        """Show profiles menu with save/load options."""
        from PyQt5.QtWidgets import QMenu
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QMenu::separator {
                height: 1px;
                background: #45475a;
                margin: 4px 8px;
            }
        """)
        
        # Save current profile
        save_action = menu.addAction("Save Current as Profile...")
        save_action.triggered.connect(self._save_profile)
        
        menu.addSeparator()
        
        # Load existing profiles
        profiles = self._get_saved_profiles()
        if profiles:
            for profile_name in profiles:
                load_action = menu.addAction(f"Load: {profile_name}")
                load_action.triggered.connect(lambda checked, p=profile_name: self._load_profile(p))
            
            menu.addSeparator()
            
            # Delete profile submenu
            delete_menu = menu.addMenu("Delete Profile")
            for profile_name in profiles:
                del_action = delete_menu.addAction(profile_name)
                del_action.triggered.connect(lambda checked, p=profile_name: self._delete_profile(p))
        else:
            no_profiles = menu.addAction("No saved profiles")
            no_profiles.setEnabled(False)
        
        menu.addSeparator()
        
        # Preset profiles
        preset_menu = menu.addMenu("Presets")
        
        minimal_action = preset_menu.addAction("Minimal (Text only)")
        minimal_action.triggered.connect(lambda: self._apply_preset('minimal'))
        
        standard_action = preset_menu.addAction("Standard (Chat + Memory)")
        standard_action.triggered.connect(lambda: self._apply_preset('standard'))
        
        creative_action = preset_menu.addAction("Creative (All Generation)")
        creative_action.triggered.connect(lambda: self._apply_preset('creative'))
        
        full_action = preset_menu.addAction("Full (Everything)")
        full_action.triggered.connect(lambda: self._apply_preset('full'))
        
        menu.exec_(self.mapToGlobal(self.sender().pos()))
    
    def _get_profiles_path(self):
        """Get the path to the profiles configuration file."""
        from pathlib import Path
        config_dir = Path.home() / ".enigma_engine"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "module_profiles.json"
    
    def _get_saved_profiles(self) -> list:
        """Get list of saved profile names."""
        import json
        path = self._get_profiles_path()
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    return list(data.get('profiles', {}).keys())
            except Exception:
                pass
        return []
    
    def _save_profile(self):
        """Save current loaded modules as a profile."""
        from PyQt5.QtWidgets import QInputDialog
        import json
        
        name, ok = QInputDialog.getText(
            self, "Save Profile", "Profile name:",
        )
        if ok and name.strip():
            name = name.strip()
            
            # Get currently loaded modules
            loaded = []
            if self.module_manager:
                try:
                    loaded = list(self.module_manager.list_loaded())
                except Exception:
                    pass
            
            # Load existing profiles
            path = self._get_profiles_path()
            data = {'profiles': {}}
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                except Exception:
                    pass
            
            # Save new profile
            data['profiles'][name] = {
                'modules': loaded,
                'created': str(datetime.now())
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self._log(f"Saved profile '{name}' with {len(loaded)} modules")
            QMessageBox.information(self, "Profile Saved", f"Profile '{name}' saved with {len(loaded)} modules.")
    
    def _load_profile(self, profile_name: str):
        """Load a saved profile."""
        import json
        
        path = self._get_profiles_path()
        if not path.exists():
            return
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            profile = data.get('profiles', {}).get(profile_name)
            if not profile:
                return
            
            modules = profile.get('modules', [])
            
            # Check for conflicts before loading
            conflicts = self._check_profile_conflicts(modules)
            if conflicts:
                reply = QMessageBox.warning(
                    self, "Conflicts Detected",
                    f"Loading this profile may cause conflicts:\n\n{conflicts}\n\nContinue anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
            
            # Unload all current modules first
            if self.module_manager:
                try:
                    loaded = list(self.module_manager.list_loaded())
                    for mod_id in loaded:
                        self.module_manager.unload(mod_id)
                except Exception:
                    pass
            
            # Load profile modules in order
            loaded_count = 0
            for mod_id in modules:
                if self.module_manager:
                    try:
                        if self.module_manager.load(mod_id):
                            loaded_count += 1
                    except Exception as e:
                        self._log(f"Failed to load {mod_id}: {e}")
            
            self._refresh_status()
            self._log(f"Loaded profile '{profile_name}': {loaded_count}/{len(modules)} modules")
            
        except Exception as e:
            self._log(f"Error loading profile: {e}")
    
    def _delete_profile(self, profile_name: str):
        """Delete a saved profile."""
        import json
        
        reply = QMessageBox.question(
            self, "Delete Profile",
            f"Delete profile '{profile_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        
        path = self._get_profiles_path()
        if not path.exists():
            return
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            if profile_name in data.get('profiles', {}):
                del data['profiles'][profile_name]
                
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self._log(f"Deleted profile '{profile_name}'")
        except Exception as e:
            self._log(f"Error deleting profile: {e}")
    
    def _apply_preset(self, preset_name: str):
        """Apply a preset module configuration."""
        presets = {
            'minimal': ['tokenizer'],
            'standard': ['model', 'tokenizer', 'inference', 'memory'],
            'creative': ['model', 'tokenizer', 'inference', 'memory', 
                        'image_gen_local', 'code_gen_local', 'audio_gen_local'],
            'full': ['model', 'tokenizer', 'inference', 'training', 'memory',
                    'image_gen_local', 'code_gen_local', 'video_gen_local',
                    'audio_gen_local', 'threed_gen_local', 'voice_input', 
                    'voice_output', 'vision', 'web_tools', 'file_tools']
        }
        
        modules = presets.get(preset_name, [])
        if not modules:
            return
        
        # Confirm
        reply = QMessageBox.question(
            self, "Apply Preset",
            f"This will unload all current modules and load the '{preset_name}' preset.\n\n"
            f"Modules: {', '.join(modules)}\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        
        # Unload all
        if self.module_manager:
            try:
                loaded = list(self.module_manager.list_loaded())
                for mod_id in loaded:
                    self.module_manager.unload(mod_id)
            except Exception:
                pass
        
        # Load preset modules
        loaded_count = 0
        for mod_id in modules:
            if self.module_manager:
                try:
                    if self.module_manager.load(mod_id):
                        loaded_count += 1
                except Exception as e:
                    self._log(f"Failed to load {mod_id}: {e}")
        
        self._refresh_status()
        self._log(f"Applied '{preset_name}' preset: {loaded_count}/{len(modules)} modules")

    # =========================================================================
    # CONFLICT DETECTION
    # =========================================================================
    
    def _check_profile_conflicts(self, modules: list) -> str:
        """Check for conflicts in a list of modules."""
        conflicts = []
        
        try:
            from ...modules.registry import MODULE_REGISTRY
            
            # Build provides map
            provides_map = {}
            for mod_id in modules:
                if mod_id in MODULE_REGISTRY:
                    info = MODULE_REGISTRY[mod_id].get_info()
                    for capability in info.provides:
                        if capability in provides_map:
                            conflicts.append(
                                f"'{mod_id}' and '{provides_map[capability]}' "
                                f"both provide '{capability}'"
                            )
                        else:
                            provides_map[capability] = mod_id
                    
                    # Check explicit conflicts
                    for conflict_id in info.conflicts:
                        if conflict_id in modules:
                            conflicts.append(f"'{mod_id}' conflicts with '{conflict_id}'")
        except ImportError:
            pass
        
        return "\n".join(conflicts) if conflicts else ""
    
    def _extract_conflict_module(self, reason: str) -> str:
        """Extract the conflicting module name from an error message."""
        import re
        
        # Match patterns like "conflicts with loaded module 'xxx'" or "provided by 'xxx'"
        patterns = [
            r"conflicts with(?: loaded module)? '(\w+)'",
            r"provided by '(\w+)'",
            r"already provided by '(\w+)'",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, reason)
            if match:
                return match.group(1)
        
        return ""
