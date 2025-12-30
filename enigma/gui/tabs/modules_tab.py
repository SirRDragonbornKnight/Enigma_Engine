"""
Module Manager Tab - Control all Enigma capabilities
"""
import sys
from typing import Dict, Optional

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea,
        QLabel, QPushButton, QFrame, QGroupBox, QCheckBox,
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit,
        QProgressBar, QMessageBox, QSplitter, QTextEdit
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtGui import QFont, QColor, QPalette
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


class ModuleCard(QFrame):
    """Visual card for a single module."""
    
    toggled = None  # Will be pyqtSignal
    configure_clicked = None  # Will be pyqtSignal
    
    def __init__(self, module_name: str, module_info: dict, parent=None):
        super().__init__(parent)
        self.module_name = module_name
        self.module_info = module_info
        
        self.setup_ui()
        
    def setup_ui(self):
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Header with name and toggle
        header = QHBoxLayout()
        
        # Module name
        name_label = QLabel(self.module_info.get('name', self.module_name))
        name_label.setFont(QFont('Arial', 11, QFont.Bold))
        header.addWidget(name_label)
        
        header.addStretch()
        
        # Category badge
        category = self.module_info.get('category', 'unknown')
        cat_label = QLabel(f"[{category}]")
        cat_label.setStyleSheet(self._get_category_style(category))
        header.addWidget(cat_label)
        
        # Toggle checkbox
        self.toggle = QCheckBox("Enabled")
        self.toggle.setChecked(self.module_info.get('loaded', False))
        header.addWidget(self.toggle)
        
        layout.addLayout(header)
        
        # Description
        desc = self.module_info.get('description', 'No description available.')
        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(desc_label)
        
        # Requirements
        reqs = self.module_info.get('requirements', [])
        if reqs:
            req_text = f"Requires: {', '.join(reqs)}"
            req_label = QLabel(req_text)
            req_label.setStyleSheet("color: #888; font-size: 9px; font-style: italic;")
            layout.addWidget(req_label)
        
        # Status bar
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Not loaded")
        self.status_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        # Configure button
        self.config_btn = QPushButton("Configure")
        self.config_btn.setEnabled(self.module_info.get('has_config', False))
        status_layout.addWidget(self.config_btn)
        
        layout.addLayout(status_layout)
        
        # Resource usage (if loaded)
        self.resource_bar = QProgressBar()
        self.resource_bar.setMaximum(100)
        self.resource_bar.setValue(0)
        self.resource_bar.setTextVisible(True)
        self.resource_bar.setFormat("Resources: %p%")
        self.resource_bar.setVisible(False)
        layout.addWidget(self.resource_bar)
        
    def _get_category_style(self, category: str) -> str:
        colors = {
            'core': '#e74c3c',
            'memory': '#3498db',
            'interface': '#2ecc71',
            'perception': '#9b59b6',
            'output': '#f39c12',
            'tools': '#1abc9c',
            'network': '#e67e22',
            'extension': '#95a5a6',
        }
        color = colors.get(category.lower(), '#95a5a6')
        return f"background-color: {color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 9px;"
    
    def set_status(self, status: str, is_loaded: bool):
        self.status_label.setText(status)
        if is_loaded:
            self.status_label.setStyleSheet("color: #27ae60;")
            self.resource_bar.setVisible(True)
        else:
            self.status_label.setStyleSheet("color: #666;")
            self.resource_bar.setVisible(False)
    
    def set_resource_usage(self, percent: int):
        self.resource_bar.setValue(percent)


class ModulesTab(QWidget):
    """Tab for managing all Enigma modules."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.module_cards: Dict[str, ModuleCard] = {}
        self.setup_ui()
        
        # Refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_status)
        self.refresh_timer.start(5000)  # Every 5 seconds
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title and controls
        header = QHBoxLayout()
        
        title = QLabel("Module Manager")
        title.setFont(QFont('Arial', 16, QFont.Bold))
        header.addWidget(title)
        
        header.addStretch()
        
        # Filter by category
        header.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(['All', 'Core', 'Memory', 'Interface', 'Perception', 'Output', 'Tools', 'Network', 'Extension'])
        self.filter_combo.currentTextChanged.connect(self.filter_modules)
        header.addWidget(self.filter_combo)
        
        # Quick actions
        self.load_all_btn = QPushButton("Load All")
        self.load_all_btn.clicked.connect(self.load_all)
        header.addWidget(self.load_all_btn)
        
        self.unload_all_btn = QPushButton("Unload All")
        self.unload_all_btn.clicked.connect(self.unload_all)
        header.addWidget(self.unload_all_btn)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_status)
        header.addWidget(self.refresh_btn)
        
        layout.addLayout(header)
        
        # Main content - splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Module grid
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        
        scroll.setWidget(self.grid_widget)
        left_layout.addWidget(scroll)
        
        splitter.addWidget(left_widget)
        
        # Right: Details/Log panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # System status
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout(status_group)
        
        self.cpu_bar = self._create_resource_bar("CPU")
        status_layout.addWidget(self.cpu_bar)
        
        self.mem_bar = self._create_resource_bar("Memory")
        status_layout.addWidget(self.mem_bar)
        
        self.gpu_bar = self._create_resource_bar("GPU")
        status_layout.addWidget(self.gpu_bar)
        
        self.vram_bar = self._create_resource_bar("VRAM")
        status_layout.addWidget(self.vram_bar)
        
        right_layout.addWidget(status_group)
        
        # Log
        log_group = QGroupBox("Module Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont('Consolas', 9))
        log_layout.addWidget(self.log_text)
        
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_text.clear)
        log_layout.addWidget(clear_btn)
        
        right_layout.addWidget(log_group)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 300])
        
        layout.addWidget(splitter)
        
        # Populate with modules
        self.populate_modules()
        
    def _create_resource_bar(self, name: str) -> QProgressBar:
        bar = QProgressBar()
        bar.setMaximum(100)
        bar.setValue(0)
        bar.setTextVisible(True)
        bar.setFormat(f"{name}: %p%")
        return bar
    
    def populate_modules(self):
        """Populate the grid with module cards."""
        # Clear existing
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.module_cards.clear()
        
        # Get modules from registry
        modules = self._get_all_modules()
        
        # Add cards in grid (3 columns)
        row, col = 0, 0
        for name, info in modules.items():
            card = ModuleCard(name, info)
            card.toggle.stateChanged.connect(lambda state, n=name: self._on_toggle(n, state))
            card.config_btn.clicked.connect(lambda _, n=name: self._on_configure(n))
            
            self.grid_layout.addWidget(card, row, col)
            self.module_cards[name] = card
            
            col += 1
            if col >= 3:
                col = 0
                row += 1
        
        # Add stretch at bottom
        self.grid_layout.setRowStretch(row + 1, 1)
        
    def _get_all_modules(self) -> dict:
        """Get all available modules."""
        # This would connect to the actual module registry
        # For now, return a representative list
        return {
            'model': {
                'name': 'Model Engine',
                'category': 'core',
                'description': 'Core transformer model for inference and training. Supports sizes from nano to omega.',
                'requirements': ['torch'],
                'has_config': True,
                'loaded': False,
            },
            'tokenizer': {
                'name': 'Tokenizer',
                'category': 'core',
                'description': 'BPE tokenizer with vocabulary management and encoding/decoding.',
                'requirements': [],
                'has_config': True,
                'loaded': False,
            },
            'training': {
                'name': 'Training Engine',
                'category': 'core',
                'description': 'Model training with AMP, gradient accumulation, and distributed support.',
                'requirements': ['torch', 'model', 'tokenizer'],
                'has_config': True,
                'loaded': False,
            },
            'inference': {
                'name': 'Inference Engine',
                'category': 'core',
                'description': 'Text generation with streaming, chat mode, and KV-cache.',
                'requirements': ['torch', 'model', 'tokenizer'],
                'has_config': True,
                'loaded': False,
            },
            'memory': {
                'name': 'Memory Manager',
                'category': 'memory',
                'description': 'Conversation storage, retrieval, and context management.',
                'requirements': [],
                'has_config': True,
                'loaded': False,
            },
            'vector_memory': {
                'name': 'Vector Memory',
                'category': 'memory',
                'description': 'Semantic search using embeddings for relevant context retrieval.',
                'requirements': ['torch', 'memory'],
                'has_config': True,
                'loaded': False,
            },
            'voice_input': {
                'name': 'Voice Input (STT)',
                'category': 'interface',
                'description': 'Speech-to-text using various backends (Whisper, Google, etc.).',
                'requirements': [],
                'has_config': True,
                'loaded': False,
            },
            'voice_output': {
                'name': 'Voice Output (TTS)',
                'category': 'output',
                'description': 'Text-to-speech with multiple voice profiles and engines.',
                'requirements': [],
                'has_config': True,
                'loaded': False,
            },
            'vision': {
                'name': 'Vision System',
                'category': 'perception',
                'description': 'Image processing, OCR, and visual understanding.',
                'requirements': ['torch', 'PIL'],
                'has_config': True,
                'loaded': False,
            },
            'avatar': {
                'name': 'Avatar System',
                'category': 'output',
                'description': 'Visual avatar representation with expressions and animations.',
                'requirements': [],
                'has_config': True,
                'loaded': False,
            },
            'web_tools': {
                'name': 'Web Tools',
                'category': 'tools',
                'description': 'Web search, page fetching, and URL processing.',
                'requirements': ['requests'],
                'has_config': False,
                'loaded': False,
            },
            'file_tools': {
                'name': 'File Tools',
                'category': 'tools',
                'description': 'File operations, reading, writing, and management.',
                'requirements': [],
                'has_config': False,
                'loaded': False,
            },
            'document_tools': {
                'name': 'Document Tools',
                'category': 'tools',
                'description': 'PDF, Office, and document processing capabilities.',
                'requirements': [],
                'has_config': False,
                'loaded': False,
            },
            'api_server': {
                'name': 'API Server',
                'category': 'network',
                'description': 'REST API for external access to Enigma capabilities.',
                'requirements': ['flask', 'inference'],
                'has_config': True,
                'loaded': False,
            },
            'network': {
                'name': 'Multi-Device Network',
                'category': 'network',
                'description': 'Distributed computing across multiple Enigma instances.',
                'requirements': [],
                'has_config': True,
                'loaded': False,
            },
            'protocols': {
                'name': 'Protocol Engine',
                'category': 'extension',
                'description': 'Custom behavior protocols for robots, games, and APIs.',
                'requirements': [],
                'has_config': True,
                'loaded': False,
            },
        }
    
    def _on_toggle(self, module_name: str, state: int):
        """Handle module toggle."""
        enabled = state == Qt.Checked
        if enabled:
            self.log(f"Loading module: {module_name}...")
            # TODO: Actually load the module
            self.module_cards[module_name].set_status("Loaded", True)
            self.log(f"✓ Module {module_name} loaded")
        else:
            self.log(f"Unloading module: {module_name}...")
            # TODO: Actually unload the module
            self.module_cards[module_name].set_status("Not loaded", False)
            self.log(f"✓ Module {module_name} unloaded")
    
    def _on_configure(self, module_name: str):
        """Show configuration dialog for a module."""
        QMessageBox.information(
            self, 
            f"Configure {module_name}",
            f"Configuration dialog for {module_name} coming soon.\n\n"
            "This will show all configurable options for this module."
        )
    
    def filter_modules(self, category: str):
        """Filter modules by category."""
        for name, card in self.module_cards.items():
            if category == 'All':
                card.setVisible(True)
            else:
                card.setVisible(card.module_info.get('category', '').lower() == category.lower())
    
    def load_all(self):
        """Load all modules."""
        self.log("Loading all modules...")
        for name, card in self.module_cards.items():
            card.toggle.setChecked(True)
        self.log("✓ All modules loaded")
    
    def unload_all(self):
        """Unload all modules."""
        self.log("Unloading all modules...")
        for name, card in self.module_cards.items():
            card.toggle.setChecked(False)
        self.log("✓ All modules unloaded")
    
    def refresh_status(self):
        """Refresh module and system status."""
        try:
            import psutil
            self.cpu_bar.setValue(int(psutil.cpu_percent()))
            self.mem_bar.setValue(int(psutil.virtual_memory().percent))
            
            # GPU stats if available
            try:
                import torch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_reserved() * 100 if torch.cuda.max_memory_reserved() > 0 else 0
                    self.gpu_bar.setValue(50)  # Would need pynvml for actual usage
                    self.vram_bar.setValue(int(allocated))
                else:
                    self.gpu_bar.setValue(0)
                    self.vram_bar.setValue(0)
            except ImportError:
                self.gpu_bar.setValue(0)
                self.vram_bar.setValue(0)
                
        except ImportError:
            pass
    
    def log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)


if not HAS_PYQT:
    # Stub class for when PyQt5 is not available
    class ModulesTab:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt5 is required for the Module Manager GUI")
