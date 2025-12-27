"""
Enhanced PyQt5 GUI for Enigma with Setup Wizard

Features:
  - First-run setup wizard to create/name your AI
  - Model selection and management
  - Backup before risky operations
  - Grow/shrink models with confirmation
  - Chat, Training, Voice integration
  - Dark/Light/Shadow/Midnight mode toggle
  - Avatar control panel
  - Screen vision preview with camera support
  - Training data editor
  - Per-AI conversation history
  - Multi-AI support (run multiple models)
  - Image upload in chat/vision tabs
  - Selectable (read-only) text throughout
"""
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLineEdit, QLabel, QListWidget, QTabWidget, QFileDialog, QMessageBox,
    QDialog, QComboBox, QProgressBar, QGroupBox, QRadioButton, QButtonGroup,
    QSpinBox, QCheckBox, QDialogButtonBox, QWizard, QWizardPage, QFormLayout,
    QSlider, QSplitter, QPlainTextEdit, QToolTip, QFrame, QScrollArea, QInputDialog,
    QListWidgetItem, QActionGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QImage, QTextCursor
import time


# === THEME STYLESHEETS ===
# Theme: Dark (Catppuccin Mocha)
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QTextEdit, QPlainTextEdit, QLineEdit, QListWidget {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px;
}
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #b4befe;
}
QPushButton:pressed {
    background-color: #74c7ec;
}
QPushButton:disabled {
    background-color: #45475a;
    color: #6c7086;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
}
QGroupBox::title {
    color: #89b4fa;
    subcontrol-origin: margin;
    left: 10px;
}
QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #313244;
    color: #cdd6f4;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QProgressBar {
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #a6e3a1;
}
QMenuBar {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QMenuBar::item:selected {
    background-color: #313244;
}
QMenu {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
}
QMenu::item:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QSpinBox, QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px;
}
QSlider::groove:horizontal {
    background: #45475a;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QScrollBar:vertical {
    background: #313244;
    width: 12px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 6px;
}
QLabel#header {
    font-size: 16px;
    font-weight: bold;
    color: #89b4fa;
}
QLabel {
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
}
"""

LIGHT_STYLE = """
QMainWindow, QWidget {
    background-color: #eff1f5;
    color: #4c4f69;
}
QTextEdit, QPlainTextEdit, QLineEdit, QListWidget {
    background-color: #ffffff;
    color: #4c4f69;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 4px;
}
QPushButton {
    background-color: #1e66f5;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #7287fd;
}
QPushButton:pressed {
    background-color: #04a5e5;
}
QPushButton:disabled {
    background-color: #ccd0da;
    color: #9ca0b0;
}
QGroupBox {
    border: 1px solid #ccd0da;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
}
QGroupBox::title {
    color: #1e66f5;
    subcontrol-origin: margin;
    left: 10px;
}
QTabWidget::pane {
    border: 1px solid #ccd0da;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #e6e9ef;
    color: #4c4f69;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #1e66f5;
    color: #ffffff;
}
QProgressBar {
    border: 1px solid #ccd0da;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #40a02b;
}
QMenuBar {
    background-color: #eff1f5;
    color: #4c4f69;
}
QMenuBar::item:selected {
    background-color: #e6e9ef;
}
QMenu {
    background-color: #ffffff;
    color: #4c4f69;
    border: 1px solid #ccd0da;
}
QMenu::item:selected {
    background-color: #1e66f5;
    color: #ffffff;
}
QSpinBox, QComboBox {
    background-color: #ffffff;
    color: #4c4f69;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 4px;
}
QSlider::groove:horizontal {
    background: #ccd0da;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #1e66f5;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QLabel#header {
    font-size: 16px;
    font-weight: bold;
    color: #1e66f5;
}
QLabel {
    selection-background-color: #1e66f5;
    selection-color: #ffffff;
}
"""

# Theme: Shadow (Very dark with purple accents)
SHADOW_STYLE = """
QMainWindow, QWidget {
    background-color: #0d0d0d;
    color: #b8b8b8;
}
QTextEdit, QPlainTextEdit, QLineEdit, QListWidget {
    background-color: #1a1a1a;
    color: #d0d0d0;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 4px;
}
QPushButton {
    background-color: #6b21a8;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #7c3aed;
}
QPushButton:pressed {
    background-color: #581c87;
}
QPushButton:disabled {
    background-color: #2a2a2a;
    color: #4a4a4a;
}
QGroupBox {
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
}
QGroupBox::title {
    color: #9333ea;
    subcontrol-origin: margin;
    left: 10px;
}
QTabWidget::pane {
    border: 1px solid #2a2a2a;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #1a1a1a;
    color: #b8b8b8;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #6b21a8;
    color: #ffffff;
}
QProgressBar {
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #9333ea;
}
QMenuBar {
    background-color: #0d0d0d;
    color: #b8b8b8;
}
QMenuBar::item:selected {
    background-color: #1a1a1a;
}
QMenu {
    background-color: #1a1a1a;
    color: #b8b8b8;
    border: 1px solid #2a2a2a;
}
QMenu::item:selected {
    background-color: #6b21a8;
    color: #ffffff;
}
QSpinBox, QComboBox {
    background-color: #1a1a1a;
    color: #d0d0d0;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 4px;
}
QSlider::groove:horizontal {
    background: #2a2a2a;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #9333ea;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QScrollBar:vertical {
    background: #1a1a1a;
    width: 12px;
}
QScrollBar::handle:vertical {
    background: #2a2a2a;
    border-radius: 6px;
}
QLabel#header {
    font-size: 16px;
    font-weight: bold;
    color: #9333ea;
}
QLabel {
    selection-background-color: #6b21a8;
    selection-color: #ffffff;
}
"""

# Theme: Midnight (Deep blue/black with cyan accents)
MIDNIGHT_STYLE = """
QMainWindow, QWidget {
    background-color: #030712;
    color: #e2e8f0;
}
QTextEdit, QPlainTextEdit, QLineEdit, QListWidget {
    background-color: #0f172a;
    color: #e2e8f0;
    border: 1px solid #1e293b;
    border-radius: 4px;
    padding: 4px;
}
QPushButton {
    background-color: #0891b2;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #22d3ee;
}
QPushButton:pressed {
    background-color: #0e7490;
}
QPushButton:disabled {
    background-color: #1e293b;
    color: #475569;
}
QGroupBox {
    border: 1px solid #1e293b;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
}
QGroupBox::title {
    color: #22d3ee;
    subcontrol-origin: margin;
    left: 10px;
}
QTabWidget::pane {
    border: 1px solid #1e293b;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #0f172a;
    color: #e2e8f0;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #0891b2;
    color: #ffffff;
}
QProgressBar {
    border: 1px solid #1e293b;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #22d3ee;
}
QMenuBar {
    background-color: #030712;
    color: #e2e8f0;
}
QMenuBar::item:selected {
    background-color: #0f172a;
}
QMenu {
    background-color: #0f172a;
    color: #e2e8f0;
    border: 1px solid #1e293b;
}
QMenu::item:selected {
    background-color: #0891b2;
    color: #ffffff;
}
QSpinBox, QComboBox {
    background-color: #0f172a;
    color: #e2e8f0;
    border: 1px solid #1e293b;
    border-radius: 4px;
    padding: 4px;
}
QSlider::groove:horizontal {
    background: #1e293b;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #22d3ee;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QScrollBar:vertical {
    background: #0f172a;
    width: 12px;
}
QScrollBar::handle:vertical {
    background: #1e293b;
    border-radius: 6px;
}
QLabel#header {
    font-size: 16px;
    font-weight: bold;
    color: #22d3ee;
}
QLabel {
    selection-background-color: #0891b2;
    selection-color: #ffffff;
}
"""

# Theme dictionary for easy access
THEMES = {
    "dark": DARK_STYLE,
    "light": LIGHT_STYLE,
    "shadow": SHADOW_STYLE,
    "midnight": MIDNIGHT_STYLE,
}

# Import enigma modules
try:
    from ..core.model_registry import ModelRegistry
    from ..core.model_config import MODEL_PRESETS, get_model_config
    from ..core.model_scaling import grow_model, shrink_model
    from ..core.inference import EnigmaEngine
    from ..memory.manager import ConversationManager
    from ..config import CONFIG
except ImportError:
    # Running standalone
    pass


class SetupWizard(QWizard):
    """First-run setup wizard for creating a new AI."""
    
    def __init__(self, registry: ModelRegistry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.setWindowTitle("Enigma Setup Wizard")
        self.setWizardStyle(QWizard.ModernStyle)
        self.resize(600, 450)
        
        # Detect hardware BEFORE creating pages
        self.hw_profile = self._detect_hardware()
        
        # Add pages
        self.addPage(self._create_welcome_page())
        self.addPage(self._create_name_page())
        self.addPage(self._create_size_page())
        self.addPage(self._create_confirm_page())
        
        self.model_name = None
        self.model_size = self.hw_profile.get("recommended", "tiny")
    
    def _detect_hardware(self) -> dict:
        """Detect hardware capabilities for model size recommendations."""
        try:
            from ..core.hardware import HardwareProfile
            hw = HardwareProfile()
            profile = hw.profile
            
            ram_gb = profile.get("memory", {}).get("total_gb", 2)
            available_gb = profile.get("memory", {}).get("available_gb", 1)
            vram_gb = profile.get("gpu", {}).get("vram_gb", 0)
            is_pi = profile.get("platform", {}).get("is_raspberry_pi", False)
            is_mobile = profile.get("platform", {}).get("is_mobile", False)
            has_gpu = profile.get("gpu", {}).get("cuda_available", False)
            device_type = "Raspberry Pi" if is_pi else ("Mobile" if is_mobile else "PC")
            
            # Determine max safe size based on available memory
            effective_mem = vram_gb if has_gpu else min(available_gb, ram_gb * 0.4)
            
            if effective_mem < 1:
                max_size = "tiny"
                recommended = "tiny"
            elif effective_mem < 2:
                max_size = "small"
                recommended = "tiny"
            elif effective_mem < 4:
                max_size = "medium"
                recommended = "small"
            elif effective_mem < 8:
                max_size = "large"
                recommended = "medium"
            else:
                max_size = "xl"
                recommended = "large"
            
            # Force tiny for Pi/mobile
            if is_pi or is_mobile:
                max_size = "small"
                recommended = "tiny"
            
            return {
                "ram_gb": ram_gb,
                "available_gb": available_gb,
                "vram_gb": vram_gb,
                "has_gpu": has_gpu,
                "is_pi": is_pi,
                "is_mobile": is_mobile,
                "device_type": device_type,
                "max_size": max_size,
                "recommended": recommended,
                "effective_mem": effective_mem,
            }
        except Exception as e:
            # Fallback: assume limited hardware
            return {
                "ram_gb": 2,
                "available_gb": 1,
                "vram_gb": 0,
                "has_gpu": False,
                "is_pi": True,
                "is_mobile": False,
                "device_type": "Unknown",
                "max_size": "small",
                "recommended": "tiny",
                "effective_mem": 1,
                "error": str(e),
            }
    
    def _create_welcome_page(self):
        page = QWizardPage()
        page.setTitle("Welcome to Enigma")
        page.setSubTitle("Let's set up your AI")
        
        layout = QVBoxLayout()
        
        welcome_text = QLabel("""
        <h3>Welcome!</h3>
        <p>This wizard will help you create your first AI model.</p>
        <p>Your AI starts as a <b>blank slate</b> - it will learn only from 
        the data you train it on. No pre-programmed emotions or personality.</p>
        <p><b>What you'll do:</b></p>
        <ul>
            <li>Give your AI a name</li>
            <li>Choose a model size based on your hardware</li>
            <li>Create the initial model (ready for training)</li>
        </ul>
        <p>Click <b>Next</b> to begin.</p>
        """)
        welcome_text.setWordWrap(True)
        layout.addWidget(welcome_text)
        
        page.setLayout(layout)
        return page
    
    def _create_name_page(self):
        page = QWizardPage()
        page.setTitle("Name Your AI")
        page.setSubTitle("Choose a unique name for this model")
        
        layout = QFormLayout()
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., artemis, apollo, atlas...")
        self.name_input.textChanged.connect(self._validate_name)
        
        self.name_status = QLabel("")
        
        layout.addRow("AI Name:", self.name_input)
        layout.addRow("", self.name_status)
        
        description_label = QLabel("""
        <p><b>Tips:</b></p>
        <ul>
            <li>Use lowercase letters, numbers, underscores</li>
            <li>Each AI gets its own folder in models/</li>
            <li>You can create multiple AIs with different names</li>
        </ul>
        """)
        description_label.setWordWrap(True)
        layout.addRow(description_label)
        
        # Register field for validation
        page.registerField("model_name*", self.name_input)
        
        page.setLayout(layout)
        return page
    
    def _validate_name(self, text):
        name = text.lower().strip().replace(" ", "_")
        if not name:
            self.name_status.setText("")
        elif name in self.registry.registry.get("models", {}):
            self.name_status.setText("[!] Name already exists!")
            self.name_status.setStyleSheet("color: orange")
        elif not name.replace("_", "").isalnum():
            self.name_status.setText("[X] Use only letters, numbers, underscores")
            self.name_status.setStyleSheet("color: red")
        else:
            self.name_status.setText("[OK] Name available")
            self.name_status.setStyleSheet("color: green")
    
    def _create_size_page(self):
        page = QWizardPage()
        page.setTitle("Choose Model Size")
        page.setSubTitle("Select based on your hardware")
        
        layout = QVBoxLayout()
        
        self.size_group = QButtonGroup()
        
        # Size definitions with memory requirements
        sizes = [
            ("tiny", "Tiny (~0.5M params)", "Any device", "<1GB", 0.5),
            ("small", "Small (~10M params)", "4GB+ RAM", "2GB", 2),
            ("medium", "Medium (~50M params)", "8GB+ RAM or GPU", "4GB", 4),
            ("large", "Large (~150M params)", "GPU with 8GB+ VRAM", "8GB", 8),
        ]
        
        # Size order for comparison
        size_order = ["tiny", "small", "medium", "large", "xl"]
        max_idx = size_order.index(self.hw_profile.get("max_size", "tiny"))
        recommended = self.hw_profile.get("recommended", "tiny")
        
        for i, (size_id, name, hw, mem, req_gb) in enumerate(sizes):
            can_use = size_order.index(size_id) <= max_idx
            is_recommended = (size_id == recommended)
            
            label = f"{name}\n    {hw} | Needs: {mem}"
            if is_recommended:
                label += " [!] RECOMMENDED"
            if not can_use:
                label += " [!] TOO LARGE"
            
            radio = QRadioButton(label)
            radio.size_id = size_id
            radio.setEnabled(can_use)
            self.size_group.addButton(radio, i)
            layout.addWidget(radio)
            
            if is_recommended and can_use:
                radio.setChecked(True)
        
        # If nothing checked, check tiny
        if not self.size_group.checkedButton():
            for btn in self.size_group.buttons():
                if btn.size_id == "tiny":
                    btn.setChecked(True)
                    break
        
        # Hardware info from detection
        hw = self.hw_profile
        hw_text = f"""
        <hr>
        <p><b>Your Hardware:</b> {hw.get('device_type', 'Unknown')}</p>
        <ul>
            <li>RAM: {hw.get('ram_gb', '?')} GB (available: {hw.get('available_gb', '?')} GB)</li>
            <li>GPU VRAM: {hw.get('vram_gb', 0)} GB {'[OK]' if hw.get('has_gpu') else '(no GPU)'}</li>
            <li>Effective memory for models: ~{hw.get('effective_mem', 1):.1f} GB</li>
        </ul>
        <p><b>Recommendation:</b> <span style="color: green;">{recommended.upper()}</span></p>
        <p><i>You can grow your model later with better hardware!</i></p>
        """
        note = QLabel(hw_text)
        note.setWordWrap(True)
        layout.addWidget(note)
        
        page.setLayout(layout)
        return page
    
    def _create_confirm_page(self):
        page = QWizardPage()
        page.setTitle("Confirm Setup")
        page.setSubTitle("Review your choices")
        
        layout = QVBoxLayout()
        
        self.confirm_label = QLabel()
        self.confirm_label.setWordWrap(True)
        layout.addWidget(self.confirm_label)
        
        page.setLayout(layout)
        return page
    
    def initializePage(self, page_id):
        """Called when a page is shown."""
        if page_id == 3:  # Confirm page
            name = self.name_input.text().lower().strip().replace(" ", "_")
            
            checked = self.size_group.checkedButton()
            size = checked.size_id if checked else "small"
            
            config = MODEL_PRESETS.get(size, {})
            
            self.confirm_label.setText(f"""
            <h3>Ready to Create Your AI</h3>
            <table>
                <tr><td><b>Name:</b></td><td>{name}</td></tr>
                <tr><td><b>Size:</b></td><td>{size}</td></tr>
                <tr><td><b>Dimensions:</b></td><td>{config.get('dim', '?')}</td></tr>
                <tr><td><b>Layers:</b></td><td>{config.get('depth', '?')}</td></tr>
                <tr><td><b>Min VRAM:</b></td><td>{config.get('min_vram_gb', '?')} GB</td></tr>
            </table>
            <br>
            <p>Click <b>Finish</b> to create your AI.</p>
            <p>The model will be saved in: <code>models/{name}/</code></p>
            """)
            
            self.model_name = name
            self.model_size = size
    
    def get_result(self):
        """Get the wizard result."""
        return {
            "name": self.model_name,
            "size": self.model_size,
        }


class ModelManagerDialog(QDialog):
    """Dialog for managing models - grow, shrink, backup, delete."""
    
    def __init__(self, registry: ModelRegistry, current_model: str = None, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.current_model = current_model
        
        self.setWindowTitle("Model Manager")
        self.resize(500, 400)
        self._build_ui()
        self._refresh_list()
    
    def _build_ui(self):
        layout = QVBoxLayout()
        
        # Model list
        layout.addWidget(QLabel("Registered Models:"))
        self.model_list = QListWidget()
        self.model_list.itemClicked.connect(self._on_select_model)
        layout.addWidget(self.model_list)
        
        # Info display
        self.info_label = QLabel("Select a model to see details")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_new = QPushButton("New Model")
        self.btn_new.clicked.connect(self._on_new_model)
        
        self.btn_backup = QPushButton("Backup")
        self.btn_backup.clicked.connect(self._on_backup)
        
        self.btn_grow = QPushButton("Grow >>")
        self.btn_grow.clicked.connect(self._on_grow)
        
        self.btn_shrink = QPushButton("<< Shrink")
        self.btn_shrink.clicked.connect(self._on_shrink)
        
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._on_delete)
        self.btn_delete.setStyleSheet("color: red")
        
        btn_layout.addWidget(self.btn_new)
        btn_layout.addWidget(self.btn_backup)
        btn_layout.addWidget(self.btn_grow)
        btn_layout.addWidget(self.btn_shrink)
        btn_layout.addWidget(self.btn_delete)
        
        layout.addLayout(btn_layout)
        
        # Load button
        self.btn_load = QPushButton("Load Selected Model")
        self.btn_load.clicked.connect(self.accept)
        layout.addWidget(self.btn_load)
        
        self.setLayout(layout)
    
    def _refresh_list(self):
        self.model_list.clear()
        for name, info in self.registry.registry.get("models", {}).items():
            status = "[+]" if info.get("has_weights") else "[ ]"
            self.model_list.addItem(f"{status} {name} ({info.get('size', '?')})")
    
    def _on_select_model(self, item):
        text = item.text()
        # Extract name from "[+] name (size)"
        name = text.split()[1]
        
        try:
            info = self.registry.get_model_info(name)
            meta = info.get("metadata", {})
            config = info.get("config", {})
            
            self.info_label.setText(f"""
            <b>{name}</b><br>
            Size: {info['registry'].get('size', '?')}<br>
            Created: {meta.get('created', '?')[:10]}<br>
            Last trained: {meta.get('last_trained', 'Never')}<br>
            Epochs: {meta.get('total_epochs', 0)}<br>
            Parameters: {meta.get('estimated_parameters', '?'):,}<br>
            Checkpoints: {len(info.get('checkpoints', []))}
            """)
            
            self.selected_model = name
        except Exception as e:
            self.info_label.setText(f"Error loading info: {e}")
    
    def _on_new_model(self):
        wizard = SetupWizard(self.registry, self)
        if wizard.exec_() == QWizard.Accepted:
            result = wizard.get_result()
            try:
                self.registry.create_model(
                    result["name"],
                    size=result["size"],
                    vocab_size=32000
                )
                self._refresh_list()
                QMessageBox.information(self, "Success", f"Created model '{result['name']}'")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
    
    def _on_backup(self):
        if not hasattr(self, 'selected_model'):
            QMessageBox.warning(self, "No Selection", "Select a model first")
            return
        
        name = self.selected_model
        model_dir = Path(self.registry.models_dir) / name
        backup_dir = Path(self.registry.models_dir) / f"{name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            shutil.copytree(model_dir, backup_dir)
            QMessageBox.information(self, "Backup Complete", f"Backed up to:\n{backup_dir}")
            self._refresh_list()
        except Exception as e:
            QMessageBox.warning(self, "Backup Failed", str(e))
    
    def _on_grow(self):
        if not hasattr(self, 'selected_model'):
            QMessageBox.warning(self, "No Selection", "Select a model first")
            return
        
        # Check if model still exists
        if self.selected_model not in self.registry.registry.get("models", {}):
            QMessageBox.warning(self, "Model Not Found", "Selected model no longer exists. Please select another.")
            delattr(self, 'selected_model')
            self._refresh_list()
            return
        
        # Show size selection dialog
        sizes = ["small", "medium", "large", "xl"]
        current_size = self.registry.registry["models"][self.selected_model].get("size", "tiny")
        
        # Filter to only larger sizes
        current_idx = sizes.index(current_size) if current_size in sizes else -1
        available = sizes[current_idx + 1:] if current_idx >= 0 else sizes
        
        if not available:
            QMessageBox.information(self, "Max Size", "Model is already at maximum size")
            return
        
        size, ok = self._show_size_dialog("Grow Model", available, 
            f"Current size: {current_size}\nSelect target size:")
        
        if ok and size:
            # Confirm with backup warning
            reply = QMessageBox.question(
                self, "Confirm Grow",
                f"Grow '{self.selected_model}' from {current_size} to {size}?\n\n"
                "A backup will be created automatically.\n"
                "The grown model will keep existing knowledge.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Auto-backup first
                self._on_backup()
                
                # Grow - replace the model in place
                try:
                    from ..core.model_scaling import grow_registered_model
                    grow_registered_model(
                        self.registry,
                        self.selected_model,
                        self.selected_model,  # Keep same name
                        size
                    )
                    self._refresh_list()
                    QMessageBox.information(self, "Success", 
                        f"Model '{self.selected_model}' grown to {size}!\n"
                        f"A backup was created.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", str(e))
    
    def _on_shrink(self):
        if not hasattr(self, 'selected_model'):
            QMessageBox.warning(self, "No Selection", "Select a model first")
            return
        
        # Check if model still exists
        if self.selected_model not in self.registry.registry.get("models", {}):
            QMessageBox.warning(self, "Model Not Found", "Selected model no longer exists. Please select another.")
            delattr(self, 'selected_model')
            self._refresh_list()
            return
        
        sizes = ["tiny", "small", "medium", "large"]
        current_size = self.registry.registry["models"][self.selected_model].get("size", "xl")
        
        current_idx = sizes.index(current_size) if current_size in sizes else len(sizes)
        available = sizes[:current_idx]
        
        if not available:
            QMessageBox.information(self, "Min Size", "Model is already at minimum size")
            return
        
        size, ok = self._show_size_dialog("Shrink Model", available,
            f"Current size: {current_size}\nSelect target size:\n\n"
            "WARNING: Shrinking loses some capacity!")
        
        if ok and size:
            reply = QMessageBox.warning(
                self, "Confirm Shrink",
                f"Shrink '{self.selected_model}' from {current_size} to {size}?\n\n"
                "A backup will be created first.\n"
                "WARNING: Some knowledge may be lost in shrinking.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    # Auto-backup first
                    self._on_backup()
                    
                    # Load model
                    model, config = self.registry.load_model(self.selected_model)
                    
                    # Shrink
                    shrunk = shrink_model(model, size, config["vocab_size"])
                    
                    # Update the existing model
                    self.registry.save_model(self.selected_model, shrunk)
                    self.registry.registry["models"][self.selected_model]["size"] = size
                    self.registry._save_registry()
                    
                    self._refresh_list()
                    QMessageBox.information(self, "Success",
                        f"Model '{self.selected_model}' shrunk to {size}!\n"
                        f"A backup was created.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", str(e))
    
    def _on_delete(self):
        if not hasattr(self, 'selected_model'):
            QMessageBox.warning(self, "No Selection", "Select a model first")
            return
        
        # Check if model still exists
        if self.selected_model not in self.registry.registry.get("models", {}):
            QMessageBox.warning(self, "Model Not Found", "Selected model no longer exists.")
            delattr(self, 'selected_model')
            self._refresh_list()
            return
        
        reply = QMessageBox.warning(
            self, "Confirm Delete",
            f"DELETE model '{self.selected_model}'?\n\n"
            "TIP: If you made a backup, it will still exist.\n"
            "WARNING: The model folder and weights will be removed!",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Double confirm
            reply2 = QMessageBox.critical(
                self, "FINAL WARNING",
                f"Are you ABSOLUTELY SURE you want to delete '{self.selected_model}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply2 == QMessageBox.Yes:
                try:
                    self.registry.delete_model(self.selected_model, confirm=True)
                    self._refresh_list()
                    QMessageBox.information(self, "Deleted", "Model deleted.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", str(e))
    
    def _show_size_dialog(self, title, sizes, message):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel(message))
        
        combo = QComboBox()
        combo.addItems(sizes)
        layout.addWidget(combo)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            return combo.currentText(), True
        return None, False
    
    def get_selected_model(self):
        return getattr(self, 'selected_model', None)


class EnhancedMainWindow(QMainWindow):
    """Enhanced main window with setup wizard and model management."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enigma Engine")
        self.resize(1000, 700)
        
        # Initialize registry
        self.registry = ModelRegistry()
        self.current_model_name = None
        self.engine = None
        
        # Initialize toggle states
        self.auto_speak = False
        self.microphone_enabled = False
        
        # Check if first run (no models)
        if not self.registry.registry.get("models"):
            self._run_setup_wizard()
        else:
            self._show_model_selector()
        
        self._build_ui()
    
    def _run_setup_wizard(self):
        """Run first-time setup wizard."""
        wizard = SetupWizard(self.registry, self)
        if wizard.exec_() == QWizard.Accepted:
            result = wizard.get_result()
            try:
                self.registry.create_model(
                    result["name"],
                    size=result["size"],
                    vocab_size=32000,
                    description="Created via setup wizard"
                )
                self.current_model_name = result["name"]
                self._load_current_model()
            except Exception as e:
                QMessageBox.critical(self, "Setup Failed", str(e))
                sys.exit(1)
        else:
            # User cancelled - exit
            sys.exit(0)
    
    def _show_model_selector(self):
        """Show model selection on startup."""
        models = list(self.registry.registry.get("models", {}).keys())
        if len(models) == 1:
            self.current_model_name = models[0]
        else:
            dialog = ModelManagerDialog(self.registry, parent=self)
            if dialog.exec_() == QDialog.Accepted:
                self.current_model_name = dialog.get_selected_model()
            
            if not self.current_model_name and models:
                self.current_model_name = models[0]
        
        self._load_current_model()
    
    def _load_current_model(self):
        """Load the current model into the engine."""
        if self.current_model_name:
            try:
                # Create engine with selected model
                model, config = self.registry.load_model(self.current_model_name)
                
                # Create a custom engine with this model
                from ..core.inference import EnigmaEngine
                self.engine = EnigmaEngine.__new__(EnigmaEngine)
                self.engine.device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
                self.engine.model = model
                self.engine.model.to(self.engine.device)
                self.engine.model.eval()
                from ..core.tokenizer import load_tokenizer
                self.engine.tokenizer = load_tokenizer()
                
                # Initialize the AI's brain for learning
                from ..core.ai_brain import get_brain
                self.brain = get_brain(
                    self.current_model_name, 
                    auto_learn=getattr(self, 'learn_while_chatting', True)
                )
                
                self.setWindowTitle(f"Enigma Engine - {self.current_model_name}")
                
                # Update training tab label
                if hasattr(self, 'training_model_label'):
                    self.training_model_label.setText(f"Model: {self.current_model_name}")
                
                # Refresh notes files for new model
                if hasattr(self, 'notes_file_combo'):
                    self._refresh_notes_files()
                    
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Could not load model: {e}")
                self.engine = None
                self.brain = None
    
    def _build_ui(self):
        """Build the main UI."""
        # Menu bar
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New Model...", self._on_new_model)
        file_menu.addAction("Open Model...", self._on_open_model)
        file_menu.addSeparator()
        file_menu.addAction("Backup Current Model", self._on_backup_current)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        
        # Options menu with toggles
        options_menu = menubar.addMenu("Options")
        
        self.dark_mode_action = options_menu.addAction("Dark Mode (ON)")
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setChecked(True)
        self.dark_mode_action.triggered.connect(self._toggle_dark_mode)
        
        options_menu.addSeparator()
        
        self.avatar_action = options_menu.addAction("Avatar (OFF)")
        self.avatar_action.setCheckable(True)
        self.avatar_action.setChecked(False)
        self.avatar_action.triggered.connect(self._toggle_avatar)
        
        options_menu.addSeparator()
        
        self.auto_speak_action = options_menu.addAction("AI Auto-Speak (OFF)")
        self.auto_speak_action.setCheckable(True)
        self.auto_speak_action.setChecked(False)
        self.auto_speak_action.triggered.connect(self._toggle_auto_speak)
        
        self.microphone_action = options_menu.addAction("Microphone (OFF)")
        self.microphone_action.setCheckable(True)
        self.microphone_action.setChecked(False)
        self.microphone_action.triggered.connect(self._toggle_microphone)
        
        options_menu.addSeparator()
        
        # Learn while chatting toggle
        self.learn_action = options_menu.addAction("Learn While Chatting (ON)")
        self.learn_action.setCheckable(True)
        self.learn_action.setChecked(True)  # On by default
        self.learn_action.triggered.connect(self._toggle_learning)
        self.learn_while_chatting = True
        
        # Status bar with clickable model selector
        self.model_status_btn = QPushButton(f"Model: {self.current_model_name or 'None'}  v")
        self.model_status_btn.setFlat(True)
        self.model_status_btn.setCursor(Qt.PointingHandCursor)
        self.model_status_btn.clicked.connect(self._on_open_model)
        self.model_status_btn.setToolTip("Click to change model")
        self.model_status_btn.setStyleSheet("""
            QPushButton {
                border: none;
                padding: 2px 8px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: rgba(137, 180, 250, 0.3);
                border-radius: 4px;
            }
        """)
        self.statusBar().addWidget(self.model_status_btn)
        
        # Apply dark mode by default
        self.setStyleSheet(DARK_STYLE)
        
        # Import tabs from separate modules
        from .tabs import (
            create_chat_tab, create_training_tab, create_avatar_tab,
            create_vision_tab, create_sessions_tab, create_instructions_tab,
            create_terminal_tab
        )
        
        # Main tabs
        tabs = QTabWidget()
        self.tabs = tabs  # Store reference for AI control
        tabs.addTab(create_chat_tab(self), "[>] Chat")
        tabs.addTab(create_training_tab(self), "[+] Train")
        tabs.addTab(create_avatar_tab(self), "[*] Avatar")
        tabs.addTab(create_vision_tab(self), "[o] Vision")
        tabs.addTab(create_terminal_tab(self), "[#] Terminal")
        tabs.addTab(create_sessions_tab(self), "[=] History")
        tabs.addTab(create_instructions_tab(self), "[~] Files")
        
        self.setCentralWidget(tabs)
    
    def _toggle_dark_mode(self, checked):
        """Toggle between dark and light themes."""
        if checked:
            self.setStyleSheet(DARK_STYLE)
            self.dark_mode_action.setText("Dark Mode (ON)")
        else:
            self.setStyleSheet(LIGHT_STYLE)
            self.dark_mode_action.setText("Dark Mode (OFF)")
    
    def _toggle_auto_speak(self, checked):
        """Toggle auto-speak mode."""
        self.auto_speak = checked
        if hasattr(self, 'auto_speak_action'):
            if checked:
                self.auto_speak_action.setText("AI Auto-Speak (ON)")
            else:
                self.auto_speak_action.setText("AI Auto-Speak (OFF)")
    
    def _toggle_microphone(self, checked):
        """Toggle microphone listening."""
        self.microphone_enabled = checked
        if hasattr(self, 'microphone_action'):
            if checked:
                self.microphone_action.setText("Microphone (ON)")
            else:
                self.microphone_action.setText("Microphone (OFF)")
    
    def _toggle_learning(self, checked):
        """Toggle learn-while-chatting mode."""
        self.learn_while_chatting = checked
        if hasattr(self, 'learn_action'):
            if checked:
                self.learn_action.setText("Learn While Chatting (ON)")
            else:
                self.learn_action.setText("Learn While Chatting (OFF)")
        
        # Update brain if loaded
        if hasattr(self, 'brain') and self.brain:
            self.brain.auto_learn = checked
    
    def _toggle_avatar(self, checked):
        """Toggle avatar enabled/disabled."""
        try:
            if checked:
                self._enable_avatar()
                self.avatar_action.setText("Avatar (ON)")
            else:
                self._disable_avatar()
                self.avatar_action.setText("Avatar (OFF)")
        except Exception as e:
            # Don't crash if avatar fails
            self.avatar_action.setChecked(False)
            self.avatar_action.setText("Avatar (OFF)")
    
    def _toggle_screen_watching(self, checked):
        """Toggle continuous screen watching."""
        if checked:
            self.btn_start_watching.setText("[x] Stop Watching")
            interval_ms = self.vision_interval_spin.value() * 1000
            self.vision_timer.start(interval_ms)
            self._do_continuous_capture()
        else:
            self.btn_start_watching.setText("[o] Start Watching")
            self.vision_timer.stop()
    
    def _do_single_capture(self):
        """Do a single screen capture."""
        self._capture_screen()
    
    def _do_continuous_capture(self):
        """Capture for continuous watching."""
        self._capture_screen()
    
    def _capture_camera(self):
        """Capture image from webcam/camera."""
        try:
            import cv2
            
            # Try to open camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.vision_preview.setText("Camera not available\\n\\nMake sure a camera is connected.")
                return
            
            # Capture frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                self.vision_preview.setText("Failed to capture from camera")
                return
            
            # Convert BGR to RGB
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Store for analysis
            self._last_screenshot = img
            self.current_vision_image = "camera"
            
            # Resize for display
            display_img = img.copy()
            display_img.thumbnail((640, 400))
            
            # Convert to QPixmap
            import io
            buffer = io.BytesIO()
            display_img.save(buffer, format="PNG")
            buffer.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.read())
            self.vision_preview.setPixmap(pixmap)
            
            # Info
            width, height = img.size
            from datetime import datetime
            info = f"Camera: {width}x{height} | Captured: {datetime.now().strftime('%H:%M:%S')}"
            self.vision_text.setPlainText(info)
            
        except ImportError:
            self.vision_preview.setText("Camera capture requires OpenCV\\n\\nInstall: pip install opencv-python")
        except Exception as e:
            self.vision_preview.setText(f"Camera error: {e}")
    
    def _load_vision_image(self):
        """Load an image file for analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            from PIL import Image
            img = Image.open(file_path)
            
            # Store for analysis
            self._last_screenshot = img
            self.current_vision_image = file_path
            
            # Resize for display
            display_img = img.copy()
            display_img.thumbnail((640, 400))
            
            # Convert to QPixmap
            import io
            buffer = io.BytesIO()
            display_img.save(buffer, format="PNG")
            buffer.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.read())
            self.vision_preview.setPixmap(pixmap)
            
            # Info
            width, height = img.size
            from pathlib import Path
            info = f"Image: {Path(file_path).name} | {width}x{height}"
            self.vision_text.setPlainText(info)
            
        except Exception as e:
            self.vision_preview.setText(f"Error loading image: {e}")
    
    def _analyze_vision_image(self):
        """Have AI analyze the current image."""
        if not hasattr(self, '_last_screenshot') or self._last_screenshot is None:
            self.vision_text.setPlainText("No image to analyze. Capture or load an image first.")
            return
        
        # Get OCR text
        ocr_text = ""
        try:
            from ..tools.simple_ocr import extract_text
            ocr_text = extract_text(self._last_screenshot)
        except:
            pass
        
        # Build analysis
        analysis = []
        analysis.append(f"Image size: {self._last_screenshot.size[0]}x{self._last_screenshot.size[1]}")
        
        if ocr_text:
            analysis.append(f"\\nDetected text:\\n{ocr_text}")
        else:
            analysis.append("\\nNo text detected in image.")
        
        # If AI is available, get description
        if self.engine:
            try:
                prompt = "Describe what you might see in a screenshot or image."
                # Note: Real vision would need multi-modal model
                analysis.append(f"\\n(AI vision analysis requires multi-modal model)")
            except:
                pass
        
        self.vision_text.setPlainText("\\n".join(analysis))
    
    def _capture_screen(self):
        """Capture screen and display it. Uses scrot on Linux (Wayland/Pi friendly)."""
        try:
            img = None
            error_msg = None
            
            # On Linux, use scrot (works on Wayland, X11, and Pi)
            import platform
            import shutil
            
            if platform.system() == "Linux" and shutil.which("scrot"):
                try:
                    import subprocess
                    import tempfile
                    import os
                    from PIL import Image
                    
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                        tmp_path = f.name
                    
                    # Run scrot with overwrite flag
                    result = subprocess.run(
                        ['scrot', '-o', tmp_path], 
                        capture_output=True, 
                        text=True, 
                        timeout=10
                    )
                    
                    if result.returncode == 0 and os.path.exists(tmp_path):
                        img = Image.open(tmp_path)
                        img = img.copy()  # Load into memory
                        os.unlink(tmp_path)  # Clean up temp file
                    else:
                        error_msg = f"scrot failed: {result.stderr}"
                except Exception as e:
                    error_msg = f"scrot error: {e}"
            
            # macOS - use screencapture
            elif platform.system() == "Darwin" and shutil.which("screencapture"):
                try:
                    import subprocess
                    import tempfile
                    import os
                    from PIL import Image
                    
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                        tmp_path = f.name
                    
                    subprocess.run(['screencapture', '-x', tmp_path], timeout=10)
                    
                    if os.path.exists(tmp_path):
                        img = Image.open(tmp_path)
                        img = img.copy()
                        os.unlink(tmp_path)
                except Exception as e:
                    error_msg = f"screencapture error: {e}"
            
            # Fallback: Try PIL ImageGrab (Windows, some X11)
            if img is None:
                try:
                    from PIL import ImageGrab
                    img = ImageGrab.grab()
                except Exception as e:
                    if error_msg:
                        error_msg += f", ImageGrab: {e}"
                    else:
                        error_msg = f"ImageGrab error: {e}"
            
            # Last resort: mss (may fail on Wayland)
            if img is None:
                try:
                    import mss
                    from PIL import Image
                    with mss.mss() as sct:
                        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                        screenshot = sct.grab(monitor)
                        img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
                except Exception as e:
                    if error_msg:
                        error_msg += f", mss: {e}"
                    else:
                        error_msg = f"mss error: {e}"
            
            if img is None:
                install_hint = ""
                if platform.system() == "Linux":
                    install_hint = "\n\nInstall scrot: sudo apt install scrot"
                self.vision_preview.setText(f"Screenshot failed\n\n{error_msg}{install_hint}")
                return
            
            # Save full image for AI analysis
            self._last_screenshot = img
            
            # Resize for display
            display_img = img.copy()
            display_img.thumbnail((640, 400))
            
            # Convert to QPixmap
            import io
            buffer = io.BytesIO()
            display_img.save(buffer, format="PNG")
            buffer.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.read())
            self.vision_preview.setPixmap(pixmap)
            
            # Basic info
            width, height = img.size
            from datetime import datetime
            info = f"Screen: {width}x{height} | Captured: {datetime.now().strftime('%H:%M:%S')}"
            self.vision_text.setPlainText(info)
            
        except Exception as e:
            self.vision_preview.setText(f"Error: {e}")
    
    # ========== TERMINAL METHODS ==========
    
    def _clear_terminal(self):
        """Clear the terminal output."""
        if hasattr(self, 'terminal_output'):
            self.terminal_output.clear()
            self._terminal_lines = []
    
    def _update_terminal_filter(self):
        """Update the terminal log level filter."""
        if hasattr(self, 'terminal_log_level'):
            self._terminal_log_level = self.terminal_log_level.currentText()
    
    def log_terminal(self, message, level="info"):
        """Log a message to the AI terminal display."""
        from .tabs.terminal_tab import log_to_terminal
        log_to_terminal(self, message, level)
    
    def update_terminal_stats(self, tokens_per_sec=None, memory_mb=None, model_name=None):
        """Update the terminal statistics display."""
        if tokens_per_sec is not None and hasattr(self, 'terminal_tps_label'):
            self.terminal_tps_label.setText(f"Tokens/sec: {tokens_per_sec:.1f}")
        if memory_mb is not None and hasattr(self, 'terminal_memory_label'):
            self.terminal_memory_label.setText(f"Memory: {memory_mb:.1f} MB")
        if model_name is not None and hasattr(self, 'terminal_model_label'):
            self.terminal_model_label.setText(f"Model: {model_name}")
    
    # ========== AI WATCHING METHODS ==========
    
    def ai_start_watching(self):
        """AI can start continuous screen watching."""
        if not self.btn_start_watching.isChecked():
            self.btn_start_watching.setChecked(True)
            self._toggle_screen_watching()
        return "Started screen watching"
    
    def ai_stop_watching(self):
        """AI can stop continuous screen watching."""
        if self.btn_start_watching.isChecked():
            self.btn_start_watching.setChecked(False)
            self._toggle_screen_watching()
        return "Stopped screen watching"
    
    def ai_capture_screen(self):
        """AI can capture a single screenshot."""
        self._capture_screen()
        return "Screen captured"
    
    def ai_get_screen_text(self):
        """AI can get OCR text from last screenshot."""
        if not hasattr(self, '_last_screenshot') or self._last_screenshot is None:
            return "No screenshot available. Use ai_capture_screen() first."
        try:
            from .tools.simple_ocr import extract_text
            text = extract_text(self._last_screenshot)
            return text if text else "No text detected in screenshot"
        except:
            return "OCR not available"
    
    # === Session Actions ===
    
    def _populate_history_ai_selector(self):
        """Populate the AI selector dropdown in history tab."""
        if not hasattr(self, 'history_ai_selector'):
            return
        self.history_ai_selector.clear()
        self.history_ai_selector.addItem("All AIs")
        for name in self.registry.registry.get("models", {}).keys():
            self.history_ai_selector.addItem(name)
        # Select current model if available
        if self.current_model_name:
            idx = self.history_ai_selector.findText(self.current_model_name)
            if idx >= 0:
                self.history_ai_selector.setCurrentIndex(idx)
    
    def _on_history_ai_changed(self, ai_name):
        """Handle AI selection change in history tab."""
        self._refresh_sessions()
    
    def _get_sessions_dir(self):
        """Get the sessions directory based on selected AI."""
        if hasattr(self, 'history_ai_selector'):
            ai_name = self.history_ai_selector.currentText()
            if ai_name and ai_name != "All AIs":
                # Per-AI sessions
                return Path(CONFIG.get("models_dir", "models")) / ai_name / "brain" / "conversations"
        # Default global sessions
        return Path(CONFIG.get("data_dir", "data")) / "conversations"
    
    def _refresh_sessions(self):
        """Refresh the list of saved sessions."""
        if not hasattr(self, 'sessions_list'):
            return
        self.sessions_list.clear()
        
        selected_ai = ""
        if hasattr(self, 'history_ai_selector'):
            selected_ai = self.history_ai_selector.currentText()
        
        if selected_ai == "All AIs" or not selected_ai:
            # Show sessions from all AIs
            all_sessions = []
            
            # Global sessions
            global_dir = Path(CONFIG.get("data_dir", "data")) / "conversations"
            if global_dir.exists():
                for f in global_dir.glob("*.json"):
                    all_sessions.append((f.stat().st_mtime, f.stem, "global"))
            
            # Per-AI sessions
            models_dir = Path(CONFIG.get("models_dir", "models"))
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    conv_dir = model_dir / "brain" / "conversations"
                    if conv_dir.exists():
                        for f in conv_dir.glob("*.json"):
                            all_sessions.append((f.stat().st_mtime, f.stem, model_dir.name))
            
            # Sort by time and display
            for mtime, name, ai in sorted(all_sessions, reverse=True):
                display = f"[{ai}] {name}" if ai != "global" else name
                self.sessions_list.addItem(display)
        else:
            # Show sessions for selected AI only
            conv_dir = self._get_sessions_dir()
            conv_dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(conv_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
                self.sessions_list.addItem(f.stem)
    
    def _load_session(self, item):
        """Load a session's content into the viewer."""
        if not item:
            return
        session_name = item.text()
        conv_dir = Path(CONFIG.get("data_dir", "data")) / "conversations"
        session_file = conv_dir / f"{session_name}.json"
        
        if session_file.exists():
            try:
                data = json.loads(session_file.read_text())
                html = f"<h3>{session_name}</h3>"
                for msg in data.get("messages", []):
                    role = msg.get("role", "user")
                    text = msg.get("text", "")
                    if role == "user":
                        html += f"<p><b>You:</b> {text}</p>"
                    else:
                        html += f"<p><b>AI:</b> {text}</p>"
                self.session_viewer.setHtml(html)
                self._current_session = session_name
            except Exception as e:
                self.session_viewer.setPlainText(f"Error loading session: {e}")
    
    def _new_session(self):
        """Create a new chat session."""
        name, ok = QInputDialog.getText(self, "New Session", "Session name:")
        if ok and name:
            if hasattr(self, 'chat_messages') and self.chat_messages:
                self._save_current_chat(name)
            else:
                conv_dir = Path(CONFIG.get("data_dir", "data")) / "conversations"
                conv_dir.mkdir(parents=True, exist_ok=True)
                session_file = conv_dir / f"{name}.json"
                session_file.write_text(json.dumps({
                    "name": name,
                    "saved_at": time.time(),
                    "messages": []
                }))
            self._refresh_sessions()
            self.chat_messages = []
    
    def _rename_session(self):
        """Rename the selected session."""
        item = self.sessions_list.currentItem()
        if not item:
            QMessageBox.warning(self, "No Selection", "Select a session to rename")
            return
        old_name = item.text()
        new_name, ok = QInputDialog.getText(self, "Rename Session", "New name:", text=old_name)
        if ok and new_name and new_name != old_name:
            conv_dir = Path(CONFIG.get("data_dir", "data")) / "conversations"
            old_file = conv_dir / f"{old_name}.json"
            new_file = conv_dir / f"{new_name}.json"
            if old_file.exists():
                old_file.rename(new_file)
                self._refresh_sessions()
    
    def _delete_session(self):
        """Delete the selected session."""
        item = self.sessions_list.currentItem()
        if not item:
            QMessageBox.warning(self, "No Selection", "Select a session to delete")
            return
        session_name = item.text()
        reply = QMessageBox.question(
            self, "Delete Session",
            f"Delete session '{session_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            conv_dir = Path(CONFIG.get("data_dir", "data")) / "conversations"
            session_file = conv_dir / f"{session_name}.json"
            if session_file.exists():
                session_file.unlink()
            self._refresh_sessions()
            self.session_viewer.clear()
    
    def _load_session_into_chat(self):
        """Load the selected session into the chat tab."""
        if not hasattr(self, '_current_session'):
            QMessageBox.warning(self, "No Session", "Select a session first")
            return
        conv_dir = Path(CONFIG.get("data_dir", "data")) / "conversations"
        session_file = conv_dir / f"{self._current_session}.json"
        if session_file.exists():
            try:
                data = json.loads(session_file.read_text())
                self.chat_display.clear()
                self.chat_messages = data.get("messages", [])
                for msg in self.chat_messages:
                    role = msg.get("role", "user")
                    text = msg.get("text", "")
                    if role == "user":
                        self.chat_display.append(f"<b>You:</b> {text}")
                    else:
                        self.chat_display.append(f"<b>{self.current_model_name}:</b> {text}")
                self.tabs.setCurrentIndex(0)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load session: {e}")
    
    def _save_current_chat(self, name=None):
        """Save current chat to a session file."""
        if not hasattr(self, 'chat_messages'):
            self.chat_messages = []
        if not name:
            name = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        conv_dir = Path(CONFIG.get("data_dir", "data")) / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        session_file = conv_dir / f"{name}.json"
        session_file.write_text(json.dumps({
            "name": name,
            "saved_at": time.time(),
            "messages": self.chat_messages
        }))
    
    # === Data Editor Actions (for Training tab) ===
    
    
    def _refresh_data_files(self):
        """Refresh list of training data files."""
        if not hasattr(self, 'data_file_combo'):
            return
        self.data_file_combo.clear()
        
        # Get AI's data directory
        if self.current_model_name:
            model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
            data_dir = model_info.get("data_dir") or (Path(model_info.get("path", "")) / "data")
            if isinstance(data_dir, str):
                data_dir = Path(data_dir)
        else:
            data_dir = Path(CONFIG.get("data_dir", "data"))
        
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure training.txt exists
        training_file = data_dir / "training.txt"
        if not training_file.exists():
            training_file.write_text("# Training Data\n# Add Q&A pairs below\n\nQ: Hello\nA: Hi there!\n")
        
        # Add files (training data files, not instructions)
        for f in sorted(data_dir.glob("*.txt")):
            if f.name not in ["instructions.txt", "notes.txt"]:
                self.data_file_combo.addItem(f.name, str(f))
        
        # Select first file if available
        if self.data_file_combo.count() > 0:
            self.data_file_combo.setCurrentIndex(0)
    
    def _load_data_file(self, index):
        """Load a data file into the training editor."""
        if index < 0 or not hasattr(self, 'data_file_combo'):
            return
        
        filepath = self.data_file_combo.itemData(index)
        if filepath and Path(filepath).exists():
            self.data_editor.setPlainText(Path(filepath).read_text())
            self._current_data_file = filepath
            self.training_data_path = filepath  # Auto-set for training
    
    def _save_data_file(self):
        """Save the training data file."""
        if not hasattr(self, '_current_data_file') or not self._current_data_file:
            QMessageBox.warning(self, "No File", "Select a file first")
            return
        
        try:
            Path(self._current_data_file).write_text(self.data_editor.toPlainText())
            QMessageBox.information(self, "Saved", "File saved!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save: {e}")
    
    def _create_data_file(self):
        """Create a new training data file."""
        name, ok = QInputDialog.getText(self, "New Training File", "File name (without .txt):")
        if ok and name:
            if not name.endswith(".txt"):
                name += ".txt"
            
            # Get data directory
            if self.current_model_name:
                model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
                data_dir = model_info.get("data_dir") or (Path(model_info.get("path", "")) / "data")
                if isinstance(data_dir, str):
                    data_dir = Path(data_dir)
            else:
                data_dir = Path(CONFIG.get("data_dir", "data"))
            
            data_dir.mkdir(parents=True, exist_ok=True)
            new_file = data_dir / name
            
            if new_file.exists():
                QMessageBox.warning(self, "Exists", f"{name} already exists")
                return
            
            new_file.write_text("# Training Data\n# Add Q&A pairs below\n\n")
            self._refresh_data_files()
            
            # Select the new file
            idx = self.data_file_combo.findText(name)
            if idx >= 0:
                self.data_file_combo.setCurrentIndex(idx)
    
    # === Avatar Actions ===
    
    def _refresh_avatar_status(self):
        """Update avatar status - now handled by avatar_tab."""
        # Initialize avatar expressions dict if needed
        if not hasattr(self, 'avatar_expressions'):
            self.avatar_expressions = {}
            self.current_expression = "neutral"
        
        # Try to load default avatar
        self._load_default_avatar()
    
    def _load_default_avatar(self):
        """Try to load avatar image from model's avatar folder."""
        if not self.current_model_name:
            return
            
        model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
        model_path = Path(model_info.get("path", ""))
        avatar_dir = model_path / "avatar"
        
        if not avatar_dir.exists():
            avatar_dir.mkdir(exist_ok=True)
            return
        
        # Load all expression images
        for img_file in avatar_dir.glob("*.png"):
            expr_name = img_file.stem.lower()
            self.avatar_expressions[expr_name] = str(img_file)
        for img_file in avatar_dir.glob("*.jpg"):
            expr_name = img_file.stem.lower()
            self.avatar_expressions[expr_name] = str(img_file)
        
        # Display neutral or first available
        if "neutral" in self.avatar_expressions:
            self._display_avatar_image(self.avatar_expressions["neutral"])
        elif self.avatar_expressions:
            first = list(self.avatar_expressions.values())[0]
            self._display_avatar_image(first)
    
    def _load_avatar_image(self):
        """Load a custom avatar image."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Avatar Image", "", "Images (*.png *.jpg *.jpeg *.gif);;All Files (*)"
        )
        if filepath:
            # Copy to model's avatar folder
            if self.current_model_name:
                try:
                    model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
                    model_path = Path(model_info.get("path", ""))
                    avatar_dir = model_path / "avatar"
                    avatar_dir.mkdir(exist_ok=True)
                    
                    import shutil
                    dest = avatar_dir / f"neutral{Path(filepath).suffix}"
                    shutil.copy(filepath, dest)
                    filepath = str(dest)
                except Exception as e:
                    pass  # Use original filepath
            
            # Display the image
            self._display_avatar_image(filepath)
            
            # Update expression dict
            if hasattr(self, 'avatar_expressions'):
                self.avatar_expressions["neutral"] = filepath
    
    def _display_avatar_image(self, filepath):
        """Display an avatar image."""
        pixmap = QPixmap(filepath)
        if not pixmap.isNull():
            scaled = pixmap.scaled(380, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.avatar_image_label.setPixmap(scaled)
            self.avatar_image_label.setStyleSheet("border: 2px solid #89b4fa; border-radius: 12px; background: #1e1e2e;")
            if hasattr(self, 'avatar_status_label'):
                self.avatar_status_label.setText(f"Avatar loaded: {Path(filepath).name}")
        else:
            self.avatar_image_label.setText("Failed to load image")
            if hasattr(self, 'avatar_status_label'):
                self.avatar_status_label.setText("Failed to load avatar")
    
    def _enable_avatar(self):
        """Enable avatar display."""
        self._refresh_avatar_status()
    
    def _disable_avatar(self):
        """Disable avatar display."""
        self.avatar_image_label.clear()
        self.avatar_image_label.setText("Avatar disabled\n\nEnable in Options -> Avatar")
        if hasattr(self, 'avatar_status_label'):
            self.avatar_status_label.setText("Avatar disabled")
    
    # === Vision Actions ===
    
    def _capture_screen(self):
        """Capture and display screen."""
        try:
            from ..tools.vision import ScreenCapture
            capture = ScreenCapture()
            img = capture.capture()
            
            if img:
                # Check if image is all black (common on Wayland)
                import numpy as np
                img_array = np.array(img)
                if img_array.max() < 10:  # Nearly all black
                    self.vision_preview.setText(
                        "WARNING: Screenshot appears black\\n\\n"
                        "This often happens on Wayland (Raspberry Pi default).\\n\\n"
                        "Try:\\n"
                        "1. Install: pip install pyscreenshot mss\\n"
                        "2. Or switch to X11 session\\n"
                        "3. Or use scrot: sudo apt install scrot"
                    )
                    return
                
                # Convert PIL to QPixmap safely
                img = img.resize((640, 360))  # Resize for display
                img = img.convert("RGB")  # Ensure RGB mode
                
                # Use BytesIO for safer conversion
                import io
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)
                
                pixmap = QPixmap()
                pixmap.loadFromData(buffer.read())
                self.vision_preview.setPixmap(pixmap)
                self.vision_preview.setToolTip(f"Captured at {datetime.now().strftime('%H:%M:%S')}")
            else:
                self.vision_preview.setText("Failed to capture screen\\n\\nTry: pip install mss pyscreenshot")
        except Exception as e:
            self.vision_preview.setText(f"Error: {e}\\n\\nTry: pip install pillow mss")
    
    def _analyze_screen(self):
        """Analyze screen with OCR."""
        try:
            from ..tools.vision import get_screen_vision
            vision = get_screen_vision()
            result = vision.see(describe=True, detect_text=True)
            
            output = []
            if result.get("success"):
                output.append(f"Resolution: {result['size']['width']}x{result['size']['height']}")
                if result.get("description"):
                    output.append(f"\nDescription: {result['description']}")
                if result.get("text_content"):
                    output.append(f"\n--- Detected Text ---\n{result['text_content'][:500]}")
            else:
                output.append(f"Error: {result.get('error', 'Unknown')}")
            
            self.vision_text.setPlainText("\n".join(output))
            
            # Also capture for preview
            self._capture_screen()
        except Exception as e:
            self.vision_text.setPlainText(f"Error: {e}")
    
    def _refresh_models_list(self):
        """Refresh models list if it exists (Models tab removed)."""
        if not hasattr(self, 'models_list'):
            return
        self.models_list.clear()
        for name, info in self.registry.registry.get("models", {}).items():
            status = "[+]" if info.get("has_weights") else "[ ]"
            current = " << ACTIVE" if name == self.current_model_name else ""
            self.models_list.addItem(f"{status} {name} ({info.get('size', '?')}){current}")
    
    # === Actions ===
    
    def _on_send(self):
        text = self.chat_input.text().strip()
        if not text or not self.engine:
            return
        
        # Initialize chat messages list if needed
        if not hasattr(self, 'chat_messages'):
            self.chat_messages = []
        
        self.chat_display.append(f"<b>You:</b> {text}")
        self.chat_input.clear()
        
        # Track user message
        self.chat_messages.append({
            "role": "user",
            "text": text,
            "ts": time.time()
        })
        
        try:
            response = self.engine.generate(text, max_gen=50)
            self.chat_display.append(f"<b>{self.current_model_name}:</b> {response}")
            self.last_response = response
            
            # Track AI response
            self.chat_messages.append({
                "role": "assistant",
                "text": response,
                "ts": time.time()
            })
            
            # Learn from this interaction if enabled
            if getattr(self, 'learn_while_chatting', True) and hasattr(self, 'brain') and self.brain:
                self.brain.record_interaction(text, response)
                
                # Check if we should auto-train
                if self.brain.should_auto_train():
                    self.statusBar().showMessage(
                        f"[+] Learned {self.brain.interactions_since_train} new things! "
                        "Training will improve responses.", 5000
                    )
            
            # Auto-speak if enabled
            if getattr(self, 'auto_speak', False):
                self._speak_text(response)
        except Exception as e:
            self.chat_display.append(f"<i>Error: {e}</i>")
    
    def _speak_text(self, text):
        """Speak text using TTS."""
        try:
            from ..voice import speak
            speak(text)
        except Exception as e:
            pass  # Silent fail for auto-speak
    
    def _on_speak_last(self):
        if hasattr(self, 'last_response'):
            try:
                from ..voice import speak
                speak(self.last_response)
            except Exception as e:
                QMessageBox.warning(self, "TTS Error", str(e))
    
    def _on_new_model(self):
        wizard = SetupWizard(self.registry, self)
        if wizard.exec_() == QWizard.Accepted:
            result = wizard.get_result()
            try:
                self.registry.create_model(result["name"], size=result["size"], vocab_size=32000)
                self._refresh_models_list()
                QMessageBox.information(self, "Success", f"Created model '{result['name']}'")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
    
    def _on_open_model(self):
        dialog = ModelManagerDialog(self.registry, self.current_model_name, self)
        if dialog.exec_() == QDialog.Accepted:
            selected = dialog.get_selected_model()
            if selected:
                self.current_model_name = selected
                self._load_current_model()
                self.model_status_btn.setText(f"Model: {self.current_model_name}  v")
                self.setWindowTitle(f"Enigma Engine - {self.current_model_name}")
    
    def _on_backup_current(self):
        if not self.current_model_name:
            QMessageBox.warning(self, "No Model", "No model is currently loaded.")
            return
        
        model_dir = Path(self.registry.models_dir) / self.current_model_name
        backup_dir = Path(self.registry.models_dir) / f"{self.current_model_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            shutil.copytree(model_dir, backup_dir)
            QMessageBox.information(self, "Backup Complete", f"Backed up to:\n{backup_dir}")
        except Exception as e:
            QMessageBox.warning(self, "Backup Failed", str(e))
    
    def _on_select_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Training Data", "", "Text Files (*.txt)")
        if path:
            self.training_data_path = path
            self.data_path_label.setText(f"Data: {Path(path).name}")
    
    def _on_start_training(self):
        if not self.current_model_name:
            QMessageBox.warning(self, "No Model", "No model loaded")
            return
        
        # Save training file first if editor exists
        if hasattr(self, 'training_editor') and hasattr(self, '_current_training_file'):
            try:
                content = self.training_editor.toPlainText()
                Path(self._current_training_file).write_text(content)
            except:
                pass
        
        if not hasattr(self, 'training_data_path') or not self.training_data_path:
            QMessageBox.warning(self, "No Data", "Select a training file first.")
            return
        
        # Update button and progress
        self.btn_train.setEnabled(False)
        self.btn_train.setText("Training...")
        self.train_progress.setValue(0)
        QApplication.processEvents()
        
        try:
            from ..core.trainer import EnigmaTrainer
            
            model, config = self.registry.load_model(self.current_model_name)
            
            trainer = EnigmaTrainer(
                model=model,
                model_name=self.current_model_name,
                registry=self.registry,
                data_path=self.training_data_path,
                batch_size=self.batch_spin.value(),
                learning_rate=float(self.lr_input.text()),
            )
            
            epochs = self.epochs_spin.value()
            for epoch in range(epochs):
                trainer.train(epochs=1)
                progress = int((epoch + 1) / epochs * 100)
                self.train_progress.setValue(progress)
                QApplication.processEvents()
            
            # Reload model
            self._load_current_model()
            
            self.train_progress.setValue(100)
            self.btn_train.setText("Train")
            self.btn_train.setEnabled(True)
            QMessageBox.information(self, "Done", "Training finished!")
        except Exception as e:
            self.btn_train.setText("Train")
            self.btn_train.setEnabled(True)
            QMessageBox.warning(self, "Training Error", str(e))
    
    # === AI Control Methods ===
    # These methods allow the AI to control the GUI programmatically
    
    def ai_create_model(self, name: str, size: str = "tiny"):
        """AI can create a new model."""
        try:
            self.registry.create_model(name, size=size, vocab_size=32000)
            self._refresh_models_list()
            return f"Created model '{name}' with size '{size}'"
        except Exception as e:
            return f"Error creating model: {e}"
    
    def ai_switch_model(self, name: str):
        """AI can switch to a different model."""
        if name in self.registry.registry.get("models", {}):
            self.current_model_name = name
            self._load_current_model()
            self.model_status_btn.setText(f"Model: {name}  v")
            return f"Switched to model '{name}'"
        return f"Model '{name}' not found"
    
    def ai_send_message(self, text: str):
        """AI can send a chat message (for testing/demos)."""
        self.chat_input.setText(text)
        self._on_send()
        return "Message sent"
    
    def ai_switch_tab(self, tab_name: str):
        """AI can switch tabs."""
        tab_map = {
            "chat": 0, "train": 1, "training": 1, "avatar": 2, 
            "vision": 3, "history": 4, "sessions": 4, "files": 5, "help": 5, "notes": 5
        }
        idx = tab_map.get(tab_name.lower())
        if idx is not None:
            self.tabs.setCurrentIndex(idx)
            return f"Switched to {tab_name} tab"
        return f"Unknown tab: {tab_name}"
    
    def ai_save_session(self, name: str = None):
        """AI can save the current chat session."""
        self._save_current_chat(name)
        return f"Session saved as '{name or 'auto-named'}'"
    
    def ai_send_to_game(self, command: str):
        """AI can send commands to connected game."""
        if hasattr(self, 'game_connection') and self.game_connection:
            try:
                # TODO: Send via WebSocket/HTTP based on connection type
                if hasattr(self, 'game_log'):
                    self.game_log.append(f"AI >> {command}")
                return f"Sent to game: {command}"
            except Exception as e:
                return f"Failed to send: {e}"
        return "Not connected to any game"
    
    def ai_send_to_robot(self, command: str):
        """AI can send commands to connected robot."""
        if hasattr(self, 'robot_connection') and self.robot_connection:
            try:
                self.robot_connection.write(f"{command}\n".encode())
                if hasattr(self, 'robot_log'):
                    self.robot_log.append(f"AI >> {command}")
                return f"Sent to robot: {command}"
            except Exception as e:
                return f"Failed to send: {e}"
        return "Not connected to any robot"
    
    def ai_get_available_actions(self):
        """Return list of actions the AI can perform."""
        return [
            "ai_create_model(name, size='tiny'|'small'|'medium'|'large')",
            "ai_switch_model(name)",
            "ai_send_message(text)",
            "ai_switch_tab('chat'|'train'|'avatar'|'vision'|'history'|'files')",
            "ai_save_session(name)",
            "ai_capture_screen()",
            "ai_start_watching()",
            "ai_stop_watching()",
            "ai_get_screen_text()",
            "ai_send_to_game(command)",
            "ai_send_to_robot(command)",
        ]


def run_app():
    """Run the enhanced GUI application."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = EnhancedMainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
