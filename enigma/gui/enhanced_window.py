"""
Enhanced PyQt5 GUI for Enigma with Setup Wizard

Features:
  - First-run setup wizard to create/name your AI
  - Model selection and management
  - Backup before risky operations
  - Grow/shrink models with confirmation
  - Chat, Training, Voice integration
  - Dark/Light mode toggle
  - Avatar control panel
  - Screen vision preview
  - Training data editor
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
    QSlider, QSplitter, QPlainTextEdit, QToolTip, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QImage
import time


# === DARK/LIGHT THEME STYLESHEETS ===
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
            self.name_status.setText("⚠ Name already exists!")
            self.name_status.setStyleSheet("color: orange")
        elif not name.replace("_", "").isalnum():
            self.name_status.setText("✗ Use only letters, numbers, underscores")
            self.name_status.setStyleSheet("color: red")
        else:
            self.name_status.setText("✓ Name available")
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
                label += " ⭐ RECOMMENDED"
            if not can_use:
                label += " ⚠️ TOO LARGE"
            
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
            <li>GPU VRAM: {hw.get('vram_gb', 0)} GB {'✓' if hw.get('has_gpu') else '(no GPU)'}</li>
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
        
        self.btn_grow = QPushButton("Grow →")
        self.btn_grow.clicked.connect(self._on_grow)
        
        self.btn_shrink = QPushButton("← Shrink")
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
            status = "✓" if info.get("has_weights") else "○"
            self.model_list.addItem(f"{status} {name} ({info.get('size', '?')})")
    
    def _on_select_model(self, item):
        text = item.text()
        # Extract name from "✓ name (size)"
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
                
                # Grow
                try:
                    from ..core.model_scaling import grow_registered_model
                    new_name = f"{self.selected_model}_{size}"
                    grow_registered_model(
                        self.registry,
                        self.selected_model,
                        new_name,
                        size
                    )
                    self._refresh_list()
                    QMessageBox.information(self, "Success", 
                        f"Created grown model '{new_name}'\n"
                        f"Original '{self.selected_model}' unchanged.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", str(e))
    
    def _on_shrink(self):
        if not hasattr(self, 'selected_model'):
            QMessageBox.warning(self, "No Selection", "Select a model first")
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
            "⚠ Shrinking loses some capacity!")
        
        if ok and size:
            reply = QMessageBox.warning(
                self, "Confirm Shrink",
                f"Shrink '{self.selected_model}' from {current_size} to {size}?\n\n"
                "⚠ This will create a COPY - original is preserved.\n"
                "⚠ Some knowledge may be lost in shrinking.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    # Load model
                    model, config = self.registry.load_model(self.selected_model)
                    
                    # Shrink
                    shrunk = shrink_model(model, size, config["vocab_size"])
                    
                    # Save as new model
                    new_name = f"{self.selected_model}_{size}"
                    self.registry.create_model(new_name, size=size, vocab_size=config["vocab_size"])
                    self.registry.save_model(new_name, shrunk)
                    
                    self._refresh_list()
                    QMessageBox.information(self, "Success",
                        f"Created shrunk model '{new_name}'")
                except Exception as e:
                    QMessageBox.warning(self, "Error", str(e))
    
    def _on_delete(self):
        if not hasattr(self, 'selected_model'):
            QMessageBox.warning(self, "No Selection", "Select a model first")
            return
        
        reply = QMessageBox.warning(
            self, "Confirm Delete",
            f"DELETE model '{self.selected_model}'?\n\n"
            "⚠ This CANNOT be undone!\n"
            "⚠ All weights and checkpoints will be lost!",
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
                
                self.setWindowTitle(f"Enigma Engine - {self.current_model_name}")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Could not load model: {e}")
                self.engine = None
    
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
        
        # View menu with dark mode toggle
        view_menu = menubar.addMenu("View")
        self.dark_mode_action = view_menu.addAction("🌙 Dark Mode")
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setChecked(True)  # Default to dark mode
        self.dark_mode_action.triggered.connect(self._toggle_dark_mode)
        
        # Status bar with clickable model selector
        self.model_status_btn = QPushButton(f"Model: {self.current_model_name or 'None'}  ▼")
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
        
        # Main tabs
        tabs = QTabWidget()
        tabs.addTab(self._chat_tab(), "💬 Chat")
        tabs.addTab(self._training_tab(), "🎓 Training")
        tabs.addTab(self._avatar_tab(), "🤖 Avatar")
        tabs.addTab(self._vision_tab(), "👁️ Vision")
        tabs.addTab(self._instructions_tab(), "📖 Instructions")
        
        self.setCentralWidget(tabs)
    
    def _toggle_dark_mode(self, checked):
        """Toggle between dark and light themes."""
        if checked:
            self.setStyleSheet(DARK_STYLE)
            self.dark_mode_action.setText("🌙 Dark Mode")
        else:
            self.setStyleSheet(LIGHT_STYLE)
            self.dark_mode_action.setText("☀️ Light Mode")
    
    def _chat_tab(self):
        """Chat interface tab."""
        w = QWidget()
        layout = QVBoxLayout()
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)
        
        # Input
        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type a message...")
        self.chat_input.returnPressed.connect(self._on_send)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._on_send)
        
        self.speak_btn = QPushButton("🔊 Speak")
        self.speak_btn.clicked.connect(self._on_speak_last)
        
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(self.send_btn)
        input_layout.addWidget(self.speak_btn)
        
        layout.addLayout(input_layout)
        w.setLayout(layout)
        return w
    
    def _training_tab(self):
        """Training controls tab with integrated data editor."""
        w = QWidget()
        layout = QVBoxLayout()
        
        # Use a splitter to show data editor and training controls side by side
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Data Editor
        data_widget = QWidget()
        data_layout = QVBoxLayout()
        
        header = QLabel("Training Data")
        header.setObjectName("header")
        data_layout.addWidget(header)
        
        # File selection in a compact row
        file_layout = QHBoxLayout()
        self.data_file_combo = QComboBox()
        self.data_file_combo.setMinimumWidth(200)
        self._refresh_data_files()
        self.data_file_combo.setCurrentIndex(-1)
        self.data_file_combo.currentIndexChanged.connect(self._load_data_file)
        
        btn_new_file = QPushButton("New")
        btn_new_file.setMaximumWidth(50)
        btn_new_file.clicked.connect(self._create_data_file)
        
        file_layout.addWidget(self.data_file_combo)
        file_layout.addWidget(btn_new_file)
        data_layout.addLayout(file_layout)
        
        # Editor
        self.data_editor = QPlainTextEdit()
        self.data_editor.setPlaceholderText(
            "Select a file above or create a new one.\n\n"
            "FORMAT:\n"
            "Q: What is your name?\n"
            "A: My name is [AI name]\n\n"
            "User: Hello!\n"
            "Assistant: Hi there!"
        )
        data_layout.addWidget(self.data_editor)
        
        # Save button
        btn_save = QPushButton("💾 Save")
        btn_save.clicked.connect(self._save_data_file)
        data_layout.addWidget(btn_save)
        
        data_widget.setLayout(data_layout)
        splitter.addWidget(data_widget)
        
        # Right side: Training Controls
        train_widget = QWidget()
        train_layout = QVBoxLayout()
        
        train_header = QLabel("Training Controls")
        train_header.setObjectName("header")
        train_layout.addWidget(train_header)
        
        # Current data file indicator
        self.data_path_label = QLabel("No data file selected")
        train_layout.addWidget(self.data_path_label)
        
        btn_use_data = QPushButton("📚 Use Selected File for Training")
        btn_use_data.clicked.connect(self._use_data_for_training)
        train_layout.addWidget(btn_use_data)
        
        # Training parameters - simple layout without nested group
        train_layout.addWidget(QLabel("<b>Parameters:</b>"))
        
        params_layout = QFormLayout()
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(10)
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(4)
        
        self.lr_input = QLineEdit("0.0001")
        
        params_layout.addRow("Epochs:", self.epochs_spin)
        params_layout.addRow("Batch Size:", self.batch_spin)
        params_layout.addRow("Learning Rate:", self.lr_input)
        train_layout.addLayout(params_layout)
        
        # Parameter help (expandable)
        help_btn = QPushButton("ℹ️ Parameter Help")
        help_btn.setCheckable(True)
        help_btn.clicked.connect(self._toggle_param_help)
        train_layout.addWidget(help_btn)
        
        self.param_help_label = QLabel(
            "<b>Epochs:</b> Times to go through all data. More = better learning but slower.<br>"
            "<b>Batch Size:</b> Examples per step. Pi: 1-2, GPU: 4-16.<br>"
            "<b>Learning Rate:</b> How fast AI learns. Default 0.0001 is usually good."
        )
        self.param_help_label.setWordWrap(True)
        self.param_help_label.setVisible(False)
        train_layout.addWidget(self.param_help_label)
        
        # Progress
        self.train_progress = QProgressBar()
        self.train_status = QLabel("Ready")
        train_layout.addWidget(self.train_progress)
        train_layout.addWidget(self.train_status)
        
        # Train button
        self.btn_train = QPushButton("▶️ Start Training")
        self.btn_train.clicked.connect(self._on_start_training)
        train_layout.addWidget(self.btn_train)
        
        train_layout.addStretch()
        train_widget.setLayout(train_layout)
        splitter.addWidget(train_widget)
        
        # Set initial splitter sizes
        splitter.setSizes([500, 300])
        
        layout.addWidget(splitter)
        w.setLayout(layout)
        return w
    
    def _toggle_param_help(self, checked):
        """Toggle parameter help visibility."""
        self.param_help_label.setVisible(checked)
    
    def _avatar_tab(self):
        """Avatar control panel - AI controlled."""
        w = QWidget()
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("Avatar")
        header.setObjectName("header")
        layout.addWidget(header)
        
        # Avatar preview
        avatar_group = QGroupBox("Avatar Preview")
        avatar_layout = QVBoxLayout()
        
        self.avatar_image_label = QLabel("No avatar loaded")
        self.avatar_image_label.setMinimumSize(250, 250)
        self.avatar_image_label.setMaximumSize(350, 350)
        self.avatar_image_label.setAlignment(Qt.AlignCenter)
        self.avatar_image_label.setStyleSheet("border: 2px dashed #45475a; border-radius: 8px;")
        avatar_layout.addWidget(self.avatar_image_label, alignment=Qt.AlignCenter)
        
        # Upload buttons
        btn_row = QHBoxLayout()
        btn_load_image = QPushButton("📷 Load Image")
        btn_load_image.clicked.connect(self._load_avatar_image)
        btn_load_model = QPushButton("📦 Load Avatar Model")
        btn_load_model.clicked.connect(self._load_avatar_model)
        btn_row.addWidget(btn_load_image)
        btn_row.addWidget(btn_load_model)
        avatar_layout.addLayout(btn_row)
        
        avatar_group.setLayout(avatar_layout)
        layout.addWidget(avatar_group)
        
        # Status (AI controlled)
        status_group = QGroupBox("Status (AI Controlled)")
        status_layout = QFormLayout()
        
        self.avatar_status_label = QLabel("Not initialized")
        self.avatar_state_label = QLabel("Idle")
        self.avatar_expression_label = QLabel("neutral")
        
        status_layout.addRow("Status:", self.avatar_status_label)
        status_layout.addRow("State:", self.avatar_state_label)
        status_layout.addRow("Expression:", self.avatar_expression_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Enable/disable only
        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QHBoxLayout()
        
        self.btn_avatar_enable = QPushButton("✅ Enable")
        self.btn_avatar_enable.clicked.connect(self._enable_avatar)
        self.btn_avatar_disable = QPushButton("❌ Disable")
        self.btn_avatar_disable.clicked.connect(self._disable_avatar)
        
        ctrl_layout.addWidget(self.btn_avatar_enable)
        ctrl_layout.addWidget(self.btn_avatar_disable)
        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)
        
        # Info
        info = QLabel(
            "<b>ℹ️ About Avatar:</b><br>"
            "The avatar is controlled by the AI automatically.<br>"
            "It will change expressions and speak based on responses.<br><br>"
            "<b>Supported formats:</b><br>"
            "• Image: PNG, JPG, GIF<br>"
            "• Model: Live2D, VRM (place in models/[ai_name]/avatar/)"
        )
        info.setWordWrap(True)
        info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(info)
        
        layout.addStretch()
        
        # Initialize
        self._refresh_avatar_status()
        
        w.setLayout(layout)
        return w
    
    def _vision_tab(self):
        """Screen vision - AI can watch the screen."""
        w = QWidget()
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("Screen Vision")
        header.setObjectName("header")
        layout.addWidget(header)
        
        # Preview area
        preview_group = QGroupBox("Screen Preview")
        preview_layout = QVBoxLayout()
        
        self.vision_preview = QLabel("Click 'Start Watching' to enable AI vision")
        self.vision_preview.setMinimumHeight(300)
        self.vision_preview.setAlignment(Qt.AlignCenter)
        self.vision_preview.setStyleSheet("border: 1px solid #45475a; border-radius: 4px;")
        self.vision_preview.setTextInteractionFlags(Qt.TextSelectableByMouse)
        preview_layout.addWidget(self.vision_preview)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.btn_start_watching = QPushButton("👁️ Start Watching")
        self.btn_start_watching.setCheckable(True)
        self.btn_start_watching.clicked.connect(self._toggle_screen_watching)
        
        self.btn_single_capture = QPushButton("📷 Single Capture")
        self.btn_single_capture.clicked.connect(self._do_single_capture)
        
        btn_layout.addWidget(self.btn_start_watching)
        btn_layout.addWidget(self.btn_single_capture)
        preview_layout.addLayout(btn_layout)
        
        # Watch interval
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Update interval:"))
        self.vision_interval_spin = QSpinBox()
        self.vision_interval_spin.setRange(1, 60)
        self.vision_interval_spin.setValue(5)
        self.vision_interval_spin.setSuffix(" sec")
        interval_layout.addWidget(self.vision_interval_spin)
        interval_layout.addStretch()
        preview_layout.addLayout(interval_layout)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Analysis output
        analysis_group = QGroupBox("AI Analysis")
        analysis_layout = QVBoxLayout()
        
        self.vision_text = QPlainTextEdit()
        self.vision_text.setReadOnly(True)
        self.vision_text.setPlaceholderText("AI's description of what it sees will appear here...")
        self.vision_text.setMaximumHeight(150)
        analysis_layout.addWidget(self.vision_text)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Info
        info = QLabel(
            "<b>ℹ️ About Vision:</b><br>"
            "When watching is enabled, the AI can see your screen.<br>"
            "It will describe what it sees and can help with tasks.<br><br>"
            "<b>Note:</b> On Wayland (Pi default), you may need:<br>"
            "<code>sudo apt install scrot</code>"
        )
        info.setWordWrap(True)
        info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(info)
        
        # Timer for continuous watching
        self.vision_timer = QTimer()
        self.vision_timer.timeout.connect(self._do_continuous_capture)
        
        w.setLayout(layout)
        return w
    
    def _toggle_screen_watching(self, checked):
        """Toggle continuous screen watching."""
        if checked:
            self.btn_start_watching.setText("⏹️ Stop Watching")
            interval_ms = self.vision_interval_spin.value() * 1000
            self.vision_timer.start(interval_ms)
            self._do_continuous_capture()  # Capture immediately
        else:
            self.btn_start_watching.setText("👁️ Start Watching")
            self.vision_timer.stop()
    
    def _do_single_capture(self):
        """Do a single screen capture."""
        self._capture_and_analyze()
    
    def _do_continuous_capture(self):
        """Capture for continuous watching."""
        self._capture_and_analyze()
    
    def _capture_and_analyze(self):
        """Capture screen and analyze it."""
        try:
            # Try multiple capture methods
            img = None
            
            # Method 1: Try mss (fast, works on most systems)
            try:
                import mss
                with mss.mss() as sct:
                    monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                    screenshot = sct.grab(monitor)
                    from PIL import Image
                    img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
            except Exception:
                pass
            
            # Method 2: Try scrot (works on Wayland/Pi)
            if img is None:
                try:
                    import subprocess
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                        tmp_path = f.name
                    subprocess.run(['scrot', tmp_path], check=True, capture_output=True)
                    from PIL import Image
                    img = Image.open(tmp_path)
                    import os
                    os.unlink(tmp_path)
                except Exception:
                    pass
            
            # Method 3: Try PIL ImageGrab
            if img is None:
                try:
                    from PIL import ImageGrab
                    img = ImageGrab.grab()
                except Exception:
                    pass
            
            if img is None:
                self.vision_preview.setText(
                    "❌ Screenshot failed\n\n"
                    "Install one of:\n"
                    "• pip install mss\n"
                    "• sudo apt install scrot"
                )
                return
            
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
            
            # Simple analysis - describe what's visible
            width, height = img.size
            analysis = f"Screen: {width}x{height}\n"
            analysis += f"Captured: {datetime.now().strftime('%H:%M:%S')}\n"
            
            # Try OCR if available
            try:
                from ..tools.simple_ocr import extract_text
                text = extract_text(img)
                if text and text.strip():
                    analysis += f"\n--- Detected Text ---\n{text[:500]}"
            except Exception:
                analysis += "\n(Install pytesseract for text detection)"
            
            self.vision_text.setPlainText(analysis)
            
        except Exception as e:
            self.vision_preview.setText(f"Error: {e}")
            self.vision_text.setPlainText(f"Capture failed: {e}")
    
    def _instructions_tab(self):
        """Instructions tab - editable instructions for the AI."""
        w = QWidget()
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("AI Instructions")
        header.setObjectName("header")
        layout.addWidget(header)
        
        info = QLabel(
            "Define how your AI should behave. These instructions are loaded at startup."
        )
        info.setWordWrap(True)
        info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(info)
        
        # Instructions editor
        self.instructions_editor = QPlainTextEdit()
        self.instructions_editor.setPlaceholderText(
            "Enter instructions for your AI here...\n\n"
            "Example:\n"
            "You are a helpful assistant named [name].\n"
            "You are friendly and concise.\n"
            "You help with programming, writing, and general questions.\n"
            "You always respond in a polite manner."
        )
        
        # Load existing instructions
        self._load_instructions()
        layout.addWidget(self.instructions_editor)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        btn_save = QPushButton("💾 Save Instructions")
        btn_save.clicked.connect(self._save_instructions)
        
        btn_reset = QPushButton("🔄 Reset to Default")
        btn_reset.clicked.connect(self._reset_instructions)
        
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_reset)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Tips
        tips = QLabel(
            "<b>💡 Tips:</b><br>"
            "• Be specific about personality and tone<br>"
            "• Include what the AI should and shouldn't do<br>"
            "• These instructions are saved with your model"
        )
        tips.setWordWrap(True)
        tips.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(tips)
        
        w.setLayout(layout)
        return w
    
    def _load_instructions(self):
        """Load instructions from model folder."""
        if self.current_model_name:
            model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
            instructions_path = Path(model_info.get("path", "")) / "instructions.txt"
            if instructions_path.exists():
                self.instructions_editor.setPlainText(instructions_path.read_text())
                return
        
        # Default instructions
        default = f"""You are {self.current_model_name or 'an AI assistant'}.
You are helpful, friendly, and concise.
You assist with questions and tasks to the best of your ability.
You are honest about what you don't know."""
        self.instructions_editor.setPlainText(default)
    
    def _save_instructions(self):
        """Save instructions to model folder and optionally to GitHub."""
        if not self.current_model_name:
            QMessageBox.warning(self, "No Model", "Load a model first")
            return
        
        model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
        instructions_path = Path(model_info.get("path", "")) / "instructions.txt"
        
        try:
            instructions_path.write_text(self.instructions_editor.toPlainText())
            QMessageBox.information(self, "Saved", f"Instructions saved to {instructions_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def _reset_instructions(self):
        """Reset instructions to default."""
        reply = QMessageBox.question(
            self, "Reset Instructions",
            "Reset to default instructions? This will overwrite current text.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            default = f"""You are {self.current_model_name or 'an AI assistant'}.
You are helpful, friendly, and concise.
You assist with questions and tasks to the best of your ability.
You are honest about what you don't know."""
            self.instructions_editor.setPlainText(default)
    
    # === Data Editor Actions ===
    
    def _refresh_data_files(self):
        """Refresh list of data files - shows AI's own data first."""
        self.data_file_combo.clear()
        
        # First, add current AI's data files if one is loaded
        if self.current_model_name:
            model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
            model_data_dir = model_info.get("data_dir") or (Path(model_info.get("path", "")) / "data")
            if isinstance(model_data_dir, str):
                model_data_dir = Path(model_data_dir)
            
            if model_data_dir.exists():
                for f in model_data_dir.glob("*.txt"):
                    self.data_file_combo.addItem(f"📌 {self.current_model_name}: {f.name}", str(f))
        
        # Add separator if we have AI files
        if self.data_file_combo.count() > 0:
            self.data_file_combo.insertSeparator(self.data_file_combo.count())
        
        # Then add global data files
        data_dir = Path(CONFIG.get("data_dir", "data"))
        data_dir.mkdir(parents=True, exist_ok=True)
        
        for f in data_dir.glob("*.txt"):
            self.data_file_combo.addItem(f"📁 Global: {f.name}", str(f))
    
    def _load_data_file(self, index):
        """Load a data file into editor."""
        if index < 0:
            return
        filepath = self.data_file_combo.itemData(index)
        if not filepath:
            # Fallback for old format
            filename = self.data_file_combo.currentText()
            if not filename or filename.startswith("---"):
                return
            data_dir = Path(CONFIG.get("data_dir", "data"))
            filepath = str(data_dir / filename)
        
        try:
            self.data_editor.setPlainText(Path(filepath).read_text())
            self._current_data_file = filepath
        except Exception as e:
            self.data_editor.setPlainText(f"Error loading file: {e}")
    
    def _save_data_file(self):
        """Save current editor content."""
        if not hasattr(self, '_current_data_file') or not self._current_data_file:
            QMessageBox.warning(self, "No File", "Select or create a file first")
            return
        
        filepath = Path(self._current_data_file)
        try:
            filepath.write_text(self.data_editor.toPlainText())
            QMessageBox.information(self, "Saved", f"Saved to {filepath.name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def _create_data_file(self):
        """Create a new data file in the current AI's folder."""
        from PyQt5.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New File", "Filename (without .txt):")
        if ok and name:
            try:
                name = name.strip().replace(" ", "_")
                if not name.endswith(".txt"):
                    name += ".txt"
                
                # Save to AI's data folder if available, else global
                if self.current_model_name:
                    model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
                    data_dir = Path(model_info.get("path", "")) / "data"
                else:
                    data_dir = Path(CONFIG.get("data_dir", "data"))
                
                data_dir.mkdir(parents=True, exist_ok=True)
                filepath = data_dir / name
                
                ai_name = self.current_model_name or "Assistant"
                filepath.write_text(f"# Training data for {ai_name}\n# Add your Q&A pairs or conversations below\n\n")
                
                self._refresh_data_files()
                # Select the new file
                for i in range(self.data_file_combo.count()):
                    if self.data_file_combo.itemData(i) == str(filepath):
                        self.data_file_combo.setCurrentIndex(i)
                        break
                QMessageBox.information(self, "Created", f"Created {name} in {data_dir}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to create file: {e}")
    
    def _use_data_for_training(self):
        """Set current file as training data."""
        if not hasattr(self, '_current_data_file') or not self._current_data_file:
            QMessageBox.warning(self, "No File", "Select a data file first")
            return
        
        self.training_data_path = self._current_data_file
        filename = Path(self._current_data_file).name
        self.data_path_label.setText(f"Selected: {filename}")
        QMessageBox.information(self, "Ready", f"'{filename}' selected for training.\nGo to Training tab to start.")
    
    def _import_data_file(self):
        """Import an external file into the AI's data folder."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Training Data", "", "Text Files (*.txt);;All Files (*)"
        )
        if filepath:
            try:
                # Determine destination
                if self.current_model_name:
                    model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
                    dest_dir = Path(model_info.get("path", "")) / "data"
                else:
                    dest_dir = Path(CONFIG.get("data_dir", "data"))
                
                dest_dir.mkdir(parents=True, exist_ok=True)
                src = Path(filepath)
                dest = dest_dir / src.name
                
                # Copy content
                dest.write_text(src.read_text())
                
                self._refresh_data_files()
                QMessageBox.information(self, "Imported", f"Imported {src.name} to {dest_dir}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to import: {e}")
    
    # === Avatar Actions ===
    
    def _refresh_avatar_status(self):
        """Update avatar status display."""
        try:
            from ..avatar.avatar_api import AvatarController
            self.avatar = AvatarController()
            self.avatar_status_label.setText("✅ Initialized")
            self.avatar_state_label.setText(self.avatar.state.get("status", "unknown"))
        except Exception as e:
            self.avatar_status_label.setText(f"❌ Error: {e}")
            self.avatar = None
        
        # Try to load avatar image if one exists
        self._load_default_avatar()
    
    def _load_default_avatar(self):
        """Try to load avatar image from model folder or avatar folder."""
        avatar_paths = []
        
        # Check model-specific avatar
        if self.current_model_name:
            model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
            model_path = Path(model_info.get("path", ""))
            avatar_paths.extend([
                model_path / "avatar.png",
                model_path / "avatar.jpg",
                model_path / "avatar.jpeg",
            ])
        
        # Check global avatar folder
        avatar_dir = Path("avatar")
        avatar_paths.extend([
            avatar_dir / "default.png",
            avatar_dir / "avatar.png",
        ])
        
        for path in avatar_paths:
            if path.exists():
                self._display_avatar_image(str(path))
                return
    
    def _load_avatar_image(self):
        """Load a custom avatar image."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Avatar Image", "", "Images (*.png *.jpg *.jpeg *.gif);;All Files (*)"
        )
        if filepath:
            # Copy to model folder if we have a current model
            if self.current_model_name:
                try:
                    model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
                    model_path = Path(model_info.get("path", ""))
                    dest = model_path / f"avatar{Path(filepath).suffix}"
                    
                    import shutil
                    shutil.copy(filepath, dest)
                    filepath = str(dest)
                    QMessageBox.information(self, "Saved", f"Avatar saved to {dest}")
                except Exception as e:
                    QMessageBox.warning(self, "Warning", f"Couldn't save to model folder: {e}")
            
            self._display_avatar_image(filepath)
    
    def _load_avatar_model(self):
        """Load an avatar model (Live2D, VRM, etc.)."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Avatar Model Folder",
            "", QFileDialog.ShowDirsOnly
        )
        if folder:
            if self.current_model_name:
                try:
                    model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
                    model_path = Path(model_info.get("path", ""))
                    avatar_dir = model_path / "avatar"
                    avatar_dir.mkdir(exist_ok=True)
                    
                    # Copy all files from selected folder
                    src_folder = Path(folder)
                    for item in src_folder.iterdir():
                        dest = avatar_dir / item.name
                        if item.is_file():
                            shutil.copy(item, dest)
                        elif item.is_dir():
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.copytree(item, dest)
                    
                    QMessageBox.information(self, "Imported", f"Avatar model imported to {avatar_dir}")
                    
                    # Try to find and display preview image
                    for ext in ['.png', '.jpg', '.jpeg']:
                        preview = avatar_dir / f"preview{ext}"
                        if preview.exists():
                            self._display_avatar_image(str(preview))
                            break
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to import: {e}")
            else:
                QMessageBox.warning(self, "No Model", "Load an AI model first")
    
    def _display_avatar_image(self, filepath):
        """Display an avatar image."""
        pixmap = QPixmap(filepath)
        if not pixmap.isNull():
            scaled = pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.avatar_image_label.setPixmap(scaled)
            self.avatar_image_label.setStyleSheet("border: 2px solid #89b4fa; border-radius: 8px;")
        else:
            self.avatar_image_label.setText("Failed to load image")
    
    def _enable_avatar(self):
        if self.avatar:
            self.avatar.enable()
            self._refresh_avatar_status()
    
    def _disable_avatar(self):
        if self.avatar:
            self.avatar.disable()
            self._refresh_avatar_status()
    
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
                        "⚠️ Screenshot appears black\\n\\n"
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
            status = "✓" if info.get("has_weights") else "○"
            current = " ← ACTIVE" if name == self.current_model_name else ""
            self.models_list.addItem(f"{status} {name} ({info.get('size', '?')}){current}")
    
    # === Actions ===
    
    def _on_send(self):
        text = self.chat_input.text().strip()
        if not text or not self.engine:
            return
        
        self.chat_display.append(f"<b>You:</b> {text}")
        self.chat_input.clear()
        
        try:
            response = self.engine.generate(text, max_gen=50)
            self.chat_display.append(f"<b>{self.current_model_name}:</b> {response}")
            self.last_response = response
        except Exception as e:
            self.chat_display.append(f"<i>Error: {e}</i>")
    
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
                self.model_status_btn.setText(f"Model: {self.current_model_name}  ▼")
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
        
        if not hasattr(self, 'training_data_path'):
            QMessageBox.warning(self, "No Data", "Select training data first")
            return
        
        # This should run in a thread - simplified version here
        self.train_status.setText("Training... (UI may freeze)")
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
            
            trainer.train(epochs=self.epochs_spin.value())
            
            # Reload model
            self._load_current_model()
            
            self.train_status.setText("Training complete!")
            QMessageBox.information(self, "Done", "Training finished!")
        except Exception as e:
            self.train_status.setText(f"Error: {e}")
            QMessageBox.warning(self, "Training Error", str(e))


def run_app():
    """Run the enhanced GUI application."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = EnhancedMainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
