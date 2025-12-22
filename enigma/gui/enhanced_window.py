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
        self.resize(600, 400)
        
        # Add pages
        self.addPage(self._create_welcome_page())
        self.addPage(self._create_name_page())
        self.addPage(self._create_size_page())
        self.addPage(self._create_confirm_page())
        
        self.model_name = None
        self.model_size = "small"
    
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
        self.name_input.setPlaceholderText("e.g., enigma, artemis, apollo...")
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
        
        sizes = [
            ("tiny", "Tiny - Testing only (~9M params)", "Raspberry Pi, any laptop", "1GB RAM"),
            ("small", "Small - Learning (~21M params)", "RTX 2080 or similar", "4GB VRAM"),
            ("medium", "Medium - Real use (~58M params)", "RTX 2080 (tight fit)", "6GB VRAM"),
            ("large", "Large - Serious (~134M params)", "RTX 3080/3090", "10GB VRAM"),
        ]
        
        for i, (size_id, name, hw, mem) in enumerate(sizes):
            radio = QRadioButton(f"{name}\n    Hardware: {hw} | Memory: {mem}")
            radio.size_id = size_id
            self.size_group.addButton(radio, i)
            layout.addWidget(radio)
            
            if size_id == "small":
                radio.setChecked(True)
        
        # Hardware note
        note = QLabel("""
        <p><b>Your hardware:</b> RTX 2080 (8GB) → Recommended: <b>Small</b> or <b>Medium</b></p>
        <p><i>You can always grow your model later as you get better hardware!</i></p>
        """)
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
        file_menu.addAction("Exit", self.close)
        
        model_menu = menubar.addMenu("Model")
        model_menu.addAction("Model Manager...", self._on_model_manager)
        model_menu.addAction("Backup Current", self._on_backup_current)
        model_menu.addSeparator()
        model_menu.addAction("Grow Model...", self._on_grow_current)
        model_menu.addAction("Shrink Model...", self._on_shrink_current)
        
        # View menu with dark mode toggle
        view_menu = menubar.addMenu("View")
        self.dark_mode_action = view_menu.addAction("🌙 Dark Mode")
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setChecked(True)  # Default to dark mode
        self.dark_mode_action.triggered.connect(self._toggle_dark_mode)
        
        # Status bar
        self.statusBar().showMessage(f"Model: {self.current_model_name or 'None'}")
        
        # Apply dark mode by default
        self.setStyleSheet(DARK_STYLE)
        
        # Main tabs
        tabs = QTabWidget()
        tabs.addTab(self._chat_tab(), "💬 Chat")
        tabs.addTab(self._training_tab(), "🎓 Training")
        tabs.addTab(self._data_editor_tab(), "📝 Data Editor")
        tabs.addTab(self._avatar_tab(), "🤖 Avatar")
        tabs.addTab(self._vision_tab(), "👁️ Vision")
        tabs.addTab(self._models_tab(), "📦 Models")
        
        self.setCentralWidget(tabs)
    
    def _toggle_dark_mode(self, checked):
        """Toggle between dark and light themes."""
        if checked:
            self.setStyleSheet(DARK_STYLE)
            self.dark_mode_action.setText("🌙 Dark Mode")
        else:
            self.setStyleSheet(LIGHT_STYLE)
            self.dark_mode_action.setText("☀️ Light Mode")
        
        self.setCentralWidget(tabs)
    
    def _chat_tab(self):
        """Chat interface tab."""
        w = QWidget()
        layout = QVBoxLayout()
        
        # Model indicator
        self.model_label = QLabel(f"<b>Active Model:</b> {self.current_model_name or 'None'}")
        layout.addWidget(self.model_label)
        
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
        """Training controls tab."""
        w = QWidget()
        layout = QVBoxLayout()
        
        # Training data
        data_group = QGroupBox("Training Data")
        data_layout = QVBoxLayout()
        
        self.data_path_label = QLabel("No data file selected")
        btn_select_data = QPushButton("📂 Select Training Data...")
        btn_select_data.clicked.connect(self._on_select_data)
        
        data_layout.addWidget(self.data_path_label)
        data_layout.addWidget(btn_select_data)
        
        # Quick tip
        tip_label = QLabel("<i>💡 Tip: Use the Data Editor tab to create/edit training files</i>")
        tip_label.setWordWrap(True)
        data_layout.addWidget(tip_label)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Training params with tooltips
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout()
        
        # Epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(10)
        self.epochs_spin.setToolTip(
            "<b>Epochs</b><br>"
            "Number of times to go through ALL training data.<br><br>"
            "• <b>More epochs</b> = Better learning, but takes longer<br>"
            "• <b>Too many</b> = Overfitting (memorizes instead of learning)<br><br>"
            "Start with 5-10, increase if responses are poor."
        )
        epochs_label = QLabel("Epochs:")
        epochs_label.setToolTip(self.epochs_spin.toolTip())
        
        # Batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(4)
        self.batch_spin.setToolTip(
            "<b>Batch Size</b><br>"
            "How many examples to process at once.<br><br>"
            "• <b>Larger batch</b> = Faster training, needs more VRAM<br>"
            "• <b>Smaller batch</b> = Slower, but uses less memory<br><br>"
            "Raspberry Pi: Use 1-2<br>"
            "GPU (8GB): Use 4-8<br>"
            "GPU (24GB): Use 16-32"
        )
        batch_label = QLabel("Batch Size:")
        batch_label.setToolTip(self.batch_spin.toolTip())
        
        # Learning rate
        self.lr_input = QLineEdit("0.0001")
        self.lr_input.setToolTip(
            "<b>Learning Rate</b><br>"
            "How fast the AI adjusts its knowledge.<br><br>"
            "• <b>Too high</b> = Unstable, AI 'forgets' things<br>"
            "• <b>Too low</b> = Very slow learning<br><br>"
            "Default 0.0001 is usually good.<br>"
            "If training loss stays high, try 0.0003<br>"
            "If training is unstable, try 0.00003"
        )
        lr_label = QLabel("Learning Rate:")
        lr_label.setToolTip(self.lr_input.toolTip())
        
        params_layout.addRow(epochs_label, self.epochs_spin)
        params_layout.addRow(batch_label, self.batch_spin)
        params_layout.addRow(lr_label, self.lr_input)
        
        # Help text
        help_label = QLabel("💡 <i>Hover over each parameter name for explanation</i>")
        params_layout.addRow(help_label)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Progress
        self.train_progress = QProgressBar()
        self.train_status = QLabel("Ready")
        layout.addWidget(self.train_progress)
        layout.addWidget(self.train_status)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_train = QPushButton("Start Training")
        self.btn_train.clicked.connect(self._on_start_training)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        
        btn_layout.addWidget(self.btn_train)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)
        
        layout.addStretch()
        w.setLayout(layout)
        return w
    
    def _models_tab(self):
        """Models overview tab."""
        w = QWidget()
        layout = QVBoxLayout()
        
        btn_manager = QPushButton("Open Model Manager")
        btn_manager.clicked.connect(self._on_model_manager)
        layout.addWidget(btn_manager)
        
        # Quick list
        layout.addWidget(QLabel("<b>Registered Models:</b>"))
        self.models_list = QListWidget()
        self._refresh_models_list()
        layout.addWidget(self.models_list)
        
        w.setLayout(layout)
        return w
    
    def _data_editor_tab(self):
        """Data editor for training files."""
        w = QWidget()
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("Training Data Editor")
        header.setObjectName("header")
        layout.addWidget(header)
        
        # Info about current AI
        self.data_editor_ai_label = QLabel(f"<b>Current AI:</b> {self.current_model_name or 'None'}")
        layout.addWidget(self.data_editor_ai_label)
        
        # File selection
        file_layout = QHBoxLayout()
        self.data_file_combo = QComboBox()
        self.data_file_combo.setMinimumWidth(300)
        self._refresh_data_files()
        self.data_file_combo.currentIndexChanged.connect(self._load_data_file)
        
        btn_refresh = QPushButton("🔄")
        btn_refresh.setMaximumWidth(40)
        btn_refresh.clicked.connect(self._refresh_data_files)
        
        btn_new_file = QPushButton("➕ New File")
        btn_new_file.clicked.connect(self._create_data_file)
        
        file_layout.addWidget(QLabel("File:"))
        file_layout.addWidget(self.data_file_combo)
        file_layout.addWidget(btn_refresh)
        file_layout.addWidget(btn_new_file)
        file_layout.addStretch()
        layout.addLayout(file_layout)
        
        # Editor
        self.data_editor = QPlainTextEdit()
        self.data_editor.setPlaceholderText(
            "Enter training data here...\n\n"
            "FORMAT OPTIONS:\n"
            "1. Plain text (AI learns patterns)\n"
            "2. Q&A format:\n"
            "   Q: What is your name?\n"
            "   A: My name is [your AI's name]\n\n"
            "3. Conversation format:\n"
            "   User: Hello\n"
            "   Assistant: Hello! How can I help?\n\n"
            "The more examples, the better the AI learns!"
        )
        layout.addWidget(self.data_editor)
        
        # Save button
        btn_layout = QHBoxLayout()
        self.btn_save_data = QPushButton("💾 Save File")
        self.btn_save_data.clicked.connect(self._save_data_file)
        
        btn_use_for_training = QPushButton("📚 Use for Training")
        btn_use_for_training.clicked.connect(self._use_data_for_training)
        
        btn_layout.addWidget(self.btn_save_data)
        btn_layout.addWidget(btn_use_for_training)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Tips
        tips = QLabel(
            "<b>💡 Tips:</b><br>"
            "• More diverse examples = smarter AI<br>"
            "• Include many variations of common questions<br>"
            "• Add personality through response style<br>"
            "• Save before training!"
        )
        tips.setWordWrap(True)
        layout.addWidget(tips)
        
        w.setLayout(layout)
        return w
    
    def _avatar_tab(self):
        """Avatar control panel."""
        w = QWidget()
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("Avatar Control")
        header.setObjectName("header")
        layout.addWidget(header)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QFormLayout()
        
        self.avatar_status_label = QLabel("Not initialized")
        self.avatar_state_label = QLabel("Unknown")
        
        status_layout.addRow("Status:", self.avatar_status_label)
        status_layout.addRow("State:", self.avatar_state_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Controls
        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout()
        
        # Enable/disable
        btn_row1 = QHBoxLayout()
        self.btn_avatar_enable = QPushButton("✅ Enable Avatar")
        self.btn_avatar_enable.clicked.connect(self._enable_avatar)
        self.btn_avatar_disable = QPushButton("❌ Disable Avatar")
        self.btn_avatar_disable.clicked.connect(self._disable_avatar)
        btn_row1.addWidget(self.btn_avatar_enable)
        btn_row1.addWidget(self.btn_avatar_disable)
        ctrl_layout.addLayout(btn_row1)
        
        # Expressions
        expr_layout = QHBoxLayout()
        expr_layout.addWidget(QLabel("Expression:"))
        self.avatar_expr_combo = QComboBox()
        self.avatar_expr_combo.addItems(["neutral", "happy", "sad", "thinking", "surprised"])
        self.avatar_expr_combo.currentTextChanged.connect(self._set_avatar_expression)
        expr_layout.addWidget(self.avatar_expr_combo)
        ctrl_layout.addLayout(expr_layout)
        
        # Speak test
        speak_layout = QHBoxLayout()
        self.avatar_speak_input = QLineEdit()
        self.avatar_speak_input.setPlaceholderText("Enter text for avatar to speak...")
        btn_speak = QPushButton("🔊 Speak")
        btn_speak.clicked.connect(self._avatar_speak)
        speak_layout.addWidget(self.avatar_speak_input)
        speak_layout.addWidget(btn_speak)
        ctrl_layout.addLayout(speak_layout)
        
        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)
        
        # Info
        info = QLabel(
            "<b>ℹ️ About Avatar:</b><br>"
            "The avatar is a visual representation of your AI.<br>"
            "It can display expressions and speak using TTS.<br><br>"
            "To use a custom avatar model:<br>"
            "1. Place model files in avatar/ folder<br>"
            "2. Update avatar_api.py with your renderer"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        layout.addStretch()
        
        # Initialize avatar status
        self._refresh_avatar_status()
        
        w.setLayout(layout)
        return w
    
    def _vision_tab(self):
        """Screen vision preview."""
        w = QWidget()
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("Screen Vision")
        header.setObjectName("header")
        layout.addWidget(header)
        
        # Preview area
        preview_group = QGroupBox("Screen Preview")
        preview_layout = QVBoxLayout()
        
        self.vision_preview = QLabel("Click 'Capture' to see what the AI sees")
        self.vision_preview.setMinimumHeight(300)
        self.vision_preview.setAlignment(Qt.AlignCenter)
        self.vision_preview.setStyleSheet("border: 1px solid #45475a; border-radius: 4px;")
        preview_layout.addWidget(self.vision_preview)
        
        # Capture button
        btn_capture = QPushButton("📷 Capture Screen")
        btn_capture.clicked.connect(self._capture_screen)
        preview_layout.addWidget(btn_capture)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # OCR/Analysis
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout()
        
        self.vision_text = QPlainTextEdit()
        self.vision_text.setReadOnly(True)
        self.vision_text.setPlaceholderText("OCR text and analysis will appear here...")
        self.vision_text.setMaximumHeight(150)
        analysis_layout.addWidget(self.vision_text)
        
        btn_analyze = QPushButton("🔍 Analyze Screen")
        btn_analyze.clicked.connect(self._analyze_screen)
        analysis_layout.addWidget(btn_analyze)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Info
        info = QLabel(
            "<b>ℹ️ About Vision:</b><br>"
            "Your AI can 'see' the screen through screenshots.<br>"
            "OCR extracts text from images for understanding.<br><br>"
            "Use cases:<br>"
            "• Read documents on screen<br>"
            "• Navigate applications<br>"
            "• Find buttons/text"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        w.setLayout(layout)
        return w
    
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
        """Create a new data file."""
        from PyQt5.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New File", "Filename (without .txt):")
        if ok and name:
            name = name.strip().replace(" ", "_")
            if not name.endswith(".txt"):
                name += ".txt"
            data_dir = Path(CONFIG.get("data_dir", "data"))
            filepath = data_dir / name
            filepath.write_text("# Training data for your AI\n\n")
            self._refresh_data_files()
            self.data_file_combo.setCurrentText(name)
    
    def _use_data_for_training(self):
        """Set current file as training data."""
        filename = self.data_file_combo.currentText()
        if not filename:
            return
        data_dir = Path(CONFIG.get("data_dir", "data"))
        self.training_data_path = str(data_dir / filename)
        self.data_path_label.setText(f"Selected: {filename}")
        QMessageBox.information(self, "Ready", f"'{filename}' selected for training.\nGo to Training tab to start.")
    
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
    
    def _enable_avatar(self):
        if self.avatar:
            self.avatar.enable()
            self._refresh_avatar_status()
    
    def _disable_avatar(self):
        if self.avatar:
            self.avatar.disable()
            self._refresh_avatar_status()
    
    def _set_avatar_expression(self, expression):
        if self.avatar:
            self.avatar.set_expression(expression)
    
    def _avatar_speak(self):
        text = self.avatar_speak_input.text().strip()
        if text and self.avatar:
            self.avatar.speak(text)
            self.avatar_speak_input.clear()
    
    # === Vision Actions ===
    
    def _capture_screen(self):
        """Capture and display screen."""
        try:
            from ..tools.vision import ScreenCapture
            capture = ScreenCapture()
            img = capture.capture()
            
            if img:
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
            else:
                self.vision_preview.setText("Failed to capture screen")
        except Exception as e:
            self.vision_preview.setText(f"Error: {e}")
    
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
                self._refresh_models_list()  # Update arrow indicator
                self.model_label.setText(f"<b>Active Model:</b> {self.current_model_name}")
                self.model_label.setText(f"<b>Active Model:</b> {self.current_model_name}")
                self.statusBar().showMessage(f"Model: {self.current_model_name}")
                self._refresh_models_list()
    
    def _on_model_manager(self):
        dialog = ModelManagerDialog(self.registry, self.current_model_name, self)
        dialog.exec_()
        self._refresh_models_list()
    
    def _on_backup_current(self):
        if not self.current_model_name:
            return
        
        model_dir = Path(self.registry.models_dir) / self.current_model_name
        backup_dir = Path(self.registry.models_dir) / f"{self.current_model_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            shutil.copytree(model_dir, backup_dir)
            QMessageBox.information(self, "Backup Complete", f"Backed up to:\n{backup_dir}")
        except Exception as e:
            QMessageBox.warning(self, "Backup Failed", str(e))
    
    def _on_grow_current(self):
        QMessageBox.information(self, "Grow", "Use Model Manager to grow models")
        self._on_model_manager()
    
    def _on_shrink_current(self):
        QMessageBox.information(self, "Shrink", "Use Model Manager to shrink models")
        self._on_model_manager()
    
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
