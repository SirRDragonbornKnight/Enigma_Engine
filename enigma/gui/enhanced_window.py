"""
Enhanced PyQt5 GUI for Enigma with Setup Wizard

Features:
  - First-run setup wizard to create/name your AI
  - Model selection and management
  - Backup before risky operations
  - Grow/shrink models with confirmation
  - Chat, Training, Voice integration
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
    QSpinBox, QCheckBox, QDialogButtonBox, QWizard, QWizardPage, QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import time

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
        
        # Status bar
        self.statusBar().showMessage(f"Model: {self.current_model_name or 'None'}")
        
        # Main tabs
        tabs = QTabWidget()
        tabs.addTab(self._chat_tab(), "Chat")
        tabs.addTab(self._training_tab(), "Training")
        tabs.addTab(self._models_tab(), "Models")
        
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
        btn_select_data = QPushButton("Select Training Data...")
        btn_select_data.clicked.connect(self._on_select_data)
        
        data_layout.addWidget(self.data_path_label)
        data_layout.addWidget(btn_select_data)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Training params
        params_group = QGroupBox("Training Parameters")
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
