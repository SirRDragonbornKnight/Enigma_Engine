"""Train tab for Enigma Engine GUI."""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QLineEdit, QProgressBar, QComboBox, QFileDialog,
    QPlainTextEdit, QMessageBox
)

from ...config import CONFIG


def create_training_tab(parent):
    """Create the train tab with model training controls."""
    w = QWidget()
    layout = QVBoxLayout()
    
    # Header
    header = QLabel("Train Your AI")
    header.setObjectName("header")
    layout.addWidget(header)
    
    # Current model info
    parent.training_model_label = QLabel("No model loaded")
    parent.training_model_label.setStyleSheet("color: #89b4fa; font-weight: bold;")
    layout.addWidget(parent.training_model_label)
    
    # Training data file selector
    file_layout = QHBoxLayout()
    file_layout.addWidget(QLabel("Training File:"))
    
    parent.training_file_combo = QComboBox()
    parent.training_file_combo.setMinimumWidth(150)
    parent.training_file_combo.currentIndexChanged.connect(lambda idx: _load_training_file(parent, idx))
    file_layout.addWidget(parent.training_file_combo)
    
    btn_browse = QPushButton("Browse...")
    btn_browse.clicked.connect(lambda: _browse_training_file(parent))
    file_layout.addWidget(btn_browse)
    
    btn_save = QPushButton("Save")
    btn_save.clicked.connect(lambda: _save_training_file(parent))
    file_layout.addWidget(btn_save)
    
    file_layout.addStretch()
    layout.addLayout(file_layout)
    
    # File content editor
    parent.training_editor = QPlainTextEdit()
    parent.training_editor.setPlaceholderText("Training data will appear here...\n\nFormat:\nQ: Question?\nA: Answer.\n\nOr plain text for the AI to learn from.")
    layout.addWidget(parent.training_editor, stretch=1)
    
    # Parameters row (inline)
    params_layout = QHBoxLayout()
    
    params_layout.addWidget(QLabel("Epochs:"))
    parent.epochs_spin = QSpinBox()
    parent.epochs_spin.setRange(1, 10000)
    parent.epochs_spin.setValue(10)
    parent.epochs_spin.setToolTip("How many times to go through all data")
    parent.epochs_spin.setMaximumWidth(80)
    params_layout.addWidget(parent.epochs_spin)
    
    params_layout.addWidget(QLabel("Batch:"))
    parent.batch_spin = QSpinBox()
    parent.batch_spin.setRange(1, 64)
    parent.batch_spin.setValue(4)
    parent.batch_spin.setToolTip("Examples per step (Pi: 1-2, GPU: 4-16)")
    parent.batch_spin.setMaximumWidth(60)
    params_layout.addWidget(parent.batch_spin)
    
    params_layout.addWidget(QLabel("LR:"))
    parent.lr_input = QLineEdit("0.0001")
    parent.lr_input.setToolTip("How fast AI learns (lower = slower but stable)")
    parent.lr_input.setMaximumWidth(80)
    params_layout.addWidget(parent.lr_input)
    
    params_layout.addStretch()
    layout.addLayout(params_layout)
    
    # Progress bar
    parent.train_progress = QProgressBar()
    parent.train_progress.setValue(0)
    layout.addWidget(parent.train_progress)
    
    # Train and Stop buttons row
    btn_layout = QHBoxLayout()
    
    parent.btn_train = QPushButton("Train")
    parent.btn_train.clicked.connect(parent._on_start_training)
    btn_layout.addWidget(parent.btn_train)
    
    parent.btn_stop_train = QPushButton("Stop")
    parent.btn_stop_train.setToolTip("Stop training after current epoch")
    parent.btn_stop_train.clicked.connect(parent._on_stop_training)
    parent.btn_stop_train.setEnabled(False)
    parent.btn_stop_train.setStyleSheet("background-color: #dc2626;")
    btn_layout.addWidget(parent.btn_stop_train)
    
    layout.addLayout(btn_layout)
    
    # Hidden data path label (for compatibility)
    parent.data_path_label = QLabel("")
    parent.data_path_label.setVisible(False)
    layout.addWidget(parent.data_path_label)
    
    # Populate training file list and load first file
    _refresh_training_files(parent)
    
    w.setLayout(layout)
    return w


def _refresh_training_files(parent):
    """Populate the training file dropdown."""
    parent.training_file_combo.blockSignals(True)
    parent.training_file_combo.clear()
    
    # Always use global data directory for training files
    global_data_dir = Path(CONFIG.get("data_dir", "data"))
    global_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Also check model-specific data dir if a model is loaded
    model_data_dir = None
    if parent.current_model_name:
        model_info = parent.registry.registry.get("models", {}).get(parent.current_model_name, {})
        model_data_dir = model_info.get("data_dir") or (Path(model_info.get("path", "")) / "data")
        if isinstance(model_data_dir, str):
            model_data_dir = Path(model_data_dir)
        if model_data_dir.exists():
            model_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure training.txt exists in global data dir
    training_file = global_data_dir / "training.txt"
    if not training_file.exists():
        training_file.write_text("# Training Data\n\nQ: Hello\nA: Hi there!\n")
    
    # Collect all txt files from both directories
    seen_files = set()
    all_files = []
    
    # Add files from global data directory first
    for f in sorted(global_data_dir.glob("*.txt")):
        if f.name not in seen_files:
            seen_files.add(f.name)
            all_files.append((f.name, str(f)))
    
    # Add model-specific files (if different directory)
    if model_data_dir and model_data_dir != global_data_dir and model_data_dir.exists():
        for f in sorted(model_data_dir.glob("*.txt")):
            if f.name not in seen_files:
                seen_files.add(f.name)
                all_files.append((f"[Model] {f.name}", str(f)))
    
    # Add all files to combo
    for name, path in all_files:
        parent.training_file_combo.addItem(name, path)
    
    # Select training.txt by default
    idx = parent.training_file_combo.findText("training.txt")
    if idx >= 0:
        parent.training_file_combo.setCurrentIndex(idx)
    
    parent.training_file_combo.blockSignals(False)
    
    # Set default path and load content
    parent.training_data_path = str(training_file)
    parent.data_path_label.setText(str(training_file))
    _load_training_file(parent, parent.training_file_combo.currentIndex())


def _load_training_file(parent, index):
    """Load a training file into the editor."""
    if index < 0:
        return
    
    filepath = parent.training_file_combo.itemData(index)
    if filepath and Path(filepath).exists():
        try:
            content = Path(filepath).read_text(encoding='utf-8', errors='replace')
            parent.training_editor.setPlainText(content)
            parent.training_data_path = filepath
            parent.data_path_label.setText(filepath)
            parent._current_training_file = filepath
        except Exception as e:
            parent.training_editor.setPlainText(f"Error loading file: {e}")


def _save_training_file(parent):
    """Save the current training file."""
    if not hasattr(parent, '_current_training_file') or not parent._current_training_file:
        QMessageBox.warning(parent, "No File", "Select a file first")
        return
    
    try:
        content = parent.training_editor.toPlainText()
        Path(parent._current_training_file).write_text(content, encoding='utf-8')
        QMessageBox.information(parent, "Saved", "Training file saved!")
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Failed to save: {e}")


def _browse_training_file(parent):
    """Browse for a training file."""
    filepath, _ = QFileDialog.getOpenFileName(
        parent, "Select Training File", "", "Text Files (*.txt);;All Files (*)"
    )
    if filepath:
        parent.training_data_path = filepath
        parent.data_path_label.setText(filepath)
        # Add to combo if not there
        name = Path(filepath).name
        if parent.training_file_combo.findText(name) < 0:
            parent.training_file_combo.addItem(name, filepath)
        idx = parent.training_file_combo.findData(filepath)
        if idx >= 0:
            parent.training_file_combo.setCurrentIndex(idx)
