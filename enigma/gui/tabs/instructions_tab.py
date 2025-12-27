"""Files tab for Enigma Engine GUI - edit model data files."""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QPlainTextEdit, QMessageBox, QInputDialog
)
from PyQt5.QtCore import Qt

from ...config import CONFIG

# Default instructions content
DEFAULT_INSTRUCTIONS = """# Enigma Engine - Quick Start Guide

## Getting Started
1. Create a Model: File -> New Model
2. Add Training Data: Go to Train tab, select a file
3. Train: Set parameters and click Train
4. Chat: Talk to your AI in the Chat tab!

## Training Data Format
Your AI learns from conversations. Use these formats:

Q: What is your name?
A: I'm your AI assistant.

User: Hello!
AI: Hi there! How can I help?

More data = smarter AI. Add lots of examples!

## Training Parameters
- Epochs: Times through data (10-50 to start)
- Batch: Pi: 1-2, GPU: 4-16
- LR: Learning rate (0.0001 is usually good)

## Model Sizes
- Tiny: ~0.5M params - Works on any device
- Small: ~10M params - Good for Pi 4GB+
- Medium: ~50M params - Needs 8GB+ RAM
- Large: ~150M params - Needs GPU 8GB+

## Avatar
Load an image or model in the Avatar tab.
Enable display via Options -> Avatar

## Vision
Take screenshots for the AI to analyze.
On Raspberry Pi, install: sudo apt install scrot

## Voice Features
- Options -> AI Auto-Speak: AI speaks responses
- Options -> Microphone: Voice input

## Tips
- Back up your model before making big changes
- Train in small batches - you can always train more
- The more diverse your training data, the better
- Check the models/ folder for your AI's files
"""


def create_instructions_tab(parent):
    """Create the files tab - edit model data and notes."""
    w = QWidget()
    layout = QVBoxLayout()
    
    # Header
    header = QLabel("Model Files")
    header.setObjectName("header")
    layout.addWidget(header)
    
    # File selector row
    file_layout = QHBoxLayout()
    file_layout.addWidget(QLabel("File:"))
    
    parent.instructions_file_combo = QComboBox()
    parent.instructions_file_combo.setMinimumWidth(200)
    parent.instructions_file_combo.currentIndexChanged.connect(
        lambda idx: _load_instructions_file(parent, idx)
    )
    
    btn_new = QPushButton("[+] New")
    btn_new.setToolTip("Create new notes file")
    btn_new.clicked.connect(lambda: _create_instructions_file(parent))
    
    btn_save = QPushButton("[S] Save")
    btn_save.setToolTip("Save current file")
    btn_save.clicked.connect(lambda: _save_instructions_file(parent))
    
    file_layout.addWidget(parent.instructions_file_combo)
    file_layout.addWidget(btn_new)
    file_layout.addWidget(btn_save)
    layout.addLayout(file_layout)
    
    # Editor
    parent.instructions_editor = QPlainTextEdit()
    parent.instructions_editor.setPlaceholderText("Select a file above...")
    layout.addWidget(parent.instructions_editor, stretch=1)
    
    # Refresh files list - instructions.txt shown first
    _refresh_instructions_files(parent)
    
    w.setLayout(layout)
    return w


def _refresh_instructions_files(parent):
    """Refresh list of instruction/notes files."""
    parent.instructions_file_combo.clear()
    
    # Get data directory
    if parent.current_model_name:
        model_info = parent.registry.registry.get("models", {}).get(parent.current_model_name, {})
        data_dir = model_info.get("data_dir") or (Path(model_info.get("path", "")) / "data")
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
    else:
        data_dir = Path(CONFIG.get("data_dir", "data"))
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure instructions.txt exists with default content
    instructions_file = data_dir / "instructions.txt"
    if not instructions_file.exists():
        instructions_file.write_text(DEFAULT_INSTRUCTIONS)
    
    # Ensure training.txt exists
    training_file = data_dir / "training.txt"
    if not training_file.exists():
        training_file.write_text("# Training Data\\n# Add Q&A pairs below\\n\\nQ: Hello\\nA: Hi there!\\n")
    
    # Add instructions.txt first
    parent.instructions_file_combo.addItem("instructions.txt", str(instructions_file))
    
    # Add training.txt second
    parent.instructions_file_combo.addItem("training.txt", str(training_file))
    
    # Add other files
    for f in sorted(data_dir.glob("*.txt")):
        if f.name not in ["instructions.txt", "training.txt"]:
            parent.instructions_file_combo.addItem(f.name, str(f))
    
    # Select instructions.txt by default
    parent.instructions_file_combo.setCurrentIndex(0)


def _load_instructions_file(parent, index):
    """Load an instructions file into the editor."""
    if index < 0:
        return
    
    filepath = parent.instructions_file_combo.itemData(index)
    if filepath and Path(filepath).exists():
        parent.instructions_editor.setPlainText(Path(filepath).read_text())
        parent._current_instructions_file = filepath


def _save_instructions_file(parent):
    """Save the current instructions file."""
    if not hasattr(parent, '_current_instructions_file') or not parent._current_instructions_file:
        QMessageBox.warning(parent, "No File", "Select a file first")
        return
    
    try:
        Path(parent._current_instructions_file).write_text(parent.instructions_editor.toPlainText())
        QMessageBox.information(parent, "Saved", "File saved successfully!")
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Failed to save: {e}")


def _create_instructions_file(parent):
    """Create a new notes file."""
    name, ok = QInputDialog.getText(parent, "New File", "File name (without .txt):")
    if ok and name:
        if not name.endswith(".txt"):
            name += ".txt"
        
        # Get data directory
        if parent.current_model_name:
            model_info = parent.registry.registry.get("models", {}).get(parent.current_model_name, {})
            data_dir = model_info.get("data_dir") or (Path(model_info.get("path", "")) / "data")
            if isinstance(data_dir, str):
                data_dir = Path(data_dir)
        else:
            data_dir = Path(CONFIG.get("data_dir", "data"))
        
        data_dir.mkdir(parents=True, exist_ok=True)
        new_file = data_dir / name
        
        if new_file.exists():
            QMessageBox.warning(parent, "Exists", f"{name} already exists")
            return
        
        new_file.write_text("")
        _refresh_instructions_files(parent)
        
        # Select the new file
        idx = parent.instructions_file_combo.findText(name)
        if idx >= 0:
            parent.instructions_file_combo.setCurrentIndex(idx)
