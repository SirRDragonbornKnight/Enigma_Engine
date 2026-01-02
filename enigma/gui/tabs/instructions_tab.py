"""Files tab for Enigma Engine GUI - edit model data files."""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QPlainTextEdit, QMessageBox, QFileDialog,
    QTreeWidget, QTreeWidgetItem, QSplitter, QGroupBox
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

## Model Scaling - How It Works

Enigma supports 15 model sizes from nano to omega:

### Size Tiers:
| Tier       | Sizes                 | Hardware Needed              |
|------------|----------------------|------------------------------|
| Embedded   | nano (~1M), micro (~2M) | Microcontrollers, IoT       |
| Edge       | tiny (~5M), mini (~10M) | Raspberry Pi, Mobile        |
| Consumer   | small (~27M), medium (~85M), base (~125M) | Desktop GPU |
| Prosumer   | large (~200M), xl (~600M) | RTX 3080+, RTX 4090        |
| Server     | xxl (~1.5B), huge (~3B)   | Multi-GPU Server           |
| Datacenter | giant (~7B), colossal (~13B) | A100/H100 Clusters       |
| Ultimate   | titan (~30B), omega (~70B+)  | Full Datacenter Racks     |

### Growing a Model:
You can grow a model while preserving learned knowledge!
- Train a small model on your laptop
- Grow it to medium when you get a better GPU
- Continue training the larger model

### Shrinking a Model:
You can shrink for deployment (some capacity lost):
- Train a large model on a powerful machine
- Shrink it to run on a Raspberry Pi

### Knowledge Distillation:
Train a small "student" model to mimic a large "teacher"
for efficient edge deployment.

## Connecting to Games/Robots/APIs
Protocol configs are in: data/protocols/
  - game/   : Game connections (Unity, Godot, etc)
  - robot/  : Robot connections (Arduino, ROS, GPIO)
  - api/    : API connections (REST, MQTT, etc)

To add a new connection:
1. Copy an example JSON from the folder
2. Edit the settings (host, port, protocol)
3. Set "enabled": true
4. Restart or click Refresh in the tab

See data/protocols/README.txt for full details.

## Avatar
Load an image or model in the Avatar tab.
Enable display via Options -> Avatar

## Vision
Take screenshots for the AI to analyze.
On Raspberry Pi, install: sudo apt install scrot

## Voice Features
- Options -> AI Auto-Speak: AI speaks responses
- Options -> Microphone: Voice input

## Module System
Everything in Enigma is a toggleable module:
- Core: model, tokenizer, training, inference
- Generation: image_gen, code_gen, video_gen, audio_gen
- Memory: memory, embeddings
- Perception: voice_input, vision
- Output: voice_output, avatar

Use the Modules tab to enable/disable capabilities.

## Tips
- Back up your model before making big changes
- Train in small batches - you can always train more
- The more diverse your training data, the better
- Check the models/ folder for your AI's files
"""


def create_instructions_tab(parent):
    """Create the files tab - edit model data and notes with folder browser."""
    w = QWidget()
    layout = QVBoxLayout()
    
    # Header
    header = QLabel("📁 File Browser")
    header.setObjectName("header")
    layout.addWidget(header)
    
    # Create splitter for file tree and editor
    splitter = QSplitter(Qt.Horizontal)
    splitter.setChildrenCollapsible(False)
    
    # Left side: File tree browser
    left_widget = QWidget()
    left_widget.setMinimumWidth(200)
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 5, 0)
    
    # File tree header
    tree_header = QHBoxLayout()
    tree_label = QLabel("📂 Folders")
    tree_label.setStyleSheet("font-weight: bold;")
    tree_header.addWidget(tree_label)
    
    refresh_btn = QPushButton("")
    refresh_btn.setToolTip("Refresh file list")
    refresh_btn.setMaximumWidth(30)
    refresh_btn.clicked.connect(lambda: _refresh_file_tree(parent))
    tree_header.addWidget(refresh_btn)
    
    left_layout.addLayout(tree_header)
    
    # File tree widget
    parent.file_tree = QTreeWidget()
    parent.file_tree.setHeaderHidden(True)
    parent.file_tree.setMinimumWidth(180)
    parent.file_tree.itemClicked.connect(lambda item, col: _on_tree_item_clicked(parent, item))
    left_layout.addWidget(parent.file_tree)
    
    splitter.addWidget(left_widget)
    
    # Right side: Editor
    right_widget = QWidget()
    right_widget.setMinimumWidth(400)
    right_layout = QVBoxLayout(right_widget)
    right_layout.setContentsMargins(5, 0, 0, 0)
    
    # File buttons row
    file_layout = QHBoxLayout()
    
    btn_open = QPushButton("📂 Open File...")
    btn_open.setToolTip("Open a file from your system")
    btn_open.clicked.connect(lambda: _open_instructions_file(parent))
    file_layout.addWidget(btn_open)
    
    btn_save = QPushButton("Save")
    btn_save.setToolTip("Save current file")
    btn_save.clicked.connect(lambda: _save_instructions_file(parent))
    file_layout.addWidget(btn_save)
    
    btn_new = QPushButton("New File")
    btn_new.setToolTip("Create a new file")
    btn_new.clicked.connect(lambda: _create_new_instructions_file(parent))
    file_layout.addWidget(btn_new)
    
    file_layout.addStretch()
    right_layout.addLayout(file_layout)
    
    # Current file name display
    parent.instructions_file_label = QLabel("No file open")
    parent.instructions_file_label.setStyleSheet("color: #a6e3a1; font-style: italic; padding: 4px;")
    right_layout.addWidget(parent.instructions_file_label)
    
    # Editor
    parent.instructions_editor = QPlainTextEdit()
    parent.instructions_editor.setPlaceholderText("Select a file from the tree or click 'Open File...'")
    right_layout.addWidget(parent.instructions_editor, stretch=1)
    
    splitter.addWidget(right_widget)
    splitter.setSizes([200, 600])
    
    layout.addWidget(splitter)
    
    w.setLayout(layout)
    
    # Initialize file tree
    _refresh_file_tree(parent)
    
    # Load default instructions file on startup
    _load_default_instructions(parent)
    
    return w


def _refresh_file_tree(parent):
    """Refresh the file tree with .txt files from data and docs folders."""
    parent.file_tree.clear()
    
    # Get paths
    data_dir = Path(CONFIG.get("data_dir", "data"))
    
    # Find project root by looking for marker files
    def find_project_root(start_path: Path) -> Path:
        """Find project root by looking for common marker files."""
        current = start_path
        markers = ['pyproject.toml', 'setup.py', 'README.md', '.git']
        for _ in range(10):  # Limit search depth
            for marker in markers:
                if (current / marker).exists():
                    return current
            if current.parent == current:
                break
            current = current.parent
        # Fallback: go up 4 levels from this file
        return start_path.parent.parent.parent.parent
    
    enigma_root = find_project_root(Path(__file__).resolve())
    docs_dir = enigma_root / "docs"
    
    # Add data folder section
    if data_dir.exists():
        data_item = QTreeWidgetItem(parent.file_tree, ["📁 Data Files"])
        data_item.setData(0, Qt.UserRole, str(data_dir))
        data_item.setExpanded(True)
        _add_txt_files_to_tree(data_item, data_dir)
    
    # Add docs folder section
    if docs_dir.exists():
        docs_item = QTreeWidgetItem(parent.file_tree, ["📚 Documentation"])
        docs_item.setData(0, Qt.UserRole, str(docs_dir))
        docs_item.setExpanded(True)
        _add_txt_files_to_tree(docs_item, docs_dir, include_md=True)
    
    # Add root level .txt files
    root_txt_files = list(enigma_root.glob("*.txt"))
    if root_txt_files:
        root_item = QTreeWidgetItem(parent.file_tree, ["Root Files"])
        root_item.setData(0, Qt.UserRole, str(enigma_root))
        root_item.setExpanded(True)
        for txt_file in sorted(root_txt_files):
            file_item = QTreeWidgetItem(root_item, [f"{txt_file.name}"])
            file_item.setData(0, Qt.UserRole, str(txt_file))


def _add_txt_files_to_tree(parent_item, folder: Path, include_md: bool = False):
    """Add .txt files (and optionally .md files) from a folder to the tree."""
    if not folder.exists():
        return
    
    # Get all relevant files
    files = list(folder.glob("*.txt"))
    if include_md:
        files.extend(folder.glob("*.md"))
    
    for txt_file in sorted(files, key=lambda x: x.name.lower()):
        icon = "" if txt_file.suffix == ".txt" else "📝"
        file_item = QTreeWidgetItem(parent_item, [f"{icon} {txt_file.name}"])
        file_item.setData(0, Qt.UserRole, str(txt_file))


def _on_tree_item_clicked(parent, item):
    """Handle tree item click - load file if it's a file."""
    filepath = item.data(0, Qt.UserRole)
    if filepath:
        path = Path(filepath)
        if path.is_file():
            _load_file_into_editor(parent, path)


def _load_file_into_editor(parent, filepath: Path):
    """Load a file into the editor."""
    parent._current_instructions_file = str(filepath)
    parent.instructions_file_label.setText(f"{filepath.name}")
    
    try:
        content = filepath.read_text(encoding='utf-8', errors='replace')
        parent.instructions_editor.setPlainText(content)
    except Exception as e:
        parent.instructions_editor.setPlainText(f"Error loading file: {e}")


def _load_default_instructions(parent):
    """Load the default instructions.txt file."""
    global_data_dir = Path(CONFIG.get("data_dir", "data"))
    global_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure instructions.txt exists with default content
    instructions_file = global_data_dir / "instructions.txt"
    if not instructions_file.exists():
        instructions_file.write_text(DEFAULT_INSTRUCTIONS)
    
    # Load it
    _load_file_into_editor(parent, instructions_file)


def _open_instructions_file(parent):
    """Open a file using system file dialog."""
    start_dir = str(Path(CONFIG.get("data_dir", "data")))
    
    filepath, _ = QFileDialog.getOpenFileName(
        parent, "Open File", start_dir, "Text Files (*.txt *.md);;All Files (*)"
    )
    
    if filepath:
        _load_file_into_editor(parent, Path(filepath))
        _refresh_file_tree(parent)


def _save_instructions_file(parent):
    """Save the current instructions file."""
    if not hasattr(parent, '_current_instructions_file') or not parent._current_instructions_file:
        QMessageBox.warning(parent, "No File", "Open a file first")
        return
    
    try:
        Path(parent._current_instructions_file).write_text(parent.instructions_editor.toPlainText(), encoding='utf-8')
        QMessageBox.information(parent, "Saved", "File saved successfully!")
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Failed to save: {e}")


def _create_new_instructions_file(parent):
    """Create a new file."""
    from PyQt5.QtWidgets import QInputDialog
    
    name, ok = QInputDialog.getText(
        parent, "New File", 
        "Enter filename (without .txt):",
        text="my_notes"
    )
    
    if not ok or not name.strip():
        return
    
    name = name.strip()
    if not name.endswith('.txt'):
        name += '.txt'
    
    # Save to data directory
    data_dir = Path(CONFIG.get("data_dir", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    new_file = data_dir / name
    
    if new_file.exists():
        reply = QMessageBox.question(
            parent, "File Exists",
            f"'{name}' already exists. Open it instead?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return
    else:
        # Create new empty file
        new_file.write_text("# Notes\n\n", encoding='utf-8')
    
    # Load the file
    _load_file_into_editor(parent, new_file)
    _refresh_file_tree(parent)
