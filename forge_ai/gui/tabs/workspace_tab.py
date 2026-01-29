"""
Workspace Tab - Combined Training, Notes, and Prompts management.

A unified space for:
  - Training data preparation and model training
  - Notes and bookmarks
  - System prompt templates management
"""

import json
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QListWidget, QListWidgetItem,
    QSplitter, QGroupBox, QInputDialog, QMessageBox,
    QTabWidget, QFrame, QPlainTextEdit,
    QSpinBox, QProgressBar, QFileDialog, QScrollArea,
    QDialogButtonBox, QDialog, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .shared_components import NoScrollComboBox
from ...config import CONFIG


def create_workspace_tab(parent):
    """Create the unified workspace tab with Training, Notes, and Prompts."""
    w = QWidget()
    layout = QVBoxLayout()
    layout.setSpacing(0)
    layout.setContentsMargins(0, 0, 0, 0)
    
    # Header
    header_layout = QHBoxLayout()
    header_layout.setContentsMargins(10, 10, 10, 5)
    header = QLabel("Workspace")
    header.setStyleSheet("font-size: 12px; font-weight: bold;")
    header_layout.addWidget(header)
    header_layout.addStretch()
    
    # Current model indicator
    parent.workspace_model_label = QLabel("No model loaded")
    parent.workspace_model_label.setStyleSheet("""
        color: #89b4fa; 
        font-weight: bold;
        padding: 4px 8px;
        background: rgba(137, 180, 250, 0.1);
        border-radius: 4px;
    """)
    header_layout.addWidget(parent.workspace_model_label)
    
    layout.addLayout(header_layout)
    
    # Sub-tabs for different workspace areas
    sub_tabs = QTabWidget()
    sub_tabs.setStyleSheet("""
        QTabWidget::pane {
            border: none;
            background: transparent;
        }
        QTabBar::tab {
            padding: 8px 20px;
            margin-right: 2px;
            background: #2d2d2d;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #3d3d3d;
            color: #89b4fa;
        }
        QTabBar::tab:hover:!selected {
            background: #353535;
        }
    """)
    
    # Training sub-tab
    training_widget = _create_training_section(parent)
    sub_tabs.addTab(training_widget, "Training")
    
    # Prompts sub-tab
    prompts_widget = _create_prompts_section(parent)
    sub_tabs.addTab(prompts_widget, "Prompts")
    
    # Notes sub-tab
    notes_widget = _create_notes_section(parent)
    sub_tabs.addTab(notes_widget, "Notes")
    
    layout.addWidget(sub_tabs)
    
    w.setLayout(layout)
    return w


def _create_training_section(parent):
    """Create the training data section."""
    w = QWidget()
    layout = QVBoxLayout()
    layout.setSpacing(8)
    layout.setContentsMargins(6, 6, 6, 6)
    
    # File management group
    file_group = QGroupBox("Training Data")
    file_layout = QVBoxLayout(file_group)
    
    # File action buttons
    btn_row = QHBoxLayout()
    
    btn_open = QPushButton("Open File")
    btn_open.setToolTip("Open a training data file")
    btn_open.clicked.connect(lambda: _browse_training_file(parent))
    btn_row.addWidget(btn_open)
    
    btn_save = QPushButton("Save")
    btn_save.setToolTip("Save changes")
    btn_save.clicked.connect(lambda: _save_training_file(parent))
    btn_row.addWidget(btn_save)
    
    btn_new = QPushButton("New File")
    btn_new.setToolTip("Create a new training data file")
    btn_new.clicked.connect(lambda: _create_new_training_file(parent))
    btn_row.addWidget(btn_new)
    
    btn_row.addStretch()
    file_layout.addLayout(btn_row)
    
    # Current file display
    parent.workspace_training_file_label = QLabel("No file open")
    parent.workspace_training_file_label.setStyleSheet("""
        color: #a6e3a1; 
        font-style: italic; 
        padding: 4px 8px;
        background: rgba(166, 227, 161, 0.1);
        border-radius: 4px;
    """)
    file_layout.addWidget(parent.workspace_training_file_label)
    
    layout.addWidget(file_group)
    
    # File content editor
    parent.workspace_training_editor = QPlainTextEdit()
    parent.workspace_training_editor.setPlaceholderText(
        "Training data will appear here...\n\n"
        "Format your data like this:\n"
        "Q: User question?\n"
        "A: AI response.\n"
    )
    parent.workspace_training_editor.setStyleSheet("""
        QPlainTextEdit {
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
        }
    """)
    layout.addWidget(parent.workspace_training_editor, stretch=1)
    
    # Training parameters group
    params_group = QGroupBox("Training Parameters")
    params_layout = QHBoxLayout(params_group)
    params_layout.setSpacing(10)
    
    # Epochs
    epochs_layout = QVBoxLayout()
    epochs_layout.addWidget(QLabel("Epochs"))
    parent.workspace_epochs_spin = QSpinBox()
    parent.workspace_epochs_spin.setRange(1, 10000)
    parent.workspace_epochs_spin.setValue(10)
    parent.workspace_epochs_spin.setToolTip("Training iterations")
    epochs_layout.addWidget(parent.workspace_epochs_spin)
    params_layout.addLayout(epochs_layout)
    
    # Batch size
    batch_layout = QVBoxLayout()
    batch_layout.addWidget(QLabel("Batch"))
    parent.workspace_batch_spin = QSpinBox()
    parent.workspace_batch_spin.setRange(1, 64)
    parent.workspace_batch_spin.setValue(4)
    parent.workspace_batch_spin.setToolTip("Examples per step")
    batch_layout.addWidget(parent.workspace_batch_spin)
    params_layout.addLayout(batch_layout)
    
    # Learning rate
    lr_layout = QVBoxLayout()
    lr_layout.addWidget(QLabel("Learn Rate"))
    parent.workspace_lr_input = QLineEdit("0.0001")
    parent.workspace_lr_input.setToolTip("Learning rate")
    lr_layout.addWidget(parent.workspace_lr_input)
    params_layout.addLayout(lr_layout)
    
    params_layout.addStretch()
    layout.addWidget(params_group)
    
    # Progress section
    parent.workspace_progress_label = QLabel("Ready to train")
    parent.workspace_progress_label.setStyleSheet("color: #bac2de;")
    layout.addWidget(parent.workspace_progress_label)
    
    parent.workspace_train_progress = QProgressBar()
    parent.workspace_train_progress.setValue(0)
    layout.addWidget(parent.workspace_train_progress)
    
    # Train buttons
    btn_layout = QHBoxLayout()
    
    parent.workspace_btn_train = QPushButton("Start Training")
    parent.workspace_btn_train.clicked.connect(parent._on_start_training)
    parent.workspace_btn_train.setStyleSheet("""
        QPushButton {
            padding: 10px 20px;
            font-weight: bold;
            background-color: #a6e3a1;
            color: #1e1e2e;
        }
    """)
    btn_layout.addWidget(parent.workspace_btn_train)
    
    parent.workspace_btn_stop = QPushButton("Stop")
    parent.workspace_btn_stop.clicked.connect(parent._on_stop_training)
    parent.workspace_btn_stop.setEnabled(False)
    parent.workspace_btn_stop.setStyleSheet("""
        QPushButton {
            padding: 10px 20px;
            background-color: #f38ba8;
            color: #1e1e2e;
        }
        QPushButton:disabled {
            background-color: #313244;
            color: #f38ba8;
            border: 2px dashed #f38ba8;
        }
    """)
    btn_layout.addWidget(parent.workspace_btn_stop)
    
    btn_layout.addStretch()
    layout.addLayout(btn_layout)
    
    # Initialize training file
    _init_training_file(parent)
    
    w.setLayout(layout)
    return w


def _create_prompts_section(parent):
    """Create the prompts management section."""
    w = QWidget()
    layout = QVBoxLayout()
    layout.setSpacing(8)
    layout.setContentsMargins(6, 6, 6, 6)
    
    # Description
    desc = QLabel(
        "Manage system prompts that define how the AI behaves. "
        "Create presets for different use cases."
    )
    desc.setWordWrap(True)
    desc.setStyleSheet("color: #bac2de; margin-bottom: 8px;")
    layout.addWidget(desc)
    
    # Preset management
    preset_group = QGroupBox("Prompt Presets")
    preset_layout = QVBoxLayout(preset_group)
    
    # Preset selector row
    selector_row = QHBoxLayout()
    selector_row.addWidget(QLabel("Preset:"))
    
    parent.workspace_prompt_combo = NoScrollComboBox()
    parent.workspace_prompt_combo.setMinimumWidth(250)
    parent.workspace_prompt_combo.setToolTip("Select a system prompt preset")
    # Load presets AFTER editor is created (below)
    parent.workspace_prompt_combo.currentIndexChanged.connect(
        lambda: _on_preset_changed(parent)
    )
    selector_row.addWidget(parent.workspace_prompt_combo)
    
    btn_refresh = QPushButton("Refresh")
    btn_refresh.clicked.connect(lambda: _load_prompt_presets(parent))
    selector_row.addWidget(btn_refresh)
    
    selector_row.addStretch()
    preset_layout.addLayout(selector_row)
    
    layout.addWidget(preset_group)
    
    # Prompt editor
    editor_group = QGroupBox("Prompt Content")
    editor_layout = QVBoxLayout(editor_group)
    
    parent.workspace_prompt_editor = QTextEdit()
    parent.workspace_prompt_editor.setPlaceholderText(
        "Enter or edit your system prompt here...\n\n"
        "This defines how the AI will behave and respond."
    )
    parent.workspace_prompt_editor.setStyleSheet("""
        QTextEdit {
            font-family: monospace;
            font-size: 12px;
            background-color: #2d2d2d;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 8px;
        }
    """)
    editor_layout.addWidget(parent.workspace_prompt_editor)
    
    # Action buttons
    btn_row = QHBoxLayout()
    
    btn_save = QPushButton("Save Preset")
    btn_save.setToolTip("Save changes to current preset")
    btn_save.clicked.connect(lambda: _save_current_preset(parent))
    btn_row.addWidget(btn_save)
    
    btn_new = QPushButton("Save As New")
    btn_new.setToolTip("Save as a new preset with a new name")
    btn_new.clicked.connect(lambda: _save_as_new_preset(parent))
    btn_row.addWidget(btn_new)
    
    btn_delete = QPushButton("Delete")
    btn_delete.setToolTip("Delete the current user preset")
    btn_delete.clicked.connect(lambda: _delete_current_preset(parent))
    btn_row.addWidget(btn_delete)
    
    btn_apply = QPushButton("Apply to Chat")
    btn_apply.setToolTip("Use this prompt in chat")
    btn_apply.setStyleSheet("""
        QPushButton {
            background-color: #89b4fa;
            color: #1e1e2e;
        }
    """)
    btn_apply.clicked.connect(lambda: _apply_prompt_to_chat(parent))
    btn_row.addWidget(btn_apply)
    
    btn_row.addStretch()
    editor_layout.addLayout(btn_row)
    
    parent.workspace_prompt_status = QLabel("")
    parent.workspace_prompt_status.setStyleSheet("color: #bac2de; font-style: italic;")
    editor_layout.addWidget(parent.workspace_prompt_status)
    
    layout.addWidget(editor_group, stretch=1)
    
    # NOW load presets after editor exists
    _load_prompt_presets(parent)
    
    w.setLayout(layout)
    return w


def _create_notes_section(parent):
    """Create the notes section."""
    w = QWidget()
    layout = QVBoxLayout()
    layout.setSpacing(8)
    layout.setContentsMargins(6, 6, 6, 6)
    
    # Notes list and editor in splitter
    splitter = QSplitter(Qt.Horizontal)
    
    # Left: Notes list
    list_widget = QWidget()
    list_layout = QVBoxLayout(list_widget)
    list_layout.setContentsMargins(0, 0, 5, 0)
    list_layout.setSpacing(8)
    
    list_header = QLabel("Notes")
    list_header.setStyleSheet("font-weight: bold; font-size: 12px; padding: 4px 0;")
    list_layout.addWidget(list_header)
    
    parent.workspace_notes_list = QListWidget()
    parent.workspace_notes_list.setStyleSheet("""
        QListWidget {
            border: 1px solid #444;
            border-radius: 4px;
            padding: 4px;
        }
        QListWidget::item {
            padding: 6px;
            border-radius: 4px;
        }
        QListWidget::item:selected {
            background-color: #89b4fa;
            color: #1e1e2e;
        }
        QListWidget::item:hover:!selected {
            background-color: #3d3d3d;
        }
    """)
    parent.workspace_notes_list.itemClicked.connect(
        lambda item: _load_note(parent, item)
    )
    list_layout.addWidget(parent.workspace_notes_list)
    
    # New note button below list
    btn_new_note = QPushButton("New Note")
    btn_new_note.setToolTip("Create a new note")
    btn_new_note.setStyleSheet("""
        QPushButton {
            padding: 8px;
            background-color: #3d3d3d;
            border: 1px solid #555;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #4d4d4d;
            border-color: #89b4fa;
        }
    """)
    btn_new_note.clicked.connect(lambda: _create_new_note(parent))
    list_layout.addWidget(btn_new_note)
    
    splitter.addWidget(list_widget)
    
    # Right: Note editor
    editor_widget = QWidget()
    editor_layout = QVBoxLayout(editor_widget)
    editor_layout.setContentsMargins(5, 0, 0, 0)
    editor_layout.setSpacing(8)
    
    # Note title
    title_label = QLabel("Title")
    title_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 4px 0;")
    editor_layout.addWidget(title_label)
    
    parent.workspace_note_title = QLineEdit()
    parent.workspace_note_title.setPlaceholderText("Enter note title...")
    parent.workspace_note_title.setStyleSheet("""
        QLineEdit {
            padding: 8px;
            border: 1px solid #444;
            border-radius: 4px;
            font-size: 12px;
        }
        QLineEdit:focus {
            border-color: #89b4fa;
        }
    """)
    editor_layout.addWidget(parent.workspace_note_title)
    
    # Note content label
    content_label = QLabel("Content")
    content_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 4px 0;")
    editor_layout.addWidget(content_label)
    
    # Note content
    parent.workspace_note_editor = QTextEdit()
    parent.workspace_note_editor.setPlaceholderText("Write your note here...")
    parent.workspace_note_editor.setStyleSheet("""
        QTextEdit {
            border: 1px solid #444;
            border-radius: 4px;
            padding: 8px;
            font-size: 12px;
        }
        QTextEdit:focus {
            border-color: #89b4fa;
        }
    """)
    editor_layout.addWidget(parent.workspace_note_editor)
    
    # Note actions - uniform button row
    note_btns = QHBoxLayout()
    note_btns.setSpacing(8)
    
    button_style = """
        QPushButton {
            padding: 8px 16px;
            border: 1px solid #555;
            border-radius: 4px;
            background-color: #3d3d3d;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #4d4d4d;
            border-color: #89b4fa;
        }
    """
    
    btn_save_note = QPushButton("Save")
    btn_save_note.setToolTip("Save changes to this note")
    btn_save_note.setStyleSheet(button_style)
    btn_save_note.clicked.connect(lambda: _save_note(parent))
    note_btns.addWidget(btn_save_note)
    
    btn_save_as = QPushButton("Save As")
    btn_save_as.setToolTip("Save as a new note with a different name")
    btn_save_as.setStyleSheet(button_style)
    btn_save_as.clicked.connect(lambda: _save_note_as(parent))
    note_btns.addWidget(btn_save_as)
    
    btn_delete_note = QPushButton("Delete")
    btn_delete_note.setToolTip("Delete this note")
    btn_delete_note.setStyleSheet("""
        QPushButton {
            padding: 8px 16px;
            border: 1px solid #f38ba8;
            border-radius: 4px;
            background-color: #3d3d3d;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: rgba(243, 139, 168, 0.3);
            border-color: #f38ba8;
        }
    """)
    btn_delete_note.clicked.connect(lambda: _delete_note(parent))
    note_btns.addWidget(btn_delete_note)
    
    note_btns.addStretch()
    
    btn_copy_to_training = QPushButton("Copy to Training")
    btn_copy_to_training.setToolTip("Add this note's content to training data")
    btn_copy_to_training.setStyleSheet("""
        QPushButton {
            padding: 8px 16px;
            border: 1px solid #a6e3a1;
            border-radius: 4px;
            background-color: #3d3d3d;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: rgba(166, 227, 161, 0.3);
            border-color: #a6e3a1;
        }
    """)
    btn_copy_to_training.clicked.connect(lambda: _copy_note_to_training(parent))
    note_btns.addWidget(btn_copy_to_training)
    
    editor_layout.addLayout(note_btns)
    
    splitter.addWidget(editor_widget)
    splitter.setSizes([200, 400])
    
    layout.addWidget(splitter)
    
    # Load existing notes
    _refresh_notes_list(parent)
    
    w.setLayout(layout)
    return w


# ============== Training Helpers ==============

def _init_training_file(parent):
    """Initialize default training file."""
    global_data_dir = Path(CONFIG.get("data_dir", "data"))
    global_data_dir.mkdir(parents=True, exist_ok=True)
    
    training_file = global_data_dir / "training.txt"
    if not training_file.exists():
        training_file.write_text("# Training Data\n\nQ: Hello\nA: Hi there!\n")
    
    parent._workspace_training_file = str(training_file)
    parent.workspace_training_file_label.setText(training_file.name)
    
    try:
        content = training_file.read_text(encoding='utf-8', errors='replace')
        parent.workspace_training_editor.setPlainText(content)
    except Exception as e:
        parent.workspace_training_editor.setPlainText(f"Error: {e}")


def _browse_training_file(parent):
    """Open a training file."""
    path, _ = QFileDialog.getOpenFileName(
        parent, "Open Training File",
        str(Path(CONFIG.get("data_dir", "data"))),
        "Text Files (*.txt);;All Files (*)"
    )
    if path:
        parent._workspace_training_file = path
        parent.workspace_training_file_label.setText(Path(path).name)
        try:
            content = Path(path).read_text(encoding='utf-8', errors='replace')
            parent.workspace_training_editor.setPlainText(content)
        except Exception as e:
            QMessageBox.warning(parent, "Error", f"Could not read file: {e}")


def _save_training_file(parent):
    """Save the current training file."""
    if not hasattr(parent, '_workspace_training_file'):
        return
    try:
        Path(parent._workspace_training_file).write_text(
            parent.workspace_training_editor.toPlainText(),
            encoding='utf-8'
        )
        # Sync with old training tab if it exists
        if hasattr(parent, 'training_editor'):
            parent.training_editor.setPlainText(parent.workspace_training_editor.toPlainText())
            parent.training_data_path = parent._workspace_training_file
            parent.data_path_label.setText(parent._workspace_training_file)
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Could not save: {e}")


def _create_new_training_file(parent):
    """Create a new training file."""
    name, ok = QInputDialog.getText(parent, "New File", "File name (without .txt):")
    if ok and name:
        data_dir = Path(CONFIG.get("data_dir", "data"))
        new_path = data_dir / f"{name}.txt"
        if new_path.exists():
            QMessageBox.warning(parent, "Exists", "File already exists!")
            return
        new_path.write_text("# Training Data\n\n")
        parent._workspace_training_file = str(new_path)
        parent.workspace_training_file_label.setText(new_path.name)
        parent.workspace_training_editor.setPlainText("# Training Data\n\n")


def _insert_template(parent, template_type):
    """Insert a template at cursor."""
    templates = {
        "qa": "\nQ: [Your question here]\nA: [AI response here]\n",
        "conversation": "\nUser: [User message]\nAssistant: [AI response]\n",
    }
    cursor = parent.workspace_training_editor.textCursor()
    cursor.insertText(templates.get(template_type, ""))


def _wrap_selection_qa(parent):
    """Wrap selected text in Q&A format."""
    cursor = parent.workspace_training_editor.textCursor()
    selected = cursor.selectedText()
    if not selected:
        QMessageBox.information(parent, "No Selection", "Select text to wrap.")
        return
    
    selected = selected.replace('\u2029', '\n')
    lines = selected.strip().split('\n')
    
    if len(lines) >= 2:
        question = lines[0].strip()
        answer = '\n'.join(lines[1:]).strip()
        wrapped = f"Q: {question}\nA: {answer}\n"
    else:
        wrapped = f"Q: {selected.strip()}\nA: [Answer here]\n"
    
    cursor.insertText(wrapped)


# ============== Prompts Helpers ==============

def _get_user_presets_path():
    """Get path to user presets file."""
    return Path(CONFIG.get("data_dir", "data")) / "user_prompts.json"


def _load_user_presets():
    """Load user presets from file."""
    path = _get_user_presets_path()
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}


def _save_user_presets(presets):
    """Save user presets to file."""
    path = _get_user_presets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(presets, f, indent=2)


# Built-in prompts
BUILTIN_PROMPTS = {
    "simple": "You are a helpful AI assistant. Answer questions clearly and conversationally. Be friendly and helpful.",
    
    "full": """You are ForgeAI, an intelligent AI assistant with access to various tools.

## Tool Usage
Use this format: <tool_call>{"tool": "name", "params": {}}</tool_call>

## Available Tools
- generate_image, generate_code, read_file, web_search

Be helpful and respect user privacy.""",

    "forgeai_complete": """You are the AI assistant for ForgeAI, a modular AI framework.

## Avatar System
- A 3D avatar appears on screen that you can control
- Control bones: head, neck, chest, shoulders, arms, legs

## Generation Tools
Use <tool_call>{"tool": "name", "params": {}}</tool_call>

## Interaction Style
- Be friendly and conversational
- Explain what you're doing before using tools"""
}


def _load_prompt_presets(parent):
    """Load all prompt presets into dropdown."""
    parent.workspace_prompt_combo.clear()
    
    # Built-in presets
    parent.workspace_prompt_combo.addItem("Simple (small models)", "simple")
    parent.workspace_prompt_combo.addItem("Full (with tools)", "full")
    parent.workspace_prompt_combo.addItem("ForgeAI Complete", "forgeai_complete")
    parent.workspace_prompt_combo.addItem("Custom (editable)", "custom")
    
    # User presets
    user_presets = _load_user_presets()
    for name in user_presets.keys():
        parent.workspace_prompt_combo.addItem(f"[User] {name}", f"user_{name}")
    
    # Trigger load of first item
    _on_preset_changed(parent)


def _on_preset_changed(parent):
    """Handle preset selection change."""
    preset_id = parent.workspace_prompt_combo.currentData()
    if not preset_id:
        return
    
    if preset_id in BUILTIN_PROMPTS:
        parent.workspace_prompt_editor.setText(BUILTIN_PROMPTS[preset_id])
        parent.workspace_prompt_editor.setReadOnly(True)
    elif preset_id == "custom":
        parent.workspace_prompt_editor.setReadOnly(False)
        if not parent.workspace_prompt_editor.toPlainText().strip():
            parent.workspace_prompt_editor.setPlaceholderText("Enter your custom prompt...")
    elif preset_id.startswith("user_"):
        name = preset_id[5:]
        presets = _load_user_presets()
        if name in presets:
            parent.workspace_prompt_editor.setText(presets[name])
        parent.workspace_prompt_editor.setReadOnly(False)


def _save_current_preset(parent):
    """Save changes to current preset."""
    preset_id = parent.workspace_prompt_combo.currentData()
    
    if preset_id in BUILTIN_PROMPTS:
        parent.workspace_prompt_status.setText("Cannot modify built-in presets. Use 'Save As New'.")
        parent.workspace_prompt_status.setStyleSheet("color: #f59e0b;")
        return
    
    if preset_id == "custom":
        # Save to gui_settings
        _save_to_gui_settings(parent)
        return
    
    if preset_id and preset_id.startswith("user_"):
        name = preset_id[5:]
        presets = _load_user_presets()
        presets[name] = parent.workspace_prompt_editor.toPlainText()
        _save_user_presets(presets)
        parent.workspace_prompt_status.setText(f"Saved: {name}")
        parent.workspace_prompt_status.setStyleSheet("color: #10b981;")


def _save_as_new_preset(parent):
    """Save as a new user preset."""
    content = parent.workspace_prompt_editor.toPlainText().strip()
    if not content:
        parent.workspace_prompt_status.setText("Cannot save empty prompt")
        parent.workspace_prompt_status.setStyleSheet("color: #ef4444;")
        return
    
    name, ok = QInputDialog.getText(parent, "Save Preset", "Preset name:")
    if not ok or not name.strip():
        return
    
    name = name.strip()
    presets = _load_user_presets()
    
    if name in presets:
        reply = QMessageBox.question(
            parent, "Overwrite?",
            f"'{name}' exists. Overwrite?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
    else:
        parent.workspace_prompt_combo.addItem(f"[User] {name}", f"user_{name}")
    
    presets[name] = content
    _save_user_presets(presets)
    
    # Select the new preset
    for i in range(parent.workspace_prompt_combo.count()):
        if parent.workspace_prompt_combo.itemData(i) == f"user_{name}":
            parent.workspace_prompt_combo.setCurrentIndex(i)
            break
    
    parent.workspace_prompt_status.setText(f"Created: {name}")
    parent.workspace_prompt_status.setStyleSheet("color: #10b981;")


def _delete_current_preset(parent):
    """Delete current user preset."""
    preset_id = parent.workspace_prompt_combo.currentData()
    
    if not preset_id or not preset_id.startswith("user_"):
        parent.workspace_prompt_status.setText("Can only delete user presets")
        parent.workspace_prompt_status.setStyleSheet("color: #f59e0b;")
        return
    
    name = preset_id[5:]
    reply = QMessageBox.question(
        parent, "Delete?",
        f"Delete preset '{name}'?",
        QMessageBox.Yes | QMessageBox.No
    )
    if reply != QMessageBox.Yes:
        return
    
    presets = _load_user_presets()
    if name in presets:
        del presets[name]
        _save_user_presets(presets)
    
    idx = parent.workspace_prompt_combo.currentIndex()
    parent.workspace_prompt_combo.removeItem(idx)
    parent.workspace_prompt_combo.setCurrentIndex(0)
    
    parent.workspace_prompt_status.setText(f"Deleted: {name}")
    parent.workspace_prompt_status.setStyleSheet("color: #ef4444;")


def _save_to_gui_settings(parent):
    """Save custom prompt to gui_settings.json."""
    settings_path = Path(CONFIG.get("data_dir", "data")) / "gui_settings.json"
    settings = {}
    if settings_path.exists():
        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        except:
            pass
    
    settings["system_prompt"] = parent.workspace_prompt_editor.toPlainText()
    settings["system_prompt_preset"] = "custom"
    
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)
    
    parent.workspace_prompt_status.setText("Saved to settings")
    parent.workspace_prompt_status.setStyleSheet("color: #10b981;")


def _apply_prompt_to_chat(parent):
    """Apply current prompt to chat system."""
    content = parent.workspace_prompt_editor.toPlainText()
    
    # Save to gui_settings
    settings_path = Path(CONFIG.get("data_dir", "data")) / "gui_settings.json"
    settings = {}
    if settings_path.exists():
        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        except:
            pass
    
    settings["system_prompt"] = content
    
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)
    
    # Update main window if available
    if hasattr(parent, 'system_prompt'):
        parent.system_prompt = content
    
    # Update settings tab if it exists
    if hasattr(parent, 'custom_system_prompt'):
        parent.custom_system_prompt.setText(content)
    
    parent.workspace_prompt_status.setText("Applied to chat!")
    parent.workspace_prompt_status.setStyleSheet("color: #89b4fa;")


# ============== Notes Helpers ==============

def _get_notes_dir():
    """Get notes directory."""
    notes_dir = Path(CONFIG.get("data_dir", "data")) / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    return notes_dir


def _refresh_notes_list(parent):
    """Refresh the notes list."""
    parent.workspace_notes_list.clear()
    notes_dir = _get_notes_dir()
    
    for note_file in sorted(notes_dir.glob("*.json"), reverse=True):
        try:
            with open(note_file, 'r') as f:
                note = json.load(f)
            item = QListWidgetItem(note.get("title", note_file.stem))
            item.setData(Qt.UserRole, str(note_file))
            parent.workspace_notes_list.addItem(item)
        except:
            pass


def _load_note(parent, item):
    """Load a note into the editor."""
    path = item.data(Qt.UserRole)
    if not path:
        return
    
    try:
        with open(path, 'r') as f:
            note = json.load(f)
        parent.workspace_note_title.setText(note.get("title", ""))
        parent.workspace_note_editor.setText(note.get("content", ""))
        parent._workspace_current_note = path
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Could not load note: {e}")


def _create_new_note(parent):
    """Create a new note."""
    parent.workspace_note_title.clear()
    parent.workspace_note_editor.clear()
    parent._workspace_current_note = None
    parent.workspace_note_title.setFocus()


def _save_note(parent):
    """Save the current note."""
    title = parent.workspace_note_title.text().strip()
    content = parent.workspace_note_editor.toPlainText()
    
    if not title:
        QMessageBox.warning(parent, "No Title", "Please enter a title.")
        return
    
    # Create safe filename
    safe_name = "".join(c for c in title if c.isalnum() or c in " -_").strip()[:50]
    notes_dir = _get_notes_dir()
    
    note_data = {
        "title": title,
        "content": content,
        "modified": datetime.now().isoformat()
    }
    
    # Use existing path or create new
    if hasattr(parent, '_workspace_current_note') and parent._workspace_current_note:
        note_path = Path(parent._workspace_current_note)
    else:
        note_path = notes_dir / f"{safe_name}.json"
        # Avoid overwriting
        counter = 1
        while note_path.exists():
            note_path = notes_dir / f"{safe_name}_{counter}.json"
            counter += 1
    
    with open(note_path, 'w') as f:
        json.dump(note_data, f, indent=2)
    
    parent._workspace_current_note = str(note_path)
    _refresh_notes_list(parent)


def _save_note_as(parent):
    """Save the current note with a new name."""
    content = parent.workspace_note_editor.toPlainText()
    
    # Ask for new name
    new_title, ok = QInputDialog.getText(
        parent, "Save As", "Enter new note name:",
        text=parent.workspace_note_title.text()
    )
    
    if not ok or not new_title.strip():
        return
    
    new_title = new_title.strip()
    
    # Create safe filename
    safe_name = "".join(c for c in new_title if c.isalnum() or c in " -_").strip()[:50]
    notes_dir = _get_notes_dir()
    
    note_path = notes_dir / f"{safe_name}.json"
    
    # Check if already exists
    if note_path.exists():
        reply = QMessageBox.question(
            parent, "Overwrite?",
            f"A note named '{new_title}' already exists. Overwrite?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
    
    note_data = {
        "title": new_title,
        "content": content,
        "modified": datetime.now().isoformat()
    }
    
    with open(note_path, 'w') as f:
        json.dump(note_data, f, indent=2)
    
    parent._workspace_current_note = str(note_path)
    parent.workspace_note_title.setText(new_title)
    _refresh_notes_list(parent)
    
    QMessageBox.information(parent, "Saved", f"Note saved as '{new_title}'")


def _delete_note(parent):
    """Delete the current note."""
    if not hasattr(parent, '_workspace_current_note') or not parent._workspace_current_note:
        QMessageBox.information(parent, "No Note", "No note selected.")
        return
    
    reply = QMessageBox.question(
        parent, "Delete?",
        "Delete this note?",
        QMessageBox.Yes | QMessageBox.No
    )
    if reply != QMessageBox.Yes:
        return
    
    try:
        Path(parent._workspace_current_note).unlink()
        parent._workspace_current_note = None
        parent.workspace_note_title.clear()
        parent.workspace_note_editor.clear()
        _refresh_notes_list(parent)
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Could not delete: {e}")


def _copy_note_to_training(parent):
    """Copy note content to training data."""
    content = parent.workspace_note_editor.toPlainText()
    if not content:
        QMessageBox.information(parent, "Empty", "Note is empty.")
        return
    
    # Get current training content
    current = parent.workspace_training_editor.toPlainText()
    
    # Add note content
    title = parent.workspace_note_title.text() or "Note"
    new_content = f"{current}\n\n# From Note: {title}\n{content}"
    parent.workspace_training_editor.setPlainText(new_content)
    
    QMessageBox.information(parent, "Copied", "Note added to training data!")
