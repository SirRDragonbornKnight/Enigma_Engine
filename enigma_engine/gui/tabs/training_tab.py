"""Train tab for Enigma AI Engine GUI."""

import logging
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...config import CONFIG
from .shared_components import NoScrollComboBox


def create_training_tab(parent):
    """Create the train tab with model training controls."""
    w = QWidget()
    layout = QVBoxLayout()
    layout.setSpacing(8)
    layout.setContentsMargins(6, 6, 6, 6)
    
    # Header with model info
    header_layout = QHBoxLayout()
    
    header = QLabel("Train Your AI")
    header.setObjectName("header")
    header.setStyleSheet("font-size: 12px; font-weight: bold;")
    header_layout.addWidget(header)
    
    header_layout.addStretch()
    
    # Current model info
    parent.training_model_label = QLabel("No model loaded")
    parent.training_model_label.setStyleSheet("""
        color: #89b4fa; 
        font-weight: bold;
        padding: 4px 8px;
        background: rgba(137, 180, 250, 0.1);
        border-radius: 4px;
    """)
    header_layout.addWidget(parent.training_model_label)
    
    layout.addLayout(header_layout)
    
    # === QUICK START SECTION ===
    quickstart_group = QGroupBox("Quick Start - Train Your AI in 3 Steps")
    quickstart_group.setStyleSheet("""
        QGroupBox {
            border: 2px solid #a6e3a1;
            border-radius: 8px;
            margin-top: 8px;
            padding: 8px;
            font-weight: bold;
        }
        QGroupBox::title {
            color: #a6e3a1;
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
    """)
    quickstart_layout = QVBoxLayout(quickstart_group)
    
    # Step 1: Load base knowledge
    step1_row = QHBoxLayout()
    step1_label = QLabel("1. Give AI basic knowledge:")
    step1_label.setStyleSheet("font-weight: bold; min-width: 180px;")
    step1_row.addWidget(step1_label)
    
    btn_load_base = QPushButton("Load Base Knowledge")
    btn_load_base.setToolTip("Load essential Q&A pairs that teach the AI fundamentals (greetings, basic facts, empathy, etc.)")
    btn_load_base.setStyleSheet("background: #a6e3a1; color: #1e1e2e; font-weight: bold; padding: 6px 12px;")
    btn_load_base.clicked.connect(lambda: _load_base_knowledge(parent))
    step1_row.addWidget(btn_load_base)
    
    btn_view_base = QPushButton("View")
    btn_view_base.setToolTip("Preview the base knowledge file")
    btn_view_base.clicked.connect(lambda: _view_base_knowledge(parent))
    step1_row.addWidget(btn_view_base)
    
    step1_row.addStretch()
    quickstart_layout.addLayout(step1_row)
    
    # Step 2: Add your own knowledge
    step2_row = QHBoxLayout()
    step2_label = QLabel("2. Add your knowledge:")
    step2_label.setStyleSheet("font-weight: bold; min-width: 180px;")
    step2_row.addWidget(step2_label)
    
    btn_paste_convert = QPushButton("Paste Text -> Q&A")
    btn_paste_convert.setToolTip("Paste any text and auto-convert it to training Q&A pairs using the Trainer AI")
    btn_paste_convert.setStyleSheet("background: #89b4fa; color: #1e1e2e; font-weight: bold; padding: 6px 12px;")
    btn_paste_convert.clicked.connect(lambda: _paste_and_convert(parent))
    step2_row.addWidget(btn_paste_convert)
    
    btn_import_url = QPushButton("Import URL")
    btn_import_url.setToolTip("Fetch a webpage and convert to training data")
    btn_import_url.clicked.connect(lambda: _import_from_url(parent))
    step2_row.addWidget(btn_import_url)
    
    step2_row.addStretch()
    quickstart_layout.addLayout(step2_row)
    
    # Step 3: Train
    step3_row = QHBoxLayout()
    step3_label = QLabel("3. Train the AI:")
    step3_label.setStyleSheet("font-weight: bold; min-width: 180px;")
    step3_row.addWidget(step3_label)
    
    btn_quick_train = QPushButton("Quick Train (Recommended Settings)")
    btn_quick_train.setToolTip("Start training with safe default settings (10 epochs, batch 4, lr 0.0001)")
    btn_quick_train.setStyleSheet("background: #f5c2e7; color: #1e1e2e; font-weight: bold; padding: 6px 12px;")
    btn_quick_train.clicked.connect(lambda: _quick_train(parent))
    step3_row.addWidget(btn_quick_train)
    
    step3_row.addStretch()
    quickstart_layout.addLayout(step3_row)
    
    # Status line
    parent.quickstart_status = QLabel("Load base knowledge first, then add your own, then train!")
    parent.quickstart_status.setStyleSheet("color: #6c7086; font-style: italic; padding: 4px;")
    quickstart_layout.addWidget(parent.quickstart_status)
    
    layout.addWidget(quickstart_group)
    
    # File management group
    file_group = QGroupBox("Training Data")
    file_layout = QVBoxLayout(file_group)
    
    # File action buttons
    btn_row = QHBoxLayout()
    
    btn_open = QPushButton("Open File")
    btn_open.setToolTip("Open a training data file from your system")
    btn_open.clicked.connect(lambda: _browse_training_file(parent))
    btn_row.addWidget(btn_open)
    
    btn_save = QPushButton("Save")
    btn_save.setToolTip("Save changes to the current file")
    btn_save.clicked.connect(lambda: _save_training_file(parent))
    btn_row.addWidget(btn_save)
    
    btn_new = QPushButton("New File")
    btn_new.setToolTip("Create a new training data file")
    btn_new.clicked.connect(lambda: _create_new_training_file(parent))
    btn_row.addWidget(btn_new)
    
    btn_url = QPushButton("Import from URL")
    btn_url.setToolTip("Fetch content from a webpage and convert to training data")
    btn_url.clicked.connect(lambda: _import_from_url(parent))
    btn_url.setStyleSheet("""
        QPushButton {
            background-color: #89b4fa;
            color: #1e1e2e;
        }
        QPushButton:hover {
            background-color: #b4d0ff;
        }
    """)
    btn_row.addWidget(btn_url)
    
    btn_row.addStretch()
    file_layout.addLayout(btn_row)
    
    # Current file display
    parent.training_file_label = QLabel("No file open")
    parent.training_file_label.setStyleSheet("""
        color: #a6e3a1; 
        font-style: italic; 
        padding: 4px 8px;
        background: rgba(166, 227, 161, 0.1);
        border-radius: 4px;
    """)
    parent.training_file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    file_layout.addWidget(parent.training_file_label)
    
    layout.addWidget(file_group)
    
    # Prompt Templates section (collapsible)
    prompt_group = QGroupBox("Prompt Templates")
    prompt_group.setCheckable(True)
    prompt_group.setChecked(False)  # Collapsed by default
    prompt_layout = QVBoxLayout(prompt_group)
    
    # Template selector row
    template_row = QHBoxLayout()
    template_row.addWidget(QLabel("Template:"))
    parent.training_prompt_combo = NoScrollComboBox()
    parent.training_prompt_combo.setMinimumWidth(200)
    parent.training_prompt_combo.setToolTip("Select a prompt template to insert into your training data")
    _populate_prompt_templates(parent)
    parent.training_prompt_combo.currentIndexChanged.connect(
        lambda: _preview_prompt_template(parent)
    )
    template_row.addWidget(parent.training_prompt_combo)
    
    btn_refresh = QPushButton("Refresh")
    btn_refresh.setToolTip("Reload templates from Settings")
    btn_refresh.clicked.connect(lambda: _populate_prompt_templates(parent))
    template_row.addWidget(btn_refresh)
    
    template_row.addStretch()
    prompt_layout.addLayout(template_row)
    
    # Preview area
    parent.prompt_preview = QTextEdit()
    parent.prompt_preview.setReadOnly(True)
    parent.prompt_preview.setMaximumHeight(80)
    parent.prompt_preview.setPlaceholderText("Select a template to preview...")
    parent.prompt_preview.setStyleSheet("""
        QTextEdit {
            background-color: #252525;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 4px;
            font-family: monospace;
            font-size: 12px;
            color: #aaa;
        }
    """)
    prompt_layout.addWidget(parent.prompt_preview)
    
    # Insert buttons
    insert_row = QHBoxLayout()
    
    btn_insert_start = QPushButton("Insert at Start")
    btn_insert_start.setToolTip("Add this prompt at the beginning of training data")
    btn_insert_start.clicked.connect(lambda: _insert_prompt_template(parent, "start"))
    insert_row.addWidget(btn_insert_start)
    
    btn_insert_cursor = QPushButton("Insert at Cursor")
    btn_insert_cursor.setToolTip("Insert prompt at current cursor position")
    btn_insert_cursor.clicked.connect(lambda: _insert_prompt_template(parent, "cursor"))
    insert_row.addWidget(btn_insert_cursor)
    
    btn_wrap = QPushButton("Wrap Selection")
    btn_wrap.setToolTip("Wrap selected text with Q:/A: format")
    btn_wrap.clicked.connect(lambda: _wrap_selection_qa(parent))
    insert_row.addWidget(btn_wrap)
    
    insert_row.addStretch()
    prompt_layout.addLayout(insert_row)
    
    layout.addWidget(prompt_group)
    
    # Training Data Preview Group
    preview_group = QGroupBox("Training Data Preview")
    preview_group.setCheckable(True)
    preview_group.setChecked(False)  # Collapsed by default
    preview_group.setStyleSheet("""
        QGroupBox {
            border: 2px solid #f9e2af;
            border-radius: 8px;
            margin-top: 8px;
            padding: 8px;
            font-weight: bold;
        }
        QGroupBox::title {
            color: #f9e2af;
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
    """)
    preview_layout = QVBoxLayout(preview_group)
    
    # Preview description
    preview_desc = QLabel("Preview how training data will be parsed before training.")
    preview_desc.setWordWrap(True)
    preview_desc.setStyleSheet("color: #a6adc8; font-weight: normal; padding: 4px;")
    preview_layout.addWidget(preview_desc)
    
    # Preview button row
    preview_btn_row = QHBoxLayout()
    btn_analyze = QPushButton("Analyze Data")
    btn_analyze.setToolTip("Parse current training data and show detected Q&A pairs")
    btn_analyze.clicked.connect(lambda: _analyze_training_data(parent))
    preview_btn_row.addWidget(btn_analyze)
    
    btn_validate = QPushButton("Validate Format")
    btn_validate.setToolTip("Check for formatting issues")
    btn_validate.clicked.connect(lambda: _validate_training_data(parent))
    preview_btn_row.addWidget(btn_validate)
    
    preview_btn_row.addStretch()
    
    # Stats label
    parent.data_stats_label = QLabel("No data analyzed")
    parent.data_stats_label.setStyleSheet("color: #6c7086; font-size: 11px;")
    preview_btn_row.addWidget(parent.data_stats_label)
    
    preview_layout.addLayout(preview_btn_row)
    
    # Preview text area
    parent.data_preview_text = QTextEdit()
    parent.data_preview_text.setReadOnly(True)
    parent.data_preview_text.setMaximumHeight(150)
    parent.data_preview_text.setPlaceholderText("Click 'Analyze Data' to see parsed training pairs...")
    parent.data_preview_text.setStyleSheet("""
        QTextEdit {
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 6px;
            padding: 4px;
            font-family: 'Consolas', monospace;
            font-size: 11px;
        }
    """)
    preview_layout.addWidget(parent.data_preview_text)
    
    layout.addWidget(preview_group)
    
    # File content editor
    parent.training_editor = QPlainTextEdit()
    parent.training_editor.setPlaceholderText(
        "Training data will appear here...\n\n"
        "Format your data like this:\n"
        "Q: User question?\n"
        "A: AI response.\n\n"
        "Or plain text for the AI to learn patterns from."
    )
    parent.training_editor.setStyleSheet("""
        QPlainTextEdit {
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            line-height: 1.4;
        }
    """)
    layout.addWidget(parent.training_editor, stretch=1)
    
    # Training parameters group
    params_group = QGroupBox("Training Parameters")
    params_layout = QHBoxLayout(params_group)
    params_layout.setSpacing(10)
    
    # Epochs
    epochs_layout = QVBoxLayout()
    epochs_label = QLabel("Epochs")
    epochs_label.setStyleSheet("font-size: 12px; color: #bac2de;")
    epochs_layout.addWidget(epochs_label)
    parent.epochs_spin = QSpinBox()
    parent.epochs_spin.setRange(1, 10000)
    parent.epochs_spin.setValue(10)
    parent.epochs_spin.setToolTip("How many times to go through all data")
    parent.epochs_spin.setMinimumWidth(80)
    epochs_layout.addWidget(parent.epochs_spin)
    params_layout.addLayout(epochs_layout)
    
    # Batch size
    batch_layout = QVBoxLayout()
    batch_label = QLabel("Batch Size")
    batch_label.setStyleSheet("font-size: 12px; color: #bac2de;")
    batch_layout.addWidget(batch_label)
    parent.batch_spin = QSpinBox()
    parent.batch_spin.setRange(1, 64)
    parent.batch_spin.setValue(4)
    parent.batch_spin.setToolTip("Examples per step (Pi: 1-2, GPU: 4-16)")
    parent.batch_spin.setMinimumWidth(70)
    batch_layout.addWidget(parent.batch_spin)
    params_layout.addLayout(batch_layout)
    
    # Learning rate
    lr_layout = QVBoxLayout()
    lr_label = QLabel("Learning Rate")
    lr_label.setStyleSheet("font-size: 12px; color: #bac2de;")
    lr_layout.addWidget(lr_label)
    parent.lr_input = QLineEdit("0.0001")
    parent.lr_input.setToolTip("How fast AI learns (lower = slower but stable)")
    parent.lr_input.setMinimumWidth(80)
    lr_layout.addWidget(parent.lr_input)
    params_layout.addLayout(lr_layout)
    
    # Data format selector
    format_layout = QVBoxLayout()
    format_label = QLabel("Data Format")
    format_label.setStyleSheet("font-size: 12px; color: #bac2de;")
    format_layout.addWidget(format_label)
    parent.format_combo = NoScrollComboBox()
    parent.format_combo.addItem("Auto-detect", "auto")
    parent.format_combo.addItem("Q&A (Q: A:)", "qa")
    parent.format_combo.addItem("JSONL", "jsonl")
    parent.format_combo.addItem("Conversation", "conversation")
    parent.format_combo.addItem("Instruction", "instruction")
    parent.format_combo.addItem("ChatML", "chatml")
    parent.format_combo.addItem("Plain Text", "plain")
    parent.format_combo.setToolTip(
        "Training data format:\n"
        "- Auto-detect: Automatically detect format\n"
        "- Q&A: Q: question\\nA: answer\n"
        "- JSONL: {\"input\": ..., \"output\": ...}\n"
        "- Conversation: User: ... Assistant: ...\n"
        "- Instruction: ### Instruction\\n...\\n### Response\n"
        "- ChatML: <|im_start|>user\\n..."
    )
    parent.format_combo.setMinimumWidth(100)
    format_layout.addWidget(parent.format_combo)
    params_layout.addLayout(format_layout)
    
    params_layout.addStretch()
    layout.addWidget(params_group)
    
    # === SYSTEM PROMPT EDITOR SECTION ===
    prompt_editor_group = QGroupBox("System Prompt (AI Personality)")
    prompt_editor_group.setCheckable(True)
    prompt_editor_group.setChecked(False)  # Collapsed by default for space
    prompt_editor_group.setStyleSheet("""
        QGroupBox {
            border: 2px solid #cba6f7;
            border-radius: 8px;
            margin-top: 8px;
            padding: 8px;
            font-weight: bold;
        }
        QGroupBox::title {
            color: #cba6f7;
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
    """)
    prompt_editor_layout = QVBoxLayout(prompt_editor_group)
    
    # Description
    prompt_desc = QLabel(
        "Define your AI's personality and behavior. This prompt is prepended to all interactions."
    )
    prompt_desc.setWordWrap(True)
    prompt_desc.setStyleSheet("color: #a6adc8; font-weight: normal; padding: 4px;")
    prompt_editor_layout.addWidget(prompt_desc)
    
    # Preset selector row
    preset_row = QHBoxLayout()
    preset_row.addWidget(QLabel("Preset:"))
    parent.system_prompt_preset = NoScrollComboBox()
    parent.system_prompt_preset.addItem("Default Assistant", "default")
    parent.system_prompt_preset.addItem("Helpful Expert", "expert")
    parent.system_prompt_preset.addItem("Friendly Casual", "casual")
    parent.system_prompt_preset.addItem("Technical Precise", "technical")
    parent.system_prompt_preset.addItem("Creative Writer", "creative")
    parent.system_prompt_preset.addItem("Custom", "custom")
    parent.system_prompt_preset.setToolTip("Select a personality preset or choose Custom to write your own")
    parent.system_prompt_preset.currentIndexChanged.connect(lambda: _load_system_prompt_preset(parent))
    preset_row.addWidget(parent.system_prompt_preset)
    
    btn_load_persona = QPushButton("Load from Persona")
    btn_load_persona.setToolTip("Load system prompt from an existing persona")
    btn_load_persona.clicked.connect(lambda: _load_prompt_from_persona(parent))
    preset_row.addWidget(btn_load_persona)
    
    preset_row.addStretch()
    prompt_editor_layout.addLayout(preset_row)
    
    # System prompt editor
    parent.system_prompt_editor = QPlainTextEdit()
    parent.system_prompt_editor.setMaximumHeight(120)
    parent.system_prompt_editor.setPlaceholderText(
        "Enter your AI's system prompt here...\n\n"
        "Example: You are a helpful assistant named Forge. "
        "You are knowledgeable, friendly, and always try to help users."
    )
    parent.system_prompt_editor.setStyleSheet("""
        QPlainTextEdit {
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            background: #1e1e2e;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 4px;
        }
    """)
    # Load default system prompt
    parent.system_prompt_editor.setPlainText(
        "You are Forge, a helpful AI assistant. "
        "You provide clear, accurate, and helpful responses. "
        "You are friendly but professional."
    )
    prompt_editor_layout.addWidget(parent.system_prompt_editor)
    
    # Action buttons
    prompt_btn_row = QHBoxLayout()
    
    btn_save_prompt = QPushButton("Save to Persona")
    btn_save_prompt.setToolTip("Save this system prompt to the current persona")
    btn_save_prompt.clicked.connect(lambda: _save_system_prompt_to_persona(parent))
    prompt_btn_row.addWidget(btn_save_prompt)
    
    btn_inject_prompt = QPushButton("Inject into Training Data")
    btn_inject_prompt.setToolTip("Add this system prompt as the first entry in your training data")
    btn_inject_prompt.setStyleSheet("background: #89b4fa; color: #1e1e2e;")
    btn_inject_prompt.clicked.connect(lambda: _inject_system_prompt(parent))
    prompt_btn_row.addWidget(btn_inject_prompt)
    
    btn_test_prompt = QPushButton("Test Response")
    btn_test_prompt.setToolTip("Test how the AI responds with this system prompt")
    btn_test_prompt.clicked.connect(lambda: _test_system_prompt(parent))
    prompt_btn_row.addWidget(btn_test_prompt)
    
    prompt_btn_row.addStretch()
    prompt_editor_layout.addLayout(prompt_btn_row)
    
    layout.addWidget(prompt_editor_group)
    
    # Progress section
    progress_layout = QVBoxLayout()
    
    # Progress bar with label
    progress_header = QHBoxLayout()
    parent.training_progress_label = QLabel("Ready to train")
    parent.training_progress_label.setStyleSheet("color: #bac2de;")
    progress_header.addWidget(parent.training_progress_label)
    progress_header.addStretch()
    progress_layout.addLayout(progress_header)
    
    parent.train_progress = QProgressBar()
    parent.train_progress.setValue(0)
    parent.train_progress.setStyleSheet("""
        QProgressBar {
            border-radius: 4px;
            text-align: center;
            height: 20px;
        }
        QProgressBar::chunk {
            background-color: #a6e3a1;
            border-radius: 4px;
        }
    """)
    progress_layout.addWidget(parent.train_progress)
    
    layout.addLayout(progress_layout)
    
    # Train and Stop buttons row
    btn_layout = QHBoxLayout()
    btn_layout.setSpacing(10)
    
    parent.btn_train = QPushButton("Start Training")
    parent.btn_train.clicked.connect(parent._on_start_training)
    parent.btn_train.setToolTip("Start training the model on your data.\\nTraining will run for the specified number of epochs.\\nYou can stop training at any time.")
    parent.btn_train.setStyleSheet("""
        QPushButton {
            padding: 12px 24px;
            font-size: 12px;
            font-weight: bold;
            background-color: #a6e3a1;
            color: #1e1e2e;
        }
        QPushButton:hover {
            background-color: #b4f0b4;
        }
    """)
    btn_layout.addWidget(parent.btn_train)
    
    parent.btn_stop_train = QPushButton("Stop")
    parent.btn_stop_train.setToolTip("Stop training after the current epoch completes.\\nPartial training progress will be saved.")
    parent.btn_stop_train.clicked.connect(parent._on_stop_training)
    parent.btn_stop_train.setEnabled(False)
    parent.btn_stop_train.setStyleSheet("""
        QPushButton {
            padding: 12px 24px;
            font-size: 12px;
            background-color: #f38ba8;
            color: #1e1e2e;
        }
        QPushButton:disabled {
            background-color: #313244;
            color: #f38ba8;
            border: 2px dashed #f38ba8;
        }
    """)
    btn_layout.addWidget(parent.btn_stop_train)
    
    btn_layout.addStretch()
    layout.addLayout(btn_layout)
    
    # Hidden data path label (for compatibility)
    parent.data_path_label = QLabel("")
    parent.data_path_label.setVisible(False)
    layout.addWidget(parent.data_path_label)
    
    # Initialize training file
    _refresh_training_files(parent)
    
    w.setLayout(layout)
    return w


def _refresh_training_files(parent):
    """Initialize training data - open default file if exists."""
    # Always use global data directory for training files
    global_data_dir = Path(CONFIG.get("data_dir", "data"))
    global_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure training.txt exists in global data dir
    training_file = global_data_dir / "training.txt"
    if not training_file.exists():
        training_file.write_text("# Training Data\n\nQ: Hello\nA: Hi there!\n")
    
    # Load the default training file
    parent.training_data_path = str(training_file)
    parent.data_path_label.setText(str(training_file))
    parent._current_training_file = str(training_file)
    parent.training_file_label.setText(f"{training_file.name}")
    
    try:
        content = training_file.read_text(encoding='utf-8', errors='replace')
        parent.training_editor.setPlainText(content)
    except Exception as e:
        parent.training_editor.setPlainText(f"Error loading file: {e}")


def _load_training_file(parent, index):
    """Load a training file into the editor - deprecated, kept for compatibility."""


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
    """Browse for a training file using system file dialog."""
    # Start in data directory
    start_dir = str(Path(CONFIG.get("data_dir", "data")))
    
    filepath, _ = QFileDialog.getOpenFileName(
        parent, "Open Training File", start_dir, "Text Files (*.txt);;All Files (*)"
    )
    
    if filepath:
        # Update paths
        parent.training_data_path = filepath
        parent.data_path_label.setText(filepath)
        parent._current_training_file = filepath
        parent.training_file_label.setText(f"{Path(filepath).name}")
        
        # Load file content
        try:
            content = Path(filepath).read_text(encoding='utf-8', errors='replace')
            parent.training_editor.setPlainText(content)
        except Exception as e:
            parent.training_editor.setPlainText(f"Error loading file: {e}")


def _create_new_training_file(parent):
    """Create a new training data file."""
    # Get filename from user
    name, ok = QInputDialog.getText(
        parent, "New Training File", 
        "Enter filename (without .txt):",
        text="my_training_data"
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
        # Create new file with template
        template = """# Training Data
# Add Q&A pairs below for your AI to learn from

Q: Hello
A: Hi there! How can I help you today?

Q: What is your name?
A: I'm your AI assistant.

# Add more Q&A pairs here...
"""
        new_file.write_text(template, encoding='utf-8')
    
    # Load the file
    parent.training_data_path = str(new_file)
    parent.data_path_label.setText(str(new_file))
    parent._current_training_file = str(new_file)
    parent.training_file_label.setText(f"{new_file.name}")
    
    try:
        content = new_file.read_text(encoding='utf-8', errors='replace')
        parent.training_editor.setPlainText(content)
    except Exception as e:
        parent.training_editor.setPlainText(f"Error loading file: {e}")


def _import_from_url(parent):
    """Import training data from a URL."""
    # Create dialog for URL input
    dialog = QDialog(parent)
    dialog.setWindowTitle("Import Training Data from URL")
    dialog.setMinimumWidth(500)
    dialog.setMinimumHeight(400)
    
    layout = QVBoxLayout(dialog)
    
    # URL input
    url_layout = QHBoxLayout()
    url_label = QLabel("URL:")
    url_layout.addWidget(url_label)
    url_input = QLineEdit()
    url_input.setPlaceholderText("https://example.com/article")
    url_layout.addWidget(url_input)
    layout.addLayout(url_layout)
    
    # Options
    options_group = QGroupBox("Conversion Options")
    options_layout = QVBoxLayout(options_group)
    
    format_qa = QCheckBox("Convert to Q&A format (extract paragraphs as answers)")
    format_qa.setChecked(True)
    format_qa.setToolTip("Converts paragraphs into Q: What is [topic]?\\nA: [paragraph]")
    options_layout.addWidget(format_qa)
    
    keep_headers = QCheckBox("Include headers as topics")
    keep_headers.setChecked(True)
    keep_headers.setToolTip("Use H1/H2/H3 headers to create topic-based questions")
    options_layout.addWidget(keep_headers)
    
    clean_text = QCheckBox("Clean up formatting (remove extra whitespace)")
    clean_text.setChecked(True)
    options_layout.addWidget(clean_text)
    
    layout.addWidget(options_group)
    
    # Preview area
    preview_label = QLabel("Preview (click 'Fetch' to load content):")
    layout.addWidget(preview_label)
    
    preview_text = QTextEdit()
    preview_text.setPlaceholderText("Fetched content will appear here...")
    preview_text.setReadOnly(True)
    layout.addWidget(preview_text)
    
    # Buttons
    btn_layout = QHBoxLayout()
    
    fetch_btn = QPushButton("Fetch")
    fetch_btn.setStyleSheet("background-color: #89b4fa; color: #1e1e2e;")
    
    def on_fetch():
        url = url_input.text().strip()
        if not url:
            QMessageBox.warning(dialog, "Error", "Please enter a URL")
            return
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            url_input.setText(url)
        
        preview_text.setPlainText("Fetching...")
        fetch_btn.setEnabled(False)
        
        # Run fetch in background thread to prevent UI freeze
        import threading
        def do_fetch():
            try:
                # Fetch the webpage
                headers = {"User-Agent": "Mozilla/5.0 (compatible; ForgeBot/1.0)"}
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as response:
                    html = response.read().decode('utf-8', errors='ignore')
                
                # Extract text content
                content = _extract_training_content(
                    html, 
                    as_qa=format_qa.isChecked(),
                    include_headers=keep_headers.isChecked(),
                    clean=clean_text.isChecked()
                )
                
                # Update UI on main thread
                from PyQt5.QtCore import Q_ARG, QMetaObject, Qt
                QMetaObject.invokeMethod(
                    preview_text, "setPlainText",
                    Qt.QueuedConnection,
                    Q_ARG(str, content)
                )
                
            except urllib.error.HTTPError as e:
                from PyQt5.QtCore import Q_ARG, QMetaObject, Qt
                QMetaObject.invokeMethod(
                    preview_text, "setPlainText",
                    Qt.QueuedConnection,
                    Q_ARG(str, f"HTTP Error {e.code}: {e.reason}")
                )
            except urllib.error.URLError as e:
                from PyQt5.QtCore import Q_ARG, QMetaObject, Qt
                QMetaObject.invokeMethod(
                    preview_text, "setPlainText",
                    Qt.QueuedConnection,
                    Q_ARG(str, f"Network Error: {e}")
                )
            except Exception as e:
                from PyQt5.QtCore import Q_ARG, QMetaObject, Qt
                QMetaObject.invokeMethod(
                    preview_text, "setPlainText",
                    Qt.QueuedConnection,
                    Q_ARG(str, f"Error: {e}")
                )
            finally:
                from PyQt5.QtCore import Q_ARG, QMetaObject, Qt
                QMetaObject.invokeMethod(
                    fetch_btn, "setEnabled",
                    Qt.QueuedConnection,
                    Q_ARG(bool, True)
                )
        
        thread = threading.Thread(target=do_fetch, daemon=True)
        thread.start()
    
    fetch_btn.clicked.connect(on_fetch)
    btn_layout.addWidget(fetch_btn)
    
    btn_layout.addStretch()
    
    # Dialog buttons
    button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    btn_layout.addWidget(button_box)
    
    layout.addLayout(btn_layout)
    
    # Show dialog
    if dialog.exec_() == QDialog.Accepted:
        content = preview_text.toPlainText()
        if content and not content.startswith(("Fetching...", "HTTP Error", "Network Error", "Error:")):
            # Append to current editor content
            current = parent.training_editor.toPlainText()
            if current.strip():
                new_content = current + "\n\n# Imported from URL\n" + content
            else:
                new_content = "# Imported from URL\n" + content
            parent.training_editor.setPlainText(new_content)
            QMessageBox.information(parent, "Success", "Content imported! Don't forget to save.")


def _extract_training_content(html: str, as_qa: bool = True, include_headers: bool = True, clean: bool = True) -> str:
    """
    Extract and convert webpage HTML to training data format.
    
    Args:
        html: Raw HTML content
        as_qa: Convert to Q&A format
        include_headers: Use headers for topics
        clean: Clean up whitespace
        
    Returns:
        Formatted training data
    """
    # Remove script, style, nav, footer, etc.
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<aside[^>]*>.*?</aside>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    
    # Extract title
    title_match = re.search(r'<title>([^<]+)</title>', html, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else "Unknown Topic"
    
    lines = []
    current_topic = title
    
    if include_headers:
        # Extract headers and following paragraphs
        # Find all headers (h1-h4) and paragraphs
        pattern = r'<(h[1-4])[^>]*>(.*?)</\1>|<p[^>]*>(.*?)</p>'
        matches = re.finditer(pattern, html, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            if match.group(1):  # It's a header
                header_text = re.sub(r'<[^>]+>', '', match.group(2)).strip()
                if header_text:
                    current_topic = header_text
            elif match.group(3):  # It's a paragraph
                para_text = re.sub(r'<[^>]+>', '', match.group(3)).strip()
                if para_text and len(para_text) > 30:  # Skip very short paragraphs
                    if as_qa:
                        # Create Q&A format
                        question = f"What is {current_topic}?" if current_topic != title else f"Tell me about {title}"
                        lines.append(f"Q: {question}")
                        lines.append(f"A: {para_text}")
                        lines.append("")
                    else:
                        lines.append(para_text)
                        lines.append("")
    else:
        # Just extract all text
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences/paragraphs
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 30:
                if as_qa and i % 2 == 0:
                    lines.append(f"Q: {sentence}")
                elif as_qa:
                    lines.append(f"A: {sentence}")
                    lines.append("")
                else:
                    lines.append(sentence)
    
    result = '\n'.join(lines)
    
    if clean:
        # Clean up excessive whitespace
        result = re.sub(r'\n{3,}', '\n\n', result)
        result = re.sub(r' {2,}', ' ', result)
    
    return result.strip()


def _populate_prompt_templates(parent):
    """Populate the prompt templates dropdown with built-in and user presets."""
    import json
    from pathlib import Path
    
    parent.training_prompt_combo.clear()
    
    # Built-in templates
    parent.training_prompt_combo.addItem("-- Select a template --", None)
    parent.training_prompt_combo.addItem("Simple Assistant", "simple")
    parent.training_prompt_combo.addItem("Full (with tools)", "full")
    parent.training_prompt_combo.addItem("Enigma AI Engine Complete", "Enigma AI Engine_full")
    parent.training_prompt_combo.addItem("Q&A Format Example", "qa_example")
    parent.training_prompt_combo.addItem("Conversation Format", "conversation")
    parent.training_prompt_combo.addItem("Custom (from Settings)", "custom")
    
    # Load user presets from user_prompts.json
    user_presets_path = Path(CONFIG.get("data_dir", "data")) / "user_prompts.json"
    if user_presets_path.exists():
        try:
            with open(user_presets_path, encoding='utf-8') as f:
                user_presets = json.load(f)
            for name in user_presets.keys():
                parent.training_prompt_combo.addItem(f"[User] {name}", f"user_{name}")
        except Exception as e:
            logger.debug("Failed to load user presets from %s: %s", user_presets_path, e)


def _get_template_content(template_id):
    """Get the content of a template by ID."""
    import json
    from pathlib import Path
    
    templates = {
        "simple": "You are a helpful AI assistant. Answer questions clearly and conversationally. Be friendly and helpful.",
        
        "full": """You are Enigma AI Engine, an intelligent AI assistant with access to various tools and capabilities.

## Tool Usage
When you need to perform an action, use this format:
<tool_call>{"tool": "tool_name", "params": {"param1": "value1"}}</tool_call>

## Available Tools
- generate_image: Create an image from text
- generate_code: Generate code for a task
- read_file: Read contents of a file
- web_search: Search the web

Be helpful, accurate, and respect user privacy.""",

        "Enigma AI Engine_full": """You are the AI assistant for Enigma AI Engine, a modular AI framework.

## Avatar System
- A 3D avatar appears on screen that you can control
- Control bones: head, neck, chest, shoulders, arms, legs

## Generation Tools
Use <tool_call>{"tool": "name", "params": {}}</tool_call> format:
- generate_image, generate_video, generate_code, generate_audio

## Interaction Style
- Be friendly and conversational
- Explain what you're doing before using tools""",

        "qa_example": """Q: What is your name?
A: I am Enigma AI Engine, your helpful AI assistant.

Q: What can you help me with?
A: I can help with coding, creative tasks, answering questions, and more.

Q: How do I train you?
A: Add more Q&A pairs like this to teach me new information!""",

        "conversation": """User: Hello there!
Assistant: Hi! How can I help you today?

User: I have a question about coding.
Assistant: Of course! I'd be happy to help with coding. What would you like to know?""",
    }
    
    if template_id in templates:
        return templates[template_id]
    
    # Check for user preset
    if template_id and template_id.startswith("user_"):
        preset_name = template_id[5:]
        user_presets_path = Path(CONFIG.get("data_dir", "data")) / "user_prompts.json"
        if user_presets_path.exists():
            try:
                with open(user_presets_path, encoding='utf-8') as f:
                    user_presets = json.load(f)
                if preset_name in user_presets:
                    return user_presets[preset_name]
            except Exception as e:
                logger.debug("Failed to load user preset '%s': %s", preset_name, e)
    
    # Check for custom from gui_settings
    if template_id == "custom":
        settings_path = Path(CONFIG.get("data_dir", "data")) / "gui_settings.json"
        if settings_path.exists():
            try:
                with open(settings_path, encoding='utf-8') as f:
                    settings = json.load(f)
                return settings.get("system_prompt", "")
            except Exception as e:
                logger.debug("Failed to load custom template from gui_settings: %s", e)
    
    return ""


def _preview_prompt_template(parent):
    """Preview the selected template."""
    template_id = parent.training_prompt_combo.currentData()
    if not template_id:
        parent.prompt_preview.clear()
        return
    
    content = _get_template_content(template_id)
    # Show first 300 chars with ellipsis
    preview = content[:300] + "..." if len(content) > 300 else content
    parent.prompt_preview.setText(preview)


def _insert_prompt_template(parent, position="cursor"):
    """Insert the selected template into the training editor."""
    template_id = parent.training_prompt_combo.currentData()
    if not template_id:
        QMessageBox.information(parent, "No Template", "Please select a template first.")
        return
    
    content = _get_template_content(template_id)
    if not content:
        return
    
    # Add header comment
    template_name = parent.training_prompt_combo.currentText()
    insert_text = f"# System Prompt: {template_name}\n{content}\n\n"
    
    if position == "start":
        current = parent.training_editor.toPlainText()
        parent.training_editor.setPlainText(insert_text + current)
    else:  # cursor
        cursor = parent.training_editor.textCursor()
        cursor.insertText(insert_text)


def _wrap_selection_qa(parent):
    """Wrap the selected text in Q&A format."""
    cursor = parent.training_editor.textCursor()
    selected = cursor.selectedText()
    
    if not selected:
        QMessageBox.information(parent, "No Selection", "Select some text to wrap in Q&A format.")
        return
    
    # Replace Unicode paragraph separators with newlines
    selected = selected.replace('\u2029', '\n')
    
    # Try to split into Q and A
    lines = selected.strip().split('\n')
    
    if len(lines) >= 2:
        # Assume first line is question, rest is answer
        question = lines[0].strip()
        answer = '\n'.join(lines[1:]).strip()
        wrapped = f"Q: {question}\nA: {answer}\n"
    else:
        # Single line - make it a question with placeholder answer
        wrapped = f"Q: {selected.strip()}\nA: [Your answer here]\n"
    
    cursor.insertText(wrapped)


# ============================================================================
# Training Data Preview Functions
# ============================================================================

def _analyze_training_data(parent):
    """Analyze training data and show parsed Q&A pairs."""
    content = parent.training_editor.toPlainText()
    
    if not content.strip():
        parent.data_preview_text.setPlainText("No training data to analyze.")
        parent.data_stats_label.setText("Empty")
        return
    
    # Parse Q&A pairs (handles multi-line answers and code blocks)
    pairs = []
    lines = content.split('\n')
    current_q = None
    current_a = []
    in_answer = False  # Track answer mode even when first line is empty
    
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith('Q:'):
            # Save previous pair if exists
            if current_q is not None and in_answer:
                pairs.append((current_q, '\n'.join(current_a).strip()))
            current_q = line_stripped[2:].strip()
            current_a = []
            in_answer = False
        elif line_stripped.startswith('A:') and current_q is not None:
            in_answer = True
            first_line = line_stripped[2:].strip()
            if first_line:
                current_a.append(first_line)
        elif current_q is not None and in_answer:
            # Continue collecting answer (preserve indentation for code blocks)
            current_a.append(line.rstrip())
    
    # Save last pair
    if current_q is not None and in_answer:
        pairs.append((current_q, '\n'.join(current_a).strip()))
    
    # Build preview HTML
    preview_html = []
    for i, (q, a) in enumerate(pairs[:10], 1):  # Show first 10
        q_short = q[:60] + "..." if len(q) > 60 else q
        a_short = a[:80] + "..." if len(a) > 80 else a
        preview_html.append(
            f"<p><b style='color: #89b4fa;'>Q{i}:</b> {q_short}<br>"
            f"<b style='color: #a6e3a1;'>A{i}:</b> {a_short}</p>"
        )
    
    if len(pairs) > 10:
        preview_html.append(f"<p style='color: #6c7086;'>... and {len(pairs) - 10} more pairs</p>")
    
    if not pairs:
        preview_html.append("<p style='color: #f38ba8;'>No Q&A pairs detected. Check your format:</p>")
        preview_html.append("<p style='color: #a6adc8;'>Q: Your question here<br>A: Your answer here</p>")
    
    parent.data_preview_text.setHtml('\n'.join(preview_html))
    
    # Update stats
    char_count = len(content)
    word_count = len(content.split())
    parent.data_stats_label.setText(f"{len(pairs)} pairs | {word_count:,} words | {char_count:,} chars")


def _validate_training_data(parent):
    """Validate training data for common issues."""
    content = parent.training_editor.toPlainText()
    
    if not content.strip():
        QMessageBox.warning(parent, "Empty", "No training data to validate.")
        return
    
    issues = []
    lines = content.split('\n')
    
    # Check for orphan Q: or A:
    prev_was_q = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('Q:'):
            if prev_was_q:
                issues.append(f"Line {i-1}: Question without answer")
            prev_was_q = True
        elif stripped.startswith('A:'):
            if not prev_was_q:
                issues.append(f"Line {i}: Answer without question")
            prev_was_q = False
        elif stripped and prev_was_q and not stripped.startswith('#'):
            # Non-empty line after Q but no A: marker
            pass  # Allow continuation
    
    # Check for very short answers (parse with code block support)
    pairs = []
    current_q = None
    current_a = []
    in_answer = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('Q:'):
            if current_q is not None and in_answer:
                pairs.append((current_q, '\n'.join(current_a).strip()))
            current_q = stripped[2:].strip()
            current_a = []
            in_answer = False
        elif stripped.startswith('A:') and current_q:
            in_answer = True
            first_line = stripped[2:].strip()
            if first_line:
                current_a.append(first_line)
        elif current_q and in_answer:
            # Continue collecting (preserve code blocks)
            current_a.append(line.rstrip())
    if current_q and in_answer:
        pairs.append((current_q, '\n'.join(current_a).strip()))
    
    for i, (q, a) in enumerate(pairs, 1):
        if len(a) < 5:
            issues.append(f"Q&A pair {i}: Answer is very short ({len(a)} chars)")
        if len(q) < 3:
            issues.append(f"Q&A pair {i}: Question is very short ({len(q)} chars)")
    
    # Report results
    if issues:
        issues_text = '\n'.join(issues[:20])
        if len(issues) > 20:
            issues_text += f"\n... and {len(issues) - 20} more issues"
        QMessageBox.warning(parent, "Validation Issues", 
            f"Found {len(issues)} potential issues:\n\n{issues_text}")
    else:
        QMessageBox.information(parent, "Validation Passed",
            f"Training data looks good!\n\n"
            f"Found {len(pairs)} Q&A pairs ready for training.")


# ============================================================================
# Quick Start Helper Functions
# ============================================================================

def _load_base_knowledge(parent):
    """Load the base_knowledge.txt into the training editor."""
    from pathlib import Path
    
    base_path = Path(__file__).parent.parent.parent.parent / "data" / "base_knowledge.txt"
    
    if not base_path.exists():
        QMessageBox.warning(parent, "Not Found", 
            f"Base knowledge file not found at:\n{base_path}\n\n"
            "This file should contain foundational Q&A pairs.")
        return
    
    try:
        content = base_path.read_text(encoding='utf-8')
        
        # Confirm if editor has content
        current = parent.training_editor.toPlainText().strip()
        if current:
            reply = QMessageBox.question(
                parent, "Replace Content?",
                "The editor already has content. Do you want to:\n\n"
                "Yes - Replace all content\n"
                "No - Append to existing content",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.No:
                content = current + "\n\n" + content
        
        parent.training_editor.setPlainText(content)
        
        # Count Q&A pairs
        qa_count = content.count("\nQ:") + (1 if content.startswith("Q:") else 0)
        QMessageBox.information(parent, "Loaded",
            f"Loaded {qa_count} Q&A pairs from base knowledge.\n\n"
            "This covers basic self-awareness, conversation, and reasoning.\n"
            "Click 'Save' then 'Train' to train with this data!")
            
    except Exception as e:
        QMessageBox.critical(parent, "Error", f"Failed to load: {e}")


def _view_base_knowledge(parent):
    """Preview the base knowledge file in a dialog."""
    from pathlib import Path
    
    base_path = Path(__file__).parent.parent.parent.parent / "data" / "base_knowledge.txt"
    
    if not base_path.exists():
        QMessageBox.warning(parent, "Not Found", "Base knowledge file not found.")
        return
    
    try:
        content = base_path.read_text(encoding='utf-8')
        
        # Create preview dialog
        dialog = QDialog(parent)
        dialog.setWindowTitle("Base Knowledge Preview")
        dialog.resize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Info label
        qa_count = content.count("\nQ:") + (1 if content.startswith("Q:") else 0)
        info = QLabel(f"Contains {qa_count} Q&A pairs covering foundational knowledge")
        layout.addWidget(info)
        
        # Text display
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(content)
        layout.addWidget(text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()
        
    except Exception as e:
        QMessageBox.critical(parent, "Error", f"Failed to read: {e}")


def _paste_and_convert(parent):
    """Open dialog to paste text and convert to Q&A format using Trainer AI."""
    dialog = QDialog(parent)
    dialog.setWindowTitle("Convert Text to Q&A Training Data")
    dialog.resize(700, 600)
    
    layout = QVBoxLayout(dialog)
    
    # Instructions
    instructions = QLabel(
        "Paste any text (articles, documentation, notes) below.\n"
        "The Trainer AI will automatically generate Q&A pairs from it."
    )
    layout.addWidget(instructions)
    
    # Input text area
    input_label = QLabel("Input Text:")
    layout.addWidget(input_label)
    
    input_text = QTextEdit()
    input_text.setPlaceholderText(
        "Paste your text here...\n\n"
        "Examples of good content:\n"
        "- Wikipedia articles\n"
        "- Documentation\n"
        "- Study notes\n"
        "- How-to guides\n"
        "- Any factual content"
    )
    layout.addWidget(input_text)
    
    # Topic hint
    topic_layout = QHBoxLayout()
    topic_layout.addWidget(QLabel("Topic Hint (optional):"))
    topic_input = QLineEdit()
    topic_input.setPlaceholderText("e.g., Python programming, History, Science")
    topic_layout.addWidget(topic_input)
    layout.addLayout(topic_layout)
    
    # Output area
    output_label = QLabel("Generated Q&A Pairs:")
    layout.addWidget(output_label)
    
    output_text = QTextEdit()
    output_text.setReadOnly(True)
    output_text.setPlaceholderText("Q&A pairs will appear here after conversion...")
    layout.addWidget(output_text)
    
    # Buttons
    btn_layout = QHBoxLayout()
    
    convert_btn = QPushButton("Convert to Q&A")
    add_btn = QPushButton("Add to Training Data")
    add_btn.setEnabled(False)
    close_btn = QPushButton("Close")
    
    btn_layout.addWidget(convert_btn)
    btn_layout.addWidget(add_btn)
    btn_layout.addStretch()
    btn_layout.addWidget(close_btn)
    layout.addLayout(btn_layout)
    
    def do_convert():
        text = input_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(dialog, "No Text", "Please paste some text to convert.")
            return
        
        topic = topic_input.text().strip() or None
        
        convert_btn.setEnabled(False)
        convert_btn.setText("Converting...")
        QApplication.processEvents()
        
        try:
            # Try to use Trainer AI
            from enigma_engine.core.tool_router import get_router
            router = get_router()
            qa_pairs = router.generate_training_data(text, topic=topic, num_pairs=10)
            
            if qa_pairs:
                # Format output
                output = ""
                for q, a in qa_pairs:
                    output += f"Q: {q}\nA: {a}\n\n"
                output_text.setPlainText(output.strip())
                add_btn.setEnabled(True)
            else:
                output_text.setPlainText("No Q&A pairs could be generated. Try different text.")
                
        except Exception as e:
            # Fallback: simple rule-based extraction
            output_text.setPlainText(f"Trainer AI unavailable, using basic extraction...\n\n")
            _do_basic_extraction(text, output_text)
            add_btn.setEnabled(True)
            
        finally:
            convert_btn.setEnabled(True)
            convert_btn.setText("Convert to Q&A")
    
    def _do_basic_extraction(text, output_widget):
        """Basic Q&A extraction without AI (preserves code blocks)."""
        import re
        
        qa_output = ""
        
        # First, extract and preserve code blocks
        code_blocks = []
        code_pattern = r'```[\w]*\n(.*?)```'
        
        def save_code(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"
        
        text_no_code = re.sub(code_pattern, save_code, text, flags=re.DOTALL)
        
        # Split into paragraphs (not sentences to preserve code context)
        paragraphs = re.split(r'\n\s*\n', text_no_code)
        
        for i, para in enumerate(paragraphs[:15]):  # Limit to 15 paragraphs
            para = para.strip()
            if len(para) < 20:  # Skip very short paragraphs
                continue
            
            # Restore code blocks in paragraph
            for j, block in enumerate(code_blocks):
                para = para.replace(f"__CODE_BLOCK_{j}__", block)
            
            # Create Q&A from paragraph
            words = para.split()[:5]  # First 5 words for topic
            topic = ' '.join(w for w in words if w.isalpha())[:30]
            
            if topic:
                qa_output += f"Q: Explain about {topic.lower()}?\n"
                qa_output += f"A: {para}\n\n"
        
        current = output_widget.toPlainText()
        output_widget.setPlainText(current + qa_output)
    
    def do_add():
        qa_content = output_text.toPlainText().strip()
        if not qa_content:
            return
        
        # Add to training editor
        current = parent.training_editor.toPlainText()
        if current.strip():
            parent.training_editor.setPlainText(current + "\n\n" + qa_content)
        else:
            parent.training_editor.setPlainText(qa_content)
        
        QMessageBox.information(dialog, "Added", 
            "Q&A pairs added to training data.\n"
            "Click 'Save' then 'Train' to train the model!")
        dialog.accept()
    
    convert_btn.clicked.connect(do_convert)
    add_btn.clicked.connect(do_add)
    close_btn.clicked.connect(dialog.reject)
    
    dialog.exec_()


def _quick_train(parent):
    """Start training with recommended beginner settings."""
    # Verify there's training data
    content = parent.training_editor.toPlainText().strip()
    
    if not content:
        QMessageBox.warning(parent, "No Training Data",
            "Please add some training data first!\n\n"
            "Options:\n"
            "1. Click 'Load Base Knowledge' to load starter data\n"
            "2. Click 'Paste Text -> Q&A' to convert any text\n"
            "3. Manually type Q&A pairs")
        return
    
    # Count Q&A pairs
    qa_count = content.count("\nQ:") + (1 if content.startswith("Q:") else 0)
    
    if qa_count < 10:
        reply = QMessageBox.question(parent, "Limited Data",
            f"Only {qa_count} Q&A pairs found.\n\n"
            "For best results, we recommend at least 50+ pairs.\n"
            "Training with limited data may not produce good results.\n\n"
            "Continue anyway?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No:
            return
    
    # Check if data is saved
    reply = QMessageBox.information(parent, "Quick Train",
        f"Ready to train with {qa_count} Q&A pairs!\n\n"
        "Recommended settings for beginners:\n"
        "- Epochs: 3 (more = better but slower)\n"
        "- Learning Rate: 0.0001 (safe default)\n"
        "- Batch Size: 4 (works on most hardware)\n\n"
        "The training will:\n"
        "1. Save your training data\n"
        "2. Train the model (may take a few minutes)\n"
        "3. Show progress in the log\n\n"
        "Start training?",
        QMessageBox.Ok | QMessageBox.Cancel)
    
    if reply == QMessageBox.Cancel:
        return
    
    # Save first
    _save_training_data(parent)
    
    # Set recommended values
    if hasattr(parent, 'epochs_input'):
        parent.epochs_input.setValue(3)
    if hasattr(parent, 'batch_input'):
        parent.batch_input.setValue(4)
    if hasattr(parent, 'lr_input'):
        parent.lr_input.setValue(0.0001)
    
    # Start training
    _start_training(parent)


# =============================================================================
# SYSTEM PROMPT EDITOR FUNCTIONS
# =============================================================================

# Presets for system prompts
SYSTEM_PROMPT_PRESETS = {
    "default": (
        "You are Forge, a helpful AI assistant. "
        "You provide clear, accurate, and helpful responses. "
        "You are friendly but professional."
    ),
    "expert": (
        "You are an expert AI assistant with deep knowledge across many domains. "
        "You provide detailed, well-researched answers with explanations. "
        "You cite sources when relevant and acknowledge uncertainty."
    ),
    "casual": (
        "You are a friendly, casual AI buddy. "
        "You chat naturally, use simple language, and have a warm personality. "
        "You're helpful but also fun to talk to!"
    ),
    "technical": (
        "You are a precise technical assistant. "
        "You provide accurate, detailed technical information. "
        "You use proper terminology and provide code examples when relevant. "
        "You prioritize correctness over brevity."
    ),
    "creative": (
        "You are a creative AI with an artistic soul. "
        "You help with writing, storytelling, and creative projects. "
        "You're imaginative, expressive, and encourage creativity. "
        "You can write in various styles and help brainstorm ideas."
    ),
    "custom": ""  # User will provide their own
}


def _load_system_prompt_preset(parent):
    """Load a system prompt preset into the editor."""
    if not hasattr(parent, 'system_prompt_preset'):
        return
    
    preset_key = parent.system_prompt_preset.currentData()
    if preset_key and preset_key != "custom":
        prompt = SYSTEM_PROMPT_PRESETS.get(preset_key, "")
        if prompt:
            parent.system_prompt_editor.setPlainText(prompt)


def _load_prompt_from_persona(parent):
    """Load system prompt from an existing persona."""
    try:
        from ...core.persona import PersonaManager
        
        manager = PersonaManager()
        personas = manager.list_personas()
        
        if not personas:
            QMessageBox.information(parent, "No Personas", 
                "No personas found. Create one in the Persona tab first.")
            return
        
        # Create selection dialog
        dialog = QDialog(parent)
        dialog.setWindowTitle("Select Persona")
        dialog.setMinimumWidth(300)
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("Select a persona to load its system prompt:"))
        
        persona_list = QListWidget()
        for persona in personas:
            item = QListWidgetItem(persona.name)
            item.setData(Qt.UserRole, persona.id)
            persona_list.addItem(item)
        layout.addWidget(persona_list)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            selected = persona_list.currentItem()
            if selected:
                persona_id = selected.data(Qt.UserRole)
                persona = manager.get_persona(persona_id)
                if persona and persona.system_prompt:
                    parent.system_prompt_editor.setPlainText(persona.system_prompt)
                    parent.system_prompt_preset.setCurrentIndex(
                        parent.system_prompt_preset.findData("custom")
                    )
                    QMessageBox.information(parent, "Loaded",
                        f"Loaded system prompt from persona: {persona.name}")
                else:
                    QMessageBox.warning(parent, "No Prompt",
                        "This persona has no system prompt defined.")
    
    except Exception as e:
        logger.warning(f"Could not load persona: {e}")
        QMessageBox.warning(parent, "Error", f"Could not load persona: {e}")


def _save_system_prompt_to_persona(parent):
    """Save current system prompt to the active persona."""
    try:
        from ...core.persona import PersonaManager
        
        prompt = parent.system_prompt_editor.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(parent, "Empty Prompt", 
                "Please enter a system prompt first.")
            return
        
        manager = PersonaManager()
        current = manager.get_current_persona()
        
        if not current:
            # Create default persona
            reply = QMessageBox.question(parent, "No Active Persona",
                "No active persona found. Create a default one?",
                QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                current = manager.create_persona(
                    name="My AI",
                    system_prompt=prompt
                )
                manager.set_current_persona(current.id)
                QMessageBox.information(parent, "Created",
                    f"Created persona 'My AI' with your system prompt.")
            return
        
        # Update current persona
        current.system_prompt = prompt
        current.last_modified = datetime.now().isoformat()
        manager.save_persona(current)
        
        QMessageBox.information(parent, "Saved",
            f"System prompt saved to persona: {current.name}")
    
    except Exception as e:
        logger.warning(f"Could not save to persona: {e}")
        QMessageBox.warning(parent, "Error", f"Could not save: {e}")


def _inject_system_prompt(parent):
    """Inject system prompt into training data."""
    prompt = parent.system_prompt_editor.toPlainText().strip()
    if not prompt:
        QMessageBox.warning(parent, "Empty Prompt",
            "Please enter a system prompt first.")
        return
    
    # Format as training data header
    header = f"# System Prompt\n# {prompt}\n\n"
    
    current = parent.training_editor.toPlainText()
    
    # Check if already has system prompt
    if current.strip().startswith("# System Prompt"):
        reply = QMessageBox.question(parent, "Replace Prompt?",
            "Training data already has a system prompt.\n"
            "Replace it with the new one?",
            QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Remove old header (everything until first Q: or double newline)
            lines = current.split("\n")
            start_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("Q:") or (line.strip() and not line.startswith("#")):
                    start_idx = i
                    break
            content = "\n".join(lines[start_idx:])
            parent.training_editor.setPlainText(header + content)
    else:
        parent.training_editor.setPlainText(header + current)
    
    QMessageBox.information(parent, "Injected",
        "System prompt added to training data.\n"
        "Save and train to teach the AI this personality!")


def _test_system_prompt(parent):
    """Test AI response with current system prompt."""
    prompt = parent.system_prompt_editor.toPlainText().strip()
    if not prompt:
        QMessageBox.warning(parent, "Empty Prompt",
            "Please enter a system prompt first.")
        return
    
    # Create test dialog
    dialog = QDialog(parent)
    dialog.setWindowTitle("Test System Prompt")
    dialog.setMinimumSize(500, 400)
    layout = QVBoxLayout(dialog)
    
    # Show system prompt
    layout.addWidget(QLabel("System Prompt:"))
    prompt_display = QTextEdit()
    prompt_display.setPlainText(prompt)
    prompt_display.setReadOnly(True)
    prompt_display.setMaximumHeight(80)
    layout.addWidget(prompt_display)
    
    # Test input
    layout.addWidget(QLabel("Test Message:"))
    test_input = QLineEdit()
    test_input.setPlaceholderText("Enter a test message for the AI...")
    test_input.setText("Hello! Who are you?")
    layout.addWidget(test_input)
    
    # Response area
    layout.addWidget(QLabel("AI Response:"))
    response_area = QTextEdit()
    response_area.setReadOnly(True)
    response_area.setPlaceholderText("Click 'Test' to see how the AI responds...")
    layout.addWidget(response_area)
    
    # Buttons
    btn_layout = QHBoxLayout()
    test_btn = QPushButton("Test")
    close_btn = QPushButton("Close")
    btn_layout.addWidget(test_btn)
    btn_layout.addStretch()
    btn_layout.addWidget(close_btn)
    layout.addLayout(btn_layout)
    
    def do_test():
        test_msg = test_input.text().strip()
        if not test_msg:
            return
        
        response_area.setPlainText("Generating response...")
        QApplication.processEvents()
        
        try:
            # Try to use the inference engine
            from ...core.inference import EnigmaEngine
            
            engine = EnigmaEngine()
            
            # Format with system prompt
            full_prompt = f"System: {prompt}\n\nUser: {test_msg}\n\nAssistant:"
            
            response = engine.generate(full_prompt, max_gen=150)
            
            # Clean up response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            response_area.setPlainText(response)
            
        except Exception as e:
            response_area.setPlainText(
                f"Could not generate response: {e}\n\n"
                "Note: Testing requires a trained model.\n"
                "Train your model first, then test the prompt."
            )
    
    test_btn.clicked.connect(do_test)
    close_btn.clicked.connect(dialog.accept)
    
    dialog.exec_()
