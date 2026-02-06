"""Train tab for ForgeAI GUI."""

import logging
import re
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
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
    
    params_layout.addStretch()
    layout.addWidget(params_group)
    
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
    pass


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
    parent.training_prompt_combo.addItem("ForgeAI Complete", "forgeai_full")
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
        
        "full": """You are ForgeAI, an intelligent AI assistant with access to various tools and capabilities.

## Tool Usage
When you need to perform an action, use this format:
<tool_call>{"tool": "tool_name", "params": {"param1": "value1"}}</tool_call>

## Available Tools
- generate_image: Create an image from text
- generate_code: Generate code for a task
- read_file: Read contents of a file
- web_search: Search the web

Be helpful, accurate, and respect user privacy.""",

        "forgeai_full": """You are the AI assistant for ForgeAI, a modular AI framework.

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
A: I am ForgeAI, your helpful AI assistant.

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
            from forge_ai.core.tool_router import get_router
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
        """Basic Q&A extraction without AI."""
        lines = text.split('.')
        qa_output = ""
        
        for i, sentence in enumerate(lines[:15]):  # Limit to 15 sentences
            sentence = sentence.strip()
            if len(sentence) > 20:  # Skip very short sentences
                # Create a question from the sentence
                words = sentence.split()
                if len(words) > 3:
                    # Simple pattern: "What is X?" where X is the subject
                    if words[0].lower() in ['the', 'a', 'an']:
                        subject = ' '.join(words[1:4])
                    else:
                        subject = ' '.join(words[:3])
                    
                    qa_output += f"Q: What do you know about {subject.lower()}?\n"
                    qa_output += f"A: {sentence}.\n\n"
        
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
