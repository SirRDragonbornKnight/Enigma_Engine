"""Train tab for Enigma Engine GUI."""

import re
import urllib.request
import urllib.error
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QLineEdit, QProgressBar, QFileDialog,
    QPlainTextEdit, QMessageBox, QInputDialog, QGroupBox,
    QFrame, QDialog, QTextEdit, QDialogButtonBox, QCheckBox
)
from PyQt5.QtCore import Qt

from ...config import CONFIG


def create_training_tab(parent):
    """Create the train tab with model training controls."""
    w = QWidget()
    layout = QVBoxLayout()
    layout.setSpacing(12)
    layout.setContentsMargins(10, 10, 10, 10)
    
    # Header with model info
    header_layout = QHBoxLayout()
    
    header = QLabel("Train Your AI")
    header.setObjectName("header")
    header.setStyleSheet("font-size: 16px; font-weight: bold;")
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
    params_layout.setSpacing(15)
    
    # Epochs
    epochs_layout = QVBoxLayout()
    epochs_label = QLabel("Epochs")
    epochs_label.setStyleSheet("font-size: 11px; color: #6c7086;")
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
    batch_label.setStyleSheet("font-size: 11px; color: #6c7086;")
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
    lr_label.setStyleSheet("font-size: 11px; color: #6c7086;")
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
    parent.training_progress_label.setStyleSheet("color: #6c7086;")
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
    parent.btn_train.setStyleSheet("""
        QPushButton {
            padding: 12px 24px;
            font-size: 14px;
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
    parent.btn_stop_train.setToolTip("Stop training after current epoch")
    parent.btn_stop_train.clicked.connect(parent._on_stop_training)
    parent.btn_stop_train.setEnabled(False)
    parent.btn_stop_train.setStyleSheet("""
        QPushButton {
            padding: 12px 24px;
            font-size: 14px;
            background-color: #f38ba8;
            color: #1e1e2e;
        }
        QPushButton:disabled {
            background-color: #45475a;
            color: #6c7086;
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
        try:
            # Fetch the webpage
            headers = {"User-Agent": "Mozilla/5.0 (compatible; EnigmaBot/1.0)"}
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
            
            preview_text.setPlainText(content)
            
        except urllib.error.HTTPError as e:
            preview_text.setPlainText(f"HTTP Error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            preview_text.setPlainText(f"Network Error: {e}")
        except Exception as e:
            preview_text.setPlainText(f"Error: {e}")
    
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
