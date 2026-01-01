"""AI Terminal tab for Enigma Engine GUI - view AI processing in real-time."""

from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QPlainTextEdit, QCheckBox, QComboBox, QGroupBox, QSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QTextCursor


def create_terminal_tab(parent):
    """Create the AI terminal visualization tab."""
    w = QWidget()
    layout = QVBoxLayout()
    
    # Terminal header
    header_layout = QHBoxLayout()
    
    title = QLabel("AI Processing Terminal")
    title.setStyleSheet("font-weight: bold; font-size: 14px;")
    header_layout.addWidget(title)
    
    header_layout.addStretch()
    
    # Auto-scroll checkbox
    parent.terminal_autoscroll = QCheckBox("Auto-scroll")
    parent.terminal_autoscroll.setChecked(True)
    header_layout.addWidget(parent.terminal_autoscroll)
    
    # Clear button
    clear_btn = QPushButton("Clear")
    clear_btn.clicked.connect(lambda: parent._clear_terminal())
    header_layout.addWidget(clear_btn)
    
    layout.addLayout(header_layout)
    
    # Main terminal display
    parent.terminal_output = QPlainTextEdit()
    parent.terminal_output.setReadOnly(True)
    parent.terminal_output.setTextInteractionFlags(
        Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
    )
    
    # Monospace font for terminal feel
    font = QFont("Consolas", 10)
    if not font.exactMatch():
        font = QFont("Courier New", 10)
    parent.terminal_output.setFont(font)
    
    # Terminal styling
    parent.terminal_output.setStyleSheet("""
        QPlainTextEdit {
            background-color: #0d1117;
            color: #58a6ff;
            border: 1px solid #30363d;
            padding: 8px;
        }
    """)
    
    parent.terminal_output.setPlaceholderText(
        "AI processing logs will appear here...\n\n"
        "Shows:\n"
        "  - Token processing\n"
        "  - Inference steps\n"
        "  - Model activations\n"
        "  - Generation progress\n"
        "  - Error messages"
    )
    layout.addWidget(parent.terminal_output)
    
    # Stats panel
    stats_group = QGroupBox("Live Statistics")
    stats_layout = QHBoxLayout()
    
    # Tokens per second
    parent.terminal_tps_label = QLabel("Tokens/sec: --")
    stats_layout.addWidget(parent.terminal_tps_label)
    
    # Memory usage
    parent.terminal_memory_label = QLabel("Memory: --")
    stats_layout.addWidget(parent.terminal_memory_label)
    
    # Active model
    parent.terminal_model_label = QLabel("Model: --")
    stats_layout.addWidget(parent.terminal_model_label)
    
    stats_layout.addStretch()
    stats_group.setLayout(stats_layout)
    layout.addWidget(stats_group)
    
    # Control panel
    control_group = QGroupBox("Logging Controls")
    control_layout = QHBoxLayout()
    
    # Log level
    control_layout.addWidget(QLabel("Log Level:"))
    parent.terminal_log_level = QComboBox()
    parent.terminal_log_level.addItems(["All", "Info", "Debug", "Tokens", "Errors Only"])
    parent.terminal_log_level.currentTextChanged.connect(
        lambda: parent._update_terminal_filter()
    )
    control_layout.addWidget(parent.terminal_log_level)
    
    control_layout.addStretch()
    
    # Max lines
    control_layout.addWidget(QLabel("Max Lines:"))
    parent.terminal_max_lines = QSpinBox()
    parent.terminal_max_lines.setRange(100, 10000)
    parent.terminal_max_lines.setValue(1000)
    parent.terminal_max_lines.setSingleStep(100)
    control_layout.addWidget(parent.terminal_max_lines)
    
    control_group.setLayout(control_layout)
    layout.addWidget(control_group)
    
    # Initialize terminal state
    parent._terminal_lines = []
    parent._terminal_log_level = "All"
    
    w.setLayout(layout)
    return w


def log_to_terminal(parent, message, level="info"):
    """
    Log a message to the AI terminal.
    
    Args:
        parent: The main window with terminal_output
        message: The message to log
        level: One of 'info', 'debug', 'token', 'error', 'success'
    """
    if not hasattr(parent, 'terminal_output'):
        return
    
    # Filter by log level
    current_filter = getattr(parent, '_terminal_log_level', 'All')
    if current_filter == "Errors Only" and level != "error":
        return
    if current_filter == "Tokens" and level not in ("token", "error"):
        return
    if current_filter == "Debug" and level not in ("debug", "error", "info"):
        return
    if current_filter == "Info" and level not in ("info", "error", "success"):
        return
    
    # Format timestamp
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    # Level prefixes
    prefixes = {
        "info": "[INFO]",
        "debug": "[DEBUG]",
        "token": "[TOKEN]",
        "error": "[ERROR]",
        "success": "[OK]"
    }
    prefix = prefixes.get(level, "[LOG]")
    
    # Format line
    line = f"{timestamp} {prefix} {message}"
    
    # Store line
    if not hasattr(parent, '_terminal_lines'):
        parent._terminal_lines = []
    parent._terminal_lines.append(line)
    
    # Trim to max lines
    max_lines = getattr(parent, 'terminal_max_lines', None)
    if max_lines and hasattr(max_lines, 'value'):
        max_val = max_lines.value()
        if len(parent._terminal_lines) > max_val:
            parent._terminal_lines = parent._terminal_lines[-max_val:]
    
    # Append to display
    parent.terminal_output.appendPlainText(line)
    
    # Auto-scroll
    if hasattr(parent, 'terminal_autoscroll') and parent.terminal_autoscroll.isChecked():
        cursor = parent.terminal_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        parent.terminal_output.setTextCursor(cursor)
