"""Chat tab for Enigma Engine GUI."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLineEdit, QLabel, QFrame, QSplitter,
    QGroupBox, QSizePolicy
)
from PyQt5.QtCore import Qt


def create_chat_tab(parent):
    """Create the chat interface tab with improved UX."""
    w = QWidget()
    layout = QVBoxLayout()
    layout.setSpacing(10)
    layout.setContentsMargins(10, 10, 10, 10)
    
    # Header with model info and controls
    header_layout = QHBoxLayout()
    
    # Model indicator - check if model is already loaded
    initial_model_text = "No model loaded"
    if hasattr(parent, 'current_model_name') and parent.current_model_name:
        initial_model_text = f"[AI] {parent.current_model_name}"
    parent.chat_model_label = QLabel(initial_model_text)
    parent.chat_model_label.setStyleSheet("""
        QLabel {
            color: #89b4fa;
            font-weight: bold;
            font-size: 13px;
            padding: 4px 8px;
            background: rgba(137, 180, 250, 0.1);
            border-radius: 4px;
        }
    """)
    header_layout.addWidget(parent.chat_model_label)
    
    header_layout.addStretch()
    
    # Quick action buttons
    parent.btn_clear_chat = QPushButton("Clear")
    parent.btn_clear_chat.setToolTip("Clear chat history")
    parent.btn_clear_chat.setMaximumWidth(70)
    parent.btn_clear_chat.clicked.connect(lambda: _clear_chat(parent))
    parent.btn_clear_chat.setStyleSheet("""
        QPushButton {
            background-color: #45475a;
            padding: 4px 8px;
        }
    """)
    header_layout.addWidget(parent.btn_clear_chat)
    
    parent.btn_save_chat = QPushButton("Save")
    parent.btn_save_chat.setToolTip("Save conversation")
    parent.btn_save_chat.setMaximumWidth(70)
    parent.btn_save_chat.clicked.connect(lambda: _save_chat(parent))
    parent.btn_save_chat.setStyleSheet("""
        QPushButton {
            background-color: #45475a;
            padding: 4px 8px;
        }
    """)
    header_layout.addWidget(parent.btn_save_chat)
    
    layout.addLayout(header_layout)
    
    # Chat display - selectable text with better styling
    parent.chat_display = QTextEdit()
    parent.chat_display.setReadOnly(True)
    parent.chat_display.setTextInteractionFlags(
        Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
    )
    parent.chat_display.setPlaceholderText(
        "Start chatting with your AI...\n\n"
        "Tips:\n"
        "- Train your model first in the Train tab\n"
        "- Use Q&A format for best results\n"
        "- The AI learns from your conversations"
    )
    parent.chat_display.setStyleSheet("""
        QTextEdit {
            font-size: 13px;
            line-height: 1.5;
            padding: 10px;
        }
    """)
    layout.addWidget(parent.chat_display, stretch=1)
    
    # Input area with better layout
    input_frame = QFrame()
    input_frame.setStyleSheet("""
        QFrame {
            background: rgba(49, 50, 68, 0.5);
            border-radius: 8px;
            padding: 8px;
        }
    """)
    input_layout = QHBoxLayout(input_frame)
    input_layout.setContentsMargins(8, 8, 8, 8)
    input_layout.setSpacing(8)
    
    # Text input
    parent.chat_input = QLineEdit()
    parent.chat_input.setPlaceholderText("Type your message here... (Press Enter to send)")
    parent.chat_input.returnPressed.connect(parent._on_send)
    parent.chat_input.setStyleSheet("""
        QLineEdit {
            padding: 10px 12px;
            font-size: 13px;
            border-radius: 6px;
        }
    """)
    input_layout.addWidget(parent.chat_input, stretch=1)
    
    # Speak button (for TTS)
    parent.btn_speak = QPushButton("Voice")
    parent.btn_speak.setToolTip("Speak last response")
    parent.btn_speak.setMinimumWidth(65)
    parent.btn_speak.setMaximumWidth(70)
    parent.btn_speak.clicked.connect(parent._on_speak_last)
    input_layout.addWidget(parent.btn_speak)
    
    # Send button
    parent.send_btn = QPushButton("Send")
    parent.send_btn.clicked.connect(parent._on_send)
    parent.send_btn.setStyleSheet("""
        QPushButton {
            padding: 10px 20px;
            font-weight: bold;
            min-width: 80px;
        }
    """)
    input_layout.addWidget(parent.send_btn)
    
    layout.addWidget(input_frame)
    
    # Status bar at bottom
    status_layout = QHBoxLayout()
    
    parent.chat_status = QLabel("")
    parent.chat_status.setStyleSheet("color: #6c7086; font-size: 11px;")
    status_layout.addWidget(parent.chat_status)
    
    status_layout.addStretch()
    
    # Learning indicator
    parent.learning_indicator = QLabel("Learning: ON")
    parent.learning_indicator.setStyleSheet("color: #a6e3a1; font-size: 11px;")
    parent.learning_indicator.setToolTip("AI learns from conversations when enabled")
    status_layout.addWidget(parent.learning_indicator)
    
    layout.addLayout(status_layout)
    
    # Initialize auto_speak
    parent.auto_speak = False
    
    w.setLayout(layout)
    return w


def _clear_chat(parent):
    """Clear the chat display and history."""
    parent.chat_display.clear()
    parent.chat_messages = []
    parent.chat_status.setText("Chat cleared")


def _save_chat(parent):
    """Save the current chat session."""
    if hasattr(parent, '_save_current_chat'):
        parent._save_current_chat()
        parent.chat_status.setText("Chat saved!")
    else:
        parent.chat_status.setText("Save not available")
