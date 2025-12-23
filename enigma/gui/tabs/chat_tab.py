"""Chat tab for Enigma Engine GUI."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLineEdit
)


def create_chat_tab(parent):
    """Create the chat interface tab."""
    w = QWidget()
    layout = QVBoxLayout()
    
    # Chat display
    parent.chat_display = QTextEdit()
    parent.chat_display.setReadOnly(True)
    layout.addWidget(parent.chat_display)
    
    # Input row
    input_layout = QHBoxLayout()
    parent.chat_input = QLineEdit()
    parent.chat_input.setPlaceholderText("Ask something...")
    parent.chat_input.returnPressed.connect(parent._on_send)
    
    parent.send_btn = QPushButton("Send")
    parent.send_btn.clicked.connect(parent._on_send)
    
    input_layout.addWidget(parent.chat_input)
    input_layout.addWidget(parent.send_btn)
    layout.addLayout(input_layout)
    
    # Initialize auto_speak
    parent.auto_speak = False
    
    w.setLayout(layout)
    return w
