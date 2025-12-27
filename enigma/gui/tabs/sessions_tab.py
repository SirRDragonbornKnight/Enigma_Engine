"""History tab for Enigma Engine GUI - view saved chat sessions per AI."""

import json
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QTextEdit, QMessageBox, QInputDialog, QSplitter,
    QComboBox
)
from PyQt5.QtCore import Qt

from ...config import CONFIG


def create_sessions_tab(parent):
    """Create the history tab - with AI selector and side by side chat viewer."""
    w = QWidget()
    layout = QVBoxLayout()
    
    # Header
    header = QLabel("Chat History")
    header.setObjectName("header")
    layout.addWidget(header)
    
    # AI selector row
    ai_layout = QHBoxLayout()
    ai_layout.addWidget(QLabel("Select AI:"))
    parent.history_ai_selector = QComboBox()
    parent.history_ai_selector.setMinimumWidth(150)
    parent.history_ai_selector.currentTextChanged.connect(parent._on_history_ai_changed)
    ai_layout.addWidget(parent.history_ai_selector)
    ai_layout.addStretch()
    layout.addLayout(ai_layout)
    
    # Session controls row
    btn_layout = QHBoxLayout()
    
    btn_new_session = QPushButton("New")
    btn_new_session.setToolTip("Start a new chat session")
    btn_new_session.clicked.connect(parent._new_session)
    
    btn_rename_session = QPushButton("Rename")
    btn_rename_session.setToolTip("Rename selected session")
    btn_rename_session.clicked.connect(parent._rename_session)
    
    btn_delete_session = QPushButton("Delete")
    btn_delete_session.setToolTip("Delete selected session")
    btn_delete_session.clicked.connect(parent._delete_session)
    
    btn_refresh = QPushButton("Refresh")
    btn_refresh.setToolTip("Refresh session list")
    btn_refresh.clicked.connect(parent._refresh_sessions)
    
    btn_layout.addWidget(btn_new_session)
    btn_layout.addWidget(btn_rename_session)
    btn_layout.addWidget(btn_delete_session)
    btn_layout.addWidget(btn_refresh)
    btn_layout.addStretch()
    layout.addLayout(btn_layout)
    
    # Side by side: Content on LEFT, Sessions on RIGHT
    splitter = QSplitter(Qt.Horizontal)
    
    # LEFT: Session content viewer
    left_widget = QWidget()
    left_layout = QVBoxLayout()
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_label = QLabel("Conversation:")
    left_layout.addWidget(left_label)
    parent.session_viewer = QTextEdit()
    parent.session_viewer.setReadOnly(True)
    parent.session_viewer.setTextInteractionFlags(
        Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
    )
    parent.session_viewer.setPlaceholderText("Select a saved chat to view...")
    parent.session_viewer.setTextInteractionFlags(
        Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
    )
    left_layout.addWidget(parent.session_viewer)
    left_widget.setLayout(left_layout)
    splitter.addWidget(left_widget)
    
    # RIGHT: Session list
    right_widget = QWidget()
    right_layout = QVBoxLayout()
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_label = QLabel("Saved Chats:")
    right_layout.addWidget(right_label)
    parent.sessions_list = QListWidget()
    parent.sessions_list.itemClicked.connect(parent._load_session)
    right_layout.addWidget(parent.sessions_list)
    right_widget.setLayout(right_layout)
    splitter.addWidget(right_widget)
    
    # Set initial sizes (content larger than list)
    splitter.setSizes([400, 200])
    layout.addWidget(splitter, stretch=1)
    
    # Populate AI selector and refresh sessions list
    parent._populate_history_ai_selector()
    parent._refresh_sessions()
    
    w.setLayout(layout)
    return w
