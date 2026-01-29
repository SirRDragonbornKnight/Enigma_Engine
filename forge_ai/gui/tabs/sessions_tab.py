"""History tab for ForgeAI GUI - view saved chat sessions per AI."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QTextEdit, QSplitter, QGroupBox, QFrame
)
from PyQt5.QtCore import Qt

from .shared_components import NoScrollComboBox



def create_sessions_tab(parent):
    """Create the history tab - with AI selector and side by side chat viewer."""
    w = QWidget()
    layout = QVBoxLayout()
    layout.setSpacing(12)
    layout.setContentsMargins(10, 10, 10, 10)
    
    # Header with AI selector
    header_layout = QHBoxLayout()
    
    header = QLabel("Chat History")
    header.setStyleSheet("font-size: 12px; font-weight: bold;")
    header_layout.addWidget(header)
    
    header_layout.addStretch()
    
    # AI selector
    header_layout.addWidget(QLabel("AI:"))
    parent.history_ai_selector = NoScrollComboBox()
    parent.history_ai_selector.setMinimumWidth(150)
    parent.history_ai_selector.setToolTip("Select AI model to view chat history")
    parent.history_ai_selector.setStyleSheet("""
        QComboBox {
            padding: 4px 8px;
            background: rgba(137, 180, 250, 0.1);
            border-radius: 4px;
        }
    """)
    parent.history_ai_selector.currentTextChanged.connect(parent._on_history_ai_changed)
    header_layout.addWidget(parent.history_ai_selector)
    
    layout.addLayout(header_layout)
    
    # Main content area with splitter
    splitter = QSplitter(Qt.Horizontal)
    
    # LEFT: Session list
    left_frame = QFrame()
    left_frame.setStyleSheet("""
        QFrame {
            background: rgba(49, 50, 68, 0.3);
            border-radius: 8px;
        }
    """)
    left_layout = QVBoxLayout(left_frame)
    left_layout.setContentsMargins(10, 10, 10, 10)
    
    list_header = QHBoxLayout()
    list_label = QLabel("Saved Chats")
    list_label.setStyleSheet("font-weight: bold;")
    list_header.addWidget(list_label)
    
    btn_refresh = QPushButton("Refresh")
    btn_refresh.setMaximumWidth(70)
    btn_refresh.setToolTip("Refresh list")
    btn_refresh.clicked.connect(parent._refresh_sessions)
    list_header.addWidget(btn_refresh)
    left_layout.addLayout(list_header)
    
    parent.sessions_list = QListWidget()
    parent.sessions_list.itemClicked.connect(parent._load_session)
    parent.sessions_list.setStyleSheet("""
        QListWidget {
            border: none;
            background: transparent;
        }
        QListWidget::item {
            padding: 8px;
            border-radius: 4px;
            margin: 2px 0;
        }
        QListWidget::item:selected {
            background: rgba(137, 180, 250, 0.3);
        }
        QListWidget::item:hover {
            background: rgba(255, 255, 255, 0.05);
        }
    """)
    left_layout.addWidget(parent.sessions_list)
    
    # Session action buttons
    btn_row = QHBoxLayout()
    
    btn_new = QPushButton("New")
    btn_new.setToolTip("Start a new chat session")
    btn_new.clicked.connect(parent._new_session)
    btn_row.addWidget(btn_new)
    
    btn_load = QPushButton("Load")
    btn_load.setToolTip("Load session into chat")
    btn_load.clicked.connect(parent._load_session_into_chat)
    btn_row.addWidget(btn_load)
    
    left_layout.addLayout(btn_row)
    
    btn_row2 = QHBoxLayout()
    
    btn_rename = QPushButton("Rename")
    btn_rename.setToolTip("Rename selected session")
    btn_rename.clicked.connect(parent._rename_session)
    btn_row2.addWidget(btn_rename)
    
    btn_delete = QPushButton("Delete")
    btn_delete.setToolTip("Delete selected session")
    btn_delete.clicked.connect(parent._delete_session)
    btn_delete.setStyleSheet("""
        QPushButton {
            background-color: rgba(243, 139, 168, 0.3);
        }
        QPushButton:hover {
            background-color: rgba(243, 139, 168, 0.5);
        }
    """)
    btn_row2.addWidget(btn_delete)
    
    left_layout.addLayout(btn_row2)
    
    splitter.addWidget(left_frame)
    
    # RIGHT: Conversation viewer
    right_frame = QFrame()
    right_frame.setStyleSheet("""
        QFrame {
            background: rgba(49, 50, 68, 0.3);
            border-radius: 8px;
        }
    """)
    right_layout = QVBoxLayout(right_frame)
    right_layout.setContentsMargins(10, 10, 10, 10)
    
    viewer_label = QLabel("Conversation")
    viewer_label.setStyleSheet("font-weight: bold;")
    right_layout.addWidget(viewer_label)
    
    parent.session_viewer = QTextEdit()
    parent.session_viewer.setReadOnly(True)
    parent.session_viewer.setTextInteractionFlags(
        Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
    )
    parent.session_viewer.setPlaceholderText(
        "Select a saved chat to view its contents...\n\n"
        "Double-click to load into the Chat tab."
    )
    parent.session_viewer.setStyleSheet("""
        QTextEdit {
            border: none;
            background: transparent;
            font-size: 12px;
        }
    """)
    right_layout.addWidget(parent.session_viewer)
    
    splitter.addWidget(right_frame)
    
    # Set initial sizes (list narrower than content)
    splitter.setSizes([250, 450])
    layout.addWidget(splitter, stretch=1)
    
    # Populate AI selector and refresh sessions list
    parent._populate_history_ai_selector()
    parent._refresh_sessions()
    
    w.setLayout(layout)
    return w
