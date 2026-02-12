"""
================================================================================
THE CONVERSATION HALL - CHAT TAB INTERFACE
================================================================================

Welcome, traveler, to the grand Conversation Hall! This is where humans and
AI meet to exchange words, ideas, and understanding. The hall is adorned with
comfortable cushions (buttons) and a great speaking scroll (chat display)
where all conversations are recorded for posterity.

FILE: enigma_engine/gui/tabs/chat_tab.py
TYPE: GUI Component - Chat Interface
MAIN FUNCTION: create_chat_tab()

    THE HALL'S FEATURES:
    
    +--------------------------------------------------+
    |  [Model: small_forge]      [+New] [Clear] [Save] |  <- The Header Bar
    +--------------------------------------------------+
    |                                                  |
    |  User: Hello, AI!                               |
    |  AI: Greetings, noble traveler!                 |  <- The Great Scroll
    |                                                  |
    +--------------------------------------------------+
    |  [Thinking...]                                   |  <- Status Panel
    +--------------------------------------------------+
    |  [ Type your message here...        ] [Send]     |  <- Input Chamber
    +--------------------------------------------------+

CONNECTED REALMS:
    PARENT:     enigma_engine/gui/enhanced_window.py - The main castle
    INVOKES:    enigma_engine/core/inference.py - For AI responses
    STORES IN:  enigma_engine/memory/manager.py - Conversation archives

USAGE:
    from enigma_engine.gui.tabs.chat_tab import create_chat_tab
    
    chat_widget = create_chat_tab(parent_window)
    tabs.addTab(chat_widget, "Chat")
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QShortcut,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

# =============================================================================
# STYLE CONSTANTS
# =============================================================================

# UI text truncation length
MAX_DISPLAY_LENGTH = 200

# Note: History limiting is now handled by token-based truncation in
# enigma_engine/core/inference.py (_truncate_history method) which dynamically
# calculates how many messages fit based on actual token counts.

STYLE_MODEL_LABEL = """
    QLabel {
        color: #89b4fa;
        font-weight: bold;
        font-size: 12px;
        padding: 4px 8px;
        background: rgba(137, 180, 250, 0.1);
        border-radius: 4px;
    }
"""

STYLE_NEW_CHAT_BTN = """
    QPushButton {
        background-color: #a6e3a1;
        color: #1e1e2e;
        font-weight: bold;
        padding: 4px 8px;
    }
    QPushButton:hover {
        background-color: #94e2d5;
    }
"""

STYLE_SECONDARY_BTN = """
    QPushButton {
        background-color: #89b4fa;
        color: #1e1e2e;
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 4px;
        border: none;
    }
    QPushButton:hover {
        background-color: #b4befe;
    }
    QPushButton:disabled {
        background-color: #313244;
        color: #f38ba8;
        border: 2px dashed #f38ba8;
    }
"""

STYLE_THINKING_FRAME = """
    QFrame {
        background: rgba(249, 226, 175, 0.15);
        border: 1px solid #f9e2af;
        border-radius: 6px;
        padding: 4px;
    }
"""

STYLE_INPUT_FRAME = """
    QFrame {
        background: rgba(49, 50, 68, 0.7);
        border: 1px solid #45475a;
        border-radius: 8px;
        padding: 8px;
    }
"""

STYLE_CHAT_INPUT = """
    QLineEdit {
        background-color: rgba(50, 50, 50, 0.9);
        border: 1px solid #555;
        border-radius: 8px;
        padding: 10px 15px;
        font-size: 12px;
        color: white;
    }
    QLineEdit:focus {
        border-color: #3498db;
    }
"""

STYLE_SEND_BTN = """
    QPushButton {
        background-color: #3498db;
        border: none;
        border-radius: 8px;
        color: white;
        font-size: 11px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #2980b9;
    }
    QPushButton:pressed {
        background-color: #1c5980;
    }
"""

STYLE_STOP_BTN = """
    QPushButton {
        background-color: #e74c3c;
        border: none;
        border-radius: 8px;
        color: white;
        font-size: 11px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #c0392b;
    }
    QPushButton:pressed {
        background-color: #922b21;
    }
"""

STYLE_REC_BTN = """
    QPushButton {
        background-color: #444;
        border: 2px solid #555;
        border-radius: 8px;
        color: #bac2de;
        font-size: 12px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #555;
        border-color: #e74c3c;
        color: #e74c3c;
    }
    QPushButton:checked {
        background-color: #e74c3c;
        border-color: #c0392b;
        color: white;
    }
    QPushButton:checked:hover {
        background-color: #c0392b;
    }
"""

STYLE_TTS_BTN = """
    QPushButton {
        background-color: #cba6f7;
        color: #1e1e2e;
        font-weight: bold;
        font-size: 11px;
        border-radius: 8px;
    }
    QPushButton:hover {
        background-color: #f5c2e7;
    }
"""

# Button dimensions - consistent sizing for all input buttons
BUTTON_WIDTH_SMALL = 80
BUTTON_WIDTH_MEDIUM = 110
BUTTON_HEIGHT = 36
TTS_BTN_SIZE = (80, 36)  # Match Send button size
REC_BTN_SIZE = (80, 36)  # Match Send button size
ATTACH_BTN_SIZE = (36, 36)

# Attachment styles
STYLE_ATTACH_BTN = """
    QPushButton {
        background-color: #45475a;
        border: 1px solid #585b70;
        border-radius: 6px;
        color: #cdd6f4;
        font-size: 14px;
    }
    QPushButton:hover {
        background-color: #585b70;
        border-color: #89b4fa;
    }
"""

STYLE_ATTACHMENT_PREVIEW = """
    QFrame {
        background: rgba(69, 71, 90, 0.5);
        border: 1px dashed #585b70;
        border-radius: 6px;
        padding: 4px;
    }
"""

STYLE_ATTACHMENT_ITEM = """
    QFrame {
        background: #313244;
        border: 1px solid #45475a;
        border-radius: 4px;
        padding: 2px;
    }
"""

# TTS state tracking to prevent multiple runs
_tts_is_speaking = False
_tts_stop_requested = False


# =============================================================================
# HELPER FUNCTIONS - Build Individual UI Sections
# =============================================================================

def _create_header_section(parent, layout):
    """Build the header bar with model indicator and action buttons."""
    header_layout = QHBoxLayout()
    
    # Model indicator
    initial_model_text = "No model loaded"
    if hasattr(parent, 'current_model_name') and parent.current_model_name:
        initial_model_text = f"[AI] {parent.current_model_name}"
    
    parent.chat_model_label = QLabel(initial_model_text)
    parent.chat_model_label.setStyleSheet(STYLE_MODEL_LABEL)
    header_layout.addWidget(parent.chat_model_label)
    
    # Persona dropdown selector
    parent.persona_combo = QComboBox()
    parent.persona_combo.setToolTip("Switch AI persona during conversation")
    parent.persona_combo.setMinimumWidth(120)
    parent.persona_combo.setMaximumWidth(180)
    parent.persona_combo.setStyleSheet("""
        QComboBox {
            color: #a6e3a1;
            font-weight: bold;
            font-size: 11px;
            padding: 4px 8px;
            background: rgba(166, 227, 161, 0.15);
            border: 1px solid rgba(166, 227, 161, 0.3);
            border-radius: 4px;
        }
        QComboBox:hover {
            background: rgba(166, 227, 161, 0.25);
            border-color: rgba(166, 227, 161, 0.5);
        }
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid #a6e3a1;
            margin-right: 5px;
        }
        QComboBox QAbstractItemView {
            background: #313244;
            border: 1px solid #45475a;
            selection-background-color: rgba(166, 227, 161, 0.3);
            color: #cdd6f4;
        }
    """)
    
    # Populate persona dropdown
    _populate_persona_combo(parent)
    
    # Connect persona change
    parent.persona_combo.currentTextChanged.connect(lambda name: _on_persona_changed(parent, name))
    
    header_layout.addWidget(parent.persona_combo)
    
    header_layout.addStretch()
    
    # Prompt History button
    parent.btn_prompt_history = QPushButton("History")
    parent.btn_prompt_history.setToolTip("Browse and reuse previous prompts")
    parent.btn_prompt_history.setMaximumWidth(BUTTON_WIDTH_SMALL)
    parent.btn_prompt_history.clicked.connect(lambda: _show_prompt_history(parent))
    parent.btn_prompt_history.setStyleSheet(STYLE_SECONDARY_BTN)
    header_layout.addWidget(parent.btn_prompt_history)
    
    # New Chat button
    parent.btn_new_chat = QPushButton("+ New Chat")
    parent.btn_new_chat.setToolTip("Start a fresh conversation (saves current chat first)")
    parent.btn_new_chat.setMaximumWidth(BUTTON_WIDTH_MEDIUM)
    parent.btn_new_chat.clicked.connect(lambda: _new_chat(parent))
    parent.btn_new_chat.setStyleSheet(STYLE_NEW_CHAT_BTN)
    header_layout.addWidget(parent.btn_new_chat)
    
    # Clear button
    parent.btn_clear_chat = QPushButton("Clear")
    parent.btn_clear_chat.setToolTip("Clear chat history")
    parent.btn_clear_chat.setMaximumWidth(BUTTON_WIDTH_SMALL)
    parent.btn_clear_chat.clicked.connect(lambda: _clear_chat(parent))
    parent.btn_clear_chat.setStyleSheet(STYLE_SECONDARY_BTN)
    header_layout.addWidget(parent.btn_clear_chat)
    
    # Save button
    parent.btn_save_chat = QPushButton("Save")
    parent.btn_save_chat.setToolTip("Save conversation")
    parent.btn_save_chat.setMaximumWidth(BUTTON_WIDTH_SMALL)
    parent.btn_save_chat.clicked.connect(lambda: _save_chat(parent))
    parent.btn_save_chat.setStyleSheet(STYLE_SECONDARY_BTN)
    header_layout.addWidget(parent.btn_save_chat)
    
    # Summarize button - compress conversation for context/handoff
    parent.btn_summarize = QPushButton("Summary")
    parent.btn_summarize.setToolTip("Summarize conversation for context or handoff to another AI")
    parent.btn_summarize.setMaximumWidth(BUTTON_WIDTH_SMALL)
    parent.btn_summarize.clicked.connect(lambda: _summarize_chat(parent))
    parent.btn_summarize.setStyleSheet(STYLE_SECONDARY_BTN)
    header_layout.addWidget(parent.btn_summarize)
    
    layout.addLayout(header_layout)


def _create_chat_display(parent, layout):
    """Build the main chat display area with search."""
    # Search bar (hidden by default, toggle with Ctrl+F)
    parent.search_frame = QFrame()
    parent.search_frame.setStyleSheet("""
        QFrame {
            background: #313244;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 4px;
        }
    """)
    search_layout = QHBoxLayout(parent.search_frame)
    search_layout.setContentsMargins(8, 4, 8, 4)
    search_layout.setSpacing(8)
    
    search_label = QLabel("Find:")
    search_label.setStyleSheet("color: #cdd6f4; font-size: 11px;")
    search_layout.addWidget(search_label)
    
    parent.search_input = QLineEdit()
    parent.search_input.setPlaceholderText("Search in conversation...")
    parent.search_input.setStyleSheet("""
        QLineEdit {
            background: #1e1e2e;
            border: 1px solid #45475a;
            border-radius: 3px;
            padding: 4px 8px;
            color: #cdd6f4;
            font-size: 11px;
        }
    """)
    parent.search_input.returnPressed.connect(lambda: _search_next(parent))
    parent.search_input.textChanged.connect(lambda: _highlight_search(parent))
    search_layout.addWidget(parent.search_input, stretch=1)
    
    prev_btn = QPushButton("Prev")
    prev_btn.setFixedWidth(50)
    prev_btn.setStyleSheet("""
        QPushButton {
            background: #45475a;
            border: none;
            border-radius: 3px;
            color: #cdd6f4;
            padding: 4px 8px;
            font-size: 10px;
        }
        QPushButton:hover { background: #585b70; }
    """)
    prev_btn.clicked.connect(lambda: _search_prev(parent))
    search_layout.addWidget(prev_btn)
    
    next_btn = QPushButton("Next")
    next_btn.setFixedWidth(50)
    next_btn.setStyleSheet(prev_btn.styleSheet())
    next_btn.clicked.connect(lambda: _search_next(parent))
    search_layout.addWidget(next_btn)
    
    parent.search_count = QLabel("")
    parent.search_count.setStyleSheet("color: #6c7086; font-size: 10px;")
    search_layout.addWidget(parent.search_count)
    
    close_search_btn = QPushButton("X")
    close_search_btn.setFixedSize(20, 20)
    close_search_btn.setStyleSheet("""
        QPushButton {
            background: transparent;
            border: none;
            color: #6c7086;
            font-size: 12px;
        }
        QPushButton:hover { color: #f38ba8; }
    """)
    close_search_btn.clicked.connect(lambda: _toggle_search(parent, False))
    search_layout.addWidget(close_search_btn)
    
    parent.search_frame.hide()
    layout.addWidget(parent.search_frame)
    
    # Initialize search state
    parent._search_positions = []
    parent._search_index = 0
    
    # Chat display
    parent.chat_display = QTextBrowser()
    parent.chat_display.setReadOnly(True)
    parent.chat_display.setTextInteractionFlags(
        Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard | Qt.LinksAccessibleByMouse
    )
    parent.chat_display.setOpenExternalLinks(False)
    parent.chat_display.anchorClicked.connect(lambda url: _handle_chat_link(parent, url))
    
    # Show welcome message for first-time users
    no_model = not (hasattr(parent, 'current_model_name') and parent.current_model_name)
    if no_model:
        parent.chat_display.setHtml("""
<div style="text-align: center; padding: 40px;">
<h2 style="color: #89b4fa;">Welcome to Enigma AI Engine!</h2>
<p style="color: #cdd6f4; font-size: 14px;">
Create your first AI to start chatting.
</p>
<p style="color: #bac2de; font-size: 12px; margin-top: 20px;">
<b>Getting Started:</b><br>
1. Go to <b>Model Manager</b> (bottom-left)<br>
2. Click <b>+ Template</b> to create from a ready-made kit<br>
3. Or click <b>+ New</b> for a blank AI
</p>
<p style="color: #6c7086; font-size: 11px; margin-top: 30px;">
Tip: The "Friendly Chatbot" template includes training data<br>
so your AI can chat right away!
</p>
</div>
""")
    else:
        parent.chat_display.setPlaceholderText(
            "Start chatting with your AI...\n\n"
            "Tips:\n"
            "- Just ask naturally - 'Generate an image of a sunset'\n"
            "- The AI auto-detects what you want to create\n"
            "- Rate responses to help the AI learn\n"
            "- Click 'Regenerate' on AI responses to get alternate versions\n"
            "- Press Ctrl+F to search"
        )
    parent.chat_display.setStyleSheet("""
        QTextEdit {
            font-size: 12px;
            line-height: 1.5;
            padding: 10px;
        }
    """)
    layout.addWidget(parent.chat_display, stretch=1)


def _create_thinking_panel(parent, layout):
    """Build the thinking indicator panel."""
    parent.thinking_frame = QFrame()
    parent.thinking_frame.setStyleSheet(STYLE_THINKING_FRAME)
    
    thinking_layout = QHBoxLayout(parent.thinking_frame)
    thinking_layout.setContentsMargins(8, 4, 8, 4)
    
    parent.thinking_label = QLabel("Thinking...")
    parent.thinking_label.setStyleSheet("color: #f9e2af; font-size: 12px;")
    thinking_layout.addWidget(parent.thinking_label)
    thinking_layout.addStretch()
    
    parent.thinking_frame.hide()
    layout.addWidget(parent.thinking_frame)


def _create_input_section(parent, layout):
    """Build the message input area with all buttons and attachment support."""
    # Initialize attachment tracking
    parent._attachments = []  # List of file paths or image data
    
    # Attachment preview area (hidden by default)
    parent.attachment_frame = QFrame()
    parent.attachment_frame.setStyleSheet(STYLE_ATTACHMENT_PREVIEW)
    parent.attachment_frame.setMaximumHeight(60)
    attachment_layout = QHBoxLayout(parent.attachment_frame)
    attachment_layout.setContentsMargins(6, 4, 6, 4)
    attachment_layout.setSpacing(6)
    
    parent.attachment_label = QLabel("Attachments:")
    parent.attachment_label.setStyleSheet("color: #6c7086; font-size: 10px;")
    attachment_layout.addWidget(parent.attachment_label)
    
    # Container for attachment previews
    parent.attachment_container = QWidget()
    parent.attachment_container_layout = QHBoxLayout(parent.attachment_container)
    parent.attachment_container_layout.setContentsMargins(0, 0, 0, 0)
    parent.attachment_container_layout.setSpacing(4)
    attachment_layout.addWidget(parent.attachment_container, stretch=1)
    
    # Clear attachments button
    clear_attach_btn = QPushButton("X")
    clear_attach_btn.setFixedSize(20, 20)
    clear_attach_btn.setToolTip("Clear all attachments")
    clear_attach_btn.setStyleSheet("""
        QPushButton {
            background: transparent;
            border: none;
            color: #f38ba8;
            font-size: 12px;
            font-weight: bold;
        }
        QPushButton:hover { color: #eba0ac; }
    """)
    clear_attach_btn.clicked.connect(lambda: _clear_attachments(parent))
    attachment_layout.addWidget(clear_attach_btn)
    
    parent.attachment_frame.hide()
    layout.addWidget(parent.attachment_frame)
    
    # Main input frame
    input_frame = QFrame()
    input_frame.setStyleSheet(STYLE_INPUT_FRAME)
    input_frame.setAcceptDrops(True)
    
    # Override drag events for drop support
    input_frame.dragEnterEvent = lambda e: _drag_enter(parent, e)
    input_frame.dropEvent = lambda e: _drop_event(parent, e)
    
    input_layout = QHBoxLayout(input_frame)
    input_layout.setContentsMargins(8, 8, 8, 8)
    input_layout.setSpacing(8)
    
    # Attach button (before input) - uses link icon
    from pathlib import Path
    parent.attach_btn = QPushButton()
    icon_path = Path(__file__).parent.parent.parent.parent / "assets" / "icons" / "link.svg"
    if icon_path.exists():
        parent.attach_btn.setIcon(QIcon(str(icon_path)))
    else:
        parent.attach_btn.setText("ATT")  # Fallback text
    parent.attach_btn.setFixedSize(*ATTACH_BTN_SIZE)
    parent.attach_btn.setToolTip("Attach file, image, or video (Ctrl+Shift+V to paste)")
    parent.attach_btn.setStyleSheet(STYLE_ATTACH_BTN)
    parent.attach_btn.clicked.connect(lambda: _attach_file(parent))
    input_layout.addWidget(parent.attach_btn)
    
    # Text input
    parent.chat_input = QLineEdit()
    parent.chat_input.setPlaceholderText("Chat here... (drag files or Ctrl+V to paste images)")
    parent.chat_input.returnPressed.connect(parent._on_send)
    parent.chat_input.setToolTip("Type your message and press Enter or click Send")
    parent.chat_input.setStyleSheet(STYLE_CHAT_INPUT)
    parent.chat_input.textChanged.connect(lambda text: _update_token_count(parent, text))
    input_layout.addWidget(parent.chat_input, stretch=1)
    
    # Token counter label
    parent.token_count_label = QLabel("0 chars")
    parent.token_count_label.setStyleSheet("color: #6c7086; font-size: 10px; min-width: 65px;")
    parent.token_count_label.setToolTip("Approximate character/token count")
    input_layout.addWidget(parent.token_count_label)
    
    # Send button
    parent.send_btn = QPushButton("Send")
    parent.send_btn.setFixedSize(BUTTON_WIDTH_SMALL, BUTTON_HEIGHT)
    parent.send_btn.clicked.connect(parent._on_send)
    parent.send_btn.setToolTip("Send your message (Enter)")
    parent.send_btn.setStyleSheet(STYLE_SEND_BTN)
    input_layout.addWidget(parent.send_btn)
    
    # Stop button (hidden by default)
    parent.stop_btn = QPushButton("Stop")
    parent.stop_btn.setToolTip("Stop AI generation")
    parent.stop_btn.setFixedSize(BUTTON_WIDTH_SMALL, BUTTON_HEIGHT)
    parent.stop_btn.setStyleSheet(STYLE_STOP_BTN)
    parent.stop_btn.clicked.connect(lambda: _stop_generation(parent))
    parent.stop_btn.hide()
    input_layout.addWidget(parent.stop_btn)
    
    # Voice record button
    parent.rec_btn = QPushButton("REC")
    parent.rec_btn.setFixedSize(*REC_BTN_SIZE)
    parent.rec_btn.setToolTip("Record - Click to speak (Shift+Click to save as voice message)")
    parent.rec_btn.setCheckable(True)
    parent.rec_btn.setStyleSheet(STYLE_REC_BTN)
    parent.rec_btn.clicked.connect(lambda: _toggle_voice_input(parent))
    input_layout.addWidget(parent.rec_btn)
    
    # TTS button - toggles voice mode
    parent.btn_speak = QPushButton("Voice")
    parent.btn_speak.setToolTip("Toggle voice mode - AI will speak responses aloud")
    parent.btn_speak.setFixedSize(*TTS_BTN_SIZE)
    parent.btn_speak.setCheckable(True)
    parent.btn_speak.setStyleSheet(STYLE_TTS_BTN)
    parent.btn_speak.clicked.connect(lambda: _toggle_voice_mode(parent))
    input_layout.addWidget(parent.btn_speak)
    
    # TTS Stop button (hidden by default)
    parent.btn_stop_tts = QPushButton("Stop")
    parent.btn_stop_tts.setToolTip("Stop speech")
    parent.btn_stop_tts.setFixedSize(*TTS_BTN_SIZE)
    parent.btn_stop_tts.setStyleSheet(STYLE_STOP_BTN)
    parent.btn_stop_tts.clicked.connect(lambda: _stop_tts(parent))
    parent.btn_stop_tts.hide()
    input_layout.addWidget(parent.btn_stop_tts)
    
    layout.addWidget(input_frame)
    
    # Setup Ctrl+V paste shortcut for images


def _create_status_bar(parent, layout):
    """Build the bottom status bar with learning indicator and token counter."""
    bottom_layout = QHBoxLayout()
    bottom_layout.setSpacing(8)
    
    parent.chat_status = QLabel("")
    parent.chat_status.setStyleSheet("color: #bac2de; font-size: 11px;")
    bottom_layout.addWidget(parent.chat_status)
    bottom_layout.addStretch()
    
    # Token usage display - shows context window usage
    parent.token_label = QLabel("Tokens: 0 / 4096")
    parent.token_label.setStyleSheet("color: #89b4fa; font-size: 11px;")
    parent.token_label.setToolTip(
        "Context window usage - shows how much of the AI's memory is being used.\n\n"
        "When this fills up, older messages may be forgotten.\n"
        "Click to see detailed breakdown."
    )
    parent.token_label.setCursor(Qt.PointingHandCursor)
    parent.token_label.mousePressEvent = lambda e: _show_token_details(parent)
    bottom_layout.addWidget(parent.token_label)
    
    # Token progress bar
    parent.token_bar = QProgressBar()
    parent.token_bar.setMaximumWidth(100)
    parent.token_bar.setMaximumHeight(12)
    parent.token_bar.setTextVisible(False)
    parent.token_bar.setRange(0, 100)
    parent.token_bar.setValue(0)
    parent.token_bar.setStyleSheet("""
        QProgressBar {
            border: 1px solid #45475a;
            border-radius: 4px;
            background: #313244;
        }
        QProgressBar::chunk {
            border-radius: 3px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #a6e3a1, stop:0.5 #f9e2af, stop:0.75 #fab387, stop:1 #f38ba8);
        }
    """)
    parent.token_bar.setToolTip("Context usage: Green (low) -> Yellow (medium) -> Orange (high) -> Red (critical)")
    bottom_layout.addWidget(parent.token_bar)
    
    # Initialize context tracker for this chat
    _init_context_tracker(parent)
    
    # Learning indicator
    parent.learning_indicator = QLabel("Learning: ON")
    parent.learning_indicator.setStyleSheet("color: #a6e3a1; font-size: 11px;")
    parent.learning_indicator.setToolTip(
        "When Learning is ON, the AI records your conversations and uses them to improve.\n\n"
        "How it works:\n"
        "- Each Q&A pair is saved to the model's training data\n"
        "- After enough interactions, the model can be retrained\n"
        "- This helps the AI learn your preferences and style\n\n"
        "Note: Learning only works with local Forge models.\n"
        "HuggingFace models (GPT-2, Mistral, etc.) don't use this feature.\n\n"
        "Toggle in Settings menu or click here to toggle."
    )
    parent.learning_indicator.setCursor(Qt.PointingHandCursor)
    parent.learning_indicator.mousePressEvent = lambda e: _toggle_learning(parent)
    bottom_layout.addWidget(parent.learning_indicator)
    
    # Note: Voice toggle moved to input area (btn_speak) to reduce redundant indicators
    
    layout.addLayout(bottom_layout)


def _init_context_tracker(parent):
    """Initialize the context tracker for the chat."""
    try:
        from ...utils.context_window import get_context_tracker, reset_context_tracker
        
        # Get model context size if available
        max_tokens = 4096  # Default
        if hasattr(parent, 'current_model_name') and parent.current_model_name:
            from ...utils.context_window import get_context_size
            max_tokens = get_context_size(parent.current_model_name)
        
        # Reset and get fresh tracker
        reset_context_tracker()
        parent._context_tracker = get_context_tracker(max_tokens=max_tokens)
        
        # Add callback for auto-continue warnings
        parent._context_tracker.add_usage_callback(lambda usage: _on_context_usage_update(parent, usage))
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to init context tracker: {e}")
        parent._context_tracker = None


def _on_context_usage_update(parent, usage):
    """Called when context usage changes."""
    try:
        _update_token_display(parent)
        
        # Check for warnings
        if usage.percentage >= 90:
            parent.chat_status.setText("Warning: Context nearly full - AI may forget earlier messages")
            parent.chat_status.setStyleSheet("color: #f38ba8; font-size: 11px; font-weight: bold;")
            
            # Show popup warning only once per session
            if not getattr(parent, '_critical_warning_shown', False):
                parent._critical_warning_shown = True
                _show_hallucination_warning(parent, usage)
                
        elif usage.percentage >= 75:
            parent.chat_status.setText("Context usage high")
            parent.chat_status.setStyleSheet("color: #fab387; font-size: 11px;")
            # Reset critical warning flag when usage drops
            parent._critical_warning_shown = False
        else:
            parent.chat_status.setStyleSheet("color: #bac2de; font-size: 11px;")
            # Reset critical warning flag when usage drops
            parent._critical_warning_shown = False
        
        # Check for auto-continue trigger
        if hasattr(parent, '_context_tracker') and parent._context_tracker:
            if parent._context_tracker.should_auto_continue():
                if not getattr(parent, '_auto_continue_triggered', False):
                    parent._auto_continue_triggered = True
                    _trigger_auto_continue(parent)
            
    except Exception:
        pass  # Intentionally silent


def _show_hallucination_warning(parent, usage):
    """Show a warning popup when context is nearly full."""
    from PyQt5.QtWidgets import QMessageBox
    
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("Context Memory Warning")
    msg.setText("The AI is running low on context memory.")
    msg.setInformativeText(
        f"Context usage: {usage.percentage:.0f}%\n\n"
        "Earlier parts of this conversation may be forgotten, "
        "leading to inconsistent or repeated responses.\n\n"
        "What would you like to do?"
    )
    
    continue_btn = msg.addButton("Continue Anyway", QMessageBox.AcceptRole)
    new_chat_btn = msg.addButton("Start Fresh Chat", QMessageBox.ActionRole)
    smart_btn = msg.addButton("Smart Continue", QMessageBox.ActionRole)
    smart_btn.setToolTip("Save conversation summary and continue in new chat")
    
    msg.exec_()
    
    clicked = msg.clickedButton()
    if clicked == new_chat_btn:
        _clear_chat(parent)
    elif clicked == smart_btn:
        _smart_continue(parent)


def _trigger_auto_continue(parent):
    """Automatically continue conversation when context is full."""
    from ...config import CONFIG
    
    ctx_config = CONFIG.get("context_window", {})
    if not ctx_config.get("auto_continue_enabled", True):
        return
    
    # Show notification
    if hasattr(parent, 'chat_status'):
        parent.chat_status.setText("Auto-continuing in new chat...")
        parent.chat_status.setStyleSheet("color: #89b4fa; font-size: 11px; font-weight: bold;")
    
    _smart_continue(parent, show_notification=True)


def _smart_continue(parent, show_notification=False):
    """Continue conversation with summary in a new chat."""
    from PyQt5.QtWidgets import QMessageBox
    from ...config import CONFIG
    
    if not hasattr(parent, '_context_tracker') or parent._context_tracker is None:
        return
    
    try:
        ctx_config = CONFIG.get("context_window", {})
        
        # Get summary and messages to keep
        summary, messages = parent._context_tracker.prepare_auto_continue()
        
        # Save old chat to history if enabled
        if ctx_config.get("auto_save_on_continue", True):
            try:
                from ...memory.manager import ConversationManager
                history_manager = ConversationManager()
                history_manager.save_conversation(
                    parent._conversation_history if hasattr(parent, '_conversation_history') else [],
                    metadata={"auto_continued": True, "summary": summary[:200]}
                )
            except Exception:
                pass  # History save is optional
        
        # Clear the chat
        _clear_chat(parent, reset_tracker=True)
        
        # Add summary as system context if available
        if summary and ctx_config.get("auto_continue_include_summary", True):
            # Add summary to chat display
            parent.chat_display.append(
                f'<div style="color: #89b4fa; font-style: italic; padding: 8px; '
                f'border-left: 3px solid #89b4fa; margin: 8px 0;">'
                f'<strong>Continued from previous conversation:</strong><br/>'
                f'{summary.replace(chr(10), "<br/>")}</div>'
            )
            
            # Add to conversation history
            if hasattr(parent, '_conversation_history'):
                parent._conversation_history.append({
                    "role": "system",
                    "content": f"[Conversation Summary]\n{summary}"
                })
        
        # Add kept messages back
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                parent.chat_display.append(
                    f'<div style="color: #89dceb;"><strong>You:</strong> {content}</div>'
                )
            elif role == "assistant":
                parent.chat_display.append(
                    f'<div style="color: #a6e3a1;"><strong>AI:</strong> {content}</div>'
                )
            
            # Add to conversation history
            if hasattr(parent, '_conversation_history'):
                parent._conversation_history.append(msg)
            
            # Re-track in context tracker
            if hasattr(parent, '_context_tracker') and parent._context_tracker:
                parent._context_tracker.add_message(role, content)
        
        # Reset auto-continue flag
        parent._auto_continue_triggered = False
        parent._critical_warning_shown = False
        
        # Show notification
        if show_notification:
            parent.chat_status.setText("Conversation continued in new context")
            parent.chat_status.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Smart continue failed: {e}")
        QMessageBox.warning(parent, "Error", f"Could not continue conversation: {e}")


def _update_token_display(parent):
    """Update the token counter display."""
    if not hasattr(parent, '_context_tracker') or parent._context_tracker is None:
        return
    
    try:
        usage = parent._context_tracker.get_usage()
        
        # Update label
        parent.token_label.setText(f"Tokens: {usage.used_tokens:,} / {usage.max_tokens:,}")
        
        # Update progress bar
        parent.token_bar.setValue(int(min(100, usage.percentage)))
        
        # Color-code based on usage level
        from ...utils.context_window import UsageLevel
        color_map = {
            UsageLevel.LOW: "#a6e3a1",      # Green
            UsageLevel.MEDIUM: "#f9e2af",   # Yellow
            UsageLevel.HIGH: "#fab387",     # Orange
            UsageLevel.CRITICAL: "#f38ba8"  # Red
        }
        color = color_map.get(usage.level, "#89b4fa")
        parent.token_label.setStyleSheet(f"color: {color}; font-size: 11px;")
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Token display update failed: {e}")


def _show_token_details(parent):
    """Show detailed token breakdown popup."""
    from PyQt5.QtWidgets import QMessageBox
    
    if not hasattr(parent, '_context_tracker') or parent._context_tracker is None:
        QMessageBox.information(parent, "Token Usage", "Context tracker not available.")
        return
    
    try:
        summary = parent._context_tracker.get_summary()
        tokens = summary['tokens']
        
        details = f"""Context Window Usage

Total: {tokens['used']:,} / {tokens['max']:,} tokens ({summary['percentage']:.1f}%)
{summary['progress_bar']}

Breakdown:
  System Prompt:    {tokens['system']:,} tokens
  User Messages:    {tokens['user']:,} tokens
  AI Responses:     {tokens['assistant']:,} tokens
  Remaining:        {tokens['remaining']:,} tokens

Messages: {summary['message_count']}
Status: {summary['level']}

Tip: Start a new chat if context is running out."""
        
        QMessageBox.information(parent, "Token Usage Details", details)
        
    except Exception as e:
        QMessageBox.warning(parent, "Error", f"Could not get token details: {e}")


# =============================================================================
# ATTACHMENT HANDLING FUNCTIONS
# =============================================================================

def _attach_file(parent):
    """Open file dialog to attach files."""
    
    file_filter = (
        "All Files (*);;"
        "Images (*.png *.jpg *.jpeg *.gif *.bmp *.webp);;"
        "Videos (*.mp4 *.webm *.mov *.avi *.mkv);;"
        "Audio (*.mp3 *.wav *.ogg *.m4a *.flac);;"
        "Documents (*.txt *.pdf *.md *.doc *.docx)"
    )
    files, _ = QFileDialog.getOpenFileNames(
        parent, "Attach Files", "", file_filter
    )
    
    for file_path in files:
        _add_attachment(parent, file_path)


def _add_attachment(parent, source):
    """
    Add an attachment to the list.
    
    Args:
        parent: The parent widget
        source: File path (str) or QImage for clipboard pastes
    """
    import tempfile
    from pathlib import Path

    # If source is a QImage (from clipboard), save it temporarily
    if isinstance(source, QImage):
        temp_dir = Path(tempfile.gettempdir()) / "enigma_engine_attachments"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"paste_{len(parent._attachments)}.png"
        source.save(str(temp_path))
        source = str(temp_path)
    
    # Check if already attached
    if source in parent._attachments:
        return
    
    parent._attachments.append(source)
    _update_attachment_preview(parent)


def _update_attachment_preview(parent):
    """Update the attachment preview area."""
    from pathlib import Path

    # Clear existing previews
    while parent.attachment_container_layout.count() > 0:
        item = parent.attachment_container_layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
    
    if not parent._attachments:
        parent.attachment_frame.hide()
        return
    
    parent.attachment_frame.show()
    
    for i, path in enumerate(parent._attachments):
        item_frame = QFrame()
        item_frame.setStyleSheet(STYLE_ATTACHMENT_ITEM)
        item_layout = QHBoxLayout(item_frame)
        item_layout.setContentsMargins(4, 2, 4, 2)
        item_layout.setSpacing(4)
        
        # Check if it's an image
        path_obj = Path(path)
        is_image = path_obj.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
        is_video = path_obj.suffix.lower() in ['.mp4', '.webm', '.mov', '.avi', '.mkv']
        is_audio = path_obj.suffix.lower() in ['.mp3', '.wav', '.ogg', '.m4a', '.flac']
        
        if is_image:
            # Show thumbnail
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                thumb_label = QLabel()
                thumb_label.setPixmap(pixmap)
                item_layout.addWidget(thumb_label)
        elif is_video:
            # Show video icon
            video_label = QLabel("VID")
            video_label.setFixedSize(40, 40)
            video_label.setAlignment(Qt.AlignCenter)
            video_label.setStyleSheet("""
                background: #94e2d5; color: #1e1e2e; font-weight: bold;
                font-size: 10px; border-radius: 4px;
            """)
            item_layout.addWidget(video_label)
        elif is_audio:
            # Show audio icon
            audio_label = QLabel("AUD")
            audio_label.setFixedSize(40, 40)
            audio_label.setAlignment(Qt.AlignCenter)
            audio_label.setStyleSheet("""
                background: #f5c2e7; color: #1e1e2e; font-weight: bold;
                font-size: 10px; border-radius: 4px;
            """)
            item_layout.addWidget(audio_label)
        else:
            # Show file icon
            file_label = QLabel("FILE")
            file_label.setFixedSize(40, 40)
            file_label.setAlignment(Qt.AlignCenter)
            file_label.setStyleSheet("""
                background: #6c7086; color: #1e1e2e; font-weight: bold;
                font-size: 9px; border-radius: 4px;
            """)
            item_layout.addWidget(file_label)
        
        # Show filename
        name_label = QLabel(path_obj.name[:15] + "..." if len(path_obj.name) > 15 else path_obj.name)
        name_label.setStyleSheet("color: #cdd6f4; font-size: 9px;")
        name_label.setToolTip(str(path))
        item_layout.addWidget(name_label)
        
        # Remove button
        remove_btn = QPushButton("x")
        remove_btn.setFixedSize(16, 16)
        remove_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #6c7086;
                font-size: 10px;
            }
            QPushButton:hover { color: #f38ba8; }
        """)
        remove_btn.clicked.connect(lambda checked, idx=i: _remove_attachment(parent, idx))
        item_layout.addWidget(remove_btn)
        
        parent.attachment_container_layout.addWidget(item_frame)
    
    parent.attachment_container_layout.addStretch()


def _remove_attachment(parent, index: int):
    """Remove an attachment by index."""
    if 0 <= index < len(parent._attachments):
        parent._attachments.pop(index)
        _update_attachment_preview(parent)


def _clear_attachments(parent):
    """Clear all attachments."""
    parent._attachments.clear()
    _update_attachment_preview(parent)


def cleanup_temp_attachments():
    """
    Clean up temporary attachment files on application exit.
    
    This removes the enigma_engine_attachments directory from temp folder
    to prevent accumulation of pasted images.
    """
    import tempfile
    import shutil
    from pathlib import Path
    
    temp_dir = Path(tempfile.gettempdir()) / "enigma_engine_attachments"
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            # Silently ignore cleanup failures (files may be in use)
            pass


def _drag_enter(parent, event):
    """Handle drag enter event for file drops."""
    if event.mimeData().hasUrls() or event.mimeData().hasImage():
        event.acceptProposedAction()
    else:
        event.ignore()


def _drop_event(parent, event):
    """Handle drop event for files and images."""
    mime = event.mimeData()
    
    if mime.hasUrls():
        for url in mime.urls():
            if url.isLocalFile():
                _add_attachment(parent, url.toLocalFile())
    elif mime.hasImage():
        image = QImage(mime.imageData())
        if not image.isNull():
            _add_attachment(parent, image)
    
    event.acceptProposedAction()


def _check_clipboard_for_image(parent):
    """Check clipboard for image and add if found."""
    clipboard = QApplication.clipboard()
    mime = clipboard.mimeData()
    
    if mime.hasImage():
        image = clipboard.image()
        if not image.isNull():
            _add_attachment(parent, image)
            parent.chat_status.setText("Image pasted from clipboard")
            return True
    return False


def get_attachments(parent):
    """Get list of current attachments (for use when sending message)."""
    return list(getattr(parent, '_attachments', []))


def format_attachments_html(attachments: list, max_width: int = 300) -> str:
    """
    Format attachments as HTML for display in chat.
    
    Args:
        attachments: List of file paths
        max_width: Maximum image width in pixels
        
    Returns:
        HTML string with embedded images/videos or file links
    """
    import base64
    from pathlib import Path
    
    if not attachments:
        return ""
    
    html_parts = []
    
    for path in attachments:
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()
        
        # Image types
        if suffix in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            try:
                # Read and encode image as base64 for inline display
                with open(path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                
                mime_type = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.bmp': 'image/bmp',
                    '.webp': 'image/webp'
                }.get(suffix, 'image/png')
                
                html_parts.append(f'''
                <div style="margin: 8px 0;">
                    <img src="data:{mime_type};base64,{img_data}" 
                         style="max-width: {max_width}px; max-height: 300px; border-radius: 8px; cursor: pointer;"
                         title="Click to view full size: {path_obj.name}"
                         onclick="window.open('file:///{path.replace(chr(92), '/')}', '_blank')"/>
                    <div style="color: #6c7086; font-size: 10px; margin-top: 2px;">{path_obj.name}</div>
                </div>
                ''')
            except Exception as e:
                html_parts.append(f'<div style="color: #f38ba8; font-size: 10px;">Failed to load image: {path_obj.name}</div>')
        
        # Video types
        elif suffix in ['.mp4', '.webm', '.mov', '.avi', '.mkv']:
            # For videos, show a clickable link/thumbnail (base64 embedding videos is too heavy)
            html_parts.append(f'''
            <div style="margin: 8px 0; padding: 10px; background: #45475a; border-radius: 8px; display: inline-block;">
                <div style="color: #94e2d5; font-weight: bold;">Video Attachment</div>
                <div style="color: #cdd6f4; font-size: 12px;">{path_obj.name}</div>
                <div style="color: #6c7086; font-size: 10px; margin-top: 4px;">
                    <a href="file:///{path.replace(chr(92), '/')}" style="color: #89b4fa;">Click to open</a>
                </div>
            </div>
            ''')
        
        # Audio types
        elif suffix in ['.mp3', '.wav', '.ogg', '.m4a', '.flac']:
            html_parts.append(f'''
            <div style="margin: 8px 0; padding: 10px; background: #45475a; border-radius: 8px; display: inline-block;">
                <div style="color: #f5c2e7; font-weight: bold;">Audio Attachment</div>
                <div style="color: #cdd6f4; font-size: 12px;">{path_obj.name}</div>
                <div style="color: #6c7086; font-size: 10px; margin-top: 4px;">
                    <a href="file:///{path.replace(chr(92), '/')}" style="color: #89b4fa;">Click to play</a>
                </div>
            </div>
            ''')
        
        # Other files
        else:
            html_parts.append(f'''
            <div style="margin: 8px 0; padding: 8px; background: #45475a; border-radius: 6px; display: inline-block;">
                <span style="color: #6c7086;">File:</span>
                <a href="file:///{path.replace(chr(92), '/')}" style="color: #89b4fa;">{path_obj.name}</a>
            </div>
            ''')
    
    return '\n'.join(html_parts)


def process_attachments_for_ai(attachments: list) -> dict:
    """
    Process attachments for AI consumption (vision analysis, etc).
    
    Args:
        attachments: List of file paths
        
    Returns:
        Dict with 'images', 'videos', 'audio', 'files' lists
    """
    from pathlib import Path
    
    result = {
        'images': [],
        'videos': [],
        'audio': [],
        'files': []
    }
    
    for path in attachments:
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()
        
        if suffix in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            result['images'].append(str(path))
        elif suffix in ['.mp4', '.webm', '.mov', '.avi', '.mkv']:
            result['videos'].append(str(path))
        elif suffix in ['.mp3', '.wav', '.ogg', '.m4a', '.flac']:
            result['audio'].append(str(path))
        else:
            result['files'].append(str(path))
    
    return result


# =============================================================================
# THE HALL CONSTRUCTION - Main Tab Creation
# =============================================================================

def create_chat_tab(parent):
    """
    Construct the grand Conversation Hall.
    
    This function assembles the chat interface piece by piece, creating
    a welcoming space where human and AI may converse freely.
    
    Args:
        parent: The main window that will house this tab.
    
    Returns:
        QWidget: The fully constructed chat tab.
    """
    chat_widget = QWidget()
    main_layout = QVBoxLayout()
    main_layout.setSpacing(6)
    main_layout.setContentsMargins(6, 6, 6, 6)
    
    # Build each section using helper functions
    _create_header_section(parent, main_layout)
    
    # Add quick actions bar if GUI mode manager is available
    if hasattr(parent, 'gui_mode_manager'):
        from ..widgets.quick_actions import QuickActionsBar
        quick_actions = QuickActionsBar(parent)
        # Connect signals to parent methods if they exist
        if hasattr(parent, '_on_screenshot_clicked'):
            quick_actions.screenshot_clicked.connect(parent._on_screenshot_clicked)
        if hasattr(parent, '_on_voice_clicked'):
            quick_actions.voice_clicked.connect(parent._on_voice_clicked)
        if hasattr(parent, '_on_game_mode_clicked'):
            quick_actions.game_mode_clicked.connect(parent._on_game_mode_clicked)
        if hasattr(parent, 'btn_new_chat'):
            quick_actions.new_chat_clicked.connect(lambda: parent.btn_new_chat.click())
        if hasattr(parent, '_on_quick_generate_clicked'):
            quick_actions.quick_generate_clicked.connect(parent._on_quick_generate_clicked)
        main_layout.addWidget(quick_actions)
    
    _create_chat_display(parent, main_layout)
    
    _create_thinking_panel(parent, main_layout)
    _create_input_section(parent, main_layout)
    _create_status_bar(parent, main_layout)
    
    # Initialize voice thread tracking
    if not hasattr(parent, '_voice_thread'):
        parent._voice_thread = None
    
    # Update voice button state from saved settings
    _update_voice_button_state(parent)
    
    # Setup keyboard shortcuts
    search_shortcut = QShortcut(QKeySequence("Ctrl+F"), chat_widget)
    search_shortcut.activated.connect(lambda: _toggle_search(parent, True))
    
    # Escape to close search
    escape_shortcut = QShortcut(QKeySequence("Escape"), chat_widget)
    escape_shortcut.activated.connect(lambda: _toggle_search(parent, False) if parent.search_frame.isVisible() else None)
    
    chat_widget.setLayout(main_layout)
    return chat_widget


def _on_feedback(parent, is_positive):
    """Handle feedback button clicks."""
    feedback_type = "positive" if is_positive else "negative"
    # Log feedback for future analytics integration
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"User feedback received: {feedback_type} for last AI response")


def _toggle_voice_mode(parent):
    """Toggle voice mode from the TTS button."""
    parent.auto_speak = not getattr(parent, 'auto_speak', False)
    _update_voice_button_state(parent)
    _update_tts_button_state(parent)
    
    # Also sync with the auto_speak_action menu item if it exists
    if hasattr(parent, 'auto_speak_action'):
        parent.auto_speak_action.blockSignals(True)
        parent.auto_speak_action.setChecked(parent.auto_speak)
        parent.auto_speak_action.setText(f"AI Auto-Speak ({'ON' if parent.auto_speak else 'OFF'})")
        parent.auto_speak_action.blockSignals(False)
    
    # Show status
    if parent.auto_speak:
        parent.chat_status.setText("Voice mode ON - AI will speak responses")
    else:
        parent.chat_status.setText("Voice mode OFF")


def _toggle_voice_output(parent):
    """Toggle voice output on/off (from status bar button)."""
    _toggle_voice_mode(parent)


def _update_tts_button_state(parent):
    """Update the TTS/Voice button appearance based on voice mode state."""
    if not hasattr(parent, 'btn_speak'):
        return
    
    is_on = getattr(parent, 'auto_speak', False)
    parent.btn_speak.setChecked(is_on)
    
    if is_on:
        parent.btn_speak.setText("Voice")
        parent.btn_speak.setToolTip("Voice mode ON - AI speaks responses\nClick to turn off")
        parent.btn_speak.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #94d990;
            }
        """)
    else:
        parent.btn_speak.setText("Voice")
        parent.btn_speak.setToolTip("Voice mode OFF\nClick to turn on")
        parent.btn_speak.setStyleSheet(STYLE_TTS_BTN)


def _update_voice_button_state(parent):
    """Update the voice button appearance based on state."""
    is_on = getattr(parent, 'auto_speak', False)
    
    # Update the TTS button state (the only voice indicator now)
    _update_tts_button_state(parent)


def _update_token_count(parent, text: str):
    """Update the token counter label as user types."""
    char_count = len(text)
    # Rough token estimate: ~4 chars per token for English
    token_estimate = char_count // 4
    
    if char_count == 0:
        parent.token_count_label.setText("0 chars")
        parent.token_count_label.setStyleSheet("color: #6c7086; font-size: 10px; min-width: 65px;")
    elif char_count < 500:
        parent.token_count_label.setText(f"{char_count} chars")
        parent.token_count_label.setStyleSheet("color: #6c7086; font-size: 10px; min-width: 65px;")
    elif char_count < 2000:
        parent.token_count_label.setText(f"~{token_estimate} tokens")
        parent.token_count_label.setStyleSheet("color: #f9e2af; font-size: 10px; min-width: 65px;")
    else:
        parent.token_count_label.setText(f"~{token_estimate} tokens")
        parent.token_count_label.setStyleSheet("color: #f38ba8; font-size: 10px; min-width: 65px;")


def _toggle_voice_input(parent):
    """Toggle voice input (microphone recording)."""
    is_listening = parent.rec_btn.isChecked()
    
    if is_listening:
        parent.rec_btn.setToolTip("Listening... (click to stop)")
        parent.chat_status.setText("Listening...")
        
        # Try to start voice recognition
        try:
            if hasattr(parent, '_voice_thread') and parent._voice_thread:
                return
            
            import threading

            # Check if voice message mode is enabled (hold Shift while clicking)
            from PyQt5.QtWidgets import QApplication
            modifiers = QApplication.keyboardModifiers()
            save_recording = bool(modifiers & Qt.ShiftModifier)
            
            parent._voice_thread = threading.Thread(
                target=lambda: _do_voice_input(parent, save_recording=save_recording), 
                daemon=True
            )
            parent._voice_thread.start()
        except Exception as e:
            parent.rec_btn.setChecked(False)
            parent.chat_status.setText(f"Voice error: {e}")
    else:
        parent.rec_btn.setToolTip("Record - Click to speak (Shift+Click to save recording)")
        parent.chat_status.setText("Ready")
        parent._voice_thread = None


def _do_voice_input(parent, save_recording: bool = False):
    """
    Background voice recognition - automatically sends message after capture.
    
    Args:
        parent: Parent widget
        save_recording: If True, save audio file and attach instead of transcribing
    """
    try:
        import os
        import sys
        from datetime import datetime
        from pathlib import Path

        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        
        # Suppress PyAudio stderr spam when opening microphone
        old_stderr = sys.stderr
        devnull = open(os.devnull, 'w')
        try:
            sys.stderr = devnull
            mic = sr.Microphone()
        except Exception:
            raise
        finally:
            sys.stderr = old_stderr
            devnull.close()
        
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)
        
        if save_recording:
            # Save as audio file and attach
            voice_dir = Path.home() / ".enigma_engine" / "voice_messages"
            voice_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = voice_dir / f"voice_{timestamp}.wav"
            
            # Save raw audio data as WAV
            with open(audio_path, "wb") as f:
                f.write(audio.get_wav_data())
            
            # Attach the voice message
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, lambda: _voice_message_saved(parent, str(audio_path)))
        else:
            # Regular transcription
            text = recognizer.recognize_google(audio)
            
            # Auto-send the voice input (more alive, don't put in chat box)
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, lambda: _voice_input_and_send(parent, text))
        
    except Exception as e:
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: _voice_input_error(parent, str(e)))


def _voice_message_saved(parent, audio_path: str):
    """Handle saved voice message."""
    parent.rec_btn.setChecked(False)
    parent.rec_btn.setToolTip("Record - Click to speak (Shift+Click to save recording)")
    parent._voice_thread = None
    
    # Add as attachment
    if hasattr(parent, '_attachments'):
        parent._attachments.append(audio_path)
        _update_attachment_preview(parent)
        parent.chat_status.setText(f"Voice message saved and attached")
    else:
        parent.chat_status.setText(f"Voice saved: {audio_path}")


def _voice_input_and_send(parent, text: str):
    """Process voice input and automatically send to AI."""
    parent.rec_btn.setChecked(False)
    parent.rec_btn.setToolTip("Record - Click to speak")
    parent._voice_thread = None
    
    if not text or not text.strip():
        parent.chat_status.setText("No speech detected")
        return
    
    parent.chat_status.setText(f"Heard: {text[:50]}..." if len(text) > 50 else f"Heard: {text}")
    
    # Don't put in chat box - send directly for a more alive feel
    # This triggers the AI to respond immediately
    if hasattr(parent, '_on_send'):
        # Temporarily set the input text and send
        parent.chat_input.setText(text)
        parent._on_send()  # This will read from chat_input and process


# =============================================================================
# DEPRECATED FUNCTIONS (kept for reference, no longer used)
# =============================================================================

def _voice_input_done(parent):
    """
    DEPRECATED: No longer used with auto-send voice input.
    Was called when voice input completed successfully.
    Voice input now uses _do_voice_input() which auto-sends.
    """
    parent.rec_btn.setChecked(False)
    parent.rec_btn.setToolTip("Record - Click to speak")
    parent.chat_status.setText("Voice captured")
    parent._voice_thread = None
    parent.chat_input.setFocus()


def _on_speak_last_safe(parent):
    """
    DEPRECATED: No longer used. TTS is now controlled via auto-speak mode.
    See _toggle_voice_mode() for current implementation.
    Was: Speak last AI response with double-click protection.
    """
    global _tts_is_speaking, _tts_stop_requested
    
    # Prevent double-clicks while TTS is running
    if _tts_is_speaking:
        parent.chat_status.setText("Already speaking - click Stop to cancel")
        return
    
    if not hasattr(parent, 'last_response') or not parent.last_response:
        parent.chat_status.setText("No response to speak")
        return
    
    _tts_is_speaking = True
    _tts_stop_requested = False
    
    # Update UI
    parent.btn_speak.setEnabled(False)
    parent.btn_speak.setText("...")
    parent.btn_stop_tts.show()
    parent.chat_status.setText("Speaking...")
    
    # Run TTS in background thread
    import threading
    thread = threading.Thread(target=lambda: _do_tts(parent, parent.last_response), daemon=True)
    thread.start()


def _do_tts(parent, text: str):
    """Perform TTS in background thread with better voice quality."""
    global _tts_is_speaking, _tts_stop_requested
    
    try:
        import re

        # Clean text for TTS
        clean_text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
        clean_text = re.sub(r'<tool_call>.*?</tool_call>', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'```[\s\S]*?```', '', clean_text)  # Remove code blocks
        clean_text = clean_text.strip()[:500]  # Limit length
        
        if not clean_text or clean_text.startswith("[Warning]"):
            return
        
        if _tts_stop_requested:
            return
        
        # Try to use voice profile system for better quality
        try:
            from ..voice.voice_profile import get_engine
            engine = get_engine()
            
            # Check if avatar has a custom voice
            if hasattr(parent, 'avatar') and parent.avatar:
                avatar_voice = getattr(parent.avatar, 'voice_profile', None)
                if avatar_voice:
                    engine.set_profile(avatar_voice)
            
            engine.speak(clean_text)
        except Exception:
            # Fallback to simple speak
            try:
                from ..voice import speak
                speak(clean_text)
            except Exception:
                pass  # Intentionally silent
    finally:
        # Reset state from main thread
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: _tts_finished(parent))


def _tts_finished(parent):
    """Called when TTS finishes."""
    global _tts_is_speaking
    _tts_is_speaking = False
    
    parent.btn_speak.setEnabled(True)
    parent.btn_speak.setText("TTS")
    parent.btn_stop_tts.hide()
    parent.chat_status.setText("Ready")


# =============================================================================
# MESSAGE BRANCHING SYSTEM
# =============================================================================

def _init_branching(parent):
    """Initialize the message branching system."""
    if not hasattr(parent, '_message_branches') or not isinstance(parent._message_branches, dict):
        # Maps message index -> list of alternate responses
        parent._message_branches = {}
    if not hasattr(parent, '_current_branch') or not isinstance(parent._current_branch, dict):
        # Maps message index -> current branch index (0 = original)
        parent._current_branch = {}


def _add_branch(parent, message_idx: int, response: str):
    """
    Add an alternate response branch for a message.
    
    Args:
        parent: Parent widget
        message_idx: Index of the AI message to branch
        response: The alternate response text
    """
    _init_branching(parent)
    
    if message_idx not in parent._message_branches:
        # First branch - save original response
        if hasattr(parent, 'chat_messages') and message_idx < len(parent.chat_messages):
            original = parent.chat_messages[message_idx].get('content', '')
            parent._message_branches[message_idx] = [original]
            parent._current_branch[message_idx] = 0
    
    parent._message_branches[message_idx].append(response)
    # Auto-switch to new branch
    parent._current_branch[message_idx] = len(parent._message_branches[message_idx]) - 1
    
    return parent._current_branch[message_idx]


def _get_branch_count(parent, message_idx: int) -> int:
    """Get the number of branches for a message."""
    _init_branching(parent)
    return len(parent._message_branches.get(message_idx, []))


def _switch_branch(parent, message_idx: int, direction: int):
    """
    Switch to a different branch for a message.
    
    Args:
        parent: Parent widget
        message_idx: Index of the message
        direction: 1 for next, -1 for previous
    """
    _init_branching(parent)
    
    if message_idx not in parent._message_branches:
        return
    
    branches = parent._message_branches[message_idx]
    current = parent._current_branch.get(message_idx, 0)
    new_idx = (current + direction) % len(branches)
    parent._current_branch[message_idx] = new_idx
    
    # Update the message in chat_messages
    if hasattr(parent, 'chat_messages') and message_idx < len(parent.chat_messages):
        parent.chat_messages[message_idx]['content'] = branches[new_idx]
        _refresh_chat_display(parent)


def _regenerate_response(parent, message_idx: int = None):
    """
    Regenerate the AI response for a message, creating a new branch.
    
    Args:
        parent: Parent widget
        message_idx: Index of AI message to regenerate (None = last AI message)
    """
    _init_branching(parent)
    
    if not hasattr(parent, 'chat_messages') or not parent.chat_messages:
        parent.chat_status.setText("No messages to regenerate")
        return
    
    # Find the last AI message if no index provided
    if message_idx is None:
        for i in range(len(parent.chat_messages) - 1, -1, -1):
            if parent.chat_messages[i].get('role') == 'assistant':
                message_idx = i
                break
    
    if message_idx is None:
        parent.chat_status.setText("No AI response to regenerate")
        return
    
    # Find the user message before this AI response
    user_msg_idx = None
    for i in range(message_idx - 1, -1, -1):
        if parent.chat_messages[i].get('role') == 'user':
            user_msg_idx = i
            break
    
    if user_msg_idx is None:
        parent.chat_status.setText("Could not find original prompt")
        return
    
    user_prompt = parent.chat_messages[user_msg_idx].get('content', '')
    
    # Generate new response
    parent.chat_status.setText("Regenerating response...")
    
    import threading
    def generate_branch():
        try:
            # Use the brain/inference engine to generate
            if hasattr(parent, 'brain') and parent.brain:
                response = parent.brain.chat(user_prompt)
            elif hasattr(parent, '_inference_engine') and parent._inference_engine:
                response = parent._inference_engine.generate(user_prompt)
            else:
                response = "[Could not generate - no inference engine available]"
            
            # Add as branch
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, lambda: _on_regenerate_complete(parent, message_idx, response))
        except Exception as e:
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, lambda: parent.chat_status.setText(f"Regenerate failed: {e}"))
    
    thread = threading.Thread(target=generate_branch, daemon=True)
    thread.start()


def _on_regenerate_complete(parent, message_idx: int, response: str):
    """Handle completed regeneration."""
    branch_idx = _add_branch(parent, message_idx, response)
    branch_count = _get_branch_count(parent, message_idx)
    
    parent.chat_status.setText(f"Generated branch {branch_idx + 1}/{branch_count}")
    _refresh_chat_display(parent)


def _refresh_chat_display(parent):
    """Refresh the chat display with current branch selections."""
    if not hasattr(parent, 'chat_display') or not hasattr(parent, 'chat_messages'):
        return
    
    _init_branching(parent)
    
    # Rebuild the display with branch navigation
    html_parts = []
    
    for i, msg in enumerate(parent.chat_messages):
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'assistant':
            # Check for branches
            branch_count = _get_branch_count(parent, i)
            current_branch = parent._current_branch.get(i, 0)
            
            # Add branch navigation if multiple branches exist
            if branch_count > 1:
                nav_html = f'''
                <div style="color: #6c7086; font-size: 10px; margin-bottom: 4px;">
                    <a href="branch_prev_{i}" style="color: #89b4fa; text-decoration: none;">&lt;</a>
                    Branch {current_branch + 1}/{branch_count}
                    <a href="branch_next_{i}" style="color: #89b4fa; text-decoration: none;">&gt;</a>
                    <a href="regenerate_{i}" style="color: #a6e3a1; text-decoration: none; margin-left: 8px;">Regenerate</a>
                </div>
                '''
            else:
                nav_html = f'''
                <div style="color: #6c7086; font-size: 10px; margin-bottom: 4px;">
                    <a href="regenerate_{i}" style="color: #a6e3a1; text-decoration: none;">Regenerate</a>
                </div>
                '''
            
            html_parts.append(f'''
            <div style="background: #313244; padding: 10px; margin: 5px 0; border-radius: 8px;">
                <b style="color: #89b4fa;">AI:</b>
                {nav_html}
                <div style="color: #cdd6f4;">{content}</div>
            </div>
            ''')
        else:
            html_parts.append(f'''
            <div style="background: #45475a; padding: 10px; margin: 5px 0; border-radius: 8px;">
                <b style="color: #f9e2af;">You:</b>
                <div style="color: #cdd6f4;">{content}</div>
            </div>
            ''')
    
    parent.chat_display.setHtml(''.join(html_parts))
    
    # Scroll to bottom
    scrollbar = parent.chat_display.verticalScrollBar()
    scrollbar.setValue(scrollbar.maximum())


def _handle_branch_link(parent, url):
    """Handle clicks on branch navigation links."""
    url_str = url.toString() if hasattr(url, 'toString') else str(url)
    
    if url_str.startswith('branch_prev_'):
        try:
            idx = int(url_str.replace('branch_prev_', ''))
            _switch_branch(parent, idx, -1)
        except ValueError:
            pass  # Intentionally silent
    elif url_str.startswith('branch_next_'):
        try:
            idx = int(url_str.replace('branch_next_', ''))
            _switch_branch(parent, idx, 1)
        except ValueError:
            pass  # Intentionally silent
    elif url_str.startswith('regenerate_'):
        try:
            idx = int(url_str.replace('regenerate_', ''))
            _regenerate_response(parent, idx)
        except ValueError:
            pass  # Intentionally silent


def _stop_tts(parent):
    """Stop TTS playback."""
    global _tts_stop_requested, _tts_is_speaking
    _tts_stop_requested = True
    
    # Try to stop the TTS engine
    try:
        from ..voice.voice_profile import get_engine
        engine = get_engine()
        if hasattr(engine, '_engine') and engine._engine:
            engine._engine.stop()
    except Exception:
        pass  # Intentionally silent
    
    _tts_is_speaking = False
    parent.btn_speak.setEnabled(True)
    parent.btn_speak.setText("TTS")
    parent.btn_stop_tts.hide()
    parent.chat_status.setText("Speech stopped")


def _voice_input_error(parent, error: str):
    """Called when voice input fails."""
    parent.rec_btn.setChecked(False)
    parent.rec_btn.setToolTip("Record - Click to speak")
    parent.chat_status.setText(f"Voice error: {error[:40]}")
    parent._voice_thread = None


def _toggle_learning(parent):
    """Toggle the learning mode on/off."""
    current = getattr(parent, 'learn_while_chatting', True)
    parent.learn_while_chatting = not current
    
    if parent.learn_while_chatting:
        parent.learning_indicator.setText("Learning: ON")
        parent.learning_indicator.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        parent.chat_status.setText("Learning enabled - AI will learn from conversations")
    else:
        parent.learning_indicator.setText("Learning: OFF")
        parent.learning_indicator.setStyleSheet("color: #bac2de; font-size: 11px;")
        parent.chat_status.setText("Learning disabled - conversations won't be saved for training")
    
    # Update brain if available
    if hasattr(parent, 'brain') and parent.brain:
        parent.brain.auto_learn = parent.learn_while_chatting


def _clear_chat(parent, reset_tracker=False):
    """Clear the chat display and history.
    
    Args:
        parent: The parent widget
        reset_tracker: If True, also reset the context tracker
    """
    parent.chat_display.clear()
    parent.chat_messages = []
    
    # Clear conversation history if it exists
    if hasattr(parent, '_conversation_history'):
        parent._conversation_history = []
    
    # Reset context tracker if requested
    if reset_tracker and hasattr(parent, '_context_tracker') and parent._context_tracker:
        parent._context_tracker.clear()
    
    parent.chat_status.setText("Chat cleared")


def _show_prompt_history(parent):
    """Show dialog with prompt history for reuse."""
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QListWidget, QListWidgetItem, 
        QPushButton, QHBoxLayout, QLineEdit, QLabel
    )
    
    dialog = QDialog(parent)
    dialog.setWindowTitle("Prompt History")
    dialog.setMinimumSize(500, 400)
    dialog.setStyleSheet("""
        QDialog {
            background: #1e1e2e;
            color: #cdd6f4;
        }
        QListWidget {
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 6px;
            padding: 4px;
        }
        QListWidget::item {
            padding: 8px;
            border-bottom: 1px solid #333;
        }
        QListWidget::item:selected {
            background: #89b4fa;
            color: #1e1e2e;
        }
        QLineEdit {
            padding: 8px;
            border: 1px solid #444;
            border-radius: 6px;
            background: #1a1a1a;
        }
    """)
    
    layout = QVBoxLayout(dialog)
    
    # Search box
    search_layout = QHBoxLayout()
    search_layout.addWidget(QLabel("Search:"))
    search_input = QLineEdit()
    search_input.setPlaceholderText("Filter prompts...")
    search_layout.addWidget(search_input)
    layout.addLayout(search_layout)
    
    # Prompt list
    prompt_list = QListWidget()
    
    # Load prompt history
    prompts = _get_prompt_history()
    for prompt in prompts:
        item = QListWidgetItem(prompt[:100] + "..." if len(prompt) > 100 else prompt)
        item.setData(Qt.ItemDataRole.UserRole, prompt)
        item.setToolTip(prompt[:300])
        prompt_list.addItem(item)
    
    layout.addWidget(prompt_list)
    
    # Filter function
    def filter_prompts(text):
        for i in range(prompt_list.count()):
            item = prompt_list.item(i)
            full_prompt = item.data(Qt.ItemDataRole.UserRole)
            item.setHidden(text.lower() not in full_prompt.lower())
    
    search_input.textChanged.connect(filter_prompts)
    
    # Buttons
    btn_layout = QHBoxLayout()
    
    use_btn = QPushButton("Use Prompt")
    use_btn.setStyleSheet("background: #89b4fa; color: #1e1e2e; font-weight: bold;")
    use_btn.clicked.connect(lambda: _use_selected_prompt(dialog, prompt_list, parent))
    btn_layout.addWidget(use_btn)
    
    delete_btn = QPushButton("Delete")
    delete_btn.clicked.connect(lambda: _delete_selected_prompt(prompt_list))
    btn_layout.addWidget(delete_btn)
    
    clear_btn = QPushButton("Clear All")
    clear_btn.setStyleSheet("background: #f38ba8;")
    clear_btn.clicked.connect(lambda: _clear_prompt_history(prompt_list))
    btn_layout.addWidget(clear_btn)
    
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(dialog.accept)
    btn_layout.addWidget(close_btn)
    
    layout.addLayout(btn_layout)
    
    # Double-click to use
    prompt_list.itemDoubleClicked.connect(
        lambda item: _use_prompt(dialog, item.data(Qt.ItemDataRole.UserRole), parent)
    )
    
    dialog.exec_()


def _get_prompt_history() -> list:
    """Load prompt history from storage."""
    import json
    from pathlib import Path
    
    history_path = Path.home() / ".enigma_engine" / "prompt_history.json"
    if history_path.exists():
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('prompts', [])
        except Exception:
            pass  # Intentionally silent
    return []


def _save_prompt_history(prompts: list):
    """Save prompt history to storage."""
    import json
    from pathlib import Path
    
    history_path = Path.home() / ".enigma_engine" / "prompt_history.json"
    history_path.parent.mkdir(exist_ok=True)
    
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({'prompts': prompts[-100:]}, f, indent=2)  # Keep last 100


def _add_to_prompt_history(prompt: str):
    """Add a prompt to history (called when sending messages)."""
    if not prompt.strip():
        return
    
    prompts = _get_prompt_history()
    # Remove duplicates
    if prompt in prompts:
        prompts.remove(prompt)
    prompts.insert(0, prompt)  # Add to front
    _save_prompt_history(prompts)


def _use_selected_prompt(dialog, prompt_list, parent):
    """Use the selected prompt from history."""
    current = prompt_list.currentItem()
    if current:
        prompt = current.data(Qt.ItemDataRole.UserRole)
        _use_prompt(dialog, prompt, parent)


def _use_prompt(dialog, prompt: str, parent):
    """Set the prompt in the chat input."""
    if hasattr(parent, 'chat_input'):
        parent.chat_input.setText(prompt)
    dialog.accept()


def _delete_selected_prompt(prompt_list):
    """Delete the selected prompt from history."""
    current = prompt_list.currentItem()
    if current:
        prompt = current.data(Qt.ItemDataRole.UserRole)
        prompts = _get_prompt_history()
        if prompt in prompts:
            prompts.remove(prompt)
            _save_prompt_history(prompts)
        prompt_list.takeItem(prompt_list.row(current))


def _clear_prompt_history(prompt_list):
    """Clear all prompt history."""
    from PyQt5.QtWidgets import QMessageBox
    
    reply = QMessageBox.question(
        prompt_list.parent(), "Clear History",
        "Clear all prompt history?",
        QMessageBox.Yes | QMessageBox.No
    )
    if reply == QMessageBox.Yes:
        _save_prompt_history([])
        prompt_list.clear()


def _populate_persona_combo(parent):
    """Populate the persona dropdown with available personas."""
    try:
        from ...utils.personas import PersonaManager
        
        manager = PersonaManager()
        all_personas = manager.list_personas()
        
        parent.persona_combo.blockSignals(True)
        parent.persona_combo.clear()
        
        # Add all available personas
        for name, persona in sorted(all_personas.items()):
            parent.persona_combo.addItem(persona.name, name)
        
        # Try to select current persona
        try:
            from ...core.persona import get_persona_manager
            pm = get_persona_manager()
            current = pm.get_current_persona()
            if current:
                idx = parent.persona_combo.findText(current.name)
                if idx >= 0:
                    parent.persona_combo.setCurrentIndex(idx)
        except Exception:
            # Default to first persona
            if parent.persona_combo.count() > 0:
                parent.persona_combo.setCurrentIndex(0)
        
        parent.persona_combo.blockSignals(False)
        
    except Exception as e:
        print(f"[Chat] Failed to populate personas: {e}")
        parent.persona_combo.addItem("Default", "default")


def _on_persona_changed(parent, persona_name: str):
    """Handle persona selection change."""
    if not persona_name:
        return
        
    try:
        from ...utils.personas import PersonaManager
        
        manager = PersonaManager()
        # Find persona key by display name
        all_personas = manager.list_personas()
        persona_key = None
        for key, persona in all_personas.items():
            if persona.name == persona_name:
                persona_key = key
                break
        
        if not persona_key:
            return
        
        persona = manager.get_persona(persona_key)
        if not persona:
            return
        
        # Update the core persona manager (uses string IDs)
        try:
            from ...core.persona import get_persona_manager
            pm = get_persona_manager()
            # Core persona manager uses persona_id strings
            # Check if this persona exists in core, if not create it
            if not pm.persona_exists(persona_key):
                # Create the persona in core system with utils persona data
                pm.create_persona(
                    persona_id=persona_key,
                    name=persona.name,
                    description=persona.description,
                    system_prompt=persona.system_prompt,
                    traits=persona.traits
                )
            pm.set_current_persona(persona_key)
        except Exception as e:
            print(f"[Chat] Core persona manager error: {e}")
            pass  # Core persona manager may not be available
        
        # Update status
        if hasattr(parent, 'chat_status'):
            parent.chat_status.setText(f"Switched to persona: {persona.name}")
        
        # Log the change
        if hasattr(parent, 'log_terminal'):
            parent.log_terminal(f"Persona changed to: {persona.name}", "info")
            
    except Exception as e:
        print(f"[Chat] Failed to change persona: {e}")


def _new_chat(parent):
    """Start a new chat - save current chat first, then clear both main chat and Quick Chat."""
    # Save current chat if there's content
    if hasattr(parent, 'chat_messages') and parent.chat_messages:
        if hasattr(parent, '_save_current_chat'):
            parent._save_current_chat()
            parent.chat_status.setText("Previous chat saved. Starting new conversation...")
    
    # Clear the main chat
    parent.chat_display.clear()
    parent.chat_messages = []
    
    # Also clear Quick Chat via ChatSync
    try:
        from ..chat_sync import ChatSync
        chat_sync = ChatSync.instance()
        chat_sync.clear_chat()  # This clears both main and quick chat displays
    except Exception:
        pass  # Intentionally silent
    
    # ─────────────────────────────────────────────────────────────────────────
    # CLEAR KV-CACHE: Prevents hallucinations from stale context!
    # This is critical - without this, the model may reference old conversations
    # ─────────────────────────────────────────────────────────────────────────
    if hasattr(parent, 'engine') and parent.engine:
        # Clear KV-cache for fresh context
        if hasattr(parent.engine, 'clear_kv_cache'):
            try:
                parent.engine.clear_kv_cache()
                if hasattr(parent, 'log_terminal'):
                    parent.log_terminal("Cleared KV-cache for new conversation", "debug")
            except Exception:
                pass  # Intentionally silent
        
        # Reset HuggingFace conversation history
        if hasattr(parent.engine, 'model') and hasattr(parent.engine.model, 'reset_conversation'):
            try:
                parent.engine.model.reset_conversation()
            except (AttributeError, RuntimeError):
                pass  # Model doesn't support conversation reset
    
    # Clear tool output memory for fresh session
    try:
        from ...tools.history import get_output_memory
        get_output_memory().clear()
    except (ImportError, AttributeError):
        pass  # Tool history module not available
    
    # Show welcome message in main chat
    model_name = parent.current_model_name if hasattr(parent, 'current_model_name') else "AI"
    parent.chat_display.append(
        f'<div style="color: #a6e3a1; padding: 8px;">'
        f'<b>New conversation started with {model_name}</b><br>'
        f'<span style="color: #6c7086;">Previous chat has been saved. Type a message to begin.</span>'
        f'</div><hr>'
    )
    parent.chat_status.setText("New chat started")


def _save_chat(parent):
    """Save the current chat session."""
    if hasattr(parent, '_save_current_chat'):
        parent._save_current_chat()
        parent.chat_status.setText("Chat saved!")
    else:
        parent.chat_status.setText("Save not available")


def _summarize_chat(parent):
    """
    Summarize the current conversation.
    
    This creates a compact summary that can be:
    - Used to continue the conversation later
    - Handed off to another AI for context
    - Copied for sharing or documentation
    """
    from PyQt5.QtWidgets import (
        QApplication,
        QDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
    )
    
    if not hasattr(parent, 'chat_messages') or not parent.chat_messages:
        parent.chat_status.setText("No conversation to summarize")
        return
    
    if len(parent.chat_messages) < 2:
        parent.chat_status.setText("Need more messages to summarize")
        return
    
    parent.chat_status.setText("Generating summary...")
    
    try:
        from ...memory.conversation_summary import (
            export_for_handoff,
            summarize_conversation,
        )

        # Generate summary
        summary = summarize_conversation(parent.chat_messages, use_ai=False)
        
        # Create dialog to show summary
        dialog = QDialog(parent)
        dialog.setWindowTitle("Conversation Summary")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Summary info
        info_label = QLabel(f"Messages: {summary.message_count} | Topics: {', '.join(summary.topics[:3]) or 'General chat'}")
        info_label.setStyleSheet("color: #89b4fa; font-weight: bold;")
        layout.addWidget(info_label)
        
        # Summary text
        summary_text = QTextEdit()
        summary_text.setReadOnly(True)
        summary_text.setPlainText(summary.to_context_string() or summary.summary_text)
        layout.addWidget(summary_text)
        
        # Handoff context (for other AIs)
        layout.addWidget(QLabel("Context for handoff to another AI:"))
        handoff_text = QTextEdit()
        handoff_text.setReadOnly(True)
        handoff_text.setPlainText(export_for_handoff(parent.chat_messages))
        handoff_text.setMaximumHeight(150)
        layout.addWidget(handoff_text)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        copy_btn = QPushButton("Copy Summary")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(summary_text.toPlainText()))
        btn_layout.addWidget(copy_btn)
        
        copy_handoff_btn = QPushButton("Copy Handoff")
        copy_handoff_btn.clicked.connect(lambda: QApplication.clipboard().setText(handoff_text.toPlainText()))
        btn_layout.addWidget(copy_handoff_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        dialog.exec_()
        parent.chat_status.setText("Summary generated")
        
    except Exception as e:
        parent.chat_status.setText(f"Summary failed: {str(e)[:30]}")


def _stop_generation(parent):
    """Stop the current AI generation - handles both AIGenerationWorker and ChatSync."""
    stopped = False
    
    # First, try to stop the AIGenerationWorker (used by main chat tab's _on_send)
    if hasattr(parent, '_ai_worker') and parent._ai_worker:
        if parent._ai_worker.isRunning():
            parent._ai_worker.stop()
            stopped = True
    
    # Also stop ChatSync (used by quick chat and shared generation)
    try:
        from ..chat_sync import ChatSync
        chat_sync = ChatSync.instance()
        if chat_sync.is_generating:
            chat_sync.stop_generation()
            stopped = True
    except Exception:
        pass  # Intentionally silent
    
    # Update UI
    if stopped:
        parent.chat_status.setText("Stopping generation...")
        if hasattr(parent, 'stop_btn'):
            parent.stop_btn.setEnabled(False)
            parent.stop_btn.setText("...")
        if hasattr(parent, 'thinking_frame'):
            parent.thinking_frame.hide()
        if hasattr(parent, 'send_btn'):
            parent.send_btn.setEnabled(True)
            parent.send_btn.setText("Send")
    else:
        parent.chat_status.setText("Nothing to stop")


def _handle_chat_link(parent, url):
    """
    Universal handler for all links clicked in chat display.
    Routes to appropriate handler based on link type.
    """
    url_str = url.toString() if hasattr(url, 'toString') else str(url)
    
    # Branch navigation links
    if url_str.startswith('branch_') or url_str.startswith('regenerate_'):
        _handle_branch_link(parent, url)
        return
    
    # Feedback and code copy links
    _handle_feedback_link(parent, url)


def _handle_feedback_link(parent, url):
    """Handle feedback links clicked in chat."""
    from PyQt5.QtWidgets import QInputDialog
    
    url_str = url.toString() if hasattr(url, 'toString') else str(url)
    
    # Handle copy:code_hash links for code blocks
    if url_str.startswith('copy:'):
        code_hash = url_str[5:]
        _copy_code_block(parent, code_hash)
        return
    
    if not url_str.startswith('feedback:'):
        return
    
    parts = url_str.split(':')
    if len(parts) < 3:
        return
    
    feedback_type = parts[1]
    response_id = parts[2]
    
    # Get the response data
    response_data = None
    if hasattr(parent, '_response_history'):
        response_data = parent._response_history.get(int(response_id))
    
    if feedback_type == 'good':
        parent.chat_status.setText("Thanks for the positive feedback!")
        # Integrate with learning engine for real learning
        if response_data:
            _record_positive_feedback(parent, response_data)
            parent.chat_status.setText("Feedback saved - AI will learn from this good example!")
    
    elif feedback_type == 'bad':
        parent.chat_status.setText("Sorry about that. What went wrong?")
        # Ask for quick reason
        reason, ok = QInputDialog.getItem(
            parent,
            "What was wrong?",
            "Please select what was wrong with the response:",
            ["Incorrect/Wrong info", "Off-topic", "Too long/verbose", "Too short", "Confusing", "Other"],
            0, False
        )
        if ok and reason:
            parent.chat_status.setText(f"Feedback noted: {reason}")
            if response_data:
                _record_negative_feedback(parent, response_data, reason)
    
    elif feedback_type == 'critique':
        # Open detailed critique dialog
        _show_critique_dialog(parent, response_id, response_data)
    
    elif feedback_type == 'regenerate':
        # Regenerate the response with the same input
        _regenerate_response(parent, response_id, response_data)


def _regenerate_response(parent, response_id: str, response_data: dict):
    """Regenerate a response using the original user input."""
    if not response_data:
        parent.chat_status.setText("Cannot regenerate - original input not found")
        return
    
    original_input = response_data.get('user_input', '')
    if not original_input:
        parent.chat_status.setText("Cannot regenerate - no original input")
        return
    
    # Add a note to chat
    parent.chat_display.append(
        '<div style="color: #cba6f7; padding: 4px; font-size: 12px;"><i>Regenerating response...</i></div>'
    )
    
    # Set the input and trigger send
    parent.chat_input.setText(original_input)
    if hasattr(parent, '_on_send'):
        parent._on_send()
    
    parent.chat_status.setText("Regenerating response...")


def _copy_code_block(parent, code_hash: str):
    """
    Copy a code block to clipboard by its hash.
    
    Args:
        parent: Parent window
        code_hash: MD5 hash of the code content
    """
    from PyQt5.QtGui import QGuiApplication

    # Try to find the code in the chat display HTML
    html = parent.chat_display.toHtml() if hasattr(parent, 'chat_display') else ""
    
    # Look for code blocks by their data-code attribute
    import re
    pattern = rf'<code[^>]*data-code="{code_hash}"[^>]*>([^<]*)</code>'
    match = re.search(pattern, html, re.DOTALL)
    
    if match:
        import html as html_module
        code = html_module.unescape(match.group(1))
        
        # Copy to clipboard
        clipboard = QGuiApplication.clipboard()
        if clipboard:
            clipboard.setText(code)
            parent.chat_status.setText("Code copied to clipboard!")
        else:
            parent.chat_status.setText("Could not access clipboard")
    else:
        # Fallback: search in stored code blocks if we have them
        if hasattr(parent, '_code_blocks') and code_hash in parent._code_blocks:
            code = parent._code_blocks[code_hash]
            clipboard = QGuiApplication.clipboard()
            if clipboard:
                clipboard.setText(code)
                parent.chat_status.setText("Code copied to clipboard!")
        else:
            parent.chat_status.setText("Code block not found")


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def _toggle_search(parent, show=None):
    """Toggle the search bar visibility."""
    if show is None:
        show = not parent.search_frame.isVisible()
    
    parent.search_frame.setVisible(show)
    if show:
        parent.search_input.setFocus()
        parent.search_input.selectAll()
    else:
        # Clear highlighting
        _clear_search_highlight(parent)


def _highlight_search(parent):
    """Highlight all occurrences of search text."""
    search_text = parent.search_input.text()
    
    if not search_text:
        _clear_search_highlight(parent)
        parent.search_count.setText("")
        return
    
    # Get plain text from display
    text = parent.chat_display.toPlainText()
    
    # Find all occurrences
    parent._search_positions = []
    start = 0
    search_lower = search_text.lower()
    text_lower = text.lower()
    
    while True:
        pos = text_lower.find(search_lower, start)
        if pos == -1:
            break
        parent._search_positions.append(pos)
        start = pos + 1
    
    # Update count
    count = len(parent._search_positions)
    if count > 0:
        parent._search_index = 0
        parent.search_count.setText(f"1 of {count}")
        _go_to_search_position(parent, 0)
    else:
        parent.search_count.setText("No results")
        parent._search_index = 0


def _search_next(parent):
    """Go to next search result."""
    if not parent._search_positions:
        return
    
    parent._search_index = (parent._search_index + 1) % len(parent._search_positions)
    _go_to_search_position(parent, parent._search_index)
    parent.search_count.setText(f"{parent._search_index + 1} of {len(parent._search_positions)}")


def _search_prev(parent):
    """Go to previous search result."""
    if not parent._search_positions:
        return
    
    parent._search_index = (parent._search_index - 1) % len(parent._search_positions)
    _go_to_search_position(parent, parent._search_index)
    parent.search_count.setText(f"{parent._search_index + 1} of {len(parent._search_positions)}")


def _go_to_search_position(parent, index):
    """Navigate to a specific search result position."""
    if not parent._search_positions or index >= len(parent._search_positions):
        return
    
    from PyQt5.QtGui import QTextCursor
    
    pos = parent._search_positions[index]
    search_len = len(parent.search_input.text())
    
    # Move cursor to position and select
    cursor = parent.chat_display.textCursor()
    cursor.setPosition(pos)
    cursor.setPosition(pos + search_len, QTextCursor.KeepAnchor)
    parent.chat_display.setTextCursor(cursor)
    
    # Ensure visible
    parent.chat_display.ensureCursorVisible()


def _clear_search_highlight(parent):
    """Clear search highlighting."""
    parent._search_positions = []
    parent._search_index = 0


def _record_feedback_helper(parent, response_data, feedback_type, extra_metadata=None):
    """
    Shared helper for recording feedback to avoid code duplication.
    
    Args:
        parent: Parent window with model info
        response_data: Response data dictionary
        feedback_type: 'positive' or 'negative'
        extra_metadata: Optional additional metadata dict
    """
    try:
        from enigma_engine.core.self_improvement import get_learning_engine
        
        model_name = getattr(parent, 'current_model_name', None)
        if not model_name:
            return
        
        metadata = {'source': 'chat_ui', 'timestamp': response_data.get('timestamp')}
        if extra_metadata:
            metadata.update(extra_metadata)
        
        engine = get_learning_engine(model_name)
        engine.record_feedback(
            input_text=response_data['user_input'],
            output_text=response_data['ai_response'],
            feedback=feedback_type,
            metadata=metadata
        )
        
        # Also save to brain if available (legacy support)
        if hasattr(parent, 'brain') and parent.brain:
            if feedback_type == 'positive':
                parent.brain.record_interaction(
                    response_data['user_input'],
                    response_data['ai_response'],
                    quality=1.0  # High quality
                )
            elif feedback_type == 'negative':
                reason = extra_metadata.get('reason', 'negative') if extra_metadata else 'negative'
                parent.brain.add_memory(
                    f"BAD RESPONSE - {reason}: Q: {response_data['user_input'][:100]}",
                    importance=0.3,
                    category="negative_feedback"
                )
    except Exception as e:
        import logging
        logging.error(f"Error recording {feedback_type} feedback: {e}")


def _record_positive_feedback(parent, response_data):
    """Record positive feedback using the learning engine."""
    _record_feedback_helper(parent, response_data, 'positive')


def _record_negative_feedback(parent, response_data, reason):
    """Record negative feedback using the learning engine."""
    _record_feedback_helper(parent, response_data, 'negative', {'reason': reason})


def _show_critique_dialog(parent, response_id, response_data):
    """Show a dialog for detailed critique of a response."""
    from PyQt5.QtWidgets import (
        QDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
    )

    from .shared_components import NoScrollComboBox
    
    dialog = QDialog(parent)
    dialog.setWindowTitle("Critique Response")
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(parent.styleSheet())
    
    layout = QVBoxLayout(dialog)
    
    # Show the original exchange
    if response_data:
        layout.addWidget(QLabel(f"<b>Your message:</b> {response_data['user_input'][:200]}..."))
        layout.addWidget(QLabel(f"<b>AI response:</b> {response_data['ai_response'][:200]}..."))
    
    layout.addWidget(QLabel("<b>What should the AI have said instead?</b>"))
    
    correction_input = QTextEdit()
    correction_input.setPlaceholderText(
        "Write the ideal response here...\n\n"
        "This will be saved as training data to improve the AI."
    )
    correction_input.setMaximumHeight(150)
    layout.addWidget(correction_input)
    
    # Issue type
    issue_layout = QHBoxLayout()
    issue_layout.addWidget(QLabel("Issue type:"))
    issue_combo = NoScrollComboBox()
    issue_combo.setToolTip("Select the type of issue with the AI response")
    issue_combo.addItems([
        "Factually incorrect",
        "Misunderstood question", 
        "Tone/style wrong",
        "Too verbose",
        "Not helpful",
        "Should have used tool",
        "Other"
    ])
    issue_layout.addWidget(issue_combo)
    issue_layout.addStretch()
    layout.addLayout(issue_layout)
    
    # Buttons
    btn_layout = QHBoxLayout()
    
    save_btn = QPushButton("Save Correction")
    save_btn.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: bold;")
    
    cancel_btn = QPushButton("Cancel")
    
    btn_layout.addStretch()
    btn_layout.addWidget(cancel_btn)
    btn_layout.addWidget(save_btn)
    layout.addLayout(btn_layout)
    
    def save_critique():
        correction = correction_input.toPlainText().strip()
        issue = issue_combo.currentText()
        
        if correction and response_data:
            # Save corrected example using learning engine
            try:
                from enigma_engine.core.self_improvement import (
                    LearningExample,
                    LearningSource,
                    Priority,
                    get_learning_engine,
                )
                
                model_name = getattr(parent, 'current_model_name', None)
                if model_name:
                    engine = get_learning_engine(model_name)
                    
                    # Evaluate the correction quality
                    quality_metrics = engine.evaluate_response_quality(
                        response_data['user_input'],
                        correction
                    )
                    
                    # Create a high-priority learning example from the correction
                    example = LearningExample(
                        input_text=response_data['user_input'],
                        output_text=correction,
                        source=LearningSource.CORRECTION,
                        priority=Priority.CRITICAL,  # User corrections are most important!
                        quality_score=quality_metrics['overall'],
                        relevance=quality_metrics['relevance'],
                        coherence=quality_metrics['coherence'],
                        repetition=quality_metrics['repetition'],
                        metadata={
                            'original_response': response_data['ai_response'][:200],
                            'issue_type': issue,
                            'source': 'user_correction'
                        }
                    )
                    engine.add_learning_example(example)
                    
                    parent.chat_status.setText(f"Correction saved! AI will prioritize learning this better response.")
            except Exception as e:
                import logging
                logging.error(f"Error saving correction: {e}")
                parent.chat_status.setText("Error saving correction, but continuing...")
            
            # Also save to brain if available (legacy support)
            if hasattr(parent, 'brain') and parent.brain:
                # Save the corrected version with high quality
                parent.brain.record_interaction(
                    response_data['user_input'],
                    correction,
                    quality=1.0
                )
            
            # Show in chat
            parent.chat_display.append(
                f'<div style="background-color: #313244; padding: 8px; margin: 4px 0; border-radius: 8px; border-left: 3px solid #89b4fa;">'
                f'<b style="color: #89b4fa;">Correction saved ({issue}):</b><br>'
                f'<i style="color: #a6e3a1;">Better response:</i> {correction[:MAX_DISPLAY_LENGTH]}...</div>'
            )
        
        dialog.accept()
    
    save_btn.clicked.connect(save_critique)
    cancel_btn.clicked.connect(dialog.reject)
    
    dialog.exec_()
