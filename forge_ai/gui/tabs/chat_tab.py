"""
================================================================================
THE CONVERSATION HALL - CHAT TAB INTERFACE
================================================================================

Welcome, traveler, to the grand Conversation Hall! This is where humans and
AI meet to exchange words, ideas, and understanding. The hall is adorned with
comfortable cushions (buttons) and a great speaking scroll (chat display)
where all conversations are recorded for posterity.

FILE: forge_ai/gui/tabs/chat_tab.py
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
    PARENT:     forge_ai/gui/enhanced_window.py - The main castle
    INVOKES:    forge_ai/core/inference.py - For AI responses
    STORES IN:  forge_ai/memory/manager.py - Conversation archives

USAGE:
    from forge_ai.gui.tabs.chat_tab import create_chat_tab
    
    chat_widget = create_chat_tab(parent_window)
    tabs.addTab(chat_widget, "Chat")
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QTextBrowser, QLineEdit, QLabel, QFrame, QSplitter,
    QGroupBox, QSizePolicy, QShortcut
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence


# =============================================================================
# STYLE CONSTANTS
# =============================================================================

# UI text truncation length
MAX_DISPLAY_LENGTH = 200

# Note: History limiting is now handled by token-based truncation in
# forge_ai/core/inference.py (_truncate_history method) which dynamically
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

STYLE_VOICE_TOGGLE = """
    QPushButton {
        background-color: #333;
        border: 1px solid #555;
        border-radius: 4px;
        color: #bac2de;
        font-size: 12px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #444;
        border-color: #2ecc71;
    }
    QPushButton:checked {
        background-color: #2ecc71;
        border-color: #27ae60;
        color: white;
    }
"""

# Button dimensions
BUTTON_WIDTH_SMALL = 80
BUTTON_WIDTH_MEDIUM = 110
BUTTON_HEIGHT = 36
VOICE_TOGGLE_SIZE = (50, 28)
TTS_BTN_SIZE = (55, 36)
REC_BTN_SIZE = (60, 36)

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
    
    # Persona indicator
    try:
        from ...core.persona import get_persona_manager
        manager = get_persona_manager()
        persona = manager.get_current_persona()
        persona_text = f"[Persona] {persona.name}"
    except Exception:
        persona_text = "[Persona] Default"
    
    parent.chat_persona_label = QLabel(persona_text)
    parent.chat_persona_label.setStyleSheet("""
        QLabel {
            color: #a6e3a1;
            font-weight: bold;
            font-size: 12px;
            padding: 4px 8px;
            background: rgba(166, 227, 161, 0.1);
            border-radius: 4px;
        }
    """)
    parent.chat_persona_label.setToolTip("Current AI persona - manage in Persona tab")
    header_layout.addWidget(parent.chat_persona_label)
    
    header_layout.addStretch()
    
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
    parent.chat_display.anchorClicked.connect(lambda url: _handle_feedback_link(parent, url))
    parent.chat_display.setPlaceholderText(
        "Start chatting with your AI...\n\n"
        "Tips:\n"
        "- Just ask naturally - 'Generate an image of a sunset'\n"
        "- The AI auto-detects what you want to create\n"
        "- Rate responses to help the AI learn\n"
        "- Click Critique to give detailed feedback\n"
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
    """Build the message input area with all buttons."""
    input_frame = QFrame()
    input_frame.setStyleSheet(STYLE_INPUT_FRAME)
    
    input_layout = QHBoxLayout(input_frame)
    input_layout.setContentsMargins(8, 8, 8, 8)
    input_layout.setSpacing(8)
    
    # Text input
    parent.chat_input = QLineEdit()
    parent.chat_input.setPlaceholderText("Chat here...")
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
    parent.rec_btn.setToolTip("Record - Click to speak")
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


def _create_status_bar(parent, layout):
    """Build the bottom status bar with learning and voice indicators."""
    bottom_layout = QHBoxLayout()
    bottom_layout.setSpacing(8)
    
    parent.chat_status = QLabel("")
    parent.chat_status.setStyleSheet("color: #bac2de; font-size: 11px;")
    bottom_layout.addWidget(parent.chat_status)
    bottom_layout.addStretch()
    
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
    
    # Voice toggle button
    parent.voice_toggle_btn = QPushButton("OFF")
    parent.voice_toggle_btn.setFixedSize(*VOICE_TOGGLE_SIZE)
    parent.voice_toggle_btn.setCheckable(True)
    parent.voice_toggle_btn.setToolTip("AI Voice: Click to toggle auto-speak")
    parent.voice_toggle_btn.setStyleSheet(STYLE_VOICE_TOGGLE)
    parent.voice_toggle_btn.clicked.connect(lambda: _toggle_voice_output(parent))
    bottom_layout.addWidget(parent.voice_toggle_btn)
    
    layout.addLayout(bottom_layout)


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
    
    # Add feedback buttons below chat display if GUI mode manager is available
    if hasattr(parent, 'gui_mode_manager'):
        from ..widgets.quick_actions import FeedbackButtons
        parent.feedback_buttons = FeedbackButtons(parent)
        parent.feedback_buttons.good_feedback.connect(lambda: _on_feedback(parent, True))
        parent.feedback_buttons.bad_feedback.connect(lambda: _on_feedback(parent, False))
        main_layout.addWidget(parent.feedback_buttons)
    
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
    """Update the voice toggle button appearance based on state."""
    is_on = getattr(parent, 'auto_speak', False)
    
    # Update status bar voice toggle button
    if hasattr(parent, 'voice_toggle_btn'):
        parent.voice_toggle_btn.setChecked(is_on)
        
        if is_on:
            parent.voice_toggle_btn.setText("ON")
            parent.voice_toggle_btn.setToolTip("AI Voice: ON\nAI will speak responses")
        else:
            parent.voice_toggle_btn.setText("OFF")
            parent.voice_toggle_btn.setToolTip("AI Voice: OFF")
    
    # Also update the TTS button state
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
            parent._voice_thread = threading.Thread(target=lambda: _do_voice_input(parent), daemon=True)
            parent._voice_thread.start()
        except Exception as e:
            parent.rec_btn.setChecked(False)
            parent.chat_status.setText(f"Voice error: {e}")
    else:
        parent.rec_btn.setToolTip("Record - Click to speak")
        parent.chat_status.setText("Ready")
        parent._voice_thread = None


def _do_voice_input(parent):
    """Background voice recognition - automatically sends message after capture."""
    try:
        import speech_recognition as sr
        import sys
        import os
        
        recognizer = sr.Recognizer()
        
        # Suppress PyAudio stderr spam when opening microphone
        old_stderr = sys.stderr
        try:
            devnull = open(os.devnull, 'w')
            sys.stderr = devnull
            mic = sr.Microphone()
            sys.stderr = old_stderr
            devnull.close()
        except Exception:
            sys.stderr = old_stderr
            raise
        
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
        
        text = recognizer.recognize_google(audio)
        
        # Auto-send the voice input (more alive, don't put in chat box)
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: _voice_input_and_send(parent, text))
        
    except Exception as e:
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: _voice_input_error(parent, str(e)))


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
                pass
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
        pass
    
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


def _clear_chat(parent):
    """Clear the chat display and history."""
    parent.chat_display.clear()
    parent.chat_messages = []
    parent.chat_status.setText("Chat cleared")


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
        pass
    
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
                pass
        
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
    from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout, QLabel, QApplication
    
    if not hasattr(parent, 'chat_messages') or not parent.chat_messages:
        parent.chat_status.setText("No conversation to summarize")
        return
    
    if len(parent.chat_messages) < 2:
        parent.chat_status.setText("Need more messages to summarize")
        return
    
    parent.chat_status.setText("Generating summary...")
    
    try:
        from ...memory.conversation_summary import summarize_conversation, export_for_handoff
        
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
        pass
    
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


def _handle_feedback_link(parent, url):
    """Handle feedback links clicked in chat."""
    from PyQt5.QtWidgets import QInputDialog, QMessageBox, QApplication
    
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
    from PyQt5.QtWidgets import QApplication
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
    
    from PyQt5.QtGui import QTextCursor, QColor, QTextCharFormat
    
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
        from forge_ai.core.self_improvement import get_learning_engine
        
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
    from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton
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
                from forge_ai.core.self_improvement import get_learning_engine, LearningExample, LearningSource, Priority
                
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
