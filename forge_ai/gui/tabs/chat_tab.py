"""Chat tab for ForgeAI GUI."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QTextBrowser, QLineEdit, QLabel, QFrame, QSplitter,
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
    parent.btn_new_chat = QPushButton("+ New Chat")
    parent.btn_new_chat.setToolTip("Start a fresh conversation (saves current chat first)")
    parent.btn_new_chat.setMaximumWidth(90)
    parent.btn_new_chat.clicked.connect(lambda: _new_chat(parent))
    parent.btn_new_chat.setStyleSheet("""
        QPushButton {
            background-color: #a6e3a1;
            color: #1e1e2e;
            font-weight: bold;
            padding: 4px 8px;
        }
        QPushButton:hover {
            background-color: #94e2d5;
        }
    """)
    header_layout.addWidget(parent.btn_new_chat)
    
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
    parent.chat_display = QTextBrowser()
    parent.chat_display.setReadOnly(True)
    parent.chat_display.setTextInteractionFlags(
        Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard | Qt.LinksAccessibleByMouse
    )
    parent.chat_display.setOpenExternalLinks(False)  # Handle links ourselves
    parent.chat_display.anchorClicked.connect(lambda url: _handle_feedback_link(parent, url))
    parent.chat_display.setPlaceholderText(
        "Start chatting with your AI...\n\n"
        "Tips:\n"
        "• Just ask naturally - 'Generate an image of a sunset'\n"
        "• The AI auto-detects what you want to create\n"
        "• Rate responses with 👍👎 to help the AI learn\n"
        "• Click ✏️ Critique to give detailed feedback"
    )
    parent.chat_display.setStyleSheet("""
        QTextEdit {
            font-size: 13px;
            line-height: 1.5;
            padding: 10px;
        }
    """)
    layout.addWidget(parent.chat_display, stretch=1)
    
    # Thinking/Status panel (shown during generation) - ABOVE the input
    parent.thinking_frame = QFrame()
    parent.thinking_frame.setStyleSheet("""
        QFrame {
            background: rgba(249, 226, 175, 0.15);
            border: 1px solid #f9e2af;
            border-radius: 6px;
            padding: 4px;
        }
    """)
    thinking_layout = QHBoxLayout(parent.thinking_frame)
    thinking_layout.setContentsMargins(8, 4, 8, 4)
    
    parent.thinking_label = QLabel("🤔 Thinking...")
    parent.thinking_label.setStyleSheet("color: #f9e2af; font-size: 12px;")
    thinking_layout.addWidget(parent.thinking_label)
    
    thinking_layout.addStretch()
    
    parent.thinking_frame.hide()  # Hidden by default
    layout.addWidget(parent.thinking_frame)
    
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
    
    # Stop button (hidden by default, shown during generation)
    parent.stop_btn = QPushButton("⏹ Stop")
    parent.stop_btn.setToolTip("Stop AI generation")
    parent.stop_btn.setMinimumWidth(70)
    parent.stop_btn.setMinimumHeight(40)
    parent.stop_btn.setStyleSheet("""
        QPushButton {
            background-color: #f38ba8;
            color: #1e1e2e;
            font-weight: bold;
            font-size: 13px;
            border-radius: 6px;
            padding: 8px 12px;
        }
        QPushButton:hover {
            background-color: #eba0ac;
        }
    """)
    parent.stop_btn.clicked.connect(lambda: _stop_generation(parent))
    parent.stop_btn.hide()  # Hidden by default
    input_layout.addWidget(parent.stop_btn)
    
    # Speak button (for TTS)
    parent.btn_speak = QPushButton("Voice")
    parent.btn_speak.setToolTip("Speak last response")
    parent.btn_speak.setMinimumWidth(90)
    parent.btn_speak.setMinimumHeight(40)
    parent.btn_speak.setStyleSheet("""
        QPushButton {
            background-color: #cba6f7;
            color: #1e1e2e;
            font-weight: bold;
            font-size: 13px;
            border-radius: 6px;
            padding: 8px 12px;
        }
        QPushButton:hover {
            background-color: #f5c2e7;
        }
    """)
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
    
    # Learning indicator with detailed tooltip
    parent.learning_indicator = QLabel("📚 Learning: ON")
    parent.learning_indicator.setStyleSheet("color: #a6e3a1; font-size: 11px;")
    parent.learning_indicator.setToolTip(
        "When Learning is ON, the AI records your conversations and uses them to improve.\n\n"
        "How it works:\n"
        "• Each Q&A pair is saved to the model's training data\n"
        "• After enough interactions, the model can be retrained\n"
        "• This helps the AI learn your preferences and style\n\n"
        "Note: Learning only works with local Forge models.\n"
        "HuggingFace models (GPT-2, Mistral, etc.) don't use this feature.\n\n"
        "Toggle in Settings menu or click here to toggle."
    )
    parent.learning_indicator.setCursor(Qt.PointingHandCursor)
    parent.learning_indicator.mousePressEvent = lambda e: _toggle_learning(parent)
    status_layout.addWidget(parent.learning_indicator)
    
    layout.addLayout(status_layout)
    
    # Initialize auto_speak
    parent.auto_speak = False
    
    w.setLayout(layout)
    return w


def _toggle_learning(parent):
    """Toggle the learning mode on/off."""
    current = getattr(parent, 'learn_while_chatting', True)
    parent.learn_while_chatting = not current
    
    if parent.learn_while_chatting:
        parent.learning_indicator.setText("📚 Learning: ON")
        parent.learning_indicator.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        parent.chat_status.setText("Learning enabled - AI will learn from conversations")
    else:
        parent.learning_indicator.setText("📚 Learning: OFF")
        parent.learning_indicator.setStyleSheet("color: #6c7086; font-size: 11px;")
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
    """Start a new chat - save current chat first, then clear."""
    # Save current chat if there's content
    if hasattr(parent, 'chat_messages') and parent.chat_messages:
        if hasattr(parent, '_save_current_chat'):
            parent._save_current_chat()
            parent.chat_status.setText("Previous chat saved. Starting new conversation...")
    
    # Clear the chat
    parent.chat_display.clear()
    parent.chat_messages = []
    
    # Reset any HuggingFace conversation history
    if hasattr(parent, 'engine') and parent.engine:
        if hasattr(parent.engine, 'model') and hasattr(parent.engine.model, 'reset_conversation'):
            try:
                parent.engine.model.reset_conversation()
            except:
                pass
    
    # Clear tool output memory for fresh session
    try:
        from ...tools.history import get_output_memory
        get_output_memory().clear()
    except:
        pass
    
    # Show welcome message
    model_name = parent.current_model_name if hasattr(parent, 'current_model_name') else "AI"
    parent.chat_display.append(
        f'<div style="color: #a6e3a1; padding: 8px;">'
        f'<b>✨ New conversation started with {model_name}</b><br>'
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


def _stop_generation(parent):
    """Stop the current AI generation."""
    if hasattr(parent, '_ai_worker') and parent._ai_worker and parent._ai_worker.isRunning():
        parent._ai_worker.stop()
        parent.chat_status.setText("⏹ Stopping generation...")
        parent.stop_btn.setEnabled(False)
        parent.stop_btn.setText("Stopping...")


def _handle_feedback_link(parent, url):
    """Handle feedback links clicked in chat."""
    from PyQt5.QtWidgets import QInputDialog, QMessageBox
    
    url_str = url.toString() if hasattr(url, 'toString') else str(url)
    
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
        parent.chat_status.setText("👍 Thanks for the positive feedback!")
        # Save good example with high quality score
        if hasattr(parent, 'brain') and parent.brain and response_data:
            parent.brain.record_interaction(
                response_data['user_input'],
                response_data['ai_response'],
                quality=1.0  # High quality
            )
            parent.chat_status.setText("👍 Feedback saved - AI will learn from this good example!")
    
    elif feedback_type == 'bad':
        parent.chat_status.setText("👎 Sorry about that. What went wrong?")
        # Ask for quick reason
        reason, ok = QInputDialog.getItem(
            parent,
            "What was wrong?",
            "Please select what was wrong with the response:",
            ["Incorrect/Wrong info", "Off-topic", "Too long/verbose", "Too short", "Confusing", "Other"],
            0, False
        )
        if ok and reason:
            parent.chat_status.setText(f"👎 Feedback noted: {reason}")
            # Don't save this as training data (or save with low quality)
            if hasattr(parent, 'brain') and parent.brain and response_data:
                # Record with low quality so it's deprioritized
                parent.brain.add_memory(
                    f"BAD RESPONSE - {reason}: Q: {response_data['user_input'][:100]}",
                    importance=0.3,
                    category="negative_feedback"
                )
    
    elif feedback_type == 'critique':
        # Open detailed critique dialog
        _show_critique_dialog(parent, response_id, response_data)


def _show_critique_dialog(parent, response_id, response_data):
    """Show a dialog for detailed critique of a response."""
    from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton, QComboBox
    
    dialog = QDialog(parent)
    dialog.setWindowTitle("✏️ Critique Response")
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
    issue_combo = QComboBox()
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
            # Save corrected example as training data
            if hasattr(parent, 'brain') and parent.brain:
                # Save the corrected version with high quality
                parent.brain.record_interaction(
                    response_data['user_input'],
                    correction,
                    quality=1.0
                )
                parent.chat_status.setText(f"✏️ Correction saved! AI will learn the better response.")
            
            # Show in chat
            parent.chat_display.append(
                f'<div style="background-color: #313244; padding: 8px; margin: 4px 0; border-radius: 8px; border-left: 3px solid #89b4fa;">'
                f'<b style="color: #89b4fa;">📝 Correction saved ({issue}):</b><br>'
                f'<i style="color: #a6e3a1;">Better response:</i> {correction[:200]}...</div>'
            )
        
        dialog.accept()
    
    save_btn.clicked.connect(save_critique)
    cancel_btn.clicked.connect(dialog.reject)
    
    dialog.exec_()
