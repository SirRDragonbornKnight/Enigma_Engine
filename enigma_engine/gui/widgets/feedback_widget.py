"""
Feedback Widget - User feedback collection for self-improvement.

Provides thumbs up/down buttons for AI responses with optional text feedback.
"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget


class FeedbackWidget(QWidget):
    """
    Widget for collecting user feedback on AI responses.
    
    Features:
    - Thumbs up/down buttons
    - Optional text feedback for detailed explanations
    - Visual confirmation when feedback is given
    
    Signals:
        feedback_given: Emitted when user provides feedback (rating, text)
    """
    
    feedback_given = pyqtSignal(float, str)  # (rating, feedback_text)
    
    def __init__(self, message_id: str, parent=None):
        """
        Initialize feedback widget.
        
        Args:
            message_id: Unique ID for the message being rated
            parent: Parent widget
        """
        super().__init__(parent)
        self.message_id = message_id
        self.feedback_submitted = False
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)
        
        # Feedback label
        self.feedback_label = QLabel("Was this helpful?")
        self.feedback_label.setStyleSheet("""
            color: #a6adc8;
            font-size: 10px;
            padding: 2px 4px;
        """)
        layout.addWidget(self.feedback_label)
        
        # Thumbs up button
        self.thumbs_up = QPushButton("👍")
        self.thumbs_up.setFixedSize(30, 24)
        self.thumbs_up.setToolTip("Good response")
        self.thumbs_up.setStyleSheet("""
            QPushButton {
                background-color: #45475a;
                border: 1px solid #6c7086;
                border-radius: 4px;
                font-size: 12px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #585b70;
                border-color: #a6e3a1;
            }
            QPushButton:pressed {
                background-color: #a6e3a1;
            }
        """)
        self.thumbs_up.clicked.connect(lambda: self.give_feedback(1.0))
        layout.addWidget(self.thumbs_up)
        
        # Thumbs down button
        self.thumbs_down = QPushButton("👎")
        self.thumbs_down.setFixedSize(30, 24)
        self.thumbs_down.setToolTip("Poor response - click to explain why")
        self.thumbs_down.setStyleSheet("""
            QPushButton {
                background-color: #45475a;
                border: 1px solid #6c7086;
                border-radius: 4px;
                font-size: 12px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #585b70;
                border-color: #f38ba8;
            }
            QPushButton:pressed {
                background-color: #f38ba8;
            }
        """)
        self.thumbs_down.clicked.connect(lambda: self.show_feedback_input(-1.0))
        layout.addWidget(self.thumbs_down)
        
        # Optional detailed feedback text input (hidden by default)
        self.feedback_text = QLineEdit()
        self.feedback_text.setPlaceholderText("Why? (helps AI learn)")
        self.feedback_text.setMaxLength(200)
        self.feedback_text.setStyleSheet("""
            QLineEdit {
                background-color: #313244;
                border: 1px solid #6c7086;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 10px;
                color: #cdd6f4;
            }
        """)
        self.feedback_text.setVisible(False)
        self.feedback_text.returnPressed.connect(self.submit_text_feedback)
        layout.addWidget(self.feedback_text, stretch=1)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def give_feedback(self, rating: float):
        """
        Give positive feedback.
        
        Args:
            rating: Rating value (1.0 for positive, -1.0 for negative)
        """
        if self.feedback_submitted:
            return
        
        feedback_text = self.feedback_text.text() if self.feedback_text.isVisible() else ""
        self.feedback_given.emit(rating, feedback_text)
        self.feedback_submitted = True
        
        # Visual confirmation
        if rating > 0:
            self.thumbs_up.setStyleSheet("""
                QPushButton {
                    background-color: #a6e3a1;
                    color: #1e1e2e;
                    border: 2px solid #a6e3a1;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 2px;
                }
            """)
            self.feedback_label.setText("Thanks!")
            self.feedback_label.setStyleSheet("color: #a6e3a1; font-size: 10px; font-weight: bold;")
        else:
            self.thumbs_down.setStyleSheet("""
                QPushButton {
                    background-color: #f38ba8;
                    color: #1e1e2e;
                    border: 2px solid #f38ba8;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 2px;
                }
            """)
            self.feedback_label.setText("Thanks!")
            self.feedback_label.setStyleSheet("color: #f38ba8; font-size: 10px; font-weight: bold;")
        
        # Disable both buttons
        self.thumbs_up.setEnabled(False)
        self.thumbs_down.setEnabled(False)
        self.feedback_text.setVisible(False)
    
    def show_feedback_input(self, rating: float):
        """
        Show text input for negative feedback explanation.
        
        Args:
            rating: Rating value (should be -1.0 for negative)
        """
        if self.feedback_submitted:
            return
        
        self.pending_rating = rating
        self.feedback_text.setVisible(True)
        self.feedback_text.setFocus()
        
        # If they click thumbs down without explaining, submit anyway after a delay
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(10000, lambda: self.give_feedback(rating) if not self.feedback_submitted else None)
    
    def submit_text_feedback(self):
        """Submit feedback with text explanation."""
        if hasattr(self, 'pending_rating'):
            self.give_feedback(self.pending_rating)
