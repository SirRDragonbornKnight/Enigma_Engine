"""
Quick Actions Bar - Floating action bar for common tasks.

Provides quick access to frequently used features like voice input,
screenshot capture, and game mode toggle.
"""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class QuickActionsBar(QWidget):
    """
    Floating action bar for common tasks.
    
    Provides buttons for:
    - Screenshot capture (send to AI)
    - Voice input
    - Game mode toggle
    - New conversation
    - Quick image generation
    """
    
    # Signals
    screenshot_clicked = pyqtSignal()
    voice_clicked = pyqtSignal()
    game_mode_clicked = pyqtSignal()
    new_chat_clicked = pyqtSignal()
    quick_generate_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the quick actions UI."""
        self.setObjectName("quickActionsBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        # Style the container
        self.setStyleSheet("""
            QWidget#quickActionsBar {
                background-color: #12121a;
                border: 1px solid #1e293b;
                border-radius: 8px;
            }
            QPushButton {
                background-color: #1a1a24;
                color: #e2e8f0;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #6366f1;
                color: white;
            }
            QPushButton:pressed {
                background-color: #4f46e5;
            }
        """)
        
        # Define actions (label, tooltip, signal)
        actions = [
            ("Screenshot", "Send screenshot to AI (Ctrl+Shift+S)", self.screenshot_clicked),
            ("Voice", "Voice input (Ctrl+Shift+V)", self.voice_clicked),
            ("Game Mode", "Toggle game mode (Ctrl+Shift+Space)", self.game_mode_clicked),
            ("New Chat", "New conversation (Ctrl+N)", self.new_chat_clicked),
            ("Quick Gen", "Quick image generation (Ctrl+Shift+G)", self.quick_generate_clicked),
        ]
        
        for label, tooltip, signal in actions:
            btn = QPushButton(label)
            btn.setToolTip(tooltip)
            btn.clicked.connect(signal.emit)
            btn.setFixedHeight(32)
            layout.addWidget(btn)
        
        layout.addStretch()


class FeedbackButtons(QWidget):
    """
    Feedback buttons for rating AI responses.
    
    Provides thumbs up/down buttons for users to rate the last response.
    """
    
    # Signals
    good_feedback = pyqtSignal()
    bad_feedback = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the feedback UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Label
        label = QLabel("Feedback:")
        label.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(label)
        
        # Good button
        self.good_btn = QPushButton("Good")
        self.good_btn.setToolTip("This response was helpful")
        self.good_btn.clicked.connect(self._on_good)
        self.good_btn.setFixedWidth(80)
        self.good_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a1a24;
                color: #e2e8f0;
                border: 1px solid #1e293b;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #22c55e;
                color: white;
                border-color: #22c55e;
            }
            QPushButton:pressed {
                background-color: #16a34a;
            }
            QPushButton[active="true"] {
                background-color: #22c55e;
                color: white;
                border-color: #22c55e;
            }
        """)
        layout.addWidget(self.good_btn)
        
        # Bad button
        self.bad_btn = QPushButton("Bad")
        self.bad_btn.setToolTip("This response was not helpful")
        self.bad_btn.clicked.connect(self._on_bad)
        self.bad_btn.setFixedWidth(80)
        self.bad_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a1a24;
                color: #e2e8f0;
                border: 1px solid #1e293b;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #ef4444;
                color: white;
                border-color: #ef4444;
            }
            QPushButton:pressed {
                background-color: #dc2626;
            }
            QPushButton[active="true"] {
                background-color: #ef4444;
                color: white;
                border-color: #ef4444;
            }
        """)
        layout.addWidget(self.bad_btn)
        
        layout.addStretch()
        
        self._active_button = None
    
    def _on_good(self):
        """Handle good feedback."""
        self._active_button = "good"
        self.good_btn.setProperty("active", "true")
        self.bad_btn.setProperty("active", "false")
        self.good_btn.style().unpolish(self.good_btn)
        self.good_btn.style().polish(self.good_btn)
        self.bad_btn.style().unpolish(self.bad_btn)
        self.bad_btn.style().polish(self.bad_btn)
        self.good_feedback.emit()
    
    def _on_bad(self):
        """Handle bad feedback."""
        self._active_button = "bad"
        self.good_btn.setProperty("active", "false")
        self.bad_btn.setProperty("active", "true")
        self.good_btn.style().unpolish(self.good_btn)
        self.good_btn.style().polish(self.good_btn)
        self.bad_btn.style().unpolish(self.bad_btn)
        self.bad_btn.style().polish(self.bad_btn)
        self.bad_feedback.emit()
    
    def reset(self):
        """Reset feedback buttons."""
        self._active_button = None
        self.good_btn.setProperty("active", "false")
        self.bad_btn.setProperty("active", "false")
        self.good_btn.style().unpolish(self.good_btn)
        self.good_btn.style().polish(self.good_btn)
        self.bad_btn.style().unpolish(self.bad_btn)
        self.bad_btn.style().polish(self.bad_btn)


class GameModeIndicator(QWidget):
    """
    Game mode status indicator.
    
    Shows when game mode is active and displays resource usage.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._is_active = False
    
    def _setup_ui(self):
        """Set up the indicator UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a24;
                border: 1px solid #1e293b;
                border-radius: 6px;
            }
            QLabel {
                color: #e2e8f0;
                font-size: 11px;
            }
        """)
        
        self.status_label = QLabel("Game Mode: Inactive")
        layout.addWidget(self.status_label)
        
        self.cpu_label = QLabel("CPU: --")
        layout.addWidget(self.cpu_label)
        
        self.hide()  # Hidden by default
    
    def set_active(self, active: bool):
        """Set game mode active state."""
        self._is_active = active
        if active:
            self.status_label.setText("Game Mode: Active")
            self.status_label.setStyleSheet("color: #22c55e; font-size: 11px; font-weight: bold;")
            self.show()
        else:
            self.status_label.setText("Game Mode: Inactive")
            self.status_label.setStyleSheet("color: #64748b; font-size: 11px;")
            self.hide()
    
    def update_cpu(self, cpu_percent: float):
        """Update CPU usage display."""
        self.cpu_label.setText(f"CPU: {cpu_percent:.1f}%")
        
        # Color code based on usage
        if cpu_percent < 20:
            color = "#22c55e"  # Green
        elif cpu_percent < 50:
            color = "#f59e0b"  # Orange
        else:
            color = "#ef4444"  # Red
        
        self.cpu_label.setStyleSheet(f"color: {color}; font-size: 11px;")
