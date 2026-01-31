"""
Interactive Avatar Demo

Shows the avatar responding to AI in real-time.
Run this to see the avatar express itself!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QColor, QPainter, QFont

from forge_ai.avatar import (
    UnifiedAvatar, AvatarMode, AvatarType,
    AIAvatarBridge, create_avatar_bridge
)


class AvatarDemo(QMainWindow):
    """Demo window showing AI-controlled avatar."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ForgeAI Avatar Demo")
        self.setMinimumSize(800, 600)
        
        # Create avatar
        self.avatar = UnifiedAvatar()
        self.avatar.set_mode(AvatarMode.PNG_BOUNCE)
        
        # Create AI bridge
        self.bridge = create_avatar_bridge(self.avatar)
        
        # Setup UI
        self._setup_ui()
        
        # Create a test avatar image
        self._create_test_avatar()
        
    def _setup_ui(self):
        """Setup the UI."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Left side - Avatar display
        avatar_group = QGroupBox("Avatar")
        avatar_layout = QVBoxLayout(avatar_group)
        
        # Avatar widget
        self.avatar_widget = self.avatar.get_widget()
        self.avatar_widget.setFixedSize(256, 256)
        avatar_layout.addWidget(self.avatar_widget, alignment=Qt.AlignCenter)
        
        # Status label
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        avatar_layout.addWidget(self.status_label)
        
        # Emotion label
        self.emotion_label = QLabel("Emotion: neutral")
        self.emotion_label.setAlignment(Qt.AlignCenter)
        avatar_layout.addWidget(self.emotion_label)
        
        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["PNG Bounce", "2D Animated", "3D Skeletal"])
        self.mode_combo.currentIndexChanged.connect(self._change_mode)
        mode_layout.addWidget(self.mode_combo)
        avatar_layout.addLayout(mode_layout)
        
        # Type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Human", "Animal", "Robot", "Fantasy"])
        self.type_combo.currentIndexChanged.connect(self._change_type)
        type_layout.addWidget(self.type_combo)
        avatar_layout.addLayout(type_layout)
        
        # Manual controls
        controls_layout = QHBoxLayout()
        
        btn_happy = QPushButton("Happy")
        btn_happy.clicked.connect(lambda: self._set_emotion("happy"))
        controls_layout.addWidget(btn_happy)
        
        btn_sad = QPushButton("Sad")
        btn_sad.clicked.connect(lambda: self._set_emotion("sad"))
        controls_layout.addWidget(btn_sad)
        
        btn_wave = QPushButton("Wave")
        btn_wave.clicked.connect(lambda: self.avatar.gesture("wave"))
        controls_layout.addWidget(btn_wave)
        
        avatar_layout.addLayout(controls_layout)
        
        layout.addWidget(avatar_group)
        
        # Right side - Chat
        chat_group = QGroupBox("Chat with AI")
        chat_layout = QVBoxLayout(chat_group)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 10))
        chat_layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a message...")
        self.input_field.returnPressed.connect(self._send_message)
        input_layout.addWidget(self.input_field)
        
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._send_message)
        input_layout.addWidget(send_btn)
        
        chat_layout.addLayout(input_layout)
        
        # Quick test buttons
        test_layout = QHBoxLayout()
        
        btn_greet = QPushButton("Test: Greeting")
        btn_greet.clicked.connect(lambda: self._simulate_response(
            "Hello! I'm so happy to meet you! 😊 Welcome to ForgeAI!"
        ))
        test_layout.addWidget(btn_greet)
        
        btn_think = QPushButton("Test: Thinking")
        btn_think.clicked.connect(lambda: self._simulate_response(
            "Hmm, let me think about that... I believe the answer might be interesting."
        ))
        test_layout.addWidget(btn_think)
        
        btn_sad = QPushButton("Test: Apology")
        btn_sad.clicked.connect(lambda: self._simulate_response(
            "I'm sorry, I'm afraid I can't do that. Unfortunately, it's not possible."
        ))
        test_layout.addWidget(btn_sad)
        
        btn_surprise = QPushButton("Test: Surprise")
        btn_surprise.clicked.connect(lambda: self._simulate_response(
            "Wow! That's amazing! I can't believe it!! This is incredible!"
        ))
        test_layout.addWidget(btn_surprise)
        
        chat_layout.addLayout(test_layout)
        
        # Explicit command test buttons (AI intentionally controls avatar)
        explicit_layout = QHBoxLayout()
        explicit_layout.addWidget(QLabel("Explicit AI Commands:"))
        
        btn_explicit_greet = QPushButton("Wave + Happy")
        btn_explicit_greet.clicked.connect(lambda: self._simulate_response(
            "[emotion:happy][gesture:wave] Hello! Great to see you!"
        ))
        explicit_layout.addWidget(btn_explicit_greet)
        
        btn_explicit_think = QPushButton("Think + Nod")
        btn_explicit_think.clicked.connect(lambda: self._simulate_response(
            "[emotion:thinking][gesture:nod][action:think] Let me consider that carefully..."
        ))
        explicit_layout.addWidget(btn_explicit_think)
        
        btn_explicit_excited = QPushButton("Excited + Clap")
        btn_explicit_excited.clicked.connect(lambda: self._simulate_response(
            "[emotion:excited][gesture:clap] That's fantastic news! Congratulations!"
        ))
        explicit_layout.addWidget(btn_explicit_excited)
        
        btn_explicit_shrug = QPushButton("Confused + Shrug")
        btn_explicit_shrug.clicked.connect(lambda: self._simulate_response(
            "[emotion:confused][gesture:shrug] Hmm, I'm not entirely sure about that..."
        ))
        explicit_layout.addWidget(btn_explicit_shrug)
        
        chat_layout.addLayout(explicit_layout)
        
        layout.addWidget(chat_group)
    
    def _create_test_avatar(self):
        """Create a simple test avatar image."""
        # Create a cute circular avatar
        pixmap = QPixmap(256, 256)
        pixmap.fill(QColor(0, 0, 0, 0))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Body/face - blue circle
        painter.setBrush(QColor(100, 150, 255))
        painter.setPen(QColor(80, 120, 220))
        painter.drawEllipse(28, 28, 200, 200)
        
        # Eyes - white circles
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QColor(200, 200, 200))
        painter.drawEllipse(70, 80, 45, 45)
        painter.drawEllipse(140, 80, 45, 45)
        
        # Pupils - black circles
        painter.setBrush(QColor(30, 30, 30))
        painter.drawEllipse(85, 95, 20, 20)
        painter.drawEllipse(155, 95, 20, 20)
        
        # Smile - arc
        painter.setPen(QColor(50, 50, 50))
        painter.setBrush(Qt.NoBrush)
        from PyQt5.QtGui import QPen
        pen = QPen(QColor(50, 50, 50), 4)
        painter.setPen(pen)
        from PyQt5.QtCore import QRect
        painter.drawArc(QRect(80, 120, 96, 60), -30 * 16, -120 * 16)
        
        painter.end()
        
        # Save and load
        pixmap.save('demo_avatar.png')
        self.avatar.load('demo_avatar.png')
        
        self.chat_display.append("Avatar loaded! Try the test buttons or type a message.\n")
    
    def _change_mode(self, index):
        """Change avatar mode."""
        modes = [AvatarMode.PNG_BOUNCE, AvatarMode.ANIMATED_2D, AvatarMode.SKELETAL_3D]
        self.avatar.set_mode(modes[index])
        self.status_label.setText(f"Mode: {modes[index].name}")
    
    def _change_type(self, index):
        """Change avatar type."""
        types = [AvatarType.HUMAN, AvatarType.ANIMAL, AvatarType.ROBOT, AvatarType.FANTASY]
        self.avatar.set_avatar_type(types[index])
        self.status_label.setText(f"Type: {types[index].name}")
    
    def _set_emotion(self, emotion):
        """Manually set emotion."""
        self.avatar.set_emotion(emotion)
        self.emotion_label.setText(f"Emotion: {emotion}")
    
    def _send_message(self):
        """Send a message (simulated AI response)."""
        text = self.input_field.text().strip()
        if not text:
            return
        
        self.chat_display.append(f"You: {text}\n")
        self.input_field.clear()
        
        # Check for user greeting - avatar reacts
        if any(g in text.lower() for g in ['hello', 'hi', 'hey']):
            self.avatar.gesture('wave')
        
        # Simulate AI thinking then responding
        self.status_label.setText("Status: Thinking...")
        self.avatar.set_emotion("thinking")
        
        QTimer.singleShot(1000, lambda: self._generate_response(text))
    
    def _generate_response(self, user_text):
        """Generate a simulated AI response."""
        # Simple response logic
        responses = {
            "hello": "Hello! I'm so happy to meet you! How can I help you today? 😊",
            "hi": "Hi there! Welcome! What would you like to talk about?",
            "how are you": "I'm doing great, thank you for asking! I'm excited to chat with you!",
            "help": "Of course! I'd be glad to help. Let me think about what you need...",
            "bye": "Goodbye! It was wonderful talking with you. See you soon! 👋",
            "thank": "You're welcome! I'm happy I could help! 😊",
        }
        
        # Find matching response
        response = "That's interesting! Let me think about that... I believe we can figure this out together!"
        for key, val in responses.items():
            if key in user_text.lower():
                response = val
                break
        
        self._simulate_response(response)
    
    def _simulate_response(self, response: str):
        """Simulate streaming AI response with avatar expression."""
        self.chat_display.append("AI: ")
        
        # Start talking
        self.bridge.on_response_start()
        self.status_label.setText("Status: Talking")
        
        # Stream response - but commands get stripped by the bridge
        self._stream_index = 0
        self._stream_response = response
        self._stream_buffer = ""
        self._stream_timer = QTimer()
        self._stream_timer.timeout.connect(self._stream_next_char)
        self._stream_timer.start(30)  # 30ms per character
    
    def _stream_next_char(self):
        """Stream next character of response."""
        if self._stream_index < len(self._stream_response):
            char = self._stream_response[self._stream_index]
            self._stream_index += 1
            
            # Buffer characters to handle commands spanning multiple chars
            self._stream_buffer += char
            
            # Check if we have a complete command or regular text
            # Send to bridge - it returns cleaned text
            cleaned = self.bridge.on_response_chunk(self._stream_buffer)
            
            # If cleaned text returned, display it
            if cleaned.strip():
                cursor = self.chat_display.textCursor()
                cursor.movePosition(cursor.End)
                cursor.insertText(cleaned)
                self.chat_display.setTextCursor(cursor)
            
            # Clear buffer after processing
            self._stream_buffer = ""
            
            # Update emotion label
            self.emotion_label.setText(f"Emotion: {self.bridge._last_emotion}")
            
            # Show any explicit commands that were executed
            cmds = self.bridge.get_commands_for_response()
            if cmds:
                cmd_str = ", ".join(f"{c.command_type}:{c.value}" for c in cmds[-3:])
                self.status_label.setText(f"Commands: {cmd_str}")
        else:
            # Done streaming
            self._stream_timer.stop()
            self.bridge.on_response_end()
            self.status_label.setText("Status: Idle")
            self.chat_display.append("\n")


def main():
    app = QApplication(sys.argv)
    
    # Set style
    app.setStyle('Fusion')
    
    window = AvatarDemo()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
