"""
AI-Avatar Bridge

Connects the AI chat system to the avatar, so the avatar expresses
what the AI is saying in real-time.

When you chat with the AI:
- Avatar starts talking animation when AI begins responding
- Avatar shows emotions based on response content
- Avatar does gestures for greetings, questions, etc.
- Avatar stops talking when AI finishes

Usage:
    from forge_ai.avatar.ai_bridge import AIAvatarBridge
    
    # Create bridge with your avatar
    bridge = AIAvatarBridge(avatar)
    
    # Connect to AI response events
    bridge.on_response_start()  # Call when AI starts generating
    bridge.on_response_chunk("Hello!")  # Call for each text chunk
    bridge.on_response_end()  # Call when AI finishes
    
    # Or use the convenience wrapper
    response = bridge.generate_with_avatar(engine, prompt)
"""

import re
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field

try:
    from PyQt5.QtCore import QObject, pyqtSignal, QTimer
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QObject = object
    pyqtSignal = lambda *args: None


@dataclass
class EmotionKeywords:
    """Keywords that indicate emotions in AI responses."""
    
    HAPPY: List[str] = field(default_factory=lambda: [
        "happy", "glad", "great", "awesome", "wonderful", "fantastic",
        "excited", "love", "enjoy", "amazing", "excellent", "perfect",
        "yay", "hooray", "nice", "good", "pleased", "delighted",
        "!", "😊", "😄", "🎉", ":)", ":D", "haha", "lol"
    ])
    
    SAD: List[str] = field(default_factory=lambda: [
        "sorry", "sad", "unfortunately", "regret", "apologize",
        "can't", "cannot", "unable", "impossible", "difficult",
        "worry", "concern", "afraid", "disappointed", "miss",
        "😢", "😔", ":("
    ])
    
    SURPRISED: List[str] = field(default_factory=lambda: [
        "wow", "whoa", "amazing", "incredible", "unbelievable",
        "surprising", "unexpected", "really", "seriously",
        "oh my", "no way", "what", "!!", "😮", "😲", "🤯"
    ])
    
    THINKING: List[str] = field(default_factory=lambda: [
        "hmm", "let me think", "consider", "perhaps", "maybe",
        "possibly", "i think", "i believe", "interesting",
        "analyzing", "processing", "calculating", "...", "🤔"
    ])
    
    CONFUSED: List[str] = field(default_factory=lambda: [
        "confused", "unclear", "don't understand", "not sure",
        "what do you mean", "could you clarify", "?", "hmm",
        "🤨", "😕"
    ])


@dataclass  
class GestureKeywords:
    """Keywords that trigger gestures."""
    
    WAVE: List[str] = field(default_factory=lambda: [
        "hello", "hi", "hey", "greetings", "welcome",
        "goodbye", "bye", "see you", "farewell", "👋"
    ])
    
    NOD: List[str] = field(default_factory=lambda: [
        "yes", "correct", "right", "exactly", "indeed",
        "agree", "sure", "of course", "absolutely", "definitely"
    ])
    
    SHAKE: List[str] = field(default_factory=lambda: [
        "no", "incorrect", "wrong", "disagree", "not really",
        "i don't think so", "nope"
    ])
    
    SHRUG: List[str] = field(default_factory=lambda: [
        "i don't know", "not sure", "maybe", "perhaps",
        "hard to say", "depends", "🤷"
    ])
    
    POINT: List[str] = field(default_factory=lambda: [
        "look at", "check out", "see here", "notice",
        "here is", "this is", "that is", "👉"
    ])


class AIAvatarBridge(QObject if HAS_PYQT else object):
    """
    Bridge between AI chat and avatar expression.
    
    Analyzes AI responses in real-time and controls the avatar
    to express emotions and gestures appropriately.
    """
    
    # Signals
    if HAS_PYQT:
        emotion_detected = pyqtSignal(str)
        gesture_triggered = pyqtSignal(str)
    
    def __init__(self, avatar=None):
        if HAS_PYQT:
            super().__init__()
        
        self.avatar = avatar
        self._is_responding = False
        self._response_buffer = ""
        self._last_emotion = "neutral"
        self._gesture_cooldown = False
        
        # Keyword matchers
        self._emotion_keywords = EmotionKeywords()
        self._gesture_keywords = GestureKeywords()
        
        # Cooldown timer for gestures (don't spam)
        if HAS_PYQT:
            self._gesture_timer = QTimer()
            self._gesture_timer.timeout.connect(self._reset_gesture_cooldown)
            self._gesture_timer.setSingleShot(True)
    
    def set_avatar(self, avatar):
        """Set or change the avatar."""
        self.avatar = avatar
    
    # =========================================================================
    # RESPONSE LIFECYCLE
    # =========================================================================
    
    def on_response_start(self):
        """Called when AI starts generating a response."""
        self._is_responding = True
        self._response_buffer = ""
        self._last_emotion = "neutral"
        
        if self.avatar:
            self.avatar.start_talking()
    
    def on_response_chunk(self, text: str):
        """
        Called for each chunk of AI response (streaming).
        
        Analyzes text for emotions and gestures in real-time.
        """
        if not self._is_responding:
            return
        
        self._response_buffer += text
        
        # Analyze for emotion (check periodically, not every character)
        if len(self._response_buffer) % 20 == 0 or text.endswith(('.', '!', '?')):
            emotion = self._detect_emotion(self._response_buffer)
            if emotion != self._last_emotion:
                self._last_emotion = emotion
                if self.avatar:
                    self.avatar.set_emotion(emotion)
                if HAS_PYQT:
                    self.emotion_detected.emit(emotion)
        
        # Check for gestures (with cooldown)
        if not self._gesture_cooldown:
            gesture = self._detect_gesture(text)
            if gesture:
                self._trigger_gesture(gesture)
    
    def on_response_end(self):
        """Called when AI finishes generating response."""
        self._is_responding = False
        
        # Final emotion check on complete response
        final_emotion = self._detect_emotion(self._response_buffer)
        if self.avatar:
            self.avatar.set_emotion(final_emotion)
            self.avatar.stop_talking()
        
        self._response_buffer = ""
    
    # =========================================================================
    # EMOTION DETECTION
    # =========================================================================
    
    def _detect_emotion(self, text: str) -> str:
        """Detect emotion from text content."""
        text_lower = text.lower()
        
        # Count keyword matches for each emotion
        scores = {
            "happy": 0,
            "sad": 0,
            "surprised": 0,
            "thinking": 0,
        }
        
        for keyword in self._emotion_keywords.HAPPY:
            if keyword.lower() in text_lower:
                scores["happy"] += 1
        
        for keyword in self._emotion_keywords.SAD:
            if keyword.lower() in text_lower:
                scores["sad"] += 1
        
        for keyword in self._emotion_keywords.SURPRISED:
            if keyword.lower() in text_lower:
                scores["surprised"] += 1
        
        for keyword in self._emotion_keywords.THINKING:
            if keyword.lower() in text_lower:
                scores["thinking"] += 1
        
        # Get highest scoring emotion
        max_score = max(scores.values())
        if max_score == 0:
            return "neutral"
        
        for emotion, score in scores.items():
            if score == max_score:
                return emotion
        
        return "neutral"
    
    def _detect_gesture(self, text: str) -> Optional[str]:
        """Detect if text should trigger a gesture."""
        text_lower = text.lower()
        
        for keyword in self._gesture_keywords.WAVE:
            if keyword.lower() in text_lower:
                return "wave"
        
        for keyword in self._gesture_keywords.NOD:
            if keyword.lower() in text_lower:
                return "nod"
        
        for keyword in self._gesture_keywords.SHAKE:
            if keyword.lower() in text_lower:
                return "shake"
        
        for keyword in self._gesture_keywords.SHRUG:
            if keyword.lower() in text_lower:
                return "shrug"
        
        return None
    
    def _trigger_gesture(self, gesture: str):
        """Trigger a gesture with cooldown."""
        if self._gesture_cooldown:
            return
        
        self._gesture_cooldown = True
        
        if self.avatar:
            self.avatar.gesture(gesture)
        
        if HAS_PYQT:
            self.gesture_triggered.emit(gesture)
            self._gesture_timer.start(2000)  # 2 second cooldown
    
    def _reset_gesture_cooldown(self):
        """Reset gesture cooldown."""
        self._gesture_cooldown = False
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def generate_with_avatar(self, engine, prompt: str, **kwargs) -> str:
        """
        Generate AI response with avatar expression.
        
        Wraps the ForgeEngine generate call to automatically
        control the avatar during generation.
        
        Args:
            engine: ForgeEngine instance
            prompt: The user's prompt
            **kwargs: Additional args for engine.generate()
            
        Returns:
            The AI's response text
        """
        self.on_response_start()
        
        try:
            # Check if engine supports streaming
            if hasattr(engine, 'generate_stream'):
                # Streaming mode
                response_parts = []
                for chunk in engine.generate_stream(prompt, **kwargs):
                    response_parts.append(chunk)
                    self.on_response_chunk(chunk)
                response = ''.join(response_parts)
            else:
                # Non-streaming mode
                response = engine.generate(prompt, **kwargs)
                self.on_response_chunk(response)
            
            return response
            
        finally:
            self.on_response_end()
    
    def wrap_callback(self, original_callback: Callable) -> Callable:
        """
        Wrap a streaming callback to include avatar control.
        
        Usage:
            def my_callback(chunk):
                print(chunk, end='')
            
            wrapped = bridge.wrap_callback(my_callback)
            engine.generate_stream(prompt, callback=wrapped)
        """
        def wrapped(chunk):
            self.on_response_chunk(chunk)
            if original_callback:
                original_callback(chunk)
        return wrapped


class AvatarChatIntegration:
    """
    Full integration for chat windows.
    
    Drop-in integration for PyQt chat interfaces.
    """
    
    def __init__(self, chat_widget=None, avatar=None):
        self.chat_widget = chat_widget
        self.bridge = AIAvatarBridge(avatar)
        
        # Connect signals if chat widget has them
        if chat_widget and hasattr(chat_widget, 'response_started'):
            chat_widget.response_started.connect(self.bridge.on_response_start)
        if chat_widget and hasattr(chat_widget, 'response_chunk'):
            chat_widget.response_chunk.connect(self.bridge.on_response_chunk)
        if chat_widget and hasattr(chat_widget, 'response_finished'):
            chat_widget.response_finished.connect(self.bridge.on_response_end)
    
    def set_avatar(self, avatar):
        """Set the avatar to control."""
        self.bridge.set_avatar(avatar)
    
    def process_user_input(self, text: str):
        """
        Process user input to prepare avatar.
        
        Can detect user emotion/intent to have avatar react.
        """
        text_lower = text.lower()
        
        # User saying hi - avatar waves
        if any(g in text_lower for g in ['hello', 'hi', 'hey']):
            if self.bridge.avatar:
                self.bridge.avatar.gesture('wave')
        
        # User asking question - avatar looks attentive
        if '?' in text:
            if self.bridge.avatar:
                self.bridge.avatar.listen()


# =============================================================================
# CONVENIENCE FUNCTIONS  
# =============================================================================

def create_avatar_bridge(avatar) -> AIAvatarBridge:
    """Create an AI-Avatar bridge."""
    return AIAvatarBridge(avatar)


def integrate_avatar_with_chat(chat_widget, avatar) -> AvatarChatIntegration:
    """Integrate avatar with a chat widget."""
    return AvatarChatIntegration(chat_widget, avatar)
