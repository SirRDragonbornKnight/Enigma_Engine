"""
AI-Avatar Bridge

Connects the AI chat system to the avatar, so the avatar expresses
what the AI is saying in real-time.

TWO CONTROL MODES:
==================
1. AUTOMATIC: Detects emotions from AI text via SentimentAnalyzer (advanced NLP)
2. EXPLICIT: AI uses special commands to intentionally control avatar

EXPLICIT CONTROL COMMANDS (AI can include these in output):
    [emotion:happy]     - Set avatar emotion
    [gesture:wave]      - Trigger gesture animation
    [action:think]      - Trigger action state
    [expression:wink]   - Trigger facial expression
    
Commands are STRIPPED from displayed text, so users see clean responses.
The AI can include them anywhere in its output.

SUPPORTED COMMANDS:
    Emotions: happy, sad, surprised, thinking, confused, neutral, angry, excited
    Gestures: wave, nod, shake, shrug, point, bow, clap, thumbsup
    Actions: think, listen, talk, idle, celebrate, acknowledge
    Expressions: wink, blink, eyebrow_raise, smile, frown, smirk

Example AI output:
    "[emotion:excited][gesture:wave] Hello! I'm so glad to see you!"
    User sees: "Hello! I'm so glad to see you!" (avatar waves excitedly)

Usage:
    from enigma_engine.avatar.ai_bridge import AIAvatarBridge
    
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

# Try to import the advanced sentiment analyzer
try:
    from .sentiment_analyzer import SentimentAnalyzer, analyze_for_avatar
    HAS_SENTIMENT_ANALYZER = True
except ImportError:
    HAS_SENTIMENT_ANALYZER = False
    SentimentAnalyzer = None
    analyze_for_avatar = None
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    from PyQt5.QtCore import QObject, QTimer, pyqtSignal
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QObject = object
    pyqtSignal = lambda *args: None


# =============================================================================
# EXPLICIT COMMAND SYSTEM
# =============================================================================

@dataclass
class AvatarCommand:
    """A parsed avatar control command."""
    command_type: str  # 'emotion', 'gesture', 'action', 'expression'
    value: str         # 'happy', 'wave', etc.
    raw_text: str      # Original '[emotion:happy]' for removal


@dataclass
class ExplicitCommands:
    """
    Valid explicit control commands the AI can use.
    
    The AI can include these in its output to intentionally control the avatar.
    Commands are stripped from displayed text.
    """
    
    # Emotions - how the avatar feels
    EMOTIONS: list[str] = field(default_factory=lambda: [
        "happy", "sad", "surprised", "thinking", "confused",
        "neutral", "angry", "excited", "curious", "worried",
        "proud", "embarrassed", "loving", "mischievous", "focused"
    ])
    
    # Gestures - physical movements
    GESTURES: list[str] = field(default_factory=lambda: [
        "wave", "nod", "shake", "shrug", "point", "bow",
        "clap", "thumbsup", "peace", "facepalm", "salute",
        "dance", "spin", "jump"
    ])
    
    # Actions - state changes
    ACTIONS: list[str] = field(default_factory=lambda: [
        "think", "listen", "talk", "idle", "celebrate",
        "acknowledge", "attention", "relax", "alert", "sleep"
    ])
    
    # Facial expressions
    EXPRESSIONS: list[str] = field(default_factory=lambda: [
        "wink", "blink", "eyebrow_raise", "smile", "frown",
        "smirk", "pout", "gasp", "eye_roll", "tongue_out"
    ])


# Regex pattern for explicit commands: [type:value]
COMMAND_PATTERN = re.compile(r'\[(\w+):(\w+)\]', re.IGNORECASE)


def parse_explicit_commands(text: str) -> tuple[str, list[AvatarCommand]]:
    """
    Parse explicit AI commands from text.
    
    Returns:
        Tuple of (cleaned_text, list_of_commands)
        
    Example:
        text = "[emotion:happy][gesture:wave] Hello there!"
        cleaned, commands = parse_explicit_commands(text)
        # cleaned = "Hello there!"
        # commands = [AvatarCommand('emotion', 'happy', ...), AvatarCommand('gesture', 'wave', ...)]
    """
    commands = []
    valid_commands = ExplicitCommands()
    
    for match in COMMAND_PATTERN.finditer(text):
        cmd_type = match.group(1).lower()
        cmd_value = match.group(2).lower()
        raw_text = match.group(0)
        
        # Validate command type and value
        is_valid = False
        if cmd_type == 'emotion' and cmd_value in valid_commands.EMOTIONS:
            is_valid = True
        elif cmd_type == 'gesture' and cmd_value in valid_commands.GESTURES:
            is_valid = True
        elif cmd_type == 'action' and cmd_value in valid_commands.ACTIONS:
            is_valid = True
        elif cmd_type == 'expression' and cmd_value in valid_commands.EXPRESSIONS:
            is_valid = True
        
        if is_valid:
            commands.append(AvatarCommand(cmd_type, cmd_value, raw_text))
    
    # Strip all commands from text (valid or not - don't show malformed commands)
    cleaned = COMMAND_PATTERN.sub('', text).strip()
    
    return cleaned, commands


def get_command_reference() -> str:
    """Get a reference string of all available commands for AI prompting."""
    cmds = ExplicitCommands()
    return f"""
AVATAR CONTROL COMMANDS (include in your response to control the avatar):
- Emotions: {', '.join(f'[emotion:{e}]' for e in cmds.EMOTIONS[:8])}...
- Gestures: {', '.join(f'[gesture:{g}]' for g in cmds.GESTURES[:8])}...
- Actions: {', '.join(f'[action:{a}]' for a in cmds.ACTIONS[:6])}...
- Expressions: {', '.join(f'[expression:{e}]' for e in cmds.EXPRESSIONS[:6])}...

Example: "[emotion:excited][gesture:wave] Hello! Great to see you!"
Commands are hidden from users - they only see: "Hello! Great to see you!"
"""


@dataclass
class EmotionKeywords:
    """Keywords that indicate emotions in AI responses."""
    
    HAPPY: list[str] = field(default_factory=lambda: [
        "happy", "glad", "great", "awesome", "wonderful", "fantastic",
        "excited", "love", "enjoy", "amazing", "excellent", "perfect",
        "yay", "hooray", "nice", "good", "pleased", "delighted",
        "!", "😊", "😄", "🎉", ":)", ":D", "haha", "lol"
    ])
    
    SAD: list[str] = field(default_factory=lambda: [
        "sorry", "sad", "unfortunately", "regret", "apologize",
        "can't", "cannot", "unable", "impossible", "difficult",
        "worry", "concern", "afraid", "disappointed", "miss",
        "😢", "😔", ":("
    ])
    
    SURPRISED: list[str] = field(default_factory=lambda: [
        "wow", "whoa", "amazing", "incredible", "unbelievable",
        "surprising", "unexpected", "really", "seriously",
        "oh my", "no way", "what", "!!", "😮", "😲", "🤯"
    ])
    
    THINKING: list[str] = field(default_factory=lambda: [
        "hmm", "let me think", "consider", "perhaps", "maybe",
        "possibly", "i think", "i believe", "interesting",
        "analyzing", "processing", "calculating", "...", "🤔"
    ])
    
    CONFUSED: list[str] = field(default_factory=lambda: [
        "confused", "unclear", "don't understand", "not sure",
        "what do you mean", "could you clarify", "?", "hmm",
        "🤨", "😕"
    ])


@dataclass  
class GestureKeywords:
    """Keywords that trigger gestures."""
    
    WAVE: list[str] = field(default_factory=lambda: [
        "hello", "hi", "hey", "greetings", "welcome",
        "goodbye", "bye", "see you", "farewell", "👋"
    ])
    
    NOD: list[str] = field(default_factory=lambda: [
        "yes", "correct", "right", "exactly", "indeed",
        "agree", "sure", "of course", "absolutely", "definitely"
    ])
    
    SHAKE: list[str] = field(default_factory=lambda: [
        "no", "incorrect", "wrong", "disagree", "not really",
        "i don't think so", "nope"
    ])
    
    SHRUG: list[str] = field(default_factory=lambda: [
        "i don't know", "not sure", "maybe", "perhaps",
        "hard to say", "depends", "🤷"
    ])
    
    POINT: list[str] = field(default_factory=lambda: [
        "look at", "check out", "see here", "notice",
        "here is", "this is", "that is", "👉"
    ])


class AIAvatarBridge(QObject if HAS_PYQT else object):
    """
    Bridge between AI chat and avatar expression.
    
    Analyzes AI responses in real-time and controls the avatar
    to express emotions and gestures appropriately.
    
    TWO CONTROL MODES:
    1. AUTOMATIC: Detects emotions from text via keyword matching
    2. EXPLICIT: AI uses [emotion:happy], [gesture:wave], etc. commands
    
    Explicit commands take priority over automatic detection.
    Commands are stripped from the displayed response text.
    """
    
    # Signals
    if HAS_PYQT:
        emotion_detected = pyqtSignal(str)
        gesture_triggered = pyqtSignal(str)
        action_triggered = pyqtSignal(str)
        expression_triggered = pyqtSignal(str)
        text_cleaned = pyqtSignal(str)  # Emits text with commands stripped
    
    def __init__(self, avatar=None, enable_explicit_commands: bool = True):
        if HAS_PYQT:
            super().__init__()
        
        self.avatar = avatar
        self._is_responding = False
        self._response_buffer = ""
        self._last_emotion = "neutral"
        self._gesture_cooldown = False
        self._enable_explicit = enable_explicit_commands
        
        # Track explicit commands received this response
        self._explicit_commands: list[AvatarCommand] = []
        
        # Keyword matchers for automatic detection
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
    
    def enable_explicit_commands(self, enable: bool = True):
        """Enable or disable explicit command parsing."""
        self._enable_explicit = enable
    
    # =========================================================================
    # RESPONSE LIFECYCLE
    # =========================================================================
    
    def on_response_start(self):
        """Called when AI starts generating a response."""
        self._is_responding = True
        self._response_buffer = ""
        self._last_emotion = "neutral"
        self._explicit_commands = []
        
        if self.avatar:
            self.avatar.start_talking()
    
    def on_response_chunk(self, text: str) -> str:
        """
        Called for each chunk of AI response (streaming).
        
        Analyzes text for emotions and gestures in real-time.
        If explicit commands are enabled, parses and executes them,
        returning the cleaned text with commands stripped.
        
        Args:
            text: The raw AI response chunk
            
        Returns:
            Cleaned text with explicit commands removed
        """
        if not self._is_responding:
            return text
        
        cleaned_text = text
        
        # Check for explicit commands first (if enabled)
        if self._enable_explicit:
            cleaned_text, commands = parse_explicit_commands(text)
            
            for cmd in commands:
                self._explicit_commands.append(cmd)
                self._execute_command(cmd)
            
            # Emit cleaned text signal
            if commands and HAS_PYQT:
                self.text_cleaned.emit(cleaned_text)
        
        self._response_buffer += cleaned_text
        
        # Only use automatic detection if no explicit emotion command was given
        has_explicit_emotion = any(c.command_type == 'emotion' for c in self._explicit_commands)
        
        # Analyze for emotion (check periodically, not every character)
        if not has_explicit_emotion:
            if len(self._response_buffer) % 20 == 0 or cleaned_text.endswith(('.', '!', '?')):
                emotion = self._detect_emotion(self._response_buffer)
                if emotion != self._last_emotion:
                    self._last_emotion = emotion
                    if self.avatar:
                        self.avatar.set_emotion(emotion)
                    if HAS_PYQT:
                        self.emotion_detected.emit(emotion)
        
        # Check for gestures (with cooldown) - only if no explicit gesture
        has_explicit_gesture = any(c.command_type == 'gesture' for c in self._explicit_commands)
        if not has_explicit_gesture and not self._gesture_cooldown:
            gesture = self._detect_gesture(cleaned_text)
            if gesture:
                self._trigger_gesture(gesture)
        
        return cleaned_text
    
    def _execute_command(self, cmd: AvatarCommand):
        """Execute an explicit avatar command."""
        if cmd.command_type == 'emotion':
            self._last_emotion = cmd.value
            if self.avatar:
                self.avatar.set_emotion(cmd.value)
            if HAS_PYQT:
                self.emotion_detected.emit(cmd.value)
                
        elif cmd.command_type == 'gesture':
            self._trigger_gesture(cmd.value)
            
        elif cmd.command_type == 'action':
            if self.avatar:
                # Try different action methods
                if hasattr(self.avatar, 'set_action'):
                    self.avatar.set_action(cmd.value)
                elif hasattr(self.avatar, cmd.value):
                    getattr(self.avatar, cmd.value)()
            if HAS_PYQT:
                self.action_triggered.emit(cmd.value)
                
        elif cmd.command_type == 'expression':
            if self.avatar:
                if hasattr(self.avatar, 'set_expression'):
                    self.avatar.set_expression(cmd.value)
                elif hasattr(self.avatar, cmd.value):
                    getattr(self.avatar, cmd.value)()
            if HAS_PYQT:
                self.expression_triggered.emit(cmd.value)
    
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
        """
        Detect emotion from text content.
        
        Uses the advanced SentimentAnalyzer if available, falling back to
        simple keyword matching if not.
        """
        # Use advanced sentiment analyzer if available
        if HAS_SENTIMENT_ANALYZER and analyze_for_avatar:
            try:
                expression, confidence = analyze_for_avatar(text)
                if confidence > 0.3:  # Only use if reasonably confident
                    return expression
            except Exception:
                pass  # Fall back to keyword matching
        
        # Fallback: simple keyword matching
        return self._detect_emotion_keywords(text)
    
    def _detect_emotion_keywords(self, text: str) -> str:
        """Fallback: Detect emotion using simple keyword matching."""
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
        
        Wraps the EnigmaEngine generate call to automatically
        control the avatar during generation.
        
        Args:
            engine: EnigmaEngine instance
            prompt: The user's prompt
            **kwargs: Additional args for engine.generate()
            
        Returns:
            The AI's response text (cleaned of explicit commands if enabled)
        """
        self.on_response_start()
        
        try:
            # Check if engine supports streaming
            if hasattr(engine, 'generate_stream'):
                # Streaming mode
                response_parts = []
                for chunk in engine.generate_stream(prompt, **kwargs):
                    cleaned = self.on_response_chunk(chunk)
                    response_parts.append(cleaned)
                response = ''.join(response_parts)
            else:
                # Non-streaming mode
                response = engine.generate(prompt, **kwargs)
                response = self.on_response_chunk(response)
            
            return response
            
        finally:
            self.on_response_end()
    
    def generate_with_avatar_commands(self, engine, prompt: str, 
                                       include_command_reference: bool = True,
                                       **kwargs) -> str:
        """
        Generate AI response with explicit avatar command support.
        
        This version injects the command reference into the system prompt
        so the AI knows how to control the avatar.
        
        Args:
            engine: EnigmaEngine instance
            prompt: The user's prompt
            include_command_reference: If True, adds command reference to prompt
            **kwargs: Additional args for engine.generate()
            
        Returns:
            The AI's response text (cleaned of explicit commands)
        """
        if include_command_reference:
            # Add command reference to system context
            command_ref = get_command_reference()
            enhanced_prompt = f"{command_ref}\n\nUser: {prompt}"
        else:
            enhanced_prompt = prompt
        
        return self.generate_with_avatar(engine, enhanced_prompt, **kwargs)
    
    def get_commands_for_response(self) -> list[AvatarCommand]:
        """Get all explicit commands that were processed for the current/last response."""
        return self._explicit_commands.copy()
    
    def wrap_callback(self, original_callback: Callable) -> Callable:
        """
        Wrap a streaming callback to include avatar control.
        
        The callback receives cleaned text (commands stripped).
        
        Usage:
            def my_callback(chunk):
                print(chunk, end='')
            
            wrapped = bridge.wrap_callback(my_callback)
            engine.generate_stream(prompt, callback=wrapped)
        """
        def wrapped(chunk):
            cleaned = self.on_response_chunk(chunk)
            if original_callback:
                original_callback(cleaned)
        return wrapped
    
    def wrap_callback_with_raw(self, original_callback: Callable, 
                                command_callback: Callable = None) -> Callable:
        """
        Wrap a streaming callback with both cleaned text and command notifications.
        
        Usage:
            def on_text(chunk):
                print(chunk, end='')
            
            def on_command(cmd):
                print(f"Command: {cmd.command_type}:{cmd.value}")
            
            wrapped = bridge.wrap_callback_with_raw(on_text, on_command)
        """
        prev_cmd_count = [0]  # Mutable to track in closure
        
        def wrapped(chunk):
            cleaned = self.on_response_chunk(chunk)
            
            # Check for new commands
            if command_callback:
                current_count = len(self._explicit_commands)
                if current_count > prev_cmd_count[0]:
                    for cmd in self._explicit_commands[prev_cmd_count[0]:]:
                        command_callback(cmd)
                    prev_cmd_count[0] = current_count
            
            if original_callback:
                original_callback(cleaned)
        
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

def create_avatar_bridge(avatar, enable_explicit: bool = True) -> AIAvatarBridge:
    """
    Create an AI-Avatar bridge.
    
    Args:
        avatar: The avatar instance to control
        enable_explicit: Enable explicit command parsing (default True)
    """
    return AIAvatarBridge(avatar, enable_explicit_commands=enable_explicit)


def integrate_avatar_with_chat(chat_widget, avatar) -> AvatarChatIntegration:
    """Integrate avatar with a chat widget."""
    return AvatarChatIntegration(chat_widget, avatar)


def get_avatar_command_prompt() -> str:
    """
    Get a system prompt snippet that teaches the AI to control the avatar.
    
    Add this to your system prompt when you want the AI to express itself.
    """
    return get_command_reference()


def process_ai_response(text: str, avatar=None) -> str:
    """
    Simple one-shot processing of AI response.
    
    Parses explicit commands, executes them on avatar, returns cleaned text.
    
    Args:
        text: Raw AI response (may contain [emotion:happy] etc.)
        avatar: Avatar to control (optional)
        
    Returns:
        Cleaned text with commands stripped
    """
    cleaned, commands = parse_explicit_commands(text)
    
    if avatar:
        for cmd in commands:
            if cmd.command_type == 'emotion':
                if hasattr(avatar, 'set_emotion'):
                    avatar.set_emotion(cmd.value)
            elif cmd.command_type == 'gesture':
                if hasattr(avatar, 'gesture'):
                    avatar.gesture(cmd.value)
            elif cmd.command_type == 'action':
                if hasattr(avatar, 'set_action'):
                    avatar.set_action(cmd.value)
                elif hasattr(avatar, cmd.value):
                    getattr(avatar, cmd.value)()
            elif cmd.command_type == 'expression':
                if hasattr(avatar, 'set_expression'):
                    avatar.set_expression(cmd.value)
    
    return cleaned


def list_avatar_commands() -> dict[str, list[str]]:
    """
    Get all available explicit commands.
    
    Returns:
        Dict mapping command types to available values
    """
    cmds = ExplicitCommands()
    return {
        'emotion': cmds.EMOTIONS,
        'gesture': cmds.GESTURES,
        'action': cmds.ACTIONS,
        'expression': cmds.EXPRESSIONS
    }
