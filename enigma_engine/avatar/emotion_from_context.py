"""
Emotion from Context System

AI automatically sets avatar emotion based on conversation context.
Analyzes AI responses to determine appropriate emotional expression.

FILE: enigma_engine/avatar/emotion_from_context.py
TYPE: Avatar Emotion System
MAIN CLASSES: EmotionDetector, ContextEmotionSync, EmotionState
"""

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class Emotion(Enum):
    """Available avatar emotions."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    CONFUSED = "confused"
    THINKING = "thinking"
    EXCITED = "excited"
    CONCERNED = "concerned"
    EMPATHETIC = "empathetic"
    FOCUSED = "focused"
    PLAYFUL = "playful"
    PROUD = "proud"


@dataclass
class EmotionState:
    """Current emotional state with confidence."""
    emotion: Emotion
    confidence: float = 1.0
    intensity: float = 0.7  # 0.0 to 1.0
    duration: float = 3.0   # seconds to hold
    timestamp: float = field(default_factory=time.time)
    source: str = "context"  # context, user, explicit


# Emotion detection patterns
EMOTION_PATTERNS: dict[Emotion, list[str]] = {
    Emotion.HAPPY: [
        r'\b(happy|glad|pleased|delighted|wonderful|great|awesome|fantastic|'
        r'excellent|perfect|love|enjoy|excited|thrilled|yay|hooray|congratulations)\b',
        r'(:\)|😊|😄|🎉|❤️|💖)',
        r'\b(haha|lol|lmao)\b'
    ],
    Emotion.SAD: [
        r'\b(sad|sorry|unfortunate|regret|apologize|disappointed|condolences|'
        r'sympathy|miss|lost|grief|mourn)\b',
        r'(:\(|😢|😭|💔)',
        r'\b(unfortunately|sadly)\b'
    ],
    Emotion.SURPRISED: [
        r'\b(wow|whoa|amazing|incredible|unbelievable|shocking|astonishing|'
        r'mind-blowing|extraordinary)\b',
        r'(😮|😲|🤯|!!)',
        r'\b(oh my|really\?|seriously\?)\b'
    ],
    Emotion.CONFUSED: [
        r'\b(confused|unclear|uncertain|puzzled|perplexed|strange|odd|weird)\b',
        r'(🤔|\?\?)',
        r'\b(hmm|huh|what\?|not sure)\b'
    ],
    Emotion.THINKING: [
        r'\b(thinking|consider|analyze|evaluate|examine|reflect|ponder|'
        r'let me think|interesting question)\b',
        r'(🤔|💭)',
        r'\b(well|hmm|let\'s see)\b'
    ],
    Emotion.EXCITED: [
        r'\b(excited|thrilled|can\'t wait|amazing|incredible|fantastic|'
        r'brilliant|outstanding)\b',
        r'(🎉|🚀|⚡|✨|!{2,})',
        r'\b(yes!|awesome!|let\'s go)\b'
    ],
    Emotion.CONCERNED: [
        r'\b(concern|worried|careful|caution|warning|danger|risk|'
        r'please be|note that|important)\b',
        r'(⚠️|😟|🙁)',
        r'\b(however|but|although|be aware)\b'
    ],
    Emotion.EMPATHETIC: [
        r'\b(understand|feel|relate|sympathize|empathize|here for you|'
        r'support|care|listen)\b',
        r'(💕|🤗|❤️)',
        r'\b(i hear you|that must be|i can imagine)\b'
    ],
    Emotion.FOCUSED: [
        r'\b(focus|concentrate|specifically|exactly|precisely|detail|'
        r'step by step|carefully)\b',
        r'(🎯|📝|🔍)',
        r'\b(first|then|next|finally)\b.*\b(first|then|next|finally)\b'
    ],
    Emotion.PLAYFUL: [
        r'\b(fun|joke|kidding|playful|silly|humor|laugh|tease)\b',
        r'(😄|😜|🎮|😏)',
        r'\b(hehe|teehee|just kidding)\b'
    ],
    Emotion.PROUD: [
        r'\b(proud|accomplished|achieved|succeeded|well done|good job|'
        r'impressive|excellent work)\b',
        r'(🏆|⭐|👏|💪)',
        r'\b(great job|nice work|you did it)\b'
    ]
}

# Emotion transition preferences (current -> next is natural?)
EMOTION_TRANSITIONS: dict[Emotion, list[Emotion]] = {
    Emotion.NEUTRAL: list(Emotion),  # Can go to any
    Emotion.HAPPY: [Emotion.EXCITED, Emotion.PLAYFUL, Emotion.PROUD, Emotion.NEUTRAL],
    Emotion.SAD: [Emotion.EMPATHETIC, Emotion.CONCERNED, Emotion.NEUTRAL],
    Emotion.SURPRISED: [Emotion.EXCITED, Emotion.CONFUSED, Emotion.HAPPY, Emotion.NEUTRAL],
    Emotion.CONFUSED: [Emotion.THINKING, Emotion.NEUTRAL, Emotion.FOCUSED],
    Emotion.THINKING: [Emotion.FOCUSED, Emotion.NEUTRAL, Emotion.HAPPY],
    Emotion.EXCITED: [Emotion.HAPPY, Emotion.PLAYFUL, Emotion.NEUTRAL],
    Emotion.CONCERNED: [Emotion.EMPATHETIC, Emotion.THINKING, Emotion.NEUTRAL],
    Emotion.EMPATHETIC: [Emotion.SAD, Emotion.CONCERNED, Emotion.NEUTRAL],
    Emotion.FOCUSED: [Emotion.THINKING, Emotion.PROUD, Emotion.NEUTRAL],
    Emotion.PLAYFUL: [Emotion.HAPPY, Emotion.EXCITED, Emotion.NEUTRAL],
    Emotion.PROUD: [Emotion.HAPPY, Emotion.EXCITED, Emotion.NEUTRAL]
}


class EmotionDetector:
    """Detects emotion from text content."""
    
    def __init__(self):
        self._compiled_patterns: dict[Emotion, list[re.Pattern]] = {}
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        for emotion, patterns in EMOTION_PATTERNS.items():
            self._compiled_patterns[emotion] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
            
    def detect(self, text: str) -> EmotionState:
        """
        Detect emotion from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            EmotionState with detected emotion
        """
        if not text:
            return EmotionState(Emotion.NEUTRAL, confidence=1.0)
            
        scores: dict[Emotion, float] = {e: 0.0 for e in Emotion}
        
        # Score each emotion based on pattern matches
        for emotion, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                scores[emotion] += len(matches) * 0.3  # 0.3 per match
                
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            for emotion in scores:
                scores[emotion] /= total
        else:
            scores[Emotion.NEUTRAL] = 1.0
            
        # Get highest scoring emotion
        best_emotion = max(scores, key=scores.get)
        confidence = scores[best_emotion]
        
        # Default to neutral if confidence is too low
        if confidence < 0.2:
            best_emotion = Emotion.NEUTRAL
            confidence = 1.0
            
        # Calculate intensity based on punctuation and emphasis
        intensity = self._calculate_intensity(text)
        
        return EmotionState(
            emotion=best_emotion,
            confidence=confidence,
            intensity=intensity
        )
        
    def _calculate_intensity(self, text: str) -> float:
        """Calculate emotional intensity from text markers."""
        intensity = 0.5  # Base
        
        # Exclamation marks increase intensity
        exclamations = text.count('!')
        intensity += min(exclamations * 0.1, 0.3)
        
        # All caps words increase intensity
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        intensity += min(caps_words * 0.05, 0.2)
        
        # Multiple punctuation increases intensity
        multi_punct = len(re.findall(r'[!?]{2,}', text))
        intensity += min(multi_punct * 0.1, 0.2)
        
        return min(intensity, 1.0)


class ContextEmotionSync:
    """Synchronizes avatar emotion with conversation context."""
    
    def __init__(self, avatar_controller=None):
        """
        Initialize emotion sync.
        
        Args:
            avatar_controller: Avatar controller instance (optional)
        """
        self._avatar = avatar_controller
        self._detector = EmotionDetector()
        self._current_state = EmotionState(Emotion.NEUTRAL)
        self._history: list[EmotionState] = []
        self._max_history = 10
        self._callbacks: list[Callable[[EmotionState], None]] = []
        self._enabled = True
        self._lock = threading.Lock()
        self._transition_timer: Optional[threading.Timer] = None
        
    def set_avatar(self, avatar_controller):
        """Set the avatar controller."""
        self._avatar = avatar_controller
        
    def process_response(self, ai_response: str) -> EmotionState:
        """
        Process AI response and update emotion.
        
        Args:
            ai_response: The AI's response text
            
        Returns:
            New emotion state
        """
        if not self._enabled:
            return self._current_state
            
        with self._lock:
            # Detect emotion from response
            new_state = self._detector.detect(ai_response)
            
            # Check if transition is natural
            if not self._is_natural_transition(self._current_state.emotion, new_state.emotion):
                # Smooth transition through neutral
                self._transition_through_neutral(new_state)
            else:
                self._set_emotion(new_state)
                
            return self._current_state
            
    def process_user_input(self, user_text: str) -> EmotionState:
        """
        Process user input for emotional context.
        
        Args:
            user_text: The user's message
            
        Returns:
            Detected emotion state
        """
        # Detect user's apparent emotion
        user_emotion = self._detector.detect(user_text)
        user_emotion.source = "user"
        
        # Potentially mirror empathetic response
        if user_emotion.emotion in [Emotion.SAD, Emotion.CONCERNED]:
            empathetic = EmotionState(
                emotion=Emotion.EMPATHETIC,
                confidence=user_emotion.confidence * 0.8,
                intensity=user_emotion.intensity * 0.7
            )
            self._set_emotion(empathetic)
            
        return user_emotion
        
    def _is_natural_transition(self, from_emotion: Emotion, to_emotion: Emotion) -> bool:
        """Check if emotion transition is natural."""
        if from_emotion == to_emotion:
            return True
        natural_next = EMOTION_TRANSITIONS.get(from_emotion, [])
        return to_emotion in natural_next
        
    def _transition_through_neutral(self, target: EmotionState):
        """Transition through neutral state."""
        # First go to neutral
        neutral = EmotionState(Emotion.NEUTRAL, intensity=0.3, duration=0.5)
        self._set_emotion(neutral)
        
        # Schedule transition to target
        if self._transition_timer:
            self._transition_timer.cancel()
        self._transition_timer = threading.Timer(0.5, lambda: self._set_emotion(target))
        self._transition_timer.start()
        
    def _set_emotion(self, state: EmotionState):
        """Set current emotion state."""
        self._current_state = state
        
        # Add to history
        self._history.append(state)
        if len(self._history) > self._max_history:
            self._history.pop(0)
            
        # Update avatar if available
        if self._avatar and hasattr(self._avatar, 'set_emotion'):
            try:
                self._avatar.set_emotion(state.emotion.value, intensity=state.intensity)
            except Exception as e:
                logger.warning(f"Failed to set avatar emotion: {e}")
                
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Emotion callback error: {e}")
                
    def on_emotion_change(self, callback: Callable[[EmotionState], None]):
        """Register callback for emotion changes."""
        self._callbacks.append(callback)
        
    def set_enabled(self, enabled: bool):
        """Enable or disable automatic emotion sync."""
        self._enabled = enabled
        
    def get_current_emotion(self) -> EmotionState:
        """Get current emotion state."""
        return self._current_state
        
    def get_emotion_history(self) -> list[EmotionState]:
        """Get recent emotion history."""
        return self._history.copy()
        
    def set_emotion_explicit(self, emotion: Emotion, intensity: float = 0.7):
        """
        Explicitly set emotion (overrides context detection).
        
        Args:
            emotion: Emotion to set
            intensity: Emotion intensity (0.0 to 1.0)
        """
        state = EmotionState(
            emotion=emotion,
            confidence=1.0,
            intensity=intensity,
            source="explicit"
        )
        self._set_emotion(state)
        
    def reset_to_neutral(self):
        """Reset emotion to neutral."""
        self._set_emotion(EmotionState(Emotion.NEUTRAL))


# Singleton instance
_emotion_sync: Optional[ContextEmotionSync] = None


def get_emotion_sync(avatar_controller=None) -> ContextEmotionSync:
    """Get or create the emotion sync singleton."""
    global _emotion_sync
    if _emotion_sync is None:
        _emotion_sync = ContextEmotionSync(avatar_controller)
    elif avatar_controller is not None:
        _emotion_sync.set_avatar(avatar_controller)
    return _emotion_sync


def detect_emotion(text: str) -> tuple[str, float]:
    """
    Convenience function to detect emotion from text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Tuple of (emotion_name, confidence)
    """
    detector = EmotionDetector()
    state = detector.detect(text)
    return state.emotion.value, state.confidence


__all__ = [
    'Emotion',
    'EmotionState',
    'EmotionDetector',
    'ContextEmotionSync',
    'get_emotion_sync',
    'detect_emotion',
    'EMOTION_PATTERNS',
    'EMOTION_TRANSITIONS'
]
