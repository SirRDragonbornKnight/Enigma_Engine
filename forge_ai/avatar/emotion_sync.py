"""
Emotion-Expression Synchronization

Automatically updates avatar expression based on AI mood and emotional state.
"""

import logging
import threading
import time
from threading import Lock
from typing import Optional, Callable

from ..core.personality import AIPersonality

logger = logging.getLogger(__name__)


class EmotionExpressionSync:
    """
    Automatically updates avatar expression based on AI mood.
    
    Monitors the AI's personality mood and updates avatar expressions
    to match the emotional state.
    """
    
    # Mapping from AI mood to avatar expression
    MOOD_TO_EXPRESSION = {
        "happy": "happy",
        "excited": "excited",
        "curious": "thinking",
        "thoughtful": "thinking",
        "concerned": "worried",
        "worried": "sad",
        "sad": "sad",
        "surprised": "surprised",
        "confused": "confused",
        "neutral": "neutral",
    }
    
    # Emotion keywords in text → expressions
    TEXT_EMOTION_KEYWORDS = {
        "happy": ["happy", "joy", "great", "wonderful", "excellent", "yay", "hooray"],
        "excited": ["excited", "amazing", "awesome", "wow", "incredible"],
        "sad": ["sad", "unfortunately", "sorry", "regret", "disappointed"],
        "thinking": ["think", "consider", "wonder", "perhaps", "maybe"],
        "confused": ["confused", "unclear", "puzzled", "not sure", "don't understand"],
        "surprised": ["surprised", "unexpected", "wow", "oh", "whoa"],
    }
    
    def __init__(self, avatar, personality: Optional[AIPersonality] = None):
        """
        Initialize emotion sync.
        
        Args:
            avatar: AvatarController instance
            personality: AIPersonality to monitor
        """
        self.avatar = avatar
        self.personality = personality
        self._sync_thread: Optional[threading.Thread] = None
        self._running = False
        self._state_lock = Lock()  # Thread safety for state changes
        self._last_mood: Optional[str] = None
        self._callbacks: list = []
    
    def start_sync(self):
        """Start background mood monitoring."""
        if self._running:
            return
        
        self._running = True
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        logger.info("Started mood monitoring")
    
    def stop_sync(self):
        """Stop background mood monitoring."""
        self._running = False
        if self._sync_thread:
            self._sync_thread.join(timeout=1.0)
        logger.info("Stopped mood monitoring")
    
    def _sync_loop(self):
        """Background loop to monitor mood changes."""
        while self._running:
            if self.personality:
                current_mood = self.personality.mood
                
                with self._state_lock:
                    if current_mood != self._last_mood:
                        self.on_mood_change(self._last_mood, current_mood)
                        self._last_mood = current_mood
            
            time.sleep(0.5)  # Check every 0.5 seconds
    
    def on_mood_change(self, old_mood: Optional[str], new_mood: str):
        """
        Called when AI mood changes - update expression.
        
        Args:
            old_mood: Previous mood
            new_mood: New mood
        """
        expression = self.MOOD_TO_EXPRESSION.get(new_mood.lower(), "neutral")
        
        if self.avatar.is_enabled:
            self.avatar.set_expression(expression)
            logger.debug(f"Mood changed: {old_mood} → {new_mood}, expression: {expression}")
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(old_mood, new_mood, expression)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
    
    def on_speaking(self, text: str):
        """
        Called when AI is speaking - trigger speaking animation with emotional context.
        
        Args:
            text: Text being spoken
        """
        # Detect emotion from text
        emotion = self.detect_emotion_from_text(text)
        
        # Set appropriate expression if emotion detected
        if emotion and emotion != "neutral":
            expression = self.MOOD_TO_EXPRESSION.get(emotion, "neutral")
            if self.avatar.is_enabled:
                self.avatar.set_expression(expression)
        
        # Trigger speaking animation
        if self.avatar.is_enabled:
            self.avatar.speak(text, animate=True)
    
    def detect_emotion_from_text(self, text: str) -> str:
        """
        Analyze AI response text for emotion cues.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected emotion (or "neutral")
        """
        text_lower = text.lower()
        
        # Count matches for each emotion
        emotion_scores = {}
        
        for emotion, keywords in self.TEXT_EMOTION_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Return emotion with highest score
        if emotion_scores:
            return max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        return "neutral"
    
    def force_expression(self, expression: str):
        """
        Manually set expression (overrides mood sync temporarily).
        
        Args:
            expression: Expression to set
        """
        if self.avatar.is_enabled:
            self.avatar.set_expression(expression)
    
    def register_callback(self, callback: Callable):
        """
        Register callback for mood changes.
        
        Callback signature: callback(old_mood, new_mood, expression)
        
        Args:
            callback: Function to call on mood change
        """
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable):
        """
        Unregister a callback.
        
        Args:
            callback: Function to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
