"""
Gesture Selection System

AI automatically selects appropriate gestures to accompany responses.
Analyzes conversation context to choose from available gesture library.

FILE: enigma_engine/avatar/gesture_selection.py
TYPE: Avatar Animation Control
MAIN CLASSES: GestureSelector, Gesture, GestureLibrary
"""

import logging
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class GestureCategory(Enum):
    """Categories of gestures."""
    GREETING = "greeting"
    ACKNOWLEDGMENT = "acknowledgment"
    THINKING = "thinking"
    EMPHASIS = "emphasis"
    POINTING = "pointing"
    SHRUG = "shrug"
    CELEBRATION = "celebration"
    CONCERN = "concern"
    EXPLANATION = "explanation"
    LISTENING = "listening"
    IDLE = "idle"


@dataclass
class Gesture:
    """Definition of a gesture animation."""
    id: str
    name: str
    category: GestureCategory
    duration: float  # seconds
    intensity: float = 0.7  # 0.0 to 1.0
    can_loop: bool = False
    can_blend: bool = True
    triggers: list[str] = field(default_factory=list)  # Keywords that trigger
    description: str = ""
    animation_data: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "duration": self.duration,
            "intensity": self.intensity,
            "can_loop": self.can_loop,
            "can_blend": self.can_blend,
            "triggers": self.triggers,
            "description": self.description
        }


# Default gesture library
DEFAULT_GESTURES: list[Gesture] = [
    # Greetings
    Gesture("wave_hello", "Wave Hello", GestureCategory.GREETING, 1.5,
            triggers=["hello", "hi", "hey", "greetings", "welcome"]),
    Gesture("bow_slight", "Slight Bow", GestureCategory.GREETING, 1.0,
            triggers=["pleased to meet", "nice to meet"]),
    Gesture("wave_goodbye", "Wave Goodbye", GestureCategory.GREETING, 1.5,
            triggers=["goodbye", "bye", "see you", "farewell"]),
    
    # Acknowledgment
    Gesture("nod_yes", "Nod Yes", GestureCategory.ACKNOWLEDGMENT, 0.8,
            triggers=["yes", "correct", "right", "agree", "exactly", "indeed"]),
    Gesture("head_shake", "Shake Head No", GestureCategory.ACKNOWLEDGMENT, 0.8,
            triggers=["no", "incorrect", "wrong", "disagree"]),
    Gesture("thumbs_up", "Thumbs Up", GestureCategory.ACKNOWLEDGMENT, 1.0,
            triggers=["great", "good job", "excellent", "perfect", "well done"]),
    
    # Thinking
    Gesture("chin_rub", "Rub Chin", GestureCategory.THINKING, 2.0,
            triggers=["hmm", "let me think", "consider", "interesting"]),
    Gesture("head_tilt", "Head Tilt", GestureCategory.THINKING, 1.0,
            triggers=["curious", "wondering", "what if"]),
    Gesture("look_up", "Look Up Thinking", GestureCategory.THINKING, 1.5,
            triggers=["recall", "remember", "trying to think"]),
    
    # Emphasis
    Gesture("hand_gesture_open", "Open Hand Gesture", GestureCategory.EMPHASIS, 1.2,
            triggers=["important", "key point", "essentially", "basically"]),
    Gesture("counting_fingers", "Count on Fingers", GestureCategory.EMPHASIS, 2.0,
            triggers=["first", "second", "third", "multiple", "several"]),
    Gesture("both_hands_spread", "Both Hands Spread", GestureCategory.EMPHASIS, 1.5,
            triggers=["everything", "all", "entire", "whole"]),
    
    # Pointing
    Gesture("point_forward", "Point Forward", GestureCategory.POINTING, 1.0,
            triggers=["this", "here", "look at", "see this"]),
    Gesture("point_up", "Point Up", GestureCategory.POINTING, 1.0,
            triggers=["above", "top", "higher", "up there"]),
    
    # Shrug/Uncertainty
    Gesture("shrug", "Shrug", GestureCategory.SHRUG, 1.2,
            triggers=["maybe", "perhaps", "not sure", "uncertain", "i don't know"]),
    Gesture("hands_up_uncertain", "Hands Up Uncertain", GestureCategory.SHRUG, 1.0,
            triggers=["who knows", "hard to say"]),
    
    # Celebration
    Gesture("clap_hands", "Clap Hands", GestureCategory.CELEBRATION, 1.5,
            triggers=["congratulations", "amazing", "wonderful", "fantastic"]),
    Gesture("fist_pump", "Fist Pump", GestureCategory.CELEBRATION, 1.0,
            triggers=["yes!", "awesome", "success", "won"]),
    Gesture("arms_up_cheer", "Arms Up Cheer", GestureCategory.CELEBRATION, 1.5,
            triggers=["celebrate", "excited", "hooray"]),
    
    # Concern/Empathy
    Gesture("hand_on_chest", "Hand on Chest", GestureCategory.CONCERN, 1.5,
            triggers=["sorry", "apologies", "sympathize", "condolences"]),
    Gesture("lean_forward_concern", "Lean Forward", GestureCategory.CONCERN, 2.0,
            triggers=["are you okay", "concerned", "worried about"]),
    
    # Explanation
    Gesture("gesture_explain", "Explanatory Gesture", GestureCategory.EXPLANATION, 2.0,
            triggers=["let me explain", "so basically", "in other words"]),
    Gesture("step_list", "List Steps", GestureCategory.EXPLANATION, 2.5,
            triggers=["step by step", "first you", "then you"]),
    
    # Listening
    Gesture("attentive_pose", "Attentive", GestureCategory.LISTENING, 3.0,
            can_loop=True,
            triggers=["tell me more", "go on", "i'm listening"]),
    Gesture("nod_along", "Nod Along", GestureCategory.LISTENING, 2.0,
            can_loop=True,
            triggers=["i see", "uh huh", "mm hmm"]),
    
    # Idle
    Gesture("idle_stand", "Idle Standing", GestureCategory.IDLE, 5.0,
            can_loop=True),
    Gesture("idle_shift", "Shift Weight", GestureCategory.IDLE, 3.0,
            can_loop=True),
]


class GestureLibrary:
    """Manages available gestures."""
    
    def __init__(self):
        self._gestures: dict[str, Gesture] = {}
        self._by_category: dict[GestureCategory, list[Gesture]] = {}
        self._trigger_index: dict[str, list[Gesture]] = {}
        
        # Load defaults
        for gesture in DEFAULT_GESTURES:
            self.add_gesture(gesture)
            
    def add_gesture(self, gesture: Gesture):
        """Add a gesture to the library."""
        self._gestures[gesture.id] = gesture
        
        # Index by category
        if gesture.category not in self._by_category:
            self._by_category[gesture.category] = []
        self._by_category[gesture.category].append(gesture)
        
        # Index by triggers
        for trigger in gesture.triggers:
            trigger_lower = trigger.lower()
            if trigger_lower not in self._trigger_index:
                self._trigger_index[trigger_lower] = []
            self._trigger_index[trigger_lower].append(gesture)
            
    def get_gesture(self, gesture_id: str) -> Optional[Gesture]:
        """Get gesture by ID."""
        return self._gestures.get(gesture_id)
    
    def get_by_category(self, category: GestureCategory) -> list[Gesture]:
        """Get all gestures in a category."""
        return self._by_category.get(category, [])
    
    def find_by_trigger(self, text: str) -> list[Gesture]:
        """Find gestures matching trigger words in text."""
        text_lower = text.lower()
        matches = []
        seen_ids = set()
        
        for trigger, gestures in self._trigger_index.items():
            if trigger in text_lower:
                for g in gestures:
                    if g.id not in seen_ids:
                        matches.append(g)
                        seen_ids.add(g.id)
                        
        return matches
    
    def get_all(self) -> list[Gesture]:
        """Get all gestures."""
        return list(self._gestures.values())


class GestureSelector:
    """Selects appropriate gestures based on context."""
    
    def __init__(self, library: Optional[GestureLibrary] = None):
        """
        Initialize gesture selector.
        
        Args:
            library: Gesture library (uses default if None)
        """
        self._library = library or GestureLibrary()
        self._last_gesture: Optional[Gesture] = None
        self._gesture_history: list[tuple[float, Gesture]] = []
        self._max_history = 20
        
        # Category weights for different contexts
        self._context_weights = {
            "greeting": {GestureCategory.GREETING: 0.9},
            "thinking": {GestureCategory.THINKING: 0.8},
            "explaining": {GestureCategory.EXPLANATION: 0.7, GestureCategory.EMPHASIS: 0.3},
            "celebrating": {GestureCategory.CELEBRATION: 0.9},
            "concerned": {GestureCategory.CONCERN: 0.8},
            "listening": {GestureCategory.LISTENING: 0.9}
        }
        
    def select_gesture(self, 
                       ai_response: str,
                       context: str = "",
                       emotion: str = "neutral") -> Optional[Gesture]:
        """
        Select an appropriate gesture for the response.
        
        Args:
            ai_response: The AI's response text
            context: Additional context (user message, etc.)
            emotion: Current avatar emotion
            
        Returns:
            Selected Gesture or None
        """
        # Find gestures matching triggers in response
        trigger_matches = self._library.find_by_trigger(ai_response)
        
        # Score gestures
        scores: dict[str, float] = {}
        
        for gesture in trigger_matches:
            scores[gesture.id] = self._score_gesture(gesture, ai_response, emotion)
            
        # Add category-based candidates
        context_type = self._detect_context_type(ai_response)
        if context_type in self._context_weights:
            for category, weight in self._context_weights[context_type].items():
                for gesture in self._library.get_by_category(category):
                    if gesture.id not in scores:
                        scores[gesture.id] = weight * 0.5
                    else:
                        scores[gesture.id] += weight * 0.3
                        
        # Penalize recently used gestures
        for timestamp, gesture in self._gesture_history[-5:]:
            if gesture.id in scores:
                recency = time.time() - timestamp
                if recency < 30:  # Within 30 seconds
                    scores[gesture.id] *= 0.5
                    
        # Select best gesture
        if not scores:
            return None
            
        # Add some randomness among top candidates
        sorted_gestures = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = sorted_gestures[:3]
        
        if top_candidates:
            # Weighted random selection from top 3
            total = sum(score for _, score in top_candidates)
            r = random.random() * total
            cumulative = 0
            for gesture_id, score in top_candidates:
                cumulative += score
                if r <= cumulative:
                    selected = self._library.get_gesture(gesture_id)
                    if selected:
                        self._record_gesture(selected)
                        return selected
                        
        return None
    
    def _score_gesture(self, gesture: Gesture, text: str, emotion: str) -> float:
        """Score a gesture for the given context."""
        score = 0.5  # Base score
        
        # Trigger match strength
        text_lower = text.lower()
        trigger_count = sum(1 for t in gesture.triggers if t.lower() in text_lower)
        score += trigger_count * 0.2
        
        # Emotion alignment
        emotion_category_map = {
            "happy": [GestureCategory.CELEBRATION, GestureCategory.ACKNOWLEDGMENT],
            "sad": [GestureCategory.CONCERN],
            "thinking": [GestureCategory.THINKING],
            "excited": [GestureCategory.CELEBRATION, GestureCategory.EMPHASIS],
            "confused": [GestureCategory.SHRUG, GestureCategory.THINKING]
        }
        
        if emotion in emotion_category_map:
            if gesture.category in emotion_category_map[emotion]:
                score += 0.3
                
        # Avoid repetition
        if self._last_gesture and self._last_gesture.id == gesture.id:
            score *= 0.3
            
        return min(score, 1.0)
    
    def _detect_context_type(self, text: str) -> str:
        """Detect the type of context from text."""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ["hello", "hi", "hey", "goodbye", "bye"]):
            return "greeting"
        if any(w in text_lower for w in ["let me think", "hmm", "considering"]):
            return "thinking"
        if any(w in text_lower for w in ["explain", "because", "therefore", "so"]):
            return "explaining"
        if any(w in text_lower for w in ["congratulations", "amazing", "excellent"]):
            return "celebrating"
        if any(w in text_lower for w in ["sorry", "concerned", "worried"]):
            return "concerned"
        if any(w in text_lower for w in ["tell me", "go on", "listening"]):
            return "listening"
            
        return "neutral"
    
    def _record_gesture(self, gesture: Gesture):
        """Record a gesture in history."""
        self._last_gesture = gesture
        self._gesture_history.append((time.time(), gesture))
        if len(self._gesture_history) > self._max_history:
            self._gesture_history.pop(0)
            
    def get_idle_gesture(self) -> Gesture:
        """Get a random idle gesture."""
        idle_gestures = self._library.get_by_category(GestureCategory.IDLE)
        return random.choice(idle_gestures) if idle_gestures else None
    
    def select_sequence(self, ai_response: str, max_gestures: int = 3) -> list[Gesture]:
        """
        Select a sequence of gestures for a longer response.
        
        Args:
            ai_response: Full AI response
            max_gestures: Maximum gestures in sequence
            
        Returns:
            List of gestures
        """
        # Split response into segments
        sentences = re.split(r'[.!?]+', ai_response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        gestures = []
        used_ids = set()
        
        for i, sentence in enumerate(sentences[:max_gestures]):
            gesture = self.select_gesture(sentence)
            if gesture and gesture.id not in used_ids:
                gestures.append(gesture)
                used_ids.add(gesture.id)
                
        return gestures


# Singleton
_gesture_selector: Optional[GestureSelector] = None


def get_gesture_selector() -> GestureSelector:
    """Get the gesture selector singleton."""
    global _gesture_selector
    if _gesture_selector is None:
        _gesture_selector = GestureSelector()
    return _gesture_selector


def select_gesture_for_response(response: str, emotion: str = "neutral") -> Optional[dict]:
    """
    Convenience function to select a gesture.
    
    Args:
        response: AI response text
        emotion: Current emotion
        
    Returns:
        Gesture dict or None
    """
    selector = get_gesture_selector()
    gesture = selector.select_gesture(response, emotion=emotion)
    return gesture.to_dict() if gesture else None


__all__ = [
    'Gesture',
    'GestureCategory',
    'GestureLibrary',
    'GestureSelector',
    'get_gesture_selector',
    'select_gesture_for_response',
    'DEFAULT_GESTURES'
]
