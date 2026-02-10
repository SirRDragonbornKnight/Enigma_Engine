"""
Personality Expression System

Avatar visual style adapts to match AI personality traits.
Adjusts posture, movement style, and expression tendencies.

FILE: enigma_engine/avatar/personality_expression.py
TYPE: Avatar Behavior
MAIN CLASSES: PersonalityExpression, PersonalityTraits, ExpressionStyle
"""

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PersonalityDimension(Enum):
    """Big Five personality dimensions."""
    OPENNESS = "openness"               # Creative, curious
    CONSCIENTIOUSNESS = "conscientiousness"  # Organized, disciplined
    EXTRAVERSION = "extraversion"       # Outgoing, energetic
    AGREEABLENESS = "agreeableness"     # Friendly, compassionate
    NEUROTICISM = "neuroticism"         # Anxious, moody


@dataclass
class PersonalityTraits:
    """AI personality traits (0-1 scale)."""
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.2
    
    # Additional traits
    humor: float = 0.5           # Use of humor
    formality: float = 0.5       # Formal vs casual
    enthusiasm: float = 0.5      # Energy level
    empathy: float = 0.5         # Emotional responsiveness
    confidence: float = 0.5      # Self-assurance
    
    def to_dict(self) -> dict[str, float]:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
            "humor": self.humor,
            "formality": self.formality,
            "enthusiasm": self.enthusiasm,
            "empathy": self.empathy,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, float]) -> 'PersonalityTraits':
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    @classmethod
    def preset_friendly(cls) -> 'PersonalityTraits':
        """Friendly, helpful assistant."""
        return cls(
            openness=0.7, conscientiousness=0.6, extraversion=0.7,
            agreeableness=0.9, neuroticism=0.1, humor=0.6,
            formality=0.3, enthusiasm=0.7, empathy=0.8, confidence=0.6
        )
    
    @classmethod
    def preset_professional(cls) -> 'PersonalityTraits':
        """Professional, formal assistant."""
        return cls(
            openness=0.5, conscientiousness=0.9, extraversion=0.4,
            agreeableness=0.6, neuroticism=0.2, humor=0.2,
            formality=0.9, enthusiasm=0.4, empathy=0.5, confidence=0.8
        )
    
    @classmethod
    def preset_creative(cls) -> 'PersonalityTraits':
        """Creative, imaginative assistant."""
        return cls(
            openness=0.95, conscientiousness=0.4, extraversion=0.6,
            agreeableness=0.7, neuroticism=0.3, humor=0.7,
            formality=0.2, enthusiasm=0.8, empathy=0.7, confidence=0.7
        )
    
    @classmethod
    def preset_calm(cls) -> 'PersonalityTraits':
        """Calm, measured assistant."""
        return cls(
            openness=0.5, conscientiousness=0.7, extraversion=0.3,
            agreeableness=0.7, neuroticism=0.05, humor=0.3,
            formality=0.5, enthusiasm=0.3, empathy=0.7, confidence=0.7
        )


@dataclass
class ExpressionStyle:
    """How personality manifests in avatar behavior."""
    # Animation speeds (0.5 = slower, 1.0 = normal, 1.5 = faster)
    gesture_speed: float = 1.0
    blink_frequency: float = 1.0
    micro_movement_intensity: float = 0.5
    
    # Posture
    posture_upright: float = 0.7  # 0 = slouched, 1 = very upright
    lean_tendency: float = 0.0    # -1 = lean back, 1 = lean forward
    head_tilt_range: float = 0.3  # How much the head tilts
    
    # Expression tendencies
    smile_tendency: float = 0.5   # How often to smile
    eyebrow_expressiveness: float = 0.5
    eye_contact_duration: float = 0.7
    
    # Movement patterns
    fidget_frequency: float = 0.2  # How often to fidget
    hand_gesture_frequency: float = 0.5
    nod_frequency: float = 0.4
    
    def to_dict(self) -> dict[str, float]:
        return {
            "gesture_speed": self.gesture_speed,
            "blink_frequency": self.blink_frequency,
            "micro_movement_intensity": self.micro_movement_intensity,
            "posture_upright": self.posture_upright,
            "lean_tendency": self.lean_tendency,
            "head_tilt_range": self.head_tilt_range,
            "smile_tendency": self.smile_tendency,
            "eyebrow_expressiveness": self.eyebrow_expressiveness,
            "eye_contact_duration": self.eye_contact_duration,
            "fidget_frequency": self.fidget_frequency,
            "hand_gesture_frequency": self.hand_gesture_frequency,
            "nod_frequency": self.nod_frequency
        }


class PersonalityExpression:
    """Maps personality traits to avatar expression style."""
    
    def __init__(self, traits: Optional[PersonalityTraits] = None):
        """
        Initialize with personality traits.
        
        Args:
            traits: PersonalityTraits (uses defaults if None)
        """
        self._traits = traits or PersonalityTraits()
        self._style: Optional[ExpressionStyle] = None
        self._emotion_biases: dict[str, float] = {}
        
    def set_traits(self, traits: PersonalityTraits):
        """Update personality traits."""
        self._traits = traits
        self._style = None  # Force recalculation
        self._emotion_biases.clear()
        
    def get_expression_style(self) -> ExpressionStyle:
        """
        Calculate expression style from personality traits.
        
        Returns:
            ExpressionStyle derived from current traits
        """
        if self._style is not None:
            return self._style
            
        t = self._traits
        
        # Calculate animation speeds
        # Extroverts and high-enthusiasm = faster, neurotic = faster/irregular
        base_speed = 0.8 + (t.extraversion * 0.2) + (t.enthusiasm * 0.2)
        
        # Calculate posture
        # Confident + conscientious = upright
        upright = 0.4 + (t.confidence * 0.3) + (t.conscientiousness * 0.2) + (t.formality * 0.1)
        
        # Lean tendency
        # Agreeable + empathetic = lean forward
        lean = (t.agreeableness - 0.5) * 0.4 + (t.empathy - 0.5) * 0.3
        
        # Expression tendencies
        # Agreeable + extravert = more smiling
        smile = 0.2 + (t.agreeableness * 0.3) + (t.extraversion * 0.2) + (t.humor * 0.2)
        
        # Eye contact
        # Confident + extravert = more eye contact
        eye_contact = 0.3 + (t.confidence * 0.3) + (t.extraversion * 0.2)
        # Neurotic reduces eye contact
        eye_contact -= t.neuroticism * 0.2
        
        # Fidgeting
        # Neurotic + not conscientious = more fidgeting
        fidget = 0.1 + (t.neuroticism * 0.4) + ((1 - t.conscientiousness) * 0.2)
        
        # Hand gestures
        # Extravert + open + enthusiastic = more gestures
        gestures = 0.2 + (t.extraversion * 0.3) + (t.openness * 0.2) + (t.enthusiasm * 0.2)
        # Formal reduces gestures
        gestures -= t.formality * 0.2
        
        # Nodding
        # Agreeable + empathetic = more nodding
        nod = 0.2 + (t.agreeableness * 0.3) + (t.empathy * 0.3)
        
        self._style = ExpressionStyle(
            gesture_speed=self._clamp(base_speed),
            blink_frequency=self._clamp(0.8 + random.uniform(-0.2, 0.2)),
            micro_movement_intensity=self._clamp(0.3 + (t.neuroticism * 0.3) + (t.enthusiasm * 0.2)),
            posture_upright=self._clamp(upright),
            lean_tendency=self._clamp(lean, -1, 1),
            head_tilt_range=self._clamp(0.2 + (t.openness * 0.2) + (t.extraversion * 0.1)),
            smile_tendency=self._clamp(smile),
            eyebrow_expressiveness=self._clamp(0.3 + (t.extraversion * 0.3) + (t.enthusiasm * 0.2)),
            eye_contact_duration=self._clamp(eye_contact),
            fidget_frequency=self._clamp(fidget),
            hand_gesture_frequency=self._clamp(gestures),
            nod_frequency=self._clamp(nod)
        )
        
        return self._style
    
    def get_emotion_bias(self, emotion: str) -> float:
        """
        Get personality-based bias for an emotion.
        
        Some personalities naturally tend toward certain emotions.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Bias value (0-1, higher = more likely)
        """
        if emotion in self._emotion_biases:
            return self._emotion_biases[emotion]
            
        t = self._traits
        
        # Calculate biases
        biases = {
            "happy": 0.3 + (t.extraversion * 0.3) + (t.agreeableness * 0.2),
            "sad": 0.1 + (t.neuroticism * 0.3) + (t.empathy * 0.2),
            "angry": 0.1 + ((1 - t.agreeableness) * 0.2) + (t.neuroticism * 0.2),
            "surprised": 0.2 + (t.openness * 0.3) + (t.enthusiasm * 0.2),
            "fearful": 0.1 + (t.neuroticism * 0.4),
            "disgusted": 0.1 + ((1 - t.agreeableness) * 0.2),
            "neutral": 0.4 + (t.formality * 0.2) + ((1 - t.extraversion) * 0.2),
            "thinking": 0.3 + (t.openness * 0.2) + (t.conscientiousness * 0.2),
            "excited": 0.2 + (t.enthusiasm * 0.4) + (t.extraversion * 0.2),
            "curious": 0.3 + (t.openness * 0.4)
        }
        
        # Store and return
        for emo, bias in biases.items():
            self._emotion_biases[emo] = self._clamp(bias)
            
        return self._emotion_biases.get(emotion, 0.5)
    
    def adjust_emotion_intensity(self, emotion: str, intensity: float) -> float:
        """
        Adjust emotion intensity based on personality.
        
        Args:
            emotion: Emotion name
            intensity: Base intensity (0-1)
            
        Returns:
            Adjusted intensity
        """
        t = self._traits
        
        # Base multiplier from extraversion
        multiplier = 0.7 + (t.extraversion * 0.4)
        
        # Adjust for specific emotions
        if emotion == "happy" and t.humor > 0.5:
            multiplier += 0.2
        elif emotion in ("sad", "fearful") and t.neuroticism > 0.5:
            multiplier += 0.2
        elif emotion == "excited" and t.enthusiasm > 0.5:
            multiplier += 0.3
            
        # Formal personalities dampen expression
        multiplier -= t.formality * 0.2
        
        return self._clamp(intensity * multiplier)
    
    def get_idle_behavior_weights(self) -> dict[str, float]:
        """
        Get weights for idle behavior selection.
        
        Returns:
            Dict of behavior -> weight
        """
        t = self._traits
        
        return {
            "look_around": 0.3 + (t.openness * 0.3),
            "fidget": 0.1 + (t.neuroticism * 0.4),
            "smile": 0.2 + (t.agreeableness * 0.3),
            "stretch": 0.1 + ((1 - t.formality) * 0.2),
            "blink": 0.5,  # Always blink
            "micro_movement": 0.3 + (t.enthusiasm * 0.2),
            "check_appearance": 0.1 + (t.conscientiousness * 0.2),
            "deep_breath": 0.2 + ((1 - t.neuroticism) * 0.2),
            "head_tilt": 0.2 + (t.extraversion * 0.2)
        }
    
    def suggest_default_expression(self) -> str:
        """
        Suggest a default/resting expression based on personality.
        
        Returns:
            Expression name
        """
        t = self._traits
        
        if t.agreeableness > 0.7 and t.extraversion > 0.5:
            return "friendly_smile"
        elif t.formality > 0.7:
            return "professional_neutral"
        elif t.confidence > 0.7:
            return "confident_neutral"
        elif t.neuroticism > 0.6:
            return "slightly_anxious"
        else:
            return "relaxed_neutral"
    
    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp value to range."""
        return max(min_val, min(max_val, value))
    
    @property
    def traits(self) -> PersonalityTraits:
        return self._traits


# Preset personalities
PERSONALITY_PRESETS: dict[str, PersonalityTraits] = {
    "friendly": PersonalityTraits.preset_friendly(),
    "professional": PersonalityTraits.preset_professional(),
    "creative": PersonalityTraits.preset_creative(),
    "calm": PersonalityTraits.preset_calm(),
    "default": PersonalityTraits()
}


# Singleton
_personality_expression: Optional[PersonalityExpression] = None


def get_personality_expression(traits: PersonalityTraits = None) -> PersonalityExpression:
    """Get or create personality expression singleton."""
    global _personality_expression
    if _personality_expression is None:
        _personality_expression = PersonalityExpression(traits)
    elif traits:
        _personality_expression.set_traits(traits)
    return _personality_expression


def apply_personality_preset(preset_name: str) -> PersonalityExpression:
    """Apply a personality preset."""
    traits = PERSONALITY_PRESETS.get(preset_name, PersonalityTraits())
    return get_personality_expression(traits)


__all__ = [
    'PersonalityExpression',
    'PersonalityTraits',
    'PersonalityDimension',
    'ExpressionStyle',
    'PERSONALITY_PRESETS',
    'get_personality_expression',
    'apply_personality_preset'
]
