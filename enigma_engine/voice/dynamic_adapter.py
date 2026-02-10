"""
Dynamic Voice Adapter

Adapts voice in real-time based on emotion and context.

Features:
- Emotional voice modulation (happy, sad, excited, serious)
- Context-aware voice changes (storytelling, teaching, casual, formal)
- Mood-based parameter adjustments
- Personality-driven voice selection

Usage:
    from enigma_engine.voice.dynamic_adapter import DynamicVoiceAdapter
    
    adapter = DynamicVoiceAdapter()
    
    # Adapt for emotion
    profile = adapter.adapt_for_emotion("happy", base_profile)
    
    # Adapt for context
    profile = adapter.adapt_for_context("storytelling", base_profile)
    
    # Adapt for personality mood
    profile = adapter.adapt_for_mood("cheerful", base_profile)
"""

from typing import Optional

from .voice_effects import VoiceEffects
from .voice_profile import VoiceProfile

try:
    from ..core.personality import AIPersonality
    PERSONALITY_AVAILABLE = True
except ImportError:
    PERSONALITY_AVAILABLE = False


class DynamicVoiceAdapter:
    """
    Dynamically adapts voice based on emotional and contextual needs.
    
    Provides real-time voice modulation without permanently changing
    the base voice profile.
    """
    
    # Emotion parameter adjustments (relative changes)
    EMOTION_ADJUSTMENTS = {
        "happy": {
            "pitch_delta": 0.15,
            "speed_delta": 0.1,
            "volume_delta": 0.05,
            "effects": ["warm", "energetic"]
        },
        "sad": {
            "pitch_delta": -0.1,
            "speed_delta": -0.15,
            "volume_delta": -0.1,
            "effects": ["calm"]
        },
        "excited": {
            "pitch_delta": 0.2,
            "speed_delta": 0.2,
            "volume_delta": 0.1,
            "effects": ["energetic"]
        },
        "serious": {
            "pitch_delta": -0.05,
            "speed_delta": -0.05,
            "volume_delta": 0.0,
            "effects": ["authoritative"]
        },
        "playful": {
            "pitch_delta": 0.1,
            "speed_delta": 0.15,
            "volume_delta": 0.05,
            "effects": ["warm"]
        },
        "concerned": {
            "pitch_delta": -0.05,
            "speed_delta": -0.1,
            "volume_delta": -0.05,
            "effects": ["warm", "calm"]
        },
        "angry": {
            "pitch_delta": 0.0,
            "speed_delta": 0.1,
            "volume_delta": 0.15,
            "effects": ["authoritative"]
        },
        "curious": {
            "pitch_delta": 0.1,
            "speed_delta": 0.05,
            "volume_delta": 0.0,
            "effects": ["warm"]
        },
        "thoughtful": {
            "pitch_delta": -0.05,
            "speed_delta": -0.15,
            "volume_delta": -0.05,
            "effects": ["calm"]
        },
        "neutral": {
            "pitch_delta": 0.0,
            "speed_delta": 0.0,
            "volume_delta": 0.0,
            "effects": []
        }
    }
    
    # Context parameter adjustments
    CONTEXT_ADJUSTMENTS = {
        "storytelling": {
            "pitch_delta": 0.05,
            "speed_delta": -0.1,
            "volume_delta": 0.0,
            "effects": ["warm"],
            "description": "Storytelling mode"
        },
        "teaching": {
            "pitch_delta": 0.0,
            "speed_delta": -0.1,
            "volume_delta": 0.05,
            "effects": ["calm"],
            "description": "Teaching mode"
        },
        "casual": {
            "pitch_delta": 0.05,
            "speed_delta": 0.05,
            "volume_delta": 0.0,
            "effects": ["warm"],
            "description": "Casual conversation"
        },
        "formal": {
            "pitch_delta": -0.05,
            "speed_delta": -0.05,
            "volume_delta": 0.0,
            "effects": ["authoritative"],
            "description": "Formal mode"
        },
        "presentation": {
            "pitch_delta": 0.0,
            "speed_delta": -0.05,
            "volume_delta": 0.1,
            "effects": ["authoritative"],
            "description": "Presentation mode"
        },
        "friendly": {
            "pitch_delta": 0.1,
            "speed_delta": 0.0,
            "volume_delta": 0.0,
            "effects": ["warm"],
            "description": "Friendly mode"
        },
        "professional": {
            "pitch_delta": 0.0,
            "speed_delta": 0.0,
            "volume_delta": 0.0,
            "effects": ["cold"],
            "description": "Professional mode"
        }
    }
    
    def __init__(self):
        """Initialize dynamic voice adapter."""
        self.effects_system = VoiceEffects()
        self.current_emotion: Optional[str] = None
        self.current_context: Optional[str] = None
    
    def adapt_for_emotion(
        self,
        emotion: str,
        base_profile: Optional[VoiceProfile] = None
    ) -> VoiceProfile:
        """
        Adapt voice for a specific emotion.
        
        Args:
            emotion: Emotion name (happy, sad, excited, serious, etc.)
            base_profile: Base voice profile (creates default if None)
            
        Returns:
            Adapted VoiceProfile
        """
        if base_profile is None:
            base_profile = VoiceProfile()
        
        emotion_lower = emotion.lower()
        
        if emotion_lower not in self.EMOTION_ADJUSTMENTS:
            # Unknown emotion, return base unchanged
            return base_profile
        
        adjustments = self.EMOTION_ADJUSTMENTS[emotion_lower]
        
        # Apply adjustments
        adapted = VoiceProfile(
            name=f"{base_profile.name}_emotional",
            pitch=self._clamp_value(
                base_profile.pitch + adjustments["pitch_delta"],
                0.5, 1.5
            ),
            speed=self._clamp_value(
                base_profile.speed + adjustments["speed_delta"],
                0.5, 1.5
            ),
            volume=self._clamp_value(
                base_profile.volume + adjustments["volume_delta"],
                0.3, 1.0
            ),
            voice=base_profile.voice,
            effects=self._merge_effects(base_profile.effects, adjustments["effects"]),
            language=base_profile.language,
            description=f"{base_profile.description} (emotion: {emotion})"
        )
        
        self.current_emotion = emotion_lower
        return adapted
    
    def adapt_for_context(
        self,
        context: str,
        base_profile: Optional[VoiceProfile] = None
    ) -> VoiceProfile:
        """
        Adapt voice for a specific context.
        
        Args:
            context: Context name (storytelling, teaching, casual, formal)
            base_profile: Base voice profile
            
        Returns:
            Adapted VoiceProfile
        """
        if base_profile is None:
            base_profile = VoiceProfile()
        
        context_lower = context.lower()
        
        if context_lower not in self.CONTEXT_ADJUSTMENTS:
            return base_profile
        
        adjustments = self.CONTEXT_ADJUSTMENTS[context_lower]
        
        adapted = VoiceProfile(
            name=f"{base_profile.name}_contextual",
            pitch=self._clamp_value(
                base_profile.pitch + adjustments["pitch_delta"],
                0.5, 1.5
            ),
            speed=self._clamp_value(
                base_profile.speed + adjustments["speed_delta"],
                0.5, 1.5
            ),
            volume=self._clamp_value(
                base_profile.volume + adjustments["volume_delta"],
                0.3, 1.0
            ),
            voice=base_profile.voice,
            effects=self._merge_effects(base_profile.effects, adjustments["effects"]),
            language=base_profile.language,
            description=adjustments.get("description", f"Context: {context}")
        )
        
        self.current_context = context_lower
        return adapted
    
    def adapt_for_mood(
        self,
        mood: str,
        base_profile: Optional[VoiceProfile] = None
    ) -> VoiceProfile:
        """
        Adapt voice based on AI's current mood.
        
        This is similar to emotion but tied to personality state.
        
        Args:
            mood: Mood from personality (happy, concerned, curious, neutral)
            base_profile: Base voice profile
            
        Returns:
            Adapted VoiceProfile
        """
        # Moods map to emotions
        mood_to_emotion = {
            "happy": "happy",
            "concerned": "concerned",
            "curious": "curious",
            "thoughtful": "thoughtful",
            "neutral": "neutral"
        }
        
        emotion = mood_to_emotion.get(mood.lower(), "neutral")
        return self.adapt_for_emotion(emotion, base_profile)
    
    def adapt_for_personality(
        self,
        personality: 'AIPersonality',
        base_profile: Optional[VoiceProfile] = None
    ) -> VoiceProfile:
        """
        Adapt voice based on personality's current mood and traits.
        
        Args:
            personality: AIPersonality object
            base_profile: Base voice profile
            
        Returns:
            Adapted VoiceProfile
        """
        if not PERSONALITY_AVAILABLE:
            return base_profile or VoiceProfile()
        
        # Start with mood-based adaptation
        adapted = self.adapt_for_mood(personality.mood, base_profile)
        
        # Further adjust based on active traits
        traits = personality.get_all_effective_traits()
        
        # High playfulness -> more energetic
        if traits.get("playfulness", 0.5) > 0.7:
            if "energetic" not in adapted.effects:
                adapted.effects.append("energetic")
        
        # High formality -> more authoritative
        if traits.get("formality", 0.5) > 0.7:
            if "authoritative" not in adapted.effects:
                adapted.effects.append("authoritative")
        
        # High empathy -> warmer
        if traits.get("empathy", 0.5) > 0.7:
            if "warm" not in adapted.effects:
                adapted.effects.append("warm")
        
        adapted.description = f"Personality-adapted ({personality.mood})"
        
        return adapted
    
    def adapt_combined(
        self,
        emotion: Optional[str] = None,
        context: Optional[str] = None,
        base_profile: Optional[VoiceProfile] = None
    ) -> VoiceProfile:
        """
        Adapt voice for both emotion and context simultaneously.
        
        Args:
            emotion: Optional emotion
            context: Optional context
            base_profile: Base voice profile
            
        Returns:
            Adapted VoiceProfile with both adjustments
        """
        if base_profile is None:
            base_profile = VoiceProfile()
        
        profile = base_profile
        
        # Apply emotion first
        if emotion:
            profile = self.adapt_for_emotion(emotion, profile)
        
        # Then context (may override some emotion changes)
        if context:
            # Get context adjustments
            context_lower = context.lower()
            if context_lower in self.CONTEXT_ADJUSTMENTS:
                ctx_adj = self.CONTEXT_ADJUSTMENTS[context_lower]
                
                # Apply on top of emotional adaptation
                profile = VoiceProfile(
                    name=f"{base_profile.name}_combined",
                    pitch=self._clamp_value(
                        profile.pitch + ctx_adj["pitch_delta"],
                        0.5, 1.5
                    ),
                    speed=self._clamp_value(
                        profile.speed + ctx_adj["speed_delta"],
                        0.5, 1.5
                    ),
                    volume=self._clamp_value(
                        profile.volume + ctx_adj["volume_delta"],
                        0.3, 1.0
                    ),
                    voice=profile.voice,
                    effects=self._merge_effects(profile.effects, ctx_adj["effects"]),
                    language=profile.language,
                    description=f"Emotion: {emotion}, Context: {context}"
                )
        
        return profile
    
    def _clamp_value(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp a value to a range."""
        return max(min_val, min(max_val, value))
    
    def _merge_effects(
        self,
        base_effects: list,
        new_effects: list
    ) -> list:
        """
        Merge effect lists, avoiding duplicates and resolving conflicts.
        """
        # Start with base effects
        merged = list(base_effects)
        
        # Add new effects that aren't already present
        for effect in new_effects:
            if effect not in merged:
                merged.append(effect)
        
        # Use effects system to resolve conflicts
        merged = self.effects_system.combine_effects(merged)
        
        return merged
    
    def get_emotion_description(self, emotion: str) -> str:
        """Get description of how emotion affects voice."""
        if emotion.lower() in self.EMOTION_ADJUSTMENTS:
            adj = self.EMOTION_ADJUSTMENTS[emotion.lower()]
            effects_str = ", ".join(adj["effects"]) if adj["effects"] else "none"
            return f"Pitch: {adj['pitch_delta']:+.2f}, Speed: {adj['speed_delta']:+.2f}, Effects: {effects_str}"
        return "Unknown emotion"
    
    def get_context_description(self, context: str) -> str:
        """Get description of how context affects voice."""
        if context.lower() in self.CONTEXT_ADJUSTMENTS:
            return self.CONTEXT_ADJUSTMENTS[context.lower()].get("description", context)
        return "Unknown context"
    
    def list_emotions(self) -> list:
        """List all supported emotions."""
        return list(self.EMOTION_ADJUSTMENTS.keys())
    
    def list_contexts(self) -> list:
        """List all supported contexts."""
        return list(self.CONTEXT_ADJUSTMENTS.keys())


# Convenience functions
def adapt_voice_for_emotion(
    emotion: str,
    base_profile: Optional[VoiceProfile] = None
) -> VoiceProfile:
    """
    Adapt voice for an emotion.
    
    Args:
        emotion: Emotion name
        base_profile: Base voice profile
        
    Returns:
        Adapted VoiceProfile
    """
    adapter = DynamicVoiceAdapter()
    return adapter.adapt_for_emotion(emotion, base_profile)


def adapt_voice_for_context(
    context: str,
    base_profile: Optional[VoiceProfile] = None
) -> VoiceProfile:
    """
    Adapt voice for a context.
    
    Args:
        context: Context name
        base_profile: Base voice profile
        
    Returns:
        Adapted VoiceProfile
    """
    adapter = DynamicVoiceAdapter()
    return adapter.adapt_for_context(context, base_profile)


def adapt_voice_for_personality(
    personality: 'AIPersonality',
    base_profile: Optional[VoiceProfile] = None
) -> VoiceProfile:
    """
    Adapt voice based on personality.
    
    Args:
        personality: AIPersonality object
        base_profile: Base voice profile
        
    Returns:
        Adapted VoiceProfile
    """
    adapter = DynamicVoiceAdapter()
    return adapter.adapt_for_personality(personality, base_profile)
