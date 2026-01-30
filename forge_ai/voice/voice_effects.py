"""
Enhanced Voice Effects System

Expand beyond basic "robotic" effect to a comprehensive effects system.

Effects:
- robotic (levels 1-3)
- whisper
- echo
- warm / cold
- energetic / calm
- authoritative
- Custom effect combinations

Usage:
    from forge_ai.voice.voice_effects import VoiceEffects, apply_effect
    
    effects = VoiceEffects()
    
    # Apply single effect
    text = effects.apply_effect("Hello world", "robotic", level=2)
    
    # Apply multiple effects
    text = effects.apply_effects("Hello", ["warm", "calm"])
    
    # Get effect for emotion
    effect = effects.effect_for_emotion("happy")  # Returns "energetic"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EffectConfig:
    """Configuration for a voice effect."""
    
    name: str
    description: str
    text_transform: bool = True  # Does it modify text?
    audio_transform: bool = False  # Does it need audio processing?
    parameters: dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class VoiceEffects:
    """
    Enhanced voice effects system.
    
    Provides text-based and audio-based transformations for voice output.
    """
    
    # Effect definitions
    EFFECTS = {
        "robotic": EffectConfig(
            name="robotic",
            description="Mechanical, robotic voice with pauses",
            text_transform=True,
            parameters={"levels": 3}
        ),
        "whisper": EffectConfig(
            name="whisper",
            description="Soft, quiet whisper effect",
            text_transform=True,
            audio_transform=True,
            parameters={"volume_reduction": 0.5}
        ),
        "echo": EffectConfig(
            name="echo",
            description="Echo/reverb effect",
            text_transform=True,
            audio_transform=True,
            parameters={"repeats": 1}
        ),
        "warm": EffectConfig(
            name="warm",
            description="Warm, friendly tone",
            text_transform=True,
            parameters={"softeners": ["...", "~"]}
        ),
        "cold": EffectConfig(
            name="cold",
            description="Cold, clinical tone",
            text_transform=True,
            parameters={"formal": True}
        ),
        "energetic": EffectConfig(
            name="energetic",
            description="Energetic, enthusiastic voice",
            text_transform=True,
            parameters={"emphasis": True}
        ),
        "calm": EffectConfig(
            name="calm",
            description="Calm, soothing voice",
            text_transform=True,
            parameters={"pauses": True}
        ),
        "authoritative": EffectConfig(
            name="authoritative",
            description="Authoritative, commanding voice",
            text_transform=True,
            parameters={"emphasis": True, "formal": True}
        ),
    }
    
    # Emotion to effect mapping
    EMOTION_EFFECTS = {
        "happy": "energetic",
        "sad": "calm",
        "excited": "energetic",
        "serious": "authoritative",
        "playful": "warm",
        "concerned": "warm",
        "neutral": None,
        "angry": "authoritative",
        "curious": "warm",
        "thoughtful": "calm"
    }
    
    def __init__(self):
        """Initialize voice effects system."""
        self.active_effects: List[str] = []
    
    def apply_effect(
        self,
        text: str,
        effect: str,
        level: int = 1
    ) -> str:
        """
        Apply a single effect to text.
        
        Args:
            text: Text to transform
            effect: Effect name
            level: Effect intensity (1-3)
            
        Returns:
            Transformed text
        """
        if effect not in self.EFFECTS:
            return text  # Unknown effect, return unchanged
        
        effect_config = self.EFFECTS[effect]
        
        if not effect_config.text_transform:
            return text  # No text transformation needed
        
        # Apply effect-specific transformation
        if effect == "robotic":
            return self._apply_robotic(text, level)
        elif effect == "whisper":
            return self._apply_whisper(text)
        elif effect == "echo":
            return self._apply_echo(text, level)
        elif effect == "warm":
            return self._apply_warm(text)
        elif effect == "cold":
            return self._apply_cold(text)
        elif effect == "energetic":
            return self._apply_energetic(text)
        elif effect == "calm":
            return self._apply_calm(text)
        elif effect == "authoritative":
            return self._apply_authoritative(text)
        
        return text
    
    def apply_effects(
        self,
        text: str,
        effects: List[str],
        levels: Optional[Dict[str, int]] = None
    ) -> str:
        """
        Apply multiple effects to text.
        
        Args:
            text: Text to transform
            effects: List of effect names
            levels: Optional dict mapping effect names to levels
            
        Returns:
            Transformed text
        """
        if not effects:
            return text
        
        levels = levels or {}
        result = text
        
        for effect in effects:
            level = levels.get(effect, 1)
            result = self.apply_effect(result, effect, level)
        
        return result
    
    def _apply_robotic(self, text: str, level: int) -> str:
        """Apply robotic effect with varying intensity."""
        if level == 1:
            # Mild: Small pauses at sentence ends
            text = text.replace(". ", "... ")
            text = text.replace("? ", "?... ")
            text = text.replace("! ", "!... ")
        elif level == 2:
            # Medium: Pauses at commas too
            text = text.replace(". ", "... ")
            text = text.replace("? ", "?... ")
            text = text.replace("! ", "!... ")
            text = text.replace(", ", ",. ")
        elif level >= 3:
            # Strong: Pauses between words
            words = text.split()
            text = ". ".join(words)
        
        return text
    
    def _apply_whisper(self, text: str) -> str:
        """Apply whisper effect (mostly audio-based)."""
        # Text hint: lowercase and softer punctuation
        text = text.lower()
        text = text.replace("!", "...")
        text = text.replace("?", "...")
        return f"*{text}*"  # Indicate whisper
    
    def _apply_echo(self, text: str, level: int) -> str:
        """Apply echo effect."""
        # Add subtle word repetition for echo feel
        if level >= 2:
            words = text.split()
            if len(words) > 0:
                # Repeat last word softly
                last_word = words[-1].rstrip(".!?")
                text = f"{text}... {last_word.lower()}..."
        return text
    
    def _apply_warm(self, text: str) -> str:
        """Apply warm, friendly effect."""
        # Make punctuation softer
        text = text.replace(".", "...")
        # Could add warmth indicators
        return text
    
    def _apply_cold(self, text: str) -> str:
        """Apply cold, clinical effect."""
        # Remove emotional punctuation
        text = text.replace("!", ".")
        text = text.replace("...", ".")
        return text
    
    def _apply_energetic(self, text: str) -> str:
        """Apply energetic, enthusiastic effect."""
        # Add emphasis to important words
        # Keep existing exclamations, add energy
        if not text.endswith("!"):
            text = text.rstrip(".") + "!"
        return text
    
    def _apply_calm(self, text: str) -> str:
        """Apply calm, soothing effect."""
        # Add gentle pauses
        text = text.replace(".", "...")
        text = text.replace("!", ".")
        # Slow down with pauses
        text = text.replace(", ", ",... ")
        return text
    
    def _apply_authoritative(self, text: str) -> str:
        """Apply authoritative effect."""
        import re
        
        # Make statements more definitive
        text = text.replace("...", ".")
        
        # Remove hedging with proper word boundaries
        hedging_patterns = [
            r'\b(maybe|perhaps|possibly|might)\b',
        ]
        
        for pattern in hedging_patterns:
            # Remove hedging word and clean up extra spaces
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Clean up space before punctuation
        text = re.sub(r'\s+([,.])', r'\1', text)
        
        return text.strip()
    
    def effect_for_emotion(self, emotion: str) -> Optional[str]:
        """
        Get appropriate effect for an emotion.
        
        Args:
            emotion: Emotion name (happy, sad, excited, etc.)
            
        Returns:
            Effect name or None
        """
        return self.EMOTION_EFFECTS.get(emotion.lower())
    
    def effect_for_context(self, context: str) -> Optional[str]:
        """
        Get appropriate effect for a context.
        
        Args:
            context: Context (storytelling, teaching, casual, formal)
            
        Returns:
            Effect name or None
        """
        context_map = {
            "storytelling": "warm",
            "teaching": "calm",
            "casual": "warm",
            "formal": "authoritative",
            "presentation": "authoritative",
            "friendly": "warm",
            "professional": "cold"
        }
        return context_map.get(context.lower())
    
    def combine_effects(
        self,
        effects: List[str],
        priorities: Optional[Dict[str, int]] = None
    ) -> List[str]:
        """
        Combine compatible effects, resolving conflicts.
        
        Some effects conflict (e.g., warm + cold).
        This method resolves conflicts based on priorities.
        
        Args:
            effects: List of effects to combine
            priorities: Optional priority values (higher = preferred)
            
        Returns:
            Filtered list of compatible effects
        """
        if not effects:
            return []
        
        priorities = priorities or {}
        
        # Define conflicting effect pairs
        conflicts = [
            ("warm", "cold"),
            ("energetic", "calm"),
            ("whisper", "authoritative")
        ]
        
        # Resolve conflicts
        result = list(effects)
        
        for effect1, effect2 in conflicts:
            if effect1 in result and effect2 in result:
                # Both present - keep higher priority
                priority1 = priorities.get(effect1, 0)
                priority2 = priorities.get(effect2, 0)
                
                if priority1 > priority2:
                    result.remove(effect2)
                elif priority2 > priority1:
                    result.remove(effect1)
                else:
                    # Equal priority - remove both
                    result.remove(effect1)
                    result.remove(effect2)
        
        return result
    
    def get_effect_description(self, effect: str) -> str:
        """Get human-readable description of an effect."""
        if effect in self.EFFECTS:
            return self.EFFECTS[effect].description
        return "Unknown effect"
    
    def list_available_effects(self) -> List[str]:
        """List all available effect names."""
        return list(self.EFFECTS.keys())
    
    def get_effect_config(self, effect: str) -> Optional[EffectConfig]:
        """Get configuration for an effect."""
        return self.EFFECTS.get(effect)


# Convenience functions
def apply_effect(text: str, effect: str, level: int = 1) -> str:
    """
    Apply a voice effect to text.
    
    Args:
        text: Text to transform
        effect: Effect name
        level: Effect intensity (1-3)
        
    Returns:
        Transformed text
    """
    effects_system = VoiceEffects()
    return effects_system.apply_effect(text, effect, level)


def apply_effects(text: str, effects: List[str]) -> str:
    """
    Apply multiple effects to text.
    
    Args:
        text: Text to transform
        effects: List of effect names
        
    Returns:
        Transformed text
    """
    effects_system = VoiceEffects()
    return effects_system.apply_effects(text, effects)


def effect_for_emotion(emotion: str) -> Optional[str]:
    """Get effect name for an emotion."""
    effects_system = VoiceEffects()
    return effects_system.effect_for_emotion(emotion)


def effect_for_context(context: str) -> Optional[str]:
    """Get effect name for a context."""
    effects_system = VoiceEffects()
    return effects_system.effect_for_context(context)
