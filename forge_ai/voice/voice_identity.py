"""
AI Voice Self-Discovery System

Allows AI to experiment with and discover its own voice identity.

Features:
- AI discovers voice that "feels right" based on personality
- Parse natural language voice descriptions
- Learn from user feedback on voice choices
- Evolve voice identity over time

Usage:
    from forge_ai.voice.voice_identity import AIVoiceIdentity
    from forge_ai.core.personality import load_personality
    
    identity = AIVoiceIdentity()
    personality = load_personality("my_model")
    
    # AI discovers its voice
    voice_profile = identity.discover_voice(personality)
    
    # AI describes desired voice
    voice_profile = identity.describe_desired_voice(
        "I want a warm, slow, confident voice"
    )
    
    # Learn from feedback
    identity.learn_from_feedback("The voice feels too fast", voice_profile)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .voice_profile import VoiceProfile, PROFILES_DIR
from ..config import CONFIG

try:
    from ..core.personality import AIPersonality, PersonalityTraits
    PERSONALITY_AVAILABLE = True
except ImportError:
    PERSONALITY_AVAILABLE = False


@dataclass
class VoiceExperiment:
    """Record of voice experimentation."""
    
    profile: Dict[str, Any]
    feedback_score: float = 0.0
    user_feedback: Optional[str] = None
    iteration: int = 0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class AIVoiceIdentity:
    """
    AI discovers and evolves its own voice identity.
    
    The AI experiments with different voice parameters and learns
    what "feels right" based on its personality and user feedback.
    """
    
    def __init__(self):
        """Initialize voice identity system."""
        self.experiments: List[VoiceExperiment] = []
        self.current_identity: Optional[VoiceProfile] = None
        self.feedback_history: List[Dict[str, Any]] = []
        
    def discover_voice(
        self,
        personality: 'AIPersonality',
        num_experiments: int = 5,
        base_voice: str = "default"
    ) -> VoiceProfile:
        """
        AI experiments with different voice settings to find what feels right.
        
        Process:
        1. Generate base voice from personality
        2. Create variations by adjusting parameters
        3. "Simulate" which one feels most aligned with personality
        4. Return the best match
        
        Args:
            personality: AIPersonality object
            num_experiments: Number of variations to try
            base_voice: Base voice type to start from
            
        Returns:
            VoiceProfile that best matches the AI's personality
        """
        if not PERSONALITY_AVAILABLE:
            # Fallback to basic profile
            return VoiceProfile(
                name="discovered_voice",
                description="Voice discovered without personality module"
            )
        
        # Generate base voice from personality
        from .voice_generator import AIVoiceGenerator
        generator = AIVoiceGenerator()
        base_profile = generator.generate_voice_from_personality(personality, base_voice)
        
        # Create variations by experimenting with parameters
        candidates = [base_profile]
        traits = personality.traits
        
        for i in range(num_experiments - 1):
            # Create variation by adjusting parameters
            variation = VoiceProfile(
                name=f"experiment_{i}",
                pitch=base_profile.pitch + random.uniform(-0.1, 0.1),
                speed=base_profile.speed + random.uniform(-0.15, 0.15),
                volume=base_profile.volume + random.uniform(-0.1, 0.1),
                voice=base_profile.voice,
                effects=base_profile.effects.copy(),
                description=f"Experimental variation {i}"
            )
            candidates.append(variation)
        
        # Score each candidate based on personality alignment
        best_profile = base_profile
        best_score = 0.0
        
        for candidate in candidates:
            score = self._score_voice_alignment(candidate, traits)
            
            # Record experiment
            experiment = VoiceExperiment(
                profile=asdict(candidate),
                feedback_score=score,
                iteration=len(self.experiments)
            )
            self.experiments.append(experiment)
            
            if score > best_score:
                best_score = score
                best_profile = candidate
        
        # Finalize discovered voice
        best_profile.name = f"{personality.model_name}_discovered"
        best_profile.description = f"AI-discovered voice (confidence: {best_score:.2f})"
        
        self.current_identity = best_profile
        return best_profile
    
    def _score_voice_alignment(
        self,
        profile: VoiceProfile,
        traits: 'PersonalityTraits'
    ) -> float:
        """
        Score how well a voice profile aligns with personality traits.
        
        Args:
            profile: Voice profile to score
            traits: Personality traits to match against
            
        Returns:
            Alignment score (0.0 to 1.0)
        """
        score = 0.0
        total_weight = 0.0
        
        # Pitch alignment: higher playfulness/humor -> prefer higher pitch
        expected_pitch = 1.0 + (traits.playfulness - 0.5) * 0.4 + (traits.humor_level - 0.5) * 0.3
        pitch_diff = abs(profile.pitch - expected_pitch)
        pitch_score = max(0, 1.0 - pitch_diff)
        score += pitch_score * 2.0
        total_weight += 2.0
        
        # Speed alignment: formality -> slower, playfulness -> faster
        expected_speed = 1.0 - (traits.formality - 0.5) * 0.3 + (traits.playfulness - 0.5) * 0.2
        speed_diff = abs(profile.speed - expected_speed)
        speed_score = max(0, 1.0 - speed_diff)
        score += speed_score * 1.5
        total_weight += 1.5
        
        # Volume alignment: confidence -> louder
        expected_volume = 0.85 + (traits.confidence - 0.5) * 0.2
        volume_diff = abs(profile.volume - expected_volume)
        volume_score = max(0, 1.0 - volume_diff * 2)
        score += volume_score * 1.0
        total_weight += 1.0
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def describe_desired_voice(self, description: str) -> VoiceProfile:
        """
        Parse natural language description into voice parameters.
        
        Understands terms like:
        - Pitch: "high", "low", "deep", "light"
        - Speed: "fast", "slow", "quick", "measured", "deliberate"
        - Qualities: "warm", "cold", "energetic", "calm", "robotic", "authoritative"
        
        Args:
            description: Natural language voice description
            
        Returns:
            VoiceProfile matching the description
        """
        desc_lower = description.lower()
        
        # Start with defaults
        pitch = 1.0
        speed = 1.0
        volume = 0.85
        voice = "default"
        effects = []
        
        # Parse pitch
        if any(word in desc_lower for word in ["low", "deep", "bass"]):
            pitch = 0.7
        elif any(word in desc_lower for word in ["high", "light", "bright"]):
            pitch = 1.3
        
        # Parse speed
        if any(word in desc_lower for word in ["slow", "measured", "deliberate", "careful"]):
            speed = 0.8
        elif any(word in desc_lower for word in ["fast", "quick", "rapid", "energetic"]):
            speed = 1.2
        
        # Parse volume
        if any(word in desc_lower for word in ["quiet", "soft", "gentle"]):
            volume = 0.6
        elif any(word in desc_lower for word in ["loud", "strong", "powerful"]):
            volume = 1.0
        
        # Parse voice type
        if any(word in desc_lower for word in ["female", "woman", "feminine"]):
            voice = "female"
        elif any(word in desc_lower for word in ["male", "man", "masculine"]):
            voice = "male"
        
        # Parse effects/qualities
        if any(word in desc_lower for word in ["robotic", "robot", "mechanical"]):
            effects.append("robotic")
        
        if any(word in desc_lower for word in ["warm", "friendly", "gentle"]):
            effects.append("warm")
        
        if any(word in desc_lower for word in ["cold", "clinical", "detached"]):
            effects.append("cold")
        
        if any(word in desc_lower for word in ["energetic", "excited", "enthusiastic"]):
            effects.append("energetic")
            speed = max(speed, 1.1)
        
        if any(word in desc_lower for word in ["calm", "peaceful", "relaxed"]):
            effects.append("calm")
            speed = min(speed, 0.9)
        
        if any(word in desc_lower for word in ["confident", "authoritative", "commanding"]):
            effects.append("authoritative")
            pitch = min(pitch, 0.95)
            volume = max(volume, 0.9)
        
        if any(word in desc_lower for word in ["whisper", "whispering"]):
            effects.append("whisper")
            volume = 0.5
        
        # Create profile
        profile = VoiceProfile(
            name="described_voice",
            pitch=pitch,
            speed=speed,
            volume=volume,
            voice=voice,
            effects=effects,
            description=f"Voice from description: '{description}'"
        )
        
        self.current_identity = profile
        return profile
    
    def learn_from_feedback(
        self,
        feedback: str,
        current_profile: Optional[VoiceProfile] = None
    ) -> VoiceProfile:
        """
        Adjust voice based on user feedback.
        
        Understands feedback like:
        - "Too fast" / "Speak slower"
        - "Too quiet" / "Louder please"
        - "Too high pitched" / "Lower your voice"
        - "More energy" / "Too robotic"
        
        Args:
            feedback: User feedback text
            current_profile: Current voice profile (uses last identity if None)
            
        Returns:
            Adjusted VoiceProfile
        """
        if current_profile is None:
            current_profile = self.current_identity
        
        if current_profile is None:
            # No profile to adjust, create default
            current_profile = VoiceProfile()
        
        # Record feedback
        self.feedback_history.append({
            "feedback": feedback,
            "profile_before": asdict(current_profile),
            "timestamp": datetime.now().isoformat()
        })
        
        # Parse feedback and adjust
        feedback_lower = feedback.lower()
        
        # Create adjusted profile
        adjusted = VoiceProfile(
            name=current_profile.name,
            pitch=current_profile.pitch,
            speed=current_profile.speed,
            volume=current_profile.volume,
            voice=current_profile.voice,
            effects=current_profile.effects.copy(),
            description=current_profile.description
        )
        
        # Adjust speed
        if any(word in feedback_lower for word in ["too fast", "slow down", "slower"]):
            adjusted.speed = max(0.5, current_profile.speed - 0.15)
        elif any(word in feedback_lower for word in ["too slow", "faster", "speed up"]):
            adjusted.speed = min(1.5, current_profile.speed + 0.15)
        
        # Adjust volume
        if any(word in feedback_lower for word in ["too quiet", "louder", "can't hear"]):
            adjusted.volume = min(1.0, current_profile.volume + 0.15)
        elif any(word in feedback_lower for word in ["too loud", "quieter", "softer"]):
            adjusted.volume = max(0.3, current_profile.volume - 0.15)
        
        # Adjust pitch
        if any(word in feedback_lower for word in ["too high", "lower pitch", "deeper"]):
            adjusted.pitch = max(0.5, current_profile.pitch - 0.1)
        elif any(word in feedback_lower for word in ["too low", "higher pitch", "lighter"]):
            adjusted.pitch = min(1.5, current_profile.pitch + 0.1)
        
        # Adjust effects
        if any(word in feedback_lower for word in ["too robotic", "less robotic", "more natural"]):
            if "robotic" in adjusted.effects:
                adjusted.effects.remove("robotic")
        elif any(word in feedback_lower for word in ["more robotic", "more mechanical"]):
            if "robotic" not in adjusted.effects:
                adjusted.effects.append("robotic")
        
        if any(word in feedback_lower for word in ["more energy", "more enthusiastic"]):
            if "energetic" not in adjusted.effects:
                adjusted.effects.append("energetic")
            adjusted.speed = min(1.3, adjusted.speed + 0.1)
        
        if any(word in feedback_lower for word in ["calmer", "more relaxed", "less energetic"]):
            if "energetic" in adjusted.effects:
                adjusted.effects.remove("energetic")
            if "calm" not in adjusted.effects:
                adjusted.effects.append("calm")
        
        adjusted.description = f"Adjusted from feedback: {feedback[:50]}"
        self.current_identity = adjusted
        
        return adjusted
    
    def save_identity(self, model_name: str, directory: Optional[Path] = None) -> Path:
        """
        Save discovered voice identity and experiments.
        
        Args:
            model_name: Model name for organization
            directory: Optional directory (default: models/{model_name}/voice/)
            
        Returns:
            Path to saved identity file
        """
        if directory is None:
            models_dir = Path(CONFIG["models_dir"])
            directory = models_dir / model_name / "voice"
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save current identity profile
        if self.current_identity:
            # Save profile in the same directory as voice_identity
            profiles_dir = directory / "profiles"
            profiles_dir.mkdir(parents=True, exist_ok=True)
            self.current_identity.save(profiles_dir)
        
        # Save discovery history
        identity_file = directory / "voice_identity.json"
        data = {
            "current_identity": asdict(self.current_identity) if self.current_identity else None,
            "experiments": [asdict(exp) for exp in self.experiments],
            "feedback_history": self.feedback_history
        }
        
        with open(identity_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return identity_file
    
    def load_identity(self, model_name: str, directory: Optional[Path] = None) -> bool:
        """
        Load saved voice identity.
        
        Args:
            model_name: Model name
            directory: Optional directory
            
        Returns:
            True if loaded successfully
        """
        if directory is None:
            models_dir = Path(CONFIG["models_dir"])
            directory = models_dir / model_name / "voice"
        
        directory = Path(directory)
        identity_file = directory / "voice_identity.json"
        
        if not identity_file.exists():
            return False
        
        try:
            with open(identity_file, 'r') as f:
                data = json.load(f)
            
            if data.get("current_identity"):
                self.current_identity = VoiceProfile(**data["current_identity"])
            
            self.experiments = [
                VoiceExperiment(**exp) for exp in data.get("experiments", [])
            ]
            
            self.feedback_history = data.get("feedback_history", [])
            
            return True
        except Exception as e:
            print(f"Error loading voice identity: {e}")
            return False


# Convenience functions
def discover_voice(
    personality: 'AIPersonality',
    num_experiments: int = 5
) -> VoiceProfile:
    """
    AI discovers its voice based on personality.
    
    Args:
        personality: AIPersonality object
        num_experiments: Number of variations to try
        
    Returns:
        Discovered VoiceProfile
    """
    identity = AIVoiceIdentity()
    return identity.discover_voice(personality, num_experiments)


def describe_voice(description: str) -> VoiceProfile:
    """
    Create voice from natural language description.
    
    Args:
        description: Voice description (e.g., "warm, slow, confident")
        
    Returns:
        VoiceProfile matching description
    """
    identity = AIVoiceIdentity()
    return identity.describe_desired_voice(description)


def adjust_voice_from_feedback(
    feedback: str,
    current_profile: VoiceProfile
) -> VoiceProfile:
    """
    Adjust voice based on user feedback.
    
    Args:
        feedback: User feedback text
        current_profile: Current voice profile
        
    Returns:
        Adjusted VoiceProfile
    """
    identity = AIVoiceIdentity()
    return identity.learn_from_feedback(feedback, current_profile)
