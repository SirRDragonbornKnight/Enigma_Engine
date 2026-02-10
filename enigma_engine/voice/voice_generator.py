"""
================================================================================
🔊 AI VOICE GENERATOR - THE VOICE HALL
================================================================================

AI can create and evolve its own voice based on personality traits!
Generate unique voices that match your AI's character.

📍 FILE: enigma_engine/voice/voice_generator.py
🏷️ TYPE: Voice Synthesis & Profile Generation
🎯 MAIN CLASSES: AIVoiceGenerator, VoiceEvolution

┌─────────────────────────────────────────────────────────────────────────────┐
│  PERSONALITY → VOICE MAPPING:                                               │
│                                                                             │
│  High confidence  → Lower pitch, slower speed                              │
│  High playfulness → Varied pitch, faster speed                             │
│  High formality   → Neutral pitch, measured pace                           │
│  High empathy     → Warmer tone (softer volume)                            │
│  High humor       → Higher pitch, faster speed                             │
└─────────────────────────────────────────────────────────────────────────────┘

🎙️ VOICE OPTIONS:
    1. AI-Generated: AI picks parameters based on personality
    2. User-Provided: User uploads voice samples
    3. Clone: Clone a voice from audio samples (if supported)

🔗 CONNECTED FILES:
    → USES:      enigma_engine/voice/voice_profile.py (VoiceProfile storage)
    → USES:      enigma_engine/core/personality.py (AIPersonality traits)
    ← USED BY:   enigma_engine/gui/tabs/audio_tab.py (TTS tab)
    ← USED BY:   enigma_engine/avatar/controller.py (avatar speech)

📖 USAGE:
    from enigma_engine.voice.voice_generator import AIVoiceGenerator
    from enigma_engine.core.personality import load_personality
    
    generator = AIVoiceGenerator()
    personality = load_personality("my_model")
    
    # Generate voice from personality
    voice_profile = generator.generate_voice_from_personality(personality)
    
    # Or create from samples
    voice_profile = generator.create_from_samples(["sample1.wav", "sample2.wav"])

📖 SEE ALSO:
    • enigma_engine/voice/tts_simple.py   - Simple text-to-speech
    • enigma_engine/voice/listener.py    - Speech-to-text input
    • enigma_engine/voice/lip_sync.py    - Sync avatar lips to speech
    • data/voice_profiles/          - Saved voice configurations
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import CONFIG
from .voice_profile import PROFILES_DIR, VoiceProfile

logger = logging.getLogger(__name__)

try:
    from ..core.personality import AIPersonality
    PERSONALITY_AVAILABLE = True
except ImportError:
    PERSONALITY_AVAILABLE = False


@dataclass
class VoiceEvolution:
    """Track how voice changes over time."""
    
    version: int = 1
    changes: List[Dict[str, Any]] = None
    base_profile: Optional[str] = None
    
    def __post_init__(self):
        if self.changes is None:
            self.changes = []


class AIVoiceGenerator:
    """
    AI voice generator that creates and evolves voices.
    
    Features:
    - Generate voice from personality traits
    - Create voice from audio samples
    - Evolve voice as personality changes
    - Save/load voice configurations
    """
    
    def __init__(self):
        """Initialize voice generator."""
        self.evolution_history: Dict[str, VoiceEvolution] = {}
    
    def generate_voice_from_personality(
        self, 
        personality: AIPersonality,
        base_voice: str = "default"
    ) -> VoiceProfile:
        """
        Create voice profile matching personality traits.
        
        Maps personality traits to voice parameters:
        - Higher confidence → lower pitch, slower speed
        - Higher playfulness → varied pitch, faster speed
        - Higher formality → neutral pitch, measured pace
        - Higher empathy → warmer tone (softer volume)
        - Higher humor → higher pitch, faster speed
        
        Args:
            personality: AIPersonality object
            base_voice: Base voice type ("male", "female", "default")
        
        Returns:
            VoiceProfile configured for the personality
        """
        if not PERSONALITY_AVAILABLE:
            # Fallback to default
            return VoiceProfile(name=personality.model_name if hasattr(personality, 'model_name') else "default")
        
        traits = personality.traits
        
        # Calculate voice parameters from traits
        # Pitch: confidence and playfulness influence
        # Low confidence + high playfulness = higher pitch
        # High confidence + low playfulness = lower pitch
        pitch = 1.0
        pitch += (traits.playfulness - 0.5) * 0.4  # ±0.2
        pitch += (traits.humor_level - 0.5) * 0.3  # ±0.15
        pitch -= (traits.confidence - 0.5) * 0.4   # ±0.2 (inverse)
        pitch -= (traits.formality - 0.5) * 0.2    # ±0.1 (inverse)
        pitch = max(0.6, min(1.4, pitch))  # Clamp to reasonable range
        
        # Speed: formality and verbosity influence
        # High formality = slower, more measured
        # High verbosity = might speak faster to fit more in
        speed = 1.0
        speed -= (traits.formality - 0.5) * 0.3     # ±0.15
        speed += (traits.playfulness - 0.5) * 0.2   # ±0.1
        speed += (traits.humor_level - 0.5) * 0.2   # ±0.1
        speed = max(0.7, min(1.3, speed))  # Clamp
        
        # Volume: empathy and confidence influence
        # High empathy = softer, more gentle
        # High confidence = louder, more assertive
        volume = 0.85
        volume += (traits.confidence - 0.5) * 0.2   # ±0.1
        volume -= (traits.empathy - 0.5) * 0.1      # ±0.05 (inverse)
        volume = max(0.6, min(1.0, volume))  # Clamp
        
        # Voice gender/type
        # Use base_voice unless personality strongly suggests otherwise
        voice_type = base_voice
        
        # Effects based on traits
        effects = []
        if traits.creativity > 0.8:
            effects.append("expressive")
        if traits.confidence > 0.8 and traits.formality > 0.7:
            effects.append("authoritative")
        if traits.playfulness > 0.8:
            effects.append("cheerful")
        
        # Create description
        desc_parts = []
        if traits.formality > 0.7:
            desc_parts.append("formal")
        elif traits.formality < 0.3:
            desc_parts.append("casual")
        
        if traits.playfulness > 0.7:
            desc_parts.append("playful")
        if traits.confidence > 0.7:
            desc_parts.append("confident")
        if traits.empathy > 0.7:
            desc_parts.append("empathetic")
        
        description = f"AI-generated voice for {personality.model_name}"
        if desc_parts:
            description += f" ({', '.join(desc_parts)})"
        
        # Create voice profile
        profile = VoiceProfile(
            name=f"{personality.model_name}_ai_voice",
            pitch=pitch,
            speed=speed,
            volume=volume,
            voice=voice_type,
            effects=effects,
            description=description
        )
        
        return profile
    
    def create_from_samples(
        self,
        audio_files: List[str],
        name: str = "custom_voice",
        model_name: Optional[str] = None
    ) -> VoiceProfile:
        """
        Create voice from user audio samples (for TTS cloning).
        
        Now uses audio analysis to estimate voice parameters.
        Integration hooks for Coqui XTTS when available.
        
        Args:
            audio_files: List of paths to audio files (.wav, .mp3)
            name: Name for the voice
            model_name: Optional model name to associate with
        
        Returns:
            VoiceProfile with analyzed parameters
        """
        # Create storage directory
        if model_name:
            models_dir = Path(CONFIG["models_dir"])
            samples_dir = models_dir / model_name / "voice" / "samples"
        else:
            samples_dir = PROFILES_DIR / "samples" / name
        
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy/move samples to storage
        stored_samples = []
        for i, audio_file in enumerate(audio_files):
            src = Path(audio_file)
            if src.exists():
                ext = src.suffix
                dest = samples_dir / f"sample_{i:02d}{ext}"
                shutil.copy2(src, dest)
                stored_samples.append(str(dest))
        
        # Analyze samples to estimate voice parameters
        try:
            from .audio_analyzer import AudioAnalyzer
            analyzer = AudioAnalyzer()
            profile = analyzer.estimate_voice_profile(stored_samples, name)
            profile.description = f"Cloned voice from {len(stored_samples)} samples"
        except Exception as e:
            logger.warning(f"Could not analyze audio samples: {e}")
            # Fallback to defaults
            profile = VoiceProfile(
                name=name,
                pitch=1.0,
                speed=1.0,
                volume=0.9,
                voice="custom",
                description=f"Custom voice from {len(stored_samples)} samples"
            )
        
        # Save sample references with profile
        profile_data = profile.save()
        
        # Add sample info to a metadata file with Coqui TTS hooks
        metadata_file = samples_dir / "metadata.json"
        
        # Extract Coqui-compatible features if analyzer is available
        coqui_features = None
        try:
            from .audio_analyzer import AudioAnalyzer
            analyzer = AudioAnalyzer()
            coqui_features = analyzer.extract_coqui_features(stored_samples)
        except Exception as e:
            logger.debug(f"Coqui feature extraction not available: {e}")
        
        metadata = {
            "name": name,
            "samples": stored_samples,
            "created": profile_data.stat().st_mtime if profile_data.exists() else 0,
            "sample_count": len(stored_samples),
            "profile_parameters": {
                "pitch": profile.pitch,
                "speed": profile.speed,
                "volume": profile.volume,
                "voice": profile.voice
            },
            "coqui_features": coqui_features
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return profile
    
    def evolve_voice(
        self,
        profile: VoiceProfile,
        personality: AIPersonality,
        evolution_rate: float = 0.1
    ) -> VoiceProfile:
        """
        Gradually adjust voice as personality evolves.
        
        This creates a new profile that's slightly adjusted
        based on personality changes.
        
        Args:
            profile: Current voice profile
            personality: Updated personality
            evolution_rate: How much to change (0.0 to 1.0)
        
        Returns:
            Updated VoiceProfile
        """
        if not PERSONALITY_AVAILABLE:
            return profile
        
        # Generate target voice from current personality
        target = self.generate_voice_from_personality(personality, profile.voice)
        
        # Blend current profile with target
        evolved = VoiceProfile(
            name=profile.name,
            pitch=profile.pitch * (1 - evolution_rate) + target.pitch * evolution_rate,
            speed=profile.speed * (1 - evolution_rate) + target.speed * evolution_rate,
            volume=profile.volume * (1 - evolution_rate) + target.volume * evolution_rate,
            voice=profile.voice,  # Don't change voice type
            effects=profile.effects,  # Keep effects
            description=f"Evolved voice (generation {profile.name.split('_v')[-1] if '_v' in profile.name else '1'})"
        )
        
        # Track evolution
        if profile.name not in self.evolution_history:
            self.evolution_history[profile.name] = VoiceEvolution(
                version=1,
                base_profile=profile.name
            )
        
        history = self.evolution_history[profile.name]
        history.version += 1
        history.changes.append({
            "version": history.version,
            "pitch_delta": evolved.pitch - profile.pitch,
            "speed_delta": evolved.speed - profile.speed,
            "volume_delta": evolved.volume - profile.volume,
        })
        
        return evolved
    
    def get_voice_samples_dir(self, model_name: str) -> Path:
        """Get directory where voice samples are stored."""
        models_dir = Path(CONFIG["models_dir"])
        return models_dir / model_name / "voice" / "samples"
    
    def list_voice_samples(self, model_name: str) -> List[Path]:
        """List all voice samples for a model."""
        samples_dir = self.get_voice_samples_dir(model_name)
        if not samples_dir.exists():
            return []
        
        audio_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
        samples = []
        for file in samples_dir.iterdir():
            if file.suffix.lower() in audio_extensions:
                samples.append(file)
        
        return sorted(samples)
    
    def save_evolution_history(self, model_name: str):
        """Save voice evolution history."""
        models_dir = Path(CONFIG["models_dir"])
        history_file = models_dir / model_name / "voice" / "evolution.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            name: {
                "version": hist.version,
                "changes": hist.changes,
                "base_profile": hist.base_profile
            }
            for name, hist in self.evolution_history.items()
        }
        
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_evolution_history(self, model_name: str) -> bool:
        """Load voice evolution history."""
        models_dir = Path(CONFIG["models_dir"])
        history_file = models_dir / model_name / "voice" / "evolution.json"
        
        if not history_file.exists():
            return False
        
        try:
            with open(history_file) as f:
                data = json.load(f)
            
            self.evolution_history = {
                name: VoiceEvolution(
                    version=hist_data["version"],
                    changes=hist_data["changes"],
                    base_profile=hist_data["base_profile"]
                )
                for name, hist_data in data.items()
            }
            return True
        except Exception as e:
            logger.error(f"Error loading evolution history: {e}")
            return False


# Convenience functions
def generate_voice_for_personality(
    personality: AIPersonality,
    base_voice: str = "default"
) -> VoiceProfile:
    """Generate voice profile from personality."""
    generator = AIVoiceGenerator()
    return generator.generate_voice_from_personality(personality, base_voice)


def create_voice_from_samples(
    audio_files: List[str],
    name: str = "custom_voice",
    model_name: Optional[str] = None
) -> VoiceProfile:
    """Create voice from audio samples."""
    generator = AIVoiceGenerator()
    return generator.create_from_samples(audio_files, name, model_name)
