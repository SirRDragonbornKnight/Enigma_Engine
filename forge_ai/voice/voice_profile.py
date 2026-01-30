"""
Voice Profile System - Customize AI voices for different characters.

This module allows:
  1. Creating custom voice profiles (pitch, speed, voice type)
  2. Saving/loading voice profiles for characters
  3. User-configurable voice settings

Usage:
    from forge_ai.voice.voice_profile import VoiceProfile, VoiceEngine
    
    # Create a GLaDOS-like voice
    glados = VoiceProfile(
        name="GLaDOS",
        pitch=0.8,      # Lower pitch
        speed=0.9,      # Slightly slower
        voice="female", # Female voice
        effects=["robotic"]
    )
    
    # Use the voice
    engine = VoiceEngine()
    engine.set_profile(glados)
    engine.speak("Hello, test subject.")
"""

from __future__ import annotations

import json
import logging
import platform
import shlex
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from ..utils.system_messages import warning_msg, error_msg

logger = logging.getLogger(__name__)

# Check available TTS backends
HAVE_PYTTSX3 = False
try:
    import pyttsx3
    HAVE_PYTTSX3 = True
except ImportError:
    pass

HAVE_ESPEAK = shutil.which("espeak") is not None

# Voice profiles directory
PROFILES_DIR = Path(__file__).parent.parent.parent / "information" / "voice_profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class VoiceProfile:
    """
    A voice profile defines how the AI should sound.
    
    Attributes:
        name: Profile name (e.g., "GLaDOS", "Jarvis")
        pitch: Voice pitch multiplier (0.5 = low, 1.0 = normal, 1.5 = high)
        speed: Speech rate multiplier (0.5 = slow, 1.0 = normal, 2.0 = fast)
        volume: Volume level (0.0 to 1.0)
        voice: Voice type ("male", "female", or specific voice ID)
        effects: List of effects to apply (e.g., ["robotic", "echo"])
        language: Language code (e.g., "en", "en-US")
        description: Optional description of the character
    """
    name: str = "default"
    pitch: float = 1.0
    speed: float = 1.0
    volume: float = 1.0
    voice: str = "default"
    effects: List[str] = None
    language: str = "en"
    description: str = ""
    
    def __post_init__(self):
        if self.effects is None:
            self.effects = []
        # Clamp values to valid ranges
        self.pitch = max(0.1, min(2.0, self.pitch))
        self.speed = max(0.1, min(3.0, self.speed))
        self.volume = max(0.0, min(1.0, self.volume))
    
    def save(self, directory: Path = PROFILES_DIR) -> Path:
        """Save profile to JSON file."""
        filepath = directory / f"{self.name.lower().replace(' ', '_')}.json"
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        return filepath
    
    @classmethod
    def load(cls, name: str, directory: Path = PROFILES_DIR) -> 'VoiceProfile':
        """Load profile from JSON file."""
        filepath = directory / f"{name.lower().replace(' ', '_')}.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                return cls(**data)
        raise FileNotFoundError(f"Voice profile '{name}' not found")
    
    @classmethod
    def list_profiles(cls, directory: Path = PROFILES_DIR) -> List[str]:
        """List all available voice profiles."""
        profiles = []
        for f in directory.glob("*.json"):
            profiles.append(f.stem)
        return profiles


# =============================================================================
# PRESET VOICE PROFILES
# =============================================================================

PRESET_PROFILES = {
    "default": VoiceProfile(
        name="Default",
        pitch=1.0,
        speed=1.0,
        volume=1.0,
        voice="default",
        description="Standard neutral voice"
    ),
    "glados": VoiceProfile(
        name="GLaDOS",
        pitch=0.85,
        speed=0.95,
        volume=0.9,
        voice="female",
        effects=["robotic"],
        description="Cold, sarcastic AI from Portal"
    ),
    "jarvis": VoiceProfile(
        name="Jarvis",
        pitch=0.95,
        speed=1.0,
        volume=0.85,
        voice="male",
        effects=[],
        description="Formal British AI assistant"
    ),
    "wheatley": VoiceProfile(
        name="Wheatley",
        pitch=1.1,
        speed=1.2,
        volume=1.0,
        voice="male",
        effects=[],
        description="Nervous, rambling AI from Portal 2"
    ),
    "hal9000": VoiceProfile(
        name="HAL9000",
        pitch=0.9,
        speed=0.85,
        volume=0.7,
        voice="male",
        effects=["calm"],
        description="Calm, deliberate AI from 2001"
    ),
    "robot": VoiceProfile(
        name="Robot",
        pitch=0.7,
        speed=0.8,
        volume=1.0,
        voice="default",
        effects=["robotic"],
        description="Classic robotic voice"
    ),
    "cheerful": VoiceProfile(
        name="Cheerful",
        pitch=1.2,
        speed=1.1,
        volume=1.0,
        voice="female",
        effects=[],
        description="Happy, energetic assistant"
    ),
    "wise": VoiceProfile(
        name="Wise",
        pitch=0.8,
        speed=0.9,
        volume=0.85,
        voice="male",
        effects=[],
        description="Slow, thoughtful mentor voice"
    ),
}


class VoiceEngine:
    """
    Voice engine that applies voice profiles to TTS output.
    
    Usage:
        engine = VoiceEngine()
        engine.set_profile("glados")  # Use preset
        engine.speak("Hello, test subject.")
        
        # Or create custom profile
        custom = VoiceProfile(name="custom", pitch=1.2, speed=0.8)
        engine.set_profile(custom)
    """
    
    def __init__(self):
        self.profile = PRESET_PROFILES["default"]
        self._engine = None
        self._init_engine()
    
    def _init_engine(self):
        """Initialize the TTS engine."""
        if HAVE_PYTTSX3:
            try:
                self._engine = pyttsx3.init()
            except RuntimeError as e:
                logger.debug(f"pyttsx3 init failed (no audio device?): {e}")
                self._engine = None
            except Exception as e:
                logger.warning(f"Unexpected error initializing pyttsx3: {e}")
                self._engine = None
    
    def set_profile(self, profile) -> bool:
        """
        Set the voice profile.
        
        Args:
            profile: Either a VoiceProfile object, a preset name (str), 
                    or a dict with profile settings
        
        Returns:
            True if profile was set successfully
        """
        if isinstance(profile, str):
            # Try preset first
            if profile.lower() in PRESET_PROFILES:
                self.profile = PRESET_PROFILES[profile.lower()]
            else:
                # Try loading from file
                try:
                    self.profile = VoiceProfile.load(profile)
                except FileNotFoundError:
                    print(warning_msg(f"Profile '{profile}' not found, using default"))
                    self.profile = PRESET_PROFILES["default"]
                    return False
        elif isinstance(profile, dict):
            self.profile = VoiceProfile(**profile)
        elif isinstance(profile, VoiceProfile):
            self.profile = profile
        else:
            return False
        
        self._apply_profile()
        return True
    
    def _apply_profile(self):
        """Apply current profile settings to TTS engine."""
        if not self._engine:
            return
        
        try:
            # Set speech rate (words per minute)
            # Default is ~200 WPM, scale by speed multiplier
            base_rate = 200
            self._engine.setProperty('rate', int(base_rate * self.profile.speed))
            
            # Set volume
            self._engine.setProperty('volume', self.profile.volume)
            
            # Set voice (male/female/specific)
            voices = self._engine.getProperty('voices')
            if voices:
                target_voice = None
                voice_pref = self.profile.voice.lower()
                
                for voice in voices:
                    voice_name = voice.name.lower()
                    voice_id = voice.id.lower()
                    
                    # Check for specific voice ID
                    if voice_pref == voice_id or voice_pref == voice_name:
                        target_voice = voice.id
                        break
                    
                    # Check for gender preference
                    if voice_pref == "female" and ("female" in voice_name or "zira" in voice_name or "woman" in voice_name):
                        target_voice = voice.id
                        break
                    elif voice_pref == "male" and ("male" in voice_name or "david" in voice_name or "man" in voice_name):
                        target_voice = voice.id
                        break
                
                if target_voice:
                    self._engine.setProperty('voice', target_voice)
        except Exception as e:
            print(warning_msg(f"Could not apply voice settings: {e}"))
    
    def get_available_voices(self) -> List[Dict[str, str]]:
        """Get list of available system voices."""
        voices = []
        if self._engine:
            try:
                for voice in self._engine.getProperty('voices'):
                    voices.append({
                        "id": voice.id,
                        "name": voice.name,
                        "languages": getattr(voice, 'languages', []),
                        "gender": getattr(voice, 'gender', 'unknown')
                    })
            except Exception as e:
                logger.debug(f"Could not enumerate voices: {e}")
        return voices
    
    def speak(self, text: str, wait: bool = True):
        """
        Speak text using current voice profile.
        
        Args:
            text: Text to speak
            wait: If True, block until speech is complete
        """
        if not text:
            return
        
        # Apply effects to text if needed
        text = self._apply_text_effects(text)
        
        # Try pyttsx3 first
        if self._engine:
            try:
                self._apply_profile()  # Ensure settings are current
                self._engine.say(text)
                if wait:
                    self._engine.runAndWait()
                return
            except Exception:
                pass
        
        # Fallback to platform-specific TTS
        self._platform_speak(text)
    
    def _apply_text_effects(self, text: str) -> str:
        """Apply text-based effects (modifications before speech)."""
        if not self.profile.effects:
            return text
        
        # Try to use enhanced effects system
        try:
            from .voice_effects import VoiceEffects
            effects_system = VoiceEffects()
            
            # Apply each effect
            for effect in self.profile.effects:
                text = effects_system.apply_effect(text, effect, level=1)
            
            return text
        except ImportError:
            pass
        
        # Fallback to basic robotic effect
        if "robotic" in self.profile.effects:
            text = text.replace(". ", "... ")
            text = text.replace("? ", "?... ")
            text = text.replace("! ", "!... ")
        
        return text
    
    def _platform_speak(self, text: str):
        """Platform-specific TTS fallback."""
        import subprocess
        import os
        
        try:
            system = platform.system()
            
            if system == "Darwin":  # macOS
                # macOS 'say' command with rate (use shlex.quote to prevent injection)
                rate = int(180 * self.profile.speed)
                import subprocess
                subprocess.run(['say', '-r', str(rate), text], check=False)
                
            elif system == "Windows":
                # Windows SAPI with rate (escape text to prevent injection)
                rate = int(10 * (self.profile.speed - 1))  # SAPI rate: -10 to 10
                rate = max(-10, min(10, rate))
                # Escape quotes and special chars for PowerShell
                safe_text = text.replace("'", "''").replace('"', '`"')
                ps_script = f'''
                Add-Type -AssemblyName System.speech
                $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
                $s.Rate = {rate}
                $s.Volume = {int(self.profile.volume * 100)}
                $s.Speak('{safe_text}')
                '''
                subprocess.call(['powershell', '-c', ps_script], 
                              creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
                
            else:  # Linux
                if HAVE_ESPEAK:
                    # espeak with pitch and speed (use subprocess to prevent injection)
                    pitch = int(50 * self.profile.pitch)
                    speed = int(175 * self.profile.speed)
                    subprocess.run(['espeak', '-p', str(pitch), '-s', str(speed), text], check=False)
                else:
                    print(f"[TTS] {text}")
                    
        except Exception as e:
            print(error_msg(f"TTS failed: {e}"))
    
    def save_profile(self, name: str = None) -> Path:
        """Save current profile to file."""
        if name:
            self.profile.name = name
        return self.profile.save()
    
    def create_custom_profile(
        self,
        name: str,
        pitch: float = 1.0,
        speed: float = 1.0,
        volume: float = 1.0,
        voice: str = "default",
        effects: List[str] = None,
        description: str = ""
    ) -> VoiceProfile:
        """
        Create and save a custom voice profile.
        
        Args:
            name: Profile name
            pitch: 0.5 (low) to 1.5 (high), default 1.0
            speed: 0.5 (slow) to 2.0 (fast), default 1.0
            volume: 0.0 to 1.0, default 1.0
            voice: "male", "female", or specific voice ID
            effects: List of effects ["robotic", "echo", etc.]
            description: Character description
        
        Returns:
            The created VoiceProfile
        """
        profile = VoiceProfile(
            name=name,
            pitch=pitch,
            speed=speed,
            volume=volume,
            voice=voice,
            effects=effects or [],
            description=description
        )
        profile.save()
        return profile


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global engine instance
_default_engine: Optional[VoiceEngine] = None


def get_engine() -> VoiceEngine:
    """Get or create the default voice engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = VoiceEngine()
    return _default_engine


def speak(text: str, profile: str = None):
    """
    Speak text with optional voice profile.
    
    Args:
        text: Text to speak
        profile: Optional profile name to use
    """
    engine = get_engine()
    if profile:
        engine.set_profile(profile)
    engine.speak(text)


def set_voice(profile) -> bool:
    """Set the default voice profile."""
    return get_engine().set_profile(profile)


def list_presets() -> List[str]:
    """List available preset voice profiles."""
    return list(PRESET_PROFILES.keys())


def list_custom_profiles() -> List[str]:
    """List user-created voice profiles."""
    return VoiceProfile.list_profiles()


def list_system_voices() -> List[Dict[str, str]]:
    """List available system TTS voices."""
    return get_engine().get_available_voices()


# Initialize default profiles on import
def _init_default_profiles():
    """Create default profile files if they don't exist."""
    for name, profile in PRESET_PROFILES.items():
        filepath = PROFILES_DIR / f"{name}.json"
        if not filepath.exists():
            profile.save()

_init_default_profiles()
