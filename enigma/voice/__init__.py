"""
Voice Package - Unified TTS and STT interface with voice customization.

Usage:
    from enigma.voice import speak, listen, set_voice
    
    # Basic usage
    speak("Hello!")
    text = listen()
    
    # Use a character voice
    set_voice("glados")
    speak("Hello, test subject.")
    
    # Create custom voice
    from enigma.voice import VoiceProfile, VoiceEngine
    
    my_voice = VoiceProfile(
        name="MyCharacter",
        pitch=0.8,      # Lower pitch
        speed=1.1,      # Slightly faster
        voice="female"
    )
    my_voice.save()
    set_voice("MyCharacter")

Available Preset Voices:
    - default   : Standard neutral voice
    - glados    : Cold, sarcastic AI (Portal)
    - jarvis    : Formal British assistant
    - wheatley  : Nervous, rambling AI
    - hal9000   : Calm, deliberate AI
    - robot     : Classic robotic voice
    - cheerful  : Happy, energetic
    - wise      : Slow, thoughtful mentor

Components:
    - tts_simple.py: Basic text-to-speech
    - stt_simple.py: Speech-to-text (SpeechRecognition)
    - voice_profile.py: Voice customization system
    - whisper_stt.py: High-quality Whisper STT (optional)
    - natural_tts.py: Natural TTS with Coqui/Bark (optional)
"""

from .stt_simple import transcribe_from_mic as listen
from .voice_profile import (
    VoiceProfile,
    VoiceEngine,
    speak,
    set_voice,
    get_engine,
    list_presets,
    list_custom_profiles,
    list_system_voices,
    PRESET_PROFILES,
)

# New voice options (optional dependencies)
from .whisper_stt import WhisperSTT
from .natural_tts import NaturalTTS

__all__ = [
    'speak',
    'listen', 
    'set_voice',
    'VoiceProfile',
    'VoiceEngine',
    'get_engine',
    'list_presets',
    'list_custom_profiles',
    'list_system_voices',
    'PRESET_PROFILES',
    'WhisperSTT',
    'NaturalTTS',
]

