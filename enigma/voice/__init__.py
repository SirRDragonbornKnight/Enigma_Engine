"""
Voice Package - Unified TTS and STT interface.

Usage:
    from enigma.voice import speak, listen
    
    speak("Hello!")
    text = listen()

Components:
    - tts_simple.py: Text-to-speech (pyttsx3/espeak)
    - stt_simple.py: Speech-to-text (SpeechRecognition)
    - tts_adapter.py: Compatibility adapter
    - stt_adapter.py: Compatibility adapter
    - voice_system.py: Advanced voice with Piper support
"""

from .tts_simple import speak
from .stt_simple import transcribe_from_mic as listen

__all__ = ['speak', 'listen']
