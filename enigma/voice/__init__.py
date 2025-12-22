"""
Voice Package - Unified TTS and STT interface.

Usage:
    from enigma.voice import speak, listen
    
    speak("Hello!")
    text = listen()

Components:
    - tts_simple.py: Text-to-speech (pyttsx3/espeak)
    - stt_simple.py: Speech-to-text (SpeechRecognition)
"""

from .tts_simple import speak
from .stt_simple import transcribe_from_mic as listen

__all__ = ['speak', 'listen']
