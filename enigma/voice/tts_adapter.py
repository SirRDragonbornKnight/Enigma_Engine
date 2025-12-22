"""
TTS Adapter - Unified interface for text-to-speech.

Redirects to tts_simple.speak() for compatibility.
All voice output should go through this adapter.
"""

from .tts_simple import speak

__all__ = ['speak']
