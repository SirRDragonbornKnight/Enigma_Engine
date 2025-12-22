"""
STT Adapter - Unified interface for speech-to-text.

Redirects to stt_simple for compatibility.
All voice input should go through this adapter.
"""

from .stt_simple import transcribe_from_mic

def listen_for_speech(timeout=8):
    """Listen to microphone and return transcribed text."""
    return transcribe_from_mic(timeout)

__all__ = ['transcribe_from_mic', 'listen_for_speech']
