"""
Trigger Phrases / Voice Wake Module

Provides wake word detection for hands-free operation.
Allows users to wake the AI by saying phrases like "Hey Enigma".

Features:
- Configurable wake phrases
- Background listening (always-on mode)
- Phrase confidence threshold
- Multiple phrase support
- Callback system for wake events

Usage:
    from enigma.voice.trigger_phrases import TriggerPhraseDetector
    
    detector = TriggerPhraseDetector()
    detector.add_phrase("hey enigma")
    detector.add_phrase("ok enigma")
    
    detector.on_trigger(my_callback)
    detector.start_listening()
    
    # Later...
    detector.stop_listening()
"""

import logging
import threading
import time
from typing import Callable, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TriggerConfig:
    """Configuration for trigger phrase detection."""
    
    # Wake phrases to listen for
    phrases: List[str] = None
    
    # Confidence threshold (0.0 to 1.0)
    confidence_threshold: float = 0.6
    
    # Listen duration per cycle (seconds)
    listen_duration: float = 2.0
    
    # Pause between listen cycles (seconds)
    pause_duration: float = 0.1
    
    # Sound feedback when triggered
    play_sound_on_trigger: bool = True
    
    # Confirmation phrase to speak when triggered
    confirmation_phrase: str = "Yes?"
    
    def __post_init__(self):
        if self.phrases is None:
            self.phrases = ["hey enigma", "ok enigma", "enigma"]


class TriggerPhraseDetector:
    """
    Detects wake/trigger phrases in background.
    
    Listens continuously for configured phrases and triggers
    callbacks when detected.
    """
    
    def __init__(self, config: Optional[TriggerConfig] = None):
        """
        Initialize trigger phrase detector.
        
        Args:
            config: Configuration options (uses defaults if None)
        """
        self.config = config or TriggerConfig()
        
        self._listening = False
        self._listen_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[str], None]] = []
        self._stt_available = False
        
        # Check for STT availability
        self._check_stt_availability()
    
    def _check_stt_availability(self):
        """Check if speech-to-text is available."""
        try:
            # Try importing STT module
            from . import stt_simple
            self._stt_module = stt_simple
            self._stt_available = True
            logger.info("STT available for trigger phrase detection")
        except ImportError:
            self._stt_available = False
            logger.warning("STT not available - trigger phrases disabled")
    
    def add_phrase(self, phrase: str):
        """
        Add a trigger phrase to listen for.
        
        Args:
            phrase: The wake phrase (case-insensitive)
        """
        phrase_lower = phrase.lower().strip()
        if phrase_lower not in self.config.phrases:
            self.config.phrases.append(phrase_lower)
            logger.info(f"Added trigger phrase: '{phrase_lower}'")
    
    def remove_phrase(self, phrase: str) -> bool:
        """
        Remove a trigger phrase.
        
        Args:
            phrase: The phrase to remove
            
        Returns:
            True if removed, False if not found
        """
        phrase_lower = phrase.lower().strip()
        if phrase_lower in self.config.phrases:
            self.config.phrases.remove(phrase_lower)
            logger.info(f"Removed trigger phrase: '{phrase_lower}'")
            return True
        return False
    
    def get_phrases(self) -> List[str]:
        """Get list of configured trigger phrases."""
        return self.config.phrases.copy()
    
    def on_trigger(self, callback: Callable[[str], None]):
        """
        Register a callback for when a trigger phrase is detected.
        
        Args:
            callback: Function to call with the detected phrase
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[str], None]):
        """Remove a previously registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def start_listening(self) -> bool:
        """
        Start listening for trigger phrases in background.
        
        Returns:
            True if started successfully, False otherwise
        """
        if not self._stt_available:
            logger.error("Cannot start listening - STT not available")
            return False
        
        if self._listening:
            logger.warning("Already listening for trigger phrases")
            return True
        
        self._listening = True
        self._listen_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="TriggerPhraseListener"
        )
        self._listen_thread.start()
        
        logger.info("Started listening for trigger phrases")
        return True
    
    def stop_listening(self):
        """Stop listening for trigger phrases."""
        if not self._listening:
            return
        
        self._listening = False
        
        if self._listen_thread:
            self._listen_thread.join(timeout=3.0)
            self._listen_thread = None
        
        logger.info("Stopped listening for trigger phrases")
    
    def is_listening(self) -> bool:
        """Check if currently listening."""
        return self._listening
    
    def _listen_loop(self):
        """Background listening loop."""
        while self._listening:
            try:
                # Listen for speech
                text = self._stt_module.transcribe_from_mic(
                    timeout=round(self.config.listen_duration)
                )
                
                if text:
                    # Check for trigger phrases
                    text_lower = text.lower().strip()
                    detected_phrase = self._check_phrases(text_lower)
                    
                    if detected_phrase:
                        logger.info(f"Trigger phrase detected: '{detected_phrase}'")
                        self._on_trigger_detected(detected_phrase)
                
                # Brief pause before next listen cycle
                time.sleep(self.config.pause_duration)
                
            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                time.sleep(1.0)  # Longer pause on error
    
    def _check_phrases(self, text: str) -> Optional[str]:
        """
        Check if text contains a trigger phrase.
        
        Args:
            text: Transcribed text to check
            
        Returns:
            The matched phrase or None
        """
        for phrase in self.config.phrases:
            if phrase in text:
                return phrase
            
            # Check for similar phrases (fuzzy matching)
            if self._similar_phrase(text, phrase):
                return phrase
        
        return None
    
    def _similar_phrase(self, text: str, phrase: str) -> bool:
        """
        Check for similar phrases using fuzzy matching.
        
        Args:
            text: Transcribed text
            phrase: Target phrase
            
        Returns:
            True if similar enough
        """
        # Simple word overlap check
        text_words = set(text.split())
        phrase_words = set(phrase.split())
        
        if not phrase_words:
            return False
        
        overlap = len(text_words & phrase_words) / len(phrase_words)
        return overlap >= self.config.confidence_threshold
    
    def _on_trigger_detected(self, phrase: str):
        """Handle trigger phrase detection."""
        # Play confirmation sound/speak
        if self.config.play_sound_on_trigger and self.config.confirmation_phrase:
            self._speak_confirmation()
        
        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(phrase)
            except Exception as e:
                logger.error(f"Error in trigger callback: {e}")
    
    def _speak_confirmation(self):
        """Speak the confirmation phrase."""
        try:
            from .tts_simple import speak
            speak(self.config.confirmation_phrase)
        except ImportError:
            pass  # TTS not available
        except Exception as e:
            logger.warning(f"Could not speak confirmation: {e}")
    
    def check_once(self, timeout: float = 3.0) -> Optional[str]:
        """
        Listen once for a trigger phrase (blocking).
        
        Args:
            timeout: How long to listen (seconds)
            
        Returns:
            Detected phrase or None
        """
        if not self._stt_available:
            return None
        
        try:
            text = self._stt_module.transcribe_from_mic(timeout=round(timeout))
            if text:
                return self._check_phrases(text.lower().strip())
        except Exception as e:
            logger.error(f"Error in check_once: {e}")
        
        return None


# Global instance for convenience
_detector: Optional[TriggerPhraseDetector] = None


def get_detector() -> TriggerPhraseDetector:
    """Get global trigger phrase detector instance."""
    global _detector
    if _detector is None:
        _detector = TriggerPhraseDetector()
    return _detector


def start_wake_word_detection(
    phrases: Optional[List[str]] = None,
    callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Convenience function to start wake word detection.
    
    Args:
        phrases: Custom trigger phrases (uses defaults if None)
        callback: Callback function when triggered
        
    Returns:
        True if started successfully
    """
    detector = get_detector()
    
    if phrases:
        for phrase in phrases:
            detector.add_phrase(phrase)
    
    if callback:
        detector.on_trigger(callback)
    
    return detector.start_listening()


def stop_wake_word_detection():
    """Stop wake word detection."""
    global _detector
    if _detector:
        _detector.stop_listening()


def is_wake_word_active() -> bool:
    """Check if wake word detection is active."""
    global _detector
    return _detector is not None and _detector.is_listening()
