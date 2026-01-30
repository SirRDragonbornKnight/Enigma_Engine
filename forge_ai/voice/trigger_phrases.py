"""
Trigger Phrases / Voice Wake Module

Provides wake word detection for hands-free operation.
Allows users to wake the AI by saying phrases like "Hey ForgeAI".

Features:
- Configurable wake phrases
- Background listening (always-on mode)
- Phrase confidence threshold
- Multiple phrase support
- Callback system for wake events
- AI-suggested personalized wake phrases
- Custom wake word training
- Multiple wake word "personalities" (formal vs casual)

Usage:
    from forge_ai.voice.trigger_phrases import TriggerPhraseDetector, SmartWakeWords
    
    detector = TriggerPhraseDetector()
    detector.add_phrase("hey ai tester")
    detector.add_phrase("ok ai tester")
    
    detector.on_trigger(my_callback)
    detector.start_listening()
    
    # Smart wake words
    smart = SmartWakeWords()
    suggestions = smart.suggest_wake_phrases("Forge", personality)
    
    # Later...
    detector.stop_listening()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class TriggerConfig:
    """Configuration for trigger phrase detection."""
    
    # Wake phrases to listen for
    phrases: list[str] = None
    
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
            self.phrases = ["hey ai tester", "ok ai tester", "forge_ai"]


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


# =============================================================================
# SMART WAKE WORDS
# =============================================================================

try:
    from ..core.personality import AIPersonality
    PERSONALITY_AVAILABLE = True
except ImportError:
    PERSONALITY_AVAILABLE = False


class SmartWakeWords:
    """
    Smart wake word system with AI-suggested phrases and custom training.
    
    Features:
    - AI suggests wake phrases based on name and personality
    - Multiple wake word "personalities" (formal vs casual)
    - Custom wake phrase training
    - Confidence scoring improvements
    """
    
    def __init__(self):
        """Initialize smart wake words system."""
        self.custom_phrases: Dict[str, List[str]] = {}  # phrase -> audio samples
        self.phrase_personalities: Dict[str, str] = {}  # phrase -> personality type
    
    def suggest_wake_phrases(
        self,
        ai_name: str,
        personality: Optional['AIPersonality'] = None,
        num_suggestions: int = 5
    ) -> List[str]:
        """
        AI suggests personalized wake phrases.
        
        Generates wake phrases that fit the AI's name and personality.
        
        Args:
            ai_name: Name of the AI
            personality: Optional AIPersonality for personalization
            num_suggestions: Number of suggestions to generate
            
        Returns:
            List of suggested wake phrases
        """
        suggestions = []
        name_lower = ai_name.lower()
        
        # Base suggestions
        suggestions.append(f"hey {name_lower}")
        suggestions.append(f"ok {name_lower}")
        suggestions.append(name_lower)
        
        # Personality-based suggestions
        if personality and PERSONALITY_AVAILABLE:
            # Validate that personality has the required method
            if not hasattr(personality, 'get_all_effective_traits'):
                # Fallback to default suggestions
                pass
            else:
                try:
                    traits = personality.get_all_effective_traits()
                    
                    # Formal personality
                    if traits.get("formality", 0.5) > 0.7:
                        suggestions.append(f"excuse me {name_lower}")
                        suggestions.append(f"greetings {name_lower}")
                        suggestions.append(f"attention {name_lower}")
                    
                    # Casual/playful personality
                    elif traits.get("playfulness", 0.5) > 0.6:
                        suggestions.append(f"yo {name_lower}")
                        suggestions.append(f"sup {name_lower}")
                        suggestions.append(f"{name_lower} buddy")
                    
                    # Friendly personality
                    if traits.get("empathy", 0.5) > 0.7:
                        suggestions.append(f"hello {name_lower}")
                        suggestions.append(f"{name_lower} friend")
                    
                    # Professional
                    if traits.get("confidence", 0.5) > 0.7:
                        suggestions.append(f"assistant {name_lower}")
                        suggestions.append(f"{name_lower} activate")
                except (AttributeError, TypeError):
                    # If personality doesn't have expected structure, skip
                    pass
        else:
            # Default suggestions without personality
            suggestions.extend([
                f"hello {name_lower}",
                f"{name_lower} wake up",
                f"activate {name_lower}"
            ])
        
        # Return unique suggestions (limited to num_suggestions)
        unique = []
        for phrase in suggestions:
            if phrase not in unique:
                unique.append(phrase)
            if len(unique) >= num_suggestions:
                break
        
        return unique
    
    def categorize_wake_phrase(
        self,
        phrase: str,
        personality_type: str = "neutral"
    ):
        """
        Categorize a wake phrase by personality type.
        
        Args:
            phrase: Wake phrase
            personality_type: "formal", "casual", "friendly", "neutral"
        """
        self.phrase_personalities[phrase.lower()] = personality_type
    
    def get_phrases_by_personality(
        self,
        personality_type: str
    ) -> List[str]:
        """
        Get wake phrases matching a personality type.
        
        Args:
            personality_type: "formal", "casual", "friendly", "neutral"
            
        Returns:
            List of matching phrases
        """
        return [
            phrase for phrase, ptype in self.phrase_personalities.items()
            if ptype == personality_type
        ]
    
    def train_custom_phrase(
        self,
        phrase: str,
        audio_samples: Optional[List[str]] = None
    ):
        """
        Train a custom wake phrase with audio samples.
        
        Note: Actual wake word training requires specialized models
        like Porcupine, Snowboy, or custom acoustic models. This
        method stores the samples for future integration.
        
        Args:
            phrase: Custom wake phrase
            audio_samples: Optional list of audio file paths for training
        """
        phrase_lower = phrase.lower()
        
        if audio_samples:
            self.custom_phrases[phrase_lower] = audio_samples
            logger.info(f"Stored {len(audio_samples)} samples for wake phrase '{phrase}'")
        else:
            # Add phrase without samples (uses text matching)
            self.custom_phrases[phrase_lower] = []
            logger.info(f"Added wake phrase '{phrase}' (text matching only)")
    
    def get_custom_phrases(self) -> List[str]:
        """Get list of custom-trained wake phrases."""
        return list(self.custom_phrases.keys())
    
    def improve_confidence(
        self,
        phrase: str,
        detected_text: str
    ) -> float:
        """
        Improved confidence scoring for phrase detection.
        
        Uses more sophisticated matching than simple substring search.
        
        Args:
            phrase: Target wake phrase
            detected_text: Detected speech text
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        phrase_lower = phrase.lower()
        text_lower = detected_text.lower()
        
        # Exact match
        if phrase_lower == text_lower:
            return 1.0
        
        # Substring match
        if phrase_lower in text_lower:
            # Check position - prefer phrases at start
            position = text_lower.find(phrase_lower)
            if position == 0:
                return 0.95
            else:
                return 0.85
        
        # Fuzzy word matching
        phrase_words = set(phrase_lower.split())
        text_words = set(text_lower.split())
        
        if not phrase_words:
            return 0.0
        
        # Calculate overlap
        overlap = len(phrase_words & text_words)
        ratio = overlap / len(phrase_words)
        
        # Require at least 60% word overlap
        if ratio >= 0.6:
            return ratio * 0.8  # Max 0.8 for fuzzy matches
        
        # Levenshtein-like similarity for close matches
        if len(phrase_lower) > 3 and len(text_lower) > 3:
            # Simple similarity check
            common_chars = sum(1 for c in phrase_lower if c in text_lower)
            similarity = common_chars / len(phrase_lower)
            
            if similarity >= 0.7:
                return similarity * 0.6  # Max 0.6 for character similarity
        
        return 0.0


# Convenience functions for smart wake words
def suggest_wake_phrases(
    ai_name: str,
    personality: Optional['AIPersonality'] = None
) -> List[str]:
    """
    Get AI-suggested wake phrases.
    
    Args:
        ai_name: Name of the AI
        personality: Optional personality for personalization
        
    Returns:
        List of suggested wake phrases
    """
    smart = SmartWakeWords()
    return smart.suggest_wake_phrases(ai_name, personality)


def train_custom_wake_phrase(
    phrase: str,
    audio_samples: Optional[List[str]] = None
):
    """
    Train a custom wake phrase.
    
    Args:
        phrase: Wake phrase
        audio_samples: Optional audio samples for training
    """
    smart = SmartWakeWords()
    smart.train_custom_phrase(phrase, audio_samples)

