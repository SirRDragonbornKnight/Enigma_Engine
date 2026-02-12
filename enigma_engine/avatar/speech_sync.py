"""
Speech Synchronization System

Coordinates voice output with avatar lip sync animation.
When the AI speaks, this ensures the avatar's mouth moves accordingly.

Usage:
    from enigma_engine.avatar.speech_sync import SpeechSync, get_speech_sync
    
    # Get singleton instance
    sync = get_speech_sync()
    
    # Link avatar overlay
    sync.set_avatar(avatar_overlay)
    
    # Now use speak() - avatar will animate
    sync.speak("Hello, I am your AI assistant!")
    
    # Or integrate with existing voice module
    from enigma_engine.voice import speak
    speak = sync.speak  # Replace speak function
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Try to import voice module
try:
    from ..voice import get_engine
    from ..voice import speak as voice_speak
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    voice_speak = None

# Try to import adaptive animator
try:
    from .adaptive_animator import AdaptiveAnimator
    ANIMATOR_AVAILABLE = True
except ImportError:
    ANIMATOR_AVAILABLE = False


@dataclass
class SpeechEvent:
    """Represents a speech event for synchronization."""
    text: str
    duration: float = 0.0  # Estimated duration in seconds
    emotion: str = "neutral"
    priority: int = 0  # Higher = more important
    callback: Optional[Callable] = None


@dataclass
class SpeechSyncConfig:
    """Configuration for speech synchronization."""
    # Timing
    words_per_minute: float = 150.0  # Average speaking speed
    pre_speech_delay: float = 0.1    # Delay before speech starts
    post_speech_delay: float = 0.2   # Delay after speech ends
    
    # Animation
    enable_lip_sync: bool = True
    enable_emotions: bool = True
    enable_gestures: bool = True  # Nod, etc. during speech
    
    # AI command file for overlay
    command_file: str = ""


class SpeechSync:
    """
    Synchronizes voice output with avatar animation.
    
    This is the central coordinator that:
    1. Estimates speech duration
    2. Triggers avatar lip sync
    3. Manages speech queue
    4. Handles concurrent speech requests
    """
    
    def __init__(self, config: Optional[SpeechSyncConfig] = None):
        """Initialize speech sync system."""
        self.config = config or SpeechSyncConfig()
        
        # Avatar reference (set externally)
        self._avatar = None
        self._animator: Optional['AdaptiveAnimator'] = None
        
        # Speech queue
        self._speech_queue: list[SpeechEvent] = []
        self._is_speaking = False
        self._speech_lock = threading.Lock()
        
        # Voice engine
        self._voice_engine = None
        if VOICE_AVAILABLE:
            try:
                self._voice_engine = get_engine()
            except Exception:
                pass  # Intentionally silent
        
        # AI command file for avatar overlay
        if self.config.command_file:
            self._command_file = Path(self.config.command_file)
        else:
            # Default location
            from ..config import CONFIG
            self._command_file = Path(CONFIG.get("data_dir", "data")) / "avatar" / "ai_command.json"
        
        # Callbacks
        self._on_speech_start: list[Callable[[str], None]] = []
        self._on_speech_end: list[Callable[[str], None]] = []
        self._on_word: list[Callable[[str, int], None]] = []  # word, position
        
        # Worker thread
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
    
    def set_avatar(self, avatar):
        """
        Set the avatar overlay to control.
        
        Args:
            avatar: Avatar overlay instance with speak() capability
        """
        self._avatar = avatar
        
        # Check if avatar has animator
        if hasattr(avatar, '_animator') and avatar._animator:
            self._animator = avatar._animator
    
    def set_animator(self, animator: 'AdaptiveAnimator'):
        """Set the animator directly."""
        self._animator = animator
    
    def add_speech_start_callback(self, callback: Callable[[str], None]):
        """Add callback for when speech starts."""
        self._on_speech_start.append(callback)
    
    def add_speech_end_callback(self, callback: Callable[[str], None]):
        """Add callback for when speech ends."""
        self._on_speech_end.append(callback)
    
    def add_word_callback(self, callback: Callable[[str, int], None]):
        """Add callback for each word spoken."""
        self._on_word.append(callback)
    
    def estimate_duration(self, text: str) -> float:
        """
        Estimate speech duration in seconds.
        
        Args:
            text: Text to speak
            
        Returns:
            Estimated duration in seconds
        """
        # Count words
        words = text.split()
        word_count = len(words)
        
        # Calculate base duration
        minutes = word_count / self.config.words_per_minute
        duration = minutes * 60
        
        # Add time for punctuation pauses
        pause_chars = ['.', ',', '!', '?', ':', ';', '-', '—']
        pause_count = sum(text.count(c) for c in pause_chars)
        duration += pause_count * 0.15  # 150ms per pause
        
        # Add pre/post delays
        duration += self.config.pre_speech_delay + self.config.post_speech_delay
        
        return max(0.5, duration)  # Minimum 0.5 seconds
    
    def speak(
        self, 
        text: str, 
        emotion: str = "neutral",
        wait: bool = True,
        use_voice: bool = True
    ):
        """
        Speak text with synchronized avatar animation.
        
        This is the main entry point. It:
        1. Starts avatar lip sync
        2. Plays TTS audio
        3. Waits for completion
        
        Args:
            text: Text to speak
            emotion: Emotion to express (neutral, happy, sad, angry, etc.)
            wait: Whether to wait for speech to complete
            use_voice: Whether to actually play audio (False for silent animation)
        """
        if not text:
            return
        
        # Calculate duration
        duration = self.estimate_duration(text)
        
        # Notify callbacks
        for cb in self._on_speech_start:
            try:
                cb(text)
            except Exception as e:
                logger.debug(f"Speech start callback error: {e}")
        
        # Set emotion if enabled
        if self.config.enable_emotions and emotion != "neutral":
            self._set_emotion(emotion)
        
        # Start lip sync animation
        self._start_lip_sync(text, duration)
        
        # Play voice if enabled and available
        if use_voice and VOICE_AVAILABLE and self._voice_engine:
            try:
                self._voice_engine.speak(text, wait=wait)
            except Exception as e:
                print(f"Voice playback error: {e}")
                # Fallback: just wait the duration
                if wait:
                    time.sleep(duration)
        elif wait:
            # No voice, just wait
            time.sleep(duration)
        
        # Notify end
        for cb in self._on_speech_end:
            try:
                cb(text)
            except Exception as e:
                logger.debug(f"Speech end callback error: {e}")
    
    def speak_async(
        self, 
        text: str, 
        emotion: str = "neutral",
        callback: Optional[Callable] = None
    ):
        """
        Speak text asynchronously.
        
        Args:
            text: Text to speak
            emotion: Emotion to express
            callback: Function to call when speech completes
        """
        event = SpeechEvent(
            text=text,
            duration=self.estimate_duration(text),
            emotion=emotion,
            callback=callback
        )
        
        with self._speech_lock:
            self._speech_queue.append(event)
        
        # Start worker if not running
        if not self._running:
            self._start_worker()
    
    def _start_worker(self):
        """Start background worker thread."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
    
    def _worker_loop(self):
        """Background worker that processes speech queue."""
        while self._running:
            event = None
            
            with self._speech_lock:
                if self._speech_queue:
                    event = self._speech_queue.pop(0)
            
            if event:
                self._is_speaking = True
                self.speak(event.text, emotion=event.emotion, wait=True)
                self._is_speaking = False
                
                if event.callback:
                    try:
                        event.callback()
                    except Exception as e:
                        logger.debug(f"Speech callback error: {e}")
            else:
                time.sleep(0.1)  # Small sleep when queue is empty
        
    def stop(self):
        """Stop speech sync and clear queue."""
        self._running = False
        
        with self._speech_lock:
            self._speech_queue.clear()
        
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
    
    def _set_emotion(self, emotion: str):
        """Set avatar emotion."""
        # Try direct animator
        if self._animator:
            try:
                self._animator.set_emotion(emotion)
                return
            except Exception as e:
                logger.debug(f"Animator emotion error: {e}")
        
        # Try avatar method
        if self._avatar and hasattr(self._avatar, 'set_emotion'):
            try:
                self._avatar.set_emotion(emotion)
                return
            except Exception as e:
                logger.debug(f"Avatar emotion error: {e}")
        
        # Try AI command file
        self._send_command("emotion", {"emotion": emotion})
    
    def _start_lip_sync(self, text: str, duration: float):
        """Start lip sync animation."""
        # Try direct animator speak
        if self._animator:
            try:
                self._animator.speak(text, duration)
                return
            except Exception as e:
                logger.debug(f"Animator speak error: {e}")
        
        # Try avatar speak method
        if self._avatar and hasattr(self._avatar, 'animate_speak'):
            try:
                self._avatar.animate_speak(text, duration)
                return
            except Exception as e:
                logger.debug(f"Avatar animate_speak error: {e}")
        
        # Try AI command file
        self._send_command("speak", {"text": text, "duration": duration})
    
    def _send_command(self, action: str, params: dict[str, Any]):
        """Send command via AI command file."""
        try:
            self._command_file.parent.mkdir(parents=True, exist_ok=True)
            
            command = {"action": action, **params}
            with open(self._command_file, 'w') as f:
                json.dump(command, f)
                
        except Exception as e:
            print(f"Failed to send avatar command: {e}")
    
    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking
    
    @property
    def queue_length(self) -> int:
        """Get number of queued speech events."""
        with self._speech_lock:
            return len(self._speech_queue)


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_speech_sync: Optional[SpeechSync] = None


def get_speech_sync() -> SpeechSync:
    """Get the global speech sync instance."""
    global _speech_sync
    if _speech_sync is None:
        _speech_sync = SpeechSync()
    return _speech_sync


def sync_speak(
    text: str, 
    emotion: str = "neutral",
    wait: bool = True
):
    """
    Speak text with avatar sync.
    
    Convenience function that uses the global SpeechSync.
    
    Args:
        text: Text to speak
        emotion: Emotion to show
        wait: Wait for completion
    """
    sync = get_speech_sync()
    sync.speak(text, emotion=emotion, wait=wait)


def set_avatar_for_sync(avatar):
    """Set the avatar for speech sync."""
    get_speech_sync().set_avatar(avatar)


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def patch_speak_function():
    """
    Patch the enigma_engine.voice.speak function to include avatar sync.
    
    After calling this, all speak() calls will trigger avatar animation.
    """
    global voice_speak
    
    if not VOICE_AVAILABLE:
        print("Voice module not available for patching")
        return False
    
    try:
        import enigma_engine.voice.voice_profile as vp

        # Store original
        original_speak = vp.speak
        
        # Create wrapper
        def synced_speak(text: str, profile: str = None):
            sync = get_speech_sync()
            
            # Set profile if specified
            if profile:
                sync._voice_engine.set_profile(profile)
            
            # Speak with sync (this will use the voice engine internally)
            sync.speak(text, wait=True, use_voice=True)
        
        # Replace
        vp.speak = synced_speak
        
        print("✓ Patched speak() function for avatar sync")
        return True
        
    except Exception as e:
        print(f"Failed to patch speak function: {e}")
        return False


def create_voice_avatar_bridge(avatar, voice_engine=None):
    """
    Create a bridge between voice system and avatar.
    
    This sets up automatic lip sync whenever the voice engine speaks.
    
    Args:
        avatar: Avatar overlay instance
        voice_engine: Optional voice engine (uses default if None)
        
    Returns:
        SpeechSync instance configured for the bridge
    """
    sync = get_speech_sync()
    sync.set_avatar(avatar)
    
    if voice_engine:
        sync._voice_engine = voice_engine
    
    return sync
