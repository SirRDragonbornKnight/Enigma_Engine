# type: ignore
# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalMemberAccess=false
"""
================================================================================
THE LISTENER'S CHAMBER - VOICE INPUT SYSTEM
================================================================================

In the quiet depths of ForgeAI lies the Listener's Chamber, where spoken
words are transformed into text that the AI can understand. The Listener
waits patiently, ear pressed to the ether, ready to hear the sacred
wake words that summon the AI to attention.

FILE: forge_ai/voice/listener.py
TYPE: Speech-to-Text & Wake Word Detection
MAIN CLASSES: VoiceListener, VoiceConfig

    THE LISTENER'S VIGIL:
    
    [Ambient silence...]
    User: "Hey ForgeAI!"        <- Wake word detected!
    [Listener awakens]
    User: "What's the weather?"  <- Speech captured
    [Text sent to AI]

WAKE WORD OPTIONS:
    "Hey ForgeAI" - The primary invocation
    "Forge"       - A shorter summons
    Custom words can be added via VoiceConfig

REQUIREMENTS:
    pip install SpeechRecognition pyaudio
    
    Platform Notes:
    - Windows: pyaudio works out of the box
    - Linux: sudo apt install python3-pyaudio portaudio19-dev
    - macOS: brew install portaudio && pip install pyaudio

CONNECTED REALMS:
    PROVIDES TO:    forge_ai/gui/ - Voice input for chat
    WORKS WITH:     forge_ai/voice/voice_generator.py - Complete voice I/O
    USES:           speech_recognition library (Google Speech API)

USAGE:
    from forge_ai.voice.listener import VoiceListener, VoiceConfig
    
    config = VoiceConfig(wake_words=["hey forge"])
    listener = VoiceListener(config)
    
    def on_speech(text):
        print(f"You said: {text}")
    
    listener.start(callback=on_speech)
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable


# =============================================================================
# THE EARS OF THE FORGE - Speech Recognition Import
# =============================================================================
# We attempt to import the speech recognition library, which provides
# the magical ability to convert spoken words into written text.

try:
    import speech_recognition as sr
    HAS_SPEECH = True
except ImportError:
    HAS_SPEECH = False
    sr = None


# =============================================================================
# THE DEVICE ORACLE - Audio Hardware Detection
# =============================================================================

def _check_audio_device():
    """
    Consult the Device Oracle to determine if audio input exists.
    
    This sacred ritual probes the system's audio capabilities to find
    a microphone. It performs the check only once and caches the result
    to avoid repeatedly disturbing the audio subsystem.
    
    Returns:
        bool: True if an input device is found, False otherwise
    """
    import os
    
    # Allow the ritual to be bypassed for tests/headless environments
    if os.getenv('FORGE_NO_AUDIO'):
        return False
    
    if not HAS_SPEECH:
        return False
    
    try:
        # This may emit ALSA warnings once, but results are cached
        p = sr.Microphone.get_pyaudio().PyAudio()
        device_count = p.get_device_count()
        has_device = False
        for i in range(device_count):
            try:
                info = p.get_device_info_by_index(i)
                if info.get('maxInputChannels', 0) > 0:
                    has_device = True
                    break
            except (IOError, OSError):
                # Some devices fail enumeration - this is normal
                continue
        p.terminate()
        return has_device
    except Exception:
        return False


# Cache the oracle's answer - no need to ask twice
_AUDIO_DEVICE_CHECKED = False
_HAS_AUDIO_DEVICE = None


def has_audio_device():
    """
    Query whether the system possesses an audio input device.
    
    The answer is cached after the first inquiry, as audio devices
    rarely appear or vanish during a session.
    """
    global _AUDIO_DEVICE_CHECKED, _HAS_AUDIO_DEVICE
    if not _AUDIO_DEVICE_CHECKED:
        _HAS_AUDIO_DEVICE = _check_audio_device()
        _AUDIO_DEVICE_CHECKED = True
    return _HAS_AUDIO_DEVICE


# =============================================================================
# THE LISTENER'S CODEX - Configuration Settings
# =============================================================================

@dataclass
class VoiceConfig:
    """
    Configuration tome for the Voice Listener.
    
    These settings control how the Listener perceives and interprets
    spoken words from the mortal realm.
    
    Attributes:
        wake_words: The sacred phrases that awaken the Listener
        language: The tongue in which speech shall be understood
        timeout: Seconds of silence before abandoning a listen attempt
        phrase_time_limit: Maximum duration of a single utterance
        energy_threshold: Microphone sensitivity (lower = more sensitive)
        dynamic_energy: If True, auto-adjusts for ambient noise
    """
    wake_words: list[str] = None
    language: str = "en-US"
    timeout: float = 5.0
    phrase_time_limit: float = 10.0
    energy_threshold: int = 300
    dynamic_energy: bool = True
    
    def __post_init__(self):
        if self.wake_words is None:
            self.wake_words = ["hey ai tester", "forge_ai", "hey engine", "a"]


# =============================================================================
# THE ETERNAL LISTENER - Voice Detection Class
# =============================================================================

class VoiceListener:
    """
    Background voice listener with wake word detection.
    
    Usage:
        listener = VoiceListener()
        listener.on_command = lambda text: print(f"Command: {text}")
        listener.start()
        
        # Later...
        listener.stop()
    """
    
    def __init__(self, config: VoiceConfig = None):
        self.config = config or VoiceConfig()
        self._speech_available = HAS_SPEECH
        
        if HAS_SPEECH:
            self.recognizer = sr.Recognizer()
        else:
            self.recognizer = None
        self.microphone = None
        
        # Callbacks
        self.on_wake: Optional[Callable[[], None]] = None  # Called when wake word detected
        self.on_command: Optional[Callable[[str], None]] = None  # Called with transcribed command
        self.on_error: Optional[Callable[[str], None]] = None  # Called on errors
        self.on_listening: Optional[Callable[[bool], None]] = None  # Called when listening state changes
        
        # State
        self._running = False
        self._listening_for_command = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Configure recognizer (only if available)
        if self.recognizer is not None:
            self.recognizer.energy_threshold = self.config.energy_threshold
            self.recognizer.dynamic_energy_threshold = self.config.dynamic_energy
    
    def _get_microphone(self):
        """Get or create microphone instance."""
        if not HAS_SPEECH:
            raise ImportError(
                "SpeechRecognition required. Install with:\n"
                "  pip install SpeechRecognition pyaudio"
            )
        if not has_audio_device():
            raise RuntimeError(
                "No audio input device detected. "
                "Connect a microphone or disable voice input module."
            )
        if self.microphone is None:
            self.microphone = sr.Microphone()
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
        return self.microphone
    
    def start(self):
        """Start listening in background thread."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop listening."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def _listen_loop(self):
        """Main listening loop."""
        mic = self._get_microphone()
        
        while self._running and not self._stop_event.is_set():
            try:
                with mic as source:
                    # Listen for audio
                    if self.on_listening:
                        self.on_listening(True)
                    
                    try:
                        audio = self.recognizer.listen(
                            source,
                            timeout=self.config.timeout,
                            phrase_time_limit=self.config.phrase_time_limit
                        )
                    except sr.WaitTimeoutError:
                        continue
                    finally:
                        if self.on_listening:
                            self.on_listening(False)
                    
                    # Transcribe
                    try:
                        text = self.recognizer.recognize_google(
                            audio,
                            language=self.config.language
                        ).lower()
                    except sr.UnknownValueError:
                        continue  # Couldn't understand
                    except sr.RequestError as e:
                        if self.on_error:
                            self.on_error(f"Speech service error: {e}")
                        time.sleep(1)
                        continue
                    
                    # Check for wake word or process command
                    if self._listening_for_command:
                        # We're waiting for a command after wake word
                        self._listening_for_command = False
                        if self.on_command:
                            self.on_command(text)
                    else:
                        # Check for wake word
                        wake_detected = False
                        remaining_text = text
                        
                        for wake_word in self.config.wake_words:
                            if text.startswith(wake_word):
                                wake_detected = True
                                remaining_text = text[len(wake_word):].strip()
                                break
                            elif wake_word in text:
                                wake_detected = True
                                # Get text after wake word
                                idx = text.find(wake_word)
                                remaining_text = text[idx + len(wake_word):].strip()
                                break
                        
                        if wake_detected:
                            if self.on_wake:
                                self.on_wake()
                            
                            # If there's text after wake word, treat as command
                            if remaining_text and len(remaining_text) > 2:
                                if self.on_command:
                                    self.on_command(remaining_text)
                            else:
                                # Wait for next phrase as command
                                self._listening_for_command = True
                
            except Exception as e:
                if self.on_error:
                    self.on_error(str(e))
                time.sleep(0.5)
    
    def listen_once(self, timeout: float = None) -> Optional[str]:
        """
        Listen for a single phrase and return it.
        Blocking call.
        """
        timeout = timeout or self.config.timeout
        mic = self._get_microphone()
        
        try:
            with mic as source:
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=self.config.phrase_time_limit
                )
                
                text = self.recognizer.recognize_google(
                    audio,
                    language=self.config.language
                )
                return text
                
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            return None
        except sr.RequestError as e:
            if self.on_error:
                self.on_error(f"Speech service error: {e}")
            return None
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @staticmethod
    def is_available() -> bool:
        """Check if voice recognition is available."""
        if not HAS_SPEECH:
            return False
        try:
            # Try to access microphone
            sr.Microphone()
            return True
        except (OSError, AttributeError):
            return False
    
    @staticmethod
    def get_microphones() -> List[str]:
        """Get list of available microphones."""
        if not HAS_SPEECH:
            return []
        try:
            return sr.Microphone.list_microphone_names()
        except (OSError, AttributeError):
            return []


class VoiceCommander:
    """
    High-level voice command interface for Forge.
    
    Combines voice listener with command processing.
    """
    
    def __init__(self, command_callback: Callable[[str], None] = None):
        self.listener: Optional[VoiceListener] = None
        self.command_callback = command_callback
        self._notification_callback: Optional[Callable[[str, str], None]] = None
        self._status_callback: Optional[Callable[[str], None]] = None
        self.current_model = "ForgeAI"
        
    def set_notification_callback(self, callback: Callable[[str, str], None]):
        """Set callback for notifications (title, message)."""
        self._notification_callback = callback
    
    def set_status_callback(self, callback: Callable[[str], None]):
        """Set callback for status updates."""
        self._status_callback = callback
    
    def start(self) -> bool:
        """Start voice listening."""
        if not VoiceListener.is_available():
            self._notify("Voice Unavailable", 
                        "Microphone not found or SpeechRecognition not installed.\n"
                        "Install with: pip install SpeechRecognition pyaudio")
            return False
        
        try:
            self.listener = VoiceListener()
            self.listener.on_wake = self._on_wake
            self.listener.on_command = self._on_command
            self.listener.on_error = self._on_error
            self.listener.on_listening = self._on_listening
            self.listener.start()
            
            self._notify("Voice Enabled", 
                        f"Say 'Hey ForgeAI' to activate.\n"
                        f"Model: {self.current_model}")
            return True
            
        except Exception as e:
            self._notify("Voice Error", str(e))
            return False
    
    def stop(self):
        """Stop voice listening."""
        if self.listener:
            self.listener.stop()
            self.listener = None
            self._notify("Voice Disabled", "Voice activation stopped.")
    
    def _on_wake(self):
        """Called when wake word detected."""
        self._status("Listening...")
        self._notify("Forge", "I'm listening...")
    
    def _on_command(self, text: str):
        """Called when command received."""
        self._status(f"Processing: {text[:30]}...")
        
        if self.command_callback:
            self.command_callback(text)
    
    def _on_error(self, error: str):
        """Called on error."""
        self._status(f"Error: {error[:50]}")
    
    def _on_listening(self, is_listening: bool):
        """Called when listening state changes."""
        if is_listening:
            self._status("Listening...")
        else:
            self._status("Ready (say 'Hey ForgeAI')")
    
    def _notify(self, title: str, message: str):
        if self._notification_callback:
            self._notification_callback(title, message)
    
    def _status(self, text: str):
        if self._status_callback:
            self._status_callback(text)
    
    @property
    def is_running(self) -> bool:
        return self.listener is not None and self.listener.is_running


# Convenience functions
def check_voice_available() -> dict:
    """Check if voice recognition is available and return status."""
    result = {
        "available": False,
        "speech_recognition": HAS_SPEECH,
        "microphone": False,
        "microphones": [],
        "error": None
    }
    
    if not HAS_SPEECH:
        result["error"] = "SpeechRecognition not installed. Run: pip install SpeechRecognition pyaudio"
        return result
    
    try:
        mics = VoiceListener.get_microphones()
        result["microphones"] = mics
        result["microphone"] = len(mics) > 0
        result["available"] = result["microphone"]
    except Exception as e:
        result["error"] = str(e)
    
    return result
