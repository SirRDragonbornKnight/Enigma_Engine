"""
Voice-Only Mode for Enigma AI Engine

Interact with AI using only voice.

Features:
- Continuous listening
- Wake word detection
- Voice feedback
- Hands-free operation
- Command shortcuts

Usage:
    from enigma_engine.voice.voice_only_mode import VoiceOnlyMode
    
    mode = VoiceOnlyMode()
    
    # Set wake word
    mode.set_wake_word("enigma")
    
    # Start
    mode.start()
    
    # Say "enigma" then speak your command
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ListeningState(Enum):
    """Voice mode states."""
    IDLE = "idle"  # Waiting for wake word
    LISTENING = "listening"  # Actively listening
    PROCESSING = "processing"  # Processing command
    SPEAKING = "speaking"  # Outputting speech
    PAUSED = "paused"  # Temporarily paused


@dataclass
class VoiceCommand:
    """A voice command."""
    text: str
    timestamp: float = 0.0
    confidence: float = 0.0
    is_wake_word: bool = False


@dataclass
class VoiceConfig:
    """Voice mode configuration."""
    wake_word: str = "enigma"
    wake_word_sensitivity: float = 0.5  # 0-1
    listen_timeout: float = 5.0  # Seconds to listen after wake word
    silence_threshold: float = 1.5  # Seconds of silence to stop listening
    confirmation_sound: bool = True  # Play sound on wake word
    echo_commands: bool = True  # Speak back recognized commands
    continuous_mode: bool = False  # Don't require wake word
    language: str = "en-US"


class VoiceOnlyMode:
    """Voice-only interaction mode."""
    
    def __init__(
        self,
        config: Optional[VoiceConfig] = None,
        on_command: Optional[Callable[[str], str]] = None
    ):
        """
        Initialize voice-only mode.
        
        Args:
            config: Voice configuration
            on_command: Callback for processing commands
        """
        self.config = config or VoiceConfig()
        self.on_command = on_command or self._default_handler
        
        # State
        self.state = ListeningState.IDLE
        self._running = False
        
        # Threading
        self._listen_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._command_queue: queue.Queue = queue.Queue()
        
        # Audio components (lazy loaded)
        self._listener = None
        self._speaker = None
        self._recognizer = None
        
        # Callbacks
        self._state_callbacks: List[Callable[[ListeningState], None]] = []
        self._transcript_callbacks: List[Callable[[str], None]] = []
        
        # Statistics
        self._stats = {
            "commands_processed": 0,
            "wake_words_detected": 0,
            "errors": 0
        }
        
        logger.info("VoiceOnlyMode initialized")
    
    def set_wake_word(self, wake_word: str, sensitivity: float = 0.5):
        """Set the wake word."""
        self.config.wake_word = wake_word.lower()
        self.config.wake_word_sensitivity = sensitivity
    
    def set_language(self, language: str):
        """Set recognition language."""
        self.config.language = language
    
    def set_continuous_mode(self, enabled: bool):
        """Set continuous listening mode (no wake word needed)."""
        self.config.continuous_mode = enabled
    
    def start(self):
        """Start voice-only mode."""
        if self._running:
            return
        
        self._running = True
        self._init_audio()
        
        # Start listening thread
        self._listen_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )
        self._listen_thread.start()
        
        # Start processing thread
        self._process_thread = threading.Thread(
            target=self._process_loop,
            daemon=True
        )
        self._process_thread.start()
        
        self._set_state(ListeningState.IDLE)
        
        # Announce start
        if self.config.confirmation_sound:
            self._speak("Voice mode active")
        
        logger.info("Voice-only mode started")
    
    def stop(self):
        """Stop voice-only mode."""
        self._running = False
        
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
        if self._process_thread:
            self._process_thread.join(timeout=2.0)
        
        self._set_state(ListeningState.IDLE)
        
        logger.info("Voice-only mode stopped")
    
    def pause(self):
        """Pause listening."""
        self._set_state(ListeningState.PAUSED)
    
    def resume(self):
        """Resume listening."""
        self._set_state(ListeningState.IDLE)
    
    def on_state_change(self, callback: Callable[[ListeningState], None]):
        """Register state change callback."""
        self._state_callbacks.append(callback)
    
    def on_transcript(self, callback: Callable[[str], None]):
        """Register transcript callback."""
        self._transcript_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return dict(self._stats)
    
    def _init_audio(self):
        """Initialize audio components."""
        # Try to import speech recognition
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._mic = sr.Microphone()
        except ImportError:
            logger.warning("speech_recognition not installed")
            self._recognizer = None
            self._mic = None
        
        # Try to import TTS
        try:
            from enigma_engine.voice.multi_speaker import get_multi_speaker_tts
            self._speaker = get_multi_speaker_tts()
        except ImportError:
            try:
                import pyttsx3
                self._speaker = pyttsx3.init()
            except ImportError:
                logger.warning("No TTS available")
                self._speaker = None
    
    def _listen_loop(self):
        """Main listening loop."""
        while self._running:
            if self.state == ListeningState.PAUSED:
                time.sleep(0.1)
                continue
            
            if self.state == ListeningState.SPEAKING:
                time.sleep(0.1)
                continue
            
            try:
                # Listen for audio
                text = self._listen_once()
                
                if not text:
                    continue
                
                # Check for wake word
                if self.config.continuous_mode or self.state == ListeningState.LISTENING:
                    # Process command directly
                    command = VoiceCommand(
                        text=text,
                        timestamp=time.time()
                    )
                    self._command_queue.put(command)
                    
                elif self._is_wake_word(text):
                    # Wake word detected
                    self._stats["wake_words_detected"] += 1
                    
                    if self.config.confirmation_sound:
                        self._play_confirmation()
                    
                    self._set_state(ListeningState.LISTENING)
                    
                    # Listen for command
                    command_text = self._listen_for_command()
                    
                    if command_text:
                        command = VoiceCommand(
                            text=command_text,
                            timestamp=time.time()
                        )
                        self._command_queue.put(command)
                    
                    self._set_state(ListeningState.IDLE)
                    
            except Exception as e:
                logger.error(f"Listen error: {e}")
                self._stats["errors"] += 1
                time.sleep(0.5)
    
    def _process_loop(self):
        """Process commands from queue."""
        while self._running:
            try:
                command = self._command_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            self._set_state(ListeningState.PROCESSING)
            
            # Notify transcript
            for callback in self._transcript_callbacks:
                try:
                    callback(command.text)
                except Exception:
                    pass  # Intentionally silent
            
            # Process command
            try:
                response = self.on_command(command.text)
                
                if response:
                    self._set_state(ListeningState.SPEAKING)
                    self._speak(response)
                
                self._stats["commands_processed"] += 1
                
            except Exception as e:
                logger.error(f"Process error: {e}")
                self._speak("Sorry, I encountered an error")
                self._stats["errors"] += 1
            
            self._set_state(ListeningState.IDLE)
    
    def _listen_once(self) -> Optional[str]:
        """Listen for a single utterance."""
        if not self._recognizer or not self._mic:
            time.sleep(1.0)
            return None
        
        import speech_recognition as sr
        
        try:
            with self._mic as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio = self._recognizer.listen(
                    source,
                    timeout=3.0,
                    phrase_time_limit=10.0
                )
            
            # Recognize
            text = self._recognizer.recognize_google(
                audio,
                language=self.config.language
            )
            
            return text.lower().strip()
            
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            logger.error(f"Recognition error: {e}")
            return None
    
    def _listen_for_command(self) -> Optional[str]:
        """Listen for command after wake word."""
        if not self._recognizer or not self._mic:
            return None
        
        
        try:
            with self._mic as source:
                audio = self._recognizer.listen(
                    source,
                    timeout=self.config.listen_timeout,
                    phrase_time_limit=15.0
                )
            
            text = self._recognizer.recognize_google(
                audio,
                language=self.config.language
            )
            
            return text.strip()
            
        except Exception:
            return None
    
    def _is_wake_word(self, text: str) -> bool:
        """Check if text contains wake word."""
        if not text:
            return False
        
        wake_word = self.config.wake_word.lower()
        text_lower = text.lower()
        
        # Exact match
        if wake_word in text_lower:
            return True
        
        # Fuzzy match
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, wake_word, text_lower).ratio()
        
        return ratio >= self.config.wake_word_sensitivity
    
    def _speak(self, text: str):
        """Speak text."""
        if not self._speaker:
            logger.info(f"[SPEAK] {text}")
            return
        
        try:
            # Check speaker type
            if hasattr(self._speaker, 'speak'):
                # MultiSpeakerTTS
                audio = self._speaker.speak(text)
                # Play audio
                self._play_audio(audio)
            else:
                # pyttsx3
                self._speaker.say(text)
                self._speaker.runAndWait()
                
        except Exception as e:
            logger.error(f"Speech error: {e}")
    
    def _play_audio(self, audio):
        """Play generated audio."""
        try:
            import sounddevice as sd
            sd.play(audio.audio_data, audio.sample_rate)
            sd.wait()
        except ImportError:
            # Save and play with system
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            audio.save(temp_path)
            
            if os.name == 'nt':
                import winsound
                winsound.PlaySound(temp_path, winsound.SND_FILENAME)
            else:
                import subprocess
                # Try aplay first, then afplay (macOS)
                try:
                    subprocess.run(['aplay', temp_path], stderr=subprocess.DEVNULL, check=False, timeout=30)
                except FileNotFoundError:
                    subprocess.run(['afplay', temp_path], stderr=subprocess.DEVNULL, check=False, timeout=30)
            
            os.unlink(temp_path)
    
    def _play_confirmation(self):
        """Play confirmation sound."""
        # Simple beep
        try:
            import numpy as np
            
            duration = 0.1
            sample_rate = 22050
            freq = 800
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            beep = 0.3 * np.sin(2 * np.pi * freq * t)
            
            try:
                import sounddevice as sd
                sd.play(beep, sample_rate)
                sd.wait()
            except ImportError:
                pass  # Intentionally silent
                
        except Exception:
            pass  # Intentionally silent
    
    def _set_state(self, new_state: ListeningState):
        """Set state and notify."""
        self.state = new_state
        
        for callback in self._state_callbacks:
            try:
                callback(new_state)
            except Exception:
                pass  # Intentionally silent
    
    def _default_handler(self, text: str) -> str:
        """Default command handler."""
        logger.info(f"Command: {text}")
        return f"I heard: {text}"


class VoiceCommandRouter:
    """Route voice commands to handlers."""
    
    def __init__(self):
        self._commands: Dict[str, Callable[[str], str]] = {}
        self._fallback: Optional[Callable[[str], str]] = None
    
    def register(
        self,
        keywords: List[str],
        handler: Callable[[str], str]
    ):
        """
        Register command handler.
        
        Args:
            keywords: Keywords to match
            handler: Handler function
        """
        for keyword in keywords:
            self._commands[keyword.lower()] = handler
    
    def set_fallback(self, handler: Callable[[str], str]):
        """Set fallback handler for unmatched commands."""
        self._fallback = handler
    
    def route(self, text: str) -> str:
        """
        Route command to handler.
        
        Args:
            text: Command text
            
        Returns:
            Response text
        """
        text_lower = text.lower()
        
        # Check registered commands
        for keyword, handler in self._commands.items():
            if keyword in text_lower:
                return handler(text)
        
        # Fallback
        if self._fallback:
            return self._fallback(text)
        
        return "I don't understand that command"


# Pre-built command handlers
class VoiceCommands:
    """Common voice commands."""
    
    @staticmethod
    def time_handler(text: str) -> str:
        """Handle time queries."""
        from datetime import datetime
        now = datetime.now()
        return f"The time is {now.strftime('%I:%M %p')}"
    
    @staticmethod
    def date_handler(text: str) -> str:
        """Handle date queries."""
        from datetime import datetime
        now = datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}"
    
    @staticmethod
    def help_handler(text: str) -> str:
        """Handle help requests."""
        return "You can ask me about the time, date, or just talk to me"
    
    @staticmethod
    def stop_handler(text: str) -> str:
        """Handle stop command."""
        return "Goodbye"


def create_voice_mode_with_ai(ai_callback: Callable[[str], str]) -> VoiceOnlyMode:
    """
    Create voice mode with AI integration.
    
    Args:
        ai_callback: Function to get AI response
        
    Returns:
        Configured VoiceOnlyMode
    """
    router = VoiceCommandRouter()
    
    # Register built-in commands
    router.register(["what time", "current time"], VoiceCommands.time_handler)
    router.register(["what date", "today's date", "what day"], VoiceCommands.date_handler)
    router.register(["help", "what can you"], VoiceCommands.help_handler)
    
    # Set AI as fallback
    router.set_fallback(ai_callback)
    
    mode = VoiceOnlyMode(on_command=router.route)
    
    return mode


# Global instance
_voice_mode: Optional[VoiceOnlyMode] = None


def get_voice_mode() -> VoiceOnlyMode:
    """Get or create global VoiceOnlyMode instance."""
    global _voice_mode
    if _voice_mode is None:
        _voice_mode = VoiceOnlyMode()
    return _voice_mode
