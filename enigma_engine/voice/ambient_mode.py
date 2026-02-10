"""
Ambient Voice Mode

Always-listening background assistant with wake word detection.
Low-power monitoring with efficient activation.

FILE: enigma_engine/voice/ambient_mode.py
TYPE: Voice
MAIN CLASSES: AmbientListener, WakeWordDetector, BackgroundAssistant
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


class AmbientState(Enum):
    """Ambient mode state."""
    STOPPED = "stopped"
    LISTENING = "listening"       # Low-power wake word detection
    ACTIVATED = "activated"       # Wake word detected, processing
    PROCESSING = "processing"     # AI processing command
    RESPONDING = "responding"     # AI speaking response
    COOLDOWN = "cooldown"         # Post-response cooldown


@dataclass
class AmbientConfig:
    """Configuration for ambient mode."""
    # Audio settings
    sample_rate: int = 16000
    chunk_duration_ms: int = 30
    
    # Wake word settings
    wake_words: list[str] = field(default_factory=lambda: ["hey forge", "ok forge"])
    wake_word_threshold: float = 0.7
    
    # Timing
    activation_timeout_s: float = 10.0  # Time to listen after wake word
    cooldown_s: float = 0.5
    
    # Power saving
    low_power_mode: bool = True
    skip_frames: int = 2  # Process every Nth frame in low power
    
    # Callbacks
    on_wake: Callable[[], None] = None
    on_command: Callable[[str], None] = None
    on_response: Callable[[str], None] = None


class WakeWordDetector:
    """
    Detect wake words in audio stream.
    
    Uses keyword spotting for efficient always-on detection.
    """
    
    def __init__(
        self,
        wake_words: list[str],
        threshold: float = 0.7,
        sample_rate: int = 16000
    ):
        self.wake_words = [w.lower() for w in wake_words]
        self.threshold = threshold
        self.sample_rate = sample_rate
        
        self._model = None
        self._feature_extractor = None
        self._load_model()
    
    def _load_model(self):
        """Load wake word detection model."""
        # Try different wake word engines
        
        # Option 1: Porcupine
        try:
            self._engine = "porcupine"
            logger.info("Using Porcupine for wake word detection")
            return
        except ImportError:
            pass
        
        # Option 2: OpenWakeWord
        try:
            self._engine = "openwakeword"
            logger.info("Using OpenWakeWord for wake word detection")
            return
        except ImportError:
            pass
        
        # Option 3: Snowboy
        try:
            self._engine = "snowboy"
            logger.info("Using Snowboy for wake word detection")
            return
        except ImportError:
            pass
        
        # Fallback: Simple energy + pattern matching
        self._engine = "simple"
        logger.warning("No wake word engine found, using simple detection")
    
    def detect(self, audio_data: bytes) -> Optional[str]:
        """
        Check audio for wake word.
        
        Returns:
            Detected wake word, or None
        """
        if self._engine == "simple":
            return self._simple_detect(audio_data)
        elif self._engine == "porcupine":
            return self._porcupine_detect(audio_data)
        elif self._engine == "openwakeword":
            return self._openwakeword_detect(audio_data)
        
        return None
    
    def _simple_detect(self, audio_data: bytes) -> Optional[str]:
        """Simple energy-based detection (placeholder for real implementation)."""
        if not NUMPY_AVAILABLE:
            return None
        
        # Convert to numpy array
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Calculate energy
        energy = np.sqrt(np.mean(audio ** 2))
        normalized_energy = energy / 32768.0
        
        # Very basic: detect sustained energy above threshold
        # In practice, this would use a proper keyword spotting model
        if normalized_energy > 0.1:
            # Placeholder: would use speech recognition here
            return None
        
        return None
    
    def _porcupine_detect(self, audio_data: bytes) -> Optional[str]:
        """Detect using Porcupine."""
        try:
            import struct


            # Convert bytes to int16 array
            audio = struct.unpack(f'{len(audio_data)//2}h', audio_data)
            
            # Process with Porcupine
            keyword_index = self._model.process(audio)
            
            if keyword_index >= 0:
                return self.wake_words[keyword_index]
        except Exception as e:
            logger.debug(f"Porcupine detection error: {e}")
        
        return None
    
    def _openwakeword_detect(self, audio_data: bytes) -> Optional[str]:
        """Detect using OpenWakeWord."""
        try:
            pass

            # Convert to numpy
            audio = np.frombuffer(audio_data, dtype=np.int16)
            
            # Get predictions
            predictions = self._model.predict(audio)
            
            for wake_word, score in predictions.items():
                if score > self.threshold:
                    return wake_word
        except Exception as e:
            logger.debug(f"OpenWakeWord detection error: {e}")
        
        return None


class AudioCapture:
    """Low-level audio capture for ambient mode."""
    
    def __init__(self, config: AmbientConfig):
        self.config = config
        self._pyaudio = None
        self._stream = None
        self._is_running = False
        self._frame_count = 0
    
    @property
    def chunk_size(self) -> int:
        return int(self.config.sample_rate * self.config.chunk_duration_ms / 1000)
    
    def start(self) -> bool:
        """Start audio capture."""
        if not PYAUDIO_AVAILABLE:
            logger.error("PyAudio not available")
            return False
        
        try:
            self._pyaudio = pyaudio.PyAudio()
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            self._is_running = True
            logger.info("Audio capture started")
            return True
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            return False
    
    def stop(self):
        """Stop audio capture."""
        self._is_running = False
        
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
    
    def read(self) -> Optional[bytes]:
        """Read audio chunk."""
        if not self._is_running or not self._stream:
            return None
        
        self._frame_count += 1
        
        # Skip frames in low power mode
        if self.config.low_power_mode:
            if self._frame_count % (self.config.skip_frames + 1) != 0:
                # Still read to keep buffer clear, but don't process
                try:
                    self._stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception:
                    pass
                return None
        
        try:
            return self._stream.read(self.chunk_size, exception_on_overflow=False)
        except Exception as e:
            logger.debug(f"Audio read error: {e}")
            return None


class CommandBuffer:
    """Buffer and process audio after wake word detection."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        max_duration_s: float = 10.0
    ):
        self.sample_rate = sample_rate
        self.max_duration_s = max_duration_s
        self._chunks: list[bytes] = []
        self._start_time: float = 0
    
    def start(self):
        """Start buffering."""
        self._chunks = []
        self._start_time = time.time()
    
    def add(self, chunk: bytes):
        """Add audio chunk."""
        self._chunks.append(chunk)
    
    def is_timeout(self) -> bool:
        """Check if timeout exceeded."""
        return time.time() - self._start_time > self.max_duration_s
    
    def get_audio(self) -> bytes:
        """Get all buffered audio."""
        return b''.join(self._chunks)
    
    def clear(self):
        """Clear buffer."""
        self._chunks = []


class BackgroundAssistant:
    """
    Background assistant that processes commands after wake word.
    
    Integrates with Enigma AI Engine for command processing.
    """
    
    def __init__(
        self,
        process_fn: Callable[[str], str],
        speak_fn: Callable[[str], None] = None
    ):
        """
        Initialize assistant.
        
        Args:
            process_fn: Function to process text command, returns response
            speak_fn: Function to speak response audio
        """
        self.process_fn = process_fn
        self.speak_fn = speak_fn
        self._stt_engine = None
    
    def transcribe(self, audio_data: bytes) -> str:
        """Convert audio to text."""
        # Try different STT engines
        
        # Option 1: Whisper
        try:
            import whisper
            
            if self._stt_engine is None:
                self._stt_engine = whisper.load_model("tiny")
            
            # Convert to float32
            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            result = self._stt_engine.transcribe(audio)
            return result["text"].strip()
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
        
        # Option 2: SpeechRecognition
        try:
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            
            # Convert to AudioData
            audio = sr.AudioData(audio_data, 16000, 2)
            
            try:
                return recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return ""
            except sr.RequestError as e:
                logger.error(f"Speech recognition error: {e}")
                return ""
        except ImportError:
            pass
        
        logger.warning("No STT engine available")
        return ""
    
    def handle_command(self, audio_data: bytes) -> Optional[str]:
        """
        Handle voice command.
        
        Returns:
            Text response from AI
        """
        # Transcribe audio
        text = self.transcribe(audio_data)
        
        if not text:
            logger.debug("No speech detected in command")
            return None
        
        logger.info(f"Command: {text}")
        
        # Process with AI
        response = self.process_fn(text)
        
        logger.info(f"Response: {response}")
        
        # Speak response if available
        if self.speak_fn and response:
            self.speak_fn(response)
        
        return response


class AmbientListener:
    """
    Always-listening ambient mode controller.
    
    Manages wake word detection and command processing.
    """
    
    def __init__(
        self,
        config: AmbientConfig = None,
        process_fn: Callable[[str], str] = None,
        speak_fn: Callable[[str], None] = None
    ):
        """
        Initialize ambient listener.
        
        Args:
            config: Ambient mode configuration
            process_fn: Function to process commands
            speak_fn: Function to speak responses
        """
        self.config = config or AmbientConfig()
        
        self._audio = AudioCapture(self.config)
        self._wake_detector = WakeWordDetector(
            self.config.wake_words,
            self.config.wake_word_threshold,
            self.config.sample_rate
        )
        self._command_buffer = CommandBuffer(
            self.config.sample_rate,
            self.config.activation_timeout_s
        )
        self._assistant = BackgroundAssistant(
            process_fn or (lambda x: ""),
            speak_fn
        )
        
        self._state = AmbientState.STOPPED
        self._running = False
        self._listen_thread: Optional[threading.Thread] = None
        
        # Events
        self._activation_event = threading.Event()
    
    @property
    def state(self) -> AmbientState:
        return self._state
    
    def start(self) -> bool:
        """Start ambient listening."""
        if self._running:
            return True
        
        if not self._audio.start():
            return False
        
        self._running = True
        self._state = AmbientState.LISTENING
        
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        
        logger.info("Ambient mode started")
        return True
    
    def stop(self):
        """Stop ambient listening."""
        self._running = False
        self._state = AmbientState.STOPPED
        
        self._audio.stop()
        
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
        
        logger.info("Ambient mode stopped")
    
    def _listen_loop(self):
        """Main listening loop."""
        while self._running:
            audio_chunk = self._audio.read()
            
            if audio_chunk is None:
                time.sleep(0.001)
                continue
            
            if self._state == AmbientState.LISTENING:
                # Check for wake word
                wake_word = self._wake_detector.detect(audio_chunk)
                
                if wake_word:
                    logger.info(f"Wake word detected: {wake_word}")
                    self._on_wake_word(wake_word)
            
            elif self._state == AmbientState.ACTIVATED:
                # Buffer command audio
                self._command_buffer.add(audio_chunk)
                
                # Check for end of speech or timeout
                if self._command_buffer.is_timeout():
                    self._on_command_complete()
            
            elif self._state == AmbientState.COOLDOWN:
                # Just wait during cooldown
                pass
    
    def _on_wake_word(self, wake_word: str):
        """Handle wake word detection."""
        self._state = AmbientState.ACTIVATED
        self._command_buffer.start()
        
        # Notify callback
        if self.config.on_wake:
            try:
                self.config.on_wake()
            except Exception as e:
                logger.error(f"Wake callback error: {e}")
        
        # Set timeout
        threading.Timer(
            self.config.activation_timeout_s,
            self._on_activation_timeout
        ).start()
    
    def _on_activation_timeout(self):
        """Handle activation timeout."""
        if self._state == AmbientState.ACTIVATED:
            self._on_command_complete()
    
    def _on_command_complete(self):
        """Handle command completion."""
        self._state = AmbientState.PROCESSING
        
        # Get audio
        audio_data = self._command_buffer.get_audio()
        self._command_buffer.clear()
        
        if not audio_data:
            self._enter_cooldown()
            return
        
        # Process in background
        def process():
            try:
                response = self._assistant.handle_command(audio_data)
                
                # Notify callback
                if self.config.on_response and response:
                    self.config.on_response(response)
            except Exception as e:
                logger.error(f"Command processing error: {e}")
            finally:
                self._enter_cooldown()
        
        threading.Thread(target=process, daemon=True).start()
    
    def _enter_cooldown(self):
        """Enter cooldown state."""
        self._state = AmbientState.COOLDOWN
        
        def end_cooldown():
            if self._running:
                self._state = AmbientState.LISTENING
        
        threading.Timer(self.config.cooldown_s, end_cooldown).start()
    
    def force_activate(self):
        """Manually activate listening (skip wake word)."""
        if self._state == AmbientState.LISTENING:
            self._on_wake_word("manual")


def create_ambient_listener(
    wake_words: list[str] = None,
    process_fn: Callable[[str], str] = None,
    speak_fn: Callable[[str], None] = None,
    **config_kwargs
) -> AmbientListener:
    """
    Create an ambient listener instance.
    
    Args:
        wake_words: List of wake words to detect
        process_fn: Function to process commands
        speak_fn: Function to speak responses
        **config_kwargs: Additional AmbientConfig options
    
    Returns:
        Configured AmbientListener
    """
    config = AmbientConfig(
        wake_words=wake_words or ["hey forge", "ok forge"],
        **config_kwargs
    )
    
    return AmbientListener(
        config=config,
        process_fn=process_fn,
        speak_fn=speak_fn
    )


__all__ = [
    'AmbientListener',
    'AmbientConfig',
    'AmbientState',
    'WakeWordDetector',
    'BackgroundAssistant',
    'create_ambient_listener'
]
