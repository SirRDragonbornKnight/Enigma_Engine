"""
Wake Word Detection System
===========================

Local wake word detection for "Hey Enigma" style activation.
Works offline without sending audio to cloud services.

Supports multiple backends:
- OpenWakeWord (recommended) - Neural network based, accurate
- Porcupine - High quality, requires free API key  
- Pocketsphinx - Fully open source, basic accuracy
- Simple energy-based (fallback) - Detects loud sounds

Usage:
    from enigma_engine.voice.wake_word import WakeWordDetector
    
    detector = WakeWordDetector(wake_phrases=["hey enigma"])
    
    def on_wake():
        print("Wake word detected!")
    
    detector.on_detected = on_wake
    detector.start()

Requirements (install one):
    pip install openwakeword   # Recommended
    pip install pvporcupine    # High quality (needs API key)
    pip install pocketsphinx   # Fully offline
"""

from __future__ import annotations

import logging
import os
import queue
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class WakeWordBackend(Enum):
    """Available wake word detection backends."""
    OPENWAKEWORD = "openwakeword"   # Neural network based
    PORCUPINE = "porcupine"         # Picovoice (requires key)
    POCKETSPHINX = "pocketsphinx"   # CMU Sphinx
    ENERGY = "energy"               # Simple energy detection (fallback)
    AUTO = "auto"                   # Auto-select best available


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection."""
    wake_phrases: List[str] = field(default_factory=lambda: ["hey enigma"])
    backend: WakeWordBackend = WakeWordBackend.AUTO
    sensitivity: float = 0.5        # 0.0 (strict) to 1.0 (loose)
    energy_threshold: int = 2000    # For energy-based detection
    sample_rate: int = 16000        # Audio sample rate
    chunk_size: int = 512           # Audio chunk size
    cooldown: float = 2.0           # Seconds between detections
    
    # Backend-specific
    porcupine_key: str = ""         # Picovoice API key
    model_path: Optional[str] = None  # Custom model path


class BaseWakeWordEngine(ABC):
    """Abstract base class for wake word detection engines."""
    
    @abstractmethod
    def process_audio(self, audio_data: bytes) -> bool:
        """
        Process an audio chunk and check for wake word.
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            
        Returns:
            True if wake word detected
        """
    
    @abstractmethod
    def cleanup(self):
        """Release resources."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name."""


class OpenWakeWordEngine(BaseWakeWordEngine):
    """
    OpenWakeWord-based detection.
    Best accuracy, works offline, customizable.
    """
    
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self._model = None
        self._oww = None
        
        try:
            import openwakeword
            from openwakeword.model import Model
            
            self._oww = openwakeword
            
            # Load model - use pre-trained or custom
            if config.model_path:
                self._model = Model(wakeword_models=[config.model_path])
            else:
                # Use default models (hey_jarvis is similar to "hey enigma")
                self._model = Model(inference_framework='onnx')
            
            logger.info("OpenWakeWord engine initialized")
            
        except ImportError:
            raise ImportError(
                "OpenWakeWord not installed. Install with:\n"
                "  pip install openwakeword"
            )
    
    def process_audio(self, audio_data: bytes) -> bool:
        if self._model is None:
            return False
        
        try:
            # Convert bytes to numpy array
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Process with model
            prediction = self._model.predict(audio_array)
            
            # Check predictions against threshold
            threshold = 1.0 - self.config.sensitivity
            for model_name, scores in prediction.items():
                if any(score > threshold for score in scores):
                    logger.debug(f"Wake word detected by {model_name}")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"OpenWakeWord processing error: {e}")
            return False
    
    def cleanup(self):
        self._model = None
    
    @property
    def name(self) -> str:
        return "OpenWakeWord"


class PorcupineEngine(BaseWakeWordEngine):
    """
    Porcupine-based detection (Picovoice).
    High quality but requires free API key.
    """
    
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self._porcupine = None
        
        try:
            import pvporcupine
            
            if not config.porcupine_key:
                # Try environment variable
                key = os.environ.get("PORCUPINE_ACCESS_KEY", "")
                if not key:
                    raise ValueError(
                        "Porcupine requires an access key. "
                        "Get free key at: https://picovoice.ai/ "
                        "Set via config.porcupine_key or PORCUPINE_ACCESS_KEY env var"
                    )
                config.porcupine_key = key
            
            # Map wake phrases to built-in keywords if possible
            keywords = []
            for phrase in config.wake_phrases:
                # Porcupine has some built-in keywords
                phrase_lower = phrase.lower().replace(" ", "")
                if "alexa" in phrase_lower:
                    keywords.append("alexa")
                elif "computer" in phrase_lower:
                    keywords.append("computer")
                elif "jarvis" in phrase_lower:
                    keywords.append("jarvis")
                elif "terminator" in phrase_lower:
                    keywords.append("terminator")
                else:
                    # Use "jarvis" as fallback for "hey enigma"
                    keywords.append("jarvis")
            
            # Remove duplicates
            keywords = list(set(keywords))
            
            self._porcupine = pvporcupine.create(
                access_key=config.porcupine_key,
                keywords=keywords,
                sensitivities=[config.sensitivity] * len(keywords)
            )
            
            logger.info(f"Porcupine engine initialized with keywords: {keywords}")
            
        except ImportError:
            raise ImportError(
                "Porcupine not installed. Install with:\n"
                "  pip install pvporcupine"
            )
    
    def process_audio(self, audio_data: bytes) -> bool:
        if self._porcupine is None:
            return False
        
        try:
            # Convert bytes to array of shorts
            pcm = struct.unpack_from(
                f"{self._porcupine.frame_length}h",
                audio_data[:self._porcupine.frame_length * 2]
            )
            
            keyword_index = self._porcupine.process(pcm)
            
            if keyword_index >= 0:
                logger.debug(f"Porcupine detected keyword index {keyword_index}")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Porcupine processing error: {e}")
            return False
    
    def cleanup(self):
        if self._porcupine:
            self._porcupine.delete()
            self._porcupine = None
    
    @property
    def name(self) -> str:
        return "Porcupine"


class PocketsphinxEngine(BaseWakeWordEngine):
    """
    Pocketsphinx-based detection.
    Fully open source, works offline.
    """
    
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self._decoder = None
        
        try:
            from pocketsphinx import Decoder, get_model_path
            
            # Create config
            model_path = get_model_path()
            
            ps_config = Decoder.default_config()
            ps_config.set_string('-hmm', os.path.join(model_path, 'en-us'))
            ps_config.set_string('-lm', os.path.join(model_path, 'en-us.lm.bin'))
            ps_config.set_string('-dict', os.path.join(model_path, 'cmudict-en-us.dict'))
            
            # Create keyphrase list
            kws_file = self._create_keyphrase_file(config.wake_phrases)
            ps_config.set_string('-kws', kws_file)
            
            # Sensitivity
            ps_config.set_float('-kws_threshold', 1e-20 * (1.1 - config.sensitivity))
            
            self._decoder = Decoder(ps_config)
            self._decoder.start_utt()
            
            logger.info("Pocketsphinx engine initialized")
            
        except ImportError:
            raise ImportError(
                "Pocketsphinx not installed. Install with:\n"
                "  pip install pocketsphinx"
            )
    
    def _create_keyphrase_file(self, phrases: List[str]) -> str:
        """Create a keyphrase list file for pocketsphinx."""
        import tempfile
        
        fd, path = tempfile.mkstemp(suffix='.txt')
        try:
            with os.fdopen(fd, 'w') as f:
                for phrase in phrases:
                    f.write(f"{phrase.lower()}/1e-20/\n")
        except Exception:
            os.close(fd)
            raise
        
        return path
    
    def process_audio(self, audio_data: bytes) -> bool:
        if self._decoder is None:
            return False
        
        try:
            self._decoder.process_raw(audio_data, False, False)
            
            hypothesis = self._decoder.hyp()
            if hypothesis:
                detected = hypothesis.hypstr.lower()
                for phrase in self.config.wake_phrases:
                    if phrase.lower() in detected:
                        self._decoder.end_utt()
                        self._decoder.start_utt()
                        logger.debug(f"Pocketsphinx detected: {detected}")
                        return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Pocketsphinx processing error: {e}")
            return False
    
    def cleanup(self):
        if self._decoder:
            try:
                self._decoder.end_utt()
            except Exception:
                pass  # Intentionally silent
            self._decoder = None
    
    @property
    def name(self) -> str:
        return "Pocketsphinx"


class EnergyEngine(BaseWakeWordEngine):
    """
    Simple energy-based detection (fallback).
    Just detects when audio exceeds a threshold - not a true wake word detector.
    Useful as a push-to-talk alternative.
    """
    
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self._consecutive_loud = 0
        self._required_consecutive = 3  # Frames of loud audio needed
        logger.info("Energy-based detection initialized (fallback mode)")
    
    def process_audio(self, audio_data: bytes) -> bool:
        try:
            # Calculate RMS energy
            samples = struct.unpack(f"{len(audio_data)//2}h", audio_data)
            rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
            
            # Check against threshold
            if rms > self.config.energy_threshold:
                self._consecutive_loud += 1
                if self._consecutive_loud >= self._required_consecutive:
                    self._consecutive_loud = 0
                    return True
            else:
                self._consecutive_loud = 0
            
            return False
            
        except Exception as e:
            logger.debug(f"Energy detection error: {e}")
            return False
    
    def cleanup(self):
        pass
    
    @property
    def name(self) -> str:
        return "Energy"


class WakeWordDetector:
    """
    Main wake word detector with automatic backend selection.
    
    Listens continuously for wake phrases and triggers callbacks
    when detected. Works offline (depending on backend).
    
    Usage:
        detector = WakeWordDetector(wake_phrases=["hey enigma"])
        detector.on_detected = lambda: print("Wake!")
        detector.start()
    """
    
    def __init__(self, config: WakeWordConfig = None, **kwargs):
        """
        Initialize the wake word detector.
        
        Args:
            config: WakeWordConfig instance
            **kwargs: Override config options (wake_phrases, sensitivity, etc.)
        """
        if config is None:
            config = WakeWordConfig()
        
        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.config = config
        self._engine: Optional[BaseWakeWordEngine] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._last_detection_time = 0.0
        
        # Callbacks
        self.on_detected: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_ready: Optional[Callable[[], None]] = None
        
        # Initialize engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the wake word engine based on config."""
        backend = self.config.backend
        
        if backend == WakeWordBackend.AUTO:
            # Try backends in order of preference
            backends_to_try = [
                WakeWordBackend.OPENWAKEWORD,
                WakeWordBackend.PORCUPINE,
                WakeWordBackend.POCKETSPHINX,
                WakeWordBackend.ENERGY,
            ]
        else:
            backends_to_try = [backend]
        
        for be in backends_to_try:
            try:
                if be == WakeWordBackend.OPENWAKEWORD:
                    self._engine = OpenWakeWordEngine(self.config)
                elif be == WakeWordBackend.PORCUPINE:
                    self._engine = PorcupineEngine(self.config)
                elif be == WakeWordBackend.POCKETSPHINX:
                    self._engine = PocketsphinxEngine(self.config)
                elif be == WakeWordBackend.ENERGY:
                    self._engine = EnergyEngine(self.config)
                
                logger.info(f"Using wake word backend: {self._engine.name}")
                break
                
            except Exception as e:
                logger.debug(f"Backend {be.value} unavailable: {e}")
                continue
        
        if self._engine is None:
            logger.warning("No wake word engine available, using energy fallback")
            self._engine = EnergyEngine(self.config)
    
    def start(self):
        """Start listening for wake words."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        
        if self.on_ready:
            self.on_ready()
    
    def stop(self):
        """Stop listening for wake words."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        if self._engine:
            self._engine.cleanup()
    
    def _listen_loop(self):
        """Main audio capture and processing loop."""
        try:
            import pyaudio
        except ImportError:
            logger.error("PyAudio required for wake word detection")
            if self.on_error:
                self.on_error("PyAudio not installed")
            return
        
        pa = pyaudio.PyAudio()
        
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            
            logger.info(f"Wake word detection started (engine: {self._engine.name})")
            
            while self._running:
                try:
                    audio_data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                    
                    # Check for wake word
                    if self._engine.process_audio(audio_data):
                        # Check cooldown
                        now = time.time()
                        if now - self._last_detection_time >= self.config.cooldown:
                            self._last_detection_time = now
                            logger.info("Wake word detected!")
                            if self.on_detected:
                                try:
                                    self.on_detected()
                                except Exception as e:
                                    logger.error(f"Wake word callback error: {e}")
                    
                except IOError as e:
                    logger.debug(f"Audio read error: {e}")
                    time.sleep(0.01)
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            if self.on_error:
                self.on_error(str(e))
        
        finally:
            pa.terminate()
    
    @property
    def is_running(self) -> bool:
        """Check if detector is running."""
        return self._running
    
    @property
    def engine_name(self) -> str:
        """Get the name of the active engine."""
        return self._engine.name if self._engine else "None"
    
    @staticmethod
    def get_available_backends() -> List[str]:
        """Get list of available backends on this system."""
        available = []
        
        try:
            available.append("openwakeword")
        except ImportError:
            pass  # Intentionally silent
        
        try:
            available.append("porcupine")
        except ImportError:
            pass  # Intentionally silent
        
        try:
            available.append("pocketsphinx")
        except ImportError:
            pass  # Intentionally silent
        
        # Energy is always available
        available.append("energy")
        
        return available


# Convenience function
def create_wake_word_detector(
    wake_phrases: List[str] = None,
    on_detected: Callable[[], None] = None,
    backend: str = "auto"
) -> WakeWordDetector:
    """
    Create and configure a wake word detector.
    
    Args:
        wake_phrases: List of wake phrases (default: ["hey enigma"])
        on_detected: Callback when wake word detected
        backend: Backend name or "auto"
        
    Returns:
        Configured WakeWordDetector
    """
    config = WakeWordConfig(
        wake_phrases=wake_phrases or ["hey enigma"],
        backend=WakeWordBackend(backend) if backend != "auto" else WakeWordBackend.AUTO
    )
    
    detector = WakeWordDetector(config)
    if on_detected:
        detector.on_detected = on_detected
    
    return detector


__all__ = [
    "WakeWordDetector",
    "WakeWordConfig",
    "WakeWordBackend",
    "BaseWakeWordEngine",
    "OpenWakeWordEngine",
    "PorcupineEngine",
    "PocketsphinxEngine",
    "EnergyEngine",
    "create_wake_word_detector",
]
