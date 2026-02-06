"""
================================================================================
Voice Pipeline - Unified voice input/output with device awareness.
================================================================================

Combines STT and TTS with:
- Device-specific optimization (phone mic, PC mic, robot speaker)
- Wake word detection
- Noise cancellation
- Gaming mode (mute during games)
- Multi-device routing (speak on phone, listen on PC)

USAGE:
    from forge_ai.voice.voice_pipeline import VoicePipeline
    
    pipeline = VoicePipeline()
    pipeline.start()
    
    # Listen for voice
    pipeline.on_speech(lambda text: print(f"Heard: {text}"))
    
    # Speak response
    pipeline.speak("Hello! How can I help?")
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_AUDIO_CHANNELS = 1
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_WAKE_WORD = "hey forge"
DEFAULT_WAKE_SENSITIVITY = 0.5
DEFAULT_SILENCE_THRESHOLD_SECONDS = 0.3
MAX_RECORDING_SECONDS = 30.0
DEFAULT_TTS_RATE = 150
DEFAULT_TTS_VOLUME = 1.0
THREAD_JOIN_TIMEOUT = 2.0


# =============================================================================
# TTS Engine Pool - Reuse engines across pipeline instances
# =============================================================================
class TTSEnginePool:
    """
    Pool of TTS engines for efficient reuse.
    
    Creating TTS engines (especially pyttsx3) is expensive. This pool
    allows multiple VoicePipeline instances to share engines.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._engines = {}
                    cls._instance._in_use = set()
                    cls._instance._pool_lock = threading.Lock()
        return cls._instance
    
    def acquire(self, engine_type: str = "pyttsx3", config: dict = None) -> dict:
        """
        Acquire a TTS engine from the pool or create a new one.
        
        Args:
            engine_type: Type of engine ("pyttsx3", "elevenlabs", "builtin")
            config: Optional configuration for the engine
            
        Returns:
            Engine dict with type and engine instance
        """
        with self._pool_lock:
            # Find available engine of this type
            for key, engine in self._engines.items():
                if key.startswith(engine_type) and key not in self._in_use:
                    self._in_use.add(key)
                    logger.debug(f"Reusing TTS engine from pool: {key}")
                    return engine
            
            # Create new engine
            engine = self._create_engine(engine_type, config)
            if engine:
                key = f"{engine_type}_{len(self._engines)}"
                self._engines[key] = engine
                self._in_use.add(key)
                logger.debug(f"Created new TTS engine: {key}")
                return engine
            
            return {"type": "builtin"}
    
    def release(self, engine: dict):
        """Release an engine back to the pool."""
        with self._pool_lock:
            for key, eng in self._engines.items():
                if eng is engine and key in self._in_use:
                    self._in_use.discard(key)
                    logger.debug(f"Released TTS engine to pool: {key}")
                    return
    
    def _create_engine(self, engine_type: str, config: dict = None) -> dict:
        """Create a new TTS engine."""
        config = config or {}
        
        if engine_type == "pyttsx3":
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', config.get('rate', DEFAULT_TTS_RATE))
                engine.setProperty('volume', config.get('volume', DEFAULT_TTS_VOLUME))
                return {"type": "pyttsx3", "engine": engine}
            except Exception as e:
                logger.warning(f"Failed to create pyttsx3 engine: {e}")
                return None
        
        elif engine_type == "elevenlabs":
            import os
            api_key = os.environ.get("ELEVENLABS_API_KEY", "")
            if api_key:
                return {"type": "elevenlabs", "api_key": api_key}
            return None
        
        return {"type": "builtin"}
    
    def cleanup(self):
        """Clean up all engines in the pool."""
        with self._pool_lock:
            for key, engine in self._engines.items():
                if engine.get("type") == "pyttsx3" and "engine" in engine:
                    try:
                        engine["engine"].stop()
                    except Exception:
                        pass
            self._engines.clear()
            self._in_use.clear()
            logger.debug("TTS engine pool cleaned up")


# Global TTS engine pool
_tts_pool = TTSEnginePool()


class VoiceMode(Enum):
    """Voice pipeline modes."""
    FULL = auto()          # Full voice in/out
    LISTEN_ONLY = auto()   # Only speech recognition
    SPEAK_ONLY = auto()    # Only TTS
    MUTED = auto()         # All voice disabled
    GAMING = auto()        # Push-to-talk only


class VoiceDevice(Enum):
    """Types of voice devices."""
    LOCAL_MIC = auto()     # Local microphone
    LOCAL_SPEAKER = auto() # Local speaker
    REMOTE_MIC = auto()    # Remote device microphone (phone)
    REMOTE_SPEAKER = auto()# Remote device speaker
    VIRTUAL = auto()       # Virtual audio device


@dataclass
class VoiceConfig:
    """Voice pipeline configuration."""
    # Input settings
    input_device: str = "default"
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_AUDIO_CHANNELS
    chunk_size: int = DEFAULT_CHUNK_SIZE
    
    # Wake word
    wake_word: str = DEFAULT_WAKE_WORD
    wake_word_enabled: bool = True
    wake_word_sensitivity: float = DEFAULT_WAKE_SENSITIVITY
    
    # STT settings
    stt_model: str = "whisper"  # whisper, vosk, google
    stt_language: str = "en"
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD_SECONDS
    max_recording_time: float = MAX_RECORDING_SECONDS
    
    # TTS settings
    tts_engine: str = "pyttsx3"  # pyttsx3, elevenlabs, espeak
    tts_voice: str = "default"
    tts_rate: int = DEFAULT_TTS_RATE
    tts_volume: float = DEFAULT_TTS_VOLUME
    
    # Speed control settings
    speed_control_enabled: bool = True
    speed_multiplier: float = 1.0  # 0.25-3.0 (1.0 = normal)
    preserve_pitch: bool = True    # Keep pitch when stretching audio
    
    # Voice profile (for TTS customization)
    voice_profile: str = "default"  # Profile name or path
    
    # Noise reduction settings
    noise_reduction_enabled: bool = True
    noise_reduction_strength: float = 1.0  # 0.0-1.0
    
    # Echo cancellation settings
    echo_cancellation_enabled: bool = True
    
    # Audio ducking settings
    audio_ducking_enabled: bool = True
    audio_duck_level: float = 0.3  # 0.0-1.0 (lower = more ducking)
    
    # Interruption handling settings
    interruption_enabled: bool = True
    interruption_sensitivity: str = "medium"  # low, medium, high, very_high
    interruption_mode: str = "confirmed"  # immediate, confirmed, word_boundary, sentence_end
    
    # Output settings
    output_device: str = "default"
    
    # Mode settings
    default_mode: VoiceMode = VoiceMode.FULL
    gaming_mode_enabled: bool = True
    push_to_talk_key: str = "ctrl"


@dataclass
class SpeechSegment:
    """A segment of recognized speech."""
    text: str
    confidence: float
    timestamp: float
    duration: float
    is_final: bool = True


class VoicePipeline:
    """
    Unified voice input/output pipeline.
    
    Handles:
    - Speech recognition with multiple backends
    - Text-to-speech with multiple engines
    - Wake word detection
    - Gaming mode integration
    - Multi-device voice routing
    """
    
    def __init__(self, config: VoiceConfig = None):
        self.config = config or VoiceConfig()
        
        # Current mode
        self._mode = self.config.default_mode
        self._mode_lock = threading.Lock()
        
        # Speech callbacks
        self._speech_callbacks: List[Callable[[str], None]] = []
        self._wake_callbacks: List[Callable[[], None]] = []
        
        # TTS queue
        self._tts_queue: queue.Queue = queue.Queue()
        
        # State
        self._listening = False
        self._speaking = False
        self._wake_detected = False
        
        # Threads
        self._listen_thread: Optional[threading.Thread] = None
        self._speak_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Audio backends (lazy loaded)
        self._stt_engine = None
        self._tts_engine = None
        self._audio_input = None
        self._audio_output = None
        
        # Wake word detector
        self._wake_detector = None
        
        # Noise reducer (lazy loaded)
        self._noise_reducer = None
        
        # Echo canceller (lazy loaded)
        self._echo_canceller = None
        
        # Audio ducker (lazy loaded)
        self._audio_ducker = None
        
        # Speed controller (lazy loaded)
        self._speed_controller = None
        
        # Interruption handler (lazy loaded)
        self._interruption_handler = None
        
        # Flag to signal TTS stop
        self._tts_stop_requested = False
    
    def start(self):
        """Start the voice pipeline."""
        self._running = True
        
        # Initialize engines
        self._init_stt()
        self._init_tts()
        self._init_noise_reducer()
        self._init_echo_canceller()
        self._init_audio_ducker()
        self._init_speed_controller()
        self._init_interruption_handler()
        
        # Start listener thread
        if self._mode in (VoiceMode.FULL, VoiceMode.LISTEN_ONLY):
            self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._listen_thread.start()
        
        # Start speaker thread
        self._speak_thread = threading.Thread(target=self._speak_loop, daemon=True)
        self._speak_thread.start()
        
        logger.info(f"Voice pipeline started in {self._mode.name} mode")
    
    def stop(self):
        """Stop the voice pipeline."""
        self._running = False
        
        # Stop threads
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
        if self._speak_thread:
            self._tts_queue.put(None)  # Signal to stop
            self._speak_thread.join(timeout=2.0)
        
        # Cleanup
        self._cleanup_engines()
        
        logger.info("Voice pipeline stopped")
    
    def __del__(self):
        """Ensure proper cleanup when the object is garbage collected."""
        try:
            self._running = False
            
            # Release TTS engine back to pool
            if self._tts_engine:
                _tts_pool.release(self._tts_engine)
                self._tts_engine = None
            
            # Signal threads to stop (don't wait - daemon threads will die with process)
            if hasattr(self, '_tts_queue') and self._tts_queue:
                try:
                    self._tts_queue.put_nowait(None)
                except Exception:
                    pass
        except Exception:
            # Ignore errors during garbage collection
            pass
    
    def set_mode(self, mode: VoiceMode):
        """Change voice mode."""
        with self._mode_lock:
            old_mode = self._mode
            self._mode = mode
            
            logger.info(f"Voice mode: {old_mode.name} -> {mode.name}")
            
            # Start/stop listener as needed
            if mode in (VoiceMode.FULL, VoiceMode.LISTEN_ONLY):
                if not self._listen_thread or not self._listen_thread.is_alive():
                    self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
                    self._listen_thread.start()
    
    def get_mode(self) -> VoiceMode:
        """Get current voice mode."""
        with self._mode_lock:
            return self._mode
    
    def speak(self, text: str, priority: int = 0, interrupt: bool = False):
        """
        Queue text for speech.
        
        Args:
            text: Text to speak
            priority: Higher = more important
            interrupt: If True, stop current speech first
        """
        if self._mode == VoiceMode.MUTED:
            logger.debug("Voice muted, not speaking")
            return
        
        if interrupt:
            self._interrupt_speech()
        
        self._tts_queue.put((priority, text))
    
    def speak_async(self, text: str, callback: Callable[[], None] = None):
        """Speak text asynchronously with optional callback when done."""
        def speak_and_callback():
            self.speak(text)
            if callback:
                # Wait for speech to finish
                while self._speaking:
                    time.sleep(0.1)
                callback()
        
        thread = threading.Thread(target=speak_and_callback, daemon=True)
        thread.start()
    
    def on_speech(self, callback: Callable[[str], None]):
        """Register callback for recognized speech."""
        self._speech_callbacks.append(callback)
    
    def on_wake(self, callback: Callable[[], None]):
        """Register callback for wake word detection."""
        self._wake_callbacks.append(callback)
    
    def is_listening(self) -> bool:
        """Check if currently listening."""
        return self._listening
    
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._speaking
    
    def enable_gaming_mode(self):
        """Enable gaming mode (push-to-talk)."""
        self.set_mode(VoiceMode.GAMING)
    
    # =========================================================================
    # VOICE PROFILE MANAGEMENT
    # =========================================================================
    
    def set_voice_profile(self, profile_name_or_obj) -> bool:
        """
        Set the active voice profile for TTS.
        
        Args:
            profile_name_or_obj: Either a profile name string, VoiceProfile object,
                                or a dict with profile settings
        
        Returns:
            True if profile was set successfully
        """
        try:
            from .voice_profile import PRESET_PROFILES, VoiceProfile

            # Store current profile
            if isinstance(profile_name_or_obj, str):
                if profile_name_or_obj.lower() in PRESET_PROFILES:
                    self._current_voice_profile = PRESET_PROFILES[profile_name_or_obj.lower()]
                else:
                    # Try loading from file
                    self._current_voice_profile = VoiceProfile.load(profile_name_or_obj)
            elif isinstance(profile_name_or_obj, dict):
                self._current_voice_profile = VoiceProfile(**profile_name_or_obj)
            elif isinstance(profile_name_or_obj, VoiceProfile):
                self._current_voice_profile = profile_name_or_obj
            else:
                logger.warning(f"Invalid profile type: {type(profile_name_or_obj)}")
                return False
            
            # Apply to TTS engine
            self._apply_voice_profile()
            logger.info(f"Voice profile set to: {self._current_voice_profile.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set voice profile: {e}")
            return False
    
    def get_voice_profile(self) -> Optional[VoiceProfile]:
        """Get the current voice profile."""
        return getattr(self, '_current_voice_profile', None)
    
    def list_voice_profiles(self) -> List[str]:
        """
        List all available voice profiles (presets + saved).
        
        Returns:
            List of profile names
        """
        try:
            from .voice_profile import PRESET_PROFILES, VoiceProfile
            
            profiles = list(PRESET_PROFILES.keys())
            profiles.extend(VoiceProfile.list_profiles())
            return sorted(set(profiles))
        except Exception as e:
            logger.error(f"Failed to list profiles: {e}")
            return ["default"]
    
    def save_voice_profile(self, name: str = None) -> bool:
        """
        Save the current voice profile to disk.
        
        Args:
            name: Optional name for the profile
            
        Returns:
            True if saved successfully
        """
        if not hasattr(self, '_current_voice_profile') or not self._current_voice_profile:
            logger.warning("No voice profile to save")
            return False
        
        try:
            if name:
                self._current_voice_profile.name = name
            self._current_voice_profile.save()
            logger.info(f"Voice profile saved: {self._current_voice_profile.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            return False
    
    def _apply_voice_profile(self):
        """Apply the current voice profile to the TTS engine."""
        if not hasattr(self, '_current_voice_profile') or not self._current_voice_profile:
            return
        
        profile = self._current_voice_profile
        
        if self._tts_engine and self._tts_engine.get("type") == "pyttsx3":
            try:
                engine = self._tts_engine["engine"]
                
                # Apply rate (base rate * speed multiplier)
                base_rate = 200
                engine.setProperty('rate', int(base_rate * profile.speed))
                
                # Apply volume
                engine.setProperty('volume', profile.volume)
                
                # Apply voice (male/female/specific)
                voices = engine.getProperty('voices')
                if voices and profile.voice != "default":
                    voice_pref = profile.voice.lower()
                    for voice in voices:
                        if voice_pref in voice.name.lower() or voice_pref in voice.id.lower():
                            engine.setProperty('voice', voice.id)
                            break
                
                logger.debug(f"Applied voice profile: rate={int(base_rate * profile.speed)}, volume={profile.volume}")
                
            except Exception as e:
                logger.warning(f"Failed to apply voice profile to pyttsx3: {e}")
        
        elif self._tts_engine and self._tts_engine.get("type") == "elevenlabs":
            # ElevenLabs handles profiles differently via voice IDs
            logger.debug("ElevenLabs profile application not yet implemented")
    
    def disable_gaming_mode(self):
        """Disable gaming mode."""
        self.set_mode(self.config.default_mode)
    
    def _init_stt(self):
        """
        Initialize speech-to-text engine with automatic fallback.
        
        Fallback chain:
        1. User-configured STT (whisper/vosk/google)
        2. Local Whisper (if available)
        3. Vosk (if available)
        4. SpeechRecognition library
        5. Builtin (very limited)
        """
        stt_engine = None
        attempted = []
        
        # Try user's preferred STT first
        if self.config.stt_model == "whisper":
            try:
                stt_engine = self._init_whisper()
                logger.info("Initialized Whisper STT")
            except Exception as e:
                attempted.append(f"whisper: {e}")
        elif self.config.stt_model == "vosk":
            try:
                stt_engine = self._init_vosk()
                logger.info("Initialized Vosk STT")
            except Exception as e:
                attempted.append(f"vosk: {e}")
        elif self.config.stt_model == "google":
            try:
                stt_engine = self._init_speech_recognition()
                logger.info("Initialized Google Speech STT")
            except Exception as e:
                attempted.append(f"google: {e}")
        
        # Fallback chain if preferred not available
        if stt_engine is None:
            # Try local Whisper
            try:
                stt_engine = self._init_whisper()
                logger.info("Fallback: Using local Whisper STT")
            except Exception as e:
                attempted.append(f"whisper fallback: {e}")
        
        if stt_engine is None:
            # Try Vosk (good offline option)
            try:
                stt_engine = self._init_vosk()
                logger.info("Fallback: Using Vosk STT")
            except Exception as e:
                attempted.append(f"vosk fallback: {e}")
        
        if stt_engine is None:
            # Try SpeechRecognition library
            try:
                stt_engine = self._init_speech_recognition()
                logger.info("Fallback: Using SpeechRecognition STT")
            except Exception as e:
                attempted.append(f"speech_recognition fallback: {e}")
        
        if stt_engine is None:
            # Last resort: builtin (very limited)
            stt_engine = self._init_builtin_stt()
            logger.warning(f"Using limited builtin STT. Attempted: {attempted}")
        
        self._stt_engine = stt_engine
    
    def _init_tts(self):
        """Initialize text-to-speech engine using the pool."""
        try:
            config = {
                'rate': self.config.tts_rate,
                'volume': self.config.tts_volume
            }
            self._tts_engine = _tts_pool.acquire(self.config.tts_engine, config)
        except Exception as e:
            logger.warning(f"Failed to init TTS ({self.config.tts_engine}): {e}")
            self._tts_engine = {"type": "builtin"}
    
    def _init_whisper(self):
        """Initialize Whisper STT."""
        try:
            import whisper  # type: ignore
            model = whisper.load_model("base")
            return {"type": "whisper", "model": model}
        except ImportError:
            raise RuntimeError("whisper not available")
    
    def _init_vosk(self):
        """Initialize Vosk STT."""
        try:
            from vosk import KaldiRecognizer, Model
            model_path = Path(__file__).parent / "models" / "vosk-model-small"
            if model_path.exists():
                model = Model(str(model_path))
                return {"type": "vosk", "model": model}
            raise RuntimeError("Vosk model not found")
        except ImportError:
            raise RuntimeError("vosk not available")
    
    def _init_builtin_stt(self):
        """Initialize builtin STT (limited)."""
        logger.info("Using builtin STT (limited)")
        return {"type": "builtin"}
    
    def _init_speech_recognition(self):
        """
        Initialize SpeechRecognition library.
        
        This provides access to multiple backends including:
        - Google Speech API (online, free tier)
        - Sphinx (offline, less accurate)
        - Wit.ai, Bing, etc.
        """
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            return {"type": "speech_recognition", "recognizer": recognizer}
        except ImportError:
            raise RuntimeError("speech_recognition not available - pip install SpeechRecognition")
    
    def _init_pyttsx3(self):
        """Initialize pyttsx3 TTS."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', self.config.tts_rate)
            engine.setProperty('volume', self.config.tts_volume)
            return {"type": "pyttsx3", "engine": engine}
        except ImportError:
            raise RuntimeError("pyttsx3 not installed - run: pip install pyttsx3")
        except Exception as e:
            raise RuntimeError(f"pyttsx3 initialization failed: {e}")
    
    def _init_elevenlabs(self):
        """Initialize ElevenLabs TTS."""
        import os
        api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY environment variable not set")
        return {"type": "elevenlabs", "api_key": api_key}
    
    def _init_builtin_tts(self):
        """Initialize builtin TTS (limited)."""
        logger.info("Using builtin TTS (limited)")
        return {"type": "builtin"}
    
    def _init_noise_reducer(self):
        """Initialize noise reducer if enabled."""
        if not self.config.noise_reduction_enabled:
            logger.debug("Noise reduction disabled")
            return
        
        try:
            from .noise_reduction import NoiseReducer, NoiseReductionConfig
            
            nr_config = NoiseReductionConfig(
                prop_decrease=self.config.noise_reduction_strength
            )
            self._noise_reducer = NoiseReducer(nr_config)
            logger.info(f"Noise reducer initialized ({self._noise_reducer.backend_name})")
        except Exception as e:
            logger.warning(f"Failed to init noise reducer: {e}")
            self._noise_reducer = None
    
    def _apply_noise_reduction(self, audio, sample_rate: int = 16000):
        """
        Apply noise reduction to audio if enabled.
        
        Args:
            audio: Audio data (numpy array or bytes)
            sample_rate: Sample rate in Hz
        
        Returns:
            Cleaned audio in the same format as input
        """
        if self._noise_reducer is None:
            return audio
        
        try:
            import numpy as np

            # Convert to numpy if needed
            input_was_bytes = isinstance(audio, bytes)
            if input_was_bytes:
                audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_np = audio
            
            # Apply noise reduction
            cleaned = self._noise_reducer.reduce_noise(audio_np, sample_rate)
            
            # Convert back if needed
            if input_was_bytes:
                return (cleaned * 32768).astype(np.int16).tobytes()
            return cleaned
            
        except Exception as e:
            logger.debug(f"Noise reduction failed (non-fatal): {e}")
            return audio
    
    def _init_echo_canceller(self):
        """Initialize echo canceller if enabled."""
        if not self.config.echo_cancellation_enabled:
            logger.debug("Echo cancellation disabled")
            return
        
        try:
            from .echo_cancellation import EchoCancellationConfig, EchoCanceller
            
            ec_config = EchoCancellationConfig(
                sample_rate=self.config.sample_rate
            )
            self._echo_canceller = EchoCanceller(ec_config)
            logger.info(f"Echo canceller initialized ({self._echo_canceller.backend_name})")
        except Exception as e:
            logger.warning(f"Failed to init echo canceller: {e}")
            self._echo_canceller = None
    
    def _feed_speaker_reference(self, audio, sample_rate: int = None):
        """
        Feed speaker audio to echo canceller for reference.
        
        Call this when audio is played through speakers.
        
        Args:
            audio: Speaker audio being played
            sample_rate: Sample rate in Hz
        """
        if self._echo_canceller is None:
            return
        
        sample_rate = sample_rate or self.config.sample_rate
        
        try:
            import numpy as np

            # Convert to numpy if needed
            if isinstance(audio, bytes):
                audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_np = audio
            
            self._echo_canceller.feed_reference(audio_np, sample_rate)
            
        except Exception as e:
            logger.debug(f"Failed to feed speaker reference: {e}")
    
    def _apply_echo_cancellation(self, audio, sample_rate: int = None):
        """
        Apply echo cancellation to microphone audio.
        
        Args:
            audio: Microphone audio (numpy array or bytes)
            sample_rate: Sample rate in Hz
        
        Returns:
            Echo-cancelled audio in the same format as input
        """
        if self._echo_canceller is None:
            return audio
        
        sample_rate = sample_rate or self.config.sample_rate
        
        try:
            import numpy as np

            # Convert to numpy if needed
            input_was_bytes = isinstance(audio, bytes)
            if input_was_bytes:
                audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_np = audio
            
            # Apply echo cancellation
            cleaned = self._echo_canceller.process(audio_np, sample_rate)
            
            # Convert back if needed
            if input_was_bytes:
                return (cleaned * 32768).astype(np.int16).tobytes()
            return cleaned
            
        except Exception as e:
            logger.debug(f"Echo cancellation failed (non-fatal): {e}")
            return audio
    
    def _init_audio_ducker(self):
        """Initialize audio ducker if enabled."""
        if not self.config.audio_ducking_enabled:
            logger.debug("Audio ducking disabled")
            return
        
        try:
            from .audio_ducking import AudioDucker, AudioDuckingConfig
            
            duck_config = AudioDuckingConfig(
                duck_level=self.config.audio_duck_level
            )
            self._audio_ducker = AudioDucker(duck_config)
            logger.info(f"Audio ducker initialized ({self._audio_ducker.backend_name})")
        except Exception as e:
            logger.warning(f"Failed to init audio ducker: {e}")
            self._audio_ducker = None
    
    def _init_speed_controller(self):
        """Initialize speed controller if enabled."""
        if not self.config.speed_control_enabled:
            logger.debug("Speed control disabled")
            return
        
        try:
            from .speed_control import SpeedConfig, SpeedController
            
            speed_config = SpeedConfig(
                default_speed=self.config.speed_multiplier,
                preserve_pitch=self.config.preserve_pitch,
                pyttsx3_base_rate=self.config.tts_rate
            )
            self._speed_controller = SpeedController(speed_config)
            
            # Apply to TTS engine if already initialized
            if self._tts_engine and "engine" in self._tts_engine:
                self._speed_controller.apply_to_engine(self._tts_engine["engine"])
            
            logger.info(f"Speed controller initialized (speed={self.config.speed_multiplier}x)")
        except Exception as e:
            logger.warning(f"Failed to init speed controller: {e}")
            self._speed_controller = None
    
    # =========================================================================
    # Speed Control Methods
    # =========================================================================
    
    def set_speed(self, multiplier: float) -> float:
        """
        Set speech speed.
        
        Args:
            multiplier: Speed multiplier (0.25-3.0, 1.0 = normal)
        
        Returns:
            Actual speed set
        """
        if self._speed_controller:
            return self._speed_controller.set_speed(multiplier)
        
        # Fallback: adjust TTS rate directly
        self.config.speed_multiplier = max(0.25, min(3.0, multiplier))
        if self._tts_engine and "engine" in self._tts_engine:
            engine = self._tts_engine["engine"]
            if hasattr(engine, 'setProperty'):
                new_rate = int(self.config.tts_rate * self.config.speed_multiplier)
                engine.setProperty('rate', new_rate)
        return self.config.speed_multiplier
    
    def get_speed(self) -> float:
        """Get current speech speed multiplier."""
        if self._speed_controller:
            return self._speed_controller.speed
        return self.config.speed_multiplier
    
    def speed_faster(self, amount: float = 0.25) -> float:
        """Increase speech speed."""
        if self._speed_controller:
            return self._speed_controller.faster(amount)
        return self.set_speed(self.config.speed_multiplier + amount)
    
    def speed_slower(self, amount: float = 0.25) -> float:
        """Decrease speech speed."""
        if self._speed_controller:
            return self._speed_controller.slower(amount)
        return self.set_speed(self.config.speed_multiplier - amount)
    
    def speed_reset(self) -> float:
        """Reset speech speed to normal."""
        if self._speed_controller:
            return self._speed_controller.reset()
        return self.set_speed(1.0)
    
    def _init_interruption_handler(self):
        """Initialize interruption handler if enabled."""
        if not self.config.interruption_enabled:
            logger.debug("Interruption handling disabled")
            return
        
        try:
            from .interruption import (
                InterruptionConfig,
                InterruptionHandler,
                InterruptionMode,
                InterruptionSensitivity,
            )

            # Map config strings to enums
            mode_map = {
                "immediate": InterruptionMode.IMMEDIATE,
                "confirmed": InterruptionMode.CONFIRMED,
                "word_boundary": InterruptionMode.WORD_BOUNDARY,
                "sentence_end": InterruptionMode.SENTENCE_END,
            }
            sensitivity_map = {
                "low": InterruptionSensitivity.LOW,
                "medium": InterruptionSensitivity.MEDIUM,
                "high": InterruptionSensitivity.HIGH,
                "very_high": InterruptionSensitivity.VERY_HIGH,
            }
            
            int_config = InterruptionConfig(
                mode=mode_map.get(self.config.interruption_mode, InterruptionMode.CONFIRMED),
                sensitivity=sensitivity_map.get(self.config.interruption_sensitivity, InterruptionSensitivity.MEDIUM),
                sample_rate=self.config.sample_rate
            )
            self._interruption_handler = InterruptionHandler(int_config)
            
            # Set TTS stop function
            self._interruption_handler.set_tts_stop_function(self._stop_tts)
            
            # Register callback
            self._interruption_handler.on_interrupt(self._on_user_interrupt)
            
            logger.info(f"Interruption handler initialized (mode={self.config.interruption_mode}, sensitivity={self.config.interruption_sensitivity})")
        except Exception as e:
            logger.warning(f"Failed to init interruption handler: {e}")
            self._interruption_handler = None
    
    def _stop_tts(self):
        """Stop current TTS playback."""
        self._tts_stop_requested = True
        
        # Try to stop the TTS engine
        if self._tts_engine:
            engine_type = self._tts_engine.get("type")
            engine = self._tts_engine.get("engine")
            
            if engine_type == "pyttsx3" and engine:
                try:
                    engine.stop()
                except Exception as e:
                    logger.debug(f"pyttsx3 stop error: {e}")
    
    def _on_user_interrupt(self):
        """Handle user interruption callback."""
        logger.info("User interrupted - stopping TTS")
        
        # Restore audio if ducked
        if self._audio_ducker:
            try:
                self._audio_ducker.duck_end()
            except Exception:
                pass
    
    def enable_interruption(self, enabled: bool = True):
        """Enable or disable interruption handling at runtime."""
        self.config.interruption_enabled = enabled
        if enabled and not self._interruption_handler:
            self._init_interruption_handler()
        elif not enabled and self._interruption_handler:
            self._interruption_handler.stop_monitoring()
    
    def set_interruption_sensitivity(self, sensitivity: str):
        """
        Set interruption sensitivity.
        
        Args:
            sensitivity: low, medium, high, or very_high
        """
        self.config.interruption_sensitivity = sensitivity
        # Re-init to apply new sensitivity
        if self._interruption_handler:
            self._init_interruption_handler()
    
    def _cleanup_engines(self):
        """Cleanup audio engines."""
        # Release TTS engine back to pool instead of destroying it
        if self._tts_engine:
            _tts_pool.release(self._tts_engine)
            self._tts_engine = None
    
    def _listen_loop(self):
        """Main listening loop."""
        while self._running:
            try:
                with self._mode_lock:
                    mode = self._mode
                
                if mode in (VoiceMode.MUTED, VoiceMode.SPEAK_ONLY):
                    time.sleep(0.5)
                    continue
                
                if mode == VoiceMode.GAMING:
                    # Push-to-talk mode
                    if not self._check_push_to_talk():
                        time.sleep(0.1)
                        continue
                
                # Check for wake word if enabled
                if self.config.wake_word_enabled and not self._wake_detected:
                    if self._detect_wake_word():
                        self._wake_detected = True
                        for callback in self._wake_callbacks:
                            try:
                                callback()
                            except Exception as e:
                                logger.error(f"Wake callback error: {e}")
                        continue
                
                # Listen for speech
                self._listening = True
                text = self._recognize_speech()
                self._listening = False
                
                if text:
                    # Reset wake word state
                    self._wake_detected = False
                    
                    # Notify callbacks
                    for callback in self._speech_callbacks:
                        try:
                            callback(text)
                        except Exception as e:
                            logger.error(f"Speech callback error: {e}")
                
            except Exception as e:
                logger.error(f"Listen error: {e}")
                time.sleep(1.0)
    
    def _speak_loop(self):
        """Main speaking loop."""
        while self._running:
            try:
                item = self._tts_queue.get(timeout=1.0)
                if item is None:
                    break
                
                priority, text = item
                
                with self._mode_lock:
                    mode = self._mode
                
                if mode == VoiceMode.MUTED:
                    continue
                
                self._speaking = True
                self._tts_stop_requested = False
                
                # Start interruption monitoring
                if self._interruption_handler:
                    self._interruption_handler.start_monitoring()
                
                # Duck other audio during speech
                if self._audio_ducker:
                    self._audio_ducker.duck_start()
                
                try:
                    self._synthesize_speech(text)
                finally:
                    # Stop interruption monitoring
                    if self._interruption_handler:
                        self._interruption_handler.stop_monitoring()
                    
                    # Restore audio after speech
                    if self._audio_ducker:
                        self._audio_ducker.duck_end()
                
                self._speaking = False
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speak error: {e}")
                self._speaking = False
                # Ensure cleanup on error
                if self._interruption_handler:
                    try:
                        self._interruption_handler.stop_monitoring()
                    except Exception:
                        pass
                if self._audio_ducker:
                    try:
                        self._audio_ducker.duck_end()
                    except Exception:
                        pass
    
    def _recognize_speech(self) -> str:
        """Recognize speech from microphone."""
        if not self._stt_engine:
            return ""
        
        stt_type = self._stt_engine.get("type")
        
        if stt_type == "whisper":
            return self._recognize_whisper()
        elif stt_type == "vosk":
            return self._recognize_vosk()
        elif stt_type == "speech_recognition":
            return self._recognize_speech_recognition()
        else:
            return self._recognize_builtin()
    
    def _recognize_whisper(self) -> str:
        """Recognize using Whisper."""
        try:
            import numpy as np
            import sounddevice as sd

            # Record audio
            duration = 5.0  # seconds
            audio = sd.rec(
                int(duration * self.config.sample_rate),
                samplerate=self.config.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            audio_clean = audio.flatten()
            
            # Apply echo cancellation first (removes speaker feedback)
            audio_clean = self._apply_echo_cancellation(audio_clean, self.config.sample_rate)
            
            # Apply noise reduction
            audio_clean = self._apply_noise_reduction(audio_clean, self.config.sample_rate)
            
            # Transcribe
            model = self._stt_engine["model"]
            result = model.transcribe(audio_clean)
            return result.get("text", "").strip()
            
        except Exception as e:
            logger.error(f"Whisper recognition error: {e}")
            return ""
    
    def _recognize_vosk(self) -> str:
        """Recognize using Vosk."""
        try:
            import json

            import sounddevice as sd
            from vosk import KaldiRecognizer
            
            rec = KaldiRecognizer(self._stt_engine["model"], self.config.sample_rate)
            
            # Record and recognize
            with sd.RawInputStream(
                samplerate=self.config.sample_rate,
                blocksize=8000,
                channels=1,
                dtype='int16'
            ) as stream:
                while True:
                    data, _ = stream.read(4000)
                    if rec.AcceptWaveform(bytes(data)):
                        result = json.loads(rec.Result())
                        return result.get("text", "")
            
        except Exception as e:
            logger.error(f"Vosk recognition error: {e}")
            return ""
    
    def _recognize_builtin(self) -> str:
        """Builtin recognition (placeholder)."""
        # This would use a simple signal processing approach
        # For now, just log that we're using builtin
        logger.debug("Builtin STT - waiting for input")
        time.sleep(2.0)  # Simulate listening
        return ""
    
    def _recognize_speech_recognition(self) -> str:
        """
        Recognize using SpeechRecognition library.
        
        Supports multiple backends with automatic fallback:
        1. Google Speech API (online, best accuracy)
        2. Sphinx (offline, works without internet)
        """
        try:
            import speech_recognition as sr
            
            recognizer = self._stt_engine.get("recognizer")
            if not recognizer:
                recognizer = sr.Recognizer()
            
            # Use microphone
            with sr.Microphone(sample_rate=self.config.sample_rate) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                logger.debug("Listening...")
                try:
                    audio = recognizer.listen(
                        source, 
                        timeout=self.config.max_recording_time,
                        phrase_time_limit=10
                    )
                except sr.WaitTimeoutError:
                    return ""
            
            # Try Google Speech first (online)
            try:
                text = recognizer.recognize_google(audio, language=self.config.stt_language)
                return text.strip()
            except sr.RequestError:
                logger.debug("Google Speech unavailable, trying offline...")
            except sr.UnknownValueError:
                return ""  # No speech detected
            
            # Fallback to Sphinx (offline)
            try:
                text = recognizer.recognize_sphinx(audio)
                return text.strip()
            except sr.RequestError:
                logger.debug("Sphinx not available")
            except sr.UnknownValueError:
                return ""
            
            return ""
            
        except Exception as e:
            logger.error(f"SpeechRecognition error: {e}")
            return ""
    
    def _synthesize_speech(self, text: str):
        """Synthesize speech from text."""
        if not self._tts_engine:
            return
        
        tts_type = self._tts_engine.get("type")
        
        if tts_type == "pyttsx3":
            self._speak_pyttsx3(text)
        elif tts_type == "elevenlabs":
            self._speak_elevenlabs(text)
        else:
            self._speak_builtin(text)
    
    def _speak_pyttsx3(self, text: str):
        """Speak using pyttsx3."""
        try:
            engine = self._tts_engine["engine"]
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
    
    def _speak_elevenlabs(self, text: str):
        """Speak using ElevenLabs API."""
        try:
            import requests
            
            url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
            headers = {
                "xi-api-key": self._tts_engine["api_key"],
                "Content-Type": "application/json",
            }
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            if response.status_code == 200:
                # Play audio
                self._play_audio(response.content)
                
        except Exception as e:
            logger.error(f"ElevenLabs error: {e}")
    
    def _speak_builtin(self, text: str):
        """Builtin TTS (limited)."""
        # Log the text that would be spoken
        logger.info(f"TTS: {text}")
        # Simulate speaking time
        time.sleep(len(text) * 0.05)
    
    def _play_audio(self, audio_data: bytes):
        """
        Play audio data (supports MP3, WAV, OGG).
        
        Uses multiple backends for decoding:
        1. pydub (if available) - supports most formats
        2. soundfile (if available) - high quality
        3. wave (built-in) - WAV only fallback
        """
        try:
            from io import BytesIO

            import numpy as np
            import sounddevice as sd
            
            audio_array = None
            sample_rate = 44100
            
            # Try pydub first (handles MP3, OGG, etc.)
            try:
                from pydub import AudioSegment

                # Detect format from magic bytes
                if audio_data[:3] == b'ID3' or audio_data[:2] == b'\xff\xfb':
                    # MP3 format
                    audio = AudioSegment.from_mp3(BytesIO(audio_data))
                elif audio_data[:4] == b'OggS':
                    # OGG format
                    audio = AudioSegment.from_ogg(BytesIO(audio_data))
                elif audio_data[:4] == b'RIFF':
                    # WAV format
                    audio = AudioSegment.from_wav(BytesIO(audio_data))
                else:
                    # Try auto-detection
                    audio = AudioSegment.from_file(BytesIO(audio_data))
                
                # Convert to numpy array
                samples = np.array(audio.get_array_of_samples())
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2))
                audio_array = samples.astype(np.float32) / 32768.0
                sample_rate = audio.frame_rate
                
            except ImportError:
                pass
            
            # Try soundfile if pydub not available
            if audio_array is None:
                try:
                    import soundfile as sf
                    audio_array, sample_rate = sf.read(BytesIO(audio_data))
                except (ImportError, RuntimeError):
                    pass
            
            # Fallback to built-in wave for WAV files
            if audio_array is None and audio_data[:4] == b'RIFF':
                import wave
                with wave.open(BytesIO(audio_data), 'rb') as wf:
                    sample_rate = wf.getframerate()
                    n_channels = wf.getnchannels()
                    n_frames = wf.getnframes()
                    raw_data = wf.readframes(n_frames)
                    
                    # Convert to numpy
                    if wf.getsampwidth() == 2:
                        audio_array = np.frombuffer(raw_data, dtype=np.int16)
                    else:
                        audio_array = np.frombuffer(raw_data, dtype=np.int8)
                    
                    if n_channels == 2:
                        audio_array = audio_array.reshape((-1, 2))
                    audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Play if we successfully decoded
            if audio_array is not None:
                # Feed to echo canceller before playing
                self._feed_speaker_reference(audio_array, sample_rate)
                
                sd.play(audio_array, sample_rate)
                sd.wait()  # Wait until playback finishes
                logger.debug(f"Played audio: {len(audio_array)} samples at {sample_rate}Hz")
            else:
                logger.warning("Could not decode audio - no compatible decoder found")
                logger.info("Install pydub (pip install pydub) for MP3/OGG support")
                # Fallback: estimate duration and simulate
                time.sleep(max(1.0, len(audio_data) / 16000))
            
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            # Simulate playback on error
            time.sleep(2.0)
    
    def _detect_wake_word(self) -> bool:
        """
        Detect wake word in recent audio buffer.
        
        Supports multiple backends:
        1. Porcupine (if available) - most accurate
        2. Vosk (if available) - offline, good accuracy
        3. Simple keyword matching - basic fallback
        
        Returns:
            True if wake word detected
        """
        wake_word = self.config.wake_word.lower()
        
        # Try Porcupine first (commercial-grade wake word detection)
        try:
            import pvporcupine
            
            if not hasattr(self, '_porcupine') or self._porcupine is None:
                # Initialize Porcupine with built-in keyword if matching
                built_in_keywords = ['porcupine', 'bumblebee', 'alexa', 'hey siri', 'hey google', 'jarvis', 'computer']
                
                if any(kw in wake_word for kw in built_in_keywords):
                    # Use closest built-in keyword
                    for kw in built_in_keywords:
                        if kw in wake_word:
                            self._porcupine = pvporcupine.create(keywords=[kw])
                            break
                else:
                    # Can't use Porcupine without access key for custom words
                    self._porcupine = None
            
            if self._porcupine and hasattr(self, '_audio_buffer') and self._audio_buffer:
                # Process audio buffer
                result = self._porcupine.process(self._audio_buffer[-self._porcupine.frame_length:])
                if result >= 0:
                    logger.info(f"Wake word detected via Porcupine")
                    return True
                    
        except ImportError:
            pass  # Porcupine not available
        except Exception as e:
            logger.debug(f"Porcupine error: {e}")
        
        # Try Vosk for keyword spotting (free, offline)
        try:
            import json

            import vosk
            
            if not hasattr(self, '_vosk_recognizer') or self._vosk_recognizer is None:
                # Initialize Vosk with small model
                vosk.SetLogLevel(-1)  # Suppress logs
                model_path = Path.home() / '.forge_ai' / 'models' / 'vosk-model-small-en-us'
                if model_path.exists():
                    model = vosk.Model(str(model_path))
                    self._vosk_recognizer = vosk.KaldiRecognizer(model, self.config.sample_rate)
                    self._vosk_recognizer.SetWords(True)
            
            if hasattr(self, '_vosk_recognizer') and self._vosk_recognizer and hasattr(self, '_recent_audio'):
                # Feed recent audio to recognizer
                if self._vosk_recognizer.AcceptWaveform(self._recent_audio):
                    result = json.loads(self._vosk_recognizer.Result())
                    text = result.get('text', '').lower()
                    if wake_word in text or any(word in text for word in wake_word.split()):
                        logger.info(f"Wake word detected via Vosk: {text}")
                        return True
                        
        except ImportError:
            pass  # Vosk not available
        except Exception as e:
            logger.debug(f"Vosk error: {e}")
        
        # Fallback: Check if we have recent transcription that matches
        if hasattr(self, '_last_transcription') and self._last_transcription:
            text = self._last_transcription.lower()
            # Check for wake word or common variations
            wake_variants = [
                wake_word,
                wake_word.replace(' ', ''),
                wake_word.replace('hey ', ''),
                wake_word.replace('forge', 'for'),  # Common mishearing
            ]
            for variant in wake_variants:
                if variant in text:
                    logger.info(f"Wake word detected via transcription match: {text}")
                    self._last_transcription = ''  # Clear to avoid re-triggering
                    return True
        
        # Try custom trained wake word model
        if hasattr(self, '_custom_wake_model') and self._custom_wake_model:
            if hasattr(self, '_recent_audio_buffer') and self._recent_audio_buffer:
                try:
                    import numpy as np
                    audio = np.array(self._recent_audio_buffer, dtype=np.float32)
                    detected, confidence = self._custom_wake_model.detect(
                        audio, self.config.sample_rate
                    )
                    if detected:
                        logger.info(f"Wake word detected via custom model (confidence: {confidence:.2f})")
                        return True
                except Exception as e:
                    logger.debug(f"Custom wake word detection error: {e}")
        
        return False
    
    def set_custom_wake_word(self, model: WakeWordModel):
        """
        Set a custom trained wake word model.
        
        Args:
            model: Trained WakeWordModel from wake_word_trainer
        """
        self._custom_wake_model = model
        logger.info(f"Custom wake word model set for '{model.wake_phrase}'")
    
    def load_custom_wake_word(self, wake_phrase: str) -> bool:
        """
        Load a custom trained wake word model by phrase.
        
        Args:
            wake_phrase: The wake phrase to load
            
        Returns:
            True if model loaded successfully
        """
        try:
            from .wake_word_trainer import load_wake_word_model
            model = load_wake_word_model(wake_phrase)
            if model:
                self.set_custom_wake_word(model)
                return True
        except Exception as e:
            logger.error(f"Failed to load custom wake word: {e}")
        return False
    
    def _check_push_to_talk(self) -> bool:
        """Check if push-to-talk key is pressed."""
        try:
            import keyboard
            return keyboard.is_pressed(self.config.push_to_talk_key)
        except ImportError:
            logger.debug("keyboard module not available for push-to-talk")
            return False
        except Exception as e:
            logger.debug(f"Push-to-talk check failed: {e}")
            return False
    
    def _interrupt_speech(self):
        """Interrupt current speech."""
        if self._tts_engine and self._tts_engine.get("type") == "pyttsx3":
            try:
                self._tts_engine["engine"].stop()
            except Exception as e:
                logger.debug(f"Speech interrupt error (non-fatal): {e}")
        self._speaking = False


# Global pipeline instance
_pipeline: Optional[VoicePipeline] = None


def get_voice_pipeline(**kwargs) -> VoicePipeline:
    """Get or create global voice pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = VoicePipeline(**kwargs)
    return _pipeline


__all__ = [
    'VoicePipeline',
    'VoiceConfig',
    'VoiceMode',
    'VoiceDevice',
    'SpeechSegment',
    'get_voice_pipeline',
]
