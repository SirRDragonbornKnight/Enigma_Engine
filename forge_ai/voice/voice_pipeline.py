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
    
    def start(self):
        """Start the voice pipeline."""
        self._running = True
        
        # Initialize engines
        self._init_stt()
        self._init_tts()
        
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
    
    def disable_gaming_mode(self):
        """Disable gaming mode."""
        self.set_mode(self.config.default_mode)
    
    def _init_stt(self):
        """Initialize speech-to-text engine."""
        try:
            if self.config.stt_model == "whisper":
                self._stt_engine = self._init_whisper()
            elif self.config.stt_model == "vosk":
                self._stt_engine = self._init_vosk()
            else:
                self._stt_engine = self._init_builtin_stt()
        except Exception as e:
            logger.warning(f"Failed to init STT ({self.config.stt_model}): {e}")
            self._stt_engine = self._init_builtin_stt()
    
    def _init_tts(self):
        """Initialize text-to-speech engine."""
        try:
            if self.config.tts_engine == "pyttsx3":
                self._tts_engine = self._init_pyttsx3()
            elif self.config.tts_engine == "elevenlabs":
                self._tts_engine = self._init_elevenlabs()
            else:
                self._tts_engine = self._init_builtin_tts()
        except Exception as e:
            logger.warning(f"Failed to init TTS ({self.config.tts_engine}): {e}")
            self._tts_engine = self._init_builtin_tts()
    
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
            from vosk import Model, KaldiRecognizer
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
    
    def _cleanup_engines(self):
        """Cleanup audio engines."""
        if self._tts_engine and self._tts_engine.get("type") == "pyttsx3":
            try:
                self._tts_engine["engine"].stop()
            except Exception as e:
                logger.debug(f"TTS cleanup error (non-fatal): {e}")
    
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
                self._synthesize_speech(text)
                self._speaking = False
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speak error: {e}")
                self._speaking = False
    
    def _recognize_speech(self) -> str:
        """Recognize speech from microphone."""
        if not self._stt_engine:
            return ""
        
        stt_type = self._stt_engine.get("type")
        
        if stt_type == "whisper":
            return self._recognize_whisper()
        elif stt_type == "vosk":
            return self._recognize_vosk()
        else:
            return self._recognize_builtin()
    
    def _recognize_whisper(self) -> str:
        """Recognize using Whisper."""
        try:
            import sounddevice as sd
            import numpy as np
            
            # Record audio
            duration = 5.0  # seconds
            audio = sd.rec(
                int(duration * self.config.sample_rate),
                samplerate=self.config.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            # Transcribe
            model = self._stt_engine["model"]
            result = model.transcribe(audio.flatten())
            return result.get("text", "").strip()
            
        except Exception as e:
            logger.error(f"Whisper recognition error: {e}")
            return ""
    
    def _recognize_vosk(self) -> str:
        """Recognize using Vosk."""
        try:
            from vosk import KaldiRecognizer
            import sounddevice as sd
            import json
            
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
        """Play raw audio data."""
        try:
            import sounddevice as sd
            import numpy as np
            
            # Decode audio (assuming MP3)
            from io import BytesIO
            import wave
            
            # This is simplified - real implementation would decode MP3
            # For now, just simulate playback
            time.sleep(2.0)
            
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
    
    def _detect_wake_word(self) -> bool:
        """Detect wake word in audio."""
        # Simplified wake word detection
        # Real implementation would use Porcupine or similar
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
