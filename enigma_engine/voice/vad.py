"""
Voice Activity Detection (VAD) for Enigma AI Engine.

Detects speech vs silence/noise using multiple backends.
Supports: WebRTC VAD, Silero VAD, energy-based detection.

Usage:
    from enigma_engine.voice.vad import VAD, VADConfig
    
    vad = VAD()
    is_speech = vad.is_speech(audio_chunk)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class VADBackend(Enum):
    """Available VAD backends."""
    WEBRTC = "webrtc"
    SILERO = "silero"
    ENERGY = "energy"
    AUTO = "auto"


@dataclass
class VADConfig:
    """VAD configuration."""
    backend: VADBackend = VADBackend.AUTO
    sample_rate: int = 16000
    frame_duration_ms: int = 30
    aggressiveness: int = 2  # 0-3, higher = more aggressive filtering
    energy_threshold: float = 0.01
    speech_pad_ms: int = 300  # Padding around speech
    min_speech_ms: int = 250  # Minimum speech duration
    max_silence_ms: int = 500  # Max silence before end of speech


class VAD:
    """Voice Activity Detection with multiple backends."""
    
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self._backend = None
        self._backend_name = None
        self._init_backend()
        
        # State for speech segmentation
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False
    
    def _init_backend(self) -> None:
        """Initialize the best available backend."""
        if self.config.backend == VADBackend.AUTO:
            # Try backends in order of quality
            for backend in [VADBackend.SILERO, VADBackend.WEBRTC, VADBackend.ENERGY]:
                if self._try_init_backend(backend):
                    return
        else:
            if not self._try_init_backend(self.config.backend):
                logger.warning(f"Failed to init {self.config.backend}, falling back to energy")
                self._try_init_backend(VADBackend.ENERGY)
    
    def _try_init_backend(self, backend: VADBackend) -> bool:
        """Try to initialize a specific backend."""
        try:
            if backend == VADBackend.SILERO:
                return self._init_silero()
            elif backend == VADBackend.WEBRTC:
                return self._init_webrtc()
            elif backend == VADBackend.ENERGY:
                return self._init_energy()
        except Exception as e:
            logger.debug(f"Backend {backend} init failed: {e}")
        return False
    
    def _init_silero(self) -> bool:
        """Initialize Silero VAD (best quality)."""
        try:
            import torch
            model, utils = torch.hub.load(
                'snakers4/silero-vad', 'silero_vad',
                force_reload=False, trust_repo=True
            )
            self._backend = model
            self._backend_name = "silero"
            self._get_speech_timestamps = utils[0]
            logger.info("Using Silero VAD backend")
            return True
        except Exception:
            return False
    
    def _init_webrtc(self) -> bool:
        """Initialize WebRTC VAD (fast, good quality)."""
        try:
            import webrtcvad
            self._backend = webrtcvad.Vad(self.config.aggressiveness)
            self._backend_name = "webrtc"
            logger.info("Using WebRTC VAD backend")
            return True
        except ImportError:
            return False
    
    def _init_energy(self) -> bool:
        """Initialize energy-based VAD (fallback)."""
        self._backend = "energy"
        self._backend_name = "energy"
        self._noise_floor = self.config.energy_threshold
        self._adapting = True
        self._frame_count = 0
        logger.info("Using energy-based VAD backend")
        return True
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Check if audio chunk contains speech.
        
        Args:
            audio: Audio samples (float32 or int16)
        
        Returns:
            True if speech detected
        """
        if self._backend_name == "silero":
            return self._silero_detect(audio)
        elif self._backend_name == "webrtc":
            return self._webrtc_detect(audio)
        else:
            return self._energy_detect(audio)
    
    def _silero_detect(self, audio: np.ndarray) -> bool:
        """Detect speech using Silero VAD."""
        try:
            import torch
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / 32768.0
            tensor = torch.from_numpy(audio)
            confidence = self._backend(tensor, self.config.sample_rate).item()
            return confidence > 0.5
        except Exception as e:
            logger.debug(f"Silero error: {e}")
            return self._energy_detect(audio)
    
    def _webrtc_detect(self, audio: np.ndarray) -> bool:
        """Detect speech using WebRTC VAD."""
        try:
            # WebRTC needs int16
            if audio.dtype == np.float32:
                audio = (audio * 32768).astype(np.int16)
            return self._backend.is_speech(audio.tobytes(), self.config.sample_rate)
        except Exception as e:
            logger.debug(f"WebRTC error: {e}")
            return self._energy_detect(audio)
    
    def _energy_detect(self, audio: np.ndarray) -> bool:
        """Detect speech using energy threshold."""
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        energy = np.sqrt(np.mean(audio ** 2))
        
        # Adaptive noise floor
        if self._adapting and self._frame_count < 30:
            self._noise_floor = 0.9 * self._noise_floor + 0.1 * energy
            self._frame_count += 1
        
        threshold = max(self._noise_floor * 2, self.config.energy_threshold)
        return energy > threshold
    
    def process_stream(self, audio: np.ndarray) -> list[tuple]:
        """
        Process audio stream and return speech segments.
        
        Args:
            audio: Full audio array
        
        Returns:
            List of (start_sample, end_sample) tuples
        """
        frame_size = int(self.config.sample_rate * self.config.frame_duration_ms / 1000)
        segments = []
        current_start = None
        
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            is_speech = self.is_speech(frame)
            
            if is_speech:
                self._speech_frames += 1
                self._silence_frames = 0
                if not self._in_speech:
                    min_frames = self.config.min_speech_ms / self.config.frame_duration_ms
                    if self._speech_frames >= min_frames:
                        self._in_speech = True
                        # Pad start
                        pad_frames = self.config.speech_pad_ms / self.config.frame_duration_ms
                        current_start = max(0, i - int(pad_frames * frame_size))
            else:
                self._silence_frames += 1
                if self._in_speech:
                    max_silence = self.config.max_silence_ms / self.config.frame_duration_ms
                    if self._silence_frames >= max_silence:
                        # End of speech segment
                        pad_frames = self.config.speech_pad_ms / self.config.frame_duration_ms
                        end = min(len(audio), i + int(pad_frames * frame_size))
                        segments.append((current_start, end))
                        self._in_speech = False
                        self._speech_frames = 0
                        current_start = None
        
        # Handle ongoing speech at end
        if self._in_speech and current_start is not None:
            segments.append((current_start, len(audio)))
        
        return segments
    
    def reset(self) -> None:
        """Reset state."""
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False
        if self._backend_name == "energy":
            self._adapting = True
            self._frame_count = 0


def get_vad(config: Optional[VADConfig] = None) -> VAD:
    """Get a VAD instance."""
    return VAD(config)
