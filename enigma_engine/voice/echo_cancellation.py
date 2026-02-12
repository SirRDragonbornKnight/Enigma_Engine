"""
================================================================================
Echo Cancellation - Prevent feedback loops between speakers and microphone.
================================================================================

Multi-backend acoustic echo cancellation (AEC):
1. speexdsp library (best quality, C-based)
2. Adaptive filter (LMS/NLMS algorithm)
3. Simple cross-correlation subtraction (always available)

USAGE:
    from enigma_engine.voice.echo_cancellation import EchoCanceller
    
    canceller = EchoCanceller()
    
    # Process microphone input, removing speaker playback echo
    clean_mic = canceller.cancel_echo(mic_audio, speaker_audio, sample_rate=16000)
    
    # For real-time use, feed speaker audio as it plays
    canceller.feed_reference(speaker_audio, sample_rate=16000)
    clean_mic = canceller.process(mic_audio, sample_rate=16000)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)


class EchoCancellationBackend(Enum):
    """Available echo cancellation backends."""
    SPEEXDSP = auto()        # speexdsp-based AEC (best)
    ADAPTIVE_FILTER = auto()  # LMS/NLMS adaptive filter
    CROSS_CORRELATION = auto() # Simple cross-correlation subtraction
    AUTO = auto()             # Auto-select best available


@dataclass
class EchoCancellationConfig:
    """Configuration for echo cancellation."""
    
    # Backend selection
    backend: EchoCancellationBackend = EchoCancellationBackend.AUTO
    
    # Adaptive filter parameters
    filter_length: int = 1024        # Filter taps (longer = more echo delay handled)
    step_size: float = 0.1           # LMS step size (0.01-0.5, lower = more stable)
    leakage: float = 0.9999          # Leakage factor for stability
    
    # Cross-correlation parameters
    max_delay_ms: float = 500.0      # Maximum echo delay to search for
    correlation_threshold: float = 0.3  # Minimum correlation to apply cancellation
    
    # General
    frame_size: int = 256            # Processing frame size
    sample_rate: int = 16000         # Default sample rate
    reference_buffer_ms: float = 1000.0  # Reference audio buffer length


class EchoCanceller:
    """
    Acoustic echo cancellation for preventing speaker-to-mic feedback.
    
    In a typical setup:
    - Speaker plays TTS output
    - Microphone picks up both user speech AND speaker echo
    - Echo canceller removes the speaker echo from mic input
    """
    
    def __init__(self, config: EchoCancellationConfig = None):
        self.config = config or EchoCancellationConfig()
        
        # Reference audio buffer (speaker output)
        buffer_samples = int(self.config.reference_buffer_ms * self.config.sample_rate / 1000)
        self._reference_buffer: deque = deque(maxlen=buffer_samples)
        
        # Adaptive filter state
        self._filter_weights: np.ndarray | None = None
        self._filter_initialized = False
        
        # Detect available backends
        self._available_backends = self._detect_backends()
        
        # Select backend
        self._active_backend = self._select_backend()
        
        # Initialize backend-specific state
        self._init_backend()
        
        logger.info(f"EchoCanceller initialized with {self._active_backend.name} backend")
    
    def _detect_backends(self) -> list[EchoCancellationBackend]:
        """Detect which backends are available."""
        available = [EchoCancellationBackend.CROSS_CORRELATION]  # Always available
        
        # Adaptive filter always available (pure numpy)
        available.append(EchoCancellationBackend.ADAPTIVE_FILTER)
        
        try:
            import speexdsp  # type: ignore
            available.append(EchoCancellationBackend.SPEEXDSP)
        except ImportError:
            pass  # Intentionally silent
        
        return available
    
    def _select_backend(self) -> EchoCancellationBackend:
        """Select the best available backend."""
        if self.config.backend != EchoCancellationBackend.AUTO:
            if self.config.backend in self._available_backends:
                return self.config.backend
            logger.warning(
                f"Requested backend {self.config.backend.name} not available, "
                f"falling back to auto-select"
            )
        
        # Priority order: speexdsp > adaptive_filter > cross_correlation
        priority = [
            EchoCancellationBackend.SPEEXDSP,
            EchoCancellationBackend.ADAPTIVE_FILTER,
            EchoCancellationBackend.CROSS_CORRELATION
        ]
        
        for backend in priority:
            if backend in self._available_backends:
                return backend
        
        return EchoCancellationBackend.CROSS_CORRELATION
    
    def _init_backend(self):
        """Initialize backend-specific state."""
        if self._active_backend == EchoCancellationBackend.SPEEXDSP:
            self._init_speexdsp()
        elif self._active_backend == EchoCancellationBackend.ADAPTIVE_FILTER:
            self._init_adaptive_filter()
    
    def _init_speexdsp(self):
        """Initialize speexdsp echo canceller."""
        try:
            import speexdsp  # type: ignore
            self._aec = speexdsp.EchoCanceller(
                self.config.frame_size,
                self.config.filter_length,
                self.config.sample_rate
            )
            logger.debug("speexdsp AEC initialized")
        except Exception as e:
            logger.warning(f"Failed to init speexdsp: {e}, falling back")
            self._active_backend = EchoCancellationBackend.ADAPTIVE_FILTER
            self._init_adaptive_filter()
    
    def _init_adaptive_filter(self):
        """Initialize adaptive filter weights."""
        self._filter_weights = np.zeros(self.config.filter_length)
        self._filter_initialized = True
        logger.debug(f"Adaptive filter initialized with {self.config.filter_length} taps")
    
    def feed_reference(self, audio: np.ndarray, sample_rate: int = None):
        """
        Feed reference audio (speaker output) for echo cancellation.
        
        Call this whenever audio is played through speakers.
        
        Args:
            audio: Speaker audio being played
            sample_rate: Sample rate (resamples if different from config)
        """
        sample_rate = sample_rate or self.config.sample_rate
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0
        
        # Resample if needed
        if sample_rate != self.config.sample_rate:
            audio = self._resample(audio, sample_rate, self.config.sample_rate)
        
        # Add to reference buffer
        for sample in audio.flatten():
            self._reference_buffer.append(sample)
    
    def process(self, mic_audio: np.ndarray, sample_rate: int = None) -> np.ndarray:
        """
        Process microphone audio, removing echo from reference buffer.
        
        Args:
            mic_audio: Microphone input audio
            sample_rate: Sample rate
        
        Returns:
            Echo-cancelled audio
        """
        sample_rate = sample_rate or self.config.sample_rate
        
        # Get reference from buffer
        ref_len = len(mic_audio)
        if len(self._reference_buffer) < ref_len:
            # Not enough reference audio, return original
            return mic_audio
        
        # Get matching reference segment
        ref_audio = np.array(list(self._reference_buffer)[-ref_len:], dtype=np.float32)
        
        return self.cancel_echo(mic_audio, ref_audio, sample_rate)
    
    def cancel_echo(
        self,
        mic_audio: np.ndarray,
        speaker_audio: np.ndarray,
        sample_rate: int = None
    ) -> np.ndarray:
        """
        Cancel echo from microphone audio using speaker reference.
        
        Args:
            mic_audio: Microphone input (contains speech + echo)
            speaker_audio: Speaker output (the echo source)
            sample_rate: Sample rate
        
        Returns:
            Echo-cancelled microphone audio
        """
        sample_rate = sample_rate or self.config.sample_rate
        
        # Ensure float32
        if mic_audio.dtype != np.float32:
            mic_audio = mic_audio.astype(np.float32)
            if np.abs(mic_audio).max() > 1.0:
                mic_audio = mic_audio / 32768.0
        
        if speaker_audio.dtype != np.float32:
            speaker_audio = speaker_audio.astype(np.float32)
            if np.abs(speaker_audio).max() > 1.0:
                speaker_audio = speaker_audio / 32768.0
        
        # Ensure mono
        if len(mic_audio.shape) > 1:
            mic_audio = mic_audio.mean(axis=1)
        if len(speaker_audio.shape) > 1:
            speaker_audio = speaker_audio.mean(axis=1)
        
        # Route to backend
        if self._active_backend == EchoCancellationBackend.SPEEXDSP:
            return self._cancel_speexdsp(mic_audio, speaker_audio)
        elif self._active_backend == EchoCancellationBackend.ADAPTIVE_FILTER:
            return self._cancel_adaptive(mic_audio, speaker_audio)
        else:
            return self._cancel_cross_correlation(mic_audio, speaker_audio, sample_rate)
    
    def _cancel_speexdsp(
        self,
        mic_audio: np.ndarray,
        speaker_audio: np.ndarray
    ) -> np.ndarray:
        """Cancel echo using speexdsp."""
        try:
            # Convert to int16 for speexdsp
            mic_int16 = (mic_audio * 32767).astype(np.int16)
            spk_int16 = (speaker_audio * 32767).astype(np.int16)
            
            # Process in frames
            frame_size = self.config.frame_size
            output = np.zeros_like(mic_audio)
            
            for i in range(0, len(mic_audio) - frame_size + 1, frame_size):
                mic_frame = mic_int16[i:i + frame_size]
                spk_frame = spk_int16[i:i + frame_size]
                
                # Echo cancellation
                out_frame = self._aec.process(mic_frame.tobytes(), spk_frame.tobytes())
                output[i:i + frame_size] = np.frombuffer(out_frame, dtype=np.int16) / 32767.0
            
            return output.astype(np.float32)
            
        except Exception as e:
            logger.error(f"speexdsp AEC failed: {e}, falling back")
            return self._cancel_adaptive(mic_audio, speaker_audio)
    
    def _cancel_adaptive(
        self,
        mic_audio: np.ndarray,
        speaker_audio: np.ndarray
    ) -> np.ndarray:
        """
        Cancel echo using NLMS (Normalized Least Mean Squares) adaptive filter.
        
        The adaptive filter learns the acoustic path from speaker to mic
        and subtracts the estimated echo.
        """
        if not self._filter_initialized:
            self._init_adaptive_filter()
        
        n_samples = len(mic_audio)
        filter_len = self.config.filter_length
        mu = self.config.step_size
        leakage = self.config.leakage
        
        # Ensure speaker audio is long enough
        if len(speaker_audio) < n_samples:
            speaker_audio = np.pad(speaker_audio, (0, n_samples - len(speaker_audio)))
        
        # Output buffer
        output = np.zeros(n_samples, dtype=np.float32)
        
        # Process sample by sample
        w = self._filter_weights.copy()
        
        for n in range(n_samples):
            # Build reference vector (current and past speaker samples)
            if n >= filter_len:
                x = speaker_audio[n - filter_len + 1:n + 1][::-1]
            else:
                x = np.zeros(filter_len)
                x[:n + 1] = speaker_audio[:n + 1][::-1]
            
            # Estimate echo
            echo_estimate = np.dot(w, x)
            
            # Error (desired signal = mic input, we want to remove echo)
            error = mic_audio[n] - echo_estimate
            
            # NLMS update
            norm = np.dot(x, x) + 1e-10  # Avoid division by zero
            w = leakage * w + (mu / norm) * error * x
            
            # Output is the error (mic minus estimated echo)
            output[n] = error
        
        # Save updated weights for next call
        self._filter_weights = w
        
        return output
    
    def _cancel_cross_correlation(
        self,
        mic_audio: np.ndarray,
        speaker_audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Simple echo cancellation using cross-correlation.
        
        Finds the delay and scaling of the echo, then subtracts it.
        Less accurate than adaptive methods but simple and robust.
        """
        # Maximum delay in samples
        max_delay = int(self.config.max_delay_ms * sample_rate / 1000)
        
        # Ensure matching lengths
        min_len = min(len(mic_audio), len(speaker_audio))
        mic = mic_audio[:min_len]
        spk = speaker_audio[:min_len]
        
        # Find correlation at different delays
        best_corr = 0.0
        best_delay = 0
        best_scale = 0.0
        
        for delay in range(0, min(max_delay, min_len // 2)):
            if delay > 0:
                mic_segment = mic[delay:]
                spk_segment = spk[:-delay]
            else:
                mic_segment = mic
                spk_segment = spk
            
            # Normalize
            mic_norm = np.sqrt(np.sum(mic_segment ** 2) + 1e-10)
            spk_norm = np.sqrt(np.sum(spk_segment ** 2) + 1e-10)
            
            # Correlation
            corr = np.abs(np.sum(mic_segment * spk_segment)) / (mic_norm * spk_norm)
            
            if corr > best_corr:
                best_corr = corr
                best_delay = delay
                # Estimate scaling factor
                best_scale = np.sum(mic_segment * spk_segment) / (np.sum(spk_segment ** 2) + 1e-10)
        
        # Apply cancellation if correlation is significant
        if best_corr > self.config.correlation_threshold:
            logger.debug(f"Echo detected: delay={best_delay}, corr={best_corr:.3f}, scale={best_scale:.3f}")
            
            output = mic_audio.copy()
            if best_delay > 0:
                # Subtract delayed and scaled speaker signal
                output[best_delay:] -= best_scale * speaker_audio[:-best_delay]
            else:
                output -= best_scale * speaker_audio[:len(output)]
            
            return output.astype(np.float32)
        
        # No significant echo detected
        return mic_audio
    
    def _resample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample audio to target rate."""
        if from_rate == to_rate:
            return audio
        
        try:
            from scipy import signal as sig
            num_samples = int(len(audio) * to_rate / from_rate)
            return sig.resample(audio, num_samples).astype(np.float32)
        except ImportError:
            # Simple linear interpolation fallback
            ratio = to_rate / from_rate
            new_len = int(len(audio) * ratio)
            indices = np.arange(new_len) / ratio
            indices = np.clip(indices, 0, len(audio) - 1)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    
    def reset(self):
        """Reset echo canceller state."""
        self._reference_buffer.clear()
        if self._active_backend == EchoCancellationBackend.ADAPTIVE_FILTER:
            self._filter_weights = np.zeros(self.config.filter_length)
        logger.debug("Echo canceller state reset")
    
    @property
    def backend_name(self) -> str:
        """Get the name of the active backend."""
        return self._active_backend.name


# Convenience function
def cancel_echo(
    mic_audio: np.ndarray,
    speaker_audio: np.ndarray,
    sample_rate: int = 16000,
    config: EchoCancellationConfig = None
) -> np.ndarray:
    """
    Convenience function to cancel echo from microphone audio.
    
    Args:
        mic_audio: Microphone input array
        speaker_audio: Speaker output array (echo source)
        sample_rate: Sample rate in Hz
        config: Optional configuration
    
    Returns:
        Echo-cancelled audio array
    """
    canceller = EchoCanceller(config)
    return canceller.cancel_echo(mic_audio, speaker_audio, sample_rate)
