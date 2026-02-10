"""
================================================================================
Noise Reduction - Filter background noise from audio input.
================================================================================

Multi-backend noise reduction with automatic fallback:
1. noisereduce library (spectral gating, best quality)
2. scipy spectral subtraction (good fallback)
3. Simple energy-based filtering (always available)

USAGE:
    from enigma_engine.voice.noise_reduction import NoiseReducer
    
    reducer = NoiseReducer()
    
    # Reduce noise from audio
    clean_audio = reducer.reduce_noise(noisy_audio, sample_rate=16000)
    
    # With noise profile learned from sample
    reducer.learn_noise_profile(noise_sample, sample_rate=16000)
    clean_audio = reducer.reduce_noise(noisy_audio, sample_rate=16000)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)


class NoiseReductionBackend(Enum):
    """Available noise reduction backends."""
    NOISEREDUCE = auto()      # noisereduce library (best quality)
    SPECTRAL_SUBTRACT = auto()  # scipy spectral subtraction
    ENERGY_GATE = auto()       # Simple energy-based gating
    AUTO = auto()              # Auto-select best available


@dataclass
class NoiseReductionConfig:
    """Configuration for noise reduction."""
    
    # Backend selection
    backend: NoiseReductionBackend = NoiseReductionBackend.AUTO
    
    # Spectral gating parameters (for noisereduce)
    stationary: bool = True        # Assume stationary noise
    prop_decrease: float = 1.0     # Proportion to reduce noise by (1.0 = 100%)
    time_constant_s: float = 2.0   # Time constant for noise estimation
    freq_mask_smooth_hz: int = 500  # Frequency smoothing
    time_mask_smooth_ms: int = 50   # Time smoothing
    
    # Spectral subtraction parameters
    alpha: float = 2.0             # Over-subtraction factor
    beta: float = 0.01             # Spectral floor
    
    # Energy gate parameters
    threshold_db: float = -40.0    # Energy threshold in dB
    attack_ms: float = 5.0         # Attack time
    release_ms: float = 50.0       # Release time
    
    # General
    n_fft: int = 2048              # FFT size for spectral methods
    hop_length: int = 512          # Hop length for STFT


class NoiseReducer:
    """
    Multi-backend noise reduction for audio input.
    
    Automatically selects the best available backend and falls back
    gracefully if libraries aren't installed.
    """
    
    def __init__(self, config: NoiseReductionConfig = None):
        self.config = config or NoiseReductionConfig()
        
        # Noise profile for spectral subtraction
        self._noise_profile: np.ndarray | None = None
        self._noise_energy: float = 0.0
        
        # Detect available backends
        self._available_backends = self._detect_backends()
        
        # Select backend
        self._active_backend = self._select_backend()
        
        logger.info(f"NoiseReducer initialized with {self._active_backend.name} backend")
    
    def _detect_backends(self) -> list[NoiseReductionBackend]:
        """Detect which backends are available."""
        available = [NoiseReductionBackend.ENERGY_GATE]  # Always available
        
        try:
            import noisereduce  # type: ignore
            available.append(NoiseReductionBackend.NOISEREDUCE)
        except ImportError:
            pass
        
        try:
            from scipy import signal  # type: ignore
            available.append(NoiseReductionBackend.SPECTRAL_SUBTRACT)
        except ImportError:
            pass
        
        return available
    
    def _select_backend(self) -> NoiseReductionBackend:
        """Select the best available backend."""
        if self.config.backend != NoiseReductionBackend.AUTO:
            if self.config.backend in self._available_backends:
                return self.config.backend
            logger.warning(
                f"Requested backend {self.config.backend.name} not available, "
                f"falling back to auto-select"
            )
        
        # Priority order: noisereduce > spectral_subtract > energy_gate
        priority = [
            NoiseReductionBackend.NOISEREDUCE,
            NoiseReductionBackend.SPECTRAL_SUBTRACT,
            NoiseReductionBackend.ENERGY_GATE
        ]
        
        for backend in priority:
            if backend in self._available_backends:
                return backend
        
        return NoiseReductionBackend.ENERGY_GATE
    
    def reduce_noise(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        noise_sample: np.ndarray = None
    ) -> np.ndarray:
        """
        Reduce noise from audio.
        
        Args:
            audio: Input audio as numpy array (mono, float32 or int16)
            sample_rate: Audio sample rate in Hz
            noise_sample: Optional noise-only sample for better reduction
        
        Returns:
            Cleaned audio as numpy array
        """
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0  # Normalize int16
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Route to appropriate backend
        if self._active_backend == NoiseReductionBackend.NOISEREDUCE:
            return self._reduce_noisereduce(audio, sample_rate, noise_sample)
        elif self._active_backend == NoiseReductionBackend.SPECTRAL_SUBTRACT:
            return self._reduce_spectral_subtract(audio, sample_rate, noise_sample)
        else:
            return self._reduce_energy_gate(audio, sample_rate)
    
    def learn_noise_profile(
        self,
        noise_sample: np.ndarray,
        sample_rate: int = 16000
    ):
        """
        Learn a noise profile from a noise-only sample.
        
        This improves noise reduction quality by using a known
        noise reference instead of estimating from the signal.
        
        Args:
            noise_sample: Audio sample containing only noise
            sample_rate: Sample rate in Hz
        """
        # Ensure float32
        if noise_sample.dtype != np.float32:
            noise_sample = noise_sample.astype(np.float32)
            if noise_sample.max() > 1.0:
                noise_sample = noise_sample / 32768.0
        
        # Compute noise spectrum
        try:
            from scipy import signal as sig

            # Compute spectrogram of noise
            _, _, Sxx = sig.spectrogram(
                noise_sample,
                fs=sample_rate,
                nperseg=self.config.n_fft,
                noverlap=self.config.n_fft - self.config.hop_length
            )
            
            # Average noise spectrum
            self._noise_profile = np.mean(Sxx, axis=1)
            self._noise_energy = np.mean(noise_sample ** 2)
            
            logger.info(f"Learned noise profile with {len(self._noise_profile)} frequency bins")
            
        except ImportError:
            # Fallback: just store energy
            self._noise_energy = np.mean(noise_sample ** 2)
            logger.info("Learned noise energy level (scipy not available for full profile)")
    
    def _reduce_noisereduce(
        self,
        audio: np.ndarray,
        sample_rate: int,
        noise_sample: np.ndarray = None
    ) -> np.ndarray:
        """Reduce noise using noisereduce library (spectral gating)."""
        try:
            import noisereduce as nr  # type: ignore
            
            reduced = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                y_noise=noise_sample,
                stationary=self.config.stationary,
                prop_decrease=self.config.prop_decrease,
                time_constant_s=self.config.time_constant_s,
                freq_mask_smooth_hz=self.config.freq_mask_smooth_hz,
                time_mask_smooth_ms=self.config.time_mask_smooth_ms,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )
            
            return reduced.astype(np.float32)
            
        except Exception as e:
            logger.error(f"noisereduce failed: {e}, falling back")
            return self._reduce_spectral_subtract(audio, sample_rate, noise_sample)
    
    def _reduce_spectral_subtract(
        self,
        audio: np.ndarray,
        sample_rate: int,
        noise_sample: np.ndarray = None
    ) -> np.ndarray:
        """Reduce noise using spectral subtraction."""
        try:
            from scipy import signal as sig

            # STFT
            f, t, Zxx = sig.stft(
                audio,
                fs=sample_rate,
                nperseg=self.config.n_fft,
                noverlap=self.config.n_fft - self.config.hop_length
            )
            
            # Magnitude and phase
            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)
            
            # Estimate or use noise spectrum
            if noise_sample is not None:
                # Use provided noise sample
                _, _, Nxx = sig.stft(
                    noise_sample,
                    fs=sample_rate,
                    nperseg=self.config.n_fft,
                    noverlap=self.config.n_fft - self.config.hop_length
                )
                noise_mag = np.mean(np.abs(Nxx), axis=1, keepdims=True)
            elif self._noise_profile is not None:
                # Use learned profile
                noise_mag = self._noise_profile.reshape(-1, 1)
            else:
                # Estimate from first few frames (assume start is noise)
                n_noise_frames = min(10, magnitude.shape[1] // 4)
                noise_mag = np.mean(magnitude[:, :n_noise_frames], axis=1, keepdims=True)
            
            # Spectral subtraction with over-subtraction
            alpha = self.config.alpha
            beta = self.config.beta
            
            # Subtract noise spectrum
            magnitude_clean = magnitude ** 2 - alpha * (noise_mag ** 2)
            
            # Apply spectral floor
            magnitude_clean = np.maximum(magnitude_clean, beta * (noise_mag ** 2))
            
            # Take square root
            magnitude_clean = np.sqrt(magnitude_clean)
            
            # Reconstruct with original phase
            Zxx_clean = magnitude_clean * np.exp(1j * phase)
            
            # Inverse STFT
            _, audio_clean = sig.istft(
                Zxx_clean,
                fs=sample_rate,
                nperseg=self.config.n_fft,
                noverlap=self.config.n_fft - self.config.hop_length
            )
            
            # Match original length
            if len(audio_clean) > len(audio):
                audio_clean = audio_clean[:len(audio)]
            elif len(audio_clean) < len(audio):
                audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))
            
            return audio_clean.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Spectral subtraction failed: {e}, falling back")
            return self._reduce_energy_gate(audio, sample_rate)
    
    def _reduce_energy_gate(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Simple energy-based noise gate.
        
        This is the fallback method that works without any dependencies.
        It gates audio below a threshold with smooth attack/release.
        """
        # Frame-based processing
        frame_size = int(sample_rate * 0.02)  # 20ms frames
        n_frames = len(audio) // frame_size
        
        # Calculate frame energies
        frames = audio[:n_frames * frame_size].reshape(-1, frame_size)
        energies = np.mean(frames ** 2, axis=1)
        
        # Convert threshold to linear
        threshold_linear = 10 ** (self.config.threshold_db / 20)
        
        # If we have learned noise energy, use it
        if self._noise_energy > 0:
            threshold_linear = max(threshold_linear, np.sqrt(self._noise_energy) * 2)
        
        # Calculate gate envelope
        attack_frames = max(1, int(self.config.attack_ms * sample_rate / 1000 / frame_size))
        release_frames = max(1, int(self.config.release_ms * sample_rate / 1000 / frame_size))
        
        # Gate state
        envelope = np.zeros(n_frames)
        gate_open = 0.0
        
        for i in range(n_frames):
            if np.sqrt(energies[i]) > threshold_linear:
                # Attack
                gate_open = min(1.0, gate_open + 1.0 / attack_frames)
            else:
                # Release
                gate_open = max(0.0, gate_open - 1.0 / release_frames)
            envelope[i] = gate_open
        
        # Smooth envelope
        envelope = np.repeat(envelope, frame_size)
        
        # Handle remaining samples
        if len(envelope) < len(audio):
            envelope = np.pad(envelope, (0, len(audio) - len(envelope)), 
                            constant_values=envelope[-1] if len(envelope) > 0 else 0)
        elif len(envelope) > len(audio):
            envelope = envelope[:len(audio)]
        
        # Apply gate
        return (audio * envelope).astype(np.float32)
    
    def reset(self):
        """Reset noise profiles and state."""
        self._noise_profile = None
        self._noise_energy = 0.0
        logger.debug("Noise reducer state reset")
    
    @property
    def backend_name(self) -> str:
        """Get the name of the active backend."""
        return self._active_backend.name


# Convenience function
def reduce_noise(
    audio: np.ndarray,
    sample_rate: int = 16000,
    noise_sample: np.ndarray = None,
    config: NoiseReductionConfig = None
) -> np.ndarray:
    """
    Convenience function to reduce noise from audio.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate in Hz
        noise_sample: Optional noise-only sample
        config: Optional configuration
    
    Returns:
        Cleaned audio array
    """
    reducer = NoiseReducer(config)
    return reducer.reduce_noise(audio, sample_rate, noise_sample)
