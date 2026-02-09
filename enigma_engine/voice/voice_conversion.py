"""
Voice Conversion for Enigma AI Engine

Convert voice from one speaker to another.

Features:
- Speaker embedding extraction
- Voice style transfer
- Real-time conversion
- Pitch and timbre control

Usage:
    from enigma_engine.voice.voice_conversion import VoiceConverter
    
    converter = VoiceConverter()
    
    # Register target voice
    converter.register_voice("target", "samples/target_voice.wav")
    
    # Convert audio
    converted = converter.convert(
        source_audio="input.wav",
        target_voice="target"
    )
    converted.save("output.wav")
"""

import hashlib
import json
import logging
import os
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ConversionMethod(Enum):
    """Voice conversion methods."""
    PITCH_SHIFT = "pitch_shift"  # Basic pitch shifting
    SPEC_MAPPING = "spec_mapping"  # Spectral mapping
    NEURAL = "neural"  # Deep learning based
    PPGS = "ppgs"  # Phonetic features
    DIFF = "diff"  # Diffusion-based


@dataclass
class VoiceProfile:
    """Voice profile for conversion."""
    id: str
    name: str
    embedding: Optional[np.ndarray] = None  # Speaker embedding
    reference_audio: List[str] = field(default_factory=list)
    
    # Voice characteristics (extracted or estimated)
    pitch_mean: float = 0.0  # Mean F0
    pitch_std: float = 0.0  # F0 standard deviation
    energy_mean: float = 0.0
    timbre_features: Optional[np.ndarray] = None  # MFCCs or similar


@dataclass
class ConvertedAudio:
    """Converted audio data."""
    audio_data: np.ndarray
    sample_rate: int = 22050
    source_voice: str = ""
    target_voice: str = ""
    duration: float = 0.0
    
    def save(self, path: str):
        """Save to WAV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Normalize to int16
        audio_int16 = (self.audio_data * 32767).astype(np.int16)
        
        with wave.open(str(path), 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_int16.tobytes())
        
        return str(path)


class AudioProcessor:
    """Audio processing utilities."""
    
    @staticmethod
    def load_audio(path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        with wave.open(path, 'r') as wav:
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            n_channels = wav.getnchannels()
            audio_bytes = wav.readframes(n_frames)
            
            # Convert to float
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Handle stereo
            if n_channels == 2:
                audio_int16 = audio_int16.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
            audio_float = audio_int16.astype(np.float32) / 32768.0
            
            # Resample if needed
            if sample_rate != target_sr:
                audio_float = AudioProcessor.resample(audio_float, sample_rate, target_sr)
                sample_rate = target_sr
            
            return audio_float, sample_rate
    
    @staticmethod
    def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Resample audio."""
        if src_rate == dst_rate:
            return audio
        
        ratio = dst_rate / src_rate
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)
    
    @staticmethod
    def compute_stft(
        audio: np.ndarray,
        n_fft: int = 1024,
        hop_length: int = 256,
        window: str = "hann"
    ) -> np.ndarray:
        """Compute Short-Time Fourier Transform."""
        # Create window
        if window == "hann":
            win = np.hanning(n_fft)
        else:
            win = np.ones(n_fft)
        
        # Pad audio
        audio_padded = np.pad(audio, (n_fft // 2, n_fft // 2), mode='reflect')
        
        # Compute frames
        n_frames = 1 + (len(audio_padded) - n_fft) // hop_length
        stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
        
        for i in range(n_frames):
            start = i * hop_length
            frame = audio_padded[start:start + n_fft] * win
            spectrum = np.fft.rfft(frame)
            stft[:, i] = spectrum
        
        return stft
    
    @staticmethod
    def compute_istft(
        stft: np.ndarray,
        hop_length: int = 256,
        window: str = "hann"
    ) -> np.ndarray:
        """Inverse STFT."""
        n_fft = (stft.shape[0] - 1) * 2
        n_frames = stft.shape[1]
        
        if window == "hann":
            win = np.hanning(n_fft)
        else:
            win = np.ones(n_fft)
        
        output_length = n_fft + hop_length * (n_frames - 1)
        audio = np.zeros(output_length)
        window_sum = np.zeros(output_length)
        
        for i in range(n_frames):
            frame = np.fft.irfft(stft[:, i], n=n_fft)
            start = i * hop_length
            audio[start:start + n_fft] += frame * win
            window_sum[start:start + n_fft] += win ** 2
        
        # Normalize
        window_sum = np.maximum(window_sum, 1e-8)
        audio /= window_sum
        
        # Trim padding
        audio = audio[n_fft // 2:-(n_fft // 2)]
        
        return audio
    
    @staticmethod
    def estimate_f0(audio: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
        """Estimate fundamental frequency (pitch)."""
        frame_length = int(0.025 * sample_rate)  # 25ms
        hop_length = int(0.010 * sample_rate)  # 10ms
        
        n_frames = 1 + (len(audio) - frame_length) // hop_length
        f0 = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            
            # Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr) // 2:]
            
            # Find peak
            min_lag = int(sample_rate / 500)  # Max F0 = 500 Hz
            max_lag = int(sample_rate / 50)   # Min F0 = 50 Hz
            
            if max_lag > len(corr):
                max_lag = len(corr) - 1
            
            if min_lag < max_lag:
                segment = corr[min_lag:max_lag]
                if len(segment) > 0 and np.max(segment) > 0.1 * corr[0]:
                    peak = min_lag + np.argmax(segment)
                    f0[i] = sample_rate / peak
        
        return f0
    
    @staticmethod
    def compute_mfcc(
        audio: np.ndarray,
        sample_rate: int = 22050,
        n_mfcc: int = 13
    ) -> np.ndarray:
        """Compute Mel-frequency cepstral coefficients."""
        # Compute spectrogram
        stft = AudioProcessor.compute_stft(audio)
        power_spec = np.abs(stft) ** 2
        
        # Mel filterbank
        n_filters = 40
        n_fft = (stft.shape[0] - 1) * 2
        
        mel_filters = AudioProcessor._create_mel_filterbank(
            sample_rate, n_fft, n_filters
        )
        
        # Apply filterbank
        mel_spec = np.dot(mel_filters, power_spec)
        
        # Log and DCT
        mel_spec = np.maximum(mel_spec, 1e-10)
        log_mel = np.log(mel_spec)
        
        # DCT (simplified)
        mfcc = np.zeros((n_mfcc, log_mel.shape[1]))
        for i in range(n_mfcc):
            for j in range(log_mel.shape[0]):
                mfcc[i] += log_mel[j] * np.cos(np.pi * i * (j + 0.5) / log_mel.shape[0])
        
        return mfcc
    
    @staticmethod
    def _create_mel_filterbank(
        sample_rate: int,
        n_fft: int,
        n_filters: int
    ) -> np.ndarray:
        """Create mel filterbank."""
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)
        
        low_mel = hz_to_mel(0)
        high_mel = hz_to_mel(sample_rate / 2)
        
        mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
        hz_points = mel_to_hz(mel_points)
        
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
        
        filters = np.zeros((n_filters, n_fft // 2 + 1))
        
        for i in range(n_filters):
            for j in range(bin_points[i], bin_points[i + 1]):
                filters[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
            for j in range(bin_points[i + 1], bin_points[i + 2]):
                filters[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])
        
        return filters


class VoiceConverter:
    """Voice conversion system."""
    
    def __init__(
        self,
        method: ConversionMethod = ConversionMethod.PITCH_SHIFT,
        cache_dir: str = "cache/voice_conversion"
    ):
        """
        Initialize voice converter.
        
        Args:
            method: Conversion method
            cache_dir: Cache directory
        """
        self.method = method
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Voice profiles
        self._voices: Dict[str, VoiceProfile] = {}
        
        # Audio processor
        self.processor = AudioProcessor()
        
        logger.info(f"VoiceConverter initialized with method: {method.value}")
    
    def register_voice(
        self,
        voice_id: str,
        reference_audio: Union[str, List[str]],
        name: Optional[str] = None
    ) -> VoiceProfile:
        """
        Register a voice for conversion.
        
        Args:
            voice_id: Unique voice ID
            reference_audio: Reference audio file(s)
            name: Display name
            
        Returns:
            Created voice profile
        """
        if isinstance(reference_audio, str):
            reference_audio = [reference_audio]
        
        # Analyze reference audio
        pitch_values = []
        energy_values = []
        mfcc_list = []
        
        for audio_path in reference_audio:
            if not os.path.exists(audio_path):
                logger.warning(f"Reference audio not found: {audio_path}")
                continue
            
            audio, sr = self.processor.load_audio(audio_path)
            
            # Extract F0
            f0 = self.processor.estimate_f0(audio, sr)
            voiced_f0 = f0[f0 > 0]
            if len(voiced_f0) > 0:
                pitch_values.extend(voiced_f0.tolist())
            
            # Extract energy
            energy = np.sqrt(np.mean(audio ** 2))
            energy_values.append(energy)
            
            # Extract MFCCs
            mfcc = self.processor.compute_mfcc(audio, sr)
            mfcc_list.append(np.mean(mfcc, axis=1))
        
        # Compute statistics
        pitch_mean = np.mean(pitch_values) if pitch_values else 0.0
        pitch_std = np.std(pitch_values) if pitch_values else 0.0
        energy_mean = np.mean(energy_values) if energy_values else 0.0
        timbre_features = np.mean(mfcc_list, axis=0) if mfcc_list else None
        
        profile = VoiceProfile(
            id=voice_id,
            name=name or voice_id,
            reference_audio=reference_audio,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            energy_mean=energy_mean,
            timbre_features=timbre_features
        )
        
        self._voices[voice_id] = profile
        
        logger.info(f"Registered voice: {voice_id} (F0={pitch_mean:.1f}Hz)")
        return profile
    
    def get_voice(self, voice_id: str) -> Optional[VoiceProfile]:
        """Get voice profile."""
        return self._voices.get(voice_id)
    
    def list_voices(self) -> List[str]:
        """List registered voices."""
        return list(self._voices.keys())
    
    def convert(
        self,
        source_audio: Union[str, np.ndarray],
        target_voice: str,
        source_voice: Optional[str] = None,
        preserve_energy: bool = True,
        pitch_shift_only: bool = False
    ) -> ConvertedAudio:
        """
        Convert voice.
        
        Args:
            source_audio: Source audio file or array
            target_voice: Target voice ID
            source_voice: Source voice ID (for analysis)
            preserve_energy: Preserve original energy
            pitch_shift_only: Only shift pitch, no timbre
            
        Returns:
            Converted audio
        """
        # Load source audio
        if isinstance(source_audio, str):
            audio, sr = self.processor.load_audio(source_audio)
            source_name = os.path.basename(source_audio)
        else:
            audio = source_audio
            sr = 22050
            source_name = "array"
        
        # Get target profile
        target = self._voices.get(target_voice)
        if not target:
            raise ValueError(f"Target voice {target_voice} not found")
        
        # Analyze source
        if source_voice and source_voice in self._voices:
            source = self._voices[source_voice]
        else:
            # Estimate source characteristics
            f0 = self.processor.estimate_f0(audio, sr)
            voiced_f0 = f0[f0 > 0]
            source_pitch = np.mean(voiced_f0) if len(voiced_f0) > 0 else 200.0
            source = VoiceProfile(
                id="source",
                name="source",
                pitch_mean=source_pitch
            )
        
        # Apply conversion based on method
        if self.method == ConversionMethod.PITCH_SHIFT or pitch_shift_only:
            converted = self._pitch_shift_convert(audio, sr, source, target)
        elif self.method == ConversionMethod.SPEC_MAPPING:
            converted = self._spectral_convert(audio, sr, source, target)
        else:
            # Default to pitch shift
            converted = self._pitch_shift_convert(audio, sr, source, target)
        
        # Preserve energy
        if preserve_energy:
            original_energy = np.sqrt(np.mean(audio ** 2))
            converted_energy = np.sqrt(np.mean(converted ** 2))
            if converted_energy > 0:
                converted = converted * (original_energy / converted_energy)
        
        return ConvertedAudio(
            audio_data=converted,
            sample_rate=sr,
            source_voice=source.id,
            target_voice=target.id,
            duration=len(converted) / sr
        )
    
    def _pitch_shift_convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        source: VoiceProfile,
        target: VoiceProfile
    ) -> np.ndarray:
        """Convert using pitch shifting."""
        # Calculate pitch ratio
        if source.pitch_mean > 0 and target.pitch_mean > 0:
            pitch_ratio = target.pitch_mean / source.pitch_mean
        else:
            pitch_ratio = 1.0
        
        # Limit ratio to reasonable range
        pitch_ratio = np.clip(pitch_ratio, 0.5, 2.0)
        
        if abs(pitch_ratio - 1.0) < 0.01:
            return audio
        
        # PSOLA-like pitch shifting
        converted = self._psola_pitch_shift(audio, sample_rate, pitch_ratio)
        
        return converted
    
    def _psola_pitch_shift(
        self,
        audio: np.ndarray,
        sample_rate: int,
        ratio: float
    ) -> np.ndarray:
        """PSOLA-based pitch shifting."""
        # Simplified TD-PSOLA
        # Find pitch periods
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)
        
        # Compute output length
        output_length = int(len(audio) / ratio)
        output = np.zeros(output_length)
        window = np.hanning(frame_length)
        
        # Overlap-add synthesis
        src_pos = 0.0
        dst_pos = 0
        
        while dst_pos < output_length - frame_length:
            src_idx = int(src_pos)
            if src_idx + frame_length > len(audio):
                break
            
            frame = audio[src_idx:src_idx + frame_length] * window
            
            if dst_pos + frame_length <= output_length:
                output[dst_pos:dst_pos + frame_length] += frame
            
            src_pos += hop_length * ratio
            dst_pos += hop_length
        
        return output
    
    def _spectral_convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        source: VoiceProfile,
        target: VoiceProfile
    ) -> np.ndarray:
        """Convert using spectral mapping."""
        # Compute STFT
        stft = self.processor.compute_stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Pitch shift in frequency domain
        if source.pitch_mean > 0 and target.pitch_mean > 0:
            pitch_ratio = target.pitch_mean / source.pitch_mean
            pitch_ratio = np.clip(pitch_ratio, 0.5, 2.0)
            
            # Shift spectrum
            new_magnitude = np.zeros_like(magnitude)
            for i in range(magnitude.shape[0]):
                new_idx = int(i / pitch_ratio)
                if 0 <= new_idx < magnitude.shape[0]:
                    new_magnitude[new_idx] = magnitude[i]
            
            magnitude = new_magnitude
        
        # Apply timbre transfer if available
        if target.timbre_features is not None and source.pitch_mean > 0:
            # Simple envelope matching
            envelope_shift = target.energy_mean / max(source.pitch_mean, 1e-8)
            magnitude *= min(envelope_shift, 2.0)
        
        # Reconstruct
        new_stft = magnitude * np.exp(1j * phase)
        converted = self.processor.compute_istft(new_stft)
        
        return converted
    
    def convert_realtime(
        self,
        audio_chunk: np.ndarray,
        target_voice: str,
        state: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Convert audio in real-time.
        
        Args:
            audio_chunk: Audio chunk to convert
            target_voice: Target voice ID
            state: Previous state for continuity
            
        Returns:
            Tuple of (converted_chunk, new_state)
        """
        if state is None:
            state = {"buffer": np.array([]), "overlap": 256}
        
        # Add to buffer
        buffer = np.concatenate([state["buffer"], audio_chunk])
        
        # Process if enough samples
        min_length = 2048
        if len(buffer) < min_length:
            return np.array([]), {"buffer": buffer, "overlap": state["overlap"]}
        
        # Convert
        result = self.convert(buffer, target_voice)
        
        # Keep overlap for next chunk
        new_buffer = buffer[-state["overlap"]:]
        
        # Remove overlap from output
        output = result.audio_data[:-state["overlap"]] if len(result.audio_data) > state["overlap"] else result.audio_data
        
        return output, {"buffer": new_buffer, "overlap": state["overlap"]}


class VoiceMixer:
    """Mix multiple voice characteristics."""
    
    def __init__(self):
        self.converter = VoiceConverter()
    
    def blend_voices(
        self,
        voice_ids: List[str],
        weights: List[float]
    ) -> VoiceProfile:
        """
        Blend multiple voices into one.
        
        Args:
            voice_ids: List of voice IDs
            weights: Blend weights (should sum to 1)
            
        Returns:
            Blended voice profile
        """
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
        # Collect profiles
        profiles = []
        for vid in voice_ids:
            profile = self.converter.get_voice(vid)
            if profile:
                profiles.append(profile)
            else:
                raise ValueError(f"Voice {vid} not found")
        
        # Blend characteristics
        pitch_mean = sum(p.pitch_mean * w for p, w in zip(profiles, weights))
        pitch_std = sum(p.pitch_std * w for p, w in zip(profiles, weights))
        energy_mean = sum(p.energy_mean * w for p, w in zip(profiles, weights))
        
        # Blend timbre features
        timbre_list = [p.timbre_features for p in profiles if p.timbre_features is not None]
        if timbre_list:
            timbre = sum(t * w for t, w in zip(timbre_list, weights[:len(timbre_list)]))
        else:
            timbre = None
        
        blended = VoiceProfile(
            id=f"blend_{'_'.join(voice_ids)}",
            name=f"Blend of {', '.join(voice_ids)}",
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            energy_mean=energy_mean,
            timbre_features=timbre
        )
        
        return blended


# Global instance
_voice_converter: Optional[VoiceConverter] = None


def get_voice_converter() -> VoiceConverter:
    """Get or create global VoiceConverter instance."""
    global _voice_converter
    if _voice_converter is None:
        _voice_converter = VoiceConverter()
    return _voice_converter
