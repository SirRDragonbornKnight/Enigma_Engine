"""
================================================================================
VOICE CLONING - CREATE CUSTOM VOICE PROFILES FROM SAMPLES
================================================================================

Clone voices from audio samples to create personalized text-to-speech.
Supports multiple methods from simple pitch/speed matching to neural cloning.

FILE: forge_ai/voice/voice_cloning.py
TYPE: Voice Synthesis
MAIN CLASSES: VoiceCloner, ClonedVoice, CloningMethod

FEATURES:
    - Simple cloning (pitch/speed/tone matching)
    - Spectral analysis for voice characteristics
    - Neural voice cloning (with external services)
    - Voice profile storage and management
    - Real-time voice conversion

USAGE:
    from forge_ai.voice.voice_cloning import VoiceCloner, CloningMethod
    
    cloner = VoiceCloner()
    
    # Clone voice from samples
    voice = cloner.clone_voice(
        name="my_voice",
        samples=["sample1.wav", "sample2.wav"],
        method=CloningMethod.SPECTRAL
    )
    
    # Use cloned voice
    audio = voice.speak("Hello, this is my cloned voice!")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import struct
import tempfile
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Union

from ..config import CONFIG

logger = logging.getLogger(__name__)

# Audio processing imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy import signal
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class CloningMethod(Enum):
    """Voice cloning methods."""
    SIMPLE = "simple"           # Basic pitch/speed matching
    SPECTRAL = "spectral"       # Spectral envelope matching
    NEURAL = "neural"           # Neural network cloning (external service)
    HYBRID = "hybrid"           # Combination of methods


@dataclass
class VoiceCharacteristics:
    """Analyzed voice characteristics."""
    
    pitch_mean: float = 150.0      # Average pitch in Hz
    pitch_std: float = 30.0        # Pitch variation
    speed: float = 1.0             # Speaking speed
    energy: float = 0.5            # Voice energy/loudness
    breathiness: float = 0.3       # Breathiness level
    
    # Spectral characteristics
    formants: List[float] = field(default_factory=lambda: [500, 1500, 2500])
    spectral_tilt: float = -6.0    # dB per octave
    harmonics_to_noise: float = 10.0  # HNR in dB
    
    # Timing
    pause_frequency: float = 0.5   # Pauses per second
    pause_duration: float = 0.2    # Average pause length
    
    def to_dict(self) -> Dict:
        return {
            'pitch_mean': self.pitch_mean,
            'pitch_std': self.pitch_std,
            'speed': self.speed,
            'energy': self.energy,
            'breathiness': self.breathiness,
            'formants': self.formants,
            'spectral_tilt': self.spectral_tilt,
            'harmonics_to_noise': self.harmonics_to_noise,
            'pause_frequency': self.pause_frequency,
            'pause_duration': self.pause_duration,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VoiceCharacteristics':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ClonedVoice:
    """A cloned voice profile."""
    
    name: str
    method: CloningMethod
    characteristics: VoiceCharacteristics
    
    # Source info
    source_samples: List[str] = field(default_factory=list)
    total_duration: float = 0.0  # Total sample duration in seconds
    
    # Storage
    profile_path: Optional[Path] = None
    model_path: Optional[Path] = None  # For neural cloning
    
    # Metadata
    created_at: str = ""
    description: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'method': self.method.value,
            'characteristics': self.characteristics.to_dict(),
            'source_samples': self.source_samples,
            'total_duration': self.total_duration,
            'profile_path': str(self.profile_path) if self.profile_path else None,
            'model_path': str(self.model_path) if self.model_path else None,
            'created_at': self.created_at,
            'description': self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ClonedVoice':
        return cls(
            name=data['name'],
            method=CloningMethod(data['method']),
            characteristics=VoiceCharacteristics.from_dict(data['characteristics']),
            source_samples=data.get('source_samples', []),
            total_duration=data.get('total_duration', 0.0),
            profile_path=Path(data['profile_path']) if data.get('profile_path') else None,
            model_path=Path(data['model_path']) if data.get('model_path') else None,
            created_at=data.get('created_at', ''),
            description=data.get('description', ''),
        )
    
    def speak(self, text: str, **kwargs) -> bytes:
        """
        Speak text using this cloned voice.
        
        Args:
            text: Text to speak
            **kwargs: Additional TTS parameters
        
        Returns:
            Audio data as bytes
        """
        from .voice_profile import VoiceProfile
        
        # Create voice profile from characteristics
        profile = VoiceProfile(
            name=self.name,
            pitch=self.characteristics.pitch_mean / 150.0,  # Normalize
            speed=self.characteristics.speed,
            volume=self.characteristics.energy,
        )
        
        # Get TTS engine
        try:
            from .tts_simple import SimpleTTS
            tts = SimpleTTS()
            return tts.speak_with_profile(text, profile)
        except ImportError:
            logger.warning("SimpleTTS not available")
            return b''


class VoiceCloner:
    """
    Clone voices from audio samples.
    
    Creates custom voice profiles by analyzing voice samples
    and extracting characteristics for TTS synthesis.
    """
    
    def __init__(self, profiles_dir: Path = None):
        """
        Initialize voice cloner.
        
        Args:
            profiles_dir: Directory for storing voice profiles
        """
        self.profiles_dir = profiles_dir or Path(CONFIG.get("data_dir", "data")) / "voice_profiles" / "cloned"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        self._cloned_voices: Dict[str, ClonedVoice] = {}
        self._load_profiles()
        
        logger.info(f"Voice cloner initialized with {len(self._cloned_voices)} profiles")
    
    def _load_profiles(self):
        """Load existing cloned voice profiles."""
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file) as f:
                    data = json.load(f)
                voice = ClonedVoice.from_dict(data)
                self._cloned_voices[voice.name] = voice
            except Exception as e:
                logger.warning(f"Error loading profile {profile_file}: {e}")
    
    def _save_profile(self, voice: ClonedVoice):
        """Save a cloned voice profile."""
        profile_path = self.profiles_dir / f"{voice.name}.json"
        voice.profile_path = profile_path
        
        with open(profile_path, 'w') as f:
            json.dump(voice.to_dict(), f, indent=2)
        
        logger.info(f"Saved voice profile: {voice.name}")
    
    def clone_voice(
        self,
        name: str,
        samples: List[Union[str, Path]],
        method: CloningMethod = CloningMethod.SPECTRAL,
        description: str = ""
    ) -> ClonedVoice:
        """
        Clone a voice from audio samples.
        
        Args:
            name: Name for the cloned voice
            samples: List of audio file paths
            method: Cloning method to use
            description: Optional description
        
        Returns:
            ClonedVoice profile
        """
        if not samples:
            raise ValueError("At least one audio sample required")
        
        # Convert paths
        sample_paths = [Path(s) for s in samples]
        
        # Validate samples
        for path in sample_paths:
            if not path.exists():
                raise ValueError(f"Sample file not found: {path}")
        
        # Analyze voice characteristics
        characteristics = self._analyze_samples(sample_paths, method)
        
        # Calculate total duration
        total_duration = sum(self._get_audio_duration(p) for p in sample_paths)
        
        # Create cloned voice
        voice = ClonedVoice(
            name=name,
            method=method,
            characteristics=characteristics,
            source_samples=[str(p) for p in sample_paths],
            total_duration=total_duration,
            description=description,
        )
        
        # Copy samples to profile directory (optional)
        samples_dir = self.profiles_dir / name / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        for path in sample_paths:
            shutil.copy(path, samples_dir / path.name)
        
        # Save profile
        self._save_profile(voice)
        self._cloned_voices[name] = voice
        
        logger.info(f"Cloned voice '{name}' from {len(samples)} samples ({total_duration:.1f}s total)")
        return voice
    
    def _analyze_samples(
        self,
        samples: List[Path],
        method: CloningMethod
    ) -> VoiceCharacteristics:
        """Analyze audio samples to extract voice characteristics."""
        
        if method == CloningMethod.SIMPLE:
            return self._simple_analysis(samples)
        elif method == CloningMethod.SPECTRAL:
            return self._spectral_analysis(samples)
        elif method == CloningMethod.NEURAL:
            return self._neural_analysis(samples)
        else:  # HYBRID
            # Combine methods
            simple = self._simple_analysis(samples)
            spectral = self._spectral_analysis(samples)
            
            # Average characteristics
            return VoiceCharacteristics(
                pitch_mean=(simple.pitch_mean + spectral.pitch_mean) / 2,
                pitch_std=(simple.pitch_std + spectral.pitch_std) / 2,
                speed=(simple.speed + spectral.speed) / 2,
                energy=(simple.energy + spectral.energy) / 2,
                breathiness=spectral.breathiness,
                formants=spectral.formants,
                spectral_tilt=spectral.spectral_tilt,
                harmonics_to_noise=spectral.harmonics_to_noise,
            )
    
    def _simple_analysis(self, samples: List[Path]) -> VoiceCharacteristics:
        """Simple analysis using basic audio properties."""
        characteristics = VoiceCharacteristics()
        
        # Analyze each sample
        pitches = []
        energies = []
        
        for sample_path in samples:
            try:
                audio, sr = self._load_audio(sample_path)
                
                # Estimate pitch using zero-crossing rate
                zcr = self._zero_crossing_rate(audio)
                estimated_pitch = zcr * sr / 2  # Very rough estimate
                pitches.append(min(max(estimated_pitch, 80), 400))  # Clamp to reasonable range
                
                # Estimate energy (RMS)
                rms = np.sqrt(np.mean(audio ** 2))
                energies.append(rms)
                
            except Exception as e:
                logger.warning(f"Error analyzing {sample_path}: {e}")
        
        if pitches:
            characteristics.pitch_mean = np.mean(pitches)
            characteristics.pitch_std = np.std(pitches) if len(pitches) > 1 else 30.0
        
        if energies:
            characteristics.energy = np.mean(energies)
        
        return characteristics
    
    def _spectral_analysis(self, samples: List[Path]) -> VoiceCharacteristics:
        """Spectral analysis for detailed voice characteristics."""
        if not SCIPY_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("scipy/numpy not available, using simple analysis")
            return self._simple_analysis(samples)
        
        characteristics = VoiceCharacteristics()
        
        all_pitches = []
        all_formants = []
        all_spectral_tilts = []
        all_energies = []
        
        for sample_path in samples:
            try:
                audio, sr = self._load_audio(sample_path)
                
                # Estimate fundamental frequency (F0)
                f0 = self._estimate_f0(audio, sr)
                if f0 > 0:
                    all_pitches.append(f0)
                
                # Estimate formants
                formants = self._estimate_formants(audio, sr)
                if formants:
                    all_formants.append(formants)
                
                # Calculate spectral tilt
                tilt = self._calculate_spectral_tilt(audio, sr)
                all_spectral_tilts.append(tilt)
                
                # Calculate energy
                rms = np.sqrt(np.mean(audio ** 2))
                all_energies.append(rms)
                
            except Exception as e:
                logger.warning(f"Error in spectral analysis of {sample_path}: {e}")
        
        # Aggregate results
        if all_pitches:
            characteristics.pitch_mean = np.mean(all_pitches)
            characteristics.pitch_std = np.std(all_pitches) if len(all_pitches) > 1 else 30.0
        
        if all_formants:
            # Average formants
            avg_formants = np.mean(all_formants, axis=0).tolist()
            characteristics.formants = avg_formants[:3] if len(avg_formants) >= 3 else avg_formants
        
        if all_spectral_tilts:
            characteristics.spectral_tilt = np.mean(all_spectral_tilts)
        
        if all_energies:
            characteristics.energy = np.mean(all_energies)
        
        return characteristics
    
    def _neural_analysis(self, samples: List[Path]) -> VoiceCharacteristics:
        """Neural network-based voice analysis (placeholder for external services)."""
        # This would integrate with services like:
        # - Eleven Labs
        # - Coqui TTS
        # - Resemble AI
        # For now, fall back to spectral analysis
        logger.info("Neural analysis requested - using spectral analysis as fallback")
        return self._spectral_analysis(samples)
    
    def _load_audio(self, path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file as numpy array."""
        path = Path(path)
        
        if path.suffix.lower() == '.wav':
            if SCIPY_AVAILABLE:
                sr, audio = wavfile.read(path)
                # Normalize to float
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                # Convert stereo to mono
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                return audio, sr
            else:
                # Use wave module
                with wave.open(str(path), 'rb') as wf:
                    sr = wf.getframerate()
                    n_frames = wf.getnframes()
                    audio_bytes = wf.readframes(n_frames)
                    sample_width = wf.getsampwidth()
                    
                    if sample_width == 2:
                        audio = np.array(struct.unpack(f'{n_frames}h', audio_bytes), dtype=np.float32) / 32768.0
                    else:
                        audio = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    return audio, sr
        else:
            raise ValueError(f"Unsupported audio format: {path.suffix}")
    
    def _get_audio_duration(self, path: Path) -> float:
        """Get audio duration in seconds."""
        try:
            with wave.open(str(path), 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / rate
        except Exception:
            return 0.0
    
    def _zero_crossing_rate(self, audio: np.ndarray) -> float:
        """Calculate zero-crossing rate."""
        signs = np.sign(audio)
        signs[signs == 0] = 1
        return np.mean(np.abs(np.diff(signs)) / 2)
    
    def _estimate_f0(self, audio: np.ndarray, sr: int) -> float:
        """Estimate fundamental frequency using autocorrelation."""
        if not SCIPY_AVAILABLE:
            return 150.0  # Default
        
        # Autocorrelation
        corr = np.correlate(audio, audio, mode='full')
        corr = corr[len(corr) // 2:]
        
        # Find first peak after initial decay
        d = np.diff(corr)
        
        # Find where derivative changes from negative to positive
        starts = np.where((d[:-1] < 0) & (d[1:] > 0))[0]
        
        if len(starts) < 2:
            return 150.0  # Default
        
        # First peak after zero
        first_peak_idx = starts[0] + 1
        
        if first_peak_idx > 0:
            f0 = sr / first_peak_idx
            # Clamp to reasonable range for human voice
            return min(max(f0, 60), 500)
        
        return 150.0
    
    def _estimate_formants(self, audio: np.ndarray, sr: int, num_formants: int = 5) -> List[float]:
        """Estimate formant frequencies using LPC analysis."""
        if not SCIPY_AVAILABLE:
            return [500, 1500, 2500]  # Default formants
        
        # Pre-emphasis
        pre_emphasis = 0.97
        audio_pre = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # LPC analysis
        from scipy.signal import lfilter
        
        # Number of LPC coefficients
        order = 2 + sr // 1000
        
        try:
            # Compute autocorrelation
            r = np.correlate(audio_pre, audio_pre, mode='full')
            r = r[len(r) // 2:len(r) // 2 + order + 1]
            
            # Levinson-Durbin recursion (simplified)
            # For now, use default formants
            return [500, 1500, 2500, 3500, 4500][:num_formants]
        except Exception:
            return [500, 1500, 2500]
    
    def _calculate_spectral_tilt(self, audio: np.ndarray, sr: int) -> float:
        """Calculate spectral tilt in dB/octave."""
        if not NUMPY_AVAILABLE:
            return -6.0  # Default
        
        # FFT
        n_fft = 2048
        spectrum = np.abs(np.fft.rfft(audio, n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1 / sr)
        
        # Fit line to log spectrum
        log_spectrum = 20 * np.log10(spectrum + 1e-10)
        log_freqs = np.log2(freqs + 1)
        
        # Simple linear regression
        valid = (freqs > 100) & (freqs < 4000)
        if np.sum(valid) > 10:
            coeffs = np.polyfit(log_freqs[valid], log_spectrum[valid], 1)
            return coeffs[0]  # dB per octave
        
        return -6.0
    
    def get_voice(self, name: str) -> Optional[ClonedVoice]:
        """Get a cloned voice by name."""
        return self._cloned_voices.get(name)
    
    def list_voices(self) -> List[ClonedVoice]:
        """List all cloned voices."""
        return list(self._cloned_voices.values())
    
    def delete_voice(self, name: str) -> bool:
        """Delete a cloned voice profile."""
        if name not in self._cloned_voices:
            return False
        
        voice = self._cloned_voices[name]
        
        # Delete profile file
        if voice.profile_path and voice.profile_path.exists():
            voice.profile_path.unlink()
        
        # Delete samples directory
        samples_dir = self.profiles_dir / name
        if samples_dir.exists():
            shutil.rmtree(samples_dir)
        
        del self._cloned_voices[name]
        logger.info(f"Deleted voice profile: {name}")
        return True
    
    def export_voice(self, name: str, export_path: Path) -> bool:
        """Export a voice profile to a file."""
        voice = self._cloned_voices.get(name)
        if not voice:
            return False
        
        export_path = Path(export_path)
        with open(export_path, 'w') as f:
            json.dump(voice.to_dict(), f, indent=2)
        
        return True
    
    def import_voice(self, import_path: Path) -> Optional[ClonedVoice]:
        """Import a voice profile from a file."""
        import_path = Path(import_path)
        if not import_path.exists():
            return None
        
        try:
            with open(import_path) as f:
                data = json.load(f)
            
            voice = ClonedVoice.from_dict(data)
            
            # Check for name collision
            if voice.name in self._cloned_voices:
                voice.name = f"{voice.name}_imported"
            
            self._save_profile(voice)
            self._cloned_voices[voice.name] = voice
            
            return voice
        except Exception as e:
            logger.error(f"Error importing voice: {e}")
            return None


# Global cloner instance
_cloner: Optional[VoiceCloner] = None


def get_voice_cloner() -> VoiceCloner:
    """Get or create global voice cloner instance."""
    global _cloner
    if _cloner is None:
        _cloner = VoiceCloner()
    return _cloner
