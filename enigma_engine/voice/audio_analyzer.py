"""
Audio Analysis for Voice Cloning

Analyzes audio samples to extract voice characteristics.

Features:
- Audio feature extraction (pitch, speed, timbre)
- Voice similarity comparison
- Parameter estimation from audio
- Integration hooks for advanced TTS (Coqui XTTS, etc.)

Usage:
    from enigma_engine.voice.audio_analyzer import AudioAnalyzer
    
    analyzer = AudioAnalyzer()
    
    # Analyze audio samples
    features = analyzer.analyze_audio("sample.wav")
    
    # Estimate voice parameters
    profile = analyzer.estimate_voice_profile(["sample1.wav", "sample2.wav"])
    
    # Compare voices
    similarity = analyzer.compare_voices("voice1.wav", "voice2.wav")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .voice_profile import VoiceProfile

logger = logging.getLogger(__name__)

# Audio constants for fallback analysis
DEFAULT_SAMPLE_RATE = 44100  # Hz
DEFAULT_CHANNELS = 2  # Stereo
DEFAULT_BYTES_PER_SAMPLE = 2  # 16-bit audio


@dataclass
class TimbreFeatures:
    """Voice timbre characteristics for cloning accuracy."""
    
    # Formants (resonant frequencies of vocal tract)
    formants: list[float] = field(default_factory=lambda: [500, 1500, 2500])
    formant_bandwidths: list[float] = field(default_factory=lambda: [90, 110, 170])
    
    # Spectral characteristics
    spectral_centroid: float = 1500.0  # "Center of mass" of spectrum (Hz)
    spectral_bandwidth: float = 2000.0  # Spread of spectrum (Hz)
    spectral_rolloff: float = 4000.0   # Frequency below which 85% of energy is contained
    spectral_flatness: float = 0.1     # How noise-like vs tonal (0=tonal, 1=noise)
    
    # Voice quality
    jitter: float = 0.01              # Pitch perturbation (irregularity)
    shimmer: float = 0.03             # Amplitude perturbation
    harmonics_to_noise: float = 15.0  # HNR in dB (voice clarity)
    
    # Vocal tract characteristics
    vocal_tract_length: float = 17.0  # Estimated in cm (affects formant spacing)
    breathiness: float = 0.3          # 0=clear, 1=breathy
    nasality: float = 0.2             # 0=oral, 1=nasal
    
    def to_dict(self) -> dict:
        return {
            'formants': self.formants,
            'formant_bandwidths': self.formant_bandwidths,
            'spectral_centroid': self.spectral_centroid,
            'spectral_bandwidth': self.spectral_bandwidth,
            'spectral_rolloff': self.spectral_rolloff,
            'spectral_flatness': self.spectral_flatness,
            'jitter': self.jitter,
            'shimmer': self.shimmer,
            'harmonics_to_noise': self.harmonics_to_noise,
            'vocal_tract_length': self.vocal_tract_length,
            'breathiness': self.breathiness,
            'nasality': self.nasality,
        }


@dataclass
class AudioFeatures:
    """Extracted audio features."""
    
    average_pitch: float = 1.0  # Normalized pitch (1.0 = reference)
    pitch_variance: float = 0.0  # Pitch variation
    speaking_rate: float = 1.0  # Words per second, normalized
    energy: float = 0.5  # Average energy/volume
    duration: float = 0.0  # Sample duration in seconds
    sample_rate: int = 0  # Audio sample rate
    
    # Advanced features (if available)
    formants: list[float] | None = None  # Formant frequencies
    spectral_centroid: float | None = None
    zero_crossing_rate: float | None = None
    
    # Timbre features for voice cloning
    timbre: TimbreFeatures | None = None


class AudioAnalyzer:
    """
    Analyzes audio to extract voice characteristics.
    
    Supports basic audio analysis with graceful degradation when
    advanced libraries are not available.
    """
    
    def __init__(self):
        """Initialize audio analyzer."""
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check for optional audio processing libraries."""
        self.have_librosa = False
        self.have_parselmouth = False
        self.have_soundfile = False
        
        try:
            self.have_librosa = True
        except ImportError:
            pass  # Intentionally silent
        
        try:
            self.have_parselmouth = True
        except ImportError:
            pass  # Intentionally silent
        
        try:
            self.have_soundfile = True
        except ImportError:
            pass  # Intentionally silent
    
    def _estimate_speaking_rate(self, audio: np.ndarray, sr: int, duration: float) -> float:
        """
        Estimate speaking rate from audio using energy envelope peaks.
        
        This method counts syllable-like peaks in the energy envelope,
        which correlates with speaking rate.
        
        Args:
            audio: Audio samples as numpy array
            sr: Sample rate
            duration: Audio duration in seconds
            
        Returns:
            Speaking rate normalized around 1.0 (1.0 = ~150 wpm average)
        """
        import numpy as np
        
        try:
            # Calculate short-time energy (using 20ms windows)
            frame_length = int(sr * 0.02)  # 20ms
            hop_length = int(sr * 0.01)    # 10ms hop
            
            # Compute energy for each frame
            num_frames = 1 + (len(audio) - frame_length) // hop_length
            energy = np.zeros(num_frames)
            
            for i in range(num_frames):
                start = i * hop_length
                end = start + frame_length
                if end <= len(audio):
                    energy[i] = np.sum(audio[start:end] ** 2)
            
            if len(energy) < 3:
                return 1.0
            
            # Smooth the energy envelope
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(energy, kernel, mode='same')
            
            # Normalize
            if np.max(smoothed) > 0:
                smoothed = smoothed / np.max(smoothed)
            
            # Find peaks (syllable candidates)
            # A peak is where energy is higher than neighbors and above threshold
            threshold = 0.2
            peaks = []
            for i in range(1, len(smoothed) - 1):
                if (smoothed[i] > smoothed[i-1] and 
                    smoothed[i] > smoothed[i+1] and 
                    smoothed[i] > threshold):
                    peaks.append(i)
            
            # Calculate syllables per second
            syllables = len(peaks)
            syllables_per_second = syllables / duration if duration > 0 else 0
            
            # Normalize to speaking rate (average English is ~5-6 syllables/sec)
            # 1.0 = 5.5 syllables/second
            normalized_rate = syllables_per_second / 5.5
            
            # Clamp to reasonable range
            return max(0.3, min(2.5, normalized_rate))
            
        except Exception as e:
            logger.debug(f"Speaking rate estimation failed: {e}")
            return 1.0
    
    def _estimate_basic_speaking_rate(self, samples: tuple, sample_rate: int, duration: float) -> float:
        """
        Estimate speaking rate using basic audio samples (no numpy/librosa).
        
        Uses a simplified syllable counting approach.
        
        Args:
            samples: Audio samples as tuple
            sample_rate: Sample rate
            duration: Audio duration in seconds
            
        Returns:
            Speaking rate normalized around 1.0
        """
        if not samples or duration <= 0:
            return 1.0
        
        try:
            # Calculate frame-based energy using 20ms windows
            frame_size = int(sample_rate * 0.02)
            hop_size = int(sample_rate * 0.01)
            
            energy = []
            for i in range(0, len(samples) - frame_size, hop_size):
                frame = samples[i:i + frame_size]
                frame_energy = sum(s ** 2 for s in frame) / len(frame)
                energy.append(frame_energy)
            
            if len(energy) < 3:
                return 1.0
            
            # Normalize energy
            max_energy = max(energy) if energy else 1
            if max_energy > 0:
                energy = [e / max_energy for e in energy]
            
            # Simple smoothing (3-point average)
            smoothed = []
            for i in range(len(energy)):
                start = max(0, i - 1)
                end = min(len(energy), i + 2)
                smoothed.append(sum(energy[start:end]) / (end - start))
            
            # Count peaks (syllable approximations)
            threshold = 0.2
            peaks = 0
            for i in range(1, len(smoothed) - 1):
                if (smoothed[i] > smoothed[i-1] and 
                    smoothed[i] > smoothed[i+1] and 
                    smoothed[i] > threshold):
                    peaks += 1
            
            # Calculate normalized rate
            syllables_per_second = peaks / duration if duration > 0 else 0
            normalized_rate = syllables_per_second / 5.5  # Average English rate
            
            return max(0.3, min(2.5, normalized_rate))
            
        except Exception:
            return 1.0
    
    def analyze_audio(self, audio_path: str) -> AudioFeatures:
        """
        Analyze audio file to extract voice features.
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, etc.)
            
        Returns:
            AudioFeatures with extracted characteristics
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Try advanced analysis first
        if self.have_librosa and self.have_parselmouth:
            return self._analyze_with_librosa(audio_path)
        
        # Fallback to basic analysis
        return self._analyze_basic(audio_path)
    
    def _analyze_with_librosa(self, audio_path: Path) -> AudioFeatures:
        """Analyze with librosa and parselmouth (advanced)."""
        import librosa
        import numpy as np
        
        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=None)
            duration = len(y) / sr
            
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get average pitch
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            avg_pitch = np.mean(pitch_values) if pitch_values else 0
            pitch_variance = np.std(pitch_values) if pitch_values else 0
            
            # Normalize pitch (assume 200 Hz as reference)
            normalized_pitch = avg_pitch / 200.0 if avg_pitch > 0 else 1.0
            
            # Energy/volume
            energy = np.mean(librosa.feature.rms(y=y))
            normalized_energy = min(1.0, energy * 10)  # Normalize to 0-1
            
            # Speaking rate (rough estimate from zero crossings)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Estimate speaking rate from energy envelope
            speaking_rate = self._estimate_speaking_rate(y, sr, duration)
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid = np.mean(spectral_centroids)
            
            # Extract timbre features for accurate voice cloning
            timbre = self._extract_timbre_features(y, sr, avg_pitch, pitch_values)
            
            return AudioFeatures(
                average_pitch=normalized_pitch,
                pitch_variance=pitch_variance / 100.0,  # Normalize
                speaking_rate=speaking_rate,
                energy=normalized_energy,
                duration=duration,
                sample_rate=sr,
                formants=timbre.formants if timbre else None,
                spectral_centroid=spectral_centroid,
                zero_crossing_rate=float(zcr),
                timbre=timbre
            )
            
        except Exception as e:
            logger.warning(f"Advanced analysis failed: {e}")
            return self._analyze_basic(audio_path)
    
    def _extract_timbre_features(
        self, 
        y: np.ndarray, 
        sr: int,
        f0: float,
        pitch_values: list[float]
    ) -> TimbreFeatures | None:
        """
        Extract comprehensive timbre features for voice cloning.
        
        Analyzes formants, spectral characteristics, and voice quality
        parameters needed for accurate voice reproduction.
        
        Args:
            y: Audio samples
            sr: Sample rate
            f0: Fundamental frequency
            pitch_values: List of pitch measurements
            
        Returns:
            TimbreFeatures or None if extraction fails
        """
        try:
            import librosa
            import numpy as np
            
            timbre = TimbreFeatures()
            
            # === FORMANT ANALYSIS ===
            # Use LPC (Linear Predictive Coding) to estimate formants
            formants, bandwidths = self._extract_formants_lpc(y, sr)
            if formants:
                timbre.formants = formants[:3] if len(formants) >= 3 else formants
                timbre.formant_bandwidths = bandwidths[:3] if len(bandwidths) >= 3 else bandwidths
            
            # === SPECTRAL FEATURES ===
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            timbre.spectral_centroid = float(np.mean(spectral_centroid))
            
            # Spectral bandwidth (spread)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            timbre.spectral_bandwidth = float(np.mean(spectral_bandwidth))
            
            # Spectral rolloff (frequency below which 85% of energy lies)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
            timbre.spectral_rolloff = float(np.mean(spectral_rolloff))
            
            # Spectral flatness (tonality vs noise)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)
            timbre.spectral_flatness = float(np.mean(spectral_flatness))
            
            # === VOICE QUALITY ===
            # Jitter (pitch perturbation)
            if len(pitch_values) > 1:
                pitch_arr = np.array(pitch_values)
                pitch_diffs = np.abs(np.diff(pitch_arr))
                timbre.jitter = float(np.mean(pitch_diffs) / np.mean(pitch_arr)) if np.mean(pitch_arr) > 0 else 0.01
            
            # Shimmer (amplitude perturbation)
            rms = librosa.feature.rms(y=y)[0]
            if len(rms) > 1:
                rms_diffs = np.abs(np.diff(rms))
                timbre.shimmer = float(np.mean(rms_diffs) / np.mean(rms)) if np.mean(rms) > 0 else 0.03
            
            # Harmonics-to-noise ratio (HNR)
            timbre.harmonics_to_noise = self._estimate_hnr(y, sr, f0)
            
            # === VOCAL TRACT CHARACTERISTICS ===
            # Estimate vocal tract length from formant spacing
            if len(timbre.formants) >= 2:
                # Formant spacing relates inversely to vocal tract length
                # Average adult male: 17cm, female: 14cm
                formant_spacing = timbre.formants[1] - timbre.formants[0]
                # Typical F1-F2 spacing is ~1000Hz for 17cm tract
                timbre.vocal_tract_length = 17.0 * (1000.0 / formant_spacing) if formant_spacing > 0 else 17.0
                timbre.vocal_tract_length = max(12.0, min(22.0, timbre.vocal_tract_length))
            
            # Breathiness (from spectral flatness and HNR)
            timbre.breathiness = min(1.0, max(0.0, timbre.spectral_flatness * 2 + (1 - timbre.harmonics_to_noise / 30)))
            
            # Nasality (estimated from formant patterns)
            # Nasal sounds show anti-resonances and altered F1
            timbre.nasality = self._estimate_nasality(timbre.formants, timbre.spectral_flatness)
            
            return timbre
            
        except Exception as e:
            logger.debug(f"Timbre extraction failed: {e}")
            return None
    
    def _extract_formants_lpc(
        self, 
        y: np.ndarray, 
        sr: int,
        order: int = 12
    ) -> tuple[list[float], list[float]]:
        """
        Extract formant frequencies using LPC analysis.
        
        Linear Predictive Coding models the vocal tract as a filter,
        and the poles of this filter correspond to formant frequencies.
        
        Args:
            y: Audio samples
            sr: Sample rate
            order: LPC order (higher = more formants but less stable)
            
        Returns:
            Tuple of (formant_frequencies, formant_bandwidths)
        """
        import numpy as np
        
        try:
            # Pre-emphasis filter (boost high frequencies)
            pre_emphasis = 0.97
            y_emph = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
            
            # Window the signal
            frame_length = min(len(y_emph), int(0.025 * sr))  # 25ms frame
            frame = y_emph[:frame_length] * np.hamming(frame_length)
            
            # Compute LPC coefficients using autocorrelation method
            # Autocorrelation
            r = np.correlate(frame, frame, mode='full')
            r = r[len(r)//2:len(r)//2 + order + 1]
            
            # Levinson-Durbin recursion
            a = self._levinson_durbin(r, order)
            
            if a is None:
                return [500, 1500, 2500], [90, 110, 170]  # Defaults
            
            # Find roots of the LPC polynomial
            roots = np.roots(a)
            
            # Keep only roots inside unit circle with positive imaginary part
            formants = []
            bandwidths = []
            
            for root in roots:
                if np.imag(root) >= 0:
                    # Convert to frequency
                    angle = np.angle(root)
                    freq = angle * sr / (2 * np.pi)
                    
                    # Bandwidth from distance to unit circle
                    bw = -np.log(np.abs(root)) * sr / np.pi
                    
                    # Filter to reasonable formant range
                    if 200 < freq < 5000 and bw < 500:
                        formants.append(freq)
                        bandwidths.append(bw)
            
            # Sort by frequency
            if formants:
                sorted_indices = np.argsort(formants)
                formants = [formants[i] for i in sorted_indices]
                bandwidths = [bandwidths[i] for i in sorted_indices]
                return formants, bandwidths
            
            return [500, 1500, 2500], [90, 110, 170]
            
        except Exception as e:
            logger.debug(f"LPC formant extraction failed: {e}")
            return [500, 1500, 2500], [90, 110, 170]
    
    def _levinson_durbin(self, r: np.ndarray, order: int) -> np.ndarray | None:
        """
        Levinson-Durbin algorithm for solving Toeplitz system.
        
        Used to compute LPC coefficients from autocorrelation.
        """
        import numpy as np
        
        try:
            a = np.zeros(order + 1)
            a[0] = 1.0
            
            e = r[0]
            
            for i in range(1, order + 1):
                # Compute reflection coefficient
                lambda_val = 0.0
                for j in range(1, i):
                    lambda_val += a[j] * r[i - j]
                lambda_val = (r[i] - lambda_val) / e if e != 0 else 0
                
                # Update coefficients
                a_new = a.copy()
                a_new[i] = lambda_val
                for j in range(1, i):
                    a_new[j] = a[j] - lambda_val * a[i - j]
                a = a_new
                
                # Update error
                e = e * (1 - lambda_val ** 2)
                
                if e <= 0:
                    break
            
            return a
            
        except Exception:
            return None
    
    def _estimate_hnr(self, y: np.ndarray, sr: int, f0: float) -> float:
        """
        Estimate Harmonics-to-Noise Ratio (HNR).
        
        HNR measures voice clarity - higher values indicate clearer voice,
        lower values indicate breathier/hoarser voice.
        
        Args:
            y: Audio samples
            sr: Sample rate
            f0: Fundamental frequency
            
        Returns:
            HNR in dB (typically 5-25 dB for normal speech)
        """
        import numpy as np
        
        try:
            if f0 <= 0:
                f0 = 150  # Default
            
            # Use autocorrelation-based HNR estimation
            period = int(sr / f0)
            
            if period < 2 or period > len(y) // 2:
                return 15.0  # Default
            
            # Compute autocorrelation
            n_periods = min(10, len(y) // period - 1)
            if n_periods < 1:
                return 15.0
            
            acf_values = []
            for lag in range(1, n_periods + 1):
                shift = lag * period
                if shift < len(y):
                    segment1 = y[:len(y) - shift]
                    segment2 = y[shift:]
                    min_len = min(len(segment1), len(segment2))
                    acf = np.sum(segment1[:min_len] * segment2[:min_len])
                    acf /= np.sqrt(np.sum(segment1[:min_len]**2) * np.sum(segment2[:min_len]**2) + 1e-10)
                    acf_values.append(acf)
            
            if not acf_values:
                return 15.0
            
            # HNR from peak autocorrelation
            max_acf = max(acf_values)
            if max_acf > 0 and max_acf < 1:
                hnr = 10 * np.log10(max_acf / (1 - max_acf + 1e-10))
                return float(max(0, min(30, hnr)))
            
            return 15.0
            
        except Exception:
            return 15.0
    
    def _estimate_nasality(self, formants: list[float], spectral_flatness: float) -> float:
        """
        Estimate nasality from formant patterns.
        
        Nasal sounds show characteristic formant patterns with
        anti-resonances (zeros) and altered F1.
        
        Args:
            formants: Formant frequencies
            spectral_flatness: Spectral flatness value
            
        Returns:
            Nasality estimate (0-1)
        """
        if len(formants) < 2:
            return 0.2
        
        try:
            # Nasal sounds typically have:
            # 1. Lower F1 (around 250-300 Hz vs 500+ for oral)
            # 2. Anti-resonance around 1000 Hz
            # 3. Higher spectral flatness
            
            nasality = 0.0
            
            # Low F1 suggests nasality
            if formants[0] < 400:
                nasality += 0.3
            
            # Check for characteristic nasal formant spacing
            if len(formants) >= 2:
                f1_f2_ratio = formants[1] / formants[0] if formants[0] > 0 else 3
                # Nasal sounds often have larger F1-F2 ratio
                if f1_f2_ratio > 4:
                    nasality += 0.2
            
            # High spectral flatness can indicate nasality
            nasality += spectral_flatness * 0.3
            
            return min(1.0, max(0.0, nasality))
            
        except Exception:
            return 0.2

    def _analyze_basic(self, audio_path: Path) -> AudioFeatures:
        """
        Basic analysis without advanced libraries (librosa).
        
        Uses Python standard library (wave, audioop) for WAV files,
        or falls back to file-based estimates for other formats.
        """
        import struct
        
        try:
            # Try to analyze WAV files with standard library
            if str(audio_path).lower().endswith('.wav'):
                import audioop
                import wave
                
                with wave.open(str(audio_path), 'rb') as wf:
                    n_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    sample_rate = wf.getframerate()
                    n_frames = wf.getnframes()
                    
                    # Read all frames
                    frames = wf.readframes(n_frames)
                    
                    # Calculate duration
                    duration = n_frames / sample_rate
                    
                    # Calculate RMS energy (volume level)
                    try:
                        rms = audioop.rms(frames, sample_width)
                        # Normalize to 0-1 range (assume max RMS around 32767 for 16-bit)
                        max_rms = (2 ** (sample_width * 8 - 1)) - 1
                        energy = min(1.0, rms / max_rms)
                    except Exception:
                        energy = 0.5
                    
                    # Calculate zero crossing rate (indicates frequency content)
                    try:
                        # Convert to mono if stereo
                        if n_channels == 2:
                            mono_frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
                        else:
                            mono_frames = frames
                        
                        # Count zero crossings
                        if sample_width == 2:  # 16-bit
                            samples = struct.unpack(f'<{len(mono_frames)//2}h', mono_frames)
                        elif sample_width == 1:  # 8-bit
                            samples = struct.unpack(f'{len(mono_frames)}b', mono_frames)
                        else:
                            samples = []
                        
                        if samples:
                            zero_crossings = sum(1 for i in range(1, len(samples)) 
                                               if (samples[i] >= 0) != (samples[i-1] >= 0))
                            zcr = zero_crossings / len(samples) if samples else 0
                        else:
                            zcr = 0.1
                    except Exception:
                        zcr = 0.1
                    
                    # Estimate pitch from zero crossing rate
                    # Higher ZCR generally means higher pitch
                    estimated_pitch = 0.5 + (zcr * 5)  # Scale to reasonable range
                    estimated_pitch = max(0.5, min(2.0, estimated_pitch))
                    
                    # Estimate speaking rate from energy variations in basic mode
                    # Count peaks in the energy as syllable approximations
                    speaking_rate = self._estimate_basic_speaking_rate(samples, sample_rate, duration)
                    
                    return AudioFeatures(
                        average_pitch=estimated_pitch,
                        pitch_variance=0.15,
                        speaking_rate=speaking_rate,
                        energy=energy,
                        duration=duration,
                        sample_rate=sample_rate,
                        zero_crossing_rate=zcr
                    )
        except Exception as e:
            logger.debug(f"WAV analysis failed, using file estimates: {e}")
        
        # Fallback: file-based estimates for non-WAV or failed analysis
        file_size = audio_path.stat().st_size
        
        # Rough estimates using standard audio format constants
        estimated_duration = file_size / (
            DEFAULT_SAMPLE_RATE * DEFAULT_CHANNELS * DEFAULT_BYTES_PER_SAMPLE
        )
        
        # Return default features with estimates
        return AudioFeatures(
            average_pitch=1.0,
            pitch_variance=0.1,
            speaking_rate=1.0,
            energy=0.7,
            duration=estimated_duration,
            sample_rate=DEFAULT_SAMPLE_RATE
        )
    
    def estimate_voice_profile(
        self,
        audio_files: list[str],
        name: str = "analyzed_voice"
    ) -> VoiceProfile:
        """
        Estimate voice profile parameters from audio samples.
        
        Args:
            audio_files: List of audio file paths
            name: Name for the voice profile
            
        Returns:
            VoiceProfile with estimated parameters
        """
        if not audio_files:
            return VoiceProfile(name=name)
        
        # Analyze all samples
        all_features = []
        for audio_file in audio_files:
            try:
                features = self.analyze_audio(audio_file)
                all_features.append(features)
            except Exception as e:
                logger.warning(f"Could not analyze {audio_file}: {e}")
        
        if not all_features:
            logger.warning("No valid audio samples analyzed, using defaults")
            return VoiceProfile(name=name)
        
        # Average features
        avg_pitch = sum(f.average_pitch for f in all_features) / len(all_features)
        avg_energy = sum(f.energy for f in all_features) / len(all_features)
        
        # Map to voice profile parameters
        # Pitch: normalize around 1.0
        pitch = max(0.5, min(1.5, avg_pitch))
        
        # Volume: use energy
        volume = max(0.3, min(1.0, avg_energy))
        
        # Speed: default to 1.0 (would need speech recognition for accurate estimate)
        speed = 1.0
        
        # Determine voice type from pitch
        voice = "default"
        if avg_pitch < 0.85:
            voice = "male"
        elif avg_pitch > 1.15:
            voice = "female"
        
        # Create profile
        profile = VoiceProfile(
            name=name,
            pitch=pitch,
            speed=speed,
            volume=volume,
            voice=voice,
            description=f"Analyzed from {len(audio_files)} audio samples"
        )
        
        return profile
    
    def compare_voices(
        self,
        audio_file1: str,
        audio_file2: str
    ) -> float:
        """
        Compare similarity between two voice samples.
        
        Args:
            audio_file1: First audio file
            audio_file2: Second audio file
            
        Returns:
            Similarity score (0.0 to 1.0, higher = more similar)
        """
        try:
            features1 = self.analyze_audio(audio_file1)
            features2 = self.analyze_audio(audio_file2)
            
            # Compare pitch
            pitch_diff = abs(features1.average_pitch - features2.average_pitch)
            pitch_similarity = max(0, 1.0 - pitch_diff)
            
            # Compare energy
            energy_diff = abs(features1.energy - features2.energy)
            energy_similarity = max(0, 1.0 - energy_diff)
            
            # Compare variance
            variance_diff = abs(features1.pitch_variance - features2.pitch_variance)
            variance_similarity = max(0, 1.0 - variance_diff)
            
            # Weighted average
            similarity = (
                pitch_similarity * 0.5 +
                energy_similarity * 0.3 +
                variance_similarity * 0.2
            )
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error comparing voices: {e}")
            return 0.0
    
    def extract_coqui_features(
        self,
        audio_files: list[str]
    ) -> dict[str, Any]:
        """
        Extract features suitable for Coqui XTTS voice cloning.
        
        This prepares audio samples for use with Coqui TTS when available.
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            Dict with features for Coqui TTS
        """
        features = {
            "audio_files": audio_files,
            "num_samples": len(audio_files),
            "analysis": []
        }
        
        for audio_file in audio_files:
            try:
                audio_features = self.analyze_audio(audio_file)
                features["analysis"].append({
                    "file": audio_file,
                    "duration": audio_features.duration,
                    "sample_rate": audio_features.sample_rate,
                    "pitch": audio_features.average_pitch,
                    "energy": audio_features.energy
                })
            except Exception as e:
                logger.warning(f"Could not analyze {audio_file}: {e}")
        
        return features
    
    def validate_audio_quality(
        self,
        audio_path: str
    ) -> tuple[bool, list[str]]:
        """
        Validate audio quality for voice cloning.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        try:
            features = self.analyze_audio(audio_path)
            
            # Check duration (should be at least 3 seconds)
            if features.duration < 3.0:
                issues.append(f"Audio too short: {features.duration:.1f}s (need at least 3s)")
            
            # Check if too long (over 30 seconds might have too much silence)
            if features.duration > 30.0:
                issues.append(f"Audio very long: {features.duration:.1f}s (consider trimming)")
            
            # Check sample rate
            if features.sample_rate < 16000:
                issues.append(f"Low sample rate: {features.sample_rate} Hz (prefer 22050+ Hz)")
            
            # Check energy
            if features.energy < 0.1:
                issues.append("Audio too quiet")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Could not analyze audio: {e}")
            return False, issues


# Convenience functions
def analyze_audio(audio_path: str) -> AudioFeatures:
    """
    Analyze audio file to extract features.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        AudioFeatures
    """
    analyzer = AudioAnalyzer()
    return analyzer.analyze_audio(audio_path)


def estimate_voice_profile(
    audio_files: list[str],
    name: str = "analyzed_voice"
) -> VoiceProfile:
    """
    Estimate voice profile from audio samples.
    
    Args:
        audio_files: List of audio file paths
        name: Name for profile
        
    Returns:
        VoiceProfile
    """
    analyzer = AudioAnalyzer()
    return analyzer.estimate_voice_profile(audio_files, name)


def compare_voices(audio_file1: str, audio_file2: str) -> float:
    """
    Compare similarity between two voices.
    
    Args:
        audio_file1: First audio file
        audio_file2: Second audio file
        
    Returns:
        Similarity score (0-1)
    """
    analyzer = AudioAnalyzer()
    return analyzer.compare_voices(audio_file1, audio_file2)
