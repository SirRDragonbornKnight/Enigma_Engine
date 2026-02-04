"""
Audio Analysis for Voice Cloning

Analyzes audio samples to extract voice characteristics.

Features:
- Audio feature extraction (pitch, speed, timbre)
- Voice similarity comparison
- Parameter estimation from audio
- Integration hooks for advanced TTS (Coqui XTTS, etc.)

Usage:
    from forge_ai.voice.audio_analyzer import AudioAnalyzer
    
    analyzer = AudioAnalyzer()
    
    # Analyze audio samples
    features = analyzer.analyze_audio("sample.wav")
    
    # Estimate voice parameters
    profile = analyzer.estimate_voice_profile(["sample1.wav", "sample2.wav"])
    
    # Compare voices
    similarity = analyzer.compare_voices("voice1.wav", "voice2.wav")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .voice_profile import VoiceProfile

# Audio constants for fallback analysis
DEFAULT_SAMPLE_RATE = 44100  # Hz
DEFAULT_CHANNELS = 2  # Stereo
DEFAULT_BYTES_PER_SAMPLE = 2  # 16-bit audio


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
    formants: Optional[List[float]] = None  # Formant frequencies
    spectral_centroid: Optional[float] = None
    zero_crossing_rate: Optional[float] = None


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
            import librosa
            self.have_librosa = True
        except ImportError:
            pass
        
        try:
            import parselmouth
            self.have_parselmouth = True
        except ImportError:
            pass
        
        try:
            import soundfile
            self.have_soundfile = True
        except ImportError:
            pass
    
    def _estimate_speaking_rate(self, audio: 'np.ndarray', sr: int, duration: float) -> float:
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
            
            return AudioFeatures(
                average_pitch=normalized_pitch,
                pitch_variance=pitch_variance / 100.0,  # Normalize
                speaking_rate=speaking_rate,
                energy=normalized_energy,
                duration=duration,
                sample_rate=sr,
                spectral_centroid=spectral_centroid,
                zero_crossing_rate=float(zcr)
            )
            
        except Exception as e:
            print(f"Warning: Advanced analysis failed: {e}")
            return self._analyze_basic(audio_path)
    
    def _analyze_basic(self, audio_path: Path) -> AudioFeatures:
        \"\"\"
        Basic analysis without advanced libraries (librosa).
        
        Uses Python standard library (wave, audioop) for WAV files,
        or falls back to file-based estimates for other formats.
        \"\"\"
        import struct
        
        try:
            # Try to analyze WAV files with standard library
            if str(audio_path).lower().endswith('.wav'):
                import wave
                import audioop
                
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
        audio_files: List[str],
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
                print(f"Warning: Could not analyze {audio_file}: {e}")
        
        if not all_features:
            print("Warning: No valid audio samples analyzed, using defaults")
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
            print(f"Error comparing voices: {e}")
            return 0.0
    
    def extract_coqui_features(
        self,
        audio_files: List[str]
    ) -> Dict[str, Any]:
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
                print(f"Warning: Could not analyze {audio_file}: {e}")
        
        return features
    
    def validate_audio_quality(
        self,
        audio_path: str
    ) -> Tuple[bool, List[str]]:
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
    audio_files: List[str],
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
