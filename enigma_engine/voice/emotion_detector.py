"""
Emotion Detection from Voice for Enigma AI Engine

Analyze audio to detect emotional states from speech.

Provides:
- Real-time emotion detection from audio stream
- Emotion classification (happy, sad, angry, etc.)
- Intensity estimation
- Arousal/valence mapping
- Integration with TTS/avatar systems

Usage:
    from enigma_engine.voice.emotion_detector import VoiceEmotionDetector
    
    detector = VoiceEmotionDetector()
    
    # Analyze audio file
    result = detector.analyze_file("speech.wav")
    print(result.emotion)      # "happy"
    print(result.confidence)   # 0.85
    print(result.intensity)    # 0.7
    
    # Real-time streaming
    detector.start_stream()
    detector.add_callback(on_emotion_change)
    
    # Integrate with avatar
    detector.connect_to_avatar(ai_avatar_controller)

Requirements:
    pip install librosa sounddevice scipy  # Core audio
    pip install speechbrain                # Optional: neural emotion model
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)

# Try imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sd = None


class EmotionCategory(Enum):
    """Detectable emotion categories."""
    NEUTRAL = auto()
    HAPPY = auto()
    SAD = auto()
    ANGRY = auto()
    FEARFUL = auto()
    DISGUSTED = auto()
    SURPRISED = auto()
    CALM = auto()
    EXCITED = auto()


@dataclass
class EmotionResult:
    """Result of emotion analysis."""
    # Primary emotion detected
    emotion: EmotionCategory = EmotionCategory.NEUTRAL
    confidence: float = 0.0
    
    # All emotion probabilities
    probabilities: Dict[EmotionCategory, float] = field(default_factory=dict)
    
    # Dimensional model
    valence: float = 0.0       # -1 (negative) to 1 (positive)
    arousal: float = 0.0       # -1 (calm) to 1 (excited)
    dominance: float = 0.0     # -1 (submissive) to 1 (dominant)
    
    # Intensity
    intensity: float = 0.5     # 0 (subtle) to 1 (intense)
    
    # Audio features
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    energy: float = 0.0
    speech_rate: float = 0.0
    
    # Timestamp
    timestamp: float = 0.0
    duration: float = 0.0


@dataclass
class VoiceEmotionConfig:
    """Configuration for voice emotion detection."""
    # Audio settings
    sample_rate: int = 16000
    frame_length: int = 2048
    hop_length: int = 512
    
    # Analysis
    min_segment_length: float = 0.5   # Minimum audio to analyze (seconds)
    update_interval: float = 0.2       # How often to update (seconds)
    
    # Smoothing
    smooth_factor: float = 0.3         # Lower = smoother emotion transitions
    
    # Thresholds
    silence_threshold: float = 0.01    # Below this = silence
    min_confidence: float = 0.3        # Below this = uncertain


class AudioFeatureExtractor:
    """
    Extract acoustic features relevant to emotion.
    
    Features:
    - Pitch (F0) statistics
    - Energy/intensity
    - Spectral features (MFCCs, spectral centroid)
    - Temporal features (speech rate, pauses)
    """
    
    def __init__(self, config: VoiceEmotionConfig):
        self.config = config
    
    def extract(self, audio: 'np.ndarray', sr: int) -> Dict[str, float]:
        """
        Extract features from audio.
        
        Args:
            audio: Audio samples (mono, float32)
            sr: Sample rate
            
        Returns:
            Dict of feature name -> value
        """
        if not LIBROSA_AVAILABLE or not NUMPY_AVAILABLE:
            return {}
        
        features = {}
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=audio, frame_length=self.config.frame_length)[0]
        features['energy_mean'] = float(np.mean(rms))
        features['energy_std'] = float(np.std(rms))
        features['energy_max'] = float(np.max(rms))
        
        # Pitch (F0) using pyin for robustness
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) > 0:
                features['pitch_mean'] = float(np.mean(f0_valid))
                features['pitch_std'] = float(np.std(f0_valid))
                features['pitch_min'] = float(np.min(f0_valid))
                features['pitch_max'] = float(np.max(f0_valid))
                features['pitch_range'] = features['pitch_max'] - features['pitch_min']
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
        except Exception:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
        
        # MFCCs (standard for speech emotion)
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, 
            n_mfcc=13,
            n_fft=self.config.frame_length,
            hop_length=self.config.hop_length
        )
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        # Zero crossing rate (correlates with speech/noise distinction)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        
        # Speech rate estimation (from energy peaks)
        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            features['tempo'] = float(tempo)
        except Exception:
            features['tempo'] = 120.0
        
        return features


class RuleBasedEmotionClassifier:
    """
    Rule-based emotion classification from acoustic features.
    
    Uses heuristics based on prosody research:
    - Happy: Higher pitch, higher energy, faster tempo
    - Sad: Lower pitch, lower energy, slower tempo
    - Angry: Higher pitch variation, high energy, fast tempo
    - Calm: Low pitch variation, moderate energy, slow tempo
    """
    
    def __init__(self):
        # Feature baselines (can be calibrated per speaker)
        self.baseline_pitch = 150.0
        self.baseline_energy = 0.1
        self.baseline_tempo = 120.0
    
    def classify(self, features: Dict[str, float]) -> EmotionResult:
        """
        Classify emotion from features.
        
        Args:
            features: Extracted audio features
            
        Returns:
            EmotionResult with classification
        """
        pitch = features.get('pitch_mean', self.baseline_pitch)
        pitch_std = features.get('pitch_std', 0)
        energy = features.get('energy_mean', self.baseline_energy)
        tempo = features.get('tempo', self.baseline_tempo)
        
        # Normalize relative to baseline
        pitch_ratio = pitch / self.baseline_pitch if self.baseline_pitch > 0 else 1.0
        energy_ratio = energy / self.baseline_energy if self.baseline_energy > 0 else 1.0
        tempo_ratio = tempo / self.baseline_tempo if self.baseline_tempo > 0 else 1.0
        
        # Calculate dimensional values
        # Valence: pitch height + energy correlate with positive
        valence = (pitch_ratio - 1.0) * 0.3 + (energy_ratio - 1.0) * 0.2
        valence = max(-1, min(1, valence))
        
        # Arousal: energy + tempo + pitch variation
        arousal = (energy_ratio - 1.0) * 0.4 + (tempo_ratio - 1.0) * 0.3
        if pitch_std > 50:
            arousal += 0.2
        arousal = max(-1, min(1, arousal))
        
        # Dominance: energy + low pitch
        dominance = (energy_ratio - 1.0) * 0.3 - (pitch_ratio - 1.0) * 0.2
        dominance = max(-1, min(1, dominance))
        
        # Map to categorical emotions
        probs = {e: 0.0 for e in EmotionCategory}
        
        if arousal < -0.3:
            # Low arousal
            if valence > 0.2:
                probs[EmotionCategory.CALM] = 0.6
                probs[EmotionCategory.NEUTRAL] = 0.3
            elif valence < -0.2:
                probs[EmotionCategory.SAD] = 0.6
                probs[EmotionCategory.NEUTRAL] = 0.3
            else:
                probs[EmotionCategory.NEUTRAL] = 0.7
                probs[EmotionCategory.CALM] = 0.2
        
        elif arousal > 0.3:
            # High arousal
            if valence > 0.2:
                probs[EmotionCategory.HAPPY] = 0.4
                probs[EmotionCategory.EXCITED] = 0.4
            elif valence < -0.2:
                if dominance > 0.2:
                    probs[EmotionCategory.ANGRY] = 0.6
                    probs[EmotionCategory.DISGUSTED] = 0.2
                else:
                    probs[EmotionCategory.FEARFUL] = 0.5
                    probs[EmotionCategory.SURPRISED] = 0.3
            else:
                probs[EmotionCategory.SURPRISED] = 0.5
                probs[EmotionCategory.EXCITED] = 0.3
        
        else:
            # Moderate arousal
            if valence > 0.2:
                probs[EmotionCategory.HAPPY] = 0.5
                probs[EmotionCategory.NEUTRAL] = 0.3
            elif valence < -0.2:
                probs[EmotionCategory.SAD] = 0.4
                probs[EmotionCategory.ANGRY] = 0.3
            else:
                probs[EmotionCategory.NEUTRAL] = 0.6
                probs[EmotionCategory.CALM] = 0.2
        
        # Normalize probabilities
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        
        # Get primary emotion
        primary = max(probs, key=probs.get)
        confidence = probs[primary]
        
        # Intensity from energy and pitch variation
        intensity = min(1.0, energy_ratio * 0.5 + (pitch_std / 100) * 0.3)
        
        return EmotionResult(
            emotion=primary,
            confidence=confidence,
            probabilities=probs,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            intensity=intensity,
            pitch_mean=pitch,
            pitch_std=pitch_std,
            energy=energy,
            timestamp=time.time()
        )


class VoiceEmotionDetector:
    """
    Main voice emotion detection class.
    
    Can analyze audio files or stream from microphone.
    """
    
    def __init__(self, config: Optional[VoiceEmotionConfig] = None):
        self.config = config or VoiceEmotionConfig()
        
        self._feature_extractor = AudioFeatureExtractor(self.config)
        self._classifier = RuleBasedEmotionClassifier()
        
        # Streaming
        self._stream = None
        self._running = False
        self._stream_thread: Optional[threading.Thread] = None
        self._audio_buffer: List[float] = []
        self._lock = threading.Lock()
        
        # Current result
        self._current_result = EmotionResult()
        self._prev_result = EmotionResult()
        
        # Callbacks
        self._callbacks: List[Callable[[EmotionResult], None]] = []
        
        # Avatar connection
        self._avatar_controller: Optional[Any] = None
        
        logger.info("VoiceEmotionDetector initialized")
    
    def is_available(self) -> bool:
        """Check if required libraries are installed."""
        return NUMPY_AVAILABLE and LIBROSA_AVAILABLE
    
    def analyze_file(self, path: str) -> EmotionResult:
        """
        Analyze emotion from audio file.
        
        Args:
            path: Path to audio file (WAV, MP3, etc.)
            
        Returns:
            EmotionResult
        """
        if not self.is_available():
            logger.error("Required libraries not installed. Run: pip install librosa numpy")
            return EmotionResult()
        
        try:
            # Load audio
            audio, sr = librosa.load(path, sr=self.config.sample_rate, mono=True)
            
            return self.analyze_audio(audio, sr)
            
        except Exception as e:
            logger.error(f"Failed to analyze file: {e}")
            return EmotionResult()
    
    def analyze_audio(self, audio: 'np.ndarray', sr: int) -> EmotionResult:
        """
        Analyze emotion from audio array.
        
        Args:
            audio: Audio samples (mono, float32)
            sr: Sample rate
            
        Returns:
            EmotionResult
        """
        if not self.is_available():
            return EmotionResult()
        
        # Check for silence
        if np.max(np.abs(audio)) < self.config.silence_threshold:
            return EmotionResult(emotion=EmotionCategory.NEUTRAL, confidence=0.5)
        
        # Extract features
        features = self._feature_extractor.extract(audio, sr)
        
        # Classify
        result = self._classifier.classify(features)
        result.duration = len(audio) / sr
        
        return result
    
    def start_stream(self, device: Optional[int] = None) -> bool:
        """
        Start real-time emotion detection from microphone.
        
        Args:
            device: Audio input device ID (None = default)
            
        Returns:
            True if started successfully
        """
        if not SOUNDDEVICE_AVAILABLE:
            logger.error("sounddevice not installed. Run: pip install sounddevice")
            return False
        
        if self._running:
            return True
        
        try:
            self._running = True
            self._audio_buffer = []
            
            # Audio callback
            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio status: {status}")
                with self._lock:
                    self._audio_buffer.extend(indata[:, 0].tolist())
            
            # Start input stream
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype='float32',
                callback=audio_callback,
                device=device
            )
            self._stream.start()
            
            # Start analysis thread
            self._stream_thread = threading.Thread(
                target=self._analysis_loop,
                daemon=True
            )
            self._stream_thread.start()
            
            logger.info("Voice emotion stream started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stream: {e}")
            self._running = False
            return False
    
    def stop_stream(self):
        """Stop real-time detection."""
        self._running = False
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None
        
        logger.info("Voice emotion stream stopped")
    
    def _analysis_loop(self):
        """Continuous analysis of buffered audio."""
        min_samples = int(self.config.min_segment_length * self.config.sample_rate)
        
        while self._running:
            time.sleep(self.config.update_interval)
            
            # Get audio from buffer
            with self._lock:
                if len(self._audio_buffer) < min_samples:
                    continue
                
                audio = np.array(self._audio_buffer[-min_samples:], dtype=np.float32)
                # Keep some overlap
                self._audio_buffer = self._audio_buffer[-min_samples // 2:]
            
            # Analyze
            result = self.analyze_audio(audio, self.config.sample_rate)
            
            # Smooth with previous
            result = self._smooth_result(result)
            
            with self._lock:
                self._current_result = result
                self._prev_result = result
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            # Update avatar if connected
            if self._avatar_controller:
                self._update_avatar(result)
    
    def _smooth_result(self, result: EmotionResult) -> EmotionResult:
        """Apply smoothing to emotion result."""
        factor = self.config.smooth_factor
        
        result.valence = self._prev_result.valence + (result.valence - self._prev_result.valence) * factor
        result.arousal = self._prev_result.arousal + (result.arousal - self._prev_result.arousal) * factor
        result.intensity = self._prev_result.intensity + (result.intensity - self._prev_result.intensity) * factor
        
        return result
    
    def get_current_emotion(self) -> EmotionResult:
        """Get most recent emotion result."""
        with self._lock:
            return EmotionResult(**self._current_result.__dict__)
    
    def add_callback(self, callback: Callable[[EmotionResult], None]):
        """Add callback for emotion updates."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[EmotionResult], None]):
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def connect_to_avatar(self, controller: Any):
        """
        Connect to avatar controller for automatic emotion sync.
        
        Args:
            controller: AIAvatarController instance
        """
        self._avatar_controller = controller
        logger.info("Voice emotion connected to avatar")
    
    def disconnect_from_avatar(self):
        """Disconnect from avatar controller."""
        self._avatar_controller = None
    
    def _update_avatar(self, result: EmotionResult):
        """Update avatar based on detected emotion."""
        if not self._avatar_controller:
            return
        
        # Only update if confident enough
        if result.confidence < self.config.min_confidence:
            return
        
        # Map to avatar emotion
        emotion_map = {
            EmotionCategory.NEUTRAL: "NEUTRAL",
            EmotionCategory.HAPPY: "HAPPY",
            EmotionCategory.SAD: "SAD",
            EmotionCategory.ANGRY: "ANGRY",
            EmotionCategory.FEARFUL: "FEARFUL",
            EmotionCategory.SURPRISED: "SURPRISED",
            EmotionCategory.CALM: "NEUTRAL",
            EmotionCategory.EXCITED: "EXCITED",
            EmotionCategory.DISGUSTED: "DISGUSTED",
        }
        
        avatar_emotion = emotion_map.get(result.emotion, "NEUTRAL")
        
        try:
            self._avatar_controller.express_emotion(
                avatar_emotion,
                intensity=result.intensity * result.confidence
            )
        except Exception as e:
            logger.error(f"Failed to update avatar: {e}")
    
    def calibrate(self, duration: float = 5.0):
        """
        Calibrate baselines from neutral speech.
        
        Have the user speak in a neutral tone for the duration.
        
        Args:
            duration: Calibration duration in seconds
        """
        if not SOUNDDEVICE_AVAILABLE:
            logger.error("Cannot calibrate without sounddevice")
            return
        
        logger.info(f"Calibrating for {duration} seconds. Please speak neutrally...")
        
        # Record audio
        samples = int(duration * self.config.sample_rate)
        audio = sd.rec(samples, samplerate=self.config.sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        
        # Extract features
        features = self._feature_extractor.extract(audio, self.config.sample_rate)
        
        # Update baselines
        self._classifier.baseline_pitch = features.get('pitch_mean', 150.0)
        self._classifier.baseline_energy = features.get('energy_mean', 0.1)
        self._classifier.baseline_tempo = features.get('tempo', 120.0)
        
        logger.info(f"Calibration complete. Baseline pitch: {self._classifier.baseline_pitch:.1f}Hz")


# Convenience functions

def is_voice_emotion_available() -> bool:
    """Check if voice emotion detection is available."""
    return NUMPY_AVAILABLE and LIBROSA_AVAILABLE


def get_voice_emotion_requirements() -> str:
    """Get installation instructions."""
    return """
Voice Emotion Detection Requirements:

Core (required):
    pip install librosa numpy scipy

Real-time streaming:
    pip install sounddevice

Neural model (optional, more accurate):
    pip install speechbrain

How it works:
- Extracts acoustic features (pitch, energy, MFCCs)
- Maps features to emotion dimensions (valence, arousal)
- Classifies into emotion categories

For best results:
1. Use in a quiet environment
2. Calibrate for your voice
3. Speak naturally (not whispered or shouted)
"""
