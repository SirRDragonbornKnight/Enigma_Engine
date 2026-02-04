"""
================================================================================
CUSTOM WAKE WORD TRAINER
================================================================================

Train custom wake words from user voice samples.

Features:
- Record wake word samples from microphone
- Train a simple keyword detection model
- Support for multiple wake word phrases
- Export/import trained models
- Integration with VoicePipeline

FILE: forge_ai/voice/wake_word_trainer.py
TYPE: Voice Recognition Training
MAIN CLASSES: WakeWordTrainer, WakeWordModel

USAGE:
    from forge_ai.voice.wake_word_trainer import WakeWordTrainer
    
    trainer = WakeWordTrainer()
    
    # Record samples
    trainer.record_samples("hey forge", num_samples=5)
    
    # Train model
    model = trainer.train()
    
    # Use in pipeline
    pipeline.set_custom_wake_word(model)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import struct
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import CONFIG

logger = logging.getLogger(__name__)

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


# Constants
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_SAMPLE_DURATION = 2.0  # seconds per sample
MIN_SAMPLES_REQUIRED = 3
RECOMMENDED_SAMPLES = 10


@dataclass
class WakeWordSample:
    """A recorded wake word sample."""
    
    audio_path: Path
    duration: float
    sample_rate: int
    wake_phrase: str
    is_positive: bool = True  # True for wake word, False for background
    recorded_at: str = ""
    
    def __post_init__(self):
        if not self.recorded_at:
            from datetime import datetime
            self.recorded_at = datetime.now().isoformat()


@dataclass
class WakeWordModel:
    """Trained wake word detection model."""
    
    wake_phrase: str
    model_type: str = "mfcc_dtw"  # mfcc_dtw, embedding, neural
    
    # Model data
    templates: List['np.ndarray'] = field(default_factory=list)
    threshold: float = 0.7
    
    # Metadata
    num_samples: int = 0
    trained_at: str = ""
    version: str = "1.0"
    
    def __post_init__(self):
        if not self.trained_at:
            from datetime import datetime
            self.trained_at = datetime.now().isoformat()
    
    def save(self, path: Path):
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'wake_phrase': self.wake_phrase,
            'model_type': self.model_type,
            'threshold': self.threshold,
            'num_samples': self.num_samples,
            'trained_at': self.trained_at,
            'version': self.version,
        }
        
        # Save templates separately (numpy arrays)
        if self.templates and NUMPY_AVAILABLE:
            templates_list = [t.tolist() for t in self.templates]
            data['templates'] = templates_list
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved wake word model to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'WakeWordModel':
        """Load model from file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        model = cls(
            wake_phrase=data['wake_phrase'],
            model_type=data.get('model_type', 'mfcc_dtw'),
            threshold=data.get('threshold', 0.7),
            num_samples=data.get('num_samples', 0),
            trained_at=data.get('trained_at', ''),
            version=data.get('version', '1.0'),
        )
        
        # Load templates
        if 'templates' in data and NUMPY_AVAILABLE:
            model.templates = [np.array(t) for t in data['templates']]
        
        return model
    
    def detect(self, audio: 'np.ndarray', sample_rate: int) -> Tuple[bool, float]:
        """
        Detect wake word in audio.
        
        Args:
            audio: Audio samples as numpy array
            sample_rate: Sample rate
            
        Returns:
            Tuple of (detected, confidence)
        """
        if not self.templates or not NUMPY_AVAILABLE:
            return False, 0.0
        
        try:
            # Extract features from input audio
            input_features = _extract_mfcc(audio, sample_rate)
            
            if input_features is None:
                return False, 0.0
            
            # Compare to templates using DTW
            best_distance = float('inf')
            for template in self.templates:
                distance = _dtw_distance(input_features, template)
                best_distance = min(best_distance, distance)
            
            # Convert distance to confidence (lower distance = higher confidence)
            # Normalize based on typical distances
            confidence = max(0, 1 - (best_distance / 100))
            
            detected = confidence >= self.threshold
            
            return detected, confidence
            
        except Exception as e:
            logger.debug(f"Wake word detection error: {e}")
            return False, 0.0


class WakeWordTrainer:
    """
    Train custom wake word models from voice samples.
    
    Provides an easy-to-use interface for:
    - Recording wake word samples
    - Recording background/negative samples
    - Training detection models
    - Testing and validating models
    """
    
    def __init__(self, samples_dir: Path = None):
        """
        Initialize trainer.
        
        Args:
            samples_dir: Directory to store recorded samples
        """
        self.samples_dir = samples_dir or (CONFIG.data_dir / "wake_words")
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = CONFIG.models_dir / "wake_words"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Recording state
        self._recording = False
        self._recorded_samples: List[WakeWordSample] = []
        
        # Callbacks
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback for progress updates: (current, total, message)."""
        self._progress_callback = callback
    
    def _report_progress(self, current: int, total: int, message: str):
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(current, total, message)
        logger.info(f"[{current}/{total}] {message}")
    
    def record_samples(
        self,
        wake_phrase: str,
        num_samples: int = RECOMMENDED_SAMPLES,
        sample_duration: float = DEFAULT_SAMPLE_DURATION,
        delay_between: float = 1.5
    ) -> List[WakeWordSample]:
        """
        Record wake word samples from microphone.
        
        Args:
            wake_phrase: The wake word/phrase to record
            num_samples: Number of samples to record
            sample_duration: Duration of each sample in seconds
            delay_between: Delay between recordings in seconds
            
        Returns:
            List of recorded samples
        """
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio not available for recording")
        
        samples = []
        phrase_dir = self.samples_dir / _sanitize_filename(wake_phrase)
        phrase_dir.mkdir(parents=True, exist_ok=True)
        
        self._report_progress(0, num_samples, f"Preparing to record '{wake_phrase}'...")
        
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=DEFAULT_CHANNELS,
                rate=DEFAULT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=DEFAULT_CHUNK_SIZE
            )
            
            for i in range(num_samples):
                self._report_progress(i, num_samples, f"Say '{wake_phrase}' now...")
                
                # Record sample
                frames = []
                num_chunks = int(DEFAULT_SAMPLE_RATE / DEFAULT_CHUNK_SIZE * sample_duration)
                
                for _ in range(num_chunks):
                    data = stream.read(DEFAULT_CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(data)
                
                # Save to file
                timestamp = int(time.time() * 1000)
                filename = f"sample_{i+1}_{timestamp}.wav"
                filepath = phrase_dir / filename
                
                with wave.open(str(filepath), 'wb') as wf:
                    wf.setnchannels(DEFAULT_CHANNELS)
                    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(DEFAULT_SAMPLE_RATE)
                    wf.writeframes(b''.join(frames))
                
                sample = WakeWordSample(
                    audio_path=filepath,
                    duration=sample_duration,
                    sample_rate=DEFAULT_SAMPLE_RATE,
                    wake_phrase=wake_phrase,
                    is_positive=True
                )
                samples.append(sample)
                
                self._report_progress(i + 1, num_samples, f"Recorded sample {i+1}/{num_samples}")
                
                # Delay before next recording
                if i < num_samples - 1:
                    time.sleep(delay_between)
            
            stream.stop_stream()
            stream.close()
            
        finally:
            audio.terminate()
        
        self._recorded_samples.extend(samples)
        self._report_progress(num_samples, num_samples, f"Recording complete! {num_samples} samples saved.")
        
        return samples
    
    def record_background_samples(
        self,
        num_samples: int = 20,
        sample_duration: float = DEFAULT_SAMPLE_DURATION
    ) -> List[WakeWordSample]:
        """
        Record background/negative samples for better accuracy.
        
        These are samples of normal speech or ambient sound that
        should NOT trigger the wake word.
        
        Args:
            num_samples: Number of background samples
            sample_duration: Duration of each sample
            
        Returns:
            List of background samples
        """
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio not available for recording")
        
        samples = []
        bg_dir = self.samples_dir / "_background"
        bg_dir.mkdir(parents=True, exist_ok=True)
        
        self._report_progress(0, num_samples, "Recording background samples (talk normally, don't say wake word)...")
        
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=DEFAULT_CHANNELS,
                rate=DEFAULT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=DEFAULT_CHUNK_SIZE
            )
            
            for i in range(num_samples):
                # Record sample
                frames = []
                num_chunks = int(DEFAULT_SAMPLE_RATE / DEFAULT_CHUNK_SIZE * sample_duration)
                
                for _ in range(num_chunks):
                    data = stream.read(DEFAULT_CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(data)
                
                # Save to file
                timestamp = int(time.time() * 1000)
                filename = f"background_{i+1}_{timestamp}.wav"
                filepath = bg_dir / filename
                
                with wave.open(str(filepath), 'wb') as wf:
                    wf.setnchannels(DEFAULT_CHANNELS)
                    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(DEFAULT_SAMPLE_RATE)
                    wf.writeframes(b''.join(frames))
                
                sample = WakeWordSample(
                    audio_path=filepath,
                    duration=sample_duration,
                    sample_rate=DEFAULT_SAMPLE_RATE,
                    wake_phrase="_background",
                    is_positive=False
                )
                samples.append(sample)
                
                self._report_progress(i + 1, num_samples, f"Background sample {i+1}/{num_samples}")
            
            stream.stop_stream()
            stream.close()
            
        finally:
            audio.terminate()
        
        self._report_progress(num_samples, num_samples, f"Background recording complete!")
        
        return samples
    
    def add_sample_from_file(
        self,
        audio_path: Path,
        wake_phrase: str,
        is_positive: bool = True
    ) -> WakeWordSample:
        """
        Add an existing audio file as a sample.
        
        Args:
            audio_path: Path to audio file
            wake_phrase: Associated wake phrase
            is_positive: Whether this is a positive sample
            
        Returns:
            WakeWordSample object
        """
        audio_path = Path(audio_path)
        
        # Get audio info
        with wave.open(str(audio_path), 'rb') as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / sample_rate
        
        sample = WakeWordSample(
            audio_path=audio_path,
            duration=duration,
            sample_rate=sample_rate,
            wake_phrase=wake_phrase,
            is_positive=is_positive
        )
        
        self._recorded_samples.append(sample)
        return sample
    
    def train(
        self,
        wake_phrase: str = None,
        samples: List[WakeWordSample] = None,
        model_type: str = "mfcc_dtw"
    ) -> WakeWordModel:
        """
        Train a wake word detection model.
        
        Args:
            wake_phrase: Wake phrase to train (uses recorded if None)
            samples: Samples to use (uses recorded if None)
            model_type: Model type ('mfcc_dtw' for template matching)
            
        Returns:
            Trained WakeWordModel
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy required for training")
        
        # Use recorded samples if not provided
        if samples is None:
            samples = self._recorded_samples
        
        # Filter to positive samples
        positive_samples = [s for s in samples if s.is_positive]
        
        if wake_phrase:
            positive_samples = [s for s in positive_samples if s.wake_phrase == wake_phrase]
        elif positive_samples:
            wake_phrase = positive_samples[0].wake_phrase
        else:
            raise ValueError("No samples available for training")
        
        if len(positive_samples) < MIN_SAMPLES_REQUIRED:
            raise ValueError(
                f"Need at least {MIN_SAMPLES_REQUIRED} samples, got {len(positive_samples)}"
            )
        
        self._report_progress(0, len(positive_samples), f"Training model for '{wake_phrase}'...")
        
        # Extract features from all samples
        templates = []
        for i, sample in enumerate(positive_samples):
            try:
                audio = self._load_audio(sample.audio_path)
                features = _extract_mfcc(audio, sample.sample_rate)
                
                if features is not None:
                    templates.append(features)
                    
                self._report_progress(i + 1, len(positive_samples), f"Processed sample {i+1}")
                
            except Exception as e:
                logger.warning(f"Failed to process {sample.audio_path}: {e}")
        
        if len(templates) < MIN_SAMPLES_REQUIRED:
            raise ValueError(f"Only {len(templates)} valid samples after processing")
        
        # Determine optimal threshold using background samples
        negative_samples = [s for s in samples if not s.is_positive]
        threshold = self._calculate_threshold(templates, negative_samples)
        
        model = WakeWordModel(
            wake_phrase=wake_phrase,
            model_type=model_type,
            templates=templates,
            threshold=threshold,
            num_samples=len(templates)
        )
        
        # Save model
        model_path = self.models_dir / f"{_sanitize_filename(wake_phrase)}.json"
        model.save(model_path)
        
        self._report_progress(
            len(positive_samples), 
            len(positive_samples), 
            f"Training complete! Model saved to {model_path}"
        )
        
        return model
    
    def _load_audio(self, path: Path) -> 'np.ndarray':
        """Load audio file as numpy array."""
        with wave.open(str(path), 'rb') as wf:
            n_frames = wf.getnframes()
            sample_width = wf.getsampwidth()
            audio_bytes = wf.readframes(n_frames)
            
            if sample_width == 2:
                audio = np.array(
                    struct.unpack(f'{n_frames}h', audio_bytes),
                    dtype=np.float32
                ) / 32768.0
            else:
                audio = np.frombuffer(audio_bytes, dtype=np.float32)
            
            return audio
    
    def _calculate_threshold(
        self,
        templates: List['np.ndarray'],
        negative_samples: List[WakeWordSample]
    ) -> float:
        """Calculate optimal detection threshold."""
        # Calculate distances between templates (expected range)
        template_distances = []
        for i, t1 in enumerate(templates):
            for j, t2 in enumerate(templates):
                if i < j:
                    dist = _dtw_distance(t1, t2)
                    template_distances.append(dist)
        
        if not template_distances:
            return 0.7  # Default
        
        # Use mean + std of template distances to set threshold
        mean_dist = np.mean(template_distances)
        std_dist = np.std(template_distances)
        
        # Threshold should reject samples with distance > mean + 2*std
        # Convert to confidence: threshold = 1 - (mean + 2*std) / 100
        threshold = max(0.5, min(0.95, 1 - (mean_dist + 2 * std_dist) / 100))
        
        logger.debug(f"Calculated threshold: {threshold:.3f} (mean_dist={mean_dist:.2f})")
        
        return threshold
    
    def test_model(
        self,
        model: WakeWordModel,
        test_samples: List[WakeWordSample] = None
    ) -> Dict[str, Any]:
        """
        Test a trained model on samples.
        
        Args:
            model: Trained model to test
            test_samples: Samples to test on (uses recorded if None)
            
        Returns:
            Dict with accuracy metrics
        """
        if test_samples is None:
            test_samples = self._recorded_samples
        
        results = {
            'total': len(test_samples),
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'details': []
        }
        
        for sample in test_samples:
            try:
                audio = self._load_audio(sample.audio_path)
                detected, confidence = model.detect(audio, sample.sample_rate)
                
                is_match = sample.wake_phrase == model.wake_phrase
                
                if detected and is_match:
                    results['true_positives'] += 1
                elif not detected and not is_match:
                    results['true_negatives'] += 1
                elif detected and not is_match:
                    results['false_positives'] += 1
                else:
                    results['false_negatives'] += 1
                
                results['details'].append({
                    'file': str(sample.audio_path),
                    'expected': is_match,
                    'detected': detected,
                    'confidence': confidence
                })
                
            except Exception as e:
                logger.warning(f"Test failed for {sample.audio_path}: {e}")
        
        # Calculate metrics
        tp = results['true_positives']
        tn = results['true_negatives']
        fp = results['false_positives']
        fn = results['false_negatives']
        
        total = tp + tn + fp + fn
        results['accuracy'] = (tp + tn) / total if total > 0 else 0
        results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return results
    
    def list_available_models(self) -> List[str]:
        """List trained wake word models."""
        models = []
        for path in self.models_dir.glob("*.json"):
            try:
                model = WakeWordModel.load(path)
                models.append(model.wake_phrase)
            except Exception:
                pass
        return models
    
    def load_model(self, wake_phrase: str) -> Optional[WakeWordModel]:
        """Load a trained model by wake phrase."""
        model_path = self.models_dir / f"{_sanitize_filename(wake_phrase)}.json"
        
        if model_path.exists():
            return WakeWordModel.load(model_path)
        
        # Try exact filename match
        for path in self.models_dir.glob("*.json"):
            try:
                model = WakeWordModel.load(path)
                if model.wake_phrase.lower() == wake_phrase.lower():
                    return model
            except Exception:
                pass
        
        return None
    
    def delete_model(self, wake_phrase: str) -> bool:
        """Delete a trained model."""
        model_path = self.models_dir / f"{_sanitize_filename(wake_phrase)}.json"
        
        if model_path.exists():
            model_path.unlink()
            logger.info(f"Deleted model for '{wake_phrase}'")
            return True
        
        return False


# =============================================================================
# Helper Functions
# =============================================================================

def _sanitize_filename(text: str) -> str:
    """Sanitize text for use as filename."""
    # Replace spaces and special chars
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in text)
    return safe.lower()


def _extract_mfcc(
    audio: 'np.ndarray',
    sample_rate: int,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 256
) -> Optional['np.ndarray']:
    """
    Extract MFCC features from audio.
    
    MFCCs (Mel-Frequency Cepstral Coefficients) capture the
    spectral characteristics of speech in a compact form.
    """
    if not NUMPY_AVAILABLE:
        return None
    
    try:
        # Try librosa first (best quality)
        try:
            import librosa
            mfcc = librosa.feature.mfcc(
                y=audio, sr=sample_rate,
                n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
            )
            return mfcc.T  # (time, features)
        except ImportError:
            pass
        
        # Fallback: simple MFCC implementation
        return _simple_mfcc(audio, sample_rate, n_mfcc, n_fft, hop_length)
        
    except Exception as e:
        logger.debug(f"MFCC extraction failed: {e}")
        return None


def _simple_mfcc(
    audio: 'np.ndarray',
    sample_rate: int,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 256
) -> 'np.ndarray':
    """Simple MFCC implementation without librosa."""
    import numpy as np
    from scipy.fftpack import dct
    
    # Pre-emphasis
    pre_emphasis = 0.97
    emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    # Frame the signal
    frame_length = n_fft
    num_frames = 1 + (len(emphasized) - frame_length) // hop_length
    
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start = i * hop_length
        frames[i] = emphasized[start:start + frame_length] * np.hamming(frame_length)
    
    # FFT and power spectrum
    fft_frames = np.fft.rfft(frames, n_fft)
    power_spectrum = np.abs(fft_frames) ** 2
    
    # Mel filterbank
    n_mels = 26
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        for j in range(bin_points[i], bin_points[i + 1]):
            filterbank[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
        for j in range(bin_points[i + 1], bin_points[i + 2]):
            filterbank[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])
    
    # Apply filterbank
    mel_spectrum = np.dot(power_spectrum, filterbank.T)
    mel_spectrum = np.where(mel_spectrum == 0, np.finfo(float).eps, mel_spectrum)
    log_mel_spectrum = np.log(mel_spectrum)
    
    # DCT to get MFCCs
    mfcc = dct(log_mel_spectrum, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    
    return mfcc


def _dtw_distance(seq1: 'np.ndarray', seq2: 'np.ndarray') -> float:
    """
    Compute Dynamic Time Warping distance between two sequences.
    
    DTW allows comparison of sequences of different lengths by
    finding the optimal alignment between them.
    """
    import numpy as np
    
    n, m = len(seq1), len(seq2)
    
    if n == 0 or m == 0:
        return float('inf')
    
    # Cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    
    return dtw[n, m] / (n + m)  # Normalize by path length


# =============================================================================
# Convenience Functions
# =============================================================================

def create_trainer(samples_dir: Path = None) -> WakeWordTrainer:
    """Create a WakeWordTrainer instance."""
    return WakeWordTrainer(samples_dir)


def train_wake_word(
    wake_phrase: str,
    num_samples: int = RECOMMENDED_SAMPLES,
    include_background: bool = True
) -> WakeWordModel:
    """
    Convenient function to record samples and train a wake word model.
    
    Args:
        wake_phrase: The wake word/phrase to train
        num_samples: Number of wake word samples to record
        include_background: Whether to record background samples
        
    Returns:
        Trained WakeWordModel
    """
    trainer = WakeWordTrainer()
    
    # Record wake word samples
    trainer.record_samples(wake_phrase, num_samples)
    
    # Optionally record background samples
    if include_background:
        trainer.record_background_samples(num_samples=10)
    
    # Train model
    return trainer.train(wake_phrase)


def load_wake_word_model(wake_phrase: str) -> Optional[WakeWordModel]:
    """Load a previously trained wake word model."""
    trainer = WakeWordTrainer()
    return trainer.load_model(wake_phrase)
