"""
================================================================================
Audio File Input - Transcribe audio files, not just microphone.
================================================================================

Provides file-based speech-to-text transcription:
- Multiple audio format support (WAV, MP3, FLAC, OGG, M4A)
- Batch transcription for multiple files
- Automatic format conversion
- Timestamp extraction
- Speaker segmentation (optional)

USAGE:
    from enigma_engine.voice.audio_file_input import AudioFileTranscriber
    
    transcriber = AudioFileTranscriber()
    
    # Transcribe single file
    result = transcriber.transcribe("recording.wav")
    print(result.text)
    
    # Transcribe with timestamps
    result = transcriber.transcribe("audio.mp3", timestamps=True)
    for segment in result.segments:
        print(f"[{segment.start:.1f}s] {segment.text}")
    
    # Batch transcription
    results = transcriber.transcribe_batch(["file1.wav", "file2.mp3"])
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import wave
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class TranscriptionBackend(Enum):
    """Available transcription backends."""
    
    WHISPER = auto()           # OpenAI Whisper (local)
    WHISPER_API = auto()       # OpenAI Whisper API
    VOSK = auto()              # Vosk offline
    SPEECH_RECOGNITION = auto() # SpeechRecognition library
    DEEPSPEECH = auto()        # Mozilla DeepSpeech


class AudioFormat(Enum):
    """Supported audio formats."""
    
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    WEBM = "webm"
    AAC = "aac"
    WMA = "wma"
    OPUS = "opus"


@dataclass
class TranscriptionConfig:
    """Configuration for audio file transcription."""
    
    # Backend settings
    preferred_backend: TranscriptionBackend = TranscriptionBackend.WHISPER
    model_size: str = "base"      # tiny, base, small, medium, large
    
    # Language settings
    language: str = "en"          # Language code or "auto"
    
    # Output settings
    include_timestamps: bool = False
    include_confidence: bool = False
    include_word_timestamps: bool = False
    
    # Processing settings
    sample_rate: int = 16000      # Target sample rate
    normalize_audio: bool = True  # Normalize volume
    remove_silence: bool = False  # Remove silent sections
    
    # Batch settings
    parallel_jobs: int = 1        # Parallel transcription jobs
    
    # Temp file handling
    cleanup_temp: bool = True     # Delete temp files


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio with timing."""
    
    text: str
    start: float                  # Start time in seconds
    end: float                    # End time in seconds
    confidence: float = 1.0       # Confidence score
    speaker: str | None = None # Speaker ID (if diarization enabled)
    words: list[dict] = field(default_factory=list)  # Word-level timestamps


@dataclass
class TranscriptionResult:
    """Result of audio file transcription."""
    
    text: str                     # Full transcription text
    segments: list[TranscriptionSegment] = field(default_factory=list)
    duration: float = 0.0         # Audio duration in seconds
    language: str = "en"          # Detected/specified language
    confidence: float = 1.0       # Overall confidence
    
    # Metadata
    source_file: str = ""
    backend: str = ""
    processing_time: float = 0.0


class AudioFileTranscriber:
    """
    Transcribe audio files to text.
    
    Supports multiple formats and backends with automatic
    format conversion and preprocessing.
    """
    
    def __init__(self, config: TranscriptionConfig = None):
        self.config = config or TranscriptionConfig()
        
        # Backend engine (lazy loaded)
        self._engine = None
        self._backend: TranscriptionBackend = self.config.preferred_backend
        
        # Supported formats
        self._supported_formats = {fmt.value for fmt in AudioFormat}
    
    def _init_engine(self):
        """Initialize transcription engine."""
        if self._engine is not None:
            return
        
        backend = self.config.preferred_backend
        
        if backend == TranscriptionBackend.WHISPER:
            self._init_whisper()
        elif backend == TranscriptionBackend.VOSK:
            self._init_vosk()
        elif backend == TranscriptionBackend.SPEECH_RECOGNITION:
            self._init_speech_recognition()
        else:
            # Fallback chain
            if not self._init_whisper():
                if not self._init_vosk():
                    self._init_speech_recognition()
    
    def _init_whisper(self) -> bool:
        """Initialize Whisper backend."""
        try:
            import whisper
            
            model = whisper.load_model(self.config.model_size)
            self._engine = {"type": "whisper", "model": model}
            self._backend = TranscriptionBackend.WHISPER
            logger.info(f"Initialized Whisper ({self.config.model_size})")
            return True
            
        except ImportError:
            logger.debug("Whisper not available")
            return False
        except Exception as e:
            logger.warning(f"Failed to init Whisper: {e}")
            return False
    
    def _init_vosk(self) -> bool:
        """Initialize Vosk backend."""
        try:
            from vosk import KaldiRecognizer, Model

            # Find model path
            model_paths = [
                Path.home() / ".cache/vosk/model",
                Path("/usr/share/vosk/model"),
                Path("models/vosk"),
            ]
            
            model_path = None
            for p in model_paths:
                if p.exists():
                    model_path = p
                    break
            
            if not model_path:
                logger.debug("Vosk model not found")
                return False
            
            model = Model(str(model_path))
            self._engine = {"type": "vosk", "model": model, "KaldiRecognizer": KaldiRecognizer}
            self._backend = TranscriptionBackend.VOSK
            logger.info("Initialized Vosk")
            return True
            
        except ImportError:
            logger.debug("Vosk not available")
            return False
        except Exception as e:
            logger.warning(f"Failed to init Vosk: {e}")
            return False
    
    def _init_speech_recognition(self) -> bool:
        """Initialize SpeechRecognition backend."""
        try:
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            self._engine = {"type": "speech_recognition", "recognizer": recognizer, "sr": sr}
            self._backend = TranscriptionBackend.SPEECH_RECOGNITION
            logger.info("Initialized SpeechRecognition")
            return True
            
        except ImportError:
            logger.debug("SpeechRecognition not available")
            return False
    
    def transcribe(
        self,
        file_path: str | Path,
        timestamps: bool = None,
        language: str = None
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            file_path: Path to audio file
            timestamps: Include timestamps (overrides config)
            language: Language code (overrides config)
        
        Returns:
            TranscriptionResult with text and metadata
        """
        import time
        start_time = time.time()
        
        self._init_engine()
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Override config if specified
        include_timestamps = timestamps if timestamps is not None else self.config.include_timestamps
        lang = language or self.config.language
        
        # Convert to WAV if needed
        wav_path = self._ensure_wav(file_path)
        
        try:
            # Load audio
            audio, duration = self._load_audio(wav_path)
            
            # Preprocess
            if self.config.normalize_audio:
                audio = self._normalize(audio)
            
            # Transcribe based on backend
            if self._backend == TranscriptionBackend.WHISPER:
                result = self._transcribe_whisper(audio, lang, include_timestamps)
            elif self._backend == TranscriptionBackend.VOSK:
                result = self._transcribe_vosk(wav_path, lang)
            elif self._backend == TranscriptionBackend.SPEECH_RECOGNITION:
                result = self._transcribe_sr(wav_path, lang)
            else:
                result = TranscriptionResult(text="", source_file=str(file_path))
            
            # Add metadata
            result.source_file = str(file_path)
            result.duration = duration
            result.backend = self._backend.name
            result.processing_time = time.time() - start_time
            
            return result
            
        finally:
            # Cleanup temp file
            if self.config.cleanup_temp and wav_path != file_path:
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass  # Intentionally silent
    
    def transcribe_batch(
        self,
        file_paths: list[str | Path],
        progress_callback: Callable[[int, int, str], None] = None
    ) -> list[TranscriptionResult]:
        """
        Transcribe multiple audio files.
        
        Args:
            file_paths: List of audio file paths
            progress_callback: Called with (current, total, filename)
        
        Returns:
            List of TranscriptionResult objects
        """
        results = []
        total = len(file_paths)
        
        for i, path in enumerate(file_paths):
            if progress_callback:
                progress_callback(i + 1, total, str(path))
            
            try:
                result = self.transcribe(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to transcribe {path}: {e}")
                results.append(TranscriptionResult(
                    text="",
                    source_file=str(path)
                ))
        
        return results
    
    def transcribe_stream(
        self,
        file_path: str | Path,
        chunk_seconds: float = 30.0
    ) -> Generator[TranscriptionSegment]:
        """
        Stream transcription for long audio files.
        
        Yields segments as they are transcribed.
        
        Args:
            file_path: Path to audio file
            chunk_seconds: Process in chunks of this duration
        
        Yields:
            TranscriptionSegment objects
        """
        self._init_engine()
        
        file_path = Path(file_path)
        wav_path = self._ensure_wav(file_path)
        
        try:
            # Load full audio
            audio, duration = self._load_audio(wav_path)
            
            # Calculate chunk size
            chunk_samples = int(chunk_seconds * self.config.sample_rate)
            
            # Process in chunks
            offset = 0
            chunk_idx = 0
            
            while offset < len(audio):
                chunk = audio[offset:offset + chunk_samples]
                chunk_start = offset / self.config.sample_rate
                
                # Transcribe chunk
                if self._backend == TranscriptionBackend.WHISPER:
                    result = self._transcribe_whisper(chunk, self.config.language, True)
                    
                    # Adjust timestamps
                    for seg in result.segments:
                        seg.start += chunk_start
                        seg.end += chunk_start
                        yield seg
                else:
                    # Non-Whisper backends: single segment per chunk
                    result = self._transcribe_chunk_simple(chunk)
                    if result:
                        yield TranscriptionSegment(
                            text=result,
                            start=chunk_start,
                            end=chunk_start + len(chunk) / self.config.sample_rate
                        )
                
                offset += chunk_samples
                chunk_idx += 1
                
        finally:
            if self.config.cleanup_temp and wav_path != file_path:
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass  # Intentionally silent
    
    def _ensure_wav(self, file_path: Path) -> Path:
        """Convert audio to WAV format if needed."""
        suffix = file_path.suffix.lower().lstrip('.')
        
        if suffix == "wav":
            return file_path
        
        # Convert using ffmpeg
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = Path(f.name)
            
            cmd = [
                "ffmpeg", "-y", "-i", str(file_path),
                "-ar", str(self.config.sample_rate),
                "-ac", "1",
                "-f", "wav",
                str(temp_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300
            )
            
            if result.returncode == 0 and temp_path.exists():
                return temp_path
                
        except FileNotFoundError:
            logger.debug("ffmpeg not found, trying pydub")
        except Exception as e:
            logger.debug(f"ffmpeg conversion failed: {e}")
        
        # Try pydub
        try:
            from pydub import AudioSegment
            
            audio = AudioSegment.from_file(str(file_path))
            audio = audio.set_frame_rate(self.config.sample_rate)
            audio = audio.set_channels(1)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = Path(f.name)
            
            audio.export(str(temp_path), format="wav")
            return temp_path
            
        except ImportError:
            logger.debug("pydub not available")
        except Exception as e:
            logger.debug(f"pydub conversion failed: {e}")
        
        # Return original and hope for the best
        return file_path
    
    def _load_audio(self, wav_path: Path) -> tuple[np.ndarray, float]:
        """Load audio file as numpy array."""
        # Try soundfile first
        try:
            import soundfile as sf
            audio, sr = sf.read(str(wav_path))
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert to mono
            if sr != self.config.sample_rate:
                audio = self._resample(audio, sr, self.config.sample_rate)
            duration = len(audio) / self.config.sample_rate
            return audio.astype(np.float32), duration
        except ImportError:
            pass  # Intentionally silent
        
        # Try wave module
        try:
            with wave.open(str(wav_path), 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_width = wf.getsampwidth()
                sr = wf.getframerate()
                channels = wf.getnchannels()
                
                if sample_width == 2:
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 1:
                    audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                else:
                    audio = np.frombuffer(frames, dtype=np.float32)
                
                # Convert to mono
                if channels > 1:
                    audio = audio.reshape(-1, channels).mean(axis=1)
                
                if sr != self.config.sample_rate:
                    audio = self._resample(audio, sr, self.config.sample_rate)
                
                duration = len(audio) / self.config.sample_rate
                return audio, duration
                
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return np.array([], dtype=np.float32), 0.0
    
    def _resample(self, audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Resample audio to target rate."""
        if src_rate == dst_rate:
            return audio
        
        try:
            import scipy.signal
            num_samples = int(len(audio) * dst_rate / src_rate)
            return scipy.signal.resample(audio, num_samples).astype(np.float32)
        except ImportError:
            # Linear interpolation fallback
            ratio = dst_rate / src_rate
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio volume."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val * 0.95
        return audio
    
    def _transcribe_whisper(
        self,
        audio: np.ndarray,
        language: str,
        timestamps: bool
    ) -> TranscriptionResult:
        """Transcribe using Whisper."""
        model = self._engine["model"]
        
        options = {
            "language": language if language != "auto" else None,
            "task": "transcribe",
        }
        
        result = model.transcribe(audio, **options)
        
        # Build result
        segments = []
        if timestamps and "segments" in result:
            for seg in result["segments"]:
                segments.append(TranscriptionSegment(
                    text=seg["text"].strip(),
                    start=seg["start"],
                    end=seg["end"],
                    confidence=seg.get("no_speech_prob", 0.0)
                ))
        
        return TranscriptionResult(
            text=result["text"].strip(),
            segments=segments,
            language=result.get("language", language),
            confidence=1.0 - result.get("no_speech_prob", 0.0) if "no_speech_prob" in result else 1.0
        )
    
    def _transcribe_vosk(self, wav_path: Path, language: str) -> TranscriptionResult:
        """Transcribe using Vosk."""
        model = self._engine["model"]
        KaldiRecognizer = self._engine["KaldiRecognizer"]
        
        rec = KaldiRecognizer(model, self.config.sample_rate)
        rec.SetWords(True)
        
        with wave.open(str(wav_path), 'rb') as wf:
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)
        
        import json
        result = json.loads(rec.FinalResult())
        
        text = result.get("text", "")
        
        # Extract word timestamps if available
        segments = []
        if "result" in result:
            for word in result["result"]:
                segments.append(TranscriptionSegment(
                    text=word["word"],
                    start=word["start"],
                    end=word["end"],
                    confidence=word.get("conf", 1.0)
                ))
        
        return TranscriptionResult(
            text=text,
            segments=segments,
            language=language
        )
    
    def _transcribe_sr(self, wav_path: Path, language: str) -> TranscriptionResult:
        """Transcribe using SpeechRecognition."""
        recognizer = self._engine["recognizer"]
        sr = self._engine["sr"]
        
        with sr.AudioFile(str(wav_path)) as source:
            audio = recognizer.record(source)
        
        try:
            # Try Google first
            text = recognizer.recognize_google(audio, language=language)
        except Exception:
            try:
                # Fallback to Sphinx (offline)
                text = recognizer.recognize_sphinx(audio)
            except Exception:
                text = ""
        
        return TranscriptionResult(
            text=text,
            language=language
        )
    
    def _transcribe_chunk_simple(self, audio: np.ndarray) -> str:
        """Simple transcription for a chunk (non-Whisper backends)."""
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Write audio
            with wave.open(str(temp_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.config.sample_rate)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())
            
            # Transcribe
            if self._backend == TranscriptionBackend.VOSK:
                result = self._transcribe_vosk(temp_path, self.config.language)
            else:
                result = self._transcribe_sr(temp_path, self.config.language)
            
            return result.text
            
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass  # Intentionally silent
    
    def get_supported_formats(self) -> list[str]:
        """Get list of supported audio formats."""
        return list(self._supported_formats)
    
    @property
    def backend_name(self) -> str:
        """Get current backend name."""
        return self._backend.name if self._backend else "NONE"


# Convenience functions

_transcriber: AudioFileTranscriber | None = None


def get_transcriber(config: TranscriptionConfig = None) -> AudioFileTranscriber:
    """Get or create global transcriber instance."""
    global _transcriber
    if _transcriber is None:
        _transcriber = AudioFileTranscriber(config)
    return _transcriber


def transcribe_file(
    file_path: str | Path,
    language: str = "en"
) -> str:
    """
    Transcribe an audio file to text.
    
    Args:
        file_path: Path to audio file
        language: Language code
    
    Returns:
        Transcribed text
    """
    transcriber = get_transcriber()
    result = transcriber.transcribe(file_path, language=language)
    return result.text


def transcribe_with_timestamps(
    file_path: str | Path,
    language: str = "en"
) -> list[tuple[float, float, str]]:
    """
    Transcribe with timestamps.
    
    Args:
        file_path: Path to audio file
        language: Language code
    
    Returns:
        List of (start, end, text) tuples
    """
    transcriber = get_transcriber()
    result = transcriber.transcribe(file_path, timestamps=True, language=language)
    return [(seg.start, seg.end, seg.text) for seg in result.segments]
