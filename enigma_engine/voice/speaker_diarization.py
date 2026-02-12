"""
================================================================================
Speaker Diarization - Identify different speakers in audio.
================================================================================

Multi-backend speaker identification:
- pyannote-audio: State-of-the-art neural diarization (requires HuggingFace token)
- resemblyzer: Speaker embeddings with clustering
- Simple: Energy/pause-based speaker change detection

USAGE:
    from enigma_engine.voice.speaker_diarization import SpeakerDiarizer
    
    diarizer = SpeakerDiarizer()
    
    # Diarize audio file
    result = diarizer.diarize("meeting.wav")
    for segment in result.segments:
        print(f"[{segment.start:.1f}s - {segment.end:.1f}s] Speaker {segment.speaker}: {segment.text}")
    
    # Get speaker statistics
    stats = result.get_speaker_stats()
    print(f"Total speakers: {stats['num_speakers']}")
    
    # Identify known speakers
    diarizer.add_speaker_profile("Alice", "alice_sample.wav")
    result = diarizer.diarize("meeting.wav", identify_speakers=True)
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
import wave
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class DiarizationBackend(Enum):
    """Available diarization backends."""
    
    PYANNOTE = auto()      # pyannote-audio (best quality)
    RESEMBLYZER = auto()   # resemblyzer embeddings + clustering
    SPEECHBRAIN = auto()   # SpeechBrain speaker recognition
    SIMPLE = auto()        # Energy/pause-based (fallback)


@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization."""
    
    # Backend selection
    preferred_backend: DiarizationBackend = DiarizationBackend.RESEMBLYZER
    
    # Speaker detection settings
    min_speakers: int = 1           # Minimum expected speakers
    max_speakers: int = 10          # Maximum expected speakers
    auto_num_speakers: bool = True  # Auto-detect number of speakers
    
    # Segmentation settings
    min_segment_duration: float = 0.5  # Minimum segment length in seconds
    min_silence_duration: float = 0.3  # Pause to consider speaker change
    overlap_threshold: float = 0.3     # Threshold for overlapping speech
    
    # Embedding settings
    embedding_model: str = "default"   # Embedding model for speaker similarity
    similarity_threshold: float = 0.7  # Threshold for same-speaker clustering
    
    # Audio processing
    sample_rate: int = 16000
    normalize_audio: bool = True
    
    # Huggingface token for pyannote
    hf_token: str | None = None
    
    # Profile storage
    profile_dir: Path | None = None


@dataclass
class SpeakerSegment:
    """A segment of audio from a single speaker."""
    
    start: float              # Start time in seconds
    end: float                # End time in seconds
    speaker: str              # Speaker ID or name
    text: str | None = None  # Transcribed text (if available)
    confidence: float = 1.0   # Confidence in speaker assignment
    embedding: np.ndarray | None = None  # Speaker embedding for this segment
    
    @property
    def duration(self) -> float:
        """Get segment duration."""
        return self.end - self.start


@dataclass
class DiarizationResult:
    """Result of speaker diarization."""
    
    segments: list[SpeakerSegment] = field(default_factory=list)
    audio_duration: float = 0.0
    num_speakers: int = 0
    backend_used: str = ""
    
    def get_speaker_stats(self) -> dict[str, Any]:
        """Get statistics about speakers."""
        speaker_durations: dict[str, float] = {}
        speaker_segments: dict[str, int] = {}
        
        for seg in self.segments:
            speaker_durations[seg.speaker] = speaker_durations.get(seg.speaker, 0) + seg.duration
            speaker_segments[seg.speaker] = speaker_segments.get(seg.speaker, 0) + 1
        
        return {
            "num_speakers": self.num_speakers,
            "audio_duration": self.audio_duration,
            "speakers": {
                speaker: {
                    "total_duration": duration,
                    "percentage": (duration / self.audio_duration * 100) if self.audio_duration > 0 else 0,
                    "num_segments": speaker_segments[speaker],
                    "avg_segment_duration": duration / speaker_segments[speaker] if speaker_segments[speaker] > 0 else 0
                }
                for speaker, duration in speaker_durations.items()
            }
        }
    
    def get_transcript_by_speaker(self) -> dict[str, list[str]]:
        """Get all text segments grouped by speaker."""
        result: dict[str, list[str]] = {}
        for seg in self.segments:
            if seg.text:
                if seg.speaker not in result:
                    result[seg.speaker] = []
                result[seg.speaker].append(seg.text)
        return result
    
    def to_srt(self) -> str:
        """Export to SRT subtitle format with speaker labels."""
        lines = []
        for i, seg in enumerate(self.segments, 1):
            start_h = int(seg.start // 3600)
            start_m = int((seg.start % 3600) // 60)
            start_s = seg.start % 60
            end_h = int(seg.end // 3600)
            end_m = int((seg.end % 3600) // 60)
            end_s = seg.end % 60
            
            text = seg.text or f"[{seg.speaker}]"
            lines.append(f"{i}")
            lines.append(f"{start_h:02d}:{start_m:02d}:{start_s:06.3f}".replace('.', ',') + 
                        f" --> {end_h:02d}:{end_m:02d}:{end_s:06.3f}".replace('.', ','))
            lines.append(f"[{seg.speaker}] {text}")
            lines.append("")
        
        return "\n".join(lines)


class SpeakerDiarizer:
    """Multi-backend speaker diarization system."""
    
    def __init__(self, config: DiarizationConfig | None = None):
        """Initialize diarizer with config."""
        self.config = config or DiarizationConfig()
        self._backend: Any | None = None
        self._backend_type: DiarizationBackend | None = None
        self._speaker_profiles: dict[str, np.ndarray] = {}
        
        # Set up profile directory
        if self.config.profile_dir:
            self.config.profile_dir.mkdir(parents=True, exist_ok=True)
            self._load_profiles()
    
    def _init_backend(self) -> DiarizationBackend:
        """Initialize the best available backend."""
        if self._backend is not None:
            return self._backend_type
        
        backends_to_try = [
            self.config.preferred_backend,
            DiarizationBackend.PYANNOTE,
            DiarizationBackend.RESEMBLYZER,
            DiarizationBackend.SPEECHBRAIN,
            DiarizationBackend.SIMPLE
        ]
        
        for backend in backends_to_try:
            try:
                if backend == DiarizationBackend.PYANNOTE:
                    if self._init_pyannote():
                        return DiarizationBackend.PYANNOTE
                        
                elif backend == DiarizationBackend.RESEMBLYZER:
                    if self._init_resemblyzer():
                        return DiarizationBackend.RESEMBLYZER
                        
                elif backend == DiarizationBackend.SPEECHBRAIN:
                    if self._init_speechbrain():
                        return DiarizationBackend.SPEECHBRAIN
                        
                elif backend == DiarizationBackend.SIMPLE:
                    self._backend = "simple"
                    self._backend_type = DiarizationBackend.SIMPLE
                    logger.info("Using simple energy-based diarization")
                    return DiarizationBackend.SIMPLE
                    
            except Exception as e:
                logger.debug(f"Backend {backend.name} failed: {e}")
                continue
        
        # Fallback to simple
        self._backend = "simple"
        self._backend_type = DiarizationBackend.SIMPLE
        return DiarizationBackend.SIMPLE
    
    def _init_pyannote(self) -> bool:
        """Initialize pyannote-audio backend."""
        try:
            from pyannote.audio import Pipeline
            
            token = self.config.hf_token or os.environ.get("HF_TOKEN")
            if not token:
                logger.info("pyannote requires HuggingFace token (HF_TOKEN)")
                return False
            
            self._backend = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )
            self._backend_type = DiarizationBackend.PYANNOTE
            logger.info("Using pyannote-audio for speaker diarization")
            return True
            
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"pyannote init failed: {e}")
            return False
    
    def _init_resemblyzer(self) -> bool:
        """Initialize resemblyzer backend."""
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            
            self._backend = VoiceEncoder()
            self._preprocess_wav = preprocess_wav
            self._backend_type = DiarizationBackend.RESEMBLYZER
            logger.info("Using resemblyzer for speaker diarization")
            return True
            
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"resemblyzer init failed: {e}")
            return False
    
    def _init_speechbrain(self) -> bool:
        """Initialize SpeechBrain backend."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            
            self._backend = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/speechbrain_spkrec"
            )
            self._backend_type = DiarizationBackend.SPEECHBRAIN
            logger.info("Using SpeechBrain for speaker diarization")
            return True
            
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"SpeechBrain init failed: {e}")
            return False
    
    def _load_audio(self, audio_path: str | Path) -> tuple[np.ndarray, int]:
        """Load audio file as numpy array."""
        audio_path = Path(audio_path)
        
        # Try soundfile first
        try:
            import soundfile as sf
            audio, sr = sf.read(str(audio_path))
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != self.config.sample_rate:
                # Resample
                import scipy.signal
                audio = scipy.signal.resample(audio, int(len(audio) * self.config.sample_rate / sr))
            return audio.astype(np.float32), self.config.sample_rate
        except ImportError:
            pass  # Intentionally silent
        
        # Try librosa
        try:
            import librosa
            audio, sr = librosa.load(str(audio_path), sr=self.config.sample_rate, mono=True)
            return audio.astype(np.float32), sr
        except ImportError:
            pass  # Intentionally silent
        
        # Fallback: WAV with standard library
        if audio_path.suffix.lower() == '.wav':
            with wave.open(str(audio_path), 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                sr = wf.getframerate()
                if sr != self.config.sample_rate:
                    # Simple resampling
                    ratio = self.config.sample_rate / sr
                    audio = np.interp(
                        np.linspace(0, len(audio) - 1, int(len(audio) * ratio)),
                        np.arange(len(audio)),
                        audio
                    )
                return audio, self.config.sample_rate
        
        raise ValueError(f"Cannot load audio format: {audio_path.suffix}")
    
    def diarize(
        self,
        audio_path: str | Path,
        num_speakers: int | None = None,
        identify_speakers: bool = False,
        transcribe: bool = False,
        progress_callback: Callable[[float], None] | None = None
    ) -> DiarizationResult:
        """
        Perform speaker diarization on an audio file.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Fixed number of speakers (None for auto-detect)
            identify_speakers: Try to match against known speaker profiles
            transcribe: Also transcribe each segment
            progress_callback: Optional callback for progress updates
            
        Returns:
            DiarizationResult with speaker segments
        """
        backend = self._init_backend()
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        audio, sr = self._load_audio(audio_path)
        audio_duration = len(audio) / sr
        
        if progress_callback:
            progress_callback(0.1)
        
        # Perform diarization based on backend
        if backend == DiarizationBackend.PYANNOTE:
            segments = self._diarize_pyannote(audio_path, num_speakers)
        elif backend == DiarizationBackend.RESEMBLYZER:
            segments = self._diarize_resemblyzer(audio, sr, num_speakers)
        elif backend == DiarizationBackend.SPEECHBRAIN:
            segments = self._diarize_speechbrain(audio, sr, num_speakers)
        else:
            segments = self._diarize_simple(audio, sr)
        
        if progress_callback:
            progress_callback(0.7)
        
        # Identify known speakers if requested
        if identify_speakers and self._speaker_profiles:
            segments = self._identify_speakers(segments, audio, sr)
        
        if progress_callback:
            progress_callback(0.85)
        
        # Transcribe segments if requested
        if transcribe:
            segments = self._transcribe_segments(segments, audio_path)
        
        if progress_callback:
            progress_callback(1.0)
        
        # Count unique speakers
        unique_speakers = len({seg.speaker for seg in segments})
        
        return DiarizationResult(
            segments=segments,
            audio_duration=audio_duration,
            num_speakers=unique_speakers,
            backend_used=backend.name
        )
    
    def _diarize_pyannote(
        self,
        audio_path: Path,
        num_speakers: int | None
    ) -> list[SpeakerSegment]:
        """Diarize using pyannote-audio."""
        params = {}
        if num_speakers:
            params["num_speakers"] = num_speakers
        elif not self.config.auto_num_speakers:
            params["min_speakers"] = self.config.min_speakers
            params["max_speakers"] = self.config.max_speakers
        
        diarization = self._backend(str(audio_path), **params)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker
            ))
        
        return segments
    
    def _diarize_resemblyzer(
        self,
        audio: np.ndarray,
        sr: int,
        num_speakers: int | None
    ) -> list[SpeakerSegment]:
        """Diarize using resemblyzer embeddings and clustering."""
        from sklearn.cluster import AgglomerativeClustering

        # Preprocess
        wav = self._preprocess_wav(audio, sr)
        if len(wav) == 0:
            return []
        
        # Get continuous embeddings
        encoder = self._backend
        
        # Segment audio into chunks for embedding
        chunk_duration = 1.5  # seconds
        overlap = 0.75
        chunk_samples = int(chunk_duration * sr)
        hop_samples = int(chunk_samples * (1 - overlap))
        
        embeddings = []
        timestamps = []
        
        for start_sample in range(0, len(wav) - chunk_samples, hop_samples):
            end_sample = start_sample + chunk_samples
            chunk = wav[start_sample:end_sample]
            
            # Get embedding for chunk
            embed = encoder.embed_utterance(chunk)
            embeddings.append(embed)
            timestamps.append((start_sample / sr, end_sample / sr))
        
        if not embeddings:
            return []
        
        embeddings_array = np.array(embeddings)
        
        # Determine number of speakers
        if num_speakers is None and self.config.auto_num_speakers:
            # Estimate using silhouette score
            from sklearn.metrics import silhouette_score
            best_n = 2
            best_score = -1
            for n in range(2, min(self.config.max_speakers + 1, len(embeddings))):
                try:
                    clustering = AgglomerativeClustering(n_clusters=n)
                    labels = clustering.fit_predict(embeddings_array)
                    score = silhouette_score(embeddings_array, labels)
                    if score > best_score:
                        best_score = score
                        best_n = n
                except (ValueError, RuntimeError):
                    break
            num_speakers = best_n
        else:
            num_speakers = num_speakers or 2
        
        # Cluster embeddings
        clustering = AgglomerativeClustering(
            n_clusters=min(num_speakers, len(embeddings)),
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings_array)
        
        # Convert to segments (merge consecutive same-speaker chunks)
        segments = []
        current_speaker = None
        current_start = None
        
        for i, (label, (start, end)) in enumerate(zip(labels, timestamps)):
            speaker = f"SPEAKER_{label:02d}"
            
            if speaker != current_speaker:
                if current_speaker is not None:
                    segments.append(SpeakerSegment(
                        start=current_start,
                        end=timestamps[i-1][1],
                        speaker=current_speaker,
                        embedding=embeddings_array[i-1]
                    ))
                current_speaker = speaker
                current_start = start
        
        # Add last segment
        if current_speaker is not None:
            segments.append(SpeakerSegment(
                start=current_start,
                end=timestamps[-1][1],
                speaker=current_speaker,
                embedding=embeddings_array[-1]
            ))
        
        return segments
    
    def _diarize_speechbrain(
        self,
        audio: np.ndarray,
        sr: int,
        num_speakers: int | None
    ) -> list[SpeakerSegment]:
        """Diarize using SpeechBrain speaker recognition."""
        import torch
        from sklearn.cluster import AgglomerativeClustering

        # Segment audio into chunks
        chunk_duration = 2.0  # seconds
        overlap = 0.5
        chunk_samples = int(chunk_duration * sr)
        hop_samples = int(chunk_samples * (1 - overlap))
        
        embeddings = []
        timestamps = []
        
        for start_sample in range(0, len(audio) - chunk_samples, hop_samples):
            end_sample = start_sample + chunk_samples
            chunk = audio[start_sample:end_sample]
            
            # Convert to tensor
            waveform = torch.tensor(chunk).unsqueeze(0)
            
            # Get embedding
            embed = self._backend.encode_batch(waveform)
            embeddings.append(embed.squeeze().cpu().numpy())
            timestamps.append((start_sample / sr, end_sample / sr))
        
        if not embeddings:
            return []
        
        embeddings_array = np.array(embeddings)
        num_speakers = num_speakers or 2
        
        # Cluster
        clustering = AgglomerativeClustering(
            n_clusters=min(num_speakers, len(embeddings)),
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings_array)
        
        # Convert to segments
        segments = []
        current_speaker = None
        current_start = None
        
        for i, (label, (start, end)) in enumerate(zip(labels, timestamps)):
            speaker = f"SPEAKER_{label:02d}"
            
            if speaker != current_speaker:
                if current_speaker is not None:
                    segments.append(SpeakerSegment(
                        start=current_start,
                        end=timestamps[i-1][1],
                        speaker=current_speaker
                    ))
                current_speaker = speaker
                current_start = start
        
        if current_speaker is not None:
            segments.append(SpeakerSegment(
                start=current_start,
                end=timestamps[-1][1],
                speaker=current_speaker
            ))
        
        return segments
    
    def _diarize_simple(
        self,
        audio: np.ndarray,
        sr: int
    ) -> list[SpeakerSegment]:
        """Simple energy/pause based diarization (fallback)."""
        # Compute RMS energy in windows
        window_size = int(0.025 * sr)  # 25ms windows
        hop_size = int(0.010 * sr)     # 10ms hop
        
        energies = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            energies.append(rms)
        
        energies = np.array(energies)
        
        # Find speech/silence threshold
        if len(energies) == 0:
            return []
        
        threshold = np.percentile(energies, 30)  # Bottom 30% is silence
        
        # Find speech regions
        is_speech = energies > threshold
        
        # Smooth with median filter
        try:
            from scipy.ndimage import median_filter
            is_speech = median_filter(is_speech.astype(float), size=15) > 0.5
        except ImportError:
            pass  # Intentionally silent
        
        # Find boundaries
        segments = []
        in_speech = False
        speech_start = 0
        
        time_per_frame = hop_size / sr
        min_silence_frames = int(self.config.min_silence_duration / time_per_frame)
        
        silence_count = 0
        speaker_idx = 0
        
        for i, speech in enumerate(is_speech):
            time = i * time_per_frame
            
            if speech:
                if not in_speech:
                    speech_start = time
                    in_speech = True
                    # Consider speaker change after long silence
                    if silence_count > min_silence_frames * 3:
                        speaker_idx = (speaker_idx + 1) % 2
                silence_count = 0
            else:
                silence_count += 1
                if in_speech and silence_count > min_silence_frames:
                    # End of speech segment
                    if time - speech_start >= self.config.min_segment_duration:
                        segments.append(SpeakerSegment(
                            start=speech_start,
                            end=time - (silence_count * time_per_frame),
                            speaker=f"SPEAKER_{speaker_idx:02d}"
                        ))
                    in_speech = False
        
        # End final segment
        if in_speech:
            end_time = len(is_speech) * time_per_frame
            if end_time - speech_start >= self.config.min_segment_duration:
                segments.append(SpeakerSegment(
                    start=speech_start,
                    end=end_time,
                    speaker=f"SPEAKER_{speaker_idx:02d}"
                ))
        
        return segments
    
    def _identify_speakers(
        self,
        segments: list[SpeakerSegment],
        audio: np.ndarray,
        sr: int
    ) -> list[SpeakerSegment]:
        """Try to match segments against known speaker profiles."""
        if not self._speaker_profiles:
            return segments
        
        # Get embeddings for segments if not already present
        embeddings_needed = any(seg.embedding is None for seg in segments)
        
        if embeddings_needed and self._backend_type == DiarizationBackend.RESEMBLYZER:
            for seg in segments:
                if seg.embedding is None:
                    start_sample = int(seg.start * sr)
                    end_sample = int(seg.end * sr)
                    chunk = audio[start_sample:end_sample]
                    if len(chunk) > 0:
                        wav = self._preprocess_wav(chunk, sr)
                        if len(wav) > 0:
                            seg.embedding = self._backend.embed_utterance(wav)
        
        # Match each segment to known speakers
        identified_segments = []
        for seg in segments:
            if seg.embedding is not None:
                best_match = None
                best_similarity = self.config.similarity_threshold
                
                for name, profile_embed in self._speaker_profiles.items():
                    # Cosine similarity
                    similarity = np.dot(seg.embedding, profile_embed) / (
                        np.linalg.norm(seg.embedding) * np.linalg.norm(profile_embed)
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name
                
                if best_match:
                    seg.speaker = best_match
                    seg.confidence = best_similarity
            
            identified_segments.append(seg)
        
        return identified_segments
    
    def _transcribe_segments(
        self,
        segments: list[SpeakerSegment],
        audio_path: Path
    ) -> list[SpeakerSegment]:
        """Transcribe each segment using whisper or other STT."""
        try:
            from enigma_engine.voice.audio_file_input import AudioFileTranscriber
            transcriber = AudioFileTranscriber()
        except ImportError:
            logger.warning("AudioFileTranscriber not available, skipping transcription")
            return segments
        
        # Create temp files for each segment and transcribe
        audio, sr = self._load_audio(audio_path)
        
        for seg in segments:
            try:
                start_sample = int(seg.start * sr)
                end_sample = int(seg.end * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # Save temp file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                
                try:
                    import soundfile as sf
                    sf.write(temp_path, segment_audio, sr)
                except ImportError:
                    import wave
                    with wave.open(temp_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sr)
                        wf.writeframes((segment_audio * 32767).astype(np.int16).tobytes())
                
                # Transcribe
                result = transcriber.transcribe(temp_path)
                seg.text = result.text
                
                # Cleanup
                os.unlink(temp_path)
                
            except Exception as e:
                logger.debug(f"Failed to transcribe segment: {e}")
                seg.text = ""
        
        return segments
    
    def add_speaker_profile(
        self,
        name: str,
        audio_path: str | Path,
        append: bool = False
    ) -> bool:
        """
        Add a speaker profile from a sample audio file.
        
        Args:
            name: Name for the speaker
            audio_path: Path to audio sample of this speaker
            append: If True, average with existing profile; otherwise replace
            
        Returns:
            True if profile was added successfully
        """
        self._init_backend()
        
        if self._backend_type not in (DiarizationBackend.RESEMBLYZER, DiarizationBackend.SPEECHBRAIN):
            logger.warning("Speaker profiles only supported with resemblyzer or speechbrain backends")
            return False
        
        try:
            audio, sr = self._load_audio(audio_path)
            
            if self._backend_type == DiarizationBackend.RESEMBLYZER:
                wav = self._preprocess_wav(audio, sr)
                embedding = self._backend.embed_utterance(wav)
            else:
                # SpeechBrain
                import torch
                waveform = torch.tensor(audio).unsqueeze(0)
                embedding = self._backend.encode_batch(waveform).squeeze().cpu().numpy()
            
            if append and name in self._speaker_profiles:
                # Average embeddings
                old_embed = self._speaker_profiles[name]
                embedding = (old_embed + embedding) / 2
                embedding = embedding / np.linalg.norm(embedding)  # Renormalize
            
            self._speaker_profiles[name] = embedding
            
            # Save if profile_dir is set
            if self.config.profile_dir:
                self._save_profiles()
            
            logger.info(f"Added speaker profile: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add speaker profile: {e}")
            return False
    
    def remove_speaker_profile(self, name: str) -> bool:
        """Remove a speaker profile."""
        if name in self._speaker_profiles:
            del self._speaker_profiles[name]
            if self.config.profile_dir:
                self._save_profiles()
            return True
        return False
    
    def list_speaker_profiles(self) -> list[str]:
        """List all known speaker profiles."""
        return list(self._speaker_profiles.keys())
    
    def _save_profiles(self) -> None:
        """Save profiles to disk."""
        if not self.config.profile_dir:
            return
        
        profile_path = self.config.profile_dir / "speaker_profiles.pkl"
        try:
            with open(profile_path, 'wb') as f:
                pickle.dump(self._speaker_profiles, f)
        except Exception as e:
            logger.error(f"Failed to save speaker profiles: {e}")
    
    def _load_profiles(self) -> None:
        """Load profiles from disk."""
        if not self.config.profile_dir:
            return
        
        profile_path = self.config.profile_dir / "speaker_profiles.pkl"
        if profile_path.exists():
            try:
                with open(profile_path, 'rb') as f:
                    self._speaker_profiles = pickle.load(f)
                logger.info(f"Loaded {len(self._speaker_profiles)} speaker profiles")
            except Exception as e:
                logger.error(f"Failed to load speaker profiles: {e}")


def get_diarizer(config: DiarizationConfig | None = None) -> SpeakerDiarizer:
    """Get or create a singleton diarizer instance."""
    global _diarizer_instance
    if '_diarizer_instance' not in globals() or _diarizer_instance is None:
        _diarizer_instance = SpeakerDiarizer(config)
    return _diarizer_instance


# Convenience functions
def diarize_audio(
    audio_path: str | Path,
    num_speakers: int | None = None,
    transcribe: bool = False
) -> DiarizationResult:
    """Quick diarization of an audio file."""
    diarizer = get_diarizer()
    return diarizer.diarize(audio_path, num_speakers=num_speakers, transcribe=transcribe)


def identify_speaker(
    audio_path: str | Path,
    profiles_dir: str | Path
) -> str | None:
    """Identify who is speaking in an audio file."""
    config = DiarizationConfig(profile_dir=Path(profiles_dir))
    diarizer = SpeakerDiarizer(config)
    result = diarizer.diarize(audio_path, num_speakers=1, identify_speakers=True)
    
    if result.segments:
        return result.segments[0].speaker
    return None
