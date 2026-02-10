"""
Multi-Speaker TTS for Enigma AI Engine

Generate speech with multiple voices.

Features:
- Multiple TTS backends
- Speaker embedding
- Voice cloning
- Emotion control
- Batch generation

Usage:
    from enigma_engine.voice.multi_speaker import MultiSpeakerTTS
    
    tts = MultiSpeakerTTS()
    
    # Register speakers
    tts.register_speaker("alice", voice_samples=["alice_1.wav"])
    tts.register_speaker("bob", preset="male_deep")
    
    # Generate speech
    audio = tts.speak("Hello world", speaker="alice")
    
    # Conversation
    conversation = [
        {"speaker": "alice", "text": "Hi Bob!"},
        {"speaker": "bob", "text": "Hello Alice!"}
    ]
    audio = tts.speak_conversation(conversation)
"""

import hashlib
import json
import logging
import os
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class TTSBackend(Enum):
    """TTS backend types."""
    PYTTSX3 = "pyttsx3"
    COQUI = "coqui"  # TTS library
    BARK = "bark"  # Suno Bark
    ELEVENLABS = "elevenlabs"
    AZURE = "azure"
    GOOGLE = "google"
    OPENAI = "openai"


class EmotionType(Enum):
    """Speech emotion types."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    CALM = "calm"
    EXCITED = "excited"


@dataclass
class SpeakerProfile:
    """Speaker voice profile."""
    id: str
    name: str
    voice_embedding: Optional[np.ndarray] = None  # Speaker embedding vector
    voice_samples: List[str] = field(default_factory=list)  # Reference audio files
    preset: Optional[str] = None  # Built-in voice preset
    backend: TTSBackend = TTSBackend.PYTTSX3
    
    # Voice characteristics
    pitch: float = 1.0  # 0.5-2.0
    speed: float = 1.0  # 0.5-2.0
    volume: float = 1.0  # 0.0-1.0
    
    # Emotion/style
    default_emotion: EmotionType = EmotionType.NEUTRAL
    
    # Backend-specific
    backend_voice_id: Optional[str] = None  # ID for API services
    backend_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeechSegment:
    """A segment of speech."""
    speaker: str
    text: str
    emotion: EmotionType = EmotionType.NEUTRAL
    pause_before: float = 0.0  # Seconds
    pause_after: float = 0.3  # Seconds


@dataclass
class GeneratedAudio:
    """Generated audio data."""
    audio_data: np.ndarray  # Raw audio samples
    sample_rate: int = 22050
    speaker: str = ""
    text: str = ""
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


class TTSEngine(ABC):
    """Abstract TTS engine."""
    
    @abstractmethod
    def synthesize(
        self,
        text: str,
        speaker: SpeakerProfile,
        emotion: EmotionType = EmotionType.NEUTRAL
    ) -> GeneratedAudio:
        """Generate speech from text."""
    
    @abstractmethod
    def get_available_voices(self) -> List[str]:
        """Get available voice presets."""


class Pyttsx3Engine(TTSEngine):
    """pyttsx3-based TTS engine."""
    
    def __init__(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self._voices = self.engine.getProperty('voices')
        except ImportError:
            self.engine = None
            self._voices = []
            logger.warning("pyttsx3 not installed")
    
    def synthesize(
        self,
        text: str,
        speaker: SpeakerProfile,
        emotion: EmotionType = EmotionType.NEUTRAL
    ) -> GeneratedAudio:
        if not self.engine:
            # Return silence
            return GeneratedAudio(
                audio_data=np.zeros(22050),
                sample_rate=22050,
                speaker=speaker.id,
                text=text,
                duration=1.0
            )
        
        import tempfile
        
        # Set voice properties
        if speaker.backend_voice_id:
            self.engine.setProperty('voice', speaker.backend_voice_id)
        elif speaker.preset and self._voices:
            # Match preset to available voice
            for voice in self._voices:
                if speaker.preset.lower() in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
        
        self.engine.setProperty('rate', int(150 * speaker.speed))
        self.engine.setProperty('volume', speaker.volume)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        self.engine.save_to_file(text, temp_path)
        self.engine.runAndWait()
        
        # Load audio
        audio_data, sample_rate = self._load_wav(temp_path)
        
        # Cleanup
        os.unlink(temp_path)
        
        return GeneratedAudio(
            audio_data=audio_data,
            sample_rate=sample_rate,
            speaker=speaker.id,
            text=text,
            duration=len(audio_data) / sample_rate
        )
    
    def get_available_voices(self) -> List[str]:
        if not self._voices:
            return []
        return [v.name for v in self._voices]
    
    def _load_wav(self, path: str) -> Tuple[np.ndarray, int]:
        """Load WAV file."""
        with wave.open(path, 'r') as wav:
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            audio_bytes = wav.readframes(n_frames)
            
            # Convert to float
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            
            return audio_float, sample_rate


class CoquiEngine(TTSEngine):
    """Coqui TTS engine - supports voice cloning."""
    
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        self.model_name = model_name
        self.tts = None
        self._load_model()
    
    def _load_model(self):
        try:
            from TTS.api import TTS as CoquiTTS
            self.tts = CoquiTTS(model_name=self.model_name)
        except ImportError:
            logger.warning("Coqui TTS not installed")
    
    def synthesize(
        self,
        text: str,
        speaker: SpeakerProfile,
        emotion: EmotionType = EmotionType.NEUTRAL
    ) -> GeneratedAudio:
        if not self.tts:
            return GeneratedAudio(
                audio_data=np.zeros(22050),
                sample_rate=22050,
                speaker=speaker.id,
                text=text,
                duration=1.0
            )
        
        import tempfile
        
        # Generate
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        # Voice cloning if samples provided
        if speaker.voice_samples:
            self.tts.tts_to_file(
                text=text,
                speaker_wav=speaker.voice_samples[0],
                file_path=temp_path
            )
        else:
            self.tts.tts_to_file(
                text=text,
                file_path=temp_path
            )
        
        # Load result
        audio_data, sample_rate = self._load_wav(temp_path)
        os.unlink(temp_path)
        
        return GeneratedAudio(
            audio_data=audio_data,
            sample_rate=sample_rate,
            speaker=speaker.id,
            text=text,
            duration=len(audio_data) / sample_rate
        )
    
    def get_available_voices(self) -> List[str]:
        if not self.tts:
            return []
        try:
            return self.tts.speakers or []
        except Exception:
            return []
    
    def _load_wav(self, path: str) -> Tuple[np.ndarray, int]:
        """Load WAV file."""
        with wave.open(path, 'r') as wav:
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            audio_bytes = wav.readframes(n_frames)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            return audio_float, sample_rate


class MultiSpeakerTTS:
    """Multi-speaker text-to-speech system."""
    
    def __init__(
        self,
        default_backend: TTSBackend = TTSBackend.PYTTSX3,
        cache_dir: str = "cache/tts"
    ):
        """
        Initialize multi-speaker TTS.
        
        Args:
            default_backend: Default TTS backend
            cache_dir: Directory for caching generated audio
        """
        self.default_backend = default_backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Speaker registry
        self._speakers: Dict[str, SpeakerProfile] = {}
        
        # TTS engines
        self._engines: Dict[TTSBackend, TTSEngine] = {}
        
        # Initialize default engine
        self._init_engine(default_backend)
        
        # Built-in presets
        self._init_presets()
        
        logger.info(f"MultiSpeakerTTS initialized with {default_backend.value}")
    
    def _init_engine(self, backend: TTSBackend):
        """Initialize TTS engine."""
        if backend in self._engines:
            return
        
        if backend == TTSBackend.PYTTSX3:
            self._engines[backend] = Pyttsx3Engine()
        elif backend == TTSBackend.COQUI:
            self._engines[backend] = CoquiEngine()
        else:
            logger.warning(f"Backend {backend.value} not implemented")
    
    def _init_presets(self):
        """Initialize built-in voice presets."""
        presets = [
            SpeakerProfile(id="default", name="Default", preset="default"),
            SpeakerProfile(id="male_deep", name="Male Deep", pitch=0.8, speed=0.9),
            SpeakerProfile(id="male_normal", name="Male Normal", pitch=1.0),
            SpeakerProfile(id="female_high", name="Female High", pitch=1.3, speed=1.1),
            SpeakerProfile(id="female_normal", name="Female Normal", pitch=1.1),
            SpeakerProfile(id="child", name="Child", pitch=1.5, speed=1.2),
            SpeakerProfile(id="elder_male", name="Elder Male", pitch=0.85, speed=0.85),
            SpeakerProfile(id="elder_female", name="Elder Female", pitch=1.0, speed=0.85),
        ]
        
        for preset in presets:
            self._speakers[preset.id] = preset
    
    def register_speaker(
        self,
        speaker_id: str,
        name: Optional[str] = None,
        voice_samples: Optional[List[str]] = None,
        preset: Optional[str] = None,
        backend: Optional[TTSBackend] = None,
        **kwargs
    ) -> SpeakerProfile:
        """
        Register a new speaker.
        
        Args:
            speaker_id: Unique speaker ID
            name: Display name
            voice_samples: Audio files for voice cloning
            preset: Built-in preset name
            backend: TTS backend to use
            **kwargs: Additional speaker profile attributes
            
        Returns:
            Created speaker profile
        """
        # Copy from preset if specified
        if preset and preset in self._speakers:
            base = self._speakers[preset]
            profile = SpeakerProfile(
                id=speaker_id,
                name=name or speaker_id,
                pitch=base.pitch,
                speed=base.speed,
                volume=base.volume,
                backend=backend or base.backend,
                preset=preset
            )
        else:
            profile = SpeakerProfile(
                id=speaker_id,
                name=name or speaker_id,
                backend=backend or self.default_backend
            )
        
        # Set voice samples
        if voice_samples:
            profile.voice_samples = voice_samples
        
        # Apply additional kwargs
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        # Initialize engine if needed
        self._init_engine(profile.backend)
        
        self._speakers[speaker_id] = profile
        
        logger.info(f"Registered speaker: {speaker_id}")
        return profile
    
    def get_speaker(self, speaker_id: str) -> Optional[SpeakerProfile]:
        """Get speaker profile."""
        return self._speakers.get(speaker_id)
    
    def list_speakers(self) -> List[str]:
        """List registered speakers."""
        return list(self._speakers.keys())
    
    def speak(
        self,
        text: str,
        speaker: str = "default",
        emotion: EmotionType = EmotionType.NEUTRAL,
        output_path: Optional[str] = None,
        use_cache: bool = True
    ) -> GeneratedAudio:
        """
        Generate speech for text.
        
        Args:
            text: Text to speak
            speaker: Speaker ID
            emotion: Emotion style
            output_path: Save to file
            use_cache: Use cached audio if available
            
        Returns:
            Generated audio
        """
        # Get speaker profile
        profile = self._speakers.get(speaker)
        if not profile:
            logger.warning(f"Speaker {speaker} not found, using default")
            profile = self._speakers.get("default", SpeakerProfile(id="default", name="Default"))
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(text, speaker, emotion)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached
        
        # Get engine
        engine = self._engines.get(profile.backend)
        if not engine:
            self._init_engine(profile.backend)
            engine = self._engines.get(profile.backend)
        
        if not engine:
            raise ValueError(f"No engine for backend {profile.backend.value}")
        
        # Generate
        audio = engine.synthesize(text, profile, emotion)
        
        # Cache
        if use_cache:
            self._save_to_cache(cache_key, audio)
        
        # Save to file
        if output_path:
            audio.save(output_path)
        
        return audio
    
    def speak_conversation(
        self,
        segments: List[Union[SpeechSegment, Dict]],
        output_path: Optional[str] = None
    ) -> GeneratedAudio:
        """
        Generate speech for a conversation.
        
        Args:
            segments: List of speech segments
            output_path: Save combined audio to file
            
        Returns:
            Combined audio
        """
        audio_parts = []
        sample_rate = 22050
        
        for segment in segments:
            # Convert dict to SpeechSegment
            if isinstance(segment, dict):
                segment = SpeechSegment(
                    speaker=segment.get("speaker", "default"),
                    text=segment.get("text", ""),
                    emotion=segment.get("emotion", EmotionType.NEUTRAL),
                    pause_before=segment.get("pause_before", 0.0),
                    pause_after=segment.get("pause_after", 0.3)
                )
            
            # Add pause before
            if segment.pause_before > 0:
                silence = np.zeros(int(segment.pause_before * sample_rate))
                audio_parts.append(silence)
            
            # Generate speech
            audio = self.speak(
                segment.text,
                speaker=segment.speaker,
                emotion=segment.emotion
            )
            
            # Resample if needed
            if audio.sample_rate != sample_rate:
                audio.audio_data = self._resample(audio.audio_data, audio.sample_rate, sample_rate)
            
            audio_parts.append(audio.audio_data)
            
            # Add pause after
            if segment.pause_after > 0:
                silence = np.zeros(int(segment.pause_after * sample_rate))
                audio_parts.append(silence)
        
        # Combine
        combined = np.concatenate(audio_parts)
        
        result = GeneratedAudio(
            audio_data=combined,
            sample_rate=sample_rate,
            speaker="conversation",
            text=f"{len(segments)} segments",
            duration=len(combined) / sample_rate
        )
        
        if output_path:
            result.save(output_path)
        
        return result
    
    def clone_voice(
        self,
        speaker_id: str,
        reference_audio: List[str],
        name: Optional[str] = None,
        backend: TTSBackend = TTSBackend.COQUI
    ) -> SpeakerProfile:
        """
        Clone a voice from reference audio.
        
        Args:
            speaker_id: ID for new speaker
            reference_audio: List of reference audio files
            name: Display name
            backend: Backend that supports cloning
            
        Returns:
            Created speaker profile
        """
        return self.register_speaker(
            speaker_id=speaker_id,
            name=name,
            voice_samples=reference_audio,
            backend=backend
        )
    
    def get_available_voices(self, backend: Optional[TTSBackend] = None) -> List[str]:
        """Get available voices for backend."""
        if backend:
            engine = self._engines.get(backend)
            if engine:
                return engine.get_available_voices()
            return []
        
        voices = []
        for engine in self._engines.values():
            voices.extend(engine.get_available_voices())
        return voices
    
    def _get_cache_key(self, text: str, speaker: str, emotion: EmotionType) -> str:
        """Generate cache key."""
        data = f"{text}|{speaker}|{emotion.value}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _save_to_cache(self, key: str, audio: GeneratedAudio):
        """Save audio to cache."""
        cache_path = self.cache_dir / f"{key}.wav"
        audio.save(str(cache_path))
        
        # Save metadata
        meta_path = self.cache_dir / f"{key}.json"
        meta = {
            "speaker": audio.speaker,
            "text": audio.text,
            "sample_rate": audio.sample_rate,
            "duration": audio.duration
        }
        meta_path.write_text(json.dumps(meta))
    
    def _load_from_cache(self, key: str) -> Optional[GeneratedAudio]:
        """Load audio from cache."""
        cache_path = self.cache_dir / f"{key}.wav"
        meta_path = self.cache_dir / f"{key}.json"
        
        if not cache_path.exists() or not meta_path.exists():
            return None
        
        # Load metadata
        meta = json.loads(meta_path.read_text())
        
        # Load audio
        with wave.open(str(cache_path), 'r') as wav:
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            audio_bytes = wav.readframes(n_frames)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
        
        return GeneratedAudio(
            audio_data=audio_float,
            sample_rate=sample_rate,
            speaker=meta.get("speaker", ""),
            text=meta.get("text", ""),
            duration=meta.get("duration", 0.0)
        )
    
    def _resample(self, audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Simple resampling."""
        if src_rate == dst_rate:
            return audio
        
        ratio = dst_rate / src_rate
        new_length = int(len(audio) * ratio)
        
        # Linear interpolation
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)


# Global instance
_multi_speaker_tts: Optional[MultiSpeakerTTS] = None


def get_multi_speaker_tts() -> MultiSpeakerTTS:
    """Get or create global MultiSpeakerTTS instance."""
    global _multi_speaker_tts
    if _multi_speaker_tts is None:
        _multi_speaker_tts = MultiSpeakerTTS()
    return _multi_speaker_tts
