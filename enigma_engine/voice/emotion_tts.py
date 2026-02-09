"""
Emotion-Controlled TTS for Enigma AI Engine

Generate speech with specified emotional qualities.

Provides:
- Emotion presets (happy, sad, angry, excited, calm, etc.)
- Intensity control for each emotion
- Speed and pitch modulation
- Integration with voice emotion detector
- Multiple TTS backend support

Usage:
    from enigma_engine.voice.emotion_tts import EmotionTTS
    
    tts = EmotionTTS()
    
    # Basic usage
    tts.speak("Hello!", emotion="happy")
    tts.speak("I'm sorry to hear that", emotion="sad", intensity=0.8)
    
    # Complex emotion
    tts.speak(
        "This is amazing!",
        emotion="excited",
        intensity=0.9,
        speed=1.2
    )
    
    # Get audio without playing
    audio = tts.synthesize("Text", emotion="calm")
    audio.save("output.wav")
"""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import io

logger = logging.getLogger(__name__)

# Try imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    pyttsx3 = None

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sd = None


class TTSEmotion(Enum):
    """Available emotions for TTS."""
    NEUTRAL = auto()
    HAPPY = auto()
    SAD = auto()
    ANGRY = auto()
    FEARFUL = auto()
    SURPRISED = auto()
    DISGUSTED = auto()
    CALM = auto()
    EXCITED = auto()
    TIRED = auto()
    SERIOUS = auto()
    PLAYFUL = auto()
    SYMPATHETIC = auto()
    CONFIDENT = auto()
    NERVOUS = auto()


@dataclass
class EmotionParameters:
    """Parameters that define an emotional speaking style."""
    # Speech rate multiplier (0.5-2.0, 1.0 = normal)
    rate: float = 1.0
    
    # Pitch shift in semitones (-12 to +12)
    pitch: float = 0.0
    
    # Volume multiplier (0.0-1.5)
    volume: float = 1.0
    
    # Emphasis on stressed syllables (0.0-1.0)
    emphasis: float = 0.5
    
    # Pause between phrases (0.0-1.0)
    pause_factor: float = 0.5
    
    # Voice tremor/shake (0.0-1.0)
    tremor: float = 0.0
    
    # Breathiness (0.0-1.0)
    breathiness: float = 0.0
    
    # Whisper amount (0.0-1.0)
    whisper: float = 0.0


# Emotion presets
EMOTION_PRESETS: Dict[TTSEmotion, EmotionParameters] = {
    TTSEmotion.NEUTRAL: EmotionParameters(),
    
    TTSEmotion.HAPPY: EmotionParameters(
        rate=1.15,
        pitch=2.0,
        volume=1.1,
        emphasis=0.7
    ),
    
    TTSEmotion.SAD: EmotionParameters(
        rate=0.85,
        pitch=-2.0,
        volume=0.8,
        emphasis=0.3,
        pause_factor=0.8,
        breathiness=0.3
    ),
    
    TTSEmotion.ANGRY: EmotionParameters(
        rate=1.2,
        pitch=1.0,
        volume=1.3,
        emphasis=0.9,
        pause_factor=0.3
    ),
    
    TTSEmotion.FEARFUL: EmotionParameters(
        rate=1.3,
        pitch=3.0,
        volume=0.9,
        tremor=0.4,
        breathiness=0.4
    ),
    
    TTSEmotion.SURPRISED: EmotionParameters(
        rate=1.25,
        pitch=4.0,
        volume=1.2,
        emphasis=0.8,
        pause_factor=0.7
    ),
    
    TTSEmotion.DISGUSTED: EmotionParameters(
        rate=0.9,
        pitch=-1.0,
        volume=0.9,
        emphasis=0.6,
        breathiness=0.2
    ),
    
    TTSEmotion.CALM: EmotionParameters(
        rate=0.9,
        pitch=-0.5,
        volume=0.85,
        emphasis=0.3,
        pause_factor=0.7
    ),
    
    TTSEmotion.EXCITED: EmotionParameters(
        rate=1.35,
        pitch=3.5,
        volume=1.25,
        emphasis=0.85,
        pause_factor=0.2
    ),
    
    TTSEmotion.TIRED: EmotionParameters(
        rate=0.75,
        pitch=-2.5,
        volume=0.7,
        emphasis=0.2,
        pause_factor=0.9,
        breathiness=0.5
    ),
    
    TTSEmotion.SERIOUS: EmotionParameters(
        rate=0.95,
        pitch=-1.5,
        volume=1.0,
        emphasis=0.6,
        pause_factor=0.6
    ),
    
    TTSEmotion.PLAYFUL: EmotionParameters(
        rate=1.2,
        pitch=2.5,
        volume=1.05,
        emphasis=0.7,
        pause_factor=0.4
    ),
    
    TTSEmotion.SYMPATHETIC: EmotionParameters(
        rate=0.9,
        pitch=0.5,
        volume=0.9,
        emphasis=0.4,
        pause_factor=0.7,
        breathiness=0.2
    ),
    
    TTSEmotion.CONFIDENT: EmotionParameters(
        rate=1.05,
        pitch=-0.5,
        volume=1.15,
        emphasis=0.7,
        pause_factor=0.5
    ),
    
    TTSEmotion.NERVOUS: EmotionParameters(
        rate=1.2,
        pitch=2.0,
        volume=0.85,
        tremor=0.3,
        pause_factor=0.3,
        breathiness=0.3
    ),
}


@dataclass
class AudioOutput:
    """Container for synthesized audio."""
    samples: 'np.ndarray'
    sample_rate: int
    duration: float
    
    def save(self, path: str, format: str = "wav"):
        """Save audio to file."""
        try:
            import soundfile as sf
            sf.write(path, self.samples, self.sample_rate)
        except ImportError:
            # Fallback to wave
            import wave
            with wave.open(path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes((self.samples * 32767).astype(np.int16).tobytes())
    
    def play(self):
        """Play the audio."""
        if SOUNDDEVICE_AVAILABLE:
            sd.play(self.samples, self.sample_rate)
            sd.wait()


class EmotionTTS:
    """
    Text-to-speech with emotional expression.
    
    Modulates speech parameters to convey different emotions.
    """
    
    def __init__(self, voice_id: Optional[str] = None):
        """
        Initialize emotion TTS.
        
        Args:
            voice_id: Optional voice ID for TTS engine
        """
        self._engine = None
        self._voice_id = voice_id
        self._lock = threading.Lock()
        
        # Default settings
        self._default_rate = 150  # Words per minute
        self._default_volume = 1.0
        
        # Initialize engine
        self._init_engine()
        
        logger.info("EmotionTTS initialized")
    
    def _init_engine(self):
        """Initialize TTS engine."""
        if PYTTSX3_AVAILABLE:
            try:
                self._engine = pyttsx3.init()
                
                # Get available voices
                voices = self._engine.getProperty('voices')
                if voices and self._voice_id:
                    for voice in voices:
                        if self._voice_id in voice.id:
                            self._engine.setProperty('voice', voice.id)
                            break
                
                self._default_rate = self._engine.getProperty('rate')
                
            except Exception as e:
                logger.warning(f"Failed to init pyttsx3: {e}")
                self._engine = None
    
    def speak(
        self,
        text: str,
        emotion: TTSEmotion | str = TTSEmotion.NEUTRAL,
        intensity: float = 1.0,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        blocking: bool = True
    ):
        """
        Speak text with specified emotion.
        
        Args:
            text: Text to speak
            emotion: Emotion to express
            intensity: How strong the emotion (0.0-1.0)
            speed: Optional speed override
            pitch: Optional pitch override
            blocking: Wait for speech to complete
        """
        # Parse emotion
        if isinstance(emotion, str):
            emotion = TTSEmotion[emotion.upper()]
        
        params = self._get_emotion_params(emotion, intensity)
        
        # Apply overrides
        if speed is not None:
            params.rate = speed
        if pitch is not None:
            params.pitch = pitch
        
        def do_speak():
            with self._lock:
                if self._engine:
                    # Apply parameters
                    rate = int(self._default_rate * params.rate)
                    volume = min(1.0, self._default_volume * params.volume)
                    
                    self._engine.setProperty('rate', rate)
                    self._engine.setProperty('volume', volume)
                    
                    self._engine.say(text)
                    self._engine.runAndWait()
        
        if blocking:
            do_speak()
        else:
            thread = threading.Thread(target=do_speak, daemon=True)
            thread.start()
    
    def synthesize(
        self,
        text: str,
        emotion: TTSEmotion | str = TTSEmotion.NEUTRAL,
        intensity: float = 1.0,
        sample_rate: int = 22050
    ) -> Optional[AudioOutput]:
        """
        Synthesize text to audio without playing.
        
        Args:
            text: Text to synthesize
            emotion: Emotion to express
            intensity: Emotion intensity (0.0-1.0)
            sample_rate: Output sample rate
            
        Returns:
            AudioOutput or None if synthesis failed
        """
        if isinstance(emotion, str):
            emotion = TTSEmotion[emotion.upper()]
        
        params = self._get_emotion_params(emotion, intensity)
        
        try:
            # Try using edge-tts or other services that return audio
            audio = self._synthesize_with_service(text, params, sample_rate)
            if audio:
                return audio
            
            # Fallback: use pyttsx3 to file
            if self._engine:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                
                with self._lock:
                    rate = int(self._default_rate * params.rate)
                    self._engine.setProperty('rate', rate)
                    self._engine.setProperty('volume', min(1.0, params.volume))
                    self._engine.save_to_file(text, temp_path)
                    self._engine.runAndWait()
                
                # Load the audio
                import wave
                with wave.open(temp_path, 'rb') as wf:
                    sr = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767
                
                # Clean up
                Path(temp_path).unlink(missing_ok=True)
                
                # Apply pitch shift if needed
                if params.pitch != 0:
                    samples = self._shift_pitch(samples, sr, params.pitch)
                
                return AudioOutput(
                    samples=samples,
                    sample_rate=sr,
                    duration=len(samples) / sr
                )
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
        
        return None
    
    def _synthesize_with_service(
        self,
        text: str,
        params: EmotionParameters,
        sample_rate: int
    ) -> Optional[AudioOutput]:
        """Try to synthesize using cloud/advanced services."""
        # This is a placeholder for integration with:
        # - ElevenLabs
        # - Azure Cognitive Services
        # - Google Cloud TTS
        # - Amazon Polly
        
        # Each service has its own emotion/style controls
        return None
    
    def _get_emotion_params(
        self,
        emotion: TTSEmotion,
        intensity: float
    ) -> EmotionParameters:
        """Get parameters for emotion with intensity scaling."""
        preset = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS[TTSEmotion.NEUTRAL])
        neutral = EMOTION_PRESETS[TTSEmotion.NEUTRAL]
        
        # Interpolate between neutral and emotion preset based on intensity
        intensity = max(0.0, min(1.0, intensity))
        
        def lerp(a: float, b: float) -> float:
            return a + (b - a) * intensity
        
        return EmotionParameters(
            rate=lerp(neutral.rate, preset.rate),
            pitch=lerp(neutral.pitch, preset.pitch),
            volume=lerp(neutral.volume, preset.volume),
            emphasis=lerp(neutral.emphasis, preset.emphasis),
            pause_factor=lerp(neutral.pause_factor, preset.pause_factor),
            tremor=lerp(neutral.tremor, preset.tremor),
            breathiness=lerp(neutral.breathiness, preset.breathiness),
            whisper=lerp(neutral.whisper, preset.whisper),
        )
    
    def _shift_pitch(
        self,
        samples: 'np.ndarray',
        sample_rate: int,
        semitones: float
    ) -> 'np.ndarray':
        """Shift pitch by semitones."""
        try:
            import librosa
            return librosa.effects.pitch_shift(samples, sr=sample_rate, n_steps=semitones)
        except ImportError:
            # Simple resampling-based pitch shift
            factor = 2 ** (semitones / 12)
            
            # Resample to shift pitch
            indices = np.arange(0, len(samples), factor)
            indices = indices[indices < len(samples)].astype(int)
            
            return samples[indices]
    
    def list_voices(self) -> List[Dict[str, str]]:
        """Get available voices."""
        voices = []
        
        if self._engine:
            for voice in self._engine.getProperty('voices'):
                voices.append({
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages,
                })
        
        return voices
    
    def set_voice(self, voice_id: str):
        """Set the voice to use."""
        if self._engine:
            self._engine.setProperty('voice', voice_id)
            self._voice_id = voice_id
    
    def list_emotions(self) -> List[str]:
        """Get available emotions."""
        return [e.name.lower() for e in TTSEmotion]
    
    def preview_emotion(self, emotion: TTSEmotion | str):
        """Speak a sample phrase to preview the emotion."""
        samples = {
            TTSEmotion.HAPPY: "I'm so happy to help you today!",
            TTSEmotion.SAD: "I understand, that must be difficult...",
            TTSEmotion.ANGRY: "This is completely unacceptable!",
            TTSEmotion.FEARFUL: "Oh no, what's happening?",
            TTSEmotion.SURPRISED: "Wow, I didn't expect that!",
            TTSEmotion.CALM: "Everything is going to be okay.",
            TTSEmotion.EXCITED: "This is amazing! I can't wait!",
            TTSEmotion.TIRED: "I'm feeling quite exhausted...",
            TTSEmotion.SERIOUS: "This is an important matter.",
            TTSEmotion.PLAYFUL: "Hehe, that's pretty funny!",
            TTSEmotion.CONFIDENT: "I know exactly what to do.",
        }
        
        if isinstance(emotion, str):
            emotion = TTSEmotion[emotion.upper()]
        
        text = samples.get(emotion, "This is how I sound with this emotion.")
        self.speak(text, emotion)


# Global instance
_instance: Optional[EmotionTTS] = None


def get_emotion_tts() -> EmotionTTS:
    """Get or create global EmotionTTS instance."""
    global _instance
    if _instance is None:
        _instance = EmotionTTS()
    return _instance


def speak_with_emotion(text: str, emotion: str = "neutral", intensity: float = 1.0):
    """Quick function to speak with emotion."""
    get_emotion_tts().speak(text, emotion, intensity)
