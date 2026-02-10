"""
================================================================================
Emotional TTS - Emotional inflection in generated speech.
================================================================================

Add emotional expression to text-to-speech by:
1. Detecting emotion from text content
2. Applying prosody modifications (rate, pitch, volume)
3. Inserting appropriate pauses and emphasis
4. Using SSML to render emotional speech

Supported emotions:
- happy, excited, enthusiastic
- sad, melancholic, disappointed
- angry, frustrated, annoyed
- fearful, anxious, nervous
- calm, neutral, professional
- surprised, amazed
- sarcastic, ironic
- empathetic, caring
- confident, assertive

USAGE:
    from enigma_engine.voice.emotional_tts import EmotionalTTS, Emotion
    
    tts = EmotionalTTS()
    
    # Speak with specific emotion
    tts.speak("I can't believe it!", emotion=Emotion.SURPRISED)
    
    # Auto-detect emotion from text
    tts.speak_auto("This is terrible news...")
    
    # Get emotional SSML
    ssml = tts.to_ssml("Hello!", emotion=Emotion.HAPPY)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class Emotion(Enum):
    """Supported emotions for TTS."""
    
    # Positive emotions
    NEUTRAL = auto()
    HAPPY = auto()
    EXCITED = auto()
    ENTHUSIASTIC = auto()
    JOYFUL = auto()
    
    # Negative emotions
    SAD = auto()
    MELANCHOLIC = auto()
    DISAPPOINTED = auto()
    DEPRESSED = auto()
    
    # Anger spectrum
    ANGRY = auto()
    FRUSTRATED = auto()
    ANNOYED = auto()
    FURIOUS = auto()
    
    # Fear spectrum
    FEARFUL = auto()
    ANXIOUS = auto()
    NERVOUS = auto()
    TERRIFIED = auto()
    
    # Calm spectrum
    CALM = auto()
    RELAXED = auto()
    SERENE = auto()
    PROFESSIONAL = auto()
    
    # Surprise
    SURPRISED = auto()
    AMAZED = auto()
    SHOCKED = auto()
    
    # Complex emotions
    SARCASTIC = auto()
    IRONIC = auto()
    EMPATHETIC = auto()
    CARING = auto()
    CONFIDENT = auto()
    ASSERTIVE = auto()
    CURIOUS = auto()
    THOUGHTFUL = auto()
    URGENT = auto()
    WHISPER = auto()


@dataclass
class EmotionProfile:
    """Prosody profile for an emotion."""
    
    # Base modifiers (1.0 = normal)
    rate: float = 1.0           # Speech rate
    pitch: float = 1.0          # Pitch level
    pitch_range: float = 1.0    # Pitch variation
    volume: float = 1.0         # Volume level
    
    # Timing
    pause_multiplier: float = 1.0    # Scale pauses
    word_gap_ms: int = 0             # Extra gap between words
    sentence_pause_ms: int = 0       # Extra pause at sentence end
    
    # Emphasis patterns
    emphasis_words: list[str] = field(default_factory=list)  # Words to emphasize
    emphasis_pattern: str = "none"   # none, first, last, alternating
    
    # Voice quality hints (for advanced TTS)
    breathiness: float = 0.0    # 0-1
    tension: float = 0.5        # 0-1 (0=lax, 1=tense)
    
    # SSML generation hints
    use_breaks: bool = True
    break_on_punctuation: bool = True


# Emotion profiles with prosody settings
EMOTION_PROFILES: dict[Emotion, EmotionProfile] = {
    # Neutral
    Emotion.NEUTRAL: EmotionProfile(),
    
    # Happy/Positive
    Emotion.HAPPY: EmotionProfile(
        rate=1.1,
        pitch=1.15,
        pitch_range=1.2,
        volume=1.1,
        emphasis_pattern="alternating",
    ),
    Emotion.EXCITED: EmotionProfile(
        rate=1.25,
        pitch=1.2,
        pitch_range=1.4,
        volume=1.2,
        word_gap_ms=-20,  # Faster
    ),
    Emotion.ENTHUSIASTIC: EmotionProfile(
        rate=1.15,
        pitch=1.15,
        pitch_range=1.3,
        volume=1.15,
    ),
    Emotion.JOYFUL: EmotionProfile(
        rate=1.1,
        pitch=1.2,
        pitch_range=1.3,
        volume=1.1,
        breathiness=0.1,
    ),
    
    # Sad/Negative
    Emotion.SAD: EmotionProfile(
        rate=0.85,
        pitch=0.9,
        pitch_range=0.7,
        volume=0.85,
        pause_multiplier=1.3,
        sentence_pause_ms=200,
    ),
    Emotion.MELANCHOLIC: EmotionProfile(
        rate=0.8,
        pitch=0.85,
        pitch_range=0.6,
        volume=0.8,
        pause_multiplier=1.4,
        breathiness=0.2,
    ),
    Emotion.DISAPPOINTED: EmotionProfile(
        rate=0.9,
        pitch=0.9,
        pitch_range=0.8,
        volume=0.9,
        sentence_pause_ms=150,
    ),
    Emotion.DEPRESSED: EmotionProfile(
        rate=0.75,
        pitch=0.8,
        pitch_range=0.5,
        volume=0.75,
        pause_multiplier=1.5,
    ),
    
    # Angry
    Emotion.ANGRY: EmotionProfile(
        rate=1.1,
        pitch=1.1,
        pitch_range=1.3,
        volume=1.3,
        tension=0.8,
        emphasis_pattern="first",
    ),
    Emotion.FRUSTRATED: EmotionProfile(
        rate=1.05,
        pitch=1.05,
        pitch_range=1.2,
        volume=1.15,
        tension=0.7,
    ),
    Emotion.ANNOYED: EmotionProfile(
        rate=1.0,
        pitch=1.0,
        volume=1.1,
        tension=0.6,
        sentence_pause_ms=100,
    ),
    Emotion.FURIOUS: EmotionProfile(
        rate=1.2,
        pitch=1.15,
        pitch_range=1.4,
        volume=1.4,
        tension=0.9,
        word_gap_ms=-30,
    ),
    
    # Fear
    Emotion.FEARFUL: EmotionProfile(
        rate=1.15,
        pitch=1.1,
        pitch_range=1.3,
        volume=0.9,
        breathiness=0.3,
        tension=0.7,
    ),
    Emotion.ANXIOUS: EmotionProfile(
        rate=1.1,
        pitch=1.05,
        pitch_range=1.2,
        volume=0.95,
        breathiness=0.2,
    ),
    Emotion.NERVOUS: EmotionProfile(
        rate=1.05,
        pitch=1.05,
        volume=0.9,
        pause_multiplier=0.8,
        breathiness=0.15,
    ),
    Emotion.TERRIFIED: EmotionProfile(
        rate=1.3,
        pitch=1.2,
        pitch_range=1.5,
        volume=0.85,
        breathiness=0.4,
        tension=0.8,
    ),
    
    # Calm
    Emotion.CALM: EmotionProfile(
        rate=0.9,
        pitch=0.95,
        pitch_range=0.8,
        volume=0.9,
        pause_multiplier=1.2,
    ),
    Emotion.RELAXED: EmotionProfile(
        rate=0.85,
        pitch=0.95,
        pitch_range=0.7,
        volume=0.85,
        pause_multiplier=1.3,
        breathiness=0.1,
    ),
    Emotion.SERENE: EmotionProfile(
        rate=0.8,
        pitch=0.9,
        pitch_range=0.6,
        volume=0.8,
        pause_multiplier=1.4,
    ),
    Emotion.PROFESSIONAL: EmotionProfile(
        rate=0.95,
        pitch=1.0,
        pitch_range=0.9,
        volume=1.0,
    ),
    
    # Surprise
    Emotion.SURPRISED: EmotionProfile(
        rate=1.1,
        pitch=1.25,
        pitch_range=1.5,
        volume=1.15,
        emphasis_pattern="first",
    ),
    Emotion.AMAZED: EmotionProfile(
        rate=0.95,
        pitch=1.2,
        pitch_range=1.4,
        volume=1.1,
        pause_multiplier=1.2,
    ),
    Emotion.SHOCKED: EmotionProfile(
        rate=1.2,
        pitch=1.3,
        pitch_range=1.6,
        volume=1.2,
        sentence_pause_ms=300,
    ),
    
    # Complex
    Emotion.SARCASTIC: EmotionProfile(
        rate=0.9,
        pitch=1.0,
        pitch_range=1.3,
        volume=1.0,
        emphasis_pattern="last",
        pause_multiplier=1.1,
    ),
    Emotion.IRONIC: EmotionProfile(
        rate=0.95,
        pitch=1.05,
        pitch_range=1.2,
        volume=1.0,
        pause_multiplier=1.15,
    ),
    Emotion.EMPATHETIC: EmotionProfile(
        rate=0.9,
        pitch=0.95,
        pitch_range=1.1,
        volume=0.95,
        breathiness=0.1,
        pause_multiplier=1.2,
    ),
    Emotion.CARING: EmotionProfile(
        rate=0.9,
        pitch=1.0,
        pitch_range=1.1,
        volume=0.9,
        breathiness=0.15,
    ),
    Emotion.CONFIDENT: EmotionProfile(
        rate=0.95,
        pitch=0.95,
        pitch_range=1.0,
        volume=1.1,
        tension=0.6,
    ),
    Emotion.ASSERTIVE: EmotionProfile(
        rate=1.0,
        pitch=0.95,
        pitch_range=1.1,
        volume=1.15,
        tension=0.65,
        emphasis_pattern="first",
    ),
    Emotion.CURIOUS: EmotionProfile(
        rate=1.0,
        pitch=1.1,
        pitch_range=1.3,
        volume=1.0,
    ),
    Emotion.THOUGHTFUL: EmotionProfile(
        rate=0.85,
        pitch=0.95,
        pitch_range=0.9,
        volume=0.9,
        pause_multiplier=1.3,
    ),
    Emotion.URGENT: EmotionProfile(
        rate=1.3,
        pitch=1.1,
        pitch_range=1.2,
        volume=1.2,
        word_gap_ms=-40,
        tension=0.7,
    ),
    Emotion.WHISPER: EmotionProfile(
        rate=0.85,
        pitch=0.9,
        pitch_range=0.5,
        volume=0.5,
        breathiness=0.5,
    ),
}


# Emotion detection patterns
EMOTION_PATTERNS: dict[Emotion, list[str]] = {
    Emotion.HAPPY: [
        r'\b(happy|glad|pleased|delighted|wonderful|great|awesome|fantastic|yay)\b',
        r'[!]{2,}',
        r':\)|😊|😄|🎉',
    ],
    Emotion.EXCITED: [
        r'\b(excited|amazing|incredible|wow|omg|can\'t wait)\b',
        r'[!]{3,}',
        r'🎊|🤩|😍',
    ],
    Emotion.SAD: [
        r'\b(sad|sorry|unfortunately|regret|miss|lost|gone)\b',
        r'😢|😞|💔',
    ],
    Emotion.DISAPPOINTED: [
        r'\b(disappointed|let down|expected|hoped|wish)\b',
    ],
    Emotion.ANGRY: [
        r'\b(angry|furious|hate|stupid|ridiculous|unacceptable)\b',
        r'[!]{2,}.*\b(not|never|stop)\b',
        r'😠|😡|🤬',
    ],
    Emotion.FRUSTRATED: [
        r'\b(frustrated|annoying|ugh|argh|why won\'t|doesn\'t work)\b',
    ],
    Emotion.FEARFUL: [
        r'\b(scared|afraid|terrified|frightening|dangerous|help)\b',
        r'😨|😱|😰',
    ],
    Emotion.ANXIOUS: [
        r'\b(worried|anxious|nervous|concern|stress)\b',
    ],
    Emotion.SURPRISED: [
        r'\b(surprised|unexpected|shocking|can\'t believe|what)\b',
        r'\?{2,}',
        r'😮|😲|🤯',
    ],
    Emotion.CALM: [
        r'\b(calm|peaceful|relaxed|okay|fine|alright)\b',
    ],
    Emotion.SARCASTIC: [
        r'\b(oh really|sure|right|totally|obviously)\b.*[.]{3}',
        r'\b(wow|great|wonderful)\b.*\bnot\b',
    ],
    Emotion.EMPATHETIC: [
        r'\b(understand|feel|sorry to hear|must be|here for you)\b',
        r'💙|🤗|❤️',
    ],
    Emotion.CURIOUS: [
        r'\b(wonder|curious|how|why|what if)\b',
        r'\?$',
        r'🤔',
    ],
    Emotion.URGENT: [
        r'\b(urgent|immediately|now|asap|hurry|emergency|quick)\b',
        r'[!]{2,}',
    ],
}


class EmotionDetector:
    """Detect emotion from text content."""
    
    def __init__(self):
        # Compile patterns
        self._patterns: dict[Emotion, list[re.Pattern]] = {}
        for emotion, patterns in EMOTION_PATTERNS.items():
            self._patterns[emotion] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def detect(self, text: str) -> tuple[Emotion, float]:
        """
        Detect the primary emotion in text.
        
        Args:
            text: Input text
        
        Returns:
            Tuple of (emotion, confidence 0-1)
        """
        scores: dict[Emotion, float] = {}
        
        for emotion, patterns in self._patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = pattern.findall(text)
                score += len(matches) * 0.3
            
            if score > 0:
                scores[emotion] = min(1.0, score)
        
        if not scores:
            return Emotion.NEUTRAL, 1.0
        
        # Get highest scoring emotion
        best_emotion = max(scores, key=scores.get)
        return best_emotion, scores[best_emotion]
    
    def detect_multiple(self, text: str) -> list[tuple[Emotion, float]]:
        """
        Detect all emotions present in text.
        
        Returns:
            List of (emotion, confidence) tuples, sorted by confidence
        """
        scores: dict[Emotion, float] = {}
        
        for emotion, patterns in self._patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = pattern.findall(text)
                score += len(matches) * 0.3
            
            if score > 0:
                scores[emotion] = min(1.0, score)
        
        if not scores:
            return [(Emotion.NEUTRAL, 1.0)]
        
        # Sort by score descending
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class EmotionalTTS:
    """
    Text-to-speech with emotional expression.
    
    Modifies prosody based on emotion to make speech more expressive.
    """
    
    def __init__(self, tts_engine: Any = None):
        """
        Initialize emotional TTS.
        
        Args:
            tts_engine: TTS engine instance (pyttsx3, etc.)
        """
        self.tts_engine = tts_engine
        self.detector = EmotionDetector()
        
        # Import SSML processor if available
        try:
            from .ssml import SSMLProcessor
            self._ssml_processor = SSMLProcessor(tts_engine)
        except ImportError:
            self._ssml_processor = None
    
    def speak(
        self,
        text: str,
        emotion: Emotion = Emotion.NEUTRAL,
        blocking: bool = True
    ):
        """
        Speak text with specified emotion.
        
        Args:
            text: Text to speak
            emotion: Emotion to express
            blocking: Wait for speech to complete
        """
        ssml = self.to_ssml(text, emotion)
        
        if self._ssml_processor:
            self._ssml_processor.speak(ssml, blocking)
        else:
            # Fallback: apply basic prosody
            self._speak_with_prosody(text, emotion, blocking)
    
    def speak_auto(self, text: str, blocking: bool = True):
        """
        Speak text with auto-detected emotion.
        
        Args:
            text: Text to speak
            blocking: Wait for speech to complete
        """
        emotion, confidence = self.detector.detect(text)
        logger.debug(f"Detected emotion: {emotion.name} ({confidence:.2f})")
        self.speak(text, emotion, blocking)
    
    def to_ssml(self, text: str, emotion: Emotion = Emotion.NEUTRAL) -> str:
        """
        Convert text to SSML with emotional prosody.
        
        Args:
            text: Input text
            emotion: Emotion to express
        
        Returns:
            SSML markup string
        """
        profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES[Emotion.NEUTRAL])
        
        # Build prosody attributes
        prosody_attrs = []
        
        if profile.rate != 1.0:
            rate_pct = int((profile.rate - 1.0) * 100)
            if rate_pct >= 0:
                prosody_attrs.append(f'rate="+{rate_pct}%"')
            else:
                prosody_attrs.append(f'rate="{rate_pct}%"')
        
        if profile.pitch != 1.0:
            pitch_pct = int((profile.pitch - 1.0) * 100)
            if pitch_pct >= 0:
                prosody_attrs.append(f'pitch="+{pitch_pct}%"')
            else:
                prosody_attrs.append(f'pitch="{pitch_pct}%"')
        
        if profile.volume != 1.0:
            if profile.volume >= 1.5:
                prosody_attrs.append('volume="x-loud"')
            elif profile.volume >= 1.2:
                prosody_attrs.append('volume="loud"')
            elif profile.volume <= 0.5:
                prosody_attrs.append('volume="x-soft"')
            elif profile.volume <= 0.8:
                prosody_attrs.append('volume="soft"')
        
        # Process text with breaks and emphasis
        processed = self._process_text(text, profile)
        
        # Wrap in prosody if needed
        if prosody_attrs:
            prosody_str = " ".join(prosody_attrs)
            content = f'<prosody {prosody_str}>{processed}</prosody>'
        else:
            content = processed
        
        return f'<speak>{content}</speak>'
    
    def _process_text(self, text: str, profile: EmotionProfile) -> str:
        """Process text adding breaks and emphasis based on profile."""
        result = text
        
        # Add breaks on punctuation
        if profile.break_on_punctuation and profile.use_breaks:
            # Sentence endings
            pause_ms = int(400 * profile.pause_multiplier) + profile.sentence_pause_ms
            result = re.sub(
                r'([.!?])\s+',
                f'\\1<break time="{pause_ms}ms"/> ',
                result
            )
            
            # Commas
            comma_pause = int(200 * profile.pause_multiplier)
            result = re.sub(
                r',\s+',
                f',<break time="{comma_pause}ms"/> ',
                result
            )
        
        # Apply emphasis patterns
        if profile.emphasis_pattern != "none":
            result = self._apply_emphasis(result, profile)
        
        return result
    
    def _apply_emphasis(self, text: str, profile: EmotionProfile) -> str:
        """Apply emphasis pattern to text."""
        words = text.split()
        
        if not words:
            return text
        
        if profile.emphasis_pattern == "first":
            # Emphasize first word of each sentence
            result = []
            sentence_start = True
            for word in words:
                if sentence_start and word[0].isalpha():
                    result.append(f'<emphasis level="moderate">{word}</emphasis>')
                    sentence_start = False
                else:
                    result.append(word)
                
                if word.rstrip('.,!?;:').endswith(('.', '!', '?')):
                    sentence_start = True
            
            return ' '.join(result)
        
        elif profile.emphasis_pattern == "last":
            # Emphasize last word of each sentence
            result = []
            for i, word in enumerate(words):
                next_word = words[i + 1] if i + 1 < len(words) else None
                
                if word.rstrip().endswith(('.', '!', '?')) or next_word is None:
                    clean = word.rstrip('.,!?;:')
                    punct = word[len(clean):]
                    result.append(f'<emphasis level="moderate">{clean}</emphasis>{punct}')
                else:
                    result.append(word)
            
            return ' '.join(result)
        
        elif profile.emphasis_pattern == "alternating":
            # Light emphasis on alternating words (for excited speech)
            result = []
            for i, word in enumerate(words):
                if i % 3 == 0 and len(word) > 3:  # Every 3rd word
                    result.append(f'<emphasis level="reduced">{word}</emphasis>')
                else:
                    result.append(word)
            
            return ' '.join(result)
        
        return text
    
    def _speak_with_prosody(
        self,
        text: str,
        emotion: Emotion,
        blocking: bool
    ):
        """Speak with prosody applied directly to TTS engine."""
        if not self.tts_engine:
            logger.warning("No TTS engine configured")
            return
        
        profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES[Emotion.NEUTRAL])
        
        try:
            # Try pyttsx3-style API
            if hasattr(self.tts_engine, 'getProperty'):
                original_rate = self.tts_engine.getProperty('rate')
                original_volume = self.tts_engine.getProperty('volume')
                
                # Apply emotional prosody
                new_rate = int(original_rate * profile.rate)
                new_volume = min(1.0, original_volume * profile.volume)
                
                self.tts_engine.setProperty('rate', new_rate)
                self.tts_engine.setProperty('volume', new_volume)
                
                self.tts_engine.say(text)
                if blocking:
                    self.tts_engine.runAndWait()
                
                # Restore
                self.tts_engine.setProperty('rate', original_rate)
                self.tts_engine.setProperty('volume', original_volume)
            
            elif hasattr(self.tts_engine, 'speak'):
                self.tts_engine.speak(text)
            
            else:
                logger.warning("Unknown TTS engine type")
                
        except Exception as e:
            logger.error(f"Emotional TTS error: {e}")
    
    def get_emotion_profile(self, emotion: Emotion) -> EmotionProfile:
        """Get the prosody profile for an emotion."""
        return EMOTION_PROFILES.get(emotion, EMOTION_PROFILES[Emotion.NEUTRAL])
    
    def set_emotion_profile(self, emotion: Emotion, profile: EmotionProfile):
        """Set a custom prosody profile for an emotion."""
        EMOTION_PROFILES[emotion] = profile


def detect_emotion(text: str) -> tuple[Emotion, float]:
    """
    Convenience function to detect emotion in text.
    
    Returns:
        Tuple of (emotion, confidence)
    """
    detector = EmotionDetector()
    return detector.detect(text)


def emotional_ssml(text: str, emotion: Emotion = None) -> str:
    """
    Convenience function to generate emotional SSML.
    
    Args:
        text: Input text
        emotion: Emotion to use (auto-detect if None)
    
    Returns:
        SSML markup string
    """
    tts = EmotionalTTS()
    
    if emotion is None:
        emotion, _ = tts.detector.detect(text)
    
    return tts.to_ssml(text, emotion)
