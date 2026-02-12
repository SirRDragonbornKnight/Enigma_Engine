"""
Real-Time Translation for Enigma AI Engine

Translate speech between languages in real-time.

Features:
- Speech recognition
- Text translation
- TTS output
- Multiple language pairs
- Streaming mode

Usage:
    from enigma_engine.voice.realtime_translation import RealtimeTranslator
    
    translator = RealtimeTranslator()
    
    # Translate English speech to Japanese
    translator.set_languages(source="en", target="ja")
    
    # Start listening and translating
    translator.start()
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TranslationProvider(Enum):
    """Translation provider options."""
    GOOGLE = "google"
    DEEPL = "deepl"
    OPENAI = "openai"
    LIBRE = "libre"
    LOCAL = "local"  # Local models


@dataclass
class TranslationConfig:
    """Translation configuration."""
    source_language: str = "en"
    target_language: str = "ja"
    provider: TranslationProvider = TranslationProvider.GOOGLE
    
    # Voice settings
    voice_input: bool = True
    voice_output: bool = True
    
    # API settings
    api_key: str = ""
    api_endpoint: str = ""


@dataclass
class TranslatedSegment:
    """A translated speech segment."""
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.0


class TextTranslator:
    """Translate text between languages."""
    
    def __init__(
        self,
        provider: TranslationProvider = TranslationProvider.GOOGLE,
        api_key: str = ""
    ):
        self.provider = provider
        self.api_key = api_key
        
        # Cache translations
        self._cache: Dict[str, str] = {}
    
    def translate(
        self,
        text: str,
        source: str = "en",
        target: str = "ja"
    ) -> str:
        """
        Translate text.
        
        Args:
            text: Text to translate
            source: Source language code
            target: Target language code
            
        Returns:
            Translated text
        """
        # Check cache
        cache_key = f"{source}:{target}:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Translate based on provider
        if self.provider == TranslationProvider.GOOGLE:
            result = self._google_translate(text, source, target)
        elif self.provider == TranslationProvider.DEEPL:
            result = self._deepl_translate(text, source, target)
        elif self.provider == TranslationProvider.OPENAI:
            result = self._openai_translate(text, source, target)
        elif self.provider == TranslationProvider.LOCAL:
            result = self._local_translate(text, source, target)
        else:
            result = text  # Fallback
        
        # Cache
        self._cache[cache_key] = result
        
        return result
    
    def _google_translate(self, text: str, source: str, target: str) -> str:
        """Translate using Google Translate."""
        try:
            from googletrans import Translator
            translator = Translator()
            result = translator.translate(text, src=source, dest=target)
            return result.text
        except ImportError:
            logger.warning("googletrans not installed")
            return text
        except Exception as e:
            logger.error(f"Google translate error: {e}")
            return text
    
    def _deepl_translate(self, text: str, source: str, target: str) -> str:
        """Translate using DeepL API."""
        if not self.api_key:
            logger.error("DeepL API key required")
            return text
        
        try:
            import requests
            
            response = requests.post(
                "https://api-free.deepl.com/v2/translate",
                data={
                    "auth_key": self.api_key,
                    "text": text,
                    "source_lang": source.upper(),
                    "target_lang": target.upper()
                },
                timeout=30
            )
            
            if response.ok:
                data = response.json()
                return data["translations"][0]["text"]
            else:
                return text
                
        except Exception as e:
            logger.error(f"DeepL translate error: {e}")
            return text
    
    def _openai_translate(self, text: str, source: str, target: str) -> str:
        """Translate using OpenAI."""
        if not self.api_key:
            logger.error("OpenAI API key required")
            return text
        
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"Translate from {source} to {target}. Output only the translation."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI translate error: {e}")
            return text
    
    def _local_translate(self, text: str, source: str, target: str) -> str:
        """Translate using local model."""
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            model_name = f"Helsinki-NLP/opus-mt-{source}-{target}"
            
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            outputs = model.generate(**inputs)
            
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Local translate error: {e}")
            return text


class RealtimeTranslator:
    """Real-time speech translation."""
    
    def __init__(self, config: Optional[TranslationConfig] = None):
        """
        Initialize translator.
        
        Args:
            config: Translation configuration
        """
        self.config = config or TranslationConfig()
        
        # Components
        self._translator = TextTranslator(
            provider=self.config.provider,
            api_key=self.config.api_key
        )
        
        # State
        self._running = False
        self._paused = False
        
        # Queues
        self._text_queue: queue.Queue = queue.Queue()
        self._output_queue: queue.Queue = queue.Queue()
        
        # Threads
        self._listen_thread: Optional[threading.Thread] = None
        self._translate_thread: Optional[threading.Thread] = None
        self._speak_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._on_original: List[Callable[[str], None]] = []
        self._on_translated: List[Callable[[TranslatedSegment], None]] = []
        
        # Stats
        self._segments_translated = 0
        
        logger.info(f"RealtimeTranslator initialized: {self.config.source_language} -> {self.config.target_language}")
    
    def set_languages(self, source: str, target: str):
        """Set source and target languages."""
        self.config.source_language = source
        self.config.target_language = target
    
    def start(self):
        """Start real-time translation."""
        if self._running:
            return
        
        self._running = True
        self._paused = False
        
        # Start threads
        if self.config.voice_input:
            self._listen_thread = threading.Thread(
                target=self._listen_loop,
                daemon=True
            )
            self._listen_thread.start()
        
        self._translate_thread = threading.Thread(
            target=self._translate_loop,
            daemon=True
        )
        self._translate_thread.start()
        
        if self.config.voice_output:
            self._speak_thread = threading.Thread(
                target=self._speak_loop,
                daemon=True
            )
            self._speak_thread.start()
        
        logger.info("Real-time translation started")
    
    def stop(self):
        """Stop translation."""
        self._running = False
        
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
        if self._translate_thread:
            self._translate_thread.join(timeout=2.0)
        if self._speak_thread:
            self._speak_thread.join(timeout=2.0)
        
        logger.info("Real-time translation stopped")
    
    def pause(self):
        """Pause translation."""
        self._paused = True
    
    def resume(self):
        """Resume translation."""
        self._paused = False
    
    def translate_text(self, text: str) -> TranslatedSegment:
        """
        Translate text manually.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated segment
        """
        translated = self._translator.translate(
            text,
            source=self.config.source_language,
            target=self.config.target_language
        )
        
        segment = TranslatedSegment(
            original_text=text,
            translated_text=translated,
            source_lang=self.config.source_language,
            target_lang=self.config.target_language
        )
        
        self._segments_translated += 1
        
        return segment
    
    def on_original(self, callback: Callable[[str], None]):
        """Register callback for original text."""
        self._on_original.append(callback)
    
    def on_translated(self, callback: Callable[[TranslatedSegment], None]):
        """Register callback for translated text."""
        self._on_translated.append(callback)
    
    def get_stats(self) -> Dict:
        """Get translation statistics."""
        return {
            "segments_translated": self._segments_translated,
            "source_language": self.config.source_language,
            "target_language": self.config.target_language,
            "running": self._running,
            "paused": self._paused
        }
    
    def _listen_loop(self):
        """Listen for speech input."""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            mic = sr.Microphone()
        except ImportError:
            logger.error("speech_recognition not installed")
            return
        
        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue
            
            try:
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio = recognizer.listen(source, timeout=5.0, phrase_time_limit=10.0)
                
                # Recognize
                text = recognizer.recognize_google(
                    audio,
                    language=self.config.source_language
                )
                
                if text:
                    # Notify callbacks
                    for callback in self._on_original:
                        try:
                            callback(text)
                        except Exception:
                            pass  # Intentionally silent
                    
                    # Queue for translation
                    self._text_queue.put(text)
                    
            except Exception:
                pass  # Timeout or recognition error
    
    def _translate_loop(self):
        """Translate queued text."""
        while self._running:
            try:
                text = self._text_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            # Translate
            segment = self.translate_text(text)
            
            # Notify callbacks
            for callback in self._on_translated:
                try:
                    callback(segment)
                except Exception:
                    pass  # Intentionally silent
            
            # Queue for speech output
            if self.config.voice_output:
                self._output_queue.put(segment)
    
    def _speak_loop(self):
        """Speak translated text."""
        # Try to get TTS
        tts = None
        try:
            from enigma_engine.voice.multi_speaker import get_multi_speaker_tts
            tts = get_multi_speaker_tts()
        except ImportError:
            try:
                import pyttsx3
                tts = pyttsx3.init()
            except ImportError:
                logger.warning("No TTS available")
        
        while self._running:
            try:
                segment = self._output_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            if not tts:
                continue
            
            # Speak translated text
            try:
                if hasattr(tts, 'speak'):
                    audio = tts.speak(segment.translated_text)
                    # Play audio
                    try:
                        import sounddevice as sd
                        sd.play(audio.audio_data, audio.sample_rate)
                        sd.wait()
                    except Exception:
                        pass  # Intentionally silent
                else:
                    tts.say(segment.translated_text)
                    tts.runAndWait()
            except Exception as e:
                logger.error(f"TTS error: {e}")


# Language code mappings
LANGUAGE_CODES = {
    "english": "en",
    "japanese": "ja",
    "chinese": "zh",
    "korean": "ko",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "arabic": "ar",
    "hindi": "hi"
}


def create_translator(
    source: str = "en",
    target: str = "ja",
    provider: str = "google"
) -> RealtimeTranslator:
    """
    Create a configured translator.
    
    Args:
        source: Source language
        target: Target language
        provider: Translation provider
        
    Returns:
        Configured translator
    """
    config = TranslationConfig(
        source_language=LANGUAGE_CODES.get(source.lower(), source),
        target_language=LANGUAGE_CODES.get(target.lower(), target),
        provider=TranslationProvider(provider)
    )
    
    return RealtimeTranslator(config)


# Global instance
_translator: Optional[RealtimeTranslator] = None


def get_realtime_translator() -> RealtimeTranslator:
    """Get or create global translator."""
    global _translator
    if _translator is None:
        _translator = RealtimeTranslator()
    return _translator
