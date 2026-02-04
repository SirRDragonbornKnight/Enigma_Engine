"""
================================================================================
Multilingual TTS - Support multiple languages in voice output.
================================================================================

Provides language detection, voice selection, and TTS routing for 50+ languages.

Features:
- Automatic language detection from text
- Language-specific voice selection
- Multi-engine support (pyttsx3, espeak, gTTS, Azure, etc.)
- Code-switching (mixed language) handling
- Phonetic transliteration fallback

USAGE:
    from forge_ai.voice.multilingual_tts import MultilingualTTS, Language
    
    tts = MultilingualTTS()
    
    # Speak with auto-detected language
    tts.speak("Hello, how are you?")  # English
    tts.speak("Bonjour, comment allez-vous?")  # French
    tts.speak("Hola, ¿cómo estás?")  # Spanish
    
    # Force specific language
    tts.speak("Hello", language=Language.JAPANESE)  # Uses Japanese voice
    
    # Get available voices for language
    voices = tts.get_voices(Language.GERMAN)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages for TTS."""
    
    # Major world languages
    ENGLISH = "en"
    ENGLISH_US = "en-US"
    ENGLISH_UK = "en-GB"
    ENGLISH_AU = "en-AU"
    
    SPANISH = "es"
    SPANISH_ES = "es-ES"
    SPANISH_MX = "es-MX"
    SPANISH_AR = "es-AR"
    
    FRENCH = "fr"
    FRENCH_FR = "fr-FR"
    FRENCH_CA = "fr-CA"
    
    GERMAN = "de"
    GERMAN_DE = "de-DE"
    GERMAN_AT = "de-AT"
    
    ITALIAN = "it"
    PORTUGUESE = "pt"
    PORTUGUESE_BR = "pt-BR"
    PORTUGUESE_PT = "pt-PT"
    
    # Asian languages
    CHINESE = "zh"
    CHINESE_CN = "zh-CN"
    CHINESE_TW = "zh-TW"
    CHINESE_HK = "zh-HK"
    
    JAPANESE = "ja"
    KOREAN = "ko"
    
    HINDI = "hi"
    BENGALI = "bn"
    TAMIL = "ta"
    TELUGU = "te"
    MARATHI = "mr"
    GUJARATI = "gu"
    KANNADA = "kn"
    MALAYALAM = "ml"
    PUNJABI = "pa"
    URDU = "ur"
    
    THAI = "th"
    VIETNAMESE = "vi"
    INDONESIAN = "id"
    MALAY = "ms"
    TAGALOG = "tl"
    
    # European languages
    RUSSIAN = "ru"
    POLISH = "pl"
    DUTCH = "nl"
    SWEDISH = "sv"
    NORWEGIAN = "no"
    DANISH = "da"
    FINNISH = "fi"
    GREEK = "el"
    CZECH = "cs"
    HUNGARIAN = "hu"
    ROMANIAN = "ro"
    UKRAINIAN = "uk"
    TURKISH = "tr"
    
    # Middle Eastern
    ARABIC = "ar"
    HEBREW = "he"
    PERSIAN = "fa"
    
    # Other
    SWAHILI = "sw"
    AFRIKAANS = "af"
    
    # Auto-detect
    AUTO = "auto"


@dataclass
class VoiceInfo:
    """Information about a TTS voice."""
    
    id: str                      # Voice identifier
    name: str                    # Display name
    language: Language           # Primary language
    language_code: str           # Full language code (e.g., "en-US")
    gender: str = "neutral"      # male, female, neutral
    engine: str = "unknown"      # TTS engine providing this voice
    quality: str = "standard"    # standard, high, neural
    
    # Optional metadata
    age: str = "adult"           # child, teen, adult, elderly
    style: str = "general"       # general, news, conversational, etc.
    sample_rate: int = 22050     # Audio sample rate
    
    def __str__(self):
        return f"{self.name} ({self.language_code}, {self.gender})"


@dataclass
class LanguageConfig:
    """Configuration for a language."""
    
    code: str                           # ISO 639-1 code
    name: str                           # Full name
    native_name: str                    # Name in the language itself
    default_voice: Optional[str] = None # Preferred voice ID
    fallback_language: Optional[str] = None  # Fallback if voice unavailable
    
    # Script/writing system
    script: str = "Latin"               # Latin, Cyrillic, Arabic, etc.
    rtl: bool = False                   # Right-to-left
    
    # Phonetic hints
    phonetic_alphabet: str = "ipa"      # IPA, X-SAMPA, etc.


# Language configurations
LANGUAGE_CONFIGS: Dict[Language, LanguageConfig] = {
    Language.ENGLISH: LanguageConfig("en", "English", "English"),
    Language.ENGLISH_US: LanguageConfig("en-US", "English (US)", "English"),
    Language.ENGLISH_UK: LanguageConfig("en-GB", "English (UK)", "English"),
    
    Language.SPANISH: LanguageConfig("es", "Spanish", "Español"),
    Language.SPANISH_ES: LanguageConfig("es-ES", "Spanish (Spain)", "Español"),
    Language.SPANISH_MX: LanguageConfig("es-MX", "Spanish (Mexico)", "Español"),
    
    Language.FRENCH: LanguageConfig("fr", "French", "Français"),
    Language.FRENCH_FR: LanguageConfig("fr-FR", "French (France)", "Français"),
    Language.FRENCH_CA: LanguageConfig("fr-CA", "French (Canada)", "Français"),
    
    Language.GERMAN: LanguageConfig("de", "German", "Deutsch"),
    Language.ITALIAN: LanguageConfig("it", "Italian", "Italiano"),
    
    Language.PORTUGUESE: LanguageConfig("pt", "Portuguese", "Português"),
    Language.PORTUGUESE_BR: LanguageConfig("pt-BR", "Portuguese (Brazil)", "Português"),
    
    Language.CHINESE: LanguageConfig("zh", "Chinese", "中文", script="Han"),
    Language.CHINESE_CN: LanguageConfig("zh-CN", "Chinese (Simplified)", "简体中文", script="Han"),
    Language.CHINESE_TW: LanguageConfig("zh-TW", "Chinese (Traditional)", "繁體中文", script="Han"),
    
    Language.JAPANESE: LanguageConfig("ja", "Japanese", "日本語", script="Japanese"),
    Language.KOREAN: LanguageConfig("ko", "Korean", "한국어", script="Hangul"),
    
    Language.HINDI: LanguageConfig("hi", "Hindi", "हिन्दी", script="Devanagari"),
    Language.ARABIC: LanguageConfig("ar", "Arabic", "العربية", script="Arabic", rtl=True),
    Language.HEBREW: LanguageConfig("he", "Hebrew", "עברית", script="Hebrew", rtl=True),
    
    Language.RUSSIAN: LanguageConfig("ru", "Russian", "Русский", script="Cyrillic"),
    Language.GREEK: LanguageConfig("el", "Greek", "Ελληνικά", script="Greek"),
    Language.TURKISH: LanguageConfig("tr", "Turkish", "Türkçe"),
    
    Language.THAI: LanguageConfig("th", "Thai", "ไทย", script="Thai"),
    Language.VIETNAMESE: LanguageConfig("vi", "Vietnamese", "Tiếng Việt"),
}


class LanguageDetector:
    """Detect language from text content."""
    
    # Character range patterns for script detection
    SCRIPT_PATTERNS = {
        "Han": r'[\u4e00-\u9fff]',
        "Hiragana": r'[\u3040-\u309f]',
        "Katakana": r'[\u30a0-\u30ff]',
        "Hangul": r'[\uac00-\ud7af]',
        "Cyrillic": r'[\u0400-\u04ff]',
        "Arabic": r'[\u0600-\u06ff]',
        "Hebrew": r'[\u0590-\u05ff]',
        "Devanagari": r'[\u0900-\u097f]',
        "Thai": r'[\u0e00-\u0e7f]',
        "Greek": r'[\u0370-\u03ff]',
    }
    
    # Common word patterns for Latin-script languages
    WORD_PATTERNS = {
        Language.ENGLISH: [
            r'\b(the|is|are|was|were|have|has|been|will|would|could|should)\b',
            r'\b(and|but|or|if|then|that|this|what|which|how|why|when)\b',
        ],
        Language.SPANISH: [
            r'\b(el|la|los|las|un|una|es|son|está|están|que|de|en|con)\b',
            r'\b(para|por|como|cuando|donde|porque|pero|muy|más)\b',
            r'[¿¡]',
        ],
        Language.FRENCH: [
            r'\b(le|la|les|un|une|des|est|sont|que|de|en|avec|pour)\b',
            r'\b(je|tu|il|elle|nous|vous|ils|elles|ce|cette|ces)\b',
            r"[àâçéèêëîïôùûü]",
        ],
        Language.GERMAN: [
            r'\b(der|die|das|ein|eine|ist|sind|und|oder|aber|wenn)\b',
            r'\b(ich|du|er|sie|es|wir|ihr|Sie|nicht|auch|noch)\b',
            r'[äöüß]',
        ],
        Language.ITALIAN: [
            r'\b(il|la|lo|gli|le|un|una|è|sono|che|di|in|con|per)\b',
            r'\b(io|tu|lui|lei|noi|voi|loro|non|anche|molto)\b',
            r'[àèéìíòóùú]',
        ],
        Language.PORTUGUESE: [
            r'\b(o|a|os|as|um|uma|é|são|que|de|em|com|para)\b',
            r'\b(eu|tu|ele|ela|nós|vós|eles|elas|não|também)\b',
            r'[ãõáéíóúâêô]',
        ],
        Language.DUTCH: [
            r'\b(de|het|een|is|zijn|van|in|met|op|aan|voor|maar)\b',
            r'\b(ik|je|jij|hij|zij|wij|jullie|niet|ook|nog)\b',
        ],
        Language.RUSSIAN: [
            r'\b(и|в|не|на|я|что|он|она|они|мы|вы|это|как|но)\b',
        ],
        Language.TURKISH: [
            r'\b(bir|ve|bu|için|ile|de|da|ne|var|ben|sen|o)\b',
            r'[ğışöüç]',
        ],
    }
    
    def __init__(self):
        # Compile patterns
        self._script_re = {
            name: re.compile(pattern)
            for name, pattern in self.SCRIPT_PATTERNS.items()
        }
        self._word_re = {
            lang: [re.compile(p, re.IGNORECASE) for p in patterns]
            for lang, patterns in self.WORD_PATTERNS.items()
        }
    
    def detect(self, text: str) -> Tuple[Language, float]:
        """
        Detect the primary language of text.
        
        Args:
            text: Input text
        
        Returns:
            Tuple of (language, confidence 0-1)
        """
        if not text.strip():
            return Language.ENGLISH, 0.0
        
        # First check scripts
        script = self._detect_script(text)
        
        if script == "Han":
            # Could be Chinese or Japanese with kanji
            if self._has_script(text, "Hiragana") or self._has_script(text, "Katakana"):
                return Language.JAPANESE, 0.9
            return Language.CHINESE, 0.85
        
        elif script == "Hiragana" or script == "Katakana":
            return Language.JAPANESE, 0.95
        
        elif script == "Hangul":
            return Language.KOREAN, 0.95
        
        elif script == "Cyrillic":
            # Could be Russian, Ukrainian, etc.
            return Language.RUSSIAN, 0.8
        
        elif script == "Arabic":
            return Language.ARABIC, 0.9
        
        elif script == "Hebrew":
            return Language.HEBREW, 0.9
        
        elif script == "Devanagari":
            return Language.HINDI, 0.85
        
        elif script == "Thai":
            return Language.THAI, 0.95
        
        elif script == "Greek":
            return Language.GREEK, 0.9
        
        # Latin script - use word patterns
        scores: Dict[Language, float] = {}
        
        for lang, patterns in self._word_re.items():
            score = 0.0
            for pattern in patterns:
                matches = pattern.findall(text)
                score += len(matches) * 0.1
            
            if score > 0:
                scores[lang] = min(1.0, score)
        
        if not scores:
            return Language.ENGLISH, 0.5  # Default fallback
        
        best_lang = max(scores, key=scores.get)
        return best_lang, scores[best_lang]
    
    def _detect_script(self, text: str) -> str:
        """Detect the primary script used in text."""
        script_counts = {}
        
        for script, pattern in self._script_re.items():
            count = len(pattern.findall(text))
            if count > 0:
                script_counts[script] = count
        
        if not script_counts:
            return "Latin"
        
        return max(script_counts, key=script_counts.get)
    
    def _has_script(self, text: str, script: str) -> bool:
        """Check if text contains any characters from a script."""
        pattern = self._script_re.get(script)
        return pattern and bool(pattern.search(text))
    
    def detect_segments(self, text: str) -> List[Tuple[str, Language, float]]:
        """
        Detect language for each segment of text (for code-switching).
        
        Returns:
            List of (segment_text, language, confidence) tuples
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        segments = []
        for sentence in sentences:
            if sentence.strip():
                lang, conf = self.detect(sentence)
                segments.append((sentence, lang, conf))
        
        return segments if segments else [(text, Language.ENGLISH, 0.5)]


class MultilingualTTS:
    """
    Text-to-speech with multi-language support.
    
    Automatically detects language and selects appropriate voice.
    """
    
    def __init__(self, tts_engine: Any = None):
        """
        Initialize multilingual TTS.
        
        Args:
            tts_engine: TTS engine instance (pyttsx3, etc.)
        """
        self.tts_engine = tts_engine
        self.detector = LanguageDetector()
        
        # Voice cache per language
        self._voices: Dict[str, List[VoiceInfo]] = {}
        self._current_voice: Optional[VoiceInfo] = None
        
        # Discover available voices
        self._discover_voices()
    
    def _discover_voices(self):
        """Discover available voices from TTS engine."""
        if not self.tts_engine:
            return
        
        try:
            # pyttsx3 style
            if hasattr(self.tts_engine, 'getProperty'):
                voices = self.tts_engine.getProperty('voices')
                
                for voice in voices:
                    # Parse language from voice
                    lang_code = self._extract_language_code(voice)
                    
                    info = VoiceInfo(
                        id=voice.id,
                        name=voice.name,
                        language=self._code_to_language(lang_code),
                        language_code=lang_code,
                        gender=self._guess_gender(voice.name),
                        engine="pyttsx3",
                    )
                    
                    if lang_code not in self._voices:
                        self._voices[lang_code] = []
                    self._voices[lang_code].append(info)
                    
                    # Also add to base language
                    base_code = lang_code.split("-")[0]
                    if base_code != lang_code:
                        if base_code not in self._voices:
                            self._voices[base_code] = []
                        self._voices[base_code].append(info)
                
                logger.info(f"Discovered {sum(len(v) for v in self._voices.values())} voices")
                
        except Exception as e:
            logger.warning(f"Failed to discover voices: {e}")
    
    def _extract_language_code(self, voice) -> str:
        """Extract language code from voice object."""
        # Try common attributes
        if hasattr(voice, 'languages') and voice.languages:
            return voice.languages[0]
        
        if hasattr(voice, 'id'):
            # Parse from ID like "com.apple.speech.synthesis.voice.Alex"
            voice_id = voice.id.lower()
            
            # Common patterns
            lang_patterns = [
                (r'\.([a-z]{2}-[a-z]{2})\.', 1),  # .en-us.
                (r'_([a-z]{2}-[a-z]{2})$', 1),     # _en-us
                (r'\.([a-z]{2})\.', 1),            # .en.
            ]
            
            for pattern, group in lang_patterns:
                match = re.search(pattern, voice_id)
                if match:
                    return match.group(group).replace('_', '-')
        
        # Default
        return "en-US"
    
    def _guess_gender(self, name: str) -> str:
        """Guess voice gender from name."""
        name_lower = name.lower()
        
        female_names = ['female', 'woman', 'samantha', 'victoria', 'karen', 'susan', 
                       'alice', 'emily', 'emma', 'sarah', 'anna', 'maria', 'julia']
        male_names = ['male', 'man', 'alex', 'daniel', 'david', 'james', 'john',
                     'tom', 'oliver', 'george', 'william', 'michael']
        
        if any(n in name_lower for n in female_names):
            return "female"
        if any(n in name_lower for n in male_names):
            return "male"
        
        return "neutral"
    
    def _code_to_language(self, code: str) -> Language:
        """Convert language code to Language enum."""
        code_lower = code.lower()
        
        for lang in Language:
            if lang.value == code_lower:
                return lang
            if lang.value == code_lower.split("-")[0]:
                return lang
        
        return Language.ENGLISH
    
    def speak(
        self,
        text: str,
        language: Language = Language.AUTO,
        voice_id: Optional[str] = None,
        blocking: bool = True
    ):
        """
        Speak text in specified or auto-detected language.
        
        Args:
            text: Text to speak
            language: Language to use (AUTO for detection)
            voice_id: Specific voice ID to use
            blocking: Wait for speech to complete
        """
        # Detect language if auto
        if language == Language.AUTO:
            language, confidence = self.detector.detect(text)
            logger.debug(f"Detected language: {language.name} ({confidence:.2f})")
        
        # Select voice
        voice = self._select_voice(language, voice_id)
        
        if voice and self.tts_engine:
            self._set_voice(voice)
        
        # Speak
        self._speak_text(text, blocking)
    
    def speak_multilingual(self, text: str, blocking: bool = True):
        """
        Speak text with code-switching support.
        
        Detects language changes within text and switches voices.
        
        Args:
            text: Text that may contain multiple languages
            blocking: Wait for speech to complete
        """
        segments = self.detector.detect_segments(text)
        
        for segment_text, language, confidence in segments:
            if confidence > 0.5:
                voice = self._select_voice(language)
                if voice and self.tts_engine:
                    self._set_voice(voice)
            
            self._speak_text(segment_text, blocking=True)  # Always block between segments
    
    def _select_voice(
        self,
        language: Language,
        voice_id: Optional[str] = None
    ) -> Optional[VoiceInfo]:
        """Select appropriate voice for language."""
        if voice_id:
            # Find specific voice
            for voices in self._voices.values():
                for voice in voices:
                    if voice.id == voice_id:
                        return voice
        
        # Get voices for language
        lang_code = language.value
        voices = self._voices.get(lang_code, [])
        
        if not voices:
            # Try base language
            base_code = lang_code.split("-")[0]
            voices = self._voices.get(base_code, [])
        
        if not voices:
            # Fallback to English
            voices = self._voices.get("en", []) or self._voices.get("en-US", [])
        
        if voices:
            # Prefer neural/high quality voices
            for voice in voices:
                if voice.quality == "neural":
                    return voice
            return voices[0]
        
        return None
    
    def _set_voice(self, voice: VoiceInfo):
        """Set the active voice."""
        if not self.tts_engine:
            return
        
        try:
            if hasattr(self.tts_engine, 'setProperty'):
                self.tts_engine.setProperty('voice', voice.id)
                self._current_voice = voice
                logger.debug(f"Set voice: {voice.name}")
        except Exception as e:
            logger.warning(f"Failed to set voice: {e}")
    
    def _speak_text(self, text: str, blocking: bool):
        """Speak text using current engine settings."""
        if not self.tts_engine:
            logger.warning("No TTS engine configured")
            return
        
        try:
            if hasattr(self.tts_engine, 'say'):
                self.tts_engine.say(text)
                if blocking and hasattr(self.tts_engine, 'runAndWait'):
                    self.tts_engine.runAndWait()
            elif hasattr(self.tts_engine, 'speak'):
                self.tts_engine.speak(text)
        except Exception as e:
            logger.error(f"TTS speak error: {e}")
    
    def get_voices(self, language: Language = None) -> List[VoiceInfo]:
        """
        Get available voices, optionally filtered by language.
        
        Args:
            language: Filter by language (None for all)
        
        Returns:
            List of available voices
        """
        if language is None:
            # Return all voices
            all_voices = []
            for voices in self._voices.values():
                all_voices.extend(voices)
            return list(set(all_voices))  # Remove duplicates
        
        lang_code = language.value
        voices = self._voices.get(lang_code, [])
        
        if not voices:
            base_code = lang_code.split("-")[0]
            voices = self._voices.get(base_code, [])
        
        return voices
    
    def get_supported_languages(self) -> List[Language]:
        """Get list of languages with available voices."""
        languages = set()
        
        for code in self._voices.keys():
            lang = self._code_to_language(code)
            languages.add(lang)
        
        return sorted(languages, key=lambda x: x.name)
    
    def get_current_voice(self) -> Optional[VoiceInfo]:
        """Get the currently selected voice."""
        return self._current_voice


def detect_language(text: str) -> Tuple[Language, float]:
    """
    Convenience function to detect language.
    
    Returns:
        Tuple of (language, confidence)
    """
    detector = LanguageDetector()
    return detector.detect(text)


def get_language_name(language: Language) -> str:
    """Get the display name for a language."""
    config = LANGUAGE_CONFIGS.get(language)
    return config.name if config else language.name.replace("_", " ").title()


def get_native_name(language: Language) -> str:
    """Get the native name for a language."""
    config = LANGUAGE_CONFIGS.get(language)
    return config.native_name if config else language.name
