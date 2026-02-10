"""
================================================================================
Profanity Filter - Filter profanity in transcription and text.
================================================================================

Multi-backend profanity detection and filtering:
- better-profanity: Fast detection with caching
- profanity-filter: ML-based with context awareness
- alt-profanity-check: Alternative word list
- Simple: Built-in word list (fallback)

USAGE:
    from enigma_engine.voice.profanity_filter import ProfanityFilter
    
    filter = ProfanityFilter()
    
    # Check if text contains profanity
    if filter.contains_profanity("some text"):
        print("Contains profanity!")
    
    # Censor profanity
    clean = filter.censor("some bad text")
    print(clean)  # "some *** text"
    
    # Custom replacement
    clean = filter.censor("bad text", replacement="[CENSORED]")
    
    # Get list of detected words
    words = filter.detect("text with bad words")
    print(words)  # ["bad", "words"]
    
    # Custom word lists
    filter.add_words(["customword"])
    filter.remove_words(["allowedword"])
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FilterBackend(Enum):
    """Available profanity filter backends."""
    
    BETTER_PROFANITY = auto()    # better-profanity library
    PROFANITY_FILTER = auto()    # profanity-filter library  
    ALT_PROFANITY = auto()       # alt-profanity-check
    SIMPLE = auto()              # Built-in word list


class CensorStyle(Enum):
    """How to censor detected profanity."""
    
    ASTERISKS = auto()      # Replace with ***
    GRAWLIX = auto()        # Replace with @#$%!
    FIRST_LAST = auto()     # Keep first/last letters: s**t
    FULL_WORD = auto()      # Replace whole word: [CENSORED]
    BEEP = auto()           # Replace with [BEEP]
    REMOVE = auto()         # Remove the word entirely
    CUSTOM = auto()         # Custom replacement string


@dataclass
class FilterConfig:
    """Configuration for profanity filtering."""
    
    # Backend selection
    preferred_backend: FilterBackend = FilterBackend.BETTER_PROFANITY
    
    # Filter settings
    enabled: bool = True
    censor_style: CensorStyle = CensorStyle.ASTERISKS
    custom_replacement: str = "[CENSORED]"
    
    # Detection sensitivity
    include_partial_matches: bool = True   # Detect words within words
    case_sensitive: bool = False
    detect_leetspeak: bool = True          # Detect 4ss, sh1t, etc.
    detect_obfuscation: bool = True        # Detect s.h.i.t, s_h_i_t
    
    # Language
    language: str = "en"
    
    # Custom lists
    additional_words: set[str] = field(default_factory=set)
    allowed_words: set[str] = field(default_factory=set)
    
    # Word list file
    custom_wordlist_path: Path | None = None


@dataclass
class FilterResult:
    """Result of profanity filtering."""
    
    original: str
    filtered: str  
    contains_profanity: bool
    detected_words: list[str] = field(default_factory=list)
    positions: list[tuple[int, int]] = field(default_factory=list)
    
    @property
    def profanity_count(self) -> int:
        """Number of profane words detected."""
        return len(self.detected_words)


class ProfanityFilter:
    """Multi-backend profanity detection and filtering."""
    
    # Built-in minimal word list (common English profanity)
    # This is intentionally minimal - real filtering should use a proper library
    _BUILTIN_WORDS = {
        # 4-letter words
        "shit", "fuck", "damn", "hell", "crap", "piss", "dick", "cock",
        "cunt", "twat", "slut", "whore", "bitch", "arse",
        # Longer variations
        "asshole", "bastard", "bullshit", "fucking", "motherfucker",
        "shithead", "dickhead", "dumbass", "jackass",
        # Slurs (minimal list - real filter libraries have comprehensive lists)
        "nigger", "nigga", "faggot", "fag", "retard", "retarded",
        # Religious
        "goddamn", "goddamnit",
    }
    
    # Leetspeak substitutions
    _LEETSPEAK_MAP = {
        '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's',
        '7': 't', '8': 'b', '@': 'a', '$': 's', '!': 'i',
        '+': 't', '(': 'c', ')': 'c'
    }
    
    def __init__(self, config: FilterConfig | None = None):
        """Initialize filter with config."""
        self.config = config or FilterConfig()
        self._backend: Any | None = None
        self._backend_type: FilterBackend | None = None
        self._word_list: set[str] = set()
        
        # Load custom word list if provided
        if self.config.custom_wordlist_path:
            self._load_wordlist(self.config.custom_wordlist_path)
    
    def _init_backend(self) -> FilterBackend:
        """Initialize the best available backend."""
        if self._backend is not None:
            return self._backend_type
        
        backends_to_try = [
            self.config.preferred_backend,
            FilterBackend.BETTER_PROFANITY,
            FilterBackend.PROFANITY_FILTER,
            FilterBackend.ALT_PROFANITY,
            FilterBackend.SIMPLE
        ]
        
        for backend in backends_to_try:
            try:
                if backend == FilterBackend.BETTER_PROFANITY:
                    if self._init_better_profanity():
                        return FilterBackend.BETTER_PROFANITY
                        
                elif backend == FilterBackend.PROFANITY_FILTER:
                    if self._init_profanity_filter():
                        return FilterBackend.PROFANITY_FILTER
                        
                elif backend == FilterBackend.ALT_PROFANITY:
                    if self._init_alt_profanity():
                        return FilterBackend.ALT_PROFANITY
                        
                elif backend == FilterBackend.SIMPLE:
                    self._init_simple()
                    return FilterBackend.SIMPLE
                    
            except Exception as e:
                logger.debug(f"Backend {backend.name} failed: {e}")
                continue
        
        # Fallback to simple
        self._init_simple()
        return FilterBackend.SIMPLE
    
    def _init_better_profanity(self) -> bool:
        """Initialize better-profanity backend."""
        try:
            from better_profanity import profanity
            
            profanity.load_censor_words()
            
            # Add custom words
            if self.config.additional_words:
                profanity.add_censor_words(list(self.config.additional_words))
            
            self._backend = profanity
            self._backend_type = FilterBackend.BETTER_PROFANITY
            logger.info("Using better-profanity for profanity filtering")
            return True
            
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"better-profanity init failed: {e}")
            return False
    
    def _init_profanity_filter(self) -> bool:
        """Initialize profanity-filter backend."""
        try:
            from profanity_filter import ProfanityFilter as PF
            
            pf = PF()
            self._backend = pf
            self._backend_type = FilterBackend.PROFANITY_FILTER
            logger.info("Using profanity-filter for profanity filtering")
            return True
            
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"profanity-filter init failed: {e}")
            return False
    
    def _init_alt_profanity(self) -> bool:
        """Initialize alt-profanity-check backend."""
        try:
            from alt_profanity_check import predict, predict_prob
            
            self._backend = (predict, predict_prob)
            self._backend_type = FilterBackend.ALT_PROFANITY
            logger.info("Using alt-profanity-check for profanity filtering")
            return True
            
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"alt-profanity-check init failed: {e}")
            return False
    
    def _init_simple(self) -> None:
        """Initialize simple built-in backend."""
        self._word_list = self._BUILTIN_WORDS.copy()
        self._word_list.update(self.config.additional_words)
        self._word_list -= self.config.allowed_words
        
        self._backend = "simple"
        self._backend_type = FilterBackend.SIMPLE
        logger.info("Using built-in word list for profanity filtering")
    
    def _load_wordlist(self, path: Path) -> None:
        """Load custom word list from file."""
        try:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word and not word.startswith('#'):
                        self._word_list.add(word)
            logger.info(f"Loaded {len(self._word_list)} words from {path}")
        except Exception as e:
            logger.warning(f"Failed to load word list: {e}")
    
    def _decode_leetspeak(self, text: str) -> str:
        """Convert leetspeak to regular text."""
        if not self.config.detect_leetspeak:
            return text
        
        result = text.lower()
        for leet, normal in self._LEETSPEAK_MAP.items():
            result = result.replace(leet, normal)
        
        return result
    
    def _decode_obfuscation(self, text: str) -> str:
        """Remove common obfuscation patterns."""
        if not self.config.detect_obfuscation:
            return text
        
        # Remove separators between letters
        result = re.sub(r'(\w)[.\-_*]+(?=\w)', r'\1', text)
        
        # Remove repeated letters (fuuuuck -> fuck)
        result = re.sub(r'(.)\1{2,}', r'\1', result)
        
        return result
    
    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        result = text
        
        if not self.config.case_sensitive:
            result = result.lower()
        
        result = self._decode_leetspeak(result)
        result = self._decode_obfuscation(result)
        
        return result
    
    def contains_profanity(self, text: str) -> bool:
        """Check if text contains profanity."""
        if not self.config.enabled or not text:
            return False
        
        backend = self._init_backend()
        
        if backend == FilterBackend.BETTER_PROFANITY:
            return self._backend.contains_profanity(text)
            
        elif backend == FilterBackend.PROFANITY_FILTER:
            return self._backend.is_profane(text)
            
        elif backend == FilterBackend.ALT_PROFANITY:
            predict, _ = self._backend
            return predict([text])[0] == 1
            
        else:
            return self._contains_profanity_simple(text)
    
    def _contains_profanity_simple(self, text: str) -> bool:
        """Check using simple word list."""
        normalized = self._normalize(text)
        words = re.findall(r'\b\w+\b', normalized)
        
        for word in words:
            if word in self._word_list:
                return True
            
            # Check partial matches
            if self.config.include_partial_matches:
                for bad_word in self._word_list:
                    if bad_word in word:
                        return True
        
        return False
    
    def detect(self, text: str) -> list[str]:
        """Detect and return all profane words in text."""
        if not self.config.enabled or not text:
            return []
        
        backend = self._init_backend()
        
        # Most backends don't have a direct detect method, so we use simple
        normalized = self._normalize(text)
        words = re.findall(r'\b\w+\b', normalized)
        
        detected = []
        for word in words:
            if self._is_profane_word(word):
                # Find original case version
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    detected.append(match.group())
                else:
                    detected.append(word)
        
        return detected
    
    def _is_profane_word(self, word: str) -> bool:
        """Check if a single word is profane."""
        normalized = self._normalize(word)
        
        # Check allowed list first
        if normalized in self.config.allowed_words:
            return False
        
        # Direct match
        if normalized in self._word_list:
            return True
        
        # Partial match
        if self.config.include_partial_matches:
            for bad_word in self._word_list:
                if bad_word in normalized:
                    return True
        
        return False
    
    def censor(
        self,
        text: str,
        style: CensorStyle | None = None,
        replacement: str | None = None
    ) -> str:
        """
        Censor profanity in text.
        
        Args:
            text: Input text
            style: Censoring style (overrides config)
            replacement: Custom replacement (for CUSTOM style)
            
        Returns:
            Censored text
        """
        if not self.config.enabled or not text:
            return text
        
        backend = self._init_backend()
        style = style or self.config.censor_style
        replacement = replacement or self.config.custom_replacement
        
        # Use backend censoring if available
        if backend == FilterBackend.BETTER_PROFANITY:
            if style == CensorStyle.ASTERISKS:
                return self._backend.censor(text, '*')
            elif style == CensorStyle.GRAWLIX:
                return self._backend.censor(text, '@')
                
        elif backend == FilterBackend.PROFANITY_FILTER:
            return self._backend.censor(text)
        
        # Custom censoring for all styles
        return self._censor_simple(text, style, replacement)
    
    def _censor_simple(
        self,
        text: str,
        style: CensorStyle,
        replacement: str
    ) -> str:
        """Censor using simple word replacement."""
        result = text
        
        # Find all profane words with their positions
        normalized = self._normalize(text)
        
        # Build pattern for all words
        words_to_find = []
        for word in self._word_list:
            if word not in self.config.allowed_words:
                words_to_find.append(re.escape(word))
        
        if not words_to_find:
            return text
        
        pattern = re.compile(
            r'\b(' + '|'.join(words_to_find) + r')\b',
            re.IGNORECASE
        )
        
        def replace_func(match):
            word = match.group()
            return self._get_replacement(word, style, replacement)
        
        result = pattern.sub(replace_func, result)
        
        return result
    
    def _get_replacement(
        self,
        word: str,
        style: CensorStyle,
        custom: str
    ) -> str:
        """Get replacement string for a word."""
        if style == CensorStyle.ASTERISKS:
            return '*' * len(word)
            
        elif style == CensorStyle.GRAWLIX:
            grawlix = '@#$%!&*'
            return ''.join(grawlix[i % len(grawlix)] for i in range(len(word)))
            
        elif style == CensorStyle.FIRST_LAST:
            if len(word) <= 2:
                return '*' * len(word)
            return word[0] + '*' * (len(word) - 2) + word[-1]
            
        elif style == CensorStyle.FULL_WORD:
            return custom
            
        elif style == CensorStyle.BEEP:
            return "[BEEP]"
            
        elif style == CensorStyle.REMOVE:
            return ""
            
        elif style == CensorStyle.CUSTOM:
            return custom
            
        return '*' * len(word)
    
    def filter_text(
        self,
        text: str,
        style: CensorStyle | None = None,
        replacement: str | None = None
    ) -> FilterResult:
        """
        Filter text with full result details.
        
        Returns FilterResult with censored text and metadata.
        """
        if not self.config.enabled or not text:
            return FilterResult(
                original=text,
                filtered=text,
                contains_profanity=False
            )
        
        detected = self.detect(text)
        has_profanity = len(detected) > 0
        filtered = self.censor(text, style, replacement)
        
        # Find positions
        positions = []
        for word in detected:
            for match in re.finditer(re.escape(word), text, re.IGNORECASE):
                positions.append((match.start(), match.end()))
        
        return FilterResult(
            original=text,
            filtered=filtered,
            contains_profanity=has_profanity,
            detected_words=detected,
            positions=positions
        )
    
    def add_words(self, words: list[str]) -> None:
        """Add words to the filter."""
        normalized = {w.lower() for w in words}
        self.config.additional_words.update(normalized)
        
        # Update backend if initialized
        if self._backend_type == FilterBackend.BETTER_PROFANITY:
            self._backend.add_censor_words(list(normalized))
        elif self._backend_type == FilterBackend.SIMPLE:
            self._word_list.update(normalized)
    
    def remove_words(self, words: list[str]) -> None:
        """Remove words from the filter (allow them)."""
        normalized = {w.lower() for w in words}
        self.config.allowed_words.update(normalized)
        
        if self._backend_type == FilterBackend.SIMPLE:
            self._word_list -= normalized
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable filtering."""
        self.config.enabled = enabled
    
    def get_word_count(self) -> int:
        """Get number of words in filter."""
        if self._backend_type == FilterBackend.SIMPLE:
            return len(self._word_list)
        return -1  # Unknown for other backends


# Singleton instance
_filter_instance: ProfanityFilter | None = None


def get_profanity_filter(
    config: FilterConfig | None = None
) -> ProfanityFilter:
    """Get or create a singleton filter instance."""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = ProfanityFilter(config)
    return _filter_instance


# Convenience functions
def contains_profanity(text: str) -> bool:
    """Quick check for profanity."""
    return get_profanity_filter().contains_profanity(text)


def censor_profanity(text: str, replacement: str = "***") -> str:
    """Quick censoring of profanity."""
    filter_obj = get_profanity_filter()
    return filter_obj.censor(text, CensorStyle.CUSTOM, replacement)


def filter_text(text: str) -> FilterResult:
    """Quick filtering with full result."""
    return get_profanity_filter().filter_text(text)
