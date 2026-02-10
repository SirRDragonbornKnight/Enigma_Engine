"""
================================================================================
Punctuation Restoration - Add punctuation to transcribed text.
================================================================================

Multi-backend punctuation restoration for ASR output:
- deepmultilingualpunctuation: Neural punctuation model
- punctuators: Fast rule-based + transformer
- nemo: NVIDIA NeMo punctuation/capitalization
- Simple: Rule-based heuristics (fallback)

USAGE:
    from enigma_engine.voice.punctuation_restoration import PunctuationRestorer
    
    restorer = PunctuationRestorer()
    
    # Restore punctuation
    text = "hello how are you im doing fine thanks"
    result = restorer.restore(text)
    print(result)  # "Hello, how are you? I'm doing fine, thanks."
    
    # Batch processing
    texts = ["first sentence", "second sentence"]
    results = restorer.restore_batch(texts)
    
    # With confidence scores
    result = restorer.restore(text, return_confidence=True)
    print(result.text, result.confidence)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class PunctuationBackend(Enum):
    """Available punctuation restoration backends."""
    
    DEEP_PUNCTUATION = auto()    # deepmultilingualpunctuation
    PUNCTUATORS = auto()         # punctuators library
    NEMO = auto()                # NVIDIA NeMo
    TRANSFORMERS = auto()        # HuggingFace transformers
    SIMPLE = auto()              # Rule-based fallback


@dataclass
class PunctuationConfig:
    """Configuration for punctuation restoration."""
    
    # Backend selection
    preferred_backend: PunctuationBackend = PunctuationBackend.DEEP_PUNCTUATION
    
    # Language settings
    language: str = "en"
    
    # Restoration settings
    restore_capitalization: bool = True
    restore_punctuation: bool = True
    
    # Punctuation types to restore
    restore_periods: bool = True
    restore_commas: bool = True
    restore_questions: bool = True
    restore_exclamations: bool = True
    restore_colons: bool = True
    restore_semicolons: bool = True
    
    # Advanced settings
    min_confidence: float = 0.5    # Minimum confidence for punctuation
    batch_size: int = 8            # Batch size for processing
    max_length: int = 512          # Max text length per call


@dataclass
class PunctuationResult:
    """Result of punctuation restoration."""
    
    text: str                            # Restored text
    original: str                        # Original text
    confidence: float = 1.0              # Overall confidence
    punctuation_added: int = 0           # Number of punctuation marks added
    capitalizations_fixed: int = 0       # Number of capitalizations fixed
    
    @property
    def changes_made(self) -> int:
        """Total changes made."""
        return self.punctuation_added + self.capitalizations_fixed


class PunctuationRestorer:
    """Multi-backend punctuation restoration system."""
    
    def __init__(self, config: PunctuationConfig | None = None):
        """Initialize restorer with config."""
        self.config = config or PunctuationConfig()
        self._backend: Any | None = None
        self._backend_type: PunctuationBackend | None = None
        
        # Common contractions for rule-based restoration
        self._contractions = {
            "im": "I'm",
            "ive": "I've",
            "id": "I'd",
            "ill": "I'll",
            "youre": "you're",
            "youve": "you've",
            "youd": "you'd",
            "youll": "you'll",
            "hes": "he's",
            "shes": "she's",
            "its": "it's",
            "weve": "we've",
            "wed": "we'd",
            "well": "we'll",
            "theyre": "they're",
            "theyve": "they've",
            "theyd": "they'd",
            "theyll": "they'll",
            "cant": "can't",
            "couldnt": "couldn't",
            "wouldnt": "wouldn't",
            "shouldnt": "shouldn't",
            "dont": "don't",
            "doesnt": "doesn't",
            "didnt": "didn't",
            "isnt": "isn't",
            "arent": "aren't",
            "wasnt": "wasn't",
            "werent": "weren't",
            "wont": "won't",
            "havent": "haven't",
            "hasnt": "hasn't",
            "hadnt": "hadn't",
            "thats": "that's",
            "whats": "what's",
            "whos": "who's",
            "wheres": "where's",
            "hows": "how's",
            "lets": "let's",
        }
        
        # Question words for heuristic detection
        self._question_words = {
            "who", "what", "when", "where", "why", "how",
            "which", "whose", "whom", "can", "could", "would",
            "should", "will", "do", "does", "did", "is", "are",
            "was", "were", "have", "has", "had", "may", "might"
        }
    
    def _init_backend(self) -> PunctuationBackend:
        """Initialize the best available backend."""
        if self._backend is not None:
            return self._backend_type
        
        backends_to_try = [
            self.config.preferred_backend,
            PunctuationBackend.DEEP_PUNCTUATION,
            PunctuationBackend.PUNCTUATORS,
            PunctuationBackend.NEMO,
            PunctuationBackend.TRANSFORMERS,
            PunctuationBackend.SIMPLE
        ]
        
        for backend in backends_to_try:
            try:
                if backend == PunctuationBackend.DEEP_PUNCTUATION:
                    if self._init_deep_punctuation():
                        return PunctuationBackend.DEEP_PUNCTUATION
                        
                elif backend == PunctuationBackend.PUNCTUATORS:
                    if self._init_punctuators():
                        return PunctuationBackend.PUNCTUATORS
                        
                elif backend == PunctuationBackend.NEMO:
                    if self._init_nemo():
                        return PunctuationBackend.NEMO
                        
                elif backend == PunctuationBackend.TRANSFORMERS:
                    if self._init_transformers():
                        return PunctuationBackend.TRANSFORMERS
                        
                elif backend == PunctuationBackend.SIMPLE:
                    self._backend = "simple"
                    self._backend_type = PunctuationBackend.SIMPLE
                    logger.info("Using simple rule-based punctuation restoration")
                    return PunctuationBackend.SIMPLE
                    
            except Exception as e:
                logger.debug(f"Backend {backend.name} failed: {e}")
                continue
        
        # Fallback to simple
        self._backend = "simple"
        self._backend_type = PunctuationBackend.SIMPLE
        return PunctuationBackend.SIMPLE
    
    def _init_deep_punctuation(self) -> bool:
        """Initialize deepmultilingualpunctuation backend."""
        try:
            from deepmultilingualpunctuation import PunctuationModel
            
            self._backend = PunctuationModel()
            self._backend_type = PunctuationBackend.DEEP_PUNCTUATION
            logger.info("Using deepmultilingualpunctuation for punctuation restoration")
            return True
            
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"deepmultilingualpunctuation init failed: {e}")
            return False
    
    def _init_punctuators(self) -> bool:
        """Initialize punctuators backend."""
        try:
            from punctuators.models import PunctCapSegModelONNX
            
            model = PunctCapSegModelONNX.from_pretrained(
                "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
            )
            self._backend = model
            self._backend_type = PunctuationBackend.PUNCTUATORS
            logger.info("Using punctuators for punctuation restoration")
            return True
            
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"punctuators init failed: {e}")
            return False
    
    def _init_nemo(self) -> bool:
        """Initialize NVIDIA NeMo backend."""
        try:
            from nemo.collections.nlp.models import PunctuationCapitalizationModel
            
            model = PunctuationCapitalizationModel.from_pretrained(
                "punctuation_en_distilbert"
            )
            self._backend = model
            self._backend_type = PunctuationBackend.NEMO
            logger.info("Using NeMo for punctuation restoration")
            return True
            
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"NeMo init failed: {e}")
            return False
    
    def _init_transformers(self) -> bool:
        """Initialize HuggingFace transformers backend."""
        try:
            from transformers import pipeline
            
            self._backend = pipeline(
                "token-classification",
                model="oliverguhr/fullstop-punctuation-multilang-large",
                aggregation_strategy="simple"
            )
            self._backend_type = PunctuationBackend.TRANSFORMERS
            logger.info("Using transformers for punctuation restoration")
            return True
            
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"transformers init failed: {e}")
            return False
    
    def restore(
        self,
        text: str,
        return_confidence: bool = False
    ) -> PunctuationResult:
        """
        Restore punctuation to text.
        
        Args:
            text: Input text without punctuation
            return_confidence: Include per-token confidence
            
        Returns:
            PunctuationResult with restored text
        """
        backend = self._init_backend()
        original = text
        
        if not text or not text.strip():
            return PunctuationResult(text="", original=original)
        
        # Process based on backend
        if backend == PunctuationBackend.DEEP_PUNCTUATION:
            result_text, confidence = self._restore_deep_punctuation(text)
        elif backend == PunctuationBackend.PUNCTUATORS:
            result_text, confidence = self._restore_punctuators(text)
        elif backend == PunctuationBackend.NEMO:
            result_text, confidence = self._restore_nemo(text)
        elif backend == PunctuationBackend.TRANSFORMERS:
            result_text, confidence = self._restore_transformers(text)
        else:
            result_text, confidence = self._restore_simple(text)
        
        # Count changes
        punct_added = sum(1 for c in result_text if c in '.,!?;:' and c not in original)
        caps_fixed = sum(1 for a, b in zip(original.lower(), result_text.lower()) 
                        if a != b and a.isalpha())
        
        return PunctuationResult(
            text=result_text,
            original=original,
            confidence=confidence,
            punctuation_added=punct_added,
            capitalizations_fixed=caps_fixed
        )
    
    def restore_batch(self, texts: list[str]) -> list[PunctuationResult]:
        """Restore punctuation for multiple texts."""
        backend = self._init_backend()
        
        # Some backends support native batching
        if backend == PunctuationBackend.DEEP_PUNCTUATION:
            restored = self._backend.restore_punctuation(texts)
            if isinstance(restored, str):
                restored = [restored]
            return [
                PunctuationResult(text=r, original=t)
                for r, t in zip(restored, texts)
            ]
        
        # Fall back to sequential processing
        return [self.restore(text) for text in texts]
    
    def _restore_deep_punctuation(self, text: str) -> tuple[str, float]:
        """Restore using deepmultilingualpunctuation."""
        result = self._backend.restore_punctuation(text)
        return result, 0.9  # This library doesn't return confidence
    
    def _restore_punctuators(self, text: str) -> tuple[str, float]:
        """Restore using punctuators."""
        results = self._backend.infer([text])
        if results and results[0]:
            # punctuators returns list of segments
            restored = " ".join(results[0])
            return restored, 0.85
        return text, 0.0
    
    def _restore_nemo(self, text: str) -> tuple[str, float]:
        """Restore using NeMo."""
        result = self._backend.add_punctuation_capitalization([text])
        if result:
            return result[0], 0.9
        return text, 0.0
    
    def _restore_transformers(self, text: str) -> tuple[str, float]:
        """Restore using transformers pipeline."""
        # Split into chunks if too long
        words = text.split()
        result_words = []
        confidences = []
        
        chunk_size = 100  # words per chunk
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            predictions = self._backend(chunk)
            
            # Reconstruct text with punctuation
            result_text = chunk
            for pred in sorted(predictions, key=lambda x: x['end'], reverse=True):
                label = pred['entity_group']
                conf = pred['score']
                end = pred['end']
                
                # Map labels to punctuation
                punct_map = {
                    '.': '.',
                    ',': ',',
                    '?': '?',
                    ':': ':',
                    '-': '-',
                    'PERIOD': '.',
                    'COMMA': ',',
                    'QUESTION': '?',
                    'COLON': ':',
                }
                
                if label in punct_map:
                    punctuation = punct_map[label]
                    # Insert punctuation after the word
                    result_text = result_text[:end] + punctuation + result_text[end:]
                    confidences.append(conf)
            
            result_words.append(result_text)
        
        final_text = " ".join(result_words)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.7
        
        # Capitalize first letter and after sentence endings
        final_text = self._capitalize_sentences(final_text)
        
        return final_text, avg_confidence
    
    def _restore_simple(self, text: str) -> tuple[str, float]:
        """Simple rule-based punctuation restoration."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        if not text:
            return "", 0.0
        
        words = text.split()
        result_words = []
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Fix contractions
            if word_lower in self._contractions:
                word = self._contractions[word_lower]
            
            # Add word
            result_words.append(word)
            
            # Check for sentence boundary (rough heuristics)
            is_last = (i == len(words) - 1)
            next_word = words[i + 1].lower() if i + 1 < len(words) else None
            
            # Question detection
            if (i > 0 and words[0].lower() in self._question_words and is_last):
                result_words[-1] = word + "?"
            # Statement ending
            elif is_last and not word.endswith(('.', '!', '?', ',')):
                result_words[-1] = word + "."
            # Comma before conjunctions in longer sentences
            elif (next_word in ('but', 'yet', 'so') and i > 3):
                result_words[-1] = word + ","
            # Comma in lists (detecting "and" or "or" patterns)
            elif (next_word == 'and' and i > 0 and 
                  any(w.lower() == 'and' for w in words[:i])):
                if not word.endswith(','):
                    result_words[-1] = word + ","
        
        result_text = " ".join(result_words)
        result_text = self._capitalize_sentences(result_text)
        
        return result_text, 0.6
    
    def _capitalize_sentences(self, text: str) -> str:
        """Capitalize first letter of sentences."""
        if not text:
            return text
        
        # Capitalize first character
        result = text[0].upper() + text[1:] if len(text) > 0 else text
        
        # Capitalize after sentence endings
        result = re.sub(
            r'([.!?]\s+)([a-z])',
            lambda m: m.group(1) + m.group(2).upper(),
            result
        )
        
        # Capitalize "I" as standalone word
        result = re.sub(r'\bi\b', 'I', result)
        
        return result


# Singleton instance
_restorer_instance: PunctuationRestorer | None = None


def get_punctuation_restorer(
    config: PunctuationConfig | None = None
) -> PunctuationRestorer:
    """Get or create a singleton restorer instance."""
    global _restorer_instance
    if _restorer_instance is None:
        _restorer_instance = PunctuationRestorer(config)
    return _restorer_instance


# Convenience function
def restore_punctuation(text: str) -> str:
    """Quick punctuation restoration."""
    restorer = get_punctuation_restorer()
    result = restorer.restore(text)
    return result.text


def restore_punctuation_batch(texts: list[str]) -> list[str]:
    """Quick batch punctuation restoration."""
    restorer = get_punctuation_restorer()
    results = restorer.restore_batch(texts)
    return [r.text for r in results]
