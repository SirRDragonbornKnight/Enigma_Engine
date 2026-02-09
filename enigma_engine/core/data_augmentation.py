"""
Synthetic Data Augmentation for Enigma AI Engine

Generate augmented training data.

Features:
- Text paraphrasing
- Back translation
- Word substitution
- Sentence shuffling
- Template variation
- Noise injection

Usage:
    from enigma_engine.core.data_augmentation import DataAugmenter, AugmentConfig
    
    augmenter = DataAugmenter()
    
    # Augment single text
    variants = augmenter.augment("Hello, how are you?", n=5)
    
    # Augment dataset
    augmented = augmenter.augment_dataset(dataset, multiplier=3)
"""

import logging
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AugmentationType(Enum):
    """Types of data augmentation."""
    SYNONYM = "synonym"  # Replace with synonyms
    PARAPHRASE = "paraphrase"  # Rewrite sentences
    BACK_TRANSLATE = "back_translate"  # Translate and back
    NOISE = "noise"  # Add random noise
    SHUFFLE = "shuffle"  # Shuffle sentences/words
    DELETE = "delete"  # Random deletion
    INSERT = "insert"  # Random insertion
    SWAP = "swap"  # Swap adjacent words
    TEMPLATE = "template"  # Template-based variation


@dataclass
class AugmentConfig:
    """Configuration for data augmentation."""
    methods: List[AugmentationType] = field(default_factory=lambda: [
        AugmentationType.SYNONYM,
        AugmentationType.SWAP,
        AugmentationType.DELETE
    ])
    
    # Probabilities
    word_replace_prob: float = 0.1  # Probability to replace a word
    word_delete_prob: float = 0.1  # Probability to delete a word
    word_swap_prob: float = 0.1  # Probability to swap words
    word_insert_prob: float = 0.05  # Probability to insert a word
    
    # Constraints
    min_words: int = 3  # Minimum words after augmentation
    max_attempts: int = 10  # Max attempts to generate valid augmentation
    preserve_entities: bool = True  # Try to preserve named entities


# Simple synonym database (expandable)
SYNONYMS = {
    "good": ["great", "excellent", "fine", "nice", "wonderful"],
    "bad": ["poor", "terrible", "awful", "horrible", "dreadful"],
    "big": ["large", "huge", "enormous", "massive", "giant"],
    "small": ["tiny", "little", "miniature", "compact", "minute"],
    "happy": ["joyful", "pleased", "delighted", "cheerful", "content"],
    "sad": ["unhappy", "sorrowful", "gloomy", "melancholy", "dejected"],
    "fast": ["quick", "rapid", "speedy", "swift", "hasty"],
    "slow": ["sluggish", "gradual", "leisurely", "unhurried"],
    "beautiful": ["pretty", "lovely", "gorgeous", "stunning", "attractive"],
    "ugly": ["unattractive", "unsightly", "hideous", "grotesque"],
    "smart": ["intelligent", "clever", "bright", "brilliant", "wise"],
    "stupid": ["foolish", "dumb", "idiotic", "ignorant"],
    "strong": ["powerful", "mighty", "robust", "sturdy", "tough"],
    "weak": ["feeble", "frail", "fragile", "delicate"],
    "hot": ["warm", "heated", "burning", "scorching"],
    "cold": ["cool", "chilly", "freezing", "icy", "frigid"],
    "new": ["fresh", "recent", "modern", "novel"],
    "old": ["ancient", "aged", "elderly", "vintage"],
    "easy": ["simple", "straightforward", "effortless", "basic"],
    "hard": ["difficult", "challenging", "tough", "complex"],
    "important": ["significant", "crucial", "vital", "essential"],
    "interesting": ["fascinating", "intriguing", "engaging", "captivating"],
    "like": ["enjoy", "love", "appreciate", "prefer"],
    "want": ["desire", "wish", "need", "crave"],
    "make": ["create", "produce", "build", "construct"],
    "get": ["obtain", "acquire", "receive", "gain"],
    "go": ["move", "travel", "proceed", "head"],
    "come": ["arrive", "approach", "reach"],
    "see": ["observe", "view", "notice", "spot"],
    "know": ["understand", "realize", "recognize", "comprehend"],
    "think": ["believe", "consider", "assume", "suppose"],
    "say": ["state", "mention", "declare", "express"],
    "tell": ["inform", "explain", "describe", "narrate"],
    "ask": ["inquire", "question", "query", "request"],
    "use": ["utilize", "employ", "apply", "implement"],
    "find": ["discover", "locate", "detect", "uncover"],
    "give": ["provide", "offer", "grant", "supply"],
    "take": ["grab", "seize", "accept", "receive"],
    "help": ["assist", "aid", "support", "facilitate"],
    "try": ["attempt", "endeavor", "strive"],
    "work": ["function", "operate", "perform"],
    "start": ["begin", "commence", "initiate", "launch"],
    "stop": ["cease", "halt", "end", "terminate"],
}

# Filler words for insertion
FILLER_WORDS = [
    "actually", "basically", "really", "simply", "just",
    "perhaps", "maybe", "probably", "certainly", "definitely"
]


class DataAugmenter:
    """Data augmentation for text."""
    
    def __init__(
        self,
        config: Optional[AugmentConfig] = None,
        custom_synonyms: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize data augmenter.
        
        Args:
            config: Augmentation configuration
            custom_synonyms: Additional synonyms
        """
        self.config = config or AugmentConfig()
        
        # Build synonym dictionary
        self.synonyms = dict(SYNONYMS)
        if custom_synonyms:
            self.synonyms.update(custom_synonyms)
        
        # Reverse mapping for efficiency
        self._reverse_synonyms = {}
        for word, syns in self.synonyms.items():
            for syn in syns:
                if syn not in self._reverse_synonyms:
                    self._reverse_synonyms[syn] = []
                self._reverse_synonyms[syn].append(word)
        
        logger.info(
            f"DataAugmenter initialized with {len(self.config.methods)} methods"
        )
    
    def augment(
        self,
        text: str,
        n: int = 1,
        methods: Optional[List[AugmentationType]] = None
    ) -> List[str]:
        """
        Generate augmented variants of text.
        
        Args:
            text: Input text
            n: Number of variants to generate
            methods: Specific methods to use (default: all configured)
            
        Returns:
            List of augmented texts
        """
        methods = methods or self.config.methods
        variants = []
        
        for _ in range(n):
            # Pick random method
            method = random.choice(methods)
            
            # Try to generate valid augmentation
            for attempt in range(self.config.max_attempts):
                try:
                    augmented = self._apply_method(text, method)
                    
                    # Validate
                    if augmented and augmented != text:
                        words = augmented.split()
                        if len(words) >= self.config.min_words:
                            variants.append(augmented)
                            break
                except Exception as e:
                    logger.debug(f"Augmentation attempt failed: {e}")
            
            # Fallback: return original with minor changes
            if len(variants) <= _ - 1:
                variants.append(text)
        
        return variants
    
    def augment_dataset(
        self,
        dataset: List[Dict],
        text_key: str = "text",
        multiplier: int = 2,
        include_original: bool = True
    ) -> List[Dict]:
        """
        Augment a dataset.
        
        Args:
            dataset: List of samples (dicts)
            text_key: Key for text field
            multiplier: How many augmented samples per original
            include_original: Include original samples
            
        Returns:
            Augmented dataset
        """
        augmented = []
        
        for sample in dataset:
            if include_original:
                augmented.append(sample)
            
            text = sample.get(text_key, "")
            if not text:
                continue
            
            variants = self.augment(text, n=multiplier)
            
            for variant in variants:
                new_sample = sample.copy()
                new_sample[text_key] = variant
                new_sample['augmented'] = True
                augmented.append(new_sample)
        
        return augmented
    
    def _apply_method(self, text: str, method: AugmentationType) -> str:
        """Apply specific augmentation method."""
        if method == AugmentationType.SYNONYM:
            return self._synonym_replacement(text)
        elif method == AugmentationType.SWAP:
            return self._word_swap(text)
        elif method == AugmentationType.DELETE:
            return self._random_deletion(text)
        elif method == AugmentationType.INSERT:
            return self._random_insertion(text)
        elif method == AugmentationType.SHUFFLE:
            return self._shuffle_sentences(text)
        elif method == AugmentationType.NOISE:
            return self._add_noise(text)
        elif method == AugmentationType.PARAPHRASE:
            return self._simple_paraphrase(text)
        else:
            return text
    
    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms."""
        words = text.split()
        new_words = []
        
        for word in words:
            # Clean word for lookup
            clean = word.lower().strip('.,!?;:"\'-')
            
            if random.random() < self.config.word_replace_prob:
                if clean in self.synonyms:
                    synonym = random.choice(self.synonyms[clean])
                    # Preserve case
                    if word[0].isupper():
                        synonym = synonym.capitalize()
                    # Preserve punctuation
                    if word[-1] in '.,!?;:':
                        synonym += word[-1]
                    new_words.append(synonym)
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def _word_swap(self, text: str) -> str:
        """Swap adjacent words."""
        words = text.split()
        if len(words) < 2:
            return text
        
        new_words = list(words)
        n_swaps = max(1, int(len(words) * self.config.word_swap_prob))
        
        for _ in range(n_swaps):
            if len(new_words) < 2:
                break
            idx = random.randint(0, len(new_words) - 2)
            new_words[idx], new_words[idx + 1] = new_words[idx + 1], new_words[idx]
        
        return ' '.join(new_words)
    
    def _random_deletion(self, text: str) -> str:
        """Randomly delete words."""
        words = text.split()
        if len(words) <= self.config.min_words:
            return text
        
        new_words = []
        for word in words:
            if random.random() > self.config.word_delete_prob:
                new_words.append(word)
        
        if len(new_words) < self.config.min_words:
            return text
        
        return ' '.join(new_words)
    
    def _random_insertion(self, text: str) -> str:
        """Randomly insert words."""
        words = text.split()
        new_words = []
        
        for word in words:
            new_words.append(word)
            if random.random() < self.config.word_insert_prob:
                filler = random.choice(FILLER_WORDS)
                new_words.append(filler)
        
        return ' '.join(new_words)
    
    def _shuffle_sentences(self, text: str) -> str:
        """Shuffle sentences in text."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) < 2:
            return text
        
        random.shuffle(sentences)
        return ' '.join(sentences)
    
    def _add_noise(self, text: str) -> str:
        """Add character-level noise."""
        chars = list(text)
        new_chars = []
        
        for char in chars:
            if random.random() < 0.02:  # 2% noise rate
                noise_type = random.choice(['delete', 'duplicate', 'swap'])
                if noise_type == 'delete':
                    continue
                elif noise_type == 'duplicate':
                    new_chars.extend([char, char])
                    continue
                elif noise_type == 'swap' and new_chars:
                    new_chars[-1], char = char, new_chars[-1]
            new_chars.append(char)
        
        return ''.join(new_chars)
    
    def _simple_paraphrase(self, text: str) -> str:
        """Simple rule-based paraphrasing."""
        # Apply multiple transformations
        result = text
        
        # Synonym replacement
        result = self._synonym_replacement(result)
        
        # Simple grammatical transformations
        transformations = [
            (r"\bdon't\b", "do not"),
            (r"\bcan't\b", "cannot"),
            (r"\bwon't\b", "will not"),
            (r"\bI'm\b", "I am"),
            (r"\bhe's\b", "he is"),
            (r"\bshe's\b", "she is"),
            (r"\bit's\b", "it is"),
            (r"\bthey're\b", "they are"),
            (r"\bwe're\b", "we are"),
            (r"\bI've\b", "I have"),
            (r"\bthey've\b", "they have"),
            (r"\bI'll\b", "I will"),
        ]
        
        # Random expand/contract
        for pattern, replacement in transformations:
            if random.random() < 0.5:
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            else:
                result = re.sub(replacement, pattern.replace(r'\b', ''), result, flags=re.IGNORECASE)
        
        return result


class TemplateAugmenter:
    """Template-based data augmentation."""
    
    def __init__(self):
        self.templates: Dict[str, List[str]] = {}
    
    def add_template(self, name: str, templates: List[str]):
        """
        Add templates for augmentation.
        
        Templates use {slot} placeholders.
        """
        self.templates[name] = templates
    
    def augment_from_template(
        self,
        template_name: str,
        slots: Dict[str, List[str]],
        n: int = 10
    ) -> List[str]:
        """
        Generate text from template.
        
        Args:
            template_name: Name of template set
            slots: Dict of slot name -> possible values
            n: Number of variants
            
        Returns:
            List of generated texts
        """
        templates = self.templates.get(template_name, [])
        if not templates:
            return []
        
        results = []
        for _ in range(n):
            template = random.choice(templates)
            
            # Fill slots
            text = template
            for slot, values in slots.items():
                value = random.choice(values)
                text = text.replace(f"{{{slot}}}", value)
            
            results.append(text)
        
        return results


class BackTranslator:
    """Back-translation for paraphrasing."""
    
    def __init__(self, translator=None):
        """
        Initialize back translator.
        
        Args:
            translator: Translation function/model
        """
        self.translator = translator
        
        # Supported language pairs for back-translation
        self.language_pairs = [
            ('en', 'de'),  # English <-> German
            ('en', 'fr'),  # English <-> French
            ('en', 'es'),  # English <-> Spanish
        ]
    
    def translate(self, text: str, target_lang: str) -> str:
        """Translate text to target language."""
        if self.translator:
            return self.translator(text, target_lang)
        
        # Mock translation for testing
        return f"[{target_lang}]{text}[/{target_lang}]"
    
    def back_translate(self, text: str, via_lang: str = 'de') -> str:
        """
        Back-translate text.
        
        Args:
            text: Original text
            via_lang: Intermediate language
            
        Returns:
            Back-translated text
        """
        # Translate to intermediate language
        intermediate = self.translate(text, via_lang)
        
        # Translate back to original language
        back = self.translate(intermediate, 'en')
        
        return back
    
    def augment(self, text: str, n: int = 3) -> List[str]:
        """Generate variants via back-translation."""
        results = []
        
        for i in range(n):
            # Use different intermediate language each time
            via_lang = self.language_pairs[i % len(self.language_pairs)][1]
            variant = self.back_translate(text, via_lang)
            results.append(variant)
        
        return results


def augment_qa_dataset(
    questions: List[str],
    answers: List[str],
    augmenter: DataAugmenter,
    augment_questions: bool = True,
    augment_answers: bool = False,
    multiplier: int = 2
) -> Tuple[List[str], List[str]]:
    """
    Augment a QA dataset.
    
    Args:
        questions: List of questions
        answers: List of answers
        augmenter: DataAugmenter instance
        augment_questions: Whether to augment questions
        augment_answers: Whether to augment answers
        multiplier: Augmentation multiplier
        
    Returns:
        Tuple of (augmented_questions, augmented_answers)
    """
    aug_questions = []
    aug_answers = []
    
    for q, a in zip(questions, answers):
        # Original
        aug_questions.append(q)
        aug_answers.append(a)
        
        # Augmented
        for _ in range(multiplier):
            new_q = augmenter.augment(q, n=1)[0] if augment_questions else q
            new_a = augmenter.augment(a, n=1)[0] if augment_answers else a
            aug_questions.append(new_q)
            aug_answers.append(new_a)
    
    return aug_questions, aug_answers
