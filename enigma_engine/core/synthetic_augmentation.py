"""
Synthetic Data Augmentation for Enigma AI Engine

Automatically paraphrase and vary training examples.

Features:
- Paraphrase generation
- Synonym replacement
- Sentence reordering
- Back-translation
- Noise injection
- Example mixing

Usage:
    from enigma_engine.core.synthetic_augmentation import DataAugmentor
    
    augmentor = DataAugmentor()
    
    # Augment a single example
    augmented = augmentor.augment("What is the capital of France?")
    
    # Augment training dataset
    augmented_data = augmentor.augment_dataset(training_data, multiplier=3)
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)


# Common synonyms for data augmentation
SYNONYM_MAP = {
    "big": ["large", "huge", "massive", "enormous", "giant"],
    "small": ["tiny", "little", "miniature", "compact", "petite"],
    "good": ["great", "excellent", "wonderful", "fantastic", "superb"],
    "bad": ["poor", "terrible", "awful", "horrible", "dreadful"],
    "fast": ["quick", "rapid", "speedy", "swift", "hasty"],
    "slow": ["sluggish", "leisurely", "gradual", "unhurried"],
    "happy": ["joyful", "pleased", "delighted", "cheerful", "content"],
    "sad": ["unhappy", "sorrowful", "melancholy", "dejected", "gloomy"],
    "help": ["assist", "aid", "support", "guide", "facilitate"],
    "make": ["create", "build", "construct", "produce", "generate"],
    "get": ["obtain", "acquire", "receive", "fetch", "retrieve"],
    "show": ["display", "present", "demonstrate", "reveal", "exhibit"],
    "tell": ["inform", "explain", "describe", "relate", "communicate"],
    "think": ["believe", "consider", "suppose", "assume", "reckon"],
    "want": ["desire", "wish", "would like", "need", "require"],
    "use": ["utilize", "employ", "apply", "leverage", "make use of"],
    "find": ["discover", "locate", "identify", "detect", "uncover"],
    "give": ["provide", "offer", "supply", "grant", "deliver"],
    "take": ["grab", "seize", "capture", "accept", "acquire"],
}

# Question starters for variation
QUESTION_STARTERS = [
    "What", "How", "Why", "When", "Where", "Who",
    "Can you", "Could you", "Would you", "Will you",
    "Please", "I need to", "I want to", "Help me",
]


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    # Augmentation methods
    enable_synonym: bool = True
    enable_paraphrase: bool = True
    enable_reorder: bool = True
    enable_noise: bool = True
    enable_back_translation: bool = False  # Requires API
    
    # Parameters
    synonym_probability: float = 0.3
    noise_probability: float = 0.1
    reorder_probability: float = 0.2
    
    # Output
    target_multiplier: int = 3
    max_examples: int = 100000
    preserve_original: bool = True
    
    # Quality
    min_length: int = 10
    max_length: int = 2048


class TextAugmentor:
    """Text augmentation utilities."""
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self._random = random.Random()
    
    def augment(self, text: str, methods: Optional[List[str]] = None) -> List[str]:
        """
        Augment a single text.
        
        Args:
            text: Input text
            methods: Specific methods to use (all if None)
            
        Returns:
            List of augmented texts
        """
        results = []
        
        if methods is None:
            methods = []
            if self.config.enable_synonym:
                methods.append("synonym")
            if self.config.enable_paraphrase:
                methods.append("paraphrase")
            if self.config.enable_reorder:
                methods.append("reorder")
            if self.config.enable_noise:
                methods.append("noise")
        
        for method in methods:
            if method == "synonym":
                augmented = self._synonym_replacement(text)
            elif method == "paraphrase":
                augmented = self._paraphrase(text)
            elif method == "reorder":
                augmented = self._reorder_sentences(text)
            elif method == "noise":
                augmented = self._inject_noise(text)
            else:
                continue
            
            if augmented and augmented != text:
                results.append(augmented)
        
        return results
    
    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms."""
        words = text.split()
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?":;')
            
            if word_lower in SYNONYM_MAP and self._random.random() < self.config.synonym_probability:
                synonyms = SYNONYM_MAP[word_lower]
                replacement = self._random.choice(synonyms)
                
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                
                words[i] = word.replace(word_lower, replacement)
        
        return ' '.join(words)
    
    def _paraphrase(self, text: str) -> str:
        """Simple paraphrasing using templates."""
        # Question paraphrasing
        if text.endswith('?'):
            return self._paraphrase_question(text)
        
        # Statement paraphrasing
        return self._paraphrase_statement(text)
    
    def _paraphrase_question(self, text: str) -> str:
        """Paraphrase a question."""
        # Remove question mark for processing
        text = text.rstrip('?')
        
        # Common question patterns
        patterns = [
            (r'^What is (.+)$', ["Tell me about {}", "Explain {}", "Describe {}"]),
            (r'^How do I (.+)$', ["Help me {}", "I need to {}", "Assist me to {}"]),
            (r'^Can you (.+)$', ["Would you {}", "Please {}", "Could you {}"]),
            (r'^Why (.+)$', ["Explain why {}", "Tell me the reason {}"]),
        ]
        
        for pattern, templates in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                template = self._random.choice(templates)
                return template.format(match.group(1)) + '?'
        
        # Generic paraphrase
        starters = ["Can you tell me", "I want to know", "Please explain"]
        starter = self._random.choice(starters)
        return f"{starter} {text.lower()}?"
    
    def _paraphrase_statement(self, text: str) -> str:
        """Paraphrase a statement."""
        # Simple word swaps
        swaps = [
            ("is a", "represents a"),
            ("is the", "serves as the"),
            ("can be", "may be"),
            ("will", "shall"),
            ("should", "ought to"),
        ]
        
        result = text
        for old, new in swaps:
            if old in result.lower() and self._random.random() < 0.5:
                result = re.sub(old, new, result, flags=re.IGNORECASE)
                break
        
        return result
    
    def _reorder_sentences(self, text: str) -> str:
        """Reorder sentences in multi-sentence text."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) < 2:
            return text
        
        # Don't reorder if probability check fails
        if self._random.random() > self.config.reorder_probability:
            return text
        
        # Shuffle middle sentences (keep first and last in place often)
        if len(sentences) > 2 and self._random.random() < 0.7:
            middle = sentences[1:-1]
            self._random.shuffle(middle)
            sentences = [sentences[0]] + middle + [sentences[-1]]
        else:
            self._random.shuffle(sentences)
        
        return ' '.join(sentences)
    
    def _inject_noise(self, text: str) -> str:
        """Inject minor noise into text."""
        words = text.split()
        
        for i, word in enumerate(words):
            if self._random.random() < self.config.noise_probability:
                noise_type = self._random.choice([
                    "typo", "double", "case"
                ])
                
                if noise_type == "typo" and len(word) > 3:
                    # Swap two adjacent characters
                    pos = self._random.randint(1, len(word) - 2)
                    word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
                elif noise_type == "double" and len(word) > 2:
                    # Double a character
                    pos = self._random.randint(0, len(word) - 1)
                    word = word[:pos] + word[pos] + word[pos:]
                elif noise_type == "case":
                    # Random case change
                    word = ''.join(
                        c.upper() if self._random.random() < 0.3 else c.lower()
                        for c in word
                    )
                
                words[i] = word
        
        return ' '.join(words)


class DataAugmentor:
    """Augment training datasets."""
    
    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize data augmentor.
        
        Args:
            config: Augmentation configuration
            api_key: Optional API key for advanced augmentation
        """
        self.config = config or AugmentationConfig()
        self.api_key = api_key
        self.text_augmentor = TextAugmentor(self.config)
    
    def augment_example(
        self,
        example: Dict[str, str],
        input_key: str = "input",
        output_key: str = "output"
    ) -> List[Dict[str, str]]:
        """
        Augment a single training example.
        
        Args:
            example: Training example dict
            input_key: Key for input text
            output_key: Key for output text
            
        Returns:
            List of augmented examples
        """
        results = []
        
        # Preserve original
        if self.config.preserve_original:
            results.append(example.copy())
        
        input_text = example.get(input_key, "")
        output_text = example.get(output_key, "")
        
        # Augment input
        augmented_inputs = self.text_augmentor.augment(input_text)
        
        for aug_input in augmented_inputs[:self.config.target_multiplier - 1]:
            results.append({
                input_key: aug_input,
                output_key: output_text,
                **{k: v for k, v in example.items() if k not in [input_key, output_key]}
            })
        
        return results
    
    def augment_dataset(
        self,
        dataset: List[Dict],
        input_key: str = "input",
        output_key: str = "output",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict]:
        """
        Augment an entire dataset.
        
        Args:
            dataset: List of training examples
            input_key: Key for input text
            output_key: Key for output text
            progress_callback: Optional callback(current, total)
            
        Returns:
            Augmented dataset
        """
        results = []
        total = len(dataset)
        
        for i, example in enumerate(dataset):
            augmented = self.augment_example(example, input_key, output_key)
            results.extend(augmented)
            
            if progress_callback:
                progress_callback(i + 1, total)
            
            # Check max limit
            if len(results) >= self.config.max_examples:
                logger.warning(f"Reached max examples limit: {self.config.max_examples}")
                break
        
        logger.info(f"Augmented {len(dataset)} examples to {len(results)}")
        return results
    
    def augment_file(
        self,
        input_path: Path,
        output_path: Path,
        format: str = "jsonl"
    ) -> int:
        """
        Augment a training data file.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            format: File format (jsonl or json)
            
        Returns:
            Number of examples written
        """
        # Load data
        if format == "jsonl":
            dataset = []
            with open(input_path) as f:
                for line in f:
                    if line.strip():
                        dataset.append(json.loads(line))
        else:
            with open(input_path) as f:
                dataset = json.load(f)
        
        # Augment
        augmented = self.augment_dataset(dataset)
        
        # Save
        if format == "jsonl":
            with open(output_path, 'w') as f:
                for example in augmented:
                    f.write(json.dumps(example) + '\n')
        else:
            with open(output_path, 'w') as f:
                json.dump(augmented, f, indent=2)
        
        return len(augmented)


class MixupAugmentor:
    """Mix examples to create new training data."""
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize mixup augmentor.
        
        Args:
            alpha: Mixup interpolation alpha
        """
        self.alpha = alpha
    
    def mixup_text(
        self,
        text1: str,
        text2: str,
        label1: float = 0.0,
        label2: float = 1.0
    ) -> Tuple[str, float]:
        """
        Mix two text examples.
        
        Args:
            text1: First text
            text2: Second text
            label1: First label
            label2: Second label
            
        Returns:
            (mixed_text, mixed_label)
        """
        # Sample lambda from beta distribution
        lam = random.betavariate(self.alpha, self.alpha)
        
        # For text, we concatenate with separator
        words1 = text1.split()
        words2 = text2.split()
        
        # Take proportional words from each
        n1 = int(len(words1) * lam)
        n2 = int(len(words2) * (1 - lam))
        
        mixed_words = words1[:n1] + words2[-n2:] if n2 > 0 else words1[:n1]
        mixed_text = ' '.join(mixed_words)
        
        # Mix labels
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_text, mixed_label


class BackTranslator:
    """Back-translation augmentation using translation APIs."""
    
    def __init__(self, api_key: str, provider: str = "google"):
        """
        Initialize back-translator.
        
        Args:
            api_key: API key for translation service
            provider: Translation provider
        """
        self.api_key = api_key
        self.provider = provider
        
        # Intermediate languages for back-translation
        self.intermediate_languages = ["fr", "de", "es", "ja", "zh"]
    
    def back_translate(
        self,
        text: str,
        intermediate_lang: Optional[str] = None
    ) -> str:
        """
        Perform back-translation.
        
        Args:
            text: Input text
            intermediate_lang: Intermediate language (random if None)
            
        Returns:
            Back-translated text
        """
        if intermediate_lang is None:
            intermediate_lang = random.choice(self.intermediate_languages)
        
        try:
            # Translate to intermediate language
            intermediate = self._translate(text, "en", intermediate_lang)
            
            # Translate back to English
            result = self._translate(intermediate, intermediate_lang, "en")
            
            return result
            
        except Exception as e:
            logger.error(f"Back-translation failed: {e}")
            return text
    
    def _translate(self, text: str, source: str, target: str) -> str:
        """Translate text between languages."""
        # Placeholder - implement with actual translation API
        # Google Translate, DeepL, etc.
        raise NotImplementedError("Translation API not implemented")


# Convenience functions
def augment_training_data(
    input_file: Path,
    output_file: Path,
    multiplier: int = 3
) -> int:
    """
    Augment a training data file.
    
    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file
        multiplier: Target multiplier for examples
        
    Returns:
        Number of augmented examples
    """
    config = AugmentationConfig(target_multiplier=multiplier)
    augmentor = DataAugmentor(config)
    return augmentor.augment_file(input_file, output_file)


def augment_text(text: str) -> List[str]:
    """
    Augment a single text.
    
    Args:
        text: Input text
        
    Returns:
        List of augmented texts
    """
    augmentor = TextAugmentor()
    return augmentor.augment(text)
