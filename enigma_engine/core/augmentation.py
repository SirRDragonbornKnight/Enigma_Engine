"""
Data Augmentation for Enigma AI Engine

Augment training data for better model performance.

Features:
- Text paraphrasing
- Synonym replacement
- Back translation
- Sentence shuffling
- Noise injection
- Template-based generation

Usage:
    from enigma_engine.core.augmentation import DataAugmenter, get_augmenter
    
    augmenter = get_augmenter()
    
    # Augment single text
    variants = augmenter.augment("Hello, how are you?")
    
    # Augment dataset
    augmented_data = augmenter.augment_dataset(original_data)
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AugmentConfig:
    """Configuration for data augmentation."""
    # Augmentation methods to use
    synonym_replace: bool = True
    random_insert: bool = True
    random_swap: bool = True
    random_delete: bool = True
    back_translate: bool = False
    paraphrase: bool = False
    
    # Parameters
    aug_probability: float = 0.3  # Probability of augmenting each word
    min_length: int = 3  # Minimum text length to augment
    max_augmentations: int = 5  # Max augmentations per text
    
    # Noise
    typo_probability: float = 0.0
    case_change_probability: float = 0.0


# Simple synonym dictionary
SYNONYMS = {
    "good": ["great", "excellent", "fine", "nice", "wonderful"],
    "bad": ["poor", "terrible", "awful", "dreadful", "horrible"],
    "big": ["large", "huge", "enormous", "massive", "giant"],
    "small": ["tiny", "little", "minute", "miniature", "compact"],
    "fast": ["quick", "rapid", "speedy", "swift", "hasty"],
    "slow": ["sluggish", "leisurely", "unhurried", "gradual", "plodding"],
    "happy": ["glad", "pleased", "joyful", "delighted", "content"],
    "sad": ["unhappy", "sorrowful", "melancholy", "gloomy", "depressed"],
    "help": ["assist", "aid", "support", "guide", "facilitate"],
    "make": ["create", "produce", "build", "construct", "generate"],
    "see": ["view", "observe", "notice", "spot", "perceive"],
    "think": ["believe", "consider", "suppose", "assume", "reckon"],
    "say": ["tell", "state", "mention", "express", "declare"],
    "want": ["desire", "wish", "need", "require", "prefer"],
    "use": ["utilize", "employ", "apply", "operate", "implement"],
    "go": ["move", "travel", "proceed", "advance", "depart"],
    "come": ["arrive", "approach", "appear", "reach", "enter"],
    "get": ["obtain", "acquire", "receive", "gain", "fetch"],
    "give": ["provide", "offer", "supply", "grant", "present"],
    "take": ["grab", "seize", "capture", "acquire", "accept"],
    "know": ["understand", "realize", "recognize", "comprehend", "grasp"],
    "find": ["discover", "locate", "detect", "uncover", "identify"],
    "look": ["appear", "seem", "glance", "gaze", "stare"],
    "work": ["function", "operate", "perform", "labor", "toil"],
    "run": ["execute", "operate", "sprint", "dash", "race"],
    "write": ["compose", "author", "draft", "pen", "type"],
    "read": ["study", "peruse", "examine", "review", "scan"],
    "speak": ["talk", "converse", "communicate", "discuss", "chat"],
    "learn": ["study", "discover", "understand", "master", "absorb"],
    "start": ["begin", "commence", "initiate", "launch", "originate"],
    "end": ["finish", "conclude", "terminate", "complete", "cease"],
    "important": ["significant", "crucial", "vital", "essential", "critical"],
    "different": ["various", "diverse", "distinct", "unique", "separate"],
}


class SynonymReplacer:
    """Replaces words with synonyms."""
    
    def __init__(self, synonyms: Optional[Dict[str, List[str]]] = None):
        self._synonyms = synonyms or SYNONYMS
    
    def replace(self, text: str, probability: float = 0.3) -> str:
        """
        Replace words with synonyms.
        
        Args:
            text: Input text
            probability: Probability of replacing each eligible word
            
        Returns:
            Augmented text
        """
        words = text.split()
        new_words = []
        
        for word in words:
            word_lower = word.lower()
            
            if (word_lower in self._synonyms and 
                random.random() < probability):
                synonyms = self._synonyms[word_lower]
                replacement = random.choice(synonyms)
                
                # Preserve case
                if word.isupper():
                    replacement = replacement.upper()
                elif word[0].isupper():
                    replacement = replacement.capitalize()
                
                new_words.append(replacement)
            else:
                new_words.append(word)
        
        return " ".join(new_words)


class RandomInserter:
    """Randomly inserts words."""
    
    def __init__(self, synonyms: Optional[Dict[str, List[str]]] = None):
        self._synonyms = synonyms or SYNONYMS
        self._all_words = []
        for syns in self._synonyms.values():
            self._all_words.extend(syns)
    
    def insert(self, text: str, n_insertions: int = 1) -> str:
        """
        Randomly insert words.
        
        Args:
            text: Input text
            n_insertions: Number of words to insert
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        for _ in range(n_insertions):
            if not words:
                break
            
            insert_word = random.choice(self._all_words)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, insert_word)
        
        return " ".join(words)


class RandomSwapper:
    """Randomly swaps word positions."""
    
    def swap(self, text: str, n_swaps: int = 1) -> str:
        """
        Randomly swap word positions.
        
        Args:
            text: Input text
            n_swaps: Number of swaps to perform
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        if len(words) < 2:
            return text
        
        for _ in range(n_swaps):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return " ".join(words)


class RandomDeleter:
    """Randomly deletes words."""
    
    def delete(self, text: str, probability: float = 0.1) -> str:
        """
        Randomly delete words.
        
        Args:
            text: Input text
            probability: Probability of deleting each word
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        if len(words) <= 1:
            return text
        
        new_words = [w for w in words if random.random() > probability]
        
        # Keep at least one word
        if not new_words:
            new_words = [random.choice(words)]
        
        return " ".join(new_words)


class TypoGenerator:
    """Generates realistic typos."""
    
    # Keyboard adjacent keys
    ADJACENT_KEYS = {
        'q': 'wa', 'w': 'qeas', 'e': 'wrsd', 'r': 'etdf', 't': 'ryfg',
        'y': 'tugh', 'u': 'yihj', 'i': 'uojk', 'o': 'iplk', 'p': 'ol',
        'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
        'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huiknm', 'k': 'jiolm',
        'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv',
        'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
    }
    
    def add_typo(self, text: str, probability: float = 0.05) -> str:
        """
        Add realistic typos.
        
        Args:
            text: Input text
            probability: Probability of typo per character
            
        Returns:
            Text with typos
        """
        result = []
        
        for char in text:
            if char.lower() in self.ADJACENT_KEYS and random.random() < probability:
                typo_type = random.choice(['adjacent', 'delete', 'duplicate', 'transpose'])
                
                if typo_type == 'adjacent':
                    adjacent = self.ADJACENT_KEYS[char.lower()]
                    new_char = random.choice(adjacent)
                    if char.isupper():
                        new_char = new_char.upper()
                    result.append(new_char)
                
                elif typo_type == 'delete':
                    pass  # Don't add the character
                
                elif typo_type == 'duplicate':
                    result.append(char)
                    result.append(char)
                
                elif typo_type == 'transpose' and len(result) > 0:
                    last = result.pop()
                    result.append(char)
                    result.append(last)
                
                else:
                    result.append(char)
            else:
                result.append(char)
        
        return "".join(result)


class CaseChanger:
    """Changes text case."""
    
    def change(self, text: str, probability: float = 0.1) -> str:
        """
        Randomly change word cases.
        
        Args:
            text: Input text
            probability: Probability of changing each word
            
        Returns:
            Transformed text
        """
        words = text.split()
        new_words = []
        
        for word in words:
            if random.random() < probability:
                change_type = random.choice(['upper', 'lower', 'title', 'swap'])
                
                if change_type == 'upper':
                    new_words.append(word.upper())
                elif change_type == 'lower':
                    new_words.append(word.lower())
                elif change_type == 'title':
                    new_words.append(word.title())
                elif change_type == 'swap':
                    new_words.append(word.swapcase())
            else:
                new_words.append(word)
        
        return " ".join(new_words)


class DataAugmenter:
    """
    Main data augmentation system.
    """
    
    def __init__(self, config: Optional[AugmentConfig] = None):
        """
        Initialize augmenter.
        
        Args:
            config: Augmentation configuration
        """
        self._config = config or AugmentConfig()
        
        # Initialize augmenters
        self._synonym = SynonymReplacer()
        self._inserter = RandomInserter()
        self._swapper = RandomSwapper()
        self._deleter = RandomDeleter()
        self._typo = TypoGenerator()
        self._case = CaseChanger()
    
    def augment(
        self,
        text: str,
        n_augmentations: int = 3
    ) -> List[str]:
        """
        Generate augmented versions of text.
        
        Args:
            text: Text to augment
            n_augmentations: Number of augmentations to generate
            
        Returns:
            List of augmented texts
        """
        if len(text.split()) < self._config.min_length:
            return [text]
        
        n = min(n_augmentations, self._config.max_augmentations)
        augmented = []
        
        for _ in range(n):
            aug_text = text
            
            # Apply random augmentations
            if self._config.synonym_replace and random.random() < 0.5:
                aug_text = self._synonym.replace(aug_text, self._config.aug_probability)
            
            if self._config.random_insert and random.random() < 0.3:
                aug_text = self._inserter.insert(aug_text, 1)
            
            if self._config.random_swap and random.random() < 0.3:
                aug_text = self._swapper.swap(aug_text, 1)
            
            if self._config.random_delete and random.random() < 0.2:
                aug_text = self._deleter.delete(aug_text, 0.1)
            
            if self._config.typo_probability > 0:
                aug_text = self._typo.add_typo(aug_text, self._config.typo_probability)
            
            if self._config.case_change_probability > 0:
                aug_text = self._case.change(aug_text, self._config.case_change_probability)
            
            # Only add if different from original
            if aug_text != text and aug_text not in augmented:
                augmented.append(aug_text)
        
        return augmented if augmented else [text]
    
    def augment_pair(
        self,
        input_text: str,
        output_text: str,
        n_augmentations: int = 3
    ) -> List[Tuple[str, str]]:
        """
        Augment input-output pair (e.g., for instruction tuning).
        
        Args:
            input_text: Input text
            output_text: Expected output
            n_augmentations: Number of augmentations
            
        Returns:
            List of (input, output) pairs
        """
        pairs = [(input_text, output_text)]
        
        # Augment input only (output should remain consistent)
        augmented_inputs = self.augment(input_text, n_augmentations)
        
        for aug_input in augmented_inputs:
            if aug_input != input_text:
                pairs.append((aug_input, output_text))
        
        return pairs
    
    def augment_dataset(
        self,
        data: List[Dict[str, str]],
        input_key: str = "input",
        output_key: str = "output",
        augmentation_factor: int = 2
    ) -> List[Dict[str, str]]:
        """
        Augment a dataset.
        
        Args:
            data: List of data samples
            input_key: Key for input text
            output_key: Key for output text
            augmentation_factor: How many times to augment (1 = original only)
            
        Returns:
            Augmented dataset
        """
        augmented_data = []
        
        for sample in data:
            # Keep original
            augmented_data.append(sample)
            
            if input_key in sample:
                input_text = sample[input_key]
                output_text = sample.get(output_key, "")
                
                # Generate augmentations
                pairs = self.augment_pair(
                    input_text,
                    output_text,
                    augmentation_factor - 1
                )
                
                for aug_input, aug_output in pairs[1:]:  # Skip original
                    new_sample = sample.copy()
                    new_sample[input_key] = aug_input
                    if output_key in sample:
                        new_sample[output_key] = aug_output
                    augmented_data.append(new_sample)
        
        return augmented_data
    
    def augment_conversation(
        self,
        messages: List[Dict[str, str]],
        user_only: bool = True
    ) -> List[Dict[str, str]]:
        """
        Augment a conversation.
        
        Args:
            messages: List of messages with 'role' and 'content'
            user_only: Only augment user messages
            
        Returns:
            Augmented conversation
        """
        augmented = []
        
        for msg in messages:
            if msg.get("role") == "user" or not user_only:
                content = msg.get("content", "")
                augs = self.augment(content, 1)
                
                new_msg = msg.copy()
                new_msg["content"] = augs[0] if augs else content
                augmented.append(new_msg)
            else:
                augmented.append(msg)
        
        return augmented


# Global instance
_augmenter: Optional[DataAugmenter] = None


def get_augmenter() -> DataAugmenter:
    """Get or create global augmenter."""
    global _augmenter
    if _augmenter is None:
        _augmenter = DataAugmenter()
    return _augmenter
