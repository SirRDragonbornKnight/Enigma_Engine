"""
Data Filter for Federated Learning

Filter and sanitize training data before federated learning to ensure:
- Privacy: Remove personally identifiable information
- Quality: Filter low-quality or malformed data
- Safety: Remove inappropriate content
"""

import logging
import re
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class DataFilter:
    """
    Filter training data for privacy and quality.
    
    Ensures that only appropriate, safe data is used for federated
    learning without leaking private information.
    """
    
    # Common PII patterns
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
    
    # Common profanity and inappropriate words (minimal list for demo)
    INAPPROPRIATE_WORDS = {
        'spam', 'scam', 'fraud',
        # Add more as needed
    }
    
    def __init__(
        self,
        remove_pii: bool = True,
        remove_inappropriate: bool = True,
        min_length: int = 10,
        max_length: int = 10000
    ):
        """
        Initialize data filter.
        
        Args:
            remove_pii: Remove personally identifiable information
            remove_inappropriate: Remove inappropriate content
            min_length: Minimum text length
            max_length: Maximum text length
        """
        self.remove_pii = remove_pii
        self.remove_inappropriate = remove_inappropriate
        self.min_length = min_length
        self.max_length = max_length
        
        logger.debug(
            f"Initialized DataFilter: "
            f"remove_pii={remove_pii}, "
            f"remove_inappropriate={remove_inappropriate}"
        )
    
    def filter_training_data(
        self, 
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter training data.
        
        Args:
            data: List of training examples (dicts with 'input' and 'output')
            
        Returns:
            Filtered data with PII removed and inappropriate content filtered
        """
        filtered = []
        
        for i, example in enumerate(data):
            try:
                # Filter input
                if 'input' in example:
                    filtered_input = self._filter_text(example['input'])
                    if filtered_input is None:
                        logger.debug(f"Filtered out example {i} (input)")
                        continue
                    example['input'] = filtered_input
                
                # Filter output
                if 'output' in example:
                    filtered_output = self._filter_text(example['output'])
                    if filtered_output is None:
                        logger.debug(f"Filtered out example {i} (output)")
                        continue
                    example['output'] = filtered_output
                
                filtered.append(example)
                
            except Exception as e:
                logger.warning(f"Error filtering example {i}: {e}")
                continue
        
        logger.info(
            f"Filtered training data: {len(filtered)}/{len(data)} examples kept"
        )
        
        return filtered
    
    def _filter_text(self, text: str) -> str | None:
        """
        Filter a single text string.
        
        Args:
            text: Text to filter
            
        Returns:
            Filtered text, or None if text should be removed
        """
        if not isinstance(text, str):
            return None
        
        # Check length
        if len(text) < self.min_length or len(text) > self.max_length:
            return None
        
        # Remove PII
        if self.remove_pii:
            text = self._remove_pii(text)
        
        # Check for inappropriate content
        if self.remove_inappropriate:
            if self._contains_inappropriate(text):
                return None
        
        return text
    
    def _remove_pii(self, text: str) -> str:
        """
        Remove personally identifiable information.
        
        Args:
            text: Original text
            
        Returns:
            Text with PII redacted
        """
        # Replace emails
        text = self.EMAIL_PATTERN.sub('[EMAIL]', text)
        
        # Replace phone numbers
        text = self.PHONE_PATTERN.sub('[PHONE]', text)
        
        # Replace SSNs
        text = self.SSN_PATTERN.sub('[SSN]', text)
        
        # Replace credit card numbers
        text = self.CREDIT_CARD_PATTERN.sub('[CREDIT_CARD]', text)
        
        return text
    
    def _contains_inappropriate(self, text: str) -> bool:
        """
        Check if text contains inappropriate content.
        
        Args:
            text: Text to check
            
        Returns:
            True if inappropriate
        """
        text_lower = text.lower()
        
        for word in self.INAPPROPRIATE_WORDS:
            if word in text_lower:
                return True
        
        return False
    
    def add_inappropriate_word(self, word: str) -> None:
        """
        Add a word to the inappropriate words list.
        
        Args:
            word: Word to add
        """
        self.INAPPROPRIATE_WORDS.add(word.lower())
        logger.debug(f"Added inappropriate word: {word}")
    
    def remove_inappropriate_word(self, word: str) -> None:
        """
        Remove a word from the inappropriate words list.
        
        Args:
            word: Word to remove
        """
        self.INAPPROPRIATE_WORDS.discard(word.lower())
        logger.debug(f"Removed inappropriate word: {word}")
    
    def get_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the data.
        
        Args:
            data: Training data
            
        Returns:
            Statistics dict
        """
        total = len(data)
        
        # Count examples with PII
        pii_count = 0
        for example in data:
            text = str(example.get('input', '')) + str(example.get('output', ''))
            if self._has_pii(text):
                pii_count += 1
        
        # Count inappropriate examples
        inappropriate_count = 0
        for example in data:
            text = str(example.get('input', '')) + str(example.get('output', ''))
            if self._contains_inappropriate(text):
                inappropriate_count += 1
        
        return {
            "total_examples": total,
            "examples_with_pii": pii_count,
            "inappropriate_examples": inappropriate_count,
        }
    
    def _has_pii(self, text: str) -> bool:
        """Check if text contains PII."""
        return bool(
            self.EMAIL_PATTERN.search(text) or
            self.PHONE_PATTERN.search(text) or
            self.SSN_PATTERN.search(text) or
            self.CREDIT_CARD_PATTERN.search(text)
        )
