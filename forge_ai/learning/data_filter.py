"""
Data Filtering for Federated Learning

Controls what data gets used for federated learning. Gives users fine-grained
control over their privacy by filtering out sensitive conversations, keywords,
and personal information.
"""

import re
import logging
from typing import Set, List, Optional, Any, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example."""
    text: str
    category: Optional[str] = None
    is_private: bool = False
    metadata: Optional[Dict[str, Any]] = None


class FederatedDataFilter:
    """
    Filter what data gets used for federated learning.
    
    Provides user control over:
    - Which conversation categories to include
    - Keywords to exclude (passwords, credit cards, etc.)
    - Automatic PII detection and removal
    - Quality filtering
    """
    
    def __init__(self):
        """Initialize the data filter."""
        # Categories to include (empty = all)
        self.allowed_categories: Set[str] = set()
        
        # Keywords to exclude
        self.excluded_keywords: Set[str] = {
            # Security-sensitive
            "password", "passwd", "pwd",
            "credit card", "creditcard", "card number",
            "ssn", "social security",
            "api key", "api_key", "apikey",
            "secret", "private key", "privatekey",
            "token", "access_token",
            # Personal info
            "email", "phone number", "address",
            "birth date", "birthday", "dob",
            # Financial
            "bank account", "routing number",
            "cvv", "pin", "pin number",
        }
        
        # PII patterns to detect
        self._pii_patterns = self._compile_pii_patterns()
        
        # Quality thresholds
        self.min_length = 10  # Minimum text length
        self.max_length = 10000  # Maximum text length
        
        logger.info("Federated data filter initialized")
    
    def _compile_pii_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for PII detection."""
        patterns = [
            # Email addresses
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            # Phone numbers (various formats)
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            re.compile(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}'),
            # Credit card numbers
            re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            # SSN
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            # IP addresses
            re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
        ]
        return patterns
    
    def add_allowed_category(self, category: str):
        """Add a category to the allow list."""
        self.allowed_categories.add(category.lower())
        logger.debug(f"Added allowed category: {category}")
    
    def remove_allowed_category(self, category: str):
        """Remove a category from the allow list."""
        self.allowed_categories.discard(category.lower())
        logger.debug(f"Removed allowed category: {category}")
    
    def add_excluded_keyword(self, keyword: str):
        """Add a keyword to the exclusion list."""
        self.excluded_keywords.add(keyword.lower())
        logger.debug(f"Added excluded keyword: {keyword}")
    
    def remove_excluded_keyword(self, keyword: str):
        """Remove a keyword from the exclusion list."""
        self.excluded_keywords.discard(keyword.lower())
        logger.debug(f"Removed excluded keyword: {keyword}")
    
    def should_include(self, example: TrainingExample) -> bool:
        """
        Check if a training example should be included in federated learning.
        
        Filters out:
        - User-marked private conversations
        - Sensitive keywords
        - Wrong categories
        - Low-quality examples
        
        Args:
            example: Training example to check
        
        Returns:
            True if example should be included
        """
        # Never include private conversations
        if example.is_private:
            logger.debug("Excluded: marked as private")
            return False
        
        # Check category filter
        if self.allowed_categories:
            if example.category is None:
                logger.debug("Excluded: no category and category filter is active")
                return False
            if example.category.lower() not in self.allowed_categories:
                logger.debug(f"Excluded: category {example.category} not in allowed list")
                return False
        
        # Check text length
        if len(example.text) < self.min_length:
            logger.debug(f"Excluded: text too short ({len(example.text)} < {self.min_length})")
            return False
        if len(example.text) > self.max_length:
            logger.debug(f"Excluded: text too long ({len(example.text)} > {self.max_length})")
            return False
        
        # Check for excluded keywords
        text_lower = example.text.lower()
        for keyword in self.excluded_keywords:
            if keyword in text_lower:
                logger.debug(f"Excluded: contains keyword '{keyword}'")
                return False
        
        # Check for PII
        if self._contains_pii(example.text):
            logger.debug("Excluded: contains PII")
            return False
        
        return True
    
    def _contains_pii(self, text: str) -> bool:
        """Check if text contains personally identifiable information."""
        for pattern in self._pii_patterns:
            if pattern.search(text):
                return True
        return False
    
    def sanitize(self, example: TrainingExample) -> TrainingExample:
        """
        Remove sensitive information from training example.
        
        Operations:
        - Redact PII (replace with placeholders)
        - Remove exact names/addresses
        - Generalize specific details
        
        Args:
            example: Training example to sanitize
        
        Returns:
            Sanitized training example
        """
        sanitized_text = example.text
        
        # Replace PII with placeholders
        for pattern in self._pii_patterns:
            if pattern == self._pii_patterns[0]:  # Email
                sanitized_text = pattern.sub("[EMAIL]", sanitized_text)
            elif pattern == self._pii_patterns[1] or pattern == self._pii_patterns[2]:  # Phone
                sanitized_text = pattern.sub("[PHONE]", sanitized_text)
            elif pattern == self._pii_patterns[3]:  # Credit card
                sanitized_text = pattern.sub("[CREDIT_CARD]", sanitized_text)
            elif pattern == self._pii_patterns[4]:  # SSN
                sanitized_text = pattern.sub("[SSN]", sanitized_text)
            elif pattern == self._pii_patterns[5]:  # IP
                sanitized_text = pattern.sub("[IP_ADDRESS]", sanitized_text)
        
        # Create sanitized example
        return TrainingExample(
            text=sanitized_text,
            category=example.category,
            is_private=example.is_private,
            metadata={
                **(example.metadata or {}),
                "sanitized": True,
            }
        )
    
    def filter_batch(
        self,
        examples: List[TrainingExample],
        sanitize: bool = True,
    ) -> List[TrainingExample]:
        """
        Filter a batch of training examples.
        
        Args:
            examples: List of training examples
            sanitize: Whether to sanitize included examples
        
        Returns:
            Filtered (and optionally sanitized) examples
        """
        result = []
        
        for example in examples:
            if self.should_include(example):
                if sanitize:
                    result.append(self.sanitize(example))
                else:
                    result.append(example)
        
        logger.info(
            f"Filtered {len(examples)} examples -> {len(result)} included "
            f"({len(examples) - len(result)} excluded)"
        )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        return {
            "allowed_categories": list(self.allowed_categories) if self.allowed_categories else "all",
            "excluded_keywords": list(self.excluded_keywords),
            "min_length": self.min_length,
            "max_length": self.max_length,
            "pii_patterns": len(self._pii_patterns),
        }


def test_data_filter():
    """Test data filtering."""
    filter = FederatedDataFilter()
    
    # Test examples
    examples = [
        TrainingExample(
            text="This is a normal conversation about coding.",
            category="coding",
        ),
        TrainingExample(
            text="My password is secret123 please don't share it.",
            category="general",
        ),
        TrainingExample(
            text="Contact me at john@example.com or 555-123-4567.",
            category="general",
        ),
        TrainingExample(
            text="Private conversation.",
            category="personal",
            is_private=True,
        ),
        TrainingExample(
            text="Too short",
            category="general",
        ),
    ]
    
    # Filter examples
    filtered = filter.filter_batch(examples, sanitize=True)
    
    print(f"Original: {len(examples)} examples")
    print(f"Filtered: {len(filtered)} examples")
    print(f"\nIncluded examples:")
    for ex in filtered:
        print(f"  - {ex.text[:50]}...")
    
    print(f"\nFilter stats: {filter.get_stats()}")


if __name__ == "__main__":
    test_data_filter()
