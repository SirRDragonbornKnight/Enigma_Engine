"""
PII Scrubbing for Enigma AI Engine

Automatically detect and redact personal identifiable information.

Features:
- Email detection
- Phone numbers
- Credit cards
- SSN/tax IDs
- Names and addresses
- Custom patterns

Usage:
    from enigma_engine.utils.pii_scrubber import PIIScrubber, get_scrubber
    
    scrubber = get_scrubber()
    
    # Scrub text
    text = "Contact john@email.com or 555-123-4567"
    clean = scrubber.scrub(text)
    # "Contact [EMAIL] or [PHONE]"
    
    # Get detected PII
    pii = scrubber.detect(text)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Tuple

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    NAME = "name"
    ADDRESS = "address"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    URL = "url"
    CUSTOM = "custom"


@dataclass
class PIIMatch:
    """Detected PII match."""
    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class PIIPattern:
    """Pattern for PII detection."""
    pii_type: PIIType
    pattern: Pattern
    replacement: str = "[REDACTED]"
    confidence: float = 1.0


@dataclass
class ScrubConfig:
    """Scrubbing configuration."""
    # What to detect/scrub
    detect_email: bool = True
    detect_phone: bool = True
    detect_ssn: bool = True
    detect_credit_card: bool = True
    detect_ip: bool = True
    detect_url: bool = False  # Often legitimate
    
    # Replacement style
    replacement_style: str = "type"  # "type", "mask", "remove"
    mask_char: str = "*"
    
    # Minimum confidence threshold
    min_confidence: float = 0.8


class PIIDetector:
    """Detect PII in text."""
    
    # Common patterns
    PATTERNS = {
        PIIType.EMAIL: [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 1.0),
        ],
        PIIType.PHONE: [
            # US formats
            (r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', 0.9),
            (r'\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b', 0.95),
            (r'\b\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', 0.95),
            # International
            (r'\b\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b', 0.85),
        ],
        PIIType.SSN: [
            (r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', 0.9),
        ],
        PIIType.CREDIT_CARD: [
            # Visa, MC, etc.
            (r'\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 0.95),  # Visa
            (r'\b5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 0.95),  # MC
            (r'\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b', 0.95),  # Amex
            (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 0.7),  # Generic
        ],
        PIIType.IP_ADDRESS: [
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 0.9),
        ],
        PIIType.DATE_OF_BIRTH: [
            (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 0.7),
            (r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', 0.7),
        ],
        PIIType.URL: [
            (r'https?://[^\s<>"{}|\\^`\[\]]+', 0.95),
        ],
    }
    
    def __init__(self, config: Optional[ScrubConfig] = None):
        """Initialize detector."""
        self._config = config or ScrubConfig()
        self._compiled: Dict[PIIType, List[Tuple[Pattern, float]]] = {}
        self._custom_patterns: List[PIIPattern] = []
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns."""
        for pii_type, patterns in self.PATTERNS.items():
            self._compiled[pii_type] = [
                (re.compile(pattern, re.IGNORECASE), conf)
                for pattern, conf in patterns
            ]
    
    def add_custom_pattern(
        self,
        pattern: str,
        pii_type: PIIType = PIIType.CUSTOM,
        replacement: str = "[REDACTED]",
        confidence: float = 1.0
    ):
        """Add custom detection pattern."""
        self._custom_patterns.append(PIIPattern(
            pii_type=pii_type,
            pattern=re.compile(pattern),
            replacement=replacement,
            confidence=confidence
        ))
    
    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect all PII in text.
        
        Returns:
            List of PII matches
        """
        matches = []
        
        # Standard patterns
        for pii_type, patterns in self._compiled.items():
            if not self._should_detect(pii_type):
                continue
            
            for pattern, confidence in patterns:
                for match in pattern.finditer(text):
                    if confidence >= self._config.min_confidence:
                        matches.append(PIIMatch(
                            pii_type=pii_type,
                            value=match.group(),
                            start=match.start(),
                            end=match.end(),
                            confidence=confidence
                        ))
        
        # Custom patterns
        for custom in self._custom_patterns:
            for match in custom.pattern.finditer(text):
                if custom.confidence >= self._config.min_confidence:
                    matches.append(PIIMatch(
                        pii_type=custom.pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=custom.confidence
                    ))
        
        # Sort by position
        matches.sort(key=lambda m: m.start)
        
        return matches
    
    def _should_detect(self, pii_type: PIIType) -> bool:
        """Check if PII type should be detected."""
        mapping = {
            PIIType.EMAIL: self._config.detect_email,
            PIIType.PHONE: self._config.detect_phone,
            PIIType.SSN: self._config.detect_ssn,
            PIIType.CREDIT_CARD: self._config.detect_credit_card,
            PIIType.IP_ADDRESS: self._config.detect_ip,
            PIIType.URL: self._config.detect_url,
        }
        return mapping.get(pii_type, True)


class PIIScrubber:
    """Scrub PII from text."""
    
    def __init__(self, config: Optional[ScrubConfig] = None):
        """Initialize scrubber."""
        self._config = config or ScrubConfig()
        self._detector = PIIDetector(config)
        
        # Type-specific replacements
        self._replacements = {
            PIIType.EMAIL: "[EMAIL]",
            PIIType.PHONE: "[PHONE]",
            PIIType.SSN: "[SSN]",
            PIIType.CREDIT_CARD: "[CARD]",
            PIIType.IP_ADDRESS: "[IP]",
            PIIType.DATE_OF_BIRTH: "[DOB]",
            PIIType.NAME: "[NAME]",
            PIIType.ADDRESS: "[ADDRESS]",
            PIIType.URL: "[URL]",
            PIIType.CUSTOM: "[REDACTED]",
        }
        
        logger.info("PIIScrubber initialized")
    
    def scrub(self, text: str) -> str:
        """
        Scrub all PII from text.
        
        Returns:
            Text with PII replaced
        """
        matches = self._detector.detect(text)
        
        if not matches:
            return text
        
        # Build result, replacing matches
        result = []
        last_end = 0
        
        for match in matches:
            # Add text before match
            result.append(text[last_end:match.start])
            
            # Add replacement
            replacement = self._get_replacement(match)
            result.append(replacement)
            
            last_end = match.end
        
        # Add remaining text
        result.append(text[last_end:])
        
        return ''.join(result)
    
    def _get_replacement(self, match: PIIMatch) -> str:
        """Get replacement string for match."""
        if self._config.replacement_style == "type":
            return self._replacements.get(match.pii_type, "[REDACTED]")
        
        elif self._config.replacement_style == "mask":
            # Partially mask the value
            value = match.value
            if len(value) <= 4:
                return self._config.mask_char * len(value)
            else:
                # Show first and last 2 chars
                return value[:2] + self._config.mask_char * (len(value) - 4) + value[-2:]
        
        elif self._config.replacement_style == "remove":
            return ""
        
        return "[REDACTED]"
    
    def detect(self, text: str) -> List[PIIMatch]:
        """Detect PII without scrubbing."""
        return self._detector.detect(text)
    
    def contains_pii(self, text: str) -> bool:
        """Check if text contains any PII."""
        return len(self._detector.detect(text)) > 0
    
    def get_pii_summary(self, text: str) -> Dict[str, int]:
        """
        Get summary of PII types found.
        
        Returns:
            Dict of PII type -> count
        """
        matches = self._detector.detect(text)
        summary = {}
        
        for match in matches:
            key = match.pii_type.value
            summary[key] = summary.get(key, 0) + 1
        
        return summary
    
    def add_custom_pattern(
        self,
        pattern: str,
        replacement: str = "[REDACTED]",
        confidence: float = 1.0
    ):
        """Add custom scrubbing pattern."""
        self._detector.add_custom_pattern(
            pattern,
            PIIType.CUSTOM,
            replacement,
            confidence
        )
    
    def set_replacement(self, pii_type: PIIType, replacement: str):
        """Set custom replacement for PII type."""
        self._replacements[pii_type] = replacement


def scrub_training_data(
    data: List[Dict[str, str]],
    scrubber: Optional[PIIScrubber] = None
) -> List[Dict[str, str]]:
    """
    Scrub PII from training data.
    
    Args:
        data: List of training examples with 'input' and 'output' keys
        scrubber: PIIScrubber instance
        
    Returns:
        Scrubbed training data
    """
    if scrubber is None:
        scrubber = PIIScrubber()
    
    scrubbed = []
    
    for example in data:
        scrubbed_example = {}
        
        for key, value in example.items():
            if isinstance(value, str):
                scrubbed_example[key] = scrubber.scrub(value)
            else:
                scrubbed_example[key] = value
        
        scrubbed.append(scrubbed_example)
    
    return scrubbed


# Global instance
_scrubber: Optional[PIIScrubber] = None


def get_scrubber(config: Optional[ScrubConfig] = None) -> PIIScrubber:
    """Get or create global scrubber."""
    global _scrubber
    if _scrubber is None:
        _scrubber = PIIScrubber(config)
    return _scrubber
