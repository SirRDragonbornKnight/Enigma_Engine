"""
PII Scrubbing

Personal Identifiable Information detection and redaction.
Uses pattern matching and NER for comprehensive PII detection.

FILE: enigma_engine/security/pii_scrubber.py
TYPE: Security/Privacy
MAIN CLASSES: PIIScrubber, PIIDetector, PIIPattern
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from re import Pattern
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of Personal Identifiable Information."""
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
    API_KEY = "api_key"
    PASSWORD = "password"
    USERNAME = "username"
    URL = "url"


@dataclass
class PIIMatch:
    """A detected PII match."""
    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float = 1.0
    context: str = ""
    
    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class PIIPattern:
    """Pattern definition for PII detection."""
    pii_type: PIIType
    pattern: Pattern
    validator: Optional[callable] = None
    confidence: float = 0.9


@dataclass
class ScrubConfig:
    """PII scrubbing configuration."""
    enabled_types: list[PIIType] = field(default_factory=lambda: list(PIIType))
    redaction_char: str = "*"
    hash_values: bool = False  # Replace with hash instead of redaction
    preserve_format: bool = True  # Keep format like XXX-XX-XXXX
    log_detections: bool = True
    min_confidence: float = 0.7


class PIIDetector:
    """Detects PII in text using patterns and rules."""
    
    def __init__(self):
        """Initialize PII detector with patterns."""
        self._patterns = self._build_patterns()
        self._name_patterns = self._build_name_patterns()
    
    def _build_patterns(self) -> list[PIIPattern]:
        """Build regex patterns for PII detection."""
        return [
            # Email
            PIIPattern(
                PIIType.EMAIL,
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
                confidence=0.95
            ),
            
            # US Phone (various formats)
            PIIPattern(
                PIIType.PHONE,
                re.compile(r'\b(?:\+1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
                confidence=0.9
            ),
            
            # International phone
            PIIPattern(
                PIIType.PHONE,
                re.compile(r'\b\+[1-9]\d{1,14}\b'),
                confidence=0.85
            ),
            
            # SSN (US)
            PIIPattern(
                PIIType.SSN,
                re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'),
                self._validate_ssn,
                confidence=0.95
            ),
            
            # Credit Card (various formats - Luhn validated)
            PIIPattern(
                PIIType.CREDIT_CARD,
                re.compile(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'),
                self._validate_luhn,
                confidence=0.95
            ),
            
            # Credit Card (no separators)
            PIIPattern(
                PIIType.CREDIT_CARD,
                re.compile(r'\b\d{15,16}\b'),
                self._validate_luhn,
                confidence=0.9
            ),
            
            # IPv4 Address
            PIIPattern(
                PIIType.IP_ADDRESS,
                re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
                confidence=0.95
            ),
            
            # IPv6 Address
            PIIPattern(
                PIIType.IP_ADDRESS,
                re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),
                confidence=0.95
            ),
            
            # Date of Birth (various formats)
            PIIPattern(
                PIIType.DATE_OF_BIRTH,
                re.compile(r'\b(?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])[/\-.](?:19|20)\d{2}\b'),
                confidence=0.8
            ),
            
            # Date of Birth (ISO format)
            PIIPattern(
                PIIType.DATE_OF_BIRTH,
                re.compile(r'\b(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])\b'),
                confidence=0.8
            ),
            
            # US Passport
            PIIPattern(
                PIIType.PASSPORT,
                re.compile(r'\b[A-Z]\d{8}\b'),
                confidence=0.7
            ),
            
            # Bank Account (generic)
            PIIPattern(
                PIIType.BANK_ACCOUNT,
                re.compile(r'\b\d{9,18}\b'),
                confidence=0.5  # Low confidence - needs context
            ),
            
            # API Keys (common patterns)
            PIIPattern(
                PIIType.API_KEY,
                re.compile(r'\b(?:sk_live_|sk_test_|pk_live_|pk_test_)[A-Za-z0-9]{24,}\b'),  # Stripe
                confidence=0.99
            ),
            PIIPattern(
                PIIType.API_KEY,
                re.compile(r'\bAIza[0-9A-Za-z\-_]{35}\b'),  # Google
                confidence=0.99
            ),
            PIIPattern(
                PIIType.API_KEY,
                re.compile(r'\b[A-Za-z0-9]{32,}\b'),  # Generic - needs context
                confidence=0.3
            ),
            
            # Password in assignments
            PIIPattern(
                PIIType.PASSWORD,
                re.compile(r'(?:password|passwd|pwd)[\s]*[=:]\s*["\']?([^"\'\s]+)["\']?', re.IGNORECASE),
                confidence=0.95
            ),
        ]
    
    def _build_name_patterns(self) -> list[Pattern]:
        """Build patterns for name detection."""
        # Common title patterns
        return [
            re.compile(r'\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'),
            re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'),  # Two capitalized words
        ]
    
    def _validate_ssn(self, value: str) -> bool:
        """Validate SSN format and rules."""
        cleaned = re.sub(r'[-.\s]', '', value)
        if len(cleaned) != 9:
            return False
        
        # Check for invalid SSN patterns
        if cleaned[:3] in ('000', '666') or cleaned[:3].startswith('9'):
            return False
        if cleaned[3:5] == '00' or cleaned[5:] == '0000':
            return False
        
        return True
    
    def _validate_luhn(self, value: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        cleaned = re.sub(r'[-.\s]', '', value)
        
        if not cleaned.isdigit():
            return False
        
        if len(cleaned) < 13 or len(cleaned) > 19:
            return False
        
        # Luhn algorithm
        total = 0
        reverse = cleaned[::-1]
        
        for i, digit in enumerate(reverse):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        
        return total % 10 == 0
    
    def detect(self, text: str, config: ScrubConfig = None) -> list[PIIMatch]:
        """
        Detect PII in text.
        
        Args:
            text: Text to scan
            config: Detection configuration
            
        Returns:
            List of PII matches
        """
        config = config or ScrubConfig()
        matches = []
        
        for pii_pattern in self._patterns:
            if pii_pattern.pii_type not in config.enabled_types:
                continue
            
            for match in pii_pattern.pattern.finditer(text):
                value = match.group()
                
                # Run validator if present
                if pii_pattern.validator:
                    if not pii_pattern.validator(value):
                        continue
                
                confidence = pii_pattern.confidence
                if confidence < config.min_confidence:
                    continue
                
                # Get context
                start_ctx = max(0, match.start() - 20)
                end_ctx = min(len(text), match.end() + 20)
                context = text[start_ctx:end_ctx]
                
                matches.append(PIIMatch(
                    pii_type=pii_pattern.pii_type,
                    value=value,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    context=context
                ))
        
        # Deduplicate overlapping matches
        matches = self._deduplicate(matches)
        
        return matches
    
    def _deduplicate(self, matches: list[PIIMatch]) -> list[PIIMatch]:
        """Remove overlapping matches, keeping highest confidence."""
        if not matches:
            return []
        
        # Sort by start position
        sorted_matches = sorted(matches, key=lambda m: (m.start, -m.confidence))
        
        result = []
        last_end = -1
        
        for match in sorted_matches:
            if match.start >= last_end:
                result.append(match)
                last_end = match.end
        
        return result


class PIIScrubber:
    """Scrubs PII from text."""
    
    def __init__(self, config: Optional[ScrubConfig] = None):
        """
        Initialize PII scrubber.
        
        Args:
            config: Scrubbing configuration
        """
        self._config = config or ScrubConfig()
        self._detector = PIIDetector()
        self._stats = {
            "total_scanned": 0,
            "total_redacted": 0,
            "by_type": {t.value: 0 for t in PIIType}
        }
    
    def scrub(self, text: str) -> tuple[str, list[PIIMatch]]:
        """
        Scrub PII from text.
        
        Args:
            text: Text to scrub
            
        Returns:
            Tuple of (scrubbed text, list of matches)
        """
        matches = self._detector.detect(text, self._config)
        
        if not matches:
            self._stats["total_scanned"] += 1
            return text, []
        
        # Sort by position descending to replace from end
        matches_sorted = sorted(matches, key=lambda m: m.start, reverse=True)
        
        result = text
        for match in matches_sorted:
            replacement = self._generate_replacement(match)
            result = result[:match.start] + replacement + result[match.end:]
            
            # Update stats
            self._stats["total_redacted"] += 1
            self._stats["by_type"][match.pii_type.value] += 1
        
        self._stats["total_scanned"] += 1
        
        if self._config.log_detections:
            logger.info(f"Scrubbed {len(matches)} PII items from text")
        
        return result, matches
    
    def _generate_replacement(self, match: PIIMatch) -> str:
        """Generate replacement for PII match."""
        if self._config.hash_values:
            # Generate consistent hash
            hash_val = hashlib.sha256(match.value.encode()).hexdigest()[:8]
            return f"[{match.pii_type.value.upper()}:{hash_val}]"
        
        if self._config.preserve_format:
            return self._format_preserving_redact(match)
        
        return self._config.redaction_char * match.length
    
    def _format_preserving_redact(self, match: PIIMatch) -> str:
        """Redact while preserving format characters."""
        result = []
        for char in match.value:
            if char.isalnum():
                result.append(self._config.redaction_char)
            else:
                result.append(char)
        return ''.join(result)
    
    def scrub_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively scrub PII from a dictionary.
        
        Args:
            data: Dictionary to scrub
            
        Returns:
            Scrubbed dictionary
        """
        result = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                scrubbed, _ = self.scrub(value)
                result[key] = scrubbed
            elif isinstance(value, dict):
                result[key] = self.scrub_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.scrub_dict(item) if isinstance(item, dict)
                    else self.scrub(item)[0] if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result
    
    def get_stats(self) -> dict[str, Any]:
        """Get scrubbing statistics."""
        return self._stats.copy()
    
    def detect_only(self, text: str) -> list[PIIMatch]:
        """Detect PII without scrubbing."""
        return self._detector.detect(text, self._config)


def scrub_text(text: str, **kwargs) -> str:
    """
    Quick function to scrub PII from text.
    
    Args:
        text: Text to scrub
        **kwargs: ScrubConfig parameters
        
    Returns:
        Scrubbed text
    """
    config = ScrubConfig(**kwargs) if kwargs else None
    scrubber = PIIScrubber(config)
    result, _ = scrubber.scrub(text)
    return result


def detect_pii(text: str) -> list[dict[str, Any]]:
    """
    Quick function to detect PII in text.
    
    Args:
        text: Text to scan
        
    Returns:
        List of PII detection results
    """
    scrubber = PIIScrubber()
    matches = scrubber.detect_only(text)
    
    return [
        {
            "type": m.pii_type.value,
            "value": m.value,
            "start": m.start,
            "end": m.end,
            "confidence": m.confidence
        }
        for m in matches
    ]


__all__ = [
    'PIIScrubber',
    'PIIDetector',
    'PIIPattern',
    'PIIMatch',
    'PIIType',
    'ScrubConfig',
    'scrub_text',
    'detect_pii'
]
