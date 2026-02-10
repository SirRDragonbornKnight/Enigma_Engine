"""
PII Scrubbing for Enigma AI Engine

Auto-redact personal information in training data.

Features:
- Email detection
- Phone number detection
- SSN/ID detection
- Name detection
- Address detection
- Credit card detection
- Custom pattern support

Usage:
    from enigma_engine.utils.pii_scrubbing import PIIScrubber
    
    scrubber = PIIScrubber()
    
    # Scrub text
    clean = scrubber.scrub("Contact john@example.com")
    # Returns: "Contact [EMAIL]"
    
    # Scrub training data
    scrubber.scrub_file("data.txt", "data_clean.txt")
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII to detect."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "dob"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    CUSTOM = "custom"


@dataclass
class PIIMatch:
    """A detected PII match."""
    pii_type: PIIType
    start: int
    end: int
    text: str
    confidence: float = 1.0


@dataclass
class ScrubConfig:
    """PII scrubbing configuration."""
    # What to detect
    detect_emails: bool = True
    detect_phones: bool = True
    detect_ssn: bool = True
    detect_credit_cards: bool = True
    detect_ip_addresses: bool = True
    detect_names: bool = True
    detect_addresses: bool = True
    
    # How to replace
    replacement_format: str = "[{type}]"  # e.g., "[EMAIL]"
    hash_replacements: bool = False       # Replace with hash instead
    
    # Performance
    max_text_length: int = 1000000


class PIIDetector:
    """Detect PII in text."""
    
    def __init__(self):
        """Initialize PII detector."""
        # Compile patterns
        self._patterns = self._build_patterns()
        
        # Common names list (sample)
        self._common_names = self._load_common_names()
    
    def _build_patterns(self) -> Dict[PIIType, re.Pattern]:
        """Build regex patterns for PII detection."""
        return {
            PIIType.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE
            ),
            
            PIIType.PHONE: re.compile(
                r'''
                (?:
                    \+?1?[-.\s]?          # Country code
                )?
                (?:
                    \(?\d{3}\)?[-.\s]?    # Area code
                )
                \d{3}[-.\s]?\d{4}         # Number
                ''',
                re.VERBOSE
            ),
            
            PIIType.SSN: re.compile(
                r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
            ),
            
            PIIType.CREDIT_CARD: re.compile(
                r'''
                \b(?:
                    4[0-9]{12}(?:[0-9]{3})?|           # Visa
                    5[1-5][0-9]{14}|                   # MasterCard
                    3[47][0-9]{13}|                    # Amex
                    6(?:011|5[0-9]{2})[0-9]{12}        # Discover
                )\b
                ''',
                re.VERBOSE
            ),
            
            PIIType.IP_ADDRESS: re.compile(
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ),
            
            PIIType.DATE_OF_BIRTH: re.compile(
                r'''
                \b(?:
                    \d{1,2}[-/]\d{1,2}[-/]\d{2,4}|     # MM/DD/YYYY
                    \d{4}[-/]\d{1,2}[-/]\d{1,2}        # YYYY-MM-DD
                )\b
                ''',
                re.VERBOSE
            ),
            
            PIIType.ADDRESS: re.compile(
                r'''
                \b\d+\s+                               # Street number
                (?:[A-Za-z]+\s+){1,3}                  # Street name
                (?:
                    Street|St|Avenue|Ave|Road|Rd|
                    Boulevard|Blvd|Drive|Dr|Lane|Ln|
                    Court|Ct|Way|Place|Pl
                )
                \.?\b
                ''',
                re.VERBOSE | re.IGNORECASE
            ),
        }
    
    def _load_common_names(self) -> Set[str]:
        """Load common names for detection."""
        # Common first names (sample)
        return {
            'james', 'john', 'robert', 'michael', 'william', 'david',
            'mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'susan',
            'richard', 'joseph', 'thomas', 'charles', 'christopher', 'daniel',
            'matthew', 'anthony', 'mark', 'donald', 'steven', 'paul',
            'andrew', 'joshua', 'kenneth', 'kevin', 'brian', 'george',
            'sarah', 'karen', 'nancy', 'lisa', 'betty', 'margaret'
        }
    
    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect PII in text.
        
        Args:
            text: Text to scan
            
        Returns:
            List of PII matches
        """
        matches = []
        
        # Run pattern-based detection
        for pii_type, pattern in self._patterns.items():
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    start=match.start(),
                    end=match.end(),
                    text=match.group(),
                    confidence=1.0
                ))
        
        # Detect names (lower confidence)
        matches.extend(self._detect_names(text))
        
        # Remove overlapping matches (keep longest/highest confidence)
        matches = self._deduplicate_matches(matches)
        
        return matches
    
    def _detect_names(self, text: str) -> List[PIIMatch]:
        """Detect potential names in text."""
        matches = []
        
        # Look for capitalized words that are common names
        words = re.finditer(r'\b([A-Z][a-z]+)\b', text)
        
        for match in words:
            name = match.group(1).lower()
            
            if name in self._common_names:
                matches.append(PIIMatch(
                    pii_type=PIIType.NAME,
                    start=match.start(),
                    end=match.end(),
                    text=match.group(1),
                    confidence=0.7
                ))
        
        return matches
    
    def _deduplicate_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping matches."""
        if not matches:
            return []
        
        # Sort by start position
        sorted_matches = sorted(matches, key=lambda m: (m.start, -len(m.text)))
        
        result = []
        last_end = -1
        
        for match in sorted_matches:
            if match.start >= last_end:
                result.append(match)
                last_end = match.end
        
        return result


class PIIScrubber:
    """Scrub PII from text."""
    
    def __init__(self, config: Optional[ScrubConfig] = None):
        """
        Initialize PII scrubber.
        
        Args:
            config: Scrubbing configuration
        """
        self.config = config or ScrubConfig()
        self._detector = PIIDetector()
        
        # Custom patterns
        self._custom_patterns: Dict[str, re.Pattern] = {}
        
        # Stats
        self._scrub_count = 0
        self._type_counts: Dict[PIIType, int] = {}
    
    def scrub(self, text: str) -> str:
        """
        Scrub PII from text.
        
        Args:
            text: Text to scrub
            
        Returns:
            Scrubbed text
        """
        if len(text) > self.config.max_text_length:
            logger.warning(f"Text exceeds max length, truncating to {self.config.max_text_length}")
            text = text[:self.config.max_text_length]
        
        # Detect PII
        matches = self._detector.detect(text)
        
        # Also run custom patterns
        matches.extend(self._detect_custom(text))
        
        # Sort by position (reverse for replacement)
        matches = sorted(matches, key=lambda m: m.start, reverse=True)
        
        # Replace matches
        result = text
        for match in matches:
            if self._should_scrub(match.pii_type):
                replacement = self._get_replacement(match)
                result = result[:match.start] + replacement + result[match.end:]
                
                # Update stats
                self._scrub_count += 1
                self._type_counts[match.pii_type] = self._type_counts.get(match.pii_type, 0) + 1
        
        return result
    
    def scrub_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrub PII from a dictionary (recursive).
        
        Args:
            data: Dictionary to scrub
            
        Returns:
            Scrubbed dictionary
        """
        result = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.scrub(value)
            elif isinstance(value, dict):
                result[key] = self.scrub_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.scrub(v) if isinstance(v, str)
                    else self.scrub_dict(v) if isinstance(v, dict)
                    else v
                    for v in value
                ]
            else:
                result[key] = value
        
        return result
    
    def scrub_file(
        self,
        input_path: str,
        output_path: str,
        encoding: str = 'utf-8'
    ):
        """
        Scrub PII from a file.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            encoding: File encoding
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        with open(input_path, 'r', encoding=encoding) as f_in:
            with open(output_path, 'w', encoding=encoding) as f_out:
                for line in f_in:
                    scrubbed = self.scrub(line)
                    f_out.write(scrubbed)
        
        logger.info(f"Scrubbed {input_path} -> {output_path} ({self._scrub_count} replacements)")
    
    def add_custom_pattern(self, name: str, pattern: str):
        """
        Add a custom pattern to detect.
        
        Args:
            name: Pattern name
            pattern: Regex pattern
        """
        self._custom_patterns[name] = re.compile(pattern)
    
    def _should_scrub(self, pii_type: PIIType) -> bool:
        """Check if PII type should be scrubbed."""
        checks = {
            PIIType.EMAIL: self.config.detect_emails,
            PIIType.PHONE: self.config.detect_phones,
            PIIType.SSN: self.config.detect_ssn,
            PIIType.CREDIT_CARD: self.config.detect_credit_cards,
            PIIType.IP_ADDRESS: self.config.detect_ip_addresses,
            PIIType.NAME: self.config.detect_names,
            PIIType.ADDRESS: self.config.detect_addresses,
        }
        
        return checks.get(pii_type, True)
    
    def _get_replacement(self, match: PIIMatch) -> str:
        """Get replacement text for PII match."""
        if self.config.hash_replacements:
            import hashlib
            hash_val = hashlib.md5(match.text.encode()).hexdigest()[:8]
            return f"[{match.pii_type.value.upper()}:{hash_val}]"
        else:
            return self.config.replacement_format.format(
                type=match.pii_type.value.upper()
            )
    
    def _detect_custom(self, text: str) -> List[PIIMatch]:
        """Run custom pattern detection."""
        matches = []
        
        for name, pattern in self._custom_patterns.items():
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    pii_type=PIIType.CUSTOM,
                    start=match.start(),
                    end=match.end(),
                    text=match.group(),
                    confidence=1.0
                ))
        
        return matches
    
    def get_stats(self) -> Dict:
        """Get scrubbing statistics."""
        return {
            "total_scrubbed": self._scrub_count,
            "by_type": {k.value: v for k, v in self._type_counts.items()}
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self._scrub_count = 0
        self._type_counts.clear()


class TrainingDataCleaner:
    """Clean PII from training datasets."""
    
    def __init__(self, scrubber: Optional[PIIScrubber] = None):
        """
        Initialize cleaner.
        
        Args:
            scrubber: PII scrubber instance
        """
        self._scrubber = scrubber or PIIScrubber()
    
    def clean_jsonl(
        self,
        input_path: str,
        output_path: str,
        text_fields: List[str] = None
    ):
        """
        Clean JSONL training file.
        
        Args:
            input_path: Input file
            output_path: Output file
            text_fields: Fields to scrub (all if None)
        """
        import json
        
        with open(input_path, 'r') as f_in:
            with open(output_path, 'w') as f_out:
                for line in f_in:
                    if not line.strip():
                        continue
                    
                    data = json.loads(line)
                    
                    if text_fields:
                        for field in text_fields:
                            if field in data and isinstance(data[field], str):
                                data[field] = self._scrubber.scrub(data[field])
                    else:
                        data = self._scrubber.scrub_dict(data)
                    
                    f_out.write(json.dumps(data) + '\n')
    
    def clean_text_file(self, input_path: str, output_path: str):
        """Clean plain text file."""
        self._scrubber.scrub_file(input_path, output_path)


# Global instance
_scrubber: Optional[PIIScrubber] = None


def get_pii_scrubber() -> PIIScrubber:
    """Get or create global PII scrubber."""
    global _scrubber
    if _scrubber is None:
        _scrubber = PIIScrubber()
    return _scrubber


def scrub_text(text: str) -> str:
    """Quick scrub function."""
    return get_pii_scrubber().scrub(text)
