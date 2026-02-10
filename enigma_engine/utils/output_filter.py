"""
Output Filtering - Filter sensitive data from AI responses and outputs.

Provides comprehensive output filtering including:
- PII redaction (emails, phones, SSNs, credit cards)
- Secret/credential detection and redaction
- Custom pattern filtering
- URL/path filtering
- Profanity filtering
- Length/format constraints

Part of the Enigma AI Engine security utilities.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from re import Pattern
from typing import Any, Optional


class SensitiveDataType(Enum):
    """Types of sensitive data."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    API_KEY = "api_key"
    PASSWORD = "password"
    PRIVATE_KEY = "private_key"
    AWS_KEY = "aws_key"
    JWT_TOKEN = "jwt_token"
    URL = "url"
    FILE_PATH = "file_path"
    CUSTOM = "custom"


@dataclass
class RedactionResult:
    """Result of redaction operation."""
    original: str
    filtered: str
    redactions_made: int
    types_redacted: list[SensitiveDataType] = field(default_factory=list)
    redaction_details: list[dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original": self.original,
            "filtered": self.filtered,
            "redactions_made": self.redactions_made,
            "types_redacted": [t.value for t in self.types_redacted],
            "redaction_details": self.redaction_details
        }


@dataclass
class FilterRule:
    """Custom filter rule."""
    name: str
    pattern: Pattern
    replacement: str
    data_type: SensitiveDataType = SensitiveDataType.CUSTOM
    
    def apply(self, text: str) -> tuple[str, int]:
        """Apply filter rule, return (filtered_text, count)."""
        result, count = self.pattern.subn(self.replacement, text)
        return result, count


# Sensitive data patterns
SENSITIVE_PATTERNS: dict[SensitiveDataType, dict[str, Any]] = {
    SensitiveDataType.EMAIL: {
        "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "replacement": "[EMAIL REDACTED]",
        "description": "Email addresses"
    },
    SensitiveDataType.PHONE: {
        "pattern": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "replacement": "[PHONE REDACTED]",
        "description": "Phone numbers"
    },
    SensitiveDataType.SSN: {
        "pattern": r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        "replacement": "[SSN REDACTED]",
        "description": "Social Security Numbers"
    },
    SensitiveDataType.CREDIT_CARD: {
        "pattern": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "replacement": "[CARD REDACTED]",
        "description": "Credit card numbers"
    },
    SensitiveDataType.IP_ADDRESS: {
        "pattern": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "replacement": "[IP REDACTED]",
        "description": "IP addresses"
    },
    SensitiveDataType.API_KEY: {
        "pattern": r'\b(?:api[_-]?key|apikey|api_secret|api_token)["\']?\s*[:=]\s*["\']?([A-Za-z0-9_-]{16,})["\']?',
        "replacement": "[API_KEY REDACTED]",
        "description": "API keys"
    },
    SensitiveDataType.PASSWORD: {
        "pattern": r'(?:password|passwd|pwd|secret)["\']?\s*[:=]\s*["\']?([^\s"\']+)["\']?',
        "replacement": "[PASSWORD REDACTED]",
        "description": "Passwords"
    },
    SensitiveDataType.PRIVATE_KEY: {
        "pattern": r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----',
        "replacement": "[PRIVATE KEY REDACTED]",
        "description": "Private keys"
    },
    SensitiveDataType.AWS_KEY: {
        "pattern": r'\b(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b',
        "replacement": "[AWS_KEY REDACTED]",
        "description": "AWS access keys"
    },
    SensitiveDataType.JWT_TOKEN: {
        "pattern": r'\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b',
        "replacement": "[JWT REDACTED]",
        "description": "JWT tokens"
    },
}

# Additional secret patterns
SECRET_PATTERNS = [
    # GitHub tokens
    (r'\bgh[ps]_[A-Za-z0-9]{36}\b', "[GITHUB_TOKEN REDACTED]"),
    (r'\bgithub_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59}\b', "[GITHUB_PAT REDACTED]"),
    
    # Slack tokens
    (r'\bxox[baprs]-[A-Za-z0-9-]+\b', "[SLACK_TOKEN REDACTED]"),
    
    # Stripe keys
    (r'\b[sr]k_(?:live|test)_[A-Za-z0-9]{24,}\b', "[STRIPE_KEY REDACTED]"),
    
    # Google API keys
    (r'\bAIza[A-Za-z0-9_-]{35}\b', "[GOOGLE_API_KEY REDACTED]"),
    
    # Generic secrets
    (r'(?:secret|token|key|auth)["\']?\s*[:=]\s*["\']?([A-Za-z0-9_-]{20,})["\']?', "[SECRET REDACTED]"),
    
    # Bearer tokens
    (r'\bBearer\s+[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\b', "[BEARER_TOKEN REDACTED]"),
    
    # Base64 encoded secrets (likely)
    (r'(?:password|secret|key|token|auth)["\']?\s*[:=]\s*["\']?([A-Za-z0-9+/]{40,}={0,2})["\']?', "[ENCODED_SECRET REDACTED]"),
]

# Path patterns to redact
PATH_PATTERNS = [
    # Windows paths
    (r'[A-Za-z]:\\(?:Users|Documents and Settings)\\[^\\]+(?:\\[^\\<>:"|?*\s]+)*', "[PATH REDACTED]"),
    # Unix home paths
    (r'/(?:home|Users)/[^/]+(?:/[^\s/]+)*', "[PATH REDACTED]"),
    # Temp paths
    (r'(?:/tmp|/var/tmp|C:\\Temp|C:\\Windows\\Temp)(?:/[^\s/]+|\\[^\s\\]+)*', "[TEMP_PATH REDACTED]"),
]


class OutputFilter:
    """
    Filter sensitive data from AI responses and outputs.
    
    Usage:
        filter = OutputFilter()
        
        # Full filtering
        result = filter.filter(ai_response)
        safe_output = result.filtered
        
        # Specific filtering
        no_emails = filter.redact_emails(text)
        no_secrets = filter.redact_secrets(text)
        
        # Custom rules
        filter.add_rule("company_id", r"COMP-\\d{6}", "[ID REDACTED]")
        
        # Selective filtering
        result = filter.filter(text, types={SensitiveDataType.EMAIL, SensitiveDataType.PHONE})
    """
    
    def __init__(
        self,
        redact_emails: bool = True,
        redact_phones: bool = True,
        redact_ssns: bool = True,
        redact_cards: bool = True,
        redact_ips: bool = False,
        redact_secrets: bool = True,
        redact_paths: bool = False
    ):
        """
        Initialize the output filter.
        
        Args:
            redact_emails: Redact email addresses
            redact_phones: Redact phone numbers
            redact_ssns: Redact SSNs
            redact_cards: Redact credit cards
            redact_ips: Redact IP addresses
            redact_secrets: Redact API keys, passwords, tokens
            redact_paths: Redact file paths
        """
        self.settings = {
            SensitiveDataType.EMAIL: redact_emails,
            SensitiveDataType.PHONE: redact_phones,
            SensitiveDataType.SSN: redact_ssns,
            SensitiveDataType.CREDIT_CARD: redact_cards,
            SensitiveDataType.IP_ADDRESS: redact_ips,
            SensitiveDataType.API_KEY: redact_secrets,
            SensitiveDataType.PASSWORD: redact_secrets,
            SensitiveDataType.PRIVATE_KEY: redact_secrets,
            SensitiveDataType.AWS_KEY: redact_secrets,
            SensitiveDataType.JWT_TOKEN: redact_secrets,
        }
        self.redact_paths = redact_paths
        
        # Compile patterns
        self._compiled_patterns: dict[SensitiveDataType, Pattern] = {}
        for dtype, info in SENSITIVE_PATTERNS.items():
            self._compiled_patterns[dtype] = re.compile(
                info["pattern"], 
                re.IGNORECASE
            )
        
        # Compile secret patterns
        self._secret_patterns = [
            (re.compile(p, re.IGNORECASE), r) for p, r in SECRET_PATTERNS
        ]
        
        # Compile path patterns
        self._path_patterns = [
            (re.compile(p, re.IGNORECASE), r) for p, r in PATH_PATTERNS
        ]
        
        # Custom rules
        self._custom_rules: list[FilterRule] = []
    
    def filter(
        self,
        text: str,
        types: Optional[set[SensitiveDataType]] = None
    ) -> RedactionResult:
        """
        Filter sensitive data from text.
        
        Args:
            text: Text to filter
            types: Specific types to redact (uses settings if None)
            
        Returns:
            RedactionResult with filtered text and details
        """
        original = text
        filtered = text
        total_redactions = 0
        types_redacted: set[SensitiveDataType] = set()
        details: list[dict[str, Any]] = []
        
        # Determine which types to redact
        if types is None:
            types = {t for t, enabled in self.settings.items() if enabled}
        
        # Apply built-in patterns
        for dtype in types:
            if dtype in self._compiled_patterns:
                pattern = self._compiled_patterns[dtype]
                info = SENSITIVE_PATTERNS[dtype]
                
                matches = list(pattern.finditer(filtered))
                if matches:
                    count = len(matches)
                    total_redactions += count
                    types_redacted.add(dtype)
                    
                    for match in matches:
                        details.append({
                            "type": dtype.value,
                            "position": match.span(),
                            "length": len(match.group())
                        })
                    
                    filtered = pattern.sub(info["replacement"], filtered)
        
        # Apply secret patterns if API_KEY or PASSWORD in types
        if SensitiveDataType.API_KEY in types or SensitiveDataType.PASSWORD in types:
            for pattern, replacement in self._secret_patterns:
                matches = list(pattern.finditer(filtered))
                if matches:
                    count = len(matches)
                    total_redactions += count
                    types_redacted.add(SensitiveDataType.API_KEY)
                    
                    for match in matches:
                        details.append({
                            "type": "secret",
                            "position": match.span(),
                            "length": len(match.group())
                        })
                    
                    filtered = pattern.sub(replacement, filtered)
        
        # Apply path patterns
        if self.redact_paths:
            for pattern, replacement in self._path_patterns:
                matches = list(pattern.finditer(filtered))
                if matches:
                    count = len(matches)
                    total_redactions += count
                    types_redacted.add(SensitiveDataType.FILE_PATH)
                    
                    for match in matches:
                        details.append({
                            "type": "file_path",
                            "position": match.span(),
                            "length": len(match.group())
                        })
                    
                    filtered = pattern.sub(replacement, filtered)
        
        # Apply custom rules
        for rule in self._custom_rules:
            new_filtered, count = rule.apply(filtered)
            if count > 0:
                total_redactions += count
                types_redacted.add(rule.data_type)
                filtered = new_filtered
        
        return RedactionResult(
            original=original,
            filtered=filtered,
            redactions_made=total_redactions,
            types_redacted=list(types_redacted),
            redaction_details=details
        )
    
    def add_rule(
        self,
        name: str,
        pattern: str,
        replacement: str,
        data_type: SensitiveDataType = SensitiveDataType.CUSTOM
    ) -> None:
        """
        Add a custom filtering rule.
        
        Args:
            name: Rule name
            pattern: Regex pattern to match
            replacement: Replacement text
            data_type: Type of sensitive data
        """
        compiled = re.compile(pattern, re.IGNORECASE)
        rule = FilterRule(
            name=name,
            pattern=compiled,
            replacement=replacement,
            data_type=data_type
        )
        self._custom_rules.append(rule)
    
    def remove_rule(self, name: str) -> bool:
        """
        Remove a custom rule by name.
        
        Args:
            name: Rule name
            
        Returns:
            True if rule was removed
        """
        original_count = len(self._custom_rules)
        self._custom_rules = [r for r in self._custom_rules if r.name != name]
        return len(self._custom_rules) < original_count
    
    def redact_emails(self, text: str) -> str:
        """Redact email addresses from text."""
        result = self.filter(text, types={SensitiveDataType.EMAIL})
        return result.filtered
    
    def redact_phones(self, text: str) -> str:
        """Redact phone numbers from text."""
        result = self.filter(text, types={SensitiveDataType.PHONE})
        return result.filtered
    
    def redact_ssns(self, text: str) -> str:
        """Redact SSNs from text."""
        result = self.filter(text, types={SensitiveDataType.SSN})
        return result.filtered
    
    def redact_cards(self, text: str) -> str:
        """Redact credit card numbers from text."""
        result = self.filter(text, types={SensitiveDataType.CREDIT_CARD})
        return result.filtered
    
    def redact_ips(self, text: str) -> str:
        """Redact IP addresses from text."""
        result = self.filter(text, types={SensitiveDataType.IP_ADDRESS})
        return result.filtered
    
    def redact_secrets(self, text: str) -> str:
        """Redact API keys, passwords, and tokens from text."""
        result = self.filter(text, types={
            SensitiveDataType.API_KEY,
            SensitiveDataType.PASSWORD,
            SensitiveDataType.PRIVATE_KEY,
            SensitiveDataType.AWS_KEY,
            SensitiveDataType.JWT_TOKEN
        })
        return result.filtered
    
    def redact_pii(self, text: str) -> str:
        """Redact all PII (emails, phones, SSNs, cards)."""
        result = self.filter(text, types={
            SensitiveDataType.EMAIL,
            SensitiveDataType.PHONE,
            SensitiveDataType.SSN,
            SensitiveDataType.CREDIT_CARD
        })
        return result.filtered
    
    def redact_all(self, text: str) -> str:
        """Redact all known sensitive data types."""
        result = self.filter(text, types=set(SensitiveDataType))
        return result.filtered
    
    def detect_sensitive_data(
        self,
        text: str
    ) -> dict[SensitiveDataType, int]:
        """
        Detect sensitive data without redacting.
        
        Args:
            text: Text to scan
            
        Returns:
            Dict mapping data type to count found
        """
        results = {}
        
        for dtype, pattern in self._compiled_patterns.items():
            matches = pattern.findall(text)
            if matches:
                results[dtype] = len(matches)
        
        # Check secrets
        secret_count = 0
        for pattern, _ in self._secret_patterns:
            matches = pattern.findall(text)
            secret_count += len(matches)
        
        if secret_count > 0:
            results[SensitiveDataType.API_KEY] = results.get(
                SensitiveDataType.API_KEY, 0
            ) + secret_count
        
        # Check paths
        path_count = 0
        for pattern, _ in self._path_patterns:
            matches = pattern.findall(text)
            path_count += len(matches)
        
        if path_count > 0:
            results[SensitiveDataType.FILE_PATH] = path_count
        
        return results
    
    def has_sensitive_data(self, text: str) -> bool:
        """
        Check if text contains any sensitive data.
        
        Args:
            text: Text to check
            
        Returns:
            True if sensitive data detected
        """
        detected = self.detect_sensitive_data(text)
        return len(detected) > 0
    
    def mask_partial(
        self,
        text: str,
        data_type: SensitiveDataType,
        visible_chars: int = 4,
        mask_char: str = "*"
    ) -> str:
        """
        Partially mask sensitive data (show first/last chars).
        
        Args:
            text: Text to process
            data_type: Type of data to mask
            visible_chars: Number of chars to keep visible
            mask_char: Character to use for masking
            
        Returns:
            Text with partial masking
        """
        if data_type not in self._compiled_patterns:
            return text
        
        pattern = self._compiled_patterns[data_type]
        
        def masker(match):
            value = match.group()
            if len(value) <= visible_chars * 2:
                return mask_char * len(value)
            
            return (
                value[:visible_chars] +
                mask_char * (len(value) - visible_chars * 2) +
                value[-visible_chars:]
            )
        
        return pattern.sub(masker, text)
    
    def truncate(
        self,
        text: str,
        max_length: int,
        suffix: str = "..."
    ) -> str:
        """
        Truncate text to maximum length.
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add if truncated
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix


class ContentFilter:
    """
    Filter content based on categories (harmful, inappropriate, etc.).
    
    Usage:
        filter = ContentFilter()
        
        # Check content
        is_safe = filter.is_safe(text)
        
        # Get categories
        categories = filter.categorize(text)
        
        # Filter with replacement
        filtered = filter.filter_harmful(text)
    """
    
    def __init__(self):
        """Initialize content filter."""
        # Harmful content patterns
        self._harmful_patterns = [
            # Violence
            (r'\b(kill|murder|attack|assault|harm|hurt|injure)\s+(someone|people|person|him|her|them)\b', "harmful_violence"),
            # Self-harm
            (r'\b(suicide|self[- ]?harm|cut\s+myself)\b', "harmful_self_harm"),
            # Illegal activities
            (r'\b(hack|crack|steal|rob|break\s+into)\s+(account|system|computer|bank)\b', "harmful_illegal"),
        ]
        
        # Compile patterns
        self._compiled_harmful = [
            (re.compile(p, re.IGNORECASE), cat) for p, cat in self._harmful_patterns
        ]
    
    def categorize(self, text: str) -> set[str]:
        """
        Categorize content by detected issues.
        
        Args:
            text: Text to categorize
            
        Returns:
            Set of category names found
        """
        categories = set()
        
        for pattern, category in self._compiled_harmful:
            if pattern.search(text):
                categories.add(category)
        
        return categories
    
    def is_safe(self, text: str) -> bool:
        """
        Check if content is safe.
        
        Args:
            text: Text to check
            
        Returns:
            True if no harmful content detected
        """
        categories = self.categorize(text)
        return len(categories) == 0
    
    def filter_harmful(
        self,
        text: str,
        replacement: str = "[CONTENT FILTERED]"
    ) -> str:
        """
        Filter harmful content.
        
        Args:
            text: Text to filter
            replacement: Replacement text
            
        Returns:
            Filtered text
        """
        result = text
        
        for pattern, _ in self._compiled_harmful:
            result = pattern.sub(replacement, result)
        
        return result


# Convenience functions
def get_output_filter(**kwargs) -> OutputFilter:
    """Get an OutputFilter instance."""
    return OutputFilter(**kwargs)


def filter_output(text: str) -> str:
    """Quick output filtering."""
    return OutputFilter().filter(text).filtered


def redact_pii(text: str) -> str:
    """Quick PII redaction."""
    return OutputFilter().redact_pii(text)


def redact_secrets(text: str) -> str:
    """Quick secret redaction."""
    return OutputFilter().redact_secrets(text)


def detect_sensitive(text: str) -> dict[SensitiveDataType, int]:
    """Quick sensitive data detection."""
    return OutputFilter().detect_sensitive_data(text)


def has_sensitive_data(text: str) -> bool:
    """Quick check for sensitive data."""
    return OutputFilter().has_sensitive_data(text)
