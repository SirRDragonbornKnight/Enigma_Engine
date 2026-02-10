"""
Input Sanitization - Sanitize and validate user inputs against injection attacks.

Provides comprehensive input sanitization including:
- SQL injection prevention
- XSS prevention
- Command injection prevention
- Path traversal prevention
- Input type validation
- Length/format constraints

Part of the Enigma AI Engine security utilities.
"""

import html
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ThreatType(Enum):
    """Types of security threats."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    LDAP_INJECTION = "ldap_injection"
    XML_INJECTION = "xml_injection"
    TEMPLATE_INJECTION = "template_injection"
    UNICODE_ATTACK = "unicode_attack"
    NULL_BYTE = "null_byte"
    FORMAT_STRING = "format_string"


@dataclass
class SanitizationResult:
    """Result of sanitization operation."""
    original: str
    sanitized: str
    is_safe: bool
    threats_found: list[ThreatType] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    changes_made: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original": self.original,
            "sanitized": self.sanitized,
            "is_safe": self.is_safe,
            "threats_found": [t.value for t in self.threats_found],
            "warnings": self.warnings,
            "changes_made": self.changes_made
        }


@dataclass
class ValidationRule:
    """Rule for input validation."""
    name: str
    check: Callable[[str], bool]
    message: str
    
    def validate(self, value: str) -> Optional[str]:
        """Return error message if validation fails."""
        if not self.check(value):
            return self.message
        return None


# SQL injection patterns
SQL_INJECTION_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE)\b)",
    r"('|\")\s*(OR|AND)\s*('|\")?\s*=\s*('|\")?",
    r";\s*--",
    r"'\s*OR\s+'",
    r"1\s*=\s*1",
    r"'\s*=\s*'",
    r"--\s*$",
    r"/\*.*\*/",
    r"@@\w+",
    r"CHAR\s*\(",
    r"WAITFOR\s+DELAY",
    r"BENCHMARK\s*\(",
    r"LOAD_FILE\s*\(",
    r"INTO\s+(OUTFILE|DUMPFILE)",
]

# XSS patterns
XSS_PATTERNS = [
    r"<\s*script",
    r"javascript\s*:",
    r"vbscript\s*:",
    r"on\w+\s*=",
    r"<\s*iframe",
    r"<\s*frame",
    r"<\s*object",
    r"<\s*embed",
    r"<\s*applet",
    r"<\s*form",
    r"<\s*input",
    r"<\s*button",
    r"<\s*img[^>]+onerror",
    r"<\s*body[^>]+onload",
    r"<\s*svg[^>]+onload",
    r"expression\s*\(",
    r"url\s*\(\s*['\"]?\s*data:",
    r"<\s*meta[^>]+http-equiv",
    r"<\s*base[^>]+href",
    r"<\s*link[^>]+href",
]

# Command injection patterns
COMMAND_PATTERNS = [
    r"[;&|`$]",
    r"\$\(",
    r"\$\{",
    r">\s*>",
    r"<\s*<",
    r"\|\s*\|",
    r"&&",
    r"`[^`]+`",
    r"\$\([^)]+\)",
    r">\s*/dev/",
    r">\s*/tmp/",
    r"2>&1",
]

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.\\",
    r"%2e%2e[/\\]",
    r"\.%2e[/\\]",
    r"%2e\.[/\\]",
    r"\.\./",
    r"\\\.\.\\",
    r"/etc/",
    r"C:\\",
    r"/proc/",
    r"/dev/",
]

# LDAP injection patterns
LDAP_PATTERNS = [
    r"[)(|*\\]",
    r"\x00",
    r"\\[0-9a-fA-F]{2}",
]

# XML injection patterns
XML_PATTERNS = [
    r"<!ENTITY",
    r"<!DOCTYPE",
    r"<!\[CDATA\[",
    r"SYSTEM\s+['\"]",
    r"PUBLIC\s+['\"]",
    r"&[a-zA-Z]+;",
    r"&#\d+;",
    r"&#x[0-9a-fA-F]+;",
]

# Template injection patterns
TEMPLATE_PATTERNS = [
    r"\{\{.*\}\}",
    r"\{%.*%\}",
    r"\$\{.*\}",
    r"<%.*%>",
    r"#\{.*\}",
]


class InputSanitizer:
    """
    Comprehensive input sanitization for security.
    
    Usage:
        sanitizer = InputSanitizer()
        
        # Full sanitization
        result = sanitizer.sanitize(user_input)
        if result.is_safe:
            process(result.sanitized)
        
        # Specific sanitization
        safe_html = sanitizer.sanitize_html(user_input)
        safe_sql = sanitizer.sanitize_sql(user_input)
        safe_path = sanitizer.sanitize_path(user_input)
        
        # Validation
        errors = sanitizer.validate(user_input, [
            sanitizer.rules.max_length(100),
            sanitizer.rules.alphanumeric(),
        ])
    """
    
    def __init__(self, strict: bool = False):
        """
        Initialize the sanitizer.
        
        Args:
            strict: If True, reject suspicious inputs instead of sanitizing
        """
        self.strict = strict
        self.rules = ValidationRules()
        
        # Compile patterns
        self._sql_patterns = [re.compile(p, re.IGNORECASE) for p in SQL_INJECTION_PATTERNS]
        self._xss_patterns = [re.compile(p, re.IGNORECASE) for p in XSS_PATTERNS]
        self._cmd_patterns = [re.compile(p) for p in COMMAND_PATTERNS]
        self._path_patterns = [re.compile(p, re.IGNORECASE) for p in PATH_TRAVERSAL_PATTERNS]
        self._ldap_patterns = [re.compile(p) for p in LDAP_PATTERNS]
        self._xml_patterns = [re.compile(p, re.IGNORECASE) for p in XML_PATTERNS]
        self._template_patterns = [re.compile(p) for p in TEMPLATE_PATTERNS]
    
    def sanitize(
        self,
        text: str,
        threats_to_check: Optional[set[ThreatType]] = None,
        max_length: Optional[int] = None
    ) -> SanitizationResult:
        """
        Perform comprehensive sanitization.
        
        Args:
            text: Input text to sanitize
            threats_to_check: Specific threats to check (all if None)
            max_length: Maximum allowed length
            
        Returns:
            SanitizationResult with sanitized text and details
        """
        original = text
        threats_found: list[ThreatType] = []
        warnings: list[str] = []
        changes_made: list[str] = []
        
        # Check all threats by default
        if threats_to_check is None:
            threats_to_check = set(ThreatType)
        
        # Null byte check
        if ThreatType.NULL_BYTE in threats_to_check:
            if '\x00' in text or '%00' in text:
                threats_found.append(ThreatType.NULL_BYTE)
                text = text.replace('\x00', '').replace('%00', '')
                changes_made.append("Removed null bytes")
        
        # Unicode normalization
        if ThreatType.UNICODE_ATTACK in threats_to_check:
            normalized = unicodedata.normalize('NFKC', text)
            if normalized != text:
                warnings.append("Unicode normalized")
                text = normalized
        
        # SQL injection check
        if ThreatType.SQL_INJECTION in threats_to_check:
            for pattern in self._sql_patterns:
                if pattern.search(text):
                    threats_found.append(ThreatType.SQL_INJECTION)
                    break
        
        # XSS check
        if ThreatType.XSS in threats_to_check:
            for pattern in self._xss_patterns:
                if pattern.search(text):
                    threats_found.append(ThreatType.XSS)
                    text = self.sanitize_html(text)
                    changes_made.append("HTML entities escaped")
                    break
        
        # Command injection check
        if ThreatType.COMMAND_INJECTION in threats_to_check:
            for pattern in self._cmd_patterns:
                if pattern.search(text):
                    threats_found.append(ThreatType.COMMAND_INJECTION)
                    break
        
        # Path traversal check
        if ThreatType.PATH_TRAVERSAL in threats_to_check:
            for pattern in self._path_patterns:
                if pattern.search(text):
                    threats_found.append(ThreatType.PATH_TRAVERSAL)
                    text = self.sanitize_path(text)
                    changes_made.append("Path traversal sequences removed")
                    break
        
        # LDAP injection check
        if ThreatType.LDAP_INJECTION in threats_to_check:
            for pattern in self._ldap_patterns:
                if pattern.search(text):
                    threats_found.append(ThreatType.LDAP_INJECTION)
                    break
        
        # XML injection check
        if ThreatType.XML_INJECTION in threats_to_check:
            for pattern in self._xml_patterns:
                if pattern.search(text):
                    threats_found.append(ThreatType.XML_INJECTION)
                    break
        
        # Template injection check
        if ThreatType.TEMPLATE_INJECTION in threats_to_check:
            for pattern in self._template_patterns:
                if pattern.search(text):
                    threats_found.append(ThreatType.TEMPLATE_INJECTION)
                    break
        
        # Length check
        if max_length and len(text) > max_length:
            text = text[:max_length]
            warnings.append(f"Truncated to {max_length} characters")
            changes_made.append(f"Truncated from {len(original)} to {max_length}")
        
        # In strict mode, empty the text if threats found
        if self.strict and threats_found:
            text = ""
            changes_made.append("Input rejected in strict mode")
        
        return SanitizationResult(
            original=original,
            sanitized=text,
            is_safe=len(threats_found) == 0,
            threats_found=threats_found,
            warnings=warnings,
            changes_made=changes_made
        )
    
    def sanitize_html(self, text: str) -> str:
        """
        Escape HTML special characters.
        
        Args:
            text: Text to sanitize
            
        Returns:
            HTML-safe text
        """
        return html.escape(text, quote=True)
    
    def sanitize_sql(self, text: str) -> str:
        """
        Escape SQL special characters.
        
        Args:
            text: Text to sanitize
            
        Returns:
            SQL-safe text (use parameterized queries instead!)
        """
        # Note: This is a fallback - always use parameterized queries
        replacements = {
            "'": "''",
            "\\": "\\\\",
            "\x00": "",
            "\n": "\\n",
            "\r": "\\r",
            "\x1a": "\\Z",
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    def sanitize_path(self, path: str) -> str:
        """
        Remove path traversal sequences.
        
        Args:
            path: Path to sanitize
            
        Returns:
            Safe path
        """
        # Remove directory traversal
        path = re.sub(r'\.\.+[/\\]', '', path)
        path = re.sub(r'%2e%2e[/\\]', '', path, flags=re.IGNORECASE)
        
        # Remove null bytes
        path = path.replace('\x00', '').replace('%00', '')
        
        # Remove leading slashes (prevent absolute paths)
        path = path.lstrip('/\\')
        
        # Remove Windows drive letters
        path = re.sub(r'^[a-zA-Z]:', '', path)
        
        return path
    
    def sanitize_command(self, text: str) -> str:
        """
        Escape shell special characters.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Shell-safe text
        """
        # Remove dangerous characters
        dangerous = ['|', ';', '&', '$', '`', '(', ')', '{', '}', 
                    '[', ']', '!', '#', '~', '<', '>', '\n', '\r']
        
        for char in dangerous:
            text = text.replace(char, '')
        
        return text
    
    def sanitize_ldap(self, text: str) -> str:
        """
        Escape LDAP special characters.
        
        Args:
            text: Text to sanitize
            
        Returns:
            LDAP-safe text
        """
        replacements = {
            '\\': '\\5c',
            '*': '\\2a',
            '(': '\\28',
            ')': '\\29',
            '\x00': '\\00',
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    def sanitize_xml(self, text: str) -> str:
        """
        Escape XML special characters.
        
        Args:
            text: Text to sanitize
            
        Returns:
            XML-safe text
        """
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&apos;',
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    def sanitize_json(self, text: str) -> str:
        """
        Escape JSON special characters.
        
        Args:
            text: Text to sanitize
            
        Returns:
            JSON-safe text
        """
        replacements = {
            '\\': '\\\\',
            '"': '\\"',
            '\n': '\\n',
            '\r': '\\r',
            '\t': '\\t',
            '\b': '\\b',
            '\f': '\\f',
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename.
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Safe filename
        """
        # Remove path components
        filename = filename.replace('/', '_').replace('\\', '_')
        
        # Remove null bytes and dangerous chars
        filename = filename.replace('\x00', '').replace('%00', '')
        
        # Remove or replace illegal filename chars
        illegal = '<>:"|?*'
        for char in illegal:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Prevent reserved names on Windows
        reserved = ['CON', 'PRN', 'AUX', 'NUL'] + \
                  [f'COM{i}' for i in range(1, 10)] + \
                  [f'LPT{i}' for i in range(1, 10)]
        
        name_part = filename.split('.')[0].upper()
        if name_part in reserved:
            filename = '_' + filename
        
        return filename or 'unnamed'
    
    def validate(
        self,
        text: str,
        rules: list[ValidationRule]
    ) -> list[str]:
        """
        Validate text against rules.
        
        Args:
            text: Text to validate
            rules: List of validation rules
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        for rule in rules:
            error = rule.validate(text)
            if error:
                errors.append(error)
        
        return errors
    
    def is_safe_url(self, url: str) -> bool:
        """
        Check if URL is safe (no javascript:, data:, etc.).
        
        Args:
            url: URL to check
            
        Returns:
            True if URL appears safe
        """
        url_lower = url.lower().strip()
        
        dangerous_schemes = [
            'javascript:',
            'vbscript:',
            'data:',
            'file:',
        ]
        
        for scheme in dangerous_schemes:
            if url_lower.startswith(scheme):
                return False
        
        return True
    
    def strip_tags(self, text: str) -> str:
        """
        Remove all HTML tags from text.
        
        Args:
            text: Text with potential HTML
            
        Returns:
            Text with tags removed
        """
        return re.sub(r'<[^>]+>', '', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Text to normalize
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def check_threats(
        self,
        text: str
    ) -> dict[ThreatType, bool]:
        """
        Check which threat types are present.
        
        Args:
            text: Text to check
            
        Returns:
            Dict mapping threat type to presence
        """
        results = {}
        
        # SQL injection
        results[ThreatType.SQL_INJECTION] = any(
            p.search(text) for p in self._sql_patterns
        )
        
        # XSS
        results[ThreatType.XSS] = any(
            p.search(text) for p in self._xss_patterns
        )
        
        # Command injection
        results[ThreatType.COMMAND_INJECTION] = any(
            p.search(text) for p in self._cmd_patterns
        )
        
        # Path traversal
        results[ThreatType.PATH_TRAVERSAL] = any(
            p.search(text) for p in self._path_patterns
        )
        
        # LDAP injection
        results[ThreatType.LDAP_INJECTION] = any(
            p.search(text) for p in self._ldap_patterns
        )
        
        # XML injection
        results[ThreatType.XML_INJECTION] = any(
            p.search(text) for p in self._xml_patterns
        )
        
        # Template injection
        results[ThreatType.TEMPLATE_INJECTION] = any(
            p.search(text) for p in self._template_patterns
        )
        
        # Null bytes
        results[ThreatType.NULL_BYTE] = '\x00' in text or '%00' in text
        
        return results


class ValidationRules:
    """Factory for common validation rules."""
    
    @staticmethod
    def required() -> ValidationRule:
        """Value must not be empty."""
        return ValidationRule(
            name="required",
            check=lambda x: bool(x and x.strip()),
            message="This field is required"
        )
    
    @staticmethod
    def min_length(length: int) -> ValidationRule:
        """Value must be at least n characters."""
        return ValidationRule(
            name=f"min_length_{length}",
            check=lambda x: len(x) >= length,
            message=f"Must be at least {length} characters"
        )
    
    @staticmethod
    def max_length(length: int) -> ValidationRule:
        """Value must not exceed n characters."""
        return ValidationRule(
            name=f"max_length_{length}",
            check=lambda x: len(x) <= length,
            message=f"Must not exceed {length} characters"
        )
    
    @staticmethod
    def alphanumeric() -> ValidationRule:
        """Value must be alphanumeric only."""
        return ValidationRule(
            name="alphanumeric",
            check=lambda x: x.isalnum(),
            message="Must contain only letters and numbers"
        )
    
    @staticmethod
    def alphabetic() -> ValidationRule:
        """Value must be alphabetic only."""
        return ValidationRule(
            name="alphabetic",
            check=lambda x: x.isalpha(),
            message="Must contain only letters"
        )
    
    @staticmethod
    def numeric() -> ValidationRule:
        """Value must be numeric only."""
        return ValidationRule(
            name="numeric",
            check=lambda x: x.isdigit(),
            message="Must contain only numbers"
        )
    
    @staticmethod
    def email() -> ValidationRule:
        """Value must be valid email format."""
        pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        return ValidationRule(
            name="email",
            check=lambda x: bool(pattern.match(x)),
            message="Must be a valid email address"
        )
    
    @staticmethod
    def url() -> ValidationRule:
        """Value must be valid URL format."""
        pattern = re.compile(
            r'^https?://[a-zA-Z0-9][-a-zA-Z0-9]*(\.[a-zA-Z0-9][-a-zA-Z0-9]*)+'
        )
        return ValidationRule(
            name="url",
            check=lambda x: bool(pattern.match(x)),
            message="Must be a valid URL"
        )
    
    @staticmethod
    def pattern(regex: str, message: str = "Invalid format") -> ValidationRule:
        """Value must match regex pattern."""
        compiled = re.compile(regex)
        return ValidationRule(
            name="pattern",
            check=lambda x: bool(compiled.match(x)),
            message=message
        )
    
    @staticmethod
    def no_whitespace() -> ValidationRule:
        """Value must not contain whitespace."""
        return ValidationRule(
            name="no_whitespace",
            check=lambda x: not any(c.isspace() for c in x),
            message="Must not contain spaces"
        )
    
    @staticmethod
    def ascii_only() -> ValidationRule:
        """Value must be ASCII only."""
        return ValidationRule(
            name="ascii_only",
            check=lambda x: x.isascii(),
            message="Must contain only ASCII characters"
        )
    
    @staticmethod
    def printable() -> ValidationRule:
        """Value must be printable characters only."""
        return ValidationRule(
            name="printable",
            check=lambda x: x.isprintable(),
            message="Must contain only printable characters"
        )
    
    @staticmethod
    def no_special_chars(allowed: str = "") -> ValidationRule:
        """Value must not contain special characters."""
        def check(x):
            for char in x:
                if not (char.isalnum() or char in allowed):
                    return False
            return True
        
        return ValidationRule(
            name="no_special_chars",
            check=check,
            message="Must not contain special characters"
        )


# Convenience functions
def get_sanitizer(strict: bool = False) -> InputSanitizer:
    """Get an InputSanitizer instance."""
    return InputSanitizer(strict=strict)


def sanitize(text: str, max_length: Optional[int] = None) -> SanitizationResult:
    """Quick sanitization."""
    return InputSanitizer().sanitize(text, max_length=max_length)


def sanitize_html(text: str) -> str:
    """Quick HTML sanitization."""
    return InputSanitizer().sanitize_html(text)


def sanitize_sql(text: str) -> str:
    """Quick SQL sanitization (use parameterized queries instead!)."""
    return InputSanitizer().sanitize_sql(text)


def sanitize_path(path: str) -> str:
    """Quick path sanitization."""
    return InputSanitizer().sanitize_path(path)


def sanitize_filename(filename: str) -> str:
    """Quick filename sanitization."""
    return InputSanitizer().sanitize_filename(filename)


def is_safe(text: str) -> bool:
    """Quick safety check."""
    return InputSanitizer().sanitize(text).is_safe


def check_threats(text: str) -> dict[ThreatType, bool]:
    """Quick threat check."""
    return InputSanitizer().check_threats(text)
