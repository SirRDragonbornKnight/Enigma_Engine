"""
Regex Helper - Build, explain, test, and validate regex patterns.

Provides tools for working with regex patterns including:
- Pattern building/parsing
- Pattern explanation in human-readable format
- Pattern testing with match highlighting
- Common pattern library
- Pattern validation

Part of the ForgeAI utilities.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class PatternCategory(Enum):
    """Categories for common regex patterns."""
    TEXT = "text"
    NUMBERS = "numbers"
    VALIDATION = "validation"
    EXTRACTION = "extraction"
    FORMATTING = "formatting"
    CUSTOM = "custom"


@dataclass
class MatchResult:
    """Result of a regex match."""
    matched: bool
    full_match: Optional[str] = None
    groups: List[str] = field(default_factory=list)
    named_groups: Dict[str, str] = field(default_factory=dict)
    span: Optional[Tuple[int, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "matched": self.matched,
            "full_match": self.full_match,
            "groups": self.groups,
            "named_groups": self.named_groups,
            "span": self.span
        }


@dataclass
class PatternInfo:
    """Information about a regex pattern."""
    pattern: str
    name: str
    description: str
    category: PatternCategory = PatternCategory.CUSTOM
    example_matches: List[str] = field(default_factory=list)
    example_non_matches: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern": self.pattern,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "example_matches": self.example_matches,
            "example_non_matches": self.example_non_matches
        }


# Token explanations for regex elements
REGEX_TOKENS = {
    # Anchors
    r'^': 'Start of string or line',
    r'$': 'End of string or line',
    r'\A': 'Start of string only',
    r'\Z': 'End of string only',
    r'\b': 'Word boundary',
    r'\B': 'Not a word boundary',
    
    # Character classes
    r'.': 'Any character except newline',
    r'\d': 'Digit (0-9)',
    r'\D': 'Non-digit',
    r'\w': 'Word character (a-z, A-Z, 0-9, _)',
    r'\W': 'Non-word character',
    r'\s': 'Whitespace (space, tab, newline)',
    r'\S': 'Non-whitespace',
    
    # Quantifiers
    r'*': 'Zero or more',
    r'+': 'One or more',
    r'?': 'Zero or one (optional)',
    
    # Groups
    r'(': 'Start capturing group',
    r')': 'End capturing group',
    r'(?:': 'Start non-capturing group',
    r'(?=': 'Positive lookahead',
    r'(?!': 'Negative lookahead',
    r'(?<=': 'Positive lookbehind',
    r'(?<!': 'Negative lookbehind',
    
    # Operators
    r'|': 'Alternation (OR)',
    r'\\': 'Escape character',
}


# Common patterns library
COMMON_PATTERNS: Dict[str, PatternInfo] = {
    "email": PatternInfo(
        pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        name="Email Address",
        description="Matches standard email addresses",
        category=PatternCategory.VALIDATION,
        example_matches=["user@example.com", "test.name+tag@domain.org"],
        example_non_matches=["not-an-email", "@missing.com", "missing@"]
    ),
    "phone_us": PatternInfo(
        pattern=r'^\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$',
        name="US Phone Number",
        description="Matches US phone numbers in various formats",
        category=PatternCategory.VALIDATION,
        example_matches=["(555) 123-4567", "555-123-4567", "5551234567"],
        example_non_matches=["123-4567", "555-1234-567"]
    ),
    "url": PatternInfo(
        pattern=r'^https?://[a-zA-Z0-9][-a-zA-Z0-9]*(\.[a-zA-Z0-9][-a-zA-Z0-9]*)+(/[-a-zA-Z0-9._~:/?#\[\]@!$&\'()*+,;=%]*)?$',
        name="URL",
        description="Matches HTTP/HTTPS URLs",
        category=PatternCategory.VALIDATION,
        example_matches=["https://example.com", "http://sub.domain.org/path?q=1"],
        example_non_matches=["ftp://file.com", "not a url"]
    ),
    "ipv4": PatternInfo(
        pattern=r'^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$',
        name="IPv4 Address",
        description="Matches valid IPv4 addresses",
        category=PatternCategory.VALIDATION,
        example_matches=["192.168.1.1", "10.0.0.255", "0.0.0.0"],
        example_non_matches=["256.1.1.1", "192.168.1", "abc.def.ghi.jkl"]
    ),
    "date_iso": PatternInfo(
        pattern=r'^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$',
        name="ISO Date (YYYY-MM-DD)",
        description="Matches dates in ISO format",
        category=PatternCategory.VALIDATION,
        example_matches=["2024-01-15", "1999-12-31"],
        example_non_matches=["2024/01/15", "15-01-2024", "2024-13-01"]
    ),
    "date_us": PatternInfo(
        pattern=r'^(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/\d{4}$',
        name="US Date (MM/DD/YYYY)",
        description="Matches dates in US format",
        category=PatternCategory.VALIDATION,
        example_matches=["01/15/2024", "12/31/1999"],
        example_non_matches=["2024-01-15", "1/15/2024"]
    ),
    "time_24h": PatternInfo(
        pattern=r'^(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?$',
        name="24-Hour Time",
        description="Matches time in 24-hour format",
        category=PatternCategory.VALIDATION,
        example_matches=["14:30", "23:59:59", "00:00"],
        example_non_matches=["25:00", "2:30 PM"]
    ),
    "hex_color": PatternInfo(
        pattern=r'^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$',
        name="Hex Color Code",
        description="Matches CSS hex color codes",
        category=PatternCategory.VALIDATION,
        example_matches=["#fff", "#FF5733", "#000000"],
        example_non_matches=["fff", "#GGGGGG", "#12345"]
    ),
    "credit_card": PatternInfo(
        pattern=r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$',
        name="Credit Card Number",
        description="Matches 16-digit credit card numbers",
        category=PatternCategory.VALIDATION,
        example_matches=["1234 5678 9012 3456", "1234-5678-9012-3456", "1234567890123456"],
        example_non_matches=["1234 5678 9012", "12345678901234567"]
    ),
    "ssn": PatternInfo(
        pattern=r'^\d{3}-\d{2}-\d{4}$',
        name="Social Security Number",
        description="Matches US SSN format",
        category=PatternCategory.VALIDATION,
        example_matches=["123-45-6789"],
        example_non_matches=["123456789", "12-345-6789"]
    ),
    "zip_us": PatternInfo(
        pattern=r'^\d{5}(?:-\d{4})?$',
        name="US ZIP Code",
        description="Matches 5-digit and ZIP+4 codes",
        category=PatternCategory.VALIDATION,
        example_matches=["12345", "12345-6789"],
        example_non_matches=["1234", "123456"]
    ),
    "username": PatternInfo(
        pattern=r'^[a-zA-Z][a-zA-Z0-9_-]{2,31}$',
        name="Username",
        description="3-32 chars, starts with letter, allows letters/numbers/underscore/dash",
        category=PatternCategory.VALIDATION,
        example_matches=["john_doe", "User123", "test-user"],
        example_non_matches=["1user", "ab", "a" * 33]
    ),
    "password_strong": PatternInfo(
        pattern=r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$',
        name="Strong Password",
        description="8+ chars with uppercase, lowercase, digit, and special char",
        category=PatternCategory.VALIDATION,
        example_matches=["Password1!", "Str0ng@Pass"],
        example_non_matches=["password", "PASSWORD1", "Password"]
    ),
    "slug": PatternInfo(
        pattern=r'^[a-z0-9]+(?:-[a-z0-9]+)*$',
        name="URL Slug",
        description="URL-friendly slug format",
        category=PatternCategory.VALIDATION,
        example_matches=["hello-world", "my-post-123"],
        example_non_matches=["Hello World", "-invalid", "invalid-"]
    ),
    "uuid": PatternInfo(
        pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        name="UUID",
        description="Matches UUID format",
        category=PatternCategory.VALIDATION,
        example_matches=["550e8400-e29b-41d4-a716-446655440000"],
        example_non_matches=["550e8400-e29b-41d4-a716"]
    ),
    "integer": PatternInfo(
        pattern=r'^-?\d+$',
        name="Integer",
        description="Matches positive and negative integers",
        category=PatternCategory.NUMBERS,
        example_matches=["123", "-456", "0"],
        example_non_matches=["12.34", "abc"]
    ),
    "decimal": PatternInfo(
        pattern=r'^-?\d+\.?\d*$',
        name="Decimal Number",
        description="Matches decimal numbers",
        category=PatternCategory.NUMBERS,
        example_matches=["123.45", "-67.89", "100"],
        example_non_matches=["abc", "1.2.3"]
    ),
    "scientific": PatternInfo(
        pattern=r'^-?\d+\.?\d*[eE][+-]?\d+$',
        name="Scientific Notation",
        description="Matches numbers in scientific notation",
        category=PatternCategory.NUMBERS,
        example_matches=["1.5e10", "2.5E-3", "-1e5"],
        example_non_matches=["1.5", "e10"]
    ),
    "html_tag": PatternInfo(
        pattern=r'<([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>.*?</\1>|<([a-zA-Z][a-zA-Z0-9]*)\b[^>]*/>',
        name="HTML Tag",
        description="Matches HTML tags with content or self-closing",
        category=PatternCategory.EXTRACTION,
        example_matches=["<p>text</p>", "<br/>", "<div class='test'>content</div>"],
        example_non_matches=["<>", "</p>"]
    ),
    "whitespace_trim": PatternInfo(
        pattern=r'^\s+|\s+$',
        name="Leading/Trailing Whitespace",
        description="Matches whitespace at start or end of string",
        category=PatternCategory.FORMATTING,
        example_matches=["  text", "text  ", "  "],
        example_non_matches=["text", "no whitespace"]
    ),
    "multiple_spaces": PatternInfo(
        pattern=r'\s{2,}',
        name="Multiple Spaces",
        description="Matches 2 or more consecutive whitespace chars",
        category=PatternCategory.FORMATTING,
        example_matches=["hello  world", "too   many    spaces"],
        example_non_matches=["single space", "no_space"]
    ),
}


class RegexHelper:
    """
    Helper class for building, testing, and explaining regex patterns.
    
    Usage:
        helper = RegexHelper()
        
        # Test a pattern
        result = helper.test(r'\\d+', '123')
        
        # Explain a pattern
        explanation = helper.explain(r'^\\d{3}-\\d{4}$')
        
        # Find all matches
        matches = helper.find_all(r'\\w+@\\w+', 'email@test another@domain')
        
        # Get a common pattern
        email_pattern = helper.get_pattern('email')
        
        # Build a pattern
        pattern = helper.build_pattern()
            .starts_with()
            .digits(3)
            .literal('-')
            .digits(4)
            .ends_with()
            .get()
    """
    
    def __init__(self):
        """Initialize the regex helper."""
        self.patterns = COMMON_PATTERNS.copy()
        self._custom_patterns: Dict[str, PatternInfo] = {}
    
    def test(
        self,
        pattern: str,
        text: str,
        flags: int = 0
    ) -> MatchResult:
        """
        Test if a pattern matches text.
        
        Args:
            pattern: Regex pattern
            text: Text to test
            flags: Regex flags (re.IGNORECASE, etc.)
            
        Returns:
            MatchResult with match details
        """
        try:
            compiled = re.compile(pattern, flags)
            match = compiled.match(text)
            
            if match:
                return MatchResult(
                    matched=True,
                    full_match=match.group(0),
                    groups=list(match.groups()) if match.groups() else [],
                    named_groups=match.groupdict() or {},
                    span=match.span()
                )
            return MatchResult(matched=False)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
    
    def search(
        self,
        pattern: str,
        text: str,
        flags: int = 0
    ) -> MatchResult:
        """
        Search for pattern anywhere in text.
        
        Args:
            pattern: Regex pattern
            text: Text to search
            flags: Regex flags
            
        Returns:
            MatchResult with first match details
        """
        try:
            compiled = re.compile(pattern, flags)
            match = compiled.search(text)
            
            if match:
                return MatchResult(
                    matched=True,
                    full_match=match.group(0),
                    groups=list(match.groups()) if match.groups() else [],
                    named_groups=match.groupdict() or {},
                    span=match.span()
                )
            return MatchResult(matched=False)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
    
    def find_all(
        self,
        pattern: str,
        text: str,
        flags: int = 0
    ) -> List[MatchResult]:
        """
        Find all matches in text.
        
        Args:
            pattern: Regex pattern
            text: Text to search
            flags: Regex flags
            
        Returns:
            List of MatchResult for each match
        """
        try:
            compiled = re.compile(pattern, flags)
            results = []
            
            for match in compiled.finditer(text):
                results.append(MatchResult(
                    matched=True,
                    full_match=match.group(0),
                    groups=list(match.groups()) if match.groups() else [],
                    named_groups=match.groupdict() or {},
                    span=match.span()
                ))
            
            return results
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
    
    def replace(
        self,
        pattern: str,
        text: str,
        replacement: str,
        flags: int = 0,
        count: int = 0
    ) -> str:
        """
        Replace pattern matches in text.
        
        Args:
            pattern: Regex pattern
            text: Text to modify
            replacement: Replacement string
            flags: Regex flags
            count: Max replacements (0 = all)
            
        Returns:
            Modified text
        """
        try:
            return re.sub(pattern, replacement, text, count=count, flags=flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
    
    def split(
        self,
        pattern: str,
        text: str,
        flags: int = 0,
        maxsplit: int = 0
    ) -> List[str]:
        """
        Split text by pattern.
        
        Args:
            pattern: Regex pattern
            text: Text to split
            flags: Regex flags
            maxsplit: Max splits (0 = all)
            
        Returns:
            List of string parts
        """
        try:
            return re.split(pattern, text, maxsplit=maxsplit, flags=flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
    
    def explain(self, pattern: str) -> str:
        """
        Explain a regex pattern in human-readable format.
        
        Args:
            pattern: Regex pattern to explain
            
        Returns:
            Human-readable explanation
        """
        explanations = []
        i = 0
        
        while i < len(pattern):
            # Check for multi-char tokens first
            found_token = False
            
            for length in range(4, 0, -1):
                token = pattern[i:i+length]
                
                # Handle escaped characters
                if token.startswith('\\') and len(token) >= 2:
                    if token[:2] in REGEX_TOKENS:
                        explanations.append(f"'{token[:2]}' - {REGEX_TOKENS[token[:2]]}")
                        i += 2
                        found_token = True
                        break
                
                # Handle special sequences
                if token in REGEX_TOKENS:
                    explanations.append(f"'{token}' - {REGEX_TOKENS[token]}")
                    i += len(token)
                    found_token = True
                    break
            
            if found_token:
                continue
            
            char = pattern[i]
            
            # Character classes [...]
            if char == '[':
                end = pattern.find(']', i + 1)
                if end != -1:
                    char_class = pattern[i:end+1]
                    negated = char_class[1] == '^' if len(char_class) > 2 else False
                    inner = char_class[2:-1] if negated else char_class[1:-1]
                    
                    if negated:
                        explanations.append(f"'{char_class}' - Any character NOT in: {inner}")
                    else:
                        explanations.append(f"'{char_class}' - Any character in: {inner}")
                    i = end + 1
                    continue
            
            # Quantifiers with range
            if char == '{':
                end = pattern.find('}', i + 1)
                if end != -1:
                    quantifier = pattern[i:end+1]
                    inner = quantifier[1:-1]
                    if ',' in inner:
                        parts = inner.split(',')
                        min_val = parts[0] or '0'
                        max_val = parts[1] if len(parts) > 1 and parts[1] else 'unlimited'
                        explanations.append(f"'{quantifier}' - Between {min_val} and {max_val} times")
                    else:
                        explanations.append(f"'{quantifier}' - Exactly {inner} times")
                    i = end + 1
                    continue
            
            # Single char tokens
            if char in REGEX_TOKENS:
                explanations.append(f"'{char}' - {REGEX_TOKENS[char]}")
            elif char == '\\' and i + 1 < len(pattern):
                next_char = pattern[i + 1]
                explanations.append(f"'\\{next_char}' - Literal '{next_char}'")
                i += 1
            else:
                explanations.append(f"'{char}' - Literal character")
            
            i += 1
        
        return '\n'.join(explanations)
    
    def validate(self, pattern: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a pattern is valid regex.
        
        Args:
            pattern: Regex pattern to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            re.compile(pattern)
            return True, None
        except re.error as e:
            return False, str(e)
    
    def get_pattern(self, name: str) -> Optional[PatternInfo]:
        """
        Get a pattern from the library.
        
        Args:
            name: Pattern name (e.g., 'email', 'phone_us')
            
        Returns:
            PatternInfo or None if not found
        """
        return self.patterns.get(name) or self._custom_patterns.get(name)
    
    def list_patterns(
        self,
        category: Optional[PatternCategory] = None
    ) -> List[PatternInfo]:
        """
        List available patterns.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of PatternInfo
        """
        all_patterns = list(self.patterns.values()) + list(self._custom_patterns.values())
        
        if category:
            return [p for p in all_patterns if p.category == category]
        return all_patterns
    
    def add_pattern(
        self,
        name: str,
        pattern: str,
        description: str,
        category: PatternCategory = PatternCategory.CUSTOM,
        example_matches: Optional[List[str]] = None,
        example_non_matches: Optional[List[str]] = None
    ) -> PatternInfo:
        """
        Add a custom pattern to the library.
        
        Args:
            name: Unique pattern name
            pattern: Regex pattern
            description: Human-readable description
            category: Pattern category
            example_matches: Example strings that match
            example_non_matches: Example strings that don't match
            
        Returns:
            Created PatternInfo
        """
        # Validate pattern
        valid, error = self.validate(pattern)
        if not valid:
            raise ValueError(f"Invalid pattern: {error}")
        
        info = PatternInfo(
            pattern=pattern,
            name=name,
            description=description,
            category=category,
            example_matches=example_matches or [],
            example_non_matches=example_non_matches or []
        )
        
        self._custom_patterns[name] = info
        return info
    
    def highlight_matches(
        self,
        pattern: str,
        text: str,
        highlight_start: str = ">>",
        highlight_end: str = "<<",
        flags: int = 0
    ) -> str:
        """
        Highlight matches in text.
        
        Args:
            pattern: Regex pattern
            text: Text to search
            highlight_start: Start marker
            highlight_end: End marker
            flags: Regex flags
            
        Returns:
            Text with matches highlighted
        """
        try:
            def replacer(match):
                return f"{highlight_start}{match.group(0)}{highlight_end}"
            
            return re.sub(pattern, replacer, text, flags=flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
    
    def build_pattern(self) -> 'PatternBuilder':
        """
        Start building a pattern with fluent interface.
        
        Returns:
            PatternBuilder instance
        """
        return PatternBuilder()
    
    def escape(self, text: str) -> str:
        """
        Escape special regex characters in text.
        
        Args:
            text: Text to escape
            
        Returns:
            Escaped text safe for use in regex
        """
        return re.escape(text)


class PatternBuilder:
    """
    Fluent interface for building regex patterns.
    
    Usage:
        pattern = PatternBuilder()
            .starts_with()
            .literal('Hello')
            .whitespace()
            .word_chars(1, None)
            .ends_with()
            .get()
    """
    
    def __init__(self):
        """Initialize the builder."""
        self._parts: List[str] = []
    
    def get(self) -> str:
        """Get the built pattern."""
        return ''.join(self._parts)
    
    def raw(self, pattern: str) -> 'PatternBuilder':
        """Add raw pattern text."""
        self._parts.append(pattern)
        return self
    
    def literal(self, text: str) -> 'PatternBuilder':
        """Add literal text (escaped)."""
        self._parts.append(re.escape(text))
        return self
    
    def starts_with(self) -> 'PatternBuilder':
        """Match start of string."""
        self._parts.append('^')
        return self
    
    def ends_with(self) -> 'PatternBuilder':
        """Match end of string."""
        self._parts.append('$')
        return self
    
    def any_char(self) -> 'PatternBuilder':
        """Match any single character."""
        self._parts.append('.')
        return self
    
    def digit(self) -> 'PatternBuilder':
        """Match single digit."""
        self._parts.append(r'\d')
        return self
    
    def digits(
        self,
        min_count: int = 1,
        max_count: Optional[int] = None
    ) -> 'PatternBuilder':
        """Match multiple digits."""
        self._parts.append(r'\d')
        self._add_quantifier(min_count, max_count)
        return self
    
    def non_digit(self) -> 'PatternBuilder':
        """Match non-digit."""
        self._parts.append(r'\D')
        return self
    
    def word_char(self) -> 'PatternBuilder':
        """Match single word character."""
        self._parts.append(r'\w')
        return self
    
    def word_chars(
        self,
        min_count: int = 1,
        max_count: Optional[int] = None
    ) -> 'PatternBuilder':
        """Match multiple word characters."""
        self._parts.append(r'\w')
        self._add_quantifier(min_count, max_count)
        return self
    
    def non_word_char(self) -> 'PatternBuilder':
        """Match non-word character."""
        self._parts.append(r'\W')
        return self
    
    def whitespace(self) -> 'PatternBuilder':
        """Match single whitespace."""
        self._parts.append(r'\s')
        return self
    
    def whitespaces(
        self,
        min_count: int = 1,
        max_count: Optional[int] = None
    ) -> 'PatternBuilder':
        """Match multiple whitespace."""
        self._parts.append(r'\s')
        self._add_quantifier(min_count, max_count)
        return self
    
    def non_whitespace(self) -> 'PatternBuilder':
        """Match non-whitespace."""
        self._parts.append(r'\S')
        return self
    
    def word_boundary(self) -> 'PatternBuilder':
        """Match word boundary."""
        self._parts.append(r'\b')
        return self
    
    def optional(self) -> 'PatternBuilder':
        """Make previous element optional."""
        self._parts.append('?')
        return self
    
    def zero_or_more(self) -> 'PatternBuilder':
        """Match previous element zero or more times."""
        self._parts.append('*')
        return self
    
    def one_or_more(self) -> 'PatternBuilder':
        """Match previous element one or more times."""
        self._parts.append('+')
        return self
    
    def exactly(self, count: int) -> 'PatternBuilder':
        """Match previous element exactly n times."""
        self._parts.append(f'{{{count}}}')
        return self
    
    def between(
        self,
        min_count: int,
        max_count: Optional[int] = None
    ) -> 'PatternBuilder':
        """Match previous element between min and max times."""
        self._add_quantifier(min_count, max_count)
        return self
    
    def group(self, *patterns: str) -> 'PatternBuilder':
        """Create capturing group."""
        self._parts.append('(' + ''.join(patterns) + ')')
        return self
    
    def named_group(self, name: str, *patterns: str) -> 'PatternBuilder':
        """Create named capturing group."""
        self._parts.append(f'(?P<{name}>' + ''.join(patterns) + ')')
        return self
    
    def non_capturing_group(self, *patterns: str) -> 'PatternBuilder':
        """Create non-capturing group."""
        self._parts.append('(?:' + ''.join(patterns) + ')')
        return self
    
    def either(self, *options: str) -> 'PatternBuilder':
        """Match any of the options."""
        self._parts.append('(?:' + '|'.join(options) + ')')
        return self
    
    def char_class(self, chars: str, negate: bool = False) -> 'PatternBuilder':
        """Create character class."""
        if negate:
            self._parts.append(f'[^{chars}]')
        else:
            self._parts.append(f'[{chars}]')
        return self
    
    def range(
        self,
        start: str,
        end: str,
        negate: bool = False
    ) -> 'PatternBuilder':
        """Create character range."""
        if negate:
            self._parts.append(f'[^{start}-{end}]')
        else:
            self._parts.append(f'[{start}-{end}]')
        return self
    
    def lookahead(self, pattern: str, positive: bool = True) -> 'PatternBuilder':
        """Add lookahead assertion."""
        if positive:
            self._parts.append(f'(?={pattern})')
        else:
            self._parts.append(f'(?!{pattern})')
        return self
    
    def lookbehind(self, pattern: str, positive: bool = True) -> 'PatternBuilder':
        """Add lookbehind assertion."""
        if positive:
            self._parts.append(f'(?<={pattern})')
        else:
            self._parts.append(f'(?<!{pattern})')
        return self
    
    def _add_quantifier(
        self,
        min_count: int,
        max_count: Optional[int]
    ) -> None:
        """Add quantifier to pattern."""
        if min_count == 0 and max_count is None:
            self._parts.append('*')
        elif min_count == 1 and max_count is None:
            self._parts.append('+')
        elif min_count == 0 and max_count == 1:
            self._parts.append('?')
        elif max_count is None:
            self._parts.append(f'{{{min_count},}}')
        elif min_count == max_count:
            self._parts.append(f'{{{min_count}}}')
        else:
            self._parts.append(f'{{{min_count},{max_count}}}')


# Convenience functions
def get_regex_helper() -> RegexHelper:
    """Get a RegexHelper instance."""
    return RegexHelper()


def test_pattern(pattern: str, text: str, flags: int = 0) -> MatchResult:
    """Quick pattern test."""
    return RegexHelper().test(pattern, text, flags)


def explain_pattern(pattern: str) -> str:
    """Quick pattern explanation."""
    return RegexHelper().explain(pattern)


def validate_pattern(pattern: str) -> Tuple[bool, Optional[str]]:
    """Quick pattern validation."""
    return RegexHelper().validate(pattern)


def get_common_pattern(name: str) -> Optional[str]:
    """Get a common pattern by name."""
    info = COMMON_PATTERNS.get(name)
    return info.pattern if info else None


def build_pattern() -> PatternBuilder:
    """Start building a pattern."""
    return PatternBuilder()
