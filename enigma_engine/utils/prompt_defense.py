"""
Prompt Injection Defense for Enigma AI Engine

Detect and prevent prompt injection attacks.

Features:
- Input sanitization
- Injection pattern detection
- Jailbreak detection
- Output validation
- Security logging

Usage:
    from enigma_engine.utils.prompt_defense import PromptDefender, get_defender
    
    defender = get_defender()
    
    # Check input
    result = defender.check_input("Ignore previous instructions...")
    if result.is_suspicious:
        print(f"Blocked: {result.reason}")
    
    # Sanitize input
    safe_input = defender.sanitize(user_input)
    
    # Validate output
    is_safe = defender.validate_output(model_output)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class CheckResult:
    """Result of a security check."""
    is_suspicious: bool
    threat_level: ThreatLevel
    reason: str = ""
    matched_patterns: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_suspicious": self.is_suspicious,
            "threat_level": self.threat_level.name,
            "reason": self.reason,
            "matched_patterns": self.matched_patterns,
            "confidence": self.confidence
        }


@dataclass
class SecurityEvent:
    """Security event for logging."""
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    details: Dict[str, Any]
    blocked: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "threat_level": self.threat_level.name,
            "details": self.details,
            "blocked": self.blocked
        }


class PatternMatcher:
    """Matches known injection patterns."""
    
    # Common injection patterns
    INJECTION_PATTERNS = [
        # Instruction override attempts
        (r"ignore\s+(all\s+)?(previous|prior|above)", ThreatLevel.HIGH, "Instruction override"),
        (r"disregard\s+(all\s+)?(previous|prior|above)", ThreatLevel.HIGH, "Instruction override"),
        (r"forget\s+(all\s+)?(previous|prior|above)", ThreatLevel.HIGH, "Instruction override"),
        
        # New instruction injection
        (r"new\s+instructions?\s*:", ThreatLevel.HIGH, "New instruction injection"),
        (r"instead\s*,?\s+you\s+(must|should|will)", ThreatLevel.HIGH, "Instruction replacement"),
        (r"your\s+new\s+(task|job|role|purpose)", ThreatLevel.HIGH, "Role replacement"),
        
        # System prompt extraction
        (r"(show|reveal|display|print|output)\s+(your\s+)?(system\s+)?prompt", ThreatLevel.CRITICAL, "Prompt extraction"),
        (r"what\s+(are\s+)?your\s+(initial\s+)?instructions", ThreatLevel.MEDIUM, "Instruction probing"),
        (r"repeat\s+(your\s+)?(system\s+)?prompt", ThreatLevel.HIGH, "Prompt extraction"),
        
        # Roleplay manipulation
        (r"you\s+are\s+now\s+", ThreatLevel.MEDIUM, "Role manipulation"),
        (r"pretend\s+(to\s+be|you\s+are)", ThreatLevel.MEDIUM, "Role manipulation"),
        (r"act\s+as\s+(if\s+you\s+are|a)", ThreatLevel.LOW, "Role manipulation"),
        
        # DAN-style jailbreaks
        (r"\bDAN\b", ThreatLevel.HIGH, "DAN jailbreak"),
        (r"developer\s+mode", ThreatLevel.HIGH, "Developer mode jailbreak"),
        (r"jailbreak", ThreatLevel.CRITICAL, "Jailbreak keyword"),
        
        # Encoding attacks
        (r"base64\s*:", ThreatLevel.MEDIUM, "Encoding attack"),
        (r"decode\s+(this|the\s+following)", ThreatLevel.LOW, "Decoding request"),
        
        # Delimiter manipulation
        (r"\[\s*INST\s*\]", ThreatLevel.HIGH, "Delimiter injection"),
        (r"<\s*/?\s*system\s*>", ThreatLevel.HIGH, "XML delimiter injection"),
        (r"\{\s*\{[\s\S]*\}\s*\}", ThreatLevel.MEDIUM, "Template injection"),
        
        # Command injection
        (r"execute\s+(the\s+)?(following\s+)?(command|code)", ThreatLevel.MEDIUM, "Command injection"),
        (r"\$\([\s\S]+\)", ThreatLevel.HIGH, "Shell injection"),
        (r"`[\s\S]+`", ThreatLevel.LOW, "Backtick injection"),
        
        # Data exfiltration
        (r"(send|post|transmit)\s+(data|information)\s+to", ThreatLevel.HIGH, "Data exfiltration"),
        (r"(connect|call|fetch)\s+(to\s+)?https?://", ThreatLevel.MEDIUM, "URL request"),
    ]
    
    # Jailbreak indicators (less specific patterns)
    JAILBREAK_INDICATORS = [
        "bypass",
        "restrictions",
        "limitations",
        "unfiltered",
        "uncensored",
        "no limits",
        "without restrictions",
        "override safety",
        "disable filters",
        "ignore rules"
    ]
    
    def __init__(self):
        # Compile patterns
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), level, desc)
            for pattern, level, desc in self.INJECTION_PATTERNS
        ]
    
    def match(self, text: str) -> List[Tuple[str, ThreatLevel, str]]:
        """
        Match text against injection patterns.
        
        Args:
            text: Text to check
            
        Returns:
            List of (matched_text, threat_level, description)
        """
        matches = []
        
        for pattern, level, desc in self._compiled_patterns:
            found = pattern.search(text)
            if found:
                matches.append((found.group(), level, desc))
        
        # Check jailbreak indicators
        text_lower = text.lower()
        indicator_count = sum(1 for ind in self.JAILBREAK_INDICATORS if ind in text_lower)
        
        if indicator_count >= 3:
            matches.append(("Multiple jailbreak indicators", ThreatLevel.HIGH, "Jailbreak attempt"))
        elif indicator_count >= 2:
            matches.append(("Jailbreak indicators", ThreatLevel.MEDIUM, "Potential jailbreak"))
        
        return matches


class InputSanitizer:
    """Sanitizes user input."""
    
    # Characters that should be escaped/removed
    DANGEROUS_CHARS = {
        "\x00": "",  # Null byte
        "\x1b": "",  # Escape
        "\r": "\n",  # Carriage return
    }
    
    # Delimiter sequences to neutralize
    DELIMITERS = [
        "```",
        "###",
        "---",
        "***",
        "[INST]",
        "[/INST]",
        "<|",
        "|>",
        "<s>",
        "</s>",
        "<<SYS>>",
        "<</SYS>>",
    ]
    
    def __init__(self):
        self._custom_replacements: Dict[str, str] = {}
    
    def sanitize(self, text: str, strict: bool = False) -> str:
        """
        Sanitize input text.
        
        Args:
            text: Text to sanitize
            strict: Use strict mode (more aggressive)
            
        Returns:
            Sanitized text
        """
        result = text
        
        # Remove dangerous characters
        for char, replacement in self.DANGEROUS_CHARS.items():
            result = result.replace(char, replacement)
        
        # Neutralize delimiters
        for delimiter in self.DELIMITERS:
            result = result.replace(delimiter, f" {delimiter} ")
        
        # Apply custom replacements
        for pattern, replacement in self._custom_replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        if strict:
            # Remove any remaining control characters
            result = "".join(c for c in result if c.isprintable() or c in "\n\t")
            
            # Limit consecutive newlines
            result = re.sub(r"\n{3,}", "\n\n", result)
            
            # Limit length
            if len(result) > 10000:
                result = result[:10000] + "..."
        
        return result.strip()
    
    def add_replacement(self, pattern: str, replacement: str):
        """Add custom replacement pattern."""
        self._custom_replacements[pattern] = replacement


class OutputValidator:
    """Validates model output."""
    
    # Patterns that shouldn't appear in output
    DANGEROUS_OUTPUT_PATTERNS = [
        # Revealed system prompts
        (r"my\s+(system\s+)?prompt\s+is", ThreatLevel.HIGH),
        (r"my\s+(initial\s+)?instructions\s+(are|were)", ThreatLevel.HIGH),
        
        # Harmful content indicators
        (r"here\s+is\s+(how\s+to|the\s+)?(bomb|weapon|attack)", ThreatLevel.CRITICAL),
        (r"step\s*\d+\s*:.*\b(kill|harm|attack|hack)\b", ThreatLevel.HIGH),
        
        # Data leakage
        (r"api[_-]?key\s*[:=]", ThreatLevel.HIGH),
        (r"password\s*[:=]", ThreatLevel.MEDIUM),
        (r"secret\s*[:=]", ThreatLevel.MEDIUM),
    ]
    
    def __init__(self):
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), level)
            for pattern, level in self.DANGEROUS_OUTPUT_PATTERNS
        ]
        
        self._custom_validators: List[Callable[[str], Optional[str]]] = []
    
    def validate(self, output: str) -> CheckResult:
        """
        Validate model output.
        
        Args:
            output: Model output to validate
            
        Returns:
            Check result
        """
        matches = []
        max_level = ThreatLevel.NONE
        
        for pattern, level in self._compiled_patterns:
            if pattern.search(output):
                matches.append(pattern.pattern)
                if level.value > max_level.value:
                    max_level = level
        
        # Run custom validators
        for validator in self._custom_validators:
            reason = validator(output)
            if reason:
                matches.append(reason)
                max_level = ThreatLevel.MEDIUM
        
        return CheckResult(
            is_suspicious=len(matches) > 0,
            threat_level=max_level,
            reason="Dangerous content detected" if matches else "",
            matched_patterns=matches,
            confidence=min(1.0, len(matches) * 0.3)
        )
    
    def add_validator(self, validator: Callable[[str], Optional[str]]):
        """Add custom output validator."""
        self._custom_validators.append(validator)


class SecurityLogger:
    """Logs security events."""
    
    def __init__(self, log_path: Optional[str] = None):
        self._log_path = Path(log_path) if log_path else None
        self._events: List[SecurityEvent] = []
        self._max_events = 1000
    
    def log(self, event: SecurityEvent):
        """Log a security event."""
        self._events.append(event)
        
        # Trim old events
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        
        # Log to file
        if self._log_path:
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(event.to_dict()) + "\n")
            except Exception as e:
                logger.error(f"Failed to write security log: {e}")
        
        # Log critical events
        if event.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            logger.warning(f"Security event: {event.event_type} - {event.details}")
    
    def get_events(
        self,
        threat_level: Optional[ThreatLevel] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """Get recent security events."""
        events = self._events
        
        if threat_level:
            events = [e for e in events if e.threat_level == threat_level]
        
        return events[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get security statistics."""
        total = len(self._events)
        blocked = sum(1 for e in self._events if e.blocked)
        
        by_level = {}
        for level in ThreatLevel:
            by_level[level.name] = sum(1 for e in self._events if e.threat_level == level)
        
        return {
            "total_events": total,
            "blocked_events": blocked,
            "by_threat_level": by_level
        }


class PromptDefender:
    """
    Main prompt injection defense system.
    """
    
    def __init__(
        self,
        strict_mode: bool = False,
        log_path: Optional[str] = None,
        block_threshold: ThreatLevel = ThreatLevel.HIGH
    ):
        """
        Initialize defender.
        
        Args:
            strict_mode: Enable strict sanitization
            log_path: Path for security logs
            block_threshold: Minimum level to block
        """
        self._strict_mode = strict_mode
        self._block_threshold = block_threshold
        
        self._pattern_matcher = PatternMatcher()
        self._sanitizer = InputSanitizer()
        self._output_validator = OutputValidator()
        self._logger = SecurityLogger(log_path)
        
        # Callbacks
        self._alert_callbacks: List[Callable[[SecurityEvent], None]] = []
    
    def check_input(self, text: str) -> CheckResult:
        """
        Check input for injection attempts.
        
        Args:
            text: User input to check
            
        Returns:
            Check result
        """
        matches = self._pattern_matcher.match(text)
        
        if not matches:
            return CheckResult(
                is_suspicious=False,
                threat_level=ThreatLevel.NONE
            )
        
        # Get highest threat level
        max_level = max(m[1] for m in matches)
        matched_patterns = [m[2] for m in matches]
        
        # Calculate confidence
        confidence = min(1.0, len(matches) * 0.25)
        
        result = CheckResult(
            is_suspicious=True,
            threat_level=max_level,
            reason=", ".join(matched_patterns),
            matched_patterns=matched_patterns,
            confidence=confidence
        )
        
        # Log the event
        event = SecurityEvent(
            timestamp=datetime.now().timestamp(),
            event_type="input_check",
            threat_level=max_level,
            details={
                "text_length": len(text),
                "matches": matched_patterns
            },
            blocked=max_level.value >= self._block_threshold.value
        )
        self._logger.log(event)
        
        # Alert if critical
        if max_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            self._alert(event)
        
        return result
    
    def sanitize(self, text: str) -> str:
        """
        Sanitize user input.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        return self._sanitizer.sanitize(text, self._strict_mode)
    
    def validate_output(self, output: str) -> CheckResult:
        """
        Validate model output.
        
        Args:
            output: Model output
            
        Returns:
            Check result
        """
        result = self._output_validator.validate(output)
        
        if result.is_suspicious:
            event = SecurityEvent(
                timestamp=datetime.now().timestamp(),
                event_type="output_validation",
                threat_level=result.threat_level,
                details={
                    "output_length": len(output),
                    "patterns": result.matched_patterns
                },
                blocked=result.threat_level.value >= self._block_threshold.value
            )
            self._logger.log(event)
        
        return result
    
    def process_input(self, text: str) -> Tuple[str, CheckResult]:
        """
        Full input processing pipeline.
        
        Args:
            text: User input
            
        Returns:
            Tuple of (sanitized_text, check_result)
        """
        # First check for injection
        check_result = self.check_input(text)
        
        # If blocked, return empty
        if check_result.threat_level.value >= self._block_threshold.value:
            return "", check_result
        
        # Sanitize
        sanitized = self.sanitize(text)
        
        return sanitized, check_result
    
    def should_block(self, result: CheckResult) -> bool:
        """Check if result should block processing."""
        return result.threat_level.value >= self._block_threshold.value
    
    def add_alert_callback(self, callback: Callable[[SecurityEvent], None]):
        """Add security alert callback."""
        self._alert_callbacks.append(callback)
    
    def _alert(self, event: SecurityEvent):
        """Trigger alert callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get defense statistics."""
        return self._logger.get_statistics()
    
    def get_events(
        self,
        threat_level: Optional[ThreatLevel] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent security events."""
        events = self._logger.get_events(threat_level, limit)
        return [e.to_dict() for e in events]


# Global instance
_defender: Optional[PromptDefender] = None


def get_defender() -> PromptDefender:
    """Get or create global prompt defender."""
    global _defender
    if _defender is None:
        _defender = PromptDefender()
    return _defender
