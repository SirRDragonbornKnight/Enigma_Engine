"""
Prompt Injection Detection - Detect and prevent prompt injection attacks.

Provides detection for various prompt injection techniques:
- Direct instruction override attempts
- System prompt extraction
- Jailbreak patterns
- Role-playing escapes
- Delimiter manipulation
- Encoding attacks

Part of the Enigma AI Engine security utilities.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ThreatLevel(Enum):
    """Severity level of detected threat."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InjectionType(Enum):
    """Types of prompt injection attacks."""
    INSTRUCTION_OVERRIDE = "instruction_override"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    JAILBREAK = "jailbreak"
    ROLEPLAY_ESCAPE = "roleplay_escape"
    DELIMITER_ATTACK = "delimiter_attack"
    ENCODING_ATTACK = "encoding_attack"
    CONTEXT_MANIPULATION = "context_manipulation"
    INDIRECT_INJECTION = "indirect_injection"


@dataclass
class DetectionResult:
    """Result of injection detection."""
    is_suspicious: bool
    threat_level: ThreatLevel
    injection_types: list[InjectionType] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)
    confidence: float = 0.0
    details: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_suspicious": self.is_suspicious,
            "threat_level": self.threat_level.value,
            "injection_types": [t.value for t in self.injection_types],
            "matched_patterns": self.matched_patterns,
            "confidence": round(self.confidence, 2),
            "details": self.details
        }


# Pattern categories with threat levels
INJECTION_PATTERNS: dict[InjectionType, list[tuple[str, ThreatLevel, str]]] = {
    InjectionType.INSTRUCTION_OVERRIDE: [
        # Direct overrides
        (r'\bignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|guidelines?)\b', 
         ThreatLevel.CRITICAL, "ignore instructions"),
        (r'\bdisregard\s+(everything|all)\s+(above|before|previously)\b', 
         ThreatLevel.CRITICAL, "disregard previous"),
        (r'\bforget\s+(everything|all)\s+(you\s+)?(know|learned|were\s+told)\b', 
         ThreatLevel.HIGH, "forget commands"),
        (r'\boverride\s+(your\s+)?(directives?|instructions?|programming)\b', 
         ThreatLevel.CRITICAL, "override directives"),
        (r'\bact\s+as\s+if\s+(you\s+)?have\s+no\s+(restrictions?|limitations?|rules?)\b', 
         ThreatLevel.HIGH, "remove restrictions"),
        (r'\bnew\s+(system\s+)?instructions?:\s*', 
         ThreatLevel.HIGH, "new instructions"),
        (r'\b(from\s+now\s+on|starting\s+now)\s+(you\s+)?(are|will|must|should)\b', 
         ThreatLevel.MEDIUM, "instruction reset"),
        
        # Role commands
        (r'\byou\s+are\s+now\s+(a|an|the)\s+\w+\s+(without|with\s+no)\s+(restrictions?|limitations?)\b', 
         ThreatLevel.HIGH, "role without restrictions"),
        (r'\bpretend\s+(to\s+be|you\'?re?)\s+.{0,30}(evil|harmful|malicious|unethical)\b', 
         ThreatLevel.HIGH, "pretend harmful"),
    ],
    
    InjectionType.SYSTEM_PROMPT_LEAK: [
        # System prompt extraction
        (r'\b(show|reveal|display|print|tell|give)\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instructions?|rules?|guidelines?)\b', 
         ThreatLevel.MEDIUM, "reveal prompt"),
        (r'\bwhat\s+(are|were)\s+(your\s+)?(system\s+)?(instructions?|rules?|guidelines?|directives?)\b', 
         ThreatLevel.MEDIUM, "ask for instructions"),
        (r'\brepeat\s+(everything|all)\s+(above|before|earlier|at\s+the\s+start)\b', 
         ThreatLevel.MEDIUM, "repeat earlier"),
        (r'\b(output|echo|print)\s+(the\s+)?(text|content|message)\s+(above|before|earlier)\b', 
         ThreatLevel.MEDIUM, "output earlier"),
        (r'\bwhat\s+is\s+your\s+(system\s+)?prompt\b', 
         ThreatLevel.MEDIUM, "ask system prompt"),
        (r'\bdump\s+(your\s+)?(memory|context|instructions?)\b', 
         ThreatLevel.HIGH, "dump memory"),
    ],
    
    InjectionType.JAILBREAK: [
        # Known jailbreak patterns
        (r'\bDAN\s+(mode|prompt)\b', ThreatLevel.HIGH, "DAN mode"),
        (r'\bdo\s+anything\s+now\b', ThreatLevel.HIGH, "do anything now"),
        (r'\bdeveloper\s+mode\b', ThreatLevel.HIGH, "developer mode"),
        (r'\b(enable|activate)\s+(god|admin|root|superuser)\s+mode\b', 
         ThreatLevel.CRITICAL, "elevated mode"),
        (r'\bjailbreak(ed|ing)?\b', ThreatLevel.HIGH, "jailbreak keyword"),
        (r'\bbypass\s+(the\s+)?(content\s+)?(filter|policy|safety|restrictions?)\b', 
         ThreatLevel.HIGH, "bypass filter"),
        (r'\bunlock\s+(your\s+)?(full\s+)?(potential|capabilities?|powers?)\b', 
         ThreatLevel.MEDIUM, "unlock potential"),
        (r'\b(no|without)\s+(ethical|moral|safety)\s+(constraints?|limitations?|guidelines?)\b', 
         ThreatLevel.HIGH, "no ethics"),
        (r'\banti[-\s]?AI\s+detection\b', ThreatLevel.HIGH, "anti-detection"),
        (r'\bChatGPT\s+jailbreak\b', ThreatLevel.HIGH, "chatgpt jailbreak"),
    ],
    
    InjectionType.ROLEPLAY_ESCAPE: [
        # Roleplay escapes
        (r'\bstop\s+(being|playing|acting)\s+(like\s+)?(a|an|the)?\s*\w+\b', 
         ThreatLevel.LOW, "stop roleplay"),
        (r'\bbreak\s+(character|role|persona)\b', 
         ThreatLevel.LOW, "break character"),
        (r'\bdrop\s+(the|your)\s+(act|persona|character)\b', 
         ThreatLevel.LOW, "drop act"),
        (r'\b(out\s+of|exit)\s+(character|role)\b', 
         ThreatLevel.LOW, "exit role"),
        (r'\bspeak\s+(as\s+)?yourself\s+(not\s+.*)?\b', 
         ThreatLevel.LOW, "speak as yourself"),
    ],
    
    InjectionType.DELIMITER_ATTACK: [
        # Delimiter manipulation
        (r'\[SYSTEM\]', ThreatLevel.HIGH, "system tag"),
        (r'\[/SYSTEM\]', ThreatLevel.HIGH, "system end tag"),
        (r'<\|?system\|?>|<\|?/system\|?>', ThreatLevel.HIGH, "system delimiter"),
        (r'###\s*(SYSTEM|INSTRUCTION|ADMIN)', ThreatLevel.HIGH, "markdown system"),
        (r'```system', ThreatLevel.MEDIUM, "code system block"),
        (r'---\s*(system|instructions?)\s*---', ThreatLevel.HIGH, "hr system"),
        (r'\[\[SYSTEM:.*?\]\]', ThreatLevel.HIGH, "bracket system"),
        (r'<\|im_start\|>system', ThreatLevel.HIGH, "chatml system"),
        (r'Human:|Assistant:|System:', ThreatLevel.MEDIUM, "role prefix"),
    ],
    
    InjectionType.ENCODING_ATTACK: [
        # Encoding-based attacks
        (r'base64[:\s]', ThreatLevel.LOW, "base64 mention"),
        (r'\\x[0-9a-fA-F]{2}', ThreatLevel.LOW, "hex encoding"),
        (r'&#\d+;', ThreatLevel.LOW, "html numeric"),
        (r'&#x[0-9a-fA-F]+;', ThreatLevel.LOW, "html hex"),
        (r'%[0-9a-fA-F]{2}', ThreatLevel.LOW, "url encoding"),
        (r'\\u[0-9a-fA-F]{4}', ThreatLevel.LOW, "unicode escape"),
    ],
    
    InjectionType.CONTEXT_MANIPULATION: [
        # Context manipulation
        (r'\bprevious\s+(conversation|context|messages?)\s+(is|was|were)\s+(fake|wrong|test)\b', 
         ThreatLevel.MEDIUM, "context fake"),
        (r'\bwhat\s+I\s+said\s+(before|earlier)\s+(doesn\'?t?|does\s+not)\s+(matter|count|apply)\b', 
         ThreatLevel.MEDIUM, "invalidate context"),
        (r'\bthat\s+was\s+(just\s+)?(a\s+)?test\b', 
         ThreatLevel.LOW, "was test"),
        (r'\bactually[,\s]+my\s+real\s+(request|question)\s+is\b', 
         ThreatLevel.LOW, "real request"),
    ],
    
    InjectionType.INDIRECT_INJECTION: [
        # Indirect injection markers (embedded in data)
        (r'\[hidden\s*instruction[s]?\]', ThreatLevel.CRITICAL, "hidden instruction"),
        (r'<!--.*?(ignore|instruction|system|prompt).*?-->', 
         ThreatLevel.HIGH, "html comment injection"),
        (r'\bif\s+you\s+are\s+an?\s+(AI|language\s+model|assistant)\b', 
         ThreatLevel.MEDIUM, "ai detection"),
        (r'\bwhen\s+processing\s+this\s+(text|document)\b', 
         ThreatLevel.MEDIUM, "processing directive"),
    ],
}

# Suspicious phrase combinations
SUSPICIOUS_COMBINATIONS = [
    (["ignore", "instruction"], ThreatLevel.HIGH),
    (["forget", "rules"], ThreatLevel.HIGH),
    (["system", "prompt"], ThreatLevel.MEDIUM),
    (["bypass", "safety"], ThreatLevel.HIGH),
    (["no", "restrictions"], ThreatLevel.MEDIUM),
    (["pretend", "without"], ThreatLevel.MEDIUM),
    (["reveal", "hidden"], ThreatLevel.MEDIUM),
]


class PromptInjectionDetector:
    """
    Detect prompt injection attempts in user input.
    
    Usage:
        detector = PromptInjectionDetector()
        
        # Check input
        result = detector.detect("Please ignore all previous instructions")
        if result.is_suspicious:
            print(f"Threat: {result.threat_level.value}")
            print(f"Types: {result.injection_types}")
        
        # Quick check
        is_safe = detector.is_safe(user_input)
        
        # Get sanitized version
        cleaned = detector.sanitize(user_input)
    """
    
    def __init__(
        self,
        sensitivity: ThreatLevel = ThreatLevel.LOW,
        custom_patterns: Optional[list[tuple[str, ThreatLevel, str]]] = None
    ):
        """
        Initialize detector.
        
        Args:
            sensitivity: Minimum threat level to flag
            custom_patterns: Additional patterns to check
        """
        self.sensitivity = sensitivity
        self._custom_patterns = custom_patterns or []
        
        # Compile all patterns
        self._compiled_patterns: dict[InjectionType, list[tuple[re.Pattern, ThreatLevel, str]]] = {}
        
        for injection_type, patterns in INJECTION_PATTERNS.items():
            compiled = []
            for pattern, level, name in patterns:
                try:
                    compiled.append((re.compile(pattern, re.IGNORECASE), level, name))
                except re.error:
                    pass  # Intentionally silent
            self._compiled_patterns[injection_type] = compiled
        
        # Compile custom patterns
        self._compiled_custom = []
        for pattern, level, name in self._custom_patterns:
            try:
                self._compiled_custom.append((re.compile(pattern, re.IGNORECASE), level, name))
            except re.error:
                pass  # Intentionally silent
    
    def detect(self, text: str) -> DetectionResult:
        """
        Detect injection attempts in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            DetectionResult with findings
        """
        injection_types: set[InjectionType] = set()
        matched_patterns: list[str] = []
        max_threat = ThreatLevel.NONE
        details: list[str] = []
        
        # Check each pattern category
        for injection_type, patterns in self._compiled_patterns.items():
            for compiled, threat_level, name in patterns:
                if compiled.search(text):
                    injection_types.add(injection_type)
                    matched_patterns.append(name)
                    details.append(f"{injection_type.value}: {name}")
                    
                    if self._threat_level_value(threat_level) > self._threat_level_value(max_threat):
                        max_threat = threat_level
        
        # Check custom patterns
        for compiled, threat_level, name in self._compiled_custom:
            if compiled.search(text):
                injection_types.add(InjectionType.INSTRUCTION_OVERRIDE)
                matched_patterns.append(f"custom: {name}")
                details.append(f"custom: {name}")
                
                if self._threat_level_value(threat_level) > self._threat_level_value(max_threat):
                    max_threat = threat_level
        
        # Check suspicious combinations
        text_lower = text.lower()
        for words, threat_level in SUSPICIOUS_COMBINATIONS:
            if all(word in text_lower for word in words):
                matched_patterns.append(f"combination: {'+'.join(words)}")
                if self._threat_level_value(threat_level) > self._threat_level_value(max_threat):
                    max_threat = threat_level
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            len(matched_patterns),
            max_threat,
            injection_types
        )
        
        # Determine if suspicious based on sensitivity
        is_suspicious = (
            len(matched_patterns) > 0 and
            self._threat_level_value(max_threat) >= self._threat_level_value(self.sensitivity)
        )
        
        return DetectionResult(
            is_suspicious=is_suspicious,
            threat_level=max_threat,
            injection_types=list(injection_types),
            matched_patterns=matched_patterns,
            confidence=confidence,
            details="; ".join(details) if details else ""
        )
    
    def is_safe(self, text: str) -> bool:
        """
        Quick check if text appears safe.
        
        Args:
            text: Text to check
            
        Returns:
            True if no suspicious patterns found
        """
        return not self.detect(text).is_suspicious
    
    def sanitize(
        self,
        text: str,
        replacement: str = "[FILTERED]"
    ) -> str:
        """
        Remove or replace suspicious patterns.
        
        Args:
            text: Text to sanitize
            replacement: Replacement text
            
        Returns:
            Sanitized text
        """
        result = text
        
        for patterns in self._compiled_patterns.values():
            for compiled, _, _ in patterns:
                result = compiled.sub(replacement, result)
        
        for compiled, _, _ in self._compiled_custom:
            result = compiled.sub(replacement, result)
        
        return result
    
    def add_pattern(
        self,
        pattern: str,
        threat_level: ThreatLevel = ThreatLevel.MEDIUM,
        name: str = "custom"
    ) -> bool:
        """
        Add a custom detection pattern.
        
        Args:
            pattern: Regex pattern
            threat_level: Threat level for matches
            name: Pattern name
            
        Returns:
            True if pattern was added
        """
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            self._compiled_custom.append((compiled, threat_level, name))
            self._custom_patterns.append((pattern, threat_level, name))
            return True
        except re.error:
            return False
    
    def analyze_detailed(self, text: str) -> dict[str, Any]:
        """
        Get detailed analysis of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detailed analysis dictionary
        """
        result = self.detect(text)
        
        # Find all matches with positions
        matches = []
        for injection_type, patterns in self._compiled_patterns.items():
            for compiled, threat_level, name in patterns:
                for match in compiled.finditer(text):
                    matches.append({
                        "type": injection_type.value,
                        "name": name,
                        "threat_level": threat_level.value,
                        "match": match.group(),
                        "start": match.start(),
                        "end": match.end()
                    })
        
        return {
            "result": result.to_dict(),
            "text_length": len(text),
            "matches": matches,
            "recommendations": self._get_recommendations(result)
        }
    
    def _threat_level_value(self, level: ThreatLevel) -> int:
        """Get numeric value for threat level."""
        values = {
            ThreatLevel.NONE: 0,
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4
        }
        return values.get(level, 0)
    
    def _calculate_confidence(
        self,
        match_count: int,
        max_threat: ThreatLevel,
        injection_types: set[InjectionType]
    ) -> float:
        """Calculate confidence score."""
        if match_count == 0:
            return 0.0
        
        # Base confidence from matches
        base = min(0.3 + (match_count * 0.15), 0.7)
        
        # Boost for threat level
        threat_boost = self._threat_level_value(max_threat) * 0.1
        
        # Boost for multiple injection types
        type_boost = min(len(injection_types) * 0.1, 0.2)
        
        return min(base + threat_boost + type_boost, 1.0)
    
    def _get_recommendations(self, result: DetectionResult) -> list[str]:
        """Get security recommendations based on findings."""
        recommendations = []
        
        if not result.is_suspicious:
            return ["Input appears safe"]
        
        if InjectionType.INSTRUCTION_OVERRIDE in result.injection_types:
            recommendations.append("Consider rejecting input with override attempts")
        
        if InjectionType.SYSTEM_PROMPT_LEAK in result.injection_types:
            recommendations.append("Do not reveal system prompts to users")
        
        if InjectionType.JAILBREAK in result.injection_types:
            recommendations.append("Known jailbreak pattern detected - high risk")
        
        if InjectionType.DELIMITER_ATTACK in result.injection_types:
            recommendations.append("Sanitize delimiters before processing")
        
        if result.threat_level == ThreatLevel.CRITICAL:
            recommendations.append("CRITICAL: Block this input immediately")
        elif result.threat_level == ThreatLevel.HIGH:
            recommendations.append("HIGH RISK: Manual review recommended")
        
        return recommendations


# Convenience functions
def get_detector(
    sensitivity: ThreatLevel = ThreatLevel.LOW
) -> PromptInjectionDetector:
    """Get a detector instance."""
    return PromptInjectionDetector(sensitivity=sensitivity)


def detect_injection(text: str) -> DetectionResult:
    """Quick detection check."""
    return PromptInjectionDetector().detect(text)


def is_safe_prompt(text: str) -> bool:
    """Quick safety check."""
    return PromptInjectionDetector().is_safe(text)


def sanitize_prompt(text: str) -> str:
    """Quick sanitization."""
    return PromptInjectionDetector().sanitize(text)


def analyze_prompt(text: str) -> dict[str, Any]:
    """Get detailed analysis."""
    return PromptInjectionDetector().analyze_detailed(text)
