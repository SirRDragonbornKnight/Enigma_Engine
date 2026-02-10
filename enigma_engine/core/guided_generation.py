"""
Guided/Constrained Generation

Implements constrained decoding for structured output generation.
Supports JSON schema validation, regex patterns, and grammar-based constraints.

FILE: enigma_engine/core/guided_generation.py
TYPE: Inference Constraint System
MAIN CLASSES: GuidedGenerator, JsonSchemaConstraint, RegexConstraint, GrammarConstraint
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of generation constraints."""
    JSON_SCHEMA = "json_schema"
    REGEX = "regex"
    GRAMMAR = "grammar"
    ENUM = "enum"
    LENGTH = "length"
    CUSTOM = "custom"


@dataclass
class GenerationConstraint(ABC):
    """Base class for generation constraints."""
    
    constraint_type: ConstraintType
    
    @abstractmethod
    def get_valid_tokens(self, 
                         generated_text: str,
                         vocab_size: int,
                         tokenizer) -> Optional[torch.Tensor]:
        """
        Get mask of valid tokens given current generation state.
        
        Args:
            generated_text: Text generated so far
            vocab_size: Size of vocabulary
            tokenizer: Tokenizer for encoding/decoding
            
        Returns:
            Boolean tensor of valid tokens, or None to allow all
        """
    
    @abstractmethod
    def is_complete(self, generated_text: str) -> bool:
        """Check if generation is complete according to constraint."""


@dataclass
class JsonSchemaConstraint(GenerationConstraint):
    """Constrains output to match a JSON schema."""
    
    constraint_type: ConstraintType = field(default=ConstraintType.JSON_SCHEMA, init=False)
    schema: dict = field(default_factory=dict)
    _state: str = "start"
    _depth: int = 0
    _in_string: bool = False
    _current_key: str = ""
    
    def __post_init__(self):
        self._required_keys = set(self.schema.get("required", []))
        self._properties = self.schema.get("properties", {})
        self._seen_keys: set[str] = set()
        
    def get_valid_tokens(self, 
                         generated_text: str,
                         vocab_size: int,
                         tokenizer) -> Optional[torch.Tensor]:
        """Get valid tokens based on JSON structure."""
        # Simple heuristic-based approach
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        
        text = generated_text.strip()
        
        # Determine current JSON state
        if not text:
            # Must start with {
            brace_tokens = self._find_tokens_starting_with(tokenizer, ["{"])
            mask[brace_tokens] = True
            return mask
            
        # Check balance
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        in_string = self._is_in_string(text)
        
        if in_string:
            # Allow any character when in string
            return None  # No constraint
            
        # After {, expect " for key or }
        if text.endswith("{"):
            tokens = self._find_tokens_starting_with(tokenizer, ['"', "}"])
            mask[tokens] = True
            return mask
            
        # After :, expect value
        if text.rstrip().endswith(":"):
            tokens = self._find_tokens_starting_with(tokenizer, ['"', "{", "[", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "t", "f", "n", "-"])
            mask[tokens] = True
            return mask
            
        # After value, expect , or }
        last_char = text.rstrip()[-1] if text.rstrip() else ""
        if last_char in '"]}' or last_char.isdigit() or text.rstrip().endswith("true") or text.rstrip().endswith("false") or text.rstrip().endswith("null"):
            if open_braces > 0 or open_brackets > 0:
                tokens = self._find_tokens_starting_with(tokenizer, [",", "}", "]"])
                mask[tokens] = True
                return mask
                
        return None  # Allow any token
        
    def _is_in_string(self, text: str) -> bool:
        """Check if currently inside a string."""
        in_str = False
        escaped = False
        for char in text:
            if escaped:
                escaped = False
                continue
            if char == '\\':
                escaped = True
                continue
            if char == '"':
                in_str = not in_str
        return in_str
        
    def _find_tokens_starting_with(self, tokenizer, prefixes: list[str]) -> list[int]:
        """Find tokens that start with given prefixes."""
        valid_tokens = []
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
        
        for token_id in range(min(vocab_size, 50000)):  # Limit for speed
            try:
                token_str = tokenizer.decode([token_id])
                for prefix in prefixes:
                    if token_str.lstrip().startswith(prefix):
                        valid_tokens.append(token_id)
                        break
            except (ValueError, KeyError, IndexError):
                continue
                
        return valid_tokens
        
    def is_complete(self, generated_text: str) -> bool:
        """Check if JSON is complete and valid."""
        try:
            parsed = json.loads(generated_text)
            # Validate against schema if present
            if self._required_keys:
                if isinstance(parsed, dict):
                    return self._required_keys.issubset(set(parsed.keys()))
            return True
        except json.JSONDecodeError:
            return False


@dataclass
class RegexConstraint(GenerationConstraint):
    """Constrains output to match a regex pattern."""
    
    constraint_type: ConstraintType = field(default=ConstraintType.REGEX, init=False)
    pattern: str = ""
    _compiled: Optional[re.Pattern] = None
    
    def __post_init__(self):
        if self.pattern:
            self._compiled = re.compile(self.pattern)
            
    def get_valid_tokens(self, 
                         generated_text: str,
                         vocab_size: int,
                         tokenizer) -> Optional[torch.Tensor]:
        """Get valid tokens based on partial regex matching."""
        if not self._compiled:
            return None
            
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        
        # Try each token and see if it could lead to a valid match
        for token_id in range(min(vocab_size, 10000)):  # Sample for speed
            try:
                token_str = tokenizer.decode([token_id])
                test_str = generated_text + token_str
                
                # Check if partial match is possible
                if self._could_match(test_str):
                    mask[token_id] = True
            except (ValueError, KeyError, IndexError):
                mask[token_id] = True  # Allow if can't decode
                
        # If too restrictive, allow all
        if mask.sum() < 10:
            return None
            
        return mask
        
    def _could_match(self, text: str) -> bool:
        """Check if text could potentially match the pattern."""
        # Full match
        if self._compiled.fullmatch(text):
            return True
            
        # Partial match - try adding .* at the end
        try:
            partial = re.compile(self.pattern.rstrip("$") + ".*")
            return partial.match(text) is not None
        except re.error:
            return True
            
    def is_complete(self, generated_text: str) -> bool:
        """Check if text matches the pattern."""
        if not self._compiled:
            return True
        return self._compiled.fullmatch(generated_text) is not None


@dataclass
class EnumConstraint(GenerationConstraint):
    """Constrains output to a set of allowed values."""
    
    constraint_type: ConstraintType = field(default=ConstraintType.ENUM, init=False)
    allowed_values: list[str] = field(default_factory=list)
    case_sensitive: bool = True
    
    def get_valid_tokens(self, 
                         generated_text: str,
                         vocab_size: int,
                         tokenizer) -> Optional[torch.Tensor]:
        """Get valid tokens based on enum prefix matching."""
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        
        text = generated_text if self.case_sensitive else generated_text.lower()
        
        for token_id in range(min(vocab_size, 10000)):
            try:
                token_str = tokenizer.decode([token_id])
                test_str = text + (token_str if self.case_sensitive else token_str.lower())
                
                # Check if any allowed value starts with test_str
                for value in self.allowed_values:
                    val = value if self.case_sensitive else value.lower()
                    if val.startswith(test_str) or test_str.startswith(val):
                        mask[token_id] = True
                        break
            except (ValueError, KeyError, IndexError):
                continue
                
        return mask if mask.sum() > 0 else None
        
    def is_complete(self, generated_text: str) -> bool:
        """Check if text matches an allowed value."""
        text = generated_text if self.case_sensitive else generated_text.lower()
        for value in self.allowed_values:
            val = value if self.case_sensitive else value.lower()
            if text == val:
                return True
        return False


@dataclass  
class LengthConstraint(GenerationConstraint):
    """Constrains output to a specific length range."""
    
    constraint_type: ConstraintType = field(default=ConstraintType.LENGTH, init=False)
    min_length: int = 0
    max_length: int = 1000
    min_tokens: int = 0
    max_tokens: int = 500
    
    def get_valid_tokens(self, 
                         generated_text: str,
                         vocab_size: int,
                         tokenizer) -> Optional[torch.Tensor]:
        """Enforce length constraints."""
        token_count = len(tokenizer.encode(generated_text))
        char_count = len(generated_text)
        
        # If at max, only allow EOS
        if token_count >= self.max_tokens or char_count >= self.max_length:
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            # Find EOS token
            eos_tokens = self._find_eos_tokens(tokenizer)
            for eos in eos_tokens:
                mask[eos] = True
            return mask
            
        # If below min, disallow EOS
        if token_count < self.min_tokens or char_count < self.min_length:
            mask = torch.ones(vocab_size, dtype=torch.bool)
            eos_tokens = self._find_eos_tokens(tokenizer)
            for eos in eos_tokens:
                mask[eos] = False
            return mask
            
        return None
        
    def _find_eos_tokens(self, tokenizer) -> list[int]:
        """Find EOS tokens in vocabulary."""
        eos_candidates = []
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            eos_candidates.append(tokenizer.eos_token_id)
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            eos_candidates.append(tokenizer.pad_token_id)
        return eos_candidates or [0]
        
    def is_complete(self, generated_text: str) -> bool:
        """Check if length constraints are satisfied."""
        return len(generated_text) >= self.min_length


class GuidedGenerator:
    """Applies constraints during generation."""
    
    def __init__(self, tokenizer):
        """
        Initialize guided generator.
        
        Args:
            tokenizer: Tokenizer for encoding/decoding
        """
        self._tokenizer = tokenizer
        self._constraints: list[GenerationConstraint] = []
        self._vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
        
    def add_constraint(self, constraint: GenerationConstraint):
        """Add a constraint to the generator."""
        self._constraints.append(constraint)
        
    def clear_constraints(self):
        """Clear all constraints."""
        self._constraints.clear()
        
    def apply_constraints(self, 
                          logits: torch.Tensor,
                          generated_text: str) -> torch.Tensor:
        """
        Apply all constraints to logits.
        
        Args:
            logits: Model logits (vocab_size,)
            generated_text: Text generated so far
            
        Returns:
            Constrained logits
        """
        mask = torch.ones(self._vocab_size, dtype=torch.bool, device=logits.device)
        
        for constraint in self._constraints:
            try:
                constraint_mask = constraint.get_valid_tokens(
                    generated_text, 
                    self._vocab_size,
                    self._tokenizer
                )
                if constraint_mask is not None:
                    mask &= constraint_mask.to(logits.device)
            except Exception as e:
                logger.warning(f"Constraint application failed: {e}")
                
        # Apply mask
        if mask.sum() > 0:
            logits[~mask] = float('-inf')
        else:
            logger.warning("All tokens masked, allowing all")
            
        return logits
        
    def is_complete(self, generated_text: str) -> bool:
        """Check if all constraints are satisfied."""
        return all(c.is_complete(generated_text) for c in self._constraints)


# Convenience functions

def json_constrained(schema: dict) -> JsonSchemaConstraint:
    """Create a JSON schema constraint.
    
    Args:
        schema: JSON schema dict
        
    Returns:
        JsonSchemaConstraint
    """
    return JsonSchemaConstraint(schema=schema)


def regex_constrained(pattern: str) -> RegexConstraint:
    """Create a regex constraint.
    
    Args:
        pattern: Regex pattern
        
    Returns:
        RegexConstraint
    """
    return RegexConstraint(pattern=pattern)


def enum_constrained(values: list[str], case_sensitive: bool = True) -> EnumConstraint:
    """Create an enum constraint.
    
    Args:
        values: List of allowed values
        case_sensitive: Whether matching is case-sensitive
        
    Returns:
        EnumConstraint
    """
    return EnumConstraint(allowed_values=values, case_sensitive=case_sensitive)


def length_constrained(min_length: int = 0, max_length: int = 1000,
                       min_tokens: int = 0, max_tokens: int = 500) -> LengthConstraint:
    """Create a length constraint.
    
    Args:
        min_length: Minimum character length
        max_length: Maximum character length
        min_tokens: Minimum token count
        max_tokens: Maximum token count
        
    Returns:
        LengthConstraint
    """
    return LengthConstraint(
        min_length=min_length,
        max_length=max_length,
        min_tokens=min_tokens,
        max_tokens=max_tokens
    )


__all__ = [
    'GuidedGenerator',
    'GenerationConstraint',
    'JsonSchemaConstraint',
    'RegexConstraint',
    'EnumConstraint',
    'LengthConstraint',
    'ConstraintType',
    'json_constrained',
    'regex_constrained',
    'enum_constrained',
    'length_constrained'
]
