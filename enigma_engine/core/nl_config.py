"""
Natural Language Model Configuration for Enigma AI Engine

Configure model parameters using natural language.

Features:
- "Make it more creative"
- "Be more concise"
- "Speak formally"
- "Increase randomness"
- Sentiment-based adjustments
- Personality presets
- Learning user preferences

Usage:
    from enigma_engine.core.nl_config import NLModelConfig
    
    config = NLModelConfig()
    
    # Natural language adjustments
    config.apply("Make it more creative")
    config.apply("Be very precise and factual")
    config.apply("Write in a casual tone")
    
    # Get resulting parameters
    params = config.get_parameters()
    # {'temperature': 0.9, 'top_p': 0.95, ...}
    
    # Apply to engine
    config.configure_engine(engine)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Dimension(Enum):
    """Configuration dimensions."""
    CREATIVITY = auto()      # temperature, top_p
    PRECISION = auto()       # temperature (low), top_k
    VERBOSITY = auto()       # max_tokens, repetition penalty
    FORMALITY = auto()       # system prompt adjustments
    SPEED = auto()           # max_tokens, early stopping
    SAFETY = auto()          # top_k, filtering
    CONSISTENCY = auto()     # temperature, seed


@dataclass
class ParameterRange:
    """Valid range for a parameter."""
    min_val: float
    max_val: float
    default: float
    
    def clamp(self, value: float) -> float:
        return max(self.min_val, min(self.max_val, value))


# Parameter definitions
PARAMETERS = {
    'temperature': ParameterRange(0.0, 2.0, 0.7),
    'top_p': ParameterRange(0.0, 1.0, 0.9),
    'top_k': ParameterRange(1, 100, 50),
    'max_tokens': ParameterRange(1, 4096, 512),
    'repetition_penalty': ParameterRange(1.0, 2.0, 1.1),
    'frequency_penalty': ParameterRange(0.0, 2.0, 0.0),
    'presence_penalty': ParameterRange(0.0, 2.0, 0.0),
}


@dataclass
class Adjustment:
    """A parameter adjustment."""
    parameter: str
    delta: float = 0.0       # Relative change
    absolute: Optional[float] = None  # Absolute value
    factor: float = 1.0      # Multiplicative factor


@dataclass
class Intent:
    """Parsed user intent."""
    dimension: Dimension
    direction: float  # -1 to 1 (decrease to increase)
    intensity: float  # 0 to 1 (subtle to extreme)
    adjustments: List[Adjustment] = field(default_factory=list)


# Intent patterns
INTENT_PATTERNS: Dict[str, Tuple[Dimension, float]] = {
    # Creativity
    r'(?:more\s+)?creativ': (Dimension.CREATIVITY, 1.0),
    r'(?:more\s+)?imaginat': (Dimension.CREATIVITY, 1.0),
    r'(?:more\s+)?original': (Dimension.CREATIVITY, 0.8),
    r'(?:less\s+)?boring': (Dimension.CREATIVITY, 0.6),
    r'(?:more\s+)?random': (Dimension.CREATIVITY, 1.0),
    r'(?:more\s+)?varied': (Dimension.CREATIVITY, 0.7),
    r'(?:more\s+)?playful': (Dimension.CREATIVITY, 0.8),
    r'(?:less\s+)?predictabl': (Dimension.CREATIVITY, 0.7),
    
    r'(?:less\s+)?creativ': (Dimension.CREATIVITY, -0.8),
    r'(?:more\s+)?conservativ': (Dimension.CREATIVITY, -0.6),
    r'(?:more\s+)?safe': (Dimension.CREATIVITY, -0.7),
    r'(?:more\s+)?predicable': (Dimension.CREATIVITY, -0.5),
    
    # Precision
    r'(?:more\s+)?precis': (Dimension.PRECISION, 1.0),
    r'(?:more\s+)?accurat': (Dimension.PRECISION, 1.0),
    r'(?:more\s+)?factual': (Dimension.PRECISION, 0.9),
    r'(?:more\s+)?exact': (Dimension.PRECISION, 0.8),
    r'(?:more\s+)?technical': (Dimension.PRECISION, 0.7),
    r'(?:more\s+)?specific': (Dimension.PRECISION, 0.6),
    r'(?:more\s+)?detailed': (Dimension.PRECISION, 0.5),
    
    r'(?:less\s+)?precis': (Dimension.PRECISION, -0.8),
    r'(?:more\s+)?casual': (Dimension.PRECISION, -0.5),
    r'(?:more\s+)?relaxed': (Dimension.PRECISION, -0.4),
    
    # Verbosity
    r'(?:more\s+)?verbose': (Dimension.VERBOSITY, 1.0),
    r'(?:more\s+)?detailed': (Dimension.VERBOSITY, 0.8),
    r'(?:longer\s+)?response': (Dimension.VERBOSITY, 0.7),
    r'(?:more\s+)?thorough': (Dimension.VERBOSITY, 0.8),
    r'(?:more\s+)?complet': (Dimension.VERBOSITY, 0.6),
    r'expand': (Dimension.VERBOSITY, 0.7),
    r'elaborate': (Dimension.VERBOSITY, 0.8),
    
    r'(?:more\s+)?concis': (Dimension.VERBOSITY, -1.0),
    r'(?:more\s+)?brief': (Dimension.VERBOSITY, -0.9),
    r'shorter': (Dimension.VERBOSITY, -0.8),
    r'(?:more\s+)?succinct': (Dimension.VERBOSITY, -0.7),
    r'to the point': (Dimension.VERBOSITY, -0.8),
    
    # Formality
    r'(?:more\s+)?formal': (Dimension.FORMALITY, 1.0),
    r'(?:more\s+)?professional': (Dimension.FORMALITY, 0.9),
    r'(?:more\s+)?polite': (Dimension.FORMALITY, 0.6),
    r'(?:more\s+)?serious': (Dimension.FORMALITY, 0.7),
    r'business': (Dimension.FORMALITY, 0.8),
    
    r'(?:more\s+)?casual': (Dimension.FORMALITY, -0.8),
    r'(?:more\s+)?informal': (Dimension.FORMALITY, -1.0),
    r'(?:more\s+)?friendly': (Dimension.FORMALITY, -0.6),
    r'(?:more\s+)?relaxed': (Dimension.FORMALITY, -0.5),
    r'conversational': (Dimension.FORMALITY, -0.7),
    
    # Speed
    r'(?:more\s+)?quick': (Dimension.SPEED, 1.0),
    r'(?:more\s+)?fast': (Dimension.SPEED, 1.0),
    r'faster': (Dimension.SPEED, 0.8),
    r'hurry': (Dimension.SPEED, 0.7),
    
    r'(?:more\s+)?slow': (Dimension.SPEED, -0.5),
    r'take.*time': (Dimension.SPEED, -0.6),
    
    # Consistency
    r'(?:more\s+)?consistent': (Dimension.CONSISTENCY, 1.0),
    r'(?:more\s+)?stable': (Dimension.CONSISTENCY, 0.9),
    r'(?:more\s+)?reliable': (Dimension.CONSISTENCY, 0.8),
    r'same': (Dimension.CONSISTENCY, 0.7),
}

# Intensity words
INTENSITY_MODIFIERS = {
    'very': 1.5,
    'really': 1.4,
    'much': 1.3,
    'extremely': 2.0,
    'super': 1.6,
    'a lot': 1.4,
    'way': 1.3,
    'slightly': 0.5,
    'a bit': 0.4,
    'little': 0.3,
    'somewhat': 0.6,
    'kind of': 0.4,
    'sort of': 0.4,
}


# Dimension to parameter mappings
DIMENSION_MAPPINGS: Dict[Dimension, List[Adjustment]] = {
    Dimension.CREATIVITY: [
        Adjustment('temperature', delta=0.2),
        Adjustment('top_p', delta=0.05),
        Adjustment('presence_penalty', delta=0.1),
    ],
    Dimension.PRECISION: [
        Adjustment('temperature', delta=-0.15),
        Adjustment('top_k', delta=-10),
        Adjustment('repetition_penalty', delta=0.05),
    ],
    Dimension.VERBOSITY: [
        Adjustment('max_tokens', factor=1.5),
        Adjustment('repetition_penalty', delta=-0.02),
    ],
    Dimension.FORMALITY: [
        # Primarily affects system prompt
    ],
    Dimension.SPEED: [
        Adjustment('max_tokens', factor=0.7),
    ],
    Dimension.CONSISTENCY: [
        Adjustment('temperature', delta=-0.2),
        Adjustment('top_k', delta=-15),
    ],
}


# Personality presets
PRESETS: Dict[str, Dict[str, float]] = {
    'creative': {
        'temperature': 1.0,
        'top_p': 0.95,
        'presence_penalty': 0.5,
        'frequency_penalty': 0.3,
    },
    'precise': {
        'temperature': 0.3,
        'top_p': 0.85,
        'top_k': 30,
        'repetition_penalty': 1.15,
    },
    'balanced': {
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50,
        'max_tokens': 512,
    },
    'concise': {
        'temperature': 0.5,
        'max_tokens': 256,
        'repetition_penalty': 1.2,
    },
    'verbose': {
        'temperature': 0.8,
        'max_tokens': 2048,
        'repetition_penalty': 1.05,
    },
    'formal': {
        'temperature': 0.5,
        'repetition_penalty': 1.15,
    },
    'casual': {
        'temperature': 0.9,
        'top_p': 0.95,
        'presence_penalty': 0.3,
    },
}


class NLModelConfig:
    """
    Configure model parameters using natural language.
    
    Parses natural language instructions and converts them
    to appropriate parameter adjustments.
    """
    
    def __init__(self, base_params: Optional[Dict[str, float]] = None):
        """
        Initialize NL config.
        
        Args:
            base_params: Starting parameter values
        """
        self._params: Dict[str, float] = {}
        
        # Set defaults
        for name, range_def in PARAMETERS.items():
            self._params[name] = range_def.default
        
        # Apply base params
        if base_params:
            for name, value in base_params.items():
                if name in PARAMETERS:
                    self._params[name] = PARAMETERS[name].clamp(value)
        
        # System prompt modifications
        self._system_prompt_modifiers: List[str] = []
        
        # History of adjustments
        self._history: List[Tuple[str, Intent]] = []
        self._max_history: int = 100
        
        logger.info("NLModelConfig initialized")
    
    def apply(self, instruction: str) -> Dict[str, Any]:
        """
        Apply a natural language instruction.
        
        Args:
            instruction: Natural language like "make it more creative"
            
        Returns:
            Dict with applied changes
        """
        # Check for preset
        preset_name = self._check_preset(instruction)
        if preset_name:
            return self._apply_preset(preset_name)
        
        # Parse intent
        intent = self._parse_intent(instruction)
        if not intent:
            logger.warning(f"Could not parse intent from: {instruction}")
            return {'understood': False, 'instruction': instruction}
        
        # Apply adjustments
        changes = self._apply_intent(intent)
        
        # Record history
        self._history.append((instruction, intent))
        
        # Trim history if too long
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        return {
            'understood': True,
            'dimension': intent.dimension.name,
            'direction': 'increase' if intent.direction > 0 else 'decrease',
            'intensity': intent.intensity,
            'changes': changes
        }
    
    def _check_preset(self, instruction: str) -> Optional[str]:
        """Check if instruction matches a preset."""
        instruction_lower = instruction.lower()
        
        # Direct preset mentions
        for preset in PRESETS:
            if preset in instruction_lower:
                return preset
        
        # Preset aliases
        aliases = {
            'default': 'balanced',
            'normal': 'balanced',
            'standard': 'balanced',
            'professional': 'formal',
            'business': 'formal',
            'fun': 'casual',
            'friendly': 'casual',
            'short': 'concise',
            'brief': 'concise',
            'long': 'verbose',
            'detailed': 'verbose',
            'imaginative': 'creative',
            'factual': 'precise',
            'accurate': 'precise',
        }
        
        for alias, preset in aliases.items():
            if alias in instruction_lower:
                return preset
        
        return None
    
    def _apply_preset(self, preset_name: str) -> Dict[str, Any]:
        """Apply a preset configuration."""
        if preset_name not in PRESETS:
            return {'understood': False, 'error': f'Unknown preset: {preset_name}'}
        
        preset = PRESETS[preset_name]
        changes = {}
        
        for param, value in preset.items():
            if param in PARAMETERS:
                old_value = self._params[param]
                self._params[param] = PARAMETERS[param].clamp(value)
                changes[param] = {
                    'from': old_value,
                    'to': self._params[param]
                }
        
        return {
            'understood': True,
            'preset': preset_name,
            'changes': changes
        }
    
    def _parse_intent(self, instruction: str) -> Optional[Intent]:
        """Parse natural language to intent."""
        instruction_lower = instruction.lower()
        
        # Find matching pattern
        best_match = None
        best_dimension = None
        best_direction = 0.0
        
        for pattern, (dimension, direction) in INTENT_PATTERNS.items():
            if re.search(pattern, instruction_lower):
                best_match = pattern
                best_dimension = dimension
                best_direction = direction
                break
        
        if not best_dimension:
            return None
        
        # Calculate intensity
        intensity = 1.0
        for modifier, multiplier in INTENSITY_MODIFIERS.items():
            if modifier in instruction_lower:
                intensity *= multiplier
                break
        
        intensity = min(2.0, intensity)  # Cap intensity
        
        return Intent(
            dimension=best_dimension,
            direction=best_direction,
            intensity=intensity,
            adjustments=DIMENSION_MAPPINGS.get(best_dimension, [])
        )
    
    def _apply_intent(self, intent: Intent) -> Dict[str, Dict[str, float]]:
        """Apply an intent to parameters."""
        changes = {}
        
        for adj in intent.adjustments:
            if adj.parameter not in PARAMETERS:
                continue
            
            old_value = self._params[adj.parameter]
            new_value = old_value
            
            if adj.absolute is not None:
                new_value = adj.absolute
            elif adj.delta != 0:
                # Scale delta by direction and intensity
                scaled_delta = adj.delta * intent.direction * intent.intensity
                new_value = old_value + scaled_delta
            elif adj.factor != 1.0:
                # Scale factor by direction and intensity
                if intent.direction > 0:
                    scaled_factor = 1 + (adj.factor - 1) * intent.intensity
                else:
                    scaled_factor = 1 / (1 + (adj.factor - 1) * intent.intensity)
                new_value = old_value * scaled_factor
            
            # Clamp to valid range
            new_value = PARAMETERS[adj.parameter].clamp(new_value)
            
            if new_value != old_value:
                self._params[adj.parameter] = new_value
                changes[adj.parameter] = {
                    'from': old_value,
                    'to': new_value
                }
        
        return changes
    
    def get_parameters(self) -> Dict[str, float]:
        """Get current parameter values."""
        return dict(self._params)
    
    def get_parameter(self, name: str) -> Optional[float]:
        """Get a specific parameter."""
        return self._params.get(name)
    
    def set_parameter(self, name: str, value: float):
        """Set a parameter directly."""
        if name in PARAMETERS:
            self._params[name] = PARAMETERS[name].clamp(value)
    
    def reset(self):
        """Reset to defaults."""
        for name, range_def in PARAMETERS.items():
            self._params[name] = range_def.default
        self._system_prompt_modifiers.clear()
        self._history.clear()
    
    def describe_settings(self) -> str:
        """Describe current settings in natural language."""
        descriptions = []
        
        # Temperature description
        temp = self._params['temperature']
        if temp > 1.2:
            descriptions.append("very creative and unpredictable")
        elif temp > 0.9:
            descriptions.append("fairly creative")
        elif temp > 0.5:
            descriptions.append("balanced creativity")
        elif temp > 0.2:
            descriptions.append("precise and focused")
        else:
            descriptions.append("extremely deterministic")
        
        # Length description
        max_tok = self._params['max_tokens']
        if max_tok > 1500:
            descriptions.append("verbose output")
        elif max_tok < 300:
            descriptions.append("concise output")
        
        # Repetition
        rep_penalty = self._params['repetition_penalty']
        if rep_penalty > 1.3:
            descriptions.append("avoids repetition strongly")
        
        return "Currently configured for: " + ", ".join(descriptions)
    
    def configure_engine(self, engine):
        """
        Apply configuration to an inference engine.
        
        Args:
            engine: Inference engine instance
        """
        for param, value in self._params.items():
            if hasattr(engine, param):
                setattr(engine, param, value)
            elif hasattr(engine, 'config') and hasattr(engine.config, param):
                setattr(engine.config, param, value)
            elif hasattr(engine, 'generation_config'):
                setattr(engine.generation_config, param, value)
        
        logger.info(f"Applied config to engine: {self._params}")
    
    def get_history(self) -> List[Tuple[str, str]]:
        """Get history of applied instructions."""
        return [
            (instruction, intent.dimension.name)
            for instruction, intent in self._history
        ]


# Convenience functions
def configure_from_text(text: str, engine=None) -> Dict[str, Any]:
    """
    Quick function to configure from natural language.
    
    Args:
        text: Natural language instruction
        engine: Optional engine to configure
        
    Returns:
        Configuration result
    """
    config = NLModelConfig()
    result = config.apply(text)
    
    if engine:
        config.configure_engine(engine)
    
    return result


def get_preset_params(preset_name: str) -> Dict[str, float]:
    """Get parameters for a preset."""
    if preset_name in PRESETS:
        params = {}
        for name, range_def in PARAMETERS.items():
            params[name] = range_def.default
        params.update(PRESETS[preset_name])
        return params
    return {}
