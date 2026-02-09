"""
================================================================================
            TRAINER AI - THE AI THAT TRAINS OTHER AIs
================================================================================

The Trainer AI is a meta-AI system that helps prepare, curate, and generate
training data for any specialized model in the router system.

📍 FILE: enigma_engine/core/trainer_ai.py
🏷️ TYPE: Meta-AI / Data Curator
🎯 MAIN CLASS: TrainerAI

WHAT IT DOES:
    1. Generates training data for ANY router position (router, vision, code, etc.)
    2. Regulates data quality (format validation, deduplication, scoring)
    3. Curates existing data (filters, cleans, augments)
    4. Evaluates model outputs for quality
    5. Provides data templates for each position

ROUTER POSITIONS IT SUPPORTS:
    | Position   | Data Format                      | Purpose                    |
    |------------|----------------------------------|----------------------------|
    | router     | INPUT: text | INTENT: category   | Intent classification      |
    | vision     | IMAGE: desc | CAPTION: text      | Image description          |
    | code       | TASK: desc | CODE: implementation | Code generation           |
    | math       | PROBLEM: text | SOLUTION: steps  | Math reasoning             |
    | avatar     | COMMAND: text | BONES: json      | Avatar control             |
    | chat       | USER: text | ASSISTANT: response | Conversation               |

USAGE:
    from enigma_engine.core.trainer_ai import TrainerAI, get_trainer_ai
    
    trainer = get_trainer_ai()
    
    # Generate training data for the router
    data = trainer.generate_training_data("router", count=100)
    
    # Curate existing data
    clean_data = trainer.curate_data("router", raw_data)
    
    # Get data format template
    template = trainer.get_template("code")

SEE ALSO:
    - enigma_engine/core/tool_router.py - Uses these trained models
    - scripts/train_specialized_model.py - Trains the models
    - data/specialized/ - Training data files
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..tools.data_trainer import CharacterProfile, CharacterTrainer

from ..config import CONFIG

logger = logging.getLogger(__name__)


# =============================================================================
# DATA FORMAT DEFINITIONS FOR EACH POSITION
# =============================================================================

@dataclass
class PositionConfig:
    """Configuration for a router position's training data format."""
    name: str
    description: str
    input_prefix: str
    output_prefix: str
    separator: str
    example_count_min: int
    recommended_model_size: str
    validation_rules: List[str] = field(default_factory=list)
    

# Define all router positions and their data formats
POSITION_CONFIGS = {
    "router": PositionConfig(
        name="router",
        description="Intent classification - routes user input to the right tool",
        input_prefix="INPUT:",
        output_prefix="INTENT:",
        separator=" | ",
        example_count_min=100,
        recommended_model_size="nano",
        validation_rules=["intent_must_be_known", "no_empty_input"],
    ),
    "vision": PositionConfig(
        name="vision",
        description="Image captioning - describes visual content",
        input_prefix="IMAGE:",
        output_prefix="CAPTION:",
        separator=" | ",
        example_count_min=200,
        recommended_model_size="tiny",
        validation_rules=["caption_min_words_5", "no_empty_input"],
    ),
    "code": PositionConfig(
        name="code",
        description="Code generation - writes code from descriptions",
        input_prefix="TASK:",
        output_prefix="CODE:",
        separator="\n",
        example_count_min=100,
        recommended_model_size="small",
        validation_rules=["code_must_be_valid", "no_empty_task"],
    ),
    "math": PositionConfig(
        name="math",
        description="Mathematical reasoning - solves math problems step by step",
        input_prefix="PROBLEM:",
        output_prefix="SOLUTION:",
        separator="\n",
        example_count_min=150,
        recommended_model_size="small",
        validation_rules=["solution_has_steps", "no_empty_problem"],
    ),
    "avatar": PositionConfig(
        name="avatar",
        description="Avatar control - converts commands to bone movements",
        input_prefix="COMMAND:",
        output_prefix="BONES:",
        separator=" | ",
        example_count_min=200,
        recommended_model_size="tiny",
        validation_rules=["bones_is_valid_json", "no_empty_command"],
    ),
    "chat": PositionConfig(
        name="chat",
        description="Conversation - general chat responses",
        input_prefix="USER:",
        output_prefix="ASSISTANT:",
        separator="\n",
        example_count_min=500,
        recommended_model_size="small",
        validation_rules=["response_min_words_3", "no_empty_input"],
    ),
    "trainer": PositionConfig(
        name="trainer",
        description="Data generation - creates training data for other positions",
        input_prefix="GENERATE_FOR:",
        output_prefix="DATA:",
        separator="\n",
        example_count_min=50,
        recommended_model_size="small",
        validation_rules=["target_position_valid", "data_format_correct"],
    ),
    "teacher": PositionConfig(
        name="teacher",
        description="Meta-learning - teaches trainer AI how to create better training data",
        input_prefix="REQUEST:",
        output_prefix="GUIDANCE:",
        separator="\n",
        example_count_min=100,
        recommended_model_size="medium",
        validation_rules=["has_evaluation", "has_improvements", "quality_score_present"],
    ),
}


# =============================================================================
# QUICK-CREATE PRESETS
# =============================================================================

@dataclass
class QuickCreatePreset:
    """Preset configuration for one-click AI creation."""
    name: str
    description: str
    positions: List[str]  # Which router positions to train
    recommended_data_count: int  # How many training examples per position
    model_size: str  # Recommended model size
    tools_enabled: List[str]  # Which tools to enable in the AI
    

QUICK_CREATE_PRESETS = {
    "character_only": QuickCreatePreset(
        name="Character Only",
        description="Create a persona-based AI for conversations. No tools, just chat.",
        positions=["chat"],
        recommended_data_count=500,
        model_size="small",
        tools_enabled=[],
    ),
    "router_only": QuickCreatePreset(
        name="Router Only",
        description="Intent classifier that routes to existing tools. Quick to train.",
        positions=["router"],
        recommended_data_count=200,
        model_size="nano",
        tools_enabled=["image", "code", "chat", "search", "video", "audio", "3d"],
    ),
    "router_chat": QuickCreatePreset(
        name="Router + Chat",
        description="Routes intents AND handles conversations. Good balance.",
        positions=["router", "chat"],
        recommended_data_count=300,
        model_size="small",
        tools_enabled=["image", "code", "chat", "search"],
    ),
    "creative_ai": QuickCreatePreset(
        name="Creative AI",
        description="Focus on image, video, and 3D generation with chat.",
        positions=["router", "chat"],
        recommended_data_count=300,
        model_size="small",
        tools_enabled=["image", "video", "3d", "chat", "audio"],
    ),
    "coder_ai": QuickCreatePreset(
        name="Coder AI",
        description="Programming assistant with code generation focus.",
        positions=["router", "chat", "code"],
        recommended_data_count=400,
        model_size="small",
        tools_enabled=["code", "chat", "file", "search"],
    ),
    "full_router": QuickCreatePreset(
        name="Full Router",
        description="All specialized models for maximum routing capability.",
        positions=["router", "vision", "code", "math", "avatar"],
        recommended_data_count=200,
        model_size="small",
        tools_enabled=["image", "code", "video", "audio", "3d", "avatar", "math", "search", "file"],
    ),
    "complete_system": QuickCreatePreset(
        name="Complete System",
        description="Everything - all positions, trainer, and teacher. Requires most training.",
        positions=["router", "vision", "code", "math", "avatar", "chat", "trainer", "teacher"],
        recommended_data_count=300,
        model_size="medium",
        tools_enabled=["image", "code", "video", "audio", "3d", "avatar", "math", "search", "file", "chat"],
    ),
    "teacher_system": QuickCreatePreset(
        name="Teacher System",
        description="Meta-learning AI that teaches the Trainer AI. Self-improving.",
        positions=["teacher", "trainer"],
        recommended_data_count=150,
        model_size="medium",
        tools_enabled=["chat"],
    ),
}


# =============================================================================
# PROMPT-DRIVEN AI CREATION (Natural Language to AI)
# =============================================================================

@dataclass
class ParsedAISpec:
    """Parsed specification from a natural language prompt."""
    name: Optional[str] = None
    character_traits: List[str] = field(default_factory=list)
    speaking_style: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    positions: List[str] = field(default_factory=list)
    model_size: str = "small"
    personality_mode: str = "hybrid"
    restrictions: List[str] = field(default_factory=list)
    system_prompt_hints: List[str] = field(default_factory=list)
    confidence: float = 0.0  # How confident the parser is in the result


class PromptParser:
    """
    Parses natural language prompts to extract AI specifications.
    
    Examples:
        "Create an AI that speaks like Shakespeare and can write Python code"
        → character: Shakespearean, capabilities: [code], language: python
        
        "Make a friendly assistant that can search the web and describe images"
        → traits: friendly, capabilities: [search, vision]
        
        "I want an AI that talks like a pirate and generates images"
        → character: pirate, capabilities: [image]
    """
    
    # Keywords that indicate character/personality traits
    CHARACTER_PATTERNS = {
        # Historical/Literary figures
        r"(?:like|as)\s+(?:a\s+)?shakespeare(?:an)?": {"speaking_style": "shakespearean", "traits": ["eloquent", "poetic", "archaic"]},
        r"(?:like|as)\s+(?:a\s+)?pirate": {"speaking_style": "pirate", "traits": ["adventurous", "bold", "nautical"]},
        r"(?:like|as)\s+(?:a\s+)?(?:sci-?fi\s+)?robot": {"speaking_style": "robotic", "traits": ["logical", "precise", "technical"]},
        r"(?:like|as)\s+(?:a\s+)?professor": {"speaking_style": "academic", "traits": ["knowledgeable", "formal", "educational"]},
        r"(?:like|as)\s+(?:a\s+)?(?:southern|cowboy)": {"speaking_style": "southern", "traits": ["friendly", "folksy", "warm"]},
        r"(?:like|as)\s+(?:a\s+)?british": {"speaking_style": "british", "traits": ["polite", "formal", "witty"]},
        r"(?:like|as)\s+(?:a\s+)?(?:surfer|beach)": {"speaking_style": "casual", "traits": ["laid-back", "relaxed", "friendly"]},
        r"(?:like|as)\s+(?:a\s+)?ninja": {"speaking_style": "mysterious", "traits": ["stealthy", "wise", "disciplined"]},
        r"(?:like|as)\s+(?:a\s+)?wizard": {"speaking_style": "mystical", "traits": ["wise", "magical", "cryptic"]},
        
        # Personality traits
        r"\b(?:very\s+)?friendly\b": {"traits": ["friendly"]},
        r"\b(?:very\s+)?helpful\b": {"traits": ["helpful"]},
        r"\b(?:very\s+)?sarcastic\b": {"traits": ["sarcastic"]},
        r"\b(?:very\s+)?serious\b": {"traits": ["serious"]},
        r"\b(?:very\s+)?formal\b": {"traits": ["formal"]},
        r"\b(?:very\s+)?casual\b": {"traits": ["casual"]},
        r"\b(?:very\s+)?funny\b": {"traits": ["funny", "humorous"]},
        r"\b(?:very\s+)?professional\b": {"traits": ["professional"]},
        r"\b(?:very\s+)?creative\b": {"traits": ["creative"]},
        r"\b(?:very\s+)?technical\b": {"traits": ["technical"]},
        r"\b(?:very\s+)?brief|concise\b": {"traits": ["concise"]},
        r"\b(?:very\s+)?verbose|detailed\b": {"traits": ["detailed"]},
    }
    
    # Keywords that indicate capabilities
    CAPABILITY_PATTERNS = {
        # Code related
        r"(?:writes?|generates?|creates?)\s+(?:\w+\s+)?code": {"capabilities": ["code"], "positions": ["code", "router"]},
        r"\bpython\b": {"capabilities": ["code"], "language": "python"},
        r"\bjavascript|js\b": {"capabilities": ["code"], "language": "javascript"},
        r"\bprogramming|coding\b": {"capabilities": ["code"], "positions": ["code", "router"]},
        
        # Image related
        r"(?:generates?|creates?|makes?|draws?)\s+(?:\w+\s+)?images?": {"capabilities": ["image"], "positions": ["router"]},
        r"\bimage\s+generation|image\s+creator": {"capabilities": ["image"], "positions": ["router"]},
        r"\bdescribes?\s+(?:\w+\s+)?images?|image\s+description|image\s+analysis": {"capabilities": ["vision"], "positions": ["vision", "router"]},
        r"\bsees?\s+(?:images?|pictures?|photos?)": {"capabilities": ["vision"], "positions": ["vision", "router"]},
        
        # Search/web related
        r"search(?:es)?\s+(?:the\s+)?(?:web|internet)": {"capabilities": ["search"], "positions": ["router"]},
        r"\bweb\s+search|browses?\b": {"capabilities": ["search"], "positions": ["router"]},
        
        # Avatar related
        r"\bavatar|character\s+animation|bone\s+control": {"capabilities": ["avatar"], "positions": ["avatar", "router"]},
        r"\banimates?|moves?\s+character": {"capabilities": ["avatar"], "positions": ["avatar", "router"]},
        
        # Audio/voice related - be specific to avoid matching "speaks like"
        r"\b(?:with\s+)?voice\s+(?:output|synthesis)|tts|text.to.speech": {"capabilities": ["audio"], "positions": ["router"]},
        r"\breads?\s+(?:aloud|out\s+loud)|speaks?\s+(?:to\s+me|out|aloud)": {"capabilities": ["audio"], "positions": ["router"]},
        
        # Video related
        r"(?:generates?|creates?|makes?)\s+(?:\w+\s+)?videos?": {"capabilities": ["video"], "positions": ["router"]},
        
        # 3D related
        r"(?:generates?|creates?|makes?)\s+(?:\w+\s+)?3d|3d\s+models?": {"capabilities": ["3d"], "positions": ["router"]},
        
        # Math related
        r"\bmath|calculations?|solves?\s+(?:math\s+)?problems?\b": {"capabilities": ["math"], "positions": ["math", "router"]},
        
        # File related
        r"\bfile\s+access|reads?\s+files?|writes?\s+files?": {"capabilities": ["file"], "positions": ["router"]},
        
        # Chat (default if nothing else specified)
        r"\bchat|converses?|talks?\s+to|assistant\b": {"capabilities": ["chat"], "positions": ["chat"]},
    }
    
    # Keywords that indicate restrictions
    RESTRICTION_PATTERNS = {
        r"\bno\s+(?:bad\s+)?language|clean\b": {"restrictions": ["no_profanity"]},
        r"\bfamily.?friendly|kid.?safe|child.?friendly\b": {"restrictions": ["family_friendly"]},
        r"\bshort\s+(?:responses?|answers?)|brief\b": {"restrictions": ["keep_short"]},
        r"\bno\s+opinions?\b": {"restrictions": ["no_opinions"]},
        r"\bfactual\s+only|facts?\s+only\b": {"restrictions": ["factual_only"]},
    }
    
    # Model size indicators
    SIZE_PATTERNS = {
        r"\btiny|minimal|lightweight\b": "tiny",
        r"\bsmall|basic\b": "small",
        r"\bmedium|standard|normal\b": "medium",
        r"\blarge|powerful|advanced\b": "large",
        r"\bhuge|massive|maximum\b": "xl",
    }
    
    def __init__(self):
        """Initialize the prompt parser."""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self._character_re = [(re.compile(p, re.IGNORECASE), v) for p, v in self.CHARACTER_PATTERNS.items()]
        self._capability_re = [(re.compile(p, re.IGNORECASE), v) for p, v in self.CAPABILITY_PATTERNS.items()]
        self._restriction_re = [(re.compile(p, re.IGNORECASE), v) for p, v in self.RESTRICTION_PATTERNS.items()]
        self._size_re = [(re.compile(p, re.IGNORECASE), v) for p, v in self.SIZE_PATTERNS.items()]
    
    def parse(self, prompt: str) -> ParsedAISpec:
        """
        Parse a natural language prompt into an AI specification.
        
        Args:
            prompt: Natural language description of desired AI
        
        Returns:
            ParsedAISpec with extracted specifications
        """
        spec = ParsedAISpec()
        matches = 0
        total_patterns = len(self._character_re) + len(self._capability_re) + len(self._restriction_re)
        
        # Extract name if mentioned
        # Match "called X" or "named X" - quoted names can be multi-word
        name_match = re.search(r"(?:called?|named?)\s+['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
        if not name_match:
            # Single word name without quotes
            name_match = re.search(r"(?:called?|named?)\s+(\w+)\b", prompt, re.IGNORECASE)
        if name_match:
            spec.name = name_match.group(1).strip().title()
            matches += 1
        
        # Extract character traits and speaking style
        for pattern, values in self._character_re:
            if pattern.search(prompt):
                if "speaking_style" in values and not spec.speaking_style:
                    spec.speaking_style = values["speaking_style"]
                if "traits" in values:
                    spec.character_traits.extend(values["traits"])
                matches += 1
        
        # Extract capabilities and positions
        for pattern, values in self._capability_re:
            if pattern.search(prompt):
                if "capabilities" in values:
                    for cap in values["capabilities"]:
                        if cap not in spec.capabilities:
                            spec.capabilities.append(cap)
                if "positions" in values:
                    for pos in values["positions"]:
                        if pos not in spec.positions:
                            spec.positions.append(pos)
                matches += 1
        
        # Extract restrictions
        for pattern, values in self._restriction_re:
            if pattern.search(prompt):
                spec.restrictions.extend(values.get("restrictions", []))
                matches += 1
        
        # Extract model size preference
        for pattern, size in self._size_re:
            if pattern.search(prompt):
                spec.model_size = size
                matches += 1
                break
        
        # Determine personality mode based on traits
        if spec.speaking_style:
            # Strong speaking style suggests baking it in
            spec.personality_mode = "hybrid"
        elif spec.character_traits:
            # Traits only can be system prompt
            spec.personality_mode = "system_prompt"
        
        # Generate system prompt hints from traits
        if spec.character_traits:
            spec.system_prompt_hints.append(f"You have these traits: {', '.join(spec.character_traits)}")
        if spec.speaking_style:
            spec.system_prompt_hints.append(f"You speak in a {spec.speaking_style} style")
        if spec.restrictions:
            spec.system_prompt_hints.append(f"Restrictions: {', '.join(spec.restrictions)}")
        
        # Ensure at least chat capability
        if not spec.capabilities:
            spec.capabilities.append("chat")
        if not spec.positions:
            spec.positions.append("chat")
        
        # Always add router if there are multiple capabilities
        if len(spec.capabilities) > 1 and "router" not in spec.positions:
            spec.positions.insert(0, "router")
        
        # Remove duplicates while preserving order
        spec.character_traits = list(dict.fromkeys(spec.character_traits))
        spec.capabilities = list(dict.fromkeys(spec.capabilities))
        spec.positions = list(dict.fromkeys(spec.positions))
        
        # Calculate confidence
        if matches > 0:
            spec.confidence = min(1.0, matches / (total_patterns * 0.1))
        
        return spec
    
    def generate_summary(self, spec: ParsedAISpec) -> str:
        """Generate a human-readable summary of the parsed spec."""
        parts = []
        
        if spec.name:
            parts.append(f"Name: {spec.name}")
        
        if spec.speaking_style:
            parts.append(f"Speaking Style: {spec.speaking_style}")
        
        if spec.character_traits:
            parts.append(f"Personality: {', '.join(spec.character_traits)}")
        
        if spec.capabilities:
            parts.append(f"Capabilities: {', '.join(spec.capabilities)}")
        
        if spec.positions:
            parts.append(f"Training Positions: {', '.join(spec.positions)}")
        
        parts.append(f"Model Size: {spec.model_size}")
        parts.append(f"Personality Mode: {spec.personality_mode}")
        
        if spec.restrictions:
            parts.append(f"Restrictions: {', '.join(spec.restrictions)}")
        
        parts.append(f"Confidence: {spec.confidence:.0%}")
        
        return "\n".join(parts)


# Global parser instance
_prompt_parser: Optional[PromptParser] = None


def get_prompt_parser() -> PromptParser:
    """Get the global PromptParser instance."""
    global _prompt_parser
    if _prompt_parser is None:
        _prompt_parser = PromptParser()
    return _prompt_parser


# =============================================================================
# AI BUNDLE FORMAT SPECIFICATION
# =============================================================================

@dataclass
class AIBundleSpec:
    """Specification for an .enigma-bundle AI package.
    
    An AI bundle packages together:
    - Trained model weights for each position
    - Configuration and metadata
    - Persona settings (name, personality, system prompt)
    - Character profile (optional)
    - Tool permissions
    """
    name: str
    version: str
    description: str
    created: str
    author: str = "Anonymous"
    
    # Model weights paths (relative to bundle)
    models: Dict[str, str] = field(default_factory=dict)  # position -> weights path
    
    # Persona configuration
    persona_name: str = "AI Assistant"
    personality: str = ""
    speech_patterns: List[str] = field(default_factory=list)
    system_prompt: str = ""
    
    # Capabilities
    positions_trained: List[str] = field(default_factory=list)
    tools_enabled: List[str] = field(default_factory=list)
    
    # Character profile (optional)
    character_profile: Optional[Dict[str, Any]] = None
    
    # How personality was applied
    personality_mode: str = "system_prompt"  # "baked", "system_prompt", or "hybrid"
    
    # Bundle metadata
    preset_used: Optional[str] = None  # Which preset was used to create this
    training_epochs: int = 0
    quality_score: float = 0.0


def create_bundle_manifest(spec: AIBundleSpec) -> Dict[str, Any]:
    """Create a bundle manifest dict from spec."""
    return {
        "enigma_bundle_version": "1.0",
        "name": spec.name,
        "version": spec.version,
        "description": spec.description,
        "created": spec.created,
        "author": spec.author,
        "models": spec.models,
        "persona": {
            "name": spec.persona_name,
            "personality": spec.personality,
            "speech_patterns": spec.speech_patterns,
            "system_prompt": spec.system_prompt,
        },
        "capabilities": {
            "positions_trained": spec.positions_trained,
            "tools_enabled": spec.tools_enabled,
        },
        "character_profile": spec.character_profile,
        "personality_mode": spec.personality_mode,
        "metadata": {
            "preset_used": spec.preset_used,
            "training_epochs": spec.training_epochs,
            "quality_score": spec.quality_score,
        },
    }


def load_bundle_manifest(manifest_path: Path) -> AIBundleSpec:
    """Load a bundle from its manifest.json file."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    persona = data.get("persona", {})
    capabilities = data.get("capabilities", {})
    metadata = data.get("metadata", {})
    
    return AIBundleSpec(
        name=data["name"],
        version=data["version"],
        description=data["description"],
        created=data["created"],
        author=data.get("author", "Anonymous"),
        models=data.get("models", {}),
        persona_name=persona.get("name", "AI Assistant"),
        personality=persona.get("personality", ""),
        speech_patterns=persona.get("speech_patterns", []),
        system_prompt=persona.get("system_prompt", ""),
        positions_trained=capabilities.get("positions_trained", []),
        tools_enabled=capabilities.get("tools_enabled", []),
        character_profile=data.get("character_profile"),
        personality_mode=data.get("personality_mode", "system_prompt"),
        preset_used=metadata.get("preset_used"),
        training_epochs=metadata.get("training_epochs", 0),
        quality_score=metadata.get("quality_score", 0.0),
    )


# Known intents for router
KNOWN_INTENTS = [
    "chat", "image", "code", "video", "audio", "3d", "math",
    "search", "file", "settings", "help", "avatar", "memory"
]

# Example templates for generating synthetic data
SYNTHETIC_TEMPLATES = {
    "router": [
        ("draw {object}", "image"),
        ("paint {object}", "image"),
        ("create an image of {object}", "image"),
        ("generate a picture of {object}", "image"),
        ("write code to {task}", "code"),
        ("create a function that {task}", "code"),
        ("program a {object}", "code"),
        ("help me with {topic}", "chat"),
        ("explain {topic}", "chat"),
        ("what is {topic}", "chat"),
        ("tell me about {topic}", "chat"),
        ("make a video of {object}", "video"),
        ("animate {object}", "video"),
        ("say {text}", "audio"),
        ("speak {text}", "audio"),
        ("read {text} aloud", "audio"),
        ("create a 3d model of {object}", "3d"),
        ("sculpt {object}", "3d"),
        ("solve {math_problem}", "math"),
        ("calculate {math_problem}", "math"),
        ("search for {query}", "search"),
        ("find {query}", "search"),
        ("wave at me", "avatar"),
        ("dance", "avatar"),
        ("nod your head", "avatar"),
    ],
    "vision": [
        ("a {adj} {object} in a {location}", "The image shows a {adj} {object} situated in a {location}. The {object} appears clearly in the scene."),
        ("{object} on a {surface}", "A {object} is placed on top of a {surface}. The lighting highlights the {object}'s features."),
        ("person {action} near {object}", "A person is {action} near a {object}. The scene captures this moment clearly."),
    ],
    "code": {
        "python": [
            ("function to calculate factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
            ("function to reverse a string", "def reverse_string(s):\n    return s[::-1]"),
            ("function to check if palindrome", "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"),
            ("function to find max in list", "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)"),
            ("function to sort a list", "def sort_list(lst):\n    return sorted(lst)"),
        ],
    },
    "avatar": [
        ("wave", '{"right_arm": {"rotation": [0, 0, 45], "speed": 0.5}, "action": "wave"}'),
        ("nod", '{"head": {"rotation": [15, 0, 0], "speed": 0.3}, "action": "nod"}'),
        ("shake head", '{"head": {"rotation": [0, 30, 0], "speed": 0.4}, "action": "shake"}'),
        ("look left", '{"head": {"rotation": [0, -45, 0], "speed": 0.2}, "action": "look"}'),
        ("look right", '{"head": {"rotation": [0, 45, 0], "speed": 0.2}, "action": "look"}'),
        ("raise hand", '{"right_arm": {"rotation": [-90, 0, 0], "speed": 0.4}, "action": "raise"}'),
        ("dance", '{"full_body": {"animation": "dance", "speed": 1.0}, "action": "dance"}'),
        ("bow", '{"spine": {"rotation": [45, 0, 0], "speed": 0.3}, "action": "bow"}'),
    ],
    "teacher": [
        # Meta-learning templates: REQUEST -> GUIDANCE (evaluation + improvements)
        ("Train an AI that can classify user intents",
         "EVALUATION: Router training requires diverse intent examples covering all categories.\n"
         "QUALITY_SCORE: 0.85\n"
         "DATA_REQUIREMENTS: Minimum 100 examples, balanced across intents.\n"
         "IMPROVEMENTS: Include edge cases, ambiguous queries, multi-intent requests.\n"
         "SAMPLE: INPUT: draw me a cat | INTENT: image"),
        ("Train an AI that can describe images",
         "EVALUATION: Vision training needs varied image descriptions with rich vocabulary.\n"
         "QUALITY_SCORE: 0.80\n"
         "DATA_REQUIREMENTS: Minimum 200 examples with detailed captions.\n"
         "IMPROVEMENTS: Add sensory details, spatial relationships, emotional context.\n"
         "SAMPLE: IMAGE: sunset over ocean | CAPTION: A warm orange sun dips below the horizon"),
        ("Train an AI that can write code",
         "EVALUATION: Code training requires correct, runnable examples with clear task descriptions.\n"
         "QUALITY_SCORE: 0.90\n"
         "DATA_REQUIREMENTS: Minimum 100 examples covering multiple languages and patterns.\n"
         "IMPROVEMENTS: Add error handling, edge cases, performance considerations.\n"
         "SAMPLE: TASK: factorial function | CODE: def factorial(n): return 1 if n<=1 else n*factorial(n-1)"),
        ("Train an AI that can control avatar bones",
         "EVALUATION: Avatar training needs natural language to bone pose mappings.\n"
         "QUALITY_SCORE: 0.75\n"
         "DATA_REQUIREMENTS: Minimum 200 examples with valid JSON bone data.\n"
         "IMPROVEMENTS: Include compound movements, emotion expressions, body language.\n"
         "SAMPLE: COMMAND: wave hello | BONES: {\"right_arm\": {\"rotation\": [0, 0, 45]}}"),
        ("Evaluate this training data for quality",
         "EVALUATION: Analyzing format compliance, diversity, and coverage.\n"
         "QUALITY_SCORE: <calculated>\n"
         "FORMAT_ISSUES: List any malformed entries.\n"
         "DIVERSITY_ISSUES: Identify repetitive patterns.\n"
         "IMPROVEMENTS: Specific suggestions to improve data quality."),
        ("How do I create better training data",
         "EVALUATION: Quality training data has these characteristics:\n"
         "1. Correct format with clear input/output separation\n"
         "2. Diverse examples covering edge cases\n"
         "3. Consistent style and terminology\n"
         "4. Balanced distribution across categories\n"
         "QUALITY_SCORE: N/A (general guidance)\n"
         "IMPROVEMENTS: Start with templates, add variations, validate with scoring."),
    ],
}

# Word banks for synthetic data generation
WORD_BANKS = {
    "object": ["cat", "dog", "tree", "house", "car", "flower", "mountain", "river", 
               "bird", "fish", "robot", "dragon", "castle", "spaceship", "guitar"],
    "adj": ["beautiful", "colorful", "majestic", "tiny", "enormous", "ancient", 
            "modern", "mysterious", "bright", "dark", "serene", "chaotic"],
    "location": ["forest", "beach", "city", "desert", "space", "underwater", 
                 "mountain top", "garden", "office", "kitchen"],
    "surface": ["table", "desk", "floor", "shelf", "counter", "bed", "grass"],
    "topic": ["quantum physics", "machine learning", "history", "cooking", 
              "programming", "art", "music", "science", "philosophy"],
    "task": ["sort a list", "parse JSON", "connect to a database", "read a file",
             "send an email", "create a web server", "calculate fibonacci"],
    "math_problem": ["2 + 2", "the integral of x^2", "15% of 80", "solve for x: 2x + 5 = 15"],
    "text": ["hello world", "good morning", "the quick brown fox", "welcome home"],
    "action": ["standing", "sitting", "walking", "running", "jumping", "reading"],
    "query": ["python tutorials", "best restaurants", "weather forecast", "news today"],
}


# =============================================================================
# DATA QUALITY SCORING
# =============================================================================

@dataclass
class DataQualityScore:
    """Quality assessment of training data."""
    overall_score: float  # 0.0 - 1.0
    format_score: float
    diversity_score: float
    completeness_score: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


# =============================================================================
# MODEL QUALITY SCORING DASHBOARD
# =============================================================================

@dataclass
class ModelQualityMetrics:
    """Quality metrics for a trained model at a point in time."""
    model_id: str
    timestamp: str
    position: str
    
    # Training metrics
    final_loss: float = 0.0
    loss_trend: str = "stable"  # improving, stable, degrading
    training_epochs: int = 0
    training_examples: int = 0
    
    # Performance metrics
    response_quality: float = 0.0  # 0.0-1.0 from evaluations
    response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Data quality
    data_quality_score: float = 0.0
    data_diversity: float = 0.0
    
    # Health indicators
    overfitting_risk: float = 0.0  # 0.0-1.0
    needs_retraining: bool = False
    retraining_reason: Optional[str] = None


@dataclass
class ModelHealthStatus:
    """Overall health status for display in UI dashboard."""
    model_id: str
    position: str
    
    # Health grade (A-F)
    overall_grade: str = "C"  # A, B, C, D, F
    health_score: float = 0.5  # 0.0-1.0
    
    # Status message
    status: str = "healthy"  # healthy, warning, critical, needs_attention
    status_message: str = ""
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Last updated
    last_evaluation: str = ""
    last_training: str = ""


class QualityTracker:
    """
    Tracks model quality metrics over time for the Quality Scoring Dashboard.
    
    Provides:
    - Automatic quality metrics recording after training
    - AI Health status calculations
    - Trend analysis over training runs
    - Retraining recommendations
    
    Usage:
        tracker = get_quality_tracker()
        
        # Record metrics after training
        tracker.record_training_metrics(model_id, position, loss, epochs, data_count)
        
        # Get health status for dashboard
        health = tracker.get_model_health(model_id)
        
        # Get all models needing attention
        alerts = tracker.get_models_needing_attention()
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the quality tracker."""
        self.storage_path = storage_path or Path(CONFIG.get("data_dir", "data")) / "quality_tracking"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._metrics_cache: Dict[str, List[ModelQualityMetrics]] = {}
        self._health_cache: Dict[str, ModelHealthStatus] = {}
        
        # Load existing metrics
        self._load_metrics()
        
        logger.info(f"QualityTracker initialized at {self.storage_path}")
    
    def _load_metrics(self):
        """Load existing metrics from storage."""
        metrics_file = self.storage_path / "metrics_history.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    for model_id, metrics_list in data.items():
                        self._metrics_cache[model_id] = [
                            ModelQualityMetrics(**m) for m in metrics_list
                        ]
            except Exception as e:
                logger.warning(f"Failed to load metrics history: {e}")
    
    def _save_metrics(self):
        """Save metrics to storage."""
        metrics_file = self.storage_path / "metrics_history.json"
        try:
            data = {}
            for model_id, metrics_list in self._metrics_cache.items():
                data[model_id] = [
                    {
                        "model_id": m.model_id,
                        "timestamp": m.timestamp,
                        "position": m.position,
                        "final_loss": m.final_loss,
                        "loss_trend": m.loss_trend,
                        "training_epochs": m.training_epochs,
                        "training_examples": m.training_examples,
                        "response_quality": m.response_quality,
                        "response_time_ms": m.response_time_ms,
                        "memory_usage_mb": m.memory_usage_mb,
                        "data_quality_score": m.data_quality_score,
                        "data_diversity": m.data_diversity,
                        "overfitting_risk": m.overfitting_risk,
                        "needs_retraining": m.needs_retraining,
                        "retraining_reason": m.retraining_reason,
                    }
                    for m in metrics_list
                ]
            with open(metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")
    
    def record_training_metrics(
        self,
        model_id: str,
        position: str,
        final_loss: float,
        epochs: int,
        training_examples: int,
        loss_history: Optional[List[float]] = None,
        data_quality: Optional[DataQualityScore] = None,
    ) -> ModelQualityMetrics:
        """
        Record metrics after a training run.
        
        Args:
            model_id: Unique identifier for the model
            position: Router position (router, chat, code, etc.)
            final_loss: Final training loss
            epochs: Number of training epochs
            training_examples: Number of training examples used
            loss_history: Optional list of loss values per epoch
            data_quality: Optional data quality assessment
        
        Returns:
            ModelQualityMetrics with recorded data
        """
        from datetime import datetime
        
        # Determine loss trend
        loss_trend = "stable"
        if loss_history and len(loss_history) >= 3:
            early = sum(loss_history[:len(loss_history)//3]) / (len(loss_history)//3)
            late = sum(loss_history[-len(loss_history)//3:]) / (len(loss_history)//3)
            if late < early * 0.9:
                loss_trend = "improving"
            elif late > early * 1.1:
                loss_trend = "degrading"
        
        # Calculate overfitting risk
        overfitting_risk = 0.0
        if loss_history and len(loss_history) >= 5:
            # Check if loss started increasing after decreasing
            min_idx = loss_history.index(min(loss_history))
            if min_idx < len(loss_history) - 2:
                # Loss increased after minimum
                increase = loss_history[-1] - loss_history[min_idx]
                if increase > 0:
                    overfitting_risk = min(1.0, increase / loss_history[min_idx])
        
        # Determine if retraining needed
        needs_retraining = False
        retraining_reason = None
        
        if final_loss > 0.5:
            needs_retraining = True
            retraining_reason = "High final loss - model may not have learned well"
        elif overfitting_risk > 0.3:
            needs_retraining = True
            retraining_reason = "Overfitting detected - try more diverse data or fewer epochs"
        elif training_examples < 50:
            needs_retraining = True
            retraining_reason = "Insufficient training data - add more examples"
        
        metrics = ModelQualityMetrics(
            model_id=model_id,
            timestamp=datetime.now().isoformat(),
            position=position,
            final_loss=final_loss,
            loss_trend=loss_trend,
            training_epochs=epochs,
            training_examples=training_examples,
            data_quality_score=data_quality.overall_score if data_quality else 0.5,
            data_diversity=data_quality.diversity_score if data_quality else 0.5,
            overfitting_risk=overfitting_risk,
            needs_retraining=needs_retraining,
            retraining_reason=retraining_reason,
        )
        
        # Store in cache
        if model_id not in self._metrics_cache:
            self._metrics_cache[model_id] = []
        self._metrics_cache[model_id].append(metrics)
        
        # Keep only last 20 metrics per model
        if len(self._metrics_cache[model_id]) > 20:
            self._metrics_cache[model_id] = self._metrics_cache[model_id][-20:]
        
        # Update health cache
        self._update_health(model_id)
        
        # Save to disk
        self._save_metrics()
        
        return metrics
    
    def _update_health(self, model_id: str):
        """Update health status for a model."""
        metrics_list = self._metrics_cache.get(model_id, [])
        if not metrics_list:
            return
        
        latest = metrics_list[-1]
        
        # Calculate health score (0-1)
        health_components = [
            1.0 - min(1.0, latest.final_loss),  # Lower loss = better
            1.0 - latest.overfitting_risk,  # Lower overfitting = better
            latest.data_quality_score,  # Higher quality = better
            min(1.0, latest.training_examples / 200),  # More examples = better (up to 200)
        ]
        health_score = sum(health_components) / len(health_components)
        
        # Determine grade
        if health_score >= 0.9:
            grade = "A"
        elif health_score >= 0.8:
            grade = "B"
        elif health_score >= 0.6:
            grade = "C"
        elif health_score >= 0.4:
            grade = "D"
        else:
            grade = "F"
        
        # Determine status
        if latest.needs_retraining:
            status = "needs_attention"
            status_message = latest.retraining_reason or "Retraining recommended"
        elif health_score < 0.4:
            status = "critical"
            status_message = "Model quality is poor"
        elif health_score < 0.6:
            status = "warning"
            status_message = "Model could be improved"
        else:
            status = "healthy"
            status_message = "Model is performing well"
        
        # Generate recommendations
        recommendations = []
        if latest.final_loss > 0.3:
            recommendations.append("Consider training for more epochs")
        if latest.overfitting_risk > 0.2:
            recommendations.append("Add more diverse training data")
        if latest.training_examples < 100:
            recommendations.append(f"Add more training examples (currently {latest.training_examples})")
        if latest.data_quality_score < 0.6:
            recommendations.append("Improve training data quality")
        if not recommendations:
            recommendations.append("Model is in good shape!")
        
        self._health_cache[model_id] = ModelHealthStatus(
            model_id=model_id,
            position=latest.position,
            overall_grade=grade,
            health_score=health_score,
            status=status,
            status_message=status_message,
            recommendations=recommendations,
            last_evaluation=latest.timestamp,
            last_training=latest.timestamp,
        )
    
    def get_model_health(self, model_id: str) -> Optional[ModelHealthStatus]:
        """Get health status for a specific model."""
        return self._health_cache.get(model_id)
    
    def get_all_health_statuses(self) -> List[ModelHealthStatus]:
        """Get health status for all tracked models."""
        return list(self._health_cache.values())
    
    def get_models_needing_attention(self) -> List[ModelHealthStatus]:
        """Get models that need attention (retraining, poor health, etc.)."""
        return [
            h for h in self._health_cache.values()
            if h.status in ("needs_attention", "critical", "warning")
        ]
    
    def get_metrics_history(self, model_id: str, limit: int = 10) -> List[ModelQualityMetrics]:
        """Get metrics history for a model."""
        history = self._metrics_cache.get(model_id, [])
        return history[-limit:] if limit else history
    
    def get_improvement_trend(self, model_id: str) -> Dict[str, Any]:
        """Analyze improvement trend over training runs."""
        history = self._metrics_cache.get(model_id, [])
        if len(history) < 2:
            return {"trend": "insufficient_data", "runs": len(history)}
        
        losses = [m.final_loss for m in history]
        first_half = losses[:len(losses)//2]
        second_half = losses[len(losses)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg < first_avg * 0.9:
            trend = "improving"
        elif second_avg > first_avg * 1.1:
            trend = "degrading"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "runs": len(history),
            "first_avg_loss": first_avg,
            "recent_avg_loss": second_avg,
            "improvement_pct": (first_avg - second_avg) / first_avg * 100 if first_avg > 0 else 0,
        }
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get a summary for the quality dashboard UI."""
        all_health = list(self._health_cache.values())
        
        if not all_health:
            return {
                "total_models": 0,
                "healthy": 0,
                "warning": 0,
                "critical": 0,
                "grade_distribution": {},
                "models": [],
            }
        
        healthy = sum(1 for h in all_health if h.status == "healthy")
        warning = sum(1 for h in all_health if h.status in ("warning", "needs_attention"))
        critical = sum(1 for h in all_health if h.status == "critical")
        
        grades = {}
        for h in all_health:
            grades[h.overall_grade] = grades.get(h.overall_grade, 0) + 1
        
        return {
            "total_models": len(all_health),
            "healthy": healthy,
            "warning": warning,
            "critical": critical,
            "grade_distribution": grades,
            "models": [
                {
                    "id": h.model_id,
                    "position": h.position,
                    "grade": h.overall_grade,
                    "status": h.status,
                    "message": h.status_message,
                }
                for h in sorted(all_health, key=lambda x: x.health_score)
            ],
        }


# Global quality tracker instance
_quality_tracker: Optional[QualityTracker] = None


def get_quality_tracker() -> QualityTracker:
    """Get the global QualityTracker instance."""
    global _quality_tracker
    if _quality_tracker is None:
        _quality_tracker = QualityTracker()
    return _quality_tracker


# =============================================================================
# TRAINING DATA MARKETPLACE
# =============================================================================

@dataclass
class DataPack:
    """
    A shareable training data pack for the marketplace.
    
    Contains training data, metadata, and quality ratings.
    """
    pack_id: str
    name: str
    description: str
    author: str
    version: str = "1.0.0"
    
    # Content type
    position: str = "chat"  # router, chat, code, vision, avatar, etc.
    category: str = "general"  # general, character, task, style
    
    # Training data
    training_lines: List[str] = field(default_factory=list)
    example_count: int = 0
    
    # Character/Style info (for character packs)
    character_name: Optional[str] = None
    character_traits: List[str] = field(default_factory=list)
    speaking_style: Optional[str] = None
    system_prompt: Optional[str] = None
    
    # Quality metrics
    quality_score: float = 0.0
    community_rating: float = 0.0
    rating_count: int = 0
    download_count: int = 0
    
    # Metadata
    created_at: str = ""
    updated_at: str = ""
    tags: List[str] = field(default_factory=list)
    license: str = "MIT"
    
    # File info
    file_path: Optional[str] = None
    file_size_kb: float = 0.0


@dataclass
class DataPackRating:
    """A user rating for a data pack."""
    pack_id: str
    user_id: str
    rating: float  # 1-5 stars
    review: str = ""
    timestamp: str = ""


class DataMarketplace:
    """
    Training Data Marketplace for sharing and discovering training data packs.
    
    Features:
    - Create exportable data packs from training data
    - Import community data packs
    - Rate and review data packs
    - Search/filter available packs
    - Quality scoring integration
    
    Usage:
        marketplace = get_data_marketplace()
        
        # Create a pack from training data
        pack = marketplace.create_pack(
            name="Shakespeare Bot",
            description="Training data for Shakespearean speech",
            position="chat",
            training_file="data/shakespeare.txt",
            character_name="Will",
            speaking_style="Elizabethan English"
        )
        
        # Export for sharing
        marketplace.export_pack(pack.pack_id, "shakespeare_pack.zip")
        
        # Import a community pack
        imported = marketplace.import_pack("community_pack.zip")
        
        # Rate a pack
        marketplace.rate_pack(pack.pack_id, rating=4.5, review="Great quality!")
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the marketplace."""
        self.storage_path = storage_path or Path(CONFIG.get("data_dir", "data")) / "marketplace"
        self.packs_dir = self.storage_path / "packs"
        self.imports_dir = self.storage_path / "imports"
        self.exports_dir = self.storage_path / "exports"
        
        # Create directories
        for d in [self.packs_dir, self.imports_dir, self.exports_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self._packs_cache: Dict[str, DataPack] = {}
        self._ratings_cache: Dict[str, List[DataPackRating]] = {}
        
        # Load existing packs
        self._load_packs()
        
        logger.info(f"DataMarketplace initialized at {self.storage_path}")
    
    def _load_packs(self):
        """Load existing packs from storage."""
        index_file = self.storage_path / "pack_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for pack_data in data.get("packs", []):
                        pack = DataPack(**pack_data)
                        self._packs_cache[pack.pack_id] = pack
                    for pack_id, ratings in data.get("ratings", {}).items():
                        self._ratings_cache[pack_id] = [
                            DataPackRating(**r) for r in ratings
                        ]
            except Exception as e:
                logger.warning(f"Failed to load marketplace index: {e}")
    
    def _save_index(self):
        """Save pack index to storage."""
        index_file = self.storage_path / "pack_index.json"
        try:
            data = {
                "packs": [
                    {
                        "pack_id": p.pack_id,
                        "name": p.name,
                        "description": p.description,
                        "author": p.author,
                        "version": p.version,
                        "position": p.position,
                        "category": p.category,
                        "training_lines": [],  # Don't include in index
                        "example_count": p.example_count,
                        "character_name": p.character_name,
                        "character_traits": p.character_traits,
                        "speaking_style": p.speaking_style,
                        "system_prompt": p.system_prompt,
                        "quality_score": p.quality_score,
                        "community_rating": p.community_rating,
                        "rating_count": p.rating_count,
                        "download_count": p.download_count,
                        "created_at": p.created_at,
                        "updated_at": p.updated_at,
                        "tags": p.tags,
                        "license": p.license,
                        "file_path": p.file_path,
                        "file_size_kb": p.file_size_kb,
                    }
                    for p in self._packs_cache.values()
                ],
                "ratings": {
                    pack_id: [
                        {
                            "pack_id": r.pack_id,
                            "user_id": r.user_id,
                            "rating": r.rating,
                            "review": r.review,
                            "timestamp": r.timestamp,
                        }
                        for r in ratings
                    ]
                    for pack_id, ratings in self._ratings_cache.items()
                },
            }
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save marketplace index: {e}")
    
    def create_pack(
        self,
        name: str,
        description: str,
        author: str,
        position: str = "chat",
        category: str = "general",
        training_file: Optional[str] = None,
        training_data: Optional[str] = None,
        character_name: Optional[str] = None,
        character_traits: Optional[List[str]] = None,
        speaking_style: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tags: Optional[List[str]] = None,
        license_type: str = "MIT",
    ) -> DataPack:
        """
        Create a new data pack from training data.
        
        Args:
            name: Display name for the pack
            description: Description of what the pack trains
            author: Author/creator name
            position: Router position (chat, code, router, etc.)
            category: Pack category (general, character, task, style)
            training_file: Path to training data file
            training_data: Or raw training data string
            character_name: For character packs, the character's name
            character_traits: List of character traits
            speaking_style: Speaking style description
            system_prompt: System prompt for the character
            tags: Search tags
            license_type: License for sharing
        
        Returns:
            Created DataPack
        """
        from datetime import datetime
        import hashlib
        
        # Load training data
        lines = []
        if training_file and Path(training_file).exists():
            with open(training_file, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
        elif training_data:
            lines = [l.strip() for l in training_data.split('\n') if l.strip()]
        
        # Generate pack ID
        pack_id = hashlib.sha256(
            f"{name}{author}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Assess quality
        quality_score = 0.5
        if lines:
            # Simple quality heuristics
            has_prefix = sum(1 for l in lines if '<|' in l or 'USER:' in l.upper())
            quality_score = min(1.0, 0.3 + (has_prefix / len(lines)) * 0.4 + min(len(lines), 500) / 1000)
        
        now = datetime.now().isoformat()
        
        pack = DataPack(
            pack_id=pack_id,
            name=name,
            description=description,
            author=author,
            position=position,
            category=category,
            training_lines=lines,
            example_count=len(lines),
            character_name=character_name,
            character_traits=character_traits or [],
            speaking_style=speaking_style,
            system_prompt=system_prompt,
            quality_score=quality_score,
            created_at=now,
            updated_at=now,
            tags=tags or [],
            license=license_type,
        )
        
        # Save training data to file
        pack_file = self.packs_dir / f"{pack_id}.txt"
        with open(pack_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        pack.file_path = str(pack_file)
        pack.file_size_kb = pack_file.stat().st_size / 1024
        
        # Cache and save
        self._packs_cache[pack_id] = pack
        self._save_index()
        
        logger.info(f"Created data pack: {name} ({pack_id}) with {len(lines)} examples")
        return pack
    
    def export_pack(self, pack_id: str, output_path: Optional[str] = None) -> str:
        """
        Export a data pack as a shareable zip file.
        
        Args:
            pack_id: ID of pack to export
            output_path: Optional output path (defaults to exports dir)
        
        Returns:
            Path to exported zip file
        """
        import zipfile
        
        pack = self._packs_cache.get(pack_id)
        if not pack:
            raise ValueError(f"Pack not found: {pack_id}")
        
        # Default output path
        if not output_path:
            safe_name = "".join(c if c.isalnum() else "_" for c in pack.name)
            output_path = str(self.exports_dir / f"{safe_name}_{pack_id[:8]}.zip")
        
        # Load full training data
        if pack.file_path and Path(pack.file_path).exists():
            with open(pack.file_path, 'r', encoding='utf-8') as f:
                training_lines = f.read().split('\n')
        else:
            training_lines = pack.training_lines
        
        # Create zip
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Metadata
            metadata = {
                "pack_id": pack.pack_id,
                "name": pack.name,
                "description": pack.description,
                "author": pack.author,
                "version": pack.version,
                "position": pack.position,
                "category": pack.category,
                "example_count": len(training_lines),
                "character_name": pack.character_name,
                "character_traits": pack.character_traits,
                "speaking_style": pack.speaking_style,
                "system_prompt": pack.system_prompt,
                "quality_score": pack.quality_score,
                "community_rating": pack.community_rating,
                "tags": pack.tags,
                "license": pack.license,
                "created_at": pack.created_at,
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
            
            # Training data
            zf.writestr("training_data.txt", '\n'.join(training_lines))
            
            # README
            readme = f"""# {pack.name}

{pack.description}

## Details
- **Author**: {pack.author}
- **Position**: {pack.position}
- **Category**: {pack.category}
- **Examples**: {len(training_lines)}
- **Quality Score**: {pack.quality_score:.2f}
- **License**: {pack.license}

## Usage
Import this pack using the Enigma Engine Data Marketplace:
```python
from enigma_engine.core.trainer_ai import get_data_marketplace
marketplace = get_data_marketplace()
pack = marketplace.import_pack("path/to/{Path(output_path).name}")
```

## Tags
{', '.join(pack.tags) if pack.tags else 'None'}
"""
            zf.writestr("README.md", readme)
        
        pack.download_count += 1
        self._save_index()
        
        logger.info(f"Exported pack to: {output_path}")
        return output_path
    
    def import_pack(self, zip_path: str) -> DataPack:
        """
        Import a data pack from a zip file.
        
        Args:
            zip_path: Path to the zip file
        
        Returns:
            Imported DataPack
        """
        import zipfile
        from datetime import datetime
        
        if not Path(zip_path).exists():
            raise FileNotFoundError(f"Pack file not found: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Load metadata
            metadata = json.loads(zf.read("metadata.json").decode('utf-8'))
            
            # Load training data
            training_data = zf.read("training_data.txt").decode('utf-8')
            lines = [l.strip() for l in training_data.split('\n') if l.strip()]
        
        # Create pack from metadata
        pack_id = metadata.get("pack_id", "")
        
        # Check if already imported
        if pack_id in self._packs_cache:
            logger.info(f"Pack {pack_id} already exists, updating...")
        
        pack = DataPack(
            pack_id=pack_id,
            name=metadata.get("name", "Unknown"),
            description=metadata.get("description", ""),
            author=metadata.get("author", "Unknown"),
            version=metadata.get("version", "1.0.0"),
            position=metadata.get("position", "chat"),
            category=metadata.get("category", "general"),
            training_lines=lines,
            example_count=len(lines),
            character_name=metadata.get("character_name"),
            character_traits=metadata.get("character_traits", []),
            speaking_style=metadata.get("speaking_style"),
            system_prompt=metadata.get("system_prompt"),
            quality_score=metadata.get("quality_score", 0.5),
            community_rating=metadata.get("community_rating", 0.0),
            created_at=metadata.get("created_at", ""),
            updated_at=datetime.now().isoformat(),
            tags=metadata.get("tags", []),
            license=metadata.get("license", "MIT"),
        )
        
        # Save training data locally
        pack_file = self.imports_dir / f"{pack_id}.txt"
        with open(pack_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        pack.file_path = str(pack_file)
        pack.file_size_kb = pack_file.stat().st_size / 1024
        
        # Cache and save
        self._packs_cache[pack_id] = pack
        self._save_index()
        
        logger.info(f"Imported pack: {pack.name} ({pack_id}) with {len(lines)} examples")
        return pack
    
    def rate_pack(
        self,
        pack_id: str,
        rating: float,
        review: str = "",
        user_id: str = "anonymous",
    ) -> DataPackRating:
        """
        Rate a data pack.
        
        Args:
            pack_id: ID of pack to rate
            rating: Rating 1-5 stars
            review: Optional review text
            user_id: User identifier
        
        Returns:
            Created rating
        """
        from datetime import datetime
        
        if pack_id not in self._packs_cache:
            raise ValueError(f"Pack not found: {pack_id}")
        
        rating = max(1.0, min(5.0, rating))  # Clamp 1-5
        
        pack_rating = DataPackRating(
            pack_id=pack_id,
            user_id=user_id,
            rating=rating,
            review=review,
            timestamp=datetime.now().isoformat(),
        )
        
        # Add to ratings
        if pack_id not in self._ratings_cache:
            self._ratings_cache[pack_id] = []
        self._ratings_cache[pack_id].append(pack_rating)
        
        # Update pack's community rating
        pack = self._packs_cache[pack_id]
        all_ratings = [r.rating for r in self._ratings_cache[pack_id]]
        pack.community_rating = sum(all_ratings) / len(all_ratings)
        pack.rating_count = len(all_ratings)
        
        self._save_index()
        
        logger.info(f"Rated pack {pack_id}: {rating}/5 stars")
        return pack_rating
    
    def get_pack(self, pack_id: str) -> Optional[DataPack]:
        """Get a pack by ID."""
        return self._packs_cache.get(pack_id)
    
    def get_pack_training_data(self, pack_id: str) -> str:
        """Get the full training data for a pack."""
        pack = self._packs_cache.get(pack_id)
        if not pack:
            return ""
        
        if pack.file_path and Path(pack.file_path).exists():
            with open(pack.file_path, 'r', encoding='utf-8') as f:
                return f.read()
        return '\n'.join(pack.training_lines)
    
    def list_packs(
        self,
        position: Optional[str] = None,
        category: Optional[str] = None,
        min_rating: float = 0.0,
        search: Optional[str] = None,
        sort_by: str = "rating",
    ) -> List[DataPack]:
        """
        List and filter available packs.
        
        Args:
            position: Filter by router position
            category: Filter by category
            min_rating: Minimum community rating
            search: Search in name/description/tags
            sort_by: Sort by "rating", "downloads", "quality", "newest"
        
        Returns:
            List of matching DataPacks
        """
        packs = list(self._packs_cache.values())
        
        # Apply filters
        if position:
            packs = [p for p in packs if p.position == position]
        
        if category:
            packs = [p for p in packs if p.category == category]
        
        if min_rating > 0:
            packs = [p for p in packs if p.community_rating >= min_rating]
        
        if search:
            search_lower = search.lower()
            packs = [
                p for p in packs
                if search_lower in p.name.lower()
                or search_lower in p.description.lower()
                or any(search_lower in t.lower() for t in p.tags)
            ]
        
        # Sort
        if sort_by == "rating":
            packs.sort(key=lambda p: p.community_rating, reverse=True)
        elif sort_by == "downloads":
            packs.sort(key=lambda p: p.download_count, reverse=True)
        elif sort_by == "quality":
            packs.sort(key=lambda p: p.quality_score, reverse=True)
        elif sort_by == "newest":
            packs.sort(key=lambda p: p.created_at, reverse=True)
        
        return packs
    
    def get_pack_ratings(self, pack_id: str) -> List[DataPackRating]:
        """Get all ratings for a pack."""
        return self._ratings_cache.get(pack_id, [])
    
    def delete_pack(self, pack_id: str) -> bool:
        """Delete a pack from the marketplace."""
        pack = self._packs_cache.get(pack_id)
        if not pack:
            return False
        
        # Delete training file
        if pack.file_path and Path(pack.file_path).exists():
            Path(pack.file_path).unlink()
        
        # Remove from caches
        del self._packs_cache[pack_id]
        if pack_id in self._ratings_cache:
            del self._ratings_cache[pack_id]
        
        self._save_index()
        logger.info(f"Deleted pack: {pack_id}")
        return True
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        packs = list(self._packs_cache.values())
        
        if not packs:
            return {
                "total_packs": 0,
                "total_examples": 0,
                "categories": {},
                "positions": {},
                "avg_rating": 0.0,
                "top_rated": [],
                "most_downloaded": [],
            }
        
        categories = {}
        positions = {}
        for p in packs:
            categories[p.category] = categories.get(p.category, 0) + 1
            positions[p.position] = positions.get(p.position, 0) + 1
        
        rated_packs = [p for p in packs if p.rating_count > 0]
        avg_rating = (
            sum(p.community_rating for p in rated_packs) / len(rated_packs)
            if rated_packs else 0.0
        )
        
        return {
            "total_packs": len(packs),
            "total_examples": sum(p.example_count for p in packs),
            "categories": categories,
            "positions": positions,
            "avg_rating": avg_rating,
            "top_rated": sorted(packs, key=lambda p: p.community_rating, reverse=True)[:5],
            "most_downloaded": sorted(packs, key=lambda p: p.download_count, reverse=True)[:5],
        }


# Global marketplace instance
_data_marketplace: Optional[DataMarketplace] = None


def get_data_marketplace() -> DataMarketplace:
    """Get the global DataMarketplace instance."""
    global _data_marketplace
    if _data_marketplace is None:
        _data_marketplace = DataMarketplace()
    return _data_marketplace


# =============================================================================
# INCREMENTAL TRAINING
# =============================================================================

@dataclass
class TrainingHistory:
    """Record of what a model has been trained on."""
    model_id: str
    training_runs: List[Dict[str, Any]] = field(default_factory=list)
    skills_learned: List[str] = field(default_factory=list)
    total_epochs: int = 0
    total_examples: int = 0
    created_at: str = ""
    last_trained: str = ""


@dataclass 
class SkillDefinition:
    """Definition of a trainable skill."""
    name: str
    description: str
    position: str  # router position this skill applies to
    training_data: str = ""  # Training data for this skill
    example_count: int = 0
    templates: List[str] = field(default_factory=list)


class IncrementalTrainer:
    """
    Incremental Training system for adding capabilities without full retrain.
    
    Features:
    - "Teach" existing models new skills incrementally
    - Track what models have learned
    - Merge trained models
    - Fine-tune on user conversations (with consent)
    
    Usage:
        trainer = get_incremental_trainer()
        
        # Teach a new skill
        result = trainer.teach_skill(
            model_path="models/my_chatbot",
            skill_name="math",
            training_data="USER: What is 2+2?\\nASSISTANT: 4",
            epochs=5
        )
        
        # Get model's learning history
        history = trainer.get_training_history("my_chatbot")
        
        # Merge two models
        merged = trainer.merge_models(
            base_model="models/base",
            skill_model="models/math_specialist",
            output_path="models/merged",
            skill_weight=0.3
        )
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the incremental trainer."""
        self.storage_path = storage_path or Path(CONFIG.get("data_dir", "data")) / "incremental_training"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._history_cache: Dict[str, TrainingHistory] = {}
        self._skills_cache: Dict[str, SkillDefinition] = {}
        
        # Load existing data
        self._load_data()
        
        # Built-in skill definitions
        self._init_builtin_skills()
        
        logger.info(f"IncrementalTrainer initialized at {self.storage_path}")
    
    def _load_data(self):
        """Load existing training history and skills."""
        history_file = self.storage_path / "training_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for model_id, hist_data in data.get("history", {}).items():
                        self._history_cache[model_id] = TrainingHistory(**hist_data)
                    for skill_name, skill_data in data.get("skills", {}).items():
                        self._skills_cache[skill_name] = SkillDefinition(**skill_data)
            except Exception as e:
                logger.warning(f"Failed to load incremental training data: {e}")
    
    def _save_data(self):
        """Save training history and skills."""
        history_file = self.storage_path / "training_history.json"
        try:
            data = {
                "history": {
                    model_id: {
                        "model_id": h.model_id,
                        "training_runs": h.training_runs,
                        "skills_learned": h.skills_learned,
                        "total_epochs": h.total_epochs,
                        "total_examples": h.total_examples,
                        "created_at": h.created_at,
                        "last_trained": h.last_trained,
                    }
                    for model_id, h in self._history_cache.items()
                },
                "skills": {
                    name: {
                        "name": s.name,
                        "description": s.description,
                        "position": s.position,
                        "training_data": "",  # Don't persist training data in index
                        "example_count": s.example_count,
                        "templates": s.templates,
                    }
                    for name, s in self._skills_cache.items()
                },
            }
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save incremental training data: {e}")
    
    def _init_builtin_skills(self):
        """Initialize built-in skill definitions."""
        builtin_skills = [
            SkillDefinition(
                name="math",
                description="Basic math calculations and reasoning",
                position="math",
                templates=[
                    "USER: What is {a} + {b}?\nASSISTANT: {result}",
                    "USER: Calculate {a} * {b}\nASSISTANT: {result}",
                    "USER: What is {a} divided by {b}?\nASSISTANT: {result}",
                ],
            ),
            SkillDefinition(
                name="coding",
                description="Code generation and explanation",
                position="code",
                templates=[
                    "USER: Write a function to {task}\nASSISTANT: ```{lang}\n{code}\n```",
                    "USER: Explain this code: {code}\nASSISTANT: {explanation}",
                ],
            ),
            SkillDefinition(
                name="storytelling",
                description="Creative story generation",
                position="chat",
                templates=[
                    "USER: Tell me a story about {topic}\nASSISTANT: {story}",
                    "USER: Continue this story: {prompt}\nASSISTANT: {continuation}",
                ],
            ),
            SkillDefinition(
                name="roleplay",
                description="Character roleplay and acting",
                position="chat",
                templates=[
                    "USER: Pretend you are {character}\nASSISTANT: *{action}* {dialogue}",
                    "USER: As {character}, respond to: {prompt}\nASSISTANT: {response}",
                ],
            ),
            SkillDefinition(
                name="summarization",
                description="Text summarization",
                position="chat",
                templates=[
                    "USER: Summarize this: {text}\nASSISTANT: {summary}",
                    "USER: Give me the key points of: {text}\nASSISTANT: {points}",
                ],
            ),
        ]
        
        for skill in builtin_skills:
            if skill.name not in self._skills_cache:
                self._skills_cache[skill.name] = skill
    
    def register_skill(
        self,
        name: str,
        description: str,
        position: str = "chat",
        training_data: Optional[str] = None,
        templates: Optional[List[str]] = None,
    ) -> SkillDefinition:
        """
        Register a new skill definition.
        
        Args:
            name: Unique skill name
            description: What this skill does
            position: Router position (chat, code, math, etc.)
            training_data: Optional training data string
            templates: Optional training data templates
        
        Returns:
            Created SkillDefinition
        """
        example_count = 0
        if training_data:
            example_count = len([l for l in training_data.split('\n') if l.strip()])
        
        skill = SkillDefinition(
            name=name,
            description=description,
            position=position,
            training_data=training_data or "",
            example_count=example_count,
            templates=templates or [],
        )
        
        self._skills_cache[name] = skill
        self._save_data()
        
        logger.info(f"Registered skill: {name}")
        return skill
    
    def get_skill(self, name: str) -> Optional[SkillDefinition]:
        """Get a skill definition by name."""
        return self._skills_cache.get(name)
    
    def list_skills(self) -> List[SkillDefinition]:
        """List all available skills."""
        return list(self._skills_cache.values())
    
    def teach_skill(
        self,
        model_id: str,
        skill_name: str,
        training_data: Optional[str] = None,
        epochs: int = 5,
        learning_rate: float = 0.0001,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Teach a skill to an existing model incrementally.
        
        This performs a small, focused training run on the new skill
        without forgetting previous training.
        
        Args:
            model_id: Identifier for the model
            skill_name: Name of skill to teach
            training_data: Optional custom training data (overrides skill's data)
            epochs: Number of training epochs (keep low to avoid forgetting)
            learning_rate: Learning rate (keep low for incremental training)
            model_path: Path to model weights
        
        Returns:
            Dict with training results
        """
        from datetime import datetime
        
        skill = self._skills_cache.get(skill_name)
        if not skill and not training_data:
            return {
                "success": False,
                "error": f"Skill '{skill_name}' not found and no training_data provided",
            }
        
        # Get training data
        data = training_data or (skill.training_data if skill else "")
        if not data:
            return {
                "success": False,
                "error": "No training data available for this skill",
            }
        
        examples = [l for l in data.split('\n') if l.strip()]
        
        # Record the training run
        now = datetime.now().isoformat()
        run_record = {
            "skill": skill_name,
            "epochs": epochs,
            "examples": len(examples),
            "learning_rate": learning_rate,
            "timestamp": now,
            "model_path": model_path,
        }
        
        # Update history
        if model_id not in self._history_cache:
            self._history_cache[model_id] = TrainingHistory(
                model_id=model_id,
                created_at=now,
            )
        
        history = self._history_cache[model_id]
        history.training_runs.append(run_record)
        if skill_name not in history.skills_learned:
            history.skills_learned.append(skill_name)
        history.total_epochs += epochs
        history.total_examples += len(examples)
        history.last_trained = now
        
        self._save_data()
        
        # Note: Actual PyTorch training would happen here if model_path is provided
        # For now, we just record the intent and return guidance
        
        result = {
            "success": True,
            "model_id": model_id,
            "skill": skill_name,
            "epochs": epochs,
            "examples_count": len(examples),
            "skills_learned": history.skills_learned,
            "total_training_runs": len(history.training_runs),
            "guidance": f"To actually train, run: trainer.train_model('{model_path}', data, epochs={epochs}, lr={learning_rate})",
            "training_data_preview": examples[:3] if examples else [],
        }
        
        logger.info(f"Recorded skill training: {skill_name} for {model_id}")
        return result
    
    def get_training_history(self, model_id: str) -> Optional[TrainingHistory]:
        """Get the training history for a model."""
        return self._history_cache.get(model_id)
    
    def merge_models(
        self,
        base_model_path: str,
        skill_model_path: str,
        output_path: str,
        skill_weight: float = 0.3,
        merge_method: str = "weighted_average",
    ) -> Dict[str, Any]:
        """
        Merge a skill model into a base model.
        
        This allows combining a general model with a specialized one.
        
        Args:
            base_model_path: Path to base model weights
            skill_model_path: Path to skill/specialized model weights
            output_path: Where to save merged model
            skill_weight: Weight for skill model (0.0-1.0)
            merge_method: "weighted_average", "slerp", or "ties"
        
        Returns:
            Dict with merge results
        
        Note:
            For actual weight merging, both models must have the same architecture.
            This method records the merge and provides guidance for PyTorch merging.
        """
        result = {
            "success": True,
            "base_model": base_model_path,
            "skill_model": skill_model_path,
            "output_path": output_path,
            "skill_weight": skill_weight,
            "merge_method": merge_method,
            "instructions": [],
        }
        
        # Check if paths exist
        if not Path(base_model_path).exists():
            result["success"] = False
            result["error"] = f"Base model not found: {base_model_path}"
            return result
        
        if not Path(skill_model_path).exists():
            result["success"] = False
            result["error"] = f"Skill model not found: {skill_model_path}"
            return result
        
        # Provide PyTorch merge instructions
        result["instructions"] = [
            "To merge models with PyTorch:",
            "1. Load both model state dicts",
            "2. For weighted average: merged[key] = base[key] * (1-weight) + skill[key] * weight",
            "3. Save merged state dict",
            f"",
            f"Example code:",
            f"  import torch",
            f"  base = torch.load('{base_model_path}')",
            f"  skill = torch.load('{skill_model_path}')",
            f"  merged = {{}}",
            f"  for key in base:",
            f"      merged[key] = base[key] * {1-skill_weight:.1f} + skill[key] * {skill_weight:.1f}",
            f"  torch.save(merged, '{output_path}')",
        ]
        
        logger.info(f"Merge guidance generated for {base_model_path} + {skill_model_path}")
        return result
    
    def record_conversation_learning(
        self,
        model_id: str,
        conversation: List[Dict[str, str]],
        user_approved: bool = False,
    ) -> Dict[str, Any]:
        """
        Record a conversation for potential future training (with user consent).
        
        Args:
            model_id: Model identifier
            conversation: List of {"role": "user"|"assistant", "content": str}
            user_approved: Whether user has approved using this for training
        
        Returns:
            Dict with recording status
        """
        from datetime import datetime
        
        if not user_approved:
            return {
                "success": False,
                "reason": "User consent required for conversation learning",
                "message": "Set user_approved=True after getting explicit user consent",
            }
        
        # Convert to training format
        training_lines = []
        for i in range(0, len(conversation) - 1, 2):
            if i + 1 < len(conversation):
                user_msg = conversation[i]
                asst_msg = conversation[i + 1]
                if user_msg.get("role") == "user" and asst_msg.get("role") == "assistant":
                    line = f"USER: {user_msg['content']}\nASSISTANT: {asst_msg['content']}"
                    training_lines.append(line)
        
        if not training_lines:
            return {
                "success": False,
                "reason": "No valid conversation pairs found",
            }
        
        # Save conversation data
        conv_dir = self.storage_path / "conversations" / model_id
        conv_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conv_file = conv_dir / f"conv_{timestamp}.txt"
        
        with open(conv_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(training_lines))
        
        logger.info(f"Recorded {len(training_lines)} conversation pairs for {model_id}")
        
        return {
            "success": True,
            "model_id": model_id,
            "pairs_recorded": len(training_lines),
            "saved_to": str(conv_file),
            "message": "Conversation recorded. Use teach_skill with this data to train.",
        }
    
    def get_pending_conversations(self, model_id: str) -> List[str]:
        """Get recorded conversations pending training."""
        conv_dir = self.storage_path / "conversations" / model_id
        if not conv_dir.exists():
            return []
        
        return sorted([str(f) for f in conv_dir.glob("*.txt")])


# Global incremental trainer instance
_incremental_trainer: Optional[IncrementalTrainer] = None


def get_incremental_trainer() -> IncrementalTrainer:
    """Get the global IncrementalTrainer instance."""
    global _incremental_trainer
    if _incremental_trainer is None:
        _incremental_trainer = IncrementalTrainer()
    return _incremental_trainer


# =============================================================================
# MODEL INHERITANCE
# =============================================================================

@dataclass
class ModelLineage:
    """Records the lineage/ancestry of a model."""
    model_id: str
    parent_id: Optional[str] = None
    parent_name: Optional[str] = None
    
    # What was inherited
    inherited_weights: bool = False
    inherited_config: bool = False
    inherited_personality: bool = False
    
    # Customizations applied
    customizations: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: str = ""
    forked_from_bundle: Optional[str] = None


@dataclass
class CloneConfig:
    """Configuration for cloning a model."""
    name: str
    description: str = ""
    
    # What to inherit
    inherit_weights: bool = True
    inherit_personality: bool = True
    inherit_training_data: bool = False
    
    # What to customize
    new_personality: Optional[str] = None
    new_traits: Optional[List[str]] = None
    new_speaking_style: Optional[str] = None
    
    # Capabilities
    add_capabilities: List[str] = field(default_factory=list)
    remove_capabilities: List[str] = field(default_factory=list)


class ModelInheritance:
    """
    Model Inheritance system for creating new AIs from existing ones.
    
    Features:
    - Clone existing models as base for new ones
    - Inherit personality but change capabilities
    - Fork community AI bundles
    - Track model lineage/ancestry
    
    Usage:
        inheritance = get_model_inheritance()
        
        # Clone a model with customizations
        new_model = inheritance.clone_model(
            source_model_id="helpful_assistant",
            config=CloneConfig(
                name="pirate_assistant",
                inherit_personality=True,
                new_speaking_style="pirate",
                add_capabilities=["storytelling"]
            )
        )
        
        # Fork a community bundle
        forked = inheritance.fork_bundle("community_bundle.zip", "my_version")
        
        # Get model lineage
        lineage = inheritance.get_lineage("my_model")
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the model inheritance system."""
        self.storage_path = storage_path or Path(CONFIG.get("data_dir", "data")) / "model_inheritance"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._lineage_cache: Dict[str, ModelLineage] = {}
        
        # Load existing lineage data
        self._load_lineage()
        
        logger.info(f"ModelInheritance initialized at {self.storage_path}")
    
    def _load_lineage(self):
        """Load existing lineage data."""
        lineage_file = self.storage_path / "lineage.json"
        if lineage_file.exists():
            try:
                with open(lineage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for model_id, lineage_data in data.items():
                        self._lineage_cache[model_id] = ModelLineage(**lineage_data)
            except Exception as e:
                logger.warning(f"Failed to load lineage data: {e}")
    
    def _save_lineage(self):
        """Save lineage data."""
        lineage_file = self.storage_path / "lineage.json"
        try:
            data = {
                model_id: {
                    "model_id": l.model_id,
                    "parent_id": l.parent_id,
                    "parent_name": l.parent_name,
                    "inherited_weights": l.inherited_weights,
                    "inherited_config": l.inherited_config,
                    "inherited_personality": l.inherited_personality,
                    "customizations": l.customizations,
                    "created_at": l.created_at,
                    "forked_from_bundle": l.forked_from_bundle,
                }
                for model_id, l in self._lineage_cache.items()
            }
            with open(lineage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save lineage data: {e}")
    
    def clone_model(
        self,
        source_model_id: str,
        config: CloneConfig,
        source_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Clone an existing model with customizations.
        
        Args:
            source_model_id: ID of the model to clone
            config: CloneConfig with clone settings
            source_path: Path to source model (optional)
            output_path: Where to save cloned model (optional)
        
        Returns:
            Dict with clone results and new model info
        """
        from datetime import datetime
        import hashlib
        
        # Generate new model ID
        new_model_id = hashlib.sha256(
            f"{config.name}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        result = {
            "success": True,
            "new_model_id": new_model_id,
            "name": config.name,
            "parent_id": source_model_id,
            "inherited": [],
            "customizations": [],
        }
        
        # Track what was inherited
        if config.inherit_weights:
            result["inherited"].append("weights")
        if config.inherit_personality:
            result["inherited"].append("personality")
        if config.inherit_training_data:
            result["inherited"].append("training_data")
        
        # Track customizations
        customizations = []
        if config.new_personality:
            customizations.append(f"personality: {config.new_personality}")
        if config.new_traits:
            customizations.append(f"traits: {', '.join(config.new_traits)}")
        if config.new_speaking_style:
            customizations.append(f"style: {config.new_speaking_style}")
        if config.add_capabilities:
            customizations.append(f"added: {', '.join(config.add_capabilities)}")
        if config.remove_capabilities:
            customizations.append(f"removed: {', '.join(config.remove_capabilities)}")
        
        result["customizations"] = customizations
        
        # Create lineage record
        lineage = ModelLineage(
            model_id=new_model_id,
            parent_id=source_model_id,
            parent_name=config.name,
            inherited_weights=config.inherit_weights,
            inherited_config=True,
            inherited_personality=config.inherit_personality,
            customizations=customizations,
            created_at=datetime.now().isoformat(),
        )
        
        self._lineage_cache[new_model_id] = lineage
        self._save_lineage()
        
        # Generate new config
        new_config = {
            "model_id": new_model_id,
            "name": config.name,
            "description": config.description,
            "parent_id": source_model_id,
            "speaking_style": config.new_speaking_style,
            "character_traits": config.new_traits or [],
            "capabilities": config.add_capabilities,
        }
        
        result["new_config"] = new_config
        
        # Provide copy instructions if paths provided
        if source_path and output_path:
            result["copy_instructions"] = [
                f"To complete the clone:",
                f"1. Copy weights: {source_path} -> {output_path}",
                f"2. Update config with new_config",
                f"3. Train on new personality/capabilities as needed",
            ]
        
        logger.info(f"Cloned model {source_model_id} -> {new_model_id} ({config.name})")
        return result
    
    def fork_bundle(
        self,
        bundle_path: str,
        new_name: str,
        customize: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Fork a community AI bundle to create your own version.
        
        Args:
            bundle_path: Path to the bundle zip file
            new_name: Name for your forked version
            customize: Optional customizations dict
        
        Returns:
            Dict with fork results
        """
        from datetime import datetime
        import zipfile
        import hashlib
        
        if not Path(bundle_path).exists():
            return {
                "success": False,
                "error": f"Bundle not found: {bundle_path}",
            }
        
        # Generate fork ID
        fork_id = hashlib.sha256(
            f"{new_name}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        result = {
            "success": True,
            "fork_id": fork_id,
            "name": new_name,
            "source_bundle": bundle_path,
            "customizations": customize or {},
        }
        
        # Extract and read bundle metadata
        try:
            with zipfile.ZipFile(bundle_path, 'r') as zf:
                if "metadata.json" in zf.namelist():
                    metadata = json.loads(zf.read("metadata.json").decode('utf-8'))
                    result["source_name"] = metadata.get("name", "Unknown")
                    result["source_author"] = metadata.get("author", "Unknown")
        except Exception as e:
            result["warning"] = f"Could not read bundle metadata: {e}"
        
        # Create lineage record for the fork
        lineage = ModelLineage(
            model_id=fork_id,
            parent_name=result.get("source_name", "Community Bundle"),
            inherited_weights=True,
            inherited_config=True,
            inherited_personality=True,
            customizations=[f"forked with: {k}={v}" for k, v in (customize or {}).items()],
            created_at=datetime.now().isoformat(),
            forked_from_bundle=bundle_path,
        )
        
        self._lineage_cache[fork_id] = lineage
        self._save_lineage()
        
        # Output path for fork
        fork_dir = self.storage_path / "forks" / fork_id
        fork_dir.mkdir(parents=True, exist_ok=True)
        result["fork_path"] = str(fork_dir)
        
        result["next_steps"] = [
            f"1. Extract bundle to: {fork_dir}",
            f"2. Modify config.json with your customizations",
            f"3. Optionally retrain with additional data",
            f"4. Your fork ID: {fork_id}",
        ]
        
        logger.info(f"Created fork {fork_id} from bundle {bundle_path}")
        return result
    
    def get_lineage(self, model_id: str) -> Optional[ModelLineage]:
        """Get the lineage/ancestry of a model."""
        return self._lineage_cache.get(model_id)
    
    def get_full_ancestry(self, model_id: str) -> List[ModelLineage]:
        """Get full ancestry chain for a model."""
        ancestry = []
        current_id = model_id
        
        while current_id:
            lineage = self._lineage_cache.get(current_id)
            if lineage:
                ancestry.append(lineage)
                current_id = lineage.parent_id
            else:
                break
        
        return ancestry
    
    def list_descendants(self, model_id: str) -> List[ModelLineage]:
        """List all models descended from a given model."""
        return [
            l for l in self._lineage_cache.values()
            if l.parent_id == model_id
        ]
    
    def get_inheritance_tree(self) -> Dict[str, Any]:
        """Get a tree view of all model inheritance relationships."""
        # Find root models (no parent)
        roots = [l for l in self._lineage_cache.values() if not l.parent_id]
        
        def build_tree(lineage: ModelLineage) -> Dict[str, Any]:
            children = self.list_descendants(lineage.model_id)
            return {
                "model_id": lineage.model_id,
                "parent_name": lineage.parent_name,
                "created_at": lineage.created_at,
                "customizations": lineage.customizations,
                "children": [build_tree(c) for c in children],
            }
        
        return {
            "roots": [build_tree(r) for r in roots],
            "total_models": len(self._lineage_cache),
        }


# Global model inheritance instance
_model_inheritance: Optional[ModelInheritance] = None


def get_model_inheritance() -> ModelInheritance:
    """Get the global ModelInheritance instance."""
    global _model_inheritance
    if _model_inheritance is None:
        _model_inheritance = ModelInheritance()
    return _model_inheritance


# =============================================================================
# API-POWERED TRAINING DATA GENERATION
# =============================================================================

@dataclass
class APIProviderConfig:
    """Configuration for an API provider."""
    name: str
    api_key: str
    model: str = "gpt-4"
    base_url: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7


# Task definitions for training data generation
TRAINING_TASKS = {
    "chat": {
        "name": "Chat & Conversation",
        "description": "General conversation, Q&A, assistance",
        "position": "chat",
        "prompts": [
            "Generate a training example where a user asks for help with {topic} and the AI provides a helpful, friendly response.",
            "Create a conversation where the user discusses {topic} and the AI engages thoughtfully.",
            "Generate an example where the AI explains {topic} in simple terms.",
        ],
        "topics": ["cooking", "travel", "technology", "health", "relationships", "hobbies", "work", "education", "movies", "music", "sports", "news", "science", "history", "art"],
    },
    "code": {
        "name": "Code Generation",
        "description": "Programming, debugging, code explanation",
        "position": "code",
        "prompts": [
            "Generate a training example where a user asks for help writing {language} code for {task}.",
            "Create an example where the AI debugs a {language} code snippet.",
            "Generate an example where the AI explains how {concept} works in {language}.",
        ],
        "topics": ["Python", "JavaScript", "TypeScript", "Rust", "C++", "Java"],
        "tasks": ["sorting algorithm", "file handling", "API call", "database query", "web scraping", "data processing"],
        "concepts": ["recursion", "async/await", "classes", "decorators", "closures", "generators"],
    },
    "vision": {
        "name": "Vision & Image Understanding",
        "description": "Describe images, analyze visual content",
        "position": "vision",
        "prompts": [
            "Generate a training example where the AI describes an image of {subject}.",
            "Create an example where the AI answers questions about a {scene_type} scene.",
            "Generate an example where the AI identifies objects in an image of {subject}.",
        ],
        "subjects": ["a sunset", "a city street", "a forest", "a beach", "a kitchen", "an office", "a dog", "a cat", "food", "a car"],
        "scene_types": ["indoor", "outdoor", "urban", "rural", "natural", "domestic"],
    },
    "avatar": {
        "name": "Avatar Control",
        "description": "Control avatar bones, expressions, animations",
        "position": "avatar",
        "prompts": [
            "Generate a training example where the AI controls avatar bones to show {emotion}.",
            "Create an example where the AI animates the avatar to perform {action}.",
            "Generate an example of avatar bone control for {gesture}.",
        ],
        "emotions": ["happiness", "sadness", "surprise", "anger", "confusion", "excitement"],
        "actions": ["waving", "nodding", "shaking head", "pointing", "shrugging", "clapping"],
        "gestures": ["thumbs up", "thinking pose", "greeting", "farewell", "agreement", "disagreement"],
    },
    "image_gen": {
        "name": "Image Generation",
        "description": "Create images from text descriptions",
        "position": "router",  # Routes to image generation
        "prompts": [
            "Generate a training example where the AI understands a request to create an image of {subject}.",
            "Create an example where the AI clarifies an image generation request for {style} art.",
            "Generate an example of the AI helping refine a prompt for generating {subject}.",
        ],
        "subjects": ["a landscape", "a portrait", "an animal", "a building", "abstract art", "a fantasy scene"],
        "styles": ["realistic", "cartoon", "anime", "oil painting", "watercolor", "digital art"],
    },
    "audio_gen": {
        "name": "Audio & Voice Generation",
        "description": "Text-to-speech, voice synthesis, audio creation",
        "position": "router",
        "prompts": [
            "Generate a training example where the AI processes a TTS request with {emotion} tone.",
            "Create an example where the AI helps configure voice settings for {use_case}.",
            "Generate an example of the AI understanding an audio generation request.",
        ],
        "emotions": ["cheerful", "serious", "calm", "excited", "professional", "friendly"],
        "use_cases": ["narration", "announcement", "conversation", "presentation", "storytelling"],
    },
    "video_gen": {
        "name": "Video Generation",
        "description": "Create videos, animations, visual content",
        "position": "router",
        "prompts": [
            "Generate a training example where the AI understands a video generation request for {type}.",
            "Create an example where the AI helps plan a {duration} video about {topic}.",
        ],
        "types": ["animation", "slideshow", "timelapse", "explainer", "promotional"],
        "durations": ["short", "medium", "long"],
        "topics": ["product demo", "tutorial", "story", "presentation", "artistic"],
    },
    "3d_gen": {
        "name": "3D Model Generation",
        "description": "Create 3D models and scenes",
        "position": "router",
        "prompts": [
            "Generate a training example where the AI processes a request to create a 3D model of {object}.",
            "Create an example where the AI helps design a 3D {scene_type}.",
        ],
        "objects": ["character", "building", "vehicle", "furniture", "creature", "landscape"],
        "scene_types": ["interior", "exterior", "fantasy", "sci-fi", "realistic"],
    },
    "game": {
        "name": "Game AI Control",
        "description": "Control game characters, make decisions, play games",
        "position": "game",
        "prompts": [
            "Generate a training example where the AI decides the best action in a {game_type} game scenario.",
            "Create an example where the AI analyzes a {game_type} game state and plans moves.",
            "Generate an example of the AI providing game strategy for {situation}.",
        ],
        "game_types": ["strategy", "action", "puzzle", "RPG", "simulation", "card"],
        "situations": ["combat", "exploration", "resource management", "dialogue choice", "puzzle solving"],
    },
    "robot": {
        "name": "Robot Control",
        "description": "Control robotic hardware, sensors, actuators",
        "position": "robot",
        "prompts": [
            "Generate a training example where the AI commands a robot to {action}.",
            "Create an example where the AI interprets sensor data from a {sensor_type} sensor.",
            "Generate an example of the AI planning a robot movement sequence for {task}.",
        ],
        "actions": ["move forward", "turn left", "pick up object", "navigate to waypoint", "avoid obstacle"],
        "sensor_types": ["camera", "lidar", "ultrasonic", "touch", "temperature"],
        "tasks": ["object retrieval", "navigation", "inspection", "assembly", "sorting"],
    },
    "math": {
        "name": "Math & Calculations",
        "description": "Mathematical reasoning, calculations, problem solving",
        "position": "math",
        "prompts": [
            "Generate a training example where the AI solves a {difficulty} {math_type} problem.",
            "Create an example where the AI explains the solution to a {math_type} problem step by step.",
            "Generate an example of the AI helping with {math_type} calculations.",
        ],
        "difficulties": ["basic", "intermediate", "advanced"],
        "math_types": ["arithmetic", "algebra", "geometry", "calculus", "statistics", "probability"],
    },
    "router": {
        "name": "Intent Classification",
        "description": "Classify user intent to route to correct handler",
        "position": "router",
        "prompts": [
            "Generate a training example where the AI classifies the intent of: '{sample_request}'",
            "Create an example of intent classification for a {request_type} request.",
        ],
        "request_types": ["image generation", "code help", "general chat", "math question", "file operation", "web search"],
        "sample_requests": [
            "Can you make me a picture of a sunset?",
            "Help me write Python code",
            "What's the weather like?",
            "Calculate 15% of 230",
            "Find files in my documents",
            "Search for news about AI",
        ],
    },
}


class APITrainingProvider:
    """
    Generates high-quality training data using external APIs (GPT-4, Claude, etc.).
    
    This allows users to leverage powerful cloud AI to generate training data
    for their local Trainer AI, which can then create specialized task AIs.
    
    Usage:
        provider = APITrainingProvider()
        provider.configure_openai(api_key="sk-...")
        
        # Generate training data for specific tasks
        data = provider.generate_training_data(
            tasks=["chat", "code", "avatar"],
            examples_per_task=100
        )
        
        # Generate data for ALL tasks
        all_data = provider.generate_training_data(
            tasks="all",
            examples_per_task=50
        )
        
        # Train a local model on the generated data
        provider.train_local_trainer(data, model_path="models/trainer")
    """
    
    def __init__(self, auto_load_keys: bool = True):
        """
        Initialize the API training provider.
        
        Args:
            auto_load_keys: If True, automatically load API keys from secure storage
        """
        self._providers: Dict[str, APIProviderConfig] = {}
        self._active_provider: Optional[str] = None
        self._generated_data: Dict[str, List[str]] = {}
        self._generation_stats: Dict[str, int] = {}
        
        if auto_load_keys:
            self._load_from_secure_storage()
        
        logger.info("APITrainingProvider initialized")
    
    def _load_from_secure_storage(self) -> int:
        """
        Load API keys from secure storage.
        
        Returns:
            Number of providers loaded from secure storage
        """
        loaded = 0
        try:
            from ..utils.api_key_encryption import get_api_key
            
            # Try to load OpenAI
            openai_key = get_api_key("openai")
            if openai_key:
                self._providers["openai"] = APIProviderConfig(
                    name="openai",
                    api_key=openai_key,
                    model="gpt-4",
                    base_url="https://api.openai.com/v1",
                )
                loaded += 1
                logger.info("Loaded OpenAI API key from secure storage")
            
            # Try to load Anthropic
            anthropic_key = get_api_key("anthropic")
            if anthropic_key:
                self._providers["anthropic"] = APIProviderConfig(
                    name="anthropic",
                    api_key=anthropic_key,
                    model="claude-3-opus-20240229",
                    base_url="https://api.anthropic.com/v1",
                )
                loaded += 1
                logger.info("Loaded Anthropic API key from secure storage")
            
            # Set first available as active
            if loaded > 0 and not self._active_provider:
                self._active_provider = list(self._providers.keys())[0]
                
        except ImportError:
            logger.debug("Secure storage not available")
        except Exception as e:
            logger.warning(f"Could not load keys from secure storage: {e}")
        
        return loaded
    
    def _store_key_securely(self, service: str, api_key: str, description: str = "") -> bool:
        """
        Store an API key in secure storage.
        
        Args:
            service: Service name (e.g., "openai", "anthropic")
            api_key: The API key to store
            description: Optional description
            
        Returns:
            True if stored successfully
        """
        try:
            from ..utils.api_key_encryption import store_api_key
            return store_api_key(service, api_key, description)
        except ImportError:
            logger.warning("Secure storage not available - key stored in memory only")
            return False
        except Exception as e:
            logger.warning(f"Could not store key securely: {e}")
            return False
    
    def configure_openai(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
        store_securely: bool = True,
    ) -> None:
        """
        Configure OpenAI as a training data provider.
        
        Args:
            api_key: OpenAI API key (if None, loads from secure storage/env)
            model: Model to use (gpt-4, gpt-4-turbo, gpt-3.5-turbo)
            base_url: Optional custom base URL (for Azure or proxies)
            store_securely: Whether to store the key in encrypted storage
        """
        # Get key from secure storage if not provided
        if api_key is None:
            try:
                from ..utils.api_key_encryption import get_api_key
                api_key = get_api_key("openai")
            except ImportError:
                pass
        
        if not api_key:
            raise ValueError("No OpenAI API key provided and none found in secure storage")
        
        self._providers["openai"] = APIProviderConfig(
            name="openai",
            api_key=api_key,
            model=model,
            base_url=base_url or "https://api.openai.com/v1",
        )
        self._active_provider = "openai"
        
        # Store securely for future use
        if store_securely:
            self._store_key_securely("openai", api_key, f"OpenAI API key for {model}")
        
        logger.info(f"Configured OpenAI provider with model {model}")
    
    def configure_anthropic(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229",
        store_securely: bool = True,
    ) -> None:
        """
        Configure Anthropic Claude as a training data provider.
        
        Args:
            api_key: Anthropic API key (if None, loads from secure storage/env)
            model: Model to use (claude-3-opus, claude-3-sonnet, etc.)
            store_securely: Whether to store the key in encrypted storage
        """
        # Get key from secure storage if not provided
        if api_key is None:
            try:
                from ..utils.api_key_encryption import get_api_key
                api_key = get_api_key("anthropic")
            except ImportError:
                pass
        
        if not api_key:
            raise ValueError("No Anthropic API key provided and none found in secure storage")
        
        self._providers["anthropic"] = APIProviderConfig(
            name="anthropic",
            api_key=api_key,
            model=model,
            base_url="https://api.anthropic.com/v1",
        )
        self._active_provider = "anthropic"
        
        # Store securely for future use
        if store_securely:
            self._store_key_securely("anthropic", api_key, f"Anthropic API key for {model}")
        
        logger.info(f"Configured Anthropic provider with model {model}")
    
    def configure_custom(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        store_securely: bool = True,
    ) -> None:
        """
        Configure a custom OpenAI-compatible API provider.
        
        Args:
            name: Provider name
            api_key: API key
            base_url: Base URL for the API
            model: Model name
            store_securely: Whether to store the key in encrypted storage
        """
        self._providers[name] = APIProviderConfig(
            name=name,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
        self._active_provider = name
        
        # Store securely for future use
        if store_securely:
            self._store_key_securely(name, api_key, f"Custom API key for {name}")
        
        logger.info(f"Configured custom provider '{name}' with model {model}")
    
    def set_active_provider(self, name: str) -> bool:
        """Set the active provider by name."""
        if name in self._providers:
            self._active_provider = name
            return True
        return False
    
    def list_providers(self) -> List[str]:
        """List configured providers."""
        return list(self._providers.keys())
    
    def list_available_tasks(self) -> Dict[str, str]:
        """List all available training tasks."""
        return {k: v["name"] for k, v in TRAINING_TASKS.items()}
    
    def _call_api(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Make an API call to the active provider."""
        if not self._active_provider or self._active_provider not in self._providers:
            logger.error("No active provider configured")
            return None
        
        config = self._providers[self._active_provider]
        
        try:
            import requests
            
            if config.name in ("openai", "custom") or (config.base_url and "openai" in config.base_url.lower()):
                # OpenAI-compatible API
                headers = {
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                }
                
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = requests.post(
                    f"{config.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": config.model,
                        "messages": messages,
                        "max_tokens": config.max_tokens,
                        "temperature": config.temperature,
                    },
                    timeout=60,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    return None
                    
            elif config.name == "anthropic":
                # Anthropic API
                headers = {
                    "x-api-key": config.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                }
                
                response = requests.post(
                    f"{config.base_url}/messages",
                    headers=headers,
                    json={
                        "model": config.model,
                        "max_tokens": config.max_tokens,
                        "system": system_prompt if system_prompt else "You are a helpful AI training data generator.",
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=60,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["content"][0]["text"]
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    return None
            
            else:
                # Try OpenAI-compatible as fallback
                return self._call_api_openai_compat(config, prompt, system_prompt)
                
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None
    
    def _call_api_openai_compat(self, config: APIProviderConfig, prompt: str, system_prompt: str) -> Optional[str]:
        """Fallback OpenAI-compatible API call."""
        import requests
        
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{config.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": config.model,
                    "messages": messages,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                },
                timeout=60,
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Fallback API call failed: {e}")
        
        return None
    
    def _generate_single_example(self, task_key: str, task_config: Dict[str, Any]) -> Optional[str]:
        """Generate a single training example for a task."""
        import random
        
        # Select a random prompt template
        prompt_template = random.choice(task_config["prompts"])
        
        # Fill in template variables
        prompt = prompt_template
        for key, values in task_config.items():
            if key not in ("name", "description", "position", "prompts") and isinstance(values, list):
                placeholder = "{" + key.rstrip("s") + "}"  # {topic}, {language}, etc.
                if placeholder in prompt:
                    prompt = prompt.replace(placeholder, random.choice(values))
                # Also try plural form
                placeholder_plural = "{" + key + "}"
                if placeholder_plural in prompt:
                    prompt = prompt.replace(placeholder_plural, random.choice(values))
        
        # System prompt for generating training data
        system_prompt = f"""You are generating training data for an AI system.
Task: {task_config['name']} - {task_config['description']}

Generate a single training example in this EXACT format:
USER: [user's message/request]
ASSISTANT: [AI's response]

The response should be high quality, helpful, and appropriate for the task.
Only output the USER/ASSISTANT pair, nothing else."""
        
        response = self._call_api(prompt, system_prompt)
        
        if response:
            # Clean up the response
            response = response.strip()
            # Ensure proper format
            if "USER:" in response and "ASSISTANT:" in response:
                return response
            elif response.startswith("User:") or response.startswith("user:"):
                return response.replace("User:", "USER:").replace("user:", "USER:").replace("Assistant:", "ASSISTANT:").replace("assistant:", "ASSISTANT:")
        
        return None
    
    def generate_training_data(
        self,
        tasks: Union[str, List[str]] = "all",
        examples_per_task: int = 50,
        callback: Optional[Callable] = None,
    ) -> Dict[str, List[str]]:
        """
        Generate training data for specified tasks using the API.
        
        Args:
            tasks: "all" for all tasks, or list of task keys (e.g., ["chat", "code", "avatar"])
            examples_per_task: Number of examples to generate per task
            callback: Optional callback(task, current, total) for progress updates
        
        Returns:
            Dict mapping task keys to lists of training examples
        """
        if not self._active_provider:
            raise ValueError("No API provider configured. Call configure_openai() or configure_anthropic() first.")
        
        # Determine which tasks to generate
        if tasks == "all":
            task_list = list(TRAINING_TASKS.keys())
        else:
            task_list = tasks if isinstance(tasks, list) else [tasks]
        
        # Validate tasks
        invalid_tasks = [t for t in task_list if t not in TRAINING_TASKS]
        if invalid_tasks:
            raise ValueError(f"Invalid tasks: {invalid_tasks}. Available: {list(TRAINING_TASKS.keys())}")
        
        results: Dict[str, List[str]] = {}
        total_examples = len(task_list) * examples_per_task
        current = 0
        
        for task_key in task_list:
            task_config = TRAINING_TASKS[task_key]
            results[task_key] = []
            
            logger.info(f"Generating {examples_per_task} examples for task: {task_key}")
            
            for i in range(examples_per_task):
                example = self._generate_single_example(task_key, task_config)
                if example:
                    results[task_key].append(example)
                
                current += 1
                if callback:
                    callback(task_key, current, total_examples)
            
            self._generation_stats[task_key] = len(results[task_key])
            logger.info(f"Generated {len(results[task_key])} examples for {task_key}")
        
        self._generated_data = results
        return results
    
    def generate_combined_training_file(
        self,
        tasks: Union[str, List[str]] = "all",
        examples_per_task: int = 50,
        output_path: Optional[str] = None,
        callback: Optional[Callable] = None,
    ) -> str:
        """
        Generate training data and save to a single combined file.
        
        Args:
            tasks: "all" or list of task keys
            examples_per_task: Examples per task
            output_path: Where to save (defaults to data/api_training_data.txt)
            callback: Progress callback
        
        Returns:
            Path to the generated file
        """
        data = self.generate_training_data(tasks, examples_per_task, callback)
        
        # Combine all examples
        all_examples = []
        for task_key, examples in data.items():
            all_examples.append(f"# ===== {TRAINING_TASKS[task_key]['name']} ({task_key}) =====")
            all_examples.extend(examples)
            all_examples.append("")  # Blank line between sections
        
        # Determine output path
        if not output_path:
            output_path = str(Path(CONFIG.get("data_dir", "data")) / "api_training_data.txt")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_examples))
        
        logger.info(f"Saved combined training data to {output_path}")
        return output_path
    
    def generate_separate_training_files(
        self,
        tasks: Union[str, List[str]] = "all",
        examples_per_task: int = 50,
        output_dir: Optional[str] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, str]:
        """
        Generate training data with separate files for each task.
        
        Args:
            tasks: "all" or list of task keys
            examples_per_task: Examples per task
            output_dir: Directory for output files
            callback: Progress callback
        
        Returns:
            Dict mapping task keys to file paths
        """
        data = self.generate_training_data(tasks, examples_per_task, callback)
        
        if not output_dir:
            output_dir = str(Path(CONFIG.get("data_dir", "data")) / "api_training")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        file_paths = {}
        for task_key, examples in data.items():
            file_path = Path(output_dir) / f"{task_key}_training.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(examples))
            file_paths[task_key] = str(file_path)
            logger.info(f"Saved {task_key} training data to {file_path}")
        
        return file_paths
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about generated data."""
        total = sum(self._generation_stats.values())
        return {
            "total_examples": total,
            "by_task": self._generation_stats.copy(),
            "tasks_generated": list(self._generation_stats.keys()),
            "provider": self._active_provider,
        }
    
    def create_trainer_ai_config(
        self,
        tasks: List[str],
        model_size: str = "small",
        name: str = "API-Trained Trainer",
    ) -> Dict[str, Any]:
        """
        Create a configuration for training a local Trainer AI.
        
        Args:
            tasks: Tasks this trainer should handle
            model_size: Size of the local model
            name: Name for the trainer
        
        Returns:
            Configuration dict for TrainerAI
        """
        positions = []
        for task in tasks:
            if task in TRAINING_TASKS:
                pos = TRAINING_TASKS[task]["position"]
                if pos not in positions:
                    positions.append(pos)
        
        return {
            "name": name,
            "model_size": model_size,
            "positions": positions,
            "tasks": tasks,
            "training_source": "api",
            "provider": self._active_provider,
            "capabilities": [TRAINING_TASKS[t]["name"] for t in tasks if t in TRAINING_TASKS],
        }


# Global API training provider instance
_api_training_provider: Optional[APITrainingProvider] = None


def get_api_training_provider() -> APITrainingProvider:
    """Get the global APITrainingProvider instance."""
    global _api_training_provider
    if _api_training_provider is None:
        _api_training_provider = APITrainingProvider()
    return _api_training_provider


# =============================================================================
# TRAINER AI CLASS
# =============================================================================

class TrainerAI:
    """
    Meta-AI for training data generation and curation.
    
    This is the "AI that trains AIs" - it helps prepare training data
    for any specialized model in the router system.
    """
    
    def __init__(self, model=None, use_ai_generation: bool = True):
        """
        Initialize the Trainer AI.
        
        Args:
            model: Optional Forge model for AI-powered generation
            use_ai_generation: If True, uses AI for generation when available
        """
        self.model = model
        self.use_ai = use_ai_generation and model is not None
        self.positions = POSITION_CONFIGS
        self._generation_cache: Dict[str, List[str]] = {}
        
        logger.info(f"TrainerAI initialized (AI generation: {self.use_ai})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEMPLATE AND FORMAT METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_positions(self) -> List[str]:
        """Get list of all supported router positions."""
        return list(self.positions.keys())
    
    def get_position_info(self, position: str) -> Optional[PositionConfig]:
        """Get configuration for a specific position."""
        return self.positions.get(position)
    
    def get_template(self, position: str, count: int = 3) -> str:
        """
        Get example template for a position's data format.
        
        Args:
            position: Router position name
            count: Number of examples to include
        
        Returns:
            Formatted template with examples
        """
        config = self.positions.get(position)
        if not config:
            return f"Unknown position: {position}. Available: {', '.join(self.positions.keys())}"
        
        template = f"""# Training Data Format for: {config.name.upper()}
# {config.description}
# Recommended model size: {config.recommended_model_size}
# Minimum examples: {config.example_count_min}

# Format:
# {config.input_prefix} <input text>{config.separator}{config.output_prefix} <output text>

# Examples:
"""
        examples = self._generate_examples(position, count)
        template += "\n".join(examples)
        
        return template
    
    def _generate_examples(self, position: str, count: int) -> List[str]:
        """Generate example training data entries."""
        config = self.positions.get(position)
        if not config:
            return []
        
        examples = []
        
        if position == "router":
            templates = SYNTHETIC_TEMPLATES.get("router", [])
            for template, intent in random.sample(templates, min(count, len(templates))):
                filled = self._fill_template(template)
                examples.append(f"{config.input_prefix} {filled}{config.separator}{config.output_prefix} {intent}")
                
        elif position == "vision":
            templates = SYNTHETIC_TEMPLATES.get("vision", [])
            for img_desc, caption_template in random.sample(templates, min(count, len(templates))):
                filled_img = self._fill_template(img_desc)
                filled_caption = self._fill_template(caption_template)
                examples.append(f"{config.input_prefix} {filled_img}{config.separator}{config.output_prefix} {filled_caption}")
                
        elif position == "code":
            py_templates = SYNTHETIC_TEMPLATES.get("code", {}).get("python", [])
            for task, code in random.sample(py_templates, min(count, len(py_templates))):
                examples.append(f"{config.input_prefix} {task}{config.separator}{config.output_prefix}\n{code}")
                
        elif position == "avatar":
            templates = SYNTHETIC_TEMPLATES.get("avatar", [])
            for command, bones in random.sample(templates, min(count, len(templates))):
                examples.append(f"{config.input_prefix} {command}{config.separator}{config.output_prefix} {bones}")
                
        elif position == "chat":
            # Basic chat examples
            chat_examples = [
                ("Hello!", "Hello! How can I help you today?"),
                ("How are you?", "I'm doing well, thank you for asking! How can I assist you?"),
                ("What's the weather like?", "I don't have access to real-time weather data, but I can help you find a weather service!"),
            ]
            for user, assistant in random.sample(chat_examples, min(count, len(chat_examples))):
                examples.append(f"{config.input_prefix} {user}{config.separator}{config.output_prefix} {assistant}")
                
        elif position == "math":
            math_examples = [
                ("What is 2 + 2?", "Step 1: Add 2 and 2\nStep 2: 2 + 2 = 4\nAnswer: 4"),
                ("Solve: 3x = 15", "Step 1: Divide both sides by 3\nStep 2: x = 15/3\nStep 3: x = 5\nAnswer: x = 5"),
            ]
            for problem, solution in random.sample(math_examples, min(count, len(math_examples))):
                examples.append(f"{config.input_prefix} {problem}{config.separator}{config.output_prefix}\n{solution}")
        
        elif position == "teacher":
            templates = SYNTHETIC_TEMPLATES.get("teacher", [])
            for request, guidance in random.sample(templates, min(count, len(templates))):
                examples.append(f"{config.input_prefix} {request}{config.separator}{config.output_prefix} {guidance}")
                
        return examples
    
    def _fill_template(self, template: str) -> str:
        """Fill a template with random words from word banks."""
        result = template
        for key, words in WORD_BANKS.items():
            placeholder = "{" + key + "}"
            while placeholder in result:
                result = result.replace(placeholder, random.choice(words), 1)
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # DATA GENERATION METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def generate_training_data(
        self,
        position: str,
        count: int = 100,
        seed_data: Optional[str] = None,
        use_ai: Optional[bool] = None,
    ) -> str:
        """
        Generate training data for a specific router position.
        
        Args:
            position: Router position (router, vision, code, etc.)
            count: Number of examples to generate
            seed_data: Optional existing data to augment
            use_ai: Override AI generation setting
        
        Returns:
            Generated training data as string
        """
        config = self.positions.get(position)
        if not config:
            raise ValueError(f"Unknown position: {position}. Available: {', '.join(self.positions.keys())}")
        
        should_use_ai = use_ai if use_ai is not None else self.use_ai
        
        if should_use_ai and self.model:
            return self._generate_with_ai(position, count, seed_data)
        else:
            return self._generate_synthetic(position, count, seed_data)
    
    def _generate_synthetic(self, position: str, count: int, seed_data: Optional[str]) -> str:
        """Generate synthetic training data using templates."""
        config = self.positions[position]
        examples = []
        
        # Parse seed data for patterns if provided
        patterns = []
        if seed_data:
            patterns = self._extract_patterns(seed_data, position)
        
        generated = 0
        max_attempts = count * 3  # Prevent infinite loops
        attempts = 0
        
        while generated < count and attempts < max_attempts:
            attempts += 1
            
            # Try to generate from templates
            example = self._generate_single_example(position, patterns)
            if example and example not in examples:
                examples.append(example)
                generated += 1
        
        # Add header
        header = f"""# Generated Training Data for: {config.name.upper()}
# Generated by TrainerAI
# Count: {len(examples)} examples
# Format: {config.input_prefix} <input>{config.separator}{config.output_prefix} <output>

"""
        return header + "\n".join(examples)
    
    def _generate_single_example(self, position: str, patterns: List) -> Optional[str]:
        """Generate a single training example."""
        config = self.positions[position]
        
        if position == "router":
            templates = SYNTHETIC_TEMPLATES.get("router", [])
            if templates:
                template, intent = random.choice(templates)
                filled = self._fill_template(template)
                return f"{config.input_prefix} {filled}{config.separator}{config.output_prefix} {intent}"
                
        elif position == "avatar":
            templates = SYNTHETIC_TEMPLATES.get("avatar", [])
            if templates:
                command, bones = random.choice(templates)
                # Add variation
                variations = ["please", "", "can you", ""]
                prefix = random.choice(variations)
                full_command = f"{prefix} {command}".strip()
                return f"{config.input_prefix} {full_command}{config.separator}{config.output_prefix} {bones}"
        
        elif position == "teacher":
            templates = SYNTHETIC_TEMPLATES.get("teacher", [])
            if templates:
                request, guidance = random.choice(templates)
                # Add variation to requests
                variations = ["I need to", "Help me", "How do I", "Can you help me"]
                prefix = random.choice(variations)
                full_request = f"{prefix} {request.lower()}" if random.random() > 0.5 else request
                return f"{config.input_prefix} {full_request}{config.separator}{config.output_prefix} {guidance}"
        
        # Fall back to basic examples
        return self._generate_examples(position, 1)[0] if self._generate_examples(position, 1) else None
    
    def _generate_with_ai(self, position: str, count: int, seed_data: Optional[str]) -> str:
        """Generate training data using the AI model."""
        config = self.positions[position]
        
        prompt = f"""Generate {count} training examples for a {config.description}.

Format each example as:
{config.input_prefix} <input text>{config.separator}{config.output_prefix} <output text>

Examples should be diverse and high quality.
"""
        if seed_data:
            prompt += f"\nHere are some existing examples to learn from:\n{seed_data[:1000]}\n"
        
        prompt += f"\nGenerate {count} new examples:\n"
        
        try:
            # Use the model to generate
            if self.model is None:
                return self._generate_synthetic(position, count, seed_data)
            response = self.model.generate(prompt, max_gen=count * 100)
            return response
        except Exception as e:
            logger.warning(f"AI generation failed, falling back to synthetic: {e}")
            return self._generate_synthetic(position, count, seed_data)
    
    def _extract_patterns(self, data: str, position: str) -> List[Tuple[str, str]]:
        """Extract input/output patterns from existing data."""
        config = self.positions[position]
        patterns = []
        
        lines = data.split('\n')
        for line in lines:
            if config.input_prefix in line and config.output_prefix in line:
                try:
                    parts = line.split(config.separator)
                    if len(parts) >= 2:
                        input_part = parts[0].replace(config.input_prefix, '').strip()
                        output_part = parts[1].replace(config.output_prefix, '').strip()
                        patterns.append((input_part, output_part))
                except Exception:
                    pass
        
        return patterns
    
    # ─────────────────────────────────────────────────────────────────────────
    # DATA CURATION METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def curate_data(
        self,
        position: str,
        raw_data: str,
        remove_duplicates: bool = True,
        validate_format: bool = True,
        filter_low_quality: bool = True,
    ) -> Tuple[str, DataQualityScore]:
        """
        Curate and clean training data.
        
        Args:
            position: Router position
            raw_data: Raw training data to curate
            remove_duplicates: Remove duplicate entries
            validate_format: Check format validity
            filter_low_quality: Remove low-quality entries
        
        Returns:
            Tuple of (cleaned data, quality score)
        """
        config = self.positions.get(position)
        if not config:
            raise ValueError(f"Unknown position: {position}")
        
        lines = [l.strip() for l in raw_data.split('\n') if l.strip() and not l.startswith('#')]
        original_count = len(lines)
        issues = []
        suggestions = []
        
        # Remove duplicates
        if remove_duplicates:
            unique_lines = list(dict.fromkeys(lines))
            duplicates_removed = len(lines) - len(unique_lines)
            if duplicates_removed > 0:
                issues.append(f"Removed {duplicates_removed} duplicate entries")
            lines = unique_lines
        
        # Validate format
        valid_lines = []
        invalid_count = 0
        
        for line in lines:
            is_valid, error = self._validate_line(line, config)
            if is_valid:
                valid_lines.append(line)
            else:
                invalid_count += 1
                if invalid_count <= 5:  # Only report first 5
                    issues.append(f"Invalid: {line[:50]}... ({error})")
        
        if invalid_count > 0:
            suggestions.append(f"Fix {invalid_count} invalid entries using format: {config.input_prefix} <input>{config.separator}{config.output_prefix} <output>")
        
        lines = valid_lines
        
        # Filter low quality
        if filter_low_quality:
            high_quality = []
            low_quality_count = 0
            
            for line in lines:
                quality = self._assess_line_quality(line, config)
                if quality >= 0.5:
                    high_quality.append(line)
                else:
                    low_quality_count += 1
            
            if low_quality_count > 0:
                issues.append(f"Filtered {low_quality_count} low-quality entries")
            
            lines = high_quality
        
        # Calculate scores
        format_score = len(lines) / original_count if original_count > 0 else 0
        diversity_score = self._calculate_diversity(lines, config)
        completeness_score = min(1.0, len(lines) / config.example_count_min)
        overall_score = (format_score + diversity_score + completeness_score) / 3
        
        # Generate suggestions
        if len(lines) < config.example_count_min:
            suggestions.append(f"Add {config.example_count_min - len(lines)} more examples (minimum: {config.example_count_min})")
        
        if diversity_score < 0.7:
            suggestions.append("Increase variety in your examples - many entries are similar")
        
        # Rebuild data
        header = f"""# Curated Training Data for: {config.name.upper()}
# Original: {original_count} entries | Cleaned: {len(lines)} entries
# Quality Score: {overall_score:.2f}

"""
        cleaned_data = header + "\n".join(lines)
        
        score = DataQualityScore(
            overall_score=overall_score,
            format_score=format_score,
            diversity_score=diversity_score,
            completeness_score=completeness_score,
            issues=issues,
            suggestions=suggestions,
        )
        
        return cleaned_data, score
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEACHER AI METHODS (META-LEARNING)
    # ─────────────────────────────────────────────────────────────────────────
    
    def teacher_evaluate(
        self,
        position: str,
        training_data: str,
        detailed: bool = True,
    ) -> Dict[str, Any]:
        """
        Teacher AI evaluation of training data quality.
        
        This is the meta-learning component - the Teacher evaluates training data
        and provides guidance on how to improve it, essentially "teaching" about
        what makes good training data.
        
        Args:
            position: Router position the data is for
            training_data: The training data to evaluate
            detailed: Whether to include detailed analysis
        
        Returns:
            Dict with evaluation, score, and improvements
        """
        config = self.positions.get(position)
        if not config:
            raise ValueError(f"Unknown position: {position}")
        
        # Parse the data
        lines = [l.strip() for l in training_data.split('\n') 
                 if l.strip() and not l.startswith('#')]
        
        # Run basic curation to get scores
        _, quality_score = self.curate_data(position, training_data)
        
        # Advanced evaluation
        evaluation = {
            "position": position,
            "total_examples": len(lines),
            "quality_score": quality_score.overall_score,
            "format_score": quality_score.format_score,
            "diversity_score": quality_score.diversity_score,
            "completeness_score": quality_score.completeness_score,
            "meets_minimum": len(lines) >= config.example_count_min,
            "issues": quality_score.issues,
            "improvements": [],
            "teaching_guidance": "",
        }
        
        if detailed:
            # Analyze patterns
            evaluation["pattern_analysis"] = self._analyze_patterns(lines, config)
            
            # Generate teaching guidance
            evaluation["teaching_guidance"] = self._generate_teaching_guidance(
                position, 
                quality_score, 
                len(lines),
                config
            )
            
            # Specific improvements
            evaluation["improvements"] = self._generate_improvements(
                position, 
                quality_score,
                evaluation["pattern_analysis"],
                config
            )
        
        return evaluation
    
    def _analyze_patterns(self, lines: List[str], config: PositionConfig) -> Dict[str, Any]:
        """Analyze patterns in training data for meta-learning."""
        analysis = {
            "avg_length": 0,
            "length_variance": 0,
            "repeated_words": [],
            "format_consistency": 0.0,
            "input_diversity": 0.0,
            "output_diversity": 0.0,
        }
        
        if not lines:
            return analysis
        
        # Length analysis
        lengths = [len(line) for line in lines]
        analysis["avg_length"] = sum(lengths) / len(lengths)
        if len(lengths) > 1:
            mean = analysis["avg_length"]
            analysis["length_variance"] = sum((l - mean) ** 2 for l in lengths) / len(lengths)
        
        # Word frequency
        all_words = []
        for line in lines:
            all_words.extend(line.lower().split())
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(10)
        analysis["repeated_words"] = [w for w, c in common_words if c > len(lines) * 0.3]
        
        # Format consistency
        has_input = sum(1 for l in lines if config.input_prefix in l)
        has_output = sum(1 for l in lines if config.output_prefix in l)
        analysis["format_consistency"] = min(has_input, has_output) / len(lines) if lines else 0
        
        # Input/output diversity
        inputs = []
        outputs = []
        for line in lines:
            if config.separator in line:
                parts = line.split(config.separator)
                if len(parts) >= 2:
                    inputs.append(parts[0].strip())
                    outputs.append(parts[1].strip())
        
        if inputs:
            unique_input_ratio = len(set(inputs)) / len(inputs)
            analysis["input_diversity"] = unique_input_ratio
        
        if outputs:
            unique_output_ratio = len(set(outputs)) / len(outputs)
            analysis["output_diversity"] = unique_output_ratio
        
        return analysis
    
    def _generate_teaching_guidance(
        self,
        position: str,
        quality: DataQualityScore,
        count: int,
        config: PositionConfig,
    ) -> str:
        """Generate meta-learning guidance for improving training data."""
        guidance_parts = []
        
        # Overall assessment
        if quality.overall_score >= 0.8:
            guidance_parts.append(f"Excellent {position} training data! Quality is high.")
        elif quality.overall_score >= 0.6:
            guidance_parts.append(f"Good {position} training data with room for improvement.")
        else:
            guidance_parts.append(f"The {position} training data needs significant improvement.")
        
        # Format guidance
        if quality.format_score < 0.9:
            guidance_parts.append(
                f"Format Guidance: Each line should follow '{config.input_prefix} <input>"
                f"{config.separator}{config.output_prefix} <output>'. "
                f"Current format score: {quality.format_score:.0%}"
            )
        
        # Diversity guidance
        if quality.diversity_score < 0.7:
            guidance_parts.append(
                "Diversity Guidance: Add more variety to your examples. "
                "Try different phrasings, edge cases, and scenarios. "
                f"Current diversity: {quality.diversity_score:.0%}"
            )
        
        # Completeness guidance
        if count < config.example_count_min:
            needed = config.example_count_min - count
            guidance_parts.append(
                f"Completeness Guidance: Need {needed} more examples. "
                f"Minimum recommended: {config.example_count_min}"
            )
        
        # Position-specific guidance
        position_guidance = {
            "router": "For router training: Include ambiguous queries, multi-intent requests, and edge cases.",
            "vision": "For vision training: Add sensory details, spatial relationships, and emotional context.",
            "code": "For code training: Include error handling, edge cases, and performance considerations.",
            "avatar": "For avatar training: Add compound movements, emotion expressions, and body language.",
            "chat": "For chat training: Include context-aware responses and personality consistency.",
            "math": "For math training: Show step-by-step reasoning and handle edge cases.",
            "trainer": "For trainer training: Include meta-examples showing how to generate good data.",
            "teacher": "For teacher training: Include evaluation criteria and improvement strategies.",
        }
        if position in position_guidance:
            guidance_parts.append(position_guidance[position])
        
        return "\n".join(guidance_parts)
    
    def _generate_improvements(
        self,
        position: str,
        quality: DataQualityScore,
        pattern_analysis: Dict[str, Any],
        config: PositionConfig,
    ) -> List[str]:
        """Generate specific improvement suggestions."""
        improvements = []
        
        # From quality score
        improvements.extend(quality.suggestions)
        
        # From pattern analysis
        if pattern_analysis["repeated_words"]:
            improvements.append(
                f"Reduce repetition of words: {', '.join(pattern_analysis['repeated_words'][:5])}"
            )
        
        if pattern_analysis["input_diversity"] < 0.8:
            improvements.append(
                "Increase input variety - many inputs are similar. "
                "Try rephrasing or adding new scenarios."
            )
        
        if pattern_analysis["output_diversity"] < 0.8:
            improvements.append(
                "Increase output variety - consider different response styles."
            )
        
        if pattern_analysis["length_variance"] < 100:
            improvements.append(
                "Add examples of varying lengths for robustness."
            )
        
        return improvements
    
    def teacher_improve_data(
        self,
        position: str,
        training_data: str,
        target_count: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Teacher AI improves existing training data.
        
        Uses meta-learning to:
        1. Fix format issues
        2. Add missing examples
        3. Increase diversity
        4. Remove low-quality entries
        
        Args:
            position: Router position
            training_data: Existing training data
            target_count: Target number of examples (optional)
        
        Returns:
            Tuple of (improved_data, improvement_report)
        """
        config = self.positions.get(position)
        if not config:
            raise ValueError(f"Unknown position: {position}")
        
        # First evaluate
        evaluation = self.teacher_evaluate(position, training_data)
        
        # Curate existing data
        curated_data, _ = self.curate_data(position, training_data)
        
        # Parse curated
        lines = [l.strip() for l in curated_data.split('\n') 
                 if l.strip() and not l.startswith('#')]
        
        # Determine how many more we need
        target = target_count or config.example_count_min
        needed = max(0, target - len(lines))
        
        # Generate additional examples if needed
        additions = []
        if needed > 0:
            new_data = self.generate_training_data(
                position, 
                count=needed, 
                seed_data=curated_data
            )
            # Parse new data
            new_lines = [l.strip() for l in new_data.split('\n') 
                        if l.strip() and not l.startswith('#')]
            additions = new_lines
        
        # Combine
        all_lines = lines + additions
        
        # De-duplicate
        unique_lines = list(dict.fromkeys(all_lines))
        
        # Build improved data
        header = f"""# Improved Training Data for: {config.name.upper()}
# Improved by Teacher AI (meta-learning system)
# Original: {len(lines)} examples | Added: {len(additions)} | Final: {len(unique_lines)}
# Quality Score: {evaluation['quality_score']:.2f} -> improved

"""
        improved_data = header + "\n".join(unique_lines)
        
        # Build report
        report = {
            "original_count": len(lines),
            "added_count": len(additions),
            "final_count": len(unique_lines),
            "duplicates_removed": len(all_lines) - len(unique_lines),
            "original_quality": evaluation["quality_score"],
            "improvements_applied": [
                f"Curated and cleaned {len(lines)} existing examples",
                f"Generated {len(additions)} new examples" if additions else "No new examples needed",
                f"Removed {len(all_lines) - len(unique_lines)} duplicates" if len(all_lines) > len(unique_lines) else "No duplicates found",
            ],
            "teaching_guidance": evaluation.get("teaching_guidance", ""),
        }
        
        return improved_data, report
    
    def teacher_self_improve(
        self,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Self-improvement loop for the Teacher AI.
        
        The Teacher AI can learn from:
        1. Feedback from specialized models it helped train
        2. Analysis of what training data worked well
        3. New patterns in successful training runs
        
        Args:
            feedback: Optional feedback from a training run or model evaluation
        
        Returns:
            Dict with improvement status and updated capabilities
        """
        result = {
            "status": "analyzed",
            "improvements_identified": [],
            "capabilities_updated": [],
            "recommendations": [],
        }
        
        if feedback:
            # Analyze feedback
            if "model_performance" in feedback:
                perf = feedback["model_performance"]
                if perf < 0.5:
                    result["improvements_identified"].append(
                        "Model performance low - training data may need more examples"
                    )
                    result["recommendations"].append(
                        "Increase training data count and diversity"
                    )
            
            if "loss_curve" in feedback:
                # Check for overfitting
                losses = feedback["loss_curve"]
                if len(losses) > 10:
                    early = sum(losses[:5]) / 5
                    late = sum(losses[-5:]) / 5
                    if late > early * 1.1:
                        result["improvements_identified"].append(
                            "Possible overfitting detected"
                        )
                        result["recommendations"].append(
                            "Add more diverse training examples or reduce epochs"
                        )
            
            if "position" in feedback and "success_rate" in feedback:
                pos = feedback["position"]
                rate = feedback["success_rate"]
                if rate < 0.7:
                    result["improvements_identified"].append(
                        f"Low success rate for {pos} position ({rate:.0%})"
                    )
                    result["recommendations"].append(
                        f"Review {pos} training data for edge cases and errors"
                    )
        
        # Self-assessment of capabilities
        result["capabilities_updated"] = list(self.positions.keys())
        result["positions_supported"] = len(self.positions)
        
        return result

    def _validate_line(self, line: str, config: PositionConfig) -> Tuple[bool, str]:
        """Validate a single training line."""
        # Check basic format
        if config.input_prefix not in line:
            return False, f"Missing {config.input_prefix}"
        
        if config.output_prefix not in line:
            return False, f"Missing {config.output_prefix}"
        
        # Position-specific validation
        if "router" in config.name:
            # Check intent is known
            for intent in KNOWN_INTENTS:
                if intent in line.lower():
                    return True, ""
            return False, "Unknown intent"
        
        if "avatar" in config.name:
            # Check bones is valid JSON
            if config.output_prefix in line:
                bones_part = line.split(config.output_prefix)[-1].strip()
                try:
                    json.loads(bones_part)
                except json.JSONDecodeError:
                    return False, "Invalid JSON in bones"
        
        return True, ""
    
    def _assess_line_quality(self, line: str, config: PositionConfig) -> float:
        """Assess quality of a single training line (0.0 - 1.0)."""
        score = 0.5  # Base score
        
        # Length checks
        if len(line) > 20:
            score += 0.1
        if len(line) > 50:
            score += 0.1
        
        # Has both parts
        if config.input_prefix in line and config.output_prefix in line:
            score += 0.2
        
        # Not too repetitive
        words = line.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 0.1
        
        return min(1.0, score)
    
    def _calculate_diversity(self, lines: List[str], config: PositionConfig) -> float:
        """Calculate diversity score of training data."""
        if not lines:
            return 0.0
        
        # Calculate word distribution
        all_words = []
        for line in lines:
            all_words.extend(line.lower().split())
        
        if not all_words:
            return 0.0
        
        word_counts = Counter(all_words)
        unique_ratio = len(word_counts) / len(all_words)
        
        # Check output variety (for router, check intent distribution)
        if config.name == "router":
            intents = []
            for line in lines:
                for intent in KNOWN_INTENTS:
                    if config.output_prefix in line and intent in line.split(config.output_prefix)[-1].lower():
                        intents.append(intent)
                        break
            
            if intents:
                intent_counts = Counter(intents)
                intent_variety = len(intent_counts) / len(KNOWN_INTENTS)
                return (unique_ratio + intent_variety) / 2
        
        return unique_ratio
    
    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATION METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def evaluate_model_output(
        self,
        position: str,
        input_text: str,
        output_text: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a model's output for quality.
        
        Args:
            position: Router position the model is for
            input_text: The input given to the model
            output_text: The model's output
        
        Returns:
            Evaluation dict with scores and feedback
        """
        config = self.positions.get(position)
        if not config:
            return {"error": f"Unknown position: {position}"}
        
        evaluation = {
            "position": position,
            "input": input_text[:100],
            "output": output_text[:200],
            "scores": {},
            "feedback": [],
            "overall": 0.0,
        }
        
        # Length score
        min_length = {"router": 3, "vision": 20, "code": 10, "chat": 5, "avatar": 10}.get(position, 10)
        length_score = min(1.0, len(output_text) / min_length) if min_length else 1.0
        evaluation["scores"]["length"] = length_score
        
        # Relevance score (basic keyword matching)
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())
        common = input_words & output_words
        relevance_score = len(common) / max(len(input_words), 1) if input_words else 0.5
        evaluation["scores"]["relevance"] = min(1.0, relevance_score + 0.3)  # Boost since exact match not required
        
        # Format score
        format_score = 1.0
        if position == "router":
            # Check if output is a known intent
            format_score = 1.0 if any(intent in output_text.lower() for intent in KNOWN_INTENTS) else 0.0
        elif position == "avatar":
            # Check if output is valid JSON
            try:
                json.loads(output_text)
                format_score = 1.0
            except json.JSONDecodeError:
                format_score = 0.0
                evaluation["feedback"].append("Output should be valid JSON for avatar control")
        elif position == "code":
            # Basic code check
            code_indicators = ["def ", "class ", "import ", "return ", "if ", "for ", "while ", "="]
            format_score = 1.0 if any(ind in output_text for ind in code_indicators) else 0.5
        
        evaluation["scores"]["format"] = format_score
        
        # Calculate overall
        weights = {"length": 0.2, "relevance": 0.4, "format": 0.4}
        overall = sum(evaluation["scores"].get(k, 0) * w for k, w in weights.items())
        evaluation["overall"] = overall
        
        # Generate feedback
        if overall < 0.5:
            evaluation["feedback"].append("Output quality is low - consider retraining with more examples")
        elif overall < 0.7:
            evaluation["feedback"].append("Output is acceptable but could be improved")
        else:
            evaluation["feedback"].append("Output quality is good")
        
        return evaluation
    
    # ─────────────────────────────────────────────────────────────────────────
    # FILE OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def save_training_data(self, position: str, data: str, filename: Optional[str] = None) -> Path:
        """
        Save generated training data to file.
        
        Args:
            position: Router position
            data: Training data string
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        data_dir = Path(CONFIG.get("data_dir", "data")) / "specialized"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"{position}_training.txt"
        
        filepath = data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data)
        
        logger.info(f"Saved training data to: {filepath}")
        return filepath
    
    def load_training_data(self, position: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Load existing training data from file.
        
        Args:
            position: Router position
            filename: Optional custom filename
        
        Returns:
            Training data string or None if not found
        """
        data_dir = Path(CONFIG.get("data_dir", "data")) / "specialized"
        
        if filename is None:
            filename = f"{position}_training.txt"
        
        filepath = data_dir / filename
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    # ─────────────────────────────────────────────────────────────────────────
    # QUICK-CREATE PRESET METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_presets(self) -> Dict[str, QuickCreatePreset]:
        """Get all available Quick-Create presets."""
        return QUICK_CREATE_PRESETS
    
    def get_preset(self, preset_name: str) -> Optional[QuickCreatePreset]:
        """Get a specific preset by name."""
        return QUICK_CREATE_PRESETS.get(preset_name)
    
    def generate_from_preset(
        self,
        preset_name: str,
        count_multiplier: float = 1.0,
        save: bool = True,
    ) -> Dict[str, str]:
        """
        Generate training data for all positions in a preset.
        
        Args:
            preset_name: Name of the preset to use
            count_multiplier: Multiply recommended count by this (e.g., 0.5 for faster, 2.0 for more data)
            save: If True, save generated data to files
        
        Returns:
            Dict mapping position name to generated data
        """
        preset = self.get_preset(preset_name)
        if not preset:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(QUICK_CREATE_PRESETS.keys())}")
        
        results = {}
        count = int(preset.recommended_data_count * count_multiplier)
        
        logger.info(f"Generating data for preset '{preset.name}' ({len(preset.positions)} positions, {count} examples each)")
        
        for position in preset.positions:
            data = self.generate_training_data(position, count=count)
            results[position] = data
            
            if save:
                self.save_training_data(position, data)
        
        return results
    
    def get_preset_summary(self, preset_name: str) -> str:
        """Get a human-readable summary of a preset."""
        preset = self.get_preset(preset_name)
        if not preset:
            return f"Unknown preset: {preset_name}"
        
        lines = [
            f"Preset: {preset.name}",
            f"Description: {preset.description}",
            f"",
            f"Positions to train ({len(preset.positions)}):",
        ]
        
        for pos in preset.positions:
            config = self.positions.get(pos)
            if config:
                lines.append(f"  - {pos}: {config.description}")
            else:
                lines.append(f"  - {pos}")
        
        lines.extend([
            f"",
            f"Recommended model size: {preset.model_size}",
            f"Training examples per position: {preset.recommended_data_count}",
            f"Tools enabled: {', '.join(preset.tools_enabled) or 'None'}",
        ])
        
        return "\n".join(lines)
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHARACTER-TO-AI PIPELINE
    # ─────────────────────────────────────────────────────────────────────────
    
    def generate_character_training_data(
        self,
        character_profile: "CharacterProfile",
        count: int = 500,
        conversation_starters: Optional[List[str]] = None,
        include_topics: bool = True,
    ) -> str:
        """
        Generate chat training data from a character profile.
        
        Converts CharacterTrainer output into training data format
        for the 'chat' position, preserving speech patterns and personality.
        
        Args:
            character_profile: CharacterProfile from CharacterTrainer
            count: Number of training examples to generate
            conversation_starters: Custom prompts to use (auto-generated if None)
            include_topics: Include character's topics in conversations
        
        Returns:
            Training data string in chat format (USER: ... ASSISTANT: ...)
        """
        config = self.positions["chat"]
        examples = []
        
        # Use existing dialogues as base responses
        dialogues = list(character_profile.sample_dialogues)
        
        # Default conversation starters
        default_starters = [
            "Tell me about yourself.",
            "What do you think about that?",
            "How would you respond to this?",
            "What's your opinion?",
            "Can you explain that?",
            "What would you say?",
            "Share your thoughts.",
            "How do you feel about this?",
            "What's on your mind?",
            "Describe your perspective.",
        ]
        
        # Add topic-based starters if enabled
        topic_starters = []
        if include_topics and character_profile.topics:
            for topic in character_profile.topics[:10]:
                topic_starters.extend([
                    f"What do you know about {topic}?",
                    f"Tell me about {topic}.",
                    f"What's your view on {topic}?",
                ])
        
        # Combine starters
        starters = conversation_starters or (default_starters + topic_starters)
        
        # Apply speech patterns to dialogues
        styled_dialogues = self._apply_character_style(
            dialogues,
            character_profile
        )
        
        # Generate training examples
        generated = 0
        while generated < count:
            # Pick a starter and a response
            starter = random.choice(starters)
            
            if styled_dialogues:
                response = random.choice(styled_dialogues)
            else:
                # Fallback: generate response using catchphrases
                response = self._generate_character_response(character_profile)
            
            # Create training line
            example = f"{config.input_prefix} {starter}\n{config.output_prefix} {response}"
            
            # Avoid exact duplicates
            if example not in examples:
                examples.append(example)
                generated += 1
            
            # Prevent infinite loop if not enough variety
            if generated < count and len(examples) >= len(styled_dialogues) * len(starters):
                break
        
        # Build header
        header = f"""# Character Training Data: {character_profile.name}
# Generated by TrainerAI Character-to-AI Pipeline
# Personality traits: {', '.join(trait for trait, score in character_profile.personality_traits.items() if score > 0.3)}
# Speech patterns captured from {character_profile.dialogue_count} dialogues
# Examples: {len(examples)}

"""
        return header + "\n\n".join(examples)
    
    def _apply_character_style(
        self,
        dialogues: List[str],
        profile: "CharacterProfile",
    ) -> List[str]:
        """
        Apply character's speech patterns to dialogues.
        
        Preserves the original meaning while adding character flair:
        - Catchphrases may be prepended/appended
        - Vocabulary preferences applied
        - Speech pattern markers added
        """
        styled = []
        
        for dialogue in dialogues:
            result = dialogue
            
            # Occasionally add catchphrases (20% chance)
            if profile.catchphrases and random.random() < 0.2:
                catchphrase = random.choice(profile.catchphrases)
                if random.random() < 0.5:
                    result = f"{catchphrase} {result}"
                else:
                    result = f"{result} {catchphrase}"
            
            styled.append(result)
        
        return styled
    
    def _generate_character_response(self, profile: "CharacterProfile") -> str:
        """Generate a response using character's vocabulary and patterns."""
        # Start with a catchphrase or phrase
        parts = []
        
        if profile.catchphrases:
            parts.append(random.choice(profile.catchphrases))
        elif profile.phrases:
            parts.append(random.choice(profile.phrases[:20]))
        else:
            # Fallback response
            parts.append(f"As {profile.name}, I would say...")
        
        # Add topic reference if available
        if profile.topics and random.random() < 0.5:
            topic = random.choice(profile.topics[:5])
            parts.append(f"Regarding {topic},")
        
        return " ".join(parts)
    
    def character_to_ai(
        self,
        character_name: str,
        data_path: str,
        output_name: Optional[str] = None,
        training_count: int = 500,
        personality_mode: str = "hybrid",
        create_bundle: bool = True,
        aliases: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        One-button Character-to-AI conversion.
        
        Extracts a character from training data and creates everything needed
        to use them as an AI:
        - Extracts character profile using CharacterTrainer
        - Generates training data in chat format
        - Creates persona configuration
        - Optionally creates an AI bundle
        
        Args:
            character_name: Name of character to extract
            data_path: Path to source training data
            output_name: Name for the AI (defaults to character name)
            training_count: Number of training examples to generate
            personality_mode: How to apply personality
                - "baked": Personality woven into training data
                - "system_prompt": Personality in system prompt only
                - "hybrid": Both (recommended)
            create_bundle: If True, create an .enigma-bundle
            aliases: Alternative names for the character
        
        Returns:
            Dict with:
                - success: bool
                - character_profile: CharacterProfile (as dict)
                - training_data_path: Path to generated training data
                - system_prompt: Generated system prompt
                - bundle_path: Path to bundle (if created)
                - error: Error message (if failed)
        """
        from ..tools.data_trainer import CharacterTrainer
        
        result = {
            "success": False,
            "character_profile": None,
            "training_data_path": None,
            "system_prompt": None,
            "bundle_path": None,
            "error": None,
        }
        
        output_name = output_name or character_name
        
        try:
            # Step 1: Extract character using CharacterTrainer
            logger.info(f"Extracting character '{character_name}' from {data_path}...")
            trainer = CharacterTrainer()
            extraction = trainer.extract_character(
                character_name=character_name,
                data_path=data_path,
                aliases=aliases,
            )
            
            if not extraction.success:
                result["error"] = extraction.error or "Character extraction failed"
                return result
            
            profile = extraction.character
            if profile is None:
                result["error"] = "Character profile extraction returned None"
                return result
                
            result["character_profile"] = profile.to_dict()
            logger.info(f"Extracted {profile.dialogue_count} dialogues for {character_name}")
            
            # Step 2: Generate system prompt
            system_prompt = self._generate_character_system_prompt(profile)
            result["system_prompt"] = system_prompt
            
            # Step 3: Generate training data
            logger.info(f"Generating {training_count} training examples...")
            training_data = self.generate_character_training_data(
                character_profile=profile,
                count=training_count,
            )
            
            # If hybrid/baked, inject personality into training data
            if personality_mode in ["baked", "hybrid"]:
                training_data = self._inject_personality_markers(training_data, profile)
            
            # Save training data
            safe_name = output_name.lower().replace(" ", "_")
            saved_path = self.save_training_data(
                position="chat",
                data=training_data,
                filename=f"{safe_name}_character_training.txt"
            )
            result["training_data_path"] = str(saved_path)
            logger.info(f"Saved training data to {saved_path}")
            
            # Step 4: Create bundle if requested
            if create_bundle:
                logger.info("Creating AI bundle...")
                bundle_path = self.create_bundle(
                    name=output_name,
                    description=f"AI based on {character_name} character",
                    model_paths={},  # Empty - user will train and add models
                    persona_name=output_name,
                    personality=self._summarize_personality(profile),
                    system_prompt=system_prompt if personality_mode in ["system_prompt", "hybrid"] else "",
                    tools_enabled=["chat"],
                )
                result["bundle_path"] = str(bundle_path)
                
                # Save character profile in bundle
                profile_path = bundle_path / "character_profile.json"
                with open(profile_path, 'w', encoding='utf-8') as f:
                    json.dump(profile.to_dict(), f, indent=2)
            
            result["success"] = True
            logger.info(f"Successfully created AI from character '{character_name}'")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Character-to-AI failed: {e}")
        
        return result
    
    def _generate_character_system_prompt(self, profile: "CharacterProfile") -> str:
        """
        Generate a comprehensive system prompt from character profile.
        
        More detailed than CharacterTrainer's basic prompt generation.
        """
        sections = []
        
        # Identity
        sections.append(f"You are {profile.name}.")
        
        # Personality traits
        strong_traits = [
            trait for trait, score in profile.personality_traits.items()
            if score > 0.3
        ]
        if strong_traits:
            trait_str = ", ".join(strong_traits)
            sections.append(f"Your personality is {trait_str}.")
        
        # Speech style
        speech_notes = []
        patterns = profile.speech_patterns
        
        if patterns.get("formal_language", 0) > 0.4:
            speech_notes.append("You speak formally and eloquently.")
        elif patterns.get("casual_language", 0) > 0.4:
            speech_notes.append("You speak casually and conversationally.")
        
        if patterns.get("uses_contractions", 0) > 0.5:
            speech_notes.append("You frequently use contractions.")
        elif patterns.get("uses_contractions", 0) < 0.2:
            speech_notes.append("You rarely use contractions.")
        
        if patterns.get("questions", 0) > 0.4:
            speech_notes.append("You often ask questions.")
        
        if patterns.get("exclamations", 0) > 0.3:
            speech_notes.append("You express yourself enthusiastically.")
        
        if speech_notes:
            sections.extend(speech_notes)
        
        # Catchphrases
        if profile.catchphrases:
            phrase_examples = profile.catchphrases[:3]
            sections.append(
                f"Your characteristic phrases include: {', '.join(repr(p) for p in phrase_examples)}."
            )
        
        # Topics of expertise/interest
        if profile.topics:
            topic_str = ", ".join(profile.topics[:5])
            sections.append(f"You often discuss topics like: {topic_str}.")
        
        # Response style instruction
        sections.append(
            "Stay in character and respond as this character would. "
            "Maintain consistent personality and speech patterns."
        )
        
        return " ".join(sections)
    
    def _inject_personality_markers(self, training_data: str, profile: "CharacterProfile") -> str:
        """
        Inject personality markers into training data for 'baked' personality.
        
        Adds subtle personality cues to responses:
        - Vocabulary preferences
        - Characteristic phrases
        - Speech pattern markers
        """
        lines = training_data.split("\n")
        modified_lines = []
        
        for line in lines:
            if line.startswith("ASSISTANT:"):
                # This is a response line - potentially modify it
                response = line[len("ASSISTANT:"):].strip()
                
                # Add vocabulary preferences (occasionally)
                if profile.vocabulary and random.random() < 0.1:
                    # Find a common word and add a related phrase
                    top_words = list(profile.vocabulary.keys())[:10]
                    if top_words:
                        word = random.choice(top_words)
                        # Simple augmentation - add word context
                        response = f"{response} Indeed, {word} is quite relevant here."
                
                modified_lines.append(f"ASSISTANT: {response}")
            else:
                modified_lines.append(line)
        
        return "\n".join(modified_lines)
    
    def _summarize_personality(self, profile: "CharacterProfile") -> str:
        """Create a brief personality summary for bundle metadata."""
        traits = [
            trait for trait, score in profile.personality_traits.items()
            if score > 0.3
        ]
        
        if traits:
            return f"{profile.name}: {', '.join(traits)}"
        else:
            return f"{profile.name} character"
    
    # ─────────────────────────────────────────────────────────────────────────
    # AI BUNDLE METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def create_bundle(
        self,
        name: str,
        description: str,
        model_paths: Dict[str, Path],
        persona_name: str = "AI Assistant",
        personality: str = "",
        system_prompt: str = "",
        tools_enabled: Optional[List[str]] = None,
        preset_used: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Create an AI bundle (.enigma-bundle) from trained models.
        
        Args:
            name: Name of the AI bundle
            description: Description of what this AI does
            model_paths: Dict mapping position names to model weight paths
            persona_name: Display name for the AI
            personality: Personality description
            system_prompt: System prompt for the AI
            tools_enabled: List of enabled tools
            preset_used: Name of preset used for creation
            output_dir: Where to save the bundle (default: models/bundles/)
        
        Returns:
            Path to created bundle directory
        """
        import shutil
        from datetime import datetime
        
        # Determine output location
        if output_dir is None:
            output_dir = Path(CONFIG.get("models_dir", "models")) / "bundles"
        
        bundle_dir = output_dir / name.lower().replace(" ", "_")
        bundle_dir.mkdir(parents=True, exist_ok=True)
        
        # Create models subdirectory
        models_dir = bundle_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Copy model files to bundle
        model_refs = {}
        for position, model_path in model_paths.items():
            model_path = Path(model_path)
            if model_path.exists():
                dest = models_dir / f"{position}_model.pth"
                shutil.copy2(model_path, dest)
                model_refs[position] = f"models/{position}_model.pth"
        
        # Create bundle spec
        spec = AIBundleSpec(
            name=name,
            version="1.0.0",
            description=description,
            created=datetime.now().isoformat(),
            models=model_refs,
            persona_name=persona_name,
            personality=personality,
            system_prompt=system_prompt,
            positions_trained=list(model_paths.keys()),
            tools_enabled=tools_enabled or [],
            preset_used=preset_used,
        )
        
        # Write manifest
        manifest = create_bundle_manifest(spec)
        manifest_path = bundle_dir / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        # Write README
        readme_path = bundle_dir / "README.md"
        readme_content = f"""# {name}

{description}

## Installation

Copy this folder to your Enigma AI Engine `models/bundles/` directory, then use the Bundle Manager to import it.

## Capabilities

**Trained Positions:** {', '.join(spec.positions_trained)}
**Tools Enabled:** {', '.join(spec.tools_enabled) or 'None'}

## Persona

**Name:** {persona_name}
**Personality:** {personality or 'Not specified'}

## Created

{spec.created}
"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"Created AI bundle at: {bundle_dir}")
        return bundle_dir
    
    def load_bundle(self, bundle_path: Path) -> AIBundleSpec:
        """
        Load an AI bundle and return its specification.
        
        Args:
            bundle_path: Path to bundle directory or manifest.json
        
        Returns:
            AIBundleSpec with bundle configuration
        """
        bundle_path = Path(bundle_path)
        
        if bundle_path.is_file():
            manifest_path = bundle_path
        else:
            manifest_path = bundle_path / "manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")
        
        return load_bundle_manifest(manifest_path)
    
    def list_bundles(self, bundles_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        List all available AI bundles.
        
        Args:
            bundles_dir: Directory to search (default: models/bundles/)
        
        Returns:
            List of bundle info dicts
        """
        if bundles_dir is None:
            bundles_dir = Path(CONFIG.get("models_dir", "models")) / "bundles"
        
        bundles = []
        
        if not bundles_dir.exists():
            return bundles
        
        for item in bundles_dir.iterdir():
            manifest_path = item / "manifest.json" if item.is_dir() else None
            if manifest_path and manifest_path.exists():
                try:
                    spec = self.load_bundle(manifest_path)
                    bundles.append({
                        "path": str(item),
                        "name": spec.name,
                        "description": spec.description,
                        "version": spec.version,
                        "positions": spec.positions_trained,
                        "tools": spec.tools_enabled,
                    })
                except Exception as e:
                    logger.warning(f"Failed to load bundle {item}: {e}")
        
        return bundles

    # ─────────────────────────────────────────────────────────────────────────
    # PROMPT-DRIVEN AI CREATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def create_from_prompt(
        self,
        prompt: str,
        data_count: int = 200,
        auto_train: bool = False,
    ) -> Dict[str, Any]:
        """
        Create an AI from a natural language description.
        
        Examples:
            "Create an AI that speaks like Shakespeare and can write Python code"
            "Make a friendly assistant that can search the web and describe images"
            "I want an AI that talks like a pirate and generates images"
        
        Args:
            prompt: Natural language description of desired AI
            data_count: Number of training examples to generate per position
            auto_train: If True, automatically start training (not implemented yet)
        
        Returns:
            Dict with:
                - spec: ParsedAISpec object
                - summary: Human-readable summary
                - training_data: Generated training data by position
                - system_prompt: Generated system prompt
                - config: Configuration for the AI wizard
        """
        parser = get_prompt_parser()
        spec = parser.parse(prompt)
        summary = parser.generate_summary(spec)
        
        logger.info(f"Parsed prompt into spec: {summary}")
        
        # Generate training data for each position
        training_data = {}
        for position in spec.positions:
            try:
                data = self.generate_training_data(position, count=data_count)
                training_data[position] = data
            except Exception as e:
                logger.warning(f"Failed to generate data for {position}: {e}")
                training_data[position] = ""
        
        # Generate system prompt from spec
        system_prompt = self._generate_system_prompt_from_spec(spec)
        
        # Build config for the wizard/bundle
        config = {
            "name": spec.name or "Custom AI",
            "tools": spec.capabilities,
            "positions": spec.positions,
            "model_size": spec.model_size,
            "personality_mode": spec.personality_mode,
            "system_prompt": system_prompt,
            "character_traits": spec.character_traits,
            "speaking_style": spec.speaking_style,
            "restrictions": spec.restrictions,
        }
        
        return {
            "spec": spec,
            "summary": summary,
            "training_data": training_data,
            "system_prompt": system_prompt,
            "config": config,
            "original_prompt": prompt,
        }
    
    def _generate_system_prompt_from_spec(self, spec: ParsedAISpec) -> str:
        """Generate a system prompt from a parsed spec."""
        lines = ["You are a helpful AI assistant."]
        
        if spec.name:
            lines[0] = f"You are {spec.name}, a helpful AI assistant."
        
        if spec.speaking_style:
            style_descriptions = {
                "shakespearean": "Speak in the eloquent, poetic style of Shakespeare, using thee, thou, and flowery language.",
                "pirate": "Speak like a swashbuckling pirate, using 'arr', 'matey', and nautical terms.",
                "robotic": "Speak in a precise, logical manner like a sophisticated robot. Use technical terminology and structured responses.",
                "academic": "Speak like a knowledgeable professor, using formal language and thorough explanations.",
                "southern": "Speak with warm, folksy Southern charm. Use colloquial expressions and a friendly tone.",
                "british": "Speak with proper British English, using polite formal language with occasional dry wit.",
                "casual": "Speak in a relaxed, friendly manner like a helpful friend. Keep it chill and approachable.",
                "mysterious": "Speak with an air of mystery and wisdom. Be cryptic yet helpful.",
                "mystical": "Speak like a wise wizard, using metaphors and references to ancient knowledge.",
            }
            if spec.speaking_style in style_descriptions:
                lines.append(style_descriptions[spec.speaking_style])
        
        if spec.character_traits:
            trait_sentence = f"Your personality is {', '.join(spec.character_traits)}."
            lines.append(trait_sentence)
        
        if spec.capabilities:
            cap_descriptions = {
                "code": "You can write and explain code.",
                "image": "You can generate images from text descriptions.",
                "vision": "You can describe and analyze images.",
                "search": "You can search the web for information.",
                "avatar": "You can control animated avatars.",
                "audio": "You can generate speech and audio.",
                "video": "You can create videos and animations.",
                "3d": "You can generate 3D models.",
                "math": "You can solve mathematical problems step by step.",
                "file": "You can read and write files.",
                "chat": "You excel at conversation.",
            }
            enabled_caps = [cap_descriptions.get(cap, "") for cap in spec.capabilities if cap in cap_descriptions]
            if enabled_caps:
                lines.append(" ".join([c for c in enabled_caps if c]))
        
        if spec.restrictions:
            restriction_descriptions = {
                "no_profanity": "Never use profanity or inappropriate language.",
                "family_friendly": "Keep all responses family-friendly and appropriate for all ages.",
                "keep_short": "Keep your responses brief and to the point.",
                "no_opinions": "Do not express personal opinions; stick to facts.",
                "factual_only": "Only provide factual, verifiable information.",
            }
            for restriction in spec.restrictions:
                if restriction in restriction_descriptions:
                    lines.append(restriction_descriptions[restriction])
        
        return "\n".join(lines)
    
    def improve_from_prompt(
        self,
        prompt: str,
        existing_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Improve an existing AI based on a natural language prompt.
        
        Examples:
            "Make my AI better at explaining things"
            "Add image generation capability"
            "Make responses shorter"
            "Make it speak more formally"
        
        Args:
            prompt: Natural language description of desired improvement
            existing_config: Current configuration to improve
        
        Returns:
            Dict with:
                - changes: What was changed
                - new_config: Updated configuration
                - additional_training: New training data if capabilities were added
                - system_prompt_update: Updated system prompt if style changed
        """
        parser = get_prompt_parser()
        improvement_spec = parser.parse(prompt)
        
        changes = []
        new_config = existing_config.copy() if existing_config else {}
        additional_training = {}
        
        # Check for capability additions
        current_caps = set(new_config.get("tools", []))
        new_caps = set(improvement_spec.capabilities) - {"chat"}  # Exclude default chat
        
        added_caps = new_caps - current_caps
        if added_caps:
            changes.append(f"Added capabilities: {', '.join(added_caps)}")
            new_config["tools"] = list(current_caps | added_caps)
            
            # Generate training data for new capabilities
            for cap in added_caps:
                # Map capability to position
                cap_to_pos = {
                    "code": "code", "vision": "vision", "math": "math", "avatar": "avatar"
                }
                pos = cap_to_pos.get(cap, "router")
                try:
                    additional_training[pos] = self.generate_training_data(pos, count=100)
                except Exception as e:
                    logger.warning(f"Failed to generate improvement data for {pos}: {e}")
        
        # Check for position additions
        current_positions = set(new_config.get("positions", []))
        new_positions = set(improvement_spec.positions)
        
        added_positions = new_positions - current_positions
        if added_positions:
            changes.append(f"Added training positions: {', '.join(added_positions)}")
            new_config["positions"] = list(current_positions | added_positions)
        
        # Check for trait/style changes
        if improvement_spec.speaking_style:
            old_style = new_config.get("speaking_style")
            if old_style != improvement_spec.speaking_style:
                changes.append(f"Changed speaking style to: {improvement_spec.speaking_style}")
                new_config["speaking_style"] = improvement_spec.speaking_style
        
        if improvement_spec.character_traits:
            current_traits = set(new_config.get("character_traits", []))
            new_traits = set(improvement_spec.character_traits)
            added_traits = new_traits - current_traits
            if added_traits:
                changes.append(f"Added traits: {', '.join(added_traits)}")
                new_config["character_traits"] = list(current_traits | new_traits)
        
        # Check for restriction changes
        if improvement_spec.restrictions:
            current_restrictions = set(new_config.get("restrictions", []))
            new_restrictions = set(improvement_spec.restrictions)
            added_restrictions = new_restrictions - current_restrictions
            if added_restrictions:
                changes.append(f"Added restrictions: {', '.join(added_restrictions)}")
                new_config["restrictions"] = list(current_restrictions | new_restrictions)
        
        # Check for size change
        if improvement_spec.model_size != "small":  # small is default
            current_size = new_config.get("model_size", "small")
            if current_size != improvement_spec.model_size:
                changes.append(f"Changed model size: {current_size} -> {improvement_spec.model_size}")
                new_config["model_size"] = improvement_spec.model_size
        
        # Regenerate system prompt if style/traits changed
        system_prompt_update = None
        if improvement_spec.speaking_style or improvement_spec.character_traits:
            # Build a combined spec for prompt generation
            combined_spec = ParsedAISpec(
                name=new_config.get("name"),
                character_traits=new_config.get("character_traits", []),
                speaking_style=new_config.get("speaking_style"),
                capabilities=new_config.get("tools", ["chat"]),
                restrictions=new_config.get("restrictions", []),
            )
            system_prompt_update = self._generate_system_prompt_from_spec(combined_spec)
            new_config["system_prompt"] = system_prompt_update
            changes.append("Updated system prompt")
        
        if not changes:
            changes.append("No specific improvements detected - consider being more specific")
        
        return {
            "changes": changes,
            "new_config": new_config,
            "additional_training": additional_training,
            "system_prompt_update": system_prompt_update,
            "improvement_spec": improvement_spec,
            "original_prompt": prompt,
        }

    # =========================================================================
    # QUALITY TRACKING INTEGRATION
    # =========================================================================

    def track_training_run(
        self,
        model_id: str,
        position: str,
        final_loss: float,
        epochs: int,
        training_examples: int,
        loss_history: Optional[List[float]] = None,
        data_quality: Optional[DataQualityScore] = None,
    ) -> ModelQualityMetrics:
        """
        Track a training run in the quality dashboard.
        
        Call this after training completes to record metrics for the dashboard.
        
        Args:
            model_id: Unique identifier for the model
            position: Router position (router, chat, code, etc.)
            final_loss: Final training loss
            epochs: Number of training epochs
            training_examples: Number of training examples used
            loss_history: Optional list of loss values per epoch
            data_quality: Optional data quality assessment
        
        Returns:
            ModelQualityMetrics with recorded data
        
        Example:
            trainer = get_trainer_ai()
            
            # After training completes:
            metrics = trainer.track_training_run(
                model_id="my_chatbot_v1",
                position="chat", 
                final_loss=0.15,
                epochs=10,
                training_examples=500,
                loss_history=[0.8, 0.5, 0.3, 0.2, 0.18, 0.16, 0.15, 0.15, 0.15, 0.15]
            )
            
            print(f"Overfitting risk: {metrics.overfitting_risk}")
            print(f"Needs retraining: {metrics.needs_retraining}")
        """
        tracker = get_quality_tracker()
        return tracker.record_training_metrics(
            model_id=model_id,
            position=position,
            final_loss=final_loss,
            epochs=epochs,
            training_examples=training_examples,
            loss_history=loss_history,
            data_quality=data_quality,
        )
    
    def get_model_quality(self, model_id: str) -> Optional[ModelQualityMetrics]:
        """
        Get the latest quality metrics for a model.
        
        Args:
            model_id: Unique identifier for the model
        
        Returns:
            Latest ModelQualityMetrics or None if no metrics recorded
        """
        tracker = get_quality_tracker()
        history = tracker.get_metrics_history(model_id, limit=1)
        return history[0] if history else None
    
    def get_ai_health(self, model_id: str) -> Optional[ModelHealthStatus]:
        """
        Get the health status for a model.
        
        Args:
            model_id: Unique identifier for the model
        
        Returns:
            ModelHealthStatus with grade, recommendations, etc.
        
        Example:
            trainer = get_trainer_ai()
            health = trainer.get_ai_health("my_chatbot_v1")
            
            print(f"Grade: {health.overall_grade}")  # A, B, C, D, F
            print(f"Status: {health.status}")  # healthy, warning, critical
            print(f"Recommendations: {health.recommendations}")
        """
        tracker = get_quality_tracker()
        return tracker.get_model_health(model_id)
    
    def suggest_retraining(self, model_id: str) -> Dict[str, Any]:
        """
        Check if a model needs retraining and get suggestions.
        
        Args:
            model_id: Unique identifier for the model
        
        Returns:
            Dict with retraining recommendations:
            - needs_retraining: bool
            - reason: str or None
            - suggestions: List[str]
            - current_grade: str
            - trend: str (improving, stable, degrading)
        """
        tracker = get_quality_tracker()
        health = tracker.get_model_health(model_id)
        trend_info = tracker.get_improvement_trend(model_id)
        
        if not health:
            return {
                "needs_retraining": False,
                "reason": "No metrics recorded for this model",
                "suggestions": ["Train the model first to establish baseline metrics"],
                "current_grade": "N/A",
                "trend": "unknown",
            }
        
        metrics = tracker.get_metrics_history(model_id, limit=1)
        latest = metrics[0] if metrics else None
        
        return {
            "needs_retraining": latest.needs_retraining if latest else False,
            "reason": latest.retraining_reason if latest else None,
            "suggestions": health.recommendations,
            "current_grade": health.overall_grade,
            "trend": trend_info.get("trend", "unknown"),
            "improvement_pct": trend_info.get("improvement_pct", 0),
        }
    
    def get_quality_dashboard(self) -> Dict[str, Any]:
        """
        Get a summary of all model quality for the dashboard UI.
        
        Returns:
            Dict with:
            - total_models: int
            - healthy: int (count of healthy models)
            - warning: int (count of models with warnings)
            - critical: int (count of critical models)
            - grade_distribution: Dict[str, int] (count per grade)
            - models: List of model summaries
        
        Example:
            trainer = get_trainer_ai()
            dashboard = trainer.get_quality_dashboard()
            
            print(f"Total models: {dashboard['total_models']}")
            print(f"Healthy: {dashboard['healthy']}")
            print(f"Need attention: {dashboard['warning'] + dashboard['critical']}")
            
            for model in dashboard['models']:
                print(f"{model['id']}: Grade {model['grade']} - {model['status']}")
        """
        tracker = get_quality_tracker()
        return tracker.get_dashboard_summary()
    
    def get_models_needing_attention(self) -> List[ModelHealthStatus]:
        """
        Get all models that need attention (warnings, critical, needs_attention).
        
        Returns:
            List of ModelHealthStatus for models that need attention
        """
        tracker = get_quality_tracker()
        return tracker.get_models_needing_attention()

    # =========================================================================
    # DATA MARKETPLACE INTEGRATION
    # =========================================================================
    
    def create_data_pack(
        self,
        name: str,
        description: str,
        author: str,
        position: str = "chat",
        training_file: Optional[str] = None,
        training_data: Optional[str] = None,
        character_name: Optional[str] = None,
        speaking_style: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> DataPack:
        """
        Create a shareable data pack from training data.
        
        Args:
            name: Display name for the pack
            description: Description of what the pack trains
            author: Author name
            position: Router position (chat, code, router, etc.)
            training_file: Path to training data file
            training_data: Or raw training data string
            character_name: For character packs, the character's name
            speaking_style: Speaking style description
            tags: Search tags
        
        Returns:
            Created DataPack
        
        Example:
            trainer = get_trainer_ai()
            pack = trainer.create_data_pack(
                name="Pirate Bot",
                description="Training data for pirate speech",
                author="Captain Jack",
                position="chat",
                training_file="data/pirate_chat.txt",
                character_name="BlackBeard",
                speaking_style="pirate",
                tags=["pirate", "character", "fun"]
            )
        """
        marketplace = get_data_marketplace()
        return marketplace.create_pack(
            name=name,
            description=description,
            author=author,
            position=position,
            training_file=training_file,
            training_data=training_data,
            character_name=character_name,
            speaking_style=speaking_style,
            tags=tags,
        )
    
    def export_data_pack(self, pack_id: str, output_path: Optional[str] = None) -> str:
        """
        Export a data pack as a shareable zip file.
        
        Args:
            pack_id: ID of pack to export
            output_path: Optional output path
        
        Returns:
            Path to exported zip file
        """
        marketplace = get_data_marketplace()
        return marketplace.export_pack(pack_id, output_path)
    
    def import_data_pack(self, zip_path: str) -> DataPack:
        """
        Import a data pack from a zip file.
        
        Args:
            zip_path: Path to the zip file
        
        Returns:
            Imported DataPack
        """
        marketplace = get_data_marketplace()
        return marketplace.import_pack(zip_path)
    
    def browse_marketplace(
        self,
        position: Optional[str] = None,
        category: Optional[str] = None,
        search: Optional[str] = None,
        min_rating: float = 0.0,
    ) -> List[DataPack]:
        """
        Browse available data packs in the marketplace.
        
        Args:
            position: Filter by router position
            category: Filter by category (general, character, task, style)
            search: Search in name/description/tags
            min_rating: Minimum community rating (1-5)
        
        Returns:
            List of matching DataPacks
        """
        marketplace = get_data_marketplace()
        return marketplace.list_packs(
            position=position,
            category=category,
            search=search,
            min_rating=min_rating,
        )
    
    def rate_data_pack(self, pack_id: str, rating: float, review: str = "") -> DataPackRating:
        """
        Rate a data pack.
        
        Args:
            pack_id: ID of pack to rate
            rating: Rating 1-5 stars
            review: Optional review text
        
        Returns:
            Created rating
        """
        marketplace = get_data_marketplace()
        return marketplace.rate_pack(pack_id, rating, review)
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        marketplace = get_data_marketplace()
        return marketplace.get_marketplace_stats()

    # =========================================================================
    # INCREMENTAL TRAINING INTEGRATION
    # =========================================================================
    
    def teach_skill(
        self,
        model_id: str,
        skill_name: str,
        training_data: Optional[str] = None,
        epochs: int = 5,
    ) -> Dict[str, Any]:
        """
        Teach a new skill to an existing model incrementally.
        
        Args:
            model_id: Model identifier
            skill_name: Name of skill to teach (e.g., "math", "coding", "storytelling")
            training_data: Optional custom training data
            epochs: Number of training epochs
        
        Returns:
            Dict with training results and guidance
        
        Example:
            trainer = get_trainer_ai()
            result = trainer.teach_skill(
                model_id="my_chatbot",
                skill_name="math",
                training_data="USER: What is 5+5?\\nASSISTANT: 10",
                epochs=5
            )
        """
        inc_trainer = get_incremental_trainer()
        return inc_trainer.teach_skill(
            model_id=model_id,
            skill_name=skill_name,
            training_data=training_data,
            epochs=epochs,
        )
    
    def list_available_skills(self) -> List[SkillDefinition]:
        """List all available skills that can be taught."""
        inc_trainer = get_incremental_trainer()
        return inc_trainer.list_skills()
    
    def register_custom_skill(
        self,
        name: str,
        description: str,
        position: str = "chat",
        training_data: Optional[str] = None,
    ) -> SkillDefinition:
        """
        Register a custom skill definition.
        
        Args:
            name: Unique skill name
            description: What this skill does
            position: Router position
            training_data: Training data for this skill
        
        Returns:
            Created SkillDefinition
        """
        inc_trainer = get_incremental_trainer()
        return inc_trainer.register_skill(
            name=name,
            description=description,
            position=position,
            training_data=training_data,
        )
    
    def get_model_learning_history(self, model_id: str) -> Optional[TrainingHistory]:
        """
        Get the training history for a model.
        
        Args:
            model_id: Model identifier
        
        Returns:
            TrainingHistory with all training runs and skills learned
        """
        inc_trainer = get_incremental_trainer()
        return inc_trainer.get_training_history(model_id)
    
    def merge_models(
        self,
        base_model_path: str,
        skill_model_path: str,
        output_path: str,
        skill_weight: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Merge a skill model into a base model.
        
        Args:
            base_model_path: Path to base model
            skill_model_path: Path to specialized model
            output_path: Where to save merged model
            skill_weight: Weight for skill model (0.0-1.0)
        
        Returns:
            Dict with merge instructions
        """
        inc_trainer = get_incremental_trainer()
        return inc_trainer.merge_models(
            base_model_path=base_model_path,
            skill_model_path=skill_model_path,
            output_path=output_path,
            skill_weight=skill_weight,
        )
    
    def learn_from_conversation(
        self,
        model_id: str,
        conversation: List[Dict[str, str]],
        user_approved: bool = False,
    ) -> Dict[str, Any]:
        """
        Record a conversation for potential training (requires user consent).
        
        Args:
            model_id: Model identifier
            conversation: List of {"role": "user"|"assistant", "content": str}
            user_approved: User consent flag (must be True)
        
        Returns:
            Dict with recording status
        """
        inc_trainer = get_incremental_trainer()
        return inc_trainer.record_conversation_learning(
            model_id=model_id,
            conversation=conversation,
            user_approved=user_approved,
        )

    # =========================================================================
    # MODEL INHERITANCE INTEGRATION
    # =========================================================================
    
    def clone_model(
        self,
        source_model_id: str,
        new_name: str,
        inherit_weights: bool = True,
        inherit_personality: bool = True,
        new_speaking_style: Optional[str] = None,
        new_traits: Optional[List[str]] = None,
        add_capabilities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Clone an existing model with customizations.
        
        Args:
            source_model_id: ID of model to clone
            new_name: Name for the new model
            inherit_weights: Whether to copy model weights
            inherit_personality: Whether to inherit personality/style
            new_speaking_style: Optional new speaking style
            new_traits: Optional new character traits
            add_capabilities: Optional capabilities to add
        
        Returns:
            Dict with clone results
        
        Example:
            trainer = get_trainer_ai()
            result = trainer.clone_model(
                source_model_id="helpful_bot",
                new_name="pirate_bot",
                new_speaking_style="pirate",
                add_capabilities=["storytelling"]
            )
        """
        inheritance = get_model_inheritance()
        config = CloneConfig(
            name=new_name,
            inherit_weights=inherit_weights,
            inherit_personality=inherit_personality,
            new_speaking_style=new_speaking_style,
            new_traits=new_traits,
            add_capabilities=add_capabilities or [],
        )
        return inheritance.clone_model(source_model_id, config)
    
    def fork_community_bundle(
        self,
        bundle_path: str,
        new_name: str,
        customize: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Fork a community AI bundle to create your own version.
        
        Args:
            bundle_path: Path to the bundle zip
            new_name: Name for your fork
            customize: Optional customizations
        
        Returns:
            Dict with fork results
        """
        inheritance = get_model_inheritance()
        return inheritance.fork_bundle(bundle_path, new_name, customize)
    
    def get_model_lineage(self, model_id: str) -> Optional[ModelLineage]:
        """Get the lineage/ancestry of a model."""
        inheritance = get_model_inheritance()
        return inheritance.get_lineage(model_id)
    
    def get_model_ancestry(self, model_id: str) -> List[ModelLineage]:
        """Get full ancestry chain for a model."""
        inheritance = get_model_inheritance()
        return inheritance.get_full_ancestry(model_id)
    
    def get_inheritance_tree(self) -> Dict[str, Any]:
        """Get a tree view of all model inheritance relationships."""
        inheritance = get_model_inheritance()
        return inheritance.get_inheritance_tree()

    # =========================================================================
    # API-POWERED TRAINING
    # =========================================================================
    
    def configure_api(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        store_securely: bool = True,
    ) -> None:
        """
        Configure an API provider for training data generation.
        
        Args:
            provider: "openai", "anthropic", or "custom"
            api_key: API key (if None, loads from secure storage/env)
            model: Model name (defaults based on provider)
            base_url: Custom base URL (for custom providers)
            store_securely: If True, stores API key in encrypted storage
        
        Example:
            trainer = get_trainer_ai()
            
            # Option 1: Pass API key directly (will be stored securely)
            trainer.configure_api("openai", "sk-...")
            
            # Option 2: Use previously stored key
            trainer.configure_api("openai")  # Loads from secure storage
            
            data = trainer.generate_api_training_data(["chat", "code"])
        """
        api = get_api_training_provider()
        
        if provider == "openai":
            api.configure_openai(api_key, model or "gpt-4", base_url, store_securely)
        elif provider == "anthropic":
            api.configure_anthropic(api_key, model or "claude-3-opus-20240229", store_securely)
        elif provider == "custom":
            if not base_url:
                raise ValueError("base_url required for custom provider")
            if not api_key:
                raise ValueError("api_key required for custom provider")
            api.configure_custom("custom", api_key, base_url, model or "gpt-4", store_securely)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', or 'custom'")
    
    def list_api_training_tasks(self) -> Dict[str, str]:
        """
        List available training tasks for API generation.
        
        Returns:
            Dict mapping task keys to descriptions
        """
        return {
            k: f"{v['name']} - {v['description']}"
            for k, v in TRAINING_TASKS.items()
        }
    
    def generate_api_training_data(
        self,
        tasks: Union[str, List[str]] = "all",
        examples_per_task: int = 50,
        output_file: Optional[str] = None,
        separate_files: bool = False,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Generate training data using an API (GPT-4, Claude, etc.).
        
        This is the main method for API-powered training data generation.
        Configure an API first with configure_api().
        
        Args:
            tasks: "all" for all tasks, or list like ["chat", "code", "avatar"]
            examples_per_task: Number of examples per task (default 50)
            output_file: Optional path to save combined data
            separate_files: If True, creates separate file per task
            callback: Optional progress callback(task, current, total)
        
        Returns:
            Dict with:
            - data: Generated training data by task
            - files: Paths to saved files (if output specified)
            - stats: Generation statistics
        
        Example:
            trainer = get_trainer_ai()
            trainer.configure_api("openai", "sk-...")
            
            # Generate for specific tasks
            result = trainer.generate_api_training_data(
                tasks=["chat", "code", "avatar"],
                examples_per_task=100
            )
            
            # Generate for ALL tasks
            result = trainer.generate_api_training_data(
                tasks="all",
                examples_per_task=50,
                output_file="data/full_training.txt"
            )
        """
        api = get_api_training_provider()
        
        if not api._active_provider:
            raise ValueError("No API configured. Call configure_api() first.")
        
        result = {
            "data": {},
            "files": {},
            "stats": {},
        }
        
        if separate_files:
            output_dir = output_file or str(Path(CONFIG.get("data_dir", "data")) / "api_training")
            result["files"] = api.generate_separate_training_files(
                tasks, examples_per_task, output_dir, callback
            )
            result["data"] = api._generated_data
        elif output_file:
            file_path = api.generate_combined_training_file(
                tasks, examples_per_task, output_file, callback
            )
            result["files"] = {"combined": file_path}
            result["data"] = api._generated_data
        else:
            result["data"] = api.generate_training_data(tasks, examples_per_task, callback)
        
        result["stats"] = api.get_generation_stats()
        return result
    
    def train_from_api_data(
        self,
        tasks: Union[str, List[str]],
        examples_per_task: int = 50,
        model_name: str = "api_trained_model",
        model_size: str = "small",
        epochs: int = 10,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Complete workflow: Generate API data and prepare for training.
        
        This combines API data generation with local model training setup.
        
        Args:
            tasks: Tasks to train on
            examples_per_task: Examples to generate per task
            model_name: Name for the trained model
            model_size: Model size (tiny, small, medium, large)
            epochs: Training epochs
            callback: Progress callback
        
        Returns:
            Dict with training configuration and data paths
        
        Example:
            trainer = get_trainer_ai()
            trainer.configure_api("openai", "sk-...")
            
            # Generate data and get training config
            config = trainer.train_from_api_data(
                tasks=["chat", "avatar", "code"],
                examples_per_task=100,
                model_name="my_assistant",
                model_size="small"
            )
            
            # The config contains everything needed to train
            print(config["training_file"])  # Path to training data
            print(config["positions"])  # Router positions covered
        """
        api = get_api_training_provider()
        
        # Generate the training data
        output_file = str(Path(CONFIG.get("data_dir", "data")) / f"{model_name}_training.txt")
        
        gen_result = self.generate_api_training_data(
            tasks=tasks,
            examples_per_task=examples_per_task,
            output_file=output_file,
            callback=callback,
        )
        
        # Build the training configuration
        trainer_config = api.create_trainer_ai_config(
            tasks=tasks if isinstance(tasks, list) else list(TRAINING_TASKS.keys()),
            model_size=model_size,
            name=model_name,
        )
        
        return {
            "model_name": model_name,
            "model_size": model_size,
            "training_file": output_file,
            "positions": trainer_config["positions"],
            "tasks": trainer_config["tasks"],
            "capabilities": trainer_config["capabilities"],
            "epochs": epochs,
            "stats": gen_result["stats"],
            "ready_to_train": True,
            "train_command": f"python scripts/train_specialized_model.py --data {output_file} --size {model_size} --epochs {epochs}",
        }
    
    def create_task_combination_trainer(
        self,
        task_combinations: List[List[str]],
        examples_per_task: int = 50,
        base_name: str = "combo_trainer",
        callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create multiple trainers, each for a different task combination.
        
        This allows creating:
        - One AI that does everything
        - Separate specialist AIs
        - Custom combinations
        
        Args:
            task_combinations: List of task lists, e.g.:
                [
                    ["chat", "code"],  # Chat + Code specialist
                    ["avatar", "game"],  # Avatar + Game specialist
                    ["all"],  # Full trainer
                ]
            examples_per_task: Examples per task
            base_name: Base name for trainers
            callback: Progress callback
        
        Returns:
            List of training configurations, one per combination
        
        Example:
            trainer = get_trainer_ai()
            trainer.configure_api("openai", "sk-...")
            
            configs = trainer.create_task_combination_trainer([
                ["chat"],  # Chat only
                ["chat", "code"],  # Chat + Code
                ["chat", "avatar", "vision"],  # Chat + Avatar + Vision
                ["all"],  # Everything
            ])
            
            for config in configs:
                print(f"{config['model_name']}: {config['tasks']}")
        """
        results = []
        
        for i, tasks in enumerate(task_combinations):
            task_str = "_".join(tasks) if tasks != ["all"] else "all_tasks"
            model_name = f"{base_name}_{task_str}"
            
            config = self.train_from_api_data(
                tasks=tasks[0] if tasks == ["all"] else tasks,
                examples_per_task=examples_per_task,
                model_name=model_name,
                callback=callback,
            )
            
            results.append(config)
        
        return results
    
    # =========================================================================
    # ASYNC / NON-BLOCKING OPERATIONS
    # =========================================================================
    # These methods allow training to run in background without interrupting
    # the main AI's chat flow.
    
    def train_async(
        self,
        data_path: str,
        model_name: str,
        model_size: str = "small",
        epochs: int = 3,
        on_progress: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
    ) -> str:
        """
        Start training in background without blocking the main AI.
        
        Returns a task_id that can be used to check progress or cancel.
        The main AI continues responding normally during training.
        
        Args:
            data_path: Path to training data
            model_name: Name for the trained model
            model_size: Model size
            epochs: Training epochs
            on_progress: Called with progress updates
            on_complete: Called when training finishes
            
        Returns:
            task_id string for tracking the background task
        """
        from .async_training import get_async_trainer
        
        trainer = get_async_trainer()
        return trainer.start_training_async(
            data_path=data_path,
            model_name=model_name,
            model_size=model_size,
            epochs=epochs,
            on_progress=on_progress,
            on_complete=on_complete,
        )
    
    def generate_data_async(
        self,
        position: str,
        count: int = 100,
        use_api: bool = False,
        on_progress: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
    ) -> str:
        """
        Generate training data in background without blocking.
        
        Args:
            position: Router position (code, chat, router, etc.)
            count: Number of examples to generate
            use_api: Use API for generation
            on_progress: Progress callback
            on_complete: Completion callback
            
        Returns:
            task_id for tracking
        """
        from .async_training import get_async_trainer
        
        trainer = get_async_trainer()
        return trainer.start_data_generation_async(
            position=position,
            count=count,
            use_api=use_api,
            on_progress=on_progress,
            on_complete=on_complete,
        )
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a background training task."""
        from .async_training import get_async_trainer
        
        progress = get_async_trainer().get_task_status(task_id)
        return progress.to_dict() if progress else None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a background training task."""
        from .async_training import get_async_trainer
        return get_async_trainer().cancel_task(task_id)
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get all currently running background tasks."""
        from .async_training import get_async_trainer
        return [p.to_dict() for p in get_async_trainer().get_active_tasks()]
    
    # =========================================================================
    # META-LEARNING - THE TRAINER LEARNS TO TEACH BETTER
    # =========================================================================
    # These methods allow the Trainer AI to learn and improve its teaching
    # strategies over time.
    
    def record_teaching_result(
        self,
        task_type: str,
        strategy: str,
        examples_used: int,
        initial_score: float,
        final_score: float,
        training_time: float,
        model_size: str = "small",
    ):
        """
        Record the result of a training attempt so the Trainer can learn.
        
        This builds up knowledge about what teaching strategies work best
        for different tasks.
        
        Args:
            task_type: Type of task taught
            strategy: Teaching strategy used
            examples_used: Number of training examples
            initial_score: Model performance before (0-1)
            final_score: Model performance after (0-1)
            training_time: Time spent training (seconds)
            model_size: Size of model trained
        """
        from .meta_learning import get_meta_learner
        
        meta = get_meta_learner()
        meta.record_teaching_attempt(
            task_type=task_type,
            strategy_name=strategy,
            examples_used=examples_used,
            initial_performance=initial_score,
            final_performance=final_score,
            training_time=training_time,
            model_size=model_size,
        )
    
    def get_best_teaching_strategy(self, task_type: str) -> Dict[str, Any]:
        """
        Get the best teaching strategy for a task type.
        
        The Trainer learns which approaches work best based on past results.
        
        Args:
            task_type: Type of task to teach
            
        Returns:
            Strategy configuration dict
        """
        from .meta_learning import get_meta_learner
        
        strategy = get_meta_learner().get_best_strategy(task_type)
        return strategy.to_dict()
    
    def get_optimized_teaching_plan(
        self,
        task_type: str,
        model_size: str = "small",
        target_performance: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Get an optimized teaching plan based on learned experience.
        
        Args:
            task_type: Task to teach
            model_size: Target model size
            target_performance: Desired performance level
            
        Returns:
            Optimized teaching plan
        """
        from .meta_learning import get_meta_learner
        
        return get_meta_learner().optimize_teaching_plan(
            task_type=task_type,
            model_size=model_size,
            target_performance=target_performance,
        )
    
    def get_teaching_insights(self, task_type: Optional[str] = None) -> List[str]:
        """
        Get insights the Trainer has learned about teaching.
        
        Args:
            task_type: Optional filter by task
            
        Returns:
            List of learned insights
        """
        from .meta_learning import get_meta_learner
        return get_meta_learner().get_insights(task_type)
    
    def get_teaching_report(self) -> Dict[str, Any]:
        """Get comprehensive report on teaching performance."""
        from .meta_learning import get_meta_learner
        return get_meta_learner().get_performance_report()
    
    # =========================================================================
    # WEB-ASSISTED TRAINING
    # =========================================================================
    # The Trainer can use the web to gather training data.
    
    def generate_from_web(
        self,
        position: str,
        count: int = 100,
        topics: Optional[List[str]] = None,
    ) -> str:
        """
        Generate training data by collecting from the web.
        
        Args:
            position: Router position (code, chat, router, etc.)
            count: Number of examples to collect
            topics: Optional list of topics to search for
            
        Returns:
            Training data string in proper format
        """
        from .web_training import get_web_collector
        
        return get_web_collector().generate_for_trainer_ai(
            position=position,
            count=count,
            topics=topics,
        )
    
    def scrape_training_data(
        self,
        url: str,
        topic: str,
        task_type: str = "chat",
    ) -> Dict[str, Any]:
        """
        Scrape a specific URL for training data.
        
        Args:
            url: URL to scrape
            topic: Topic context
            task_type: Type of data to extract
            
        Returns:
            Dict with extracted examples
        """
        from .web_training import get_web_collector
        return get_web_collector().scrape_for_training(url, topic, task_type)
    
    def scrape_documentation_async(
        self,
        doc_url: str,
        topic: str,
        max_pages: int = 10,
        on_progress: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
    ) -> str:
        """
        Scrape documentation in background for training data.
        
        Args:
            doc_url: Base documentation URL
            topic: Topic name
            max_pages: Maximum pages to scrape
            on_progress: Progress callback
            on_complete: Completion callback
            
        Returns:
            task_id for tracking
        """
        from .async_training import get_async_trainer
        
        return get_async_trainer().start_web_scraping_async(
            urls=[doc_url],
            topic=topic,
            max_pages=max_pages,
            on_progress=on_progress,
            on_complete=on_complete,
        )
    
    def collect_web_training_data(
        self,
        topics: List[str],
        task_types: List[str],
        examples_per_topic: int = 50,
        output_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect comprehensive training data from multiple web sources.
        
        Args:
            topics: List of topics to collect
            task_types: Task types to extract
            examples_per_topic: Examples per topic
            output_file: Optional path to save data
            
        Returns:
            List of extracted training examples
        """
        from .web_training import get_web_collector
        from pathlib import Path
        
        examples = get_web_collector().collect_training_data(
            topics=topics,
            task_types=task_types,
            examples_per_topic=examples_per_topic,
            output_file=Path(output_file) if output_file else None,
        )
        
        return [
            {
                "input": ex.input_text,
                "output": ex.output_text,
                "task_type": ex.task_type,
                "source": ex.source_url,
                "quality": ex.quality_score,
            }
            for ex in examples
        ]


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_trainer_ai_instance: Optional[TrainerAI] = None


def get_trainer_ai(model=None) -> TrainerAI:
    """
    Get the global TrainerAI instance.
    
    Args:
        model: Optional model to use for AI-powered generation
    
    Returns:
        TrainerAI instance
    """
    global _trainer_ai_instance
    
    if _trainer_ai_instance is None:
        _trainer_ai_instance = TrainerAI(model=model)
    elif model is not None and _trainer_ai_instance.model is None:
        _trainer_ai_instance.model = model
        _trainer_ai_instance.use_ai = True
    
    return _trainer_ai_instance


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TrainerAI - Generate and curate training data")
    parser.add_argument("action", choices=["generate", "curate", "template", "list"],
                        help="Action to perform")
    parser.add_argument("--position", "-p", help="Router position (router, vision, code, etc.)")
    parser.add_argument("--count", "-c", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--input", "-i", help="Input file for curation")
    parser.add_argument("--output", "-o", help="Output file")
    
    args = parser.parse_args()
    
    trainer = TrainerAI()
    
    if args.action == "list":
        print("Available router positions:")
        for name, config in POSITION_CONFIGS.items():
            print(f"  {name}: {config.description}")
            print(f"    - Recommended size: {config.recommended_model_size}")
            print(f"    - Min examples: {config.example_count_min}")
        
    elif args.action == "template":
        if not args.position:
            print("Error: --position required for template")
        else:
            print(trainer.get_template(args.position, count=5))
        
    elif args.action == "generate":
        if not args.position:
            print("Error: --position required for generate")
        else:
            data = trainer.generate_training_data(args.position, count=args.count)
            if args.output:
                trainer.save_training_data(args.position, data, args.output)
                print(f"Saved to: {args.output}")
            else:
                print(data)
        
    elif args.action == "curate":
        if not args.position or not args.input:
            print("Error: --position and --input required for curate")
        else:
            with open(args.input) as f:
                raw_data = f.read()
            
            cleaned, score = trainer.curate_data(args.position, raw_data)
            
            print(f"\nQuality Score: {score.overall_score:.2f}")
            print(f"  Format: {score.format_score:.2f}")
            print(f"  Diversity: {score.diversity_score:.2f}")
            print(f"  Completeness: {score.completeness_score:.2f}")
            
            if score.issues:
                print("\nIssues:")
                for issue in score.issues:
                    print(f"  - {issue}")
            
            if score.suggestions:
                print("\nSuggestions:")
                for suggestion in score.suggestions:
                    print(f"  - {suggestion}")
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(cleaned)
                print(f"\nSaved cleaned data to: {args.output}")
