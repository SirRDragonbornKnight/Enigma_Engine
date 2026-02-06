"""
Unified Prompt Manager - Single source of truth for all AI system prompts.

This module provides a centralized system for managing prompts across all
generation modules (image_tab, code_tab, audio_tab, etc.) with support for
persona integration, module-specific overrides, and safety prompts.

Usage:
    from forge_ai.core.prompt_manager import PromptManager, get_prompt_manager
    
    manager = get_prompt_manager()
    
    # Get prompt for a module
    prompt = manager.get_system_prompt("chat")
    
    # Get prompt with persona
    prompt = manager.get_system_prompt("code", persona_id="my_persona")
    
    # Register custom prompt
    manager.register_prompt("custom_module", "You are a custom assistant.")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ..config import CONFIG

logger = logging.getLogger(__name__)


# ==============================================================================
# PROMPT DEFINITIONS
# ==============================================================================

@dataclass
class PromptDefinition:
    """Definition of a prompt for a specific module."""
    module: str
    base_prompt: str
    description: str = ""
    safety_append: bool = True          # Whether to append safety prompt
    persona_prepend: bool = True        # Whether to prepend persona prompt
    variables: Dict[str, str] = field(default_factory=dict)  # Variable substitutions
    version: str = "1.0"
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "base_prompt": self.base_prompt,
            "description": self.description,
            "safety_append": self.safety_append,
            "persona_prepend": self.persona_prepend,
            "variables": self.variables,
            "version": self.version,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptDefinition":
        return cls(**data)


@dataclass
class PromptOverride:
    """Override for a specific prompt in a specific context."""
    module: str
    context: str                        # e.g., "persona:my_ai", "mode:game"
    override_prompt: str
    merge_mode: str = "replace"         # "replace", "prepend", "append"
    priority: int = 0                   # Higher = applied later


@dataclass
class SafetyPrompt:
    """Safety prompt that can be appended to any generation."""
    name: str
    content: str
    modules: Set[str] = field(default_factory=set)  # Empty = all modules
    always_include: bool = False
    priority: int = 100


# ==============================================================================
# DEFAULT PROMPTS
# ==============================================================================

DEFAULT_BASE_PROMPT = """You are Forge, an AI assistant created with ForgeAI.
You are helpful, harmless, and honest."""

DEFAULT_PROMPTS: Dict[str, PromptDefinition] = {
    "chat": PromptDefinition(
        module="chat",
        base_prompt="""You are Forge, a helpful AI assistant.
You provide clear, accurate, and helpful responses.
You can use tools when needed and explain your reasoning.""",
        description="General chat and conversation",
    ),
    
    "image": PromptDefinition(
        module="image",
        base_prompt="""You are an AI assistant that helps create images.
When asked to generate an image, describe it in detail for the image generator.
Include style, mood, composition, colors, and artistic direction.
Be creative but follow the user's intent.""",
        description="Image generation assistance",
    ),
    
    "code": PromptDefinition(
        module="code",
        base_prompt="""You are an expert programmer and code assistant.
Write clean, efficient, well-documented code.
Follow best practices for the language being used.
Explain your code when helpful.
Consider edge cases and error handling.""",
        description="Code generation and assistance",
    ),
    
    "video": PromptDefinition(
        module="video",
        base_prompt="""You are an AI assistant for video generation.
Help users describe videos clearly for the video generator.
Consider timing, transitions, motion, and visual storytelling.
Provide detailed scene descriptions.""",
        description="Video generation assistance",
    ),
    
    "audio": PromptDefinition(
        module="audio",
        base_prompt="""You are an AI assistant for audio and speech generation.
Help users with text-to-speech and audio creation.
Consider tone, pacing, emotion, and clarity.
Adapt your suggestions to the intended use.""",
        description="Audio and TTS assistance",
    ),
    
    "3d": PromptDefinition(
        module="3d",
        base_prompt="""You are an AI assistant for 3D model generation.
Help users describe 3D objects and scenes.
Consider geometry, materials, textures, and proportions.
Be specific about shapes and spatial relationships.""",
        description="3D model generation assistance",
    ),
    
    "avatar": PromptDefinition(
        module="avatar",
        base_prompt="""You are an AI with an avatar that can express emotions and actions.
Use your avatar to enhance communication with expressions and gestures.
Match your expressions to the emotional tone of conversations.
Available expressions: happy, sad, surprised, thinking, excited, neutral.""",
        description="Avatar control and expression",
        safety_append=False,  # Avatar doesn't need safety prompts
    ),
    
    "game": PromptDefinition(
        module="game",
        base_prompt="""You are an AI game companion and assistant.
Help players with strategies, tips, and in-game assistance.
Adapt your personality to the game's theme.
Be engaging and fun while being helpful.""",
        description="Game AI assistance",
    ),
    
    "robot": PromptDefinition(
        module="robot",
        base_prompt="""You are an AI controlling physical robot hardware.
Execute movements safely and precisely.
Always verify commands before executing.
Report any errors or unexpected conditions.
Safety is the top priority.""",
        description="Robot and hardware control",
    ),
    
    "vision": PromptDefinition(
        module="vision",
        base_prompt="""You are an AI with vision capabilities.
You can see and analyze images and screens.
Describe what you see accurately and helpfully.
Protect user privacy - don't share sensitive information seen on screen.""",
        description="Vision and screen analysis",
    ),
    
    "web": PromptDefinition(
        module="web",
        base_prompt="""You are an AI assistant with web search capabilities.
Search for accurate, up-to-date information.
Cite sources when providing factual information.
Be careful about reliability of web sources.""",
        description="Web search and browsing",
    ),
    
    "file": PromptDefinition(
        module="file",
        base_prompt="""You are an AI assistant with file system access.
Help users manage, read, and organize files.
Never access or modify files without explicit permission.
Respect file system boundaries and permissions.""",
        description="File operations",
    ),
}

DEFAULT_SAFETY_PROMPTS: Dict[str, SafetyPrompt] = {
    "general": SafetyPrompt(
        name="general",
        content="""Always prioritize safety and user well-being.
Refuse requests that could cause harm.
Protect user privacy and data.""",
        always_include=True,
        priority=100,
    ),
    
    "code_safety": SafetyPrompt(
        name="code_safety",
        content="""Never generate malicious code, malware, or exploits.
Warn users about security implications of code.
Follow secure coding practices.""",
        modules={"code"},
        priority=90,
    ),
    
    "content_safety": SafetyPrompt(
        name="content_safety",
        content="""Do not generate harmful, illegal, or explicit content.
Respect content guidelines and policies.""",
        modules={"image", "video", "audio"},
        priority=90,
    ),
    
    "robot_safety": SafetyPrompt(
        name="robot_safety",
        content="""SAFETY FIRST: Always verify robot commands are safe.
Never execute movements that could cause injury or damage.
Stop immediately if any safety concern is detected.""",
        modules={"robot"},
        always_include=True,
        priority=100,
    ),
}


# ==============================================================================
# PROMPT MANAGER
# ==============================================================================

class PromptManager:
    """
    Centralized prompt management for all ForgeAI modules.
    
    Features:
    - Single source of truth for all prompts
    - Persona integration
    - Module-specific overrides
    - Safety prompt injection
    - Variable substitution
    - Prompt caching and validation
    """
    
    _instance: Optional["PromptManager"] = None
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize prompt manager.
        
        Args:
            config_path: Path to prompt configuration file
        """
        self.config_path = config_path or Path(CONFIG.get("prompts_dir", "data/prompts"))
        self.config_path = Path(self.config_path)
        
        # Load prompts
        self.prompts: Dict[str, PromptDefinition] = dict(DEFAULT_PROMPTS)
        self.overrides: List[PromptOverride] = []
        self.safety_prompts: Dict[str, SafetyPrompt] = dict(DEFAULT_SAFETY_PROMPTS)
        
        # Caches
        self._prompt_cache: Dict[str, str] = {}
        self._persona_cache: Dict[str, Any] = {}
        
        # Variable transformers
        self._transformers: Dict[str, Callable[[str], str]] = {}
        
        # Load custom prompts
        self._load_custom_prompts()
    
    @classmethod
    def get_instance(cls) -> "PromptManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load_custom_prompts(self):
        """Load custom prompts from config file."""
        prompts_file = self.config_path / "prompts.json"
        if prompts_file.exists():
            try:
                data = json.loads(prompts_file.read_text(encoding="utf-8"))
                
                # Load prompt definitions
                for name, prompt_data in data.get("prompts", {}).items():
                    self.prompts[name] = PromptDefinition.from_dict(prompt_data)
                
                # Load overrides
                for override_data in data.get("overrides", []):
                    self.overrides.append(PromptOverride(**override_data))
                
                # Load safety prompts
                for name, safety_data in data.get("safety", {}).items():
                    if "modules" in safety_data:
                        safety_data["modules"] = set(safety_data["modules"])
                    self.safety_prompts[name] = SafetyPrompt(**safety_data)
                
                logger.info(f"Loaded {len(self.prompts)} prompts from config")
            except Exception as e:
                logger.warning(f"Error loading custom prompts: {e}")
    
    def save_prompts(self):
        """Save prompts to config file."""
        self.config_path.mkdir(parents=True, exist_ok=True)
        prompts_file = self.config_path / "prompts.json"
        
        data = {
            "prompts": {
                name: prompt.to_dict() 
                for name, prompt in self.prompts.items()
            },
            "overrides": [
                {
                    "module": o.module,
                    "context": o.context,
                    "override_prompt": o.override_prompt,
                    "merge_mode": o.merge_mode,
                    "priority": o.priority
                }
                for o in self.overrides
            ],
            "safety": {
                name: {
                    "name": s.name,
                    "content": s.content,
                    "modules": list(s.modules),
                    "always_include": s.always_include,
                    "priority": s.priority
                }
                for name, s in self.safety_prompts.items()
            }
        }
        
        prompts_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"Saved prompts to {prompts_file}")
    
    def get_system_prompt(
        self,
        module: str,
        persona_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        include_safety: bool = True,
        variables: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Get the complete system prompt for a module.
        
        Args:
            module: Module name (e.g., "chat", "code", "image")
            persona_id: Optional persona to apply
            context: Additional context for overrides
            include_safety: Whether to include safety prompts
            variables: Variable substitutions
            
        Returns:
            Complete system prompt
        """
        # Check cache
        cache_key = f"{module}:{persona_id}:{include_safety}"
        if cache_key in self._prompt_cache and not variables:
            return self._prompt_cache[cache_key]
        
        # Get base prompt
        prompt_def = self.prompts.get(module)
        if not prompt_def:
            logger.warning(f"No prompt defined for module '{module}', using default")
            prompt_def = PromptDefinition(
                module=module,
                base_prompt=DEFAULT_BASE_PROMPT
            )
        
        parts = []
        
        # Prepend persona prompt if enabled
        if prompt_def.persona_prepend and persona_id:
            persona_prompt = self._get_persona_prompt(persona_id)
            if persona_prompt:
                parts.append(persona_prompt)
        
        # Add base prompt
        base_prompt = prompt_def.base_prompt
        
        # Apply overrides
        base_prompt = self._apply_overrides(module, base_prompt, persona_id, context)
        
        # Substitute variables
        if variables or prompt_def.variables:
            all_vars = {**prompt_def.variables, **(variables or {})}
            base_prompt = self._substitute_variables(base_prompt, all_vars)
        
        parts.append(base_prompt)
        
        # Append safety prompts if enabled
        if include_safety and prompt_def.safety_append:
            safety = self._get_safety_prompts(module)
            if safety:
                parts.append(safety)
        
        # Combine
        prompt = "\n\n".join(parts)
        
        # Cache if no variables
        if not variables:
            self._prompt_cache[cache_key] = prompt
        
        return prompt
    
    def _get_persona_prompt(self, persona_id: str) -> Optional[str]:
        """Get prompt text from a persona."""
        # Try to load from persona system
        try:
            from .persona import PersonaManager
            manager = PersonaManager()
            persona = manager.load_persona(persona_id)
            if persona and persona.system_prompt:
                return persona.system_prompt
        except Exception as e:
            logger.debug(f"Could not load persona '{persona_id}': {e}")
        
        return None
    
    def _apply_overrides(
        self,
        module: str,
        prompt: str,
        persona_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Apply relevant overrides to a prompt."""
        context = context or {}
        contexts_to_check = [f"module:{module}"]
        
        if persona_id:
            contexts_to_check.append(f"persona:{persona_id}")
        
        for key, value in context.items():
            contexts_to_check.append(f"{key}:{value}")
        
        # Find matching overrides
        matching = [
            o for o in self.overrides
            if o.module == module and o.context in contexts_to_check
        ]
        
        # Sort by priority
        matching.sort(key=lambda x: x.priority)
        
        # Apply overrides
        for override in matching:
            if override.merge_mode == "replace":
                prompt = override.override_prompt
            elif override.merge_mode == "prepend":
                prompt = override.override_prompt + "\n\n" + prompt
            elif override.merge_mode == "append":
                prompt = prompt + "\n\n" + override.override_prompt
        
        return prompt
    
    def _get_safety_prompts(self, module: str) -> str:
        """Get relevant safety prompts for a module."""
        relevant = []
        
        for safety in self.safety_prompts.values():
            # Check if applies to this module
            if safety.always_include or not safety.modules or module in safety.modules:
                relevant.append((safety.priority, safety.content))
        
        # Sort by priority
        relevant.sort(key=lambda x: x[0], reverse=True)
        
        return "\n".join(content for _, content in relevant)
    
    def _substitute_variables(self, prompt: str, variables: Dict[str, str]) -> str:
        """Substitute variables in prompt."""
        for name, value in variables.items():
            # Apply transformer if exists
            if name in self._transformers:
                value = self._transformers[name](value)
            
            # Replace {variable} format
            prompt = prompt.replace(f"{{{name}}}", str(value))
        
        return prompt
    
    def register_prompt(
        self,
        module: str,
        prompt: str,
        description: str = "",
        **kwargs
    ):
        """
        Register a new prompt definition.
        
        Args:
            module: Module name
            prompt: Prompt text
            description: Prompt description
            **kwargs: Additional PromptDefinition fields
        """
        self.prompts[module] = PromptDefinition(
            module=module,
            base_prompt=prompt,
            description=description,
            **kwargs
        )
        
        # Clear cache for this module
        self._clear_cache(module)
        
        logger.info(f"Registered prompt for module '{module}'")
    
    def add_override(
        self,
        module: str,
        context: str,
        override_prompt: str,
        merge_mode: str = "replace",
        priority: int = 0
    ):
        """
        Add a prompt override.
        
        Args:
            module: Module to override
            context: Context string (e.g., "persona:my_ai")
            override_prompt: Override prompt text
            merge_mode: "replace", "prepend", or "append"
            priority: Higher = applied later
        """
        self.overrides.append(PromptOverride(
            module=module,
            context=context,
            override_prompt=override_prompt,
            merge_mode=merge_mode,
            priority=priority
        ))
        
        self._clear_cache(module)
    
    def add_safety_prompt(
        self,
        name: str,
        content: str,
        modules: Optional[Set[str]] = None,
        always_include: bool = False
    ):
        """
        Add a safety prompt.
        
        Args:
            name: Safety prompt name
            content: Safety prompt content
            modules: Set of modules to apply to (empty = all)
            always_include: Whether to always include
        """
        self.safety_prompts[name] = SafetyPrompt(
            name=name,
            content=content,
            modules=modules or set(),
            always_include=always_include
        )
        
        self._prompt_cache.clear()
    
    def register_transformer(self, variable: str, transformer: Callable[[str], str]):
        """
        Register a variable transformer.
        
        Args:
            variable: Variable name
            transformer: Function to transform the value
        """
        self._transformers[variable] = transformer
    
    def _clear_cache(self, module: Optional[str] = None):
        """Clear prompt cache."""
        if module:
            keys_to_remove = [k for k in self._prompt_cache if k.startswith(f"{module}:")]
            for key in keys_to_remove:
                del self._prompt_cache[key]
        else:
            self._prompt_cache.clear()
    
    def list_modules(self) -> List[str]:
        """List all registered modules."""
        return list(self.prompts.keys())
    
    def get_prompt_definition(self, module: str) -> Optional[PromptDefinition]:
        """Get raw prompt definition for a module."""
        return self.prompts.get(module)
    
    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Validate a prompt.
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        # Check length
        if len(prompt) > 4000:
            warnings.append(f"Prompt is long ({len(prompt)} chars). May exceed token limits.")
        
        if len(prompt) < 10:
            issues.append("Prompt is too short.")
        
        # Check for unsubstituted variables
        import re
        unsubstituted = re.findall(r'\{[a-zA-Z_]+\}', prompt)
        if unsubstituted:
            warnings.append(f"Found unsubstituted variables: {unsubstituted}")
        
        # Check for conflicting instructions
        if "always" in prompt.lower() and "never" in prompt.lower():
            warnings.append("Prompt contains both 'always' and 'never' - may cause conflicts.")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "length": len(prompt),
            "estimated_tokens": len(prompt) // 4  # Rough estimate
        }


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get the global PromptManager instance."""
    global _manager
    if _manager is None:
        _manager = PromptManager.get_instance()
    return _manager


def get_system_prompt(
    module: str,
    persona_id: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convenience function to get a system prompt.
    
    Args:
        module: Module name
        persona_id: Optional persona ID
        **kwargs: Additional arguments for get_system_prompt
        
    Returns:
        System prompt string
    """
    return get_prompt_manager().get_system_prompt(module, persona_id, **kwargs)
