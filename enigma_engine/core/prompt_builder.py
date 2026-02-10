"""
================================================================================
PROMPT BUILDER - CENTRALIZED PROMPT CONSTRUCTION
================================================================================

All prompt construction goes through here to ensure consistency.
Configurable templates allow customization without code changes.

FILE: enigma_engine/core/prompt_builder.py
TYPE: Prompt Construction Utility
MAIN CLASS: PromptBuilder

USAGE:
    from enigma_engine.core.prompt_builder import PromptBuilder, get_prompt_builder
    
    builder = get_prompt_builder()
    
    # Build a chat prompt
    prompt = builder.build_chat_prompt(
        message="Hello!",
        history=[{"role": "user", "content": "Hi"}],
        system_prompt="You are helpful."
    )
    
    # Extract AI response from generated text
    response = builder.extract_response(full_output)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..config import CONFIG

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT TEMPLATES - Configurable formats
# =============================================================================

@dataclass
class PromptTemplate:
    """Template for formatting prompts."""
    
    # Role markers
    system_prefix: str = "System: "
    user_prefix: str = "User: "
    assistant_prefix: str = "Assistant: "
    
    # Separators
    message_separator: str = "\n"
    turn_separator: str = "\n"
    
    # Stop sequences (what marks end of AI turn)
    stop_sequences: list[str] = field(default_factory=lambda: [
        "\nUser:", "\n\n", "User:", "\nHuman:"
    ])
    
    # Whether to add assistant prefix at end (for generation)
    add_generation_prefix: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "system_prefix": self.system_prefix,
            "user_prefix": self.user_prefix,
            "assistant_prefix": self.assistant_prefix,
            "message_separator": self.message_separator,
            "turn_separator": self.turn_separator,
            "stop_sequences": self.stop_sequences,
            "add_generation_prefix": self.add_generation_prefix,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Preset templates for different use cases
PROMPT_TEMPLATES = {
    "default": PromptTemplate(),
    
    "chatml": PromptTemplate(
        system_prefix="<|im_start|>system\n",
        user_prefix="<|im_start|>user\n",
        assistant_prefix="<|im_start|>assistant\n",
        message_separator="<|im_end|>\n",
        turn_separator="",
        stop_sequences=["<|im_end|>", "<|im_start|>user"],
    ),
    
    "llama": PromptTemplate(
        system_prefix="[INST] <<SYS>>\n",
        user_prefix="[INST] ",
        assistant_prefix="[/INST] ",
        message_separator="\n<</SYS>>\n\n" if "system" else " [/INST]\n",
        turn_separator="</s><s>",
        stop_sequences=["</s>", "[INST]"],
    ),
    
    "simple": PromptTemplate(
        system_prefix="",
        user_prefix="Q: ",
        assistant_prefix="A: ",
        message_separator="\n",
        turn_separator="\n\n",
        stop_sequences=["\nQ:", "\n\n"],
    ),
}


# =============================================================================
# PROMPT BUILDER CLASS
# =============================================================================

class PromptBuilder:
    """
    Centralized prompt construction for consistent formatting.
    
    Supports:
    - Multiple template formats (default, chatml, llama, simple)
    - Custom templates loaded from config
    - System prompts with personality integration
    - Conversation history formatting
    - Response extraction from generated text
    """
    
    def __init__(self, template_name: str = "default"):
        """
        Initialize prompt builder.
        
        Args:
            template_name: Name of template to use (default, chatml, llama, simple)
        """
        self._template_name = template_name
        self._template = self._load_template(template_name)
        self._custom_system_prompt: Optional[str] = None
    
    def _load_template(self, name: str) -> PromptTemplate:
        """Load template by name, checking user config first."""
        # Check for user-defined template
        user_templates_path = Path(CONFIG.get("data_dir", "data")) / "prompt_templates.json"
        if user_templates_path.exists():
            try:
                with open(user_templates_path, encoding='utf-8') as f:
                    user_templates = json.load(f)
                if name in user_templates:
                    return PromptTemplate.from_dict(user_templates[name])
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load user templates: {e}")
        
        # Use built-in template
        if name in PROMPT_TEMPLATES:
            return PROMPT_TEMPLATES[name]
        
        logger.warning(f"Unknown template '{name}', using default")
        return PROMPT_TEMPLATES["default"]
    
    @property
    def template(self) -> PromptTemplate:
        """Get current template."""
        return self._template
    
    def set_template(self, name: str) -> None:
        """Change the active template."""
        self._template_name = name
        self._template = self._load_template(name)
    
    def set_system_prompt(self, prompt: Optional[str]) -> None:
        """Set a custom system prompt to use in all prompts."""
        self._custom_system_prompt = prompt
    
    def build_chat_prompt(
        self,
        message: str,
        history: Optional[list[dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        personality_prompt: Optional[str] = None,
        include_generation_prefix: bool = True
    ) -> str:
        """
        Build a complete chat prompt from components.
        
        Args:
            message: Current user message
            history: Previous messages [{"role": "user/assistant", "content": "..."}]
            system_prompt: System instruction (overrides custom if set)
            personality_prompt: Optional personality prompt to append to system
            include_generation_prefix: Add assistant prefix at end for generation
        
        Returns:
            Formatted prompt string ready for model input
        """
        t = self._template
        parts = []
        
        # Determine effective system prompt
        effective_system = system_prompt or self._custom_system_prompt
        
        # Combine system and personality prompts
        if effective_system and personality_prompt:
            effective_system = f"{effective_system}\n\n{personality_prompt}"
        elif personality_prompt:
            effective_system = personality_prompt
        
        # Add system prompt
        if effective_system:
            parts.append(f"{t.system_prefix}{effective_system}{t.message_separator}")
        
        # Add history
        if history:
            for msg in history:
                role = msg.get("role", "user").lower()
                content = msg.get("content", msg.get("text", ""))
                
                if role in ("user", "human"):
                    parts.append(f"{t.user_prefix}{content}{t.message_separator}")
                elif role in ("assistant", "ai", "bot"):
                    parts.append(f"{t.assistant_prefix}{content}{t.message_separator}")
                elif role == "system":
                    # Additional system messages in history
                    parts.append(f"{t.system_prefix}{content}{t.message_separator}")
        
        # Add current message
        parts.append(f"{t.user_prefix}{message}{t.message_separator}")
        
        # Add generation prefix
        if include_generation_prefix and t.add_generation_prefix:
            parts.append(t.assistant_prefix)
        
        return t.turn_separator.join(parts) if t.turn_separator else "".join(parts)
    
    def build_completion_prompt(
        self,
        text: str,
        instruction: Optional[str] = None
    ) -> str:
        """
        Build a simple completion prompt.
        
        Args:
            text: Text to complete
            instruction: Optional instruction prefix
        
        Returns:
            Formatted prompt
        """
        if instruction:
            return f"{self._template.system_prefix}{instruction}\n\n{text}"
        return text
    
    def get_stop_sequences(self) -> list[str]:
        """Get stop sequences for current template."""
        return self._template.stop_sequences.copy()
    
    def extract_response(self, full_output: str, original_prompt: str = "") -> str:
        """
        Extract the AI response from generated output.
        
        Args:
            full_output: Complete generated text (may include prompt)
            original_prompt: Original prompt (to remove if present)
        
        Returns:
            Cleaned AI response
        """
        response = full_output
        
        # Remove original prompt if present
        if original_prompt and response.startswith(original_prompt):
            response = response[len(original_prompt):]
        
        # Remove assistant prefix if present at start
        assistant_prefix = self._template.assistant_prefix.strip()
        if assistant_prefix and response.strip().startswith(assistant_prefix):
            response = response.strip()[len(assistant_prefix):]
        
        # Split at any assistant prefix and take last part
        if assistant_prefix in response:
            response = response.split(assistant_prefix)[-1]
        
        # Remove any trailing stop sequences
        for stop in self._template.stop_sequences:
            if stop in response:
                response = response.split(stop)[0]
        
        return response.strip()
    
    def format_training_example(
        self,
        user_input: str,
        ai_response: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format a training example in the current template style.
        
        Args:
            user_input: User's message
            ai_response: AI's response
            system_prompt: Optional system context
        
        Returns:
            Formatted training example
        """
        t = self._template
        parts = []
        
        if system_prompt:
            parts.append(f"{t.system_prefix}{system_prompt}{t.message_separator}")
        
        parts.append(f"{t.user_prefix}{user_input}{t.message_separator}")
        parts.append(f"{t.assistant_prefix}{ai_response}{t.message_separator}")
        
        return "".join(parts)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_prompt_builder: Optional[PromptBuilder] = None


def get_prompt_builder(template_name: str = "default") -> PromptBuilder:
    """
    Get or create the global prompt builder instance.
    
    Args:
        template_name: Template to use (only used on first call)
    
    Returns:
        Shared PromptBuilder instance
    """
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder(template_name)
    return _prompt_builder


def build_chat_prompt(
    message: str,
    history: Optional[list[dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    **kwargs
) -> str:
    """Convenience function to build a chat prompt."""
    return get_prompt_builder().build_chat_prompt(
        message=message,
        history=history,
        system_prompt=system_prompt,
        **kwargs
    )


def extract_response(full_output: str, original_prompt: str = "") -> str:
    """Convenience function to extract AI response."""
    return get_prompt_builder().extract_response(full_output, original_prompt)
