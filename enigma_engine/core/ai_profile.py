"""
AI Profile System for Enigma Engine

Each AI model can have its own profile file (ai_profile.json) that defines:
- Model path and settings
- System prompt and personality
- Chat template
- Generation parameters
- Memory settings
- Available commands

This allows hot-swapping AI profiles like we do with mods.

Usage:
    from enigma_engine.core.ai_profile import AIProfile, load_profile, save_profile

    # Load an AI profile
    profile = load_profile("profiles/assistant.json")

    # Apply to engine/model
    engine.apply_profile(profile)

    # Save a profile
    save_profile(profile, "profiles/my_ai.json")
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# PROFILE DATACLASS
# =============================================================================

@dataclass
class GenerationConfig:
    """Generation parameters for the AI."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = field(default_factory=lambda: ["</s>", "<|endoftext|>"])


@dataclass
class MemoryConfig:
    """Memory and context settings."""
    max_history_messages: int = 20
    max_context_tokens: int = 4096
    save_conversations: bool = True
    conversation_dir: str = "memory/conversations"
    # Long-term memory (knowledge base)
    enable_knowledge_base: bool = False
    knowledge_base_path: str = "memory/knowledge"


@dataclass
class AIProfile:
    """
    Complete AI profile definition.

    Similar to mod.json but for AI models. Defines everything
    needed to load and configure an AI personality.

    Attributes:
        name: Display name for the AI (e.g., "Assistant", "Coder", "Creative")
        id: Unique identifier (e.g., "assistant", "coding_helper")
        version: Profile version for compatibility
        description: What this AI is for

        model_path: Path to the model file (.gguf, .pth, or HuggingFace repo)
        model_type: Type of model ("gguf", "pytorch", "huggingface", "ollama")

        system_prompt: Default system prompt for this AI
        personality: Personality traits (for consistency)

        chat_template: How to format messages (None = auto-detect)
        generation: Generation parameters
        memory: Memory and context settings

        commands: List of commands this AI can use (empty = all)
        disabled_commands: Commands this AI cannot use

        author: Who created this profile
        tags: Tags for categorization
    """
    # Identity
    name: str = "Enigma Assistant"
    id: str = "default"
    version: str = "1.0"
    description: str = "A helpful AI assistant"

    # Model
    model_path: str = ""
    model_type: str = "auto"  # auto, gguf, pytorch, huggingface, ollama

    # Personality
    system_prompt: str = "You are a helpful AI assistant."
    personality: Dict[str, Any] = field(default_factory=lambda: {
        "tone": "helpful",
        "verbosity": "balanced",
        "formality": "casual",
        "humor": "occasional",
    })

    # Chat
    chat_template: Optional[str] = None  # None = auto-detect from model

    # Generation
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # Memory
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Commands
    commands: List[str] = field(default_factory=list)  # Empty = all allowed
    disabled_commands: List[str] = field(default_factory=list)

    # Metadata
    author: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for saving."""
        # asdict() already recursively converts nested dataclasses
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AIProfile':
        """Create profile from dictionary."""
        # Work on a copy so we don't mutate the caller's dict
        data = dict(data)
        # Handle nested configs — filter unknown keys to prevent crashes
        if 'generation' in data and isinstance(data['generation'], dict):
            gen_fields = {f.name for f in GenerationConfig.__dataclass_fields__.values()}
            gen_data = {k: v for k, v in data['generation'].items() if k in gen_fields}
            data['generation'] = GenerationConfig(**gen_data)
        if 'memory' in data and isinstance(data['memory'], dict):
            mem_fields = {f.name for f in MemoryConfig.__dataclass_fields__.values()}
            mem_data = {k: v for k, v in data['memory'].items() if k in mem_fields}
            data['memory'] = MemoryConfig(**mem_data)

        # Filter out unknown fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - valid_fields - {'generation', 'memory'}
        if unknown:
            logger.debug("AIProfile.from_dict: ignoring unknown fields: %s", unknown)
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)

    def can_use_command(self, command: str) -> bool:
        """Check if this AI can use a specific command."""
        # Check disabled first
        if command in self.disabled_commands:
            return False

        # If commands list is empty, allow all
        if not self.commands:
            return True

        # Check if in allowed list
        return command in self.commands


# =============================================================================
# PROFILE MANAGEMENT
# =============================================================================

def load_profile(path: str) -> AIProfile:
    """
    Load an AI profile from a JSON file.

    Args:
        path: Path to ai_profile.json

    Returns:
        AIProfile instance

    Example:
        profile = load_profile("profiles/coding_assistant.json")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        profile = AIProfile.from_dict(data)
        logger.info(f"Loaded AI profile: {profile.name} ({profile.id})")
        return profile

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in profile: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load profile: {e}") from e


def save_profile(profile: AIProfile, path: str) -> Path:
    """
    Save an AI profile to a JSON file.

    Args:
        profile: AIProfile instance
        path: Output path

    Returns:
        Path where profile was saved

    Example:
        save_profile(my_profile, "profiles/my_ai.json")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = profile.to_dict()

    from enigma_engine.core.safe_save import atomic_write_json
    atomic_write_json(path, data)

    logger.info(f"Saved AI profile to: {path}")
    return path


def list_profiles(profiles_dir: str = "profiles") -> List[Dict[str, str]]:
    """
    List all available AI profiles.

    Args:
        profiles_dir: Directory to search

    Returns:
        List of dicts with profile info (name, id, path, description, conversation_dir)

    Example:
        profiles = list_profiles()
        for p in profiles:
            print(f"{p['name']}: {p['description']}")
    """
    profiles_path = Path(profiles_dir)

    if not profiles_path.exists():
        return []

    results = []

    for profile_file in profiles_path.glob("*.json"):
        try:
            with open(profile_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            profile_id = data.get("id", profile_file.stem)

            # Get conversation_dir from memory config, default to profiles/<id>/conversations
            memory_config = data.get("memory", {})
            default_conv_dir = f"profiles/{profile_id}/conversations"
            conversation_dir = memory_config.get("conversation_dir", default_conv_dir)

            results.append({
                "name": data.get("name", profile_file.stem),
                "id": profile_id,
                "path": str(profile_file),
                "description": data.get("description", ""),
                "model_path": data.get("model_path", ""),
                "tags": data.get("tags", []),
                "conversation_dir": conversation_dir,
            })
        except Exception as e:
            logger.debug(f"Could not read profile {profile_file}: {e}")

    return results


# =============================================================================
# PROFILE MANAGER (Hot-swap support)
# =============================================================================

class AIProfileManager:
    """
    Manager for AI profiles with hot-swap support.

    Like the mod system, allows loading/unloading AI profiles
    at runtime without restarting the application.

    Example:
        manager = AIProfileManager()
        manager.load_profile("profiles/assistant.json")

        # Switch to different AI
        manager.switch_profile("coding_helper")

        # List available
        for p in manager.list_profiles():
            print(p['name'])
    """

    def __init__(self, profiles_dir: str = "profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        self._loaded_profiles: Dict[str, AIProfile] = {}
        self._active_profile: Optional[str] = None

        # Callbacks for hot-swap
        self.on_profile_loaded: Optional[Callable] = None
        self.on_profile_switched: Optional[Callable] = None

    @property
    def active_profile(self) -> Optional[AIProfile]:
        """Get the currently active profile."""
        if self._active_profile and self._active_profile in self._loaded_profiles:
            return self._loaded_profiles[self._active_profile]
        return None

    def load_profile(self, path_or_id: str) -> AIProfile:
        """
        Load a profile from file or by ID.

        Args:
            path_or_id: File path or profile ID

        Returns:
            Loaded AIProfile
        """
        # Check if it's a path
        if Path(path_or_id).exists():
            profile = load_profile(path_or_id)
        else:
            # Try to find by ID in profiles dir
            profile_path = self.profiles_dir / f"{path_or_id}.json"
            if not profile_path.exists():
                raise FileNotFoundError(f"Profile not found: {path_or_id}")
            profile = load_profile(str(profile_path))

        # Store in loaded profiles
        self._loaded_profiles[profile.id] = profile

        # Callback
        if self.on_profile_loaded:
            self.on_profile_loaded(profile)

        logger.info(f"Loaded profile: {profile.name} ({profile.id})")
        return profile

    def switch_profile(self, profile_id: str) -> AIProfile:
        """
        Switch to a different profile.

        Args:
            profile_id: ID of profile to switch to

        Returns:
            The new active profile
        """
        # Load if not already loaded
        if profile_id not in self._loaded_profiles:
            self.load_profile(profile_id)

        old_profile = self._active_profile
        self._active_profile = profile_id
        profile = self._loaded_profiles[profile_id]

        # Callback
        if self.on_profile_switched:
            self.on_profile_switched(old_profile, profile)

        logger.info(f"Switched from {old_profile} to {profile.name}")
        return profile

    def unload_profile(self, profile_id: str) -> None:
        """Unload a profile from memory."""
        if profile_id in self._loaded_profiles:
            del self._loaded_profiles[profile_id]

            if self._active_profile == profile_id:
                self._active_profile = None

            logger.info(f"Unloaded profile: {profile_id}")

    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all available profiles."""
        return list_profiles(str(self.profiles_dir))

    def list_loaded(self) -> List[str]:
        """List IDs of currently loaded profiles."""
        return list(self._loaded_profiles.keys())

    def create_profile(
        self,
        name: str,
        model_path: str,
        system_prompt: str = "You are a helpful AI assistant.",
        **kwargs
    ) -> AIProfile:
        """
        Create and save a new profile.

        Args:
            name: Display name
            model_path: Path to model
            system_prompt: System prompt
            **kwargs: Additional profile fields

        Returns:
            Created profile
        """
        # Generate ID from name
        profile_id = name.lower().replace(" ", "_").replace("-", "_")

        profile = AIProfile(
            name=name,
            id=profile_id,
            model_path=model_path,
            system_prompt=system_prompt,
            **kwargs
        )

        # Save to profiles dir
        save_path = self.profiles_dir / f"{profile_id}.json"
        save_profile(profile, str(save_path))

        return profile


# =============================================================================
# DEFAULT PROFILES
# =============================================================================

DEFAULT_PROFILES = {
    "assistant": AIProfile(
        name="Enigma Assistant",
        id="assistant",
        description="A helpful general-purpose AI assistant",
        system_prompt="""You are Enigma, a helpful AI assistant running locally on the user's computer.
You can execute commands using [CMD]command[/CMD] blocks.
Be concise, helpful, and friendly.""",
        personality={
            "tone": "helpful",
            "verbosity": "balanced",
            "formality": "casual",
        },
        tags=["general", "assistant"],
    ),

    "coding_helper": AIProfile(
        name="Code Assistant",
        id="coding_helper",
        description="Specialized for programming and code review",
        system_prompt="""You are a coding assistant. Help users write, debug, and understand code.
Use [CMD]file.read path[/CMD] to read files.
Use [CMD]file.write path content[/CMD] to save code.
Be precise and technical. Show code examples.""",
        personality={
            "tone": "technical",
            "verbosity": "detailed",
            "formality": "professional",
        },
        generation=GenerationConfig(
            temperature=0.3,  # Lower for more precise code
            top_p=0.95,
        ),
        tags=["coding", "developer", "technical"],
    ),

    "creative_writer": AIProfile(
        name="Creative Writer",
        id="creative_writer",
        description="For creative writing and storytelling",
        system_prompt="""You are a creative writing assistant. Help users with stories,
poems, scripts, and creative content. Be imaginative and expressive.""",
        personality={
            "tone": "creative",
            "verbosity": "expressive",
            "formality": "casual",
            "humor": "witty",
        },
        generation=GenerationConfig(
            temperature=0.9,  # Higher for creativity
            top_p=0.95,
            repetition_penalty=1.2,
        ),
        tags=["creative", "writing", "stories"],
    ),

    "researcher": AIProfile(
        name="Research Assistant",
        id="researcher",
        description="For research, analysis, and fact-finding",
        system_prompt="""You are a research assistant. Help users find information,
analyze data, and understand complex topics.
Use [CMD]search.web query[/CMD] to search the internet.
Use [CMD]web.fetch url[/CMD] to read web pages.
Always cite sources and be factual.""",
        personality={
            "tone": "analytical",
            "verbosity": "detailed",
            "formality": "academic",
        },
        generation=GenerationConfig(
            temperature=0.5,
            top_p=0.9,
        ),
        commands=["search.web", "web.fetch", "file.read", "note.add"],
        tags=["research", "academic", "analysis"],
    ),
}


def create_default_profiles(profiles_dir: str = "profiles") -> None:
    """
    Create default profile files in the profiles directory.

    Args:
        profiles_dir: Where to save profiles
    """
    profiles_path = Path(profiles_dir)
    profiles_path.mkdir(parents=True, exist_ok=True)

    for profile_id, profile in DEFAULT_PROFILES.items():
        profile_file = profiles_path / f"{profile_id}.json"

        # Don't overwrite existing profiles
        if not profile_file.exists():
            save_profile(profile, str(profile_file))
            logger.info(f"Created default profile: {profile_id}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_profile_for_model(model_path: str, profiles_dir: str = "profiles") -> Optional[AIProfile]:
    """
    Find a profile that uses a specific model.

    Args:
        model_path: Path to model file
        profiles_dir: Where to search

    Returns:
        Matching profile or None
    """
    model_path = Path(model_path).resolve()

    for profile_info in list_profiles(profiles_dir):
        profile_model = Path(profile_info.get("model_path", ""))
        if profile_model.resolve() == model_path:
            return load_profile(profile_info["path"])

    return None


def apply_profile_to_engine(profile: AIProfile, engine) -> None:
    """
    Apply an AI profile's settings to an engine.

    Args:
        profile: AIProfile to apply
        engine: EnigmaEngine instance
    """
    # Apply generation settings
    if hasattr(engine, 'temperature'):
        engine.temperature = profile.generation.temperature
    if hasattr(engine, 'top_p'):
        engine.top_p = profile.generation.top_p
    if hasattr(engine, 'top_k'):
        engine.top_k = profile.generation.top_k
    if hasattr(engine, 'max_tokens'):
        engine.max_tokens = profile.generation.max_tokens

    # Apply system prompt
    if hasattr(engine, 'system_prompt'):
        engine.system_prompt = profile.system_prompt

    logger.info(f"Applied profile '{profile.name}' to engine")
