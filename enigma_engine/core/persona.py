"""
AI Persona System - Complete AI Identity Management

This module allows users to:
- Create and customize their own AI personas
- Copy/paste personas to create variants
- Export/import personas as shareable files
- Merge traits from multiple personas

A persona is NOT a "character preset" - it's the full state of YOUR AI
that can be copied, exported, and shared.

Usage:
    from enigma_engine.core.persona import PersonaManager, AIPersona
    
    manager = PersonaManager()
    
    # Get current persona
    persona = manager.get_current_persona()
    
    # Copy to create variant
    new_persona = manager.copy_persona(persona.id, "My Assistant")
    
    # Export to share
    manager.export_persona(persona.id, Path("my_ai.forge-ai"))
    
    # Import from file
    imported = manager.import_persona(Path("shared_ai.forge-ai"))
"""

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..config import CONFIG
from .personality import AIPersonality, PersonalityTraits


@dataclass
class AIPersona:
    """
    Complete AI identity - everything that makes this AI unique.
    
    This is NOT a "character preset" - it's the full state of YOUR AI
    that can be copied, exported, and shared.
    """
    # Identity
    id: str                              # Unique identifier
    name: str                            # Display name (user chooses)
    created_at: str                      # When this AI was born
    
    # Personality (links to existing personality.py)
    personality_traits: dict[str, float] # humor, formality, etc.
    
    # Voice (links to existing voice system)
    voice_profile_id: str = "default"   # Which voice config to use
    
    # Avatar (links to existing avatar system)
    avatar_preset_id: str = "default"   # Which avatar config to use
    
    # Behavior
    system_prompt: str = ""             # Base instructions
    response_style: str = "balanced"    # "concise", "detailed", "casual", etc.
    
    # Knowledge
    knowledge_domains: list[str] = field(default_factory=list)  # What topics this AI knows about
    memories: list[str] = field(default_factory=list)          # Important things to remember
    
    # Learning state
    learning_data_path: str = ""        # Path to this AI's training data
    model_weights_path: str = ""        # Path to fine-tuned weights (if any)
    
    # Customization
    catchphrases: list[str] = field(default_factory=list)  # Signature phrases
    preferences: dict[str, Any] = field(default_factory=dict)  # User-defined preferences
    
    # Metadata
    version: str = "1.0"
    last_modified: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize defaults after creation."""
        if not self.last_modified:
            self.last_modified = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'AIPersona':
        """Create from dictionary."""
        return cls(**data)
    
    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.last_modified = datetime.now().isoformat()


class PersonaManager:
    """
    Manage AI personas - create, copy, export, import.
    
    Storage structure:
        data/personas/
        ├── default/
        │   ├── persona.json      # Main config
        │   ├── personality.json  # Trait values
        │   ├── memories.json     # Important memories
        │   └── learning/         # Training data for this persona
        ├── my_assistant/
        │   └── ...
        └── templates/
            └── helpful_assistant.forge-ai
    """
    
    def __init__(self, personas_dir: Optional[Path] = None):
        """
        Initialize persona manager.
        
        Args:
            personas_dir: Directory to store personas (defaults to data/personas)
        """
        if personas_dir is None:
            data_dir = Path(CONFIG.get("data_dir", "data"))
            personas_dir = data_dir / "personas"
        
        self.personas_dir = Path(personas_dir)
        self.personas_dir.mkdir(parents=True, exist_ok=True)
        
        self.templates_dir = self.personas_dir / "templates"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_persona_id: Optional[str] = None
        self._personas_cache: dict[str, AIPersona] = {}
        
        # Load or create default persona
        self._ensure_default_persona()
    
    def _ensure_default_persona(self):
        """Ensure a default persona exists."""
        default_dir = self.personas_dir / "default"
        if not default_dir.exists():
            # Create default persona
            default_persona = AIPersona(
                id="default",
                name="Forge Assistant",
                created_at=datetime.now().isoformat(),
                personality_traits=PersonalityTraits().to_dict(),
                system_prompt="You are a helpful AI assistant built with Enigma AI Engine. You are knowledgeable, friendly, and adaptable to user needs.",
                response_style="balanced",
                description="Default Enigma AI Engine assistant persona",
                tags=["default", "assistant"]
            )
            self.save_persona(default_persona)
            self.current_persona_id = "default"
        else:
            # Load default persona
            self.current_persona_id = "default"
    
    def get_current_persona(self) -> AIPersona:
        """
        Get the currently active AI persona.
        
        Returns:
            Currently active persona
        """
        if self.current_persona_id is None:
            self._ensure_default_persona()
        
        if self.current_persona_id in self._personas_cache:
            return self._personas_cache[self.current_persona_id]
        
        persona = self.load_persona(self.current_persona_id)
        if persona is None:
            # Fallback to default
            self.current_persona_id = "default"
            persona = self.load_persona("default")
        
        self._personas_cache[self.current_persona_id] = persona
        return persona
    
    def set_current_persona(self, persona_id: str) -> bool:
        """
        Set the currently active persona.
        
        Args:
            persona_id: ID of persona to activate
            
        Returns:
            True if successful, False otherwise
        """
        if self.persona_exists(persona_id):
            self.current_persona_id = persona_id
            return True
        return False
    
    def persona_exists(self, persona_id: str) -> bool:
        """Check if a persona exists."""
        persona_dir = self.personas_dir / persona_id
        persona_file = persona_dir / "persona.json"
        return persona_file.exists()
    
    def list_personas(self) -> list[dict[str, str]]:
        """
        List all available personas.
        
        Returns:
            List of persona info dicts with 'id', 'name', 'description'
        """
        personas = []
        for persona_dir in self.personas_dir.iterdir():
            if persona_dir.is_dir() and persona_dir.name != "templates":
                persona_file = persona_dir / "persona.json"
                if persona_file.exists():
                    try:
                        with open(persona_file) as f:
                            data = json.load(f)
                            personas.append({
                                'id': data.get('id', persona_dir.name),
                                'name': data.get('name', persona_dir.name),
                                'description': data.get('description', ''),
                                'created_at': data.get('created_at', ''),
                            })
                    except Exception:
                        pass
        return personas
    
    def load_persona(self, persona_id: str) -> Optional[AIPersona]:
        """
        Load a persona from disk.
        
        Args:
            persona_id: ID of persona to load
            
        Returns:
            Loaded persona or None if not found
        """
        persona_dir = self.personas_dir / persona_id
        persona_file = persona_dir / "persona.json"
        
        if not persona_file.exists():
            return None
        
        try:
            with open(persona_file) as f:
                data = json.load(f)
            return AIPersona.from_dict(data)
        except Exception as e:
            print(f"Error loading persona {persona_id}: {e}")
            return None
    
    def save_persona(self, persona: AIPersona):
        """
        Save persona to disk.
        
        Args:
            persona: Persona to save
        """
        persona.update_timestamp()
        
        persona_dir = self.personas_dir / persona.id
        persona_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main persona file
        persona_file = persona_dir / "persona.json"
        with open(persona_file, 'w') as f:
            json.dump(persona.to_dict(), f, indent=2)
        
        # Create learning directory
        learning_dir = persona_dir / "learning"
        learning_dir.mkdir(exist_ok=True)
        if not persona.learning_data_path:
            persona.learning_data_path = str(learning_dir)
        
        # Update cache
        self._personas_cache[persona.id] = persona
    
    def delete_persona(self, persona_id: str) -> bool:
        """
        Delete a persona.
        
        Args:
            persona_id: ID of persona to delete
            
        Returns:
            True if successful, False otherwise
        """
        if persona_id == "default":
            return False  # Cannot delete default persona
        
        persona_dir = self.personas_dir / persona_id
        if persona_dir.exists():
            shutil.rmtree(persona_dir)
            if persona_id in self._personas_cache:
                del self._personas_cache[persona_id]
            
            # Switch to default if deleting current
            if self.current_persona_id == persona_id:
                self.current_persona_id = "default"
            
            return True
        return False
    
    def copy_persona(self, source_id: str, new_name: str, copy_learning_data: bool = False) -> Optional[AIPersona]:
        """
        COPY/PASTE - Clone an AI to create a variant.
        
        Copies:
        - All personality traits
        - Voice settings
        - Avatar settings
        - System prompt
        - Knowledge domains
        - Catchphrases
        - Preferences
        
        Does NOT copy by default:
        - Conversation history (fresh start)
        - Learning data (optional - user chooses)
        
        Args:
            source_id: ID of persona to copy
            new_name: Name for the new persona
            copy_learning_data: If True, copy learning data too
            
        Returns:
            New persona or None if failed
        """
        source = self.load_persona(source_id)
        if source is None:
            return None
        
        # Create new persona with copied data
        new_id = str(uuid.uuid4())[:8]  # Short UUID for readability
        new_persona = AIPersona(
            id=new_id,
            name=new_name,
            created_at=datetime.now().isoformat(),
            personality_traits=source.personality_traits.copy(),
            voice_profile_id=source.voice_profile_id,
            avatar_preset_id=source.avatar_preset_id,
            system_prompt=source.system_prompt,
            response_style=source.response_style,
            knowledge_domains=source.knowledge_domains.copy(),
            memories=[],  # Fresh start
            learning_data_path="",
            model_weights_path="",
            catchphrases=source.catchphrases.copy(),
            preferences=source.preferences.copy(),
            description=f"Copy of {source.name}",
            tags=source.tags.copy() + ["copy"]
        )
        
        # Save new persona
        self.save_persona(new_persona)
        
        # Copy learning data if requested
        if copy_learning_data and source.learning_data_path:
            source_learning_dir = Path(source.learning_data_path)
            if source_learning_dir.exists():
                new_learning_dir = self.personas_dir / new_id / "learning"
                new_learning_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy all files in learning directory
                for file in source_learning_dir.iterdir():
                    if file.is_file():
                        shutil.copy2(file, new_learning_dir / file.name)
                
                new_persona.learning_data_path = str(new_learning_dir)
                self.save_persona(new_persona)
        
        return new_persona
    
    def export_persona(self, persona_id: str, output_path: Path) -> Optional[Path]:
        """
        Export persona as shareable file (.forge-ai or .json).
        
        Users can share their AI configs with others.
        
        Args:
            persona_id: ID of persona to export
            output_path: Where to save the exported file
            
        Returns:
            Path to exported file or None if failed
        """
        persona = self.load_persona(persona_id)
        if persona is None:
            return None
        
        # Prepare export data (exclude some fields)
        export_data = persona.to_dict()
        
        # Clear paths that won't be valid on another system
        export_data['learning_data_path'] = ""
        export_data['model_weights_path'] = ""
        export_data['memories'] = []  # Don't export memories
        
        # Add export metadata
        export_data['exported_at'] = datetime.now().isoformat()
        export_data['exported_from'] = 'Enigma AI Engine'
        export_data['format_version'] = '1.0'
        
        # Ensure output path has correct extension
        if not output_path.suffix:
            output_path = output_path.with_suffix('.forge-ai')
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return output_path
    
    def import_persona(self, file_path: Path, new_name: Optional[str] = None) -> Optional[AIPersona]:
        """
        Import persona from file.
        
        Creates a new AI based on someone else's shared config.
        
        Args:
            file_path: Path to .forge-ai or .json file
            new_name: Optional name override
            
        Returns:
            Imported persona or None if failed
        """
        if not file_path.exists():
            return None
        
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            # Remove export metadata that's not part of the dataclass
            data.pop('exported_at', None)
            data.pop('exported_from', None)
            data.pop('format_version', None)
            
            # Generate new ID
            new_id = str(uuid.uuid4())[:8]
            data['id'] = new_id
            
            # Override name if provided
            if new_name:
                data['name'] = new_name
            
            # Set timestamps
            data['created_at'] = datetime.now().isoformat()
            data['last_modified'] = datetime.now().isoformat()
            
            # Clear system-specific paths
            data['learning_data_path'] = ""
            data['model_weights_path'] = ""
            
            # Add import tag
            if 'tags' not in data:
                data['tags'] = []
            if 'imported' not in data['tags']:
                data['tags'].append('imported')
            
            # Create persona
            persona = AIPersona.from_dict(data)
            
            # Save it
            self.save_persona(persona)
            
            return persona
            
        except Exception as e:
            print(f"Error importing persona: {e}")
            return None
    
    def merge_personas(self, base_id: str, overlay_id: str, new_name: str) -> Optional[AIPersona]:
        """
        Merge two personas - take traits from both.
        
        Example: Take personality from one, voice from another.
        
        Args:
            base_id: ID of base persona (provides foundation)
            overlay_id: ID of overlay persona (overrides selected fields)
            new_name: Name for merged persona
            
        Returns:
            Merged persona or None if failed
        """
        base = self.load_persona(base_id)
        overlay = self.load_persona(overlay_id)
        
        if base is None or overlay is None:
            return None
        
        # Start with base
        merged = self.copy_persona(base_id, new_name, copy_learning_data=False)
        if merged is None:
            return None
        
        # Merge personality traits (average values)
        for trait_name in merged.personality_traits.keys():
            if trait_name in overlay.personality_traits:
                base_val = merged.personality_traits[trait_name]
                overlay_val = overlay.personality_traits[trait_name]
                merged.personality_traits[trait_name] = (base_val + overlay_val) / 2.0
        
        # Combine knowledge domains
        merged.knowledge_domains = list(set(merged.knowledge_domains + overlay.knowledge_domains))
        
        # Combine catchphrases
        merged.catchphrases = list(set(merged.catchphrases + overlay.catchphrases))
        
        # Merge preferences (overlay wins on conflicts)
        merged.preferences.update(overlay.preferences)
        
        # Add merge tag
        merged.tags.append("merged")
        merged.description = f"Merged from {base.name} and {overlay.name}"
        
        # Save merged persona
        self.save_persona(merged)
        
        return merged
    
    def integrate_with_personality(self, persona: AIPersona) -> AIPersonality:
        """
        Create an AIPersonality instance from a persona.
        
        Args:
            persona: Persona to convert
            
        Returns:
            AIPersonality instance configured with persona's traits
        """
        personality = AIPersonality(persona.name)
        
        # Set traits from persona
        if persona.personality_traits:
            for trait_name, value in persona.personality_traits.items():
                if hasattr(personality.traits, trait_name):
                    setattr(personality.traits, trait_name, value)
        
        # Set other personality attributes
        if persona.catchphrases:
            personality.catchphrases = persona.catchphrases.copy()
        
        if persona.knowledge_domains:
            personality.interests = persona.knowledge_domains.copy()
        
        # Set voice preferences
        personality.voice_preferences['profile_id'] = persona.voice_profile_id
        
        return personality
    
    def update_persona_from_personality(self, persona: AIPersona, personality: AIPersonality):
        """
        Update persona with evolved personality traits.
        
        Args:
            persona: Persona to update
            personality: AIPersonality with evolved traits
        """
        # Update traits
        persona.personality_traits = personality.get_all_effective_traits()
        
        # Update catchphrases if they evolved
        if personality.catchphrases:
            persona.catchphrases = personality.catchphrases.copy()
        
        # Update knowledge domains
        if personality.interests:
            persona.knowledge_domains = personality.interests.copy()
        
        # Save changes
        self.save_persona(persona)


# Singleton instance
_persona_manager: Optional[PersonaManager] = None


def get_persona_manager() -> PersonaManager:
    """
    Get the global persona manager instance.
    
    Returns:
        Global PersonaManager instance
    """
    global _persona_manager
    if _persona_manager is None:
        _persona_manager = PersonaManager()
    return _persona_manager
