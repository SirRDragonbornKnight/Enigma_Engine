"""
================================================================================
CAPABILITY REGISTRY - TRACK WHAT EACH MODEL CAN DO
================================================================================

The Capability Registry maintains a catalog of what each model and tool can do.
This enables intelligent routing and ensures requests go to capable models.

FILE: enigma_engine/core/capability_registry.py
TYPE: Model Capability Tracking
MAIN CLASS: CapabilityRegistry

CAPABILITIES:
    • text_generation      - Generate text from prompts
    • code_generation      - Write and explain code
    • vision               - Understand images
    • image_generation     - Create images from text
    • audio_generation     - Generate audio/speech
    • speech_to_text       - Transcribe audio to text
    • text_to_speech       - Convert text to audio
    • embedding            - Create vector embeddings
    • reasoning            - Complex problem solving
    • tool_calling         - Execute function calls
    • translation          - Language translation
    • summarization        - Text summarization

USAGE:
    from enigma_engine.core.capability_registry import CapabilityRegistry, Capability
    
    registry = CapabilityRegistry()
    
    # Register a model's capabilities
    registry.register_model(
        model_id="forge:small",
        capabilities=["text_generation", "reasoning"],
        metadata={"size": "27M", "device": "cpu"}
    )
    
    # Find models with specific capability
    models = registry.find_models_with_capability("code_generation")
    
    # Check if model has capability
    can_code = registry.has_capability("forge:small", "code_generation")
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CAPABILITY DEFINITIONS
# =============================================================================

@dataclass
class Capability:
    """Definition of a capability that models can have."""
    
    name: str                      # Capability identifier
    display_name: str              # Human-readable name
    description: str               # What this capability does
    requires_input: list[str]      # Required input types (e.g., "text", "image")
    produces_output: list[str]     # Output types (e.g., "text", "image")
    category: str = "general"      # Category grouping


# Built-in capability definitions
BUILT_IN_CAPABILITIES = {
    "text_generation": Capability(
        name="text_generation",
        display_name="Text Generation",
        description="Generate text from prompts and continue conversations",
        requires_input=["text"],
        produces_output=["text"],
        category="generation"
    ),
    "code_generation": Capability(
        name="code_generation",
        display_name="Code Generation",
        description="Write and explain code in various programming languages",
        requires_input=["text"],
        produces_output=["text", "code"],
        category="generation"
    ),
    "vision": Capability(
        name="vision",
        display_name="Vision/Image Understanding",
        description="Analyze and describe images, answer questions about pictures",
        requires_input=["image", "text"],
        produces_output=["text"],
        category="perception"
    ),
    "image_generation": Capability(
        name="image_generation",
        display_name="Image Generation",
        description="Create images from text descriptions",
        requires_input=["text"],
        produces_output=["image"],
        category="generation"
    ),
    "audio_generation": Capability(
        name="audio_generation",
        display_name="Audio Generation",
        description="Generate audio, music, or sound effects",
        requires_input=["text"],
        produces_output=["audio"],
        category="generation"
    ),
    "speech_to_text": Capability(
        name="speech_to_text",
        display_name="Speech to Text (STT)",
        description="Transcribe spoken audio to written text",
        requires_input=["audio"],
        produces_output=["text"],
        category="perception"
    ),
    "text_to_speech": Capability(
        name="text_to_speech",
        display_name="Text to Speech (TTS)",
        description="Convert written text to spoken audio",
        requires_input=["text"],
        produces_output=["audio"],
        category="generation"
    ),
    "embedding": Capability(
        name="embedding",
        display_name="Text Embedding",
        description="Create vector representations of text for semantic search",
        requires_input=["text"],
        produces_output=["vector"],
        category="utility"
    ),
    "reasoning": Capability(
        name="reasoning",
        display_name="Reasoning/Problem Solving",
        description="Complex logical reasoning and problem solving",
        requires_input=["text"],
        produces_output=["text"],
        category="cognition"
    ),
    "tool_calling": Capability(
        name="tool_calling",
        display_name="Tool/Function Calling",
        description="Execute function calls and use external tools",
        requires_input=["text"],
        produces_output=["function_call"],
        category="cognition"
    ),
    "translation": Capability(
        name="translation",
        display_name="Language Translation",
        description="Translate text between languages",
        requires_input=["text"],
        produces_output=["text"],
        category="language"
    ),
    "summarization": Capability(
        name="summarization",
        display_name="Text Summarization",
        description="Create concise summaries of longer texts",
        requires_input=["text"],
        produces_output=["text"],
        category="language"
    ),
    "video_generation": Capability(
        name="video_generation",
        display_name="Video Generation",
        description="Create video clips from text descriptions",
        requires_input=["text"],
        produces_output=["video"],
        category="generation"
    ),
    "3d_generation": Capability(
        name="3d_generation",
        display_name="3D Model Generation",
        description="Create 3D models and meshes",
        requires_input=["text"],
        produces_output=["3d_model"],
        category="generation"
    ),
}


# =============================================================================
# MODEL CAPABILITY ENTRY
# =============================================================================

@dataclass
class ModelCapabilityEntry:
    """Record of a model's capabilities and metadata."""
    
    model_id: str                              # Unique model identifier
    capabilities: set[str]                     # Set of capability names
    metadata: dict[str, Any] = field(default_factory=dict)  # Model info
    performance_ratings: dict[str, float] = field(default_factory=dict)  # Capability performance (0-1)
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "capabilities": list(self.capabilities),
            "metadata": self.metadata,
            "performance_ratings": self.performance_ratings,
            "registered_at": self.registered_at,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelCapabilityEntry":
        """Create from dictionary."""
        data = data.copy()
        data["capabilities"] = set(data.get("capabilities", []))
        return cls(**data)


# =============================================================================
# CAPABILITY REGISTRY
# =============================================================================

class CapabilityRegistry:
    """
    Central registry tracking model capabilities.
    
    Maintains a catalog of:
    - What each model can do
    - Performance ratings per capability
    - Model metadata (size, requirements, etc.)
    - Capability definitions
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the capability registry.
        
        Args:
            config_path: Path to save/load registry data
        """
        self._models: dict[str, ModelCapabilityEntry] = {}
        self._capabilities: dict[str, Capability] = BUILT_IN_CAPABILITIES.copy()
        self._config_path = Path(config_path) if config_path else None
        
        # Load existing registry if available
        if self._config_path and self._config_path.exists():
            self.load()
    
    # -------------------------------------------------------------------------
    # MODEL REGISTRATION
    # -------------------------------------------------------------------------
    
    def register_model(
        self,
        model_id: str,
        capabilities: list[str],
        metadata: Optional[dict[str, Any]] = None,
        performance_ratings: Optional[dict[str, float]] = None,
        auto_detect: bool = False,
    ) -> None:
        """
        Register a model and its capabilities.
        
        Args:
            model_id: Unique identifier for the model
            capabilities: List of capability names this model has
            metadata: Optional model metadata (size, device, etc.)
            performance_ratings: Optional performance scores per capability (0-1)
            auto_detect: If True, try to auto-detect capabilities from metadata
        """
        # Auto-detect capabilities if requested
        if auto_detect:
            detected = self._auto_detect_capabilities(model_id, metadata or {})
            capabilities = list(set(capabilities + detected))
        
        # Validate capabilities
        invalid = [c for c in capabilities if c not in self._capabilities]
        if invalid:
            logger.warning(f"Unknown capabilities for {model_id}: {invalid}")
            capabilities = [c for c in capabilities if c in self._capabilities]
        
        # Create or update entry
        if model_id in self._models:
            entry = self._models[model_id]
            entry.capabilities.update(capabilities)
            if metadata:
                entry.metadata.update(metadata)
            if performance_ratings:
                entry.performance_ratings.update(performance_ratings)
            entry.last_updated = datetime.now().isoformat()
        else:
            entry = ModelCapabilityEntry(
                model_id=model_id,
                capabilities=set(capabilities),
                metadata=metadata or {},
                performance_ratings=performance_ratings or {},
            )
            self._models[model_id] = entry
        
        logger.info(f"Registered model {model_id} with capabilities: {capabilities}")
    
    def unregister_model(self, model_id: str) -> bool:
        """
        Unregister a model.
        
        Args:
            model_id: Model to unregister
            
        Returns:
            True if model was unregistered, False if not found
        """
        if model_id in self._models:
            del self._models[model_id]
            logger.info(f"Unregistered model {model_id}")
            return True
        return False
    
    # -------------------------------------------------------------------------
    # CAPABILITY QUERIES
    # -------------------------------------------------------------------------
    
    def has_capability(self, model_id: str, capability: str) -> bool:
        """
        Check if a model has a specific capability.
        
        Args:
            model_id: Model to check
            capability: Capability name
            
        Returns:
            True if model has the capability
        """
        if model_id not in self._models:
            return False
        return capability in self._models[model_id].capabilities
    
    def get_capabilities(self, model_id: str) -> list[str]:
        """
        Get all capabilities of a model.
        
        Args:
            model_id: Model to query
            
        Returns:
            List of capability names
        """
        if model_id not in self._models:
            return []
        return list(self._models[model_id].capabilities)
    
    def find_models_with_capability(
        self,
        capability: str,
        min_performance: Optional[float] = None,
    ) -> list[str]:
        """
        Find all models with a specific capability.
        
        Args:
            capability: Capability to search for
            min_performance: Minimum performance rating (0-1)
            
        Returns:
            List of model IDs
        """
        results = []
        for model_id, entry in self._models.items():
            if capability not in entry.capabilities:
                continue
            
            # Check performance requirement
            if min_performance is not None:
                rating = entry.performance_ratings.get(capability, 0.0)
                if rating < min_performance:
                    continue
            
            results.append(model_id)
        
        return results
    
    def find_best_model(
        self,
        capability: str,
        requirements: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Find the best model for a capability based on performance ratings.
        
        Args:
            capability: Capability needed
            requirements: Optional requirements (e.g., {"device": "cpu"})
            
        Returns:
            Model ID of best model, or None if no suitable model found
        """
        candidates = self.find_models_with_capability(capability)
        
        if not candidates:
            return None
        
        # Filter by requirements
        if requirements:
            filtered = []
            for model_id in candidates:
                entry = self._models[model_id]
                metadata = entry.metadata
                
                # Check each requirement
                matches = True
                for key, value in requirements.items():
                    if key not in metadata or metadata[key] != value:
                        matches = False
                        break
                
                if matches:
                    filtered.append(model_id)
            
            candidates = filtered
        
        if not candidates:
            return None
        
        # Sort by performance rating
        def get_rating(model_id: str) -> float:
            entry = self._models[model_id]
            return entry.performance_ratings.get(capability, 0.5)
        
        candidates.sort(key=get_rating, reverse=True)
        return candidates[0]
    
    # -------------------------------------------------------------------------
    # CAPABILITY MANAGEMENT
    # -------------------------------------------------------------------------
    
    def register_capability(self, capability: Capability) -> None:
        """
        Register a new capability definition.
        
        Args:
            capability: Capability to register
        """
        self._capabilities[capability.name] = capability
        logger.info(f"Registered capability: {capability.name}")
    
    def get_capability_definition(self, name: str) -> Optional[Capability]:
        """
        Get definition of a capability.
        
        Args:
            name: Capability name
            
        Returns:
            Capability definition or None
        """
        return self._capabilities.get(name)
    
    def list_capabilities(self) -> list[str]:
        """Get list of all known capabilities."""
        return list(self._capabilities.keys())
    
    # -------------------------------------------------------------------------
    # MODEL INFORMATION
    # -------------------------------------------------------------------------
    
    def get_model_info(self, model_id: str) -> Optional[dict[str, Any]]:
        """
        Get complete information about a model.
        
        Args:
            model_id: Model to query
            
        Returns:
            Dictionary with model info or None
        """
        if model_id not in self._models:
            return None
        return self._models[model_id].to_dict()
    
    def list_models(self) -> list[str]:
        """Get list of all registered models."""
        return list(self._models.keys())
    
    def update_performance_rating(
        self,
        model_id: str,
        capability: str,
        rating: float,
    ) -> None:
        """
        Update performance rating for a model's capability.
        
        Args:
            model_id: Model to update
            capability: Capability to rate
            rating: Performance rating (0-1)
        """
        if model_id not in self._models:
            logger.warning(f"Model {model_id} not registered")
            return
        
        if not (0.0 <= rating <= 1.0):
            logger.warning(f"Rating must be between 0 and 1, got {rating}")
            rating = max(0.0, min(1.0, rating))
        
        self._models[model_id].performance_ratings[capability] = rating
        self._models[model_id].last_updated = datetime.now().isoformat()
    
    # -------------------------------------------------------------------------
    # AUTO-DETECTION
    # -------------------------------------------------------------------------
    
    def _auto_detect_capabilities(
        self,
        model_id: str,
        metadata: dict[str, Any],
    ) -> list[str]:
        """
        Auto-detect capabilities from model metadata.
        
        Args:
            model_id: Model identifier
            metadata: Model metadata
            
        Returns:
            List of detected capability names
        """
        capabilities = []
        model_lower = model_id.lower()
        
        # Check for vision models
        if any(x in model_lower for x in ["vl", "vision", "llava", "cogvlm", "idefics", "clip"]):
            capabilities.append("vision")
        
        # Check for code models
        if any(x in model_lower for x in ["code", "coder", "codegen", "starcoder", "deepseek-coder"]):
            capabilities.append("code_generation")
        
        # Check for embedding models
        if any(x in model_lower for x in ["embed", "sentence", "e5", "bge"]):
            capabilities.append("embedding")
        
        # Most models have basic text generation
        if not capabilities or "instruct" in model_lower or "chat" in model_lower:
            capabilities.append("text_generation")
        
        # Instruction-tuned models typically have reasoning
        if "instruct" in model_lower or "chat" in model_lower:
            capabilities.append("reasoning")
        
        return capabilities
    
    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save registry to disk.
        
        Args:
            path: Save path (uses config_path if not provided)
        """
        save_path = Path(path) if path else self._config_path
        if not save_path:
            logger.warning("No save path configured for capability registry")
            return
        
        # Prepare data
        data = {
            "models": {
                model_id: entry.to_dict()
                for model_id, entry in self._models.items()
            },
            "custom_capabilities": {
                name: asdict(cap)
                for name, cap in self._capabilities.items()
                if name not in BUILT_IN_CAPABILITIES
            },
            "saved_at": datetime.now().isoformat(),
        }
        
        # Save to file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved capability registry to {save_path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load registry from disk.
        
        Args:
            path: Load path (uses config_path if not provided)
        """
        load_path = Path(path) if path else self._config_path
        if not load_path or not load_path.exists():
            return
        
        try:
            with open(load_path, encoding="utf-8") as f:
                data = json.load(f)
            
            # Load models
            self._models = {
                model_id: ModelCapabilityEntry.from_dict(entry_data)
                for model_id, entry_data in data.get("models", {}).items()
            }
            
            # Load custom capabilities
            for name, cap_data in data.get("custom_capabilities", {}).items():
                self._capabilities[name] = Capability(**cap_data)
            
            logger.info(f"Loaded capability registry from {load_path}")
        except Exception as e:
            logger.error(f"Failed to load capability registry: {e}")


# =============================================================================
# GLOBAL REGISTRY INSTANCE
# =============================================================================

_global_registry: Optional[CapabilityRegistry] = None


def get_capability_registry(config_path: Optional[str] = None) -> CapabilityRegistry:
    """
    Get the global capability registry instance.
    
    Args:
        config_path: Optional path for registry persistence
        
    Returns:
        Global CapabilityRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        from ..config import CONFIG
        if config_path is None:
            config_path = str(Path(CONFIG["data_dir"]) / "capability_registry.json")
        _global_registry = CapabilityRegistry(config_path)
    return _global_registry
