"""
Model Registry - Manage multiple named AI models.

This allows you to:
  - Create new models with unique names
  - Train them separately with different data
  - Load any model by name
  - List all your trained models
  - Each model saves its config WITH its weights

USAGE:
    from enigma.core.model_registry import ModelRegistry

    registry = ModelRegistry()

    # Create a new AI
    registry.create_model("artemis", size="small", vocab_size=32000)

    # List all models
    registry.list_models()

    # Load a specific model
    model, config = registry.load_model("artemis")

    # Save after training
    registry.save_model("artemis", model)
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, Any

from .model import Enigma, EnigmaConfig, MODEL_PRESETS
from ..config import CONFIG


def get_model_config(size: str) -> dict:
    """Get model config dict from MODEL_PRESETS."""
    if size not in MODEL_PRESETS:
        raise ValueError(f"Unknown model size: {size}. Available: {list(MODEL_PRESETS.keys())}")
    preset = MODEL_PRESETS[size]
    return preset.to_dict()


def estimate_parameters(vocab_size: int, dim: int, n_layers: int, **kwargs) -> int:
    """Estimate parameter count for a model configuration."""
    # Embedding: vocab_size * dim
    embed_params = vocab_size * dim
    # Output: vocab_size * dim
    output_params = vocab_size * dim
    # Per layer: ~12 * dim^2 (attention + FFN approximation)
    layer_params = 12 * dim * dim * n_layers
    return embed_params + output_params + layer_params


class ModelRegistry:
    """
    Manages multiple named AI models.

    Directory structure:
        models/
            registry.json           # Index of all models
            artemis/
                config.json         # Model architecture config
                metadata.json       # Training history, creation date, etc.
                weights.pth         # Model weights
                checkpoints/        # Training checkpoints
                    epoch_100.pth
                    epoch_200.pth
            apollo/
                config.json
                metadata.json
                weights.pth
                checkpoints/
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir or CONFIG["models_dir"])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.models_dir / "registry.json"
        self._load_registry()

    def _load_registry(self):
        """Load or create the model registry."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                self.registry = json.load(f)
            
            # Clean up orphaned entries (registered but folder doesn't exist)
            orphaned = []
            for name, info in self.registry.get("models", {}).items():
                # Get path from registry, or construct from models_dir/name
                model_path_str = info.get("path", "")
                if model_path_str:
                    model_path = Path(model_path_str)
                    # Handle relative paths
                    if not model_path.is_absolute():
                        model_path = self.models_dir.parent / model_path
                else:
                    model_path = self.models_dir / name
                
                if not model_path.exists():
                    orphaned.append(name)
            
            if orphaned:
                for name in orphaned:
                    del self.registry["models"][name]
                self._save_registry()
        else:
            self.registry = {"models": {}, "created": datetime.now().isoformat()}
            self._save_registry()

    def _save_registry(self):
        """Save the registry index."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def create_model(
        self,
        name: str,
        size: str = "tiny",
        vocab_size: int = 32000,
        description: str = "",
        custom_config: Optional[Dict] = None
    ) -> Enigma:
        """
        Create a new named model.

        Args:
            name: Unique name for this AI (e.g., "artemis", "apollo")
            size: Preset size ("tiny", "small", "medium", "large", "xl", "xxl")
            vocab_size: Vocabulary size for tokenizer
            description: Optional description of this model's purpose
            custom_config: Override preset with custom dim/depth/heads

        Returns:
            Initialized (untrained) model
        """
        # Validate name
        name = name.lower().strip().replace(" ", "_")
        if name in self.registry["models"]:
            raise ValueError(f"Model '{name}' already exists. Use load_model() or delete it first.")

        # Check if folder exists but isn't registered (orphaned folder)
        model_dir = self.models_dir / name
        if model_dir.exists():
            # Clean up orphaned folder so we can create fresh
            import shutil
            shutil.rmtree(model_dir)

        # Get config
        if custom_config:
            model_config = custom_config
        else:
            model_config = get_model_config(size)

        model_config["vocab_size"] = vocab_size

        # Create model directory with full structure
        model_dir = self.models_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "checkpoints").mkdir(exist_ok=True)
        (model_dir / "data").mkdir(exist_ok=True)  # AI's own training data

        # Save config
        with open(model_dir / "config.json", "w") as f:
            json.dump(model_config, f, indent=2)

        # Create AI-specific training data file
        training_data_file = model_dir / "data" / "training.txt"
        training_data_file.write_text(f"""# Training Data
# =============
# Add your training data below. The more examples, the better!
#
# FORMAT OPTIONS:
#
# 1. Plain text (AI learns patterns and style):
#    Just write paragraphs of text the AI should learn from.
#
# 2. Q&A format:
#    Q: What is your name?
#    A: I'm {name}.
#
# 3. Conversation format:
#    User: Hello!
#    AI: Hello! How can I help you today?
#
# TIPS:
# - Include many variations of common questions
# - Add personality through response style
# - More diverse examples = smarter AI
# =============

# Example conversations (customize these!):

Q: What is your name?
A: I'm {name}.

Q: Who made you?
A: I was created using the Enigma Engine.

Q: How are you?
A: I'm doing well, thank you for asking! How can I help you today?

User: Hello!
AI: Hello! It's nice to meet you. How can I assist you?

User: Tell me about yourself.
AI: I'm {name}, an AI assistant. I'm here to help with questions, have conversations, and assist with various tasks.

# Add more training data below...

""")

        # Create AI-specific instructions file
        instructions_file = model_dir / "data" / "instructions.txt"
        instructions_file.write_text(f"""# AI Instructions
# ================
# This file defines your AI's behavior and personality.
# Edit this file to customize how your AI responds.

# STEP 1: PERSONALITY
# -------------------
# Describe the traits you want your AI to have.
# Examples: friendly, helpful, curious, professional, playful

{name} is a helpful and friendly AI assistant.
{name} gives clear and concise answers.
{name} is honest about its limitations.

# STEP 2: KNOWLEDGE AREAS
# -----------------------
# List topics or domains your AI should focus on.
# Leave blank for general knowledge.


# STEP 3: BEHAVIOR RULES
# ----------------------
# Define rules your AI should follow.

1. Be helpful and respectful
2. Give accurate information
3. Admit when unsure
4. Keep responses clear and relevant

# STEP 4: COMMUNICATION STYLE
# ---------------------------
# Describe how your AI should communicate.

{name} uses a conversational tone.
{name} avoids overly technical jargon unless asked.
{name} asks clarifying questions when needed.
""")

        # Create metadata
        metadata = {
            "name": name,
            "description": description,
            "size_preset": size,
            "created": datetime.now().isoformat(),
            "last_trained": None,
            "total_epochs": 0,
            "total_steps": 0,
            "training_history": [],
            "estimated_parameters": estimate_parameters(**model_config),
            "data_files": {
                "training": str(training_data_file),
                "instructions": str(instructions_file),
            },
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Update registry
        self.registry["models"][name] = {
            "path": str(model_dir),
            "size": size,
            "created": metadata["created"],
            "has_weights": False,
            "data_dir": str(model_dir / "data"),
        }
        self._save_registry()

        print(f"[SYSTEM] [OK] Created model '{name}' ({size})")
        print(f"[SYSTEM]   Parameters: {metadata['estimated_parameters']:,}")
        print(f"[SYSTEM]   Location: {model_dir}")
        print(f"[SYSTEM]   Training data: {training_data_file}")

        # Return None instead of instantiating model - saves memory
        # Model will be created lazily when load_model() is called
        return None

    def load_model(
        self,
        name: str,
        device: Optional[str] = None,
        checkpoint: Optional[str] = None
    ) -> Tuple[Enigma, Dict]:
        """
        Load a model by name.

        Args:
            name: Model name
            device: Device to load to ("cuda", "cpu", or None for auto)
            checkpoint: Specific checkpoint to load (e.g., "epoch_100") or None for latest

        Returns:
            (model, config_dict)
        """
        name = name.lower().strip()
        if name not in self.registry["models"]:
            available = list(self.registry['models'].keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")

        model_dir = Path(self.registry["models"][name]["path"])

        # Load config
        with open(model_dir / "config.json", "r") as f:
            config = json.load(f)

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create EnigmaConfig from saved config (handles legacy parameter names)
        model_config = EnigmaConfig.from_dict(config)

        # Create model
        model = Enigma(model_config)

        # Load weights
        if checkpoint:
            weights_path = model_dir / "checkpoints" / f"{checkpoint}.pth"
        else:
            weights_path = model_dir / "weights.pth"

        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            # Silent load - no print to avoid confusion with AI output
        else:
            print(f"[SYSTEM] [!] No weights found - model is untrained")

        model.to(device)

        return model, config

    def save_model(
        self,
        name: str,
        model: Enigma,
        epoch: Optional[int] = None,
        save_checkpoint: bool = True,
        checkpoint_name: Optional[str] = None
    ):
        """
        Save model weights.

        Args:
            name: Model name
            model: The model to save
            epoch: Current epoch (for checkpoint naming)
            save_checkpoint: Also save as a checkpoint
            checkpoint_name: Custom checkpoint name (e.g., "best")
        """
        name = name.lower().strip()
        if name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found in registry")

        model_dir = Path(self.registry["models"][name]["path"])

        # Save main weights
        torch.save(model.state_dict(), model_dir / "weights.pth")

        # Save checkpoint
        if save_checkpoint:
            checkpoints_dir = model_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)

            if checkpoint_name:
                # Custom name like "best"
                checkpoint_path = checkpoints_dir / f"{checkpoint_name}.pth"
            elif epoch is not None:
                # Epoch-based name
                checkpoint_path = checkpoints_dir / f"epoch_{epoch}.pth"
            else:
                # Default
                checkpoint_path = checkpoints_dir / "latest.pth"

            torch.save(model.state_dict(), checkpoint_path)

        # Update registry
        self.registry["models"][name]["has_weights"] = True
        self._save_registry()

        # Silent save - no print to avoid confusion with AI output

    def update_metadata(self, name: str, **kwargs):
        """Update model metadata after training."""
        name = name.lower().strip()
        model_dir = Path(self.registry["models"][name]["path"])

        with open(model_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        metadata.update(kwargs)
        metadata["last_trained"] = datetime.now().isoformat()

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def list_models(self) -> Dict[str, Any]:
        """List all registered models. Returns dict, prints summary."""
        # Return data - GUI will display it
        return self.registry["models"]

    def delete_model(self, name: str, confirm: bool = False):
        """Delete a model and all its files."""
        name = name.lower().strip()
        if name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found")

        if not confirm:
            raise ValueError(f"Confirm deletion by passing confirm=True")

        import shutil
        model_dir = Path(self.registry["models"][name]["path"])
        shutil.rmtree(model_dir)

        del self.registry["models"][name]
        self._save_registry()

    def get_model_info(self, name: str) -> Dict:
        """Get detailed info about a model."""
        name = name.lower().strip()
        model_dir = Path(self.registry["models"][name]["path"])

        with open(model_dir / "config.json", "r") as f:
            config = json.load(f)
        with open(model_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # List checkpoints
        checkpoints = list((model_dir / "checkpoints").glob("*.pth"))

        return {
            "config": config,
            "metadata": metadata,
            "checkpoints": [cp.stem for cp in checkpoints],
            "registry": self.registry["models"][name],
        }


# Convenience function
def get_registry(models_dir: Optional[str] = None) -> ModelRegistry:
    """Get the model registry instance."""
    return ModelRegistry(models_dir)


if __name__ == "__main__":
    # Demo
    registry = ModelRegistry()
    registry.list_models()
