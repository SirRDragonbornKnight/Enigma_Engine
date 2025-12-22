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

from .model import TinyEnigma
from .model_config import MODEL_PRESETS, get_model_config, estimate_parameters
from ..config import CONFIG


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
    ) -> TinyEnigma:
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
        training_data_file.write_text(f"""# Training Data for {name}
# ===========================
# Add your training data below. The more examples, the better!
#
# FORMAT OPTIONS:
#
# 1. Plain text (AI learns patterns and style):
#    Just write paragraphs of text the AI should learn from.
#
# 2. Q&A format:
#    Q: What is your name?
#    A: My name is {name}.
#
# 3. Conversation format:
#    User: Hello!
#    {name}: Hello! How can I help you today?
#
# TIPS:
# - Include many variations of common questions
# - Add personality through response style
# - More diverse examples = smarter AI
# ===========================

# Example conversations (customize these!):

Q: What is your name?
A: My name is {name}.

Q: Who made you?
A: I was created by my developer to be a helpful AI assistant.

Q: How are you?
A: I'm doing well, thank you for asking! How can I help you today?

User: Hello!
{name}: Hello! It's nice to meet you. How can I assist you?

User: Tell me about yourself.
{name}: I'm {name}, an AI assistant. I'm here to help with questions, have conversations, and assist with various tasks.

# Add more training data below...

""")
        
        # Create AI-specific instructions file
        instructions_file = model_dir / "data" / "instructions.txt"
        instructions_file.write_text(f"""# Instructions for {name}
# ===========================
# These instructions help define your AI's behavior and personality.
# Edit this file to customize how {name} responds.
# ===========================

# PERSONALITY:
# Describe the personality traits you want {name} to have.
# Example: friendly, helpful, curious, professional

{name} is a helpful and friendly AI assistant.
{name} gives clear and concise answers.
{name} is honest about its limitations.

# KNOWLEDGE:
# List topics or domains {name} should be knowledgeable about.

# RULES:
# Define any rules {name} should follow.

1. Be helpful and respectful
2. Give accurate information
3. Admit when unsure
4. Keep responses clear and relevant

# STYLE:
# Describe how {name} should communicate.

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
        
        # Create and return model
        model = TinyEnigma(**model_config)
        
        print(f"✓ Created model '{name}' ({size})")
        print(f"  Parameters: {metadata['estimated_parameters']:,}")
        print(f"  Location: {model_dir}")
        print(f"  Training data: {training_data_file}")
        
        return model
    
    def load_model(
        self, 
        name: str, 
        device: Optional[str] = None,
        checkpoint: Optional[str] = None
    ) -> Tuple[TinyEnigma, Dict]:
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
            raise ValueError(f"Model '{name}' not found. Available: {list(self.registry['models'].keys())}")
        
        model_dir = Path(self.registry["models"][name]["path"])
        
        # Load config
        with open(model_dir / "config.json", "r") as f:
            config = json.load(f)
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create model
        model = TinyEnigma(**config)
        
        # Load weights
        if checkpoint:
            weights_path = model_dir / "checkpoints" / f"{checkpoint}.pth"
        else:
            weights_path = model_dir / "weights.pth"
        
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✓ Loaded weights from {weights_path}")
        else:
            print(f"⚠ No weights found - model is untrained")
        
        model.to(device)
        
        return model, config
    
    def save_model(
        self,
        name: str,
        model: TinyEnigma,
        epoch: Optional[int] = None,
        save_checkpoint: bool = True
    ):
        """
        Save model weights.
        
        Args:
            name: Model name
            model: The model to save
            epoch: Current epoch (for checkpoint naming)
            save_checkpoint: Also save as a checkpoint
        """
        name = name.lower().strip()
        if name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found in registry")
        
        model_dir = Path(self.registry["models"][name]["path"])
        
        # Save main weights
        torch.save(model.state_dict(), model_dir / "weights.pth")
        
        # Save checkpoint
        if save_checkpoint and epoch is not None:
            checkpoint_path = model_dir / "checkpoints" / f"epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Update registry
        self.registry["models"][name]["has_weights"] = True
        self._save_registry()
        
        print(f"✓ Saved model '{name}'")
    
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
        """List all registered models."""
        print("\n" + "="*60)
        print("REGISTERED MODELS")
        print("="*60)
        
        if not self.registry["models"]:
            print("No models yet. Create one with registry.create_model('name')")
        else:
            for name, info in self.registry["models"].items():
                status = "✓ trained" if info["has_weights"] else "○ untrained"
                print(f"\n{name}")
                print(f"  Size: {info['size']}")
                print(f"  Status: {status}")
                print(f"  Created: {info['created'][:10]}")
        
        print("\n" + "="*60)
        return self.registry["models"]
    
    def delete_model(self, name: str, confirm: bool = False):
        """Delete a model and all its files."""
        name = name.lower().strip()
        if name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found")
        
        if not confirm:
            print(f"⚠ This will permanently delete model '{name}' and all its weights!")
            print(f"  Call delete_model('{name}', confirm=True) to proceed.")
            return
        
        import shutil
        model_dir = Path(self.registry["models"][name]["path"])
        shutil.rmtree(model_dir)
        
        del self.registry["models"][name]
        self._save_registry()
        
        print(f"✓ Deleted model '{name}'")
    
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
