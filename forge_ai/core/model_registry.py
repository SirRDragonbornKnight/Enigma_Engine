"""
Model Registry - Manage multiple named AI models.

This allows you to:
  - Create new models with unique names
  - Train them separately with different data
  - Load any model by name
  - List all your trained models
  - Each model saves its config WITH its weights

USAGE:
    from forge_ai.core.model_registry import ModelRegistry

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

from .model import Forge, ForgeConfig, MODEL_PRESETS
from ..config import CONFIG


def safe_load_weights(path, map_location=None):
    """
    Safely load PyTorch weights, handling CVE-2025-32434 vulnerability.
    
    Tries methods in order:
    1. Safetensors (if available and .safetensors file exists)
    2. torch.load with weights_only=True (torch >= 2.6)
    3. torch.load without weights_only (older torch, with warning)
    """
    path = Path(path)
    
    # Try safetensors first (most secure)
    safetensors_path = path.with_suffix('.safetensors')
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
            return load_file(str(safetensors_path), device=str(map_location) if map_location else 'cpu')
        except ImportError:
            pass  # safetensors not installed
    
    # Try torch.load with weights_only=True first
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        error_msg = str(e).lower()
        # Check if this is the CVE vulnerability block or version issue
        if 'vulnerability' in error_msg or 'weights_only' in error_msg or 'v2.6' in error_msg or 'upgrade' in error_msg:
            # Fall back to loading without weights_only
            # This is less secure but necessary for older torch versions
            import warnings
            warnings.warn(
                f"Loading weights without weights_only=True. "
                f"Consider upgrading torch to 2.6+ or converting to safetensors format. "
                f"To convert: pip install safetensors && python -c \"import torch; from safetensors.torch import save_file; save_file(torch.load('{path}'), '{path.with_suffix('.safetensors')}')\"",
                UserWarning
            )
            return torch.load(path, map_location=map_location, weights_only=False)
        # Re-raise if it's a different error
        raise


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
        custom_config: Optional[Dict] = None,
        base_model: Optional[str] = None
    ) -> Forge:
        """
        Create a new named model.

        Args:
            name: Unique name for this AI (e.g., "artemis", "apollo")
            size: Preset size ("tiny", "small", "medium", "large", "xl", "xxl")
            vocab_size: Vocabulary size for tokenizer
            description: Optional description of this model's purpose
            custom_config: Override preset with custom dim/depth/heads
            base_model: Optional name of existing model to use as base (transfer learning)

        Returns:
            Initialized (untrained) model
        """
        import shutil
        
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
        (model_dir / "conversations").mkdir(exist_ok=True)  # AI's own conversation history

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
A: I was created using the ForgeAI.

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
            "base_model": base_model,  # Track lineage
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Update registry
        self.registry["models"][name] = {
            "path": str(model_dir),
            "size": size,
            "source": "forge_ai",  # Mark as Forge model (not HuggingFace)
            "created": metadata["created"],
            "has_weights": False,
            "data_dir": str(model_dir / "data"),
            "base_model": base_model,
        }
        
        # If base model specified, copy its weights
        if base_model:
            base_info = self.registry["models"].get(base_model)
            if base_info and base_info.get("has_weights"):
                base_dir = Path(base_info["path"])
                base_checkpoints = base_dir / "checkpoints"
                
                # Find the latest checkpoint from base
                if base_checkpoints.exists():
                    checkpoint_files = list(base_checkpoints.glob("*.pt"))
                    if checkpoint_files:
                        # Copy the latest checkpoint
                        latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
                        dest_checkpoint = model_dir / "checkpoints" / "base_weights.pt"
                        shutil.copy2(latest, dest_checkpoint)
                        
                        # Mark as having weights
                        self.registry["models"][name]["has_weights"] = True
                        
                        print(f"[SYSTEM] [OK] Copied weights from base model '{base_model}'")
                        print(f"[SYSTEM]   Base checkpoint: {latest.name}")
                        
                        # Also copy training data if exists
                        base_training = base_dir / "data" / "training.txt"
                        if base_training.exists():
                            # Append base training data to new model's training file
                            base_content = base_training.read_text()
                            current_content = training_data_file.read_text()
                            training_data_file.write_text(
                                current_content + 
                                f"\n\n# === Inherited from base model: {base_model} ===\n\n" +
                                base_content
                            )
                            print(f"[SYSTEM]   Inherited training data from base model")
            else:
                print(f"[SYSTEM] [!] Base model '{base_model}' has no trained weights - starting fresh")
        
        self._save_registry()

        print(f"[SYSTEM] [OK] Created model '{name}' ({size})")
        print(f"[SYSTEM]   Parameters: {metadata['estimated_parameters']:,}")
        print(f"[SYSTEM]   Location: {model_dir}")
        print(f"[SYSTEM]   Training data: {training_data_file}")
        if base_model:
            print(f"[SYSTEM]   Base model: {base_model}")

        # Return None instead of instantiating model - saves memory
        # Model will be created lazily when load_model() is called
        return None

    def load_model(
        self,
        name: str,
        device: Optional[str] = None,
        checkpoint: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Tuple[Forge, Dict]:
        """
        Load a model by name.

        Args:
            name: Model name
            device: Device to load to ("cuda", "cpu", or None for auto)
            checkpoint: Specific checkpoint to load (e.g., "epoch_100") or None for latest
            progress_callback: Optional callback(message, percent) for progress updates

        Returns:
            (model, config_dict)
        """
        def report(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)
        
        name = name.lower().strip()
        if name not in self.registry["models"]:
            available = list(self.registry['models'].keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")

        reg_info = self.registry["models"][name]
        model_dir = Path(reg_info["path"])
        
        report("Checking model type...", 5)
        
        # Check if this is a HuggingFace model
        if reg_info.get("source") == "huggingface":
            hf_model_id = reg_info.get("huggingface_id")
            if not hf_model_id:
                raise ValueError(f"HuggingFace model '{name}' missing huggingface_id in registry")
            
            report(f"Loading HuggingFace model: {hf_model_id}...", 10)
            # Use HuggingFace model class
            from .huggingface_loader import HuggingFaceModel
            hf_model = HuggingFaceModel(hf_model_id, device=device)
            report("Downloading/loading model files...", 20)
            hf_model.load()
            report("HuggingFace model loaded", 35)
            # Return full registry info as config (includes use_custom_tokenizer setting)
            config = {
                "source": "huggingface",
                "model_id": hf_model_id,
                "use_custom_tokenizer": reg_info.get("use_custom_tokenizer", False),
                **{k: v for k, v in reg_info.items() if k not in ["path", "created"]}
            }
            # Return the HuggingFaceModel wrapper (has .generate(), .chat(), etc.)
            return hf_model, config

        # Load config for local Forge models
        report("Reading model configuration...", 10)
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Model '{name}' is missing config.json. "
                f"The model directory exists but wasn't properly initialized. "
                f"Try creating a new model or check if files were deleted."
            )
        
        with open(config_path, "r") as f:
            config = json.load(f)

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        report(f"Creating model architecture ({device})...", 15)

        # Create ForgeConfig from saved config (handles legacy parameter names)
        model_config = ForgeConfig.from_dict(config)

        # Create model - MUST use config= keyword argument!
        model = Forge(config=model_config)
        
        report("Model architecture created", 20)

        # Load weights
        if checkpoint:
            weights_path = model_dir / "checkpoints" / f"{checkpoint}.pth"
        else:
            weights_path = model_dir / "weights.pth"

        if weights_path.exists():
            report("Loading model weights from disk...", 25)
            state_dict = safe_load_weights(weights_path, map_location=device)
            report("Applying weights to model...", 30)
            model.load_state_dict(state_dict)
            report("Weights loaded successfully", 35)
            # Silent load - no print to avoid confusion with AI output
        else:
            print(f"[SYSTEM] [!] No weights found - model is untrained")

        report(f"Moving model to {device}...", 38)
        model.to(device)

        return model, config

    def save_model(
        self,
        name: str,
        model: Forge,
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
        # Find the actual key in the registry (case-insensitive match)
        name_lower = name.lower().strip()
        actual_key = None
        for key in self.registry["models"]:
            if key.lower() == name_lower:
                actual_key = key
                break
        
        if actual_key is None:
            raise ValueError(f"Model '{name}' not found")
        
        name = actual_key  # Use the actual key from registry

        if not confirm:
            raise ValueError(f"Confirm deletion by passing confirm=True")

        import shutil
        model_dir = Path(self.registry["models"][name]["path"])
        shutil.rmtree(model_dir)

        del self.registry["models"][name]
        self._save_registry()

    def get_model_info(self, name: str) -> Dict:
        """Get detailed info about a model."""
        # Find the actual key in the registry (case-insensitive match)
        name_lower = name.lower().strip()
        actual_key = None
        for key in self.registry["models"]:
            if key.lower() == name_lower:
                actual_key = key
                break
        
        if actual_key is None:
            raise ValueError(f"Model '{name}' not found")
        
        reg_info = self.registry["models"][actual_key]
        model_dir = Path(reg_info["path"])

        # Load config if exists, otherwise use defaults
        config = {}
        config_path = model_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
            except Exception:
                pass
        
        # Load metadata if exists, otherwise use registry info
        metadata = {}
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except Exception:
                pass
        
        # For HuggingFace models, populate metadata from registry
        if reg_info.get("source") == "huggingface":
            metadata.setdefault("created", reg_info.get("created", "Unknown"))
            metadata.setdefault("source", "huggingface")
            metadata.setdefault("huggingface_id", reg_info.get("huggingface_id", "Unknown"))

        # List checkpoints
        checkpoints_dir = model_dir / "checkpoints"
        checkpoints = []
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("*.pth"))

        return {
            "config": config,
            "metadata": metadata,
            "checkpoints": [cp.stem for cp in checkpoints],
            "registry": reg_info,
        }

    def export_to_huggingface(
        self,
        name: str,
        output_dir: Optional[str] = None,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        private: bool = False
    ) -> str:
        """
        Export a ForgeAI model to HuggingFace format.
        
        Can either save locally or push directly to HuggingFace Hub.
        
        Args:
            name: Model name in registry
            output_dir: Local directory to save (if not pushing to Hub)
            repo_id: HuggingFace repo ID (e.g., "username/model-name") to push to
            token: HuggingFace API token (or set HF_TOKEN env var)
            private: Make the HuggingFace repo private
            
        Returns:
            Path to exported model (local) or URL (if pushed to Hub)
            
        Example:
            # Export locally
            registry.export_to_huggingface("my_model", output_dir="./hf_export")
            
            # Push to Hub
            registry.export_to_huggingface(
                "my_model",
                repo_id="username/my-model",
                token="hf_..."
            )
        """
        from .huggingface_exporter import HuggingFaceExporter
        
        name = name.lower().strip()
        if name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found")
        
        reg_info = self.registry["models"][name]
        if reg_info.get("source") == "huggingface":
            raise ValueError(
                f"Model '{name}' is already a HuggingFace model. "
                "Only ForgeAI-trained models can be exported."
            )
        
        exporter = HuggingFaceExporter(str(self.models_dir))
        
        if repo_id:
            # Push to Hub
            return exporter.push_to_hub(name, repo_id, token=token, private=private)
        elif output_dir:
            # Export locally
            path = exporter.export_to_hf_format(name, output_dir)
            return str(path)
        else:
            # Default: export to models_dir/name_hf_export
            default_output = self.models_dir / f"{name}_hf_export"
            path = exporter.export_to_hf_format(name, str(default_output))
            return str(path)


# Convenience function
def get_registry(models_dir: Optional[str] = None) -> ModelRegistry:
    """Get the model registry instance."""
    return ModelRegistry(models_dir)


if __name__ == "__main__":
    # Demo
    registry = ModelRegistry()
    registry.list_models()
