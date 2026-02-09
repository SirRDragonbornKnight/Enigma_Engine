"""
Model Registry - Manage multiple named AI models.

This allows you to:
  - Create new models with unique names
  - Train them separately with different data
  - Load any model by name
  - List all your trained models
  - Each model saves its config WITH its weights

USAGE:
    from enigma_engine.core.model_registry import ModelRegistry

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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..config import CONFIG
from .model import MODEL_PRESETS, Forge, ForgeConfig


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
        # Base directory for relative paths (project root)
        self._base_dir = Path(CONFIG.get("root", self.models_dir.parent))
        self._load_registry()
    
    def _to_relative_path(self, path: Path) -> str:
        """Convert absolute path to relative path from project root."""
        try:
            return str(path.relative_to(self._base_dir))
        except ValueError:
            # Path is outside project, keep absolute
            return str(path)
    
    def _to_absolute_path(self, path_str: str) -> Path:
        """Convert stored path to absolute path."""
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self._base_dir / path

    def _load_registry(self):
        """Load or create the model registry."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                self.registry = json.load(f)
            
            # Clean up orphaned entries (registered but folder doesn't exist)
            orphaned = []
            for name, info in self.registry.get("models", {}).items():
                # Get path from registry, or construct from models_dir/name
                model_path_str = info.get("path", "")
                if model_path_str:
                    model_path = self._to_absolute_path(model_path_str)
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
        custom_config: Optional[dict] = None,
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
A: I was created using the Enigma AI Engine.

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
            # Capabilities this model is trained for
            # When a model has capabilities, router auto-assigns it to those tools
            "capabilities": ["chat"],  # Default: chat only, add more as you train
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Update registry with relative paths for portability
        self.registry["models"][name] = {
            "path": self._to_relative_path(model_dir),
            "size": size,
            "source": "enigma_engine",  # Mark as Forge model (not HuggingFace)
            "created": metadata["created"],
            "has_weights": False,
            "data_dir": self._to_relative_path(model_dir / "data"),
            "base_model": base_model,
            "capabilities": ["chat"],  # What this model can do
        }
        
        # If base model specified, copy its weights
        if base_model:
            base_info = self.registry["models"].get(base_model)
            if base_info and base_info.get("has_weights"):
                base_dir = self._to_absolute_path(base_info["path"])
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
    ) -> tuple[Forge, dict]:
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
        model_dir = self._to_absolute_path(reg_info["path"])
        
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
        
        with open(config_path) as f:
            config = json.load(f)

        # Determine device using device profiles for smarter selection
        if device is None:
            try:
                from .device_profiles import get_device_profiler
                profiler = get_device_profiler()
                device = profiler.get_torch_device()
            except ImportError:
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

        model_dir = self._to_absolute_path(self.registry["models"][name]["path"])

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
        model_dir = self._to_absolute_path(self.registry["models"][name]["path"])

        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)

        metadata.update(kwargs)
        metadata["last_trained"] = datetime.now().isoformat()

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def list_models(self) -> dict[str, Any]:
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
        model_dir = self._to_absolute_path(self.registry["models"][name]["path"])
        shutil.rmtree(model_dir)

        del self.registry["models"][name]
        self._save_registry()

    def get_model_info(self, name: str) -> dict:
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
        model_dir = self._to_absolute_path(reg_info["path"])

        # Load config if exists, otherwise use defaults
        config = {}
        config_path = model_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
            except Exception:
                pass
        
        # Load metadata if exists, otherwise use registry info
        metadata = {}
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
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
        Export a Enigma AI Engine model to HuggingFace format.
        
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
                "Only Enigma AI Engine-trained models can be exported."
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

    # =========================================================================
    # MODEL VERSIONING
    # =========================================================================
    
    def create_version(
        self,
        name: str,
        version_name: Optional[str] = None,
        notes: str = "",
    ) -> str:
        """
        Create a named version snapshot of the current model weights.
        
        Saves a copy of the current weights with version metadata, allowing
        rollback if future training degrades quality.
        
        Args:
            name: Model name
            version_name: Optional version name (default: v1, v2, etc.)
            notes: Notes about this version (what changed, quality observations)
            
        Returns:
            Version name that was created
        """
        import shutil
        
        name = name.lower().strip()
        if name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found")
        
        reg_info = self.registry["models"][name]
        model_dir = self._to_absolute_path(reg_info["path"])
        
        # Create versions directory
        versions_dir = model_dir / "versions"
        versions_dir.mkdir(exist_ok=True)
        
        # Determine version name
        if version_name is None:
            existing = list(versions_dir.glob("v*"))
            version_num = len(existing) + 1
            version_name = f"v{version_num}"
        
        version_dir = versions_dir / version_name
        if version_dir.exists():
            raise ValueError(f"Version '{version_name}' already exists")
        
        version_dir.mkdir()
        
        # Copy current weights
        weights_path = model_dir / "weights.pth"
        if weights_path.exists():
            shutil.copy2(weights_path, version_dir / "weights.pth")
        
        # Copy config
        config_path = model_dir / "config.json"
        if config_path.exists():
            shutil.copy2(config_path, version_dir / "config.json")
        
        # Load current metadata for version info
        metadata_path = model_dir / "metadata.json"
        version_metadata = {
            "version": version_name,
            "created": datetime.now().isoformat(),
            "notes": notes,
            "epochs": 0,
            "quality_score": None,
        }
        
        if metadata_path.exists():
            with open(metadata_path) as f:
                current_meta = json.load(f)
            version_metadata["epochs"] = current_meta.get("total_epochs", 0)
        
        # Save version metadata
        with open(version_dir / "version_info.json", "w") as f:
            json.dump(version_metadata, f, indent=2)
        
        # Update model's version history
        if "versions" not in reg_info:
            reg_info["versions"] = []
        reg_info["versions"].append(version_name)
        reg_info["current_version"] = version_name
        self._save_registry()
        
        print(f"[SYSTEM] [OK] Created version '{version_name}' for model '{name}'")
        return version_name
    
    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a model.
        
        Args:
            name: Model name
            
        Returns:
            List of version info dicts sorted by creation date
        """
        name = name.lower().strip()
        if name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found")
        
        reg_info = self.registry["models"][name]
        model_dir = self._to_absolute_path(reg_info["path"])
        versions_dir = model_dir / "versions"
        
        if not versions_dir.exists():
            return []
        
        versions = []
        for version_path in versions_dir.iterdir():
            if version_path.is_dir():
                info_path = version_path / "version_info.json"
                if info_path.exists():
                    with open(info_path) as f:
                        info = json.load(f)
                    info["path"] = str(version_path)
                    info["has_weights"] = (version_path / "weights.pth").exists()
                    versions.append(info)
        
        # Sort by creation date
        versions.sort(key=lambda v: v.get("created", ""))
        return versions
    
    def rollback_to_version(self, name: str, version_name: str) -> bool:
        """
        Rollback a model to a previous version.
        
        Copies the version's weights to be the current weights.
        Creates a backup of current weights first.
        
        Args:
            name: Model name
            version_name: Version to rollback to
            
        Returns:
            True if successful
        """
        import shutil
        
        name = name.lower().strip()
        if name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found")
        
        reg_info = self.registry["models"][name]
        model_dir = self._to_absolute_path(reg_info["path"])
        version_dir = model_dir / "versions" / version_name
        
        if not version_dir.exists():
            raise ValueError(f"Version '{version_name}' not found")
        
        version_weights = version_dir / "weights.pth"
        if not version_weights.exists():
            raise ValueError(f"Version '{version_name}' has no weights")
        
        # Backup current weights before rollback
        current_weights = model_dir / "weights.pth"
        if current_weights.exists():
            backup_name = f"pre_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            backup_path = model_dir / "checkpoints" / backup_name
            backup_path.parent.mkdir(exist_ok=True)
            shutil.copy2(current_weights, backup_path)
            print(f"[SYSTEM]   Backed up current weights to: {backup_name}")
        
        # Copy version weights to current
        shutil.copy2(version_weights, current_weights)
        
        # Update registry
        reg_info["current_version"] = version_name
        self._save_registry()
        
        print(f"[SYSTEM] [OK] Rolled back '{name}' to version '{version_name}'")
        return True
    
    def compare_versions(
        self,
        name: str,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """
        Compare two versions of a model.
        
        Args:
            name: Model name
            version_a: First version to compare
            version_b: Second version to compare
            
        Returns:
            Comparison dict with differences
        """
        name = name.lower().strip()
        if name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found")
        
        reg_info = self.registry["models"][name]
        model_dir = self._to_absolute_path(reg_info["path"])
        
        # Load version info for both
        version_a_dir = model_dir / "versions" / version_a
        version_b_dir = model_dir / "versions" / version_b
        
        if not version_a_dir.exists():
            raise ValueError(f"Version '{version_a}' not found")
        if not version_b_dir.exists():
            raise ValueError(f"Version '{version_b}' not found")
        
        info_a = {}
        info_b = {}
        
        info_a_path = version_a_dir / "version_info.json"
        info_b_path = version_b_dir / "version_info.json"
        
        if info_a_path.exists():
            with open(info_a_path) as f:
                info_a = json.load(f)
        
        if info_b_path.exists():
            with open(info_b_path) as f:
                info_b = json.load(f)
        
        # Calculate weight file sizes
        weights_a = version_a_dir / "weights.pth"
        weights_b = version_b_dir / "weights.pth"
        
        size_a = weights_a.stat().st_size if weights_a.exists() else 0
        size_b = weights_b.stat().st_size if weights_b.exists() else 0
        
        return {
            "model": name,
            "version_a": {
                "name": version_a,
                "created": info_a.get("created", "Unknown"),
                "epochs": info_a.get("epochs", 0),
                "notes": info_a.get("notes", ""),
                "quality_score": info_a.get("quality_score"),
                "weights_size_mb": round(size_a / (1024 * 1024), 2),
            },
            "version_b": {
                "name": version_b,
                "created": info_b.get("created", "Unknown"),
                "epochs": info_b.get("epochs", 0),
                "notes": info_b.get("notes", ""),
                "quality_score": info_b.get("quality_score"),
                "weights_size_mb": round(size_b / (1024 * 1024), 2),
            },
            "differences": {
                "epochs_delta": info_b.get("epochs", 0) - info_a.get("epochs", 0),
                "size_delta_mb": round((size_b - size_a) / (1024 * 1024), 2),
            },
        }
    
    def set_version_quality_score(
        self,
        name: str,
        version_name: str,
        score: float,
        notes: str = "",
    ) -> None:
        """
        Set a quality score for a model version.
        
        Use this after testing to mark which versions are better.
        
        Args:
            name: Model name
            version_name: Version to score
            score: Quality score (0.0 to 1.0)
            notes: Optional notes about the quality
        """
        name = name.lower().strip()
        if name not in self.registry["models"]:
            raise ValueError(f"Model '{name}' not found")
        
        reg_info = self.registry["models"][name]
        model_dir = self._to_absolute_path(reg_info["path"])
        version_dir = model_dir / "versions" / version_name
        
        if not version_dir.exists():
            raise ValueError(f"Version '{version_name}' not found")
        
        info_path = version_dir / "version_info.json"
        info = {}
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
        
        info["quality_score"] = max(0.0, min(1.0, score))
        if notes:
            info["quality_notes"] = notes
        info["quality_scored_at"] = datetime.now().isoformat()
        
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"[SYSTEM] [OK] Set quality score {score:.2f} for '{name}' version '{version_name}'")


# Convenience function
def get_registry(models_dir: Optional[str] = None) -> ModelRegistry:
    """Get the model registry instance."""
    return ModelRegistry(models_dir)


# =============================================================================
# CAPABILITY MANAGEMENT - For Multi-Capable Models
# =============================================================================

def set_model_capabilities(model_name: str, capabilities: list, models_dir: Optional[str] = None):
    """
    Set the capabilities a model is trained for.
    
    When a model has capabilities, the router can auto-assign it to those tools.
    This means one model trained on chat+code+vision only loads ONCE and handles all three.
    
    Args:
        model_name: Name of the model in registry
        capabilities: List of capability names (e.g., ["chat", "code", "vision"])
        models_dir: Optional models directory
        
    Example:
        # Train a model on multiple things, then declare its capabilities:
        set_model_capabilities("my_multimodel", ["chat", "code", "vision", "web"])
        
        # Now the router knows this ONE model can handle all four!
    """
    registry = ModelRegistry(models_dir)
    name = model_name.lower().strip()
    
    if name not in registry.registry["models"]:
        raise ValueError(f"Model '{name}' not found")
    
    # Update registry
    registry.registry["models"][name]["capabilities"] = capabilities
    registry._save_registry()
    
    # Update metadata file too
    model_dir = registry._to_absolute_path(registry.registry["models"][name]["path"])
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["capabilities"] = capabilities
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    print(f"[SYSTEM] [OK] Set capabilities for '{name}': {capabilities}")


def get_model_capabilities(model_name: str, models_dir: Optional[str] = None) -> list:
    """
    Get the capabilities a model is trained for.
    
    Args:
        model_name: Name of the model in registry
        models_dir: Optional models directory
        
    Returns:
        List of capability names
    """
    registry = ModelRegistry(models_dir)
    name = model_name.lower().strip()
    
    if name not in registry.registry["models"]:
        raise ValueError(f"Model '{name}' not found")
    
    return registry.registry["models"][name].get("capabilities", ["chat"])


def find_models_with_capability(capability: str, models_dir: Optional[str] = None) -> list:
    """
    Find all models that have a specific capability.
    
    Args:
        capability: Capability to search for (e.g., "vision", "code")
        models_dir: Optional models directory
        
    Returns:
        List of model names that have this capability
    """
    registry = ModelRegistry(models_dir)
    models = []
    
    for name, info in registry.registry["models"].items():
        caps = info.get("capabilities", ["chat"])
        if capability in caps:
            models.append(name)
    
    return models


if __name__ == "__main__":
    # Demo
    registry = ModelRegistry()
    registry.list_models()
