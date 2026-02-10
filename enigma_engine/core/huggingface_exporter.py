"""
HuggingFace Model Exporter
==========================

Export Enigma AI Engine-trained models to HuggingFace Hub format.
This allows sharing your trained Forge/Enigma models on HuggingFace.

Usage:
    from enigma_engine.core.huggingface_exporter import HuggingFaceExporter
    
    # Export a trained model
    exporter = HuggingFaceExporter()
    
    # Save locally in HuggingFace format
    exporter.export_to_hf_format("my_model", output_dir="./hf_export")
    
    # Push directly to HuggingFace Hub
    exporter.push_to_hub(
        model_name="my_model",
        repo_id="username/my-forge-model",
        token="hf_..."  # Your HuggingFace token
    )
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# Check for huggingface_hub
HAVE_HF_HUB = False
try:
    from huggingface_hub import HfApi, create_repo
    HAVE_HF_HUB = True
except ImportError:
    logger.info("huggingface_hub not installed - push_to_hub disabled. Install with: pip install huggingface-hub")

# Check for safetensors (preferred format)
HAVE_SAFETENSORS = False
try:
    from safetensors.torch import save_file as save_safetensors
    HAVE_SAFETENSORS = True
except ImportError:
    logger.info("safetensors not installed - using .bin format. Install with: pip install safetensors")


class HuggingFaceExporter:
    """
    Export Enigma AI Engine models to HuggingFace format.
    
    This converts Forge models to a format compatible with HuggingFace,
    allowing them to be shared on the Hub or loaded with transformers.
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the exporter.
        
        Args:
            models_dir: Path to Enigma AI Engine models directory (default: CONFIG["models_dir"])
        """
        from ..config import CONFIG
        self.models_dir = Path(models_dir or CONFIG["models_dir"])
        
    def _get_model_path(self, model_name: str) -> Path:
        """Get the path to a model directory."""
        model_path = self.models_dir / model_name
        if not model_path.exists():
            raise ValueError(f"Model '{model_name}' not found at {model_path}")
        return model_path
    
    def _load_forge_config(self, model_path: Path) -> dict[str, Any]:
        """Load Forge model configuration."""
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json found at {config_path}")
        
        with open(config_path) as f:
            return json.load(f)
    
    def _load_forge_metadata(self, model_path: Path) -> dict[str, Any]:
        """Load Forge model metadata."""
        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}
    
    def _convert_config_to_hf(self, forge_config: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Convert Forge config to HuggingFace-compatible config.
        
        Forge models use a custom architecture, so we create a custom config
        that preserves all the Forge-specific settings while being HF-loadable.
        """
        hf_config = {
            # Model architecture (custom Enigma AI Engine type)
            "model_type": "forge",
            "architectures": ["ForgeForCausalLM"],
            
            # Core dimensions
            "vocab_size": forge_config.get("vocab_size", 8000),
            "hidden_size": forge_config.get("dim", 512),
            "intermediate_size": forge_config.get("hidden_dim", 2048),
            "num_hidden_layers": forge_config.get("n_layers", 8),
            "num_attention_heads": forge_config.get("n_heads", 8),
            "num_key_value_heads": forge_config.get("n_kv_heads", 8),
            
            # Sequence settings
            "max_position_embeddings": forge_config.get("max_seq_len", 1024),
            
            # Architecture flags
            "hidden_act": "silu" if forge_config.get("use_swiglu", True) else "gelu",
            "use_rope": forge_config.get("use_rope", True),
            "use_rms_norm": forge_config.get("use_rms_norm", True),
            "rope_theta": forge_config.get("rope_theta", 10000.0),
            
            # Training info
            "torch_dtype": "float32",
            "tie_word_embeddings": False,
            
            # Enigma AI Engine specific (for round-trip loading)
            "_forge_config": forge_config,
            "_forge_metadata": {
                "name": metadata.get("name", "unknown"),
                "created": metadata.get("created", "unknown"),
                "total_epochs": metadata.get("total_epochs", 0),
                "estimated_parameters": metadata.get("estimated_parameters", 0),
            }
        }
        
        return hf_config
    
    def _convert_weights_to_hf(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Convert Forge weight names to HuggingFace-style names.
        
        This maintains compatibility while allowing the weights to be
        recognized by HuggingFace's loading infrastructure.
        """
        # Forge uses standard PyTorch naming, we just prefix for HF style
        hf_state_dict = {}
        
        for key, value in state_dict.items():
            # Convert Forge naming to HF-style naming
            # Forge: layers.0.attention.wq.weight -> model.layers.0.self_attn.q_proj.weight
            new_key = key
            
            # Add model prefix for HF compatibility
            if not key.startswith("model."):
                if key.startswith("embed"):
                    new_key = f"model.{key}"
                elif key.startswith("layers"):
                    new_key = f"model.{key}"
                elif key.startswith("norm"):
                    new_key = f"model.{key}"
                elif key.startswith("output"):
                    new_key = f"lm_head.{key.replace('output.', '')}"
            
            hf_state_dict[new_key] = value
        
        return hf_state_dict
    
    def export_to_hf_format(
        self,
        model_name: str,
        output_dir: str,
        include_tokenizer: bool = True,
        convert_to_safetensors: bool = True
    ) -> Path:
        """
        Export a Enigma AI Engine model to HuggingFace format.
        
        Args:
            model_name: Name of the model in Enigma AI Engine registry
            output_dir: Directory to save the HuggingFace-format model
            include_tokenizer: Include tokenizer files if available
            convert_to_safetensors: Use safetensors format (recommended)
            
        Returns:
            Path to the output directory
        """
        model_path = self._get_model_path(model_name)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting {model_name} to HuggingFace format at {output_path}")
        
        # Load Forge config and metadata
        forge_config = self._load_forge_config(model_path)
        metadata = self._load_forge_metadata(model_path)
        
        # Convert and save config
        hf_config = self._convert_config_to_hf(forge_config, metadata)
        with open(output_path / "config.json", "w") as f:
            json.dump(hf_config, f, indent=2)
        logger.info("Saved config.json")
        
        # Load and convert weights
        weights_path = model_path / "weights.pth"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            hf_state_dict = self._convert_weights_to_hf(state_dict)
            
            # Save in safetensors or pytorch format
            if convert_to_safetensors and HAVE_SAFETENSORS:
                save_safetensors(hf_state_dict, output_path / "model.safetensors")
                logger.info("Saved model.safetensors")
            else:
                torch.save(hf_state_dict, output_path / "pytorch_model.bin")
                logger.info("Saved pytorch_model.bin")
        else:
            logger.warning(f"No weights.pth found - exporting config only")
        
        # Copy/create tokenizer files if available
        if include_tokenizer:
            self._export_tokenizer(model_path, output_path)
        
        # Create model card (README.md)
        self._create_model_card(output_path, model_name, forge_config, metadata)
        
        logger.info(f"Export complete: {output_path}")
        return output_path
    
    def _export_tokenizer(self, model_path: Path, output_path: Path):
        """Export tokenizer files if available."""
        from ..config import CONFIG

        # Check for tokenizer in model directory
        tokenizer_files = ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"]
        vocab_model_dir = Path(CONFIG.get("vocab_model_dir", "enigma_engine/vocab_model"))
        
        # Try model directory first, then vocab_model_dir
        for source_dir in [model_path, vocab_model_dir]:
            for fname in tokenizer_files:
                src = source_dir / fname
                if src.exists():
                    shutil.copy(src, output_path / fname)
                    logger.info(f"Copied {fname}")
        
        # Create tokenizer_config.json if not present
        tokenizer_config_path = output_path / "tokenizer_config.json"
        if not tokenizer_config_path.exists():
            tokenizer_config = {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_type": "forge",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            }
            with open(tokenizer_config_path, "w") as f:
                json.dump(tokenizer_config, f, indent=2)
            logger.info("Created tokenizer_config.json")
    
    def _create_model_card(
        self,
        output_path: Path,
        model_name: str,
        config: dict[str, Any],
        metadata: dict[str, Any]
    ):
        """Create a HuggingFace model card (README.md)."""
        params = metadata.get("estimated_parameters", 0)
        if params >= 1_000_000_000:
            params_str = f"{params / 1_000_000_000:.1f}B"
        elif params >= 1_000_000:
            params_str = f"{params / 1_000_000:.0f}M"
        else:
            params_str = f"{params:,}"
        
        card = f"""---
language:
- en
library_name: forge-ai
tags:
- forge
- Enigma AI Engine
- causal-lm
- text-generation
license: mit
---

# {model_name}

This model was trained using [Enigma AI Engine](https://github.com/Enigma AI Engine/forge-ai), a modular AI framework.

## Model Details

- **Architecture**: Forge Transformer (LLaMA-style)
- **Parameters**: {params_str}
- **Hidden Size**: {config.get("dim", "unknown")}
- **Layers**: {config.get("n_layers", "unknown")}
- **Attention Heads**: {config.get("n_heads", "unknown")}
- **Context Length**: {config.get("max_seq_len", "unknown")}
- **Created**: {metadata.get("created", "unknown")}
- **Training Epochs**: {metadata.get("total_epochs", 0)}

## Features

- RoPE (Rotary Position Embeddings)
- RMSNorm
- SwiGLU activation
- Grouped Query Attention (GQA)
- KV Cache for efficient generation

## Usage

### With Enigma AI Engine (Recommended)

```python
from enigma_engine.core.model_registry import ModelRegistry

# If you have the model locally
registry = ModelRegistry()
model, config = registry.load_model("{model_name}")
```

### Load Weights Manually

```python
import torch
from safetensors.torch import load_file

# Load the weights
weights = load_file("model.safetensors")

# The config.json contains the full architecture specification
import json
with open("config.json") as f:
    config = json.load(f)
    
# Use _forge_config for exact Enigma AI Engine settings
forge_config = config.get("_forge_config", {{}})
```

## Training

This model was trained on custom data using the Enigma AI Engine training pipeline.

## License

MIT License
"""
        
        with open(output_path / "README.md", "w") as f:
            f.write(card)
        logger.info("Created README.md (model card)")
    
    def push_to_hub(
        self,
        model_name: str,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: str = "Upload Enigma AI Engine model",
        include_tokenizer: bool = True
    ) -> str:
        """
        Push a Enigma AI Engine model directly to HuggingFace Hub.
        
        Args:
            model_name: Name of the model in Enigma AI Engine registry
            repo_id: HuggingFace repo ID (e.g., "username/model-name")
            token: HuggingFace API token (or set HF_TOKEN env var)
            private: Make the repository private
            commit_message: Commit message for the upload
            include_tokenizer: Include tokenizer files
            
        Returns:
            URL of the uploaded model
        """
        if not HAVE_HF_HUB:
            raise ImportError(
                "huggingface_hub is required for push_to_hub. "
                "Install with: pip install huggingface-hub"
            )
        
        # Get token from env if not provided
        token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise ValueError(
                "No HuggingFace token provided. Either pass token= argument, "
                "or set HF_TOKEN environment variable. "
                "Get your token from https://huggingface.co/settings/tokens"
            )
        
        # Create temp directory for export
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export to HF format
            export_path = self.export_to_hf_format(
                model_name,
                tmpdir,
                include_tokenizer=include_tokenizer
            )
            
            # Create repo if needed
            api = HfApi(token=token)
            try:
                create_repo(repo_id, token=token, private=private, exist_ok=True)
            except Exception as e:
                logger.warning(f"Repo creation note: {e}")
            
            # Upload
            logger.info(f"Uploading to {repo_id}...")
            url = api.upload_folder(
                folder_path=str(export_path),
                repo_id=repo_id,
                token=token,
                commit_message=commit_message
            )
            
            logger.info(f"Upload complete! Model available at: https://huggingface.co/{repo_id}")
            return f"https://huggingface.co/{repo_id}"
    
    def list_exportable_models(self) -> dict[str, dict[str, Any]]:
        """List all Enigma AI Engine models that can be exported."""
        from .model_registry import ModelRegistry
        
        registry = ModelRegistry(str(self.models_dir))
        models = registry.list_models()
        
        # Filter to only Forge models (not already HuggingFace)
        exportable = {}
        for name, info in models.items():
            if info.get("source") != "huggingface":
                model_path = Path(info.get("path", ""))
                has_weights = (model_path / "weights.pth").exists() if model_path.exists() else False
                exportable[name] = {
                    **info,
                    "has_weights": has_weights,
                    "exportable": has_weights
                }
        
        return exportable


def export_model_to_hub(
    model_name: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False
) -> str:
    """
    Convenience function to export and push a model to HuggingFace Hub.
    
    Args:
        model_name: Name of the Enigma AI Engine model
        repo_id: HuggingFace repo ID (e.g., "username/my-model")
        token: HuggingFace API token
        private: Make repo private
        
    Returns:
        URL of the uploaded model
    """
    exporter = HuggingFaceExporter()
    return exporter.push_to_hub(model_name, repo_id, token=token, private=private)


def export_model_locally(model_name: str, output_dir: str) -> Path:
    """
    Convenience function to export a model to HuggingFace format locally.
    
    Args:
        model_name: Name of the Enigma AI Engine model
        output_dir: Directory to save the exported model
        
    Returns:
        Path to the exported model
    """
    exporter = HuggingFaceExporter()
    return exporter.export_to_hf_format(model_name, output_dir)
