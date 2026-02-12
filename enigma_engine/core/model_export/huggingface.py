"""
HuggingFace Hub export and import provider.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import torch

from .base import (
    ExportProvider,
    ExportResult,
    ExportStatus,
    ImportProvider,
    ImportResult,
    ImportStatus,
    ProviderConfig,
)

logger = logging.getLogger(__name__)

# Check for dependencies
HAVE_HF_HUB = False
HfApi = None
create_repo = None
hf_hub_download = None
snapshot_download = None
list_models = None

try:
    from huggingface_hub import (
        HfApi,
        create_repo,
        list_models,
        snapshot_download,
    )
    HAVE_HF_HUB = True
except ImportError:
    pass  # Intentionally silent

HAVE_SAFETENSORS = False
try:
    from safetensors.torch import save_file as save_safetensors
    HAVE_SAFETENSORS = True
except ImportError:
    pass  # Intentionally silent


class HuggingFaceProvider(ExportProvider):
    """
    Export models to HuggingFace Hub.
    
    HuggingFace is the largest open-source model hub with millions of models.
    Your Enigma AI Engine models can be shared and discovered by the community.
    
    Usage:
        provider = HuggingFaceProvider()
        result = provider.export(
            "my_model",
            repo_id="username/my-model",
            token="hf_...",
            private=False
        )
    """
    
    NAME = "huggingface"
    DESCRIPTION = "Export to HuggingFace Hub - the largest open model repository"
    REQUIRES_AUTH = True
    AUTH_ENV_VAR = "HF_TOKEN"
    SUPPORTED_FORMATS = ["pytorch", "safetensors"]
    WEBSITE = "https://huggingface.co"
    
    def _convert_config_to_hf(
        self, 
        forge_config: dict[str, Any], 
        metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert Forge config to HuggingFace format."""
        return {
            "model_type": "forge",
            "architectures": ["ForgeForCausalLM"],
            "vocab_size": forge_config.get("vocab_size", 8000),
            "hidden_size": forge_config.get("dim", 512),
            "intermediate_size": forge_config.get("hidden_dim", 2048),
            "num_hidden_layers": forge_config.get("n_layers", 8),
            "num_attention_heads": forge_config.get("n_heads", 8),
            "num_key_value_heads": forge_config.get("n_kv_heads", 8),
            "max_position_embeddings": forge_config.get("max_seq_len", 1024),
            "hidden_act": "silu" if forge_config.get("use_swiglu", True) else "gelu",
            "use_rope": forge_config.get("use_rope", True),
            "use_rms_norm": forge_config.get("use_rms_norm", True),
            "rope_theta": forge_config.get("rope_theta", 10000.0),
            "torch_dtype": "float32",
            "tie_word_embeddings": False,
            "_forge_config": forge_config,
            "_forge_metadata": {
                "name": metadata.get("name", "unknown"),
                "created": metadata.get("created", "unknown"),
                "total_epochs": metadata.get("total_epochs", 0),
            }
        }
    
    def _create_model_card(
        self,
        output_path: Path,
        model_name: str,
        config: dict[str, Any],
        metadata: dict[str, Any]
    ):
        """Create README.md model card."""
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

Trained with [Enigma AI Engine](https://github.com/Enigma AI Engine/forge-ai).

## Model Details

| Property | Value |
|----------|-------|
| Parameters | {params_str} |
| Hidden Size | {config.get("dim", "?")} |
| Layers | {config.get("n_layers", "?")} |
| Heads | {config.get("n_heads", "?")} |
| Context | {config.get("max_seq_len", "?")} |

## Usage

```python
from enigma_engine.core.model_registry import ModelRegistry

registry = ModelRegistry()
model, config = registry.load_model("{model_name}")
```
"""
        with open(output_path / "README.md", "w") as f:
            f.write(card)
    
    def export_local(
        self,
        model_name: str,
        output_dir: str,
        use_safetensors: bool = True
    ) -> ExportResult:
        """Export to local directory in HuggingFace format."""
        try:
            model_path = self._get_model_path(model_name)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Load and convert config
            config = self._load_config(model_path)
            metadata = self._load_metadata(model_path)
            hf_config = self._convert_config_to_hf(config, metadata)
            
            with open(output_path / "config.json", "w") as f:
                json.dump(hf_config, f, indent=2)
            
            # Export weights
            weights_path = model_path / "weights.pth"
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                
                if use_safetensors and HAVE_SAFETENSORS:
                    save_safetensors(state_dict, output_path / "model.safetensors")
                else:
                    torch.save(state_dict, output_path / "pytorch_model.bin")
            
            # Create model card
            self._create_model_card(output_path, model_name, config, metadata)
            
            return ExportResult(
                status=ExportStatus.SUCCESS,
                provider=self.NAME,
                model_name=model_name,
                local_path=str(output_path),
                message="Exported to HuggingFace format"
            )
            
        except Exception as e:
            return ExportResult(
                status=ExportStatus.FAILED,
                provider=self.NAME,
                model_name=model_name,
                message=str(e)
            )
    
    def export(
        self,
        model_name: str,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        private: bool = False,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> ExportResult:
        """
        Export to HuggingFace Hub or local directory.
        
        Args:
            model_name: Enigma AI Engine model name
            repo_id: HuggingFace repo (e.g., "username/model-name")
            token: HuggingFace API token
            private: Make repo private
            output_dir: Local export directory (if not pushing to Hub)
        """
        # Local export only
        if output_dir and not repo_id:
            return self.export_local(model_name, output_dir)
        
        # Push to Hub
        if not HAVE_HF_HUB:
            return ExportResult(
                status=ExportStatus.FAILED,
                provider=self.NAME,
                model_name=model_name,
                message="huggingface_hub not installed. Run: pip install huggingface-hub"
            )
        
        if not repo_id:
            return ExportResult(
                status=ExportStatus.FAILED,
                provider=self.NAME,
                model_name=model_name,
                message="repo_id required for Hub upload"
            )
        
        try:
            token = self._check_auth(token)
            
            # Export to temp directory first
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                local_result = self.export_local(model_name, tmpdir)
                if not local_result.success:
                    return local_result
                
                # Create repo and upload
                api = HfApi(token=token)
                create_repo(repo_id, token=token, private=private, exist_ok=True)
                
                api.upload_folder(
                    folder_path=tmpdir,
                    repo_id=repo_id,
                    token=token,
                    commit_message=f"Upload Enigma AI Engine model {model_name}"
                )
                
                url = f"https://huggingface.co/{repo_id}"
                return ExportResult(
                    status=ExportStatus.SUCCESS,
                    provider=self.NAME,
                    model_name=model_name,
                    url=url,
                    message=f"Uploaded to {url}"
                )
                
        except Exception as e:
            return ExportResult(
                status=ExportStatus.FAILED,
                provider=self.NAME,
                model_name=model_name,
                message=str(e)
            )


class HuggingFaceImporter(ImportProvider):
    """
    Import models from HuggingFace Hub.
    
    Download any public model from HuggingFace and register it locally.
    
    Usage:
        importer = HuggingFaceImporter()
        
        # Search for models
        results = importer.search("llama chat")
        
        # Import a model
        result = importer.import_model(
            "microsoft/DialoGPT-small",
            local_name="dialogpt"
        )
    """
    
    NAME = "huggingface"
    DESCRIPTION = "Import from HuggingFace Hub - millions of open models"
    REQUIRES_AUTH = False  # Only needed for private models
    AUTH_ENV_VAR = "HF_TOKEN"
    SUPPORTED_FORMATS = ["pytorch", "safetensors", "gguf"]
    WEBSITE = "https://huggingface.co"
    
    @classmethod
    def get_config(cls) -> ProviderConfig:
        return ProviderConfig(
            name=cls.NAME,
            description=cls.DESCRIPTION,
            requires_auth=cls.REQUIRES_AUTH,
            auth_env_var=cls.AUTH_ENV_VAR,
            supported_formats=cls.SUPPORTED_FORMATS,
            website=cls.WEBSITE,
            can_export=True,
            can_import=True,
        )
    
    def search(
        self,
        query: str,
        limit: int = 10,
        filter_task: Optional[str] = "text-generation",
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Search HuggingFace Hub for models.
        
        Args:
            query: Search query
            limit: Max results
            filter_task: Filter by task (text-generation, text-classification, etc.)
        """
        if not HAVE_HF_HUB:
            logger.warning("huggingface_hub not installed")
            return []
        
        try:
            models = list_models(
                search=query,
                limit=limit,
                task=filter_task,
                sort="downloads",
                direction=-1,
            )
            
            return [
                {
                    "id": m.id,
                    "name": m.id.split("/")[-1],
                    "author": m.author,
                    "downloads": getattr(m, "downloads", 0),
                    "likes": getattr(m, "likes", 0),
                    "task": filter_task,
                    "url": f"https://huggingface.co/{m.id}",
                }
                for m in models
            ]
        except Exception as e:
            logger.warning(f"HuggingFace search failed: {e}")
            return []
    
    def import_model(
        self,
        source_id: str,
        local_name: Optional[str] = None,
        token: Optional[str] = None,
        revision: str = "main",
        **kwargs
    ) -> ImportResult:
        """
        Import a model from HuggingFace Hub.
        
        Args:
            source_id: HuggingFace model ID (e.g., "microsoft/DialoGPT-small")
            local_name: Local name (default: model name from ID)
            token: HuggingFace token (for private models)
            revision: Git revision to download
        """
        if not HAVE_HF_HUB:
            return ImportResult(
                status=ImportStatus.FAILED,
                provider=self.NAME,
                model_name=local_name or source_id,
                source_id=source_id,
                message="huggingface_hub not installed. Run: pip install huggingface-hub"
            )
        
        try:
            # Determine local name
            if not local_name:
                local_name = source_id.replace("/", "_").lower()
            
            local_name = local_name.lower().strip().replace(" ", "_")
            model_path = self.models_dir / local_name
            
            # Check if already exists
            if model_path.exists():
                return ImportResult(
                    status=ImportStatus.ALREADY_EXISTS,
                    provider=self.NAME,
                    model_name=local_name,
                    source_id=source_id,
                    local_path=str(model_path),
                    message=f"Model already exists at {model_path}"
                )
            
            # Create model directory
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Download model files
            logger.info(f"Downloading {source_id} from HuggingFace...")
            
            try:
                # Try to get auth token
                auth_token = None
                try:
                    auth_token = self._check_auth(token)
                except ValueError:
                    pass  # No auth, that's fine for public models
                
                # Download the full model
                downloaded_path = snapshot_download(
                    repo_id=source_id,
                    revision=revision,
                    token=auth_token,
                    local_dir=str(model_path),
                    local_dir_use_symlinks=False,
                )
                
                logger.info(f"Downloaded to {downloaded_path}")
                
            except Exception as e:
                # Clean up on failure
                if model_path.exists():
                    shutil.rmtree(model_path)
                raise e
            
            # Register in Enigma AI Engine
            self._register_model(
                local_name=local_name,
                model_path=model_path,
                source="huggingface",
                source_id=source_id,
                metadata={
                    "huggingface_id": source_id,
                    "revision": revision,
                }
            )
            
            return ImportResult(
                status=ImportStatus.SUCCESS,
                provider=self.NAME,
                model_name=local_name,
                source_id=source_id,
                local_path=str(model_path),
                message=f"Imported {source_id} as '{local_name}'"
            )
            
        except Exception as e:
            logger.exception("HuggingFace import failed")
            return ImportResult(
                status=ImportStatus.FAILED,
                provider=self.NAME,
                model_name=local_name or source_id,
                source_id=source_id,
                message=str(e)
            )