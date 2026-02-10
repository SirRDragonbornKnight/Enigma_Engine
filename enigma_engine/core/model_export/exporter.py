"""
Unified Model Hub
=================

Central interface for importing AND exporting Enigma AI Engine models to various platforms.
Bidirectional model transfer - share your models AND use community models.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from .base import (
    ExportProvider,
    ExportResult,
    ExportStatus,
    ImportProvider,
    ImportResult,
    ProviderConfig,
)
from .huggingface import HuggingFaceImporter, HuggingFaceProvider
from .ollama import OllamaImporter, OllamaProvider
from .onnx import ONNXProvider
from .replicate import ReplicateImporter, ReplicateProvider
from .wandb import WandBImporter, WandBProvider

logger = logging.getLogger(__name__)


# Registry of all export providers
EXPORT_PROVIDERS: dict[str, type[ExportProvider]] = {
    "huggingface": HuggingFaceProvider,
    "hf": HuggingFaceProvider,  # Alias
    "replicate": ReplicateProvider,
    "ollama": OllamaProvider,
    "wandb": WandBProvider,
    "wb": WandBProvider,  # Alias
    "onnx": ONNXProvider,
}

# Registry of all import providers
IMPORT_PROVIDERS: dict[str, type[ImportProvider]] = {
    "huggingface": HuggingFaceImporter,
    "hf": HuggingFaceImporter,  # Alias
    "replicate": ReplicateImporter,
    "ollama": OllamaImporter,
    "wandb": WandBImporter,
    "wb": WandBImporter,  # Alias
}

# Keep old name for backwards compatibility
PROVIDERS = EXPORT_PROVIDERS


class ModelHub:
    """
    Unified interface for IMPORTING and EXPORTING Enigma AI Engine models.
    
    Bidirectional model transfer to/from multiple platforms:
    - HuggingFace Hub (share models, download community models)
    - Replicate (API-based model serving, use cloud models)
    - Ollama (local model server, pull popular models)
    - Weights & Biases (ML experiment tracking, model artifacts)
    - ONNX (export only - cross-platform deployment)
    
    Usage:
        hub = ModelHub()
        
        # === IMPORTING (Get models FROM platforms) ===
        
        # Search for models
        results = hub.search("llama", provider="huggingface")
        results = hub.search("llama", provider="ollama")
        
        # Import a model
        result = hub.import_model(
            "meta-llama/Llama-2-7b",
            provider="huggingface",
            local_name="llama2-7b"
        )
        
        # Import from Ollama
        result = hub.import_model(
            "llama2:7b",
            provider="ollama"
        )
        
        # === EXPORTING (Push models TO platforms) ===
        
        # Export to HuggingFace
        result = hub.export(
            "my_model",
            provider="huggingface",
            repo_id="username/my-model"
        )
        
        # Export to multiple platforms at once
        results = hub.export_all(
            "my_model",
            providers=["huggingface", "ollama", "onnx"],
            huggingface_kwargs={"repo_id": "user/model"},
        )
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the model hub.
        
        Args:
            models_dir: Path to Enigma AI Engine models directory
        """
        from ...config import CONFIG
        self.models_dir = Path(models_dir or CONFIG["models_dir"])
        self._export_providers: dict[str, ExportProvider] = {}
        self._import_providers: dict[str, ImportProvider] = {}
    
    # ===================
    # Provider Management
    # ===================
    
    def _get_export_provider(self, name: str) -> ExportProvider:
        """Get or create an export provider instance."""
        name = name.lower()
        if name not in EXPORT_PROVIDERS:
            available = list({p.NAME for p in EXPORT_PROVIDERS.values()})
            raise ValueError(f"Unknown export provider: {name}. Available: {available}")
        
        if name not in self._export_providers:
            provider_class = EXPORT_PROVIDERS[name]
            self._export_providers[name] = provider_class(str(self.models_dir))
        
        return self._export_providers[name]
    
    def _get_import_provider(self, name: str) -> ImportProvider:
        """Get or create an import provider instance."""
        name = name.lower()
        if name not in IMPORT_PROVIDERS:
            available = list({p.NAME for p in IMPORT_PROVIDERS.values()})
            raise ValueError(f"Unknown import provider: {name}. Available: {available}")
        
        if name not in self._import_providers:
            provider_class = IMPORT_PROVIDERS[name]
            self._import_providers[name] = provider_class(str(self.models_dir))
        
        return self._import_providers[name]
    
    def list_export_providers(self) -> list[ProviderConfig]:
        """List all available export providers."""
        seen = set()
        configs = []
        for provider_class in EXPORT_PROVIDERS.values():
            if provider_class.NAME not in seen:
                configs.append(provider_class.get_config())
                seen.add(provider_class.NAME)
        return configs
    
    def list_import_providers(self) -> list[str]:
        """List all available import providers."""
        seen = set()
        names = []
        for provider_class in IMPORT_PROVIDERS.values():
            if provider_class.NAME not in seen:
                names.append(provider_class.NAME)
                seen.add(provider_class.NAME)
        return names
    
    # ====================
    # IMPORT Functionality
    # ====================
    
    def search(
        self,
        query: str,
        provider: str = "huggingface",
        limit: int = 20,
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Search for models on a platform.
        
        Args:
            query: Search query
            provider: Platform to search ("huggingface", "ollama", "replicate", "wandb")
            limit: Maximum results to return
            **kwargs: Provider-specific arguments
            
        Returns:
            List of model info dicts
            
        Example:
            # Search HuggingFace
            models = hub.search("llama", provider="huggingface")
            
            # Search Ollama
            models = hub.search("mistral", provider="ollama")
        """
        importer = self._get_import_provider(provider)
        return importer.search(query, limit=limit, **kwargs)
    
    def import_model(
        self,
        source_id: str,
        provider: str = "huggingface",
        local_name: Optional[str] = None,
        **kwargs
    ) -> ImportResult:
        """
        Import a model FROM a platform.
        
        Args:
            source_id: Model identifier on the platform
            provider: Platform to import from
            local_name: Local name for the model (default: derived from source)
            **kwargs: Provider-specific arguments
            
        Returns:
            ImportResult with status and local path
            
        Example:
            # Import from HuggingFace
            result = hub.import_model(
                "microsoft/DialoGPT-small",
                provider="huggingface",
                local_name="dialogpt"
            )
            
            # Import from Ollama
            result = hub.import_model(
                "llama2:7b",
                provider="ollama"
            )
            
            # Import from Replicate (creates API wrapper)
            result = hub.import_model(
                "meta/llama-2-70b-chat",
                provider="replicate",
                local_name="llama2-api"
            )
        """
        importer = self._get_import_provider(provider)
        
        logger.info(f"Importing {source_id} from {provider}...")
        result = importer.import_model(source_id, local_name=local_name, **kwargs)
        
        if result.success:
            logger.info(f"Import successful: {result.local_path}")
        else:
            logger.error(f"Import failed: {result.message}")
        
        return result
    
    def list_available(self, provider: str = "ollama", **kwargs) -> list[dict[str, Any]]:
        """
        List models available on a platform.
        
        For Ollama, lists locally installed models.
        For W&B, lists models in a project.
        
        Args:
            provider: Platform to query
            **kwargs: Provider-specific arguments (e.g., project for W&B)
            
        Returns:
            List of available models
        """
        importer = self._get_import_provider(provider)
        
        # Use list_models if available (Ollama, W&B)
        if hasattr(importer, 'list_models'):
            return importer.list_models(**kwargs)
        
        # Otherwise return empty
        return []
    
    # ====================
    # EXPORT Functionality
    # ====================
    
    def get_provider_info(self, name: str) -> ProviderConfig:
        """Get information about a specific export provider."""
        provider = self._get_export_provider(name)
        return provider.get_config()
    
    def list_exportable_models(self) -> dict[str, dict[str, Any]]:
        """
        List all Enigma AI Engine models that can be exported.
        
        Returns:
            Dict of model names to their info
        """
        from ..model_registry import ModelRegistry
        
        registry = ModelRegistry(str(self.models_dir))
        models = registry.list_models()
        
        exportable = {}
        for name, info in models.items():
            # Skip HuggingFace models (already external)
            if info.get("source") == "huggingface":
                continue
            
            model_path = Path(info.get("path", ""))
            has_weights = (model_path / "weights.pth").exists() if model_path.exists() else False
            
            exportable[name] = {
                **info,
                "has_weights": has_weights,
                "exportable": has_weights,
            }
        
        return exportable
    
    def validate(self, model_name: str, provider: str) -> bool:
        """
        Check if a model can be exported to a provider.
        
        Args:
            model_name: Name of the model
            provider: Export provider name
            
        Returns:
            True if export is possible
        """
        provider_instance = self._get_export_provider(provider)
        return provider_instance.validate(model_name)
    
    def export(
        self,
        model_name: str,
        provider: str,
        **kwargs
    ) -> ExportResult:
        """
        Export a model to a specific provider.
        
        Args:
            model_name: Name of the Enigma AI Engine model
            provider: Export provider ("huggingface", "replicate", "ollama", "wandb", "onnx")
            **kwargs: Provider-specific arguments
            
        Returns:
            ExportResult with status and location
            
        Example:
            # HuggingFace
            exporter.export("my_model", "huggingface", repo_id="user/model")
            
            # Replicate
            exporter.export("my_model", "replicate", output_dir="./pkg")
            
            # Ollama
            exporter.export("my_model", "ollama", quantization="q4_k_m")
            
            # W&B
            exporter.export("my_model", "wandb", project="my-project")
            
            # ONNX
            exporter.export("my_model", "onnx", optimize=True)
        """
        provider_instance = self._get_export_provider(provider)
        
        logger.info(f"Exporting {model_name} to {provider}...")
        result = provider_instance.export(model_name, **kwargs)
        
        if result.success:
            logger.info(f"Export successful: {result}")
        else:
            logger.error(f"Export failed: {result}")
        
        return result
    
    def export_all(
        self,
        model_name: str,
        providers: Optional[list[str]] = None,
        **provider_kwargs
    ) -> dict[str, ExportResult]:
        """
        Export a model to multiple providers.
        
        Args:
            model_name: Name of the model
            providers: List of providers (default: all non-cloud providers)
            **provider_kwargs: Provider-specific kwargs as {provider}_kwargs dicts
            
        Returns:
            Dict mapping provider names to ExportResults
            
        Example:
            results = exporter.export_all(
                "my_model",
                providers=["huggingface", "ollama", "onnx"],
                huggingface_kwargs={"repo_id": "user/model"},
                ollama_kwargs={"quantization": "q4_0"},
            )
        """
        if providers is None:
            # Default to local-only providers
            providers = ["ollama", "onnx"]
        
        results = {}
        for provider in providers:
            # Get provider-specific kwargs
            kwargs = provider_kwargs.get(f"{provider}_kwargs", {})
            
            try:
                results[provider] = self.export(model_name, provider, **kwargs)
            except Exception as e:
                results[provider] = ExportResult(
                    status=ExportStatus.FAILED,
                    provider=provider,
                    model_name=model_name,
                    message=str(e)
                )
        
        return results


# Backwards compatibility alias
ModelExporter = ModelHub


# Singleton instance
_hub: Optional[ModelHub] = None


def get_hub(models_dir: Optional[str] = None) -> ModelHub:
    """Get the global ModelHub instance."""
    global _hub
    if _hub is None:
        _hub = ModelHub(models_dir)
    return _hub


# Backwards compatibility
def get_exporter(models_dir: Optional[str] = None) -> ModelHub:
    """Get the global ModelHub instance (alias for backwards compatibility)."""
    return get_hub(models_dir)


# ===================
# Convenience Functions
# ===================

def export_model(
    model_name: str,
    provider: str,
    **kwargs
) -> ExportResult:
    """
    Quick export function.
    
    Example:
        from enigma_engine.core.model_export import export_model
        
        result = export_model("my_model", "huggingface", repo_id="user/model")
    """
    return get_hub().export(model_name, provider, **kwargs)


def import_model(
    source_id: str,
    provider: str = "huggingface",
    local_name: Optional[str] = None,
    **kwargs
) -> ImportResult:
    """
    Quick import function.
    
    Example:
        from enigma_engine.core.model_export import import_model
        
        # Import from HuggingFace
        result = import_model("microsoft/DialoGPT-small", "huggingface", local_name="dialogpt")
        
        # Import from Ollama
        result = import_model("llama2:7b", "ollama")
    """
    return get_hub().import_model(source_id, provider, local_name=local_name, **kwargs)


def search_models(
    query: str,
    provider: str = "huggingface",
    limit: int = 20,
    **kwargs
) -> list[dict[str, Any]]:
    """
    Quick search function.
    
    Example:
        from enigma_engine.core.model_export import search_models
        
        # Search HuggingFace
        models = search_models("llama", "huggingface")
        
        # Search Ollama
        models = search_models("mistral", "ollama")
    """
    return get_hub().search(query, provider, limit=limit, **kwargs)


def list_export_providers() -> list[ProviderConfig]:
    """List all available export providers."""
    return get_hub().list_export_providers()


def list_import_providers() -> list[str]:
    """List all available import providers."""
    return get_hub().list_import_providers()
