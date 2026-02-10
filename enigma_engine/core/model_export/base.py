"""
Base classes for model export and import providers.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ExportStatus(Enum):
    """Status of an export operation."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # Some files exported, some failed
    PENDING = "pending"  # For async exports


class ImportStatus(Enum):
    """Status of an import operation."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    PENDING = "pending"
    ALREADY_EXISTS = "already_exists"


@dataclass
class ExportResult:
    """Result of an export operation."""
    status: ExportStatus
    provider: str
    model_name: str
    url: Optional[str] = None  # URL to access the exported model
    local_path: Optional[str] = None  # Local path if exported locally
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.status == ExportStatus.SUCCESS
    
    def __str__(self) -> str:
        if self.success:
            location = self.url or self.local_path or "unknown"
            return f"✓ Exported {self.model_name} to {self.provider}: {location}"
        return f"✗ Export failed for {self.model_name} to {self.provider}: {self.message}"


@dataclass
class ImportResult:
    """Result of an import operation."""
    status: ImportStatus
    provider: str
    model_name: str
    source_id: str = ""  # Original ID on the platform
    local_path: Optional[str] = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.status in (ImportStatus.SUCCESS, ImportStatus.ALREADY_EXISTS)
    
    def __str__(self) -> str:
        if self.success:
            return f"✓ Imported {self.model_name} from {self.provider}: {self.local_path}"
        return f"✗ Import failed for {self.model_name} from {self.provider}: {self.message}"


@dataclass 
class ProviderConfig:
    """Configuration for an export/import provider."""
    name: str
    description: str
    requires_auth: bool = False
    auth_env_var: Optional[str] = None  # Environment variable for auth token
    supported_formats: list[str] = field(default_factory=lambda: ["pytorch"])
    website: Optional[str] = None
    can_export: bool = True
    can_import: bool = True
    

class ExportProvider(ABC):
    """
    Base class for model export providers.
    
    Each provider implements export to a specific platform.
    """
    
    # Override these in subclasses
    NAME: str = "base"
    DESCRIPTION: str = "Base export provider"
    REQUIRES_AUTH: bool = False
    AUTH_ENV_VAR: Optional[str] = None
    SUPPORTED_FORMATS: list[str] = ["pytorch"]
    WEBSITE: Optional[str] = None
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the provider.
        
        Args:
            models_dir: Path to Enigma AI Engine models directory
        """
        from enigma_engine.config import CONFIG
        self.models_dir = Path(models_dir or CONFIG["models_dir"])
    
    @classmethod
    def get_config(cls) -> ProviderConfig:
        """Get provider configuration."""
        return ProviderConfig(
            name=cls.NAME,
            description=cls.DESCRIPTION,
            requires_auth=cls.REQUIRES_AUTH,
            auth_env_var=cls.AUTH_ENV_VAR,
            supported_formats=cls.SUPPORTED_FORMATS,
            website=cls.WEBSITE,
        )
    
    def _get_model_path(self, model_name: str) -> Path:
        """Get the path to a model directory."""
        model_path = self.models_dir / model_name
        if not model_path.exists():
            raise ValueError(f"Model '{model_name}' not found at {model_path}")
        return model_path
    
    def _load_config(self, model_path: Path) -> dict[str, Any]:
        """Load model configuration."""
        import json
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json found at {config_path}")
        with open(config_path) as f:
            return json.load(f)
    
    def _load_metadata(self, model_path: Path) -> dict[str, Any]:
        """Load model metadata."""
        import json
        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}
    
    def _load_weights(self, model_path: Path, device: str = "cpu") -> dict[str, Any]:
        """Load model weights."""
        import torch
        weights_path = model_path / "weights.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"No weights.pth found at {weights_path}")
        return torch.load(weights_path, map_location=device, weights_only=True)
    
    def _check_auth(self, token: Optional[str] = None) -> str:
        """Check for authentication token."""
        import os
        if token:
            return token
        if self.AUTH_ENV_VAR:
            token = os.environ.get(self.AUTH_ENV_VAR)
            if token:
                return token
        if self.REQUIRES_AUTH:
            raise ValueError(
                f"{self.NAME} requires authentication. "
                f"Provide token= argument or set {self.AUTH_ENV_VAR} environment variable."
            )
        return ""
    
    @abstractmethod
    def export(
        self,
        model_name: str,
        **kwargs
    ) -> ExportResult:
        """
        Export a model to this provider.
        
        Args:
            model_name: Name of the model in Enigma AI Engine registry
            **kwargs: Provider-specific arguments
            
        Returns:
            ExportResult with status and location
        """
    
    def validate(self, model_name: str) -> bool:
        """
        Validate that a model can be exported.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model can be exported
        """
        try:
            model_path = self._get_model_path(model_name)
            self._load_config(model_path)
            return True
        except Exception as e:
            logger.warning(f"Model validation failed: {e}")
            return False


class ImportProvider(ABC):
    """
    Base class for model import providers.
    
    Each provider implements import from a specific platform.
    """
    
    NAME: str = "base"
    DESCRIPTION: str = "Base import provider"
    REQUIRES_AUTH: bool = False
    AUTH_ENV_VAR: Optional[str] = None
    SUPPORTED_FORMATS: list[str] = ["pytorch"]
    WEBSITE: Optional[str] = None
    
    def __init__(self, models_dir: Optional[str] = None):
        from enigma_engine.config import CONFIG
        self.models_dir = Path(models_dir or CONFIG["models_dir"])
    
    @classmethod
    def get_config(cls) -> ProviderConfig:
        return ProviderConfig(
            name=cls.NAME,
            description=cls.DESCRIPTION,
            requires_auth=cls.REQUIRES_AUTH,
            auth_env_var=cls.AUTH_ENV_VAR,
            supported_formats=cls.SUPPORTED_FORMATS,
            website=cls.WEBSITE,
            can_export=False,
            can_import=True,
        )
    
    def _check_auth(self, token: Optional[str] = None) -> str:
        import os
        if token:
            return token
        if self.AUTH_ENV_VAR:
            token = os.environ.get(self.AUTH_ENV_VAR)
            if token:
                return token
        if self.REQUIRES_AUTH:
            raise ValueError(
                f"{self.NAME} requires authentication. "
                f"Provide token= argument or set {self.AUTH_ENV_VAR} environment variable."
            )
        return ""
    
    def _register_model(
        self,
        local_name: str,
        model_path: Path,
        source: str,
        source_id: str,
        metadata: Optional[dict[str, Any]] = None
    ):
        """Register an imported model in the Enigma AI Engine registry."""
        import json
        from datetime import datetime

        from ..model_registry import ModelRegistry
        
        registry = ModelRegistry(str(self.models_dir))
        
        # Create metadata file
        meta = {
            "name": local_name,
            "source": source,
            "source_id": source_id,
            "imported_at": datetime.now().isoformat(),
            **(metadata or {})
        }
        with open(model_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        # Update registry
        registry.registry["models"][local_name] = {
            "path": str(model_path),
            "source": source,
            "source_id": source_id,
            "created": meta["imported_at"],
            "has_weights": (model_path / "weights.pth").exists() or 
                          (model_path / "pytorch_model.bin").exists() or
                          (model_path / "model.safetensors").exists(),
        }
        registry._save_registry()
    
    @abstractmethod
    def import_model(
        self,
        source_id: str,
        local_name: Optional[str] = None,
        **kwargs
    ) -> ImportResult:
        """
        Import a model from this provider.
        
        Args:
            source_id: Model identifier on the platform
            local_name: Local name to save as (default: derived from source_id)
            **kwargs: Provider-specific arguments
            
        Returns:
            ImportResult with status and location
        """
    
    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Search for models on the platform.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of model info dicts
        """
    
    def list_models(self, **kwargs) -> list[dict[str, Any]]:
        """List available models (if supported)."""
        return []