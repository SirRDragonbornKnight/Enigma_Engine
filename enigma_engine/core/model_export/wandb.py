"""
Weights & Biases export and import provider.

W&B is an ML experiment tracking and model registry platform.
"""

import logging
from typing import Any, Optional


from .base import (
    ExportProvider,
    ExportResult,
    ExportStatus,
    ImportProvider,
    ImportResult,
    ImportStatus,
)

logger = logging.getLogger(__name__)

# Check for wandb
HAVE_WANDB = False
wandb = None
try:
    import wandb
    HAVE_WANDB = True
except ImportError:
    pass


class WandBProvider(ExportProvider):
    """
    Export models to Weights & Biases Model Registry.
    
    W&B provides:
    - Model versioning and lineage tracking
    - Team collaboration on models
    - Model deployment integrations
    - Experiment tracking
    
    Requirements:
        pip install wandb
    
    Usage:
        provider = WandBProvider()
        result = provider.export(
            "my_model",
            project="my-project",
            entity="my-team",  # Optional
            token="wandb_..."
        )
    """
    
    NAME = "wandb"
    DESCRIPTION = "Export to Weights & Biases - ML experiment tracking & model registry"
    REQUIRES_AUTH = True
    AUTH_ENV_VAR = "WANDB_API_KEY"
    SUPPORTED_FORMATS = ["pytorch", "safetensors"]
    WEBSITE = "https://wandb.ai"
    
    def export(
        self,
        model_name: str,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        token: Optional[str] = None,
        output_dir: Optional[str] = None,
        tags: Optional[list[str]] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> ExportResult:
        """
        Export to Weights & Biases.
        
        Args:
            model_name: Enigma AI Engine model name
            project: W&B project name
            entity: W&B entity (team/user)
            token: W&B API key
            output_dir: Local directory (for local-only export)
            tags: Tags for the model artifact
            description: Model description
        """
        if not HAVE_WANDB:
            return ExportResult(
                status=ExportStatus.FAILED,
                provider=self.NAME,
                model_name=model_name,
                message="wandb not installed. Run: pip install wandb"
            )
        
        try:
            model_path = self._get_model_path(model_name)
            config = self._load_config(model_path)
            metadata = self._load_metadata(model_path)
            
            # Get auth token
            token = self._check_auth(token)
            if token:
                wandb.login(key=token)
            
            # Set project name
            project = project or f"Enigma AI Engine-{model_name}"
            
            # Initialize W&B run
            run = wandb.init(
                project=project,
                entity=entity,
                job_type="model-export",
                config={
                    "model_name": model_name,
                    "forge_config": config,
                    "metadata": metadata,
                },
                tags=tags or ["Enigma AI Engine", "export"],
            )
            
            # Create model artifact
            artifact = wandb.Artifact(
                name=model_name,
                type="model",
                description=description or metadata.get("description", f"Enigma AI Engine model: {model_name}"),
                metadata={
                    "framework": "Enigma AI Engine",
                    "architecture": "forge-transformer",
                    **metadata
                }
            )
            
            # Add config file
            config_path = model_path / "config.json"
            if config_path.exists():
                artifact.add_file(str(config_path), name="config.json")
            
            # Add metadata file
            metadata_path = model_path / "metadata.json"
            if metadata_path.exists():
                artifact.add_file(str(metadata_path), name="metadata.json")
            
            # Add weights
            weights_path = model_path / "weights.pth"
            if weights_path.exists():
                artifact.add_file(str(weights_path), name="weights.pth")
            
            # Add any checkpoints
            checkpoints_dir = model_path / "checkpoints"
            if checkpoints_dir.exists():
                for ckpt in checkpoints_dir.glob("*.pth"):
                    artifact.add_file(str(ckpt), name=f"checkpoints/{ckpt.name}")
            
            # Log the artifact
            run.log_artifact(artifact)
            
            # Get artifact URL
            artifact_url = f"https://wandb.ai/{entity or run.entity}/{project}/artifacts/model/{model_name}"
            
            # Finish run
            run.finish()
            
            return ExportResult(
                status=ExportStatus.SUCCESS,
                provider=self.NAME,
                model_name=model_name,
                url=artifact_url,
                message=f"Uploaded to W&B: {artifact_url}",
                details={
                    "project": project,
                    "entity": entity or run.entity,
                    "artifact_name": model_name,
                }
            )
            
        except Exception as e:
            logger.exception("W&B export failed")
            return ExportResult(
                status=ExportStatus.FAILED,
                provider=self.NAME,
                model_name=model_name,
                message=str(e)
            )
    
    def list_models(
        self,
        project: str,
        entity: Optional[str] = None,
        token: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """List models in a W&B project."""
        if not HAVE_WANDB:
            return []
        
        try:
            token = self._check_auth(token)
            if token:
                wandb.login(key=token)
            
            api = wandb.Api()
            artifacts = api.artifacts(
                type_name="model",
                name=f"{entity}/{project}" if entity else project
            )
            
            return [
                {
                    "name": a.name,
                    "version": a.version,
                    "created_at": str(a.created_at),
                    "size": a.size,
                    "metadata": a.metadata,
                }
                for a in artifacts
            ]
        except Exception as e:
            logger.warning(f"Failed to list W&B models: {e}")
            return []


class WandBImporter(ImportProvider):
    """
    Import models from Weights & Biases Model Registry.
    
    Download model artifacts from W&B projects.
    
    Usage:
        importer = WandBImporter()
        
        # List models in a project
        models = importer.list_models(project="my-project")
        
        # Import a model artifact
        result = importer.import_model(
            "entity/project/model:v1",
            local_name="my_model"
        )
    """
    
    NAME = "wandb"
    DESCRIPTION = "Import from Weights & Biases Model Registry"
    REQUIRES_AUTH = True
    AUTH_ENV_VAR = "WANDB_API_KEY"
    SUPPORTED_FORMATS = ["pytorch", "safetensors"]
    WEBSITE = "https://wandb.ai"
    
    def search(
        self,
        query: str,
        limit: int = 10,
        entity: Optional[str] = None,
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Search for models in W&B.
        
        Note: W&B search is project-scoped, so this searches within
        your accessible projects.
        """
        if not HAVE_WANDB:
            return []
        
        # W&B doesn't have global search, return empty
        # User needs to specify project
        logger.info("W&B requires specifying a project. Use list_models(project='name') instead.")
        return []
    
    def list_models(
        self,
        project: str,
        entity: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs
    ) -> list[dict[str, Any]]:
        """List model artifacts in a W&B project."""
        if not HAVE_WANDB:
            return []
        
        try:
            token = self._check_auth(token)
            if token:
                wandb.login(key=token)
            
            api = wandb.Api()
            
            # Build artifact path
            if entity:
                path = f"{entity}/{project}"
            else:
                path = project
            
            artifacts = api.artifacts(type_name="model", name=path)
            
            return [
                {
                    "id": f"{a.entity}/{a.project}/{a.name}:{a.version}",
                    "name": a.name,
                    "version": a.version,
                    "created_at": str(a.created_at),
                    "size": getattr(a, "size", 0),
                    "metadata": getattr(a, "metadata", {}),
                }
                for a in artifacts
            ]
        except Exception as e:
            logger.warning(f"Failed to list W&B models: {e}")
            return []
    
    def import_model(
        self,
        source_id: str,
        local_name: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs
    ) -> ImportResult:
        """
        Import a model artifact from W&B.
        
        Args:
            source_id: W&B artifact path (e.g., "entity/project/model:v1")
            local_name: Local name for the model
            token: W&B API key
        """
        if not HAVE_WANDB:
            return ImportResult(
                status=ImportStatus.FAILED,
                provider=self.NAME,
                model_name=local_name or source_id,
                source_id=source_id,
                message="wandb not installed. Run: pip install wandb"
            )
        
        try:
            token = self._check_auth(token)
            if token:
                wandb.login(key=token)
            
            # Determine local name
            if not local_name:
                # Extract name from artifact path
                parts = source_id.split("/")
                local_name = parts[-1].split(":")[0] if parts else source_id
                local_name = f"wandb_{local_name}".lower()
            
            local_name = local_name.lower().strip().replace(" ", "_")
            model_path = self.models_dir / local_name
            
            if model_path.exists():
                return ImportResult(
                    status=ImportStatus.ALREADY_EXISTS,
                    provider=self.NAME,
                    model_name=local_name,
                    source_id=source_id,
                    local_path=str(model_path),
                    message=f"Model already exists at {model_path}"
                )
            
            # Download artifact
            logger.info(f"Downloading {source_id} from W&B...")
            
            api = wandb.Api()
            artifact = api.artifact(source_id)
            
            model_path.mkdir(parents=True, exist_ok=True)
            artifact.download(root=str(model_path))
            
            # Register
            self._register_model(
                local_name=local_name,
                model_path=model_path,
                source="wandb",
                source_id=source_id,
                metadata={
                    "wandb_artifact": source_id,
                    "artifact_metadata": getattr(artifact, "metadata", {}),
                }
            )
            
            return ImportResult(
                status=ImportStatus.SUCCESS,
                provider=self.NAME,
                model_name=local_name,
                source_id=source_id,
                local_path=str(model_path),
                message=f"Downloaded {source_id} from W&B"
            )
            
        except Exception as e:
            logger.exception("W&B import failed")
            return ImportResult(
                status=ImportStatus.FAILED,
                provider=self.NAME,
                model_name=local_name or source_id,
                source_id=source_id,
                message=str(e)
            )