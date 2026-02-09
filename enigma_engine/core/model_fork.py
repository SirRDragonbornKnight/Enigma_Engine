"""
Model Fork/Inheritance System for Enigma AI Engine

Fork existing AIs, keep attribution and lineage.

Features:
- Model forking with attribution
- Lineage tracking
- License inheritance
- Diff tracking
- Merge capabilities

Usage:
    from enigma_engine.core.model_fork import ModelFork
    
    fork = ModelFork()
    
    # Fork a model
    forked = fork.create_fork(
        source_model="models/base-small",
        fork_name="my-specialized",
        creator="username"
    )
    
    # View lineage
    lineage = fork.get_lineage(forked.model_id)
"""

import hashlib
import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelAttribution:
    """Attribution information for a model."""
    name: str
    creator: str
    license: str
    source_url: Optional[str] = None
    citation: Optional[str] = None


@dataclass
class ModelLineage:
    """Lineage information for a forked model."""
    model_id: str
    parent_id: Optional[str]
    name: str
    creator: str
    created_at: float
    
    # Attribution chain
    attributions: List[ModelAttribution] = field(default_factory=list)
    
    # Training info
    training_data_hash: Optional[str] = None
    base_model_hash: Optional[str] = None
    
    # License
    license: str = "Apache-2.0"
    
    # Changes from parent
    changes: List[str] = field(default_factory=list)


@dataclass
class ForkedModel:
    """A forked model."""
    model_id: str
    name: str
    path: Path
    lineage: ModelLineage
    
    # Metadata
    description: str = ""
    version: str = "0.1.0"
    tags: List[str] = field(default_factory=list)


class ModelFork:
    """Model forking and inheritance manager."""
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize model fork manager.
        
        Args:
            models_dir: Directory containing models
        """
        self.models_dir = models_dir or Path("models")
        self._registry: Dict[str, ForkedModel] = {}
        self._lineage_file = self.models_dir / "lineage_registry.json"
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self):
        """Load existing lineage registry."""
        if self._lineage_file.exists():
            try:
                with open(self._lineage_file) as f:
                    data = json.load(f)
                    for model_id, info in data.items():
                        lineage = ModelLineage(
                            model_id=info["model_id"],
                            parent_id=info.get("parent_id"),
                            name=info["name"],
                            creator=info["creator"],
                            created_at=info["created_at"],
                            license=info.get("license", "Apache-2.0"),
                            changes=info.get("changes", [])
                        )
                        
                        # Load attributions
                        for attr_data in info.get("attributions", []):
                            lineage.attributions.append(ModelAttribution(**attr_data))
                        
                        self._registry[model_id] = ForkedModel(
                            model_id=model_id,
                            name=info["name"],
                            path=Path(info.get("path", "")),
                            lineage=lineage,
                            description=info.get("description", ""),
                            version=info.get("version", "0.1.0"),
                            tags=info.get("tags", [])
                        )
            except Exception as e:
                logger.error(f"Failed to load lineage registry: {e}")
    
    def _save_registry(self):
        """Save lineage registry."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        data = {}
        for model_id, model in self._registry.items():
            data[model_id] = {
                "model_id": model.model_id,
                "name": model.name,
                "path": str(model.path),
                "parent_id": model.lineage.parent_id,
                "creator": model.lineage.creator,
                "created_at": model.lineage.created_at,
                "license": model.lineage.license,
                "changes": model.lineage.changes,
                "description": model.description,
                "version": model.version,
                "tags": model.tags,
                "attributions": [
                    {
                        "name": a.name,
                        "creator": a.creator,
                        "license": a.license,
                        "source_url": a.source_url,
                        "citation": a.citation
                    }
                    for a in model.lineage.attributions
                ]
            }
        
        with open(self._lineage_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_fork(
        self,
        source_path: str,
        fork_name: str,
        creator: str,
        description: str = "",
        license: str = "Apache-2.0",
        changes: Optional[List[str]] = None
    ) -> ForkedModel:
        """
        Create a fork of an existing model.
        
        Args:
            source_path: Path to source model
            fork_name: Name for the fork
            creator: Creator/organization name
            description: Description of fork
            license: License for the fork
            changes: List of changes from parent
            
        Returns:
            Forked model
        """
        source = Path(source_path)
        if not source.exists():
            raise ValueError(f"Source model not found: {source_path}")
        
        # Generate model ID
        model_id = str(uuid.uuid4())[:12]
        
        # Create fork directory
        fork_path = self.models_dir / f"{fork_name}_{model_id}"
        fork_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        if source.is_dir():
            for item in source.iterdir():
                if item.is_file():
                    shutil.copy2(item, fork_path / item.name)
        else:
            shutil.copy2(source, fork_path / source.name)
        
        # Find parent in registry
        parent_id = None
        parent_attributions = []
        
        for existing_model in self._registry.values():
            if str(existing_model.path) == str(source) or existing_model.name == source.stem:
                parent_id = existing_model.model_id
                parent_attributions = list(existing_model.lineage.attributions)
                break
        
        # Calculate base model hash
        base_hash = self._calculate_model_hash(source)
        
        # Create lineage
        lineage = ModelLineage(
            model_id=model_id,
            parent_id=parent_id,
            name=fork_name,
            creator=creator,
            created_at=time.time(),
            license=license,
            base_model_hash=base_hash,
            changes=changes or [],
            attributions=parent_attributions
        )
        
        # Add attribution for parent
        parent_name = source.stem if source.is_dir() else source.name
        lineage.attributions.append(ModelAttribution(
            name=parent_name,
            creator="original",  # Would be looked up from parent's lineage
            license=license
        ))
        
        # Create forked model
        forked = ForkedModel(
            model_id=model_id,
            name=fork_name,
            path=fork_path,
            lineage=lineage,
            description=description
        )
        
        # Save fork metadata
        self._save_fork_metadata(forked)
        
        # Register
        self._registry[model_id] = forked
        self._save_registry()
        
        logger.info(f"Created fork '{fork_name}' (ID: {model_id}) from {source_path}")
        
        return forked
    
    def _calculate_model_hash(self, path: Path) -> str:
        """Calculate hash of model files."""
        hasher = hashlib.sha256()
        
        if path.is_dir():
            for f in sorted(path.glob("**/*")):
                if f.is_file():
                    hasher.update(f.read_bytes())
        else:
            hasher.update(path.read_bytes())
        
        return hasher.hexdigest()[:16]
    
    def _save_fork_metadata(self, model: ForkedModel):
        """Save fork metadata to model directory."""
        metadata_path = model.path / "FORK_INFO.json"
        
        metadata = {
            "model_id": model.model_id,
            "name": model.name,
            "description": model.description,
            "version": model.version,
            "forked_at": datetime.now().isoformat(),
            "lineage": {
                "parent_id": model.lineage.parent_id,
                "creator": model.lineage.creator,
                "license": model.lineage.license,
                "changes": model.lineage.changes,
                "attributions": [
                    {
                        "name": a.name,
                        "creator": a.creator,
                        "license": a.license
                    }
                    for a in model.lineage.attributions
                ]
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_lineage(self, model_id: str) -> List[ModelLineage]:
        """
        Get full lineage chain for a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            List of lineage entries from oldest to newest
        """
        lineage = []
        current_id = model_id
        
        while current_id:
            model = self._registry.get(current_id)
            if not model:
                break
            
            lineage.append(model.lineage)
            current_id = model.lineage.parent_id
        
        return list(reversed(lineage))
    
    def get_descendants(self, model_id: str) -> List[ForkedModel]:
        """
        Get all models forked from a given model.
        
        Args:
            model_id: Parent model ID
            
        Returns:
            List of descendant models
        """
        descendants = []
        
        for model in self._registry.values():
            if model.lineage.parent_id == model_id:
                descendants.append(model)
                # Recursively get children's descendants
                descendants.extend(self.get_descendants(model.model_id))
        
        return descendants
    
    def get_attribution_text(self, model_id: str) -> str:
        """
        Generate attribution text for a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Attribution text
        """
        model = self._registry.get(model_id)
        if not model:
            return ""
        
        lines = [
            f"# {model.name}",
            f"Created by: {model.lineage.creator}",
            f"License: {model.lineage.license}",
            "",
            "## Attribution",
            ""
        ]
        
        for attr in model.lineage.attributions:
            lines.append(f"- **{attr.name}** by {attr.creator} ({attr.license})")
            if attr.source_url:
                lines.append(f"  Source: {attr.source_url}")
            if attr.citation:
                lines.append(f"  Citation: {attr.citation}")
        
        return "\n".join(lines)
    
    def register_existing(
        self,
        path: str,
        name: str,
        creator: str,
        license: str = "Apache-2.0",
        description: str = ""
    ) -> ForkedModel:
        """
        Register an existing model in the lineage system.
        
        Args:
            path: Model path
            name: Model name
            creator: Creator name
            license: Model license
            description: Description
            
        Returns:
            Registered model
        """
        model_path = Path(path)
        model_id = str(uuid.uuid4())[:12]
        
        lineage = ModelLineage(
            model_id=model_id,
            parent_id=None,
            name=name,
            creator=creator,
            created_at=time.time(),
            license=license,
            base_model_hash=self._calculate_model_hash(model_path)
        )
        
        model = ForkedModel(
            model_id=model_id,
            name=name,
            path=model_path,
            lineage=lineage,
            description=description
        )
        
        self._registry[model_id] = model
        self._save_registry()
        
        return model
    
    def list_models(self) -> List[ForkedModel]:
        """List all registered models."""
        return list(self._registry.values())
    
    def get_model(self, model_id: str) -> Optional[ForkedModel]:
        """Get a model by ID."""
        return self._registry.get(model_id)
    
    def render_lineage_tree(self, model_id: str) -> str:
        """
        Render lineage as ASCII tree.
        
        Args:
            model_id: Model to show tree for
            
        Returns:
            ASCII tree representation
        """
        lineage = self.get_lineage(model_id)
        if not lineage:
            return "Model not found"
        
        lines = []
        for i, entry in enumerate(lineage):
            prefix = "  " * i + ("└─ " if i > 0 else "")
            lines.append(f"{prefix}{entry.name} ({entry.model_id})")
            lines.append(f"{'  ' * (i + 1)}Creator: {entry.creator}")
            lines.append(f"{'  ' * (i + 1)}License: {entry.license}")
        
        return "\n".join(lines)


# Convenience functions
def fork_model(
    source: str,
    name: str,
    creator: str,
    **kwargs
) -> ForkedModel:
    """
    Quick fork a model.
    
    Args:
        source: Source model path
        name: Name for fork
        creator: Creator name
        **kwargs: Additional arguments
        
    Returns:
        Forked model
    """
    fork_manager = ModelFork()
    return fork_manager.create_fork(source, name, creator, **kwargs)


def get_model_lineage(model_id: str) -> List[ModelLineage]:
    """Get lineage for a model."""
    fork_manager = ModelFork()
    return fork_manager.get_lineage(model_id)
