"""
Enigma AI Engine Model Hub - Share and discover fine-tuned models

Features:
- Upload/download models
- Model versioning
- Search and discovery
- Usage statistics
- Model cards (documentation)
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Optional imports
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class ModelLicense(Enum):
    """Common model licenses"""
    MIT = "mit"
    APACHE_2 = "apache-2.0"
    GPL_3 = "gpl-3.0"
    CC_BY_4 = "cc-by-4.0"
    CC_BY_NC_4 = "cc-by-nc-4.0"
    CC_BY_SA_4 = "cc-by-sa-4.0"
    LLAMA_2 = "llama2"
    OPENRAIL = "openrail"
    CUSTOM = "custom"


class ModelCategory(Enum):
    """Model categories"""
    CHAT = "chat"
    CODE = "code"
    CREATIVE = "creative"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QA = "question-answering"
    ROLEPLAY = "roleplay"
    TOOL_USE = "tool-use"
    MULTIMODAL = "multimodal"
    OTHER = "other"


@dataclass
class ModelCard:
    """Model documentation card (similar to HuggingFace model cards)"""
    name: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    license: str = ModelLicense.MIT.value
    category: str = ModelCategory.CHAT.value
    tags: list[str] = field(default_factory=list)
    
    # Technical details
    base_model: str = ""
    model_size: str = ""  # e.g., "small", "medium", "7B"
    parameters: int = 0
    quantization: str = ""  # e.g., "int8", "int4", "fp16"
    context_length: int = 2048
    
    # Training details
    training_data: str = ""
    training_steps: int = 0
    training_compute: str = ""
    fine_tuning_method: str = ""  # e.g., "full", "lora", "qlora"
    
    # Usage
    intended_use: str = ""
    limitations: str = ""
    example_prompts: list[str] = field(default_factory=list)
    
    # Metadata
    created_at: str = ""
    updated_at: str = ""
    downloads: int = 0
    likes: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ModelCard':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def to_markdown(self) -> str:
        """Generate markdown documentation"""
        md = f"""# {self.name}

{self.description}

## Model Details

| Property | Value |
|----------|-------|
| Author | {self.author} |
| Version | {self.version} |
| License | {self.license} |
| Category | {self.category} |
| Base Model | {self.base_model or 'N/A'} |
| Parameters | {self.parameters:,} |
| Context Length | {self.context_length:,} |
| Quantization | {self.quantization or 'None'} |

## Tags

{', '.join(f'`{tag}`' for tag in self.tags)}

## Training

- **Fine-tuning Method:** {self.fine_tuning_method or 'N/A'}
- **Training Data:** {self.training_data or 'N/A'}
- **Training Steps:** {self.training_steps:,}

## Intended Use

{self.intended_use or 'General purpose language model.'}

## Limitations

{self.limitations or 'See license for usage restrictions.'}

## Example Prompts

"""
        for prompt in self.example_prompts[:5]:
            md += f"- {prompt}\n"
        
        md += f"""
## Statistics

- Downloads: {self.downloads:,}
- Likes: {self.likes:,}
- Created: {self.created_at}
- Updated: {self.updated_at}
"""
        return md


@dataclass
class ModelVersion:
    """Version information for a model"""
    version: str
    sha256: str
    size_bytes: int
    created_at: str
    download_url: str = ""
    changelog: str = ""


@dataclass
class HubModel:
    """Model in the hub"""
    id: str
    card: ModelCard
    versions: list[ModelVersion] = field(default_factory=list)
    files: list[str] = field(default_factory=list)


class LocalModelIndex:
    """
    Local index of downloaded models
    """
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.index_file = self.models_dir / "hub_index.json"
        self._index: dict[str, dict] = {}
        self._load_index()
    
    def _load_index(self) -> None:
        """Load index from file"""
        if self.index_file.exists():
            try:
                with open(self.index_file) as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load model index: {e}")
                self._index = {}
    
    def _save_index(self) -> None:
        """Save index to file"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_file, 'w') as f:
            json.dump(self._index, f, indent=2)
    
    def add_model(self, model_id: str, version: str, 
                 path: Path, card: ModelCard) -> None:
        """Add model to index"""
        self._index[model_id] = {
            "version": version,
            "path": str(path),
            "card": card.to_dict(),
            "downloaded_at": datetime.now().isoformat()
        }
        self._save_index()
    
    def remove_model(self, model_id: str) -> bool:
        """Remove model from index"""
        if model_id in self._index:
            # Remove files
            model_path = Path(self._index[model_id]["path"])
            if model_path.exists():
                shutil.rmtree(model_path)
            
            del self._index[model_id]
            self._save_index()
            return True
        return False
    
    def get_model(self, model_id: str) -> Optional[dict]:
        """Get model info from index"""
        return self._index.get(model_id)
    
    def list_models(self) -> list[dict]:
        """List all indexed models"""
        return [
            {"id": k, **v} for k, v in self._index.items()
        ]
    
    def is_downloaded(self, model_id: str, version: str = None) -> bool:
        """Check if model is downloaded"""
        if model_id not in self._index:
            return False
        if version and self._index[model_id]["version"] != version:
            return False
        return True


class ModelHubClient:
    """
    Client for interacting with model hub servers
    
    Supports:
    - Enigma AI Engine Hub (default)
    - HuggingFace Hub
    - Custom servers
    """
    
    DEFAULT_HUB_URL = "https://hub.Enigma AI Engine.dev/api"  # Placeholder
    
    def __init__(self, 
                 hub_url: str = None,
                 api_key: str = None,
                 models_dir: Union[str, Path] = None):
        self.hub_url = hub_url or self.DEFAULT_HUB_URL
        self.api_key = api_key
        self.models_dir = Path(models_dir) if models_dir else Path.home() / ".enigma_engine" / "hub_models"
        
        self.local_index = LocalModelIndex(self.models_dir)
        self._session = None
    
    async def _get_session(self):
        """Get aiohttp session"""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required for async operations")
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        """Close the client session"""
        if self._session:
            await self._session.close()
            self._session = None
    
    def _get_headers(self) -> dict[str, str]:
        """Get request headers"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def search_models(self, 
                           query: str = "",
                           category: ModelCategory = None,
                           tags: list[str] = None,
                           sort_by: str = "downloads",
                           limit: int = 20) -> list[dict]:
        """Search for models in the hub"""
        params = {
            "q": query,
            "limit": limit,
            "sort": sort_by
        }
        if category:
            params["category"] = category.value
        if tags:
            params["tags"] = ",".join(tags)
        
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.hub_url}/models",
                params=params,
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Search failed: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    async def get_model_info(self, model_id: str) -> Optional[HubModel]:
        """Get detailed model information"""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.hub_url}/models/{model_id}",
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return HubModel(
                        id=model_id,
                        card=ModelCard.from_dict(data.get("card", {})),
                        versions=[
                            ModelVersion(**v) for v in data.get("versions", [])
                        ],
                        files=data.get("files", [])
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None
    
    async def download_model(self,
                            model_id: str,
                            version: str = "latest",
                            progress_callback: Callable[[int, int], None] = None) -> Optional[Path]:
        """
        Download a model from the hub
        
        Args:
            model_id: Model identifier
            version: Version to download (or "latest")
            progress_callback: Callback for download progress (received, total)
            
        Returns:
            Path to downloaded model directory
        """
        # Get model info
        model_info = await self.get_model_info(model_id)
        if not model_info:
            logger.error(f"Model not found: {model_id}")
            return None
        
        # Find version
        if version == "latest" and model_info.versions:
            version_info = model_info.versions[0]
        else:
            version_info = next(
                (v for v in model_info.versions if v.version == version), None
            )
        
        if not version_info:
            logger.error(f"Version not found: {version}")
            return None
        
        # Create model directory
        model_dir = self.models_dir / model_id / version_info.version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            session = await self._get_session()
            
            # Download model archive
            download_url = version_info.download_url or \
                f"{self.hub_url}/models/{model_id}/download/{version_info.version}"
            
            async with session.get(
                download_url,
                headers=self._get_headers()
            ) as response:
                if response.status != 200:
                    logger.error(f"Download failed: {response.status}")
                    return None
                
                total_size = int(response.headers.get('content-length', 0))
                received = 0
                
                archive_path = model_dir / "model.zip"
                with open(archive_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
                        received += len(chunk)
                        if progress_callback:
                            progress_callback(received, total_size)
            
            # Verify checksum
            if version_info.sha256:
                actual_hash = self._compute_sha256(archive_path)
                if actual_hash != version_info.sha256:
                    logger.error("Checksum mismatch!")
                    archive_path.unlink()
                    return None
            
            # Extract archive
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(model_dir)
            archive_path.unlink()
            
            # Save model card
            card_path = model_dir / "model_card.json"
            with open(card_path, 'w') as f:
                json.dump(model_info.card.to_dict(), f, indent=2)
            
            # Update local index
            self.local_index.add_model(
                model_id, version_info.version, model_dir, model_info.card
            )
            
            logger.info(f"Downloaded model: {model_id} v{version_info.version}")
            return model_dir
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            if model_dir.exists():
                shutil.rmtree(model_dir)
            return None
    
    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    async def upload_model(self,
                          model_path: Union[str, Path],
                          card: ModelCard,
                          version: str = "1.0.0",
                          progress_callback: Callable[[int, int], None] = None) -> Optional[str]:
        """
        Upload a model to the hub
        
        Args:
            model_path: Path to model directory
            card: Model card with metadata
            version: Version string
            progress_callback: Callback for upload progress
            
        Returns:
            Model ID if successful
        """
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"Model path not found: {model_path}")
            return None
        
        if not self.api_key:
            logger.error("API key required for upload")
            return None
        
        try:
            # Create archive
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                archive_path = Path(tmp.name)
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in model_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(model_path)
                        zf.write(file_path, arcname)
            
            # Compute hash
            sha256 = self._compute_sha256(archive_path)
            file_size = archive_path.stat().st_size
            
            # Create model entry
            session = await self._get_session()
            
            # First, create model metadata
            create_data = {
                "card": card.to_dict(),
                "version": version,
                "sha256": sha256,
                "size_bytes": file_size
            }
            
            async with session.post(
                f"{self.hub_url}/models",
                json=create_data,
                headers=self._get_headers()
            ) as response:
                if response.status not in (200, 201):
                    error = await response.text()
                    logger.error(f"Failed to create model: {error}")
                    return None
                
                result = await response.json()
                model_id = result.get("id")
                upload_url = result.get("upload_url")
            
            # Upload file
            with open(archive_path, 'rb') as f:
                async with session.put(
                    upload_url,
                    data=f,
                    headers={"Content-Type": "application/zip"}
                ) as response:
                    if response.status not in (200, 201):
                        logger.error(f"Upload failed: {response.status}")
                        return None
            
            # Cleanup
            archive_path.unlink()
            
            logger.info(f"Uploaded model: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return None
    
    async def like_model(self, model_id: str) -> bool:
        """Like a model"""
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.hub_url}/models/{model_id}/like",
                headers=self._get_headers()
            ) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def get_trending(self, limit: int = 10) -> list[dict]:
        """Get trending models"""
        return await self.search_models(sort_by="trending", limit=limit)
    
    async def get_recent(self, limit: int = 10) -> list[dict]:
        """Get recently uploaded models"""
        return await self.search_models(sort_by="created_at", limit=limit)
    
    def list_downloaded(self) -> list[dict]:
        """List locally downloaded models"""
        return self.local_index.list_models()
    
    def delete_downloaded(self, model_id: str) -> bool:
        """Delete a downloaded model"""
        return self.local_index.remove_model(model_id)
    
    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get path to downloaded model"""
        info = self.local_index.get_model(model_id)
        if info:
            return Path(info["path"])
        return None


class HuggingFaceHubAdapter:
    """
    Adapter for HuggingFace Hub
    
    Allows downloading models from HuggingFace and converting them
    """
    
    HF_API_URL = "https://huggingface.co/api"
    
    def __init__(self, token: str = None):
        self.token = token or os.environ.get("HF_TOKEN")
    
    def _get_headers(self) -> dict[str, str]:
        """Get request headers"""
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    def search_models(self,
                     query: str = "",
                     filter_tags: list[str] = None,
                     limit: int = 20) -> list[dict]:
        """Search HuggingFace models"""
        if not HAS_REQUESTS:
            logger.error("requests library required")
            return []
        
        params = {
            "search": query,
            "limit": limit,
            "sort": "downloads",
            "direction": -1
        }
        if filter_tags:
            params["filter"] = ",".join(filter_tags)
        
        try:
            response = requests.get(
                f"{self.HF_API_URL}/models",
                params=params,
                headers=self._get_headers(),
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.error(f"HF search error: {e}")
            return []
    
    def get_model_info(self, model_id: str) -> Optional[dict]:
        """Get model info from HuggingFace"""
        if not HAS_REQUESTS:
            return None
        
        try:
            response = requests.get(
                f"{self.HF_API_URL}/models/{model_id}",
                headers=self._get_headers(),
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def download_model(self,
                      model_id: str,
                      local_dir: Union[str, Path],
                      revision: str = "main") -> bool:
        """
        Download model from HuggingFace
        
        Uses huggingface_hub library if available
        """
        try:
            from huggingface_hub import snapshot_download
            
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                revision=revision,
                token=self.token
            )
            return True
            
        except ImportError:
            logger.error("huggingface_hub library required")
            return False
        except Exception as e:
            logger.error(f"HF download error: {e}")
            return False


# Singleton instance
_hub_client: Optional[ModelHubClient] = None


def get_hub_client(hub_url: str = None, api_key: str = None) -> ModelHubClient:
    """Get the model hub client singleton"""
    global _hub_client
    if _hub_client is None:
        _hub_client = ModelHubClient(hub_url=hub_url, api_key=api_key)
    return _hub_client


def get_hf_adapter(token: str = None) -> HuggingFaceHubAdapter:
    """Get HuggingFace hub adapter"""
    return HuggingFaceHubAdapter(token=token)


# Convenience functions
async def download_from_hub(model_id: str, version: str = "latest") -> Optional[Path]:
    """Download a model from Enigma AI Engine Hub"""
    client = get_hub_client()
    return await client.download_model(model_id, version)


async def search_hub(query: str, category: str = None) -> list[dict]:
    """Search Enigma AI Engine Hub for models"""
    client = get_hub_client()
    cat = ModelCategory(category) if category else None
    return await client.search_models(query, category=cat)


def download_from_hf(model_id: str, local_dir: str) -> bool:
    """Download a model from HuggingFace"""
    adapter = get_hf_adapter()
    return adapter.download_model(model_id, local_dir)
