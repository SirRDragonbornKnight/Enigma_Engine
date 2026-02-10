"""
Ollama Model Import

Load and convert models from Ollama format to Enigma AI Engine.

FILE: enigma_engine/core/ollama_loader.py
TYPE: Model Loading
MAIN CLASSES: OllamaModelLoader, OllamaModelInfo
"""

import json
import logging
import os
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaQuantType(Enum):
    """Ollama quantization types."""
    F32 = "f32"
    F16 = "f16"
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    Q5_0 = "q5_0"
    Q5_1 = "q5_1"
    Q8_0 = "q8_0"
    Q2_K = "q2_K"
    Q3_K = "q3_K"
    Q4_K = "q4_K"
    Q5_K = "q5_K"
    Q6_K = "q6_K"


@dataclass
class OllamaModelInfo:
    """Information about an Ollama model."""
    name: str
    size: int  # bytes
    digest: str
    modified_at: str
    quantization: str
    parameter_size: str  # e.g., "7B", "13B"
    family: str  # e.g., "llama", "mistral"
    template: str = ""
    system_prompt: str = ""
    context_length: int = 4096
    embedding_length: int = 4096
    attention_heads: int = 32
    attention_kv_heads: int = 32
    layers: int = 32
    vocab_size: int = 32000


@dataclass
class OllamaBlob:
    """Ollama blob (layer) information."""
    digest: str
    size: int
    media_type: str
    data: bytes = None


class OllamaModelLoader:
    """
    Load models from Ollama format.
    
    Ollama stores models in ~/.ollama/models/ with:
    - manifests/: Model manifests (JSON)
    - blobs/: Model weights (GGUF format)
    """
    
    # Default Ollama paths
    DEFAULT_PATHS = {
        "linux": Path.home() / ".ollama" / "models",
        "darwin": Path.home() / ".ollama" / "models",
        "windows": Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama" / "models"
    }
    
    def __init__(self, ollama_path: str = None):
        self.ollama_path = Path(ollama_path) if ollama_path else self._find_ollama_path()
        self.manifests_path = self.ollama_path / "manifests"
        self.blobs_path = self.ollama_path / "blobs"
    
    def _find_ollama_path(self) -> Path:
        """Find Ollama models directory."""
        import platform
        system = platform.system().lower()
        
        if system in self.DEFAULT_PATHS:
            path = self.DEFAULT_PATHS[system]
            if path.exists():
                return path
        
        # Try environment variable
        if "OLLAMA_MODELS" in os.environ:
            return Path(os.environ["OLLAMA_MODELS"])
        
        raise FileNotFoundError("Ollama models directory not found")
    
    def list_models(self) -> list[OllamaModelInfo]:
        """List all available Ollama models."""
        models = []
        
        if not self.manifests_path.exists():
            return models
        
        # Iterate through registry/library
        registry_path = self.manifests_path / "registry.ollama.ai" / "library"
        
        if registry_path.exists():
            for model_dir in registry_path.iterdir():
                if model_dir.is_dir():
                    for tag_file in model_dir.iterdir():
                        if tag_file.is_file():
                            try:
                                info = self._parse_manifest(tag_file, model_dir.name)
                                if info:
                                    models.append(info)
                            except Exception as e:
                                logger.warning(f"Failed to parse {tag_file}: {e}")
        
        return models
    
    def _parse_manifest(
        self,
        manifest_path: Path,
        model_name: str
    ) -> Optional[OllamaModelInfo]:
        """Parse Ollama manifest file."""
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Get layers
        layers = manifest.get("layers", [])
        
        # Find model layer (usually gguf)
        model_layer = None
        config_layer = None
        template = ""
        system = ""
        
        for layer in layers:
            media_type = layer.get("mediaType", "")
            
            if "model" in media_type:
                model_layer = layer
            elif "params" in media_type or "config" in media_type:
                config_layer = layer
            elif "template" in media_type:
                template = self._read_blob(layer.get("digest", ""))
            elif "system" in media_type:
                system = self._read_blob(layer.get("digest", ""))
        
        if not model_layer:
            return None
        
        # Parse config
        config = {}
        if config_layer:
            config_data = self._read_blob(config_layer.get("digest", ""))
            if config_data:
                try:
                    config = json.loads(config_data)
                except json.JSONDecodeError:
                    pass
        
        # Determine quantization and size
        digest = model_layer.get("digest", "")
        size = model_layer.get("size", 0)
        
        # Parse model architecture from config
        return OllamaModelInfo(
            name=f"{model_name}:{manifest_path.name}",
            size=size,
            digest=digest,
            modified_at=manifest.get("modified", ""),
            quantization=config.get("quantization", "unknown"),
            parameter_size=self._estimate_param_size(size),
            family=config.get("architecture", config.get("model_type", "unknown")),
            template=template if isinstance(template, str) else "",
            system_prompt=system if isinstance(system, str) else "",
            context_length=config.get("context_length", 4096),
            embedding_length=config.get("embedding_length", 4096),
            attention_heads=config.get("num_attention_heads", 32),
            attention_kv_heads=config.get("num_key_value_heads", 32),
            layers=config.get("num_hidden_layers", 32),
            vocab_size=config.get("vocab_size", 32000)
        )
    
    def _read_blob(self, digest: str) -> Optional[bytes]:
        """Read blob data from digest."""
        if not digest:
            return None
        
        # Digest format: sha256:xxxx
        blob_path = self.blobs_path / digest.replace(":", "-")
        
        if blob_path.exists():
            with open(blob_path, "rb") as f:
                return f.read()
        
        return None
    
    def _estimate_param_size(self, size_bytes: int) -> str:
        """Estimate parameter size from file size."""
        # Rough estimates based on quantization
        gb = size_bytes / (1024 ** 3)
        
        if gb < 1:
            return "tiny"
        elif gb < 2:
            return "1B"
        elif gb < 4:
            return "3B"
        elif gb < 5:
            return "7B"
        elif gb < 8:
            return "13B"
        elif gb < 15:
            return "30B"
        elif gb < 40:
            return "65B"
        else:
            return "70B+"
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to model blob."""
        # Parse model name
        if ":" in model_name:
            name, tag = model_name.split(":", 1)
        else:
            name, tag = model_name, "latest"
        
        manifest_path = self.manifests_path / "registry.ollama.ai" / "library" / name / tag
        
        if not manifest_path.exists():
            return None
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        for layer in manifest.get("layers", []):
            if "model" in layer.get("mediaType", ""):
                digest = layer.get("digest", "")
                blob_path = self.blobs_path / digest.replace(":", "-")
                
                if blob_path.exists():
                    return blob_path
        
        return None
    
    def load_model(
        self,
        model_name: str,
        device: str = "cpu"
    ) -> Optional[dict[str, Any]]:
        """
        Load Ollama model into Enigma AI Engine format.
        
        Args:
            model_name: Ollama model name (e.g., "llama2:7b")
            device: Target device
            
        Returns:
            Model state dict or None
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for model loading")
        
        model_path = self.get_model_path(model_name)
        
        if not model_path:
            logger.error(f"Model not found: {model_name}")
            return None
        
        logger.info(f"Loading model from {model_path}")
        
        # Ollama uses GGUF format
        return self._load_gguf(model_path, device)
    
    def _load_gguf(
        self,
        path: Path,
        device: str
    ) -> Optional[dict[str, Any]]:
        """Load GGUF format model."""
        # Import GGUF loader
        try:
            from .gguf_loader import load_gguf_file
            return load_gguf_file(str(path), device)
        except ImportError:
            logger.warning("GGUF loader not available, attempting direct load")
        
        # Basic GGUF parsing
        with open(path, "rb") as f:
            # GGUF magic
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError("Invalid GGUF file")
            
            # Version
            version = struct.unpack("<I", f.read(4))[0]
            logger.info(f"GGUF version: {version}")
            
            # Tensor count
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_kv_count = struct.unpack("<Q", f.read(8))[0]
            
            logger.info(f"Tensors: {tensor_count}, Metadata: {metadata_kv_count}")
            
            # Parse metadata
            metadata = self._parse_gguf_metadata(f, metadata_kv_count)
            
            # Parse tensor info
            tensors = self._parse_gguf_tensors(f, tensor_count)
            
            return {
                "metadata": metadata,
                "tensors": tensors,
                "path": str(path)
            }
    
    def _parse_gguf_metadata(
        self,
        f,
        count: int
    ) -> dict[str, Any]:
        """Parse GGUF metadata."""
        metadata = {}
        
        for _ in range(count):
            # Key
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8")
            
            # Value type
            value_type = struct.unpack("<I", f.read(4))[0]
            
            # Value
            value = self._read_gguf_value(f, value_type)
            metadata[key] = value
        
        return metadata
    
    def _read_gguf_value(self, f, value_type: int) -> Any:
        """Read GGUF value based on type."""
        # GGUF types
        if value_type == 0:  # UINT8
            return struct.unpack("<B", f.read(1))[0]
        elif value_type == 1:  # INT8
            return struct.unpack("<b", f.read(1))[0]
        elif value_type == 2:  # UINT16
            return struct.unpack("<H", f.read(2))[0]
        elif value_type == 3:  # INT16
            return struct.unpack("<h", f.read(2))[0]
        elif value_type == 4:  # UINT32
            return struct.unpack("<I", f.read(4))[0]
        elif value_type == 5:  # INT32
            return struct.unpack("<i", f.read(4))[0]
        elif value_type == 6:  # FLOAT32
            return struct.unpack("<f", f.read(4))[0]
        elif value_type == 7:  # BOOL
            return bool(struct.unpack("<B", f.read(1))[0])
        elif value_type == 8:  # STRING
            str_len = struct.unpack("<Q", f.read(8))[0]
            return f.read(str_len).decode("utf-8")
        elif value_type == 9:  # ARRAY
            array_type = struct.unpack("<I", f.read(4))[0]
            array_len = struct.unpack("<Q", f.read(8))[0]
            return [self._read_gguf_value(f, array_type) for _ in range(array_len)]
        elif value_type == 10:  # UINT64
            return struct.unpack("<Q", f.read(8))[0]
        elif value_type == 11:  # INT64
            return struct.unpack("<q", f.read(8))[0]
        elif value_type == 12:  # FLOAT64
            return struct.unpack("<d", f.read(8))[0]
        
        return None
    
    def _parse_gguf_tensors(
        self,
        f,
        count: int
    ) -> list[dict[str, Any]]:
        """Parse GGUF tensor info."""
        tensors = []
        
        for _ in range(count):
            # Name
            name_len = struct.unpack("<Q", f.read(8))[0]
            name = f.read(name_len).decode("utf-8")
            
            # Dimensions
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            
            # Type
            tensor_type = struct.unpack("<I", f.read(4))[0]
            
            # Offset
            offset = struct.unpack("<Q", f.read(8))[0]
            
            tensors.append({
                "name": name,
                "dims": dims,
                "type": tensor_type,
                "offset": offset
            })
        
        return tensors
    
    def convert_to_forge(
        self,
        model_name: str,
        output_path: str,
        device: str = "cpu"
    ) -> bool:
        """
        Convert Ollama model to Enigma AI Engine format.
        
        Args:
            model_name: Ollama model name
            output_path: Output path for Enigma AI Engine model
            device: Device for conversion
            
        Returns:
            Success status
        """
        model_data = self.load_model(model_name, device)
        
        if not model_data:
            return False
        
        # Get model info
        models = self.list_models()
        info = next((m for m in models if m.name == model_name), None)
        
        # Create Enigma AI Engine config
        config = {
            "model_type": "forge",
            "architecture": info.family if info else "unknown",
            "hidden_size": info.embedding_length if info else 4096,
            "num_attention_heads": info.attention_heads if info else 32,
            "num_key_value_heads": info.attention_kv_heads if info else 32,
            "num_hidden_layers": info.layers if info else 32,
            "vocab_size": info.vocab_size if info else 32000,
            "max_position_embeddings": info.context_length if info else 4096,
            "source": f"ollama:{model_name}"
        }
        
        # Save in Enigma AI Engine format
        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(output / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Save model reference (actual weights are in GGUF)
        with open(output / "model_ref.json", "w") as f:
            json.dump({
                "format": "gguf",
                "source_path": str(self.get_model_path(model_name)),
                "metadata": model_data.get("metadata", {})
            }, f, indent=2)
        
        logger.info(f"Model converted and saved to {output_path}")
        return True


def list_ollama_models() -> list[OllamaModelInfo]:
    """List available Ollama models."""
    try:
        loader = OllamaModelLoader()
        return loader.list_models()
    except FileNotFoundError:
        logger.warning("Ollama not installed or models directory not found")
        return []


def load_ollama_model(model_name: str, device: str = "cpu"):
    """Load an Ollama model."""
    loader = OllamaModelLoader()
    return loader.load_model(model_name, device)
