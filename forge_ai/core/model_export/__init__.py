"""
Model Hub - Bidirectional Model Transfer
========================================

Import AND export models to/from various platforms:
- HuggingFace Hub (share and download models)
- Replicate (cloud model serving)
- Ollama (local serving)
- Weights & Biases (experiment tracking)
- ONNX (edge deployment, export only)

Usage:
    from forge_ai.core.model_export import ModelHub
    
    hub = ModelHub()
    
    # === IMPORT from platforms ===
    
    # Search for models
    models = hub.search("llama", provider="huggingface")
    models = hub.search("mistral", provider="ollama")
    
    # Import a model
    result = hub.import_model("microsoft/DialoGPT-small", provider="huggingface")
    result = hub.import_model("llama2:7b", provider="ollama")
    
    # === EXPORT to platforms ===
    
    hub.export("my_model", provider="huggingface", repo_id="user/model")
    hub.export("my_model", provider="ollama", model_name="my-model")
    
    # === Quick functions ===
    
    from forge_ai.core.model_export import import_model, export_model, search_models
    
    result = import_model("meta-llama/Llama-2-7b", "huggingface", local_name="llama2")
    result = export_model("my_model", "huggingface", repo_id="user/model")
    models = search_models("llama", "huggingface")
"""

from .base import (
    # Export base
    ExportProvider, ExportResult, ExportStatus, ProviderConfig,
    # Import base
    ImportProvider, ImportResult, ImportStatus,
)
from .exporter import (
    ModelHub,
    ModelExporter,  # Backwards compatibility alias
    get_hub,
    get_exporter,  # Backwards compatibility alias
    export_model,
    import_model,
    search_models,
    list_export_providers,
    list_import_providers,
)

# Export providers
from .huggingface import HuggingFaceProvider
from .replicate import ReplicateProvider
from .ollama import OllamaProvider
from .wandb import WandBProvider
from .onnx import ONNXProvider

# Import providers
from .huggingface import HuggingFaceImporter
from .replicate import ReplicateImporter
from .ollama import OllamaImporter
from .wandb import WandBImporter

__all__ = [
    # Main interface
    "ModelHub",
    "get_hub",
    
    # Quick functions
    "export_model",
    "import_model",
    "search_models",
    "list_export_providers",
    "list_import_providers",
    
    # Base classes - Export
    "ExportProvider",
    "ExportResult",
    "ExportStatus",
    "ProviderConfig",
    
    # Base classes - Import
    "ImportProvider",
    "ImportResult",
    "ImportStatus",
    
    # Export Providers
    "HuggingFaceProvider",
    "ReplicateProvider", 
    "OllamaProvider",
    "WandBProvider",
    "ONNXProvider",
    
    # Import Providers
    "HuggingFaceImporter",
    "ReplicateImporter",
    "OllamaImporter",
    "WandBImporter",
    
    # Backwards compatibility
    "ModelExporter",
    "get_exporter",
]
