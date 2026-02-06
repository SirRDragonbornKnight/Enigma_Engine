"""
GGUF Export

DEPRECATED: This module has been consolidated into forge_ai.core.gguf
Import from there instead:
    from forge_ai.core.gguf import GGUFExporter, export_to_gguf

This file re-exports for backward compatibility.
"""

# Re-export everything from the consolidated module
from forge_ai.core.gguf import (
    GGML_BLOCK_SIZES,
    GGUF_MAGIC,
    GGUF_VERSION,
    QUANT_TYPES,
    GGMLType,
    GGUFExporter,
    GGUFMetadata,
    GGUFMetadataType,
    GGUFQuantizer,
    GGUFTensor,
    GGUFValueType,
    GGUFWriter,
    convert_tensor_name,
    export_to_gguf,
)

# Backward compatibility aliases
TensorInfo = GGUFTensor

__all__ = [
    'GGMLType',
    'GGUFValueType',
    'GGUFMetadataType',
    'GGUFTensor',
    'GGUFMetadata',
    'GGUFQuantizer',
    'GGUFWriter',
    'GGUFExporter',
    'export_to_gguf',
    'convert_tensor_name',
    'TensorInfo',
    'QUANT_TYPES',
    'GGML_BLOCK_SIZES',
    'GGUF_MAGIC',
    'GGUF_VERSION',
]
