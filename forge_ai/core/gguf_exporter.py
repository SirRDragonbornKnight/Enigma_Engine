"""
GGUF Exporter

DEPRECATED: This module has been consolidated into forge_ai.core.gguf
Import from there instead:
    from forge_ai.core.gguf import GGUFExporter, export_to_gguf

This file re-exports for backward compatibility.
"""

# Re-export everything from the consolidated module
from forge_ai.core.gguf import (
    GGMLType,
    GGUFValueType,
    GGUFMetadataType,
    GGUFTensor,
    GGUFMetadata,
    GGUFQuantizer,
    GGUFWriter,
    GGUFExporter,
    export_to_gguf,
    convert_tensor_name,
    QUANT_TYPES,
    GGML_BLOCK_SIZES,
    GGUF_MAGIC,
    GGUF_VERSION,
)

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
    'QUANT_TYPES',
    'GGML_BLOCK_SIZES',
    'GGUF_MAGIC',
    'GGUF_VERSION',
]
