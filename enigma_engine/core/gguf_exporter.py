"""
GGUF Exporter

DEPRECATED: This module has been consolidated into enigma_engine.core.gguf
Import from there instead:
    from enigma_engine.core.gguf import GGUFExporter, export_to_gguf

This file re-exports for backward compatibility.
"""

# Re-export everything from the consolidated module
from enigma_engine.core.gguf import (
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
