"""
Advanced Quantization - Re-exports from dedicated quantization modules.

DEPRECATED: Use the dedicated modules directly:
    from enigma_engine.core.awq_quantization import AWQQuantizer
    from enigma_engine.core.gptq_quantization import GPTQQuantizer
    from enigma_engine.core.quantization import QuantConfig, QuantizedLinear

This file maintains backward compatibility.
"""

# Re-export from dedicated modules
from enigma_engine.core.awq_quantization import (
    AWQConfig,
    AWQLinear,
    AWQQuantizer,
)
from enigma_engine.core.gptq_quantization import QuantizedLinear  # Use GPTQ version
from enigma_engine.core.gptq_quantization import (
    GPTQ,
    GPTQConfig,
    GPTQQuantizer,
)
from enigma_engine.core.quantization import (
    GGMLQuantConfig,
    GGMLQuantType,
    QuantConfig,
)

# Backwards compat aliases
QuantizedWeight = AWQConfig  # Similar structure

__all__ = [
    # AWQ
    'AWQConfig',
    'AWQLinear', 
    'AWQQuantizer',
    # GPTQ
    'GPTQConfig',
    'GPTQ',
    'GPTQQuantizer',
    'QuantizedLinear',
    # General
    'QuantConfig',
    'QuantizedWeight',
    'GGMLQuantType',
    'GGMLQuantConfig',
]
