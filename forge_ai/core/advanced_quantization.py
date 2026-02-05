"""
Advanced Quantization - Re-exports from dedicated quantization modules.

DEPRECATED: Use the dedicated modules directly:
    from forge_ai.core.awq_quantization import AWQQuantizer
    from forge_ai.core.gptq_quantization import GPTQQuantizer
    from forge_ai.core.quantization import QuantConfig, QuantizedLinear

This file maintains backward compatibility.
"""

# Re-export from dedicated modules
from forge_ai.core.awq_quantization import (
    AWQConfig,
    AWQLinear,
    AWQQuantizer,
)

from forge_ai.core.gptq_quantization import (
    GPTQConfig,
    GPTQ,
    GPTQQuantizer,
    QuantizedLinear,  # Use GPTQ version
)

from forge_ai.core.quantization import (
    QuantConfig,
    GGMLQuantType,
    GGMLQuantConfig,
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
