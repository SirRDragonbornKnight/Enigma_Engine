"""
Hardware Detection - Re-exports from hardware_detection.py

DEPRECATED: Use forge_ai.core.hardware_detection directly.
This file maintains backward compatibility.
"""

# Re-export everything from the main module  
from forge_ai.core.hardware_detection import (
    HardwareProfile,
    clear_cached_profile,
    detect_hardware,
    estimate_memory_usage,
    get_cached_profile,
    get_optimal_config,
    recommend_model_size,
)

__all__ = [
    'HardwareProfile',
    'detect_hardware',
    'recommend_model_size',
    'get_optimal_config',
    'estimate_memory_usage',
    'get_cached_profile',
    'clear_cached_profile',
]
