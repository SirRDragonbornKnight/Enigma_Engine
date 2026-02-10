"""
GPU Utilities for Enigma AI Engine.

Provides common GPU operations to avoid code duplication across the codebase.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Cached torch module
_TORCH = None
_TORCH_CHECKED = False


def _get_torch():
    """Get torch module if available, cache result."""
    global _TORCH, _TORCH_CHECKED
    if not _TORCH_CHECKED:
        try:
            import torch
            _TORCH = torch
        except ImportError:
            _TORCH = None
        _TORCH_CHECKED = True
    return _TORCH


def clear_cuda_cache() -> bool:
    """
    Clear CUDA memory cache if available.
    
    This is a common operation used throughout Enigma AI Engine to free GPU memory
    after model unloading or heavy computations.
    
    Returns:
        True if cache was cleared, False if CUDA not available or error occurred
    """
    torch = _get_torch()
    if torch is None:
        return False
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("CUDA cache cleared")
            return True
    except Exception as e:
        logger.debug(f"Could not clear CUDA cache: {e}")
    
    return False


def get_gpu_memory_info() -> Optional[dict[str, Any]]:
    """
    Get GPU memory information.
    
    Returns:
        Dict with 'total_mb', 'used_mb', 'free_mb', 'utilization' keys,
        or None if GPU not available
    """
    torch = _get_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    
    try:
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        free = total - reserved
        
        return {
            'total_mb': total // (1024 * 1024),
            'reserved_mb': reserved // (1024 * 1024),
            'allocated_mb': allocated // (1024 * 1024),
            'free_mb': free // (1024 * 1024),
            'utilization': allocated / total if total > 0 else 0,
        }
    except Exception as e:
        logger.debug(f"Could not get GPU memory info: {e}")
        return None


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    torch = _get_torch()
    if torch is None:
        return False
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if Apple MPS (Metal) is available."""
    torch = _get_torch()
    if torch is None:
        return False
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def get_best_device() -> str:
    """
    Get the best available device string.
    
    Returns:
        'cuda', 'mps', or 'cpu'
    """
    if is_cuda_available():
        return 'cuda'
    if is_mps_available():
        return 'mps'
    return 'cpu'
