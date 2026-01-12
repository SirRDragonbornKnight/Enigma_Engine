"""
Vector utilities placeholder - DEPRECATED.
This module is deprecated. Use enigma.memory.vector_db.SimpleVectorDB instead.
"""
import warnings

# Re-export from vector_db for backward compatibility
from .vector_db import SimpleVectorDB

# Show deprecation warning when importing
warnings.warn(
    "enigma.memory.vector_utils.SimpleVectorDB is deprecated. "
    "Please use enigma.memory.vector_db.SimpleVectorDB instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['SimpleVectorDB']
