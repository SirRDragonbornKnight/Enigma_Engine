"""
Enigma Configuration Management

This package provides:
- Default configuration
- User configuration loading
- Environment variable support
"""

from .defaults import CONFIG, get_config, update_config

__all__ = ["CONFIG", "get_config", "update_config"]
