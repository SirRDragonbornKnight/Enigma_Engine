"""
================================================================================
Provider Base - Unified provider interface for all generation tabs.
================================================================================

All generation providers (Image, Code, Video, Audio, 3D, Embeddings) should
inherit from this base class to ensure consistent behavior across the app.

Key benefits:
- Consistent load/unload lifecycle
- Device-aware initialization
- Automatic fallback handling
- Memory-efficient unloading
- Standard error handling

Usage:
    from forge_ai.gui.tabs.provider_base import GenerationProvider
    
    class MyLocalProvider(GenerationProvider):
        NAME = "my_local"
        DISPLAY_NAME = "Local (My Model)"
        IS_CLOUD = False
        
        def _do_load(self) -> bool:
            # Load your model here
            self._model = load_my_model()
            return self._model is not None
        
        def _do_generate(self, **kwargs) -> dict:
            # Generate here
            result = self._model.generate(**kwargs)
            return {"success": True, "path": result}
"""

import gc
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class GenerationProvider(ABC):
    """
    Base class for all generation providers.
    
    Provides:
    - Consistent lifecycle (load/unload/generate)
    - Device-aware configuration
    - Automatic fallback to builtin
    - Memory management
    - Error handling
    """
    
    # Override these in subclasses
    NAME: str = "base"
    DISPLAY_NAME: str = "Base Provider"
    IS_CLOUD: bool = False
    REQUIRES_API_KEY: bool = False
    API_KEY_ENV_VAR: str = ""
    BUILTIN_FALLBACK_CLASS: Optional[str] = None  # e.g., "BuiltinImageGen"
    
    # Hardware requirements (for device detection)
    MIN_RAM_MB: int = 512
    MIN_VRAM_MB: int = 0
    REQUIRES_GPU: bool = False
    
    def __init__(self, **config):
        """
        Initialize provider.
        
        Args:
            **config: Provider-specific configuration
        """
        self.config = config
        self._instance = None
        self._builtin_instance = None
        self._using_builtin = False
        self._is_loaded = False
        self._load_error: Optional[str] = None
        self._device = None
        self._dtype = None
        
        # Get device info if available
        self._init_device()
    
    def _init_device(self):
        """Initialize device settings from profiler."""
        try:
            from ...core.device_profiles import get_device_profiler
            profiler = get_device_profiler()
            self._device = profiler.get_torch_device()
            self._dtype = profiler.get_torch_dtype()
        except ImportError:
            self._device = "cpu"
            self._dtype = None
    
    @property
    def is_loaded(self) -> bool:
        """Check if provider is loaded and ready."""
        return self._is_loaded
    
    @property
    def using_builtin(self) -> bool:
        """Check if using builtin fallback."""
        return self._using_builtin
    
    @property
    def device(self) -> str:
        """Get the device string (cuda/cpu/mps)."""
        return self._device or "cpu"
    
    @property
    def load_error(self) -> Optional[str]:
        """Get the last load error if any."""
        return self._load_error
    
    @property
    def instance(self):
        """Get the underlying model/client instance."""
        if self._using_builtin:
            return self._builtin_instance
        return self._instance
    
    def load(self) -> bool:
        """
        Load the provider.
        
        Returns:
            True if loaded successfully (including via fallback)
        """
        if self._is_loaded:
            return True
        
        self._load_error = None
        
        # Check API key for cloud providers
        if self.REQUIRES_API_KEY:
            api_key = self._get_api_key()
            if not api_key:
                self._load_error = f"Missing API key. Set {self.API_KEY_ENV_VAR} environment variable."
                logger.warning(self._load_error)
                return self._try_builtin_fallback()
        
        # Try primary load
        try:
            if self._do_load():
                self._is_loaded = True
                self._using_builtin = False
                logger.info(f"{self.DISPLAY_NAME} loaded successfully")
                return True
        except ImportError as e:
            self._load_error = f"Missing dependency: {e}"
            logger.warning(f"{self.DISPLAY_NAME} import error: {e}")
        except Exception as e:
            self._load_error = f"Load failed: {e}"
            logger.warning(f"{self.DISPLAY_NAME} load failed: {e}")
        
        # Try builtin fallback
        return self._try_builtin_fallback()
    
    def _try_builtin_fallback(self) -> bool:
        """Attempt to load builtin fallback."""
        if not self.BUILTIN_FALLBACK_CLASS:
            return False
        
        try:
            from ...builtin import (
                BuiltinImageGen, BuiltinCodeGen, BuiltinVideoGen,
                BuiltinTTS, BuiltinEmbeddings, Builtin3DGen
            )
            
            builtin_map = {
                "BuiltinImageGen": BuiltinImageGen,
                "BuiltinCodeGen": BuiltinCodeGen,
                "BuiltinVideoGen": BuiltinVideoGen,
                "BuiltinTTS": BuiltinTTS,
                "BuiltinEmbeddings": BuiltinEmbeddings,
                "Builtin3DGen": Builtin3DGen,
            }
            
            cls = builtin_map.get(self.BUILTIN_FALLBACK_CLASS)
            if cls:
                self._builtin_instance = cls()
                if self._builtin_instance.load():
                    self._is_loaded = True
                    self._using_builtin = True
                    logger.info(f"{self.DISPLAY_NAME} using builtin fallback")
                    return True
        except Exception as e:
            logger.warning(f"Builtin fallback failed: {e}")
        
        return False
    
    def unload(self) -> bool:
        """
        Unload the provider and free resources.
        
        Returns:
            True if unloaded successfully
        """
        if not self._is_loaded:
            return True
        
        try:
            # Unload primary instance
            if self._instance is not None:
                self._do_unload()
                self._instance = None
            
            # Unload builtin
            if self._builtin_instance is not None:
                if hasattr(self._builtin_instance, 'unload'):
                    self._builtin_instance.unload()
                self._builtin_instance = None
            
            self._is_loaded = False
            self._using_builtin = False
            
            # Force memory cleanup
            gc.collect()
            
            # Clear GPU cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info(f"{self.DISPLAY_NAME} unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading {self.DISPLAY_NAME}: {e}")
            return False
    
    def generate(self, **kwargs) -> Dict[str, Any]:
        """
        Generate content.
        
        Args:
            **kwargs: Generation parameters
            
        Returns:
            Dict with 'success', 'path', 'error', 'duration' keys
        """
        if not self._is_loaded:
            return {"success": False, "error": "Provider not loaded"}
        
        start_time = time.time()
        
        try:
            # Validate inputs
            validation = self._validate_inputs(**kwargs)
            if not validation.get("valid", True):
                return {"success": False, "error": validation.get("error", "Invalid input")}
            
            # Use builtin if applicable
            if self._using_builtin and self._builtin_instance:
                result = self._builtin_instance.generate(**kwargs)
            else:
                result = self._do_generate(**kwargs)
            
            # Add duration if not present
            if "duration" not in result:
                result["duration"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"{self.DISPLAY_NAME} generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        import os
        
        # Check config first
        if "api_key" in self.config:
            return self.config["api_key"]
        
        # Then environment
        if self.API_KEY_ENV_VAR:
            return os.environ.get(self.API_KEY_ENV_VAR)
        
        return None
    
    def _validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """
        Validate generation inputs.
        
        Override in subclasses for custom validation.
        
        Returns:
            Dict with 'valid' bool and optional 'error' string
        """
        return {"valid": True}
    
    # =========================================================================
    # Abstract methods - override in subclasses
    # =========================================================================
    
    @abstractmethod
    def _do_load(self) -> bool:
        """
        Actual load implementation.
        
        Store loaded model/client in self._instance.
        
        Returns:
            True if loaded successfully
        """
        raise NotImplementedError
    
    def _do_unload(self):
        """
        Actual unload implementation.
        
        Override if you need custom cleanup beyond deleting self._instance.
        """
        pass
    
    @abstractmethod
    def _do_generate(self, **kwargs) -> Dict[str, Any]:
        """
        Actual generation implementation.
        
        Args:
            **kwargs: Generation parameters
            
        Returns:
            Dict with 'success' and result data
        """
        raise NotImplementedError
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def get_output_path(self, prefix: str, extension: str, output_dir: Path) -> Path:
        """Generate a timestamped output path."""
        import time
        timestamp = int(time.time())
        filename = f"{prefix}_{timestamp}.{extension}"
        return output_dir / filename
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get provider information for UI."""
        return {
            "name": cls.NAME,
            "display_name": cls.DISPLAY_NAME,
            "is_cloud": cls.IS_CLOUD,
            "requires_api_key": cls.REQUIRES_API_KEY,
            "api_key_env_var": cls.API_KEY_ENV_VAR,
            "min_ram_mb": cls.MIN_RAM_MB,
            "min_vram_mb": cls.MIN_VRAM_MB,
            "requires_gpu": cls.REQUIRES_GPU,
        }


class ProviderRegistry:
    """
    Registry of available providers by type.
    
    Usage:
        registry = ProviderRegistry()
        registry.register("image", PlaceholderImage)
        registry.register("image", StableDiffusionLocal)
        
        providers = registry.get_providers("image")
        provider = registry.get_provider("image", "local")
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._providers = {}  # {type: {name: class}}
            cls._instance._instances = {}  # {type: {name: instance}}
        return cls._instance
    
    def register(self, provider_type: str, provider_class: type):
        """Register a provider class."""
        if provider_type not in self._providers:
            self._providers[provider_type] = {}
        
        name = getattr(provider_class, 'NAME', provider_class.__name__)
        self._providers[provider_type][name] = provider_class
    
    def get_providers(self, provider_type: str) -> Dict[str, type]:
        """Get all registered providers for a type."""
        return self._providers.get(provider_type, {})
    
    def get_provider(
        self,
        provider_type: str,
        name: str,
        **config
    ) -> Optional[GenerationProvider]:
        """Get or create a provider instance."""
        # Check cache
        cache_key = f"{provider_type}_{name}"
        if cache_key in self._instances:
            inst = self._instances.get(provider_type, {}).get(name)
            if inst:
                return inst
        
        # Create new instance
        providers = self.get_providers(provider_type)
        if name not in providers:
            return None
        
        inst = providers[name](**config)
        
        # Cache
        if provider_type not in self._instances:
            self._instances[provider_type] = {}
        self._instances[provider_type][name] = inst
        
        return inst
    
    def unload_all(self, provider_type: Optional[str] = None):
        """Unload all cached providers."""
        if provider_type:
            if provider_type in self._instances:
                for inst in self._instances[provider_type].values():
                    if hasattr(inst, 'unload'):
                        inst.unload()
                self._instances[provider_type] = {}
        else:
            for type_instances in self._instances.values():
                for inst in type_instances.values():
                    if hasattr(inst, 'unload'):
                        inst.unload()
            self._instances = {}


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    return ProviderRegistry()


__all__ = [
    'GenerationProvider',
    'ProviderRegistry',
    'get_provider_registry',
]
