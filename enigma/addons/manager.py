"""
Addon Manager
=============

Central manager for all Enigma addons.
Handles loading, routing requests, and coordination.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import importlib
import sys

from .base import (
    Addon, AddonResult, AddonType
)


class AddonManager:
    """
    Manages all addons in Enigma.
    
    Features:
    - Load/unload addons dynamically
    - Route requests to appropriate addons
    - Track usage and costs
    - Support multiple addons per type
    - Fallback chains
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.addons: Dict[str, Addon] = {}
        self.type_registry: Dict[AddonType, List[str]] = {t: [] for t in AddonType}
        self.default_addons: Dict[AddonType, str] = {}
        self.fallback_chains: Dict[AddonType, List[str]] = {}
        
        self.config_path = config_path or Path("data/addons_config.json")
        self._callbacks: Dict[str, List] = {}
        
    # === Registration ===
    
    def register(self, addon: Addon, set_default: bool = True) -> bool:
        """
        Register an addon with the manager.
        
        Args:
            addon: The addon instance to register
            set_default: Whether to set as default for its type
        """
        if addon.name in self.addons:
            print(f"Warning: Addon '{addon.name}' already registered, replacing")
        
        self.addons[addon.name] = addon
        
        # Add to type registry
        if addon.name not in self.type_registry[addon.addon_type]:
            self.type_registry[addon.addon_type].append(addon.name)
        
        # Set as default if requested or if first of type
        if set_default or addon.addon_type not in self.default_addons:
            self.default_addons[addon.addon_type] = addon.name
        
        self._emit('addon_registered', addon)
        return True
    
    def unregister(self, name: str) -> bool:
        """Remove an addon from the manager."""
        if name not in self.addons:
            return False
        
        addon = self.addons[name]
        
        # Unload if loaded
        if addon.is_loaded:
            addon.unload()
        
        # Remove from registries
        del self.addons[name]
        if name in self.type_registry[addon.addon_type]:
            self.type_registry[addon.addon_type].remove(name)
        if self.default_addons.get(addon.addon_type) == name:
            # Set new default if available
            remaining = self.type_registry[addon.addon_type]
            self.default_addons[addon.addon_type] = remaining[0] if remaining else None
        
        self._emit('addon_unregistered', name)
        return True
    
    # === Loading ===
    
    def load_addon(self, name: str) -> bool:
        """Load a specific addon."""
        if name not in self.addons:
            return False
        
        addon = self.addons[name]
        if addon.is_loaded:
            return True
        
        try:
            success = addon.load()
            if success:
                self._emit('addon_loaded', addon)
            return success
        except Exception as e:
            print(f"Failed to load addon '{name}': {e}")
            return False
    
    def unload_addon(self, name: str) -> bool:
        """Unload a specific addon."""
        if name not in self.addons:
            return False
        
        addon = self.addons[name]
        if not addon.is_loaded:
            return True
        
        try:
            success = addon.unload()
            if success:
                self._emit('addon_unloaded', addon)
            return success
        except Exception as e:
            print(f"Failed to unload addon '{name}': {e}")
            return False
    
    def load_all(self) -> Dict[str, bool]:
        """Load all registered addons."""
        results = {}
        for name in self.addons:
            results[name] = self.load_addon(name)
        return results
    
    def unload_all(self) -> Dict[str, bool]:
        """Unload all addons."""
        results = {}
        for name in self.addons:
            results[name] = self.unload_addon(name)
        return results
    
    # === Generation ===
    
    def generate(self, addon_type: AddonType, prompt: str, 
                 addon_name: Optional[str] = None, **kwargs) -> AddonResult:
        """
        Generate using an addon.
        
        Args:
            addon_type: Type of generation (IMAGE, CODE, VIDEO, etc.)
            prompt: The input prompt
            addon_name: Specific addon to use (uses default if None)
            **kwargs: Addon-specific parameters
        """
        # Get addon
        if addon_name:
            if addon_name not in self.addons:
                return AddonResult(success=False, error=f"Addon '{addon_name}' not found")
            addon = self.addons[addon_name]
        else:
            addon = self.get_default(addon_type)
            if not addon:
                return AddonResult(success=False, error=f"No addon available for {addon_type.name}")
        
        # Check availability
        if not addon.is_loaded:
            if not self.load_addon(addon.name):
                return AddonResult(success=False, error=f"Failed to load addon '{addon.name}'")
        
        if not addon.check_availability():
            # Try fallback
            fallback = self._get_fallback(addon_type, addon.name)
            if fallback:
                addon = fallback
            else:
                return AddonResult(success=False, error=f"Addon '{addon.name}' not available")
        
        # Generate
        try:
            result = addon.generate(prompt, **kwargs)
            result.addon_name = addon.name
            self._emit('generation_complete', result)
            return result
        except Exception as e:
            return AddonResult(success=False, error=str(e), addon_name=addon.name)
    
    # Convenience methods for common types
    
    def generate_image(self, prompt: str, width: int = 512, height: int = 512,
                       num_images: int = 1, addon_name: str = None, **kwargs) -> AddonResult:
        """Generate images."""
        return self.generate(
            AddonType.IMAGE, prompt,
            addon_name=addon_name,
            width=width, height=height, num_images=num_images,
            **kwargs
        )
    
    def generate_code(self, prompt: str, language: str = "python",
                      addon_name: str = None, **kwargs) -> AddonResult:
        """Generate code."""
        return self.generate(
            AddonType.CODE, prompt,
            addon_name=addon_name,
            language=language,
            **kwargs
        )
    
    def generate_video(self, prompt: str, duration: float = 4.0,
                       fps: int = 24, addon_name: str = None, **kwargs) -> AddonResult:
        """Generate video."""
        return self.generate(
            AddonType.VIDEO, prompt,
            addon_name=addon_name,
            duration=duration, fps=fps,
            **kwargs
        )
    
    def generate_audio(self, prompt: str, duration: float = 10.0,
                       addon_name: str = None, **kwargs) -> AddonResult:
        """Generate audio."""
        return self.generate(
            AddonType.AUDIO, prompt,
            addon_name=addon_name,
            duration=duration,
            **kwargs
        )
    
    def generate_embedding(self, text: str, addon_name: str = None, **kwargs) -> AddonResult:
        """Generate embedding vector."""
        return self.generate(
            AddonType.EMBEDDING, text,
            addon_name=addon_name,
            **kwargs
        )
    
    # === Addon Access ===
    
    def get(self, name: str) -> Optional[Addon]:
        """Get an addon by name."""
        return self.addons.get(name)
    
    def get_default(self, addon_type: AddonType) -> Optional[Addon]:
        """Get the default addon for a type."""
        name = self.default_addons.get(addon_type)
        return self.addons.get(name) if name else None
    
    def set_default(self, addon_type: AddonType, name: str) -> bool:
        """Set the default addon for a type."""
        if name not in self.addons:
            return False
        if self.addons[name].addon_type != addon_type:
            return False
        self.default_addons[addon_type] = name
        return True
    
    def list_addons(self, addon_type: Optional[AddonType] = None) -> List[str]:
        """List registered addons, optionally filtered by type."""
        if addon_type:
            return self.type_registry.get(addon_type, [])
        return list(self.addons.keys())
    
    def get_by_type(self, addon_type: AddonType) -> List[Addon]:
        """Get all addons of a specific type."""
        names = self.type_registry.get(addon_type, [])
        return [self.addons[n] for n in names if n in self.addons]
    
    # === Fallbacks ===
    
    def set_fallback_chain(self, addon_type: AddonType, chain: List[str]):
        """Set fallback chain for an addon type."""
        self.fallback_chains[addon_type] = chain
    
    def _get_fallback(self, addon_type: AddonType, exclude: str) -> Optional[Addon]:
        """Get next available fallback addon."""
        chain = self.fallback_chains.get(addon_type, self.type_registry[addon_type])
        for name in chain:
            if name != exclude and name in self.addons:
                addon = self.addons[name]
                if addon.is_loaded and addon.check_availability():
                    return addon
        return None
    
    # === Persistence ===
    
    def save_config(self):
        """Save addon configuration."""
        config = {
            'defaults': {t.name: n for t, n in self.default_addons.items() if n},
            'fallbacks': {t.name: c for t, c in self.fallback_chains.items()},
            'addons': {}
        }
        
        for name, addon in self.addons.items():
            config['addons'][name] = addon.config.to_dict()
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self):
        """Load addon configuration."""
        if not self.config_path.exists():
            return
        
        try:
            with open(self.config_path) as f:
                config = json.load(f)
            
            # Restore defaults
            for type_name, addon_name in config.get('defaults', {}).items():
                addon_type = AddonType[type_name]
                if addon_name in self.addons:
                    self.default_addons[addon_type] = addon_name
            
            # Restore fallbacks
            for type_name, chain in config.get('fallbacks', {}).items():
                addon_type = AddonType[type_name]
                self.fallback_chains[addon_type] = chain
                
        except Exception as e:
            print(f"Failed to load addon config: {e}")
    
    # === Dynamic Loading ===
    
    def load_addon_from_file(self, path: Path) -> Optional[Addon]:
        """Load an addon from a Python file."""
        try:
            spec = importlib.util.spec_from_file_location("addon_module", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["addon_module"] = module
            spec.loader.exec_module(module)
            
            # Look for addon class
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) and issubclass(obj, Addon) and obj is not Addon:
                    addon = obj()
                    self.register(addon)
                    return addon
            
            return None
        except Exception as e:
            print(f"Failed to load addon from {path}: {e}")
            return None
    
    def load_addons_from_directory(self, directory: Path) -> List[Addon]:
        """Load all addons from a directory."""
        addons = []
        for path in directory.glob("*.py"):
            if path.stem.startswith("_"):
                continue
            addon = self.load_addon_from_file(path)
            if addon:
                addons.append(addon)
        return addons
    
    # === Events ===
    
    def on(self, event: str, callback):
        """Register event callback."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def _emit(self, event: str, data: Any = None):
        """Emit event to callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                print(f"Addon manager callback error: {e}")
    
    # === Info ===
    
    def get_status(self) -> dict:
        """Get manager status."""
        return {
            'total_addons': len(self.addons),
            'loaded_addons': sum(1 for a in self.addons.values() if a.is_loaded),
            'types_available': {
                t.name: len(addons) for t, addons in self.type_registry.items() if addons
            },
            'defaults': {t.name: n for t, n in self.default_addons.items() if n},
        }
    
    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get all capabilities grouped by type."""
        caps = {}
        for addon_type, names in self.type_registry.items():
            if names:
                caps[addon_type.name] = [
                    {
                        'name': n,
                        'loaded': self.addons[n].is_loaded,
                        'available': self.addons[n].check_availability() if self.addons[n].is_loaded else False,
                        'provider': self.addons[n].provider.name,
                    }
                    for n in names
                ]
        return caps


# Global manager instance
_global_manager: Optional[AddonManager] = None


def get_addon_manager() -> AddonManager:
    """Get the global addon manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = AddonManager()
    return _global_manager


def init_addon_manager(config_path: Optional[Path] = None) -> AddonManager:
    """Initialize the global addon manager."""
    global _global_manager
    _global_manager = AddonManager(config_path)
    return _global_manager
