"""
Enigma Module Manager
=====================

Central system for managing all Enigma modules.
Handles loading, unloading, dependencies, and configuration.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional imports - cache results
_TORCH = None
_TORCH_CHECKED = False
_TORCH_WARNING_SHOWN = False

def _get_torch():
    """Get torch module if available, cache result."""
    global _TORCH, _TORCH_CHECKED, _TORCH_WARNING_SHOWN
    if not _TORCH_CHECKED:
        try:
            import torch
            _TORCH = torch
        except ImportError:
            _TORCH = None
            if not _TORCH_WARNING_SHOWN:
                logger.warning("PyTorch not available - GPU detection disabled")
                _TORCH_WARNING_SHOWN = True
        _TORCH_CHECKED = True
    return _TORCH


class ModuleState(Enum):
    """Module lifecycle states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


class ModuleCategory(Enum):
    """Module categories for organization."""
    CORE = "core"
    MEMORY = "memory"
    INTERFACE = "interface"
    PERCEPTION = "perception"
    OUTPUT = "output"
    GENERATION = "generation"  # AI generation: images, code, video, audio
    TOOLS = "tools"
    NETWORK = "network"
    EXTENSION = "extension"


@dataclass
class ModuleInfo:
    """Module metadata and configuration."""
    id: str
    name: str
    description: str
    category: ModuleCategory
    version: str = "1.0.0"
    
    # Dependencies
    requires: List[str] = field(default_factory=list)  # Required modules
    optional: List[str] = field(default_factory=list)  # Optional enhancements
    conflicts: List[str] = field(default_factory=list)  # Cannot run together
    
    # Hardware requirements
    min_ram_mb: int = 0
    min_vram_mb: int = 0
    requires_gpu: bool = False
    supports_distributed: bool = False
    
    # Capabilities provided
    provides: List[str] = field(default_factory=list)
    
    # Configuration schema
    config_schema: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime info
    state: ModuleState = ModuleState.UNLOADED
    load_time: Optional[datetime] = None
    error_message: Optional[str] = None


class Module:
    """
    Base class for all Enigma modules.
    
    Subclass this to create new modules.
    """
    
    # Override these in subclasses
    INFO = ModuleInfo(
        id="base",
        name="Base Module",
        description="Base module class",
        category=ModuleCategory.EXTENSION,
    )
    
    def __init__(self, manager: 'ModuleManager', config: Dict[str, Any] = None):
        self.manager = manager
        self.config = config or {}
        self.state = ModuleState.UNLOADED
        self._instance = None
    
    @classmethod
    def get_info(cls) -> ModuleInfo:
        """Get module information."""
        return cls.INFO
    
    def load(self) -> bool:
        """
        Load the module. Override in subclass.
        
        Returns True if successful, False otherwise.
        """
        return True
    
    def unload(self) -> bool:
        """
        Unload the module. Override in subclass.
        
        Returns True if successful, False otherwise.
        """
        return True
    
    def activate(self) -> bool:
        """
        Activate the module (start processing). Override in subclass.
        """
        return True
    
    def deactivate(self) -> bool:
        """
        Deactivate the module (stop processing). Override in subclass.
        """
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current module status."""
        return {
            'id': self.INFO.id,
            'state': self.state.value,
            'config': self.config,
        }
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Update module configuration.
        
        Args:
            config: New configuration values
            
        Returns:
            True if configuration was accepted
        """
        self.config.update(config)
        return True
    
    def get_interface(self) -> Any:
        """
        Get the module's public interface/instance.
        
        Returns the main object other modules should interact with.
        """
        return self._instance


class ModuleManager:
    """
    Central manager for all Enigma modules.
    
    Handles:
    - Module discovery and registration
    - Dependency resolution
    - Loading/unloading modules
    - Configuration management
    - Hardware compatibility checking
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.modules: Dict[str, Module] = {}
        self.module_classes: Dict[str, type] = {}
        self.config_path = config_path or Path("enigma_modules.json")
        self.hardware_profile: Dict[str, Any] = {}
        
        # Event callbacks
        self._on_load: List[Callable] = []
        self._on_unload: List[Callable] = []
        self._on_state_change: List[Callable] = []
        
        # Detect hardware
        self._detect_hardware()
    
    def _detect_hardware(self):
        """Detect available hardware capabilities."""
        # Default values
        self.hardware_profile = {
            'cpu_cores': 1,
            'ram_mb': 4096,
            'gpu_available': False,
            'gpu_name': None,
            'vram_mb': 0,
            'mps_available': False,
        }
        
        # Try to detect GPU with torch (cached)
        torch = _get_torch()
        if torch:
            try:
                self.hardware_profile['gpu_available'] = torch.cuda.is_available()
                self.hardware_profile['mps_available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                
                if torch.cuda.is_available():
                    self.hardware_profile['gpu_name'] = torch.cuda.get_device_name(0)
                    self.hardware_profile['vram_mb'] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            except Exception as e:
                logger.warning(f"Error detecting GPU: {e}")
        # Only warn once (checked in _get_torch)
        
        # Try to detect CPU/RAM with psutil
        try:
            import psutil
            self.hardware_profile['cpu_cores'] = psutil.cpu_count()
            self.hardware_profile['ram_mb'] = psutil.virtual_memory().total // (1024 * 1024)
        except ImportError:
            pass  # Silently use defaults
    
    def register(self, module_class: type) -> bool:
        """
        Register a module class.
        
        Args:
            module_class: Module subclass to register
            
        Returns:
            True if registered successfully
        """
        if not issubclass(module_class, Module):
            logger.error(f"Cannot register {module_class}: not a Module subclass")
            return False
        
        info = module_class.get_info()
        self.module_classes[info.id] = module_class
        logger.info(f"Registered module: {info.id} ({info.name})")
        return True
    
    def unregister(self, module_id: str) -> bool:
        """Unregister a module class."""
        if module_id in self.module_classes:
            # Unload if loaded
            if module_id in self.modules:
                self.unload(module_id)
            del self.module_classes[module_id]
            return True
        return False
    
    def can_load(self, module_id: str) -> tuple[bool, str]:
        """
        Check if a module can be loaded.
        
        Returns:
            (can_load, reason)
        """
        if module_id not in self.module_classes:
            return False, f"Module '{module_id}' not registered"
        
        info = self.module_classes[module_id].get_info()
        
        # Check hardware requirements
        if info.requires_gpu and not self.hardware_profile['gpu_available']:
            return False, "Module requires GPU but none available"
        
        if info.min_vram_mb > self.hardware_profile['vram_mb']:
            return False, f"Module requires {info.min_vram_mb}MB VRAM, only {self.hardware_profile['vram_mb']}MB available"
        
        if info.min_ram_mb > self.hardware_profile['ram_mb']:
            return False, f"Module requires {info.min_ram_mb}MB RAM, only {self.hardware_profile['ram_mb']}MB available"
        
        # Check explicit conflicts
        for conflict_id in info.conflicts:
            if conflict_id in self.modules and self.modules[conflict_id].state == ModuleState.LOADED:
                return False, f"Module conflicts with loaded module '{conflict_id}'"
        
        # Check capability conflicts (two modules providing same thing)
        # e.g., image_gen_local and image_gen_api both provide 'image_generation'
        for provided in info.provides:
            for loaded_id, loaded_module in self.modules.items():
                if loaded_module.state == ModuleState.LOADED:
                    loaded_info = loaded_module.get_info()
                    if provided in loaded_info.provides and loaded_id != module_id:
                        return False, f"Capability '{provided}' already provided by '{loaded_id}'. Unload it first."
        
        # Check dependencies
        for dep_id in info.requires:
            if dep_id not in self.modules or self.modules[dep_id].state != ModuleState.LOADED:
                return False, f"Required module '{dep_id}' not loaded"
        
        return True, "OK"
    
    def load(self, module_id: str, config: Dict[str, Any] = None) -> bool:
        """
        Load a module.
        
        Args:
            module_id: Module ID to load
            config: Optional configuration
            
        Returns:
            True if loaded successfully
        """
        can_load, reason = self.can_load(module_id)
        if not can_load:
            logger.error(f"Cannot load module '{module_id}': {reason}")
            return False
        
        module_class = self.module_classes[module_id]
        
        try:
            # Create instance
            module = module_class(self, config)
            module.state = ModuleState.LOADING
            
            # Load
            if module.load():
                module.state = ModuleState.LOADED
                module.get_info().load_time = datetime.now()
                self.modules[module_id] = module
                
                # Notify listeners
                for callback in self._on_load:
                    callback(module_id)
                
                logger.info(f"Loaded module: {module_id}")
                return True
            else:
                module.state = ModuleState.ERROR
                module.get_info().error_message = "load() returned False"
                return False
                
        except Exception as e:
            logger.error(f"Error loading module '{module_id}': {e}")
            return False
    
    def unload(self, module_id: str) -> bool:
        """Unload a module."""
        if module_id not in self.modules:
            return False
        
        module = self.modules[module_id]
        
        # Check if other modules depend on this one
        for other_id, other_module in self.modules.items():
            if other_id != module_id:
                info = other_module.get_info()
                if module_id in info.requires and other_module.state == ModuleState.LOADED:
                    logger.error(f"Cannot unload '{module_id}': required by '{other_id}'")
                    return False
        
        try:
            if module.unload():
                module.state = ModuleState.UNLOADED
                del self.modules[module_id]
                
                # Notify listeners
                for callback in self._on_unload:
                    callback(module_id)
                
                logger.info(f"Unloaded module: {module_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error unloading module '{module_id}': {e}")
            return False
    
    def activate(self, module_id: str) -> bool:
        """Activate a loaded module."""
        if module_id not in self.modules:
            return False
        
        module = self.modules[module_id]
        if module.state != ModuleState.LOADED:
            return False
        
        if module.activate():
            module.state = ModuleState.ACTIVE
            return True
        return False
    
    def deactivate(self, module_id: str) -> bool:
        """Deactivate an active module."""
        if module_id not in self.modules:
            return False
        
        module = self.modules[module_id]
        if module.state != ModuleState.ACTIVE:
            return False
        
        if module.deactivate():
            module.state = ModuleState.LOADED
            return True
        return False
    
    def get_module(self, module_id: str) -> Optional[Module]:
        """Get a loaded module instance."""
        return self.modules.get(module_id)
    
    def get_interface(self, module_id: str) -> Any:
        """Get a module's public interface."""
        module = self.modules.get(module_id)
        return module.get_interface() if module else None
    
    def list_modules(self, category: Optional[ModuleCategory] = None) -> List[ModuleInfo]:
        """List all registered modules."""
        modules = []
        for module_class in self.module_classes.values():
            info = module_class.get_info()
            if category is None or info.category == category:
                # Update state from loaded instance if exists
                if info.id in self.modules:
                    info.state = self.modules[info.id].state
                modules.append(info)
        return modules
    
    def list_loaded(self) -> List[str]:
        """List IDs of loaded modules."""
        return list(self.modules.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall status of all modules."""
        return {
            'hardware': self.hardware_profile,
            'registered': len(self.module_classes),
            'loaded': len(self.modules),
            'modules': {
                mid: module.get_status() 
                for mid, module in self.modules.items()
            }
        }
    
    def save_config(self, path: Optional[Path] = None):
        """Save current module configuration."""
        path = path or self.config_path
        
        config = {
            'loaded_modules': {},
            'disabled_modules': [],
        }
        
        for module_id, module in self.modules.items():
            config['loaded_modules'][module_id] = {
                'config': module.config,
                'active': module.state == ModuleState.ACTIVE,
            }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, path: Optional[Path] = None) -> bool:
        """Load and apply module configuration."""
        path = path or self.config_path
        
        if not path.exists():
            return False
        
        with open(path, 'r') as f:
            config = json.load(f)
        
        # Load modules in dependency order
        for module_id, module_config in config.get('loaded_modules', {}).items():
            if module_id in self.module_classes:
                self.load(module_id, module_config.get('config'))
                if module_config.get('active'):
                    self.activate(module_id)
        
        return True
    
    def on_load(self, callback: Callable):
        """Register callback for module load events."""
        self._on_load.append(callback)
    
    def on_unload(self, callback: Callable):
        """Register callback for module unload events."""
        self._on_unload.append(callback)
    
    def on_state_change(self, callback: Callable):
        """Register callback for module state changes."""
        self._on_state_change.append(callback)


# Global instance
_manager: Optional[ModuleManager] = None


def get_manager() -> ModuleManager:
    """Get the global module manager instance."""
    global _manager
    if _manager is None:
        _manager = ModuleManager()
    return _manager
