#!/usr/bin/env python3
"""
ForgeAI Module System Example
==============================

Complete example showing how to use ForgeAI's module system including:
- Loading and unloading modules
- Handling dependencies and conflicts
- Creating custom modules
- Module configuration

The module system is the core of ForgeAI - everything is a toggleable
module that can be enabled/disabled based on your needs and hardware.

Dependencies:
    None (pure Python)

Run: python examples/modules_example.py
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from enum import Enum, auto


# =============================================================================
# Module Types and States
# =============================================================================

class ModuleState(Enum):
    """Module lifecycle states."""
    UNLOADED = auto()
    LOADING = auto()
    LOADED = auto()
    ERROR = auto()
    DISABLED = auto()


class ModuleCategory(Enum):
    """Module categories."""
    CORE = "core"           # Essential modules
    GENERATION = "gen"      # AI generation (image, video, code)
    PERCEPTION = "percept"  # Input (vision, voice)
    OUTPUT = "output"       # Output (TTS, avatar)
    MEMORY = "memory"       # Storage and retrieval
    TOOLS = "tools"         # AI tools
    NETWORK = "network"     # Networking
    INTERFACE = "ui"        # User interface


@dataclass
class ModuleInfo:
    """Information about a module."""
    name: str
    category: ModuleCategory
    description: str
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)
    min_memory_mb: int = 0
    requires_gpu: bool = False


# =============================================================================
# Module Base Class
# =============================================================================

class Module:
    """
    Base class for ForgeAI modules.
    
    All modules inherit from this and implement load/unload.
    """
    
    def __init__(self, info: ModuleInfo):
        self.info = info
        self.state = ModuleState.UNLOADED
        self.error_message: Optional[str] = None
        self._load_time = 0.0
    
    @property
    def name(self) -> str:
        return self.info.name
    
    @property
    def is_loaded(self) -> bool:
        return self.state == ModuleState.LOADED
    
    def load(self) -> bool:
        """Load the module. Override in subclasses."""
        start_time = time.time()
        
        try:
            self.state = ModuleState.LOADING
            
            # Subclasses do their loading here
            success = self._do_load()
            
            if success:
                self.state = ModuleState.LOADED
                self._load_time = time.time() - start_time
                return True
            else:
                self.state = ModuleState.ERROR
                return False
                
        except Exception as e:
            self.state = ModuleState.ERROR
            self.error_message = str(e)
            return False
    
    def unload(self) -> bool:
        """Unload the module. Override in subclasses."""
        try:
            success = self._do_unload()
            
            if success:
                self.state = ModuleState.UNLOADED
                return True
            return False
            
        except Exception as e:
            self.error_message = str(e)
            return False
    
    def _do_load(self) -> bool:
        """Override this to implement loading logic."""
        return True
    
    def _do_unload(self) -> bool:
        """Override this to implement unloading logic."""
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            "name": self.name,
            "category": self.info.category.value,
            "state": self.state.name,
            "load_time": self._load_time,
            "error": self.error_message
        }


# =============================================================================
# Sample Modules
# =============================================================================

class ModelModule(Module):
    """Core model module - loads the AI model."""
    
    def __init__(self):
        super().__init__(ModuleInfo(
            name="model",
            category=ModuleCategory.CORE,
            description="Core AI model",
            dependencies=[],
            conflicts=[],
            provides=["inference"],
            min_memory_mb=500,
            requires_gpu=False
        ))
        self.model = None
    
    def _do_load(self) -> bool:
        print(f"  Loading AI model...")
        time.sleep(0.1)  # Simulate loading
        self.model = {"loaded": True, "params": "27M"}
        return True
    
    def _do_unload(self) -> bool:
        self.model = None
        return True


class TokenizerModule(Module):
    """Tokenizer module - text tokenization."""
    
    def __init__(self):
        super().__init__(ModuleInfo(
            name="tokenizer",
            category=ModuleCategory.CORE,
            description="Text tokenizer",
            dependencies=[],
            conflicts=[],
            provides=["tokenization"],
            min_memory_mb=50,
            requires_gpu=False
        ))
        self.tokenizer = None
    
    def _do_load(self) -> bool:
        print(f"  Loading tokenizer...")
        self.tokenizer = {"vocab_size": 50257}
        return True
    
    def _do_unload(self) -> bool:
        self.tokenizer = None
        return True


class ImageGenLocalModule(Module):
    """Local image generation with Stable Diffusion."""
    
    def __init__(self):
        super().__init__(ModuleInfo(
            name="image_gen_local",
            category=ModuleCategory.GENERATION,
            description="Local image generation (Stable Diffusion)",
            dependencies=["model"],
            conflicts=["image_gen_api"],  # Can't have both
            provides=["image_generation"],
            min_memory_mb=4000,
            requires_gpu=True
        ))
        self.pipe = None
    
    def _do_load(self) -> bool:
        print(f"  Loading Stable Diffusion...")
        time.sleep(0.1)
        self.pipe = {"model": "sd-1.5", "loaded": True}
        return True
    
    def _do_unload(self) -> bool:
        self.pipe = None
        return True
    
    def generate(self, prompt: str) -> str:
        """Generate image from prompt."""
        if not self.is_loaded:
            return "Module not loaded"
        return f"[Generated image for: {prompt}]"


class ImageGenAPIModule(Module):
    """Cloud image generation (DALL-E, Replicate)."""
    
    def __init__(self):
        super().__init__(ModuleInfo(
            name="image_gen_api",
            category=ModuleCategory.GENERATION,
            description="Cloud image generation (DALL-E)",
            dependencies=[],
            conflicts=["image_gen_local"],  # Can't have both
            provides=["image_generation"],
            min_memory_mb=10,
            requires_gpu=False
        ))
    
    def _do_load(self) -> bool:
        print(f"  Connecting to DALL-E API...")
        return True


class VoiceInputModule(Module):
    """Voice input (speech-to-text)."""
    
    def __init__(self):
        super().__init__(ModuleInfo(
            name="voice_input",
            category=ModuleCategory.PERCEPTION,
            description="Speech-to-text input",
            dependencies=[],
            conflicts=[],
            provides=["voice_input"],
            min_memory_mb=200,
            requires_gpu=False
        ))


class VoiceOutputModule(Module):
    """Voice output (text-to-speech)."""
    
    def __init__(self):
        super().__init__(ModuleInfo(
            name="voice_output",
            category=ModuleCategory.OUTPUT,
            description="Text-to-speech output",
            dependencies=[],
            conflicts=[],
            provides=["voice_output"],
            min_memory_mb=100,
            requires_gpu=False
        ))


# =============================================================================
# Module Manager
# =============================================================================

class ModuleManager:
    """
    Central module manager for ForgeAI.
    
    Handles:
    - Loading/unloading modules
    - Dependency resolution
    - Conflict prevention
    - Memory management
    """
    
    def __init__(self, available_memory_mb: int = 8000):
        """
        Initialize module manager.
        
        Args:
            available_memory_mb: Available system memory in MB
        """
        self.available_memory = available_memory_mb
        self.used_memory = 0
        
        # Registry of available modules
        self._registry: Dict[str, Module] = {}
        
        # Currently loaded modules
        self._loaded: Dict[str, Module] = {}
        
        # Provided capabilities
        self._capabilities: Dict[str, str] = {}  # capability -> module name
        
        # Register built-in modules
        self._register_builtin_modules()
    
    def _register_builtin_modules(self):
        """Register built-in modules."""
        modules = [
            ModelModule(),
            TokenizerModule(),
            ImageGenLocalModule(),
            ImageGenAPIModule(),
            VoiceInputModule(),
            VoiceOutputModule(),
        ]
        
        for module in modules:
            self._registry[module.name] = module
    
    def _log(self, message: str):
        """Log manager message."""
        print(f"[ModuleManager] {message}")
    
    def register(self, module: Module):
        """Register a custom module."""
        self._registry[module.name] = module
        self._log(f"Registered module: {module.name}")
    
    def get_available_modules(self) -> List[ModuleInfo]:
        """Get list of all available modules."""
        return [m.info for m in self._registry.values()]
    
    def get_loaded_modules(self) -> List[str]:
        """Get names of loaded modules."""
        return list(self._loaded.keys())
    
    def is_loaded(self, name: str) -> bool:
        """Check if a module is loaded."""
        return name in self._loaded
    
    def get_module(self, name: str) -> Optional[Module]:
        """Get a loaded module instance."""
        return self._loaded.get(name)
    
    def _check_dependencies(self, module: Module) -> List[str]:
        """Check if dependencies are satisfied."""
        missing = []
        for dep in module.info.dependencies:
            if dep not in self._loaded:
                missing.append(dep)
        return missing
    
    def _check_conflicts(self, module: Module) -> List[str]:
        """Check for conflicting modules."""
        conflicts = []
        for conflict in module.info.conflicts:
            if conflict in self._loaded:
                conflicts.append(conflict)
        
        # Also check if capability is already provided
        for capability in module.info.provides:
            if capability in self._capabilities:
                provider = self._capabilities[capability]
                if provider != module.name:
                    conflicts.append(f"{provider} (provides {capability})")
        
        return conflicts
    
    def _check_memory(self, module: Module) -> bool:
        """Check if enough memory is available."""
        required = module.info.min_memory_mb
        available = self.available_memory - self.used_memory
        return available >= required
    
    def load(self, name: str) -> bool:
        """
        Load a module.
        
        Args:
            name: Module name to load
            
        Returns:
            True if loaded successfully
        """
        self._log(f"Loading module: {name}")
        
        # Check module exists
        if name not in self._registry:
            self._log(f"  ERROR: Module '{name}' not found")
            return False
        
        module = self._registry[name]
        
        # Check if already loaded
        if name in self._loaded:
            self._log(f"  Already loaded")
            return True
        
        # Check dependencies
        missing_deps = self._check_dependencies(module)
        if missing_deps:
            self._log(f"  ERROR: Missing dependencies: {missing_deps}")
            return False
        
        # Check conflicts
        conflicts = self._check_conflicts(module)
        if conflicts:
            self._log(f"  ERROR: Conflicts with: {conflicts}")
            return False
        
        # Check memory
        if not self._check_memory(module):
            self._log(f"  ERROR: Insufficient memory (need {module.info.min_memory_mb}MB)")
            return False
        
        # Load the module
        if module.load():
            self._loaded[name] = module
            self.used_memory += module.info.min_memory_mb
            
            # Register provided capabilities
            for capability in module.info.provides:
                self._capabilities[capability] = name
            
            self._log(f"  Loaded successfully (took {module._load_time:.2f}s)")
            return True
        else:
            self._log(f"  ERROR: {module.error_message}")
            return False
    
    def unload(self, name: str) -> bool:
        """
        Unload a module.
        
        Args:
            name: Module name to unload
            
        Returns:
            True if unloaded successfully
        """
        self._log(f"Unloading module: {name}")
        
        if name not in self._loaded:
            self._log(f"  Not loaded")
            return True
        
        module = self._loaded[name]
        
        # Check if other modules depend on this one
        dependents = []
        for other_name, other_module in self._loaded.items():
            if name in other_module.info.dependencies:
                dependents.append(other_name)
        
        if dependents:
            self._log(f"  ERROR: Required by: {dependents}")
            return False
        
        # Unload
        if module.unload():
            del self._loaded[name]
            self.used_memory -= module.info.min_memory_mb
            
            # Unregister capabilities
            for capability in module.info.provides:
                if self._capabilities.get(capability) == name:
                    del self._capabilities[capability]
            
            self._log(f"  Unloaded successfully")
            return True
        else:
            self._log(f"  ERROR: {module.error_message}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        return {
            "available_modules": len(self._registry),
            "loaded_modules": len(self._loaded),
            "memory_used_mb": self.used_memory,
            "memory_available_mb": self.available_memory - self.used_memory,
            "capabilities": list(self._capabilities.keys()),
            "modules": [m.get_status() for m in self._loaded.values()]
        }


# =============================================================================
# Custom Module Example
# =============================================================================

class MyCustomModule(Module):
    """Example custom module."""
    
    def __init__(self):
        super().__init__(ModuleInfo(
            name="my_custom",
            category=ModuleCategory.TOOLS,
            description="My custom functionality",
            dependencies=["model"],  # Requires model to be loaded
            conflicts=[],
            provides=["custom_feature"],
            min_memory_mb=100,
            requires_gpu=False
        ))
        self.data = None
    
    def _do_load(self) -> bool:
        print(f"  Initializing custom module...")
        self.data = {"initialized": True}
        return True
    
    def _do_unload(self) -> bool:
        self.data = None
        return True
    
    def custom_function(self, input_data: str) -> str:
        """Custom functionality."""
        if not self.is_loaded:
            return "Module not loaded"
        return f"Processed: {input_data}"


# =============================================================================
# Example Usage
# =============================================================================

def example_basic_loading():
    """Basic module loading and unloading."""
    print("\n" + "="*60)
    print("Example 1: Basic Module Loading")
    print("="*60)
    
    manager = ModuleManager(available_memory_mb=8000)
    
    print("\nAvailable modules:")
    for info in manager.get_available_modules():
        print(f"  - {info.name}: {info.description}")
    
    print("\n--- Loading modules ---")
    
    # Load core modules
    manager.load("tokenizer")
    manager.load("model")
    
    print(f"\nLoaded: {manager.get_loaded_modules()}")
    
    # Unload
    print("\n--- Unloading ---")
    manager.unload("model")
    manager.unload("tokenizer")


def example_dependencies():
    """Module dependencies."""
    print("\n" + "="*60)
    print("Example 2: Dependencies")
    print("="*60)
    
    manager = ModuleManager()
    
    print("Trying to load image_gen_local (depends on model)...")
    manager.load("image_gen_local")  # Should fail
    
    print("\nLoading model first, then image_gen_local...")
    manager.load("model")
    manager.load("image_gen_local")  # Should succeed
    
    print(f"\nLoaded: {manager.get_loaded_modules()}")


def example_conflicts():
    """Module conflicts."""
    print("\n" + "="*60)
    print("Example 3: Conflicts")
    print("="*60)
    
    manager = ModuleManager()
    manager.load("model")
    
    print("Loading local image generation...")
    manager.load("image_gen_local")
    
    print("\nTrying to load API image generation (conflicts!)...")
    manager.load("image_gen_api")  # Should fail
    
    print("\nUnloading local, then loading API...")
    manager.unload("image_gen_local")
    manager.load("image_gen_api")  # Should succeed
    
    print(f"\nLoaded: {manager.get_loaded_modules()}")


def example_memory():
    """Memory management."""
    print("\n" + "="*60)
    print("Example 4: Memory Management")
    print("="*60)
    
    # Limited memory
    manager = ModuleManager(available_memory_mb=1000)
    
    print(f"Available memory: {manager.available_memory}MB")
    
    manager.load("tokenizer")  # 50MB
    manager.load("model")       # 500MB
    
    status = manager.get_status()
    print(f"Used: {status['memory_used_mb']}MB")
    print(f"Remaining: {status['memory_available_mb']}MB")
    
    print("\nTrying to load image_gen_local (needs 4000MB)...")
    manager.load("image_gen_local")  # Should fail


def example_custom_module():
    """Creating custom modules."""
    print("\n" + "="*60)
    print("Example 5: Custom Module")
    print("="*60)
    
    manager = ModuleManager()
    
    # Create and register custom module
    custom = MyCustomModule()
    manager.register(custom)
    
    # Load dependencies first
    manager.load("model")
    
    # Load custom module
    manager.load("my_custom")
    
    # Use the custom module
    module = manager.get_module("my_custom")
    if module:
        result = module.custom_function("test data")
        print(f"Custom function result: {result}")


def example_status():
    """Module status and capabilities."""
    print("\n" + "="*60)
    print("Example 6: Status and Capabilities")
    print("="*60)
    
    manager = ModuleManager()
    
    manager.load("tokenizer")
    manager.load("model")
    manager.load("voice_input")
    manager.load("voice_output")
    
    status = manager.get_status()
    
    print(f"\nModule Manager Status:")
    print(f"  Available modules: {status['available_modules']}")
    print(f"  Loaded modules: {status['loaded_modules']}")
    print(f"  Memory used: {status['memory_used_mb']}MB")
    print(f"  Capabilities: {status['capabilities']}")
    
    print(f"\nLoaded module details:")
    for mod in status['modules']:
        print(f"  - {mod['name']}: {mod['state']} ({mod['load_time']:.2f}s)")


def example_forge_integration():
    """Real ForgeAI integration."""
    print("\n" + "="*60)
    print("Example 7: ForgeAI Integration")
    print("="*60)
    
    print("For actual ForgeAI module system:")
    print("""
    from forge_ai.modules import ModuleManager
    from forge_ai.modules.registry import (
        ModelModule, TokenizerModule, ImageGenLocalModule
    )
    
    # Create manager
    manager = ModuleManager()
    
    # Load modules
    manager.load('tokenizer')
    manager.load('model')
    manager.load('image_gen_local')  # Or 'image_gen_api'
    
    # Use image generation
    image_mod = manager.get_module('image_gen_local')
    result = image_mod.generate("a sunset", width=512, height=512)
    
    # GUI has Module Manager tab for visual control
    python run.py --gui
    # Click "Modules" tab to toggle modules on/off
    """)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("ForgeAI Module System Examples")
    print("="*60)
    
    example_basic_loading()
    example_dependencies()
    example_conflicts()
    example_memory()
    example_custom_module()
    example_status()
    example_forge_integration()
    
    print("\n" + "="*60)
    print("Module System Summary:")
    print("="*60)
    print("""
Key Concepts:

1. Everything is a Module:
   - Core: model, tokenizer, inference
   - Generation: image_gen, video_gen, code_gen, audio_gen
   - Perception: voice_input, vision, camera
   - Output: voice_output, avatar
   - Memory: memory, embedding
   - Tools: web_tools, file_tools

2. Dependencies:
   - Modules can require other modules
   - Manager auto-checks dependencies
   - Load order matters

3. Conflicts:
   - Some modules conflict (can't load both)
   - e.g., image_gen_local vs image_gen_api
   - Manager prevents loading conflicts

4. Capabilities:
   - Modules provide capabilities
   - Only one module per capability
   - e.g., both image_gen modules provide "image_generation"

5. Memory Management:
   - Each module declares memory needs
   - Manager tracks total usage
   - Prevents overloading system

Creating Custom Module:
    class MyModule(Module):
        def __init__(self):
            super().__init__(ModuleInfo(
                name="my_module",
                category=ModuleCategory.TOOLS,
                dependencies=["model"],
                provides=["my_feature"]
            ))
        
        def _do_load(self):
            # Load your resources
            return True
        
        def _do_unload(self):
            # Cleanup
            return True

GUI Control:
    python run.py --gui
    # Use Modules tab to toggle modules visually
""")
