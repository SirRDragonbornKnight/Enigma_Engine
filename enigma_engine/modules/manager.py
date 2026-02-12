"""
================================================================================
⚙️ MODULE MANAGER - THE CONTROL CENTER
================================================================================

This is the BRAIN that controls what's running! Enigma AI Engine's secret power:
EVERYTHING is a module that can be loaded/unloaded dynamically.

📍 FILE: enigma_engine/modules/manager.py
🏷️ TYPE: Module Lifecycle Management
🎯 MAIN CLASSES: ModuleManager, Module, ModuleInfo, ModuleState

┌─────────────────────────────────────────────────────────────────────────────┐
│  MODULE STATES:                                                             │
│                                                                             │
│  UNLOADED → LOADING → LOADED → ACTIVE                                      │
│      ↑                            │                                         │
│      └────────────────────────────┘                                         │
│                (unload)                                                     │
│                                                                             │
│  ERROR / DISABLED (special states for problems)                            │
└─────────────────────────────────────────────────────────────────────────────┘

📦 MODULE CATEGORIES:
    • CORE       - model, tokenizer, inference
    • MEMORY     - memory, embeddings, vector_db
    • INTERFACE  - gui, voice input/output
    • PERCEPTION - vision, camera
    • OUTPUT     - avatar, tts
    • GENERATION - image, code, video, audio, 3d
    • TOOLS      - web, file, browser tools
    • NETWORK    - api_server, multi_device

⚠️ CONFLICT RULES:
    ✗ Cannot load image_gen_local AND image_gen_api together
    ✗ Cannot load code_gen_local AND code_gen_api together  
    ✓ Must load dependencies first (inference needs model + tokenizer)

🔗 CONNECTED FILES:
    → USES:      enigma_engine/modules/registry.py (all module definitions)
    → USES:      enigma_engine/modules/sandbox.py (safe execution)
    ← USED BY:   enigma_engine/gui/tabs/modules_tab.py (GUI toggle)
    ← USED BY:   All module consumers

📖 USAGE:
    from enigma_engine.modules import ModuleManager
    
    manager = ModuleManager()
    manager.load('model')           # Load core model
    manager.load('tokenizer')       # Load tokenizer
    manager.load('image_gen_local') # Load Stable Diffusion
    
    mod = manager.get_module('image_gen_local')
    result = mod.generate("a sunset", width=512)
    
    manager.unload('image_gen_local')  # Free memory

📖 SEE ALSO:
    • enigma_engine/modules/registry.py     - All available modules defined here
    • enigma_engine/gui/tabs/modules_tab.py - GUI for toggling modules
    • data/module_config.json          - Saved module settings
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .registry import Module

logger = logging.getLogger(__name__)

# Optional imports - cache results
_TORCH = None
_TORCH_CHECKED = False
_TORCH_WARNING_SHOWN = False
_DEVICE_PROFILER = None


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


def _get_device_profiler():
    """Get device profiler for hardware-aware decisions."""
    global _DEVICE_PROFILER
    if _DEVICE_PROFILER is None:
        try:
            from ..core.device_profiles import get_device_profiler
            _DEVICE_PROFILER = get_device_profiler()
        except ImportError:
            _DEVICE_PROFILER = None
    return _DEVICE_PROFILER


# =============================================================================
# 🔄 MODULE LIFECYCLE STATES
# =============================================================================
# Every module goes through these states during its lifecycle.
# The state machine ensures proper initialization and cleanup.

class ModuleState(Enum):
    """
    Module lifecycle states.
    
    📐 STATE TRANSITIONS:
    
    UNLOADED ──load()──▶ LOADING ──success──▶ LOADED ──activate()──▶ ACTIVE
        ▲                    │                  │                      │
        │                    ▼                  │                      │
        │                  ERROR                │                      │
        │                                       │                      │
        └─────────unload()──────────────────────┴──────────────────────┘
    
    DISABLED: Manually disabled by user, won't auto-load
    """
    UNLOADED = "unloaded"   # Not loaded, no resources used
    LOADING = "loading"     # Currently initializing
    LOADED = "loaded"       # Ready to use, but not actively processing
    ACTIVE = "active"       # Fully running and processing
    ERROR = "error"         # Failed to load or crashed
    DISABLED = "disabled"   # Manually disabled by user


class ModuleCategory(Enum):
    """
    Module categories for organization.
    
    📦 CATEGORIES:
    - CORE: Essential modules (model, tokenizer, inference)
    - MEMORY: Data storage (conversation history, vector DB)
    - INTERFACE: User interaction (GUI, voice)
    - PERCEPTION: Input processing (vision, camera)
    - OUTPUT: Output generation (avatar, TTS)
    - GENERATION: AI content creation (images, code, video, audio, 3D)
    - TOOLS: Utility functions (web search, file ops)
    - NETWORK: Communication (API server, multi-device)
    - EXTENSION: Third-party plugins
    """
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
    """
    Module metadata and configuration.
    
    📖 WHAT THIS HOLDS:
    Everything you need to know about a module before loading it:
    - What it is (id, name, description)
    - What it needs (dependencies, hardware)
    - What conflicts with it
    - Current state
    """
    # ─────────────────────────────────────────────────────────────────────────
    # IDENTITY: Who is this module?
    # ─────────────────────────────────────────────────────────────────────────
    id: str                       # Unique identifier (e.g., "image_gen_local")
    name: str                     # Human-readable name
    description: str              # What does it do?
    category: ModuleCategory      # Which category?
    version: str = "1.0.0"

    # ─────────────────────────────────────────────────────────────────────────
    # DEPENDENCIES: What does it need to work?
    # ─────────────────────────────────────────────────────────────────────────
    requires: List[str] = field(default_factory=list)   # MUST have these loaded first
    optional: List[str] = field(default_factory=list)   # Nice to have, not required
    conflicts: List[str] = field(default_factory=list)  # CANNOT run with these!

    # ─────────────────────────────────────────────────────────────────────────
    # HARDWARE REQUIREMENTS: What resources does it need?
    # ─────────────────────────────────────────────────────────────────────────
    min_ram_mb: int = 0           # Minimum RAM in MB
    min_vram_mb: int = 0          # Minimum GPU VRAM in MB
    requires_gpu: bool = False    # MUST have GPU?
    supports_distributed: bool = False  # Can run across multiple devices?

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVACY: Does it phone home?
    # ─────────────────────────────────────────────────────────────────────────
    is_cloud_service: bool = False  # True = sends data to external servers

    # ─────────────────────────────────────────────────────────────────────────
    # CAPABILITIES: What features does it provide?
    # ─────────────────────────────────────────────────────────────────────────
    provides: List[str] = field(default_factory=list)  # e.g., ["image_generation"]

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIGURATION: What settings can be adjusted?
    # ─────────────────────────────────────────────────────────────────────────
    config_schema: Dict[str, Any] = field(default_factory=dict)

    # ─────────────────────────────────────────────────────────────────────────
    # RUNTIME STATE: Current status
    # ─────────────────────────────────────────────────────────────────────────
    state: ModuleState = ModuleState.UNLOADED
    load_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class ModuleHealth:
    """
    Health status for a module.
    
    📖 USED FOR:
    Monitoring if modules are working correctly over time.
    """
    module_id: str
    is_healthy: bool
    last_check: datetime
    response_time_ms: float
    error_count: int
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# 🧩 MODULE BASE CLASS - The Blueprint for All Modules
# =============================================================================

class Module:
    """
    Base class for all Forge modules.
    
    📖 WHAT THIS IS:
    The template that ALL modules follow. When you create a new module,
    you inherit from this class and override the methods.
    
    📐 LIFECYCLE METHODS:
    1. load()     - Initialize resources (load model, connect to API)
    2. activate() - Start processing (begin listening, etc.)
    3. deactivate() - Stop processing (pause, save state)
    4. unload()   - Release resources (free memory, close connections)
    
    📐 EXAMPLE (creating a new module):
    
        class MyModule(Module):
            INFO = ModuleInfo(
                id="my_module",
                name="My Cool Module",
                description="Does cool stuff",
                category=ModuleCategory.EXTENSION,
            )
            
            def load(self) -> bool:
                # Initialize your module here
                self._instance = SomeCoolThing()
                return True
            
            def unload(self) -> bool:
                # Cleanup here
                self._instance = None
                return True
    
    🔗 CONNECTS TO:
      ← Inherited by all module classes in registry.py
      → Managed by ModuleManager
    """

    # ─────────────────────────────────────────────────────────────────────────
    # CLASS ATTRIBUTES: Override these in subclasses!
    # ─────────────────────────────────────────────────────────────────────────
    INFO = ModuleInfo(
        id="base",
        name="Base Module",
        description="Base module class - don't instantiate directly!",
        category=ModuleCategory.EXTENSION,
    )

    def __init__(self, manager: ModuleManager, config: Dict[str, Any] = None):
        """
        Initialize module instance.
        
        Args:
            manager: The ModuleManager that owns this module
            config: Configuration dictionary for this module
        """
        self.manager = manager        # Reference to parent manager
        self.config = config or {}    # Module-specific settings
        self.state = ModuleState.UNLOADED
        self._instance = None         # The actual working object (model, API client, etc.)

    @classmethod
    def get_info(cls) -> ModuleInfo:
        """Get module information (static metadata)."""
        return cls.INFO

    def load(self) -> bool:
        """
        Load the module - initialize resources.
        
        📖 OVERRIDE THIS IN SUBCLASS!
        Put your initialization code here:
        - Load model weights
        - Connect to APIs
        - Initialize hardware
        
        Returns:
            True if successful, False if failed
        """
        return True

    def unload(self) -> bool:
        """
        Unload the module - release resources.
        
        📖 OVERRIDE THIS IN SUBCLASS!
        Put your cleanup code here:
        - Free model from memory
        - Close connections
        - Save state
        
        Returns:
            True if successful, False if failed
        """
        return True

    def activate(self) -> bool:
        """
        Activate the module - start processing.
        
        📖 OPTIONAL TO OVERRIDE
        Start any background processes, listeners, etc.
        """
        return True

    def deactivate(self) -> bool:
        """
        Deactivate the module - stop processing.
        
        📖 OPTIONAL TO OVERRIDE
        Stop background processes, save state if needed.
        """
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current module status as a dictionary."""
        return {
            'id': self.INFO.id,
            'state': self.state.value,
            'config': self.config,
        }

    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Update module configuration.
        
        📖 OVERRIDE IF YOU NEED VALIDATION
        Default just merges the new config in.

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
        
        📖 WHAT THIS RETURNS:
        The main object other code should interact with.
        For example:
        - Model module → returns the Forge model
        - Image gen → returns the image generator object
        - Memory → returns the ConversationManager

        Returns:
            The module's working instance
        """
        return self._instance

    def is_loaded(self) -> bool:
        """
        Check if the module is loaded and usable.

        Returns:
            True if module is in LOADED or ACTIVE state
        """
        return self.state in (ModuleState.LOADED, ModuleState.ACTIVE)


# =============================================================================
# 🎛️ MODULE MANAGER - The Central Control System
# =============================================================================

class ModuleManager:
    """
    Central manager for all Forge modules.
    
    📖 WHAT THIS DOES:
    This is the BOSS that controls all modules! It handles:
    
    ⚡ KEY RESPONSIBILITIES:
    1. MODULE DISCOVERY: Find all available modules
    2. DEPENDENCY RESOLUTION: Load modules in correct order
    3. CONFLICT PREVENTION: Stop incompatible modules from loading together
    4. HARDWARE CHECKING: Verify system can run the module
    5. LIFECYCLE MANAGEMENT: Load, activate, deactivate, unload
    6. CONFIGURATION: Save/load module settings
    7. HEALTH MONITORING: Track module health over time
    
    📐 EXAMPLE USAGE:
    
        manager = ModuleManager()
        
        # Load modules in dependency order
        manager.load('model')           # Load first (core)
        manager.load('tokenizer')       # Load second (core)
        manager.load('inference')       # Load third (depends on model+tokenizer)
        manager.load('image_gen_local') # Load Stable Diffusion
        
        # Get the loaded module and use it
        image_module = manager.get_module('image_gen_local')
        image = image_module.generate("a sunset over mountains")
        
        # Unload when done to free memory
        manager.unload('image_gen_local')
    
    📐 CONFLICT EXAMPLE:
    
        manager.load('image_gen_local')  # OK - loads Stable Diffusion
        manager.load('image_gen_api')    # FAILS! - conflicts with local
        # Error: Cannot load image_gen_api - conflicts with image_gen_local
    
    🔗 CONNECTS TO:
      → Uses module classes from registry.py
      → Saves config to data/module_config.json
      ← Used by GUI modules tab, CLI, and all module consumers
    Attributes:
        modules: Dict mapping module ID to loaded ``Module`` instance.
        module_classes: Dict mapping module ID to registered module class.
        config_path: ``Path`` to the JSON file where module configuration
            is persisted.
        hardware_profile: Dict of detected hardware capabilities (CPU
            cores, RAM, GPU name, VRAM, recommended model size, etc.).
        local_only: If ``True``, cloud-service modules are blocked.
    """
    
    # Singleton instance storage
    _instance: Optional[ModuleManager] = None
    _initialized: bool = False
    
    def __new__(cls, config_path: Optional[Path] = None, local_only: bool = True):
        """
        Singleton pattern: Return existing instance if available.
        
        This ensures that all code calling ModuleManager() gets the same
        instance, preventing duplicate module registrations and state
        inconsistencies.
        """
        global _manager
        
        # If global singleton exists, return it
        if _manager is not None:
            return _manager
        
        # If class-level singleton exists, return it  
        if cls._instance is not None:
            return cls._instance
        
        # Create new instance
        instance = super().__new__(cls)
        cls._instance = instance
        _manager = instance
        return instance

    def __init__(self, config_path: Optional[Path] = None, local_only: bool = True):
        """
        Initialize the Module Manager.
        
        Args:
            config_path: Where to save/load module configuration
            local_only: If True, only allow local modules (no cloud APIs)
        """
        # Singleton guard: Only initialize once
        if ModuleManager._initialized:
            return
        ModuleManager._initialized = True
        
        # ─────────────────────────────────────────────────────────────────────
        # THREAD SAFETY: Lock for module operations
        # ─────────────────────────────────────────────────────────────────────
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._module_locks: Dict[str, threading.Lock] = {}  # Per-module locks
        
        # ─────────────────────────────────────────────────────────────────────
        # MODULE STORAGE
        # ─────────────────────────────────────────────────────────────────────
        self.modules: Dict[str, Module] = {}          # Loaded module instances
        self.module_classes: Dict[str, type] = {}     # Registered module classes
        self.config_path = config_path or Path("data/module_config.json")
        self.hardware_profile: Dict[str, Any] = {}    # Detected hardware
        self.local_only = local_only  # Privacy: block cloud modules by default
        
        # Device profile integration for smart hardware decisions
        self._device_profile = None

        # ─────────────────────────────────────────────────────────────────────
        # EVENT CALLBACKS: Notify listeners when things happen
        # ─────────────────────────────────────────────────────────────────────
        self._on_load: List[Callable] = []        # Called when module loads
        self._on_unload: List[Callable] = []      # Called when module unloads
        self._on_state_change: List[Callable] = [] # Called on any state change

        # ─────────────────────────────────────────────────────────────────────
        # HEALTH MONITORING: Background thread checks module health
        # ─────────────────────────────────────────────────────────────────────
        self._health_monitor_thread: Optional[threading.Thread] = None
        self._health_monitor_running: bool = False
        self._health_monitor_interval: int = 60  # Check every 60 seconds
        self._health_monitor_stop_event: threading.Event = threading.Event()
        self._module_error_counts: Dict[str, int] = {}

        # ─────────────────────────────────────────────────────────────────────
        # DETECT HARDWARE: Figure out what this system can run
        # ─────────────────────────────────────────────────────────────────────
        self._detect_hardware()
        
        # ─────────────────────────────────────────────────────────────────────
        # AUTO-REGISTER ALL BUILT-IN MODULES
        # ─────────────────────────────────────────────────────────────────────
        self._auto_register_modules()
        
    def _auto_register_modules(self):
        """Auto-register all built-in modules from the registry."""
        try:
            from .registry import register_all
            register_all(self)
            logger.info(f"Auto-registered {len(self.module_classes)} modules")
        except Exception as e:
            logger.warning(f"Could not auto-register modules: {e}")
        
    @property
    def device_profile(self):
        """Get device profile for hardware-aware decisions."""
        if self._device_profile is None:
            profiler = _get_device_profiler()
            if profiler:
                self._device_profile = profiler.get_profile()
        return self._device_profile
    
    def get_recommended_modules(self) -> List[str]:
        """
        Get recommended modules based on device capabilities.
        
        Returns list of module IDs that should work well on this hardware.
        """
        recommended = []
        profile = self.device_profile
        
        if not profile:
            return ["tokenizer", "model", "inference", "memory"]
        
        # Core modules - always recommended
        recommended.extend(["tokenizer", "memory"])
        
        # Model choice based on hardware
        if profile.use_gpu or profile.recommended_model_size in ["medium", "large", "xl"]:
            recommended.append("model")
            recommended.append("inference")
        else:
            # Low-power devices might prefer cloud API or GGUF
            if profile.recommended_model_size in ["nano", "tiny"]:
                recommended.append("chat_api")  # Ollama or cloud
            else:
                recommended.append("model")
                recommended.append("inference")
        
        # Generation modules based on VRAM
        if self.hardware_profile.get('vram_mb', 0) > 4000:
            recommended.append("image_gen_local")
        elif self.hardware_profile.get('vram_mb', 0) > 0:
            # Some GPU but limited VRAM
            pass  # No heavy generation
        
        # Voice - works on most devices
        recommended.extend(["voice_input", "voice_output"])
        
        return recommended
    
    def auto_configure(self, mode: str = "inference") -> Dict[str, bool]:
        """
        Automatically configure modules based on hardware.
        
        Args:
            mode: "inference" (default), "training", "server", or "minimal"
            
        Returns:
            Dict of module_id -> success status
        """
        results = {}
        profile = self.device_profile
        
        if mode == "minimal":
            # Bare minimum for chat
            modules = ["tokenizer"]
            # Prefer cloud API for minimal setups
            if self.hardware_profile.get('ram_mb', 0) < 4096:
                modules.append("chat_api")
            else:
                modules.extend(["model", "inference"])
        elif mode == "inference":
            modules = self.get_recommended_modules()
        elif mode == "training":
            modules = ["tokenizer", "model", "training", "memory"]
        elif mode == "server":
            modules = ["tokenizer", "model", "inference", "memory", "api_server"]
        else:
            modules = self.get_recommended_modules()
        
        # Load modules in dependency order
        for module_id in modules:
            if module_id in self.module_classes:
                can_load, reason = self.can_load(module_id)
                if can_load:
                    results[module_id] = self.load(module_id)
                else:
                    logger.info(f"Skipping {module_id}: {reason}")
                    results[module_id] = False
            else:
                results[module_id] = False
        
        return results

    def _detect_hardware(self):
        """
        Detect available hardware capabilities.
        
        📖 WHAT THIS CHECKS:
        - CPU cores (for parallel processing)
        - RAM (for model loading)
        - GPU availability and VRAM (for fast inference)
        - Apple MPS (for M1/M2 Macs)
        - Device class (embedded, mobile, desktop, server)
        """
        # Start with safe defaults
        self.hardware_profile = {
            'cpu_cores': 1,
            'ram_mb': 4096,
            'gpu_available': False,
            'gpu_name': None,
            'vram_mb': 0,
            'mps_available': False,
            'device_class': 'unknown',
            'recommended_model_size': 'tiny',
        }
        
        # Try to use device profiler for comprehensive detection
        profiler = _get_device_profiler()
        if profiler:
            try:
                caps = profiler.detect()
                profile = profiler.get_profile()
                
                self.hardware_profile.update({
                    'cpu_cores': caps.cpu_cores,
                    'ram_mb': caps.ram_total_mb,
                    'gpu_available': caps.has_cuda or caps.has_mps,
                    'gpu_name': caps.gpu_name,
                    'vram_mb': caps.vram_total_mb,
                    'mps_available': caps.has_mps,
                    'device_class': profiler.classify().name,
                    'recommended_model_size': profile.recommended_model_size,
                    'is_raspberry_pi': caps.is_raspberry_pi,
                    'is_android': caps.is_android,
                    'is_mobile': caps.is_android or caps.is_ios,
                })
                logger.info(f"Hardware detected: {self.hardware_profile['device_class']}, "
                           f"recommended model: {self.hardware_profile['recommended_model_size']}")
                return
            except Exception as e:
                logger.warning(f"Device profiler error: {e}, falling back to basic detection")

        # Fallback: Try to detect GPU with torch (cached to avoid repeated imports)
        torch = _get_torch()
        if torch:
            try:
                self.hardware_profile['gpu_available'] = torch.cuda.is_available()
                self.hardware_profile['mps_available'] = hasattr(
                    torch.backends, 'mps') and torch.backends.mps.is_available()

                if torch.cuda.is_available():
                    self.hardware_profile['gpu_name'] = torch.cuda.get_device_name(0)
                    self.hardware_profile['vram_mb'] = torch.cuda.get_device_properties(
                        0).total_memory // (1024 * 1024)
            except Exception as e:
                logger.warning(f"Error detecting GPU: {e}")

        # Try to detect CPU/RAM with psutil
        try:
            import psutil
            self.hardware_profile['cpu_cores'] = psutil.cpu_count()
            self.hardware_profile['ram_mb'] = psutil.virtual_memory().total // (1024 * 1024)
        except ImportError:
            logger.debug("psutil not available, using default CPU/RAM values")
        
        # Determine recommended model size based on hardware
        ram_gb = self.hardware_profile['ram_mb'] / 1024
        vram_gb = self.hardware_profile['vram_mb'] / 1024
        
        if vram_gb >= 24 or ram_gb >= 64:
            self.hardware_profile['recommended_model_size'] = 'xl'
        elif vram_gb >= 12 or ram_gb >= 32:
            self.hardware_profile['recommended_model_size'] = 'large'
        elif vram_gb >= 6 or ram_gb >= 16:
            self.hardware_profile['recommended_model_size'] = 'medium'
        elif vram_gb >= 4 or ram_gb >= 8:
            self.hardware_profile['recommended_model_size'] = 'small'
        elif ram_gb >= 4:
            self.hardware_profile['recommended_model_size'] = 'tiny'
        else:
            self.hardware_profile['recommended_model_size'] = 'nano'

    def register(self, module_class: type) -> bool:
        """
        Register a module class (make it available for loading).
        
        📖 WHAT THIS DOES:
        Adds a module class to the registry so it can be loaded later.
        This doesn't load the module - just makes it available.
        
        📐 EXAMPLE:
            manager.register(ImageGenLocalModule)
            # Now 'image_gen_local' can be loaded with manager.load()

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
        """Unregister a module class (remove from available modules)."""
        if module_id in self.module_classes:
            # Unload if loaded
            if module_id in self.modules:
                self.unload(module_id)
            del self.module_classes[module_id]
            return True
        return False

    # =========================================================================
    # 🔍 VALIDATION: Can this module be loaded?
    # =========================================================================

    def can_load(self, module_id: str) -> Tuple[bool, str]:
        """
        Check if a module can be loaded.
        
        📖 WHAT THIS CHECKS:
        This runs a series of safety checks before allowing a module to load:
        
        📐 CHECK ORDER:
        1. Does module exist? (registered?)
        2. Privacy: Is it cloud but we're in local-only mode?
        3. Hardware: Do we have GPU if required?
        4. VRAM: Do we have enough GPU memory?
        5. RAM: Do we have enough system memory?
        6. Conflicts: Are any conflicting modules loaded?
        7. Capability: Is another module already providing this feature?
        8. Dependencies: Are all required modules loaded?

        Returns:
            (can_load, reason) - True/OK if can load, False/reason if not
        """
        # ─────────────────────────────────────────────────────────────────────
        # CHECK 1: Does the module exist?
        # ─────────────────────────────────────────────────────────────────────
        if module_id not in self.module_classes:
            return False, f"Module '{module_id}' not registered"

        info = self.module_classes[module_id].get_info()

        # ─────────────────────────────────────────────────────────────────────
        # CHECK 2: Privacy - is it cloud when we want local only?
        # ─────────────────────────────────────────────────────────────────────
        if self.local_only and info.is_cloud_service:
            return False, "Module requires external cloud services. Disable local_only mode to use cloud modules."

        # ─────────────────────────────────────────────────────────────────────
        # CHECK 3: Hardware - GPU required?
        # ─────────────────────────────────────────────────────────────────────
        if info.requires_gpu and not self.hardware_profile['gpu_available']:
            return False, "Module requires GPU but none available"

        # ─────────────────────────────────────────────────────────────────────
        # CHECK 4: VRAM - enough GPU memory?
        # ─────────────────────────────────────────────────────────────────────
        if info.min_vram_mb > self.hardware_profile['vram_mb']:
            return False, f"Module requires {info.min_vram_mb}MB VRAM, only {self.hardware_profile['vram_mb']}MB available"

        # ─────────────────────────────────────────────────────────────────────
        # CHECK 5: RAM - enough system memory?
        # ─────────────────────────────────────────────────────────────────────
        if info.min_ram_mb > self.hardware_profile['ram_mb']:
            return False, f"Module requires {info.min_ram_mb}MB RAM, only {self.hardware_profile['ram_mb']}MB available"

        # ─────────────────────────────────────────────────────────────────────
        # CHECK 6: Explicit conflicts - module declares it can't run with another
        # ─────────────────────────────────────────────────────────────────────
        for conflict_id in info.conflicts:
            if conflict_id in self.modules and self.modules[conflict_id].state == ModuleState.LOADED:
                return False, f"Module conflicts with loaded module '{conflict_id}'"

        # ─────────────────────────────────────────────────────────────────────
        # CHECK 7: Capability conflicts - two modules providing same feature
        # Example: image_gen_local and image_gen_api both provide 'image_generation'
        # Only one can be loaded at a time!
        # ─────────────────────────────────────────────────────────────────────
        for provided in info.provides:
            for loaded_id, loaded_module in self.modules.items():
                if loaded_module.state == ModuleState.LOADED:
                    loaded_info = loaded_module.get_info()
                    if provided in loaded_info.provides and loaded_id != module_id:
                        return False, f"Capability '{provided}' already provided by '{loaded_id}'. Unload it first."

        # ─────────────────────────────────────────────────────────────────────
        # CHECK 8: Dependencies - are required modules loaded?
        # ─────────────────────────────────────────────────────────────────────
        for dep_id in info.requires:
            if dep_id not in self.modules or self.modules[dep_id].state != ModuleState.LOADED:
                return False, f"Required module '{dep_id}' not loaded"

        # ─────────────────────────────────────────────────────────────────────
        # ALL CHECKS PASSED!
        # ─────────────────────────────────────────────────────────────────────
        return True, "OK"

    # =========================================================================
    # 📦 LOADING: Initialize and start a module
    # =========================================================================

    def _get_module_lock(self, module_id: str) -> threading.Lock:
        """Get or create a per-module lock for fine-grained concurrency control."""
        with self._lock:
            if module_id not in self._module_locks:
                self._module_locks[module_id] = threading.Lock()
            return self._module_locks[module_id]

    def load(self, module_id: str, config: Dict[str, Any] = None) -> bool:
        """
        Load a module (thread-safe).
        
        📖 WHAT THIS DOES:
        1. Validates the module can be loaded (can_load checks)
        2. Creates an instance of the module class
        3. Calls the module's load() method to initialize resources
        4. Stores the module in our registry
        5. Notifies listeners that module was loaded
        
        📐 LOADING FLOW:
        
            can_load()  ──OK──▶  create instance  ──▶  module.load()
                │                                          │
                ▼                                          ▼
            return False                              success?
                                                       │    │
                                                      YES   NO
                                                       │    │
                                                       ▼    ▼
                                                    LOADED  ERROR

        Args:
            module_id: The unique string identifier of the module to load
                (e.g. ``"model"``, ``"image_gen_local"``).
            config: Optional dictionary of module-specific settings that
                is forwarded to the module constructor.

        Returns:
            ``True`` if the module was loaded (or was already loaded).
            ``False`` if any validation check failed or the module's own
            ``load()`` method returned ``False``.

        Raises:
            No exceptions are raised directly; failures are logged and
            ``False`` is returned.

        Example:
            >>> manager = ModuleManager()
            >>> manager.load("model")
            True
            >>> manager.load("image_gen_local", config={"resolution": 512})
            True
        """
        # Get per-module lock to prevent concurrent loading of same module
        module_lock = self._get_module_lock(module_id)
        
        with module_lock:
            # Double-check if already loaded (another thread may have loaded it)
            if module_id in self.modules and self.modules[module_id].state == ModuleState.LOADED:
                logger.debug(f"Module '{module_id}' already loaded")
                return True
            
            # First, validate with all our safety checks
            can_load, reason = self.can_load(module_id)
            if not can_load:
                logger.error(f"Cannot load module '{module_id}': {reason}")
                return False

            module_class = self.module_classes[module_id]
            module_info = module_class.get_info()

            # Privacy warning for cloud services
            if module_info.is_cloud_service:
                logger.warning(
                    f"Warning: Module '{module_id}' connects to external cloud services and requires API keys + internet.")

            try:
                # ─────────────────────────────────────────────────────────────────
                # STEP 1: Create the module instance
                # ─────────────────────────────────────────────────────────────────
                module = module_class(self, config)
                module.state = ModuleState.LOADING

                # ─────────────────────────────────────────────────────────────────
                # STEP 2: Call module's load() to initialize resources
                # This is where the module loads models, connects to APIs, etc.
                # ─────────────────────────────────────────────────────────────────
                if module.load():
                    # Success! Mark as loaded and store
                    module.state = ModuleState.LOADED
                    module.get_info().load_time = datetime.now()
                    
                    # Use main lock for modifying shared modules dict
                    with self._lock:
                        self.modules[module_id] = module

                    # ─────────────────────────────────────────────────────────────
                    # STEP 3: Register with capability registry (orchestrator)
                    # ─────────────────────────────────────────────────────────────
                    self._register_module_capabilities(module_id, module_info)

                    # ─────────────────────────────────────────────────────────────
                    # STEP 4: Notify any listeners (GUI, etc.)
                    # ─────────────────────────────────────────────────────────────
                    for callback in self._on_load:
                        try:
                            callback(module_id)
                        except Exception as e:
                            logger.warning(f"Error in load callback: {e}")

                    logger.info(f"Loaded module: {module_id}")
                    return True
                else:
                    # Module's load() returned False - something went wrong
                    module.state = ModuleState.ERROR
                    module.get_info().error_message = "load() returned False"
                    return False

            except Exception as e:
                logger.error(f"Error loading module '{module_id}': {e}")
                return False

    def load_sandboxed(
        self, 
        module_id: str, 
        sandbox_config: Optional[Any] = None,
        config: Dict[str, Any] = None
    ) -> bool:
        """
        Load a module in a sandboxed environment.
        
        📖 WHAT THIS IS:
        Sandboxing provides extra security for untrusted modules by:
        - Limiting file system access
        - Restricting network access
        - Controlling resource usage
        - Isolating the module from the main process
        
        📐 USE CASES:
        - Third-party plugins
        - Experimental modules
        - Modules from unknown sources
        
        Args:
            module_id: Module ID to load
            sandbox_config: Sandbox configuration (SandboxConfig, uses defaults if None)
            config: Optional module configuration
            
        Returns:
            True if loaded successfully
        """
        from .sandbox import ModuleSandbox, create_default_sandbox_config

        # Create default sandbox config if not provided
        if sandbox_config is None:
            sandbox_config = create_default_sandbox_config(module_id)
        
        # Check if module can be loaded
        can_load, reason = self.can_load(module_id)
        if not can_load:
            logger.error(f"Cannot load module '{module_id}': {reason}")
            return False
        
        module_class = self.module_classes[module_id]
        module_info = module_class.get_info()
        
        logger.info(f"Loading module '{module_id}' in sandbox")
        
        # Create sandbox environment
        sandbox = ModuleSandbox(module_id, sandbox_config)
        
        try:
            # Create module instance
            module = module_class(self, config)
            module.state = ModuleState.LOADING
            
            # Load module inside sandbox (restricted environment)
            def load_func():
                return module.load()
            
            success = sandbox.run_in_sandbox(load_func)
            
            if success:
                module.state = ModuleState.LOADED
                module.get_info().load_time = datetime.now()
                self.modules[module_id] = module
                
                # Keep sandbox reference for future sandboxed calls
                module._sandbox = sandbox
                
                # Notify listeners
                for callback in self._on_load:
                    callback(module_id)
                
                logger.info(f"Loaded module '{module_id}' in sandbox")
                return True
            else:
                module.state = ModuleState.ERROR
                module.get_info().error_message = "load() returned False in sandbox"
                return False
                
        except Exception as e:
            logger.error(f"Error loading module '{module_id}' in sandbox: {e}")
            return False

    # =========================================================================
    # 🗑️ UNLOADING: Release module resources
    # =========================================================================

    def unload(self, module_id: str) -> bool:
        """
        Unload a module and release its resources (thread-safe).

        Performs the following steps:

        1. Checks that no other loaded module declares a dependency on
           this one.  If module *B* requires module *A*, you must unload
           *B* before *A*.
        2. Calls the module's own ``unload()`` method so it can free GPU
           memory, close network connections, save state, etc.
        3. Removes the module from the internal ``modules`` dict.
        4. Fires all registered ``_on_unload`` callbacks to notify
           listeners (e.g. the GUI Modules tab).

        Args:
            module_id: The unique identifier of the module to unload
                (e.g. ``"image_gen_local"``).

        Returns:
            ``True`` if the module was successfully unloaded.
            ``False`` if the module was not loaded, is still required by
            another module, or its ``unload()`` method returned ``False``.

        Raises:
            No exceptions are raised directly; errors are logged and
            ``False`` is returned.

        Example:
            >>> manager.unload("image_gen_local")
            True
            >>> manager.get_module("image_gen_local") is None
            True
        """
        # Get per-module lock
        module_lock = self._get_module_lock(module_id)
        
        with module_lock:
            if module_id not in self.modules:
                return False

            module = self.modules[module_id]

            # ─────────────────────────────────────────────────────────────────────
            # DEPENDENCY CHECK: Are other modules depending on this one?
            # ─────────────────────────────────────────────────────────────────────
            with self._lock:  # Lock for reading modules dict
                for other_id, other_module in self.modules.items():
                    if other_id != module_id:
                        info = other_module.get_info()
                        if module_id in info.requires and other_module.state == ModuleState.LOADED:
                            # Can't unload! Something else needs this module
                            logger.error(f"Cannot unload '{module_id}': required by '{other_id}'")
                            return False

            try:
                # ─────────────────────────────────────────────────────────────────
                # STEP 1: Call module's unload() to release resources
                # This is where models are freed from memory, connections closed, etc.
                # ─────────────────────────────────────────────────────────────────
                if module.unload():
                    module.state = ModuleState.UNLOADED
                    
                    # Use main lock for modifying shared modules dict
                    with self._lock:
                        del self.modules[module_id]

                    # ─────────────────────────────────────────────────────────────
                    # STEP 2: Notify listeners (GUI, etc.)
                    # ─────────────────────────────────────────────────────────────
                    for callback in self._on_unload:
                        try:
                            callback(module_id)
                        except Exception as e:
                            logger.warning(f"Error in unload callback: {e}")

                    logger.info(f"Unloaded module: {module_id}")
                    return True
                return False

            except Exception as e:
                logger.error(f"Error unloading module '{module_id}': {e}")
                return False

    # =========================================================================
    # ⚡ ACTIVATION: Start/stop module processing
    # =========================================================================

    def activate(self, module_id: str) -> bool:
        """
        Activate a loaded module.
        
        📖 LOADED vs ACTIVE:
        - LOADED: Module is initialized and ready, but not processing
        - ACTIVE: Module is running background tasks, listening for events, etc.
        
        Some modules don't need activation (they work immediately when loaded).
        Others (like voice listener) need explicit activation to start processing.

        Args:
            module_id: Module ID to activate

        Returns:
            True if activated successfully
        """
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
        """
        Deactivate an active module.
        
        📖 WHAT THIS DOES:
        Stops the module's background processing but keeps it loaded.
        Useful for pausing expensive operations without full unload.
        
        Args:
            module_id: Module ID to deactivate

        Returns:
            True if deactivated successfully
        """
        if module_id not in self.modules:
            return False

        module = self.modules[module_id]
        if module.state != ModuleState.ACTIVE:
            return False

        if module.deactivate():
            module.state = ModuleState.LOADED
            return True
        return False

    # =========================================================================
    # 🔍 GETTERS: Access loaded modules
    # =========================================================================

    def get_module(self, module_id: str) -> Optional[Module]:
        """Get a loaded module instance by its unique ID.

        Returns the ``Module`` wrapper object that carries lifecycle state,
        configuration, and metadata.  To obtain the inner working object
        (e.g. the actual image generator or model), call
        ``get_interface()`` on the returned module -- or use the
        convenience method ``manager.get_interface(module_id)``.

        Args:
            module_id: The unique string identifier of the module
                (e.g. ``"image_gen_local"``, ``"model"``).

        Returns:
            The ``Module`` instance if the module is currently loaded,
            or ``None`` if it has not been loaded or has been unloaded.

        Example:
            >>> mod = manager.get_module("inference")
            >>> if mod and mod.is_loaded():
            ...     engine = mod.get_interface()
            ...     response = engine.generate("Hello world")
        """
        return self.modules.get(module_id)

    def get_interface(self, module_id: str) -> Any:
        """
        Get a module's public interface.
        
        📖 WHAT THIS RETURNS:
        The actual working object inside the module.
        This is what you use to call module functions.
        
        📐 EXAMPLE:
            image_gen = manager.get_interface('image_gen_local')
            result = image_gen.generate("a sunset")  # Use the actual generator

        Args:
            module_id: Module ID

        Returns:
            The module's working instance (model, generator, etc.)
        """
        module = self.modules.get(module_id)
        return module.get_interface() if module else None

    def is_loaded(self, module_id: str) -> bool:
        """Check whether a module is currently loaded.

        Args:
            module_id: The unique identifier of the module.

        Returns:
            ``True`` if the module exists in the loaded modules dict
            and its state is ``ModuleState.LOADED``.
        """
        return module_id in self.modules and self.modules[module_id].state == ModuleState.LOADED

    def list_modules(self, category: Optional[ModuleCategory] = None) -> List[ModuleInfo]:
        """List all registered modules, optionally filtered by category.

        Returns ``ModuleInfo`` objects for every module class that has been
        registered with the manager.  If a module is currently loaded, its
        ``state`` field is updated to reflect the live runtime state.

        Args:
            category: If provided, only modules whose
                ``ModuleInfo.category`` matches this value are returned.
                Pass ``None`` (default) to list modules from all
                categories.

        Returns:
            A list of ``ModuleInfo`` dataclass instances, one per
            registered module (or per matching module when *category* is
            set).

        Example:
            >>> # List all generation modules
            >>> gen_mods = manager.list_modules(ModuleCategory.GENERATION)
            >>> for m in gen_mods:
            ...     print(m.id, m.state.value)
            'image_gen_local' 'unloaded'
            'code_gen_local'  'loaded'
        """
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
        """Return the IDs of all currently loaded modules.

        Returns:
            A list of module ID strings in no guaranteed order.

        Example:
            >>> manager.list_loaded()
            ['model', 'tokenizer', 'inference']
        """
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
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage with all loaded modules.
        
        Returns detailed metrics about:
        - Memory (RAM and VRAM)
        - CPU usage
        - Module counts and sizes
        - Estimated overhead
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'modules_loaded': len(self.modules),
            'modules_registered': len(self.module_classes),
            'categories_active': {},
        }
        
        # Count modules by category
        for module in self.modules.values():
            info = module.get_info()
            cat = info.category.value
            metrics['categories_active'][cat] = metrics['categories_active'].get(cat, 0) + 1
        
        # Get system resource usage
        try:
            import os

            import psutil

            # Current process
            process = psutil.Process(os.getpid())
            
            # Memory usage
            mem_info = process.memory_info()
            metrics['memory'] = {
                'rss_mb': mem_info.rss / (1024 * 1024),  # Resident Set Size
                'vms_mb': mem_info.vms / (1024 * 1024),  # Virtual Memory Size
                'percent': process.memory_percent(),
                'system_total_mb': psutil.virtual_memory().total / (1024 * 1024),
                'system_available_mb': psutil.virtual_memory().available / (1024 * 1024),
                'system_used_percent': psutil.virtual_memory().percent,
            }
            
            # CPU usage
            metrics['cpu'] = {
                'percent': process.cpu_percent(interval=0.1),
                'num_threads': process.num_threads(),
                'system_percent': psutil.cpu_percent(interval=0.1),
                'cores': psutil.cpu_count(),
            }
            
        except ImportError:
            metrics['memory'] = {'note': 'psutil not available for detailed metrics'}
            metrics['cpu'] = {'note': 'psutil not available for detailed metrics'}
        except Exception as e:
            metrics['error'] = f"Error getting resource usage: {e}"
        
        # GPU/VRAM usage if available
        torch = _get_torch()
        if torch and torch.cuda.is_available():
            try:
                metrics['gpu'] = {
                    'device_name': torch.cuda.get_device_name(0),
                    'allocated_mb': torch.cuda.memory_allocated(0) / (1024 * 1024),
                    'reserved_mb': torch.cuda.memory_reserved(0) / (1024 * 1024),
                    'max_allocated_mb': torch.cuda.max_memory_allocated(0) / (1024 * 1024),
                    'total_mb': torch.cuda.get_device_properties(0).total_memory / (1024 * 1024),
                }
                metrics['gpu']['used_percent'] = (
                    metrics['gpu']['allocated_mb'] / metrics['gpu']['total_mb'] * 100
                )
            except Exception as e:
                metrics['gpu'] = {'error': str(e)}
        else:
            metrics['gpu'] = {'available': False}
        
        # Module-specific requirements
        total_min_ram = 0
        total_min_vram = 0
        gpu_modules = []
        cloud_modules = []
        
        for module in self.modules.values():
            info = module.get_info()
            total_min_ram += info.min_ram_mb
            total_min_vram += info.min_vram_mb
            if info.requires_gpu:
                gpu_modules.append(info.id)
            if info.is_cloud_service:
                cloud_modules.append(info.id)
        
        metrics['requirements'] = {
            'total_min_ram_mb': total_min_ram,
            'total_min_vram_mb': total_min_vram,
            'gpu_modules': gpu_modules,
            'gpu_module_count': len(gpu_modules),
            'cloud_modules': cloud_modules,
            'cloud_module_count': len(cloud_modules),
        }
        
        # Estimate overhead
        if 'memory' in metrics and 'rss_mb' in metrics['memory']:
            base_overhead_mb = 200  # Estimated base Python + Forge overhead
            estimated_module_memory = metrics['memory']['rss_mb'] - base_overhead_mb
            metrics['estimates'] = {
                'base_overhead_mb': base_overhead_mb,
                'modules_memory_mb': max(0, estimated_module_memory),
                'avg_per_module_mb': (
                    estimated_module_memory / len(self.modules) if self.modules else 0
                ),
            }
        
        # Performance impact assessment
        metrics['assessment'] = self._assess_performance(metrics)
        
        return metrics
    
    def _assess_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall performance and provide recommendations."""
        assessment = {
            'status': 'good',
            'warnings': [],
            'recommendations': [],
        }
        
        # Check memory
        if 'memory' in metrics and 'system_used_percent' in metrics['memory']:
            mem_percent = metrics['memory']['system_used_percent']
            if mem_percent > 90:
                assessment['status'] = 'critical'
                assessment['warnings'].append(f"System memory at {mem_percent:.1f}% - may cause instability")
                assessment['recommendations'].append("Unload unused modules or close other applications")
            elif mem_percent > 75:
                assessment['status'] = 'warning'
                assessment['warnings'].append(f"System memory at {mem_percent:.1f}% - getting high")
                assessment['recommendations'].append("Consider unloading some modules")
        
        # Check GPU memory
        if 'gpu' in metrics and 'used_percent' in metrics['gpu']:
            gpu_percent = metrics['gpu']['used_percent']
            if gpu_percent > 90:
                assessment['status'] = 'critical'
                assessment['warnings'].append(f"GPU memory at {gpu_percent:.1f}% - may cause OOM errors")
                assessment['recommendations'].append("Unload GPU-intensive modules (image/video generation)")
            elif gpu_percent > 75:
                if assessment['status'] == 'good':
                    assessment['status'] = 'warning'
                assessment['warnings'].append(f"GPU memory at {gpu_percent:.1f}%")
        
        # Check module count
        loaded_count = metrics.get('modules_loaded', 0)
        if loaded_count > 15:
            assessment['recommendations'].append(
                f"{loaded_count} modules loaded - consider unloading unused ones for better performance"
            )
        
        # Check if too many cloud services (privacy concern)
        if 'requirements' in metrics:
            cloud_count = metrics['requirements'].get('cloud_module_count', 0)
            if cloud_count > 3:
                assessment['warnings'].append(
                    f"{cloud_count} cloud modules active - data is being sent to external services"
                )
        
        return assessment

    def save_config(self, path: Optional[Path] = None):
        """Save current module configuration using atomic write to prevent corruption."""
        from ..utils.io_utils import atomic_save_json
        
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

        atomic_save_json(path, config, indent=2)

    def load_config(self, path: Optional[Path] = None) -> bool:
        """Load and apply module configuration."""
        path = path or self.config_path

        if not path.exists():
            return False

        with open(path) as f:
            config = json.load(f)

        # Load modules in dependency order
        for module_id, module_config in config.get('loaded_modules', {}).items():
            if module_id in self.module_classes:
                self.load(module_id, module_config.get('config'))
                if module_config.get('active'):
                    self.activate(module_id)

        return True

    def health_check(self, module_id: str) -> Optional[ModuleHealth]:
        """
        Run health check on a specific module.
        
        Args:
            module_id: Module ID to check
            
        Returns:
            ModuleHealth object with status, or None if module not loaded
        """
        if module_id not in self.modules:
            logger.warning(f"Cannot check health of unloaded module: {module_id}")
            return None
        
        module = self.modules[module_id]
        warnings = []
        is_healthy = True
        
        # Measure response time
        start_time = time.time()
        
        try:
            # Basic health check - try to get module status
            status = module.get_status()
            
            # Check module state
            if module.state == ModuleState.ERROR:
                is_healthy = False
                warnings.append("Module is in ERROR state")
            
            # Check if module has error message
            info = module.get_info()
            if info.error_message:
                warnings.append(f"Error message: {info.error_message}")
            
            # Check resource usage (basic checks)
            try:
                import os

                import psutil
                process = psutil.Process(os.getpid())
                mem_percent = process.memory_percent()
                
                if mem_percent > 80:
                    warnings.append(f"High memory usage: {mem_percent:.1f}%")
                
            except (ImportError, Exception) as e:
                logger.debug(f"psutil not available or error during health check: {e}")
            
        except Exception as e:
            is_healthy = False
            warnings.append(f"Health check exception: {str(e)}")
            logger.error(f"Error during health check of '{module_id}': {e}")
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Track error count
        error_count = self._module_error_counts.get(module_id, 0)
        if not is_healthy:
            error_count += 1
            self._module_error_counts[module_id] = error_count
        else:
            # Reset error count on successful check
            self._module_error_counts[module_id] = 0
        
        return ModuleHealth(
            module_id=module_id,
            is_healthy=is_healthy,
            last_check=datetime.now(),
            response_time_ms=response_time_ms,
            error_count=error_count,
            warnings=warnings
        )
    
    def health_check_all(self) -> Dict[str, ModuleHealth]:
        """
        Run health checks on all loaded modules.
        
        Returns:
            Dictionary mapping module IDs to their health status
        """
        results = {}
        
        for module_id in self.modules.keys():
            health = self.health_check(module_id)
            if health:
                results[module_id] = health
        
        return results
    
    def _health_monitor_loop(self):
        """Background thread loop for health monitoring."""
        logger.info(f"Health monitor started (interval: {self._health_monitor_interval}s)")
        
        while self._health_monitor_running:
            try:
                # Run health checks on all modules
                results = self.health_check_all()
                
                # Log any unhealthy modules
                for module_id, health in results.items():
                    if not health.is_healthy:
                        logger.warning(
                            f"Module '{module_id}' is unhealthy: "
                            f"{', '.join(health.warnings)}"
                        )
                    elif health.warnings:
                        logger.info(
                            f"Module '{module_id}' has warnings: "
                            f"{', '.join(health.warnings)}"
                        )
                
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
            
            # Interruptible sleep using Event.wait()
            # Returns True if event is set (stop requested), False on timeout
            if self._health_monitor_stop_event.wait(timeout=self._health_monitor_interval):
                break  # Stop was requested
        
        logger.info("Health monitor stopped")
    
    def start_health_monitor(self, interval_seconds: int = 60):
        """
        Start background health monitoring.
        
        Args:
            interval_seconds: How often to check module health (default: 60s)
        """
        if self._health_monitor_running:
            logger.warning("Health monitor is already running")
            return
        
        self._health_monitor_interval = interval_seconds
        self._health_monitor_running = True
        self._health_monitor_stop_event.clear()  # Reset the stop event
        
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="ModuleHealthMonitor"
        )
        self._health_monitor_thread.start()
        
        logger.info(f"Started health monitor with {interval_seconds}s interval")
    
    def stop_health_monitor(self):
        """Stop background health monitoring."""
        if not self._health_monitor_running:
            logger.warning("Health monitor is not running")
            return
        
        logger.info("Stopping health monitor...")
        self._health_monitor_running = False
        self._health_monitor_stop_event.set()  # Signal the thread to wake up and stop
        
        if self._health_monitor_thread:
            self._health_monitor_thread.join(timeout=5.0)
            self._health_monitor_thread = None
        
        logger.info("Health monitor stopped")
    
    def is_health_monitor_running(self) -> bool:
        """
        Check if health monitor is currently running.
        
        Returns:
            True if health monitor is active
        """
        return self._health_monitor_running
    
    # =========================================================================
    # 🎯 ORCHESTRATOR INTEGRATION - Capability Registration
    # =========================================================================
    
    def _register_module_capabilities(self, module_id: str, module_info: ModuleInfo) -> None:
        """
        Register module capabilities with the orchestrator.
        
        Maps module types to capabilities and registers them with the
        capability registry for intelligent routing.
        
        Args:
            module_id: Module identifier
            module_info: Module information
        """
        try:
            from ..core.capability_registry import get_capability_registry
            
            registry = get_capability_registry()
            
            # Map module categories to capabilities
            capabilities = []
            
            if module_info.category == ModuleCategory.CORE:
                if "model" in module_id:
                    capabilities.extend(["text_generation", "reasoning"])
                elif "inference" in module_id:
                    capabilities.extend(["text_generation"])
                elif "code" in module_id:
                    capabilities.extend(["code_generation"])
            
            elif module_info.category == ModuleCategory.GENERATION:
                if "image_gen" in module_id:
                    capabilities.append("image_generation")
                elif "code_gen" in module_id:
                    capabilities.append("code_generation")
                elif "video_gen" in module_id:
                    capabilities.append("video_generation")
                elif "audio_gen" in module_id:
                    capabilities.append("audio_generation")
                elif "threed_gen" in module_id:
                    capabilities.append("3d_generation")
                elif "embedding" in module_id:
                    capabilities.append("embedding")
            
            elif module_info.category == ModuleCategory.PERCEPTION:
                if "vision" in module_id:
                    capabilities.append("vision")
                elif "camera" in module_id:
                    capabilities.append("vision")
            
            elif module_info.category == ModuleCategory.OUTPUT:
                if "tts" in module_id or "voice_output" in module_id:
                    capabilities.append("text_to_speech")
            
            elif module_info.category == ModuleCategory.INTERFACE:
                if "stt" in module_id or "voice_input" in module_id:
                    capabilities.append("speech_to_text")
            
            # Register if we found any capabilities
            if capabilities:
                metadata = {
                    "module_id": module_id,
                    "category": module_info.category.value,
                    "is_local": not module_info.is_cloud_service,
                    "description": module_info.description,
                }
                
                # Use "local:" prefix for local modules, module_id otherwise
                model_id = f"module:{module_id}"
                
                registry.register_model(
                    model_id=model_id,
                    capabilities=capabilities,
                    metadata=metadata,
                    auto_detect=False,
                )
                
                logger.debug(f"Registered {module_id} with capabilities: {capabilities}")
        
        except ImportError:
            # Orchestrator not available - that's OK
            pass
        except Exception as e:
            logger.warning(f"Failed to register module {module_id} with orchestrator: {e}")

    def on_load(self, callback: Callable):
        """Register callback for module load events."""
        self._on_load.append(callback)

    def on_unload(self, callback: Callable):
        """Register callback for module unload events."""
        self._on_unload.append(callback)

    def on_state_change(self, callback: Callable):
        """Register callback for module state changes."""
        self._on_state_change.append(callback)
    
    @classmethod
    def reset_singleton(cls) -> None:
        """
        Reset the singleton instance.
        
        Use this only in testing to get a fresh ModuleManager.
        In production, the singleton should persist for the lifetime
        of the application.
        """
        global _manager
        cls._instance = None
        cls._initialized = False
        _manager = None
        logger.debug("ModuleManager singleton reset")


# Global instance
_manager: Optional[ModuleManager] = None


def get_manager() -> ModuleManager:
    """
    Get the global module manager instance (singleton).
    
    This is the preferred way to access the ModuleManager.
    Calling ModuleManager() directly also returns the same singleton.
    """
    global _manager
    if _manager is None:
        _manager = ModuleManager()
    return _manager


def set_manager(manager: ModuleManager) -> None:
    """
    Set the global module manager instance.
    
    Use this to inject a custom manager (e.g., for testing).
    """
    global _manager
    _manager = manager
    ModuleManager._instance = manager
    ModuleManager._initialized = True


def reset_manager() -> None:
    """
    Reset the global module manager.
    
    Use this in testing to get a fresh instance.
    """
    ModuleManager.reset_singleton()
