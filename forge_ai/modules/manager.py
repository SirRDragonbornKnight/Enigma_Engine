"""
ForgeAI Module Manager
=====================

Central system for managing all Forge modules.
Handles loading, unloading, dependencies, and configuration.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple

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

    # Privacy and cloud requirements
    is_cloud_service: bool = False  # True if module connects to external cloud APIs

    # Capabilities provided
    provides: List[str] = field(default_factory=list)

    # Configuration schema
    config_schema: Dict[str, Any] = field(default_factory=dict)

    # Runtime info
    state: ModuleState = ModuleState.UNLOADED
    load_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class ModuleHealth:
    """Health status for a module."""
    module_id: str
    is_healthy: bool
    last_check: datetime
    response_time_ms: float
    error_count: int
    warnings: List[str] = field(default_factory=list)


class Module:
    """
    Base class for all Forge modules.

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
    Central manager for all Forge modules.

    Handles:
    - Module discovery and registration
    - Dependency resolution
    - Loading/unloading modules
    - Configuration management
    - Hardware compatibility checking
    """

    def __init__(self, config_path: Optional[Path] = None, local_only: bool = True):
        self.modules: Dict[str, Module] = {}
        self.module_classes: Dict[str, type] = {}
        self.config_path = config_path or Path("forge_modules.json")
        self.hardware_profile: Dict[str, Any] = {}
        self.local_only = local_only  # Default to local-only for privacy

        # Event callbacks
        self._on_load: List[Callable] = []
        self._on_unload: List[Callable] = []
        self._on_state_change: List[Callable] = []

        # Health monitoring
        self._health_monitor_thread: Optional[threading.Thread] = None
        self._health_monitor_running: bool = False
        self._health_monitor_interval: int = 60
        self._health_monitor_stop_event: threading.Event = threading.Event()
        self._module_error_counts: Dict[str, int] = {}

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
                self.hardware_profile['mps_available'] = hasattr(
                    torch.backends, 'mps') and torch.backends.mps.is_available()

                if torch.cuda.is_available():
                    self.hardware_profile['gpu_name'] = torch.cuda.get_device_name(0)
                    self.hardware_profile['vram_mb'] = torch.cuda.get_device_properties(
                        0).total_memory // (1024 * 1024)
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

    def can_load(self, module_id: str) -> Tuple[bool, str]:
        """
        Check if a module can be loaded.

        Returns:
            (can_load, reason)
        """
        if module_id not in self.module_classes:
            return False, f"Module '{module_id}' not registered"

        info = self.module_classes[module_id].get_info()

        # Check local-only mode
        if self.local_only and info.is_cloud_service:
            return False, "Module requires external cloud services. Disable local_only mode to use cloud modules."

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
        module_info = module_class.get_info()

        # Warn about cloud services
        if module_info.is_cloud_service:
            logger.warning(
                f"Warning: Module '{module_id}' connects to external cloud services and requires API keys + internet.")

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

    def load_sandboxed(
        self, 
        module_id: str, 
        sandbox_config: Optional[Any] = None,
        config: Dict[str, Any] = None
    ) -> bool:
        """
        Load a module in a sandboxed environment.
        
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
        
        # Create sandbox
        sandbox = ModuleSandbox(module_id, sandbox_config)
        
        try:
            # Create module instance
            module = module_class(self, config)
            module.state = ModuleState.LOADING
            
            # Load module in sandbox
            def load_func():
                return module.load()
            
            success = sandbox.run_in_sandbox(load_func)
            
            if success:
                module.state = ModuleState.LOADED
                module.get_info().load_time = datetime.now()
                self.modules[module_id] = module
                
                # Store sandbox reference with module for future use
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

    def is_loaded(self, module_id: str) -> bool:
        """Check if a module is currently loaded."""
        return module_id in self.modules and self.modules[module_id].state == ModuleState.LOADED

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
            import psutil
            import os
            
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
                import psutil
                import os
                process = psutil.Process(os.getpid())
                mem_percent = process.memory_percent()
                
                if mem_percent > 80:
                    warnings.append(f"High memory usage: {mem_percent:.1f}%")
                
            except (ImportError, Exception):
                pass  # psutil not available or error
            
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
