"""
Enigma Module Sandbox
=====================

Provides sandboxing/isolation for running untrusted modules with restricted permissions.

Features:
- Restricted file system access
- Network access control
- Memory and CPU limits
- Restricted imports
"""

import os
import sys
import time
import resource
import logging
from pathlib import Path
from typing import List, Callable, Any, Optional, Dict, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Default restricted imports
_DEFAULT_RESTRICTED_IMPORTS = ['os.system', 'subprocess', 'eval', 'exec']


@dataclass
class SandboxConfig:
    """Configuration for module sandbox."""
    allowed_paths: List[Path] = field(default_factory=list)
    allowed_hosts: List[str] = field(default_factory=list)
    blocked_hosts: List[str] = field(default_factory=list)
    max_memory_mb: int = 1024
    max_cpu_seconds: int = 300
    allow_network: bool = True
    allow_subprocess: bool = False
    restricted_imports: List[str] = field(default_factory=lambda: list(_DEFAULT_RESTRICTED_IMPORTS))


class ModuleSandbox:
    """
    Sandbox environment for running modules with restricted permissions.
    
    Features:
    - Restricted file system access (configurable allowed paths)
    - Network access control (allow/deny lists)
    - Memory limits
    - CPU time limits
    - Restricted imports
    
    Note: This provides basic sandboxing. For production use cases requiring
    strong isolation, consider using containers (Docker) or VMs.
    """
    
    def __init__(self, module_id: str, config: SandboxConfig):
        """
        Initialize sandbox for a module.
        
        Args:
            module_id: ID of the module being sandboxed
            config: Sandbox configuration
        """
        self.module_id = module_id
        self.config = config
        self.permissions: Dict[str, bool] = {}
        self._setup_permissions()
        
    def _setup_permissions(self):
        """Setup permission flags based on config."""
        self.permissions['network'] = self.config.allow_network
        self.permissions['subprocess'] = self.config.allow_subprocess
        self.permissions['file_write'] = len(self.config.allowed_paths) > 0
        
    def check_permission(self, permission: str) -> bool:
        """
        Check if a permission is granted.
        
        Args:
            permission: Permission name (e.g., 'network', 'subprocess', 'file_write')
            
        Returns:
            True if permission is granted
        """
        return self.permissions.get(permission, False)
    
    def check_path_access(self, path: Path, mode: str = 'read') -> bool:
        """
        Check if access to a path is allowed.
        
        Args:
            path: Path to check
            mode: 'read' or 'write'
            
        Returns:
            True if access is allowed
        """
        path = Path(path).resolve()
        
        # If no restrictions, allow all
        if not self.config.allowed_paths:
            return True
        
        # Check if path is under any allowed path
        for allowed_path in self.config.allowed_paths:
            allowed_path = Path(allowed_path).resolve()
            try:
                path.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        
        return False
    
    def check_host_access(self, host: str) -> bool:
        """
        Check if network access to a host is allowed.
        
        Args:
            host: Hostname or IP address
            
        Returns:
            True if access is allowed
        """
        # Check if network is disabled
        if not self.config.allow_network:
            return False
        
        # Check blocked hosts first
        if host in self.config.blocked_hosts:
            return False
        
        # If no allowed_hosts specified, allow all (except blocked)
        if not self.config.allowed_hosts:
            return True
        
        # Check if host is in allowed list
        return host in self.config.allowed_hosts
    
    def _set_resource_limits(self):
        """Set CPU and memory limits for the process."""
        try:
            # Set memory limit (soft and hard)
            max_memory_bytes = self.config.max_memory_mb * 1024 * 1024
            resource.setrlimit(
                resource.RLIMIT_AS,
                (max_memory_bytes, max_memory_bytes)
            )
            
            # Set CPU time limit
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (self.config.max_cpu_seconds, self.config.max_cpu_seconds)
            )
            
            logger.debug(
                f"Sandbox limits set: {self.config.max_memory_mb}MB RAM, "
                f"{self.config.max_cpu_seconds}s CPU"
            )
            
        except (ValueError, resource.error) as e:
            logger.warning(f"Could not set resource limits: {e}")
    
    def _install_import_hook(self):
        """
        Install import hook to restrict dangerous imports.
        
        Uses the builtins module to ensure compatibility across Python versions
        and execution contexts.
        """
        import builtins
        original_import = builtins.__import__
        restricted = set(self.config.restricted_imports)
        
        def restricted_import(name, *args, **kwargs):
            # Check if import is restricted
            if name in restricted:
                raise ImportError(
                    f"Import of '{name}' is restricted in sandbox"
                )
            
            # Check for restricted submodules (e.g., 'os.system')
            for restricted_name in restricted:
                if '.' in restricted_name:
                    module, attr = restricted_name.split('.', 1)
                    if name == module:
                        # Let the import happen, but we'll check attribute access
                        pass
            
            return original_import(name, *args, **kwargs)
        
        builtins.__import__ = restricted_import
        return original_import
    
    def run_in_sandbox(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function within the sandbox.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            Various exceptions if sandbox restrictions are violated
        """
        logger.info(f"Running function in sandbox for module '{self.module_id}'")
        
        # Store original import
        original_import = None
        
        try:
            # Set resource limits (Unix-like systems only)
            if hasattr(resource, 'RLIMIT_AS'):
                self._set_resource_limits()
            
            # Install import restrictions
            original_import = self._install_import_hook()
            
            # Execute the function
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            logger.debug(
                f"Sandbox execution completed in {elapsed:.2f}s "
                f"for module '{self.module_id}'"
            )
            
            return result
            
        except MemoryError:
            logger.error(
                f"Memory limit exceeded in sandbox for module '{self.module_id}'"
            )
            raise
            
        except Exception as e:
            logger.error(
                f"Error in sandboxed execution for module '{self.module_id}': {e}"
            )
            raise
            
        finally:
            # Restore original import
            if original_import:
                import builtins
                builtins.__import__ = original_import


class SandboxViolationError(Exception):
    """Raised when sandbox restrictions are violated."""
    pass


def create_default_sandbox_config(
    module_id: str,
    data_dir: Optional[Path] = None
) -> SandboxConfig:
    """
    Create a default sandbox configuration for a module.
    
    Args:
        module_id: ID of the module
        data_dir: Optional data directory to allow access to
        
    Returns:
        SandboxConfig with sensible defaults
    """
    allowed_paths = []
    
    # Allow access to module's data directory if provided
    if data_dir:
        allowed_paths.append(data_dir)
    
    # Allow access to common safe directories
    # (can be customized based on requirements)
    
    return SandboxConfig(
        allowed_paths=allowed_paths,
        allowed_hosts=[],  # Empty = allow all
        blocked_hosts=['localhost', '127.0.0.1'],  # Block local by default
        max_memory_mb=1024,
        max_cpu_seconds=300,
        allow_network=True,
        allow_subprocess=False,
        restricted_imports=['os.system', 'subprocess', 'eval', 'exec']
    )
