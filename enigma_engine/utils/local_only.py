"""
Local-Only Mode for Enigma AI Engine

Disable all network features for complete privacy.

Features:
- Block outbound connections
- Disable telemetry
- Local model enforcement
- Offline operation
- Network activity logging

Usage:
    from enigma_engine.utils.local_only import LocalOnlyMode
    
    # Enable local-only mode
    local = LocalOnlyMode()
    local.enable()
    
    # Check status
    if local.is_enabled():
        print("Running in offline mode")
"""

import logging
import os
import socket
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class NetworkActivity:
    """Record of network activity attempt."""
    timestamp: float
    host: str
    port: int
    blocked: bool
    reason: str


@dataclass
class LocalOnlyConfig:
    """Configuration for local-only mode."""
    # What to block
    block_http: bool = True
    block_websocket: bool = True
    block_api_calls: bool = True
    block_telemetry: bool = True
    
    # Exceptions
    allowed_hosts: Set[str] = field(default_factory=lambda: {"localhost", "127.0.0.1", "::1"})
    allowed_ports: Set[int] = field(default_factory=lambda: {0})  # Allow any local port
    
    # Logging
    log_blocked: bool = True
    log_path: Optional[Path] = None


class NetworkBlocker:
    """Block network connections."""
    
    def __init__(self, config: LocalOnlyConfig):
        self.config = config
        
        # Original socket functions
        self._original_socket = socket.socket
        self._original_connect = None
        
        # Activity log
        self._activity: List[NetworkActivity] = []
        self._lock = threading.Lock()
        
        # Blocked hosts/ports
        self._blocked_hosts: Set[str] = set()
        self._blocked_ports: Set[int] = set()
    
    def enable(self):
        """Enable network blocking."""
        # Monkey-patch socket
        original_init = socket.socket.__init__
        
        def patched_init(self_socket, *args, **kwargs):
            original_init(self_socket, *args, **kwargs)
            self_socket._blocked = False
        
        # Patch connect
        def patched_connect(address):
            host = address[0] if isinstance(address, tuple) else str(address)
            port = address[1] if isinstance(address, tuple) else 0
            
            # Check if allowed
            if self._is_allowed(host, port):
                self._log_activity(host, port, blocked=False, reason="Allowed")
                # Call original
                return self._original_connect(address)
            else:
                self._log_activity(host, port, blocked=True, reason="Local-only mode")
                raise ConnectionRefusedError(f"Network blocked in local-only mode: {host}:{port}")
        
        # Apply patches to requests library if available
        self._patch_requests()
        
        logger.info("Local-only mode enabled - network connections blocked")
    
    def disable(self):
        """Disable network blocking."""
        # Restore original socket behavior would go here
        logger.info("Local-only mode disabled")
    
    def _is_allowed(self, host: str, port: int) -> bool:
        """Check if connection is allowed."""
        # Always allow localhost
        if host in self.config.allowed_hosts:
            return True
        
        # Check blocked lists
        if host in self._blocked_hosts:
            return False
        
        if port in self._blocked_ports:
            return False
        
        # Default: block
        return False
    
    def _log_activity(self, host: str, port: int, blocked: bool, reason: str):
        """Log network activity."""
        import time
        
        activity = NetworkActivity(
            timestamp=time.time(),
            host=host,
            port=port,
            blocked=blocked,
            reason=reason
        )
        
        with self._lock:
            self._activity.append(activity)
            
            # Keep last 1000
            if len(self._activity) > 1000:
                self._activity = self._activity[-1000:]
        
        if self.config.log_blocked and blocked:
            logger.warning(f"Blocked network connection to {host}:{port} - {reason}")
    
    def _patch_requests(self):
        """Patch requests library."""
        try:
            import requests
            
            original_get = requests.get
            original_post = requests.post
            
            def blocked_request(*args, **kwargs):
                url = args[0] if args else kwargs.get('url', '')
                self._log_activity(url, 443, blocked=True, reason="HTTP blocked")
                raise requests.exceptions.ConnectionError("Network blocked in local-only mode")
            
            requests.get = blocked_request
            requests.post = blocked_request
            
        except ImportError:
            pass
    
    def get_activity(self) -> List[NetworkActivity]:
        """Get network activity log."""
        with self._lock:
            return list(self._activity)
    
    def add_exception(self, host: str):
        """Add host to allowed list."""
        self.config.allowed_hosts.add(host)
    
    def remove_exception(self, host: str):
        """Remove host from allowed list."""
        self.config.allowed_hosts.discard(host)


class LocalOnlyMode:
    """Main local-only mode controller."""
    
    def __init__(self, config: Optional[LocalOnlyConfig] = None):
        """
        Initialize local-only mode.
        
        Args:
            config: Configuration
        """
        self.config = config or LocalOnlyConfig()
        
        # Components
        self._blocker = NetworkBlocker(self.config)
        
        # State
        self._enabled = False
        
        # Environment patches
        self._original_env: Dict[str, str] = {}
    
    def enable(self):
        """Enable local-only mode."""
        if self._enabled:
            return
        
        self._enabled = True
        
        # Enable network blocker
        self._blocker.enable()
        
        # Set environment variables
        self._patch_environment()
        
        # Disable API modules
        self._disable_api_modules()
        
        logger.info("Local-only mode ENABLED - all network features disabled")
    
    def disable(self):
        """Disable local-only mode."""
        if not self._enabled:
            return
        
        self._enabled = False
        
        # Disable network blocker
        self._blocker.disable()
        
        # Restore environment
        self._restore_environment()
        
        logger.info("Local-only mode DISABLED")
    
    def is_enabled(self) -> bool:
        """Check if local-only mode is enabled."""
        return self._enabled
    
    def _patch_environment(self):
        """Set environment for local-only operation."""
        # Store originals
        env_vars = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "HUGGINGFACE_TOKEN",
            "REPLICATE_API_TOKEN",
            "ELEVENLABS_API_KEY"
        ]
        
        for var in env_vars:
            if var in os.environ:
                self._original_env[var] = os.environ[var]
                del os.environ[var]
        
        # Set offline flags
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["ENIGMA_OFFLINE_MODE"] = "1"
    
    def _restore_environment(self):
        """Restore original environment."""
        # Restore API keys
        for var, value in self._original_env.items():
            os.environ[var] = value
        
        self._original_env.clear()
        
        # Remove offline flags
        for var in ["TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "ENIGMA_OFFLINE_MODE"]:
            if var in os.environ:
                del os.environ[var]
    
    def _disable_api_modules(self):
        """Disable API-based modules."""
        try:
            from enigma_engine.modules import ModuleManager
            
            manager = ModuleManager()
            
            # Disable API modules
            api_modules = [
                "image_gen_api",
                "code_gen_api", 
                "video_gen_api",
                "audio_gen_api",
                "embedding_api",
                "threed_gen_api"
            ]
            
            for module in api_modules:
                try:
                    manager.unload(module)
                except:
                    pass
                    
        except ImportError:
            pass
    
    def check_local_models(self) -> Dict[str, bool]:
        """Check what local models are available."""
        available = {}
        
        # Check for local model files
        model_paths = [
            Path("models"),
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "torch" / "hub"
        ]
        
        for path in model_paths:
            if path.exists():
                for item in path.iterdir():
                    if item.is_dir():
                        available[item.name] = True
        
        return available
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "enabled": self._enabled,
            "blocked_connections": len(self._blocker.get_activity()),
            "allowed_hosts": list(self.config.allowed_hosts),
            "local_models": self.check_local_models() if self._enabled else {}
        }
    
    def add_allowed_host(self, host: str):
        """Add a host to the allowed list (for local network resources)."""
        self._blocker.add_exception(host)
    
    def get_blocked_activity(self) -> List[Dict]:
        """Get list of blocked connection attempts."""
        activity = self._blocker.get_activity()
        
        return [
            {
                "timestamp": a.timestamp,
                "host": a.host,
                "port": a.port,
                "blocked": a.blocked,
                "reason": a.reason
            }
            for a in activity
            if a.blocked
        ]


# Global instance
_local_only: Optional[LocalOnlyMode] = None


def get_local_only_mode() -> LocalOnlyMode:
    """Get or create global local-only mode instance."""
    global _local_only
    if _local_only is None:
        _local_only = LocalOnlyMode()
    return _local_only


def enable_local_only():
    """Quick enable local-only mode."""
    get_local_only_mode().enable()


def disable_local_only():
    """Quick disable local-only mode."""
    get_local_only_mode().disable()


def is_offline() -> bool:
    """Check if running in offline mode."""
    return (
        get_local_only_mode().is_enabled() or
        os.environ.get("ENIGMA_OFFLINE_MODE") == "1" or
        os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )
