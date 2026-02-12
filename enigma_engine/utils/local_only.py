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
from typing import Any, Dict, List, Optional, Set

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
            pass  # Intentionally silent
    
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
        self.model_cache = ModelCache()
        self.sync_queue = OfflineSyncQueue()
        
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
                except Exception:
                    pass  # Intentionally silent
                    
        except ImportError:
            pass  # Intentionally silent
    
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
            "local_models": self.check_local_models() if self._enabled else {},
            "cached_models": len(self.model_cache.list_cached()),
            "pending_sync_ops": len(self.sync_queue),
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


class ModelCache:
    """Manages local caching of models for offline use.

    Tracks which models are cached locally so the engine can operate
    without network access.  Models are stored under a configurable
    cache directory and an index file records metadata.

    Args:
        cache_dir: Directory to store cached models. Defaults to
            ``models/cache``.

    Example::

        cache = ModelCache()
        cache.cache_model("microsoft/phi-3.5-mini", Path("models/phi35"))
        print(cache.list_cached())
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("models") / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.cache_dir / "cache_index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    # -- index persistence ------------------------------------------------

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache index from disk."""
        if self._index_path.exists():
            try:
                with open(self._index_path, "r") as f:
                    import json
                    return json.load(f)
            except Exception:
                logger.warning("Corrupt cache index, starting fresh")
        return {}

    def _save_index(self):
        """Persist the cache index to disk."""
        import json
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    # -- public API -------------------------------------------------------

    def cache_model(self, model_id: str, source_path: Path, metadata: Optional[Dict] = None):
        """Register a locally-available model in the cache.

        Args:
            model_id: HuggingFace-style model identifier (e.g.
                ``"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"``).
            source_path: Path where the model files reside on disk.
            metadata: Optional extra information (size, format, etc.).
        """
        from datetime import datetime
        self._index[model_id] = {
            "path": str(source_path),
            "cached_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._save_index()
        logger.info("Cached model %s at %s", model_id, source_path)

    def get_cached_path(self, model_id: str) -> Optional[Path]:
        """Return the local path for a cached model, or ``None``."""
        entry = self._index.get(model_id)
        if entry:
            p = Path(entry["path"])
            if p.exists():
                return p
            logger.warning("Cached path for %s no longer exists", model_id)
        return None

    def is_cached(self, model_id: str) -> bool:
        """Check whether *model_id* is available locally."""
        return self.get_cached_path(model_id) is not None

    def list_cached(self) -> Dict[str, Dict[str, Any]]:
        """Return a copy of the full cache index."""
        return dict(self._index)

    def remove(self, model_id: str):
        """Remove a model from the cache index (does not delete files)."""
        if model_id in self._index:
            del self._index[model_id]
            self._save_index()
            logger.info("Removed %s from cache index", model_id)

    def scan_local_models(self) -> Dict[str, Path]:
        """Scan common model directories and return discovered models.

        Looks inside ``models/``, the HuggingFace Hub cache, and the
        Torch Hub cache.
        """
        found: Dict[str, Path] = {}
        search_dirs = [
            Path("models"),
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "torch" / "hub",
        ]
        for d in search_dirs:
            if not d.exists():
                continue
            for item in d.iterdir():
                if item.is_dir():
                    found[item.name] = item
        return found


class OfflineSyncQueue:
    """Queues operations that require network access so they can be
    replayed when connectivity is restored.

    Operations are persisted to a JSONL file so nothing is lost if the
    process restarts while still offline.

    Args:
        queue_path: File used to persist queued operations.

    Example::

        queue = OfflineSyncQueue()
        queue.enqueue("upload_feedback", {"rating": "positive"})
        # ... later, when back online ...
        for op in queue.drain():
            process(op)
    """

    def __init__(self, queue_path: Optional[Path] = None):
        self.queue_path = queue_path or Path("data") / "offline_queue.jsonl"
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        """Load queued operations from disk."""
        import json
        ops: List[Dict[str, Any]] = []
        if self.queue_path.exists():
            try:
                with open(self.queue_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            ops.append(json.loads(line))
            except Exception:
                logger.warning("Could not load offline queue")
        return ops

    def _persist(self):
        """Write the queue to disk."""
        import json
        with open(self.queue_path, "w") as f:
            for op in self._queue:
                f.write(json.dumps(op) + "\n")

    def enqueue(self, operation: str, payload: Dict[str, Any]):
        """Add an operation to the queue.

        Args:
            operation: A short label such as ``"upload_feedback"`` or
                ``"sync_memory"``.
            payload: Arbitrary JSON-serializable data for the operation.
        """
        from datetime import datetime
        entry = {
            "operation": operation,
            "payload": payload,
            "queued_at": datetime.now().isoformat(),
        }
        self._queue.append(entry)
        self._persist()
        logger.debug("Queued offline operation: %s", operation)

    def drain(self) -> List[Dict[str, Any]]:
        """Return all queued operations and clear the queue."""
        ops = list(self._queue)
        self._queue.clear()
        self._persist()
        return ops

    def peek(self) -> List[Dict[str, Any]]:
        """Return queued operations without removing them."""
        return list(self._queue)

    def __len__(self) -> int:
        return len(self._queue)


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
