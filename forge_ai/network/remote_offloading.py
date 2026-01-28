"""
Remote Offloading - Decide when and where to offload tasks

Automatically routes heavy inference tasks to capable remote servers
when local hardware is insufficient or busy.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OffloadDecision(Enum):
    """Decision on where to run inference."""
    LOCAL = "local"          # Run on this device
    REMOTE = "remote"        # Offload to remote server
    FALLBACK = "fallback"    # Remote failed, fallback to local
    SKIP = "skip"            # Skip entirely (no capacity anywhere)


@dataclass
class OffloadCriteria:
    """Criteria for deciding when to offload."""
    model_size_mb: float = 0         # Model size in MB
    requires_gpu: bool = False        # Task needs GPU
    estimated_time_s: float = 0       # Estimated execution time
    priority: int = 5                 # Task priority (1=highest)
    allow_remote: bool = True         # Can this task be offloaded?
    prefer_remote: bool = False       # Prefer remote even if local possible


@dataclass
class ServerCapability:
    """Capabilities of a remote server."""
    address: str
    port: int
    gpu_available: bool = False
    gpu_memory_mb: int = 0
    ram_mb: int = 0
    latency_ms: float = float('inf')
    current_load: float = 0.0         # 0.0 to 1.0
    models_loaded: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)


class RemoteOffloader:
    """
    Manages remote task offloading decisions.
    
    Decides when to run tasks locally vs offload to remote servers
    based on hardware capabilities, server load, and task requirements.
    """
    
    def __init__(self, prefer_local: bool = True):
        """
        Initialize remote offloader.
        
        Args:
            prefer_local: If True, prefer local execution when possible
        """
        self.prefer_local = prefer_local
        self._servers: Dict[str, ServerCapability] = {}
        self._local_capability: Optional[ServerCapability] = None
        self._detect_local_capability()
    
    def _detect_local_capability(self):
        """Detect local hardware capabilities."""
        import psutil
        
        ram_mb = psutil.virtual_memory().total // (1024 * 1024)
        
        gpu_available = False
        gpu_memory_mb = 0
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        except ImportError:
            pass
        
        self._local_capability = ServerCapability(
            address="localhost",
            port=0,
            gpu_available=gpu_available,
            gpu_memory_mb=gpu_memory_mb,
            ram_mb=ram_mb,
            latency_ms=0,
            current_load=psutil.cpu_percent() / 100.0,
        )
        
        logger.info(
            f"Local capability: RAM={ram_mb}MB, GPU={'Yes' if gpu_available else 'No'}"
            f"{f', VRAM={gpu_memory_mb}MB' if gpu_available else ''}"
        )
    
    def register_server(self, server: ServerCapability):
        """Register a remote server."""
        key = f"{server.address}:{server.port}"
        self._servers[key] = server
        logger.info(f"Registered server: {key}")
    
    def unregister_server(self, address: str, port: int):
        """Unregister a remote server."""
        key = f"{address}:{port}"
        if key in self._servers:
            del self._servers[key]
            logger.info(f"Unregistered server: {key}")
    
    def update_server(self, address: str, port: int, **updates):
        """Update server information."""
        key = f"{address}:{port}"
        if key in self._servers:
            for attr, value in updates.items():
                if hasattr(self._servers[key], attr):
                    setattr(self._servers[key], attr, value)
            self._servers[key].last_seen = time.time()
    
    def get_available_servers(self, max_age_s: float = 60) -> List[ServerCapability]:
        """Get list of available servers."""
        now = time.time()
        return [
            server for server in self._servers.values()
            if now - server.last_seen < max_age_s
        ]
    
    def decide(self, criteria: OffloadCriteria) -> tuple[OffloadDecision, Optional[ServerCapability]]:
        """
        Decide whether to offload a task.
        
        Args:
            criteria: Task requirements and preferences
            
        Returns:
            Tuple of (decision, selected_server)
        """
        # If offloading disabled, always local
        if not criteria.allow_remote:
            return OffloadDecision.LOCAL, None
        
        # Check if local can handle it
        local_capable = self._can_handle_locally(criteria)
        
        # Get available servers sorted by preference
        servers = self._rank_servers(criteria)
        
        # Decision logic
        if criteria.prefer_remote and servers:
            # User wants remote, and we have servers
            return OffloadDecision.REMOTE, servers[0]
        
        if local_capable and self.prefer_local:
            # Local can handle it and we prefer local
            return OffloadDecision.LOCAL, None
        
        if servers:
            # Offload to best available server
            return OffloadDecision.REMOTE, servers[0]
        
        if local_capable:
            # No remote available, but local can handle it
            return OffloadDecision.LOCAL, None
        
        # Nothing can handle this task
        logger.warning(f"No capacity for task: {criteria}")
        return OffloadDecision.SKIP, None
    
    def _can_handle_locally(self, criteria: OffloadCriteria) -> bool:
        """Check if local hardware can handle the task."""
        if self._local_capability is None:
            return False
        
        # Check GPU requirement
        if criteria.requires_gpu and not self._local_capability.gpu_available:
            return False
        
        # Check model size fits in memory
        if criteria.requires_gpu:
            if criteria.model_size_mb > self._local_capability.gpu_memory_mb:
                return False
        else:
            if criteria.model_size_mb > self._local_capability.ram_mb * 0.8:  # Leave 20% headroom
                return False
        
        return True
    
    def _rank_servers(self, criteria: OffloadCriteria) -> List[ServerCapability]:
        """Rank servers by suitability for the task."""
        available = self.get_available_servers()
        
        capable = []
        for server in available:
            # Check GPU requirement
            if criteria.requires_gpu and not server.gpu_available:
                continue
            
            # Check memory
            if criteria.requires_gpu:
                if criteria.model_size_mb > server.gpu_memory_mb:
                    continue
            else:
                if criteria.model_size_mb > server.ram_mb * 0.8:
                    continue
            
            capable.append(server)
        
        # Sort by: load (ascending), then latency (ascending)
        capable.sort(key=lambda s: (s.current_load, s.latency_ms))
        
        return capable


# Global instance
_offloader: Optional[RemoteOffloader] = None


def get_remote_offloader() -> RemoteOffloader:
    """Get the global remote offloader instance."""
    global _offloader
    if _offloader is None:
        _offloader = RemoteOffloader()
    return _offloader
