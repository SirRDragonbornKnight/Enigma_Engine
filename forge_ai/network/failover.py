"""
Failover Manager - Handle server failures gracefully

Monitors server health, manages failover, and ensures high availability.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import requests

logger = logging.getLogger(__name__)


class ServerHealth(Enum):
    """Server health status."""
    HEALTHY = "healthy"         # Server responding normally
    DEGRADED = "degraded"       # Slow or partial responses
    UNHEALTHY = "unhealthy"     # Not responding
    UNKNOWN = "unknown"         # Not yet checked


@dataclass
class ServerStatus:
    """Status information for a server."""
    address: str
    port: int
    health: ServerHealth = ServerHealth.UNKNOWN
    last_check: float = 0.0
    last_success: float = 0.0
    consecutive_failures: int = 0
    latency_ms: float = 0.0
    error_message: Optional[str] = None
    
    @property
    def key(self) -> str:
        return f"{self.address}:{self.port}"


class FailoverManager:
    """
    Manages server health monitoring and failover.
    
    Features:
    - Periodic health checks
    - Automatic failover when primary fails
    - Automatic recovery when servers come back
    - Circuit breaker pattern
    - Health check callbacks
    """
    
    def __init__(
        self,
        check_interval_s: float = 10.0,
        unhealthy_threshold: int = 3,
        recovery_threshold: int = 2,
        timeout_s: float = 5.0
    ):
        """
        Initialize failover manager.
        
        Args:
            check_interval_s: Time between health checks
            unhealthy_threshold: Failures before marking unhealthy
            recovery_threshold: Successes before marking healthy again
            timeout_s: Timeout for health check requests
        """
        self.check_interval_s = check_interval_s
        self.unhealthy_threshold = unhealthy_threshold
        self.recovery_threshold = recovery_threshold
        self.timeout_s = timeout_s
        
        self._servers: Dict[str, ServerStatus] = {}
        self._primary: Optional[str] = None
        self._check_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Callbacks
        self._on_unhealthy: List[Callable[[ServerStatus], None]] = []
        self._on_recovery: List[Callable[[ServerStatus], None]] = []
        self._on_failover: List[Callable[[str, str], None]] = []  # (old, new)
        
        # Recovery tracking
        self._recovery_count: Dict[str, int] = {}
    
    def add_server(self, address: str, port: int, is_primary: bool = False):
        """Add a server to monitor."""
        status = ServerStatus(address=address, port=port)
        with self._lock:
            self._servers[status.key] = status
            if is_primary or self._primary is None:
                self._primary = status.key
        logger.info(f"Added server to failover pool: {status.key}")
    
    def remove_server(self, address: str, port: int):
        """Remove a server from monitoring."""
        key = f"{address}:{port}"
        with self._lock:
            if key in self._servers:
                del self._servers[key]
                if self._primary == key:
                    # Select new primary
                    healthy = self.get_healthy_servers()
                    self._primary = healthy[0].key if healthy else None
        logger.info(f"Removed server from failover pool: {key}")
    
    def set_primary(self, address: str, port: int):
        """Set the primary server."""
        key = f"{address}:{port}"
        with self._lock:
            if key in self._servers:
                self._primary = key
                logger.info(f"Primary server set to: {key}")
    
    def get_primary(self) -> Optional[ServerStatus]:
        """Get the current primary server."""
        with self._lock:
            if self._primary and self._primary in self._servers:
                return self._servers[self._primary]
        return None
    
    def get_healthy_servers(self) -> List[ServerStatus]:
        """Get list of healthy servers."""
        with self._lock:
            return [
                s for s in self._servers.values()
                if s.health in (ServerHealth.HEALTHY, ServerHealth.DEGRADED)
            ]
    
    def on_unhealthy(self, callback: Callable[[ServerStatus], None]):
        """Register callback for when server becomes unhealthy."""
        self._on_unhealthy.append(callback)
    
    def on_recovery(self, callback: Callable[[ServerStatus], None]):
        """Register callback for when server recovers."""
        self._on_recovery.append(callback)
    
    def on_failover(self, callback: Callable[[str, str], None]):
        """Register callback for failover events (old_primary, new_primary)."""
        self._on_failover.append(callback)
    
    def start(self):
        """Start health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._check_thread = threading.Thread(
            target=self._check_loop,
            daemon=True,
            name="FailoverHealthCheck"
        )
        self._check_thread.start()
        logger.info("Failover manager started")
    
    def stop(self):
        """Stop health monitoring."""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=self.check_interval_s + 1)
        logger.info("Failover manager stopped")
    
    def _check_loop(self):
        """Main health check loop."""
        while self._running:
            with self._lock:
                servers = list(self._servers.values())
            
            for server in servers:
                self._check_server(server)
            
            time.sleep(self.check_interval_s)
    
    def _check_server(self, server: ServerStatus):
        """Check health of a single server."""
        url = f"http://{server.address}:{server.port}/health"
        start_time = time.time()
        
        try:
            response = requests.get(url, timeout=self.timeout_s)
            latency = (time.time() - start_time) * 1000
            
            server.last_check = time.time()
            server.latency_ms = latency
            
            if response.status_code == 200:
                self._handle_success(server, latency)
            else:
                self._handle_failure(server, f"HTTP {response.status_code}")
                
        except requests.Timeout:
            self._handle_failure(server, "Timeout")
        except requests.ConnectionError:
            self._handle_failure(server, "Connection refused")
        except Exception as e:
            self._handle_failure(server, str(e))
    
    def _handle_success(self, server: ServerStatus, latency_ms: float):
        """Handle successful health check."""
        server.last_success = time.time()
        server.consecutive_failures = 0
        server.error_message = None
        
        # Track recovery
        key = server.key
        if server.health == ServerHealth.UNHEALTHY:
            self._recovery_count[key] = self._recovery_count.get(key, 0) + 1
            
            if self._recovery_count[key] >= self.recovery_threshold:
                old_health = server.health
                server.health = ServerHealth.HEALTHY
                self._recovery_count[key] = 0
                
                logger.info(f"Server recovered: {key}")
                for callback in self._on_recovery:
                    try:
                        callback(server)
                    except Exception as e:
                        logger.error(f"Recovery callback error: {e}")
        else:
            # Determine health based on latency
            if latency_ms < 500:
                server.health = ServerHealth.HEALTHY
            elif latency_ms < 2000:
                server.health = ServerHealth.DEGRADED
            else:
                server.health = ServerHealth.DEGRADED
    
    def _handle_failure(self, server: ServerStatus, error: str):
        """Handle failed health check."""
        server.consecutive_failures += 1
        server.error_message = error
        self._recovery_count[server.key] = 0
        
        if server.consecutive_failures >= self.unhealthy_threshold:
            if server.health != ServerHealth.UNHEALTHY:
                server.health = ServerHealth.UNHEALTHY
                logger.warning(f"Server unhealthy: {server.key} ({error})")
                
                for callback in self._on_unhealthy:
                    try:
                        callback(server)
                    except Exception as e:
                        logger.error(f"Unhealthy callback error: {e}")
                
                # Trigger failover if this was primary
                if server.key == self._primary:
                    self._trigger_failover()
    
    def _trigger_failover(self):
        """Trigger failover to another server."""
        old_primary = self._primary
        
        # Find best healthy server
        healthy = self.get_healthy_servers()
        if not healthy:
            logger.error("No healthy servers for failover!")
            return
        
        # Select lowest latency healthy server
        healthy.sort(key=lambda s: s.latency_ms)
        new_primary = healthy[0].key
        
        with self._lock:
            self._primary = new_primary
        
        logger.warning(f"Failover: {old_primary} -> {new_primary}")
        
        for callback in self._on_failover:
            try:
                callback(old_primary, new_primary)
            except Exception as e:
                logger.error(f"Failover callback error: {e}")
    
    def get_status(self) -> Dict:
        """Get current failover status."""
        with self._lock:
            return {
                "primary": self._primary,
                "servers": [
                    {
                        "key": s.key,
                        "health": s.health.value,
                        "latency_ms": f"{s.latency_ms:.1f}",
                        "failures": s.consecutive_failures,
                        "error": s.error_message,
                    }
                    for s in self._servers.values()
                ],
                "healthy_count": len(self.get_healthy_servers()),
            }
