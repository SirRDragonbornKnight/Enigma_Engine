"""
Load Balancer - Distribute tasks across multiple servers

Implements various strategies for distributing inference tasks:
- Round-robin
- Least connections
- Weighted (based on server capabilities)
- Latency-aware
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class BalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"       # Cycle through servers
    LEAST_CONN = "least_conn"         # Fewest active connections
    WEIGHTED = "weighted"             # Based on server capabilities
    LATENCY = "latency"               # Lowest latency first
    RANDOM = "random"                 # Random selection
    ADAPTIVE = "adaptive"             # Combines multiple factors


@dataclass
class ServerInfo:
    """Information about a server in the pool."""
    address: str
    port: int
    weight: float = 1.0               # Relative capacity (higher = more capable)
    active_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    last_success: float = field(default_factory=time.time)
    last_failure: Optional[float] = None
    is_healthy: bool = True
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests
    
    @property
    def key(self) -> str:
        """Unique server key."""
        return f"{self.address}:{self.port}"


class LoadBalancer:
    """
    Distributes tasks across multiple servers.
    
    Features:
    - Multiple balancing strategies
    - Health tracking
    - Automatic failover
    - Connection counting
    """
    
    def __init__(self, strategy: BalancingStrategy = BalancingStrategy.ADAPTIVE):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy to use
        """
        self.strategy = strategy
        self._servers: Dict[str, ServerInfo] = {}
        self._lock = threading.Lock()
        self._round_robin_index = 0
    
    def add_server(
        self,
        address: str,
        port: int,
        weight: float = 1.0
    ) -> ServerInfo:
        """Add a server to the pool."""
        server = ServerInfo(address=address, port=port, weight=weight)
        with self._lock:
            self._servers[server.key] = server
        logger.info(f"Added server to pool: {server.key}")
        return server
    
    def remove_server(self, address: str, port: int):
        """Remove a server from the pool."""
        key = f"{address}:{port}"
        with self._lock:
            if key in self._servers:
                del self._servers[key]
                logger.info(f"Removed server from pool: {key}")
    
    def get_server(self, exclude: Optional[List[str]] = None) -> Optional[ServerInfo]:
        """
        Get next server based on strategy.
        
        Args:
            exclude: List of server keys to exclude
            
        Returns:
            Selected server or None if none available
        """
        exclude = exclude or []
        
        with self._lock:
            healthy = [
                s for s in self._servers.values()
                if s.is_healthy and s.key not in exclude
            ]
            
            if not healthy:
                return None
            
            if self.strategy == BalancingStrategy.ROUND_ROBIN:
                return self._select_round_robin(healthy)
            elif self.strategy == BalancingStrategy.LEAST_CONN:
                return self._select_least_conn(healthy)
            elif self.strategy == BalancingStrategy.WEIGHTED:
                return self._select_weighted(healthy)
            elif self.strategy == BalancingStrategy.LATENCY:
                return self._select_latency(healthy)
            elif self.strategy == BalancingStrategy.RANDOM:
                return self._select_random(healthy)
            else:  # ADAPTIVE
                return self._select_adaptive(healthy)
    
    def _select_round_robin(self, servers: List[ServerInfo]) -> ServerInfo:
        """Round-robin selection."""
        self._round_robin_index = (self._round_robin_index + 1) % len(servers)
        return servers[self._round_robin_index]
    
    def _select_least_conn(self, servers: List[ServerInfo]) -> ServerInfo:
        """Select server with fewest active connections."""
        return min(servers, key=lambda s: s.active_connections)
    
    def _select_weighted(self, servers: List[ServerInfo]) -> ServerInfo:
        """Weighted random selection."""
        total_weight = sum(s.weight for s in servers)
        r = random.uniform(0, total_weight)
        cumulative = 0
        for server in servers:
            cumulative += server.weight
            if r <= cumulative:
                return server
        return servers[-1]
    
    def _select_latency(self, servers: List[ServerInfo]) -> ServerInfo:
        """Select server with lowest latency."""
        return min(servers, key=lambda s: s.avg_latency_ms)
    
    def _select_random(self, servers: List[ServerInfo]) -> ServerInfo:
        """Random selection."""
        return random.choice(servers)
    
    def _select_adaptive(self, servers: List[ServerInfo]) -> ServerInfo:
        """
        Adaptive selection combining multiple factors.
        
        Score = weight * (1 / (1 + connections)) * (1 / (1 + latency/100)) * (1 - error_rate)
        """
        def score(s: ServerInfo) -> float:
            connection_factor = 1 / (1 + s.active_connections)
            latency_factor = 1 / (1 + s.avg_latency_ms / 100)
            error_factor = 1 - s.error_rate
            return s.weight * connection_factor * latency_factor * error_factor
        
        return max(servers, key=score)
    
    def mark_request_start(self, server: ServerInfo):
        """Mark that a request is starting."""
        with self._lock:
            server.active_connections += 1
    
    def mark_request_end(
        self,
        server: ServerInfo,
        success: bool,
        latency_ms: float
    ):
        """Mark that a request has completed."""
        with self._lock:
            server.active_connections = max(0, server.active_connections - 1)
            server.total_requests += 1
            
            if success:
                server.last_success = time.time()
                # Update average latency (exponential moving average)
                alpha = 0.1
                server.avg_latency_ms = alpha * latency_ms + (1 - alpha) * server.avg_latency_ms
            else:
                server.total_errors += 1
                server.last_failure = time.time()
                
                # Mark unhealthy if too many consecutive errors
                recent_error_rate = self._get_recent_error_rate(server)
                if recent_error_rate > 0.5:
                    server.is_healthy = False
                    logger.warning(f"Server marked unhealthy: {server.key}")
    
    def _get_recent_error_rate(self, server: ServerInfo, window: int = 10) -> float:
        """Get error rate for recent requests."""
        # Simplified: just use overall error rate
        # A real implementation would track a sliding window
        return server.error_rate
    
    def mark_healthy(self, address: str, port: int):
        """Mark a server as healthy."""
        key = f"{address}:{port}"
        with self._lock:
            if key in self._servers:
                self._servers[key].is_healthy = True
                logger.info(f"Server marked healthy: {key}")
    
    def mark_unhealthy(self, address: str, port: int):
        """Mark a server as unhealthy."""
        key = f"{address}:{port}"
        with self._lock:
            if key in self._servers:
                self._servers[key].is_healthy = False
                logger.warning(f"Server marked unhealthy: {key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            return {
                "strategy": self.strategy.value,
                "total_servers": len(self._servers),
                "healthy_servers": sum(1 for s in self._servers.values() if s.is_healthy),
                "servers": [
                    {
                        "key": s.key,
                        "weight": s.weight,
                        "active": s.active_connections,
                        "total_requests": s.total_requests,
                        "error_rate": f"{s.error_rate:.2%}",
                        "avg_latency_ms": f"{s.avg_latency_ms:.1f}",
                        "healthy": s.is_healthy,
                    }
                    for s in self._servers.values()
                ]
            }
