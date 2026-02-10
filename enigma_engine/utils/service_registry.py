"""
Service Registry for Enigma AI Engine.

Provides service discovery, registration, and health monitoring.
Supports local and distributed service registration.

Usage:
    from enigma_engine.utils.service_registry import ServiceRegistry, Service
    
    registry = ServiceRegistry()
    
    # Register a service
    registry.register(Service(
        name="inference",
        endpoint="localhost:8080",
        version="1.0.0",
        metadata={"model": "forge-small"}
    ))
    
    # Discover services
    services = registry.discover("inference")
    
    # Get healthy service
    service = registry.get_healthy("inference")
    
    # With decorator
    @service_provider("image_gen")
    class ImageGenerator:
        def generate(self, prompt):
            pass
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceStatus(Enum):
    """Service health status."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    STARTING = "starting"
    STOPPING = "stopping"


@dataclass
class ServiceHealth:
    """Health information for a service."""
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_check: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    latency_ms: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "last_check": self.last_check,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "latency_ms": self.latency_ms,
            "error_message": self.error_message,
        }


@dataclass
class Service:
    """
    Service registration information.
    
    Attributes:
        name: Service name (e.g., "inference", "image_gen")
        endpoint: Service endpoint URL or address
        version: Service version string
        instance_id: Unique instance identifier
        metadata: Additional service metadata
        tags: Tags for filtering/discovery
        weight: Load balancing weight (higher = more traffic)
        zone: Deployment zone/region
        health_check_url: URL for health checks
    """
    name: str
    endpoint: str
    version: str = "1.0.0"
    instance_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    weight: int = 1
    zone: str = "default"
    health_check_url: Optional[str] = None
    
    # Internal state
    health: ServiceHealth = field(default_factory=ServiceHealth)
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Generate instance_id if not provided."""
        if not self.instance_id:
            unique = f"{self.name}:{self.endpoint}:{self.version}:{time.time()}"
            self.instance_id = hashlib.md5(unique.encode()).hexdigest()[:12]
        
        # Convert tags list to set
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
    
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.health.status in (ServiceStatus.HEALTHY, ServiceStatus.DEGRADED)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "endpoint": self.endpoint,
            "version": self.version,
            "instance_id": self.instance_id,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "weight": self.weight,
            "zone": self.zone,
            "health": self.health.to_dict(),
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Service":
        """Create from dictionary."""
        health_data = data.pop("health", {})
        data["tags"] = set(data.get("tags", []))
        
        service = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        
        if health_data:
            service.health = ServiceHealth(
                status=ServiceStatus(health_data.get("status", "unknown")),
                last_check=health_data.get("last_check", 0.0),
                consecutive_failures=health_data.get("consecutive_failures", 0),
                consecutive_successes=health_data.get("consecutive_successes", 0),
                latency_ms=health_data.get("latency_ms", 0.0),
                error_message=health_data.get("error_message"),
            )
        
        return service


class LoadBalancer:
    """Load balancing strategies for service selection."""
    
    @staticmethod
    def round_robin(services: list[Service], state: dict[str, Any]) -> Optional[Service]:
        """Round-robin selection."""
        if not services:
            return None
        
        index = state.get("rr_index", 0)
        service = services[index % len(services)]
        state["rr_index"] = index + 1
        return service
    
    @staticmethod
    def weighted(services: list[Service], state: dict[str, Any]) -> Optional[Service]:
        """Weighted random selection."""
        if not services:
            return None
        
        import random
        total_weight = sum(s.weight for s in services)
        target = random.uniform(0, total_weight)
        
        current = 0
        for service in services:
            current += service.weight
            if current >= target:
                return service
        
        return services[-1]
    
    @staticmethod
    def least_latency(services: list[Service], state: dict[str, Any]) -> Optional[Service]:
        """Select service with lowest latency."""
        if not services:
            return None
        
        return min(services, key=lambda s: s.health.latency_ms or float('inf'))
    
    @staticmethod
    def random(services: list[Service], state: dict[str, Any]) -> Optional[Service]:
        """Random selection."""
        if not services:
            return None
        
        import random
        return random.choice(services)
    
    @staticmethod
    def zone_aware(services: list[Service], state: dict[str, Any], preferred_zone: str = "default") -> Optional[Service]:
        """Prefer services in the same zone."""
        if not services:
            return None
        
        # Try same zone first
        same_zone = [s for s in services if s.zone == preferred_zone]
        if same_zone:
            return LoadBalancer.weighted(same_zone, state)
        
        # Fallback to any zone
        return LoadBalancer.weighted(services, state)


class ServiceRegistry:
    """
    Service registry for discovery and health monitoring.
    
    Features:
        - Service registration/deregistration
        - Health checking
        - Load balancing
        - Service versioning
        - Tag-based filtering
    """
    
    def __init__(
        self,
        health_check_interval: float = 30.0,
        heartbeat_timeout: float = 90.0,
        auto_deregister: bool = True,
    ):
        """
        Initialize registry.
        
        Args:
            health_check_interval: Seconds between health checks
            heartbeat_timeout: Seconds before considering service dead
            auto_deregister: Remove services after heartbeat timeout
        """
        self.health_check_interval = health_check_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.auto_deregister = auto_deregister
        
        self._services: dict[str, dict[str, Service]] = {}  # name -> instance_id -> Service
        self._lock = threading.RLock()
        self._lb_state: dict[str, dict[str, Any]] = {}  # Load balancer state per service
        
        self._health_check_thread: Optional[threading.Thread] = None
        self._running = False
        
        self._listeners: list[Callable[[str, Service, str], None]] = []
    
    def register(self, service: Service) -> bool:
        """
        Register a service.
        
        Args:
            service: Service to register
        
        Returns:
            True if registered (or updated)
        """
        with self._lock:
            if service.name not in self._services:
                self._services[service.name] = {}
                self._lb_state[service.name] = {}
            
            is_new = service.instance_id not in self._services[service.name]
            service.health.status = ServiceStatus.STARTING
            self._services[service.name][service.instance_id] = service
            
            action = "registered" if is_new else "updated"
            logger.info(f"Service {action}: {service.name}/{service.instance_id} at {service.endpoint}")
            
            self._notify_listeners("register" if is_new else "update", service)
            return True
    
    def deregister(self, name: str, instance_id: str) -> bool:
        """
        Deregister a service instance.
        
        Args:
            name: Service name
            instance_id: Instance identifier
        
        Returns:
            True if deregistered
        """
        with self._lock:
            if name in self._services and instance_id in self._services[name]:
                service = self._services[name].pop(instance_id)
                
                if not self._services[name]:
                    del self._services[name]
                    del self._lb_state[name]
                
                logger.info(f"Service deregistered: {name}/{instance_id}")
                self._notify_listeners("deregister", service)
                return True
            return False
    
    def heartbeat(self, name: str, instance_id: str) -> bool:
        """
        Update service heartbeat.
        
        Args:
            name: Service name
            instance_id: Instance identifier
        
        Returns:
            True if service found
        """
        with self._lock:
            if name in self._services and instance_id in self._services[name]:
                service = self._services[name][instance_id]
                service.last_heartbeat = time.time()
                
                if service.health.status == ServiceStatus.STARTING:
                    service.health.status = ServiceStatus.HEALTHY
                
                return True
            return False
    
    def discover(
        self,
        name: str,
        version: Optional[str] = None,
        tags: Optional[set[str]] = None,
        healthy_only: bool = True,
    ) -> list[Service]:
        """
        Discover services by criteria.
        
        Args:
            name: Service name
            version: Required version (None for any)
            tags: Required tags (None for any)
            healthy_only: Only return healthy services
        
        Returns:
            List of matching services
        """
        with self._lock:
            if name not in self._services:
                return []
            
            services = list(self._services[name].values())
            
            # Filter by version
            if version:
                services = [s for s in services if s.version == version]
            
            # Filter by tags
            if tags:
                services = [s for s in services if tags.issubset(s.tags)]
            
            # Filter by health
            if healthy_only:
                services = [s for s in services if s.is_healthy()]
            
            return services
    
    def get_healthy(
        self,
        name: str,
        strategy: str = "weighted",
        **kwargs
    ) -> Optional[Service]:
        """
        Get a healthy service using load balancing.
        
        Args:
            name: Service name
            strategy: Load balancing strategy (round_robin, weighted, least_latency, random)
            **kwargs: Additional strategy arguments
        
        Returns:
            Selected service or None
        """
        services = self.discover(name, healthy_only=True, **{k: v for k, v in kwargs.items() if k in ('version', 'tags')})
        
        if not services:
            return None
        
        with self._lock:
            lb_state = self._lb_state.get(name, {})
        
        strategies = {
            "round_robin": LoadBalancer.round_robin,
            "weighted": LoadBalancer.weighted,
            "least_latency": LoadBalancer.least_latency,
            "random": LoadBalancer.random,
        }
        
        balancer = strategies.get(strategy, LoadBalancer.weighted)
        return balancer(services, lb_state)
    
    def get_all_services(self) -> dict[str, list[Service]]:
        """Get all registered services."""
        with self._lock:
            return {
                name: list(instances.values())
                for name, instances in self._services.items()
            }
    
    def get_service(self, name: str, instance_id: str) -> Optional[Service]:
        """Get specific service instance."""
        with self._lock:
            if name in self._services and instance_id in self._services[name]:
                return self._services[name][instance_id]
            return None
    
    def update_health(
        self,
        name: str,
        instance_id: str,
        status: ServiceStatus,
        latency_ms: float = 0.0,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update service health status.
        
        Args:
            name: Service name
            instance_id: Instance identifier
            status: Health status
            latency_ms: Response latency
            error_message: Error message if unhealthy
        
        Returns:
            True if service found
        """
        with self._lock:
            service = self.get_service(name, instance_id)
            if not service:
                return False
            
            old_status = service.health.status
            service.health.status = status
            service.health.last_check = time.time()
            service.health.latency_ms = latency_ms
            service.health.error_message = error_message
            
            if status == ServiceStatus.HEALTHY:
                service.health.consecutive_successes += 1
                service.health.consecutive_failures = 0
            elif status in (ServiceStatus.UNHEALTHY, ServiceStatus.DEGRADED):
                service.health.consecutive_failures += 1
                service.health.consecutive_successes = 0
            
            if old_status != status:
                logger.info(f"Service health changed: {name}/{instance_id}: {old_status.value} -> {status.value}")
                self._notify_listeners("health_change", service)
            
            return True
    
    def add_listener(self, callback: Callable[[str, Service, str], None]) -> None:
        """
        Add event listener.
        
        Callback receives: (event_type, service, extra_info)
        Event types: register, deregister, update, health_change
        """
        self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable) -> bool:
        """Remove event listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)
            return True
        return False
    
    def _notify_listeners(self, event: str, service: Service, extra: str = "") -> None:
        """Notify all listeners of an event."""
        for callback in self._listeners:
            try:
                callback(event, service, extra)
            except Exception as e:
                logger.warning(f"Listener error: {e}")
    
    def start_health_checks(self) -> None:
        """Start background health check thread."""
        if self._running:
            return
        
        self._running = True
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="service-health-check"
        )
        self._health_check_thread.start()
        logger.info("Started service health check thread")
    
    def stop_health_checks(self) -> None:
        """Stop background health check thread."""
        self._running = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)
            self._health_check_thread = None
        logger.info("Stopped service health check thread")
    
    def _health_check_loop(self) -> None:
        """Health check loop."""
        while self._running:
            try:
                self._run_health_checks()
                self._check_heartbeat_timeouts()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            time.sleep(self.health_check_interval)
    
    def _run_health_checks(self) -> None:
        """Run health checks on all services."""
        with self._lock:
            all_services = [
                (name, service)
                for name, instances in self._services.items()
                for service in instances.values()
                if service.health_check_url
            ]
        
        for name, service in all_services:
            self._check_service_health(name, service)
    
    def _check_service_health(self, name: str, service: Service) -> None:
        """Check health of a single service."""
        if not service.health_check_url:
            return
        
        try:
            import urllib.error
            import urllib.request
            
            start = time.time()
            request = urllib.request.Request(
                service.health_check_url,
                method='GET',
                headers={'User-Agent': 'Enigma AI Engine-HealthCheck/1.0'}
            )
            
            with urllib.request.urlopen(request, timeout=10) as response:
                latency_ms = (time.time() - start) * 1000
                
                if response.status == 200:
                    self.update_health(name, service.instance_id, ServiceStatus.HEALTHY, latency_ms)
                else:
                    self.update_health(
                        name, service.instance_id, ServiceStatus.DEGRADED, latency_ms,
                        f"HTTP {response.status}"
                    )
        
        except urllib.error.URLError as e:
            self.update_health(
                name, service.instance_id, ServiceStatus.UNHEALTHY,
                error_message=str(e.reason)
            )
        except Exception as e:
            self.update_health(
                name, service.instance_id, ServiceStatus.UNHEALTHY,
                error_message=str(e)
            )
    
    def _check_heartbeat_timeouts(self) -> None:
        """Check for services that missed heartbeats."""
        if not self.auto_deregister:
            return
        
        now = time.time()
        to_remove = []
        
        with self._lock:
            for name, instances in self._services.items():
                for instance_id, service in instances.items():
                    if now - service.last_heartbeat > self.heartbeat_timeout:
                        to_remove.append((name, instance_id))
        
        for name, instance_id in to_remove:
            logger.warning(f"Service heartbeat timeout: {name}/{instance_id}")
            self.deregister(name, instance_id)
    
    def export(self) -> str:
        """Export registry to JSON."""
        with self._lock:
            data = {
                name: [s.to_dict() for s in instances.values()]
                for name, instances in self._services.items()
            }
            return json.dumps(data, indent=2)
    
    def import_from(self, json_data: str) -> int:
        """
        Import services from JSON.
        
        Returns:
            Number of services imported
        """
        data = json.loads(json_data)
        count = 0
        
        for name, services in data.items():
            for service_data in services:
                service = Service.from_dict(service_data)
                self.register(service)
                count += 1
        
        return count


# Global registry instance
_registry: Optional[ServiceRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> ServiceRegistry:
    """Get global service registry."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = ServiceRegistry()
        return _registry


def set_registry(registry: ServiceRegistry) -> None:
    """Set global service registry."""
    global _registry
    with _registry_lock:
        _registry = registry


# Decorator for registering service providers
def service_provider(
    name: str,
    version: str = "1.0.0",
    tags: Optional[set[str]] = None,
    endpoint: Optional[str] = None,
    **metadata
):
    """
    Decorator to register a class as a service provider.
    
    Args:
        name: Service name
        version: Service version
        tags: Service tags
        endpoint: Service endpoint (auto-generated if None)
        **metadata: Additional metadata
    
    Example:
        @service_provider("inference", version="2.0.0", model="forge-small")
        class InferenceService:
            def generate(self, prompt):
                pass
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            
            service = Service(
                name=name,
                endpoint=endpoint or f"local://{cls.__name__}",
                version=version,
                tags=tags or set(),
                metadata={**metadata, "class": cls.__name__}
            )
            service.health.status = ServiceStatus.HEALTHY
            
            get_registry().register(service)
            self._service_registration = service
        
        cls.__init__ = new_init
        
        # Add cleanup method
        def cleanup(self):
            if hasattr(self, '_service_registration'):
                service = self._service_registration
                get_registry().deregister(service.name, service.instance_id)
        
        cls.__service_cleanup__ = cleanup
        
        return cls
    
    return decorator


# Pre-configured service types for Enigma AI Engine
class ForgeServices:
    """Pre-defined service names for Enigma AI Engine."""
    
    INFERENCE = "inference"
    IMAGE_GEN = "image_gen"
    AUDIO_GEN = "audio_gen"
    VIDEO_GEN = "video_gen"
    VOICE_INPUT = "voice_input"
    VOICE_OUTPUT = "voice_output"
    MEMORY = "memory"
    EMBEDDING = "embedding"
    API_SERVER = "api_server"
    WEB_UI = "web_ui"
    TRAINING = "training"
    FEDERATED = "federated"
