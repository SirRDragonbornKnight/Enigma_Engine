"""
Circuit Breaker Pattern Implementation for ForgeAI.

Prevents cascade failures by failing fast when a service is down.
Three states: CLOSED (normal), OPEN (failing fast), HALF_OPEN (testing).

Usage:
    from forge_ai.utils.circuit_breaker import circuit_breaker, CircuitBreaker
    
    # Decorator approach
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    def call_external_api():
        return requests.get("https://api.example.com")
    
    # Context manager approach
    breaker = CircuitBreaker("api_service")
    with breaker:
        response = requests.get("https://api.example.com")
    
    # Manual approach
    breaker = get_breaker("database")
    if breaker.can_execute():
        try:
            result = db.query("SELECT * FROM users")
            breaker.record_success()
        except Exception as e:
            breaker.record_failure(e)
"""

import time
import threading
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict, List, Type
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing fast, requests are rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: List[tuple] = field(default_factory=list)
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "success_rate": self.success_rate(),
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
        }


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    
    States:
        - CLOSED: Normal operation, failures are counted
        - OPEN: Fast-fail mode, all calls rejected
        - HALF_OPEN: Testing mode, limited calls allowed
    
    Transitions:
        CLOSED -> OPEN: When failure_threshold consecutive failures
        OPEN -> HALF_OPEN: After recovery_timeout seconds
        HALF_OPEN -> CLOSED: On success
        HALF_OPEN -> OPEN: On failure
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        success_threshold: int = 1,
        excluded_exceptions: Optional[List[Type[Exception]]] = None,
        on_state_change: Optional[Callable[[str, CircuitState, CircuitState], None]] = None,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Identifier for this circuit
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            half_open_max_calls: Max calls in half-open state
            success_threshold: Successes in half-open to close circuit
            excluded_exceptions: Exceptions that don't count as failures
            on_state_change: Callback when state changes
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.excluded_exceptions = excluded_exceptions or []
        self.on_state_change = on_state_change
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._lock = threading.RLock()
        self._last_state_change = time.time()
        self._half_open_calls = 0
        self._half_open_successes = 0
    
    @property
    def state(self) -> CircuitState:
        """Get current state, checking for auto-transition."""
        with self._lock:
            self._check_state_transition()
            return self._state
    
    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats
    
    def _check_state_transition(self) -> None:
        """Check if state should transition automatically."""
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout elapsed
            elapsed = time.time() - self._last_state_change
            if elapsed >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if self._state == new_state:
            return
            
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()
        
        # Reset half-open counters
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._half_open_successes = 0
        
        # Record state change
        self._stats.state_changes.append((time.time(), old_state.value, new_state.value))
        
        # Limit state change history
        if len(self._stats.state_changes) > 100:
            self._stats.state_changes = self._stats.state_changes[-100:]
        
        logger.info(f"Circuit '{self.name}' state: {old_state.value} -> {new_state.value}")
        
        # Notify callback
        if self.on_state_change:
            try:
                self.on_state_change(self.name, old_state, new_state)
            except Exception as e:
                logger.warning(f"State change callback error: {e}")
    
    def can_execute(self) -> bool:
        """Check if a call can be executed."""
        with self._lock:
            self._check_state_transition()
            
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                return False
            else:  # HALF_OPEN
                return self._half_open_calls < self.half_open_max_calls
    
    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.last_success_time = time.time()
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
    
    def record_failure(self, exception: Optional[Exception] = None) -> None:
        """Record a failed call."""
        with self._lock:
            # Check if exception should be excluded
            if exception and any(isinstance(exception, exc_type) for exc_type in self.excluded_exceptions):
                logger.debug(f"Exception {type(exception).__name__} excluded from failure count")
                return
            
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.last_failure_time = time.time()
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            
            if self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                self._transition_to(CircuitState.OPEN)
    
    def record_rejection(self) -> None:
        """Record a rejected call (circuit open)."""
        with self._lock:
            self._stats.rejected_calls += 1
    
    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._stats = CircuitStats()
    
    def force_open(self) -> None:
        """Force circuit to open state."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
    
    def force_close(self) -> None:
        """Force circuit to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
    
    def __enter__(self) -> "CircuitBreaker":
        """Context manager entry."""
        if not self.can_execute():
            self.record_rejection()
            raise CircuitBreakerError(f"Circuit '{self.name}' is open")
        
        if self._state == CircuitState.HALF_OPEN:
            with self._lock:
                self._half_open_calls += 1
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure(exc_val)
        return False  # Don't suppress exceptions
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator usage."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    **kwargs
) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before testing recovery
        **kwargs: Additional CircuitBreaker arguments
    
    Returns:
        CircuitBreaker instance
    """
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                **kwargs
            )
        return _circuit_breakers[name]


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    **kwargs
) -> Callable:
    """
    Decorator to protect a function with a circuit breaker.
    
    Args:
        name: Circuit name (defaults to function name)
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before testing recovery
        **kwargs: Additional CircuitBreaker arguments
    
    Returns:
        Decorated function
    
    Example:
        @circuit_breaker(failure_threshold=3, recovery_timeout=60)
        def call_api():
            return requests.get("https://api.example.com")
    """
    def decorator(func: Callable) -> Callable:
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        breaker = get_breaker(breaker_name, failure_threshold, recovery_timeout, **kwargs)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not breaker.can_execute():
                breaker.record_rejection()
                raise CircuitBreakerError(
                    f"Circuit '{breaker_name}' is open (state: {breaker.state.value})"
                )
            
            if breaker.state == CircuitState.HALF_OPEN:
                with breaker._lock:
                    breaker._half_open_calls += 1
            
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure(e)
                raise
        
        # Attach breaker to function for inspection
        wrapper.circuit_breaker = breaker
        return wrapper
    
    return decorator


def list_breakers() -> Dict[str, dict]:
    """List all circuit breakers and their stats."""
    with _registry_lock:
        return {
            name: {
                "state": breaker.state.value,
                "stats": breaker.stats.to_dict(),
            }
            for name, breaker in _circuit_breakers.items()
        }


def reset_all_breakers() -> None:
    """Reset all circuit breakers."""
    with _registry_lock:
        for breaker in _circuit_breakers.values():
            breaker.reset()


def remove_breaker(name: str) -> bool:
    """Remove a circuit breaker from registry."""
    with _registry_lock:
        if name in _circuit_breakers:
            del _circuit_breakers[name]
            return True
        return False


# Pre-configured breakers for common services
class Breakers:
    """Pre-configured circuit breakers for ForgeAI services."""
    
    @staticmethod
    def api() -> CircuitBreaker:
        """Circuit breaker for external API calls."""
        return get_breaker("api", failure_threshold=5, recovery_timeout=30)
    
    @staticmethod
    def database() -> CircuitBreaker:
        """Circuit breaker for database operations."""
        return get_breaker("database", failure_threshold=3, recovery_timeout=10)
    
    @staticmethod
    def model() -> CircuitBreaker:
        """Circuit breaker for model inference."""
        return get_breaker("model", failure_threshold=3, recovery_timeout=60)
    
    @staticmethod
    def network() -> CircuitBreaker:
        """Circuit breaker for network operations."""
        return get_breaker("network", failure_threshold=5, recovery_timeout=30)
    
    @staticmethod
    def voice() -> CircuitBreaker:
        """Circuit breaker for voice services."""
        return get_breaker("voice", failure_threshold=3, recovery_timeout=15)
    
    @staticmethod
    def image_gen() -> CircuitBreaker:
        """Circuit breaker for image generation."""
        return get_breaker("image_gen", failure_threshold=2, recovery_timeout=60)


# Convenience function for async code
async def async_circuit_breaker(
    name: str,
    coro,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0
):
    """
    Async-compatible circuit breaker wrapper.
    
    Args:
        name: Circuit breaker name
        coro: Coroutine to execute
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before testing recovery
    
    Returns:
        Coroutine result
    
    Example:
        result = await async_circuit_breaker(
            "api_call",
            fetch_data_async()
        )
    """
    breaker = get_breaker(name, failure_threshold, recovery_timeout)
    
    if not breaker.can_execute():
        breaker.record_rejection()
        raise CircuitBreakerError(f"Circuit '{name}' is open")
    
    if breaker.state == CircuitState.HALF_OPEN:
        with breaker._lock:
            breaker._half_open_calls += 1
    
    try:
        result = await coro
        breaker.record_success()
        return result
    except Exception as e:
        breaker.record_failure(e)
        raise
