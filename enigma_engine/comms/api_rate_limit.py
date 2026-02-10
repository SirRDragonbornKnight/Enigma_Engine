"""
API Rate Limiting Middleware

Rate limiting decorator for Flask API endpoints.
Integrates with the existing RateLimiter from tools.rate_limiter.

Usage:
    from enigma_engine.comms.api_rate_limit import rate_limit, APIRateLimiter
    
    @app.route("/generate", methods=["POST"])
    @rate_limit(requests_per_minute=10)
    def generate():
        ...
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock
from typing import Callable, Optional

from flask import jsonify, request

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration for an endpoint."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10  # Allow short bursts


@dataclass
class ClientState:
    """Track rate limit state for a client."""
    minute_requests: list = field(default_factory=list)
    hour_requests: list = field(default_factory=list)
    day_requests: list = field(default_factory=list)
    last_request: float = 0.0
    blocked_until: float = 0.0


class APIRateLimiter:
    """
    Rate limiter for API endpoints.
    
    Supports per-client tracking with multiple time windows.
    """
    
    def __init__(self):
        self._clients: dict[str, ClientState] = defaultdict(ClientState)
        self._lock = Lock()
        self._endpoint_configs: dict[str, RateLimitConfig] = {}
    
    def configure_endpoint(self, endpoint: str, config: RateLimitConfig):
        """Configure rate limits for an endpoint."""
        self._endpoint_configs[endpoint] = config
    
    def get_client_id(self) -> str:
        """Get client identifier from request."""
        # Use API key if available, else IP
        api_key = (
            request.headers.get('X-API-Key') or
            request.headers.get('Authorization', '').replace('Bearer ', '')
        )
        if api_key:
            return f"key:{api_key[:16]}"
        return f"ip:{request.remote_addr}"
    
    def _cleanup_old_requests(self, state: ClientState, now: float):
        """Remove expired requests from tracking lists."""
        minute_cutoff = now - 60
        hour_cutoff = now - 3600
        day_cutoff = now - 86400
        
        state.minute_requests = [t for t in state.minute_requests if t > minute_cutoff]
        state.hour_requests = [t for t in state.hour_requests if t > hour_cutoff]
        state.day_requests = [t for t in state.day_requests if t > day_cutoff]
    
    def check_rate_limit(self, endpoint: str) -> tuple[bool, Optional[str], Optional[int]]:
        """
        Check if request is allowed.
        
        Returns:
            Tuple of (allowed, error_message, retry_after_seconds)
        """
        config = self._endpoint_configs.get(endpoint, RateLimitConfig())
        client_id = self.get_client_id()
        now = time.time()
        
        with self._lock:
            state = self._clients[client_id]
            
            # Check if client is blocked
            if state.blocked_until > now:
                retry_after = int(state.blocked_until - now) + 1
                return False, "Rate limit exceeded - temporarily blocked", retry_after
            
            # Cleanup old requests
            self._cleanup_old_requests(state, now)
            
            # Check limits
            if len(state.minute_requests) >= config.requests_per_minute:
                state.blocked_until = now + 60
                return False, f"Rate limit: {config.requests_per_minute}/minute exceeded", 60
            
            if len(state.hour_requests) >= config.requests_per_hour:
                state.blocked_until = now + 300  # 5 min cooldown
                return False, f"Rate limit: {config.requests_per_hour}/hour exceeded", 300
            
            if len(state.day_requests) >= config.requests_per_day:
                state.blocked_until = now + 3600  # 1 hour cooldown
                return False, f"Rate limit: {config.requests_per_day}/day exceeded", 3600
            
            # Record request
            state.minute_requests.append(now)
            state.hour_requests.append(now)
            state.day_requests.append(now)
            state.last_request = now
            
            return True, None, None
    
    def get_remaining(self, endpoint: str) -> dict[str, int]:
        """Get remaining requests for current client."""
        config = self._endpoint_configs.get(endpoint, RateLimitConfig())
        client_id = self.get_client_id()
        now = time.time()
        
        with self._lock:
            state = self._clients[client_id]
            self._cleanup_old_requests(state, now)
            
            return {
                "minute": max(0, config.requests_per_minute - len(state.minute_requests)),
                "hour": max(0, config.requests_per_hour - len(state.hour_requests)),
                "day": max(0, config.requests_per_day - len(state.day_requests)),
            }


# Global rate limiter instance
_api_limiter: Optional[APIRateLimiter] = None


def get_api_limiter() -> APIRateLimiter:
    """Get or create global API rate limiter."""
    global _api_limiter
    if _api_limiter is None:
        _api_limiter = APIRateLimiter()
    return _api_limiter


def rate_limit(
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    requests_per_day: int = 10000
):
    """
    Decorator to apply rate limiting to a Flask endpoint.
    
    Usage:
        @app.route("/generate", methods=["POST"])
        @rate_limit(requests_per_minute=10)
        def generate():
            ...
    """
    def decorator(f: Callable) -> Callable:
        endpoint_name = f.__name__
        
        # Configure the rate limit
        limiter = get_api_limiter()
        limiter.configure_endpoint(endpoint_name, RateLimitConfig(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            requests_per_day=requests_per_day
        ))
        
        @wraps(f)
        def wrapped(*args, **kwargs):
            limiter = get_api_limiter()
            allowed, error, retry_after = limiter.check_rate_limit(endpoint_name)
            
            if not allowed:
                response = jsonify({
                    "error": "Rate Limited",
                    "message": error,
                    "retry_after": retry_after
                })
                response.status_code = 429
                response.headers["Retry-After"] = str(retry_after)
                
                # Add rate limit headers
                remaining = limiter.get_remaining(endpoint_name)
                response.headers["X-RateLimit-Remaining-Minute"] = str(remaining["minute"])
                response.headers["X-RateLimit-Remaining-Hour"] = str(remaining["hour"])
                
                logger.warning(f"Rate limit hit for {endpoint_name}: {error}")
                return response
            
            # Add rate limit headers to successful responses
            response = f(*args, **kwargs)
            
            # If it's a tuple (response, status_code), handle that
            if isinstance(response, tuple):
                return response
            
            # Add headers if we have a response object
            try:
                remaining = limiter.get_remaining(endpoint_name)
                response.headers["X-RateLimit-Remaining-Minute"] = str(remaining["minute"])
            except Exception:
                pass
            
            return response
        
        return wrapped
    return decorator
