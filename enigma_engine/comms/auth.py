"""
Rate Limiting and Authentication for enigma_engine APIs

Production-ready security features:
- Token-based authentication (API keys)
- Rate limiting (requests/minute, tokens/day)
- IP-based throttling
- Usage tracking and quotas
- JWT support for advanced auth

Usage:
    from enigma_engine.comms.auth import AuthMiddleware, RateLimiter
    
    # Add to Flask app
    auth = AuthMiddleware(app)
    rate_limiter = RateLimiter(app, requests_per_minute=60)
    
    @app.route('/generate')
    @auth.require_auth
    @rate_limiter.limit
    def generate():
        ...
"""

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Try to import JWT, provide fallback
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False


@dataclass
class APIKey:
    """API key with metadata."""
    key: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    rate_limit: Optional[int] = None  # Requests per minute
    token_quota: Optional[int] = None  # Tokens per day
    scopes: set[str] = field(default_factory=lambda: {"generate", "embed"})
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if key is valid."""
        if not self.is_active:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    def has_scope(self, scope: str) -> bool:
        """Check if key has a scope."""
        return scope in self.scopes or "*" in self.scopes


class APIKeyManager:
    """Manage API keys."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self._keys: dict[str, APIKey] = {}
        self._storage_path = storage_path
        self._lock = threading.Lock()
        
        if storage_path and os.path.exists(storage_path):
            self._load_keys()
    
    def create_key(
        self,
        name: str,
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
        token_quota: Optional[int] = None,
        scopes: Optional[set[str]] = None
    ) -> APIKey:
        """Create a new API key."""
        # Generate secure key
        key = f"forge_{secrets.token_urlsafe(32)}"
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            key=key,
            name=name,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            rate_limit=rate_limit,
            token_quota=token_quota,
            scopes=scopes or {"generate", "embed"}
        )
        
        with self._lock:
            # Store hash of key, not key itself
            key_hash = self._hash_key(key)
            self._keys[key_hash] = api_key
        
        self._save_keys()
        
        logger.info(f"Created API key: {name}")
        return api_key
    
    def validate_key(self, key: str) -> Optional[APIKey]:
        """Validate an API key."""
        key_hash = self._hash_key(key)
        
        with self._lock:
            api_key = self._keys.get(key_hash)
        
        if api_key and api_key.is_valid():
            return api_key
        return None
    
    def revoke_key(self, key: str) -> bool:
        """Revoke an API key."""
        key_hash = self._hash_key(key)
        
        with self._lock:
            if key_hash in self._keys:
                self._keys[key_hash].is_active = False
                self._save_keys()
                return True
        return False
    
    def list_keys(self) -> list[dict[str, Any]]:
        """List all keys (without the actual key values)."""
        with self._lock:
            return [
                {
                    'name': k.name,
                    'created_at': k.created_at.isoformat(),
                    'expires_at': k.expires_at.isoformat() if k.expires_at else None,
                    'is_active': k.is_active,
                    'scopes': list(k.scopes)
                }
                for k in self._keys.values()
            ]
    
    def _hash_key(self, key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _save_keys(self):
        """Save keys to storage."""
        if not self._storage_path:
            return
        
        data = {}
        for key_hash, api_key in self._keys.items():
            data[key_hash] = {
                'key': api_key.key,  # Store actual key for reloading
                'name': api_key.name,
                'created_at': api_key.created_at.isoformat(),
                'expires_at': api_key.expires_at.isoformat() if api_key.expires_at else None,
                'rate_limit': api_key.rate_limit,
                'token_quota': api_key.token_quota,
                'scopes': list(api_key.scopes),
                'is_active': api_key.is_active
            }
        
        with open(self._storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_keys(self):
        """Load keys from storage."""
        try:
            with open(self._storage_path) as f:
                data = json.load(f)
            
            for key_hash, key_data in data.items():
                self._keys[key_hash] = APIKey(
                    key=key_data['key'],
                    name=key_data['name'],
                    created_at=datetime.fromisoformat(key_data['created_at']),
                    expires_at=datetime.fromisoformat(key_data['expires_at']) if key_data['expires_at'] else None,
                    rate_limit=key_data.get('rate_limit'),
                    token_quota=key_data.get('token_quota'),
                    scopes=set(key_data.get('scopes', ['generate', 'embed'])),
                    is_active=key_data.get('is_active', True)
                )
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    tokens_per_minute: int = 100000
    tokens_per_day: int = 1000000
    burst_multiplier: float = 1.5  # Allow bursts up to 1.5x limit


class RateLimitState:
    """Track rate limit state for a key/IP."""
    
    def __init__(self):
        self.minute_requests: list[float] = []
        self.hour_requests: list[float] = []
        self.day_requests: list[float] = []
        self.minute_tokens: list[tuple[float, int]] = []  # (timestamp, tokens)
        self.day_tokens: list[tuple[float, int]] = []
    
    def cleanup(self, now: float):
        """Remove old entries."""
        minute_ago = now - 60
        hour_ago = now - 3600
        day_ago = now - 86400
        
        self.minute_requests = [t for t in self.minute_requests if t > minute_ago]
        self.hour_requests = [t for t in self.hour_requests if t > hour_ago]
        self.day_requests = [t for t in self.day_requests if t > day_ago]
        self.minute_tokens = [(t, n) for t, n in self.minute_tokens if t > minute_ago]
        self.day_tokens = [(t, n) for t, n in self.day_tokens if t > day_ago]
    
    def get_minute_requests(self) -> int:
        return len(self.minute_requests)
    
    def get_hour_requests(self) -> int:
        return len(self.hour_requests)
    
    def get_day_requests(self) -> int:
        return len(self.day_requests)
    
    def get_minute_tokens(self) -> int:
        return sum(n for _, n in self.minute_tokens)
    
    def get_day_tokens(self) -> int:
        return sum(n for _, n in self.day_tokens)


class RateLimiter:
    """
    Rate limiter for API requests.
    
    Features:
    - Per-key and per-IP rate limiting
    - Token-based quotas
    - Sliding window algorithm
    - Configurable limits
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._states: dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = threading.Lock()
    
    def check_limit(
        self,
        identifier: str,
        tokens: int = 0,
        custom_limit: Optional[int] = None
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if request is within rate limits.
        
        Args:
            identifier: Key or IP to check
            tokens: Number of tokens for this request
            custom_limit: Custom per-minute limit for this key
        
        Returns:
            (allowed, info) where info contains limit details
        """
        now = time.time()
        
        with self._lock:
            state = self._states[identifier]
            state.cleanup(now)
            
            # Check request limits
            rpm_limit = custom_limit or self.config.requests_per_minute
            
            if state.get_minute_requests() >= rpm_limit * self.config.burst_multiplier:
                return False, {
                    'error': 'rate_limit_exceeded',
                    'limit': rpm_limit,
                    'current': state.get_minute_requests(),
                    'retry_after': 60 - (now - state.minute_requests[0]) if state.minute_requests else 60
                }
            
            if state.get_hour_requests() >= self.config.requests_per_hour:
                return False, {
                    'error': 'hourly_limit_exceeded',
                    'limit': self.config.requests_per_hour,
                    'current': state.get_hour_requests()
                }
            
            if state.get_day_requests() >= self.config.requests_per_day:
                return False, {
                    'error': 'daily_limit_exceeded',
                    'limit': self.config.requests_per_day,
                    'current': state.get_day_requests()
                }
            
            # Check token limits
            if tokens > 0:
                if state.get_minute_tokens() + tokens > self.config.tokens_per_minute:
                    return False, {
                        'error': 'token_rate_limit_exceeded',
                        'limit': self.config.tokens_per_minute,
                        'current': state.get_minute_tokens()
                    }
                
                if state.get_day_tokens() + tokens > self.config.tokens_per_day:
                    return False, {
                        'error': 'daily_token_limit_exceeded',
                        'limit': self.config.tokens_per_day,
                        'current': state.get_day_tokens()
                    }
            
            # Record request
            state.minute_requests.append(now)
            state.hour_requests.append(now)
            state.day_requests.append(now)
            
            if tokens > 0:
                state.minute_tokens.append((now, tokens))
                state.day_tokens.append((now, tokens))
            
            return True, {
                'remaining_requests': rpm_limit - state.get_minute_requests(),
                'remaining_tokens': self.config.tokens_per_minute - state.get_minute_tokens()
            }
    
    def get_usage(self, identifier: str) -> dict[str, Any]:
        """Get usage statistics for an identifier."""
        with self._lock:
            state = self._states.get(identifier)
            if not state:
                return {'requests': 0, 'tokens': 0}
            
            state.cleanup(time.time())
            return {
                'requests_minute': state.get_minute_requests(),
                'requests_hour': state.get_hour_requests(),
                'requests_day': state.get_day_requests(),
                'tokens_minute': state.get_minute_tokens(),
                'tokens_day': state.get_day_tokens()
            }


class AuthMiddleware:
    """
    Authentication middleware for Flask/FastAPI.
    
    Usage (Flask):
        auth = AuthMiddleware(key_manager)
        
        @app.route('/api/generate')
        @auth.require_auth(scope='generate')
        def generate():
            ...
    """
    
    def __init__(
        self,
        key_manager: APIKeyManager,
        rate_limiter: Optional[RateLimiter] = None,
        jwt_secret: Optional[str] = None
    ):
        self.key_manager = key_manager
        self.rate_limiter = rate_limiter or RateLimiter()
        self.jwt_secret = jwt_secret or os.urandom(32).hex()
    
    def require_auth(self, scope: str = "generate"):
        """Decorator to require authentication."""
        def decorator(f: Callable) -> Callable:
            @wraps(f)
            def wrapped(*args, **kwargs):
                # Import Flask here to avoid dependency issues
                from flask import g, jsonify, request

                # Get API key from header
                auth_header = request.headers.get('Authorization', '')
                
                if auth_header.startswith('Bearer '):
                    token = auth_header[7:]
                    
                    # Try JWT first
                    if JWT_AVAILABLE and self._validate_jwt(token):
                        return f(*args, **kwargs)
                    
                    # Try API key
                    api_key = self.key_manager.validate_key(token)
                    if api_key:
                        if not api_key.has_scope(scope):
                            return jsonify({'error': 'Insufficient permissions'}), 403
                        
                        # Check rate limit
                        allowed, info = self.rate_limiter.check_limit(
                            token,
                            custom_limit=api_key.rate_limit
                        )
                        
                        if not allowed:
                            return jsonify(info), 429
                        
                        # Store key info for handler
                        g.api_key = api_key
                        g.rate_limit_info = info
                        
                        return f(*args, **kwargs)
                
                return jsonify({'error': 'Unauthorized'}), 401
            
            return wrapped
        return decorator
    
    def _validate_jwt(self, token: str) -> bool:
        """Validate a JWT token."""
        if not JWT_AVAILABLE:
            return False
        
        try:
            jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return True
        except jwt.InvalidTokenError:
            return False
    
    def create_jwt(
        self,
        user_id: str,
        expires_in_hours: int = 24,
        scopes: Optional[list[str]] = None
    ) -> str:
        """Create a JWT token."""
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT not installed")
        
        payload = {
            'sub': user_id,
            'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
            'iat': datetime.utcnow(),
            'scopes': scopes or ['generate']
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')


class IPThrottler:
    """IP-based throttling for DDoS protection."""
    
    def __init__(
        self,
        max_requests_per_second: int = 10,
        ban_duration_seconds: int = 300
    ):
        self.max_requests_per_second = max_requests_per_second
        self.ban_duration_seconds = ban_duration_seconds
        
        self._request_counts: dict[str, list[float]] = defaultdict(list)
        self._banned_ips: dict[str, float] = {}
        self._lock = threading.Lock()
    
    def check_ip(self, ip: str) -> tuple[bool, Optional[str]]:
        """
        Check if IP is allowed.
        
        Returns:
            (allowed, reason)
        """
        now = time.time()
        
        with self._lock:
            # Check if banned
            if ip in self._banned_ips:
                if now < self._banned_ips[ip]:
                    return False, "IP temporarily banned"
                else:
                    del self._banned_ips[ip]
            
            # Clean old requests
            self._request_counts[ip] = [
                t for t in self._request_counts[ip]
                if t > now - 1
            ]
            
            # Check rate
            if len(self._request_counts[ip]) >= self.max_requests_per_second:
                self._banned_ips[ip] = now + self.ban_duration_seconds
                return False, "Rate limit exceeded, IP banned"
            
            self._request_counts[ip].append(now)
            return True, None


def create_auth_system(
    storage_path: Optional[str] = None,
    requests_per_minute: int = 60
) -> tuple[APIKeyManager, RateLimiter, AuthMiddleware]:
    """
    Create a complete authentication system.
    
    Args:
        storage_path: Path to store API keys
        requests_per_minute: Default rate limit
    
    Returns:
        (key_manager, rate_limiter, auth_middleware)
    """
    key_manager = APIKeyManager(storage_path)
    rate_limiter = RateLimiter(RateLimitConfig(requests_per_minute=requests_per_minute))
    auth_middleware = AuthMiddleware(key_manager, rate_limiter)
    
    return key_manager, rate_limiter, auth_middleware
