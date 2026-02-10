"""
API Security and Rate Limiting for Enigma AI Engine

Secure API access with rate limiting, authentication, and validation.

Features:
- Rate limiting (per-user, global)
- API key authentication
- Request validation
- IP blocking
- Usage tracking
- Cost estimation

Usage:
    from enigma_engine.comms.api_security import RateLimiter, APIAuthenticator
    
    # Rate limiting
    limiter = RateLimiter(requests_per_minute=60)
    
    @limiter.limit
    def api_endpoint(request):
        ...
    
    # Authentication
    auth = APIAuthenticator()
    key = auth.create_key("user123", tier="pro")
    
    if auth.validate(request_key):
        process_request()
"""

import logging
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Rate limit exceeded."""
    def __init__(self, message: str, retry_after: float = 60.0):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(Exception):
    """Authentication failed."""


class APITier(Enum):
    """API access tiers."""
    FREE = auto()
    BASIC = auto()
    PRO = auto()
    ENTERPRISE = auto()
    UNLIMITED = auto()


@dataclass
class TierLimits:
    """Limits for each tier."""
    requests_per_minute: int
    requests_per_day: int
    max_tokens_per_request: int
    concurrent_requests: int
    cost_per_1k_tokens: float = 0.0
    
    @classmethod
    def get_limits(cls, tier: APITier) -> "TierLimits":
        """Get limits for a tier."""
        limits = {
            APITier.FREE: cls(10, 100, 1024, 1, 0.0),
            APITier.BASIC: cls(30, 1000, 4096, 2, 0.001),
            APITier.PRO: cls(60, 10000, 8192, 5, 0.002),
            APITier.ENTERPRISE: cls(300, 100000, 32768, 20, 0.001),
            APITier.UNLIMITED: cls(1000, 1000000, 131072, 100, 0.0)
        }
        return limits.get(tier, limits[APITier.FREE])


@dataclass
class APIKey:
    """API key data."""
    key: str
    user_id: str
    tier: APITier
    created: datetime
    expires: Optional[datetime] = None
    active: bool = True
    
    # Usage tracking
    requests_today: int = 0
    tokens_used: int = 0
    last_request: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if key is valid."""
        if not self.active:
            return False
        if self.expires and datetime.now() > self.expires:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key[:8] + "...",  # Masked
            "user_id": self.user_id,
            "tier": self.tier.name,
            "created": self.created.isoformat(),
            "expires": self.expires.isoformat() if self.expires else None,
            "active": self.active,
            "requests_today": self.requests_today,
            "tokens_used": self.tokens_used
        }


@dataclass
class RequestRecord:
    """Record of an API request."""
    timestamp: float
    user_id: str
    endpoint: str
    tokens: int = 0
    success: bool = True
    latency_ms: float = 0.0


class RateLimiter:
    """
    Rate limiter with sliding window.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_day: int = 10000,
        window_seconds: int = 60
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Max requests per minute
            requests_per_day: Max requests per day
            window_seconds: Sliding window size
        """
        self._rpm = requests_per_minute
        self._rpd = requests_per_day
        self._window = window_seconds
        
        # Request tracking per user
        self._requests: Dict[str, List[float]] = {}
        self._daily_counts: Dict[str, int] = {}
        self._daily_reset: float = time.time()
        
        self._lock = threading.Lock()
    
    def check(self, user_id: str = "default") -> Tuple[bool, float]:
        """
        Check if request is allowed.
        
        Args:
            user_id: User identifier
            
        Returns:
            (allowed, retry_after_seconds)
        """
        with self._lock:
            now = time.time()
            
            # Reset daily counts if needed
            if now - self._daily_reset > 86400:  # 24 hours
                self._daily_counts = {}
                self._daily_reset = now
            
            # Get user's request history
            if user_id not in self._requests:
                self._requests[user_id] = []
            
            requests = self._requests[user_id]
            
            # Remove old requests outside window
            cutoff = now - self._window
            requests = [r for r in requests if r > cutoff]
            self._requests[user_id] = requests
            
            # Check minute limit
            if len(requests) >= self._rpm:
                oldest = min(requests)
                retry_after = oldest + self._window - now
                return False, max(0, retry_after)
            
            # Check daily limit
            daily = self._daily_counts.get(user_id, 0)
            if daily >= self._rpd:
                # Reset at midnight
                retry_after = 86400 - (now - self._daily_reset)
                return False, max(0, retry_after)
            
            return True, 0.0
    
    def record(self, user_id: str = "default"):
        """Record a request."""
        with self._lock:
            now = time.time()
            
            if user_id not in self._requests:
                self._requests[user_id] = []
            
            self._requests[user_id].append(now)
            self._daily_counts[user_id] = self._daily_counts.get(user_id, 0) + 1
    
    def limit(self, func: Callable) -> Callable:
        """Decorator to apply rate limiting."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get user_id from request
            user_id = kwargs.get("user_id", "default")
            
            allowed, retry_after = self.check(user_id)
            if not allowed:
                raise RateLimitExceeded(
                    f"Rate limit exceeded. Retry after {retry_after:.1f}s",
                    retry_after
                )
            
            result = func(*args, **kwargs)
            self.record(user_id)
            return result
        
        return wrapper
    
    def get_usage(self, user_id: str = "default") -> Dict[str, Any]:
        """Get usage stats for a user."""
        with self._lock:
            requests = self._requests.get(user_id, [])
            daily = self._daily_counts.get(user_id, 0)
            
            # Count requests in current window
            now = time.time()
            cutoff = now - self._window
            recent = len([r for r in requests if r > cutoff])
            
            return {
                "requests_in_window": recent,
                "window_limit": self._rpm,
                "requests_today": daily,
                "daily_limit": self._rpd,
                "window_seconds": self._window
            }


class APIAuthenticator:
    """
    API key authentication and management.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize authenticator.
        
        Args:
            db_path: Path to SQLite database for persistence
        """
        self._keys: Dict[str, APIKey] = {}
        self._db_path = db_path
        self._lock = threading.Lock()
        
        if db_path:
            self._init_db()
            self._load_keys()
    
    def _init_db(self):
        """Initialize database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    created TEXT NOT NULL,
                    expires TEXT,
                    active INTEGER DEFAULT 1,
                    requests_today INTEGER DEFAULT 0,
                    tokens_used INTEGER DEFAULT 0,
                    last_request TEXT
                )
            """)
            conn.commit()
    
    def _load_keys(self):
        """Load keys from database."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("SELECT * FROM api_keys")
            for row in cursor:
                key = APIKey(
                    key=row[0],
                    user_id=row[1],
                    tier=APITier[row[2]],
                    created=datetime.fromisoformat(row[3]),
                    expires=datetime.fromisoformat(row[4]) if row[4] else None,
                    active=bool(row[5]),
                    requests_today=row[6],
                    tokens_used=row[7],
                    last_request=datetime.fromisoformat(row[8]) if row[8] else None
                )
                self._keys[key.key] = key
    
    def _save_key(self, key: APIKey):
        """Save key to database."""
        if not self._db_path:
            return
        
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO api_keys
                (key, user_id, tier, created, expires, active, requests_today, tokens_used, last_request)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                key.key,
                key.user_id,
                key.tier.name,
                key.created.isoformat(),
                key.expires.isoformat() if key.expires else None,
                int(key.active),
                key.requests_today,
                key.tokens_used,
                key.last_request.isoformat() if key.last_request else None
            ))
            conn.commit()
    
    def create_key(
        self,
        user_id: str,
        tier: str | APITier = APITier.FREE,
        expires_days: Optional[int] = None
    ) -> str:
        """
        Create a new API key.
        
        Args:
            user_id: User identifier
            tier: Access tier
            expires_days: Days until expiration
            
        Returns:
            API key string
        """
        with self._lock:
            # Generate secure key
            raw_key = secrets.token_urlsafe(32)
            
            # Parse tier
            if isinstance(tier, str):
                tier = APITier[tier.upper()]
            
            # Calculate expiration
            expires = None
            if expires_days:
                expires = datetime.now() + timedelta(days=expires_days)
            
            key = APIKey(
                key=raw_key,
                user_id=user_id,
                tier=tier,
                created=datetime.now(),
                expires=expires
            )
            
            self._keys[raw_key] = key
            self._save_key(key)
            
            logger.info(f"Created API key for user {user_id} (tier: {tier.name})")
            return raw_key
    
    def validate(self, key: str) -> Optional[APIKey]:
        """
        Validate an API key.
        
        Args:
            key: API key to validate
            
        Returns:
            APIKey if valid, None if invalid
        """
        with self._lock:
            api_key = self._keys.get(key)
            
            if api_key is None:
                return None
            
            if not api_key.is_valid():
                return None
            
            return api_key
    
    def revoke(self, key: str) -> bool:
        """Revoke an API key."""
        with self._lock:
            if key in self._keys:
                self._keys[key].active = False
                self._save_key(self._keys[key])
                logger.info(f"Revoked API key: {key[:8]}...")
                return True
            return False
    
    def record_usage(self, key: str, tokens: int = 0):
        """Record usage for a key."""
        with self._lock:
            if key in self._keys:
                api_key = self._keys[key]
                api_key.requests_today += 1
                api_key.tokens_used += tokens
                api_key.last_request = datetime.now()
                self._save_key(api_key)
    
    def get_key(self, key: str) -> Optional[APIKey]:
        """Get API key info (masked)."""
        return self._keys.get(key)
    
    def list_keys(self, user_id: Optional[str] = None) -> List[APIKey]:
        """List API keys."""
        keys = list(self._keys.values())
        if user_id:
            keys = [k for k in keys if k.user_id == user_id]
        return keys


class IPBlocker:
    """
    Block suspicious IP addresses.
    """
    
    def __init__(self):
        """Initialize IP blocker."""
        self._blocked: Set[str] = set()
        self._suspicious: Dict[str, int] = {}  # IP -> strike count
        self._strike_threshold = 5
        self._lock = threading.Lock()
    
    def check(self, ip: str) -> bool:
        """
        Check if IP is allowed.
        
        Args:
            ip: IP address
            
        Returns:
            True if allowed
        """
        with self._lock:
            return ip not in self._blocked
    
    def block(self, ip: str, reason: str = ""):
        """Block an IP."""
        with self._lock:
            self._blocked.add(ip)
            logger.warning(f"Blocked IP: {ip} ({reason})")
    
    def unblock(self, ip: str):
        """Unblock an IP."""
        with self._lock:
            self._blocked.discard(ip)
            logger.info(f"Unblocked IP: {ip}")
    
    def strike(self, ip: str) -> int:
        """
        Add a strike to an IP.
        
        Args:
            ip: IP address
            
        Returns:
            Current strike count
        """
        with self._lock:
            self._suspicious[ip] = self._suspicious.get(ip, 0) + 1
            count = self._suspicious[ip]
            
            if count >= self._strike_threshold:
                self._blocked.add(ip)
                logger.warning(f"Auto-blocked IP after {count} strikes: {ip}")
            
            return count
    
    def get_blocked(self) -> List[str]:
        """Get list of blocked IPs."""
        with self._lock:
            return list(self._blocked)


class UsageTracker:
    """
    Track API usage and estimate costs.
    """
    
    # Maximum in-memory records to prevent unbounded growth
    _max_records: int = 10000
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize usage tracker."""
        self._db_path = db_path
        self._records: List[RequestRecord] = []
        self._lock = threading.Lock()
        
        if db_path:
            self._init_db()
    
    def _init_db(self):
        """Initialize database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    timestamp REAL,
                    user_id TEXT,
                    endpoint TEXT,
                    tokens INTEGER,
                    success INTEGER,
                    latency_ms REAL
                )
            """)
            conn.commit()
    
    def record(
        self,
        user_id: str,
        endpoint: str,
        tokens: int = 0,
        success: bool = True,
        latency_ms: float = 0.0
    ):
        """Record a request."""
        record = RequestRecord(
            timestamp=time.time(),
            user_id=user_id,
            endpoint=endpoint,
            tokens=tokens,
            success=success,
            latency_ms=latency_ms
        )
        
        with self._lock:
            self._records.append(record)
            # Trim oldest records if exceeding limit
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]
        
        if self._db_path:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    INSERT INTO usage (timestamp, user_id, endpoint, tokens, success, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    record.timestamp,
                    record.user_id,
                    record.endpoint,
                    record.tokens,
                    int(record.success),
                    record.latency_ms
                ))
                conn.commit()
    
    def get_stats(
        self,
        user_id: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get usage statistics."""
        cutoff = time.time() - (hours * 3600)
        
        with self._lock:
            records = [r for r in self._records if r.timestamp > cutoff]
            if user_id:
                records = [r for r in records if r.user_id == user_id]
        
        if not records:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "success_rate": 1.0,
                "avg_latency_ms": 0.0
            }
        
        total_requests = len(records)
        total_tokens = sum(r.tokens for r in records)
        success_count = sum(1 for r in records if r.success)
        avg_latency = sum(r.latency_ms for r in records) / total_requests
        
        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "success_rate": success_count / total_requests,
            "avg_latency_ms": round(avg_latency, 2),
            "time_range_hours": hours
        }
    
    def estimate_cost(
        self,
        user_id: str,
        tier: APITier,
        hours: int = 24
    ) -> float:
        """Estimate cost for a user."""
        stats = self.get_stats(user_id, hours)
        limits = TierLimits.get_limits(tier)
        
        tokens = stats["total_tokens"]
        cost = (tokens / 1000) * limits.cost_per_1k_tokens
        
        return round(cost, 4)


def secure_api_middleware(
    rate_limiter: RateLimiter,
    authenticator: APIAuthenticator,
    ip_blocker: Optional[IPBlocker] = None
) -> Callable:
    """
    Create middleware for API security.
    
    Usage with Flask:
        @app.before_request
        def security_check():
            secure_api_middleware(limiter, auth)(request)
    """
    def middleware(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get request info
            request = kwargs.get("request")
            if request is None and args:
                request = args[0]
            
            # Get IP and key from request
            ip = getattr(request, "remote_addr", "unknown") if request else "unknown"
            key = None
            
            if request:
                key = (
                    getattr(request, "headers", {}).get("Authorization", "").replace("Bearer ", "") or
                    getattr(request, "headers", {}).get("X-API-Key", "")
                )
            
            # Check IP block
            if ip_blocker and not ip_blocker.check(ip):
                raise AuthenticationError(f"IP blocked: {ip}")
            
            # Validate API key
            if not key:
                raise AuthenticationError("API key required")
            
            api_key = authenticator.validate(key)
            if not api_key:
                if ip_blocker:
                    ip_blocker.strike(ip)
                raise AuthenticationError("Invalid API key")
            
            # Check rate limit
            limits = TierLimits.get_limits(api_key.tier)
            rate_limiter._rpm = limits.requests_per_minute
            rate_limiter._rpd = limits.requests_per_day
            
            allowed, retry_after = rate_limiter.check(api_key.user_id)
            if not allowed:
                raise RateLimitExceeded(
                    "Rate limit exceeded",
                    retry_after
                )
            
            # Process request
            result = func(*args, **kwargs)
            
            # Record usage
            rate_limiter.record(api_key.user_id)
            authenticator.record_usage(key)
            
            return result
        
        return wrapper
    
    return middleware
