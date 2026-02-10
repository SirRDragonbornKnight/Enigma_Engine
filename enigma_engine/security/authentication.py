"""
Authentication System

User authentication with password hashing and session management.
Supports local accounts and OAuth providers.

FILE: enigma_engine/security/authentication.py
TYPE: Security System
MAIN CLASSES: AuthManager, User, Session
"""

import base64
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AuthProvider(Enum):
    """Authentication providers."""
    LOCAL = "local"
    OAUTH_GOOGLE = "oauth_google"
    OAUTH_GITHUB = "oauth_github"
    API_KEY = "api_key"
    SSO = "sso"


class UserRole(Enum):
    """User roles."""
    GUEST = "guest"
    USER = "user"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"


@dataclass
class User:
    """User account."""
    id: str
    username: str
    email: str
    password_hash: str = ""
    salt: str = ""
    role: UserRole = UserRole.USER
    provider: AuthProvider = AuthProvider.LOCAL
    provider_id: str = ""
    created_at: float = field(default_factory=time.time)
    last_login: float = 0.0
    is_active: bool = True
    is_verified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_sensitive: bool = False) -> dict:
        data = {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "provider": self.provider.value,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "is_active": self.is_active,
            "is_verified": self.is_verified
        }
        if include_sensitive:
            data["password_hash"] = self.password_hash
            data["salt"] = self.salt
            data["provider_id"] = self.provider_id
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'User':
        return cls(
            id=data["id"],
            username=data["username"],
            email=data.get("email", ""),
            password_hash=data.get("password_hash", ""),
            salt=data.get("salt", ""),
            role=UserRole(data.get("role", "user")),
            provider=AuthProvider(data.get("provider", "local")),
            provider_id=data.get("provider_id", ""),
            created_at=data.get("created_at", time.time()),
            last_login=data.get("last_login", 0.0),
            is_active=data.get("is_active", True),
            is_verified=data.get("is_verified", False),
            metadata=data.get("metadata", {})
        )


@dataclass
class Session:
    """Authentication session."""
    id: str
    user_id: str
    token: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    ip_address: str = ""
    user_agent: str = ""
    is_active: bool = True
    
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "ip_address": self.ip_address
        }


class PasswordHasher:
    """Secure password hashing."""
    
    ITERATIONS = 100000
    
    @classmethod
    def hash_password(cls, password: str, salt: str = None) -> tuple[str, str]:
        """
        Hash a password with salt.
        
        Args:
            password: Plain text password
            salt: Salt (generates new if None)
            
        Returns:
            Tuple of (hash, salt)
        """
        if salt is None:
            salt = secrets.token_hex(32)
        
        # PBKDF2 with SHA-256
        hash_bytes = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            cls.ITERATIONS
        )
        
        password_hash = base64.b64encode(hash_bytes).decode('utf-8')
        return password_hash, salt
    
    @classmethod
    def verify_password(cls, password: str, hash: str, salt: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Plain text password
            hash: Stored password hash
            salt: Stored salt
            
        Returns:
            True if password matches
        """
        test_hash, _ = cls.hash_password(password, salt)
        return secrets.compare_digest(test_hash, hash)


class TokenGenerator:
    """Generates secure tokens."""
    
    @staticmethod
    def generate_token(length: int = 64) -> str:
        """Generate a secure random token."""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_user_id() -> str:
        """Generate a unique user ID."""
        return f"u_{secrets.token_hex(12)}"
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate a unique session ID."""
        return f"s_{secrets.token_hex(16)}"


@dataclass
class AuthConfig:
    """Authentication configuration."""
    min_password_length: int = 8
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_symbols: bool = False
    session_duration: int = 86400 * 7  # 7 days
    max_sessions_per_user: int = 5
    allow_registration: bool = True
    require_email_verification: bool = False
    lockout_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes


class AuthManager:
    """Manages authentication and sessions."""
    
    def __init__(self, 
                 storage_path: Optional[Path] = None,
                 config: Optional[AuthConfig] = None):
        """
        Initialize auth manager.
        
        Args:
            storage_path: Path for persistent storage
            config: Auth configuration
        """
        self._storage_path = storage_path
        self._config = config or AuthConfig()
        
        self._users: dict[str, User] = {}
        self._users_by_username: dict[str, str] = {}  # username -> user_id
        self._users_by_email: dict[str, str] = {}     # email -> user_id
        self._sessions: dict[str, Session] = {}
        self._failed_attempts: dict[str, list[float]] = {}  # username -> timestamps
        
        if storage_path:
            self._load()
    
    def register(self,
                 username: str,
                 email: str,
                 password: str,
                 role: UserRole = UserRole.USER) -> tuple[Optional[User], str]:
        """
        Register a new user.
        
        Args:
            username: Desired username
            email: Email address
            password: Plain text password
            role: User role
            
        Returns:
            Tuple of (User or None, error message)
        """
        if not self._config.allow_registration:
            return None, "Registration is disabled"
        
        # Validate username
        if len(username) < 3:
            return None, "Username must be at least 3 characters"
        if username.lower() in self._users_by_username:
            return None, "Username already taken"
        
        # Validate email
        if "@" not in email or "." not in email:
            return None, "Invalid email address"
        if email.lower() in self._users_by_email:
            return None, "Email already registered"
        
        # Validate password
        valid, msg = self._validate_password(password)
        if not valid:
            return None, msg
        
        # Create user
        password_hash, salt = PasswordHasher.hash_password(password)
        
        user = User(
            id=TokenGenerator.generate_user_id(),
            username=username,
            email=email.lower(),
            password_hash=password_hash,
            salt=salt,
            role=role,
            is_verified=not self._config.require_email_verification
        )
        
        self._users[user.id] = user
        self._users_by_username[username.lower()] = user.id
        self._users_by_email[email.lower()] = user.id
        
        self._save()
        
        logger.info(f"User registered: {username}")
        return user, ""
    
    def login(self,
              username_or_email: str,
              password: str,
              ip_address: str = "",
              user_agent: str = "") -> tuple[Optional[Session], str]:
        """
        Login with username/email and password.
        
        Args:
            username_or_email: Username or email
            password: Plain text password
            ip_address: Client IP
            user_agent: Client user agent
            
        Returns:
            Tuple of (Session or None, error message)
        """
        # Check lockout
        if self._is_locked_out(username_or_email):
            return None, "Account temporarily locked due to failed attempts"
        
        # Find user
        user = self._find_user(username_or_email)
        if not user:
            self._record_failed_attempt(username_or_email)
            return None, "Invalid credentials"
        
        # Check if active
        if not user.is_active:
            return None, "Account is disabled"
        
        # Verify password
        if not PasswordHasher.verify_password(password, user.password_hash, user.salt):
            self._record_failed_attempt(username_or_email)
            return None, "Invalid credentials"
        
        # Clear failed attempts
        if username_or_email in self._failed_attempts:
            del self._failed_attempts[username_or_email]
        
        # Create session
        session = self._create_session(user, ip_address, user_agent)
        
        # Update last login
        user.last_login = time.time()
        self._save()
        
        logger.info(f"User logged in: {user.username}")
        return session, ""
    
    def logout(self, session_token: str) -> bool:
        """
        Logout (invalidate session).
        
        Args:
            session_token: Session token
            
        Returns:
            True if session was invalidated
        """
        session = self._get_session_by_token(session_token)
        if session:
            session.is_active = False
            self._save()
            return True
        return False
    
    def validate_session(self, session_token: str) -> Optional[User]:
        """
        Validate a session token.
        
        Args:
            session_token: Session token
            
        Returns:
            User if session valid, None otherwise
        """
        session = self._get_session_by_token(session_token)
        if not session:
            return None
        
        if not session.is_active or session.is_expired():
            return None
        
        user = self._users.get(session.user_id)
        if not user or not user.is_active:
            return None
        
        return user
    
    def change_password(self,
                        user_id: str,
                        old_password: str,
                        new_password: str) -> tuple[bool, str]:
        """
        Change user password.
        
        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password
            
        Returns:
            Tuple of (success, message)
        """
        user = self._users.get(user_id)
        if not user:
            return False, "User not found"
        
        # Verify old password
        if not PasswordHasher.verify_password(old_password, user.password_hash, user.salt):
            return False, "Current password is incorrect"
        
        # Validate new password
        valid, msg = self._validate_password(new_password)
        if not valid:
            return False, msg
        
        # Update password
        password_hash, salt = PasswordHasher.hash_password(new_password)
        user.password_hash = password_hash
        user.salt = salt
        
        # Invalidate all sessions
        self._invalidate_user_sessions(user_id)
        
        self._save()
        return True, "Password changed successfully"
    
    def _validate_password(self, password: str) -> tuple[bool, str]:
        """Validate password against policy."""
        cfg = self._config
        
        if len(password) < cfg.min_password_length:
            return False, f"Password must be at least {cfg.min_password_length} characters"
        
        if cfg.require_uppercase and not any(c.isupper() for c in password):
            return False, "Password must contain uppercase letter"
        
        if cfg.require_lowercase and not any(c.islower() for c in password):
            return False, "Password must contain lowercase letter"
        
        if cfg.require_numbers and not any(c.isdigit() for c in password):
            return False, "Password must contain a number"
        
        if cfg.require_symbols:
            symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in symbols for c in password):
                return False, "Password must contain a symbol"
        
        return True, ""
    
    def _find_user(self, username_or_email: str) -> Optional[User]:
        """Find user by username or email."""
        key = username_or_email.lower()
        
        user_id = self._users_by_username.get(key)
        if not user_id:
            user_id = self._users_by_email.get(key)
        
        if user_id:
            return self._users.get(user_id)
        return None
    
    def _create_session(self, 
                        user: User,
                        ip_address: str,
                        user_agent: str) -> Session:
        """Create a new session."""
        # Enforce max sessions
        self._cleanup_user_sessions(user.id)
        
        session = Session(
            id=TokenGenerator.generate_session_id(),
            user_id=user.id,
            token=TokenGenerator.generate_token(),
            expires_at=time.time() + self._config.session_duration,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self._sessions[session.token] = session
        self._save()
        
        return session
    
    def _get_session_by_token(self, token: str) -> Optional[Session]:
        """Get session by token."""
        return self._sessions.get(token)
    
    def _cleanup_user_sessions(self, user_id: str):
        """Remove old sessions if over limit."""
        user_sessions = [s for s in self._sessions.values() if s.user_id == user_id and s.is_active]
        user_sessions.sort(key=lambda s: s.created_at, reverse=True)
        
        # Keep only max_sessions_per_user - 1 (to make room for new one)
        for session in user_sessions[self._config.max_sessions_per_user - 1:]:
            session.is_active = False
    
    def _invalidate_user_sessions(self, user_id: str):
        """Invalidate all sessions for a user."""
        for session in self._sessions.values():
            if session.user_id == user_id:
                session.is_active = False
    
    def _record_failed_attempt(self, username: str):
        """Record a failed login attempt."""
        if username not in self._failed_attempts:
            self._failed_attempts[username] = []
        self._failed_attempts[username].append(time.time())
    
    def _is_locked_out(self, username: str) -> bool:
        """Check if user is locked out."""
        if username not in self._failed_attempts:
            return False
        
        # Filter recent attempts
        cutoff = time.time() - self._config.lockout_duration
        recent = [t for t in self._failed_attempts[username] if t > cutoff]
        self._failed_attempts[username] = recent
        
        return len(recent) >= self._config.lockout_attempts
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)
    
    def get_all_users(self) -> list[User]:
        """Get all users (admin only)."""
        return list(self._users.values())
    
    def _save(self):
        """Save to storage."""
        if not self._storage_path:
            return
        
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "users": [u.to_dict(include_sensitive=True) for u in self._users.values()],
            "sessions": [s.to_dict() for s in self._sessions.values()]
        }
        
        with open(self._storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        """Load from storage."""
        if not self._storage_path or not self._storage_path.exists():
            return
        
        try:
            with open(self._storage_path) as f:
                data = json.load(f)
            
            for user_data in data.get("users", []):
                user = User.from_dict(user_data)
                self._users[user.id] = user
                self._users_by_username[user.username.lower()] = user.id
                if user.email:
                    self._users_by_email[user.email.lower()] = user.id
            
            # Only load non-expired sessions
            for session_data in data.get("sessions", []):
                if session_data.get("expires_at", 0) > time.time():
                    session = Session(
                        id=session_data["id"],
                        user_id=session_data["user_id"],
                        token=session_data.get("token", ""),
                        created_at=session_data.get("created_at", 0),
                        expires_at=session_data.get("expires_at", 0),
                        ip_address=session_data.get("ip_address", "")
                    )
                    if session.token:
                        self._sessions[session.token] = session
                        
        except Exception as e:
            logger.error(f"Failed to load auth data: {e}")


# Singleton
_auth_manager: Optional[AuthManager] = None


def get_auth_manager(storage_path: Path = None) -> AuthManager:
    """Get the auth manager singleton."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager(storage_path)
    return _auth_manager


__all__ = [
    'AuthManager',
    'User',
    'Session',
    'UserRole',
    'AuthProvider',
    'AuthConfig',
    'PasswordHasher',
    'TokenGenerator',
    'get_auth_manager'
]
