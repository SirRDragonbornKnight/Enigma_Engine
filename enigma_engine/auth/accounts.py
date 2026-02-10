"""
User Accounts System

Multi-user authentication, profiles, and session management.

FILE: enigma_engine/auth/accounts.py
TYPE: Multi-User
MAIN CLASSES: UserManager, Session, AuthProvider
"""

import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False

try:
    import jwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles and permissions."""
    GUEST = "guest"  # Read-only
    USER = "user"  # Standard access
    ADMIN = "admin"  # Full access
    OWNER = "owner"  # System owner


class AuthMethod(Enum):
    """Authentication methods."""
    PASSWORD = "password"
    TOKEN = "token"
    OAUTH = "oauth"
    API_KEY = "api_key"


@dataclass
class UserProfile:
    """User profile data."""
    user_id: str
    username: str
    email: str = ""
    display_name: str = ""
    role: UserRole = UserRole.USER
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    avatar_url: str = ""
    preferences: dict[str, Any] = field(default_factory=dict)
    
    # Usage tracking
    total_messages: int = 0
    total_tokens: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["role"] = self.role.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'UserProfile':
        data = data.copy()
        if "role" in data:
            data["role"] = UserRole(data["role"])
        return cls(**data)


@dataclass
class Session:
    """User session."""
    session_id: str
    user_id: str
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    ip_address: str = ""
    user_agent: str = ""
    is_active: bool = True
    
    def __post_init__(self):
        if self.expires_at is None:
            self.expires_at = time.time() + 86400 * 7  # 7 days
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class APIKey:
    """API key for programmatic access."""
    key_id: str
    user_id: str
    key_hash: str
    name: str
    scopes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    expires_at: Optional[float] = None
    is_active: bool = True


class PasswordHasher:
    """Secure password hashing."""
    
    ITERATIONS = 100000
    ALGORITHM = "sha256"
    
    @classmethod
    def hash_password(cls, password: str, salt: Optional[bytes] = None) -> tuple:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        key = hashlib.pbkdf2_hmac(
            cls.ALGORITHM,
            password.encode("utf-8"),
            salt,
            cls.ITERATIONS
        )
        
        return salt, key
    
    @classmethod
    def verify_password(
        cls,
        password: str,
        salt: bytes,
        stored_hash: bytes
    ) -> bool:
        """Verify password against stored hash."""
        _, computed_hash = cls.hash_password(password, salt)
        return hmac.compare_digest(computed_hash, stored_hash)


class TokenManager:
    """JWT token management."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self._algorithm = "HS256"
    
    def create_token(
        self,
        user_id: str,
        expires_hours: int = 24,
        extra_claims: Optional[dict] = None
    ) -> str:
        """Create JWT token."""
        if not HAS_JWT:
            # Fallback to simple token
            return f"{user_id}:{secrets.token_hex(32)}:{int(time.time() + expires_hours * 3600)}"
        
        payload = {
            "sub": user_id,
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=expires_hours),
            "jti": str(uuid.uuid4())
        }
        
        if extra_claims:
            payload.update(extra_claims)
        
        return jwt.encode(payload, self.secret_key, algorithm=self._algorithm)
    
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode token."""
        if not HAS_JWT:
            # Fallback verification
            try:
                parts = token.split(":")
                if len(parts) == 3:
                    user_id, _, expires = parts
                    if int(expires) > time.time():
                        return {"sub": user_id}
            except (ValueError, IndexError):
                pass
            return None
        
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self._algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
        
        return None


if HAS_SQLITE:
    
    class UserDatabase:
        """SQLite-based user storage."""
        
        def __init__(self, db_path: str):
            self.db_path = db_path
            self._init_db()
        
        def _init_db(self):
            """Initialize database schema."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE,
                        display_name TEXT,
                        role TEXT DEFAULT 'user',
                        created_at REAL,
                        last_login REAL,
                        avatar_url TEXT,
                        preferences TEXT,
                        total_messages INTEGER DEFAULT 0,
                        total_tokens INTEGER DEFAULT 0
                    )
                """)
                
                # Credentials table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS credentials (
                        user_id TEXT PRIMARY KEY,
                        password_salt BLOB,
                        password_hash BLOB,
                        auth_method TEXT,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                """)
                
                # Sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        created_at REAL,
                        expires_at REAL,
                        ip_address TEXT,
                        user_agent TEXT,
                        is_active INTEGER,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                """)
                
                # API keys table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        key_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        key_hash TEXT,
                        name TEXT,
                        scopes TEXT,
                        created_at REAL,
                        last_used REAL,
                        expires_at REAL,
                        is_active INTEGER,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                """)
                
                conn.commit()
        
        def create_user(
            self,
            username: str,
            password: str,
            email: str = "",
            role: UserRole = UserRole.USER
        ) -> Optional[UserProfile]:
            """Create new user."""
            user_id = str(uuid.uuid4())
            salt, hash_val = PasswordHasher.hash_password(password)
            
            profile = UserProfile(
                user_id=user_id,
                username=username,
                email=email,
                display_name=username,
                role=role
            )
            
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT INTO users (
                            user_id, username, email, display_name, role,
                            created_at, avatar_url, preferences,
                            total_messages, total_tokens
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        profile.user_id,
                        profile.username,
                        profile.email,
                        profile.display_name,
                        profile.role.value,
                        profile.created_at,
                        profile.avatar_url,
                        json.dumps(profile.preferences),
                        profile.total_messages,
                        profile.total_tokens
                    ))
                    
                    cursor.execute("""
                        INSERT INTO credentials (user_id, password_salt, password_hash, auth_method)
                        VALUES (?, ?, ?, ?)
                    """, (user_id, salt, hash_val, AuthMethod.PASSWORD.value))
                    
                    conn.commit()
                    
                return profile
            except sqlite3.IntegrityError:
                logger.error(f"User {username} already exists")
                return None
        
        def get_user(self, user_id: str) -> Optional[UserProfile]:
            """Get user by ID."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM users WHERE user_id = ?",
                    (user_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_profile(row)
            return None
        
        def get_user_by_username(self, username: str) -> Optional[UserProfile]:
            """Get user by username."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM users WHERE username = ?",
                    (username,)
                )
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_profile(row)
            return None
        
        def verify_password(
            self,
            user_id: str,
            password: str
        ) -> bool:
            """Verify user password."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT password_salt, password_hash FROM credentials WHERE user_id = ?",
                    (user_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    salt, stored_hash = row
                    return PasswordHasher.verify_password(password, salt, stored_hash)
            return False
        
        def update_user(self, profile: UserProfile):
            """Update user profile."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users SET
                        email = ?,
                        display_name = ?,
                        role = ?,
                        last_login = ?,
                        avatar_url = ?,
                        preferences = ?,
                        total_messages = ?,
                        total_tokens = ?
                    WHERE user_id = ?
                """, (
                    profile.email,
                    profile.display_name,
                    profile.role.value,
                    profile.last_login,
                    profile.avatar_url,
                    json.dumps(profile.preferences),
                    profile.total_messages,
                    profile.total_tokens,
                    profile.user_id
                ))
                conn.commit()
        
        def delete_user(self, user_id: str):
            """Delete user and related data."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM credentials WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM api_keys WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                conn.commit()
        
        def create_session(
            self,
            user_id: str,
            ip_address: str = "",
            user_agent: str = ""
        ) -> Session:
            """Create user session."""
            session = Session(
                session_id=str(uuid.uuid4()),
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sessions (
                        session_id, user_id, created_at, expires_at,
                        ip_address, user_agent, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.user_id,
                    session.created_at,
                    session.expires_at,
                    session.ip_address,
                    session.user_agent,
                    int(session.is_active)
                ))
                conn.commit()
            
            return session
        
        def get_session(self, session_id: str) -> Optional[Session]:
            """Get session by ID."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM sessions WHERE session_id = ?",
                    (session_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return Session(
                        session_id=row[0],
                        user_id=row[1],
                        created_at=row[2],
                        expires_at=row[3],
                        ip_address=row[4],
                        user_agent=row[5],
                        is_active=bool(row[6])
                    )
            return None
        
        def invalidate_session(self, session_id: str):
            """Invalidate session."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE sessions SET is_active = 0 WHERE session_id = ?",
                    (session_id,)
                )
                conn.commit()
        
        def _row_to_profile(self, row) -> UserProfile:
            """Convert database row to UserProfile."""
            return UserProfile(
                user_id=row[0],
                username=row[1],
                email=row[2] or "",
                display_name=row[3] or "",
                role=UserRole(row[4]),
                created_at=row[5],
                last_login=row[6],
                avatar_url=row[7] or "",
                preferences=json.loads(row[8]) if row[8] else {},
                total_messages=row[9],
                total_tokens=row[10]
            )
    
    
    class UserManager:
        """
        Main user management interface.
        """
        
        def __init__(
            self,
            db_path: str = "data/users.db",
            secret_key: Optional[str] = None
        ):
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.db = UserDatabase(db_path)
            self.tokens = TokenManager(secret_key)
            
            self._active_sessions: dict[str, Session] = {}
        
        def register(
            self,
            username: str,
            password: str,
            email: str = "",
            role: UserRole = UserRole.USER
        ) -> Optional[UserProfile]:
            """Register new user."""
            # Validate
            if len(username) < 3:
                raise ValueError("Username too short")
            if len(password) < 8:
                raise ValueError("Password must be at least 8 characters")
            
            return self.db.create_user(username, password, email, role)
        
        def login(
            self,
            username: str,
            password: str,
            ip_address: str = "",
            user_agent: str = ""
        ) -> Optional[tuple]:
            """
            Login user.
            
            Returns:
                (token, session, profile) or None
            """
            profile = self.db.get_user_by_username(username)
            
            if not profile:
                return None
            
            if not self.db.verify_password(profile.user_id, password):
                return None
            
            # Update last login
            profile.last_login = time.time()
            self.db.update_user(profile)
            
            # Create session
            session = self.db.create_session(
                profile.user_id,
                ip_address,
                user_agent
            )
            
            self._active_sessions[session.session_id] = session
            
            # Create token
            token = self.tokens.create_token(
                profile.user_id,
                extra_claims={"role": profile.role.value}
            )
            
            return token, session, profile
        
        def logout(self, session_id: str):
            """Logout user session."""
            self.db.invalidate_session(session_id)
            self._active_sessions.pop(session_id, None)
        
        def authenticate(self, token: str) -> Optional[UserProfile]:
            """Authenticate user by token."""
            payload = self.tokens.verify_token(token)
            
            if payload:
                user_id = payload.get("sub")
                return self.db.get_user(user_id)
            
            return None
        
        def get_user(self, user_id: str) -> Optional[UserProfile]:
            """Get user profile."""
            return self.db.get_user(user_id)
        
        def update_profile(self, profile: UserProfile):
            """Update user profile."""
            self.db.update_user(profile)
        
        def change_password(
            self,
            user_id: str,
            old_password: str,
            new_password: str
        ) -> bool:
            """Change user password."""
            if not self.db.verify_password(user_id, old_password):
                return False
            
            if len(new_password) < 8:
                raise ValueError("Password must be at least 8 characters")
            
            salt, hash_val = PasswordHasher.hash_password(new_password)
            
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE credentials
                    SET password_salt = ?, password_hash = ?
                    WHERE user_id = ?
                """, (salt, hash_val, user_id))
                conn.commit()
            
            return True
        
        def delete_user(self, user_id: str):
            """Delete user account."""
            self.db.delete_user(user_id)
        
        def list_users(
            self,
            limit: int = 100,
            offset: int = 0
        ) -> list[UserProfile]:
            """List all users (admin only)."""
            users = []
            
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM users LIMIT ? OFFSET ?",
                    (limit, offset)
                )
                
                for row in cursor.fetchall():
                    users.append(self.db._row_to_profile(row))
            
            return users
        
        def track_usage(
            self,
            user_id: str,
            messages: int = 0,
            tokens: int = 0
        ):
            """Track user usage."""
            profile = self.db.get_user(user_id)
            if profile:
                profile.total_messages += messages
                profile.total_tokens += tokens
                self.db.update_user(profile)

else:
    class UserDatabase:
        pass
    
    class UserManager:
        pass


def create_user_manager(
    db_path: str = "data/users.db",
    secret_key: Optional[str] = None
) -> 'UserManager':
    """Create user manager instance."""
    if not HAS_SQLITE:
        raise ImportError("SQLite required for user management")
    
    return UserManager(db_path, secret_key)
