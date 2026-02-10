"""
Role-Based Access Control for Enigma AI Engine

Admin/user permissions for multi-user deployments.

Features:
- Role definitions
- Permission management
- User authentication
- Access control
- Session management

Usage:
    from enigma_engine.utils.rbac import RBACManager
    
    rbac = RBACManager()
    
    # Create roles
    rbac.create_role("admin", permissions=["all"])
    rbac.create_role("user", permissions=["chat", "view_models"])
    
    # Create user
    rbac.create_user("john", password="secret", role="user")
    
    # Check permission
    if rbac.has_permission("john", "train_model"):
        # Allow training
        pass
"""

import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# Built-in permissions
class Permission:
    """Permission constants."""
    ALL = "all"
    
    # Chat
    CHAT = "chat"
    CHAT_HISTORY = "chat_history"
    
    # Models
    VIEW_MODELS = "view_models"
    LOAD_MODEL = "load_model"
    TRAIN_MODEL = "train_model"
    DELETE_MODEL = "delete_model"
    EXPORT_MODEL = "export_model"
    
    # Modules
    VIEW_MODULES = "view_modules"
    TOGGLE_MODULES = "toggle_modules"
    
    # Settings
    VIEW_SETTINGS = "view_settings"
    CHANGE_SETTINGS = "change_settings"
    
    # API
    API_ACCESS = "api_access"
    API_ADMIN = "api_admin"
    
    # Users
    VIEW_USERS = "view_users"
    MANAGE_USERS = "manage_users"
    
    # System
    SYSTEM_INFO = "system_info"
    SYSTEM_ADMIN = "system_admin"
    
    # Data
    VIEW_DATA = "view_data"
    UPLOAD_DATA = "upload_data"
    DELETE_DATA = "delete_data"


@dataclass
class Role:
    """A user role."""
    name: str
    permissions: Set[str]
    description: str = ""
    created: float = field(default_factory=time.time)


@dataclass
class User:
    """A system user."""
    username: str
    password_hash: str
    salt: str
    role: str
    email: str = ""
    created: float = field(default_factory=time.time)
    last_login: float = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """A user session."""
    session_id: str
    username: str
    created: float
    expires: float
    ip_address: str = ""
    user_agent: str = ""


class PasswordManager:
    """Handle password hashing and verification."""
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """
        Hash a password.
        
        Args:
            password: Plain text password
            salt: Optional salt (generated if not provided)
            
        Returns:
            (hash, salt) tuple
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 with SHA256
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        
        return password_hash, salt
    
    @staticmethod
    def verify_password(password: str, password_hash: str, salt: str) -> bool:
        """
        Verify a password.
        
        Args:
            password: Password to verify
            password_hash: Stored hash
            salt: Stored salt
            
        Returns:
            True if password matches
        """
        computed_hash, _ = PasswordManager.hash_password(password, salt)
        return hmac.compare_digest(computed_hash, password_hash)


class RBACManager:
    """Role-Based Access Control manager."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize RBAC manager.
        
        Args:
            storage_path: Path to store RBAC data
        """
        self.storage_path = storage_path or Path("data/rbac")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage
        self._roles: Dict[str, Role] = {}
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Session] = {}
        
        # Password manager
        self._password_manager = PasswordManager()
        
        # Session settings
        self._session_duration = 24 * 60 * 60  # 24 hours
        
        # Load existing data
        self._load_data()
        
        # Create default roles if none exist
        if not self._roles:
            self._create_default_roles()
    
    def _create_default_roles(self):
        """Create default roles."""
        # Admin role - all permissions
        self.create_role("admin", {Permission.ALL}, "Full system access")
        
        # User role - basic permissions
        self.create_role("user", {
            Permission.CHAT,
            Permission.CHAT_HISTORY,
            Permission.VIEW_MODELS,
            Permission.LOAD_MODEL,
            Permission.VIEW_MODULES,
            Permission.VIEW_SETTINGS,
            Permission.API_ACCESS
        }, "Standard user access")
        
        # Guest role - limited permissions
        self.create_role("guest", {
            Permission.CHAT,
            Permission.VIEW_MODELS
        }, "Limited guest access")
        
        # Trainer role - can train models
        self.create_role("trainer", {
            Permission.CHAT,
            Permission.CHAT_HISTORY,
            Permission.VIEW_MODELS,
            Permission.LOAD_MODEL,
            Permission.TRAIN_MODEL,
            Permission.VIEW_MODULES,
            Permission.TOGGLE_MODULES,
            Permission.VIEW_DATA,
            Permission.UPLOAD_DATA
        }, "Model training access")
    
    def create_role(
        self,
        name: str,
        permissions: Set[str],
        description: str = ""
    ) -> Role:
        """
        Create a new role.
        
        Args:
            name: Role name
            permissions: Set of permission strings
            description: Role description
            
        Returns:
            Created role
        """
        role = Role(
            name=name,
            permissions=permissions,
            description=description
        )
        
        self._roles[name] = role
        self._save_data()
        
        logger.info(f"Created role: {name}")
        return role
    
    def delete_role(self, name: str):
        """Delete a role."""
        if name in self._roles:
            del self._roles[name]
            self._save_data()
    
    def get_role(self, name: str) -> Optional[Role]:
        """Get a role by name."""
        return self._roles.get(name)
    
    def list_roles(self) -> List[Role]:
        """List all roles."""
        return list(self._roles.values())
    
    def create_user(
        self,
        username: str,
        password: str,
        role: str,
        email: str = "",
        metadata: Optional[Dict] = None
    ) -> User:
        """
        Create a new user.
        
        Args:
            username: Username
            password: Plain text password
            role: Role name
            email: User email
            metadata: Additional metadata
            
        Returns:
            Created user
        """
        if username in self._users:
            raise ValueError(f"User '{username}' already exists")
        
        if role not in self._roles:
            raise ValueError(f"Role '{role}' does not exist")
        
        # Hash password
        password_hash, salt = self._password_manager.hash_password(password)
        
        user = User(
            username=username,
            password_hash=password_hash,
            salt=salt,
            role=role,
            email=email,
            metadata=metadata or {}
        )
        
        self._users[username] = user
        self._save_data()
        
        logger.info(f"Created user: {username} with role: {role}")
        return user
    
    def delete_user(self, username: str):
        """Delete a user."""
        if username in self._users:
            del self._users[username]
            
            # Also delete user's sessions
            sessions_to_delete = [
                sid for sid, s in self._sessions.items()
                if s.username == username
            ]
            for sid in sessions_to_delete:
                del self._sessions[sid]
            
            self._save_data()
    
    def get_user(self, username: str) -> Optional[User]:
        """Get a user by username."""
        return self._users.get(username)
    
    def list_users(self) -> List[User]:
        """List all users."""
        return list(self._users.values())
    
    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: str = "",
        user_agent: str = ""
    ) -> Optional[Session]:
        """
        Authenticate a user.
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP
            user_agent: Client user agent
            
        Returns:
            Session if successful, None otherwise
        """
        user = self._users.get(username)
        
        if not user or not user.enabled:
            logger.warning(f"Authentication failed: user not found or disabled: {username}")
            return None
        
        if not self._password_manager.verify_password(
            password, user.password_hash, user.salt
        ):
            logger.warning(f"Authentication failed: invalid password for {username}")
            return None
        
        # Create session
        session = Session(
            session_id=secrets.token_hex(32),
            username=username,
            created=time.time(),
            expires=time.time() + self._session_duration,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self._sessions[session.session_id] = session
        
        # Update last login
        user.last_login = time.time()
        self._save_data()
        
        logger.info(f"User authenticated: {username}")
        return session
    
    def validate_session(self, session_id: str) -> Optional[Session]:
        """
        Validate a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session if valid, None otherwise
        """
        session = self._sessions.get(session_id)
        
        if not session:
            return None
        
        if time.time() > session.expires:
            del self._sessions[session_id]
            return None
        
        return session
    
    def invalidate_session(self, session_id: str):
        """Invalidate a session (logout)."""
        if session_id in self._sessions:
            del self._sessions[session_id]
    
    def has_permission(
        self,
        username: str,
        permission: str,
        session_id: Optional[str] = None
    ) -> bool:
        """
        Check if user has a permission.
        
        Args:
            username: Username
            permission: Permission to check
            session_id: Optional session ID for validation
            
        Returns:
            True if permission granted
        """
        # Validate session if provided
        if session_id:
            session = self.validate_session(session_id)
            if not session or session.username != username:
                return False
        
        user = self._users.get(username)
        if not user or not user.enabled:
            return False
        
        role = self._roles.get(user.role)
        if not role:
            return False
        
        # Check for 'all' permission
        if Permission.ALL in role.permissions:
            return True
        
        return permission in role.permissions
    
    def get_user_permissions(self, username: str) -> Set[str]:
        """Get all permissions for a user."""
        user = self._users.get(username)
        if not user:
            return set()
        
        role = self._roles.get(user.role)
        if not role:
            return set()
        
        return role.permissions
    
    def change_password(
        self,
        username: str,
        old_password: str,
        new_password: str
    ) -> bool:
        """
        Change user password.
        
        Args:
            username: Username
            old_password: Current password
            new_password: New password
            
        Returns:
            True if successful
        """
        user = self._users.get(username)
        if not user:
            return False
        
        if not self._password_manager.verify_password(
            old_password, user.password_hash, user.salt
        ):
            return False
        
        # Set new password
        password_hash, salt = self._password_manager.hash_password(new_password)
        user.password_hash = password_hash
        user.salt = salt
        
        self._save_data()
        return True
    
    def change_role(self, username: str, new_role: str) -> bool:
        """Change a user's role."""
        user = self._users.get(username)
        if not user:
            return False
        
        if new_role not in self._roles:
            return False
        
        user.role = new_role
        self._save_data()
        return True
    
    def _save_data(self):
        """Save RBAC data to disk."""
        # Save roles
        roles_data = {
            name: {
                "permissions": list(role.permissions),
                "description": role.description,
                "created": role.created
            }
            for name, role in self._roles.items()
        }
        
        (self.storage_path / "roles.json").write_text(
            json.dumps(roles_data, indent=2)
        )
        
        # Save users (without sessions)
        users_data = {
            username: {
                "password_hash": user.password_hash,
                "salt": user.salt,
                "role": user.role,
                "email": user.email,
                "created": user.created,
                "last_login": user.last_login,
                "enabled": user.enabled,
                "metadata": user.metadata
            }
            for username, user in self._users.items()
        }
        
        (self.storage_path / "users.json").write_text(
            json.dumps(users_data, indent=2)
        )
    
    def _load_data(self):
        """Load RBAC data from disk."""
        # Load roles
        roles_file = self.storage_path / "roles.json"
        if roles_file.exists():
            try:
                roles_data = json.loads(roles_file.read_text())
                for name, data in roles_data.items():
                    self._roles[name] = Role(
                        name=name,
                        permissions=set(data.get("permissions", [])),
                        description=data.get("description", ""),
                        created=data.get("created", time.time())
                    )
            except Exception as e:
                logger.error(f"Error loading roles: {e}")
        
        # Load users
        users_file = self.storage_path / "users.json"
        if users_file.exists():
            try:
                users_data = json.loads(users_file.read_text())
                for username, data in users_data.items():
                    self._users[username] = User(
                        username=username,
                        password_hash=data["password_hash"],
                        salt=data["salt"],
                        role=data["role"],
                        email=data.get("email", ""),
                        created=data.get("created", time.time()),
                        last_login=data.get("last_login", 0),
                        enabled=data.get("enabled", True),
                        metadata=data.get("metadata", {})
                    )
            except Exception as e:
                logger.error(f"Error loading users: {e}")


# Decorator for permission checking
def require_permission(permission: str):
    """Decorator to require a permission."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Get username from kwargs or first arg
            username = kwargs.get('username') or (args[0] if args else None)
            session_id = kwargs.get('session_id')
            
            rbac = get_rbac_manager()
            
            if not rbac.has_permission(username, permission, session_id):
                raise PermissionError(f"Permission denied: {permission}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global instance
_rbac_manager: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    """Get or create global RBAC manager."""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager()
    return _rbac_manager
