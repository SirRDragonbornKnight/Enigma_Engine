"""
Role-Based Access Control (RBAC)

Manages permissions and access control based on user roles.
Defines what actions each role can perform.

FILE: enigma_engine/security/access_control.py
TYPE: Security System
MAIN CLASSES: AccessControl, Permission, RolePermissions
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions."""
    # Chat permissions
    CHAT_READ = "chat:read"
    CHAT_WRITE = "chat:write"
    CHAT_DELETE = "chat:delete"
    
    # Model permissions
    MODEL_USE = "model:use"
    MODEL_TRAIN = "model:train"
    MODEL_DOWNLOAD = "model:download"
    MODEL_UPLOAD = "model:upload"
    MODEL_DELETE = "model:delete"
    
    # Memory permissions
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    MEMORY_DELETE = "memory:delete"
    MEMORY_EXPORT = "memory:export"
    
    # Tool permissions
    TOOLS_BASIC = "tools:basic"
    TOOLS_FILE = "tools:file"
    TOOLS_WEB = "tools:web"
    TOOLS_CODE = "tools:code"
    TOOLS_SYSTEM = "tools:system"
    
    # Generation permissions
    GEN_TEXT = "gen:text"
    GEN_IMAGE = "gen:image"
    GEN_AUDIO = "gen:audio"
    GEN_VIDEO = "gen:video"
    GEN_CODE = "gen:code"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_ROLES = "admin:roles"
    ADMIN_SETTINGS = "admin:settings"
    ADMIN_LOGS = "admin:logs"
    ADMIN_PLUGINS = "admin:plugins"
    ADMIN_MODULES = "admin:modules"
    
    # API permissions
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"
    
    # System permissions
    SYSTEM_RESTART = "system:restart"
    SYSTEM_UPDATE = "system:update"
    SYSTEM_CONFIGURE = "system:configure"


class Role(Enum):
    """User roles with increasing privilege levels."""
    GUEST = "guest"
    USER = "user"
    POWER_USER = "power_user"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"


# Default permissions for each role
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.GUEST: {
        Permission.CHAT_READ,
        Permission.MODEL_USE,
        Permission.GEN_TEXT,
    },
    
    Role.USER: {
        Permission.CHAT_READ,
        Permission.CHAT_WRITE,
        Permission.MODEL_USE,
        Permission.MEMORY_READ,
        Permission.MEMORY_WRITE,
        Permission.TOOLS_BASIC,
        Permission.GEN_TEXT,
        Permission.GEN_IMAGE,
        Permission.API_READ,
    },
    
    Role.POWER_USER: {
        Permission.CHAT_READ,
        Permission.CHAT_WRITE,
        Permission.CHAT_DELETE,
        Permission.MODEL_USE,
        Permission.MODEL_DOWNLOAD,
        Permission.MEMORY_READ,
        Permission.MEMORY_WRITE,
        Permission.MEMORY_DELETE,
        Permission.MEMORY_EXPORT,
        Permission.TOOLS_BASIC,
        Permission.TOOLS_FILE,
        Permission.TOOLS_WEB,
        Permission.TOOLS_CODE,
        Permission.GEN_TEXT,
        Permission.GEN_IMAGE,
        Permission.GEN_AUDIO,
        Permission.GEN_CODE,
        Permission.API_READ,
        Permission.API_WRITE,
    },
    
    Role.MODERATOR: {
        Permission.CHAT_READ,
        Permission.CHAT_WRITE,
        Permission.CHAT_DELETE,
        Permission.MODEL_USE,
        Permission.MODEL_DOWNLOAD,
        Permission.MEMORY_READ,
        Permission.MEMORY_WRITE,
        Permission.MEMORY_DELETE,
        Permission.MEMORY_EXPORT,
        Permission.TOOLS_BASIC,
        Permission.TOOLS_FILE,
        Permission.TOOLS_WEB,
        Permission.TOOLS_CODE,
        Permission.GEN_TEXT,
        Permission.GEN_IMAGE,
        Permission.GEN_AUDIO,
        Permission.GEN_VIDEO,
        Permission.GEN_CODE,
        Permission.ADMIN_LOGS,
        Permission.API_READ,
        Permission.API_WRITE,
    },
    
    Role.ADMIN: {
        Permission.CHAT_READ,
        Permission.CHAT_WRITE,
        Permission.CHAT_DELETE,
        Permission.MODEL_USE,
        Permission.MODEL_TRAIN,
        Permission.MODEL_DOWNLOAD,
        Permission.MODEL_UPLOAD,
        Permission.MODEL_DELETE,
        Permission.MEMORY_READ,
        Permission.MEMORY_WRITE,
        Permission.MEMORY_DELETE,
        Permission.MEMORY_EXPORT,
        Permission.TOOLS_BASIC,
        Permission.TOOLS_FILE,
        Permission.TOOLS_WEB,
        Permission.TOOLS_CODE,
        Permission.TOOLS_SYSTEM,
        Permission.GEN_TEXT,
        Permission.GEN_IMAGE,
        Permission.GEN_AUDIO,
        Permission.GEN_VIDEO,
        Permission.GEN_CODE,
        Permission.ADMIN_USERS,
        Permission.ADMIN_SETTINGS,
        Permission.ADMIN_LOGS,
        Permission.ADMIN_PLUGINS,
        Permission.ADMIN_MODULES,
        Permission.API_READ,
        Permission.API_WRITE,
        Permission.API_ADMIN,
        Permission.SYSTEM_CONFIGURE,
    },
    
    Role.SUPERADMIN: set(Permission),  # All permissions
}


@dataclass
class ResourcePermission:
    """Permission for a specific resource."""
    resource_type: str
    resource_id: str
    permissions: set[Permission]
    granted_by: str = ""
    granted_at: float = 0.0


@dataclass
class UserPermissions:
    """Complete permissions for a user."""
    user_id: str
    role: Role
    additional_permissions: set[Permission] = field(default_factory=set)
    denied_permissions: set[Permission] = field(default_factory=set)
    resource_permissions: list[ResourcePermission] = field(default_factory=list)
    
    def get_effective_permissions(self) -> set[Permission]:
        """Get all effective permissions."""
        base = ROLE_PERMISSIONS.get(self.role, set()).copy()
        base.update(self.additional_permissions)
        base -= self.denied_permissions
        return base
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in self.get_effective_permissions()
    
    def has_any_permission(self, permissions: list[Permission]) -> bool:
        """Check if user has any of the permissions."""
        effective = self.get_effective_permissions()
        return any(p in effective for p in permissions)
    
    def has_all_permissions(self, permissions: list[Permission]) -> bool:
        """Check if user has all the permissions."""
        effective = self.get_effective_permissions()
        return all(p in effective for p in permissions)


class AccessControl:
    """Manages access control and permissions."""
    
    def __init__(self):
        """Initialize access control."""
        self._user_permissions: dict[str, UserPermissions] = {}
        self._role_permissions = ROLE_PERMISSIONS.copy()
        self._permission_callbacks: dict[Permission, Callable] = {}
    
    def set_user_role(self, user_id: str, role: Role):
        """Set a user's role."""
        if user_id not in self._user_permissions:
            self._user_permissions[user_id] = UserPermissions(
                user_id=user_id,
                role=role
            )
        else:
            self._user_permissions[user_id].role = role
    
    def get_user_permissions(self, user_id: str) -> Optional[UserPermissions]:
        """Get a user's permissions."""
        return self._user_permissions.get(user_id)
    
    def grant_permission(self, user_id: str, permission: Permission):
        """Grant an additional permission to a user."""
        if user_id not in self._user_permissions:
            self._user_permissions[user_id] = UserPermissions(
                user_id=user_id,
                role=Role.USER
            )
        self._user_permissions[user_id].additional_permissions.add(permission)
    
    def revoke_permission(self, user_id: str, permission: Permission):
        """Revoke a permission from a user."""
        if user_id in self._user_permissions:
            perms = self._user_permissions[user_id]
            perms.additional_permissions.discard(permission)
            perms.denied_permissions.add(permission)
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """
        Check if a user has a permission.
        
        Args:
            user_id: User ID
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        user_perms = self._user_permissions.get(user_id)
        if not user_perms:
            # Default to guest permissions
            return permission in ROLE_PERMISSIONS.get(Role.GUEST, set())
        
        return user_perms.has_permission(permission)
    
    def require_permission(self, permission: Permission):
        """
        Decorator to require a permission for a function.
        
        Args:
            permission: Required permission
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, user_id: str = None, **kwargs):
                if user_id is None:
                    raise PermissionError("User ID required")
                
                if not self.check_permission(user_id, permission):
                    raise PermissionError(f"Permission denied: {permission.value}")
                
                return func(*args, user_id=user_id, **kwargs)
            return wrapper
        return decorator
    
    def require_any_permission(self, permissions: list[Permission]):
        """Decorator to require any of the permissions."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, user_id: str = None, **kwargs):
                if user_id is None:
                    raise PermissionError("User ID required")
                
                user_perms = self._user_permissions.get(user_id)
                if not user_perms or not user_perms.has_any_permission(permissions):
                    raise PermissionError(f"Permission denied: requires one of {[p.value for p in permissions]}")
                
                return func(*args, user_id=user_id, **kwargs)
            return wrapper
        return decorator
    
    def get_role_permissions(self, role: Role) -> set[Permission]:
        """Get all permissions for a role."""
        return self._role_permissions.get(role, set()).copy()
    
    def modify_role_permissions(self, 
                                role: Role, 
                                add: list[Permission] = None,
                                remove: list[Permission] = None):
        """
        Modify permissions for a role.
        
        Args:
            role: Role to modify
            add: Permissions to add
            remove: Permissions to remove
        """
        if role not in self._role_permissions:
            self._role_permissions[role] = set()
        
        if add:
            self._role_permissions[role].update(add)
        if remove:
            self._role_permissions[role] -= set(remove)
    
    def get_all_permissions(self) -> list[Permission]:
        """Get all available permissions."""
        return list(Permission)
    
    def get_permissions_by_category(self) -> dict[str, list[Permission]]:
        """Get permissions grouped by category."""
        categories: dict[str, list[Permission]] = {}
        
        for perm in Permission:
            category = perm.value.split(":")[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(perm)
        
        return categories
    
    def can_assign_role(self, 
                        assigner_user_id: str, 
                        target_role: Role) -> bool:
        """
        Check if a user can assign a role to another user.
        
        Args:
            assigner_user_id: User doing the assignment
            target_role: Role being assigned
            
        Returns:
            True if permitted
        """
        # Must have admin:users permission
        if not self.check_permission(assigner_user_id, Permission.ADMIN_USERS):
            return False
        
        # Can't assign superadmin unless you are superadmin
        assigner_perms = self._user_permissions.get(assigner_user_id)
        if target_role == Role.SUPERADMIN:
            return assigner_perms and assigner_perms.role == Role.SUPERADMIN
        
        # Can only assign roles at or below your level
        role_levels = {
            Role.GUEST: 0,
            Role.USER: 1,
            Role.POWER_USER: 2,
            Role.MODERATOR: 3,
            Role.ADMIN: 4,
            Role.SUPERADMIN: 5
        }
        
        if assigner_perms:
            return role_levels.get(target_role, 0) < role_levels.get(assigner_perms.role, 0)
        
        return False


class PermissionError(Exception):
    """Raised when permission is denied."""


# Singleton
_access_control: Optional[AccessControl] = None


def get_access_control() -> AccessControl:
    """Get the access control singleton."""
    global _access_control
    if _access_control is None:
        _access_control = AccessControl()
    return _access_control


def require_permission(permission: Permission):
    """Shortcut decorator for permission requirement."""
    return get_access_control().require_permission(permission)


def check_permission(user_id: str, permission: Permission) -> bool:
    """Shortcut to check permission."""
    return get_access_control().check_permission(user_id, permission)


__all__ = [
    'AccessControl',
    'Permission',
    'Role',
    'UserPermissions',
    'ResourcePermission',
    'PermissionError',
    'ROLE_PERMISSIONS',
    'get_access_control',
    'require_permission',
    'check_permission'
]
