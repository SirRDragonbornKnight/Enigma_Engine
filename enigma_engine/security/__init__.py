"""
Enigma AI Engine Security Module

Authentication, authorization, and access control for Enigma AI Engine.
"""

from .access_control import (
    ROLE_PERMISSIONS,
    AccessControl,
    Permission,
    PermissionError,
    ResourcePermission,
    Role,
    UserPermissions,
    check_permission,
    get_access_control,
    require_permission,
)
from .authentication import (
    AuthConfig,
    AuthManager,
    AuthProvider,
    PasswordHasher,
    Session,
    TokenGenerator,
    User,
    UserRole,
    get_auth_manager,
)

__all__ = [
    # Authentication
    'AuthManager',
    'User',
    'Session',
    'UserRole',
    'AuthProvider',
    'AuthConfig',
    'PasswordHasher',
    'TokenGenerator',
    'get_auth_manager',
    # Access Control
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
