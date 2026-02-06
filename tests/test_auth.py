"""
Tests for authentication and user account systems.

Tests forge_ai/auth/accounts.py and forge_ai/security/authentication.py
"""

import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestPasswordHasher:
    """Tests for password hashing functionality."""
    
    def test_hash_password_returns_salt_and_hash(self):
        """Test that hash_password returns both salt and hash."""
        from forge_ai.auth.accounts import PasswordHasher
        
        salt, key = PasswordHasher.hash_password("test_password")
        
        assert salt is not None
        assert key is not None
        assert len(salt) == 32  # 32 bytes
        assert len(key) > 0
    
    def test_same_password_different_salt(self):
        """Test that same password with different salt gives different hash."""
        from forge_ai.auth.accounts import PasswordHasher
        
        salt1, hash1 = PasswordHasher.hash_password("test_password")
        salt2, hash2 = PasswordHasher.hash_password("test_password")
        
        # Salts should be different (random)
        assert salt1 != salt2
        # Hashes should be different due to different salts
        assert hash1 != hash2
    
    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        from forge_ai.auth.accounts import PasswordHasher
        
        password = "my_secure_password_123!"
        salt, stored_hash = PasswordHasher.hash_password(password)
        
        assert PasswordHasher.verify_password(password, salt, stored_hash)
    
    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        from forge_ai.auth.accounts import PasswordHasher
        
        password = "correct_password"
        wrong_password = "wrong_password"
        salt, stored_hash = PasswordHasher.hash_password(password)
        
        assert not PasswordHasher.verify_password(wrong_password, salt, stored_hash)
    
    def test_empty_password_handling(self):
        """Test handling of empty password."""
        from forge_ai.auth.accounts import PasswordHasher
        
        # Should not raise, even for empty password
        salt, key = PasswordHasher.hash_password("")
        assert salt is not None
        assert key is not None
    
    def test_unicode_password(self):
        """Test handling of unicode characters in password."""
        from forge_ai.auth.accounts import PasswordHasher
        
        password = "密码パスワード🔐"
        salt, key = PasswordHasher.hash_password(password)
        
        assert PasswordHasher.verify_password(password, salt, key)


class TestUserProfile:
    """Tests for UserProfile dataclass."""
    
    def test_create_user_profile(self):
        """Test creating a basic user profile."""
        from forge_ai.auth.accounts import UserProfile, UserRole
        
        profile = UserProfile(
            user_id="user_123",
            username="testuser",
            email="test@example.com"
        )
        
        assert profile.user_id == "user_123"
        assert profile.username == "testuser"
        assert profile.email == "test@example.com"
        assert profile.role == UserRole.USER  # Default
    
    def test_user_profile_to_dict(self):
        """Test converting user profile to dictionary."""
        from forge_ai.auth.accounts import UserProfile, UserRole
        
        profile = UserProfile(
            user_id="user_123",
            username="testuser",
            email="test@example.com",
            role=UserRole.ADMIN
        )
        
        data = profile.to_dict()
        
        assert data["user_id"] == "user_123"
        assert data["username"] == "testuser"
        assert data["role"] == "admin"  # Enum value
    
    def test_user_profile_from_dict(self):
        """Test creating user profile from dictionary."""
        from forge_ai.auth.accounts import UserProfile, UserRole
        
        data = {
            "user_id": "user_456",
            "username": "fromdict",
            "email": "dict@example.com",
            "role": "admin"
        }
        
        profile = UserProfile.from_dict(data)
        
        assert profile.user_id == "user_456"
        assert profile.role == UserRole.ADMIN


class TestSession:
    """Tests for Session management."""
    
    def test_create_session(self):
        """Test creating a session."""
        from forge_ai.auth.accounts import Session
        
        session = Session(
            session_id="sess_123",
            user_id="user_456"
        )
        
        assert session.session_id == "sess_123"
        assert session.user_id == "user_456"
        assert session.is_active
    
    def test_session_expiration(self):
        """Test session expiration logic."""
        from forge_ai.auth.accounts import Session
        
        # Create an already-expired session
        session = Session(
            session_id="sess_expired",
            user_id="user_789",
            expires_at=time.time() - 3600  # Expired 1 hour ago
        )
        
        assert session.is_expired
    
    def test_session_not_expired(self):
        """Test that fresh session is not expired."""
        from forge_ai.auth.accounts import Session
        
        session = Session(
            session_id="sess_fresh",
            user_id="user_abc"
        )
        
        assert not session.is_expired


class TestAPIKey:
    """Tests for API key functionality."""
    
    def test_create_api_key(self):
        """Test creating an API key."""
        from forge_ai.auth.accounts import APIKey
        
        api_key = APIKey(
            key_id="key_123",
            user_id="user_456",
            key_hash="hashed_key_value",
            name="Test API Key",
            scopes=["read", "write"]
        )
        
        assert api_key.key_id == "key_123"
        assert api_key.name == "Test API Key"
        assert "read" in api_key.scopes
        assert api_key.is_active


class TestUserRole:
    """Tests for UserRole enum."""
    
    def test_user_roles_exist(self):
        """Test that expected user roles exist."""
        from forge_ai.auth.accounts import UserRole
        
        assert UserRole.GUEST
        assert UserRole.USER
        assert UserRole.ADMIN
        assert UserRole.OWNER
    
    def test_role_values(self):
        """Test role string values."""
        from forge_ai.auth.accounts import UserRole
        
        assert UserRole.GUEST.value == "guest"
        assert UserRole.USER.value == "user"
        assert UserRole.ADMIN.value == "admin"


class TestSecurityAuthentication:
    """Tests for security/authentication.py."""
    
    def test_auth_provider_enum(self):
        """Test AuthProvider enum values."""
        from forge_ai.security.authentication import AuthProvider
        
        assert AuthProvider.LOCAL.value == "local"
        assert AuthProvider.API_KEY.value == "api_key"
    
    def test_user_to_dict_excludes_sensitive(self):
        """Test that User.to_dict excludes sensitive data by default."""
        from forge_ai.security.authentication import User, UserRole, AuthProvider
        
        user = User(
            id="user_123",
            username="testuser",
            email="test@example.com",
            password_hash="secret_hash",
            salt="secret_salt"
        )
        
        data = user.to_dict(include_sensitive=False)
        
        assert "password_hash" not in data
        assert "salt" not in data
        assert data["username"] == "testuser"
    
    def test_user_to_dict_includes_sensitive_when_requested(self):
        """Test that User.to_dict includes sensitive data when requested."""
        from forge_ai.security.authentication import User
        
        user = User(
            id="user_123",
            username="testuser",
            email="test@example.com",
            password_hash="secret_hash",
            salt="secret_salt"
        )
        
        data = user.to_dict(include_sensitive=True)
        
        assert data["password_hash"] == "secret_hash"
        assert data["salt"] == "secret_salt"
    
    def test_user_from_dict(self):
        """Test creating User from dictionary."""
        from forge_ai.security.authentication import User, UserRole
        
        data = {
            "id": "u_789",
            "username": "dictuser",
            "email": "dict@test.com",
            "role": "admin"
        }
        
        user = User.from_dict(data)
        
        assert user.id == "u_789"
        assert user.role == UserRole.ADMIN


class TestInputSanitization:
    """Tests for input sanitization (if exists)."""
    
    def test_sanitizer_module_exists(self):
        """Test that input sanitizer exists."""
        try:
            from forge_ai.utils.input_sanitizer import sanitize
            # If it exists, test basic functionality
            result = sanitize("<script>alert('xss')</script>")
            assert "<script>" not in result.sanitized
        except ImportError:
            # Module may not exist - that's okay
            pytest.skip("input_sanitizer module not found")
    
    def test_script_tags_removed(self):
        """Test that script tags are sanitized."""
        try:
            from forge_ai.utils.input_sanitizer import sanitize_html
            result = sanitize_html("<p>Hello</p><script>evil()</script>")
            assert "<script>" not in result
            assert "Hello" in result
        except ImportError:
            pytest.skip("sanitize_html not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
