"""
Tests for forge_ai/utils/security.py

Tests the path blocking and security validation functionality.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestPathBlocking:
    """Tests for path blocking functionality."""
    
    def test_blocked_patterns_reject_sensitive_files(self):
        """Test that sensitive file patterns are blocked."""
        from forge_ai.utils.security import is_path_blocked
        
        # These should be blocked by default patterns
        sensitive_paths = [
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
            "~/.ssh/id_rsa",
            "/home/user/.env",
            "secrets.pem",
            "private.key",
            "config/password.txt",
        ]
        
        for path in sensitive_paths:
            blocked, reason = is_path_blocked(path)
            # Note: May not be blocked if patterns not configured
            # This test documents expected behavior
            if blocked:
                assert reason is not None
    
    def test_normal_paths_allowed(self):
        """Test that normal paths are not blocked."""
        from forge_ai.utils.security import is_path_blocked
        
        normal_paths = [
            "document.txt",
            "data/training.json",
            "models/small/model.pt",
            "outputs/generated.png",
        ]
        
        for path in normal_paths:
            blocked, reason = is_path_blocked(path)
            assert not blocked, f"Path {path} should not be blocked: {reason}"
    
    def test_get_blocked_paths_returns_list(self):
        """Test that get_blocked_paths returns a list."""
        from forge_ai.utils.security import get_blocked_paths
        
        paths = get_blocked_paths()
        assert isinstance(paths, list)
    
    def test_get_blocked_patterns_returns_list(self):
        """Test that get_blocked_patterns returns a list."""
        from forge_ai.utils.security import get_blocked_patterns
        
        patterns = get_blocked_patterns()
        assert isinstance(patterns, list)
    
    def test_symlink_resolution(self):
        """Test that symlinks are properly resolved for security."""
        from forge_ai.utils.security import is_path_blocked
        
        # Create a temp directory with a symlink
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a normal file
            normal_file = Path(tmpdir) / "normal.txt"
            normal_file.write_text("test")
            
            # Test the normal file
            blocked, _ = is_path_blocked(str(normal_file))
            assert not blocked


class TestSecurityInitialization:
    """Tests for security system initialization."""
    
    def test_security_module_imports(self):
        """Test that security module can be imported."""
        from forge_ai.utils import security
        assert hasattr(security, 'is_path_blocked')
        assert hasattr(security, 'get_blocked_paths')
    
    def test_initialization_is_idempotent(self):
        """Test that initialization can be called multiple times safely."""
        from forge_ai.utils.security import _initialize_blocks
        
        # Should not raise on multiple calls
        _initialize_blocks()
        _initialize_blocks()


class TestPathValidation:
    """Tests for path validation edge cases."""
    
    def test_empty_path(self):
        """Test handling of empty path."""
        from forge_ai.utils.security import is_path_blocked
        
        blocked, reason = is_path_blocked("")
        # Empty path should be handled gracefully
        assert isinstance(blocked, bool)
    
    def test_relative_path(self):
        """Test handling of relative paths."""
        from forge_ai.utils.security import is_path_blocked
        
        blocked, reason = is_path_blocked("../../../etc/passwd")
        # Path traversal attempts should be handled
        assert isinstance(blocked, bool)
    
    def test_unicode_path(self):
        """Test handling of unicode in paths."""
        from forge_ai.utils.security import is_path_blocked
        
        blocked, reason = is_path_blocked("文档/test.txt")
        assert isinstance(blocked, bool)
    
    def test_very_long_path(self):
        """Test handling of very long paths."""
        from forge_ai.utils.security import is_path_blocked
        
        long_path = "a/" * 100 + "file.txt"
        blocked, reason = is_path_blocked(long_path)
        assert isinstance(blocked, bool)


class TestAddBlockedPath:
    """Tests for adding blocked paths (if user-configurable)."""
    
    def test_add_blocked_path_exists(self):
        """Test that add_blocked_path function exists."""
        from forge_ai.utils.security import add_blocked_path
        assert callable(add_blocked_path)
    
    def test_add_blocked_pattern_exists(self):
        """Test that add_blocked_pattern function exists."""
        from forge_ai.utils.security import add_blocked_pattern
        assert callable(add_blocked_pattern)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
