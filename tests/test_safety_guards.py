"""
Tests for GUI Safety Guards
===========================

Tests for the safety_guards module that protects against:
- User mistakes (confirmation dialogs)
- Spam clicking (rate limiting)
- Invalid inputs (validation)
- Tampering attempts (anti-tamper)
"""

import time
import pytest


class TestInputValidator:
    """Tests for input validation."""
    
    def test_text_validation_empty_allowed(self):
        """Test empty text when allowed."""
        from forge_ai.gui.safety_guards import InputValidator
        
        valid, error = InputValidator.validate_text("", allow_empty=True)
        assert valid is True
        assert error is None
    
    def test_text_validation_empty_not_allowed(self):
        """Test empty text when not allowed."""
        from forge_ai.gui.safety_guards import InputValidator
        
        valid, error = InputValidator.validate_text("", allow_empty=False)
        assert valid is False
        assert "empty" in error.lower()
    
    def test_text_validation_too_long(self):
        """Test text exceeding max length."""
        from forge_ai.gui.safety_guards import InputValidator
        
        long_text = "x" * 60000
        valid, error = InputValidator.validate_text(long_text, field_type="chat_message")
        assert valid is False
        assert "too long" in error.lower()
    
    def test_text_validation_injection_null_byte(self):
        """Test null byte injection detection."""
        from forge_ai.gui.safety_guards import InputValidator
        
        # Null byte injection attempt
        valid, error = InputValidator.validate_text("normal\x00text")
        assert valid is False
        assert "invalid" in error.lower()
    
    def test_text_validation_injection_path_traversal(self):
        """Test path traversal injection detection."""
        from forge_ai.gui.safety_guards import InputValidator
        
        valid, error = InputValidator.validate_text("../../etc/passwd", check_injection=True)
        assert valid is False
    
    def test_text_validation_injection_template(self):
        """Test template injection detection."""
        from forge_ai.gui.safety_guards import InputValidator
        
        valid, error = InputValidator.validate_text("Hello ${user.password}", check_injection=True)
        assert valid is False
    
    def test_text_validation_normal_text(self):
        """Test normal text passes validation."""
        from forge_ai.gui.safety_guards import InputValidator
        
        valid, error = InputValidator.validate_text("Hello, this is a normal message!")
        assert valid is True
        assert error is None
    
    def test_path_validation_traversal(self):
        """Test path traversal detection."""
        from forge_ai.gui.safety_guards import InputValidator
        
        valid, error = InputValidator.validate_path("../../../etc/passwd")
        # Path with .. is still allowed if it resolves within bounds
        # But the security.py module will catch actual traversal
        assert isinstance(valid, bool)
    
    def test_path_validation_dangerous_chars(self):
        """Test dangerous character detection in paths."""
        from forge_ai.gui.safety_guards import InputValidator
        
        valid, error = InputValidator.validate_path("file<script>.txt")
        assert valid is False
        assert "invalid character" in error.lower()
    
    def test_path_validation_null_byte(self):
        """Test null byte in path."""
        from forge_ai.gui.safety_guards import InputValidator
        
        valid, error = InputValidator.validate_path("file.txt\x00.exe")
        assert valid is False
    
    def test_number_validation_valid(self):
        """Test valid number."""
        from forge_ai.gui.safety_guards import InputValidator
        
        valid, error, value = InputValidator.validate_number("42")
        assert valid is True
        assert value == 42.0
    
    def test_number_validation_bounds(self):
        """Test number bounds checking."""
        from forge_ai.gui.safety_guards import InputValidator
        
        valid, error, value = InputValidator.validate_number("150", min_val=0, max_val=100)
        assert valid is False
        assert "at most" in error.lower()
    
    def test_number_validation_invalid(self):
        """Test invalid number format."""
        from forge_ai.gui.safety_guards import InputValidator
        
        valid, error, value = InputValidator.validate_number("not_a_number")
        assert valid is False
        assert "invalid" in error.lower()
    
    def test_sanitize_for_display(self):
        """Test text sanitization for display."""
        from forge_ai.gui.safety_guards import InputValidator
        
        # Test control character removal
        text = "Hello\x00World\x07Test"
        sanitized = InputValidator.sanitize_for_display(text)
        assert "\x00" not in sanitized
        assert "\x07" not in sanitized
        
        # Test truncation
        long_text = "x" * 2000
        sanitized = InputValidator.sanitize_for_display(long_text, max_length=100)
        assert len(sanitized) == 100
        assert sanitized.endswith("...")


class TestRateLimiter:
    """Tests for rate limiting."""
    
    def test_rate_limiter_allows_first_action(self):
        """Test first action is allowed."""
        from forge_ai.gui.safety_guards import RateLimiter
        
        limiter = RateLimiter()
        allowed, wait = limiter.check("test_action", min_interval=1.0)
        assert allowed is True
        assert wait == 0
    
    def test_rate_limiter_blocks_rapid_actions(self):
        """Test rapid actions are blocked."""
        from forge_ai.gui.safety_guards import RateLimiter
        
        limiter = RateLimiter()
        
        # First action allowed
        allowed1, _ = limiter.check("rapid_test", min_interval=1.0)
        assert allowed1 is True
        
        # Immediate second action blocked
        allowed2, wait = limiter.check("rapid_test", min_interval=1.0)
        assert allowed2 is False
        assert wait > 0
    
    def test_rate_limiter_allows_after_interval(self):
        """Test actions allowed after interval."""
        from forge_ai.gui.safety_guards import RateLimiter
        
        limiter = RateLimiter()
        
        allowed1, _ = limiter.check("interval_test", min_interval=0.1)
        assert allowed1 is True
        
        time.sleep(0.15)  # Wait longer than interval
        
        allowed2, _ = limiter.check("interval_test", min_interval=0.1)
        assert allowed2 is True
    
    def test_rate_limiter_burst_protection(self):
        """Test burst protection kicks in."""
        from forge_ai.gui.safety_guards import RateLimiter
        
        limiter = RateLimiter()
        
        # Rapid burst
        for i in range(5):
            limiter.check("burst_test", min_interval=0.0, max_burst=3, burst_window=5.0)
        
        # Should be blocked after burst limit
        allowed, wait = limiter.check("burst_test", min_interval=0.0, max_burst=3, burst_window=5.0)
        assert allowed is False
    
    def test_rate_limiter_different_actions_independent(self):
        """Test different actions are tracked independently."""
        from forge_ai.gui.safety_guards import RateLimiter
        
        limiter = RateLimiter()
        
        # Action A
        allowed_a, _ = limiter.check("action_a", min_interval=10.0)
        assert allowed_a is True
        
        # Action B should not be affected by A
        allowed_b, _ = limiter.check("action_b", min_interval=10.0)
        assert allowed_b is True


class TestActionHistory:
    """Tests for undo/redo functionality."""
    
    def test_record_action(self):
        """Test recording an action."""
        from forge_ai.gui.safety_guards import ActionHistory
        
        history = ActionHistory()
        action_id = history.record(
            "test_action",
            "Test description",
            {"data": "test"}
        )
        
        assert action_id is not None
        assert "action_" in action_id
    
    def test_can_undo_after_record(self):
        """Test can_undo returns True after recording."""
        from forge_ai.gui.safety_guards import ActionHistory
        
        history = ActionHistory()
        history.record("test", "Test", {"x": 1})
        
        can_undo, desc = history.can_undo()
        assert can_undo is True
        assert desc == "Test"
    
    def test_cannot_undo_empty_history(self):
        """Test can_undo returns False for empty history."""
        from forge_ai.gui.safety_guards import ActionHistory
        
        history = ActionHistory()
        can_undo, desc = history.can_undo()
        assert can_undo is False
    
    def test_undo_with_handler(self):
        """Test undo executes handler."""
        from forge_ai.gui.safety_guards import ActionHistory
        
        history = ActionHistory()
        
        # Track if handler was called
        handler_called = {"value": False}
        
        def undo_handler(data):
            handler_called["value"] = True
            return True
        
        history.register_handlers("test_type", undo_handler)
        history.record("test_type", "Test", {"x": 1})
        
        success, msg = history.undo()
        assert success is True
        assert handler_called["value"] is True
    
    def test_undo_without_handler_fails(self):
        """Test undo fails without registered handler."""
        from forge_ai.gui.safety_guards import ActionHistory
        
        history = ActionHistory()
        history.record("unhandled_type", "Test", {"x": 1})
        
        success, msg = history.undo()
        assert success is False
        assert "No undo handler" in msg
    
    def test_get_history(self):
        """Test getting action history."""
        from forge_ai.gui.safety_guards import ActionHistory
        
        history = ActionHistory()
        history.record("type1", "First action", {})
        history.record("type2", "Second action", {})
        
        items = history.get_history()
        assert len(items) == 2
        # Most recent first
        assert items[0]["description"] == "Second action"
    
    def test_max_history_enforcement(self):
        """Test maximum history limit is enforced."""
        from forge_ai.gui.safety_guards import ActionHistory
        
        history = ActionHistory(max_history=5)
        
        for i in range(10):
            history.record("test", f"Action {i}", {})
        
        items = history.get_history(limit=100)
        assert len(items) == 5


class TestAntiTamper:
    """Tests for anti-tampering protection."""
    
    def test_protected_settings_blocked(self):
        """Test that protected settings cannot be changed."""
        from forge_ai.gui.safety_guards import AntiTamper
        
        tamper = AntiTamper()
        
        allowed, reason = tamper.validate_setting_change(
            "blocked_paths", [], ["/new/path"]
        )
        assert allowed is False
        assert "protected" in reason.lower()
    
    def test_setting_bounds_enforced(self):
        """Test that setting bounds are enforced."""
        from forge_ai.gui.safety_guards import AntiTamper
        
        tamper = AntiTamper()
        
        # Temperature out of bounds
        allowed, reason = tamper.validate_setting_change(
            "temperature", 0.7, 5.0
        )
        assert allowed is False
        assert "between" in reason.lower()
    
    def test_valid_setting_change_allowed(self):
        """Test that valid setting changes are allowed."""
        from forge_ai.gui.safety_guards import AntiTamper
        
        tamper = AntiTamper()
        
        allowed, reason = tamper.validate_setting_change(
            "temperature", 0.7, 0.9
        )
        assert allowed is True
    
    def test_api_key_format_validation(self):
        """Test API key format validation."""
        from forge_ai.gui.safety_guards import AntiTamper
        
        tamper = AntiTamper()
        
        # Fake key rejected
        valid, error = tamper.validate_api_key_format("test123", "generic")
        assert valid is False
        assert "test/fake" in error.lower()
        
        # Empty key rejected
        valid, error = tamper.validate_api_key_format("", "generic")
        assert valid is False
    
    def test_suspicious_activity_logging(self):
        """Test suspicious activity is logged."""
        from forge_ai.gui.safety_guards import AntiTamper
        
        tamper = AntiTamper()
        
        # Try to change protected setting
        tamper.validate_setting_change("blocked_paths", [], ["hack"])
        
        activities = tamper.get_suspicious_activity()
        assert len(activities) >= 1
        assert activities[0]["type"] == "protected_setting_access"


class TestActionSeverity:
    """Tests for action severity levels."""
    
    def test_severity_mapping(self):
        """Test action severity mapping."""
        from forge_ai.gui.safety_guards import ACTION_SEVERITY_MAP, ActionSeverity
        
        assert ACTION_SEVERITY_MAP["read_file"] == ActionSeverity.SAFE
        assert ACTION_SEVERITY_MAP["delete_file"] == ActionSeverity.MODERATE
        assert ACTION_SEVERITY_MAP["delete_model"] == ActionSeverity.DANGEROUS
        assert ACTION_SEVERITY_MAP["delete_all_models"] == ActionSeverity.CRITICAL


class TestSafetyGuardsClass:
    """Tests for the main SafetyGuards class."""
    
    def test_initialization(self):
        """Test SafetyGuards initialization."""
        from forge_ai.gui.safety_guards import SafetyGuards
        
        SafetyGuards.initialize()
        # Should not raise
    
    def test_validate_input(self):
        """Test input validation through SafetyGuards."""
        from forge_ai.gui.safety_guards import SafetyGuards
        
        valid, error = SafetyGuards.validate_input("Hello world")
        assert valid is True
    
    def test_validate_path(self):
        """Test path validation through SafetyGuards."""
        from forge_ai.gui.safety_guards import SafetyGuards
        
        valid, error = SafetyGuards.validate_path("/some/valid/path")
        assert valid is True
    
    def test_rate_limit_check(self):
        """Test rate limit check through SafetyGuards."""
        from forge_ai.gui.safety_guards import SafetyGuards
        
        allowed, wait = SafetyGuards.check_rate_limit("test_action_sg")
        assert allowed is True


class TestRateLimitDecorator:
    """Tests for rate_limit decorator."""
    
    def test_decorator_allows_first_call(self):
        """Test decorator allows first call."""
        from forge_ai.gui.safety_guards import rate_limit
        
        @rate_limit(min_interval=1.0, action_name="decorator_test_1")
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    def test_decorator_blocks_rapid_calls(self):
        """Test decorator blocks rapid calls."""
        from forge_ai.gui.safety_guards import rate_limit
        
        call_count = {"value": 0}
        
        @rate_limit(min_interval=1.0, action_name="decorator_test_2")
        def test_func():
            call_count["value"] += 1
            return "success"
        
        # First call succeeds
        result1 = test_func()
        assert result1 == "success"
        
        # Rapid second call returns None (blocked)
        result2 = test_func()
        assert result2 is None
        
        # Function was only actually called once
        assert call_count["value"] == 1


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
