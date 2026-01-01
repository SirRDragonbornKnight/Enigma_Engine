#!/usr/bin/env python3
"""
Tests for new features: system messages, text formatting, URL safety, power mode, and autonomous mode.

Run with: python tests/test_new_features.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSystemMessages:
    """Tests for system message utilities."""
    
    def test_import_system_messages(self):
        """Test that system messages module imports correctly."""
        from enigma.utils.system_messages import system_msg, error_msg, warning_msg, info_msg
        assert system_msg is not None
        assert error_msg is not None
        print("✓ System messages import test passed")
    
    def test_system_msg_format(self):
        """Test system message formatting."""
        from enigma.utils.system_messages import system_msg
        result = system_msg("Test message")
        assert "[System]" in result
        assert "Test message" in result
        print("✓ System message format test passed")
    
    def test_error_msg_format(self):
        """Test error message formatting."""
        from enigma.utils.system_messages import error_msg
        result = error_msg("Error occurred")
        assert "[Error]" in result
        assert "Error occurred" in result
        print("✓ Error message format test passed")
    
    def test_warning_msg_format(self):
        """Test warning message formatting."""
        from enigma.utils.system_messages import warning_msg
        result = warning_msg("Warning message")
        assert "[Warning]" in result
        assert "Warning message" in result
        print("✓ Warning message format test passed")
    
    def test_info_msg_format(self):
        """Test info message formatting."""
        from enigma.utils.system_messages import info_msg
        result = info_msg("Information")
        assert "[Info]" in result
        assert "Information" in result
        print("✓ Info message format test passed")


class TestTextFormatting:
    """Tests for text formatting utilities."""
    
    def test_import_text_formatter(self):
        """Test that text formatter imports correctly."""
        from enigma.utils.text_formatting import TextFormatter
        assert TextFormatter is not None
        print("✓ Text formatter import test passed")
    
    def test_bold_formatting(self):
        """Test bold text formatting."""
        from enigma.utils.text_formatting import TextFormatter
        text = "This is **bold** text"
        result = TextFormatter.to_html(text)
        assert "<b>bold</b>" in result
        print("✓ Bold formatting test passed")
    
    def test_italic_formatting(self):
        """Test italic text formatting."""
        from enigma.utils.text_formatting import TextFormatter
        text = "This is *italic* text"
        result = TextFormatter.to_html(text)
        assert "<i>italic</i>" in result
        print("✓ Italic formatting test passed")
    
    def test_underline_formatting(self):
        """Test underline text formatting."""
        from enigma.utils.text_formatting import TextFormatter
        text = "This is __underlined__ text"
        result = TextFormatter.to_html(text)
        assert "<u>underlined</u>" in result
        print("✓ Underline formatting test passed")
    
    def test_code_formatting(self):
        """Test code text formatting."""
        from enigma.utils.text_formatting import TextFormatter
        text = "Use `print()` function"
        result = TextFormatter.to_html(text)
        assert "<code>print()</code>" in result
        print("✓ Code formatting test passed")
    
    def test_strip_formatting(self):
        """Test stripping formatting from text."""
        from enigma.utils.text_formatting import TextFormatter
        text = "This is **bold** and *italic* text"
        result = TextFormatter.strip_formatting(text)
        assert "**" not in result
        assert "*" not in result
        assert "bold" in result
        assert "italic" in result
        print("✓ Strip formatting test passed")


class TestURLSafety:
    """Tests for URL safety utilities."""
    
    def test_import_url_safety(self):
        """Test that URL safety module imports correctly."""
        from enigma.tools.url_safety import URLSafety, ContentFilter
        assert URLSafety is not None
        assert ContentFilter is not None
        print("✓ URL safety import test passed")
    
    def test_safe_url(self):
        """Test that safe URLs pass."""
        from enigma.tools.url_safety import URLSafety
        safety = URLSafety()
        assert safety.is_safe("https://github.com/test")
        assert safety.is_safe("https://python.org")
        print("✓ Safe URL test passed")
    
    def test_blocked_domain(self):
        """Test that blocked domains are caught."""
        from enigma.tools.url_safety import URLSafety
        safety = URLSafety()
        assert not safety.is_safe("https://malware-site.com/test")
        print("✓ Blocked domain test passed")
    
    def test_blocked_pattern(self):
        """Test that blocked patterns are caught."""
        from enigma.tools.url_safety import URLSafety
        safety = URLSafety()
        assert not safety.is_safe("https://example.com/file.exe")
        assert not safety.is_safe("https://example.com/download-crack.zip")
        print("✓ Blocked pattern test passed")
    
    def test_trusted_domain(self):
        """Test trusted domain checking."""
        from enigma.tools.url_safety import URLSafety
        safety = URLSafety()
        assert safety.is_trusted("https://github.com/repo")
        assert safety.is_trusted("https://stackoverflow.com/questions")
        print("✓ Trusted domain test passed")
    
    def test_filter_urls(self):
        """Test filtering a list of URLs."""
        from enigma.tools.url_safety import URLSafety
        safety = URLSafety()
        urls = [
            "https://github.com/test",
            "https://malware-site.com/bad",
            "https://python.org/docs"
        ]
        safe_urls = safety.filter_urls(urls)
        assert len(safe_urls) == 2
        assert "malware-site.com" not in str(safe_urls)
        print("✓ Filter URLs test passed")
    
    def test_content_filter(self):
        """Test content filtering."""
        from enigma.tools.url_safety import ContentFilter
        filter = ContentFilter()
        text = "Good content\nAdvertisement: Click here to win!\nMore good content"
        assert filter.is_ad_content("Click here to win!")
        assert not filter.is_ad_content("Good content")
        print("✓ Content filter test passed")
    
    def test_filter_content(self):
        """Test filtering ad content from text."""
        from enigma.tools.url_safety import ContentFilter
        filter = ContentFilter()
        text = "Good line\nClick here to buy now\nAnother good line"
        filtered = filter.filter_content(text)
        assert "Good line" in filtered
        assert "buy now" not in filtered
        print("✓ Filter content test passed")


class TestPowerMode:
    """Tests for power mode management."""
    
    def test_import_power_mode(self):
        """Test that power mode module imports correctly."""
        try:
            from enigma.core.power_mode import PowerManager, PowerLevel, get_power_manager
            assert PowerManager is not None
            assert PowerLevel is not None
            assert get_power_manager is not None
            print("✓ Power mode import test passed")
        except ImportError as e:
            if "torch" in str(e):
                print("⊘ Power mode test skipped (torch not available)")
            else:
                raise
    
    def test_power_manager_singleton(self):
        """Test that PowerManager is a singleton."""
        try:
            from enigma.core.power_mode import get_power_manager
            pm1 = get_power_manager()
            pm2 = get_power_manager()
            assert pm1 is pm2
            print("✓ Power manager singleton test passed")
        except ImportError:
            print("⊘ Power manager test skipped (dependencies not available)")
    
    def test_power_levels_exist(self):
        """Test that all power levels exist."""
        try:
            from enigma.core.power_mode import PowerLevel
            assert PowerLevel.FULL
            assert PowerLevel.BALANCED
            assert PowerLevel.LOW
            assert PowerLevel.GAMING
            assert PowerLevel.BACKGROUND
            print("✓ Power levels exist test passed")
        except ImportError:
            print("⊘ Power levels test skipped (dependencies not available)")
    
    def test_set_power_level(self):
        """Test setting power level."""
        try:
            from enigma.core.power_mode import get_power_manager, PowerLevel
            pm = get_power_manager()
            pm.set_level(PowerLevel.LOW)
            assert pm.level == PowerLevel.LOW
            print("✓ Set power level test passed")
        except ImportError:
            print("⊘ Set power level test skipped (dependencies not available)")
    
    def test_power_settings(self):
        """Test power settings for different levels."""
        try:
            from enigma.core.power_mode import get_power_manager, PowerLevel
            pm = get_power_manager()
            
            # Test FULL mode
            pm.set_level(PowerLevel.FULL)
            assert pm.settings.max_batch_size == 16
            assert pm.settings.use_gpu == True
            
            # Test GAMING mode
            pm.set_level(PowerLevel.GAMING)
            assert pm.settings.max_batch_size == 1
            assert pm.settings.use_gpu == False
            print("✓ Power settings test passed")
        except ImportError:
            print("⊘ Power settings test skipped (dependencies not available)")
    
    def test_pause_resume(self):
        """Test pausing and resuming."""
        try:
            from enigma.core.power_mode import get_power_manager
            pm = get_power_manager()
            
            assert not pm.is_paused
            pm.pause()
            assert pm.is_paused
            pm.resume()
            assert not pm.is_paused
            print("✓ Pause/resume test passed")
        except ImportError:
            print("⊘ Pause/resume test skipped (dependencies not available)")


class TestAutonomousMode:
    """Tests for autonomous mode."""
    
    def test_import_autonomous(self):
        """Test that autonomous module imports correctly."""
        try:
            from enigma.core.autonomous import AutonomousMode, AutonomousManager
            assert AutonomousMode is not None
            assert AutonomousManager is not None
            print("✓ Autonomous mode import test passed")
        except ImportError as e:
            if "ai_brain" in str(e) or "personality" in str(e):
                print("⊘ Autonomous mode test skipped (optional dependencies not available)")
            else:
                raise
    
    def test_create_autonomous_mode(self):
        """Test creating autonomous mode instance."""
        try:
            from enigma.core.autonomous import AutonomousMode
            am = AutonomousMode("test_model")
            assert am.model_name == "test_model"
            assert am.enabled == False
            print("✓ Create autonomous mode test passed")
        except ImportError:
            print("⊘ Create autonomous mode test skipped (dependencies not available)")
    
    def test_autonomous_manager(self):
        """Test autonomous manager."""
        try:
            from enigma.core.autonomous import AutonomousManager
            am1 = AutonomousManager.get("model1")
            am2 = AutonomousManager.get("model1")
            assert am1 is am2  # Same instance for same model
            
            am3 = AutonomousManager.get("model2")
            assert am3 is not am1  # Different instance for different model
            print("✓ Autonomous manager test passed")
        except ImportError:
            print("⊘ Autonomous manager test skipped (dependencies not available)")
    
    def test_start_stop(self):
        """Test starting and stopping autonomous mode."""
        try:
            from enigma.core.autonomous import AutonomousMode
            am = AutonomousMode("test_model")
            
            assert am.enabled == False
            am.start()
            assert am.enabled == True
            am.stop()
            assert am.enabled == False
            print("✓ Start/stop autonomous mode test passed")
        except ImportError:
            print("⊘ Start/stop autonomous test skipped (dependencies not available)")


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing System Messages")
    print("=" * 60)
    test_sys_msg = TestSystemMessages()
    test_sys_msg.test_import_system_messages()
    test_sys_msg.test_system_msg_format()
    test_sys_msg.test_error_msg_format()
    test_sys_msg.test_warning_msg_format()
    test_sys_msg.test_info_msg_format()
    
    print("\n" + "=" * 60)
    print("Testing Text Formatting")
    print("=" * 60)
    test_text = TestTextFormatting()
    test_text.test_import_text_formatter()
    test_text.test_bold_formatting()
    test_text.test_italic_formatting()
    test_text.test_underline_formatting()
    test_text.test_code_formatting()
    test_text.test_strip_formatting()
    
    print("\n" + "=" * 60)
    print("Testing URL Safety")
    print("=" * 60)
    test_url = TestURLSafety()
    test_url.test_import_url_safety()
    test_url.test_safe_url()
    test_url.test_blocked_domain()
    test_url.test_blocked_pattern()
    test_url.test_trusted_domain()
    test_url.test_filter_urls()
    test_url.test_content_filter()
    test_url.test_filter_content()
    
    print("\n" + "=" * 60)
    print("Testing Power Mode")
    print("=" * 60)
    test_power = TestPowerMode()
    test_power.test_import_power_mode()
    test_power.test_power_manager_singleton()
    test_power.test_power_levels_exist()
    test_power.test_set_power_level()
    test_power.test_power_settings()
    test_power.test_pause_resume()
    
    print("\n" + "=" * 60)
    print("Testing Autonomous Mode")
    print("=" * 60)
    test_auto = TestAutonomousMode()
    test_auto.test_import_autonomous()
    test_auto.test_create_autonomous_mode()
    test_auto.test_autonomous_manager()
    test_auto.test_start_stop()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
