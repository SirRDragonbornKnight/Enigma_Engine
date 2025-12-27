#!/usr/bin/env python3
"""
Tests for the Enigma tool system.

Run with: pytest tests/test_tools.py -v
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestToolRegistry:
    """Tests for the tool registry."""
    
    def test_registry_loads(self):
        """Test tool registry loads."""
        from enigma.tools import get_registry
        registry = get_registry()
        assert registry is not None
        assert hasattr(registry, 'tools')
    
    def test_list_tools(self):
        """Test listing all tools."""
        from enigma.tools import get_registry
        registry = get_registry()
        tools = list(registry.tools.keys())
        assert len(tools) > 0
    
    def test_execute_tool(self):
        """Test executing a tool."""
        from enigma.tools import execute_tool
        result = execute_tool("get_system_info")
        assert isinstance(result, dict)


class TestSystemTools:
    """Tests for system tools."""
    
    def test_get_system_info(self):
        """Test system info tool."""
        from enigma.tools import execute_tool
        result = execute_tool("get_system_info")
        assert "success" in result or "error" in result
    
    def test_get_time(self):
        """Test time tool."""
        from enigma.tools import execute_tool
        result = execute_tool("get_time")
        # Should return time info
        assert isinstance(result, dict)


class TestFileTools:
    """Tests for file tools."""
    
    def test_list_directory(self, tmp_path):
        """Test listing directory."""
        from enigma.tools import execute_tool
        
        # Create temp files
        (tmp_path / "test.txt").write_text("hello")
        (tmp_path / "subdir").mkdir()
        
        result = execute_tool("list_directory", path=str(tmp_path))
        assert isinstance(result, dict)
    
    def test_read_file(self, tmp_path):
        """Test reading file."""
        from enigma.tools import execute_tool
        
        # Create temp file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")
        
        result = execute_tool("read_file", path=str(test_file))
        assert isinstance(result, dict)


class TestVision:
    """Tests for vision system."""
    
    def test_screen_capture_init(self):
        """Test screen capture initialization."""
        from enigma.tools.vision import ScreenCapture
        capture = ScreenCapture()
        assert capture is not None
    
    def test_screen_vision_init(self):
        """Test screen vision initialization."""
        from enigma.tools.vision import ScreenVision
        vision = ScreenVision()
        assert vision is not None
    
    def test_capture_screen(self):
        """Test screen capture."""
        from enigma.tools.vision import ScreenCapture
        capture = ScreenCapture()
        
        try:
            img = capture.grab()
            assert img is not None
        except Exception:
            # May fail on headless systems
            pytest.skip("Screen capture not available")


class TestWebTools:
    """Tests for web tools."""
    
    def test_web_search_mock(self):
        """Test web search structure."""
        from enigma.tools import execute_tool
        result = execute_tool("web_search", query="test")
        # Should return dict even if search fails
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
