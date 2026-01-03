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
        # Check for expected built-in tools
        expected_tools = ["web_search", "read_file", "write_file", "get_system_info"]
        for tool_name in expected_tools:
            assert tool_name in tools, f"Missing expected tool: {tool_name}"
    
    def test_get_tool_by_name(self):
        """Test getting a tool by name."""
        from enigma.tools import get_registry
        registry = get_registry()
        tool = registry.get("get_system_info")
        assert tool is not None
        assert tool.name == "get_system_info"
    
    def test_tool_not_found(self):
        """Test executing a non-existent tool."""
        from enigma.tools import execute_tool
        result = execute_tool("nonexistent_tool_xyz")
        assert result["success"] is False
        assert "not found" in result.get("error", "").lower()
    
    def test_execute_tool(self):
        """Test executing a tool."""
        from enigma.tools import execute_tool
        result = execute_tool("get_system_info")
        assert isinstance(result, dict)
        assert "success" in result
    
    def test_tool_to_dict(self):
        """Test tool export to dict for AI consumption."""
        from enigma.tools import get_registry
        registry = get_registry()
        tool = registry.get("get_system_info")
        tool_dict = tool.to_dict()
        assert "name" in tool_dict
        assert "description" in tool_dict
        assert "parameters" in tool_dict


class TestSystemTools:
    """Tests for system tools."""
    
    def test_get_system_info(self):
        """Test system info tool returns expected data."""
        from enigma.tools import execute_tool
        result = execute_tool("get_system_info")
        assert result.get("success") is True
        info = result.get("info", {})
        # Check for expected system info fields
        assert "os" in info
        assert "python_version" in info
        assert "cpu_count" in info
    
    def test_run_command_safe(self):
        """Test running a safe command."""
        from enigma.tools import execute_tool
        result = execute_tool("run_command", command="echo hello")
        assert result.get("success") is True
        assert "hello" in result.get("stdout", "")
    
    def test_run_command_blocked(self):
        """Test that dangerous commands are blocked."""
        from enigma.tools import execute_tool
        result = execute_tool("run_command", command="rm -rf /")
        assert result.get("success") is False
        assert "blocked" in result.get("error", "").lower()


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
        assert result.get("success") is True
    
    def test_read_file(self, tmp_path):
        """Test reading file content."""
        from enigma.tools import execute_tool
        
        # Create temp file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")
        
        result = execute_tool("read_file", path=str(test_file))
        assert isinstance(result, dict)
        assert result.get("success") is True
        assert result.get("content") == "Hello World"
    
    def test_read_file_not_found(self, tmp_path):
        """Test reading non-existent file."""
        from enigma.tools import execute_tool
        
        result = execute_tool("read_file", path=str(tmp_path / "nonexistent.txt"))
        assert result.get("success") is False
        assert "not found" in result.get("error", "").lower()
    
    def test_write_file(self, tmp_path):
        """Test writing file content."""
        from enigma.tools import execute_tool
        
        test_file = tmp_path / "output.txt"
        result = execute_tool("write_file", path=str(test_file), content="Test content")
        assert result.get("success") is True
        assert test_file.read_text() == "Test content"
    
    def test_write_file_append(self, tmp_path):
        """Test appending to file."""
        from enigma.tools import execute_tool
        
        test_file = tmp_path / "append.txt"
        test_file.write_text("Line 1\n")
        
        result = execute_tool("write_file", path=str(test_file), content="Line 2", mode="append")
        assert result.get("success") is True
        assert "Line 1" in test_file.read_text()
        assert "Line 2" in test_file.read_text()
    
    def test_move_file(self, tmp_path):
        """Test moving/renaming a file."""
        from enigma.tools import execute_tool
        
        src = tmp_path / "source.txt"
        dst = tmp_path / "destination.txt"
        src.write_text("content")
        
        result = execute_tool("move_file", source=str(src), destination=str(dst))
        assert result.get("success") is True
        assert dst.exists()
        assert not src.exists()
    
    def test_delete_file(self, tmp_path):
        """Test deleting a file."""
        from enigma.tools import execute_tool
        
        test_file = tmp_path / "to_delete.txt"
        test_file.write_text("delete me")
        
        result = execute_tool("delete_file", path=str(test_file), confirm="yes")
        assert result.get("success") is True
        assert not test_file.exists()


class TestVision:
    """Tests for vision system."""
    
    def test_screen_capture_init(self):
        """Test screen capture initialization."""
        from enigma.tools.vision import ScreenCapture
        capture = ScreenCapture()
        assert capture is not None
        assert hasattr(capture, '_backend')
    
    def test_screen_vision_init(self):
        """Test screen vision initialization."""
        from enigma.tools.vision import ScreenVision
        vision = ScreenVision()
        assert vision is not None
    
    def test_capture_screen(self):
        """Test screen capture on systems with display."""
        from enigma.tools.vision import ScreenCapture
        capture = ScreenCapture()
        
        if capture._backend == "none":
            pytest.skip("No screenshot backend available")
        
        try:
            img = capture.capture()
            # May return None on headless systems
            if img is not None:
                assert hasattr(img, 'width')
                assert hasattr(img, 'height')
        except Exception:
            pytest.skip("Screen capture not available on this system")


class TestWebTools:
    """Tests for web tools."""
    
    def test_web_search_structure(self):
        """Test web search returns proper structure."""
        from enigma.tools import execute_tool
        result = execute_tool("web_search", query="test")
        # Should return dict even if search fails
        assert isinstance(result, dict)
        assert "success" in result
    
    def test_web_search_empty_query(self):
        """Test web search with empty query."""
        from enigma.tools import execute_tool
        result = execute_tool("web_search", query="")
        assert result.get("success") is False
        assert "empty" in result.get("error", "").lower()


class TestToolDefinitions:
    """Tests for tool definitions (AI tool use system)."""
    
    def test_get_all_tools(self):
        """Test getting all tool definitions."""
        from enigma.tools import get_all_tools
        tools = get_all_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0
    
    def test_get_tools_by_category(self):
        """Test filtering tools by category."""
        from enigma.tools import get_tools_by_category
        gen_tools = get_tools_by_category("generation")
        assert isinstance(gen_tools, list)
    
    def test_get_available_tools_for_prompt(self):
        """Test getting tool descriptions for AI prompt."""
        from enigma.tools import get_available_tools_for_prompt
        prompt_text = get_available_tools_for_prompt()
        assert isinstance(prompt_text, str)
        assert len(prompt_text) > 0
    
    def test_tool_definition_schema(self):
        """Test tool definition exports proper schema."""
        from enigma.tools import get_tool_definition
        tool_def = get_tool_definition("generate_image")
        if tool_def:
            schema = tool_def.get_schema()
            assert "generate_image" in schema
            assert "prompt" in schema.lower()


class TestToolExecutor:
    """Tests for the AI tool executor."""
    
    def test_executor_init(self):
        """Test tool executor initialization."""
        from enigma.tools import ToolExecutor
        executor = ToolExecutor()
        assert executor is not None
    
    def test_execute_tool_from_text(self):
        """Test parsing and executing tool calls from text."""
        from enigma.tools import execute_tool_from_text
        # This tests the text parsing capability
        text = '<tool>get_system_info</tool>'
        modified_text, results = execute_tool_from_text(text)
        assert isinstance(results, list)
        assert isinstance(modified_text, str)


class TestInteractiveTools:
    """Tests for interactive/personal assistant tools."""
    
    def test_create_checklist(self, tmp_path):
        """Test creating a checklist."""
        from enigma.tools.interactive_tools import ChecklistManager
        manager = ChecklistManager(storage_path=tmp_path / "checklists.json")
        result = manager.create_checklist("Test List", ["Item 1", "Item 2"])
        assert result["success"] is True
        assert result["name"] == "Test List"
        assert result["items"] == 2
    
    def test_list_checklists(self, tmp_path):
        """Test listing checklists."""
        from enigma.tools.interactive_tools import ChecklistManager
        manager = ChecklistManager(storage_path=tmp_path / "checklists.json")
        manager.create_checklist("List 1", ["A", "B"])
        manager.create_checklist("List 2", ["C"])
        
        lists = manager.list_checklists()
        assert len(lists) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
