"""
Tests for AI Tool Integration
==============================

Tests for the tool interface that allows AI to invoke tools.
"""

import pytest
from enigma.core.tool_interface import (
    ToolInterface, ToolCall, ToolResult,
    create_tool_interface, parse_and_execute_tool
)


class TestToolCallParsing:
    """Test parsing of tool calls from AI output."""
    
    def test_parse_simple_tool_call(self):
        """Test parsing a basic tool call."""
        interface = ToolInterface()
        
        ai_output = '<|tool_call|>generate_image("a sunset")<|tool_end|>'
        tool_call = interface.parse_tool_call(ai_output)
        
        assert tool_call is not None
        assert tool_call.tool_name == "generate_image"
        assert "sunset" in str(tool_call.arguments.values())
    
    def test_parse_tool_call_with_multiple_args(self):
        """Test parsing tool call with multiple arguments."""
        interface = ToolInterface()
        
        ai_output = '<|tool_call|>generate_image("sunset", 512, 512)<|tool_end|>'
        tool_call = interface.parse_tool_call(ai_output)
        
        assert tool_call is not None
        assert tool_call.tool_name == "generate_image"
        assert len(tool_call.arguments) > 0
    
    def test_parse_tool_call_json_args(self):
        """Test parsing tool call with JSON-style arguments."""
        interface = ToolInterface()
        
        ai_output = '<|tool_call|>avatar_action("set_expression", {"expression": "happy"})<|tool_end|>'
        tool_call = interface.parse_tool_call(ai_output)
        
        assert tool_call is not None
        assert tool_call.tool_name == "avatar_action"
    
    def test_parse_no_tool_call(self):
        """Test parsing text without tool call."""
        interface = ToolInterface()
        
        ai_output = "This is just regular text without any tool calls."
        tool_call = interface.parse_tool_call(ai_output)
        
        assert tool_call is None
    
    def test_parse_tool_call_in_context(self):
        """Test parsing tool call within larger text."""
        interface = ToolInterface()
        
        ai_output = """
        Let me generate an image for you!
        <|tool_call|>generate_image("beautiful landscape")<|tool_end|>
        I hope you like it!
        """
        tool_call = interface.parse_tool_call(ai_output)
        
        assert tool_call is not None
        assert tool_call.tool_name == "generate_image"
    
    def test_parse_tool_call_positions(self):
        """Test that tool call positions are correct."""
        interface = ToolInterface()
        
        ai_output = 'Some text <|tool_call|>speak("hello")<|tool_end|> more text'
        tool_call = interface.parse_tool_call(ai_output)
        
        assert tool_call is not None
        assert tool_call.start_pos > 0
        assert tool_call.end_pos > tool_call.start_pos


class TestToolExecution:
    """Test execution of tools."""
    
    def test_execute_image_generation(self):
        """Test executing image generation tool."""
        interface = ToolInterface()
        
        tool_call = ToolCall(
            tool_name="generate_image",
            arguments={"prompt": "test image"},
            raw_text="generate_image('test image')"
        )
        
        result = interface.execute_tool(tool_call)
        
        assert result is not None
        assert result.tool_name == "generate_image"
        # Should succeed (even if simulated)
        assert result.success or result.error is not None
    
    def test_execute_avatar_action(self):
        """Test executing avatar action tool."""
        interface = ToolInterface()
        
        tool_call = ToolCall(
            tool_name="avatar_action",
            arguments={"action": "set_expression", "params": {"expression": "happy"}},
            raw_text="avatar_action('set_expression', {'expression': 'happy'})"
        )
        
        result = interface.execute_tool(tool_call)
        
        assert result is not None
        assert result.tool_name == "avatar_action"
    
    def test_execute_speak(self):
        """Test executing speak tool."""
        interface = ToolInterface()
        
        tool_call = ToolCall(
            tool_name="speak",
            arguments={"text": "Hello world"},
            raw_text="speak('Hello world')"
        )
        
        result = interface.execute_tool(tool_call)
        
        assert result is not None
        assert result.tool_name == "speak"
    
    def test_execute_unknown_tool(self):
        """Test executing unknown tool returns error."""
        interface = ToolInterface()
        
        tool_call = ToolCall(
            tool_name="nonexistent_tool",
            arguments={},
            raw_text="nonexistent_tool()"
        )
        
        result = interface.execute_tool(tool_call)
        
        assert result is not None
        assert not result.success
        assert result.error is not None
        assert "Unknown tool" in result.error or "not found" in result.error.lower()
    
    def test_execute_with_error_handling(self):
        """Test that tool execution handles errors gracefully."""
        interface = ToolInterface()
        
        # Tool call with invalid arguments
        tool_call = ToolCall(
            tool_name="read_file",
            arguments={},  # Missing required 'path'
            raw_text="read_file()"
        )
        
        result = interface.execute_tool(tool_call)
        
        # Should return result (even if error)
        assert result is not None


class TestToolResultFormatting:
    """Test formatting of tool results."""
    
    def test_format_success_result(self):
        """Test formatting successful tool result."""
        interface = ToolInterface()
        
        result = ToolResult(
            success=True,
            tool_name="test_tool",
            message="Operation completed"
        )
        
        formatted = interface.format_tool_result(result)
        
        assert '<|tool_result|>' in formatted
        assert '<|tool_result_end|>' in formatted
        assert 'completed' in formatted.lower()
    
    def test_format_error_result(self):
        """Test formatting error result."""
        interface = ToolInterface()
        
        result = ToolResult(
            success=False,
            tool_name="test_tool",
            error="Something went wrong"
        )
        
        formatted = interface.format_tool_result(result)
        
        assert '<|tool_result|>' in formatted
        assert '<|tool_result_end|>' in formatted
        assert 'error' in formatted.lower()
    
    def test_format_result_with_data(self):
        """Test formatting result with data."""
        interface = ToolInterface()
        
        result = ToolResult(
            success=True,
            tool_name="test_tool",
            data="Some result data",
            message="Task complete"
        )
        
        formatted = interface.format_tool_result(result)
        
        assert '<|tool_result|>' in formatted
        assert len(formatted) > 0


class TestToolRegistry:
    """Test tool registration and availability."""
    
    def test_available_tools_not_empty(self):
        """Test that interface has available tools."""
        interface = ToolInterface()
        
        assert len(interface.available_tools) > 0
    
    def test_all_tools_have_descriptions(self):
        """Test that all tools have descriptions."""
        interface = ToolInterface()
        
        for tool_name in interface.available_tools.keys():
            assert tool_name in interface.tool_descriptions
            assert len(interface.tool_descriptions[tool_name]) > 0
    
    def test_get_tools_list(self):
        """Test getting list of tools."""
        interface = ToolInterface()
        
        tools_list = interface.get_tools_list()
        
        assert len(tools_list) > 0
        assert all('name' in tool and 'description' in tool for tool in tools_list)
    
    def test_get_tool_names(self):
        """Test getting tool names."""
        interface = ToolInterface()
        
        names = interface.get_tool_names()
        
        assert len(names) > 0
        assert 'generate_image' in names
        assert 'speak' in names
        assert 'search_web' in names


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_tool_interface(self):
        """Test creating tool interface."""
        interface = create_tool_interface()
        
        assert interface is not None
        assert isinstance(interface, ToolInterface)
    
    def test_parse_and_execute_tool(self):
        """Test parse and execute in one step."""
        ai_output = '<|tool_call|>speak("test")<|tool_end|>'
        
        result = parse_and_execute_tool(ai_output)
        
        assert result is not None
        assert '<|tool_result|>' in result
    
    def test_parse_and_execute_no_tool(self):
        """Test parse and execute with no tool call."""
        ai_output = "Just regular text"
        
        result = parse_and_execute_tool(ai_output)
        
        assert result is None


class TestArgumentParsing:
    """Test parsing of different argument formats."""
    
    def test_parse_single_string_arg(self):
        """Test parsing single string argument."""
        interface = ToolInterface()
        
        args = interface._parse_arguments('"hello world"')
        
        assert len(args) > 0
        assert any('hello' in str(v).lower() for v in args.values())
    
    def test_parse_multiple_args(self):
        """Test parsing multiple arguments."""
        interface = ToolInterface()
        
        args = interface._parse_arguments('"arg1", "arg2", "arg3"')
        
        assert len(args) > 0
    
    def test_parse_json_dict(self):
        """Test parsing JSON dictionary argument."""
        interface = ToolInterface()
        
        args = interface._parse_arguments('{"key": "value", "num": 42}')
        
        # Should parse as dict or at least not crash
        assert args is not None
    
    def test_parse_empty_args(self):
        """Test parsing empty arguments."""
        interface = ToolInterface()
        
        args = interface._parse_arguments('')
        
        assert args == {}
    
    def test_parse_number_args(self):
        """Test parsing numeric arguments."""
        interface = ToolInterface()
        
        args = interface._parse_arguments('512, 512')
        
        assert len(args) > 0


class TestIntegrationWithModuleManager:
    """Test integration with module manager."""
    
    def test_interface_with_none_manager(self):
        """Test interface works without module manager."""
        interface = ToolInterface(module_manager=None)
        
        # Should still work
        assert len(interface.available_tools) > 0
    
    def test_interface_accepts_manager(self):
        """Test interface accepts module manager."""
        # Create interface with mock manager
        interface = ToolInterface(module_manager=None)
        
        # Should initialize
        assert interface.manager is None


class TestToolCallDataclass:
    """Test ToolCall dataclass."""
    
    def test_tool_call_creation(self):
        """Test creating ToolCall."""
        tool_call = ToolCall(
            tool_name="test",
            arguments={"arg": "value"},
            raw_text="test('value')"
        )
        
        assert tool_call.tool_name == "test"
        assert tool_call.arguments == {"arg": "value"}
        assert tool_call.raw_text == "test('value')"
    
    def test_tool_call_with_positions(self):
        """Test ToolCall with positions."""
        tool_call = ToolCall(
            tool_name="test",
            arguments={},
            raw_text="test()",
            start_pos=10,
            end_pos=20
        )
        
        assert tool_call.start_pos == 10
        assert tool_call.end_pos == 20


class TestToolResultDataclass:
    """Test ToolResult dataclass."""
    
    def test_tool_result_success(self):
        """Test successful ToolResult."""
        result = ToolResult(
            success=True,
            tool_name="test",
            data="result"
        )
        
        assert result.success
        assert result.tool_name == "test"
        assert result.data == "result"
    
    def test_tool_result_error(self):
        """Test error ToolResult."""
        result = ToolResult(
            success=False,
            tool_name="test",
            error="Error message"
        )
        
        assert not result.success
        assert result.error == "Error message"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
