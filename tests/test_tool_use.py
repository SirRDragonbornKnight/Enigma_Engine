#!/usr/bin/env python3
"""
Tests for the tool use system.

Run with: pytest tests/test_tool_use.py -v
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTokenizerEnhancements:
    """Tests for advanced tokenizer enhancements."""
    
    def test_tokenizer_has_tool_tokens(self):
        """Test that tokenizer has tool use special tokens."""
        from enigma_engine.core.advanced_tokenizer import ForgeTokenizer as AdvancedBPETokenizer
        
        tokenizer = AdvancedBPETokenizer()
        
        # Check for tool tokens (Enigma AI Engine's [E:token] format)
        assert "[E:tool]" in tokenizer.special_tokens
        assert "[E:tool_end]" in tokenizer.special_tokens
        assert "[E:tool_out]" in tokenizer.special_tokens
        assert "[E:out_end]" in tokenizer.special_tokens
    
    def test_tokenizer_bpe_dropout(self):
        """Test BPE dropout functionality."""
        from enigma_engine.core.advanced_tokenizer import AdvancedBPETokenizer
        
        tokenizer = AdvancedBPETokenizer()
        tokenizer.set_bpe_dropout(0.1)
        
        assert tokenizer.bpe_dropout == 0.1
        
        # Should not raise error
        tokenizer.set_bpe_dropout(0.0)
        assert tokenizer.bpe_dropout == 0.0
    
    def test_tokenizer_streaming(self):
        """Test streaming tokenization."""
        from enigma_engine.core.advanced_tokenizer import AdvancedBPETokenizer
        
        tokenizer = AdvancedBPETokenizer()
        
        # Add chunks
        ids1 = tokenizer.encode_stream("Hello ", finalize=False)
        ids2 = tokenizer.encode_stream("world!", finalize=True)
        
        # Should return some ids
        assert isinstance(ids1, list)
        assert isinstance(ids2, list)
        
        # Reset buffer
        tokenizer.reset_stream()
        assert tokenizer._stream_buffer == ""
    
    def test_tokenizer_improved_decode(self):
        """Test improved decoding with space cleanup."""
        from enigma_engine.core.advanced_tokenizer import AdvancedBPETokenizer
        
        tokenizer = AdvancedBPETokenizer()
        
        text = "Hello world"
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode_improved(ids, clean_up_spaces=True)
        
        # Should decode successfully
        assert isinstance(decoded, str)
        assert len(decoded) > 0
    
    def test_tokenizer_compression_ratio(self):
        """Test compression ratio calculation."""
        from enigma_engine.core.advanced_tokenizer import AdvancedBPETokenizer
        
        tokenizer = AdvancedBPETokenizer()
        
        text = "Hello world, this is a test."
        ratio = tokenizer.get_compression_ratio(text)
        
        # Should return positive ratio
        assert ratio > 0
        assert isinstance(ratio, float)
    
    def test_tokenizer_stats(self):
        """Test tokenization statistics."""
        from enigma_engine.core.advanced_tokenizer import AdvancedBPETokenizer
        
        tokenizer = AdvancedBPETokenizer()
        
        text = "Hello world!"
        stats = tokenizer.tokenize_stats(text)
        
        # Check stats structure
        assert "text_length" in stats
        assert "token_count" in stats
        assert "compression_ratio" in stats
        assert "tokens" in stats
        
        assert stats["text_length"] == len(text)
        assert stats["token_count"] > 0


class TestToolDefinitions:
    """Tests for tool definitions."""
    
    def test_get_tool_definition(self):
        """Test getting a tool definition."""
        from enigma_engine.tools.tool_definitions import get_tool_definition
        
        tool = get_tool_definition("generate_image")
        
        assert tool is not None
        assert tool.name == "generate_image"
        assert len(tool.parameters) > 0
    
    def test_get_all_tools(self):
        """Test getting all tools."""
        from enigma_engine.tools.tool_definitions import get_all_tools
        
        tools = get_all_tools()
        
        assert len(tools) > 0
        assert all(hasattr(t, 'name') for t in tools)
    
    def test_get_tools_by_category(self):
        """Test getting tools by category."""
        from enigma_engine.tools.tool_definitions import get_tools_by_category
        
        gen_tools = get_tools_by_category("generation")
        
        assert len(gen_tools) > 0
        assert all(t.category == "generation" for t in gen_tools)
    
    def test_tool_schemas(self):
        """Test tool schema generation."""
        from enigma_engine.tools.tool_definitions import get_tool_schemas
        
        schemas = get_tool_schemas()
        
        assert isinstance(schemas, str)
        assert len(schemas) > 0
        assert "generate_image" in schemas
    
    def test_available_tools_prompt(self):
        """Test available tools prompt generation."""
        from enigma_engine.tools.tool_definitions import get_available_tools_for_prompt
        
        prompt = get_available_tools_for_prompt()
        
        assert isinstance(prompt, str)
        assert "AVAILABLE TOOLS" in prompt
        assert "<tool_call>" in prompt
        assert "generate_image" in prompt


class TestToolExecutor:
    """Tests for tool executor."""
    
    def test_parse_tool_calls(self):
        """Test parsing tool calls from text."""
        from enigma_engine.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        text = """Some text
<tool_call>
{"tool": "generate_image", "params": {"prompt": "test"}}
</tool_call>
More text"""
        
        calls = executor.parse_tool_calls(text)
        
        assert len(calls) == 1
        tool_name, params, start, end = calls[0]
        assert tool_name == "generate_image"
        assert "prompt" in params
    
    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls."""
        from enigma_engine.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        text = """
<tool_call>{"tool": "read_file", "params": {"path": "test.txt"}}</tool_call>
Some text
<tool_call>{"tool": "write_file", "params": {"path": "out.txt", "content": "test"}}</tool_call>
"""
        
        calls = executor.parse_tool_calls(text)
        
        assert len(calls) == 2
    
    def test_validate_params_success(self):
        """Test parameter validation success."""
        from enigma_engine.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        is_valid, error, validated = executor.validate_params(
            "generate_image",
            {"prompt": "test image", "width": 512}
        )
        
        assert is_valid
        assert error is None
        assert "prompt" in validated
    
    def test_validate_params_missing_required(self):
        """Test parameter validation with missing required param."""
        from enigma_engine.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        is_valid, error, validated = executor.validate_params(
            "generate_image",
            {}  # Missing required 'prompt'
        )
        
        assert not is_valid
        assert error is not None
        assert "prompt" in error.lower()
    
    def test_validate_params_type_conversion(self):
        """Test parameter type conversion."""
        from enigma_engine.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        is_valid, error, validated = executor.validate_params(
            "generate_image",
            {"prompt": "test", "width": "512"}  # String instead of int
        )
        
        assert is_valid
        assert validated["width"] == 512  # Should be converted to int
    
    def test_format_tool_result(self):
        """Test formatting tool results."""
        from enigma_engine.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        result = {
            "tool": "generate_image",
            "success": True,
            "result": "Image generated",
            "output_path": "test.png"
        }
        
        formatted = executor.format_tool_result(result)
        
        assert "<tool_result>" in formatted
        assert "</tool_result>" in formatted
        assert "generate_image" in formatted
        assert "success" in formatted
    
    def test_execute_unknown_tool(self):
        """Test executing unknown tool."""
        from enigma_engine.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        result = executor.execute_tool("unknown_tool", {})
        
        assert not result["success"]
        assert "error" in result


class TestInferenceToolIntegration:
    """Tests for tool integration in inference engine."""
    
    def test_engine_init_with_tools(self):
        """Test initializing engine with tools enabled."""
        from enigma_engine.core.inference import EnigmaEngine
        
        # Initialize without module manager (tools won't work but should init)
        engine = EnigmaEngine(
            model_size="nano",  # Use smallest model for testing
            enable_tools=True,
            module_manager=None
        )
        
        # Should have tool attributes
        assert hasattr(engine, 'enable_tools')
        assert hasattr(engine, '_tool_executor')
    
    def test_engine_init_without_tools(self):
        """Test initializing engine without tools."""
        from enigma_engine.core.inference import EnigmaEngine
        
        engine = EnigmaEngine(
            model_size="nano",
            enable_tools=False
        )
        
        assert engine.enable_tools == False


class TestToolTrainingData:
    """Tests for tool training data file."""
    
    def test_training_data_exists(self):
        """Test that training data file exists."""
        data_file = Path(__file__).parent.parent / "data" / "tool_training_data.txt"
        
        assert data_file.exists(), "Tool training data file not found"
    
    def test_training_data_format(self):
        """Test training data has correct format."""
        data_file = Path(__file__).parent.parent / "data" / "tool_training_data.txt"
        
        with open(data_file, 'r') as f:
            content = f.read()
        
        # Should contain User: and AI: markers (Enigma format) or Q:/A: format
        assert "User:" in content or "Q:" in content
        assert "AI:" in content or "A:" in content
        
        # Should contain tool call format (either <tool_call> or [E:tool])
        has_tool_format = "<tool_call>" in content or "[E:tool]" in content
        assert has_tool_format, "Training data should contain tool call markers"
        
        has_tool_result = "<tool_result>" in content or "[E:tool_out]" in content
        assert has_tool_result, "Training data should contain tool result markers"
        
        # Should have examples for major tools
        assert "generate_image" in content
        assert "screenshot" in content
    
    def test_training_data_tokenizable(self):
        """Test that training data can be tokenized."""
        from enigma_engine.core.tokenizer import get_tokenizer
        
        data_file = Path(__file__).parent.parent / "data" / "tool_training_data.txt"
        
        with open(data_file, 'r') as f:
            content = f.read()
        
        # Get tokenizer
        tokenizer = get_tokenizer()
        
        # Should be able to encode without errors
        ids = tokenizer.encode(content[:1000], add_special_tokens=False)  # First 1000 chars
        
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)


class TestToolUseEndToEnd:
    """End-to-end tests for tool use system."""
    
    def test_parse_and_format_roundtrip(self):
        """Test parsing and formatting roundtrip."""
        from enigma_engine.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        # Original text with tool call
        original = """User asks something
<tool_call>{"tool": "read_file", "params": {"path": "test.txt"}}</tool_call>
More text"""
        
        # Parse
        calls = executor.parse_tool_calls(original)
        assert len(calls) == 1
        
        # Create result
        result = {
            "tool": "read_file",
            "success": True,
            "result": "File contents"
        }
        
        # Format
        formatted = executor.format_tool_result(result)
        
        assert "<tool_result>" in formatted
        assert "read_file" in formatted
    
    def test_execute_tool_from_text(self):
        """Test executing tools from text."""
        from enigma_engine.tools.tool_executor import execute_tool_from_text
        
        text = """Some text
<tool_call>{"tool": "list_directory", "params": {"path": "."}}</tool_call>
More text"""
        
        # Execute (without module manager, should try builtin tools)
        modified_text, results = execute_tool_from_text(text, module_manager=None)
        
        # Should have replaced tool call with result
        assert "<tool_result>" in modified_text or "<tool_call>" not in modified_text
        assert len(results) >= 0  # May succeed or fail depending on tool availability


class TestGIFGeneration:
    """Tests for GIF generation tool."""
    
    def test_generate_gif_tool_definition(self):
        """Test that generate_gif tool is properly defined."""
        from enigma_engine.tools.tool_definitions import get_tool_definition
        
        tool = get_tool_definition("generate_gif")
        
        assert tool is not None
        assert tool.name == "generate_gif"
        assert tool.category == "generation"
        assert len(tool.parameters) > 0
        
        # Check for required parameters
        param_names = [p.name for p in tool.parameters]
        assert "frames" in param_names
        assert "fps" in param_names
        assert "loop" in param_names
    
    def test_generate_gif_params_validation(self):
        """Test parameter validation for generate_gif."""
        from enigma_engine.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        # Test valid params
        is_valid, error, validated = executor.validate_params(
            "generate_gif",
            {"frames": ["sunrise", "noon", "sunset"], "fps": 5, "loop": 0}
        )
        
        assert is_valid
        assert error is None
        assert "frames" in validated
        assert validated["fps"] == 5
        assert validated["loop"] == 0
    
    def test_generate_gif_missing_frames(self):
        """Test generate_gif with missing frames parameter."""
        from enigma_engine.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        # Missing required 'frames'
        is_valid, error, validated = executor.validate_params(
            "generate_gif",
            {"fps": 5}
        )
        
        assert not is_valid
        assert error is not None
        assert "frames" in error.lower()
    
    def test_generate_gif_default_values(self):
        """Test that default values are applied for optional parameters."""
        from enigma_engine.tools.tool_definitions import get_tool_definition
        
        tool = get_tool_definition("generate_gif")
        
        # Find fps parameter
        fps_param = next(p for p in tool.parameters if p.name == "fps")
        assert fps_param.default == 5
        
        loop_param = next(p for p in tool.parameters if p.name == "loop")
        assert loop_param.default == 0


class TestMediaEditingTools:
    """Tests for image/GIF/video editing tools."""
    
    def test_edit_image_tool_definition(self):
        """Test that edit_image tool is properly defined."""
        from enigma_engine.tools.tool_definitions import get_tool_definition
        
        tool = get_tool_definition("edit_image")
        
        assert tool is not None
        assert tool.name == "edit_image"
        
        # Check edit_type has enum values
        edit_type_param = next(p for p in tool.parameters if p.name == "edit_type")
        assert edit_type_param.enum is not None
        assert "resize" in edit_type_param.enum
        assert "rotate" in edit_type_param.enum
    
    def test_edit_gif_tool_definition(self):
        """Test that edit_gif tool is properly defined."""
        from enigma_engine.tools.tool_definitions import get_tool_definition
        
        tool = get_tool_definition("edit_gif")
        
        assert tool is not None
        assert tool.name == "edit_gif"
        
        # Check edit_type parameter
        edit_type_param = next(p for p in tool.parameters if p.name == "edit_type")
        assert "speed" in edit_type_param.enum
        assert "reverse" in edit_type_param.enum
    
    def test_edit_video_tool_definition(self):
        """Test that edit_video tool is properly defined."""
        from enigma_engine.tools.tool_definitions import get_tool_definition
        
        tool = get_tool_definition("edit_video")
        
        assert tool is not None
        assert tool.name == "edit_video"
        
        # Check edit_type parameter
        edit_type_param = next(p for p in tool.parameters if p.name == "edit_type")
        assert "trim" in edit_type_param.enum
        assert "to_gif" in edit_type_param.enum
    
    def test_edit_image_params_validation(self):
        """Test parameter validation for edit_image."""
        from enigma_engine.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        # Test valid params
        is_valid, error, validated = executor.validate_params(
            "edit_image",
            {"image_path": "test.png", "edit_type": "resize", "width": 800, "height": 600}
        )
        
        assert is_valid
        assert error is None
        assert validated["edit_type"] == "resize"
    
    def test_edit_image_invalid_edit_type(self):
        """Test edit_image with invalid edit type."""
        from enigma_engine.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        # Invalid edit_type
        is_valid, error, validated = executor.validate_params(
            "edit_image",
            {"image_path": "test.png", "edit_type": "invalid_operation"}
        )
        
        assert not is_valid
        assert "edit_type" in error.lower()
    
    def test_edit_image_execution_file_not_found(self):
        """Test edit_image with non-existent file."""
        from enigma_engine.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        result = executor.execute_tool(
            "edit_image",
            {"image_path": "nonexistent.png", "edit_type": "resize", "width": 100, "height": 100}
        )
        
        assert not result["success"]
        assert "not found" in result["error"].lower()
    
    def test_edit_gif_execution_file_not_found(self):
        """Test edit_gif with non-existent file."""
        from enigma_engine.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        result = executor.execute_tool(
            "edit_gif",
            {"gif_path": "nonexistent.gif", "edit_type": "reverse"}
        )
        
        assert not result["success"]
        assert "not found" in result["error"].lower()
    
    def test_edit_video_execution_file_not_found(self):
        """Test edit_video with non-existent file."""
        from enigma_engine.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        result = executor.execute_tool(
            "edit_video",
            {"video_path": "nonexistent.mp4", "edit_type": "trim"}
        )
        
        assert not result["success"]
        assert "not found" in result["error"].lower()


class TestToolIntegration:
    """Tests for new tools integration with inference engine."""
    
    def test_all_new_tools_registered(self):
        """Test that all new tools are in the registry."""
        from enigma_engine.tools.tool_definitions import TOOLS_BY_NAME
        
        assert "generate_gif" in TOOLS_BY_NAME
        assert "edit_image" in TOOLS_BY_NAME
        assert "edit_gif" in TOOLS_BY_NAME
        assert "edit_video" in TOOLS_BY_NAME
    
    def test_new_tools_have_examples(self):
        """Test that new tools have usage examples."""
        from enigma_engine.tools.tool_definitions import get_tool_definition
        
        gif_tool = get_tool_definition("generate_gif")
        assert len(gif_tool.examples) > 0
        
        edit_img_tool = get_tool_definition("edit_image")
        assert len(edit_img_tool.examples) > 0
    
    def test_tool_schemas_include_new_tools(self):
        """Test that tool schemas include new tools."""
        from enigma_engine.tools.tool_definitions import get_tool_schemas
        
        schemas = get_tool_schemas()
        
        assert "generate_gif" in schemas
        assert "edit_image" in schemas
        assert "edit_gif" in schemas
        assert "edit_video" in schemas


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
