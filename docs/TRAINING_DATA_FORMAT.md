# Training Data Format Guide

## Overview

All training data files in the AI Tester AI Engine follow standardized formats to ensure consistency and proper model training. This guide documents the official formats.

## File Structure

### Supported Training Files
- `tool_training_data.txt` - Tool usage examples with AI responses
- `tool_training_examples.txt` - Comprehensive tool examples (all tools)
- `personality_development.txt` - Personality trait development
- `self_awareness_training.txt` - Self-reflection and awareness
- `combined_action_training.txt` - Multi-tool complex interactions

## Standard Format

### Conversation Format

**Always use:**
```
Q: User question or statement
A: AI response
```

**Do NOT use:**
- `User:` / `Assistant:`
- `Human:` / `AI:`
- Any other variations

### Tool Call Format (STANDARDIZED)

**JSON-based format** (standard as of 2024):
```
<tool_call>{"tool": "tool_name", "params": {"param1": "value1", "param2": "value2"}}</tool_call>
<tool_result>{"tool": "tool_name", "success": true, "result": "Result text here"}</tool_result>
```

**Example:**
```
Q: Generate an image of a sunset
A: I'll create that for you!
<tool_call>{"tool": "generate_image", "params": {"prompt": "beautiful sunset over ocean", "width": 512, "height": 512}}</tool_call>
<tool_result>{"tool": "generate_image", "success": true, "result": "Image generated successfully", "output_path": "outputs/sunset.png"}</tool_result>
Here's your sunset image! Saved to outputs/sunset.png.
```

### Tool Call Components

#### Tool Call Structure
- **Opening tag**: `<tool_call>`
- **JSON object**:
  - `tool` (string, required): Tool name (e.g., "generate_image")
  - `params` (object, required): Parameters as key-value pairs
- **Closing tag**: `</tool_call>`

#### Tool Result Structure
- **Opening tag**: `<tool_result>`
- **JSON object**:
  - `tool` (string, optional): Tool name (for reference)
  - `success` (boolean, required): Whether tool execution succeeded
  - `result` (string, required): Result message or description
  - `output_path` (string, optional): Path to generated file
  - `error` (string, optional): Error message if success=false
  - Additional tool-specific fields as needed
- **Closing tag**: `</tool_result>`

## Parameter Guidelines

### Common Tool Parameters

#### Image Generation
```json
{
  "tool": "generate_image",
  "params": {
    "prompt": "description of image",
    "width": 512,
    "height": 512,
    "steps": 20
  }
}
```

#### GIF Generation
```json
{
  "tool": "generate_gif",
  "params": {
    "frames": ["frame 1 description", "frame 2 description"],
    "fps": 5,
    "loop": 0,
    "width": 512,
    "height": 512
  }
}
```

#### Image Editing
```json
{
  "tool": "edit_image",
  "params": {
    "image_path": "path/to/image.jpg",
    "edit_type": "resize",
    "width": 800,
    "height": 600
  }
}
```

#### File Operations
```json
{
  "tool": "write_file",
  "params": {
    "path": "/path/to/file.txt",
    "content": "File content here"
  }
}
```

```json
{
  "tool": "read_file",
  "params": {
    "path": "/path/to/file.txt"
  }
}
```

## Best Practices

### 1. Natural Conversation Flow
- Make AI responses conversational and natural
- Acknowledge the user's request before calling tools
- Explain results after tool execution
- Use appropriate tone (helpful, friendly, professional)

### 2. Error Handling
```
Q: Read a file that doesn't exist
A: Let me try to read that file for you.
<tool_call>{"tool": "read_file", "params": {"path": "nonexistent.txt"}}</tool_call>
<tool_result>{"tool": "read_file", "success": false, "error": "File not found: nonexistent.txt"}</tool_result>
I'm sorry, but I couldn't find the file "nonexistent.txt". Could you check the filename and path?
```

### 3. Multi-Tool Sequences
```
Q: Research Python and save a summary
A: I'll search for Python information first.
<tool_call>{"tool": "web_search", "params": {"query": "Python programming language overview"}}</tool_call>
<tool_result>{"tool": "web_search", "success": true, "result": "Python is a high-level programming language..."}</tool_result>
Great! Now let me save this for you.
<tool_call>{"tool": "write_file", "params": {"path": "python_info.txt", "content": "Python Overview:\n\nPython is a high-level programming language..."}}</tool_call>
<tool_result>{"tool": "write_file", "success": true, "result": "File written successfully"}</tool_result>
Done! I've researched Python and saved a summary to python_info.txt.
```

### 4. Tool Results
- Always include `success` field
- Provide meaningful `result` text
- Include `output_path` for generated files
- Use `error` field when success=false
- Add tool-specific fields as needed (e.g., `frame_count` for GIFs)

## File Headers

Each training file should start with a header explaining its format:

```python
"""
File Description
================

Brief description of what this file teaches.

Format:
  Q: User query
  A: AI response with tool calls

Tool Call Format:
  <tool_call>{"tool": "name", "params": {...}}</tool_call>
  <tool_result>{"success": true, "result": "..."}</tool_result>

Note: This format is consistent with tool_executor.py and tool_definitions.py
"""
```

## Validation

### Check Format Compliance
```bash
# Check for old format markers
grep '<|tool_call|>' data/*.txt
grep 'User:' data/*.txt
grep 'Assistant:' data/*.txt

# Should find none!

# Verify correct format
grep '<tool_call>' data/*.txt | head -5
grep '"tool":' data/*.txt | head -5
```

### Test with Tokenizer
```python
from ai_tester.core.tokenizer import get_tokenizer

tokenizer = get_tokenizer()
with open('data/tool_training_data.txt', 'r') as f:
    content = f.read()
    
# Should encode without errors
ids = tokenizer.encode(content[:1000])
print(f"Encoded {len(ids)} tokens")
```

## Migration from Old Formats

### Old Format (deprecated)
```
# DO NOT USE
User: Generate an image
Assistant: <|tool_call|>generate_image("sunset")<|tool_end|>
<|tool_result|>Image generated<|tool_result_end|>
Done!
```

### New Format (current standard)
```
# CORRECT FORMAT
Q: Generate an image
A: <tool_call>{"tool": "generate_image", "params": {"prompt": "sunset"}}</tool_call>
<tool_result>{"tool": "generate_image", "success": true, "result": "Image generated"}</tool_result>
Done!
```

## Updates and Changes

### Version History
- **v2.0 (Jan 2026)**: Standardized to JSON format for all tools
- **v1.5**: Added GIF generation and media editing tools
- **v1.0**: Initial tool use system with multiple format variants

### Future Compatibility
- This format is designed to be parseable by `tool_executor.py`
- Compatible with all tools in `tool_definitions.py`
- Validated by test suite in `tests/test_tool_use.py`

---

**Last Updated**: January 2026

**Maintained By**: AI Tester Engine Development Team

**Questions?** See `docs/TOOL_USE.md` for usage documentation
