# AI Tool Integration - Implementation Guide

## Overview

This implementation adds comprehensive AI tool integration to Enigma Engine, allowing the AI model to invoke external tools during generation. This bridges the gap between "tools exist" and "AI can actually use them."

## What's New

### 1. Enhanced Tokenizer (`enigma/core/advanced_tokenizer.py`)

**39 Special Tokens** for comprehensive tool use:

```python
# Tool invocation
<|tool_call|>, <|tool_result|>, <|tool_end|>, <|tool_result_end|>

# Modalities
<|image|>, <|audio|>, <|video|>, <|vision|>

# Actions/Capabilities
<|generate_image|>, <|avatar_action|>, <|speak|>, <|listen|>
<|search_web|>, <|read_file|>, <|write_file|>, <|capture_screen|>, <|run_code|>

# Conversation roles
<|system|>, <|user|>, <|assistant|>

# Formatting
<|code|>, <|code_end|>, <|newline|>, <|tab|>

# Meta
<|thinking|>, <|thinking_end|>, <|error|>, <|success|>
```

**New Features:**
- Improved BPE algorithm with better regex pre-tokenization for code, numbers, and special characters
- Streaming encode/decode support (`encode_stream()`, `decode_stream()`)
- Separate merges file support for better reproducibility
- Enhanced unicode and emoji handling
- Optimized caching

**Usage:**
```python
from enigma.core.advanced_tokenizer import AdvancedBPETokenizer

tokenizer = AdvancedBPETokenizer()

# Encode with special tokens
text = '<|tool_call|>generate_image("sunset")<|tool_end|>'
ids = tokenizer.encode(text)

# Save with separate merges
tokenizer.save("tokenizer.json", save_merges_separately=True)
# Creates: tokenizer.json and tokenizer.merges

# Stream encode
for ids in tokenizer.encode_stream(text_chunks):
    process(ids)
```

### 2. Tool Interface (`enigma/core/tool_interface.py`)

Central interface for AI to invoke tools. Handles:
- Parsing tool calls from AI output
- Executing tools
- Formatting results for AI understanding

**Supported Tools:**
1. `generate_image(prompt)` - Image generation
2. `avatar_action(action, params)` - Avatar control
3. `capture_screen()` - Screen capture and vision
4. `speak(text)` - Text-to-speech
5. `search_web(query)` - Web search
6. `read_file(path)` - Read files
7. `write_file(path, content)` - Write files
8. `list_directory(path)` - List directories

**Usage:**
```python
from enigma.core.tool_interface import ToolInterface

interface = ToolInterface(module_manager)

# Parse AI output
ai_output = '<|tool_call|>generate_image("a sunset")<|tool_end|>'
tool_call = interface.parse_tool_call(ai_output)

# Execute
if tool_call:
    result = interface.execute_tool(tool_call)
    formatted = interface.format_tool_result(result)
    print(formatted)
    # <|tool_result|>Image generated successfully<|tool_result_end|>
```

**One-liner:**
```python
from enigma.core.tool_interface import parse_and_execute_tool

result = parse_and_execute_tool(ai_output)
```

### 3. Tool Prompts (`enigma/core/tool_prompts.py`)

System prompts that teach the AI how to use tools.

**Usage:**
```python
from enigma.core.tool_prompts import get_tool_enabled_system_prompt

# Get full system prompt
system_prompt = get_tool_enabled_system_prompt()

# Use in chat
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Show me a dragon"}
]
```

The system prompt includes:
- Instructions for tool use
- Examples of each tool
- Proper format for tool calls
- Guidelines for when to use tools

### 4. Tool-Aware Generation (`enigma/core/inference.py`)

New methods in `EnigmaEngine` for generation with tool support:

**`generate_with_tools()`**
```python
from enigma.core.inference import EnigmaEngine

engine = EnigmaEngine()

response = engine.generate_with_tools(
    prompt="Show me a dragon",
    module_manager=manager,
    max_gen=200,
    max_tool_iterations=5
)
# AI generates tool call, it executes, AI continues with result
```

**`stream_generate_with_tools()`**
```python
for token in engine.stream_generate_with_tools(
    prompt="Search for AI news",
    module_manager=manager
):
    print(token, end="", flush=True)
# Streams tokens, pausing for tool execution
```

**How it works:**
1. AI generates text as normal
2. When `<|tool_call|>...<|tool_end|>` is detected, execution pauses
3. Tool is parsed and executed
4. Result is formatted as `<|tool_result|>...<|tool_result_end|>`
5. Generation continues with result in context
6. Supports multiple tool calls in sequence

### 5. Training Data (`data/tool_training_examples.txt`)

**70+ Training Examples** covering all tools:

- 15 image generation examples
- 12 avatar control examples
- 8 vision/screen capture examples
- 8 voice/TTS examples
- 10 web search examples
- 12 file operation examples
- 4 multi-tool complex scenarios

**Format:**
```
Q: User query
A: AI response with tool calls
<|tool_call|>tool_name("args")<|tool_end|>
<|tool_result|>Result from tool<|tool_result_end|>
AI continues based on result
```

**Training:**
```bash
# Include in your training data
python scripts/train.py --data data/tool_training_examples.txt --vocab-size 8000

# Or append to existing training data
cat data/tool_training_examples.txt >> data/training_data.txt
```

## Quick Start

### 1. Train Tokenizer with Special Tokens

```python
from enigma.core.advanced_tokenizer import AdvancedBPETokenizer

tokenizer = AdvancedBPETokenizer()
tokenizer.train(
    texts=["your", "training", "texts"],
    vocab_size=8000,
    verbose=True
)
tokenizer.save("models/tokenizer.json")
```

### 2. Train Model with Tool Examples

```bash
python scripts/train.py \
    --data data/tool_training_examples.txt \
    --tokenizer models/tokenizer.json \
    --model-size small \
    --epochs 10
```

### 3. Use Tool-Aware Generation

```python
from enigma.core.inference import EnigmaEngine
from enigma.modules import ModuleManager

# Load model
engine = EnigmaEngine(
    model_path="models/enigma.pth",
    tokenizer_path="models/tokenizer.json"
)

# Load modules for tools (optional but recommended)
manager = ModuleManager()
manager.load('image_gen_local')  # For generate_image

# Generate with tools
response = engine.generate_with_tools(
    prompt="Create an image of a dragon breathing fire",
    module_manager=manager,
    max_gen=200
)

print(response)
```

### 4. Run Demo

```bash
python demo_tool_integration.py
```

## Architecture

```
User Input
    ↓
EnigmaEngine.generate_with_tools()
    ↓
[AI generates tokens] → [Tool call detected]
    ↓                           ↓
Continue generation    ToolInterface.parse_tool_call()
    ↑                           ↓
    |                  ToolInterface.execute_tool()
    |                           ↓
    |                  Execute via ModuleManager/Addons
    |                           ↓
    |                  ToolInterface.format_tool_result()
    |                           ↓
    └─────────── Feed result back to generation

Multiple iterations supported (max_tool_iterations parameter)
```

## Testing

All functionality is thoroughly tested:

```bash
# Test tokenizer upgrades (21 tests)
pytest tests/test_tokenizer_upgrades.py -v

# Test tool interface (32 tests)  
pytest tests/test_tool_interface.py -v

# Run all new tests (53 tests)
pytest tests/test_tokenizer_upgrades.py tests/test_tool_interface.py -v
```

**Test Coverage:**
- Special tokens encoding/decoding
- Streaming encode/decode
- Merges file support
- Tool call parsing (all formats)
- Tool execution (all tools)
- Result formatting
- Error handling
- Integration points

## Examples

### Example 1: Image Generation

```python
# AI learns to generate images
Q: Show me a dragon
A: I'll create that for you!
<|tool_call|>generate_image("majestic dragon with scales, breathing fire")<|tool_end|>
<|tool_result|>Image generated successfully<|tool_result_end|>
Here's your dragon! A majestic creature with shimmering scales.
```

### Example 2: Web Search

```python
# AI learns to search web
Q: What's the latest news about AI?
A: Let me search for that.
<|tool_call|>search_web("latest AI news 2024")<|tool_end|>
<|tool_result|>Found: Recent breakthroughs in LLMs and robotics<|tool_result_end|>
According to the latest news, there have been exciting breakthroughs...
```

### Example 3: Multi-Tool Use

```python
# AI learns to chain tools
Q: Search for Python tutorials and save them
A: I'll search and save the results for you.
<|tool_call|>search_web("Python programming tutorials")<|tool_end|>
<|tool_result|>Found: Python.org, Real Python, Codecademy<|tool_result_end|>
<|tool_call|>write_file("python_resources.txt", "Python Resources:\n1. Python.org\n2. Real Python\n3. Codecademy")<|tool_end|>
<|tool_result|>File written successfully<|tool_result_end|>
Done! I've saved the best Python tutorials to a file.
```

## Best Practices

### Training
1. **Mix tool examples with regular conversations** - Don't train only on tool data
2. **Use diverse examples** - Cover all tools and use cases
3. **Include failures** - Show AI how to handle tool errors gracefully
4. **Progressive training** - Start with basic tool use, add complex scenarios

### Inference
1. **Set reasonable max_tool_iterations** (3-5) to prevent loops
2. **Load required modules** before enabling tools
3. **Handle tool failures gracefully** in your application
4. **Monitor tool usage** for unexpected behavior

### Prompting
1. **Include system prompt** when tool use is expected
2. **Be clear about available tools** in context
3. **Show examples** of desired tool usage in few-shot prompts

## Troubleshooting

### Issue: Tools not executing
**Solution:** Ensure modules are loaded:
```python
manager = ModuleManager()
manager.load('image_gen_local')
engine.generate_with_tools(..., module_manager=manager)
```

### Issue: Tool calls not parsed
**Solution:** Check tokenizer has special tokens:
```python
tokenizer = AdvancedBPETokenizer()
assert '<|tool_call|>' in tokenizer.special_tokens
```

### Issue: Model doesn't use tools
**Solution:** Train with tool examples:
```bash
python scripts/train.py --data data/tool_training_examples.txt
```

## File Reference

### Core Implementation
- `enigma/core/advanced_tokenizer.py` - Enhanced tokenizer
- `enigma/core/tool_interface.py` - Tool interface and execution
- `enigma/core/tool_prompts.py` - System prompts for tools
- `enigma/core/inference.py` - Tool-aware generation methods

### Training & Testing
- `data/tool_training_examples.txt` - 70+ training examples
- `tests/test_tokenizer_upgrades.py` - 21 tokenizer tests
- `tests/test_tool_interface.py` - 32 tool interface tests
- `demo_tool_integration.py` - Demo script

## Future Enhancements

Potential additions:
- More tool types (database, API calls, shell commands)
- Tool approval system (ask user before executing)
- Tool usage analytics and logging
- Parallel tool execution
- Custom tool registration API
- Tool composition (tools calling other tools)

## Contributing

When adding new tools:
1. Register in `ToolInterface._register_tools()`
2. Implement execution method
3. Add to `tool_prompts.py`
4. Create training examples in `tool_training_examples.txt`
5. Add tests in `test_tool_interface.py`

## License

Same as Enigma Engine main project.

## Credits

Implemented as part of the Enigma Engine AI Tool Integration upgrade.
