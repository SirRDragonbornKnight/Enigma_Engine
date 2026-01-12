# Tool Use System Documentation

## Overview

The AI Tester Engine now includes a powerful **Tool Use System** that allows the AI to execute external capabilities (image generation, vision, avatar control, file operations, etc.) naturally during conversations.

When trained on tool use examples, the AI learns to:
1. **Recognize** when a tool is needed
2. **Format** tool calls correctly
3. **Interpret** tool results
4. **Respond** naturally after using tools

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AI Tester Model                         │
│              (Trained on tool use examples)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Generates text with
                         │ <tool_call> tags
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Tool Executor                             │
│  - Parses tool calls from AI output                         │
│  - Validates parameters                                     │
│  - Executes appropriate module/function                     │
│  - Returns structured results                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Results injected back
                         │ as <tool_result> tags
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   AI Tester Model                           │
│          (Continues generation with results)                │
└─────────────────────────────────────────────────────────────┘
```

## Tool Call Format

The AI outputs tool calls in a structured format:

```
<tool_call>
{"tool": "tool_name", "params": {"param1": "value1", "param2": "value2"}}
</tool_call>
```

### Example:
```
<tool_call>
{"tool": "generate_image", "params": {"prompt": "sunset over mountains", "width": 512, "height": 512}}
</tool_call>
```

## Tool Result Format

After execution, results are injected back as:

```
<tool_result>
{"tool": "tool_name", "success": true/false, "result": "..."}
</tool_result>
```

### Success Example:
```
<tool_result>
{"tool": "generate_image", "success": true, "result": "Image generated successfully", "output_path": "outputs/image_001.png"}
</tool_result>
```

### Error Example:
```
<tool_result>
{"tool": "read_file", "success": false, "error": "File not found: missing.txt"}
</tool_result>
```

## Available Tools

### Generation Tools

#### generate_image
Generate images from text descriptions.
- **Module:** `image_gen_local` or `image_gen_api`
- **Parameters:**
  - `prompt` (string, required): Description of image to generate
  - `width` (int, optional, default=512): Image width in pixels
  - `height` (int, optional, default=512): Image height in pixels
  - `steps` (int, optional, default=20): Number of inference steps

#### generate_code
Generate code in various programming languages.
- **Module:** `code_gen_local` or `code_gen_api`
- **Parameters:**
  - `description` (string, required): What the code should do
  - `language` (string, optional, default="python"): Programming language

#### generate_video
Generate videos from text descriptions.
- **Module:** `video_gen_local` or `video_gen_api`
- **Parameters:**
  - `prompt` (string, required): Description of video
  - `duration` (float, optional, default=3.0): Duration in seconds
  - `fps` (int, optional, default=24): Frames per second

#### generate_audio
Generate audio or music from descriptions.
- **Module:** `audio_gen_local` or `audio_gen_api`
- **Parameters:**
  - `prompt` (string, required): Description of audio
  - `duration` (float, optional, default=5.0): Duration in seconds

#### generate_gif
Generate an animated GIF from a list of image prompts.
- **Module:** `image_gen_local` or `image_gen_api`
- **Parameters:**
  - `frames` (list, required): List of text prompts for each frame
  - `fps` (int, optional, default=5): Frames per second
  - `loop` (int, optional, default=0): Number of loops (0 = infinite)
  - `width` (int, optional, default=512): Width of each frame
  - `height` (int, optional, default=512): Height of each frame
- **Examples:**
  - "Create a GIF showing sunrise, noon, and sunset"
  - "Generate an animated GIF of a cat walking"
  - "Make a GIF of a flower blooming"

### Media Editing Tools

#### edit_image
Edit an existing image with various transformations.
- **Module:** Built-in (uses Pillow)
- **Parameters:**
  - `image_path` (string, required): Path to the image file
  - `edit_type` (string, required): Type of edit - "resize", "rotate", "flip", "brightness", "contrast", "blur", "sharpen", "grayscale", "crop"
  - `width` (int, optional): New width for resize
  - `height` (int, optional): New height for resize
  - `angle` (int, optional): Rotation angle in degrees
  - `direction` (string, optional): Flip direction - "horizontal" or "vertical"
  - `factor` (float, optional, default=1.0): Adjustment factor for brightness/contrast
  - `crop_box` (list, optional): Crop coordinates [left, top, right, bottom]
- **Examples:**
  - "Resize the image to 800x600"
  - "Rotate the image 90 degrees"
  - "Increase the brightness"

#### edit_gif
Edit an existing GIF animation.
- **Module:** Built-in (uses Pillow)
- **Parameters:**
  - `gif_path` (string, required): Path to the GIF file
  - `edit_type` (string, required): Type of edit - "speed", "reverse", "crop", "resize", "extract_frames"
  - `speed_factor` (float, optional, default=1.0): Speed multiplier (2.0 = 2x faster)
  - `width` (int, optional): New width for resize
  - `height` (int, optional): New height for resize
  - `crop_box` (list, optional): Crop coordinates [left, top, right, bottom]
- **Examples:**
  - "Make the GIF play twice as fast"
  - "Reverse the GIF animation"
  - "Extract all frames from the GIF"

#### edit_video
Edit an existing video file.
- **Module:** Built-in (requires moviepy)
- **Parameters:**
  - `video_path` (string, required): Path to the video file
  - `edit_type` (string, required): Type of edit - "trim", "speed", "extract_frames", "resize", "to_gif"
  - `start_time` (float, optional, default=0.0): Start time for trim
  - `end_time` (float, optional): End time for trim
  - `speed_factor` (float, optional, default=1.0): Speed multiplier
  - `width` (int, optional): New width for resize
  - `height` (int, optional): New height for resize
  - `fps` (int, optional, default=10): Frames per second for extract_frames or to_gif
- **Examples:**
  - "Trim the video from 10 to 30 seconds"
  - "Convert video to GIF"
  - "Speed up the video 2x"

### Perception Tools

#### analyze_image
Analyze and describe images using computer vision.
- **Module:** `vision`
- **Parameters:**
  - `image_path` (string, required): Path to image file
  - `detail_level` (string, optional, default="normal"): "brief", "normal", or "detailed"

#### find_on_screen
Find elements on screen using vision.
- **Module:** `vision`
- **Parameters:**
  - `query` (string, required): What to look for on screen

### Control Tools

#### control_avatar
Control avatar expressions and animations.
- **Module:** `avatar`
- **Parameters:**
  - `action` (string, required): "expression", "animation", "gesture", or "speak"
  - `value` (string, required): Value for the action

#### speak
Speak text out loud using text-to-speech.
- **Module:** `voice_output`
- **Parameters:**
  - `text` (string, required): Text to speak
  - `voice` (string, optional, default="default"): Voice to use

### System Tools

#### read_file
Read file contents.
- **Parameters:**
  - `path` (string, required): File path

#### write_file
Write content to a file.
- **Parameters:**
  - `path` (string, required): File path
  - `content` (string, required): Content to write

#### list_directory
List files in a directory.
- **Parameters:**
  - `path` (string, optional, default="."): Directory path

#### web_search
Search the internet.
- **Parameters:**
  - `query` (string, required): Search query
  - `num_results` (int, optional, default=5): Number of results

#### fetch_webpage
Get content from a webpage.
- **Parameters:**
  - `url` (string, required): Webpage URL

## Usage

### 1. Training the AI

Train the AI on tool use examples:

```python
from ai_tester.core.tokenizer import train_tokenizer
from ai_tester.core.trainer import train_model

# Train tokenizer on tool use data
tokenizer = train_tokenizer(
    data_paths=["data/tool_training_data.txt"],
    vocab_size=8000,
    output_path="models/tokenizer.json"
)

# Train model
train_model(
    data_path="data/tool_training_data.txt",
    model_size="small",
    epochs=30
)
```

### 2. Enabling Tool Execution

Initialize the inference engine with tool support:

```python
from ai_tester.core.inference import AITesterEngine
from ai_tester.modules import ModuleManager

# Initialize module manager
manager = ModuleManager()

# Load modules you want to use
manager.load('model')
manager.load('tokenizer')
manager.load('image_gen_local')  # For image generation
manager.load('vision')            # For image analysis
manager.load('avatar')            # For avatar control

# Create inference engine with tools enabled
engine = AITesterEngine(
    model_path="models/enigma.pth",
    enable_tools=True,
    module_manager=manager
)
```

### 3. Using Tools in Conversation

The AI will automatically use tools when appropriate:

```python
# User asks for image generation
response = engine.generate(
    "Can you generate an image of a sunset over mountains?",
    max_gen=200,
    execute_tools=True  # Enable tool execution
)

print(response)
# Output:
# I'll create that image for you.
# <tool_call>{"tool": "generate_image", ...}</tool_call>
# <tool_result>{"success": true, ...}</tool_result>
# Done! I've generated a beautiful sunset image...
```

### 4. Manual Tool Execution

You can also execute tools directly:

```python
from ai_tester.tools import ToolExecutor

executor = ToolExecutor(module_manager=manager)

# Execute a tool
result = executor.execute_tool(
    "generate_image",
    {"prompt": "a cat wearing a hat", "width": 512, "height": 512}
)

print(result)
# {"success": true, "result": "Image generated...", "output_path": "..."}
```

## Adding New Tools

### 1. Define the Tool

Add to `enigma/tools/tool_definitions.py`:

```python
MY_TOOL = ToolDefinition(
    name="my_tool",
    description="What my tool does",
    category="generation",  # or "perception", "control", "system"
    module="my_module",     # Module that provides this
    parameters=[
        ToolParameter(
            name="param1",
            type="string",
            description="What this parameter does",
            required=True,
        ),
    ],
    examples=[
        "Example usage of my tool",
    ],
)

# Add to ALL_TOOLS list
ALL_TOOLS = [
    # ... existing tools ...
    MY_TOOL,
]
```

### 2. Implement Execution

Add to `enigma/tools/tool_executor.py`:

```python
def _execute_my_tool(self, module, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute my custom tool."""
    try:
        param1 = params.get("param1", "")
        
        # Call module's method
        if hasattr(module, "my_method"):
            result = module.my_method(param1=param1)
        else:
            return {
                "success": False,
                "error": "Module does not have my_method()",
                "tool": "my_tool",
            }
        
        return {
            "success": True,
            "result": str(result),
            "tool": "my_tool",
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool": "my_tool",
        }
```

### 3. Add Training Examples

Add to `data/tool_training_data.txt`:

```
Q: Use my tool
A: I'll use my tool for you.
<tool_call>{"tool": "my_tool", "params": {"param1": "value"}}</tool_call>
<tool_result>{"tool": "my_tool", "success": true, "result": "..."}</tool_result>
Done! My tool has been executed successfully.
```

### 4. Retrain the Model

Train the model with the updated data so it learns to use the new tool.

## Configuration

### Tool Execution Settings

Control tool execution behavior:

```python
engine.generate(
    prompt="Your prompt",
    execute_tools=True,        # Enable/disable tool execution
    max_tool_iterations=5,     # Maximum tool execution loops
)
```

### Tool-Specific Settings

Configure individual tools through modules:

```python
# Example: Configure image generation
manager.configure('image_gen_local', {
    'default_steps': 30,
    'default_size': 768,
})
```

## Best Practices

### 1. Training Data Quality

- Provide diverse examples of tool usage
- Include error handling examples
- Show multi-step tool chains
- Use natural language in responses

### 2. Tool Design

- Keep parameters simple and clear
- Provide good default values
- Return structured, parseable results
- Handle errors gracefully

### 3. Performance

- Tools are executed sequentially
- Each tool execution adds latency
- Consider caching results when appropriate
- Limit `max_tool_iterations` to prevent loops

### 4. Safety

- Validate all tool parameters
- Implement proper error handling
- Consider access controls for sensitive tools
- Log all tool executions for auditing

## Troubleshooting

### Tools Not Being Called

1. **Check training**: Ensure model was trained on tool use examples
2. **Verify modules**: Confirm required modules are loaded
3. **Enable tools**: Set `enable_tools=True` in AITesterEngine
4. **Check prompt**: Include clear requests that require tools

### Tool Execution Errors

1. **Module not loaded**: Load the required module first
2. **Invalid parameters**: Check parameter types and values
3. **Module method missing**: Ensure module implements required methods
4. **Permissions**: Verify file/system permissions for system tools

### Poor Tool Usage

1. **Insufficient training**: Train longer or with more examples
2. **Ambiguous prompts**: Use clearer, more specific prompts
3. **Missing examples**: Add training data for specific use cases
4. **Wrong tool chosen**: Add more examples distinguishing tools

## Examples

See `data/tool_training_data.txt` for comprehensive examples of:
- Image generation
- Vision/image analysis
- Avatar control
- Text-to-speech
- Code generation
- File operations
- Web search
- Multi-step tool chains
- Error handling

## API Reference

### ToolExecutor

```python
class ToolExecutor:
    def __init__(self, module_manager=None)
    def parse_tool_calls(self, text: str) -> List[Tuple]
    def validate_params(self, tool_name: str, params: Dict) -> Tuple
    def execute_tool(self, tool_name: str, params: Dict) -> Dict
    def format_tool_result(self, result: Dict) -> str
```

### AITesterEngine (Tool Support)

```python
class AITesterEngine:
    def __init__(
        self,
        ...,
        enable_tools: bool = False,
        module_manager: Optional[Any] = None
    )
    
    def generate(
        self,
        ...,
        execute_tools: bool = None,
        max_tool_iterations: int = 5
    ) -> str
```

## Future Enhancements

Planned improvements:
- Parallel tool execution
- Tool result caching
- Conditional tool execution
- Tool authorization/permissions
- Tool usage analytics
- Streaming tool execution
- Tool composition (tools calling tools)

## Contributing

To contribute new tools:
1. Define the tool in `tool_definitions.py`
2. Implement execution in `tool_executor.py`
3. Add training examples in `data/tool_training_data.txt`
4. Update this documentation
5. Add tests in `tests/test_tool_use.py`

---

For more information, see:
- `enigma/tools/tool_definitions.py` - Tool schemas
- `enigma/tools/tool_executor.py` - Execution logic
- `enigma/core/inference.py` - Integration with AI
- `data/tool_training_data.txt` - Training examples
