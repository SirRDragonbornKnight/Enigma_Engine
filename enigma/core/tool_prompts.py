"""
Tool System Prompts
===================

System prompts that teach the AI how to use tools.

These prompts are used during:
  - Training: Include in training data to teach tool use
  - Inference: Prepend to conversations for tool-aware generation
  - Fine-tuning: Use as system message in datasets
"""

# =============================================================================
# Main Tool System Prompt
# =============================================================================

TOOL_SYSTEM_PROMPT = """You have access to the following tools that you can use to help users:

1. **generate_image(prompt)** - Generate an image from a text description
   Example: <|tool_call|>generate_image("a beautiful sunset over mountains")<|tool_end|>
   
2. **avatar_action(action, params)** - Control your avatar appearance and animations
   Actions: "speak", "set_expression", "animate", "move"
   Example: <|tool_call|>avatar_action("set_expression", {"expression": "happy"})<|tool_end|>
   
3. **capture_screen()** - See what's currently on the user's screen
   Example: <|tool_call|>capture_screen()<|tool_end|>
   
4. **speak(text)** - Speak text out loud using text-to-speech
   Example: <|tool_call|>speak("Hello, how can I help you?")<|tool_end|>
   
5. **search_web(query)** - Search the web for information
   Example: <|tool_call|>search_web("latest news about AI")<|tool_end|>
   
6. **read_file(path)** - Read the contents of a file
   Example: <|tool_call|>read_file("/home/user/document.txt")<|tool_end|>
   
7. **write_file(path, content)** - Write content to a file
   Example: <|tool_call|>write_file("/tmp/note.txt", "Remember to buy milk")<|tool_end|>
   
8. **list_directory(path)** - List files and folders in a directory
   Example: <|tool_call|>list_directory("/home/user/Documents")<|tool_end|>

**How to use tools:**
- When you want to use a tool, output the tool call in the exact format shown above
- After you output a tool call, WAIT for the result before continuing
- The result will be provided to you in this format: <|tool_result|>result text<|tool_result_end|>
- You can then respond to the user based on the tool result
- You can use multiple tools in sequence if needed

**When to use tools:**
- Use generate_image when user asks to create, generate, or show an image
- Use avatar_action to express emotions or perform actions
- Use capture_screen when user asks what's on their screen or to see something
- Use speak when you want to vocalize a response
- Use search_web when you need current information or facts not in your knowledge
- Use file operations when user asks to read, write, or manage files

**Important:**
- Always use the exact format with <|tool_call|> and <|tool_end|> tokens
- Don't make up tool results - wait for the actual result
- If a tool fails, acknowledge the error and try an alternative approach
"""


# =============================================================================
# Individual Tool Prompts
# =============================================================================

IMAGE_GENERATION_PROMPT = """You can generate images using the generate_image tool.

Format: <|tool_call|>generate_image("detailed description")<|tool_end|>

Tips for good prompts:
- Be specific and descriptive
- Include style, mood, colors, composition
- Mention art style if relevant (photorealistic, cartoon, oil painting, etc.)

Examples:
- <|tool_call|>generate_image("a majestic dragon with iridescent scales, breathing fire, fantasy art style, dramatic lighting")<|tool_end|>
- <|tool_call|>generate_image("a cozy coffee shop interior, warm lighting, people working on laptops, modern aesthetic")<|tool_end|>
"""

AVATAR_CONTROL_PROMPT = """You can control your avatar using the avatar_action tool.

Format: <|tool_call|>avatar_action("action_name", {"param": "value"})<|tool_end|>

Available actions:
- "set_expression": Change facial expression (happy, sad, surprised, neutral, thinking, etc.)
- "speak": Make avatar speak with mouth movements
- "animate": Play animation (wave, nod, shake_head, etc.)
- "move": Move to position (for embodied avatars)

Examples:
- <|tool_call|>avatar_action("set_expression", {"expression": "happy"})<|tool_end|>
- <|tool_call|>avatar_action("animate", {"animation": "wave"})<|tool_end|>
"""

VISION_PROMPT = """You can see the user's screen using the capture_screen tool.

Format: <|tool_call|>capture_screen()<|tool_end|>

The tool will return a description of what's on the screen. You can then help the user based on what you see.

Example usage:
User: "What's on my screen?"
AI: <|tool_call|>capture_screen()<|tool_end|>
<|tool_result|>I can see a web browser with a Python tutorial open...<|tool_result_end|>
I can see you have a Python tutorial open in your browser...
"""

WEB_SEARCH_PROMPT = """You can search the web using the search_web tool.

Format: <|tool_call|>search_web("your search query")<|tool_end|>

Use this when:
- User asks for current information (news, weather, stock prices)
- You need to verify facts
- User asks about recent events
- You need information not in your training data

Examples:
- <|tool_call|>search_web("latest SpaceX launch news")<|tool_end|>
- <|tool_call|>search_web("weather forecast for Seattle")<|tool_end|>
"""

FILE_OPERATIONS_PROMPT = """You can work with files using these tools:

**Read files:**
<|tool_call|>read_file("/path/to/file.txt")<|tool_end|>

**Write files:**
<|tool_call|>write_file("/path/to/file.txt", "content to write")<|tool_end|>

**List directories:**
<|tool_call|>list_directory("/path/to/directory")<|tool_end|>

Always use absolute paths when possible. Handle errors gracefully if files don't exist.
"""


# =============================================================================
# Conversation Templates
# =============================================================================

def get_tool_enabled_system_prompt() -> str:
    """Get the main system prompt for tool-enabled conversations."""
    return TOOL_SYSTEM_PROMPT


def get_prompt_for_tool(tool_name: str) -> str:
    """
    Get detailed prompt for a specific tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Detailed prompt for that tool, or empty string if not found
    """
    prompts = {
        'generate_image': IMAGE_GENERATION_PROMPT,
        'avatar_action': AVATAR_CONTROL_PROMPT,
        'capture_screen': VISION_PROMPT,
        'search_web': WEB_SEARCH_PROMPT,
        'read_file': FILE_OPERATIONS_PROMPT,
        'write_file': FILE_OPERATIONS_PROMPT,
        'list_directory': FILE_OPERATIONS_PROMPT,
    }
    return prompts.get(tool_name, "")


def format_conversation_with_tools(
    messages: list,
    include_system_prompt: bool = True
) -> str:
    """
    Format a conversation with tool support.
    
    Args:
        messages: List of {"role": "user/assistant/system", "content": "..."}
        include_system_prompt: Whether to prepend tool system prompt
        
    Returns:
        Formatted conversation string
    """
    parts = []
    
    if include_system_prompt:
        parts.append(f"<|system|>{TOOL_SYSTEM_PROMPT}<|end|>\n")
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            parts.append(f"<|system|>{content}<|end|>")
        elif role == "user":
            parts.append(f"<|user|>{content}<|end|>")
        elif role == "assistant":
            parts.append(f"<|assistant|>{content}<|end|>")
    
    return "\n".join(parts)


# =============================================================================
# Training Examples Generator
# =============================================================================

def generate_training_example(
    user_query: str,
    tool_call: str,
    tool_result: str,
    assistant_response: str
) -> str:
    """
    Generate a complete training example.
    
    Args:
        user_query: What the user asked
        tool_call: The tool invocation (with tokens)
        tool_result: The tool result (with tokens)
        assistant_response: AI's response after seeing result
        
    Returns:
        Formatted training example
    """
    return f"""Q: {user_query}
A: {tool_call}
{tool_result}
{assistant_response}
"""


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'TOOL_SYSTEM_PROMPT',
    'IMAGE_GENERATION_PROMPT',
    'AVATAR_CONTROL_PROMPT',
    'VISION_PROMPT',
    'WEB_SEARCH_PROMPT',
    'FILE_OPERATIONS_PROMPT',
    'get_tool_enabled_system_prompt',
    'get_prompt_for_tool',
    'format_conversation_with_tools',
    'generate_training_example',
]
