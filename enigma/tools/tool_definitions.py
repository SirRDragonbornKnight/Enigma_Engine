"""
Tool Definitions for Enigma AI
===============================

Defines all available tools that the AI can use, including their schemas,
parameters, and which modules provide them.

This allows the AI to:
- Know what tools are available
- Understand what each tool does
- Know what parameters each tool requires
- Execute tools through the module system
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "int", "float", "bool", "list", "dict"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None  # For choice parameters


@dataclass
class ToolDefinition:
    """Definition of a tool that the AI can use."""
    name: str
    description: str
    parameters: List[ToolParameter]
    module: Optional[str] = None  # Module that provides this capability
    category: str = "general"  # "generation", "perception", "control", "system", "general"
    examples: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for AI consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                p.name: {
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum,
                }
                for p in self.parameters
            },
            "category": self.category,
        }
    
    def get_schema(self) -> str:
        """Get human-readable schema for the tool."""
        params = []
        for p in self.parameters:
            req = "required" if p.required else "optional"
            default = f" (default: {p.default})" if p.default is not None else ""
            params.append(f"  - {p.name} ({p.type}, {req}){default}: {p.description}")
        
        return f"{self.name}\n  {self.description}\nParameters:\n" + "\n".join(params)


# =============================================================================
# Tool Definitions
# =============================================================================

# --- Image Generation Tools ---

GENERATE_IMAGE = ToolDefinition(
    name="generate_image",
    description="Generate an image from a text description using AI image generation",
    category="generation",
    module="image_gen_local",  # or image_gen_api
    parameters=[
        ToolParameter(
            name="prompt",
            type="string",
            description="Detailed description of the image to generate",
            required=True,
        ),
        ToolParameter(
            name="width",
            type="int",
            description="Width of the image in pixels",
            required=False,
            default=512,
        ),
        ToolParameter(
            name="height",
            type="int",
            description="Height of the image in pixels",
            required=False,
            default=512,
        ),
        ToolParameter(
            name="steps",
            type="int",
            description="Number of inference steps (more = higher quality)",
            required=False,
            default=20,
        ),
    ],
    examples=[
        "Generate an image of a sunset over mountains",
        "Create a picture of a cat wearing a wizard hat",
        "Make an image of a futuristic city at night",
    ],
)

# --- Vision/Image Analysis Tools ---

ANALYZE_IMAGE = ToolDefinition(
    name="analyze_image",
    description="Analyze an image and describe what's in it using computer vision",
    category="perception",
    module="vision",
    parameters=[
        ToolParameter(
            name="image_path",
            type="string",
            description="Path to the image file to analyze",
            required=True,
        ),
        ToolParameter(
            name="detail_level",
            type="string",
            description="Level of detail in description",
            required=False,
            default="normal",
            enum=["brief", "normal", "detailed"],
        ),
    ],
    examples=[
        "What's in this image?",
        "Analyze the uploaded image",
        "Describe what you see in this picture",
    ],
)

FIND_ON_SCREEN = ToolDefinition(
    name="find_on_screen",
    description="Find and locate specific elements on the screen using vision",
    category="perception",
    module="vision",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="What to look for on screen (e.g., 'the Save button', 'text that says Menu')",
            required=True,
        ),
    ],
    examples=[
        "Find the Submit button on screen",
        "Locate the text that says 'Welcome'",
    ],
)

# --- Avatar Control Tools ---

CONTROL_AVATAR = ToolDefinition(
    name="control_avatar",
    description="Control the avatar's expression, animation, or behavior",
    category="control",
    module="avatar",
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="Action to perform",
            required=True,
            enum=["expression", "animation", "gesture", "speak"],
        ),
        ToolParameter(
            name="value",
            type="string",
            description="Value for the action (e.g., 'happy', 'wave', 'nod')",
            required=True,
        ),
    ],
    examples=[
        "Make the avatar smile",
        "Have the avatar wave",
        "Set avatar expression to surprised",
    ],
)

# --- Text-to-Speech Tools ---

SPEAK = ToolDefinition(
    name="speak",
    description="Speak text out loud using text-to-speech",
    category="control",
    module="voice_output",
    parameters=[
        ToolParameter(
            name="text",
            type="string",
            description="Text to speak aloud",
            required=True,
        ),
        ToolParameter(
            name="voice",
            type="string",
            description="Voice to use",
            required=False,
            default="default",
        ),
    ],
    examples=[
        "Say 'Hello there!'",
        "Read this text aloud",
        "Speak the answer",
    ],
)

# --- Code Generation Tools ---

GENERATE_CODE = ToolDefinition(
    name="generate_code",
    description="Generate code in a specific programming language",
    category="generation",
    module="code_gen_local",  # or code_gen_api
    parameters=[
        ToolParameter(
            name="description",
            type="string",
            description="What the code should do",
            required=True,
        ),
        ToolParameter(
            name="language",
            type="string",
            description="Programming language",
            required=False,
            default="python",
            enum=["python", "javascript", "java", "cpp", "go", "rust"],
        ),
    ],
    examples=[
        "Write a Python function to calculate fibonacci numbers",
        "Generate JavaScript code to validate an email address",
    ],
)

# --- File Operation Tools ---

READ_FILE = ToolDefinition(
    name="read_file",
    description="Read the contents of a file",
    category="system",
    module=None,  # Built-in tool
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to read",
            required=True,
        ),
    ],
    examples=[
        "Read the file README.md",
        "Show me the contents of config.json",
    ],
)

WRITE_FILE = ToolDefinition(
    name="write_file",
    description="Write content to a file",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to write",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Content to write to the file",
            required=True,
        ),
    ],
    examples=[
        "Save this text to notes.txt",
        "Write the code to main.py",
    ],
)

LIST_DIRECTORY = ToolDefinition(
    name="list_directory",
    description="List files and directories in a path",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Directory path to list",
            required=False,
            default=".",
        ),
    ],
    examples=[
        "List files in the current directory",
        "Show me what's in the Documents folder",
    ],
)

# --- Web Tools ---

WEB_SEARCH = ToolDefinition(
    name="web_search",
    description="Search the internet for information",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Search query",
            required=True,
        ),
        ToolParameter(
            name="num_results",
            type="int",
            description="Number of results to return",
            required=False,
            default=5,
        ),
    ],
    examples=[
        "Search for Python tutorials",
        "Look up the weather forecast",
    ],
)

FETCH_WEBPAGE = ToolDefinition(
    name="fetch_webpage",
    description="Fetch and extract content from a webpage",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="URL of the webpage",
            required=True,
        ),
    ],
    examples=[
        "Get the content from https://example.com",
        "Fetch the webpage",
    ],
)

# --- Video Generation Tools ---

GENERATE_VIDEO = ToolDefinition(
    name="generate_video",
    description="Generate a video from a text description or animate an image",
    category="generation",
    module="video_gen_local",  # or video_gen_api
    parameters=[
        ToolParameter(
            name="prompt",
            type="string",
            description="Description of the video to generate",
            required=True,
        ),
        ToolParameter(
            name="duration",
            type="float",
            description="Duration in seconds",
            required=False,
            default=3.0,
        ),
        ToolParameter(
            name="fps",
            type="int",
            description="Frames per second",
            required=False,
            default=24,
        ),
    ],
    examples=[
        "Generate a video of waves crashing on a beach",
        "Create an animation of a spinning cube",
    ],
)

# --- Audio Generation Tools ---

GENERATE_AUDIO = ToolDefinition(
    name="generate_audio",
    description="Generate audio or music from a text description",
    category="generation",
    module="audio_gen_local",  # or audio_gen_api
    parameters=[
        ToolParameter(
            name="prompt",
            type="string",
            description="Description of the audio to generate",
            required=True,
        ),
        ToolParameter(
            name="duration",
            type="float",
            description="Duration in seconds",
            required=False,
            default=5.0,
        ),
    ],
    examples=[
        "Generate the sound of rain",
        "Create piano music",
    ],
)

# --- GIF Generation Tools ---

GENERATE_GIF = ToolDefinition(
    name="generate_gif",
    description="Generate an animated GIF from a list of image prompts",
    category="generation",
    module="image_gen_local",  # or image_gen_api
    parameters=[
        ToolParameter(
            name="frames",
            type="list",
            description="List of text prompts for each frame of the GIF",
            required=True,
        ),
        ToolParameter(
            name="fps",
            type="int",
            description="Frames per second for the GIF animation",
            required=False,
            default=5,
        ),
        ToolParameter(
            name="loop",
            type="int",
            description="Number of times to loop (0 = infinite loop)",
            required=False,
            default=0,
        ),
        ToolParameter(
            name="width",
            type="int",
            description="Width of each frame in pixels",
            required=False,
            default=512,
        ),
        ToolParameter(
            name="height",
            type="int",
            description="Height of each frame in pixels",
            required=False,
            default=512,
        ),
    ],
    examples=[
        "Create a GIF showing sunrise, noon, and sunset",
        "Generate an animated GIF of a cat walking",
        "Make a GIF of a flower blooming",
    ],
)

# --- Media Editing Tools ---

EDIT_IMAGE = ToolDefinition(
    name="edit_image",
    description="Edit an existing image with various transformations and enhancements",
    category="generation",
    module=None,  # Built-in tool using Pillow
    parameters=[
        ToolParameter(
            name="image_path",
            type="string",
            description="Path to the image file to edit",
            required=True,
        ),
        ToolParameter(
            name="edit_type",
            type="string",
            description="Type of edit to perform",
            required=True,
            enum=["resize", "rotate", "flip", "brightness", "contrast", "blur", "sharpen", "grayscale", "crop"],
        ),
        ToolParameter(
            name="width",
            type="int",
            description="New width for resize operation",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="height",
            type="int",
            description="New height for resize operation",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="angle",
            type="int",
            description="Rotation angle in degrees",
            required=False,
            default=0,
        ),
        ToolParameter(
            name="direction",
            type="string",
            description="Flip direction: 'horizontal' or 'vertical'",
            required=False,
            enum=["horizontal", "vertical"],
        ),
        ToolParameter(
            name="factor",
            type="float",
            description="Adjustment factor (e.g., 1.5 for brightness, 0.5-2.0 range)",
            required=False,
            default=1.0,
        ),
        ToolParameter(
            name="crop_box",
            type="list",
            description="Crop coordinates [left, top, right, bottom]",
            required=False,
            default=None,
        ),
    ],
    examples=[
        "Resize the image to 800x600",
        "Rotate the image 90 degrees",
        "Increase the brightness of the image",
        "Apply blur to the image",
    ],
)

EDIT_GIF = ToolDefinition(
    name="edit_gif",
    description="Edit an existing GIF animation (speed, reverse, crop, etc.)",
    category="generation",
    module=None,  # Built-in tool using Pillow
    parameters=[
        ToolParameter(
            name="gif_path",
            type="string",
            description="Path to the GIF file to edit",
            required=True,
        ),
        ToolParameter(
            name="edit_type",
            type="string",
            description="Type of edit to perform",
            required=True,
            enum=["speed", "reverse", "crop", "resize", "extract_frames"],
        ),
        ToolParameter(
            name="speed_factor",
            type="float",
            description="Speed multiplier (2.0 = 2x faster, 0.5 = half speed)",
            required=False,
            default=1.0,
        ),
        ToolParameter(
            name="width",
            type="int",
            description="New width for resize operation",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="height",
            type="int",
            description="New height for resize operation",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="crop_box",
            type="list",
            description="Crop coordinates [left, top, right, bottom]",
            required=False,
            default=None,
        ),
    ],
    examples=[
        "Make the GIF play twice as fast",
        "Reverse the GIF animation",
        "Resize the GIF to 400x400",
    ],
)

EDIT_VIDEO = ToolDefinition(
    name="edit_video",
    description="Edit an existing video file (trim, speed, extract frames, etc.)",
    category="generation",
    module=None,  # Built-in tool, requires moviepy
    parameters=[
        ToolParameter(
            name="video_path",
            type="string",
            description="Path to the video file to edit",
            required=True,
        ),
        ToolParameter(
            name="edit_type",
            type="string",
            description="Type of edit to perform",
            required=True,
            enum=["trim", "speed", "extract_frames", "resize", "to_gif"],
        ),
        ToolParameter(
            name="start_time",
            type="float",
            description="Start time in seconds for trim operation",
            required=False,
            default=0.0,
        ),
        ToolParameter(
            name="end_time",
            type="float",
            description="End time in seconds for trim operation",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="speed_factor",
            type="float",
            description="Speed multiplier (2.0 = 2x faster, 0.5 = half speed)",
            required=False,
            default=1.0,
        ),
        ToolParameter(
            name="width",
            type="int",
            description="New width for resize operation",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="height",
            type="int",
            description="New height for resize operation",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="fps",
            type="int",
            description="Frames per second for extract_frames or to_gif operations",
            required=False,
            default=10,
        ),
    ],
    examples=[
        "Trim the video from 10 to 30 seconds",
        "Convert video to GIF",
        "Extract frames from the video",
        "Speed up the video 2x",
    ],
)


# =============================================================================
# Registry of All Tools
# =============================================================================

ALL_TOOLS = [
    # Generation
    GENERATE_IMAGE,
    GENERATE_GIF,
    GENERATE_CODE,
    GENERATE_VIDEO,
    GENERATE_AUDIO,
    
    # Editing
    EDIT_IMAGE,
    EDIT_GIF,
    EDIT_VIDEO,
    
    # Perception
    ANALYZE_IMAGE,
    FIND_ON_SCREEN,
    
    # Control
    CONTROL_AVATAR,
    SPEAK,
    
    # System
    READ_FILE,
    WRITE_FILE,
    LIST_DIRECTORY,
    WEB_SEARCH,
    FETCH_WEBPAGE,
]

# Dictionary for fast lookup
TOOLS_BY_NAME: Dict[str, ToolDefinition] = {
    tool.name: tool for tool in ALL_TOOLS
}

TOOLS_BY_CATEGORY: Dict[str, List[ToolDefinition]] = {}
for tool in ALL_TOOLS:
    if tool.category not in TOOLS_BY_CATEGORY:
        TOOLS_BY_CATEGORY[tool.category] = []
    TOOLS_BY_CATEGORY[tool.category].append(tool)


# =============================================================================
# Helper Functions
# =============================================================================

def get_tool_definition(tool_name: str) -> Optional[ToolDefinition]:
    """Get tool definition by name."""
    return TOOLS_BY_NAME.get(tool_name)


def get_all_tools() -> List[ToolDefinition]:
    """Get all tool definitions."""
    return ALL_TOOLS


def get_tools_by_category(category: str) -> List[ToolDefinition]:
    """Get all tools in a category."""
    return TOOLS_BY_CATEGORY.get(category, [])


def get_tool_schemas() -> str:
    """Get all tool schemas as a formatted string."""
    schemas = []
    for category, tools in sorted(TOOLS_BY_CATEGORY.items()):
        schemas.append(f"\n=== {category.upper()} TOOLS ===\n")
        for tool in tools:
            schemas.append(tool.get_schema())
            schemas.append("")
    return "\n".join(schemas)


def get_available_tools_for_prompt() -> str:
    """
    Get a formatted description of available tools for including in AI prompts.
    
    Returns:
        Formatted string describing all available tools
    """
    lines = [
        "AVAILABLE TOOLS",
        "=" * 80,
        "",
        "You have access to the following tools. To use a tool, output:",
        "<tool_call>",
        '{"tool": "tool_name", "params": {"param1": "value1", "param2": "value2"}}',
        "</tool_call>",
        "",
        "The system will execute the tool and provide results as:",
        "<tool_result>",
        '{"tool": "tool_name", "success": true, "result": "..."}',
        "</tool_result>",
        "",
        "=" * 80,
        "",
    ]
    
    for category, tools in sorted(TOOLS_BY_CATEGORY.items()):
        lines.append(f"\n{category.upper()} TOOLS:")
        lines.append("-" * 40)
        
        for tool in tools:
            lines.append(f"\n{tool.name}:")
            lines.append(f"  {tool.description}")
            lines.append("  Parameters:")
            for param in tool.parameters:
                req = "*required*" if param.required else "optional"
                default = f" = {param.default}" if param.default is not None else ""
                lines.append(f"    - {param.name} ({param.type}, {req}){default}")
                lines.append(f"      {param.description}")
            if tool.examples:
                lines.append("  Examples:")
                for ex in tool.examples[:2]:  # Limit to 2 examples
                    lines.append(f"    - {ex}")
    
    return "\n".join(lines)


__all__ = [
    "ToolDefinition",
    "ToolParameter",
    "get_tool_definition",
    "get_all_tools",
    "get_tools_by_category",
    "get_tool_schemas",
    "get_available_tools_for_prompt",
    "TOOLS_BY_NAME",
    "TOOLS_BY_CATEGORY",
]
