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
    version: str = "1.0.0"  # Tool version (semantic versioning)
    deprecated: bool = False  # Whether tool is deprecated
    deprecated_message: Optional[str] = None  # Deprecation message
    added_in: Optional[str] = None  # Version when tool was added
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for AI consumption."""
        result = {
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
            "version": self.version,
        }
        
        if self.deprecated:
            result["deprecated"] = True
            if self.deprecated_message:
                result["deprecated_message"] = self.deprecated_message
        
        if self.added_in:
            result["added_in"] = self.added_in
        
        return result
    
    def get_schema(self) -> str:
        """Get human-readable schema for the tool."""
        params = []
        for p in self.parameters:
            req = "required" if p.required else "optional"
            default = f" (default: {p.default})" if p.default is not None else ""
            params.append(f"  - {p.name} ({p.type}, {req}){default}: {p.description}")
        
        schema = f"{self.name} (v{self.version})"
        if self.deprecated:
            schema += " [DEPRECATED]"
        schema += f"\n  {self.description}\nParameters:\n" + "\n".join(params)
        
        if self.deprecated and self.deprecated_message:
            schema += f"\n  NOTE: {self.deprecated_message}"
        
        return schema
    
    def is_compatible(self, required_version: str) -> bool:
        """
        Check if tool version is compatible with required version.
        
        Uses semantic versioning: MAJOR.MINOR.PATCH
        - Major version must match
        - Minor version must be >= required
        - Patch version doesn't matter for compatibility
        
        Args:
            required_version: Required version string (e.g., "1.2.0")
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            # Parse versions
            current_parts = [int(x) for x in self.version.split('.')]
            required_parts = [int(x) for x in required_version.split('.')]
            
            # Pad with zeros if needed
            while len(current_parts) < 3:
                current_parts.append(0)
            while len(required_parts) < 3:
                required_parts.append(0)
            
            current_major, current_minor, _ = current_parts[:3]
            required_major, required_minor, _ = required_parts[:3]
            
            # Major version must match
            if current_major != required_major:
                return False
            
            # Minor version must be >= required
            if current_minor < required_minor:
                return False
            
            return True
        
        except (ValueError, IndexError):
            # If parsing fails, assume compatible
            logger.warning(f"Failed to parse versions: {self.version}, {required_version}")
            return True


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
        ToolParameter(
            name="max_lines",
            type="int",
            description="Maximum number of lines to read",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="encoding",
            type="string",
            description="File encoding (default: utf-8)",
            required=False,
            default="utf-8",
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
        ToolParameter(
            name="mode",
            type="string",
            description="Write mode: 'overwrite' or 'append'",
            required=False,
            default="overwrite",
            enum=["overwrite", "append"],
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

MOVE_FILE = ToolDefinition(
    name="move_file",
    description="Move a file to a new location or rename it",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="source",
            type="string",
            description="Source file path",
            required=True,
        ),
        ToolParameter(
            name="destination",
            type="string",
            description="Destination file path",
            required=True,
        ),
    ],
    examples=[
        "Move file.txt to backup folder",
        "Rename old.txt to new.txt",
    ],
)

DELETE_FILE = ToolDefinition(
    name="delete_file",
    description="Delete a file (requires confirmation)",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to delete",
            required=True,
        ),
        ToolParameter(
            name="confirm",
            type="string",
            description="Set to 'yes' to confirm deletion",
            required=False,
            default="no",
        ),
    ],
    examples=[
        "Delete temp.txt",
        "Remove the old backup file",
    ],
)

# --- Document Tools ---

READ_DOCUMENT = ToolDefinition(
    name="read_document",
    description="Read a document file (txt, pdf, epub, docx, html, md)",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the document file",
            required=True,
        ),
    ],
    examples=[
        "Read the PDF document",
        "Open the Word document",
    ],
)

EXTRACT_TEXT = ToolDefinition(
    name="extract_text",
    description="Extract plain text from a file, removing formatting",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file",
            required=True,
        ),
        ToolParameter(
            name="max_chars",
            type="int",
            description="Maximum characters to extract",
            required=False,
            default=None,
        ),
    ],
    examples=[
        "Extract text from document.pdf",
        "Get the text content from this file",
    ],
)

# --- System Tools ---

GET_SYSTEM_INFO = ToolDefinition(
    name="get_system_info",
    description="Get system information including OS, CPU, memory, disk, and GPU details",
    category="system",
    module=None,
    parameters=[],
    examples=[
        "What are my system specs?",
        "Check my computer info",
        "How much RAM do I have?",
    ],
)

RUN_COMMAND = ToolDefinition(
    name="run_command",
    description="Execute a shell command and return the output",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="command",
            type="string",
            description="Shell command to execute",
            required=True,
        ),
        ToolParameter(
            name="timeout",
            type="int",
            description="Maximum seconds to wait",
            required=False,
            default=30,
        ),
        ToolParameter(
            name="cwd",
            type="string",
            description="Working directory",
            required=False,
            default=None,
        ),
    ],
    examples=[
        "Run 'python --version'",
        "Execute 'dir' command",
        "Run a shell command",
    ],
)

SCREENSHOT = ToolDefinition(
    name="screenshot",
    description="Take a screenshot and save it to a file",
    category="perception",
    module=None,
    parameters=[
        ToolParameter(
            name="output_path",
            type="string",
            description="Path to save the screenshot",
            required=False,
            default="screenshot.png",
        ),
        ToolParameter(
            name="region",
            type="string",
            description="Region as 'x,y,width,height' (optional, default: full screen)",
            required=False,
            default=None,
        ),
    ],
    examples=[
        "Take a screenshot",
        "Capture the screen",
    ],
)

SEE_SCREEN = ToolDefinition(
    name="see_screen",
    description="Look at the screen and describe what's visible",
    category="perception",
    module=None,
    parameters=[],
    examples=[
        "What's on my screen?",
        "Look at my screen",
        "What do you see?",
    ],
)

# --- Interactive/Task Management Tools ---

CREATE_CHECKLIST = ToolDefinition(
    name="create_checklist",
    description="Create a new checklist with items",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="name",
            type="string",
            description="Name of the checklist",
            required=True,
        ),
        ToolParameter(
            name="items",
            type="list",
            description="List of checklist items",
            required=True,
        ),
    ],
    examples=[
        "Create a shopping list",
        "Make a todo checklist",
    ],
)

LIST_CHECKLISTS = ToolDefinition(
    name="list_checklists",
    description="List all created checklists and their status",
    category="system",
    module=None,
    parameters=[],
    examples=[
        "Show my checklists",
        "What checklists do I have?",
    ],
)

ADD_TASK = ToolDefinition(
    name="add_task",
    description="Add a task with optional due date and priority",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="title",
            type="string",
            description="Task title",
            required=True,
        ),
        ToolParameter(
            name="description",
            type="string",
            description="Task description",
            required=False,
            default="",
        ),
        ToolParameter(
            name="due_date",
            type="string",
            description="Due date in ISO format",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="priority",
            type="string",
            description="Priority level",
            required=False,
            default="medium",
            enum=["low", "medium", "high"],
        ),
    ],
    examples=[
        "Add a task to call mom",
        "Create a high priority task",
    ],
)

LIST_TASKS = ToolDefinition(
    name="list_tasks",
    description="List all tasks with optional filtering",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="show_completed",
            type="bool",
            description="Whether to show completed tasks",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="priority",
            type="string",
            description="Filter by priority",
            required=False,
            default=None,
            enum=["low", "medium", "high"],
        ),
    ],
    examples=[
        "Show my tasks",
        "List high priority tasks",
    ],
)

COMPLETE_TASK = ToolDefinition(
    name="complete_task",
    description="Mark a task as complete",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="task_id",
            type="string",
            description="The ID of the task to complete",
            required=True,
        ),
    ],
    examples=[
        "Complete task_1",
        "Mark task as done",
    ],
)

SET_REMINDER = ToolDefinition(
    name="set_reminder",
    description="Set a reminder for a specific time",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="message",
            type="string",
            description="Reminder message",
            required=True,
        ),
        ToolParameter(
            name="remind_at",
            type="string",
            description="When to remind in ISO format",
            required=True,
        ),
        ToolParameter(
            name="repeat",
            type="string",
            description="Repeat frequency",
            required=False,
            default=None,
            enum=["daily", "weekly", "monthly"],
        ),
    ],
    examples=[
        "Remind me at 3pm",
        "Set a reminder for tomorrow",
    ],
)

LIST_REMINDERS = ToolDefinition(
    name="list_reminders",
    description="List all active reminders",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="active_only",
            type="bool",
            description="Show only active reminders",
            required=False,
            default=True,
        ),
    ],
    examples=[
        "Show my reminders",
        "What reminders do I have?",
    ],
)

CHECK_REMINDERS = ToolDefinition(
    name="check_reminders",
    description="Check for reminders that are due right now",
    category="system",
    module=None,
    parameters=[],
    examples=[
        "Any reminders due?",
        "Check my reminders",
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

# --- Module Management Tools ---

LOAD_MODULE = ToolDefinition(
    name="load_module",
    description="Load/enable a module to add AI capabilities. Check resources first if unsure.",
    category="system",
    module=None,  # Built-in, uses ModuleManager
    parameters=[
        ToolParameter(
            name="module_id",
            type="string",
            description="Module ID to load (e.g., 'image_gen_local', 'voice_output')",
            required=True,
        ),
    ],
    examples=[
        "Load image generation capability",
        "Enable voice output",
        "Turn on the vision module",
    ],
)

UNLOAD_MODULE = ToolDefinition(
    name="unload_module",
    description="Unload/disable a module to free up resources",
    category="system",
    module=None,
    parameters=[
        ToolParameter(
            name="module_id",
            type="string",
            description="Module ID to unload",
            required=True,
        ),
    ],
    examples=[
        "Unload image generation to save memory",
        "Disable video generation",
        "Turn off the avatar module",
    ],
)

LIST_MODULES = ToolDefinition(
    name="list_modules",
    description="List all available modules and their status (loaded/unloaded)",
    category="system",
    module=None,
    parameters=[],
    examples=[
        "What modules are available?",
        "Show me what's currently loaded",
        "List all AI capabilities",
    ],
)

CHECK_RESOURCES = ToolDefinition(
    name="check_resources",
    description="Check current system resource usage (RAM, VRAM, CPU) and get recommendations",
    category="system",
    module=None,
    parameters=[],
    examples=[
        "Check if I have enough resources",
        "Can I run video generation?",
        "What's my current memory usage?",
        "Am I using too much RAM?",
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
    SCREENSHOT,
    SEE_SCREEN,
    
    # Control
    CONTROL_AVATAR,
    SPEAK,
    
    # System - File Operations
    READ_FILE,
    WRITE_FILE,
    LIST_DIRECTORY,
    MOVE_FILE,
    DELETE_FILE,
    READ_DOCUMENT,
    EXTRACT_TEXT,
    
    # System - Web
    WEB_SEARCH,
    FETCH_WEBPAGE,
    
    # System - Commands & Info
    GET_SYSTEM_INFO,
    RUN_COMMAND,
    
    # System - Task Management
    CREATE_CHECKLIST,
    LIST_CHECKLISTS,
    ADD_TASK,
    LIST_TASKS,
    COMPLETE_TASK,
    SET_REMINDER,
    LIST_REMINDERS,
    CHECK_REMINDERS,
    
    # Module Management
    LOAD_MODULE,
    UNLOAD_MODULE,
    LIST_MODULES,
    CHECK_RESOURCES,
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
