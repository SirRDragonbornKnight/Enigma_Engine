"""
Tool Definitions for Enigma AI Engine
===============================

Defines all available tools that the AI can use, including their schemas,
parameters, and which modules provide them.

This allows the AI to:
- Know what tools are available
- Understand what each tool does
- Know what parameters each tool requires
- Execute tools through the module system
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "int", "float", "bool", "list", "dict"
    description: str
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None  # For choice parameters


@dataclass
class ToolDefinition:
    """Definition of a tool that the AI can use."""
    name: str
    description: str
    parameters: list[ToolParameter]
    module: str | None = None  # Module that provides this capability
    category: str = "general"  # "generation", "perception", "control", "system", "general"
    examples: list[str] = field(default_factory=list)
    version: str = "1.0.0"  # Tool version (semantic versioning)
    deprecated: bool = False  # Whether tool is deprecated
    deprecated_message: str | None = None  # Deprecation message
    added_in: str | None = None  # Version when tool was added
    
    def to_dict(self) -> dict[str, Any]:
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
    module=None,  # Direct tool - auto-loads Stable Diffusion without module system
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

# --- Avatar Control Tools (Advanced/Manual Override) ---
# NOTE: Primary avatar control is through inline commands in AI responses:
#   [emotion:happy] [gesture:wave] [action:think] etc.
# These tools are for advanced/programmatic control.

CONTROL_AVATAR = ToolDefinition(
    name="control_avatar",
    description="[ADVANCED] Manual override for avatar control. For normal expression, use inline commands like [emotion:happy] in your response instead. This tool is for programmatic control like moving position or showing/hiding.",
    category="control",
    module="avatar",
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="Action to perform",
            required=True,
            enum=["show", "hide", "jump", "pin", "unpin", "move", "resize", "orientation"],
        ),
        ToolParameter(
            name="value",
            type="string",
            description="Value for the action. For 'move': 'x,y' coordinates. For 'resize': pixel size like '250'. For 'orientation': 'front', 'back', 'left', 'right'.",
            required=False,
            default="",
        ),
    ],
    examples=[
        "Show my avatar on desktop",
        "Move to position 100,200",
        "Resize to 300 pixels",
        "Turn to face left",
    ],
)

CUSTOMIZE_AVATAR = ToolDefinition(
    name="customize_avatar",
    description="[ADVANCED] Change avatar visual appearance programmatically (colors, lighting). For expressions, use inline [emotion:X] commands instead.",
    category="control",
    module="avatar",
    parameters=[
        ToolParameter(
            name="setting",
            type="string",
            description="Setting to change",
            required=True,
            enum=["primary_color", "secondary_color", "accent_color", "light_intensity", 
                  "ambient_strength", "wireframe", "show_grid", "rotate_speed", "auto_rotate", "reset"],
        ),
        ToolParameter(
            name="value",
            type="string",
            description="Value for the setting (color hex like '#ff0000', number 0-100, or 'true'/'false' for booleans)",
            required=True,
        ),
    ],
    examples=[
        "Change avatar primary color to red",
        "Turn on wireframe mode",
        "Set lighting to 80%",
        "Enable auto-rotation",
        "Reset avatar settings",
    ],
)

CONTROL_AVATAR_BONES = ToolDefinition(
    name="control_avatar_bones",
    description="[ADVANCED] Direct bone manipulation for rigged 3D avatars. For normal gestures, use inline [gesture:X] commands in your response instead.",
    category="control",
    module="avatar",
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="Type of bone control action",
            required=True,
            enum=["move_bone", "gesture", "reset_pose"],
        ),
        ToolParameter(
            name="bone_name",
            type="string",
            description="Name of bone to move (for move_bone action)",
            required=False,
            enum=["head", "neck", "spine", "chest", "hips", "left_shoulder", "right_shoulder",
                  "left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm",
                  "left_hand", "right_hand", "left_upper_leg", "right_upper_leg",
                  "left_lower_leg", "right_lower_leg", "left_foot", "right_foot"],
        ),
        ToolParameter(
            name="pitch",
            type="float",
            description="Pitch rotation in degrees (nodding up/down). Range: typically -45 to 45",
            required=False,
        ),
        ToolParameter(
            name="yaw",
            type="float",
            description="Yaw rotation in degrees (turning left/right). Range: typically -80 to 80",
            required=False,
        ),
        ToolParameter(
            name="roll",
            type="float",
            description="Roll rotation in degrees (tilting side to side). Range: typically -30 to 30",
            required=False,
        ),
        ToolParameter(
            name="gesture_name",
            type="string",
            description="Predefined gesture name (for gesture action)",
            required=False,
            enum=["nod", "shake", "wave", "shrug", "point", "thinking", "bow", "stretch"],
        ),
    ],
    examples=[
        "Make the avatar nod",
        "Avatar, wave hello",
        "Make the avatar do a thinking pose",
        "Move the avatar's head to look left",
        "Avatar shrug gesture",
        "Reset avatar to neutral position",
    ],
)

MANAGE_SCENE_OBJECTS = ToolDefinition(
    name="manage_scene_objects",
    description="Add, remove, or manipulate 3D objects/props in the avatar scene. Place items around the avatar for decoration, storytelling, or interaction.",
    category="control",
    module="avatar",
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="Action to perform on scene objects",
            required=True,
            enum=["add", "remove", "move", "scale", "rotate", "list", "clear"],
        ),
        ToolParameter(
            name="object_path",
            type="string",
            description="Path to 3D model file (for 'add' action). Supports GLB, GLTF, OBJ, FBX.",
            required=False,
        ),
        ToolParameter(
            name="object_id",
            type="string",
            description="ID of existing object (for 'remove', 'move', 'scale', 'rotate' actions)",
            required=False,
        ),
        ToolParameter(
            name="position",
            type="string",
            description="Position as 'x,y,z' coordinates (for 'add' or 'move' actions). Example: '0.5,-0.5,0.3'",
            required=False,
        ),
        ToolParameter(
            name="scale",
            type="float",
            description="Scale factor (for 'add' or 'scale' actions). Default: 1.0",
            required=False,
            default=1.0,
        ),
        ToolParameter(
            name="rotation",
            type="string",
            description="Rotation as 'pitch,yaw,roll' in degrees (for 'add' or 'rotate' actions)",
            required=False,
        ),
    ],
    examples=[
        "Add a hat object to the scene",
        "Place a chair next to the avatar",
        "Move the table object to position 0.5, 0, 0.5",
        "Remove the cup from the scene",
        "List all objects in the scene",
        "Clear all scene objects",
        "Scale the book to half size",
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

CREATE_VOICE_PROFILE = ToolDefinition(
    name="create_voice_profile",
    description="Create a new voice profile for an avatar based on personality description. The AI analyzes the description and generates matching voice parameters.",
    category="generation",
    module="voice_output",
    parameters=[
        ToolParameter(
            name="name",
            type="string",
            description="Name for the voice profile",
            required=True,
        ),
        ToolParameter(
            name="personality",
            type="string",
            description="Description of the character/personality (e.g., 'confident, deep-voiced leader who speaks calmly')",
            required=True,
        ),
        ToolParameter(
            name="base_voice",
            type="string",
            description="Base voice type",
            required=False,
            default="default",
            enum=["default", "male", "female"],
        ),
    ],
    examples=[
        "Create a voice for a wise old mentor",
        "Make a cheerful energetic assistant voice",
        "Generate a robotic AI voice",
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
# GUI Control Tools - AI can control the user interface
# =============================================================================

SWITCH_TAB = ToolDefinition(
    name="switch_tab",
    description="Navigate to a specific tab in the GUI. Use this to help users find features or switch context during conversations.",
    category="control",
    module=None,
    parameters=[
        ToolParameter(
            name="tab_name",
            type="string",
            description="Name of the tab to switch to",
            required=True,
            enum=["chat", "train", "history", "scale", "modules", "image", "code", 
                  "video", "audio", "search", "avatar", "vision", "personality",
                  "terminal", "files", "examples", "settings"],
        ),
    ],
    examples=[
        "Let me take you to the image generation tab",
        "Switch to settings so we can configure this",
        "Open the training tab",
    ],
)

ADJUST_SETTING = ToolDefinition(
    name="adjust_setting",
    description="Adjust a user preference or GUI setting. The AI can help optimize the user experience.",
    category="control",
    module=None,
    parameters=[
        ToolParameter(
            name="setting",
            type="string",
            description="The setting to adjust",
            required=True,
            enum=["theme", "chat_zoom", "auto_speak", "learn_while_chatting",
                  "always_on_top", "system_prompt_preset", "avatar_auto_run"],
        ),
        ToolParameter(
            name="value",
            type="string",
            description="New value for the setting (type depends on setting)",
            required=True,
        ),
    ],
    examples=[
        "Enable auto-speak so I can talk to you",
        "Turn on learning mode",
        "Make the chat text bigger",
        "Switch to dark theme",
    ],
)

GET_SETTING = ToolDefinition(
    name="get_setting",
    description="Get the current value of a user preference or setting",
    category="control",
    module=None,
    parameters=[
        ToolParameter(
            name="setting",
            type="string",
            description="The setting to retrieve",
            required=True,
            enum=["theme", "chat_zoom", "auto_speak", "learn_while_chatting",
                  "always_on_top", "system_prompt_preset", "avatar_auto_run",
                  "last_model", "last_tab"],
        ),
    ],
    examples=[
        "What's the current theme?",
        "Is auto-speak enabled?",
        "Check if learning mode is on",
    ],
)

MANAGE_CONVERSATION = ToolDefinition(
    name="manage_conversation",
    description="Manage chat conversations - save, rename, delete, list, or load conversations",
    category="control",
    module=None,
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="Action to perform on conversations",
            required=True,
            enum=["save", "rename", "delete", "list", "load", "new"],
        ),
        ToolParameter(
            name="name",
            type="string",
            description="Conversation name (for save/rename/delete/load)",
            required=False,
        ),
        ToolParameter(
            name="new_name",
            type="string",
            description="New name (for rename action)",
            required=False,
        ),
    ],
    examples=[
        "Save this conversation as 'Project Discussion'",
        "Show me my saved conversations",
        "Start a new chat",
        "Load the conversation about Python",
    ],
)

SHOW_HELP = ToolDefinition(
    name="show_help",
    description="Show help information or tips about a specific feature or the application",
    category="control",
    module=None,
    parameters=[
        ToolParameter(
            name="topic",
            type="string",
            description="Topic to get help about",
            required=False,
            enum=["getting_started", "chat", "training", "image_generation", 
                  "code_generation", "voice", "avatar", "modules", "settings",
                  "keyboard_shortcuts", "tips"],
            default="getting_started",
        ),
    ],
    examples=[
        "How do I train my AI?",
        "Show me keyboard shortcuts",
        "Help with image generation",
        "Give me some tips",
    ],
)

OPTIMIZE_FOR_HARDWARE = ToolDefinition(
    name="optimize_for_hardware",
    description="Automatically adjust settings based on detected hardware capabilities",
    category="control",
    module=None,
    parameters=[
        ToolParameter(
            name="mode",
            type="string",
            description="Optimization mode",
            required=False,
            enum=["auto", "performance", "balanced", "power_saver", "gaming"],
            default="auto",
        ),
    ],
    examples=[
        "Optimize settings for my computer",
        "Set up for best performance",
        "Use power-saving mode",
        "Configure for gaming (less resources)",
    ],
)


# =============================================================================
# Teacher-Student AI Training Tools
# =============================================================================

TRAIN_MODEL = ToolDefinition(
    name="train_model",
    description="Train an AI model on a data file. Returns training results including final loss.",
    category="training",
    module=None,
    parameters=[
        ToolParameter(
            name="data_file",
            type="string",
            description="Path to training data file (.txt with Q:/A: format)",
            required=True,
        ),
        ToolParameter(
            name="model_name",
            type="string",
            description="Name for the trained model (will be saved as models/{name}.pth)",
            required=False,
            default="student",
        ),
        ToolParameter(
            name="model_size",
            type="string",
            description="Size of model to train",
            required=False,
            enum=["nano", "micro", "tiny", "small", "medium", "large"],
            default="small",
        ),
        ToolParameter(
            name="epochs",
            type="int",
            description="Number of training epochs",
            required=False,
            default=30,
        ),
        ToolParameter(
            name="batch_size",
            type="int",
            description="Batch size for training",
            required=False,
            default=4,
        ),
        ToolParameter(
            name="learning_rate",
            type="float",
            description="Learning rate",
            required=False,
            default=0.0001,
        ),
    ],
    examples=[
        "Train a model on my data",
        "Create student AI from training.txt",
        "Train small model for 50 epochs",
    ],
)

CHAT_WITH_MODEL = ToolDefinition(
    name="chat_with_model",
    description="Send a message to a specific model and get its response. Used for testing models.",
    category="training",
    module=None,
    parameters=[
        ToolParameter(
            name="model_name",
            type="string",
            description="Name of the model to chat with (e.g., 'student', 'small_forge')",
            required=True,
        ),
        ToolParameter(
            name="message",
            type="string",
            description="Message to send to the model",
            required=True,
        ),
        ToolParameter(
            name="max_tokens",
            type="int",
            description="Maximum tokens in response",
            required=False,
            default=100,
        ),
        ToolParameter(
            name="temperature",
            type="float",
            description="Sampling temperature (0.1-1.5)",
            required=False,
            default=0.8,
        ),
    ],
    examples=[
        "Test the student model",
        "Ask student 'What is 2+2?'",
        "Chat with my trained model",
    ],
)

GENERATE_TRAINING_DATA = ToolDefinition(
    name="generate_training_data",
    description="Generate Q:/A: training data from web content or a topic.",
    category="training",
    module=None,
    parameters=[
        ToolParameter(
            name="topic",
            type="string",
            description="Topic to generate training data about",
            required=True,
        ),
        ToolParameter(
            name="num_pairs",
            type="int",
            description="Number of Q:/A: pairs to generate",
            required=False,
            default=50,
        ),
        ToolParameter(
            name="output_file",
            type="string",
            description="Path to save the training data",
            required=False,
            default="data/generated_training.txt",
        ),
        ToolParameter(
            name="use_web",
            type="bool",
            description="Whether to search the web for information",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="difficulty",
            type="string",
            description="Difficulty level of questions",
            required=False,
            enum=["basic", "intermediate", "advanced", "mixed"],
            default="mixed",
        ),
    ],
    examples=[
        "Generate training data about Python",
        "Create 100 Q&A pairs about machine learning",
        "Make training data from web search on cooking",
    ],
)

EVALUATE_MODEL = ToolDefinition(
    name="evaluate_model",
    description="Evaluate a model's performance on a test file. Returns accuracy score.",
    category="training",
    module=None,
    parameters=[
        ToolParameter(
            name="model_name",
            type="string",
            description="Name of the model to evaluate",
            required=True,
        ),
        ToolParameter(
            name="test_file",
            type="string",
            description="Path to test file with Q:/A: pairs",
            required=False,
            default="data/test_questions.txt",
        ),
        ToolParameter(
            name="num_questions",
            type="int",
            description="Number of questions to test (0 = all)",
            required=False,
            default=0,
        ),
    ],
    examples=[
        "Test student model accuracy",
        "Evaluate my trained model",
        "Run 20 test questions on student",
    ],
)

LIST_MODELS = ToolDefinition(
    name="list_models",
    description="List all available trained models in the models directory.",
    category="training",
    module=None,
    parameters=[],
    examples=[
        "What models do I have?",
        "Show available models",
        "List trained AIs",
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
    CONTROL_AVATAR_BONES,
    MANAGE_SCENE_OBJECTS,
    CUSTOMIZE_AVATAR,
    SPEAK,
    CREATE_VOICE_PROFILE,
    
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
    
    # GUI Control - AI can control the interface
    SWITCH_TAB,
    ADJUST_SETTING,
    GET_SETTING,
    MANAGE_CONVERSATION,
    SHOW_HELP,
    OPTIMIZE_FOR_HARDWARE,
    
    # Training - Teacher-Student AI System
    TRAIN_MODEL,
    CHAT_WITH_MODEL,
    GENERATE_TRAINING_DATA,
    EVALUATE_MODEL,
    LIST_MODELS,
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


def get_available_tools_for_prompt(include_guides: bool = True) -> str:
    """
    Get a formatted description of available tools for including in AI prompts.
    
    Args:
        include_guides: Whether to include AI control guide txt files from data/ai_control/
    
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
    
    # Include AI control guides from data/ai_control/ folder
    if include_guides:
        guide_content = _load_ai_control_guides()
        if guide_content:
            lines.append("\n\n" + "=" * 80)
            lines.append("AI CONTROL GUIDES")
            lines.append("=" * 80)
            lines.append(guide_content)
    
    return "\n".join(lines)


def _load_ai_control_guides() -> str:
    """
    Load all AI control guide txt files from data/ai_control/ folder.
    
    Returns:
        Combined content of all guide files
    """
    from pathlib import Path
    
    # Find the data/ai_control directory
    guide_dir = Path(__file__).parent.parent.parent / "data" / "ai_control"
    
    if not guide_dir.exists():
        return ""
    
    guides = []
    for txt_file in sorted(guide_dir.glob("*.txt")):
        try:
            content = txt_file.read_text(encoding="utf-8")
            guides.append(f"\n--- {txt_file.stem.upper().replace('_', ' ')} ---\n{content}")
        except Exception:
            continue
    
    return "\n".join(guides)


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
