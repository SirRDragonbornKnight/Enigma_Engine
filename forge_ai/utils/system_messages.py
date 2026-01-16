"""
Forge System Message Utilities
================================

Clearly labeled system vs ForgeAI messages.
Uses ForgeAI's unique message format.
"""


class MessagePrefix:
    """Forge standard message prefixes for different message types."""
    SYSTEM = "[Forge:System]"
    AI = "[Forge]"
    USER = "[User]"
    ERROR = "[Forge:Error]"
    WARNING = "[Forge:Warning]"
    INFO = "[Forge:Info]"
    DEBUG = "[Forge:Debug]"
    THINKING = "[Forge:Thinking]"
    MEMORY = "[Forge:Memory]"


def system_msg(text: str) -> str:
    """Format an Forge system message."""
    return f"{MessagePrefix.SYSTEM} {text}"


def error_msg(text: str) -> str:
    """Format an Forge error message."""
    return f"{MessagePrefix.ERROR} {text}"


def warning_msg(text: str) -> str:
    """Format an Forge warning message."""
    return f"{MessagePrefix.WARNING} {text}"


def info_msg(text: str) -> str:
    """Format an Forge info message."""
    return f"{MessagePrefix.INFO} {text}"


def debug_msg(text: str) -> str:
    """Format an Forge debug message."""
    return f"{MessagePrefix.DEBUG} {text}"


def ai_msg(text: str) -> str:
    """Format an ForgeAI message."""
    return f"{MessagePrefix.AI} {text}"


def forge_msg(text: str) -> str:
    """Format a ForgeAI message (preferred name)."""
    return f"{MessagePrefix.AI} {text}"


# Legacy alias for backwards compatibility
enigma_msg = forge_msg


def user_msg(text: str) -> str:
    """Format a user message."""
    return f"{MessagePrefix.USER} {text}"


def thinking_msg(text: str) -> str:
    """Format an Forge thinking/reasoning message."""
    return f"{MessagePrefix.THINKING} {text}"


def memory_msg(text: str) -> str:
    """Format an Forge memory recall message."""
    return f"{MessagePrefix.MEMORY} {text}"
