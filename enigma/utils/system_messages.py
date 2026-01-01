"""
System Message Utilities - Clearly label system vs AI messages.
"""


class MessagePrefix:
    """Standard message prefixes for different message types."""
    SYSTEM = "[System]"
    AI = "[AI]"
    USER = "[User]"
    ERROR = "[Error]"
    WARNING = "[Warning]"
    INFO = "[Info]"
    DEBUG = "[Debug]"


def system_msg(text: str) -> str:
    """Format a system message."""
    return f"{MessagePrefix.SYSTEM} {text}"


def error_msg(text: str) -> str:
    """Format an error message."""
    return f"{MessagePrefix.ERROR} {text}"


def warning_msg(text: str) -> str:
    """Format a warning message."""
    return f"{MessagePrefix.WARNING} {text}"


def info_msg(text: str) -> str:
    """Format an info message."""
    return f"{MessagePrefix.INFO} {text}"


def debug_msg(text: str) -> str:
    """Format a debug message."""
    return f"{MessagePrefix.DEBUG} {text}"


def ai_msg(text: str) -> str:
    """Format an AI message."""
    return f"{MessagePrefix.AI} {text}"


def user_msg(text: str) -> str:
    """Format a user message."""
    return f"{MessagePrefix.USER} {text}"
