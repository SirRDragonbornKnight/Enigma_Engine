"""
Enigma System Message Utilities
================================

Clearly labeled system vs AI Tester messages.
Uses AI Tester's unique message format.
"""


class MessagePrefix:
    """Enigma standard message prefixes for different message types."""
    SYSTEM = "[Enigma:System]"
    AI = "[Enigma]"
    USER = "[User]"
    ERROR = "[Enigma:Error]"
    WARNING = "[Enigma:Warning]"
    INFO = "[Enigma:Info]"
    DEBUG = "[Enigma:Debug]"
    THINKING = "[Enigma:Thinking]"
    MEMORY = "[Enigma:Memory]"


def system_msg(text: str) -> str:
    """Format an Enigma system message."""
    return f"{MessagePrefix.SYSTEM} {text}"


def error_msg(text: str) -> str:
    """Format an Enigma error message."""
    return f"{MessagePrefix.ERROR} {text}"


def warning_msg(text: str) -> str:
    """Format an Enigma warning message."""
    return f"{MessagePrefix.WARNING} {text}"


def info_msg(text: str) -> str:
    """Format an Enigma info message."""
    return f"{MessagePrefix.INFO} {text}"


def debug_msg(text: str) -> str:
    """Format an Enigma debug message."""
    return f"{MessagePrefix.DEBUG} {text}"


def ai_msg(text: str) -> str:
    """Format an AI Tester message."""
    return f"{MessagePrefix.AI} {text}"


def enigma_msg(text: str) -> str:
    """Format an AI Tester message (preferred name)."""
    return f"{MessagePrefix.AI} {text}"


def user_msg(text: str) -> str:
    """Format a user message."""
    return f"{MessagePrefix.USER} {text}"


def thinking_msg(text: str) -> str:
    """Format an Enigma thinking/reasoning message."""
    return f"{MessagePrefix.THINKING} {text}"


def memory_msg(text: str) -> str:
    """Format an Enigma memory recall message."""
    return f"{MessagePrefix.MEMORY} {text}"
