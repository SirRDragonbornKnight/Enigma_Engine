"""
================================================================================
THE HERALD'S CHAMBER - MESSAGE FORMATTING UTILITIES
================================================================================

In the grand halls of Enigma AI Engine, the Herald stands ready to announce all
communications. Each message bears a sacred prefix that identifies its
origin - whether from the System itself, the AI's mind, or the User's hand.

FILE: enigma_engine/utils/system_messages.py
TYPE: Message Formatting & Prefixing
MAIN CLASS: MessagePrefix

    THE HERALD'S PROCLAMATIONS:
    
    [Forge:System]   - Decrees from the system itself
    [Forge]          - Words from the AI's consciousness  
    [User]           - Messages from the human traveler
    [Forge:Error]    - Warnings of danger encountered
    [Forge:Warning]  - Cautions to heed
    [Forge:Info]     - Knowledge to be shared
    [Forge:Thinking] - The AI's inner contemplation
    [Forge:Memory]   - Echoes from remembered conversations

USAGE:
    from enigma_engine.utils.system_messages import system_msg, ai_msg, error_msg
    
    print(system_msg("The forge awakens"))
    # Output: [Forge:System] The forge awakens
    
    print(ai_msg("Hello, traveler!"))
    # Output: [Forge] Hello, traveler!

CONNECTED REALMS:
    USED BY:   enigma_engine/core/training.py - Training progress messages
    USED BY:   enigma_engine/core/inference.py - Generation status
    USED BY:   enigma_engine/gui/ - User interface messages
"""


# =============================================================================
# THE HERALD'S SEALS - Message Prefix Constants
# =============================================================================

class MessagePrefix:
    """
    The Herald's collection of sacred seals.
    
    Each prefix serves as a seal of authenticity, marking messages
    with their true origin so all who read may know the source.
    """
    SYSTEM = "[Forge:System]"
    AI = "[Forge]"
    USER = "[User]"
    ERROR = "[Forge:Error]"
    WARNING = "[Forge:Warning]"
    INFO = "[Forge:Info]"
    DEBUG = "[Forge:Debug]"
    THINKING = "[Forge:Thinking]"
    MEMORY = "[Forge:Memory]"


# =============================================================================
# THE HERALD'S PROCLAMATIONS - Message Formatting Functions
# =============================================================================

def system_msg(text: str) -> str:
    """
    Proclaim a system decree.
    
    Used for messages from Enigma AI Engine's core systems - initialization,
    shutdown, configuration changes, and operational status.
    """
    return f"{MessagePrefix.SYSTEM} {text}"


def error_msg(text: str) -> str:
    """
    Sound the alarm of danger.
    
    Used when something has gone wrong and requires attention.
    These messages indicate failures that may need intervention.
    """
    return f"{MessagePrefix.ERROR} {text}"


def warning_msg(text: str) -> str:
    """
    Issue a caution to travelers.
    
    Used for potential issues that don't halt operation but
    should be noted - deprecations, unusual states, recoverable errors.
    """
    return f"{MessagePrefix.WARNING} {text}"


def info_msg(text: str) -> str:
    """
    Share knowledge with the realm.
    
    Used for informational messages that help users understand
    what's happening - progress updates, status reports, helpful tips.
    """
    return f"{MessagePrefix.INFO} {text}"


def debug_msg(text: str) -> str:
    """
    Whisper secrets to the developers.
    
    Used for detailed technical information helpful during
    development and troubleshooting. Hidden from normal view.
    """
    return f"{MessagePrefix.DEBUG} {text}"


def ai_msg(text: str) -> str:
    """
    Speak with the AI's voice.
    
    Used when the AI itself is communicating - responses,
    thoughts, and generated content bear this seal.
    """
    return f"{MessagePrefix.AI} {text}"


def forge_msg(text: str) -> str:
    """
    The preferred invocation for AI messages.
    
    Identical to ai_msg, but with a name that better reflects
    the Enigma AI Engine identity. Use this for new code.
    """
    return f"{MessagePrefix.AI} {text}"


# Legacy alias preserved for ancient scrolls (backwards compatibility)
enigma_msg = forge_msg


def user_msg(text: str) -> str:
    """
    Speak with the human's voice.
    
    Used to echo or format messages from the user.
    """
    return f"{MessagePrefix.USER} {text}"


def thinking_msg(text: str) -> str:
    """
    Reveal the AI's inner contemplation.
    
    Used when showing the AI's reasoning process - chain of thought,
    planning steps, and decision-making visible to the user.
    """
    return f"{MessagePrefix.THINKING} {text}"


def memory_msg(text: str) -> str:
    """
    Echo from the halls of memory.
    
    Used when the AI recalls information from past conversations
    or long-term storage. Helps users understand context sources.
    """
    return f"{MessagePrefix.MEMORY} {text}"
