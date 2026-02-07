"""
================================================================================
AGENTS PACKAGE
================================================================================

Multi-agent system for Enigma AI Engine.
"""

from .multi_agent import (
    PRESET_AGENTS,
    Agent,
    AgentConversation,
    AgentMessage,
    AgentPersonality,
    AgentRole,
    MultiAgentSystem,
    get_multi_agent_system,
)

__all__ = [
    'MultiAgentSystem',
    'Agent',
    'AgentRole',
    'AgentPersonality',
    'AgentMessage',
    'AgentConversation',
    'get_multi_agent_system',
    'PRESET_AGENTS',
]
