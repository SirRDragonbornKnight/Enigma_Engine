"""
================================================================================
AGENTS PACKAGE
================================================================================

Multi-agent system for ForgeAI.
"""

from .multi_agent import (
    MultiAgentSystem,
    Agent,
    AgentRole,
    AgentPersonality,
    AgentMessage,
    AgentConversation,
    get_multi_agent_system,
    PRESET_AGENTS,
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
