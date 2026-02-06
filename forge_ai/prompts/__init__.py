"""
ForgeAI Prompt Library Module

Create, manage, and share prompt templates.
"""

from .prompt_library import (
    BUILTIN_PROMPTS,
    PromptCategory,
    PromptCollection,
    PromptHubClient,
    PromptLibrary,
    PromptTemplate,
    PromptVariable,
    get_prompt_hub,
    get_prompt_library,
    install_builtin_prompts,
)

__all__ = [
    'PromptTemplate',
    'PromptVariable',
    'PromptCollection',
    'PromptCategory',
    'PromptLibrary',
    'PromptHubClient',
    'get_prompt_library',
    'get_prompt_hub',
    'install_builtin_prompts',
    'BUILTIN_PROMPTS',
]
