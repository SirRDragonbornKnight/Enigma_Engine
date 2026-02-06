"""
ForgeAI Personality Module

Systems for AI personality, behavior, and proactive engagement.

Components:
- curiosity: AI asks questions to learn about the user
"""

from .curiosity import (
    AICuriosity,
    CuriosityConfig,
    Question,
    QuestionCategory,
    add_conversation_topic,
    ask_user_question,
    get_curiosity_system,
    record_user_answer,
)

__all__ = [
    # Main class
    "AICuriosity",
    "CuriosityConfig",
    "Question",
    "QuestionCategory",
    
    # Convenience functions
    "get_curiosity_system",
    "ask_user_question",
    "record_user_answer",
    "add_conversation_topic",
]
