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
    get_curiosity_system,
    ask_user_question,
    record_user_answer,
    add_conversation_topic,
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
