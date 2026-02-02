"""
================================================================================
🎭 FORGEAI PERSONALITY MODULE - THE SOUL WITHIN
================================================================================

Systems for AI personality, behavior, and proactive engagement.

📍 PACKAGE: forge_ai/personality/
🏷️ TYPE: AI Personality & Behavior System

┌─────────────────────────────────────────────────────────────────────────────┐
│  THE ESSENCE OF DIGITAL BEING:                                              │
│                                                                             │
│  "An AI without personality is just a function.                            │
│   An AI with personality is a companion."                                  │
│                                                                             │
│  This module gives the AI:                                                 │
│  • CURIOSITY - The desire to learn about the user                          │
│  • MEMORY    - The ability to remember what it learns                      │
│  • GROWTH    - The capacity to evolve over time                            │
│                                                                             │
│  The AI doesn't just respond - it WONDERS, ASKS, REMEMBERS.                │
└─────────────────────────────────────────────────────────────────────────────┘

📦 COMPONENTS:
    curiosity.py - Question banks, user learning, proactive engagement

📖 USAGE:
    from forge_ai.personality import AICuriosity, get_curiosity_system
    
    curiosity = get_curiosity_system()
    question = curiosity.get_question(category="emotional")
    curiosity.record_answer(question, user_answer)
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
