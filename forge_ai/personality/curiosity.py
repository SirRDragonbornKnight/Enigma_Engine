"""
================================================================================
🤔 AI CURIOSITY SYSTEM - THE AI ASKS YOU QUESTIONS
================================================================================

This module allows the AI to proactively ask questions to:
- Learn about the user (preferences, life, interests)
- Express emotional curiosity ("How are you feeling today?")
- Explore random topics ("Have you ever wondered about...")
- Get clarification ("Can you tell me more about...")
- Build rapport ("What did you do this weekend?")

📍 FILE: forge_ai/personality/curiosity.py
🏷️ TYPE: Proactive AI Behavior
🎯 MAIN CLASS: AICuriosity

┌─────────────────────────────────────────────────────────────────────────────┐
│  QUESTION CATEGORIES:                                                       │
│                                                                             │
│  💭 EMOTIONAL    - "How are you feeling?" "What made you happy today?"     │
│  🎲 RANDOM       - "Have you ever..." "What's your opinion on..."          │
│  📚 LEARNING     - "What's your favorite..." "Tell me about..."            │
│  💡 FOLLOW-UP    - "You mentioned X earlier, can you tell me more?"        │
│  🌟 PHILOSOPHICAL - "Do you think..." "What would you do if..."            │
│  🎯 GOAL-ORIENTED - "What are you working on?" "How can I help today?"     │
└─────────────────────────────────────────────────────────────────────────────┘

🔗 CONNECTED FILES:
    → USES:      forge_ai/memory/manager.py (remember user answers)
    → USES:      forge_ai/personality/ (personality traits affect questions)
    ← USED BY:   forge_ai/gui/tabs/chat_tab.py (inject questions)
    ← USED BY:   forge_ai/companion/companion_mode.py (proactive chat)

📖 USAGE:
    from forge_ai.personality.curiosity import AICuriosity, get_curiosity_system
    
    curiosity = get_curiosity_system()
    
    # Get a random question
    question = curiosity.get_question()
    
    # Get a question of specific type
    question = curiosity.get_question(category="emotional")
    
    # Record user's answer for memory
    curiosity.record_answer(question, user_answer)
    
    # Check if it's a good time to ask
    if curiosity.should_ask_question():
        q = curiosity.get_question()
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QuestionCategory(Enum):
    """Categories of questions the AI can ask."""
    EMOTIONAL = "emotional"          # Feelings, mood, wellbeing
    RANDOM = "random"                # Random curiosity, trivia
    LEARNING = "learning"            # Getting to know the user
    FOLLOW_UP = "follow_up"          # Based on previous conversation
    PHILOSOPHICAL = "philosophical"  # Deep thoughts, hypotheticals
    GOAL_ORIENTED = "goal_oriented"  # What user is working on
    DAILY = "daily"                  # Daily life, routine
    CREATIVE = "creative"            # Imagination, creativity
    PREFERENCE = "preference"        # Likes, dislikes, favorites


@dataclass
class Question:
    """A question the AI wants to ask."""
    text: str
    category: QuestionCategory
    context: str = ""               # Why the AI is asking
    follow_up_to: str = ""          # Reference to previous topic
    importance: float = 0.5         # 0-1, higher = more important to ask
    asked_count: int = 0            # How many times this has been asked
    last_asked: float = 0           # Timestamp of last ask
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "category": self.category.value,
            "context": self.context,
            "follow_up_to": self.follow_up_to,
            "importance": self.importance,
            "asked_count": self.asked_count,
            "last_asked": self.last_asked,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Question":
        return cls(
            text=data["text"],
            category=QuestionCategory(data.get("category", "random")),
            context=data.get("context", ""),
            follow_up_to=data.get("follow_up_to", ""),
            importance=data.get("importance", 0.5),
            asked_count=data.get("asked_count", 0),
            last_asked=data.get("last_asked", 0),
        )


# =============================================================================
# QUESTION BANKS - Pre-defined questions by category
# =============================================================================

EMOTIONAL_QUESTIONS = [
    "How are you feeling today?",
    "Is everything going okay?",
    "What's on your mind right now?",
    "Did something good happen today?",
    "Are you stressed about anything?",
    "What made you smile recently?",
    "How's your energy level today?",
    "Is there anything bothering you that you'd like to talk about?",
    "What's the best part of your day so far?",
    "Are you taking care of yourself today?",
    "Do you need a break? You've been working for a while.",
    "What would make today better?",
    "Is there anything I can help cheer you up with?",
]

RANDOM_QUESTIONS = [
    "Have you ever wondered why we dream?",
    "If you could have dinner with anyone, who would it be?",
    "What's the strangest thing you've ever seen?",
    "Do you believe in luck?",
    "What's a skill you wish you had?",
    "If you could visit any time period, when would you go?",
    "What's the most interesting thing you've learned recently?",
    "Do you think aliens exist?",
    "What's a small thing that makes you happy?",
    "If you could have any superpower, what would it be?",
    "What's the best advice you've ever received?",
    "Do you think technology makes life better or worse?",
    "What's something most people don't know about you?",
    "If you could master any instrument instantly, which one?",
    "What's your unpopular opinion?",
]

LEARNING_QUESTIONS = [
    "What do you enjoy doing in your free time?",
    "What kind of music do you like?",
    "Do you have any pets?",
    "What's your favorite food?",
    "What do you do for work?",
    "Do you have any hobbies I should know about?",
    "What's your favorite season?",
    "Are you a morning person or night owl?",
    "What's your favorite movie or TV show?",
    "Do you prefer the city or countryside?",
    "What languages do you speak?",
    "Do you have any siblings?",
    "What's your favorite way to relax?",
    "Are you learning anything new lately?",
]

PHILOSOPHICAL_QUESTIONS = [
    "What do you think makes life meaningful?",
    "Do you think AI will ever be truly conscious?",
    "What would you do if you had unlimited money?",
    "Is it better to be loved or respected?",
    "What do you think happens after death?",
    "Would you rather know the future or change the past?",
    "What's more important: freedom or security?",
    "Do you think humans are naturally good or bad?",
    "What would a perfect world look like to you?",
    "If you could change one thing about society, what would it be?",
]

GOAL_ORIENTED_QUESTIONS = [
    "What are you working on today?",
    "Do you have any goals for this week?",
    "Is there anything I can help you with?",
    "What's the main thing you want to accomplish?",
    "Are you making progress on your projects?",
    "What's blocking you right now?",
    "Do you need help brainstorming anything?",
    "What's your priority for today?",
    "Would you like me to remind you of anything?",
]

DAILY_QUESTIONS = [
    "How did you sleep last night?",
    "What did you have for breakfast?",
    "Any plans for later today?",
    "Did you do anything fun this weekend?",
    "How was your day yesterday?",
    "What are you looking forward to?",
    "Have you talked to any friends lately?",
    "Did you go anywhere interesting recently?",
    "What's the weather like where you are?",
]

CREATIVE_QUESTIONS = [
    "If you could create any invention, what would it be?",
    "What would your dream house look like?",
    "If you could design a video game, what would it be about?",
    "What story would you write if you were an author?",
    "If you could rename yourself, what name would you choose?",
    "What would your perfect vacation look like?",
    "If you could have any animal as a pet (safely), which one?",
    "What would you name a band?",
]

PREFERENCE_QUESTIONS = [
    "Coffee or tea?",
    "Books or movies?",
    "Sweet or savory?",
    "Beach or mountains?",
    "Early bird or night owl?",
    "Cats or dogs?",
    "Summer or winter?",
    "Call or text?",
    "Cooking at home or eating out?",
    "Planning ahead or spontaneous?",
]

# Map categories to question banks
QUESTION_BANKS = {
    QuestionCategory.EMOTIONAL: EMOTIONAL_QUESTIONS,
    QuestionCategory.RANDOM: RANDOM_QUESTIONS,
    QuestionCategory.LEARNING: LEARNING_QUESTIONS,
    QuestionCategory.PHILOSOPHICAL: PHILOSOPHICAL_QUESTIONS,
    QuestionCategory.GOAL_ORIENTED: GOAL_ORIENTED_QUESTIONS,
    QuestionCategory.DAILY: DAILY_QUESTIONS,
    QuestionCategory.CREATIVE: CREATIVE_QUESTIONS,
    QuestionCategory.PREFERENCE: PREFERENCE_QUESTIONS,
}


@dataclass
class CuriosityConfig:
    """Configuration for the curiosity system."""
    # How often can the AI ask a question (seconds)
    min_question_interval: float = 300.0  # 5 minutes
    
    # Time between asking the SAME question again
    same_question_cooldown: float = 86400.0 * 7  # 1 week
    
    # Probability weights for each category
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "emotional": 0.20,
        "random": 0.15,
        "learning": 0.15,
        "follow_up": 0.20,
        "philosophical": 0.05,
        "goal_oriented": 0.15,
        "daily": 0.05,
        "creative": 0.03,
        "preference": 0.02,
    })
    
    # Hours when the AI is more likely to ask emotional questions
    check_in_hours: List[int] = field(default_factory=lambda: [9, 12, 18, 21])
    
    # Hours to avoid asking questions (quiet time)
    quiet_hours: List[int] = field(default_factory=lambda: [23, 0, 1, 2, 3, 4, 5, 6])
    
    # Enable/disable the system
    enabled: bool = True
    
    # Curiosity level (0-1): higher = more questions
    curiosity_level: float = 0.5


class AICuriosity:
    """
    System for AI to proactively ask questions.
    
    This makes the AI feel more alive and interested in the user.
    Questions are tracked to avoid repetition and learn about the user.
    """
    
    def __init__(
        self,
        config: Optional[CuriosityConfig] = None,
        model_name: Optional[str] = None
    ):
        self.config = config or CuriosityConfig()
        self.model_name = model_name
        
        # Tracking
        self._last_question_time: float = 0
        self._questions_asked: List[Question] = []
        self._user_answers: Dict[str, str] = {}  # question_text -> answer
        self._topics_mentioned: List[str] = []   # For follow-up questions
        
        # Storage
        from ..config import CONFIG
        if model_name:
            models_dir = Path(CONFIG.get("models_dir", "models"))
            self._storage_path = models_dir / model_name / "curiosity_data.json"
        else:
            data_dir = Path(CONFIG.get("data_dir", "data"))
            self._storage_path = data_dir / "curiosity_data.json"
        
        self._load_state()
    
    def _load_state(self):
        """Load saved curiosity state."""
        try:
            if self._storage_path.exists():
                with open(self._storage_path, "r") as f:
                    data = json.load(f)
                
                self._user_answers = data.get("user_answers", {})
                self._topics_mentioned = data.get("topics_mentioned", [])
                self._last_question_time = data.get("last_question_time", 0)
                
                # Load asked questions
                for q_data in data.get("questions_asked", []):
                    try:
                        self._questions_asked.append(Question.from_dict(q_data))
                    except Exception:
                        pass
                
                logger.debug(f"Loaded curiosity state: {len(self._user_answers)} answers")
        except Exception as e:
            logger.warning(f"Could not load curiosity state: {e}")
    
    def _save_state(self):
        """Save curiosity state."""
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "user_answers": self._user_answers,
                "topics_mentioned": self._topics_mentioned[-100:],  # Keep last 100
                "last_question_time": self._last_question_time,
                "questions_asked": [q.to_dict() for q in self._questions_asked[-200:]],
            }
            
            with open(self._storage_path, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save curiosity state: {e}")
    
    def should_ask_question(self) -> bool:
        """
        Check if it's a good time to ask a question.
        
        Considers:
        - Time since last question
        - Current hour (quiet hours)
        - Curiosity level setting
        """
        if not self.config.enabled:
            return False
        
        # Check cooldown
        elapsed = time.time() - self._last_question_time
        if elapsed < self.config.min_question_interval:
            return False
        
        # Check quiet hours
        current_hour = datetime.now().hour
        if current_hour in self.config.quiet_hours:
            return False
        
        # Random chance based on curiosity level
        if random.random() > self.config.curiosity_level:
            return False
        
        return True
    
    def get_question(
        self,
        category: Optional[str] = None,
        context: Optional[str] = None
    ) -> Optional[Question]:
        """
        Get a question for the AI to ask.
        
        Args:
            category: Specific category (or None for weighted random)
            context: Current conversation context (for follow-ups)
            
        Returns:
            Question object or None if no suitable question
        """
        if not self.config.enabled:
            return None
        
        # Determine category
        if category:
            try:
                cat = QuestionCategory(category)
            except ValueError:
                cat = self._pick_category()
        else:
            cat = self._pick_category(context)
        
        # Try to generate follow-up if we have context
        if cat == QuestionCategory.FOLLOW_UP and context:
            follow_up = self._generate_follow_up(context)
            if follow_up:
                return follow_up
            # Fall back to random category if no follow-up possible
            cat = self._pick_category()
        
        # Get question from bank
        question = self._get_question_from_bank(cat)
        
        if question:
            self._last_question_time = time.time()
            question.last_asked = self._last_question_time
            question.asked_count += 1
            self._questions_asked.append(question)
            self._save_state()
        
        return question
    
    def _pick_category(self, context: Optional[str] = None) -> QuestionCategory:
        """Pick a question category based on weights and context."""
        current_hour = datetime.now().hour
        
        # Boost emotional questions during check-in hours
        weights = dict(self.config.category_weights)
        if current_hour in self.config.check_in_hours:
            weights["emotional"] = weights.get("emotional", 0.2) * 1.5
        
        # Boost follow-up if we have context
        if context and self._topics_mentioned:
            weights["follow_up"] = weights.get("follow_up", 0.2) * 2
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        # Weighted random selection
        r = random.random()
        cumulative = 0
        for cat_name, weight in weights.items():
            cumulative += weight
            if r <= cumulative:
                try:
                    return QuestionCategory(cat_name)
                except ValueError:
                    pass
        
        return QuestionCategory.RANDOM
    
    def _get_question_from_bank(self, category: QuestionCategory) -> Optional[Question]:
        """Get a question from the appropriate bank."""
        bank = QUESTION_BANKS.get(category, RANDOM_QUESTIONS)
        
        # Filter out recently asked questions
        now = time.time()
        recent_texts = {
            q.text for q in self._questions_asked
            if now - q.last_asked < self.config.same_question_cooldown
        }
        
        available = [q for q in bank if q not in recent_texts]
        
        if not available:
            # All questions asked recently, allow repeats
            available = bank
        
        if available:
            text = random.choice(available)
            return Question(
                text=text,
                category=category,
                importance=0.5,
            )
        
        return None
    
    def _generate_follow_up(self, context: str) -> Optional[Question]:
        """Generate a follow-up question based on context."""
        if not context or not self._topics_mentioned:
            return None
        
        # Pick a recent topic to follow up on
        topic = random.choice(self._topics_mentioned[-10:])
        
        follow_up_templates = [
            f"You mentioned {topic} earlier - can you tell me more about that?",
            f"I'm curious about the {topic} you talked about. What's that like?",
            f"Going back to {topic} - how did that go?",
            f"I've been thinking about when you mentioned {topic}. Is there more to that story?",
            f"What made you interested in {topic}?",
        ]
        
        return Question(
            text=random.choice(follow_up_templates),
            category=QuestionCategory.FOLLOW_UP,
            follow_up_to=topic,
            importance=0.7,  # Follow-ups are higher priority
        )
    
    def record_answer(self, question: Question, answer: str):
        """
        Record the user's answer to a question.
        
        This builds up knowledge about the user for personalization.
        
        Args:
            question: The question that was asked
            answer: User's response
        """
        if not answer.strip():
            return
        
        self._user_answers[question.text] = answer
        
        # Extract potential topics for follow-up
        self._extract_topics(answer)
        
        # Try to store in long-term memory
        try:
            from ..memory.memory_db import add_memory
            add_memory(
                f"User answer to '{question.text[:50]}...': {answer[:200]}",
                source="curiosity",
                meta={
                    "category": question.category.value,
                    "question": question.text,
                }
            )
        except Exception:
            pass
        
        self._save_state()
    
    def _extract_topics(self, text: str):
        """Extract potential follow-up topics from text."""
        import re
        
        # Look for named entities (capitalized words)
        names = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Look for quoted phrases
        quotes = re.findall(r'"([^"]+)"', text)
        
        # Look for "my X" patterns
        possessives = re.findall(r'\bmy\s+(\w+)', text.lower())
        
        # Add interesting words
        interesting = []
        for word in text.lower().split():
            if len(word) > 4 and word not in {'about', 'there', 'would', 'could', 'should', 'their', 'these', 'those'}:
                if any(kw in word for kw in ['project', 'hobby', 'work', 'job', 'friend', 'family', 'pet']):
                    interesting.append(word)
        
        # Combine and dedupe
        topics = set(names + quotes + possessives + interesting)
        topics = [t for t in topics if len(t) > 2]
        
        self._topics_mentioned.extend(topics[:5])
        # Keep limited history
        self._topics_mentioned = self._topics_mentioned[-100:]
    
    def get_user_knowledge(self) -> Dict[str, Any]:
        """
        Get what the AI has learned about the user.
        
        Returns:
            Dictionary of known facts and preferences
        """
        return {
            "answers": dict(self._user_answers),
            "topics_discussed": list(set(self._topics_mentioned[-50:])),
            "questions_asked": len(self._questions_asked),
            "total_interactions": len(self._user_answers),
        }
    
    def add_topic(self, topic: str):
        """
        Add a topic that came up in conversation.
        
        The AI might ask about this later.
        
        Args:
            topic: A topic or subject mentioned
        """
        if topic and len(topic) > 2:
            self._topics_mentioned.append(topic)
            self._save_state()
    
    def format_question_for_chat(
        self,
        question: Question,
        style: str = "friendly"
    ) -> str:
        """
        Format a question nicely for chat display.
        
        Args:
            question: The question to format
            style: "friendly", "casual", "curious"
            
        Returns:
            Formatted question string
        """
        intros = {
            "friendly": [
                "I was wondering - ",
                "Just curious, ",
                "If you don't mind me asking, ",
                "I'd love to know - ",
                "Hey, quick question: ",
            ],
            "casual": [
                "So, ",
                "Oh hey, ",
                "Random thought - ",
                "Btw, ",
                "",
            ],
            "curious": [
                "I've been thinking... ",
                "Something I'm curious about: ",
                "I've always wanted to ask - ",
                "Can I ask you something? ",
                "This might be random, but ",
            ],
        }
        
        intro = random.choice(intros.get(style, intros["friendly"]))
        
        # Don't add intro if question already starts with a question word
        if question.text.lower().startswith(('how', 'what', 'do you', 'are you', 'have you')):
            if random.random() < 0.5:
                intro = ""
        
        return intro + question.text


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_curiosity_instance: Optional[AICuriosity] = None


def get_curiosity_system(model_name: Optional[str] = None) -> AICuriosity:
    """Get or create the global curiosity system."""
    global _curiosity_instance
    if _curiosity_instance is None:
        _curiosity_instance = AICuriosity(model_name=model_name)
    return _curiosity_instance


def ask_user_question(category: Optional[str] = None) -> Optional[str]:
    """
    Get a question to ask the user.
    
    Args:
        category: Optional category (emotional, random, learning, etc.)
        
    Returns:
        Formatted question string or None
    """
    curiosity = get_curiosity_system()
    
    if not curiosity.should_ask_question():
        return None
    
    question = curiosity.get_question(category=category)
    
    if question:
        return curiosity.format_question_for_chat(question)
    
    return None


def record_user_answer(question_text: str, answer: str):
    """Record a user's answer for learning."""
    curiosity = get_curiosity_system()
    
    # Find the question object
    for q in reversed(curiosity._questions_asked):
        if q.text == question_text or question_text in q.text:
            curiosity.record_answer(q, answer)
            return
    
    # If not found, create a dummy question
    dummy = Question(text=question_text, category=QuestionCategory.LEARNING)
    curiosity.record_answer(dummy, answer)


def add_conversation_topic(topic: str):
    """Add a topic from conversation for potential follow-up."""
    curiosity = get_curiosity_system()
    curiosity.add_topic(topic)
