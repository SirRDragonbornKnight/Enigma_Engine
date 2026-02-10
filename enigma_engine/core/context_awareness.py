"""
Context Awareness Enhancement for Multi-Turn Conversations
Tracks conversation context and provides fallback clarifications.
"""
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    entities: list[str] = None  # Extracted entities (names, places, etc.)
    topics: list[str] = None  # Detected topics
    intent: str = None  # Detected intent
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.topics is None:
            self.topics = []


class ContextTracker:
    """Tracks and manages conversation context."""
    
    def __init__(self, max_context_turns: int = 10):
        """
        Initialize context tracker.
        
        Args:
            max_context_turns: Maximum number of turns to keep in context
        """
        self.max_context_turns = max_context_turns
        self.conversation_history: list[ConversationTurn] = []
        self.current_topic: Optional[str] = None
        self.entities_mentioned: dict[str, int] = {}  # entity -> mention count
        self.unclear_count: int = 0  # Track unclear queries
        
    def add_turn(
        self,
        role: str,
        content: str,
        entities: Optional[list[str]] = None,
        topics: Optional[list[str]] = None
    ) -> ConversationTurn:
        """
        Add a conversation turn and update context.
        
        Args:
            role: 'user' or 'assistant'
            content: The message content
            entities: Extracted entities
            topics: Detected topics
            
        Returns:
            The created ConversationTurn
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now().timestamp(),
            entities=entities or [],
            topics=topics or []
        )
        
        # Extract simple entities if not provided
        if not turn.entities:
            turn.entities = self._extract_simple_entities(content)
        
        # Update entity tracking
        for entity in turn.entities:
            self.entities_mentioned[entity] = self.entities_mentioned.get(entity, 0) + 1
        
        # Update current topic
        if turn.topics:
            self.current_topic = turn.topics[0]
        
        # Add to history
        self.conversation_history.append(turn)
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_context_turns:
            removed = self.conversation_history.pop(0)
            # Decrease entity counts from removed turn
            for entity in removed.entities:
                if entity in self.entities_mentioned:
                    self.entities_mentioned[entity] -= 1
                    if self.entities_mentioned[entity] <= 0:
                        del self.entities_mentioned[entity]
        
        return turn
    
    def _extract_simple_entities(self, text: str) -> list[str]:
        """Extract simple entities like capitalized words."""
        # Very basic entity extraction - capitalized words that aren't at start of sentence
        words = text.split()
        entities = []
        
        for i, word in enumerate(words):
            # Skip first word and common words
            if i == 0 or word.lower() in ['the', 'a', 'an', 'this', 'that', 'these', 'those']:
                continue
            
            # Check if capitalized
            if word and word[0].isupper() and len(word) > 1:
                clean_word = re.sub(r'[^\w\s]', '', word)
                if clean_word:
                    entities.append(clean_word)
        
        return entities
    
    def detect_unclear_context(self, user_input: str) -> tuple[bool, Optional[str]]:
        """
        Detect if the user input lacks clear context.
        
        Args:
            user_input: The user's input
            
        Returns:
            (is_unclear, clarification_suggestion)
        """
        unclear_indicators = [
            # Pronouns without clear referents
            r'\b(it|this|that|he|she|they|them)\b',
            # Question words alone
            r'^(what|why|how|where|when)\??$',
            # Very short queries
            r'^\w{1,3}$',
        ]
        
        user_lower = user_input.lower().strip()
        
        # Check length
        if len(user_input.split()) <= 2:
            # Very short input might lack context
            for pattern in unclear_indicators[:2]:  # Check pronouns and question words
                if re.search(pattern, user_lower):
                    self.unclear_count += 1
                    return True, self._generate_clarification(user_input)
        
        # Check for pronouns without recent context
        if re.search(unclear_indicators[0], user_lower):
            if len(self.conversation_history) < 2:
                self.unclear_count += 1
                return True, self._generate_clarification(user_input)
        
        return False, None
    
    def _generate_clarification(self, user_input: str) -> str:
        """Generate a clarification prompt."""
        clarifications = [
            "Could you provide more details?",
            "I want to make sure I understand correctly. Can you rephrase that?",
            "Can you be more specific?",
            "Could you elaborate on that?",
            "I'm not quite sure I follow. Could you explain more?",
        ]
        
        # Choose based on unclear count to vary responses
        return clarifications[self.unclear_count % len(clarifications)]
    
    def get_relevant_context(self, current_input: str, max_turns: int = 5) -> list[ConversationTurn]:
        """
        Get relevant conversation history for the current input.
        
        Args:
            current_input: Current user input
            max_turns: Maximum number of turns to return
            
        Returns:
            List of relevant conversation turns
        """
        if not self.conversation_history:
            return []
        
        # Score turns by relevance to current input
        scored_turns = []
        input_words = set(current_input.lower().split())
        
        for i, turn in enumerate(self.conversation_history):
            score = 0.0
            
            # Recency score (more recent = higher)
            recency = (i + 1) / len(self.conversation_history)
            score += recency * 0.3  # 30% weight for recency
            
            # Word overlap score
            turn_text = f"{turn.user_input} {turn.ai_response}".lower()
            turn_words = set(turn_text.split())
            overlap = len(input_words & turn_words)
            if input_words:
                overlap_ratio = overlap / len(input_words)
                score += overlap_ratio * 0.4  # 40% weight for word overlap
            
            # Entity overlap score
            turn_entities = set()
            for word in turn_text.split():
                if word.istitle() and len(word) > 2:
                    turn_entities.add(word.lower())
            for word in current_input.split():
                if word.istitle() and word.lower() in turn_entities:
                    score += 0.1  # Bonus for shared entities
            
            # Topic continuity score
            if turn.detected_topic and self.current_topic:
                if turn.detected_topic == self.current_topic:
                    score += 0.2  # Bonus for same topic
            
            scored_turns.append((score, i, turn))
        
        # Sort by score (descending) and get top turns
        scored_turns.sort(key=lambda x: x[0], reverse=True)
        relevant_indices = sorted([t[1] for t in scored_turns[:max_turns]])
        
        # Return in chronological order
        return [self.conversation_history[i] for i in relevant_indices]
    
    def get_context_summary(self) -> str:
        """Get a summary of the current context."""
        if not self.conversation_history:
            return "No conversation context yet."
        
        parts = []
        
        if self.current_topic:
            parts.append(f"Current topic: {self.current_topic}")
        
        # Most mentioned entities
        if self.entities_mentioned:
            top_entities = sorted(
                self.entities_mentioned.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            entities_str = ", ".join([f"{e} ({c}x)" for e, c in top_entities])
            parts.append(f"Entities discussed: {entities_str}")
        
        parts.append(f"Conversation turns: {len(self.conversation_history)}")
        
        return " | ".join(parts)
    
    def format_context_for_prompt(self, include_last_n: int = 5) -> str:
        """
        Format context for inclusion in AI prompt.
        
        Args:
            include_last_n: Number of recent turns to include
            
        Returns:
            Formatted context string
        """
        if not self.conversation_history:
            return ""
        
        recent = self.conversation_history[-include_last_n:]
        
        lines = ["Previous conversation:"]
        for turn in recent:
            role_label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role_label}: {turn.content}")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset context tracker."""
        self.conversation_history.clear()
        self.current_topic = None
        self.entities_mentioned.clear()
        self.unclear_count = 0


class ContextAwareConversation:
    """Manages context-aware conversations with fallback clarifications."""
    
    def __init__(self, max_context_turns: int = 10):
        """
        Initialize context-aware conversation manager.
        
        Args:
            max_context_turns: Maximum turns to keep in context
        """
        self.context_tracker = ContextTracker(max_context_turns)
        self.clarification_threshold = 3  # After 3 unclear queries, suggest restart
        
    def process_user_input(self, user_input: str) -> dict[str, Any]:
        """
        Process user input with context awareness.
        
        Args:
            user_input: The user's input
            
        Returns:
            Dictionary with:
                - needs_clarification: bool
                - clarification_prompt: Optional[str]
                - context: str (formatted context)
                - turn: ConversationTurn
        """
        # Check if input is unclear
        is_unclear, clarification = self.context_tracker.detect_unclear_context(user_input)
        
        # Add turn to context
        turn = self.context_tracker.add_turn('user', user_input)
        
        # Get context
        context = self.context_tracker.format_context_for_prompt()
        
        result = {
            'needs_clarification': is_unclear,
            'clarification_prompt': clarification,
            'context': context,
            'turn': turn,
            'context_summary': self.context_tracker.get_context_summary()
        }
        
        # Check if we should suggest restarting conversation
        if self.context_tracker.unclear_count >= self.clarification_threshold:
            result['suggest_restart'] = True
            result['restart_prompt'] = (
                "We seem to be having trouble understanding each other. "
                "Would you like to start fresh? You can also provide more context to help me understand better."
            )
        
        return result
    
    def add_assistant_response(self, response: str):
        """Add assistant response to context."""
        self.context_tracker.add_turn('assistant', response)
    
    def get_context_for_prompt(self) -> str:
        """Get formatted context for AI prompt."""
        return self.context_tracker.format_context_for_prompt()
    
    def reset_conversation(self):
        """Reset the conversation context."""
        self.context_tracker.reset()
        logger.info("Conversation context reset")


def create_context_aware_system(max_turns: int = 10) -> ContextAwareConversation:
    """
    Factory function to create a context-aware conversation system.
    
    Args:
        max_turns: Maximum number of turns to keep in context
        
    Returns:
        ContextAwareConversation instance
    """
    return ContextAwareConversation(max_turns)
