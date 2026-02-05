"""
================================================================================
Quick Replies - Suggested follow-up questions and responses.
================================================================================

Generates contextual follow-up suggestions:
- Analyzes conversation context for relevant follow-ups  
- Pattern-based suggestion generation
- Customizable suggestion rules
- Learning from user selections

USAGE:
    from forge_ai.utils.quick_replies import QuickReplyGenerator, get_quick_reply_generator
    
    generator = get_quick_reply_generator()
    
    # Get suggestions based on conversation
    suggestions = generator.get_suggestions(
        messages=[
            {"role": "user", "content": "How do I sort a list in Python?"},
            {"role": "assistant", "content": "You can use list.sort() or sorted()..."}
        ]
    )
    # Returns: ["What's the difference between sort() and sorted()?", 
    #           "How do I sort by a custom key?", ...]
    
    # Track which suggestions users click to improve rankings
    generator.record_selection("What's the difference...")
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SuggestionRule:
    """A rule for generating suggestions."""
    name: str
    triggers: List[str]  # Keywords/patterns that activate this rule
    suggestions: List[str]  # Suggestions to show
    priority: int = 0  # Higher = shown first
    category: str = ""
    is_regex: bool = False  # If True, triggers are regex patterns


@dataclass
class QuickReplyConfig:
    """Configuration for the quick reply generator."""
    max_suggestions: int = 5
    min_relevance_score: float = 0.1
    include_generic: bool = True
    learn_from_selections: bool = True


# Built-in suggestion rules
SUGGESTION_RULES: List[SuggestionRule] = [
    # ===== Coding Topics =====
    SuggestionRule(
        name="python_basics",
        triggers=["python", "list", "dict", "string", "function", "class"],
        suggestions=[
            "Can you show me an example?",
            "What are common mistakes to avoid?",
            "How does this work under the hood?",
            "What's the time complexity?",
            "Are there any alternatives?"
        ],
        priority=8,
        category="coding"
    ),
    SuggestionRule(
        name="code_explanation",
        triggers=["explain", "how does", "what does", "why does"],
        suggestions=[
            "Can you give another example?",
            "What would happen if I changed X?",
            "How would I modify this for a different use case?",
            "What are the edge cases?",
            "Can you break it down further?"
        ],
        priority=7,
        category="coding"
    ),
    SuggestionRule(
        name="debugging",
        triggers=["error", "bug", "doesn't work", "not working", "exception", "traceback"],
        suggestions=[
            "What's causing this error?",
            "How do I fix it?",
            "How can I prevent this in the future?",
            "Can you explain the error message?",
            "Should I add error handling?"
        ],
        priority=9,
        category="debugging"
    ),
    SuggestionRule(
        name="code_review",
        triggers=["review", "feedback", "improve", "refactor", "optimize"],
        suggestions=[
            "What's the most important issue to fix?",
            "Can you show the refactored version?",
            "Are there any security concerns?",
            "How would you structure this differently?",
            "What tests should I add?"
        ],
        priority=8,
        category="coding"
    ),
    
    # ===== Learning =====
    SuggestionRule(
        name="learning",
        triggers=["learn", "understand", "explain", "what is", "how to"],
        suggestions=[
            "Can you give a real-world example?",
            "What should I learn next?",
            "Are there any good resources?",
            "What are common misconceptions?",
            "How is this used in practice?"
        ],
        priority=6,
        category="learning"
    ),
    SuggestionRule(
        name="comparison",
        triggers=["difference", "compare", "vs", "versus", "better"],
        suggestions=[
            "When should I use each one?",
            "What are the pros and cons?",
            "Which is more performant?",
            "Which is more popular?",
            "Can you show examples of both?"
        ],
        priority=7,
        category="analysis"
    ),
    
    # ===== Writing =====
    SuggestionRule(
        name="writing_help",
        triggers=["write", "draft", "email", "message", "document"],
        suggestions=[
            "Can you make it more formal?",
            "Can you make it shorter?",
            "How about a different tone?",
            "Can you add more detail?",
            "Can you help with the subject line?"
        ],
        priority=6,
        category="writing"
    ),
    SuggestionRule(
        name="summarize",
        triggers=["summary", "summarize", "tldr", "main points"],
        suggestions=[
            "Can you expand on point X?",
            "What's the most important takeaway?",
            "Can you make it even shorter?",
            "Are there any key details missing?",
            "Can you add bullet points?"
        ],
        priority=6,
        category="writing"
    ),
    
    # ===== Analysis =====
    SuggestionRule(
        name="decision_help",
        triggers=["should i", "best way", "recommend", "advice", "decision"],
        suggestions=[
            "What factors should I consider?",
            "What would you do in my situation?",
            "What are the risks?",
            "Is there a middle ground?",
            "What questions should I ask myself?"
        ],
        priority=7,
        category="analysis"
    ),
    SuggestionRule(
        name="brainstorm",
        triggers=["ideas", "brainstorm", "creative", "suggest", "options"],
        suggestions=[
            "Can you give me more ideas?",
            "Which idea is most feasible?",
            "Can you combine some of these?",
            "What's the most creative option?",
            "How would I start implementing idea X?"
        ],
        priority=6,
        category="creative"
    ),
    
    # ===== Project Management =====
    SuggestionRule(
        name="project",
        triggers=["project", "plan", "timeline", "milestone", "task"],
        suggestions=[
            "What should I prioritize?",
            "What might go wrong?",
            "How should I track progress?",
            "What resources do I need?",
            "Can you break this down further?"
        ],
        priority=6,
        category="productivity"
    ),
    
    # ===== Generic Follow-ups =====
    SuggestionRule(
        name="generic_continue",
        triggers=[".*"],  # Matches anything
        suggestions=[
            "Can you explain more?",
            "Can you give an example?",
            "What else should I know?",
            "How do I get started?",
            "Are there any alternatives?"
        ],
        priority=1,
        category="generic",
        is_regex=True
    ),
]


class QuickReplyGenerator:
    """
    Generates contextual follow-up suggestions.
    """
    
    def __init__(
        self,
        config: Optional[QuickReplyConfig] = None,
        data_path: Optional[Path] = None
    ):
        """
        Initialize the quick reply generator.
        
        Args:
            config: Generation configuration
            data_path: Path to store learned data
        """
        self.config = config or QuickReplyConfig()
        self._data_path = data_path or Path("data/quick_replies")
        self._data_path.mkdir(parents=True, exist_ok=True)
        
        self._rules: List[SuggestionRule] = list(SUGGESTION_RULES)
        self._custom_rules_file = self._data_path / "custom_rules.json"
        self._selection_history_file = self._data_path / "selection_history.json"
        
        self._selection_history: Counter = Counter()
        self._context_selections: Dict[str, Counter] = {}
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load custom rules and selection history."""
        # Load custom rules
        if self._custom_rules_file.exists():
            try:
                with open(self._custom_rules_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data.get("rules", []):
                        rule = SuggestionRule(**item)
                        self._rules.append(rule)
            except Exception as e:
                logger.error(f"Failed to load custom rules: {e}")
        
        # Load selection history
        if self._selection_history_file.exists():
            try:
                with open(self._selection_history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._selection_history = Counter(data.get("global", {}))
                    self._context_selections = {
                        k: Counter(v) for k, v in data.get("context", {}).items()
                    }
            except Exception as e:
                logger.error(f"Failed to load selection history: {e}")
    
    def _save_data(self) -> None:
        """Save selection history."""
        try:
            data = {
                "global": dict(self._selection_history),
                "context": {k: dict(v) for k, v in self._context_selections.items()}
            }
            with open(self._selection_history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save selection history: {e}")
    
    def get_suggestions(
        self,
        messages: List[Dict[str, str]],
        max_suggestions: Optional[int] = None
    ) -> List[str]:
        """
        Generate follow-up suggestions based on conversation.
        
        Args:
            messages: Conversation history [{role, content}]
            max_suggestions: Override max suggestions
            
        Returns:
            List of suggested follow-up questions
        """
        if not messages:
            return self._get_generic_suggestions(max_suggestions)
        
        # Get the last exchange
        last_user_msg = ""
        last_assistant_msg = ""
        
        for msg in reversed(messages):
            if msg["role"] == "user" and not last_user_msg:
                last_user_msg = msg["content"].lower()
            elif msg["role"] == "assistant" and not last_assistant_msg:
                last_assistant_msg = msg["content"].lower()
            if last_user_msg and last_assistant_msg:
                break
        
        combined_text = f"{last_user_msg} {last_assistant_msg}"
        
        # Score and collect suggestions
        scored_suggestions: List[Tuple[float, str]] = []
        seen_suggestions: Set[str] = set()
        
        for rule in self._rules:
            if self._rule_matches(rule, combined_text):
                for suggestion in rule.suggestions:
                    if suggestion in seen_suggestions:
                        continue
                    
                    score = self._score_suggestion(suggestion, rule, combined_text)
                    
                    if score >= self.config.min_relevance_score:
                        scored_suggestions.append((score, suggestion))
                        seen_suggestions.add(suggestion)
        
        # Sort by score (descending)
        scored_suggestions.sort(reverse=True, key=lambda x: x[0])
        
        # Return top suggestions
        limit = max_suggestions or self.config.max_suggestions
        return [s for _, s in scored_suggestions[:limit]]
    
    def _rule_matches(self, rule: SuggestionRule, text: str) -> bool:
        """Check if a rule's triggers match the text."""
        text_lower = text.lower()
        
        for trigger in rule.triggers:
            if rule.is_regex:
                if re.search(trigger, text_lower):
                    return True
            else:
                if trigger.lower() in text_lower:
                    return True
        
        return False
    
    def _score_suggestion(
        self,
        suggestion: str,
        rule: SuggestionRule,
        context: str
    ) -> float:
        """Score a suggestion based on relevance."""
        score = rule.priority / 10.0  # Base score from priority
        
        # Boost based on selection history
        if self.config.learn_from_selections:
            global_count = self._selection_history.get(suggestion, 0)
            score += min(global_count * 0.1, 0.5)  # Cap at 0.5 boost
            
            # Context-specific boost
            context_key = self._get_context_key(context)
            if context_key in self._context_selections:
                context_count = self._context_selections[context_key].get(suggestion, 0)
                score += min(context_count * 0.2, 0.3)  # Higher weight for context
        
        # Penalize generic suggestions if we have specific ones
        if rule.category == "generic":
            score *= 0.5
        
        return score
    
    def _get_context_key(self, text: str) -> str:
        """Extract a context key from text for tracking."""
        # Extract main topic words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        # Filter common words and get most important
        common = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                  'could', 'should', 'can', 'may', 'might', 'to', 'of', 'in',
                  'for', 'on', 'with', 'at', 'by', 'from', 'or', 'and', 'not',
                  'it', 'this', 'that', 'i', 'you', 'he', 'she', 'they', 'we'}
        keywords = [w for w in words if w not in common and len(w) > 2]
        return '_'.join(sorted(set(keywords[:5])))
    
    def _get_generic_suggestions(self, max_suggestions: Optional[int] = None) -> List[str]:
        """Get generic suggestions when no context."""
        if not self.config.include_generic:
            return []
        
        generic = [
            "What can you help me with?",
            "Tell me about your capabilities",
            "Help me write some code",
            "Explain a concept to me",
            "Help me with a project"
        ]
        
        limit = max_suggestions or self.config.max_suggestions
        return generic[:limit]
    
    def record_selection(self, suggestion: str, context: str = "") -> None:
        """
        Record that a user selected a suggestion.
        
        Args:
            suggestion: The suggestion that was selected
            context: Optional context for contextual learning
        """
        if not self.config.learn_from_selections:
            return
        
        self._selection_history[suggestion] += 1
        
        if context:
            context_key = self._get_context_key(context)
            if context_key not in self._context_selections:
                self._context_selections[context_key] = Counter()
            self._context_selections[context_key][suggestion] += 1
        
        self._save_data()
    
    def add_rule(
        self,
        name: str,
        triggers: List[str],
        suggestions: List[str],
        priority: int = 5,
        category: str = "custom",
        is_regex: bool = False
    ) -> SuggestionRule:
        """
        Add a custom suggestion rule.
        
        Args:
            name: Rule name
            triggers: Keywords/patterns that activate
            suggestions: Suggestions to show
            priority: Priority (0-10)
            category: Category name
            is_regex: If triggers are regex patterns
            
        Returns:
            The created rule
        """
        rule = SuggestionRule(
            name=name,
            triggers=triggers,
            suggestions=suggestions,
            priority=priority,
            category=category,
            is_regex=is_regex
        )
        
        self._rules.append(rule)
        self._save_custom_rules()
        
        return rule
    
    def _save_custom_rules(self) -> None:
        """Save custom rules to disk."""
        try:
            builtin_names = {r.name for r in SUGGESTION_RULES}
            custom_rules = [
                {
                    "name": r.name,
                    "triggers": r.triggers,
                    "suggestions": r.suggestions,
                    "priority": r.priority,
                    "category": r.category,
                    "is_regex": r.is_regex
                }
                for r in self._rules if r.name not in builtin_names
            ]
            
            with open(self._custom_rules_file, 'w', encoding='utf-8') as f:
                json.dump({"rules": custom_rules}, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save custom rules: {e}")
    
    def get_coding_follow_ups(self, language: str = "") -> List[str]:
        """Get coding-specific follow-ups."""
        suggestions = [
            f"Show me best practices for {language}" if language else "What are best practices?",
            "How would you test this?",
            "What are potential edge cases?",
            "Can you optimize this code?",
            "How would you handle errors here?"
        ]
        return suggestions[:self.config.max_suggestions]
    
    def get_clarifying_questions(self) -> List[str]:
        """Get questions to clarify requirements."""
        return [
            "What's the expected input/output?",
            "Are there any constraints?",
            "What have you tried so far?",
            "What's the use case?",
            "Do you have a specific example?"
        ][:self.config.max_suggestions]


# Singleton instance
_quick_reply_instance: Optional[QuickReplyGenerator] = None


def get_quick_reply_generator(
    config: Optional[QuickReplyConfig] = None
) -> QuickReplyGenerator:
    """Get or create the singleton generator instance."""
    global _quick_reply_instance
    if _quick_reply_instance is None:
        _quick_reply_instance = QuickReplyGenerator(config)
    return _quick_reply_instance


# Convenience functions
def get_quick_suggestions(messages: List[Dict[str, str]], max_results: int = 5) -> List[str]:
    """Quick access to get suggestions."""
    return get_quick_reply_generator().get_suggestions(messages, max_results)


def record_suggestion_click(suggestion: str, context: str = "") -> None:
    """Record that a suggestion was clicked."""
    get_quick_reply_generator().record_selection(suggestion, context)
