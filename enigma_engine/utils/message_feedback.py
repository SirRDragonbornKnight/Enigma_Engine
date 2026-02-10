"""
================================================================================
Message Feedback System - Track user reactions and feedback on messages.
================================================================================

Feedback features:
- Thumbs up/down reactions
- Star ratings
- Custom reactions
- Feedback categories (helpful, accurate, creative, etc.)
- Analytics and reporting
- Export for training

USAGE:
    from enigma_engine.utils.message_feedback import FeedbackManager, get_feedback_manager
    
    feedback = get_feedback_manager()
    
    # Add reaction
    feedback.react(message_id, ReactionType.THUMBS_UP)
    
    # Add detailed feedback
    feedback.add_feedback(message_id, rating=5, categories=["helpful", "accurate"])
    
    # Get feedback stats
    stats = feedback.get_stats()
    
    # Export for training
    data = feedback.export_training_data()
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReactionType(str, Enum):
    """Types of quick reactions."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    HEART = "heart"
    STAR = "star"
    LIGHTBULB = "lightbulb"  # Insightful
    QUESTION = "question"    # Confused
    CHECK = "check"          # Resolved
    FLAG = "flag"            # Report


class FeedbackCategory(str, Enum):
    """Categories for detailed feedback."""
    HELPFUL = "helpful"
    ACCURATE = "accurate"
    CREATIVE = "creative"
    CLEAR = "clear"
    RELEVANT = "relevant"
    COMPLETE = "complete"
    FAST = "fast"
    
    # Negative categories
    UNHELPFUL = "unhelpful"
    INACCURATE = "inaccurate"
    CONFUSING = "confusing"
    OFF_TOPIC = "off_topic"
    INCOMPLETE = "incomplete"
    REPETITIVE = "repetitive"
    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"


@dataclass
class MessageReaction:
    """A reaction on a message."""
    message_id: str
    reaction_type: ReactionType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "reaction_type": self.reaction_type.value,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageReaction:
        return cls(
            message_id=data["message_id"],
            reaction_type=ReactionType(data["reaction_type"]),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )


@dataclass
class MessageFeedback:
    """Detailed feedback for a message."""
    id: str
    message_id: str
    
    # Rating (1-5 stars)
    rating: int = 0
    
    # Reaction
    reaction: ReactionType | None = None
    
    # Categories that apply
    categories: list[str] = field(default_factory=list)
    
    # Freeform comment
    comment: str = ""
    
    # Context
    conversation_id: str = ""
    prompt: str = ""        # The user's prompt
    response: str = ""      # The AI's response
    model_name: str = ""    # Model that generated response
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.reaction:
            data["reaction"] = self.reaction.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageFeedback:
        if "reaction" in data and data["reaction"]:
            data["reaction"] = ReactionType(data["reaction"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @property
    def is_positive(self) -> bool:
        """Check if this is positive feedback."""
        if self.rating > 0:
            return self.rating >= 4
        if self.reaction:
            return self.reaction in [ReactionType.THUMBS_UP, ReactionType.HEART, 
                                     ReactionType.STAR, ReactionType.CHECK]
        return False
    
    @property
    def is_negative(self) -> bool:
        """Check if this is negative feedback."""
        if self.rating > 0:
            return self.rating <= 2
        if self.reaction:
            return self.reaction in [ReactionType.THUMBS_DOWN, ReactionType.FLAG]
        return False


@dataclass
class FeedbackStats:
    """Statistics about collected feedback."""
    total_feedback: int = 0
    total_reactions: int = 0
    
    # Reaction counts
    thumbs_up: int = 0
    thumbs_down: int = 0
    hearts: int = 0
    stars: int = 0
    
    # Rating distribution
    rating_counts: dict[int, int] = field(default_factory=dict)
    average_rating: float = 0.0
    
    # Category counts
    category_counts: dict[str, int] = field(default_factory=dict)
    
    # Ratio
    positive_ratio: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeedbackManager:
    """
    Manage message feedback and reactions.
    """
    
    def __init__(self, data_path: Path | None = None):
        """
        Initialize feedback manager.
        
        Args:
            data_path: Path to store feedback data
        """
        self._data_path = data_path or Path("data/feedback")
        self._data_path.mkdir(parents=True, exist_ok=True)
        
        self._feedback_file = self._data_path / "feedback.json"
        self._reactions_file = self._data_path / "reactions.json"
        
        self._feedback: dict[str, MessageFeedback] = {}
        self._reactions: dict[str, list[MessageReaction]] = {}  # message_id -> reactions
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load feedback data from disk."""
        # Load feedback
        if self._feedback_file.exists():
            try:
                with open(self._feedback_file, encoding='utf-8') as f:
                    data = json.load(f)
                    for fb_data in data.get("feedback", []):
                        fb = MessageFeedback.from_dict(fb_data)
                        self._feedback[fb.id] = fb
            except Exception as e:
                logger.error(f"Failed to load feedback: {e}")
        
        # Load reactions
        if self._reactions_file.exists():
            try:
                with open(self._reactions_file, encoding='utf-8') as f:
                    data = json.load(f)
                    for msg_id, reactions_data in data.get("reactions", {}).items():
                        self._reactions[msg_id] = [
                            MessageReaction.from_dict(r) for r in reactions_data
                        ]
            except Exception as e:
                logger.error(f"Failed to load reactions: {e}")
    
    def _save_data(self) -> None:
        """Save feedback data to disk."""
        try:
            # Save feedback
            fb_data = {
                "feedback": [fb.to_dict() for fb in self._feedback.values()],
                "last_updated": datetime.now().isoformat()
            }
            with open(self._feedback_file, 'w', encoding='utf-8') as f:
                json.dump(fb_data, f, indent=2)
            
            # Save reactions
            rx_data = {
                "reactions": {
                    msg_id: [r.to_dict() for r in reactions]
                    for msg_id, reactions in self._reactions.items()
                },
                "last_updated": datetime.now().isoformat()
            }
            with open(self._reactions_file, 'w', encoding='utf-8') as f:
                json.dump(rx_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feedback data: {e}")
    
    def react(
        self,
        message_id: str,
        reaction_type: ReactionType,
        toggle: bool = True
    ) -> bool:
        """
        Add or toggle a reaction on a message.
        
        Args:
            message_id: Message ID
            reaction_type: Type of reaction
            toggle: If True, removes reaction if already exists
            
        Returns:
            True if reaction was added, False if removed
        """
        if message_id not in self._reactions:
            self._reactions[message_id] = []
        
        # Check if reaction already exists
        existing = None
        for i, r in enumerate(self._reactions[message_id]):
            if r.reaction_type == reaction_type:
                existing = i
                break
        
        if existing is not None and toggle:
            # Remove existing reaction
            self._reactions[message_id].pop(existing)
            self._save_data()
            return False
        elif existing is None:
            # Add new reaction
            reaction = MessageReaction(
                message_id=message_id,
                reaction_type=reaction_type
            )
            self._reactions[message_id].append(reaction)
            self._save_data()
            return True
        
        return True  # Already exists, not toggled
    
    def get_reactions(self, message_id: str) -> list[MessageReaction]:
        """Get all reactions for a message."""
        return self._reactions.get(message_id, [])
    
    def add_feedback(
        self,
        message_id: str,
        rating: int = 0,
        reaction: ReactionType | None = None,
        categories: list[str] | None = None,
        comment: str = "",
        conversation_id: str = "",
        prompt: str = "",
        response: str = "",
        model_name: str = ""
    ) -> MessageFeedback:
        """
        Add detailed feedback for a message.
        
        Args:
            message_id: Message ID
            rating: Star rating (1-5, 0 for no rating)
            reaction: Quick reaction
            categories: Applicable feedback categories
            comment: Freeform comment
            conversation_id: Parent conversation ID
            prompt: The user's original prompt
            response: The AI's response
            model_name: Model that generated the response
            
        Returns:
            The created feedback
        """
        feedback_id = f"{message_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        feedback = MessageFeedback(
            id=feedback_id,
            message_id=message_id,
            rating=max(0, min(5, rating)),
            reaction=reaction,
            categories=categories or [],
            comment=comment,
            conversation_id=conversation_id,
            prompt=prompt,
            response=response,
            model_name=model_name
        )
        
        self._feedback[feedback_id] = feedback
        
        # Also add as a reaction if provided
        if reaction:
            self.react(message_id, reaction, toggle=False)
        
        self._save_data()
        return feedback
    
    def get_feedback(self, feedback_id: str) -> MessageFeedback | None:
        """Get feedback by ID."""
        return self._feedback.get(feedback_id)
    
    def get_message_feedback(self, message_id: str) -> list[MessageFeedback]:
        """Get all feedback for a message."""
        return [fb for fb in self._feedback.values() if fb.message_id == message_id]
    
    def thumbs_up(self, message_id: str) -> bool:
        """Quick thumbs up."""
        return self.react(message_id, ReactionType.THUMBS_UP)
    
    def thumbs_down(self, message_id: str) -> bool:
        """Quick thumbs down."""
        return self.react(message_id, ReactionType.THUMBS_DOWN)
    
    def rate(self, message_id: str, rating: int) -> MessageFeedback:
        """Quick star rating."""
        return self.add_feedback(message_id, rating=rating)
    
    def get_stats(self) -> FeedbackStats:
        """Get feedback statistics."""
        stats = FeedbackStats()
        
        # Count reactions
        for reactions in self._reactions.values():
            for r in reactions:
                stats.total_reactions += 1
                if r.reaction_type == ReactionType.THUMBS_UP:
                    stats.thumbs_up += 1
                elif r.reaction_type == ReactionType.THUMBS_DOWN:
                    stats.thumbs_down += 1
                elif r.reaction_type == ReactionType.HEART:
                    stats.hearts += 1
                elif r.reaction_type == ReactionType.STAR:
                    stats.stars += 1
        
        # Count feedback
        stats.total_feedback = len(self._feedback)
        total_rating = 0
        rating_count = 0
        positive = 0
        negative = 0
        
        for fb in self._feedback.values():
            if fb.rating > 0:
                stats.rating_counts[fb.rating] = stats.rating_counts.get(fb.rating, 0) + 1
                total_rating += fb.rating
                rating_count += 1
            
            for cat in fb.categories:
                stats.category_counts[cat] = stats.category_counts.get(cat, 0) + 1
            
            if fb.is_positive:
                positive += 1
            elif fb.is_negative:
                negative += 1
        
        if rating_count > 0:
            stats.average_rating = total_rating / rating_count
        
        # Calculate positive ratio including reactions
        total_positive = positive + stats.thumbs_up + stats.hearts
        total_negative = negative + stats.thumbs_down
        total = total_positive + total_negative
        if total > 0:
            stats.positive_ratio = total_positive / total
        
        return stats
    
    def get_positive_feedback(self, limit: int = 100) -> list[MessageFeedback]:
        """Get positive feedback entries."""
        positive = [fb for fb in self._feedback.values() if fb.is_positive]
        return sorted(positive, key=lambda f: f.created_at, reverse=True)[:limit]
    
    def get_negative_feedback(self, limit: int = 100) -> list[MessageFeedback]:
        """Get negative feedback entries."""
        negative = [fb for fb in self._feedback.values() if fb.is_negative]
        return sorted(negative, key=lambda f: f.created_at, reverse=True)[:limit]
    
    def export_training_data(
        self,
        output_path: Path | None = None,
        min_rating: int = 4,
        include_negative: bool = False
    ) -> list[dict[str, Any]]:
        """
        Export feedback data for training.
        
        Args:
            output_path: Optional file path to save to
            min_rating: Minimum rating to include (for positive samples)
            include_negative: Whether to include negative feedback
            
        Returns:
            List of training examples
        """
        training_data = []
        
        for fb in self._feedback.values():
            # Skip if no prompt/response
            if not fb.prompt or not fb.response:
                continue
            
            # Filter by rating
            if fb.rating > 0:
                if fb.rating >= min_rating:
                    training_data.append({
                        "prompt": fb.prompt,
                        "response": fb.response,
                        "rating": fb.rating,
                        "label": "positive",
                        "categories": fb.categories,
                        "model": fb.model_name
                    })
                elif include_negative and fb.rating <= 2:
                    training_data.append({
                        "prompt": fb.prompt,
                        "response": fb.response,
                        "rating": fb.rating,
                        "label": "negative",
                        "categories": fb.categories,
                        "model": fb.model_name
                    })
            # Use reactions if no rating
            elif fb.is_positive:
                training_data.append({
                    "prompt": fb.prompt,
                    "response": fb.response,
                    "rating": 5,
                    "label": "positive",
                    "categories": fb.categories,
                    "model": fb.model_name
                })
            elif include_negative and fb.is_negative:
                training_data.append({
                    "prompt": fb.prompt,
                    "response": fb.response,
                    "rating": 1,
                    "label": "negative",
                    "categories": fb.categories,
                    "model": fb.model_name
                })
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2)
        
        return training_data
    
    def get_model_stats(self) -> dict[str, dict[str, Any]]:
        """Get feedback stats grouped by model."""
        model_stats: dict[str, dict[str, Any]] = {}
        
        for fb in self._feedback.values():
            if not fb.model_name:
                continue
            
            if fb.model_name not in model_stats:
                model_stats[fb.model_name] = {
                    "count": 0,
                    "total_rating": 0,
                    "positive": 0,
                    "negative": 0
                }
            
            stats = model_stats[fb.model_name]
            stats["count"] += 1
            
            if fb.rating > 0:
                stats["total_rating"] += fb.rating
            
            if fb.is_positive:
                stats["positive"] += 1
            elif fb.is_negative:
                stats["negative"] += 1
        
        # Calculate averages
        for model, stats in model_stats.items():
            if stats["count"] > 0:
                stats["average_rating"] = stats["total_rating"] / stats["count"]
                total = stats["positive"] + stats["negative"]
                stats["positive_ratio"] = stats["positive"] / total if total > 0 else 0
        
        return model_stats
    
    def clear_old_feedback(self, days: int = 90) -> int:
        """
        Remove feedback older than specified days.
        
        Args:
            days: Age threshold in days
            
        Returns:
            Number of entries removed
        """
        from datetime import timedelta
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        removed = 0
        
        to_remove = [
            fb_id for fb_id, fb in self._feedback.items()
            if fb.created_at < cutoff
        ]
        
        for fb_id in to_remove:
            del self._feedback[fb_id]
            removed += 1
        
        if removed > 0:
            self._save_data()
        
        return removed


# Singleton instance
_feedback_instance: FeedbackManager | None = None


def get_feedback_manager(data_path: Path | None = None) -> FeedbackManager:
    """Get or create the singleton feedback manager."""
    global _feedback_instance
    if _feedback_instance is None:
        _feedback_instance = FeedbackManager(data_path)
    return _feedback_instance


# Convenience functions
def thumbs_up(message_id: str) -> bool:
    """Quick thumbs up on a message."""
    return get_feedback_manager().thumbs_up(message_id)


def thumbs_down(message_id: str) -> bool:
    """Quick thumbs down on a message."""
    return get_feedback_manager().thumbs_down(message_id)


def rate_message(message_id: str, rating: int) -> MessageFeedback:
    """Rate a message (1-5 stars)."""
    return get_feedback_manager().rate(message_id, rating)


def get_feedback_stats() -> FeedbackStats:
    """Get feedback statistics."""
    return get_feedback_manager().get_stats()
