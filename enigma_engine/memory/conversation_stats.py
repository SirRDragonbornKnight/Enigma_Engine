"""
================================================================================
Conversation Statistics - Track and analyze conversation metrics.
================================================================================

Comprehensive conversation analytics:
- Token usage tracking over time
- Message counts by role
- Response time metrics
- Conversation duration
- Topic analysis
- Usage patterns

USAGE:
    from enigma_engine.memory.conversation_stats import ConversationStats, get_conversation_stats
    
    stats = get_conversation_stats()
    
    # Track a message
    stats.track_message(
        role="user",
        content="Hello!",
        tokens=5,
        response_time=0.0
    )
    
    # Get statistics
    summary = stats.get_summary()
    print(f"Total messages: {summary['total_messages']}")
    print(f"Total tokens: {summary['total_tokens']}")
    
    # Get usage over time
    daily_usage = stats.get_usage_by_day()
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MessageStats:
    """Statistics for a single message."""
    timestamp: str
    role: str
    tokens: int
    char_count: int
    word_count: int
    response_time: float = 0.0  # seconds
    conversation_id: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageStats:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConversationSummary:
    """Summary of a conversation."""
    conversation_id: str
    started_at: str
    ended_at: str
    duration_seconds: float
    message_count: int
    user_messages: int
    assistant_messages: int
    total_tokens: int
    user_tokens: int
    assistant_tokens: int
    avg_response_time: float
    topics: list[str] = field(default_factory=list)


@dataclass
class UsageMetrics:
    """Aggregated usage metrics."""
    period: str  # "day", "week", "month", "all"
    start_date: str
    end_date: str
    total_conversations: int
    total_messages: int
    total_tokens: int
    avg_messages_per_conversation: float
    avg_tokens_per_message: float
    avg_response_time: float
    busiest_hour: int
    most_active_day: str
    user_messages: int
    assistant_messages: int


class ConversationStats:
    """
    Track and analyze conversation statistics.
    """
    
    def __init__(self, data_path: Path | None = None):
        """
        Initialize conversation stats tracker.
        
        Args:
            data_path: Path to store statistics data
        """
        self._data_path = data_path or Path("data/stats")
        self._data_path.mkdir(parents=True, exist_ok=True)
        
        self._stats_file = self._data_path / "conversation_stats.json"
        self._messages: list[MessageStats] = []
        self._conversations: dict[str, list[MessageStats]] = defaultdict(list)
        
        # Current conversation tracking
        self._current_conversation_id: str | None = None
        self._conversation_start: datetime | None = None
        
        # Aggregated counters
        self._total_tokens = 0
        self._total_messages = 0
        self._response_times: list[float] = []
        
        self._load_stats()
    
    def _load_stats(self) -> None:
        """Load statistics from disk."""
        if self._stats_file.exists():
            try:
                with open(self._stats_file, encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for msg_data in data.get("messages", []):
                        msg = MessageStats.from_dict(msg_data)
                        self._messages.append(msg)
                        self._conversations[msg.conversation_id].append(msg)
                    
                    self._total_tokens = data.get("total_tokens", 0)
                    self._total_messages = data.get("total_messages", 0)
                    self._response_times = data.get("response_times", [])
                    
                logger.info(f"Loaded {len(self._messages)} message stats")
            except Exception as e:
                logger.error(f"Failed to load stats: {e}")
    
    def _save_stats(self) -> None:
        """Save statistics to disk."""
        try:
            # Keep only last 10000 messages to prevent unbounded growth
            recent_messages = self._messages[-10000:] if len(self._messages) > 10000 else self._messages
            recent_times = self._response_times[-10000:] if len(self._response_times) > 10000 else self._response_times
            
            data = {
                "messages": [m.to_dict() for m in recent_messages],
                "total_tokens": self._total_tokens,
                "total_messages": self._total_messages,
                "response_times": recent_times,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self._stats_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
                
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
    
    def start_conversation(self, conversation_id: str | None = None) -> str:
        """
        Start tracking a new conversation.
        
        Args:
            conversation_id: Optional ID, auto-generated if not provided
            
        Returns:
            Conversation ID
        """
        import uuid
        
        self._current_conversation_id = conversation_id or str(uuid.uuid4())[:8]
        self._conversation_start = datetime.now()
        
        logger.debug(f"Started conversation: {self._current_conversation_id}")
        return self._current_conversation_id
    
    def end_conversation(self) -> ConversationSummary | None:
        """
        End the current conversation and get summary.
        
        Returns:
            ConversationSummary or None
        """
        if not self._current_conversation_id:
            return None
        
        conv_id = self._current_conversation_id
        messages = self._conversations.get(conv_id, [])
        
        if not messages:
            self._current_conversation_id = None
            return None
        
        # Calculate summary
        user_msgs = [m for m in messages if m.role == "user"]
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        
        response_times = [m.response_time for m in assistant_msgs if m.response_time > 0]
        avg_response = sum(response_times) / len(response_times) if response_times else 0.0
        
        summary = ConversationSummary(
            conversation_id=conv_id,
            started_at=messages[0].timestamp,
            ended_at=messages[-1].timestamp,
            duration_seconds=(datetime.now() - self._conversation_start).total_seconds()
                if self._conversation_start else 0,
            message_count=len(messages),
            user_messages=len(user_msgs),
            assistant_messages=len(assistant_msgs),
            total_tokens=sum(m.tokens for m in messages),
            user_tokens=sum(m.tokens for m in user_msgs),
            assistant_tokens=sum(m.tokens for m in assistant_msgs),
            avg_response_time=avg_response
        )
        
        self._current_conversation_id = None
        self._conversation_start = None
        self._save_stats()
        
        return summary
    
    def track_message(
        self,
        role: str,
        content: str,
        tokens: int = 0,
        response_time: float = 0.0,
        conversation_id: str | None = None
    ) -> MessageStats:
        """
        Track a message.
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content
            tokens: Token count (estimated if 0)
            response_time: Response generation time
            conversation_id: Conversation ID
            
        Returns:
            MessageStats for the tracked message
        """
        conv_id = conversation_id or self._current_conversation_id or "default"
        
        # Estimate tokens if not provided
        if tokens == 0:
            tokens = len(content) // 4  # Rough estimate
        
        stats = MessageStats(
            timestamp=datetime.now().isoformat(),
            role=role,
            tokens=tokens,
            char_count=len(content),
            word_count=len(content.split()),
            response_time=response_time,
            conversation_id=conv_id
        )
        
        self._messages.append(stats)
        self._conversations[conv_id].append(stats)
        self._total_tokens += tokens
        self._total_messages += 1
        
        if response_time > 0:
            self._response_times.append(response_time)
        
        # Auto-save periodically
        if self._total_messages % 50 == 0:
            self._save_stats()
        
        return stats
    
    def get_summary(self) -> dict[str, Any]:
        """
        Get overall statistics summary.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self._messages:
            return {
                "total_messages": 0,
                "total_tokens": 0,
                "total_conversations": 0,
                "avg_tokens_per_message": 0,
                "avg_response_time": 0,
                "user_messages": 0,
                "assistant_messages": 0
            }
        
        user_msgs = [m for m in self._messages if m.role == "user"]
        assistant_msgs = [m for m in self._messages if m.role == "assistant"]
        
        valid_times = [t for t in self._response_times if t > 0]
        avg_time = sum(valid_times) / len(valid_times) if valid_times else 0
        
        return {
            "total_messages": self._total_messages,
            "total_tokens": self._total_tokens,
            "total_conversations": len(self._conversations),
            "avg_tokens_per_message": self._total_tokens / self._total_messages if self._total_messages else 0,
            "avg_response_time": avg_time,
            "user_messages": len(user_msgs),
            "assistant_messages": len(assistant_msgs),
            "user_tokens": sum(m.tokens for m in user_msgs),
            "assistant_tokens": sum(m.tokens for m in assistant_msgs),
            "avg_words_per_message": sum(m.word_count for m in self._messages) / len(self._messages),
            "first_message": self._messages[0].timestamp if self._messages else None,
            "last_message": self._messages[-1].timestamp if self._messages else None
        }
    
    def get_usage_by_day(self, days: int = 30) -> list[dict[str, Any]]:
        """
        Get usage statistics grouped by day.
        
        Args:
            days: Number of days to include
            
        Returns:
            List of daily statistics
        """
        cutoff = datetime.now() - timedelta(days=days)
        daily_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "messages": 0,
            "tokens": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "conversations": set()
        })
        
        for msg in self._messages:
            try:
                msg_date = datetime.fromisoformat(msg.timestamp)
                if msg_date < cutoff:
                    continue
                
                day_key = msg_date.strftime("%Y-%m-%d")
                daily_stats[day_key]["messages"] += 1
                daily_stats[day_key]["tokens"] += msg.tokens
                daily_stats[day_key]["conversations"].add(msg.conversation_id)
                
                if msg.role == "user":
                    daily_stats[day_key]["user_messages"] += 1
                elif msg.role == "assistant":
                    daily_stats[day_key]["assistant_messages"] += 1
                    
            except (ValueError, TypeError):
                continue
        
        # Convert sets to counts
        result = []
        for day, stats in sorted(daily_stats.items()):
            result.append({
                "date": day,
                "messages": stats["messages"],
                "tokens": stats["tokens"],
                "conversations": len(stats["conversations"]),
                "user_messages": stats["user_messages"],
                "assistant_messages": stats["assistant_messages"]
            })
        
        return result
    
    def get_usage_by_hour(self) -> dict[int, int]:
        """
        Get message distribution by hour of day.
        
        Returns:
            Dictionary mapping hour (0-23) to message count
        """
        hourly: Counter = Counter()
        
        for msg in self._messages:
            try:
                msg_date = datetime.fromisoformat(msg.timestamp)
                hourly[msg_date.hour] += 1
            except (ValueError, TypeError):
                continue
        
        return dict(hourly)
    
    def get_conversation_stats(self, conversation_id: str) -> ConversationSummary | None:
        """
        Get statistics for a specific conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            ConversationSummary or None
        """
        messages = self._conversations.get(conversation_id)
        if not messages:
            return None
        
        user_msgs = [m for m in messages if m.role == "user"]
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        
        response_times = [m.response_time for m in assistant_msgs if m.response_time > 0]
        avg_response = sum(response_times) / len(response_times) if response_times else 0.0
        
        # Calculate duration
        try:
            start = datetime.fromisoformat(messages[0].timestamp)
            end = datetime.fromisoformat(messages[-1].timestamp)
            duration = (end - start).total_seconds()
        except (ValueError, IndexError) as e:
            logger.debug(f"Could not calculate duration: {e}")
            duration = 0
        
        return ConversationSummary(
            conversation_id=conversation_id,
            started_at=messages[0].timestamp,
            ended_at=messages[-1].timestamp,
            duration_seconds=duration,
            message_count=len(messages),
            user_messages=len(user_msgs),
            assistant_messages=len(assistant_msgs),
            total_tokens=sum(m.tokens for m in messages),
            user_tokens=sum(m.tokens for m in user_msgs),
            assistant_tokens=sum(m.tokens for m in assistant_msgs),
            avg_response_time=avg_response
        )
    
    def get_top_conversations(self, limit: int = 10, by: str = "tokens") -> list[ConversationSummary]:
        """
        Get top conversations by a metric.
        
        Args:
            limit: Number of conversations
            by: Metric to sort by (tokens, messages, duration)
            
        Returns:
            List of ConversationSummary
        """
        summaries = []
        
        for conv_id in self._conversations:
            summary = self.get_conversation_stats(conv_id)
            if summary:
                summaries.append(summary)
        
        # Sort by metric
        sort_key = {
            "tokens": lambda s: s.total_tokens,
            "messages": lambda s: s.message_count,
            "duration": lambda s: s.duration_seconds
        }.get(by, lambda s: s.total_tokens)
        
        return sorted(summaries, key=sort_key, reverse=True)[:limit]
    
    def get_metrics(self, period: str = "all") -> UsageMetrics:
        """
        Get aggregated metrics for a time period.
        
        Args:
            period: "day", "week", "month", or "all"
            
        Returns:
            UsageMetrics
        """
        now = datetime.now()
        
        if period == "day":
            cutoff = now - timedelta(days=1)
        elif period == "week":
            cutoff = now - timedelta(weeks=1)
        elif period == "month":
            cutoff = now - timedelta(days=30)
        else:
            cutoff = datetime.min
        
        # Filter messages
        filtered = []
        for msg in self._messages:
            try:
                msg_date = datetime.fromisoformat(msg.timestamp)
                if msg_date >= cutoff:
                    filtered.append(msg)
            except ValueError:
                continue
        
        if not filtered:
            return UsageMetrics(
                period=period,
                start_date=cutoff.isoformat() if cutoff != datetime.min else "",
                end_date=now.isoformat(),
                total_conversations=0,
                total_messages=0,
                total_tokens=0,
                avg_messages_per_conversation=0,
                avg_tokens_per_message=0,
                avg_response_time=0,
                busiest_hour=0,
                most_active_day="",
                user_messages=0,
                assistant_messages=0
            )
        
        # Calculate metrics
        conversations = {m.conversation_id for m in filtered}
        user_msgs = [m for m in filtered if m.role == "user"]
        assistant_msgs = [m for m in filtered if m.role == "assistant"]
        response_times = [m.response_time for m in assistant_msgs if m.response_time > 0]
        
        # Hour distribution
        hours = Counter(
            datetime.fromisoformat(m.timestamp).hour 
            for m in filtered 
            if m.timestamp
        )
        busiest_hour = hours.most_common(1)[0][0] if hours else 0
        
        # Day distribution
        days = Counter(
            datetime.fromisoformat(m.timestamp).strftime("%A")
            for m in filtered
            if m.timestamp
        )
        most_active_day = days.most_common(1)[0][0] if days else ""
        
        total_tokens = sum(m.tokens for m in filtered)
        
        return UsageMetrics(
            period=period,
            start_date=filtered[0].timestamp if filtered else "",
            end_date=filtered[-1].timestamp if filtered else "",
            total_conversations=len(conversations),
            total_messages=len(filtered),
            total_tokens=total_tokens,
            avg_messages_per_conversation=len(filtered) / len(conversations) if conversations else 0,
            avg_tokens_per_message=total_tokens / len(filtered) if filtered else 0,
            avg_response_time=sum(response_times) / len(response_times) if response_times else 0,
            busiest_hour=busiest_hour,
            most_active_day=most_active_day,
            user_messages=len(user_msgs),
            assistant_messages=len(assistant_msgs)
        )
    
    def export_stats(self, format: str = "json") -> str:
        """
        Export statistics.
        
        Args:
            format: "json" or "csv"
            
        Returns:
            Formatted statistics string
        """
        if format == "csv":
            lines = ["timestamp,role,tokens,char_count,word_count,response_time,conversation_id"]
            for msg in self._messages:
                lines.append(f"{msg.timestamp},{msg.role},{msg.tokens},{msg.char_count},{msg.word_count},{msg.response_time},{msg.conversation_id}")
            return "\n".join(lines)
        
        return json.dumps({
            "summary": self.get_summary(),
            "messages": [m.to_dict() for m in self._messages[-1000:]]
        }, indent=2)
    
    def clear_stats(self) -> None:
        """Clear all statistics."""
        self._messages.clear()
        self._conversations.clear()
        self._total_tokens = 0
        self._total_messages = 0
        self._response_times.clear()
        self._save_stats()


# Singleton instance
_stats_instance: ConversationStats | None = None


def get_conversation_stats(data_path: Path | None = None) -> ConversationStats:
    """Get or create the singleton stats instance."""
    global _stats_instance
    if _stats_instance is None:
        _stats_instance = ConversationStats(data_path)
    return _stats_instance


# Convenience functions
def track_message(role: str, content: str, tokens: int = 0, response_time: float = 0.0) -> None:
    """Quick message tracking."""
    get_conversation_stats().track_message(role, content, tokens, response_time)


def get_stats_summary() -> dict[str, Any]:
    """Get quick summary."""
    return get_conversation_stats().get_summary()
