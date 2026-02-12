"""
Semantic Memory Search for Enigma AI Engine

Natural language queries over conversation history and memories.

Features:
- "What did I say about X last week?"
- "Find conversations about Python"
- "When did we discuss the project?"
- Temporal filtering (today, this week, last month)
- Relevance ranking
- Context retrieval

Usage:
    from enigma_engine.memory.semantic_search import SemanticMemorySearch
    
    search = SemanticMemorySearch(conversation_manager)
    
    # Natural language search
    results = search.query("What did I say about machine learning?")
    
    # With time filter
    results = search.query("our discussion about APIs", time_filter="this week")
    
    # Get context around a result
    context = search.get_context(result_id, window=5)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Try imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class TimeFilter(Enum):
    """Time filter presets."""
    ALL = auto()
    TODAY = auto()
    YESTERDAY = auto()
    THIS_WEEK = auto()
    LAST_WEEK = auto()
    THIS_MONTH = auto()
    LAST_MONTH = auto()
    LAST_3_MONTHS = auto()
    THIS_YEAR = auto()


@dataclass
class SearchResult:
    """A single search result."""
    # Unique identifier
    id: str
    
    # The matching message/memory
    content: str
    
    # Who said it (user/assistant/system)
    role: str
    
    # Relevance score (0-1)
    score: float
    
    # When it was said
    timestamp: datetime
    
    # Conversation ID
    conversation_id: str
    
    # Surrounding context
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchQuery:
    """Parsed search query."""
    # Main search text
    text: str
    
    # Time filter
    time_filter: Optional[TimeFilter] = None
    
    # Role filter (user/assistant/both)
    role_filter: Optional[str] = None
    
    # Minimum relevance
    min_score: float = 0.0
    
    # Max results
    limit: int = 10
    
    # Include context
    include_context: bool = True
    context_window: int = 3


class SemanticMemorySearch:
    """
    Natural language search over conversation history.
    
    Uses embeddings and keyword matching to find relevant
    past conversations.
    """
    
    def __init__(
        self,
        conversation_manager=None,
        embedding_model: Optional[str] = None,
        cache_embeddings: bool = True
    ):
        """
        Initialize semantic search.
        
        Args:
            conversation_manager: ConversationManager instance
            embedding_model: Model to use for embeddings
            cache_embeddings: Whether to cache computed embeddings
        """
        self._conv_manager = conversation_manager
        self._embedding_model = embedding_model
        self._cache_embeddings = cache_embeddings
        
        # Embedding cache: message_id -> embedding
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        # Try to load vector DB
        self._vector_db = None
        try:
            from enigma_engine.memory.vector_db import get_vector_db
            self._vector_db = get_vector_db()
        except ImportError:
            logger.warning("Vector DB not available, using keyword search only")
        
        # Time filter patterns
        self._time_patterns = [
            (r'\btoday\b', TimeFilter.TODAY),
            (r'\byesterday\b', TimeFilter.YESTERDAY),
            (r'\bthis week\b', TimeFilter.THIS_WEEK),
            (r'\blast week\b', TimeFilter.LAST_WEEK),
            (r'\bthis month\b', TimeFilter.THIS_MONTH),
            (r'\blast month\b', TimeFilter.LAST_MONTH),
            (r'\blast (?:3|three) months?\b', TimeFilter.LAST_3_MONTHS),
            (r'\bthis year\b', TimeFilter.THIS_YEAR),
            (r'\brecently\b', TimeFilter.THIS_WEEK),
        ]
        
        logger.info("SemanticMemorySearch initialized")
    
    def query(
        self,
        query: str,
        time_filter: Optional[Union[str, TimeFilter]] = None,
        role_filter: Optional[str] = None,
        limit: int = 10,
        min_score: float = 0.1,
        include_context: bool = True
    ) -> List[SearchResult]:
        """
        Search memories with natural language query.
        
        Args:
            query: Natural language query
            time_filter: Time filter (string like "this week" or TimeFilter enum)
            role_filter: Filter by role ("user", "assistant", or None for both)
            limit: Maximum results to return
            min_score: Minimum relevance score (0-1)
            include_context: Include surrounding messages
            
        Returns:
            List of SearchResult objects
        """
        # Parse query
        parsed = self._parse_query(query, time_filter, role_filter, limit, min_score)
        
        # Get messages within time range
        messages = self._get_messages_in_range(parsed.time_filter)
        
        if not messages:
            return []
        
        # Score messages
        scored = self._score_messages(messages, parsed.text)
        
        # Filter and sort
        results = []
        for msg, score in sorted(scored, key=lambda x: x[1], reverse=True):
            if score < parsed.min_score:
                continue
            
            if parsed.role_filter and msg.get('role') != parsed.role_filter:
                continue
            
            result = SearchResult(
                id=msg.get('id', str(hash(msg.get('content', '')))),
                content=msg.get('content', ''),
                role=msg.get('role', 'unknown'),
                score=score,
                timestamp=self._parse_timestamp(msg.get('timestamp')),
                conversation_id=msg.get('conversation_id', ''),
                metadata=msg.get('metadata', {})
            )
            
            # Add context if requested
            if include_context:
                result.context_before, result.context_after = self._get_context(
                    msg, parsed.context_window
                )
            
            results.append(result)
            
            if len(results) >= parsed.limit:
                break
        
        return results
    
    def find_mentions(self, topic: str, limit: int = 20) -> List[SearchResult]:
        """
        Find all mentions of a specific topic.
        
        Args:
            topic: Topic to search for
            limit: Maximum results
            
        Returns:
            List of results mentioning the topic
        """
        return self.query(
            topic,
            min_score=0.3,  # Higher threshold for topic search
            limit=limit
        )
    
    def find_questions(self, about: Optional[str] = None) -> List[SearchResult]:
        """
        Find questions asked by the user.
        
        Args:
            about: Optional topic filter
            
        Returns:
            List of questions
        """
        messages = self._get_all_messages()
        questions = []
        
        for msg in messages:
            content = msg.get('content', '')
            if msg.get('role') == 'user' and '?' in content:
                # Score by relevance to topic if specified
                if about:
                    _, score = self._score_messages([msg], about)[0]
                    if score < 0.2:
                        continue
                else:
                    score = 1.0
                
                questions.append(SearchResult(
                    id=msg.get('id', str(hash(content))),
                    content=content,
                    role='user',
                    score=score,
                    timestamp=self._parse_timestamp(msg.get('timestamp')),
                    conversation_id=msg.get('conversation_id', '')
                ))
        
        return sorted(questions, key=lambda x: x.timestamp, reverse=True)[:20]
    
    def find_similar(self, text: str, limit: int = 5) -> List[SearchResult]:
        """
        Find messages similar to the given text.
        
        Args:
            text: Reference text
            limit: Maximum results
            
        Returns:
            List of similar messages
        """
        if self._vector_db:
            # Use vector similarity
            try:
                similar = self._vector_db.search(text, k=limit)
                results = []
                for doc, score in similar:
                    results.append(SearchResult(
                        id=doc.get('id', ''),
                        content=doc.get('content', ''),
                        role=doc.get('role', 'unknown'),
                        score=score,
                        timestamp=self._parse_timestamp(doc.get('timestamp')),
                        conversation_id=doc.get('conversation_id', '')
                    ))
                return results
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
        
        # Fallback to keyword search
        return self.query(text, limit=limit)
    
    def get_context(
        self,
        result: SearchResult,
        window: int = 5
    ) -> Tuple[List[str], List[str]]:
        """
        Get extended context around a search result.
        
        Args:
            result: SearchResult to get context for
            window: Number of messages before/after
            
        Returns:
            Tuple of (messages_before, messages_after)
        """
        # Find the original message
        messages = self._get_messages_for_conversation(result.conversation_id)
        
        for i, msg in enumerate(messages):
            if msg.get('content') == result.content:
                before = [m.get('content', '') for m in messages[max(0, i-window):i]]
                after = [m.get('content', '') for m in messages[i+1:i+1+window]]
                return before, after
        
        return [], []
    
    def summarize_topic(self, topic: str) -> str:
        """
        Summarize what has been discussed about a topic.
        
        Args:
            topic: Topic to summarize
            
        Returns:
            Summary string
        """
        results = self.find_mentions(topic, limit=20)
        
        if not results:
            return f"No discussions found about '{topic}'."
        
        # Group by date
        by_date = {}
        for r in results:
            date_str = r.timestamp.strftime('%Y-%m-%d')
            if date_str not in by_date:
                by_date[date_str] = []
            by_date[date_str].append(r)
        
        # Build summary
        lines = [f"Found {len(results)} mentions of '{topic}':"]
        
        for date_str in sorted(by_date.keys(), reverse=True)[:5]:
            items = by_date[date_str]
            lines.append(f"\n{date_str}: {len(items)} mentions")
            
            # Show top items
            for item in sorted(items, key=lambda x: x.score, reverse=True)[:2]:
                preview = item.content[:100] + "..." if len(item.content) > 100 else item.content
                lines.append(f"  - [{item.role}]: {preview}")
        
        return "\n".join(lines)
    
    def _parse_query(
        self,
        query: str,
        time_filter: Optional[Union[str, TimeFilter]],
        role_filter: Optional[str],
        limit: int,
        min_score: float
    ) -> SearchQuery:
        """Parse a natural language query."""
        parsed = SearchQuery(
            text=query,
            time_filter=None,
            role_filter=role_filter,
            limit=limit,
            min_score=min_score
        )
        
        # Handle explicit time filter
        if time_filter:
            if isinstance(time_filter, str):
                time_filter_lower = time_filter.lower().replace('_', ' ')
                for pattern, tf in self._time_patterns:
                    if re.search(pattern, time_filter_lower, re.IGNORECASE):
                        parsed.time_filter = tf
                        break
            else:
                parsed.time_filter = time_filter
        else:
            # Extract time filter from query
            cleaned_query = query
            for pattern, tf in self._time_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    parsed.time_filter = tf
                    cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
                    break
            
            parsed.text = cleaned_query.strip()
        
        # Extract role filter from query
        if 'i said' in query.lower() or 'my question' in query.lower():
            parsed.role_filter = 'user'
        elif 'you said' in query.lower() or 'your response' in query.lower():
            parsed.role_filter = 'assistant'
        
        return parsed
    
    def _get_time_range(self, time_filter: Optional[TimeFilter]) -> Tuple[datetime, datetime]:
        """Get datetime range for a time filter."""
        now = datetime.now()
        end = now
        
        if time_filter == TimeFilter.TODAY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_filter == TimeFilter.YESTERDAY:
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_filter == TimeFilter.THIS_WEEK:
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_filter == TimeFilter.LAST_WEEK:
            end = now - timedelta(days=now.weekday())
            start = end - timedelta(days=7)
        elif time_filter == TimeFilter.THIS_MONTH:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif time_filter == TimeFilter.LAST_MONTH:
            first_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = first_of_month
            start = (first_of_month - timedelta(days=1)).replace(day=1)
        elif time_filter == TimeFilter.LAST_3_MONTHS:
            start = now - timedelta(days=90)
        elif time_filter == TimeFilter.THIS_YEAR:
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            # ALL - go back far
            start = datetime(2000, 1, 1)
        
        return start, end
    
    def _get_messages_in_range(self, time_filter: Optional[TimeFilter]) -> List[Dict]:
        """Get messages within time range."""
        start, end = self._get_time_range(time_filter)
        
        all_messages = self._get_all_messages()
        
        filtered = []
        for msg in all_messages:
            ts = self._parse_timestamp(msg.get('timestamp'))
            if start <= ts <= end:
                filtered.append(msg)
        
        return filtered
    
    def _get_all_messages(self) -> List[Dict]:
        """Get all messages from conversation manager."""
        if self._conv_manager is None:
            return []
        
        try:
            # Try to get all conversations
            if hasattr(self._conv_manager, 'get_all_conversations'):
                convs = self._conv_manager.get_all_conversations()
                messages = []
                for conv in convs:
                    conv_id = conv.get('id', '')
                    for msg in conv.get('messages', []):
                        msg['conversation_id'] = conv_id
                        messages.append(msg)
                return messages
            
            # Fallback to current conversation
            if hasattr(self._conv_manager, 'get_history'):
                return self._conv_manager.get_history()
            
        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
        
        return []
    
    def _get_messages_for_conversation(self, conv_id: str) -> List[Dict]:
        """Get messages for a specific conversation."""
        if self._conv_manager is None:
            return []
        
        try:
            if hasattr(self._conv_manager, 'get_conversation'):
                conv = self._conv_manager.get_conversation(conv_id)
                return conv.get('messages', []) if conv else []
        except Exception as e:
            logger.error(f"Failed to get conversation: {e}")
        
        return []
    
    def _score_messages(
        self,
        messages: List[Dict],
        query: str
    ) -> List[Tuple[Dict, float]]:
        """Score messages by relevance to query."""
        scored = []
        
        # Prepare query
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for msg in messages:
            content = msg.get('content', '')
            content_lower = content.lower()
            content_words = set(content_lower.split())
            
            # Calculate score
            score = 0.0
            
            # Exact substring match (high score)
            if query_lower in content_lower:
                score += 0.5
            
            # Word overlap
            overlap = len(query_words & content_words)
            if query_words:
                word_score = overlap / len(query_words)
                score += word_score * 0.3
            
            # Use embeddings if available
            if self._vector_db and NUMPY_AVAILABLE:
                try:
                    embedding_score = self._get_embedding_similarity(query, content)
                    score += embedding_score * 0.2
                except Exception:
                    pass  # Intentionally silent
            
            scored.append((msg, min(1.0, score)))
        
        return scored
    
    def _get_embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate embedding similarity between two texts."""
        if not self._vector_db:
            return 0.0
        
        try:
            emb1 = self._get_embedding(text1)
            emb2 = self._get_embedding(text2)
            
            if emb1 is not None and emb2 is not None:
                # Cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                return float(max(0, similarity))
        except Exception:
            pass  # Intentionally silent
        
        return 0.0
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get or compute embedding for text."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        if self._vector_db and hasattr(self._vector_db, 'embed'):
            try:
                embedding = self._vector_db.embed(text)
                if self._cache_embeddings:
                    self._embedding_cache[text] = embedding
                return embedding
            except Exception:
                pass  # Intentionally silent
        
        return None
    
    def _get_context(self, msg: Dict, window: int) -> Tuple[List[str], List[str]]:
        """Get context around a message."""
        conv_id = msg.get('conversation_id', '')
        if not conv_id:
            return [], []
        
        messages = self._get_messages_for_conversation(conv_id)
        content = msg.get('content', '')
        
        for i, m in enumerate(messages):
            if m.get('content') == content:
                before = [x.get('content', '') for x in messages[max(0, i-window):i]]
                after = [x.get('content', '') for x in messages[i+1:i+1+window]]
                return before, after
        
        return [], []
    
    def _parse_timestamp(self, ts: Any) -> datetime:
        """Parse timestamp to datetime."""
        if isinstance(ts, datetime):
            return ts
        
        if isinstance(ts, str):
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
                try:
                    return datetime.strptime(ts, fmt)
                except ValueError:
                    continue
        
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts)
        
        return datetime.now()


# Convenience functions
def search_memories(query: str, **kwargs) -> List[SearchResult]:
    """Quick function to search memories."""
    try:
        from enigma_engine.memory.manager import get_conversation_manager
        manager = get_conversation_manager()
        search = SemanticMemorySearch(manager)
        return search.query(query, **kwargs)
    except ImportError:
        logger.warning("Memory manager not available")
        return []


def what_did_i_say_about(topic: str) -> List[SearchResult]:
    """Find what the user said about a topic."""
    return search_memories(topic, role_filter='user')


def what_was_discussed(topic: str) -> str:
    """Get summary of what was discussed about a topic."""
    try:
        from enigma_engine.memory.manager import get_conversation_manager
        manager = get_conversation_manager()
        search = SemanticMemorySearch(manager)
        return search.summarize_topic(topic)
    except ImportError:
        return "Memory system not available"
