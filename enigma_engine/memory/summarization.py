"""
Automatic Conversation Summarization for Enigma AI Engine

Compress old conversations automatically.

Features:
- Extractive summarization
- Abstractive summarization
- Progressive compression
- Key point extraction
- Summary chaining

Usage:
    from enigma_engine.memory.summarization import ConversationSummarizer
    
    summarizer = ConversationSummarizer()
    
    # Summarize a conversation
    summary = summarizer.summarize(messages, max_length=200)
    
    # Progressive summarization
    summary = summarizer.progressive_summarize(messages)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A conversation message."""
    role: str  # user, assistant, system
    content: str
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Summary:
    """A conversation summary."""
    text: str
    original_count: int
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    key_points: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    compression_ratio: float = 0.0


class ConversationSummarizer:
    """Summarize conversations."""
    
    def __init__(self, model: Optional[Any] = None):
        """
        Initialize summarizer.
        
        Args:
            model: Optional language model for abstractive summarization
        """
        self.model = model
        
        # Stop words for extractive summarization
        self._stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'i', 'me', 'my', 'myself', 'you', 'your', 'yourself',
            'he', 'him', 'his', 'she', 'her', 'it', 'its', 'we',
            'they', 'them', 'their', 'this', 'that', 'these', 'those'
        }
    
    def summarize(
        self,
        messages: List[Message],
        max_length: int = 200,
        method: str = "extractive"
    ) -> Summary:
        """
        Summarize a conversation.
        
        Args:
            messages: List of messages
            max_length: Maximum summary length
            method: "extractive" or "abstractive"
            
        Returns:
            Summary
        """
        if method == "abstractive" and self.model:
            return self._abstractive_summarize(messages, max_length)
        else:
            return self._extractive_summarize(messages, max_length)
    
    def _extractive_summarize(
        self,
        messages: List[Message],
        max_length: int
    ) -> Summary:
        """Extract key sentences for summary."""
        # Combine messages
        all_text = " ".join(m.content for m in messages)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', all_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Score sentences
        scored = []
        for sentence in sentences:
            score = self._score_sentence(sentence, all_text)
            scored.append((score, sentence))
        
        # Sort by score
        scored.sort(reverse=True)
        
        # Select top sentences until max_length
        summary_sentences = []
        total_length = 0
        
        for score, sentence in scored:
            if total_length + len(sentence) <= max_length:
                summary_sentences.append(sentence)
                total_length += len(sentence) + 1
        
        # Reorder by original position
        ordered = sorted(
            summary_sentences,
            key=lambda s: all_text.find(s)
        )
        
        summary_text = ". ".join(ordered)
        if summary_text and not summary_text.endswith('.'):
            summary_text += "."
        
        # Extract key points
        key_points = self._extract_key_points(messages)
        
        # Extract topics
        topics = self._extract_topics(all_text)
        
        return Summary(
            text=summary_text,
            original_count=len(messages),
            key_points=key_points,
            topics=topics,
            compression_ratio=len(summary_text) / max(1, len(all_text))
        )
    
    def _score_sentence(self, sentence: str, full_text: str) -> float:
        """Score a sentence for importance."""
        words = sentence.lower().split()
        if not words:
            return 0.0
        
        # Remove stop words
        content_words = [w for w in words if w not in self._stop_words]
        
        if not content_words:
            return 0.0
        
        # Score based on word frequency in full text
        full_lower = full_text.lower()
        score = 0.0
        
        for word in content_words:
            frequency = full_lower.count(word)
            score += frequency
        
        # Normalize by sentence length
        score /= len(words)
        
        # Bonus for question marks (likely important)
        if '?' in sentence:
            score *= 1.2
        
        # Bonus for named entities (capitalized words)
        caps = len([w for w in sentence.split() if w[0].isupper()])
        score += caps * 0.5
        
        return score
    
    def _abstractive_summarize(
        self,
        messages: List[Message],
        max_length: int
    ) -> Summary:
        """Generate abstractive summary using language model."""
        if not self.model:
            return self._extractive_summarize(messages, max_length)
        
        # Build prompt
        conversation = "\n".join(
            f"{m.role}: {m.content}"
            for m in messages
        )
        
        prompt = f"""Summarize this conversation in {max_length} characters or less:

{conversation}

Summary:"""
        
        try:
            if hasattr(self.model, 'generate'):
                summary_text = self.model.generate(prompt, max_length=max_length)
            elif hasattr(self.model, '__call__'):
                summary_text = self.model(prompt)[:max_length]
            else:
                return self._extractive_summarize(messages, max_length)
            
            return Summary(
                text=summary_text,
                original_count=len(messages),
                key_points=self._extract_key_points(messages),
                topics=self._extract_topics(conversation),
                compression_ratio=len(summary_text) / max(1, len(conversation))
            )
            
        except Exception as e:
            logger.error(f"Abstractive summarization failed: {e}")
            return self._extractive_summarize(messages, max_length)
    
    def _extract_key_points(self, messages: List[Message]) -> List[str]:
        """Extract key points from messages."""
        key_points = []
        
        for msg in messages:
            content = msg.content
            
            # Look for bullet points
            bullets = re.findall(r'[•\-\*]\s*(.+)', content)
            key_points.extend(bullets[:3])
            
            # Look for numbered items
            numbered = re.findall(r'\d+[\.\)]\s*(.+)', content)
            key_points.extend(numbered[:3])
            
            # Look for "important" markers
            important = re.findall(r'(?:important|key|main|note):\s*(.+)', content, re.I)
            key_points.extend(important)
        
        return key_points[:10]  # Limit to 10 key points
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text."""
        words = text.lower().split()
        
        # Remove stop words
        content_words = [w for w in words if w not in self._stop_words and len(w) > 3]
        
        # Count word frequency
        freq: Dict[str, int] = {}
        for word in content_words:
            # Clean word
            word = re.sub(r'[^\w]', '', word)
            if word:
                freq[word] = freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(freq.items(), key=lambda x: -x[1])
        
        # Return top topics
        return [word for word, count in sorted_words[:5]]
    
    def progressive_summarize(
        self,
        messages: List[Message],
        chunk_size: int = 10,
        final_max_length: int = 500
    ) -> Summary:
        """
        Progressively summarize a long conversation.
        
        Args:
            messages: All messages
            chunk_size: Messages per chunk
            final_max_length: Max length of final summary
            
        Returns:
            Final summary
        """
        if len(messages) <= chunk_size:
            return self.summarize(messages, final_max_length)
        
        # Split into chunks
        chunks = [
            messages[i:i + chunk_size]
            for i in range(0, len(messages), chunk_size)
        ]
        
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = self.summarize(chunk, max_length=200)
            chunk_summaries.append(Message(
                role="summary",
                content=summary.text
            ))
        
        # Summarize the summaries
        if len(chunk_summaries) > chunk_size:
            return self.progressive_summarize(chunk_summaries, chunk_size, final_max_length)
        
        return self.summarize(chunk_summaries, final_max_length)


class MemoryImportanceScorer:
    """Score memory importance for retention."""
    
    def __init__(self):
        """Initialize scorer."""
        # Importance keywords
        self._high_importance_keywords = {
            'remember', 'important', 'never forget', 'always',
            'key', 'critical', 'essential', 'must', 'priority',
            'deadline', 'password', 'secret', 'personal'
        }
        
        self._medium_importance_keywords = {
            'like', 'prefer', 'favorite', 'hobby', 'interest',
            'birthday', 'name', 'location', 'work', 'job'
        }
    
    def score(self, message: Message, context: Optional[Dict] = None) -> float:
        """
        Score a message's importance.
        
        Args:
            message: The message to score
            context: Optional context (usage frequency, etc.)
            
        Returns:
            Importance score (0-1)
        """
        score = 0.5  # Base score
        content_lower = message.content.lower()
        
        # Check for high importance keywords
        for keyword in self._high_importance_keywords:
            if keyword in content_lower:
                score += 0.15
        
        # Check for medium importance keywords
        for keyword in self._medium_importance_keywords:
            if keyword in content_lower:
                score += 0.05
        
        # User messages are generally more important
        if message.role == "user":
            score += 0.1
        
        # Questions are important
        if '?' in message.content:
            score += 0.1
        
        # Length factor (very short or very long may be less important)
        length = len(message.content)
        if 50 <= length <= 500:
            score += 0.1
        
        # Context factors
        if context:
            # Frequently accessed memories are important
            access_count = context.get('access_count', 0)
            score += min(0.2, access_count * 0.02)
            
            # Recently accessed memories
            recency = context.get('recency_score', 0)
            score += recency * 0.1
        
        return min(1.0, max(0.0, score))
    
    def batch_score(
        self,
        messages: List[Message],
        contexts: Optional[List[Dict]] = None
    ) -> List[float]:
        """Score multiple messages."""
        if contexts is None:
            contexts = [None] * len(messages)
        
        return [
            self.score(msg, ctx)
            for msg, ctx in zip(messages, contexts)
        ]


# Convenience functions
def summarize_conversation(messages: List[Dict], max_length: int = 200) -> str:
    """Quick summarize a conversation."""
    summarizer = ConversationSummarizer()
    msg_objs = [Message(role=m.get('role', 'user'), content=m.get('content', '')) for m in messages]
    summary = summarizer.summarize(msg_objs, max_length)
    return summary.text


def get_importance_score(content: str, role: str = "user") -> float:
    """Quick get importance score."""
    scorer = MemoryImportanceScorer()
    return scorer.score(Message(role=role, content=content))
