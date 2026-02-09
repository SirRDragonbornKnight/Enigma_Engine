"""
Automatic Summarization for Enigma AI Engine

Compress old conversations to save memory while preserving context.

Features:
- Extractive summarization
- Abstractive summarization
- Hierarchical compression
- Key point extraction
- Context preservation

Usage:
    from enigma_engine.memory.summarizer import Summarizer, get_summarizer
    
    summarizer = get_summarizer()
    
    # Summarize conversation
    summary = summarizer.summarize(conversation_history)
    
    # Hierarchical compression
    compressed = summarizer.compress_history(messages, max_tokens=1000)
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SummaryMethod(Enum):
    """Summarization methods."""
    EXTRACTIVE = "extractive"  # Select important sentences
    ABSTRACTIVE = "abstractive"  # Generate new summary
    HYBRID = "hybrid"  # Combine both
    HIERARCHICAL = "hierarchical"  # Progressive compression


@dataclass
class SummaryConfig:
    """Configuration for summarization."""
    method: SummaryMethod = SummaryMethod.HYBRID
    
    # Length control
    target_ratio: float = 0.3  # Target 30% of original
    min_length: int = 50
    max_length: int = 500
    
    # Extractive
    num_sentences: int = 5
    
    # Key points
    extract_key_points: bool = True
    max_key_points: int = 5
    
    # Preservation
    preserve_entities: bool = True
    preserve_numbers: bool = True


@dataclass
class Summary:
    """Summary result."""
    text: str
    method: str
    
    # Statistics
    original_length: int = 0
    summary_length: int = 0
    compression_ratio: float = 0.0
    
    # Extracted info
    key_points: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "method": self.method,
            "original_length": self.original_length,
            "summary_length": self.summary_length,
            "compression_ratio": round(self.compression_ratio, 2),
            "key_points": self.key_points,
            "entities": self.entities
        }


class TextPreprocessor:
    """Preprocess text for summarization."""
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simple pattern matching)."""
        # Capitalized words that aren't sentence starts
        entities = []
        
        sentences = self.split_sentences(text)
        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words[1:], 1):  # Skip first word
                if word[0].isupper() and word.isalpha():
                    entities.append(word)
        
        return list(set(entities))
    
    def extract_numbers(self, text: str) -> List[str]:
        """Extract numbers and quantities."""
        patterns = [
            r'\$[\d,]+(?:\.\d+)?',  # Money
            r'\d+(?:\.\d+)?%',  # Percentages
            r'\d{4}',  # Years
            r'\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand))?',  # Numbers
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numbers.extend(matches)
        
        return list(set(numbers))


class ExtractivesSummarizer:
    """Extractive summarization using sentence scoring."""
    
    def __init__(self):
        self._preprocessor = TextPreprocessor()
        
        # Stop words
        self._stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because",
            "until", "while", "although", "though", "after", "before"
        }
    
    def score_sentences(
        self,
        sentences: List[str],
        word_freq: Dict[str, float]
    ) -> List[Tuple[int, float]]:
        """
        Score sentences by word importance.
        
        Returns:
            List of (sentence_index, score)
        """
        scored = []
        
        for i, sentence in enumerate(sentences):
            words = self._preprocessor.tokenize(sentence)
            
            if not words:
                scored.append((i, 0.0))
                continue
            
            # Score based on word frequency
            score = sum(word_freq.get(w, 0) for w in words if w not in self._stop_words)
            
            # Normalize by sentence length
            score = score / len(words)
            
            # Boost for position (first/last sentences often important)
            if i == 0:
                score *= 1.2
            elif i == len(sentences) - 1:
                score *= 1.1
            
            scored.append((i, score))
        
        return scored
    
    def summarize(
        self,
        text: str,
        num_sentences: int = 5
    ) -> str:
        """
        Extract top sentences as summary.
        
        Args:
            text: Input text
            num_sentences: Number of sentences to extract
            
        Returns:
            Extractive summary
        """
        sentences = self._preprocessor.split_sentences(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Build word frequency
        all_words = self._preprocessor.tokenize(text)
        word_freq = Counter(all_words)
        
        # Normalize frequencies
        max_freq = max(word_freq.values()) if word_freq else 1
        word_freq = {w: f / max_freq for w, f in word_freq.items()}
        
        # Score sentences
        scored = self.score_sentences(sentences, word_freq)
        
        # Get top sentences (preserving order)
        top_indices = sorted(
            [idx for idx, _ in sorted(scored, key=lambda x: -x[1])[:num_sentences]]
        )
        
        return " ".join(sentences[i] for i in top_indices)


class AbstractiveSummarizer:
    """Abstractive summarization using model generation."""
    
    def __init__(self, model: Optional[Any] = None):
        self._model = model
    
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        model: Optional[Any] = None
    ) -> str:
        """
        Generate abstractive summary.
        
        Args:
            text: Input text
            max_length: Max summary length
            model: Override model
            
        Returns:
            Generated summary
        """
        model = model or self._model
        
        if model is None:
            # Fallback to extractive
            return ExtractivesSummarizer().summarize(text, num_sentences=3)
        
        prompt = f"""Summarize the following text concisely:

Text: {text}

Summary:"""
        
        if hasattr(model, 'generate'):
            summary = model.generate(prompt, max_new_tokens=max_length)
            # Extract just the summary part
            if "Summary:" in summary:
                summary = summary.split("Summary:")[-1].strip()
            return summary
        
        return text[:max_length]


class KeyPointExtractor:
    """Extract key points from text."""
    
    def __init__(self):
        self._preprocessor = TextPreprocessor()
        
        # Patterns that often indicate key points
        self._key_patterns = [
            r'(?:important|key|main|crucial|essential|significant)(?:ly)?\s+(.+?)(?:\.|$)',
            r'(?:first|second|third|finally|lastly)[,\s]+(.+?)(?:\.|$)',
            r'(?:in conclusion|to summarize|in summary)[,\s]+(.+?)(?:\.|$)',
            r'(?:the main|the key|the most)[^.]+(?:\.|$)'
        ]
    
    def extract(self, text: str, max_points: int = 5) -> List[str]:
        """
        Extract key points from text.
        
        Args:
            text: Input text
            max_points: Maximum number of points
            
        Returns:
            List of key points
        """
        key_points = []
        
        # Match patterns
        for pattern in self._key_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_points.extend(matches)
        
        # Also include sentences with important words
        sentences = self._preprocessor.split_sentences(text)
        important_words = {'important', 'key', 'main', 'note', 'remember', 'critical'}
        
        for sentence in sentences:
            words = set(self._preprocessor.tokenize(sentence))
            if words & important_words:
                key_points.append(sentence)
        
        # Deduplicate and limit
        seen = set()
        unique_points = []
        for point in key_points:
            normalized = point.lower().strip()
            if normalized not in seen and len(normalized) > 10:
                seen.add(normalized)
                unique_points.append(point.strip())
        
        return unique_points[:max_points]


class ConversationCompressor:
    """Compress conversation history."""
    
    def __init__(self, config: SummaryConfig):
        self._config = config
        self._extractive = ExtractivesSummarizer()
        self._key_extractor = KeyPointExtractor()
        self._preprocessor = TextPreprocessor()
    
    def compress_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000
    ) -> List[Dict[str, str]]:
        """
        Compress conversation messages.
        
        Args:
            messages: List of {role, content} dicts
            max_tokens: Target token limit
            
        Returns:
            Compressed messages
        """
        # Estimate current token count (rough: 1 token per 4 chars)
        def estimate_tokens(msgs):
            return sum(len(m.get("content", "")) // 4 for m in msgs)
        
        current_tokens = estimate_tokens(messages)
        
        if current_tokens <= max_tokens:
            return messages
        
        # Strategy: Keep recent, summarize old
        compressed = []
        
        # Keep last few messages intact
        keep_recent = min(4, len(messages) // 2)
        recent = messages[-keep_recent:]
        old = messages[:-keep_recent]
        
        if old:
            # Summarize old messages
            old_text = "\n".join(
                f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                for m in old
            )
            
            summary = self._extractive.summarize(
                old_text,
                num_sentences=self._config.num_sentences
            )
            
            # Extract key points
            key_points = self._key_extractor.extract(old_text, self._config.max_key_points)
            
            # Create summary message
            summary_content = f"[Earlier conversation summary]\n{summary}"
            if key_points:
                summary_content += "\n\nKey points:\n" + "\n".join(f"- {p}" for p in key_points)
            
            compressed.append({
                "role": "system",
                "content": summary_content
            })
        
        # Add recent messages
        compressed.extend(recent)
        
        return compressed
    
    def hierarchical_compress(
        self,
        messages: List[Dict[str, str]],
        levels: int = 3
    ) -> str:
        """
        Hierarchical compression over multiple levels.
        
        Args:
            messages: Conversation messages
            levels: Number of compression levels
            
        Returns:
            Highly compressed summary
        """
        # Combine all messages
        full_text = "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}"
            for m in messages
        )
        
        current = full_text
        
        for level in range(levels):
            # Each level compresses further
            target_sentences = max(3, self._config.num_sentences - level * 2)
            current = self._extractive.summarize(current, num_sentences=target_sentences)
        
        return current


class Summarizer:
    """High-level summarization interface."""
    
    def __init__(
        self,
        config: Optional[SummaryConfig] = None,
        model: Optional[Any] = None
    ):
        """
        Initialize summarizer.
        
        Args:
            config: Summarization configuration
            model: Optional model for abstractive summarization
        """
        self._config = config or SummaryConfig()
        
        self._extractive = ExtractivesSummarizer()
        self._abstractive = AbstractiveSummarizer(model)
        self._key_extractor = KeyPointExtractor()
        self._compressor = ConversationCompressor(self._config)
        self._preprocessor = TextPreprocessor()
    
    def summarize(
        self,
        text: str,
        method: Optional[SummaryMethod] = None
    ) -> Summary:
        """
        Summarize text.
        
        Args:
            text: Input text
            method: Override method
            
        Returns:
            Summary result
        """
        method = method or self._config.method
        original_length = len(text)
        
        if method == SummaryMethod.EXTRACTIVE:
            summary_text = self._extractive.summarize(
                text,
                num_sentences=self._config.num_sentences
            )
        elif method == SummaryMethod.ABSTRACTIVE:
            summary_text = self._abstractive.summarize(
                text,
                max_length=self._config.max_length
            )
        elif method == SummaryMethod.HYBRID:
            # First extractive, then abstractive polish
            extracted = self._extractive.summarize(
                text,
                num_sentences=self._config.num_sentences + 2
            )
            summary_text = self._abstractive.summarize(
                extracted,
                max_length=self._config.max_length
            )
        else:
            summary_text = self._extractive.summarize(text)
        
        # Extract metadata
        key_points = []
        entities = []
        
        if self._config.extract_key_points:
            key_points = self._key_extractor.extract(text, self._config.max_key_points)
        
        if self._config.preserve_entities:
            entities = self._preprocessor.extract_entities(text)
        
        summary_length = len(summary_text)
        
        return Summary(
            text=summary_text,
            method=method.value,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=summary_length / original_length if original_length > 0 else 0,
            key_points=key_points,
            entities=entities
        )
    
    def summarize_conversation(
        self,
        messages: List[Dict[str, str]]
    ) -> Summary:
        """Summarize a conversation."""
        full_text = "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}"
            for m in messages
        )
        return self.summarize(full_text)
    
    def compress_history(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000
    ) -> List[Dict[str, str]]:
        """
        Compress conversation history.
        
        Args:
            messages: Conversation messages
            max_tokens: Target token limit
            
        Returns:
            Compressed messages
        """
        return self._compressor.compress_messages(messages, max_tokens)
    
    def get_key_points(self, text: str) -> List[str]:
        """Extract key points from text."""
        return self._key_extractor.extract(text, self._config.max_key_points)


# Global instance
_summarizer: Optional[Summarizer] = None


def get_summarizer(
    config: Optional[SummaryConfig] = None,
    model: Optional[Any] = None
) -> Summarizer:
    """Get or create global summarizer."""
    global _summarizer
    if _summarizer is None or config is not None:
        _summarizer = Summarizer(config, model)
    return _summarizer
