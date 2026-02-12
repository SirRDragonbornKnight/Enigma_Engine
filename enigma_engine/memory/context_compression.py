"""
Context Compression for Enigma AI Engine

Fit more content in the context window through compression.

Features:
- Extractive summarization
- Importance-based filtering
- Hierarchical compression
- Token-aware truncation
- Semantic deduplication
- Key information extraction

Usage:
    from enigma_engine.memory.context_compression import ContextCompressor, compress_context
    
    # Quick compression
    compressed = compress_context(long_text, max_tokens=1000)
    
    # Detailed control
    compressor = ContextCompressor()
    result = compressor.compress(
        context=long_text,
        max_tokens=1000,
        preserve_recent=True
    )
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class CompressionStrategy(Enum):
    """Compression strategies."""
    TRUNCATE = auto()           # Simple truncation
    SUMMARIZE = auto()          # Extractive summarization
    IMPORTANCE = auto()         # Keep important parts
    HIERARCHICAL = auto()       # Multi-level compression
    SEMANTIC = auto()           # Semantic deduplication
    HYBRID = auto()             # Combination


@dataclass
class CompressionConfig:
    """Configuration for context compression."""
    strategy: CompressionStrategy = CompressionStrategy.HYBRID
    
    # Token limits
    max_tokens: int = 2048
    reserve_tokens: int = 100  # Reserve for response
    
    # Importance weights
    recency_weight: float = 0.3
    relevance_weight: float = 0.4
    information_weight: float = 0.3
    
    # Preservation
    preserve_system: bool = True
    preserve_recent_turns: int = 2
    preserve_keywords: List[str] = field(default_factory=list)
    
    # Summarization
    summary_ratio: float = 0.3  # Target compression ratio
    min_sentence_importance: float = 0.3
    
    # Deduplication
    similarity_threshold: float = 0.85


@dataclass
class CompressionResult:
    """Result of context compression."""
    text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    
    # Details
    preserved_sections: List[str] = field(default_factory=list)
    removed_sections: List[str] = field(default_factory=list)
    summary_created: bool = False
    
    def __str__(self):
        return self.text


class ContextCompressor:
    """
    Compresses context to fit within token limits.
    """
    
    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
        tokenizer=None
    ):
        """
        Initialize compressor.
        
        Args:
            config: Compression configuration
            tokenizer: Optional tokenizer for accurate counting
        """
        self._config = config or CompressionConfig()
        self._tokenizer = tokenizer
        
        # Load tokenizer if not provided
        if not self._tokenizer:
            try:
                from enigma_engine.core.tokenizer import get_tokenizer
                self._tokenizer = get_tokenizer()
            except Exception:
                pass  # Intentionally silent
        
        # Importance keywords
        self._important_keywords = [
            "important", "critical", "must", "never", "always",
            "remember", "note", "warning", "error", "key",
        ]
        
        # Stop words for keyword extraction
        self._stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may",
            "might", "must", "shall", "can", "need", "dare",
            "ought", "used", "to", "of", "in", "for", "on",
            "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between",
            "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each",
            "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or",
            "because", "until", "while", "what", "which", "who",
            "whom", "this", "that", "these", "those", "am", "it",
        }
    
    def compress(
        self,
        context: str | List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        query: str = "",
        **kwargs
    ) -> CompressionResult:
        """
        Compress context to fit token limit.
        
        Args:
            context: Text or list of chat messages
            max_tokens: Optional override for max tokens
            query: Optional query for relevance scoring
            **kwargs: Additional options
            
        Returns:
            Compression result
        """
        max_tokens = max_tokens or self._config.max_tokens
        effective_limit = max_tokens - self._config.reserve_tokens
        
        # Handle chat format
        if isinstance(context, list):
            context = self._messages_to_text(context)
        
        # Count original tokens
        original_tokens = self._count_tokens(context)
        
        if original_tokens <= effective_limit:
            # No compression needed
            return CompressionResult(
                text=context,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0
            )
        
        # Apply compression based on strategy
        if self._config.strategy == CompressionStrategy.TRUNCATE:
            compressed = self._compress_truncate(context, effective_limit)
        
        elif self._config.strategy == CompressionStrategy.SUMMARIZE:
            compressed = self._compress_summarize(context, effective_limit)
        
        elif self._config.strategy == CompressionStrategy.IMPORTANCE:
            compressed = self._compress_importance(context, effective_limit, query)
        
        elif self._config.strategy == CompressionStrategy.HIERARCHICAL:
            compressed = self._compress_hierarchical(context, effective_limit, query)
        
        elif self._config.strategy == CompressionStrategy.SEMANTIC:
            compressed = self._compress_semantic(context, effective_limit)
        
        else:  # HYBRID
            compressed = self._compress_hybrid(context, effective_limit, query)
        
        # Count compressed tokens
        compressed_tokens = self._count_tokens(compressed.text)
        compressed.original_tokens = original_tokens
        compressed.compressed_tokens = compressed_tokens
        compressed.compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        
        logger.debug(
            f"Compressed {original_tokens} -> {compressed_tokens} tokens "
            f"({compressed.compression_ratio:.1%})"
        )
        
        return compressed
    
    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to text format."""
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass  # Intentionally silent
        
        # Rough estimate: ~4 chars per token
        return len(text) // 4
    
    def _compress_truncate(self, text: str, max_tokens: int) -> CompressionResult:
        """Simple truncation from the beginning."""
        if self._tokenizer:
            tokens = self._tokenizer.encode(text)
            if len(tokens) > max_tokens:
                # Keep end (most recent)
                tokens = tokens[-max_tokens:]
                text = self._tokenizer.decode(tokens)
        else:
            # Character estimation
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                text = "..." + text[-max_chars:]
        
        return CompressionResult(
            text=text,
            original_tokens=0,
            compressed_tokens=0,
            compression_ratio=0,
        )
    
    def _compress_summarize(self, text: str, max_tokens: int) -> CompressionResult:
        """Extractive summarization."""
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 3:
            return self._compress_truncate(text, max_tokens)
        
        # Score sentences by importance
        scored = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, i, len(sentences))
            scored.append((sentence, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences within token limit
        selected = []
        total_tokens = 0
        
        for sentence, score in scored:
            sentence_tokens = self._count_tokens(sentence)
            if total_tokens + sentence_tokens <= max_tokens:
                selected.append((sentence, score))
                total_tokens += sentence_tokens
        
        # Restore original order
        # Get original indices
        sentence_indices = {s: i for i, s in enumerate(sentences)}
        selected.sort(key=lambda x: sentence_indices.get(x[0], 0))
        
        compressed_text = " ".join(s for s, _ in selected)
        
        return CompressionResult(
            text=compressed_text,
            original_tokens=0,
            compressed_tokens=total_tokens,
            compression_ratio=0,
            summary_created=True,
        )
    
    def _compress_importance(
        self,
        text: str,
        max_tokens: int,
        query: str = ""
    ) -> CompressionResult:
        """Keep important parts based on scoring."""
        # Split into paragraphs/sections
        sections = self._split_sections(text)
        
        # Score each section
        scored_sections = []
        for i, section in enumerate(sections):
            score = self._score_section(section, i, len(sections), query)
            scored_sections.append((section, score))
        
        # Select sections by importance
        scored_sections.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        total_tokens = 0
        preserved = []
        removed = []
        
        for section, score in scored_sections:
            section_tokens = self._count_tokens(section)
            
            if total_tokens + section_tokens <= max_tokens:
                selected.append((section, score))
                total_tokens += section_tokens
                preserved.append(section[:50] + "..." if len(section) > 50 else section)
            else:
                removed.append(section[:50] + "..." if len(section) > 50 else section)
        
        # Restore order
        section_indices = {s: i for i, s in enumerate(sections)}
        selected.sort(key=lambda x: section_indices.get(x[0], 0))
        
        compressed_text = "\n\n".join(s for s, _ in selected)
        
        return CompressionResult(
            text=compressed_text,
            original_tokens=0,
            compressed_tokens=total_tokens,
            compression_ratio=0,
            preserved_sections=preserved,
            removed_sections=removed,
        )
    
    def _compress_hierarchical(
        self,
        text: str,
        max_tokens: int,
        query: str = ""
    ) -> CompressionResult:
        """Multi-level compression."""
        current_tokens = self._count_tokens(text)
        
        # Level 1: Remove redundancy
        if current_tokens > max_tokens:
            text = self._remove_redundancy(text)
            current_tokens = self._count_tokens(text)
        
        # Level 2: Summarize verbose sections
        if current_tokens > max_tokens:
            text = self._summarize_verbose(text, max_tokens)
            current_tokens = self._count_tokens(text)
        
        # Level 3: Importance-based filtering
        if current_tokens > max_tokens:
            result = self._compress_importance(text, max_tokens, query)
            return result
        
        return CompressionResult(
            text=text,
            original_tokens=0,
            compressed_tokens=current_tokens,
            compression_ratio=0,
        )
    
    def _compress_semantic(self, text: str, max_tokens: int) -> CompressionResult:
        """Semantic deduplication."""
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 3:
            return self._compress_truncate(text, max_tokens)
        
        # Simple similarity check (without embeddings)
        unique_sentences = []
        seen_keywords = set()
        
        for sentence in sentences:
            keywords = self._extract_keywords(sentence)
            
            # Check overlap with seen
            overlap = len(keywords & seen_keywords) / max(1, len(keywords))
            
            if overlap < self._config.similarity_threshold:
                unique_sentences.append(sentence)
                seen_keywords.update(keywords)
        
        # Build compressed text
        compressed_text = " ".join(unique_sentences)
        
        # Further truncate if needed
        if self._count_tokens(compressed_text) > max_tokens:
            return self._compress_importance(compressed_text, max_tokens)
        
        return CompressionResult(
            text=compressed_text,
            original_tokens=0,
            compressed_tokens=self._count_tokens(compressed_text),
            compression_ratio=0,
        )
    
    def _compress_hybrid(
        self,
        text: str,
        max_tokens: int,
        query: str = ""
    ) -> CompressionResult:
        """Hybrid compression combining multiple strategies."""
        preserved = []
        
        # 1. Preserve system instructions and recent turns
        system_text, remaining = self._extract_system(text)
        recent_text, older_text = self._split_recent(remaining)
        
        if system_text:
            preserved.append(system_text)
        
        # 2. Calculate token budget
        preserved_tokens = sum(self._count_tokens(p) for p in preserved)
        preserved_tokens += self._count_tokens(recent_text)
        
        remaining_budget = max_tokens - preserved_tokens
        
        # 3. Compress older content
        if remaining_budget > 0 and older_text:
            # First deduplicate
            older_text = self._remove_redundancy(older_text)
            
            # Then importance-based compression
            if self._count_tokens(older_text) > remaining_budget:
                result = self._compress_importance(older_text, remaining_budget, query)
                older_compressed = result.text
            else:
                older_compressed = older_text
        else:
            older_compressed = ""
        
        # 4. Combine
        parts = []
        if system_text:
            parts.append(system_text)
        if older_compressed:
            parts.append("[Earlier conversation summarized]\n" + older_compressed)
        if recent_text:
            parts.append(recent_text)
        
        compressed_text = "\n\n".join(parts)
        
        return CompressionResult(
            text=compressed_text,
            original_tokens=0,
            compressed_tokens=self._count_tokens(compressed_text),
            compression_ratio=0,
            summary_created=bool(older_compressed and older_text != older_compressed),
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_sections(self, text: str) -> List[str]:
        """Split text into sections."""
        # Split on double newlines or explicit markers
        sections = re.split(r'\n\n+|(?=\n[#*-]|\nuser:|\nassistant:)', text)
        return [s.strip() for s in sections if s.strip()]
    
    def _score_sentence(self, sentence: str, position: int, total: int) -> float:
        """Score sentence importance."""
        score = 0.0
        sentence_lower = sentence.lower()
        
        # Recency score (later = more important)
        recency = (position / max(1, total - 1)) if total > 1 else 1.0
        score += recency * self._config.recency_weight
        
        # Keyword importance
        keyword_hits = sum(1 for kw in self._important_keywords if kw in sentence_lower)
        score += min(1.0, keyword_hits * 0.2) * self._config.information_weight
        
        # Length bonus (substantial content)
        words = len(sentence.split())
        if 10 <= words <= 50:
            score += 0.1
        
        # Questions are often important
        if '?' in sentence:
            score += 0.15
        
        return score
    
    def _score_section(
        self,
        section: str,
        position: int,
        total: int,
        query: str = ""
    ) -> float:
        """Score section importance."""
        score = 0.0
        section_lower = section.lower()
        
        # Recency
        recency = (position / max(1, total - 1)) if total > 1 else 1.0
        score += recency * self._config.recency_weight
        
        # Query relevance
        if query:
            query_words = set(query.lower().split()) - self._stop_words
            section_words = set(section_lower.split()) - self._stop_words
            overlap = len(query_words & section_words)
            relevance = overlap / max(1, len(query_words))
            score += relevance * self._config.relevance_weight
        
        # Information density
        keywords = self._extract_keywords(section)
        density = len(keywords) / max(1, len(section.split()))
        score += min(1.0, density * 10) * self._config.information_weight
        
        # Preserve markers
        if any(kw in section_lower for kw in self._config.preserve_keywords):
            score += 0.5
        
        return score
    
    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if w not in self._stop_words and len(w) > 2}
    
    def _remove_redundancy(self, text: str) -> str:
        """Remove redundant content."""
        sentences = self._split_sentences(text)
        
        seen = set()
        unique = []
        
        for sentence in sentences:
            # Normalize for comparison
            normalized = ' '.join(sentence.lower().split())
            
            if normalized not in seen:
                seen.add(normalized)
                unique.append(sentence)
        
        return ' '.join(unique)
    
    def _summarize_verbose(self, text: str, target_tokens: int) -> str:
        """Summarize verbose sections."""
        sections = self._split_sections(text)
        
        summarized = []
        for section in sections:
            section_tokens = self._count_tokens(section)
            
            # If section is too long, summarize it
            if section_tokens > target_tokens // 4:
                # Keep first and last parts
                sentences = self._split_sentences(section)
                if len(sentences) > 4:
                    summary = ' '.join(sentences[:2]) + ' [...] ' + ' '.join(sentences[-2:])
                    summarized.append(summary)
                else:
                    summarized.append(section)
            else:
                summarized.append(section)
        
        return '\n\n'.join(summarized)
    
    def _extract_system(self, text: str) -> Tuple[str, str]:
        """Extract system instructions from text."""
        # Look for system message patterns
        patterns = [
            r'^(system:\s*.+?)(?=\nuser:|$)',
            r'^(You are .+?)(?=\n\n)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                system = match.group(1)
                remaining = text[match.end():].strip()
                return system, remaining
        
        return "", text
    
    def _split_recent(self, text: str) -> Tuple[str, str]:
        """Split into recent and older content."""
        # Split on message boundaries
        parts = re.split(r'\n(?=(?:user|assistant):)', text, flags=re.IGNORECASE)
        
        if len(parts) <= self._config.preserve_recent_turns * 2:
            return text, ""
        
        recent_count = self._config.preserve_recent_turns * 2
        recent = '\n'.join(parts[-recent_count:])
        older = '\n'.join(parts[:-recent_count])
        
        return recent, older


# Convenience functions

def compress_context(
    context: str | List[Dict[str, str]],
    max_tokens: int = 2048,
    strategy: str = "hybrid",
    query: str = ""
) -> str:
    """
    Quick function to compress context.
    
    Args:
        context: Text or messages to compress
        max_tokens: Token limit
        strategy: Compression strategy
        query: Optional relevance query
        
    Returns:
        Compressed text
    """
    config = CompressionConfig(
        strategy=CompressionStrategy[strategy.upper()],
        max_tokens=max_tokens
    )
    
    compressor = ContextCompressor(config)
    result = compressor.compress(context, query=query)
    
    return result.text
