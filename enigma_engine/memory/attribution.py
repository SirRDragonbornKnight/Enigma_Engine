"""
Memory Attribution for Enigma AI Engine

Track which memories contributed to responses.

Features:
- Source tracking
- Contribution scoring
- Citation generation
- Memory visualization
- Provenance chain

Usage:
    from enigma_engine.memory.attribution import AttributionTracker, get_tracker
    
    tracker = get_tracker()
    
    # Track memory usage
    tracker.track_usage(response_id="r1", memory_ids=["m1", "m2"])
    
    # Get attributions
    attributions = tracker.get_attributions("r1")
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class AttributionType(Enum):
    """Types of memory attribution."""
    DIRECT = "direct"  # Memory directly used
    INDIRECT = "indirect"  # Memory influenced reasoning
    RETRIEVED = "retrieved"  # Memory retrieved but maybe not used
    GENERATED = "generated"  # New memory created from response


@dataclass
class MemorySource:
    """Source information for a memory."""
    memory_id: str
    source_type: str  # "conversation", "document", "web", "user", etc.
    source_id: Optional[str] = None
    source_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Attribution:
    """Attribution of a memory to a response."""
    memory_id: str
    response_id: str
    attribution_type: AttributionType
    
    # Scoring
    contribution_score: float = 0.0  # 0-1, how much it contributed
    confidence: float = 1.0  # Confidence in attribution
    
    # Context
    text_span: Optional[Tuple[int, int]] = None  # Where in response
    memory_snippet: Optional[str] = None  # Relevant part of memory
    
    # Metadata
    timestamp: float = field(default_factory=time.time)


@dataclass
class AttributionReport:
    """Report of all attributions for a response."""
    response_id: str
    response_text: str
    
    attributions: List[Attribution] = field(default_factory=list)
    total_contribution: float = 0.0
    citation_text: str = ""
    
    # Provenance
    memory_chain: List[str] = field(default_factory=list)  # Memory ancestors


class ContributionScorer:
    """Score memory contributions to responses."""
    
    def __init__(
        self,
        similarity_fn: Optional[Callable[[str, str], float]] = None
    ):
        self._similarity_fn = similarity_fn or self._default_similarity
    
    def _default_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def score_contribution(
        self,
        memory_text: str,
        response_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Score how much a memory contributed to a response.
        
        Returns:
            Contribution score (0-1)
        """
        # Base similarity
        similarity = self._similarity_fn(memory_text, response_text)
        
        # Boost if memory words appear in response
        memory_words = set(memory_text.lower().split())
        response_words = set(response_text.lower().split())
        
        word_overlap = len(memory_words & response_words)
        overlap_ratio = word_overlap / len(memory_words) if memory_words else 0
        
        # Combine scores
        score = 0.6 * similarity + 0.4 * overlap_ratio
        
        return min(1.0, score)
    
    def find_text_spans(
        self,
        memory_text: str,
        response_text: str,
        min_length: int = 3
    ) -> List[Tuple[int, int]]:
        """Find where memory content appears in response."""
        spans = []
        memory_lower = memory_text.lower()
        response_lower = response_text.lower()
        
        # Find matching phrases
        memory_words = memory_lower.split()
        
        for i in range(len(memory_words)):
            for j in range(i + min_length, len(memory_words) + 1):
                phrase = ' '.join(memory_words[i:j])
                
                start = response_lower.find(phrase)
                if start != -1:
                    spans.append((start, start + len(phrase)))
        
        # Merge overlapping spans
        spans = self._merge_spans(spans)
        
        return spans
    
    def _merge_spans(
        self,
        spans: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Merge overlapping spans."""
        if not spans:
            return []
        
        sorted_spans = sorted(spans)
        merged = [sorted_spans[0]]
        
        for start, end in sorted_spans[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(end, last_end))
            else:
                merged.append((start, end))
        
        return merged


class CitationGenerator:
    """Generate citations for memory attributions."""
    
    def __init__(self, style: str = "inline"):
        """
        Initialize citation generator.
        
        Args:
            style: Citation style ("inline", "footnote", "endnote")
        """
        self._style = style
    
    def generate_citation(
        self,
        attributions: List[Attribution],
        sources: Dict[str, MemorySource]
    ) -> str:
        """
        Generate citation text for attributions.
        
        Returns:
            Citation string
        """
        if not attributions:
            return ""
        
        if self._style == "inline":
            return self._inline_citations(attributions, sources)
        elif self._style == "footnote":
            return self._footnote_citations(attributions, sources)
        else:
            return self._endnote_citations(attributions, sources)
    
    def _inline_citations(
        self,
        attributions: List[Attribution],
        sources: Dict[str, MemorySource]
    ) -> str:
        """Generate inline citations like [1], [2]."""
        citations = []
        
        for i, attr in enumerate(attributions, 1):
            source = sources.get(attr.memory_id)
            if source:
                name = source.source_name or source.source_id or attr.memory_id
                citations.append(f"[{i}: {name}]")
            else:
                citations.append(f"[{i}]")
        
        return " ".join(citations)
    
    def _footnote_citations(
        self,
        attributions: List[Attribution],
        sources: Dict[str, MemorySource]
    ) -> str:
        """Generate footnote-style citations."""
        lines = ["\n---\nSources:"]
        
        for i, attr in enumerate(attributions, 1):
            source = sources.get(attr.memory_id)
            if source:
                name = source.source_name or source.source_type
                line = f"{i}. {name}"
                if attr.memory_snippet:
                    line += f': "{attr.memory_snippet[:50]}..."'
                lines.append(line)
            else:
                lines.append(f"{i}. Memory {attr.memory_id}")
        
        return "\n".join(lines)
    
    def _endnote_citations(
        self,
        attributions: List[Attribution],
        sources: Dict[str, MemorySource]
    ) -> str:
        """Generate endnote-style citations."""
        lines = ["\n\nReferences:"]
        
        for i, attr in enumerate(attributions, 1):
            source = sources.get(attr.memory_id)
            if source:
                timestamp = time.strftime(
                    "%Y-%m-%d",
                    time.localtime(source.timestamp)
                )
                name = source.source_name or source.source_type
                lines.append(f"[{i}] {name} ({source.source_type}), {timestamp}")
            else:
                lines.append(f"[{i}] Memory {attr.memory_id}")
        
        return "\n".join(lines)


class AttributionTracker:
    """Track memory attributions for responses."""
    
    def __init__(
        self,
        citation_style: str = "inline",
        similarity_fn: Optional[Callable[[str, str], float]] = None
    ):
        """
        Initialize attribution tracker.
        
        Args:
            citation_style: Style for citations
            similarity_fn: Custom similarity function
        """
        self._scorer = ContributionScorer(similarity_fn)
        self._citation = CitationGenerator(citation_style)
        
        # Storage
        self._sources: Dict[str, MemorySource] = {}
        self._attributions: Dict[str, List[Attribution]] = defaultdict(list)
        self._response_texts: Dict[str, str] = {}
        self._memory_texts: Dict[str, str] = {}
        
        # Provenance
        self._memory_parents: Dict[str, List[str]] = defaultdict(list)
    
    def register_source(
        self,
        memory_id: str,
        source_type: str,
        source_id: Optional[str] = None,
        source_name: Optional[str] = None,
        memory_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register a memory source.
        
        Args:
            memory_id: Memory identifier
            source_type: Type of source
            source_id: Source identifier
            source_name: Human-readable name
            memory_text: Text content for attribution scoring
            metadata: Additional metadata
        """
        self._sources[memory_id] = MemorySource(
            memory_id=memory_id,
            source_type=source_type,
            source_id=source_id,
            source_name=source_name,
            metadata=metadata or {}
        )
        
        if memory_text:
            self._memory_texts[memory_id] = memory_text
    
    def track_usage(
        self,
        response_id: str,
        memory_ids: List[str],
        response_text: Optional[str] = None,
        attribution_type: AttributionType = AttributionType.RETRIEVED
    ):
        """
        Track which memories were used for a response.
        
        Args:
            response_id: Response identifier
            memory_ids: List of memory IDs used
            response_text: Response text for scoring
            attribution_type: Type of attribution
        """
        if response_text:
            self._response_texts[response_id] = response_text
        
        for memory_id in memory_ids:
            # Calculate contribution score
            score = 0.5  # Default
            memory_snippet = None
            text_span = None
            
            memory_text = self._memory_texts.get(memory_id)
            if memory_text and response_text:
                score = self._scorer.score_contribution(
                    memory_text, response_text
                )
                
                spans = self._scorer.find_text_spans(
                    memory_text, response_text
                )
                if spans:
                    text_span = spans[0]
                
                # Extract relevant snippet
                memory_snippet = memory_text[:100]
            
            attribution = Attribution(
                memory_id=memory_id,
                response_id=response_id,
                attribution_type=attribution_type,
                contribution_score=score,
                text_span=text_span,
                memory_snippet=memory_snippet
            )
            
            self._attributions[response_id].append(attribution)
    
    def mark_direct_attribution(
        self,
        response_id: str,
        memory_id: str,
        contribution_score: Optional[float] = None
    ):
        """Mark a memory as directly contributing to response."""
        for attr in self._attributions[response_id]:
            if attr.memory_id == memory_id:
                attr.attribution_type = AttributionType.DIRECT
                if contribution_score is not None:
                    attr.contribution_score = contribution_score
                break
    
    def add_provenance(
        self,
        memory_id: str,
        parent_ids: List[str]
    ):
        """Add parent memories for provenance tracking."""
        self._memory_parents[memory_id].extend(parent_ids)
    
    def get_attributions(
        self,
        response_id: str
    ) -> List[Attribution]:
        """Get attributions for a response."""
        return self._attributions.get(response_id, [])
    
    def get_report(
        self,
        response_id: str,
        include_citations: bool = True
    ) -> AttributionReport:
        """
        Generate attribution report for a response.
        
        Returns:
            AttributionReport with all attribution details
        """
        attributions = self._attributions.get(response_id, [])
        response_text = self._response_texts.get(response_id, "")
        
        # Calculate total contribution
        total = sum(a.contribution_score for a in attributions)
        
        # Build provenance chain
        chain: Set[str] = set()
        for attr in attributions:
            self._build_chain(attr.memory_id, chain)
        
        # Generate citations
        citation_text = ""
        if include_citations and attributions:
            citation_text = self._citation.generate_citation(
                attributions, self._sources
            )
        
        return AttributionReport(
            response_id=response_id,
            response_text=response_text,
            attributions=attributions,
            total_contribution=total,
            citation_text=citation_text,
            memory_chain=list(chain)
        )
    
    def _build_chain(
        self,
        memory_id: str,
        chain: Set[str],
        depth: int = 0
    ):
        """Recursively build provenance chain."""
        if depth > 10 or memory_id in chain:
            return
        
        chain.add(memory_id)
        
        for parent_id in self._memory_parents.get(memory_id, []):
            self._build_chain(parent_id, chain, depth + 1)
    
    def get_top_sources(
        self,
        response_id: str,
        n: int = 5
    ) -> List[Tuple[str, MemorySource, float]]:
        """
        Get top contributing sources.
        
        Returns:
            List of (memory_id, source, score) tuples
        """
        attributions = self._attributions.get(response_id, [])
        
        sorted_attrs = sorted(
            attributions,
            key=lambda a: a.contribution_score,
            reverse=True
        )[:n]
        
        result = []
        for attr in sorted_attrs:
            source = self._sources.get(attr.memory_id)
            if source:
                result.append((attr.memory_id, source, attr.contribution_score))
        
        return result
    
    def get_memory_usage_stats(
        self,
        memory_id: str
    ) -> Dict[str, Any]:
        """Get statistics for how a memory has been used."""
        usage_count = 0
        avg_contribution = 0.0
        response_ids = []
        
        for resp_id, attrs in self._attributions.items():
            for attr in attrs:
                if attr.memory_id == memory_id:
                    usage_count += 1
                    avg_contribution += attr.contribution_score
                    response_ids.append(resp_id)
        
        if usage_count > 0:
            avg_contribution /= usage_count
        
        return {
            "memory_id": memory_id,
            "usage_count": usage_count,
            "avg_contribution": avg_contribution,
            "response_ids": response_ids,
            "source": self._sources.get(memory_id)
        }
    
    def clear_response(self, response_id: str):
        """Clear attributions for a response."""
        if response_id in self._attributions:
            del self._attributions[response_id]
        if response_id in self._response_texts:
            del self._response_texts[response_id]


# Global instance
_tracker: Optional[AttributionTracker] = None


def get_tracker(
    citation_style: str = "inline"
) -> AttributionTracker:
    """Get or create global attribution tracker."""
    global _tracker
    if _tracker is None:
        _tracker = AttributionTracker(citation_style)
    return _tracker
