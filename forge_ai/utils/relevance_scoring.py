"""
Context Relevance Scoring - Semantic relevance scoring for context.

Provides relevance scoring for:
- Semantic similarity between queries and context
- TF-IDF based relevance
- BM25 ranking
- Combined scoring strategies
- Context window optimization

Part of the ForgeAI memory utilities.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Tuple
from enum import Enum


class ScoringStrategy(Enum):
    """Relevance scoring strategies."""
    TFIDF = "tfidf"
    BM25 = "bm25"
    JACCARD = "jaccard"
    COSINE = "cosine"
    COMBINED = "combined"


@dataclass
class ScoredContext:
    """Context with relevance score."""
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    strategy: Optional[str] = None
    breakdown: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "strategy": self.strategy,
            "breakdown": self.breakdown
        }


class TextProcessor:
    """Text preprocessing utilities."""
    
    # Common English stopwords
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
        'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 'can', 'just', 'should', 'now', 'i', 'you', 'your', 'we', 'our'
    }
    
    @staticmethod
    def tokenize(text: str, lowercase: bool = True, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_stopwords: Remove common stopwords
            
        Returns:
            List of tokens
        """
        if lowercase:
            text = text.lower()
        
        # Split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text)
        
        if remove_stopwords:
            tokens = [t for t in tokens if t not in TextProcessor.STOPWORDS]
        
        return tokens
    
    @staticmethod
    def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Generate n-grams from tokens."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


class TFIDFScorer:
    """TF-IDF based relevance scoring."""
    
    def __init__(self):
        """Initialize TF-IDF scorer."""
        self._documents: List[List[str]] = []
        self._idf_cache: Dict[str, float] = {}
        self._doc_count = 0
    
    def add_document(self, text: str):
        """Add a document to the corpus."""
        tokens = TextProcessor.tokenize(text)
        self._documents.append(tokens)
        self._doc_count += 1
        self._idf_cache.clear()  # Invalidate cache
    
    def add_documents(self, texts: List[str]):
        """Add multiple documents."""
        for text in texts:
            self.add_document(text)
    
    def _tf(self, term: str, document: List[str]) -> float:
        """Calculate term frequency."""
        if not document:
            return 0.0
        return document.count(term) / len(document)
    
    def _idf(self, term: str) -> float:
        """Calculate inverse document frequency."""
        if term in self._idf_cache:
            return self._idf_cache[term]
        
        doc_freq = sum(1 for doc in self._documents if term in doc)
        if doc_freq == 0:
            idf = 0.0
        else:
            idf = math.log(self._doc_count / doc_freq)
        
        self._idf_cache[term] = idf
        return idf
    
    def score(self, query: str, document: str) -> float:
        """
        Calculate TF-IDF relevance score.
        
        Args:
            query: Query text
            document: Document text
            
        Returns:
            Relevance score (0-1)
        """
        query_tokens = TextProcessor.tokenize(query)
        doc_tokens = TextProcessor.tokenize(document)
        
        if not query_tokens or not doc_tokens:
            return 0.0
        
        score = 0.0
        for term in query_tokens:
            tf = self._tf(term, doc_tokens)
            idf = self._idf(term) if self._documents else 1.0
            score += tf * idf
        
        # Normalize
        max_score = len(query_tokens) * math.log(max(self._doc_count, 2))
        if max_score > 0:
            score = min(score / max_score, 1.0)
        
        return score


class BM25Scorer:
    """BM25 (Okapi BM25) ranking algorithm."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 scorer.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self._documents: List[List[str]] = []
        self._doc_lengths: List[int] = []
        self._avg_doc_length: float = 0.0
        self._doc_freqs: Dict[str, int] = {}
        self._doc_count = 0
    
    def add_document(self, text: str):
        """Add a document to the corpus."""
        tokens = TextProcessor.tokenize(text)
        self._documents.append(tokens)
        self._doc_lengths.append(len(tokens))
        self._doc_count += 1
        
        # Update document frequencies
        seen = set()
        for token in tokens:
            if token not in seen:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                seen.add(token)
        
        # Update average length
        self._avg_doc_length = sum(self._doc_lengths) / self._doc_count
    
    def add_documents(self, texts: List[str]):
        """Add multiple documents."""
        for text in texts:
            self.add_document(text)
    
    def _idf(self, term: str) -> float:
        """Calculate IDF for BM25."""
        df = self._doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query: str, document: str) -> float:
        """
        Calculate BM25 relevance score.
        
        Args:
            query: Query text
            document: Document text
            
        Returns:
            Relevance score
        """
        query_tokens = TextProcessor.tokenize(query)
        doc_tokens = TextProcessor.tokenize(document)
        doc_length = len(doc_tokens)
        
        if not query_tokens or not doc_tokens:
            return 0.0
        
        term_freqs = Counter(doc_tokens)
        avg_dl = self._avg_doc_length if self._avg_doc_length > 0 else doc_length
        
        score = 0.0
        for term in query_tokens:
            tf = term_freqs.get(term, 0)
            idf = self._idf(term) if self._documents else math.log(2)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / avg_dl)
            
            if denominator > 0:
                score += idf * numerator / denominator
        
        return score


class JaccardScorer:
    """Jaccard similarity scoring."""
    
    @staticmethod
    def score(query: str, document: str) -> float:
        """
        Calculate Jaccard similarity.
        
        Args:
            query: Query text
            document: Document text
            
        Returns:
            Similarity score (0-1)
        """
        query_set = set(TextProcessor.tokenize(query))
        doc_set = set(TextProcessor.tokenize(document))
        
        if not query_set or not doc_set:
            return 0.0
        
        intersection = len(query_set & doc_set)
        union = len(query_set | doc_set)
        
        return intersection / union if union > 0 else 0.0


class CosineScorer:
    """Cosine similarity scoring."""
    
    @staticmethod
    def score(query: str, document: str) -> float:
        """
        Calculate cosine similarity using term frequencies.
        
        Args:
            query: Query text
            document: Document text
            
        Returns:
            Similarity score (0-1)
        """
        query_tokens = TextProcessor.tokenize(query)
        doc_tokens = TextProcessor.tokenize(document)
        
        if not query_tokens or not doc_tokens:
            return 0.0
        
        query_freq = Counter(query_tokens)
        doc_freq = Counter(doc_tokens)
        
        # Get all unique terms
        all_terms = set(query_freq.keys()) | set(doc_freq.keys())
        
        # Calculate dot product and magnitudes
        dot_product = 0.0
        query_magnitude = 0.0
        doc_magnitude = 0.0
        
        for term in all_terms:
            q = query_freq.get(term, 0)
            d = doc_freq.get(term, 0)
            dot_product += q * d
            query_magnitude += q * q
            doc_magnitude += d * d
        
        magnitude = math.sqrt(query_magnitude) * math.sqrt(doc_magnitude)
        
        return dot_product / magnitude if magnitude > 0 else 0.0


class RelevanceScorer:
    """
    Combined relevance scoring system.
    
    Usage:
        scorer = RelevanceScorer()
        
        # Add context documents for corpus stats
        scorer.add_documents(["doc1 content", "doc2 content"])
        
        # Score single document
        score = scorer.score("user query", "document to score")
        
        # Score and rank multiple contexts
        contexts = ["context1", "context2", "context3"]
        ranked = scorer.rank_contexts("query", contexts, top_k=2)
        
        # Get scored context objects
        results = scorer.score_contexts("query", contexts)
        for ctx in results:
            print(f"{ctx.score:.3f}: {ctx.content[:50]}...")
    """
    
    def __init__(
        self,
        strategy: ScoringStrategy = ScoringStrategy.COMBINED,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize relevance scorer.
        
        Args:
            strategy: Scoring strategy to use
            weights: Weights for combined strategy
        """
        self.strategy = strategy
        self.weights = weights or {
            "tfidf": 0.3,
            "bm25": 0.4,
            "jaccard": 0.15,
            "cosine": 0.15
        }
        
        self._tfidf = TFIDFScorer()
        self._bm25 = BM25Scorer()
    
    def add_document(self, text: str):
        """Add document to corpus for IDF calculations."""
        self._tfidf.add_document(text)
        self._bm25.add_document(text)
    
    def add_documents(self, texts: List[str]):
        """Add multiple documents to corpus."""
        for text in texts:
            self.add_document(text)
    
    def score(
        self,
        query: str,
        document: str,
        strategy: Optional[ScoringStrategy] = None
    ) -> float:
        """
        Calculate relevance score.
        
        Args:
            query: Query text
            document: Document text
            strategy: Override default strategy
            
        Returns:
            Relevance score
        """
        strat = strategy or self.strategy
        
        if strat == ScoringStrategy.TFIDF:
            return self._tfidf.score(query, document)
        elif strat == ScoringStrategy.BM25:
            return self._bm25.score(query, document)
        elif strat == ScoringStrategy.JACCARD:
            return JaccardScorer.score(query, document)
        elif strat == ScoringStrategy.COSINE:
            return CosineScorer.score(query, document)
        elif strat == ScoringStrategy.COMBINED:
            return self._combined_score(query, document)
        
        return 0.0
    
    def _combined_score(self, query: str, document: str) -> float:
        """Calculate combined weighted score."""
        scores = {
            "tfidf": self._tfidf.score(query, document),
            "bm25": self._bm25.score(query, document),
            "jaccard": JaccardScorer.score(query, document),
            "cosine": CosineScorer.score(query, document)
        }
        
        # Normalize BM25 (can exceed 1.0)
        if scores["bm25"] > 1.0:
            scores["bm25"] = 1.0 / (1.0 + math.exp(-scores["bm25"]))
        
        total_weight = sum(self.weights.values())
        weighted_score = sum(
            scores[s] * self.weights.get(s, 0)
            for s in scores
        )
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def score_contexts(
        self,
        query: str,
        contexts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[ScoredContext]:
        """
        Score multiple contexts.
        
        Args:
            query: Query text
            contexts: List of context strings
            metadata: Optional metadata for each context
            
        Returns:
            List of scored contexts
        """
        results = []
        
        for i, context in enumerate(contexts):
            score = self.score(query, context)
            
            # Get score breakdown for combined
            breakdown = {}
            if self.strategy == ScoringStrategy.COMBINED:
                breakdown = {
                    "tfidf": self._tfidf.score(query, context),
                    "bm25": self._bm25.score(query, context),
                    "jaccard": JaccardScorer.score(query, context),
                    "cosine": CosineScorer.score(query, context)
                }
            
            results.append(ScoredContext(
                content=context,
                score=score,
                metadata=metadata[i] if metadata else {},
                strategy=self.strategy.value,
                breakdown=breakdown
            ))
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def rank_contexts(
        self,
        query: str,
        contexts: List[str],
        top_k: Optional[int] = None,
        threshold: float = 0.0
    ) -> List[str]:
        """
        Rank and filter contexts by relevance.
        
        Args:
            query: Query text
            contexts: List of context strings
            top_k: Return only top k results
            threshold: Minimum score threshold
            
        Returns:
            Ranked list of contexts
        """
        scored = self.score_contexts(query, contexts)
        
        # Filter by threshold
        filtered = [s for s in scored if s.score >= threshold]
        
        # Limit to top_k
        if top_k:
            filtered = filtered[:top_k]
        
        return [s.content for s in filtered]
    
    def select_context_window(
        self,
        query: str,
        contexts: List[str],
        max_tokens: int = 2000,
        avg_chars_per_token: int = 4
    ) -> List[str]:
        """
        Select contexts that fit within token budget.
        
        Args:
            query: Query text
            contexts: Available contexts
            max_tokens: Maximum token budget
            avg_chars_per_token: Estimated chars per token
            
        Returns:
            Selected contexts fitting budget
        """
        max_chars = max_tokens * avg_chars_per_token
        scored = self.score_contexts(query, contexts)
        
        selected = []
        total_chars = 0
        
        for ctx in scored:
            ctx_chars = len(ctx.content)
            if total_chars + ctx_chars <= max_chars:
                selected.append(ctx.content)
                total_chars += ctx_chars
            else:
                break
        
        return selected


# Global scorer instance
_global_scorer: Optional[RelevanceScorer] = None


def get_relevance_scorer() -> RelevanceScorer:
    """Get the global relevance scorer."""
    global _global_scorer
    if _global_scorer is None:
        _global_scorer = RelevanceScorer()
    return _global_scorer


def score_relevance(query: str, document: str) -> float:
    """Score document relevance to query."""
    return get_relevance_scorer().score(query, document)


def rank_by_relevance(
    query: str,
    contexts: List[str],
    top_k: Optional[int] = None
) -> List[str]:
    """Rank contexts by relevance to query."""
    return get_relevance_scorer().rank_contexts(query, contexts, top_k)
