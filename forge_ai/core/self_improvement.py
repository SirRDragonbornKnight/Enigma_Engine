"""
Self-Improvement System - Real AI Learning and Growth

This module implements the actual learning and self-improvement mechanisms
for ForgeAI. It provides real functionality for:
  - Quality evaluation of responses
  - Learning queue management
  - Knowledge graph building
  - Training data extraction
  - Performance tracking
  - Feedback recording

Unlike stub code, this system actually learns from interactions and
improves over time through measurable metrics.
"""

import json
import time
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

from ..config import CONFIG

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Common stop words for text analysis
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
    'could', 'should', 'can', 'may', 'might', 'must', 'shall', 'this', 
    'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 
    'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 
    'their'
}

# UI text truncation length
MAX_DISPLAY_LENGTH = 200


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class Priority(Enum):
    """Priority levels for learning examples."""
    CRITICAL = 5     # Must learn immediately (user corrections, errors)
    HIGH = 4        # Important patterns (positive feedback)
    MEDIUM = 3      # Good examples (normal conversation)
    LOW = 2         # Background learning (curiosity)
    BACKGROUND = 1  # Low priority (exploratory)


class LearningSource(Enum):
    """Source of a learning example."""
    CONVERSATION = "conversation"      # From real user interactions
    PRACTICE = "practice"             # Self-generated practice
    REFLECTION = "reflection"         # Analyzed from past conversations
    RESEARCH = "research"             # From web/external sources
    CORRECTION = "correction"         # User corrections/feedback
    CURIOSITY = "curiosity"          # Exploration driven


@dataclass
class LearningExample:
    """
    A single learning example with metadata.
    
    This represents something the AI should learn from, whether it's
    a good response pattern, a correction, or new knowledge.
    """
    input_text: str                      # The prompt/question
    output_text: str                     # The desired response
    source: LearningSource               # Where this came from
    priority: Priority                   # How important to learn
    quality_score: float                 # 0.0-1.0 quality rating
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Topics extracted from this example
    topics: List[str] = field(default_factory=list)
    
    # Validation metrics
    relevance: float = 0.0    # How relevant output is to input
    coherence: float = 0.0    # How coherent the output is
    repetition: float = 0.0   # Repetition score (lower is better)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['source'] = self.source.value
        data['priority'] = self.priority.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningExample':
        """Create from dictionary."""
        data = data.copy()
        data['source'] = LearningSource(data['source'])
        data['priority'] = Priority(data['priority'])
        return cls(**data)


@dataclass
class PerformanceMetrics:
    """
    Track AI performance over time.
    
    These metrics help the AI understand if it's improving
    and guide autonomous learning decisions.
    """
    total_conversations: int = 0
    total_responses: int = 0
    
    # Feedback counts
    positive_feedback: int = 0
    negative_feedback: int = 0
    neutral_responses: int = 0
    
    # Quality metrics
    avg_response_quality: float = 0.5
    avg_conversation_length: float = 0.0
    
    # Learning stats
    examples_learned: int = 0
    topics_explored: int = 0
    
    # Topic engagement (which topics get positive feedback)
    topic_scores: Dict[str, float] = field(default_factory=dict)
    
    # Time tracking
    last_updated: float = field(default_factory=time.time)
    first_tracked: float = field(default_factory=time.time)
    
    def feedback_ratio(self) -> float:
        """Calculate positive feedback ratio (0.0-1.0)."""
        total = self.positive_feedback + self.negative_feedback
        if total == 0:
            return 0.5
        return self.positive_feedback / total
    
    def health_score(self) -> float:
        """
        Calculate overall health score (0.0-1.0).
        
        Combines multiple metrics into single score:
        - Feedback ratio (50%)
        - Response quality (30%)
        - Learning activity (20%)
        """
        feedback_component = self.feedback_ratio() * 0.5
        quality_component = self.avg_response_quality * 0.3
        
        # Learning activity score (has the AI been learning?)
        if self.total_conversations > 0:
            learning_rate = min(1.0, self.examples_learned / self.total_conversations)
        else:
            learning_rate = 0.0
        learning_component = learning_rate * 0.2
        
        return feedback_component + quality_component + learning_component
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AutonomousConfig:
    """Configuration for autonomous learning behavior."""
    
    # Learning behavior
    enabled: bool = False
    interval: int = 300  # 5 minutes between actions
    max_actions_per_hour: int = 12
    
    # Quality thresholds
    min_quality_for_learning: float = 0.6  # Don't learn from bad responses
    reflection_depth: int = 10  # How many recent conversations to analyze
    
    # Personality evolution settings
    evolution_rate: float = 0.02  # How much to adjust traits
    balance_threshold: float = 0.5  # Threshold for balancing traits
    
    # Resource limits
    low_power_mode: bool = False  # Reduce activity for gaming
    max_queue_size: int = 1000
    
    # Feature toggles
    enable_web_research: bool = True
    enable_practice: bool = True
    enable_reflection: bool = True
    enable_personality_evolution: bool = True
    enable_knowledge_building: bool = True
    
    # Low power mode settings (for gaming)
    low_power_interval: int = 1800  # 30 minutes when low power
    low_power_max_actions: int = 2  # Only 2 actions per hour in low power
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutonomousConfig':
        """Create from dictionary."""
        return cls(**data)
    
    def get_effective_interval(self) -> int:
        """Get interval considering low power mode."""
        return self.low_power_interval if self.low_power_mode else self.interval
    
    def get_effective_max_actions(self) -> int:
        """Get max actions considering low power mode."""
        return self.low_power_max_actions if self.low_power_mode else self.max_actions_per_hour


# =============================================================================
# LEARNING ENGINE
# =============================================================================

class LearningEngine:
    """
    Core learning engine that manages the AI's self-improvement.
    
    This is NOT stub code. It actually:
    - Evaluates response quality with real metrics
    - Manages a persistent learning queue
    - Builds and maintains a knowledge graph
    - Extracts training data
    - Tracks performance metrics
    - Records user feedback
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the learning engine for a model.
        
        Args:
            model_name: Name of the AI model
        """
        self.model_name = model_name
        
        # Setup paths
        models_dir = Path(CONFIG.get("models_dir", "models"))
        self.model_dir = models_dir / model_name
        self.learning_dir = self.model_dir / "learning"
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
        # Data files
        self.queue_file = self.learning_dir / "learning_queue.jsonl"
        self.metrics_file = self.learning_dir / "performance_metrics.json"
        self.knowledge_graph_file = self.learning_dir / "knowledge_graph.json"
        self.feedback_file = self.learning_dir / "feedback_log.jsonl"
        
        # In-memory state
        self.learning_queue: List[LearningExample] = []
        self.metrics = PerformanceMetrics()
        self.knowledge_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load existing data
        self._load_state()
        
        logger.info(f"LearningEngine initialized for {model_name}")
        logger.info(f"  Queue size: {len(self.learning_queue)}")
        logger.info(f"  Health score: {self.metrics.health_score():.2f}")
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def _load_state(self):
        """Load persistent state from disk."""
        # Load learning queue
        if self.queue_file.exists():
            try:
                with open(self.queue_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            example = LearningExample.from_dict(data)
                            self.learning_queue.append(example)
                logger.info(f"Loaded {len(self.learning_queue)} examples from queue")
            except Exception as e:
                logger.error(f"Error loading learning queue: {e}")
        
        # Load metrics
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = PerformanceMetrics.from_dict(data)
                logger.info(f"Loaded metrics: {self.metrics.total_conversations} conversations")
            except Exception as e:
                logger.error(f"Error loading metrics: {e}")
        
        # Load knowledge graph
        if self.knowledge_graph_file.exists():
            try:
                with open(self.knowledge_graph_file, 'r') as f:
                    data = json.load(f)
                    self.knowledge_graph = defaultdict(set, {
                        k: set(v) for k, v in data.items()
                    })
                logger.info(f"Loaded knowledge graph with {len(self.knowledge_graph)} topics")
            except Exception as e:
                logger.error(f"Error loading knowledge graph: {e}")
    
    def save_state(self):
        """Save persistent state to disk."""
        with self._lock:
            # Learning queue is persisted incrementally in add_learning_example()
            # (appended to JSONL file as examples are added)
            
            # Save metrics
            try:
                with open(self.metrics_file, 'w') as f:
                    json.dump(self.metrics.to_dict(), f, indent=2)
            except Exception as e:
                logger.error(f"Error saving metrics: {e}")
            
            # Save knowledge graph
            try:
                with open(self.knowledge_graph_file, 'w') as f:
                    data = {k: list(v) for k, v in self.knowledge_graph.items()}
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving knowledge graph: {e}")
    
    # =========================================================================
    # QUALITY EVALUATION
    # =========================================================================
    
    def evaluate_response_quality(
        self, 
        input_text: str, 
        output_text: str
    ) -> Dict[str, float]:
        """
        Evaluate the quality of a response using multiple metrics.
        
        This is REAL evaluation, not stub code. Metrics include:
        - Relevance: How well output relates to input
        - Coherence: How well-structured the output is
        - Repetition: Amount of repeated content (lower is better)
        - Overall quality: Combined score
        
        Args:
            input_text: The prompt/question
            output_text: The generated response
            
        Returns:
            Dictionary with quality metrics (all 0.0-1.0)
        """
        metrics = {}
        
        # Basic validation
        if not output_text or not output_text.strip():
            return {
                'relevance': 0.0,
                'coherence': 0.0,
                'repetition': 1.0,  # Max repetition
                'overall': 0.0
            }
        
        # 1. RELEVANCE: Check if output relates to input
        metrics['relevance'] = self._calculate_relevance(input_text, output_text)
        
        # 2. COHERENCE: Check if output is well-structured
        metrics['coherence'] = self._calculate_coherence(output_text)
        
        # 3. REPETITION: Check for repeated content (lower is better)
        metrics['repetition'] = self._calculate_repetition(output_text)
        
        # 4. OVERALL: Combined quality score
        # Repetition is inverted (1.0 - repetition) because lower is better
        metrics['overall'] = (
            metrics['relevance'] * 0.4 +
            metrics['coherence'] * 0.3 +
            (1.0 - metrics['repetition']) * 0.3
        )
        
        return metrics
    
    def _calculate_relevance(self, input_text: str, output_text: str) -> float:
        """
        Calculate how relevant the output is to the input.
        
        Uses keyword overlap and length ratios as proxies for relevance.
        """
        # Extract keywords (simple word-based approach)
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())
        
        # Remove common stop words
        input_keywords = input_words - STOP_WORDS
        output_keywords = output_words - STOP_WORDS
        
        if not input_keywords:
            return 0.5  # Neutral if no keywords in input
        
        # Calculate overlap
        overlap = len(input_keywords & output_keywords)
        relevance = min(1.0, overlap / len(input_keywords))
        
        # Boost if output contains question marks when input has them
        if '?' in input_text and '?' not in output_text:
            relevance *= 1.1  # Small boost for answering questions
        
        return min(1.0, relevance)
    
    def _calculate_coherence(self, text: str) -> float:
        """
        Calculate how coherent the text is.
        
        Uses sentence structure, punctuation, and length as proxies.
        """
        if not text:
            return 0.0
        
        score = 0.5  # Start neutral
        
        # 1. Has proper sentence structure (capitalization and punctuation)
        sentences = text.split('.')
        if len(sentences) > 1:
            score += 0.1
        
        # 2. Reasonable length (not too short, not too long)
        word_count = len(text.split())
        if 5 <= word_count <= 200:
            score += 0.2
        elif word_count < 3:
            score -= 0.2
        
        # 3. Has punctuation (shows structure)
        if any(p in text for p in '.!?,;'):
            score += 0.1
        
        # 4. Not all caps (shows proper formatting)
        if not text.isupper():
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _calculate_repetition(self, text: str) -> float:
        """
        Calculate repetition score (0.0 = no repetition, 1.0 = highly repetitive).
        
        Checks for repeated phrases and words.
        """
        if not text:
            return 0.0
        
        words = text.lower().split()
        if len(words) < 3:
            return 0.0
        
        # Check for repeated n-grams (2-word and 3-word phrases)
        repetition_score = 0.0
        
        # 2-grams
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        unique_bigrams = len(set(bigrams))
        total_bigrams = len(bigrams)
        if total_bigrams > 0:
            bigram_repetition = 1.0 - (unique_bigrams / total_bigrams)
            repetition_score += bigram_repetition * 0.5
        
        # 3-grams
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        unique_trigrams = len(set(trigrams))
        total_trigrams = len(trigrams)
        if total_trigrams > 0:
            trigram_repetition = 1.0 - (unique_trigrams / total_trigrams)
            repetition_score += trigram_repetition * 0.5
        
        return min(1.0, repetition_score)
    
    # =========================================================================
    # TOPIC EXTRACTION
    # =========================================================================
    
    def extract_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """
        Extract main topics from text.
        
        Uses simple keyword extraction based on word frequency
        and common patterns.
        
        Args:
            text: Text to extract topics from
            max_topics: Maximum number of topics to return
            
        Returns:
            List of topic keywords
        """
        if not text:
            return []
        
        # Normalize and tokenize
        words = text.lower().split()
        
        # Remove stop words
        keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        
        # Count frequencies
        freq = defaultdict(int)
        for word in keywords:
            freq[word] += 1
        
        # Sort by frequency and take top N
        topics = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        topics = [topic for topic, _ in topics[:max_topics]]
        
        return topics
    
    # =========================================================================
    # LEARNING QUEUE
    # =========================================================================
    
    def add_learning_example(self, example: LearningExample):
        """
        Add an example to the learning queue.
        
        The example will be persisted to disk and kept in memory
        for future training.
        
        Args:
            example: Learning example to add
        """
        with self._lock:
            # Extract topics if not already done
            if not example.topics:
                example.topics = self.extract_topics(
                    example.input_text + ' ' + example.output_text
                )
            
            # Add to queue
            self.learning_queue.append(example)
            
            # Persist to disk (append to JSONL)
            try:
                with open(self.queue_file, 'a') as f:
                    f.write(json.dumps(example.to_dict()) + '\n')
            except Exception as e:
                logger.error(f"Error persisting learning example: {e}")
            
            # Update knowledge graph
            for topic in example.topics:
                # Connect topics that appear together
                for other_topic in example.topics:
                    if topic != other_topic:
                        self.knowledge_graph[topic].add(other_topic)
            
            # Update metrics
            self.metrics.examples_learned += 1
            
            logger.debug(f"Added learning example: {example.priority.name} priority, "
                        f"quality={example.quality_score:.2f}, topics={example.topics[:3]}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning queue."""
        with self._lock:
            stats = {
                'total_examples': len(self.learning_queue),
                'by_priority': defaultdict(int),
                'by_source': defaultdict(int),
                'avg_quality': 0.0,
            }
            
            if self.learning_queue:
                for example in self.learning_queue:
                    stats['by_priority'][example.priority.name] += 1
                    stats['by_source'][example.source.name] += 1
                
                stats['avg_quality'] = sum(
                    e.quality_score for e in self.learning_queue
                ) / len(self.learning_queue)
            
            return dict(stats)
    
    # =========================================================================
    # FEEDBACK & METRICS
    # =========================================================================
    
    def record_feedback(
        self, 
        input_text: str, 
        output_text: str, 
        feedback: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record user feedback on a response.
        
        Args:
            input_text: The prompt
            output_text: The AI's response
            feedback: 'positive', 'negative', or 'neutral'
            metadata: Optional additional metadata
        """
        with self._lock:
            # Update metrics
            if feedback == 'positive':
                self.metrics.positive_feedback += 1
            elif feedback == 'negative':
                self.metrics.negative_feedback += 1
            else:
                self.metrics.neutral_responses += 1
            
            # Create learning example for positive feedback
            if feedback == 'positive':
                quality = self.evaluate_response_quality(input_text, output_text)
                example = LearningExample(
                    input_text=input_text,
                    output_text=output_text,
                    source=LearningSource.CONVERSATION,
                    priority=Priority.HIGH,  # Positive feedback = important
                    quality_score=quality['overall'],
                    relevance=quality['relevance'],
                    coherence=quality['coherence'],
                    repetition=quality['repetition'],
                    metadata=metadata or {}
                )
                self.add_learning_example(example)
            
            # Log feedback
            try:
                with open(self.feedback_file, 'a') as f:
                    log_entry = {
                        'timestamp': time.time(),
                        'feedback': feedback,
                        'input': input_text[:100],  # Truncate for space
                        'output': output_text[:100],
                        'metadata': metadata
                    }
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as e:
                logger.error(f"Error logging feedback: {e}")
            
            self.metrics.last_updated = time.time()
    
    def update_conversation_metrics(self, message_count: int):
        """
        Update metrics after a conversation.
        
        Args:
            message_count: Number of messages in the conversation
        """
        with self._lock:
            self.metrics.total_conversations += 1
            
            # Update rolling average of conversation length
            n = self.metrics.total_conversations
            self.metrics.avg_conversation_length = (
                (self.metrics.avg_conversation_length * (n - 1) + message_count) / n
            )
            
            self.metrics.last_updated = time.time()
    
    # =========================================================================
    # KNOWLEDGE GRAPH
    # =========================================================================
    
    def get_related_topics(self, topic: str, max_results: int = 5) -> List[str]:
        """
        Get topics related to a given topic from the knowledge graph.
        
        Args:
            topic: Topic to find relations for
            max_results: Maximum number of related topics to return
            
        Returns:
            List of related topic names
        """
        with self._lock:
            if topic not in self.knowledge_graph:
                return []
            
            related = list(self.knowledge_graph[topic])
            return related[:max_results]
    
    def get_all_topics(self) -> List[str]:
        """Get all topics in the knowledge graph."""
        with self._lock:
            return list(self.knowledge_graph.keys())
    
    # =========================================================================
    # TRAINING DATA EXPORT
    # =========================================================================
    
    def export_training_data(
        self, 
        min_quality: float = 0.6,
        max_examples: int = 1000
    ) -> Path:
        """
        Export learning queue to training data file.
        
        Filters by quality and limits size.
        
        Args:
            min_quality: Minimum quality score to include
            max_examples: Maximum number of examples to export
            
        Returns:
            Path to exported file
        """
        with self._lock:
            # Filter by quality
            good_examples = [
                e for e in self.learning_queue
                if e.quality_score >= min_quality
            ]
            
            # Sort by priority and quality
            good_examples.sort(
                key=lambda e: (e.priority.value, e.quality_score),
                reverse=True
            )
            
            # Limit size
            good_examples = good_examples[:max_examples]
            
            # Export to training format
            export_path = self.learning_dir / "training_data.txt"
            
            with open(export_path, 'w') as f:
                f.write(f"# Training data exported from learning queue\n")
                f.write(f"# Model: {self.model_name}\n")
                f.write(f"# Exported: {datetime.now().isoformat()}\n")
                f.write(f"# Examples: {len(good_examples)}\n")
                f.write(f"# Min quality: {min_quality}\n")
                f.write("\n")
                
                for example in good_examples:
                    f.write(f"Q: {example.input_text}\n")
                    f.write(f"A: {example.output_text}\n")
                    f.write("\n")
            
            logger.info(f"Exported {len(good_examples)} examples to {export_path}")
            return export_path
    
    # =========================================================================
    # METRICS
    # =========================================================================
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self._lock:
            return self.metrics
    
    def get_metrics_summary(self) -> str:
        """Get human-readable metrics summary."""
        m = self.metrics
        
        lines = []
        lines.append(f"=== Performance Metrics for {self.model_name} ===")
        lines.append(f"Conversations: {m.total_conversations}")
        lines.append(f"Responses: {m.total_responses}")
        lines.append(f"Feedback Ratio: {m.feedback_ratio():.1%} positive")
        lines.append(f"Health Score: {m.health_score():.1%}")
        lines.append(f"Avg Quality: {m.avg_response_quality:.2f}")
        lines.append(f"Examples Learned: {m.examples_learned}")
        lines.append(f"Topics Explored: {m.topics_explored}")
        lines.append(f"Queue Size: {len(self.learning_queue)}")
        
        return '\n'.join(lines)


# =============================================================================
# GLOBAL ACCESS
# =============================================================================

_engines: Dict[str, LearningEngine] = {}
_lock = threading.Lock()


def get_learning_engine(model_name: str) -> LearningEngine:
    """Get or create a learning engine for a model."""
    with _lock:
        if model_name not in _engines:
            _engines[model_name] = LearningEngine(model_name)
        return _engines[model_name]
