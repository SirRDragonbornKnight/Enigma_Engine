"""
Tests for the self-improvement and autonomous learning system.

Tests cover:
- LearningEngine functionality
- Quality evaluation metrics
- Learning queue management
- Performance metrics tracking
- Autonomous actions
- Feedback recording
- Knowledge graph building
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import time

from forge_ai.core.self_improvement import (
    LearningEngine,
    LearningExample,
    LearningSource,
    Priority,
    PerformanceMetrics,
    AutonomousConfig,
    get_learning_engine
)


class TestLearningExample:
    """Test LearningExample dataclass."""
    
    def test_create_example(self):
        """Test creating a learning example."""
        example = LearningExample(
            input_text="What is AI?",
            output_text="AI is artificial intelligence.",
            source=LearningSource.CONVERSATION,
            priority=Priority.MEDIUM,
            quality_score=0.8
        )
        
        assert example.input_text == "What is AI?"
        assert example.output_text == "AI is artificial intelligence."
        assert example.source == LearningSource.CONVERSATION
        assert example.priority == Priority.MEDIUM
        assert example.quality_score == 0.8
    
    def test_to_dict(self):
        """Test converting example to dictionary."""
        example = LearningExample(
            input_text="Test",
            output_text="Response",
            source=LearningSource.PRACTICE,
            priority=Priority.HIGH,
            quality_score=0.9
        )
        
        data = example.to_dict()
        assert isinstance(data, dict)
        assert data['input_text'] == "Test"
        assert data['source'] == 'practice'
        assert data['priority'] == 4
    
    def test_from_dict(self):
        """Test creating example from dictionary."""
        data = {
            'input_text': "Test",
            'output_text': "Response",
            'source': 'conversation',
            'priority': 3,
            'quality_score': 0.7,
            'timestamp': time.time(),
            'metadata': {},
            'topics': [],
            'relevance': 0.8,
            'coherence': 0.7,
            'repetition': 0.1
        }
        
        example = LearningExample.from_dict(data)
        assert example.input_text == "Test"
        assert example.source == LearningSource.CONVERSATION
        assert example.priority == Priority.MEDIUM


class TestPerformanceMetrics:
    """Test PerformanceMetrics tracking."""
    
    def test_create_metrics(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics()
        assert metrics.total_conversations == 0
        assert metrics.positive_feedback == 0
        assert metrics.negative_feedback == 0
    
    def test_feedback_ratio(self):
        """Test calculating feedback ratio."""
        metrics = PerformanceMetrics()
        
        # No feedback yet
        assert metrics.feedback_ratio() == 0.5
        
        # Add some feedback
        metrics.positive_feedback = 7
        metrics.negative_feedback = 3
        assert metrics.feedback_ratio() == 0.7
    
    def test_health_score(self):
        """Test calculating health score."""
        metrics = PerformanceMetrics()
        metrics.total_conversations = 10
        metrics.positive_feedback = 8
        metrics.negative_feedback = 2
        metrics.examples_learned = 5
        metrics.avg_response_quality = 0.8
        
        health = metrics.health_score()
        assert 0.0 <= health <= 1.0
        assert health > 0.5  # Should be healthy with good metrics


class TestAutonomousConfig:
    """Test AutonomousConfig."""
    
    def test_create_config(self):
        """Test creating configuration."""
        config = AutonomousConfig()
        assert not config.enabled
        assert config.interval == 300
        assert not config.low_power_mode
    
    def test_effective_interval(self):
        """Test getting effective interval based on power mode."""
        config = AutonomousConfig()
        
        # Normal mode
        assert config.get_effective_interval() == 300
        
        # Low power mode
        config.low_power_mode = True
        assert config.get_effective_interval() == 1800
    
    def test_effective_max_actions(self):
        """Test getting effective max actions based on power mode."""
        config = AutonomousConfig()
        
        # Normal mode
        assert config.get_effective_max_actions() == 12
        
        # Low power mode
        config.low_power_mode = True
        assert config.get_effective_max_actions() == 2


class TestLearningEngine:
    """Test LearningEngine functionality."""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary model directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def engine(self, temp_model_dir, monkeypatch):
        """Create a learning engine with temporary directory."""
        # Mock CONFIG to use temp directory
        from forge_ai import config
        monkeypatch.setitem(config.CONFIG, 'models_dir', temp_model_dir)
        
        engine = LearningEngine("test_model")
        return engine
    
    def test_create_engine(self, engine):
        """Test creating learning engine."""
        assert engine.model_name == "test_model"
        assert isinstance(engine.learning_queue, list)
        assert isinstance(engine.metrics, PerformanceMetrics)
    
    def test_evaluate_response_quality(self, engine):
        """Test evaluating response quality."""
        quality = engine.evaluate_response_quality(
            "What is Python?",
            "Python is a high-level programming language known for its simplicity and readability."
        )
        
        assert 'relevance' in quality
        assert 'coherence' in quality
        assert 'repetition' in quality
        assert 'overall' in quality
        
        # All metrics should be between 0 and 1
        for metric in quality.values():
            assert 0.0 <= metric <= 1.0
    
    def test_evaluate_poor_response(self, engine):
        """Test evaluating a poor quality response."""
        quality = engine.evaluate_response_quality(
            "What is Python?",
            "The weather is nice today."
        )
        
        # Should have low relevance
        assert quality['relevance'] < 0.5
    
    def test_evaluate_repetitive_response(self, engine):
        """Test detecting repetition."""
        quality = engine.evaluate_response_quality(
            "Tell me about cats",
            "Cats are great cats are great cats are great cats are great"
        )
        
        # Should detect high repetition
        assert quality['repetition'] > 0.5
    
    def test_extract_topics(self, engine):
        """Test extracting topics from text."""
        topics = engine.extract_topics(
            "Machine learning and artificial intelligence are transforming technology",
            max_topics=3
        )
        
        assert isinstance(topics, list)
        assert len(topics) <= 3
        # Should extract meaningful words
        assert any(word in ['machine', 'learning', 'artificial', 'intelligence', 'technology'] 
                   for word in topics)
    
    def test_add_learning_example(self, engine):
        """Test adding learning example to queue."""
        example = LearningExample(
            input_text="What is AI?",
            output_text="AI is artificial intelligence.",
            source=LearningSource.CONVERSATION,
            priority=Priority.HIGH,
            quality_score=0.8
        )
        
        initial_count = len(engine.learning_queue)
        engine.add_learning_example(example)
        
        assert len(engine.learning_queue) == initial_count + 1
        assert engine.metrics.examples_learned > 0
    
    def test_record_feedback(self, engine):
        """Test recording user feedback."""
        engine.record_feedback(
            input_text="Hello",
            output_text="Hi there!",
            feedback='positive'
        )
        
        assert engine.metrics.positive_feedback == 1
        
        engine.record_feedback(
            input_text="Test",
            output_text="Response",
            feedback='negative'
        )
        
        assert engine.metrics.negative_feedback == 1
    
    def test_update_conversation_metrics(self, engine):
        """Test updating conversation metrics."""
        initial_count = engine.metrics.total_conversations
        
        engine.update_conversation_metrics(message_count=10)
        
        assert engine.metrics.total_conversations == initial_count + 1
        assert engine.metrics.avg_conversation_length > 0
    
    def test_knowledge_graph(self, engine):
        """Test knowledge graph building."""
        # Add examples with topics
        example1 = LearningExample(
            input_text="AI and ML",
            output_text="AI uses ML",
            source=LearningSource.RESEARCH,
            priority=Priority.LOW,
            quality_score=0.7,
            topics=['ai', 'ml']
        )
        
        example2 = LearningExample(
            input_text="ML and data",
            output_text="ML needs data",
            source=LearningSource.RESEARCH,
            priority=Priority.LOW,
            quality_score=0.7,
            topics=['ml', 'data']
        )
        
        engine.add_learning_example(example1)
        engine.add_learning_example(example2)
        
        # Check relationships
        related = engine.get_related_topics('ml')
        assert 'ai' in related or 'data' in related
    
    def test_export_training_data(self, engine, temp_model_dir):
        """Test exporting training data."""
        # Add some examples
        for i in range(5):
            example = LearningExample(
                input_text=f"Question {i}",
                output_text=f"Answer {i}",
                source=LearningSource.CONVERSATION,
                priority=Priority.MEDIUM,
                quality_score=0.7
            )
            engine.add_learning_example(example)
        
        export_path = engine.export_training_data(min_quality=0.5)
        
        assert export_path.exists()
        content = export_path.read_text()
        assert "Question" in content
        assert "Answer" in content
    
    def test_get_queue_stats(self, engine):
        """Test getting queue statistics."""
        # Add some examples
        engine.add_learning_example(LearningExample(
            input_text="Test",
            output_text="Response",
            source=LearningSource.CONVERSATION,
            priority=Priority.HIGH,
            quality_score=0.8
        ))
        
        stats = engine.get_queue_stats()
        
        assert 'total_examples' in stats
        assert 'by_priority' in stats
        assert 'by_source' in stats
        assert 'avg_quality' in stats
        assert stats['total_examples'] > 0
    
    def test_persistence(self, engine, temp_model_dir):
        """Test state persistence."""
        # Add some data
        example = LearningExample(
            input_text="Test persistence",
            output_text="Data should persist",
            source=LearningSource.PRACTICE,
            priority=Priority.MEDIUM,
            quality_score=0.75
        )
        engine.add_learning_example(example)
        engine.metrics.positive_feedback = 5
        
        # Save state
        engine.save_state()
        
        # Create new engine instance (should load saved state)
        from forge_ai import config
        engine2 = LearningEngine("test_model")
        
        # Check data persisted
        assert len(engine2.learning_queue) > 0
        assert engine2.metrics.positive_feedback == 5


class TestGetLearningEngine:
    """Test global engine access."""
    
    def test_get_engine(self, monkeypatch):
        """Test getting or creating engine instance."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            from forge_ai import config
            monkeypatch.setitem(config.CONFIG, 'models_dir', temp_dir)
            
            # Get engine for first time
            engine1 = get_learning_engine("test_model")
            assert engine1 is not None
            
            # Get same engine again
            engine2 = get_learning_engine("test_model")
            assert engine1 is engine2  # Should be same instance
            
            # Get different engine
            engine3 = get_learning_engine("other_model")
            assert engine3 is not engine1
        finally:
            shutil.rmtree(temp_dir)


# Integration tests would go here but require full ForgeAI setup
# These are unit tests focusing on the self-improvement components
