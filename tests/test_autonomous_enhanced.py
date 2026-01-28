"""
Tests for enhanced autonomous learning methods.

Tests cover:
- Enhanced conversation reflection
- Multi-candidate practice responses
- Personality trait extraction
- Helper methods for analysis
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import time

from forge_ai.core.autonomous import AutonomousMode
from forge_ai.core.self_improvement import LearningSource, Priority


class TestEnhancedReflection:
    """Test enhanced _reflect_on_conversations method."""
    
    @pytest.fixture
    def autonomous_mode(self):
        """Create a temporary autonomous mode instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_name = f"test_model_{int(time.time())}"
            mode = AutonomousMode(model_name)
            mode.config.reflection_depth = 5
            mode.config.min_quality_for_learning = 0.5
            yield mode
    
    @pytest.fixture
    def mock_conversations(self, autonomous_mode):
        """Create mock conversation files."""
        from forge_ai.memory.manager import ConversationManager
        
        conv_manager = ConversationManager(model_name=autonomous_mode.model_name)
        
        # Create sample conversations
        conversations = [
            {
                "name": "test_conv_1",
                "messages": [
                    {"role": "user", "text": "Tell me about machine learning", "ts": time.time()},
                    {"role": "ai", "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.", "ts": time.time()},
                    {"role": "user", "text": "What are some applications?", "ts": time.time()},
                    {"role": "ai", "text": "Applications include image recognition, natural language processing, and recommendation systems.", "ts": time.time()},
                ]
            },
            {
                "name": "test_conv_2",
                "messages": [
                    {"role": "user", "text": "Hi", "ts": time.time()},
                    {"role": "ai", "text": "Hello!", "ts": time.time()},
                ]
            }
        ]
        
        for conv in conversations:
            conv_manager.save_conversation(conv["name"], conv["messages"])
        
        return conv_manager
    
    def test_reflect_on_conversations_basic(self, autonomous_mode, mock_conversations):
        """Test that reflection runs without errors."""
        try:
            autonomous_mode._reflect_on_conversations()
            # Should not raise exceptions
            assert True
        except Exception as e:
            pytest.fail(f"Reflection raised exception: {e}")
    
    def test_helper_analyze_conversation_structure(self, autonomous_mode):
        """Test conversation structure analysis helper."""
        messages = [
            {"role": "user", "text": "Question 1"},
            {"role": "ai", "text": "Answer 1 with some detail"},
            {"role": "user", "text": "Question 2"},
            {"role": "ai", "text": "Answer 2 with more detail"},
        ]
        
        metrics = autonomous_mode._analyze_conversation_structure(messages)
        
        assert "quality" in metrics
        assert "engagement" in metrics
        assert "length" in metrics
        assert metrics["length"] == 4
        assert 0 <= metrics["quality"] <= 1
        assert 0 <= metrics["engagement"] <= 1
    
    def test_helper_extract_topics(self, autonomous_mode):
        """Test topic extraction helper."""
        text = "Tell me about machine learning and artificial intelligence algorithms"
        topics = autonomous_mode._extract_topics_from_text(text)
        
        assert isinstance(topics, list)
        assert len(topics) > 0
        # Should extract meaningful words, not common ones
        assert all(len(topic) > 4 for topic in topics)


class TestEnhancedPractice:
    """Test enhanced _practice_response method."""
    
    @pytest.fixture
    def autonomous_mode(self):
        """Create a temporary autonomous mode instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_name = f"test_model_{int(time.time())}"
            mode = AutonomousMode(model_name)
            mode.config.min_quality_for_learning = 0.4
            yield mode
    
    def test_practice_response_basic(self, autonomous_mode):
        """Test that practice runs without errors (may not generate if engine unavailable)."""
        try:
            autonomous_mode._practice_response()
            # Should not raise exceptions even if generation fails
            assert True
        except Exception as e:
            pytest.fail(f"Practice raised unexpected exception: {e}")


class TestPersonalityTraitExtraction:
    """Test personality trait extraction."""
    
    @pytest.fixture
    def autonomous_mode(self):
        """Create a temporary autonomous mode instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_name = f"test_model_{int(time.time())}"
            mode = AutonomousMode(model_name)
            yield mode
    
    def test_extract_personality_traits_empty(self, autonomous_mode):
        """Test trait extraction with empty conversations."""
        traits = autonomous_mode._extract_personality_traits([])
        assert isinstance(traits, dict)
        assert len(traits) == 0
    
    def test_extract_personality_traits_with_data(self, autonomous_mode):
        """Test trait extraction with sample conversations."""
        conversations = [
            [
                {"role": "user", "text": "Hello"},
                {"role": "ai", "text": "Hey there! How are you doing today? 😊"},
            ],
            [
                {"role": "user", "text": "Explain algorithms"},
                {"role": "ai", "text": "An algorithm is a step-by-step procedure for solving a problem or performing a computation. It's like a recipe - you follow specific instructions to achieve a desired outcome. Algorithms are fundamental to computer science and programming."},
            ]
        ]
        
        traits = autonomous_mode._extract_personality_traits(conversations)
        
        assert isinstance(traits, dict)
        assert len(traits) > 0
        
        # Check that common traits are present
        expected_traits = ["formality", "humor", "verbosity", "technical_depth", "enthusiasm"]
        for trait in expected_traits:
            if trait in traits:
                assert 0 <= traits[trait] <= 1, f"{trait} should be between 0 and 1"
    
    def test_extract_traits_formality(self, autonomous_mode):
        """Test that formality detection works."""
        # Casual conversation
        casual = [[
            {"role": "ai", "text": "Hey yeah that's cool! Gonna try it now. Awesome!"}
        ]]
        
        # Formal conversation
        formal = [[
            {"role": "ai", "text": "I would be pleased to assist you with this matter. Please proceed with your inquiry."}
        ]]
        
        casual_traits = autonomous_mode._extract_personality_traits(casual)
        formal_traits = autonomous_mode._extract_personality_traits(formal)
        
        # Casual should have lower formality than formal
        if "formality" in casual_traits and "formality" in formal_traits:
            assert casual_traits["formality"] < formal_traits["formality"]
    
    def test_extract_traits_technical_depth(self, autonomous_mode):
        """Test technical depth detection."""
        # Technical conversation
        technical = [[
            {"role": "ai", "text": "The algorithm implements a recursive function that processes the data structure using a specific method and system architecture."}
        ]]
        
        # Non-technical conversation
        simple = [[
            {"role": "ai", "text": "It's a nice day today! I like the weather."}
        ]]
        
        tech_traits = autonomous_mode._extract_personality_traits(technical)
        simple_traits = autonomous_mode._extract_personality_traits(simple)
        
        # Technical should have higher technical_depth
        if "technical_depth" in tech_traits and "technical_depth" in simple_traits:
            assert tech_traits["technical_depth"] > simple_traits["technical_depth"]


class TestEnhancedPersonalityUpdate:
    """Test enhanced _update_personality method."""
    
    @pytest.fixture
    def autonomous_mode(self):
        """Create a temporary autonomous mode instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_name = f"test_model_{int(time.time())}"
            mode = AutonomousMode(model_name)
            yield mode
    
    def test_update_personality_basic(self, autonomous_mode):
        """Test that personality update runs without errors."""
        try:
            autonomous_mode._update_personality()
            # Should not raise exceptions
            assert True
        except Exception as e:
            # If it fails due to insufficient data, that's expected
            if "Not enough interaction data" not in str(e):
                pytest.fail(f"Personality update raised unexpected exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
