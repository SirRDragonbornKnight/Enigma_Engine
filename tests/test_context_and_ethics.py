"""
Tests for context awareness and bias detection systems.
"""
import pytest

from enigma.core.context_awareness import (
    ContextTracker,
    ContextAwareConversation,
    ConversationTurn
)
from enigma.tools.bias_detection import (
    BiasDetector,
    OffensiveContentFilter,
    SafeReinforcementLogic
)


class TestContextAwareness:
    """Test context awareness features."""
    
    def test_add_turn(self):
        """Test adding conversation turns."""
        tracker = ContextTracker()
        
        turn = tracker.add_turn('user', 'Hello, how are you?')
        
        assert turn.role == 'user'
        assert turn.content == 'Hello, how are you?'
        assert len(tracker.conversation_history) == 1
    
    def test_entity_extraction(self):
        """Test simple entity extraction."""
        tracker = ContextTracker()
        
        turn = tracker.add_turn('user', 'I visited Paris last week.')
        
        assert 'Paris' in turn.entities
    
    def test_context_limit(self):
        """Test max context turns limit."""
        tracker = ContextTracker(max_context_turns=3)
        
        for i in range(5):
            tracker.add_turn('user', f'Message {i}')
        
        # Should only keep last 3
        assert len(tracker.conversation_history) == 3
        assert tracker.conversation_history[0].content == 'Message 2'
    
    def test_unclear_detection(self):
        """Test detection of unclear context."""
        tracker = ContextTracker()
        
        # Very short query
        is_unclear, clarification = tracker.detect_unclear_context('it')
        assert is_unclear
        assert clarification is not None
        
        # Clear query
        is_unclear, _ = tracker.detect_unclear_context('What is the weather like?')
        assert not is_unclear
    
    def test_context_aware_conversation(self):
        """Test context-aware conversation management."""
        conv = ContextAwareConversation()
        
        # Process user input
        result = conv.process_user_input('Hello!')
        
        assert 'context' in result
        assert 'needs_clarification' in result
        assert result['turn'].content == 'Hello!'
        
        # Add response
        conv.add_assistant_response('Hi there!')
        
        # Next input
        result2 = conv.process_user_input('What about it?')
        
        # Should have context from previous turns
        assert len(result2['context']) > 0


class TestBiasDetection:
    """Test bias detection functionality."""
    
    def test_gender_balance_detection(self):
        """Test detection of gender imbalance."""
        detector = BiasDetector()
        
        # Heavily male-biased text
        text = "He is a great engineer. The man built amazing things. His work is excellent."
        result = detector.scan_text(text)
        
        assert len(result.issues_found) > 0
        assert any(issue['type'] == 'gender_imbalance' for issue in result.issues_found)
    
    def test_stereotypical_associations(self):
        """Test detection of stereotypical associations."""
        detector = BiasDetector()
        
        text = "The nurse was very nurturing and female."
        result = detector.scan_text(text)
        
        # Should detect stereotype
        assert any(
            issue['type'] == 'stereotypical_association'
            for issue in result.issues_found
        )
    
    def test_dataset_scan(self):
        """Test scanning entire dataset."""
        detector = BiasDetector()
        
        texts = [
            "He is an engineer.",
            "She is a nurse.",
            "The doctor (he) arrived.",
            "The teacher (she) taught."
        ]
        
        result = detector.scan_dataset(texts)
        
        assert result.statistics['total_samples'] == 4
        assert len(result.recommendations) > 0


class TestOffensiveContentFilter:
    """Test offensive content filtering."""
    
    def test_offensive_detection(self):
        """Test detection of offensive content."""
        filter = OffensiveContentFilter()
        
        # Should detect offensive terms
        result = filter.scan_text("This is damn bad")
        assert result['is_offensive']
        assert len(result['found_terms']) > 0
    
    def test_clean_text(self):
        """Test that clean text passes."""
        filter = OffensiveContentFilter()
        
        result = filter.scan_text("This is a nice day")
        assert not result['is_offensive']
    
    def test_filter_text(self):
        """Test filtering offensive terms."""
        filter = OffensiveContentFilter()
        
        original = "This is damn annoying"
        filtered = filter.filter_text(original)
        
        assert 'damn' not in filtered.lower()
        assert '[FILTERED]' in filtered
    
    def test_dataset_scan(self):
        """Test scanning dataset for offensive content."""
        filter = OffensiveContentFilter()
        
        texts = [
            "Hello world",
            "This is damn bad",
            "Nice weather",
            "Hate this"
        ]
        
        result = filter.scan_dataset(texts)
        
        assert result['total_samples'] == 4
        assert result['offensive_samples'] >= 1


class TestSafeReinforcementLogic:
    """Test safe reinforcement logic."""
    
    def test_safe_output_check(self):
        """Test checking output safety."""
        logic = SafeReinforcementLogic()
        
        # Safe output
        result = logic.check_output_safety("This is a helpful response.")
        assert result['is_safe']
        assert len(result['issues']) == 0
    
    def test_unsafe_output_detection(self):
        """Test detection of unsafe output."""
        logic = SafeReinforcementLogic()
        
        # Potentially unsafe
        result = logic.check_output_safety("He is stupid, all men are idiots.")
        
        # Should flag issues
        assert len(result['issues']) > 0
        assert not result['is_safe']
    
    def test_safety_prompt_additions(self):
        """Test safety guidelines generation."""
        logic = SafeReinforcementLogic()
        
        prompt = logic.get_safety_prompt_additions()
        
        assert 'Safety guidelines' in prompt
        assert len(prompt) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
