"""
Tests for personality system enhancements.
"""
import pytest
from pathlib import Path
import tempfile

from enigma.core.personality import (
    AIPersonality,
    PersonalityTraits
)


class TestPersonalityEnhancements:
    """Test enhanced personality features."""
    
    def test_user_override(self):
        """Test setting user overrides for traits."""
        personality = AIPersonality("test_model")
        
        # Set original trait
        personality.traits.humor_level = 0.5
        
        # Apply user override
        personality.set_user_override('humor_level', 0.9)
        
        # Effective trait should be override
        assert personality.get_effective_trait('humor_level') == 0.9
        
        # Original trait unchanged
        assert personality.traits.humor_level == 0.5
    
    def test_clear_override(self):
        """Test clearing user overrides."""
        personality = AIPersonality("test_model")
        
        personality.set_user_override('humor_level', 0.9)
        assert personality.get_effective_trait('humor_level') == 0.9
        
        personality.clear_user_override('humor_level')
        assert personality.get_effective_trait('humor_level') == personality.traits.humor_level
    
    def test_preset_loading(self):
        """Test loading personality presets."""
        personality = AIPersonality("test_model")
        
        # Apply professional preset
        personality.set_preset('professional')
        
        # Check that traits are set as expected
        assert personality.get_effective_trait('formality') == 0.8
        assert personality.get_effective_trait('humor_level') == 0.2
    
    def test_evolution_respects_overrides(self):
        """Test that evolution doesn't change overridden traits."""
        personality = AIPersonality("test_model")
        
        # Set override
        personality.set_user_override('humor_level', 0.9)
        
        # Try to evolve
        personality.evolve_from_interaction(
            "Hello",
            "Hi there! 😊",
            feedback="positive"
        )
        
        # Override should remain
        assert personality.get_effective_trait('humor_level') == 0.9
    
    def test_personality_description(self):
        """Test personality description includes overrides."""
        personality = AIPersonality("test_model")
        
        personality.set_user_override('humor_level', 0.9)
        
        description = personality.get_personality_description()
        
        assert '*' in description  # Override marker
        assert 'test_model' in description
    
    def test_save_and_load_with_overrides(self):
        """Test saving and loading with overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            personality = AIPersonality("test_model")
            personality.set_user_override('humor_level', 0.9)
            personality.set_user_override('formality', 0.2)
            
            # Save
            personality.save(Path(tmpdir))
            
            # Load
            personality2 = AIPersonality("test_model")
            assert personality2.load(Path(tmpdir))
            
            # Check overrides preserved
            assert personality2.get_effective_trait('humor_level') == 0.9
            assert personality2.get_effective_trait('formality') == 0.2
    
    def test_allow_evolution_flag(self):
        """Test allow_evolution flag."""
        personality = AIPersonality("test_model")
        
        initial_humor = personality.traits.humor_level
        
        # Disable evolution
        personality.allow_evolution = False
        
        # Try to evolve
        personality.evolve_from_interaction(
            "Hello",
            "Hi! 😊😊😊",
            feedback="positive"
        )
        
        # Should not have changed
        assert personality.traits.humor_level == initial_humor
    
    def test_get_all_effective_traits(self):
        """Test getting all effective traits."""
        personality = AIPersonality("test_model")
        
        personality.set_user_override('humor_level', 0.9)
        personality.set_user_override('formality', 0.1)
        
        traits = personality.get_all_effective_traits()
        
        assert traits['humor_level'] == 0.9
        assert traits['formality'] == 0.1
        assert 'empathy' in traits  # Should include all traits


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
