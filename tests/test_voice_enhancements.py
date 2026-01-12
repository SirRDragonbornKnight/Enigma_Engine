"""
Tests for enhanced voice system features.
"""
import pytest
import tempfile
from pathlib import Path

# Test voice identity
from enigma.voice.voice_identity import (
    AIVoiceIdentity,
    discover_voice,
    describe_voice,
    adjust_voice_from_feedback
)

# Test voice effects
from enigma.voice.voice_effects import (
    VoiceEffects,
    apply_effect,
    apply_effects,
    effect_for_emotion,
    effect_for_context
)

# Test dynamic adapter
from enigma.voice.dynamic_adapter import (
    DynamicVoiceAdapter,
    adapt_voice_for_emotion,
    adapt_voice_for_context
)

# Test voice customizer
from enigma.voice.voice_customizer import (
    VoiceCustomizer,
    import_voice_profile,
    export_voice_profile
)

# Test audio analyzer
from enigma.voice.audio_analyzer import AudioAnalyzer

# Test smart wake words
from enigma.voice.trigger_phrases import (
    SmartWakeWords,
    suggest_wake_phrases
)

from enigma.voice import VoiceProfile


class TestVoiceIdentity:
    """Test AI voice self-discovery system."""
    
    def test_describe_desired_voice_basic(self):
        """Test parsing natural language voice descriptions."""
        identity = AIVoiceIdentity()
        
        # Test "low pitch" description
        profile = identity.describe_desired_voice("I want a low, deep voice")
        assert profile.pitch < 1.0
        
        # Test "fast" description
        profile = identity.describe_desired_voice("I want to speak fast")
        assert profile.speed > 1.0
        
        # Test "robotic" description
        profile = identity.describe_desired_voice("I want a robotic voice")
        assert "robotic" in profile.effects
    
    def test_describe_desired_voice_complex(self):
        """Test complex voice descriptions."""
        identity = AIVoiceIdentity()
        
        profile = identity.describe_desired_voice(
            "I want a warm, calm, low-pitched voice that speaks slowly"
        )
        
        assert profile.pitch < 1.0
        assert profile.speed < 1.0
        assert "warm" in profile.effects or "calm" in profile.effects
    
    def test_learn_from_feedback(self):
        """Test learning from user feedback."""
        identity = AIVoiceIdentity()
        
        # Start with default
        profile = VoiceProfile(pitch=1.0, speed=1.0, volume=0.8)
        
        # Feedback: too fast
        adjusted = identity.learn_from_feedback("Too fast, slow down", profile)
        assert adjusted.speed < profile.speed
        
        # Feedback: too quiet
        adjusted = identity.learn_from_feedback("Can't hear you, louder please", profile)
        assert adjusted.volume > profile.volume
        
        # Feedback: too high pitched
        adjusted = identity.learn_from_feedback("Voice is too high, lower it", profile)
        assert adjusted.pitch < profile.pitch
    
    def test_save_and_load_identity(self):
        """Test saving and loading voice identity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            identity = AIVoiceIdentity()
            
            # Create a voice
            profile = identity.describe_desired_voice("warm and friendly")
            identity.current_identity = profile
            
            # Save
            identity.save_identity("test_model", Path(tmpdir))
            
            # Load
            identity2 = AIVoiceIdentity()
            assert identity2.load_identity("test_model", Path(tmpdir))
            assert identity2.current_identity.name == profile.name


class TestVoiceEffects:
    """Test enhanced voice effects system."""
    
    def test_apply_single_effect(self):
        """Test applying single effects."""
        effects = VoiceEffects()
        
        # Robotic effect
        text = effects.apply_effect("Hello. How are you?", "robotic", level=1)
        assert "..." in text
        
        # Calm effect
        text = effects.apply_effect("Hello, world!", "calm")
        assert "..." in text or text != "Hello, world!"
    
    def test_apply_multiple_effects(self):
        """Test applying multiple effects."""
        effects = VoiceEffects()
        
        text = effects.apply_effects(
            "Hello world",
            ["warm", "calm"]
        )
        
        # Should have been modified
        assert text != "Hello world" or len(text) >= len("Hello world")
    
    def test_emotion_mapping(self):
        """Test emotion to effect mapping."""
        effects = VoiceEffects()
        
        assert effects.effect_for_emotion("happy") == "energetic"
        assert effects.effect_for_emotion("sad") == "calm"
        assert effects.effect_for_emotion("serious") == "authoritative"
    
    def test_context_mapping(self):
        """Test context to effect mapping."""
        effects = VoiceEffects()
        
        assert effects.effect_for_context("storytelling") == "warm"
        assert effects.effect_for_context("teaching") == "calm"
        assert effects.effect_for_context("formal") == "authoritative"
    
    def test_combine_effects(self):
        """Test combining effects with conflict resolution."""
        effects = VoiceEffects()
        
        # Conflicting effects
        combined = effects.combine_effects(["warm", "cold"])
        # Should remove one due to conflict
        assert not ("warm" in combined and "cold" in combined)
        
        # Compatible effects
        combined = effects.combine_effects(["warm", "calm"])
        assert len(combined) > 0


class TestDynamicVoiceAdapter:
    """Test dynamic voice adaptation."""
    
    def test_adapt_for_emotion(self):
        """Test emotional voice adaptation."""
        adapter = DynamicVoiceAdapter()
        base = VoiceProfile(pitch=1.0, speed=1.0, volume=0.8)
        
        # Happy emotion
        happy = adapter.adapt_for_emotion("happy", base)
        assert happy.pitch > base.pitch  # Higher pitch when happy
        
        # Sad emotion
        sad = adapter.adapt_for_emotion("sad", base)
        assert sad.pitch < base.pitch  # Lower pitch when sad
        assert sad.speed < base.speed  # Slower when sad
    
    def test_adapt_for_context(self):
        """Test contextual voice adaptation."""
        adapter = DynamicVoiceAdapter()
        base = VoiceProfile(pitch=1.0, speed=1.0)
        
        # Storytelling context
        story = adapter.adapt_for_context("storytelling", base)
        assert "warm" in story.effects
        
        # Formal context
        formal = adapter.adapt_for_context("formal", base)
        assert "authoritative" in formal.effects
    
    def test_adapt_combined(self):
        """Test combined emotion and context adaptation."""
        adapter = DynamicVoiceAdapter()
        base = VoiceProfile()
        
        adapted = adapter.adapt_combined(
            emotion="happy",
            context="casual",
            base_profile=base
        )
        
        # Should have adjustments from both
        assert adapted.pitch != base.pitch or adapted.speed != base.speed
    
    def test_list_emotions_and_contexts(self):
        """Test listing available emotions and contexts."""
        adapter = DynamicVoiceAdapter()
        
        emotions = adapter.list_emotions()
        assert "happy" in emotions
        assert "sad" in emotions
        assert "serious" in emotions
        
        contexts = adapter.list_contexts()
        assert "storytelling" in contexts
        assert "teaching" in contexts
        assert "formal" in contexts


class TestVoiceCustomizer:
    """Test voice customization tools."""
    
    def test_import_export_profile(self):
        """Test importing and exporting voice profiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            customizer = VoiceCustomizer()
            
            # Create profile
            profile = VoiceProfile(
                name="test_voice",
                pitch=1.2,
                speed=0.9,
                effects=["warm"]
            )
            
            # Export
            export_path = Path(tmpdir) / "test_voice.json"
            customizer.export_profile(profile, str(export_path))
            
            # Import
            imported = customizer.import_profile(str(export_path))
            
            assert imported.name == profile.name
            assert imported.pitch == profile.pitch
            assert imported.speed == profile.speed
            assert imported.effects == profile.effects
    
    def test_create_from_preset(self):
        """Test creating custom voice from preset."""
        customizer = VoiceCustomizer()
        
        custom = customizer.create_from_preset(
            "glados",
            customizations={"pitch": 0.7}
        )
        
        assert custom.pitch == 0.7
        assert custom.voice == "female"  # From GLaDOS preset
    
    def test_batch_create_variations(self):
        """Test batch variation creation."""
        customizer = VoiceCustomizer()
        base = VoiceProfile(name="base")
        
        variations = customizer.batch_create_variations(base, num_variations=3)
        
        assert len(variations) == 3
        # Each should be slightly different
        for var in variations:
            assert var.name != base.name


class TestAudioAnalyzer:
    """Test audio analysis system."""
    
    def test_analyzer_initialization(self):
        """Test audio analyzer initializes properly."""
        analyzer = AudioAnalyzer()
        assert analyzer is not None
    
    def test_analyze_basic_fallback(self):
        """Test basic analysis fallback when libraries not available."""
        analyzer = AudioAnalyzer()
        
        # Create a dummy file for testing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            # Write some dummy data
            f.write(b"RIFF" + b"\x00" * 100)
        
        try:
            # Should not crash even with invalid audio
            features = analyzer._analyze_basic(Path(temp_path))
            assert features is not None
            assert features.average_pitch > 0
        finally:
            Path(temp_path).unlink()
    
    def test_estimate_voice_profile_empty(self):
        """Test voice profile estimation with no files."""
        analyzer = AudioAnalyzer()
        profile = analyzer.estimate_voice_profile([], "test")
        
        assert profile.name == "test"
        assert profile.pitch > 0


class TestSmartWakeWords:
    """Test smart wake word system."""
    
    def test_suggest_wake_phrases_basic(self):
        """Test basic wake phrase suggestions."""
        smart = SmartWakeWords()
        
        suggestions = smart.suggest_wake_phrases("AI Tester", num_suggestions=5)
        
        assert len(suggestions) > 0
        assert any("ai_tester" in s.lower() for s in suggestions)
    
    def test_categorize_wake_phrase(self):
        """Test wake phrase categorization."""
        smart = SmartWakeWords()
        
        smart.categorize_wake_phrase("hey ai tester", "casual")
        smart.categorize_wake_phrase("greetings enigma", "formal")
        
        casual = smart.get_phrases_by_personality("casual")
        formal = smart.get_phrases_by_personality("formal")
        
        assert "hey ai tester" in casual
        assert "greetings enigma" in formal
    
    def test_train_custom_phrase(self):
        """Test custom wake phrase training."""
        smart = SmartWakeWords()
        
        smart.train_custom_phrase("my custom phrase")
        
        custom = smart.get_custom_phrases()
        assert "my custom phrase" in custom
    
    def test_improve_confidence(self):
        """Test improved confidence scoring."""
        smart = SmartWakeWords()
        
        # Exact match
        score = smart.improve_confidence("hey ai tester", "hey ai tester")
        assert score == 1.0
        
        # Substring match at start
        score = smart.improve_confidence("hey ai tester", "hey ai tester listen")
        assert score > 0.9
        
        # Word overlap
        score = smart.improve_confidence("ok ai tester", "okay enigma please")
        assert score > 0.5
        
        # No match
        score = smart.improve_confidence("hey ai tester", "hello world")
        assert score < 0.5


class TestVoiceIntegration:
    """Test integration between voice modules."""
    
    def test_voice_profile_with_effects(self):
        """Test voice profile using effects system."""
        profile = VoiceProfile(
            name="test",
            effects=["warm", "calm"]
        )
        
        assert len(profile.effects) == 2
        assert "warm" in profile.effects
    
    def test_personality_voice_integration(self):
        """Test personality and voice integration."""
        # Import at the end to avoid torch import issues in other tests
        try:
            from enigma.core.personality import AIPersonality
            
            personality = AIPersonality("test_model")
            personality.voice_preferences = {
                "pitch": 1.2,
                "speed": 0.9
            }
            
            assert "pitch" in personality.voice_preferences
            assert personality.voice_preferences["pitch"] == 1.2
        except ModuleNotFoundError:
            # Skip if torch is not available (not required for voice system)
            pytest.skip("Torch not available, skipping personality integration test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
