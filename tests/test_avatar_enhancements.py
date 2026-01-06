"""
Tests for enhanced avatar system.
"""
import pytest
from pathlib import Path
import tempfile
import json

from enigma.avatar import (
    AvatarController,
    AIAvatarIdentity,
    AvatarAppearance,
    EmotionExpressionSync,
    LipSync,
    AvatarCustomizer,
)
from enigma.core.personality import AIPersonality


class TestAvatarAppearance:
    """Test AvatarAppearance dataclass."""
    
    def test_default_appearance(self):
        """Test default appearance creation."""
        appearance = AvatarAppearance()
        
        assert appearance.style == "default"
        assert appearance.primary_color == "#6366f1"
        assert appearance.shape == "rounded"
        assert appearance.size == "medium"
        assert appearance.accessories == []
    
    def test_appearance_to_dict(self):
        """Test conversion to dictionary."""
        appearance = AvatarAppearance(
            style="anime",
            primary_color="#ff0000",
            accessories=["hat", "glasses"]
        )
        
        data = appearance.to_dict()
        assert data["style"] == "anime"
        assert data["primary_color"] == "#ff0000"
        assert data["accessories"] == ["hat", "glasses"]
    
    def test_appearance_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "style": "robot",
            "primary_color": "#00ff00",
            "size": "large",
            "accessories": ["tie"]
        }
        
        appearance = AvatarAppearance.from_dict(data)
        assert appearance.style == "robot"
        assert appearance.primary_color == "#00ff00"
        assert appearance.size == "large"
        assert appearance.accessories == ["tie"]


class TestAIAvatarIdentity:
    """Test AI avatar identity system."""
    
    def test_create_identity(self):
        """Test creating identity without personality."""
        identity = AIAvatarIdentity()
        
        assert identity.appearance is not None
        assert identity.evolution_history == []
    
    def test_design_from_personality_playful(self):
        """Test appearance design from playful personality."""
        personality = AIPersonality("test_model")
        personality.traits.playfulness = 0.9
        personality.traits.humor_level = 0.8
        
        identity = AIAvatarIdentity(personality)
        appearance = identity.design_from_personality()
        
        # Playful should result in rounded shape
        assert appearance.shape == "rounded"
        # Should have reasoning
        assert identity.reasoning != ""
        assert "playful" in identity.reasoning.lower() or "rounded" in identity.reasoning.lower()
    
    def test_design_from_personality_formal(self):
        """Test appearance design from formal personality."""
        personality = AIPersonality("test_model")
        personality.traits.formality = 0.9
        personality.traits.confidence = 0.8
        
        identity = AIAvatarIdentity(personality)
        appearance = identity.design_from_personality()
        
        # Formal + confident should result in angular shape
        assert appearance.shape == "angular"
        # Should have accessories like tie
        assert "tie" in appearance.accessories
    
    def test_design_from_personality_creative(self):
        """Test appearance design from creative personality."""
        personality = AIPersonality("test_model")
        personality.traits.creativity = 0.9
        
        identity = AIAvatarIdentity(personality)
        appearance = identity.design_from_personality()
        
        # Creative should result in unique/abstract style
        assert appearance.style in ["abstract", "unique"]
    
    def test_describe_desired_appearance(self):
        """Test natural language description."""
        identity = AIAvatarIdentity()
        
        appearance = identity.describe_desired_appearance("I want to look friendly and approachable")
        
        assert appearance.shape == "rounded"
        assert appearance.default_expression == "friendly"
        assert appearance.size == "small"  # Approachable = smaller
    
    def test_describe_professional(self):
        """Test professional description."""
        identity = AIAvatarIdentity()
        
        appearance = identity.describe_desired_appearance("I want a professional formal look")
        
        assert appearance.style == "realistic"
        assert "tie" in appearance.accessories
    
    def test_choose_expression_for_mood(self):
        """Test mood-to-expression mapping."""
        identity = AIAvatarIdentity()
        
        assert identity.choose_expression_for_mood("happy") == "happy"
        assert identity.choose_expression_for_mood("curious") == "thinking"
        assert identity.choose_expression_for_mood("concerned") == "worried"
        assert identity.choose_expression_for_mood("unknown") == "neutral"
    
    def test_explain_appearance(self):
        """Test appearance explanation."""
        personality = AIPersonality("test_model")
        personality.traits.playfulness = 0.8
        
        identity = AIAvatarIdentity(personality)
        identity.design_from_personality()
        
        explanation = identity.explain_appearance_choices()
        assert explanation != ""
        assert len(explanation) > 20
    
    def test_save_and_load(self):
        """Test saving and loading identity."""
        personality = AIPersonality("test_model")
        identity = AIAvatarIdentity(personality)
        identity.design_from_personality()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "identity.json"
            
            # Save
            identity.save(filepath)
            assert filepath.exists()
            
            # Load
            new_identity = AIAvatarIdentity()
            success = new_identity.load(filepath)
            
            assert success
            assert new_identity.appearance.style == identity.appearance.style
            assert new_identity.appearance.primary_color == identity.appearance.primary_color


class TestEmotionExpressionSync:
    """Test emotion synchronization."""
    
    def test_create_sync(self):
        """Test creating emotion sync."""
        avatar = AvatarController()
        personality = AIPersonality("test_model")
        
        sync = EmotionExpressionSync(avatar, personality)
        
        assert sync.avatar == avatar
        assert sync.personality == personality
    
    def test_mood_to_expression_mapping(self):
        """Test mood-to-expression mapping."""
        avatar = AvatarController()
        sync = EmotionExpressionSync(avatar, None)
        
        assert sync.MOOD_TO_EXPRESSION["happy"] == "happy"
        assert sync.MOOD_TO_EXPRESSION["curious"] == "thinking"
        assert sync.MOOD_TO_EXPRESSION["sad"] == "sad"
    
    def test_detect_emotion_from_text(self):
        """Test emotion detection from text."""
        avatar = AvatarController()
        sync = EmotionExpressionSync(avatar, None)
        
        # Happy text
        emotion = sync.detect_emotion_from_text("I'm so happy and excited!")
        assert emotion in ["happy", "excited"]
        
        # Thinking text
        emotion = sync.detect_emotion_from_text("Let me think about this...")
        assert emotion == "thinking"
        
        # Neutral text
        emotion = sync.detect_emotion_from_text("The sky is blue.")
        assert emotion == "neutral"


class TestLipSync:
    """Test lip sync system."""
    
    def test_create_lip_sync(self):
        """Test creating lip sync."""
        lip_sync = LipSync()
        
        assert lip_sync.current_viseme == "silence"
    
    def test_text_to_visemes(self):
        """Test converting text to visemes."""
        lip_sync = LipSync()
        
        visemes = lip_sync.text_to_visemes("Hello world")
        
        assert len(visemes) > 0
        # Should have viseme tuples with (name, duration)
        for viseme_name, duration in visemes:
            assert isinstance(viseme_name, str)
            assert isinstance(duration, float)
            assert duration > 0
    
    def test_animate_speaking(self):
        """Test generating speaking animation frames."""
        lip_sync = LipSync()
        
        frames = lip_sync.animate_speaking("Hello there")
        
        assert len(frames) > 0
        # Should alternate between speaking frames
        assert "speaking_1" in frames or "speaking_2" in frames
        # Should end with idle
        assert frames[-1] == "idle"


class TestAvatarCustomizer:
    """Test avatar customizer."""
    
    def test_create_customizer(self):
        """Test creating customizer."""
        avatar = AvatarController()
        customizer = AvatarCustomizer(avatar)
        
        assert customizer.avatar == avatar
    
    def test_set_style(self):
        """Test setting avatar style."""
        avatar = AvatarController()
        avatar._identity = AIAvatarIdentity()
        customizer = AvatarCustomizer(avatar)
        
        success = customizer.set_style("anime")
        assert success
        assert avatar._identity.appearance.style == "anime"
    
    def test_set_invalid_style(self):
        """Test setting invalid style."""
        avatar = AvatarController()
        avatar._identity = AIAvatarIdentity()
        customizer = AvatarCustomizer(avatar)
        
        success = customizer.set_style("invalid_style")
        assert not success
    
    def test_set_colors(self):
        """Test setting colors."""
        avatar = AvatarController()
        avatar._identity = AIAvatarIdentity()
        customizer = AvatarCustomizer(avatar)
        
        success = customizer.set_colors(primary="#ff0000", accent="#00ff00")
        assert success
        assert avatar._identity.appearance.primary_color == "#ff0000"
        assert avatar._identity.appearance.accent_color == "#00ff00"
    
    def test_apply_color_preset(self):
        """Test applying color preset."""
        avatar = AvatarController()
        avatar._identity = AIAvatarIdentity()
        customizer = AvatarCustomizer(avatar)
        
        success = customizer.apply_color_preset("warm")
        assert success
        # Warm preset should have warm colors
        assert avatar._identity.appearance.primary_color == "#f59e0b"
    
    def test_add_remove_accessory(self):
        """Test adding and removing accessories."""
        avatar = AvatarController()
        avatar._identity = AIAvatarIdentity()
        customizer = AvatarCustomizer(avatar)
        
        # Add accessory
        success = customizer.add_accessory("hat")
        assert success
        assert "hat" in avatar._identity.appearance.accessories
        
        # Remove accessory
        success = customizer.remove_accessory("hat")
        assert success
        assert "hat" not in avatar._identity.appearance.accessories
    
    def test_export_import_appearance(self):
        """Test exporting and importing appearance."""
        avatar = AvatarController()
        avatar._identity = AIAvatarIdentity()
        avatar._identity.appearance.style = "anime"
        avatar._identity.appearance.primary_color = "#ff0000"
        
        customizer = AvatarCustomizer(avatar)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "appearance.json"
            
            # Export
            success = customizer.export_appearance(str(filepath))
            assert success
            assert filepath.exists()
            
            # Modify appearance
            avatar._identity.appearance.style = "default"
            
            # Import
            success = customizer.import_appearance(str(filepath))
            assert success
            assert avatar._identity.appearance.style == "anime"
            assert avatar._identity.appearance.primary_color == "#ff0000"


class TestAvatarController:
    """Test avatar controller integration."""
    
    def test_link_personality(self):
        """Test linking personality to avatar."""
        avatar = AvatarController()
        personality = AIPersonality("test_model")
        
        avatar.link_personality(personality)
        
        assert avatar._identity is not None
        assert avatar._emotion_sync is not None
        assert avatar._identity.personality == personality
    
    def test_auto_design(self):
        """Test AI auto-design."""
        avatar = AvatarController()
        personality = AIPersonality("test_model")
        personality.traits.playfulness = 0.9
        
        avatar.link_personality(personality)
        appearance = avatar.auto_design()
        
        assert appearance is not None
        assert appearance.shape == "rounded"  # Playful = rounded
    
    def test_describe_desired_appearance(self):
        """Test describing desired appearance."""
        avatar = AvatarController()
        
        appearance = avatar.describe_desired_appearance("I want to be friendly")
        
        assert appearance is not None
        assert appearance.shape == "rounded"
    
    def test_get_customizer(self):
        """Test getting customizer."""
        avatar = AvatarController()
        
        customizer = avatar.get_customizer()
        
        assert customizer is not None
        assert isinstance(customizer, AvatarCustomizer)
        
        # Should return same instance
        customizer2 = avatar.get_customizer()
        assert customizer is customizer2
    
    def test_set_appearance(self):
        """Test setting appearance directly."""
        avatar = AvatarController()
        
        appearance = AvatarAppearance(
            style="robot",
            primary_color="#0000ff"
        )
        
        avatar.set_appearance(appearance)
        
        assert avatar._identity is not None
        assert avatar._identity.appearance.style == "robot"
        assert avatar._identity.appearance.primary_color == "#0000ff"
    
    def test_explain_appearance(self):
        """Test explaining appearance."""
        avatar = AvatarController()
        personality = AIPersonality("test_model")
        
        avatar.link_personality(personality)
        avatar.auto_design()
        
        explanation = avatar.explain_appearance()
        
        assert explanation != ""
        assert len(explanation) > 20
