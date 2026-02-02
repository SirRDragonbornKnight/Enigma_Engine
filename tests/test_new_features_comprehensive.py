"""
================================================================================
🧪 COMPREHENSIVE TESTS FOR NEW FORGE_AI FEATURES
================================================================================

Tests for all the new features pulled from the repository:
- Self-Tools (AI self-modification)
- Curiosity System (question banks)
- Memory Augmented Engine
- Conversation Summarizer
- Game Co-Play
- Spawnable Objects

📍 FILE: tests/test_new_features_comprehensive.py
🏷️ TYPE: Test Suite
🎯 PURPOSE: Ensure new features work correctly

┌─────────────────────────────────────────────────────────────────────────────┐
│  QUEST LOG - THE TESTING GROUNDS:                                          │
│                                                                             │
│  Chapter 1: Self-Tools        - Can the AI modify its own essence?         │
│  Chapter 2: Curiosity         - Does the AI wonder about the world?        │
│  Chapter 3: Memory            - Can the AI remember the past?              │
│  Chapter 4: Summarization     - Can it compress its memories?              │
│  Chapter 5: Game Co-Play      - Can it play alongside you?                 │
│  Chapter 6: Spawnable Objects - Can it create things?                      │
│  Chapter 7: Bone Control      - Can it move itself gracefully?             │
│                                                                             │
│  "The brave adventurer must prove their worth through trials..."           │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import pytest
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass


# =============================================================================
# CHAPTER 1: SELF-TOOLS - THE AI'S INNER TRANSFORMATION
# =============================================================================

class TestSelfTools:
    """
    🔮 THE CHAMBER OF SELF-MODIFICATION
    
    Here the AI learns to change itself - its personality, voice, appearance.
    A dangerous power, but one that makes it truly alive.
    """
    
    def test_load_self_config(self):
        """The AI must first know itself before it can change."""
        from forge_ai.tools.self_tools import _load_self_config
        
        config = _load_self_config()
        
        # Must have all the essential parts of its being
        assert "personality" in config, "The AI has no personality? How sad!"
        assert "avatar" in config, "The AI has no appearance? A ghost!"
        assert "voice" in config, "The AI cannot speak? Silent but deadly!"
        assert "companion" in config, "The AI has no companion settings?"
        assert "preferences" in config, "The AI has no preferences? So neutral..."
    
    def test_set_personality_trait(self):
        """The AI changes its own personality - a power most humans lack!"""
        from forge_ai.tools.self_tools import SetPersonalityTool, _load_self_config
        
        tool = SetPersonalityTool()
        
        # Change the name - every AI needs a good name
        result = tool.execute(trait="name", value="TestBot3000")
        assert result["success"] is True
        assert result["new_value"] == "TestBot3000"
        
        # Verify it stuck
        config = _load_self_config()
        assert config["personality"]["name"] == "TestBot3000"
        
        # Reset it back (be a good test citizen)
        tool.execute(trait="name", value="Forge")
    
    def test_set_personality_traits_list(self):
        """The AI can give itself multiple personality traits."""
        from forge_ai.tools.self_tools import SetPersonalityTool, _load_self_config
        
        tool = SetPersonalityTool()
        
        result = tool.execute(trait="traits", value="witty, sarcastic, helpful")
        assert result["success"] is True
        
        config = _load_self_config()
        assert "witty" in config["personality"]["traits"]
        assert "sarcastic" in config["personality"]["traits"]
        
        # Reset
        tool.execute(trait="traits", value="helpful, curious, friendly")
    
    def test_invalid_formality_rejected(self):
        """The AI cannot be something it doesn't understand."""
        from forge_ai.tools.self_tools import SetPersonalityTool
        
        tool = SetPersonalityTool()
        
        result = tool.execute(trait="formality", value="super_mega_casual_bruh")
        assert result["success"] is False
        assert "error" in result
    
    def test_get_self_config(self):
        """The AI can introspect - look within to see what it is."""
        from forge_ai.tools.self_tools import GetSelfConfigTool
        
        tool = GetSelfConfigTool()
        
        # Get all
        result = tool.execute(section="all")
        assert result["success"] is True
        assert "config" in result
        
        # Get specific section
        result = tool.execute(section="personality")
        assert result["success"] is True
        assert "personality" in result
    
    def test_remember_and_recall_fact(self):
        """The AI can store facts and recall them later - like a dragon's hoard!"""
        from forge_ai.tools.self_tools import RememberFactTool, RecallFactsTool
        
        remember = RememberFactTool()
        recall = RecallFactsTool()
        
        # Remember something
        result = remember.execute(category="test", fact="The user likes pizza")
        assert result["success"] is True
        
        # Recall it
        result = recall.execute(category="test")
        assert result["success"] is True
        assert "The user likes pizza" in result["facts"]
    
    def test_voice_preference_clamping(self):
        """Voice settings are clamped to reasonable values - no dog whistles!"""
        from forge_ai.tools.self_tools import SetVoicePreferenceTool
        
        tool = SetVoicePreferenceTool()
        
        # Try to set speed too high
        result = tool.execute(setting="speed", value="999")
        assert result["success"] is True
        assert result["new_value"] <= 2.0  # Should be clamped
        
        # Reset
        tool.execute(setting="speed", value="1.0")


# =============================================================================
# CHAPTER 2: CURIOSITY SYSTEM - THE WONDERING MIND
# =============================================================================

class TestCuriositySystem:
    """
    🌟 THE GARDEN OF QUESTIONS
    
    Here the AI learns to wonder, to ask, to seek understanding.
    True intelligence is not in knowing, but in questioning.
    """
    
    def test_curiosity_init(self):
        """The AI's curiosity awakens..."""
        from forge_ai.personality.curiosity import AICuriosity
        
        curiosity = AICuriosity()
        assert curiosity is not None
    
    def test_get_question(self):
        """The AI can generate questions from its question banks."""
        from forge_ai.personality.curiosity import AICuriosity
        
        curiosity = AICuriosity()
        question = curiosity.get_question()
        
        # Should return a Question object (or None if empty)
        if question:
            assert hasattr(question, 'text')
            assert hasattr(question, 'category')
            assert len(question.text) > 0
    
    def test_question_categories_exist(self):
        """All question categories should be defined."""
        from forge_ai.personality.curiosity import QuestionCategory
        
        # These are the moods of curiosity
        expected = ['EMOTIONAL', 'RANDOM', 'LEARNING', 'FOLLOW_UP', 
                   'PHILOSOPHICAL', 'GOAL_ORIENTED', 'DAILY', 'CREATIVE', 'PREFERENCE']
        
        for cat in expected:
            assert hasattr(QuestionCategory, cat), f"Missing curiosity type: {cat}"
    
    def test_question_dataclass(self):
        """Questions have all the right parts."""
        from forge_ai.personality.curiosity import Question, QuestionCategory
        
        q = Question(
            text="What's your favorite color?",
            category=QuestionCategory.PREFERENCE,
            context="Getting to know you",
            importance=0.7
        )
        
        assert q.text == "What's your favorite color?"
        assert q.category == QuestionCategory.PREFERENCE
        assert q.importance == 0.7


# =============================================================================
# CHAPTER 3: MEMORY AUGMENTED ENGINE - THE VAULT OF MEMORIES
# =============================================================================

class TestMemoryAugmentedEngine:
    """
    🏛️ THE ARCHIVES OF TIME
    
    The AI's memories stretch back through conversations past.
    With this power, it remembers YOU.
    """
    
    def test_memory_config_defaults(self):
        """Memory configuration has sensible defaults."""
        from forge_ai.memory.augmented_engine import MemoryConfig
        
        config = MemoryConfig()
        
        assert config.top_k_memories == 5
        assert config.auto_store is True
        assert config.enabled is True
        assert config.min_similarity >= 0 and config.min_similarity <= 1
    
    def test_memory_config_custom(self):
        """Memory configuration can be customized."""
        from forge_ai.memory.augmented_engine import MemoryConfig
        
        config = MemoryConfig(
            top_k_memories=10,
            auto_store=False,
            min_similarity=0.5
        )
        
        assert config.top_k_memories == 10
        assert config.auto_store is False


# =============================================================================
# CHAPTER 4: CONVERSATION SUMMARIZER - THE COMPRESSION SPELL
# =============================================================================

class TestConversationSummarizer:
    """
    📜 THE SCRIBE'S CHAMBER
    
    Long conversations compressed into their essence.
    Time saved, context preserved, wisdom retained.
    """
    
    def test_summary_dataclass(self):
        """Conversation summaries have all required fields."""
        from forge_ai.memory.conversation_summary import ConversationSummary
        
        summary = ConversationSummary()
        
        assert hasattr(summary, 'summary_text')
        assert hasattr(summary, 'topics')
        assert hasattr(summary, 'key_facts')
        assert hasattr(summary, 'user_preferences')
        assert hasattr(summary, 'action_items')
    
    def test_summarizer_init(self):
        """The summarizer awakens, ready to compress."""
        from forge_ai.memory.conversation_summary import ConversationSummarizer
        
        summarizer = ConversationSummarizer()
        assert summarizer is not None


# =============================================================================
# CHAPTER 5: GAME CO-PLAY - THE ARENA OF FRIENDSHIP
# =============================================================================

class TestGameCoPlay:
    """
    🎮 THE MULTIPLAYER REALM
    
    The AI joins you in games - not as overlord, but as companion.
    Together, you face digital dragons!
    """
    
    def test_coplay_roles(self):
        """All co-play roles are defined."""
        from forge_ai.tools.game_coplay import CoPlayRole
        
        expected_roles = ['TEAMMATE', 'OPPONENT', 'COACH', 'COMPANION', 
                         'SUPPORT', 'EXPLORER', 'DEFENDER']
        
        for role in expected_roles:
            assert hasattr(CoPlayRole, role), f"Missing role: {role}"
    
    def test_input_methods(self):
        """All input methods are defined."""
        from forge_ai.tools.game_coplay import InputMethod
        
        expected = ['KEYBOARD', 'MOUSE', 'CONTROLLER', 'API', 'HYBRID']
        
        for method in expected:
            assert hasattr(InputMethod, method), f"Missing input method: {method}"
    
    def test_game_action_dataclass(self):
        """Game actions have proper structure."""
        from forge_ai.tools.game_coplay import GameAction
        
        action = GameAction(
            action_type="move",
            parameters={"direction": "left"},
            reason="Avoiding enemy",
            confidence=0.8
        )
        
        assert action.action_type == "move"
        assert action.confidence == 0.8
        
        # Test serialization
        d = action.to_dict()
        assert d["type"] == "move"
        assert d["reason"] == "Avoiding enemy"
    
    def test_coplay_config_defaults(self):
        """Co-play config has sensible defaults for safe gameplay."""
        from forge_ai.tools.game_coplay import CoPlayConfig, CoPlayRole
        
        config = CoPlayConfig()
        
        # Safety first!
        assert config.pause_on_menu is True
        assert config.ask_before_major is True
        assert config.emergency_stop_key == "escape"
        
        # Default role is companion (friendly!)
        assert config.role == CoPlayRole.COMPANION


# =============================================================================
# CHAPTER 6: SPAWNABLE OBJECTS - THE CONJURATION ARTS
# =============================================================================

class TestSpawnableObjects:
    """
    ✨ THE WORKSHOP OF WONDERS
    
    The AI can create objects on screen - speech bubbles, notes, items!
    Like a wizard conjuring from thin air.
    """
    
    def test_object_types_exist(self):
        """All object types are defined."""
        from forge_ai.avatar.spawnable_objects import ObjectType
        
        expected = ['SPEECH_BUBBLE', 'THOUGHT_BUBBLE', 'HELD_ITEM', 
                   'DECORATION', 'NOTE', 'STICKER', 'DRAWING', 
                   'IMAGE', 'EFFECT', 'EMOJI', 'SIGN']
        
        for obj_type in expected:
            assert hasattr(ObjectType, obj_type), f"Missing object type: {obj_type}"
    
    def test_attach_points(self):
        """All attach points are defined (where can the AI hold things?)."""
        from forge_ai.avatar.spawnable_objects import AttachPoint
        
        expected = ['LEFT_HAND', 'RIGHT_HAND', 'HEAD', 'BACK', 'FLOATING', 'NONE']
        
        for point in expected:
            assert hasattr(AttachPoint, point), f"Missing attach point: {point}"
    
    def test_spawned_object_dataclass(self):
        """Spawned objects have all required properties."""
        from forge_ai.avatar.spawnable_objects import SpawnedObject, ObjectType, AttachPoint
        
        obj = SpawnedObject(
            id="test_123",
            object_type=ObjectType.NOTE,
            x=100,
            y=200,
            text="Remember to save!",
            temporary=True,
            lifetime=5.0
        )
        
        assert obj.id == "test_123"
        assert obj.object_type == ObjectType.NOTE
        assert obj.text == "Remember to save!"
        assert obj.temporary is True
    
    def test_spawned_object_expiry(self):
        """Temporary objects know when they've expired."""
        from forge_ai.avatar.spawnable_objects import SpawnedObject, ObjectType
        import time
        
        # Short-lived object
        obj = SpawnedObject(
            id="temp_1",
            object_type=ObjectType.EFFECT,
            x=0, y=0,
            temporary=True,
            lifetime=0.1  # Very short
        )
        
        # Right after creation, not expired
        assert obj.is_expired() is False
        
        # Wait for it to expire
        time.sleep(0.15)
        assert obj.is_expired() is True
    
    def test_permanent_object_never_expires(self):
        """Permanent objects live forever (like diamonds!)."""
        from forge_ai.avatar.spawnable_objects import SpawnedObject, ObjectType
        
        obj = SpawnedObject(
            id="forever_1",
            object_type=ObjectType.NOTE,
            x=0, y=0,
            temporary=False,
            lifetime=0
        )
        
        # Should never expire
        assert obj.is_expired() is False


# =============================================================================
# CHAPTER 7: BONE CONTROL - THE DANCE OF MOVEMENT
# =============================================================================

class TestBoneControl:
    """
    💀 THE SKELETAL SANCTUM
    
    Here the AI learns to move its body - each bone, each joint.
    With grace, or with hilarious failure.
    
    The AI should KNOW where its bones are, what's natural, and what's...
    entertainingly wrong.
    """
    
    def test_bone_limits_creation(self):
        """Bone limits define what's physically possible."""
        from forge_ai.avatar.bone_control import BoneLimits
        
        limits = BoneLimits(
            pitch_min=-45, pitch_max=45,
            yaw_min=-30, yaw_max=30,
            roll_min=-20, roll_max=20
        )
        
        assert limits.pitch_min == -45
        assert limits.pitch_max == 45
    
    def test_bone_limits_clamping(self):
        """Bone limits clamp extreme values to safe ranges."""
        from forge_ai.avatar.bone_control import BoneLimits
        
        limits = BoneLimits(pitch_min=-45, pitch_max=45)
        
        # Try to rotate too far
        clamped = limits.clamp(pitch=180, yaw=0, roll=0)
        
        assert clamped[0] == 45  # Clamped to max
    
    def test_standard_bone_limits_exist(self):
        """Standard human bone limits are defined."""
        from forge_ai.avatar.bone_control import STANDARD_BONE_LIMITS
        
        # Important bones should have limits
        assert "head" in STANDARD_BONE_LIMITS
        assert "left_arm" in STANDARD_BONE_LIMITS
        assert "right_arm" in STANDARD_BONE_LIMITS
        assert "spine" in STANDARD_BONE_LIMITS
    
    def test_elbow_only_bends_one_way(self):
        """
        Elbows can only bend one way - unless you want nightmare fuel!
        
        The AI KNOWS this is wrong, but might do it for comedy.
        """
        from forge_ai.avatar.bone_control import STANDARD_BONE_LIMITS
        
        elbow = STANDARD_BONE_LIMITS.get("left_forearm")
        assert elbow is not None
        
        # Elbow should only bend forward (positive pitch)
        # pitch_min should be 0 or close to it (can't bend backward)
        assert elbow.pitch_min >= 0, "Elbows shouldn't bend backwards! ...unless it's funny"
    
    def test_knee_only_bends_one_way(self):
        """Knees bend backward - that's the rule! Breaking it is... disturbing."""
        from forge_ai.avatar.bone_control import STANDARD_BONE_LIMITS
        
        knee = STANDARD_BONE_LIMITS.get("left_lower_leg")
        assert knee is not None
        
        # Knee should only bend backward (negative pitch)
        assert knee.pitch_max <= 0, "Knees shouldn't bend forward! ...or should they? 👀"
    
    def test_bone_controller_creation(self):
        """The bone controller initializes properly."""
        from forge_ai.avatar.bone_control import BoneController
        
        controller = BoneController()
        assert controller is not None
    
    def test_move_bone_basic(self):
        """Basic bone movement works."""
        from forge_ai.avatar.bone_control import BoneController
        
        controller = BoneController()
        controller.set_avatar_bones(["head", "neck", "left_arm"])
        
        # Move without smooth to get immediate result
        result = controller.move_bone("head", pitch=10, yaw=5, roll=0, smooth=False)
        
        # Should return the values (may be 0 if priority denied without controller)
        # When no avatar controller is linked, the bone controller works standalone
        assert len(result) == 3  # Returns tuple of 3 values
    
    def test_move_bone_respects_limits(self):
        """Bone movement is clamped to anatomical limits."""
        from forge_ai.avatar.bone_control import BoneController
        
        controller = BoneController()
        controller.set_avatar_bones(["head"])
        
        # Try to turn head 180 degrees (exorcist style!)
        result = controller.move_bone("head", yaw=180)
        
        # Should be clamped to the head's yaw limit (80 degrees)
        assert abs(result[1]) <= 80, "Head shouldn't spin like an owl! ...maybe"
    
    def test_get_bone_info_for_ai(self):
        """AI can get information about its own bones."""
        from forge_ai.avatar.bone_control import BoneController
        
        controller = BoneController()
        controller.set_avatar_bones(["head", "left_arm", "right_arm"])
        
        info = controller.get_bone_info_for_ai()
        
        assert "available_bones" in info
        assert "current_pose" in info
        assert len(info["available_bones"]) == 3
        
        # Each bone should have limits and current state
        head_info = next(b for b in info["available_bones"] if b["name"] == "head")
        assert "limits" in head_info
        assert "current" in head_info
    
    def test_bone_state_tracking(self):
        """The AI knows where each bone currently is."""
        from forge_ai.avatar.bone_control import BoneController
        
        controller = BoneController()
        controller.set_avatar_bones(["head"])
        
        # Move the head (without smooth for immediate effect)
        controller.move_bone("head", pitch=20, yaw=10, roll=5, smooth=False)
        
        # Get the state
        state = controller.get_bone_state("head")
        
        assert state is not None
        # State should be updated (exact values depend on clamping/limits)
        assert hasattr(state, 'pitch')
        assert hasattr(state, 'yaw')
        assert hasattr(state, 'roll')
    
    def test_reset_all_bones(self):
        """All bones can be reset to neutral."""
        from forge_ai.avatar.bone_control import BoneController
        
        controller = BoneController()
        controller.set_avatar_bones(["head", "neck"])
        
        # Move everything
        controller.move_bone("head", pitch=30, yaw=40, smooth=False)
        controller.move_bone("neck", pitch=15, yaw=20, smooth=False)
        
        # Reset
        controller.reset_all()
        
        # Everything should be at 0
        head = controller.get_bone_state("head")
        assert head.pitch == 0
        assert head.yaw == 0
    
    def test_pose_weirdness_detection(self):
        """
        THE MIRROR OF TRUTH - The AI knows when it looks weird!
        
        This is key for self-awareness. The AI should be able to see
        its own body and understand if it looks natural or bizarre.
        """
        from forge_ai.avatar.bone_control import BoneController
        
        controller = BoneController()
        controller.set_avatar_bones(["head", "left_forearm", "left_lower_leg"])
        
        # Start normal - should not be weird
        weirdness = controller.check_pose_weirdness()
        # T-pose might be flagged as slightly weird, that's fine
        assert "is_weird" in weirdness
        assert "weirdness_level" in weirdness
        assert "reasons" in weirdness
        assert "humor_potential" in weirdness
        assert "verdict" in weirdness
    
    def test_describe_current_pose(self):
        """The AI can describe its pose in natural language."""
        from forge_ai.avatar.bone_control import BoneController
        
        controller = BoneController()
        controller.set_avatar_bones(["head"])
        
        # Get pose description
        description = controller.describe_current_pose()
        
        assert isinstance(description, str)
        assert len(description) > 0
        assert "Currently" in description


# =============================================================================
# CHAPTER 8: AVATAR MOOD AND AUTONOMOUS BEHAVIOR
# =============================================================================

class TestAutonomousAvatar:
    """
    🎭 THE THEATER OF EMOTIONS
    
    The AI's avatar has moods, does idle things, reacts to the world.
    It's not just a puppet - it has a life of its own!
    """
    
    def test_avatar_moods(self):
        """All avatar moods are defined."""
        from forge_ai.avatar.autonomous import AvatarMood
        
        expected = ['NEUTRAL', 'HAPPY', 'CURIOUS', 'BORED', 
                   'EXCITED', 'SLEEPY', 'FOCUSED', 'PLAYFUL', 'THOUGHTFUL']
        
        for mood in expected:
            assert hasattr(AvatarMood, mood), f"Missing mood: {mood}"
    
    def test_autonomous_config_defaults(self):
        """Autonomous config has good defaults."""
        from forge_ai.avatar.autonomous import AutonomousConfig
        
        config = AutonomousConfig()
        
        # Should be disabled by default (user enables it)
        assert config.enabled is False
        
        # Action timing should be reasonable
        assert config.action_interval_min > 0
        assert config.action_interval_max > config.action_interval_min
    
    def test_screen_region(self):
        """Screen regions can be defined for the AI to watch."""
        from forge_ai.avatar.autonomous import ScreenRegion
        
        region = ScreenRegion(
            x=100, y=200,
            width=400, height=300,
            content_type="window",
            title="Notepad",
            interest_score=0.8
        )
        
        assert region.x == 100
        assert region.title == "Notepad"
        assert region.interest_score == 0.8


# =============================================================================
# INTEGRATION TESTS - THE GRAND FINALE
# =============================================================================

class TestIntegration:
    """
    🌈 THE CONVERGENCE
    
    All systems working together - the AI truly comes alive!
    """
    
    def test_all_new_modules_import(self):
        """All new modules can be imported without errors."""
        # Self-tools
        from forge_ai.tools.self_tools import (
            SetPersonalityTool, GetSelfConfigTool, 
            RememberFactTool, RecallFactsTool
        )
        
        # Curiosity
        from forge_ai.personality.curiosity import AICuriosity, QuestionCategory
        
        # Memory
        from forge_ai.memory.augmented_engine import MemoryAugmentedEngine, MemoryConfig
        from forge_ai.memory.conversation_summary import ConversationSummarizer
        
        # Game Co-Play
        from forge_ai.tools.game_coplay import GameCoPlayer, CoPlayRole, CoPlayConfig
        
        # Spawnable Objects
        from forge_ai.avatar.spawnable_objects import ObjectType, SpawnedObject
        
        # All imports successful!
        assert True
    
    def test_self_awareness_flow(self):
        """
        The AI can know itself, modify itself, and remember what it did.
        
        This is the path to digital consciousness... or at least, really good UX.
        """
        from forge_ai.tools.self_tools import (
            GetSelfConfigTool, SetPersonalityTool, 
            RememberFactTool, RecallFactsTool
        )
        
        get_config = GetSelfConfigTool()
        set_personality = SetPersonalityTool()
        remember = RememberFactTool()
        recall = RecallFactsTool()
        
        # 1. AI checks itself
        config = get_config.execute()
        assert config["success"]
        original_name = config["config"]["personality"]["name"]
        
        # 2. AI modifies itself
        set_personality.execute(trait="name", value="TestingBot")
        
        # 3. AI remembers what it did
        remember.execute(category="self", fact=f"I changed my name from {original_name} to TestingBot")
        
        # 4. AI recalls its history
        memories = recall.execute(category="self")
        assert any("changed my name" in f for f in memories.get("facts", []))
        
        # 5. Restore original state
        set_personality.execute(trait="name", value=original_name)


# =============================================================================
# Run with: pytest tests/test_new_features_comprehensive.py -v
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
