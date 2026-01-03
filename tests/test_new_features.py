"""
Basic tests for new comprehensive features.

Run with: pytest tests/test_new_features.py -v
"""

import pytest
from pathlib import Path
import tempfile
import shutil


class TestPersonality:
    """Test AI personality system."""
    
    def test_import(self):
        """Test that personality module can be imported."""
        from enigma.core.personality import AIPersonality, PersonalityTraits
        assert AIPersonality is not None
        assert PersonalityTraits is not None
    
    def test_personality_creation(self):
        """Test creating a personality."""
        from enigma.core.personality import AIPersonality
        
        personality = AIPersonality("test_model")
        assert personality.model_name == "test_model"
        assert 0.0 <= personality.traits.humor_level <= 1.0
        assert personality.mood == "neutral"
    
    def test_personality_presets(self):
        """Test personality presets."""
        from enigma.core.personality import AIPersonality
        
        presets = ["professional", "friendly", "creative", "analytical"]
        for preset in presets:
            personality = AIPersonality.create_preset("test", preset)
            assert personality is not None
            # Verify traits are in valid range
            for trait in ['humor_level', 'formality', 'verbosity']:
                value = getattr(personality.traits, trait)
                assert 0.0 <= value <= 1.0
    
    def test_personality_evolution(self):
        """Test personality evolution."""
        from enigma.core.personality import AIPersonality
        
        personality = AIPersonality("test")
        initial_humor = personality.traits.humor_level
        
        # Positive feedback should reinforce traits
        personality.evolve_from_interaction(
            user_input="Tell me a joke",
            ai_response="Why did the AI cross the road? 😄",
            feedback="positive"
        )
        
        # Conversation count should increase
        assert personality.conversation_count == 1
    
    def test_personality_save_load(self):
        """Test saving and loading personality."""
        from enigma.core.personality import AIPersonality
        
        with tempfile.TemporaryDirectory() as tmpdir:
            personality = AIPersonality("test")
            personality.traits.humor_level = 0.8
            personality.add_opinion("Python", "I love it!")
            
            # Save
            save_path = personality.save(Path(tmpdir))
            assert save_path.exists()
            
            # Load
            loaded = AIPersonality("test")
            loaded.load(Path(tmpdir))
            assert loaded.traits.humor_level == 0.8
            assert "Python" in loaded.opinions


class TestVoiceGenerator:
    """Test voice generation system."""
    
    def test_import(self):
        """Test that voice generator can be imported."""
        from enigma.voice.voice_generator import AIVoiceGenerator
        assert AIVoiceGenerator is not None
    
    def test_voice_from_personality(self):
        """Test generating voice from personality."""
        from enigma.core.personality import AIPersonality
        from enigma.voice.voice_generator import AIVoiceGenerator
        
        personality = AIPersonality("test")
        personality.traits.confidence = 0.9
        personality.traits.playfulness = 0.2
        
        generator = AIVoiceGenerator()
        voice = generator.generate_voice_from_personality(personality)
        
        assert voice is not None
        assert voice.name == "test_ai_voice"
        # High confidence should result in lower pitch
        assert voice.pitch < 1.1


class TestInstanceManager:
    """Test multi-instance support."""
    
    def test_import(self):
        """Test that instance manager can be imported."""
        from enigma.core.instance_manager import InstanceManager
        assert InstanceManager is not None
    
    def test_instance_creation(self):
        """Test creating an instance."""
        from enigma.core.instance_manager import InstanceManager
        
        manager = InstanceManager()
        assert manager.instance_id is not None
        assert len(manager.instance_id) == 8
        manager.shutdown()
    
    def test_model_locking(self):
        """Test model locking mechanism."""
        from enigma.core.instance_manager import InstanceManager
        
        manager1 = InstanceManager("test1")
        manager2 = InstanceManager("test2")
        
        try:
            # First instance should get the lock
            assert manager1.acquire_model_lock("test_model")
            
            # Second instance should not get it (no timeout)
            assert not manager2.acquire_model_lock("test_model", timeout=0.1)
            
            # Release and second should get it
            manager1.release_model_lock("test_model")
            assert manager2.acquire_model_lock("test_model")
        finally:
            manager1.shutdown()
            manager2.shutdown()


class TestHuggingFaceAddons:
    """Test HuggingFace integration through GUI tabs."""
    
    def test_import(self):
        """Test that HuggingFace tab providers can be imported."""
        # HuggingFace integration is now through GUI tabs
        try:
            from enigma.gui.tabs.image_tab import StableDiffusionLocal
            from enigma.gui.tabs.embeddings_tab import LocalEmbedding
            assert StableDiffusionLocal is not None
            assert LocalEmbedding is not None
        except ImportError as e:
            pytest.skip(f"GUI tabs not available: {e}")
    
    def test_addon_creation(self):
        """Test that tab-based providers can be instantiated."""
        try:
            from enigma.gui.tabs.embeddings_tab import LocalEmbedding
            # LocalEmbedding can be created without loading the model
            embedder = LocalEmbedding()
            assert embedder is not None
            assert hasattr(embedder, 'load')
            assert hasattr(embedder, 'embed')
        except ImportError as e:
            pytest.skip(f"Embeddings tab not available: {e}")
    
    def test_builtin_registration(self):
        """Test that generation tabs are available."""
        # Test that the new tab-based providers exist
        try:
            from enigma.gui.tabs.image_tab import StableDiffusionLocal, OpenAIImage
            from enigma.gui.tabs.code_tab import EnigmaCode, OpenAICode
            from enigma.gui.tabs.embeddings_tab import LocalEmbedding, OpenAIEmbedding
            assert StableDiffusionLocal is not None
            assert OpenAIImage is not None
            assert EnigmaCode is not None
            assert OpenAICode is not None
            assert LocalEmbedding is not None
            assert OpenAIEmbedding is not None
        except ImportError:
            pytest.skip("GUI tabs not fully available")


class TestWebDashboard:
    """Test web dashboard."""
    
    def test_import(self):
        """Test that web app can be imported."""
        try:
            from enigma.web.app import app
            assert app is not None
        except ImportError:
            pytest.skip("Flask not installed")
    
    def test_routes_exist(self):
        """Test that main routes are defined."""
        try:
            from enigma.web.app import app
            
            # Get all route rules
            routes = [str(rule) for rule in app.url_map.iter_rules()]
            
            # Check main routes exist
            assert '/' in routes
            assert '/chat' in routes
            assert '/train' in routes
            assert '/settings' in routes
            assert '/api/status' in routes
        except ImportError:
            pytest.skip("Flask not installed")


class TestMobileAPI:
    """Test mobile API."""
    
    def test_import(self):
        """Test that mobile API can be imported."""
        try:
            from enigma.mobile import mobile_app, MobileAPI
            # mobile_app may be None if Flask not available
            assert MobileAPI is not None
        except ImportError:
            pytest.skip("Flask not installed")
    
    def test_endpoints_exist(self):
        """Test that API endpoints are defined."""
        try:
            from enigma.mobile import MobileAPI
            
            api = MobileAPI(port=5099)  # Use unusual port for test
            
            # Get all route rules
            routes = [str(rule) for rule in api.app.url_map.iter_rules()]
            
            # Check API endpoints exist (comms/mobile_api uses these routes)
            assert '/chat' in routes
            assert '/status' in routes
            assert '/voice/input' in routes
            assert '/voice/output' in routes
        except ImportError:
            pytest.skip("Flask not installed")


class TestTrainingData:
    """Test that training data files exist."""
    
    def test_personality_development_exists(self):
        """Test that personality development training data exists."""
        # Training data is in data/ folder at project root
        data_dir = Path(__file__).parent.parent / "data"
        file_path = data_dir / "personality_development.txt"
        assert file_path.exists(), f"File not found: {file_path}"
        assert file_path.stat().st_size > 500  # Should have content
    
    def test_self_awareness_exists(self):
        """Test that self-awareness training data exists."""
        data_dir = Path(__file__).parent.parent / "data"
        file_path = data_dir / "self_awareness_training.txt"
        assert file_path.exists(), f"File not found: {file_path}"
        assert file_path.stat().st_size > 500
    
    def test_combined_action_exists(self):
        """Test that combined action training data exists."""
        data_dir = Path(__file__).parent.parent / "data"
        file_path = data_dir / "combined_action_training.txt"
        assert file_path.exists(), f"File not found: {file_path}"
        assert file_path.stat().st_size > 500


class TestDocumentation:
    """Test that documentation exists."""
    
    def test_docs_exist(self):
        """Test that all documentation files exist."""
        docs = [
            "PERSONALITY.md",
            "VOICE_CUSTOMIZATION.md",
            "HUGGINGFACE.md",
            "MULTI_INSTANCE.md",
            "WEB_MOBILE.md"
        ]
        
        for doc in docs:
            doc_path = Path("docs") / doc
            assert doc_path.exists(), f"{doc} not found"
            assert doc_path.stat().st_size > 500, f"{doc} is too small"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
