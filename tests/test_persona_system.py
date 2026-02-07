"""
Tests for the AI Persona System.
"""
import pytest
import tempfile
import json
from pathlib import Path

from enigma_engine.core.persona import (
    AIPersona,
    PersonaManager,
    get_persona_manager
)


class TestAIPersona:
    """Test AIPersona dataclass."""
    
    def test_persona_creation(self):
        """Test creating a basic persona."""
        persona = AIPersona(
            id="test_persona",
            name="Test AI",
            created_at="2024-01-01T00:00:00",
            personality_traits={"humor_level": 0.5, "formality": 0.6},
            system_prompt="Test prompt"
        )
        
        assert persona.id == "test_persona"
        assert persona.name == "Test AI"
        assert persona.personality_traits["humor_level"] == 0.5
        assert persona.system_prompt == "Test prompt"
    
    def test_persona_to_dict(self):
        """Test converting persona to dictionary."""
        persona = AIPersona(
            id="test",
            name="Test",
            created_at="2024-01-01T00:00:00",
            personality_traits={"humor_level": 0.5}
        )
        
        data = persona.to_dict()
        assert isinstance(data, dict)
        assert data["id"] == "test"
        assert data["name"] == "Test"
        assert data["personality_traits"]["humor_level"] == 0.5
    
    def test_persona_from_dict(self):
        """Test creating persona from dictionary."""
        data = {
            "id": "test",
            "name": "Test",
            "created_at": "2024-01-01T00:00:00",
            "personality_traits": {"humor_level": 0.5},
            "voice_profile_id": "default",
            "avatar_preset_id": "default",
            "system_prompt": "",
            "response_style": "balanced",
            "knowledge_domains": [],
            "memories": [],
            "learning_data_path": "",
            "model_weights_path": "",
            "catchphrases": [],
            "preferences": {},
            "version": "1.0",
            "last_modified": "2024-01-01T00:00:00",
            "description": "",
            "tags": []
        }
        
        persona = AIPersona.from_dict(data)
        assert persona.id == "test"
        assert persona.name == "Test"
        assert persona.personality_traits["humor_level"] == 0.5


class TestPersonaManager:
    """Test PersonaManager class."""
    
    @pytest.fixture
    def temp_manager(self):
        """Create a persona manager with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PersonaManager(personas_dir=Path(tmpdir))
            yield manager
    
    def test_manager_initialization(self, temp_manager):
        """Test that manager initializes with default persona."""
        assert temp_manager.personas_dir.exists()
        assert temp_manager.templates_dir.exists()
        assert temp_manager.current_persona_id is not None
        
        # Default persona should exist
        default_persona = temp_manager.get_current_persona()
        assert default_persona is not None
        assert default_persona.id == "default"
    
    def test_save_and_load_persona(self, temp_manager):
        """Test saving and loading a persona."""
        persona = AIPersona(
            id="test_save",
            name="Test Save",
            created_at="2024-01-01T00:00:00",
            personality_traits={"humor_level": 0.7},
            system_prompt="Test save prompt"
        )
        
        # Save
        temp_manager.save_persona(persona)
        
        # Load
        loaded = temp_manager.load_persona("test_save")
        assert loaded is not None
        assert loaded.id == "test_save"
        assert loaded.name == "Test Save"
        assert loaded.personality_traits["humor_level"] == 0.7
    
    def test_copy_persona(self, temp_manager):
        """Test copying a persona."""
        original = temp_manager.get_current_persona()
        
        # Copy
        copy = temp_manager.copy_persona(original.id, "Copy of Default")
        
        assert copy is not None
        assert copy.id != original.id
        assert copy.name == "Copy of Default"
        assert copy.personality_traits == original.personality_traits
        assert copy.system_prompt == original.system_prompt
    
    def test_export_persona(self, temp_manager):
        """Test exporting a persona to file."""
        persona = temp_manager.get_current_persona()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "exported.forge-ai"
            result = temp_manager.export_persona(persona.id, export_path)
            
            assert result is not None
            assert result.exists()
            
            # Verify content
            with open(result, 'r') as f:
                data = json.load(f)
                assert data["name"] == persona.name
                assert "exported_at" in data
    
    def test_import_persona(self, temp_manager):
        """Test importing a persona from file."""
        # Create export data
        export_data = {
            "id": "imported",
            "name": "Imported AI",
            "created_at": "2024-01-01T00:00:00",
            "personality_traits": {"humor_level": 0.8},
            "voice_profile_id": "default",
            "avatar_preset_id": "default",
            "system_prompt": "Imported prompt",
            "response_style": "casual",
            "knowledge_domains": ["test"],
            "memories": [],
            "learning_data_path": "",
            "model_weights_path": "",
            "catchphrases": ["Hello!"],
            "preferences": {},
            "version": "1.0",
            "last_modified": "2024-01-01T00:00:00",
            "description": "Test import",
            "tags": ["imported"],
            "exported_at": "2024-01-01T00:00:00",
            "exported_from": "Enigma AI Engine",
            "format_version": "1.0"
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            import_file = Path(tmpdir) / "import.forge-ai"
            with open(import_file, 'w') as f:
                json.dump(export_data, f)
            
            # Import
            imported = temp_manager.import_persona(import_file)
            
            assert imported is not None
            assert imported.name == "Imported AI"
            assert imported.personality_traits["humor_level"] == 0.8
            assert imported.catchphrases == ["Hello!"]
    
    def test_delete_persona(self, temp_manager):
        """Test deleting a persona."""
        # Create a test persona
        persona = AIPersona(
            id="to_delete",
            name="Delete Me",
            created_at="2024-01-01T00:00:00",
            personality_traits={}
        )
        temp_manager.save_persona(persona)
        
        # Verify it exists
        assert temp_manager.persona_exists("to_delete")
        
        # Delete it
        result = temp_manager.delete_persona("to_delete")
        assert result is True
        assert not temp_manager.persona_exists("to_delete")
    
    def test_cannot_delete_default(self, temp_manager):
        """Test that default persona cannot be deleted."""
        result = temp_manager.delete_persona("default")
        assert result is False
        assert temp_manager.persona_exists("default")
    
    def test_list_personas(self, temp_manager):
        """Test listing all personas."""
        # Create some test personas
        for i in range(3):
            persona = AIPersona(
                id=f"test_{i}",
                name=f"Test {i}",
                created_at="2024-01-01T00:00:00",
                personality_traits={}
            )
            temp_manager.save_persona(persona)
        
        personas = temp_manager.list_personas()
        assert len(personas) >= 4  # At least default + 3 test personas
        
        # Check structure
        for p in personas:
            assert "id" in p
            assert "name" in p
            assert "description" in p
    
    def test_set_current_persona(self, temp_manager):
        """Test setting the current persona."""
        # Create a test persona
        persona = AIPersona(
            id="new_current",
            name="New Current",
            created_at="2024-01-01T00:00:00",
            personality_traits={}
        )
        temp_manager.save_persona(persona)
        
        # Set as current
        result = temp_manager.set_current_persona("new_current")
        assert result is True
        assert temp_manager.current_persona_id == "new_current"
        
        # Get current should return it
        current = temp_manager.get_current_persona()
        assert current.id == "new_current"
    
    def test_merge_personas(self, temp_manager):
        """Test merging two personas."""
        # Create two personas with different traits
        persona1 = AIPersona(
            id="merge1",
            name="Persona 1",
            created_at="2024-01-01T00:00:00",
            personality_traits={"humor_level": 0.2, "formality": 0.8},
            knowledge_domains=["domain1"]
        )
        persona2 = AIPersona(
            id="merge2",
            name="Persona 2",
            created_at="2024-01-01T00:00:00",
            personality_traits={"humor_level": 0.8, "formality": 0.2},
            knowledge_domains=["domain2"]
        )
        
        temp_manager.save_persona(persona1)
        temp_manager.save_persona(persona2)
        
        # Merge them
        merged = temp_manager.merge_personas("merge1", "merge2", "Merged AI")
        
        assert merged is not None
        assert merged.name == "Merged AI"
        
        # Traits should be averaged
        assert 0.4 <= merged.personality_traits["humor_level"] <= 0.6
        
        # Domains should be combined
        assert "domain1" in merged.knowledge_domains
        assert "domain2" in merged.knowledge_domains


class TestPersonaIntegration:
    """Test integration with personality system."""
    
    def test_integrate_with_personality(self):
        """Test creating AIPersonality from persona."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PersonaManager(personas_dir=Path(tmpdir))
            persona = manager.get_current_persona()
            
            # Update persona traits
            persona.personality_traits = {
                "humor_level": 0.7,
                "formality": 0.3,
                "verbosity": 0.6,
                "curiosity": 0.8,
                "empathy": 0.7,
                "creativity": 0.8,
                "confidence": 0.6,
                "playfulness": 0.7
            }
            persona.catchphrases = ["Hello!", "How can I help?"]
            manager.save_persona(persona)
            
            # Convert to AIPersonality
            personality = manager.integrate_with_personality(persona)
            
            assert personality is not None
            assert personality.traits.humor_level == 0.7
            assert personality.traits.formality == 0.3
            assert "Hello!" in personality.catchphrases


def test_get_persona_manager():
    """Test singleton persona manager."""
    manager1 = get_persona_manager()
    manager2 = get_persona_manager()
    
    assert manager1 is manager2  # Should be same instance
