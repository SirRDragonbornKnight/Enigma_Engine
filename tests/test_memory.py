#!/usr/bin/env python3
"""
Tests for the AI Tester memory system.

Run with: pytest tests/test_memory.py -v
"""
import pytest
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConversationManager:
    """Tests for the conversation manager."""
    
    def test_manager_init(self):
        """Test conversation manager initialization."""
        from ai_tester.memory.manager import ConversationManager
        manager = ConversationManager()
        assert manager is not None
    
    def test_save_conversation(self):
        """Test saving conversation."""
        from ai_tester.memory.manager import ConversationManager
        manager = ConversationManager()
        
        messages = [
            {"role": "user", "text": "Hello"},
            {"role": "assistant", "text": "Hi!"},
        ]
        
        result = manager.save_conversation("test_conv", messages)
        assert result is not None
    
    def test_load_conversation(self):
        """Test loading conversation."""
        from ai_tester.memory.manager import ConversationManager
        manager = ConversationManager()
        
        messages = [
            {"role": "user", "text": "Test"},
        ]
        
        manager.save_conversation("test_load", messages)
        loaded = manager.load_conversation("test_load")
        
        assert loaded is not None
        assert "messages" in loaded
    
    def test_list_conversations(self):
        """Test listing conversations."""
        from ai_tester.memory.manager import ConversationManager
        manager = ConversationManager()
        
        # Save a conversation
        manager.save_conversation("test_list", [{"role": "user", "text": "Test"}])
        
        conversations = manager.list_conversations()
        assert isinstance(conversations, list)


class TestSimpleVectorDB:
    """Tests for vector database."""
    
    def test_db_init(self):
        """Test vector DB initialization."""
        from ai_tester.memory.vector_utils import SimpleVectorDB
        db = SimpleVectorDB(dim=64)
        assert db is not None
        assert db.dim == 64
    
    def test_add_and_search(self):
        """Test adding and searching vectors."""
        from ai_tester.memory.vector_utils import SimpleVectorDB
        db = SimpleVectorDB(dim=3)
        
        # Add some vectors
        db.add([1.0, 0.0, 0.0], "vec1")
        db.add([0.0, 1.0, 0.0], "vec2")
        db.add([0.9, 0.1, 0.0], "vec3")  # Similar to vec1
        
        # Search for similar to [1, 0, 0]
        results = db.search([1.0, 0.0, 0.0], topk=2)
        assert len(results) == 2
        # vec1 should be most similar (exact match)
        assert results[0][0] == "vec1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
