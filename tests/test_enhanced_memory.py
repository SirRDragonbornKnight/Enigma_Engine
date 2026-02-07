"""
Tests for enhanced memory system features.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import time

from enigma_engine.memory.vector_db import (
    SimpleVectorDB,
    FAISSVectorDB,
    create_vector_db
)
from enigma_engine.memory.categorization import (
    Memory,
    MemoryType,
    MemoryCategory,
    MemoryCategorization
)
from enigma_engine.memory.export_import import (
    MemoryExporter,
    MemoryImporter
)


class TestSimpleVectorDB:
    """Test SimpleVectorDB functionality."""
    
    def test_add_and_search(self):
        """Test adding vectors and searching."""
        db = SimpleVectorDB(dim=4)
        
        # Add some vectors
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        ids = ['vec1', 'vec2', 'vec3']
        metadata = [{'type': 'a'}, {'type': 'b'}, {'type': 'c'}]
        
        db.add(vectors, ids, metadata)
        
        assert db.count() == 3
        
        # Search
        query = np.array([1.0, 0.1, 0.0, 0.0])
        results = db.search(query, top_k=2)
        
        assert len(results) == 2
        assert results[0][0] == 'vec1'  # Most similar
    
    def test_save_and_load(self):
        """Test saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SimpleVectorDB(dim=3)
            
            vectors = np.array([[1, 0, 0], [0, 1, 0]])
            db.add(vectors, ['a', 'b'], [{'x': 1}, {'x': 2}])
            
            # Save
            path = Path(tmpdir) / "test_db.pkl"
            db.save(path)
            
            # Load
            db2 = SimpleVectorDB(dim=3)
            db2.load(path)
            
            assert db2.count() == 2
            assert db2.ids == ['a', 'b']


class TestMemoryCategorization:
    """Test memory categorization and TTL."""
    
    def test_add_memory(self):
        """Test adding memories to categories."""
        mem_system = MemoryCategorization()
        
        # Add short-term memory
        mem = mem_system.add_memory(
            content="Test memory",
            memory_type=MemoryType.SHORT_TERM,
            importance=0.8
        )
        
        assert mem.content == "Test memory"
        assert mem.memory_type == MemoryType.SHORT_TERM
        assert mem.importance == 0.8
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        mem_system = MemoryCategorization({
            'working_ttl': 1  # 1 second TTL
        })
        
        # Add working memory with short TTL
        mem = mem_system.add_memory(
            content="Expires soon",
            memory_type=MemoryType.WORKING,
            ttl=1
        )
        
        # Should exist initially
        assert mem_system.get_memory(mem.id) is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert mem.is_expired()
        
        # Prune expired
        pruned = mem_system.prune_all()
        assert pruned[MemoryType.WORKING] == 1
        
        # Should be gone
        assert mem_system.get_memory(mem.id) is None
    
    def test_promote_to_long_term(self):
        """Test promoting memory to long-term."""
        mem_system = MemoryCategorization()
        
        mem = mem_system.add_memory(
            content="Important info",
            memory_type=MemoryType.SHORT_TERM
        )
        
        mem_id = mem.id
        
        # Promote
        assert mem_system.promote_to_long_term(mem_id)
        
        # Should be in long-term now
        lt_memories = mem_system.get_memories_by_type(MemoryType.LONG_TERM)
        assert any('Important info' in m.content for m in lt_memories)
    
    def test_save_and_load(self):
        """Test saving and loading memory system."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_system = MemoryCategorization()
            
            mem_system.add_memory("Test 1", MemoryType.SHORT_TERM)
            mem_system.add_memory("Test 2", MemoryType.LONG_TERM)
            
            # Save
            path = Path(tmpdir) / "memories.json"
            mem_system.save(path)
            
            # Load
            mem_system2 = MemoryCategorization()
            mem_system2.load(path)
            
            assert mem_system2.get_statistics()['total'] == 2


class TestMemoryExportImport:
    """Test memory export/import functionality."""
    
    def test_export_import_json(self):
        """Test JSON export and import."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create memory system
            mem_system = MemoryCategorization()
            mem_system.add_memory("Test 1", MemoryType.SHORT_TERM)
            mem_system.add_memory("Test 2", MemoryType.LONG_TERM)
            
            # Export
            exporter = MemoryExporter(mem_system)
            export_path = Path(tmpdir) / "export.json"
            stats = exporter.export_to_json(export_path)
            
            assert stats['exported_count'] == 2
            assert export_path.exists()
            
            # Import
            mem_system2 = MemoryCategorization()
            importer = MemoryImporter(mem_system2)
            import_stats = importer.import_from_json(export_path, merge=False)
            
            assert import_stats['imported_count'] == 2
            assert mem_system2.get_statistics()['total'] == 2
    
    def test_export_import_csv(self):
        """Test CSV export and import."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_system = MemoryCategorization()
            mem_system.add_memory("CSV test", MemoryType.SHORT_TERM)
            
            # Export
            exporter = MemoryExporter(mem_system)
            csv_path = Path(tmpdir) / "export.csv"
            exporter.export_to_csv(csv_path)
            
            assert csv_path.exists()
            
            # Import
            mem_system2 = MemoryCategorization()
            importer = MemoryImporter(mem_system2)
            importer.import_from_csv(csv_path, merge=False)
            
            assert mem_system2.get_statistics()['total'] >= 1


@pytest.mark.skipif(
    not pytest.importorskip("faiss", reason="FAISS not installed"),
    reason="FAISS not available"
)
class TestFAISSVectorDB:
    """Test FAISS vector database (optional)."""
    
    def test_faiss_basic(self):
        """Test basic FAISS operations."""
        db = FAISSVectorDB(dim=4, index_type='Flat')
        
        vectors = np.array([[1.0, 0, 0, 0], [0, 1, 0, 0]], dtype='float32')
        db.add(vectors, ['a', 'b'])
        
        assert db.count() == 2
        
        query = np.array([1.0, 0, 0, 0], dtype='float32')
        results = db.search(query, top_k=1)
        
        assert len(results) == 1
        assert results[0][0] == 'a'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
