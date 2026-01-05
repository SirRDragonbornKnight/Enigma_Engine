#!/usr/bin/env python3
"""
Comprehensive tests for the complete memory system overhaul.

Tests all new features: RAG, embeddings, consolidation, async, search, 
deduplication, compression, visualization, analytics, encryption, and backup.

Run with: pytest tests/test_memory_complete.py -v
"""
import pytest
import sys
import json
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMemoryDatabase:
    """Tests for the refactored MemoryDatabase class."""
    
    def test_database_init(self):
        """Test database initialization."""
        from enigma.memory.memory_db import MemoryDatabase
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = MemoryDatabase(db_path)
            
            assert db is not None
            assert db.db_path == db_path
            assert db_path.exists()
    
    def test_add_and_get(self):
        """Test adding and retrieving memories."""
        from enigma.memory.memory_db import MemoryDatabase
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MemoryDatabase(Path(tmpdir) / "test.db")
            
            try:
                # Add memory
                mem_id = db.add_memory("Test content", source="test", meta={"key": "value"})
                assert mem_id > 0
                
                # Retrieve
                memory = db.get_by_id(mem_id)
                assert memory is not None
                assert memory['text'] == "Test content"
                assert memory['source'] == "test"
                assert memory['meta']['key'] == "value"
            finally:
                # Close connections
                db.close()
    
    def test_search(self):
        """Test memory search."""
        from enigma.memory.memory_db import MemoryDatabase
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MemoryDatabase(Path(tmpdir) / "test.db")
            
            try:
                db.add_memory("The quick brown fox")
                db.add_memory("jumps over the lazy dog")
                db.add_memory("Python programming")
                
                # Search
                results = db.search("fox", limit=10)
                assert len(results) == 1
                assert "fox" in results[0]['text']
            finally:
                # Close connections
                db.close()


class TestRAGSystem:
    """Tests for RAG (Retrieval-Augmented Generation) system."""
    
    def test_rag_init(self):
        """Test RAG system initialization."""
        from enigma.memory.rag import RAGSystem
        from enigma.memory.vector_db import SimpleVectorDB
        
        vector_db = SimpleVectorDB(dim=128)
        rag = RAGSystem(vector_db)
        
        assert rag is not None
        assert rag.vector_db is vector_db
    
    def test_document_chunking(self):
        """Test document chunking."""
        from enigma.memory.rag import RAGSystem
        from enigma.memory.vector_db import SimpleVectorDB
        
        vector_db = SimpleVectorDB(dim=128)
        rag = RAGSystem(vector_db)
        
        # Add document
        text = "This is a test document. " * 100  # Long document
        chunks = rag.add_document(text, chunk_size=100, overlap=10)
        
        assert len(chunks) > 1  # Should be chunked
        assert vector_db.count() == len(chunks)


class TestEmbeddings:
    """Tests for embedding generation."""
    
    def test_embedding_generator(self):
        """Test embedding generator."""
        from enigma.memory.embeddings import EmbeddingGenerator
        
        # Should fall back to hash-based embeddings if sentence-transformers not available
        embedder = EmbeddingGenerator(model="local")
        
        assert embedder is not None
        assert embedder.dimension > 0
    
    def test_embed_text(self):
        """Test text embedding."""
        from enigma.memory.embeddings import EmbeddingGenerator
        
        embedder = EmbeddingGenerator(model="local")
        
        embedding = embedder.embed("Test text")
        assert embedding is not None
        assert len(embedding) == embedder.dimension
    
    def test_auto_embedding_vector_db(self):
        """Test auto-embedding vector DB."""
        from enigma.memory.embeddings import AutoEmbeddingVectorDB, EmbeddingGenerator
        from enigma.memory.vector_db import SimpleVectorDB
        
        vector_db = SimpleVectorDB(dim=128)
        embedder = EmbeddingGenerator(model="local")
        embedder._dimension = 128  # Match vector DB dimension
        
        auto_db = AutoEmbeddingVectorDB(vector_db, embedder)
        
        # Add text (should auto-embed)
        auto_db.add_text("Test content", "test_id", {"meta": "data"})
        
        assert auto_db.count() == 1


class TestConsolidation:
    """Tests for memory consolidation."""
    
    def test_consolidator_init(self):
        """Test consolidator initialization."""
        from enigma.memory.consolidation import MemoryConsolidator
        from enigma.memory.categorization import MemoryCategorization
        
        memory_system = MemoryCategorization()
        consolidator = MemoryConsolidator(memory_system)
        
        assert consolidator is not None
        assert consolidator.memory_system is memory_system
    
    def test_merge_similar_memories(self):
        """Test merging similar memories."""
        from enigma.memory.consolidation import MemoryConsolidator
        from enigma.memory.categorization import MemoryCategorization, MemoryType
        
        memory_system = MemoryCategorization()
        
        # Add similar memories
        memory_system.add_memory("Hello world", MemoryType.SHORT_TERM)
        memory_system.add_memory("Hello world", MemoryType.SHORT_TERM)  # Duplicate
        
        consolidator = MemoryConsolidator(memory_system)
        merged = consolidator.merge_similar_memories(similarity_threshold=0.9)
        
        assert merged >= 0


class TestAsyncMemory:
    """Tests for async memory operations."""
    
    @pytest.mark.asyncio
    async def test_async_database(self):
        """Test async database operations."""
        try:
            from enigma.memory.async_memory import AsyncMemoryDatabase
            
            with tempfile.TemporaryDirectory() as tmpdir:
                db = AsyncMemoryDatabase(Path(tmpdir) / "test.db")
                
                # Add memory
                mem_id = await db.add_memory("Test content", source="test")
                assert mem_id > 0
                
                # Retrieve
                memories = await db.get_recent(n=10)
                assert len(memories) == 1
                assert memories[0]['text'] == "Test content"
        
        except ImportError:
            pytest.skip("aiosqlite not installed")


class TestMemorySearch:
    """Tests for memory search system."""
    
    def test_search_init(self):
        """Test search initialization."""
        from enigma.memory.search import MemorySearch
        from enigma.memory.memory_db import MemoryDatabase
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MemoryDatabase(Path(tmpdir) / "test.db")
            search = MemorySearch(memory_db=db)
            
            assert search is not None


class TestDeduplication:
    """Tests for memory deduplication."""
    
    def test_deduplicator_init(self):
        """Test deduplicator initialization."""
        from enigma.memory.deduplication import MemoryDeduplicator
        from enigma.memory.categorization import MemoryCategorization
        
        memory_system = MemoryCategorization()
        dedup = MemoryDeduplicator(memory_system)
        
        assert dedup is not None
    
    def test_find_duplicates(self):
        """Test finding duplicates."""
        from enigma.memory.deduplication import MemoryDeduplicator
        from enigma.memory.categorization import MemoryCategorization, MemoryType
        
        memory_system = MemoryCategorization()
        
        # Add duplicates
        memory_system.add_memory("Duplicate content", MemoryType.SHORT_TERM)
        memory_system.add_memory("Duplicate content", MemoryType.SHORT_TERM)
        memory_system.add_memory("Unique content", MemoryType.SHORT_TERM)
        
        dedup = MemoryDeduplicator(memory_system)
        duplicates = dedup.find_duplicates()
        
        assert len(duplicates) >= 1


class TestExportImport:
    """Tests for export/import with compression."""
    
    def test_export_compressed(self):
        """Test compressed export."""
        from enigma.memory.export_import import MemoryExporter
        from enigma.memory.categorization import MemoryCategorization, MemoryType
        
        memory_system = MemoryCategorization()
        memory_system.add_memory("Test memory", MemoryType.SHORT_TERM)
        
        exporter = MemoryExporter(memory_system)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "export.json"
            
            # Export compressed
            stats = exporter.export_compressed(export_path)
            
            assert stats['compressed'] is True
            assert Path(stats['file_path']).exists()
            assert stats['file_path'].endswith('.gz')


class TestVisualization:
    """Tests for memory visualization."""
    
    def test_visualizer_init(self):
        """Test visualizer initialization."""
        from enigma.memory.visualization import MemoryVisualizer
        from enigma.memory.categorization import MemoryCategorization
        
        memory_system = MemoryCategorization()
        viz = MemoryVisualizer(memory_system)
        
        assert viz is not None
    
    def test_generate_timeline(self):
        """Test timeline generation."""
        from enigma.memory.visualization import MemoryVisualizer
        from enigma.memory.categorization import MemoryCategorization, MemoryType
        
        memory_system = MemoryCategorization()
        memory_system.add_memory("Test 1", MemoryType.SHORT_TERM)
        time.sleep(0.1)
        memory_system.add_memory("Test 2", MemoryType.SHORT_TERM)
        
        viz = MemoryVisualizer(memory_system)
        timeline = viz.generate_timeline()
        
        assert 'events' in timeline
        assert len(timeline['events']) == 2
    
    def test_export_html(self):
        """Test HTML export."""
        from enigma.memory.visualization import MemoryVisualizer
        from enigma.memory.categorization import MemoryCategorization, MemoryType
        
        memory_system = MemoryCategorization()
        memory_system.add_memory("Test", MemoryType.SHORT_TERM)
        
        viz = MemoryVisualizer(memory_system)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "viz.html"
            viz.export_to_html(html_path)
            
            assert html_path.exists()
            assert html_path.stat().st_size > 0


class TestAnalytics:
    """Tests for memory analytics."""
    
    def test_analytics_init(self):
        """Test analytics initialization."""
        from enigma.memory.analytics import MemoryAnalytics
        from enigma.memory.categorization import MemoryCategorization
        
        memory_system = MemoryCategorization()
        analytics = MemoryAnalytics(memory_system)
        
        assert analytics is not None
    
    def test_get_statistics(self):
        """Test statistics generation."""
        from enigma.memory.analytics import MemoryAnalytics
        from enigma.memory.categorization import MemoryCategorization, MemoryType
        
        memory_system = MemoryCategorization()
        memory_system.add_memory("Test", MemoryType.SHORT_TERM, importance=0.8)
        
        analytics = MemoryAnalytics(memory_system)
        stats = analytics.get_statistics()
        
        assert stats['total_memories'] == 1
        assert 'by_type' in stats
    
    def test_generate_report(self):
        """Test report generation."""
        from enigma.memory.analytics import MemoryAnalytics
        from enigma.memory.categorization import MemoryCategorization, MemoryType
        
        memory_system = MemoryCategorization()
        memory_system.add_memory("Test", MemoryType.SHORT_TERM)
        
        analytics = MemoryAnalytics(memory_system)
        report = analytics.generate_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "MEMORY ANALYTICS REPORT" in report


class TestEncryption:
    """Tests for memory encryption."""
    
    def test_encryption_init(self):
        """Test encryption initialization."""
        try:
            from enigma.memory.encryption import MemoryEncryption
            
            encryption = MemoryEncryption()
            
            assert encryption is not None
            assert encryption.key is not None
        
        except ImportError:
            pytest.skip("cryptography not installed")
    
    def test_encrypt_decrypt(self):
        """Test encryption and decryption."""
        try:
            from enigma.memory.encryption import MemoryEncryption
            
            encryption = MemoryEncryption()
            
            original = "Secret content"
            encrypted = encryption.encrypt(original)
            decrypted = encryption.decrypt(encrypted)
            
            assert encrypted != original.encode()
            assert decrypted == original
        
        except ImportError:
            pytest.skip("cryptography not installed")


class TestBackup:
    """Tests for backup scheduling."""
    
    def test_backup_scheduler_init(self):
        """Test backup scheduler initialization."""
        from enigma.memory.backup import MemoryBackupScheduler
        from enigma.memory.categorization import MemoryCategorization
        
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_system = MemoryCategorization()
            scheduler = MemoryBackupScheduler(memory_system, Path(tmpdir))
            
            assert scheduler is not None
            assert scheduler.backup_dir.exists()
    
    def test_create_backup(self):
        """Test creating a backup."""
        from enigma.memory.backup import MemoryBackupScheduler
        from enigma.memory.categorization import MemoryCategorization, MemoryType
        
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_system = MemoryCategorization()
            memory_system.add_memory("Test", MemoryType.SHORT_TERM)
            
            scheduler = MemoryBackupScheduler(memory_system, Path(tmpdir))
            
            backup_path = scheduler.create_backup("test_backup")
            
            assert backup_path.exists()
            assert backup_path.suffix == ".zip"
    
    def test_list_backups(self):
        """Test listing backups."""
        from enigma.memory.backup import MemoryBackupScheduler
        from enigma.memory.categorization import MemoryCategorization, MemoryType
        
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_system = MemoryCategorization()
            memory_system.add_memory("Test", MemoryType.SHORT_TERM)
            
            scheduler = MemoryBackupScheduler(memory_system, Path(tmpdir))
            scheduler.create_backup("backup1")
            
            backups = scheduler.list_backups()
            
            assert len(backups) >= 1
            assert 'name' in backups[0]
            assert 'size_bytes' in backups[0]


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with legacy API."""
    
    def test_legacy_functions(self):
        """Test legacy add_memory and recent functions."""
        import tempfile
        from pathlib import Path
        
        # Temporarily override CONFIG for this test
        from enigma.config import CONFIG
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save original path
            original_db_path = CONFIG.get("db_path")
            
            try:
                # Set temporary db path
                CONFIG["db_path"] = str(Path(tmpdir) / "test.db")
                
                # Force reload of memory_db module to pick up new path
                import sys
                if 'enigma.memory.memory_db' in sys.modules:
                    del sys.modules['enigma.memory.memory_db']
                
                from enigma.memory.memory_db import add_memory, recent
                
                # These should still work
                add_memory("Legacy test", source="test")
                
                memories = recent(n=10)
                assert isinstance(memories, list)
                
            finally:
                # Restore original path
                if original_db_path:
                    CONFIG["db_path"] = original_db_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
