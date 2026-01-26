"""
Tests for the engine pool module.
"""

import unittest
import time
from unittest.mock import MagicMock, patch


class TestEnginePoolConfig(unittest.TestCase):
    """Test EnginePoolConfig dataclass."""
    
    def test_default_config(self):
        """Default config should have sane values."""
        from forge_ai.core.engine_pool import EnginePoolConfig
        
        config = EnginePoolConfig()
        
        self.assertEqual(config.max_pool_size, 2)
        self.assertEqual(config.max_idle_seconds, 300.0)
        self.assertTrue(config.preload_on_first_access)
        self.assertTrue(config.enable_auto_cleanup)
    
    def test_custom_config(self):
        """Should accept custom values."""
        from forge_ai.core.engine_pool import EnginePoolConfig
        
        config = EnginePoolConfig(
            max_pool_size=5,
            max_idle_seconds=60.0,
            enable_auto_cleanup=False
        )
        
        self.assertEqual(config.max_pool_size, 5)
        self.assertEqual(config.max_idle_seconds, 60.0)
        self.assertFalse(config.enable_auto_cleanup)


class TestPooledEngine(unittest.TestCase):
    """Test PooledEngine wrapper."""
    
    def test_pooled_engine_creation(self):
        """Should create pooled engine wrapper."""
        from forge_ai.core.engine_pool import PooledEngine
        
        mock_engine = MagicMock()
        pe = PooledEngine(engine=mock_engine, model_path="/test/path")
        
        self.assertEqual(pe.model_path, "/test/path")
        self.assertEqual(pe.use_count, 0)
        self.assertFalse(pe.in_use)
    
    def test_mark_used(self):
        """Should update state when marked as used."""
        from forge_ai.core.engine_pool import PooledEngine
        
        pe = PooledEngine(engine=MagicMock(), model_path="/test")
        pe.mark_used()
        
        self.assertEqual(pe.use_count, 1)
        self.assertTrue(pe.in_use)
    
    def test_mark_released(self):
        """Should update state when released."""
        from forge_ai.core.engine_pool import PooledEngine
        
        pe = PooledEngine(engine=MagicMock(), model_path="/test")
        pe.mark_used()
        pe.mark_released()
        
        self.assertFalse(pe.in_use)
        self.assertEqual(pe.use_count, 1)  # Count preserved
    
    def test_idle_seconds(self):
        """Should track idle time."""
        from forge_ai.core.engine_pool import PooledEngine
        
        pe = PooledEngine(engine=MagicMock(), model_path="/test")
        time.sleep(0.1)
        
        self.assertGreaterEqual(pe.idle_seconds, 0.1)


class TestEnginePoolSingleton(unittest.TestCase):
    """Test EnginePool singleton behavior."""
    
    def test_singleton_pattern(self):
        """Should return same instance."""
        from forge_ai.core.engine_pool import EnginePool, EnginePoolConfig
        
        # Reset singleton for clean test
        EnginePool._instance = None
        
        config = EnginePoolConfig(enable_auto_cleanup=False)
        pool1 = EnginePool(config)
        pool2 = EnginePool()
        
        self.assertIs(pool1, pool2)
        
        # Cleanup
        pool1.shutdown()
        EnginePool._instance = None


class TestEnginePoolStats(unittest.TestCase):
    """Test engine pool statistics."""
    
    def test_get_stats_empty(self):
        """Should return stats for empty pool."""
        from forge_ai.core.engine_pool import EnginePool, EnginePoolConfig
        
        # Reset singleton
        EnginePool._instance = None
        
        config = EnginePoolConfig(enable_auto_cleanup=False)
        pool = EnginePool(config)
        
        stats = pool.get_stats()
        
        self.assertEqual(stats["total_engines"], 0)
        self.assertEqual(stats["in_use"], 0)
        self.assertEqual(stats["idle"], 0)
        
        # Cleanup
        pool.shutdown()
        EnginePool._instance = None


class TestFallbackResponse(unittest.TestCase):
    """Test fallback response generation."""
    
    def test_create_fallback_response(self):
        """Should create user-friendly fallback message."""
        from forge_ai.core.engine_pool import create_fallback_response
        
        response = create_fallback_response("Model not loaded")
        
        self.assertIn("trouble processing", response)
        self.assertIn("Model not loaded", response)
    
    def test_default_fallback_response(self):
        """Should use default error if none provided."""
        from forge_ai.core.engine_pool import create_fallback_response
        
        response = create_fallback_response()
        
        self.assertIn("AI unavailable", response)


class TestConvenienceFunctions(unittest.TestCase):
    """Test module-level convenience functions."""
    
    def test_get_pool_returns_pool(self):
        """get_pool should return EnginePool instance."""
        from forge_ai.core.engine_pool import get_pool, EnginePool
        
        # Reset singleton
        EnginePool._instance = None
        
        pool = get_pool()
        
        self.assertIsInstance(pool, EnginePool)
        
        # Cleanup
        pool.shutdown()
        EnginePool._instance = None


class TestClearPool(unittest.TestCase):
    """Test pool clearing functionality."""
    
    def test_clear_pool(self):
        """Should clear all engines from pool."""
        from forge_ai.core.engine_pool import EnginePool, EnginePoolConfig
        
        # Reset singleton
        EnginePool._instance = None
        
        config = EnginePoolConfig(enable_auto_cleanup=False)
        pool = EnginePool(config)
        
        # Add mock engines
        pool._engines["/test"] = [MagicMock()]
        
        pool.clear_pool()
        
        self.assertEqual(len(pool._engines), 0)
        
        # Cleanup
        pool.shutdown()
        EnginePool._instance = None


class TestShutdown(unittest.TestCase):
    """Test shutdown behavior."""
    
    def test_shutdown_clears_pool(self):
        """Shutdown should clear all engines."""
        from forge_ai.core.engine_pool import EnginePool, EnginePoolConfig
        
        # Reset singleton
        EnginePool._instance = None
        
        config = EnginePoolConfig(enable_auto_cleanup=False)
        pool = EnginePool(config)
        
        pool.shutdown()
        
        self.assertTrue(pool._shutdown)
        self.assertEqual(len(pool._engines), 0)
        
        EnginePool._instance = None


if __name__ == "__main__":
    unittest.main()
