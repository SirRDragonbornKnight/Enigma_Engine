"""
Unit tests for Game Mode functionality.

Tests the game mode system including process detection, resource limits, and mode activation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProcessMonitor(unittest.TestCase):
    """Test ProcessMonitor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        from enigma_engine.core.process_monitor import ProcessMonitor
        self.monitor = ProcessMonitor()
    
    def test_initialization(self):
        """Test ProcessMonitor initializes correctly."""
        self.assertIsNotNone(self.monitor)
        self.assertIsInstance(self.monitor.get_known_game_processes(), list)
    
    def test_known_games_list(self):
        """Test that known games list is populated."""
        games = self.monitor.get_known_game_processes()
        self.assertGreater(len(games), 0, "Should have known games")
        self.assertIn('csgo.exe', games, "Should include CS:GO")
        self.assertIn('minecraft.exe', games, "Should include Minecraft")
    
    def test_custom_game_addition(self):
        """Test adding custom games."""
        self.monitor.add_custom_game("MyCustomGame.exe")
        games = self.monitor.get_known_game_processes()
        self.assertIn('mycustomgame.exe', games, "Should include custom game (lowercase)")
    
    def test_custom_game_removal(self):
        """Test removing custom games."""
        self.monitor.add_custom_game("TempGame.exe")
        self.monitor.remove_custom_game("TempGame.exe")
        games = self.monitor.get_known_game_processes()
        self.assertNotIn('tempgame.exe', games, "Should not include removed game")


class TestResourceLimits(unittest.TestCase):
    """Test ResourceLimits dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from enigma_engine.core.resource_limiter import ResourceLimits
            self.ResourceLimits = ResourceLimits
            self.has_psutil = True
        except ImportError:
            self.has_psutil = False
    
    def test_default_limits(self):
        """Test default resource limits."""
        if not self.has_psutil:
            self.skipTest("psutil not available")
        
        limits = self.ResourceLimits()
        self.assertEqual(limits.max_cpu_percent, 5.0)
        self.assertEqual(limits.max_memory_mb, 500)
        self.assertFalse(limits.gpu_allowed)
    
    def test_custom_limits(self):
        """Test custom resource limits."""
        if not self.has_psutil:
            self.skipTest("psutil not available")
        
        limits = self.ResourceLimits(
            max_cpu_percent=10.0,
            max_memory_mb=1000,
            gpu_allowed=True,
            background_tasks=True
        )
        self.assertEqual(limits.max_cpu_percent, 10.0)
        self.assertEqual(limits.max_memory_mb, 1000)
        self.assertTrue(limits.gpu_allowed)
        self.assertTrue(limits.background_tasks)


class TestGameMode(unittest.TestCase):
    """Test GameMode functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # We can't fully test without mocking dependencies
        pass
    
    def test_game_mode_status(self):
        """Test game mode status reporting."""
        from enigma_engine.core.game_mode import GameMode
        
        game_mode = GameMode()
        status = game_mode.get_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('enabled', status)
    
    def test_game_mode_enable_disable(self):
        """Test enabling and disabling game mode."""
        from enigma_engine.core.game_mode import GameMode
        
        game_mode = GameMode()
        
        # Test enable
        game_mode.enable(aggressive=False)
        self.assertTrue(game_mode.is_enabled())
        self.assertFalse(game_mode._aggressive)
        
        # Test disable
        game_mode.disable()
        self.assertFalse(game_mode.is_enabled())


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestProcessMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestResourceLimits))
    suite.addTests(loader.loadTestsFromTestCase(TestGameMode))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
