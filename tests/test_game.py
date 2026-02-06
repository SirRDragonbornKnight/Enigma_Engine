"""
Tests for forge_ai.game module.

Tests game mode functionality including:
- Game detection and profiles
- Overlay system
- Game advice and stats
- Streaming integration
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestGameProfiles:
    """Test game profile management."""
    
    def test_profile_loading(self):
        """Test loading game profiles."""
        from forge_ai.game.profiles import GameProfileManager
        
        manager = GameProfileManager()
        assert manager is not None
    
    def test_default_profiles(self):
        """Test that default profiles exist."""
        from forge_ai.game.profiles import GameProfileManager
        
        manager = GameProfileManager()
        profiles = manager.list_profiles()
        
        assert isinstance(profiles, list)
    
    def test_profile_creation(self):
        """Test creating a new game profile."""
        from forge_ai.game.profiles import GameProfileManager, GameProfile
        
        manager = GameProfileManager()
        
        profile = GameProfile(
            name="Test Game",
            executable="test.exe",
            genre="action"
        )
        
        manager.add_profile(profile)
        
        # Should be retrievable
        retrieved = manager.get_profile("Test Game")
        assert retrieved is not None
        assert retrieved.name == "Test Game"
    
    def test_profile_deletion(self):
        """Test deleting a game profile."""
        from forge_ai.game.profiles import GameProfileManager, GameProfile
        
        manager = GameProfileManager()
        
        profile = GameProfile(name="ToDelete", executable="del.exe")
        manager.add_profile(profile)
        
        manager.remove_profile("ToDelete")
        
        assert manager.get_profile("ToDelete") is None


class TestGameOverlay:
    """Test game overlay system."""
    
    def test_overlay_creation(self):
        """Test creating game overlay."""
        from forge_ai.game.overlay import GameOverlay
        
        overlay = GameOverlay(headless=True)
        assert overlay is not None
    
    def test_overlay_visibility(self):
        """Test overlay show/hide."""
        from forge_ai.game.overlay import GameOverlay
        
        overlay = GameOverlay(headless=True)
        
        overlay.show()
        assert overlay.is_visible
        
        overlay.hide()
        assert not overlay.is_visible
    
    def test_overlay_positioning(self):
        """Test overlay position settings."""
        from forge_ai.game.overlay import GameOverlay, OverlayPosition
        
        overlay = GameOverlay(headless=True)
        
        overlay.set_position(OverlayPosition.TOP_RIGHT)
        assert overlay.position == OverlayPosition.TOP_RIGHT


class TestGameAdvice:
    """Test game advice system."""
    
    def test_advice_generator(self):
        """Test advice generation."""
        from forge_ai.game.advice import AdviceGenerator
        
        generator = AdviceGenerator()
        assert generator is not None
    
    def test_context_aware_advice(self):
        """Test context-aware advice generation."""
        from forge_ai.game.advice import AdviceGenerator
        
        generator = AdviceGenerator()
        
        context = {
            'game': 'Test Game',
            'situation': 'boss fight',
            'health': 50
        }
        
        advice = generator.get_advice(context)
        assert isinstance(advice, str)


class TestGameStats:
    """Test game statistics tracking."""
    
    def test_stats_tracker_creation(self):
        """Test creating stats tracker."""
        from forge_ai.game.stats import GameStatsTracker
        
        tracker = GameStatsTracker()
        assert tracker is not None
    
    def test_session_tracking(self):
        """Test tracking a gaming session."""
        from forge_ai.game.stats import GameStatsTracker
        
        tracker = GameStatsTracker()
        
        # Start session
        session_id = tracker.start_session("Test Game")
        assert session_id is not None
        
        # End session
        tracker.end_session(session_id)
        
        # Should have recorded duration
        stats = tracker.get_session(session_id)
        assert stats is not None
        assert stats['duration'] >= 0
    
    def test_cumulative_stats(self):
        """Test cumulative statistics."""
        from forge_ai.game.stats import GameStatsTracker
        
        tracker = GameStatsTracker()
        
        # Get total playtime
        total = tracker.get_total_playtime("Test Game")
        assert total >= 0


class TestGameStreaming:
    """Test streaming integration."""
    
    def test_streaming_mode(self):
        """Test streaming mode configuration."""
        from forge_ai.game.streaming import StreamingMode
        
        mode = StreamingMode()
        assert mode is not None
    
    def test_chat_commands(self):
        """Test chat command handling."""
        from forge_ai.game.streaming import StreamingMode
        
        mode = StreamingMode()
        
        # Register command
        mode.register_command("!test", lambda: "Test response")
        
        # Handle command
        response = mode.handle_command("!test")
        assert response == "Test response"
    
    def test_invalid_command(self):
        """Test handling invalid commands."""
        from forge_ai.game.streaming import StreamingMode
        
        mode = StreamingMode()
        
        response = mode.handle_command("!nonexistent")
        assert response is None


class TestGameDetection:
    """Test game detection functionality."""
    
    def test_detector_creation(self):
        """Test creating game detector."""
        from forge_ai.game.profiles import GameDetector
        
        detector = GameDetector()
        assert detector is not None
    
    def test_running_games(self):
        """Test detecting running games."""
        from forge_ai.game.profiles import GameDetector
        
        detector = GameDetector()
        
        # Should return list (may be empty)
        games = detector.get_running_games()
        assert isinstance(games, list)


class TestGameModeIntegration:
    """Test game mode integration with core system."""
    
    def test_mode_activation(self):
        """Test activating game mode."""
        from forge_ai.core.game_mode import GameMode
        
        mode = GameMode()
        
        mode.activate()
        assert mode.is_active
        
        mode.deactivate()
        assert not mode.is_active
    
    def test_auto_detect(self):
        """Test auto-detection of games."""
        from forge_ai.core.game_mode import GameMode
        
        mode = GameMode()
        
        # Enable auto-detect
        mode.set_auto_detect(True)
        assert mode.auto_detect_enabled
