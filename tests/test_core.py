"""
Tests for Enigma Engine core functionality.

Run with: python -m pytest tests/ -v
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestCoreImports:
    """Test that core modules import correctly."""

    def test_core_modules(self):
        from enigma_engine import CONFIG
        from enigma_engine.core import EnigmaEngine
        from enigma_engine.core import get_hardware
        assert isinstance(CONFIG, dict)
        assert EnigmaEngine is not None
        assert get_hardware is not None


class TestCommandSystem:
    """Test the command processing system."""

    def test_parse_commands(self):
        from enigma_engine.core.commands import parse_commands
        text = "Hello [CMD]system.info[/CMD] world"
        clean, commands = parse_commands(text)
        assert len(commands) == 1
        assert commands[0] == "system.info"
        assert "[CMD]" not in clean
        assert "Hello" in clean and "world" in clean

    def test_parse_multiple_commands(self):
        from enigma_engine.core.commands import parse_commands
        text = "[CMD]gui.tab.switch chat[/CMD] and [CMD]system.info[/CMD]"
        _, commands = parse_commands(text)
        assert len(commands) == 2

    def test_registry(self):
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        available = registry.list_commands()
        assert len(available) > 0


class TestAIProfile:
    """Test AI profile system."""

    def test_profile_create_and_list(self):
        from enigma_engine.core.ai_profile import AIProfile, AIProfileManager
        profile = AIProfile(
            id="test_profile", name="Test Profile",
            system_prompt="You are a test assistant.")
        assert profile.id == "test_profile"
        assert "test assistant" in profile.system_prompt
        manager = AIProfileManager()
        assert isinstance(manager.list_profiles(), list)


class TestRouter:
    """Test the router module."""

    def test_router_basics(self):
        from enigma_engine.router import BrickRouter
        assert hasattr(BrickRouter, "get_prompt")
        assert hasattr(BrickRouter, "set_prompt")


class TestModelRegistry:
    """Test model registry."""

    def test_registry_list(self):
        from enigma_engine.core.model_registry import ModelRegistry
        registry = ModelRegistry()
        models = registry.list_models()
        assert isinstance(models, dict)


