"""Tests for gui_logic_chat.py pure-logic methods (no tkinter required)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Stub host object ────────────────────────────────────────────────────────

def _make_stub_host(history=None, tmp_path=None):
    """Create a minimal stub that satisfies LogicChatMixin attribute access."""
    host = SimpleNamespace()
    host.history = history if history is not None else []
    host._chat_images = []
    host._is_generating = False
    host._current_session_path = ""
    host.engine = None
    host.config_overrides = {}
    return host


# ── _trim_chat_history ───────────────────────────────────────────────────────

class TestTrimChatHistory:
    """Test in-memory history trimming."""

    def test_within_limit_untouched(self):
        """History within cap should not be modified."""
        from enigma_engine.gui.media import MAX_CHAT_HISTORY
        host = _make_stub_host(history=list(range(MAX_CHAT_HISTORY - 10)))
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        LogicChatMixin._trim_chat_history(host)
        assert len(host.history) == MAX_CHAT_HISTORY - 10

    def test_exceeds_limit_trimmed(self):
        """History above cap should be trimmed to cap."""
        from enigma_engine.gui.media import MAX_CHAT_HISTORY
        excess = 50
        host = _make_stub_host(history=list(range(MAX_CHAT_HISTORY + excess)))
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        LogicChatMixin._trim_chat_history(host)
        assert len(host.history) == MAX_CHAT_HISTORY

    def test_trim_keeps_newest(self):
        """Trimming should keep the most recent messages."""
        from enigma_engine.gui.media import MAX_CHAT_HISTORY
        total = MAX_CHAT_HISTORY + 20
        host = _make_stub_host(history=list(range(total)))
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        LogicChatMixin._trim_chat_history(host)
        assert host.history[-1] == total - 1
        assert host.history[0] == 20

    def test_empty_history(self):
        """Empty history should not crash."""
        host = _make_stub_host(history=[])
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        LogicChatMixin._trim_chat_history(host)
        assert host.history == []


# ── _get_memory_mode ─────────────────────────────────────────────────────────

class TestGetMemoryMode:
    """Test memory mode retrieval from gui_settings.json."""

    def test_returns_automatic_default(self, tmp_path, monkeypatch):
        """Should return 'automatic' when no settings file exists."""
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        host = _make_stub_host()
        result = mod.LogicChatMixin._get_memory_mode(host)
        assert result == "automatic"

    def test_reads_from_settings(self, tmp_path, monkeypatch):
        """Should read mode from gui_settings.json."""
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        settings_path = tmp_path / "gui_settings.json"
        settings_path.write_text(
            json.dumps({"memory_mode": "disabled"}), encoding="utf-8")
        host = _make_stub_host()
        result = mod.LogicChatMixin._get_memory_mode(host)
        assert result == "disabled"

    def test_invalid_mode_defaults(self, tmp_path, monkeypatch):
        """Invalid mode value should return 'automatic'."""
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        settings_path = tmp_path / "gui_settings.json"
        settings_path.write_text(
            json.dumps({"memory_mode": "bogus"}), encoding="utf-8")
        host = _make_stub_host()
        result = mod.LogicChatMixin._get_memory_mode(host)
        assert result == "automatic"

    def test_corrupted_json_defaults(self, tmp_path, monkeypatch):
        """Corrupted JSON should return 'automatic'."""
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        settings_path = tmp_path / "gui_settings.json"
        settings_path.write_text("{broken json!!!", encoding="utf-8")
        host = _make_stub_host()
        result = mod.LogicChatMixin._get_memory_mode(host)
        assert result == "automatic"


# ── _load_system_prompt ──────────────────────────────────────────────────────

class TestLoadSystemPrompt:
    """Test system prompt loading from prompts.json."""

    def test_default_when_no_file(self, tmp_path, monkeypatch):
        """Should return default prompt when prompts.json doesn't exist."""
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        host = _make_stub_host()
        result = mod.LogicChatMixin._load_system_prompt(host)
        assert result == "You are a helpful AI assistant."

    def test_reads_from_prompts_json(self, tmp_path, monkeypatch):
        """Should read system_prompt from prompts.json."""
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        prompts_path = tmp_path / "prompts.json"
        prompts_path.write_text(
            json.dumps({"current": {"system_prompt": "Custom prompt"}}),
            encoding="utf-8")
        host = _make_stub_host()
        result = mod.LogicChatMixin._load_system_prompt(host)
        assert result == "Custom prompt"

    def test_corrupted_json_returns_default(self, tmp_path, monkeypatch):
        """Corrupted prompts.json should return default."""
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        prompts_path = tmp_path / "prompts.json"
        prompts_path.write_text("not valid json", encoding="utf-8")
        host = _make_stub_host()
        result = mod.LogicChatMixin._load_system_prompt(host)
        assert result == "You are a helpful AI assistant."

    def test_missing_key_returns_default(self, tmp_path, monkeypatch):
        """prompts.json without system_prompt key should return default."""
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        prompts_path = tmp_path / "prompts.json"
        prompts_path.write_text(
            json.dumps({"current": {}}), encoding="utf-8")
        host = _make_stub_host()
        result = mod.LogicChatMixin._load_system_prompt(host)
        assert result == "You are a helpful AI assistant."


# ── _init_input_history ──────────────────────────────────────────────────────

class TestInitInputHistory:
    """Test input history initialization."""

    def test_initializes_empty(self):
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        host = _make_stub_host()
        LogicChatMixin._init_input_history(host)
        assert host._input_history == []
        assert host._input_hist_idx == -1
        assert host._input_hist_draft == ""
