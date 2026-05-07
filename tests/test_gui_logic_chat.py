"""Tests for gui_logic_chat.py pure-logic methods (no tkinter required)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace


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


# ── _build_gui_context emotional state injection ─────────────────────────────

class TestBuildGuiContextEmotionalState:
    """N-22: emotional_state must appear in _build_gui_context output."""

    def _make_gui_stub(self, emotional_state=None):
        """Minimal stub satisfying _build_gui_context attribute access."""
        host = SimpleNamespace()
        host.engine = None
        host.model_path = None
        host.route_assignments = {}
        host.models_data = []
        host.mods_data = []
        host.config_overrides = {}
        host.web_access = False
        host.history = []
        host._rag_index = None
        host.model_context = None

        if emotional_state is not None:
            ctx = SimpleNamespace()
            ctx._snapshot_emotional_state = lambda: dict(emotional_state)
            host.model_context = ctx

        return host

    def test_no_model_context_omits_emotional_line(self):
        """When model_context is None, no Internal State line appears."""
        from enigma_engine.gui.gui_logic import LogicMixin
        host = self._make_gui_stub(emotional_state=None)
        result = LogicMixin._build_gui_context(host)
        assert "Internal State" not in result

    def test_emotional_state_appears_in_output(self):
        """When model_context has emotional state, it appears in context."""
        from enigma_engine.gui.gui_logic import LogicMixin
        emo = {
            "valence": 0.5,
            "arousal": 0.2,
            "engagement": 0.8,
            "trust": 0.6,
            "frustration": 0.1,
        }
        host = self._make_gui_stub(emotional_state=emo)
        result = LogicMixin._build_gui_context(host)
        assert "Internal State" in result
        assert "valence=positive" in result
        assert "arousal=calm" in result
        assert "engagement=high" in result

    def test_negative_valence_label(self):
        """Valence below -0.3 is labelled 'negative'."""
        from enigma_engine.gui.gui_logic import LogicMixin
        emo = {"valence": -0.5, "arousal": 0.5, "engagement": 0.5,
               "trust": 0.5, "frustration": 0.1}
        host = self._make_gui_stub(emotional_state=emo)
        result = LogicMixin._build_gui_context(host)
        assert "valence=negative" in result

    def test_elevated_frustration_label(self):
        """Frustration above 0.6 is labelled 'elevated'."""
        from enigma_engine.gui.gui_logic import LogicMixin
        emo = {"valence": 0.0, "arousal": 0.5, "engagement": 0.5,
               "trust": 0.5, "frustration": 0.8}
        host = self._make_gui_stub(emotional_state=emo)
        result = LogicMixin._build_gui_context(host)
        assert "frustration=elevated" in result

    def test_snapshot_exception_does_not_crash(self):
        """If _snapshot_emotional_state raises, _build_gui_context still returns."""
        from enigma_engine.gui.gui_logic import LogicMixin
        ctx = SimpleNamespace()
        ctx._snapshot_emotional_state = lambda: (_ for _ in ()).throw(
            RuntimeError("broken"))
        host = self._make_gui_stub()
        host.model_context = ctx
        result = LogicMixin._build_gui_context(host)
        # Should not raise; Internal State line silently omitted
        assert "END SYSTEM CONTEXT" in result


# ── AutoResearch-2 Stage A wiring (Pass 154) ────────────────────────────────

class TestAutoResearch2Wiring:
    """Pass 154: post-generation uncertainty gate must be wired into
    the chat _send_message flow.

    Structural test (inspect.getsource) — last-resort per AA rules.
    LogicChatMixin._send_message launches a background thread that
    requires a fully-built Tk host (chat_display, history_list, send_btn,
    typewriter, etc.); a behavioral test would need ~30 stub fields and
    still couldn't drive the threaded _gen() reliably. The contract this
    test enforces is the wiring exists at all — the helper functions
    themselves have full behavioral coverage in test_core.TestAutoResearch.
    """

    def _send_source(self):
        import inspect
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        return inspect.getsource(LogicChatMixin._send_message)

    def test_should_retry_with_research_imported(self):
        """The retry helper is imported in the post-gen path."""
        assert "should_retry_with_research" in self._send_source()

    def test_retry_only_when_web_access_on(self):
        """Retry path is gated by `_web_access_on`."""
        src = self._send_source()
        # The gate condition must mention the flag we set pre-thread
        assert "_web_access_on" in src

    def test_retry_skipped_if_pre_gen_research_ran(self):
        """If pre-gen auto_research already added context (web_research_ctx
        truthy), do not retry — avoids double network fetch and possible loop.
        """
        src = self._send_source()
        # The post-gen block must check `not web_research_ctx`
        assert "not web_research_ctx" in src

    def test_retry_respects_stop_request(self):
        """Stop button must short-circuit the retry path."""
        src = self._send_source()
        assert "_stop_requested" in src

    def test_retry_failure_is_swallowed(self):
        """Network/auto_research failure must not break the original reply."""
        src = self._send_source()
        # The post-gen block is wrapped in try/except so a failure falls
        # back to the original `resp`.
        assert "AutoResearch-2 retry failed" in src


class TestTeach1aCorrectionStore:
    """TEACH-1a: persistent correction capture from CORE chat."""

    def test_append_correction_jsonl_creates_file(self, tmp_path, monkeypatch):
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin

        host = _make_stub_host()
        host._chat_system = lambda _msg: None
        host._chat_error = lambda _msg: None

        row = {
            "prompt": "p",
            "wrong_response": "w",
            "right_response": "r",
            "timestamp": "2026-05-07T00:00:00+00:00",
            "modality": "text",
            "model_path": "",
            "session_path": "",
        }
        target = tmp_path / "corrections.jsonl"
        LogicChatMixin._append_correction_jsonl(host, row, target)
        lines = target.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["right_response"] == "r"

    def test_append_correction_jsonl_preserves_existing_rows(
            self, tmp_path, monkeypatch):
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin

        host = _make_stub_host()
        host._chat_system = lambda _msg: None
        host._chat_error = lambda _msg: None

        target = tmp_path / "corrections.jsonl"
        first = {
            "prompt": "p1",
            "wrong_response": "w1",
            "right_response": "r1",
            "timestamp": "2026-05-07T00:00:00+00:00",
            "modality": "text",
            "model_path": "",
            "session_path": "",
        }
        second = {
            "prompt": "p2",
            "wrong_response": "w2",
            "right_response": "r2",
            "timestamp": "2026-05-07T00:00:01+00:00",
            "modality": "text",
            "model_path": "",
            "session_path": "",
        }
        LogicChatMixin._append_correction_jsonl(host, first, target)
        LogicChatMixin._append_correction_jsonl(host, second, target)

        rows = [json.loads(x) for x in target.read_text(
            encoding="utf-8").splitlines()]
        assert [r["prompt"] for r in rows] == ["p1", "p2"]

    def test_append_correction_jsonl_inserts_missing_newline(
            self, tmp_path, monkeypatch):
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin

        host = _make_stub_host()
        host._chat_system = lambda _msg: None
        host._chat_error = lambda _msg: None

        target = tmp_path / "corrections.jsonl"
        target.write_text('{"prompt":"legacy"}', encoding="utf-8")
        row = {
            "prompt": "p2",
            "wrong_response": "w2",
            "right_response": "r2",
            "timestamp": "2026-05-07T00:00:01+00:00",
            "modality": "text",
            "model_path": "",
            "session_path": "",
        }

        LogicChatMixin._append_correction_jsonl(host, row, target)
        lines = target.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["prompt"] == "legacy"
        assert json.loads(lines[1])["prompt"] == "p2"

    def test_record_correction_uses_latest_user_assistant_pair(
            self, tmp_path, monkeypatch):
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin

        host = _make_stub_host(history=[
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "5"},
        ])
        host.model_path = "models/smoke.pth"
        host._current_session_path = "memory/session_x.json"
        host._chat_msgs = []
        host._chat_errs = []
        host._chat_system = lambda m: host._chat_msgs.append(m)
        host._chat_error = lambda m: host._chat_errs.append(m)
        host._append_correction_jsonl = (
            lambda row, target_path=None:
            LogicChatMixin._append_correction_jsonl(
                host, row, target_path))

        ok = LogicChatMixin._record_correction_for_last_exchange(host, "4")
        assert ok is True
        assert not host._chat_errs
        target = tmp_path / "corrections.jsonl"
        rows = [json.loads(x) for x in target.read_text(
            encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["prompt"] == "What is 2+2?"
        assert rows[0]["wrong_response"] == "5"
        assert rows[0]["right_response"] == "4"
        assert rows[0]["modality"] == "text"

    def test_record_correction_rejects_missing_pair(self):
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin

        host = _make_stub_host(history=[])
        host._chat_msgs = []
        host._chat_system = lambda m: host._chat_msgs.append(m)
        host._chat_error = lambda _m: None

        ok = LogicChatMixin._record_correction_for_last_exchange(host, "4")
        assert ok is False
        assert any("No assistant reply" in m for m in host._chat_msgs)

    def test_record_correction_marks_vision_with_image_path(
            self, tmp_path, monkeypatch):
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin

        host = _make_stub_host(history=[
            {"role": "user", "content": "what is in this image?"},
            {"role": "assistant", "content": "it is a cat"},
        ])
        host._chat_system = lambda _m: None
        host._chat_error = lambda _m: None
        host._last_exchange_prompt = "what is in this image?"
        host._last_exchange_wrong_response = "it is a cat"
        host._last_exchange_image_path = "C:/tmp/frame.png"
        host._append_correction_jsonl = (
            lambda row, target_path=None:
            LogicChatMixin._append_correction_jsonl(
                host, row, target_path))

        ok = LogicChatMixin._record_correction_for_last_exchange(
            host, "it is a dog")
        assert ok is True
        rows = [json.loads(x) for x in (tmp_path / "corrections.jsonl").read_text(
            encoding="utf-8").splitlines()]
        assert rows[0]["modality"] == "vision"
        assert rows[0]["image_path"] == "C:/tmp/frame.png"

    def test_record_correction_uses_text_modality_without_image_context(
            self, tmp_path, monkeypatch):
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin

        host = _make_stub_host(history=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        host._chat_system = lambda _m: None
        host._chat_error = lambda _m: None
        host._append_correction_jsonl = (
            lambda row, target_path=None:
            LogicChatMixin._append_correction_jsonl(
                host, row, target_path))

        ok = LogicChatMixin._record_correction_for_last_exchange(host, "hey")
        assert ok is True
        rows = [json.loads(x) for x in (tmp_path / "corrections.jsonl").read_text(
            encoding="utf-8").splitlines()]
        assert rows[0]["modality"] == "text"
        assert "image_path" not in rows[0]

    def test_save_from_input_clears_box_on_success(self, tmp_path, monkeypatch):
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin

        class _InputStub:
            def __init__(self, text):
                self.text = text
                self.deleted = False

            def get(self, _a, _b):
                return self.text

            def delete(self, _a, _b):
                self.deleted = True
                self.text = ""

        host = _make_stub_host(history=[
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "wrong"},
        ])
        host.chat_input = _InputStub("right")
        host._chat_system = lambda _m: None
        host._chat_error = lambda _m: None
        host._auto_resize_input = lambda: None
        host._append_correction_jsonl = (
            lambda row, target_path=None:
            LogicChatMixin._append_correction_jsonl(
                host, row, target_path))
        host._record_correction_for_last_exchange = (
            lambda text:
            LogicChatMixin._record_correction_for_last_exchange(
                host, text))

        LogicChatMixin._save_last_correction_from_input(host)
        assert host.chat_input.deleted is True

    def test_core_page_wires_fix_button(self):
        """Structural gate: CORE toolbar must wire FIX to
        `_save_last_correction_from_input`."""
        import inspect
        from enigma_engine.gui import gui_pages
        src = inspect.getsource(gui_pages.PagesMixin._build_page_core)
        assert "_correct_btn" in src
        assert "_save_last_correction_from_input" in src

    def test_attach_image_sets_pending_correction_image_path(self):
        """TEACH-1b wire-site: attaching media must persist a pending
        image path so the next correction can tag modality=vision."""
        import inspect
        from enigma_engine.gui.gui_logic_media import LogicMediaMixin

        src = inspect.getsource(LogicMediaMixin._attach_image)
        assert "_pending_correction_image_path" in src

    def test_new_chat_clears_correction_provenance(self, tmp_path, monkeypatch):
        """New chat must clear pending/last-exchange correction state."""
        import enigma_engine.gui.gui_logic_chat as mod
        monkeypatch.setattr(mod, "MEMORY_DIR", tmp_path)
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin

        host = _make_stub_host(history=[{"role": "user", "content": "q"}])
        host._pending_correction_image_path = "C:/tmp/pending.png"
        host._last_exchange_image_path = "C:/tmp/last.png"
        host._last_exchange_prompt = "q"
        host._last_exchange_wrong_response = "wrong"
        host._reset_display = lambda: None
        host._update_token_counter = lambda: None
        host._save_model_context = lambda: None
        host._refresh_history_list = lambda: None
        host._chat_system = lambda _m: None
        host.engine = None

        LogicChatMixin._new_chat(host)

        assert host._pending_correction_image_path == ""
        assert host._last_exchange_image_path == ""
        assert host._last_exchange_prompt == ""
        assert host._last_exchange_wrong_response == ""
