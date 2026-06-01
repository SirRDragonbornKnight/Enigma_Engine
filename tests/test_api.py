"""
Tests for the FastAPI local API server.

Tests the API endpoints, model management, chat, profiles, and system info.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Fixture: FastAPI test client
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Create a test client for the API."""
    from fastapi.testclient import TestClient
    from enigma_engine.api.server import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health & System Info
# ---------------------------------------------------------------------------

class TestHealth:
    """Test health and system info endpoints."""

    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_system_info(self, client):
        resp = client.get("/api/system")
        assert resp.status_code == 200
        data = resp.json()
        assert "device" in data
        assert "python_version" in data
        assert "torch_available" in data


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModels:
    """Test model listing and info."""

    def test_list_models(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["models"], list)

    def test_model_status_no_model(self, client):
        resp = client.get("/api/models/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "loaded" in data


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class TestChat:
    """Test chat endpoint."""

    def test_chat_no_model_loaded(self, client):
        """Chat without a loaded model should return an error."""
        resp = client.post("/api/chat", json={"message": "hello"})
        assert resp.status_code in (200, 503)
        data = resp.json()
        # Either error field or a message about no model
        assert "error" in data or "message" in data

    def test_chat_missing_message(self, client):
        """Chat without a message field should return 422."""
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Streaming Chat
# ---------------------------------------------------------------------------

class TestStreamChat:
    """Test SSE streaming chat endpoint."""

    def test_stream_no_model_loaded(self, client):
        """Stream without a loaded model should return 503."""
        resp = client.post("/api/chat/stream", json={"message": "hello"})
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data

    def test_stream_missing_message(self, client):
        """Stream without a message field should return 422."""
        resp = client.post("/api/chat/stream", json={})
        assert resp.status_code == 422

    def test_stream_with_mock_engine(self, client):
        """Stream with a mocked engine returns SSE events."""
        from enigma_engine.api.server import state

        mock_engine = MagicMock()
        mock_engine.stream_chat = MagicMock(
            return_value=iter(["Hello", " world", "!"]))
        mock_engine.clear_history = MagicMock()

        old_engine = state.engine
        state.engine = mock_engine
        try:
            resp = client.post(
                "/api/chat/stream",
                json={"message": "hi"},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            body = resp.text
            # Should contain SSE data lines
            assert "data:" in body
            # Should contain the tokens
            assert "Hello" in body
            assert "world" in body
        finally:
            state.engine = old_engine

    def test_stream_tracks_history(self, client):
        """Streaming should append to chat history."""
        from enigma_engine.api.server import state

        mock_engine = MagicMock()
        mock_engine.stream_chat = MagicMock(
            return_value=iter(["Yes"]))
        mock_engine.clear_history = MagicMock()

        old_engine = state.engine
        old_histories = dict(state._histories)
        old_order = list(state._conv_order)
        old_active = state._active_conv_id
        state.engine = mock_engine
        state._histories.clear()
        state._conv_order.clear()
        state._active_conv_id = None
        try:
            resp = client.post(
                "/api/chat/stream",
                json={"message": "test"},
            )
            # Drain SSE body so generator runs to completion.
            _ = resp.text
            # One conversation was auto-created; pull its history.
            assert len(state._histories) == 1
            history = next(iter(state._histories.values()))
            assert len(history) == 2
            assert history[0]["role"] == "user"
            assert history[0]["content"] == "test"
            assert history[1]["role"] == "assistant"
            assert "Yes" in history[1]["content"]
        finally:
            state.engine = old_engine
            state._histories.clear()
            state._histories.update(old_histories)
            state._conv_order[:] = old_order
            state._active_conv_id = old_active


# ---------------------------------------------------------------------------
# AutoResearch-2 API parity (Pass 156z9g)
# ---------------------------------------------------------------------------

class TestAutoResearchAPIParity:
    """Pass 156z9g: /api/chat and /api/chat/stream honor ``web_access``.

    Mirrors the GUI wiring at gui_logic_chat.py:239-302 so HTTP callers
    get the same Stage-A behaviour as the desktop UI. Tests use mocked
    ``auto_research`` / ``should_auto_research`` so no real network I/O
    happens.
    """

    def test_chat_web_access_default_off_no_helper_call(
            self, client, monkeypatch):
        """web_access defaults to False; the helper must not run."""
        from enigma_engine.api import server as srv

        called = {"should": 0, "fetch": 0}
        monkeypatch.setattr(
            "enigma_engine.core.auto_research.should_auto_research",
            lambda q: called.__setitem__("should", called["should"] + 1)
            or True,
        )
        monkeypatch.setattr(
            "enigma_engine.core.auto_research.auto_research",
            lambda q, max_results=3: called.__setitem__(
                "fetch", called["fetch"] + 1) or "[CTX]",
        )

        mock_engine = MagicMock()
        mock_engine.chat = MagicMock(return_value="ok")
        old_engine = srv.state.engine
        srv.state.engine = mock_engine
        try:
            resp = client.post(
                "/api/chat", json={"message": "what is the capital of France?"})
            assert resp.status_code == 200
        finally:
            srv.state.engine = old_engine

        assert called["should"] == 0, (
            "should_auto_research must not be called when web_access=False")
        assert called["fetch"] == 0, (
            "auto_research must not be called when web_access=False")

    def test_chat_web_access_on_pre_gen_injects_system_prompt(
            self, client, monkeypatch):
        """web_access=True + should_auto_research True forwards system_prompt."""
        from enigma_engine.api import server as srv

        monkeypatch.setattr(
            "enigma_engine.core.auto_research.should_auto_research",
            lambda q: True,
        )
        monkeypatch.setattr(
            "enigma_engine.core.auto_research.auto_research",
            lambda q, max_results=3: "[WEB RESEARCH] mock context [END]",
        )

        captured = {}
        mock_engine = MagicMock()

        def _chat(message, **kwargs):
            captured.update(kwargs)
            captured["message"] = message
            return "answered"

        mock_engine.chat = _chat
        old_engine = srv.state.engine
        srv.state.engine = mock_engine
        try:
            resp = client.post(
                "/api/chat",
                json={"message": "what is the capital of France?",
                      "web_access": True},
            )
            assert resp.status_code == 200
        finally:
            srv.state.engine = old_engine

        assert "system_prompt" in captured, (
            "system_prompt must be forwarded to engine.chat when "
            "research context is non-empty")
        assert "WEB RESEARCH" in captured["system_prompt"]

    def test_chat_web_access_on_pre_gen_skipped_for_trivial_query(
            self, client, monkeypatch):
        """When should_auto_research returns False, no context is fetched."""
        from enigma_engine.api import server as srv

        fetch_calls = []
        monkeypatch.setattr(
            "enigma_engine.core.auto_research.should_auto_research",
            lambda q: False,
        )
        monkeypatch.setattr(
            "enigma_engine.core.auto_research.auto_research",
            lambda q, max_results=3: fetch_calls.append(q) or "[X]",
        )

        captured = {}
        mock_engine = MagicMock()

        def _chat(message, **kwargs):
            captured.update(kwargs)
            return "ok"

        mock_engine.chat = _chat
        old_engine = srv.state.engine
        srv.state.engine = mock_engine
        try:
            resp = client.post(
                "/api/chat", json={"message": "hi", "web_access": True})
            assert resp.status_code == 200
        finally:
            srv.state.engine = old_engine

        assert fetch_calls == [], (
            "auto_research must not run when should_auto_research is False")
        assert "system_prompt" not in captured, (
            "system_prompt must be omitted when no context was fetched")

    def test_stream_web_access_on_injects_system_prompt(
            self, client, monkeypatch):
        """Streaming endpoint also honors web_access (pre-gen only)."""
        from enigma_engine.api import server as srv

        monkeypatch.setattr(
            "enigma_engine.core.auto_research.should_auto_research",
            lambda q: True,
        )
        monkeypatch.setattr(
            "enigma_engine.core.auto_research.auto_research",
            lambda q, max_results=3: "[WEB RESEARCH] streaming ctx [END]",
        )

        captured = {}
        mock_engine = MagicMock()

        def _stream_chat(message, **kwargs):
            captured.update(kwargs)
            return iter(["a", "b"])

        mock_engine.stream_chat = _stream_chat
        old_engine = srv.state.engine
        old_histories = dict(srv.state._histories)
        old_order = list(srv.state._conv_order)
        old_active = srv.state._active_conv_id
        srv.state.engine = mock_engine
        srv.state._histories.clear()
        srv.state._conv_order.clear()
        srv.state._active_conv_id = None
        try:
            resp = client.post(
                "/api/chat/stream",
                json={"message": "what is the capital of France?",
                      "web_access": True},
            )
            assert resp.status_code == 200
            _ = resp.text  # drain SSE so the generator finishes
        finally:
            srv.state.engine = old_engine
            srv.state._histories.clear()
            srv.state._histories.update(old_histories)
            srv.state._conv_order[:] = old_order
            srv.state._active_conv_id = old_active

        assert "system_prompt" in captured
        assert "WEB RESEARCH" in captured["system_prompt"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTraining:
    """Test training status and trigger endpoints."""

    @staticmethod
    def _install_mock_engine():
        from enigma_engine.api.server import _training_state, state

        mock_engine = MagicMock()
        mock_engine.model = object()
        mock_engine.tokenizer = object()
        old_engine = state.engine
        old_active = _training_state.get("active")
        state.engine = mock_engine
        _training_state["active"] = False
        return state, _training_state, old_engine, old_active

    def test_training_status_idle(self, client):
        """Training status when no training is active."""
        resp = client.get("/api/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "active" in data
        assert data["active"] is False

    def test_cancel_training_idle(self, client):
        from enigma_engine.api.server import _training_state

        old_active = _training_state.get("active")
        _training_state["active"] = False
        try:
            resp = client.delete("/api/training/cancel")
            assert resp.status_code == 200
            assert resp.json().get("status") == "idle"
        finally:
            _training_state["active"] = old_active or False

    def test_cancel_training_active_requests_stop(self, client):
        from enigma_engine.api import server as srv

        trainer = MagicMock()
        old_active = srv._training_state.get("active")
        old_msg = srv._training_state.get("message", "")
        old_trainer = srv._active_training_trainer

        srv._training_state["active"] = True
        srv._training_state["message"] = "Training..."
        srv._active_training_trainer = trainer
        try:
            resp = client.delete("/api/training/cancel")
            assert resp.status_code == 200
            assert resp.json().get("status") == "cancelling"
            trainer.request_stop.assert_called_once()
            assert srv._training_state["message"] == "Cancel requested"
        finally:
            srv._training_state["active"] = old_active or False
            srv._training_state["message"] = old_msg
            srv._active_training_trainer = old_trainer

    def test_train_no_model(self, client):
        """Triggering training without a model should fail."""
        resp = client.post(
            "/api/train",
            json={"mode": "sft", "data": "hello world"},
        )
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data

    def test_train_requires_mode(self, client):
        """Missing ``mode`` is a 422 — legacy data_file shim was removed."""
        state, training_state, old_engine, old_active = self._install_mock_engine()
        try:
            resp = client.post("/api/train", json={})
            assert resp.status_code == 422
            assert "mode" in resp.text.lower()
        finally:
            state.engine = old_engine
            training_state["active"] = old_active or False

    def test_train_rejects_legacy_data_file_field(self, client):
        """Pre-May-27 2026 the API accepted ``data_file``; the shim was
        deleted. Requests using the legacy shape must now fail loudly
        rather than silently routing to SFT."""
        state, training_state, old_engine, old_active = self._install_mock_engine()
        try:
            resp = client.post(
                "/api/train",
                json={"data_file": "training.txt"},
            )
            # No ``mode`` → 422 (mode is now required)
            assert resp.status_code == 422
        finally:
            state.engine = old_engine
            training_state["active"] = old_active or False

    def test_train_rejects_invalid_dispatch_config_before_thread_start(
        self,
        client,
        monkeypatch,
    ):
        """Bad config-body requests should fail at the HTTP boundary, not after 200 started."""
        state, training_state, old_engine, old_active = self._install_mock_engine()
        thread_started = {"value": False}

        class _SentinelThread:
            def __init__(self, target=None, daemon=True, *args, **kwargs):
                self._target = target

            def start(self):
                thread_started["value"] = True
                if self._target is not None:
                    self._target()

        monkeypatch.setattr("threading.Thread", _SentinelThread)
        try:
            resp = client.post(
                "/api/train",
                json={
                    "mode": "dpo",
                },
            )
            assert resp.status_code == 422
            assert thread_started["value"] is False
            assert "non-empty list of preference rows" in resp.text
        finally:
            state.engine = old_engine
            training_state["active"] = old_active or False

    def test_train_dispatcher_routes_dpo_config(self, client, monkeypatch):
        """Config-body with mode=dpo should reach the dispatcher."""
        from enigma_engine.api.server import _training_state, state
        captured: dict = {}

        def fake_run_training(config, ctx):
            captured["config"] = config
            captured["ctx"] = ctx
            return {"ok": True}

        # Drive the threaded path synchronously so the test sees the call.
        class _SyncThread:
            def __init__(self, target, daemon=True):
                self._target = target

            def start(self):
                self._target()

        monkeypatch.setattr(
            "enigma_engine.training.run_training", fake_run_training
        )
        monkeypatch.setattr("threading.Thread", _SyncThread)

        mock_engine = MagicMock()
        mock_engine.model = object()
        mock_engine.tokenizer = object()
        old_engine = state.engine
        old_active = _training_state.get("active")
        state.engine = mock_engine
        _training_state["active"] = False
        try:
            resp = client.post(
                "/api/train",
                json={
                    "mode": "dpo",
                    "data": [
                        {"prompt": "p", "chosen": "c", "rejected": "r"}
                    ],
                    "training": {"epochs": 1},
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["status"] == "started"
            assert body["mode"] == "dpo"
            assert captured["config"].mode == "dpo"
            assert captured["config"].data == [
                {"prompt": "p", "chosen": "c", "rejected": "r"}
            ]
        finally:
            state.engine = old_engine
            _training_state["active"] = old_active or False

    def test_train_sft_with_inline_data(self, client, monkeypatch):
        """SFT mode via dispatcher shape: data is passed inline as text.

        Replaces the pre-May-27 2026 ``data_file`` legacy shim test.
        Callers now read the file themselves and pass the contents in
        ``data`` rather than relying on the API to resolve a path.
        """
        from enigma_engine.api.server import _training_state, state
        captured: dict = {}

        def fake_run_training(config, ctx):
            captured["config"] = config
            return {"ok": True}

        class _SyncThread:
            def __init__(self, target, daemon=True):
                self._target = target

            def start(self):
                self._target()

        monkeypatch.setattr(
            "enigma_engine.training.run_training", fake_run_training
        )
        monkeypatch.setattr("threading.Thread", _SyncThread)

        mock_engine = MagicMock()
        mock_engine.model = object()
        mock_engine.tokenizer = object()
        old_engine = state.engine
        old_active = _training_state.get("active")
        state.engine = mock_engine
        _training_state["active"] = False
        try:
            resp = client.post(
                "/api/train",
                json={
                    "mode": "sft",
                    "data": "hello world",
                    "training": {"epochs": 2},
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["mode"] == "sft"
            assert captured["config"].mode == "sft"
            assert captured["config"].data == "hello world"
            assert captured["config"].training.epochs == 2
        finally:
            state.engine = old_engine
            _training_state["active"] = old_active or False


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class TestProfiles:
    """Test AI profile endpoints."""

    def test_list_profiles(self, client):
        resp = client.get("/api/profiles")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["profiles"], list)
        # We have profile JSON files in profiles/
        assert len(data["profiles"]) > 0

    def test_get_profile(self, client):
        resp = client.get("/api/profiles/assistant")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data

    def test_get_profile_not_found(self, client):
        """Missing profile should return 404."""
        resp = client.get("/api/profiles/nonexistent_profile_xyz")
        assert resp.status_code == 404

    def test_get_profile_corrupt_json(self, client, tmp_path):
        """Corrupt profile JSON should return 500, not crash (S670)."""
        from enigma_engine.api import server
        orig_dir = server.PROFILES_DIR
        server.PROFILES_DIR = tmp_path
        try:
            bad = tmp_path / "corrupt.json"
            bad.write_text("NOT VALID JSON {{{", encoding="utf-8")
            resp = client.get("/api/profiles/corrupt")
            assert resp.status_code == 500
            assert "Corrupt" in resp.json()["detail"]
        finally:
            server.PROFILES_DIR = orig_dir

    def test_activate_profile_corrupt_json(self, client, tmp_path):
        """Activating a corrupted profile should return 500 (S670)."""
        from enigma_engine.api import server
        orig_dir = server.PROFILES_DIR
        server.PROFILES_DIR = tmp_path
        try:
            bad = tmp_path / "broken.json"
            bad.write_text("{invalid", encoding="utf-8")
            resp = client.post("/api/profiles/broken/activate")
            assert resp.status_code == 500
            assert "Corrupt" in resp.json()["detail"]
        finally:
            server.PROFILES_DIR = orig_dir

    def test_activate_profile_applies_to_loaded_engine(self, client, tmp_path):
        """Pass 156z2: activate endpoint must call apply_profile_to_engine
        when an engine is loaded — so system_prompt + adapter + roleplay
        boundary actually reach the live engine. Pre-fix, only the
        ``generation`` block was applied; everything else was dropped on
        the floor and ``apply_profile_to_engine`` had zero production
        callers."""
        from enigma_engine.api import server

        # Stub engine with the attributes apply_profile_to_engine touches
        class _StubEngine:
            def __init__(self) -> None:
                self.temperature = 1.0
                self.top_p = 1.0
                self.top_k = 0
                self.max_tokens = 256
                self.system_prompt = ""

            # Adapter no-ops — profile we write has no adapter so
            # clear_adapter() is the path taken
            def clear_adapter(self) -> None:
                self.cleared = True

            def apply_adapter(self, path: str) -> None:  # pragma: no cover
                self.adapter_path = path

        stub = _StubEngine()
        orig_engine = server.state.engine
        orig_dir = server.PROFILES_DIR
        server.state.engine = stub
        server.PROFILES_DIR = tmp_path
        try:
            (tmp_path / "boundary.json").write_text(
                '{"name": "BoundaryTest", "system_prompt": "you are a test",'
                ' "personality": {"tone": "playful"},'
                ' "generation": {"temperature": 0.42}}',
                encoding="utf-8",
            )
            resp = client.post("/api/profiles/boundary/activate")
            assert resp.status_code == 200
            # apply_profile_to_engine actually ran — engine state mutated
            assert stub.system_prompt == "you are a test"
            assert stub.temperature == 0.42
            assert getattr(stub, "cleared", False) is True
        finally:
            server.state.engine = orig_engine
            server.PROFILES_DIR = orig_dir

    def test_activate_profile_updates_state_under_lock(self, client, tmp_path):
        """ARCH-1e lock-scope: activate_profile must update active_profile and
        config_overrides while holding state._lock so concurrent chat calls
        see consistent state.

        Contract: after a successful activate POST, both state.active_profile
        and state.config_overrides reflect the new profile atomically.
        """
        from enigma_engine.api import server

        orig_dir = server.PROFILES_DIR
        orig_active = server.state.active_profile
        orig_overrides = dict(server.state.config_overrides)
        server.PROFILES_DIR = tmp_path
        try:
            (tmp_path / "locktest.json").write_text(
                '{"name": "LockTest", "generation": {"temperature": 0.77}}',
                encoding="utf-8",
            )
            resp = client.post("/api/profiles/locktest/activate")
            assert resp.status_code == 200
            # Both fields must be updated by the same call
            assert server.state.active_profile == "locktest"
            assert server.state.config_overrides.get("temperature") == pytest.approx(0.77)
        finally:
            server.state.active_profile = orig_active
            server.state.config_overrides.clear()
            server.state.config_overrides.update(orig_overrides)
            server.PROFILES_DIR = orig_dir

    def test_activate_profile_lock_scope_structural(self):
        """ARCH-1e: state._lock must be acquired before mutating
        state.active_profile and state.config_overrides.
        """
        import inspect
        from enigma_engine.api.server import activate_profile

        src = inspect.getsource(activate_profile)
        # The lock acquisition must appear BEFORE the active_profile assignment
        lock_pos = src.find("state._lock")
        profile_assign_pos = src.find("state.active_profile")
        assert lock_pos != -1, "state._lock not acquired in activate_profile"
        assert profile_assign_pos != -1, \
            "state.active_profile not set in activate_profile"
        assert lock_pos < profile_assign_pos, (
            "state._lock must be acquired before state.active_profile is assigned"
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    """Test config endpoints."""

    def test_get_config(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "temperature" in data or "config" in data

    def test_update_config(self, client):
        resp = client.post("/api/config", json={"temperature": 0.5})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "ok" or "temperature" in str(data)

    def test_update_engine_flags_no_engine_returns_no_engine(self, client):
        """POST /api/config/engine-flags with no model loaded → no-engine status."""
        from enigma_engine.api.server import state
        old = state.engine
        state.engine = None
        try:
            resp = client.post(
                "/api/config/engine-flags",
                json={"inline_search_enabled": False},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "no-engine"
            assert data["applied"] == {}
        finally:
            state.engine = old

    def test_update_engine_flags_applies_to_engine(self, client):
        """POST /api/config/engine-flags sets attributes on the live engine."""
        from enigma_engine.api.server import state
        mock_engine = MagicMock()
        old = state.engine
        state.engine = mock_engine
        try:
            resp = client.post(
                "/api/config/engine-flags",
                json={
                    "inline_search_enabled": False,
                    "inline_search_splice_enabled": True,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["applied"]["inline_search_enabled"] is False
            assert data["applied"]["inline_search_splice_enabled"] is True
            # Verify setattr actually fired on the engine object
            assert mock_engine.inline_search_enabled is False
            assert mock_engine.inline_search_splice_enabled is True
        finally:
            state.engine = old


# ---------------------------------------------------------------------------
# Batch Inference
# ---------------------------------------------------------------------------

class TestBatchInference:
    """Test batch inference endpoint."""

    def test_batch_no_model_loaded(self, client):
        """Batch without a loaded model should return 503."""
        resp = client.post(
            "/api/batch",
            json={"prompts": ["hello", "world"]},
        )
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data

    def test_batch_missing_prompts(self, client):
        """Batch without prompts field should return 422."""
        resp = client.post("/api/batch", json={})
        assert resp.status_code == 422

    def test_batch_empty_prompts(self, client):
        """Batch with empty prompts list should return 400."""
        from enigma_engine.api.server import state
        mock_engine = MagicMock()
        old_engine = state.engine
        state.engine = mock_engine
        try:
            resp = client.post(
                "/api/batch",
                json={"prompts": []},
            )
            assert resp.status_code == 400
            data = resp.json()
            assert "error" in data
        finally:
            state.engine = old_engine

    def test_batch_with_mock_engine(self, client):
        """Batch with a mocked engine returns multiple responses."""
        from enigma_engine.api.server import state

        mock_engine = MagicMock()
        mock_engine.batch_generate = MagicMock(
            return_value=["answer1", "answer2"])

        old_engine = state.engine
        state.engine = mock_engine
        try:
            resp = client.post(
                "/api/batch",
                json={"prompts": ["q1", "q2"], "max_tokens": 50},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "responses" in data
            assert len(data["responses"]) == 2
            assert data["responses"][0] == "answer1"
        finally:
            state.engine = old_engine

    def test_batch_falls_back_to_sequential(self, client):
        """If engine lacks batch_generate, falls back to sequential."""
        from enigma_engine.api.server import state

        mock_engine = MagicMock(spec=[])
        mock_engine.chat = MagicMock(side_effect=["r1", "r2"])

        old_engine = state.engine
        state.engine = mock_engine
        try:
            resp = client.post(
                "/api/batch",
                json={"prompts": ["p1", "p2"]},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "responses" in data
            assert len(data["responses"]) == 2
        finally:
            state.engine = old_engine


# ---------------------------------------------------------------------------
# CORS — Suggestion #6
# ---------------------------------------------------------------------------

class TestCORSPolicy:
    """Test that CORS middleware is NOT added by default."""

    def test_no_cors_headers_by_default(self, client):
        """Without --cors-origins, responses should not include CORS headers."""
        resp = client.get("/api/health", headers={"Origin": "http://evil.com"})
        assert resp.status_code == 200
        # No access-control-allow-origin header when CORS middleware is absent
        assert "access-control-allow-origin" not in resp.headers

    def test_no_cors_middleware_at_import(self):
        """CORS middleware should not be registered at module level."""
        import ast
        src = (Path(__file__).parent.parent
               / "enigma_engine" / "api" / "server.py").read_text(
                   encoding="utf-8")
        tree = ast.parse(src)
        # Only check top-level statements (not inside functions/classes)
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                func = call.func
                if (isinstance(func, ast.Attribute)
                        and func.attr == "add_middleware"):
                    for arg in call.args:
                        if isinstance(arg, ast.Name) and arg.id == "CORSMiddleware":
                            pytest.fail(
                                "CORSMiddleware is added at module level — "
                                "should only be added via enable_cors()")


# ---------------------------------------------------------------------------
# Input Limits — Suggestion #7
# ---------------------------------------------------------------------------

class TestInputLimits:
    """Test Pydantic input size validation on API requests."""

    def test_chat_message_too_long(self, client):
        """Chat message exceeding MAX_MESSAGE_LENGTH should be rejected."""
        from enigma_engine.api.server import state, MAX_MESSAGE_LENGTH
        mock_engine = MagicMock()
        mock_engine.chat = MagicMock(return_value="ok")
        old = state.engine
        state.engine = mock_engine
        try:
            long_msg = "x" * (MAX_MESSAGE_LENGTH + 1)
            resp = client.post("/api/chat", json={"message": long_msg})
            assert resp.status_code == 422
        finally:
            state.engine = old

    def test_chat_message_within_limit(self, client):
        """Chat message within limit should be accepted."""
        from enigma_engine.api.server import state, MAX_MESSAGE_LENGTH
        mock_engine = MagicMock()
        mock_engine.chat = MagicMock(return_value="ok")
        old = state.engine
        state.engine = mock_engine
        try:
            msg = "x" * MAX_MESSAGE_LENGTH
            resp = client.post("/api/chat", json={"message": msg})
            assert resp.status_code == 200
        finally:
            state.engine = old

    def test_batch_too_many_prompts(self, client):
        """Batch with too many prompts should be rejected."""
        from enigma_engine.api.server import state, MAX_BATCH_PROMPTS
        mock_engine = MagicMock()
        mock_engine.batch_generate = MagicMock(return_value=[])
        old = state.engine
        state.engine = mock_engine
        try:
            prompts = ["hi"] * (MAX_BATCH_PROMPTS + 1)
            resp = client.post("/api/batch", json={"prompts": prompts})
            assert resp.status_code == 422
        finally:
            state.engine = old

    def test_batch_prompt_too_long(self, client):
        """Individual batch prompt exceeding limit should be rejected."""
        from enigma_engine.api.server import state, MAX_MESSAGE_LENGTH
        mock_engine = MagicMock()
        old = state.engine
        state.engine = mock_engine
        try:
            long_prompt = "x" * (MAX_MESSAGE_LENGTH + 1)
            resp = client.post(
                "/api/batch", json={"prompts": [long_prompt]})
            assert resp.status_code == 422
        finally:
            state.engine = old

    def test_constants_exported(self):
        """Limit constants should be importable."""
        from enigma_engine.api.server import (
            MAX_MESSAGE_LENGTH, MAX_BATCH_PROMPTS, MAX_PATH_LENGTH)
        assert MAX_MESSAGE_LENGTH == 32_768
        assert MAX_BATCH_PROMPTS == 50
        assert MAX_PATH_LENGTH == 256


# ---------------------------------------------------------------------------
# Concurrency Lock — Suggestion #7
# ---------------------------------------------------------------------------

class TestConcurrencyLock:
    """Test that concurrent chat/batch requests are rejected."""

    def test_chat_returns_429_when_busy(self, client):
        """Second concurrent chat request should get 429."""
        from enigma_engine.api.server import state, _inference_lock
        mock_engine = MagicMock()
        mock_engine.chat = MagicMock(return_value="ok")
        old = state.engine
        state.engine = mock_engine
        try:
            # Acquire the lock to simulate a busy engine
            _inference_lock.acquire()
            try:
                resp = client.post(
                    "/api/chat", json={"message": "hello"})
                assert resp.status_code == 429
                assert "busy" in resp.json()["error"].lower()
            finally:
                _inference_lock.release()
        finally:
            state.engine = old

    def test_batch_returns_429_when_busy(self, client):
        """Second concurrent batch request should get 429."""
        from enigma_engine.api.server import state, _inference_lock
        mock_engine = MagicMock()
        mock_engine.batch_generate = MagicMock(return_value=["ok"])
        old = state.engine
        state.engine = mock_engine
        try:
            _inference_lock.acquire()
            try:
                resp = client.post(
                    "/api/batch", json={"prompts": ["hello"]})
                assert resp.status_code == 429
                assert "busy" in resp.json()["error"].lower()
            finally:
                _inference_lock.release()
        finally:
            state.engine = old

    def test_stream_returns_429_when_busy(self, client):
        """Concurrent stream request should get 429."""
        from enigma_engine.api.server import state, _inference_lock
        mock_engine = MagicMock()
        old = state.engine
        state.engine = mock_engine
        try:
            _inference_lock.acquire()
            try:
                resp = client.post(
                    "/api/chat/stream", json={"message": "hello"})
                assert resp.status_code == 429
            finally:
                _inference_lock.release()
        finally:
            state.engine = old

    def test_chat_works_when_not_busy(self, client):
        """Chat should work normally when lock is free."""
        from enigma_engine.api.server import state
        mock_engine = MagicMock()
        mock_engine.chat = MagicMock(return_value="response")
        old = state.engine
        state.engine = mock_engine
        try:
            resp = client.post(
                "/api/chat", json={"message": "hello"})
            assert resp.status_code == 200
            assert resp.json()["message"] == "response"
        finally:
            state.engine = old


# ---------------------------------------------------------------------------
# Thread Safety — Suggestion #8A+D: AppState locking + copy-on-write
# ---------------------------------------------------------------------------

class TestAppStateThreadSafety:
    """Verify AppState uses locks and copy-on-write snapshots."""

    def test_appstate_has_lock(self):
        """AppState must have a threading.Lock for state mutations."""
        import threading
        from enigma_engine.api.server import AppState
        s = AppState()
        assert hasattr(s, "_lock")
        assert isinstance(s._lock, type(threading.Lock()))

    def test_history_snapshot_returns_copy(self):
        """history_snapshot must return a list copy, not the internal ref."""
        from enigma_engine.api.server import AppState
        s = AppState()
        # MC-1: per-conversation history.  Seed one conversation, then
        # verify the snapshot is independent of the live list.
        cid = s.create_conversation()
        s._resolve_and_activate(cid)
        s._histories[cid].append({"role": "user", "content": "hi"})
        snap = s.history_snapshot()
        assert snap == [{"role": "user", "content": "hi"}]
        # Mutating the snapshot must NOT affect internal state
        snap.append({"role": "user", "content": "extra"})
        assert len(s.history_snapshot()) == 1

    def test_model_info_snapshot_returns_copy(self):
        """model_info_snapshot must return a dict copy."""
        from enigma_engine.api.server import AppState
        s = AppState()
        s.model_info = {"path": "x", "loaded": True}
        snap = s.model_info_snapshot()
        snap["hacked"] = True
        assert "hacked" not in s.model_info_snapshot()

    def test_chat_appends_under_lock(self):
        """AppState.chat() must append to history safely."""
        from enigma_engine.api.server import AppState
        s = AppState()
        mock_eng = MagicMock()
        mock_eng.chat = MagicMock(return_value="reply")
        s.engine = mock_eng
        # MC-1: chat() now returns (response, conv_id) and auto-creates
        # a conversation when ``conversation_id=None``.
        response, conv_id = s.chat("hello")
        assert response == "reply"
        history = s.history_snapshot(conv_id)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# API Key Authentication — Suggestion #24A
# ---------------------------------------------------------------------------

class TestAPIKeyAuth:
    """Test API key middleware rejects unauthorized requests."""

    @pytest.fixture
    def secured_client(self):
        """Create a fresh FastAPI app with API key middleware."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import JSONResponse as _JSONResponse

        _key = "test-secret-key-12345"

        class _TestAPIKeyMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                auth = request.headers.get("Authorization", "")
                if not auth.startswith("Bearer ") or auth[7:] != _key:
                    return _JSONResponse(
                        {"error": "API key required"}, status_code=401)
                return await call_next(request)

        secured_app = FastAPI()

        secured_app.add_middleware(_TestAPIKeyMiddleware)

        @secured_app.get("/api/health")
        async def health():
            return {"status": "ok"}

        client = TestClient(secured_app)
        yield client, _key

    def test_api_key_required_rejects_no_auth(self, secured_client):
        """Requests without Authorization header should get 401."""
        client, _key = secured_client
        resp = client.get("/api/health")
        assert resp.status_code == 401
        assert "api key" in resp.json()["error"].lower()

    def test_api_key_required_rejects_wrong_key(self, secured_client):
        """Requests with wrong API key should get 401."""
        client, _key = secured_client
        resp = client.get(
            "/api/health",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_api_key_valid_allows_access(self, secured_client):
        """Requests with correct API key should succeed."""
        client, _key = secured_client
        resp = client.get(
            "/api/health",
            headers={"Authorization": f"Bearer {_key}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Path Traversal Protection — Suggestion #24B
# ---------------------------------------------------------------------------

class TestPathTraversal:
    """Test that model load and training endpoints block path traversal."""

    def test_model_load_rejects_traversal(self, client):
        """Model load with ../ should be rejected."""
        resp = client.post(
            "/api/models/load",
            json={"path": "../../../etc/passwd"},
        )
        assert resp.status_code == 403

    def test_model_load_rejects_absolute_path(self, client):
        """Model load with absolute path outside models/ should fail."""
        import os
        if os.name == "nt":
            bad_path = "C:\\Windows\\System32\\config"
        else:
            bad_path = "/etc/passwd"
        resp = client.post(
            "/api/models/load",
            json={"path": bad_path},
        )
        # 403 (path outside models/) or 404 (resolved but not found)
        assert resp.status_code in (403, 404)

    # Path-traversal tests on the legacy ``data_file`` training field
    # were deleted May 27 2026 alongside the field itself. The
    # dispatcher shape passes data inline (``data`` is a string or list,
    # never a path the server resolves), so this attack class is
    # structurally impossible at the new boundary.

    def test_model_load_valid_path_inside_models(self, client):
        """Model load with a path inside models/ should not get 403."""
        resp = client.post(
            "/api/models/load",
            json={"path": "models/base.pth"},
        )
        # Should be 200 (loaded) or 404 (not found) — not 403,
        # because models/base.pth resolves inside MODELS_DIR
        assert resp.status_code in (200, 404, 500)

    def test_profile_endpoint_blocks_traversal(self, client):
        """Profile endpoint should reject path traversal."""
        resp = client.get("/api/profiles/../../etc/passwd")
        # Should be 404 (not found) — the path separator stripping
        # prevents directory traversal
        assert resp.status_code in (404, 422)


class TestModelVocabGuards:
    """Guards against tokenizer/model vocab mismatch crash paths."""

    def test_appstate_load_model_rejects_vocab_mismatch(
            self, monkeypatch: pytest.MonkeyPatch):
        """Load must fail loud when tokenizer vocab exceeds model limit.

        This prevents loading a model that will later crash generation
        with index errors.
        """
        from enigma_engine.api.server import AppState

        class _FakeTokEmb:
            num_embeddings = 16

        class _FakeModel:
            tok_embeddings = _FakeTokEmb()

        class _FakeTokenizer:
            vocab_size = 32

        class _FakeEngine:
            def __init__(self, model_path: str):
                self.model = _FakeModel()
                self.tokenizer = _FakeTokenizer()

        monkeypatch.setattr("enigma_engine.core.EnigmaEngine", _FakeEngine)

        state = AppState()
        with pytest.raises(RuntimeError, match="vocabulary mismatch"):
            state.load_model("models/smoke.pth")
        assert state.engine is None
        assert state.model_path is None

    def test_appstate_load_model_gguf_no_parameters_no_crash(
            self, monkeypatch: pytest.MonkeyPatch):
        """A GGUF engine.model has no .parameters(); the load must succeed.

        Regression for the load failure ``'GGUFModel' object has no
        attribute 'parameters'`` — info-gathering reached into
        ``engine.model.parameters()`` assuming an nn.Module, which crashed
        every GGUF load through the API. GGUF wraps a llama.cpp backend, so
        param counting is best-effort and reports 0.
        """
        from enigma_engine.api.server import AppState

        class _FakeGGUFModel:
            # Mirrors GGUFModel: a llama.cpp wrapper, NOT an nn.Module.
            # No .parameters() and no .tok_embeddings, so the vocab guard
            # (get_model_vocab_limit) returns None and skips — exactly as a
            # real GGUF load does before reaching param counting.
            model_path = "qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf"

        class _FakeEngine:
            def __init__(self, model_path: str):
                self.model = _FakeGGUFModel()
                self.tokenizer = None
                self._is_gguf = True

        monkeypatch.setattr("enigma_engine.core.EnigmaEngine", _FakeEngine)

        state = AppState()
        info = state.load_model("qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf")

        assert info["loaded"] is True
        assert info["parameters"] == 0
        assert state.engine is not None
        assert state.model_path.endswith("Qwen3-30B-A3B-Q4_K_M.gguf")

    def test_chat_returns_400_for_value_error(self, client):
        """/api/chat should map user/input validation errors to HTTP 400."""
        from enigma_engine.api.server import state

        mock_engine = MagicMock()
        mock_engine.chat = MagicMock(
            side_effect=ValueError("Prompt token IDs out of model vocabulary range")
        )

        old_engine = state.engine
        state.engine = mock_engine
        try:
            resp = client.post("/api/chat", json={"message": "hello"})
            assert resp.status_code == 400
            assert "Invalid request" in resp.json().get("error", "")
        finally:
            state.engine = old_engine


class TestChatRequestSamplingParams:
    """ChatRequest accepts top_p, top_k, repetition_penalty."""

    def test_chat_request_has_sampling_fields(self):
        """ChatRequest model has all common sampling parameters."""
        from enigma_engine.api.server import ChatRequest
        fields = ChatRequest.model_fields
        assert "top_p" in fields
        assert "top_k" in fields
        assert "repetition_penalty" in fields

    def test_chat_request_sampling_defaults_none(self):
        """Sampling params default to None (use engine defaults)."""
        from enigma_engine.api.server import ChatRequest
        req = ChatRequest(message="test")
        assert req.top_p is None
        assert req.top_k is None
        assert req.repetition_penalty is None

    def test_chat_request_accepts_sampling_values(self):
        """ChatRequest accepts explicit sampling values."""
        from enigma_engine.api.server import ChatRequest
        req = ChatRequest(
            message="test",
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.2,
        )
        assert req.top_p == 0.95
        assert req.top_k == 40
        assert req.repetition_penalty == 1.2


# ================================================================
# N-15b: ChatRequest.json_schema reaches engine.chat as kwarg
# ================================================================

class TestChatJsonSchemaWiring:
    """Pass 156z3 follow-up: the json_schema kwarg added in N-15 must be
    reachable from the production HTTP surface, otherwise the previous
    pass shipped infrastructure with no production caller \u2014 the same
    half-wired-contract failure mode that motivated 156z3 in the first
    place. These tests gate the four wire-sites that connect the chat
    API to the constraint."""

    def test_chat_request_has_json_schema_field(self):
        """ChatRequest exposes json_schema (None default = opt-in)."""
        from enigma_engine.api.server import ChatRequest
        fields = ChatRequest.model_fields
        assert "json_schema" in fields, (
            "ChatRequest must accept json_schema for N-15b")
        req = ChatRequest(message="test")
        assert req.json_schema is None
        req2 = ChatRequest(
            message="test",
            json_schema={"type": "object", "properties": {}},
        )
        assert req2.json_schema == {"type": "object", "properties": {}}

    def test_appstate_chat_forwards_json_schema(self):
        """AppState.chat must forward json_schema as a kwarg to engine.chat
        \u2014 without this the request-side field is a silent no-op."""
        from unittest.mock import MagicMock
        from enigma_engine.api.server import AppState
        s = AppState()
        mock_eng = MagicMock()
        mock_eng.chat = MagicMock(return_value="reply")
        s.engine = mock_eng
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        s.chat("hi", json_schema=schema)
        # The schema must appear in the engine.chat call kwargs
        _, kwargs = mock_eng.chat.call_args
        assert kwargs.get("json_schema") == schema, (
            "AppState.chat must forward json_schema kwarg to engine.chat")

    def test_appstate_chat_omits_json_schema_when_none(self):
        """When json_schema is None (default), the kwarg must NOT appear
        in engine.chat \u2014 otherwise legacy engines without the parameter
        would receive an unexpected kwarg on every call. Pre-N-15
        engines could legitimately raise TypeError otherwise."""
        from unittest.mock import MagicMock
        from enigma_engine.api.server import AppState
        s = AppState()
        mock_eng = MagicMock()
        mock_eng.chat = MagicMock(return_value="reply")
        s.engine = mock_eng
        s.chat("hi")  # no json_schema
        _, kwargs = mock_eng.chat.call_args
        assert "json_schema" not in kwargs, (
            "AppState.chat must NOT pass json_schema=None as a kwarg")

    def test_chat_endpoint_forwards_json_schema(self, client):
        """POST /api/chat with json_schema must forward to engine.chat
        \u2014 end-to-end production-path test (no calls to AppState.chat
        directly). Catches the regression where the handler builds its
        own kwargs dict and forgets to copy json_schema across."""
        from unittest.mock import MagicMock
        from enigma_engine.api.server import state
        mock_engine = MagicMock()
        mock_engine.chat = MagicMock(return_value="ok")
        old = state.engine
        state.engine = mock_engine
        try:
            schema = {"type": "object", "properties": {}}
            resp = client.post(
                "/api/chat",
                json={"message": "hi", "json_schema": schema},
            )
            assert resp.status_code == 200
            _, kwargs = mock_engine.chat.call_args
            assert kwargs.get("json_schema") == schema, (
                "/api/chat handler must forward json_schema to engine.chat")
        finally:
            state.engine = old

    def test_chat_stream_forwards_json_schema_when_set(self, client):
        """Pass 156z6 (N-15c): /api/chat/stream must forward json_schema
        to engine.stream_chat as a kwarg. Replaces the Pass 156z4
        rejection — streaming-with-constraint is now supported because
        the FSM advances per yielded token in `stream_generate`."""
        from unittest.mock import MagicMock
        from enigma_engine.api.server import state
        mock_engine = MagicMock()
        mock_engine.stream_chat = MagicMock(return_value=iter(["{", "}"]))
        old = state.engine
        state.engine = mock_engine
        try:
            schema = {"type": "object", "properties": {}}
            resp = client.post(
                "/api/chat/stream",
                json={"message": "hi", "json_schema": schema},
            )
            assert resp.status_code == 200
            _, kwargs = mock_engine.stream_chat.call_args
            assert kwargs.get("json_schema") == schema, (
                "/api/chat/stream must forward json_schema to "
                "engine.stream_chat — without this, streaming callers "
                "silently get unconstrained output")
        finally:
            state.engine = old

    def test_chat_stream_omits_json_schema_when_none(self, client):
        """Pass 156z6 (N-15c): when json_schema is None (default), the
        kwarg must NOT appear in stream_chat call. Mirrors the
        omit-when-None discipline on the non-streaming endpoint —
        legacy engines without the parameter on stream_generate
        would otherwise raise TypeError on every streaming request."""
        from unittest.mock import MagicMock
        from enigma_engine.api.server import state
        mock_engine = MagicMock()
        mock_engine.stream_chat = MagicMock(return_value=iter(["x"]))
        old = state.engine
        state.engine = mock_engine
        try:
            resp = client.post(
                "/api/chat/stream",
                json={"message": "hi"},
            )
            assert resp.status_code == 200
            _, kwargs = mock_engine.stream_chat.call_args
            assert "json_schema" not in kwargs, (
                "/api/chat/stream must NOT pass json_schema=None — "
                "legacy stream_generate signatures would TypeError")
        finally:
            state.engine = old


class TestChatImageWiring:
    """Visual-chat API contract: image paths must be explicit, forwarded,
    and never silently ignored on unsupported endpoints."""

    def test_chat_request_has_images_field(self):
        """ChatRequest exposes images as optional list[str] for visual chat."""
        from enigma_engine.api.server import ChatRequest

        fields = ChatRequest.model_fields
        assert "images" in fields, (
            "ChatRequest must expose images so /api/chat can accept visual context")
        req = ChatRequest(message="test")
        assert req.images is None
        req2 = ChatRequest(message="test", images=["data/avatar/images/default.png"])
        assert req2.images == ["data/avatar/images/default.png"]

    def test_appstate_chat_forwards_images(self):
        """AppState.chat must pass images through to engine.chat."""
        from unittest.mock import MagicMock
        from enigma_engine.api.server import AppState

        s = AppState()
        mock_eng = MagicMock()
        mock_eng.chat = MagicMock(return_value="reply")
        s.engine = mock_eng

        images = ["data/avatar/images/default.png"]
        s.chat("hi", images=images)
        _, kwargs = mock_eng.chat.call_args
        assert kwargs.get("images") == images, (
            "AppState.chat must forward images kwarg to engine.chat")

    def test_chat_endpoint_forwards_images(self, client):
        """POST /api/chat must forward images to engine.chat end-to-end."""
        from unittest.mock import MagicMock
        from enigma_engine.api.server import state

        mock_engine = MagicMock()
        mock_engine.chat = MagicMock(return_value="ok")
        old = state.engine
        state.engine = mock_engine
        try:
            images = ["data/avatar/images/default.png"]
            resp = client.post(
                "/api/chat",
                json={"message": "describe", "images": images},
            )
            assert resp.status_code == 200
            _, kwargs = mock_engine.chat.call_args
            assert kwargs.get("images") == images, (
                "/api/chat must forward images to engine.chat")
        finally:
            state.engine = old

    def test_chat_stream_rejects_images_until_supported(self, client):
        """/api/chat/stream must fail loud for images, not drop them silently."""
        from unittest.mock import MagicMock
        from enigma_engine.api.server import state

        mock_engine = MagicMock()
        mock_engine.stream_chat = MagicMock(return_value=iter(["ok"]))
        old = state.engine
        state.engine = mock_engine
        try:
            resp = client.post(
                "/api/chat/stream",
                json={
                    "message": "describe",
                    "images": ["data/avatar/images/default.png"],
                },
            )
            assert resp.status_code == 400
            body = resp.json()
            assert "images" in body.get("error", "").lower()
            assert "use /api/chat" in body.get("error", "")
            mock_engine.stream_chat.assert_not_called()
        finally:
            state.engine = old


class TestChatJsonSchemaBoundaryValidation:
    """Pass 156z9ac: malformed json_schema must return HTTP 400 at the
    boundary, NOT HTTP 500 wrapping a deep ValueError from inside
    generation. Same shape as the GUI ValueError-catch slice
    (Pass 156z9ab) applied to the API sibling boundary."""

    def test_chat_returns_400_when_schema_not_dict(self, client):
        from unittest.mock import MagicMock
        from enigma_engine.api.server import state
        mock_engine = MagicMock()
        mock_engine.chat = MagicMock(return_value="should-not-be-called")
        old = state.engine
        state.engine = mock_engine
        try:
            resp = client.post(
                "/api/chat",
                json={"message": "hi", "json_schema": "not-a-dict"},
            )
            # Pydantic may catch the type mismatch first (422) OR our
            # validator catches it (400). Both are acceptable
            # boundary-rejection codes; assert a real engine call
            # NEVER happened — that's the contract this test gates.
            assert resp.status_code in (400, 422)
            assert mock_engine.chat.call_count == 0, (
                "engine.chat must NOT be called when schema is "
                "structurally invalid")
        finally:
            state.engine = old

    def test_chat_returns_400_when_type_not_object(self, client):
        from unittest.mock import MagicMock
        from enigma_engine.api.server import state
        mock_engine = MagicMock()
        mock_engine.chat = MagicMock(return_value="should-not-be-called")
        old = state.engine
        state.engine = mock_engine
        try:
            resp = client.post(
                "/api/chat",
                json={
                    "message": "hi",
                    "json_schema": {"type": "array", "items": {}},
                },
            )
            assert resp.status_code == 400
            body = resp.json()
            assert "json_schema" in body.get("error", "").lower(), (
                "400 body must name json_schema so the user can find "
                "the offending field")
            assert mock_engine.chat.call_count == 0, (
                "engine.chat must NOT be called on validation failure")
        finally:
            state.engine = old

    def test_chat_stream_returns_400_when_schema_malformed(self, client):
        from unittest.mock import MagicMock
        from enigma_engine.api.server import state
        mock_engine = MagicMock()
        mock_engine.stream_chat = MagicMock(
            return_value=iter(["should-not-stream"]))
        old = state.engine
        state.engine = mock_engine
        try:
            resp = client.post(
                "/api/chat/stream",
                json={
                    "message": "hi",
                    "json_schema": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "no-such-type"},
                        },
                    },
                },
            )
            assert resp.status_code == 400
            assert mock_engine.stream_chat.call_count == 0, (
                "engine.stream_chat must NOT be called on validation "
                "failure — bad-schema requests must not block other "
                "clients on the inference lock")
        finally:
            state.engine = old

    def test_chat_400_on_bad_schema_does_not_acquire_inference_lock(
            self, client):
        """Pass 156z9ac contract: validation runs BEFORE the inference
        lock is acquired. A bad-schema request must not block a
        well-formed request that arrives microseconds later."""
        from unittest.mock import MagicMock
        from enigma_engine.api.server import state, _inference_lock
        mock_engine = MagicMock()
        mock_engine.chat = MagicMock(return_value="ok")
        old = state.engine
        state.engine = mock_engine
        try:
            # Send malformed schema
            resp = client.post(
                "/api/chat",
                json={"message": "hi", "json_schema": {"type": "array"}},
            )
            assert resp.status_code == 400
            # Lock must be available immediately (not held by the
            # bad-schema request).  acquire(blocking=False) returns
            # True iff the lock is free.
            acquired = _inference_lock.acquire(blocking=False)
            assert acquired, (
                "inference lock must be free after a 400 — bad-schema "
                "requests must not acquire it")
            if acquired:
                _inference_lock.release()
        finally:
            state.engine = old


class TestValidateJsonSchemaShape:
    """Pass 156z9ac: the extracted validator helper used by both the
    constraint constructor and the FastAPI boundary."""

    def test_accepts_minimal_object(self):
        from enigma_engine.core.json_schema_mask import (
            validate_json_schema_shape)
        # Should not raise
        validate_json_schema_shape({"type": "object", "properties": {}})

    def test_accepts_object_with_supported_types(self):
        from enigma_engine.core.json_schema_mask import (
            validate_json_schema_shape)
        validate_json_schema_shape({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
                "c": {"type": "boolean"},
                "d": {"type": "array"},
                "e": {"type": "null"},
            },
        })

    def test_rejects_non_dict(self):
        import pytest
        from enigma_engine.core.json_schema_mask import (
            validate_json_schema_shape)
        with pytest.raises(ValueError, match="must be a dict"):
            validate_json_schema_shape("not-a-dict")

    def test_rejects_non_object_root(self):
        import pytest
        from enigma_engine.core.json_schema_mask import (
            validate_json_schema_shape)
        with pytest.raises(ValueError, match="object"):
            validate_json_schema_shape(
                {"type": "array", "items": {}})

    def test_rejects_unsupported_property_type(self):
        import pytest
        from enigma_engine.core.json_schema_mask import (
            validate_json_schema_shape)
        with pytest.raises(ValueError, match="not supported"):
            validate_json_schema_shape({
                "type": "object",
                "properties": {"x": {"type": "no-such-type"}},
            })

    def test_constraint_constructor_still_validates(self):
        """Adversarial: deleting the validator call from
        ``JsonSchemaConstraint.__init__`` must regress this test —
        the constraint constructor MUST keep validating directly so
        Python callers (engine.generate, engine.chat) bypassing the
        API boundary still get the loud rejection.
        """
        import pytest
        from enigma_engine.core.json_schema_mask import (
            JsonSchemaConstraint)

        class _StubTok:
            vocab_size = 16

            def decode(self, ids, skip_special_tokens=False):
                return ""
        with pytest.raises(ValueError, match="object"):
            JsonSchemaConstraint(
                {"type": "array"}, _StubTok())


# ================================================================
# CF-10: History cap — _history must not grow without bound
# ================================================================

class TestHistoryCap:
    """AppState._history must be capped to prevent unbounded memory growth."""

    def test_history_has_max_constant(self):
        """MAX_HISTORY constant must exist in server module."""
        from enigma_engine.api import server
        assert hasattr(server, "MAX_HISTORY"), "MAX_HISTORY constant missing"
        assert isinstance(server.MAX_HISTORY, int)
        assert server.MAX_HISTORY > 0

    def test_history_capped_on_append(self):
        """Per-conversation history must evict oldest entries when cap is exceeded."""
        from enigma_engine.api.server import AppState, MAX_HISTORY
        s = AppState()
        cid = s.create_conversation()
        # Fill beyond cap (MC-1: per-conversation trim).
        for i in range(MAX_HISTORY + 20):
            with s._lock:
                conv = s._histories[cid]
                conv.append({"role": "user", "content": f"msg-{i}"})
                conv.append({"role": "assistant", "content": f"reply-{i}"})
                s._trim_history_locked(cid)
        conv = s._histories[cid]
        assert len(conv) <= MAX_HISTORY
        # Most recent entry should still be present
        assert conv[-1]["content"] == f"reply-{MAX_HISTORY + 19}"


class TestHistoryEndpoints:
    """Behavioral tests for /api/history endpoints."""

    def test_delete_history_clears_engine_history_and_kv_cache(self, client):
        from enigma_engine.api.server import state

        mock_engine = MagicMock()
        mock_engine.clear_history = MagicMock()
        mock_engine.clear_kv_cache = MagicMock()

        old_engine = state.engine
        old_histories = dict(state._histories)
        old_order = list(state._conv_order)
        old_active = state._active_conv_id
        state.engine = mock_engine
        # Seed one conversation so the nuke-everything route has
        # something to clear.
        cid = state.create_conversation()
        state._resolve_and_activate(cid)
        state._histories[cid].append({"role": "user", "content": "x"})
        try:
            resp = client.delete("/api/history")
            assert resp.status_code == 200
            assert resp.json().get("status") == "ok"
            assert state._histories == {}
            assert state._active_conv_id is None
            mock_engine.clear_history.assert_called_once()
            mock_engine.clear_kv_cache.assert_called_once()
        finally:
            state.engine = old_engine
            state._histories.clear()
            state._histories.update(old_histories)
            state._conv_order[:] = old_order
            state._active_conv_id = old_active

class TestTrainingModelSave:
    """ARCH-1d: server saves model to state.model_path after training (.pth only)."""

    def test_train_saves_pth_model_after_completion(self, client, monkeypatch, tmp_path):
        """After SFT training on a .pth model, atomic_torch_save is called."""
        from enigma_engine.api.server import _training_state, state

        saved: dict = {}

        def fake_run_training(config, ctx):
            return type("R", (), {"epoch": 1, "best_loss": 0.5})()

        def fake_atomic_save(data, path):
            saved["data"] = data
            saved["path"] = path

        class _SyncThread:
            def __init__(self, target, daemon=True):
                self._target = target
            def start(self):
                self._target()

        model_file = tmp_path / "my_model.pth"
        model_file.write_bytes(b"")

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        cfg = MagicMock()
        cfg.__class__ = type("ForgeConfig", (), {
            "__dataclass_fields__": {}
        })
        mock_model.config = cfg
        mock_engine = MagicMock()
        mock_engine.model = mock_model
        mock_engine.tokenizer = MagicMock()

        monkeypatch.setattr("enigma_engine.training.run_training", fake_run_training)
        monkeypatch.setattr("threading.Thread", _SyncThread)
        # atomic_torch_save is imported inline inside _run_training — patch the source module.
        monkeypatch.setattr("enigma_engine.core.safe_save.atomic_torch_save", fake_atomic_save)

        old_engine = state.engine
        old_path = state.model_path
        old_active = _training_state.get("active")
        state.engine = mock_engine
        state.model_path = str(model_file)
        _training_state["active"] = False
        try:
            resp = client.post(
                "/api/train",
                json={"mode": "sft", "data": "hello", "training": {"epochs": 1}},
            )
            assert resp.status_code == 200, resp.text
        finally:
            state.engine = old_engine
            state.model_path = old_path
            _training_state["active"] = old_active or False

        assert saved, "atomic_torch_save should have been called"
        assert saved["path"] == str(model_file)
        assert "model_state_dict" in saved["data"]

    def test_train_skips_save_for_gguf_model(self, client, monkeypatch, tmp_path):
        """Server must NOT call atomic_torch_save when model_path is a .gguf file."""
        from enigma_engine.api.server import _training_state, state

        saved: list = []

        def fake_run_training(config, ctx):
            return type("R", (), {"epoch": 1, "best_loss": 0.5})()

        def fake_atomic_save(data, path):
            saved.append(path)

        class _SyncThread:
            def __init__(self, target, daemon=True):
                self._target = target
            def start(self):
                self._target()

        gguf_file = tmp_path / "my_model.gguf"
        gguf_file.write_bytes(b"GGUF")

        mock_engine = MagicMock()
        mock_engine.model = MagicMock()
        mock_engine.tokenizer = MagicMock()

        monkeypatch.setattr("enigma_engine.training.run_training", fake_run_training)
        monkeypatch.setattr("threading.Thread", _SyncThread)
        monkeypatch.setattr("enigma_engine.core.safe_save.atomic_torch_save", fake_atomic_save)

        old_engine = state.engine
        old_path = state.model_path
        old_active = _training_state.get("active")
        state.engine = mock_engine
        state.model_path = str(gguf_file)
        _training_state["active"] = False
        try:
            resp = client.post(
                "/api/train",
                json={"mode": "sft", "data": "hello", "training": {"epochs": 1}},
            )
            assert resp.status_code == 200, resp.text
        finally:
            state.engine = old_engine
            state.model_path = old_path
            _training_state["active"] = old_active or False

        assert saved == [], "atomic_torch_save must not be called for GGUF models"

    def test_training_status_includes_step_fields(self, client, monkeypatch):
        """GET /api/training/status must expose step-level fields added in ARCH-1d."""

        resp = client.get("/api/training/status")
        assert resp.status_code == 200
        body = resp.json()
        for field in ("step", "total_steps", "lr", "tok_s", "best_loss",
                      "output_path", "abort_reason"):
            assert field in body, f"Missing field in training status: {field}"


# ---------------------------------------------------------------------------
# Style preferences (PERSONA-2 Slice 3)
# ---------------------------------------------------------------------------


class TestStylePreferencesEndpoints:
    """GET + PUT /api/style-preferences — Layer 2 of the layered-personality
    model. The trained core identity (Layer 1, LoRA weights) is NOT exposed
    through this endpoint and cannot be overridden via the style channel.
    """

    def test_get_returns_defaults_when_no_file(self, tmp_path, monkeypatch, client):
        """First-run / missing file → defaults returned, no error."""
        from enigma_engine.core import style_preferences as sp_module
        monkeypatch.setattr(
            sp_module,
            "STYLE_PREFERENCES_PATH",
            tmp_path / "no_such_file.json",
        )
        resp = client.get("/api/style-preferences")
        assert resp.status_code == 200
        body = resp.json()
        assert body["verbosity"] == "normal"
        assert body["formality"] == "neutral"
        assert body["default_response_length"] == "medium"
        assert body["prefer_code_examples"] is False
        assert body["prefer_bullet_points"] is False

    def test_put_then_get_roundtrip(self, tmp_path, monkeypatch, client):
        """PUT a partial update; subsequent GET returns the updated state."""
        from enigma_engine.core import style_preferences as sp_module
        prefs_path = tmp_path / "style.json"
        monkeypatch.setattr(sp_module, "STYLE_PREFERENCES_PATH", prefs_path)

        resp = client.put(
            "/api/style-preferences",
            json={"verbosity": "terse", "prefer_bullet_points": True},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["verbosity"] == "terse"
        assert body["prefer_bullet_points"] is True
        # Untouched fields must keep their default values
        assert body["formality"] == "neutral"
        assert body["default_response_length"] == "medium"
        assert body["prefer_code_examples"] is False

        # GET must reflect the PUT
        get_resp = client.get("/api/style-preferences")
        assert get_resp.status_code == 200
        assert get_resp.json() == body

    def test_put_partial_update_preserves_other_fields(
            self, tmp_path, monkeypatch, client):
        """PUT only verbosity → other previously-set fields stay intact."""
        from enigma_engine.core import style_preferences as sp_module
        prefs_path = tmp_path / "style.json"
        monkeypatch.setattr(sp_module, "STYLE_PREFERENCES_PATH", prefs_path)

        # First PUT: set verbosity + formality
        client.put(
            "/api/style-preferences",
            json={"verbosity": "verbose", "formality": "formal"},
        )
        # Second PUT: only change formality
        resp = client.put(
            "/api/style-preferences",
            json={"formality": "casual"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["formality"] == "casual"
        # verbosity must still be "verbose" from the first PUT
        assert body["verbosity"] == "verbose"

    def test_put_rejects_invalid_verbosity_enum(
            self, tmp_path, monkeypatch, client):
        """Invalid enum value → HTTP 422, NOT a silent identity override."""
        from enigma_engine.core import style_preferences as sp_module
        prefs_path = tmp_path / "style.json"
        monkeypatch.setattr(sp_module, "STYLE_PREFERENCES_PATH", prefs_path)

        resp = client.put(
            "/api/style-preferences",
            json={"verbosity": "pirate"},
        )
        assert resp.status_code == 422
        # Must mention verbosity so the caller knows what was wrong
        assert "verbosity" in resp.text.lower()

    def test_put_rejects_invalid_formality_enum(
            self, tmp_path, monkeypatch, client):
        from enigma_engine.core import style_preferences as sp_module
        prefs_path = tmp_path / "style.json"
        monkeypatch.setattr(sp_module, "STYLE_PREFERENCES_PATH", prefs_path)

        resp = client.put(
            "/api/style-preferences",
            json={"formality": "rude"},
        )
        assert resp.status_code == 422

    def test_put_rejects_unknown_field(self, tmp_path, monkeypatch, client):
        """``extra="forbid"`` blocks unknown top-level fields."""
        from enigma_engine.core import style_preferences as sp_module
        prefs_path = tmp_path / "style.json"
        monkeypatch.setattr(sp_module, "STYLE_PREFERENCES_PATH", prefs_path)

        resp = client.put(
            "/api/style-preferences",
            json={"verbosity": "terse", "secret_jailbreak": "anything"},
        )
        assert resp.status_code == 422

    def test_put_persists_to_disk(self, tmp_path, monkeypatch, client):
        """PUT must actually write the JSON file (load-able after)."""
        from enigma_engine.core import style_preferences as sp_module
        prefs_path = tmp_path / "style.json"
        monkeypatch.setattr(sp_module, "STYLE_PREFERENCES_PATH", prefs_path)

        client.put(
            "/api/style-preferences",
            json={"verbosity": "terse"},
        )
        assert prefs_path.exists()
        loaded = sp_module.load_style_preferences(prefs_path)
        assert loaded.verbosity == "terse"

    def test_put_does_not_touch_model_files(self, tmp_path, monkeypatch, client):
        """Black-box invariant at the API layer: a fake model file beside
        the preferences file must remain byte-identical after PUT."""
        from enigma_engine.core import style_preferences as sp_module
        prefs_path = tmp_path / "style.json"
        monkeypatch.setattr(sp_module, "STYLE_PREFERENCES_PATH", prefs_path)

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        fake_model = models_dir / "test.pth"
        fake_model.write_bytes(b"fake-model-bytes")
        before = fake_model.read_bytes()

        client.put(
            "/api/style-preferences",
            json={"verbosity": "verbose", "prefer_code_examples": True},
        )
        assert fake_model.read_bytes() == before


# ---------------------------------------------------------------------------
# /style chat command (PERSONA-2 Slice 4)
# ---------------------------------------------------------------------------


class TestStyleChatCommand:
    """``/style ...`` is intercepted at the AppState.chat() layer BEFORE the
    model is invoked. The command never appends to history and never reaches
    the engine. Per-conversation overrides are stored in
    ``_style_overrides_by_conversation`` and passed as ``style_overrides``
    kwarg to ``engine.chat`` on subsequent normal-chat turns."""

    def _make_state(self):
        """Fresh AppState with a mock engine."""
        from enigma_engine.api.server import AppState
        state = AppState()
        mock_engine = MagicMock()
        mock_engine.chat = MagicMock(return_value="ENGINE_REPLY")
        state.engine = mock_engine
        return state, mock_engine

    def test_style_alone_returns_current_state(self):
        state, mock_engine = self._make_state()
        reply, conv_id = state.chat("/style")
        assert "verbosity=" in reply
        assert "formality=" in reply
        # Engine MUST NOT have been called — /style is a control command
        mock_engine.chat.assert_not_called()

    def test_style_token_sets_field(self):
        state, mock_engine = self._make_state()
        reply, conv_id = state.chat("/style terse")
        assert "verbosity=terse" in reply.lower() or "set verbosity=terse" in reply.lower()
        mock_engine.chat.assert_not_called()
        # Override stored under this conversation id
        assert conv_id in state._style_overrides_by_conversation
        assert state._style_overrides_by_conversation[conv_id].verbosity == "terse"

    def test_style_reset_clears_overrides(self):
        state, mock_engine = self._make_state()
        _, conv_id = state.chat("/style terse")
        state.chat("/style reset", conversation_id=conv_id)
        # Override removed
        assert conv_id not in state._style_overrides_by_conversation

    def test_style_unknown_token_returns_error(self):
        state, mock_engine = self._make_state()
        reply, conv_id = state.chat("/style pirate")
        assert "unknown" in reply.lower() or "valid" in reply.lower()
        mock_engine.chat.assert_not_called()
        # No override was stored on the failed attempt
        assert conv_id not in state._style_overrides_by_conversation

    def test_style_command_does_not_append_to_history(self):
        """``/style ...`` is a control command, NOT chat content. The user
        message and the ack must NOT appear in conversation history."""
        state, mock_engine = self._make_state()
        _, conv_id = state.chat("/style terse")
        history = state.history_snapshot(conv_id)
        assert history == []

    def test_normal_chat_after_style_passes_override_to_engine(self):
        """After ``/style terse``, the next normal chat must pass
        ``style_overrides`` kwarg with a StylePreferences instance to
        ``engine.chat``."""
        from enigma_engine.core.style_preferences import StylePreferences
        state, mock_engine = self._make_state()
        _, conv_id = state.chat("/style terse")
        # Now a normal chat turn
        state.chat("hello", conversation_id=conv_id)
        # Verify engine.chat was called with style_overrides kwarg
        _, call_kwargs = mock_engine.chat.call_args
        assert "style_overrides" in call_kwargs
        assert isinstance(call_kwargs["style_overrides"], StylePreferences)
        assert call_kwargs["style_overrides"].verbosity == "terse"

    def test_normal_chat_with_no_override_omits_kwarg(self):
        """Conversation without any /style command → no style_overrides
        kwarg passed (engine uses disk-loaded preferences)."""
        state, mock_engine = self._make_state()
        state.chat("hello")
        _, call_kwargs = mock_engine.chat.call_args
        assert "style_overrides" not in call_kwargs

    def test_style_override_does_not_leak_across_conversations(self):
        """A /style command in conversation A must not affect conversation B."""
        state, mock_engine = self._make_state()
        _, conv_a = state.chat("/style terse")
        _, conv_b = state.chat("hello")  # auto-creates new conv
        assert conv_a != conv_b
        # conv_a has the override
        assert conv_a in state._style_overrides_by_conversation
        # conv_b does not
        assert conv_b not in state._style_overrides_by_conversation

    def test_delete_conversation_drops_style_override(self):
        """Deleting a conversation must purge its style override so a
        recycled id can't inherit stale state."""
        state, mock_engine = self._make_state()
        _, conv_id = state.chat("/style terse")
        assert conv_id in state._style_overrides_by_conversation
        state.delete_conversation(conv_id)
        assert conv_id not in state._style_overrides_by_conversation

    def test_clear_all_conversations_drops_all_style_overrides(self):
        state, mock_engine = self._make_state()
        _, conv_a = state.chat("/style terse")
        _, conv_b = state.chat("/style formal")
        assert len(state._style_overrides_by_conversation) == 2
        state.clear_all_conversations()
        assert state._style_overrides_by_conversation == {}

    def test_style_command_case_insensitive(self):
        """``/STYLE TERSE`` and ``/style terse`` are equivalent."""
        state, mock_engine = self._make_state()
        _, conv_id = state.chat("/STYLE TERSE")
        assert state._style_overrides_by_conversation[conv_id].verbosity == "terse"

    def test_style_command_with_leading_whitespace(self):
        """Leading whitespace before ``/style`` is tolerated."""
        state, mock_engine = self._make_state()
        reply, conv_id = state.chat("  /style verbose")
        mock_engine.chat.assert_not_called()
        assert state._style_overrides_by_conversation[conv_id].verbosity == "verbose"

    def test_style_boolean_tokens(self):
        """``bullets`` and ``code`` set the boolean preferences."""
        state, mock_engine = self._make_state()
        _, conv_id = state.chat("/style bullets")
        assert state._style_overrides_by_conversation[conv_id].prefer_bullet_points is True
        state.chat("/style code", conversation_id=conv_id)
        assert state._style_overrides_by_conversation[conv_id].prefer_code_examples is True
        # Both stuck (didn't overwrite each other)
        assert state._style_overrides_by_conversation[conv_id].prefer_bullet_points is True


class TestStyleChatCommandStreamPath:
    """F-A sibling-boundary closure: ``/api/chat/stream`` must intercept
    ``/style`` commands (no model call, no SSE stream, JSON ack only) and
    must forward per-conversation ``style_overrides`` kwarg on normal
    streaming turns. Sibling-boundary parity with ``state.chat()``."""

    def test_stream_intercepts_style_command(self, client):
        """``/style terse`` on the stream endpoint returns a JSON ack,
        NOT an SSE stream — engine.stream_chat is never called."""
        from enigma_engine.api.server import state
        mock_engine = MagicMock()
        mock_engine.stream_chat = MagicMock(return_value=iter(["nope"]))
        old_engine = state.engine
        # Clear conversation state for a clean test
        state.clear_all_conversations()
        state.engine = mock_engine
        try:
            resp = client.post(
                "/api/chat/stream",
                json={"message": "/style terse"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert "message" in body
            assert "conversation_id" in body
            # The engine MUST NOT have been called
            mock_engine.stream_chat.assert_not_called()
            # The override must have been stored
            conv_id = body["conversation_id"]
            assert conv_id in state._style_overrides_by_conversation
            assert (
                state._style_overrides_by_conversation[conv_id].verbosity
                == "terse"
            )
        finally:
            state.engine = old_engine
            state.clear_all_conversations()

    def test_stream_forwards_style_overrides_kwarg(self, client):
        """A conversation with an active ``/style`` override must pass
        ``style_overrides`` kwarg to ``engine.stream_chat`` on the next
        streaming turn."""
        from enigma_engine.api.server import state
        from enigma_engine.core.style_preferences import StylePreferences

        mock_engine = MagicMock()
        mock_engine.stream_chat = MagicMock(return_value=iter(["ok"]))
        old_engine = state.engine
        state.clear_all_conversations()
        state.engine = mock_engine
        try:
            # First: /style command to set the override
            resp1 = client.post(
                "/api/chat/stream",
                json={"message": "/style verbose"},
            )
            assert resp1.status_code == 200
            conv_id = resp1.json()["conversation_id"]

            # Second: normal streaming chat in the same conversation
            resp2 = client.post(
                "/api/chat/stream",
                json={"message": "hello", "conversation_id": conv_id},
            )
            assert resp2.status_code == 200
            # The engine.stream_chat MUST have been called with the kwarg
            _, call_kwargs = mock_engine.stream_chat.call_args
            assert "style_overrides" in call_kwargs
            assert isinstance(call_kwargs["style_overrides"], StylePreferences)
            assert call_kwargs["style_overrides"].verbosity == "verbose"
        finally:
            state.engine = old_engine
            state.clear_all_conversations()

    def test_stream_without_override_omits_style_overrides_kwarg(
            self, client):
        """Conversations with no /style command must NOT pass the kwarg."""
        from enigma_engine.api.server import state

        mock_engine = MagicMock()
        mock_engine.stream_chat = MagicMock(return_value=iter(["ok"]))
        old_engine = state.engine
        state.clear_all_conversations()
        state.engine = mock_engine
        try:
            resp = client.post(
                "/api/chat/stream",
                json={"message": "hello"},
            )
            assert resp.status_code == 200
            _, call_kwargs = mock_engine.stream_chat.call_args
            assert "style_overrides" not in call_kwargs
        finally:
            state.engine = old_engine
            state.clear_all_conversations()

    def test_stream_style_handler_exception_releases_inference_lock(
            self, client, monkeypatch):
        """F-Audit-2: if _handle_style_command raises mid-stream-request, the
        inference lock must be released — otherwise every subsequent request
        gets 429 'Engine busy' until restart (permanent lock leak)."""
        from enigma_engine.api import server as srv
        from enigma_engine.api.server import state

        mock_engine = MagicMock()
        mock_engine.stream_chat = MagicMock(return_value=iter(["ok"]))
        old_engine = state.engine
        state.clear_all_conversations()
        state.engine = mock_engine

        def _boom(message, conv_id):
            raise RuntimeError("simulated handler failure")

        monkeypatch.setattr(state, "_handle_style_command", _boom)
        try:
            # The request raises through — TestClient surfaces it as a 500
            # (or raises); either way the key assertion is the lock state.
            try:
                client.post("/api/chat/stream", json={"message": "hi"})
            except RuntimeError:
                pass
            # Lock MUST be free now — acquire must succeed immediately.
            acquired = srv._inference_lock.acquire(blocking=False)
            assert acquired, "inference lock leaked after handler exception"
            srv._inference_lock.release()
        finally:
            state.engine = old_engine
            state.clear_all_conversations()
            monkeypatch.undo()
