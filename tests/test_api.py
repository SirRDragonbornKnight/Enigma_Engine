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
        old_history = list(state._history)
        state.engine = mock_engine
        state._history.clear()
        try:
            client.post(
                "/api/chat/stream",
                json={"message": "test"},
            )
            assert len(state._history) == 2
            assert state._history[0]["role"] == "user"
            assert state._history[0]["content"] == "test"
            assert state._history[1]["role"] == "assistant"
            assert "Yes" in state._history[1]["content"]
        finally:
            state.engine = old_engine
            state._history[:] = old_history


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
        old_history = list(srv.state._history)
        srv.state.engine = mock_engine
        srv.state._history.clear()
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
            srv.state._history[:] = old_history

        assert "system_prompt" in captured
        assert "WEB RESEARCH" in captured["system_prompt"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTraining:
    """Test training status and trigger endpoints."""

    def test_training_status_idle(self, client):
        """Training status when no training is active."""
        resp = client.get("/api/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "active" in data
        assert data["active"] is False

    def test_train_no_model(self, client):
        """Triggering training without a model should fail."""
        resp = client.post(
            "/api/train",
            json={"data_file": "training.txt"},
        )
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data


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

    def test_train_data_file_too_long(self, client):
        """Training data_file path exceeding limit should be rejected."""
        from enigma_engine.api.server import state
        mock_engine = MagicMock()
        old = state.engine
        state.engine = mock_engine
        try:
            long_path = "a" * 300 + ".txt"
            resp = client.post("/api/train", json={"data_file": long_path})
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
        s.history.append({"role": "user", "content": "hi"})
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
        s.chat("hello")
        assert len(s.history) == 2
        assert s.history[0]["role"] == "user"
        assert s.history[1]["role"] == "assistant"


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

    def test_train_rejects_traversal(self, client):
        """Training with ../ data file should be rejected."""
        from enigma_engine.api.server import state
        mock_engine = MagicMock()
        old = state.engine
        state.engine = mock_engine
        try:
            resp = client.post(
                "/api/train",
                json={"data_file": "../../etc/passwd"},
            )
            assert resp.status_code == 403
        finally:
            state.engine = old

    def test_train_rejects_dot_dot_in_filename(self, client):
        """Training data_file with embedded .. should be rejected."""
        from enigma_engine.api.server import state
        mock_engine = MagicMock()
        old = state.engine
        state.engine = mock_engine
        try:
            resp = client.post(
                "/api/train",
                json={"data_file": "subdir/../../../secret.txt"},
            )
            assert resp.status_code in (403, 404)
        finally:
            state.engine = old

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
        """History must evict oldest entries when cap is exceeded."""
        from enigma_engine.api.server import AppState, MAX_HISTORY
        s = AppState()
        # Fill beyond cap
        for i in range(MAX_HISTORY + 20):
            with s._lock:
                s._history.append({"role": "user", "content": f"msg-{i}"})
                s._history.append({"role": "assistant", "content": f"reply-{i}"})
                s._trim_history()
        assert len(s._history) <= MAX_HISTORY
        # Most recent entry should still be present
        assert s._history[-1]["content"] == f"reply-{MAX_HISTORY + 19}"
