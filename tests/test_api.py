"""
Tests for the FastAPI local API server.

Tests the API endpoints, model management, chat, profiles, and system info.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


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

    def test_add_cors_middleware_function_exists(self):
        """The enable_cors() helper should exist for opt-in CORS."""
        from enigma_engine.api.server import enable_cors
        assert callable(enable_cors)

    def test_no_cors_middleware_at_import(self):
        """CORS middleware should not be registered at module level."""
        import ast
        from pathlib import Path
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
        import threading
        from enigma_engine.api.server import AppState
        s = AppState()
        mock_eng = MagicMock()
        mock_eng.chat = MagicMock(return_value="reply")
        s.engine = mock_eng
        s.chat("hello")
        assert len(s.history) == 2
        assert s.history[0]["role"] == "user"
        assert s.history[1]["role"] == "assistant"

    def test_training_state_has_lock(self):
        """_training_state access should be guarded by _training_lock."""
        import threading
        from enigma_engine.api import server
        assert hasattr(server, "_training_lock")
        assert isinstance(server._training_lock, type(threading.Lock()))


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

        secured_app = FastAPI()

        class _TestAPIKeyMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                if request.method == "OPTIONS":
                    return await call_next(request)
                if request.url.path.startswith("/api/"):
                    auth = request.headers.get("authorization", "")
                    if auth != f"Bearer {_key}":
                        return _JSONResponse(
                            {"error": "Invalid or missing API key"},
                            status_code=401,
                        )
                return await call_next(request)

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

    def test_api_key_middleware_defined_in_run_server(self):
        """run_server should install API key middleware when key is set."""
        import inspect
        from enigma_engine.api.server import run_server
        source = inspect.getsource(run_server)
        assert "api_key" in source
        assert "_APIKeyMiddleware" in source


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
