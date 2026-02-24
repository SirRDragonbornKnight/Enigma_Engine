"""
Tests for the FastAPI web server.

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
# Static / Frontend
# ---------------------------------------------------------------------------

class TestFrontend:
    """Test that the web frontend is served."""

    def test_index_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Enigma" in resp.text
