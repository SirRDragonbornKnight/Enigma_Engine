"""Tests for the stdlib HTTP Enigma client (ARCH-1b)."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from enigma_engine.client import EnigmaClient


class _Resp:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __iter__(self):
        return iter(io.BytesIO(self._body))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_chat_posts_message_and_returns_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def _fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["method"] = req.get_method()
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp(json.dumps({"message": "hello"}))

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    client = EnigmaClient("http://localhost:8080", timeout=12)
    out = client.chat("hi", top_k=30, web_access=True)

    assert out == "hello"
    assert seen["url"] == "http://localhost:8080/api/chat"
    assert seen["timeout"] == 12
    assert seen["method"] == "POST"
    assert seen["body"]["message"] == "hi"
    assert seen["body"]["top_k"] == 30
    assert seen["body"]["web_access"] is True


def test_chat_raises_on_missing_message_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: _Resp("{}"))
    client = EnigmaClient()
    with pytest.raises(RuntimeError, match="missing message"):
        client.chat("hi")


def test_http_error_uses_error_field_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    err_body = json.dumps({"error": "No model loaded"}).encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(
            req.full_url,
            503,
            "service unavailable",
            hdrs=None,
            fp=io.BytesIO(err_body),
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    client = EnigmaClient()
    with pytest.raises(RuntimeError, match="HTTP 503: No model loaded"):
        client.chat("hi")


def test_load_model_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def _fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp(json.dumps({"status": "ok", "model": {"loaded": True}}))

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    client = EnigmaClient("http://127.0.0.1:9000")
    out = client.load_model("models/smoke.pth")

    assert out["status"] == "ok"
    assert seen["url"] == "http://127.0.0.1:9000/api/models/load"
    assert seen["body"] == {"path": "models/smoke.pth"}


def test_activate_profile_url_encodes_profile_id(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def _fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        return _Resp(json.dumps({"status": "ok"}))

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    client = EnigmaClient("http://localhost:8080")
    out = client.activate_profile("dev/tools")

    assert out["status"] == "ok"
    assert seen["url"] == "http://localhost:8080/api/profiles/dev%2Ftools/activate"


def test_train_requires_dict_config() -> None:
    client = EnigmaClient()
    with pytest.raises(TypeError, match="config must be a dict"):
        client.train("not-a-dict")  # type: ignore[arg-type]


def test_train_posts_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def _fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp(json.dumps({"status": "started", "mode": "sft"}))

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    client = EnigmaClient()
    out = client.train({"mode": "sft", "data": "User: a\n\nAssistant: b"})

    assert out["status"] == "started"
    assert seen["url"].endswith("/api/train")
    assert seen["body"]["mode"] == "sft"


def test_chat_stream_yields_only_token_events(monkeypatch: pytest.MonkeyPatch) -> None:
    sse = (
        "event: start\n"
        'data: {"content":"","event":"start"}\n\n'
        "event: token\n"
        'data: {"content":"Hello","event":"token"}\n\n'
        "event: token\n"
        'data: {"content":" world","event":"token"}\n\n'
        "event: end\n"
        'data: {"content":"Hello world","event":"end"}\n\n'
    )

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: _Resp(sse))
    client = EnigmaClient()
    tokens = list(client.chat_stream("hi"))
    assert tokens == ["Hello", " world"]


def test_chat_stream_raises_on_error_event(monkeypatch: pytest.MonkeyPatch) -> None:
    sse = (
        "event: start\n"
        'data: {"content":"","event":"start"}\n\n'
        "event: error\n"
        'data: {"content":"boom","event":"error"}\n\n'
    )

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: _Resp(sse))
    client = EnigmaClient()
    with pytest.raises(RuntimeError, match="stream error: boom"):
        list(client.chat_stream("hi"))
