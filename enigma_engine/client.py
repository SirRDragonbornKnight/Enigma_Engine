"""Lightweight HTTP client for the local Enigma API server.

ARCH-1b: provides one canonical client surface for chat/model/training
operations so GUI/CLI can call the daemon over HTTP instead of importing
engine internals directly.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any


class EnigmaClient:
    """Minimal stdlib client for Enigma local API endpoints."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            try:
                err_obj = json.loads(detail)
                if isinstance(err_obj, dict) and "error" in err_obj:
                    detail = str(err_obj["error"])
            except json.JSONDecodeError:
                pass
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API unreachable at {url}: {exc.reason}") from exc

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"API returned non-JSON response: {raw[:200]}") from exc
        if not isinstance(obj, dict):
            raise RuntimeError(f"API returned non-object JSON: {raw[:200]}")
        return obj

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/health")

    def list_models(self) -> dict[str, Any]:
        return self._request("GET", "/api/models")

    def model_status(self) -> dict[str, Any]:
        return self._request("GET", "/api/models/status")

    def load_model(self, path: str) -> dict[str, Any]:
        return self._request("POST", "/api/models/load", {"path": path})

    def unload_model(self) -> dict[str, Any]:
        return self._request("POST", "/api/models/unload")

    def activate_profile(self, profile_id: str) -> dict[str, Any]:
        safe = urllib.parse.quote(profile_id, safe="")
        return self._request("POST", f"/api/profiles/{safe}/activate")

    def chat(
        self,
        message: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        web_access: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "message": message,
            "web_access": bool(web_access),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if repetition_penalty is not None:
            payload["repetition_penalty"] = float(repetition_penalty)
        if json_schema is not None:
            payload["json_schema"] = json_schema

        obj = self._request("POST", "/api/chat", payload)
        if "message" not in obj:
            raise RuntimeError(f"API /api/chat missing message field: {obj}")
        return str(obj["message"])

    def train(self, config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")
        return self._request("POST", "/api/train", config)

    def training_status(self) -> dict[str, Any]:
        return self._request("GET", "/api/training/status")

    def cancel_training(self) -> dict[str, Any]:
        return self._request("DELETE", "/api/training/cancel")

    def clear_history(self) -> dict[str, Any]:
        return self._request("DELETE", "/api/history")

    def chat_stream(
        self,
        message: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        web_access: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        payload: dict[str, Any] = {
            "message": message,
            "web_access": bool(web_access),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if repetition_penalty is not None:
            payload["repetition_penalty"] = float(repetition_penalty)
        if json_schema is not None:
            payload["json_schema"] = json_schema

        url = f"{self.base_url}/api/chat/stream"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                event_name = ""
                data_line = ""
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        if not data_line:
                            event_name = ""
                            continue
                        payload_obj = json.loads(data_line)
                        event = str(payload_obj.get("event", event_name or ""))
                        content = str(payload_obj.get("content", ""))
                        if event == "token":
                            yield content
                        elif event == "error":
                            raise RuntimeError(f"stream error: {content}")
                        elif event == "end":
                            return
                        event_name = ""
                        data_line = ""
                        continue

                    if line.startswith("event:"):
                        event_name = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        data_line = line[len("data:") :].strip()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API unreachable at {url}: {exc.reason}") from exc
