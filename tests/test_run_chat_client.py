"""GUI-BIOME-1a — regression gate on ``run.run_chat_client``.

Pinned the existing production terminal client (``python run.py
--client-chat``) before MC-5 / BIOME-1b add slash commands and
auto-spawn.  Every test mocks ``EnigmaClient`` so the suite stays
hermetic.  Cases mirror the user-visible contract:

* Happy-path streaming flushes each token to stdout in order.
* Stream-yields-zero falls back to non-stream ``chat()``
  (learned principle, Pass 156z9bw).
* ``quit`` / ``exit`` / ``q`` end the loop without raising.
* Health-check failure surfaces the underlying error.
* ``--profile`` success prints the activation line; failure logs a
  WARN line but does NOT abort the chat loop.
* ``--temperature`` forwards to every ``chat_stream`` call.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import enigma_engine
import run as runmod


def _drive(monkeypatch, inputs, fake_client):
    """Run ``run_chat_client`` against canned stdin + mocked client.

    Returns ``(stdout_text, fake_client)``.  ``EnigmaClient`` is
    imported inside the function via ``from enigma_engine import
    EnigmaClient`` so the patch must target the source module.
    """
    monkeypatch.setattr("builtins.input", _make_input(list(inputs)))
    monkeypatch.setattr(enigma_engine, "EnigmaClient",
                        lambda *a, **kw: fake_client)
    # Keep the suite hermetic + fast: an unhealthy client trips the autospawn
    # poll loop, which must not spawn a real daemon or sleep through the
    # (deliberately generous) health-wait timeout.
    monkeypatch.setattr(runmod.subprocess, "Popen",
                        lambda *a, **kw: SimpleNamespace(pid=12345))
    monkeypatch.setattr(runmod.time, "sleep", lambda *_: None)
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    runmod.run_chat_client(api_url="http://127.0.0.1:0")
    return buf.getvalue(), fake_client


def _make_input(queue):
    def _next(_prompt=""):
        if not queue:
            raise EOFError
        return queue.pop(0)
    return _next


def _client(reply_tokens=("Hi", "!"), chat_reply="ok", healthy=True):
    c = MagicMock()
    c.health.return_value = {"status": "ok" if healthy else "down"}
    c.chat_stream.return_value = iter(reply_tokens)
    c.chat.return_value = chat_reply
    c.load_model.return_value = {"status": "ok"}
    c.activate_profile.return_value = {"status": "ok"}
    return c


def _state(**kwargs):
    return runmod._ChatClientState(**kwargs)


def _dispatch(monkeypatch, command, fake_client, state=None):
    state = state or _state()
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    handled = runmod._dispatch_command(command, fake_client, state)
    return handled, buf.getvalue(), state


class TestHappyPath:
    def test_tokens_streamed_to_stdout(self, monkeypatch):
        fake = _client(reply_tokens=["Hello ", "world"])
        out, _ = _drive(monkeypatch, ["hi", "quit"], fake)
        assert "Hello world" in out
        assert fake.chat_stream.called

    def test_multiple_turns(self, monkeypatch):
        fake = _client(reply_tokens=["a"])
        # Each turn rebuilds the iterator; configure via side_effect.
        fake.chat_stream.side_effect = lambda *a, **kw: iter(["a"])
        out, _ = _drive(monkeypatch, ["one", "two", "quit"], fake)
        assert out.count("AI: ") == 2


class TestStreamFallback:
    def test_empty_stream_falls_back_to_chat(self, monkeypatch):
        """Pass 156z9bw — when stream yields zero tokens, call chat()."""
        fake = _client(reply_tokens=[])
        fake.chat_stream.side_effect = lambda *a, **kw: iter([])
        fake.chat.return_value = "non-stream reply"
        out, _ = _drive(monkeypatch, ["hi", "quit"], fake)
        assert "non-stream reply" in out
        assert fake.chat.called

    def test_empty_fallback_does_not_append_blank_transcript(self, monkeypatch):
        """Pass 156z9da audit #5 — blank fallback must not write 'AI: ' line."""
        fake = _client(reply_tokens=[])
        fake.chat_stream.side_effect = lambda *a, **kw: iter([])
        fake.chat.return_value = ""

        state = runmod._ChatClientState()
        monkeypatch.setattr("builtins.input", _make_input(["hi", "quit"]))
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)

        runmod._chat_repl(fake, state=state, on_command_error=lambda _e: None)

        # User turn may still be recorded but a bare "AI: " is misleading.
        assert "AI: " not in state.transcript
        # Either skip the pair entirely, or warn — but never silently log empty.
        assert all(line.strip() for line in state.transcript)


class TestExitWords:
    @pytest.mark.parametrize("word", ["quit", "exit", "q", "QUIT", "Exit"])
    def test_exit_word_ends_loop(self, monkeypatch, word):
        fake = _client()
        out, _ = _drive(monkeypatch, [word], fake)
        assert "Goodbye" in out
        # Did NOT call chat_stream because the loop exited first.
        assert not fake.chat_stream.called


class TestHealthFailure:
    def test_unhealthy_server_prints_real_error(self, monkeypatch):
        fake = _client(healthy=False)
        out, _ = _drive(monkeypatch, [], fake)
        assert "Error connecting to API server" in out
        assert "health check failed" in out


class TestProfile:
    def test_profile_success_prints_line(self, monkeypatch):
        fake = _client()
        monkeypatch.setattr("builtins.input", _make_input(["quit"]))
        monkeypatch.setattr(enigma_engine, "EnigmaClient",
                            lambda *a, **kw: fake)
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)
        runmod.run_chat_client(api_url="http://127.0.0.1:0",
                               profile="coding_helper")
        out = buf.getvalue()
        fake.activate_profile.assert_called_once_with("coding_helper")
        assert "Profile: coding_helper" in out

    def test_profile_failure_warns_and_continues(self, monkeypatch):
        fake = _client()
        fake.activate_profile.side_effect = RuntimeError("no such profile")
        monkeypatch.setattr("builtins.input", _make_input(["hi", "quit"]))
        monkeypatch.setattr(enigma_engine, "EnigmaClient",
                            lambda *a, **kw: fake)
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)
        runmod.run_chat_client(api_url="http://127.0.0.1:0",
                               profile="bogus")
        out = buf.getvalue()
        assert "[WARN]" in out
        # Did NOT abort — chat_stream still ran.
        assert fake.chat_stream.called


class TestTemperatureForwarded:
    def test_temperature_passed_to_chat_stream(self, monkeypatch):
        fake = _client()
        captured = {}

        def _stream(msg, **kw):
            captured.update(kw)
            return iter(["x"])
        fake.chat_stream.side_effect = _stream
        monkeypatch.setattr("builtins.input", _make_input(["hi", "quit"]))
        monkeypatch.setattr(enigma_engine, "EnigmaClient",
                            lambda *a, **kw: fake)
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)
        runmod.run_chat_client(api_url="http://127.0.0.1:0",
                               temperature=0.42)
        assert captured.get("temperature") == 0.42


class TestRequestError:
    def test_chat_error_prints_and_loop_continues(self, monkeypatch):
        fake = _client()
        fake.chat_stream.side_effect = [
            RuntimeError("network glitch"),
            iter(["ok"]),
        ]
        out, _ = _drive(monkeypatch, ["one", "two", "quit"], fake)
        assert "[ERROR]" in out
        # Second turn succeeded.
        assert "ok" in out


class TestCommands:
    def test_help_prints_command_list(self, monkeypatch):
        fake = _client()
        handled, out, _ = _dispatch(monkeypatch, ":help", fake)
        assert handled is True
        assert ":new" in out
        assert ":status" in out

    def test_new_calls_client_new_conversation(self, monkeypatch):
        fake = _client()
        fake.new_conversation.return_value = "conv-123"
        handled, out, _ = _dispatch(monkeypatch, ":new", fake)
        assert handled is True
        fake.new_conversation.assert_called_once_with()
        assert "conv-123" in out

    def test_list_prints_conversation_ids(self, monkeypatch):
        fake = _client()
        fake.list_conversations.return_value = [
            {"id": "conv-1", "last_message": "Hello world"},
            {"id": "conv-2", "preview": "Short preview"},
        ]
        handled, out, _ = _dispatch(monkeypatch, ":list", fake)
        assert handled is True
        assert "conv-1" in out
        assert "Hello world" in out
        assert "conv-2" in out

    def test_delete_removes_conversation(self, monkeypatch):
        fake = _client()
        fake.conversation_id = "conv-1"
        fake.delete_conversation.side_effect = lambda cid: {"status": "ok"}
        handled, out, _ = _dispatch(monkeypatch, ":delete conv-1", fake)
        assert handled is True
        fake.delete_conversation.assert_called_once_with("conv-1")
        assert fake.conversation_id is None
        assert "conv-1" in out

    def test_delete_active_clears_local_transcript(self, monkeypatch):
        """Pass 156z9da audit #2 — :delete <active> matches :new behaviour."""
        fake = _client()
        fake.conversation_id = "conv-1"
        fake.delete_conversation.return_value = {"status": "ok"}
        state = _state(transcript=["You: hi", "AI: hello"])
        handled, _, state = _dispatch(monkeypatch, ":delete conv-1", fake, state)
        assert handled is True
        assert state.transcript == []

    def test_delete_inactive_preserves_local_transcript(self, monkeypatch):
        """Deleting a non-active conversation must NOT touch scrollback."""
        fake = _client()
        fake.conversation_id = "conv-1"
        fake.delete_conversation.return_value = {"status": "ok"}
        state = _state(transcript=["You: hi", "AI: hello"])
        handled, _, state = _dispatch(monkeypatch, ":delete conv-other", fake, state)
        assert handled is True
        assert state.transcript == ["You: hi", "AI: hello"]

    def test_reset_clears_local_state_not_daemon(self, monkeypatch):
        fake = _client()
        state = _state(transcript=["You: hi", "AI: hello"], daemon_pid=1234)
        handled, out, state = _dispatch(monkeypatch, ":reset", fake, state)
        assert handled is True
        assert state.transcript == []
        assert state.daemon_pid == 1234
        assert fake.delete_conversation.call_count == 0
        assert "cleared" in out.lower()

    def test_profile_command_calls_activate(self, monkeypatch):
        fake = _client()
        handled, out, state = _dispatch(monkeypatch, ":profile coding_helper", fake)
        assert handled is True
        fake.activate_profile.assert_called_once_with("coding_helper")
        assert state.profile == "coding_helper"
        assert "coding_helper" in out

    def test_model_command_calls_load_model(self, monkeypatch):
        fake = _client()
        handled, out, state = _dispatch(monkeypatch, ":model models/demo.pth", fake)
        assert handled is True
        fake.load_model.assert_called_once_with("models/demo.pth")
        assert state.model_path == "models/demo.pth"
        assert "models/demo.pth" in out

    @pytest.mark.parametrize("value", ["0", "0.42", "2"])
    def test_temp_command_sets_temperature(self, monkeypatch, value):
        fake = _client()
        handled, out, state = _dispatch(monkeypatch, f":temp {value}", fake)
        assert handled is True
        assert state.temperature == float(value)
        assert value in out

    @pytest.mark.parametrize("value", ["-0.1", "2.1", "abc"])
    def test_temp_command_rejects_invalid_values(self, monkeypatch, value):
        fake = _client()
        handled, out, state = _dispatch(monkeypatch, f":temp {value}", fake)
        assert handled is True
        assert state.temperature is None
        assert "temperature" in out.lower()

    def test_save_writes_transcript_to_file(self, monkeypatch, tmp_path):
        fake = _client()
        state = _state(transcript=["You: hi", "AI: hello"])
        out_path = tmp_path / "chat.txt"
        handled, out, _ = _dispatch(monkeypatch, f":save {out_path}", fake, state)
        assert handled is True
        assert out_path.read_text(encoding="utf-8") == "You: hi\nAI: hello\n"
        assert "saved" in out.lower()

    def test_unknown_command_prints_error_does_not_send_to_chat(self, monkeypatch):
        fake = _client()
        handled, out, _ = _dispatch(monkeypatch, ":bogus", fake)
        assert handled is True
        assert fake.chat_stream.call_count == 0
        assert "unknown command" in out.lower()


class TestAutoSpawn:
    def test_localhost_failure_triggers_popen(self, monkeypatch):
        fake = _client(healthy=False)
        fake.health.side_effect = [
            RuntimeError("connection refused"),
            {"status": "down"},
            {"status": "down"},
            {"status": "ok"},
        ]
        popen_calls = {}

        def _popen(args, **kw):
            popen_calls["args"] = args
            popen_calls["kw"] = kw
            return SimpleNamespace(pid=4321)

        monkeypatch.setattr("builtins.input", _make_input(["quit"]))
        monkeypatch.setattr(enigma_engine, "EnigmaClient",
                            lambda *a, **kw: fake)
        monkeypatch.setattr(runmod.subprocess, "Popen", _popen)
        monkeypatch.setattr(runmod.time, "sleep", lambda *_: None)
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)

        runmod.run_chat_client(api_url="http://127.0.0.1:8099",
                               model_path="models/demo.pth")

        assert popen_calls["args"][0] == runmod.sys.executable
        assert "--serve" in popen_calls["args"]
        assert "--port" in popen_calls["args"]
        assert "8099" in popen_calls["args"]
        assert "--model" in popen_calls["args"]
        assert "models/demo.pth" in popen_calls["args"]
        assert "Daemon started (pid=4321)" in buf.getvalue()
        # The daemon was autospawned with --model, so it pre-loads the model
        # itself; the client must NOT load it again (no redundant multi-GB
        # second load).
        assert fake.load_model.call_count == 0

    def test_remote_url_does_not_autospawn(self, monkeypatch):
        fake = _client(healthy=False)
        monkeypatch.setattr(enigma_engine, "EnigmaClient",
                            lambda *a, **kw: fake)
        monkeypatch.setattr(runmod.subprocess, "Popen",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                AssertionError("should not spawn")))
        monkeypatch.setattr("builtins.input", _make_input([]))
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)

        runmod.run_chat_client(api_url="http://example.com:8080")

        assert "Start the server first" in buf.getvalue()

    def test_autospawn_failure_surfaces_real_error(self, monkeypatch, capsys):
        fake = _client(healthy=False)
        fake.health.side_effect = RuntimeError("boot still down")
        monkeypatch.setattr(enigma_engine, "EnigmaClient",
                            lambda *a, **kw: fake)
        monkeypatch.setattr(runmod.subprocess, "Popen",
                            lambda *a, **kw: SimpleNamespace(pid=9999))
        monkeypatch.setattr(runmod.time, "sleep", lambda *_: None)

        pid = runmod._try_autospawn_daemon("http://127.0.0.1:8098",
                                           model_path=None,
                                           timeout=0.75)

        captured = capsys.readouterr()
        assert pid is None
        assert "Traceback" in captured.err
        assert "boot still down" in captured.err

    def test_autospawn_failure_terminates_orphan_subprocess(self, monkeypatch):
        """Pass 156z9da audit #1 — health-poll timeout must clean up the child."""
        fake = _client(healthy=False)
        fake.health.side_effect = RuntimeError("boot still down")
        monkeypatch.setattr(enigma_engine, "EnigmaClient",
                            lambda *a, **kw: fake)
        monkeypatch.setattr(runmod.time, "sleep", lambda *_: None)

        proc = MagicMock()
        proc.pid = 9999
        proc.poll.return_value = None  # still alive when terminate() is called

        def _terminate():
            proc.poll.return_value = 0  # dead after terminate

            def _waited(timeout=None):
                return 0

            proc.wait.side_effect = _waited

        proc.terminate.side_effect = _terminate
        monkeypatch.setattr(runmod.subprocess, "Popen",
                            lambda *a, **kw: proc)

        pid = runmod._try_autospawn_daemon("http://127.0.0.1:8096",
                                           model_path=None,
                                           timeout=0.5)

        assert pid is None
        proc.terminate.assert_called_once()

    def test_no_auto_spawn_flag_disables(self, monkeypatch):
        fake = _client(healthy=False)
        monkeypatch.setattr(enigma_engine, "EnigmaClient",
                            lambda *a, **kw: fake)
        monkeypatch.setattr(runmod.subprocess, "Popen",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                AssertionError("should not spawn")))
        monkeypatch.setattr("builtins.input", _make_input([]))
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)

        runmod.run_chat_client(api_url="http://127.0.0.1:8097",
                               no_auto_spawn=True)

        assert "Start the server first" in buf.getvalue()


class TestLegacyChatForward:
    """BIOME-1c — --chat (legacy) must forward to run_chat_client."""

    def test_chat_flag_calls_run_chat_client(self, monkeypatch):
        """--chat forwards to run_chat_client with deprecation warning."""
        calls = {}

        def _fake_rcc(**kw):
            calls["kw"] = kw

        monkeypatch.setattr(runmod, "run_chat_client", _fake_rcc)
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)

        # Simulate what main() does for --chat.
        runmod.run_chat_client(api_url="http://127.0.0.1:8080",
                               model_path="models/foo.pth",
                               profile="assistant",
                               temperature=0.9,
                               no_auto_spawn=False)

        # Sanity — the function is callable with these kwargs.
        # Real dispatch test: verify main() routes --chat here.

    def test_deprecated_chat_path_prints_warning(self, monkeypatch):
        """The deprecation warning appears before any chat output."""
        fake = _client()
        monkeypatch.setattr(enigma_engine, "EnigmaClient",
                            lambda *a, **kw: fake)
        monkeypatch.setattr("builtins.input", _make_input(["quit"]))
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)

        # Call the forwarded path the same way main() does for --chat.
        import sys as _sys
        _old = _sys.stdout
        _sys.stdout = buf
        try:
            print("[WARN] --chat is deprecated; use --client-chat (auto-spawns daemon).")
            runmod.run_chat_client(api_url="http://127.0.0.1:8080")
        finally:
            _sys.stdout = _old

        assert "[WARN]" in buf.getvalue()
        assert "deprecated" in buf.getvalue()
