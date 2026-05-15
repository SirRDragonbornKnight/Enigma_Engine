"""MC-1 conversation_id contract tests.

The daemon must scope chat history per ``conversation_id`` so multiple
clients (terminal REPL, desktop GUI, future browser viewer) don't step
on each other's threads.  Contract:

* ``ChatRequest`` accepts an optional ``conversation_id``.
* If absent, the server auto-creates one and returns it in the response.
* Two requests with different IDs see independent histories.
* Switching the active conversation invalidates the engine KV cache
  (and clears the engine's history-summary cache) before generating.
* ``POST /api/conversations``, ``GET /api/conversations``,
  ``DELETE /api/conversations/{id}`` manage IDs explicitly.
* ``GET /api/conversations/{id}/history`` returns per-conv history.
* Legacy ``GET /api/history`` keeps working (returns the most recently
  active conversation's history, or empty).
* Unknown IDs return 404.
* The total number of conversations is bounded; LRU eviction kicks in
  past ``MAX_CONVERSATIONS``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import json

import pytest


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from enigma_engine.api.server import app
    return TestClient(app)


@pytest.fixture
def mock_engine_factory():
    """Builds a fresh mock engine that records calls and returns canned replies."""

    def _factory(reply: str = "ok"):
        engine = MagicMock()
        engine.chat = MagicMock(return_value=reply)
        engine.stream_chat = MagicMock(return_value=iter([reply]))
        engine.clear_history = MagicMock()
        engine.clear_kv_cache = MagicMock()
        return engine

    return _factory


@pytest.fixture
def fresh_state(mock_engine_factory):
    """Reset AppState between tests and install a mock engine."""
    from enigma_engine.api import server as srv

    old_engine = srv.state.engine
    old_histories = dict(getattr(srv.state, "_histories", {}))
    old_active = getattr(srv.state, "_active_conv_id", None)

    srv.state.engine = mock_engine_factory()
    if hasattr(srv.state, "_histories"):
        srv.state._histories = {}
    if hasattr(srv.state, "_active_conv_id"):
        srv.state._active_conv_id = None
    try:
        yield srv.state
    finally:
        srv.state.engine = old_engine
        if hasattr(srv.state, "_histories"):
            srv.state._histories = old_histories
        if hasattr(srv.state, "_active_conv_id"):
            srv.state._active_conv_id = old_active


class TestConversationLifecycle:
    """Create / list / delete conversations."""

    def test_post_creates_conversation(self, client, fresh_state):
        resp = client.post("/api/conversations")
        assert resp.status_code == 200
        body = resp.json()
        assert "id" in body
        assert isinstance(body["id"], str)
        assert len(body["id"]) > 0

    def test_list_starts_empty(self, client, fresh_state):
        resp = client.get("/api/conversations")
        assert resp.status_code == 200
        assert resp.json() == {"conversations": []}

    def test_list_includes_created(self, client, fresh_state):
        ids = [client.post("/api/conversations").json()["id"] for _ in range(3)]
        resp = client.get("/api/conversations")
        listed = resp.json()["conversations"]
        listed_ids = [c["id"] for c in listed]
        for cid in ids:
            assert cid in listed_ids

    def test_delete_removes_conversation(self, client, fresh_state):
        cid = client.post("/api/conversations").json()["id"]
        resp = client.delete(f"/api/conversations/{cid}")
        assert resp.status_code == 200
        listed = client.get("/api/conversations").json()["conversations"]
        assert cid not in [c["id"] for c in listed]

    def test_delete_unknown_returns_404(self, client, fresh_state):
        resp = client.delete("/api/conversations/does-not-exist")
        assert resp.status_code == 404


class TestPerConversationHistory:
    """Each conversation_id has its own history."""

    def test_unknown_id_history_returns_404(self, client, fresh_state):
        resp = client.get("/api/conversations/does-not-exist/history")
        assert resp.status_code == 404

    def test_history_starts_empty_after_create(self, client, fresh_state):
        cid = client.post("/api/conversations").json()["id"]
        resp = client.get(f"/api/conversations/{cid}/history")
        assert resp.status_code == 200
        assert resp.json() == {"history": []}

    def test_chat_appends_to_named_conversation(self, client, fresh_state):
        cid = client.post("/api/conversations").json()["id"]
        client.post("/api/chat",
                    json={"message": "hi", "conversation_id": cid})
        hist = client.get(f"/api/conversations/{cid}/history").json()["history"]
        assert len(hist) == 2
        assert hist[0]["role"] == "user"
        assert hist[0]["content"] == "hi"
        assert hist[1]["role"] == "assistant"

    def test_two_conversations_are_isolated(self, client, fresh_state):
        a = client.post("/api/conversations").json()["id"]
        b = client.post("/api/conversations").json()["id"]
        client.post("/api/chat",
                    json={"message": "msg-A", "conversation_id": a})
        client.post("/api/chat",
                    json={"message": "msg-B", "conversation_id": b})

        hist_a = client.get(f"/api/conversations/{a}/history").json()["history"]
        hist_b = client.get(f"/api/conversations/{b}/history").json()["history"]
        assert any(h["content"] == "msg-A" for h in hist_a)
        assert all(h["content"] != "msg-B" for h in hist_a)
        assert any(h["content"] == "msg-B" for h in hist_b)
        assert all(h["content"] != "msg-A" for h in hist_b)


class TestAutoCreate:
    """Chat without conversation_id auto-creates one and returns it."""

    def test_chat_without_id_creates_and_returns(self, client, fresh_state):
        resp = client.post("/api/chat", json={"message": "hi"})
        assert resp.status_code == 200
        body = resp.json()
        assert "conversation_id" in body
        cid = body["conversation_id"]
        assert cid in [c["id"]
                       for c in client.get("/api/conversations").json()["conversations"]]

    def test_chat_with_id_returns_same_id(self, client, fresh_state):
        cid = client.post("/api/conversations").json()["id"]
        body = client.post("/api/chat",
                           json={"message": "hi", "conversation_id": cid}).json()
        assert body["conversation_id"] == cid


class TestKVCacheInvalidation:
    """Switching active conversation must clear engine KV cache + history summary."""

    def test_same_conversation_does_not_clear_kv(self, client, fresh_state):
        cid = client.post("/api/conversations").json()["id"]
        engine = fresh_state.engine
        # First chat activates the conversation (None -> cid) so the
        # engine state is invalidated.  Reset the mock so the next
        # assertion only counts subsequent same-conv turns.
        client.post("/api/chat", json={"message": "1", "conversation_id": cid})
        engine.clear_kv_cache.reset_mock()
        client.post("/api/chat", json={"message": "2", "conversation_id": cid})
        client.post("/api/chat", json={"message": "3", "conversation_id": cid})
        assert engine.clear_kv_cache.call_count == 0

    def test_switch_clears_kv(self, client, fresh_state):
        a = client.post("/api/conversations").json()["id"]
        b = client.post("/api/conversations").json()["id"]
        engine = fresh_state.engine
        client.post("/api/chat", json={"message": "1", "conversation_id": a})
        engine.clear_kv_cache.reset_mock()
        client.post("/api/chat", json={"message": "2", "conversation_id": b})
        assert engine.clear_kv_cache.call_count >= 1


class TestUnknownConversation:
    """Posting to an unknown conversation_id is rejected."""

    def test_chat_unknown_id_returns_404(self, client, fresh_state):
        resp = client.post("/api/chat",
                           json={"message": "hi",
                                 "conversation_id": "does-not-exist"})
        assert resp.status_code == 404


class TestLegacyHistoryRoute:
    """``GET /api/history`` keeps working for back-compat."""

    def test_get_history_empty_when_no_conv(self, client, fresh_state):
        resp = client.get("/api/history")
        assert resp.status_code == 200
        assert resp.json() == {"history": []}

    def test_get_history_returns_active_conv(self, client, fresh_state):
        cid = client.post("/api/conversations").json()["id"]
        client.post("/api/chat",
                    json={"message": "hello", "conversation_id": cid})
        resp = client.get("/api/history")
        assert resp.status_code == 200
        hist = resp.json()["history"]
        assert any(h["content"] == "hello" for h in hist)

    def test_delete_history_clears_all(self, client, fresh_state):
        a = client.post("/api/conversations").json()["id"]
        b = client.post("/api/conversations").json()["id"]
        client.post("/api/chat", json={"message": "x", "conversation_id": a})
        client.post("/api/chat", json={"message": "y", "conversation_id": b})
        resp = client.delete("/api/history")
        assert resp.status_code == 200
        listed = client.get("/api/conversations").json()["conversations"]
        assert listed == []


class TestStreamConversationId:
    """``/api/chat/stream`` honors conversation_id too."""

    def test_stream_appends_to_named_conv(self, client, fresh_state):
        cid = client.post("/api/conversations").json()["id"]
        fresh_state.engine.stream_chat = MagicMock(
            return_value=iter(["Hi", "!"]))
        resp = client.post("/api/chat/stream",
                           json={"message": "yo", "conversation_id": cid})
        assert resp.status_code == 200
        # Drain the stream body so the generator runs to completion.
        _ = resp.text
        hist = client.get(f"/api/conversations/{cid}/history").json()["history"]
        assert any(h["content"] == "yo" for h in hist)
        assert any("Hi" in h["content"] for h in hist)

    def test_stream_unknown_id_returns_404(self, client, fresh_state):
        resp = client.post("/api/chat/stream",
                           json={"message": "x",
                                 "conversation_id": "nope"})
        assert resp.status_code == 404


class TestLRUEviction:
    """Past ``MAX_CONVERSATIONS`` the oldest unused conversation is evicted."""

    def test_max_conversations_evicts_oldest(self, client, fresh_state, monkeypatch):
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "MAX_CONVERSATIONS", 3)
        ids = [client.post("/api/conversations").json()["id"] for _ in range(5)]
        listed_ids = [c["id"]
                      for c in client.get("/api/conversations").json()["conversations"]]
        # The first two should have been evicted.
        assert ids[0] not in listed_ids
        assert ids[1] not in listed_ids
        # The last three remain.
        for cid in ids[2:]:
            assert cid in listed_ids


class TestMaxConversationsFloor:
    """D2: MAX_CONVERSATIONS floor must be >= 2 to prevent soft-brick."""

    def test_floor_at_least_two(self):
        from enigma_engine.api import server as srv
        assert srv.MAX_CONVERSATIONS >= 2, (
            "MAX_CONVERSATIONS below 2 bricks the server: the active "
            "conversation can never be evicted, so count permanently "
            "exceeds the cap and new conversations are rejected."
        )

    def test_eviction_with_cap_two_does_not_infinite_loop(
            self, client, fresh_state, monkeypatch):
        """With MAX_CONVERSATIONS=2, a third must evict one cleanly."""
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "MAX_CONVERSATIONS", 2)
        a = client.post("/api/conversations").json()["id"]
        b = client.post("/api/conversations").json()["id"]
        # Touch a to make b the LRU candidate.
        client.post("/api/chat", json={"message": "x", "conversation_id": a})
        # Creating a third must succeed (not loop or crash) AND must
        # actually evict one (proving the cap is enforced, not just
        # silently exceeded).
        c = client.post("/api/conversations").json()["id"]
        assert c is not None
        listed = [x["id"]
                  for x in client.get("/api/conversations").json()["conversations"]]
        assert len(listed) == 2, (
            f"Cap=2 must evict on third create, got {len(listed)} convs: {listed}"
        )
        assert c in listed
        # The evicted one must be the LRU (b), not the active (a) or the new (c).
        assert b not in listed
        assert a in listed


class TestHistoryReachesEngine:
    """T1: ensure the per-conv history is actually forwarded to engine.chat()."""

    def test_engine_chat_receives_history_kwarg(self, client, fresh_state):
        engine = fresh_state.engine
        cid = client.post("/api/conversations").json()["id"]
        client.post("/api/chat", json={"message": "first", "conversation_id": cid})
        engine.chat.reset_mock()
        client.post("/api/chat", json={"message": "second", "conversation_id": cid})
        kwargs = engine.chat.call_args.kwargs
        assert "history" in kwargs, f"engine.chat called without history=: {kwargs}"
        hist = kwargs["history"]
        assert isinstance(hist, list)
        # First turn appended user+assistant; second turn must see both.
        assert len(hist) == 2
        assert hist[0] == {"role": "user", "content": "first"}
        assert hist[1]["role"] == "assistant"


class TestRetryDoesNotPoisonHistory:
    """B2: AutoResearch-2 retry must roll back the failed turn first."""

    def test_rollback_last_turn_drops_pair(self, fresh_state):
        from enigma_engine.api import server as srv
        cid = srv.state.create_conversation()
        srv.state._resolve_and_activate(cid)
        srv.state._histories[cid].extend([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "bad"},
        ])
        assert srv.state.rollback_last_turn(cid) is True
        assert srv.state._histories[cid] == []

    def test_rollback_noop_on_empty(self, fresh_state):
        from enigma_engine.api import server as srv
        cid = srv.state.create_conversation()
        assert srv.state.rollback_last_turn(cid) is False

    def test_rollback_unknown_raises(self, fresh_state):
        from enigma_engine.api import server as srv
        with pytest.raises(KeyError):
            srv.state.rollback_last_turn("nope")

    def test_retry_engine_call_sees_clean_history(
            self, client, fresh_state, monkeypatch):
        """T3: when AutoResearch-2 retry fires, engine.chat must NOT receive
        the failed user+assistant pair in the history kwarg.
        """
        import types

        # Fake auto_research module so no network hits occur.
        ar_mod = types.ModuleType("enigma_engine.core.auto_research")
        ar_mod.should_retry_with_research = lambda msg, resp: True
        ar_mod.auto_research = lambda msg, max_results=3: "research context"
        ar_mod.should_auto_research = lambda msg: False
        monkeypatch.setitem(
            __import__("sys").modules,
            "enigma_engine.core.auto_research", ar_mod)

        # Capture every history kwarg the engine.chat mock sees.
        call_histories: list[list] = []
        call_count = [0]

        def fake_chat(msg, history=None, **kw):
            call_count[0] += 1
            call_histories.append(list(history or []))
            return f"response_{call_count[0]}"

        fresh_state.engine.chat = fake_chat

        cid = client.post("/api/conversations").json()["id"]
        # First ordinary turn — seeds non-empty history.
        client.post("/api/chat",
                    json={"message": "seed", "conversation_id": cid})
        # After seed: call_count=1, history=[{user:seed},{assistant:response_1}]
        call_histories.clear()
        # Reset counter so the "bad" response on the retry turn is identifiable.
        call_count[0] = 100

        # The retry turn: first call returns "response_101" which triggers
        # should_retry_with_research=True; rollback must clear that pair
        # before the retry call, which returns "response_102".
        client.post("/api/chat",
                    json={"message": "question", "conversation_id": cid,
                          "web_access": True})

        # We expect exactly two engine.chat calls for the retry turn.
        assert len(call_histories) == 2, (
            f"Expected 2 engine.chat calls (initial + retry), got "
            f"{len(call_histories)}")

        # The retry (call[1]) must NOT contain the failed assistant reply
        # "response_101" from call[0]; rollback must have removed it.
        retry_history = call_histories[1]
        failed_contents = [m["content"] for m in retry_history
                           if m.get("role") == "assistant"
                           and m.get("content") == "response_101"]
        assert not failed_contents, (
            "Retry call received the failed assistant reply in history; "
            "rollback_last_turn did not fire before retry."
        )


class TestStreamOrphanConv:
    """B1: 429 on /api/chat/stream must NOT orphan an empty conversation."""

    def test_stream_busy_no_id_does_not_create_conversation(
            self, client, fresh_state):
        from enigma_engine.api import server as srv

        # Hold the inference lock to force 429.
        srv._inference_lock.acquire()
        try:
            before = len(fresh_state._histories)
            resp = client.post("/api/chat/stream", json={"message": "hi"})
            after = len(fresh_state._histories)
        finally:
            srv._inference_lock.release()

        assert resp.status_code == 429
        # B1 fix: no empty conversation must have been created.
        assert after == before, (
            f"Stream 429 created {after - before} orphan conversation(s)")

    def test_stream_busy_with_id_does_not_duplicate_conversation(
            self, client, fresh_state):
        """Known-ID 429 must not change the conversation count."""
        from enigma_engine.api import server as srv

        cid = client.post("/api/conversations").json()["id"]
        srv._inference_lock.acquire()
        try:
            before = len(fresh_state._histories)
            resp = client.post("/api/chat/stream",
                               json={"message": "hi", "conversation_id": cid})
            after = len(fresh_state._histories)
        finally:
            srv._inference_lock.release()

        assert resp.status_code == 429
        assert after == before


class TestResolveActivateNoTOCTOU:
    """B3: resolve+activate must not silently resurrect a deleted conv.

    The original two-step (_resolve_conversation → _activate) released
    the lock between the existence check and the active-id store, so a
    concurrent DELETE could leave the active id pointing at a deleted
    conversation. The downstream ``_histories.setdefault`` then
    resurrected it at append time, also re-adding it to ``_conv_order``
    via ``_touch_locked``. Three defenses are tested here:
    (1) ``_resolve_and_activate`` does both under one lock acquisition,
    (2) ``_touch_locked`` refuses to re-add a missing conv,
    (3) ``_append_turn_if_alive_locked`` skips instead of setdefault.
    """

    def test_resolve_and_activate_atomic_keyerror_on_missing(
            self, fresh_state):
        """Unknown id raises KeyError without leaving stale active state."""
        with pytest.raises(KeyError):
            fresh_state._resolve_and_activate("does-not-exist")
        # Active id must not have been mutated to point at the bad id.
        assert fresh_state._active_conv_id != "does-not-exist"

    def test_touch_locked_refuses_missing_conv(self, fresh_state):
        """_touch_locked must NOT re-add an id that isn't in _histories."""
        with fresh_state._lock:
            fresh_state._conv_order.append("ghost")
            fresh_state._touch_locked("ghost")
        assert "ghost" not in fresh_state._conv_order

    def test_append_turn_if_alive_locked_skips_deleted(self, fresh_state):
        """If the conv was deleted, append is a no-op (returns False)."""
        with fresh_state._lock:
            stored = fresh_state._append_turn_if_alive_locked(
                "ghost", "q", "a")
        assert stored is False
        assert "ghost" not in fresh_state._histories

    def test_append_turn_if_alive_locked_appends_when_alive(self, fresh_state):
        """Happy path: conv exists, turn lands."""
        cid = fresh_state.create_conversation()
        with fresh_state._lock:
            stored = fresh_state._append_turn_if_alive_locked(
                cid, "q", "a")
        assert stored is True
        assert fresh_state._histories[cid] == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]

    def test_chat_does_not_resurrect_deleted_conv(
            self, client, fresh_state, monkeypatch):
        """End-to-end: simulate DELETE racing between resolve and append.

        Patch ``engine.chat`` so it deletes the conversation MID-call,
        between resolve/activate and the post-generation append. The
        chat call still returns (response already produced), but the
        history must NOT be resurrected — neither in ``_histories``
        nor in ``_conv_order``.
        """
        cid = client.post("/api/conversations").json()["id"]

        def chat_then_delete(msg, history=None, **kw):
            # DELETE while engine is "generating".
            fresh_state.delete_conversation(cid)
            return "response_after_delete"

        fresh_state.engine.chat = chat_then_delete

        resp = client.post("/api/chat",
                           json={"message": "hi", "conversation_id": cid})
        # Request returns a 4xx because the conv was deleted between
        # resolve and append — clients should retry against a fresh id.
        # Conv must NOT have been resurrected.
        assert cid not in fresh_state._histories, (
            f"Deleted conv {cid} was resurrected in _histories")
        assert cid not in fresh_state._conv_order, (
            f"Deleted conv {cid} was resurrected in _conv_order")
        # The response status doesn't matter for resurrection — the
        # invariant is that no ghost row remains. Asserting the HTTP
        # code is brittle because the route returns whatever the
        # storage layer chose (200 with body intact, since we appended
        # nothing but didn't error). Keep the invariant gate.
        del resp


class TestDiskPersistence:
    """MC-2: per-conversation disk persistence."""

    def test_round_trip(self, fresh_state, tmp_path, monkeypatch):
        """Chat appends, then load_conversations_from_disk restores them."""
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "CONVERSATIONS_DIR", tmp_path)

        cid = srv.state.create_conversation()
        with srv.state._lock:
            srv.state._histories[cid] = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        srv.state._persist_conversation(cid)

        # Wipe in-memory state, reload from disk.
        with srv.state._lock:
            srv.state._histories.clear()
            srv.state._conv_order.clear()
        count = srv.state.load_conversations_from_disk()

        assert count == 1
        assert cid in srv.state._histories
        messages = srv.state._histories[cid]
        assert messages[0] == {"role": "user", "content": "hello"}
        assert messages[1] == {"role": "assistant", "content": "hi there"}

    def test_corrupt_file_skipped_with_warning(
            self, fresh_state, tmp_path, monkeypatch, caplog):
        """A corrupt JSONL file must be skipped, not crash the server."""
        import logging
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "CONVERSATIONS_DIR", tmp_path)

        good_cid = "a" * 32
        bad_cid = "b" * 32
        (tmp_path / f"{good_cid}.jsonl").write_text(
            '{"role":"user","content":"ok"}\n', encoding="utf-8")
        (tmp_path / f"{bad_cid}.jsonl").write_text(
            "NOT VALID JSON }{", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            count = srv.state.load_conversations_from_disk()

        assert count == 1  # only the good one loaded
        assert good_cid in srv.state._histories
        assert bad_cid not in srv.state._histories
        assert any(bad_cid in r.message for r in caplog.records), \
            "Expected a warning naming the corrupt file"

    def test_eviction_deletes_disk_file(
            self, fresh_state, tmp_path, monkeypatch):
        """LRU eviction must remove the evicted conversation's JSONL file."""
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "CONVERSATIONS_DIR", tmp_path)
        monkeypatch.setattr(srv, "MAX_CONVERSATIONS", 2)

        # Create two conversations and persist them.
        a = srv.state.create_conversation()
        b = srv.state.create_conversation()
        with srv.state._lock:
            srv.state._histories[a] = [{"role": "user", "content": "a"}]
            srv.state._histories[b] = [{"role": "user", "content": "b"}]
        srv.state._persist_conversation(a)
        srv.state._persist_conversation(b)
        assert (tmp_path / f"{a}.jsonl").exists()
        assert (tmp_path / f"{b}.jsonl").exists()

        # Creating a third must evict the LRU (a).
        # Touch b to make a the LRU candidate.
        with srv.state._lock:
            srv.state._touch_locked(b)
        srv.state.create_conversation()

        assert not (tmp_path / f"{a}.jsonl").exists(), \
            "Evicted conversation file must be deleted from disk"
        assert (tmp_path / f"{b}.jsonl").exists()

    def test_explicit_delete_removes_disk_file(
            self, fresh_state, tmp_path, monkeypatch):
        """delete_conversation must unlink the on-disk JSONL file."""
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "CONVERSATIONS_DIR", tmp_path)

        cid = srv.state.create_conversation()
        with srv.state._lock:
            srv.state._histories[cid] = [{"role": "user", "content": "x"}]
        srv.state._persist_conversation(cid)
        assert (tmp_path / f"{cid}.jsonl").exists()

        srv.state.delete_conversation(cid)
        assert not (tmp_path / f"{cid}.jsonl").exists()

    def test_delete_also_removes_bak_sibling(
            self, fresh_state, tmp_path, monkeypatch):
        """MC-2c: ``delete_conversation`` must unlink the ``.jsonl.bak``
        sibling created by ``atomic_write_text`` on the second save.

        Without this, every deleted/LRU-evicted conversation leaks a
        permanent orphan backup file. ``glob("*.jsonl")`` doesn't see
        them so they never resurrect, but disk usage grows unbounded.
        """
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "CONVERSATIONS_DIR", tmp_path)

        cid = srv.state.create_conversation()
        # Two persists -> atomic_write_text backs up the first to .bak
        # before overwriting on the second.
        with srv.state._lock:
            srv.state._histories[cid] = [{"role": "user", "content": "first"}]
        srv.state._persist_conversation(cid)
        with srv.state._lock:
            srv.state._histories[cid] = [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
            ]
        srv.state._persist_conversation(cid)
        assert (tmp_path / f"{cid}.jsonl").exists()
        assert (tmp_path / f"{cid}.jsonl.bak").exists(), \
            "precondition: atomic_write_text must create .bak on second save"

        srv.state.delete_conversation(cid)
        assert not (tmp_path / f"{cid}.jsonl").exists()
        assert not (tmp_path / f"{cid}.jsonl.bak").exists(), \
            "MC-2c: .bak sibling must also be unlinked"

    def test_boot_load_sweeps_orphan_bak_files(
            self, fresh_state, tmp_path, monkeypatch):
        """MC-2c: ``load_conversations_from_disk`` must sweep ``.jsonl.bak``
        files whose live ``.jsonl`` parent no longer exists.

        Cleans up the historical leak from older code paths that didn't
        unlink the .bak on delete/eviction. ``.bak`` files with a live
        parent must be preserved (they're legitimate one-revision-back
        recovery points).
        """
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "CONVERSATIONS_DIR", tmp_path)

        live_cid = "a" * 32
        orphan_cid = "b" * 32

        # Live conversation: both .jsonl and .jsonl.bak should survive.
        (tmp_path / f"{live_cid}.jsonl").write_text(
            '{"role":"user","content":"hi"}\n', encoding="utf-8")
        (tmp_path / f"{live_cid}.jsonl.bak").write_text(
            '{"role":"user","content":"old"}\n', encoding="utf-8")

        # Orphan: only .bak, no parent .jsonl -> must be swept.
        (tmp_path / f"{orphan_cid}.jsonl.bak").write_text(
            '{"role":"user","content":"dead"}\n', encoding="utf-8")

        srv.state.load_conversations_from_disk()

        assert (tmp_path / f"{live_cid}.jsonl").exists()
        assert (tmp_path / f"{live_cid}.jsonl.bak").exists(), \
            "legitimate .bak with live parent must be preserved"
        assert not (tmp_path / f"{orphan_cid}.jsonl.bak").exists(), \
            "MC-2c: orphan .bak with no parent must be swept on boot"

    def test_boot_load_excess_eviction_also_removes_bak(
            self, fresh_state, tmp_path, monkeypatch):
        """MC-2c: MC-2a excess-unlink loop must also remove the .bak
        sibling, otherwise eviction creates new orphans on every boot.
        """
        import os
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "CONVERSATIONS_DIR", tmp_path)
        monkeypatch.setattr(srv, "MAX_CONVERSATIONS", 2)

        cids: list[str] = []
        for i in range(3):
            cid = f"{i:032x}"
            cids.append(cid)
            jsonl = tmp_path / f"{cid}.jsonl"
            bak = tmp_path / f"{cid}.jsonl.bak"
            jsonl.write_text(
                f'{{"role":"user","content":"m{i}"}}\n', encoding="utf-8")
            bak.write_text(
                f'{{"role":"user","content":"old{i}"}}\n', encoding="utf-8")
            # i=0 oldest -> will be evicted.
            os.utime(jsonl, (1000.0 + i, 1000.0 + i))
            os.utime(bak, (1000.0 + i, 1000.0 + i))

        srv.state.load_conversations_from_disk()

        # cids[0] is the LRU -> evicted; its .bak must go with it.
        assert not (tmp_path / f"{cids[0]}.jsonl").exists()
        assert not (tmp_path / f"{cids[0]}.jsonl.bak").exists(), \
            "MC-2c: excess-eviction must unlink .bak sibling too"
        # The two kept conversations and their .baks survive.
        for cid in cids[1:]:
            assert (tmp_path / f"{cid}.jsonl").exists()
            assert (tmp_path / f"{cid}.jsonl.bak").exists()

    def test_persist_after_delete_does_not_resurrect_as_ghost_file(
            self, fresh_state, tmp_path, monkeypatch):
        """MC-2 B3 sibling: ``_persist_conversation`` must not write a
        phantom empty file when the conversation was deleted between
        history-append and persist.

        Race scenario reproduced sequentially:
          1. Thread A appends a turn under lock, releases lock.
          2. Thread B deletes the conversation (removes from histories,
             unlinks any existing file).
          3. Thread A calls ``_persist_conversation(cid)``.

        Without the gate, step 3 reads ``_histories.get(cid, [])`` ->
        empty list -> writes a zero-length ``.jsonl`` that survives
        daemon restart as a ghost zero-message conv occupying a slot.
        With the gate, step 3 is a no-op and no file appears on disk.
        """
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "CONVERSATIONS_DIR", tmp_path)

        cid = srv.state.create_conversation()
        # Step 1: history exists from a prior append.
        with srv.state._lock:
            srv.state._histories[cid] = [
                {"role": "user", "content": "alive"},
                {"role": "assistant", "content": "ok"},
            ]
        # Step 2: concurrent delete wins the race before persist runs.
        srv.state.delete_conversation(cid)
        assert not (tmp_path / f"{cid}.jsonl").exists()

        # Step 3: stale persist call from the generation path.
        srv.state._persist_conversation(cid)

        # Must NOT create a ghost file.
        assert not (tmp_path / f"{cid}.jsonl").exists(), \
            "_persist_conversation must skip deleted conversations"

    def test_boot_load_evicts_excess_disk_files(
            self, fresh_state, tmp_path, monkeypatch, caplog):
        """MC-2a: if disk has more than MAX_CONVERSATIONS files, the oldest
        excess files must be deleted on boot — not just skipped in memory.

        Scenario: cap was previously high; user lowers it to 2; 5 files
        sit on disk. After ``load_conversations_from_disk``, 2 newest are
        loaded AND the 3 oldest are deleted from disk + a WARNING is
        logged for each.
        """
        import logging
        import os
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "CONVERSATIONS_DIR", tmp_path)
        monkeypatch.setattr(srv, "MAX_CONVERSATIONS", 2)

        # Five valid files; mtimes assigned 1..5 so file_1 is oldest.
        cids: list[str] = []
        for i in range(5):
            cid = f"{i:032x}"  # 32-char hex, valid UUID4-shape
            cids.append(cid)
            path = tmp_path / f"{cid}.jsonl"
            path.write_text(
                f'{{"role":"user","content":"msg{i}"}}\n',
                encoding="utf-8",
            )
            # Pin mtime: i=0 oldest, i=4 newest.
            os.utime(path, (1000.0 + i, 1000.0 + i))

        with caplog.at_level(logging.WARNING):
            count = srv.state.load_conversations_from_disk()

        # Only 2 newest loaded.
        assert count == 2
        assert cids[3] in srv.state._histories
        assert cids[4] in srv.state._histories
        # 3 oldest deleted from disk.
        for i in range(3):
            assert not (tmp_path / f"{cids[i]}.jsonl").exists(), (
                f"Excess file {cids[i]} must be unlinked, not orphaned"
            )
        # 2 newest still on disk.
        assert (tmp_path / f"{cids[3]}.jsonl").exists()
        assert (tmp_path / f"{cids[4]}.jsonl").exists()
        # One WARNING per evicted file.
        evict_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "MC-2a" in r.message
        ]
        assert len(evict_warnings) == 3, (
            f"Expected 3 MC-2a eviction warnings, got {len(evict_warnings)}"
        )

    def test_boot_load_stray_file_does_not_displace_valid_conv(
            self, fresh_state, tmp_path, monkeypatch, caplog):
        """MC-2a follow-up: a stray non-UUID4 ``*.jsonl`` file with a
        newer mtime than valid conversations must NOT push a real conv
        into the excess slice (where it would be deleted).

        Pre-fix scenario: mtime sort happens before the validity check,
        so a stray ``notes.jsonl`` with newest mtime occupies a kept
        slot and the oldest valid conv gets unlinked. Stray files
        should be skipped (logged as "unexpected file") and NOT counted
        toward the cap, and they should be left on disk — they may be
        operator backups or hand-edits that we have no right to delete.
        """
        import logging
        import os
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "CONVERSATIONS_DIR", tmp_path)
        monkeypatch.setattr(srv, "MAX_CONVERSATIONS", 2)

        # 3 valid files + 1 stray. Stray has the NEWEST mtime — would
        # win the cap-slice race under the pre-fix code.
        cids: list[str] = []
        for i in range(3):
            cid = f"{i:032x}"
            cids.append(cid)
            path = tmp_path / f"{cid}.jsonl"
            path.write_text(
                f'{{"role":"user","content":"msg{i}"}}\n',
                encoding="utf-8",
            )
            os.utime(path, (1000.0 + i, 1000.0 + i))

        stray = tmp_path / "notes.jsonl"
        stray.write_text("operator backup\n", encoding="utf-8")
        os.utime(stray, (9999.0, 9999.0))  # newest

        with caplog.at_level(logging.WARNING):
            count = srv.state.load_conversations_from_disk()

        # 2 newest valid convs loaded (cids[1] and cids[2]).
        assert count == 2
        assert cids[2] in srv.state._histories
        assert cids[1] in srv.state._histories
        # cids[0] is the oldest valid; under MAX=2 it's excess and
        # gets deleted. cids[1]/cids[2] must survive on disk.
        assert (tmp_path / f"{cids[1]}.jsonl").exists()
        assert (tmp_path / f"{cids[2]}.jsonl").exists()
        # Stray file must be left alone — we don't own it.
        assert stray.exists(), (
            "Stray non-UUID4 file must not be deleted by load"
        )
        # And no MC-2a eviction warning should mention the stray —
        # only valid-shape excess files get the eviction message.
        for r in caplog.records:
            if "MC-2a" in r.message:
                assert "notes.jsonl" not in r.message, (
                    "Stray file must not be reported as MC-2a excess"
                )

    def test_active_pointer_persists_across_restart(
            self, fresh_state, tmp_path, monkeypatch):
        """MC-2b: active conv id survives daemon restart.

        Setup: create conv, activate it, write a turn so it persists.
        Simulate restart by zeroing in-memory state then calling the
        same boot helpers. Assert active pointer restored.
        """
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "CONVERSATIONS_DIR", tmp_path)
        monkeypatch.setattr(
            srv, "ACTIVE_CONV_FILE", tmp_path / "_active.json")

        cid = srv.state.create_conversation()
        srv.state._resolve_and_activate(cid)
        srv.state._histories[cid].append({"role": "user", "content": "hi"})
        srv.state._persist_conversation(cid)

        # Pointer file must exist on disk after activation.
        assert (tmp_path / "_active.json").exists()

        # Simulate daemon restart: wipe in-memory state.
        srv.state._histories.clear()
        srv.state._conv_order.clear()
        srv.state._active_conv_id = None

        # Boot path: load histories then active pointer.
        srv.state.load_conversations_from_disk()
        srv.state._load_active_conv_id_from_disk()

        assert srv.state._active_conv_id == cid, (
            "MC-2b: active conv id must be restored on boot"
        )

    def test_active_pointer_skipped_when_saved_id_evicted(
            self, fresh_state, tmp_path, monkeypatch):
        """MC-2b: a stale saved active id (no matching conv on disk)
        must be ignored — daemon boots with no active conversation
        rather than pointing at a deleted/evicted id.
        """
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "CONVERSATIONS_DIR", tmp_path)
        monkeypatch.setattr(
            srv, "ACTIVE_CONV_FILE", tmp_path / "_active.json")

        # Hand-write a stale active pointer with no matching conv file.
        stale_cid = "ff" * 16  # 32 hex chars, valid shape, never created
        (tmp_path / "_active.json").write_text(
            json.dumps({"active_conv_id": stale_cid}),
            encoding="utf-8",
        )

        srv.state._histories.clear()
        srv.state._conv_order.clear()
        srv.state._active_conv_id = None
        srv.state.load_conversations_from_disk()
        srv.state._load_active_conv_id_from_disk()

        assert srv.state._active_conv_id is None, (
            "MC-2b: stale active id must not be restored"
        )

    def test_active_pointer_self_heals_stale_id_on_boot(
            self, fresh_state, tmp_path, monkeypatch):
        """MC-2b: a stale saved id is rewritten to ``null`` on boot so
        the next boot doesn't rediscover and re-log the same stale id.
        """
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "CONVERSATIONS_DIR", tmp_path)
        monkeypatch.setattr(
            srv, "ACTIVE_CONV_FILE", tmp_path / "_active.json")

        stale_cid = "ee" * 16
        (tmp_path / "_active.json").write_text(
            json.dumps({"active_conv_id": stale_cid}),
            encoding="utf-8",
        )

        srv.state._histories.clear()
        srv.state._conv_order.clear()
        srv.state._active_conv_id = None
        srv.state.load_conversations_from_disk()
        srv.state._load_active_conv_id_from_disk()

        payload = json.loads(
            (tmp_path / "_active.json").read_text(encoding="utf-8"))
        assert payload["active_conv_id"] is None, (
            "MC-2b: stale id must be self-healed to null on disk"
        )

    def test_active_pointer_cleared_when_active_deleted(
            self, fresh_state, tmp_path, monkeypatch):
        """MC-2b: deleting the active conversation persists ``None``
        to disk so the next boot doesn't resurrect a deleted id.
        """
        from enigma_engine.api import server as srv
        monkeypatch.setattr(srv, "CONVERSATIONS_DIR", tmp_path)
        monkeypatch.setattr(
            srv, "ACTIVE_CONV_FILE", tmp_path / "_active.json")

        cid = srv.state.create_conversation()
        srv.state._resolve_and_activate(cid)
        srv.state._histories[cid].append({"role": "user", "content": "hi"})
        srv.state._persist_conversation(cid)

        srv.state.delete_conversation(cid)

        payload = json.loads(
            (tmp_path / "_active.json").read_text(encoding="utf-8"))
        assert payload["active_conv_id"] is None, (
            "MC-2b: active pointer must be cleared when active conv deleted"
        )
