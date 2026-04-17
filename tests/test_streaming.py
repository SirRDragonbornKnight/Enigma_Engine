"""Tests for streaming system: TokenBuffer, StreamingResponse, TokenStreamer, CallbackStreamer."""
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── StreamChunk ──────────────────────────────────────────────────────────────

class TestStreamChunk:
    """Test StreamChunk data class."""

    def test_to_dict(self):
        """to_dict should return all fields."""
        from enigma_engine.core.streaming import StreamChunk, StreamEvent
        chunk = StreamChunk(content="hello", event=StreamEvent.TOKEN, index=0,
                            token_id=42, logprob=-1.5)
        d = chunk.to_dict()
        assert d["content"] == "hello"
        assert d["event"] == "token"
        assert d["index"] == 0
        assert d["token_id"] == 42
        assert d["logprob"] == -1.5

    def test_to_sse_format(self):
        """to_sse should produce valid SSE format."""
        from enigma_engine.core.streaming import StreamChunk, StreamEvent
        chunk = StreamChunk(content="world", event=StreamEvent.TOKEN, index=1)
        sse = chunk.to_sse()
        assert sse.startswith("event: token\n")
        assert "data: " in sse
        assert sse.endswith("\n\n")

    def test_to_sse_contains_json(self):
        """SSE data line should contain valid JSON."""
        import json
        from enigma_engine.core.streaming import StreamChunk, StreamEvent
        chunk = StreamChunk(content="test", event=StreamEvent.CHUNK, index=0)
        sse = chunk.to_sse()
        data_line = [l for l in sse.split("\n") if l.startswith("data: ")][0]
        parsed = json.loads(data_line[len("data: "):])
        assert parsed["content"] == "test"
        assert parsed["event"] == "chunk"


# ── TokenBuffer ──────────────────────────────────────────────────────────────

class TestTokenBuffer:
    """Test token buffering and flushing."""

    def test_no_buffer_immediate_flush(self):
        """With size=0, every token should flush immediately."""
        from enigma_engine.core.streaming import TokenBuffer
        buf = TokenBuffer(size=0)
        result = buf.add("hello")
        assert result == "hello"

    def test_buffered_accumulation(self):
        """With size>1, tokens should accumulate until buffer full."""
        from enigma_engine.core.streaming import TokenBuffer
        buf = TokenBuffer(size=3)
        assert buf.add("a") is None
        assert buf.add("b") is None
        result = buf.add("c")
        assert result == "abc"

    def test_manual_flush(self):
        """flush() should return accumulated content."""
        from enigma_engine.core.streaming import TokenBuffer
        buf = TokenBuffer(size=10)
        buf.add("hello")
        buf.add(" world")
        result = buf.flush()
        assert result == "hello world"

    def test_flush_empty_buffer(self):
        """flush() on empty buffer should return empty string."""
        from enigma_engine.core.streaming import TokenBuffer
        buf = TokenBuffer(size=5)
        assert buf.flush() == ""

    def test_has_content(self):
        """has_content should reflect buffer state."""
        from enigma_engine.core.streaming import TokenBuffer
        buf = TokenBuffer(size=10)
        assert not buf.has_content()
        buf.add("x")
        assert buf.has_content()
        buf.flush()
        assert not buf.has_content()


# ── StreamingResponse lifecycle ──────────────────────────────────────────────

class TestStreamingResponse:
    """Test StreamingResponse start → push → finish lifecycle."""

    def test_start_marks_started(self):
        """start() should mark stream as started."""
        from enigma_engine.core.streaming import StreamingResponse
        resp = StreamingResponse()
        resp.start()
        assert resp._started

    def test_push_auto_starts(self):
        """push() should auto-start if not started."""
        from enigma_engine.core.streaming import StreamingResponse
        resp = StreamingResponse()
        resp.push("token")
        assert resp._started

    def test_push_accumulates_tokens(self):
        """Pushed tokens should accumulate."""
        from enigma_engine.core.streaming import StreamingResponse
        resp = StreamingResponse()
        resp.push("hello")
        resp.push(" world")
        assert resp._tokens == ["hello", " world"]

    def test_finish_marks_complete(self):
        """finish() should mark stream as complete."""
        from enigma_engine.core.streaming import StreamingResponse
        resp = StreamingResponse()
        resp.start()
        resp.push("test")
        resp.finish()
        assert resp.is_complete

    def test_finish_idempotent(self):
        """Calling finish() twice should not crash."""
        from enigma_engine.core.streaming import StreamingResponse
        resp = StreamingResponse()
        resp.start()
        resp.finish()
        resp.finish()  # Should not crash
        assert resp.is_complete

    def test_push_after_finish_ignored(self):
        """Tokens pushed after finish should be ignored."""
        from enigma_engine.core.streaming import StreamingResponse
        resp = StreamingResponse()
        resp.start()
        resp.push("a")
        resp.finish()
        resp.push("b")  # Should be ignored
        assert resp._tokens == ["a"]

    def test_iter_yields_chunks(self):
        """Iterating should yield chunks including START and END."""
        from enigma_engine.core.streaming import StreamingResponse, StreamEvent
        resp = StreamingResponse()
        resp.start()
        resp.push("hello")
        resp.finish()
        chunks = list(resp)
        events = [c.event for c in chunks]
        assert StreamEvent.START in events
        assert StreamEvent.END in events

    def test_iter_tokens_yields_content_only(self):
        """iter_tokens should yield only token/chunk content."""
        from enigma_engine.core.streaming import StreamingResponse
        resp = StreamingResponse()
        resp.start()
        resp.push("a")
        resp.push("b")
        resp.push("c")
        resp.finish()
        tokens = list(resp.iter_tokens())
        assert "".join(tokens) == "abc"

    def test_get_text_returns_full_text(self):
        """get_text should return concatenated tokens."""
        from enigma_engine.core.streaming import StreamingResponse
        resp = StreamingResponse()
        resp.start()
        resp.push("hello")
        resp.push(" world")
        resp.finish()
        assert resp.get_text() == "hello world"


# ── StreamingResponse error handling ─────────────────────────────────────────

class TestStreamingResponseErrors:
    """Test error handling in StreamingResponse."""

    def test_error_marks_has_error(self):
        """error() should set has_error flag."""
        from enigma_engine.core.streaming import StreamingResponse
        resp = StreamingResponse()
        resp.start()
        resp.error(RuntimeError("test error"))
        assert resp.has_error

    def test_error_marks_finished(self):
        """error() should also mark stream as finished."""
        from enigma_engine.core.streaming import StreamingResponse
        resp = StreamingResponse()
        resp.start()
        resp.error(RuntimeError("boom"))
        assert resp.is_complete

    def test_iter_stops_on_error(self):
        """Iteration should stop when error event received."""
        from enigma_engine.core.streaming import StreamingResponse, StreamEvent
        resp = StreamingResponse()
        resp.start()
        resp.push("token")
        resp.error(ValueError("bad"))
        chunks = list(resp)
        last_event = chunks[-1].event
        assert last_event == StreamEvent.ERROR


# ── StreamingResponse statistics ─────────────────────────────────────────────

class TestStreamingStats:
    """Test get_stats() returns meaningful metrics."""

    def test_stats_after_completion(self):
        """Stats should reflect token count and completion."""
        from enigma_engine.core.streaming import StreamingResponse
        resp = StreamingResponse()
        resp.start()
        for i in range(10):
            resp.push(f"t{i}")
        resp.finish()
        stats = resp.get_stats()
        assert stats["total_tokens"] == 10
        assert stats["is_complete"] is True
        assert stats["has_error"] is False
        assert stats["duration_seconds"] >= 0

    def test_stats_before_start(self):
        """Stats before start should show zero tokens."""
        from enigma_engine.core.streaming import StreamingResponse
        resp = StreamingResponse()
        stats = resp.get_stats()
        assert stats["total_tokens"] == 0
        assert stats["is_complete"] is False


# ── StreamingConfig callbacks ────────────────────────────────────────────────

class TestStreamingCallbacks:
    """Test that callbacks fire correctly."""

    def test_on_token_callback(self):
        """on_token callback should fire for each token."""
        from enigma_engine.core.streaming import StreamingResponse, StreamingConfig
        received = []
        config = StreamingConfig(on_token=lambda t: received.append(t))
        resp = StreamingResponse(config)
        resp.start()
        resp.push("a")
        resp.push("b")
        resp.finish()
        assert received == ["a", "b"]

    def test_on_start_callback(self):
        """on_start callback should fire on start."""
        from enigma_engine.core.streaming import StreamingResponse, StreamingConfig
        started = []
        config = StreamingConfig(on_start=lambda: started.append(True))
        resp = StreamingResponse(config)
        resp.start()
        assert started == [True]

    def test_on_end_callback(self):
        """on_end callback should fire with full text on finish."""
        from enigma_engine.core.streaming import StreamingResponse, StreamingConfig
        ended = []
        config = StreamingConfig(on_end=lambda t: ended.append(t))
        resp = StreamingResponse(config)
        resp.start()
        resp.push("hello")
        resp.finish()
        assert ended == ["hello"]

    def test_on_error_callback(self):
        """on_error callback should fire on error."""
        from enigma_engine.core.streaming import StreamingResponse, StreamingConfig
        errors = []
        config = StreamingConfig(on_error=lambda e: errors.append(str(e)))
        resp = StreamingResponse(config)
        resp.start()
        resp.error(RuntimeError("boom"))
        assert errors == ["boom"]


# ── TokenStreamer ────────────────────────────────────────────────────────────

class TestTokenStreamer:
    """Test TokenStreamer wrapping a generator."""

    def test_stream_from_generator(self):
        """stream() should create a StreamingResponse from a generator."""
        from enigma_engine.core.streaming import TokenStreamer

        def gen():
            yield "hello"
            yield " world"

        streamer = TokenStreamer()
        resp = streamer.stream(gen())
        text = resp.get_text()
        assert text == "hello world"

    def test_stream_handles_generator_error(self):
        """stream() should capture generator errors."""
        from enigma_engine.core.streaming import TokenStreamer

        def bad_gen():
            yield "ok"
            raise RuntimeError("generator failed")

        streamer = TokenStreamer()
        resp = streamer.stream(bad_gen())
        # Consume to trigger the error
        text = resp.get_text()
        assert resp.has_error


# ── CallbackStreamer ─────────────────────────────────────────────────────────

class TestCallbackStreamer:
    """Test CallbackStreamer accumulation and callbacks."""

    def test_callback_receives_tokens(self):
        """Callback should receive each token."""
        from enigma_engine.core.streaming import CallbackStreamer
        received = []
        streamer = CallbackStreamer(callback=lambda t: received.append(t))
        streamer("hello")
        streamer(" world")
        assert received == ["hello", " world"]

    def test_get_text_returns_accumulated(self):
        """get_text should return all accumulated tokens."""
        from enigma_engine.core.streaming import CallbackStreamer
        streamer = CallbackStreamer(callback=lambda t: None)
        streamer("a")
        streamer("b")
        streamer("c")
        assert streamer.get_text() == "abc"

    def test_finish_prevents_further_tokens(self):
        """After finish(), new tokens should be ignored."""
        from enigma_engine.core.streaming import CallbackStreamer
        received = []
        streamer = CallbackStreamer(callback=lambda t: received.append(t))
        streamer("a")
        streamer.finish()
        streamer("b")  # Should be ignored
        assert received == ["a"]

    def test_auto_starts_on_first_token(self):
        """First token should auto-start the streamer."""
        from enigma_engine.core.streaming import CallbackStreamer
        streamer = CallbackStreamer(callback=lambda t: None)
        assert not streamer._started
        streamer("x")
        assert streamer._started


# ── SSE iteration ────────────────────────────────────────────────────────────

class TestSSEIteration:
    """Test Server-Sent Events iteration."""

    def test_iter_sse_produces_valid_sse(self):
        """iter_sse should yield valid SSE formatted strings."""
        from enigma_engine.core.streaming import StreamingResponse
        resp = StreamingResponse()
        resp.start()
        resp.push("test")
        resp.finish()
        sse_chunks = list(resp.iter_sse())
        for chunk in sse_chunks:
            assert "event: " in chunk
            assert "data: " in chunk
            assert chunk.endswith("\n\n")


# ── StreamEvent enum ─────────────────────────────────────────────────────────

class TestStreamEvent:
    """Test StreamEvent enum values."""

    def test_all_events_have_string_values(self):
        """All StreamEvent members should have string values."""
        from enigma_engine.core.streaming import StreamEvent
        for event in StreamEvent:
            assert isinstance(event.value, str)

    def test_expected_events_exist(self):
        """Expected event types should be present."""
        from enigma_engine.core.streaming import StreamEvent
        expected = {"token", "chunk", "start", "end", "error", "metadata"}
        actual = {e.value for e in StreamEvent}
        assert expected == actual
