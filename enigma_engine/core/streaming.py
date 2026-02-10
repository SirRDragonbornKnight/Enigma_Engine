"""
Response Streaming System

Stream LLM responses token-by-token with support for
async generation, SSE, WebSocket, and callback-based streaming.

FILE: enigma_engine/core/streaming.py
TYPE: Core/Inference
MAIN CLASSES: StreamingResponse, TokenStreamer, StreamingConfig
"""

import asyncio
import json
import logging
import queue
import threading
import time
from collections.abc import AsyncIterator, Generator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamEvent(Enum):
    """Stream event types."""
    TOKEN = "token"
    CHUNK = "chunk"
    START = "start"
    END = "end"
    ERROR = "error"
    METADATA = "metadata"


@dataclass
class StreamChunk:
    """A streaming chunk."""
    content: str
    event: StreamEvent = StreamEvent.TOKEN
    index: int = 0
    timestamp: float = field(default_factory=time.time)
    
    # Token info
    token_id: Optional[int] = None
    logprob: Optional[float] = None
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "event": self.event.value,
            "index": self.index,
            "timestamp": self.timestamp,
            "token_id": self.token_id,
            "logprob": self.logprob,
            "metadata": self.metadata
        }
    
    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        data = json.dumps(self.to_dict())
        return f"event: {self.event.value}\ndata: {data}\n\n"


@dataclass
class StreamingConfig:
    """Streaming configuration."""
    # Buffering
    buffer_size: int = 0  # 0 = no buffering
    flush_interval: float = 0.0  # Seconds, 0 = immediate
    
    # Timing
    timeout: float = 60.0
    token_delay: float = 0.0  # Artificial delay between tokens
    
    # Output
    include_logprobs: bool = False
    include_token_ids: bool = False
    
    # Callbacks
    on_token: Optional[Callable[[str], None]] = None
    on_start: Optional[Callable[[], None]] = None
    on_end: Optional[Callable[[str], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None


class TokenBuffer:
    """Buffer for accumulating tokens before flushing."""
    
    def __init__(self, size: int = 0, interval: float = 0.0):
        self.size = size
        self.interval = interval
        self._buffer: list[str] = []
        self._last_flush = time.time()
        self._lock = threading.Lock()
    
    def add(self, token: str) -> Optional[str]:
        """
        Add token to buffer.
        
        Returns:
            Flushed content if buffer full, else None
        """
        with self._lock:
            self._buffer.append(token)
            
            should_flush = (
                (self.size > 0 and len(self._buffer) >= self.size) or
                (self.interval > 0 and time.time() - self._last_flush >= self.interval)
            )
            
            if should_flush or self.size == 0:
                return self.flush()
        
        return None
    
    def flush(self) -> str:
        """Flush buffer and return content."""
        with self._lock:
            content = "".join(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()
            return content
    
    def has_content(self) -> bool:
        """Check if buffer has content."""
        return len(self._buffer) > 0


class StreamingResponse:
    """
    Streaming response handler.
    
    Collects tokens and provides various output formats
    including iterator, async iterator, SSE, and WebSocket.
    """
    
    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
        
        self._queue: queue.Queue[StreamChunk] = queue.Queue()
        self._async_queue: asyncio.Queue = None
        self._buffer = TokenBuffer(
            size=self.config.buffer_size,
            interval=self.config.flush_interval
        )
        
        self._tokens: list[str] = []
        self._chunks: list[StreamChunk] = []
        self._started = False
        self._finished = False
        self._error: Optional[Exception] = None
        
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        
        # Callbacks
        self._on_token = self.config.on_token
        self._on_start = self.config.on_start
        self._on_end = self.config.on_end
        self._on_error = self.config.on_error
    
    def start(self, metadata: dict[str, Any] = None):
        """Start the stream."""
        if self._started:
            return
        
        self._started = True
        self._start_time = time.time()
        
        chunk = StreamChunk(
            content="",
            event=StreamEvent.START,
            metadata=metadata or {}
        )
        self._emit(chunk)
        
        if self._on_start:
            try:
                self._on_start()
            except Exception as e:
                logger.debug(f"Start callback error: {e}")
    
    def push(
        self,
        token: str,
        token_id: int = None,
        logprob: float = None,
        metadata: dict[str, Any] = None
    ):
        """
        Push a token to the stream.
        
        Args:
            token: Token string
            token_id: Token ID
            logprob: Log probability
            metadata: Additional metadata
        """
        if self._finished:
            return
        
        # Auto-start if needed
        if not self._started:
            self.start()
        
        self._tokens.append(token)
        
        # Buffer if configured
        flushed = self._buffer.add(token)
        
        if flushed is not None:
            chunk = StreamChunk(
                content=flushed,
                event=StreamEvent.CHUNK if self.config.buffer_size > 0 else StreamEvent.TOKEN,
                index=len(self._chunks),
                token_id=token_id if self.config.include_token_ids else None,
                logprob=logprob if self.config.include_logprobs else None,
                metadata=metadata or {}
            )
            
            self._emit(chunk)
            
            if self._on_token:
                try:
                    self._on_token(flushed)
                except Exception as e:
                    logger.debug(f"Token callback error: {e}")
        
        # Artificial delay
        if self.config.token_delay > 0:
            time.sleep(self.config.token_delay)
    
    def finish(self, metadata: dict[str, Any] = None):
        """Finish the stream."""
        if self._finished:
            return
        
        # Flush remaining buffer
        if self._buffer.has_content():
            flushed = self._buffer.flush()
            chunk = StreamChunk(
                content=flushed,
                event=StreamEvent.CHUNK,
                index=len(self._chunks)
            )
            self._emit(chunk)
        
        self._finished = True
        self._end_time = time.time()
        
        full_text = "".join(self._tokens)
        
        chunk = StreamChunk(
            content=full_text,
            event=StreamEvent.END,
            index=len(self._chunks),
            metadata={
                **(metadata or {}),
                "total_tokens": len(self._tokens),
                "duration_ms": (self._end_time - self._start_time) * 1000
            }
        )
        self._emit(chunk)
        
        if self._on_end:
            try:
                self._on_end(full_text)
            except Exception as e:
                logger.debug(f"End callback error: {e}")
    
    def error(self, exception: Exception):
        """Signal an error in the stream."""
        self._error = exception
        self._finished = True
        
        chunk = StreamChunk(
            content=str(exception),
            event=StreamEvent.ERROR,
            index=len(self._chunks),
            metadata={"error_type": type(exception).__name__}
        )
        self._emit(chunk)
        
        if self._on_error:
            try:
                self._on_error(exception)
            except Exception as e:
                logger.debug(f"Error callback error: {e}")
    
    def _emit(self, chunk: StreamChunk):
        """Emit a chunk to all outputs."""
        self._chunks.append(chunk)
        self._queue.put_nowait(chunk)
        
        if self._async_queue is not None:
            try:
                self._async_queue.put_nowait(chunk)
            except Exception:
                pass  # Queue full or closed
    
    def __iter__(self) -> Iterator[StreamChunk]:
        """Iterate over chunks."""
        while True:
            try:
                chunk = self._queue.get(timeout=self.config.timeout)
                yield chunk
                
                if chunk.event in (StreamEvent.END, StreamEvent.ERROR):
                    break
            except queue.Empty:
                break
    
    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Async iterate over chunks."""
        if self._async_queue is None:
            self._async_queue = asyncio.Queue()
            
            # Copy existing chunks
            for chunk in self._chunks:
                await self._async_queue.put(chunk)
        
        while True:
            try:
                chunk = await asyncio.wait_for(
                    self._async_queue.get(),
                    timeout=self.config.timeout
                )
                yield chunk
                
                if chunk.event in (StreamEvent.END, StreamEvent.ERROR):
                    break
            except asyncio.TimeoutError:
                break
    
    def iter_tokens(self) -> Iterator[str]:
        """Iterate over token strings only."""
        for chunk in self:
            if chunk.event in (StreamEvent.TOKEN, StreamEvent.CHUNK):
                yield chunk.content
    
    async def aiter_tokens(self) -> AsyncIterator[str]:
        """Async iterate over token strings."""
        async for chunk in self:
            if chunk.event in (StreamEvent.TOKEN, StreamEvent.CHUNK):
                yield chunk.content
    
    def iter_sse(self) -> Iterator[str]:
        """Iterate as Server-Sent Events."""
        for chunk in self:
            yield chunk.to_sse()
    
    async def aiter_sse(self) -> AsyncIterator[str]:
        """Async iterate as Server-Sent Events."""
        async for chunk in self:
            yield chunk.to_sse()
    
    def get_text(self) -> str:
        """Get full text (blocking until complete)."""
        for _ in self:
            pass
        return "".join(self._tokens)
    
    async def aget_text(self) -> str:
        """Async get full text."""
        async for _ in self:
            pass
        return "".join(self._tokens)
    
    @property
    def is_complete(self) -> bool:
        """Check if stream is complete."""
        return self._finished
    
    @property
    def has_error(self) -> bool:
        """Check if stream has error."""
        return self._error is not None
    
    def get_stats(self) -> dict[str, Any]:
        """Get streaming statistics."""
        duration = 0.0
        if self._start_time:
            end = self._end_time or time.time()
            duration = end - self._start_time
        
        tokens_per_sec = len(self._tokens) / duration if duration > 0 else 0.0
        
        return {
            "total_tokens": len(self._tokens),
            "total_chunks": len(self._chunks),
            "duration_seconds": duration,
            "tokens_per_second": tokens_per_sec,
            "is_complete": self._finished,
            "has_error": self._error is not None
        }


class TokenStreamer:
    """
    Wrapper for streaming token generation.
    
    Integrates with model inference to provide streaming output.
    """
    
    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
    
    def stream(
        self,
        generator: Generator[str, None, None],
        metadata: dict[str, Any] = None
    ) -> StreamingResponse:
        """
        Create streaming response from generator.
        
        Args:
            generator: Token generator
            metadata: Stream metadata
        
        Returns:
            StreamingResponse
        """
        response = StreamingResponse(self.config)
        
        def run_generator():
            response.start(metadata)
            try:
                for token in generator:
                    response.push(token)
                response.finish()
            except Exception as e:
                response.error(e)
        
        # Run in background thread
        thread = threading.Thread(target=run_generator, daemon=True)
        thread.start()
        
        return response
    
    async def astream(
        self,
        generator: AsyncIterator[str],
        metadata: dict[str, Any] = None
    ) -> StreamingResponse:
        """
        Create streaming response from async generator.
        
        Args:
            generator: Async token generator
            metadata: Stream metadata
        
        Returns:
            StreamingResponse
        """
        response = StreamingResponse(self.config)
        response.start(metadata)
        
        try:
            async for token in generator:
                response.push(token)
            response.finish()
        except Exception as e:
            response.error(e)
        
        return response
    
    def stream_callback(
        self,
        callback: Callable[[str], None],
        metadata: dict[str, Any] = None
    ) -> 'CallbackStreamer':
        """
        Create callback-based streamer.
        
        Args:
            callback: Token callback function
            metadata: Stream metadata
        
        Returns:
            CallbackStreamer
        """
        return CallbackStreamer(callback, self.config, metadata)


class CallbackStreamer:
    """Callback-based token streamer."""
    
    def __init__(
        self,
        callback: Callable[[str], None],
        config: StreamingConfig = None,
        metadata: dict[str, Any] = None
    ):
        self.callback = callback
        self.config = config or StreamingConfig()
        self.metadata = metadata or {}
        
        self._tokens: list[str] = []
        self._started = False
        self._finished = False
    
    def __call__(self, token: str):
        """Stream a token."""
        if self._finished:
            return
        
        if not self._started:
            self._started = True
            if self.config.on_start:
                self.config.on_start()
        
        self._tokens.append(token)
        self.callback(token)
        
        if self.config.token_delay > 0:
            time.sleep(self.config.token_delay)
    
    def finish(self):
        """Finish streaming."""
        self._finished = True
        if self.config.on_end:
            self.config.on_end("".join(self._tokens))
    
    def get_text(self) -> str:
        """Get accumulated text."""
        return "".join(self._tokens)


def stream_print(
    generator: Generator[str, None, None],
    end: str = "",
    flush: bool = True
) -> str:
    """
    Stream tokens to print.
    
    Args:
        generator: Token generator
        end: End string
        flush: Whether to flush after each token
    
    Returns:
        Full text
    """
    tokens = []
    for token in generator:
        tokens.append(token)
        print(token, end=end, flush=flush)
    
    if end == "":
        print()  # Final newline
    
    return "".join(tokens)


async def astream_print(
    generator: AsyncIterator[str],
    end: str = "",
    flush: bool = True
) -> str:
    """
    Async stream tokens to print.
    
    Args:
        generator: Async token generator
        end: End string
        flush: Whether to flush after each token
    
    Returns:
        Full text
    """
    tokens = []
    async for token in generator:
        tokens.append(token)
        print(token, end=end, flush=flush)
    
    if end == "":
        print()
    
    return "".join(tokens)


# Global streamer instance
_streamer: Optional[TokenStreamer] = None


def get_token_streamer(config: StreamingConfig = None) -> TokenStreamer:
    """Get or create global token streamer."""
    global _streamer
    if _streamer is None:
        _streamer = TokenStreamer(config)
    return _streamer
