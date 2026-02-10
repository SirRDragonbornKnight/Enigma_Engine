"""
Tool Result Streaming
======================

Provides streaming results for long-running tools.
"""

import logging
import queue
import threading
from collections.abc import Iterator
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """State of a streaming result."""
    PENDING = "pending"
    STREAMING = "streaming"
    COMPLETE = "complete"
    ERROR = "error"


class StreamingToolResult:
    """
    Streaming result container for long-running tools.
    
    Allows tools to yield partial results as they become available.
    """
    
    def __init__(self, tool_name: str, max_queue_size: int = 100):
        """
        Initialize streaming result.
        
        Args:
            tool_name: Name of the tool producing results
            max_queue_size: Maximum items in result queue
        """
        self.tool_name = tool_name
        self.max_queue_size = max_queue_size
        
        # Result queue
        self._queue = queue.Queue(maxsize=max_queue_size)
        
        # State tracking
        self._state = StreamState.PENDING
        self._error: Optional[str] = None
        self._metadata: dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.debug(f"StreamingToolResult created for {tool_name}")
    
    def put(self, item: Any, timeout: Optional[float] = None):
        """
        Add an item to the stream.
        
        Args:
            item: Item to add to stream
            timeout: Optional timeout for queue.put
        """
        with self._lock:
            if self._state == StreamState.COMPLETE:
                raise RuntimeError("Cannot put items after stream is complete")
            
            if self._state == StreamState.PENDING:
                self._state = StreamState.STREAMING
        
        try:
            self._queue.put(item, timeout=timeout)
        except queue.Full:
            logger.warning(f"Stream queue full for {self.tool_name}")
            raise
    
    def done(self, metadata: Optional[dict[str, Any]] = None):
        """
        Mark stream as complete.
        
        Args:
            metadata: Optional metadata about completion
        """
        with self._lock:
            self._state = StreamState.COMPLETE
            if metadata:
                self._metadata.update(metadata)
        
        # Put sentinel value to unblock iterators
        self._queue.put(None)
        
        logger.debug(f"Stream complete for {self.tool_name}")
    
    def error(self, error_message: str):
        """
        Mark stream as errored.
        
        Args:
            error_message: Error message
        """
        with self._lock:
            self._state = StreamState.ERROR
            self._error = error_message
        
        # Put sentinel value to unblock iterators
        self._queue.put(None)
        
        logger.error(f"Stream error for {self.tool_name}: {error_message}")
    
    def __iter__(self) -> Iterator[Any]:
        """Iterate over stream results."""
        while True:
            try:
                item = self._queue.get(timeout=1.0)
                
                # Check for sentinel (None indicates done)
                if item is None:
                    break
                
                yield item
            
            except queue.Empty:
                # Check if stream is complete
                with self._lock:
                    if self._state in (StreamState.COMPLETE, StreamState.ERROR):
                        break
                continue
    
    def get_state(self) -> StreamState:
        """Get current stream state."""
        with self._lock:
            return self._state
    
    def get_error(self) -> Optional[str]:
        """Get error message if stream errored."""
        with self._lock:
            return self._error
    
    def get_metadata(self) -> dict[str, Any]:
        """Get stream metadata."""
        with self._lock:
            return self._metadata.copy()
    
    def is_complete(self) -> bool:
        """Check if stream is complete."""
        with self._lock:
            return self._state in (StreamState.COMPLETE, StreamState.ERROR)


class StreamingToolExecutor:
    """
    Execute tools with streaming results.
    
    Wraps regular tool execution to support streaming.
    """
    
    def __init__(self, tool_executor=None):
        """
        Initialize streaming executor.
        
        Args:
            tool_executor: ToolExecutor instance to use
        """
        self.tool_executor = tool_executor
        logger.info("StreamingToolExecutor initialized")
    
    def _get_tool_executor(self):
        """Get or create tool executor instance."""
        if self.tool_executor is None:
            from .tool_executor import ToolExecutor
            self.tool_executor = ToolExecutor()
        return self.tool_executor
    
    def execute_streaming(
        self,
        tool_name: str,
        params: dict[str, Any],
        chunk_size: Optional[int] = None
    ) -> StreamingToolResult:
        """
        Execute a tool with streaming results.
        
        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters
            chunk_size: Optional chunk size for streaming
            
        Returns:
            StreamingToolResult object
        """
        stream = StreamingToolResult(tool_name)
        
        # Execute in background thread
        def execute_worker():
            try:
                executor = self._get_tool_executor()
                result = executor.execute_tool(tool_name, params)
                
                # For tools that don't natively stream, we can still wrap the result
                if result.get("success"):
                    # Try to chunk the result if it's large
                    result_data = result.get("result", "")
                    
                    if isinstance(result_data, str) and chunk_size:
                        # Stream in chunks
                        for i in range(0, len(result_data), chunk_size):
                            chunk = result_data[i:i+chunk_size]
                            stream.put({"chunk": chunk, "index": i})
                    else:
                        # Put entire result
                        stream.put(result)
                    
                    stream.done({"total_size": len(str(result_data))})
                else:
                    stream.error(result.get("error", "Unknown error"))
            
            except Exception as e:
                logger.exception(f"Error in streaming execution: {e}")
                stream.error(str(e))
        
        # Start background thread
        thread = threading.Thread(target=execute_worker, daemon=True)
        thread.start()
        
        return stream
    
    def execute_streaming_batch(
        self,
        tool_calls: list,
        chunk_size: Optional[int] = None
    ) -> dict[int, StreamingToolResult]:
        """
        Execute multiple tools with streaming results.
        
        Args:
            tool_calls: List of (tool_name, params) tuples
            chunk_size: Optional chunk size for streaming
            
        Returns:
            Dictionary mapping index to StreamingToolResult
        """
        streams = {}
        
        for i, (tool_name, params) in enumerate(tool_calls):
            stream = self.execute_streaming(tool_name, params, chunk_size)
            streams[i] = stream
        
        return streams


__all__ = [
    "StreamingToolResult",
    "StreamingToolExecutor",
    "StreamState",
]
