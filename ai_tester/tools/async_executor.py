"""
Async Tool Executor for AI Tester
==================================

Provides asynchronous execution of tools using ThreadPoolExecutor.
Enables non-blocking tool execution and concurrent operations.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import time

logger = logging.getLogger(__name__)


class AsyncToolExecutor:
    """
    Execute tools asynchronously using ThreadPoolExecutor.
    
    Allows for non-blocking tool execution and concurrent operations,
    which is especially useful for tools that take a long time to complete
    or when executing multiple tools in parallel.
    """
    
    def __init__(self, tool_executor=None, max_workers: int = 4):
        """
        Initialize async tool executor.
        
        Args:
            tool_executor: ToolExecutor instance to use for actual execution
            max_workers: Maximum number of concurrent workers (default: 4)
        """
        self.tool_executor = tool_executor
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"AsyncToolExecutor initialized with {max_workers} workers")
    
    def _get_tool_executor(self):
        """Get or create tool executor instance."""
        if self.tool_executor is None:
            from .tool_executor import ToolExecutor
            self.tool_executor = ToolExecutor()
        return self.tool_executor
    
    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a single tool asynchronously.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            timeout: Optional timeout in seconds
            
        Returns:
            Result dictionary from tool execution
        """
        loop = asyncio.get_event_loop()
        executor_instance = self._get_tool_executor()
        
        try:
            # Run sync tool execution in thread pool
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.executor,
                    executor_instance.execute_tool,
                    tool_name,
                    params
                ),
                timeout=timeout
            )
            
            return result
        
        except asyncio.TimeoutError:
            logger.warning(f"Tool {tool_name} timed out after {timeout}s")
            return {
                "success": False,
                "error": f"Tool execution timed out after {timeout} seconds",
                "tool": tool_name,
                "timeout": True,
            }
        
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name} asynchronously: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
            }
    
    async def execute_multiple(
        self,
        tool_calls: List[tuple],
        timeout: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tools concurrently.
        
        Args:
            tool_calls: List of (tool_name, params) tuples
            timeout: Optional timeout for each tool in seconds
            
        Returns:
            List of result dictionaries in same order as input
        """
        tasks = [
            self.execute_tool(tool_name, params, timeout=timeout)
            for tool_name, params in tool_calls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tool_name = tool_calls[i][0]
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "tool": tool_name,
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def execute_with_timeout(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: float
    ) -> Dict[str, Any]:
        """
        Execute a tool with a strict timeout.
        
        This is a convenience method that wraps execute_tool with a timeout.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            timeout: Timeout in seconds
            
        Returns:
            Result dictionary from tool execution
        """
        return await self.execute_tool(tool_name, params, timeout=timeout)
    
    async def execute_with_callback(
        self,
        tool_name: str,
        params: Dict[str, Any],
        callback: Callable[[Dict[str, Any]], None],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool and call a callback function when done.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            callback: Function to call with the result
            timeout: Optional timeout in seconds
            
        Returns:
            Result dictionary from tool execution
        """
        result = await self.execute_tool(tool_name, params, timeout=timeout)
        
        try:
            callback(result)
        except Exception as e:
            logger.exception(f"Error in callback for tool {tool_name}: {e}")
        
        return result
    
    def execute_tool_sync(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool asynchronously but wait for the result (blocking).
        
        Useful when you want to use the async executor from synchronous code.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            timeout: Optional timeout in seconds
            
        Returns:
            Result dictionary from tool execution
        """
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new one in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.execute_tool(tool_name, params, timeout)
                    )
                    return future.result()
            else:
                # Run in existing loop
                return loop.run_until_complete(
                    self.execute_tool(tool_name, params, timeout)
                )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.execute_tool(tool_name, params, timeout))
    
    def execute_multiple_sync(
        self,
        tool_calls: List[tuple],
        timeout: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tools concurrently but wait for all results (blocking).
        
        Args:
            tool_calls: List of (tool_name, params) tuples
            timeout: Optional timeout for each tool in seconds
            
        Returns:
            List of result dictionaries in same order as input
        """
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new one in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.execute_multiple(tool_calls, timeout)
                    )
                    return future.result()
            else:
                # Run in existing loop
                return loop.run_until_complete(
                    self.execute_multiple(tool_calls, timeout)
                )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.execute_multiple(tool_calls, timeout))
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the executor and cleanup resources.
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        self.executor.shutdown(wait=wait)
        logger.info("AsyncToolExecutor shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown(wait=True)
        return False


__all__ = [
    "AsyncToolExecutor",
]
