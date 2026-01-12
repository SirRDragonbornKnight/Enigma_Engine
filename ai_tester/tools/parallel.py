"""
Parallel Tool Execution
=======================

Execute multiple tools concurrently for improved performance.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)


class ParallelToolExecutor:
    """
    Execute multiple tools in parallel.
    
    Features:
    - Parallel execution of different tools
    - Batch execution of same tool with different parameters
    - Timeout support
    - Result ordering preservation
    """
    
    def __init__(self, tool_executor=None, max_workers: int = 4):
        """
        Initialize parallel tool executor.
        
        Args:
            tool_executor: ToolExecutor instance to use
            max_workers: Maximum concurrent workers (default: 4)
        """
        self.tool_executor = tool_executor
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"ParallelToolExecutor initialized with {max_workers} workers")
    
    def _get_tool_executor(self):
        """Get or create tool executor instance."""
        if self.tool_executor is None:
            from .tool_executor import ToolExecutor
            self.tool_executor = ToolExecutor()
        return self.tool_executor
    
    def execute_parallel(
        self,
        tool_calls: List[tuple],
        timeout: Optional[float] = None,
        return_exceptions: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple different tools in parallel.
        
        Args:
            tool_calls: List of (tool_name, params) tuples
            timeout: Optional timeout for each tool in seconds
            return_exceptions: If True, exceptions are returned as error results
            
        Returns:
            List of result dictionaries in same order as input
        """
        if not tool_calls:
            return []
        
        executor_instance = self._get_tool_executor()
        
        # Submit all tasks
        future_to_index = {}
        for i, (tool_name, params) in enumerate(tool_calls):
            future = self.executor.submit(
                self._execute_with_timeout,
                executor_instance,
                tool_name,
                params,
                timeout
            )
            future_to_index[future] = i
        
        # Collect results in order
        results = [None] * len(tool_calls)
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            tool_name = tool_calls[index][0]
            
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                logger.exception(f"Error executing {tool_name} in parallel: {e}")
                if return_exceptions:
                    results[index] = {
                        "success": False,
                        "error": str(e),
                        "tool": tool_name,
                    }
                else:
                    raise
        
        return results
    
    def execute_batch(
        self,
        tool_name: str,
        params_list: List[Dict[str, Any]],
        timeout: Optional[float] = None,
        return_exceptions: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute the same tool with different parameters in parallel.
        
        Useful for batch processing (e.g., analyzing multiple images).
        
        Args:
            tool_name: Name of the tool to execute
            params_list: List of parameter dictionaries
            timeout: Optional timeout for each execution in seconds
            return_exceptions: If True, exceptions are returned as error results
            
        Returns:
            List of result dictionaries in same order as params_list
        """
        tool_calls = [(tool_name, params) for params in params_list]
        return self.execute_parallel(tool_calls, timeout, return_exceptions)
    
    def execute_with_callback(
        self,
        tool_calls: List[tuple],
        callback: Callable[[int, Dict[str, Any]], None],
        timeout: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute tools in parallel with a callback for each completion.
        
        Args:
            tool_calls: List of (tool_name, params) tuples
            callback: Function(index, result) called when each tool completes
            timeout: Optional timeout for each tool in seconds
            
        Returns:
            List of result dictionaries in same order as input
        """
        if not tool_calls:
            return []
        
        executor_instance = self._get_tool_executor()
        
        # Submit all tasks
        future_to_index = {}
        for i, (tool_name, params) in enumerate(tool_calls):
            future = self.executor.submit(
                self._execute_with_timeout,
                executor_instance,
                tool_name,
                params,
                timeout
            )
            future_to_index[future] = i
        
        # Collect results with callbacks
        results = [None] * len(tool_calls)
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            tool_name = tool_calls[index][0]
            
            try:
                result = future.result()
                results[index] = result
                
                # Call callback
                try:
                    callback(index, result)
                except Exception as e:
                    logger.exception(f"Error in callback for {tool_name}: {e}")
            
            except Exception as e:
                logger.exception(f"Error executing {tool_name} in parallel: {e}")
                result = {
                    "success": False,
                    "error": str(e),
                    "tool": tool_name,
                }
                results[index] = result
                
                try:
                    callback(index, result)
                except Exception as cb_e:
                    logger.exception(f"Error in callback: {cb_e}")
        
        return results
    
    def _execute_with_timeout(
        self,
        executor_instance,
        tool_name: str,
        params: Dict[str, Any],
        timeout: Optional[float]
    ) -> Dict[str, Any]:
        """Execute a single tool with optional timeout."""
        start_time = time.time()
        
        try:
            if timeout:
                # Use tool executor's timeout method if available
                if hasattr(executor_instance, 'execute_tool_with_timeout'):
                    result = executor_instance.execute_tool_with_timeout(
                        tool_name,
                        params,
                        timeout=int(timeout)
                    )
                else:
                    result = executor_instance.execute_tool(tool_name, params)
            else:
                result = executor_instance.execute_tool(tool_name, params)
            
            duration = (time.time() - start_time) * 1000
            
            # Add duration to result
            if isinstance(result, dict):
                result["duration_ms"] = round(duration, 2)
            
            return result
        
        except Exception as e:
            logger.exception(f"Error in _execute_with_timeout: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "duration_ms": round((time.time() - start_time) * 1000, 2),
            }
    
    def execute_with_progress(
        self,
        tool_calls: List[tuple],
        progress_callback: Callable[[int, int], None],
        timeout: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute tools in parallel with progress tracking.
        
        Args:
            tool_calls: List of (tool_name, params) tuples
            progress_callback: Function(completed, total) called on progress
            timeout: Optional timeout for each tool in seconds
            
        Returns:
            List of result dictionaries in same order as input
        """
        if not tool_calls:
            return []
        
        total = len(tool_calls)
        completed = 0
        
        def callback(index, result):
            nonlocal completed
            completed += 1
            try:
                progress_callback(completed, total)
            except Exception as e:
                logger.exception(f"Error in progress callback: {e}")
        
        return self.execute_with_callback(tool_calls, callback, timeout)
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the executor and cleanup resources.
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        self.executor.shutdown(wait=wait)
        logger.info("ParallelToolExecutor shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown(wait=True)
        return False


__all__ = [
    "ParallelToolExecutor",
]
