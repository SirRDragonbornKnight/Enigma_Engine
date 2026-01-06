"""
Tool Execution History and Logging
===================================

Records tool execution history for debugging, analytics, and auditing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ToolExecution:
    """Record of a single tool execution."""
    
    tool_name: str
    params: Dict[str, Any]
    success: bool
    result: Any
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: Optional[float] = None
    cached: bool = False
    rate_limited: bool = False
    timeout: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ToolExecutionHistory:
    """
    Track and query tool execution history.
    
    Features:
    - In-memory history with configurable size limit
    - Optional file logging
    - Query by tool, time range, success/failure
    - Statistics generation
    """
    
    def __init__(
        self,
        max_history: int = 1000,
        log_file: Optional[Path] = None,
        enable_file_logging: bool = False
    ):
        """
        Initialize execution history.
        
        Args:
            max_history: Maximum executions to keep in memory
            log_file: Optional path for file logging
            enable_file_logging: Whether to log to file
        """
        self.max_history = max_history
        self.enable_file_logging = enable_file_logging
        
        # In-memory history (circular buffer)
        self.history: List[ToolExecution] = []
        
        # Log file
        if log_file is None and enable_file_logging:
            log_file = Path("data/tool_execution.log")
        self.log_file = log_file
        
        if self.enable_file_logging and self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ToolExecutionHistory initialized (max={max_history}, file_logging={enable_file_logging})")
    
    def record(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Dict[str, Any],
        duration_ms: Optional[float] = None
    ):
        """
        Record a tool execution.
        
        Args:
            tool_name: Name of the tool
            params: Tool parameters
            result: Tool result
            duration_ms: Execution duration in milliseconds
        """
        execution = ToolExecution(
            tool_name=tool_name,
            params=params,
            success=result.get("success", False),
            result=result.get("result"),
            error=result.get("error"),
            duration_ms=duration_ms,
            cached=result.get("cached", False),
            rate_limited=result.get("rate_limited", False),
            timeout=result.get("timeout", False),
        )
        
        # Add to memory history
        self.history.append(execution)
        
        # Maintain max size (circular buffer)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Log to file if enabled
        if self.enable_file_logging and self.log_file:
            self._log_to_file(execution)
    
    def _log_to_file(self, execution: ToolExecution):
        """Write execution to log file."""
        try:
            with open(self.log_file, 'a') as f:
                log_entry = json.dumps(execution.to_dict())
                f.write(log_entry + '\n')
        except Exception as e:
            logger.warning(f"Failed to write to execution log: {e}")
    
    def get_recent(self, count: int = 10) -> List[ToolExecution]:
        """
        Get most recent executions.
        
        Args:
            count: Number of executions to return
            
        Returns:
            List of recent executions
        """
        return self.history[-count:]
    
    def get_failures(self, count: Optional[int] = None) -> List[ToolExecution]:
        """
        Get failed executions.
        
        Args:
            count: Maximum number to return (None for all)
            
        Returns:
            List of failed executions
        """
        failures = [ex for ex in self.history if not ex.success]
        
        if count is not None:
            return failures[-count:]
        return failures
    
    def get_by_tool(self, tool_name: str, count: Optional[int] = None) -> List[ToolExecution]:
        """
        Get executions for a specific tool.
        
        Args:
            tool_name: Name of the tool
            count: Maximum number to return (None for all)
            
        Returns:
            List of executions for the tool
        """
        executions = [ex for ex in self.history if ex.tool_name == tool_name]
        
        if count is not None:
            return executions[-count:]
        return executions
    
    def get_by_time_range(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[ToolExecution]:
        """
        Get executions within a time range.
        
        Args:
            start_time: Start of time range (None for no lower bound)
            end_time: End of time range (None for no upper bound)
            
        Returns:
            List of executions in range
        """
        results = []
        
        for ex in self.history:
            ex_time = datetime.fromisoformat(ex.timestamp)
            
            if start_time and ex_time < start_time:
                continue
            if end_time and ex_time > end_time:
                continue
            
            results.append(ex)
        
        return results
    
    def get_statistics(
        self,
        tool_name: Optional[str] = None,
        time_period: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Args:
            tool_name: Filter by tool name (None for all)
            time_period: Filter by time period (None for all)
            
        Returns:
            Statistics dictionary
        """
        # Filter executions
        if time_period:
            cutoff_time = datetime.now() - time_period
            executions = self.get_by_time_range(start_time=cutoff_time)
        else:
            executions = self.history
        
        if tool_name:
            executions = [ex for ex in executions if ex.tool_name == tool_name]
        
        if not executions:
            return {
                "total_executions": 0,
                "success_count": 0,
                "failure_count": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "tools_used": [],
            }
        
        # Calculate statistics
        success_count = sum(1 for ex in executions if ex.success)
        failure_count = len(executions) - success_count
        
        # Calculate average duration (exclude None values)
        durations = [ex.duration_ms for ex in executions if ex.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        
        # Count per tool
        tool_counts = defaultdict(int)
        tool_failures = defaultdict(int)
        tool_durations = defaultdict(list)
        
        for ex in executions:
            tool_counts[ex.tool_name] += 1
            if not ex.success:
                tool_failures[ex.tool_name] += 1
            if ex.duration_ms is not None:
                tool_durations[ex.tool_name].append(ex.duration_ms)
        
        # Per-tool stats
        per_tool_stats = {}
        for tool, count in tool_counts.items():
            failures = tool_failures[tool]
            successes = count - failures
            success_rate = (successes / count * 100) if count > 0 else 0
            
            tool_duration_list = tool_durations[tool]
            avg_tool_duration = sum(tool_duration_list) / len(tool_duration_list) if tool_duration_list else 0
            
            per_tool_stats[tool] = {
                "count": count,
                "success_count": successes,
                "failure_count": failures,
                "success_rate": round(success_rate, 2),
                "avg_duration_ms": round(avg_tool_duration, 2),
            }
        
        # Count special cases
        cached_count = sum(1 for ex in executions if ex.cached)
        rate_limited_count = sum(1 for ex in executions if ex.rate_limited)
        timeout_count = sum(1 for ex in executions if ex.timeout)
        
        return {
            "total_executions": len(executions),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": round(success_count / len(executions) * 100, 2),
            "avg_duration_ms": round(avg_duration, 2),
            "cached_count": cached_count,
            "rate_limited_count": rate_limited_count,
            "timeout_count": timeout_count,
            "tools_used": list(tool_counts.keys()),
            "per_tool": per_tool_stats,
        }
    
    def clear(self, tool_name: Optional[str] = None):
        """
        Clear execution history.
        
        Args:
            tool_name: Clear only for this tool (None for all)
        """
        if tool_name is None:
            self.history.clear()
            logger.info("Cleared all execution history")
        else:
            self.history = [ex for ex in self.history if ex.tool_name != tool_name]
            logger.info(f"Cleared execution history for {tool_name}")
    
    def export_to_json(self, filepath: Path):
        """
        Export history to JSON file.
        
        Args:
            filepath: Path to export file
        """
        try:
            data = [ex.to_dict() for ex in self.history]
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported {len(data)} executions to {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to export history: {e}")


__all__ = [
    "ToolExecution",
    "ToolExecutionHistory",
]
