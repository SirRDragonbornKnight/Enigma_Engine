"""
Tool Execution History and Logging
===================================

Records tool execution history for debugging, analytics, and auditing.
"""

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolExecution:
    """Record of a single tool execution."""
    
    tool_name: str
    params: dict[str, Any]
    success: bool
    result: Any
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: Optional[float] = None
    cached: bool = False
    rate_limited: bool = False
    timeout: bool = False
    
    def to_dict(self) -> dict[str, Any]:
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
        self.history: list[ToolExecution] = []
        
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
        params: dict[str, Any],
        result: dict[str, Any],
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
    
    def get_recent(self, count: int = 10) -> list[ToolExecution]:
        """
        Get most recent executions.
        
        Args:
            count: Number of executions to return
            
        Returns:
            List of recent executions
        """
        return self.history[-count:]
    
    def get_failures(self, count: Optional[int] = None) -> list[ToolExecution]:
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
    
    def get_by_tool(self, tool_name: str, count: Optional[int] = None) -> list[ToolExecution]:
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
    ) -> list[ToolExecution]:
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
    ) -> dict[str, Any]:
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


class ToolOutputMemory:
    """
    Track outputs from tools to enable editing and referencing previous results.
    
    Allows the AI to say "edit the last image" or "make that video faster"
    by remembering what was generated and where it was saved.
    
    Features:
    - Track last output per tool type (image, video, audio, code, etc.)
    - Track output history (last N outputs per type)
    - Get output by reference ("last", "previous", index)
    - Session-based (clears on new chat)
    """
    
    # Tool categories for grouping outputs
    TOOL_CATEGORIES = {
        'image': ['generate_image', 'edit_image', 'image_gen'],
        'video': ['generate_video', 'edit_video', 'video_gen'],
        'audio': ['generate_audio', 'speak', 'tts', 'audio_gen'],
        'code': ['generate_code', 'code_gen'],
        'gif': ['generate_gif', 'edit_gif', 'gif_gen'],
        '3d': ['generate_3d', 'threed_gen'],
        'embedding': ['generate_embedding', 'embed'],
        'screenshot': ['capture_screen', 'screenshot'],
        'file': ['read_file', 'write_file', 'list_directory'],
    }
    
    def __init__(self, max_per_type: int = 10):
        """
        Initialize output memory.
        
        Args:
            max_per_type: Maximum outputs to remember per tool type
        """
        self.max_per_type = max_per_type
        
        # Last output per category
        self._last_output: dict[str, dict[str, Any]] = {}
        
        # Output history per category (newest first)
        self._history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        
        # All outputs in order (for "the 3rd thing you made")
        self._all_outputs: list[dict[str, Any]] = []
        
        logger.info(f"ToolOutputMemory initialized (max_per_type={max_per_type})")
    
    def _get_category(self, tool_name: str) -> str:
        """Get the category for a tool name."""
        tool_lower = tool_name.lower()
        for category, tools in self.TOOL_CATEGORIES.items():
            if any(t in tool_lower for t in tools):
                return category
        return 'other'
    
    def record_output(
        self,
        tool_name: str,
        output_path: Optional[str] = None,
        output_data: Any = None,
        params: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None
    ):
        """
        Record a tool output for later reference.
        
        Args:
            tool_name: Name of the tool that produced the output
            output_path: File path if output was saved to disk
            output_data: The actual output data (if small enough to keep in memory)
            params: Parameters used to generate the output
            metadata: Additional metadata (dimensions, duration, etc.)
        """
        category = self._get_category(tool_name)
        
        record = {
            'tool': tool_name,
            'category': category,
            'path': output_path,
            'data': output_data,
            'params': params or {},
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'index': len(self._all_outputs),
        }
        
        # Update last output for this category
        self._last_output[category] = record
        
        # Add to category history
        self._history[category].insert(0, record)
        if len(self._history[category]) > self.max_per_type:
            self._history[category].pop()
        
        # Add to all outputs
        self._all_outputs.append(record)
        
        logger.debug(f"Recorded {category} output: {output_path or 'in-memory'}")
    
    def get_last(self, category: Optional[str] = None) -> Optional[dict[str, Any]]:
        """
        Get the last output, optionally filtered by category.
        
        Args:
            category: Filter by category (image, video, etc.) or None for any
            
        Returns:
            Last output record or None
        """
        if category:
            return self._last_output.get(category)
        elif self._all_outputs:
            return self._all_outputs[-1]
        return None
    
    def get_by_reference(self, reference: str) -> Optional[dict[str, Any]]:
        """
        Get output by natural language reference.
        
        Supports:
        - "last image", "previous video", "that audio"
        - "image 2", "video 3" (1-indexed from most recent)
        - "the 5th thing" (overall index)
        
        Args:
            reference: Natural language reference
            
        Returns:
            Output record or None
        """
        ref_lower = reference.lower().strip()
        
        # Check for category keywords
        for category in self.TOOL_CATEGORIES.keys():
            if category in ref_lower:
                # Check for index
                import re
                index_match = re.search(r'(\d+)', ref_lower)
                if index_match:
                    idx = int(index_match.group(1)) - 1  # Convert to 0-indexed
                    history = self._history.get(category, [])
                    if 0 <= idx < len(history):
                        return history[idx]
                
                # Default to last
                if 'last' in ref_lower or 'previous' in ref_lower or 'that' in ref_lower:
                    return self._last_output.get(category)
                
                # Just category mentioned, return last
                return self._last_output.get(category)
        
        # Check for overall index ("the 3rd thing", "output 5")
        import re
        ordinal_match = re.search(r'(\d+)(?:st|nd|rd|th)?', ref_lower)
        if ordinal_match:
            idx = int(ordinal_match.group(1)) - 1
            if 0 <= idx < len(self._all_outputs):
                return self._all_outputs[idx]
        
        # Default to absolute last
        if 'last' in ref_lower or 'previous' in ref_lower or 'that' in ref_lower:
            return self.get_last()
        
        return None
    
    def get_history(self, category: Optional[str] = None, limit: int = 5) -> list[dict[str, Any]]:
        """
        Get output history.
        
        Args:
            category: Filter by category or None for all
            limit: Maximum number of outputs to return
            
        Returns:
            List of output records (newest first)
        """
        if category:
            return self._history.get(category, [])[:limit]
        else:
            return self._all_outputs[-limit:][::-1]  # Newest first
    
    def get_editable_path(self, reference: str) -> Optional[str]:
        """
        Get the file path for an output that can be edited.
        
        Args:
            reference: Natural language reference
            
        Returns:
            File path or None
        """
        output = self.get_by_reference(reference)
        if output and output.get('path'):
            path = Path(output['path'])
            if path.exists():
                return str(path)
        return None
    
    def get_context_for_ai(self) -> str:
        """
        Generate context string for AI about recent outputs.
        
        Returns:
            Context string describing recent outputs
        """
        if not self._all_outputs:
            return "No previous outputs in this session."
        
        lines = ["Recent outputs in this session:"]
        
        # Show last output per category
        for category, output in self._last_output.items():
            path = output.get('path', 'in-memory')
            params = output.get('params', {})
            prompt = params.get('prompt', params.get('text', ''))[:50]
            lines.append(f"- Last {category}: {path}")
            if prompt:
                lines.append(f"  Prompt: \"{prompt}...\"")
        
        return "\n".join(lines)
    
    def clear(self, category: Optional[str] = None):
        """
        Clear output memory.
        
        Args:
            category: Clear only this category or None for all
        """
        if category:
            self._history[category].clear()
            self._last_output.pop(category, None)
            logger.info(f"Cleared {category} output memory")
        else:
            self._history.clear()
            self._last_output.clear()
            self._all_outputs.clear()
            logger.info("Cleared all output memory")


# Global instance for easy access
_output_memory: Optional[ToolOutputMemory] = None


def get_output_memory() -> ToolOutputMemory:
    """Get or create the global output memory instance."""
    global _output_memory
    if _output_memory is None:
        _output_memory = ToolOutputMemory()
    return _output_memory


def record_tool_output(
    tool_name: str,
    output_path: Optional[str] = None,
    output_data: Any = None,
    params: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None
):
    """Convenience function to record a tool output."""
    get_output_memory().record_output(tool_name, output_path, output_data, params, metadata)


def get_last_output(category: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Convenience function to get last output."""
    return get_output_memory().get_last(category)


def get_output_by_reference(reference: str) -> Optional[dict[str, Any]]:
    """Convenience function to get output by reference."""
    return get_output_memory().get_by_reference(reference)


__all__ = [
    "ToolExecution",
    "ToolExecutionHistory",
    "ToolOutputMemory",
    "get_output_memory",
    "record_tool_output",
    "get_last_output",
    "get_output_by_reference",
]
