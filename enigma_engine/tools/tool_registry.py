"""
Tool System for Forge

Provides a unified interface for all AI capabilities:
  - Web search
  - File operations
  - Document reading
  - Screenshots
  - System commands

Each tool has:
  - name: Unique identifier
  - description: What it does (for AI to understand)
  - execute(): Run the tool with parameters

USAGE:
    from enigma_engine.tools import ToolRegistry
    
    tools = ToolRegistry()
    
    # List available tools
    tools.list_tools()
    
    # Execute a tool
    result = tools.execute("web_search", query="python tutorials")
    result = tools.execute("read_file", path="/home/user/document.txt")
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RichParameter:
    """
    Rich parameter definition for tools.
    
    This provides detailed information about each parameter so the AI
    knows exactly what to pass and users understand what's expected.
    """
    name: str
    type: str  # "string", "int", "float", "bool", "list", "dict"
    description: str
    required: bool = True
    default: Any = None
    enum: List[Any] = field(default_factory=list)  # Valid choices
    min_value: Optional[float] = None  # For numeric types
    max_value: Optional[float] = None  # For numeric types
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for AI schemas."""
        result = {
            "type": self.type,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            result["default"] = self.default
        if self.enum:
            result["enum"] = self.enum
        if self.min_value is not None:
            result["min"] = self.min_value
        if self.max_value is not None:
            result["max"] = self.max_value
        return result


class Tool(ABC):
    """
    Base class for all tools.
    
    Tools can define parameters in two ways:
    1. Simple: parameters = {"name": "description"}
    2. Rich: rich_parameters = [RichParameter(...), ...]
    
    Rich parameters provide more detail for the AI to understand
    parameter types, valid values, defaults, etc.
    """
    
    name: str = "base_tool"
    description: str = "Base tool - override this"
    parameters: dict[str, str] = {}  # Simple: param_name: description
    rich_parameters: List[RichParameter] = []  # Rich parameter definitions
    category: str = "general"  # Tool category for grouping
    examples: List[str] = []  # Usage examples for the AI
    version: str = "1.0.0"  # Tool version
    
    @abstractmethod
    def execute(self, **kwargs) -> dict[str, Any]:
        """
        Execute the tool.
        
        Returns:
            Dict with 'success', 'result', and optionally 'error'
        """
    
    def to_dict(self) -> Dict:
        """Export tool info for AI consumption."""
        result = {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
        }
        
        # Use rich parameters if available, otherwise fall back to simple
        if self.rich_parameters:
            result["parameters"] = {
                p.name: p.to_dict() for p in self.rich_parameters
            }
        else:
            result["parameters"] = self.parameters
        
        if self.examples:
            result["examples"] = self.examples
        
        return result
    
    def get_schema(self) -> str:
        """Get human-readable schema for the tool."""
        lines = [f"{self.name} (v{self.version})", f"  {self.description}", ""]
        
        if self.rich_parameters:
            lines.append("  Parameters:")
            for p in self.rich_parameters:
                req = "required" if p.required else "optional"
                default = f" = {p.default}" if p.default is not None else ""
                lines.append(f"    {p.name} ({p.type}, {req}){default}")
                lines.append(f"      {p.description}")
                if p.enum:
                    lines.append(f"      Choices: {', '.join(str(e) for e in p.enum)}")
        elif self.parameters:
            lines.append("  Parameters:")
            for name, desc in self.parameters.items():
                lines.append(f"    {name}: {desc}")
        
        if self.examples:
            lines.append("")
            lines.append("  Examples:")
            for ex in self.examples[:3]:
                lines.append(f"    - {ex}")
        
        return "\n".join(lines)


class ToolRegistry:
    """
    Registry of all available tools.
    
    Respects tool_manager settings - disabled tools won't be registered.
    """
    
    # Tools that can safely be cached (read-only operations)
    CACHEABLE_TOOLS = {
        "get_system_info",
        "wikipedia_search",
        "arxiv_search",
    }
    
    def __init__(self, respect_manager: bool = True):
        self.tools: Dict[str, Tool] = {}
        self._respect_manager = respect_manager
        self._cache: Dict[str, tuple] = {}  # key -> (result, timestamp)
        self._cache_ttl = 300  # 5 minutes default
        self._register_builtin_tools()
    
    def _is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled in tool_manager."""
        if not self._respect_manager:
            return True
        try:
            from .tool_manager import get_tool_manager
            manager = get_tool_manager()
            return manager.is_enabled(tool_name)
        except Exception:
            return True  # Default to enabled if manager fails
    
    def _register_builtin_tools(self):
        """Register all built-in tools (respects tool_manager settings)."""
        # New tool imports
        from .automation_tools import (
            ClipboardHistoryTool,
            ClipboardReadTool,
            ClipboardWriteTool,
            ListSchedulesTool,
            PlayMacroTool,
            RecordMacroTool,
            RemoveScheduleTool,
            ScheduleTaskTool,
        )

        # Avatar tools - AI can control and customize the desktop avatar
        from .avatar_tools import (
            AdjustIdleAnimationTool,
            AvatarControlTool,
            AvatarCustomizeTool,
            AvatarEmotionTool,
            AvatarGestureTool,
        )
        from .browser_tools import (
            BrowserFocusTool,
            BrowserMediaInfoTool,
            BrowserMediaMuteTool,
            BrowserMediaPauseTool,
            BrowserMediaSkipTool,
            BrowserMediaStopTool,
            BrowserMediaVolumeTool,
            BrowserTabListTool,
        )
        from .communication_tools import OCRImageTool, TranslateTextTool
        from .data_tools import (
            CSVAnalyzeTool,
            CSVQueryTool,
            DataConvertTool,
            JSONQueryTool,
            PlotChartTool,
            SQLExecuteTool,
            SQLQueryTool,
        )
        from .document_tools import ExtractTextTool, ReadDocumentTool
        from .file_tools import (
            DeleteFileTool,
            ListDirectoryTool,
            MoveFileTool,
            ReadFileTool,
            WriteFileTool,
        )

        # Gaming tools removed - AI can roleplay, tell stories, and play games natively
        # Keep only DnD dice roller since it needs true randomness
        from .gaming_tools import DnDRollTool
        from .interactive_tools import (
            AddTaskTool,
            CheckRemindersTool,
            CompleteTaskTool,
            CreateChecklistTool,
            ListChecklistsTool,
            ListRemindersTool,
            ListTasksTool,
            SetReminderTool,
        )
        from .iot_tools import (
            CameraCaptureTool,
            CameraListTool,
            CameraStreamTool,
            GPIOPWMTool,
            GPIOReadTool,
            GPIOWriteTool,
            HomeAssistantControlTool,
            HomeAssistantSetupTool,
            HomeAssistantStatusTool,
            MQTTPublishTool,
            MQTTSubscribeTool,
        )
        from .knowledge_tools import (
            ArxivSearchTool,
            PDFExtractTool,
            WikipediaSearchTool,
        )
        from .media_tools import (
            AudioVisualizeTool,
            ConvertAudioTool,
            ExtractAudioTool,
            MusicGenerateTool,
            RemoveBackgroundTool,
            StyleTransferTool,
            UpscaleImageTool,
        )
        from .productivity_tools import (
            DockerControlTool,
            DockerListTool,
            GitCommitTool,
            GitPullTool,
            GitPushTool,
            GitStatusTool,
            ProcessKillTool,
            ProcessListTool,
            SSHExecuteTool,
            SystemMonitorTool,
        )
        from .robot_tools import (
            RobotGripperTool,
            RobotHomeTool,
            RobotMoveTool,
            RobotStatusTool,
        )

        # Self-modification tools - AI can customize itself
        from .self_tools import (
            FullscreenModeControlTool,
            GenerateAvatarTool,
            GetSelfConfigTool,
            ListAvatarsTool,
            ListEffectAssetsTool,
            ListSpawnedObjectsTool,
            OpenAvatarInBlenderTool,
            RecallFactsTool,
            RememberFactTool,
            RemoveObjectTool,
            SetAvatarPreferenceTool,
            SetAvatarTool,
            SetCompanionBehaviorTool,
            SetPersonalityTool,
            SetPreferenceTool,
            SetVoicePreferenceTool,
            SpawnObjectTool,
            SpawnScreenEffectTool,
            StopScreenEffectTool,
        )
        from .system_tools import GetSystemInfoTool, RunCommandTool, ScreenshotTool
        from .vision import FindOnScreenTool, ScreenVisionTool
        from .web_tools import FetchWebpageTool, WebSearchTool
        
        # Memory tools - AI can search and manage conversation history
        from .memory_tools import (
            ExportMemoryTool,
            ImportMemoryTool,
            MemoryStatsTool,
            SearchMemoryTool,
        )
        
        builtin = [
            # Web
            WebSearchTool(),
            FetchWebpageTool(),
            # Files
            ReadFileTool(),
            WriteFileTool(),
            ListDirectoryTool(),
            MoveFileTool(),
            DeleteFileTool(),
            # Documents
            ReadDocumentTool(),
            ExtractTextTool(),
            # System
            RunCommandTool(),
            ScreenshotTool(),
            GetSystemInfoTool(),
            # Vision
            ScreenVisionTool(),
            FindOnScreenTool(),
            # Robot
            RobotMoveTool(),
            RobotGripperTool(),
            RobotStatusTool(),
            RobotHomeTool(),
            # Interactive/Personal Assistant
            CreateChecklistTool(),
            ListChecklistsTool(),
            AddTaskTool(),
            ListTasksTool(),
            CompleteTaskTool(),
            SetReminderTool(),
            ListRemindersTool(),
            CheckRemindersTool(),
            # Automation
            ScheduleTaskTool(),
            ListSchedulesTool(),
            RemoveScheduleTool(),
            ClipboardReadTool(),
            ClipboardWriteTool(),
            ClipboardHistoryTool(),
            RecordMacroTool(),
            PlayMacroTool(),
            # Knowledge
            WikipediaSearchTool(),
            ArxivSearchTool(),
            PDFExtractTool(),
            # Communication
            TranslateTextTool(),
            # DetectLanguageTool removed - AI detects language natively
            OCRImageTool(),
            # Media
            MusicGenerateTool(),
            RemoveBackgroundTool(),
            UpscaleImageTool(),
            StyleTransferTool(),
            ConvertAudioTool(),
            ExtractAudioTool(),
            AudioVisualizeTool(),
            # Productivity
            SystemMonitorTool(),
            ProcessListTool(),
            ProcessKillTool(),
            SSHExecuteTool(),
            DockerListTool(),
            DockerControlTool(),
            GitStatusTool(),
            GitCommitTool(),
            GitPushTool(),
            GitPullTool(),
            # IoT
            HomeAssistantSetupTool(),
            HomeAssistantControlTool(),
            HomeAssistantStatusTool(),
            GPIOReadTool(),
            GPIOWriteTool(),
            GPIOPWMTool(),
            MQTTPublishTool(),
            MQTTSubscribeTool(),
            CameraCaptureTool(),
            CameraListTool(),
            CameraStreamTool(),
            # Data
            CSVAnalyzeTool(),
            CSVQueryTool(),
            PlotChartTool(),
            JSONQueryTool(),
            SQLQueryTool(),
            SQLExecuteTool(),
            DataConvertTool(),
            # Gaming - Only dice roller kept (needs true randomness)
            # Other gaming tools removed - AI can roleplay, tell stories, play trivia natively
            DnDRollTool(),
            # Browser Media Control
            BrowserMediaPauseTool(),
            BrowserMediaMuteTool(),
            BrowserMediaSkipTool(),
            BrowserMediaStopTool(),
            BrowserMediaVolumeTool(),
            BrowserMediaInfoTool(),
            BrowserTabListTool(),
            BrowserFocusTool(),
            # Avatar Control - AI controls the desktop avatar
            AvatarControlTool(),
            AvatarCustomizeTool(),
            AvatarGestureTool(),
            AvatarEmotionTool(),
            AdjustIdleAnimationTool(),
            # Self-Modification - AI customizes itself
            SetPersonalityTool(),
            SetAvatarPreferenceTool(),
            SetVoicePreferenceTool(),
            SetCompanionBehaviorTool(),
            SetPreferenceTool(),
            GetSelfConfigTool(),
            RememberFactTool(),
            RecallFactsTool(),
            # Avatar Generation - AI can generate and manage 3D avatars
            GenerateAvatarTool(),
            OpenAvatarInBlenderTool(),
            ListAvatarsTool(),
            SetAvatarTool(),
            # Object Spawning - AI can spawn objects on screen
            SpawnObjectTool(),
            RemoveObjectTool(),
            ListSpawnedObjectsTool(),
            # Screen Effects - AI can spawn visual effects anywhere
            SpawnScreenEffectTool(),
            StopScreenEffectTool(),
            ListEffectAssetsTool(),
            # Fullscreen Mode - AI can control visibility during fullscreen apps
            FullscreenModeControlTool(),
            # Memory - AI can search and manage conversation history
            SearchMemoryTool(),
            MemoryStatsTool(),
            ExportMemoryTool(),
            ImportMemoryTool(),
        ]
        
        for tool in builtin:
            # Only register if enabled in tool_manager
            if self._is_tool_enabled(tool.name):
                self.register(tool)
    
    def register(self, tool: Tool, force: bool = False):
        """Register a tool. Set force=True to bypass tool_manager check."""
        if force or self._is_tool_enabled(tool.name):
            self.tools[tool.name] = tool
    
    def unregister(self, name: str):
        """Remove a tool."""
        if name in self.tools:
            del self.tools[name]
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Tool name (renamed from 'name' to avoid conflicts with tool params)
            **kwargs: Tool parameters
            
        Returns:
            Tool result dict
        """
        tool = self.get(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}
        
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def cached_execute(self, tool_name: str, ttl: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool with caching for cacheable tools.
        
        Args:
            tool_name: Tool name
            ttl: Cache time-to-live in seconds (default: 300)
            **kwargs: Tool parameters
            
        Returns:
            Tool result dict (may be cached)
        """
        # Non-cacheable tools execute directly
        if tool_name not in self.CACHEABLE_TOOLS:
            return self.execute(tool_name, **kwargs)
        
        # Create cache key from tool name and sorted params
        cache_key = f"{tool_name}:{hash(frozenset(kwargs.items()))}"
        ttl = ttl or self._cache_ttl
        
        # Check cache
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < ttl:
                return result  # Return cached result
        
        # Execute and cache
        result = self.execute(tool_name, **kwargs)
        if result.get("success", False):  # Only cache successful results
            self._cache[cache_key] = (result, time.time())
        
        return result
    
    def clear_cache(self, tool_name: Optional[str] = None):
        """Clear tool result cache."""
        if tool_name:
            # Clear specific tool's cache entries
            keys_to_delete = [k for k in self._cache if k.startswith(f"{tool_name}:")]
            for k in keys_to_delete:
                del self._cache[k]
        else:
            self._cache.clear()
    
    def list_tools(self) -> List[Dict]:
        """List all available tools."""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def get_tools_prompt(self) -> str:
        """
        Get a prompt describing all tools for the AI.
        Uses rich parameter info when available.
        """
        lines = ["AVAILABLE TOOLS", "=" * 60, ""]
        
        # Group by category
        by_category: Dict[str, List[Tool]] = {}
        for tool in self.tools.values():
            cat = getattr(tool, 'category', 'general')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(tool)
        
        for category in sorted(by_category.keys()):
            lines.append(f"\n{category.upper()} TOOLS:")
            lines.append("-" * 40)
            
            for tool in by_category[category]:
                lines.append(f"\n{tool.name}:")
                lines.append(f"  {tool.description}")
                
                # Use rich parameters if available
                if hasattr(tool, 'rich_parameters') and tool.rich_parameters:
                    lines.append("  Parameters:")
                    for p in tool.rich_parameters:
                        req = "*required*" if p.required else "optional"
                        default = f" = {p.default}" if p.default is not None else ""
                        lines.append(f"    {p.name} ({p.type}, {req}){default}")
                        lines.append(f"      {p.description}")
                        if p.enum:
                            lines.append(f"      Choices: {', '.join(str(e) for e in p.enum)}")
                elif tool.parameters:
                    lines.append("  Parameters:")
                    for param, desc in tool.parameters.items():
                        lines.append(f"    {param}: {desc}")
                
                # Show examples if available
                if hasattr(tool, 'examples') and tool.examples:
                    lines.append("  Examples:")
                    for ex in tool.examples[:2]:
                        lines.append(f"    - {ex}")
        
        return "\n".join(lines)


class ToolProfiler:
    """
    Profiler for tracking tool usage and performance.
    
    Tracks:
    - Tool call counts
    - Execution times (min, max, avg)
    - Success/failure rates
    - Most used tools
    
    Usage:
        profiler = ToolProfiler()
        registry = ToolRegistry()
        registry.set_profiler(profiler)
        
        # After using tools
        print(profiler.get_report())
    """
    
    def __init__(self):
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._enabled = True
    
    def record(self, tool_name: str, duration: float, success: bool):
        """Record a tool execution."""
        if not self._enabled:
            return
            
        if tool_name not in self._stats:
            self._stats[tool_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
            }
        
        stats = self._stats[tool_name]
        stats["calls"] += 1
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        stats["total_time"] += duration
        stats["min_time"] = min(stats["min_time"], duration)
        stats["max_time"] = max(stats["max_time"], duration)
    
    def get_stats(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get stats for a specific tool."""
        if tool_name not in self._stats:
            return None
        
        stats = self._stats[tool_name].copy()
        if stats["calls"] > 0:
            stats["avg_time"] = stats["total_time"] / stats["calls"]
            stats["success_rate"] = stats["successes"] / stats["calls"] * 100
        return stats
    
    def get_top_tools(self, n: int = 10) -> List[tuple]:
        """Get the most used tools."""
        return sorted(
            [(name, s["calls"]) for name, s in self._stats.items()],
            key=lambda x: -x[1]
        )[:n]
    
    def get_slowest_tools(self, n: int = 10) -> List[tuple]:
        """Get the slowest tools by average execution time."""
        tools_with_avg = []
        for name, stats in self._stats.items():
            if stats["calls"] > 0:
                avg = stats["total_time"] / stats["calls"]
                tools_with_avg.append((name, avg))
        return sorted(tools_with_avg, key=lambda x: -x[1])[:n]
    
    def get_report(self) -> str:
        """Get a human-readable report of tool usage."""
        lines = ["=== Tool Usage Report ===", ""]
        
        total_calls = sum(s["calls"] for s in self._stats.values())
        total_time = sum(s["total_time"] for s in self._stats.values())
        
        lines.append(f"Total tool calls: {total_calls}")
        lines.append(f"Total execution time: {total_time:.2f}s")
        lines.append("")
        
        lines.append("Top 5 Most Used Tools:")
        for name, calls in self.get_top_tools(5):
            pct = calls / total_calls * 100 if total_calls > 0 else 0
            lines.append(f"  {name}: {calls} calls ({pct:.1f}%)")
        
        lines.append("")
        lines.append("Top 5 Slowest Tools (avg):")
        for name, avg in self.get_slowest_tools(5):
            lines.append(f"  {name}: {avg*1000:.1f}ms")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all statistics."""
        self._stats.clear()
    
    def enable(self):
        """Enable profiling."""
        self._enabled = True
    
    def disable(self):
        """Disable profiling."""
        self._enabled = False


# Global registry instance
_registry = None
_profiler = None

def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry

def get_profiler() -> ToolProfiler:
    """Get the global tool profiler."""
    global _profiler
    if _profiler is None:
        _profiler = ToolProfiler()
    return _profiler

def execute_tool(name: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to execute a tool with profiling."""
    import time
    profiler = get_profiler()
    registry = get_registry()
    
    start = time.perf_counter()
    result = registry.execute(name, **kwargs)
    duration = time.perf_counter() - start
    
    success = result.get("success", False) if isinstance(result, dict) else True
    profiler.record(name, duration, success)
    
    return result


def batch_execute_tools(tool_calls: List[tuple], parallel: bool = False) -> List[Dict[str, Any]]:
    """
    Execute multiple tools in batch.
    
    Args:
        tool_calls: List of (tool_name, kwargs_dict) tuples
        parallel: If True, execute tools concurrently using threads
        
    Returns:
        List of results in the same order as tool_calls
        
    Example:
        results = batch_execute_tools([
            ("list_directory", {"path": "."}),
            ("get_system_info", {}),
            ("web_search", {"query": "python"}),
        ], parallel=True)
    """
    if parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = [None] * len(tool_calls)
        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 8)) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(execute_tool, name, **kwargs): i
                for i, (name, kwargs) in enumerate(tool_calls)
            }
            
            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {"success": False, "error": str(e)}
        
        return results
    else:
        # Sequential execution
        return [execute_tool(name, **kwargs) for name, kwargs in tool_calls]


def get_tool_summary() -> Dict[str, Any]:
    """
    Get a summary of the tool system status.
    
    Returns:
        Dict with tool counts, categories, and profiler stats
    """
    registry = get_registry()
    profiler = get_profiler()
    
    # Count by category
    categories = {}
    for tool in registry.tools.values():
        cat = getattr(tool, 'category', 'general')
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        "total_tools": len(registry.tools),
        "categories": categories,
        "profiler_stats": {
            "total_calls": sum(s["calls"] for s in profiler._stats.values()),
            "total_time": sum(s["total_time"] for s in profiler._stats.values()),
            "top_tools": profiler.get_top_tools(5),
        }
    }
