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
    from forge_ai.tools import ToolRegistry
    
    tools = ToolRegistry()
    
    # List available tools
    tools.list_tools()
    
    # Execute a tool
    result = tools.execute("web_search", query="python tutorials")
    result = tools.execute("read_file", path="/home/user/document.txt")
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


class Tool(ABC):
    """Base class for all tools."""
    
    name: str = "base_tool"
    description: str = "Base tool - override this"
    parameters: Dict[str, str] = {}  # param_name: description
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool.
        
        Returns:
            Dict with 'success', 'result', and optionally 'error'
        """
    
    def to_dict(self) -> Dict:
        """Export tool info for AI consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """
    Registry of all available tools.
    
    Respects tool_manager settings - disabled tools won't be registered.
    """
    
    def __init__(self, respect_manager: bool = True):
        self.tools: Dict[str, Tool] = {}
        self._respect_manager = respect_manager
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
        from .web_tools import WebSearchTool, FetchWebpageTool
        from .file_tools import ReadFileTool, WriteFileTool, ListDirectoryTool, MoveFileTool, DeleteFileTool
        from .document_tools import ReadDocumentTool, ExtractTextTool
        from .system_tools import RunCommandTool, ScreenshotTool, GetSystemInfoTool
        from .vision import ScreenVisionTool, FindOnScreenTool
        from .robot_tools import RobotMoveTool, RobotGripperTool, RobotStatusTool, RobotHomeTool
        from .interactive_tools import (CreateChecklistTool, ListChecklistsTool, AddTaskTool, 
                                       ListTasksTool, CompleteTaskTool, SetReminderTool, 
                                       ListRemindersTool, CheckRemindersTool)
        
        # New tool imports
        from .automation_tools import (ScheduleTaskTool, ListSchedulesTool, RemoveScheduleTool,
                                       ClipboardReadTool, ClipboardWriteTool, ClipboardHistoryTool,
                                       RecordMacroTool, PlayMacroTool)
        from .knowledge_tools import WikipediaSearchTool, ArxivSearchTool, PDFExtractTool
        from .communication_tools import TranslateTextTool, OCRImageTool
        from .media_tools import (MusicGenerateTool, RemoveBackgroundTool, UpscaleImageTool,
                                  StyleTransferTool, ConvertAudioTool, ExtractAudioTool,
                                  AudioVisualizeTool)
        from .productivity_tools import (SystemMonitorTool, ProcessListTool, ProcessKillTool,
                                         SSHExecuteTool, DockerListTool, DockerControlTool,
                                         GitStatusTool, GitCommitTool, GitPushTool, GitPullTool)
        from .iot_tools import (HomeAssistantSetupTool, HomeAssistantControlTool,
                                HomeAssistantStatusTool, GPIOReadTool, GPIOWriteTool,
                                GPIOPWMTool, MQTTPublishTool, MQTTSubscribeTool,
                                CameraCaptureTool, CameraListTool, CameraStreamTool)
        from .browser_tools import (BrowserMediaPauseTool, BrowserMediaMuteTool,
                                    BrowserMediaSkipTool, BrowserMediaStopTool,
                                    BrowserMediaVolumeTool, BrowserMediaInfoTool,
                                    BrowserTabListTool, BrowserFocusTool)
        from .data_tools import (CSVAnalyzeTool, CSVQueryTool, PlotChartTool, JSONQueryTool,
                                 SQLQueryTool, SQLExecuteTool, DataConvertTool)
        # Avatar tools - AI can control and customize the desktop avatar
        from .avatar_tools import (AvatarControlTool, AvatarCustomizeTool, 
                                   AvatarGestureTool, AvatarEmotionTool)
        # Self-modification tools - AI can customize itself
        from .self_tools import (SetPersonalityTool, SetAvatarPreferenceTool,
                                 SetVoicePreferenceTool, SetCompanionBehaviorTool,
                                 SetPreferenceTool, GetSelfConfigTool,
                                 RememberFactTool, RecallFactsTool)
        # Gaming tools removed - AI can roleplay, tell stories, and play games natively
        # Keep only DnD dice roller since it needs true randomness
        from .gaming_tools import DnDRollTool
        
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
            # Self-Modification - AI customizes itself
            SetPersonalityTool(),
            SetAvatarPreferenceTool(),
            SetVoicePreferenceTool(),
            SetCompanionBehaviorTool(),
            SetPreferenceTool(),
            GetSelfConfigTool(),
            RememberFactTool(),
            RecallFactsTool(),
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
    
    def list_tools(self) -> List[Dict]:
        """List all available tools."""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def get_tools_prompt(self) -> str:
        """
        Get a prompt describing all tools for the AI.
        Useful for training or prompting.
        """
        lines = ["Available tools:\n"]
        for tool in self.tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
            if tool.parameters:
                for param, desc in tool.parameters.items():
                    lines.append(f"    - {param}: {desc}")
        return "\n".join(lines)


# Global registry instance
_registry = None

def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def execute_tool(name: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to execute a tool."""
    return get_registry().execute(name, **kwargs)


# Convenience alias for direct import
tool_registry = get_registry()
