"""
LangChain Tools Integration
============================

Expose Enigma AI Engine tools as LangChain Tools for use in LangChain workflows,
agents, and chains.

This allows Enigma tools to be used with:
- LangChain agents (ReAct, OpenAI Functions, etc.)
- LangChain chains
- LangGraph workflows
- Any LangChain-compatible framework

Usage:
    from enigma_engine.integrations.langchain_tools import (
        get_langchain_tools,
        EnigmaToolkit,
        enigma_tool_to_langchain,
    )
    
    # Get all tools as LangChain tools
    tools = get_langchain_tools()
    
    # Use with an agent
    from langchain.agents import AgentExecutor, create_react_agent
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    
    # Or use the toolkit
    toolkit = EnigmaToolkit()
    tools = toolkit.get_tools()

Requirements:
    pip install langchain langchain-core
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Type

logger = logging.getLogger(__name__)

# Check if langchain is available
_LANGCHAIN_AVAILABLE = False
try:
    from langchain_core.tools import BaseTool, StructuredTool
    from langchain_core.callbacks import CallbackManagerForToolRun
    from pydantic import BaseModel, Field, create_model
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        # Fallback to older langchain import
        from langchain.tools import BaseTool, StructuredTool
        from langchain.callbacks.manager import CallbackManagerForToolRun
        from pydantic import BaseModel, Field, create_model
        _LANGCHAIN_AVAILABLE = True
    except ImportError:
        logger.debug("LangChain not installed - integration disabled")
        BaseTool = object
        StructuredTool = object
        BaseModel = object


def is_langchain_available() -> bool:
    """Check if LangChain is installed and available."""
    return _LANGCHAIN_AVAILABLE


def _enigma_type_to_python(type_str: str) -> Type:
    """Convert Enigma type string to Python type."""
    type_map = {
        "string": str,
        "str": str,
        "int": int,
        "integer": int,
        "float": float,
        "number": float,
        "bool": bool,
        "boolean": bool,
        "list": list,
        "array": list,
        "dict": dict,
        "object": dict,
        "any": Any,
    }
    return type_map.get(type_str.lower(), str)


def _create_pydantic_model(tool_name: str, parameters: list) -> Type[BaseModel]:
    """
    Create a Pydantic model from Enigma tool parameters.
    
    Args:
        tool_name: Name of the tool (used for model name)
        parameters: List of ToolParameter objects
        
    Returns:
        Pydantic model class for the tool's arguments
    """
    if not _LANGCHAIN_AVAILABLE:
        return None
    
    field_definitions = {}
    
    for param in parameters:
        python_type = _enigma_type_to_python(param.type)
        
        if param.required:
            if param.default is not None:
                field_definitions[param.name] = (
                    python_type,
                    Field(default=param.default, description=param.description)
                )
            else:
                field_definitions[param.name] = (
                    python_type,
                    Field(..., description=param.description)
                )
        else:
            default = param.default if param.default is not None else None
            field_definitions[param.name] = (
                Optional[python_type],
                Field(default=default, description=param.description)
            )
    
    # Create model with sanitized name
    model_name = f"{tool_name.replace('_', ' ').title().replace(' ', '')}Args"
    return create_model(model_name, **field_definitions)


class EnigmaTool(BaseTool if _LANGCHAIN_AVAILABLE else object):
    """
    LangChain Tool wrapper for an Enigma tool.
    
    Wraps an Enigma ToolDefinition so it can be used in LangChain agents.
    """
    
    name: str = ""
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None
    
    # Enigma-specific
    _tool_name: str = ""
    _executor: Any = None
    _category: str = ""
    
    def __init__(
        self,
        tool_name: str,
        description: str,
        args_schema: Optional[Type[BaseModel]] = None,
        executor: Any = None,
        category: str = "",
        **kwargs
    ):
        """
        Initialize an Enigma tool for LangChain.
        
        Args:
            tool_name: Enigma tool name
            description: Tool description
            args_schema: Pydantic model for arguments
            executor: ToolExecutor instance (or None to create one)
            category: Tool category
        """
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for this feature. "
                "Install with: pip install langchain langchain-core"
            )
        
        super().__init__(
            name=tool_name,
            description=description,
            args_schema=args_schema,
            **kwargs
        )
        self._tool_name = tool_name
        self._executor = executor
        self._category = category
    
    def _get_executor(self):
        """Get or create tool executor."""
        if self._executor is None:
            from enigma_engine.tools.tool_executor import ToolExecutor
            self._executor = ToolExecutor()
        return self._executor
    
    def _run(
        self,
        *args,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """
        Execute the tool synchronously.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Tool result as string
        """
        executor = self._get_executor()
        
        # Execute the tool
        result = executor.execute(self._tool_name, kwargs)
        
        if result.get("success"):
            return str(result.get("result", "Tool executed successfully"))
        else:
            error = result.get("error", "Unknown error")
            return f"Error: {error}"
    
    async def _arun(
        self,
        *args,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """
        Execute the tool asynchronously.
        
        Currently just wraps the sync version.
        """
        # For now, just run sync - could be made truly async later
        return self._run(*args, run_manager=run_manager, **kwargs)


def enigma_tool_to_langchain(
    tool_def,
    executor: Any = None
) -> Optional[EnigmaTool]:
    """
    Convert an Enigma ToolDefinition to a LangChain Tool.
    
    Args:
        tool_def: Enigma ToolDefinition object
        executor: Optional ToolExecutor instance to share
        
    Returns:
        LangChain Tool or None if conversion fails
    """
    if not _LANGCHAIN_AVAILABLE:
        logger.warning("LangChain not available - cannot convert tool")
        return None
    
    try:
        # Create Pydantic model for arguments
        args_schema = _create_pydantic_model(tool_def.name, tool_def.parameters)
        
        # Create the LangChain tool
        return EnigmaTool(
            tool_name=tool_def.name,
            description=tool_def.description,
            args_schema=args_schema,
            executor=executor,
            category=tool_def.category,
        )
    
    except Exception as e:
        logger.warning(f"Failed to convert tool {tool_def.name}: {e}")
        return None


def get_langchain_tools(
    categories: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    include_deprecated: bool = False
) -> List[EnigmaTool]:
    """
    Get all Enigma tools as LangChain tools.
    
    Args:
        categories: Only include tools from these categories (None = all)
        exclude: Tool names to exclude
        include_deprecated: Include deprecated tools
        
    Returns:
        List of LangChain Tool objects
    """
    if not _LANGCHAIN_AVAILABLE:
        logger.warning(
            "LangChain not installed. Install with: pip install langchain langchain-core"
        )
        return []
    
    from enigma_engine.tools.tool_definitions import get_all_tools
    from enigma_engine.tools.tool_executor import ToolExecutor
    
    # Create shared executor
    executor = ToolExecutor()
    
    tools = []
    exclude_set = set(exclude or [])
    
    for tool_def in get_all_tools():
        # Filter by category
        if categories and tool_def.category not in categories:
            continue
        
        # Exclude specific tools
        if tool_def.name in exclude_set:
            continue
        
        # Skip deprecated unless requested
        if tool_def.deprecated and not include_deprecated:
            continue
        
        # Convert to LangChain tool
        lc_tool = enigma_tool_to_langchain(tool_def, executor)
        if lc_tool:
            tools.append(lc_tool)
    
    logger.info(f"Created {len(tools)} LangChain tools from Enigma")
    return tools


def get_langchain_tool(tool_name: str) -> Optional[EnigmaTool]:
    """
    Get a single Enigma tool as a LangChain tool.
    
    Args:
        tool_name: Name of the tool to get
        
    Returns:
        LangChain Tool or None if not found
    """
    if not _LANGCHAIN_AVAILABLE:
        return None
    
    from enigma_engine.tools.tool_definitions import get_tool_definition
    
    tool_def = get_tool_definition(tool_name)
    if tool_def:
        return enigma_tool_to_langchain(tool_def)
    return None


class EnigmaToolkit:
    """
    LangChain Toolkit containing Enigma AI Engine tools.
    
    Provides a convenient way to get tools for LangChain agents.
    
    Usage:
        toolkit = EnigmaToolkit(categories=["generation", "perception"])
        tools = toolkit.get_tools()
        
        # Use with agent
        agent = create_react_agent(llm, tools, prompt)
    """
    
    def __init__(
        self,
        categories: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        include_deprecated: bool = False
    ):
        """
        Initialize the toolkit.
        
        Args:
            categories: Only include tools from these categories
            exclude: Tool names to exclude
            include_deprecated: Include deprecated tools
        """
        self.categories = categories
        self.exclude = exclude or []
        self.include_deprecated = include_deprecated
        self._tools: Optional[List[EnigmaTool]] = None
    
    def get_tools(self) -> List[EnigmaTool]:
        """
        Get all tools in the toolkit.
        
        Returns:
            List of LangChain Tool objects
        """
        if self._tools is None:
            self._tools = get_langchain_tools(
                categories=self.categories,
                exclude=self.exclude,
                include_deprecated=self.include_deprecated
            )
        return self._tools
    
    def get_tool_names(self) -> List[str]:
        """Get names of all tools in the toolkit."""
        return [t.name for t in self.get_tools()]
    
    def add_exclusion(self, tool_name: str):
        """Add a tool to the exclusion list (rebuilds on next get_tools)."""
        self.exclude.append(tool_name)
        self._tools = None
    
    def filter_categories(self, categories: List[str]):
        """Filter to specific categories (rebuilds on next get_tools)."""
        self.categories = categories
        self._tools = None


# Convenience functions for quick access

def get_generation_tools() -> List[EnigmaTool]:
    """Get only generation tools (image, code, audio, video, 3D)."""
    return get_langchain_tools(categories=["generation"])


def get_perception_tools() -> List[EnigmaTool]:
    """Get only perception tools (vision, screen capture)."""
    return get_langchain_tools(categories=["perception"])


def get_control_tools() -> List[EnigmaTool]:
    """Get only control tools (system, robot, game)."""
    return get_langchain_tools(categories=["control"])


def get_web_tools() -> List[EnigmaTool]:
    """Get only web-related tools (search, fetch, browse)."""
    return get_langchain_tools(categories=["web"])


def get_file_tools() -> List[EnigmaTool]:
    """Get only file-related tools (read, write, list)."""
    return get_langchain_tools(categories=["file"])


__all__ = [
    "is_langchain_available",
    "enigma_tool_to_langchain",
    "get_langchain_tools",
    "get_langchain_tool",
    "EnigmaTool",
    "EnigmaToolkit",
    "get_generation_tools",
    "get_perception_tools",
    "get_control_tools",
    "get_web_tools",
    "get_file_tools",
]
