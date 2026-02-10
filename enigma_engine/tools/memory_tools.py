"""
Memory Tools - AI can search and manage its conversation memory.

Provides tools for:
- Searching conversations by text
- Getting memory statistics  
- Exporting/importing conversation backups
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .tool_registry import RichParameter, Tool

if TYPE_CHECKING:
    from ..memory.manager import ConversationManager

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_memory_manager() -> "ConversationManager":
    """Get cached ConversationManager instance."""
    from ..memory.manager import ConversationManager
    return ConversationManager()


class SearchMemoryTool(Tool):
    """
    Search through conversation history.
    
    Allows the AI to recall past conversations by searching
    for keywords or phrases across all stored messages.
    """
    
    name = "search_memory"
    description = "Search my conversation history for past discussions about a topic."
    category = "memory"
    
    rich_parameters = [
        RichParameter(
            name="query",
            type="string",
            description="Text to search for in conversations",
            required=True,
        ),
        RichParameter(
            name="limit",
            type="integer",
            description="Maximum results to return",
            required=False,
            default=10,
            min_value=1,
            max_value=100,
        ),
    ]
    
    examples = [
        "search_memory query='Python programming'",
        "search_memory query='project deadline' limit=5",
    ]
    
    def execute(self, query: str = "", limit: int = 10, **kwargs) -> dict[str, Any]:
        """Search conversations for matching text."""
        if not query:
            return {"success": False, "error": "Query cannot be empty"}
        
        try:
            manager = _get_memory_manager()
            results = manager.search_conversations(query, limit=limit)
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
            }
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return {"success": False, "error": str(e)}


class MemoryStatsTool(Tool):
    """
    Get statistics about conversation memory.
    
    Returns counts and sizes of stored conversations
    to help understand memory usage.
    """
    
    name = "memory_stats"
    description = "Get statistics about my conversation memory - conversation count, message count, storage size."
    category = "memory"
    
    rich_parameters = []
    
    examples = [
        "memory_stats",
    ]
    
    def execute(self, **kwargs) -> dict[str, Any]:
        """Return memory statistics."""
        try:
            manager = _get_memory_manager()
            stats = manager.get_stats()
            return {"success": True, **stats}
        except Exception as e:
            logger.error(f"Memory stats failed: {e}")
            return {"success": False, "error": str(e)}


class ExportMemoryTool(Tool):
    """
    Export all conversations to a backup file.
    
    Creates a JSON or JSONL file containing all stored
    conversations for backup or transfer purposes.
    """
    
    name = "export_memory"
    description = "Export all my conversations to a backup file."
    category = "memory"
    
    rich_parameters = [
        RichParameter(
            name="output_path",
            type="string",
            description="Path to save the export file",
            required=True,
        ),
        RichParameter(
            name="format",
            type="string",
            description="Export format",
            required=False,
            default="json",
            enum=["json", "jsonl"],
        ),
    ]
    
    examples = [
        "export_memory output_path='./backup.json'",
        "export_memory output_path='./backup.jsonl' format='jsonl'",
    ]
    
    def execute(self, output_path: str = "", format: str = "json", **kwargs) -> dict[str, Any]:
        """Export conversations to file."""
        if not output_path:
            return {"success": False, "error": "Output path required"}
        
        try:
            manager = _get_memory_manager()
            result = manager.export_all(Path(output_path), format=format)
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"Memory export failed: {e}")
            return {"success": False, "error": str(e)}


class ImportMemoryTool(Tool):
    """
    Import conversations from a backup file.
    
    Restores conversations from a previously exported
    JSON or JSONL file.
    """
    
    name = "import_memory"
    description = "Import conversations from a backup file."
    category = "memory"
    
    rich_parameters = [
        RichParameter(
            name="input_path",
            type="string",
            description="Path to the backup file",
            required=True,
        ),
        RichParameter(
            name="overwrite",
            type="boolean",
            description="Overwrite existing conversations with same name",
            required=False,
            default=False,
        ),
    ]
    
    examples = [
        "import_memory input_path='./backup.json'",
        "import_memory input_path='./backup.json' overwrite=true",
    ]
    
    def execute(self, input_path: str = "", overwrite: bool = False, **kwargs) -> dict[str, Any]:
        """Import conversations from file."""
        if not input_path:
            return {"success": False, "error": "Input path required"}
        
        path = Path(input_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {input_path}"}
        
        try:
            manager = _get_memory_manager()
            result = manager.import_all(path, overwrite=overwrite)
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"Memory import failed: {e}")
            return {"success": False, "error": str(e)}


# All memory tools
MEMORY_TOOLS = [
    SearchMemoryTool(),
    MemoryStatsTool(),
    ExportMemoryTool(),
    ImportMemoryTool(),
]


def get_memory_tools() -> list[Tool]:
    """Get all memory tools."""
    return MEMORY_TOOLS
