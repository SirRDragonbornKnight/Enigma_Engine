"""
File Tools - Read, write, move, delete files and directories.

Tools:
  - read_file: Read content from a file
  - write_file: Write content to a file
  - list_directory: List files in a directory
  - move_file: Move or rename a file
  - delete_file: Delete a file
  
Security:
  - Respects blocked_paths and blocked_patterns from config
  - AI cannot access files in blocked locations
"""

from __future__ import annotations

import logging
import platform
import shutil
from pathlib import Path
from typing import Any, Optional

from .tool_registry import RichParameter, Tool

logger = logging.getLogger(__name__)

# Import security check
try:
    from ..utils.security import is_path_blocked
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False
    def is_path_blocked(path):
        return False, None

# Constants
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB limit


def _check_path_allowed(path: str, resolve_first: bool = True) -> dict[str, Any]:
    """
    Check if path is allowed. Returns error dict if blocked.
    
    Args:
        path: The path to check
        resolve_first: If True, resolve the path before checking (prevents traversal attacks)
        
    Returns:
        Error dict if path is blocked, None if allowed
    """
    if resolve_first:
        try:
            # Resolve path to prevent traversal attacks (e.g., "../../etc/passwd")
            resolved_path = str(Path(path).expanduser().resolve())
        except Exception as e:
            # If path can't be resolved, use original
            logger.debug(f"Could not resolve path '{path}': {e}")
            resolved_path = path
    else:
        resolved_path = path
    
    if HAS_SECURITY:
        blocked, reason = is_path_blocked(resolved_path)
        if blocked:
            return {"success": False, "error": f"Access denied: {reason}"}
    return None


class ReadFileTool(Tool):
    """Read content from a file."""
    
    name = "read_file"
    description = "Read the contents of a text file. Returns the file content as a string."
    parameters = {
        "path": "The absolute or relative path to the file",
        "max_lines": "Maximum lines to read (default: all)",
        "encoding": "File encoding (default: utf-8)",
    }
    category = "file"
    rich_parameters = [
        RichParameter(
            name="path",
            type="string",
            description="Path to the file to read",
            required=True,
        ),
        RichParameter(
            name="max_lines",
            type="integer",
            description="Maximum number of lines to read (reads all if not set)",
            required=False,
            min_value=1,
        ),
        RichParameter(
            name="encoding",
            type="string",
            description="File encoding",
            required=False,
            default="utf-8",
        ),
    ]
    examples = [
        "read_file(path='config.json') - Read entire config file",
        "read_file(path='log.txt', max_lines=100) - Read first 100 lines",
    ]
    
    def execute(self, path: str, max_lines: Optional[int] = None, encoding: str = "utf-8", **kwargs) -> dict[str, Any]:
        """
        Execute the read file operation.
        
        Args:
            path: Path to the file to read
            max_lines: Optional maximum number of lines to read
            encoding: File encoding (default: utf-8)
            
        Returns:
            Dictionary with success status and file content or error message
        """
        try:
            if not path:
                return {"success": False, "error": "Path cannot be empty"}
            
            # Security check
            blocked = _check_path_allowed(path)
            if blocked:
                return blocked
            
            path = Path(path).expanduser().resolve()
            
            if not path.exists():
                return {"success": False, "error": f"File not found: {path}"}
            
            if not path.is_file():
                return {"success": False, "error": f"Not a file: {path}"}
            
            # Check file size to prevent reading huge files
            file_size = path.stat().st_size
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                return {"success": False, "error": f"File too large: {file_size} bytes (limit: 100MB)"}
            
            with open(path, encoding=encoding) as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    content = "".join(lines)
                else:
                    content = f.read()
            
            return {
                "success": True,
                "path": str(path),
                "size_bytes": file_size,
                "num_lines": content.count('\n') + 1,
                "content": content
            }
            
        except UnicodeDecodeError as e:
            return {"success": False, "error": f"Cannot decode file with {encoding} encoding: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class WriteFileTool(Tool):
    """Write content to a file."""
    
    name = "write_file"
    description = "Write text content to a file. Creates the file if it doesn't exist."
    parameters = {
        "path": "The path to write to",
        "content": "The text content to write",
        "mode": "Write mode: 'overwrite' or 'append' (default: overwrite)",
    }
    category = "file"
    rich_parameters = [
        RichParameter(
            name="path",
            type="string",
            description="Path to write to",
            required=True,
        ),
        RichParameter(
            name="content",
            type="string",
            description="Text content to write",
            required=True,
        ),
        RichParameter(
            name="mode",
            type="string",
            description="Write mode",
            required=False,
            default="overwrite",
            enum=["overwrite", "append"]
        ),
    ]
    examples = [
        "write_file(path='output.txt', content='Hello World') - Write to file",
        "write_file(path='log.txt', content='New entry', mode='append') - Append to file",
    ]
    
    def execute(self, path: str, content: str, mode: str = "overwrite", **kwargs) -> dict[str, Any]:
        """
        Execute the write file operation.
        
        Args:
            path: Path to write to
            content: Content to write
            mode: Write mode ('overwrite' or 'append')
            
        Returns:
            Dictionary with success status and file info or error message
        """
        try:
            if not path:
                return {"success": False, "error": "Path cannot be empty"}
            
            # Security check
            blocked = _check_path_allowed(path)
            if blocked:
                return blocked
            
            if mode not in ("overwrite", "append"):
                return {"success": False, "error": f"Invalid mode: {mode}. Use 'overwrite' or 'append'"}
            
            path = Path(path).expanduser().resolve()
            
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            file_mode = "w" if mode == "overwrite" else "a"
            
            with open(path, file_mode, encoding="utf-8") as f:
                f.write(content)
            
            return {
                "success": True,
                "path": str(path),
                "bytes_written": len(content.encode('utf-8')),
                "mode": mode
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class ListDirectoryTool(Tool):
    """List files in a directory."""
    
    name = "list_directory"
    description = "List all files and folders in a directory."
    parameters = {
        "path": "The directory path to list",
        "recursive": "Whether to list subdirectories recursively (default: false)",
        "pattern": "Filter files by pattern like '*.txt' (default: all files)",
    }
    category = "file"
    rich_parameters = [
        RichParameter(
            name="path",
            type="string",
            description="Directory path to list",
            required=True,
        ),
        RichParameter(
            name="recursive",
            type="boolean",
            description="List subdirectories recursively",
            required=False,
            default=False,
        ),
        RichParameter(
            name="pattern",
            type="string",
            description="Glob pattern to filter files (e.g., '*.txt')",
            required=False,
            default="*",
        ),
    ]
    examples = [
        "list_directory(path='.') - List current directory",
        "list_directory(path='src', recursive=true) - List all files recursively",
        "list_directory(path='docs', pattern='*.md') - List only markdown files",
    ]
    
    def execute(self, path: str, recursive: bool = False, pattern: str = "*", **kwargs) -> dict[str, Any]:
        try:
            # Security check - prevent access to blocked paths
            blocked = _check_path_allowed(path)
            if blocked:
                return blocked
            
            path = Path(path).expanduser().resolve()
            
            if not path.exists():
                return {"success": False, "error": f"Directory not found: {path}"}
            
            if not path.is_dir():
                return {"success": False, "error": f"Not a directory: {path}"}
            
            items = []
            
            if recursive:
                for item in path.rglob(pattern):
                    rel_path = item.relative_to(path)
                    items.append({
                        "name": str(rel_path),
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else 0,
                    })
            else:
                for item in path.glob(pattern):
                    items.append({
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else 0,
                    })
            
            return {
                "success": True,
                "path": str(path),
                "num_items": len(items),
                "items": items
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class MoveFileTool(Tool):
    """Move or rename a file."""
    
    name = "move_file"
    description = "Move a file to a new location or rename it."
    parameters = {
        "source": "The current path of the file",
        "destination": "The new path for the file",
    }
    category = "file"
    rich_parameters = [
        RichParameter(
            name="source",
            type="string",
            description="Current path of the file",
            required=True,
        ),
        RichParameter(
            name="destination",
            type="string",
            description="New path for the file",
            required=True,
        ),
    ]
    examples = [
        "move_file(source='old.txt', destination='new.txt') - Rename file",
        "move_file(source='file.txt', destination='archive/file.txt') - Move to folder",
    ]
    
    def execute(self, source: str, destination: str, **kwargs) -> dict[str, Any]:
        try:
            source = Path(source).expanduser().resolve()
            destination = Path(destination).expanduser().resolve()
            
            # Security checks for both source and destination
            blocked = _check_path_allowed(str(source))
            if blocked:
                return blocked
            blocked = _check_path_allowed(str(destination))
            if blocked:
                return blocked
            
            if not source.exists():
                return {"success": False, "error": f"Source not found: {source}"}
            
            # Create destination directory if needed
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(source), str(destination))
            
            return {
                "success": True,
                "source": str(source),
                "destination": str(destination),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class DeleteFileTool(Tool):
    """Delete a file (with safety checks)."""
    
    name = "delete_file"
    description = "Delete a file. Use with caution!"
    parameters = {
        "path": "The path of the file to delete",
        "confirm": "Must be 'yes' to actually delete (safety check)",
    }
    category = "file"
    rich_parameters = [
        RichParameter(
            name="path",
            type="string",
            description="Path of the file to delete",
            required=True,
        ),
        RichParameter(
            name="confirm",
            type="string",
            description="Safety confirmation - must be 'yes' to delete",
            required=True,
            enum=["yes", "no"]
        ),
    ]
    examples = [
        "delete_file(path='temp.txt', confirm='yes') - Delete file",
        "delete_file(path='folder/', confirm='yes') - Delete folder and contents",
    ]
    
    # Protected paths that cannot be deleted - OS-specific
    PROTECTED_PATHS_UNIX = [
        "/", "/home", "/usr", "/bin", "/etc", "/var", "/root",
        "/boot", "/lib", "/lib64", "/sbin", "/opt", "/sys", "/proc"
    ]
    
    PROTECTED_PATHS_WINDOWS = [
        "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
        "C:\\Users\\Public", "C:\\ProgramData", "C:\\",
    ]
    
    @property
    def protected_paths(self):
        """Get protected paths for current OS."""
        if platform.system() == "Windows":
            return self.PROTECTED_PATHS_WINDOWS
        return self.PROTECTED_PATHS_UNIX
    
    def _is_protected(self, path: Path) -> bool:
        """Check if path is protected from deletion."""
        path_str = str(path).lower()
        for protected in self.protected_paths:
            protected_lower = protected.lower()
            # Check exact match or if path is inside protected directory
            if path_str == protected_lower:
                return True
            # Check if it's a direct child of a protected path
            try:
                if path.resolve().parent == Path(protected).resolve():
                    return True
            except (OSError, ValueError):
                pass  # Intentionally silent
        return False
    
    def execute(self, path: str, confirm: str = "no", **kwargs) -> dict[str, Any]:
        try:
            if confirm.lower() != "yes":
                return {
                    "success": False, 
                    "error": "Deletion not confirmed. Set confirm='yes' to delete."
                }
            
            path = Path(path).expanduser().resolve()
            
            # Safety check - protected paths
            if self._is_protected(path):
                logger.warning(f"Attempted deletion of protected path: {path}")
                return {"success": False, "error": f"Cannot delete protected path: {path}"}
            
            # Check security module
            blocked = _check_path_allowed(str(path))
            if blocked:
                return blocked
            
            if not path.exists():
                return {"success": False, "error": f"File not found: {path}"}
            
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            
            return {
                "success": True,
                "deleted": str(path),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Test
    import json
    logging.basicConfig(level=logging.DEBUG)
    
    list_tool = ListDirectoryTool()
    result = list_tool.execute(".")
    logger.info(json.dumps(result, indent=2))
