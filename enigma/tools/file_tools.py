"""
File Tools - Read, write, move, delete files and directories.

Tools:
  - read_file: Read content from a file
  - write_file: Write content to a file
  - list_directory: List files in a directory
  - move_file: Move or rename a file
  - delete_file: Delete a file
"""

import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from .tool_registry import Tool


class ReadFileTool(Tool):
    """Read content from a file."""
    
    name = "read_file"
    description = "Read the contents of a text file. Returns the file content as a string."
    parameters = {
        "path": "The absolute or relative path to the file",
        "max_lines": "Maximum lines to read (default: all)",
        "encoding": "File encoding (default: utf-8)",
    }
    
    def execute(self, path: str, max_lines: Optional[int] = None, encoding: str = "utf-8", **kwargs) -> Dict[str, Any]:
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
            
            path = Path(path).expanduser().resolve()
            
            if not path.exists():
                return {"success": False, "error": f"File not found: {path}"}
            
            if not path.is_file():
                return {"success": False, "error": f"Not a file: {path}"}
            
            # Check file size to prevent reading huge files
            file_size = path.stat().st_size
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                return {"success": False, "error": f"File too large: {file_size} bytes (limit: 100MB)"}
            
            with open(path, "r", encoding=encoding) as f:
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
    
    def execute(self, path: str, content: str, mode: str = "overwrite", **kwargs) -> Dict[str, Any]:
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
    
    def execute(self, path: str, recursive: bool = False, pattern: str = "*", **kwargs) -> Dict[str, Any]:
        try:
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
    
    def execute(self, source: str, destination: str, **kwargs) -> Dict[str, Any]:
        try:
            source = Path(source).expanduser().resolve()
            destination = Path(destination).expanduser().resolve()
            
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
    
    # Protected paths that cannot be deleted
    PROTECTED_PATHS = [
        "/", "/home", "/usr", "/bin", "/etc", "/var", "/root",
        "/boot", "/lib", "/lib64", "/sbin", "/opt", "/sys", "/proc"
    ]
    
    def execute(self, path: str, confirm: str = "no", **kwargs) -> Dict[str, Any]:
        try:
            if confirm.lower() != "yes":
                return {
                    "success": False, 
                    "error": "Deletion not confirmed. Set confirm='yes' to delete."
                }
            
            path = Path(path).expanduser().resolve()
            
            # Safety check
            for protected in self.PROTECTED_PATHS:
                if str(path) == protected or str(path).startswith(protected + "/") and path.parent == Path(protected):
                    return {"success": False, "error": f"Cannot delete protected path: {path}"}
            
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
    
    list_tool = ListDirectoryTool()
    result = list_tool.execute(".")
    print(json.dumps(result, indent=2))
