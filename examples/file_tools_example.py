"""
File Tools Example for ForgeAI

This example shows how to use ForgeAI's file operations.
The AI can read, write, list, move, and delete files safely.

CAPABILITIES:
- Read files (text, binary, partial)
- Write files (create, append)
- List directories
- Move/rename files
- Delete files
- File search

SECURITY:
- Blocked paths prevent access to sensitive locations
- AI cannot access system files, ssh keys, etc.

USAGE:
    python examples/file_tools_example.py
    
Or import in your own code:
    from examples.file_tools_example import read_file, write_file, list_dir
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


# =============================================================================
# SECURITY (Blocked Paths)
# =============================================================================

# Paths the AI should NEVER access
BLOCKED_PATHS = [
    # System
    "/etc/passwd",
    "/etc/shadow",
    "/etc/ssh",
    "/root",
    
    # User sensitive
    "~/.ssh",
    "~/.gnupg",
    "~/.config/",
    "~/.local/share/keyrings",
    
    # Credentials
    "*.pem",
    "*.key",
    "*_rsa",
    "*_ed25519",
    "*.env",
    ".env",
    
    # Browser data
    "~/.mozilla",
    "~/.config/google-chrome",
    "~/.config/chromium",
]

BLOCKED_PATTERNS = [
    "*password*",
    "*secret*",
    "*credential*",
    "*token*",
    "*.key",
    "*.pem",
]


def is_path_blocked(path: str) -> tuple:
    """
    Check if path is blocked.
    
    Returns:
        (is_blocked: bool, reason: str)
    """
    path = str(Path(path).expanduser().resolve())
    path_lower = path.lower()
    
    # Check blocked paths
    for blocked in BLOCKED_PATHS:
        blocked_expanded = str(Path(blocked).expanduser())
        if blocked.startswith("*"):
            # Pattern matching
            pattern = blocked[1:].lower()
            if path_lower.endswith(pattern):
                return True, f"Matches blocked pattern: {blocked}"
        elif path.startswith(blocked_expanded) or path == blocked_expanded:
            return True, f"Path is blocked: {blocked}"
    
    # Check blocked patterns
    filename = Path(path).name.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern.startswith("*") and pattern.endswith("*"):
            # Contains
            if pattern[1:-1] in filename:
                return True, f"Filename matches blocked pattern: {pattern}"
        elif pattern.startswith("*"):
            # Ends with
            if filename.endswith(pattern[1:]):
                return True, f"Filename matches blocked pattern: {pattern}"
        elif pattern.endswith("*"):
            # Starts with
            if filename.startswith(pattern[:-1]):
                return True, f"Filename matches blocked pattern: {pattern}"
    
    return False, "Path is allowed"


# =============================================================================
# FILE READING
# =============================================================================

class FileReader:
    """
    Safe file reading with security checks.
    """
    
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit
    
    def read(self, path: str, max_lines: int = None, 
             encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Read a text file.
        
        Args:
            path: Path to file
            max_lines: Maximum lines to read (None for all)
            encoding: File encoding
        
        Returns:
            Dict with content and metadata
        """
        # Security check
        blocked, reason = is_path_blocked(path)
        if blocked:
            return {"success": False, "error": f"Access denied: {reason}"}
        
        path = Path(path).expanduser().resolve()
        
        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        if not path.is_file():
            return {"success": False, "error": f"Not a file: {path}"}
        
        # Size check
        size = path.stat().st_size
        if size > self.MAX_FILE_SIZE:
            return {"success": False, "error": f"File too large: {size} bytes"}
        
        try:
            with open(path, 'r', encoding=encoding, errors='ignore') as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    content = ''.join(lines)
                else:
                    content = f.read()
            
            return {
                "success": True,
                "path": str(path),
                "content": content,
                "size": size,
                "lines": content.count('\n') + 1,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def read_binary(self, path: str, max_bytes: int = None) -> Dict[str, Any]:
        """Read binary file."""
        blocked, reason = is_path_blocked(path)
        if blocked:
            return {"success": False, "error": f"Access denied: {reason}"}
        
        path = Path(path).expanduser().resolve()
        
        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        try:
            with open(path, 'rb') as f:
                if max_bytes:
                    content = f.read(max_bytes)
                else:
                    content = f.read()
            
            return {
                "success": True,
                "path": str(path),
                "content": content,
                "size": len(content),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def read_lines(self, path: str, start: int = 1, end: int = None) -> Dict[str, Any]:
        """Read specific lines from file."""
        blocked, reason = is_path_blocked(path)
        if blocked:
            return {"success": False, "error": f"Access denied: {reason}"}
        
        path = Path(path).expanduser().resolve()
        
        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Adjust indices (1-based to 0-based)
            start_idx = max(0, start - 1)
            end_idx = end if end else len(lines)
            
            selected = lines[start_idx:end_idx]
            
            return {
                "success": True,
                "path": str(path),
                "content": ''.join(selected),
                "start_line": start,
                "end_line": end_idx,
                "total_lines": len(lines),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# FILE WRITING
# =============================================================================

class FileWriter:
    """
    Safe file writing with security checks.
    """
    
    def write(self, path: str, content: str, 
              mode: str = 'w', encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Write to a file.
        
        Args:
            path: Path to file
            content: Content to write
            mode: 'w' for overwrite, 'a' for append
            encoding: File encoding
        
        Returns:
            Dict with success status
        """
        blocked, reason = is_path_blocked(path)
        if blocked:
            return {"success": False, "error": f"Access denied: {reason}"}
        
        path = Path(path).expanduser().resolve()
        
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, mode, encoding=encoding) as f:
                f.write(content)
            
            return {
                "success": True,
                "path": str(path),
                "bytes_written": len(content.encode(encoding)),
                "mode": "overwrite" if mode == 'w' else "append",
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def write_binary(self, path: str, content: bytes, mode: str = 'wb') -> Dict[str, Any]:
        """Write binary file."""
        blocked, reason = is_path_blocked(path)
        if blocked:
            return {"success": False, "error": f"Access denied: {reason}"}
        
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, mode) as f:
                f.write(content)
            
            return {
                "success": True,
                "path": str(path),
                "bytes_written": len(content),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# DIRECTORY OPERATIONS
# =============================================================================

class DirectoryManager:
    """
    Directory operations with security checks.
    """
    
    def list_dir(self, path: str, pattern: str = None, 
                 recursive: bool = False) -> Dict[str, Any]:
        """
        List directory contents.
        
        Args:
            path: Directory path
            pattern: Glob pattern to filter (e.g., "*.py")
            recursive: Include subdirectories
        
        Returns:
            Dict with file listing
        """
        blocked, reason = is_path_blocked(path)
        if blocked:
            return {"success": False, "error": f"Access denied: {reason}"}
        
        path = Path(path).expanduser().resolve()
        
        if not path.exists():
            return {"success": False, "error": f"Path not found: {path}"}
        
        if not path.is_dir():
            return {"success": False, "error": f"Not a directory: {path}"}
        
        try:
            entries = []
            
            if pattern:
                if recursive:
                    files = path.rglob(pattern)
                else:
                    files = path.glob(pattern)
            else:
                if recursive:
                    files = path.rglob("*")
                else:
                    files = path.iterdir()
            
            for f in files:
                # Skip blocked paths
                if is_path_blocked(str(f))[0]:
                    continue
                
                entry = {
                    "name": f.name,
                    "path": str(f),
                    "type": "directory" if f.is_dir() else "file",
                }
                
                if f.is_file():
                    stat = f.stat()
                    entry["size"] = stat.st_size
                    entry["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                
                entries.append(entry)
            
            return {
                "success": True,
                "path": str(path),
                "count": len(entries),
                "entries": entries,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_dir(self, path: str) -> Dict[str, Any]:
        """Create directory (and parents if needed)."""
        blocked, reason = is_path_blocked(path)
        if blocked:
            return {"success": False, "error": f"Access denied: {reason}"}
        
        path = Path(path).expanduser().resolve()
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            return {"success": True, "path": str(path)}
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# FILE OPERATIONS
# =============================================================================

class FileOperations:
    """
    Move, copy, delete operations.
    """
    
    def move(self, src: str, dst: str) -> Dict[str, Any]:
        """Move or rename a file."""
        for path in [src, dst]:
            blocked, reason = is_path_blocked(path)
            if blocked:
                return {"success": False, "error": f"Access denied: {reason}"}
        
        src = Path(src).expanduser().resolve()
        dst = Path(dst).expanduser().resolve()
        
        if not src.exists():
            return {"success": False, "error": f"Source not found: {src}"}
        
        try:
            # Create parent directories if needed
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(src), str(dst))
            
            return {
                "success": True,
                "source": str(src),
                "destination": str(dst),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def copy(self, src: str, dst: str) -> Dict[str, Any]:
        """Copy a file."""
        for path in [src, dst]:
            blocked, reason = is_path_blocked(path)
            if blocked:
                return {"success": False, "error": f"Access denied: {reason}"}
        
        src = Path(src).expanduser().resolve()
        dst = Path(dst).expanduser().resolve()
        
        if not src.exists():
            return {"success": False, "error": f"Source not found: {src}"}
        
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            if src.is_dir():
                shutil.copytree(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            
            return {
                "success": True,
                "source": str(src),
                "destination": str(dst),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete(self, path: str, confirm: bool = False) -> Dict[str, Any]:
        """
        Delete a file or directory.
        
        Args:
            path: Path to delete
            confirm: Must be True to actually delete
        """
        blocked, reason = is_path_blocked(path)
        if blocked:
            return {"success": False, "error": f"Access denied: {reason}"}
        
        if not confirm:
            return {"success": False, "error": "Set confirm=True to delete"}
        
        path = Path(path).expanduser().resolve()
        
        if not path.exists():
            return {"success": False, "error": f"Path not found: {path}"}
        
        try:
            if path.is_dir():
                shutil.rmtree(str(path))
            else:
                path.unlink()
            
            return {"success": True, "deleted": str(path)}
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# FILE SEARCH
# =============================================================================

class FileSearch:
    """
    Search for files by name or content.
    """
    
    def find_by_name(self, directory: str, pattern: str, 
                     recursive: bool = True) -> List[str]:
        """
        Find files by name pattern.
        
        Args:
            directory: Starting directory
            pattern: Glob pattern (e.g., "*.py", "test_*.txt")
            recursive: Search subdirectories
        
        Returns:
            List of matching file paths
        """
        blocked, reason = is_path_blocked(directory)
        if blocked:
            return []
        
        directory = Path(directory).expanduser().resolve()
        
        if not directory.exists():
            return []
        
        results = []
        
        try:
            if recursive:
                files = directory.rglob(pattern)
            else:
                files = directory.glob(pattern)
            
            for f in files:
                if not is_path_blocked(str(f))[0]:
                    results.append(str(f))
        except Exception as e:
            print(f"[FILE] Search error: {e}")
        
        return results
    
    def find_by_content(self, directory: str, text: str,
                        extensions: List[str] = None,
                        recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Find files containing specific text.
        
        Args:
            directory: Starting directory
            text: Text to search for
            extensions: File extensions to search (e.g., [".py", ".txt"])
            recursive: Search subdirectories
        
        Returns:
            List of matches with file path and line numbers
        """
        blocked, reason = is_path_blocked(directory)
        if blocked:
            return []
        
        directory = Path(directory).expanduser().resolve()
        
        if not directory.exists():
            return []
        
        results = []
        text_lower = text.lower()
        extensions = extensions or ['.txt', '.py', '.md', '.json', '.yaml', '.yml']
        
        try:
            pattern = "**/*" if recursive else "*"
            
            for f in directory.glob(pattern):
                if not f.is_file():
                    continue
                
                if f.suffix.lower() not in extensions:
                    continue
                
                if is_path_blocked(str(f))[0]:
                    continue
                
                try:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                        lines = file.readlines()
                        
                        matches = []
                        for i, line in enumerate(lines, 1):
                            if text_lower in line.lower():
                                matches.append({
                                    "line": i,
                                    "content": line.strip()[:100]
                                })
                        
                        if matches:
                            results.append({
                                "path": str(f),
                                "matches": matches
                            })
                except:
                    pass
                    
        except Exception as e:
            print(f"[FILE] Content search error: {e}")
        
        return results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def read_file(path: str, **kwargs) -> Dict[str, Any]:
    """Quick file read."""
    reader = FileReader()
    return reader.read(path, **kwargs)


def write_file(path: str, content: str, append: bool = False) -> Dict[str, Any]:
    """Quick file write."""
    writer = FileWriter()
    mode = 'a' if append else 'w'
    return writer.write(path, content, mode=mode)


def list_dir(path: str, pattern: str = None) -> Dict[str, Any]:
    """Quick directory listing."""
    manager = DirectoryManager()
    return manager.list_dir(path, pattern=pattern)


def find_files(directory: str, pattern: str) -> List[str]:
    """Quick file search by name."""
    search = FileSearch()
    return search.find_by_name(directory, pattern)


def search_content(directory: str, text: str) -> List[Dict]:
    """Quick content search."""
    search = FileSearch()
    return search.find_by_content(directory, text)


# =============================================================================
# MAIN - Run this file directly to test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ForgeAI File Tools Example")
    print("=" * 60)
    
    # Test directory
    test_dir = Path.home() / ".forge_ai" / "file_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Test file writing
    print("\n[1] Testing file write...")
    result = write_file(str(test_dir / "test.txt"), "Hello from ForgeAI!\nLine 2\nLine 3")
    print(f"  Write: {result}")
    
    # Test file reading
    print("\n[2] Testing file read...")
    result = read_file(str(test_dir / "test.txt"))
    print(f"  Read: {result['content'][:50]}...")
    
    # Test line reading
    print("\n[3] Testing line read...")
    reader = FileReader()
    result = reader.read_lines(str(test_dir / "test.txt"), start=2, end=3)
    print(f"  Lines 2-3: {result['content']}")
    
    # Test directory listing
    print("\n[4] Testing directory listing...")
    result = list_dir(str(test_dir))
    print(f"  Found {result['count']} items")
    for entry in result['entries']:
        print(f"    - {entry['name']} ({entry['type']})")
    
    # Test file search
    print("\n[5] Testing file search...")
    files = find_files(str(test_dir), "*.txt")
    print(f"  Found {len(files)} .txt files")
    
    # Test content search
    print("\n[6] Testing content search...")
    write_file(str(test_dir / "search_test.txt"), "This contains the word ForgeAI")
    results = search_content(str(test_dir), "ForgeAI")
    print(f"  Found 'ForgeAI' in {len(results)} files")
    
    # Test security
    print("\n[7] Testing security blocks...")
    blocked_paths = ["~/.ssh/id_rsa", "/etc/passwd", "secrets.key"]
    for path in blocked_paths:
        blocked, reason = is_path_blocked(path)
        print(f"  {path}: {'BLOCKED' if blocked else 'allowed'}")
    
    # Cleanup
    print("\n[8] Cleaning up test files...")
    ops = FileOperations()
    ops.delete(str(test_dir), confirm=True)
    print("  Cleaned up test directory")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nSecurity note: AI file access is restricted.")
    print("Blocked: ~/.ssh, /etc/passwd, *.key, etc.")
