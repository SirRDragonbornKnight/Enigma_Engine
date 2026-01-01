"""
System Tools - Screenshots, commands, and system info.

Tools:
  - run_command: Execute a shell command
  - screenshot: Take a screenshot
  - get_system_info: Get system information
"""

import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any
from .tool_registry import Tool


class RunCommandTool(Tool):
    """
    Execute a shell command.
    Use with caution - can execute arbitrary code!
    """
    
    name = "run_command"
    description = "Execute a shell command and return the output. Use carefully!"
    parameters = {
        "command": "The shell command to execute",
        "timeout": "Maximum seconds to wait (default: 30)",
        "cwd": "Working directory (default: current directory)",
    }
    
    # Dangerous commands that are blocked
    BLOCKED_COMMANDS = [
        "rm -rf /", "rm -rf ~", "mkfs", "dd if=", ":(){:|:&};:",
        "chmod -R 777 /", "chown -R", "> /dev/sda",
    ]
    
    def execute(self, command: str, timeout: int = 30, cwd: str = None, **kwargs) -> Dict[str, Any]:
        try:
            # Safety check
            for blocked in self.BLOCKED_COMMANDS:
                if blocked in command:
                    return {"success": False, "error": f"Blocked dangerous command: {blocked}"}
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )
            
            return {
                "success": result.returncode == 0,
                "command": command,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ScreenshotTool(Tool):
    """Take a screenshot using vision module's ScreenCapture."""
    
    name = "screenshot"
    description = "Take a screenshot and save it to a file."
    parameters = {
        "output_path": "Where to save the screenshot (default: screenshot.png)",
        "region": "Optional region as 'x,y,width,height' (default: full screen)",
    }
    
    def __init__(self):
        super().__init__()
        self._capture = None
    
    def _get_capture(self):
        """Lazy load ScreenCapture to avoid circular imports."""
        if self._capture is None:
            from .vision import ScreenCapture
            self._capture = ScreenCapture()
        return self._capture
    
    def execute(self, output_path: str = "screenshot.png", region: str = None, **kwargs) -> Dict[str, Any]:
        try:
            output_path = Path(output_path).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Parse region if provided
            region_tuple = None
            if region:
                parts = [int(x) for x in region.split(",")]
                if len(parts) == 4:
                    region_tuple = tuple(parts)
            
            # Use vision.ScreenCapture (consolidated backend)
            capture = self._get_capture()
            img = capture.capture(region=region_tuple)
            
            if img:
                img.save(str(output_path))
                return {
                    "success": True,
                    "path": str(output_path),
                    "size": f"{img.width}x{img.height}",
                    "backend": capture._backend,
                }
            else:
                return {
                    "success": False,
                    "error": "No screenshot method available. Install Pillow, mss, or scrot."
                }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class GetSystemInfoTool(Tool):
    """Get system information."""
    
    name = "get_system_info"
    description = "Get information about the system (OS, CPU, memory, disk, etc.)"
    parameters = {}
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        try:
            pass
            
            info = {
                "os": platform.system(),
                "os_release": platform.release(),
                "os_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
            }
            
            # CPU count
            info["cpu_count"] = os.cpu_count()
            
            # Memory (try psutil)
            try:
                import psutil
                mem = psutil.virtual_memory()
                info["memory_total_gb"] = round(mem.total / (1024**3), 2)
                info["memory_available_gb"] = round(mem.available / (1024**3), 2)
                info["memory_percent_used"] = mem.percent
                
                # Disk
                disk = psutil.disk_usage("/")
                info["disk_total_gb"] = round(disk.total / (1024**3), 2)
                info["disk_free_gb"] = round(disk.free / (1024**3), 2)
                info["disk_percent_used"] = disk.percent
            except ImportError:
                info["memory"] = "Install psutil for memory info"
            
            # GPU (try torch or nvidia-smi)
            try:
                import torch
                if torch.cuda.is_available():
                    info["gpu_available"] = True
                    info["gpu_count"] = torch.cuda.device_count()
                    info["gpu_name"] = torch.cuda.get_device_name(0)
                    info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                else:
                    info["gpu_available"] = False
            except ImportError:
                info["gpu"] = "Install torch for GPU info"
            
            return {
                "success": True,
                "info": info
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import json
    
    # Test system info
    tool = GetSystemInfoTool()
    result = tool.execute()
    print(json.dumps(result, indent=2))
