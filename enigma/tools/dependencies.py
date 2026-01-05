"""
Tool Dependency Checker
========================

Checks if required dependencies (Python modules and CLI commands) are available
for tool execution. Provides installation instructions when dependencies are missing.
"""

import logging
import subprocess
import sys
import importlib
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# Python module dependencies per tool
TOOL_DEPENDENCIES = {
    "web_search": ["requests", "beautifulsoup4"],
    "fetch_webpage": ["requests", "beautifulsoup4"],
    "read_document": ["pypdf2", "python-docx", "ebooklib"],
    "extract_text": ["pypdf2", "python-docx"],
    "generate_image": ["torch", "diffusers", "transformers"],
    "generate_video": ["torch", "diffusers", "transformers", "moviepy"],
    "generate_audio": ["torch", "transformers"],
    "analyze_image": ["torch", "transformers", "pillow"],
    "vision": ["pillow", "mss", "pyscreenshot"],
    "edit_video": ["moviepy"],
    "speak": ["pyttsx3"],
    "voice_input": ["SpeechRecognition", "pyaudio", "vosk"],
    "screenshot": ["mss", "pillow", "pyscreenshot"],
    "avatar": ["pygame"],
    "motion_tracking": ["mediapipe", "opencv-python"],
    "ocr": ["pytesseract", "pillow"],
}


# CLI command dependencies per tool
TOOL_COMMANDS = {
    "run_command": ["bash"],  # or cmd on Windows
    "screenshot": ["scrot", "gnome-screenshot"],  # Optional, fallback to Python
    "ocr": ["tesseract"],  # For pytesseract
    "video_tools": ["ffmpeg"],  # For moviepy
}


# Installation instructions for common packages
INSTALL_INSTRUCTIONS = {
    "requests": "pip install requests",
    "beautifulsoup4": "pip install beautifulsoup4",
    "pypdf2": "pip install pypdf2",
    "python-docx": "pip install python-docx",
    "ebooklib": "pip install ebooklib",
    "torch": "pip install torch torchvision torchaudio (see https://pytorch.org for platform-specific instructions)",
    "diffusers": "pip install diffusers",
    "transformers": "pip install transformers",
    "moviepy": "pip install moviepy",
    "pillow": "pip install pillow",
    "mss": "pip install mss",
    "pyscreenshot": "pip install pyscreenshot",
    "pyttsx3": "pip install pyttsx3",
    "SpeechRecognition": "pip install SpeechRecognition",
    "pyaudio": "pip install pyaudio (may require system dependencies)",
    "vosk": "pip install vosk",
    "pygame": "pip install pygame",
    "mediapipe": "pip install mediapipe",
    "opencv-python": "pip install opencv-python",
    "pytesseract": "pip install pytesseract",
    
    # CLI tools
    "tesseract": "Install Tesseract OCR from https://github.com/tesseract-ocr/tesseract",
    "ffmpeg": "Install FFmpeg from https://ffmpeg.org or via package manager",
    "scrot": "Install via package manager (apt install scrot)",
    "gnome-screenshot": "Install via package manager (apt install gnome-screenshot)",
}


class ToolDependencyChecker:
    """
    Check dependencies for tool execution.
    
    Features:
    - Check Python module availability
    - Check CLI command availability
    - Provide installation instructions
    - Cache check results
    """
    
    def __init__(self):
        """Initialize dependency checker."""
        # Cache for checked modules/commands
        self._module_cache: Dict[str, bool] = {}
        self._command_cache: Dict[str, bool] = {}
        
        logger.info("ToolDependencyChecker initialized")
    
    def _check_module(self, module_name: str) -> bool:
        """
        Check if a Python module is available.
        
        Args:
            module_name: Name of the module
            
        Returns:
            True if available, False otherwise
        """
        if module_name in self._module_cache:
            return self._module_cache[module_name]
        
        try:
            importlib.import_module(module_name)
            self._module_cache[module_name] = True
            return True
        except ImportError:
            self._module_cache[module_name] = False
            return False
    
    def _check_command(self, command_name: str) -> bool:
        """
        Check if a CLI command is available.
        
        Args:
            command_name: Name of the command
            
        Returns:
            True if available, False otherwise
        """
        if command_name in self._command_cache:
            return self._command_cache[command_name]
        
        try:
            # Try to run the command with --version or --help
            if sys.platform == "win32":
                result = subprocess.run(
                    ["where", command_name],
                    capture_output=True,
                    timeout=5
                )
            else:
                result = subprocess.run(
                    ["which", command_name],
                    capture_output=True,
                    timeout=5
                )
            
            available = result.returncode == 0
            self._command_cache[command_name] = available
            return available
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._command_cache[command_name] = False
            return False
    
    def check_tool(self, tool_name: str) -> Tuple[bool, List[str]]:
        """
        Check if all dependencies for a tool are available.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            (all_available, missing_dependencies)
        """
        missing = []
        
        # Check Python modules
        if tool_name in TOOL_DEPENDENCIES:
            for module in TOOL_DEPENDENCIES[tool_name]:
                if not self._check_module(module):
                    missing.append(f"Python module: {module}")
        
        # Check CLI commands
        if tool_name in TOOL_COMMANDS:
            for command in TOOL_COMMANDS[tool_name]:
                if not self._check_command(command):
                    missing.append(f"Command: {command}")
        
        all_available = len(missing) == 0
        return all_available, missing
    
    def check_all_tools(self) -> Dict[str, Tuple[bool, List[str]]]:
        """
        Check dependencies for all tools.
        
        Returns:
            Dictionary mapping tool_name to (available, missing_deps)
        """
        results = {}
        
        all_tools = set(TOOL_DEPENDENCIES.keys()) | set(TOOL_COMMANDS.keys())
        
        for tool_name in all_tools:
            results[tool_name] = self.check_tool(tool_name)
        
        return results
    
    def get_install_instructions(self, dependency: str) -> str:
        """
        Get installation instructions for a dependency.
        
        Args:
            dependency: Name of the dependency (module or command)
            
        Returns:
            Installation instruction string
        """
        # Clean up dependency name (remove "Python module: " prefix etc)
        clean_name = dependency.replace("Python module: ", "").replace("Command: ", "").strip()
        
        if clean_name in INSTALL_INSTRUCTIONS:
            return INSTALL_INSTRUCTIONS[clean_name]
        else:
            return f"pip install {clean_name} (installation instructions not available)"
    
    def get_missing_report(self, tool_name: Optional[str] = None) -> str:
        """
        Generate a human-readable report of missing dependencies.
        
        Args:
            tool_name: Check specific tool, or None for all tools
            
        Returns:
            Formatted report string
        """
        if tool_name:
            available, missing = self.check_tool(tool_name)
            
            if available:
                return f"✓ All dependencies available for '{tool_name}'"
            else:
                report = [f"✗ Missing dependencies for '{tool_name}':"]
                for dep in missing:
                    instructions = self.get_install_instructions(dep)
                    report.append(f"  - {dep}")
                    report.append(f"    Install: {instructions}")
                return "\n".join(report)
        else:
            # Check all tools
            all_results = self.check_all_tools()
            
            available_tools = []
            unavailable_tools = []
            
            for tool, (avail, missing) in all_results.items():
                if avail:
                    available_tools.append(tool)
                else:
                    unavailable_tools.append((tool, missing))
            
            report = ["Tool Dependency Report", "=" * 50]
            report.append(f"\n✓ Available tools ({len(available_tools)}): {', '.join(sorted(available_tools))}")
            
            if unavailable_tools:
                report.append(f"\n✗ Unavailable tools ({len(unavailable_tools)}):\n")
                
                for tool, missing in unavailable_tools:
                    report.append(f"  {tool}:")
                    for dep in missing:
                        instructions = self.get_install_instructions(dep)
                        report.append(f"    - {dep}")
                        report.append(f"      Install: {instructions}")
                    report.append("")
            
            return "\n".join(report)
    
    def clear_cache(self):
        """Clear dependency check cache."""
        self._module_cache.clear()
        self._command_cache.clear()
        logger.info("Cleared dependency check cache")
    
    def get_statistics(self) -> Dict:
        """Get dependency checker statistics."""
        all_results = self.check_all_tools()
        
        available_count = sum(1 for avail, _ in all_results.values() if avail)
        unavailable_count = len(all_results) - available_count
        
        return {
            "total_tools_checked": len(all_results),
            "available_tools": available_count,
            "unavailable_tools": unavailable_count,
            "cached_modules": len(self._module_cache),
            "cached_commands": len(self._command_cache),
        }


__all__ = [
    "ToolDependencyChecker",
    "TOOL_DEPENDENCIES",
    "TOOL_COMMANDS",
    "INSTALL_INSTRUCTIONS",
]
