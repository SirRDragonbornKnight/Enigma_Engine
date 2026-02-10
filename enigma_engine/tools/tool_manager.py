"""
Tool Manager - Enable/disable tools for lightweight deployments.

Allows users to:
  - Enable/disable individual tools
  - Use preset profiles (minimal, standard, full)
  - See tool dependencies and requirements
  - Create custom profiles for specific use cases

Perfect for:
  - Raspberry Pi / embedded devices
  - Router deployments (single-purpose AI)
  - Memory-constrained environments
  - Single-purpose installations (just camera, just image gen, etc.)

USAGE:
    from enigma_engine.tools.tool_manager import ToolManager
    
    manager = ToolManager()
    
    # Apply a preset
    manager.apply_preset("minimal")  # Just core tools
    manager.apply_preset("camera_only")  # Just camera tools
    manager.apply_preset("full")  # Everything
    
    # Enable/disable individual tools
    manager.disable_tool("dnd_roll")
    manager.enable_tool("camera_capture")
    
    # Check what's enabled
    manager.list_enabled()
    manager.list_disabled()
    
    # Get tool info
    manager.get_tool_info("image_generate")
    
    # Save/load custom profiles
    manager.save_profile("my_router_setup")
    manager.load_profile("my_router_setup")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Config path
CONFIG_DIR = Path.home() / ".enigma_engine"
TOOL_CONFIG_FILE = CONFIG_DIR / "tool_config.json"
PROFILES_DIR = CONFIG_DIR / "tool_profiles"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
PROFILES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# TOOL METADATA - Categories, dependencies, descriptions
# ============================================================================

TOOL_CATEGORIES = {
    # Core tools (always recommended)
    "core": [
        "web_search", "fetch_webpage", "read_file", "write_file", 
        "list_directory", "move_file", "delete_file", "read_document",
        "extract_text", "run_command", "screenshot", "get_system_info",
    ],
    
    # Interactive/Assistant tools
    "assistant": [
        "create_checklist", "list_checklists", "add_task", "list_tasks",
        "complete_task", "set_reminder", "list_reminders", "check_reminders",
    ],
    
    # Vision tools
    "vision": [
        "screen_vision", "find_on_screen",
    ],
    
    # Robot tools
    "robot": [
        "robot_move", "robot_gripper", "robot_status", "robot_home",
    ],
    
    # Automation tools
    "automation": [
        "schedule_task", "list_schedules", "remove_schedule",
        "clipboard_read", "clipboard_write", "clipboard_history",
        "record_macro", "play_macro", "list_macros",
        "watch_folder", "stop_watch", "list_watches",
    ],
    
    # Knowledge tools
    "knowledge": [
        "wikipedia_search", "arxiv_search", "pdf_extract",
        "bookmark_save", "bookmark_search", "bookmark_list", "bookmark_delete",
        "note_save", "note_get", "note_search", "note_list", "note_delete",
    ],
    
    # Communication tools
    "communication": [
        "translate_text",  # External API more accurate than AI
        "ocr_image", "ocr_screenshot",  # Needs OCR library
    ],
    
    # Media tools
    "media": [
        "music_generate", "remove_background", "upscale_image",
        "style_transfer", "convert_audio", "extract_audio", "audio_visualize",
    ],
    
    # Productivity tools
    "productivity": [
        "system_monitor", "process_list", "process_kill",
        "ssh_execute", "ssh_copy", "docker_list", "docker_control",
        "docker_images", "git_status", "git_commit", "git_diff",
        "git_push", "git_pull",
    ],
    
    # IoT tools
    "iot": [
        "homeassistant_setup", "homeassistant_control", "homeassistant_status",
        "gpio_read", "gpio_write", "gpio_pwm",
        "mqtt_publish", "mqtt_subscribe",
        "camera_capture", "camera_list", "camera_stream",
    ],
    
    # Data tools
    "data": [
        "csv_analyze", "csv_query", "plot_chart",
        "json_query", "json_transform",
        "sql_query", "sql_execute", "sql_tables", "data_convert",
    ],
    
    # Gaming tools - Only dice (needs true randomness)
    # Other gaming tools removed - AI can roleplay, tell stories, play games natively
    "gaming": [
        "dnd_roll",  # True random dice - AI can't do real randomness
    ],
    
    # Browser Media Control tools
    "browser": [
        "browser_media_pause", "browser_media_mute", "browser_media_skip",
        "browser_media_stop", "browser_media_volume", "browser_media_info",
        "browser_tab_list", "browser_focus",
    ],
    
    # Avatar control tools - AI controls the desktop avatar
    "avatar": [
        "control_avatar", "customize_avatar", "avatar_gesture", "avatar_emotion",
    ],
    
    # Self-modification tools - AI customizes itself
    "self": [
        "set_personality", "set_avatar_preference", "set_voice_preference",
        "set_companion_behavior", "set_preference", "get_self_config",
        "remember_fact", "recall_facts",
        "generate_avatar", "open_avatar_in_blender", "list_avatars", "set_avatar",
        "spawn_object", "remove_object", "list_spawned_objects",
        "spawn_screen_effect", "stop_screen_effect", "list_effect_assets",
        "fullscreen_mode_control",
    ],
    
    # Memory tools - AI can search and manage conversation history
    "memory": [
        "search_memory", "memory_stats", "export_memory", "import_memory",
    ],
}

# Tool dependencies (Python packages required)
TOOL_DEPENDENCIES = {
    # Web
    "web_search": ["requests"],
    "fetch_webpage": ["requests", "beautifulsoup4"],
    
    # Documents
    "read_document": ["PyPDF2", "python-docx", "ebooklib"],
    "pdf_extract": ["PyPDF2", "pdfplumber"],
    
    # Vision
    "screen_vision": ["pillow", "mss"],
    "find_on_screen": ["pillow", "opencv-python"],
    "ocr_image": ["pytesseract", "pillow"],
    "ocr_screenshot": ["pytesseract", "pillow", "mss"],
    
    # Media
    "music_generate": ["scipy", "numpy"],
    "remove_background": ["rembg", "pillow"],
    "upscale_image": ["pillow", "opencv-python"],
    "style_transfer": ["pillow", "numpy"],
    "convert_audio": ["pydub"],
    "extract_audio": ["moviepy"],
    "audio_visualize": ["matplotlib", "scipy", "numpy"],
    
    # Data
    "csv_analyze": ["pandas"],
    "csv_query": ["pandas"],
    "plot_chart": ["matplotlib", "pandas"],
    "sql_query": ["sqlite3"],  # Built-in
    
    # IoT
    "gpio_read": ["RPi.GPIO"],
    "gpio_write": ["RPi.GPIO"],
    "gpio_pwm": ["RPi.GPIO"],
    "mqtt_publish": ["paho-mqtt"],
    "mqtt_subscribe": ["paho-mqtt"],
    "camera_capture": ["opencv-python"],
    "camera_stream": ["opencv-python", "flask"],
    
    # Productivity
    "ssh_execute": ["paramiko"],
    "ssh_copy": ["paramiko"],
    "docker_list": ["docker"],
    "docker_control": ["docker"],
    "docker_images": ["docker"],
    
    # Communication
    "translate_text": ["deep-translator"],
    "detect_language": ["langdetect"],
    
    # Automation
    "clipboard_read": ["pyperclip"],
    "clipboard_write": ["pyperclip"],
    "clipboard_history": ["pyperclip"],
}

# Presets for different use cases
PRESETS = {
    "minimal": {
        "description": "Minimal tools for embedded/router deployments",
        "categories": ["core"],
        "extra_tools": [],
    },
    "standard": {
        "description": "Standard tools for desktop use",
        "categories": ["core", "assistant", "knowledge", "productivity", "avatar", "self"],
        "extra_tools": [],
    },
    "full": {
        "description": "All tools enabled",
        "categories": list(TOOL_CATEGORIES.keys()),
        "extra_tools": [],
    },
    "camera_only": {
        "description": "Just camera and vision tools for security/monitoring",
        "categories": [],
        "extra_tools": ["camera_capture", "camera_list", "camera_stream", 
                       "screen_vision", "find_on_screen"],
    },
    "iot_hub": {
        "description": "IoT/Smart Home hub setup",
        "categories": ["core", "iot"],
        "extra_tools": ["schedule_task", "list_schedules"],
    },
    "data_analyst": {
        "description": "Data analysis and visualization",
        "categories": ["core", "data", "knowledge"],
        "extra_tools": ["pdf_extract"],
    },
    "developer": {
        "description": "Development and DevOps tools",
        "categories": ["core", "productivity"],
        "extra_tools": ["ssh_execute", "ssh_copy"],
    },
    "assistant": {
        "description": "Personal assistant mode",
        "categories": ["core", "assistant", "knowledge", "communication", "avatar"],
        "extra_tools": ["schedule_task", "set_reminder"],
    },
    "gaming": {
        "description": "Fun and games mode",
        "categories": ["core", "gaming"],
        "extra_tools": [],
    },
    "training_only": {
        "description": "Minimal for AI training - no tools needed",
        "categories": [],
        "extra_tools": [],
    },
}


class ToolManager:
    """
    Manages which tools are enabled/disabled.
    
    Allows lightweight deployments by disabling unused tools.
    """
    
    def __init__(self):
        self.enabled_tools: Set[str] = set()
        self.disabled_tools: Set[str] = set()
        self.current_profile: str = "full"
        self._load_config()
    
    def _get_all_tools(self) -> Set[str]:
        """Get all available tool names."""
        all_tools = set()
        for tools in TOOL_CATEGORIES.values():
            all_tools.update(tools)
        return all_tools
    
    def _load_config(self):
        """Load tool configuration from file."""
        if TOOL_CONFIG_FILE.exists():
            try:
                with open(TOOL_CONFIG_FILE) as f:
                    config = json.load(f)
                    self.enabled_tools = set(config.get("enabled", []))
                    self.disabled_tools = set(config.get("disabled", []))
                    self.current_profile = config.get("profile", "full")
            except Exception:
                # Default to full
                self._apply_preset_internal("full")
        else:
            # Default to full on first run
            self._apply_preset_internal("full")
    
    def _save_config(self):
        """Save tool configuration to file."""
        config = {
            "enabled": list(self.enabled_tools),
            "disabled": list(self.disabled_tools),
            "profile": self.current_profile,
            "updated": datetime.now().isoformat(),
        }
        with open(TOOL_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _apply_preset_internal(self, preset_name: str):
        """Apply a preset without saving (internal use)."""
        if preset_name not in PRESETS:
            return
        
        preset = PRESETS[preset_name]
        self.enabled_tools = set()
        
        # Add tools from categories
        for category in preset["categories"]:
            if category in TOOL_CATEGORIES:
                self.enabled_tools.update(TOOL_CATEGORIES[category])
        
        # Add extra tools
        self.enabled_tools.update(preset.get("extra_tools", []))
        
        # Set disabled as everything not enabled
        all_tools = self._get_all_tools()
        self.disabled_tools = all_tools - self.enabled_tools
        self.current_profile = preset_name
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def apply_preset(self, preset_name: str) -> Dict[str, Any]:
        """
        Apply a preset configuration.
        
        Presets: minimal, standard, full, camera_only, iot_hub, 
                 data_analyst, developer, assistant, gaming, training_only
        """
        if preset_name not in PRESETS:
            return {
                "success": False,
                "error": f"Unknown preset '{preset_name}'",
                "available": list(PRESETS.keys()),
            }
        
        self._apply_preset_internal(preset_name)
        self._save_config()
        
        return {
            "success": True,
            "preset": preset_name,
            "description": PRESETS[preset_name]["description"],
            "enabled_count": len(self.enabled_tools),
            "disabled_count": len(self.disabled_tools),
        }
    
    def enable_tool(self, tool_name: str) -> Dict[str, Any]:
        """Enable a specific tool."""
        all_tools = self._get_all_tools()
        
        if tool_name not in all_tools:
            return {"success": False, "error": f"Unknown tool '{tool_name}'"}
        
        self.enabled_tools.add(tool_name)
        self.disabled_tools.discard(tool_name)
        self.current_profile = "custom"
        self._save_config()
        
        return {
            "success": True,
            "tool": tool_name,
            "status": "enabled",
            "dependencies": TOOL_DEPENDENCIES.get(tool_name, []),
        }
    
    def disable_tool(self, tool_name: str) -> Dict[str, Any]:
        """Disable a specific tool."""
        all_tools = self._get_all_tools()
        
        if tool_name not in all_tools:
            return {"success": False, "error": f"Unknown tool '{tool_name}'"}
        
        self.disabled_tools.add(tool_name)
        self.enabled_tools.discard(tool_name)
        self.current_profile = "custom"
        self._save_config()
        
        return {
            "success": True,
            "tool": tool_name,
            "status": "disabled",
        }
    
    def enable_category(self, category: str) -> Dict[str, Any]:
        """Enable all tools in a category."""
        if category not in TOOL_CATEGORIES:
            return {
                "success": False,
                "error": f"Unknown category '{category}'",
                "available": list(TOOL_CATEGORIES.keys()),
            }
        
        tools = TOOL_CATEGORIES[category]
        for tool in tools:
            self.enabled_tools.add(tool)
            self.disabled_tools.discard(tool)
        
        self.current_profile = "custom"
        self._save_config()
        
        return {
            "success": True,
            "category": category,
            "tools_enabled": tools,
            "count": len(tools),
        }
    
    def disable_category(self, category: str) -> Dict[str, Any]:
        """Disable all tools in a category."""
        if category not in TOOL_CATEGORIES:
            return {
                "success": False,
                "error": f"Unknown category '{category}'",
                "available": list(TOOL_CATEGORIES.keys()),
            }
        
        tools = TOOL_CATEGORIES[category]
        for tool in tools:
            self.disabled_tools.add(tool)
            self.enabled_tools.discard(tool)
        
        self.current_profile = "custom"
        self._save_config()
        
        return {
            "success": True,
            "category": category,
            "tools_disabled": tools,
            "count": len(tools),
        }
    
    def is_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled."""
        return tool_name in self.enabled_tools
    
    def list_enabled(self) -> Dict[str, Any]:
        """List all enabled tools."""
        # Group by category
        by_category = {}
        for category, tools in TOOL_CATEGORIES.items():
            enabled_in_cat = [t for t in tools if t in self.enabled_tools]
            if enabled_in_cat:
                by_category[category] = enabled_in_cat
        
        return {
            "success": True,
            "profile": self.current_profile,
            "total_enabled": len(self.enabled_tools),
            "by_category": by_category,
            "tools": sorted(self.enabled_tools),
        }
    
    def list_disabled(self) -> Dict[str, Any]:
        """List all disabled tools."""
        by_category = {}
        for category, tools in TOOL_CATEGORIES.items():
            disabled_in_cat = [t for t in tools if t in self.disabled_tools]
            if disabled_in_cat:
                by_category[category] = disabled_in_cat
        
        return {
            "success": True,
            "total_disabled": len(self.disabled_tools),
            "by_category": by_category,
            "tools": sorted(self.disabled_tools),
        }
    
    def list_categories(self) -> Dict[str, Any]:
        """List all tool categories with counts."""
        categories = []
        for name, tools in TOOL_CATEGORIES.items():
            enabled_count = sum(1 for t in tools if t in self.enabled_tools)
            categories.append({
                "name": name,
                "total": len(tools),
                "enabled": enabled_count,
                "disabled": len(tools) - enabled_count,
            })
        
        return {
            "success": True,
            "categories": categories,
        }
    
    def list_presets(self) -> Dict[str, Any]:
        """List available presets."""
        presets = []
        for name, info in PRESETS.items():
            # Calculate tool count for this preset
            count = 0
            for cat in info["categories"]:
                if cat in TOOL_CATEGORIES:
                    count += len(TOOL_CATEGORIES[cat])
            count += len(info.get("extra_tools", []))
            
            presets.append({
                "name": name,
                "description": info["description"],
                "tool_count": count,
                "active": name == self.current_profile,
            })
        
        return {
            "success": True,
            "current": self.current_profile,
            "presets": presets,
        }
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed info about a tool."""
        all_tools = self._get_all_tools()
        
        if tool_name not in all_tools:
            return {"success": False, "error": f"Unknown tool '{tool_name}'"}
        
        # Find category
        category = None
        for cat, tools in TOOL_CATEGORIES.items():
            if tool_name in tools:
                category = cat
                break
        
        return {
            "success": True,
            "name": tool_name,
            "category": category,
            "enabled": tool_name in self.enabled_tools,
            "dependencies": TOOL_DEPENDENCIES.get(tool_name, []),
        }
    
    def get_dependencies(self, tool_name: str = None) -> Dict[str, Any]:
        """Get dependencies for a tool or all enabled tools."""
        if tool_name:
            deps = TOOL_DEPENDENCIES.get(tool_name, [])
            return {
                "success": True,
                "tool": tool_name,
                "dependencies": deps,
            }
        
        # Get all dependencies for enabled tools
        all_deps = set()
        tool_deps = {}
        
        for tool in self.enabled_tools:
            deps = TOOL_DEPENDENCIES.get(tool, [])
            if deps:
                tool_deps[tool] = deps
                all_deps.update(deps)
        
        return {
            "success": True,
            "total_packages": len(all_deps),
            "packages": sorted(all_deps),
            "by_tool": tool_deps,
        }
    
    def save_profile(self, profile_name: str) -> Dict[str, Any]:
        """Save current configuration as a custom profile."""
        profile = {
            "name": profile_name,
            "description": f"Custom profile created {datetime.now().strftime('%Y-%m-%d')}",
            "enabled_tools": list(self.enabled_tools),
            "created": datetime.now().isoformat(),
        }
        
        profile_file = PROFILES_DIR / f"{profile_name}.json"
        with open(profile_file, 'w') as f:
            json.dump(profile, f, indent=2)
        
        return {
            "success": True,
            "profile": profile_name,
            "path": str(profile_file),
            "tool_count": len(self.enabled_tools),
        }
    
    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """Load a custom profile."""
        profile_file = PROFILES_DIR / f"{profile_name}.json"
        
        if not profile_file.exists():
            # Check built-in presets
            if profile_name in PRESETS:
                return self.apply_preset(profile_name)
            return {"success": False, "error": f"Profile '{profile_name}' not found"}
        
        try:
            with open(profile_file) as f:
                profile = json.load(f)
            
            self.enabled_tools = set(profile.get("enabled_tools", []))
            all_tools = self._get_all_tools()
            self.disabled_tools = all_tools - self.enabled_tools
            self.current_profile = profile_name
            self._save_config()
            
            return {
                "success": True,
                "profile": profile_name,
                "description": profile.get("description", ""),
                "enabled_count": len(self.enabled_tools),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_profiles(self) -> Dict[str, Any]:
        """List all saved custom profiles."""
        profiles = []
        
        for file in PROFILES_DIR.glob("*.json"):
            try:
                with open(file) as f:
                    profile = json.load(f)
                    profiles.append({
                        "name": profile.get("name", file.stem),
                        "description": profile.get("description", ""),
                        "tool_count": len(profile.get("enabled_tools", [])),
                        "created": profile.get("created", ""),
                    })
            except (json.JSONDecodeError, OSError, KeyError) as e:
                logger.warning(f"Could not load profile {file}: {e}")
        
        return {
            "success": True,
            "custom_profiles": profiles,
            "builtin_presets": list(PRESETS.keys()),
        }
    
    def delete_profile(self, profile_name: str) -> Dict[str, Any]:
        """Delete a custom profile."""
        profile_file = PROFILES_DIR / f"{profile_name}.json"
        
        if not profile_file.exists():
            return {"success": False, "error": f"Profile '{profile_name}' not found"}
        
        profile_file.unlink()
        
        return {
            "success": True,
            "deleted": profile_name,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool statistics."""
        all_tools = self._get_all_tools()
        
        # Count dependencies
        total_deps = set()
        for tool in self.enabled_tools:
            total_deps.update(TOOL_DEPENDENCIES.get(tool, []))
        
        return {
            "success": True,
            "profile": self.current_profile,
            "total_tools": len(all_tools),
            "enabled": len(self.enabled_tools),
            "disabled": len(self.disabled_tools),
            "categories": len(TOOL_CATEGORIES),
            "required_packages": len(total_deps),
        }


# Global instance
_manager: Optional[ToolManager] = None

def get_tool_manager() -> ToolManager:
    """Get the global tool manager instance."""
    global _manager
    if _manager is None:
        _manager = ToolManager()
    return _manager


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for tool management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Forge Tool Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List tools")
    list_parser.add_argument("--enabled", action="store_true", help="Show only enabled")
    list_parser.add_argument("--disabled", action="store_true", help="Show only disabled")
    list_parser.add_argument("--category", help="Filter by category")
    
    # Enable command
    enable_parser = subparsers.add_parser("enable", help="Enable tool(s)")
    enable_parser.add_argument("target", help="Tool name or category")
    enable_parser.add_argument("--category", action="store_true", help="Enable entire category")
    
    # Disable command
    disable_parser = subparsers.add_parser("disable", help="Disable tool(s)")
    disable_parser.add_argument("target", help="Tool name or category")
    disable_parser.add_argument("--category", action="store_true", help="Disable entire category")
    
    # Preset command
    preset_parser = subparsers.add_parser("preset", help="Apply a preset")
    preset_parser.add_argument("name", nargs="?", help="Preset name")
    preset_parser.add_argument("--list", action="store_true", help="List presets")
    
    # Profile commands
    profile_parser = subparsers.add_parser("profile", help="Manage profiles")
    profile_parser.add_argument("action", choices=["save", "load", "list", "delete"])
    profile_parser.add_argument("name", nargs="?", help="Profile name")
    
    # Stats command
    subparsers.add_parser("stats", help="Show statistics")
    
    # Deps command
    deps_parser = subparsers.add_parser("deps", help="Show dependencies")
    deps_parser.add_argument("tool", nargs="?", help="Specific tool (optional)")
    
    args = parser.parse_args()
    manager = get_tool_manager()
    
    if args.command == "list":
        if args.enabled:
            result = manager.list_enabled()
        elif args.disabled:
            result = manager.list_disabled()
        else:
            result = manager.list_categories()
        print(json.dumps(result, indent=2))
    
    elif args.command == "enable":
        if args.category:
            result = manager.enable_category(args.target)
        else:
            result = manager.enable_tool(args.target)
        print(json.dumps(result, indent=2))
    
    elif args.command == "disable":
        if args.category:
            result = manager.disable_category(args.target)
        else:
            result = manager.disable_tool(args.target)
        print(json.dumps(result, indent=2))
    
    elif args.command == "preset":
        if args.list or not args.name:
            result = manager.list_presets()
        else:
            result = manager.apply_preset(args.name)
        print(json.dumps(result, indent=2))
    
    elif args.command == "profile":
        if args.action == "list":
            result = manager.list_profiles()
        elif args.action == "save" and args.name:
            result = manager.save_profile(args.name)
        elif args.action == "load" and args.name:
            result = manager.load_profile(args.name)
        elif args.action == "delete" and args.name:
            result = manager.delete_profile(args.name)
        else:
            result = {"error": "Missing profile name"}
        print(json.dumps(result, indent=2))
    
    elif args.command == "stats":
        result = manager.get_stats()
        print(json.dumps(result, indent=2))
    
    elif args.command == "deps":
        result = manager.get_dependencies(args.tool)
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
