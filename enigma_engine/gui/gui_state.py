"""
================================================================================
🎛️ GUI STATE MANAGER - AI ↔ GUI Bridge
================================================================================

Singleton that provides a clean interface for AI tools to interact with the GUI.
This allows the AI to control the interface, read settings, and help users
navigate - all without tight coupling to the window implementation.

📍 FILE: enigma_engine/gui/gui_state.py
🏷️ TYPE: Singleton State Manager
🎯 MAIN CLASS: GUIStateManager

┌─────────────────────────────────────────────────────────────────────────────┐
│  ARCHITECTURE:                                                              │
│                                                                             │
│  AI Tools (tool_executor.py)                                               │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────┐                                                   │
│  │  GUIStateManager    │ ◄── Singleton, thread-safe                        │
│  │  - switch_tab()     │                                                   │
│  │  - get_setting()    │                                                   │
│  │  - set_setting()    │                                                   │
│  │  - manage_chat()    │                                                   │
│  └─────────────────────┘                                                   │
│       │                                                                     │
│       ▼                                                                     │
│  EnhancedMainWindow (enhanced_window.py)                                   │
│       │                                                                     │
│       ▼                                                                     │
│  gui_settings.json (persisted preferences)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

🔗 CONNECTED FILES:
    → USES:      data/gui_settings.json (settings persistence)
    → USES:      enigma_engine/config/defaults.py (CONFIG)
    ← USED BY:   enigma_engine/tools/tool_executor.py (AI tool execution)
    ← USED BY:   enigma_engine/gui/enhanced_window.py (window registration)

📖 USAGE:
    # From enhanced_window.py (on startup):
    from enigma_engine.gui.gui_state import get_gui_state
    gui_state = get_gui_state()
    gui_state.set_window(self)
    
    # From tool_executor.py:
    from enigma_engine.gui.gui_state import get_gui_state
    gui_state = get_gui_state()
    gui_state.switch_tab("image")
    gui_state.set_setting("chat_zoom", 12)
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Singleton instance
_gui_state_instance: Optional['GUIStateManager'] = None
_gui_state_lock = threading.RLock()


def get_gui_state() -> 'GUIStateManager':
    """Get the singleton GUIStateManager instance."""
    global _gui_state_instance
    with _gui_state_lock:
        if _gui_state_instance is None:
            _gui_state_instance = GUIStateManager()
        return _gui_state_instance


@dataclass
class HelpTopic:
    """Help content for a specific topic."""
    title: str
    content: str
    related: list[str] = field(default_factory=list)


class GUIStateManager:
    """
    Singleton manager for GUI state and AI interactions.
    
    📖 PURPOSE:
    Provides a clean, thread-safe interface for AI tools to:
    - Navigate the GUI (switch tabs)
    - Read/write user preferences
    - Manage conversations
    - Get contextual help
    
    📐 THREAD SAFETY:
    All methods use locking to ensure safe concurrent access from
    AI generation threads and the GUI thread.
    """
    
    def __init__(self):
        """Initialize the GUI state manager."""
        self._lock = threading.RLock()
        self._window = None  # Reference to EnhancedMainWindow
        self._settings_path: Optional[Path] = None
        self._callbacks: dict[str, list[Callable]] = {}
        self._help_topics = self._init_help_topics()
        
        # Try to determine settings path
        try:
            from ..config import CONFIG
            self._settings_path = Path(CONFIG.get("data_dir", "data")) / "gui_settings.json"
        except Exception:
            self._settings_path = Path("data/gui_settings.json")
    
    # =========================================================================
    # Window Registration
    # =========================================================================
    
    def set_window(self, window) -> None:
        """
        Register the main window instance.
        
        Called by EnhancedMainWindow during initialization.
        """
        with self._lock:
            self._window = window
            logger.debug("GUIStateManager: Window registered")
    
    def has_window(self) -> bool:
        """Check if a window is registered."""
        with self._lock:
            return self._window is not None
    
    # =========================================================================
    # Tab Navigation
    # =========================================================================
    
    def switch_tab(self, tab_name: str) -> dict[str, Any]:
        """
        Switch to a specific tab.
        
        Args:
            tab_name: Name of the tab (e.g., "chat", "image", "settings")
            
        Returns:
            Dict with success status and message
        """
        with self._lock:
            if self._window is None:
                return {"success": False, "error": "GUI not available"}
            
            try:
                result = self._window.ai_switch_tab(tab_name)
                return {"success": True, "result": result, "tab": tab_name}
            except Exception as e:
                logger.error(f"Failed to switch tab: {e}")
                return {"success": False, "error": str(e)}
    
    def get_current_tab(self) -> str:
        """Get the name of the currently active tab."""
        with self._lock:
            if self._window is None:
                return "unknown"
            
            try:
                # Get current tab index and map to name
                idx = self._window.tabs.currentIndex()
                tab_names = [
                    "chat", "train", "history", "scale", "modules",
                    "image", "code", "video", "audio", "search",
                    "avatar", "vision", "personality", "terminal",
                    "files", "examples", "settings"
                ]
                if 0 <= idx < len(tab_names):
                    return tab_names[idx]
                return f"tab_{idx}"
            except Exception:
                return "unknown"
    
    def get_available_tabs(self) -> list[str]:
        """Get list of available tab names."""
        return [
            "chat", "train", "history", "scale", "modules",
            "image", "code", "video", "audio", "search",
            "avatar", "vision", "personality", "terminal",
            "files", "examples", "settings"
        ]
    
    # =========================================================================
    # Settings Management
    # =========================================================================
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a GUI setting value.
        
        Args:
            key: Setting key (e.g., "chat_zoom", "auto_speak")
            default: Default value if not found
            
        Returns:
            Setting value or default
        """
        with self._lock:
            settings = self._load_settings()
            return settings.get(key, default)
    
    def set_setting(self, key: str, value: Any) -> dict[str, Any]:
        """
        Set a GUI setting value.
        
        Args:
            key: Setting key
            value: New value
            
        Returns:
            Dict with success status
        """
        with self._lock:
            # Validate setting
            valid_settings = {
                "chat_zoom": (int, 8, 20),
                "auto_speak": (bool, None, None),
                "learn_while_chatting": (bool, None, None),
                "always_on_top": (bool, None, None),
                "avatar_auto_run": (bool, None, None),
                "avatar_auto_load": (bool, None, None),
                "system_prompt_preset": (str, None, None),
                "custom_system_prompt": (str, None, None),
                "theme": (str, None, None),
                "user_display_name": (str, None, None),
                "mini_chat_always_on_top": (bool, None, None),
            }
            
            if key not in valid_settings:
                return {"success": False, "error": f"Unknown setting: {key}"}
            
            expected_type, min_val, max_val = valid_settings[key]
            
            # Type conversion
            try:
                if expected_type == bool:
                    if isinstance(value, str):
                        value = value.lower() in ("true", "1", "yes", "on")
                    else:
                        value = bool(value)
                elif expected_type == int:
                    value = int(value)
                    if min_val is not None and value < min_val:
                        value = min_val
                    if max_val is not None and value > max_val:
                        value = max_val
                elif expected_type == str:
                    value = str(value)
            except (ValueError, TypeError) as e:
                return {"success": False, "error": f"Invalid value type: {e}"}
            
            # Load, update, save
            settings = self._load_settings()
            old_value = settings.get(key)
            settings[key] = value
            self._save_settings(settings)
            
            # Apply immediately if window available
            self._apply_setting_live(key, value)
            
            # Trigger callbacks
            self._trigger_callbacks("setting_changed", key, value, old_value)
            
            return {
                "success": True, 
                "setting": key, 
                "old_value": old_value,
                "new_value": value
            }
    
    def get_all_settings(self) -> dict[str, Any]:
        """Get all current settings."""
        with self._lock:
            return self._load_settings()
    
    def _load_settings(self) -> dict[str, Any]:
        """Load settings from JSON file."""
        if self._settings_path and self._settings_path.exists():
            try:
                return json.loads(self._settings_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load settings: {e}")
        return {}
    
    def _save_settings(self, settings: dict[str, Any]) -> None:
        """Save settings to JSON file."""
        if self._settings_path:
            try:
                self._settings_path.parent.mkdir(parents=True, exist_ok=True)
                self._settings_path.write_text(
                    json.dumps(settings, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
            except Exception as e:
                logger.error(f"Failed to save settings: {e}")
    
    def _apply_setting_live(self, key: str, value: Any) -> None:
        """Apply a setting change immediately to the running GUI."""
        if self._window is None:
            return
        
        try:
            if key == "chat_zoom":
                if hasattr(self._window, 'chat_display'):
                    font = self._window.chat_display.font()
                    font.setPointSize(value)
                    self._window.chat_display.setFont(font)
            
            elif key == "auto_speak":
                if hasattr(self._window, 'auto_speak_btn'):
                    self._window.auto_speak_btn.setChecked(value)
            
            elif key == "always_on_top":
                from PyQt5.QtCore import Qt
                flags = self._window.windowFlags()
                if value:
                    self._window.setWindowFlags(flags | Qt.WindowStaysOnTopHint)
                else:
                    self._window.setWindowFlags(flags & ~Qt.WindowStaysOnTopHint)
                self._window.show()
            
            elif key == "theme":
                if hasattr(self._window, '_apply_theme'):
                    self._window._apply_theme(value)
                    
        except Exception as e:
            logger.warning(f"Failed to apply setting {key} live: {e}")
    
    # =========================================================================
    # Conversation Management
    # =========================================================================
    
    def manage_conversation(self, action: str, name: Optional[str] = None, 
                           new_name: Optional[str] = None) -> dict[str, Any]:
        """
        Manage chat conversations.
        
        Args:
            action: One of "save", "rename", "delete", "list", "load", "new"
            name: Conversation name (for save/rename/delete/load)
            new_name: New name (for rename)
            
        Returns:
            Dict with action result
        """
        with self._lock:
            try:
                if action == "list":
                    return self._list_conversations()
                elif action == "save":
                    return self._save_conversation(name)
                elif action == "load":
                    return self._load_conversation(name)
                elif action == "new":
                    return self._new_conversation()
                elif action == "rename":
                    return self._rename_conversation(name, new_name)
                elif action == "delete":
                    return self._delete_conversation(name)
                else:
                    return {"success": False, "error": f"Unknown action: {action}"}
            except Exception as e:
                logger.error(f"Conversation management error: {e}")
                return {"success": False, "error": str(e)}
    
    def _list_conversations(self) -> dict[str, Any]:
        """List all saved conversations."""
        try:
            from ..config import CONFIG
            conv_dir = Path(CONFIG.get("data_dir", "data")) / "conversations"
            
            if not conv_dir.exists():
                return {"success": True, "conversations": [], "count": 0}
            
            conversations = []
            for f in conv_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    conversations.append({
                        "name": f.stem,
                        "messages": len(data.get("messages", [])),
                        "model": data.get("model", "unknown"),
                        "modified": f.stat().st_mtime,
                    })
                except Exception:
                    conversations.append({"name": f.stem, "messages": 0})
            
            # Sort by modification time, newest first
            conversations.sort(key=lambda x: x.get("modified", 0), reverse=True)
            
            return {
                "success": True, 
                "conversations": conversations,
                "count": len(conversations)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _save_conversation(self, name: Optional[str]) -> dict[str, Any]:
        """Save current conversation."""
        if self._window is None:
            return {"success": False, "error": "GUI not available"}
        
        try:
            result = self._window.ai_save_session(name)
            return {"success": True, "result": result, "name": name}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _load_conversation(self, name: Optional[str]) -> dict[str, Any]:
        """Load a saved conversation."""
        if not name:
            return {"success": False, "error": "Conversation name required"}
        
        if self._window is None:
            return {"success": False, "error": "GUI not available"}
        
        try:
            # Find the conversation file
            from ..config import CONFIG
            conv_path = Path(CONFIG.get("data_dir", "data")) / "conversations" / f"{name}.json"
            
            if not conv_path.exists():
                return {"success": False, "error": f"Conversation '{name}' not found"}
            
            # Load via window method if available
            if hasattr(self._window, '_load_session'):
                self._window._load_session(name)
                return {"success": True, "name": name}
            
            return {"success": False, "error": "Load method not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _new_conversation(self) -> dict[str, Any]:
        """Start a new conversation."""
        if self._window is None:
            return {"success": False, "error": "GUI not available"}
        
        try:
            if hasattr(self._window, 'ai_clear_chat'):
                self._window.ai_clear_chat()
            return {"success": True, "message": "Started new conversation"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _rename_conversation(self, old_name: Optional[str], 
                            new_name: Optional[str]) -> dict[str, Any]:
        """Rename a conversation."""
        if not old_name or not new_name:
            return {"success": False, "error": "Both old and new names required"}
        
        try:
            from ..config import CONFIG
            conv_dir = Path(CONFIG.get("data_dir", "data")) / "conversations"
            old_path = conv_dir / f"{old_name}.json"
            new_path = conv_dir / f"{new_name}.json"
            
            if not old_path.exists():
                return {"success": False, "error": f"Conversation '{old_name}' not found"}
            
            if new_path.exists():
                return {"success": False, "error": f"Conversation '{new_name}' already exists"}
            
            old_path.rename(new_path)
            return {"success": True, "old_name": old_name, "new_name": new_name}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _delete_conversation(self, name: Optional[str]) -> dict[str, Any]:
        """Delete a conversation."""
        if not name:
            return {"success": False, "error": "Conversation name required"}
        
        try:
            from ..config import CONFIG
            conv_path = Path(CONFIG.get("data_dir", "data")) / "conversations" / f"{name}.json"
            
            if not conv_path.exists():
                return {"success": False, "error": f"Conversation '{name}' not found"}
            
            conv_path.unlink()
            return {"success": True, "deleted": name}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # =========================================================================
    # Help System
    # =========================================================================
    
    def get_help(self, topic: str = "getting_started") -> dict[str, Any]:
        """
        Get help content for a topic.
        
        Args:
            topic: Help topic name
            
        Returns:
            Dict with help content
        """
        topic_lower = topic.lower().replace(" ", "_")
        
        if topic_lower in self._help_topics:
            help_info = self._help_topics[topic_lower]
            return {
                "success": True,
                "topic": topic_lower,
                "title": help_info.title,
                "content": help_info.content,
                "related": help_info.related,
            }
        
        # Return list of available topics if not found
        return {
            "success": False,
            "error": f"Topic '{topic}' not found",
            "available_topics": list(self._help_topics.keys()),
        }
    
    def _init_help_topics(self) -> dict[str, HelpTopic]:
        """Initialize help topic content."""
        return {
            "getting_started": HelpTopic(
                title="Getting Started with Enigma AI Engine",
                content="""Welcome to Enigma AI Engine! Here's how to get started:

1. **Chat Tab** - Talk to your AI naturally. Just type and press Enter.

2. **Train Tab** - Teach your AI new things by training on your data.

3. **Generation Tabs** - Create images, code, videos, and more!

4. **Settings** - Customize your experience (themes, voice, etc.)

Tips:
- Ask me anything - I can help with many tasks
- Say "generate an image of..." to create pictures
- Use "help [topic]" for specific guidance""",
                related=["chat", "training", "tips"],
            ),
            
            "chat": HelpTopic(
                title="Chat Features",
                content="""The Chat tab is your main way to interact with me.

**Features:**
- Natural conversation - just type normally
- Auto-detection - I'll figure out what you want
- Learning mode - I can learn from our conversations
- Voice input/output - speak and listen

**Commands:**
- Type naturally to chat
- "Generate an image of..." for images
- "Write code for..." for programming help
- "Help" for assistance

**Tips:**
- Rate my responses to help me improve
- Use New Chat to start fresh
- Save important conversations""",
                related=["tips", "voice", "training"],
            ),
            
            "training": HelpTopic(
                title="Training Your AI",
                content="""Train your AI to be smarter and more personalized!

**How to Train:**
1. Go to the Train tab
2. Add training data (text files or paste directly)
3. Choose settings (epochs, learning rate)
4. Click Start Training

**Training Data Tips:**
- Use Q&A format: "Q: question\\nA: answer"
- More data = better results (1000+ lines recommended)
- Include diverse examples
- Keep it consistent in style

**Learning Mode:**
Enable "Learn while chatting" in Settings to let me learn from our conversations automatically.""",
                related=["chat", "settings"],
            ),
            
            "image_generation": HelpTopic(
                title="Image Generation",
                content="""Create images from text descriptions!

**How to Use:**
1. Go to the Image tab, or
2. Just say "generate an image of..." in chat

**Tips for Good Images:**
- Be descriptive: "a majestic mountain at sunset with snow"
- Specify style: "digital art", "photorealistic", "watercolor"
- Include details: colors, lighting, mood

**Settings:**
- Width/Height: Image dimensions
- Steps: More = better quality, slower
- CFG Scale: How closely to follow your prompt""",
                related=["code_generation", "tips"],
            ),
            
            "code_generation": HelpTopic(
                title="Code Generation",
                content="""I can help you write code!

**How to Use:**
- "Write a Python function that..."
- "Create a script to..."
- Go to the Code tab for more options

**Supported Languages:**
Python, JavaScript, TypeScript, HTML/CSS, SQL, and more!

**Tips:**
- Be specific about what you need
- Mention the language if it matters
- Ask me to explain the code
- I can help debug too!""",
                related=["chat", "tips"],
            ),
            
            "voice": HelpTopic(
                title="Voice Features",
                content="""Talk to me using your voice!

**Voice Input:**
- Click the microphone button
- Speak clearly
- Wait for transcription

**Voice Output (TTS):**
- Enable "Auto-speak" in Settings
- Or click the speaker icon on messages

**Voice Profiles:**
- Create custom voices in Audio tab
- Adjust pitch, speed, and style
- Save profiles for different moods""",
                related=["settings", "chat"],
            ),
            
            "avatar": HelpTopic(
                title="Avatar System",
                content="""Your AI companion can have a visual presence!

**Features:**
- 2D or 3D avatars
- Lip sync with speech
- Emotion expressions
- Autonomous behavior

**Setup:**
1. Go to Avatar tab
2. Choose an avatar model
3. Enable "Show Avatar"
4. Optional: Enable autonomous mode

**Control:**
- I can control the avatar during chat
- Avatar reacts to conversation mood
- Customize appearance in settings""",
                related=["settings", "voice"],
            ),
            
            "modules": HelpTopic(
                title="Module System",
                content="""Enigma AI Engine is fully modular - load only what you need!

**Why Modules?**
- Save memory by unloading unused features
- Prevent conflicts between similar capabilities
- Scale from Raspberry Pi to datacenter

**How to Use:**
1. Go to Modules tab
2. Toggle modules on/off
3. Check resource usage

**Categories:**
- Core: Model, tokenizer, inference
- Generation: Image, code, video, audio
- Perception: Vision, camera
- Output: Voice, avatar""",
                related=["settings", "tips"],
            ),
            
            "settings": HelpTopic(
                title="Settings & Customization",
                content="""Customize Enigma AI Engine to your preferences!

**Display:**
- Theme: Dark, Light, Midnight, Shadow
- Chat zoom: Adjust text size
- Always on top: Keep window visible

**Behavior:**
- Auto-speak: Voice responses
- Learn while chatting: Continuous learning
- System prompt: Customize AI personality

**Performance:**
- Resource mode: Minimal/Balanced/Performance
- GPU settings: Memory allocation
- Background processing""",
                related=["modules", "tips"],
            ),
            
            "keyboard_shortcuts": HelpTopic(
                title="Keyboard Shortcuts",
                content="""Speed up your workflow with shortcuts!

**Global:**
- Ctrl+Enter: Send message
- Ctrl+N: New chat
- Ctrl+S: Save conversation
- Escape: Stop generation

**Navigation:**
- Ctrl+1-9: Switch to tab 1-9
- Ctrl+Tab: Next tab
- Ctrl+Shift+Tab: Previous tab

**Chat:**
- Up/Down: Browse history
- Ctrl+L: Clear chat
- Ctrl+C: Copy selected text""",
                related=["tips", "chat"],
            ),
            
            "tips": HelpTopic(
                title="Tips & Tricks",
                content="""Get the most out of Enigma AI Engine!

**Productivity:**
- Use natural language - I understand context
- Rate responses to help me improve
- Save useful conversations for reference

**Performance:**
- Unload unused modules to save memory
- Use smaller models for faster responses
- Enable GPU acceleration if available

**Quality:**
- Be specific in prompts
- Provide examples when asking for formats
- Use system prompts for consistent behavior

**Hidden Features:**
- Drag & drop images into chat
- Right-click for context menus
- Double-click messages to edit""",
                related=["keyboard_shortcuts", "settings"],
            ),
        }
    
    # =========================================================================
    # Hardware Optimization
    # =========================================================================
    
    def optimize_for_hardware(self, mode: str = "auto") -> dict[str, Any]:
        """
        Optimize settings based on hardware capabilities.
        
        Args:
            mode: Optimization mode (auto, performance, balanced, power_saver, gaming)
            
        Returns:
            Dict with applied optimizations
        """
        with self._lock:
            try:
                # Detect hardware
                hardware = self._detect_hardware()
                
                # Determine optimal settings
                if mode == "auto":
                    if hardware.get("gpu_vram_gb", 0) >= 8:
                        mode = "performance"
                    elif hardware.get("gpu_available"):
                        mode = "balanced"
                    elif hardware.get("ram_gb", 0) >= 16:
                        mode = "balanced"
                    else:
                        mode = "power_saver"
                
                # Apply mode settings
                optimizations = self._get_mode_settings(mode, hardware)
                
                for key, value in optimizations["settings"].items():
                    self.set_setting(key, value)
                
                return {
                    "success": True,
                    "mode": mode,
                    "hardware": hardware,
                    "applied": optimizations["settings"],
                    "recommendations": optimizations.get("recommendations", []),
                }
                
            except Exception as e:
                logger.error(f"Hardware optimization failed: {e}")
                return {"success": False, "error": str(e)}
    
    def _detect_hardware(self) -> dict[str, Any]:
        """Detect available hardware."""
        hardware = {
            "gpu_available": False,
            "gpu_name": None,
            "gpu_vram_gb": 0,
            "ram_gb": 0,
            "cpu_cores": 1,
        }
        
        try:
            import psutil
            hardware["ram_gb"] = psutil.virtual_memory().total / (1024**3)
            hardware["cpu_cores"] = psutil.cpu_count()
        except ImportError:
            pass  # Intentionally silent
        
        try:
            import torch
            if torch.cuda.is_available():
                hardware["gpu_available"] = True
                hardware["gpu_name"] = torch.cuda.get_device_name(0)
                hardware["gpu_vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass  # Intentionally silent
        
        return hardware
    
    def _get_mode_settings(self, mode: str, hardware: dict[str, Any]) -> dict[str, Any]:
        """Get settings for optimization mode."""
        modes = {
            "performance": {
                "settings": {
                    "learn_while_chatting": True,
                    "avatar_auto_run": True,
                },
                "recommendations": [
                    "Using maximum quality settings",
                    "All features enabled",
                    "Consider loading larger models",
                ],
            },
            "balanced": {
                "settings": {
                    "learn_while_chatting": True,
                    "avatar_auto_run": False,
                },
                "recommendations": [
                    "Good balance of quality and performance",
                    "Avatar disabled to save resources",
                ],
            },
            "power_saver": {
                "settings": {
                    "learn_while_chatting": False,
                    "avatar_auto_run": False,
                    "auto_speak": False,
                },
                "recommendations": [
                    "Minimal resource usage",
                    "Consider using smaller models",
                    "Background features disabled",
                ],
            },
            "gaming": {
                "settings": {
                    "learn_while_chatting": False,
                    "avatar_auto_run": False,
                    "auto_speak": False,
                },
                "recommendations": [
                    "Reduced resource usage for gaming",
                    "Consider minimizing to system tray",
                    "Use quick-chat overlay instead of main window",
                ],
            },
        }
        
        return modes.get(mode, modes["balanced"])
    
    # =========================================================================
    # Callbacks
    # =========================================================================
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for an event."""
        with self._lock:
            if event not in self._callbacks:
                self._callbacks[event] = []
            self._callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, *args, **kwargs) -> None:
        """Trigger all callbacks for an event."""
        callbacks = self._callbacks.get(event, [])
        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Callback error for {event}: {e}")
