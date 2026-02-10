"""
Session Management - Auto-save drafts, export, config editing, focus mode.

Surface-level user experience features:
- Auto-save message drafts on close
- Export conversations to Obsidian/Notion/Markdown
- Visual config editor
- Focus mode for distraction-free chat

Part of the Enigma AI Engine GUI features.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# AUTO-SAVE DRAFTS
# =============================================================================

@dataclass
class MessageDraft:
    """An unsent message draft."""
    conversation_id: str
    text: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cursor_position: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "text": self.text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "cursor_position": self.cursor_position
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'MessageDraft':
        return cls(
            conversation_id=data.get("conversation_id", ""),
            text=data.get("text", ""),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            cursor_position=data.get("cursor_position", 0)
        )


class DraftManager:
    """
    Manage message drafts with auto-save.
    
    Features:
    - Auto-save drafts every N seconds
    - Persist on window close
    - Restore drafts on app start
    - Per-conversation drafts
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        auto_save_interval: int = 5
    ):
        """
        Initialize draft manager.
        
        Args:
            storage_path: Path to store drafts
            auto_save_interval: Auto-save interval in seconds
        """
        self.storage_path = storage_path
        self.auto_save_interval = auto_save_interval
        self.drafts: dict[str, MessageDraft] = {}
        self._last_save: float = 0
        self._dirty: bool = False
        
        if storage_path and storage_path.exists():
            self._load()
    
    def _load(self):
        """Load drafts from storage."""
        if not self.storage_path:
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            for draft_data in data.get("drafts", []):
                draft = MessageDraft.from_dict(draft_data)
                self.drafts[draft.conversation_id] = draft
        except Exception as e:
            logger.warning(f"Failed to load drafts: {e}")
    
    def _save(self):
        """Save drafts to storage."""
        if not self.storage_path:
            return
        try:
            data = {"drafts": [d.to_dict() for d in self.drafts.values()]}
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._last_save = time.time()
            self._dirty = False
        except Exception as e:
            logger.error(f"Failed to save drafts: {e}")
    
    def update_draft(
        self,
        conversation_id: str,
        text: str,
        cursor_position: int = 0
    ):
        """
        Update or create a draft.
        
        Args:
            conversation_id: ID of the conversation
            text: Draft message text
            cursor_position: Cursor position in text
        """
        if conversation_id in self.drafts:
            self.drafts[conversation_id].text = text
            self.drafts[conversation_id].updated_at = time.time()
            self.drafts[conversation_id].cursor_position = cursor_position
        else:
            self.drafts[conversation_id] = MessageDraft(
                conversation_id=conversation_id,
                text=text,
                cursor_position=cursor_position
            )
        
        self._dirty = True
        
        # Auto-save if interval passed
        if time.time() - self._last_save >= self.auto_save_interval:
            self._save()
    
    def get_draft(self, conversation_id: str) -> Optional[MessageDraft]:
        """Get draft for a conversation."""
        return self.drafts.get(conversation_id)
    
    def delete_draft(self, conversation_id: str):
        """Delete a draft after sending."""
        if conversation_id in self.drafts:
            del self.drafts[conversation_id]
            self._dirty = True
            self._save()
    
    def get_all_drafts(self) -> list[MessageDraft]:
        """Get all drafts."""
        return list(self.drafts.values())
    
    def save_now(self):
        """Force immediate save."""
        if self._dirty:
            self._save()
    
    def clear_old_drafts(self, max_age_days: int = 7):
        """Clear drafts older than max_age_days."""
        cutoff = time.time() - (max_age_days * 24 * 60 * 60)
        to_delete = [
            cid for cid, draft in self.drafts.items()
            if draft.updated_at < cutoff
        ]
        for cid in to_delete:
            del self.drafts[cid]
        if to_delete:
            self._save()


# =============================================================================
# EXPORT FUNCTIONALITY
# =============================================================================

class ExportFormat(Enum):
    """Supported export formats."""
    MARKDOWN = auto()
    JSON = auto()
    HTML = auto()
    OBSIDIAN = auto()
    NOTION = auto()
    TEXT = auto()


@dataclass
class ExportConfig:
    """Configuration for export."""
    format: ExportFormat = ExportFormat.MARKDOWN
    include_metadata: bool = True
    include_timestamps: bool = True
    include_system_messages: bool = False
    frontmatter: bool = True  # For Obsidian
    notion_database_id: Optional[str] = None


class ConversationExporter:
    """
    Export conversations to various formats.
    
    Features:
    - Markdown export
    - Obsidian-compatible export with frontmatter
    - Notion export (API integration)
    - JSON backup
    - HTML export
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize exporter.
        
        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
    
    def export(
        self,
        conversation_name: str,
        messages: list[dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Export a conversation.
        
        Args:
            conversation_name: Name of the conversation
            messages: List of messages
            output_path: Optional output file path
            
        Returns:
            Exported content as string
        """
        if self.config.format == ExportFormat.MARKDOWN:
            content = self._export_markdown(conversation_name, messages)
        elif self.config.format == ExportFormat.OBSIDIAN:
            content = self._export_obsidian(conversation_name, messages)
        elif self.config.format == ExportFormat.JSON:
            content = self._export_json(conversation_name, messages)
        elif self.config.format == ExportFormat.HTML:
            content = self._export_html(conversation_name, messages)
        elif self.config.format == ExportFormat.TEXT:
            content = self._export_text(conversation_name, messages)
        else:
            content = self._export_markdown(conversation_name, messages)
        
        if output_path:
            output_path.write_text(content, encoding="utf-8")
        
        return content
    
    def _export_markdown(
        self,
        name: str,
        messages: list[dict[str, Any]]
    ) -> str:
        """Export to Markdown format."""
        lines = [f"# {name}\n"]
        
        if self.config.include_metadata:
            lines.append(f"*Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        lines.append("---\n")
        
        for msg in messages:
            role = msg.get("role", "unknown")
            text = msg.get("text", msg.get("content", ""))
            ts = msg.get("ts", msg.get("timestamp"))
            
            if role == "system" and not self.config.include_system_messages:
                continue
            
            role_label = {"user": "You", "ai": "AI", "assistant": "AI", "system": "System"}.get(role, role)
            
            if self.config.include_timestamps and ts:
                ts_str = time.strftime('%H:%M', time.localtime(ts))
                lines.append(f"**{role_label}** *({ts_str})*:\n")
            else:
                lines.append(f"**{role_label}**:\n")
            
            lines.append(f"{text}\n\n")
        
        return "\n".join(lines)
    
    def _export_obsidian(
        self,
        name: str,
        messages: list[dict[str, Any]]
    ) -> str:
        """Export to Obsidian-compatible Markdown with frontmatter."""
        lines = []
        
        # Frontmatter
        if self.config.frontmatter:
            lines.append("---")
            lines.append(f"title: {name}")
            lines.append(f"date: {time.strftime('%Y-%m-%d')}")
            lines.append("type: conversation")
            lines.append("tags: [Enigma AI Engine, chat]")
            lines.append(f"message_count: {len(messages)}")
            lines.append("---\n")
        
        lines.append(f"# {name}\n")
        
        # Messages as callouts
        for msg in messages:
            role = msg.get("role", "unknown")
            text = msg.get("text", msg.get("content", ""))
            ts = msg.get("ts", msg.get("timestamp"))
            
            if role == "system" and not self.config.include_system_messages:
                continue
            
            # Use Obsidian callouts
            if role in ["user", "human"]:
                callout = "> [!question] You"
            elif role in ["ai", "assistant"]:
                callout = "> [!info] AI"
            else:
                callout = f"> [!note] {role}"
            
            if self.config.include_timestamps and ts:
                ts_str = time.strftime('%H:%M', time.localtime(ts))
                callout += f" ({ts_str})"
            
            lines.append(callout)
            for line in text.split("\n"):
                lines.append(f"> {line}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _export_json(
        self,
        name: str,
        messages: list[dict[str, Any]]
    ) -> str:
        """Export to JSON format."""
        data = {
            "name": name,
            "exported_at": time.time(),
            "exported_at_str": time.strftime('%Y-%m-%d %H:%M:%S'),
            "message_count": len(messages),
            "messages": messages
        }
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def _export_html(
        self,
        name: str,
        messages: list[dict[str, Any]]
    ) -> str:
        """Export to HTML format."""
        html = [
            "<!DOCTYPE html>",
            "<html><head>",
            f"<title>{name}</title>",
            "<style>",
            "body { font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }",
            ".message { margin: 10px 0; padding: 10px; border-radius: 8px; }",
            ".user { background: #e3f2fd; }",
            ".ai { background: #f5f5f5; }",
            ".role { font-weight: bold; margin-bottom: 5px; }",
            ".time { color: #666; font-size: 0.8em; }",
            "pre { background: #1e1e1e; color: #fff; padding: 10px; border-radius: 4px; overflow-x: auto; }",
            "</style>",
            "</head><body>",
            f"<h1>{name}</h1>"
        ]
        
        for msg in messages:
            role = msg.get("role", "unknown")
            text = msg.get("text", msg.get("content", ""))
            ts = msg.get("ts", msg.get("timestamp"))
            
            if role == "system" and not self.config.include_system_messages:
                continue
            
            role_class = "user" if role in ["user", "human"] else "ai"
            role_label = {"user": "You", "ai": "AI", "assistant": "AI"}.get(role, role)
            
            html.append(f'<div class="message {role_class}">')
            html.append(f'<div class="role">{role_label}')
            if self.config.include_timestamps and ts:
                ts_str = time.strftime('%H:%M', time.localtime(ts))
                html.append(f' <span class="time">({ts_str})</span>')
            html.append('</div>')
            
            # Escape HTML and preserve code blocks
            escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            escaped = escaped.replace('\n', '<br>')
            html.append(f'<div class="content">{escaped}</div>')
            html.append('</div>')
        
        html.append("</body></html>")
        return "\n".join(html)
    
    def _export_text(
        self,
        name: str,
        messages: list[dict[str, Any]]
    ) -> str:
        """Export to plain text format."""
        lines = [f"{name}", "=" * len(name), ""]
        
        for msg in messages:
            role = msg.get("role", "unknown")
            text = msg.get("text", msg.get("content", ""))
            
            if role == "system" and not self.config.include_system_messages:
                continue
            
            role_label = {"user": "You", "ai": "AI", "assistant": "AI"}.get(role, role)
            lines.append(f"[{role_label}]")
            lines.append(text)
            lines.append("")
        
        return "\n".join(lines)
    
    def export_to_obsidian_vault(
        self,
        vault_path: Path,
        conversation_name: str,
        messages: list[dict[str, Any]],
        folder: str = "Enigma AI Engine"
    ) -> Path:
        """
        Export directly to Obsidian vault.
        
        Args:
            vault_path: Path to Obsidian vault
            conversation_name: Name of conversation
            messages: Messages to export
            folder: Subfolder in vault
            
        Returns:
            Path to created file
        """
        # Sanitize name for filename
        safe_name = "".join(c for c in conversation_name if c.isalnum() or c in " -_").strip()
        safe_name = safe_name.replace(" ", "_")
        
        # Create folder if needed
        output_dir = vault_path / folder
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file
        output_path = output_dir / f"{safe_name}.md"
        
        old_format = self.config.format
        self.config.format = ExportFormat.OBSIDIAN
        self.export(conversation_name, messages, output_path)
        self.config.format = old_format
        
        return output_path


# =============================================================================
# CONFIG EDITOR
# =============================================================================

class ConfigValueType(Enum):
    """Types of config values."""
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()
    PATH = auto()
    SELECT = auto()
    COLOR = auto()


@dataclass
class ConfigField:
    """A configuration field definition."""
    key: str
    label: str
    value_type: ConfigValueType
    description: str = ""
    default: Any = None
    options: Optional[list[str]] = None  # For SELECT type
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    category: str = "General"
    
    def validate(self, value: Any) -> bool:
        """Validate a value for this field."""
        if self.value_type == ConfigValueType.NUMBER:
            if not isinstance(value, (int, float)):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
        elif self.value_type == ConfigValueType.SELECT:
            if self.options and value not in self.options:
                return False
        elif self.value_type == ConfigValueType.PATH:
            # Path validation - just check it's a string
            if not isinstance(value, str):
                return False
        
        return True


class ConfigEditor:
    """
    Visual configuration editor.
    
    Features:
    - Define editable config fields
    - Field validation
    - Category organization
    - Save/load config
    """
    
    # Common Enigma AI Engine config fields
    COMMON_FIELDS = [
        ConfigField("model_size", "Model Size", ConfigValueType.SELECT,
                   "Size of the AI model", "small",
                   ["nano", "micro", "tiny", "small", "medium", "large", "xl"],
                   category="Model"),
        ConfigField("temperature", "Temperature", ConfigValueType.NUMBER,
                   "Controls response randomness (0.0-2.0)", 0.7,
                   min_value=0.0, max_value=2.0, category="Generation"),
        ConfigField("max_tokens", "Max Tokens", ConfigValueType.NUMBER,
                   "Maximum response length", 512,
                   min_value=1, max_value=4096, category="Generation"),
        ConfigField("top_p", "Top P", ConfigValueType.NUMBER,
                   "Nucleus sampling threshold", 0.95,
                   min_value=0.0, max_value=1.0, category="Generation"),
        ConfigField("data_dir", "Data Directory", ConfigValueType.PATH,
                   "Directory for storing data", "data", category="Paths"),
        ConfigField("models_dir", "Models Directory", ConfigValueType.PATH,
                   "Directory for model files", "models", category="Paths"),
        ConfigField("theme", "Theme", ConfigValueType.SELECT,
                   "UI theme", "dark",
                   ["light", "dark", "system"], category="Appearance"),
        ConfigField("auto_save", "Auto-save", ConfigValueType.BOOLEAN,
                   "Automatically save conversations", True, category="General"),
        ConfigField("voice_enabled", "Voice Enabled", ConfigValueType.BOOLEAN,
                   "Enable voice input/output", False, category="Voice"),
    ]
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config editor.
        
        Args:
            config_path: Path to config file
        """
        self.config_path = config_path
        self.fields: dict[str, ConfigField] = {}
        self.values: dict[str, Any] = {}
        
        # Register common fields
        for field in self.COMMON_FIELDS:
            self.register_field(field)
        
        if config_path and config_path.exists():
            self._load()
    
    def register_field(self, field: ConfigField):
        """Register a config field."""
        self.fields[field.key] = field
        if field.key not in self.values:
            self.values[field.key] = field.default
    
    def _load(self):
        """Load config from file."""
        if not self.config_path:
            return
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            for key, value in data.items():
                if key in self.fields:
                    if self.fields[key].validate(value):
                        self.values[key] = value
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    
    def _save(self):
        """Save config to file."""
        if not self.config_path:
            return
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(
                json.dumps(self.values, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value."""
        return self.values.get(key, default)
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set config value.
        
        Args:
            key: Config key
            value: New value
            
        Returns:
            True if value was valid and set
        """
        if key not in self.fields:
            self.values[key] = value
            return True
        
        if self.fields[key].validate(value):
            self.values[key] = value
            self._save()
            return True
        
        return False
    
    def get_fields_by_category(self) -> dict[str, list[ConfigField]]:
        """Get fields organized by category."""
        categories: dict[str, list[ConfigField]] = {}
        for field in self.fields.values():
            if field.category not in categories:
                categories[field.category] = []
            categories[field.category].append(field)
        return categories
    
    def get_all_values(self) -> dict[str, Any]:
        """Get all config values."""
        return dict(self.values)
    
    def reset_to_defaults(self):
        """Reset all values to defaults."""
        for field in self.fields.values():
            self.values[field.key] = field.default
        self._save()
    
    def export_config(self) -> str:
        """Export config as JSON string."""
        return json.dumps(self.values, indent=2)
    
    def import_config(self, config_str: str) -> bool:
        """
        Import config from JSON string.
        
        Args:
            config_str: JSON config string
            
        Returns:
            True if import successful
        """
        try:
            data = json.loads(config_str)
            for key, value in data.items():
                self.set(key, value)
            return True
        except Exception as e:
            logger.error(f"Failed to import config: {e}")
            return False


# =============================================================================
# FOCUS MODE
# =============================================================================

class FocusMode:
    """
    Focus mode for distraction-free chat.
    
    Features:
    - Hide UI elements
    - Fullscreen option
    - Minimal interface
    - Auto-activate on typing
    """
    
    def __init__(self):
        """Initialize focus mode."""
        self.enabled: bool = False
        self.hide_sidebar: bool = True
        self.hide_toolbar: bool = True
        self.hide_status_bar: bool = True
        self.dim_background: bool = True
        self.fullscreen: bool = False
        self.auto_activate_typing: bool = False
        self._callbacks: list[Callable[[bool], None]] = []
    
    def enable(self, fullscreen: bool = False):
        """Enable focus mode."""
        self.enabled = True
        self.fullscreen = fullscreen
        self._notify_callbacks()
    
    def disable(self):
        """Disable focus mode."""
        self.enabled = False
        self.fullscreen = False
        self._notify_callbacks()
    
    def toggle(self) -> bool:
        """Toggle focus mode."""
        if self.enabled:
            self.disable()
        else:
            self.enable()
        return self.enabled
    
    def on_change(self, callback: Callable[[bool], None]):
        """Register callback for mode changes."""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self):
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(self.enabled)
            except Exception as e:
                logger.warning(f"Focus mode callback error: {e}")
    
    def get_hidden_elements(self) -> list[str]:
        """Get list of UI elements to hide."""
        hidden = []
        if self.enabled:
            if self.hide_sidebar:
                hidden.append("sidebar")
            if self.hide_toolbar:
                hidden.append("toolbar")
            if self.hide_status_bar:
                hidden.append("status_bar")
        return hidden
    
    def get_state(self) -> dict[str, Any]:
        """Get current focus mode state."""
        return {
            "enabled": self.enabled,
            "hide_sidebar": self.hide_sidebar,
            "hide_toolbar": self.hide_toolbar,
            "hide_status_bar": self.hide_status_bar,
            "dim_background": self.dim_background,
            "fullscreen": self.fullscreen,
            "auto_activate_typing": self.auto_activate_typing
        }
    
    def restore_state(self, state: dict[str, Any]):
        """Restore focus mode state."""
        self.hide_sidebar = state.get("hide_sidebar", True)
        self.hide_toolbar = state.get("hide_toolbar", True)
        self.hide_status_bar = state.get("hide_status_bar", True)
        self.dim_background = state.get("dim_background", True)
        self.auto_activate_typing = state.get("auto_activate_typing", False)
        # Don't restore enabled/fullscreen - start fresh


# =============================================================================
# COMBINED MANAGER
# =============================================================================

class SessionManager:
    """
    Combined manager for session-related features.
    
    Combines:
    - Draft management
    - Export functionality
    - Config editing
    - Focus mode
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize session manager.
        
        Args:
            data_dir: Directory for storing data
        """
        draft_path = data_dir / "drafts.json" if data_dir else None
        config_path = data_dir / "user_config.json" if data_dir else None
        
        self.drafts = DraftManager(draft_path)
        self.exporter = ConversationExporter()
        self.config = ConfigEditor(config_path)
        self.focus = FocusMode()
    
    def save_on_close(self):
        """Save all pending data before closing."""
        self.drafts.save_now()
    
    def get_quick_settings(self) -> dict[str, Any]:
        """Get commonly accessed settings."""
        return {
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 512),
            "theme": self.config.get("theme", "dark"),
            "auto_save": self.config.get("auto_save", True),
            "focus_mode": self.focus.enabled
        }


# Singleton instance
_session_manager: Optional[SessionManager] = None


def get_session_manager(data_dir: Optional[Path] = None) -> SessionManager:
    """Get or create the session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(data_dir)
    return _session_manager
