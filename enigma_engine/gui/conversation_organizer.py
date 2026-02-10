"""
Conversation Organization - Folders, markdown preview, and generation presets.

Surface-level features:
- Conversation folders for organizing chats
- Markdown preview toggle (rendered vs raw)
- Temperature presets (creative/balanced/precise)
- Typing indicator support

Part of the Enigma AI Engine GUI features.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONVERSATION FOLDERS
# =============================================================================

@dataclass
class ConversationFolder:
    """A folder for organizing conversations."""
    id: str
    name: str
    color: str = "#4488ff"
    icon: str = "folder"  # Icon name or emoji
    parent_id: Optional[str] = None  # For nested folders
    created_at: float = field(default_factory=time.time)
    sort_order: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "icon": self.icon,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "sort_order": self.sort_order
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ConversationFolder':
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            color=data.get("color", "#4488ff"),
            icon=data.get("icon", "folder"),
            parent_id=data.get("parent_id"),
            created_at=data.get("created_at", time.time()),
            sort_order=data.get("sort_order", 0)
        )


class FolderManager:
    """
    Manage conversation folders and organization.
    
    Features:
    - Create/rename/delete folders
    - Nested folder structure
    - Move conversations between folders  
    - Folder colors and icons
    - Smart folder suggestions
    """
    
    DEFAULT_FOLDERS = [
        ConversationFolder("inbox", "Inbox", "#808080", "inbox", sort_order=0),
        ConversationFolder("work", "Work", "#4488ff", "briefcase", sort_order=1),
        ConversationFolder("personal", "Personal", "#44ff44", "user", sort_order=2),
        ConversationFolder("archived", "Archived", "#888888", "archive", sort_order=99)
    ]
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize folder manager.
        
        Args:
            storage_path: Path to store folder data
        """
        self.storage_path = storage_path
        self.folders: dict[str, ConversationFolder] = {}
        self._conversation_folders: dict[str, str] = {}  # conv_name -> folder_id
        
        if storage_path and storage_path.exists():
            self._load()
        else:
            for folder in self.DEFAULT_FOLDERS:
                self.folders[folder.id] = folder
    
    def _load(self):
        """Load folders from storage."""
        if not self.storage_path:
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            for folder_data in data.get("folders", []):
                folder = ConversationFolder.from_dict(folder_data)
                self.folders[folder.id] = folder
            self._conversation_folders = data.get("conversation_folders", {})
        except Exception as e:
            logger.warning(f"Failed to load folders: {e}")
            for folder in self.DEFAULT_FOLDERS:
                self.folders[folder.id] = folder
    
    def _save(self):
        """Save folders to storage."""
        if not self.storage_path:
            return
        try:
            data = {
                "folders": [f.to_dict() for f in self.folders.values()],
                "conversation_folders": self._conversation_folders
            }
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save folders: {e}")
    
    def create_folder(
        self,
        name: str,
        color: str = "#4488ff",
        icon: str = "folder",
        parent_id: Optional[str] = None
    ) -> ConversationFolder:
        """Create a new folder."""
        folder_id = name.lower().replace(" ", "_")
        base_id = folder_id
        counter = 1
        while folder_id in self.folders:
            folder_id = f"{base_id}_{counter}"
            counter += 1
        
        folder = ConversationFolder(
            id=folder_id,
            name=name,
            color=color,
            icon=icon,
            parent_id=parent_id,
            sort_order=len(self.folders)
        )
        self.folders[folder_id] = folder
        self._save()
        return folder
    
    def rename_folder(self, folder_id: str, new_name: str):
        """Rename a folder."""
        if folder_id in self.folders:
            self.folders[folder_id].name = new_name
            self._save()
    
    def delete_folder(self, folder_id: str, move_to: str = "inbox"):
        """
        Delete a folder, moving conversations to another folder.
        
        Args:
            folder_id: Folder to delete
            move_to: Folder to move conversations to
        """
        if folder_id not in self.folders:
            return
        if folder_id == move_to:
            move_to = "inbox"
        
        # Move conversations
        for conv_name, conv_folder in list(self._conversation_folders.items()):
            if conv_folder == folder_id:
                self._conversation_folders[conv_name] = move_to
        
        # Delete folder
        del self.folders[folder_id]
        self._save()
    
    def move_conversation(self, conversation_name: str, folder_id: str):
        """Move a conversation to a folder."""
        if folder_id not in self.folders:
            return
        self._conversation_folders[conversation_name] = folder_id
        self._save()
    
    def get_folder(self, folder_id: str) -> Optional[ConversationFolder]:
        """Get a folder by ID."""
        return self.folders.get(folder_id)
    
    def get_conversation_folder(self, conversation_name: str) -> Optional[ConversationFolder]:
        """Get the folder containing a conversation."""
        folder_id = self._conversation_folders.get(conversation_name, "inbox")
        return self.folders.get(folder_id)
    
    def get_conversations_in_folder(
        self,
        folder_id: str,
        all_conversations: list[str]
    ) -> list[str]:
        """Get all conversations in a specific folder."""
        result = []
        for conv in all_conversations:
            conv_folder = self._conversation_folders.get(conv, "inbox")
            if conv_folder == folder_id:
                result.append(conv)
        return result
    
    def get_all_folders(self, include_nested: bool = True) -> list[ConversationFolder]:
        """
        Get all folders.
        
        Args:
            include_nested: If True, return flat list. If False, only top-level.
        """
        folders = list(self.folders.values())
        if not include_nested:
            folders = [f for f in folders if f.parent_id is None]
        return sorted(folders, key=lambda f: f.sort_order)
    
    def get_folder_tree(self) -> dict[str, Any]:
        """Get folders as a nested tree structure."""
        tree: dict[str, Any] = {}
        
        # Build tree
        for folder in self.folders.values():
            if folder.parent_id is None:
                tree[folder.id] = {
                    "folder": folder,
                    "children": {}
                }
        
        # Add children
        for folder in self.folders.values():
            if folder.parent_id and folder.parent_id in tree:
                tree[folder.parent_id]["children"][folder.id] = {
                    "folder": folder,
                    "children": {}
                }
        
        return tree
    
    def suggest_folder(
        self,
        conversation_name: str,
        messages: Optional[list[dict[str, str]]] = None
    ) -> Optional[str]:
        """
        Suggest a folder for a conversation based on content.
        
        Args:
            conversation_name: Name of the conversation
            messages: Optional message history for analysis
            
        Returns:
            Suggested folder ID or None
        """
        name_lower = conversation_name.lower()
        
        # Name-based suggestions
        if any(kw in name_lower for kw in ["work", "meeting", "project", "task"]):
            return "work"
        if any(kw in name_lower for kw in ["personal", "home", "family", "friend"]):
            return "personal"
        
        # Message content analysis
        if messages:
            combined = " ".join(m.get("content", "") for m in messages[:10]).lower()
            
            if any(kw in combined for kw in ["deadline", "client", "report", "meeting"]):
                return "work"
            if any(kw in combined for kw in ["recipe", "movie", "weekend", "vacation"]):
                return "personal"
        
        return None


# =============================================================================
# MARKDOWN PREVIEW
# =============================================================================

class MarkdownMode(Enum):
    """Markdown display modes."""
    RENDERED = auto()  # Show rendered markdown
    RAW = auto()       # Show raw markdown text
    SPLIT = auto()     # Show both side by side


@dataclass
class MarkdownPreviewConfig:
    """Configuration for markdown preview."""
    mode: MarkdownMode = MarkdownMode.RENDERED
    syntax_highlighting: bool = True
    link_detection: bool = True
    code_block_theme: str = "default"  # "default", "dark", "monokai", etc.
    max_code_height: int = 400  # Max height for code blocks in pixels


class MarkdownPreview:
    """
    Markdown preview with toggle support.
    
    Features:
    - Toggle between rendered and raw views
    - Syntax highlighting for code blocks
    - Link detection and clickable URLs
    - Code copy functionality
    """
    
    # Simple regex patterns for markdown elements
    CODE_BLOCK_PATTERN = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    INLINE_CODE_PATTERN = re.compile(r'`([^`]+)`')
    BOLD_PATTERN = re.compile(r'\*\*(.+?)\*\*')
    ITALIC_PATTERN = re.compile(r'\*(.+?)\*')
    LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
    
    def __init__(self, config: Optional[MarkdownPreviewConfig] = None):
        """
        Initialize markdown preview.
        
        Args:
            config: Preview configuration
        """
        self.config = config or MarkdownPreviewConfig()
    
    def set_mode(self, mode: MarkdownMode):
        """Set display mode."""
        self.config.mode = mode
    
    def toggle_mode(self) -> MarkdownMode:
        """Toggle between rendered and raw mode."""
        if self.config.mode == MarkdownMode.RENDERED:
            self.config.mode = MarkdownMode.RAW
        else:
            self.config.mode = MarkdownMode.RENDERED
        return self.config.mode
    
    def render(self, text: str) -> str:
        """
        Render markdown to HTML.
        
        Args:
            text: Raw markdown text
            
        Returns:
            HTML string
        """
        if self.config.mode == MarkdownMode.RAW:
            return f"<pre>{self._escape_html(text)}</pre>"
        
        html = text
        
        # Code blocks first (to avoid processing code content)
        code_blocks: list[str] = []
        
        def save_code_block(match):
            lang = match.group(1) or "text"
            code = match.group(2)
            idx = len(code_blocks)
            code_blocks.append((lang, code))
            return f"__CODE_BLOCK_{idx}__"
        
        html = self.CODE_BLOCK_PATTERN.sub(save_code_block, html)
        
        # Inline code
        html = self.INLINE_CODE_PATTERN.sub(r'<code>\1</code>', html)
        
        # Headers
        def replace_header(match):
            level = len(match.group(1))
            content = match.group(2)
            return f'<h{level}>{content}</h{level}>'
        
        html = self.HEADER_PATTERN.sub(replace_header, html)
        
        # Bold and italic
        html = self.BOLD_PATTERN.sub(r'<strong>\1</strong>', html)
        html = self.ITALIC_PATTERN.sub(r'<em>\1</em>', html)
        
        # Links
        html = self.LINK_PATTERN.sub(r'<a href="\2">\1</a>', html)
        
        # URL detection
        if self.config.link_detection:
            def linkify(match):
                url = match.group(0)
                return f'<a href="{url}">{url}</a>'
            
            html = self.URL_PATTERN.sub(linkify, html)
        
        # Restore code blocks
        for idx, (lang, code) in enumerate(code_blocks):
            code_html = self._render_code_block(lang, code)
            html = html.replace(f"__CODE_BLOCK_{idx}__", code_html)
        
        # Line breaks
        html = html.replace('\n\n', '</p><p>')
        html = f'<p>{html}</p>'
        
        return html
    
    def _render_code_block(self, lang: str, code: str) -> str:
        """Render a code block with optional syntax highlighting."""
        escaped = self._escape_html(code.strip())
        
        header = f'<div class="code-header">{lang}</div>' if lang != "text" else ""
        
        return f'''
        <div class="code-block" data-language="{lang}">
            {header}
            <pre><code class="language-{lang}">{escaped}</code></pre>
            <button class="copy-button" onclick="copyCode(this)">Copy</button>
        </div>
        '''
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;'))
    
    def extract_code_blocks(self, text: str) -> list[dict[str, str]]:
        """Extract all code blocks from markdown."""
        blocks = []
        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            blocks.append({
                "language": match.group(1) or "text",
                "code": match.group(2).strip()
            })
        return blocks
    
    def get_preview_stats(self, text: str) -> dict[str, Any]:
        """Get statistics about the markdown content."""
        code_blocks = self.CODE_BLOCK_PATTERN.findall(text)
        links = self.LINK_PATTERN.findall(text)
        headers = self.HEADER_PATTERN.findall(text)
        
        return {
            "char_count": len(text),
            "word_count": len(text.split()),
            "code_blocks": len(code_blocks),
            "links": len(links),
            "headers": len(headers)
        }


# =============================================================================
# TEMPERATURE PRESETS
# =============================================================================

class TemperaturePreset(Enum):
    """Temperature presets for generation."""
    PRECISE = "precise"      # Low temperature, deterministic
    BALANCED = "balanced"    # Medium temperature, default
    CREATIVE = "creative"    # High temperature, varied


@dataclass
class GenerationPreset:
    """Complete generation preset with all parameters."""
    name: str
    temperature: float
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    description: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "description": self.description
        }


class TemperaturePresetManager:
    """
    Manage temperature and generation presets.
    
    Features:
    - Built-in presets (precise/balanced/creative)
    - Custom presets
    - Quick toggle buttons
    - Parameter explanations
    """
    
    PRESETS: dict[str, GenerationPreset] = {
        "precise": GenerationPreset(
            name="Precise",
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            description="Deterministic, factual responses. Best for coding, math, and facts."
        ),
        "balanced": GenerationPreset(
            name="Balanced",
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.0,
            description="Default setting. Good for general conversation."
        ),
        "creative": GenerationPreset(
            name="Creative",
            temperature=1.0,
            top_p=1.0,
            top_k=100,
            repetition_penalty=0.9,
            description="Varied, imaginative responses. Best for brainstorming and storytelling."
        ),
        "code": GenerationPreset(
            name="Code",
            temperature=0.2,
            top_p=0.85,
            top_k=30,
            repetition_penalty=1.15,
            description="Optimized for code generation with minimal hallucination."
        ),
        "chat": GenerationPreset(
            name="Chat",
            temperature=0.8,
            top_p=0.95,
            top_k=60,
            repetition_penalty=1.05,
            description="Natural conversational style with good variety."
        )
    }
    
    def __init__(self, default_preset: str = "balanced"):
        """
        Initialize preset manager.
        
        Args:
            default_preset: Default preset name
        """
        self.current_preset_name = default_preset
        self.custom_presets: dict[str, GenerationPreset] = {}
    
    @property
    def current_preset(self) -> GenerationPreset:
        """Get current preset."""
        if self.current_preset_name in self.custom_presets:
            return self.custom_presets[self.current_preset_name]
        return self.PRESETS.get(self.current_preset_name, self.PRESETS["balanced"])
    
    def set_preset(self, preset_name: str):
        """Set current preset by name."""
        if preset_name in self.PRESETS or preset_name in self.custom_presets:
            self.current_preset_name = preset_name
    
    def cycle_preset(self, presets: Optional[list[str]] = None) -> GenerationPreset:
        """
        Cycle through presets.
        
        Args:
            presets: List of preset names to cycle through.
                     If None, uses ["precise", "balanced", "creative"]
        """
        cycle_list = presets or ["precise", "balanced", "creative"]
        
        try:
            current_idx = cycle_list.index(self.current_preset_name)
            next_idx = (current_idx + 1) % len(cycle_list)
        except ValueError:
            next_idx = 0
        
        self.current_preset_name = cycle_list[next_idx]
        return self.current_preset
    
    def get_preset(self, name: str) -> Optional[GenerationPreset]:
        """Get preset by name."""
        if name in self.custom_presets:
            return self.custom_presets[name]
        return self.PRESETS.get(name)
    
    def get_all_presets(self) -> dict[str, GenerationPreset]:
        """Get all presets (built-in + custom)."""
        all_presets = dict(self.PRESETS)
        all_presets.update(self.custom_presets)
        return all_presets
    
    def create_custom_preset(
        self,
        name: str,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        description: str = ""
    ) -> GenerationPreset:
        """Create a custom preset."""
        preset = GenerationPreset(
            name=name,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            description=description
        )
        self.custom_presets[name.lower()] = preset
        return preset
    
    def delete_custom_preset(self, name: str):
        """Delete a custom preset."""
        name_lower = name.lower()
        if name_lower in self.custom_presets:
            del self.custom_presets[name_lower]
            if self.current_preset_name == name_lower:
                self.current_preset_name = "balanced"
    
    def get_generation_params(self) -> dict[str, Any]:
        """Get current generation parameters as dict."""
        preset = self.current_preset
        return {
            "temperature": preset.temperature,
            "top_p": preset.top_p,
            "top_k": preset.top_k,
            "repetition_penalty": preset.repetition_penalty
        }
    
    @staticmethod
    def explain_parameter(param: str) -> str:
        """Get explanation for a generation parameter."""
        explanations = {
            "temperature": (
                "Controls randomness. Lower values (0.1-0.5) make output more "
                "focused and deterministic. Higher values (0.7-1.0) make output "
                "more creative and varied."
            ),
            "top_p": (
                "Nucleus sampling. Considers only tokens with cumulative probability "
                "up to this value. Lower values focus on likely tokens."
            ),
            "top_k": (
                "Limits token selection to top K most likely tokens. Lower values "
                "reduce randomness."
            ),
            "repetition_penalty": (
                "Penalizes repeated tokens. Values > 1.0 reduce repetition. "
                "Values < 1.0 encourage repetition."
            )
        }
        return explanations.get(param, "No explanation available.")


# =============================================================================
# TYPING INDICATOR
# =============================================================================

class TypingState(Enum):
    """States for typing indicator."""
    IDLE = auto()
    THINKING = auto()
    GENERATING = auto()
    PROCESSING = auto()


@dataclass
class TypingIndicator:
    """
    Typing/thinking indicator for AI responses.
    
    Shows different states:
    - Thinking (processing input)
    - Generating (producing response)
    - Processing (running tools)
    """
    state: TypingState = TypingState.IDLE
    message: str = ""
    start_time: Optional[float] = None
    
    def start(self, state: TypingState = TypingState.THINKING, message: str = ""):
        """Start showing indicator."""
        self.state = state
        self.message = message or self._default_message(state)
        self.start_time = time.time()
    
    def stop(self):
        """Stop showing indicator."""
        self.state = TypingState.IDLE
        self.message = ""
        self.start_time = None
    
    def update(self, state: TypingState, message: str = ""):
        """Update indicator state."""
        self.state = state
        self.message = message or self._default_message(state)
    
    def get_display(self) -> str:
        """Get display text for current state."""
        if self.state == TypingState.IDLE:
            return ""
        
        elapsed = time.time() - (self.start_time or time.time())
        dots = "." * (int(elapsed * 2) % 4)
        
        return f"{self.message}{dots}"
    
    def _default_message(self, state: TypingState) -> str:
        """Get default message for state."""
        return {
            TypingState.IDLE: "",
            TypingState.THINKING: "Thinking",
            TypingState.GENERATING: "Writing",
            TypingState.PROCESSING: "Processing"
        }.get(state, "")
    
    @property
    def is_active(self) -> bool:
        """Check if indicator is active."""
        return self.state != TypingState.IDLE


# =============================================================================
# COMBINED MANAGER
# =============================================================================

class ConversationOrganizer:
    """
    Combined manager for conversation organization features.
    
    Combines:
    - Folder management
    - Markdown preview
    - Temperature presets
    - Typing indicator
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize organizer.
        
        Args:
            data_dir: Directory for storing data
        """
        folder_path = data_dir / "folders.json" if data_dir else None
        
        self.folders = FolderManager(folder_path)
        self.markdown = MarkdownPreview()
        self.presets = TemperaturePresetManager()
        self.typing = TypingIndicator()
    
    def get_state(self) -> dict[str, Any]:
        """Get current state for persistence."""
        return {
            "markdown_mode": self.markdown.config.mode.name,
            "temperature_preset": self.presets.current_preset_name
        }
    
    def restore_state(self, state: dict[str, Any]):
        """Restore state from persistence."""
        mode_name = state.get("markdown_mode", "RENDERED")
        self.markdown.config.mode = MarkdownMode[mode_name]
        
        preset = state.get("temperature_preset", "balanced")
        self.presets.set_preset(preset)


# Singleton instance
_organizer: Optional[ConversationOrganizer] = None


def get_organizer(data_dir: Optional[Path] = None) -> ConversationOrganizer:
    """Get or create the conversation organizer."""
    global _organizer
    if _organizer is None:
        _organizer = ConversationOrganizer(data_dir)
    return _organizer
