"""
Chat Features - Message editing, pinning, tagging, and response control.

Surface-level chat improvements:
- Message editing with history
- Pin important messages
- Conversation tags for organization
- Response length control

Part of the ForgeAI GUI features.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Callable
from pathlib import Path
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class ResponseLength(Enum):
    """Response length presets."""
    BRIEF = auto()      # 1-2 sentences
    SHORT = auto()      # A few sentences
    MEDIUM = auto()     # Default, 1-2 paragraphs
    DETAILED = auto()   # Multiple paragraphs
    COMPREHENSIVE = auto()  # Full detailed response


@dataclass
class ResponseConfig:
    """Configuration for response generation."""
    length: ResponseLength = ResponseLength.MEDIUM
    max_tokens: Optional[int] = None  # Override from length
    temperature_preset: Optional[str] = None  # "creative", "balanced", "precise"
    
    def get_max_tokens(self) -> int:
        """Get max tokens based on length preset."""
        if self.max_tokens:
            return self.max_tokens
        return {
            ResponseLength.BRIEF: 50,
            ResponseLength.SHORT: 150,
            ResponseLength.MEDIUM: 500,
            ResponseLength.DETAILED: 1000,
            ResponseLength.COMPREHENSIVE: 2000
        }.get(self.length, 500)
    
    def get_system_hint(self) -> str:
        """Get system prompt hint for response length."""
        hints = {
            ResponseLength.BRIEF: "Keep your response very brief, 1-2 sentences maximum.",
            ResponseLength.SHORT: "Keep your response concise, just a few sentences.",
            ResponseLength.MEDIUM: "Provide a moderate length response.",
            ResponseLength.DETAILED: "Provide a detailed, comprehensive response.",
            ResponseLength.COMPREHENSIVE: "Provide the most thorough and complete response possible."
        }
        return hints.get(self.length, "")


@dataclass
class MessageEdit:
    """Record of a message edit."""
    original_text: str
    edited_text: str
    edited_at: float = field(default_factory=time.time)
    edit_reason: Optional[str] = None


@dataclass 
class Message:
    """
    Enhanced message with editing, pinning, and metadata support.
    
    Extends the basic {"role", "text", "ts"} format with:
    - Unique ID for tracking
    - Edit history
    - Pin status
    - Tags
    - Regeneration support
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = "user"  # "user" or "ai"
    text: str = ""
    timestamp: float = field(default_factory=time.time)
    
    # Edit support
    edit_history: List[MessageEdit] = field(default_factory=list)
    is_edited: bool = False
    
    # Pin support
    is_pinned: bool = False
    pin_note: Optional[str] = None
    
    # Tags
    tags: Set[str] = field(default_factory=set)
    
    # Regeneration
    regeneration_parent: Optional[str] = None  # ID of message this regenerated from
    regeneration_children: List[str] = field(default_factory=list)
    
    # Response config used when generating this message
    response_config: Optional[ResponseConfig] = None
    
    def edit(self, new_text: str, reason: Optional[str] = None):
        """
        Edit the message text, preserving history.
        
        Args:
            new_text: New message text
            reason: Optional reason for edit
        """
        if new_text != self.text:
            self.edit_history.append(MessageEdit(
                original_text=self.text,
                edited_text=new_text,
                edit_reason=reason
            ))
            self.text = new_text
            self.is_edited = True
    
    def pin(self, note: Optional[str] = None):
        """Pin this message with optional note."""
        self.is_pinned = True
        self.pin_note = note
    
    def unpin(self):
        """Unpin this message."""
        self.is_pinned = False
        self.pin_note = None
    
    def add_tag(self, tag: str):
        """Add a tag to this message."""
        self.tags.add(tag.lower().strip())
    
    def remove_tag(self, tag: str):
        """Remove a tag from this message."""
        self.tags.discard(tag.lower().strip())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "role": self.role,
            "text": self.text,
            "ts": self.timestamp,
            "edit_history": [
                {
                    "original": e.original_text,
                    "edited": e.edited_text,
                    "edited_at": e.edited_at,
                    "reason": e.edit_reason
                }
                for e in self.edit_history
            ],
            "is_edited": self.is_edited,
            "is_pinned": self.is_pinned,
            "pin_note": self.pin_note,
            "tags": list(self.tags),
            "regeneration_parent": self.regeneration_parent,
            "regeneration_children": self.regeneration_children
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create Message from dictionary."""
        msg = cls(
            id=data.get("id", str(uuid.uuid4())),
            role=data.get("role", "user"),
            text=data.get("text", ""),
            timestamp=data.get("ts", time.time()),
            is_edited=data.get("is_edited", False),
            is_pinned=data.get("is_pinned", False),
            pin_note=data.get("pin_note"),
            regeneration_parent=data.get("regeneration_parent"),
            regeneration_children=data.get("regeneration_children", [])
        )
        
        # Restore edit history
        for edit_data in data.get("edit_history", []):
            msg.edit_history.append(MessageEdit(
                original_text=edit_data.get("original", ""),
                edited_text=edit_data.get("edited", ""),
                edited_at=edit_data.get("edited_at", 0),
                edit_reason=edit_data.get("reason")
            ))
        
        # Restore tags
        for tag in data.get("tags", []):
            msg.tags.add(tag)
        
        return msg
    
    @classmethod
    def from_legacy(cls, legacy: Dict[str, Any]) -> 'Message':
        """
        Convert legacy message format to enhanced Message.
        
        Legacy format: {"role": "user", "text": "Hello", "ts": 12345}
        """
        return cls(
            id=legacy.get("id", str(uuid.uuid4())),
            role=legacy.get("role", "user"),
            text=legacy.get("text", ""),
            timestamp=legacy.get("ts", time.time())
        )


@dataclass
class ConversationTag:
    """Tag for organizing conversations."""
    name: str
    color: str = "#808080"  # Default gray
    description: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "color": self.color,
            "description": self.description,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTag':
        return cls(
            name=data.get("name", ""),
            color=data.get("color", "#808080"),
            description=data.get("description"),
            created_at=data.get("created_at", time.time())
        )


class ConversationTagManager:
    """
    Manage conversation tags across the application.
    
    Tags help organize conversations by:
    - Project/topic
    - Priority
    - Status
    - Custom categories
    """
    
    DEFAULT_TAGS = [
        ConversationTag("important", "#ff4444", "High priority conversations"),
        ConversationTag("work", "#4488ff", "Work-related conversations"),
        ConversationTag("personal", "#44ff44", "Personal conversations"),
        ConversationTag("code", "#ff8844", "Programming discussions"),
        ConversationTag("research", "#8844ff", "Research topics"),
        ConversationTag("archived", "#888888", "Old/inactive conversations")
    ]
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize tag manager.
        
        Args:
            storage_path: Path to store tags. If None, uses memory only.
        """
        self.storage_path = storage_path
        self.tags: Dict[str, ConversationTag] = {}
        self._conversation_tags: Dict[str, Set[str]] = {}  # conv_name -> set of tag names
        
        # Load or initialize
        if storage_path and storage_path.exists():
            self._load()
        else:
            # Initialize with defaults
            for tag in self.DEFAULT_TAGS:
                self.tags[tag.name] = tag
    
    def _load(self):
        """Load tags from storage."""
        if not self.storage_path:
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            for tag_data in data.get("tags", []):
                tag = ConversationTag.from_dict(tag_data)
                self.tags[tag.name] = tag
            self._conversation_tags = {
                k: set(v) for k, v in data.get("conversation_tags", {}).items()
            }
        except Exception as e:
            logger.warning(f"Failed to load tags: {e}")
            for tag in self.DEFAULT_TAGS:
                self.tags[tag.name] = tag
    
    def _save(self):
        """Save tags to storage."""
        if not self.storage_path:
            return
        try:
            data = {
                "tags": [t.to_dict() for t in self.tags.values()],
                "conversation_tags": {
                    k: list(v) for k, v in self._conversation_tags.items()
                }
            }
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save tags: {e}")
    
    def create_tag(self, name: str, color: str = "#808080", description: Optional[str] = None) -> ConversationTag:
        """Create a new tag."""
        name = name.lower().strip()
        if name in self.tags:
            return self.tags[name]
        
        tag = ConversationTag(name=name, color=color, description=description)
        self.tags[name] = tag
        self._save()
        return tag
    
    def delete_tag(self, name: str):
        """Delete a tag."""
        name = name.lower().strip()
        if name in self.tags:
            del self.tags[name]
            # Remove from all conversations
            for conv_tags in self._conversation_tags.values():
                conv_tags.discard(name)
            self._save()
    
    def get_tag(self, name: str) -> Optional[ConversationTag]:
        """Get a tag by name."""
        return self.tags.get(name.lower().strip())
    
    def get_all_tags(self) -> List[ConversationTag]:
        """Get all tags."""
        return list(self.tags.values())
    
    def tag_conversation(self, conversation_name: str, tag_name: str):
        """Add a tag to a conversation."""
        tag_name = tag_name.lower().strip()
        if tag_name not in self.tags:
            self.create_tag(tag_name)
        
        if conversation_name not in self._conversation_tags:
            self._conversation_tags[conversation_name] = set()
        self._conversation_tags[conversation_name].add(tag_name)
        self._save()
    
    def untag_conversation(self, conversation_name: str, tag_name: str):
        """Remove a tag from a conversation."""
        tag_name = tag_name.lower().strip()
        if conversation_name in self._conversation_tags:
            self._conversation_tags[conversation_name].discard(tag_name)
            self._save()
    
    def get_conversation_tags(self, conversation_name: str) -> List[ConversationTag]:
        """Get all tags for a conversation."""
        tag_names = self._conversation_tags.get(conversation_name, set())
        return [self.tags[name] for name in tag_names if name in self.tags]
    
    def get_conversations_by_tag(self, tag_name: str) -> List[str]:
        """Get all conversations with a specific tag."""
        tag_name = tag_name.lower().strip()
        return [
            conv for conv, tags in self._conversation_tags.items()
            if tag_name in tags
        ]
    
    def filter_conversations(
        self,
        all_conversations: List[str],
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None
    ) -> List[str]:
        """
        Filter conversations by tags.
        
        Args:
            all_conversations: List of conversation names
            include_tags: Only include conversations with ANY of these tags
            exclude_tags: Exclude conversations with ANY of these tags
            
        Returns:
            Filtered list of conversation names
        """
        result = []
        
        include_set = set(t.lower().strip() for t in (include_tags or []))
        exclude_set = set(t.lower().strip() for t in (exclude_tags or []))
        
        for conv in all_conversations:
            conv_tags = self._conversation_tags.get(conv, set())
            
            # Check exclusion first
            if exclude_set and conv_tags & exclude_set:
                continue
            
            # Check inclusion
            if include_set:
                if conv_tags & include_set:
                    result.append(conv)
            else:
                result.append(conv)
        
        return result


class MessageEditor:
    """
    Handles message editing with regeneration support.
    
    When a user edits a message:
    1. The edit is recorded in history
    2. Optionally, the AI response can be regenerated from that point
    """
    
    def __init__(self, on_regenerate: Optional[Callable[[str, List[Message]], None]] = None):
        """
        Initialize message editor.
        
        Args:
            on_regenerate: Callback when regeneration is requested.
                           Called with (message_id, messages_before)
        """
        self.on_regenerate = on_regenerate
    
    def edit_message(
        self,
        messages: List[Message],
        message_id: str,
        new_text: str,
        regenerate: bool = False
    ) -> List[Message]:
        """
        Edit a message in the conversation.
        
        Args:
            messages: List of all messages
            message_id: ID of message to edit
            new_text: New text content
            regenerate: If True, regenerate AI response after edit
            
        Returns:
            Updated message list (may be truncated if regenerating)
        """
        # Find the message
        msg_idx = None
        for i, msg in enumerate(messages):
            if msg.id == message_id:
                msg_idx = i
                break
        
        if msg_idx is None:
            raise ValueError(f"Message not found: {message_id}")
        
        # Edit the message
        messages[msg_idx].edit(new_text)
        
        if regenerate and messages[msg_idx].role == "user":
            # Truncate conversation to this point
            messages = messages[:msg_idx + 1]
            
            # Trigger regeneration callback
            if self.on_regenerate:
                self.on_regenerate(message_id, messages)
        
        return messages
    
    def get_edit_history(self, message: Message) -> List[Dict[str, Any]]:
        """Get formatted edit history for a message."""
        return [
            {
                "version": i + 1,
                "text": edit.original_text if i == 0 else messages[i-1].edited_text,
                "edited_at": edit.edited_at,
                "reason": edit.edit_reason
            }
            for i, edit in enumerate(message.edit_history)
        ] + [{"version": len(message.edit_history) + 1, "text": message.text, "current": True}]


class PinManager:
    """
    Manage pinned messages across conversations.
    
    Pinned messages:
    - Appear in a quick-access sidebar
    - Can have notes explaining why they're pinned
    - Are preserved even if conversation is deleted
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize pin manager.
        
        Args:
            storage_path: Path to store pins. If None, uses memory only.
        """
        self.storage_path = storage_path
        self.pins: Dict[str, Dict[str, Message]] = {}  # conv_name -> {msg_id: Message}
        
        if storage_path and storage_path.exists():
            self._load()
    
    def _load(self):
        """Load pins from storage."""
        if not self.storage_path:
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            for conv_name, pins in data.items():
                self.pins[conv_name] = {
                    msg_id: Message.from_dict(msg_data)
                    for msg_id, msg_data in pins.items()
                }
        except Exception as e:
            logger.warning(f"Failed to load pins: {e}")
    
    def _save(self):
        """Save pins to storage."""
        if not self.storage_path:
            return
        try:
            data = {
                conv_name: {
                    msg_id: msg.to_dict()
                    for msg_id, msg in pins.items()
                }
                for conv_name, pins in self.pins.items()
            }
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save pins: {e}")
    
    def pin_message(self, conversation_name: str, message: Message, note: Optional[str] = None):
        """Pin a message."""
        message.pin(note)
        
        if conversation_name not in self.pins:
            self.pins[conversation_name] = {}
        self.pins[conversation_name][message.id] = message
        self._save()
    
    def unpin_message(self, conversation_name: str, message_id: str):
        """Unpin a message."""
        if conversation_name in self.pins:
            if message_id in self.pins[conversation_name]:
                del self.pins[conversation_name][message_id]
                self._save()
    
    def get_pinned_messages(self, conversation_name: Optional[str] = None) -> List[Message]:
        """
        Get pinned messages.
        
        Args:
            conversation_name: If provided, only pins from this conversation.
                             If None, all pins across all conversations.
        """
        if conversation_name:
            return list(self.pins.get(conversation_name, {}).values())
        
        all_pins = []
        for pins in self.pins.values():
            all_pins.extend(pins.values())
        return sorted(all_pins, key=lambda m: m.timestamp, reverse=True)
    
    def is_pinned(self, conversation_name: str, message_id: str) -> bool:
        """Check if a message is pinned."""
        return message_id in self.pins.get(conversation_name, {})
    
    def get_pin_count(self, conversation_name: Optional[str] = None) -> int:
        """Get count of pinned messages."""
        if conversation_name:
            return len(self.pins.get(conversation_name, {}))
        return sum(len(pins) for pins in self.pins.values())


class ResponseLengthController:
    """
    Control response length with presets and custom values.
    
    Provides:
    - Named presets (brief, short, medium, detailed, comprehensive)
    - Custom token limits
    - System prompt hints for length
    """
    
    PRESETS = {
        "brief": ResponseConfig(ResponseLength.BRIEF),
        "short": ResponseConfig(ResponseLength.SHORT),
        "medium": ResponseConfig(ResponseLength.MEDIUM),
        "detailed": ResponseConfig(ResponseLength.DETAILED),
        "comprehensive": ResponseConfig(ResponseLength.COMPREHENSIVE)
    }
    
    def __init__(self, default_preset: str = "medium"):
        """
        Initialize controller.
        
        Args:
            default_preset: Default response length preset
        """
        self.current_config = self.PRESETS.get(default_preset, self.PRESETS["medium"])
    
    def set_preset(self, preset_name: str):
        """Set response length to a preset."""
        if preset_name in self.PRESETS:
            self.current_config = self.PRESETS[preset_name]
    
    def set_custom_tokens(self, max_tokens: int):
        """Set custom max token limit."""
        self.current_config = ResponseConfig(
            length=ResponseLength.MEDIUM,
            max_tokens=max_tokens
        )
    
    def get_config(self) -> ResponseConfig:
        """Get current response config."""
        return self.current_config
    
    def get_max_tokens(self) -> int:
        """Get current max tokens."""
        return self.current_config.get_max_tokens()
    
    def get_system_hint(self) -> str:
        """Get system prompt hint for current length."""
        return self.current_config.get_system_hint()
    
    def apply_to_prompt(self, system_prompt: str) -> str:
        """Apply length hint to system prompt."""
        hint = self.get_system_hint()
        if hint:
            return f"{system_prompt}\n\n{hint}"
        return system_prompt


class ChatFeaturesManager:
    """
    Unified manager for all chat features.
    
    Combines:
    - Message editing
    - Pins
    - Tags
    - Response length control
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize chat features.
        
        Args:
            data_dir: Directory for storing feature data
        """
        self.data_dir = data_dir
        
        # Initialize components
        tag_path = data_dir / "tags.json" if data_dir else None
        pin_path = data_dir / "pins.json" if data_dir else None
        
        self.tag_manager = ConversationTagManager(tag_path)
        self.pin_manager = PinManager(pin_path)
        self.response_controller = ResponseLengthController()
        self.message_editor = MessageEditor()
    
    def convert_messages(self, legacy_messages: List[Dict[str, Any]]) -> List[Message]:
        """Convert legacy message format to enhanced Messages."""
        return [Message.from_legacy(m) for m in legacy_messages]
    
    def export_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Export Messages to storage format."""
        return [m.to_dict() for m in messages]


# Singleton instance
_features_manager: Optional[ChatFeaturesManager] = None


def get_chat_features(data_dir: Optional[Path] = None) -> ChatFeaturesManager:
    """Get or create the chat features manager."""
    global _features_manager
    if _features_manager is None:
        _features_manager = ChatFeaturesManager(data_dir)
    return _features_manager
