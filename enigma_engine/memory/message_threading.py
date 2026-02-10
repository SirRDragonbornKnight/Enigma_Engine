"""
Message Threading System

Allows replying to specific messages in a conversation.
Implements threaded discussions with parent-child relationships.

FILE: enigma_engine/memory/message_threading.py
TYPE: Conversation Management
MAIN CLASSES: ThreadedMessage, MessageThread, ThreadManager
"""

import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ThreadedMessage:
    """A message that can be part of a thread."""
    id: str
    content: str
    role: str  # "user" or "assistant"
    timestamp: float = field(default_factory=time.time)
    parent_id: Optional[str] = None  # ID of message being replied to
    thread_id: Optional[str] = None  # If part of a thread
    reply_count: int = 0
    reactions: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    edited: bool = False
    edit_timestamp: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp,
            "parent_id": self.parent_id,
            "thread_id": self.thread_id,
            "reply_count": self.reply_count,
            "reactions": self.reactions,
            "metadata": self.metadata,
            "edited": self.edited,
            "edit_timestamp": self.edit_timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ThreadedMessage':
        return cls(
            id=data["id"],
            content=data["content"],
            role=data["role"],
            timestamp=data.get("timestamp", time.time()),
            parent_id=data.get("parent_id"),
            thread_id=data.get("thread_id"),
            reply_count=data.get("reply_count", 0),
            reactions=data.get("reactions", {}),
            metadata=data.get("metadata", {}),
            edited=data.get("edited", False),
            edit_timestamp=data.get("edit_timestamp")
        )


@dataclass
class MessageThread:
    """A thread of related messages."""
    id: str
    root_message_id: str
    message_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    is_resolved: bool = False
    title: str = ""
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "root_message_id": self.root_message_id,
            "message_ids": self.message_ids,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "is_resolved": self.is_resolved,
            "title": self.title
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MessageThread':
        return cls(
            id=data["id"],
            root_message_id=data["root_message_id"],
            message_ids=data.get("message_ids", []),
            created_at=data.get("created_at", time.time()),
            last_activity=data.get("last_activity", time.time()),
            is_resolved=data.get("is_resolved", False),
            title=data.get("title", "")
        )


class ThreadManager:
    """Manages threaded conversations."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize thread manager.
        
        Args:
            storage_path: Path for persistent storage
        """
        self._messages: dict[str, ThreadedMessage] = {}
        self._threads: dict[str, MessageThread] = {}
        self._message_to_thread: dict[str, str] = {}  # message_id -> thread_id
        self._children: dict[str, list[str]] = defaultdict(list)  # parent_id -> child_ids
        self._storage_path = storage_path
        
        if storage_path and storage_path.exists():
            self._load()
            
    def add_message(self, content: str, role: str,
                    reply_to: Optional[str] = None,
                    metadata: dict[str, Any] = None) -> ThreadedMessage:
        """
        Add a message, optionally as a reply.
        
        Args:
            content: Message content
            role: "user" or "assistant"
            reply_to: ID of message to reply to
            metadata: Additional metadata
            
        Returns:
            The created message
        """
        msg_id = str(uuid.uuid4())
        
        message = ThreadedMessage(
            id=msg_id,
            content=content,
            role=role,
            parent_id=reply_to,
            metadata=metadata or {}
        )
        
        self._messages[msg_id] = message
        
        # Handle reply
        if reply_to and reply_to in self._messages:
            parent = self._messages[reply_to]
            parent.reply_count += 1
            self._children[reply_to].append(msg_id)
            
            # Create or join thread
            if parent.thread_id:
                # Join existing thread
                thread = self._threads[parent.thread_id]
                thread.message_ids.append(msg_id)
                thread.last_activity = time.time()
                message.thread_id = parent.thread_id
            else:
                # Create new thread
                thread = self._create_thread(reply_to, msg_id)
                parent.thread_id = thread.id
                message.thread_id = thread.id
                
        return message
    
    def _create_thread(self, root_id: str, first_reply_id: str) -> MessageThread:
        """Create a new thread."""
        thread_id = str(uuid.uuid4())
        thread = MessageThread(
            id=thread_id,
            root_message_id=root_id,
            message_ids=[root_id, first_reply_id]
        )
        self._threads[thread_id] = thread
        self._message_to_thread[root_id] = thread_id
        self._message_to_thread[first_reply_id] = thread_id
        
        # Set thread title from root message
        root = self._messages.get(root_id)
        if root:
            thread.title = root.content[:50] + "..." if len(root.content) > 50 else root.content
            
        return thread
    
    def get_message(self, msg_id: str) -> Optional[ThreadedMessage]:
        """Get a message by ID."""
        return self._messages.get(msg_id)
    
    def get_replies(self, msg_id: str) -> list[ThreadedMessage]:
        """Get all direct replies to a message."""
        child_ids = self._children.get(msg_id, [])
        return [self._messages[cid] for cid in child_ids if cid in self._messages]
    
    def get_thread(self, thread_id: str) -> Optional[MessageThread]:
        """Get a thread by ID."""
        return self._threads.get(thread_id)
    
    def get_thread_messages(self, thread_id: str) -> list[ThreadedMessage]:
        """Get all messages in a thread, ordered by timestamp."""
        thread = self._threads.get(thread_id)
        if not thread:
            return []
        messages = [self._messages[mid] for mid in thread.message_ids if mid in self._messages]
        return sorted(messages, key=lambda m: m.timestamp)
    
    def get_thread_for_message(self, msg_id: str) -> Optional[MessageThread]:
        """Get the thread a message belongs to."""
        thread_id = self._message_to_thread.get(msg_id)
        if thread_id:
            return self._threads.get(thread_id)
        return None
    
    def get_conversation_with_threads(self) -> list[dict]:
        """
        Get conversation structure with thread info.
        
        Returns:
            List of messages with thread metadata
        """
        # Get root level messages (no parent)
        root_messages = [
            m for m in self._messages.values() 
            if m.parent_id is None
        ]
        root_messages.sort(key=lambda m: m.timestamp)
        
        result = []
        for msg in root_messages:
            msg_data = msg.to_dict()
            if msg.reply_count > 0:
                msg_data["replies"] = [
                    r.to_dict() for r in self.get_replies(msg.id)
                ]
            result.append(msg_data)
            
        return result
    
    def edit_message(self, msg_id: str, new_content: str) -> bool:
        """
        Edit a message's content.
        
        Args:
            msg_id: Message ID
            new_content: New content
            
        Returns:
            True if successful
        """
        msg = self._messages.get(msg_id)
        if msg:
            msg.content = new_content
            msg.edited = True
            msg.edit_timestamp = time.time()
            return True
        return False
    
    def add_reaction(self, msg_id: str, reaction: str) -> bool:
        """
        Add a reaction to a message.
        
        Args:
            msg_id: Message ID
            reaction: Reaction emoji/text
            
        Returns:
            True if successful
        """
        msg = self._messages.get(msg_id)
        if msg:
            msg.reactions[reaction] = msg.reactions.get(reaction, 0) + 1
            return True
        return False
    
    def resolve_thread(self, thread_id: str):
        """Mark a thread as resolved."""
        thread = self._threads.get(thread_id)
        if thread:
            thread.is_resolved = True
            
    def get_active_threads(self) -> list[MessageThread]:
        """Get all unresolved threads."""
        return [t for t in self._threads.values() if not t.is_resolved]
    
    def delete_message(self, msg_id: str, delete_replies: bool = False):
        """
        Delete a message.
        
        Args:
            msg_id: Message ID
            delete_replies: Also delete all replies
        """
        msg = self._messages.get(msg_id)
        if not msg:
            return
            
        # Handle replies
        if delete_replies:
            for reply_id in self._children.get(msg_id, []).copy():
                self.delete_message(reply_id, delete_replies=True)
        else:
            # Re-parent replies to grandparent
            for reply_id in self._children.get(msg_id, []):
                reply = self._messages.get(reply_id)
                if reply:
                    reply.parent_id = msg.parent_id
                    if msg.parent_id:
                        self._children[msg.parent_id].append(reply_id)
                        
        # Update parent's reply count
        if msg.parent_id and msg.parent_id in self._messages:
            parent = self._messages[msg.parent_id]
            parent.reply_count = max(0, parent.reply_count - 1)
            
        # Remove from children tracking
        if msg.parent_id:
            self._children[msg.parent_id] = [
                cid for cid in self._children[msg.parent_id] if cid != msg_id
            ]
            
        # Remove from thread
        if msg.thread_id and msg.thread_id in self._threads:
            thread = self._threads[msg.thread_id]
            if msg_id in thread.message_ids:
                thread.message_ids.remove(msg_id)
                
        del self._messages[msg_id]
        self._message_to_thread.pop(msg_id, None)
        
    def _save(self):
        """Save to disk."""
        if not self._storage_path:
            return
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "messages": {k: v.to_dict() for k, v in self._messages.items()},
                "threads": {k: v.to_dict() for k, v in self._threads.items()},
                "message_to_thread": self._message_to_thread,
                "children": dict(self._children)
            }
            with open(self._storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save threads: {e}")
            
    def _load(self):
        """Load from disk."""
        try:
            with open(self._storage_path) as f:
                data = json.load(f)
            self._messages = {
                k: ThreadedMessage.from_dict(v) 
                for k, v in data.get("messages", {}).items()
            }
            self._threads = {
                k: MessageThread.from_dict(v)
                for k, v in data.get("threads", {}).items()
            }
            self._message_to_thread = data.get("message_to_thread", {})
            self._children = defaultdict(list, data.get("children", {}))
        except Exception as e:
            logger.error(f"Failed to load threads: {e}")


# Singleton
_thread_manager: Optional[ThreadManager] = None


def get_thread_manager(storage_path: Optional[Path] = None) -> ThreadManager:
    """Get the thread manager singleton."""
    global _thread_manager
    if _thread_manager is None:
        _thread_manager = ThreadManager(storage_path)
    return _thread_manager


__all__ = [
    'ThreadedMessage',
    'MessageThread',
    'ThreadManager',
    'get_thread_manager'
]
