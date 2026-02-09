"""
Conversation Branching for Enigma AI Engine

Explore alternative conversation paths.

Features:
- Branch conversations at any point
- Compare different responses
- Merge branches
- Tree visualization
- History navigation

Usage:
    from enigma_engine.memory.branching import ConversationTree, Branch
    
    # Create tree
    tree = ConversationTree()
    
    # Add messages
    tree.add_message("user", "Hello")
    tree.add_message("assistant", "Hi there!")
    
    # Branch for alternatives
    branch_id = tree.branch()
    tree.add_message("assistant", "Hello! How can I help?")
    
    # Switch branches
    tree.switch(branch_id)
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A conversation message."""
    id: str
    role: str  # user, assistant, system
    content: str
    timestamp: float = field(default_factory=time.time)
    branch_id: str = "main"
    parent_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # For branching
    children: List[str] = field(default_factory=list)  # Child message IDs
    is_branch_point: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "branch_id": self.branch_id,
            "parent_id": self.parent_id,
            "children": self.children,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            branch_id=data.get("branch_id", "main"),
            parent_id=data.get("parent_id"),
            children=data.get("children", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class Branch:
    """A conversation branch."""
    id: str
    name: str
    parent_branch: Optional[str] = None
    fork_point: Optional[str] = None  # Message ID where branch forked
    created: float = field(default_factory=time.time)
    
    # State
    is_active: bool = True
    is_merged: bool = False
    
    # Statistics
    message_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "parent_branch": self.parent_branch,
            "fork_point": self.fork_point,
            "created": self.created,
            "is_active": self.is_active,
            "message_count": self.message_count
        }


class ConversationTree:
    """
    Tree-structured conversation with branching support.
    """
    
    def __init__(self, conversation_id: Optional[str] = None):
        """
        Initialize conversation tree.
        
        Args:
            conversation_id: Optional ID for this conversation
        """
        self.id = conversation_id or str(uuid.uuid4())[:8]
        self.created = time.time()
        
        # Storage
        self._messages: Dict[str, Message] = {}
        self._branches: Dict[str, Branch] = {}
        
        # Current state
        self._current_branch = "main"
        self._current_message: Optional[str] = None  # Last message ID in current branch
        
        # Create main branch
        self._branches["main"] = Branch(
            id="main",
            name="Main",
            created=self.created
        )
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to current branch.
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata
            
        Returns:
            The new message
        """
        msg_id = str(uuid.uuid4())[:8]
        
        message = Message(
            id=msg_id,
            role=role,
            content=content,
            branch_id=self._current_branch,
            parent_id=self._current_message,
            metadata=metadata or {}
        )
        
        # Update parent's children
        if self._current_message and self._current_message in self._messages:
            parent = self._messages[self._current_message]
            parent.children.append(msg_id)
            if len(parent.children) > 1:
                parent.is_branch_point = True
        
        # Store message
        self._messages[msg_id] = message
        self._current_message = msg_id
        
        # Update branch stats
        self._branches[self._current_branch].message_count += 1
        
        return message
    
    def branch(
        self,
        name: Optional[str] = None,
        from_message: Optional[str] = None
    ) -> str:
        """
        Create a new branch.
        
        Args:
            name: Branch name
            from_message: Message ID to branch from (default: current)
            
        Returns:
            New branch ID
        """
        branch_id = str(uuid.uuid4())[:8]
        
        # Determine fork point
        fork_point = from_message or self._current_message
        
        # Create branch
        branch = Branch(
            id=branch_id,
            name=name or f"Branch {len(self._branches) + 1}",
            parent_branch=self._current_branch,
            fork_point=fork_point
        )
        
        self._branches[branch_id] = branch
        
        # Mark fork point
        if fork_point and fork_point in self._messages:
            self._messages[fork_point].is_branch_point = True
        
        # Switch to new branch
        self._current_branch = branch_id
        self._current_message = fork_point
        
        logger.info(f"Created branch: {branch_id} from {fork_point}")
        return branch_id
    
    def switch(self, branch_id: str) -> bool:
        """
        Switch to a different branch.
        
        Args:
            branch_id: Target branch ID
            
        Returns:
            True if successful
        """
        if branch_id not in self._branches:
            return False
        
        self._current_branch = branch_id
        
        # Find last message in this branch
        self._current_message = self._find_last_message(branch_id)
        
        logger.info(f"Switched to branch: {branch_id}")
        return True
    
    def _find_last_message(self, branch_id: str) -> Optional[str]:
        """Find the last message in a branch."""
        messages = [
            msg for msg in self._messages.values()
            if msg.branch_id == branch_id
        ]
        
        if not messages:
            # Check parent branch for fork point
            branch = self._branches.get(branch_id)
            if branch and branch.fork_point:
                return branch.fork_point
            return None
        
        # Find message with no children in this branch
        for msg in sorted(messages, key=lambda m: m.timestamp, reverse=True):
            if not any(
                c for c in msg.children
                if c in self._messages and self._messages[c].branch_id == branch_id
            ):
                return msg.id
        
        return messages[-1].id if messages else None
    
    def get_history(
        self,
        branch_id: Optional[str] = None,
        include_parent: bool = True
    ) -> List[Message]:
        """
        Get conversation history for a branch.
        
        Args:
            branch_id: Branch ID (default: current)
            include_parent: Include messages from parent branches
            
        Returns:
            List of messages in order
        """
        branch_id = branch_id or self._current_branch
        branch = self._branches.get(branch_id)
        
        if not branch:
            return []
        
        # Get messages in this branch
        messages = []
        
        # Include parent branch messages up to fork point
        if include_parent and branch.parent_branch and branch.fork_point:
            parent_messages = self._get_path_to_message(branch.fork_point)
            messages.extend(parent_messages)
        
        # Add messages from this branch
        branch_messages = [
            msg for msg in self._messages.values()
            if msg.branch_id == branch_id
        ]
        
        messages.extend(sorted(branch_messages, key=lambda m: m.timestamp))
        
        return messages
    
    def _get_path_to_message(self, message_id: str) -> List[Message]:
        """Get all messages leading to a specific message."""
        path = []
        current_id = message_id
        
        while current_id:
            if current_id in self._messages:
                msg = self._messages[current_id]
                path.append(msg)
                current_id = msg.parent_id
            else:
                break
        
        return list(reversed(path))
    
    def compare_branches(
        self,
        branch_a: str,
        branch_b: str
    ) -> Dict[str, Any]:
        """
        Compare two branches.
        
        Args:
            branch_a: First branch ID
            branch_b: Second branch ID
            
        Returns:
            Comparison data
        """
        history_a = self.get_history(branch_a)
        history_b = self.get_history(branch_b)
        
        # Find common ancestor
        ids_a = {m.id for m in history_a}
        ids_b = {m.id for m in history_b}
        common = ids_a & ids_b
        
        # Find divergence point
        diverge_point = None
        for msg in history_a:
            if msg.id in common and msg.is_branch_point:
                diverge_point = msg.id
                break
        
        return {
            "branch_a": branch_a,
            "branch_b": branch_b,
            "messages_a": len(history_a),
            "messages_b": len(history_b),
            "common_messages": len(common),
            "unique_to_a": len(ids_a - ids_b),
            "unique_to_b": len(ids_b - ids_a),
            "diverge_point": diverge_point,
            "history_a": [m.to_dict() for m in history_a],
            "history_b": [m.to_dict() for m in history_b]
        }
    
    def merge(
        self,
        source_branch: str,
        target_branch: str,
        strategy: str = "append"
    ) -> bool:
        """
        Merge branches.
        
        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into
            strategy: Merge strategy (append, interleave)
            
        Returns:
            True if successful
        """
        if source_branch not in self._branches or target_branch not in self._branches:
            return False
        
        source_history = self.get_history(source_branch, include_parent=False)
        
        # Find merge point in target
        target_last = self._find_last_message(target_branch)
        
        # Copy messages with new branch ID
        for msg in source_history:
            new_msg = Message(
                id=str(uuid.uuid4())[:8],
                role=msg.role,
                content=msg.content,
                branch_id=target_branch,
                parent_id=target_last,
                metadata={**msg.metadata, "merged_from": source_branch}
            )
            self._messages[new_msg.id] = new_msg
            target_last = new_msg.id
        
        # Mark source as merged
        self._branches[source_branch].is_merged = True
        
        logger.info(f"Merged {source_branch} into {target_branch}")
        return True
    
    def delete_branch(self, branch_id: str) -> bool:
        """Delete a branch and its unique messages."""
        if branch_id == "main":
            return False  # Can't delete main
        
        if branch_id not in self._branches:
            return False
        
        # Delete messages unique to this branch
        to_delete = [
            msg_id for msg_id, msg in self._messages.items()
            if msg.branch_id == branch_id
        ]
        
        for msg_id in to_delete:
            del self._messages[msg_id]
        
        del self._branches[branch_id]
        
        # Switch to main if needed
        if self._current_branch == branch_id:
            self.switch("main")
        
        return True
    
    def get_branches(self) -> List[Branch]:
        """Get all branches."""
        return list(self._branches.values())
    
    def get_branch_points(self) -> List[Message]:
        """Get all branch points in the conversation."""
        return [
            msg for msg in self._messages.values()
            if msg.is_branch_point
        ]
    
    def goto_message(self, message_id: str) -> bool:
        """
        Navigate to a specific message.
        
        Args:
            message_id: Target message ID
            
        Returns:
            True if successful
        """
        if message_id not in self._messages:
            return False
        
        msg = self._messages[message_id]
        self._current_branch = msg.branch_id
        self._current_message = message_id
        
        return True
    
    def regenerate(
        self,
        generator: Callable[[List[Dict[str, str]]], str],
        count: int = 3
    ) -> List[str]:
        """
        Regenerate last response with alternatives.
        
        Args:
            generator: Function to generate response
            count: Number of alternatives
            
        Returns:
            List of branch IDs with alternatives
        """
        if not self._current_message:
            return []
        
        current_msg = self._messages.get(self._current_message)
        if not current_msg or current_msg.role != "assistant":
            return []
        
        # Get history up to before current message
        parent_id = current_msg.parent_id
        if not parent_id:
            return []
        
        history = self._get_path_to_message(parent_id)
        history_dicts = [{"role": m.role, "content": m.content} for m in history]
        
        # Generate alternatives
        branch_ids = []
        for i in range(count):
            # Create branch
            branch_id = self.branch(f"Alt {i + 1}", from_message=parent_id)
            
            # Generate new response
            response = generator(history_dicts)
            self.add_message("assistant", response)
            
            branch_ids.append(branch_id)
        
        return branch_ids
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire tree to dictionary."""
        return {
            "id": self.id,
            "created": self.created,
            "current_branch": self._current_branch,
            "current_message": self._current_message,
            "messages": {k: v.to_dict() for k, v in self._messages.items()},
            "branches": {k: v.to_dict() for k, v in self._branches.items()}
        }
    
    def save(self, path: str):
        """Save tree to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ConversationTree":
        """Load tree from file."""
        with open(path) as f:
            data = json.load(f)
        
        tree = cls(data["id"])
        tree.created = data["created"]
        tree._current_branch = data["current_branch"]
        tree._current_message = data["current_message"]
        
        tree._messages = {
            k: Message.from_dict(v)
            for k, v in data["messages"].items()
        }
        
        tree._branches = {
            k: Branch(**v)
            for k, v in data["branches"].items()
        }
        
        return tree
    
    def visualize(self) -> str:
        """Create ASCII visualization of the tree."""
        lines = ["Conversation Tree"]
        lines.append("=" * 40)
        
        # Find root messages (no parent)
        roots = [
            msg for msg in self._messages.values()
            if not msg.parent_id
        ]
        
        def print_message(msg: Message, prefix: str = "", is_last: bool = True):
            connection = "└── " if is_last else "├── "
            marker = "*" if msg.id == self._current_message else ""
            branch_marker = f"[{msg.branch_id}]" if msg.is_branch_point else ""
            
            lines.append(f"{prefix}{connection}{msg.role}: {msg.content[:30]}... {branch_marker}{marker}")
            
            new_prefix = prefix + ("    " if is_last else "│   ")
            
            # Print children
            children = [
                self._messages[c] for c in msg.children
                if c in self._messages
            ]
            
            for i, child in enumerate(children):
                print_message(child, new_prefix, i == len(children) - 1)
        
        for root in roots:
            print_message(root)
        
        return "\n".join(lines)
    
    @property
    def current_branch(self) -> str:
        """Get current branch ID."""
        return self._current_branch
    
    @property
    def current_message(self) -> Optional[Message]:
        """Get current message."""
        if self._current_message:
            return self._messages.get(self._current_message)
        return None


# Convenience function
def branch_conversation() -> ConversationTree:
    """Create a new branching conversation."""
    return ConversationTree()
