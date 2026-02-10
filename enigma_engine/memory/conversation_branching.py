"""
Conversation Branching System

Allows forking conversations to explore different response paths.
Implements a tree-based conversation structure with branch visualization.

FILE: enigma_engine/memory/conversation_branching.py
TYPE: Conversation Management
MAIN CLASSES: ConversationTree, BranchNode, BranchManager
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Message sender role."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """A single message in the conversation."""
    id: str
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        return cls(
            id=data["id"],
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {})
        )


@dataclass
class BranchNode:
    """A node in the conversation tree."""
    id: str
    message: Message
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    is_active: bool = True
    branch_label: str = ""
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "message": self.message.to_dict(),
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "is_active": self.is_active,
            "branch_label": self.branch_label,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BranchNode':
        return cls(
            id=data["id"],
            message=Message.from_dict(data["message"]),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            is_active=data.get("is_active", True),
            branch_label=data.get("branch_label", ""),
            created_at=data.get("created_at", time.time())
        )


class ConversationTree:
    """Tree-structured conversation with branching support."""
    
    def __init__(self, conversation_id: Optional[str] = None):
        """
        Initialize conversation tree.
        
        Args:
            conversation_id: Unique ID for the conversation
        """
        self.id = conversation_id or str(uuid.uuid4())
        self._nodes: dict[str, BranchNode] = {}
        self._root_id: Optional[str] = None
        self._current_id: Optional[str] = None
        self._created_at = time.time()
        self._metadata: dict[str, Any] = {}
        
    def add_message(self, role: MessageRole, content: str, 
                    parent_id: Optional[str] = None,
                    branch_label: str = "",
                    metadata: dict[str, Any] = None) -> BranchNode:
        """
        Add a message to the tree.
        
        Args:
            role: Message role
            content: Message content
            parent_id: Parent node ID (None for root or auto-append)
            branch_label: Label for this branch
            metadata: Additional message metadata
            
        Returns:
            The created node
        """
        msg_id = str(uuid.uuid4())
        message = Message(
            id=msg_id,
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        # Determine parent
        if parent_id is None and self._current_id is not None:
            parent_id = self._current_id
            
        node = BranchNode(
            id=msg_id,
            message=message,
            parent_id=parent_id,
            branch_label=branch_label
        )
        
        self._nodes[msg_id] = node
        
        # Update parent's children
        if parent_id and parent_id in self._nodes:
            self._nodes[parent_id].children_ids.append(msg_id)
        elif self._root_id is None:
            self._root_id = msg_id
            
        self._current_id = msg_id
        return node
    
    def create_branch(self, from_node_id: str, 
                      new_content: str,
                      role: MessageRole = MessageRole.ASSISTANT,
                      label: str = "") -> BranchNode:
        """
        Create a new branch from an existing node.
        
        Args:
            from_node_id: Node to branch from (the parent)
            new_content: Content for the new branch's first message
            role: Role for the new message
            label: Branch label
            
        Returns:
            The new branch node
        """
        parent_node = self._nodes.get(from_node_id)
        if not parent_node:
            raise ValueError(f"Node {from_node_id} not found")
            
        # If branching from a user message, create alternate assistant response
        # If branching from assistant message, go to its parent (user message)
        branch_parent_id = from_node_id
        if parent_node.message.role == MessageRole.ASSISTANT:
            branch_parent_id = parent_node.parent_id
            
        # Auto-generate label if not provided
        if not label:
            parent = self._nodes.get(branch_parent_id)
            if parent:
                branch_count = len(parent.children_ids) + 1
                label = f"Branch {branch_count}"
                
        return self.add_message(
            role=role,
            content=new_content,
            parent_id=branch_parent_id,
            branch_label=label
        )
    
    def switch_to_branch(self, node_id: str):
        """
        Switch the current position to a different branch.
        
        Args:
            node_id: Node ID to switch to
        """
        if node_id in self._nodes:
            self._current_id = node_id
            
    def get_current_path(self) -> list[BranchNode]:
        """Get messages from root to current position."""
        path = []
        node_id = self._current_id
        
        while node_id:
            node = self._nodes.get(node_id)
            if node:
                path.insert(0, node)
                node_id = node.parent_id
            else:
                break
                
        return path
    
    def get_current_messages(self) -> list[Message]:
        """Get messages in chat format (for inference)."""
        return [node.message for node in self.get_current_path()]
    
    def get_branches_at(self, node_id: str) -> list[BranchNode]:
        """Get all branches from a node."""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[cid] for cid in node.children_ids if cid in self._nodes]
    
    def get_all_branches(self) -> list[tuple[str, list[BranchNode]]]:
        """Get all branch points in the tree."""
        branch_points = []
        for node_id, node in self._nodes.items():
            if len(node.children_ids) > 1:
                branches = self.get_branches_at(node_id)
                branch_points.append((node_id, branches))
        return branch_points
    
    def delete_branch(self, node_id: str, delete_descendants: bool = True):
        """
        Delete a branch.
        
        Args:
            node_id: Node to delete
            delete_descendants: Also delete all children
        """
        node = self._nodes.get(node_id)
        if not node:
            return
            
        # Remove from parent's children
        if node.parent_id and node.parent_id in self._nodes:
            parent = self._nodes[node.parent_id]
            if node_id in parent.children_ids:
                parent.children_ids.remove(node_id)
                
        # Delete descendants
        if delete_descendants:
            def delete_recursive(nid: str):
                n = self._nodes.get(nid)
                if n:
                    for child_id in n.children_ids.copy():
                        delete_recursive(child_id)
                    del self._nodes[nid]
            delete_recursive(node_id)
        else:
            del self._nodes[node_id]
            
        # Update current if needed
        if self._current_id == node_id:
            self._current_id = node.parent_id
            
    def get_tree_structure(self) -> dict:
        """Get tree structure for visualization."""
        def build_tree(node_id: str, depth: int = 0) -> dict:
            node = self._nodes.get(node_id)
            if not node:
                return {}
            return {
                "id": node.id,
                "role": node.message.role.value,
                "preview": node.message.content[:50] + "..." if len(node.message.content) > 50 else node.message.content,
                "label": node.branch_label,
                "is_current": node.id == self._current_id,
                "depth": depth,
                "children": [build_tree(cid, depth + 1) for cid in node.children_ids]
            }
            
        if self._root_id:
            return build_tree(self._root_id)
        return {}
    
    def to_dict(self) -> dict:
        """Serialize tree to dictionary."""
        return {
            "id": self.id,
            "root_id": self._root_id,
            "current_id": self._current_id,
            "created_at": self._created_at,
            "metadata": self._metadata,
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()}
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ConversationTree':
        """Deserialize tree from dictionary."""
        tree = cls(data.get("id"))
        tree._root_id = data.get("root_id")
        tree._current_id = data.get("current_id")
        tree._created_at = data.get("created_at", time.time())
        tree._metadata = data.get("metadata", {})
        tree._nodes = {
            nid: BranchNode.from_dict(ndata) 
            for nid, ndata in data.get("nodes", {}).items()
        }
        return tree


class BranchManager:
    """Manages multiple conversation trees with persistence."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize branch manager.
        
        Args:
            storage_dir: Directory for storing conversation trees
        """
        self._storage_dir = storage_dir or Path("data/conversations/trees")
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._trees: dict[str, ConversationTree] = {}
        self._active_tree_id: Optional[str] = None
        
    def create_tree(self, conversation_id: Optional[str] = None) -> ConversationTree:
        """Create a new conversation tree."""
        tree = ConversationTree(conversation_id)
        self._trees[tree.id] = tree
        self._active_tree_id = tree.id
        return tree
    
    def get_tree(self, tree_id: str) -> Optional[ConversationTree]:
        """Get a conversation tree by ID."""
        if tree_id in self._trees:
            return self._trees[tree_id]
        # Try loading from disk
        return self._load_tree(tree_id)
    
    def get_active_tree(self) -> Optional[ConversationTree]:
        """Get the currently active tree."""
        if self._active_tree_id:
            return self.get_tree(self._active_tree_id)
        return None
    
    def set_active_tree(self, tree_id: str):
        """Set the active tree."""
        if tree_id in self._trees or self._load_tree(tree_id):
            self._active_tree_id = tree_id
            
    def save_tree(self, tree_id: str):
        """Save a tree to disk."""
        tree = self._trees.get(tree_id)
        if tree:
            path = self._storage_dir / f"{tree_id}.json"
            with open(path, 'w') as f:
                json.dump(tree.to_dict(), f, indent=2)
            logger.debug(f"Saved conversation tree: {tree_id}")
            
    def _load_tree(self, tree_id: str) -> Optional[ConversationTree]:
        """Load a tree from disk."""
        path = self._storage_dir / f"{tree_id}.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                tree = ConversationTree.from_dict(data)
                self._trees[tree_id] = tree
                return tree
            except Exception as e:
                logger.error(f"Failed to load tree {tree_id}: {e}")
        return None
    
    def list_trees(self) -> list[dict]:
        """List all available conversation trees."""
        trees = []
        for path in self._storage_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                trees.append({
                    "id": data.get("id"),
                    "created_at": data.get("created_at"),
                    "metadata": data.get("metadata", {})
                })
            except (json.JSONDecodeError, OSError, KeyError) as e:
                logger.debug(f"Failed to load tree from {path}: {e}")
                continue
        return trees
    
    def delete_tree(self, tree_id: str):
        """Delete a conversation tree."""
        if tree_id in self._trees:
            del self._trees[tree_id]
        path = self._storage_dir / f"{tree_id}.json"
        if path.exists():
            path.unlink()
        if self._active_tree_id == tree_id:
            self._active_tree_id = None


# Singleton instance
_branch_manager: Optional[BranchManager] = None


def get_branch_manager(storage_dir: Optional[Path] = None) -> BranchManager:
    """Get the branch manager singleton."""
    global _branch_manager
    if _branch_manager is None:
        _branch_manager = BranchManager(storage_dir)
    return _branch_manager


__all__ = [
    'ConversationTree',
    'BranchNode',
    'BranchManager',
    'Message',
    'MessageRole',
    'get_branch_manager'
]
