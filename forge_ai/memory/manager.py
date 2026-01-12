"""
Conversation manager and long-term memory bridge.
Stores conversations to disk (json), pushes embeddings into SimpleVectorDB (if provided),
and uses memory_db for short message storage.
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from .vector_db import SimpleVectorDB
from ..config import CONFIG
from ..memory.memory_db import add_memory

CONV_DIR = Path(CONFIG["data_dir"]) / "conversations"
CONV_DIR.mkdir(parents=True, exist_ok=True)

class ConversationManager:
    """
    Manages conversations and provides long-term memory capabilities.
    
    Attributes:
        conv_dir: Directory for storing conversation files
        vector_db: Vector database for semantic search
    """
    
    def __init__(self, vector_db: Optional[SimpleVectorDB] = None):
        """
        Initialize the conversation manager.
        
        Args:
            vector_db: Optional vector database instance. If None, creates a new one.
        """
        self.conv_dir = CONV_DIR
        self.vector_db = vector_db or SimpleVectorDB(dim=CONFIG.get("embed_dim", 128))

    def save_conversation(self, name: str, messages: List[Dict[str, Any]]) -> str:
        """
        Save a conversation to disk and optionally to memory DB.
        
        Args:
            name: Name of the conversation
            messages: List of message dictionaries with keys: role, text, ts
            
        Returns:
            Path to saved conversation file
            
        Raises:
            ValueError: If name is empty or contains invalid characters
            IOError: If file cannot be written
        """
        if not name:
            raise ValueError("Conversation name cannot be empty")
        
        # Sanitize filename
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')  # Replace spaces with underscores
        if not safe_name:
            raise ValueError(f"Invalid conversation name: {name}")
        
        fname = self.conv_dir / f"{safe_name}.json"
        data = {"name": name, "saved_at": time.time(), "messages": messages}
        
        try:
            fname.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except (IOError, OSError) as e:
            raise IOError(f"Failed to save conversation to {fname}: {e}") from e
        
        # Optionally push to memory DB
        for m in messages:
            try:
                add_memory(m.get("text", ""), source=m.get("role", "user"), meta={"conv": name})
            except Exception as e:
                # Log but don't fail the save operation
                print(f"Warning: Failed to add message to memory DB: {e}")
        
        return str(fname)

    def load_conversation(self, name: str) -> Dict[str, Any]:
        """
        Load a conversation from disk.
        
        Args:
            name: Name of the conversation
            
        Returns:
            Dictionary containing conversation data
            
        Raises:
            ValueError: If name is empty
            FileNotFoundError: If conversation file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        if not name:
            raise ValueError("Conversation name cannot be empty")
        
        fname = self.conv_dir / f"{name}.json"
        if not fname.exists():
            raise FileNotFoundError(f"Conversation not found: {fname}")
        
        try:
            return json.loads(fname.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in conversation file {fname}",
                e.doc,
                e.pos
            ) from e

    def list_conversations(self) -> List[str]:
        """
        List all saved conversations, sorted by modification time (newest first).
        
        Returns:
            List of conversation names
        """
        try:
            return [
                p.stem 
                for p in sorted(
                    self.conv_dir.glob("*.json"), 
                    key=lambda x: x.stat().st_mtime, 
                    reverse=True
                )
            ]
        except OSError as e:
            print(f"Warning: Error listing conversations: {e}")
            return []

    def add_to_vector_db(self, id_: str, vector: Any) -> None:
        """
        Add a vector to the vector database.
        
        Args:
            id_: Identifier for the vector
            vector: Vector to add
        """
        if not id_:
            raise ValueError("Vector ID cannot be empty")
        self.vector_db.add(vector, id_)

    def search_vectors(self, query_vec: Any, topk: int = 5) -> List[Any]:
        """
        Search for similar vectors in the database.
        
        Args:
            query_vec: Query vector
            topk: Number of results to return
            
        Returns:
            List of search results
        """
        if topk <= 0:
            raise ValueError("topk must be positive")
        return self.vector_db.search(query_vec, topk=topk)
