"""
================================================================================
💾 CONVERSATION MANAGER - THE MEMORY VAULT
================================================================================

This is where the AI stores its memories! Conversations are saved to disk
and pushed to a vector database for intelligent semantic search.

📍 FILE: forge_ai/memory/manager.py
🏷️ TYPE: Conversation Storage & Retrieval
🎯 MAIN CLASS: ConversationManager

┌─────────────────────────────────────────────────────────────────────────────┐
│  MEMORY FLOW:                                                               │
│                                                                             │
│  User: "Tell me about cats"                                                │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────────────┐                                                   │
│  │ ConversationManager │                                                   │
│  └──────────┬──────────┘                                                   │
│             │                                                               │
│     ┌───────┴───────┐                                                       │
│     ▼               ▼                                                       │
│  [JSON File]   [VectorDB]                                                  │
│  data/conv/    semantic                                                     │
│  my_chat.json  embeddings                                                   │
└─────────────────────────────────────────────────────────────────────────────┘

📁 STORAGE LOCATION: data/conversations/*.json

🔗 CONNECTED FILES:
    → USES:      forge_ai/memory/vector_db.py (SimpleVectorDB for search)
    → USES:      forge_ai/memory/memory_db.py (add_memory function)
    → USES:      forge_ai/config/ (CONFIG for paths)
    ← USED BY:   forge_ai/gui/tabs/chat_tab.py (save/load conversations)
    ← USED BY:   forge_ai/gui/enhanced_window.py (history panel)

📖 USAGE:
    from forge_ai.memory.manager import ConversationManager
    
    manager = ConversationManager()
    
    # Save conversation
    messages = [
        {"role": "user", "text": "Hello!", "ts": 12345},
        {"role": "ai", "text": "Hi there!", "ts": 12346}
    ]
    manager.save_conversation("my_chat", messages)
    
    # Load conversation
    data = manager.load_conversation("my_chat")

📖 SEE ALSO:
    • forge_ai/memory/vector_db.py  - Semantic search (find by meaning)
    • forge_ai/memory/embeddings.py - Convert text to vectors
    • forge_ai/memory/rag.py        - Retrieval-augmented generation
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


# =============================================================================
# 💾 CONVERSATION MANAGER CLASS
# =============================================================================

class ConversationManager:
    """
    Manages conversations and provides long-term memory capabilities.
    
    📖 WHAT THIS DOES:
    The ConversationManager is your AI's MEMORY VAULT!
    It handles saving and loading chat conversations, plus integrates
    with the vector database for semantic (meaning-based) search.
    
    📐 TWO STORAGE SYSTEMS:
    
    ┌──────────────────────────────────────────────────────────────────────┐
    │  1. JSON FILES (Simple Storage)                                      │
    │     Location: data/conversations/*.json                              │
    │     Format: {"name": "...", "saved_at": ..., "messages": [...]}     │
    │     Use: Load/save entire conversations by name                      │
    │                                                                      │
    │  2. VECTOR DATABASE (Semantic Search)                               │
    │     What: Stores text as mathematical vectors (embeddings)          │
    │     Use: Find similar messages by MEANING, not just keywords        │
    │     Example: "pets" matches "I have a cat" even without "pet" word  │
    └──────────────────────────────────────────────────────────────────────┘
    
    📐 MESSAGE FORMAT:
    Each message is a dictionary with these fields:
    {
        "role": "user" or "ai",    # Who said it
        "text": "Hello!",           # What was said
        "ts": 1699999999            # Timestamp (Unix seconds)
    }
    
    🔗 CONNECTS TO:
      → vector_db.py: For semantic search
      → memory_db.py: For long-term memory storage
      ← chat_tab.py: Saves/loads conversations
      ← enhanced_window.py: Shows conversation history
    
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
        # ─────────────────────────────────────────────────────────────────────
        # STORAGE PATHS
        # ─────────────────────────────────────────────────────────────────────
        self.conv_dir = CONV_DIR  # data/conversations/
        
        # ─────────────────────────────────────────────────────────────────────
        # VECTOR DATABASE: For semantic search (find by meaning)
        # dim=128 means each text is converted to a 128-number vector
        # ─────────────────────────────────────────────────────────────────────
        self.vector_db = vector_db or SimpleVectorDB(dim=CONFIG.get("embed_dim", 128))

    def save_conversation(self, name: str, messages: List[Dict[str, Any]]) -> str:
        """
        Save a conversation to disk and optionally to memory DB.
        
        📖 WHAT THIS DOES:
        1. Sanitizes the name (removes unsafe characters)
        2. Writes JSON file to data/conversations/{name}.json
        3. Pushes each message to long-term memory DB
        
        📐 FILE FORMAT:
        {
            "name": "My Chat",
            "saved_at": 1699999999.123,
            "messages": [
                {"role": "user", "text": "Hello!", "ts": 12345},
                {"role": "ai", "text": "Hi there!", "ts": 12346}
            ]
        }
        
        Args:
            name: Name of the conversation (will be sanitized)
            messages: List of message dictionaries with keys: role, text, ts
            
        Returns:
            Path to saved conversation file
            
        Raises:
            ValueError: If name is empty or contains invalid characters
            IOError: If file cannot be written
        """
        # ─────────────────────────────────────────────────────────────────────
        # VALIDATION: Make sure we have a valid name
        # ─────────────────────────────────────────────────────────────────────
        if not name:
            raise ValueError("Conversation name cannot be empty")
        
        # ─────────────────────────────────────────────────────────────────────
        # SANITIZE FILENAME: Remove dangerous characters
        # "My Chat! @#$" → "My_Chat"
        # This prevents path injection attacks and filesystem errors
        # ─────────────────────────────────────────────────────────────────────
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')  # Replace spaces with underscores
        if not safe_name:
            raise ValueError(f"Invalid conversation name: {name}")
        
        # ─────────────────────────────────────────────────────────────────────
        # PREPARE DATA: Package messages with metadata
        # ─────────────────────────────────────────────────────────────────────
        fname = self.conv_dir / f"{safe_name}.json"
        data = {"name": name, "saved_at": time.time(), "messages": messages}
        
        # ─────────────────────────────────────────────────────────────────────
        # WRITE FILE: Save to disk as JSON
        # ─────────────────────────────────────────────────────────────────────
        try:
            fname.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except (IOError, OSError) as e:
            raise IOError(f"Failed to save conversation to {fname}: {e}") from e
        
        # ─────────────────────────────────────────────────────────────────────
        # PUSH TO MEMORY DB: For long-term semantic search
        # This lets the AI remember past conversations and find relevant ones
        # ─────────────────────────────────────────────────────────────────────
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
        
        📖 WHAT THIS DOES:
        Reads the JSON file for the named conversation and returns it.
        
        Args:
            name: Name of the conversation
            
        Returns:
            Dictionary containing conversation data:
            {"name": "...", "saved_at": ..., "messages": [...]}
            
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
        
        📖 WHAT THIS RETURNS:
        A list of conversation names (without .json extension).
        Sorted so the most recently modified conversation is first.
        
        📐 EXAMPLE:
        ["chat_today", "project_discussion", "old_chat"]
        
        Returns:
            List of conversation names
        """
        try:
            return [
                p.stem  # Get filename without .json extension
                for p in sorted(
                    self.conv_dir.glob("*.json"),  # Find all JSON files
                    key=lambda x: x.stat().st_mtime,  # Sort by modification time
                    reverse=True  # Newest first
                )
            ]
        except OSError as e:
            print(f"Warning: Error listing conversations: {e}")
            return []

    # =========================================================================
    # 🔍 VECTOR DATABASE METHODS - Semantic Search
    # =========================================================================

    def add_to_vector_db(self, id_: str, vector: Any) -> None:
        """
        Add a vector to the vector database.
        
        📖 WHAT ARE VECTORS?
        Vectors are lists of numbers that represent the MEANING of text.
        Similar meanings have similar vectors (close together in space).
        
        📐 EXAMPLE:
        "I love cats" → [0.2, 0.8, 0.1, ...] (128 numbers)
        "I adore felines" → [0.21, 0.79, 0.12, ...] (similar!)
        "The weather is nice" → [0.9, 0.1, 0.3, ...] (different)
        
        Args:
            id_: Identifier for the vector (usually the original text)
            vector: Vector to add (list of floats)
        """
        if not id_:
            raise ValueError("Vector ID cannot be empty")
        self.vector_db.add(vector, id_)

    def search_vectors(self, query_vec: Any, topk: int = 5) -> List[Any]:
        """
        Search for similar vectors in the database.
        
        📖 HOW SEMANTIC SEARCH WORKS:
        1. Your query text is converted to a vector
        2. We find the K vectors in the database closest to your query
        3. Return the text/IDs associated with those vectors
        
        📐 DISTANCE METRICS:
        We measure "closeness" using cosine similarity:
        - 1.0 = identical meaning
        - 0.0 = completely unrelated
        - -1.0 = opposite meaning
        
        Args:
            query_vec: Query vector (same dimension as stored vectors)
            topk: Number of results to return
            
        Returns:
            List of (id, score) tuples, sorted by similarity
        """
        if topk <= 0:
            raise ValueError("topk must be positive")
        return self.vector_db.search(query_vec, topk=topk)
