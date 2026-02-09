"""
================================================================================
🧠 MEMORY-AUGMENTED INFERENCE ENGINE
================================================================================

This wraps EnigmaEngine to automatically retrieve and inject relevant memories
before generating responses. The AI now has REAL memory!

📍 FILE: enigma_engine/memory/augmented_engine.py
🏷️ TYPE: AI Memory Integration
🎯 MAIN CLASS: MemoryAugmentedEngine

┌─────────────────────────────────────────────────────────────────────────────┐
│  MEMORY-AUGMENTED FLOW:                                                     │
│                                                                             │
│  User: "What did I say about cats?"                                        │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────┐                                               │
│  │ 1. Search Vector DB     │ ← Find relevant past messages                 │
│  │ 2. Retrieve memories    │ ← "User loves cats, has 2 named..."          │
│  │ 3. Augment prompt       │ ← Prepend context to prompt                  │
│  │ 4. Generate response    │ ← AI now has context!                        │
│  │ 5. Store new message    │ ← Save this interaction                      │
│  └─────────────────────────┘                                               │
│         │                                                                   │
│         ▼                                                                   │
│  AI: "You mentioned you have 2 cats named Whiskers and Mittens!"           │
└─────────────────────────────────────────────────────────────────────────────┘

🔗 CONNECTED FILES:
    → USES:      enigma_engine/core/inference.py (EnigmaEngine)
    → USES:      enigma_engine/memory/vector_db.py (SimpleVectorDB)
    → USES:      enigma_engine/memory/embeddings.py (EmbeddingGenerator)
    → USES:      enigma_engine/memory/rag.py (RAGSystem)
    ← USED BY:   enigma_engine/gui/tabs/chat_tab.py (via generate_with_memory)
"""
from __future__ import annotations

import logging
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory-augmented generation."""
    
    # How many relevant memories to retrieve
    top_k_memories: int = 5
    
    # Maximum tokens for memory context
    max_memory_tokens: int = 500
    
    # Minimum similarity score for memory retrieval (0.0-1.0)
    min_similarity: float = 0.3
    
    # Auto-save messages to vector DB
    auto_store: bool = True
    
    # Memory types to search
    search_types: list[str] = field(default_factory=lambda: [
        "conversation", "fact", "preference", "instruction"
    ])
    
    # How to format memory context
    memory_template: str = "Relevant context from memory:\n{memories}\n\n"
    
    # Per-memory format
    memory_item_template: str = "- {content}"
    
    # Enable/disable memory augmentation
    enabled: bool = True


class MemoryAugmentedEngine:
    """
    Wraps EnigmaEngine to add automatic memory retrieval and storage.
    
    📖 WHAT THIS DOES:
    - Before generating: Searches for relevant past conversations
    - During generation: Injects context into the prompt  
    - After generating: Stores the new interaction for future recall
    
    📐 USAGE:
        from enigma_engine.memory.augmented_engine import MemoryAugmentedEngine
        
        engine = MemoryAugmentedEngine()
        
        # Generate with automatic memory retrieval
        response = engine.generate("What did I tell you about my cat?")
        # AI now recalls past conversations about cats!
        
        # Or use the chat method
        response = engine.chat("Remember my preferences")
    
    🔗 CONNECTS TO:
      → EnigmaEngine for text generation
      → VectorDB for semantic memory search
      → RAGSystem for retrieval-augmented generation
    """
    
    def __init__(
        self,
        engine: Any | None = None,
        vector_db: Any | None = None,
        embedding_generator: Any | None = None,
        config: MemoryConfig | None = None,
        model_name: str | None = None
    ):
        """
        Initialize memory-augmented engine.
        
        Args:
            engine: EnigmaEngine instance (creates one if None)
            vector_db: VectorDB instance for memory storage
            embedding_generator: For converting text to vectors
            config: Memory configuration
            model_name: Model name for per-model memory storage
        """
        self.config = config or MemoryConfig()
        self.model_name = model_name
        
        # ─────────────────────────────────────────────────────────────────────
        # INITIALIZE ENGINE
        # ─────────────────────────────────────────────────────────────────────
        if engine:
            self.engine = engine
        else:
            # Lazy import to avoid circular imports
            from ..core.inference import EnigmaEngine
            try:
                self.engine = EnigmaEngine()
            except FileNotFoundError:
                logger.warning("No trained model found, engine will not be available")
                self.engine = None
        
        # ─────────────────────────────────────────────────────────────────────
        # INITIALIZE EMBEDDING GENERATOR
        # ─────────────────────────────────────────────────────────────────────
        if embedding_generator:
            self.embedding_generator = embedding_generator
        else:
            try:
                from .embeddings import EmbeddingGenerator
                self.embedding_generator = EmbeddingGenerator(model="local")
            except Exception as e:
                logger.warning(f"Could not load embedding generator: {e}")
                self.embedding_generator = None
        
        # Get embedding dimension
        self._dim = 128  # Default
        if self.embedding_generator:
            self._dim = getattr(self.embedding_generator, 'dimension', 128)
        
        # ─────────────────────────────────────────────────────────────────────
        # INITIALIZE VECTOR DATABASE
        # ─────────────────────────────────────────────────────────────────────
        if vector_db:
            self.vector_db = vector_db
        else:
            try:
                # Per-model storage path
                from ..config import CONFIG
                from .vector_db import SimpleVectorDB
                if model_name:
                    models_dir = Path(CONFIG.get("models_dir", "models"))
                    db_path = models_dir / model_name / "memory_vectors.npz"
                else:
                    data_dir = Path(CONFIG.get("data_dir", "data"))
                    db_path = data_dir / "memory_vectors.npz"
                
                self.vector_db = SimpleVectorDB(dim=self._dim)
                
                # Load existing vectors if available
                if db_path.exists():
                    try:
                        self.vector_db.load(db_path)
                        logger.info(f"Loaded {self.vector_db.count()} memories from {db_path}")
                    except Exception as e:
                        logger.warning(f"Could not load vector DB: {e}")
                
                self._db_path = db_path
                
            except Exception as e:
                logger.warning(f"Could not initialize vector DB: {e}")
                self.vector_db = None
                self._db_path = None
        
        # ─────────────────────────────────────────────────────────────────────
        # INITIALIZE RAG SYSTEM
        # ─────────────────────────────────────────────────────────────────────
        self.rag_system = None
        if self.vector_db and self.embedding_generator:
            try:
                from .rag import RAGSystem
                self.rag_system = RAGSystem(
                    self.vector_db,
                    embedding_model="local"
                )
                self.rag_system._embedding_generator = self.embedding_generator
            except Exception as e:
                logger.warning(f"Could not initialize RAG: {e}")
        
        # ─────────────────────────────────────────────────────────────────────
        # CONVERSATION HISTORY (for current session)
        # ─────────────────────────────────────────────────────────────────────
        self.conversation_history: list[dict[str, str]] = []
        
        logger.info(
            f"MemoryAugmentedEngine initialized "
            f"(vectors={self.vector_db is not None}, "
            f"rag={self.rag_system is not None})"
        )
    
    def _embed_text(self, text: str):
        """Generate embedding for text."""
        if self.embedding_generator:
            return self.embedding_generator.embed(text)
        
        # Fallback: hash-based embedding
        import hashlib

        import numpy as np
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype('float32')
        if len(embedding) < self._dim:
            embedding = np.pad(embedding, (0, self._dim - len(embedding)))
        else:
            embedding = embedding[:self._dim]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def store_memory(
        self,
        content: str,
        memory_type: str = "conversation",
        metadata: dict | None = None
    ) -> str:
        """
        Store a message in the vector database for future retrieval.
        
        Args:
            content: Text content to store
            memory_type: Type of memory (conversation, fact, preference, etc.)
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        if not self.vector_db:
            logger.warning("No vector DB available, cannot store memory")
            return ""
        
        try:
            import numpy as np

            # Generate unique ID
            memory_id = f"mem_{int(time.time() * 1000)}_{hash(content) % 100000}"
            
            # Embed the content
            embedding = self._embed_text(content)
            embedding = np.array([embedding], dtype='float32')
            
            # Build metadata
            meta = {
                "content": content,
                "type": memory_type,
                "timestamp": time.time(),
                "model": self.model_name or "default",
                **(metadata or {})
            }
            
            # Store in vector DB
            self.vector_db.add(embedding, [memory_id], [meta])
            
            logger.debug(f"Stored memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return ""
    
    def retrieve_memories(
        self,
        query: str,
        top_k: int | None = None,
        memory_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant memories for a query.
        
        Args:
            query: Query text
            top_k: Number of memories to retrieve (uses config default)
            memory_types: Filter by memory types
            
        Returns:
            List of memory dictionaries with content and metadata
        """
        if not self.vector_db:
            return []
        
        top_k = top_k or self.config.top_k_memories
        memory_types = memory_types or self.config.search_types
        
        try:
            # Embed query
            query_embedding = self._embed_text(query)
            
            # Search vector DB
            results = self.vector_db.search(query_embedding, top_k=top_k * 2)
            
            # Filter and format results
            memories = []
            for mem_id, score, metadata in results:
                # Check similarity threshold
                if score < self.config.min_similarity:
                    continue
                
                # Check memory type filter
                mem_type = metadata.get("type", "conversation")
                if memory_types and mem_type not in memory_types:
                    continue
                
                memories.append({
                    "id": mem_id,
                    "content": metadata.get("content", ""),
                    "type": mem_type,
                    "score": score,
                    "timestamp": metadata.get("timestamp", 0),
                    "metadata": metadata
                })
                
                if len(memories) >= top_k:
                    break
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    def _format_memory_context(self, memories: list[dict]) -> str:
        """Format retrieved memories as context string."""
        if not memories:
            return ""
        
        # Format each memory
        memory_lines = []
        for mem in memories:
            content = mem.get("content", "")
            if content:
                line = self.config.memory_item_template.format(content=content)
                memory_lines.append(line)
        
        if not memory_lines:
            return ""
        
        # Combine and truncate to max tokens
        memories_text = "\n".join(memory_lines)
        
        # Simple truncation (could use tokenizer for better accuracy)
        max_chars = self.config.max_memory_tokens * 4  # ~4 chars per token
        if len(memories_text) > max_chars:
            memories_text = memories_text[:max_chars] + "..."
        
        return self.config.memory_template.format(memories=memories_text)
    
    def _augment_prompt(self, prompt: str) -> str:
        """
        Augment a prompt with relevant memory context.
        
        Args:
            prompt: Original user prompt
            
        Returns:
            Augmented prompt with memory context prepended
        """
        if not self.config.enabled:
            return prompt
        
        # Retrieve relevant memories
        memories = self.retrieve_memories(prompt)
        
        if not memories:
            return prompt
        
        # Format memory context
        context = self._format_memory_context(memories)
        
        if context:
            logger.debug(f"Augmented prompt with {len(memories)} memories")
            return context + prompt
        
        return prompt
    
    def generate(
        self,
        prompt: str,
        max_gen: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        use_memory: bool = True,
        store_interaction: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text with automatic memory retrieval.
        
        Args:
            prompt: Input prompt
            max_gen: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            use_memory: Whether to retrieve and inject memories
            store_interaction: Whether to store this interaction
            **kwargs: Additional arguments for EnigmaEngine
            
        Returns:
            Generated text
        """
        if not self.engine:
            return "[No model loaded]"
        
        # Augment prompt with memories
        augmented_prompt = self._augment_prompt(prompt) if use_memory else prompt
        
        # Generate response
        response = self.engine.generate(
            augmented_prompt,
            max_gen=max_gen,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs
        )
        
        # Store the interaction
        if store_interaction and self.config.auto_store:
            # Store user message
            self.store_memory(
                f"User: {prompt}",
                memory_type="conversation",
                metadata={"role": "user"}
            )
            # Store AI response
            self.store_memory(
                f"AI: {response}",
                memory_type="conversation", 
                metadata={"role": "assistant"}
            )
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def stream_generate(
        self,
        prompt: str,
        use_memory: bool = True,
        **kwargs
    ) -> Generator[str]:
        """
        Stream generate with memory augmentation.
        
        Args:
            prompt: Input prompt
            use_memory: Whether to use memory augmentation
            **kwargs: Arguments for EnigmaEngine.stream_generate
            
        Yields:
            Generated tokens
        """
        if not self.engine:
            yield "[No model loaded]"
            return
        
        # Augment prompt
        augmented_prompt = self._augment_prompt(prompt) if use_memory else prompt
        
        # Stream tokens
        full_response = ""
        for token in self.engine.stream_generate(augmented_prompt, **kwargs):
            full_response += token
            yield token
        
        # Store after completion
        if self.config.auto_store:
            self.store_memory(f"User: {prompt}", memory_type="conversation")
            self.store_memory(f"AI: {full_response}", memory_type="conversation")
        
        # Update history
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": full_response})
    
    def chat(
        self,
        message: str,
        system_prompt: str | None = None,
        use_memory: bool = True,
        **kwargs
    ) -> str:
        """
        Chat with memory-aware context.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            use_memory: Whether to use memory
            **kwargs: Additional generation arguments
            
        Returns:
            AI response
        """
        if not self.engine:
            return "[No model loaded]"
        
        # Build conversation context
        context_parts = []
        
        if system_prompt:
            context_parts.append(f"System: {system_prompt}")
        
        # Add recent conversation history
        for msg in self.conversation_history[-6:]:  # Last 3 exchanges
            role = msg["role"].capitalize()
            context_parts.append(f"{role}: {msg['content']}")
        
        # Add memory context
        if use_memory:
            memories = self.retrieve_memories(message)
            if memories:
                memory_context = self._format_memory_context(memories)
                context_parts.insert(0, memory_context)
        
        # Build full prompt
        context_parts.append(f"User: {message}")
        context_parts.append("AI:")
        
        full_prompt = "\n".join(context_parts)
        
        # Generate
        response = self.engine.generate(full_prompt, **kwargs)
        
        # Clean response
        response = response.strip()
        if response.startswith("AI:"):
            response = response[3:].strip()
        
        # Store
        if self.config.auto_store:
            self.store_memory(f"User: {message}", memory_type="conversation")
            self.store_memory(f"AI: {response}", memory_type="conversation")
        
        # Update history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def store_fact(self, fact: str, importance: float = 0.7) -> str:
        """Store an important fact for long-term memory."""
        return self.store_memory(
            fact,
            memory_type="fact",
            metadata={"importance": importance}
        )
    
    def store_preference(self, preference: str) -> str:
        """Store a user preference."""
        return self.store_memory(
            preference,
            memory_type="preference",
            metadata={"importance": 0.8}
        )
    
    def store_instruction(self, instruction: str) -> str:
        """Store a permanent instruction for the AI."""
        return self.store_memory(
            instruction,
            memory_type="instruction",
            metadata={"importance": 1.0, "permanent": True}
        )
    
    def clear_history(self):
        """Clear current session conversation history."""
        self.conversation_history = []
    
    def save_memories(self):
        """Save vector database to disk."""
        if self.vector_db and self._db_path:
            try:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                self.vector_db.save(self._db_path)
                logger.info(f"Saved {self.vector_db.count()} memories to {self._db_path}")
            except Exception as e:
                logger.error(f"Failed to save memories: {e}")
    
    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        return {
            "total_memories": self.vector_db.count() if self.vector_db else 0,
            "session_messages": len(self.conversation_history),
            "embedding_dim": self._dim,
            "config": {
                "top_k": self.config.top_k_memories,
                "min_similarity": self.config.min_similarity,
                "auto_store": self.config.auto_store,
                "enabled": self.config.enabled
            }
        }
    
    def __del__(self):
        """Save memories on cleanup."""
        try:
            self.save_memories()
        except Exception:
            pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_engine: MemoryAugmentedEngine | None = None


def get_memory_engine(model_name: str | None = None) -> MemoryAugmentedEngine:
    """
    Get or create the default memory-augmented engine.
    
    Args:
        model_name: Optional model name for per-model memory
        
    Returns:
        MemoryAugmentedEngine instance
    """
    global _default_engine
    
    if _default_engine is None:
        _default_engine = MemoryAugmentedEngine(model_name=model_name)
    
    return _default_engine


def generate_with_memory(
    prompt: str,
    use_memory: bool = True,
    **kwargs
) -> str:
    """
    Convenience function for memory-augmented generation.
    
    Args:
        prompt: Input prompt
        use_memory: Whether to use memory augmentation
        **kwargs: Additional generation arguments
        
    Returns:
        Generated text
    """
    engine = get_memory_engine()
    return engine.generate(prompt, use_memory=use_memory, **kwargs)


def chat_with_memory(message: str, **kwargs) -> str:
    """
    Convenience function for memory-augmented chat.
    
    Args:
        message: User message
        **kwargs: Additional arguments
        
    Returns:
        AI response
    """
    engine = get_memory_engine()
    return engine.chat(message, **kwargs)


def store_memory(content: str, memory_type: str = "conversation") -> str:
    """
    Store a memory in the vector database.
    
    Args:
        content: Memory content
        memory_type: Type (conversation, fact, preference, instruction)
        
    Returns:
        Memory ID
    """
    engine = get_memory_engine()
    return engine.store_memory(content, memory_type)


def search_memories(query: str, top_k: int = 5) -> list[dict]:
    """
    Search for relevant memories.
    
    Args:
        query: Search query
        top_k: Number of results
        
    Returns:
        List of memory dictionaries
    """
    engine = get_memory_engine()
    return engine.retrieve_memories(query, top_k=top_k)
