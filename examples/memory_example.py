#!/usr/bin/env python3
"""
ForgeAI Memory System Example
=============================

Complete example showing how to use ForgeAI's memory system including:
- Conversation storage and retrieval
- Vector database for semantic search
- RAG (Retrieval-Augmented Generation)
- Embeddings for text similarity

The memory system allows your AI to remember past conversations and find
relevant context using semantic search (finding by meaning, not keywords).

Dependencies:
    pip install numpy sentence-transformers faiss-cpu  # Optional: for FAISS

Run: python examples/memory_example.py
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


# =============================================================================
# Simulated Memory Classes (for standalone testing)
# In real use, import from forge_ai.memory
# =============================================================================

SIMULATED = True  # Set to False when using actual ForgeAI

@dataclass
class Memory:
    """A single memory entry."""
    id: str
    content: str
    memory_type: str = "short_term"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleVectorDB:
    """
    Pure Python vector database for semantic search.
    
    Converts text to vectors and finds similar items by 
    mathematical distance (cosine similarity).
    """
    
    def __init__(self, dim: int = 384):
        """
        Initialize vector database.
        
        Args:
            dim: Vector dimension (384 for sentence-transformers)
        """
        self.dim = dim
        self.vectors: List[np.ndarray] = []
        self.ids: List[str] = []
        self.metadata: List[Dict] = []
    
    def add(self, vectors: np.ndarray, ids: List[str], 
            metadata: Optional[List[Dict]] = None) -> None:
        """Add vectors to the database."""
        for i, (vec, id_) in enumerate(zip(vectors, ids)):
            self.vectors.append(vec)
            self.ids.append(id_)
            meta = metadata[i] if metadata else {}
            self.metadata.append(meta)
        print(f"Added {len(ids)} vectors to database")
    
    def search(self, query_vector: np.ndarray, 
               top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for similar vectors using cosine similarity."""
        if not self.vectors:
            return []
        
        # Calculate cosine similarity with all vectors
        scores = []
        for i, vec in enumerate(self.vectors):
            # Cosine similarity
            dot = np.dot(query_vector, vec)
            norm = np.linalg.norm(query_vector) * np.linalg.norm(vec)
            similarity = dot / norm if norm > 0 else 0
            scores.append((self.ids[i], similarity, self.metadata[i]))
        
        # Sort by similarity (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def count(self) -> int:
        """Get number of vectors."""
        return len(self.vectors)
    
    def save(self, path: Path) -> None:
        """Save database to disk."""
        data = {
            'dim': self.dim,
            'vectors': [v.tolist() for v in self.vectors],
            'ids': self.ids,
            'metadata': self.metadata
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Saved vector database to {path}")
    
    def load(self, path: Path) -> None:
        """Load database from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.dim = data['dim']
        self.vectors = [np.array(v) for v in data['vectors']]
        self.ids = data['ids']
        self.metadata = data['metadata']
        print(f"Loaded {self.count()} vectors from {path}")


class EmbeddingGenerator:
    """
    Generate text embeddings (convert text to vectors).
    
    Uses sentence-transformers if available, otherwise falls
    back to a simple hash-based method for testing.
    """
    
    def __init__(self, model: str = "local"):
        """
        Initialize embedding generator.
        
        Args:
            model: "local" for sentence-transformers, 
                   "openai" for OpenAI embeddings
        """
        self.model_name = model
        self.model = None
        self.dim = 384  # Default dimension
        
        if model == "local":
            self._load_local_model()
    
    def _load_local_model(self):
        """Try to load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dim = 384
            print("Loaded sentence-transformers model")
        except ImportError:
            print("sentence-transformers not installed, using fallback")
            self.model = None
    
    def embed(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        if self.model is not None:
            return self.model.encode(text)
        else:
            # Fallback: simple hash-based embedding
            return self._hash_embed(text)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Convert multiple texts to embeddings."""
        if self.model is not None:
            return self.model.encode(texts)
        else:
            return np.array([self._hash_embed(t) for t in texts])
    
    def _hash_embed(self, text: str) -> np.ndarray:
        """Simple hash-based embedding for testing without dependencies."""
        import hashlib
        # Create deterministic pseudo-random embedding from text
        h = hashlib.sha256(text.lower().encode()).digest()
        # Expand hash to embedding dimension
        np.random.seed(int.from_bytes(h[:4], 'big'))
        embedding = np.random.randn(self.dim).astype(np.float32)
        # Normalize
        return embedding / np.linalg.norm(embedding)


class ConversationManager:
    """
    Manages conversation storage and retrieval.
    
    Saves conversations to JSON files and optionally indexes
    them in a vector database for semantic search.
    """
    
    def __init__(self, data_dir: str = "data/conversations", 
                 use_vector_db: bool = True):
        """
        Initialize conversation manager.
        
        Args:
            data_dir: Directory to store conversation files
            use_vector_db: Whether to enable semantic search
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_vector_db = use_vector_db
        if use_vector_db:
            self.embedder = EmbeddingGenerator()
            self.vector_db = SimpleVectorDB(dim=self.embedder.dim)
        else:
            self.embedder = None
            self.vector_db = None
    
    def save_conversation(self, name: str, messages: List[Dict]) -> Path:
        """
        Save a conversation to disk.
        
        Args:
            name: Conversation name/identifier
            messages: List of message dicts with 'role', 'text', 'ts' keys
            
        Returns:
            Path to saved file
        """
        file_path = self.data_dir / f"{name}.json"
        
        data = {
            'name': name,
            'created': time.time(),
            'messages': messages
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved conversation '{name}' with {len(messages)} messages")
        
        # Index messages in vector database for semantic search
        if self.use_vector_db:
            self._index_messages(name, messages)
        
        return file_path
    
    def _index_messages(self, conv_name: str, messages: List[Dict]):
        """Index messages in vector database."""
        texts = []
        ids = []
        metadata = []
        
        for i, msg in enumerate(messages):
            text = msg.get('text', '')
            if text:
                texts.append(text)
                ids.append(f"{conv_name}_{i}")
                metadata.append({
                    'content': text,
                    'role': msg.get('role', 'unknown'),
                    'conversation': conv_name,
                    'timestamp': msg.get('ts', time.time())
                })
        
        if texts:
            embeddings = self.embedder.embed_batch(texts)
            self.vector_db.add(embeddings, ids, metadata)
    
    def load_conversation(self, name: str) -> Optional[Dict]:
        """Load a conversation from disk."""
        file_path = self.data_dir / f"{name}.json"
        
        if not file_path.exists():
            print(f"Conversation '{name}' not found")
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded conversation '{name}'")
        return data
    
    def list_conversations(self) -> List[str]:
        """List all saved conversations."""
        files = self.data_dir.glob("*.json")
        return [f.stem for f in files]
    
    def search_memories(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for relevant memories using semantic search.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of (id, similarity_score, metadata) tuples
        """
        if not self.use_vector_db:
            print("Vector database not enabled")
            return []
        
        query_embedding = self.embedder.embed(query)
        results = self.vector_db.search(query_embedding, top_k)
        
        return results


class RAGSystem:
    """
    Retrieval-Augmented Generation system.
    
    Combines memory retrieval with prompt augmentation to give
    the AI relevant context when answering questions.
    """
    
    def __init__(self, conversation_manager: ConversationManager):
        """
        Initialize RAG system.
        
        Args:
            conversation_manager: Manager with indexed conversations
        """
        self.manager = conversation_manager
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Memory]:
        """Retrieve relevant memories for a query."""
        results = self.manager.search_memories(query, top_k)
        
        memories = []
        for id_, score, meta in results:
            memory = Memory(
                id=id_,
                content=meta.get('content', ''),
                metadata={'score': score, **meta}
            )
            memories.append(memory)
        
        return memories
    
    def augment_prompt(self, query: str, memories: List[Memory], 
                       max_context_length: int = 2000) -> str:
        """
        Augment a prompt with relevant context from memories.
        
        Args:
            query: Original user query
            memories: Retrieved relevant memories
            max_context_length: Maximum characters of context
            
        Returns:
            Augmented prompt with context
        """
        if not memories:
            return query
        
        # Build context string
        context_parts = []
        total_length = 0
        
        for mem in memories:
            content = mem.content
            if total_length + len(content) > max_context_length:
                break
            context_parts.append(f"- {content}")
            total_length += len(content) + 3
        
        context = "\n".join(context_parts)
        
        # Create augmented prompt
        augmented = f"""Based on these relevant past conversations:
{context}

Please answer: {query}"""
        
        return augmented
    
    def query(self, user_query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Full RAG query - retrieve context and augment prompt.
        
        Args:
            user_query: User's question
            top_k: Number of memories to retrieve
            
        Returns:
            Dict with augmented_prompt, memories, and timing info
        """
        start_time = time.time()
        
        # Retrieve relevant memories
        memories = self.retrieve(user_query, top_k)
        
        # Augment prompt with context
        augmented_prompt = self.augment_prompt(user_query, memories)
        
        retrieval_time = time.time() - start_time
        
        return {
            'original_query': user_query,
            'augmented_prompt': augmented_prompt,
            'memories': memories,
            'retrieval_time': retrieval_time
        }


# =============================================================================
# Example Usage
# =============================================================================

def example_basic_memory():
    """Basic conversation storage and retrieval."""
    print("\n" + "="*60)
    print("Example 1: Basic Conversation Storage")
    print("="*60)
    
    # Initialize manager
    manager = ConversationManager(
        data_dir="data/conversations_example",
        use_vector_db=True
    )
    
    # Create sample conversations
    conversation1 = [
        {"role": "user", "text": "Tell me about cats", "ts": time.time()},
        {"role": "ai", "text": "Cats are fascinating feline companions! They sleep 12-16 hours per day and have excellent night vision.", "ts": time.time()},
        {"role": "user", "text": "What do they eat?", "ts": time.time()},
        {"role": "ai", "text": "Cats are obligate carnivores, meaning they need meat to survive. A balanced diet includes protein from fish, chicken, or commercial cat food.", "ts": time.time()}
    ]
    
    conversation2 = [
        {"role": "user", "text": "How do I train a dog?", "ts": time.time()},
        {"role": "ai", "text": "Dog training uses positive reinforcement. Reward good behavior with treats and praise. Consistency is key!", "ts": time.time()},
        {"role": "user", "text": "What commands should I teach first?", "ts": time.time()},
        {"role": "ai", "text": "Start with basic commands: sit, stay, come, and down. These form the foundation for more advanced training.", "ts": time.time()}
    ]
    
    # Save conversations
    manager.save_conversation("cats_chat", conversation1)
    manager.save_conversation("dogs_chat", conversation2)
    
    # List all conversations
    print(f"\nSaved conversations: {manager.list_conversations()}")
    
    # Load a conversation
    loaded = manager.load_conversation("cats_chat")
    if loaded:
        print(f"Loaded {len(loaded['messages'])} messages from 'cats_chat'")


def example_semantic_search():
    """Search memories by meaning, not just keywords."""
    print("\n" + "="*60)
    print("Example 2: Semantic Search")
    print("="*60)
    
    # Initialize with vector database
    manager = ConversationManager(
        data_dir="data/conversations_example",
        use_vector_db=True
    )
    
    # Add some memories to search
    memories = [
        {"role": "ai", "text": "Cats are independent pets that groom themselves", "ts": time.time()},
        {"role": "ai", "text": "Python is a programming language great for AI", "ts": time.time()},
        {"role": "ai", "text": "Dogs are loyal companions that need daily walks", "ts": time.time()},
        {"role": "ai", "text": "JavaScript runs in web browsers", "ts": time.time()},
        {"role": "ai", "text": "Feline animals have retractable claws", "ts": time.time()}
    ]
    
    manager.save_conversation("mixed_topics", memories)
    
    # Search by meaning - should find "Cats" and "Feline" entries even
    # though we search for "kitty" (semantically similar)
    print("\nSearching for 'kitty pets'...")
    results = manager.search_memories("kitty pets", top_k=3)
    
    for id_, score, meta in results:
        print(f"  Score: {score:.3f} - {meta.get('content', '')[:60]}...")


def example_rag():
    """Retrieval-Augmented Generation example."""
    print("\n" + "="*60)
    print("Example 3: RAG (Retrieval-Augmented Generation)")
    print("="*60)
    
    # Initialize manager and RAG
    manager = ConversationManager(
        data_dir="data/conversations_example",
        use_vector_db=True
    )
    
    # Index some knowledge
    knowledge = [
        {"role": "ai", "text": "ForgeAI supports multiple model sizes from nano (1M params) to omega (70B+ params)", "ts": time.time()},
        {"role": "ai", "text": "Training requires at least 1000 lines of text data for good results", "ts": time.time()},
        {"role": "ai", "text": "The module manager prevents loading conflicting modules automatically", "ts": time.time()},
        {"role": "ai", "text": "Image generation supports Stable Diffusion locally or DALL-E via API", "ts": time.time()},
        {"role": "ai", "text": "Voice input uses speech recognition, voice output uses TTS engines", "ts": time.time()}
    ]
    
    manager.save_conversation("forgeai_knowledge", knowledge)
    
    # Create RAG system
    rag = RAGSystem(manager)
    
    # Query with context retrieval
    print("\nRAG Query: 'How do I generate images?'")
    result = rag.query("How do I generate images?", top_k=2)
    
    print(f"Retrieval time: {result['retrieval_time']*1000:.1f}ms")
    print(f"Found {len(result['memories'])} relevant memories")
    print("\nAugmented prompt:")
    print("-" * 40)
    print(result['augmented_prompt'])


def example_faiss_backend():
    """Using FAISS for faster vector search (optional)."""
    print("\n" + "="*60)
    print("Example 4: FAISS Vector Database (if installed)")
    print("="*60)
    
    try:
        import faiss
        
        # Create FAISS index
        dim = 384
        index = faiss.IndexFlatL2(dim)  # L2 distance (Euclidean)
        
        # Generate some test vectors
        np.random.seed(42)
        vectors = np.random.random((100, dim)).astype('float32')
        
        # Add to index
        index.add(vectors)
        
        print(f"FAISS index created with {index.ntotal} vectors")
        
        # Search
        query = np.random.random((1, dim)).astype('float32')
        distances, indices = index.search(query, k=5)
        
        print(f"Found {len(indices[0])} nearest neighbors")
        print(f"Indices: {indices[0]}")
        print(f"Distances: {distances[0]}")
        
    except ImportError:
        print("FAISS not installed. Install with: pip install faiss-cpu")
        print("FAISS provides much faster search for large vector databases.")


def example_with_forge():
    """Using actual ForgeAI memory system (not simulated)."""
    print("\n" + "="*60)
    print("Example 5: Integration with ForgeAI")
    print("="*60)
    
    if SIMULATED:
        print("Running in simulated mode. To use real ForgeAI:")
        print("  from forge_ai.memory.manager import ConversationManager")
        print("  from forge_ai.memory.vector_db import SimpleVectorDB, FAISSVectorDB")
        print("  from forge_ai.memory.rag import RAGSystem")
        print("  from forge_ai.memory.embeddings import EmbeddingGenerator")
        return
    
    # Real ForgeAI usage
    try:
        from forge_ai.memory.manager import ConversationManager
        from forge_ai.memory.rag import RAGSystem
        
        # Use ForgeAI's actual memory system
        manager = ConversationManager()
        rag = RAGSystem(manager.vector_db)
        
        # Your conversations are automatically indexed
        # Just use the chat interface and memories are stored!
        
    except ImportError as e:
        print(f"ForgeAI not available: {e}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("ForgeAI Memory System Examples")
    print("="*60)
    
    # Run all examples
    example_basic_memory()
    example_semantic_search()
    example_rag()
    example_faiss_backend()
    example_with_forge()
    
    print("\n" + "="*60)
    print("Memory System Summary:")
    print("="*60)
    print("""
Key Concepts:
    
1. Conversation Storage:
   - Save chats to JSON files
   - Load and list saved conversations
   - Automatic timestamping
    
2. Vector Database:
   - Convert text to vectors (embeddings)
   - Find similar content by meaning
   - Supports SimpleVectorDB or FAISS
    
3. Semantic Search:
   - Search by meaning, not keywords
   - "kitty" finds "cat" and "feline"
   - Uses cosine similarity
    
4. RAG (Retrieval-Augmented Generation):
   - Retrieve relevant past context
   - Augment prompts with memories
   - Better answers with context
    
5. Backends:
   - SimpleVectorDB: Pure Python, no deps
   - FAISSVectorDB: Fast, for large data
   - Can add cloud backends (Pinecone)

For real ForgeAI integration:
    from forge_ai.memory.manager import ConversationManager
    from forge_ai.memory.vector_db import SimpleVectorDB
    from forge_ai.memory.rag import RAGSystem
""")
