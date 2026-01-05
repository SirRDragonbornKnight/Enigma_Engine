"""
RAG (Retrieval-Augmented Generation) System for Enigma Engine
Integrates memory retrieval with LLM generation for context-aware responses.
"""
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .categorization import Memory, MemoryType, MemoryCategorization
from .vector_db import VectorDBInterface

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result from RAG query."""
    query: str
    context_memories: List[Memory]
    augmented_prompt: str
    retrieval_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGSystem:
    """Retrieval-Augmented Generation for memory-enhanced responses."""
    
    def __init__(
        self,
        vector_db: VectorDBInterface,
        memory_system: Optional[MemoryCategorization] = None,
        embedding_model: str = "local"
    ):
        """
        Initialize RAG system.
        
        Args:
            vector_db: Vector database for similarity search
            memory_system: Optional memory categorization system for metadata
            embedding_model: Embedding model to use ('local' or API name)
        """
        self.vector_db = vector_db
        self.memory_system = memory_system
        self.embedding_model = embedding_model
        self._embedding_generator = None
        
        # Lazy load embedding generator
        if embedding_model == "local":
            try:
                from .embeddings import EmbeddingGenerator
                self._embedding_generator = EmbeddingGenerator(model="local")
            except ImportError:
                logger.warning("Could not load local embedding generator")
    
    def _get_embedding(self, text: str):
        """Get embedding for text."""
        if self._embedding_generator:
            return self._embedding_generator.embed(text)
        else:
            # Fallback: simple hash-based embedding
            import hashlib
            import numpy as np
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()
            embedding = np.frombuffer(hash_bytes[:self.vector_db.dim * 4], dtype=np.float32)
            # Normalize
            if len(embedding) < self.vector_db.dim:
                embedding = np.pad(embedding, (0, self.vector_db.dim - len(embedding)))
            else:
                embedding = embedding[:self.vector_db.dim]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Memory]:
        """
        Retrieve relevant memories for a query.
        
        Args:
            query: Query text
            top_k: Number of memories to retrieve
            
        Returns:
            List of relevant Memory objects
        """
        # Generate query embedding
        query_embedding = self._get_embedding(query)
        
        # Search vector database
        results = self.vector_db.search(query_embedding, top_k=top_k)
        
        # Convert results to Memory objects if memory_system available
        memories = []
        for mem_id, score, metadata in results:
            if self.memory_system:
                # Try to get full memory from system
                memory = self.memory_system.get_memory(mem_id)
                if memory:
                    memories.append(memory)
                else:
                    # Create memory from metadata
                    content = metadata.get('content', '')
                    mem_type = metadata.get('type', MemoryType.SHORT_TERM.value)
                    if isinstance(mem_type, str):
                        mem_type = MemoryType(mem_type)
                    memory = Memory(
                        id=mem_id,
                        content=content,
                        memory_type=mem_type,
                        timestamp=metadata.get('timestamp', time.time()),
                        metadata={'score': score, **metadata}
                    )
                    memories.append(memory)
            else:
                # Create basic memory from metadata
                content = metadata.get('content', '')
                memory = Memory(
                    id=mem_id,
                    content=content,
                    memory_type=MemoryType.SHORT_TERM,
                    timestamp=time.time(),
                    metadata={'score': score, **metadata}
                )
                memories.append(memory)
        
        return memories
    
    def augment_prompt(
        self,
        query: str,
        context_memories: List[Memory],
        max_context_length: int = 2000
    ) -> str:
        """
        Augment a prompt with retrieved context.
        
        Args:
            query: Original query
            context_memories: Retrieved memories
            max_context_length: Maximum length of context to include
            
        Returns:
            Augmented prompt
        """
        if not context_memories:
            return query
        
        # Build context from memories
        context_parts = []
        current_length = 0
        
        for memory in context_memories:
            content = memory.content
            if current_length + len(content) > max_context_length:
                # Truncate if needed
                remaining = max_context_length - current_length
                if remaining > 100:  # Only add if significant space remains
                    content = content[:remaining] + "..."
                    context_parts.append(content)
                break
            
            context_parts.append(content)
            current_length += len(content)
        
        # Format augmented prompt
        context_text = "\n\n".join(context_parts)
        augmented_prompt = f"""Context from memory:
{context_text}

Query: {query}

Please answer the query using the context provided above when relevant."""
        
        return augmented_prompt
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        max_context_length: int = 2000
    ) -> RAGResult:
        """
        Full RAG pipeline: embed -> retrieve -> format.
        
        Args:
            query: Query text
            top_k: Number of memories to retrieve
            max_context_length: Maximum context length
            
        Returns:
            RAGResult with query, memories, and augmented prompt
        """
        start_time = time.time()
        
        # Retrieve relevant memories
        context_memories = self.retrieve(query, top_k=top_k)
        
        # Augment prompt
        augmented_prompt = self.augment_prompt(
            query,
            context_memories,
            max_context_length=max_context_length
        )
        
        retrieval_time = time.time() - start_time
        
        result = RAGResult(
            query=query,
            context_memories=context_memories,
            augmented_prompt=augmented_prompt,
            retrieval_time=retrieval_time,
            metadata={
                'num_memories': len(context_memories),
                'context_length': len(augmented_prompt)
            }
        )
        
        return result
    
    def add_document(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[str]:
        """
        Add a document to the RAG system with chunking.
        
        Args:
            text: Document text
            metadata: Document metadata
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of chunk IDs
        """
        if not text:
            return []
        
        metadata = metadata or {}
        
        # Split into chunks with overlap
        chunks = []
        start = 0
        chunk_num = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                for punct in ['. ', '! ', '? ', '\n\n']:
                    last_punct = chunk.rfind(punct)
                    if last_punct > chunk_size * 0.7:  # At least 70% through
                        chunk = chunk[:last_punct + 1]
                        end = start + last_punct + 1
                        break
            
            chunk_id = f"doc_{hash(text[:50])}_{chunk_num}"
            chunk_metadata = {
                **metadata,
                'chunk_num': chunk_num,
                'content': chunk,
                'timestamp': time.time()
            }
            
            # Generate embedding and add to vector DB
            embedding = self._get_embedding(chunk)
            self.vector_db.add(embedding, [chunk_id], [chunk_metadata])
            
            # Also add to memory system if available
            if self.memory_system:
                self.memory_system.add_memory(
                    content=chunk,
                    memory_type=MemoryType.SEMANTIC,
                    importance=0.6,
                    metadata=chunk_metadata,
                    id_=chunk_id
                )
            
            chunks.append(chunk_id)
            chunk_num += 1
            start = end - overlap
        
        logger.info(f"Added document with {len(chunks)} chunks")
        return chunks
    
    def add_documents_from_file(
        self,
        path: Path,
        chunk_size: int = 512,
        metadata: Optional[Dict] = None
    ) -> List[str]:
        """
        Load and add documents from a file.
        
        Args:
            path: Path to file (txt, md, json)
            chunk_size: Size of each chunk
            metadata: Additional metadata
            
        Returns:
            List of chunk IDs
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        metadata = metadata or {}
        metadata['source_file'] = str(path)
        metadata['file_name'] = path.name
        
        # Read file based on extension
        if path.suffix in ['.txt', '.md']:
            text = path.read_text(encoding='utf-8')
        elif path.suffix == '.json':
            import json
            data = json.loads(path.read_text(encoding='utf-8'))
            # Convert JSON to text representation
            text = json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        return self.add_document(text, metadata=metadata, chunk_size=chunk_size)
    
    def clear(self):
        """Clear all documents from the RAG system (use with caution)."""
        logger.warning("Clearing RAG system is not fully supported - vector DB may need manual cleanup")
        # Note: VectorDB interface doesn't have a clear method
        # This would need to be implemented per backend
