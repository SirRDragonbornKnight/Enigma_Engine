# memory package - Conversation Storage, Vector Search, Categorization, Export/Import, RAG, and more

from .analytics import MemoryAnalytics
from .async_memory import AsyncMemoryDatabase, AsyncVectorDB
from .augmented_engine import (
    MemoryAugmentedEngine,
    MemoryConfig,
    chat_with_memory,
    generate_with_memory,
    get_memory_engine,
    search_memories,
    store_memory,
)
from .backup import MemoryBackupScheduler
from .categorization import Memory, MemoryCategorization, MemoryCategory, MemoryType
from .consolidation import MemoryConsolidator
from .conversation_summary import (
    ConversationSummarizer,
    ConversationSummary,
    export_for_handoff,
    get_continuation_context,
    get_summarizer,
    summarize_conversation,
)
from .deduplication import MemoryDeduplicator
from .embeddings import AutoEmbeddingVectorDB, EmbeddingGenerator
from .encryption import EncryptedMemoryCategory, MemoryEncryption
from .export_import import MemoryExporter, MemoryImporter
from .manager import ConversationManager
from .memory_db import MemoryDatabase, add_memory, recent
from .rag import RAGResult, RAGSystem
from .search import MemorySearch
from .vector_db import (
    FAISSVectorDB,
    PineconeVectorDB,
    SimpleVectorDB,
    VectorDBInterface,
    create_vector_db,
)
from .visualization import MemoryVisualizer

__all__ = [
    # Legacy API
    "ConversationManager",
    "add_memory",
    "recent",
    
    # Memory Database
    "MemoryDatabase",
    
    # Vector databases
    "VectorDBInterface",
    "FAISSVectorDB",
    "PineconeVectorDB",
    "SimpleVectorDB",
    "create_vector_db",
    
    # Categorization
    "Memory",
    "MemoryType",
    "MemoryCategory",
    "MemoryCategorization",
    
    # Export/Import
    "MemoryExporter",
    "MemoryImporter",
    
    # RAG
    "RAGSystem",
    "RAGResult",
    
    # Embeddings
    "EmbeddingGenerator",
    "AutoEmbeddingVectorDB",
    
    # Consolidation
    "MemoryConsolidator",
    
    # Async
    "AsyncMemoryDatabase",
    "AsyncVectorDB",
    
    # Search
    "MemorySearch",
    
    # Deduplication
    "MemoryDeduplicator",
    
    # Visualization
    "MemoryVisualizer",
    
    # Analytics
    "MemoryAnalytics",
    
    # Encryption
    "MemoryEncryption",
    "EncryptedMemoryCategory",
    
    # Backup
    "MemoryBackupScheduler",
    
    # Memory-Augmented Engine (NEW - AI with real memory!)
    "MemoryAugmentedEngine",
    "MemoryConfig",
    "get_memory_engine",
    "generate_with_memory",
    "chat_with_memory",
    "store_memory",
    "search_memories",
    
    # Conversation Summarization (context compression & handoff)
    "ConversationSummarizer",
    "ConversationSummary",
    "get_summarizer",
    "summarize_conversation",
    "get_continuation_context",
    "export_for_handoff",
]
