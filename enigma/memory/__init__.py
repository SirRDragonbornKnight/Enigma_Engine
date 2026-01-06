# memory package - Conversation Storage, Vector Search, Categorization, Export/Import, RAG, and more

from .manager import ConversationManager
from .memory_db import add_memory, recent, MemoryDatabase
from .vector_db import (
    VectorDBInterface,
    FAISSVectorDB,
    PineconeVectorDB,
    SimpleVectorDB,
    create_vector_db
)
from .categorization import (
    Memory,
    MemoryType,
    MemoryCategory,
    MemoryCategorization
)
from .export_import import (
    MemoryExporter,
    MemoryImporter
)
from .rag import (
    RAGSystem,
    RAGResult
)
from .embeddings import (
    EmbeddingGenerator,
    AutoEmbeddingVectorDB
)
from .consolidation import (
    MemoryConsolidator
)
from .async_memory import (
    AsyncMemoryDatabase,
    AsyncVectorDB
)
from .search import (
    MemorySearch
)
from .deduplication import (
    MemoryDeduplicator
)
from .visualization import (
    MemoryVisualizer
)
from .analytics import (
    MemoryAnalytics
)
from .encryption import (
    MemoryEncryption,
    EncryptedMemoryCategory
)
from .backup import (
    MemoryBackupScheduler
)

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
]
