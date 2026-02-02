"""
================================================================================
🧠 FORGEAI MEMORY MODULE - THE VAULT OF MEMORIES
================================================================================

Memory systems for conversation storage, vector search, RAG, and more.

📍 PACKAGE: forge_ai/memory/
🏷️ TYPE: AI Memory & Context System

┌─────────────────────────────────────────────────────────────────────────────┐
│  THE ARCHIVES OF DIGITAL MEMORY:                                            │
│                                                                             │
│  "Without memory, the AI lives only in the present moment.                 │
│   With memory, it becomes your companion across time."                     │
│                                                                             │
│  🗃️ STORAGE        - Conversations, facts, preferences                     │
│  🔍 SEARCH         - Find relevant memories instantly                      │
│  📊 EMBEDDINGS     - Turn text into searchable vectors                     │
│  🎯 RAG            - Retrieve context for better responses                 │
│  📝 SUMMARIZATION  - Compress long conversations                           │
│  🔐 ENCRYPTION     - Keep memories private                                 │
│  💾 BACKUP         - Never lose what was learned                           │
│                                                                             │
│  NEW FEATURES (Feb 2026):                                                  │
│  • MemoryAugmentedEngine - AI with REAL memory recall!                     │
│  • ConversationSummarizer - Compress context, never forget                 │
└─────────────────────────────────────────────────────────────────────────────┘

📖 QUICK START:
    # AI with automatic memory
    from forge_ai.memory import generate_with_memory
    response = generate_with_memory("What did I tell you about cats?")
    
    # Summarize long conversations
    from forge_ai.memory import summarize_conversation
    summary = summarize_conversation(messages)
"""

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
from .augmented_engine import (
    MemoryAugmentedEngine,
    MemoryConfig,
    get_memory_engine,
    generate_with_memory,
    chat_with_memory,
    store_memory,
    search_memories
)
from .conversation_summary import (
    ConversationSummarizer,
    ConversationSummary,
    get_summarizer,
    summarize_conversation,
    get_continuation_context,
    export_for_handoff
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
