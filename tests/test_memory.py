"""Tests for persistent memory, memory commands, RAG, and document readers."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestDocumentReaders:
    """Verify document_readers module structure and graceful fallbacks."""

    def test_module_imports(self):
        """document_readers should import without any required deps."""
        from enigma_engine.core.document_readers import (
            SUPPORTED_EXTENSIONS,
        )
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS

    def test_read_pdf_raises_without_lib(self):
        """read_pdf raises ImportError when pymupdf is missing."""
        from enigma_engine.core import document_readers as dr
        if dr.pdf_available():
            pytest.skip("pymupdf is installed — skip missing-lib test")
        with pytest.raises(ImportError, match="pymupdf"):
            dr.read_pdf("fake.pdf")

    def test_read_docx_raises_without_lib(self):
        """read_docx raises ImportError when python-docx is missing."""
        from enigma_engine.core import document_readers as dr
        if dr.docx_available():
            pytest.skip("python-docx is installed — skip missing-lib test")
        with pytest.raises(ImportError, match="python-docx"):
            dr.read_docx("fake.docx")

    def test_read_document_returns_none_on_missing_lib(self):
        """read_document returns None for unsupported/unavailable formats."""
        from enigma_engine.core.document_readers import read_document
        # Unknown extension always returns None
        assert read_document("file.xyz") is None

    def test_read_pdf_file_not_found(self):
        """read_pdf raises FileNotFoundError for missing file."""
        from enigma_engine.core import document_readers as dr
        if not dr.pdf_available():
            pytest.skip("pymupdf not installed")
        with pytest.raises(FileNotFoundError):
            dr.read_pdf("nonexistent.pdf")

    def test_read_docx_file_not_found(self):
        """read_docx raises FileNotFoundError for missing file."""
        from enigma_engine.core import document_readers as dr
        if not dr.docx_available():
            pytest.skip("python-docx not installed")
        with pytest.raises(FileNotFoundError):
            dr.read_docx("nonexistent.docx")


class TestRAG:
    """Verify RAG module: chunking, TF-IDF, index, query."""

    def test_rag_imports(self):
        """Core RAG classes should import without errors."""
        from enigma_engine.core.rag import (
            CHUNK_SIZE, TOP_K_DEFAULT,
        )
        assert CHUNK_SIZE > 0
        assert TOP_K_DEFAULT > 0

    def test_chunk_text_basic(self):
        """chunk_text splits long text into overlapping chunks."""
        from enigma_engine.core.rag import chunk_text
        text = "Hello world. " * 200  # much longer than CHUNK_SIZE
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        # All chunks should be non-empty
        assert all(c.strip() for c in chunks)

    def test_chunk_text_empty(self):
        """chunk_text returns empty list for empty text."""
        from enigma_engine.core.rag import chunk_text
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_chunk_text_short(self):
        """Short text produces exactly one chunk."""
        from enigma_engine.core.rag import chunk_text
        chunks = chunk_text("Short text.", chunk_size=500)
        assert len(chunks) == 1

    def test_tfidf_vectorizer(self):
        """TfidfVectorizer computes non-zero vectors."""
        from enigma_engine.core.rag import TfidfVectorizer
        docs = [
            "the cat sat on the mat",
            "the dog ran in the park",
            "fish swim in the water",
        ]
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(docs)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] > 0
        # Each row should have non-zero entries
        import numpy as np
        for i in range(3):
            row = matrix[i]
            if hasattr(row, 'toarray'):
                row = row.toarray()
            assert np.any(np.asarray(row) != 0)

    def test_tfidf_serialization(self):
        """TfidfVectorizer round-trips through to_dict/from_dict."""
        from enigma_engine.core.rag import TfidfVectorizer
        docs = ["hello world", "foo bar baz"]
        vec = TfidfVectorizer()
        vec.fit(docs)
        data = vec.to_dict()
        vec2 = TfidfVectorizer.from_dict(data)
        assert vec2.vocab == vec.vocab

    def test_rag_index_end_to_end(self):
        """Full RAG flow: add docs, build, query, format."""
        from enigma_engine.core.rag import RAGIndex
        index = RAGIndex()
        index.add_document("doc1.txt",
                           "Python is a programming language used for AI.")
        index.add_document("doc2.txt",
                           "Cats and dogs are popular household pets.")
        index.build()
        assert index.is_built
        assert index.chunk_count == 2

        results = index.query("programming language")
        assert len(results) > 0
        assert results[0]["source"] == "doc1.txt"

        ctx = RAGIndex.format_context(results)
        assert "doc1.txt" in ctx

    def test_format_context_respects_max_chars(self):
        """S763: format_context output must not exceed max_chars."""
        from enigma_engine.core.rag import RAGIndex
        results = [
            {"source": "a.txt", "chunk": "x" * 200, "score": 1.0},
            {"source": "b.txt", "chunk": "y" * 200, "score": 0.9},
            {"source": "c.txt", "chunk": "z" * 200, "score": 0.8},
            {"source": "d.txt", "chunk": "w" * 200, "score": 0.7},
        ]
        ctx = RAGIndex.format_context(results, max_chars=500)
        assert len(ctx) <= 500, (
            f"format_context output {len(ctx)} chars exceeds max_chars=500"
        )

    def test_rag_index_empty_query(self):
        """Querying unbuilt index returns empty list."""
        from enigma_engine.core.rag import RAGIndex
        index = RAGIndex()
        assert index.query("anything") == []

    def test_rag_index_save_load(self, tmp_path):
        """RAG index persists and loads correctly."""
        from enigma_engine.core.rag import RAGIndex
        index = RAGIndex()
        index.add_document("test.md", "Machine learning is a subset of AI.")
        index.build()

        save_path = tmp_path / "test_index.json"
        index.save(save_path)
        assert save_path.exists()

        loaded = RAGIndex.load(save_path)
        assert loaded.is_built
        assert loaded.chunk_count == index.chunk_count
        results = loaded.query("machine learning")
        assert len(results) > 0

    def test_adaptive_chunk_preserves_delimiters(self):
        """S728: adaptive_chunk_text must preserve markdown headers."""
        from enigma_engine.core.rag import adaptive_chunk_text
        text = "Intro paragraph.\n## Section One\nBody of section one."
        chunks = adaptive_chunk_text(text, target_size=5000)
        combined = " ".join(chunks)
        assert "##" in combined or "Section One" in combined
        # The header marker must survive splitting
        assert any("##" in c for c in chunks)

    def test_adaptive_chunk_preserves_code_fences(self):
        """S728: adaptive_chunk_text must preserve code fences."""
        from enigma_engine.core.rag import adaptive_chunk_text
        text = "Before code.\n```python\nprint('hi')\n```\nAfter code."
        chunks = adaptive_chunk_text(text, target_size=5000)
        combined = " ".join(chunks)
        assert "```" in combined

    def test_transform_warns_on_missing_idf(self, caplog):
        """S729: transform() logs warning when IDF is missing."""
        import logging
        from enigma_engine.core.rag import TfidfVectorizer
        vec = TfidfVectorizer()
        vec.vocab = {"hello": 0}  # vocab set but no idf
        with caplog.at_level(logging.WARNING):
            result = vec.transform(["hello world"])
        assert result.shape == (1, 0)
        assert any("idf" in r.message.lower() or "IDF" in r.message
                    for r in caplog.records)


class TestPersistentMemory:
    """Tests for enigma_engine.core.memory module."""

    def test_add_and_retrieve(self, tmp_path):
        """Facts can be added and retrieved."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        assert mem.add("User's name is Alex")
        assert mem.count == 1
        assert "Alex" in mem.facts[0]

    def test_deduplication(self, tmp_path):
        """Duplicate facts are rejected."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        assert mem.add("User likes Python")
        assert not mem.add("User likes Python")
        assert not mem.add("user likes python")  # case-insensitive
        assert mem.count == 1

    def test_replace_outdated(self, tmp_path):
        """Updated facts replace old ones about the same topic."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        mem.add("User's name is Bob")
        mem.add("User's name is Alex")
        assert mem.count == 1
        assert "Alex" in mem.facts[0]

    def test_remove_by_content(self, tmp_path):
        """Facts can be removed by substring match."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        mem.add("User works at NASA")
        mem.add("User likes coffee")
        assert mem.remove("NASA")
        assert mem.count == 1
        assert "coffee" in mem.facts[0]

    def test_remove_by_index(self, tmp_path):
        """Facts can be removed by index."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        mem.add("Fact A")
        mem.add("Fact B")
        assert mem.remove(0)
        assert mem.count == 1
        assert "Fact B" in mem.facts[0]

    def test_clear(self, tmp_path):
        """Clear removes all facts."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        mem.add("Fact 1")
        mem.add("Fact 2")
        mem.clear()
        assert mem.count == 0

    def test_persistence(self, tmp_path):
        """Facts survive save/reload cycle."""
        path = tmp_path / "mem.md"
        from enigma_engine.core.memory import PersistentMemory
        mem1 = PersistentMemory(memory_path=path)
        mem1.add("User is a developer")
        mem1.add("User prefers dark mode")
        # Create a new instance from same path
        mem2 = PersistentMemory(memory_path=path)
        assert mem2.count == 2
        assert "developer" in mem2.facts[0]
        assert "dark mode" in mem2.facts[1]

    def test_build_context(self, tmp_path):
        """build_context produces formatted output."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        assert mem.build_context() == ""  # empty when no facts
        mem.add("User's name is Alex")
        ctx = mem.build_context()
        assert "[MEMORY" in ctx
        assert "Alex" in ctx
        assert "[END MEMORY]" in ctx

    def test_build_context_token_cap(self, tmp_path):
        """build_context respects token budget."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        # Add many long facts
        for i in range(30):
            mem.add(f"This is a moderately long fact number {i} about something the user said")
        ctx = mem.build_context(max_tokens=100)
        # Should be capped — not all 30 facts should appear
        assert ctx.count("- ") < 30

    def test_extract_facts_name(self, tmp_path):
        """extract_facts catches 'my name is X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("Hey, my name is Alex")
        assert len(added) >= 1
        assert any("Alex" in f for f in added)
        assert mem.count >= 1

    def test_extract_facts_short_name(self, tmp_path):
        """extract_facts catches short names like Jo, Al."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("my name is Jo")
        assert len(added) >= 1
        assert any("Jo" in f for f in added)

    def test_extract_facts_accented_name(self, tmp_path):
        """extract_facts catches accented names like Andre."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("my name is Andre")
        assert len(added) >= 1
        assert any("Andre" in f for f in added)

    def test_extract_facts_hyphenated_name(self, tmp_path):
        """extract_facts catches hyphenated names like Mary-Jane."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("my name is Mary-Jane")
        assert len(added) >= 1
        assert any("Mary-Jane" in f for f in added)

    def test_extract_facts_workplace(self, tmp_path):
        """extract_facts catches 'I work at X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I work at NASA doing research.")
        assert len(added) >= 1
        assert any("NASA" in f for f in added)

    def test_extract_facts_preference(self, tmp_path):
        """extract_facts catches 'I prefer X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I prefer Python over JavaScript")
        assert len(added) >= 1
        assert any("Python" in f for f in added)

    def test_extract_facts_remember_request(self, tmp_path):
        """extract_facts catches 'remember that X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("Please remember that my dog is named Max")
        assert len(added) >= 1
        assert any("Max" in f for f in added)

    def test_extract_facts_nothing(self, tmp_path):
        """extract_facts returns empty on uninteresting messages."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("What is the weather today?")
        assert len(added) == 0

    def test_max_facts_trim(self, tmp_path):
        """Oldest facts are trimmed when exceeding MAX_FACTS."""
        from enigma_engine.core.memory import PersistentMemory, MAX_FACTS
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        for i in range(MAX_FACTS + 10):
            mem.add(f"Unique fact number {i}")
        assert mem.count == MAX_FACTS
        # First 10 should have been trimmed
        assert "number 0" not in mem.facts[0]

    def test_get_memory_singleton(self):
        """get_memory returns a singleton."""
        from enigma_engine.core.memory import get_memory
        import enigma_engine.core.memory as mem_module
        # Reset singleton for test isolation
        mem_module._instance = None
        m1 = get_memory()
        m2 = get_memory()
        assert m1 is m2
        mem_module._instance = None  # cleanup

    def test_hand_editability(self, tmp_path):
        """User can hand-edit the memory file."""
        path = tmp_path / "mem.md"
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=path)
        mem.add("Original fact")
        # Simulate user hand-editing the file
        path.write_text(
            "# AI Memory Notes\n\n"
            "- Hand-written fact by user\n"
            "- Another user note\n",
            encoding="utf-8")
        mem.reload()
        assert mem.count == 2
        assert "Hand-written fact" in mem.facts[0]

    # --- Expanded fact extraction patterns ---

    def test_extract_facts_hobby(self, tmp_path):
        """extract_facts catches 'I enjoy X' / 'I love X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I enjoy hiking on weekends")
        assert len(added) >= 1
        assert any("hiking" in f.lower() for f in added)

    def test_extract_facts_love(self, tmp_path):
        """extract_facts catches 'I love X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I really love cooking Italian food")
        assert len(added) >= 1
        assert any("cooking" in f.lower() for f in added)

    def test_extract_facts_age(self, tmp_path):
        """extract_facts catches 'I'm X years old'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I'm 28 years old by the way")
        assert len(added) >= 1
        assert any("28" in f for f in added)

    def test_extract_facts_birthday(self, tmp_path):
        """extract_facts catches 'my birthday is X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("My birthday is March 15th")
        assert len(added) >= 1
        assert any("March" in f for f in added)

    def test_extract_facts_pet(self, tmp_path):
        """extract_facts catches 'I have a dog/cat named X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I have a dog named Buddy")
        assert len(added) >= 1
        assert any("Buddy" in f for f in added)

    def test_extract_facts_family(self, tmp_path):
        """extract_facts catches family members."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("My wife's name is Sarah")
        assert len(added) >= 1
        assert any("Sarah" in f for f in added)

    def test_extract_facts_education(self, tmp_path):
        """extract_facts catches 'I studied at X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I studied at MIT")
        assert len(added) >= 1
        assert any("MIT" in f for f in added)

    def test_extract_facts_dislike(self, tmp_path):
        """extract_facts catches 'I hate X' / 'I don't like X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I hate spiders honestly")
        assert len(added) >= 1
        assert any("spiders" in f.lower() for f in added)

    def test_extract_facts_language(self, tmp_path):
        """extract_facts catches 'I speak X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I speak Spanish and French fluently")
        assert len(added) >= 1
        assert any("Spanish" in f for f in added)

    def test_extract_facts_timezone(self, tmp_path):
        """extract_facts catches timezone info."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I'm in the EST timezone")
        assert len(added) >= 1
        assert any("EST" in f for f in added)

    def test_extract_facts_degree(self, tmp_path):
        """extract_facts catches 'I have a degree in X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I have a degree in computer science")
        assert len(added) >= 1
        assert any("computer science" in f.lower() for f in added)


class TestMemoryBuiltinCommands:
    """Tests for memory.remember/forget/notes builtin commands."""

    def test_remember_command_registered(self):
        """memory.remember command exists in registry."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmds = [c.name for c in registry.list_commands()]
        assert "memory.remember" in cmds

    def test_forget_command_registered(self):
        """memory.forget command exists in registry."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmds = [c.name for c in registry.list_commands()]
        assert "memory.forget" in cmds

    def test_notes_command_registered(self):
        """memory.notes command exists in registry."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmds = [c.name for c in registry.list_commands()]
        assert "memory.notes" in cmds


# =========================================================================
# Deep-dive audit fix tests — config.set, memory traversal, AI profile
# =========================================================================

class TestMemorySaveLoadTraversal:
    """Tests that memory.save/load sanitise names to prevent path traversal."""

    def test_save_rejects_path_traversal(self, tmp_path):
        """memory.save with '../evil' should strip to just 'evil'."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmd = next(c for c in registry.list_commands() if c.name == "memory.save")
        ctx = {
            "memory_dir": tmp_path,
            "chat_messages": [{"role": "user", "content": "hi"}],
        }
        result = cmd.handler(["../evil"], ctx)
        assert result.success
        # File should be in tmp_path, not parent
        assert (tmp_path / "evil.json").exists()
        assert not (tmp_path.parent / "evil.json").exists()

    def test_load_rejects_path_traversal(self, tmp_path):
        """memory.load with '../evil' should strip to just 'evil'."""
        import json
        # Create the file in tmp_path (the valid dir)
        (tmp_path / "evil.json").write_text(
            json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
            encoding="utf-8",
        )
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmd = next(c for c in registry.list_commands() if c.name == "memory.load")
        ctx = {"memory_dir": tmp_path}
        result = cmd.handler(["../evil"], ctx)
        assert result.success
        assert len(ctx["chat_messages"]) == 1

    def test_save_rejects_dot_dot_name(self):
        """memory.save with '..' as name should fail."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmd = next(c for c in registry.list_commands() if c.name == "memory.save")
        result = cmd.handler([".."], {"chat_messages": [{"role": "user", "content": "x"}]})
        assert not result.success


class TestRAGBM25:
    """Test BM25 scoring, stop word filtering, and sparse matrix support."""

    def test_bm25_idf_formula(self):
        """BM25 IDF should differ from classic TF-IDF."""
        from enigma_engine.core.rag import TfidfVectorizer

        vec = TfidfVectorizer()
        docs = ["cat sat", "dog ran", "cat ran fast"]
        vec.fit(docs)
        # "cat" appears in 2 of 3 docs
        # BM25 IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        assert "cat" in vec.vocab
        idx_cat = vec.vocab["cat"]
        assert vec.idf is not None
        assert vec.idf[idx_cat] > 0
        # "fast" appears in 1 doc → higher IDF
        idx_fast = vec.vocab["fast"]
        assert vec.idf[idx_fast] > vec.idf[idx_cat]

    def test_bm25_k1_b_stored(self):
        """BM25 parameters k1 and b should be stored in vectorizer."""
        from enigma_engine.core.rag import TfidfVectorizer

        vec = TfidfVectorizer(k1=2.0, b=0.5)
        assert vec.k1 == 2.0
        assert vec.b == 0.5

    def test_bm25_serialization(self):
        """to_dict/from_dict should round-trip BM25 state."""
        from enigma_engine.core.rag import TfidfVectorizer

        vec = TfidfVectorizer(k1=1.2, b=0.8)
        docs = ["hello world", "foo bar hello"]
        vec.fit(docs)
        d = vec.to_dict()

        vec2 = TfidfVectorizer.from_dict(d)
        assert vec2.k1 == 1.2
        assert vec2.b == 0.8
        assert vec2.avg_dl == vec.avg_dl
        assert vec2.doc_lens is not None and vec.doc_lens is not None
        assert list(vec2.doc_lens) == list(vec.doc_lens)

    def test_bm25_backward_compat_from_dict(self):
        """from_dict without BM25 keys should use defaults."""
        from enigma_engine.core.rag import TfidfVectorizer

        # Old format: idf as a flat list, no k1/b/doc_lens/avg_dl
        d = {"vocab": {"aa": 0, "bb": 1}, "idf": [1.0, 0.5]}
        vec = TfidfVectorizer.from_dict(d)
        assert vec.k1 == 1.5  # Default
        assert vec.b == 0.75  # Default
        assert vec.avg_dl == 0.0

    def test_stop_words_filtered(self):
        """Tokenizer should filter common stop words."""
        from enigma_engine.core.rag import _tokenize

        tokens = _tokenize("the cat is on a mat")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "on" not in tokens
        assert "cat" in tokens
        assert "mat" in tokens

    def test_stop_words_preserves_content_words(self):
        """Stop word filter should keep meaningful words."""
        from enigma_engine.core.rag import _tokenize

        tokens = _tokenize("machine learning algorithms")
        assert "machine" in tokens
        assert "learning" in tokens
        assert "algorithms" in tokens

    def test_transform_returns_array(self):
        """transform() should return an array-like regardless of scipy."""
        import numpy as np
        from enigma_engine.core.rag import TfidfVectorizer

        vec = TfidfVectorizer()
        docs = ["cat dog", "fish bird"]
        vec.fit(docs)
        result = vec.transform(["cat bird"])
        # Whether sparse or dense, we should be able to get a 2-D array
        if hasattr(result, 'toarray'):
            arr = result.toarray()
        elif hasattr(result, 'A'):
            arr = np.asarray(result.A)
        else:
            arr = np.asarray(result)
        assert arr.shape[0] == 1
        assert arr.shape[1] == len(vec.vocab)


# ================================================================
# Suggestion 17: _load_gguf / _load_pytorch extraction
# ================================================================

