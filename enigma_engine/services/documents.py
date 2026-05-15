"""Document reading and RAG ingestion service.

Wraps :mod:`enigma_engine.core.document_readers` and :mod:`enigma_engine.core.rag`.

Phase 0 (this slice): one thin function shipped; RAG ingestion will land when
the DOCS page migrates.
"""

from __future__ import annotations

from pathlib import Path


def read_document(path: str | Path) -> str:
    """Read a document (PDF/TXT/MD/etc.) to plain text.

    Delegates to :func:`enigma_engine.core.document_readers.read_document`.
    """

    from enigma_engine.core.document_readers import read_document as _read_document

    return _read_document(str(path))
