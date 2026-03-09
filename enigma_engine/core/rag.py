"""Lightweight RAG (Retrieval-Augmented Generation) pipeline.

Uses TF-IDF vectors with cosine similarity for retrieval — runs
entirely on numpy (already a dependency), no external vector-store
or sentence-transformer needed.

Flow:
    1. **Index**: Chunk documents → compute TF-IDF vectors → store
    2. **Query**: Vectorize query → cosine similarity → top-K chunks
    3. **Inject**: Prepend retrieved chunks to the AI system prompt

The index is rebuilt on demand and cached in memory.  Optionally
it can be persisted to ``data/rag_index.json`` so subsequent
starts are instant.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# Optional scipy for sparse matrices (10-50x memory reduction)
try:
    from scipy.sparse import csr_matrix as _csr_matrix
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 512          # characters per chunk (overlap = CHUNK_SIZE // 4)
CHUNK_OVERLAP = 128       # character overlap between consecutive chunks
TOP_K_DEFAULT = 5         # default number of chunks to retrieve
MAX_VOCAB = 8000          # cap vocabulary for TF-IDF to keep memory low
INDEX_FILE = "rag_index.json"

# BM25 parameters (Okapi BM25 defaults)
BM25_K1 = 1.5            # term frequency saturation
BM25_B = 0.75            # document length normalization

# ---------------------------------------------------------------------------
# Stop words — filtered during tokenization to improve retrieval quality
# ---------------------------------------------------------------------------
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "if", "then", "than", "that", "this", "these", "those", "it",
    "its", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "who", "whom", "which", "what",
    "when", "where", "how", "why", "all", "each", "every", "both", "few",
    "more", "most", "some", "any", "other", "into", "over", "after",
    "before", "between", "under", "above", "up", "down", "out", "off",
    "about", "just", "also", "very", "too", "only", "own", "same",
    "here", "there", "again", "once", "such", "as",
})


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping chunks of roughly *chunk_size* chars.

    Attempts to break at sentence boundaries when possible.
    """
    if not text or not text.strip():
        return []

    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Try to snap to a sentence boundary within last 20% of chunk
        if end < length:
            search_start = max(start, end - chunk_size // 5)
            for sep in (". ", ".\n", "! ", "? ", "\n\n", "\n"):
                idx = text.rfind(sep, search_start, end)
                if idx > start:
                    end = idx + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < length else length

    return chunks


# ---------------------------------------------------------------------------
# TF-IDF vectorizer (pure numpy)
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"\W+")


def _tokenize(text: str) -> list[str]:
    """Lowercase split on non-word chars, discard short tokens and stop words."""
    return [t for t in _SPLIT_RE.split(text.lower())
            if len(t) > 1 and t not in _STOP_WORDS]


class TfidfVectorizer:
    """BM25-scored vectorizer backed by sparse matrices when scipy is available.

    Despite the class name (kept for backward compatibility), this uses
    Okapi BM25 scoring which consistently outperforms plain TF-IDF for
    document retrieval.
    """

    def __init__(self, max_vocab: int = MAX_VOCAB,
                 k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.max_vocab = max_vocab
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.doc_lens: np.ndarray | None = None
        self.avg_dl: float = 0.0

    def fit(self, documents: list[str]) -> None:
        """Build vocabulary and compute IDF from a list of documents."""
        n_docs = len(documents)
        if n_docs == 0:
            return

        # Count document frequency for each term
        df: Counter[str] = Counter()
        doc_lengths: list[int] = []
        for doc in documents:
            tokens = _tokenize(doc)
            doc_lengths.append(len(tokens))
            df.update(set(tokens))

        # Keep only top-N by document frequency
        most_common = df.most_common(self.max_vocab)
        self.vocab = {term: i for i, (term, _) in enumerate(most_common)}

        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)  (BM25 IDF)
        idf_arr = np.zeros(len(self.vocab), dtype=np.float32)
        for term, idx in self.vocab.items():
            d = df[term]
            idf_arr[idx] = math.log((n_docs - d + 0.5) / (d + 0.5) + 1)
        self.idf = idf_arr

        # Document length stats for BM25 normalization
        self.doc_lens = np.array(doc_lengths, dtype=np.float32)
        self.avg_dl = float(np.mean(self.doc_lens)) if doc_lengths else 1.0

    def transform(self, documents: list[str]) -> Any:
        """Compute BM25-scored matrix (n_docs x vocab_size).

        Returns a scipy CSR sparse matrix when scipy is available,
        otherwise a dense numpy array.
        """
        if not self.vocab or self.idf is None:
            return np.zeros((len(documents), 0), dtype=np.float32)

        n = len(documents)
        v = len(self.vocab)

        # Build raw TF and doc lengths for these documents
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for i, doc in enumerate(documents):
            tokens = _tokenize(doc)
            if not tokens:
                continue
            doc_len = len(tokens)
            tf: Counter[str] = Counter(tokens)
            for term, count in tf.items():
                if term in self.vocab:
                    idx = self.vocab[term]
                    # BM25 score for this term in this doc
                    numerator = count * (self.k1 + 1)
                    denominator = count + self.k1 * (
                        1 - self.b + self.b * doc_len / max(self.avg_dl, 1.0)
                    )
                    bm25_tf = numerator / denominator
                    score = self.idf[idx] * bm25_tf
                    rows.append(i)
                    cols.append(idx)
                    vals.append(score)

        if _HAS_SCIPY:
            data = np.array(vals, dtype=np.float32) if vals else np.array([], dtype=np.float32)
            row_arr = np.array(rows, dtype=np.int32) if rows else np.array([], dtype=np.int32)
            col_arr = np.array(cols, dtype=np.int32) if cols else np.array([], dtype=np.int32)
            return _csr_matrix((data, (row_arr, col_arr)), shape=(n, v))

        # Dense fallback
        matrix = np.zeros((n, v), dtype=np.float32)
        for r, c, val in zip(rows, cols, vals):
            matrix[r, c] = val
        return matrix

    def fit_transform(self, documents: list[str]) -> Any:
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "vocab": self.vocab,
            "idf": self.idf.tolist() if self.idf is not None else [],
            "max_vocab": self.max_vocab,
            "k1": self.k1,
            "b": self.b,
            "doc_lens": self.doc_lens.tolist() if self.doc_lens is not None else [],
            "avg_dl": self.avg_dl,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TfidfVectorizer":
        """Restore from a dict produced by ``to_dict``."""
        obj = cls(
            max_vocab=data.get("max_vocab", MAX_VOCAB),
            k1=data.get("k1", BM25_K1),
            b=data.get("b", BM25_B),
        )
        obj.vocab = data.get("vocab", {})
        idf_raw = data.get("idf", [])
        # Backward compat: old format stored idf as {term: float} dict
        if isinstance(idf_raw, dict):
            idf_list = [idf_raw.get(t, 0.0) for t in sorted(
                obj.vocab, key=obj.vocab.get  # type: ignore[arg-type]
            )]
        else:
            idf_list = idf_raw
        obj.idf = np.array(idf_list, dtype=np.float32) if idf_list else None
        dl_list = data.get("doc_lens", [])
        obj.doc_lens = np.array(dl_list, dtype=np.float32) if dl_list else None
        obj.avg_dl = data.get("avg_dl", 0.0)
        return obj


# ---------------------------------------------------------------------------
# RAG Index
# ---------------------------------------------------------------------------

class RAGIndex:
    """In-memory retrieval index over chunked documents.

    Usage::

        index = RAGIndex()
        index.add_document("path/to/file.txt", text_content)
        index.build()
        results = index.query("how does training work?", top_k=5)
        context = index.format_context(results)
    """

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.sources: list[str] = []       # source path per chunk
        self.vectorizer = TfidfVectorizer()
        self.matrix: np.ndarray | None = None
        self._built = False

    # -- Building ----------------------------------------------------------

    def add_document(self, source: str, text: str,
                     chunk_size: int = CHUNK_SIZE,
                     overlap: int = CHUNK_OVERLAP) -> int:
        """Add a document's chunks to the index (call ``build()`` after).

        Returns the number of chunks added.
        """
        new_chunks = chunk_text(text, chunk_size, overlap)
        self.chunks.extend(new_chunks)
        self.sources.extend([source] * len(new_chunks))
        self._built = False
        return len(new_chunks)

    def build(self) -> None:
        """Fit TF-IDF on all added chunks and prepare for queries."""
        if not self.chunks:
            logger.warning("RAGIndex.build() called with no chunks")
            return
        self.matrix = self.vectorizer.fit_transform(self.chunks)
        self._built = True
        logger.info(
            "RAG index built: %d chunks, vocab %d",
            len(self.chunks), len(self.vectorizer.vocab)
        )

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    # -- Querying ----------------------------------------------------------

    def query(self, text: str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
        """Retrieve the *top_k* most relevant chunks for *text*.

        Returns a list of dicts with keys:
        ``chunk``, ``source``, ``score``, ``index``.
        """
        if not self._built or self.matrix is None:
            return []

        # Build binary query vector (1 where term present)
        tokens = set(_tokenize(text))
        v = len(self.vectorizer.vocab)
        q_vec = np.zeros((1, v), dtype=np.float32)
        for token in tokens:
            if token in self.vectorizer.vocab:
                q_vec[0, self.vectorizer.vocab[token]] = 1.0

        # Score documents: sum of BM25 weights for matching query terms
        raw = q_vec @ self.matrix.T
        # Normalise to 1-D dense array regardless of backend
        if hasattr(raw, 'toarray'):
            scores = np.asarray(raw.toarray()).flatten()
        elif hasattr(raw, 'A'):
            scores = np.asarray(raw.A).flatten()
        else:
            scores = np.asarray(raw).flatten()

        # Top-K indices (descending score)
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append({
                "chunk": self.chunks[idx],
                "source": self.sources[idx],
                "score": round(score, 4),
                "index": int(idx),
            })

        return results

    @staticmethod
    def format_context(results: list[dict], max_chars: int = 3000) -> str:
        """Format retrieved chunks into a context string for the AI.

        Truncates to *max_chars* total to avoid flooding the context.
        """
        if not results:
            return ""

        parts: list[str] = []
        total = 0
        for r in results:
            source = Path(r["source"]).name
            chunk = r["chunk"]
            entry = f"[From {source}] {chunk}"
            if total + len(entry) > max_chars:
                remaining = max_chars - total
                if remaining > 50:
                    parts.append(entry[:remaining] + "...")
                break
            parts.append(entry)
            total += len(entry) + 1  # +1 for newline

        return "\n\n".join(parts)

    # -- Persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save index to a JSON file."""
        data = {
            "chunks": self.chunks,
            "sources": self.sources,
            "vectorizer": self.vectorizer.to_dict(),
        }
        from enigma_engine.core.safe_save import atomic_write_json
        atomic_write_json(path, data, indent=0)

    @classmethod
    def load(cls, path: str | Path) -> "RAGIndex":
        """Load a previously saved index."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        idx = cls()
        idx.chunks = data["chunks"]
        idx.sources = data["sources"]
        idx.vectorizer = TfidfVectorizer.from_dict(data["vectorizer"])
        # Rebuild matrix from saved vectorizer
        if idx.chunks:
            idx.matrix = idx.vectorizer.transform(idx.chunks)
            idx._built = True
        return idx


# ---------------------------------------------------------------------------
# Helper: index a directory of files
# ---------------------------------------------------------------------------

def index_directory(directory: str | Path,
                    extensions: tuple[str, ...] = (
                        ".txt", ".md", ".jsonl",
                        ".pdf", ".docx",
                    )) -> RAGIndex:
    """Recursively index all matching files in *directory*.

    Automatically uses ``document_readers`` for PDF/DOCX when available.
    """
    from .document_readers import read_document, SUPPORTED_EXTENSIONS

    directory = Path(directory)
    index = RAGIndex()

    for p in sorted(directory.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in extensions:
            continue

        text: str | None = None

        if ext in SUPPORTED_EXTENSIONS:
            text = read_document(p)
        else:
            try:
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                logger.debug("Skipping unreadable file: %s", p)
                continue

        if text and text.strip():
            n = index.add_document(str(p), text)
            logger.debug("Indexed %s → %d chunks", p.name, n)

    if index.chunks:
        index.build()

    return index
