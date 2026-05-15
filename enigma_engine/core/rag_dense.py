"""Dense semantic retrieval index — drop-in alternative to ``RAGIndex``.

Uses ``sentence-transformers`` for chunk/query embeddings and
``faiss-cpu`` for nearest-neighbour search.  Both are optional
soft-imports; if either is missing, :func:`is_available` returns
``False`` and the factory function :func:`enigma_engine.core.rag.make_rag_index`
falls back to the BM25 ``RAGIndex``.

Local-only contract: ``sentence-transformers`` downloads model weights
from HuggingFace on FIRST use (~80 MB for ``all-MiniLM-L6-v2``, ~130 MB
for ``bge-small-en-v1.5``).  After the initial cache is populated
(``~/.cache/huggingface/``) the model loads from disk — no runtime
cloud dependency on subsequent runs.  Users who want a strictly air-
gapped install should pre-download the model on a connected machine,
copy the cache directory, then set ``HF_HUB_OFFLINE=1``.

Protocol compatibility with :class:`enigma_engine.core.rag.RAGIndex`:

* :meth:`add_document` — accepts ``(source, text, chunk_size, overlap)``
* :meth:`build` — embeds all chunks, constructs the FAISS index
* :attr:`is_built` / :attr:`chunk_count` — property booleans
* :meth:`query` — returns ``list[dict]`` with ``chunk``, ``source``,
  ``score``, ``index`` (same shape as ``RAGIndex.query``)
* :meth:`save` / :meth:`load` — JSON metadata + sidecar ``.npy`` for
  the embedding matrix
* :meth:`format_context` — re-exported from :class:`RAGIndex` for
  identical output formatting

Pass 156z9dr (N-14): initial slice.  Single-shot factory wiring at
``enigma_engine.core.rag.make_rag_index`` based on
``CONFIG['rag_backend']`` (``"bm25"`` default, ``"dense"`` opt-in).
Soft-fail-to-BM25 on dep miss keeps the slice safe to ship without
adding hard requirements.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .rag import CHUNK_OVERLAP, CHUNK_SIZE, TOP_K_DEFAULT, RAGIndex, chunk_text

logger = logging.getLogger(__name__)

# --- Soft imports -----------------------------------------------------------
# Kept at module scope so tests can monkeypatch ``_st`` / ``_faiss``
# with fakes via ``sys.modules`` injection without paying the real
# import cost.

_HAS_ST = False
_HAS_FAISS = False
try:
    import sentence_transformers as _st  # type: ignore[import-not-found]
    _HAS_ST = True
except ImportError:
    _st = None  # type: ignore[assignment]
try:
    import faiss as _faiss  # type: ignore[import-not-found]
    _HAS_FAISS = True
except ImportError:
    _faiss = None  # type: ignore[assignment]


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# Pass 156z9dr: 33M params, 384-d embeddings, retrieval-tuned.  Falls
# back to ``all-MiniLM-L6-v2`` (22M / 384-d, weaker but smaller) if
# the user pins it via the factory ``model_name`` arg.  Decision logged
# in SUGGESTIONS.md RAG-2 (RESOLVED).


def is_available() -> bool:
    """True when both ``sentence-transformers`` and ``faiss-cpu`` import."""
    return _HAS_ST and _HAS_FAISS


class DenseRAGIndex:
    """FAISS + sentence-transformers retrieval index.

    Constructed via the factory :func:`enigma_engine.core.rag.make_rag_index`
    when ``rag_backend=="dense"`` and deps are present.  Direct
    construction is supported for tests; raises :class:`RuntimeError`
    when deps are missing so the failure is loud rather than silent.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        if not is_available():
            raise RuntimeError(
                "DenseRAGIndex requires sentence-transformers and "
                "faiss-cpu.  Install with: pip install "
                "sentence-transformers faiss-cpu"
            )
        self.model_name = model_name
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self._embedder: Any = None        # lazy: created on first build/query
        self._index: Any = None           # faiss.IndexFlatIP
        self._embeddings: np.ndarray | None = None
        self._built = False
        self._dim: int | None = None

    # -- Building ----------------------------------------------------------

    def _ensure_embedder(self) -> None:
        if self._embedder is None:
            # Soft-import the real class at call time so tests that
            # patch ``_st`` get the patched ``SentenceTransformer``.
            self._embedder = _st.SentenceTransformer(self.model_name)

    def add_document(
        self,
        source: str,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> int:
        """Append ``text``'s chunks to the index.  Call :meth:`build` after."""
        new = chunk_text(text, chunk_size, overlap)
        self.chunks.extend(new)
        self.sources.extend([source] * len(new))
        self._built = False
        return len(new)

    def build(self) -> None:
        """Embed all chunks and construct the FAISS inner-product index."""
        if not self.chunks:
            logger.warning(
                "DenseRAGIndex.build() called with no chunks"
            )
            return
        self._ensure_embedder()
        emb = self._embedder.encode(
            self.chunks,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] != len(self.chunks):
            raise RuntimeError(
                f"Embedder returned unexpected shape {emb.shape!r} for "
                f"{len(self.chunks)} chunks"
            )
        self._embeddings = emb
        self._dim = int(emb.shape[1])
        # Cosine similarity via inner product on L2-normalised vectors.
        self._index = _faiss.IndexFlatIP(self._dim)
        self._index.add(emb)
        self._built = True
        logger.info(
            "Dense RAG index built: %d chunks, dim=%d, model=%s",
            len(self.chunks), self._dim, self.model_name,
        )

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    # -- Querying ----------------------------------------------------------

    def query(self, text: str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
        """Retrieve the ``top_k`` chunks most similar to ``text``.

        Returns dicts shaped exactly like :meth:`RAGIndex.query` so
        downstream consumers (``_maybe_rag_splice``, GUI chat tail,
        ``format_context``) need no branching on backend.
        """
        if not self._built or self._index is None:
            return []
        self._ensure_embedder()
        q = self._embedder.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q = np.asarray(q, dtype=np.float32)
        if q.ndim != 2:
            return []
        k = min(top_k, len(self.chunks))
        if k <= 0:
            return []
        scores, indices = self._index.search(q, k)
        results: list[dict] = []
        for i, idx in enumerate(indices[0]):
            idx_int = int(idx)
            if idx_int < 0 or idx_int >= len(self.chunks):
                continue
            score = float(scores[0][i])
            if score <= 0:
                continue
            results.append({
                "chunk": self.chunks[idx_int],
                "source": self.sources[idx_int],
                "score": round(score, 4),
                "index": idx_int,
            })
        return results

    # Identical output formatting as BM25 path — no branching needed
    # at consumer sites.
    format_context = staticmethod(RAGIndex.format_context)

    # -- Persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save chunks + sources to JSON and the embedding matrix to ``<path>.npy``.

        The two-file layout keeps the JSON readable for inspection
        while the bulk float matrix stays in a compact binary file.
        """
        path = Path(path)
        emb_path = path.with_name(path.name + ".npy")
        meta = {
            "chunks": self.chunks,
            "sources": self.sources,
            "model_name": self.model_name,
            "dim": self._dim,
        }
        from .safe_save import atomic_write_json
        atomic_write_json(path, meta, indent=0)
        if self._embeddings is not None:
            # ``np.save`` is itself atomic on POSIX/Windows for small
            # matrices; the metadata file is the source of truth and
            # is written atomically above.
            np.save(emb_path, self._embeddings)

    @classmethod
    def load(cls, path: str | Path) -> "DenseRAGIndex":
        """Restore an index previously written by :meth:`save`."""
        path = Path(path)
        emb_path = path.with_name(path.name + ".npy")
        data = json.loads(path.read_text(encoding="utf-8"))
        idx = cls(model_name=data.get("model_name", DEFAULT_MODEL))
        idx.chunks = list(data.get("chunks", []))
        idx.sources = list(data.get("sources", []))
        dim_raw = data.get("dim")
        idx._dim = int(dim_raw) if dim_raw else None
        if emb_path.exists() and idx._dim:
            emb = np.load(emb_path).astype(np.float32)
            if emb.ndim == 2 and emb.shape[0] == len(idx.chunks):
                idx._embeddings = emb
                idx._index = _faiss.IndexFlatIP(idx._dim)
                idx._index.add(emb)
                idx._built = True
            else:
                logger.warning(
                    "DenseRAGIndex.load: embedding matrix shape %r does "
                    "not match %d chunks; index left unbuilt",
                    emb.shape, len(idx.chunks),
                )
        return idx
