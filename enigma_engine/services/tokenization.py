"""Tokenizer construction and loading service.

Consolidates ``core.tokenizer.get_tokenizer`` + ``core.bpe_tokenizer.BPETokenizer``
imports (appears 8+ times across the GUI today).

Phase 0 (this slice): signatures only; bodies forward verbatim to core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def get_tokenizer(path: str | Path | None = None) -> Any:
    """Load the default tokenizer (or a tokenizer at ``path``).

    Delegates to :func:`enigma_engine.core.tokenizer.get_tokenizer`. Argument
    forwarding is verbatim — the optional/required nature of ``path`` is
    defined by the core helper, not the service.
    """

    from enigma_engine.core.tokenizer import get_tokenizer as _get_tokenizer

    if path is None:
        return _get_tokenizer()
    return _get_tokenizer(str(path))


def load_bpe_tokenizer(path: str | Path) -> Any:
    """Load a :class:`~enigma_engine.core.bpe_tokenizer.BPETokenizer` from disk.

    Phase 0: thin wrapper. Future work may unify this with :func:`get_tokenizer`.
    """

    from enigma_engine.core.bpe_tokenizer import BPETokenizer

    tok = BPETokenizer()
    tok.load(str(path))
    return tok
