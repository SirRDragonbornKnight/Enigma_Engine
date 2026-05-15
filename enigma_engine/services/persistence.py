"""Atomic on-disk persistence (JSON, text, torch state).

Service contract for the GUI. Delegates to ``enigma_engine.core.safe_save``.
The atomic-save pattern (write tmp + rename) is a Stable Engineering Pattern
(see [AA code maker.md](../../AA%20code%20maker.md) §3) — every GUI write must
go through these wrappers, not raw ``open(..., "w")``.

Phase 0 (this slice): signatures only; bodies forward verbatim to core. No
GUI callers migrated yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_json(path: str | Path, data: Any) -> None:
    """Atomically write a JSON document to ``path``.

    Delegates to :func:`enigma_engine.core.safe_save.atomic_write_json`.
    Caller-side serialization (sort_keys, indent) follows the core helper.
    """

    from enigma_engine.core.safe_save import atomic_write_json

    atomic_write_json(str(path), data)


def write_text(path: str | Path, text: str) -> None:
    """Atomically write a text file to ``path``.

    Delegates to :func:`enigma_engine.core.safe_save.atomic_write_text`.
    """

    from enigma_engine.core.safe_save import atomic_write_text

    atomic_write_text(str(path), text)


def save_torch(path: str | Path, obj: Any) -> None:
    """Atomically write a torch state dict / checkpoint to ``path``.

    Delegates to :func:`enigma_engine.core.safe_save.atomic_torch_save`.
    """

    from enigma_engine.core.safe_save import atomic_torch_save

    atomic_torch_save(obj, str(path))
