"""Chat-session state service.

Wraps :mod:`enigma_engine.core.model_context` (load_model_context,
``_EMOTIONAL_RANGES`` for the runtime-state readout) — these are AI-computed
state surfaces, NOT user-authored personality (see ARCH_DECISION.md §3 C3 and
the PAGE_INVENTORY drift-check).

Phase 0 (this slice): signatures only; bodies forward verbatim to core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_context(path: str | Path) -> Any:
    """Load a model context (emotional state, memory references, etc.).

    Delegates to :func:`enigma_engine.core.model_context.load_model_context`.
    """

    from enigma_engine.core.model_context import load_model_context

    return load_model_context(str(path))


def emotional_ranges() -> Any:
    """Return the runtime emotional-state range table for read-only display.

    **Display-only — never use for write-back.** Per ARCH_DECISION.md C3, the
    AI's emotional state is AI-computed, not user-authored. The GUI may
    display the live value (the AI knowing itself) but must not accept user
    edits to it.

    Delegates to the ``_EMOTIONAL_RANGES`` constant in
    :mod:`enigma_engine.core.model_context`.
    """

    from enigma_engine.core.model_context import _EMOTIONAL_RANGES

    return _EMOTIONAL_RANGES
