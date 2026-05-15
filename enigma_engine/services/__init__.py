"""GUI-facing service contracts (Phase 0c skeleton).

This package is the **service contract layer** between the GUI client and
``enigma_engine.core``. It exists so the GUI rewrite (Phase 1–4 of the GUI
modernization plan; see [information/gui/ARCH_DECISION.md](../../information/gui/ARCH_DECISION.md))
has a stable target surface that does NOT change when CustomTkinter is replaced
by Qt/Tauri/etc.

Design rules (also recorded in ARCH_DECISION.md non-goals):

1.  Services **adapt to core**, never the reverse. Core APIs are not changed to
    suit the service surface.
2.  Every public function/class here delegates to ``enigma_engine.core.*``
    internally. Bodies stay thin. No business logic lives here in Phase 0.
3.  Imports of ``core.*`` are **deferred to call-time** so importing this
    package stays cheap and GUI cold-start is not pessimised.
4.  Signatures are stable — once a service exists, breaking its signature is
    a cross-cut migration (every page using it must follow). Adding new
    services or new kwargs (with defaults) is fine.
5.  No GUI callers are migrated to this package in Phase 0. That is Phase 4
    work, one page at a time.

Why this layer exists:

- Today: ``enigma_engine/gui/*.py`` imports ~30 distinct ``enigma_engine.core``
  modules directly (mostly via deferred imports inside functions; verified by
  grep on Pass 156z9df). Cutover to a new GUI stack would mean rewriting every
  one of those import sites, twice (once in PySide6, once in Tauri front-end).
- After Phase 4: the GUI imports only ``enigma_engine.services``. The 30+ core
  surfaces collapse to ~8 service modules. Swapping the GUI stack is a
  per-page port, not a fan-out across 30 imports.
- After ARCH-1 (engine ↔ GUI process split, future slice): swapping
  in-process delegation for an IPC client is a one-layer change. The service
  signatures stay the same.

Phase 0c ships only the **module skeleton and signatures**. Bodies forward to
core directly. No tests are added (no behavior change to test). No GUI page is
migrated yet.
"""

from __future__ import annotations

__all__ = [
    "persistence",
    "model_lifecycle",
    "tokenization",
    "inference",
    "training_dispatch",
    "hardware",
    "documents",
    "chat_state",
]
