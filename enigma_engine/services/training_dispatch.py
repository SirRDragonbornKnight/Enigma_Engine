"""Training-dispatch service.

Wraps the centralized training dispatcher (``run_training`` / trainer entry
points) so GUI launchers do not need to instantiate trainer classes directly
— an anti-pattern the §4 Learned Principles flag explicitly: *"When a
dispatcher mode already exists (run_training supports it), GUI launchers for
that same mode must call the dispatcher seam instead of instantiating
trainer classes directly — mixed routing creates sibling drift."*

Phase 0 (this slice): signature placeholder; first GUI consumer to migrate
will pin the call shape.
"""

from __future__ import annotations

from typing import Any


def run(ctx: Any, **kwargs: Any) -> Any:
    """Dispatch a training run via the central dispatcher.

    Phase 0 placeholder. The first GUI launcher to migrate via this seam
    will pin the ``ctx`` type (likely ``DispatchContext`` from
    ``enigma_engine.core.training.dispatch``) and confirm whether kwargs
    forwarding is needed at all.
    """

    raise NotImplementedError(
        "training_dispatch.run() is a Phase 0 placeholder. First GUI "
        "launcher to migrate via this seam will pin the ctx type and "
        "kwargs shape."
    )
