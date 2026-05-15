"""Hardware-detection service.

Wraps :mod:`enigma_engine.core.hardware_detection`. Used by FORGE pages, CMD
page, and CONFIG page to render hardware summaries and choose memory budgets.

Phase 0 (this slice): signature placeholder; first GUI consumer to migrate
will pin the return shape.
"""

from __future__ import annotations

from typing import Any


def detect() -> dict[str, Any]:
    """Return a hardware-summary dict.

    Phase 0 placeholder. The first GUI consumer to migrate this call will
    pin the dict shape (currently varies across call sites) and decide
    whether to forward to ``detect_hardware()`` or a higher-level helper.
    """

    raise NotImplementedError(
        "hardware.detect() is a Phase 0 placeholder. First GUI consumer "
        "to migrate this call will pin the return shape."
    )
