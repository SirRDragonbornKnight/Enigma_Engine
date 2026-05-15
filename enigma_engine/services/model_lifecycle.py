"""Model build / load / save / list service.

Consolidates the four-import quartet (``model.Enigma`` + ``model_presets.ForgeConfig``
+ ``model_registry`` + ``tokenizer.get_tokenizer``) that appears 10+ times across
the GUI today into a single surface.

Phase 0 (this slice): signatures only; bodies forward verbatim to core. No
GUI callers migrated yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_model(forge_config: Any) -> Any:
    """Construct a fresh :class:`~enigma_engine.core.model.Enigma` from a
    :class:`~enigma_engine.core.model_presets.ForgeConfig`.

    Phase 0: forwards to core constructors verbatim.
    """

    from enigma_engine.core.model import Enigma

    return Enigma(forge_config)


def load_weights(model: Any, state_path: str | Path, *, strict: bool = True) -> Any:
    """Load a checkpoint into ``model`` via the registry's safe loader.

    Delegates to :func:`enigma_engine.core.model_registry.safe_load_weights`.
    """

    from enigma_engine.core.model_registry import safe_load_weights

    return safe_load_weights(model, str(state_path), strict=strict)


def get_state_dict(model: Any) -> dict[str, Any]:
    """Return a snapshot state dict suitable for atomic torch save.

    Delegates to :func:`enigma_engine.core.model_registry.get_state_dict`.
    """

    from enigma_engine.core.model_registry import get_state_dict as _get_state_dict

    return _get_state_dict(model)


def list_models() -> list[dict[str, Any]]:
    """Return registry entries for all locally available models.

    Delegates to ``enigma_engine.core.model_registry`` listing helper. The
    exact helper name is intentionally not named in the signature so the
    service stays decoupled from the registry's internal API surface; this
    function will be filled in when the first GUI consumer migrates.
    """

    raise NotImplementedError(
        "list_models() is a Phase 0 placeholder. First GUI consumer to "
        "migrate this call will define the return shape and select the "
        "underlying registry helper."
    )
