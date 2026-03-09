"""
Atomic save helpers for model weights and data files.

Writes to a temporary file first, then atomically replaces
the target. This prevents corruption if the process is killed,
the system loses power, or the user closes the window mid-save.

Usage:
    from enigma_engine.core.safe_save import atomic_torch_save, atomic_safetensors_save
    atomic_torch_save({"model_state_dict": sd, "config": cfg}, path)
    atomic_safetensors_save(state_dict, path)    # preferred — no pickle

    from enigma_engine.core.safe_save import atomic_write_text, atomic_write_json
    atomic_write_text(path, content)             # plain text files
    atomic_write_json(path, data)                # JSON files
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def atomic_torch_save(data: dict, path: str | Path) -> None:
    """Save a PyTorch checkpoint atomically.

    Writes to ``<path>.tmp`` first, then replaces the target via
    ``os.replace`` (atomic on both Windows NTFS and Linux ext4).
    On failure the temp file is cleaned up and the original is
    left untouched.
    """
    import torch

    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up partial temp file on any failure
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_safetensors_save(
    tensors: Dict[str, "torch.Tensor"],  # noqa: F821
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """Save tensors in safetensors format atomically.

    Preferred over ``atomic_torch_save`` for pure state dicts —
    safetensors uses no pickle so loading cannot execute code.

    Args:
        tensors: Flat dict of ``{name: Tensor}``.
        path: Target ``.safetensors`` file.
        metadata: Optional string-to-string metadata header.
    """
    from safetensors.torch import save_file

    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(tmp_path), metadata=metadata)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | Path, content: str) -> None:
    """Write a text file atomically with fsync and backup rotation.

    1. Backs up the existing file to ``<path>.bak`` (if it exists).
    2. Writes *content* to ``<path>.tmp`` with ``fsync`` for durability.
    3. Atomically replaces the target via ``os.replace``.

    On failure the temp file is cleaned up and the original is
    left untouched.

    Args:
        path: Target file path.
        content: Text content to write.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        # Backup existing file before overwriting
        if path.exists():
            bak_path = path.with_suffix(path.suffix + ".bak")
            try:
                import shutil
                shutil.copy2(str(path), str(bak_path))
            except OSError as exc:
                logger.debug("Backup copy failed for %s: %s", path, exc)

        # Write to temp file with fsync for durability
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_json(
    path: str | Path,
    data: object,
    indent: int = 2,
    ensure_ascii: bool = False,
    default: object = None,
) -> None:
    """Write a JSON file atomically with fsync and backup rotation.

    Convenience wrapper around :func:`atomic_write_text` that handles
    ``json.dumps`` serialization.

    Args:
        path: Target file path.
        data: JSON-serializable data.
        indent: JSON indentation (default 2).
        ensure_ascii: Whether to escape non-ASCII (default False).
        default: Fallback serializer for non-standard types (e.g., ``str``).
    """
    import json

    content = json.dumps(
        data,
        indent=indent,
        ensure_ascii=ensure_ascii,
        default=default,
    )
    atomic_write_text(path, content)
