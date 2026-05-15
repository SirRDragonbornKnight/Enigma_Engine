"""Persist personality-probe summaries to disk so successive distill
runs can be compared on-disk.

Unlocks the Personality-5 Row G loss-half gate, which requires the
consistency metric to "show measurable pre->post drift in two
consecutive distill runs first" (SUGGESTIONS.md Pass 156z9dg
parked text).  Without on-disk persistence the GUI log was the only
record and ran 1 was gone by run 2.

Pure stdlib (json, pathlib, time).  Atomic writes via core.safe_save.
Two probe kinds: ``"identity"`` (drift signal from
``personality_data.summarize_identity_probe``) and ``"consistency"``
(drift signal from ``personality_consistency.summarize_consistency``).
Both share the same on-disk layout so a future loss-half slice can
load either with one helper.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

from enigma_engine.core.safe_save import atomic_write_text

ProbeKind = Literal["identity", "consistency"]
_VALID_KINDS: tuple[str, ...] = ("identity", "consistency")


def _checkpoints_dir() -> Path:
    """Return ``models/checkpoints/``.

    Resolved lazily so tests can monkeypatch ``scanners.MODELS_DIR``.
    """
    from enigma_engine.gui import scanners

    return Path(scanners.MODELS_DIR) / "checkpoints"


def save_probe_summary(
    summary: dict,
    *,
    stem: str,
    kind: ProbeKind,
    ts: int | None = None,
) -> Path:
    """Persist ``summary`` for a given ``(stem, kind)`` to disk.

    File path: ``models/checkpoints/{stem}_{kind}_{ts}.json``.

    The on-disk payload wraps ``summary`` with provenance fields
    (``kind``, ``stem``, ``ts``) so a later loader can verify the
    file matches its filename and so two runs can be compared by
    ``ts`` ordering rather than relying on filesystem mtime.

    Parameters
    ----------
    summary : dict
        The probe summary dict to persist.  Typically the return
        value of ``summarize_identity_probe`` or
        ``summarize_consistency``.
    stem : str
        Student model stem (e.g. ``Path(student_path).stem``).  Used
        as the filename prefix so two students do not collide.
    kind : {"identity", "consistency"}
        Which probe family produced the summary.
    ts : int, optional
        Unix epoch seconds.  Defaults to ``int(time.time())``.

    Returns
    -------
    Path
        Absolute path of the written file.

    Raises
    ------
    ValueError
        If ``kind`` is not in ``("identity", "consistency")`` or if
        ``stem`` is empty.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"Invalid probe kind {kind!r}; expected one of {_VALID_KINDS}"
        )
    if not stem:
        raise ValueError("stem must be non-empty")
    if ts is None:
        ts = int(time.time())
    out_dir = _checkpoints_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_{kind}_{ts}.json"
    payload = {
        "kind": kind,
        "stem": stem,
        "ts": ts,
        "summary": summary,
    }
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_recent_probe_summaries(
    stem: str,
    kind: ProbeKind,
    *,
    n: int = 2,
) -> list[dict]:
    """Load up to ``n`` most-recent probe summaries for ``(stem, kind)``.

    Ordering: newest first by the ``ts`` recorded inside each file
    (NOT by filesystem mtime, since restores or copies can break
    mtime).  Files whose JSON is malformed or whose ``kind``/``stem``
    fields disagree with the filename are skipped silently — this
    helper is operational, not validating.

    Returns ``[]`` if the checkpoints directory does not exist or no
    files match the pattern.

    Each returned dict has the shape written by
    :func:`save_probe_summary`: ``{"kind", "stem", "ts", "summary"}``.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"Invalid probe kind {kind!r}; expected one of {_VALID_KINDS}"
        )
    if not stem:
        raise ValueError("stem must be non-empty")
    if n <= 0:
        return []
    out_dir = _checkpoints_dir()
    if not out_dir.exists():
        return []
    pattern = f"{stem}_{kind}_*.json"
    results: list[dict] = []
    for p in out_dir.glob(pattern):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("kind") != kind or data.get("stem") != stem:
            continue
        if not isinstance(data.get("ts"), int):
            continue
        results.append(data)
    results.sort(key=lambda d: d["ts"], reverse=True)
    return results[:n]
