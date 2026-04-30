"""Migrate legacy ``.pth`` LoRA files to a quarantine directory.

Pass 156t (LoRA-1b UX): the previous ``LoraTrainer.save_adapter``
manual-fallback path emitted ``.pth`` files containing only the
``param.requires_grad`` weights — legitimate LoRA tensors in a custom
format without ``adapter_config.json``, so PEFT cannot load them. Pass
156s replaced that fallback with a PEFT-directory-only save path.

This script moves any pre-156s ``.pth`` files out of the active scan
roots into ``models/checkpoints/legacy_lora_pth/`` so they don't
clutter the new MODELS-page LoRA list, and writes a ``NOTICE.txt``
explaining the format change. Running the script twice is safe: it
only moves files that aren't already in the quarantine directory.

Scope (deliberately conservative — we do NOT touch ``models/``):
- ``models/lora_adapters/*.pth``     (top-level loose files)
- ``models/checkpoints/*_lora.pth``  (named-pattern matches only)

A ``*_lora.pth`` file inside ``models/checkpoints/`` is the only
pattern we touch in that directory, because checkpoints/ is also used
for legitimate base-model intermediate snapshots (Pass 122 protected
checkpoints) and indiscriminate moves there would lose training state.

Usage:
    python migrate_legacy_lora.py            # dry-run, prints plan
    python migrate_legacy_lora.py --apply    # actually moves files
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
QUARANTINE_DIR = MODELS_DIR / "checkpoints" / "legacy_lora_pth"

NOTICE_TEXT = """\
Legacy LoRA `.pth` files (Pass 156s+ quarantine)
=================================================

These files were produced by the pre-Pass-156s
`LoraTrainer.save_adapter` manual-fallback path. The format stored
only the trainable adapter tensors (`param.requires_grad`) WITHOUT
the `adapter_config.json` metadata that PEFT needs to load them.

The current chat engine (`EnigmaEngine.apply_adapter`) requires PEFT
directory format (a folder with `adapter_config.json` +
`adapter_model.safetensors`). These `.pth` files cannot be applied
to your model at chat time.

What you can do:
- **Retrain**: re-run LoRA training with the current `LoraTrainer`,
  which now always writes a PEFT directory.
- **Discard**: if you no longer need the weights, delete this
  directory.

This NOTICE was created by `migrate_legacy_lora.py`.
"""


def _find_legacy_files() -> list[Path]:
    """Return absolute paths of legacy ``.pth`` LoRA files.

    Scans the two locations with the highest probability of containing
    pre-156s LoRA artifacts. ``models/`` itself is intentionally
    excluded — that directory holds full base models.
    """
    found: list[Path] = []

    lora_dir = MODELS_DIR / "lora_adapters"
    if lora_dir.is_dir():
        for p in lora_dir.iterdir():
            if p.is_file() and p.suffix == ".pth":
                found.append(p)

    ckpt_dir = MODELS_DIR / "checkpoints"
    if ckpt_dir.is_dir():
        for p in ckpt_dir.iterdir():
            if p.is_file() and p.name.endswith("_lora.pth"):
                found.append(p)

    return found


def _move_one(src: Path, dest_dir: Path) -> Path:
    """Move ``src`` into ``dest_dir``, renaming on collision.

    Returns the final destination path. Idempotent: skips when ``src``
    is already inside ``dest_dir`` (no-op return of ``src``).
    """
    if dest_dir in src.parents:
        return src

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / src.name
    if target.exists():
        # Avoid clobbering — append a numeric suffix.
        stem, suffix = target.stem, target.suffix
        i = 1
        while True:
            candidate = dest_dir / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                target = candidate
                break
            i += 1
    shutil.move(str(src), str(target))
    return target


def migrate(apply: bool = False) -> dict:
    """Run the migration. Returns a summary dict.

    Args:
        apply: When ``True``, actually move files. When ``False``
            (default), just print the plan.

    Returns:
        ``{"found": [Path...], "moved": [Path...], "applied": bool}``
    """
    found = _find_legacy_files()
    moved: list[Path] = []

    if not found:
        return {"found": [], "moved": [], "applied": apply}

    if apply:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        notice = QUARANTINE_DIR / "NOTICE.txt"
        if not notice.exists():
            notice.write_text(NOTICE_TEXT, encoding="utf-8")

        for src in found:
            dest = _move_one(src, QUARANTINE_DIR)
            if dest != src:
                moved.append(dest)

    return {"found": found, "moved": moved, "applied": apply}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate legacy `.pth` LoRA files to "
            "models/checkpoints/legacy_lora_pth/."))
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually move files. Without this flag, runs in "
             "dry-run mode and only prints the plan.")
    args = parser.parse_args()

    result = migrate(apply=args.apply)
    found = result["found"]
    moved = result["moved"]

    if not found:
        print("No legacy `.pth` LoRA files found.")
        return 0

    print(f"Found {len(found)} legacy `.pth` LoRA file(s):")
    for p in found:
        try:
            rel = p.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = p
        print(f"  {rel}")

    if not args.apply:
        print()
        print("Dry-run only — re-run with --apply to move them to:")
        try:
            rel_quar = QUARANTINE_DIR.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_quar = QUARANTINE_DIR
        print(f"  {rel_quar}")
        return 0

    print()
    print(f"Moved {len(moved)} file(s) to:")
    try:
        rel_quar = QUARANTINE_DIR.relative_to(PROJECT_ROOT)
    except ValueError:
        rel_quar = QUARANTINE_DIR
    print(f"  {rel_quar}")
    print(f"NOTICE.txt: {(QUARANTINE_DIR / 'NOTICE.txt').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
