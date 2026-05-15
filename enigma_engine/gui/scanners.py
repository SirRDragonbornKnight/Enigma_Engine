"""
Enigma Engine - GUI Data Scanners
===================================

Pure functions for discovering mods, models,
training data, and saved sessions from the filesystem.
Also holds path constants, config limits, and clamping.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODS_DIR = PROJECT_ROOT / "mods"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
MEMORY_DIR = PROJECT_ROOT / "memory"
SESSIONS_DIR = DATA_DIR / "sessions"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INFO_DIR = PROJECT_ROOT / "information"
TRAINER_DIR = INFO_DIR / "trainer"
PROMPTS_DIR = DATA_DIR / "prompts"
NOTES_DIR = DATA_DIR / "notes"

# Editable path keys with display names and defaults
PATH_SETTINGS: dict[str, tuple[str, Path]] = {
    "models_dir": ("Models Directory", MODELS_DIR),
    "data_dir": ("Training Data", DATA_DIR),
    "outputs_dir": ("Outputs Directory", OUTPUTS_DIR),
    "sessions_dir": ("Sessions Directory", SESSIONS_DIR),
    "memory_dir": ("Memory Directory", MEMORY_DIR),
    "mods_dir": ("Mods Directory", MODS_DIR),
}

_PATHS_FILE = DATA_DIR / "path_settings.json"
_ROUTES_FILE = DATA_DIR / "route_assignments.json"


def load_path_settings() -> dict[str, str]:
    """Load custom path overrides from path_settings.json."""
    if _PATHS_FILE.exists():
        try:
            data = json.loads(
                _PATHS_FILE.read_text(encoding="utf-8"))
            return {k: v for k, v in data.items()
                    if isinstance(v, str)}
        except json.JSONDecodeError as e:
            logger.error("Path settings corrupted: %s", e)
        except OSError as e:
            logger.warning("Cannot read path settings: %s", e)
    return {}


def save_path_settings(paths: dict[str, str]) -> None:
    """Save custom path overrides to path_settings.json."""
    # Validate paths before saving
    validated = {}
    for key, value in paths.items():
        if not isinstance(key, str) or not isinstance(value, str):
            logger.warning("Skipping invalid path setting: %s=%r", key, value)
            continue
        if not value.strip():
            continue  # Skip empty values (use default)
        p = Path(value)
        if not p.is_absolute():
            logger.warning("Skipping non-absolute path for %s: %s", key, value)
            continue
        validated[key] = value
    _PATHS_FILE.parent.mkdir(parents=True, exist_ok=True)
    from enigma_engine.core.safe_save import atomic_write_json
    atomic_write_json(_PATHS_FILE, validated)


def get_path(key: str) -> Path:
    """Get the current path for a setting, respecting overrides."""
    overrides = load_path_settings()
    if overrides.get(key):
        return Path(overrides[key])
    _, default = PATH_SETTINGS.get(key, ("", PROJECT_ROOT))
    return default


def load_route_assignments() -> dict[str, str]:
    """Load saved route assignments from route_assignments.json."""
    if _ROUTES_FILE.exists():
        try:
            data = json.loads(
                _ROUTES_FILE.read_text(encoding="utf-8"))
            return {k: v for k, v in data.items()
                    if isinstance(v, str)}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_route_assignments(assignments: dict[str, str | None]) -> None:
    """Save route assignments to route_assignments.json."""
    # Only save non-None assignments
    clean = {k: v for k, v in assignments.items() if v}
    _ROUTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    from enigma_engine.core.safe_save import atomic_write_json
    atomic_write_json(_ROUTES_FILE, clean)


# -------------------------------------------------------------------
# Config limits and descriptions
# -------------------------------------------------------------------
CONFIG_LIMITS: dict[str, tuple[float, float, float]] = {
    "temperature": (0.0, 2.0, 0.1),
    "top_p": (0.0, 1.0, 0.05),
    "top_k": (1, 200, 1),
    "max_tokens": (1, 1000000, 1),
    "repetition_penalty": (1.0, 2.0, 0.05),
}

CONFIG_DISPLAY_NAMES: dict[str, str] = {
    "temperature": "Creativity",
    "top_p": "Diversity",
    "top_k": "Word Choices",
    "max_tokens": "Response Length",
    "repetition_penalty": "Repetition Control",
}

CONFIG_DESCRIPTIONS: dict[str, str] = {
    "temperature": (
        "How creative the AI is. Lower values give focused, "
        "predictable answers. Higher values give more varied, "
        "creative responses."),
    "top_p": (
        "Controls response diversity. Lower values stick to the "
        "most likely words. Higher values allow more variety."),
    "top_k": (
        "How many word choices the AI considers at each step. "
        "Lower is more focused, higher is more diverse."),
    "max_tokens": (
        "Maximum length of each AI response in tokens. "
        "Roughly 1 token equals 3/4 of a word."),
    "repetition_penalty": (
        "How strongly the AI avoids repeating itself. "
        "Higher values reduce repetition."),
}


# Built-in route targets for model assignment
ROUTE_KEYS: list[str] = ["chat", "trainer", "student"]


def clamp_config(name: str, value: float) -> float:
    """Clamp a config value to its valid range."""
    lo, hi, _step = CONFIG_LIMITS.get(name, (value, value, 1))
    return max(lo, min(hi, value))


# -------------------------------------------------------------------
# Scanners - filesystem queries
# -------------------------------------------------------------------

def scan_mods() -> list[dict[str, Any]]:
    """Discover mods from mods/ directory."""
    mods: list[dict[str, Any]] = []
    if not MODS_DIR.exists():
        return mods
    for d in sorted(MODS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        config_path = d / "mod.json"
        if not config_path.exists():
            continue
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            mods.append({
                "id": data.get("id", d.name),
                "name": data.get("name", d.name),
                "description": data.get("description", ""),
                "version": data.get("version", "?"),
                "port": data.get("port", 0),
                "commands": [
                    c.get("name", "") for c in data.get("commands", [])
                ],
                "commands_full": data.get("commands", []),
                "ui": data.get("ui", {}),
                "prompt": data.get("prompt", ""),
                "rules": data.get("rules", []),
                "dependencies": data.get("dependencies", []),
                "settings": data.get("settings", {}),
                "path": str(d),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return mods


# Maximum file size (bytes) to load via torch.load for param counting.
# Files larger than this use a file-size heuristic instead, avoiding
# multi-GB RAM spikes from loading huge checkpoints.
_PARAM_COUNT_LOAD_LIMIT = 2 * 1024 ** 3  # 2 GB


def _count_params_native(path: Path) -> str | None:
    """Return a human-readable param count for a native .pth/.pt model.

    If the checkpoint contains a ``target_size`` key (saved at creation
    time), that value is normalised to ``"XB"`` format and returned
    directly so the display matches what the user typed.

    For files larger than ~2 GB the function avoids ``torch.load``
    entirely — it peeks at zip metadata for ``target_size`` and falls
    back to a file-size heuristic.  This prevents loading 70 GB+
    checkpoints into RAM just to count parameters.

    Otherwise falls back to counting parameters from the state dict
    and returns e.g. ``"19.08B"`` or ``"1.50B"`` on success, ``None``
    on failure.
    """
    file_size = path.stat().st_size

    # --- Fast path: peek at zip metadata for target_size ---
    target = _peek_target_size(path)
    if target:
        return _normalise_size_label(target)

    # --- Large files: file-size heuristic (avoid torch.load) ---
    if file_size > _PARAM_COUNT_LOAD_LIMIT:
        return _estimate_params_from_size(path, file_size)

    # --- Small files: load with torch for accurate count ---
    try:
        from enigma_engine.core.model_registry import (
            safe_load_weights)
        checkpoint = safe_load_weights(
            str(path), map_location="cpu")

        sd = checkpoint.get("model_state_dict") or checkpoint
        total = sum(v.numel() for v in sd.values()
                    if hasattr(v, "numel"))
        del checkpoint, sd  # free immediately
        if total <= 0:
            return None
        return _format_param_count(total)
    except Exception:
        return _estimate_params_from_size(path, file_size)


def _peek_target_size(path: Path) -> str | None:
    """Read target_size from a .pth zip without loading tensors."""
    import zipfile
    try:
        if not zipfile.is_zipfile(str(path)):
            return None
        with zipfile.ZipFile(str(path), "r") as zf:
            # PyTorch saves metadata as <archive_name>/data.pkl
            pkl_names = [n for n in zf.namelist()
                         if n.endswith("/data.pkl") or n == "data.pkl"]
            if not pkl_names:
                return None
            import pickle
            import io
            raw = zf.read(pkl_names[0])
            # Only read the first 8 KB — target_size is a top-level
            # dict key written early; no need to unpickle tensors.
            # Use a restricted unpickler to block arbitrary code execution.
            class _SafeUnpickler(pickle.Unpickler):
                _ALLOWED = {
                    "collections": {"OrderedDict"},
                    "torch._utils": {"_rebuild_tensor_v2"},
                    "torch": {"FloatStorage", "LongStorage",
                              "IntStorage", "HalfStorage",
                              "DoubleStorage", "BFloat16Storage",
                              "ShortStorage", "ByteStorage",
                              "Size"},
                }

                def find_class(self, module: str, name: str):
                    allowed = self._ALLOWED.get(module)
                    if allowed and name in allowed:
                        return super().find_class(module, name)
                    raise pickle.UnpicklingError(
                        f"Blocked: {module}.{name}")

            data = _SafeUnpickler(io.BytesIO(raw)).load()
            if isinstance(data, dict):
                ts = data.get("target_size")
                if ts and isinstance(ts, str):
                    return ts
    except Exception as exc:
        logger.debug("Could not peek target_size from %s: %s", path, exc)
    return None


def _estimate_params_from_size(path: Path, file_size: int) -> str | None:
    """Estimate param count from total tensor data in a .pth zip.

    Sums the uncompressed sizes of all ``data/*`` entries (tensor
    storage files) and divides by 2 (fp16) for a rough param count.
    Most modern models use fp16/bf16. Falls back to raw file size / 2
    for non-zip files.
    """
    import zipfile
    try:
        if zipfile.is_zipfile(str(path)):
            tensor_bytes = 0
            with zipfile.ZipFile(str(path), "r") as zf:
                for info in zf.infolist():
                    if "/data/" in info.filename:
                        tensor_bytes += info.file_size
            if tensor_bytes > 0:
                total = tensor_bytes // 2  # fp16/bf16
                return _format_param_count(total)
    except Exception as exc:
        logger.debug("Zip inspection failed for param count: %s", exc)
    # Non-zip fallback: raw file size / 2
    total = file_size // 2
    if total > 0:
        return _format_param_count(total)
    return None


def _format_param_count(total: int) -> str:
    """Format an integer param count as human-readable string."""
    if total >= 1_000_000_000:
        return f"{total / 1e9:.2f}B"
    if total >= 1_000_000:
        return f"{total / 1e6:.1f}M"
    return f"{total:,}"


def _normalise_size_label(text: str) -> str:
    """Normalise a user-entered size like ``'8b'`` to ``'8B'``.

    Handles ``'8b'`` \u2192 ``'8B'``, ``'1.5b'`` \u2192 ``'1.5B'``,
    ``'500m'`` \u2192 ``'0.50B'``, ``'small'`` \u2192 ``'small'``.
    """
    import re
    t = text.strip().lower()
    m = re.match(r'^(\d+(?:\.\d+)?)\s*(b|m)?$', t)
    if not m:
        return text  # preset name like "small" \u2014 return as-is
    num = float(m.group(1))
    suffix = m.group(2)
    if suffix == "m":
        num_b = num / 1000
        return f"{num_b:.2f}B"
    if suffix == "b":
        # Strip unnecessary trailing zeros: 8.0 -> 8, 1.50 -> 1.5
        if num == int(num):
            return f"{int(num)}B"
        return f"{num:g}B"
    # Raw number (presumably params) \u2014 convert to B or M
    if num >= 1_000_000_000:
        return f"{num / 1e9:.2f}B"
    if num >= 1_000_000:
        return f"{num / 1e6:.1f}M"
    return text


def scan_models() -> list[dict[str, Any]]:
    """Discover model files from models/ directory.

    Sharded safetensors (model-00001-of-00005.safetensors, etc.)
    are grouped into a single entry with the combined size.
    """
    models: list[dict[str, Any]] = []
    if not MODELS_DIR.exists():
        return models
    exts = ("*.pth", "*.pt", "*.gguf", "*.bin", "*.safetensors")
    # Scan root and one level of subdirectories
    search_dirs = [("", MODELS_DIR)]
    for subdir in MODELS_DIR.iterdir():
        if subdir.is_dir():
            search_dirs.append((f"{subdir.name}/", subdir))

    # Track sharded safetensors to merge into one entry
    import re
    _shard_re = re.compile(r'^(.+)-(\d+)-of-(\d+)$')

    for prefix, directory in search_dirs:
        # Collect shards per directory: base_name → list of paths
        shard_groups: dict[str, list[Path]] = {}
        standalone: list[Path] = []

        for ext in exts:
            for p in directory.glob(ext):
                stem = p.stem
                m = _shard_re.match(stem)
                if m and p.suffix == ".safetensors":
                    # Sharded safetensors — group by base name
                    base = m.group(1)
                    shard_groups.setdefault(base, []).append(p)
                else:
                    standalone.append(p)

        # Add standalone files as individual entries
        for p in standalone:
            size_mb = p.stat().st_size / (1024 * 1024)
            fmt = p.suffix.lstrip(".")
            # Estimate param count from file size
            # For .pth: ~4 bytes/param (fp32) or ~2 bytes/param (fp16)
            # For .gguf: ~0.5-2 bytes/param depending on quantisation
            params = None
            try:
                size_bytes = p.stat().st_size
                if fmt == "gguf":
                    # Q4 ≈ 0.5 bytes/param→ params ≈ size * 2
                    params = int(size_bytes * 2)
                elif fmt in ("pth", "pt", "bin"):
                    # fp16 ≈ 2 bytes/param (most common save format)
                    params = int(size_bytes / 2)
            except OSError:
                pass
            models.append({
                "name": f"{prefix}{p.stem}",
                "path": str(p),
                "size_mb": round(size_mb, 1),
                "format": fmt,
                "params": params,
            })

        # Merge sharded safetensors into one entry per group
        for base, shard_paths in shard_groups.items():
            total_mb = sum(
                s.stat().st_size / (1024 * 1024) for s in shard_paths)
            total_bytes = sum(
                s.stat().st_size for s in shard_paths)
            count = len(shard_paths)
            # Use parent directory name if shards are in a subdirectory
            display = prefix.rstrip("/") if prefix else base
            # safetensors use fp16 ≈ 2 bytes/param
            params = int(total_bytes / 2)
            models.append({
                "name": display,
                "path": str(shard_paths[0].parent),
                "size_mb": round(total_mb, 1),
                "format": f"safetensors ({count} shards)",
                "params": params,
            })

    return models


def scan_lora_adapters(
    base_model_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Discover PEFT-format LoRA adapters under ``models/``.

    Pass 156s (LoRA-1b foundation): scans for directories containing
    an ``adapter_config.json`` (the PEFT canonical format). Each
    adapter entry carries the metadata needed to apply it at chat time
    without sidecar files: rank, alpha, target_modules, base model.

    Args:
        base_model_path: When provided, only adapters whose recorded
            ``base_model_name_or_path`` matches the **stem** of this
            path are returned. Pass ``None`` to list every adapter on
            disk (used by the audit/migration view; the chat dropdown
            should always pass the active base model so the user
            cannot apply a mismatched adapter).

    Returns:
        List of dicts: ``{name, path, base, rank, alpha, target_modules, size_kb}``.

    Notes:
        - Searches ``models/checkpoints/`` and ``models/lora_adapters/``
          one directory deep — adapter directories are flat.
        - Skips directories whose ``adapter_config.json`` is missing,
          unreadable, or malformed (logged at WARNING).
        - ``base_model_name_or_path`` may be a path or a HuggingFace
          repo id; we match by stem so both styles work.
    """
    adapters: list[dict[str, Any]] = []
    if not MODELS_DIR.exists():
        return adapters

    search_roots = [
        MODELS_DIR / "checkpoints",
        MODELS_DIR / "lora_adapters",
    ]

    base_stem = (
        Path(base_model_path).stem if base_model_path else None
    )

    for root in search_roots:
        if not root.exists():
            continue
        for adapter_dir in root.iterdir():
            if not adapter_dir.is_dir():
                continue
            cfg_path = adapter_dir / "adapter_config.json"
            if not cfg_path.exists():
                continue
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "Skipping malformed adapter at %s: %s",
                    adapter_dir, e)
                continue

            recorded_base = cfg.get("base_model_name_or_path") or ""
            if base_stem is not None:
                if not recorded_base:
                    # Adapter has no recorded base — refuse to match
                    # any specific base; skip when filtering.
                    continue
                if Path(recorded_base).stem != base_stem:
                    continue

            # Sum bytes across the directory for the size column.
            try:
                total_bytes = sum(
                    p.stat().st_size for p in adapter_dir.iterdir()
                    if p.is_file())
                size_kb = round(total_bytes / 1024, 1)
            except OSError:
                size_kb = 0.0

            target_modules = cfg.get("target_modules") or []
            if isinstance(target_modules, str):
                target_modules = [target_modules]

            adapters.append({
                "name": adapter_dir.name,
                "path": str(adapter_dir),
                "base": recorded_base,
                "rank": int(cfg.get("r", 0) or 0),
                "alpha": int(cfg.get("lora_alpha", 0) or 0),
                "target_modules": list(target_modules),
                "size_kb": size_kb,
            })

    adapters.sort(key=lambda a: a["name"].lower())
    return adapters


def scan_training_data() -> list[dict[str, Any]]:
    """Discover training data files from data/ and one-level subdirectories."""
    files: list[dict[str, Any]] = []
    if not DATA_DIR.exists():
        return files
    skip = {"gui_settings.json", "prompts.json",
            "route_assignments.json", "path_settings.json",
            "progress.json"}
    for pattern in ("*.txt", "*.jsonl", "*.json", "*.pdf", "*.docx"):
        # Top-level data/ files
        for p in DATA_DIR.glob(pattern):
            if p.name in skip:
                continue
            size_kb = p.stat().st_size / 1024
            files.append({
                "name": p.name,
                "path": str(p),
                "size_kb": round(size_kb, 1),
            })
        # One-level-deep: data/<subdir>/file (e.g. pretrain/combined.txt)
        for p in DATA_DIR.glob(f"*/{pattern}"):
            if p.name in skip:
                continue
            rel = p.relative_to(DATA_DIR)
            size_kb = p.stat().st_size / 1024
            files.append({
                "name": str(rel).replace("\\", "/"),
                "path": str(p),
                "size_kb": round(size_kb, 1),
            })
    return files


def _pick_first_match(
        files: list[dict[str, Any]],
        preferred_tails: list[str]) -> str:
    """Generic smart-default picker for FORGE data-file dropdowns.

    D-11c (Pass 156l): generalises `_pick_default_training_file` so the
    same "prefer collected output if present, else first-scanned"
    pattern can be reused by other FORGE pickers (DPO pair-data,
    pre-training data, vision data). Match is path-tail-suffix on a
    forward-slash-normalised path so both Windows and POSIX path styles
    resolve the same way. The order of `preferred_tails` is the
    preference order — first hit wins.

    Args:
        files: list of dicts each containing "name" and "path" keys
            (the shape `scan_*` helpers in this module emit).
        preferred_tails: ordered list of path-tail suffixes (forward-
            slash-separated) to prefer. First match wins.

    Returns: the matched path, or `files[0]["path"]` when no preferred
    tail matches, or "" when `files` is empty.
    """
    if not files:
        return ""
    for tail in preferred_tails:
        for f in files:
            name = f.get("name", "")
            path = f.get("path", "")
            normalised = path.replace("\\", "/")
            if (name.replace("\\", "/").endswith(tail)
                    or normalised.endswith(tail)):
                return path
    return files[0].get("path", "")


def _pick_default_training_file(files: list[dict[str, Any]]) -> str:
    """Pick the smartest default for the FORGE training-data picker.

    D-11b (Pass 156i9): when the user has run `collect_finetuning_data.py`
    and `data/finetune/combined_finetune.txt` is on disk, prefer it over
    the placeholder `data/training.txt` — the user already opted into
    reasoning data by running the collector, so surface it without
    making them navigate the directory tree. Otherwise fall back to the
    first scanned file (legacy behaviour). Empty list → "".

    Pass 156l (D-11c): now a thin wrapper over `_pick_first_match` so
    the same logic powers other FORGE pickers without duplication.
    """
    return _pick_first_match(
        files, ["finetune/combined_finetune.txt"])


def _pick_default_dpo_data_file(files: list[dict[str, Any]]) -> str:
    """Pick the smartest default for the FORGE DPO/APO pair-data picker.

    D-11c (Pass 156l): mirrors the training-data picker. When the user
    has shipped curated preference pairs to `data/dpo/combined.jsonl`
    (or `data/finetune/dpo_pairs.jsonl`), prefer those over scattered
    scratch files. Same fall-back-to-first behaviour. Empty list → "".
    """
    return _pick_first_match(
        files,
        [
            "dpo/combined.jsonl",
            "finetune/dpo_pairs.jsonl",
        ])


def _pick_default_pretrain_file(files: list[dict[str, Any]]) -> str:
    """Pick the smartest default for the FORGE pre-training data picker.

    D-11c (Pass 156l): when `data/pretrain/combined.txt` exists (output
    of `collect_pretraining_data.py --combine-only`), prefer it over
    scattered scratch files. Same fall-back-to-first behaviour. Empty
    list → "".
    """
    return _pick_first_match(
        files,
        [
            "pretrain/combined.txt",
            "pretrain/combined_pretrain.txt",
        ])


# D-11c-DPO (Pass 156q): modes that consume DPO/APO-style preference
# pairs from the shared `train_data_var` picker. When the user switches
# into one of these modes, the smart-default picker should prefer the
# DPO file layout over the SFT corpus default. The set is the union of
# alignment modes plus RLHF/Self-Play (which also expect JSONL
# preference data via the same picker).
_PREFERENCE_MODES: frozenset[str] = frozenset({
    "RLHF", "Self-Play", "GRPO", "ReMax", "SimPO", "ORPO", "APO",
})


def _pick_default_train_data_for_mode(
        files: list[dict[str, Any]], mode: str) -> str:
    """Pick the smartest default for the shared FORGE data picker per mode.

    D-11c-DPO (Pass 156q): the FORGE training-data picker is shared
    across SFT modes (Basic/LoRA) and preference-pair modes
    (RLHF/Self-Play/GRPO/ReMax/SimPO/ORPO/APO). Before this helper, the
    picker always defaulted to the SFT default — switching into APO
    surfaced an irrelevant text file rather than `data/dpo/combined.jsonl`.

    This helper routes the lookup to the mode-appropriate picker:
      - preference-pair modes → `_pick_default_dpo_data_file`
      - everything else (Basic/LoRA/Distill/AI-Guided/etc.)
        → `_pick_default_training_file`

    The caller is responsible for deciding *whether* to apply the new
    default (it should preserve any user-customised path); this helper
    only computes what the default *would* be for the given mode.
    """
    if mode in _PREFERENCE_MODES:
        return _pick_default_dpo_data_file(files)
    return _pick_default_training_file(files)


def _resolve_anchor_path(saved: str | None) -> Path | None:
    """Resolve the anchor JSONL path to use for `BackgroundTrainer`.

    Continuous-3b (Pass 156o): the GUI persists `anchor_data_path` in
    `gui_settings.json` as either a non-empty path string (user
    override) or empty/missing (use repo default). This helper
    centralises the three-branch resolution so the desktop launcher
    and the config-page status label agree:

      - non-empty `saved` → return `Path(saved)` as-is, even if it
        does not exist (status label flags missing files loudly,
        rather than silently falling back to the default).
      - empty/None → return the repo default
        (`<project>/data/anchor_examples.jsonl`) when it exists,
        else None (replay rehearses recent chat only).
    """
    if saved:
        return Path(saved)
    default = DATA_DIR / "anchor_examples.jsonl"
    return default if default.exists() else None


def scan_sessions() -> list[dict[str, Any]]:
    """Discover saved chat sessions from memory/ directory."""
    sessions: list[dict[str, Any]] = []
    if not MEMORY_DIR.exists():
        return sessions
    for p in sorted(MEMORY_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            msg_count = data.get(
                "message_count",
                len(data.get("messages", [])))
            sessions.append({
                "name": data.get("name", p.stem),
                "path": str(p),
                "messages": msg_count,
                "saved_at": data.get("saved_at", 0),
            })
        except (json.JSONDecodeError, OSError):
            continue
    # Most recent first (by saved_at timestamp)
    sessions.sort(key=lambda s: s.get("saved_at", 0), reverse=True)
    return sessions


_DOC_EXTS = (".md", ".txt", ".pdf", ".docx", ".json")

def scan_docs() -> list[dict[str, Any]]:
    """Discover documentation files from information/ and mod docs.

    Returns a flat list of doc entries grouped by category:
    - "guides" from information/ (.md, .txt, .pdf, .docx) — top-level files only
    - "trainer" from information/trainer/ (.md, .txt, .pdf, .docx)
    - "mod:<id>" from mods/<id>/docs/ (.md, .txt, .pdf, .docx)
    """
    docs: list[dict[str, Any]] = []

    # Engine docs from information/ (top-level files only)
    if INFO_DIR.exists():
        for p in sorted(INFO_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in _DOC_EXTS:
                docs.append({
                    "name": p.stem.replace("_", " ").title(),
                    "path": str(p),
                    "category": "guides",
                    "filename": p.name,
                })

    # Trainer docs from information/trainer/
    if TRAINER_DIR.exists():
        for p in sorted(TRAINER_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in _DOC_EXTS:
                docs.append({
                    "name": p.stem.replace("_", " ").title(),
                    "path": str(p),
                    "category": "trainer",
                    "filename": p.name,
                })

    # Training data files from data/
    if DATA_DIR.exists():
        skip = {"gui_settings.json", "prompts.json",
                "route_assignments.json", "path_settings.json"}
        for pattern in ("*.txt", "*.jsonl", "*.pdf", "*.docx"):
            for p in sorted(DATA_DIR.glob(pattern)):
                if p.name in skip:
                    continue
                size_kb = round(p.stat().st_size / 1024, 1)
                docs.append({
                    "name": f"{p.name} ({size_kb} KB)",
                    "path": str(p),
                    "category": "data",
                    "filename": p.name,
                })

    # Mod docs from mods/<id>/docs/
    if MODS_DIR.exists():
        for d in sorted(MODS_DIR.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            docs_dir = d / "docs"
            if not docs_dir.exists():
                continue
            for p in sorted(docs_dir.iterdir()):
                if p.suffix.lower() in _DOC_EXTS:
                    docs.append({
                        "name": p.stem.replace("_", " ").title(),
                        "path": str(p),
                        "category": f"mod:{d.name}",
                        "filename": p.name,
                    })

    # Route prompts from data/prompts/
    if PROMPTS_DIR.exists():
        for p in sorted(PROMPTS_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in (".md", ".txt"):
                docs.append({
                    "name": p.stem.replace("_", " ").title(),
                    "path": str(p),
                    "category": "prompts",
                    "filename": p.name,
                })

    # Notes from data/notes/ (memory, user notes)
    if NOTES_DIR.exists():
        for p in sorted(NOTES_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in (".md", ".txt"):
                docs.append({
                    "name": p.stem.replace("_", " ").title(),
                    "path": str(p),
                    "category": "notes",
                    "filename": p.name,
                })

    return docs


def load_route_prompt(route: str) -> str:
    """Load the user-editable prompt for a route (chat, trainer, student).

    Looks for ``data/prompts/<route>.md`` or ``<route>.txt``.
    Returns the file contents stripped, or empty string if not found.
    """
    for suffix in (".md", ".txt"):
        path = PROMPTS_DIR / f"{route}{suffix}"
        if path.exists():
            try:
                return path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# VISION DATA SCANNING
# ─────────────────────────────────────────────────────────────────────────────

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


def _extract_video_frames(
    video_path: Path,
    max_frames: int = 8,
) -> list:
    """
    Extract evenly-spaced frames from a video file as PIL Images.

    Requires OpenCV (``pip install opencv-python``). Returns an empty
    list if cv2 is not available or the video cannot be opened.

    Args:
        video_path: Path to the video file.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of PIL Image objects (RGB).
    """
    try:
        import cv2
    except ImportError:
        logger.debug("opencv-python not installed — skipping video: %s", video_path)
        return []

    try:
        from PIL import Image
    except ImportError:
        logger.debug("Pillow not installed — skipping video: %s", video_path)
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return []

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            return []

        n_sample = min(max_frames, total_frames)
        indices = [int(i * total_frames / n_sample) for i in range(n_sample)]

        frames: list = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)

        return frames
    finally:
        cap.release()


def scan_vision_data(directory: str | Path) -> list[dict]:
    """
    Discover image-text and video-text training pairs in a directory.

    Supports three formats:
    1. Paired files: image.png + image.txt in the same folder
    2. JSONL: each line has {"image": "path", "text": "caption"}
    3. Video files: video.mp4 + video.txt — auto-extracts frames,
       each frame paired with the caption text.

    Args:
        directory: Path to scan for vision training data.

    Returns:
        List of dicts with "image" (path str or PIL Image) and "text" keys.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    pairs: list[dict] = []

    # -- Strategy 1: JSONL files with image+text fields --
    for jsonl_path in directory.glob("*.jsonl"):
        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "image" in record and "text" in record:
                        pairs.append({
                            "image": str(record["image"]),
                            "text": str(record["text"]),
                        })
        except OSError:
            logger.debug(f"Could not read JSONL: {jsonl_path}")

    # -- Strategy 2: Paired files (image + same-name .txt) --
    for img_path in directory.iterdir():
        if img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            try:
                caption = txt_path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if caption:
                pairs.append({
                    "image": str(img_path),
                    "text": caption,
                })

    # -- Strategy 3: Video files (video + same-name .txt) --
    for vid_path in directory.iterdir():
        if vid_path.suffix.lower() not in _VIDEO_EXTENSIONS:
            continue
        txt_path = vid_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        try:
            caption = txt_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not caption:
            continue

        frames = _extract_video_frames(vid_path)
        if not frames:
            logger.debug("No frames extracted from video: %s", vid_path)
            continue

        logger.info("Extracted %d frames from video: %s", len(frames), vid_path.name)
        for frame in frames:
            pairs.append({
                "image": frame,  # PIL Image object
                "text": caption,
            })

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO DATA SCANNING (ARCH-1d)
# ─────────────────────────────────────────────────────────────────────────────


def scan_audio_data(directory: str | Path) -> list[dict]:
    """
    Discover audio-text training pairs in a directory.

    Supports two formats:
    1. JSONL: each line has {"audio": "path", "text": "transcript"}
    2. Paired files: clip.wav + clip.txt in the same folder

    Args:
        directory: Path to scan for audio training data.

    Returns:
        List of dicts with "audio" (path string) and "text" keys —
        the exact shape ``Trainer.train_audio`` and the ``mode="audio"``
        dispatcher expect.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    pairs: list[dict] = []

    # -- Strategy 1: JSONL files with audio+text fields --
    for jsonl_path in directory.glob("*.jsonl"):
        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "audio" in record and "text" in record:
                        pairs.append({
                            "audio": str(record["audio"]),
                            "text": str(record["text"]),
                        })
        except OSError:
            logger.debug(f"Could not read JSONL: {jsonl_path}")

    # -- Strategy 2: Paired files (audio + same-name .txt) --
    for clip_path in directory.iterdir():
        if clip_path.suffix.lower() not in _AUDIO_EXTENSIONS:
            continue
        txt_path = clip_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        try:
            transcript = txt_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if transcript:
            pairs.append({
                "audio": str(clip_path),
                "text": transcript,
            })

    return pairs
