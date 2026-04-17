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
    if key in overrides and overrides[key]:
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
