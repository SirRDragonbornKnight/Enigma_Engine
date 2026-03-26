"""
Pre-training dataset utilities — download, process, and clean text data.
===========================================================================

Provides tools for preparing text data for language model pre-training:
- process_text_corpus: Load and clean text from files or directories
- clean_text: Normalize whitespace, remove control chars
- estimate_token_count: Quick token count estimate
- download_dataset: Download known datasets via huggingface_hub
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Maximum file size (bytes) for individual files during corpus processing.
# Files exceeding this limit are skipped to prevent OOM crashes.
MAX_FILE_SIZE: int = 512_000_000  # 512 MB

# =========================================================================
# Known datasets — registry of downloadable pre-training corpora
# =========================================================================

KNOWN_DATASETS: dict[str, dict] = {
    "tinystories": {
        "name": "TinyStories",
        "description": (
            "Short children's stories (~500M tokens). "
            "Good for initial pre-training validation."
        ),
        "repo_id": "roneneldan/TinyStories",
        "repo_type": "dataset",
        "size_estimate": "~2 GB uncompressed text",
    },
    "tinystories-instruct": {
        "name": "TinyStories Instruct",
        "description": (
            "Instruction-following version of TinyStories. "
            "Stories with prompts and completions."
        ),
        "repo_id": "roneneldan/TinyStories",
        "repo_type": "dataset",
        "size_estimate": "~2 GB uncompressed text",
    },
}


# =========================================================================
# Text cleaning
# =========================================================================

def clean_text(text: str) -> str:
    """Clean raw text for pre-training.

    - Remove null bytes and control characters (except newline/tab)
    - Normalize excessive whitespace runs
    - Strip trailing whitespace per line
    - Collapse 3+ consecutive blank lines to 2

    Args:
        text: Raw text to clean.

    Returns:
        Cleaned text.
    """
    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove control chars except \n \r \t
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize \r\n to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse 3+ consecutive newlines to 2 (one blank line max)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =========================================================================
# Token count estimation
# =========================================================================

def estimate_token_count(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count from text length.

    Uses a rough heuristic: ~4 characters per token for English text.
    This matches typical BPE tokenizer behavior.

    Args:
        text: Text to estimate.
        chars_per_token: Average characters per token (default 4.0).

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


# =========================================================================
# Text corpus processing
# =========================================================================

def process_text_corpus(
    source: Path | str,
    *,
    text_key: str = "text",
) -> str:
    """Load and clean text data from a file or directory.

    Handles multiple formats:
    - .txt files: read as-is and clean
    - .jsonl files: extract text from each JSON object using text_key
    - Directories: process all .txt and .jsonl files found

    Args:
        source: Path to a file or directory of text data.
        text_key: JSON key to extract text from in JSONL files.

    Returns:
        Cleaned text ready for pre-training.
    """
    source = Path(source)

    if source.is_dir():
        return _process_directory(source, text_key=text_key)
    elif source.is_file():
        return _process_file(source, text_key=text_key)
    else:
        logger.warning("Source path does not exist: %s", source)
        return ""


def _process_file(path: Path, *, text_key: str = "text") -> str:
    """Process a single file."""
    # Guard against oversized files that would OOM
    try:
        file_size = path.stat().st_size
    except OSError:
        file_size = 0
    if file_size > MAX_FILE_SIZE:
        logger.warning(
            "Skipping %s (%d MB) — exceeds MAX_FILE_SIZE (%d MB)",
            path, file_size // 1_000_000, MAX_FILE_SIZE // 1_000_000)
        return ""

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return _process_jsonl(path, text_key=text_key)
    elif suffix == ".json":
        return _process_json(path, text_key=text_key)
    else:
        # Default: treat as plain text
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", path, exc)
            return ""
        return clean_text(text)


def _process_jsonl(path: Path, *, text_key: str = "text") -> str:
    """Extract text from a JSONL file."""
    texts: list[str] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    val = obj.get(text_key, "")
                    if isinstance(val, str) and val.strip():
                        texts.append(val.strip())
    except OSError as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return ""

    return clean_text("\n\n".join(texts))


def _process_json(path: Path, *, text_key: str = "text") -> str:
    """Extract text from a JSON file (array of objects)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return ""

    if isinstance(data, list):
        texts = []
        for obj in data:
            if isinstance(obj, dict):
                val = obj.get(text_key, "")
                if isinstance(val, str) and val.strip():
                    texts.append(val.strip())
        return clean_text("\n\n".join(texts))

    return ""


def _process_directory(
    dir_path: Path,
    *,
    text_key: str = "text",
) -> str:
    """Process all text/JSONL files in a directory."""
    parts: list[str] = []

    # Collect and sort for deterministic order
    files = sorted(dir_path.iterdir())
    for f in files:
        if not f.is_file():
            continue
        suffix = f.suffix.lower()
        if suffix in (".txt", ".jsonl", ".json"):
            chunk = _process_file(f, text_key=text_key)
            if chunk:
                parts.append(chunk)

    return "\n\n".join(parts)


# =========================================================================
# Dataset download
# =========================================================================

def download_dataset(
    name: str,
    dest_dir: Path | str,
    *,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Path:
    """Download a known pre-training dataset.

    Uses huggingface_hub (already a project dependency) to download
    dataset files.  Returns the path to the downloaded directory.

    Args:
        name: Key from KNOWN_DATASETS (e.g. "tinystories").
        dest_dir: Directory to store downloaded files.
        progress_fn: Optional callback for progress messages.

    Returns:
        Path to the directory containing downloaded files.

    Raises:
        ValueError: If name is not in KNOWN_DATASETS.
        ImportError: If huggingface_hub is not installed.
        RuntimeError: If download fails.
    """
    if name not in KNOWN_DATASETS:
        available = ", ".join(KNOWN_DATASETS.keys())
        raise ValueError(
            f"Unknown dataset: {name}. Available: {available}")

    info = KNOWN_DATASETS[name]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if progress_fn:
        progress_fn(f"Downloading {info['name']}...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for dataset download. "
            "Install with: pip install huggingface-hub"
        ) from None

    try:
        local_dir = dest_dir / name
        snapshot_download(
            repo_id=info["repo_id"],
            repo_type=info.get("repo_type", "dataset"),
            local_dir=str(local_dir),
            ignore_patterns=["*.md", "*.py", ".gitattributes"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {info['name']}: {exc}"
        ) from exc

    if progress_fn:
        progress_fn(f"Downloaded {info['name']} to {local_dir}")

    return local_dir
