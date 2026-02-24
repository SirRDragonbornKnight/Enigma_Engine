"""
Enigma Engine - GUI Data Scanners
===================================

Pure functions for discovering bricks, models, profiles,
training data, and saved sessions from the filesystem.
Also holds path constants, config limits, and clamping.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
BRICKS_DIR = PROJECT_ROOT / "bricks"
MODELS_DIR = PROJECT_ROOT / "models"
PROFILES_DIR = PROJECT_ROOT / "profiles"
DATA_DIR = PROJECT_ROOT / "data"
MEMORY_DIR = PROJECT_ROOT / "memory"
SESSIONS_DIR = DATA_DIR / "sessions"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INFO_DIR = PROJECT_ROOT / "information"

# Editable path keys with display names and defaults
PATH_SETTINGS: dict[str, tuple[str, Path]] = {
    "models_dir": ("Models Directory", MODELS_DIR),
    "data_dir": ("Training Data", DATA_DIR),
    "outputs_dir": ("Outputs Directory", OUTPUTS_DIR),
    "profiles_dir": ("Profiles Directory", PROFILES_DIR),
    "sessions_dir": ("Sessions Directory", SESSIONS_DIR),
    "memory_dir": ("Memory Directory", MEMORY_DIR),
    "bricks_dir": ("Bricks Directory", BRICKS_DIR),
}

_PATHS_FILE = DATA_DIR / "path_settings.json"


def load_path_settings() -> dict[str, str]:
    """Load custom path overrides from path_settings.json."""
    if _PATHS_FILE.exists():
        try:
            data = json.loads(
                _PATHS_FILE.read_text(encoding="utf-8"))
            return {k: v for k, v in data.items()
                    if isinstance(v, str)}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_path_settings(paths: dict[str, str]) -> None:
    """Save custom path overrides to path_settings.json."""
    _PATHS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PATHS_FILE.write_text(
        json.dumps(paths, indent=2), encoding="utf-8")


def get_path(key: str) -> Path:
    """Get the current path for a setting, respecting overrides."""
    overrides = load_path_settings()
    if key in overrides and overrides[key]:
        return Path(overrides[key])
    _, default = PATH_SETTINGS.get(key, ("", PROJECT_ROOT))
    return default


# -------------------------------------------------------------------
# Config limits and descriptions
# -------------------------------------------------------------------
CONFIG_LIMITS: dict[str, tuple[float, float, float]] = {
    "temperature": (0.0, 2.0, 0.1),
    "top_p": (0.0, 1.0, 0.05),
    "top_k": (1, 200, 1),
    "max_tokens": (16, 4096, 16),
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
ROUTE_KEYS: list[str] = ["chat", "trainer"]


def clamp_config(name: str, value: float) -> float:
    """Clamp a config value to its valid range."""
    lo, hi, _step = CONFIG_LIMITS.get(name, (value, value, 1))
    return max(lo, min(hi, value))


# -------------------------------------------------------------------
# Scanners - filesystem queries
# -------------------------------------------------------------------

def scan_bricks() -> list[dict[str, Any]]:
    """Discover bricks from bricks/ directory."""
    bricks: list[dict[str, Any]] = []
    if not BRICKS_DIR.exists():
        return bricks
    for d in sorted(BRICKS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        config_path = d / "brick.json"
        if not config_path.exists():
            continue
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            bricks.append({
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
                "dependencies": data.get("dependencies", []),
                "settings": data.get("settings", {}),
                "path": str(d),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return bricks


def scan_models() -> list[dict[str, Any]]:
    """Discover model files from models/ directory."""
    models: list[dict[str, Any]] = []
    if not MODELS_DIR.exists():
        return models
    exts = ("*.pth", "*.pt", "*.gguf", "*.bin", "*.safetensors")
    # Scan root and one level of subdirectories
    search_dirs = [("", MODELS_DIR)]
    for subdir in MODELS_DIR.iterdir():
        if subdir.is_dir():
            search_dirs.append((f"{subdir.name}/", subdir))
    for prefix, directory in search_dirs:
        for ext in exts:
            for p in directory.glob(ext):
                size_mb = p.stat().st_size / (1024 * 1024)
                models.append({
                    "name": f"{prefix}{p.stem}",
                    "path": str(p),
                    "size_mb": round(size_mb, 1),
                    "format": p.suffix.lstrip("."),
                })
    return models


def scan_profiles() -> list[dict[str, Any]]:
    """Discover AI profiles from profiles/ directory."""
    profiles: list[dict[str, Any]] = []
    if not PROFILES_DIR.exists():
        return profiles
    for p in sorted(PROFILES_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            profiles.append({
                "id": p.stem,
                "name": data.get("name", p.stem),
                "description": data.get("description", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return profiles


def scan_training_data() -> list[dict[str, Any]]:
    """Discover training data files from data/ directory."""
    files: list[dict[str, Any]] = []
    if not DATA_DIR.exists():
        return files
    skip = {"gui_settings.json", "overlay_settings.json",
            "ui_settings.json", "prompts.json"}
    for pattern in ("*.txt", "*.jsonl", "*.json"):
        for p in DATA_DIR.glob(pattern):
            if p.name in skip:
                continue
            size_kb = p.stat().st_size / 1024
            files.append({
                "name": p.name,
                "path": str(p),
                "size_kb": round(size_kb, 1),
            })
    return files


def scan_sessions() -> list[dict[str, Any]]:
    """Discover saved chat sessions from memory/ directory."""
    sessions: list[dict[str, Any]] = []
    for search_dir in (MEMORY_DIR, SESSIONS_DIR):
        if not search_dir.exists():
            continue
        for p in sorted(search_dir.glob("*.json")):
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
    return sessions


def scan_docs() -> list[dict[str, Any]]:
    """Discover documentation files from information/, profiles/, and brick docs.

    Returns a flat list of doc entries grouped by category:
    - "guides" from information/ (.md, .txt)
    - "profiles" from profiles/ (.json)
    - "brick:<id>" from bricks/<id>/docs/ (.md, .txt)
    """
    docs: list[dict[str, Any]] = []

    # Engine docs from information/
    if INFO_DIR.exists():
        for p in sorted(INFO_DIR.iterdir()):
            if p.suffix.lower() in (".md", ".txt"):
                docs.append({
                    "name": p.stem.replace("_", " ").title(),
                    "path": str(p),
                    "category": "guides",
                    "filename": p.name,
                })

    # Profile files
    if PROFILES_DIR.exists():
        for p in sorted(PROFILES_DIR.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                docs.append({
                    "name": data.get("name", p.stem),
                    "path": str(p),
                    "category": "profiles",
                    "filename": p.name,
                })
            except (json.JSONDecodeError, OSError):
                continue

    # Brick docs from bricks/<id>/docs/
    if BRICKS_DIR.exists():
        for d in sorted(BRICKS_DIR.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            docs_dir = d / "docs"
            if not docs_dir.exists():
                continue
            for p in sorted(docs_dir.iterdir()):
                if p.suffix.lower() in (".md", ".txt"):
                    docs.append({
                        "name": p.stem.replace("_", " ").title(),
                        "path": str(p),
                        "category": f"brick:{d.name}",
                        "filename": p.name,
                    })

    return docs
