"""
Enigma Engine - Chat Media Support
=====================================

Utilities for rendering images, animated GIFs, video thumbnails,
and clickable links inline in the chat display.

Uses Pillow for images/GIFs and OpenCV (optional) for video
thumbnail extraction. All imports are optional with graceful
fallback — if Pillow is missing, media features are silently
disabled.

Supported media:
- Images: .png, .jpg, .jpeg, .bmp, .webp, .tiff
- GIFs: .gif (animated with frame cycling)
- Videos: .mp4, .avi, .mov, .mkv, .webm (thumbnail + open in player)
- Links: http/https URLs (clickable, image URLs rendered inline)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Optional imaging imports
try:
    from PIL import Image, ImageTk, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).parent.parent.parent

# -------------------------------------------------------------------
# Limits and defaults
# -------------------------------------------------------------------
MAX_GIF_FRAMES = 120
MAX_IMAGE_DOWNLOAD_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_GIF_DOWNLOAD_BYTES = 20 * 1024 * 1024     # 20 MB
MAX_CHAT_IMAGES = 200  # Cap retained PhotoImage refs per conversation
MAX_CHAT_HISTORY = 500  # Cap in-memory messages (prevents RAM leak during long sessions)
MEDIA_DOWNLOAD_TIMEOUT = 10                   # seconds

# -------------------------------------------------------------------
# File extension sets
# -------------------------------------------------------------------
IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff",
})
GIF_EXTENSIONS = frozenset({".gif"})
VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
})
ALL_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | GIF_EXTENSIONS | VIDEO_EXTENSIONS

# Regex for detecting file paths with media extensions
# Matches: relative paths, absolute paths (C:\...), Unix paths (/.../)
_MEDIA_PATH_RE = re.compile(
    r'(?:'
    # Absolute Windows path: C:\dir\file.ext
    r'[A-Za-z]:\\(?:[^\s\\]+\\)*[^\s\\]+\.(?:' +
    "|".join(e.lstrip(".") for e in ALL_MEDIA_EXTENSIONS) +
    r')'
    r'|'
    # Absolute Unix or relative path: /dir/file.ext or dir/file.ext
    r'(?:[A-Za-z0-9_.][A-Za-z0-9_./\\-]*)?[A-Za-z0-9_-]+\.(?:' +
    "|".join(e.lstrip(".") for e in ALL_MEDIA_EXTENSIONS) +
    r')'
    r')',
    re.IGNORECASE,
)

# Regex for detecting URLs
_URL_RE = re.compile(
    r'https?://[^\s<>"\')\]]+',
    re.IGNORECASE,
)
# Regex for markdown image syntax: ![alt text](url)
_MARKDOWN_IMG_RE = re.compile(
    r'!\[([^\]]*)\]\((https?://[^)]+)\)',
    re.IGNORECASE,
)
# Image URL extensions for inline rendering
_IMAGE_URL_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
})


# -------------------------------------------------------------------
# Detection functions
# -------------------------------------------------------------------

def detect_media_refs(text: str) -> list[dict[str, str]]:
    """Find media file references (paths and image URLs) in text.

    Handles:
    - Local file paths (absolute and relative)
    - Bare image/GIF/video URLs
    - Markdown image syntax: ![alt](url)

    Returns list of dicts with keys:
    - path: the matched path or URL string
    - type: "image", "gif", or "video"
    - source: "file" or "url"
    - alt: alt text (only for markdown images)
    """
    refs: list[dict[str, str]] = []
    seen: set[str] = set()

    # Check for markdown image syntax first: ![alt](url)
    for match in _MARKDOWN_IMG_RE.finditer(text):
        alt = match.group(1)
        url = match.group(2)
        if url in seen:
            continue
        seen.add(url)
        url_path = url.split("?")[0]
        ext = Path(url_path).suffix.lower()
        if ext in GIF_EXTENSIONS:
            mtype = "gif"
        elif ext in VIDEO_EXTENSIONS:
            mtype = "video"
        else:
            # Default to image for markdown images
            mtype = "image"
        ref: dict[str, str] = {
            "path": url, "type": mtype, "source": "url"}
        if alt:
            ref["alt"] = alt
        refs.append(ref)

    # Check for local file paths (skip URL-like strings)
    for match in _MEDIA_PATH_RE.finditer(text):
        path_str = match.group(0)
        if path_str in seen:
            continue
        # Skip false matches that look like URL components
        # (e.g. "raw.githubusercontent.com/.../file.jpg")
        if "://" in text[max(0, match.start() - 10):match.end()]:
            continue
        if ".com/" in path_str or ".org/" in path_str or ".io/" in path_str:
            continue
        seen.add(path_str)
        ext = Path(path_str).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            mtype = "image"
        elif ext in GIF_EXTENSIONS:
            mtype = "gif"
        elif ext in VIDEO_EXTENSIONS:
            mtype = "video"
        else:
            continue
        refs.append({
            "path": path_str, "type": mtype, "source": "file"})

    # Check for bare image/media URLs (not already found via markdown)
    for match in _URL_RE.finditer(text):
        url = match.group(0)
        if url in seen:
            continue
        seen.add(url)
        # Check if URL points to media
        url_path = url.split("?")[0]  # Strip query params
        ext = Path(url_path).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            refs.append({
                "path": url, "type": "image", "source": "url"})
        elif ext in GIF_EXTENSIONS:
            refs.append({
                "path": url, "type": "gif", "source": "url"})
        elif ext in VIDEO_EXTENSIONS:
            refs.append({
                "path": url, "type": "video", "source": "url"})

    return refs


def detect_urls(text: str) -> list[str]:
    """Find all http/https URLs in text."""
    return _URL_RE.findall(text)


# -------------------------------------------------------------------
# Image loading
# -------------------------------------------------------------------

def _resolve_path(path_str: str) -> Path | None:
    """Resolve a media path to an absolute Path.

    Tries the literal path first, then relative to PROJECT_ROOT.
    Returns None if the file does not exist.
    """
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to project root
    rel = PROJECT_ROOT / path_str
    if rel.exists():
        return rel
    # Try in outputs directory sub-paths
    if not p.is_absolute():
        outputs = PROJECT_ROOT / "outputs" / path_str
        if outputs.exists():
            return outputs
    return None


def load_chat_image(
    path_str: str, max_width: int = 400, max_height: int = 300,
) -> Any | None:
    """Load an image file and return a tk.PhotoImage for chat display.

    Resizes to fit within max_width x max_height while preserving
    aspect ratio. Returns None if the file cannot be loaded.
    """
    if not HAS_PIL:
        return None
    try:
        resolved = _resolve_path(path_str)
        if resolved is None:
            return None
        img = Image.open(resolved)
        img = img.convert("RGBA")
        # Resize to fit within bounds
        img.thumbnail((max_width, max_height), Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


def load_url_image(
    url: str, max_width: int = 400, max_height: int = 300,
) -> Any | None:
    """Download an image from URL and return a PhotoImage.

    Returns None if download or conversion fails.
    """
    if not HAS_PIL:
        return None
    try:
        import urllib.request
        import io
        req = urllib.request.Request(
            url, headers={"User-Agent": "EnigmaEngine/1.1"})
        with urllib.request.urlopen(req, timeout=MEDIA_DOWNLOAD_TIMEOUT) as resp:
            data = resp.read(MAX_IMAGE_DOWNLOAD_BYTES)
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGBA")
        img.thumbnail((max_width, max_height), Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


# -------------------------------------------------------------------
# GIF animation
# -------------------------------------------------------------------

def extract_gif_frames(
    path_str: str, max_width: int = 400, max_height: int = 300,
) -> list[tuple[Any, int]] | None:
    """Extract frames from an animated GIF.

    Returns a list of (PhotoImage, duration_ms) tuples for each
    frame, or None if the file cannot be loaded.
    """
    if not HAS_PIL:
        return None
    try:
        resolved = _resolve_path(path_str)
        if resolved is None:
            # Try URL
            if path_str.startswith(("http://", "https://")):
                return _extract_gif_frames_url(
                    path_str, max_width, max_height)
            return None
        img = Image.open(resolved)
        return _process_gif_frames(img, max_width, max_height)
    except Exception:
        return None


def _extract_gif_frames_url(
    url: str, max_width: int, max_height: int,
) -> list[tuple[Any, int]] | None:
    """Download and extract GIF frames from a URL."""
    try:
        import urllib.request
        import io
        req = urllib.request.Request(
            url, headers={"User-Agent": "EnigmaEngine/1.1"})
        with urllib.request.urlopen(req, timeout=MEDIA_DOWNLOAD_TIMEOUT) as resp:
            data = resp.read(MAX_GIF_DOWNLOAD_BYTES)
        img = Image.open(io.BytesIO(data))
        return _process_gif_frames(img, max_width, max_height)
    except Exception:
        return None


def _process_gif_frames(
    img: Any, max_width: int, max_height: int,
) -> list[tuple[Any, int]]:
    """Process PIL Image GIF into list of (PhotoImage, duration_ms)."""
    frames: list[tuple[Any, int]] = []
    try:
        n_frames = getattr(img, "n_frames", 1)
        # Cap frames to prevent memory issues
        max_frames = min(n_frames, MAX_GIF_FRAMES)
        for i in range(max_frames):
            img.seek(i)
            frame = img.copy().convert("RGBA")
            frame.thumbnail((max_width, max_height), Image.LANCZOS)
            duration = img.info.get("duration", 100)
            if duration < 10:
                duration = 100  # Sensible default
            photo = ImageTk.PhotoImage(frame)
            frames.append((photo, int(duration)))
    except EOFError:
        pass
    return frames if frames else None


# -------------------------------------------------------------------
# Video thumbnails
# -------------------------------------------------------------------

def extract_video_thumbnail(
    path_str: str, max_width: int = 400, max_height: int = 300,
) -> Any | None:
    """Extract first frame from a video as a PhotoImage.

    Adds a play button overlay so users know it's clickable.
    Returns None if the video cannot be read.
    """
    if not HAS_CV2 or not HAS_PIL:
        return None
    try:
        resolved = _resolve_path(path_str)
        if resolved is None:
            return None
        cap = cv2.VideoCapture(str(resolved))
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((max_width, max_height), Image.LANCZOS)
        # Add play button overlay
        img = _add_play_overlay(img)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


def _add_play_overlay(img: Any) -> Any:
    """Draw a semi-transparent play triangle on a video thumbnail."""
    try:
        # Ensure RGBA so alpha_composite works
        base = img.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = base.size
        # Draw semi-transparent circle in center
        cx, cy = w // 2, h // 2
        r = min(w, h) // 6
        draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            fill=(0, 0, 0, 140))
        # Draw play triangle
        tri_size = r * 0.6
        points = [
            (cx - tri_size * 0.4, cy - tri_size),
            (cx - tri_size * 0.4, cy + tri_size),
            (cx + tri_size * 0.8, cy),
        ]
        draw.polygon(points, fill=(255, 255, 255, 200))
        return Image.alpha_composite(base, overlay)
    except Exception:
        return img


# -------------------------------------------------------------------
# Text processing — split text into segments
# -------------------------------------------------------------------

def split_text_and_media(text: str) -> list[dict[str, str]]:
    """Split text into alternating text and media reference segments.

    Returns a list of dicts with:
    - type: "text", "image", "gif", "video", or "link"
    - content: the text content or path/url
    - source: "file" or "url" (for media types)
    """
    segments: list[dict[str, str]] = []
    # Combine all pattern matches with positions
    matches: list[tuple[int, int, dict[str, str]]] = []

    # Markdown images: ![alt](url) — highest priority
    for m in _MARKDOWN_IMG_RE.finditer(text):
        url = m.group(2)
        url_path = url.split("?")[0]
        ext = Path(url_path).suffix.lower()
        if ext in GIF_EXTENSIONS:
            mtype = "gif"
        elif ext in VIDEO_EXTENSIONS:
            mtype = "video"
        else:
            mtype = "image"
        matches.append((
            m.start(), m.end(),
            {"type": mtype, "content": url, "source": "url"}))

    # Media file paths (skip URL-like strings)
    for m in _MEDIA_PATH_RE.finditer(text):
        path_str = m.group(0)
        ext = Path(path_str).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            mtype = "image"
        elif ext in GIF_EXTENSIONS:
            mtype = "gif"
        elif ext in VIDEO_EXTENSIONS:
            mtype = "video"
        else:
            continue
        # Skip false matches that look like URL components
        if "://" in text[max(0, m.start() - 10):m.end()]:
            continue
        if ".com/" in path_str or ".org/" in path_str or ".io/" in path_str:
            continue
        # Skip if overlaps with an existing match
        already = False
        for s, e, _ in matches:
            if not (m.end() <= s or m.start() >= e):
                already = True
                break
        if already:
            continue
        matches.append((
            m.start(), m.end(),
            {"type": mtype, "content": path_str, "source": "file"}))

    # URLs
    for m in _URL_RE.finditer(text):
        url = m.group(0)
        url_path = url.split("?")[0]
        ext = Path(url_path).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            mtype = "image"
        elif ext in GIF_EXTENSIONS:
            mtype = "gif"
        elif ext in VIDEO_EXTENSIONS:
            mtype = "video"
        else:
            mtype = "link"
        # Skip if overlaps with an existing match
        already = False
        for s, e, _ in matches:
            if not (m.end() <= s or m.start() >= e):
                already = True
                break
        if already:
            continue
        matches.append((
            m.start(), m.end(),
            {"type": mtype, "content": url, "source": "url"}))

    # Sort by position
    matches.sort(key=lambda x: x[0])

    # Build segments
    pos = 0
    for start, end, ref in matches:
        if start > pos:
            segments.append({
                "type": "text",
                "content": text[pos:start],
                "source": "inline"})
        segments.append(ref)
        pos = end
    if pos < len(text):
        segments.append({
            "type": "text",
            "content": text[pos:],
            "source": "inline"})

    return segments if segments else [
        {"type": "text", "content": text, "source": "inline"}]


def get_media_chat_width(widget_width: int) -> int:
    """Calculate max image width based on chat widget size."""
    # Leave some padding on both sides
    return max(100, min(widget_width - 60, 500))
