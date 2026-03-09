"""Document readers for PDF and DOCX files.

Provides ``read_pdf()`` and ``read_docx()`` with graceful fallbacks
when the optional dependencies (``pymupdf`` / ``python-docx``) are
not installed.  A unified ``read_document()`` dispatches by extension.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------
_HAS_PYMUPDF = False
_HAS_DOCX = False

try:
    import fitz  # pymupdf
    _HAS_PYMUPDF = True
except ImportError:
    pass

try:
    import docx as _docx  # python-docx
    _HAS_DOCX = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def read_pdf(path: str | Path) -> str:
    """Extract text from a PDF file.

    Requires ``pymupdf`` (``pip install pymupdf``).

    Returns:
        Extracted text with pages separated by newlines.

    Raises:
        ImportError: If pymupdf is not installed.
        FileNotFoundError: If the file does not exist.
    """
    if not _HAS_PYMUPDF:
        raise ImportError(
            "pymupdf is required for PDF reading. "
            "Install it with: pip install pymupdf"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[str] = []
    with fitz.open(str(path)) as doc:  # type: ignore[arg-type]
        for page in doc:
            text = page.get_text()
            if text and text.strip():
                pages.append(text.strip())

    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def read_docx(path: str | Path) -> str:
    """Extract text from a Word (.docx) file.

    Requires ``python-docx`` (``pip install python-docx``).

    Returns:
        Extracted text with paragraphs separated by newlines.

    Raises:
        ImportError: If python-docx is not installed.
        FileNotFoundError: If the file does not exist.
    """
    if not _HAS_DOCX:
        raise ImportError(
            "python-docx is required for DOCX reading. "
            "Install it with: pip install python-docx"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX not found: {path}")

    document = _docx.Document(str(path))
    paragraphs: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

# Extensions handled by this module
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def read_document(path: str | Path) -> str | None:
    """Read a document by extension, returning text or *None* on failure.

    Supports ``.pdf`` and ``.docx``.  Returns *None* (and logs a
    warning) when the required library is missing or the file cannot
    be parsed — callers can safely skip the file.
    """
    path = Path(path)
    ext = path.suffix.lower()

    try:
        if ext == ".pdf":
            return read_pdf(path)
        if ext == ".docx":
            return read_docx(path)
    except ImportError as exc:
        logger.warning("Skipping %s: %s", path.name, exc)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path.name, exc)

    return None


def pdf_available() -> bool:
    """Return True if PDF reading is available."""
    return _HAS_PYMUPDF


def docx_available() -> bool:
    """Return True if DOCX reading is available."""
    return _HAS_DOCX
