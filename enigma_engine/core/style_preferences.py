"""Style preferences — Layer 2 of the PERSONA-2 layered personality model.

This module owns the user-side surface-style settings (verbosity, formality,
response length, output format hints). It does NOT own identity — that lives
in the LoRA adapter weights and is locked to training. See AA code maker.md
§Project Goal Constraints and SUGGESTIONS.md Block 4.5.

Black-box invariant
-------------------
The model artifact is never touched by this module. Style preferences live in
``data/style_preferences.json`` as user-side runtime config that follows the
user, not the model. If the user shares a LoRA adapter with someone else, that
person gets the clean trained core with no inherited preferences.

Defaults preserve current behavior
----------------------------------
When every field is at its default, :meth:`StylePreferences.is_default` returns
True and the prompt-injection layer (Slice 2) skips injection entirely. This
gives existing users a zero-overhead path and prevents accidental regressions.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Final

from .safe_save import atomic_write_json

logger = logging.getLogger(__name__)


# Allowed enum values, pinned as module-level constants so the GUI / chat
# command surface can validate against the same single source of truth.
VERBOSITY_VALUES: Final[tuple[str, ...]] = ("terse", "normal", "verbose")
FORMALITY_VALUES: Final[tuple[str, ...]] = ("casual", "neutral", "formal")
LENGTH_VALUES: Final[tuple[str, ...]] = ("short", "medium", "long")


# Canonical on-disk location. Module-level so tests can monkeypatch it
# (same pattern as ``router._DEFAULT_ANCHOR_PATH``).
STYLE_PREFERENCES_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent.parent / "data" / "style_preferences.json"
)

# Block markers — the LLM sees this as instructions, not user input,
# and the user can see exactly what's being injected.
_BLOCK_OPEN: Final[str] = "[USER STYLE PREFERENCES]"
_BLOCK_CLOSE: Final[str] = "[/USER STYLE PREFERENCES]"


@dataclass
class StylePreferences:
    """Layer 2 of PERSONA-2: user-side surface style preferences.

    All defaults preserve the pre-PERSONA-2 behavior. Existing users who do
    not customize anything see no change in prompts, model behavior, or
    response shape.

    Attributes:
        verbosity: How much the AI says by default.
            ``"terse"`` | ``"normal"`` (default) | ``"verbose"``.
        formality: Conversational register.
            ``"casual"`` | ``"neutral"`` (default) | ``"formal"``.
        default_response_length: Preferred default length.
            ``"short"`` | ``"medium"`` (default) | ``"long"``.
        prefer_code_examples: Add code examples when explaining technical
            concepts. Default False.
        prefer_bullet_points: Prefer bullet lists over prose for multi-part
            answers. Default False.

    Raises:
        ValueError: If any enum field is set to a value outside its allowed
            set. Validation runs in ``__post_init__`` so invalid state cannot
            be constructed; chat-command overrides and GUI controls validate
            against the same constants.
    """

    verbosity: str = "normal"
    formality: str = "neutral"
    default_response_length: str = "medium"
    prefer_code_examples: bool = False
    prefer_bullet_points: bool = False

    def __post_init__(self) -> None:
        if self.verbosity not in VERBOSITY_VALUES:
            raise ValueError(
                f"verbosity must be one of {VERBOSITY_VALUES}, "
                f"got {self.verbosity!r}"
            )
        if self.formality not in FORMALITY_VALUES:
            raise ValueError(
                f"formality must be one of {FORMALITY_VALUES}, "
                f"got {self.formality!r}"
            )
        if self.default_response_length not in LENGTH_VALUES:
            raise ValueError(
                f"default_response_length must be one of {LENGTH_VALUES}, "
                f"got {self.default_response_length!r}"
            )

    def is_default(self) -> bool:
        """True when every field matches the dataclass default.

        Used by the prompt-injection layer (Slice 2) to skip the
        ``[USER STYLE PREFERENCES]`` block entirely when the user has not
        customized anything — zero overhead for users who do not use this
        feature.
        """
        return self == StylePreferences()


def load_style_preferences(path: str | Path) -> "StylePreferences":
    """Load style preferences from JSON; fall back to defaults on any error.

    Behavior matrix (loud on real issue, silent on normal path):

    - File does not exist → defaults, no log (normal first-run case).
    - File exists but is corrupt JSON → defaults, WARNING.
    - File exists but is not a JSON object → defaults, WARNING.
    - File exists with valid JSON but invalid enum value → defaults, WARNING.
    - File exists with valid JSON containing unknown fields → known fields are
      loaded, unknown fields are silently dropped (forward-compat with future
      schema additions).

    Args:
        path: JSON file path.

    Returns:
        :class:`StylePreferences` instance.
    """
    path = Path(path)
    if not path.exists():
        return StylePreferences()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Could not read style preferences from %s: %s. Using defaults.",
            path,
            exc,
        )
        return StylePreferences()

    if not isinstance(data, dict):
        logger.warning(
            "Style preferences file %s did not contain a JSON object "
            "(got %s). Using defaults.",
            path,
            type(data).__name__,
        )
        return StylePreferences()

    # Forward-compat: drop fields the current schema does not know about.
    valid_field_names = {f.name for f in fields(StylePreferences)}
    filtered = {k: v for k, v in data.items() if k in valid_field_names}

    try:
        return StylePreferences(**filtered)
    except (ValueError, TypeError) as exc:
        logger.warning(
            "Invalid style preferences in %s: %s. Using defaults.",
            path,
            exc,
        )
        return StylePreferences()


def save_style_preferences(prefs: "StylePreferences", path: str | Path) -> None:
    """Save style preferences atomically.

    Black-box invariant: this only writes the target JSON file. It does NOT
    touch any file under ``models/`` or any model artifact. The
    ``tests/test_style_preferences.py::TestStylePreferencesBlackBoxInvariant``
    suite gates this contract at the file-system level.

    Args:
        prefs: :class:`StylePreferences` to save.
        path: Target JSON file path.
    """
    atomic_write_json(path, asdict(prefs))


def render_style_preferences_block(prefs: "StylePreferences") -> str:
    """Render style preferences as a system-prompt block.

    Used by ``_prepare_chat`` (Slice 2 injection) to add a
    ``[USER STYLE PREFERENCES]`` block to the system prompt when the
    user has customized any preference. Returns the empty string when
    ``prefs.is_default()`` so the prompt is unchanged for users who
    haven't customized anything — zero overhead.

    The block format:

    - Open marker ``[USER STYLE PREFERENCES]``
    - One directive line per non-default field (defaults skipped)
    - Close marker ``[/USER STYLE PREFERENCES]``

    Bracketed markers serve two audiences: the LLM sees the block as
    instructions (a distinct frame from user input), and the user can
    see exactly what's being applied without reading code.

    Args:
        prefs: :class:`StylePreferences` to render.

    Returns:
        Rendered block, or ``""`` when all fields are at default.
    """
    if prefs.is_default():
        return ""

    directives: list[str] = []

    # Verbosity — only emit a directive when non-default.
    if prefs.verbosity == "terse":
        directives.append(
            "- Be terse: keep responses brief and to the point."
        )
    elif prefs.verbosity == "verbose":
        directives.append(
            "- Be thorough: include context, examples, and reasoning."
        )

    if prefs.formality == "casual":
        directives.append("- Use a casual, conversational register.")
    elif prefs.formality == "formal":
        directives.append("- Use a formal, professional register.")

    if prefs.default_response_length == "short":
        directives.append(
            "- Prefer short responses unless the topic demands more detail."
        )
    elif prefs.default_response_length == "long":
        directives.append(
            "- Prefer thorough responses with supporting detail."
        )

    if prefs.prefer_code_examples:
        directives.append(
            "- Add code examples when explaining technical concepts."
        )

    if prefs.prefer_bullet_points:
        directives.append(
            "- Prefer bullet points for multi-part answers."
        )

    if not directives:
        # Defensive: ``is_default()`` already guards this branch, but if
        # a future field has only one "non-default" value that maps to no
        # directive, return empty rather than a marker-only block.
        return ""

    body = "\n".join(directives)
    return f"{_BLOCK_OPEN}\n{body}\n{_BLOCK_CLOSE}"


def get_style_preferences_block_for_prompt(
    path: "str | Path | None" = None,
) -> str:
    """Convenience: load current preferences and render the prompt block.

    Used at the chat seam (``engine_chat._prepare_chat``) to inject
    style preferences into the system prompt without the caller having
    to know about file paths or default-skip logic.

    Args:
        path: Optional override path. ``None`` uses
            :data:`STYLE_PREFERENCES_PATH`.

    Returns:
        Rendered block, or ``""`` when preferences are at defaults
        (whether due to a missing file, defaults on disk, or load
        failure — see :func:`load_style_preferences`).
    """
    target = Path(path) if path is not None else STYLE_PREFERENCES_PATH
    prefs = load_style_preferences(target)
    return render_style_preferences_block(prefs)


__all__ = [
    "VERBOSITY_VALUES",
    "FORMALITY_VALUES",
    "LENGTH_VALUES",
    "STYLE_PREFERENCES_PATH",
    "StylePreferences",
    "load_style_preferences",
    "save_style_preferences",
    "render_style_preferences_block",
    "get_style_preferences_block_for_prompt",
]
