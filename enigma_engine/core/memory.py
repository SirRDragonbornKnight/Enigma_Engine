"""
Persistent AI Memory — Long-Term Fact Storage
==============================================

Gives the AI persistent memory across conversations by maintaining
a human-readable notes file (``data/notes/memory.md``).

Facts are added two ways:
  1. **Automatic extraction** — pattern matching on each exchange
     catches "my name is X", "I prefer X", etc. Works with any model.
  2. **AI command** — ``[CMD]memory.remember <fact>[/CMD]`` lets the
     AI voluntarily save things. Works well with capable models.

Memory retrieval is now active at runtime: the AI uses
``[CMD]memory.search <query>[/CMD]`` when contextually relevant,
rather than receiving all facts in every system prompt.
The ``build_context()`` helper still exists for optional/manual use.

The user can also hand-edit ``data/notes/memory.md`` at any time —
full transparency, no black box.

Directory layout::

    data/notes/memory.md   — human-readable persistent memory

Usage::

    from enigma_engine.core.memory import PersistentMemory

    mem = PersistentMemory()
    mem.add("User's name is Alex")
    mem.add("User prefers dark themes")
    context = mem.build_context(max_tokens=400)
    # → "[MEMORY — Things you remember]\\nUser's name is Alex\\n..."

    # Automatic extraction from user messages
    new_facts = mem.extract_facts("By the way, my name is Alex and I work at NASA")
    # → ["User's name is Alex", "User works at NASA"]
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Default memory file location
_NOTES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "notes"
_MEMORY_FILE = _NOTES_DIR / "memory.md"

# Maximum number of facts to keep (oldest trimmed first)
MAX_FACTS = 50

# Patterns that indicate memorable facts in user messages.
# Each tuple: (compiled regex, format string with group references)
_FACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Name patterns
    (re.compile(
        r"\bmy\s+name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        re.IGNORECASE),
     "User's name is {1}"),
    (re.compile(
        r"\bcall\s+me\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        re.IGNORECASE),
     "User wants to be called {1}"),

    # Work / occupation
    (re.compile(
        r"\bi\s+work\s+(?:at|for)\s+(.+?)(?:\.|,|$)",
        re.IGNORECASE),
     "User works at {1}"),
    (re.compile(
        r"\bi(?:'m|\s+am)\s+a(?:n)?\s+([\w\s]+?)(?:\.|,|\s+and\b|$)",
        re.IGNORECASE),
     "User is a {1}"),

    # Preferences
    (re.compile(
        r"\bi\s+(?:really\s+)?prefer\s+(.+?)(?:\.|,|$)",
        re.IGNORECASE),
     "User prefers {1}"),
    (re.compile(
        r"\bmy\s+(?:favorite|favourite)\s+(\w+)\s+is\s+(.+?)(?:\.|,|$)",
        re.IGNORECASE),
     "User's favorite {1} is {2}"),

    # Location
    (re.compile(
        r"\bi(?:'m|\s+am)\s+(?:from|in|based\s+in)\s+([A-Z][\w\s,]+?)(?:\.|,|$)",
        re.IGNORECASE),
     "User is from {1}"),

    # Explicit remember requests
    (re.compile(
        r"\bremember\s+that\s+(.+?)(?:\.|$)",
        re.IGNORECASE),
     "{1}"),
    (re.compile(
        r"\bdon'?t\s+forget\s+(?:that\s+)?(.+?)(?:\.|$)",
        re.IGNORECASE),
     "{1}"),

    # Projects / languages
    (re.compile(
        r"\bi(?:'m|\s+am)\s+(?:working\s+on|building)\s+(.+?)(?:\.|,|$)",
        re.IGNORECASE),
     "User is working on {1}"),
    (re.compile(
        r"\bi\s+(?:use|program\s+in|code\s+in)\s+(.+?)(?:\.|,|$)",
        re.IGNORECASE),
     "User uses {1}"),
]


class PersistentMemory:
    """Manages the AI's persistent memory file.

    Stores facts as a simple markdown list in ``data/notes/memory.md``.
    Thread-safe for reads and writes via internal lock.
    """

    def __init__(self, memory_path: Path | None = None) -> None:
        self.path = memory_path or _MEMORY_FILE
        self._facts: list[str] = []
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        """Read facts from the memory file."""
        if not self.path.exists():
            self._facts = []
            return
        try:
            text = self.path.read_text(encoding="utf-8")
            facts: list[str] = []
            for line in text.splitlines():
                stripped = line.strip()
                # Parse markdown list items: "- fact" or "* fact"
                if stripped.startswith(("- ", "* ")):
                    fact = stripped[2:].strip()
                    if fact:
                        facts.append(fact)
                # Also accept plain non-empty lines that aren't headers
                elif stripped and not stripped.startswith("#"):
                    facts.append(stripped)
            self._facts = facts
            logger.info("Loaded %d memory facts from %s",
                        len(facts), self.path)
        except OSError as exc:
            logger.warning("Failed to load memory: %s", exc)
            self._facts = []

    # ------------------------------------------------------------------ save
    def _save(self) -> None:
        """Write facts to the memory file."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lines = ["# AI Memory Notes",
                     "# This file is read by the AI at the start of "
                     "every conversation.",
                     "# You can edit it by hand — add, remove, or "
                     "correct anything.",
                     ""]
            for fact in self._facts:
                lines.append(f"- {fact}")
            lines.append("")  # trailing newline
            from enigma_engine.core.safe_save import atomic_write_text
            atomic_write_text(self.path, "\n".join(lines))
        except OSError as exc:
            logger.error("Failed to save memory: %s", exc)

    # --------------------------------------------------------------- add
    def add(self, fact: str) -> bool:
        """Add a fact to memory. Returns True if it was new.

        Deduplicates against existing facts (case-insensitive).
        Trims oldest facts if over MAX_FACTS.
        """
        fact = fact.strip()
        if not fact:
            return False

        with self._lock:
            # Check for duplicates (case-insensitive)
            lower_facts = [f.lower() for f in self._facts]
            if fact.lower() in lower_facts:
                return False

            # Check for near-duplicates — if the new fact updates an old one
            # about the same topic, replace the old one.
            replaced = self._try_replace_outdated(fact)

            if not replaced:
                self._facts.append(fact)

            # Trim oldest if over limit
            while len(self._facts) > MAX_FACTS:
                removed = self._facts.pop(0)
                logger.info("Memory trimmed oldest fact: %s", removed)

            self._save()
            return True

    def _try_replace_outdated(self, new_fact: str) -> bool:
        """Replace an older fact if the new one updates the same topic.

        For example, if memory has "User's name is Bob" and new fact
        is "User's name is Alex", replace the old one.
        Returns True if a replacement was made.
        """
        # Extract a topic prefix — everything before "is", "are", "="
        topic_match = re.match(
            r"^((?:User'?s?\s+)?[\w\s]+?)\s+(?:is|are|=)\s+",
            new_fact, re.IGNORECASE)
        if not topic_match:
            return False

        topic = topic_match.group(1).strip().lower()
        if len(topic) < 4:
            return False

        for i, existing in enumerate(self._facts):
            existing_topic = re.match(
                r"^((?:User'?s?\s+)?[\w\s]+?)\s+(?:is|are|=)\s+",
                existing, re.IGNORECASE)
            if existing_topic:
                if existing_topic.group(1).strip().lower() == topic:
                    logger.info("Memory updated: %r → %r",
                                existing, new_fact)
                    self._facts[i] = new_fact
                    return True
        return False

    # --------------------------------------------------------------- remove
    def remove(self, fact_or_index: str | int) -> bool:
        """Remove a fact by content (substring match) or index.

        Returns True if something was removed.
        """
        with self._lock:
            if isinstance(fact_or_index, int):
                if 0 <= fact_or_index < len(self._facts):
                    self._facts.pop(fact_or_index)
                    self._save()
                    return True
                return False

            # Substring match (case-insensitive)
            needle = fact_or_index.lower()
            for i, f in enumerate(self._facts):
                if needle in f.lower():
                    self._facts.pop(i)
                    self._save()
                    return True
            return False

    # --------------------------------------------------------------- clear
    def clear(self) -> None:
        """Remove all facts."""
        with self._lock:
            self._facts.clear()
            self._save()

    # --------------------------------------------------------------- query
    @property
    def facts(self) -> list[str]:
        """Return a copy of all stored facts."""
        return list(self._facts)

    @property
    def count(self) -> int:
        """Number of stored facts."""
        return len(self._facts)

    def reload(self) -> None:
        """Re-read facts from disk (in case user hand-edited the file)."""
        self._load()

    # -------------------------------------------------------- build_context
    def build_context(self, max_tokens: int = 400) -> str:
        """Build a context string for injection into the system prompt.

        Estimates tokens at ~4 chars per token. Returns empty string
        if no facts are stored.

        Args:
            max_tokens: Approximate token budget for the memory section.

        Returns:
            Formatted string ready to prepend to a system prompt,
            or empty string if nothing to remember.
        """
        if not self._facts:
            return ""

        lines = ["[MEMORY — Things you remember about the user]"]
        char_budget = max_tokens * 4  # rough estimate
        used = len(lines[0])

        for fact in self._facts:
            entry = f"- {fact}"
            if used + len(entry) + 1 > char_budget:
                break
            lines.append(entry)
            used += len(entry) + 1

        lines.append("[END MEMORY]")
        return "\n".join(lines)

    # ------------------------------------------------- extract_facts
    def extract_facts(self, user_message: str) -> list[str]:
        """Extract memorable facts from a user message using patterns.

        This is the automatic extraction layer — no model inference
        needed. Runs simple regexes over the user's text to catch
        common self-disclosures.

        Args:
            user_message: The raw text the user sent.

        Returns:
            List of newly added facts (empty if nothing found).
        """
        if not user_message or len(user_message) < 5:
            return []

        added: list[str] = []
        for pattern, template in _FACT_PATTERNS:
            match = pattern.search(user_message)
            if match:
                # Build the fact string from the template
                groups = match.groups()
                fact = template
                for i, group in enumerate(groups, 1):
                    fact = fact.replace(f"{{{i}}}", group.strip())
                # Clean up
                fact = fact.strip().rstrip(".,;")
                if fact and len(fact) > 3:
                    if self.add(fact):
                        added.append(fact)
                        logger.info("Auto-extracted memory: %s", fact)

        return added

    # ------------------------------------------------- string repr
    def __repr__(self) -> str:
        return f"PersistentMemory(facts={self.count}, path={self.path})"


# =====================================================================
# Module-level singleton — shared across the app
# =====================================================================

_instance: PersistentMemory | None = None
_instance_lock = threading.Lock()


def get_memory(memory_path: Path | None = None) -> PersistentMemory:
    """Get or create the global PersistentMemory instance."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = PersistentMemory(memory_path)
    return _instance
