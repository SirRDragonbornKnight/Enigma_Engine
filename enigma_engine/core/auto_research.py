"""
Auto Research — Proactive web research for AI context.

When web access is enabled, automatically searches for information
relevant to the user's query and injects results into the AI's
context so it has fresh data to work with.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting — minimum interval between web searches
# ---------------------------------------------------------------------------
_MIN_SEARCH_INTERVAL = 5.0  # seconds
_last_search_time = 0.0
_rate_lock = threading.Lock()

# ---------------------------------------------------------------------------
# LRU cache — avoid duplicate searches within a session
# ---------------------------------------------------------------------------
_CACHE_MAX = 100
_search_cache: dict[str, str] = {}  # normalized query → result string
_cache_lock = threading.Lock()


def _normalize_query(query: str) -> str:
    """Normalize a query for cache keying."""
    return " ".join(query.lower().split())


def _cache_get(key: str) -> str | None:
    """Thread-safe cache lookup."""
    with _cache_lock:
        return _search_cache.get(key)


def _cache_put(key: str, value: str) -> None:
    """Thread-safe cache insert with LRU eviction."""
    with _cache_lock:
        if len(_search_cache) >= _CACHE_MAX:
            # Evict oldest entry (first key in insertion order)
            oldest = next(iter(_search_cache))
            del _search_cache[oldest]
        _search_cache[key] = value


def _check_rate_limit() -> bool:
    """Return True if enough time has passed since last search."""
    global _last_search_time
    with _rate_lock:
        now = time.monotonic()
        if now - _last_search_time < _MIN_SEARCH_INTERVAL:
            return False
        _last_search_time = now
        return True


def _fetch_one(url: str, max_chars: int) -> str:
    """Fetch text from a single URL (for parallel execution)."""
    try:
        from enigma_engine.core.web_utils import fetch_page_text
        text = fetch_page_text(url, max_chars=max_chars)
        if text and len(text) > 50:
            return text[:max_chars]
    except Exception:
        pass
    return ""


def auto_research(query: str, max_results: int = 3,
                  max_chars_per_result: int = 500) -> str:
    """Search the web for context relevant to a query.

    Returns a formatted context string for injection into the
    system prompt.  Returns empty string if no results found or
    web utilities are unavailable.

    Features:
        - LRU cache (100 entries) avoids duplicate searches
        - Rate limiting (5s minimum between searches)
        - Parallel page fetching via ThreadPoolExecutor
    """
    if not query or len(query.strip()) < 3:
        return ""

    # Check cache first
    cache_key = _normalize_query(query)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Rate limiting
    if not _check_rate_limit():
        logger.debug("Auto-research rate-limited, skipping")
        return ""

    try:
        from enigma_engine.core.web_utils import ddg_search, ddg_image_search
    except ImportError:
        logger.debug("web_utils not available for auto-research")
        return ""

    try:
        results = ddg_search(query, max_results=max_results)
    except Exception as exc:
        logger.debug("Auto-research search failed: %s", exc)
        return ""

    if not results:
        _cache_put(cache_key, "")
        return ""

    context_parts: list[str] = []
    urls_to_fetch: list[tuple[int, str]] = []

    for i, r in enumerate(results[:max_results]):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")

        if title or snippet:
            entry = f"• {title}"
            if snippet:
                entry += f": {snippet}"
            context_parts.append(entry)

        if url:
            urls_to_fetch.append((i, url))

    # Fetch page text in parallel
    if urls_to_fetch:
        with ThreadPoolExecutor(max_workers=min(3, len(urls_to_fetch))) as pool:
            futures = {
                pool.submit(_fetch_one, url, max_chars_per_result): idx
                for idx, url in urls_to_fetch
            }
            fetched: dict[int, str] = {}
            for future in as_completed(futures, timeout=60):
                idx = futures[future]
                try:
                    text = future.result(timeout=30)
                except Exception:
                    logger.debug("Web fetch failed for idx=%d", idx)
                    text = None
                if text:
                    fetched[idx] = text

            # Insert fetched text in original order
            for idx in sorted(fetched):
                context_parts.append(f"  {fetched[idx]}")

    if not context_parts:
        _cache_put(cache_key, "")
        return ""

    # Pull 1-2 relevant images so the AI can embed them
    try:
        images = ddg_image_search(query, max_results=2)
        for img in images:
            img_title = img.get("title", "image")
            img_url = img.get("url", "")
            if img_url:
                context_parts.append(
                    f"• Image: ![{img_title}]({img_url})")
    except Exception:
        logger.debug("Auto-research image search failed")

    result = (
        "[WEB RESEARCH — auto-retrieved context]\n"
        + "\n".join(context_parts)
        + "\n[END WEB RESEARCH]"
    )

    _cache_put(cache_key, result)
    return result


def should_auto_research(query: str) -> bool:
    """Determine if a query would benefit from web research.

    Returns True for queries that ask about facts, current events,
    or topics where web data would help.  Skips short, trivial,
    or code-only messages.
    """
    if not query or len(query.strip()) < 10:
        return False

    q = query.lower().strip()

    # Skip code-only messages
    if q.startswith("```") or q.startswith("def ") or q.startswith("class "):
        return False

    # Skip simple greetings / commands
    skip_prefixes = (
        "hi", "hello", "hey", "thanks", "thank you",
        "ok", "yes", "no", "sure", "bye",
    )
    for prefix in skip_prefixes:
        if q == prefix or (q.startswith(prefix + " ") and len(q) < 20):
            return False

    # Research-worthy indicators
    research_words = (
        "what is", "what are", "how to", "how do",
        "why does", "why is", "when did", "when was",
        "who is", "who was", "where is", "where was",
        "explain", "difference between", "compare",
        "latest", "current", "recent", "news",
        "best way to", "recommend", "tutorial",
    )
    for word in research_words:
        if word in q:
            return True

    # Questions (ending with ?)
    return bool(q.endswith("?"))
