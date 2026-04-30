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
from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# AutoResearch-2 Stage A — post-generation uncertainty gate.
#
# Spec: SUGGESTIONS.md R-UNPREDICT-1 (Pass 146), AutoResearch-2.
# Signal-driven (calibrated uncertainty markers + low-evidence patterns),
# deterministic — no RNG. Caller chooses threshold.
#
# Stage B (inline `<search>` token in generation loop) is separate work
# that requires logits access and training-loop changes.
# ---------------------------------------------------------------------------

# Calibrated uncertainty / hedge markers. Substring match (case-insensitive).
_HEDGE_PHRASES: tuple[str, ...] = (
    "i'm not sure", "i am not sure", "not sure",
    "i don't know", "i do not know", "don't know",
    "not certain", "uncertain", "unclear",
    "i think", "i believe", "i guess",
    "might be", "may be", "could be",
    "perhaps", "possibly",
    "i can't say", "i cannot say", "hard to say",
)

# Refusal / apology patterns — stronger uncertainty signal than hedging.
_REFUSAL_PHRASES: tuple[str, ...] = (
    "i apologize", "i'm sorry", "i am sorry",
    "i cannot answer", "i can't answer",
    "i'm unable", "i am unable",
    "i don't have", "i do not have",
    "no information", "no data",
)


@dataclass(frozen=True)
class UncertaintyResult:
    """Score from `score_uncertainty()`.

    Attributes:
        score: 0.0 (confident) to 1.0 (highly uncertain).
        reasons: Which signals fired — for logging / debugging.
    """
    score: float
    reasons: tuple[str, ...]


def score_uncertainty(query: str, response: str) -> UncertaintyResult:
    """Score a generated response for uncertainty.

    Stage A of AutoResearch-2. Combines:
      - hedge phrase hits (capped),
      - refusal / apology hits (capped),
      - length anomaly (substantive query, very short response),
      - question-echo (response largely repeats the query).

    Higher score = more uncertain. Caller decides threshold via
    `should_retry_with_research()`.
    """
    if not response or not response.strip():
        return UncertaintyResult(score=1.0, reasons=("empty_response",))

    r = response.lower()
    q = (query or "").lower().strip()
    score = 0.0
    reasons: list[str] = []

    # Hedge phrases — cap so a hedge-heavy reply does not saturate alone.
    hedge_hits = sum(1 for p in _HEDGE_PHRASES if p in r)
    if hedge_hits:
        score += min(0.6, hedge_hits * 0.2)
        reasons.append(f"hedge_phrases={hedge_hits}")

    # Refusal / apology — slightly stronger weight per hit, also capped.
    refusal_hits = sum(1 for p in _REFUSAL_PHRASES if p in r)
    if refusal_hits:
        score += min(0.6, refusal_hits * 0.4)
        reasons.append(f"refusal={refusal_hits}")

    # Length anomaly: substantive question with a tiny response.
    if len(q) > 30 and len(response.strip()) < 50:
        score += 0.3
        reasons.append("short_response")

    # Question echo: response largely repeats the query with little new content.
    if q and len(q) > 20:
        q_clean = q.rstrip("?.! ").strip()
        if q_clean and q_clean in r:
            extra_chars = len(response.strip()) - len(q_clean)
            if extra_chars < 30:
                score += 0.3
                reasons.append("question_echo")

    score = min(1.0, score)
    return UncertaintyResult(score=score, reasons=tuple(reasons))


def should_retry_with_research(
    query: str,
    response: str,
    threshold: float = 0.55,
    enabled: bool = True,
) -> bool:
    """Decide whether to retry generation with web-research context attached.

    Stage A of AutoResearch-2 (Pass 146 spec). Signal-driven; no RNG.

    Args:
        query: The user's original message.
        response: The model's first-pass response.
        threshold: Score >= threshold triggers retry. Default 0.55.
        enabled: Hard off-switch. False always returns False.

    Returns:
        True if the caller should re-run with `auto_research()` context.
    """
    if not enabled:
        return False
    return score_uncertainty(query, response).score >= threshold
