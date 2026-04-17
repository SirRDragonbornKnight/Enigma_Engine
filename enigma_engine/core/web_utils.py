"""
Shared web search and page fetching utilities.
===============================================

Extracts the duplicated DuckDuckGo HTML parser and text
extractor into reusable functions.

Used by:
- builtin_commands.py  (search.web, web.fetch commands)
- gui_forge.py         (_web_learn web scraping)

All functions require the ``requests`` library.  Callers
should catch ``ImportError`` and surface install instructions.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from html.parser import HTMLParser
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Maximum response body size (1 MB) to prevent memory exhaustion
_MAX_RESPONSE_BYTES = 1_048_576

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36"
    ),
}


# ================================================================
# URL validation (SSRF protection)
# ================================================================

def _validate_url(url: str) -> None:
    """Reject URLs that target private/reserved IP addresses.

    Raises ``ValueError`` if the URL scheme is not http/https or
    the hostname resolves to a private, reserved, loopback, or
    link-local IP address.
    """
    parsed = urlparse(url)

    # Only allow http and https schemes
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported URL scheme: {parsed.scheme!r} (only http/https allowed)"
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"No hostname in URL: {url!r}")

    # Resolve hostname and check every address
    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve hostname {hostname!r}: {exc}") from exc

    for family, _type, _proto, _canon, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_link_local:
            raise ValueError(
                f"URL {url!r} resolves to private/reserved address {ip}"
            )


# ================================================================
# Public API
# ================================================================

def ddg_search(query: str, max_results: int = 10) -> list[dict]:
    """Search DuckDuckGo HTML and return results.

    Each result is ``{"title": str, "url": str, "snippet": str}``.
    Returns an empty list on error.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
    """
    import requests
    from urllib.parse import quote_plus

    url = (
        "https://html.duckduckgo.com/html/"
        f"?q={quote_plus(query)}")
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()

    parser = _DDGParser()
    parser.feed(resp.text)
    return parser.results[:max_results]


def ddg_image_search(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo for images and return results.

    Each result is ``{"title": str, "url": str, "thumbnail": str}``.
    Returns an empty list on error.

    Uses DuckDuckGo's image search via the ``vqd`` token mechanism.

    Args:
        query: Image search query string.
        max_results: Maximum number of results to return.
    """
    import requests
    from urllib.parse import quote_plus

    # Step 1: Get vqd token from DDG search page
    token_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        token_resp = requests.get(token_url, headers=_HEADERS, timeout=10)
        token_resp.raise_for_status()
    except Exception as exc:
        logger.warning("DDG image search token request failed: %s", exc)
        return []

    # Extract vqd token
    import re
    vqd_match = re.search(r'vqd="([^"]+)"', token_resp.text)
    if not vqd_match:
        vqd_match = re.search(r"vqd=([^&'\"]+)", token_resp.text)
    if not vqd_match:
        logger.warning("DDG image search: could not extract vqd token")
        return []

    vqd = vqd_match.group(1)

    # Step 2: Fetch image results JSON
    img_url = (
        f"https://duckduckgo.com/i.js?l=us-en&o=json"
        f"&q={quote_plus(query)}&vqd={vqd}"
        f"&f=,,,,,&p=1"
    )
    img_headers = {**_HEADERS, "Referer": "https://duckduckgo.com/"}
    try:
        img_resp = requests.get(img_url, headers=img_headers, timeout=10)
        img_resp.raise_for_status()
        data = img_resp.json()
    except Exception as exc:
        logger.warning("DDG image search request failed: %s", exc)
        return []

    results: list[dict] = []
    for item in data.get("results", []):
        image_url = item.get("image", "")
        thumbnail = item.get("thumbnail", "")
        title = item.get("title", "")
        if image_url:
            results.append({
                "title": title,
                "url": image_url,
                "thumbnail": thumbnail,
            })
        if len(results) >= max_results:
            break

    return results


def fetch_page_text(url: str, max_chars: int = 3000) -> str:
    """Fetch a URL and extract readable text content.

    Returns extracted text trimmed to *max_chars*, or an empty
    string on error.

    Raises ``ValueError`` for private/reserved IPs or non-http(s) schemes.
    """
    import requests

    _validate_url(url)

    # Stream the response to enforce a size cap
    with requests.get(
        url, headers=_HEADERS, timeout=15, stream=True
    ) as resp:
        resp.raise_for_status()

        chunks: list[bytes] = []
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=8192):
            chunks.append(chunk)
            downloaded += len(chunk)
            if downloaded >= _MAX_RESPONSE_BYTES:
                logger.warning(
                    "Response from %s exceeded %d bytes, truncating",
                    url, _MAX_RESPONSE_BYTES,
                )
                break

        html = b"".join(chunks).decode(
            resp.encoding or "utf-8", errors="replace"
        )

    text = extract_html_text(html)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def extract_html_text(html: str) -> str:
    """Extract readable text from HTML.

    Strips ``<script>``, ``<style>``, ``<nav>``, ``<footer>``,
    ``<header>``, and ``<aside>`` content.  Ignores fragments
    shorter than 3 characters.
    """
    extractor = _TextExtractor()
    extractor.feed(html)
    return " ".join(extractor.text)


# ================================================================
# Internal HTML parsers
# ================================================================

class _DDGParser(HTMLParser):
    """Parse DuckDuckGo HTML search results page."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._title = ""
        self._url = ""
        self._snippet = ""
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attrs_d = dict(attrs)
        if tag == "a" and "result__a" in attrs_d.get("class", ""):
            self._in_title = True
            self._url = attrs_d.get("href", "")
        elif (tag == "a"
                and "result__snippet" in attrs_d.get("class", "")):
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
        if tag == "a" and self._in_snippet:
            self._in_snippet = False
            if self._title.strip() and self._snippet.strip():
                self.results.append({
                    "title": self._title.strip(),
                    "url": self._url,
                    "snippet": self._snippet.strip(),
                })
            self._title = ""
            self._url = ""
            self._snippet = ""

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title += data
        if self._in_snippet:
            self._snippet += data


class _TextExtractor(HTMLParser):
    """Extract readable text from HTML, skipping boilerplate."""

    _SKIP_TAGS = frozenset({
        "script", "style", "nav", "footer", "header", "aside",
    })

    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            t = data.strip()
            if t:
                self.text.append(t)
