"""
Web Tools Example for ForgeAI

This example shows how to use ForgeAI's web capabilities.
The AI can search the internet, fetch web pages, and extract content.

CAPABILITIES:
- Web search (DuckDuckGo, no API key needed)
- Webpage fetching and parsing
- Content extraction
- URL safety checking

USAGE:
    python examples/web_tools_example.py
    
Or import in your own code:
    from examples.web_tools_example import search_web, fetch_page
"""

import json
import re
import urllib.parse
import urllib.request
from typing import Dict, Any, List, Optional
from html.parser import HTMLParser


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; ForgeAI/1.0)"
DEFAULT_TIMEOUT = 10
MAX_CONTENT_SIZE = 100000  # 100KB


# =============================================================================
# WEB SEARCH
# =============================================================================

class WebSearch:
    """
    Search the web using DuckDuckGo.
    No API key required!
    """
    
    def __init__(self):
        self.user_agent = DEFAULT_USER_AGENT
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search the web.
        
        Args:
            query: Search query string
            num_results: Number of results to return
        
        Returns:
            List of results: [{"title": ..., "url": ..., "snippet": ...}, ...]
        """
        if not query or not query.strip():
            return []
        
        try:
            # Use DuckDuckGo Lite (simpler HTML, easier to parse)
            encoded_query = urllib.parse.quote(query.strip())
            url = f"https://lite.duckduckgo.com/lite/?q={encoded_query}"
            
            headers = {"User-Agent": self.user_agent}
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as response:
                html = response.read().decode('utf-8')
            
            # Parse results
            results = self._parse_ddg_lite(html)
            return results[:num_results]
            
        except Exception as e:
            print(f"[WEB] Search failed: {e}")
            return []
    
    def _parse_ddg_lite(self, html: str) -> List[Dict[str, str]]:
        """Parse DuckDuckGo Lite HTML."""
        results = []
        
        # Find result links
        link_pattern = r'<a[^>]*rel="nofollow"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
        matches = re.findall(link_pattern, html, re.IGNORECASE)
        
        for url, title in matches:
            # Skip internal DDG links
            if 'duckduckgo.com' in url:
                continue
            
            # Clean up
            url = url.strip()
            title = title.strip()
            
            if url and title and url.startswith('http'):
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": ""  # Lite doesn't have snippets
                })
        
        return results
    
    def search_and_summarize(self, query: str, num_results: int = 3) -> str:
        """
        Search and return a text summary of results.
        
        Returns:
            Formatted string with search results
        """
        results = self.search(query, num_results)
        
        if not results:
            return f"No results found for: {query}"
        
        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   {r['url']}")
            if r.get('snippet'):
                lines.append(f"   {r['snippet']}")
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# WEBPAGE FETCHING
# =============================================================================

class TextExtractor(HTMLParser):
    """Extract readable text from HTML."""
    
    def __init__(self):
        super().__init__()
        self.text = []
        self._skip_tags = {'script', 'style', 'head', 'title', 'meta', 'noscript'}
        self._current_tag = None
    
    def handle_starttag(self, tag, attrs):
        self._current_tag = tag.lower()
    
    def handle_endtag(self, tag):
        self._current_tag = None
    
    def handle_data(self, data):
        if self._current_tag not in self._skip_tags:
            text = data.strip()
            if text:
                self.text.append(text)
    
    def get_text(self) -> str:
        return ' '.join(self.text)


class WebFetcher:
    """
    Fetch and parse web pages.
    """
    
    def __init__(self):
        self.user_agent = DEFAULT_USER_AGENT
    
    def fetch(self, url: str, extract_text: bool = True) -> Dict[str, Any]:
        """
        Fetch a webpage.
        
        Args:
            url: URL to fetch
            extract_text: If True, extract readable text from HTML
        
        Returns:
            Dict with: success, url, title, content, content_type, etc.
        """
        if not url:
            return {"success": False, "error": "URL cannot be empty"}
        
        # Ensure URL has scheme
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        try:
            headers = {"User-Agent": self.user_agent}
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as response:
                content_type = response.headers.get('Content-Type', '')
                
                # Read content (with size limit)
                raw_content = response.read(MAX_CONTENT_SIZE)
                
                # Try to decode as text
                try:
                    content = raw_content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        content = raw_content.decode('latin-1')
                    except:
                        content = raw_content.decode('utf-8', errors='ignore')
            
            result = {
                "success": True,
                "url": url,
                "content_type": content_type,
                "raw_length": len(raw_content),
            }
            
            # Extract text from HTML
            if extract_text and 'html' in content_type.lower():
                extractor = TextExtractor()
                extractor.feed(content)
                text = extractor.get_text()
                
                # Also try to get title
                title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
                if title_match:
                    result["title"] = title_match.group(1).strip()
                
                result["content"] = text
                result["text_length"] = len(text)
            else:
                result["content"] = content
            
            return result
            
        except urllib.error.HTTPError as e:
            return {"success": False, "error": f"HTTP {e.code}: {e.reason}", "url": url}
        except urllib.error.URLError as e:
            return {"success": False, "error": f"URL error: {e.reason}", "url": url}
        except Exception as e:
            return {"success": False, "error": str(e), "url": url}
    
    def fetch_json(self, url: str) -> Dict[str, Any]:
        """
        Fetch JSON from a URL.
        
        Args:
            url: URL to fetch
        
        Returns:
            Dict with: success, data, error
        """
        result = self.fetch(url, extract_text=False)
        
        if not result.get("success"):
            return result
        
        try:
            data = json.loads(result["content"])
            return {"success": True, "url": url, "data": data}
        except json.JSONDecodeError as e:
            return {"success": False, "url": url, "error": f"Invalid JSON: {e}"}


# =============================================================================
# URL SAFETY
# =============================================================================

class URLSafety:
    """
    Check URLs for safety before fetching.
    """
    
    # Known suspicious TLDs and patterns
    SUSPICIOUS_TLDS = {'.xyz', '.top', '.click', '.link', '.work', '.gq', '.ml', '.cf'}
    BLOCKED_DOMAINS = {'malware.com', 'phishing.com'}  # Add your blocklist
    
    def __init__(self):
        pass
    
    def is_safe(self, url: str) -> tuple:
        """
        Check if URL appears safe.
        
        Returns:
            (is_safe: bool, reason: str)
        """
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check scheme
            if parsed.scheme not in ('http', 'https'):
                return False, f"Unsupported scheme: {parsed.scheme}"
            
            # Check for IP addresses (potentially suspicious)
            if re.match(r'^\d+\.\d+\.\d+\.\d+', domain):
                return False, "Direct IP addresses may be suspicious"
            
            # Check blocklist
            for blocked in self.BLOCKED_DOMAINS:
                if blocked in domain:
                    return False, f"Domain is blocked: {blocked}"
            
            # Check suspicious TLDs
            for tld in self.SUSPICIOUS_TLDS:
                if domain.endswith(tld):
                    return False, f"Suspicious TLD: {tld}"
            
            return True, "URL appears safe"
            
        except Exception as e:
            return False, f"Could not parse URL: {e}"
    
    def safe_fetch(self, url: str, fetcher: WebFetcher = None) -> Dict[str, Any]:
        """
        Check URL safety then fetch.
        
        Args:
            url: URL to fetch
            fetcher: WebFetcher instance (creates one if not provided)
        
        Returns:
            Fetch result or error
        """
        is_safe, reason = self.is_safe(url)
        
        if not is_safe:
            return {"success": False, "error": f"URL blocked: {reason}", "url": url}
        
        fetcher = fetcher or WebFetcher()
        return fetcher.fetch(url)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Quick web search.
    
    Args:
        query: Search query
        num_results: Number of results
    
    Returns:
        List of search results
    """
    searcher = WebSearch()
    return searcher.search(query, num_results)


def fetch_page(url: str) -> Dict[str, Any]:
    """
    Quick page fetch.
    
    Args:
        url: URL to fetch
    
    Returns:
        Page content and metadata
    """
    fetcher = WebFetcher()
    return fetcher.fetch(url)


def fetch_safe(url: str) -> Dict[str, Any]:
    """
    Fetch with safety check.
    
    Args:
        url: URL to fetch
    
    Returns:
        Page content or error
    """
    safety = URLSafety()
    return safety.safe_fetch(url)


def search_and_fetch(query: str, num_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search and fetch content from top results.
    
    Args:
        query: Search query
        num_results: Number of pages to fetch
    
    Returns:
        List of page contents
    """
    searcher = WebSearch()
    fetcher = WebFetcher()
    
    results = searcher.search(query, num_results)
    
    pages = []
    for result in results:
        url = result['url']
        page = fetcher.fetch(url)
        page['search_title'] = result['title']
        pages.append(page)
    
    return pages


# =============================================================================
# MAIN - Run this file directly to test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ForgeAI Web Tools Example")
    print("=" * 60)
    
    # Test web search
    print("\n[1] Testing web search...")
    searcher = WebSearch()
    
    query = "Python programming tutorial"
    results = searcher.search(query, num_results=3)
    
    print(f"Search: '{query}'")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['title']}")
        print(f"     {r['url']}")
    
    # Test page fetching
    print("\n[2] Testing page fetch...")
    fetcher = WebFetcher()
    
    test_url = "https://example.com"
    result = fetcher.fetch(test_url)
    
    if result['success']:
        print(f"Fetched: {test_url}")
        print(f"  Title: {result.get('title', 'N/A')}")
        print(f"  Content length: {result.get('text_length', len(result.get('content', '')))}")
        print(f"  Preview: {result['content'][:200]}...")
    else:
        print(f"Failed: {result['error']}")
    
    # Test URL safety
    print("\n[3] Testing URL safety...")
    safety = URLSafety()
    
    test_urls = [
        "https://google.com",
        "https://suspicious.xyz/malware",
        "http://192.168.1.1/admin",
    ]
    
    for url in test_urls:
        is_safe, reason = safety.is_safe(url)
        status = "SAFE" if is_safe else "BLOCKED"
        print(f"  {status}: {url}")
        if not is_safe:
            print(f"    Reason: {reason}")
    
    # Test JSON fetching
    print("\n[4] Testing JSON fetch...")
    
    json_url = "https://api.github.com/zen"
    result = fetcher.fetch(json_url, extract_text=False)
    if result['success']:
        print(f"GitHub Zen: {result['content']}")
    
    # Test search and summarize
    print("\n[5] Testing search summary...")
    summary = searcher.search_and_summarize("ForgeAI framework", num_results=2)
    print(summary)
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nNo additional dependencies required!")
    print("All web tools use Python's built-in urllib.")
