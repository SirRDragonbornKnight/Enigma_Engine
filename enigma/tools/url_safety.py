"""
URL Safety - Block malicious websites and filter content.
"""

import re
from pathlib import Path
from typing import Set, List


class URLSafety:
    """Filter and validate URLs for safe browsing."""
    
    # Built-in blocklist
    BLOCKED_DOMAINS = {
        # Malware/phishing (examples)
        "malware-site.com",
        "phishing-example.com",
        # Add more as needed
    }
    
    BLOCKED_PATTERNS = [
        r".*\.exe$",           # Executables
        r".*\.msi$",           # Installers
        r".*\.bat$",           # Batch files
        r".*\.scr$",           # Screensavers (often malware)
        r".*download.*crack.*", # Piracy/malware
        r".*free.*download.*",  # Suspicious downloads
    ]
    
    ALLOWED_DOMAINS = {
        # Trusted sources
        "wikipedia.org",
        "github.com",
        "stackoverflow.com",
        "python.org",
        "pytorch.org",
        "huggingface.co",
    }
    
    def __init__(self, custom_blocklist_path: Path = None):
        self.blocked_domains = self.BLOCKED_DOMAINS.copy()
        self.blocked_patterns = [re.compile(p) for p in self.BLOCKED_PATTERNS]
        
        # Load custom blocklist if provided
        if custom_blocklist_path and custom_blocklist_path.exists():
            self._load_custom_blocklist(custom_blocklist_path)
    
    def _load_custom_blocklist(self, path: Path):
        """Load additional blocked domains from file."""
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                self.blocked_domains.add(line.lower())
    
    def is_safe(self, url: str) -> bool:
        """Check if URL is safe to visit."""
        url_lower = url.lower()
        
        # Check blocked domains
        for domain in self.blocked_domains:
            if domain in url_lower:
                return False
        
        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.match(url_lower):
                return False
        
        return True
    
    def is_trusted(self, url: str) -> bool:
        """Check if URL is from a trusted source."""
        url_lower = url.lower()
        for domain in self.ALLOWED_DOMAINS:
            if domain in url_lower:
                return True
        return False
    
    def filter_urls(self, urls: List[str]) -> List[str]:
        """Filter list of URLs, keeping only safe ones."""
        return [url for url in urls if self.is_safe(url)]
    
    def get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        try:
            return urlparse(url).netloc
        except:
            return ""


# Content filtering for extracted text
class ContentFilter:
    """Filter out ads, popups, and filler content."""
    
    AD_PATTERNS = [
        r"advertisement",
        r"sponsored content",
        r"click here to",
        r"subscribe now",
        r"sign up for",
        r"limited time offer",
        r"act now",
        r"buy now",
        r"free trial",
        r"download now",
        r"cookie policy",
        r"we use cookies",
        r"accept all cookies",
    ]
    
    def __init__(self):
        self.ad_patterns = [re.compile(p, re.IGNORECASE) for p in self.AD_PATTERNS]
    
    def is_ad_content(self, text: str) -> bool:
        """Check if text appears to be advertising."""
        for pattern in self.ad_patterns:
            if pattern.search(text):
                return True
        return False
    
    def filter_content(self, text: str) -> str:
        """Remove likely ad content from text."""
        lines = text.split('\n')
        filtered = [line for line in lines if not self.is_ad_content(line)]
        return '\n'.join(filtered)
    
    def extract_main_content(self, html: str) -> str:
        """Extract main content, skipping navigation/ads/footer."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # If BeautifulSoup not available, return as-is
            return html
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove known non-content elements
        for tag in soup.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style', 'noscript']):
            tag.decompose()
        
        # Remove elements with ad-related classes/ids
        ad_indicators = ['ad', 'ads', 'advertisement', 'sponsor', 'promo', 'banner', 'popup', 'modal', 'cookie', 'newsletter']
        for indicator in ad_indicators:
            for tag in soup.find_all(class_=re.compile(indicator, re.I)):
                tag.decompose()
            for tag in soup.find_all(id=re.compile(indicator, re.I)):
                tag.decompose()
        
        # Get main content
        main = soup.find('main') or soup.find('article') or soup.find('body')
        if main:
            return main.get_text(separator='\n', strip=True)
        return soup.get_text(separator='\n', strip=True)
