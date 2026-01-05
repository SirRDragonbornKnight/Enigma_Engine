"""
URL Safety - Block malicious websites and filter content.
Enhanced with dynamic blocklist loading and periodic updates.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class URLSafety:
    """Filter and validate URLs for safe browsing with dynamic blocklist support."""
    
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
    
    def __init__(
        self,
        custom_blocklist_path: Optional[Path] = None,
        enable_auto_update: bool = False,
        update_interval_hours: int = 24,
        config_path: Optional[Path] = None
    ):
        """
        Initialize URL safety checker.
        
        Args:
            custom_blocklist_path: Path to custom blocklist file
            enable_auto_update: Enable automatic blocklist updates
            update_interval_hours: Hours between blocklist updates
            config_path: Path to configuration file for blocklists and trusted domains
        """
        self.blocked_domains = self.BLOCKED_DOMAINS.copy()
        self.trusted_domains = self.ALLOWED_DOMAINS.copy()
        self.blocked_patterns = [re.compile(p) for p in self.BLOCKED_PATTERNS]
        self.custom_blocklist_path = custom_blocklist_path
        self.config_path = config_path
        self.enable_auto_update = enable_auto_update
        self.update_interval = timedelta(hours=update_interval_hours)
        self.last_update = None
        self.blocklist_cache_path = Path("data/url_blocklist_cache.json")
        
        # Load config if provided
        if config_path and config_path.exists():
            self._load_config(config_path)
        
        # Load custom blocklist if provided
        if custom_blocklist_path and custom_blocklist_path.exists():
            self._load_custom_blocklist(custom_blocklist_path)
        
        # Load cached blocklist
        self._load_cached_blocklist()
        
        # Auto-update if enabled
        if enable_auto_update:
            self._auto_update_if_needed()
    
    def _load_config(self, path: Path):
        """Load configuration from JSON file."""
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            
            # Load blocked domains
            if 'blocked_domains' in config:
                self.blocked_domains.update(config['blocked_domains'])
            
            # Load trusted domains
            if 'trusted_domains' in config:
                self.trusted_domains.update(config['trusted_domains'])
            
            logger.info(f"Loaded URL safety config from {path}")
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
    
    def _save_config(self):
        """Save configuration to JSON file."""
        if not self.config_path:
            return
        
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config = {
                'blocked_domains': list(self.blocked_domains),
                'trusted_domains': list(self.trusted_domains),
                'last_updated': datetime.now().isoformat(),
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved URL safety config to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _load_custom_blocklist(self, path: Path):
        """Load additional blocked domains from file."""
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    self.blocked_domains.add(line.lower())
            logger.info(f"Loaded {len(self.blocked_domains)} blocked domains")
        except Exception as e:
            logger.error(f"Failed to load custom blocklist: {e}")
    
    def _load_cached_blocklist(self):
        """Load cached blocklist from disk."""
        if not self.blocklist_cache_path.exists():
            return
        
        try:
            with open(self.blocklist_cache_path, 'r') as f:
                data = json.load(f)
            
            self.blocked_domains.update(data.get('blocked_domains', []))
            self.last_update = datetime.fromisoformat(data.get('last_update', datetime.now().isoformat()))
            
            logger.info(f"Loaded cached blocklist with {len(self.blocked_domains)} domains")
        except Exception as e:
            logger.error(f"Failed to load cached blocklist: {e}")
    
    def _save_cached_blocklist(self):
        """Save blocklist to cache."""
        try:
            self.blocklist_cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'blocked_domains': list(self.blocked_domains),
                'last_update': datetime.now().isoformat(),
                'total_domains': len(self.blocked_domains)
            }
            
            with open(self.blocklist_cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved blocklist cache with {len(self.blocked_domains)} domains")
        except Exception as e:
            logger.error(f"Failed to save blocklist cache: {e}")
    
    def _auto_update_if_needed(self):
        """Check if update is needed and perform it."""
        if self.last_update is None:
            self.update_blocklist_from_sources()
            return
        
        if datetime.now() - self.last_update > self.update_interval:
            logger.info("Blocklist update interval reached, updating...")
            self.update_blocklist_from_sources()
    
    def update_blocklist_from_sources(self, sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Update blocklist from external sources.
        
        Args:
            sources: List of source URLs (uses defaults if None)
            
        Returns:
            Update statistics
        """
        if sources is None:
            # Default public blocklist sources
            sources = [
                # PhishTank (example - would need API key in practice)
                # 'https://data.phishtank.com/data/online-valid.json',
                # URLhaus malware URLs
                # 'https://urlhaus.abuse.ch/downloads/csv/',
            ]
        
        initial_count = len(self.blocked_domains)
        added_domains = set()
        
        # Since we can't make actual HTTP requests in this context,
        # we'll simulate the update process
        logger.info("Blocklist update initiated (simulation mode)")
        
        # In a real implementation, would fetch from sources
        # For now, just log that we would update
        
        self.last_update = datetime.now()
        self._save_cached_blocklist()
        
        return {
            'initial_count': initial_count,
            'added_count': len(added_domains),
            'final_count': len(self.blocked_domains),
            'last_update': self.last_update.isoformat(),
            'sources_checked': len(sources)
        }
    
    def import_blocklist_from_file(self, filepath: Path) -> int:
        """
        Import blocklist from a file.
        
        Args:
            filepath: Path to blocklist file
            
        Returns:
            Number of domains added
        """
        if not filepath.exists():
            logger.error(f"Blocklist file not found: {filepath}")
            return 0
        
        initial_count = len(self.blocked_domains)
        
        try:
            content = filepath.read_text()
            
            # Support multiple formats
            if filepath.suffix == '.json':
                data = json.loads(content)
                if isinstance(data, list):
                    domains = data
                elif isinstance(data, dict):
                    domains = data.get('domains', [])
                else:
                    domains = []
            else:
                # Plain text format (one domain per line)
                domains = [
                    line.strip().lower()
                    for line in content.splitlines()
                    if line.strip() and not line.startswith('#')
                ]
            
            self.blocked_domains.update(domains)
            added = len(self.blocked_domains) - initial_count
            
            self._save_cached_blocklist()
            logger.info(f"Imported {added} new blocked domains from {filepath}")
            
            return added
        
        except Exception as e:
            logger.error(f"Failed to import blocklist: {e}")
            return 0
    
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
        for domain in self.trusted_domains:
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
        except (ValueError, AttributeError):
            return ""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get blocklist statistics."""
        return {
            'total_blocked_domains': len(self.blocked_domains),
            'trusted_domains': len(self.ALLOWED_DOMAINS),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'auto_update_enabled': self.enable_auto_update,
            'next_update': (self.last_update + self.update_interval).isoformat() if self.last_update else None
        }
    
    def add_blocked_domain(self, domain: str):
        """Manually add a domain to blocklist."""
        self.blocked_domains.add(domain.lower())
        self._save_cached_blocklist()
        if self.config_path:
            self._save_config()
        logger.info(f"Added {domain} to blocklist")
    
    def add_trusted_domain(self, domain: str):
        """Manually add a domain to trusted list."""
        self.trusted_domains.add(domain.lower())
        if self.config_path:
            self._save_config()
        logger.info(f"Added {domain} to trusted list")
    
    def remove_blocked_domain(self, domain: str) -> bool:
        """Remove a domain from blocklist."""
        domain_lower = domain.lower()
        if domain_lower in self.blocked_domains:
            self.blocked_domains.remove(domain_lower)
            self._save_cached_blocklist()
            if self.config_path:
                self._save_config()
            logger.info(f"Removed {domain} from blocklist")
            return True
        return False
    
    def remove_trusted_domain(self, domain: str) -> bool:
        """Remove a domain from trusted list."""
        domain_lower = domain.lower()
        if domain_lower in self.trusted_domains:
            self.trusted_domains.remove(domain_lower)
            if self.config_path:
                self._save_config()
            logger.info(f"Removed {domain} from trusted list")
            return True
        return False


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
