"""
Game Wiki Knowledge for Enigma AI Engine

Auto-load wikis per game.

Features:
- Wiki article fetching
- Game-specific knowledge bases
- Search functionality
- Caching
- Offline mode

Usage:
    from enigma_engine.tools.game_wiki import GameWiki
    
    wiki = GameWiki()
    
    # Load wiki for a game
    wiki.load_game("minecraft")
    
    # Search for info
    results = wiki.search("how to make diamond pickaxe")
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass
class WikiArticle:
    """A wiki article."""
    title: str
    content: str
    url: str
    game: str
    timestamp: float = field(default_factory=time.time)
    sections: Dict[str, str] = field(default_factory=dict)
    related: List[str] = field(default_factory=list)


@dataclass
class WikiSource:
    """Wiki source configuration."""
    name: str
    game: str
    base_url: str
    search_url: str
    api_url: Optional[str] = None


class WikiCache:
    """Cache for wiki articles."""
    
    def __init__(self, cache_dir: Optional[Path] = None, ttl_hours: int = 24):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory for cache files
            ttl_hours: Cache TTL in hours
        """
        self.cache_dir = cache_dir or Path("data/wiki_cache")
        self.ttl_seconds = ttl_hours * 3600
        self._memory_cache: Dict[str, WikiArticle] = {}
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, game: str, title: str) -> Optional[WikiArticle]:
        """Get article from cache."""
        key = f"{game}:{title}"
        
        # Check memory cache
        if key in self._memory_cache:
            article = self._memory_cache[key]
            if time.time() - article.timestamp < self.ttl_seconds:
                return article
        
        # Check disk cache
        cache_file = self.cache_dir / f"{game}_{self._sanitize(title)}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    if time.time() - data["timestamp"] < self.ttl_seconds:
                        article = WikiArticle(**data)
                        self._memory_cache[key] = article
                        return article
            except Exception as e:
                logger.error(f"Cache read error: {e}")
        
        return None
    
    def set(self, article: WikiArticle):
        """Store article in cache."""
        key = f"{article.game}:{article.title}"
        self._memory_cache[key] = article
        
        # Persist to disk
        cache_file = self.cache_dir / f"{article.game}_{self._sanitize(article.title)}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    "title": article.title,
                    "content": article.content,
                    "url": article.url,
                    "game": article.game,
                    "timestamp": article.timestamp,
                    "sections": article.sections,
                    "related": article.related
                }, f)
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    def _sanitize(self, text: str) -> str:
        """Sanitize text for filename."""
        return re.sub(r'[^\w\-]', '_', text)[:50]
    
    def clear(self, game: Optional[str] = None):
        """Clear cache."""
        if game:
            # Clear specific game
            for key in list(self._memory_cache.keys()):
                if key.startswith(f"{game}:"):
                    del self._memory_cache[key]
            
            for f in self.cache_dir.glob(f"{game}_*.json"):
                f.unlink()
        else:
            # Clear all
            self._memory_cache.clear()
            for f in self.cache_dir.glob("*.json"):
                f.unlink()


class GameWiki:
    """Game wiki knowledge base."""
    
    # Built-in wiki sources
    WIKI_SOURCES = {
        "minecraft": WikiSource(
            name="Minecraft Wiki",
            game="minecraft",
            base_url="https://minecraft.wiki",
            search_url="https://minecraft.wiki/w/Special:Search?search=",
            api_url="https://minecraft.wiki/api.php"
        ),
        "terraria": WikiSource(
            name="Terraria Wiki",
            game="terraria",
            base_url="https://terraria.wiki.gg",
            search_url="https://terraria.wiki.gg/wiki/Special:Search?search=",
            api_url="https://terraria.wiki.gg/api.php"
        ),
        "stardew": WikiSource(
            name="Stardew Valley Wiki",
            game="stardew",
            base_url="https://stardewvalleywiki.com",
            search_url="https://stardewvalleywiki.com/mediawiki/index.php?search=",
            api_url="https://stardewvalleywiki.com/mediawiki/api.php"
        ),
        "factorio": WikiSource(
            name="Factorio Wiki",
            game="factorio",
            base_url="https://wiki.factorio.com",
            search_url="https://wiki.factorio.com/index.php?search=",
            api_url="https://wiki.factorio.com/api.php"
        ),
        "generic": WikiSource(
            name="Fandom",
            game="generic",
            base_url="https://www.fandom.com",
            search_url="https://www.fandom.com/search?query="
        )
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize game wiki.
        
        Args:
            cache_dir: Cache directory
        """
        self._cache = WikiCache(cache_dir)
        self._current_game: Optional[str] = None
        self._current_source: Optional[WikiSource] = None
        self._custom_sources: Dict[str, WikiSource] = {}
    
    def load_game(self, game: str):
        """
        Load wiki for a specific game.
        
        Args:
            game: Game identifier
        """
        game_lower = game.lower()
        
        # Check custom sources first
        if game_lower in self._custom_sources:
            self._current_source = self._custom_sources[game_lower]
        elif game_lower in self.WIKI_SOURCES:
            self._current_source = self.WIKI_SOURCES[game_lower]
        else:
            logger.warning(f"No wiki source for {game}, using generic")
            self._current_source = self.WIKI_SOURCES["generic"]
        
        self._current_game = game_lower
        logger.info(f"Loaded wiki for {game}: {self._current_source.name}")
    
    def add_source(self, source: WikiSource):
        """Add a custom wiki source."""
        self._custom_sources[source.game.lower()] = source
    
    def search(self, query: str, limit: int = 5) -> List[WikiArticle]:
        """
        Search the wiki.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching articles
        """
        if not self._current_source:
            self.load_game("generic")
        
        # Try API search first
        if self._current_source.api_url:
            return self._api_search(query, limit)
        
        # Fall back to scraping
        return self._scrape_search(query, limit)
    
    def _api_search(self, query: str, limit: int) -> List[WikiArticle]:
        """Search using MediaWiki API."""
        import urllib.request
        import urllib.error
        
        try:
            params = f"?action=opensearch&search={quote(query)}&limit={limit}&format=json"
            url = self._current_source.api_url + params
            
            req = urllib.request.Request(url, headers={"User-Agent": "EnigmaEngine/1.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            # OpenSearch returns [query, [titles], [descriptions], [urls]]
            results = []
            if len(data) >= 4:
                for title, desc, url in zip(data[1], data[2], data[3]):
                    article = WikiArticle(
                        title=title,
                        content=desc,
                        url=url,
                        game=self._current_game
                    )
                    results.append(article)
            
            return results
            
        except Exception as e:
            logger.error(f"Wiki API search failed: {e}")
            return []
    
    def _scrape_search(self, query: str, limit: int) -> List[WikiArticle]:
        """Search by scraping (fallback)."""
        # This is a placeholder - actual implementation would need HTML parsing
        logger.warning("Scrape search not fully implemented")
        return []
    
    def get_article(self, title: str) -> Optional[WikiArticle]:
        """
        Get a specific article.
        
        Args:
            title: Article title
            
        Returns:
            Article or None
        """
        if not self._current_source:
            self.load_game("generic")
        
        # Check cache
        cached = self._cache.get(self._current_game, title)
        if cached:
            return cached
        
        # Fetch article
        article = self._fetch_article(title)
        
        if article:
            self._cache.set(article)
        
        return article
    
    def _fetch_article(self, title: str) -> Optional[WikiArticle]:
        """Fetch article from wiki."""
        import urllib.request
        import urllib.error
        
        if not self._current_source.api_url:
            return None
        
        try:
            params = f"?action=query&titles={quote(title)}&prop=extracts&explaintext=1&format=json"
            url = self._current_source.api_url + params
            
            req = urllib.request.Request(url, headers={"User-Agent": "EnigmaEngine/1.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            pages = data.get("query", {}).get("pages", {})
            
            for page_id, page in pages.items():
                if page_id == "-1":
                    return None
                
                content = page.get("extract", "")
                article_url = f"{self._current_source.base_url}/wiki/{quote(title)}"
                
                return WikiArticle(
                    title=page.get("title", title),
                    content=content,
                    url=article_url,
                    game=self._current_game,
                    sections=self._parse_sections(content)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch article: {e}")
            return None
    
    def _parse_sections(self, content: str) -> Dict[str, str]:
        """Parse article into sections."""
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        for line in content.split("\n"):
            # Check for section headers (== Header ==)
            if line.startswith("==") and line.endswith("=="):
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content)
                
                # Start new section
                current_section = line.strip("= ")
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content)
        
        return sections
    
    def get_crafting_recipe(self, item: str) -> Optional[Dict]:
        """
        Get crafting recipe for an item.
        
        Args:
            item: Item name
            
        Returns:
            Recipe dict or None
        """
        article = self.get_article(item)
        if not article:
            return None
        
        # Look for crafting section
        crafting_text = article.sections.get("Crafting", "")
        
        if not crafting_text:
            return None
        
        # Basic recipe extraction (game-specific parsing would be better)
        return {
            "item": item,
            "description": crafting_text[:500]
        }
    
    def ask(self, question: str) -> str:
        """
        Ask a question about the game.
        
        Args:
            question: Natural language question
            
        Returns:
            Answer from wiki
        """
        # Extract key terms
        terms = self._extract_terms(question)
        
        # Search wiki
        results = self.search(" ".join(terms), limit=3)
        
        if not results:
            return f"I couldn't find information about that in the {self._current_source.name}."
        
        # Get full article for best result
        article = self.get_article(results[0].title)
        
        if article:
            # Return a summary
            return f"From {article.title}:\n\n{article.content[:500]}..."
        
        return results[0].content or "No detailed information available."
    
    def _extract_terms(self, question: str) -> List[str]:
        """Extract search terms from question."""
        # Remove common question words
        stop_words = {
            'how', 'to', 'what', 'is', 'are', 'where', 'can', 'i', 
            'do', 'make', 'find', 'get', 'the', 'a', 'an', 'in'
        }
        
        words = re.findall(r'\w+', question.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]


# Convenience functions
def search_game_wiki(game: str, query: str) -> List[WikiArticle]:
    """Quick search a game wiki."""
    wiki = GameWiki()
    wiki.load_game(game)
    return wiki.search(query)


def ask_game_wiki(game: str, question: str) -> str:
    """Quick ask a question about a game."""
    wiki = GameWiki()
    wiki.load_game(game)
    return wiki.ask(question)
