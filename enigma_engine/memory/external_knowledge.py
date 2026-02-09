"""
External Knowledge Bases for Enigma AI Engine

Connect to Wikipedia, Wikidata, and other knowledge sources.

Features:
- Wikipedia search/retrieval
- Wikidata queries
- DBpedia integration
- Caching
- Rate limiting

Usage:
    from enigma_engine.memory.external_knowledge import ExternalKnowledge
    
    kb = ExternalKnowledge()
    
    # Search Wikipedia
    results = kb.search_wikipedia("artificial intelligence")
    
    # Get Wikidata entity
    entity = kb.get_wikidata_entity("Q11660")  # AI
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeSource:
    """A knowledge source result."""
    source: str  # wikipedia, wikidata, etc.
    title: str
    content: str
    url: str = ""
    entity_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieved_at: float = field(default_factory=time.time)


@dataclass
class WikipediaArticle:
    """A Wikipedia article."""
    title: str
    summary: str
    content: str
    url: str
    categories: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    page_id: int = 0


@dataclass
class WikidataEntity:
    """A Wikidata entity."""
    entity_id: str
    label: str
    description: str
    aliases: List[str] = field(default_factory=list)
    claims: Dict[str, Any] = field(default_factory=dict)
    sitelinks: Dict[str, str] = field(default_factory=dict)


class WikipediaClient:
    """Wikipedia API client."""
    
    def __init__(self, language: str = "en"):
        """
        Initialize Wikipedia client.
        
        Args:
            language: Language code
        """
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        
        # Rate limiting
        self._last_request = 0
        self._min_interval = 0.1  # seconds
    
    def search(self, query: str, limit: int = 5) -> List[str]:
        """
        Search Wikipedia articles.
        
        Args:
            query: Search query
            limit: Max results
            
        Returns:
            List of article titles
        """
        self._rate_limit()
        
        try:
            import requests
            
            params = {
                "action": "opensearch",
                "search": query,
                "limit": limit,
                "format": "json"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            return data[1] if len(data) > 1 else []
            
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return []
    
    def get_article(self, title: str) -> Optional[WikipediaArticle]:
        """
        Get a Wikipedia article.
        
        Args:
            title: Article title
            
        Returns:
            Article data or None
        """
        self._rate_limit()
        
        try:
            import requests
            
            # Get summary
            params = {
                "action": "query",
                "titles": title,
                "prop": "extracts|categories|links",
                "exintro": True,
                "explaintext": True,
                "format": "json"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            pages = data.get("query", {}).get("pages", {})
            
            for page_id, page in pages.items():
                if page_id == "-1":
                    return None
                
                return WikipediaArticle(
                    title=page.get("title", title),
                    summary=page.get("extract", ""),
                    content=page.get("extract", ""),
                    url=f"https://{self.language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    categories=[c.get("title", "") for c in page.get("categories", [])],
                    links=[l.get("title", "") for l in page.get("links", [])],
                    page_id=int(page_id)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Wikipedia article fetch failed: {e}")
            return None
    
    def get_summary(self, title: str) -> str:
        """Get article summary."""
        article = self.get_article(title)
        return article.summary if article else ""
    
    def _rate_limit(self):
        """Apply rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()


class WikidataClient:
    """Wikidata API client."""
    
    def __init__(self):
        """Initialize Wikidata client."""
        self.base_url = "https://www.wikidata.org/w/api.php"
        self.sparql_url = "https://query.wikidata.org/sparql"
        
        self._last_request = 0
        self._min_interval = 0.2
    
    def get_entity(self, entity_id: str) -> Optional[WikidataEntity]:
        """
        Get a Wikidata entity.
        
        Args:
            entity_id: Entity ID (e.g., "Q42" for Douglas Adams)
            
        Returns:
            Entity data or None
        """
        self._rate_limit()
        
        try:
            import requests
            
            params = {
                "action": "wbgetentities",
                "ids": entity_id,
                "format": "json",
                "languages": "en"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            entities = data.get("entities", {})
            entity_data = entities.get(entity_id, {})
            
            if not entity_data or "missing" in entity_data:
                return None
            
            labels = entity_data.get("labels", {})
            descriptions = entity_data.get("descriptions", {})
            aliases_data = entity_data.get("aliases", {})
            
            return WikidataEntity(
                entity_id=entity_id,
                label=labels.get("en", {}).get("value", ""),
                description=descriptions.get("en", {}).get("value", ""),
                aliases=[a["value"] for a in aliases_data.get("en", [])],
                claims=entity_data.get("claims", {}),
                sitelinks=entity_data.get("sitelinks", {})
            )
            
        except Exception as e:
            logger.error(f"Wikidata entity fetch failed: {e}")
            return None
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search Wikidata entities.
        
        Args:
            query: Search query
            limit: Max results
            
        Returns:
            List of entity info dicts
        """
        self._rate_limit()
        
        try:
            import requests
            
            params = {
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "limit": limit,
                "format": "json"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            results = []
            for item in data.get("search", []):
                results.append({
                    "id": item.get("id", ""),
                    "label": item.get("label", ""),
                    "description": item.get("description", ""),
                    "url": item.get("url", "")
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Wikidata search failed: {e}")
            return []
    
    def sparql_query(self, query: str) -> List[Dict]:
        """
        Execute a SPARQL query.
        
        Args:
            query: SPARQL query
            
        Returns:
            Query results
        """
        self._rate_limit()
        
        try:
            import requests
            
            headers = {"Accept": "application/json"}
            
            response = requests.get(
                self.sparql_url,
                params={"query": query},
                headers=headers,
                timeout=30
            )
            
            data = response.json()
            
            results = []
            bindings = data.get("results", {}).get("bindings", [])
            
            for binding in bindings:
                result = {}
                for key, value in binding.items():
                    result[key] = value.get("value", "")
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return []
    
    def _rate_limit(self):
        """Apply rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()


class KnowledgeCache:
    """Cache for knowledge base results."""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_age: int = 86400):
        """
        Initialize cache.
        
        Args:
            cache_dir: Cache directory
            max_age: Max cache age in seconds
        """
        self.cache_dir = cache_dir or Path("data/kb_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached value."""
        cache_file = self.cache_dir / f"{self._hash_key(key)}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            data = json.loads(cache_file.read_text())
            
            # Check age
            if time.time() - data.get("cached_at", 0) > self.max_age:
                cache_file.unlink()
                return None
            
            return data.get("value")
            
        except Exception:
            return None
    
    def set(self, key: str, value: Any):
        """Set cached value."""
        cache_file = self.cache_dir / f"{self._hash_key(key)}.json"
        
        data = {
            "key": key,
            "value": value,
            "cached_at": time.time()
        }
        
        cache_file.write_text(json.dumps(data, default=str))
    
    def _hash_key(self, key: str) -> str:
        """Hash cache key."""
        import hashlib
        return hashlib.md5(key.encode()).hexdigest()


class ExternalKnowledge:
    """Unified external knowledge interface."""
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize external knowledge.
        
        Args:
            use_cache: Enable caching
        """
        self.wikipedia = WikipediaClient()
        self.wikidata = WikidataClient()
        
        self.cache = KnowledgeCache() if use_cache else None
    
    def search(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[KnowledgeSource]:
        """
        Search across knowledge sources.
        
        Args:
            query: Search query
            sources: Sources to search (default: all)
            limit: Max results per source
            
        Returns:
            List of knowledge sources
        """
        sources = sources or ["wikipedia", "wikidata"]
        results = []
        
        if "wikipedia" in sources:
            wiki_results = self.search_wikipedia(query, limit)
            results.extend(wiki_results)
        
        if "wikidata" in sources:
            wd_results = self.search_wikidata(query, limit)
            results.extend(wd_results)
        
        return results
    
    def search_wikipedia(self, query: str, limit: int = 5) -> List[KnowledgeSource]:
        """Search Wikipedia."""
        # Check cache
        cache_key = f"wiki_search:{query}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return [KnowledgeSource(**s) for s in cached]
        
        titles = self.wikipedia.search(query, limit)
        
        results = []
        for title in titles:
            article = self.wikipedia.get_article(title)
            if article:
                results.append(KnowledgeSource(
                    source="wikipedia",
                    title=article.title,
                    content=article.summary,
                    url=article.url,
                    metadata={"categories": article.categories}
                ))
        
        # Cache results
        if self.cache:
            self.cache.set(cache_key, [r.__dict__ for r in results])
        
        return results
    
    def search_wikidata(self, query: str, limit: int = 5) -> List[KnowledgeSource]:
        """Search Wikidata."""
        # Check cache
        cache_key = f"wd_search:{query}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return [KnowledgeSource(**s) for s in cached]
        
        entities = self.wikidata.search(query, limit)
        
        results = []
        for entity in entities:
            results.append(KnowledgeSource(
                source="wikidata",
                title=entity.get("label", ""),
                content=entity.get("description", ""),
                url=entity.get("url", ""),
                entity_id=entity.get("id", "")
            ))
        
        if self.cache:
            self.cache.set(cache_key, [r.__dict__ for r in results])
        
        return results
    
    def get_wikipedia_article(self, title: str) -> Optional[WikipediaArticle]:
        """Get Wikipedia article."""
        return self.wikipedia.get_article(title)
    
    def get_wikidata_entity(self, entity_id: str) -> Optional[WikidataEntity]:
        """Get Wikidata entity."""
        return self.wikidata.get_entity(entity_id)
    
    def enrich_context(self, text: str, max_sources: int = 3) -> str:
        """
        Enrich text with external knowledge.
        
        Args:
            text: Input text
            max_sources: Max sources to include
            
        Returns:
            Text with knowledge snippets
        """
        # Extract potential topics (simple: use nouns)
        words = text.split()
        # Just search for key terms
        key_terms = [w for w in words if len(w) > 5 and w[0].isupper()][:2]
        
        knowledge_snippets = []
        
        for term in key_terms:
            results = self.search(term, limit=1)
            if results:
                knowledge_snippets.append(f"[{results[0].title}]: {results[0].content[:200]}")
        
        if knowledge_snippets:
            return text + "\n\nRelevant Knowledge:\n" + "\n".join(knowledge_snippets[:max_sources])
        
        return text


# Global instance
_kb: Optional[ExternalKnowledge] = None


def get_knowledge_base() -> ExternalKnowledge:
    """Get or create global knowledge base."""
    global _kb
    if _kb is None:
        _kb = ExternalKnowledge()
    return _kb
