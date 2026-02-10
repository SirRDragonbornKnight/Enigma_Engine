"""
================================================================================
            WEB TRAINING - Web-Assisted Training Data Collection
================================================================================

Enables the Trainer AI to use the web to gather, curate, and generate
training data for any task type.

FILE: enigma_engine/core/web_training.py
TYPE: Training Data Collection

KEY FEATURES:
    - Scrape documentation sites for code examples
    - Gather Q&A pairs from Stack Overflow, Reddit
    - Extract tutorials and explanations
    - Mine conversational data from forums
    - Filter and clean web data for training
    - Respect robots.txt and rate limits

USAGE:
    from enigma_engine.core.web_training import WebTrainingCollector
    
    collector = WebTrainingCollector()
    
    # Scrape a documentation site for code training data
    code_data = collector.scrape_documentation(
        "https://docs.python.org/3/",
        topic="python"
    )
    
    # Search for Q&A pairs
    qa_data = collector.search_qa_pairs(
        query="how to use async await",
        source="stackoverflow"
    )
    
    # Gather tutorial content
    tutorials = collector.gather_tutorials(
        topic="machine learning",
        max_results=20
    )
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


# Try to import web scraping libraries
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WebContent:
    """Scraped web content."""
    url: str
    title: str
    content: str
    content_type: str  # text, code, qa, tutorial, conversation
    timestamp: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "content_type": self.content_type,
            "timestamp": self.timestamp,
            "source": self.source,
            "metadata": self.metadata,
            "quality_score": self.quality_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebContent':
        return cls(**data)


@dataclass
class TrainingExample:
    """A training example extracted from web content."""
    input_text: str
    output_text: str
    task_type: str
    source_url: str
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_training_line(self, prefix: str = "INPUT:", suffix: str = "OUTPUT:") -> str:
        """Convert to training data format."""
        return f"{prefix} {self.input_text} | {suffix} {self.output_text}"


@dataclass
class ScrapeConfig:
    """Configuration for web scraping."""
    max_pages: int = 10
    delay_seconds: float = 1.0
    respect_robots: bool = True
    user_agent: str = "EnigmaEngine-TrainingBot/1.0 (+https://github.com/enigma-engine)"
    timeout: int = 30
    min_content_length: int = 100
    max_content_length: int = 50000


# =============================================================================
# WEB CONTENT EXTRACTORS
# =============================================================================

class ContentExtractor:
    """Extract specific content types from web pages."""
    
    @staticmethod
    def extract_code_blocks(html_content: str) -> List[Dict[str, str]]:
        """Extract code blocks from HTML."""
        if not HAS_BS4:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        code_blocks = []
        
        # Look for <pre><code> patterns
        for pre in soup.find_all('pre'):
            code = pre.find('code')
            if code:
                lang = code.get('class', [''])[0].replace('language-', '') if code.get('class') else ''
                code_blocks.append({
                    'code': code.get_text(strip=True),
                    'language': lang,
                })
        
        # Also look for standalone <code> elements
        for code in soup.find_all('code'):
            if code.parent.name != 'pre':  # Skip already extracted
                text = code.get_text(strip=True)
                if len(text) > 20:  # Only substantial code
                    code_blocks.append({
                        'code': text,
                        'language': '',
                    })
        
        return code_blocks
    
    @staticmethod
    def extract_qa_pairs(html_content: str, page_url: str) -> List[Dict[str, str]]:
        """Extract Q&A pairs from pages like Stack Overflow."""
        if not HAS_BS4:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        qa_pairs = []
        
        # Stack Overflow style
        if 'stackoverflow.com' in page_url:
            question = soup.find('div', class_='question')
            if question:
                q_body = question.find('div', class_='js-post-body')
                q_text = q_body.get_text(strip=True) if q_body else ""
                
                # Get accepted answer or top answer
                answers = soup.find_all('div', class_='answer')
                for answer in answers:
                    a_body = answer.find('div', class_='js-post-body')
                    if a_body:
                        a_text = a_body.get_text(strip=True)
                        qa_pairs.append({
                            'question': q_text[:500],
                            'answer': a_text[:2000],
                        })
                        break  # Just take the first good answer
        
        # Generic FAQ extraction
        else:
            # Look for definition lists
            for dl in soup.find_all('dl'):
                dts = dl.find_all('dt')
                dds = dl.find_all('dd')
                for dt, dd in zip(dts, dds):
                    qa_pairs.append({
                        'question': dt.get_text(strip=True),
                        'answer': dd.get_text(strip=True),
                    })
            
            # Look for headers followed by paragraphs
            for header in soup.find_all(['h2', 'h3', 'h4']):
                if '?' in header.get_text():
                    next_p = header.find_next('p')
                    if next_p:
                        qa_pairs.append({
                            'question': header.get_text(strip=True),
                            'answer': next_p.get_text(strip=True),
                        })
        
        return qa_pairs
    
    @staticmethod
    def extract_tutorials(html_content: str) -> List[Dict[str, Any]]:
        """Extract tutorial/guide content."""
        if not HAS_BS4:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        tutorials = []
        
        # Look for article/tutorial structure
        article = soup.find('article') or soup.find('main') or soup.find('div', class_=re.compile(r'content|tutorial|guide'))
        
        if article:
            sections = []
            current_section = {"title": "", "content": []}
            
            for elem in article.children:
                if hasattr(elem, 'name'):
                    if elem.name in ['h1', 'h2', 'h3']:
                        if current_section['content']:
                            sections.append(current_section)
                        current_section = {
                            "title": elem.get_text(strip=True),
                            "content": []
                        }
                    elif elem.name in ['p', 'ul', 'ol', 'pre']:
                        current_section['content'].append(elem.get_text(strip=True))
            
            if current_section['content']:
                sections.append(current_section)
            
            for section in sections:
                if section['title'] and section['content']:
                    tutorials.append({
                        'title': section['title'],
                        'steps': section['content'],
                    })
        
        return tutorials


# =============================================================================
# WEB TRAINING DATA COLLECTOR
# =============================================================================

class WebTrainingCollector:
    """
    Collects training data from the web.
    
    Supports multiple sources and content types for diverse training data.
    """
    
    def __init__(self, config: Optional[ScrapeConfig] = None):
        if not HAS_REQUESTS:
            logger.warning("requests library not installed - web scraping disabled")
        if not HAS_BS4:
            logger.warning("beautifulsoup4 library not installed - HTML parsing disabled")
        
        self.config = config or ScrapeConfig()
        self.session = requests.Session() if HAS_REQUESTS else None
        if self.session:
            self.session.headers.update({'User-Agent': self.config.user_agent})
        
        self.cache_dir = Path("data/web_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.extractor = ContentExtractor()
        
        # Rate limiting
        self._last_request_time: Dict[str, float] = {}
    
    # ─────────────────────────────────────────────────────────────────────────
    # LOW-LEVEL FETCHING
    # ─────────────────────────────────────────────────────────────────────────
    
    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch a web page with rate limiting and caching."""
        if not self.session:
            return None
        
        # Check cache
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.html"
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 86400:  # 24 hour cache
                return cache_file.read_text(encoding='utf-8')
        
        # Rate limiting
        domain = urlparse(url).netloc
        if domain in self._last_request_time:
            elapsed = time.time() - self._last_request_time[domain]
            if elapsed < self.config.delay_seconds:
                time.sleep(self.config.delay_seconds - elapsed)
        
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            
            self._last_request_time[domain] = time.time()
            
            # Cache the response
            cache_file.write_text(response.text, encoding='utf-8')
            
            return response.text
            
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Decode HTML entities
        text = html.unescape(text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common noise
        text = re.sub(r'Cookie\s+Policy|Privacy\s+Policy|Terms\s+of\s+Service', '', text, flags=re.I)
        return text.strip()
    
    # ─────────────────────────────────────────────────────────────────────────
    # HIGH-LEVEL COLLECTION METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def scrape_for_training(
        self,
        url: str,
        topic: str,
        task_type: str = "chat",
    ) -> Dict[str, Any]:
        """
        Scrape a URL and extract training data.
        
        Args:
            url: URL to scrape
            topic: Topic for context
            task_type: Type of training data to extract
            
        Returns:
            Dictionary with extracted content and training examples
        """
        html_content = self._fetch_page(url)
        if not html_content:
            return {"error": "Failed to fetch page", "examples": []}
        
        result = {
            "url": url,
            "topic": topic,
            "task_type": task_type,
            "timestamp": time.time(),
            "examples": [],
        }
        
        if task_type == "code":
            code_blocks = self.extractor.extract_code_blocks(html_content)
            for block in code_blocks:
                if len(block['code']) > 20:
                    result["examples"].append(TrainingExample(
                        input_text=f"Write {block['language'] or 'code'} for {topic}",
                        output_text=block['code'],
                        task_type="code",
                        source_url=url,
                        quality_score=self._score_code_quality(block['code']),
                    ))
        
        elif task_type == "qa" or task_type == "chat":
            qa_pairs = self.extractor.extract_qa_pairs(html_content, url)
            for qa in qa_pairs:
                result["examples"].append(TrainingExample(
                    input_text=qa['question'],
                    output_text=qa['answer'],
                    task_type=task_type,
                    source_url=url,
                    quality_score=self._score_qa_quality(qa),
                ))
        
        elif task_type == "tutorial":
            tutorials = self.extractor.extract_tutorials(html_content)
            for tut in tutorials:
                # Convert tutorial to instruction-following format
                steps = '\n'.join(f"Step {i+1}: {s}" for i, s in enumerate(tut['steps']))
                result["examples"].append(TrainingExample(
                    input_text=f"How do I {tut['title'].lower()}?",
                    output_text=steps,
                    task_type="chat",
                    source_url=url,
                    quality_score=0.7,
                ))
        
        return result
    
    def scrape_documentation(
        self,
        base_url: str,
        topic: str,
        max_pages: Optional[int] = None,
    ) -> List[TrainingExample]:
        """
        Scrape a documentation site for training data.
        
        Args:
            base_url: Base URL of documentation
            topic: Topic name (e.g., "python", "pytorch")
            max_pages: Maximum pages to scrape
            
        Returns:
            List of training examples
        """
        if not HAS_BS4 or not self.session:
            logger.warning("Web scraping not available")
            return []
        
        max_pages = max_pages or self.config.max_pages
        visited = set()
        to_visit = [base_url]
        examples = []
        
        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            
            visited.add(url)
            logger.debug(f"Scraping: {url}")
            
            html_content = self._fetch_page(url)
            if not html_content:
                continue
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract code examples
            code_blocks = self.extractor.extract_code_blocks(html_content)
            for block in code_blocks:
                # Find nearby context
                context = self._find_code_context(soup, block['code'])
                if context:
                    examples.append(TrainingExample(
                        input_text=f"{context}",
                        output_text=block['code'],
                        task_type="code",
                        source_url=url,
                        metadata={"language": block.get('language', topic)},
                    ))
            
            # Find links to other documentation pages
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                
                # Only follow internal links
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    if full_url not in visited:
                        to_visit.append(full_url)
        
        logger.info(f"Scraped {len(visited)} pages, extracted {len(examples)} examples")
        return examples
    
    def search_qa_pairs(
        self,
        query: str,
        source: str = "web",
        max_results: int = 20,
    ) -> List[TrainingExample]:
        """
        Search for Q&A pairs on the web.
        
        Args:
            query: Search query
            source: Source preference (web, stackoverflow, etc.)
            max_results: Maximum results
            
        Returns:
            List of Q&A training examples
        """
        examples = []
        
        # Search URLs for different sources
        search_urls = []
        
        if source in ["web", "stackoverflow"]:
            # We'd normally use an API, but for demo, use site search
            search_urls.append(
                f"https://stackoverflow.com/search?q={query.replace(' ', '+')}"
            )
        
        for url in search_urls:
            content = self._fetch_page(url)
            if content and HAS_BS4:
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract question links
                question_links = []
                for link in soup.find_all('a', class_=re.compile(r'question-hyperlink|s-link')):
                    href = link.get('href', '')
                    if '/questions/' in href:
                        full_url = urljoin(url, href)
                        question_links.append(full_url)
                
                # Fetch individual questions
                for q_url in question_links[:max_results]:
                    result = self.scrape_for_training(q_url, query, "qa")
                    examples.extend(result.get("examples", []))
        
        return examples
    
    def gather_tutorials(
        self,
        topic: str,
        max_results: int = 10,
    ) -> List[TrainingExample]:
        """
        Gather tutorial content for a topic.
        
        Args:
            topic: Topic to find tutorials for
            max_results: Maximum tutorials
            
        Returns:
            List of tutorial-based training examples
        """
        examples = []
        
        # Common tutorial sites
        sites = [
            f"https://realpython.com/search?q={topic.replace(' ', '+')}",
            f"https://www.w3schools.com/search/search_result.asp?q={topic.replace(' ', '+')}",
        ]
        
        for site_url in sites:
            result = self.scrape_for_training(site_url, topic, "tutorial")
            examples.extend(result.get("examples", []))
            
            if len(examples) >= max_results:
                break
        
        return examples[:max_results]
    
    # ─────────────────────────────────────────────────────────────────────────
    # QUALITY SCORING
    # ─────────────────────────────────────────────────────────────────────────
    
    def _score_code_quality(self, code: str) -> float:
        """Score code quality for training."""
        score = 0.5  # Base score
        
        # Length checks
        if 20 < len(code) < 1000:
            score += 0.1
        
        # Has comments
        if '#' in code or '//' in code or '"""' in code:
            score += 0.1
        
        # Has function/class definitions
        if re.search(r'def\s+\w+|class\s+\w+|function\s+\w+', code):
            score += 0.2
        
        # Not just boilerplate
        if not re.match(r'^(import|from|#!)', code):
            score += 0.1
        
        return min(1.0, score)
    
    def _score_qa_quality(self, qa: Dict[str, str]) -> float:
        """Score Q&A pair quality."""
        score = 0.5
        
        q_len = len(qa.get('question', ''))
        a_len = len(qa.get('answer', ''))
        
        # Question length
        if 10 < q_len < 500:
            score += 0.1
        
        # Answer length (substantial but not too long)
        if 50 < a_len < 2000:
            score += 0.2
        
        # Question has question mark
        if '?' in qa.get('question', ''):
            score += 0.1
        
        # Answer is longer than question
        if a_len > q_len:
            score += 0.1
        
        return min(1.0, score)
    
    def _find_code_context(self, soup: BeautifulSoup, code: str) -> Optional[str]:
        """Find context text near a code block."""
        # Look for the element containing this code
        code_elem = soup.find(string=re.compile(re.escape(code[:50]) if len(code) > 50 else re.escape(code)))
        
        if code_elem:
            # Look for preceding paragraph
            parent = code_elem.find_parent()
            if parent:
                prev = parent.find_previous_sibling(['p', 'h1', 'h2', 'h3', 'li'])
                if prev:
                    return self._clean_text(prev.get_text())[:200]
        
        return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # BULK COLLECTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def collect_training_data(
        self,
        topics: List[str],
        task_types: List[str],
        examples_per_topic: int = 50,
        output_file: Optional[Path] = None,
    ) -> List[TrainingExample]:
        """
        Collect training data for multiple topics and task types.
        
        Args:
            topics: List of topics to collect data for
            task_types: List of task types (code, chat, qa, etc.)
            examples_per_topic: Target examples per topic
            output_file: Optional file to save collected data
            
        Returns:
            All collected training examples
        """
        all_examples = []
        
        for topic in topics:
            logger.info(f"Collecting data for topic: {topic}")
            
            for task_type in task_types:
                if task_type == "code":
                    # Try popular documentation sites
                    doc_urls = self._get_doc_urls_for_topic(topic)
                    for url in doc_urls[:3]:  # Limit per topic
                        examples = self.scrape_documentation(url, topic, max_pages=5)
                        all_examples.extend(examples[:examples_per_topic // 3])
                
                elif task_type in ["qa", "chat"]:
                    examples = self.search_qa_pairs(topic, max_results=examples_per_topic)
                    all_examples.extend(examples)
                
                elif task_type == "tutorial":
                    examples = self.gather_tutorials(topic, max_results=examples_per_topic // 2)
                    all_examples.extend(examples)
        
        # Remove duplicates
        unique_examples = self._deduplicate_examples(all_examples)
        
        # Save if output file specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            training_lines = [ex.to_training_line() for ex in unique_examples]
            output_file.write_text('\n'.join(training_lines), encoding='utf-8')
            logger.info(f"Saved {len(unique_examples)} examples to {output_file}")
        
        return unique_examples
    
    def _get_doc_urls_for_topic(self, topic: str) -> List[str]:
        """Get documentation URLs for a topic."""
        topic_lower = topic.lower()
        
        known_docs = {
            "python": ["https://docs.python.org/3/library/", "https://realpython.com/"],
            "pytorch": ["https://pytorch.org/docs/stable/", "https://pytorch.org/tutorials/"],
            "javascript": ["https://developer.mozilla.org/en-US/docs/Web/JavaScript/"],
            "react": ["https://reactjs.org/docs/", "https://react.dev/"],
            "rust": ["https://doc.rust-lang.org/book/", "https://doc.rust-lang.org/std/"],
        }
        
        return known_docs.get(topic_lower, [f"https://www.google.com/search?q={topic}+tutorial"])
    
    def _deduplicate_examples(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Remove duplicate training examples."""
        seen = set()
        unique = []
        
        for ex in examples:
            key = hashlib.md5(f"{ex.input_text}{ex.output_text}".encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(ex)
        
        return unique
    
    # ─────────────────────────────────────────────────────────────────────────
    # INTEGRATION WITH TRAINER AI
    # ─────────────────────────────────────────────────────────────────────────
    
    def generate_for_trainer_ai(
        self,
        position: str,
        count: int = 100,
        topics: Optional[List[str]] = None,
    ) -> str:
        """
        Generate training data in Trainer AI format.
        
        Args:
            position: Router position (code, chat, router, etc.)
            count: Number of examples to generate
            topics: Optional list of topics to focus on
            
        Returns:
            Training data as string in proper format
        """
        # Default topics per position
        default_topics = {
            "code": ["python", "javascript", "programming patterns"],
            "chat": ["general knowledge", "helpful responses", "conversation"],
            "router": ["intent classification", "task routing"],
            "math": ["algebra", "calculus", "statistics"],
            "vision": ["image description", "visual analysis"],
        }
        
        topics = topics or default_topics.get(position, ["general"])
        
        # Map position to task types
        task_type_map = {
            "code": ["code"],
            "chat": ["qa", "chat"],
            "router": ["qa"],
            "math": ["qa", "tutorial"],
            "vision": ["tutorial"],
        }
        
        task_types = task_type_map.get(position, ["chat"])
        
        # Collect examples
        examples = self.collect_training_data(
            topics=topics,
            task_types=task_types,
            examples_per_topic=count // len(topics),
        )
        
        # Get position config for proper format
        from .trainer_ai import POSITION_CONFIGS
        config = POSITION_CONFIGS.get(position)
        
        if config:
            prefix = config.input_prefix
            suffix = config.output_prefix
            sep = config.separator
        else:
            prefix = "INPUT:"
            suffix = "OUTPUT:"
            sep = " | "
        
        # Format training data
        lines = []
        for ex in examples[:count]:
            line = f"{prefix} {ex.input_text}{sep}{suffix} {ex.output_text}"
            lines.append(line)
        
        return '\n'.join(lines)


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_web_collector: Optional[WebTrainingCollector] = None


def get_web_collector() -> WebTrainingCollector:
    """Get the global web training collector."""
    global _web_collector
    if _web_collector is None:
        _web_collector = WebTrainingCollector()
    return _web_collector


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def scrape_training_data(url: str, topic: str, task_type: str = "chat") -> Dict[str, Any]:
    """Quick function to scrape a URL for training data."""
    return get_web_collector().scrape_for_training(url, topic, task_type)


def collect_training_data(topics: List[str], task_types: List[str], **kwargs) -> List[TrainingExample]:
    """Collect training data from the web."""
    return get_web_collector().collect_training_data(topics, task_types, **kwargs)


def generate_web_training_data(position: str, count: int = 100) -> str:
    """Generate training data using web sources."""
    return get_web_collector().generate_for_trainer_ai(position, count)
