"""
Knowledge & Research Tools - External API lookups for factual information.

Tools (External APIs only - things AI can't learn):
  - wikipedia_search: Search and summarize Wikipedia articles
  - arxiv_search: Search academic papers on arXiv  
  - pdf_extract: Extract text from PDFs (requires library)

Removed (AI should learn file operations instead):
  - bookmark_* tools → Use file_tools or AI learns JSON handling
  - note_* tools → Use file_tools or AI learns file handling
"""

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .tool_registry import Tool, RichParameter


class WikipediaSearchTool(Tool):
    """Search and summarize Wikipedia articles."""
    
    name = "wikipedia_search"
    description = "Search Wikipedia for factual information. Returns article summary and URL."
    parameters = {
        "query": "Search query or article title",
        "sentences": "Sentences to return (default: 5)",
        "language": "Language code: en, es, de, fr, ja, zh (default: en)",
    }
    category = "knowledge"
    rich_parameters = [
        RichParameter(
            name="query",
            type="string",
            description="Search query or article title",
            required=True,
        ),
        RichParameter(
            name="sentences",
            type="integer",
            description="Number of sentences to return",
            required=False,
            default=5,
            min_value=1,
            max_value=20,
        ),
        RichParameter(
            name="language",
            type="string",
            description="Wikipedia language edition",
            required=False,
            default="en",
            enum=["en", "es", "de", "fr", "ja", "zh", "ru", "pt", "it"],
        ),
    ]
    examples = [
        "wikipedia_search(query='Albert Einstein')",
        "wikipedia_search(query='machine learning', sentences=10)",
        "wikipedia_search(query='Paris', language='fr')",
    ]
    
    def execute(self, query: str, sentences: int = 5, language: str = "en", **kwargs) -> dict[str, Any]:
        try:
            # Search for page
            search_url = f"https://{language}.wikipedia.org/w/api.php"
            params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 5}
            
            with urllib.request.urlopen(f"{search_url}?{urllib.parse.urlencode(params)}", timeout=10) as r:
                search_data = json.loads(r.read().decode())
            
            results = search_data.get("query", {}).get("search", [])
            if not results:
                return {"success": False, "error": "No Wikipedia articles found"}
            
            title = results[0]["title"]
            
            # Get summary
            summary_url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
            
            with urllib.request.urlopen(summary_url, timeout=10) as r:
                data = json.loads(r.read().decode())
            
            # Truncate to sentences
            extract = data.get("extract", "")
            if sentences > 0:
                extract = ' '.join(re.split(r'(?<=[.!?])\s+', extract)[:sentences])
            
            return {
                "success": True,
                "title": data.get("title", title),
                "summary": extract,
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "related": [r["title"] for r in results[1:4]],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class ArxivSearchTool(Tool):
    """Search arXiv for academic papers."""
    
    name = "arxiv_search"
    description = "Search arXiv.org for academic papers. Returns titles, authors, abstracts."
    parameters = {
        "query": "Search query",
        "max_results": "Max results (default: 5, max: 20)",
        "sort_by": "Sort: relevance, lastUpdatedDate, submittedDate",
    }
    category = "knowledge"
    rich_parameters = [
        RichParameter(
            name="query",
            type="string",
            description="Search query for papers",
            required=True,
        ),
        RichParameter(
            name="max_results",
            type="integer",
            description="Maximum number of results",
            required=False,
            default=5,
            min_value=1,
            max_value=20,
        ),
        RichParameter(
            name="sort_by",
            type="string",
            description="Sort order for results",
            required=False,
            default="relevance",
            enum=["relevance", "lastUpdatedDate", "submittedDate"],
        ),
    ]
    examples = [
        "arxiv_search(query='transformer neural networks')",
        "arxiv_search(query='quantum computing', max_results=10, sort_by='submittedDate')",
    ]
    
    def execute(self, query: str, max_results: int = 5, sort_by: str = "relevance", **kwargs) -> dict[str, Any]:
        try:
            import xml.etree.ElementTree as ET
            
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": min(int(max_results), 20),
                "sortBy": sort_by,
                "sortOrder": "descending",
            }
            
            url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
            
            with urllib.request.urlopen(url, timeout=15) as r:
                xml_data = r.read().decode()
            
            root = ET.fromstring(xml_data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            
            papers = []
            for entry in root.findall("atom:entry", ns):
                # Extract fields
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                published = entry.find("atom:published", ns)
                
                authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns) 
                          if a.find("atom:name", ns) is not None]
                
                links = entry.findall("atom:link", ns)
                pdf_link = next((l.get("href") for l in links if l.get("title") == "pdf"), None)
                
                categories = [c.get("term") for c in entry.findall("atom:category", ns)]
                
                papers.append({
                    "title": title.text.strip().replace('\n', ' ') if title is not None else "",
                    "authors": authors[:5],  # Limit authors
                    "abstract": (summary.text.strip().replace('\n', ' ')[:500] + "...") if summary is not None else "",
                    "published": published.text[:10] if published is not None else "",
                    "pdf": pdf_link,
                    "categories": categories[:3],
                })
            
            return {"success": True, "count": len(papers), "papers": papers}
        except Exception as e:
            return {"success": False, "error": str(e)}


class PDFExtractTool(Tool):
    """Extract text from PDF files."""
    
    name = "pdf_extract"
    description = "Extract text from PDF files. Requires pymupdf, pdfplumber, or pypdf library."
    parameters = {
        "path": "Path to PDF file",
        "pages": "Pages to extract: 'all', '1-5', '1,3,5' (default: first 10)",
        "max_pages": "Max pages (default: 10)",
    }
    category = "knowledge"
    rich_parameters = [
        RichParameter(
            name="path",
            type="string",
            description="Path to PDF file",
            required=True,
        ),
        RichParameter(
            name="pages",
            type="string",
            description="Pages to extract: 'all', '1-5', '1,3,5'",
            required=False,
            default="all",
        ),
        RichParameter(
            name="max_pages",
            type="integer",
            description="Maximum pages to extract",
            required=False,
            default=10,
            min_value=1,
            max_value=100,
        ),
    ]
    examples = [
        "pdf_extract(path='research_paper.pdf')",
        "pdf_extract(path='book.pdf', pages='1-20', max_pages=20)",
    ]
    
    def execute(self, path: str, pages: str = "all", max_pages: int = 10, **kwargs) -> dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            if not path.exists():
                return {"success": False, "error": f"File not found: {path}"}
            if not path.suffix.lower() == '.pdf':
                return {"success": False, "error": "Not a PDF file"}
            
            # Parse page selection
            page_set = None
            if pages != "all":
                page_set = set()
                for part in pages.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        page_set.update(range(start-1, end))
                    else:
                        page_set.add(int(part) - 1)
            
            text, page_count, method = "", 0, ""
            
            # Try PyMuPDF (fastest)
            try:
                import fitz
                doc = fitz.open(str(path))
                page_count = len(doc)
                texts = []
                for i, page in enumerate(doc):
                    if i >= max_pages: break
                    if page_set and i not in page_set: continue
                    texts.append(f"--- Page {i+1} ---\n{page.get_text()}")
                text = "\n".join(texts)
                method = "pymupdf"
                doc.close()
            except ImportError:
                # Try pdfplumber
                try:
                    import pdfplumber
                    with pdfplumber.open(str(path)) as pdf:
                        page_count = len(pdf.pages)
                        texts = []
                        for i, page in enumerate(pdf.pages):
                            if i >= max_pages: break
                            if page_set and i not in page_set: continue
                            texts.append(f"--- Page {i+1} ---\n{page.extract_text() or ''}")
                        text = "\n".join(texts)
                        method = "pdfplumber"
                except ImportError:
                    # Try pypdf
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(str(path))
                        page_count = len(reader.pages)
                        texts = []
                        for i, page in enumerate(reader.pages):
                            if i >= max_pages: break
                            if page_set and i not in page_set: continue
                            texts.append(f"--- Page {i+1} ---\n{page.extract_text()}")
                        text = "\n".join(texts)
                        method = "pypdf"
                    except ImportError:
                        return {"success": False, "error": "No PDF library. Install: pip install pymupdf"}
            
            return {
                "success": True,
                "path": str(path),
                "page_count": page_count,
                "pages_extracted": len(page_set) if page_set else min(page_count, max_pages),
                "method": method,
                "text": text[:50000],
                "text_length": len(text),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# Export tools - only the ones that need external APIs/libraries
KNOWLEDGE_TOOLS = [
    WikipediaSearchTool(),
    ArxivSearchTool(),
    PDFExtractTool(),
]

def get_knowledge_tools():
    """Get all knowledge tools."""
    return KNOWLEDGE_TOOLS
