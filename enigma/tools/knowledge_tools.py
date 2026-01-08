"""
Knowledge & Research Tools - Wikipedia, arXiv, PDFs, bookmarks, notes.

Tools:
  - wikipedia_search: Search and summarize Wikipedia articles
  - arxiv_search: Search academic papers on arXiv
  - pdf_extract: Extract text and tables from PDFs
  - bookmark_save: Save a bookmark with tags
  - bookmark_search: Search saved bookmarks
  - note_save: Save a persistent note
  - note_get: Retrieve a note
  - note_search: Search notes
  - note_list: List all notes
"""

import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from .tool_registry import Tool

# Storage paths
BOOKMARKS_FILE = Path.home() / ".enigma" / "bookmarks.json"
NOTES_DIR = Path.home() / ".enigma" / "notes"
WIKI_CACHE_DIR = Path.home() / ".enigma" / "cache" / "wikipedia"

BOOKMARKS_FILE.parent.mkdir(parents=True, exist_ok=True)
NOTES_DIR.mkdir(parents=True, exist_ok=True)
WIKI_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# WIKIPEDIA TOOLS
# ============================================================================

class WikipediaSearchTool(Tool):
    """Search and summarize Wikipedia articles."""
    
    name = "wikipedia_search"
    description = "Search Wikipedia and get article summaries. Great for factual information."
    parameters = {
        "query": "The search query or article title",
        "sentences": "Number of sentences to return (default: 5)",
        "language": "Wikipedia language code (default: en)",
    }
    
    def execute(self, query: str, sentences: int = 5, language: str = "en", **kwargs) -> Dict[str, Any]:
        try:
            import urllib.request
            import urllib.parse
            
            # Use Wikipedia API
            base_url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/"
            
            # First, search for the page
            search_url = f"https://{language}.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 5,
            }
            
            url = search_url + "?" + urllib.parse.urlencode(search_params)
            
            with urllib.request.urlopen(url, timeout=10) as response:
                search_data = json.loads(response.read().decode())
            
            if not search_data.get("query", {}).get("search"):
                return {"success": False, "error": "No Wikipedia articles found"}
            
            # Get the top result
            top_result = search_data["query"]["search"][0]
            title = top_result["title"]
            
            # Get the summary
            summary_url = base_url + urllib.parse.quote(title)
            
            with urllib.request.urlopen(summary_url, timeout=10) as response:
                summary_data = json.loads(response.read().decode())
            
            # Truncate to requested sentences
            extract = summary_data.get("extract", "")
            if sentences and sentences > 0:
                sentence_list = re.split(r'(?<=[.!?])\s+', extract)
                extract = ' '.join(sentence_list[:sentences])
            
            return {
                "success": True,
                "title": summary_data.get("title", title),
                "summary": extract,
                "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "description": summary_data.get("description", ""),
                "related_searches": [r["title"] for r in search_data["query"]["search"][1:4]],
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# ARXIV TOOLS
# ============================================================================

class ArxivSearchTool(Tool):
    """Search arXiv for academic papers."""
    
    name = "arxiv_search"
    description = "Search arXiv.org for academic papers and research. Returns titles, authors, and abstracts."
    parameters = {
        "query": "Search query (supports arXiv search syntax)",
        "max_results": "Maximum results to return (default: 5)",
        "sort_by": "Sort by: 'relevance', 'lastUpdatedDate', 'submittedDate' (default: relevance)",
    }
    
    def execute(self, query: str, max_results: int = 5, sort_by: str = "relevance", **kwargs) -> Dict[str, Any]:
        try:
            import urllib.request
            import urllib.parse
            import xml.etree.ElementTree as ET
            
            # Build arXiv API URL
            base_url = "http://export.arxiv.org/api/query?"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": min(int(max_results), 20),
                "sortBy": sort_by,
                "sortOrder": "descending",
            }
            
            url = base_url + urllib.parse.urlencode(params)
            
            with urllib.request.urlopen(url, timeout=15) as response:
                xml_data = response.read().decode()
            
            # Parse XML
            root = ET.fromstring(xml_data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            
            papers = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                published = entry.find("atom:published", ns)
                
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None:
                        authors.append(name.text)
                
                # Get PDF link
                pdf_link = ""
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_link = link.get("href", "")
                        break
                
                # Get arXiv ID
                arxiv_id = entry.find("atom:id", ns)
                
                papers.append({
                    "title": title.text.strip() if title is not None else "",
                    "authors": authors[:5],  # Limit authors
                    "abstract": summary.text.strip()[:500] if summary is not None else "",
                    "published": published.text[:10] if published is not None else "",
                    "arxiv_id": arxiv_id.text.split("/")[-1] if arxiv_id is not None else "",
                    "pdf_url": pdf_link,
                })
            
            return {
                "success": True,
                "count": len(papers),
                "papers": papers,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# PDF TOOLS
# ============================================================================

class PDFExtractTool(Tool):
    """Extract text from PDF files."""
    
    name = "pdf_extract"
    description = "Extract text content from a PDF file. Can also extract specific pages."
    parameters = {
        "path": "Path to the PDF file",
        "pages": "Page numbers to extract (comma-separated, e.g., '1,2,5'). Default: all",
        "max_pages": "Maximum pages to extract (default: 50)",
    }
    
    def execute(self, path: str, pages: str = None, max_pages: int = 50, **kwargs) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            
            if not path.exists():
                return {"success": False, "error": f"File not found: {path}"}
            
            if path.suffix.lower() != '.pdf':
                return {"success": False, "error": "Not a PDF file"}
            
            # Try PyMuPDF (fitz) first, then pdfplumber, then pypdf
            text = ""
            page_count = 0
            method_used = ""
            
            # Parse page selection
            page_set = None
            if pages:
                page_set = set(int(p.strip()) - 1 for p in pages.split(','))  # Convert to 0-indexed
            
            # Try PyMuPDF
            try:
                import fitz
                doc = fitz.open(str(path))
                page_count = len(doc)
                
                texts = []
                for i, page in enumerate(doc):
                    if i >= int(max_pages):
                        break
                    if page_set and i not in page_set:
                        continue
                    texts.append(f"--- Page {i+1} ---\n{page.get_text()}")
                
                text = "\n".join(texts)
                method_used = "PyMuPDF"
                doc.close()
                
            except ImportError:
                # Try pdfplumber
                try:
                    import pdfplumber
                    with pdfplumber.open(str(path)) as pdf:
                        page_count = len(pdf.pages)
                        texts = []
                        for i, page in enumerate(pdf.pages):
                            if i >= int(max_pages):
                                break
                            if page_set and i not in page_set:
                                continue
                            page_text = page.extract_text() or ""
                            texts.append(f"--- Page {i+1} ---\n{page_text}")
                        text = "\n".join(texts)
                        method_used = "pdfplumber"
                        
                except ImportError:
                    # Try pypdf
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(str(path))
                        page_count = len(reader.pages)
                        
                        texts = []
                        for i, page in enumerate(reader.pages):
                            if i >= int(max_pages):
                                break
                            if page_set and i not in page_set:
                                continue
                            texts.append(f"--- Page {i+1} ---\n{page.extract_text()}")
                        
                        text = "\n".join(texts)
                        method_used = "pypdf"
                        
                    except ImportError:
                        return {
                            "success": False, 
                            "error": "No PDF library installed. Install: pip install pymupdf pdfplumber pypdf"
                        }
            
            return {
                "success": True,
                "path": str(path),
                "page_count": page_count,
                "pages_extracted": len(page_set) if page_set else min(page_count, int(max_pages)),
                "method": method_used,
                "text": text[:50000],  # Limit output
                "text_length": len(text),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# BOOKMARK TOOLS
# ============================================================================

class BookmarkManager:
    """Manages bookmarks."""
    
    def __init__(self):
        self.bookmarks: List[Dict] = []
        self._load_bookmarks()
    
    def _load_bookmarks(self):
        if BOOKMARKS_FILE.exists():
            try:
                with open(BOOKMARKS_FILE, 'r') as f:
                    self.bookmarks = json.load(f)
            except:
                self.bookmarks = []
    
    def _save_bookmarks(self):
        with open(BOOKMARKS_FILE, 'w') as f:
            json.dump(self.bookmarks, f, indent=2)
    
    def add(self, url: str, title: str, tags: List[str] = None, notes: str = None) -> Dict:
        """Add a bookmark."""
        bookmark = {
            "id": len(self.bookmarks) + 1,
            "url": url,
            "title": title,
            "tags": tags or [],
            "notes": notes or "",
            "created": datetime.now().isoformat(),
        }
        self.bookmarks.append(bookmark)
        self._save_bookmarks()
        return bookmark
    
    def search(self, query: str, tags: List[str] = None) -> List[Dict]:
        """Search bookmarks."""
        results = []
        query_lower = query.lower() if query else ""
        
        for bm in self.bookmarks:
            # Tag filter
            if tags:
                if not any(t in bm.get('tags', []) for t in tags):
                    continue
            
            # Text search
            if query_lower:
                searchable = f"{bm['title']} {bm['url']} {bm.get('notes', '')} {' '.join(bm.get('tags', []))}".lower()
                if query_lower not in searchable:
                    continue
            
            results.append(bm)
        
        return results
    
    def delete(self, bookmark_id: int):
        """Delete a bookmark."""
        self.bookmarks = [b for b in self.bookmarks if b['id'] != bookmark_id]
        self._save_bookmarks()
    
    def list_all(self) -> List[Dict]:
        """List all bookmarks."""
        return self.bookmarks


class BookmarkSaveTool(Tool):
    """Save a bookmark."""
    
    name = "bookmark_save"
    description = "Save a URL as a bookmark with optional tags and notes for later reference."
    parameters = {
        "url": "The URL to bookmark",
        "title": "Title for the bookmark",
        "tags": "Comma-separated tags (e.g., 'python,tutorial,ai')",
        "notes": "Optional notes about the bookmark",
    }
    
    def execute(self, url: str, title: str, tags: str = None, notes: str = None, **kwargs) -> Dict[str, Any]:
        try:
            tag_list = [t.strip() for t in tags.split(',')] if tags else []
            
            manager = BookmarkManager()
            bookmark = manager.add(url, title, tag_list, notes)
            
            return {
                "success": True,
                "message": f"Bookmark saved: {title}",
                "bookmark": bookmark,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class BookmarkSearchTool(Tool):
    """Search bookmarks."""
    
    name = "bookmark_search"
    description = "Search saved bookmarks by text or tags."
    parameters = {
        "query": "Search text (searches title, URL, notes)",
        "tags": "Filter by tags (comma-separated)",
    }
    
    def execute(self, query: str = None, tags: str = None, **kwargs) -> Dict[str, Any]:
        try:
            tag_list = [t.strip() for t in tags.split(',')] if tags else None
            
            manager = BookmarkManager()
            results = manager.search(query, tag_list)
            
            return {
                "success": True,
                "count": len(results),
                "bookmarks": results,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class BookmarkListTool(Tool):
    """List all bookmarks."""
    
    name = "bookmark_list"
    description = "List all saved bookmarks."
    parameters = {}
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        try:
            manager = BookmarkManager()
            bookmarks = manager.list_all()
            
            return {
                "success": True,
                "count": len(bookmarks),
                "bookmarks": bookmarks,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class BookmarkDeleteTool(Tool):
    """Delete a bookmark."""
    
    name = "bookmark_delete"
    description = "Delete a bookmark by its ID."
    parameters = {
        "bookmark_id": "The ID of the bookmark to delete",
    }
    
    def execute(self, bookmark_id: int, **kwargs) -> Dict[str, Any]:
        try:
            manager = BookmarkManager()
            manager.delete(int(bookmark_id))
            return {"success": True, "message": f"Deleted bookmark {bookmark_id}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# NOTES TOOLS
# ============================================================================

class NoteManager:
    """Manages persistent notes."""
    
    def __init__(self):
        self.index_file = NOTES_DIR / "index.json"
        self.index: Dict[str, Dict] = {}
        self._load_index()
    
    def _load_index(self):
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            except:
                self.index = {}
    
    def _save_index(self):
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _get_note_path(self, name: str) -> Path:
        safe_name = re.sub(r'[^\w\-]', '_', name)
        return NOTES_DIR / f"{safe_name}.md"
    
    def save(self, name: str, content: str, tags: List[str] = None) -> Dict:
        """Save or update a note."""
        note_path = self._get_note_path(name)
        
        with open(note_path, 'w') as f:
            f.write(content)
        
        now = datetime.now().isoformat()
        
        if name in self.index:
            self.index[name]["updated"] = now
            self.index[name]["tags"] = tags or self.index[name].get("tags", [])
        else:
            self.index[name] = {
                "name": name,
                "path": str(note_path),
                "created": now,
                "updated": now,
                "tags": tags or [],
            }
        
        self._save_index()
        return self.index[name]
    
    def get(self, name: str) -> Optional[Dict]:
        """Get a note by name."""
        if name not in self.index:
            return None
        
        note_path = self._get_note_path(name)
        if not note_path.exists():
            return None
        
        with open(note_path, 'r') as f:
            content = f.read()
        
        return {
            **self.index[name],
            "content": content,
        }
    
    def search(self, query: str = None, tags: List[str] = None) -> List[Dict]:
        """Search notes."""
        results = []
        query_lower = query.lower() if query else ""
        
        for name, meta in self.index.items():
            # Tag filter
            if tags:
                if not any(t in meta.get('tags', []) for t in tags):
                    continue
            
            # Load content for search
            note_path = self._get_note_path(name)
            content = ""
            if note_path.exists():
                with open(note_path, 'r') as f:
                    content = f.read()
            
            # Text search
            if query_lower:
                searchable = f"{name} {content} {' '.join(meta.get('tags', []))}".lower()
                if query_lower not in searchable:
                    continue
            
            results.append({
                **meta,
                "preview": content[:200] + "..." if len(content) > 200 else content,
            })
        
        return results
    
    def delete(self, name: str):
        """Delete a note."""
        note_path = self._get_note_path(name)
        if note_path.exists():
            note_path.unlink()
        
        if name in self.index:
            del self.index[name]
            self._save_index()
    
    def list_all(self) -> List[Dict]:
        """List all notes."""
        return list(self.index.values())


class NoteSaveTool(Tool):
    """Save a note."""
    
    name = "note_save"
    description = "Save a persistent note that I can reference later. Great for remembering things across conversations."
    parameters = {
        "name": "Name/title for the note",
        "content": "The note content (supports markdown)",
        "tags": "Optional tags (comma-separated)",
    }
    
    def execute(self, name: str, content: str, tags: str = None, **kwargs) -> Dict[str, Any]:
        try:
            tag_list = [t.strip() for t in tags.split(',')] if tags else []
            
            manager = NoteManager()
            note = manager.save(name, content, tag_list)
            
            return {
                "success": True,
                "message": f"Note saved: {name}",
                "note": note,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class NoteGetTool(Tool):
    """Get a note by name."""
    
    name = "note_get"
    description = "Retrieve a saved note by its name."
    parameters = {
        "name": "The name of the note to retrieve",
    }
    
    def execute(self, name: str, **kwargs) -> Dict[str, Any]:
        try:
            manager = NoteManager()
            note = manager.get(name)
            
            if note:
                return {"success": True, "note": note}
            else:
                return {"success": False, "error": f"Note not found: {name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class NoteSearchTool(Tool):
    """Search notes."""
    
    name = "note_search"
    description = "Search saved notes by text or tags."
    parameters = {
        "query": "Search text (searches name and content)",
        "tags": "Filter by tags (comma-separated)",
    }
    
    def execute(self, query: str = None, tags: str = None, **kwargs) -> Dict[str, Any]:
        try:
            tag_list = [t.strip() for t in tags.split(',')] if tags else None
            
            manager = NoteManager()
            results = manager.search(query, tag_list)
            
            return {
                "success": True,
                "count": len(results),
                "notes": results,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class NoteListTool(Tool):
    """List all notes."""
    
    name = "note_list"
    description = "List all saved notes."
    parameters = {}
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        try:
            manager = NoteManager()
            notes = manager.list_all()
            
            return {
                "success": True,
                "count": len(notes),
                "notes": notes,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class NoteDeleteTool(Tool):
    """Delete a note."""
    
    name = "note_delete"
    description = "Delete a saved note by its name."
    parameters = {
        "name": "The name of the note to delete",
    }
    
    def execute(self, name: str, **kwargs) -> Dict[str, Any]:
        try:
            manager = NoteManager()
            manager.delete(name)
            return {"success": True, "message": f"Deleted note: {name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
