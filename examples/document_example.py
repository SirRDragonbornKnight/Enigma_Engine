"""
Document Tools Example for ForgeAI

This example shows how to read various document formats.
The AI can read PDFs, eBooks, Word docs, and more.

SUPPORTED FORMATS:
- .txt, .md - Plain text and Markdown
- .pdf - PDF documents
- .epub - eBooks
- .docx - Word documents
- .html - Web pages

USAGE:
    python examples/document_example.py
    
Or import in your own code:
    from examples.document_example import read_document, read_pdf
"""

from pathlib import Path
from typing import Dict, Any, Optional


# =============================================================================
# DOCUMENT READERS
# =============================================================================

class DocumentReader:
    """
    Read various document formats.
    
    Auto-detects format based on file extension.
    """
    
    SUPPORTED_EXTENSIONS = {
        '.txt': 'text',
        '.md': 'markdown',
        '.pdf': 'pdf',
        '.epub': 'epub',
        '.docx': 'docx',
        '.doc': 'doc',
        '.html': 'html',
        '.htm': 'html',
        '.rtf': 'rtf',
    }
    
    def __init__(self):
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check which optional libraries are available."""
        self.has_pypdf = False
        self.has_epub = False
        self.has_docx = False
        
        try:
            import PyPDF2
            self.has_pypdf = True
        except ImportError:
            pass
        
        try:
            import ebooklib
            self.has_epub = True
        except ImportError:
            pass
        
        try:
            import docx
            self.has_docx = True
        except ImportError:
            pass
    
    def read(self, path: str, max_chars: int = 50000, **kwargs) -> Dict[str, Any]:
        """
        Read a document file.
        
        Args:
            path: Path to document
            max_chars: Maximum characters to return
            **kwargs: Format-specific options
        
        Returns:
            Dict with: success, content, format, etc.
        """
        path = Path(path).expanduser().resolve()
        
        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        ext = path.suffix.lower()
        
        if ext not in self.SUPPORTED_EXTENSIONS:
            # Try as plain text
            ext = '.txt'
        
        format_type = self.SUPPORTED_EXTENSIONS.get(ext, 'text')
        
        try:
            if format_type in ('text', 'markdown'):
                content = self._read_text(path)
            elif format_type == 'pdf':
                content = self._read_pdf(path, **kwargs)
            elif format_type == 'epub':
                content = self._read_epub(path)
            elif format_type == 'docx':
                content = self._read_docx(path)
            elif format_type == 'html':
                content = self._read_html(path)
            else:
                content = self._read_text(path)
            
            # Truncate if needed
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n... [truncated]"
            
            return {
                "success": True,
                "path": str(path),
                "format": format_type,
                "content": content,
                "content_length": len(content),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "path": str(path)}
    
    def _read_text(self, path: Path) -> str:
        """Read plain text file."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _read_pdf(self, path: Path, start_page: int = 1, end_page: int = None) -> str:
        """Read PDF file."""
        if not self.has_pypdf:
            return "[PDF reading requires PyPDF2: pip install PyPDF2]"
        
        import PyPDF2
        
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            
            # Adjust page range
            start = max(1, start_page) - 1  # Convert to 0-indexed
            end = min(end_page or num_pages, num_pages)
            
            text = []
            for i in range(start, end):
                page = reader.pages[i]
                page_text = page.extract_text() or ""
                text.append(f"--- Page {i + 1} ---\n{page_text}")
            
            return "\n\n".join(text)
    
    def _read_epub(self, path: Path) -> str:
        """Read EPUB ebook."""
        if not self.has_epub:
            return "[EPUB reading requires ebooklib: pip install ebooklib]"
        
        from ebooklib import epub
        from bs4 import BeautifulSoup
        
        book = epub.read_epub(str(path))
        
        text = []
        for item in book.get_items():
            if item.get_type() == 9:  # XHTML content
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text.append(soup.get_text())
        
        return "\n\n".join(text)
    
    def _read_docx(self, path: Path) -> str:
        """Read Word document."""
        if not self.has_docx:
            return "[DOCX reading requires python-docx: pip install python-docx]"
        
        import docx
        
        doc = docx.Document(str(path))
        
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        
        return "\n\n".join(text)
    
    def _read_html(self, path: Path) -> str:
        """Read HTML file."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            html = f.read()
        
        # Try BeautifulSoup first
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            return soup.get_text()
        except ImportError:
            pass
        
        # Fallback: simple regex-based extraction
        import re
        
        # Remove script and style blocks
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


# =============================================================================
# PDF UTILITIES
# =============================================================================

class PDFTools:
    """
    Additional PDF utilities.
    """
    
    def __init__(self):
        self._check_pypdf()
    
    def _check_pypdf(self):
        try:
            import PyPDF2
            self.available = True
        except ImportError:
            self.available = False
    
    def get_info(self, path: str) -> Dict[str, Any]:
        """Get PDF metadata and info."""
        if not self.available:
            return {"error": "PyPDF2 not installed"}
        
        import PyPDF2
        
        try:
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                info = {
                    "pages": len(reader.pages),
                    "encrypted": reader.is_encrypted,
                }
                
                if reader.metadata:
                    info["title"] = reader.metadata.get('/Title', '')
                    info["author"] = reader.metadata.get('/Author', '')
                    info["subject"] = reader.metadata.get('/Subject', '')
                    info["creator"] = reader.metadata.get('/Creator', '')
                
                return info
                
        except Exception as e:
            return {"error": str(e)}
    
    def extract_pages(self, path: str, pages: list, output_path: str) -> bool:
        """Extract specific pages to new PDF."""
        if not self.available:
            return False
        
        import PyPDF2
        
        try:
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                writer = PyPDF2.PdfWriter()
                
                for page_num in pages:
                    if 0 <= page_num - 1 < len(reader.pages):
                        writer.add_page(reader.pages[page_num - 1])
                
                with open(output_path, 'wb') as out:
                    writer.write(out)
            
            return True
            
        except Exception as e:
            print(f"[PDF] Extract failed: {e}")
            return False
    
    def search_text(self, path: str, query: str) -> list:
        """Search for text in PDF, return page numbers."""
        if not self.available:
            return []
        
        import PyPDF2
        
        results = []
        query_lower = query.lower()
        
        try:
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if query_lower in text.lower():
                        results.append({
                            "page": i + 1,
                            "preview": self._get_context(text, query)
                        })
            
            return results
            
        except Exception as e:
            print(f"[PDF] Search failed: {e}")
            return []
    
    def _get_context(self, text: str, query: str, context_chars: int = 100) -> str:
        """Get text around the query."""
        idx = text.lower().find(query.lower())
        if idx == -1:
            return ""
        
        start = max(0, idx - context_chars)
        end = min(len(text), idx + len(query) + context_chars)
        
        return "..." + text[start:end] + "..."


# =============================================================================
# TEXT UTILITIES
# =============================================================================

class TextTools:
    """
    Text processing utilities.
    """
    
    @staticmethod
    def word_count(text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    @staticmethod
    def line_count(text: str) -> int:
        """Count lines in text."""
        return len(text.splitlines())
    
    @staticmethod
    def summarize(text: str, max_sentences: int = 5) -> str:
        """
        Simple extractive summarization.
        Returns first N sentences.
        """
        import re
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Return first N
        summary = ' '.join(sentences[:max_sentences])
        
        if len(sentences) > max_sentences:
            summary += "..."
        
        return summary
    
    @staticmethod
    def extract_urls(text: str) -> list:
        """Extract URLs from text."""
        import re
        pattern = r'https?://[^\s<>"\'{}|\\^`\[\]]+'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_emails(text: str) -> list:
        """Extract email addresses from text."""
        import re
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return re.findall(pattern, text)
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list:
        """
        Split text into overlapping chunks.
        Useful for processing with LLMs.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def read_document(path: str, **kwargs) -> Dict[str, Any]:
    """
    Quick document reading.
    
    Args:
        path: Path to document
        **kwargs: Reader options
    
    Returns:
        Dict with content and metadata
    """
    reader = DocumentReader()
    return reader.read(path, **kwargs)


def read_pdf(path: str, start_page: int = 1, end_page: int = None) -> str:
    """
    Quick PDF reading.
    
    Args:
        path: Path to PDF
        start_page: First page (1-indexed)
        end_page: Last page (None for all)
    
    Returns:
        Extracted text
    """
    reader = DocumentReader()
    result = reader.read(path, start_page=start_page, end_page=end_page)
    return result.get('content', result.get('error', ''))


def search_pdf(path: str, query: str) -> list:
    """
    Search for text in PDF.
    
    Returns:
        List of pages containing the query
    """
    tools = PDFTools()
    return tools.search_text(path, query)


def summarize_document(path: str, max_sentences: int = 10) -> str:
    """
    Read and summarize a document.
    """
    result = read_document(path)
    if not result.get('success'):
        return result.get('error', 'Failed to read document')
    
    return TextTools.summarize(result['content'], max_sentences)


# =============================================================================
# MAIN - Run this file directly to test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ForgeAI Document Tools Example")
    print("=" * 60)
    
    # Check available libraries
    print("\n[1] Checking dependencies...")
    reader = DocumentReader()
    
    print(f"  PyPDF2 (PDF):     {'Yes' if reader.has_pypdf else 'No - pip install PyPDF2'}")
    print(f"  ebooklib (EPUB):  {'Yes' if reader.has_epub else 'No - pip install ebooklib'}")
    print(f"  python-docx:      {'Yes' if reader.has_docx else 'No - pip install python-docx'}")
    
    # Test with a simple text file
    print("\n[2] Testing text file reading...")
    
    # Create a test file
    test_dir = Path.home() / ".forge_ai" / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "test_document.txt"
    
    test_content = """
    This is a test document for ForgeAI.
    
    It contains multiple paragraphs and sentences.
    The document tools can read various formats including
    PDF, EPUB, DOCX, and plain text files.
    
    Contact: test@example.com
    Website: https://example.com/docs
    
    This is the final paragraph of the test document.
    """
    
    test_file.write_text(test_content)
    
    result = read_document(str(test_file))
    print(f"  Read: {result['path']}")
    print(f"  Format: {result['format']}")
    print(f"  Length: {result['content_length']} chars")
    
    # Test text utilities
    print("\n[3] Testing text utilities...")
    text = result['content']
    
    print(f"  Word count: {TextTools.word_count(text)}")
    print(f"  Line count: {TextTools.line_count(text)}")
    print(f"  URLs found: {TextTools.extract_urls(text)}")
    print(f"  Emails found: {TextTools.extract_emails(text)}")
    
    # Test summarization
    print("\n[4] Testing summarization...")
    summary = TextTools.summarize(text, max_sentences=2)
    print(f"  Summary: {summary}")
    
    # Test chunking
    print("\n[5] Testing text chunking...")
    chunks = TextTools.chunk_text(text, chunk_size=100, overlap=20)
    print(f"  Split into {len(chunks)} chunks")
    
    # Test PDF tools (if available)
    print("\n[6] PDF tools...")
    pdf_tools = PDFTools()
    if pdf_tools.available:
        print("  PDF tools available!")
        print("  Usage:")
        print("    info = pdf_tools.get_info('document.pdf')")
        print("    results = pdf_tools.search_text('document.pdf', 'keyword')")
    else:
        print("  Install PyPDF2 for PDF support: pip install PyPDF2")
    
    # Cleanup
    test_file.unlink()
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nSupported formats:")
    print("  - .txt, .md   - Always available")
    print("  - .html       - Always available")
    print("  - .pdf        - pip install PyPDF2")
    print("  - .epub       - pip install ebooklib beautifulsoup4")
    print("  - .docx       - pip install python-docx")
