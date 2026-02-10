"""
Document Tools - Read books, PDFs, and other documents.

Tools:
  - read_document: Read various document formats (txt, pdf, epub, etc.)
  - extract_text: Extract plain text from any document
"""

from pathlib import Path
from typing import Any

from .tool_registry import Tool, RichParameter


class ReadDocumentTool(Tool):
    """
    Read various document formats.
    
    Supports:
      - .txt: Plain text
      - .md: Markdown
      - .pdf: PDF (requires PyPDF2 or pdfplumber)
      - .epub: EPUB books (requires ebooklib)
      - .docx: Word documents (requires python-docx)
      - .html: HTML files
    """
    
    name = "read_document"
    description = "Read a document file (txt, pdf, epub, docx, html, md). Good for reading books or papers."
    parameters = {
        "path": "Path to the document file",
        "max_chars": "Maximum characters to return (default: 10000)",
        "start_page": "For PDFs: starting page number (default: 1)",
        "end_page": "For PDFs: ending page number (default: all)",
    }
    category = "document"
    rich_parameters = [
        RichParameter(
            name="path",
            type="string",
            description="Path to the document file",
            required=True,
        ),
        RichParameter(
            name="max_chars",
            type="integer",
            description="Maximum characters to return",
            required=False,
            default=10000,
            min_value=100,
            max_value=1000000,
        ),
        RichParameter(
            name="start_page",
            type="integer",
            description="For PDFs: starting page number",
            required=False,
            default=1,
            min_value=1,
        ),
        RichParameter(
            name="end_page",
            type="integer",
            description="For PDFs: ending page number (all if not specified)",
            required=False,
        ),
    ]
    examples = [
        "read_document(path='book.pdf')",
        "read_document(path='paper.pdf', start_page=5, end_page=10)",
        "read_document(path='novel.epub', max_chars=50000)",
    ]
    
    def execute(self, path: str, max_chars: int = 10000, 
                start_page: int = 1, end_page: int = None, **kwargs) -> dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            
            if not path.exists():
                return {"success": False, "error": f"File not found: {path}"}
            
            ext = path.suffix.lower()
            
            if ext == ".txt" or ext == ".md":
                content = self._read_text(path)
            elif ext == ".pdf":
                content = self._read_pdf(path, start_page, end_page)
            elif ext == ".epub":
                content = self._read_epub(path)
            elif ext == ".docx":
                content = self._read_docx(path)
            elif ext in [".html", ".htm"]:
                content = self._read_html(path)
            else:
                # Try as plain text
                content = self._read_text(path)
            
            # Truncate if needed
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n... [truncated]"
            
            return {
                "success": True,
                "path": str(path),
                "format": ext,
                "content_length": len(content),
                "content": content
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _read_text(self, path: Path) -> str:
        """Read plain text file."""
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    
    def _read_pdf(self, path: Path, start_page: int, end_page: int) -> str:
        """Read PDF file."""
        try:
            import PyPDF2
            
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                
                end = end_page if end_page else num_pages
                start = max(1, start_page) - 1  # Convert to 0-indexed
                end = min(end, num_pages)
                
                text = []
                for i in range(start, end):
                    page = reader.pages[i]
                    text.append(f"--- Page {i+1} ---\n")
                    text.append(page.extract_text() or "")
                
                return "\n".join(text)
                
        except ImportError:
            # Try pdfplumber as alternative
            try:
                import pdfplumber
                
                with pdfplumber.open(path) as pdf:
                    text = []
                    end = end_page if end_page else len(pdf.pages)
                    
                    for i, page in enumerate(pdf.pages):
                        if i + 1 < start_page:
                            continue
                        if i + 1 > end:
                            break
                        text.append(f"--- Page {i+1} ---\n")
                        text.append(page.extract_text() or "")
                    
                    return "\n".join(text)
                    
            except ImportError:
                return "[PDF reading requires PyPDF2 or pdfplumber. Install with: pip install PyPDF2]"
    
    def _read_epub(self, path: Path) -> str:
        """Read EPUB ebook."""
        try:
            import ebooklib
            from bs4 import BeautifulSoup
            from ebooklib import epub
            
            book = epub.read_epub(str(path))
            
            text = []
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text.append(soup.get_text())
            
            return "\n\n".join(text)
            
        except ImportError:
            return "[EPUB reading requires ebooklib and beautifulsoup4. Install with: pip install ebooklib beautifulsoup4]"
    
    def _read_docx(self, path: Path) -> str:
        """Read Word document."""
        try:
            import docx
            
            doc = docx.Document(str(path))
            text = []
            
            for para in doc.paragraphs:
                text.append(para.text)
            
            return "\n".join(text)
            
        except ImportError:
            return "[DOCX reading requires python-docx. Install with: pip install python-docx]"
    
    def _read_html(self, path: Path) -> str:
        """Read HTML file and extract text."""
        import re
        
        with open(path, encoding="utf-8", errors="ignore") as f:
            html = f.read()
        
        # Remove script and style
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove tags
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class ExtractTextTool(Tool):
    """
    Extract plain text from any file.
    More aggressive text extraction than read_document.
    """
    
    name = "extract_text"
    description = "Extract all plain text from a file, removing formatting. Works on most file types."
    parameters = {
        "path": "Path to the file",
        "max_chars": "Maximum characters to return (default: 10000)",
    }
    category = "document"
    rich_parameters = [
        RichParameter(
            name="path",
            type="string",
            description="Path to the file to extract text from",
            required=True,
        ),
        RichParameter(
            name="max_chars",
            type="integer",
            description="Maximum characters to return",
            required=False,
            default=10000,
            min_value=100,
            max_value=1000000,
        ),
    ]
    examples = [
        "extract_text(path='document.pdf')",
        "extract_text(path='data.bin', max_chars=5000)",
    ]
    
    def execute(self, path: str, max_chars: int = 10000, **kwargs) -> dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            
            if not path.exists():
                return {"success": False, "error": f"File not found: {path}"}
            
            # Try to read as text
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError):
                # Try binary read and decode
                with open(path, "rb") as f:
                    raw = f.read()
                    content = raw.decode("utf-8", errors="ignore")
            
            # Clean up
            import re

            # Remove null bytes and control characters
            content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
            
            # Truncate
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n... [truncated]"
            
            return {
                "success": True,
                "path": str(path),
                "content_length": len(content),
                "content": content
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import json

    # Test with a text file
    tool = ReadDocumentTool()
    result = tool.execute("README.md", max_chars=500)
    print(json.dumps(result, indent=2))
