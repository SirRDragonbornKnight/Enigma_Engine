"""
================================================================================
Code Snippets Manager - Save, organize, and reuse code snippets.
================================================================================

Code snippet features:
- Save snippets with metadata
- Organize by language, tags, categories
- Search and filter snippets
- Template variables in snippets
- Import/export collections
- Syntax detection

USAGE:
    from enigma_engine.utils.code_snippets import SnippetManager, get_snippet_manager
    
    manager = get_snippet_manager()
    
    # Save a snippet
    snippet = manager.add(
        title="Python Hello World",
        code="print('Hello, World!')",
        language="python",
        tags=["basic", "print"]
    )
    
    # Search snippets
    results = manager.search("hello")
    
    # Get snippet by ID
    snippet = manager.get(snippet_id)
    
    # Use template snippet
    code = manager.apply_template(snippet_id, {"name": "World"})
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Language detection patterns
LANGUAGE_PATTERNS = {
    "python": [
        r'\bdef\s+\w+\s*\(',
        r'\bclass\s+\w+.*:',
        r'^import\s+\w+',
        r'^from\s+\w+\s+import',
        r'\bprint\s*\(',
        r'^\s*@\w+',
    ],
    "javascript": [
        r'\bfunction\s+\w+\s*\(',
        r'\bconst\s+\w+\s*=',
        r'\blet\s+\w+\s*=',
        r'\bvar\s+\w+\s*=',
        r'=>',
        r'\bconsole\.\w+\(',
        r'\bdocument\.\w+',
    ],
    "typescript": [
        r':\s*(string|number|boolean|any|void)',
        r'\binterface\s+\w+',
        r'\btype\s+\w+\s*=',
        r'<\w+>',
    ],
    "java": [
        r'\bpublic\s+(class|interface|enum)',
        r'\bprivate\s+\w+',
        r'\bstatic\s+void\s+main',
        r'System\.out\.print',
    ],
    "csharp": [
        r'\bnamespace\s+\w+',
        r'\bpublic\s+class\s+\w+',
        r'\bConsole\.Write',
        r'\bvar\s+\w+\s*=',
    ],
    "cpp": [
        r'#include\s*<\w+>',
        r'\bstd::',
        r'\bint\s+main\s*\(',
        r'\bcout\s*<<',
        r'\bvector<',
    ],
    "c": [
        r'#include\s*<\w+\.h>',
        r'\bprintf\s*\(',
        r'\bint\s+main\s*\(',
        r'\bmalloc\s*\(',
    ],
    "rust": [
        r'\bfn\s+\w+\s*\(',
        r'\blet\s+mut\s+',
        r'\bimpl\s+\w+',
        r'\bpub\s+fn',
        r'->',
    ],
    "go": [
        r'\bfunc\s+\w+\s*\(',
        r'\bpackage\s+\w+',
        r'\bfmt\.\w+',
        r':=',
    ],
    "ruby": [
        r'\bdef\s+\w+',
        r'\bend\s*$',
        r'\bputs\s+',
        r'\bclass\s+\w+\s*<',
    ],
    "php": [
        r'<\?php',
        r'\$\w+\s*=',
        r'\becho\s+',
        r'\bfunction\s+\w+\s*\(',
    ],
    "sql": [
        r'\bSELECT\s+',
        r'\bFROM\s+',
        r'\bWHERE\s+',
        r'\bINSERT\s+INTO',
        r'\bCREATE\s+TABLE',
    ],
    "bash": [
        r'^#!/bin/(ba)?sh',
        r'\becho\s+',
        r'\bif\s+\[\s+',
        r'\bfi\s*$',
    ],
    "html": [
        r'<!DOCTYPE\s+html',
        r'<html',
        r'<div',
        r'<\/\w+>',
    ],
    "css": [
        r'\.\w+\s*\{',
        r'#\w+\s*\{',
        r'\w+:\s*\w+;',
        r'@media',
    ],
    "json": [
        r'^\s*\{',
        r'^\s*\[',
        r'"\w+":\s*',
    ],
    "yaml": [
        r'^\w+:\s*$',
        r'^\s+-\s+\w+',
        r'^\s+\w+:\s+\w+',
    ],
    "markdown": [
        r'^#{1,6}\s+',
        r'^\s*[\*\-]\s+',
        r'\[.+\]\(.+\)',
    ],
}


@dataclass
class CodeSnippet:
    """A code snippet with metadata."""
    id: str
    title: str
    code: str
    language: str
    
    # Metadata
    description: str = ""
    tags: list[str] = field(default_factory=list)
    category: str = "general"
    
    # Tracking
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = ""
    use_count: int = 0
    last_used: str = ""
    
    # Template support
    is_template: bool = False
    variables: list[str] = field(default_factory=list)
    
    # User data
    favorite: bool = False
    
    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = self.created_at
        
        # Auto-detect template variables: ${variable} or {{variable}}
        if self.is_template and not self.variables:
            vars1 = set(re.findall(r'\$\{(\w+)\}', self.code))
            vars2 = set(re.findall(r'\{\{(\w+)\}\}', self.code))
            self.variables = list(vars1 | vars2)
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeSnippet:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @property
    def preview(self) -> str:
        """Get a preview of the code."""
        lines = self.code.split('\n')
        if len(lines) <= 3:
            return self.code
        return '\n'.join(lines[:3]) + f'\n... ({len(lines) - 3} more lines)'
    
    @property
    def line_count(self) -> int:
        """Get the number of lines."""
        return self.code.count('\n') + 1


def detect_language(code: str) -> str:
    """
    Detect the programming language of code.
    
    Args:
        code: Code to analyze
        
    Returns:
        Detected language name
    """
    scores: dict[str, int] = {}
    
    for language, patterns in LANGUAGE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                score += 1
        if score > 0:
            scores[language] = score
    
    if not scores:
        return "text"
    
    return max(scores.keys(), key=lambda k: scores[k])


class SnippetManager:
    """
    Manage code snippets with persistence.
    """
    
    def __init__(self, data_path: Path | None = None):
        """
        Initialize snippet manager.
        
        Args:
            data_path: Path to store snippets
        """
        self._data_path = data_path or Path("data/snippets")
        self._data_path.mkdir(parents=True, exist_ok=True)
        self._snippets_file = self._data_path / "snippets.json"
        
        self._snippets: dict[str, CodeSnippet] = {}
        self._load_snippets()
    
    def _load_snippets(self) -> None:
        """Load snippets from disk."""
        if self._snippets_file.exists():
            try:
                with open(self._snippets_file, encoding='utf-8') as f:
                    data = json.load(f)
                    for snippet_data in data.get("snippets", []):
                        snippet = CodeSnippet.from_dict(snippet_data)
                        self._snippets[snippet.id] = snippet
                logger.info(f"Loaded {len(self._snippets)} snippets")
            except Exception as e:
                logger.error(f"Failed to load snippets: {e}")
    
    def _save_snippets(self) -> None:
        """Save snippets to disk."""
        try:
            data = {
                "snippets": [s.to_dict() for s in self._snippets.values()],
                "last_updated": datetime.now().isoformat()
            }
            with open(self._snippets_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save snippets: {e}")
    
    def _generate_id(self, title: str, code: str) -> str:
        """Generate a unique ID for a snippet."""
        content = f"{title}:{code[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def add(
        self,
        title: str,
        code: str,
        language: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        category: str = "general",
        is_template: bool = False
    ) -> CodeSnippet:
        """
        Add a new snippet.
        
        Args:
            title: Snippet title
            code: The code
            language: Programming language (auto-detected if not provided)
            description: Optional description
            tags: Optional tags
            category: Category for organization
            is_template: Whether this is a template with variables
            
        Returns:
            The created snippet
        """
        snippet_id = self._generate_id(title, code)
        
        # Auto-detect language if not provided
        if not language:
            language = detect_language(code)
        
        snippet = CodeSnippet(
            id=snippet_id,
            title=title,
            code=code,
            language=language,
            description=description,
            tags=tags or [],
            category=category,
            is_template=is_template
        )
        
        self._snippets[snippet_id] = snippet
        self._save_snippets()
        
        return snippet
    
    def get(self, snippet_id: str) -> CodeSnippet | None:
        """Get a snippet by ID."""
        return self._snippets.get(snippet_id)
    
    def update(
        self,
        snippet_id: str,
        title: str | None = None,
        code: str | None = None,
        language: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        category: str | None = None
    ) -> CodeSnippet | None:
        """Update an existing snippet."""
        snippet = self._snippets.get(snippet_id)
        if not snippet:
            return None
        
        if title is not None:
            snippet.title = title
        if code is not None:
            snippet.code = code
        if language is not None:
            snippet.language = language
        if description is not None:
            snippet.description = description
        if tags is not None:
            snippet.tags = tags
        if category is not None:
            snippet.category = category
        
        snippet.updated_at = datetime.now().isoformat()
        self._save_snippets()
        
        return snippet
    
    def delete(self, snippet_id: str) -> bool:
        """Delete a snippet."""
        if snippet_id in self._snippets:
            del self._snippets[snippet_id]
            self._save_snippets()
            return True
        return False
    
    def use(self, snippet_id: str) -> str | None:
        """
        Mark a snippet as used and return its code.
        
        Args:
            snippet_id: Snippet ID
            
        Returns:
            The snippet code or None
        """
        snippet = self._snippets.get(snippet_id)
        if snippet:
            snippet.use_count += 1
            snippet.last_used = datetime.now().isoformat()
            self._save_snippets()
            return snippet.code
        return None
    
    def apply_template(
        self,
        snippet_id: str,
        variables: dict[str, str]
    ) -> str | None:
        """
        Apply variables to a template snippet.
        
        Args:
            snippet_id: Snippet ID
            variables: Variable values to substitute
            
        Returns:
            Code with variables substituted
        """
        snippet = self._snippets.get(snippet_id)
        if not snippet:
            return None
        
        code = snippet.code
        
        # Replace ${variable} syntax
        for var, value in variables.items():
            code = code.replace(f'${{{var}}}', value)
        
        # Replace {{variable}} syntax
        for var, value in variables.items():
            code = code.replace(f'{{{{{var}}}}}', value)
        
        snippet.use_count += 1
        snippet.last_used = datetime.now().isoformat()
        self._save_snippets()
        
        return code
    
    def list_all(
        self,
        language: str | None = None,
        category: str | None = None,
        tag: str | None = None,
        favorites_only: bool = False
    ) -> list[CodeSnippet]:
        """
        List snippets with optional filtering.
        
        Args:
            language: Filter by language
            category: Filter by category
            tag: Filter by tag
            favorites_only: Only return favorites
            
        Returns:
            List of matching snippets
        """
        results = list(self._snippets.values())
        
        if language:
            results = [s for s in results if s.language.lower() == language.lower()]
        
        if category:
            results = [s for s in results if s.category.lower() == category.lower()]
        
        if tag:
            results = [s for s in results if tag.lower() in [t.lower() for t in s.tags]]
        
        if favorites_only:
            results = [s for s in results if s.favorite]
        
        return sorted(results, key=lambda s: s.updated_at, reverse=True)
    
    def search(
        self,
        query: str,
        limit: int = 20
    ) -> list[CodeSnippet]:
        """
        Search snippets by title, code, or description.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            Matching snippets
        """
        query_lower = query.lower()
        results = []
        
        for snippet in self._snippets.values():
            searchable = f"{snippet.title} {snippet.code} {snippet.description} {' '.join(snippet.tags)}"
            if query_lower in searchable.lower():
                results.append(snippet)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_languages(self) -> list[str]:
        """Get all unique languages."""
        return sorted({s.language for s in self._snippets.values()})
    
    def get_categories(self) -> list[str]:
        """Get all unique categories."""
        return sorted({s.category for s in self._snippets.values()})
    
    def get_tags(self) -> list[str]:
        """Get all unique tags."""
        all_tags: set[str] = set()
        for snippet in self._snippets.values():
            all_tags.update(snippet.tags)
        return sorted(all_tags)
    
    def favorite(self, snippet_id: str, is_favorite: bool = True) -> bool:
        """Toggle favorite status."""
        snippet = self._snippets.get(snippet_id)
        if snippet:
            snippet.favorite = is_favorite
            self._save_snippets()
            return True
        return False
    
    def add_tag(self, snippet_id: str, tag: str) -> bool:
        """Add a tag to a snippet."""
        snippet = self._snippets.get(snippet_id)
        if snippet and tag not in snippet.tags:
            snippet.tags.append(tag)
            self._save_snippets()
            return True
        return False
    
    def remove_tag(self, snippet_id: str, tag: str) -> bool:
        """Remove a tag from a snippet."""
        snippet = self._snippets.get(snippet_id)
        if snippet and tag in snippet.tags:
            snippet.tags.remove(tag)
            self._save_snippets()
            return True
        return False
    
    def get_most_used(self, limit: int = 10) -> list[CodeSnippet]:
        """Get most frequently used snippets."""
        return sorted(
            self._snippets.values(),
            key=lambda s: s.use_count,
            reverse=True
        )[:limit]
    
    def get_recent(self, limit: int = 10) -> list[CodeSnippet]:
        """Get recently used snippets."""
        used = [s for s in self._snippets.values() if s.last_used]
        return sorted(used, key=lambda s: s.last_used, reverse=True)[:limit]
    
    def export_collection(
        self,
        output_path: Path,
        snippet_ids: list[str] | None = None
    ) -> int:
        """
        Export snippets to a JSON file.
        
        Args:
            output_path: Output file path
            snippet_ids: Specific IDs to export (all if None)
            
        Returns:
            Number of exported snippets
        """
        if snippet_ids:
            snippets = [s for s in self._snippets.values() if s.id in snippet_ids]
        else:
            snippets = list(self._snippets.values())
        
        data = {
            "snippets": [s.to_dict() for s in snippets],
            "exported_at": datetime.now().isoformat(),
            "count": len(snippets)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return len(snippets)
    
    def import_collection(
        self,
        input_path: Path,
        overwrite: bool = False
    ) -> int:
        """
        Import snippets from a JSON file.
        
        Args:
            input_path: Input file path
            overwrite: Overwrite existing snippets with same ID
            
        Returns:
            Number of imported snippets
        """
        with open(input_path, encoding='utf-8') as f:
            data = json.load(f)
        
        imported = 0
        for snippet_data in data.get("snippets", []):
            snippet = CodeSnippet.from_dict(snippet_data)
            
            if snippet.id in self._snippets and not overwrite:
                continue
            
            self._snippets[snippet.id] = snippet
            imported += 1
        
        self._save_snippets()
        return imported
    
    def get_stats(self) -> dict[str, Any]:
        """Get snippet statistics."""
        total = len(self._snippets)
        
        language_counts: dict[str, int] = {}
        category_counts: dict[str, int] = {}
        total_uses = 0
        
        for snippet in self._snippets.values():
            language_counts[snippet.language] = language_counts.get(snippet.language, 0) + 1
            category_counts[snippet.category] = category_counts.get(snippet.category, 0) + 1
            total_uses += snippet.use_count
        
        return {
            "total_snippets": total,
            "total_uses": total_uses,
            "favorites": sum(1 for s in self._snippets.values() if s.favorite),
            "templates": sum(1 for s in self._snippets.values() if s.is_template),
            "languages": language_counts,
            "categories": category_counts,
            "unique_tags": len(self.get_tags())
        }


# Singleton instance
_manager_instance: SnippetManager | None = None


def get_snippet_manager(data_path: Path | None = None) -> SnippetManager:
    """Get or create the singleton snippet manager."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = SnippetManager(data_path)
    return _manager_instance


# Convenience functions
def add_snippet(
    title: str,
    code: str,
    language: str | None = None,
    tags: list[str] | None = None
) -> CodeSnippet:
    """Add a new snippet."""
    return get_snippet_manager().add(title, code, language, tags=tags)


def search_snippets(query: str) -> list[CodeSnippet]:
    """Search snippets."""
    return get_snippet_manager().search(query)


def get_snippet(snippet_id: str) -> CodeSnippet | None:
    """Get a snippet by ID."""
    return get_snippet_manager().get(snippet_id)
