"""
================================================================================
Text Diff Utility - Compare and visualize text differences.
================================================================================

Text comparison features:
- Line-by-line diff
- Word-level diff
- Character-level diff
- Unified diff format
- Side-by-side comparison
- HTML output for visualization

USAGE:
    from enigma_engine.utils.text_diff import TextDiff, get_text_diff
    
    diff = get_text_diff()
    
    # Compare two texts
    result = diff.compare(old_text, new_text)
    print(result.unified)
    
    # Get HTML output for display
    html = diff.to_html(old_text, new_text)
    
    # Compare code with syntax highlighting
    code_diff = diff.compare_code(old_code, new_code, language="python")
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import Enum


class DiffType(str, Enum):
    """Type of diff output."""
    UNIFIED = "unified"
    CONTEXT = "context"
    NDIFF = "ndiff"
    HTML = "html"
    SIDE_BY_SIDE = "side_by_side"


class ChangeType(str, Enum):
    """Type of change in a diff."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class DiffLine:
    """A single line in a diff."""
    line_num_old: int | None
    line_num_new: int | None
    change_type: ChangeType
    content: str
    old_content: str = ""


@dataclass
class DiffChunk:
    """A chunk of changes in a diff."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[DiffLine] = field(default_factory=list)


@dataclass
class DiffResult:
    """Result of a text diff operation."""
    old_text: str
    new_text: str
    chunks: list[DiffChunk]
    
    # Statistics
    additions: int = 0
    deletions: int = 0
    modifications: int = 0
    unchanged: int = 0
    
    # Pre-formatted outputs
    unified: str = ""
    html: str = ""
    
    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return self.additions > 0 or self.deletions > 0 or self.modifications > 0
    
    @property
    def similarity(self) -> float:
        """Get similarity ratio (0.0 to 1.0)."""
        if not self.old_text and not self.new_text:
            return 1.0
        matcher = difflib.SequenceMatcher(None, self.old_text, self.new_text)
        return matcher.ratio()
    
    @property
    def summary(self) -> str:
        """Get a summary of changes."""
        parts = []
        if self.additions:
            parts.append(f"+{self.additions}")
        if self.deletions:
            parts.append(f"-{self.deletions}")
        if self.modifications:
            parts.append(f"~{self.modifications}")
        return ", ".join(parts) if parts else "No changes"


class TextDiff:
    """
    Text comparison and diff generation.
    """
    
    def __init__(self):
        """Initialize the text diff utility."""
    
    def compare(
        self,
        old_text: str,
        new_text: str,
        context_lines: int = 3
    ) -> DiffResult:
        """
        Compare two texts and generate a diff.
        
        Args:
            old_text: Original text
            new_text: New text
            context_lines: Number of context lines to include
            
        Returns:
            DiffResult with all diff information
        """
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        
        # Generate unified diff
        unified = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile='original',
            tofile='modified',
            n=context_lines
        ))
        
        # Parse into chunks
        chunks = self._parse_unified_diff(unified, old_lines, new_lines)
        
        # Count changes
        additions = 0
        deletions = 0
        modifications = 0
        unchanged = 0
        
        for chunk in chunks:
            for line in chunk.lines:
                if line.change_type == ChangeType.ADDED:
                    additions += 1
                elif line.change_type == ChangeType.REMOVED:
                    deletions += 1
                elif line.change_type == ChangeType.MODIFIED:
                    modifications += 1
                else:
                    unchanged += 1
        
        result = DiffResult(
            old_text=old_text,
            new_text=new_text,
            chunks=chunks,
            additions=additions,
            deletions=deletions,
            modifications=modifications,
            unchanged=unchanged,
            unified=''.join(unified),
            html=self._generate_html(chunks, old_text, new_text)
        )
        
        return result
    
    def _parse_unified_diff(
        self,
        diff_lines: list[str],
        old_lines: list[str],
        new_lines: list[str]
    ) -> list[DiffChunk]:
        """Parse unified diff output into chunks."""
        chunks = []
        current_chunk = None
        old_line_num = 0
        new_line_num = 0
        
        for line in diff_lines:
            if line.startswith('@@'):
                # Parse chunk header: @@ -start,count +start,count @@
                match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2)) if match.group(2) else 1
                    new_start = int(match.group(3))
                    new_count = int(match.group(4)) if match.group(4) else 1
                    
                    current_chunk = DiffChunk(
                        old_start=old_start,
                        old_count=old_count,
                        new_start=new_start,
                        new_count=new_count
                    )
                    chunks.append(current_chunk)
                    old_line_num = old_start
                    new_line_num = new_start
            
            elif current_chunk is not None:
                if line.startswith('---') or line.startswith('+++'):
                    continue
                
                content = line[1:] if len(line) > 1 else ""
                
                if line.startswith('+'):
                    current_chunk.lines.append(DiffLine(
                        line_num_old=None,
                        line_num_new=new_line_num,
                        change_type=ChangeType.ADDED,
                        content=content
                    ))
                    new_line_num += 1
                
                elif line.startswith('-'):
                    current_chunk.lines.append(DiffLine(
                        line_num_old=old_line_num,
                        line_num_new=None,
                        change_type=ChangeType.REMOVED,
                        content=content
                    ))
                    old_line_num += 1
                
                elif line.startswith(' '):
                    current_chunk.lines.append(DiffLine(
                        line_num_old=old_line_num,
                        line_num_new=new_line_num,
                        change_type=ChangeType.UNCHANGED,
                        content=content
                    ))
                    old_line_num += 1
                    new_line_num += 1
        
        return chunks
    
    def _generate_html(
        self,
        chunks: list[DiffChunk],
        old_text: str,
        new_text: str
    ) -> str:
        """Generate HTML visualization of diff."""
        html_parts = ["""
<style>
.diff-container { font-family: monospace; font-size: 13px; }
.diff-header { background: #f1f1f1; padding: 8px; border-bottom: 1px solid #ddd; }
.diff-chunk { margin: 10px 0; }
.diff-line { padding: 2px 8px; white-space: pre-wrap; }
.diff-added { background: #e6ffec; color: #22863a; }
.diff-removed { background: #ffebe9; color: #cb2431; }
.diff-unchanged { background: #f6f8fa; color: #24292e; }
.diff-line-num { color: #6e7781; width: 50px; display: inline-block; text-align: right; padding-right: 10px; }
.diff-stats { padding: 8px; background: #f6f8fa; border-top: 1px solid #ddd; }
.diff-stats-add { color: #22863a; }
.diff-stats-del { color: #cb2431; }
</style>
<div class="diff-container">
"""]
        
        if not chunks:
            html_parts.append('<div class="diff-header">No changes</div>')
        else:
            for chunk in chunks:
                html_parts.append(f'<div class="diff-chunk">')
                html_parts.append(f'<div class="diff-header">@@ -{chunk.old_start},{chunk.old_count} +{chunk.new_start},{chunk.new_count} @@</div>')
                
                for line in chunk.lines:
                    css_class = {
                        ChangeType.ADDED: "diff-added",
                        ChangeType.REMOVED: "diff-removed",
                        ChangeType.UNCHANGED: "diff-unchanged",
                        ChangeType.MODIFIED: "diff-added"
                    }.get(line.change_type, "diff-unchanged")
                    
                    prefix = {
                        ChangeType.ADDED: "+",
                        ChangeType.REMOVED: "-",
                        ChangeType.UNCHANGED: " ",
                        ChangeType.MODIFIED: "~"
                    }.get(line.change_type, " ")
                    
                    old_num = str(line.line_num_old) if line.line_num_old else ""
                    new_num = str(line.line_num_new) if line.line_num_new else ""
                    
                    escaped_content = (
                        line.content
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    
                    html_parts.append(
                        f'<div class="diff-line {css_class}">'
                        f'<span class="diff-line-num">{old_num}</span>'
                        f'<span class="diff-line-num">{new_num}</span>'
                        f'{prefix}{escaped_content}</div>'
                    )
                
                html_parts.append('</div>')
        
        html_parts.append('</div>')
        return ''.join(html_parts)
    
    def compare_words(
        self,
        old_text: str,
        new_text: str
    ) -> list[tuple[str, ChangeType]]:
        """
        Compare texts word by word.
        
        Args:
            old_text: Original text
            new_text: New text
            
        Returns:
            List of (word, change_type) tuples
        """
        old_words = old_text.split()
        new_words = new_text.split()
        
        matcher = difflib.SequenceMatcher(None, old_words, new_words)
        result = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for word in old_words[i1:i2]:
                    result.append((word, ChangeType.UNCHANGED))
            elif tag == 'replace':
                for word in old_words[i1:i2]:
                    result.append((word, ChangeType.REMOVED))
                for word in new_words[j1:j2]:
                    result.append((word, ChangeType.ADDED))
            elif tag == 'delete':
                for word in old_words[i1:i2]:
                    result.append((word, ChangeType.REMOVED))
            elif tag == 'insert':
                for word in new_words[j1:j2]:
                    result.append((word, ChangeType.ADDED))
        
        return result
    
    def compare_chars(
        self,
        old_text: str,
        new_text: str
    ) -> list[tuple[str, ChangeType]]:
        """
        Compare texts character by character.
        
        Args:
            old_text: Original text
            new_text: New text
            
        Returns:
            List of (char, change_type) tuples
        """
        matcher = difflib.SequenceMatcher(None, old_text, new_text)
        result = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for char in old_text[i1:i2]:
                    result.append((char, ChangeType.UNCHANGED))
            elif tag == 'replace':
                for char in old_text[i1:i2]:
                    result.append((char, ChangeType.REMOVED))
                for char in new_text[j1:j2]:
                    result.append((char, ChangeType.ADDED))
            elif tag == 'delete':
                for char in old_text[i1:i2]:
                    result.append((char, ChangeType.REMOVED))
            elif tag == 'insert':
                for char in new_text[j1:j2]:
                    result.append((char, ChangeType.ADDED))
        
        return result
    
    def side_by_side(
        self,
        old_text: str,
        new_text: str,
        width: int = 80
    ) -> str:
        """
        Generate side-by-side comparison.
        
        Args:
            old_text: Original text
            new_text: New text
            width: Width of each column
            
        Returns:
            Side-by-side formatted string
        """
        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()
        
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        result = []
        
        half_width = width // 2 - 2
        separator = " | "
        
        # Header
        result.append("=" * width)
        result.append(f"{'Original':<{half_width}}{separator}{'Modified':<{half_width}}")
        result.append("=" * width)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for i in range(i2 - i1):
                    old_line = old_lines[i1 + i][:half_width]
                    new_line = new_lines[j1 + i][:half_width]
                    result.append(f"{old_line:<{half_width}}{separator}{new_line:<{half_width}}")
            
            elif tag == 'replace':
                max_len = max(i2 - i1, j2 - j1)
                for i in range(max_len):
                    old_line = old_lines[i1 + i][:half_width] if i < i2 - i1 else ""
                    new_line = new_lines[j1 + i][:half_width] if i < j2 - j1 else ""
                    prefix_old = "- " if old_line else "  "
                    prefix_new = "+ " if new_line else "  "
                    result.append(f"{prefix_old}{old_line:<{half_width-2}}{separator}{prefix_new}{new_line:<{half_width-2}}")
            
            elif tag == 'delete':
                for i in range(i2 - i1):
                    old_line = old_lines[i1 + i][:half_width]
                    result.append(f"- {old_line:<{half_width-2}}{separator}{'':<{half_width}}")
            
            elif tag == 'insert':
                for i in range(j2 - j1):
                    new_line = new_lines[j1 + i][:half_width]
                    result.append(f"{'':<{half_width}}{separator}+ {new_line:<{half_width-2}}")
        
        result.append("=" * width)
        return "\n".join(result)
    
    def to_html(
        self,
        old_text: str,
        new_text: str,
        inline_changes: bool = True
    ) -> str:
        """
        Generate HTML diff with optional inline change highlighting.
        
        Args:
            old_text: Original text
            new_text: New text
            inline_changes: Highlight word-level changes within lines
            
        Returns:
            HTML string
        """
        result = self.compare(old_text, new_text)
        
        if inline_changes:
            return self._generate_inline_html(old_text, new_text)
        
        return result.html
    
    def _generate_inline_html(self, old_text: str, new_text: str) -> str:
        """Generate HTML with inline word-level highlighting."""
        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()
        
        html_parts = ["""
<style>
.inline-diff { font-family: monospace; font-size: 13px; line-height: 1.5; }
.inline-diff-line { padding: 2px 8px; }
.inline-added { background: #acf2bd; }
.inline-removed { background: #fdb8c0; text-decoration: line-through; }
.line-added { background: #e6ffec; }
.line-removed { background: #ffebe9; }
</style>
<div class="inline-diff">
"""]
        
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for line in old_lines[i1:i2]:
                    escaped = self._escape_html(line)
                    html_parts.append(f'<div class="inline-diff-line">{escaped}</div>')
            
            elif tag == 'replace':
                # Show inline changes for replaced lines
                for i in range(max(i2 - i1, j2 - j1)):
                    old_line = old_lines[i1 + i] if i < i2 - i1 else ""
                    new_line = new_lines[j1 + i] if i < j2 - j1 else ""
                    
                    if old_line and new_line:
                        # Show inline diff
                        inline_html = self._inline_word_diff(old_line, new_line)
                        html_parts.append(f'<div class="inline-diff-line">{inline_html}</div>')
                    elif old_line:
                        escaped = self._escape_html(old_line)
                        html_parts.append(f'<div class="inline-diff-line line-removed"><span class="inline-removed">{escaped}</span></div>')
                    elif new_line:
                        escaped = self._escape_html(new_line)
                        html_parts.append(f'<div class="inline-diff-line line-added"><span class="inline-added">{escaped}</span></div>')
            
            elif tag == 'delete':
                for line in old_lines[i1:i2]:
                    escaped = self._escape_html(line)
                    html_parts.append(f'<div class="inline-diff-line line-removed"><span class="inline-removed">{escaped}</span></div>')
            
            elif tag == 'insert':
                for line in new_lines[j1:j2]:
                    escaped = self._escape_html(line)
                    html_parts.append(f'<div class="inline-diff-line line-added"><span class="inline-added">{escaped}</span></div>')
        
        html_parts.append('</div>')
        return ''.join(html_parts)
    
    def _inline_word_diff(self, old_line: str, new_line: str) -> str:
        """Generate inline word-level diff HTML."""
        old_words = self._tokenize_for_diff(old_line)
        new_words = self._tokenize_for_diff(new_line)
        
        matcher = difflib.SequenceMatcher(None, old_words, new_words)
        result = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                result.append(self._escape_html(''.join(old_words[i1:i2])))
            elif tag == 'replace':
                old_text = self._escape_html(''.join(old_words[i1:i2]))
                new_text = self._escape_html(''.join(new_words[j1:j2]))
                result.append(f'<span class="inline-removed">{old_text}</span>')
                result.append(f'<span class="inline-added">{new_text}</span>')
            elif tag == 'delete':
                old_text = self._escape_html(''.join(old_words[i1:i2]))
                result.append(f'<span class="inline-removed">{old_text}</span>')
            elif tag == 'insert':
                new_text = self._escape_html(''.join(new_words[j1:j2]))
                result.append(f'<span class="inline-added">{new_text}</span>')
        
        return ''.join(result)
    
    def _tokenize_for_diff(self, text: str) -> list[str]:
        """Tokenize text for diffing, preserving whitespace."""
        return re.findall(r'\S+|\s+', text)
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML entities."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Get similarity ratio between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity ratio (0.0 to 1.0)
        """
        if not text1 and not text2:
            return 1.0
        matcher = difflib.SequenceMatcher(None, text1, text2)
        return matcher.ratio()
    
    def find_similar_lines(
        self,
        text: str,
        pattern: str,
        threshold: float = 0.6
    ) -> list[tuple[int, str, float]]:
        """
        Find lines similar to a pattern.
        
        Args:
            text: Text to search in
            pattern: Pattern to match
            threshold: Minimum similarity threshold
            
        Returns:
            List of (line_number, line, similarity) tuples
        """
        results = []
        for i, line in enumerate(text.splitlines(), 1):
            similarity = self.get_similarity(line.lower(), pattern.lower())
            if similarity >= threshold:
                results.append((i, line, similarity))
        
        return sorted(results, key=lambda x: x[2], reverse=True)


# Singleton instance
_diff_instance: TextDiff | None = None


def get_text_diff() -> TextDiff:
    """Get or create the singleton text diff utility."""
    global _diff_instance
    if _diff_instance is None:
        _diff_instance = TextDiff()
    return _diff_instance


# Convenience functions
def compare_texts(old_text: str, new_text: str) -> DiffResult:
    """Compare two texts and return diff result."""
    return get_text_diff().compare(old_text, new_text)


def diff_to_html(old_text: str, new_text: str) -> str:
    """Generate HTML diff visualization."""
    return get_text_diff().to_html(old_text, new_text)


def get_similarity(text1: str, text2: str) -> float:
    """Get similarity between two texts."""
    return get_text_diff().get_similarity(text1, text2)
