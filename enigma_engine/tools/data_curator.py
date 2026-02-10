"""
Data Curator - Scan, index, and manage training data files.

This tool provides functionality to:
- Scan and index all training data files
- Search by topic, character, style, or sentiment
- Tag and categorize data automatically
- Detect duplicate or low-quality entries
- Merge/split datasets with smart deduplication

Usage:
    from enigma_engine.tools.data_curator import DataCurator
    
    curator = DataCurator()
    
    # Index training data
    curator.index_directory("data/training/")
    
    # Search for content
    results = curator.search("sherlock holmes", category="character")
    
    # Detect duplicates
    duplicates = curator.find_duplicates()
    
    # Merge datasets
    curator.merge_datasets(["file1.txt", "file2.txt"], "merged.txt")
"""

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DataEntry:
    """A single entry in the training data."""
    id: str
    content: str
    source_file: str
    line_number: int
    entry_type: str = "text"            # "qa", "chat", "text", "json"
    tags: List[str] = field(default_factory=list)
    category: str = ""
    quality_score: float = 1.0
    content_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()
        if not self.id:
            self.id = f"{self.content_hash[:8]}_{self.line_number}"


@dataclass
class DataFile:
    """Metadata about a training data file."""
    path: str
    name: str
    size: int
    entry_count: int
    format: str                         # "txt", "json", "jsonl", "csv"
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    quality_avg: float = 1.0
    last_indexed: str = ""
    sample_entries: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "size": self.size,
            "entry_count": self.entry_count,
            "format": self.format,
            "categories": self.categories,
            "tags": self.tags,
            "quality_avg": self.quality_avg,
            "last_indexed": self.last_indexed,
            "sample_entries": self.sample_entries[:5]
        }


@dataclass
class SearchResult:
    """Result from a search query."""
    entry: DataEntry
    score: float
    matched_terms: List[str]


@dataclass
class DuplicateGroup:
    """Group of duplicate entries."""
    hash: str
    entries: List[DataEntry]
    similarity: float


class QualityChecker:
    """Check quality of training data entries."""
    
    # Quality factors
    MIN_LENGTH = 10
    MAX_LENGTH = 5000
    MIN_WORDS = 3
    
    # Problematic patterns
    SPAM_PATTERNS = [
        r'https?://\S+',                # URLs (too many = spam)
        r'[A-Z]{10,}',                  # All caps text
        r'(.)\1{5,}',                   # Repeated characters
        r'!\s*!' * 3,                   # Multiple exclamations  
    ]
    
    def check_quality(self, content: str) -> Tuple[float, List[str]]:
        """
        Check quality of content.
        
        Returns:
            (quality_score, list of issues)
        """
        issues = []
        score = 1.0
        
        # Length checks
        if len(content) < self.MIN_LENGTH:
            issues.append("Too short")
            score -= 0.3
        elif len(content) > self.MAX_LENGTH:
            issues.append("Too long")
            score -= 0.1
        
        # Word count
        words = content.split()
        if len(words) < self.MIN_WORDS:
            issues.append("Too few words")
            score -= 0.2
        
        # Spam patterns
        for pattern in self.SPAM_PATTERNS:
            if re.search(pattern, content):
                issues.append(f"Matches spam pattern")
                score -= 0.2
                break
        
        # Check for encoding issues
        if '�' in content or '\\x' in content:
            issues.append("Encoding issues")
            score -= 0.3
        
        # Check for empty or whitespace-only
        if not content.strip():
            issues.append("Empty or whitespace only")
            score = 0.0
        
        # Normalize score
        score = max(0.0, min(1.0, score))
        
        return score, issues


class DataCurator:
    """
    Scan, index, and manage training data files.
    
    Provides tools for organizing, cleaning, and preparing
    training data for AI model training.
    """
    
    # Supported file formats
    SUPPORTED_FORMATS = {".txt", ".json", ".jsonl", ".csv", ".md"}
    
    # Category keywords for auto-categorization
    CATEGORY_KEYWORDS = {
        "conversation": ["user:", "assistant:", "human:", "ai:", "q:", "a:"],
        "code": ["def ", "class ", "function", "import ", "from ", "```"],
        "story": ["once upon", "chapter", "the end", "said", "asked"],
        "factual": ["is a", "was born", "located in", "founded in", "wikipedia"],
        "instruction": ["step ", "first", "then", "finally", "how to"],
        "dialogue": ["\"", "'", ":", "said", "replied", "asked"],
    }
    
    def __init__(self, index_path: Optional[Path] = None):
        """
        Initialize data curator.
        
        Args:
            index_path: Path to store/load index
        """
        self.index_path = index_path or Path("data/.curator_index.json")
        
        # Data storage
        self.files: Dict[str, DataFile] = {}
        self.entries: Dict[str, DataEntry] = {}
        self.hash_to_entries: Dict[str, List[str]] = defaultdict(list)
        
        # Search indices
        self.word_index: Dict[str, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.category_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Quality checker
        self.quality_checker = QualityChecker()
        
        # Load existing index
        self._load_index()
    
    def _load_index(self):
        """Load index from disk."""
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                
                # Restore files
                for path, file_data in data.get("files", {}).items():
                    self.files[path] = DataFile(**file_data)
                
                logger.info(f"Loaded index with {len(self.files)} files")
            except Exception as e:
                logger.warning(f"Error loading index: {e}")
    
    def save_index(self):
        """Save index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "files": {path: file.to_dict() for path, file in self.files.items()},
            "indexed_at": datetime.now().isoformat()
        }
        
        self.index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"Saved index to {self.index_path}")
    
    def index_directory(
        self,
        directory: str,
        recursive: bool = True,
        reindex: bool = False
    ) -> Dict[str, Any]:
        """
        Index all training data files in a directory.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            reindex: Whether to re-index existing files
            
        Returns:
            Statistics about indexed files
        """
        path = Path(directory)
        if not path.exists():
            logger.error(f"Directory not found: {directory}")
            return {"error": "Directory not found"}
        
        stats = {
            "files_scanned": 0,
            "files_indexed": 0,
            "entries_found": 0,
            "duplicates_found": 0,
            "low_quality": 0
        }
        
        # Get files
        if recursive:
            files = [f for f in path.rglob("*") if f.suffix.lower() in self.SUPPORTED_FORMATS]
        else:
            files = [f for f in path.glob("*") if f.suffix.lower() in self.SUPPORTED_FORMATS]
        
        logger.info(f"Found {len(files)} data files to index")
        
        for file_path in files:
            stats["files_scanned"] += 1
            
            # Skip if already indexed and not reindexing
            str_path = str(file_path)
            if str_path in self.files and not reindex:
                continue
            
            # Index file
            file_stats = self._index_file(file_path)
            
            if file_stats["success"]:
                stats["files_indexed"] += 1
                stats["entries_found"] += file_stats["entries"]
                stats["duplicates_found"] += file_stats["duplicates"]
                stats["low_quality"] += file_stats["low_quality"]
        
        # Save index
        self.save_index()
        
        logger.info(f"Indexed {stats['files_indexed']} files, {stats['entries_found']} entries")
        return stats
    
    def _index_file(self, file_path: Path) -> Dict[str, Any]:
        """Index a single file."""
        stats = {
            "success": False,
            "entries": 0,
            "duplicates": 0,
            "low_quality": 0
        }
        
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            suffix = file_path.suffix.lower()
            
            # Parse based on format
            if suffix == ".json":
                entries = self._parse_json(content, str(file_path))
            elif suffix == ".jsonl":
                entries = self._parse_jsonl(content, str(file_path))
            elif suffix == ".csv":
                entries = self._parse_csv(content, str(file_path))
            else:
                entries = self._parse_text(content, str(file_path))
            
            # Process entries
            categories = set()
            tags = set()
            quality_scores = []
            
            for entry in entries:
                # Check for duplicates
                if entry.content_hash in self.hash_to_entries:
                    stats["duplicates"] += 1
                    entry.tags.append("duplicate")
                
                # Auto-categorize
                entry.category = self._auto_categorize(entry.content)
                categories.add(entry.category)
                
                # Auto-tag
                entry.tags.extend(self._auto_tag(entry.content))
                tags.update(entry.tags)
                
                # Check quality
                quality, issues = self.quality_checker.check_quality(entry.content)
                entry.quality_score = quality
                quality_scores.append(quality)
                
                if quality < 0.5:
                    stats["low_quality"] += 1
                    entry.tags.append("low_quality")
                
                # Add to indices
                self._add_to_indices(entry)
                self.entries[entry.id] = entry
                self.hash_to_entries[entry.content_hash].append(entry.id)
            
            # Create file record
            self.files[str(file_path)] = DataFile(
                path=str(file_path),
                name=file_path.name,
                size=file_path.stat().st_size,
                entry_count=len(entries),
                format=suffix[1:],
                categories=list(categories),
                tags=list(tags)[:20],
                quality_avg=sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                last_indexed=datetime.now().isoformat(),
                sample_entries=[e.content[:100] for e in entries[:5]]
            )
            
            stats["success"] = True
            stats["entries"] = len(entries)
            
        except Exception as e:
            logger.warning(f"Error indexing {file_path}: {e}")
        
        return stats
    
    def _parse_text(self, content: str, source_file: str) -> List[DataEntry]:
        """Parse plain text file."""
        entries = []
        
        # Try to detect format
        lines = content.split("\n")
        
        # Check for Q&A format
        if any(line.strip().lower().startswith(("q:", "question:")) for line in lines):
            return self._parse_qa_format(content, source_file)
        
        # Check for chat format
        if any(line.strip().lower().startswith(("user:", "assistant:", "human:", "ai:")) for line in lines):
            return self._parse_chat_format(content, source_file)
        
        # Default: each paragraph is an entry
        paragraphs = content.split("\n\n")
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                entries.append(DataEntry(
                    id="",
                    content=para,
                    source_file=source_file,
                    line_number=i,
                    entry_type="text"
                ))
        
        return entries
    
    def _parse_qa_format(self, content: str, source_file: str) -> List[DataEntry]:
        """Parse Q&A format text."""
        entries = []
        lines = content.split("\n")
        
        current_q = None
        current_a = []
        line_num = 0
        
        for i, line in enumerate(lines):
            lower = line.strip().lower()
            
            if lower.startswith(("q:", "question:")):
                # Save previous pair
                if current_q and current_a:
                    entries.append(DataEntry(
                        id="",
                        content=f"Q: {current_q}\nA: {' '.join(current_a)}",
                        source_file=source_file,
                        line_number=line_num,
                        entry_type="qa"
                    ))
                
                # Start new question
                current_q = line.split(":", 1)[1].strip() if ":" in line else line[2:].strip()
                current_a = []
                line_num = i
                
            elif lower.startswith(("a:", "answer:")):
                answer = line.split(":", 1)[1].strip() if ":" in line else line[2:].strip()
                current_a.append(answer)
            elif current_a:  # Continuation of answer
                current_a.append(line.strip())
        
        # Save last pair
        if current_q and current_a:
            entries.append(DataEntry(
                id="",
                content=f"Q: {current_q}\nA: {' '.join(current_a)}",
                source_file=source_file,
                line_number=line_num,
                entry_type="qa"
            ))
        
        return entries
    
    def _parse_chat_format(self, content: str, source_file: str) -> List[DataEntry]:
        """Parse chat format text."""
        entries = []
        lines = content.split("\n")
        
        current_messages = []
        line_num = 0
        
        for i, line in enumerate(lines):
            lower = line.strip().lower()
            
            if any(lower.startswith(prefix) for prefix in ["user:", "assistant:", "human:", "ai:", "system:"]):
                current_messages.append(line.strip())
                if line_num == 0:
                    line_num = i
            elif line.strip() == "" and current_messages:
                # End of conversation
                entries.append(DataEntry(
                    id="",
                    content="\n".join(current_messages),
                    source_file=source_file,
                    line_number=line_num,
                    entry_type="chat"
                ))
                current_messages = []
                line_num = 0
            elif current_messages:
                # Continuation
                current_messages[-1] += " " + line.strip()
        
        # Save remaining
        if current_messages:
            entries.append(DataEntry(
                id="",
                content="\n".join(current_messages),
                source_file=source_file,
                line_number=line_num,
                entry_type="chat"
            ))
        
        return entries
    
    def _parse_json(self, content: str, source_file: str) -> List[DataEntry]:
        """Parse JSON file."""
        entries = []
        
        try:
            data = json.loads(content)
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        # Try common field names
                        text = item.get("text") or item.get("content") or item.get("message") or str(item)
                        entries.append(DataEntry(
                            id="",
                            content=text,
                            source_file=source_file,
                            line_number=i,
                            entry_type="json",
                            metadata=item if isinstance(item, dict) else {}
                        ))
            elif isinstance(data, dict):
                entries.append(DataEntry(
                    id="",
                    content=json.dumps(data),
                    source_file=source_file,
                    line_number=0,
                    entry_type="json",
                    metadata=data
                ))
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
        
        return entries
    
    def _parse_jsonl(self, content: str, source_file: str) -> List[DataEntry]:
        """Parse JSONL file."""
        entries = []
        
        for i, line in enumerate(content.strip().split("\n")):
            try:
                item = json.loads(line)
                text = item.get("text") or item.get("content") or item.get("message") or str(item)
                entries.append(DataEntry(
                    id="",
                    content=text,
                    source_file=source_file,
                    line_number=i,
                    entry_type="json",
                    metadata=item if isinstance(item, dict) else {}
                ))
            except json.JSONDecodeError:
                continue
        
        return entries
    
    def _parse_csv(self, content: str, source_file: str) -> List[DataEntry]:
        """Parse CSV file."""
        entries = []
        
        lines = content.strip().split("\n")
        if len(lines) < 2:
            return entries
        
        # Simple CSV parsing (no complex escaping)
        headers = lines[0].split(",")
        
        # Find text column
        text_col = None
        for i, h in enumerate(headers):
            if h.lower().strip() in ["text", "content", "message", "input", "output"]:
                text_col = i
                break
        
        if text_col is None:
            text_col = 0
        
        for i, line in enumerate(lines[1:], 1):
            fields = line.split(",")
            if len(fields) > text_col:
                text = fields[text_col].strip().strip('"')
                entries.append(DataEntry(
                    id="",
                    content=text,
                    source_file=source_file,
                    line_number=i,
                    entry_type="csv"
                ))
        
        return entries
    
    def _auto_categorize(self, content: str) -> str:
        """Auto-categorize content based on keywords."""
        content_lower = content.lower()
        
        category_scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores.keys(), key=lambda k: category_scores[k])
        
        return "general"
    
    def _auto_tag(self, content: str) -> List[str]:
        """Auto-tag content."""
        tags = []
        content_lower = content.lower()
        
        # Language detection (simple)
        if "def " in content or "import " in content:
            tags.append("python")
        if "function" in content and ("{" in content or "=>" in content):
            tags.append("javascript")
        
        # Content type
        if "?" in content:
            tags.append("question")
        if len(content) > 1000:
            tags.append("long")
        if len(content) < 50:
            tags.append("short")
        
        # Sentiment (very simple)
        positive_words = ["good", "great", "excellent", "happy", "love", "wonderful"]
        negative_words = ["bad", "terrible", "hate", "awful", "sad", "angry"]
        
        pos_count = sum(1 for w in positive_words if w in content_lower)
        neg_count = sum(1 for w in negative_words if w in content_lower)
        
        if pos_count > neg_count:
            tags.append("positive")
        elif neg_count > pos_count:
            tags.append("negative")
        
        return tags
    
    def _add_to_indices(self, entry: DataEntry):
        """Add entry to search indices."""
        # Word index
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', entry.content.lower()))
        for word in words:
            self.word_index[word].add(entry.id)
        
        # Tag index
        for tag in entry.tags:
            self.tag_index[tag].add(entry.id)
        
        # Category index
        if entry.category:
            self.category_index[entry.category].add(entry.id)
    
    def search(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_quality: float = 0.0,
        limit: int = 100
    ) -> List[SearchResult]:
        """
        Search for entries.
        
        Args:
            query: Search query
            category: Filter by category
            tags: Filter by tags
            min_quality: Minimum quality score
            limit: Maximum results
            
        Returns:
            List of search results
        """
        results = []
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        
        # Find matching entry IDs
        matching_ids = set()
        
        # Word matching
        for word in query_words:
            if word in self.word_index:
                if not matching_ids:
                    matching_ids = self.word_index[word].copy()
                else:
                    # Intersection for AND search
                    matching_ids &= self.word_index[word]
        
        # Category filter
        if category and category in self.category_index:
            if matching_ids:
                matching_ids &= self.category_index[category]
            else:
                matching_ids = self.category_index[category].copy()
        
        # Tag filter
        if tags:
            for tag in tags:
                if tag in self.tag_index:
                    if matching_ids:
                        matching_ids &= self.tag_index[tag]
                    else:
                        matching_ids = self.tag_index[tag].copy()
        
        # Score and filter results
        for entry_id in matching_ids:
            if entry_id not in self.entries:
                continue
            
            entry = self.entries[entry_id]
            
            # Quality filter
            if entry.quality_score < min_quality:
                continue
            
            # Calculate score
            matched_terms = [w for w in query_words if w in entry.content.lower()]
            score = len(matched_terms) / len(query_words) if query_words else 0
            score *= entry.quality_score  # Weight by quality
            
            results.append(SearchResult(
                entry=entry,
                score=score,
                matched_terms=matched_terms
            ))
        
        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def find_duplicates(self, similarity_threshold: float = 1.0) -> List[DuplicateGroup]:
        """
        Find duplicate entries.
        
        Args:
            similarity_threshold: 1.0 = exact duplicates only
            
        Returns:
            List of duplicate groups
        """
        groups = []
        
        for hash_value, entry_ids in self.hash_to_entries.items():
            if len(entry_ids) > 1:
                entries = [self.entries[eid] for eid in entry_ids if eid in self.entries]
                groups.append(DuplicateGroup(
                    hash=hash_value,
                    entries=entries,
                    similarity=1.0
                ))
        
        return groups
    
    def merge_datasets(
        self,
        input_files: List[str],
        output_file: str,
        deduplicate: bool = True,
        min_quality: float = 0.5
    ) -> Dict[str, int]:
        """
        Merge multiple datasets into one.
        
        Args:
            input_files: List of input file paths
            output_file: Output file path
            deduplicate: Remove duplicates
            min_quality: Minimum quality threshold
            
        Returns:
            Merge statistics
        """
        stats = {
            "files_merged": 0,
            "entries_added": 0,
            "duplicates_removed": 0,
            "low_quality_removed": 0
        }
        
        seen_hashes = set()
        output_entries = []
        
        for file_path in input_files:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            # Read and parse file
            content = path.read_text(encoding="utf-8", errors="ignore")
            entries = self._parse_text(content, file_path)
            
            for entry in entries:
                # Check quality
                quality, _ = self.quality_checker.check_quality(entry.content)
                if quality < min_quality:
                    stats["low_quality_removed"] += 1
                    continue
                
                # Check duplicates
                if deduplicate and entry.content_hash in seen_hashes:
                    stats["duplicates_removed"] += 1
                    continue
                
                seen_hashes.add(entry.content_hash)
                output_entries.append(entry.content)
                stats["entries_added"] += 1
            
            stats["files_merged"] += 1
        
        # Write output
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n\n".join(output_entries), encoding="utf-8")
        
        logger.info(f"Merged {stats['files_merged']} files -> {output_file}")
        return stats
    
    def split_dataset(
        self,
        input_file: str,
        output_dir: str,
        split_by: str = "category",
        entries_per_file: int = 1000
    ) -> Dict[str, str]:
        """
        Split a dataset into multiple files.
        
        Args:
            input_file: Input file path
            output_dir: Output directory
            split_by: "category", "size", or "random"
            entries_per_file: Entries per output file (for size split)
            
        Returns:
            Mapping of split names to output files
        """
        path = Path(input_file)
        if not path.exists():
            logger.error(f"File not found: {input_file}")
            return {}
        
        # Parse input
        content = path.read_text(encoding="utf-8", errors="ignore")
        entries = self._parse_text(content, input_file)
        
        # Categorize entries
        for entry in entries:
            entry.category = self._auto_categorize(entry.content)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        
        if split_by == "category":
            # Group by category
            by_category = defaultdict(list)
            for entry in entries:
                by_category[entry.category].append(entry.content)
            
            for category, contents in by_category.items():
                out_file = output_path / f"{category}.txt"
                out_file.write_text("\n\n".join(contents), encoding="utf-8")
                outputs[category] = str(out_file)
        
        elif split_by == "size":
            # Split into fixed-size chunks
            for i in range(0, len(entries), entries_per_file):
                chunk = entries[i:i + entries_per_file]
                out_file = output_path / f"part_{i // entries_per_file + 1}.txt"
                out_file.write_text(
                    "\n\n".join(e.content for e in chunk),
                    encoding="utf-8"
                )
                outputs[f"part_{i // entries_per_file + 1}"] = str(out_file)
        
        logger.info(f"Split {len(entries)} entries into {len(outputs)} files")
        return outputs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about indexed data."""
        if not self.files:
            return {"status": "No data indexed"}
        
        total_entries = sum(f.entry_count for f in self.files.values())
        total_size = sum(f.size for f in self.files.values())
        
        categories = Counter()
        for entry in self.entries.values():
            categories[entry.category] += 1
        
        return {
            "files_indexed": len(self.files),
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "unique_entries": len(self.entries),
            "duplicate_groups": len([h for h, ids in self.hash_to_entries.items() if len(ids) > 1]),
            "categories": dict(categories),
            "formats": Counter(f.format for f in self.files.values()),
            "avg_quality": sum(e.quality_score for e in self.entries.values()) / len(self.entries) if self.entries else 0
        }


# Convenience function
def get_data_curator() -> DataCurator:
    """Get a DataCurator instance."""
    return DataCurator()
