"""
Productivity Tools - Commit messages, task extraction, SQL builder, model switching.

Developer productivity features:
- Generate git commit messages from diff
- Extract action items/tasks from conversations  
- Natural language to SQL query builder
- Quick model switching for retries

Part of the ForgeAI productivity suite.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Tuple, Union
from pathlib import Path
from enum import Enum, auto
import logging
import subprocess

logger = logging.getLogger(__name__)


# =============================================================================
# COMMIT MESSAGE GENERATOR
# =============================================================================

class CommitType(Enum):
    """Conventional commit types."""
    FEAT = "feat"       # New feature
    FIX = "fix"         # Bug fix
    DOCS = "docs"       # Documentation
    STYLE = "style"     # Formatting, no code change
    REFACTOR = "refactor"  # Refactoring
    PERF = "perf"       # Performance improvement
    TEST = "test"       # Adding tests
    BUILD = "build"     # Build system changes
    CI = "ci"           # CI configuration
    CHORE = "chore"     # Maintenance tasks
    REVERT = "revert"   # Revert previous commit


@dataclass
class CommitSuggestion:
    """A suggested commit message."""
    title: str          # Short commit title
    body: str = ""      # Extended description
    commit_type: CommitType = CommitType.CHORE
    scope: str = ""     # Optional scope (e.g., "api", "ui")
    breaking: bool = False  # Breaking change
    confidence: float = 0.5
    
    def format_conventional(self) -> str:
        """Format as conventional commit."""
        type_str = self.commit_type.value
        if self.scope:
            type_str = f"{type_str}({self.scope})"
        if self.breaking:
            type_str = f"{type_str}!"
        
        msg = f"{type_str}: {self.title}"
        
        if self.body:
            msg = f"{msg}\n\n{self.body}"
        
        return msg
    
    def format_simple(self) -> str:
        """Format as simple commit message."""
        msg = self.title
        if self.body:
            msg = f"{msg}\n\n{self.body}"
        return msg


class CommitMessageGenerator:
    """
    Generate commit messages from git diff or change descriptions.
    
    Features:
    - Analyze git diff for changes
    - Detect commit type from changes
    - Conventional commit format
    - Multiple suggestion styles
    """
    
    # Keywords that suggest different commit types
    TYPE_KEYWORDS = {
        CommitType.FEAT: ["add", "new", "create", "implement", "feature"],
        CommitType.FIX: ["fix", "bug", "error", "issue", "resolve", "correct"],
        CommitType.DOCS: ["doc", "readme", "comment", "documentation"],
        CommitType.STYLE: ["format", "style", "lint", "whitespace"],
        CommitType.REFACTOR: ["refactor", "restructure", "reorganize", "clean"],
        CommitType.PERF: ["optimize", "performance", "speed", "cache"],
        CommitType.TEST: ["test", "spec", "coverage"],
        CommitType.BUILD: ["build", "webpack", "package", "dependency"],
        CommitType.CI: ["ci", "pipeline", "workflow", "action"],
    }
    
    # File extension to scope mapping
    SCOPE_MAP = {
        ".py": "python",
        ".js": "js",
        ".ts": "typescript",
        ".vue": "vue",
        ".jsx": "react",
        ".tsx": "react",
        ".css": "style",
        ".scss": "style",
        ".html": "ui",
        ".md": "docs",
        ".yml": "config",
        ".yaml": "config",
        ".json": "config",
    }
    
    def __init__(self, repo_path: Optional[Path] = None):
        """
        Initialize generator.
        
        Args:
            repo_path: Path to git repository
        """
        self.repo_path = repo_path or Path.cwd()
    
    def get_staged_diff(self) -> str:
        """Get diff of staged changes."""
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--stat"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout
        except Exception as e:
            logger.warning(f"Failed to get git diff: {e}")
            return ""
    
    def get_staged_files(self) -> List[str]:
        """Get list of staged files."""
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout.strip().split("\n") if result.stdout else []
        except Exception as e:
            logger.warning(f"Failed to get staged files: {e}")
            return []
    
    def analyze_changes(self, diff: str, files: List[str]) -> Dict[str, Any]:
        """
        Analyze changes to determine commit type and scope.
        
        Args:
            diff: Git diff output
            files: List of changed files
            
        Returns:
            Analysis dict with type, scope, stats
        """
        analysis = {
            "type": CommitType.CHORE,
            "scope": "",
            "files_changed": len(files),
            "is_breaking": False,
            "keywords_found": []
        }
        
        # Detect type from diff content
        diff_lower = diff.lower()
        for commit_type, keywords in self.TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in diff_lower:
                    analysis["type"] = commit_type
                    analysis["keywords_found"].append(keyword)
                    break
        
        # Detect scope from file extensions
        extensions = set()
        for f in files:
            ext = Path(f).suffix
            if ext in self.SCOPE_MAP:
                extensions.add(self.SCOPE_MAP[ext])
        
        if len(extensions) == 1:
            analysis["scope"] = extensions.pop()
        elif len(extensions) > 1:
            # Multiple scopes, use most common
            analysis["scope"] = ""
        
        # Check for breaking change indicators
        if "breaking" in diff_lower or "BREAKING" in diff:
            analysis["is_breaking"] = True
        
        return analysis
    
    def generate_suggestions(
        self,
        diff: Optional[str] = None,
        description: Optional[str] = None,
        max_suggestions: int = 3
    ) -> List[CommitSuggestion]:
        """
        Generate commit message suggestions.
        
        Args:
            diff: Git diff (auto-fetched if None)
            description: Optional manual description of changes
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of commit suggestions
        """
        if diff is None:
            diff = self.get_staged_diff()
        
        files = self.get_staged_files()
        analysis = self.analyze_changes(diff, files)
        
        suggestions = []
        
        # Generate based on file changes
        if files:
            # Primary suggestion based on analysis
            title = self._generate_title(files, analysis)
            suggestions.append(CommitSuggestion(
                title=title,
                body=self._generate_body(files, diff),
                commit_type=analysis["type"],
                scope=analysis["scope"],
                breaking=analysis["is_breaking"],
                confidence=0.8
            ))
            
            # Alternative phrasings
            alt_titles = self._generate_alternatives(files, analysis)
            for alt_title in alt_titles[:max_suggestions - 1]:
                suggestions.append(CommitSuggestion(
                    title=alt_title,
                    commit_type=analysis["type"],
                    scope=analysis["scope"],
                    confidence=0.6
                ))
        
        # If description provided, add suggestion from it
        if description:
            suggestions.insert(0, CommitSuggestion(
                title=description[:72],  # Truncate to conventional limit
                body=description[72:] if len(description) > 72 else "",
                commit_type=analysis["type"],
                confidence=0.9
            ))
        
        return suggestions[:max_suggestions]
    
    def _generate_title(self, files: List[str], analysis: Dict[str, Any]) -> str:
        """Generate commit title from analysis."""
        if len(files) == 1:
            filename = Path(files[0]).name
            verb = self._get_verb(analysis["type"])
            return f"{verb} {filename}"
        elif len(files) <= 3:
            names = [Path(f).name for f in files]
            verb = self._get_verb(analysis["type"])
            return f"{verb} {', '.join(names)}"
        else:
            verb = self._get_verb(analysis["type"])
            # Group by directory
            dirs = set(str(Path(f).parent) for f in files)
            if len(dirs) == 1:
                return f"{verb} files in {list(dirs)[0]}"
            return f"{verb} {len(files)} files"
    
    def _generate_body(self, files: List[str], diff: str) -> str:
        """Generate commit body with details."""
        if len(files) <= 5:
            return ""
        
        lines = [f"Changed {len(files)} files:", ""]
        for f in files[:10]:
            lines.append(f"- {f}")
        if len(files) > 10:
            lines.append(f"- ... and {len(files) - 10} more")
        
        return "\n".join(lines)
    
    def _generate_alternatives(self, files: List[str], analysis: Dict[str, Any]) -> List[str]:
        """Generate alternative title phrasings."""
        alternatives = []
        
        verbs = {
            CommitType.FEAT: ["Add", "Implement", "Create"],
            CommitType.FIX: ["Fix", "Resolve", "Correct"],
            CommitType.DOCS: ["Update", "Add", "Improve"],
            CommitType.REFACTOR: ["Refactor", "Clean up", "Restructure"],
        }
        
        verb_list = verbs.get(analysis["type"], ["Update", "Modify"])
        
        for verb in verb_list[1:]:  # Skip first (already used)
            if len(files) == 1:
                alternatives.append(f"{verb} {Path(files[0]).name}")
            else:
                alternatives.append(f"{verb} {len(files)} files")
        
        return alternatives
    
    def _get_verb(self, commit_type: CommitType) -> str:
        """Get primary verb for commit type."""
        return {
            CommitType.FEAT: "Add",
            CommitType.FIX: "Fix",
            CommitType.DOCS: "Update docs for",
            CommitType.STYLE: "Format",
            CommitType.REFACTOR: "Refactor",
            CommitType.PERF: "Optimize",
            CommitType.TEST: "Add tests for",
            CommitType.BUILD: "Update build for",
            CommitType.CI: "Update CI for",
            CommitType.CHORE: "Update",
            CommitType.REVERT: "Revert",
        }.get(commit_type, "Update")


# =============================================================================
# TASK EXTRACTION
# =============================================================================

class TaskPriority(Enum):
    """Task priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ExtractedTask:
    """A task extracted from conversation."""
    title: str
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    due_date: Optional[str] = None
    assignee: Optional[str] = None
    source_message: str = ""
    confidence: float = 0.5
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "due_date": self.due_date,
            "assignee": self.assignee,
            "tags": self.tags,
            "confidence": self.confidence
        }


class TaskExtractor:
    """
    Extract action items and tasks from conversations.
    
    Features:
    - Detect action verbs (todo, need to, should, must)
    - Extract deadlines and dates
    - Prioritize by urgency keywords
    - Group related tasks
    """
    
    # Action verb patterns
    ACTION_PATTERNS = [
        r"(?:I |we |you )?(?:need to|have to|must|should|will|gonna|going to) (.+?)(?:\.|$)",
        r"(?:I |we |you )?(?:TODO|FIXME|HACK)[ :]+(.+?)(?:\.|$)",
        r"(?:remember to|don't forget to|make sure to) (.+?)(?:\.|$)",
        r"(?:action item|task)[ :]+(.+?)(?:\.|$)",
        r"(?:let me|let's|I'll) (.+?)(?:\.|$)",
    ]
    
    # Priority keywords
    PRIORITY_KEYWORDS = {
        TaskPriority.HIGH: ["urgent", "asap", "critical", "important", "immediately", "today"],
        TaskPriority.MEDIUM: ["soon", "when possible", "this week"],
        TaskPriority.LOW: ["eventually", "someday", "later", "when you have time"],
    }
    
    # Date patterns
    DATE_PATTERNS = [
        r"by (tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"by (\d{1,2}/\d{1,2}(?:/\d{2,4})?)",
        r"by (next week|next month|end of (?:week|month|day))",
        r"(this week|this month|this weekend)",
    ]
    
    def __init__(self):
        """Initialize task extractor."""
        self._action_regexes = [
            re.compile(p, re.IGNORECASE) for p in self.ACTION_PATTERNS
        ]
        self._date_regexes = [
            re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS
        ]
    
    def extract_tasks(
        self,
        messages: List[Dict[str, str]],
        min_confidence: float = 0.3
    ) -> List[ExtractedTask]:
        """
        Extract tasks from conversation messages.
        
        Args:
            messages: List of conversation messages
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of extracted tasks
        """
        tasks = []
        
        for msg in messages:
            content = msg.get("content", msg.get("text", ""))
            role = msg.get("role", "")
            
            # Focus on user messages and AI responses with action items
            if role not in ["user", "assistant", "ai"]:
                continue
            
            msg_tasks = self._extract_from_text(content)
            for task in msg_tasks:
                if task.confidence >= min_confidence:
                    task.source_message = content[:100]
                    tasks.append(task)
        
        return self._deduplicate_tasks(tasks)
    
    def _extract_from_text(self, text: str) -> List[ExtractedTask]:
        """Extract tasks from a single text."""
        tasks = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)
        
        for sentence in sentences:
            for regex in self._action_regexes:
                match = regex.search(sentence)
                if match:
                    task_text = match.group(1).strip()
                    if len(task_text) > 10:  # Filter out too-short matches
                        task = ExtractedTask(
                            title=task_text[:100],
                            description=sentence,
                            confidence=0.6
                        )
                        
                        # Detect priority
                        task.priority = self._detect_priority(sentence)
                        
                        # Detect due date
                        task.due_date = self._detect_date(sentence)
                        
                        # Adjust confidence
                        if task.due_date:
                            task.confidence += 0.1
                        if task.priority == TaskPriority.HIGH:
                            task.confidence += 0.1
                        
                        tasks.append(task)
                        break  # One task per sentence
        
        return tasks
    
    def _detect_priority(self, text: str) -> TaskPriority:
        """Detect task priority from text."""
        text_lower = text.lower()
        
        for priority, keywords in self.PRIORITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return priority
        
        return TaskPriority.MEDIUM
    
    def _detect_date(self, text: str) -> Optional[str]:
        """Detect due date from text."""
        for regex in self._date_regexes:
            match = regex.search(text)
            if match:
                return match.group(1)
        return None
    
    def _deduplicate_tasks(self, tasks: List[ExtractedTask]) -> List[ExtractedTask]:
        """Remove duplicate tasks."""
        seen_titles = set()
        unique_tasks = []
        
        for task in tasks:
            # Normalize title for comparison
            normalized = task.title.lower().strip()
            if normalized not in seen_titles:
                seen_titles.add(normalized)
                unique_tasks.append(task)
        
        return unique_tasks
    
    def format_task_list(
        self,
        tasks: List[ExtractedTask],
        format_type: str = "markdown"
    ) -> str:
        """
        Format tasks as a list.
        
        Args:
            tasks: List of tasks
            format_type: "markdown", "text", or "html"
        """
        if format_type == "markdown":
            lines = ["## Tasks\n"]
            for task in tasks:
                priority_emoji = {"high": "!", "medium": "-", "low": "?"}
                emoji = priority_emoji.get(task.priority.value, "-")
                line = f"- [{emoji}] {task.title}"
                if task.due_date:
                    line += f" (due: {task.due_date})"
                lines.append(line)
            return "\n".join(lines)
        
        elif format_type == "text":
            lines = ["Tasks:", ""]
            for i, task in enumerate(tasks, 1):
                line = f"{i}. {task.title}"
                if task.due_date:
                    line += f" - Due: {task.due_date}"
                lines.append(line)
            return "\n".join(lines)
        
        return str(tasks)


# =============================================================================
# SQL QUERY BUILDER
# =============================================================================

class SQLDialect(Enum):
    """SQL dialects."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MSSQL = "mssql"


@dataclass
class SQLQuery:
    """A generated SQL query."""
    query: str
    dialect: SQLDialect = SQLDialect.SQLITE
    explanation: str = ""
    tables_used: List[str] = field(default_factory=list)
    confidence: float = 0.5


class SQLQueryBuilder:
    """
    Natural language to SQL query builder.
    
    Features:
    - Parse natural language queries
    - Generate SELECT, INSERT, UPDATE queries
    - Schema-aware generation
    - Multiple SQL dialects
    """
    
    # Common query patterns
    PATTERNS = {
        "select_all": [
            r"(?:show|get|list|display|find)(?: all)? (\w+)",
            r"(?:what are|show me)(?: all)?(?: the)? (\w+)",
        ],
        "select_where": [
            r"(?:show|get|find) (\w+) where (\w+) (?:is|=|equals?) (.+)",
            r"(?:show|get|find) (\w+) with (\w+) (.+)",
        ],
        "count": [
            r"(?:how many|count) (\w+)",
            r"(?:number of|total) (\w+)",
        ],
        "top_n": [
            r"(?:top|first) (\d+) (\w+)",
            r"(?:show|get) (\d+) (\w+)",
        ],
        "order_by": [
            r"(\w+) (?:sorted|ordered) by (\w+)(?: (asc|desc))?",
            r"(\w+) by (\w+)(?: (highest|lowest|asc|desc))?",
        ],
    }
    
    def __init__(
        self,
        schema: Optional[Dict[str, List[str]]] = None,
        dialect: SQLDialect = SQLDialect.SQLITE
    ):
        """
        Initialize SQL builder.
        
        Args:
            schema: Table schema {table_name: [column_names]}
            dialect: SQL dialect to use
        """
        self.schema = schema or {}
        self.dialect = dialect
        
        # Compile patterns
        self._patterns: Dict[str, List[re.Pattern]] = {}
        for pattern_type, patterns in self.PATTERNS.items():
            self._patterns[pattern_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def set_schema(self, schema: Dict[str, List[str]]):
        """Set database schema."""
        self.schema = schema
    
    def build_query(self, natural_language: str) -> SQLQuery:
        """
        Build SQL query from natural language.
        
        Args:
            natural_language: Natural language query
            
        Returns:
            Generated SQL query
        """
        nl_lower = natural_language.lower()
        
        # Try each pattern type
        for pattern_type, regexes in self._patterns.items():
            for regex in regexes:
                match = regex.search(nl_lower)
                if match:
                    return self._build_from_pattern(pattern_type, match)
        
        # Fallback: try to extract table name and build simple SELECT
        table = self._find_table(nl_lower)
        if table:
            return SQLQuery(
                query=f"SELECT * FROM {table}",
                dialect=self.dialect,
                explanation=f"List all records from {table}",
                tables_used=[table],
                confidence=0.4
            )
        
        return SQLQuery(
            query="-- Could not parse query",
            explanation="Unable to understand the natural language query",
            confidence=0.0
        )
    
    def _build_from_pattern(
        self,
        pattern_type: str,
        match: re.Match
    ) -> SQLQuery:
        """Build query from matched pattern."""
        
        if pattern_type == "select_all":
            table = self._normalize_table(match.group(1))
            return SQLQuery(
                query=f"SELECT * FROM {table}",
                dialect=self.dialect,
                explanation=f"List all records from {table}",
                tables_used=[table],
                confidence=0.8
            )
        
        elif pattern_type == "select_where":
            table = self._normalize_table(match.group(1))
            column = match.group(2)
            value = match.group(3).strip("'\"")
            
            # Quote string values
            if not value.isdigit():
                value = f"'{value}'"
            
            return SQLQuery(
                query=f"SELECT * FROM {table} WHERE {column} = {value}",
                dialect=self.dialect,
                explanation=f"Find {table} where {column} equals {value}",
                tables_used=[table],
                confidence=0.7
            )
        
        elif pattern_type == "count":
            table = self._normalize_table(match.group(1))
            return SQLQuery(
                query=f"SELECT COUNT(*) FROM {table}",
                dialect=self.dialect,
                explanation=f"Count total records in {table}",
                tables_used=[table],
                confidence=0.9
            )
        
        elif pattern_type == "top_n":
            limit = int(match.group(1))
            table = self._normalize_table(match.group(2))
            
            if self.dialect == SQLDialect.MSSQL:
                query = f"SELECT TOP {limit} * FROM {table}"
            else:
                query = f"SELECT * FROM {table} LIMIT {limit}"
            
            return SQLQuery(
                query=query,
                dialect=self.dialect,
                explanation=f"Get first {limit} records from {table}",
                tables_used=[table],
                confidence=0.8
            )
        
        elif pattern_type == "order_by":
            table = self._normalize_table(match.group(1))
            column = match.group(2)
            direction = match.group(3) if len(match.groups()) > 2 else "ASC"
            
            if direction:
                direction = direction.upper()
                if direction in ["HIGHEST", "DESC"]:
                    direction = "DESC"
                else:
                    direction = "ASC"
            else:
                direction = "ASC"
            
            return SQLQuery(
                query=f"SELECT * FROM {table} ORDER BY {column} {direction}",
                dialect=self.dialect,
                explanation=f"List {table} sorted by {column} {direction.lower()}",
                tables_used=[table],
                confidence=0.7
            )
        
        return SQLQuery(query="-- Pattern not implemented", confidence=0.0)
    
    def _normalize_table(self, name: str) -> str:
        """Normalize table name and find best match in schema."""
        name = name.strip().lower()
        
        # Remove common suffixes
        if name.endswith("s") and name[:-1] in self.schema:
            return name[:-1]
        
        # Check schema
        if name in self.schema:
            return name
        
        # Find similar
        for table in self.schema:
            if table.lower() == name or table.lower() == name + "s":
                return table
        
        return name
    
    def _find_table(self, text: str) -> Optional[str]:
        """Find table name in text."""
        for table in self.schema:
            if table.lower() in text:
                return table
        return None
    
    def suggest_queries(self, table: str) -> List[SQLQuery]:
        """Suggest common queries for a table."""
        suggestions = [
            SQLQuery(
                query=f"SELECT * FROM {table}",
                explanation=f"Get all {table}",
                confidence=1.0
            ),
            SQLQuery(
                query=f"SELECT COUNT(*) FROM {table}",
                explanation=f"Count {table}",
                confidence=1.0
            ),
            SQLQuery(
                query=f"SELECT * FROM {table} LIMIT 10",
                explanation=f"Get first 10 {table}",
                confidence=1.0
            ),
        ]
        
        if table in self.schema and self.schema[table]:
            first_col = self.schema[table][0]
            suggestions.append(SQLQuery(
                query=f"SELECT * FROM {table} ORDER BY {first_col}",
                explanation=f"List {table} ordered by {first_col}",
                confidence=0.8
            ))
        
        return suggestions


# =============================================================================
# MODEL SWITCHING FOR RETRY
# =============================================================================

@dataclass
class ModelOption:
    """A model available for selection."""
    name: str
    display_name: str
    description: str = ""
    size: str = ""  # "small", "medium", "large"
    capabilities: List[str] = field(default_factory=list)
    is_local: bool = True
    api_key_required: bool = False


class ModelSwitcher:
    """
    Quick model switching for retrying responses.
    
    Features:
    - List available models
    - Switch models mid-conversation
    - Model comparison
    - Quick retry with different model
    """
    
    # Default model options
    DEFAULT_MODELS = [
        ModelOption("forge_small", "Forge Small", "Fast local model", "small", ["chat"], True),
        ModelOption("forge_medium", "Forge Medium", "Balanced local model", "medium", ["chat", "code"], True),
        ModelOption("forge_large", "Forge Large", "Quality local model", "large", ["chat", "code", "reasoning"], True),
    ]
    
    def __init__(self, models: Optional[List[ModelOption]] = None):
        """
        Initialize model switcher.
        
        Args:
            models: List of available models
        """
        self.models = {m.name: m for m in (models or self.DEFAULT_MODELS)}
        self.current_model: Optional[str] = None
        self._switch_callbacks: List[Callable[[str, str], None]] = []
    
    def get_available_models(self) -> List[ModelOption]:
        """Get all available models."""
        return list(self.models.values())
    
    def get_model(self, name: str) -> Optional[ModelOption]:
        """Get a model by name."""
        return self.models.get(name)
    
    def add_model(self, model: ModelOption):
        """Add a model to available models."""
        self.models[model.name] = model
    
    def switch_model(self, new_model: str) -> bool:
        """
        Switch to a different model.
        
        Args:
            new_model: Name of model to switch to
            
        Returns:
            True if switch successful
        """
        if new_model not in self.models:
            return False
        
        old_model = self.current_model
        self.current_model = new_model
        
        # Notify callbacks
        for callback in self._switch_callbacks:
            try:
                callback(old_model or "", new_model)
            except Exception as e:
                logger.warning(f"Model switch callback error: {e}")
        
        return True
    
    def on_switch(self, callback: Callable[[str, str], None]):
        """Register callback for model switches."""
        self._switch_callbacks.append(callback)
    
    def get_models_for_capability(self, capability: str) -> List[ModelOption]:
        """Get models that have a specific capability."""
        return [m for m in self.models.values() if capability in m.capabilities]
    
    def suggest_model_for_task(self, task_type: str) -> Optional[ModelOption]:
        """
        Suggest best model for a task.
        
        Args:
            task_type: "chat", "code", "reasoning", etc.
        """
        # Find models with capability, prefer smaller for speed
        capable = self.get_models_for_capability(task_type)
        if not capable:
            capable = list(self.models.values())
        
        # Sort by size (smaller first for speed)
        size_order = {"small": 0, "medium": 1, "large": 2}
        capable.sort(key=lambda m: size_order.get(m.size, 1))
        
        return capable[0] if capable else None


# =============================================================================
# COMBINED PRODUCTIVITY TOOLS
# =============================================================================

class ProductivityTools:
    """
    Combined productivity tools manager.
    
    Combines:
    - Commit message generator
    - Task extractor
    - SQL query builder
    - Model switcher
    """
    
    def __init__(self, repo_path: Optional[Path] = None):
        """
        Initialize productivity tools.
        
        Args:
            repo_path: Path to git repository
        """
        self.commit_gen = CommitMessageGenerator(repo_path)
        self.task_extractor = TaskExtractor()
        self.sql_builder = SQLQueryBuilder()
        self.model_switcher = ModelSwitcher()


# Singleton
_productivity_tools: Optional[ProductivityTools] = None


def get_productivity_tools(repo_path: Optional[Path] = None) -> ProductivityTools:
    """Get or create productivity tools."""
    global _productivity_tools
    if _productivity_tools is None:
        _productivity_tools = ProductivityTools(repo_path)
    return _productivity_tools
