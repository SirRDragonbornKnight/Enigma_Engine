"""
Search and Filter Utilities - Advanced search and filtering for data.

Features:
- Full-text search with ranking
- Multi-field filtering
- Sort expressions
- Query parsing
- Fuzzy matching

Part of the Enigma AI Engine data processing suite.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# FILTER OPERATORS
# =============================================================================

class FilterOp(Enum):
    """Filter comparison operators."""
    EQ = "eq"           # Equal
    NE = "ne"           # Not equal
    GT = "gt"           # Greater than
    GTE = "gte"         # Greater than or equal
    LT = "lt"           # Less than
    LTE = "lte"         # Less than or equal
    IN = "in"           # In list
    NIN = "nin"         # Not in list
    CONTAINS = "contains"      # Contains substring
    STARTS = "starts"          # Starts with
    ENDS = "ends"              # Ends with
    REGEX = "regex"            # Regex match
    EXISTS = "exists"          # Field exists
    RANGE = "range"            # Between two values


class SortDir(Enum):
    """Sort direction."""
    ASC = "asc"
    DESC = "desc"


# =============================================================================
# FILTER CONDITION
# =============================================================================

@dataclass
class FilterCondition:
    """A single filter condition."""
    field: str
    operator: FilterOp
    value: Any
    case_sensitive: bool = False
    
    def matches(self, item: Any) -> bool:
        """Check if item matches this condition."""
        # Get field value
        field_value = self._get_field(item, self.field)
        
        if self.operator == FilterOp.EXISTS:
            return (field_value is not None) == self.value
        
        if field_value is None:
            return False
        
        # Normalize strings for case-insensitive comparison
        compare_value = self.value
        if not self.case_sensitive and isinstance(field_value, str):
            field_value = field_value.lower()
            if isinstance(compare_value, str):
                compare_value = compare_value.lower()
            elif isinstance(compare_value, list):
                compare_value = [v.lower() if isinstance(v, str) else v for v in compare_value]
        
        # Apply operator
        if self.operator == FilterOp.EQ:
            return field_value == compare_value
        elif self.operator == FilterOp.NE:
            return field_value != compare_value
        elif self.operator == FilterOp.GT:
            return field_value > compare_value
        elif self.operator == FilterOp.GTE:
            return field_value >= compare_value
        elif self.operator == FilterOp.LT:
            return field_value < compare_value
        elif self.operator == FilterOp.LTE:
            return field_value <= compare_value
        elif self.operator == FilterOp.IN:
            return field_value in compare_value
        elif self.operator == FilterOp.NIN:
            return field_value not in compare_value
        elif self.operator == FilterOp.CONTAINS:
            return compare_value in str(field_value)
        elif self.operator == FilterOp.STARTS:
            return str(field_value).startswith(str(compare_value))
        elif self.operator == FilterOp.ENDS:
            return str(field_value).endswith(str(compare_value))
        elif self.operator == FilterOp.REGEX:
            try:
                flags = 0 if self.case_sensitive else re.IGNORECASE
                return bool(re.search(compare_value, str(field_value), flags))
            except re.error:
                return False
        elif self.operator == FilterOp.RANGE:
            if isinstance(compare_value, (list, tuple)) and len(compare_value) == 2:
                return compare_value[0] <= field_value <= compare_value[1]
            return False
        
        return False
    
    def _get_field(self, item: Any, field_path: str) -> Any:
        """Get field value from item, supporting dot notation."""
        parts = field_path.split(".")
        value = item
        
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            elif hasattr(value, "__getitem__"):
                try:
                    value = value[part]
                except (KeyError, IndexError, TypeError):
                    return None
            else:
                return None
            
            if value is None:
                return None
        
        return value


# =============================================================================
# FILTER BUILDER
# =============================================================================

class Filter(Generic[T]):
    """
    Fluent filter builder.
    
    Example:
        filter = Filter[User]()
        filter.where("age").gte(18).and_where("status").eq("active")
        results = filter.apply(users)
    """
    
    def __init__(self):
        """Initialize filter."""
        self._conditions: list[FilterCondition] = []
        self._or_groups: list[list[FilterCondition]] = []
        self._current_field: Optional[str] = None
        self._case_sensitive: bool = False
    
    def where(self, field: str) -> "Filter[T]":
        """Start a filter condition on a field."""
        self._current_field = field
        return self
    
    def and_where(self, field: str) -> "Filter[T]":
        """Add AND condition."""
        return self.where(field)
    
    def or_where(self, field: str) -> "Filter[T]":
        """Start OR group."""
        if self._conditions:
            self._or_groups.append(self._conditions.copy())
            self._conditions = []
        self._current_field = field
        return self
    
    def case_sensitive(self, enabled: bool = True) -> "Filter[T]":
        """Set case sensitivity for string comparisons."""
        self._case_sensitive = enabled
        return self
    
    def _add_condition(self, op: FilterOp, value: Any) -> "Filter[T]":
        """Add a condition."""
        if self._current_field:
            self._conditions.append(FilterCondition(
                field=self._current_field,
                operator=op,
                value=value,
                case_sensitive=self._case_sensitive
            ))
        return self
    
    def eq(self, value: Any) -> "Filter[T]":
        """Equal to."""
        return self._add_condition(FilterOp.EQ, value)
    
    def ne(self, value: Any) -> "Filter[T]":
        """Not equal to."""
        return self._add_condition(FilterOp.NE, value)
    
    def gt(self, value: Any) -> "Filter[T]":
        """Greater than."""
        return self._add_condition(FilterOp.GT, value)
    
    def gte(self, value: Any) -> "Filter[T]":
        """Greater than or equal."""
        return self._add_condition(FilterOp.GTE, value)
    
    def lt(self, value: Any) -> "Filter[T]":
        """Less than."""
        return self._add_condition(FilterOp.LT, value)
    
    def lte(self, value: Any) -> "Filter[T]":
        """Less than or equal."""
        return self._add_condition(FilterOp.LTE, value)
    
    def is_in(self, values: list[Any]) -> "Filter[T]":
        """Value in list."""
        return self._add_condition(FilterOp.IN, values)
    
    def not_in(self, values: list[Any]) -> "Filter[T]":
        """Value not in list."""
        return self._add_condition(FilterOp.NIN, values)
    
    def contains(self, value: str) -> "Filter[T]":
        """Contains substring."""
        return self._add_condition(FilterOp.CONTAINS, value)
    
    def starts_with(self, value: str) -> "Filter[T]":
        """Starts with."""
        return self._add_condition(FilterOp.STARTS, value)
    
    def ends_with(self, value: str) -> "Filter[T]":
        """Ends with."""
        return self._add_condition(FilterOp.ENDS, value)
    
    def matches(self, pattern: str) -> "Filter[T]":
        """Matches regex pattern."""
        return self._add_condition(FilterOp.REGEX, pattern)
    
    def exists(self, exists: bool = True) -> "Filter[T]":
        """Field exists."""
        return self._add_condition(FilterOp.EXISTS, exists)
    
    def between(self, min_val: Any, max_val: Any) -> "Filter[T]":
        """Value between range."""
        return self._add_condition(FilterOp.RANGE, [min_val, max_val])
    
    def apply(self, items: list[T]) -> list[T]:
        """Apply filter to items."""
        # Collect all OR groups
        all_groups = self._or_groups.copy()
        if self._conditions:
            all_groups.append(self._conditions)
        
        if not all_groups:
            return items
        
        results = []
        for item in items:
            # Item matches if ANY OR group matches (all conditions in group must match)
            matches_any_group = False
            for group in all_groups:
                matches_all_in_group = all(cond.matches(item) for cond in group)
                if matches_all_in_group:
                    matches_any_group = True
                    break
            
            if matches_any_group:
                results.append(item)
        
        return results


# =============================================================================
# SORT BUILDER
# =============================================================================

@dataclass
class SortSpec:
    """Sort specification."""
    field: str
    direction: SortDir = SortDir.ASC
    nulls_last: bool = True


class Sorter(Generic[T]):
    """
    Fluent sort builder.
    
    Example:
        sorter = Sorter[User]()
        sorter.by("name").then_by("created_at", desc=True)
        sorted_items = sorter.apply(users)
    """
    
    def __init__(self):
        """Initialize sorter."""
        self._specs: list[SortSpec] = []
    
    def by(self, field: str, desc: bool = False) -> "Sorter[T]":
        """Sort by field."""
        self._specs.append(SortSpec(
            field=field,
            direction=SortDir.DESC if desc else SortDir.ASC
        ))
        return self
    
    def then_by(self, field: str, desc: bool = False) -> "Sorter[T]":
        """Additional sort field."""
        return self.by(field, desc)
    
    def asc(self, field: str) -> "Sorter[T]":
        """Sort ascending."""
        return self.by(field, desc=False)
    
    def desc(self, field: str) -> "Sorter[T]":
        """Sort descending."""
        return self.by(field, desc=True)
    
    def apply(self, items: list[T]) -> list[T]:
        """Apply sort to items."""
        if not self._specs:
            return items
        
        def get_sort_key(item: T) -> tuple:
            values = []
            for spec in self._specs:
                value = self._get_field(item, spec.field)
                
                # Handle None values
                if value is None:
                    if spec.nulls_last:
                        value = (1, None)  # Sort after non-null
                    else:
                        value = (0, None)  # Sort before non-null
                else:
                    value = (0, value) if spec.nulls_last else (1, value)
                
                # Reverse for descending
                if spec.direction == SortDir.DESC:
                    if isinstance(value[1], (int, float)):
                        value = (value[0], -value[1])
                    elif isinstance(value[1], str):
                        # For strings, we'll handle in the sort
                        pass
                
                values.append(value)
            
            return tuple(values)
        
        # Sort with reverse for desc string handling
        result = sorted(items, key=get_sort_key)
        
        return result
    
    def _get_field(self, item: Any, field_path: str) -> Any:
        """Get field value."""
        parts = field_path.split(".")
        value = item
        
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None
            
            if value is None:
                return None
        
        return value


# =============================================================================
# FULL-TEXT SEARCH
# =============================================================================

@dataclass
class SearchResult(Generic[T]):
    """A search result with score."""
    item: T
    score: float
    matches: dict[str, list[str]] = field(default_factory=dict)


class TextSearch(Generic[T]):
    """
    Full-text search with ranking.
    
    Example:
        search = TextSearch[Article](fields=["title", "content"])
        results = search.search("python tutorial", articles)
    """
    
    def __init__(
        self,
        fields: list[str],
        field_weights: Optional[dict[str, float]] = None,
        min_score: float = 0.0
    ):
        """
        Initialize text search.
        
        Args:
            fields: Fields to search
            field_weights: Weight multipliers per field
            min_score: Minimum score to include
        """
        self.fields = fields
        self.field_weights = field_weights or {}
        self.min_score = min_score
    
    def search(
        self,
        query: str,
        items: list[T],
        limit: Optional[int] = None,
        fuzzy: bool = False
    ) -> list[SearchResult[T]]:
        """
        Search items for query.
        
        Args:
            query: Search query
            items: Items to search
            limit: Max results
            fuzzy: Enable fuzzy matching
            
        Returns:
            Ranked search results
        """
        if not query.strip():
            return []
        
        # Tokenize query
        query_terms = self._tokenize(query)
        
        results = []
        for item in items:
            score, matches = self._score_item(item, query_terms, fuzzy)
            
            if score >= self.min_score:
                results.append(SearchResult(
                    item=item,
                    score=score,
                    matches=matches
                ))
        
        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        
        if limit:
            results = results[:limit]
        
        return results
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into search terms."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        terms = re.split(r"[^\w]+", text)
        return [t for t in terms if t and len(t) > 1]
    
    def _score_item(
        self,
        item: T,
        query_terms: list[str],
        fuzzy: bool
    ) -> tuple[float, dict[str, list[str]]]:
        """Score an item against query terms."""
        total_score = 0.0
        all_matches: dict[str, list[str]] = {}
        
        for field_name in self.fields:
            field_value = self._get_field(item, field_name)
            if not field_value:
                continue
            
            field_text = str(field_value).lower()
            field_weight = self.field_weights.get(field_name, 1.0)
            
            field_matches = []
            field_score = 0.0
            
            for term in query_terms:
                if term in field_text:
                    # Exact match
                    field_score += 1.0 * field_weight
                    field_matches.append(term)
                    
                    # Bonus for word boundary match
                    if re.search(rf"\b{re.escape(term)}\b", field_text):
                        field_score += 0.5 * field_weight
                    
                    # Bonus for title/start match
                    if field_text.startswith(term):
                        field_score += 0.3 * field_weight
                        
                elif fuzzy:
                    # Fuzzy matching
                    fuzzy_score = self._fuzzy_match(term, field_text)
                    if fuzzy_score > 0.6:
                        field_score += fuzzy_score * 0.5 * field_weight
                        field_matches.append(f"~{term}")
            
            total_score += field_score
            if field_matches:
                all_matches[field_name] = field_matches
        
        return total_score, all_matches
    
    def _fuzzy_match(self, term: str, text: str) -> float:
        """Simple fuzzy matching using character overlap."""
        if len(term) < 3:
            return 0.0
        
        # Check for partial matches
        for i in range(len(text) - len(term) + 1):
            window = text[i:i + len(term)]
            matching = sum(1 for a, b in zip(term, window) if a == b)
            ratio = matching / len(term)
            if ratio > 0.7:
                return ratio
        
        return 0.0
    
    def _get_field(self, item: Any, field_path: str) -> Any:
        """Get field value."""
        parts = field_path.split(".")
        value = item
        
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None
            
            if value is None:
                return None
        
        return value


# =============================================================================
# QUERY PARSER
# =============================================================================

class QueryParser:
    """
    Parse search query strings into structured queries.
    
    Supports:
    - field:value syntax (status:active)
    - Quoted phrases ("exact match")
    - Operators (> < >= <= !=)
    - AND/OR keywords
    """
    
    # Regex patterns
    FIELD_VALUE = re.compile(r'(\w+):([^\s"]+|"[^"]*")')
    QUOTED = re.compile(r'"([^"]*)"')
    OPERATOR = re.compile(r'(\w+)(>=|<=|!=|>|<)(\S+)')
    
    def parse(self, query: str) -> dict[str, Any]:
        """
        Parse query string.
        
        Args:
            query: Query string
            
        Returns:
            Parsed query dict with fields, terms, and operators
        """
        result = {
            "fields": {},
            "terms": [],
            "operators": [],
            "raw": query
        }
        
        remaining = query
        
        # Extract field:value pairs
        for match in self.FIELD_VALUE.finditer(query):
            field, value = match.groups()
            value = value.strip('"')
            result["fields"][field] = value
            remaining = remaining.replace(match.group(), "")
        
        # Extract operators
        for match in self.OPERATOR.finditer(remaining):
            field, op, value = match.groups()
            result["operators"].append({
                "field": field,
                "op": op,
                "value": value
            })
            remaining = remaining.replace(match.group(), "")
        
        # Extract quoted phrases
        for match in self.QUOTED.finditer(remaining):
            result["terms"].append(match.group(1))
            remaining = remaining.replace(match.group(), "")
        
        # Remaining words are plain terms
        words = remaining.split()
        for word in words:
            word = word.strip()
            if word and word.lower() not in ("and", "or"):
                result["terms"].append(word)
        
        return result
    
    def to_filter(self, parsed: dict[str, Any]) -> Filter:
        """Convert parsed query to Filter."""
        f = Filter()
        
        # Add field conditions
        for field, value in parsed["fields"].items():
            f.where(field).eq(value)
        
        # Add operator conditions
        for op_spec in parsed["operators"]:
            field = op_spec["field"]
            op = op_spec["op"]
            value = op_spec["value"]
            
            # Try to convert numeric values
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass
            
            f.where(field)
            if op == ">":
                f.gt(value)
            elif op == ">=":
                f.gte(value)
            elif op == "<":
                f.lt(value)
            elif op == "<=":
                f.lte(value)
            elif op == "!=":
                f.ne(value)
        
        return f


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def filter_items(items: list[T], **conditions) -> list[T]:
    """
    Quick filter function.
    
    Args:
        items: Items to filter
        **conditions: Field=value conditions
        
    Returns:
        Filtered items
    """
    f = Filter()
    for field, value in conditions.items():
        f.where(field).eq(value)
    return f.apply(items)


def sort_items(items: list[T], *fields: str, desc: bool = False) -> list[T]:
    """
    Quick sort function.
    
    Args:
        items: Items to sort
        *fields: Fields to sort by
        desc: Sort descending
        
    Returns:
        Sorted items
    """
    s = Sorter()
    for field in fields:
        s.by(field, desc=desc)
    return s.apply(items)


def search_items(
    items: list[T],
    query: str,
    fields: list[str],
    limit: int = 20
) -> list[T]:
    """
    Quick search function.
    
    Args:
        items: Items to search
        query: Search query
        fields: Fields to search
        limit: Max results
        
    Returns:
        Matching items (sorted by relevance)
    """
    searcher = TextSearch(fields=fields)
    results = searcher.search(query, items, limit=limit)
    return [r.item for r in results]
