"""
Knowledge Graph Memory for Enigma AI Engine

Build and query knowledge graphs from conversations.

Features:
- Entity extraction from text
- Relationship mapping
- Graph-based reasoning
- Multi-hop queries
- Temporal knowledge tracking
- Integration with vector memory

Usage:
    from enigma_engine.memory.knowledge_graph import KnowledgeGraph, get_graph
    
    kg = KnowledgeGraph()
    
    # Extract and add knowledge
    kg.add_from_text("Alice works at OpenAI. Bob is Alice's friend.")
    
    # Query
    results = kg.query("Who works at OpenAI?")
    relations = kg.get_relations("Alice")
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of knowledge entities."""
    PERSON = auto()
    ORGANIZATION = auto()
    LOCATION = auto()
    CONCEPT = auto()
    EVENT = auto()
    OBJECT = auto()
    TIME = auto()
    UNKNOWN = auto()


class RelationType(Enum):
    """Types of relationships."""
    # Work/Professional
    WORKS_AT = "works_at"
    MANAGES = "manages"
    REPORTS_TO = "reports_to"
    COLLABORATES_WITH = "collaborates_with"
    
    # Social
    FRIEND_OF = "friend_of"
    FAMILY_OF = "family_of"
    KNOWS = "knows"
    
    # Location
    LOCATED_IN = "located_in"
    LIVES_IN = "lives_in"
    BORN_IN = "born_in"
    
    # Ownership/Association
    OWNS = "owns"
    PART_OF = "part_of"
    MEMBER_OF = "member_of"
    CREATED_BY = "created_by"
    
    # Properties
    HAS_PROPERTY = "has_property"
    IS_A = "is_a"
    RELATED_TO = "related_to"
    
    # Temporal
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"


@dataclass
class Entity:
    """A knowledge entity."""
    id: str
    name: str
    entity_type: EntityType = EntityType.UNKNOWN
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Aliases
    aliases: List[str] = field(default_factory=list)
    
    # Tracking
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    mention_count: int = 1
    
    # Confidence
    confidence: float = 1.0
    
    # Source
    sources: List[str] = field(default_factory=list)


@dataclass
class Relation:
    """A relationship between entities."""
    source_id: str
    target_id: str
    relation_type: RelationType
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence and source
    confidence: float = 1.0
    source: str = ""
    
    # Temporal
    timestamp: datetime = field(default_factory=datetime.now)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None


@dataclass
class QueryResult:
    """Result from a knowledge graph query."""
    entities: List[Entity]
    relations: List[Relation]
    paths: List[List[str]]  # Entity ID paths
    confidence: float = 1.0
    
    def __bool__(self):
        return bool(self.entities or self.relations)


class KnowledgeGraph:
    """
    Knowledge graph for storing and querying facts.
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize knowledge graph.
        
        Args:
            persist_path: Optional path for persistence
        """
        self._persist_path = Path(persist_path) if persist_path else None
        
        # Storage
        self._entities: Dict[str, Entity] = {}
        self._relations: List[Relation] = []
        
        # Indices
        self._name_index: Dict[str, str] = {}  # name/alias -> entity_id
        self._type_index: Dict[EntityType, Set[str]] = defaultdict(set)
        self._relation_index: Dict[str, List[int]] = defaultdict(list)  # entity_id -> relation indices
        
        # Extraction patterns
        self._relation_patterns = self._build_relation_patterns()
        
        # Load if exists
        if self._persist_path and self._persist_path.exists():
            self.load()
    
    def _build_relation_patterns(self) -> List[Tuple[str, RelationType]]:
        """Build patterns for relation extraction."""
        return [
            # Work relations
            (r"(\w+) works at (\w+)", RelationType.WORKS_AT),
            (r"(\w+) is employed by (\w+)", RelationType.WORKS_AT),
            (r"(\w+) manages (\w+)", RelationType.MANAGES),
            (r"(\w+) reports to (\w+)", RelationType.REPORTS_TO),
            
            # Social relations
            (r"(\w+) is (\w+)'s friend", RelationType.FRIEND_OF),
            (r"(\w+) and (\w+) are friends", RelationType.FRIEND_OF),
            (r"(\w+) knows (\w+)", RelationType.KNOWS),
            (r"(\w+) is married to (\w+)", RelationType.FAMILY_OF),
            (r"(\w+) is (\w+)'s (brother|sister|parent|child)", RelationType.FAMILY_OF),
            
            # Location
            (r"(\w+) lives in (\w+)", RelationType.LIVES_IN),
            (r"(\w+) is located in (\w+)", RelationType.LOCATED_IN),
            (r"(\w+) was born in (\w+)", RelationType.BORN_IN),
            
            # Ownership/Association
            (r"(\w+) owns (\w+)", RelationType.OWNS),
            (r"(\w+) is part of (\w+)", RelationType.PART_OF),
            (r"(\w+) is a member of (\w+)", RelationType.MEMBER_OF),
            (r"(\w+) created (\w+)", RelationType.CREATED_BY),
            (r"(\w+) was created by (\w+)", RelationType.CREATED_BY),
            
            # Type relations
            (r"(\w+) is a (\w+)", RelationType.IS_A),
            (r"(\w+) is an (\w+)", RelationType.IS_A),
        ]
    
    def add_entity(
        self,
        name: str,
        entity_type: EntityType = EntityType.UNKNOWN,
        properties: Optional[Dict[str, Any]] = None,
        source: str = ""
    ) -> Entity:
        """
        Add or update an entity.
        
        Args:
            name: Entity name
            entity_type: Type of entity
            properties: Optional properties
            source: Source of information
            
        Returns:
            Entity object
        """
        # Check if exists
        entity_id = self._name_index.get(name.lower())
        
        if entity_id and entity_id in self._entities:
            # Update existing
            entity = self._entities[entity_id]
            entity.mention_count += 1
            entity.last_seen = datetime.now()
            
            if properties:
                entity.properties.update(properties)
            if source and source not in entity.sources:
                entity.sources.append(source)
            
            # Update type if more specific
            if entity_type != EntityType.UNKNOWN and entity.entity_type == EntityType.UNKNOWN:
                entity.entity_type = entity_type
                self._type_index[entity_type].add(entity_id)
            
            return entity
        
        # Create new
        entity_id = self._generate_id(name)
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            sources=[source] if source else []
        )
        
        self._entities[entity_id] = entity
        self._name_index[name.lower()] = entity_id
        self._type_index[entity_type].add(entity_id)
        
        return entity
    
    def add_relation(
        self,
        source_name: str,
        target_name: str,
        relation_type: RelationType | str,
        properties: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        source: str = ""
    ) -> Relation:
        """
        Add a relationship between entities.
        
        Args:
            source_name: Source entity name
            target_name: Target entity name
            relation_type: Type of relation
            properties: Optional relation properties
            confidence: Confidence score
            source: Source of information
            
        Returns:
            Relation object
        """
        # Ensure entities exist
        source_entity = self.add_entity(source_name)
        target_entity = self.add_entity(target_name)
        
        # Parse relation type
        if isinstance(relation_type, str):
            try:
                relation_type = RelationType(relation_type)
            except ValueError:
                relation_type = RelationType.RELATED_TO
        
        # Check for duplicate
        for rel in self._relations:
            if (rel.source_id == source_entity.id and
                rel.target_id == target_entity.id and
                rel.relation_type == relation_type):
                # Update confidence
                rel.confidence = max(rel.confidence, confidence)
                return rel
        
        # Create relation
        relation = Relation(
            source_id=source_entity.id,
            target_id=target_entity.id,
            relation_type=relation_type,
            properties=properties or {},
            confidence=confidence,
            source=source
        )
        
        # Add to storage
        rel_idx = len(self._relations)
        self._relations.append(relation)
        
        # Update indices
        self._relation_index[source_entity.id].append(rel_idx)
        self._relation_index[target_entity.id].append(rel_idx)
        
        return relation
    
    def add_from_text(self, text: str, source: str = "") -> int:
        """
        Extract and add knowledge from text.
        
        Args:
            text: Text to extract from
            source: Source identifier
            
        Returns:
            Number of relations extracted
        """
        relations_added = 0
        
        # Extract using patterns
        for pattern, rel_type in self._relation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    source_name = groups[0]
                    target_name = groups[1]
                    
                    self.add_relation(
                        source_name,
                        target_name,
                        rel_type,
                        source=source
                    )
                    relations_added += 1
        
        # Extract additional entities (capitalized words)
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        for word in set(words):
            self.add_entity(word, source=source)
        
        return relations_added
    
    def add_triple(self, subject: str, predicate: str, obj: str, source: str = "") -> Relation:
        """
        Add a knowledge triple.
        
        Args:
            subject: Subject entity
            predicate: Predicate/relation
            obj: Object entity
            source: Source
            
        Returns:
            Created relation
        """
        # Try to map predicate to RelationType
        predicate_lower = predicate.lower().replace(" ", "_")
        
        try:
            rel_type = RelationType(predicate_lower)
        except ValueError:
            rel_type = RelationType.HAS_PROPERTY
        
        return self.add_relation(subject, obj, rel_type, source=source)
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name."""
        entity_id = self._name_index.get(name.lower())
        return self._entities.get(entity_id)
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)
    
    def get_relations(
        self,
        entity_name: str,
        relation_type: Optional[RelationType] = None,
        as_source: bool = True,
        as_target: bool = True
    ) -> List[Tuple[Entity, Relation, Entity]]:
        """
        Get relations involving an entity.
        
        Args:
            entity_name: Entity name
            relation_type: Optional filter by type
            as_source: Include where entity is source
            as_target: Include where entity is target
            
        Returns:
            List of (source_entity, relation, target_entity) tuples
        """
        entity = self.get_entity(entity_name)
        if not entity:
            return []
        
        results = []
        rel_indices = self._relation_index.get(entity.id, [])
        
        for idx in rel_indices:
            rel = self._relations[idx]
            
            # Filter by type
            if relation_type and rel.relation_type != relation_type:
                continue
            
            # Filter by direction
            if rel.source_id == entity.id and not as_source:
                continue
            if rel.target_id == entity.id and not as_target:
                continue
            
            source_entity = self._entities.get(rel.source_id)
            target_entity = self._entities.get(rel.target_id)
            
            if source_entity and target_entity:
                results.append((source_entity, rel, target_entity))
        
        return results
    
    def query(
        self,
        question: str,
        max_hops: int = 2
    ) -> QueryResult:
        """
        Query the knowledge graph with natural language.
        
        Args:
            question: Natural language question
            max_hops: Maximum relation hops for inference
            
        Returns:
            Query results
        """
        question_lower = question.lower()
        
        # Extract key entities from question
        entities_mentioned = []
        for name, entity_id in self._name_index.items():
            if name in question_lower:
                entity = self._entities.get(entity_id)
                if entity:
                    entities_mentioned.append(entity)
        
        if not entities_mentioned:
            return QueryResult(entities=[], relations=[], paths=[])
        
        # Determine query type
        is_who_query = "who" in question_lower
        is_where_query = "where" in question_lower
        is_what_query = "what" in question_lower
        
        results = QueryResult(entities=[], relations=[], paths=[])
        
        for entity in entities_mentioned:
            relations = self.get_relations(entity.name)
            
            for source, rel, target in relations:
                # Filter based on question type
                if is_who_query and target.entity_type not in [EntityType.PERSON, EntityType.UNKNOWN]:
                    continue
                if is_where_query and target.entity_type not in [EntityType.LOCATION, EntityType.UNKNOWN]:
                    continue
                
                # Add to results
                if source not in results.entities:
                    results.entities.append(source)
                if target not in results.entities:
                    results.entities.append(target)
                results.relations.append(rel)
                results.paths.append([source.id, target.id])
        
        # Multi-hop inference
        if max_hops > 1:
            self._expand_paths(results, max_hops - 1)
        
        return results
    
    def _expand_paths(self, results: QueryResult, remaining_hops: int):
        """Expand query results with multi-hop inference."""
        if remaining_hops <= 0:
            return
        
        # Get entities at path ends
        end_entities = set()
        for path in results.paths:
            if path:
                end_entities.add(path[-1])
        
        # Expand from each
        for entity_id in end_entities:
            entity = self._entities.get(entity_id)
            if not entity:
                continue
            
            relations = self.get_relations(entity.name)
            
            for source, rel, target in relations:
                if rel not in results.relations:
                    results.relations.append(rel)
                
                if target not in results.entities:
                    results.entities.append(target)
                
                # Extend paths
                for path in results.paths:
                    if path and path[-1] == entity_id:
                        results.paths.append(path + [target.id])
    
    def find_path(
        self,
        source_name: str,
        target_name: str,
        max_depth: int = 4
    ) -> Optional[List[str]]:
        """
        Find connection path between two entities.
        
        Args:
            source_name: Source entity name
            target_name: Target entity name
            max_depth: Maximum path length
            
        Returns:
            Path as list of entity IDs, or None if not found
        """
        source = self.get_entity(source_name)
        target = self.get_entity(target_name)
        
        if not source or not target:
            return None
        
        # BFS
        visited = {source.id}
        queue = [[source.id]]
        
        while queue:
            path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            current_id = path[-1]
            
            # Get neighbors
            for rel_idx in self._relation_index.get(current_id, []):
                rel = self._relations[rel_idx]
                
                # Get other end
                if rel.source_id == current_id:
                    neighbor_id = rel.target_id
                else:
                    neighbor_id = rel.source_id
                
                if neighbor_id == target.id:
                    return path + [neighbor_id]
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(path + [neighbor_id])
        
        return None
    
    def get_subgraph(
        self,
        center_entity: str,
        radius: int = 2
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Get subgraph around an entity.
        
        Args:
            center_entity: Center entity name
            radius: Hop radius
            
        Returns:
            (entities, relations) tuple
        """
        center = self.get_entity(center_entity)
        if not center:
            return [], []
        
        entities = {center.id: center}
        relations = []
        
        frontier = {center.id}
        
        for _ in range(radius):
            new_frontier = set()
            
            for entity_id in frontier:
                for rel_idx in self._relation_index.get(entity_id, []):
                    rel = self._relations[rel_idx]
                    
                    # Add relation
                    if rel not in relations:
                        relations.append(rel)
                    
                    # Add connected entities
                    for eid in [rel.source_id, rel.target_id]:
                        if eid not in entities:
                            entity = self._entities.get(eid)
                            if entity:
                                entities[eid] = entity
                                new_frontier.add(eid)
            
            frontier = new_frontier
        
        return list(entities.values()), relations
    
    def _generate_id(self, name: str) -> str:
        """Generate unique entity ID."""
        base_id = name.lower().replace(" ", "_")[:20]
        
        if base_id not in self._entities:
            return base_id
        
        i = 1
        while f"{base_id}_{i}" in self._entities:
            i += 1
        
        return f"{base_id}_{i}"
    
    def save(self, path: Optional[str] = None):
        """Save graph to file."""
        path = Path(path) if path else self._persist_path
        if not path:
            return
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "entities": {
                eid: {
                    "id": e.id,
                    "name": e.name,
                    "type": e.entity_type.name,
                    "properties": e.properties,
                    "aliases": e.aliases,
                    "mention_count": e.mention_count,
                    "confidence": e.confidence,
                    "sources": e.sources,
                }
                for eid, e in self._entities.items()
            },
            "relations": [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "type": r.relation_type.value,
                    "properties": r.properties,
                    "confidence": r.confidence,
                    "source": r.source,
                }
                for r in self._relations
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved knowledge graph: {len(self._entities)} entities, {len(self._relations)} relations")
    
    def load(self, path: Optional[str] = None):
        """Load graph from file."""
        path = Path(path) if path else self._persist_path
        if not path or not path.exists():
            return
        
        with open(path) as f:
            data = json.load(f)
        
        # Load entities
        for eid, e_data in data.get("entities", {}).items():
            entity = Entity(
                id=e_data["id"],
                name=e_data["name"],
                entity_type=EntityType[e_data.get("type", "UNKNOWN")],
                properties=e_data.get("properties", {}),
                aliases=e_data.get("aliases", []),
                mention_count=e_data.get("mention_count", 1),
                confidence=e_data.get("confidence", 1.0),
                sources=e_data.get("sources", []),
            )
            
            self._entities[eid] = entity
            self._name_index[entity.name.lower()] = eid
            self._type_index[entity.entity_type].add(eid)
        
        # Load relations
        for r_data in data.get("relations", []):
            relation = Relation(
                source_id=r_data["source_id"],
                target_id=r_data["target_id"],
                relation_type=RelationType(r_data["type"]),
                properties=r_data.get("properties", {}),
                confidence=r_data.get("confidence", 1.0),
                source=r_data.get("source", ""),
            )
            
            rel_idx = len(self._relations)
            self._relations.append(relation)
            self._relation_index[relation.source_id].append(rel_idx)
            self._relation_index[relation.target_id].append(rel_idx)
        
        logger.info(f"Loaded knowledge graph: {len(self._entities)} entities, {len(self._relations)} relations")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        type_counts = {t.name: len(ids) for t, ids in self._type_index.items() if ids}
        relation_counts = defaultdict(int)
        for rel in self._relations:
            relation_counts[rel.relation_type.value] += 1
        
        return {
            "total_entities": len(self._entities),
            "total_relations": len(self._relations),
            "entity_types": type_counts,
            "relation_types": dict(relation_counts),
        }


# Global instance
_graph: Optional[KnowledgeGraph] = None


def get_graph(persist_path: Optional[str] = None) -> KnowledgeGraph:
    """Get or create global knowledge graph."""
    global _graph
    if _graph is None:
        default_path = Path.home() / ".enigma_engine" / "knowledge_graph.json"
        _graph = KnowledgeGraph(persist_path or str(default_path))
    return _graph
