"""
================================================================================
Entity Memory - Remember people, places, things, and facts.
================================================================================

Knowledge persistence system:
- Extract and store entities from conversations
- Remember facts about entities
- Build relationship graphs
- Query entity knowledge
- Update and merge information

USAGE:
    from forge_ai.memory.entity_memory import EntityMemory, get_entity_memory
    
    memory = get_entity_memory()
    
    # Add an entity
    memory.add_entity(
        name="John",
        entity_type="person",
        facts=["Works at Google", "Likes Python"]
    )
    
    # Add a relationship
    memory.add_relationship("John", "Alice", "knows")
    
    # Query
    john = memory.get_entity("John")
    print(john.facts)
    
    # Find related entities
    related = memory.get_related("John")
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class EntityType:
    """Standard entity types."""
    PERSON = "person"
    PLACE = "place"
    ORGANIZATION = "organization"
    THING = "thing"
    CONCEPT = "concept"
    EVENT = "event"
    PROJECT = "project"
    TECHNOLOGY = "technology"
    CUSTOM = "custom"


@dataclass
class Entity:
    """An entity with associated knowledge."""
    name: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: str = ""
    updated_at: str = ""
    mention_count: int = 0
    confidence: float = 1.0  # How certain we are about this entity
    source: str = ""  # Where we learned about this
    
    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        self.updated_at = now
    
    def add_fact(self, fact: str) -> bool:
        """Add a fact if not duplicate."""
        fact_lower = fact.lower().strip()
        if not any(f.lower().strip() == fact_lower for f in self.facts):
            self.facts.append(fact)
            self.updated_at = datetime.now().isoformat()
            return True
        return False
    
    def add_alias(self, alias: str) -> bool:
        """Add an alias."""
        alias_lower = alias.lower()
        if alias_lower != self.name.lower() and alias_lower not in [a.lower() for a in self.aliases]:
            self.aliases.append(alias)
            self.updated_at = datetime.now().isoformat()
            return True
        return False
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute."""
        self.attributes[key] = value
        self.updated_at = datetime.now().isoformat()
    
    def matches_name(self, query: str) -> bool:
        """Check if query matches name or aliases."""
        query_lower = query.lower()
        if self.name.lower() == query_lower:
            return True
        return any(a.lower() == query_lower for a in self.aliases)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Relationship:
    """A relationship between two entities."""
    source: str  # Entity name
    target: str  # Entity name
    relation_type: str  # "knows", "works_at", "located_in", etc.
    bidirectional: bool = False
    strength: float = 1.0  # 0.0 to 1.0
    facts: List[str] = field(default_factory=list)
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class EntityMemory:
    """
    Long-term memory for entities and their relationships.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize entity memory.
        
        Args:
            data_path: Path to store entity data
        """
        self._data_path = data_path or Path("data/entities")
        self._data_path.mkdir(parents=True, exist_ok=True)
        
        self._entities_file = self._data_path / "entities.json"
        self._relationships_file = self._data_path / "relationships.json"
        
        self._entities: Dict[str, Entity] = {}  # name_lower -> Entity
        self._relationships: List[Relationship] = []
        
        # Index for fast lookup
        self._alias_index: Dict[str, str] = {}  # alias_lower -> canonical_name_lower
        self._type_index: Dict[str, Set[str]] = defaultdict(set)  # type -> {names}
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load entities and relationships from disk."""
        # Load entities
        if self._entities_file.exists():
            try:
                with open(self._entities_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data.get("entities", []):
                        entity = Entity.from_dict(item)
                        self._index_entity(entity)
                logger.info(f"Loaded {len(self._entities)} entities")
            except Exception as e:
                logger.error(f"Failed to load entities: {e}")
        
        # Load relationships
        if self._relationships_file.exists():
            try:
                with open(self._relationships_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data.get("relationships", []):
                        rel = Relationship.from_dict(item)
                        self._relationships.append(rel)
                logger.info(f"Loaded {len(self._relationships)} relationships")
            except Exception as e:
                logger.error(f"Failed to load relationships: {e}")
    
    def _save_data(self) -> None:
        """Save entities and relationships to disk."""
        try:
            # Save entities
            entity_data = {
                "entities": [e.to_dict() for e in self._entities.values()],
                "last_updated": datetime.now().isoformat()
            }
            with open(self._entities_file, 'w', encoding='utf-8') as f:
                json.dump(entity_data, f, indent=2)
            
            # Save relationships
            rel_data = {
                "relationships": [r.to_dict() for r in self._relationships],
                "last_updated": datetime.now().isoformat()
            }
            with open(self._relationships_file, 'w', encoding='utf-8') as f:
                json.dump(rel_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save entity data: {e}")
    
    def _index_entity(self, entity: Entity) -> None:
        """Index an entity for fast lookup."""
        name_lower = entity.name.lower()
        self._entities[name_lower] = entity
        self._type_index[entity.entity_type].add(name_lower)
        
        # Index aliases
        for alias in entity.aliases:
            self._alias_index[alias.lower()] = name_lower
    
    def _resolve_name(self, name: str) -> Optional[str]:
        """Resolve a name to canonical form."""
        name_lower = name.lower()
        
        # Direct match
        if name_lower in self._entities:
            return name_lower
        
        # Alias match
        if name_lower in self._alias_index:
            return self._alias_index[name_lower]
        
        return None
    
    def add_entity(
        self,
        name: str,
        entity_type: str = EntityType.THING,
        facts: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        source: str = ""
    ) -> Entity:
        """
        Add or update an entity.
        
        Args:
            name: Entity name
            entity_type: Type of entity
            facts: List of facts about the entity
            aliases: Alternative names
            attributes: Key-value attributes
            source: Source of information
            
        Returns:
            The entity (created or updated)
        """
        resolved = self._resolve_name(name)
        
        if resolved:
            # Update existing entity
            entity = self._entities[resolved]
            
            if facts:
                for fact in facts:
                    entity.add_fact(fact)
            
            if aliases:
                for alias in aliases:
                    if entity.add_alias(alias):
                        self._alias_index[alias.lower()] = resolved
            
            if attributes:
                for key, value in attributes.items():
                    entity.set_attribute(key, value)
            
            entity.mention_count += 1
        else:
            # Create new entity
            entity = Entity(
                name=name,
                entity_type=entity_type,
                facts=facts or [],
                aliases=aliases or [],
                attributes=attributes or {},
                source=source,
                mention_count=1
            )
            self._index_entity(entity)
        
        self._save_data()
        return entity
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """
        Get an entity by name or alias.
        
        Args:
            name: Entity name or alias
            
        Returns:
            Entity or None
        """
        resolved = self._resolve_name(name)
        if resolved:
            entity = self._entities[resolved]
            entity.mention_count += 1
            return entity
        return None
    
    def delete_entity(self, name: str) -> bool:
        """Delete an entity."""
        resolved = self._resolve_name(name)
        if not resolved:
            return False
        
        entity = self._entities[resolved]
        
        # Remove from indexes
        del self._entities[resolved]
        self._type_index[entity.entity_type].discard(resolved)
        
        for alias in entity.aliases:
            self._alias_index.pop(alias.lower(), None)
        
        # Remove relationships involving this entity
        self._relationships = [
            r for r in self._relationships 
            if r.source.lower() != resolved and r.target.lower() != resolved
        ]
        
        self._save_data()
        return True
    
    def add_fact(self, entity_name: str, fact: str) -> bool:
        """
        Add a fact to an entity.
        
        Args:
            entity_name: Entity name
            fact: Fact to add
            
        Returns:
            True if added, False if duplicate or entity not found
        """
        entity = self.get_entity(entity_name)
        if entity and entity.add_fact(fact):
            self._save_data()
            return True
        return False
    
    def add_relationship(
        self,
        source: str,
        target: str,
        relation_type: str,
        bidirectional: bool = False,
        strength: float = 1.0,
        facts: Optional[List[str]] = None
    ) -> Optional[Relationship]:
        """
        Add a relationship between entities.
        
        Args:
            source: Source entity name
            target: Target entity name
            relation_type: Type of relationship
            bidirectional: If true, relationship goes both ways
            strength: Relationship strength (0-1)
            facts: Facts about the relationship
            
        Returns:
            Relationship or None if entities not found
        """
        # Resolve names
        source_resolved = self._resolve_name(source)
        target_resolved = self._resolve_name(target)
        
        # Auto-create entities if they don't exist
        if not source_resolved:
            self.add_entity(source)
            source_resolved = source.lower()
        
        if not target_resolved:
            self.add_entity(target)
            target_resolved = target.lower()
        
        # Check for existing relationship
        for rel in self._relationships:
            if (rel.source.lower() == source_resolved and 
                rel.target.lower() == target_resolved and
                rel.relation_type == relation_type):
                # Update existing
                rel.strength = max(rel.strength, strength)
                if facts:
                    rel.facts.extend(facts)
                self._save_data()
                return rel
        
        # Create new relationship
        relationship = Relationship(
            source=self._entities[source_resolved].name,
            target=self._entities[target_resolved].name,
            relation_type=relation_type,
            bidirectional=bidirectional,
            strength=strength,
            facts=facts or []
        )
        
        self._relationships.append(relationship)
        self._save_data()
        return relationship
    
    def get_relationships(
        self,
        entity_name: str,
        relation_type: Optional[str] = None
    ) -> List[Relationship]:
        """
        Get relationships for an entity.
        
        Args:
            entity_name: Entity name
            relation_type: Optional filter by type
            
        Returns:
            List of relationships
        """
        resolved = self._resolve_name(entity_name)
        if not resolved:
            return []
        
        results = []
        for rel in self._relationships:
            is_source = rel.source.lower() == resolved
            is_target = rel.target.lower() == resolved and rel.bidirectional
            
            if is_source or is_target:
                if relation_type is None or rel.relation_type == relation_type:
                    results.append(rel)
        
        return results
    
    def get_related(
        self,
        entity_name: str,
        relation_type: Optional[str] = None
    ) -> List[Entity]:
        """
        Get entities related to an entity.
        
        Args:
            entity_name: Entity name
            relation_type: Optional filter by type
            
        Returns:
            List of related entities
        """
        relationships = self.get_relationships(entity_name, relation_type)
        resolved = self._resolve_name(entity_name)
        
        related = []
        seen = set()
        
        for rel in relationships:
            # Get the other entity in the relationship
            other_name = rel.target if rel.source.lower() == resolved else rel.source
            other_lower = other_name.lower()
            
            if other_lower not in seen and other_lower in self._entities:
                related.append(self._entities[other_lower])
                seen.add(other_lower)
        
        return related
    
    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Entity]:
        """
        Search entities by name or fact content.
        
        Args:
            query: Search query
            entity_type: Optional type filter
            limit: Max results
            
        Returns:
            List of matching entities
        """
        query_lower = query.lower()
        results: List[Tuple[float, Entity]] = []
        
        for entity in self._entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            
            score = 0.0
            
            # Name match (highest weight)
            if query_lower in entity.name.lower():
                score += 10.0
            
            # Alias match
            for alias in entity.aliases:
                if query_lower in alias.lower():
                    score += 5.0
                    break
            
            # Fact match
            for fact in entity.facts:
                if query_lower in fact.lower():
                    score += 1.0
            
            # Attribute match
            for value in entity.attributes.values():
                if query_lower in str(value).lower():
                    score += 0.5
            
            if score > 0:
                results.append((score, entity))
        
        # Sort by score and return
        results.sort(reverse=True, key=lambda x: x[0])
        return [e for _, e in results[:limit]]
    
    def list_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a type."""
        names = self._type_index.get(entity_type, set())
        return [self._entities[n] for n in names if n in self._entities]
    
    def list_types(self) -> List[str]:
        """Get all entity types in use."""
        return list(self._type_index.keys())
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract potential entities from text (simple heuristic).
        
        Args:
            text: Text to analyze
            
        Returns:
            List of (name, guessed_type) tuples
        """
        entities = []
        
        # Find capitalized words (potential names)
        # Simple heuristic - in production you'd use NER
        words = text.split()
        current_name = []
        
        for word in words:
            # Strip punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            
            if clean_word and clean_word[0].isupper() and not word.startswith('"'):
                current_name.append(clean_word)
            else:
                if current_name and len(current_name) <= 3:
                    name = ' '.join(current_name)
                    # Skip common words
                    if name.lower() not in {'the', 'a', 'an', 'i', 'we', 'they', 'he', 'she', 'it'}:
                        # Guess type
                        entity_type = EntityType.THING
                        if len(current_name) >= 2:
                            entity_type = EntityType.PERSON
                        entities.append((name, entity_type))
                current_name = []
        
        # Handle trailing name
        if current_name:
            name = ' '.join(current_name)
            if name.lower() not in {'the', 'a', 'an', 'i', 'we', 'they', 'he', 'she', 'it'}:
                entity_type = EntityType.PERSON if len(current_name) >= 2 else EntityType.THING
                entities.append((name, entity_type))
        
        return entities
    
    def process_message(self, content: str, source: str = "conversation") -> List[Entity]:
        """
        Process a message to extract and store entities.
        
        Args:
            content: Message content
            source: Source label
            
        Returns:
            List of entities found/created
        """
        extracted = self.extract_entities(content)
        entities = []
        
        for name, guessed_type in extracted:
            existing = self.get_entity(name)
            if existing:
                existing.mention_count += 1
                entities.append(existing)
            else:
                entity = self.add_entity(
                    name=name,
                    entity_type=guessed_type,
                    source=source
                )
                entities.append(entity)
        
        return entities
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of entity memory."""
        return {
            "total_entities": len(self._entities),
            "total_relationships": len(self._relationships),
            "entity_types": {t: len(names) for t, names in self._type_index.items()},
            "top_mentioned": [
                {"name": e.name, "mentions": e.mention_count}
                for e in sorted(
                    self._entities.values(),
                    key=lambda x: x.mention_count,
                    reverse=True
                )[:10]
            ]
        }
    
    def export_graph(self) -> Dict[str, Any]:
        """
        Export entity graph for visualization.
        
        Returns:
            Graph data with nodes and edges
        """
        nodes = [
            {
                "id": e.name,
                "type": e.entity_type,
                "mentions": e.mention_count
            }
            for e in self._entities.values()
        ]
        
        edges = [
            {
                "source": r.source,
                "target": r.target,
                "type": r.relation_type,
                "strength": r.strength
            }
            for r in self._relationships
        ]
        
        return {"nodes": nodes, "edges": edges}
    
    def clear(self) -> None:
        """Clear all entity memory."""
        self._entities.clear()
        self._relationships.clear()
        self._alias_index.clear()
        self._type_index.clear()
        self._save_data()


# Singleton instance
_entity_memory_instance: Optional[EntityMemory] = None


def get_entity_memory(data_path: Optional[Path] = None) -> EntityMemory:
    """Get or create the singleton entity memory."""
    global _entity_memory_instance
    if _entity_memory_instance is None:
        _entity_memory_instance = EntityMemory(data_path)
    return _entity_memory_instance


# Convenience functions
def remember_entity(name: str, entity_type: str = EntityType.THING, facts: List[str] = None) -> Entity:
    """Quick entity creation."""
    return get_entity_memory().add_entity(name, entity_type, facts)


def recall_entity(name: str) -> Optional[Entity]:
    """Quick entity lookup."""
    return get_entity_memory().get_entity(name)


def remember_relationship(source: str, target: str, relation: str) -> Optional[Relationship]:
    """Quick relationship creation."""
    return get_entity_memory().add_relationship(source, target, relation)
