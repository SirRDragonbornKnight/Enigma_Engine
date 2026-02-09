"""
Agent Persistence for Enigma AI Engine

Save and restore agent states across sessions.

Features:
- Agent state serialization
- Memory persistence
- Config preservation
- Quick restore
- Version tracking

Usage:
    from enigma_engine.agents.persistence import AgentPersistence
    
    persistence = AgentPersistence()
    
    # Save agent state
    persistence.save_agent("researcher", agent)
    
    # Restore agent
    agent = persistence.load_agent("researcher")
"""

import json
import logging
import pickle
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Serializable agent state."""
    agent_id: str
    agent_type: str
    name: str
    config: Dict[str, Any]
    memory: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1


@dataclass
class AgentSnapshot:
    """Point-in-time agent snapshot."""
    snapshot_id: str
    agent_id: str
    state: AgentState
    timestamp: float = field(default_factory=time.time)
    description: str = ""


class AgentPersistence:
    """Persist and restore agent states."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize persistence manager.
        
        Args:
            storage_dir: Directory for storing agent states
        """
        self.storage_dir = storage_dir or Path("data/agents")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Snapshots directory
        self.snapshots_dir = self.storage_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        
        # State cache
        self._cache: Dict[str, AgentState] = {}
        
        # Agent type registry
        self._agent_types: Dict[str, Type] = {}
    
    def register_agent_type(self, name: str, agent_class: Type):
        """Register an agent type for loading."""
        self._agent_types[name] = agent_class
    
    def save_agent(
        self,
        agent_id: str,
        agent: Any,
        create_snapshot: bool = False,
        snapshot_desc: str = ""
    ):
        """
        Save an agent's state.
        
        Args:
            agent_id: Unique agent identifier
            agent: Agent instance to save
            create_snapshot: Whether to create a snapshot
            snapshot_desc: Snapshot description
        """
        # Extract state from agent
        state = self._extract_state(agent_id, agent)
        
        # Save to disk
        self._save_state(agent_id, state)
        
        # Update cache
        self._cache[agent_id] = state
        
        # Create snapshot if requested
        if create_snapshot:
            self._create_snapshot(agent_id, state, snapshot_desc)
        
        logger.info(f"Saved agent: {agent_id}")
    
    def load_agent(
        self,
        agent_id: str,
        snapshot_id: Optional[str] = None
    ) -> Optional[Any]:
        """
        Load an agent from saved state.
        
        Args:
            agent_id: Agent identifier
            snapshot_id: Optional snapshot to restore
            
        Returns:
            Restored agent instance or None
        """
        # Load from snapshot if specified
        if snapshot_id:
            state = self._load_snapshot(snapshot_id)
        else:
            state = self._load_state(agent_id)
        
        if not state:
            logger.warning(f"No saved state for agent: {agent_id}")
            return None
        
        # Reconstruct agent
        agent = self._reconstruct_agent(state)
        
        logger.info(f"Loaded agent: {agent_id}")
        return agent
    
    def _extract_state(self, agent_id: str, agent: Any) -> AgentState:
        """Extract serializable state from agent."""
        state = AgentState(
            agent_id=agent_id,
            agent_type=type(agent).__name__,
            name=getattr(agent, 'name', agent_id),
            config={},
            memory={},
            metadata={}
        )
        
        # Extract config
        if hasattr(agent, 'config'):
            config = agent.config
            if hasattr(config, '__dict__'):
                state.config = {
                    k: v for k, v in config.__dict__.items()
                    if self._is_serializable(v)
                }
            elif isinstance(config, dict):
                state.config = config
        
        # Extract memory
        if hasattr(agent, 'memory'):
            memory = agent.memory
            if hasattr(memory, 'to_dict'):
                state.memory = memory.to_dict()
            elif isinstance(memory, dict):
                state.memory = memory
            elif hasattr(memory, '__dict__'):
                state.memory = {
                    k: v for k, v in memory.__dict__.items()
                    if self._is_serializable(v)
                }
        
        # Extract metadata
        state.metadata = {
            'created_at': getattr(agent, 'created_at', time.time()),
            'total_interactions': getattr(agent, 'total_interactions', 0),
            'last_active': getattr(agent, 'last_active', time.time())
        }
        
        return state
    
    def _reconstruct_agent(self, state: AgentState) -> Optional[Any]:
        """Reconstruct agent from state."""
        agent_type = state.agent_type
        
        # Try registered types first
        if agent_type in self._agent_types:
            agent_class = self._agent_types[agent_type]
            
            try:
                # Create instance
                agent = agent_class(**state.config)
                
                # Restore memory
                if hasattr(agent, 'memory') and state.memory:
                    if hasattr(agent.memory, 'from_dict'):
                        agent.memory.from_dict(state.memory)
                    elif isinstance(agent.memory, dict):
                        agent.memory.update(state.memory)
                
                # Restore metadata
                for key, value in state.metadata.items():
                    if hasattr(agent, key):
                        setattr(agent, key, value)
                
                return agent
                
            except Exception as e:
                logger.error(f"Failed to reconstruct agent: {e}")
        
        # Return raw state if can't reconstruct
        logger.warning(f"Unknown agent type: {agent_type}, returning state dict")
        return state
    
    def _save_state(self, agent_id: str, state: AgentState):
        """Save state to disk."""
        state_file = self.storage_dir / f"{agent_id}.json"
        
        state_dict = {
            'agent_id': state.agent_id,
            'agent_type': state.agent_type,
            'name': state.name,
            'config': state.config,
            'memory': state.memory,
            'metadata': state.metadata,
            'created_at': state.created_at,
            'updated_at': time.time(),
            'version': state.version
        }
        
        state_file.write_text(json.dumps(state_dict, indent=2, default=str))
    
    def _load_state(self, agent_id: str) -> Optional[AgentState]:
        """Load state from disk."""
        # Check cache first
        if agent_id in self._cache:
            return self._cache[agent_id]
        
        state_file = self.storage_dir / f"{agent_id}.json"
        
        if not state_file.exists():
            return None
        
        try:
            state_dict = json.loads(state_file.read_text())
            
            state = AgentState(
                agent_id=state_dict['agent_id'],
                agent_type=state_dict['agent_type'],
                name=state_dict['name'],
                config=state_dict['config'],
                memory=state_dict['memory'],
                metadata=state_dict.get('metadata', {}),
                created_at=state_dict.get('created_at', time.time()),
                updated_at=state_dict.get('updated_at', time.time()),
                version=state_dict.get('version', 1)
            )
            
            self._cache[agent_id] = state
            return state
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None
    
    def _create_snapshot(
        self,
        agent_id: str,
        state: AgentState,
        description: str = ""
    ):
        """Create a state snapshot."""
        snapshot_id = f"{agent_id}_{int(time.time())}"
        
        snapshot = AgentSnapshot(
            snapshot_id=snapshot_id,
            agent_id=agent_id,
            state=state,
            description=description
        )
        
        # Save snapshot
        snapshot_file = self.snapshots_dir / f"{snapshot_id}.json"
        
        snapshot_dict = {
            'snapshot_id': snapshot.snapshot_id,
            'agent_id': snapshot.agent_id,
            'timestamp': snapshot.timestamp,
            'description': snapshot.description,
            'state': {
                'agent_id': state.agent_id,
                'agent_type': state.agent_type,
                'name': state.name,
                'config': state.config,
                'memory': state.memory,
                'metadata': state.metadata,
                'version': state.version
            }
        }
        
        snapshot_file.write_text(json.dumps(snapshot_dict, indent=2, default=str))
        logger.info(f"Created snapshot: {snapshot_id}")
    
    def _load_snapshot(self, snapshot_id: str) -> Optional[AgentState]:
        """Load a snapshot."""
        snapshot_file = self.snapshots_dir / f"{snapshot_id}.json"
        
        if not snapshot_file.exists():
            return None
        
        try:
            data = json.loads(snapshot_file.read_text())
            state_dict = data['state']
            
            return AgentState(
                agent_id=state_dict['agent_id'],
                agent_type=state_dict['agent_type'],
                name=state_dict['name'],
                config=state_dict['config'],
                memory=state_dict['memory'],
                metadata=state_dict.get('metadata', {}),
                version=state_dict.get('version', 1)
            )
            
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            return None
    
    def list_agents(self) -> List[Dict]:
        """List all saved agents."""
        agents = []
        
        for state_file in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(state_file.read_text())
                agents.append({
                    'agent_id': data['agent_id'],
                    'name': data['name'],
                    'type': data['agent_type'],
                    'updated_at': data.get('updated_at', 0)
                })
            except Exception:
                continue
        
        return sorted(agents, key=lambda x: x['updated_at'], reverse=True)
    
    def list_snapshots(self, agent_id: Optional[str] = None) -> List[Dict]:
        """List snapshots, optionally filtered by agent."""
        snapshots = []
        
        for snapshot_file in self.snapshots_dir.glob("*.json"):
            try:
                data = json.loads(snapshot_file.read_text())
                
                if agent_id and data['agent_id'] != agent_id:
                    continue
                
                snapshots.append({
                    'snapshot_id': data['snapshot_id'],
                    'agent_id': data['agent_id'],
                    'timestamp': data['timestamp'],
                    'description': data.get('description', '')
                })
            except Exception:
                continue
        
        return sorted(snapshots, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_agent(self, agent_id: str, delete_snapshots: bool = False):
        """Delete an agent's saved state."""
        state_file = self.storage_dir / f"{agent_id}.json"
        
        if state_file.exists():
            state_file.unlink()
        
        if agent_id in self._cache:
            del self._cache[agent_id]
        
        if delete_snapshots:
            for snapshot_file in self.snapshots_dir.glob(f"{agent_id}_*.json"):
                snapshot_file.unlink()
        
        logger.info(f"Deleted agent: {agent_id}")
    
    def _is_serializable(self, value: Any) -> bool:
        """Check if value is JSON serializable."""
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False


# Global instance
_persistence: Optional[AgentPersistence] = None


def get_persistence() -> AgentPersistence:
    """Get or create global persistence manager."""
    global _persistence
    if _persistence is None:
        _persistence = AgentPersistence()
    return _persistence
