"""
Discovery Mode - Autonomous research and exploration

When idle, the AI can autonomously:
- Research interesting topics
- Explore curiosities
- Learn new things
- Generate creative ideas
- Organize knowledge
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DiscoveryMode:
    """
    Autonomous exploration and learning system.
    
    Features:
    - Topic suggestions based on interests
    - Scheduled discovery sessions
    - Knowledge organization
    - Curiosity tracking
    """
    
    # Interesting topics to explore
    DISCOVERY_TOPICS = [
        # Science & Technology
        "recent breakthroughs in quantum computing",
        "advances in renewable energy",
        "new discoveries in astronomy",
        "emerging programming languages",
        "developments in artificial intelligence",
        "innovations in biotechnology",
        
        # Arts & Culture
        "interesting art movements",
        "notable books published recently",
        "emerging music genres",
        "cultural festivals around the world",
        "architectural wonders",
        
        # History & Society
        "lesser-known historical events",
        "ancient civilizations",
        "social movements and their impact",
        "evolution of technology",
        "philosophical concepts",
        
        # Nature & Environment
        "unique animal behaviors",
        "rare plant species",
        "geological formations",
        "climate phenomena",
        "conservation success stories",
        
        # Practical Skills
        "productivity techniques",
        "learning methods",
        "creative problem-solving",
        "communication strategies",
        "time management systems"
    ]
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize discovery mode.
        
        Args:
            storage_path: Path to store discoveries
        """
        if storage_path is None:
            from ..config import CONFIG
            storage_path = Path(CONFIG["data_dir"]) / "discoveries.json"
        
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.discoveries = self._load()
        self.enabled = False
        self.idle_threshold = 300  # Seconds before starting discovery
        self.last_activity = datetime.now()
    
    def _load(self) -> List[Dict]:
        """Load past discoveries."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return []
        return []
    
    def _save(self):
        """Save discoveries."""
        with open(self.storage_path, 'w') as f:
            json.dump(self.discoveries, f, indent=2)
    
    def enable(self, idle_threshold: int = 300):
        """
        Enable discovery mode.
        
        Args:
            idle_threshold: Seconds of inactivity before starting discovery
        """
        self.enabled = True
        self.idle_threshold = idle_threshold
        self.last_activity = datetime.now()
    
    def disable(self):
        """Disable discovery mode."""
        self.enabled = False
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def is_idle(self) -> bool:
        """Check if system has been idle long enough."""
        if not self.enabled:
            return False
        
        idle_time = (datetime.now() - self.last_activity).total_seconds()
        return idle_time >= self.idle_threshold
    
    def get_discovery_topic(self) -> str:
        """Get a topic to explore, using AI to pick something relevant."""
        # Try AI-driven topic selection
        try:
            from ..core.inference import EnigmaEngine
            engine = EnigmaEngine.get_instance()
            
            if engine and engine.model:
                # Build context from past discoveries
                context = ""
                if self.discoveries:
                    recent = [d['topic'] for d in self.discoveries[-5:]]
                    context = f"Recently explored: {', '.join(recent)}. "
                
                prompt = f"""{context}Pick ONE topic from this list that would be most interesting to learn about next: {', '.join(self.DISCOVERY_TOPICS[:20])}

Consider what builds on recent topics or fills knowledge gaps. Reply with ONLY the topic name."""
                
                response = engine.generate(prompt, max_length=30, temperature=0.7)
                topic = response.strip()
                
                # Validate
                for t in self.DISCOVERY_TOPICS:
                    if t.lower() in topic.lower() or topic.lower() in t.lower():
                        return t
        except Exception:
            pass
        
        # Fallback: Check if we have related topics from last discovery
        if self.discoveries and 'related_topics' in self.discoveries[-1]:
            related = self.discoveries[-1]['related_topics']
            if related:
                return related[0]  # Pick first related rather than random
        
        # Last resort: Return first unexplored topic
        explored = {d['topic'] for d in self.discoveries}
        for topic in self.DISCOVERY_TOPICS:
            if topic not in explored:
                return topic
        return self.DISCOVERY_TOPICS[0]
    
    def suggest_research_query(self, topic: str) -> str:
        """
        Generate a research query for a topic using AI when available.
        
        Args:
            topic: The topic to research
            
        Returns:
            A formatted research query
        """
        # Try AI-generated query
        try:
            from ..core.inference import EnigmaEngine
            engine = EnigmaEngine.get_instance()
            
            if engine and engine.model:
                prompt = f"""Generate ONE interesting research question about '{topic}' that would lead to learning something new and practical. Be specific and curious. Reply with ONLY the question."""
                
                response = engine.generate(prompt, max_length=60, temperature=0.8)
                if response and '?' in response:
                    return response.strip()
        except Exception:
            pass
        
        # Fallback: Simple template
        return f"What are the most interesting aspects of {topic}?"
    
    def log_discovery(self, topic: str, findings: str, 
                     related_topics: Optional[List[str]] = None):
        """
        Log a discovery.
        
        Args:
            topic: The topic explored
            findings: What was discovered
            related_topics: Related topics to explore later
        """
        discovery = {
            'timestamp': datetime.now().isoformat(),
            'topic': topic,
            'findings': findings[:1000],  # Truncate
            'related_topics': related_topics or []
        }
        
        self.discoveries.append(discovery)
        
        # Keep only recent discoveries
        if len(self.discoveries) > 100:
            self.discoveries = self.discoveries[-100:]
        
        self._save()
    
    def get_recent_discoveries(self, limit: int = 10) -> List[Dict]:
        """Get recent discoveries."""
        return self.discoveries[-limit:]
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of discovery activity."""
        if not self.discoveries:
            return {
                'total_discoveries': 0,
                'topics_explored': [],
                'most_recent': None
            }
        
        # Extract topics
        topics = list({d['topic'] for d in self.discoveries})
        
        return {
            'total_discoveries': len(self.discoveries),
            'topics_explored': topics,
            'most_recent': self.discoveries[-1]['timestamp'],
            'enabled': self.enabled
        }
    
    def export_discoveries(self, output_path: Path) -> int:
        """
        Export discoveries as training data.
        
        Args:
            output_path: Path to export to
            
        Returns:
            Number of discoveries exported
        """
        lines = []
        for discovery in self.discoveries:
            lines.append(f"# Discovery: {discovery['topic']}")
            lines.append(f"# Date: {discovery['timestamp']}")
            lines.append(discovery['findings'])
            lines.append("")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return len(self.discoveries)


class AutoSaveManager:
    """
    Manages automatic saving of work in progress.
    
    Features:
    - Auto-save conversations
    - Auto-save training progress
    - Auto-save configurations
    - Recovery from crashes
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize auto-save manager.
        
        Args:
            storage_dir: Directory for auto-saves
        """
        if storage_dir is None:
            from ..config import CONFIG
            storage_dir = Path(CONFIG["data_dir"]) / "autosave"
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.enabled = True
        self.save_interval = 60  # Seconds between auto-saves
        self.last_save = {}  # Track last save time per item
    
    def enable(self, interval: int = 60):
        """
        Enable auto-save.
        
        Args:
            interval: Seconds between auto-saves
        """
        self.enabled = True
        self.save_interval = interval
    
    def disable(self):
        """Disable auto-save."""
        self.enabled = False
    
    def should_save(self, item_id: str) -> bool:
        """Check if an item should be auto-saved."""
        if not self.enabled:
            return False
        
        if item_id not in self.last_save:
            return True
        
        elapsed = (datetime.now() - self.last_save[item_id]).total_seconds()
        return elapsed >= self.save_interval
    
    def save_conversation(self, conversation_id: str, messages: List[Dict]) -> bool:
        """
        Auto-save a conversation.
        
        Args:
            conversation_id: Unique ID for conversation
            messages: List of message dicts
            
        Returns:
            True if saved successfully
        """
        if not self.should_save(f"conv_{conversation_id}"):
            return False
        
        try:
            save_path = self.storage_dir / f"conversation_{conversation_id}.json"
            data = {
                'id': conversation_id,
                'timestamp': datetime.now().isoformat(),
                'messages': messages,
                'auto_save': True
            }
            
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.last_save[f"conv_{conversation_id}"] = datetime.now()
            return True
        except Exception as e:
            logger.warning(f"Auto-save conversation failed: {e}")
            return False
    
    def save_training_state(self, model_name: str, state: Dict) -> bool:
        """
        Auto-save training progress.
        
        Args:
            model_name: Name of model being trained
            state: Training state dict
            
        Returns:
            True if saved successfully
        """
        if not self.should_save(f"train_{model_name}"):
            return False
        
        try:
            save_path = self.storage_dir / f"training_{model_name}.json"
            data = {
                'model': model_name,
                'timestamp': datetime.now().isoformat(),
                'state': state,
                'auto_save': True
            }
            
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.last_save[f"train_{model_name}"] = datetime.now()
            return True
        except Exception as e:
            logger.warning(f"Auto-save training state failed: {e}")
            return False
    
    def save_config(self, config_name: str, config: Dict) -> bool:
        """
        Auto-save configuration.
        
        Args:
            config_name: Name of configuration
            config: Configuration dict
            
        Returns:
            True if saved successfully
        """
        if not self.should_save(f"config_{config_name}"):
            return False
        
        try:
            save_path = self.storage_dir / f"config_{config_name}.json"
            data = {
                'name': config_name,
                'timestamp': datetime.now().isoformat(),
                'config': config,
                'auto_save': True
            }
            
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.last_save[f"config_{config_name}"] = datetime.now()
            return True
        except Exception as e:
            logger.warning(f"Auto-save config failed: {e}")
            return False
    
    def list_auto_saves(self) -> List[Dict[str, Any]]:
        """List all auto-save files."""
        saves = []
        for file in self.storage_dir.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                saves.append({
                    'file': file.name,
                    'timestamp': data.get('timestamp'),
                    'type': 'conversation' if 'messages' in data else
                            'training' if 'state' in data else
                            'config'
                })
            except (json.JSONDecodeError, OSError):
                continue
        
        # Sort by timestamp
        saves.sort(key=lambda x: x['timestamp'] or '', reverse=True)
        return saves
    
    def recover_conversation(self, conversation_id: str) -> Optional[List[Dict]]:
        """
        Recover an auto-saved conversation.
        
        Args:
            conversation_id: Conversation ID to recover
            
        Returns:
            List of messages or None
        """
        save_path = self.storage_dir / f"conversation_{conversation_id}.json"
        if save_path.exists():
            try:
                with open(save_path) as f:
                    data = json.load(f)
                return data.get('messages', [])
            except (json.JSONDecodeError, OSError):
                return None
        return None
    
    def clean_old_saves(self, days: int = 7):
        """
        Remove auto-saves older than specified days.
        
        Args:
            days: Number of days to keep
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        for file in self.storage_dir.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                timestamp = datetime.fromisoformat(data.get('timestamp', ''))
                if timestamp < cutoff:
                    file.unlink()
            except (json.JSONDecodeError, OSError, ValueError):
                continue


if __name__ == "__main__":
    # Test discovery mode
    print("Discovery Mode Test")
    print("=" * 60)
    
    discovery = DiscoveryMode()
    discovery.enable(idle_threshold=5)
    
    # Simulate getting topics
    for i in range(5):
        topic = discovery.get_discovery_topic()
        query = discovery.suggest_research_query(topic)
        print(f"\nTopic {i+1}: {topic}")
        print(f"Query: {query}")
    
    # Test auto-save
    print("\n\nAuto-Save Test")
    print("=" * 60)
    
    autosave = AutoSaveManager()
    
    # Save a conversation
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    success = autosave.save_conversation("test_conv", messages)
    print(f"Conversation saved: {success}")
    
    # List saves
    print("\nAuto-saves:")
    for save in autosave.list_auto_saves():
        print(f"  {save['type']}: {save['file']} ({save['timestamp']})")
