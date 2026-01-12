"""
AI Brain - The Core Learning System

This module manages everything an AI knows and learns:
  - Training data storage
  - Conversation memory
  - Continuous learning from interactions
  - Curiosity-driven exploration

Each AI instance has its own brain folder with all its data.
The AI learns from every interaction when continuous learning is enabled.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
import threading

from ..config import CONFIG


class AIBrain:
    """
    The brain of an AI - manages all learning and memory.

    Each AI has its own folder structure:
        models/{name}/
            brain/
                training_data.jsonl     # All training examples
                conversations/          # Saved conversation sessions
                memories.jsonl          # Long-term memories
                curiosities.jsonl       # Things the AI wants to learn about
                learned_patterns.json   # Patterns it has discovered
            config.json                 # AI personality/settings
            checkpoints/               # Model weights
    """

    def __init__(self, model_name: str, auto_learn: bool = True):
        self.model_name = model_name
        self.auto_learn = auto_learn

        # Paths
        self.model_dir = Path(CONFIG["models_dir"]) / model_name
        self.brain_dir = self.model_dir / "brain"
        self.data_dir = self.model_dir / "data"  # Legacy support

        # Create directories
        self.brain_dir.mkdir(parents=True, exist_ok=True)
        (self.brain_dir / "conversations").mkdir(exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Data files
        self.training_file = self.brain_dir / "training_data.jsonl"
        self.memories_file = self.brain_dir / "memories.jsonl"
        self.curiosities_file = self.brain_dir / "curiosities.jsonl"
        self.patterns_file = self.brain_dir / "learned_patterns.json"

        # In-memory state
        self.current_conversation: List[Dict] = []
        self.pending_training: List[Dict] = []
        self.curiosities: List[str] = []

        # Load existing data
        self._load_curiosities()

        # Stats
        self.interactions_since_train = 0
        self.train_threshold = 10  # Auto-train after this many interactions

        # Thread safety
        self._lock = threading.Lock()

    def record_interaction(self, user_input: str, ai_response: str,
                           quality: Optional[float] = None):
        """
        Record a conversation exchange. If auto_learn is enabled,
        this will be added to training data.

        Args:
            user_input: What the user said
            ai_response: What the AI responded
            quality: Optional quality score (0-1) for weighted learning
        """
        with self._lock:
            timestamp = time.time()

            # Add to current conversation
            self.current_conversation.append({
                "role": "user",
                "text": user_input,
                "ts": timestamp
            })
            self.current_conversation.append({
                "role": "assistant",
                "text": ai_response,
                "ts": timestamp
            })

            # If auto-learning, add to pending training
            if self.auto_learn:
                training_entry = {
                    "input": user_input,
                    "output": ai_response,
                    "timestamp": timestamp,
                    "quality": quality or 0.5,
                    "source": "conversation"
                }
                self.pending_training.append(training_entry)
                self._append_to_jsonl(self.training_file, training_entry)

                self.interactions_since_train += 1

    def save_conversation(self, name: Optional[str] = None) -> str:
        """Save current conversation to a file."""
        if not self.current_conversation:
            return ""

        if not name:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")

        conv_file = self.brain_dir / "conversations" / f"{name}.json"

        data = {
            "name": name,
            "model": self.model_name,
            "saved_at": time.time(),
            "messages": self.current_conversation,
            "message_count": len(self.current_conversation)
        }

        conv_file.write_text(json.dumps(data, indent=2))
        return str(conv_file)

    def new_conversation(self):
        """Start a fresh conversation (optionally save current first)."""
        if self.current_conversation:
            self.save_conversation()
        self.current_conversation = []

    def add_memory(self, text: str, importance: float = 0.5,
                   category: str = "general"):
        """Add something to long-term memory."""
        memory = {
            "text": text,
            "importance": importance,
            "category": category,
            "timestamp": time.time(),
            "recalled_count": 0
        }
        self._append_to_jsonl(self.memories_file, memory)

    def add_curiosity(self, topic: str):
        """Add something the AI wants to learn about."""
        if topic not in self.curiosities:
            self.curiosities.append(topic)
            self._append_to_jsonl(self.curiosities_file, {
                "topic": topic,
                "added": time.time(),
                "explored": False
            })

    def get_curiosities(self) -> List[str]:
        """Get list of things the AI is curious about."""
        return self.curiosities.copy()

    def mark_curiosity_explored(self, topic: str):
        """Mark a curiosity as explored."""
        if topic in self.curiosities:
            self.curiosities.remove(topic)

    def get_training_data(self) -> List[Dict]:
        """Get all training data for this AI."""
        data = []
        if self.training_file.exists():
            for line in self.training_file.read_text().strip().split('\n'):
                if line:
                    try:
                        data.append(json.loads(line))
                    except BaseException:
                        pass
        return data

    def get_training_count(self) -> int:
        """Get count of training examples."""
        return len(self.get_training_data())

    def export_for_training(self) -> str:
        """
        Export training data in format suitable for training.
        Returns path to exported file.
        """
        # Also include legacy training.txt format
        export_path = self.data_dir / "training.txt"

        lines = []
        lines.append("# Auto-generated training data from conversations")
        lines.append(f"# Model: {self.model_name}")
        lines.append(f"# Exported: {datetime.now().isoformat()}")
        lines.append(f"# Total examples: {self.get_training_count()}")
        lines.append("")

        for entry in self.get_training_data():
            user_text = entry.get("input", "")
            ai_text = entry.get("output", "")
            if user_text and ai_text:
                lines.append(f"Q: {user_text}")
                lines.append(f"A: {ai_text}")
                lines.append("")

        export_path.write_text('\n'.join(lines))
        return str(export_path)

    def should_auto_train(self) -> bool:
        """Check if we should trigger auto-training."""
        return (self.auto_learn and
                self.interactions_since_train >= self.train_threshold)

    def mark_trained(self):
        """Mark that training has occurred."""
        self.interactions_since_train = 0
        self.pending_training = []

    def get_stats(self) -> Dict[str, Any]:
        """Get brain statistics."""
        return {
            "model_name": self.model_name,
            "auto_learn": self.auto_learn,
            "training_examples": self.get_training_count(),
            "current_conversation_length": len(self.current_conversation),
            "pending_training": len(self.pending_training),
            "interactions_since_train": self.interactions_since_train,
            "curiosities_count": len(self.curiosities)
        }

    def _append_to_jsonl(self, filepath: Path, data: Dict):
        """Append JSON line to file."""
        with open(filepath, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _load_curiosities(self):
        """Load curiosities from file."""
        self.curiosities = []
        if self.curiosities_file.exists():
            for line in self.curiosities_file.read_text().strip().split('\n'):
                if line:
                    try:
                        entry = json.loads(line)
                        if not entry.get("explored", False):
                            self.curiosities.append(entry["topic"])
                    except BaseException:
                        pass


# Global brain instance cache
_brains: Dict[str, AIBrain] = {}


def get_brain(model_name: str, auto_learn: bool = True) -> AIBrain:
    """Get or create a brain for a model."""
    if model_name not in _brains:
        _brains[model_name] = AIBrain(model_name, auto_learn)
    return _brains[model_name]


def set_auto_learn(model_name: str, enabled: bool):
    """Enable/disable auto-learning for a model."""
    brain = get_brain(model_name)
    brain.auto_learn = enabled
