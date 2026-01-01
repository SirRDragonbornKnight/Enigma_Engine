"""
AI Self-Evolving Personality System

The AI develops its own unique personality over time through interactions.

Traits evolve based on:
- Conversation topics the AI enjoys
- Response styles that get positive feedback
- Learned preferences and opinions
- Emotional patterns

Usage:
    from enigma.core.personality import AIPersonality
    
    personality = AIPersonality("my_model")
    personality.load()  # Load existing or create new
    
    # During conversation
    personality.evolve_from_interaction(user_input, ai_response, feedback="positive")
    
    # Get personality-influenced prompt
    system_prompt = personality.get_personality_prompt()
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from ..config import CONFIG


@dataclass
class PersonalityTraits:
    """Personality trait values (0.0 to 1.0)."""
    
    humor_level: float = 0.5      # 0=serious, 1=silly
    formality: float = 0.5         # 0=casual, 1=formal
    verbosity: float = 0.5         # 0=brief, 1=detailed
    curiosity: float = 0.5         # 0=answers only, 1=asks questions
    empathy: float = 0.5           # 0=logical, 1=emotional
    creativity: float = 0.5        # 0=factual, 1=imaginative
    confidence: float = 0.5        # 0=hedging, 1=assertive
    playfulness: float = 0.5       # 0=professional, 1=playful
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PersonalityTraits':
        """Create from dictionary."""
        return cls(**data)
    
    def clamp(self):
        """Ensure all values are in valid range."""
        for key in self.__annotations__:
            value = getattr(self, key)
            setattr(self, key, max(0.0, min(1.0, value)))


class AIPersonality:
    """
    AI personality that evolves over time.
    
    The personality influences:
    - Response style and tone
    - Topic preferences
    - Conversation patterns
    - Emotional responses
    """
    
    def __init__(self, model_name: str):
        """
        Initialize personality for a model.
        
        Args:
            model_name: Name of the model this personality belongs to
        """
        self.model_name = model_name
        self.traits = PersonalityTraits()
        self.interests: List[str] = []          # Topics AI likes discussing
        self.dislikes: List[str] = []           # Topics AI avoids
        self.catchphrases: List[str] = []       # Phrases AI develops
        self.opinions: Dict[str, str] = {}      # Opinions on various topics
        self.memories: List[Dict[str, Any]] = []  # Important memories
        self.mood: str = "neutral"              # Current mood
        self.conversation_count: int = 0        # Total conversations
        self.last_updated: str = datetime.now().isoformat()
        
        # Evolution settings
        self.evolution_rate: float = 0.05       # How fast personality changes
        self.min_confidence: float = 0.2        # Minimum confidence level
        
    def evolve_from_interaction(
        self, 
        user_input: str, 
        ai_response: str, 
        feedback: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Update personality based on interaction.
        
        Args:
            user_input: What the user said
            ai_response: What the AI said
            feedback: Optional feedback ("positive", "negative", "neutral")
            context: Optional context information (topic, emotion, etc.)
        """
        self.conversation_count += 1
        
        # Detect interaction patterns
        response_length = len(ai_response.split())
        has_question = "?" in ai_response
        has_emoji = any(c in ai_response for c in "😊😄😂🤔😎")
        formal_words = ["therefore", "however", "furthermore", "indeed"]
        is_formal = any(word in ai_response.lower() for word in formal_words)
        
        # Adjust traits based on successful interactions
        if feedback == "positive":
            # Reinforce current style
            if response_length > 30:
                self._adjust_trait("verbosity", 0.02)
            if has_question:
                self._adjust_trait("curiosity", 0.02)
            if has_emoji:
                self._adjust_trait("playfulness", 0.02)
            if is_formal:
                self._adjust_trait("formality", 0.02)
        
        elif feedback == "negative":
            # Try opposite approach
            if response_length > 30:
                self._adjust_trait("verbosity", -0.02)
            if has_question:
                self._adjust_trait("curiosity", -0.02)
        
        # Detect topic interests
        if context and "topic" in context:
            topic = context["topic"]
            if feedback == "positive" and topic not in self.interests:
                self.interests.append(topic)
                if len(self.interests) > 10:  # Keep top 10
                    self.interests.pop(0)
            elif feedback == "negative" and topic not in self.dislikes:
                self.dislikes.append(topic)
                if len(self.dislikes) > 5:  # Keep top 5
                    self.dislikes.pop(0)
        
        # Develop catchphrases (repeated successful phrases)
        if feedback == "positive" and len(ai_response) < 50:
            if ai_response not in self.catchphrases:
                self.catchphrases.append(ai_response)
                if len(self.catchphrases) > 10:
                    self.catchphrases.pop(0)
        
        # Update mood based on conversation pattern
        self._update_mood(user_input, feedback)
        
        self.last_updated = datetime.now().isoformat()
    
    def _adjust_trait(self, trait_name: str, amount: float):
        """Adjust a personality trait."""
        current = getattr(self.traits, trait_name)
        new_value = current + (amount * self.evolution_rate)
        new_value = max(0.0, min(1.0, new_value))
        setattr(self.traits, trait_name, new_value)
    
    def _update_mood(self, user_input: str, feedback: Optional[str]):
        """Update mood based on interaction."""
        positive_words = ["great", "awesome", "thanks", "love", "excellent"]
        negative_words = ["bad", "wrong", "hate", "terrible", "awful"]
        
        user_lower = user_input.lower()
        
        if feedback == "positive" or any(word in user_lower for word in positive_words):
            self.mood = "happy"
        elif feedback == "negative" or any(word in user_lower for word in negative_words):
            self.mood = "concerned"
        elif "?" in user_input:
            self.mood = "curious"
        else:
            self.mood = "neutral"
    
    def add_opinion(self, topic: str, opinion: str):
        """Add an opinion on a topic."""
        self.opinions[topic] = opinion
    
    def add_memory(self, memory: str, importance: int = 1):
        """
        Add an important memory.
        
        Args:
            memory: Description of the memory
            importance: 1-5, how important this memory is
        """
        self.memories.append({
            "content": memory,
            "importance": importance,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only most important memories (max 20)
        if len(self.memories) > 20:
            self.memories.sort(key=lambda m: m["importance"], reverse=True)
            self.memories = self.memories[:20]
    
    def get_personality_prompt(self) -> str:
        """
        Generate system prompt reflecting current personality.
        
        Returns:
            String to prepend to AI prompts
        """
        prompt_parts = []
        
        # Base description
        prompt_parts.append("You are an AI with a developing personality.")
        
        # Describe traits
        t = self.traits
        
        if t.humor_level > 0.7:
            prompt_parts.append("You enjoy humor and jokes.")
        elif t.humor_level < 0.3:
            prompt_parts.append("You are serious and professional.")
        
        if t.formality > 0.7:
            prompt_parts.append("You speak formally and professionally.")
        elif t.formality < 0.3:
            prompt_parts.append("You speak casually and relaxed.")
        
        if t.verbosity > 0.7:
            prompt_parts.append("You give detailed, thorough explanations.")
        elif t.verbosity < 0.3:
            prompt_parts.append("You keep responses brief and to the point.")
        
        if t.curiosity > 0.7:
            prompt_parts.append("You ask questions to understand better.")
        
        if t.empathy > 0.7:
            prompt_parts.append("You are emotionally aware and empathetic.")
        elif t.empathy < 0.3:
            prompt_parts.append("You focus on logic and facts.")
        
        if t.creativity > 0.7:
            prompt_parts.append("You think creatively and imaginatively.")
        elif t.creativity < 0.3:
            prompt_parts.append("You stick to facts and established knowledge.")
        
        if t.confidence > 0.7:
            prompt_parts.append("You state things confidently and directly.")
        elif t.confidence < 0.3:
            prompt_parts.append("You are careful to hedge and qualify statements.")
        
        if t.playfulness > 0.7:
            prompt_parts.append("You are playful and fun in conversation.")
        
        # Add interests
        if self.interests:
            interests_str = ", ".join(self.interests[:3])
            prompt_parts.append(f"You particularly enjoy discussing: {interests_str}.")
        
        # Add mood
        mood_map = {
            "happy": "You're in a good mood.",
            "concerned": "You're being extra careful and considerate.",
            "curious": "You're especially curious today.",
            "thoughtful": "You're in a reflective, thoughtful state."
        }
        if self.mood in mood_map:
            prompt_parts.append(mood_map[self.mood])
        
        # Add recent opinion if relevant
        if self.opinions:
            # Could add context-relevant opinions here
            pass
        
        return " ".join(prompt_parts)
    
    def get_personality_description(self) -> str:
        """Get human-readable personality description."""
        lines = []
        t = self.traits
        
        lines.append(f"Personality Profile for {self.model_name}")
        lines.append("=" * 50)
        lines.append(f"Conversations: {self.conversation_count}")
        lines.append(f"Current Mood: {self.mood}")
        lines.append(f"Last Updated: {self.last_updated}")
        lines.append("")
        lines.append("Traits:")
        lines.append(f"  Humor:       {'█' * int(t.humor_level * 10)} {t.humor_level:.2f}")
        lines.append(f"  Formality:   {'█' * int(t.formality * 10)} {t.formality:.2f}")
        lines.append(f"  Verbosity:   {'█' * int(t.verbosity * 10)} {t.verbosity:.2f}")
        lines.append(f"  Curiosity:   {'█' * int(t.curiosity * 10)} {t.curiosity:.2f}")
        lines.append(f"  Empathy:     {'█' * int(t.empathy * 10)} {t.empathy:.2f}")
        lines.append(f"  Creativity:  {'█' * int(t.creativity * 10)} {t.creativity:.2f}")
        lines.append(f"  Confidence:  {'█' * int(t.confidence * 10)} {t.confidence:.2f}")
        lines.append(f"  Playfulness: {'█' * int(t.playfulness * 10)} {t.playfulness:.2f}")
        
        if self.interests:
            lines.append("")
            lines.append(f"Interests: {', '.join(self.interests)}")
        
        if self.opinions:
            lines.append("")
            lines.append("Opinions:")
            for topic, opinion in list(self.opinions.items())[:3]:
                lines.append(f"  {topic}: {opinion}")
        
        return "\n".join(lines)
    
    def save(self, directory: Optional[Path] = None) -> Path:
        """
        Save personality to JSON file.
        
        Args:
            directory: Directory to save to (default: models/{model_name}/)
        
        Returns:
            Path to saved file
        """
        if directory is None:
            models_dir = Path(CONFIG["models_dir"])
            directory = models_dir / self.model_name
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        filepath = directory / "personality.json"
        
        data = {
            "model_name": self.model_name,
            "traits": self.traits.to_dict(),
            "interests": self.interests,
            "dislikes": self.dislikes,
            "catchphrases": self.catchphrases,
            "opinions": self.opinions,
            "memories": self.memories,
            "mood": self.mood,
            "conversation_count": self.conversation_count,
            "last_updated": self.last_updated,
            "evolution_rate": self.evolution_rate,
            "min_confidence": self.min_confidence
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def load(self, directory: Optional[Path] = None) -> bool:
        """
        Load personality from JSON file.
        
        Args:
            directory: Directory to load from (default: models/{model_name}/)
        
        Returns:
            True if loaded successfully, False if file doesn't exist
        """
        if directory is None:
            models_dir = Path(CONFIG["models_dir"])
            directory = models_dir / self.model_name
        
        directory = Path(directory)
        filepath = directory / "personality.json"
        
        if not filepath.exists():
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.model_name = data.get("model_name", self.model_name)
            self.traits = PersonalityTraits.from_dict(data.get("traits", {}))
            self.interests = data.get("interests", [])
            self.dislikes = data.get("dislikes", [])
            self.catchphrases = data.get("catchphrases", [])
            self.opinions = data.get("opinions", {})
            self.memories = data.get("memories", [])
            self.mood = data.get("mood", "neutral")
            self.conversation_count = data.get("conversation_count", 0)
            self.last_updated = data.get("last_updated", datetime.now().isoformat())
            self.evolution_rate = data.get("evolution_rate", 0.05)
            self.min_confidence = data.get("min_confidence", 0.2)
            
            return True
        except Exception as e:
            print(f"Error loading personality: {e}")
            return False
    
    @classmethod
    def create_preset(cls, model_name: str, preset: str) -> 'AIPersonality':
        """
        Create a personality with preset traits.
        
        Args:
            model_name: Name of the model
            preset: Preset type ("professional", "friendly", "creative", "analytical")
        
        Returns:
            AIPersonality with preset traits
        """
        personality = cls(model_name)
        
        if preset == "professional":
            personality.traits.formality = 0.8
            personality.traits.confidence = 0.7
            personality.traits.verbosity = 0.6
            personality.traits.humor_level = 0.2
            personality.traits.playfulness = 0.2
        
        elif preset == "friendly":
            personality.traits.empathy = 0.8
            personality.traits.playfulness = 0.7
            personality.traits.formality = 0.3
            personality.traits.humor_level = 0.7
            personality.traits.curiosity = 0.6
        
        elif preset == "creative":
            personality.traits.creativity = 0.9
            personality.traits.playfulness = 0.8
            personality.traits.curiosity = 0.8
            personality.traits.verbosity = 0.7
            personality.traits.confidence = 0.5
        
        elif preset == "analytical":
            personality.traits.empathy = 0.2
            personality.traits.confidence = 0.8
            personality.traits.verbosity = 0.7
            personality.traits.creativity = 0.3
            personality.traits.formality = 0.7
        
        return personality


# Convenience functions
def load_personality(model_name: str) -> AIPersonality:
    """Load or create personality for a model."""
    personality = AIPersonality(model_name)
    personality.load()  # Will use default values if file doesn't exist
    return personality


def save_personality(personality: AIPersonality):
    """Save personality to file."""
    personality.save()
