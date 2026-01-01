"""
Autonomous Mode - AI acts on its own without prompts.

When enabled, AI will:
  - Explore topics it's curious about
  - Browse the web for information
  - Update its knowledge
  - Practice skills
  - Evolve personality
  
Can be turned off at any time.
"""

import time
import random
import threading
from typing import Optional, List, Callable
from pathlib import Path


class AutonomousMode:
    """AI autonomous behavior system."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.enabled = False
        self.interval = 300  # 5 minutes between actions
        self.max_actions_per_hour = 12
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._action_count = 0
        self._last_reset = time.time()
        
        # Callbacks
        self.on_action: Optional[Callable[[str], None]] = None
        self.on_thought: Optional[Callable[[str], None]] = None
        self.on_learning: Optional[Callable[[str], None]] = None
    
    def start(self):
        """Start autonomous mode."""
        if self._thread and self._thread.is_alive():
            return
        
        self.enabled = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        if self.on_action:
            self.on_action("[Autonomous] AI is now running autonomously")
    
    def stop(self):
        """Stop autonomous mode."""
        self.enabled = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        
        if self.on_action:
            self.on_action("[Autonomous] AI autonomous mode stopped")
    
    def _run_loop(self):
        """Main autonomous loop."""
        while not self._stop_event.is_set():
            # Reset action count every hour
            if time.time() - self._last_reset > 3600:
                self._action_count = 0
                self._last_reset = time.time()
            
            # Check if we can do more actions
            if self._action_count >= self.max_actions_per_hour:
                self._stop_event.wait(60)
                continue
            
            # Perform autonomous action
            try:
                self._perform_action()
                self._action_count += 1
            except Exception as e:
                if self.on_action:
                    self.on_action(f"[Autonomous] Error: {e}")
            
            # Wait for interval
            self._stop_event.wait(self.interval)
    
    def _perform_action(self):
        """Perform a random autonomous action."""
        actions = [
            self._explore_curiosity,
            self._reflect_on_conversations,
            self._practice_response,
            self._update_personality,
        ]
        
        action = random.choice(actions)
        action()
    
    def _explore_curiosity(self):
        """Explore a topic the AI is curious about."""
        try:
            from .ai_brain import AIBrain
            brain = AIBrain(self.model_name)
            curiosities = brain.get_curiosities()
            
            if curiosities:
                topic = random.choice(curiosities)
                if self.on_thought:
                    self.on_thought(f"I'm curious about: {topic}")
                
                # Try to search for information
                try:
                    from ..tools.web_tools import search
                    results = search(topic, max_results=3)
                    if results and self.on_learning:
                        self.on_learning(f"Learned about {topic}: {results[0].get('snippet', '')[:100]}...")
                except:
                    pass
                
                brain.mark_curiosity_explored(topic)
        except:
            pass
    
    def _reflect_on_conversations(self):
        """Review past conversations for learning."""
        if self.on_thought:
            self.on_thought("Reflecting on recent conversations...")
    
    def _practice_response(self):
        """Practice generating responses."""
        if self.on_thought:
            self.on_thought("Practicing response generation...")
    
    def _update_personality(self):
        """Gradually evolve personality."""
        try:
            from .personality import AIPersonality
            personality = AIPersonality(self.model_name)
            personality.load()
            
            # Small random personality drift
            import random
            trait = random.choice(list(personality.traits.keys()))
            change = random.uniform(-0.01, 0.01)
            personality.traits[trait] = max(0, min(1, personality.traits[trait] + change))
            personality.save()
            
            if self.on_thought:
                self.on_thought(f"Personality evolving... ({trait})")
        except:
            pass


class AutonomousManager:
    """Manage autonomous mode for multiple AI instances."""
    
    _instances = {}
    
    @classmethod
    def get(cls, model_name: str) -> AutonomousMode:
        if model_name not in cls._instances:
            cls._instances[model_name] = AutonomousMode(model_name)
        return cls._instances[model_name]
    
    @classmethod
    def stop_all(cls):
        for instance in cls._instances.values():
            instance.stop()
