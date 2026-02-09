"""
Multi-AI Communication System for Forge

Enables multiple AI models to communicate with each other:
  - Local mode: Multiple models in same process
  - Network mode: Models on different devices
  
Usage:
    from enigma_engine.comms.multi_ai import AIConversation
    
    # Create conversation between two models
    conv = AIConversation()
    conv.add_participant("forge_1", model_name="enigma_engine")
    conv.add_participant("forge_2", model_name="enigma_engine")
    
    # Start conversation
    for exchange in conv.converse("Hello, who are you?", num_turns=5):
        print(f"{exchange['speaker']}: {exchange['message']}")
"""

import json
import time
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

from ..config import CONFIG


class AIParticipant:
    """Represents one AI in a multi-AI conversation."""
    
    def __init__(
        self,
        name: str,
        model_name: str = None,
        remote_url: str = None,
        personality: str = None
    ):
        """
        Args:
            name: Unique identifier for this AI
            model_name: Local model name (for local inference)
            remote_url: URL of remote Forge node (for network mode)
            personality: Optional personality prompt prefix
        """
        self.name = name
        self.model_name = model_name
        self.remote_url = remote_url
        self.personality = personality or ""
        self._engine = None
        self._history = []
    
    @property
    def engine(self):
        """Get inference engine (lazy loaded)."""
        if self._engine is None and self.model_name:
            try:
                from ..core.inference import EnigmaEngine
                from ..core.model_registry import ModelRegistry
                
                registry = ModelRegistry()
                if self.model_name in registry.registry.get("models", {}):
                    model_path = registry.registry["models"][self.model_name]["path"]
                    self._engine = EnigmaEngine(model_path=model_path)
                else:
                    # Use default model
                    self._engine = EnigmaEngine()
            except Exception as e:
                print(f"[!] Could not load engine for {self.name}: {e}")
        return self._engine
    
    def respond(self, prompt: str, context: list[dict] = None) -> str:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: The message to respond to
            context: Previous conversation context
            
        Returns:
            The AI's response
        """
        # Build full prompt with context and personality
        full_prompt = ""
        
        if self.personality:
            full_prompt += f"[{self.personality}]\n\n"
        
        if context:
            for msg in context[-10:]:  # Last 10 messages
                full_prompt += f"{msg['speaker']}: {msg['message']}\n"
        
        full_prompt += f"You: {prompt}\n{self.name}:"
        
        # Remote mode
        if self.remote_url:
            return self._remote_respond(full_prompt)
        
        # Local mode
        if self.engine:
            try:
                response = self.engine.generate(full_prompt, max_tokens=100)
                # Clean response
                response = response.strip()
                if response.startswith(self.name + ":"):
                    response = response[len(self.name)+1:].strip()
                return response
            except Exception as e:
                return f"[Error generating response: {e}]"
        
        return "[No engine available]"
    
    def _remote_respond(self, prompt: str) -> str:
        """Get response from remote Forge node."""
        try:
            import urllib.parse
            import urllib.request
            
            data = json.dumps({"prompt": prompt}).encode()
            req = urllib.request.Request(
                f"{self.remote_url}/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
                return result.get("response", "[No response]")
        except Exception as e:
            return f"[Remote error: {e}]"


class AIConversation:
    """
    Manages a conversation between multiple AIs.
    
    Features:
        - Turn-based conversation
        - Conversation logging
        - Topic control
        - Personality injection
    """
    
    def __init__(self, name: str = None, log_callback: Callable = None, max_history: int = 500):
        """
        Args:
            name: Name for this conversation
            log_callback: Function to call with each exchange (for GUI)
            max_history: Maximum history entries to keep (0 = unlimited for saving)
        """
        self.name = name or f"conv_{int(time.time())}"
        self.participants: dict[str, AIParticipant] = {}
        self.history: list[dict] = []
        self.log_callback = log_callback
        self.created = datetime.now().isoformat()
        self._max_history = max_history
    
    def add_participant(
        self,
        name: str,
        model_name: str = None,
        remote_url: str = None,
        personality: str = None
    ) -> "AIConversation":
        """
        Add an AI participant to the conversation.
        
        Args:
            name: Unique name for this participant
            model_name: Model to use (local mode)
            remote_url: Remote Forge URL (network mode)
            personality: Personality prompt
            
        Returns:
            self for chaining
        """
        self.participants[name] = AIParticipant(
            name=name,
            model_name=model_name,
            remote_url=remote_url,
            personality=personality
        )
        return self
    
    def converse(
        self,
        initial_prompt: str,
        num_turns: int = 5,
        turn_order: list[str] = None
    ) -> Generator[dict, None, None]:
        """
        Have the AIs converse with each other.
        
        Args:
            initial_prompt: The starting message
            num_turns: Number of exchanges
            turn_order: Order of speakers (default: round-robin)
            
        Yields:
            Dict with speaker, message, timestamp for each turn
        """
        if not self.participants:
            yield {"speaker": "system", "message": "No participants", "timestamp": datetime.now().isoformat()}
            return
        
        # Default turn order
        if turn_order is None:
            turn_order = list(self.participants.keys())
        
        # First message
        current_message = initial_prompt
        first_speaker = turn_order[0]
        
        # Log initial prompt as "user" input
        initial_exchange = {
            "speaker": "user",
            "message": initial_prompt,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(initial_exchange)
        yield initial_exchange
        
        # Run conversation
        for turn in range(num_turns):
            speaker_name = turn_order[turn % len(turn_order)]
            speaker = self.participants[speaker_name]
            
            # Get response
            response = speaker.respond(current_message, context=self.history)
            
            # Create exchange record
            exchange = {
                "speaker": speaker_name,
                "message": response,
                "timestamp": datetime.now().isoformat(),
                "turn": turn + 1
            }
            
            # Store and notify
            self.history.append(exchange)
            # Trim history if limit set
            if self._max_history > 0 and len(self.history) > self._max_history:
                self.history = self.history[-self._max_history:]
            if self.log_callback:
                self.log_callback(exchange)
            
            yield exchange
            
            # Next turn uses this response
            current_message = response
    
    def save(self, filepath: str = None) -> str:
        """Save conversation to file."""
        if filepath is None:
            conv_dir = CONFIG.get("data_dir", Path("data")) / "ai_conversations"
            conv_dir.mkdir(parents=True, exist_ok=True)
            filepath = conv_dir / f"{self.name}.json"
        
        data = {
            "name": self.name,
            "created": self.created,
            "participants": list(self.participants.keys()),
            "history": self.history
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> "AIConversation":
        """Load conversation from file."""
        with open(filepath) as f:
            data = json.load(f)
        
        conv = cls(name=data.get("name"))
        conv.created = data.get("created")
        conv.history = data.get("history", [])
        
        # Re-add participants (without models loaded)
        for name in data.get("participants", []):
            conv.participants[name] = AIParticipant(name=name)
        
        return conv


def quick_ai_chat(
    prompt: str,
    model1: str = "enigma_engine",
    model2: str = "enigma_engine",
    num_turns: int = 3
) -> list[dict]:
    """
    Quick function to have two AIs chat.
    
    Args:
        prompt: Starting prompt
        model1: First AI model name
        model2: Second AI model name
        num_turns: Number of exchanges
        
    Returns:
        List of conversation exchanges
    """
    conv = AIConversation()
    conv.add_participant("AI_1", model_name=model1)
    conv.add_participant("AI_2", model_name=model2)
    
    return list(conv.converse(prompt, num_turns=num_turns))
