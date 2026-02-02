"""
Built-in Chat Bot

A simple rule-based chatbot that works without any AI model.
Provides basic conversation when torch/models aren't available.
"""

import re
import random
import time
from datetime import datetime
from typing import Dict, Any, List, Optional


class BuiltinChat:
    """
    Built-in rule-based chatbot.
    No neural network - just pattern matching and templates.
    """
    
    def __init__(self):
        self.is_loaded = False
        self.conversation_history: List[Dict[str, str]] = []
        self.user_name = "User"
        self.bot_name = "Forge"
        
        # Response patterns: (regex, responses)
        self.patterns = [
            # Greetings
            (r'\b(hi|hello|hey|howdy|greetings)\b', [
                "Hello! How can I help you today?",
                "Hi there! What's on your mind?",
                "Hey! Nice to chat with you.",
                "Hello! I'm Forge, your assistant.",
            ]),
            
            # How are you
            (r'\bhow are you\b', [
                "I'm doing well, thanks for asking! How about you?",
                "I'm great! Ready to help with whatever you need.",
                "Functioning perfectly! What can I do for you?",
            ]),
            
            # Name questions
            (r'\b(what.?s your name|who are you)\b', [
                f"I'm {self.bot_name}, your AI assistant!",
                f"My name is {self.bot_name}. I'm here to help!",
                f"I go by {self.bot_name}. How can I assist you?",
            ]),
            
            # Capability questions
            (r'\b(what can you do|help me|capabilities)\b', [
                "I can chat with you, answer questions, and help with various tasks. "
                "For full AI capabilities, make sure torch is installed!",
                "I'm a basic assistant. I can have conversations and provide information. "
                "Install the full dependencies for AI-powered features!",
            ]),
            
            # Thanks
            (r'\b(thanks|thank you|thx)\b', [
                "You're welcome!",
                "Happy to help!",
                "Anytime! Let me know if you need anything else.",
                "No problem at all!",
            ]),
            
            # Goodbye
            (r'\b(bye|goodbye|see you|later)\b', [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Bye! Come back anytime!",
            ]),
            
            # Time/date
            (r'\b(what time|current time)\b', [
                lambda: f"The current time is {datetime.now().strftime('%I:%M %p')}.",
            ]),
            (r'\b(what.?s the date|today.?s date|what day)\b', [
                lambda: f"Today is {datetime.now().strftime('%A, %B %d, %Y')}.",
            ]),
            
            # Math
            (r'(\d+)\s*[\+]\s*(\d+)', [
                lambda m: f"That equals {int(m.group(1)) + int(m.group(2))}.",
            ]),
            (r'(\d+)\s*[\-]\s*(\d+)', [
                lambda m: f"That equals {int(m.group(1)) - int(m.group(2))}.",
            ]),
            (r'(\d+)\s*[\*x]\s*(\d+)', [
                lambda m: f"That equals {int(m.group(1)) * int(m.group(2))}.",
            ]),
            (r'(\d+)\s*[\/]\s*(\d+)', [
                lambda m: f"That equals {int(m.group(1)) / int(m.group(2)):.2f}." if int(m.group(2)) != 0 else "Can't divide by zero!",
            ]),
            
            # Jokes
            (r'\b(tell.+joke|joke)\b', [
                "Why do programmers prefer dark mode? Because light attracts bugs!",
                "Why did the AI go to therapy? It had too many deep issues!",
                "What's a computer's favorite snack? Microchips!",
                "Why was the JavaScript developer sad? Because he didn't Node how to Express himself!",
                "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
            ]),
            
            # Weather (mock)
            (r'\b(weather|temperature)\b', [
                "I can't check real weather without an API, but I hope it's nice where you are!",
                "I don't have weather data, but you can check weather.com for forecasts!",
            ]),
            
            # Feelings
            (r'\bi.?m (sad|depressed|unhappy)\b', [
                "I'm sorry to hear that. Remember, it's okay to feel down sometimes. Is there anything I can help with?",
                "That's tough. Would you like to talk about it, or would you prefer a distraction?",
            ]),
            (r'\bi.?m (happy|excited|great)\b', [
                "That's wonderful to hear! What's making you feel so good?",
                "Awesome! It's great that you're in good spirits!",
            ]),
            
            # Programming help
            (r'\b(python|javascript|code|programming)\b', [
                "I can help with basic coding questions! For AI-powered code generation, "
                "check the Code tab or install the full dependencies.",
                "Programming is fun! What language are you working with?",
            ]),
            
            # ForgeAI specific
            (r'\b(forge|forgeai)\b', [
                "ForgeAI is a modular AI framework! You can train models, generate images, "
                "create code, and more. Check the tabs for different features!",
                "I'm part of ForgeAI! This is the built-in fallback chat when the full AI model isn't loaded.",
            ]),
            
            # Yes/No
            (r'^(yes|yeah|yep|yup)$', [
                "Great! What would you like to do next?",
                "Alright! How can I help?",
            ]),
            (r'^(no|nope|nah)$', [
                "Okay, no problem. Let me know if you change your mind!",
                "Alright! Is there something else I can help with?",
            ]),
        ]
        
        # Default responses for when nothing matches
        self.default_responses = [
            "Interesting! Tell me more about that.",
            "I see. Could you elaborate?",
            "That's a good point. What else is on your mind?",
            "I'm not sure I fully understand. Could you rephrase that?",
            "Hmm, I'm just a simple chatbot. For complex questions, you might need the full AI model!",
            "I'm listening! Go on...",
            "That's beyond my basic capabilities. Try loading a trained model for better responses!",
        ]
    
    def load(self) -> bool:
        """Load the chatbot."""
        self.is_loaded = True
        return True
    
    def unload(self):
        """Unload."""
        self.is_loaded = False
        self.conversation_history.clear()
    
    def _find_response(self, text: str) -> str:
        """Find a matching response for the input."""
        text_lower = text.lower().strip()
        
        # Track response indices for cycling
        if not hasattr(self, '_response_cycle_indices'):
            self._response_cycle_indices = {}
        
        for i, (pattern, responses) in enumerate(self.patterns):
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                # Cycle through responses for this pattern
                pattern_key = f"pattern_{i}"
                if pattern_key not in self._response_cycle_indices:
                    self._response_cycle_indices[pattern_key] = 0
                
                idx = self._response_cycle_indices[pattern_key] % len(responses)
                response = responses[idx]
                self._response_cycle_indices[pattern_key] = (idx + 1) % len(responses)
                
                if callable(response):
                    try:
                        return response(match) if match.groups() else response()
                    except Exception:
                        continue
                return response
        
        # Cycle through default responses
        if 'default' not in self._response_cycle_indices:
            self._response_cycle_indices['default'] = 0
        
        idx = self._response_cycle_indices['default'] % len(self.default_responses)
        response = self.default_responses[idx]
        self._response_cycle_indices['default'] = (idx + 1) % len(self.default_responses)
        return response
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response to the prompt."""
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        
        if not prompt.strip():
            return {"success": False, "error": "Empty prompt"}
        
        try:
            start = time.time()
            
            # Store in history
            self.conversation_history.append({"role": "user", "content": prompt})
            
            # Generate response
            response = self._find_response(prompt)
            
            # Store response
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Keep history manageable
            if len(self.conversation_history) > 50:
                self.conversation_history = self.conversation_history[-50:]
            
            return {
                "success": True,
                "response": response,
                "duration": time.time() - start,
                "is_builtin": True,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def chat(self, message: str) -> str:
        """Simple chat interface returning just the response string."""
        result = self.generate(message)
        return result.get("response", "I'm not sure how to respond to that.")
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()
