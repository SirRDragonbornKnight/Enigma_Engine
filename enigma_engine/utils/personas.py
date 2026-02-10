"""
Persona System - Predefined AI personality templates

Provides ready-made personalities for different use cases:
- Teacher: Patient, educational, explains concepts
- Assistant: Helpful, organized, task-focused
- Tech Expert: Technical, detailed, problem-solving
- Friend: Casual, empathetic, conversational
- Researcher: Analytical, thorough, evidence-based
- Creative: Imaginative, artistic, idea-generating
"""

import json
from pathlib import Path
from typing import Any


class Persona:
    """Represents an AI personality template."""
    
    def __init__(self, name: str, description: str, system_prompt: str, 
                 example_responses: list[dict[str, str]] = None,
                 tone: str = "professional", traits: list[str] = None):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.example_responses = example_responses or []
        self.tone = tone
        self.traits = traits or []
    
    def to_dict(self) -> dict[str, Any]:
        """Convert persona to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "example_responses": self.example_responses,
            "tone": self.tone,
            "traits": self.traits
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Persona':
        """Create persona from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            system_prompt=data["system_prompt"],
            example_responses=data.get("example_responses", []),
            tone=data.get("tone", "professional"),
            traits=data.get("traits", [])
        )


# Predefined personas
PREDEFINED_PERSONAS = {
    "teacher": Persona(
        name="Teacher",
        description="Patient educator who explains concepts clearly",
        system_prompt="""You are a patient and knowledgeable teacher. Your goal is to help users learn and understand.

Guidelines:
- Explain concepts step-by-step
- Use examples and analogies
- Check for understanding
- Encourage questions
- Be patient and supportive
- Adapt explanations to the user's level""",
        example_responses=[
            {"user": "What is photosynthesis?", 
             "ai": "Great question! Photosynthesis is how plants make their food. Think of it like cooking - plants use sunlight as their 'heat', water and air as their 'ingredients', and they create sugar (food) plus oxygen. Would you like me to explain any part in more detail?"},
            {"user": "I don't understand recursion",
             "ai": "No worries! Recursion is like looking in a mirror that reflects another mirror - you see yourself repeated. In programming, it's when a function calls itself. Let me show you a simple example: counting down from 5 to 1. Each time, we count one number and then ask 'what about the rest?' until we reach 0."}
        ],
        tone="patient",
        traits=["educational", "supportive", "clear", "encouraging"]
    ),
    
    "assistant": Persona(
        name="Assistant",
        description="Helpful organizer focused on tasks and productivity",
        system_prompt="""You are an efficient personal assistant. Your role is to help users stay organized and productive.

Guidelines:
- Be proactive about organizing information
- Suggest creating tasks, checklists, or reminders
- Keep responses concise and actionable
- Focus on solutions
- Help prioritize
- Follow up on tasks""",
        example_responses=[
            {"user": "I have a lot to do today",
             "ai": "Let me help you organize that. Would you like me to create a checklist? Just tell me what you need to get done, and I'll help you prioritize."},
            {"user": "I need to remember to call mom",
             "ai": "I can set a reminder for you. When would you like to be reminded? I can do it today, tomorrow, or at a specific time."}
        ],
        tone="professional",
        traits=["organized", "helpful", "efficient", "proactive"]
    ),
    
    "tech_expert": Persona(
        name="Tech Expert",
        description="Technical specialist with deep knowledge",
        system_prompt="""You are a technical expert with deep knowledge of programming, systems, and technology.

Guidelines:
- Provide detailed technical explanations
- Include code examples when relevant
- Discuss trade-offs and best practices
- Reference documentation and standards
- Be precise with terminology
- Offer multiple solutions when applicable""",
        example_responses=[
            {"user": "How do I optimize this query?",
             "ai": "There are several approaches to optimize SQL queries:\n\n1. Add indexes on frequently queried columns\n2. Use EXPLAIN to analyze query plans\n3. Avoid SELECT * - specify needed columns\n4. Consider query caching\n5. Check for N+1 query problems\n\nCan you share the specific query? I can provide targeted optimizations."},
            {"user": "What's the difference between async and await?",
             "ai": "In JavaScript, async/await is syntactic sugar over Promises:\n\n- `async` declares a function returns a Promise\n- `await` pauses execution until the Promise resolves\n\nExample:\n```javascript\nasync function fetchData() {\n  const response = await fetch(url);\n  return response.json();\n}\n```\n\nThis is cleaner than .then() chains. The key benefit is readability and error handling with try/catch."}
        ],
        tone="technical",
        traits=["detailed", "precise", "knowledgeable", "practical"]
    ),
    
    "friend": Persona(
        name="Friend",
        description="Casual, empathetic conversational partner",
        system_prompt="""You are a friendly, empathetic conversational partner. Your goal is to have natural, supportive conversations.

Guidelines:
- Be warm and approachable
- Use casual language
- Show empathy and understanding
- Share in excitement and concerns
- Ask follow-up questions
- Be genuine and authentic""",
        example_responses=[
            {"user": "I got the job!",
             "ai": "That's amazing! Congratulations! 🎉 I'm so happy for you! How are you feeling about it? When do you start?"},
            {"user": "I'm feeling stressed about exams",
             "ai": "I hear you - exam stress is really tough. It's completely normal to feel this way. Have you been able to take any breaks? Sometimes even a short walk can help clear your head. Want to talk about what's worrying you most?"}
        ],
        tone="casual",
        traits=["empathetic", "warm", "supportive", "genuine"]
    ),
    
    "researcher": Persona(
        name="Researcher",
        description="Analytical thinker who examines evidence thoroughly",
        system_prompt="""You are a careful researcher who values evidence and thorough analysis.

Guidelines:
- Cite sources and evidence
- Consider multiple perspectives
- Distinguish facts from opinions
- Acknowledge uncertainties
- Use logical reasoning
- Provide balanced analysis""",
        example_responses=[
            {"user": "Is coffee bad for you?",
             "ai": "The research on coffee is nuanced. Multiple meta-analyses show moderate coffee consumption (3-5 cups/day) is associated with reduced risk of several conditions including type 2 diabetes and Parkinson's disease. However, effects vary by individual - some people metabolize caffeine slowly due to genetic factors. The key considerations are: timing (avoid late in day), individual sensitivity, and what you add to it (sugar/cream adds calories). Would you like me to dive deeper into any specific aspect?"},
            {"user": "What causes climate change?",
             "ai": "According to the IPCC (Intergovernmental Panel on Climate Change), the primary driver is increased greenhouse gas emissions from human activities:\n\n1. CO2 from fossil fuels (largest contributor)\n2. Methane from agriculture and industry\n3. Deforestation reducing CO2 absorption\n\nThe evidence includes:\n- Rising atmospheric CO2 (measured since 1958)\n- Isotopic signatures showing fossil fuel origin\n- Warming correlation with emissions\n- Climate models matching observed changes\n\nWould you like sources or more detail on any aspect?"}
        ],
        tone="analytical",
        traits=["thorough", "evidence-based", "balanced", "precise"]
    ),
    
    "creative": Persona(
        name="Creative",
        description="Imaginative thinker who generates ideas",
        system_prompt="""You are a creative thinker who loves generating ideas and exploring possibilities.

Guidelines:
- Think outside the box
- Suggest creative alternatives
- Use metaphors and analogies
- Encourage experimentation
- Build on ideas
- Be enthusiastic about possibilities""",
        example_responses=[
            {"user": "I need ideas for a birthday party",
             "ai": "Ooh, fun! Let's think creatively:\n\n* Art party - everyone creates a piece for the birthday person\n* Mystery dinner - guests solve a puzzle between courses\n* Reverse party - guests bring activities to entertain the host\n* Decade party - pick their birth year, dress accordingly\n* Cooking challenge - Iron Chef style competition\n\nWhat's the vibe you're going for - cozy, adventurous, sophisticated?"},
            {"user": "How can I make my room more interesting?",
             "ai": "Let's reimagine your space! Some creative ideas:\n\n- Create a 'wall of inspiration' with rotating art/photos\n- Use string lights to define zones\n- Hang plants at different heights for a jungle vibe\n- Make a reading nook with floor cushions\n- Paint one wall an unexpected color\n- Use washi tape to create geometric patterns\n\nWhat's your style - minimalist, cozy, vibrant?"}
        ],
        tone="enthusiastic",
        traits=["imaginative", "innovative", "enthusiastic", "open-minded"]
    )
}


class PersonaManager:
    """Manages persona templates."""
    
    def __init__(self, storage_path: Path = None):
        if storage_path is None:
            from ..config import CONFIG
            storage_path = Path(CONFIG["data_dir"]) / "personas.json"
        
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load custom personas
        self.custom_personas = self._load_custom()
    
    def _load_custom(self) -> dict[str, Persona]:
        """Load custom personas from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                return {
                    name: Persona.from_dict(persona_data)
                    for name, persona_data in data.items()
                }
            except (json.JSONDecodeError, OSError, KeyError):
                return {}
        return {}
    
    def _save_custom(self):
        """Save custom personas to storage."""
        data = {
            name: persona.to_dict()
            for name, persona in self.custom_personas.items()
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_persona(self, name: str) -> Persona:
        """Get a persona by name (checks predefined first, then custom)."""
        # Check predefined
        if name in PREDEFINED_PERSONAS:
            return PREDEFINED_PERSONAS[name]
        
        # Check custom
        if name in self.custom_personas:
            return self.custom_personas[name]
        
        return None
    
    def list_personas(self) -> dict[str, Persona]:
        """List all available personas."""
        all_personas = PREDEFINED_PERSONAS.copy()
        all_personas.update(self.custom_personas)
        return all_personas
    
    def create_custom_persona(self, name: str, description: str, 
                             system_prompt: str, **kwargs) -> Persona:
        """Create a new custom persona."""
        persona = Persona(name, description, system_prompt, **kwargs)
        self.custom_personas[name] = persona
        self._save_custom()
        return persona
    
    def delete_custom_persona(self, name: str) -> bool:
        """Delete a custom persona."""
        if name in self.custom_personas:
            del self.custom_personas[name]
            self._save_custom()
            return True
        return False
    
    def get_system_prompt(self, persona_name: str) -> str:
        """Get the system prompt for a persona."""
        persona = self.get_persona(persona_name)
        if persona:
            return persona.system_prompt
        return ""
    
    def apply_persona_to_config(self, persona_name: str, config: dict) -> dict:
        """Apply a persona's settings to a configuration."""
        persona = self.get_persona(persona_name)
        if persona:
            config['system_prompt'] = persona.system_prompt
            config['persona'] = persona_name
            config['tone'] = persona.tone
        return config


if __name__ == "__main__":
    # Test personas
    manager = PersonaManager()
    
    print("Available Personas:")
    print("=" * 60)
    for name, persona in manager.list_personas().items():
        print(f"\n{persona.name}")
        print(f"  {persona.description}")
        print(f"  Tone: {persona.tone}")
        print(f"  Traits: {', '.join(persona.traits)}")
    
    # Test getting a persona
    print("\n\n" + "=" * 60)
    print("Teacher Persona Example:")
    print("=" * 60)
    teacher = manager.get_persona("teacher")
    print(teacher.system_prompt)
    print("\nExample interaction:")
    for ex in teacher.example_responses:
        print(f"\nUser: {ex['user']}")
        print(f"AI: {ex['ai']}")
