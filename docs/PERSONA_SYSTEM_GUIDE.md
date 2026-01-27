# AI Persona System Guide

## Overview

The AI Persona System allows you to create, customize, copy, and share complete AI identities. Each persona includes personality traits, voice settings, avatar configuration, and behavior patterns.

## Key Features

- **Copy/Paste AI Cloning**: Duplicate any persona to create variants
- **Export/Import**: Share your AI configurations with others as `.forge-ai` files
- **Merge Personas**: Combine traits from multiple personas
- **Template Personas**: Start with pre-configured examples
- **Integration**: Works with existing personality, voice, and avatar systems

## Quick Start

### Using the GUI

1. Open ForgeAI and navigate to the **Persona** tab
2. You'll see your current persona listed (default: "Forge Assistant")
3. Use the buttons to:
   - **Copy Persona**: Create a variant of any persona
   - **Export to File**: Save as `.forge-ai` file to share
   - **Import from File**: Load a shared persona
   - **Load Template**: Start with a pre-made persona

### Using Python

```python
from forge_ai.core.persona import PersonaManager, AIPersona

# Get the manager
manager = PersonaManager()

# Get current persona
persona = manager.get_current_persona()
print(f"Current: {persona.name}")

# Create a copy
copy = manager.copy_persona(persona.id, "My Assistant")

# Export to share
manager.export_persona(copy.id, Path("my_ai.forge-ai"))

# Import from file
imported = manager.import_persona(Path("shared_ai.forge-ai"))

# Switch to a different persona
manager.set_current_persona(imported.id)
```

## Persona Structure

Each persona includes:

### Identity
- **ID**: Unique identifier
- **Name**: Display name
- **Description**: What this AI is about
- **Created/Modified**: Timestamps

### Personality
- **Personality Traits**: humor, formality, verbosity, curiosity, empathy, creativity, confidence, playfulness
- **Catchphrases**: Signature phrases
- **Knowledge Domains**: Topics this AI knows about

### Behavior
- **System Prompt**: Base instructions
- **Response Style**: concise, detailed, casual, balanced
- **Preferences**: User-defined settings

### Appearance & Voice
- **Voice Profile ID**: Which voice to use
- **Avatar Preset ID**: Which avatar to use

### Learning
- **Learning Data Path**: Training data for this persona
- **Model Weights Path**: Fine-tuned weights (optional)
- **Memories**: Important things to remember

## Templates

Four starter templates are included:

1. **Helpful Assistant** - Balanced, professional helper
2. **Creative Companion** - Playful, imaginative brainstormer
3. **Technical Expert** - Precise, detailed technical guide
4. **Casual Friend** - Relaxed, friendly conversationalist

Load any template in the GUI via "Load Template" button.

## Integration with Personality System

The persona system integrates with the existing `AIPersonality` class:

```python
from forge_ai.core.persona import get_persona_manager
from forge_ai.core.personality import personality_from_persona

# Get current persona
manager = get_persona_manager()
persona = manager.get_current_persona()

# Convert to AIPersonality
personality = manager.integrate_with_personality(persona)

# Or use convenience function
personality = personality_from_persona(persona.name)

# Personality now has traits from persona
print(f"Humor level: {personality.traits.humor_level}")
```

## File Format

Personas are stored as JSON files with `.forge-ai` extension:

```json
{
  "id": "unique_id",
  "name": "My AI",
  "personality_traits": {
    "humor_level": 0.7,
    "formality": 0.3,
    ...
  },
  "voice_profile_id": "default",
  "avatar_preset_id": "default",
  "system_prompt": "You are...",
  "response_style": "balanced",
  "knowledge_domains": ["topic1", "topic2"],
  "catchphrases": ["Hello!", "How can I help?"],
  ...
}
```

## Storage Structure

```
data/personas/
├── default/
│   ├── persona.json      # Main config
│   ├── personality.json  # Trait values (if saved)
│   ├── memories.json     # Important memories (if saved)
│   └── learning/         # Training data for this persona
├── my_assistant/
│   └── ...
└── templates/
    ├── helpful_assistant.forge-ai
    ├── creative_companion.forge-ai
    ├── technical_expert.forge-ai
    └── casual_friend.forge-ai
```

## Advanced Usage

### Merging Personas

Combine traits from two personas:

```python
# Merge persona 1 and persona 2
merged = manager.merge_personas(
    base_id="persona1_id",
    overlay_id="persona2_id",
    new_name="Merged AI"
)

# Traits are averaged, domains and phrases are combined
```

### Updating from Personality Evolution

If your `AIPersonality` evolves through conversations, update the persona:

```python
# After personality has evolved
manager.update_persona_from_personality(persona, personality)
```

### Copy with Learning Data

When copying, you can optionally include training data:

```python
copy = manager.copy_persona(
    source_id="original_id",
    new_name="Copy",
    copy_learning_data=True  # Include training data
)
```

## Best Practices

1. **Start with a Template**: Use a template as a starting point
2. **Iterate**: Copy and modify personas to find what works
3. **Export Often**: Save your best configurations
4. **Descriptive Names**: Use clear names like "Technical Helper" not "AI 1"
5. **Document Changes**: Use the description field to note modifications
6. **Test Before Sharing**: Try your persona before exporting

## Troubleshooting

### Persona not showing in list
- Check `data/personas/` directory exists
- Verify `persona.json` file is valid JSON
- Restart the application

### Import fails
- Ensure file is valid JSON
- Check that required fields are present
- Try importing a template first to verify system works

### Changes not saved
- Click "Save Changes" button after editing
- Check file permissions on `data/personas/` directory

## Examples

### Create a Gaming Persona

```python
from forge_ai.core.persona import PersonaManager, AIPersona

manager = PersonaManager()

gamer = AIPersona(
    id="gamer",
    name="Gaming Buddy",
    personality_traits={
        "humor_level": 0.8,
        "formality": 0.2,
        "playfulness": 0.9,
        "empathy": 0.6,
        "curiosity": 0.7,
        "creativity": 0.7,
        "confidence": 0.7,
        "verbosity": 0.4,
    },
    system_prompt="You're a gaming buddy who loves video games and helps with strategies.",
    response_style="casual",
    knowledge_domains=["gaming", "esports", "game design"],
    catchphrases=["Let's game!", "GG!", "One more round?"]
)

manager.save_persona(gamer)
```

### Create a Learning Tutor

```python
tutor = AIPersona(
    id="tutor",
    name="Study Companion",
    personality_traits={
        "humor_level": 0.4,
        "formality": 0.6,
        "playfulness": 0.3,
        "empathy": 0.8,
        "curiosity": 0.7,
        "creativity": 0.5,
        "confidence": 0.7,
        "verbosity": 0.8,
    },
    system_prompt="You're a patient tutor who helps students learn complex topics.",
    response_style="detailed",
    knowledge_domains=["education", "teaching", "learning strategies"],
    catchphrases=["Let's break this down", "Great question!", "You're making progress!"]
)

manager.save_persona(tutor)
```

## API Reference

See `forge_ai/core/persona.py` for complete API documentation.

Key classes:
- `AIPersona`: Dataclass representing a complete AI identity
- `PersonaManager`: Main interface for persona operations
- `get_persona_manager()`: Get global manager instance

Key methods:
- `get_current_persona()`: Get active persona
- `save_persona(persona)`: Save to disk
- `copy_persona(source_id, name)`: Clone a persona
- `export_persona(id, path)`: Export to file
- `import_persona(path)`: Import from file
- `merge_personas(base, overlay, name)`: Combine personas
- `integrate_with_personality(persona)`: Convert to AIPersonality
