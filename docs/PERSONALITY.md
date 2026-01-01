# AI Personality System

## Overview

Enigma Engine includes a self-evolving personality system that allows your AI to develop its own unique character over time through interactions.

## Features

- **8 Core Traits**: Humor, formality, verbosity, curiosity, empathy, creativity, confidence, playfulness
- **Dynamic Evolution**: Personality changes based on conversation patterns and feedback
- **Interests & Opinions**: AI develops preferences for topics and forms opinions
- **Mood System**: Current emotional state affects responses
- **Memory System**: Important interactions are remembered
- **Catchphrases**: AI develops its own unique expressions

## Getting Started

### Creating a Personality

```python
from enigma.core.personality import AIPersonality

# Create new personality
personality = AIPersonality("my_model")

# Or create with preset
personality = AIPersonality.create_preset("my_model", "friendly")
# Presets: professional, friendly, creative, analytical
```

### Loading Existing Personality

```python
from enigma.core.personality import load_personality

personality = load_personality("my_model")
print(personality.get_personality_description())
```

### Evolving the Personality

```python
# During conversation
user_input = "Tell me a joke"
ai_response = "Why did the AI go to school? To improve its neural networks!"

# Update personality based on interaction
personality.evolve_from_interaction(
    user_input=user_input,
    ai_response=ai_response,
    feedback="positive",  # or "negative" or "neutral"
    context={"topic": "humor"}
)

# Save changes
personality.save()
```

### Using Personality in Generation

```python
# Get personality-influenced prompt
system_prompt = personality.get_personality_prompt()

# Use in your model
full_prompt = f"{system_prompt}\n\nUser: {user_input}\nAI:"
response = model.generate(full_prompt)
```

## Personality Traits

### Humor Level (0.0 to 1.0)
- **0.0-0.3**: Serious and professional
- **0.4-0.6**: Occasionally playful
- **0.7-1.0**: Frequently uses humor and jokes

### Formality (0.0 to 1.0)
- **0.0-0.3**: Very casual, uses slang
- **0.4-0.6**: Balanced tone
- **0.7-1.0**: Formal and professional

### Verbosity (0.0 to 1.0)
- **0.0-0.3**: Brief and concise
- **0.4-0.6**: Moderate detail
- **0.7-1.0**: Detailed and thorough

### Curiosity (0.0 to 1.0)
- **0.0-0.3**: Answers only what's asked
- **0.4-0.6**: Occasionally asks questions
- **0.7-1.0**: Frequently asks follow-up questions

### Empathy (0.0 to 1.0)
- **0.0-0.3**: Logical and fact-focused
- **0.4-0.6**: Balanced emotional awareness
- **0.7-1.0**: Highly emotionally aware

### Creativity (0.0 to 1.0)
- **0.0-0.3**: Sticks to facts
- **0.4-0.6**: Occasionally creative
- **0.7-1.0**: Highly imaginative

### Confidence (0.0 to 1.0)
- **0.0-0.3**: Hedges and qualifies statements
- **0.4-0.6**: Moderately assertive
- **0.7-1.0**: Direct and confident

### Playfulness (0.0 to 1.0)
- **0.0-0.3**: Professional and serious
- **0.4-0.6**: Occasionally playful
- **0.7-1.0**: Fun and lighthearted

## Advanced Features

### Adding Opinions

```python
personality.add_opinion(
    topic="Python",
    opinion="I love Python! It's elegant and powerful."
)
```

### Adding Memories

```python
personality.add_memory(
    memory="User helped me understand recursion",
    importance=5  # 1-5 scale
)
```

### Customizing Evolution Rate

```python
# Faster evolution
personality.evolution_rate = 0.1  # default: 0.05

# Slower evolution
personality.evolution_rate = 0.02
```

## Training Data

To train your model with personality awareness, include examples from:

- `data/personality_development.txt` - Personality evolution examples
- `data/self_awareness_training.txt` - Self-awareness examples
- `data/combined_action_training.txt` - Personality-influenced actions

## Integration with Other Systems

### Voice Integration

Personality automatically influences voice profile generation:

```python
from enigma.voice.voice_generator import generate_voice_for_personality

voice_profile = generate_voice_for_personality(personality)
```

### GUI Integration

The personality can be viewed and edited in the GUI Settings tab.

### Web Dashboard

Access personality settings at `/settings` in the web dashboard.

## Best Practices

1. **Give Feedback**: Regularly provide feedback to guide evolution
2. **Be Consistent**: Consistent feedback leads to clearer personality
3. **Save Often**: Personality changes are only persisted when you save
4. **Monitor Growth**: Check personality description periodically
5. **Balance Traits**: Extreme values (0.0 or 1.0) can lead to rigid behavior

## Example: Complete Workflow

```python
from enigma.core.personality import AIPersonality
from enigma.core.inference import InferenceEngine

# Setup
model = "my_chatbot"
personality = AIPersonality(model)
personality.load()  # Load existing or create new
engine = InferenceEngine(model_name=model)

# Chat loop
while True:
    user_input = input("You: ")
    
    # Get personality-influenced prompt
    system_prompt = personality.get_personality_prompt()
    full_prompt = f"{system_prompt}\n\nUser: {user_input}\nAI:"
    
    # Generate
    response = engine.generate(full_prompt)
    print(f"AI: {response}")
    
    # Get feedback (optional)
    feedback = input("Feedback (good/bad/enter to skip): ").strip()
    if feedback:
        personality.evolve_from_interaction(
            user_input=user_input,
            ai_response=response,
            feedback="positive" if feedback == "good" else "negative"
        )
        personality.save()
```

## Troubleshooting

### Personality Not Evolving
- Check that `evolution_rate > 0`
- Ensure you're calling `save()` after changes
- Provide feedback regularly

### Personality File Not Found
- Make sure `models/{model_name}/` directory exists
- Call `personality.save()` to create the file

### Traits Not Affecting Output
- Ensure you're using `get_personality_prompt()` in your generation
- Train your model with personality-aware data

## Technical Details

### File Format

Personality is saved as `models/{model_name}/personality.json`:

```json
{
  "model_name": "my_model",
  "traits": {
    "humor_level": 0.7,
    "formality": 0.3,
    ...
  },
  "interests": ["technology", "science"],
  "opinions": {...},
  "mood": "excited",
  "conversation_count": 150,
  ...
}
```

### Memory Structure

```python
personality.memories = [
    {
        "content": "User taught me about quantum computing",
        "importance": 4,
        "timestamp": "2024-01-01T12:00:00"
    },
    ...
]
```

## Further Reading

- [Voice Customization](VOICE_CUSTOMIZATION.md) - Personality-driven voices
- [Training Guide](../HOW_TO_MAKE_AI.txt) - Training with personality data
- [Web Dashboard](WEB_MOBILE.md) - Web interface for personality

## Support

For questions or issues, please:
1. Check the troubleshooting section
2. Review the example code
3. Create an issue on GitHub
