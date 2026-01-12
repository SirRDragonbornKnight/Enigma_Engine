# Voice Customization Guide

## Overview

Enigma supports voice customization through:
1. AI-generated voices from personality traits
2. User-provided voice samples
3. Voice evolution as personality changes

## Quick Start

### Generate Voice from Personality

```python
from ai_tester.core.personality import load_personality
from ai_tester.voice.voice_generator import generate_voice_for_personality

personality = load_personality("my_model")
voice = generate_voice_for_personality(personality)

# Use the voice
from ai_tester.voice.voice_profile import VoiceEngine
engine = VoiceEngine()
engine.set_profile(voice)
engine.speak("Hello! This is my personality-matched voice!")
```

### Create Voice from Samples

```python
from ai_tester.voice.voice_generator import create_voice_from_samples

# Provide audio samples (.wav, .mp3)
voice = create_voice_from_samples(
    audio_files=["sample1.wav", "sample2.wav", "sample3.wav"],
    name="my_custom_voice",
    model_name="my_model"
)
```

## Voice Parameters

Generated voices map personality traits to voice parameters:

| Personality Trait | Voice Effect |
|------------------|--------------|
| High Confidence | Lower pitch, slower speed |
| High Playfulness | Higher pitch, faster speed, more variation |
| High Formality | Neutral pitch, measured pace |
| High Empathy | Softer volume, warmer tone |
| High Humor | Higher pitch, faster speed |

## Preset Voice Profiles

Available presets:
- **default**: Neutral, balanced
- **glados**: Cold, sarcastic (Portal AI)
- **jarvis**: Formal British assistant
- **wheatley**: Nervous, fast-talking
- **hal9000**: Calm, deliberate
- **robot**: Classic robotic voice
- **cheerful**: Happy, energetic
- **wise**: Slow, thoughtful

```python
from ai_tester.voice.voice_profile import speak, set_voice

set_voice("glados")
speak("Hello, test subject.")
```

## Voice Evolution

Voices can evolve with personality:

```python
from ai_tester.voice.voice_generator import AIVoiceGenerator

generator = AIVoiceGenerator()
updated_voice = generator.evolve_voice(
    profile=current_voice,
    personality=updated_personality,
    evolution_rate=0.1
)
```

## Storage Location

Voice data is stored in:
- Profiles: `data/voice_profiles/{name}.json`
- Samples: `models/{model_name}/voice/samples/`
- Evolution history: `models/{model_name}/voice/evolution.json`

## See Also

- [PERSONALITY.md](PERSONALITY.md) - AI personality system
- [Voice Profile API](../ai_tester/voice/voice_profile.py)
