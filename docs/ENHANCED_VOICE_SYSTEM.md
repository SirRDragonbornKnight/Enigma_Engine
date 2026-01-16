# Enhanced Voice System Documentation

## Overview

The Enhanced Voice System allows the AI to create its own voice OR the user to fully customize it - this applies to all voice features. This document covers all new voice capabilities.

## Table of Contents

1. [AI Voice Self-Discovery](#ai-voice-self-discovery)
2. [Natural Language Voice Descriptions](#natural-language-voice-descriptions)
3. [Enhanced Voice Effects](#enhanced-voice-effects)
4. [Dynamic Voice Adaptation](#dynamic-voice-adaptation)
5. [Smart Wake Words](#smart-wake-words)
6. [Voice Cloning](#voice-cloning)
7. [Interactive Voice Customization](#interactive-voice-customization)
8. [Personality Integration](#personality-integration)

---

## AI Voice Self-Discovery

The AI can experiment with voice settings and discover what "feels right" based on its personality.

### Basic Usage

```python
from forge_ai.voice import discover_voice
from forge_ai.core.personality import load_personality

# Load AI personality
personality = load_personality("my_model")

# AI discovers its voice
voice_profile = discover_voice(personality, num_experiments=5)

# Use the discovered voice
from forge_ai.voice import set_voice, speak
set_voice(voice_profile)
speak("Hello! This is my discovered voice.")
```

### Advanced Discovery

```python
from forge_ai.voice.voice_identity import AIVoiceIdentity

identity = AIVoiceIdentity()

# Discover voice with more control
voice = identity.discover_voice(
    personality=personality,
    num_experiments=10,
    base_voice="female"
)

# Save the discovered identity
identity.save_identity("my_model")

# Later, load it back
identity.load_identity("my_model")
```

### How It Works

The AI:
1. Generates a base voice from personality traits
2. Creates variations by adjusting pitch, speed, and volume
3. Scores each variation based on personality alignment
4. Returns the best-matching voice profile

---

## Natural Language Voice Descriptions

The AI can describe what voice it wants in natural language, which is parsed into parameters.

### Examples

```python
from forge_ai.voice import describe_voice, speak

# Warm and calm
voice = describe_voice("I want a warm, calm voice that speaks slowly")
# Result: pitch=0.7, speed=0.8, effects=['warm', 'calm']

# Energetic and fast
voice = describe_voice("I want an energetic, fast, high-pitched voice")
# Result: pitch=1.3, speed=1.2, effects=['energetic']

# Deep and authoritative
voice = describe_voice("I want a deep, authoritative, confident voice")
# Result: pitch=0.7, effects=['authoritative']

# Robotic
voice = describe_voice("Give me a robotic, mechanical voice")
# Result: effects=['robotic']
```

### Supported Keywords

**Pitch:**
- Low/deep/bass → Lower pitch
- High/light/bright → Higher pitch

**Speed:**
- Slow/measured/deliberate → Slower speed
- Fast/quick/rapid → Faster speed

**Volume:**
- Quiet/soft/gentle → Lower volume
- Loud/strong/powerful → Higher volume

**Voice Type:**
- Female/woman/feminine
- Male/man/masculine

**Effects:**
- Robotic/robot/mechanical
- Warm/friendly/gentle
- Cold/clinical/detached
- Energetic/excited/enthusiastic
- Calm/peaceful/relaxed
- Confident/authoritative/commanding
- Whisper/whispering

---

## Enhanced Voice Effects

The system supports multiple voice effects that can be combined and applied dynamically.

### Available Effects

| Effect | Description | Text Transform |
|--------|-------------|----------------|
| `robotic` | Mechanical voice with pauses | Yes (levels 1-3) |
| `whisper` | Soft, quiet whisper | Yes + audio |
| `echo` | Echo/reverb effect | Yes + audio |
| `warm` | Warm, friendly tone | Yes |
| `cold` | Cold, clinical tone | Yes |
| `energetic` | Energetic, enthusiastic | Yes |
| `calm` | Calm, soothing | Yes |
| `authoritative` | Authoritative, commanding | Yes |

### Basic Usage

```python
from forge_ai.voice import apply_effect, apply_effects

text = "Hello, how are you today?"

# Apply single effect
robotic_text = apply_effect(text, "robotic", level=2)
# Result: "Hello,. how are you today?"

# Apply multiple effects
text_with_effects = apply_effects(text, ["warm", "calm"])
```

### Effect Levels

The `robotic` effect supports intensity levels:

```python
from forge_ai.voice.voice_effects import VoiceEffects

effects = VoiceEffects()

# Level 1: Mild pauses
text = effects.apply_effect("Hello. How are you?", "robotic", level=1)
# "Hello... How are you?"

# Level 2: Medium pauses
text = effects.apply_effect("Hello, friend.", "robotic", level=2)
# "Hello,. friend..."

# Level 3: Strong pauses between words
text = effects.apply_effect("Hello world", "robotic", level=3)
# "Hello. world"
```

### Conflict Resolution

Some effects conflict (e.g., warm + cold). The system resolves these automatically:

```python
effects = VoiceEffects()

# Conflicting effects
combined = effects.combine_effects(["warm", "cold"])
# Result: One is removed based on priority

# Compatible effects
combined = effects.combine_effects(["warm", "calm"])
# Result: Both effects are kept
```

---

## Dynamic Voice Adaptation

Voice changes automatically based on emotion, context, or personality mood.

### Emotional Adaptation

```python
from forge_ai.voice import adapt_voice_for_emotion, VoiceProfile

base_voice = VoiceProfile(pitch=1.0, speed=1.0)

# Happy emotion
happy_voice = adapt_voice_for_emotion("happy", base_voice)
# Result: Higher pitch, faster speed, "energetic" effect

# Sad emotion
sad_voice = adapt_voice_for_emotion("sad", base_voice)
# Result: Lower pitch, slower speed, "calm" effect
```

### Supported Emotions

- `happy` - Higher pitch, faster, energetic
- `sad` - Lower pitch, slower, calm
- `excited` - Much higher pitch, much faster, energetic
- `serious` - Slightly lower pitch, authoritative
- `playful` - Higher pitch, faster, warm
- `concerned` - Lower pitch, slower, warm + calm
- `angry` - Normal pitch, faster, louder, authoritative
- `curious` - Higher pitch, warm
- `thoughtful` - Lower pitch, much slower, calm
- `neutral` - No changes

### Contextual Adaptation

```python
from forge_ai.voice import adapt_voice_for_context

# Storytelling mode
story_voice = adapt_voice_for_context("storytelling", base_voice)
# Result: Slightly higher pitch, slower, "warm" effect

# Teaching mode
teach_voice = adapt_voice_for_context("teaching", base_voice)
# Result: Normal pitch, slower, louder, "calm" effect

# Formal mode
formal_voice = adapt_voice_for_context("formal", base_voice)
# Result: Lower pitch, "authoritative" effect
```

### Supported Contexts

- `storytelling` - Warm, slower
- `teaching` - Calm, slower, louder
- `casual` - Warm, slightly faster
- `formal` - Authoritative, lower pitch
- `presentation` - Authoritative, louder
- `friendly` - Warm, higher pitch
- `professional` - Cold, neutral

### Combined Adaptation

```python
from forge_ai.voice.dynamic_adapter import DynamicVoiceAdapter

adapter = DynamicVoiceAdapter()

# Adapt for both emotion AND context
voice = adapter.adapt_combined(
    emotion="happy",
    context="casual",
    base_profile=base_voice
)
```

### Personality-Based Adaptation

```python
from forge_ai.voice import adapt_voice_for_personality
from forge_ai.core.personality import load_personality

personality = load_personality("my_model")

# Voice adapts to personality's current mood and traits
adapted = adapt_voice_for_personality(personality, base_voice)
```

---

## Smart Wake Words

The system can suggest personalized wake phrases and train custom wake words.

### AI-Suggested Wake Phrases

```python
from forge_ai.voice import suggest_wake_phrases
from forge_ai.core.personality import load_personality

personality = load_personality("my_model")

# Get personalized suggestions
suggestions = suggest_wake_phrases("Enigma", personality)
# Result: ['hey enigma', 'ok enigma', 'enigma', 'hello enigma', ...]
```

### Wake Phrase Personalities

Suggestions adapt to AI personality:

```python
from forge_ai.voice.trigger_phrases import SmartWakeWords

smart = SmartWakeWords()

# Formal personality → formal phrases
suggestions = smart.suggest_wake_phrases("Enigma", formal_personality)
# Result: ['excuse me enigma', 'greetings enigma', 'attention enigma']

# Casual personality → casual phrases
suggestions = smart.suggest_wake_phrases("Enigma", casual_personality)
# Result: ['yo enigma', 'sup enigma', 'enigma buddy']
```

### Custom Wake Word Training

```python
from forge_ai.voice import train_custom_wake_phrase

# Train with audio samples (for future integration with Porcupine/Snowboy)
train_custom_wake_phrase(
    "my special phrase",
    audio_samples=["sample1.wav", "sample2.wav", "sample3.wav"]
)

# Or add without samples (uses text matching)
train_custom_wake_phrase("my special phrase")
```

### Improved Confidence Scoring

```python
smart = SmartWakeWords()

# Exact match
score = smart.improve_confidence("hey enigma", "hey enigma")
# Result: 1.0

# Substring match at start
score = smart.improve_confidence("hey enigma", "hey enigma listen")
# Result: 0.95

# Word overlap
score = smart.improve_confidence("ok enigma", "okay enigma please")
# Result: >0.6

# No match
score = smart.improve_confidence("hey enigma", "hello world")
# Result: <0.5
```

---

## Voice Cloning

Create custom voices from audio samples with analysis and Coqui XTTS integration hooks.

### Basic Voice Cloning

```python
from forge_ai.voice import create_voice_from_samples

# Provide audio samples
audio_files = ["sample1.wav", "sample2.wav", "sample3.wav"]

# Create voice profile from samples
voice = create_voice_from_samples(
    audio_files=audio_files,
    name="my_voice",
    model_name="my_model"  # Optional: associate with model
)

# Use the cloned voice
from forge_ai.voice import set_voice, speak
set_voice(voice)
speak("This is my cloned voice!")
```

### Audio Analysis

```python
from forge_ai.voice import analyze_audio

# Analyze a single audio file
features = analyze_audio("sample.wav")

print(f"Pitch: {features.average_pitch}")
print(f"Speed: {features.speaking_rate}")
print(f"Energy: {features.energy}")
print(f"Duration: {features.duration}s")
```

### Voice Comparison

```python
from forge_ai.voice import compare_voice_audio

# Compare two voice samples
similarity = compare_voice_audio("voice1.wav", "voice2.wav")
# Result: 0.0 to 1.0 (higher = more similar)

if similarity > 0.7:
    print("Voices are very similar!")
```

### Audio Quality Validation

```python
from forge_ai.voice.audio_analyzer import AudioAnalyzer

analyzer = AudioAnalyzer()

# Check if audio is suitable for cloning
is_valid, issues = analyzer.validate_audio_quality("sample.wav")

if not is_valid:
    print("Audio issues:")
    for issue in issues:
        print(f"  - {issue}")
```

### Coqui XTTS Integration

The system includes hooks for Coqui XTTS (when available):

```python
from forge_ai.voice.audio_analyzer import AudioAnalyzer

analyzer = AudioAnalyzer()

# Extract features for Coqui TTS
coqui_features = analyzer.extract_coqui_features(audio_files)

# Features include:
# - Audio file paths
# - Duration, sample rate
# - Pitch and energy analysis
# - Ready for Coqui TTS voice cloning
```

---

## Interactive Voice Customization

Tools for users to interactively create and tune voice profiles.

### Interactive CLI Tuning

```python
from forge_ai.voice import interactive_tuning

# Start interactive tuning session
voice = interactive_tuning(
    base_profile=None,  # Start from default
    preview_text="Hello, this is a test."
)

# Commands available:
# - pitch: Adjust pitch (0.5-1.5)
# - speed: Adjust speed (0.5-1.5)
# - volume: Adjust volume (0.3-1.0)
# - voice: Select voice type
# - effects: Select effects
# - preview: Hear current voice
# - done: Finish tuning
```

### Import/Export Profiles

```python
from forge_ai.voice import import_voice_profile, export_voice_profile, VoiceProfile

# Create a voice
voice = VoiceProfile(
    name="my_voice",
    pitch=1.2,
    speed=0.9,
    effects=["warm", "calm"]
)

# Export to file
export_voice_profile(voice, "my_voice.json")

# Import from file
loaded_voice = import_voice_profile("my_voice.json")

# Share with others!
```

### Voice Comparison

```python
from forge_ai.voice import compare_voices
from forge_ai.voice import PRESET_PROFILES

# Compare multiple voices side-by-side
voices = [
    PRESET_PROFILES["glados"],
    PRESET_PROFILES["jarvis"],
    PRESET_PROFILES["hal9000"]
]

compare_voices(voices, "Hello, this is a comparison test.")
# Prompts to hear each voice in sequence
```

### Preset Customization

```python
from forge_ai.voice.voice_customizer import VoiceCustomizer

customizer = VoiceCustomizer()

# Start from a preset
custom = customizer.create_from_preset(
    "glados",
    customizations={
        "pitch": 0.75,
        "speed": 0.85,
        "effects": ["robotic", "cold"]
    }
)
```

### Batch Variations

```python
customizer = VoiceCustomizer()

# Create multiple variations
variations = customizer.batch_create_variations(
    base_profile=voice,
    num_variations=5
)

# Listen to all variations and pick your favorite
compare_voices(variations, "Test text")
```

---

## Personality Integration

Voice preferences are stored with personality and evolve over time.

### Voice Preferences

```python
from forge_ai.core.personality import load_personality

personality = load_personality("my_model")

# Store voice preferences
personality.voice_preferences = {
    "pitch": 1.2,
    "speed": 0.9,
    "effects": ["warm", "calm"],
    "discovered_voice": "my_model_discovered"
}

# Track voice evolution
personality.voice_evolution_history.append({
    "timestamp": "2024-01-15T10:30:00",
    "change": "Increased pitch from 1.0 to 1.2",
    "reason": "User feedback: voice was too deep"
})

# Save personality (saves voice preferences too)
personality.save()
```

### Voice Evolution

```python
from forge_ai.voice.voice_generator import AIVoiceGenerator

generator = AIVoiceGenerator()

# Evolve voice as personality changes
current_voice = VoiceProfile.load("my_voice")
updated_voice = generator.evolve_voice(
    profile=current_voice,
    personality=personality,
    evolution_rate=0.1  # 10% change
)
```

### Sync Voice with Personality

```python
from forge_ai.voice import adapt_voice_for_personality

# Voice automatically adapts to personality state
voice = adapt_voice_for_personality(personality)

# Personality mood affects voice
personality.mood = "happy"
happy_voice = adapt_voice_for_personality(personality)

personality.mood = "thoughtful"
thoughtful_voice = adapt_voice_for_personality(personality)
```

---

## Advanced Usage

### Learning from Feedback

```python
from forge_ai.voice import adjust_voice_from_feedback

voice = VoiceProfile(pitch=1.2, speed=1.3, volume=0.7)

# User provides feedback
voice = adjust_voice_from_feedback("Too fast, slow down", voice)
# Voice speed is reduced

voice = adjust_voice_from_feedback("Can't hear you", voice)
# Volume is increased

voice = adjust_voice_from_feedback("Voice is too high", voice)
# Pitch is lowered
```

### Emotion + Context Mapping

```python
from forge_ai.voice import effect_for_emotion, effect_for_context

# Get appropriate effect for emotion
effect = effect_for_emotion("happy")
# Result: "energetic"

# Get appropriate effect for context
effect = effect_for_context("teaching")
# Result: "calm"
```

### Full Workflow Example

```python
from forge_ai.voice import *
from forge_ai.core.personality import load_personality

# 1. Load personality
personality = load_personality("my_model")

# 2. AI discovers its voice
voice = discover_voice(personality)

# 3. User provides feedback
voice = adjust_voice_from_feedback("A bit too fast", voice)

# 4. Save as preferred voice
voice.save()
personality.voice_preferences["discovered_voice"] = voice.name
personality.save()

# 5. Use the voice with dynamic adaptation
set_voice(voice)

# Voice adapts to context
casual_voice = adapt_voice_for_context("casual", voice)
set_voice(casual_voice)
speak("Hey, how's it going?")

formal_voice = adapt_voice_for_context("formal", voice)
set_voice(formal_voice)
speak("Good afternoon. How may I assist you?")

# Voice adapts to emotion
excited_voice = adapt_voice_for_emotion("excited", voice)
set_voice(excited_voice)
speak("This is amazing!")
```

---

## Requirements & Dependencies

### Required (Always Available)
- No external dependencies required for basic functionality
- All features work with graceful degradation

### Optional (Enhanced Features)
- **librosa** - Advanced audio analysis
- **parselmouth** - Pitch extraction
- **soundfile** - Audio file loading
- **Coqui TTS** - High-quality voice cloning
- **Porcupine/Snowboy** - Custom wake word training

### Installation

```bash
# Basic installation (no optional dependencies)
# All features work with fallbacks

# For advanced audio analysis
pip install librosa soundfile

# For pitch extraction
pip install praat-parselmouth

# For high-quality TTS (optional)
pip install TTS

# For custom wake words (optional)
pip install pvporcupine
```

---

## Troubleshooting

### Voice Not Changing

If the voice doesn't seem to change:
1. Ensure you're calling `set_voice()` after creating the profile
2. Check that TTS engine is initialized (`get_engine()`)
3. Try restarting the engine: `get_engine()._init_engine()`

### Audio Analysis Not Working

If audio analysis fails:
1. System falls back to basic estimation
2. Install optional dependencies for better analysis
3. Check audio file format (WAV preferred)

### Effects Not Applied

If effects don't seem to work:
1. Effects are text-based transformations
2. Some effects require audio processing (future feature)
3. Check that effects are in the profile: `profile.effects`

---

## API Reference

See inline documentation in each module:
- `forge_ai.voice.voice_identity` - AI voice discovery
- `forge_ai.voice.voice_effects` - Effects system
- `forge_ai.voice.dynamic_adapter` - Voice adaptation
- `forge_ai.voice.voice_customizer` - User tools
- `forge_ai.voice.audio_analyzer` - Audio analysis
- `forge_ai.voice.trigger_phrases` - Wake words

All modules include comprehensive docstrings and examples.
