"""
Example usage of the enhanced voice system.

Demonstrates:
- AI voice self-discovery
- Natural language voice descriptions
- Voice adaptation for emotions/contexts
- Smart wake word suggestions
- Interactive voice customization
"""

from forge_ai.voice import (
    VoiceProfile,
    discover_voice,
    describe_voice,
    adjust_voice_from_feedback,
    adapt_voice_for_emotion,
    adapt_voice_for_context,
    apply_effect,
    suggest_wake_phrases,
)


def example_ai_voice_discovery():
    """Example: AI discovers its own voice."""
    print("\n=== AI Voice Self-Discovery ===")
    
    # Note: This example requires an existing personality model
    # Create one first with: personality = AIPersonality("example_model"); personality.save()
    print("\n1. Discovering voice with personality...")
    try:
        from forge_ai.core.personality import AIPersonality
        
        # Create a temporary personality for demonstration
        personality = AIPersonality("example_model")
        personality.traits.playfulness = 0.8
        personality.traits.confidence = 0.7
        personality.traits.formality = 0.3
        
        voice = discover_voice(personality, num_experiments=3)
        print(f"   Discovered voice: {voice.name}")
        print(f"   - Pitch: {voice.pitch:.2f}")
        print(f"   - Speed: {voice.speed:.2f}")
        print(f"   - Effects: {', '.join(voice.effects) if voice.effects else 'none'}")
    except Exception as e:
        print(f"   (Personality module not available: {e})")


def example_natural_language_voice():
    """Example: Create voice from natural language."""
    print("\n=== Natural Language Voice Description ===")
    
    descriptions = [
        "I want a warm, calm voice that speaks slowly",
        "I want an energetic, fast, high-pitched voice",
        "I want a deep, authoritative, confident voice",
        "Give me a robotic, mechanical voice"
    ]
    
    for desc in descriptions:
        voice = describe_voice(desc)
        print(f"\n   Description: '{desc}'")
        print(f"   Result: pitch={voice.pitch:.2f}, speed={voice.speed:.2f}")
        print(f"   Effects: {', '.join(voice.effects) if voice.effects else 'none'}")


def example_voice_adaptation():
    """Example: Adapt voice for emotions and contexts."""
    print("\n=== Dynamic Voice Adaptation ===")
    
    base_voice = VoiceProfile(name="base", pitch=1.0, speed=1.0)
    
    # Emotional adaptation
    print("\n1. Emotional Adaptation:")
    emotions = ["happy", "sad", "excited", "serious"]
    
    for emotion in emotions:
        adapted = adapt_voice_for_emotion(emotion, base_voice)
        print(f"   {emotion.capitalize()}: pitch={adapted.pitch:.2f}, speed={adapted.speed:.2f}")
    
    # Contextual adaptation
    print("\n2. Contextual Adaptation:")
    contexts = ["storytelling", "teaching", "casual", "formal"]
    
    for context in contexts:
        adapted = adapt_voice_for_context(context, base_voice)
        print(f"   {context.capitalize()}: {', '.join(adapted.effects)}")


def example_voice_effects():
    """Example: Apply voice effects."""
    print("\n=== Voice Effects ===")
    
    text = "Hello, how are you today?"
    effects = ["robotic", "whisper", "calm", "energetic", "authoritative"]
    
    print(f"\n   Original: '{text}'")
    
    for effect in effects:
        modified = apply_effect(text, effect)
        print(f"   {effect.capitalize()}: '{modified}'")


def example_feedback_learning():
    """Example: Learn from user feedback."""
    print("\n=== Learning from Feedback ===")
    
    voice = VoiceProfile(name="initial", pitch=1.2, speed=1.3, volume=0.7)
    
    print(f"\n   Initial voice: pitch={voice.pitch:.2f}, speed={voice.speed:.2f}, volume={voice.volume:.2f}")
    
    feedbacks = [
        "Too fast, slow down",
        "Too quiet, speak louder",
        "Voice is too high pitched"
    ]
    
    for feedback in feedbacks:
        voice = adjust_voice_from_feedback(feedback, voice)
        print(f"\n   Feedback: '{feedback}'")
        print(f"   Adjusted: pitch={voice.pitch:.2f}, speed={voice.speed:.2f}, volume={voice.volume:.2f}")


def example_smart_wake_words():
    """Example: Smart wake word suggestions."""
    print("\n=== Smart Wake Word Suggestions ===")
    
    ai_names = ["ForgeAI", "Atlas", "Nova"]
    
    for name in ai_names:
        suggestions = suggest_wake_phrases(name)
        print(f"\n   AI Name: {name}")
        print(f"   Suggestions: {', '.join(suggestions[:5])}")


def example_voice_customization():
    """Example: Voice customization workflow."""
    print("\n=== Voice Customization Workflow ===")
    
    # Start with a preset
    from forge_ai.voice import PRESET_PROFILES
    
    print("\n   Available presets:")
    for name, profile in list(PRESET_PROFILES.items())[:5]:
        print(f"   - {name}: {profile.description}")
    
    # Customize a preset
    from forge_ai.voice import VoiceCustomizer
    customizer = VoiceCustomizer()
    
    custom = customizer.create_from_preset(
        "glados",
        customizations={
            "pitch": 0.75,
            "speed": 0.85,
            "effects": ["robotic", "cold"]
        }
    )
    
    print(f"\n   Customized GLaDOS:")
    print(f"   - Pitch: {custom.pitch:.2f}")
    print(f"   - Speed: {custom.speed:.2f}")
    print(f"   - Effects: {', '.join(custom.effects)}")


def main():
    """Run all examples."""
    print("=" * 70)
    print("Enhanced Voice System Examples")
    print("=" * 70)
    
    try:
        example_ai_voice_discovery()
    except Exception as e:
        print(f"Error in voice discovery: {e}")
    
    try:
        example_natural_language_voice()
    except Exception as e:
        print(f"Error in natural language: {e}")
    
    try:
        example_voice_adaptation()
    except Exception as e:
        print(f"Error in adaptation: {e}")
    
    try:
        example_voice_effects()
    except Exception as e:
        print(f"Error in effects: {e}")
    
    try:
        example_feedback_learning()
    except Exception as e:
        print(f"Error in feedback learning: {e}")
    
    try:
        example_smart_wake_words()
    except Exception as e:
        print(f"Error in wake words: {e}")
    
    try:
        example_voice_customization()
    except Exception as e:
        print(f"Error in customization: {e}")
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
