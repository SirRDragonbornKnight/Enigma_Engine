#!/usr/bin/env python3
"""
Avatar System Demo

Demonstrates the enhanced avatar system with AI self-design and user customization.
"""

import sys
from pathlib import Path

# Add enigma to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_tester.avatar import (
    get_avatar,
    AIAvatarIdentity,
    AvatarAppearance,
)
from ai_tester.core.personality import AIPersonality


def demo_ai_design():
    """Demo: AI designs its own appearance from personality."""
    print("\n" + "="*60)
    print("DEMO 1: AI Self-Design from Personality")
    print("="*60)
    
    # Create AI personality
    personality = AIPersonality("demo_model")
    personality.traits.playfulness = 0.9
    personality.traits.creativity = 0.8
    personality.traits.empathy = 0.7
    personality.traits.humor_level = 0.8
    
    print(f"\nPersonality Traits:")
    print(f"  Playfulness: {personality.traits.playfulness}")
    print(f"  Creativity: {personality.traits.creativity}")
    print(f"  Empathy: {personality.traits.empathy}")
    print(f"  Humor: {personality.traits.humor_level}")
    
    # Get avatar and link personality
    avatar = get_avatar()
    avatar.link_personality(personality)
    
    # Let AI design its appearance
    print(f"\nLetting AI design its own appearance...")
    appearance = avatar.auto_design()
    
    print(f"\nAI's Chosen Appearance:")
    print(f"  Style: {appearance.style}")
    print(f"  Shape: {appearance.shape}")
    print(f"  Size: {appearance.size}")
    print(f"  Primary Color: {appearance.primary_color}")
    print(f"  Secondary Color: {appearance.secondary_color}")
    print(f"  Accent Color: {appearance.accent_color}")
    print(f"  Accessories: {', '.join(appearance.accessories) if appearance.accessories else 'None'}")
    print(f"  Idle Animation: {appearance.idle_animation}")
    print(f"  Movement Style: {appearance.movement_style}")
    
    print(f"\nAI's Explanation:")
    print(f"  {avatar.explain_appearance()}")


def demo_natural_language():
    """Demo: AI describes desired appearance in natural language."""
    print("\n" + "="*60)
    print("DEMO 2: Natural Language Description")
    print("="*60)
    
    avatar = get_avatar()
    
    descriptions = [
        "I want to look friendly and approachable",
        "I want a professional formal look",
        "I want to be creative and artistic",
        "I want to look playful and fun",
    ]
    
    for desc in descriptions:
        print(f"\nDescription: '{desc}'")
        appearance = avatar.describe_desired_appearance(desc)
        
        print(f"  → Style: {appearance.style}")
        print(f"  → Shape: {appearance.shape}")
        print(f"  → Size: {appearance.size}")
        print(f"  → Primary Color: {appearance.primary_color}")
        print(f"  → Accessories: {', '.join(appearance.accessories) if appearance.accessories else 'None'}")


def demo_user_customization():
    """Demo: User customization tools."""
    print("\n" + "="*60)
    print("DEMO 3: User Customization")
    print("="*60)
    
    avatar = get_avatar()
    customizer = avatar.get_customizer()
    
    print("\nStarting with default appearance...")
    avatar.set_appearance(AvatarAppearance())
    
    print("\nApplying customizations:")
    
    # Change style
    print("  1. Setting style to 'anime'")
    customizer.set_style("anime")
    
    # Apply color preset
    print("  2. Applying 'sunset' color preset")
    customizer.apply_color_preset("sunset")
    
    # Add accessories
    print("  3. Adding accessories: hat, glasses")
    customizer.add_accessory("hat")
    customizer.add_accessory("glasses")
    
    # Set size
    print("  4. Setting size to 'large'")
    customizer.set_size("large")
    
    # Set animations
    print("  5. Setting animations: bounce idle, bounce movement")
    customizer.set_animations(idle="bounce", movement="bounce")
    
    print(f"\nFinal Appearance:")
    appearance = avatar._identity.appearance
    print(f"  Style: {appearance.style}")
    print(f"  Size: {appearance.size}")
    print(f"  Colors: {appearance.primary_color}, {appearance.secondary_color}, {appearance.accent_color}")
    print(f"  Accessories: {', '.join(appearance.accessories)}")
    print(f"  Animations: {appearance.idle_animation} (idle), {appearance.movement_style} (movement)")


def demo_color_presets():
    """Demo: Available color presets."""
    print("\n" + "="*60)
    print("DEMO 4: Color Presets")
    print("="*60)
    
    avatar = get_avatar()
    customizer = avatar.get_customizer()
    
    print("\nAvailable Color Presets:")
    for preset_name, colors in customizer.COLOR_PRESETS.items():
        print(f"\n  {preset_name.upper()}:")
        print(f"    Primary:   {colors['primary']}")
        print(f"    Secondary: {colors['secondary']}")
        print(f"    Accent:    {colors['accent']}")


def demo_emotion_sync():
    """Demo: Emotion synchronization."""
    print("\n" + "="*60)
    print("DEMO 5: Emotion Synchronization")
    print("="*60)
    
    from ai_tester.avatar import EmotionExpressionSync
    
    avatar = get_avatar()
    personality = AIPersonality("demo_model")
    
    # Create emotion sync
    sync = EmotionExpressionSync(avatar, personality)
    
    print("\nMood-to-Expression Mappings:")
    for mood, expression in sync.MOOD_TO_EXPRESSION.items():
        print(f"  {mood:12s} → {expression}")
    
    print("\nDetecting Emotions from Text:")
    test_texts = [
        "I'm so happy and excited to help you!",
        "Let me think about this problem...",
        "Unfortunately, that won't work.",
        "Wow! That's amazing!",
    ]
    
    for text in test_texts:
        emotion = sync.detect_emotion_from_text(text)
        expression = sync.MOOD_TO_EXPRESSION.get(emotion, "neutral")
        print(f"\n  Text: '{text}'")
        print(f"    → Emotion: {emotion}")
        print(f"    → Expression: {expression}")


def demo_sprite_generation():
    """Demo: Sprite generation."""
    print("\n" + "="*60)
    print("DEMO 6: Built-in Sprites")
    print("="*60)
    
    from ai_tester.avatar.renderers import SPRITE_TEMPLATES, generate_sprite
    
    print(f"\nAvailable Built-in Sprites: {len(SPRITE_TEMPLATES)}")
    for sprite_name in SPRITE_TEMPLATES.keys():
        print(f"  - {sprite_name}")
    
    print(f"\nGenerating custom-colored sprite...")
    svg = generate_sprite(
        "happy",
        primary_color="#ff0000",
        secondary_color="#00ff00",
        accent_color="#0000ff"
    )
    
    print(f"  Generated SVG length: {len(svg)} characters")
    print(f"  Contains custom colors: ", end="")
    print("✓" if "#ff0000" in svg else "✗")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("AI TESTER - ENHANCED AVATAR SYSTEM DEMO")
    print("="*60)
    
    try:
        demo_ai_design()
        demo_natural_language()
        demo_user_customization()
        demo_color_presets()
        demo_emotion_sync()
        demo_sprite_generation()
        
        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
