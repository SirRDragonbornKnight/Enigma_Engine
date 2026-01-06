# Enhanced Avatar System - Usage Guide

## Overview

The Enhanced Avatar System allows AI to design its own visual appearance based on personality traits, or lets users fully customize the avatar. The system includes:

- **AI Self-Design**: Avatar automatically designs itself from personality
- **Natural Language**: Describe desired appearance in plain English
- **User Customization**: Full manual control over all appearance aspects
- **Emotion Sync**: Avatar expressions automatically match AI mood
- **Built-in Sprites**: 9 SVG sprites included (no external assets needed)
- **Multiple Renderers**: Console, PyQt5 overlay, Web dashboard

## Quick Start

### 1. AI Self-Design from Personality

```python
from enigma.avatar import get_avatar
from enigma.core.personality import load_personality

# Get avatar and load AI personality
avatar = get_avatar()
personality = load_personality("my_model")

# Link personality and let AI design itself
avatar.link_personality(personality)
appearance = avatar.auto_design()

# AI explains its choices
print(avatar.explain_appearance())
# Output: "I chose this appearance because: rounded shape for playful nature, 
#          warm colors for empathy, bouncy animations for playful energy..."
```

### 2. Natural Language Description

```python
from enigma.avatar import get_avatar

avatar = get_avatar()

# AI interprets natural language
appearance = avatar.describe_desired_appearance(
    "I want to look friendly and approachable"
)
# Result: Rounded shape, warm colors, small size, friendly expression
```

### 3. User Customization

```python
from enigma.avatar import get_avatar

avatar = get_avatar()
customizer = avatar.get_customizer()

# Change style
customizer.set_style("anime")

# Apply color preset
customizer.apply_color_preset("sunset")

# Add accessories
customizer.add_accessory("hat")
customizer.add_accessory("glasses")

# Set size and animations
customizer.set_size("large")
customizer.set_animations(idle="bounce", movement="bounce")
```

### 4. Emotion Synchronization

```python
from enigma.avatar import get_avatar
from enigma.core.personality import load_personality

avatar = get_avatar()
personality = load_personality("my_model")

# Link personality - emotion sync starts automatically
avatar.link_personality(personality)

# Avatar expressions now automatically match AI mood
personality.mood = "happy"    # Avatar shows happy expression
personality.mood = "thinking" # Avatar shows thinking expression
```

## Personality-to-Appearance Mapping

The AI uses these mappings when designing its appearance:

| Personality Trait | High Value Results | Low Value Results |
|-------------------|-------------------|-------------------|
| **Playfulness** | Rounded shape, bright colors, bounce animation | Angular shape, muted colors, still animation |
| **Formality** | Realistic style, tie accessory, professional colors | Casual style, no accessories, varied colors |
| **Creativity** | Abstract style, unique elements, varied colors | Standard style, classic elements, uniform colors |
| **Confidence** | Large size, bold colors, upright posture | Small size, soft colors, relaxed posture |
| **Empathy** | Warm colors, friendly expression, cute eyes | Cool colors, neutral expression, sharp eyes |
| **Humor** | Bright colors, playful style, happy expression | Serious colors, minimal style, neutral expression |

## Built-in Sprites

9 SVG sprites are included (no downloads required):

1. **idle** - Neutral resting state
2. **happy** - Big smile, happy eyes
3. **sad** - Frown, sad eyes
4. **thinking** - Looking up, thought bubble
5. **surprised** - Wide eyes, open mouth
6. **confused** - Asymmetric eyes, question mark
7. **excited** - Sparkly eyes, big smile, sparkles
8. **speaking_1** - Semi-open mouth
9. **speaking_2** - Slightly closed mouth

All sprites support custom colors (primary, secondary, accent).

## Color Presets

9 built-in color presets:

- **default** - Balanced indigo/purple
- **warm** - Amber/red tones
- **cool** - Blue/cyan tones
- **nature** - Green tones
- **sunset** - Amber/pink/purple gradient
- **ocean** - Cyan/blue tones
- **fire** - Red/amber/yellow
- **dark** - Dark slate professional
- **pastel** - Soft purple/pink

## Customization Options

### Styles
- `default` - Balanced, versatile
- `anime` - Expressive, cute
- `realistic` - Professional, serious
- `robot` - Mechanical, tech
- `abstract` - Unique, artistic
- `minimal` - Simple, clean

### Shapes
- `rounded` - Soft, friendly
- `angular` - Sharp, professional
- `mixed` - Balanced

### Sizes
- `small` - Approachable, cute
- `medium` - Balanced
- `large` - Confident, bold

### Accessories
`hat`, `glasses`, `tie`, `bow`, `scarf`, `headphones`, `crown`, `halo`, `horns`, `creative_element`, `bold_outline`, `sparkles`

### Animations
- **Idle**: `breathe`, `float`, `pulse`, `still`, `bounce`
- **Movement**: `float`, `walk`, `bounce`, `teleport`

## Renderers

### Sprite Renderer (Default)
Works everywhere, console output:

```python
from enigma.avatar.renderers import SpriteRenderer

renderer = SpriteRenderer(controller)
renderer.show()  # Prints to console
```

### PyQt5 Renderer
Transparent overlay window:

```python
from enigma.avatar.renderers import QtAvatarRenderer

renderer = QtAvatarRenderer(controller)
renderer.show()  # Shows draggable window
```

### Web Renderer
For web dashboard:

```python
from enigma.avatar.renderers import WebAvatarRenderer

renderer = WebAvatarRenderer(controller, socketio)
renderer.show()  # Sends to web clients
```

## Export/Import

### Export Appearance

```python
customizer = avatar.get_customizer()
customizer.export_appearance("my_avatar.json")
```

### Import Appearance

```python
customizer = avatar.get_customizer()
customizer.import_appearance("my_avatar.json")
```

### Appearance File Format

```json
{
  "style": "anime",
  "primary_color": "#f59e0b",
  "secondary_color": "#ec4899",
  "accent_color": "#22d3ee",
  "shape": "rounded",
  "size": "large",
  "accessories": ["hat", "glasses"],
  "default_expression": "happy",
  "idle_animation": "bounce",
  "movement_style": "bounce"
}
```

## Advanced Usage

### Preview Before Applying

```python
customizer = avatar.get_customizer()

# Create test appearance
test_appearance = AvatarAppearance(
    style="robot",
    primary_color="#0000ff"
)

# Preview (temporary)
customizer.preview(test_appearance)

# Apply or cancel
customizer.apply_preview()  # Keep changes
# OR
customizer.cancel_preview()  # Revert
```

### Interactive Customization Wizard

```python
customizer = avatar.get_customizer()
appearance = customizer.interactive_customize()
# Guides user through step-by-step customization
```

### Generate All Sprites

```python
from enigma.avatar.renderers import SpriteRenderer

renderer = SpriteRenderer(controller)
renderer.export_sprites("./my_sprites/")
# Exports all 9 sprites as SVG files
```

## Examples

See `examples/avatar_system_demo.py` for comprehensive demonstrations of all features.

## API Reference

### AvatarController Methods

- `link_personality(personality)` - Link to AI personality
- `auto_design()` - Let AI design appearance
- `describe_desired_appearance(description)` - Natural language design
- `get_customizer()` - Get customization tools
- `set_appearance(appearance)` - Set appearance directly
- `explain_appearance()` - Get AI's explanation
- `set_expression(expression)` - Change expression
- `enable()` / `disable()` - Turn avatar on/off

### AvatarCustomizer Methods

- `set_style(style)` - Change style
- `set_colors(primary, secondary, accent)` - Set colors
- `apply_color_preset(preset)` - Use color preset
- `set_size(size)` - Change size
- `set_shape(shape)` - Change shape
- `add_accessory(accessory)` - Add accessory
- `remove_accessory(accessory)` - Remove accessory
- `set_animations(idle, movement)` - Set animations
- `export_appearance(path)` - Export to file
- `import_appearance(path)` - Import from file
- `preview(appearance)` - Preview temporarily
- `apply_preview()` / `cancel_preview()` - Confirm/cancel preview

### AIAvatarIdentity Methods

- `design_from_personality()` - Generate from personality
- `describe_desired_appearance(description)` - Parse natural language
- `choose_expression_for_mood(mood)` - Get expression for mood
- `evolve_appearance(feedback)` - Evolve over time
- `explain_appearance_choices()` - Explain choices
- `save(filepath)` / `load(filepath)` - Persistence

## Testing

Run tests with:

```bash
python -m pytest tests/test_avatar_enhancements.py -v
```

All 31 tests should pass.
