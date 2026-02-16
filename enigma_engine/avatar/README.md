# Enhanced Avatar System

## What's New

The avatar system has been completely enhanced with AI self-design capabilities and comprehensive user customization tools.

### Key Features

🤖 **AI Self-Design**
- AI automatically designs its own appearance based on personality traits
- Natural language interface: "I want to look friendly and approachable"
- Personality-driven appearance evolution

🎨 **User Customization**
- 6 styles, 3 shapes, 3 sizes
- 12+ accessories (hat, glasses, tie, etc.)
- 9 color presets + custom colors
- 10 animation options

😊 **Emotion Synchronization**
- Avatar expressions automatically match AI mood
- Text emotion detection
- Real-time mood monitoring

🎭 **Built-in Assets**
- 9 SVG sprites included (no downloads needed)
- Customizable colors for all sprites
- Works out of the box

🖥️ **Multiple Renderers**
- Console (universal)
- PyQt5 overlay window
- Web dashboard integration

## Quick Examples

### Let AI Design Itself

```python
from enigma_engine.avatar import get_avatar
from enigma_engine.core.personality import load_personality

avatar = get_avatar()
personality = load_personality("my_model")

# Link personality and let AI design
avatar.link_personality(personality)
appearance = avatar.auto_design()

# AI explains its choices
print(avatar.explain_appearance())
```

### Natural Language

```python
avatar.describe_desired_appearance("I want to be friendly and creative")
# → Rounded shape, warm colors, creative elements
```

### User Customization

```python
customizer = avatar.get_customizer()
customizer.set_style("anime")
customizer.apply_color_preset("sunset")
customizer.add_accessory("hat")
```

### Movement

```python
avatar = get_avatar()
avatar.enable()

# Move to specific screen coordinates
avatar.move_to(500, 300)

# Move relative to current position
avatar.move_relative(dx=100, dy=-50)  # Right 100px, up 50px

# Center on screen (auto-detects screen size)
avatar.center_on_screen()
```

### Gestures & Actions

```python
# Built-in gestures
avatar.control("wave")          # Wave hello
avatar.control("nod")           # Nod yes
avatar.control("shake_head")    # Shake head no
avatar.control("jump")          # Jump animation

# Lip-sync speaking
avatar.speak("Hello, how are you?")  # With TTS
avatar.animate_speak("Silent lip sync", duration=2.0)  # Without audio

# Thinking animation
avatar.think(duration=3.0)

# Set emotion/expression
avatar.set_expression("happy")  # happy, sad, thinking, surprised, neutral
avatar.control("emote", "excited")
```

### Bone Control (3D Rigged Models)

```python
from enigma_engine.avatar.bone_control import get_bone_controller

bones = get_bone_controller()

# Move individual bones (rotations in degrees)
bones.move_bone("head", pitch=15, yaw=-10)      # Look down-right
bones.move_bone("left_arm", pitch=45, roll=20)  # Raise arm
bones.move_bone("right_hand", yaw=30)           # Turn wrist

# System auto-clamps to safe limits (elbows don't bend backwards, etc.)
```

## Files & How They Work

```
enigma_engine/avatar/
```

### Core Control

| File | How It Works |
|------|-------------|
| `controller.py` | Main brain - manages avatar state machine (OFF/IDLE/SPEAKING/THINKING/MOVING), coordinates all subsystems via priority system. Higher priority systems (bone control=100) override lower ones (idle animation=30). Writes commands to `data/avatar/ai_command.json` for GUI to read. |
| `bone_control.py` | Directly manipulates 3D skeleton joints. Each bone has rotation limits based on human anatomy (elbow only bends one way, head turns max 80 degrees). Clamps unsafe movements, limits speed to prevent jerkiness. |
| `adaptive_animator.py` | Detects model capabilities (has arms? has head bone? can blink?) then plays appropriate animations. Falls back gracefully (no arms = wiggle whole body for wave). |
| `animation_system.py` | Queue-based animation player. Animations are dicts like `{type: "wave", duration: 1.0}`. Background thread processes queue at 20 FPS. |

### AI Integration

| File | How It Works |
|------|-------------|
| `avatar_identity.py` | Maps AI personality traits to appearance. High playfulness = rounded shape + bright colors. High formality = tie accessory + muted tones. Uses trait vectors (0-1) to interpolate between extremes. |
| `emotion_sync.py` | Monitors AI text output for emotion keywords/sentiment. Automatically triggers expression changes. "I'm excited!" = switch to excited sprite. |
| `lip_sync.py` | Analyzes speech audio for phonemes (mouth shapes). Maps phonemes to mouth sprites. Syncs animation timing with TTS audio. |

### Rendering

| File | How It Works |
|------|-------------|
| `desktop_pet.py` | Creates frameless, transparent PyQt5 window that floats on desktop. Handles click-through, always-on-top, mouse dragging. |
| `unified_avatar.py` | Abstracts 2D sprites vs 3D models. Same API regardless of avatar type - picks correct renderer internally. |
| `renderers/qt_renderer.py` | PyQt5 OpenGL renderer for 3D models. Uses QOpenGLWidget for hardware acceleration. |
| `renderers/sprite_renderer.py` | Simple 2D sprite display. Cycles through PNG/SVG frames for animation. Falls back to console ASCII if no GUI. |

### Customization

| File | How It Works |
|------|-------------|
| `customizer.py` | Stateful appearance builder. Stacks modifications: base style + colors + accessories + size. Export/import as JSON for sharing. |
| `avatar_manager.py` | Multi-avatar support. Load multiple avatars, switch between them, sync states across instances. |

## Documentation

- **Usage Guide**: `docs/AVATAR_SYSTEM_GUIDE.md` - Comprehensive guide
- **Demo Script**: `examples/avatar_system_demo.py` - 6 interactive demos
- **Tests**: `tests/test_avatar_enhancements.py` - 31 tests (all passing)

## Testing

```bash
# Run tests
python -m pytest tests/test_avatar_enhancements.py -v

# Run demo
python examples/avatar_system_demo.py
```

## Personality-to-Appearance Mapping

| Trait | High Value | Low Value |
|-------|-----------|-----------|
| **Playfulness** | Rounded, bright, bounce | Angular, muted, still |
| **Formality** | Tie, realistic, professional | Casual, varied colors |
| **Creativity** | Abstract, unique elements | Standard, classic |
| **Confidence** | Large, bold colors | Small, soft colors |
| **Empathy** | Warm, friendly, cute eyes | Cool, neutral, sharp eyes |

## Built-in Sprites

9 expressions included (all customizable colors):
- `idle` - Neutral resting
- `happy` - Big smile
- `sad` - Frown
- `thinking` - Thought bubble
- `surprised` - Wide eyes
- `confused` - Question mark
- `excited` - Sparkly
- `speaking_1` - Semi-open mouth
- `speaking_2` - Closed mouth

## Color Presets

- `default` - Balanced indigo/purple
- `warm` - Amber/red tones
- `cool` - Blue/cyan tones  
- `nature` - Green tones
- `sunset` - Gradient colors
- `ocean` - Blue tones
- `fire` - Red/yellow tones
- `dark` - Professional slate
- `pastel` - Soft colors

## API Highlights

```python
# Core controller
avatar = get_avatar()
avatar.link_personality(personality)
avatar.auto_design()
avatar.describe_desired_appearance("friendly")
avatar.get_customizer()
avatar.set_appearance(appearance)
avatar.explain_appearance()

# Customizer
customizer.set_style("anime")
customizer.set_colors("#ff0000", "#00ff00", "#0000ff")
customizer.apply_color_preset("sunset")
customizer.set_size("large")
customizer.add_accessory("hat")
customizer.export_appearance("avatar.json")
customizer.import_appearance("avatar.json")

# AI Identity
identity.design_from_personality()
identity.describe_desired_appearance("creative and bold")
identity.choose_expression_for_mood("happy")
identity.explain_appearance_choices()
```

## Statistics

- **Lines of Code**: ~3,500
- **Tests**: 31 (all passing)
- **Built-in Sprites**: 9
- **Color Presets**: 9
- **Customization Options**: 40+
- **Renderers**: 3

## Design Principles

✅ Works out of box (no external downloads)
✅ Graceful degradation (console fallback)
✅ AI autonomy (self-design)
✅ User control (full customization)
✅ Personality sync (reflects traits)
✅ Mood sync (auto expressions)
✅ Cross-platform (multiple renderers)

## Next Steps

1. Enable the avatar: `avatar.enable()`
2. Link to your AI's personality
3. Let it design itself or customize manually
4. Watch expressions sync with mood automatically

For more details, see `docs/AVATAR_SYSTEM_GUIDE.md`
