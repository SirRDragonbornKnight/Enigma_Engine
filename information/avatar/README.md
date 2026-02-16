# Avatar System Documentation

> **Main Documentation:** See [enigma_engine/avatar/README.md](../../enigma_engine/avatar/README.md) for full docs.

## Quick Reference

### Enable & Move

```python
from enigma_engine.avatar import get_avatar

avatar = get_avatar()
avatar.enable()

# Movement
avatar.move_to(500, 300)           # Absolute position
avatar.move_relative(100, -50)     # Relative move
avatar.center_on_screen()          # Center on screen

# Gestures
avatar.control("wave")             # Wave hello
avatar.control("nod")              # Nod yes
avatar.control("shake_head")       # Shake head no
avatar.control("jump")             # Jump

# Speaking
avatar.speak("Hello!")             # With TTS
avatar.think(duration=2.0)         # Thinking animation

# Expressions
avatar.set_expression("happy")     # happy, sad, thinking, surprised, neutral

avatar.disable()
```

### Bone Control (3D Models)

```python
from enigma_engine.avatar.bone_control import get_bone_controller

bones = get_bone_controller()
bones.move_bone("head", pitch=15, yaw=-10)  # Look down-right
bones.move_bone("left_arm", pitch=45)       # Raise arm
```

### AI Self-Design

```python
avatar.link_personality(personality)
avatar.auto_design()  # AI designs its own appearance
avatar.describe_desired_appearance("friendly and creative")
```

## Key Files

| File | Purpose |
|------|---------|
| `controller.py` | Main state machine + priority system |
| `bone_control.py` | 3D skeleton joint control with anatomical limits |
| `adaptive_animator.py` | Capability-aware animations (wave, nod, blink) |
| `avatar_identity.py` | Personality to appearance mapping |
| `emotion_sync.py` | Auto-expression from AI text sentiment |
| `desktop_pet.py` | Floating transparent window overlay |

## Default State

**Avatar is OFF by default.** User must call `avatar.enable()` to show it.
