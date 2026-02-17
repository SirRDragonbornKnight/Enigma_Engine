
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
