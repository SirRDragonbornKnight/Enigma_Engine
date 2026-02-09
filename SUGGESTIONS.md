# Enigma AI Engine - TODO Checklist

**Last Updated:** February 9, 2026

---

## Quick Wins (< 4 hours each)

- [x] **Add more avatar generation styles** (~1-2 hours) DONE
  - Added to `self_tools.py`: anime, pixel, chibi, furry, mecha
  - File: `enigma_engine/tools/self_tools.py`

- [x] **Add `adjust_idle_animation` tool** (~2 hours) DONE
  - AI can control breathing rate, sway, blink rate
  - Added `AdjustIdleAnimationTool` to `avatar_tools.py`
  - Registered in `tool_registry.py`

- [x] **GUI persona switcher dropdown** (~2-3 hours) DONE
  - Added `QComboBox` persona selector to Chat tab header
  - Populates from `utils/personas.py` PersonaManager
  - Syncs with `core/persona.py` PersonaManager
  - Users can switch AI personality mid-conversation
  - File: `chat_tab.py`

---

## Medium Tasks (4-8 hours each)

- [x] **Pixel-perfect click-through for spawned objects** (~4-6 hours) DONE
  - Added `nativeEvent` + `_is_pixel_opaque()` to `ObjectWindow` in `spawnable_objects.py`
  - Copied pattern from `avatar_display.py`
  - Caches rendered pixmap for hit testing

- [x] **AI object spawn toggles** (~4-6 hours) DONE
  - Added `SpawnSettings` dataclass with toggles: `allow_spawned_objects`, `allow_held_items`, `allow_screen_effects`, `allow_notes`, `allow_bubbles`, `gaming_mode`
  - AI gets blocked feedback via `SpawnedObject.blocked` and `SpawnedObject.blocked_reason`
  - File: `spawnable_objects.py`

- [x] **Touch interaction reactions (headpats)** (~6-8 hours) DONE
  - Added `touched` signal to `AvatarOverlayWindow` and `BoneHitRegion`
  - Touch types: 'tap', 'double_tap', 'hold', 'pet' (repeated taps for headpats!)
  - Touch events written to `data/avatar/touch_event.json` for AI to read
  - `write_touch_event_for_ai()` and `get_recent_touch_event()` in `persistence.py`
  - Files: `avatar_display.py`, `persistence.py`

- [x] **Avatar hot-swap (file watcher)** (~4-6 hours MVP) DONE
  - Added `QFileSystemWatcher` to watch avatar file for changes
  - Auto-reload with debouncing (300ms wait for file to settle)
  - Crossfade transition during avatar changes (~320ms)
  - `set_hotswap_enabled()` to toggle feature
  - File: `avatar_display.py`

---

## Large Tasks (1-3 days each)

- [x] **Fullscreen app mode enhancements** (~1-2 days) DONE
  - [x] FullscreenController class with settings/methods
  - [x] Per-monitor control logic
  - [x] Object category toggles logic (avatar, spawned_objects, effects, particles)
  - [x] Smooth fade transition code
  - [x] Global hotkey registration code (Windows)
  - [x] Fullscreen detection (Windows)
  - [x] AI tool: `fullscreen_mode_control`
  - [x] File: `enigma_engine/core/fullscreen_mode.py`
  - [x] Hooked `avatar_display.py` - registers in showEvent
  - [x] Hooked `spawnable_objects.py` - registers/unregisters on spawn/remove
  - [x] Hooked `screen_effects.py` - registers overlays on creation
  - [x] Integrated with `gaming_mode.py` - profile.avatar_enabled controls visibility
  - [x] GUI settings panel for configuring visibility preferences
  - [x] Settings persistence (auto-load on startup)
  - File: `enigma_engine/gui/tabs/settings_tab.py` (Fullscreen Visibility Control section)

- [x] **Real-time avatar editing (full version)** (~2-3 days) DONE
  - [x] Part-by-part editing (swap hair, eyes, clothes while visible)
  - [x] Morphing transitions between avatars
  - [x] Layer-based composition with z-ordering
  - [x] Per-part transforms (offset, scale, rotation, opacity)
  - [x] Tint/color support per part
  - [x] Variant system for swappable parts
  - [x] Preset save/load
  - File: `enigma_engine/avatar/part_editor.py`

- [x] **Mesh manipulation** (~1 day min, 1 week full) DONE
  - [x] Vertex-level manipulation (move, stretch)
  - [x] Morph targets / blend shapes
  - [x] Region-based scaling (HEAD, TORSO, LEGS, etc.)
  - [x] Soft selection with falloff
  - [x] Animated morph transitions
  - [x] OBJ import/export
  - [x] trimesh integration
  - File: `enigma_engine/avatar/mesh_manipulation.py`

---

## Major Features (40+ hours)

- [ ] **Portal gun system** (~30+ hours) - *AI can implement using existing tools*
  - Building blocks available:
    - Particle effects: `screen_effects.py` (magic, sparkle presets)
    - Avatar teleport: `avatar_display.py` (move_to, fade transitions)
    - Screen overlays: transparent click-through windows ready
  - AI can combine these to create portal effect when requested
  - Full render-through-portal would need OpenGL shaders (future)

- [ ] **Trainer fine-tuning workflow for pre-trained models** (~4-8 hours)
  - Current state:
    - ✅ Resume training from checkpoints works (`resume_from_checkpoint()`)
    - ✅ LoRA fine-tuning works (`lora_utils.py`)
    - ✅ HuggingFace conversion works (`convert_huggingface_to_forge()`)
  - Missing:
    - [ ] `register_huggingface_model()` in `ModelRegistry` - auto-convert HF model and register for training
    - [ ] Unified `fine_tune_pretrained()` function - handles full workflow (download → convert → register → train)
    - [ ] GUI tab for importing external models (HuggingFace, GGUF) into registry
  - Files: `enigma_engine/core/model_registry.py`, `enigma_engine/core/huggingface_loader.py`

- [x] **Fullscreen effect overlay system** DONE
  - Single transparent fullscreen overlay for effects
  - Click-through by default (WA_TransparentForMouseEvents)
  - 12 effect presets: sparkle, fire, snow, rain, explosion, confetti, hearts, magic, smoke, bubble, lightning, ripple
  - Custom textures support via `assets/effects/textures/`
  - Custom presets support via `assets/effects/presets/`
  - `set_gaming_mode()` method available (manual call, not auto-connected to gaming_mode.py)
  - Multi-monitor support (spawn_effect_on_screen, spawn_effect_all_screens)
  - AI tools: `spawn_screen_effect`, `stop_screen_effect`, `list_effect_assets`
  - File: `enigma_engine/avatar/screen_effects.py`

---

everything needs to be real time and fully integrated and fix the workflow

---

## Already Exists (Reference)

These features are DONE and don't need implementation:

| Feature | Location | Notes |
|---------|----------|-------|
| Avatar movement/scaling | `control_avatar` tool | move_to, walk_to, resize, look_at, gestures |
| Bone control | `bone_control.py` | Direct skeleton manipulation |
| Breathing/idle animation | `procedural_animation.py` | BreathingController, IdleAnimator, blinking |
| Persona system | `utils/personas.py` | Create/switch AI personalities |
| Content rating (NSFW) | `content_rating.py` | SFW/MATURE/NSFW modes, text filtering |
| Gaming mode (basic) | `gaming_mode.py` | Game detection, resource throttling, profiles |
| Generate avatar | `self_tools.py` | 10 styles: realistic, cartoon, robot, creature, abstract, anime, pixel, chibi, furry, mecha |
| Spawnable objects | `spawnable_objects.py` | Speech bubbles, notes, held items, effects |
| Screen effects | `screen_effects.py` | 12 presets + custom textures/presets via `assets/effects/` |
| BoneHitManager | `avatar_display.py` | 6 body region click detection |
| User-teachable behaviors | `behavior_preferences.py` | "Whenever you X, do Y first" |
| Physics simulation | `physics_simulation.py` | Hair/cloth springs, gravity, bounce |

---

## Archived: Completed Fixes

<details>
<summary>151 bug fixes from code review (click to expand)</summary>

See original SUGGESTIONS.md for full list of:
- 19 memory leak fixes (unbounded history lists)
- 49 subprocess timeout fixes
- 9 HTTP timeout fixes
- File leak fixes
- Division by zero fixes
- Duplicate removal
- And more...

</details>
