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

- [ ] **Gaming mode enhancements** (~1-2 days)
  - Per-monitor control: "Show avatar on monitor 2 only"
  - Object category toggles: avatar, spawned_objects, portal_effects, particles
  - Smooth fade transitions instead of instant hide
  - Hotkey override for instant toggle
  - Files: `gaming_mode.py`, `avatar_display.py`

- [ ] **Real-time avatar editing (full version)** (~2-3 days)
  - Part-by-part editing (swap hair, eyes, clothes while visible)
  - Morphing transitions between avatars
  - AI describes changes, system generates just that part
  - Files: new `avatar/part_editor.py`, `tools/avatar_tools.py`

- [ ] **Mesh manipulation** (~1 day min, 1 week full)
  - Vertex-level manipulation (stretch, squash, pull)
  - Morph targets / blend shapes
  - Simple version: scale body regions
  - Full version: proper blend shape system
  - `trimesh` library available

---

## Major Features (40+ hours)

- [ ] **Portal gun visual effects system** (~40+ hours)
  - Portal projectile animation (particle system)
  - Portal surface rendering (render-to-texture, shaders)
  - See-through effect (render destination, project onto portal)
  - Avatar teleport animation (fade into portal, appear at exit)
  - Two modes: 
    - Simple: visual effect only, instant teleport
    - Full: actual render-through-portal (OpenGL/shader work)
  - Sound effects

- [ ] **Fullscreen effect overlay system**
  - Single transparent fullscreen overlay for effects
  - Click-through by default, solid for dramatic moments
  - Can draw: portals, particles, explosions, spell effects
  - Auto-hides in gaming mode
  - Works across multi-monitor setups

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
| Generate avatar | `self_tools.py` | 5 styles: realistic, cartoon, robot, creature, abstract |
| Spawnable objects | `spawnable_objects.py` | Speech bubbles, notes, held items, effects |
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
