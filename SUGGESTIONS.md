# Avatar System Suggestions

## Implementation Status

### Completed
- **Skeleton integration** - skeleton.py loads during model load in avatar_rendering.py
- **Exact bone name matching** - _find_bone() accepts raw bone names directly
- **FINGER_POSES dict** - Added to bone_control.py for gesture presets
- **Bone controller** - BoneController class with move_bone(), callbacks

### Still Needed
- Terminal control script (was prototyped, needs refinement)
- Physics simulation (jiggle, springs)
- Scene objects (props, furniture)
- Keyframe animation API
- Bone mapping cache

---

## Current Problem
- Avatar control uses hard-coded behaviors (emotion presets, gesture mappings)
- AI can only do predefined actions, can't create new behaviors
- No physics simulation
- No scene object support (furniture, props)
- No standalone terminal interface

## Proposed Changes

### 1. Discovery-Based Control
Instead of hard-coded commands like "emotion happy", the AI should:
- Query the model to discover what bones exist
- Learn the skeleton structure dynamically
- Create poses based on what it finds
- Work with ANY model, not just specific rigs

### 2. Remove Limits
- No rotation limits on bones
- AI has full raw access to bone transforms
- Can set any bone to any angle

### 3. Physics System
Add physics simulation for secondary motion:
- Jiggle physics (hair, accessories)
- Spring physics (bouncy movement)
- Pendulum physics (swinging/dangling parts)
- AI can add physics to any bone

### 4. Scene Objects
Allow AI to manipulate the 3D scene:
- Add props (chairs, tables, items)
- Move/rotate/scale objects
- Attach objects to bones (hats, items in hand)
- Remove objects

### 5. Keyframe Animation
AI should be able to create animations:
- Define keyframes with bone poses at specific times
- Play/stop animations
- Save/load animations to files
- Loop or one-shot playback

### 6. Terminal Control
Run avatar standalone without full GUI:
- JSON commands via stdin
- JSON responses via stdout
- Standalone script to launch just the avatar overlay

### 7. JSON Protocol
All commands should be JSON for programmatic control:
- Discovery: info, bones, bone_info
- Control: set, set_multi, add, reset
- Physics: physics_add, physics_remove, physics_impulse
- Objects: object_add, object_move, object_remove
- Animation: anim_create, anim_keyframe, anim_play
- State: pose_export, pose_save, pose_load

## Bone Name Matching Problem

Current `_find_bone()` uses hardcoded aliases for human bones:
- Works: mixamorig:Head, Head, head_jnt
- Fails: gladoshead_main_12, custom_bone_xyz

### Proposed Solutions

**1. Fuzzy Matching**
Use string similarity (Levenshtein distance) to find closest match:
- "head" → "gladoshead_main_12" (contains "head")
- "neck" → "gladosneck_11" (contains "neck")

**2. AI-Built Mapping**
When model loads, AI analyzes bone names and builds its own mapping:
- Discovery returns all bone names
- AI decides which bone is "head" based on name + hierarchy
- Stores mapping for that model

**3. Hierarchy-Based Detection**
Identify bones by position in hierarchy, not name:
- Root → first child is likely spine/hips
- End of chain = hand, foot, head
- Branches from spine = arms, legs

**4. User Override**
Allow manual mapping via config file:
```json
{
  "glados": {
    "head": "gladoshead_main_12",
    "neck": "gladosneck_11"
  }
}
```

**5. Universal Partial Match**
Current system already does this but could be smarter:
- If "head" appears ANYWHERE in bone name → match
- Priority: exact > contains > fuzzy

### Best Approach for AI Control
The AI should:
1. Call `{"cmd": "bones"}` to get ALL bone names
2. Analyze names itself to figure out what's what
3. Not rely on predefined mappings at all
4. Work with raw bone names from the model

This way ANY model works - human, robot, creature, whatever.

### Caching (Don't Re-analyze Every Time)
Store discovered mappings per model:

**Location:** `data/avatar/bone_mappings/<model_hash>.json`

**Flow:**
1. Model loads → hash the file path or model name
2. Check if mapping cache exists
3. If yes → load cached mapping instantly
4. If no → AI analyzes once, saves mapping

**Cache file example:**
```json
{
  "model": "glados",
  "path": "models/avatars/glados/scene.gltf",
  "created": "2026-02-16",
  "bones": {
    "head": "gladoshead_main_12",
    "neck": "gladosneck_11",
    "eye_left": "gladoseye_L_13",
    "eye_right": "gladoseye_R_14"
  }
}
```

**Benefits:**
- First load: AI learns (few seconds)
- Every load after: instant (read JSON)
- User can manually edit if AI got it wrong
- Shareable between users (same model = same cache)

## Files to Modify
- skeleton.py - Improve matching, add translation/scale, physics hooks
- avatar_rendering.py - Add scene object rendering
- bone_control.py - Simplify or remove limits
- New: scripts/avatar.py - Terminal interface

## Key Principle
**AI discovers and creates, never limited to presets.**
