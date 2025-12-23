# Avatar System Roadmap

## Current State
- Simple image display
- Load static PNG/JPG/GIF images
- Basic placeholder for avatar integration

## Vision: AI-Controlled Avatars in Virtual Environments

### Phase 1: Static Avatars (Current)
- [x] Image loading
- [x] Display in GUI
- [ ] Expression states (happy, sad, thinking, etc.)
- [ ] Lip-sync with TTS

### Phase 2: 2D Animated Avatars
- [ ] Live2D model support (.moc3 files)
- [ ] VTuber-style avatars
- [ ] Facial expression mapping
- [ ] Head/eye tracking simulation

### Phase 3: 3D Avatar Support
- [ ] VRM model loading (vrmodels.store compatible)
- [ ] VRoid Studio models
- [ ] Bone/skeleton animation
- [ ] Blend shape expressions

### Phase 4: AI Motor Control
- [ ] Reinforcement learning for movement
- [ ] Learn to control different avatar rigs
- [ ] Inverse kinematics understanding
- [ ] Gesture generation from speech

### Phase 5: Environment Integration
- [ ] Virtual room/space rendering
- [ ] Physics simulation
- [ ] Object interaction
- [ ] Spatial awareness

---

## Avatar Format Options

### 1. VRM Models (Recommended for 3D)
**Source**: https://vrmodels.store, VRoid Studio, Booth.pm
**Format**: .vrm files (based on glTF)
**Pros**:
- Standardized humanoid skeleton
- Built-in blend shapes for expressions
- Wide model availability
- VTuber ecosystem compatibility

**Python Libraries**:
- `pygltflib` - Load glTF/VRM files
- `trimesh` - 3D mesh processing
- `moderngl` or `pyglet` - Rendering

### 2. Live2D (2D Animation)
**Format**: .moc3, .model3.json
**Pros**:
- Lightweight
- Smooth anime-style animation
- Popular for VTubers

**Integration**:
- Live2D Cubism SDK (requires license for commercial)
- Can render to texture for PyQt display

### 3. Simple Image + States
**Format**: PNG/GIF with multiple expression images
**Pros**:
- Easiest to implement
- Low resource usage
- Works on Raspberry Pi

**Implementation**:
```
avatar/
  neutral.png
  happy.png
  thinking.png
  speaking.gif
```

---

## Environment Simulation Options

### Option 1: PyGame / Panda3D (Python Native)
**Pros**: Pure Python, easy integration
**Cons**: Limited graphics, more work for 3D
**Best for**: Simple 2D environments, retro style

### Option 2: Godot Engine
**Pros**: 
- Free and open source
- GDScript is Python-like
- Can communicate via HTTP/WebSocket
- Good 3D support

**Integration**:
```
Enigma Engine <--HTTP/WebSocket--> Godot Game
     AI                           Avatar + World
```

### Option 3: Unity + ML-Agents
**Pros**:
- Industry standard
- ML-Agents for reinforcement learning
- Vast asset store
- VRM support via UniVRM

**Best for**: Training AI to control avatars in complex environments

### Option 4: Blender (Animation/Rigging)
**Use for**:
- Creating custom avatars
- Animating sequences
- Rendering cutscenes
- NOT real-time simulation

### Option 5: Garry's Mod (GMod)
**Pros**:
- Huge mod community
- Physics sandbox
- Lua scripting

**Integration**:
```lua
-- GMod Lua addon
local socket = require("socket")
-- Connect to Enigma Engine API
-- Receive movement commands
-- Control playermodel/NPC
```

**Challenges**:
- Lua ↔ Python communication
- Real-time sync
- Limited to Source engine

### Option 6: VRChat / ChilloutVR
**Pros**: Social VR, avatar ecosystems
**Cons**: Closed platforms, TOS restrictions

---

## Recommended Architecture

```
┌─────────────────────────────────────────────────┐
│                 Enigma Engine                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │   AI Core   │  │   Vision    │  │  Voice  │ │
│  │  (Brain)    │  │ (Eyes)      │  │ (Ears/  │ │
│  │             │  │             │  │  Mouth) │ │
│  └──────┬──────┘  └──────┬──────┘  └────┬────┘ │
│         │                │              │       │
│         └────────┬───────┴──────────────┘       │
│                  │                              │
│         ┌───────▼────────┐                     │
│         │  Avatar Bridge │                     │
│         │  (Commands)    │                     │
│         └───────┬────────┘                     │
└─────────────────┼───────────────────────────────┘
                  │ WebSocket/HTTP
                  ▼
┌─────────────────────────────────────────────────┐
│           Environment Runtime                    │
│  (Godot / Unity / PyGame / Custom)              │
│                                                  │
│  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Avatar    │  │     Virtual World       │  │
│  │  (Body)     │  │  (Physics, Objects)     │  │
│  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## Implementation Priority

### Short Term (Now)
1. Fix screenshot capture for training
2. Image-based avatar with expressions
3. Expression changes based on AI state

### Medium Term (1-3 months)
1. VRM model loading and display
2. Basic animation playback
3. Godot prototype for environment

### Long Term (3-6 months)
1. AI learns to control avatar movements
2. Reinforcement learning environment
3. Multi-avatar support
4. Physics interaction

---

## Quick Start: Simple Avatar System

```python
# avatar_simple.py
class SimpleAvatar:
    def __init__(self, avatar_dir):
        self.expressions = {}
        for img in Path(avatar_dir).glob("*.png"):
            self.expressions[img.stem] = img
        self.current = "neutral"
    
    def set_expression(self, expr):
        if expr in self.expressions:
            self.current = expr
            return self.expressions[expr]
        return self.expressions.get("neutral")
    
    def get_expression_from_text(self, text):
        # Simple sentiment → expression mapping
        if any(w in text.lower() for w in ["happy", "great", "awesome"]):
            return "happy"
        elif any(w in text.lower() for w in ["sad", "sorry", "unfortunately"]):
            return "sad"
        elif any(w in text.lower() for w in ["think", "hmm", "let me"]):
            return "thinking"
        return "neutral"
```

---

## Resources

- **VRM**: https://vrm.dev/en/
- **VRoid Studio**: https://vroid.com/en/studio (free avatar creator)
- **vrmodels.store**: https://vrmodels.store/
- **Live2D**: https://www.live2d.com/
- **Godot**: https://godotengine.org/
- **Unity ML-Agents**: https://unity.com/products/machine-learning-agents
