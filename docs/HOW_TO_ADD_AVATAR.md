# How to Add a New Avatar to Enigma AI Engine

## Quick Start

There are **3 ways** to add avatars to Enigma AI Engine:

### Method 1: Add Existing Files (Easiest)

1. **Locate the avatars folder:**
   ```
   /home/pi/Enigma AI Engine/data/avatar/
   ```

2. **Place your avatar files:**
   - **2D Images**: `data/avatar/images/`
     - Supported: PNG, JPG, GIF, WebP
     - Example: `my_avatar.png`
   
   - **3D Models**: `data/avatar/models/`
     - Supported: GLB, GLTF, OBJ, FBX
     - Example: `my_character.glb`
     - **For bone control**: Use rigged models (GLB/GLTF/FBX with skeleton)

3. **Refresh in GUI:**
   - Open Enigma AI Engine GUI
   - Go to **Avatar** tab
   - Click dropdown menu
   - Your avatar should appear!

---

### Method 2: Use Built-in Avatar Creator (GUI)

1. **Open Enigma AI Engine GUI:**
   ```bash
   python run.py --gui
   ```

2. **Go to Avatar Tab**

3. **Create New Avatar:**
   - Click "Create Avatar" or "+" button
   - Choose avatar type:
     - **SVG Sprite** - Simple 2D character
     - **From Image** - Upload your own image
     - **3D Model** - Import 3D file
     - **AI Generated** - Let AI create one

4. **Customize:**
   - Set name, colors, size
   - Configure expressions
   - Set idle animations

5. **Save:**
   - Avatar is automatically added to library
   - Appears in dropdown menu

---

### Method 3: Install Avatar Bundle (.fav file)

Enigma AI Engine uses `.fav` (Enigma AI Engine Avatar) bundle format for sharing avatars.

1. **Get a .fav bundle:**
   - Download from community
   - Create your own (see below)

2. **Install via GUI:**
   - Open Enigma AI Engine GUI → Avatar tab
   - Click "Install Avatar Bundle"
   - Select `.fav` file
   - Avatar installed to `data/avatar/installed/`

3. **Install via code:**
   ```python
   from enigma_engine.avatar.avatar_bundle import install_avatar
   
   avatar_dir = install_avatar("path/to/avatar.fav")
   print(f"Installed to: {avatar_dir}")
   ```

---

## Avatar Folder Structure

```
data/avatar/
├── images/           # 2D avatar images (PNG, JPG, GIF)
│   ├── robot.png
│   ├── wizard.gif
│   └── my_avatar.jpg
│
├── models/           # 3D models (GLB, GLTF, OBJ, FBX)
│   ├── character.glb
│   ├── robot.obj
│   └── rigged_model.fbx  # For bone control
│
├── installed/        # Installed avatar bundles
│   ├── cool_avatar_1/
│   └── another_avatar/
│
└── samples/          # Sample/demo avatars
    └── ...
```

---

## Creating Rigged 3D Avatars (For Bone Control)

To use the AI bone control system, your 3D model needs a skeleton:

### Required Bones (Standard Rig)

```
- head
- neck
- spine, spine1, spine2
- chest
- hips / pelvis
- left_shoulder, right_shoulder
- left_upper_arm, right_upper_arm
- left_forearm, right_forearm
- left_hand, right_hand
- left_upper_leg, right_upper_leg
- left_lower_leg, right_lower_leg
- left_foot, right_foot
```

### Creating in Blender

1. **Model your character**

2. **Add Armature (Rig):**
   - Add → Armature
   - Edit mode → Create bones
   - Name bones following standard names above

3. **Parent to Armature:**
   - Select mesh → Select armature
   - Ctrl+P → Automatic Weights

4. **Export:**
   - File → Export → glTF 2.0 (.glb)
   - Check: Include → Skinning
   - Export GLB

5. **Place in Enigma AI Engine:**
   ```bash
   cp my_rigged_model.glb /home/pi/Enigma AI Engine/data/avatar/models/
   ```

6. **Load in GUI:**
   - Avatar tab → Select your model
   - Console shows: "Bone controller initialized with X bones"
   - AI can now control bones!

---

## Creating Avatar Bundles (.fav)

Share your avatars with others!

### Via Code

```python
from enigma_engine.avatar.avatar_bundle import AvatarBundle
from pathlib import Path

# Create bundle
bundle = AvatarBundle.create(
    name="My Cool Avatar",
    avatar_type="3d_model",  # or "image", "sprite"
    files={
        "model": Path("my_avatar.glb"),
        "texture": Path("texture.png"),
        "icon": Path("icon.png")
    },
    author="Your Name",
    description="An awesome avatar",
    tags=["robot", "scifi", "rigged"]
)

# Save bundle
bundle.save("my_cool_avatar.fav")

# Others can install it:
# install_avatar("my_cool_avatar.fav")
```

### Bundle Metadata

```json
{
  "name": "My Avatar",
  "type": "3d_model",
  "author": "Creator Name",
  "version": "1.0.0",
  "description": "Description here",
  "tags": ["tag1", "tag2"],
  "files": {
    "model": "avatar.glb",
    "icon": "icon.png"
  },
  "capabilities": {
    "bone_control": true,
    "expressions": ["happy", "sad", "neutral"],
    "animations": ["idle", "wave"]
  }
}
```

---

## Testing Your Avatar

### 1. Load in GUI
```bash
python run.py --gui
# Avatar tab → Select your avatar
```

### 2. Test Bone Control (if rigged)
In chat, try:
```
"Wave hello"
"Nod your head"
"Look left"
"Point at something"
```

### 3. Test Expressions
```
"Show me a happy face"
"Look sad"
```

### 4. Check Console
Look for:
```
[Avatar] Loading model: my_avatar.glb
[BoneController] Detected bones: head, neck, spine, ...
[Avatar] Bone controller initialized with 23 bones
```

---

## Troubleshooting

### Avatar Doesn't Appear in Dropdown
- Check file is in correct folder (`data/avatar/images/` or `data/avatar/models/`)
- Refresh GUI (restart or click refresh button)
- Check file permissions (readable)

### 3D Model Loads But No Bone Control
- Model might not have skeleton/armature
- Check console for "Bone controller initialized" message
- Verify bone names match standard names
- Use GLB/GLTF format (best support)

### Avatar Too Big/Small
- GUI: Use size controls in Avatar tab
- Code: Adjust scale in avatar settings
- 3D models: Scale in Blender before export

### Bones Not Moving Correctly
- Check bone naming (must match standard names)
- Verify bone hierarchy (parent-child relationships)
- Check armature is properly bound to mesh

---

## Example: Quick Avatar Setup

```bash
# 1. Create avatar folder (if needed)
mkdir -p ~/Enigma AI Engine/data/avatar/models
mkdir -p ~/Enigma AI Engine/data/avatar/images

# 2. Add a 2D image avatar
cp my_cool_image.png ~/Enigma AI Engine/data/avatar/images/

# 3. Add a 3D model
cp rigged_character.glb ~/Enigma AI Engine/data/avatar/models/

# 4. Start GUI
cd ~/Enigma AI Engine
python run.py --gui

# 5. Select avatar in Avatar tab dropdown
```

---

## Avatar Configuration File

Each avatar can have a `config.json`:

```json
{
  "name": "My Avatar",
  "type": "3d_model",
  "scale": 1.0,
  "position": [0, 0, 0],
  "rotation": [0, 0, 0],
  "expressions": {
    "happy": {"mouth": "smile", "eyes": "open"},
    "sad": {"mouth": "frown", "eyes": "narrow"}
  },
  "animations": {
    "idle": "bounce",
    "speaking": "subtle_nod"
  },
  "bone_mappings": {
    "custom_head_bone": "head",
    "custom_arm_L": "left_upper_arm"
  }
}
```

---

## Resources

### Finding Avatars
- **Sketchfab** - 3D models (many free, rigged characters)
- **Mixamo** - Free rigged characters + animations
- **Ready Player Me** - Generate custom avatars
- **VRoid Hub** - Anime-style avatars

### Creating Avatars
- **Blender** - Free 3D modeling (blender.org)
- **VRoid Studio** - Anime character creator
- **Character Creator** - Reallusion (paid)
- **Mixamo Fuse** - Adobe character creator

### Avatar Formats
- **Best**: GLB (GLTF Binary) - Includes everything
- **Good**: GLTF (JSON + bins) - More editable
- **Okay**: FBX - Widely supported
- **Basic**: OBJ - No animations/bones
- **2D**: PNG, GIF - Simple, no 3D needed

---

**Happy avatar creating! 🎨**
