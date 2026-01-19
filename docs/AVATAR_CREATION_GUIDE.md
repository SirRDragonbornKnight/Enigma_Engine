# ForgeAI Avatar Creation Guide

Complete guide to creating custom avatars for ForgeAI.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Avatar Types](#avatar-types)
3. [Creating from Scratch](#creating-from-scratch)
4. [Using Templates](#using-templates)
5. [Emotion System](#emotion-system)
6. [Bundle Format](#bundle-format)
7. [Advanced: 3D Avatars](#advanced-3d-avatars)
8. [AI Control Commands](#ai-control-commands)

---

## Quick Start

### Easiest Way: Use Sample Avatars
1. Open ForgeAI GUI
2. Go to **Avatar** tab
3. Click **Generate Samples**
4. Select one from the gallery

### Import Existing Images
1. Prepare 8 images (256x256 PNG recommended)
2. Name them: `neutral.png`, `happy.png`, `sad.png`, etc.
3. Put them in a folder
4. Click **Import Avatar...** in Avatar tab
5. Drag the folder or select it

---

## Avatar Types

ForgeAI supports several avatar types:

| Type | Description | Best For |
|------|-------------|----------|
| `HUMAN` | Realistic or anime human | Personal assistants |
| `ANIMAL` | Cats, dogs, birds, etc. | Friendly companions |
| `ROBOT` | Mechanical, sci-fi | Technical assistants |
| `FANTASY` | Mythical creatures | Creative AI |
| `ABSTRACT` | Shapes, patterns | Minimalist style |

Each type has specific emotion mappings. For example, a robot shows "status_ok" for happy, while a human shows a smile.

---

## Creating from Scratch

### Step 1: Choose Your Size
- **Recommended**: 256x256 pixels
- **Minimum**: 64x64 pixels  
- **Maximum**: 512x512 pixels (larger slows rendering)

### Step 2: Create the Base Image
Draw your avatar in a neutral pose. Tips:
- Keep the character centered
- Leave padding around edges (10%)
- Use transparent background (PNG)
- Consistent lighting direction

### Step 3: Create Emotion Variants
Copy your base and modify for each emotion:

```
neutral.png    - Default, calm expression
happy.png      - Smile, raised eyebrows
sad.png        - Downturned mouth, droopy eyes
surprised.png  - Wide eyes, open mouth
thinking.png   - One eyebrow raised, looking up/away
confused.png   - Tilted head, squiggly mouth
angry.png      - Furrowed brows, frown
excited.png    - Big smile, sparkly eyes
```

### Step 4: Create manifest.json
```json
{
  "name": "My Avatar",
  "version": "1.0",
  "author": "Your Name",
  "description": "A custom avatar for ForgeAI",
  "avatar_type": "HUMAN",
  "emotions": {
    "neutral": "neutral.png",
    "happy": "happy.png",
    "sad": "sad.png",
    "surprised": "surprised.png",
    "thinking": "thinking.png",
    "confused": "confused.png",
    "angry": "angry.png",
    "excited": "excited.png"
  }
}
```

### Step 5: Folder Structure
```
my_avatar/
├── manifest.json
├── neutral.png
├── happy.png
├── sad.png
├── surprised.png
├── thinking.png
├── confused.png
├── angry.png
└── excited.png
```

### Step 6: Import
Drag the folder onto the Avatar tab import area, or use **Import Avatar...**.

---

## Using Templates

### Generate Templates in GUI
1. Avatar tab → **Generate Samples**
2. This creates template folders in `data/avatar/samples/`

### Generate Templates via Code
```python
from forge_ai.avatar.template_generator import generate_template

# Create templates for "my_robot" avatar
result = generate_template(
    name="my_robot",
    size=256,
    emotions=["neutral", "happy", "sad", "error", "processing"]
)

print(f"Created: {result['individual']}")
```

### Template Guidelines
The generated templates include:
- Red dashed lines: Center and margin guides
- Blue dotted lines: Face proportion guides
- Emotion label in top-left

Draw over these guides, then delete the template layer.

---

## Emotion System

### Standard Emotions (8)
Every avatar should have these:
1. `neutral` - Default state
2. `happy` - Positive response
3. `sad` - Negative/apologetic
4. `surprised` - Unexpected input
5. `thinking` - Processing
6. `confused` - Unclear input
7. `angry` - Frustrated (use sparingly)
8. `excited` - Very positive

### Extended Emotions (Optional)
Add more nuance:
- `curious` - Asking questions
- `sleepy` - Low activity
- `loving` - Appreciation
- `worried` - Concern
- `determined` - Focused
- `mischievous` - Playful
- `embarrassed` - Mistakes
- `proud` - Achievements

### Automatic Detection
ForgeAI automatically detects emotions from AI responses using keywords:
- "happy", "great", "wonderful" → `happy`
- "sorry", "unfortunately" → `sad`
- "hmm", "let me think" → `thinking`
- etc.

### Manual AI Control
The AI can explicitly control the avatar:
```
[emotion:happy] I'm glad to help!
[gesture:wave] Hello there!
[pose:excited] [emotion:excited] That's amazing news!
```

---

## Bundle Format

### .forgeavatar Format
Bundle your avatar for sharing:

1. **Structure**
```
my_avatar.forgeavatar (ZIP file)
├── manifest.json
├── emotions/
│   ├── neutral.png
│   ├── happy.png
│   └── ...
├── animations/ (optional)
│   └── idle.gif
└── models/ (optional)
    └── model.glb
```

2. **Create a Bundle**
```python
from forge_ai.avatar import AvatarBundleCreator

creator = AvatarBundleCreator()
bundle_path = creator.create_from_folder(
    folder_path="my_avatar",
    output_path="my_avatar.forgeavatar"
)
```

3. **Install a Bundle**
```python
from forge_ai.avatar import install_avatar_bundle

install_avatar_bundle("my_avatar.forgeavatar")
```

---

## Advanced: 3D Avatars

### Supported Formats
- `.glb` / `.gltf` - Recommended (GLB is single-file)
- `.obj` - Basic meshes
- `.fbx` - With animations (limited)

### Creating 3D Avatars

1. **Using Blender**
   - Model your character
   - Create shape keys for emotions (optional)
   - Export as GLB
   - Place in avatar folder

2. **Structure**
```
my_3d_avatar/
├── manifest.json
├── model.glb
└── emotions/
    └── neutral.png  (fallback for 2D mode)
```

3. **Manifest for 3D**
```json
{
  "name": "3D Robot",
  "avatar_type": "ROBOT",
  "render_mode": "3D",
  "model": "model.glb",
  "emotions": {
    "neutral": "emotions/neutral.png"
  }
}
```

### VRM Support (Coming Soon)
VRM format support for anime-style avatars is planned.

---

## AI Control Commands

### Emotion Commands
```
[emotion:happy]     Set happy expression
[emotion:sad]       Set sad expression
[emotion:thinking]  Set thinking expression
```

### Gesture Commands
```
[gesture:wave]      Wave animation
[gesture:nod]       Nodding animation
[gesture:shake]     Head shake
[gesture:shrug]     Shrugging
```

### Pose Commands
```
[pose:excited]      Excited pose
[pose:relaxed]      Relaxed stance
[pose:alert]        Alert posture
```

### Training AI to Use Commands
Add to your training data:
```
User: Hello!
AI: [gesture:wave] [emotion:happy] Hello! Great to see you!

User: I'm having a bad day
AI: [emotion:sad] I'm sorry to hear that. [emotion:caring] Would you like to talk about it?
```

---

## Troubleshooting

### Avatar Not Showing
- Check if avatar module is enabled in Modules tab
- Ensure "Show Desktop Avatar" is toggled on
- Verify image paths in manifest.json

### Wrong Expression Showing
- Check emotion name spelling in manifest
- Ensure file exists at specified path
- Try re-importing the avatar

### 3D Model Sideways
- Use orientation sliders in 3D Viewer Settings
- Or fix rotation in Blender before export

### Animations Not Playing
- GIF animations need to be in animations/ folder
- Check manifest references the correct file

---

## Resources

- **Sample Avatars**: Click "Generate Samples" in GUI
- **Templates**: `data/avatar/templates/`
- **Installed Avatars**: `data/avatar/installed/`
- **Community**: Share .forgeavatar files with others!

Happy avatar creating! 🎨
