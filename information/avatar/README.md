# Avatar System Documentation

## Overview

The Enigma Avatar System provides a visual representation of the AI that can:
- Display on screen (image-based or overlay)
- Express emotions through different expression images
- Move around the desktop (when overlay enabled)
- "Interact" with windows and files visually
- Integrate with game/robot control

## Architecture

```
enigma/avatar/
├── __init__.py          # Package exports
└── controller.py        # AvatarController - main backend logic

enigma/gui/tabs/
├── avatar_tab.py        # Container with sub-tabs
└── avatar/
    ├── __init__.py
    └── avatar_display.py  # GUI for avatar display
```

## Components

### AvatarController (enigma.avatar.controller)
The main backend controller for avatar functionality:
- **Enable/Disable**: Turn avatar on/off (default: OFF)
- **Movement**: Move around screen, follow mouse
- **Speaking**: Animate when AI speaks
- **Expressions**: Set facial expressions
- **Window Interaction**: Visual effects of "touching" windows
- **Configuration**: Save/load settings

### Avatar Display (GUI Tab)
The GUI component that shows:
- Avatar image display
- Expression state
- Enable/disable toggle
- Config file selector

## Usage

### Basic Usage
```python
from enigma.avatar import get_avatar, enable_avatar, disable_avatar

# Get the global avatar instance
avatar = get_avatar()

# Enable (turn on)
avatar.enable()

# Move avatar
avatar.move_to(500, 300)

# Show expression
avatar.set_expression("happy")

# Speak (animates mouth)
avatar.speak("Hello, I'm Enigma!")

# Disable (turn off)
avatar.disable()
```

### In AI Responses
The AI can control the avatar during responses:
```python
from enigma.gui.tabs.avatar import set_avatar_expression

# When AI generates a happy response
set_avatar_expression(main_window, "happy")

# When AI is thinking
set_avatar_expression(main_window, "thinking")
```

## Configuration

Avatar configs are stored in `data/avatar/` as JSON files:

```json
{
    "name": "My Avatar",
    "image": "avatar.png",
    "expressions": {
        "neutral": "avatar.png",
        "happy": "avatar_happy.png",
        "sad": "avatar_sad.png",
        "thinking": "avatar_thinking.png",
        "surprised": "avatar_surprised.png",
        "confused": "avatar_confused.png"
    },
    "description": "My custom avatar"
}
```

## Supported Expressions
- `neutral` - Default state
- `happy` - Positive response
- `sad` - Negative/empathetic response
- `thinking` - Processing/generating
- `surprised` - Unexpected input
- `confused` - Unclear input
- `angry` - (optional) Frustrated state

## Future Enhancements
See `docs/avatar_roadmap.md` for planned features:
- Live2D animated avatars
- 3D VRM model support
- Lip-sync with TTS
- Motion tracking integration
- Environment simulation

## Default State
**IMPORTANT**: Avatar is OFF by default. User must explicitly enable it.
This prevents unexpected overlay windows on user's desktop.
