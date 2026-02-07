# Global Hotkey System

## Overview

The global hotkey system allows users to summon Enigma AI Engine from anywhere, even when the application is not focused or when playing fullscreen games. This is implemented through platform-specific backends that register system-wide keyboard shortcuts.

## Architecture

### Core Components

1. **HotkeyManager** (`enigma_engine/core/hotkey_manager.py`)
   - Central management of all hotkeys
   - Platform detection and backend initialization
   - Hotkey registration/unregistration
   - Singleton pattern for global access

2. **Platform Backends** (`enigma_engine/core/hotkeys/`)
   - **Windows** (`windows.py`): Uses Windows API via ctypes
   - **Linux** (`linux.py`): Uses python-xlib or keyboard library
   - **macOS** (`macos.py`): Uses Quartz Event Taps or keyboard library

3. **HotkeyActions** (`enigma_engine/core/hotkey_actions.py`)
   - Callable actions triggered by hotkeys
   - Integration with GUI overlay system
   - Voice input, screenshot, and game mode actions

4. **Configuration Widget** (`enigma_engine/gui/widgets/hotkey_config.py`)
   - GUI for rebinding hotkeys
   - Conflict detection
   - Visual feedback

## Default Hotkeys

| Action | Default Binding | Description |
|--------|----------------|-------------|
| Summon AI | `Ctrl+Shift+Space` | Show AI overlay |
| Dismiss AI | `Escape` | Hide AI overlay |
| Push to Talk | `Ctrl+Shift+T` | Hold to speak |
| Toggle Game Mode | `Ctrl+Shift+G` | Toggle game mode |
| Quick Command | `Ctrl+Shift+C` | Quick command input |
| Screenshot to AI | `Ctrl+Shift+S` | Capture screen and analyze |

## Configuration

Hotkeys can be configured in:

1. **Config file** (`forge_config.json`):
```json
{
  "enable_hotkeys": true,
  "hotkeys": {
    "summon_ai": "Ctrl+Shift+Space",
    "dismiss_ai": "Escape",
    "push_to_talk": "Ctrl+Shift+T",
    "toggle_game_mode": "Ctrl+Shift+G",
    "quick_command": "Ctrl+Shift+C",
    "screenshot_to_ai": "Ctrl+Shift+S"
  }
}
```

2. **GUI Settings Tab**:
   - Navigate to Settings → Global Hotkeys
   - Click "Rebind Selected" to change a hotkey
   - Press new key combination
   - Changes are saved automatically

## Usage

### Programmatic Usage

```python
from enigma_engine.core.hotkey_manager import get_hotkey_manager, DEFAULT_HOTKEYS
from enigma_engine.core.hotkey_actions import get_hotkey_actions

# Get manager instance
manager = get_hotkey_manager()

# Get actions instance
actions = get_hotkey_actions()

# Register a hotkey
manager.register(
    hotkey="Ctrl+Shift+A",
    callback=actions.summon_overlay,
    name="custom_hotkey"
)

# Start listening
manager.start()

# Stop listening
manager.stop()

# Unregister
manager.unregister("custom_hotkey")
```

### Custom Actions

```python
from enigma_engine.core.hotkey_manager import get_hotkey_manager

def my_custom_action():
    print("Custom hotkey pressed!")

manager = get_hotkey_manager()
manager.register("Ctrl+Shift+X", my_custom_action, "my_action")
manager.start()
```

## Platform-Specific Notes

### Windows
- Uses Windows API `RegisterHotKey`
- Requires no additional dependencies
- Works on Windows 7+

### Linux
- Prefers python-xlib for X11 systems
- Falls back to keyboard library for Wayland
- Install with: `pip install python-xlib` or `pip install keyboard`

### macOS
- Prefers PyObjC for Quartz Event Taps
- Falls back to keyboard library
- Install with: `pip install pyobjc` or `pip install keyboard`

## Known Limitations

1. **System Conflicts**: Some hotkeys may conflict with system shortcuts
   - Windows: Win+*, Ctrl+Alt+Delete
   - Linux: Super+*
   - macOS: Cmd+Space, Cmd+Tab

2. **Permissions**: Some platforms require accessibility permissions
   - macOS: Grant accessibility permissions in System Preferences
   - Linux: May require running with appropriate permissions

3. **Game Overlays**: Some games block global hotkeys
   - Anti-cheat systems may interfere
   - Test in windowed mode first

## Testing

Run the manual test script:
```bash
python test_hotkeys_manual.py
```

Or run the pytest suite:
```bash
pytest tests/test_hotkeys.py -v
```

## Troubleshooting

### Hotkeys Not Working

1. Check if hotkeys are enabled:
   - Settings → Global Hotkeys → Enable checkbox

2. Check for conflicts:
   - Try a different key combination
   - Check system shortcuts

3. Platform-specific issues:
   - **Windows**: Ensure no other app is using the hotkey
   - **Linux**: Install python-xlib or keyboard library
   - **macOS**: Grant accessibility permissions

### Backend Not Loading

Check the logs for initialization errors:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from enigma_engine.core.hotkey_manager import get_hotkey_manager
manager = get_hotkey_manager()
```

## Security Considerations

- Hotkeys work globally, even when Enigma AI Engine is not focused
- Be careful with sensitive operations
- Consider disabling hotkeys in secure environments
- Hotkey actions should validate permissions

## Future Enhancements

- [ ] Hotkey recording in GUI (hold to record multi-key)
- [ ] Per-application hotkey profiles
- [ ] Cloud sync of hotkey preferences
- [ ] Gesture-based shortcuts (mouse + keyboard)
- [ ] Voice activation as hotkey alternative
