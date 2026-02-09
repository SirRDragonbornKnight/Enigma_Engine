# AI Overlay System - Gaming and Multitasking Interface

## Overview

The AI Overlay is a transparent, always-on-top window that allows you to interact with Enigma AI Engine while gaming or using other applications. It floats above all windows, providing quick AI access without leaving your current application.

## Features

### Display Modes

The overlay supports multiple display modes to suit different use cases:

1. **MINIMAL** - Just AI avatar and last response
   - Perfect for quick status checks
   - Minimal screen real estate
   - Size: 300x60 pixels

2. **COMPACT** - Avatar, response, and input field
   - Best for gaming
   - Quick chat without leaving game
   - Size: 350x150 pixels

3. **FULL** - Complete chat interface
   - Full chat history
   - Input field with send button
   - Window controls
   - Size: 450x400 pixels

4. **HIDDEN** - Overlay not visible
   - Can be toggled with hotkey
   - Preserves settings

### Themes

Six preset themes are available:

- **dark** - Dark background, white text (default)
- **light** - Light background, dark text
- **gaming** - Low opacity, green accent, monospace font
- **minimal** - Very subtle, minimal visual impact
- **cyberpunk** - Black with cyan/magenta accents
- **stealth** - Nearly invisible, gray tones

### Positioning

The overlay can be positioned at:
- Top Left
- Top Right (default)
- Bottom Left
- Bottom Right
- Center
- Custom (user-defined coordinates)

Position is automatically saved and restored between sessions.

### Transparency

Adjustable opacity from 10% to 100%:
- Higher opacity (80-100%) - Better readability
- Lower opacity (20-50%) - Less intrusive for gaming

### Click-Through Mode

When enabled, mouse clicks pass through the overlay to the game/application beneath:
- Useful for monitoring AI responses during gameplay
- Can still interact with overlay by clicking on specific elements
- Toggle on/off as needed

### Always-On-Top

The overlay stays above all windows including:
- Fullscreen games
- Borderless windowed applications
- Other applications

Can be toggled if you need it to behave like a normal window.

## Usage

### Opening the Overlay

#### From Main GUI:
1. Open Enigma AI Engine
2. Go to Settings tab
3. Find "AI Overlay" section
4. Click "Show Overlay"

#### From Code:
```python
from enigma_engine.gui.overlay import AIOverlay, OverlayMode

# Create overlay
overlay = AIOverlay()

# Show in compact mode
overlay.set_mode(OverlayMode.COMPACT)
overlay.show()
```

### Configuring the Overlay

#### From GUI:
1. Go to Settings tab
2. Find "AI Overlay" section
3. Click "Configure Overlay"
4. Adjust settings:
   - Display mode
   - Position
   - Theme
   - Opacity
   - Click-through
   - Always on top
   - Hotkey

#### From Code:
```python
from enigma_engine.gui.overlay import AIOverlay, OverlayMode, OverlayPosition

overlay = AIOverlay()

# Change mode
overlay.set_mode(OverlayMode.FULL)

# Change position
overlay.set_position(OverlayPosition.TOP_RIGHT)

# Change theme
overlay.set_theme("gaming")

# Change opacity
overlay.set_opacity(0.8)  # 80%

# Enable click-through
overlay.set_click_through(True)
```

### Using with AI Engine

```python
from enigma_engine.core.inference import EnigmaEngine
from enigma_engine.gui.overlay import AIOverlay

# Create engine
engine = EnigmaEngine(model, tokenizer)

# Create and connect overlay
overlay = AIOverlay()
overlay.set_engine(engine)
overlay.show()

# Now the overlay can send messages to AI and receive responses
```

## Configuration

### Default Settings

Settings are stored in `enigma_engine/config/defaults.py`:

```python
"overlay": {
    "enabled": True,
    "mode": "compact",
    "position": "top_right",
    "opacity": 0.9,
    "click_through": False,
    "always_on_top": True,
    "theme": "gaming",
    "hotkey": "Ctrl+Shift+A",
    "remember_position": True,
    "show_on_startup": False,
}
```

### User Settings

Settings are saved to `data/overlay_settings.json` and persist between sessions:
- Current mode
- Position (if remember_position is enabled)
- Theme
- Opacity
- Other preferences

## Game Compatibility

### Supported Game Modes

- **Fullscreen** - Overlay appears on top (may require special handling)
- **Borderless Windowed** - Overlay works perfectly
- **Windowed** - Overlay works perfectly

### Anti-Cheat Compatibility

Some games with anti-cheat software may block overlays:
- BattlEye
- Easy Anti-Cheat
- Vanguard
- GameGuard

The overlay includes detection for known anti-cheat systems and will warn if potential conflicts are detected.

### Recommended Settings for Gaming

```python
{
    "mode": "compact",          # Not too large
    "theme": "gaming",          # Low opacity, readable
    "position": "top_right",    # Out of the way
    "opacity": 0.5,             # Semi-transparent
    "click_through": True,      # Don't block clicks
    "always_on_top": True,      # Always visible
}
```

## Troubleshooting

### Overlay Not Appearing

1. Check if overlay is enabled in config:
   ```python
   from enigma_engine.config import CONFIG
   print(CONFIG.get("overlay", {}).get("enabled"))
   ```

2. Check if overlay is in HIDDEN mode:
   ```python
   overlay.set_mode(OverlayMode.COMPACT)
   overlay.show()
   ```

3. Ensure PyQt5 is installed:
   ```bash
   pip install PyQt5
   ```

### Overlay Blocks Game Clicks

Enable click-through mode:
```python
overlay.set_click_through(True)
```

Or from Settings → Configure Overlay → Check "Click-through"

### Overlay Not Staying On Top

Enable always-on-top:
```python
overlay.settings.always_on_top = True
overlay._setup_window()  # Reapply window flags
```

Or from Settings → Configure Overlay → Check "Always on top"

### Overlay Too Transparent/Opaque

Adjust opacity:
```python
overlay.set_opacity(0.9)  # 90% opaque
```

Or use opacity slider in overlay settings.

## API Reference

### AIOverlay Class

```python
class AIOverlay(QWidget):
    """Main overlay window."""
    
    def __init__(self, config_path: Optional[str] = None)
    def set_mode(self, mode: OverlayMode)
    def set_opacity(self, opacity: float)  # 0.0 to 1.0
    def set_click_through(self, enabled: bool)
    def set_position(self, position: OverlayPosition, x: int = None, y: int = None)
    def set_theme(self, theme_name: str)
    def set_engine(self, engine)
```

### OverlayChatBridge Class

```python
class OverlayChatBridge(QObject):
    """Bridge between overlay and chat system."""
    
    def set_engine(self, engine)
    def send_message(self, text: str)
    def receive_response(self, response: str)
    def sync_history(self, history: List[Dict])
    def get_history(self) -> List[Dict]
    def clear_history()
```

### OverlayCompatibility Class

```python
class OverlayCompatibility:
    """Game and application compatibility."""
    
    def detect_game_mode(self) -> str  # "fullscreen", "borderless", "windowed"
    def adjust_for_game(self, game_mode: str) -> Dict
    def check_game_overlay_support(self, game_exe: str) -> Dict
    def get_recommended_settings(self) -> Dict
```

## Testing

Run the test script to try out all overlay features:

```bash
python test_overlay.py
```

This will open a control panel where you can test:
- Different display modes
- All themes
- Position presets
- Opacity levels

## Platform-Specific Notes

### Windows
- Layered windows for transparency
- Handles DPI scaling automatically
- Works with DirectX/OpenGL games

### Linux
- X11 compositor support required for transparency
- Wayland support via layer-shell (if available)
- May need compositor running for transparency

### macOS
- NSWindow with appropriate level
- Handles Retina displays automatically
- May require accessibility permissions

## Contributing

When adding new features to the overlay:

1. Keep it minimal - gaming overlays should be lightweight
2. Test with multiple games and display modes
3. Ensure transparency works correctly
4. Don't block user input unnecessarily
5. Follow the existing theme system

## License

Part of the Enigma AI Engine project. See main LICENSE file.
