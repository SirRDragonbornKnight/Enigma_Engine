# AI Overlay - Quick Reference

## What is it?

A transparent, always-on-top window for AI interaction while gaming or using other apps.

## Quick Start

### Enable Overlay

1. Open ForgeAI
2. Go to **Settings** tab
3. Find **AI Overlay** section
4. Click **Show Overlay**

### Configure

Click **Configure Overlay** in Settings to adjust:
- Display mode (Minimal/Compact/Full)
- Position
- Theme
- Opacity
- Click-through
- Hotkey

## Display Modes

### Minimal (300x60)
```
┌─────────────────────┐
│ 🤖 AI response here │
└─────────────────────┘
```
Perfect for: Glancing at AI responses during gameplay

### Compact (350x150) - Default
```
┌─────────────────────────┐
│ 🤖 AI Name         [×]  │
├─────────────────────────┤
│ AI response here...     │
├─────────────────────────┤
│ [Type here...    ] [➤]  │
└─────────────────────────┘
```
Perfect for: Quick AI chat while gaming

### Full (450x400)
```
┌───────────────────────────┐
│ 🤖 AI Name         [─][×] │
├───────────────────────────┤
│ Chat history              │
│ ...                       │
│ ...                       │
├───────────────────────────┤
│ [Type here...      ] [➤]  │
└───────────────────────────┘
```
Perfect for: Extended conversations

## Themes

- **dark** - Black background, white text
- **light** - White background, black text  
- **gaming** - Semi-transparent, green accent (recommended)
- **minimal** - Very subtle appearance
- **cyberpunk** - Cyan/magenta accents
- **stealth** - Nearly invisible

## Hotkeys

Default: **Ctrl+Shift+A** (configurable)

## Gaming Tips

### Best Settings for Gaming
```
Mode: Compact
Theme: Gaming
Position: Top Right
Opacity: 50%
Click-through: ON
```

### Recommended Positions
- **FPS Games**: Top Right (doesn't block crosshair)
- **Strategy Games**: Bottom Left (doesn't block minimap)
- **MOBA**: Top Left (doesn't block minimap)

## Features

✓ Always-on-top (even over fullscreen games)  
✓ Transparent background  
✓ Click-through mode (clicks pass to game)  
✓ Multiple display modes  
✓ 6 preset themes  
✓ Position memory  
✓ Hotkey toggle  
✓ Draggable (click header to move)

## Compatibility

### ✓ Works With
- Borderless windowed games
- Windowed games  
- Most applications

### ⚠ May Have Issues
- Fullscreen exclusive mode (some games)
- Anti-cheat software (BattlEye, EAC, Vanguard)

## Troubleshooting

**Overlay not showing?**
- Check if mode is set to HIDDEN
- Verify overlay is enabled in config
- Try pressing hotkey (Ctrl+Shift+A)

**Blocking game clicks?**
- Enable click-through mode
- Settings → Configure Overlay → Check "Click-through"

**Can't see overlay text?**
- Increase opacity (Settings → Opacity slider)
- Try a different theme (Gaming theme is readable)

**Overlay hidden behind game?**
- Enable "Always on top"
- Some fullscreen games may block overlays

## Advanced

### Python API
```python
from forge_ai.gui.overlay import AIOverlay, OverlayMode

# Create overlay
overlay = AIOverlay()

# Configure
overlay.set_mode(OverlayMode.COMPACT)
overlay.set_theme("gaming")
overlay.set_opacity(0.8)
overlay.set_click_through(True)

# Show
overlay.show()
```

### Testing
```bash
# Run overlay tests
python tests/test_overlay.py

# Run interactive test
python test_overlay.py
```

## Full Documentation

See `docs/OVERLAY_GUIDE.md` for complete documentation.

## Platform Support

- ✓ Windows (Tested)
- ✓ Linux (X11 compositor required)
- ✓ macOS (Tested)
