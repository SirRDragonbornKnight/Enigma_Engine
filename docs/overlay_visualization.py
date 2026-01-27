"""
ASCII art visualization of the AI Overlay system for documentation.
"""

MINIMAL_MODE = """
MINIMAL MODE (300x60 pixels)
═══════════════════════════════════════

┌────────────────────────────────────┐
│ 🤖 [▼] [×]  Ready for your input  │
└────────────────────────────────────┘

Features:
- Just avatar + status/response
- Minimal screen space
- Quick glance info
- Can expand to COMPACT
"""

COMPACT_MODE = """
COMPACT MODE (350x150 pixels) - DEFAULT
═══════════════════════════════════════

┌──────────────────────────────────────────┐
│ 🤖 AI Assistant      [▼] [▲] [×]         │
├──────────────────────────────────────────┤
│                                          │
│  AI: Hello! How can I help you           │
│  while you game?                         │
│                                          │
├──────────────────────────────────────────┤
│ [Type your message here...      ] [➤]    │
└──────────────────────────────────────────┘

Features:
- Response area (3 lines)
- Input field
- Send button
- Expand/minimize buttons
"""

FULL_MODE = """
FULL MODE (450x400 pixels)
═══════════════════════════════════════

┌────────────────────────────────────────────────┐
│ 🤖 AI Assistant                    [▼] [×]     │
├────────────────────────────────────────────────┤
│ You: What's my quest objective?               │
│                                                │
│ AI: Based on your current game state,         │
│ your primary objective is to reach the         │
│ northern temple. You'll need the golden        │
│ key from the merchant first.                   │
│                                                │
│ You: Where's the merchant?                     │
│                                                │
│ AI: The merchant is in the town square,        │
│ near the fountain. Look for the blue           │
│ tent with the star symbol.                     │
│                                                │
│ [Scroll for more history...]                   │
├────────────────────────────────────────────────┤
│ [Type your message here...          ] [➤]      │
└────────────────────────────────────────────────┘

Features:
- Full chat history
- Scrollable
- Complete conversation context
- Resizable window
"""

THEME_COMPARISON = """
THEME VARIATIONS
═══════════════════════════════════════

DARK THEME (Default)
┌─────────────────────────────────┐
│ 🤖  Black bg, white text        │
│     Blue accent, 70% opacity    │
└─────────────────────────────────┘

LIGHT THEME
┌─────────────────────────────────┐
│ 🤖  White bg, black text        │
│     Blue accent, 90% opacity    │
└─────────────────────────────────┘

GAMING THEME (Recommended for Games)
┌─────────────────────────────────┐
│ 🤖  Black bg, GREEN text        │
│     Monospace, 50% opacity      │
└─────────────────────────────────┘

MINIMAL THEME
┌─────────────────────────────────┐
│ 🤖  Black bg, white text        │
│     Gray accent, 30% opacity    │
└─────────────────────────────────┘

CYBERPUNK THEME
┌─────────────────────────────────┐
│ 🤖  Black bg, CYAN text         │
│     Magenta accent, 80% opacity │
└─────────────────────────────────┘

STEALTH THEME
┌─────────────────────────────────┐
│ 🤖  Black bg, gray text         │
│     Dark gray, 20% opacity      │
└─────────────────────────────────┘
"""

POSITION_GUIDE = """
SCREEN POSITION OPTIONS
═══════════════════════════════════════

╔═══════════════════════════════════════════╗
║ TOP_LEFT          SCREEN          TOP_RIGHT║
║   ┌────┐                            ┌────┐ ║
║   │ AI │                            │ AI │ ║
║   └────┘                            └────┘ ║
║                                             ║
║                   CENTER                    ║
║                   ┌────┐                    ║
║                   │ AI │                    ║
║                   └────┘                    ║
║                                             ║
║ BOTTOM_LEFT                    BOTTOM_RIGHT ║
║   ┌────┐                            ┌────┐ ║
║   │ AI │                            │ AI │ ║
║   └────┘                            └────┘ ║
╚═══════════════════════════════════════════╝

GAMING RECOMMENDATIONS:
- FPS: TOP_RIGHT (doesn't block crosshair)
- Strategy: BOTTOM_LEFT (away from minimap)
- MOBA: TOP_LEFT (away from minimap at bottom right)
- RPG: Any position (flexible)
"""

USAGE_FLOW = """
OVERLAY USAGE FLOW
═══════════════════════════════════════

1. LAUNCH GAME
   │
   ├─► Game starts in fullscreen/windowed
   │
2. ACTIVATE OVERLAY
   │
   ├─► Press hotkey (Ctrl+Shift+A)
   │   OR
   ├─► Toggle from Settings tab
   │
3. INTERACT WITH AI
   │
   ├─► Type question in input field
   ├─► Press Enter or click Send
   ├─► AI responds in overlay
   │
4. CONTINUE GAMING
   │
   ├─► Overlay stays on top
   ├─► Optional: Enable click-through
   ├─► Switch modes as needed
   │
5. HIDE WHEN DONE
   │
   └─► Press hotkey again
       OR
       Click × button
"""

CLICK_THROUGH_DEMO = """
CLICK-THROUGH MODE
═══════════════════════════════════════

NORMAL MODE (Click-through OFF)
┌──────────────────────────────┐
│  Clicks hit overlay          │
│  Can interact with AI        │
│  Game underneath is blocked  │
└──────────────────────────────┘
         ▼ Click goes to overlay
  ╔═══════════════════╗
  ║  GAME SCREEN     ║  ← Blocked
  ╚═══════════════════╝


CLICK-THROUGH MODE (Click-through ON)
┌──────────────────────────────┐
│  Most clicks pass through    │
│  Only AI elements catch      │
│  Game remains interactive    │
└──────────────────────────────┘
         ▼ Click passes through
  ╔═══════════════════╗
  ║  GAME SCREEN     ║  ← Interactive!
  ╚═══════════════════╝

Perfect for monitoring AI responses
while actively playing!
"""

if __name__ == "__main__":
    print("AI OVERLAY VISUALIZATION")
    print("=" * 60)
    print()
    print(MINIMAL_MODE)
    print()
    print(COMPACT_MODE)
    print()
    print(FULL_MODE)
    print()
    print(THEME_COMPARISON)
    print()
    print(POSITION_GUIDE)
    print()
    print(USAGE_FLOW)
    print()
    print(CLICK_THROUGH_DEMO)
