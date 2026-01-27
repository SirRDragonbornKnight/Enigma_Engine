# ForgeAI GUI Mode System

## Overview

ForgeAI now features a flexible GUI mode system that adapts the interface to different user needs and use cases. The system provides four distinct modes, each optimized for specific workflows.

## GUI Modes

### 1. Simple Mode
**Target Users:** Beginners, casual users  
**Features:** Essential chat and basic controls only  
**Visible Tabs:** Chat, Settings

Perfect for users who just want to talk to their AI without complexity.

### 2. Standard Mode (Recommended)
**Target Users:** Regular users, content creators  
**Features:** Balanced feature set with consolidated tabs  
**Visible Tabs:** Chat, Workspace, History, Create (consolidated), AI (consolidated), Settings

The recommended mode for most users, providing access to common features through an organized interface.

### 3. Advanced Mode
**Target Users:** Power users, developers  
**Features:** All features exposed  
**Visible Tabs:** All 27+ tabs available

Full access to every feature and configuration option.

### 4. Gaming Mode
**Target Users:** Gamers  
**Features:** Minimal interface optimized for low resource usage  
**Visible Tabs:** Chat, Settings  
**Special Features:** Resource monitoring, quick summon hotkey

Designed to run in the background while gaming with minimal CPU/GPU impact.

## Key Features

### Tab Consolidation
In Standard mode, related tabs are grouped together:
- **Create Tab:** Image, Video, Audio, Code generation in one place
- **AI Tab:** Avatar, Modules, Scaling, Training in one place

### Quick Actions Bar
Floating toolbar for common tasks (visible in all modes):
- Screenshot capture → Send to AI
- Voice input toggle
- Game mode toggle
- New conversation
- Quick image generation

### Feedback Buttons
Rate AI responses with thumbs up/down to improve learning.

### Keyboard Shortcuts
Full keyboard navigation for power users:
- `Ctrl+1/2/3` - Switch between main tabs
- `Ctrl+N` - New conversation
- `Ctrl+,` - Open Settings
- `Ctrl+Enter` - Send message
- `Ctrl+Shift+Space` - Toggle game mode / Summon AI
- `F1` - Quick help

### Gaming Theme
Modern dark theme with indigo/cyan accents:
- Smooth transitions
- Optimized for readability
- Optional hover effects
- Clean, professional appearance

## Usage

### Switching Modes
1. Open ForgeAI
2. Go to **View** menu in the menu bar
3. Select **GUI Mode** submenu
4. Choose your desired mode

The interface will instantly update to show/hide tabs based on the selected mode.

### First Run Setup
New users are greeted with a setup wizard that:
1. Explains the different modes
2. Helps choose the right mode for their needs
3. Configures basic settings
4. Sets up hotkeys

### Configuration
GUI mode preference is saved automatically and persists across sessions. You can change modes at any time from the View menu.

## Technical Details

### Files Added
```
forge_ai/gui/
├── gui_modes.py           # Mode system and manager
├── themes/
│   ├── gaming_theme.py    # Gaming aesthetic stylesheet
│   └── __init__.py
├── widgets/
│   ├── quick_actions.py   # Quick actions bar, feedback buttons
│   └── __init__.py
├── wizards/
│   ├── first_run.py       # First-run setup wizard
│   └── __init__.py
└── tabs/
    ├── create_tab.py      # Consolidated creation tab
    └── ai_tab.py          # Consolidated AI configuration tab
```

### Files Modified
- `forge_ai/gui/enhanced_window.py` - Integrated mode system
- `forge_ai/gui/tabs/__init__.py` - Added new consolidated tabs
- `forge_ai/gui/tabs/chat_tab.py` - Added quick actions and feedback
- `forge_ai/config/defaults.py` - Added GUI configuration defaults

### Configuration Options
In `forge_ai/config/defaults.py`:
```python
"gui_mode": "standard",           # Current GUI mode
"gui_theme": "dark",              # Theme to use
"enable_quick_actions": True,     # Show quick actions bar
"enable_feedback_buttons": True,  # Show feedback buttons
"show_game_mode_indicator": True, # Show game mode status
```

## Benefits

### For Beginners
- Less overwhelming interface
- Guided onboarding
- Clear path to basic features

### For Regular Users
- Organized, logical tab structure
- Quick access to common features
- Consistent, modern appearance

### For Power Users
- Full feature access when needed
- Keyboard-driven workflows
- Customizable preferences

### For Gamers
- Minimal resource footprint
- Quick summon from any game
- Low distraction interface

## Future Enhancements

Planned improvements:
- [ ] Custom mode configurations
- [ ] Per-mode theme preferences
- [ ] Responsive layouts for different screen sizes
- [ ] Touch-optimized mobile mode
- [ ] Voice-only mode
- [ ] Accessibility improvements

## Feedback

We welcome feedback on the new GUI mode system! Please report issues or suggestions through GitHub Issues.
