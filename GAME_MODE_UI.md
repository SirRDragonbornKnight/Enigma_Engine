# Game Mode UI Changes

## 1. Settings Tab - Game Mode Section

```
┌─────────────────────────────────────────────────────────────────────┐
│  Game Mode - Zero Lag Gaming                                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Game Mode automatically detects when you're gaming and reduces AI     │
│  resource usage to prevent frame drops. AI stays responsive but uses   │
│  minimal CPU/GPU.                                                       │
│                                                                         │
│  [✓] Enable Game Mode          Game Mode: Watching for games           │
│                                                                         │
│  [ ] Aggressive Mode (maximum performance)                             │
│      Maximum performance: AI uses absolute minimum resources.          │
│      Balanced: AI can do light background tasks.                       │
│                                                                         │
│  ┌─ Current Limits ─────────────────────────────────────────┐         │
│  │ CPU: <100%, GPU: Allowed, Background Tasks: Enabled      │         │
│  └──────────────────────────────────────────────────────────┘         │
│                                                                         │
│  [ Toggle Game Mode Manually ]                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────┘
```

**When Game is Active:**
```
┌─────────────────────────────────────────────────────────────────────┐
│  Game Mode - Zero Lag Gaming                                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [✓] Enable Game Mode          Game Mode: ACTIVE                       │
│                                                 (green, bold)           │
│  [✓] Aggressive Mode                                                   │
│                                                                         │
│  ┌─ Current Limits ─────────────────────────────────────────┐         │
│  │ CPU: <5%, GPU: Disabled, Background Tasks: Disabled      │         │
│  │                                    (green, bold)          │         │
│  └──────────────────────────────────────────────────────────┘         │
│                                                                         │
│  [ Toggle Game Mode Manually ]                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Status Bar - Game Mode Indicator

**Normal State (Game Mode Disabled):**
```
┌─────────────────────────────────────────────────────────────────────────┐
│ Model: enigma_engine ▼     AI: Ready     Game Mode: OFF                      │
│                                           (gray)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Watching State (Game Mode Enabled, No Game):**
```
┌─────────────────────────────────────────────────────────────────────────┐
│ Model: enigma_engine ▼     AI: Ready     Game Mode: Watching                 │
│                                           (blue)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Active State (Game Detected):**
```
┌─────────────────────────────────────────────────────────────────────────┐
│ Model: enigma_engine ▼     AI: Ready     Game Mode: ACTIVE                   │
│                                           (green, bold)                  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Click Behavior:**
- Clicking "Game Mode: OFF" enables Game Mode (balanced)
- Clicking "Game Mode: Watching" or "Game Mode: ACTIVE" disables Game Mode
- Provides quick toggle without opening Settings tab

## 3. Integration Points

### Autonomous Mode (enigma_engine/core/autonomous.py)
```python
def _run_loop(self):
    while not self._stop_event.is_set():
        # ✓ CHECK GAME MODE
        try:
            from .game_mode import get_game_mode
            game_mode = get_game_mode()
            
            if game_mode.is_active():
                limits = game_mode.get_resource_limits()
                if not limits.background_tasks:
                    # Pause autonomous actions
                    self._stop_event.wait(30)
                    continue
        except Exception as e:
            logger.debug(f"Could not check game mode: {e}")
        
        # Perform autonomous action...
```

### Inference Engine (enigma_engine/core/inference.py)
```python
def generate(self, prompt: str, max_gen: int = 100, ...):
    # ✓ CHECK GAME MODE AND APPLY LIMITS
    try:
        from .game_mode import get_game_mode
        game_mode = get_game_mode()
        
        if game_mode.is_active():
            limits = game_mode.get_resource_limits()
            
            # Check if inference is allowed
            if not limits.inference_allowed:
                return "AI is paused during game mode."
            
            # Apply token limit for faster responses
            if limits.max_response_tokens > 0:
                max_gen = min(max_gen, limits.max_response_tokens)
    except Exception as e:
        logger.debug(f"Could not check game mode: {e}")
    
    # Continue with generation...
```

## 4. User Experience Flow

### Scenario 1: Automatic Detection
```
1. User enables "Enable Game Mode" in Settings
   Status bar shows: "Game Mode: Watching" (blue)

2. User launches CS:GO
   - ProcessMonitor detects csgo.exe
   - GameMode activates automatically
   - Status bar shows: "Game Mode: ACTIVE" (green, bold)
   - Settings tab shows: CPU: <5%, GPU: Disabled

3. User plays game
   - AI uses <5% CPU
   - Autonomous mode paused
   - AI still responds to questions (shorter responses)
   - No frame drops or lag

4. User closes game
   - GameMode waits 5 seconds
   - No game detected, deactivates
   - Status bar shows: "Game Mode: Watching" (blue)
   - Full AI functionality restored
```

### Scenario 2: Manual Toggle
```
1. User launches unknown game (not in detection list)
   Status bar: "Game Mode: OFF"

2. User notices lag, clicks "Game Mode: OFF" in status bar
   - Enables Game Mode immediately
   - Status bar: "Game Mode: ACTIVE"
   - Lag stops

3. User closes game, clicks "Game Mode: ACTIVE"
   - Disables Game Mode
   - Status bar: "Game Mode: OFF"
   - Full performance restored
```

### Scenario 3: Pre-Gaming Setup
```
1. User planning to game, opens Settings
2. Enables "Enable Game Mode"
3. Checks "Aggressive Mode" for maximum performance
4. Launches game
5. GameMode activates automatically with aggressive limits
6. CPU: <5%, GPU: Disabled, Background: Disabled
```

## 5. Visual Indicators

### Colors
- **Gray** (#6c7086): Game Mode OFF (disabled)
- **Blue** (#3b82f6): Game Mode Watching (enabled, no game)
- **Green** (#22c55e): Game Mode ACTIVE (game detected, bold)
- **Orange** (#f59e0b): Warning/Error states

### Font Styles
- Normal: OFF, Watching states
- Bold: ACTIVE state (emphasizes game is running)

### Interactive Elements
- All Game Mode UI elements are clickable
- Status bar indicator provides quick toggle
- Settings tab provides full configuration
