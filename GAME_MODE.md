# Game Mode - Zero Lag Gaming with AI Companion

Game Mode allows Enigma AI Engine to run in the background while you play games without causing frame drops or lag. The AI automatically detects when you're gaming and reduces its resource usage to near-zero, while still remaining responsive when you need it.

## Features

### Automatic Game Detection
- **40+ Known Games**: Automatically recognizes popular games including CS:GO, VALORANT, Minecraft, Fortnite, Cyberpunk 2077, Elden Ring, and more
- **Fullscreen Detection**: Detects any fullscreen application
- **Custom Games**: Add your own games to the detection list
- **Cross-Platform**: Works on Windows, Linux, and macOS

### Resource Management
- **CPU Limiting**: Reduces AI CPU usage to <5% (aggressive) or <10% (balanced)
- **GPU Offloading**: Forces CPU-only inference when game needs GPU
- **Memory Control**: Limits AI RAM usage to 300-500MB
- **Background Tasks**: Automatically pauses autonomous mode during gaming

### Two Modes

**Balanced Mode (Default)**
- AI uses up to 10% CPU
- 500MB RAM limit
- CPU-only inference
- Background tasks disabled
- AI responds when called with shorter responses

**Aggressive Mode**
- AI uses up to 5% CPU
- 300MB RAM limit
- CPU-only inference
- Background tasks disabled
- Maximum performance, AI only responds when explicitly called

## Usage

### Enable Game Mode

**From GUI (Settings Tab):**
1. Go to Settings tab
2. Find "Game Mode - Zero Lag Gaming" section
3. Check "Enable Game Mode"
4. Optionally check "Aggressive Mode" for maximum performance

**From Status Bar:**
- Click the "Game Mode: OFF" indicator in the bottom right to quick-toggle

### Manual Control

If auto-detection doesn't work for your game:
1. Enable Game Mode in Settings
2. Click "Toggle Game Mode Manually" to activate/deactivate

### Configuration

Game Mode settings are saved in `data/game_mode_config.json`:
```json
{
  "enabled": true,
  "aggressive": false,
  "custom_games": ["MyGame.exe"]
}
```

## How It Works

### Detection Loop
Game Mode runs a background watcher that checks every 5 seconds for:
1. Known game processes (from the built-in list)
2. Fullscreen applications
3. Custom games you've added

### Resource Enforcement
When a game is detected:
1. **Autonomous Mode**: Pauses completely if game mode is active
2. **Inference Engine**: 
   - Limits response tokens to 50-100 (faster responses)
   - Forces CPU inference if GPU not allowed
   - Checks resource limits before generating
3. **GUI**: Shows "Game Mode: ACTIVE" in status bar

### Restoration
When the game closes:
1. Waits 5 seconds (in case you're restarting)
2. Checks again for games
3. If no games found, restores full AI functionality

## Integration Points

### Autonomous Mode (`enigma_engine/core/autonomous.py`)
```python
# Checks game mode before each action
if game_mode.is_active():
    limits = game_mode.get_resource_limits()
    if not limits.background_tasks:
        # Pause autonomous actions
        continue
```

### Inference Engine (`enigma_engine/core/inference.py`)
```python
# Applies resource limits during generation
if game_mode.is_active():
    limits = game_mode.get_resource_limits()
    if not limits.inference_allowed:
        return "AI is paused during game mode."
    max_gen = min(max_gen, limits.max_response_tokens)
```

### GUI (`enigma_engine/gui/enhanced_window.py`)
```python
# Status bar indicator
self.game_mode_indicator = QLabel("Game Mode: OFF")
# Updates on game detection
game_mode.on_game_detected(self._on_game_detected)
```

## Known Games List

The following games are automatically detected:

**Competitive FPS:**
- CS:GO, CS2
- VALORANT
- Overwatch
- Apex Legends
- Fortnite
- Call of Duty

**VR Games:**
- VRChat
- Beat Saber
- Half-Life: Alyx
- Boneworks

**RPG/Adventure:**
- Cyberpunk 2077
- Elden Ring
- Baldur's Gate 3
- The Witcher 3
- Starfield
- Skyrim, Fallout 4

**Strategy:**
- Stellaris
- Civilization VI
- Total War series
- Age of Empires IV
- Hearts of Iron IV
- Europa Universalis IV

**Launchers:**
- Steam
- Epic Games Launcher
- Battle.net
- Origin
- EA Desktop
- Ubisoft Connect

...and more! See `enigma_engine/core/process_monitor.py` for the complete list.

## API Reference

### GameMode Class
```python
from enigma_engine.core.game_mode import get_game_mode

game_mode = get_game_mode()

# Enable/disable
game_mode.enable(aggressive=False)  # Balanced mode
game_mode.enable(aggressive=True)   # Aggressive mode
game_mode.disable()

# Check status
game_mode.is_enabled()  # Returns bool
game_mode.is_active()   # Returns bool (game detected)
game_mode.get_status()  # Returns dict

# Get current limits
limits = game_mode.get_resource_limits()
print(limits.max_cpu_percent)    # 5.0 or 10.0
print(limits.gpu_allowed)        # False in game mode
print(limits.background_tasks)   # False in game mode
```

### ProcessMonitor Class
```python
from enigma_engine.core.process_monitor import get_process_monitor

monitor = get_process_monitor()

# Check for games
if monitor.is_game_running():
    print("Game detected!")

# Get running games
games = monitor.get_running_games()  # Returns set of process names

# Get fullscreen app
app = monitor.get_fullscreen_app()   # Returns process name or None

# Add custom game
monitor.add_custom_game("MyGame.exe")
```

### ResourceLimits Class
```python
from enigma_engine.core.resource_limiter import ResourceLimits

# Create custom limits
limits = ResourceLimits(
    max_cpu_percent=5.0,
    max_memory_mb=500,
    gpu_allowed=False,
    background_tasks=False,
    inference_allowed=True,
    max_response_tokens=50,
    batch_processing=False
)
```

## Platform-Specific Notes

### Windows
- Full support for process detection via `tasklist`
- Fullscreen detection via Windows API
- GPU usage monitoring via `nvidia-smi` (NVIDIA GPUs)

### Linux
- Process detection via `ps`
- Fullscreen detection via X11/Wayland (requires `xprop`)
- GPU monitoring via `nvidia-smi` (NVIDIA GPUs)

### macOS
- Process detection via `ps`
- Fullscreen detection via AppleScript
- GPU monitoring not yet implemented

## Performance Impact

### Before Game Mode
- AI using 15-25% CPU
- AI using 2-4GB RAM
- Occasional frame drops when AI generates responses
- Background tasks competing for resources

### After Game Mode
- AI using <5% CPU (aggressive) or <10% (balanced)
- AI using 300-500MB RAM
- No noticeable frame drops
- AI still responds when called
- Background tasks paused

## Troubleshooting

### Game Not Detected
1. Check if game process is in the known games list
2. Add your game manually: Settings > Game Mode > Add custom game
3. Ensure game is actually fullscreen (not borderless windowed)

### AI Still Using Too Much Resources
1. Enable "Aggressive Mode" in Settings
2. Manually activate Game Mode before launching game
3. Check resource usage with Task Manager/Activity Monitor

### AI Not Responding During Game
- This is normal if using aggressive mode
- Try balanced mode instead
- You can still ask questions, responses will just be shorter

## Files Changed

### Core Files
- `enigma_engine/core/game_mode.py` - Main GameMode class
- `enigma_engine/core/process_monitor.py` - Process and game detection
- `enigma_engine/core/resource_limiter.py` - Resource limit enforcement
- `enigma_engine/core/autonomous.py` - Modified to respect game mode
- `enigma_engine/core/inference.py` - Modified to apply resource limits

### GUI Files
- `enigma_engine/gui/tabs/settings_tab.py` - Added game mode controls
- `enigma_engine/gui/enhanced_window.py` - Added status bar indicator

### Configuration
- `enigma_engine/config/defaults.py` - Added game mode config section

## Dependencies

- `psutil>=5.9.0` - For CPU/memory monitoring (already in requirements.txt)
- Platform-specific process detection tools (built-in)

## Future Enhancements

Potential improvements for future versions:
- Per-game profiles (different settings for different games)
- FPS monitoring to dynamically adjust limits
- Integration with game launchers
- Steam API integration
- Notification when game starts/ends
- Hotkey support for quick toggle
- Game-specific AI personalities
