# Avatar Control Priority System - Implementation Summary

## Problem Solved

Multiple avatar control systems were conflicting with each other:
- **BoneController** - Direct bone manipulation for rigged 3D models
- **AutonomousAvatar** - Self-acting behaviors
- **AvatarController** - Manual control (move_to, set_expression, etc.)
- **Various Animators** - Animation systems

No coordination meant they could override each other unpredictably.

## Solution Implemented

### Priority-Based Control System

Added a priority hierarchy to AvatarController where **bone animation is PRIMARY**:

```python
class ControlPriority(IntEnum):
    BONE_ANIMATION = 100    # PRIMARY: Direct bone control for rigged models
    USER_MANUAL = 80        # User dragging/clicking avatar
    AI_TOOL_CALL = 70       # AI explicit commands via tools
    AUTONOMOUS = 50         # Autonomous behaviors (FALLBACK)
    IDLE_ANIMATION = 30     # Background idle animations
    FALLBACK = 10          # Last resort (for non-avatar-trained models)
```

### How It Works

1. **Request Control**: Any system must request control before acting
   ```python
   avatar.request_control("bone_controller", ControlPriority.BONE_ANIMATION, duration=1.0)
   ```

2. **Priority Rules**:
   - Higher priority ALWAYS overrides lower priority
   - Control expires after `duration` seconds (default 2.0s)
   - Same requester can extend their own control
   - Lower priority requests are DENIED while higher priority active

3. **Automatic Fallback**: When bone controller isn't active, other systems work normally

## Files Modified

### 1. forge_ai/avatar/controller.py
- Added `ControlPriority` enum
- Added `_control_lock`, `_current_controller`, `_current_priority` tracking
- Added `request_control()` and `release_control()` methods
- Added `current_controller` property to check who's in control
- Modified `move_to()` and `set_expression()` to request control first

### 2. forge_ai/avatar/bone_control.py
- Updated docstring to mark as PRIMARY control system
- Added `link_avatar_controller()` method
- Modified `move_bone()` to request BONE_ANIMATION priority
- Updated `get_bone_controller()` to accept avatar_controller parameter for linking

### 3. forge_ai/avatar/autonomous.py
- Updated docstring to mark as FALLBACK system
- Modified mood changes to use AUTONOMOUS priority
- Modified expression changes to use AUTONOMOUS priority
- Added try/except for backward compatibility

### 4. forge_ai/avatar/__init__.py
- Exported `ControlPriority` for use in other modules

### 5. .github/copilot-instructions.md
- Documented the priority system
- Clarified bone animation as PRIMARY control
- Listed priority values for reference

## New Files

### examples/bone_control_priority_demo.py
Demo showing:
- Bone controller taking control (highest priority)
- Manual control being denied while bone controller active
- Autonomous control being denied (lower priority)
- Control expiring and becoming available again
- Bone controller overriding at any time

## Usage Example

```python
from forge_ai.avatar import get_avatar, ControlPriority
from forge_ai.avatar.bone_control import get_bone_controller

# Setup
avatar = get_avatar()
avatar.enable()

# Link bone controller as primary (highest priority)
bone_controller = get_bone_controller(avatar_controller=avatar)
bone_controller.set_avatar_bones(["head", "left_arm", "right_arm"])

# Bone control takes precedence
bone_controller.move_bone("head", pitch=15, yaw=10, roll=0)
# Now avatar.current_controller == "bone_controller"

# Manual control is denied while bone controller active
avatar.move_to(100, 100)  # Will be blocked!

# After 1 second (control timeout), manual control works again
time.sleep(1.1)
avatar.move_to(100, 100)  # Now works!
```

## Backward Compatibility

Old code that doesn't use the priority system still works:
- Default `requester="manual"` and `priority=ControlPriority.USER_MANUAL`
- Autonomous system has try/except fallback for old signature
- All existing code continues to function

## Benefits

1. ✅ **No More Conflicts** - Clear priority hierarchy prevents fighting between systems
2. ✅ **Bone Animation Primary** - Rigged model control always takes precedence
3. ✅ **Automatic Fallback** - When bone control not active, other systems work
4. ✅ **User Feedback** - Can check `avatar.current_controller` to see who's in control
5. ✅ **Flexible** - Easy to add new control systems with appropriate priorities
6. ✅ **Safe** - Lower priority systems automatically yield to higher priority

## Future Enhancements

Potential improvements:
- GUI indicator showing current controller
- Visual feedback when control requests are denied
- Priority override button for user to force manual control
- Logging/telemetry of control handoffs
- Per-bone control priorities (head higher priority than arms, etc.)
