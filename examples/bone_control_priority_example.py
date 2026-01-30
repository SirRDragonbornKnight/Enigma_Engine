"""
Bone Control Priority System Demo

Shows how bone animation is the PRIMARY control method, with other systems
as fallbacks. Demonstrates priority system preventing conflicts.
"""

from forge_ai.avatar import get_avatar, ControlPriority
from forge_ai.avatar.bone_control import get_bone_controller
from forge_ai.avatar.autonomous import get_autonomous_avatar
import time

def main():
    print("=== Avatar Control Priority Demo ===\n")
    
    # Get avatar and enable it
    avatar = get_avatar()
    avatar.enable()
    print("✓ Avatar enabled\n")
    
    # Get bone controller and link it (PRIMARY control)
    bone_controller = get_bone_controller(avatar_controller=avatar)
    bone_controller.set_avatar_bones(["head", "left_arm", "right_arm"])
    print("✓ Bone controller initialized as PRIMARY control\n")
    
    # Try manual control (should work when bone controller not active)
    print("1. Manual control (USER_MANUAL priority = 80)")
    avatar.move_to(100, 100, requester="manual", priority=ControlPriority.USER_MANUAL)
    print(f"   Current controller: {avatar.current_controller}")
    time.sleep(0.5)
    
    # Bone controller takes over (highest priority = 100)
    print("\n2. Bone controller activates (BONE_ANIMATION priority = 100)")
    bone_controller.move_bone("head", pitch=15, yaw=10, roll=0)
    print(f"   Current controller: {avatar.current_controller}")
    print("   Bone controller has HIGHEST priority - takes control")
    time.sleep(0.5)
    
    # Try manual control again - should be DENIED
    print("\n3. Try manual control while bone controller active")
    result = avatar.request_control("manual", ControlPriority.USER_MANUAL, duration=1.0)
    print(f"   Control granted: {result}")
    print(f"   Current controller: {avatar.current_controller}")
    print("   DENIED - bone controller has higher priority")
    time.sleep(0.5)
    
    # Try autonomous (even lower priority = 50)
    print("\n4. Try autonomous control (AUTONOMOUS priority = 50)")
    autonomous = get_autonomous_avatar(avatar)
    result = avatar.request_control("autonomous", ControlPriority.AUTONOMOUS, duration=1.0)
    print(f"   Control granted: {result}")
    print(f"   Current controller: {avatar.current_controller}")
    print("   DENIED - bone controller still has control")
    time.sleep(0.5)
    
    # Wait for bone controller timeout
    print("\n5. Wait for bone controller control to expire...")
    time.sleep(2.5)
    print(f"   Current controller: {avatar.current_controller}")
    print("   Control expired, now 'none' has control")
    
    # Now manual control should work
    print("\n6. Manual control after timeout (should work)")
    result = avatar.request_control("manual", ControlPriority.USER_MANUAL, duration=1.0)
    print(f"   Control granted: {result}")
    print(f"   Current controller: {avatar.current_controller}")
    print("   SUCCESS - no higher priority controller active")
    
    # Bone controller can override at any time
    print("\n7. Bone controller overrides manual control")
    bone_controller.move_bone("left_arm", pitch=45, yaw=0, roll=10)
    print(f"   Current controller: {avatar.current_controller}")
    print("   OVERRIDE - bone controller takes back control")
    
    print("\n=== Priority Hierarchy ===")
    print("BONE_ANIMATION (100)  - PRIMARY, always wins")
    print("USER_MANUAL (80)      - Second, for direct user input")
    print("AI_TOOL_CALL (70)     - AI explicit commands")
    print("AUTONOMOUS (50)       - Autonomous behaviors (fallback)")
    print("IDLE_ANIMATION (30)   - Background animations")
    print("FALLBACK (10)         - For non-avatar-trained models")
    
    print("\n✓ Demo complete")
    avatar.disable()

if __name__ == "__main__":
    main()
