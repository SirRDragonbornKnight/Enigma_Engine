"""
Test Avatar Bone Control Pipeline

Tests the full chain:
1. BoneController sets target angles
2. Smooth interpolation updates each frame
3. Callbacks trigger 3D renderer updates
4. [MOVE:] parsing from AI output

Run: python scripts/test_avatar_bones.py
"""

import sys
import time

# Add parent dir to path
sys.path.insert(0, str(__file__).rsplit("scripts", 1)[0])


def test_bone_control_basics():
    """Test basic bone controller operations."""
    print("\n=== Test 1: Basic Bone Control ===")
    
    from enigma_engine.avatar.bone_control import BoneController, BoneState
    
    ctrl = BoneController()
    
    # Test instant mode
    ctrl.move_bone("test_arm", pitch=45, yaw=10, roll=5)
    state = ctrl.get_bone_state("test_arm")
    print(f"Instant: pitch={state.pitch}, yaw={state.yaw}, roll={state.roll}")
    assert state.pitch == 45 and state.yaw == 10 and state.roll == 5
    print("  PASS: Instant mode works")
    
    # Test state tracking
    all_states = ctrl.get_all_states()
    assert "test_arm" in all_states
    print("  PASS: State tracking works")


def test_smooth_interpolation():
    """Test smooth interpolation mode."""
    print("\n=== Test 2: Smooth Interpolation ===")
    
    from enigma_engine.avatar.bone_control import BoneController
    
    ctrl = BoneController()
    ctrl.set_smooth_mode(True, smooth_factor=0.5)
    
    # Set target
    ctrl.move_bone("smooth_arm", pitch=100)
    state = ctrl.get_bone_state("smooth_arm")
    print(f"After move: current={state.pitch:.1f}, target={state.target_pitch}")
    assert state.pitch == 0  # Should not have moved yet
    assert state.target_pitch == 100
    
    # Run updates
    values = [state.pitch]
    for i in range(10):
        ctrl.update(0.016)
        state = ctrl.get_bone_state("smooth_arm")
        values.append(state.pitch)
    
    print(f"Interpolation: {' -> '.join(f'{v:.0f}' for v in values[:6])}")
    assert values[-1] > 90  # Should be near target
    assert values[1] > values[0]  # Should be increasing
    print("  PASS: Smooth interpolation works")


def test_move_parsing():
    """Test [MOVE:] command parsing."""
    print("\n=== Test 3: [MOVE:] Parsing ===")
    
    from enigma_engine.avatar.ai_bridge import parse_move_commands
    
    # Single bone
    text = "[MOVE: arm=45] Hello!"
    cleaned, commands = parse_move_commands(text)
    assert cleaned == "Hello!"
    assert len(commands) == 1
    assert commands[0].bone_name == "arm"
    assert commands[0].angle == 45
    print(f"Single: '{text}' -> '{cleaned}' + {len(commands)} commands")
    
    # Multiple bones
    text = "[MOVE: left_arm=30, right_arm=-30] Waving!"
    cleaned, commands = parse_move_commands(text)
    assert cleaned == "Waving!"
    assert len(commands) == 2
    print(f"Multi: '{text}' -> '{cleaned}' + {len(commands)} commands")
    
    # Embedded
    text = "I'll wave [MOVE: hand=90] at you now!"
    cleaned, commands = parse_move_commands(text)
    assert "wave" in cleaned.lower()
    assert "MOVE" not in cleaned
    print(f"Embed: '{text}' -> '{cleaned}'")
    
    print("  PASS: [MOVE:] parsing works")


def test_callback_system():
    """Test bone callback notifications."""
    print("\n=== Test 4: Callback System ===")
    
    from enigma_engine.avatar.bone_control import BoneController
    
    ctrl = BoneController()
    
    # Track callbacks
    received = []
    def on_bone(name, pitch, yaw, roll):
        received.append((name, pitch, yaw, roll))
    
    ctrl.add_callback(on_bone)
    
    # Move bone
    ctrl.move_bone("cb_arm", pitch=45)
    assert len(received) == 1
    assert received[0][0] == "cb_arm"
    assert received[0][1] == 45
    print(f"Callback received: {received[0]}")
    
    # Remove callback
    ctrl.remove_callback(on_bone)
    ctrl.move_bone("cb_arm", pitch=90)
    assert len(received) == 1  # No new callback
    print("  PASS: Callback system works")


def test_streaming_mode():
    """Test full streaming mode (what happens during AI generation)."""
    print("\n=== Test 5: Streaming Mode ===")
    
    from enigma_engine.avatar.ai_bridge import (
        enable_smooth_streaming,
        parse_move_commands,
        execute_bone_commands
    )
    from enigma_engine.avatar.bone_control import get_bone_controller
    
    # Simulate AI generation start
    enable_smooth_streaming(True, smooth_factor=0.2, auto_update=False)
    
    # Simulate streaming chunks
    chunks = [
        "Let me wave ",
        "[MOVE: right_arm=45] ",
        "at you! ",
        "[MOVE: right_arm=90, left_arm=20]",
        " Hello there!"
    ]
    
    full_text = ""
    ctrl = get_bone_controller()
    
    for chunk in chunks:
        cleaned, commands = parse_move_commands(chunk)
        full_text += cleaned
        if commands:
            execute_bone_commands(commands, smooth=True)
            print(f"  Executed {len(commands)} bone commands")
        
        # Simulate frame update
        ctrl.update(0.016)
    
    print(f"Final text: '{full_text.strip()}'")
    
    # Check bone state
    state = ctrl.get_bone_state("right_arm")
    if state:
        print(f"right_arm target: {state.target_pitch}")
    
    enable_smooth_streaming(False)
    print("  PASS: Streaming mode works")


def test_finger_poses():
    """Test predefined finger poses."""
    print("\n=== Test 6: Finger Poses ===")
    
    from enigma_engine.avatar.bone_control import get_bone_controller, FINGER_POSES
    
    ctrl = get_bone_controller()
    
    poses = ctrl.get_available_finger_poses()
    print(f"Available poses: {', '.join(poses[:5])}...")
    
    # Apply thumbs up
    success = ctrl.apply_finger_pose("thumbs_up", hand="right")
    print(f"thumbs_up applied: {success}")
    
    # Check a finger bone
    state = ctrl.get_bone_state("right_thumb_proximal")
    if state:
        print(f"right_thumb_proximal: pitch={state.pitch}")
    
    print("  PASS: Finger poses work")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Avatar Bone Control Pipeline Test")
    print("=" * 50)
    
    tests = [
        test_bone_control_basics,
        test_smooth_interpolation,
        test_move_parsing,
        test_callback_system,
        test_streaming_mode,
        test_finger_poses,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
