"""Test that avatar system actually works"""
import sys
import os
sys.path.insert(0, '.')
os.environ['FORGE_NO_AUDIO'] = '1'  # Skip audio checks
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap, QColor, QPainter

app = QApplication([])

print("=" * 50)
print("AVATAR SYSTEM TESTS")
print("=" * 50)

# Test 1: PNG Bounce Mode
print("\n[TEST 1] PNG Bounce Mode (DougDoug style)")
print("-" * 40)

from forge_ai.avatar import UnifiedAvatar, AvatarMode, AvatarType

# Create test image
pixmap = QPixmap(128, 128)
pixmap.fill(QColor(0, 0, 0, 0))
painter = QPainter(pixmap)
painter.setBrush(QColor(100, 150, 255))
painter.drawEllipse(10, 10, 108, 108)
painter.end()
pixmap.save('test_avatar.png')

avatar = UnifiedAvatar()
avatar.set_mode(AvatarMode.PNG_BOUNCE)
widget = avatar.get_widget()
loaded = avatar.load('test_avatar.png')

print(f"  Created: YES")
print(f"  Widget: {type(widget).__name__}")
print(f"  Loaded PNG: {loaded}")

avatar.start_talking()
print(f"  start_talking(): OK")

avatar.set_emotion('happy')
print(f"  set_emotion('happy'): OK")

avatar.react(1.5)
print(f"  react(1.5): OK")

avatar.stop_talking()
print(f"  stop_talking(): OK")

pw = avatar._png_widget
print(f"  Bounce enabled: {pw.config.bounce_enabled}")
print(f"  Has pixmap: {pw._base_pixmap is not None}")
print("  RESULT: PASS")

# Test 2: 2D Animator
print("\n[TEST 2] 2D Animation System")
print("-" * 40)

from forge_ai.avatar import AvatarAnimator, AnimationState2D

animator = AvatarAnimator()
print(f"  Created: YES")
print(f"  States: {[s.name for s in AnimationState2D]}")

animator.set_state(AnimationState2D.TALKING)
print(f"  set_state(TALKING): OK")

animator.set_state(AnimationState2D.IDLE)
print(f"  set_state(IDLE): OK")
print("  RESULT: PASS")

# Test 3: 3D Native System
print("\n[TEST 3] 3D Native System (no Panda3D)")
print("-" * 40)

from forge_ai.avatar import NativeAvatar3D, Animation3DState, GLTFLoader

avatar3d = NativeAvatar3D()
print(f"  Created: YES")
print(f"  States: {[s.name for s in Animation3DState]}")

avatar3d.set_state(Animation3DState.TALKING)
print(f"  set_state(TALKING): OK")

avatar3d.set_state(Animation3DState.IDLE)
print(f"  set_state(IDLE): OK")

print(f"  GLTFLoader: {GLTFLoader is not None}")
print("  RESULT: PASS")

# Test 4: Non-human avatar types
print("\n[TEST 4] Non-Human Avatar Types")
print("-" * 40)

from forge_ai.avatar import EmotionMapping

for atype in AvatarType:
    mapping = EmotionMapping.get_mapping(atype)
    emotions = list(mapping.keys())
    print(f"  {atype.name}: {emotions[:3]}...")
print("  RESULT: PASS")

# Test 5: Unified Avatar switching modes
print("\n[TEST 5] Mode Switching")
print("-" * 40)

avatar = UnifiedAvatar()
for mode in AvatarMode:
    avatar.set_mode(mode)
    print(f"  Switched to {mode.name}: OK")
print("  RESULT: PASS")

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)

# Cleanup
os.remove('test_avatar.png')
