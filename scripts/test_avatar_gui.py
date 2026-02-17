"""Quick avatar test - standalone window with direct bone controls."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QComboBox, QGroupBox
)

from enigma_engine.avatar.animation_3d_native import NativeAvatar3D
from enigma_engine.avatar.bone_control import get_bone_controller


class AvatarTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Avatar Bone Test")
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget
        main = QWidget()
        self.setCentralWidget(main)
        layout = QHBoxLayout(main)
        
        # Left: Avatar 3D view
        self.avatar = NativeAvatar3D()
        avatar_widget = self.avatar.get_widget()
        avatar_widget.setMinimumSize(500, 500)
        layout.addWidget(avatar_widget, stretch=2)
        
        # Right: Controls
        controls = QWidget()
        ctrl_layout = QVBoxLayout(controls)
        layout.addWidget(controls, stretch=1)
        
        # Load model button
        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model)
        ctrl_layout.addWidget(load_btn)
        
        # Bone selector
        bone_group = QGroupBox("Bone Control")
        bone_layout = QVBoxLayout(bone_group)
        
        self.bone_combo = QComboBox()
        bone_layout.addWidget(QLabel("Select Bone:"))
        bone_layout.addWidget(self.bone_combo)
        
        # Pitch slider
        bone_layout.addWidget(QLabel("Pitch:"))
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setRange(-90, 90)
        self.pitch_slider.setValue(0)
        self.pitch_slider.valueChanged.connect(self.on_pitch_change)
        bone_layout.addWidget(self.pitch_slider)
        self.pitch_label = QLabel("0")
        bone_layout.addWidget(self.pitch_label)
        
        # Yaw slider  
        bone_layout.addWidget(QLabel("Yaw:"))
        self.yaw_slider = QSlider(Qt.Horizontal)
        self.yaw_slider.setRange(-90, 90)
        self.yaw_slider.setValue(0)
        self.yaw_slider.valueChanged.connect(self.on_yaw_change)
        bone_layout.addWidget(self.yaw_slider)
        self.yaw_label = QLabel("0")
        bone_layout.addWidget(self.yaw_label)
        
        ctrl_layout.addWidget(bone_group)
        
        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        wave_btn = QPushButton("Wave")
        wave_btn.clicked.connect(self.action_wave)
        actions_layout.addWidget(wave_btn)
        
        nod_btn = QPushButton("Nod")
        nod_btn.clicked.connect(self.action_nod)
        actions_layout.addWidget(nod_btn)
        
        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self.action_reset)
        actions_layout.addWidget(reset_btn)
        
        ctrl_layout.addWidget(actions_group)
        ctrl_layout.addStretch()
        
        # Status
        self.status = QLabel("Ready - Click 'Load Model' to start")
        ctrl_layout.addWidget(self.status)
        
        # Bone controller reference
        self.bone_ctrl = None
        
        # Animation timer
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.update_animation)
        self.anim_step = 0
        
    def load_model(self):
        """Load the default avatar model."""
        # Try to find a model
        model_paths = [
            Path("models/avatars/character.glb"),
            Path("models/avatars/scene.gltf"),
            Path("assets/avatars/default.glb"),
        ]
        
        for path in model_paths:
            if path.exists():
                self.status.setText(f"Loading {path.name}...")
                QApplication.processEvents()
                
                self.avatar.load_model(str(path))
                self.bone_ctrl = get_bone_controller()
                
                # Get bone names
                info = self.bone_ctrl.get_bone_info_for_ai()
                bones = info.get("available_bones", [])
                
                # Handle if bones are dicts or strings
                bone_names = []
                for b in bones:
                    if isinstance(b, dict):
                        bone_names.append(b.get("name", str(b)))
                    else:
                        bone_names.append(str(b))
                
                self.bone_combo.clear()
                self.bone_combo.addItems(bone_names)
                
                self.status.setText(f"Loaded! {len(bone_names)} bones available")
                return
        
        self.status.setText("No model found - put a .glb in models/avatars/")
    
    def on_pitch_change(self, value):
        self.pitch_label.setText(str(value))
        self._apply_bone()
    
    def on_yaw_change(self, value):
        self.yaw_label.setText(str(value))
        self._apply_bone()
    
    def _apply_bone(self):
        if not self.bone_ctrl:
            return
        bone = self.bone_combo.currentText()
        if bone:
            self.bone_ctrl.move_bone(
                bone,
                pitch=self.pitch_slider.value(),
                yaw=self.yaw_slider.value()
            )
    
    def action_wave(self):
        """Animate a wave."""
        if not self.bone_ctrl:
            self.status.setText("Load model first!")
            return
        
        self.status.setText("Waving...")
        self.anim_step = 0
        self.anim_timer.start(50)  # 20fps
        self.current_anim = "wave"
    
    def action_nod(self):
        """Animate a nod."""
        if not self.bone_ctrl:
            self.status.setText("Load model first!")
            return
        
        self.status.setText("Nodding...")
        self.anim_step = 0
        self.anim_timer.start(50)
        self.current_anim = "nod"
    
    def action_reset(self):
        """Reset all bones to neutral."""
        if not self.bone_ctrl:
            return
        
        info = self.bone_ctrl.get_bone_info_for_ai()
        for bone in info.get("available_bones", []):
            self.bone_ctrl.move_bone(bone, pitch=0, yaw=0, roll=0)
        
        self.pitch_slider.setValue(0)
        self.yaw_slider.setValue(0)
        self.status.setText("Reset complete")
    
    def update_animation(self):
        """Animation frame update."""
        self.anim_step += 1
        
        if self.current_anim == "wave":
            # Find arm bones
            info = self.bone_ctrl.get_bone_info_for_ai()
            bones = info.get("available_bones", [])
            
            arm_bone = None
            for b in bones:
                if "arm" in b.lower() and ("r" in b.lower() or "right" in b.lower()):
                    arm_bone = b
                    break
            if not arm_bone and bones:
                arm_bone = bones[0]  # fallback
            
            if arm_bone:
                # Wave motion: up-down oscillation
                import math
                angle = 45 + 30 * math.sin(self.anim_step * 0.3)
                self.bone_ctrl.move_bone(arm_bone, pitch=angle)
            
            if self.anim_step > 40:  # ~2 seconds
                self.anim_timer.stop()
                self.action_reset()
                self.status.setText("Wave complete!")
        
        elif self.current_anim == "nod":
            # Find head bone
            info = self.bone_ctrl.get_bone_info_for_ai()
            bones = info.get("available_bones", [])
            
            head_bone = None
            for b in bones:
                if "head" in b.lower() or "neck" in b.lower():
                    head_bone = b
                    break
            if not head_bone and bones:
                head_bone = bones[0]
            
            if head_bone:
                # Nod motion
                import math
                angle = 15 * math.sin(self.anim_step * 0.4)
                self.bone_ctrl.move_bone(head_bone, pitch=angle)
            
            if self.anim_step > 30:
                self.anim_timer.stop()
                self.action_reset()
                self.status.setText("Nod complete!")


def main():
    app = QApplication(sys.argv)
    window = AvatarTestWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
