"""
Bone Test Window - Popup for testing avatar skeletal controls.

Opens from Avatar tab via "Bone Test" button.
Controls individual bones on rigged 3D models.
"""

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QSlider, QComboBox, QLineEdit, QTextEdit,
    QTabWidget, QWidget, QApplication
)


class BoneTestWindow(QDialog):
    """Popup window for testing avatar bone controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Avatar Bone Control")
        self.setMinimumSize(750, 650)
        self.setModal(False)
        
        # Store reference to main window's 3D widget
        self._main_window = parent
        self._widget_3d = getattr(parent, 'avatar_preview_3d', None) if parent else None
        
        # Current bone
        self._current_bone = ""
        
        self._build_ui()
        self._refresh_bones()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        tabs = QTabWidget()
        
        # === TAB 1: Bone Control ===
        controls_tab = QWidget()
        controls_layout = QVBoxLayout(controls_tab)
        
        # Status
        self.status_label = QLabel("Loading skeleton...")
        self.status_label.setStyleSheet("color: #cba6f7; font-weight: bold; padding: 5px;")
        controls_layout.addWidget(self.status_label)
        
        # Bone selector
        bone_group = QGroupBox("Select Bone")
        bone_layout = QVBoxLayout(bone_group)
        
        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("Bone:"))
        self.bone_combo = QComboBox()
        self.bone_combo.setMinimumWidth(300)
        self.bone_combo.currentTextChanged.connect(self._on_bone_selected)
        select_row.addWidget(self.bone_combo, stretch=1)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_bones)
        refresh_btn.setStyleSheet("background: #89b4fa; color: #1e1e2e;")
        select_row.addWidget(refresh_btn)
        bone_layout.addLayout(select_row)
        
        controls_layout.addWidget(bone_group)
        
        # Rotation controls
        rot_group = QGroupBox("Bone Rotation")
        rot_layout = QVBoxLayout(rot_group)
        
        for axis, name, default_range in [
            ("pitch", "Pitch (X - Nod)", (-90, 90)),
            ("yaw", "Yaw (Y - Turn)", (-90, 90)), 
            ("roll", "Roll (Z - Tilt)", (-45, 45))
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{name}:"))
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-180, 180)
            slider.setValue(0)
            slider.valueChanged.connect(lambda v, a=axis: self._on_bone_slider(a, v))
            setattr(self, f"{axis}_slider", slider)
            row.addWidget(slider, stretch=1)
            
            label = QLabel("0 deg")
            label.setFixedWidth(60)
            setattr(self, f"{axis}_label", label)
            row.addWidget(label)
            
            rot_layout.addLayout(row)
        
        controls_layout.addWidget(rot_group)
        
        # Quick movement buttons (direct numeric control)
        quick_group = QGroupBox("Quick Moves (Selected Bone)")
        quick_layout = QVBoxLayout(quick_group)
        
        # Pitch row
        pitch_row = QHBoxLayout()
        pitch_row.addWidget(QLabel("Pitch:"))
        for angle in [-30, -15, 15, 30]:
            btn = QPushButton(f"{angle:+d}")
            color = "#f38ba8" if angle < 0 else "#a6e3a1"
            btn.setStyleSheet(f"background: {color}; color: #1e1e2e; font-weight: bold; padding: 6px 12px;")
            btn.clicked.connect(lambda _, a=angle: self._quick_move("pitch", a))
            pitch_row.addWidget(btn)
        quick_layout.addLayout(pitch_row)
        
        # Yaw row
        yaw_row = QHBoxLayout()
        yaw_row.addWidget(QLabel("Yaw:"))
        for angle in [-30, -15, 15, 30]:
            btn = QPushButton(f"{angle:+d}")
            color = "#f38ba8" if angle < 0 else "#a6e3a1"
            btn.setStyleSheet(f"background: {color}; color: #1e1e2e; font-weight: bold; padding: 6px 12px;")
            btn.clicked.connect(lambda _, a=angle: self._quick_move("yaw", a))
            yaw_row.addWidget(btn)
        quick_layout.addLayout(yaw_row)
        
        # Roll row
        roll_row = QHBoxLayout()
        roll_row.addWidget(QLabel("Roll:"))
        for angle in [-30, -15, 15, 30]:
            btn = QPushButton(f"{angle:+d}")
            color = "#f38ba8" if angle < 0 else "#a6e3a1"
            btn.setStyleSheet(f"background: {color}; color: #1e1e2e; font-weight: bold; padding: 6px 12px;")
            btn.clicked.connect(lambda _, a=angle: self._quick_move("roll", a))
            roll_row.addWidget(btn)
        quick_layout.addLayout(roll_row)
        
        controls_layout.addWidget(quick_group)
        
        # Command input
        cmd_group = QGroupBox("Direct Control")
        cmd_layout = QVBoxLayout(cmd_group)
        
        cmd_row = QHBoxLayout()
        self.cmd_input = QLineEdit()
        self.cmd_input.setPlaceholderText("head pitch 15, arm yaw -30, reset")
        self.cmd_input.setStyleSheet("""
            QLineEdit {
                background: #313244;
                color: #cdd6f4;
                border: 2px solid #45475a;
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #89b4fa;
            }
        """)
        self.cmd_input.returnPressed.connect(self._execute_command)
        cmd_row.addWidget(self.cmd_input, stretch=1)
        
        exec_btn = QPushButton("Move")
        exec_btn.setStyleSheet("background: #a6e3a1; color: #1e1e2e; font-weight: bold; padding: 8px;")
        exec_btn.clicked.connect(self._execute_command)
        cmd_row.addWidget(exec_btn)
        
        reset_btn = QPushButton("Reset")
        reset_btn.setStyleSheet("background: #f38ba8; color: #1e1e2e; font-weight: bold; padding: 8px;")
        reset_btn.clicked.connect(self._reset_all_bones)
        cmd_row.addWidget(reset_btn)
        
        cmd_layout.addLayout(cmd_row)
        
        hint_label = QLabel("Format: [bone] [axis] [angle] - Example: head pitch 15")
        hint_label.setStyleSheet("color: #6c7086; font-size: 10px;")
        cmd_layout.addWidget(hint_label)
        
        controls_layout.addWidget(cmd_group)
        
        # Log
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background: #1e1e2e;
                color: #a6e3a1;
                font-family: monospace;
                font-size: 10px;
            }
        """)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        controls_layout.addWidget(log_group)
        
        tabs.addTab(controls_tab, "Bone Control")
        
        # === TAB 2: Skeleton Info ===
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("""
            QTextEdit {
                background: #1e1e2e;
                color: #cdd6f4;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        info_layout.addWidget(self.info_text)
        
        copy_btn = QPushButton("Copy Bone List")
        copy_btn.setStyleSheet("background: #cba6f7; color: #1e1e2e; font-weight: bold;")
        copy_btn.clicked.connect(self._copy_bone_list)
        info_layout.addWidget(copy_btn)
        
        tabs.addTab(info_tab, "Skeleton Info")
        
        # === TAB 3: Command Reference ===
        ref_tab = QWidget()
        ref_layout = QVBoxLayout(ref_tab)
        
        ref_text = QTextEdit()
        ref_text.setReadOnly(True)
        ref_text.setStyleSheet("""
            QTextEdit {
                background: #1e1e2e;
                color: #cdd6f4;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        ref_text.setPlainText("""=== AI BONE CONTROL SYNTAX ===

The AI can embed movement commands in responses:

[MOVE: bone_name=axis=angle]
[MOVE: head=pitch=-15]
[MOVE: right_arm=pitch=45, right_arm=yaw=30]

=== AXES ===
pitch - Tilt forward/back (like nodding)
yaw   - Turn left/right (like shaking head)
roll  - Lean sideways

=== ANGLES ===
Values in degrees: -180 to 180
Positive = forward/right/clockwise
Negative = backward/left/counterclockwise

=== EXAMPLES ===

# Nod (look down then up)
[MOVE: head=pitch=20]
[MOVE: head=pitch=-10]
[MOVE: head=pitch=0]

# Wave
[MOVE: right_upper_arm=pitch=-60]
[MOVE: right_forearm=pitch=45]

# Turn head to look left
[MOVE: head=yaw=30]
[MOVE: neck=yaw=15]

=== STREAMING ===
Multiple commands in conversation:

"Hi there! [MOVE: head=pitch=-10] 
*waves* [MOVE: right_arm=pitch=-60]
Nice to meet you! [MOVE: head=yaw=15]"
""")
        ref_layout.addWidget(ref_text)
        
        tabs.addTab(ref_tab, "Command Reference")
        
        layout.addWidget(tabs)
        
        # Close button
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setMinimumWidth(100)
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)
    
    def _refresh_bones(self):
        """Refresh bone list from loaded model."""
        self.bone_combo.clear()
        
        if not self._widget_3d:
            self.status_label.setText("No 3D widget - load a model first")
            self.status_label.setStyleSheet("color: #f38ba8;")
            return
        
        # Get bone names from the widget's skeleton
        bone_names = []
        skeleton_info = ""
        
        if hasattr(self._widget_3d, 'get_skeleton_info'):
            info = self._widget_3d.get_skeleton_info()
            if info.get('skeleton_loaded'):
                bones = info.get('available_bones', [])
                for b in bones:
                    if isinstance(b, dict):
                        bone_names.append(b.get('name', ''))
                    else:
                        bone_names.append(str(b))
                
                # Check if skinning can work (vertex counts must match)
                has_weights = info.get('has_weights', False)
                num_weights = info.get('num_weights', 0)
                num_verts = info.get('num_vertices', 0)
                
                if has_weights and num_weights == num_verts:
                    skeleton_info = f"Skeleton: {len(bone_names)} bones (skinning active)"
                    self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
                elif has_weights:
                    skeleton_info = f"Skeleton: {len(bone_names)} bones (vertex mismatch: {num_weights} vs {num_verts})"
                    self.status_label.setStyleSheet("color: #f9e2af; font-weight: bold;")
                else:
                    skeleton_info = f"Skeleton: {len(bone_names)} bones (no weights)"
                    self.status_label.setStyleSheet("color: #89b4fa; font-weight: bold;")
            else:
                skeleton_info = "No skeleton in model"
                self.status_label.setStyleSheet("color: #f38ba8;")
        else:
            skeleton_info = "Widget doesn't support skeleton"
            self.status_label.setStyleSheet("color: #f38ba8;")
        
        self.status_label.setText(skeleton_info)
        
        # Populate combo
        for name in bone_names:
            self.bone_combo.addItem(name)
        
        # Update info tab
        self._update_skeleton_info(bone_names)
        
        self.log_text.append(f"Found {len(bone_names)} bones")
    
    def _update_skeleton_info(self, bone_names: list):
        """Update the skeleton info tab."""
        if not bone_names:
            self.info_text.setPlainText("No skeleton loaded.\n\nLoad a rigged 3D model (GLTF/GLB) to see bones.")
            return
        
        lines = [f"=== SKELETON ({len(bone_names)} bones) ===\n"]
        
        # Group bones by type
        head_bones = []
        arm_bones = []
        leg_bones = []
        spine_bones = []
        other_bones = []
        
        for name in bone_names:
            nl = name.lower()
            if any(x in nl for x in ['head', 'neck', 'jaw', 'eye']):
                head_bones.append(name)
            elif any(x in nl for x in ['arm', 'hand', 'finger', 'thumb', 'shoulder', 'elbow', 'wrist']):
                arm_bones.append(name)
            elif any(x in nl for x in ['leg', 'foot', 'toe', 'hip', 'knee', 'ankle', 'thigh']):
                leg_bones.append(name)
            elif any(x in nl for x in ['spine', 'chest', 'pelvis', 'torso', 'body']):
                spine_bones.append(name)
            else:
                other_bones.append(name)
        
        if head_bones:
            lines.append("HEAD/NECK:")
            for b in head_bones:
                lines.append(f"  {b}")
            lines.append("")
        
        if spine_bones:
            lines.append("SPINE/TORSO:")
            for b in spine_bones:
                lines.append(f"  {b}")
            lines.append("")
        
        if arm_bones:
            lines.append("ARMS/HANDS:")
            for b in arm_bones:
                lines.append(f"  {b}")
            lines.append("")
        
        if leg_bones:
            lines.append("LEGS/FEET:")
            for b in leg_bones:
                lines.append(f"  {b}")
            lines.append("")
        
        if other_bones:
            lines.append("OTHER:")
            for b in other_bones:
                lines.append(f"  {b}")
        
        self.info_text.setPlainText("\n".join(lines))
    
    def _on_bone_selected(self, bone_name: str):
        """Handle bone selection."""
        self._current_bone = bone_name
        # Reset sliders
        self.pitch_slider.blockSignals(True)
        self.yaw_slider.blockSignals(True)
        self.roll_slider.blockSignals(True)
        self.pitch_slider.setValue(0)
        self.yaw_slider.setValue(0)
        self.roll_slider.setValue(0)
        self.pitch_slider.blockSignals(False)
        self.yaw_slider.blockSignals(False)
        self.roll_slider.blockSignals(False)
        
        self.pitch_label.setText("0 deg")
        self.yaw_label.setText("0 deg")
        self.roll_label.setText("0 deg")
        
        if bone_name:
            self.log_text.append(f"Selected: {bone_name}")
    
    def _quick_move(self, axis: str, angle: int):
        """Apply quick movement to selected bone."""
        if not self._current_bone:
            self.log_text.append("Select a bone first")
            return
        
        if not self._widget_3d or not hasattr(self._widget_3d, 'move_bone'):
            self.log_text.append("No 3D widget")
            return
        
        kwargs = {axis: angle}
        success = self._widget_3d.move_bone(self._current_bone, **kwargs)
        if success:
            self.log_text.append(f"{self._current_bone} {axis}={angle}")
            # Update slider to match
            slider = getattr(self, f"{axis}_slider", None)
            if slider:
                slider.blockSignals(True)
                slider.setValue(angle)
                slider.blockSignals(False)
            label = getattr(self, f"{axis}_label", None)
            if label:
                label.setText(f"{angle} deg")
        else:
            self.log_text.append(f"Failed: {self._current_bone}")
    
    def _on_bone_slider(self, axis: str, value: int):
        """Handle bone rotation slider change."""
        label = getattr(self, f"{axis}_label", None)
        if label:
            label.setText(f"{value} deg")
        
        if not self._current_bone:
            return
        
        if not self._widget_3d:
            return
        
        # Call the widget's move_bone method
        if hasattr(self._widget_3d, 'move_bone'):
            # Get current values for all axes
            pitch = self.pitch_slider.value()
            yaw = self.yaw_slider.value()
            roll = self.roll_slider.value()
            
            success = self._widget_3d.move_bone(self._current_bone, pitch=pitch, yaw=yaw, roll=roll)
            if success:
                self.log_text.append(f"{self._current_bone}: {axis}={value}")
            else:
                self.log_text.append(f"Failed to move {self._current_bone}")
    
    def _execute_command(self):
        """Execute a typed command: [bone] [axis] [angle] or reset."""
        cmd = self.cmd_input.text().strip().lower()
        if not cmd:
            return
        
        self.cmd_input.clear()
        self.log_text.append(f"> {cmd}")
        
        # Handle reset
        if cmd in ("reset", "reset all"):
            self._reset_all_bones()
            return
        
        # Parse: <bone> <axis> <angle>
        parts = cmd.split()
        if len(parts) < 3:
            self.log_text.append("Format: [bone] [axis] [angle]")
            return
        
        bone_query = parts[0]
        axis = parts[1]
        try:
            angle = int(parts[2])
        except ValueError:
            self.log_text.append("Angle must be a number")
            return
        
        # Find matching bone
        info = self._widget_3d.get_skeleton_info() if self._widget_3d and hasattr(self._widget_3d, 'get_skeleton_info') else {}
        bones = info.get('available_bones', [])
        bone_names = [b.get('name', str(b)) if isinstance(b, dict) else str(b) for b in bones]
        
        target_bone = None
        for bn in bone_names:
            if bone_query in bn.lower():
                target_bone = bn
                break
        
        if not target_bone:
            self.log_text.append(f"No bone matching '{bone_query}'")
            return
        
        if axis not in ("pitch", "yaw", "roll"):
            self.log_text.append("Axis must be: pitch, yaw, or roll")
            return
        
        # Execute bone move
        if self._widget_3d and hasattr(self._widget_3d, 'move_bone'):
            kwargs = {axis: angle}
            success = self._widget_3d.move_bone(target_bone, **kwargs)
            if success:
                self.log_text.append(f"{target_bone} {axis}={angle}")
            else:
                self.log_text.append(f"Failed: {target_bone}")
    
    def _reset_all_bones(self):
        """Reset all bones to bind pose."""
        if self._widget_3d and hasattr(self._widget_3d, 'reset_pose'):
            self._widget_3d.reset_pose()
            self.log_text.append("Reset to bind pose")
        
        # Reset sliders
        self.pitch_slider.setValue(0)
        self.yaw_slider.setValue(0)
        self.roll_slider.setValue(0)
    
    def _copy_bone_list(self):
        """Copy bone list to clipboard."""
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(self.info_text.toPlainText())
            self.log_text.append("Copied to clipboard")


# Global instance
_test_window: Optional[BoneTestWindow] = None


def show_bone_test_window(parent=None):
    """Show the bone test window."""
    global _test_window
    if _test_window is None or not _test_window.isVisible():
        _test_window = BoneTestWindow(parent)
    _test_window.show()
    _test_window.raise_()
    _test_window._refresh_bones()
    return _test_window
