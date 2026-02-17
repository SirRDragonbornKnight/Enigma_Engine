#!/usr/bin/env python3
"""
Standalone Desktop Avatar with Terminal Control

Usage:
    python scripts/avatar.py [model_path]
    
Commands via JSON stdin, responses via JSON stdout.
"""

import json
import sys
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt5.QtCore import pyqtSignal, QObject, QTimer
from PyQt5.QtWidgets import QApplication


class CommandSignal(QObject):
    """Bridge between stdin thread and Qt main thread."""
    command = pyqtSignal(str)


class AvatarAPI:
    """JSON API for avatar control."""
    
    def __init__(self, overlay):
        self.overlay = overlay
        self.gl = overlay._gl_widget
        self.physics_objects = {}
        self.animations = {}
        self.scene_objects = {}
        
        # Physics timer
        self._physics_timer = QTimer()
        self._physics_timer.timeout.connect(self._update_physics)
        self._physics_enabled = False
        
        # Animation timer
        self._anim_timer = QTimer()
        self._anim_timer.timeout.connect(self._update_animations)
    
    @property
    def skeleton(self):
        return getattr(self.gl, '_skeleton', None)
    
    def handle(self, cmd: dict) -> dict:
        """Handle a command, return response."""
        action = cmd.get('cmd', '')
        
        try:
            # Discovery
            if action == 'info':
                return self._cmd_info()
            if action == 'bones':
                return self._cmd_bones()
            if action == 'bone_info':
                return self._cmd_bone_info(cmd.get('name'))
            
            # Bone control
            if action == 'set':
                return self._cmd_set(cmd)
            if action == 'set_multi':
                return self._cmd_set_multi(cmd.get('bones', {}))
            if action == 'add':
                return self._cmd_add(cmd)
            if action == 'get':
                return self._cmd_get(cmd.get('bone'))
            if action == 'reset':
                return self._cmd_reset(cmd.get('bone'))
            
            # Physics
            if action == 'physics_add':
                return self._cmd_physics_add(cmd)
            if action == 'physics_remove':
                return self._cmd_physics_remove(cmd.get('bone'))
            if action == 'physics_impulse':
                return self._cmd_physics_impulse(cmd)
            if action == 'physics_list':
                return self._cmd_physics_list()
            
            # Animation
            if action == 'anim_create':
                return self._cmd_anim_create(cmd)
            if action == 'anim_keyframe':
                return self._cmd_anim_keyframe(cmd)
            if action == 'anim_play':
                return self._cmd_anim_play(cmd)
            if action == 'anim_stop':
                return self._cmd_anim_stop(cmd.get('name'))
            if action == 'anim_list':
                return {'animations': list(self.animations.keys())}
            
            # State
            if action == 'pose_export':
                return self._cmd_pose_export()
            if action == 'pose_import':
                return self._cmd_pose_import(cmd.get('pose', {}))
            if action == 'pose_save':
                return self._cmd_pose_save(cmd.get('path'))
            if action == 'pose_load':
                return self._cmd_pose_load(cmd.get('path'))
            
            # Camera
            if action == 'camera':
                return self._cmd_camera(cmd)
            if action == 'camera_reset':
                self.gl.reset_view()
                return {'ok': True}
            
            # Window
            if action == 'window_move':
                self.overlay.move(cmd.get('x', 100), cmd.get('y', 100))
                return {'ok': True}
            if action == 'window_size':
                size = cmd.get('size', 250)
                self.overlay.setFixedSize(size, size)
                return {'ok': True}
            if action == 'quit':
                QApplication.quit()
                return {'ok': True}
            
            return {'error': f'Unknown command: {action}'}
        except Exception as e:
            return {'error': str(e)}
    
    # === Discovery ===
    
    def _cmd_info(self) -> dict:
        """Get full model info."""
        info = {
            'model': getattr(self.gl, 'model_name', 'unknown'),
            'path': getattr(self.gl, '_model_path', ''),
            'vertices': len(self.gl.vertices) if self.gl.vertices is not None else 0,
            'faces': len(self.gl.faces) if self.gl.faces is not None else 0,
        }
        
        if self.skeleton:
            info['bone_count'] = len(self.skeleton.joints)
            info['bones'] = [
                {'name': j['name'], 'parent_idx': j.get('parent_idx', -1)}
                for j in self.skeleton.joints
            ]
            info['has_weights'] = self.skeleton.vertex_weights is not None
        else:
            info['bone_count'] = 0
            info['bones'] = []
            info['has_weights'] = False
        
        return info
    
    def _cmd_bones(self) -> dict:
        """List all bone names."""
        if not self.skeleton:
            return {'bones': [], 'error': 'No skeleton loaded'}
        return {'bones': self.skeleton.joint_names}
    
    def _cmd_bone_info(self, name: str) -> dict:
        """Get info about a specific bone."""
        if not self.skeleton:
            return {'error': 'No skeleton loaded'}
        if not name:
            return {'error': 'No bone name provided'}
        
        # Try exact match first, then fuzzy
        matched = self._find_bone(name)
        if not matched:
            return {'error': f'Bone not found: {name}'}
        
        current = self.skeleton.bone_transforms.get(matched, {'pitch': 0, 'yaw': 0, 'roll': 0})
        
        # Find parent
        idx = self.skeleton.joint_map.get(matched, -1)
        parent_idx = self.skeleton.joints[idx].get('parent_idx', -1) if idx >= 0 else -1
        parent_name = self.skeleton.joint_names[parent_idx] if 0 <= parent_idx < len(self.skeleton.joint_names) else None
        
        return {
            'name': matched,
            'matched_from': name,
            'parent': parent_name,
            'current': current,
        }
    
    def _find_bone(self, name: str) -> str:
        """Find bone - exact match first, then skeleton's fuzzy match."""
        if not self.skeleton:
            return None
        
        # Exact match
        if name in self.skeleton.joint_names:
            return name
        
        # Use skeleton's fuzzy matching
        return self.skeleton._find_bone(name)
    
    # === Bone Control ===
    
    def _cmd_set(self, cmd: dict) -> dict:
        """Set bone rotation."""
        if not self.skeleton:
            return {'error': 'No skeleton loaded'}
        
        bone = cmd.get('bone')
        matched = self._find_bone(bone)
        if not matched:
            return {'error': f'Bone not found: {bone}'}
        
        self.skeleton.bone_transforms[matched] = {
            'pitch': cmd.get('pitch', 0),
            'yaw': cmd.get('yaw', 0),
            'roll': cmd.get('roll', 0),
        }
        self._apply_and_redraw()
        return {'ok': True, 'bone': matched}
    
    def _cmd_set_multi(self, bones: dict) -> dict:
        """Set multiple bones at once."""
        if not self.skeleton:
            return {'error': 'No skeleton loaded'}
        
        set_bones = []
        for name, rotations in bones.items():
            matched = self._find_bone(name)
            if matched:
                self.skeleton.bone_transforms[matched] = {
                    'pitch': rotations.get('pitch', 0),
                    'yaw': rotations.get('yaw', 0),
                    'roll': rotations.get('roll', 0),
                }
                set_bones.append(matched)
        
        self._apply_and_redraw()
        return {'ok': True, 'bones': set_bones}
    
    def _cmd_add(self, cmd: dict) -> dict:
        """Add to current bone rotation."""
        if not self.skeleton:
            return {'error': 'No skeleton loaded'}
        
        bone = cmd.get('bone')
        matched = self._find_bone(bone)
        if not matched:
            return {'error': f'Bone not found: {bone}'}
        
        current = self.skeleton.bone_transforms.get(matched, {'pitch': 0, 'yaw': 0, 'roll': 0})
        self.skeleton.bone_transforms[matched] = {
            'pitch': current.get('pitch', 0) + cmd.get('pitch', 0),
            'yaw': current.get('yaw', 0) + cmd.get('yaw', 0),
            'roll': current.get('roll', 0) + cmd.get('roll', 0),
        }
        self._apply_and_redraw()
        return {'ok': True, 'bone': matched, 'current': self.skeleton.bone_transforms[matched]}
    
    def _cmd_get(self, bone: str) -> dict:
        """Get current bone rotation."""
        if not self.skeleton:
            return {'error': 'No skeleton loaded'}
        
        matched = self._find_bone(bone)
        if not matched:
            return {'error': f'Bone not found: {bone}'}
        
        current = self.skeleton.bone_transforms.get(matched, {'pitch': 0, 'yaw': 0, 'roll': 0})
        return {'bone': matched, **current}
    
    def _cmd_reset(self, bone: str = None) -> dict:
        """Reset bones to bind pose."""
        if not self.skeleton:
            return {'error': 'No skeleton loaded'}
        
        if bone:
            matched = self._find_bone(bone)
            if matched:
                self.skeleton.bone_transforms[matched] = {'pitch': 0, 'yaw': 0, 'roll': 0}
        else:
            self.skeleton.reset_pose()
        
        self._apply_and_redraw()
        return {'ok': True}
    
    # === Physics ===
    
    def _cmd_physics_add(self, cmd: dict) -> dict:
        """Add physics to a bone."""
        bone = cmd.get('bone')
        ptype = cmd.get('type', 'jiggle')
        params = cmd.get('params', {})
        
        self.physics_objects[bone] = {
            'type': ptype,
            'params': params,
            'velocity': [0.0, 0.0, 0.0],
            'target': [0.0, 0.0, 0.0],
        }
        
        if not self._physics_enabled:
            self._physics_enabled = True
            self._physics_timer.start(16)
        
        return {'ok': True, 'bone': bone, 'type': ptype}
    
    def _cmd_physics_remove(self, bone: str) -> dict:
        """Remove physics from a bone."""
        if bone in self.physics_objects:
            del self.physics_objects[bone]
            if not self.physics_objects:
                self._physics_timer.stop()
                self._physics_enabled = False
            return {'ok': True}
        return {'error': f'No physics on: {bone}'}
    
    def _cmd_physics_impulse(self, cmd: dict) -> dict:
        """Apply impulse to a bone with physics."""
        bone = cmd.get('bone')
        force = cmd.get('force', [0, 0, 0])
        if bone in self.physics_objects:
            for i in range(3):
                self.physics_objects[bone]['velocity'][i] += force[i]
            return {'ok': True}
        return {'error': f'No physics on: {bone}'}
    
    def _cmd_physics_list(self) -> dict:
        """List all physics objects."""
        return {'physics': [{'bone': k, 'type': v['type'], 'params': v['params']} 
                           for k, v in self.physics_objects.items()]}
    
    def _update_physics(self):
        """Update physics simulation."""
        if not self.skeleton or not self.physics_objects:
            return
        
        dt = 0.016
        changed = False
        
        for bone_name, phys in self.physics_objects.items():
            matched = self._find_bone(bone_name)
            if not matched:
                continue
            
            current = self.skeleton.bone_transforms.get(matched, {'pitch': 0, 'yaw': 0, 'roll': 0})
            params = phys['params']
            
            stiffness = params.get('stiffness', 0.5)
            damping = params.get('damping', 0.3)
            gravity = params.get('gravity', 0)
            
            for i, axis in enumerate(['pitch', 'yaw', 'roll']):
                target = phys['target'][i]
                pos = current.get(axis, 0)
                vel = phys['velocity'][i]
                
                force = (target - pos) * stiffness * 100
                if axis == 'pitch':
                    force += gravity * 50
                
                vel += force * dt
                vel *= (1 - damping)
                pos += vel * dt
                
                phys['velocity'][i] = vel
                current[axis] = pos
            
            self.skeleton.bone_transforms[matched] = current
            changed = True
        
        if changed:
            self._apply_and_redraw()
    
    # === Animation ===
    
    def _cmd_anim_create(self, cmd: dict) -> dict:
        """Create a new animation."""
        name = cmd.get('name')
        self.animations[name] = {
            'duration': cmd.get('duration', 1.0),
            'loop': cmd.get('loop', False),
            'keyframes': [],
            'playing': False,
            'time': 0,
            'speed': 1.0,
        }
        return {'ok': True, 'name': name}
    
    def _cmd_anim_keyframe(self, cmd: dict) -> dict:
        """Add keyframe to animation."""
        name = cmd.get('name')
        if name not in self.animations:
            return {'error': f'Animation not found: {name}'}
        
        self.animations[name]['keyframes'].append({
            'time': cmd.get('time', 0),
            'bones': cmd.get('bones', {}),
        })
        self.animations[name]['keyframes'].sort(key=lambda k: k['time'])
        return {'ok': True}
    
    def _cmd_anim_play(self, cmd: dict) -> dict:
        """Play animation."""
        name = cmd.get('name')
        if name not in self.animations:
            return {'error': f'Animation not found: {name}'}
        
        anim = self.animations[name]
        anim['playing'] = True
        anim['time'] = 0
        anim['speed'] = cmd.get('speed', 1.0)
        
        if not self._anim_timer.isActive():
            self._anim_timer.start(16)
        
        return {'ok': True, 'name': name}
    
    def _cmd_anim_stop(self, name: str) -> dict:
        """Stop animation."""
        if name in self.animations:
            self.animations[name]['playing'] = False
            return {'ok': True}
        return {'error': f'Animation not found: {name}'}
    
    def _update_animations(self):
        """Update playing animations."""
        dt = 0.016
        any_playing = False
        
        for name, anim in self.animations.items():
            if not anim['playing']:
                continue
            any_playing = True
            
            anim['time'] += dt * anim['speed']
            
            if anim['time'] >= anim['duration']:
                if anim['loop']:
                    anim['time'] = 0
                else:
                    anim['playing'] = False
                    continue
            
            keyframes = anim['keyframes']
            if not keyframes:
                continue
            
            # Find surrounding keyframes
            prev_kf = keyframes[0]
            next_kf = keyframes[-1]
            for kf in keyframes:
                if kf['time'] <= anim['time']:
                    prev_kf = kf
                if kf['time'] >= anim['time']:
                    next_kf = kf
                    break
            
            # Interpolate
            if prev_kf['time'] == next_kf['time']:
                t = 0
            else:
                t = (anim['time'] - prev_kf['time']) / (next_kf['time'] - prev_kf['time'])
            
            # Apply interpolated pose
            for bone_name in set(prev_kf['bones'].keys()) | set(next_kf['bones'].keys()):
                matched = self._find_bone(bone_name)
                if not matched:
                    continue
                
                prev_rot = prev_kf['bones'].get(bone_name, {'pitch': 0, 'yaw': 0, 'roll': 0})
                next_rot = next_kf['bones'].get(bone_name, {'pitch': 0, 'yaw': 0, 'roll': 0})
                
                self.skeleton.bone_transforms[matched] = {
                    'pitch': prev_rot.get('pitch', 0) * (1-t) + next_rot.get('pitch', 0) * t,
                    'yaw': prev_rot.get('yaw', 0) * (1-t) + next_rot.get('yaw', 0) * t,
                    'roll': prev_rot.get('roll', 0) * (1-t) + next_rot.get('roll', 0) * t,
                }
            
            self._apply_and_redraw()
        
        if not any_playing:
            self._anim_timer.stop()
    
    # === State ===
    
    def _cmd_pose_export(self) -> dict:
        """Export current pose."""
        if not self.skeleton:
            return {'error': 'No skeleton loaded'}
        
        bones = {}
        for name in self.skeleton.joint_names:
            t = self.skeleton.bone_transforms.get(name, {})
            if t.get('pitch', 0) != 0 or t.get('yaw', 0) != 0 or t.get('roll', 0) != 0:
                bones[name] = dict(t)
        return {'bones': bones}
    
    def _cmd_pose_import(self, pose: dict) -> dict:
        """Import a pose."""
        bones = pose.get('bones', {})
        return self._cmd_set_multi(bones)
    
    def _cmd_pose_save(self, path: str) -> dict:
        """Save pose to file."""
        pose = self._cmd_pose_export()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(pose, f, indent=2)
        return {'ok': True, 'path': path}
    
    def _cmd_pose_load(self, path: str) -> dict:
        """Load pose from file."""
        try:
            with open(path) as f:
                pose = json.load(f)
            return self._cmd_pose_import(pose)
        except Exception as e:
            return {'error': str(e)}
    
    # === Camera ===
    
    def _cmd_camera(self, cmd: dict) -> dict:
        """Set camera parameters."""
        if 'rotation_x' in cmd:
            self.gl.rotation_x = cmd['rotation_x']
        if 'rotation_y' in cmd:
            self.gl.rotation_y = cmd['rotation_y']
        if 'zoom' in cmd:
            self.gl.zoom = cmd['zoom']
        if 'pan_x' in cmd:
            self.gl.pan_x = cmd['pan_x']
        if 'pan_y' in cmd:
            self.gl.pan_y = cmd['pan_y']
        self.gl.update()
        return {'ok': True}
    
    # === Helpers ===
    
    def _apply_and_redraw(self):
        """Apply skinning and redraw."""
        if not self.skeleton or not self.gl:
            return
        
        if self.skeleton.vertex_weights is not None and self.gl.vertices is not None:
            if self.skeleton.bind_vertices is None:
                self.skeleton.set_bind_pose(self.gl.vertices.copy(), 
                                           self.gl.normals.copy() if self.gl.normals is not None else None)
            self.gl.vertices = self.skeleton.apply_skinning(self.skeleton.bind_vertices)
        
        self.gl.update()


def find_model() -> str:
    """Find first available avatar model."""
    search_dirs = [
        Path("models/avatars"),
        Path("data/avatar/models"),
    ]
    for d in search_dirs:
        if d.exists():
            for f in d.rglob("*.glb"):
                return str(f)
            for f in d.rglob("*.gltf"):
                return str(f)
    return None


def stdin_reader(signal: CommandSignal):
    """Read JSON commands from stdin."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if line:
                signal.command.emit(line)
        except:
            break


def main():
    # Get model path
    model_path = sys.argv[1] if len(sys.argv) > 1 else find_model()
    if not model_path:
        print(json.dumps({'error': 'No model found. Usage: python scripts/avatar.py <model.glb>'}))
        return 1
    
    app = QApplication(sys.argv)
    
    # Import here to avoid circular imports
    from enigma_engine.gui.tabs.avatar.avatar_display import Avatar3DOverlayWindow
    
    # Create overlay
    overlay = Avatar3DOverlayWindow()
    overlay.load_model(model_path)
    overlay.show()
    
    # Create API
    api = AvatarAPI(overlay)
    signal = CommandSignal()
    
    def handle_command(line):
        try:
            cmd = json.loads(line)
            response = api.handle(cmd)
        except json.JSONDecodeError:
            response = {'error': 'Invalid JSON'}
        print(json.dumps(response), flush=True)
    
    signal.command.connect(handle_command)
    
    # Print ready message
    info = api._cmd_info()
    print(json.dumps({'ready': True, **info}), flush=True)
    
    # Start stdin reader
    thread = threading.Thread(target=stdin_reader, args=(signal,), daemon=True)
    thread.start()
    
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
