"""
Blender Bridge - Real-time avatar control in Blender

Allows ForgeAI to control 3D models in Blender in real-time:
- Send pose/bone transformations
- Control blend shapes/shape keys for expressions
- Animate lip sync
- Stream AI-driven movements

Two modes:
1. Blender Addon Mode: Install addon in Blender, connects via socket
2. Script Mode: Run Blender with --python flag

Requirements:
- Blender 3.0+ with Python 3.10+
- Socket connection on port 9876 (configurable)

Usage:
    # In ForgeAI
    from forge_ai.avatar.blender_bridge import BlenderBridge
    
    bridge = BlenderBridge()
    bridge.connect()
    
    # Send expression
    bridge.set_expression("happy")
    
    # Send pose
    bridge.set_bone_rotation("head", pitch=10, yaw=5)
    
    # Lip sync
    bridge.set_viseme("AA")  # Mouth shape for "ah"
"""

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from queue import Queue

from ..config import CONFIG


# Default connection settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9876


class BlenderCommand(Enum):
    """Commands sent to Blender."""
    # Connection
    PING = "ping"
    HANDSHAKE = "handshake"
    DISCONNECT = "disconnect"
    
    # Model
    LOAD_MODEL = "load_model"
    GET_MODEL_INFO = "get_model_info"
    LIST_BONES = "list_bones"
    LIST_SHAPE_KEYS = "list_shape_keys"
    
    # Pose
    SET_BONE_ROTATION = "set_bone_rotation"
    SET_BONE_POSITION = "set_bone_position"
    SET_BONE_SCALE = "set_bone_scale"
    RESET_POSE = "reset_pose"
    
    # Expressions (shape keys)
    SET_SHAPE_KEY = "set_shape_key"
    SET_EXPRESSION = "set_expression"
    RESET_EXPRESSION = "reset_expression"
    
    # Animation
    PLAY_ANIMATION = "play_animation"
    STOP_ANIMATION = "stop_animation"
    SET_FRAME = "set_frame"
    
    # Lip sync
    SET_VISEME = "set_viseme"
    
    # Rendering
    RENDER_FRAME = "render_frame"
    SET_CAMERA = "set_camera"


@dataclass
class BlenderModelInfo:
    """Information about the loaded Blender model."""
    name: str = ""
    filepath: str = ""
    bones: List[str] = field(default_factory=list)
    shape_keys: List[str] = field(default_factory=list)
    animations: List[str] = field(default_factory=list)
    has_armature: bool = False
    vertex_count: int = 0


@dataclass 
class BlenderBridgeConfig:
    """Configuration for Blender bridge."""
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    auto_reconnect: bool = True
    reconnect_interval: float = 5.0
    command_timeout: float = 5.0
    
    # Expression mappings (ForgeAI expression -> Blender shape keys)
    expression_mappings: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Viseme mappings (phoneme -> shape key weights)
    viseme_mappings: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Bone mappings (standard name -> model-specific name)
    bone_mappings: Dict[str, str] = field(default_factory=dict)


class BlenderBridge:
    """
    Bridge for real-time control of Blender avatars.
    
    Communicates with Blender via socket to:
    - Control bone rotations (head tracking, body pose)
    - Set shape key values (facial expressions)
    - Trigger animations
    - Sync lip movements with speech
    """
    
    def __init__(self, config: Optional[BlenderBridgeConfig] = None):
        self.config = config or BlenderBridgeConfig()
        
        self._socket: Optional[socket.socket] = None
        self._connected = False
        self._running = False
        
        self._receive_thread: Optional[threading.Thread] = None
        self._reconnect_thread: Optional[threading.Thread] = None
        
        self._command_queue: Queue = Queue()
        self._response_queue: Queue = Queue()
        self._pending_responses: Dict[str, Any] = {}
        
        self._model_info: Optional[BlenderModelInfo] = None
        self._current_expression = "neutral"
        self._current_pose: Dict[str, Dict[str, float]] = {}
        
        # Callbacks
        self._on_connected: List[Callable] = []
        self._on_disconnected: List[Callable] = []
        self._on_model_loaded: List[Callable] = []
        self._on_error: List[Callable] = []
        
        # Load default mappings
        self._load_default_mappings()
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def model_info(self) -> Optional[BlenderModelInfo]:
        return self._model_info
    
    def _load_default_mappings(self):
        """Load default expression and viseme mappings."""
        # Default expression to shape key mappings
        # These work with many VRM/VRoid models
        self.config.expression_mappings = {
            "neutral": {},
            "happy": {"Fcl_MTH_A": 0.3, "Fcl_EYE_Joy": 1.0},
            "sad": {"Fcl_MTH_U": 0.5, "Fcl_EYE_Sorrow": 1.0},
            "angry": {"Fcl_MTH_E": 0.4, "Fcl_EYE_Angry": 1.0},
            "surprised": {"Fcl_MTH_O": 0.8, "Fcl_EYE_Surprised": 1.0},
            "thinking": {"Fcl_MTH_U": 0.2, "Fcl_EYE_Close_L": 0.3},
            "excited": {"Fcl_MTH_A": 0.6, "Fcl_EYE_Joy": 0.8},
            "sleeping": {"Fcl_EYE_Close": 1.0},
            "winking": {"Fcl_EYE_Close_R": 1.0, "Fcl_MTH_A": 0.2},
            "love": {"Fcl_EYE_Heart": 1.0, "Fcl_MTH_A": 0.3},
        }
        
        # Viseme mappings for lip sync (standard visemes)
        self.config.viseme_mappings = {
            "sil": {},  # Silence
            "PP": {"Fcl_MTH_Close": 0.8},  # p, b, m
            "FF": {"Fcl_MTH_I": 0.4, "Fcl_MTH_Close": 0.3},  # f, v
            "TH": {"Fcl_MTH_I": 0.5},  # th
            "DD": {"Fcl_MTH_I": 0.3, "Fcl_MTH_A": 0.2},  # t, d
            "kk": {"Fcl_MTH_A": 0.3},  # k, g
            "CH": {"Fcl_MTH_I": 0.6},  # ch, j, sh
            "SS": {"Fcl_MTH_I": 0.5},  # s, z
            "nn": {"Fcl_MTH_Close": 0.4, "Fcl_MTH_A": 0.1},  # n, l
            "RR": {"Fcl_MTH_O": 0.3},  # r
            "AA": {"Fcl_MTH_A": 1.0},  # ah
            "E": {"Fcl_MTH_E": 0.8},  # eh
            "I": {"Fcl_MTH_I": 0.8},  # ee
            "O": {"Fcl_MTH_O": 0.9},  # oh
            "U": {"Fcl_MTH_U": 0.9},  # oo
        }
        
        # Standard bone mappings
        self.config.bone_mappings = {
            "head": "Head",
            "neck": "Neck",
            "spine": "Spine",
            "chest": "Chest",
            "hips": "Hips",
            "left_eye": "LeftEye",
            "right_eye": "RightEye",
            "left_shoulder": "LeftShoulder",
            "right_shoulder": "RightShoulder",
            "left_arm": "LeftUpperArm",
            "right_arm": "RightUpperArm",
            "left_forearm": "LeftLowerArm",
            "right_forearm": "RightLowerArm",
            "left_hand": "LeftHand",
            "right_hand": "RightHand",
        }
    
    def connect(self, host: Optional[str] = None, port: Optional[int] = None) -> bool:
        """
        Connect to Blender addon.
        
        Returns True if connected successfully.
        """
        host = host or self.config.host
        port = port or self.config.port
        
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.config.command_timeout)
            self._socket.connect((host, port))
            
            self._connected = True
            self._running = True
            
            # Start receive thread
            self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._receive_thread.start()
            
            # Handshake
            response = self._send_command(BlenderCommand.HANDSHAKE, {"client": "ForgeAI"})
            if response and response.get("status") == "ok":
                print(f"[BlenderBridge] Connected to Blender at {host}:{port}")
                
                # Get model info
                self._refresh_model_info()
                
                for cb in self._on_connected:
                    try:
                        cb()
                    except Exception as e:
                        print(f"[BlenderBridge] Callback error: {e}")
                
                return True
            else:
                self.disconnect()
                return False
                
        except Exception as e:
            print(f"[BlenderBridge] Connection failed: {e}")
            self._connected = False
            
            if self.config.auto_reconnect:
                self._start_reconnect()
            
            return False
    
    def disconnect(self):
        """Disconnect from Blender."""
        self._running = False
        
        if self._connected:
            try:
                self._send_command(BlenderCommand.DISCONNECT, {})
            except:
                pass
        
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
        
        self._connected = False
        
        for cb in self._on_disconnected:
            try:
                cb()
            except:
                pass
        
        print("[BlenderBridge] Disconnected")
    
    def _start_reconnect(self):
        """Start auto-reconnect thread."""
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return
        
        self._reconnect_thread = threading.Thread(target=self._reconnect_loop, daemon=True)
        self._reconnect_thread.start()
    
    def _reconnect_loop(self):
        """Try to reconnect periodically."""
        while not self._connected and self.config.auto_reconnect:
            time.sleep(self.config.reconnect_interval)
            if not self._connected:
                print("[BlenderBridge] Attempting to reconnect...")
                self.connect()
    
    def _receive_loop(self):
        """Receive messages from Blender."""
        buffer = ""
        
        while self._running and self._socket:
            try:
                data = self._socket.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                
                # Process complete messages (newline-delimited JSON)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            msg = json.loads(line)
                            self._handle_message(msg)
                        except json.JSONDecodeError:
                            pass
                            
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[BlenderBridge] Receive error: {e}")
                break
        
        if self._running:
            self._connected = False
            for cb in self._on_disconnected:
                try:
                    cb()
                except:
                    pass
            
            if self.config.auto_reconnect:
                self._start_reconnect()
    
    def _handle_message(self, msg: dict):
        """Handle received message."""
        msg_type = msg.get("type", "response")
        
        if msg_type == "response":
            # Response to a command
            cmd_id = msg.get("id")
            if cmd_id and cmd_id in self._pending_responses:
                self._pending_responses[cmd_id] = msg
        elif msg_type == "event":
            # Async event from Blender
            event = msg.get("event")
            if event == "model_loaded":
                self._refresh_model_info()
            elif event == "animation_complete":
                pass  # Could trigger callback
    
    def _send_command(self, cmd: BlenderCommand, data: dict, wait_response: bool = True) -> Optional[dict]:
        """Send command to Blender."""
        if not self._connected or not self._socket:
            return None
        
        import uuid
        cmd_id = str(uuid.uuid4())[:8]
        
        message = {
            "id": cmd_id,
            "command": cmd.value,
            "data": data
        }
        
        try:
            self._socket.send((json.dumps(message) + '\n').encode('utf-8'))
            
            if wait_response:
                # Wait for response
                self._pending_responses[cmd_id] = None
                timeout = time.time() + self.config.command_timeout
                
                while time.time() < timeout:
                    if self._pending_responses.get(cmd_id) is not None:
                        response = self._pending_responses.pop(cmd_id)
                        return response
                    time.sleep(0.01)
                
                self._pending_responses.pop(cmd_id, None)
                return None
            
            return {"status": "sent"}
            
        except Exception as e:
            print(f"[BlenderBridge] Send error: {e}")
            return None
    
    def _refresh_model_info(self):
        """Get current model info from Blender."""
        response = self._send_command(BlenderCommand.GET_MODEL_INFO, {})
        if response and response.get("status") == "ok":
            data = response.get("data", {})
            self._model_info = BlenderModelInfo(
                name=data.get("name", ""),
                filepath=data.get("filepath", ""),
                bones=data.get("bones", []),
                shape_keys=data.get("shape_keys", []),
                animations=data.get("animations", []),
                has_armature=data.get("has_armature", False),
                vertex_count=data.get("vertex_count", 0)
            )
            
            for cb in self._on_model_loaded:
                try:
                    cb(self._model_info)
                except:
                    pass
    
    # === Model Control ===
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a 3D model in Blender.
        
        Supports: .blend, .glb, .gltf, .fbx, .obj, .vrm
        """
        response = self._send_command(BlenderCommand.LOAD_MODEL, {"filepath": filepath})
        if response and response.get("status") == "ok":
            self._refresh_model_info()
            return True
        return False
    
    def get_bones(self) -> List[str]:
        """Get list of bones in current armature."""
        if self._model_info:
            return self._model_info.bones
        
        response = self._send_command(BlenderCommand.LIST_BONES, {})
        if response and response.get("status") == "ok":
            return response.get("data", {}).get("bones", [])
        return []
    
    def get_shape_keys(self) -> List[str]:
        """Get list of shape keys (blend shapes)."""
        if self._model_info:
            return self._model_info.shape_keys
        
        response = self._send_command(BlenderCommand.LIST_SHAPE_KEYS, {})
        if response and response.get("status") == "ok":
            return response.get("data", {}).get("shape_keys", [])
        return []
    
    # === Pose Control ===
    
    def set_bone_rotation(self, bone: str, pitch: float = 0, yaw: float = 0, roll: float = 0,
                          relative: bool = False) -> bool:
        """
        Set bone rotation.
        
        Args:
            bone: Bone name (or standard name from bone_mappings)
            pitch: X rotation in degrees
            yaw: Y rotation in degrees
            roll: Z rotation in degrees
            relative: If True, adds to current rotation
        """
        # Map standard name to model-specific name
        actual_bone = self.config.bone_mappings.get(bone, bone)
        
        response = self._send_command(BlenderCommand.SET_BONE_ROTATION, {
            "bone": actual_bone,
            "rotation": [pitch, yaw, roll],
            "relative": relative
        })
        
        if response and response.get("status") == "ok":
            self._current_pose[bone] = {"pitch": pitch, "yaw": yaw, "roll": roll}
            return True
        return False
    
    def set_bone_position(self, bone: str, x: float = 0, y: float = 0, z: float = 0,
                          relative: bool = False) -> bool:
        """Set bone position (for IK or root motion)."""
        actual_bone = self.config.bone_mappings.get(bone, bone)
        
        response = self._send_command(BlenderCommand.SET_BONE_POSITION, {
            "bone": actual_bone,
            "position": [x, y, z],
            "relative": relative
        })
        return response and response.get("status") == "ok"
    
    def reset_pose(self) -> bool:
        """Reset all bones to rest pose."""
        response = self._send_command(BlenderCommand.RESET_POSE, {})
        if response and response.get("status") == "ok":
            self._current_pose.clear()
            return True
        return False
    
    def look_at(self, x: float, y: float, screen_width: int = 1920, screen_height: int = 1080):
        """
        Make avatar look at a screen position.
        
        Converts screen coordinates to head/eye rotation.
        """
        # Convert screen position to rotation
        # Center of screen = 0,0 rotation
        center_x = screen_width / 2
        center_y = screen_height / 2
        
        # Calculate angles (max ~30 degrees)
        yaw = ((x - center_x) / center_x) * 30
        pitch = ((center_y - y) / center_y) * 20  # Inverted, limited
        
        # Set head rotation
        self.set_bone_rotation("head", pitch=pitch, yaw=yaw)
        
        # Subtle eye movement (more range)
        eye_yaw = yaw * 1.5
        eye_pitch = pitch * 1.2
        self.set_bone_rotation("left_eye", pitch=eye_pitch, yaw=eye_yaw)
        self.set_bone_rotation("right_eye", pitch=eye_pitch, yaw=eye_yaw)
    
    # === Expression Control ===
    
    def set_shape_key(self, name: str, value: float) -> bool:
        """
        Set a single shape key value.
        
        Args:
            name: Shape key name
            value: 0.0 to 1.0
        """
        response = self._send_command(BlenderCommand.SET_SHAPE_KEY, {
            "name": name,
            "value": max(0.0, min(1.0, value))
        })
        return response and response.get("status") == "ok"
    
    def set_expression(self, expression: str, intensity: float = 1.0) -> bool:
        """
        Set facial expression using predefined mappings.
        
        Args:
            expression: Expression name (happy, sad, angry, etc.)
            intensity: 0.0 to 1.0, scales all shape key values
        """
        # Reset previous expression
        if self._current_expression != expression:
            prev_mapping = self.config.expression_mappings.get(self._current_expression, {})
            for key in prev_mapping:
                self.set_shape_key(key, 0.0)
        
        # Apply new expression
        mapping = self.config.expression_mappings.get(expression, {})
        if mapping:
            for key, value in mapping.items():
                self.set_shape_key(key, value * intensity)
            self._current_expression = expression
            return True
        
        return False
    
    def reset_expression(self) -> bool:
        """Reset all expression shape keys to 0."""
        response = self._send_command(BlenderCommand.RESET_EXPRESSION, {})
        if response and response.get("status") == "ok":
            self._current_expression = "neutral"
            return True
        return False
    
    # === Lip Sync ===
    
    def set_viseme(self, viseme: str, intensity: float = 1.0) -> bool:
        """
        Set mouth shape for lip sync.
        
        Args:
            viseme: Viseme code (AA, E, I, O, U, PP, FF, etc.)
            intensity: 0.0 to 1.0
        """
        mapping = self.config.viseme_mappings.get(viseme.upper(), {})
        
        # Reset mouth shape keys first
        for v_mapping in self.config.viseme_mappings.values():
            for key in v_mapping:
                self.set_shape_key(key, 0.0)
        
        # Apply viseme
        for key, value in mapping.items():
            self.set_shape_key(key, value * intensity)
        
        return True
    
    def speak_text(self, text: str, duration: float = 0.1):
        """
        Simple text-to-viseme for basic lip sync.
        
        For better results, use phoneme data from TTS.
        """
        # Simple vowel detection for basic lip sync
        vowel_map = {
            'a': 'AA', 'e': 'E', 'i': 'I', 'o': 'O', 'u': 'U',
            'p': 'PP', 'b': 'PP', 'm': 'PP',
            'f': 'FF', 'v': 'FF',
            's': 'SS', 'z': 'SS',
        }
        
        def _animate():
            for char in text.lower():
                viseme = vowel_map.get(char, 'sil')
                self.set_viseme(viseme)
                time.sleep(duration)
            self.set_viseme('sil')
        
        threading.Thread(target=_animate, daemon=True).start()
    
    # === Animation ===
    
    def play_animation(self, name: str, loop: bool = False) -> bool:
        """Play a named animation."""
        response = self._send_command(BlenderCommand.PLAY_ANIMATION, {
            "name": name,
            "loop": loop
        })
        return response and response.get("status") == "ok"
    
    def stop_animation(self) -> bool:
        """Stop current animation."""
        response = self._send_command(BlenderCommand.STOP_ANIMATION, {})
        return response and response.get("status") == "ok"
    
    # === Callbacks ===
    
    def on_connected(self, callback: Callable):
        """Register callback for connection."""
        self._on_connected.append(callback)
    
    def on_disconnected(self, callback: Callable):
        """Register callback for disconnection."""
        self._on_disconnected.append(callback)
    
    def on_model_loaded(self, callback: Callable):
        """Register callback for model load."""
        self._on_model_loaded.append(callback)


# === Blender Addon Code (save as addon) ===

BLENDER_ADDON_CODE = '''
"""
ForgeAI Blender Addon

Install this in Blender: Edit > Preferences > Add-ons > Install
Then enable "ForgeAI Avatar Bridge"

Creates a socket server that ForgeAI connects to for real-time control.
"""

bl_info = {
    "name": "ForgeAI Avatar Bridge",
    "author": "ForgeAI",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > ForgeAI",
    "description": "Real-time avatar control from ForgeAI",
    "category": "Animation",
}

import bpy
import json
import socket
import threading
import mathutils
from math import radians

# Server state
_server_socket = None
_client_socket = None
_server_thread = None
_running = False
_port = 9876


def get_armature():
    """Get active armature."""
    obj = bpy.context.active_object
    if obj and obj.type == 'ARMATURE':
        return obj
    # Find first armature
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


def get_mesh_with_shape_keys():
    """Get mesh with shape keys."""
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.data.shape_keys:
            return obj
    return None


def handle_command(cmd_data):
    """Handle a command from ForgeAI."""
    command = cmd_data.get("command", "")
    data = cmd_data.get("data", {})
    cmd_id = cmd_data.get("id", "")
    
    result = {"id": cmd_id, "type": "response", "status": "ok", "data": {}}
    
    try:
        if command == "ping":
            result["data"] = {"pong": True}
            
        elif command == "handshake":
            result["data"] = {"server": "Blender", "version": bpy.app.version_string}
            
        elif command == "get_model_info":
            armature = get_armature()
            mesh = get_mesh_with_shape_keys()
            
            bones = []
            if armature:
                bones = [b.name for b in armature.pose.bones]
            
            shape_keys = []
            if mesh and mesh.data.shape_keys:
                shape_keys = [k.name for k in mesh.data.shape_keys.key_blocks]
            
            result["data"] = {
                "name": armature.name if armature else "",
                "bones": bones,
                "shape_keys": shape_keys,
                "has_armature": armature is not None,
                "animations": [a.name for a in bpy.data.actions] if bpy.data.actions else []
            }
            
        elif command == "list_bones":
            armature = get_armature()
            if armature:
                result["data"]["bones"] = [b.name for b in armature.pose.bones]
                
        elif command == "list_shape_keys":
            mesh = get_mesh_with_shape_keys()
            if mesh and mesh.data.shape_keys:
                result["data"]["shape_keys"] = [k.name for k in mesh.data.shape_keys.key_blocks]
                
        elif command == "set_bone_rotation":
            armature = get_armature()
            if armature:
                bone_name = data.get("bone", "")
                rotation = data.get("rotation", [0, 0, 0])
                relative = data.get("relative", False)
                
                if bone_name in armature.pose.bones:
                    bone = armature.pose.bones[bone_name]
                    rot = mathutils.Euler((radians(rotation[0]), radians(rotation[1]), radians(rotation[2])))
                    
                    if relative:
                        bone.rotation_euler.x += rot.x
                        bone.rotation_euler.y += rot.y
                        bone.rotation_euler.z += rot.z
                    else:
                        bone.rotation_mode = 'XYZ'
                        bone.rotation_euler = rot
                else:
                    result["status"] = "error"
                    result["error"] = f"Bone not found: {bone_name}"
                    
        elif command == "set_shape_key":
            mesh = get_mesh_with_shape_keys()
            if mesh and mesh.data.shape_keys:
                key_name = data.get("name", "")
                value = data.get("value", 0.0)
                
                key_blocks = mesh.data.shape_keys.key_blocks
                if key_name in key_blocks:
                    key_blocks[key_name].value = value
                else:
                    result["status"] = "error"
                    result["error"] = f"Shape key not found: {key_name}"
                    
        elif command == "reset_pose":
            armature = get_armature()
            if armature:
                for bone in armature.pose.bones:
                    bone.rotation_euler = (0, 0, 0)
                    bone.rotation_quaternion = (1, 0, 0, 0)
                    bone.location = (0, 0, 0)
                    
        elif command == "reset_expression":
            mesh = get_mesh_with_shape_keys()
            if mesh and mesh.data.shape_keys:
                for key in mesh.data.shape_keys.key_blocks:
                    if key.name != "Basis":
                        key.value = 0.0
                        
        elif command == "play_animation":
            anim_name = data.get("name", "")
            loop = data.get("loop", False)
            
            if anim_name in bpy.data.actions:
                armature = get_armature()
                if armature:
                    if not armature.animation_data:
                        armature.animation_data_create()
                    armature.animation_data.action = bpy.data.actions[anim_name]
                    bpy.context.scene.frame_set(1)
            else:
                result["status"] = "error"
                result["error"] = f"Animation not found: {anim_name}"
                
        elif command == "load_model":
            filepath = data.get("filepath", "")
            if filepath.endswith(".glb") or filepath.endswith(".gltf"):
                bpy.ops.import_scene.gltf(filepath=filepath)
            elif filepath.endswith(".fbx"):
                bpy.ops.import_scene.fbx(filepath=filepath)
            elif filepath.endswith(".obj"):
                bpy.ops.wm.obj_import(filepath=filepath)
            elif filepath.endswith(".blend"):
                with bpy.data.libraries.load(filepath) as (data_from, data_to):
                    data_to.objects = data_from.objects
                for obj in data_to.objects:
                    if obj is not None:
                        bpy.context.collection.objects.link(obj)
                        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def server_thread():
    """Server thread that accepts connections."""
    global _server_socket, _client_socket, _running
    
    _server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _server_socket.bind(("127.0.0.1", _port))
    _server_socket.listen(1)
    _server_socket.settimeout(1.0)
    
    print(f"[ForgeAI] Server listening on port {_port}")
    
    while _running:
        try:
            _client_socket, addr = _server_socket.accept()
            print(f"[ForgeAI] Client connected: {addr}")
            
            buffer = ""
            while _running and _client_socket:
                try:
                    data = _client_socket.recv(4096).decode('utf-8')
                    if not data:
                        break
                    
                    buffer += data
                    while '\\n' in buffer:
                        line, buffer = buffer.split('\\n', 1)
                        if line.strip():
                            try:
                                cmd = json.loads(line)
                                result = handle_command(cmd)
                                _client_socket.send((json.dumps(result) + '\\n').encode('utf-8'))
                            except json.JSONDecodeError:
                                pass
                                
                except socket.timeout:
                    continue
                except:
                    break
                    
            if _client_socket:
                _client_socket.close()
                _client_socket = None
            print("[ForgeAI] Client disconnected")
            
        except socket.timeout:
            continue
        except:
            break
    
    if _server_socket:
        _server_socket.close()
    print("[ForgeAI] Server stopped")


class FORGEAI_OT_start_server(bpy.types.Operator):
    bl_idname = "forgeai.start_server"
    bl_label = "Start Server"
    bl_description = "Start ForgeAI connection server"
    
    def execute(self, context):
        global _server_thread, _running
        
        if _running:
            self.report({'WARNING'}, "Server already running")
            return {'CANCELLED'}
        
        _running = True
        _server_thread = threading.Thread(target=server_thread, daemon=True)
        _server_thread.start()
        
        self.report({'INFO'}, f"Server started on port {_port}")
        return {'FINISHED'}


class FORGEAI_OT_stop_server(bpy.types.Operator):
    bl_idname = "forgeai.stop_server"
    bl_label = "Stop Server"
    bl_description = "Stop ForgeAI connection server"
    
    def execute(self, context):
        global _running, _client_socket, _server_socket
        
        _running = False
        
        if _client_socket:
            try:
                _client_socket.close()
            except:
                pass
        
        self.report({'INFO'}, "Server stopped")
        return {'FINISHED'}


class FORGEAI_PT_panel(bpy.types.Panel):
    bl_label = "ForgeAI Avatar Bridge"
    bl_idname = "FORGEAI_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'ForgeAI'
    
    def draw(self, context):
        layout = self.layout
        
        layout.label(text=f"Port: {_port}")
        
        if _running:
            layout.label(text="Status: Running", icon='CHECKMARK')
            layout.operator("forgeai.stop_server", icon='PAUSE')
        else:
            layout.label(text="Status: Stopped", icon='X')
            layout.operator("forgeai.start_server", icon='PLAY')
        
        layout.separator()
        layout.label(text="Model Info:")
        
        armature = get_armature()
        if armature:
            layout.label(text=f"  Armature: {armature.name}")
            layout.label(text=f"  Bones: {len(armature.pose.bones)}")
        else:
            layout.label(text="  No armature found")
        
        mesh = get_mesh_with_shape_keys()
        if mesh and mesh.data.shape_keys:
            layout.label(text=f"  Shape Keys: {len(mesh.data.shape_keys.key_blocks)}")


classes = [
    FORGEAI_OT_start_server,
    FORGEAI_OT_stop_server,
    FORGEAI_PT_panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    global _running
    _running = False
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
'''


def save_blender_addon(output_dir: Optional[str] = None) -> str:
    """
    Save the Blender addon to a file.
    
    Returns the path to the saved addon.
    """
    if output_dir is None:
        output_dir = Path(CONFIG.get("data_dir", "data")) / "blender"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    addon_path = output_dir / "forgeai_blender_addon.py"
    
    with open(addon_path, 'w') as f:
        f.write(BLENDER_ADDON_CODE)
    
    print(f"[BlenderBridge] Addon saved to: {addon_path}")
    print("Install in Blender: Edit > Preferences > Add-ons > Install")
    
    return str(addon_path)


# Singleton instance
_bridge_instance: Optional[BlenderBridge] = None


def get_blender_bridge() -> BlenderBridge:
    """Get the singleton BlenderBridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = BlenderBridge()
    return _bridge_instance
