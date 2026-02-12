"""
3D Avatar Animation System (Local)

Real-time 3D skeletal animation using Panda3D - completely local, no cloud.

Features:
- Load FBX, glTF, or Panda3D .egg/.bam models with skeletal animation
- AI-controlled animation states (idle, talking, gestures)
- Render to PyQt widget for desktop overlay
- Transparent background support
- Dynamic camera, lighting, posing

Supported Formats:
- FBX (via fbx2egg converter or panda3d-gltf)
- glTF/GLB (via panda3d-gltf)
- Panda3D native: .egg, .bam

Usage:
    from enigma_engine.avatar.animation_3d import Avatar3DAnimator, AI3DAvatarController
    
    # Create animator
    animator = Avatar3DAnimator()
    
    # Load model with animations
    animator.load_model("data/avatar/models/character.glb")
    
    # Or load animations separately
    animator.load_animation("idle", "data/avatar/animations/idle.glb")
    animator.load_animation("talking", "data/avatar/animations/talking.glb")
    
    # Get PyQt widget to embed
    widget = animator.get_qt_widget()
    layout.addWidget(widget)
    
    # AI controls
    controller = AI3DAvatarController(animator)
    controller.start_talking()
    controller.gesture("wave")

Exporting from Unreal Engine:
    1. Select your character with skeleton
    2. File > Export > FBX
    3. Enable "Export Animations" and select animation sequences
    4. Import FBX to Blender, export as glTF (better compatibility)
    Or use Unreal's native glTF exporter plugin
"""

import os
import queue
import threading
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

# Check for Panda3D
try:
    from direct.actor.Actor import Actor
    from direct.showbase.ShowBase import ShowBase
    from direct.task import Task
    from panda3d.core import (
        AmbientLight,
        DirectionalLight,
        FrameBufferProperties,
        GraphicsOutput,
        GraphicsPipe,
        LColor,
        NodePath,
        Texture,
        TransparencyAttrib,
        WindowProperties,
        loadPrcFileData,
    )
    HAS_PANDA3D = True
except ImportError:
    HAS_PANDA3D = False
    print("[3D Animator] Panda3D not installed. Install with: pip install panda3d panda3d-gltf")

# PyQt integration
try:
    from PyQt5.QtCore import QObject, QTimer, pyqtSignal
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QObject = object
    pyqtSignal = lambda *args: None


class Animation3DState(Enum):
    """3D Avatar animation states."""
    IDLE = auto()
    TALKING = auto()
    LISTENING = auto()
    THINKING = auto()
    HAPPY = auto()
    SAD = auto()
    SURPRISED = auto()
    GESTURE = auto()


@dataclass
class Animation3DClip:
    """A single animation clip."""
    name: str
    duration: float = 0.0
    loop: bool = True
    blend_in: float = 0.2  # Blend time in seconds
    blend_out: float = 0.2


class Avatar3DAnimator:
    """
    Real-time 3D avatar animator using Panda3D.
    
    Renders 3D character with skeletal animation to a texture,
    which can be displayed in a PyQt widget.
    """
    
    def __init__(self, width: int = 512, height: int = 512):
        if not HAS_PANDA3D:
            raise ImportError("Panda3D required. Install with: pip install panda3d panda3d-gltf")
        
        self.width = width
        self.height = height
        
        # Animation storage
        self._animations: dict[str, Animation3DClip] = {}
        self._state_animations: dict[Animation3DState, str] = {}
        
        # State
        self._current_state = Animation3DState.IDLE
        self._current_animation: Optional[str] = None
        self._gesture_queue: list[str] = []
        self._return_state = Animation3DState.IDLE
        
        # Panda3D components (initialized in separate thread)
        self._base: Optional[ShowBase] = None
        self._actor: Optional[Actor] = None
        self._render_texture: Optional[Texture] = None
        self._camera: Optional[NodePath] = None
        
        # Threading
        self._panda_thread: Optional[threading.Thread] = None
        self._command_queue: queue.Queue = queue.Queue()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._running = False
        
        # Callbacks
        self._on_animation_complete: Optional[Callable] = None
        
        # PyQt widget
        self._qt_widget: Optional[QWidget] = None
        self._frame_timer: Optional[QTimer] = None
        
    def start(self):
        """Start the 3D rendering engine in background thread."""
        if self._running:
            return
            
        self._running = True
        self._panda_thread = threading.Thread(target=self._panda_main, daemon=True)
        self._panda_thread.start()
        
        # Start frame update timer for Qt
        if HAS_PYQT and self._qt_widget:
            self._frame_timer = QTimer()
            self._frame_timer.timeout.connect(self._update_qt_frame)
            self._frame_timer.start(33)  # ~30 FPS
    
    def stop(self):
        """Stop the 3D rendering engine."""
        self._running = False
        self._command_queue.put(("shutdown", None))
        if self._panda_thread:
            self._panda_thread.join(timeout=2.0)
        if self._frame_timer:
            self._frame_timer.stop()
    
    def _panda_main(self):
        """Main Panda3D loop running in separate thread."""
        # Configure Panda3D for offscreen rendering
        loadPrcFileData("", """
            window-type offscreen
            audio-library-name null
            load-display pandagl
            framebuffer-alpha true
            background-color 0 0 0 0
            want-pstats false
        """)
        
        # Try to load glTF support
        try:
            import panda3d_gltf
            panda3d_gltf.patch_loader()
        except ImportError:
            print("[3D Animator] panda3d-gltf not installed. glTF support limited.")
        
        # Create ShowBase
        self._base = ShowBase()
        self._base.disableMouse()
        
        # Disable signal handling (doesn't work in threads)
        self._base.taskMgr.setupTaskChain('default', numThreads=0, tickClock=False,
                                           threadPriority=None, frameBudget=-1,
                                           frameSync=False, timeslicePriority=False)
        
        # Create offscreen buffer for rendering
        fb_props = FrameBufferProperties()
        fb_props.setRgbaBits(8, 8, 8, 8)
        fb_props.setDepthBits(24)
        
        win_props = WindowProperties.size(self.width, self.height)
        
        self._render_buffer = self._base.graphicsEngine.makeOutput(
            self._base.pipe,
            "offscreen",
            -100,
            fb_props,
            win_props,
            GraphicsPipe.BFRefuseWindow,
            self._base.win.getGsg(),
            self._base.win
        )
        
        # Create texture to render to
        self._render_texture = Texture()
        self._render_buffer.addRenderTexture(
            self._render_texture,
            GraphicsOutput.RTMCopyRam,
            GraphicsOutput.RTPColor
        )
        
        # Set up camera for the buffer
        self._camera = self._base.makeCamera(self._render_buffer)
        self._camera.reparentTo(self._base.render)
        self._camera.setPos(0, -5, 1)  # Default position
        self._camera.lookAt(0, 0, 1)   # Look at character center
        
        # Set up lighting
        self._setup_lighting()
        
        # Add task to process commands and capture frames
        self._base.taskMgr.add(self._process_commands_task, "process_commands")
        self._base.taskMgr.add(self._capture_frame_task, "capture_frame")
        
        # Run Panda3D loop manually to avoid signal handler issues in threads
        try:
            while self._running:
                self._base.taskMgr.step()
                self._base.graphicsEngine.renderFrame()
        except (SystemExit, Exception) as e:
            if self._running:
                print(f"[3D Animator] Loop error: {e}")
    
    def _setup_lighting(self):
        """Set up default lighting."""
        # Ambient light
        ambient = AmbientLight("ambient")
        ambient.setColor(LColor(0.4, 0.4, 0.4, 1))
        ambient_np = self._base.render.attachNewNode(ambient)
        self._base.render.setLight(ambient_np)
        
        # Key light
        key = DirectionalLight("key")
        key.setColor(LColor(0.8, 0.8, 0.8, 1))
        key_np = self._base.render.attachNewNode(key)
        key_np.setHpr(-45, -45, 0)
        self._base.render.setLight(key_np)
        
        # Fill light
        fill = DirectionalLight("fill")
        fill.setColor(LColor(0.3, 0.3, 0.4, 1))
        fill_np = self._base.render.attachNewNode(fill)
        fill_np.setHpr(45, -30, 0)
        self._base.render.setLight(fill_np)
    
    def _process_commands_task(self, task):
        """Process commands from main thread."""
        while not self._command_queue.empty():
            try:
                cmd, args = self._command_queue.get_nowait()
                
                if cmd == "shutdown":
                    self._base.userExit()
                    return Task.done
                    
                elif cmd == "load_model":
                    self._do_load_model(args)
                    
                elif cmd == "play_animation":
                    self._do_play_animation(args)
                    
                elif cmd == "set_camera":
                    pos, look_at = args
                    self._camera.setPos(*pos)
                    self._camera.lookAt(*look_at)
                    
            except queue.Empty:
                break
        
        return Task.cont
    
    def _capture_frame_task(self, task):
        """Capture rendered frame to queue for Qt."""
        if not self._running:
            return Task.done
        
        # Get frame from texture
        if self._render_texture and self._render_texture.hasRamImage():
            # Convert to QImage-compatible format
            data = self._render_texture.getRamImageAs("RGBA")
            if data:
                try:
                    self._frame_queue.put_nowait((
                        bytes(data),
                        self._render_texture.getXSize(),
                        self._render_texture.getYSize()
                    ))
                except queue.Full:
                    pass  # Skip frame if queue full
        
        return Task.cont
    
    def _do_load_model(self, path: str):
        """Load model in Panda3D thread."""
        try:
            # Remove existing actor
            if self._actor:
                self._actor.cleanup()
                self._actor.removeNode()
            
            # Load as Actor for animation support
            self._actor = Actor(path)
            self._actor.reparentTo(self._base.render)
            self._actor.setTransparency(TransparencyAttrib.MAlpha)
            
            # Get available animations from the model
            anims = self._actor.getAnimNames()
            for anim_name in anims:
                duration = self._actor.getDuration(anim_name)
                self._animations[anim_name] = Animation3DClip(
                    name=anim_name,
                    duration=duration,
                    loop=True
                )
            
            print(f"[3D Animator] Loaded model with animations: {anims}")
            
            # Auto-frame the model
            bounds = self._actor.getTightBounds()
            if bounds:
                min_pt, max_pt = bounds
                center = (min_pt + max_pt) / 2
                size = (max_pt - min_pt).length()
                self._camera.setPos(center.x, center.y - size * 2, center.z)
                self._camera.lookAt(center)
                
        except Exception as e:
            print(f"[3D Animator] Failed to load model: {e}")
    
    def _do_play_animation(self, args: tuple[str, bool, float]):
        """Play animation in Panda3D thread."""
        anim_name, loop, blend = args
        
        if not self._actor:
            return
        
        if anim_name in self._animations:
            if loop:
                self._actor.loop(anim_name, blendTime=blend)
            else:
                self._actor.play(anim_name, blendTime=blend)
                # Set up callback for when animation completes
                duration = self._animations[anim_name].duration
                self._base.taskMgr.doMethodLater(
                    duration, 
                    self._on_anim_done, 
                    "anim_done",
                    extraArgs=[anim_name]
                )
            self._current_animation = anim_name
    
    def _on_anim_done(self, anim_name: str, task=None):
        """Called when a non-looping animation completes."""
        # Process gesture queue or return to state
        if self._gesture_queue:
            next_gesture = self._gesture_queue.pop(0)
            self.play_gesture(next_gesture, self._return_state)
        elif self._current_state == Animation3DState.GESTURE:
            self.set_state(self._return_state)
        
        if self._on_animation_complete:
            self._on_animation_complete(anim_name)
        
        return Task.done
    
    def _update_qt_frame(self):
        """Update Qt widget with latest frame."""
        try:
            data, width, height = self._frame_queue.get_nowait()
            
            # Create QImage from RGBA data
            qimg = QImage(data, width, height, width * 4, QImage.Format_RGBA8888)
            # Flip vertically (OpenGL vs Qt coordinate system)
            qimg = qimg.mirrored(False, True)
            
            pixmap = QPixmap.fromImage(qimg)
            
            if self._qt_widget:
                label = self._qt_widget.findChild(QLabel, "frame_label")
                if label:
                    label.setPixmap(pixmap)
                    
        except queue.Empty:
            pass  # Intentionally silent
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def load_model(self, path: str) -> bool:
        """
        Load a 3D model with animations.
        
        Supported formats:
        - glTF/GLB (recommended - has animations embedded)
        - FBX (requires conversion or panda3d-fbx)
        - Panda3D native: .egg, .bam
        
        Args:
            path: Path to model file
            
        Returns:
            True if command was queued
        """
        path = str(Path(path).resolve())
        if not os.path.exists(path):
            print(f"[3D Animator] Model not found: {path}")
            return False
        
        self._command_queue.put(("load_model", path))
        return True
    
    def load_animation(self, name: str, path: str, loop: bool = True) -> bool:
        """
        Load an animation from a separate file.
        
        Useful when animations are stored separately from the base model.
        
        Args:
            name: Animation name
            path: Path to animation file
            loop: Whether animation loops
        """
        # For separate animation files, we'd need to bind them to the actor
        # This is more complex - for now, recommend embedding anims in model
        clip = Animation3DClip(name=name, loop=loop)
        self._animations[name] = clip
        return True
    
    def map_state_to_animation(self, state: Animation3DState, animation_name: str):
        """Map an animation state to a loaded animation."""
        self._state_animations[state] = animation_name
    
    def set_state(self, state: Animation3DState):
        """Set the current animation state (AI control)."""
        if state == self._current_state and state != Animation3DState.GESTURE:
            return
        
        self._current_state = state
        
        anim_name = self._state_animations.get(state)
        if anim_name and anim_name in self._animations:
            clip = self._animations[anim_name]
            self._command_queue.put(("play_animation", (anim_name, clip.loop, clip.blend_in)))
        else:
            # Fallback to idle
            idle_name = self._state_animations.get(Animation3DState.IDLE)
            if idle_name:
                self._command_queue.put(("play_animation", (idle_name, True, 0.2)))
    
    def play_gesture(self, gesture_name: str, then_state: Animation3DState = Animation3DState.IDLE):
        """Play a one-shot gesture, then return to state."""
        if gesture_name not in self._animations:
            print(f"[3D Animator] Unknown gesture: {gesture_name}")
            return
        
        self._return_state = then_state
        self._current_state = Animation3DState.GESTURE
        
        clip = self._animations[gesture_name]
        self._command_queue.put(("play_animation", (gesture_name, False, clip.blend_in)))
    
    def queue_gesture(self, gesture_name: str):
        """Queue a gesture to play after current animation."""
        if gesture_name in self._animations:
            self._gesture_queue.append(gesture_name)
    
    def set_camera(self, position: tuple[float, float, float], 
                   look_at: tuple[float, float, float] = (0, 0, 1)):
        """Set camera position and target."""
        self._command_queue.put(("set_camera", (position, look_at)))
    
    def get_qt_widget(self) -> QWidget:
        """Get a PyQt widget that displays the 3D render."""
        if not HAS_PYQT:
            raise ImportError("PyQt5 required for Qt widget")
        
        if not self._qt_widget:
            self._qt_widget = QWidget()
            layout = QVBoxLayout(self._qt_widget)
            layout.setContentsMargins(0, 0, 0, 0)
            
            label = QLabel()
            label.setObjectName("frame_label")
            label.setFixedSize(self.width, self.height)
            label.setStyleSheet("background: transparent;")
            layout.addWidget(label)
            
            # Start frame timer if engine is running
            if self._running and not self._frame_timer:
                self._frame_timer = QTimer()
                self._frame_timer.timeout.connect(self._update_qt_frame)
                self._frame_timer.start(33)
        
        return self._qt_widget
    
    def on_animation_complete(self, callback: Callable):
        """Set callback for animation completion."""
        self._on_animation_complete = callback
    
    def get_current_state(self) -> Animation3DState:
        """Get current animation state."""
        return self._current_state
    
    def get_available_animations(self) -> list[str]:
        """Get list of loaded animations."""
        return list(self._animations.keys())


class AI3DAvatarController:
    """
    High-level controller for AI to manage 3D avatar.
    
    Same interface as 2D controller for easy swapping.
    """
    
    def __init__(self, animator: Avatar3DAnimator):
        self.animator = animator
        self._is_talking = False
        self._current_emotion = "neutral"
    
    def start_talking(self):
        """Called when AI starts speaking."""
        self._is_talking = True
        self.animator.set_state(Animation3DState.TALKING)
    
    def stop_talking(self):
        """Called when AI finishes speaking."""
        self._is_talking = False
        self._apply_emotion()
    
    def set_emotion(self, emotion: str):
        """Set avatar emotion."""
        self._current_emotion = emotion.lower()
        if not self._is_talking:
            self._apply_emotion()
    
    def _apply_emotion(self):
        """Apply current emotion as animation state."""
        emotion_map = {
            "neutral": Animation3DState.IDLE,
            "happy": Animation3DState.HAPPY,
            "sad": Animation3DState.SAD,
            "surprised": Animation3DState.SURPRISED,
            "thinking": Animation3DState.THINKING,
        }
        state = emotion_map.get(self._current_emotion, Animation3DState.IDLE)
        self.animator.set_state(state)
    
    def gesture(self, gesture_name: str):
        """Play a gesture animation."""
        emotion_map = {
            "neutral": Animation3DState.IDLE,
            "happy": Animation3DState.HAPPY,
            "sad": Animation3DState.SAD,
        }
        return_state = emotion_map.get(self._current_emotion, Animation3DState.IDLE)
        self.animator.play_gesture(gesture_name, return_state)
    
    def listen(self):
        """Put avatar in listening mode."""
        self.animator.set_state(Animation3DState.LISTENING)
    
    def think(self):
        """Put avatar in thinking mode."""
        self.animator.set_state(Animation3DState.THINKING)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_3d_avatar(model_path: str, width: int = 512, height: int = 512) -> tuple[Avatar3DAnimator, AI3DAvatarController]:
    """
    Convenience function to create a 3D avatar system.
    
    Args:
        model_path: Path to 3D model file (glTF, FBX, etc.)
        width: Render width
        height: Render height
        
    Returns:
        Tuple of (animator, controller)
    """
    animator = Avatar3DAnimator(width, height)
    animator.start()
    animator.load_model(model_path)
    
    controller = AI3DAvatarController(animator)
    return animator, controller
