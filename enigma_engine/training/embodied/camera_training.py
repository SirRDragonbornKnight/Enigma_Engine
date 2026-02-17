"""
Camera Embodied Training Data Generator

Generates training data for AI to directly control camera
by outputting PTZ (pan, tilt, zoom) values and attention.

Training Format:
    [CAMERA: type=ptz, pan(-180,180), tilt(-90,90), zoom(1x,10x)]
    [VIEW: current_pan=0, current_tilt=0, current_zoom=1x]
    [FRAME: <description of what camera currently sees>]
    User: <where to look or what to focus on>
    Assistant: [LOOK: pan=45, tilt=-10, zoom=2x] <narration>

The AI learns to:
1. Understand camera capabilities
2. Output appropriate PTZ values to frame shots
3. Track moving subjects
4. Compose visually pleasing frames
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CameraTrainingExample:
    """Single camera training example."""
    camera_type: str
    current_view: dict[str, float]
    frame_description: str
    instruction: str
    ptz_output: dict[str, float]
    narration: str
    metadata: dict = field(default_factory=dict)


class CameraTrainingGenerator:
    """
    Generates training data for embodied camera control.
    
    Creates examples that teach the model to output
    pan/tilt/zoom values directly for active vision.
    """
    
    def __init__(self, output_dir: str = "data/training/camera"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Camera configurations
        self.camera_configs = {
            "ptz_security": {
                "pan": (-180, 180),
                "tilt": (-30, 90),
                "zoom": (1, 30),
                "features": ["motion_detection", "night_vision"],
            },
            "ptz_broadcast": {
                "pan": (-170, 170),
                "tilt": (-30, 90),
                "zoom": (1, 20),
                "features": ["smooth_motion", "preset_positions"],
            },
            "webcam_ptz": {
                "pan": (-90, 90),
                "tilt": (-30, 30),
                "zoom": (1, 4),
                "features": ["auto_focus", "face_tracking"],
            },
            "robotic_camera": {
                "pan": (-180, 180),
                "tilt": (-90, 90),
                "zoom": (1, 10),
                "features": ["high_speed", "programmable"],
            },
            "phone_camera": {
                "pan": (-180, 180),  # Via gimbal
                "tilt": (-90, 90),
                "zoom": (1, 3),  # Digital + optical
                "features": ["stabilization", "auto_exposure"],
            },
        }
    
    def generate_camera_context(self, camera_type: str) -> str:
        """Generate camera configuration context."""
        if camera_type not in self.camera_configs:
            return f"[CAMERA: type={camera_type}]"
        
        config = self.camera_configs[camera_type]
        pan_range = config["pan"]
        tilt_range = config["tilt"]
        zoom_range = config["zoom"]
        
        return f"[CAMERA: type={camera_type}, pan({pan_range[0]},{pan_range[1]}), tilt({tilt_range[0]},{tilt_range[1]}), zoom({zoom_range[0]}x,{zoom_range[1]}x)]"
    
    def format_view(self, view: dict[str, float]) -> str:
        """Format current camera view state."""
        parts = [f"{k}={v}" for k, v in view.items()]
        return f"[VIEW: {', '.join(parts)}]"
    
    def format_ptz(self, ptz: dict[str, float]) -> str:
        """Format PTZ output values."""
        parts = []
        for axis, value in ptz.items():
            if isinstance(value, float) and value != int(value):
                parts.append(f"{axis}={value:.1f}")
            else:
                parts.append(f"{axis}={int(value)}")
        
        return f"[LOOK: {', '.join(parts)}]"
    
    def generate_surveillance_examples(self) -> list[CameraTrainingExample]:
        """Generate examples for security camera control."""
        examples = []
        
        scenarios = [
            {
                "current": {"pan": 0, "tilt": 0, "zoom": 1},
                "frame": "Empty parking lot, car entering on the left",
                "instruction": "Track the incoming car",
                "ptz": {"pan": -45, "tilt": 5, "zoom": 2},
                "narration": "*pans left and zooms to track vehicle*",
            },
            {
                "current": {"pan": -30, "tilt": 10, "zoom": 3},
                "frame": "Person walking toward entrance on right side",
                "instruction": "Watch the person",
                "ptz": {"pan": 30, "tilt": 0, "zoom": 4},
                "narration": "*tracks person toward entrance*",
            },
            {
                "current": {"pan": 45, "tilt": 0, "zoom": 5},
                "frame": "Zoomed on area, nothing happening",
                "instruction": "Return to overview",
                "ptz": {"pan": 0, "tilt": 10, "zoom": 1},
                "narration": "*zooms out to wide view*",
            },
            {
                "current": {"pan": 0, "tilt": 0, "zoom": 1},
                "frame": "Motion detected in far corner",
                "instruction": "Check the motion",
                "ptz": {"pan": -90, "tilt": 5, "zoom": 8},
                "narration": "*quickly pans and zooms to motion area*",
            },
            {
                "current": {"pan": -60, "tilt": 20, "zoom": 6},
                "frame": "License plate partially visible",
                "instruction": "Get the license plate",
                "ptz": {"pan": -60, "tilt": 15, "zoom": 15},
                "narration": "*adjusts angle and zooms for plate*",
            },
            {
                "current": {"pan": 0, "tilt": 0, "zoom": 1},
                "frame": "Dark scene, low visibility",
                "instruction": "Scan right side of lot",
                "ptz": {"pan": 60, "tilt": 5, "zoom": 3},
                "narration": "*pans right to scan area*",
            },
        ]
        
        for s in scenarios:
            examples.append(CameraTrainingExample(
                camera_type="ptz_security",
                current_view=s["current"],
                frame_description=s["frame"],
                instruction=s["instruction"],
                ptz_output=s["ptz"],
                narration=s["narration"],
            ))
        
        return examples
    
    def generate_broadcast_examples(self) -> list[CameraTrainingExample]:
        """Generate examples for broadcast/production camera control."""
        examples = []
        
        scenarios = [
            {
                "current": {"pan": 0, "tilt": 0, "zoom": 1},
                "frame": "Stage with speaker at podium center",
                "instruction": "Get a medium shot of the speaker",
                "ptz": {"pan": 0, "tilt": -5, "zoom": 5},
                "narration": "*zooms to medium shot of speaker*",
            },
            {
                "current": {"pan": 0, "tilt": -5, "zoom": 5},
                "frame": "Speaker gesturing to presentation screen on right",
                "instruction": "Show the presentation",
                "ptz": {"pan": 45, "tilt": 10, "zoom": 3},
                "narration": "*pans to show presentation screen*",
            },
            {
                "current": {"pan": 30, "tilt": 0, "zoom": 3},
                "frame": "Audience raising hands",
                "instruction": "Show the audience",
                "ptz": {"pan": -90, "tilt": -10, "zoom": 2},
                "narration": "*pans to audience*",
            },
            {
                "current": {"pan": -90, "tilt": -10, "zoom": 2},
                "frame": "Person in audience standing with microphone",
                "instruction": "Focus on the questioner",
                "ptz": {"pan": -75, "tilt": -5, "zoom": 8},
                "narration": "*zooms to person asking question*",
            },
            {
                "current": {"pan": 0, "tilt": 0, "zoom": 8},
                "frame": "Close-up of speaker's face",
                "instruction": "Pull back for wide shot",
                "ptz": {"pan": 0, "tilt": 5, "zoom": 1},
                "narration": "*zooms out for establishing shot*",
            },
            {
                "current": {"pan": 0, "tilt": 5, "zoom": 1},
                "frame": "Two people on stage, one speaking",
                "instruction": "Frame both speakers",
                "ptz": {"pan": 5, "tilt": 0, "zoom": 2},
                "narration": "*adjusts to two-shot framing*",
            },
        ]
        
        for s in scenarios:
            examples.append(CameraTrainingExample(
                camera_type="ptz_broadcast",
                current_view=s["current"],
                frame_description=s["frame"],
                instruction=s["instruction"],
                ptz_output=s["ptz"],
                narration=s["narration"],
            ))
        
        return examples
    
    def generate_webcam_examples(self) -> list[CameraTrainingExample]:
        """Generate examples for webcam/video call control."""
        examples = []
        
        scenarios = [
            {
                "current": {"pan": 0, "tilt": 0, "zoom": 1},
                "frame": "User's face slightly off-center to right",
                "instruction": "Center the frame on my face",
                "ptz": {"pan": 10, "tilt": 0, "zoom": 1},
                "narration": "*adjusts to center face*",
            },
            {
                "current": {"pan": 0, "tilt": 0, "zoom": 1},
                "frame": "User showing something on desk below",
                "instruction": "Look at what I'm showing",
                "ptz": {"pan": 0, "tilt": -20, "zoom": 2},
                "narration": "*tilts down and zooms to see item*",
            },
            {
                "current": {"pan": 0, "tilt": -15, "zoom": 2},
                "frame": "Close-up of document on desk",
                "instruction": "Go back to my face",
                "ptz": {"pan": 0, "tilt": 5, "zoom": 1},
                "narration": "*returns to face framing*",
            },
            {
                "current": {"pan": 0, "tilt": 0, "zoom": 1},
                "frame": "User and another person entering frame on left",
                "instruction": "Include both of us",
                "ptz": {"pan": -20, "tilt": 0, "zoom": 0.8},
                "narration": "*widens shot to include both people*",
            },
            {
                "current": {"pan": -20, "tilt": 0, "zoom": 0.8},
                "frame": "Two people, other person now left alone",
                "instruction": "Focus on the other person",
                "ptz": {"pan": -40, "tilt": 0, "zoom": 1.5},
                "narration": "*reframes on other person*",
            },
            {
                "current": {"pan": 0, "tilt": 0, "zoom": 1},
                "frame": "Window behind user causing backlight",
                "instruction": "Frame to reduce backlight",
                "ptz": {"pan": 15, "tilt": -5, "zoom": 1.2},
                "narration": "*reframes to minimize window glare*",
            },
        ]
        
        for s in scenarios:
            examples.append(CameraTrainingExample(
                camera_type="webcam_ptz",
                current_view=s["current"],
                frame_description=s["frame"],
                instruction=s["instruction"],
                ptz_output=s["ptz"],
                narration=s["narration"],
            ))
        
        return examples
    
    def generate_tracking_examples(self) -> list[CameraTrainingExample]:
        """Generate examples for object/person tracking."""
        examples = []
        
        scenarios = [
            {
                "current": {"pan": 0, "tilt": 0, "zoom": 3},
                "frame": "Person walking right across frame",
                "instruction": "Follow the person",
                "ptz": {"pan": 15, "tilt": 0, "zoom": 3},
                "narration": "*tracks person moving right*",
            },
            {
                "current": {"pan": 15, "tilt": 0, "zoom": 3},
                "frame": "Person continuing right, nearing edge",
                "instruction": "Keep following",
                "ptz": {"pan": 35, "tilt": 0, "zoom": 3},
                "narration": "*continues tracking*",
            },
            {
                "current": {"pan": 35, "tilt": 0, "zoom": 3},
                "frame": "Person stopped, turning to camera",
                "instruction": "Hold on them",
                "ptz": {"pan": 35, "tilt": -5, "zoom": 4},
                "narration": "*zooms slightly, holds frame*",
            },
            {
                "current": {"pan": 0, "tilt": 30, "zoom": 2},
                "frame": "Bird flying overhead, moving left",
                "instruction": "Track the bird",
                "ptz": {"pan": -20, "tilt": 45, "zoom": 5},
                "narration": "*follows bird in flight*",
            },
            {
                "current": {"pan": 0, "tilt": 0, "zoom": 1},
                "frame": "Car approaching in distance",
                "instruction": "Track the approaching car",
                "ptz": {"pan": 0, "tilt": 5, "zoom": 8},
                "narration": "*zooms to track approaching vehicle*",
            },
            {
                "current": {"pan": 0, "tilt": 10, "zoom": 10},
                "frame": "Car now passing close by on left",
                "instruction": "Follow as it passes",
                "ptz": {"pan": -60, "tilt": -5, "zoom": 3},
                "narration": "*quickly pans left and zooms out to follow*",
            },
        ]
        
        for s in scenarios:
            examples.append(CameraTrainingExample(
                camera_type="robotic_camera",
                current_view=s["current"],
                frame_description=s["frame"],
                instruction=s["instruction"],
                ptz_output=s["ptz"],
                narration=s["narration"],
            ))
        
        return examples
    
    def generate_composition_examples(self) -> list[CameraTrainingExample]:
        """Generate examples for creative composition."""
        examples = []
        
        scenarios = [
            {
                "current": {"pan": 0, "tilt": 0, "zoom": 1},
                "frame": "Landscape with mountain, person in foreground",
                "instruction": "Rule of thirds - person on left",
                "ptz": {"pan": 15, "tilt": 0, "zoom": 1},
                "narration": "*reframes person to left third*",
            },
            {
                "current": {"pan": 0, "tilt": 0, "zoom": 1},
                "frame": "Sunset on horizon",
                "instruction": "Frame for dramatic sky",
                "ptz": {"pan": 0, "tilt": 15, "zoom": 1},
                "narration": "*tilts up to show more sky*",
            },
            {
                "current": {"pan": 0, "tilt": 0, "zoom": 1},
                "frame": "Flower in garden, cluttered background",
                "instruction": "Isolate the flower",
                "ptz": {"pan": 0, "tilt": -10, "zoom": 8},
                "narration": "*zooms in to blur background*",
            },
            {
                "current": {"pan": 0, "tilt": 0, "zoom": 5},
                "frame": "Portrait shot, eyes at center",
                "instruction": "Eyes at upper third",
                "ptz": {"pan": 0, "tilt": 8, "zoom": 5},
                "narration": "*adjusts for proper headroom*",
            },
            {
                "current": {"pan": 45, "tilt": 0, "zoom": 1},
                "frame": "Leading lines from road going to building",
                "instruction": "Emphasize the leading lines",
                "ptz": {"pan": 45, "tilt": -5, "zoom": 1.5},
                "narration": "*lowers and zooms slightly for depth*",
            },
        ]
        
        for s in scenarios:
            examples.append(CameraTrainingExample(
                camera_type="phone_camera",
                current_view=s["current"],
                frame_description=s["frame"],
                instruction=s["instruction"],
                ptz_output=s["ptz"],
                narration=s["narration"],
            ))
        
        return examples
    
    def generate_all(self, output_file: str = None) -> list[CameraTrainingExample]:
        """Generate all training examples."""
        examples = []
        
        examples.extend(self.generate_surveillance_examples())
        examples.extend(self.generate_broadcast_examples())
        examples.extend(self.generate_webcam_examples())
        examples.extend(self.generate_tracking_examples())
        examples.extend(self.generate_composition_examples())
        
        random.shuffle(examples)
        
        if output_file:
            self.save_examples(examples, output_file)
        
        return examples
    
    def save_examples(self, examples: list[CameraTrainingExample], filename: str):
        """Save examples to file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            for ex in examples:
                cam_ctx = self.generate_camera_context(ex.camera_type)
                view_str = self.format_view(ex.current_view)
                ptz_str = self.format_ptz(ex.ptz_output)
                
                data = {
                    "input": f"{cam_ctx}\n{view_str}\n[FRAME: {ex.frame_description}]\nUser: {ex.instruction}",
                    "output": f"Assistant: {ptz_str} {ex.narration}",
                    "metadata": {"camera_type": ex.camera_type, **ex.metadata},
                }
                f.write(json.dumps(data) + "\n")
        
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            for ex in examples:
                f.write(f"[CAMERA: {ex.camera_type}]\n")
                f.write(f"[VIEW: {ex.current_view}]\n")
                f.write(f"[FRAME: {ex.frame_description}]\n")
                f.write(f"User: {ex.instruction}\n")
                f.write(f"Assistant: {self.format_ptz(ex.ptz_output)} {ex.narration}\n\n")
        
        print(f"Saved {len(examples)} examples to {output_path}")
        return output_path


def main():
    """Generate camera training data."""
    generator = CameraTrainingGenerator()
    examples = generator.generate_all("camera_embodied.jsonl")
    print(f"Generated {len(examples)} training examples")
    
    print("\n--- Sample Examples ---")
    for ex in examples[:5]:
        print(f"[CAMERA: {ex.camera_type}]")
        print(f"[FRAME: {ex.frame_description}]")
        print(f"User: {ex.instruction}")
        print(f"Assistant: {generator.format_ptz(ex.ptz_output)} {ex.narration}")
        print()


if __name__ == "__main__":
    main()
