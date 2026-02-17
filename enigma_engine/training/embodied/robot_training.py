"""
Robot Embodied Training Data Generator

Generates training data for AI to directly control robot motors
by outputting continuous values for motors/actuators.

Training Format:
    [ROBOT: type=arm, motors={shoulder(-90,90), elbow(0,135), wrist(-45,45), gripper(0,1)}]
    [SENSORS: distance=50cm, battery=80%, gripper_force=0N]
    [CAMERA: <description or base64>]
    User: <task or instruction>
    Assistant: [MOTOR: shoulder=45, elbow=90, gripper=0.8] <action description>

The AI learns to:
1. Understand robot capabilities from motor list
2. Read sensor data to inform decisions
3. Output appropriate motor values
4. Coordinate multiple motors for smooth motion
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RobotTrainingExample:
    """Single robot training example."""
    robot_type: str
    sensor_state: dict[str, Any]
    visual_state: str
    instruction: str
    motor_outputs: dict[str, float]
    narration: str
    metadata: dict = field(default_factory=dict)


class RobotTrainingGenerator:
    """
    Generates training data for embodied robot control.
    
    Creates examples for various robot types (arm, wheeled, humanoid)
    that teach the model to output motor values directly.
    """
    
    def __init__(self, output_dir: str = "data/training/robot"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Robot configurations
        self.robot_configs = {
            "arm_6dof": {
                "motors": {
                    "base": (-180, 180),
                    "shoulder": (-90, 90),
                    "elbow": (0, 135),
                    "wrist_pitch": (-90, 90),
                    "wrist_roll": (-180, 180),
                    "gripper": (0, 1),
                },
                "sensors": ["gripper_force", "joint_positions", "camera"],
            },
            "wheeled_2wd": {
                "motors": {
                    "left_wheel": (-100, 100),
                    "right_wheel": (-100, 100),
                },
                "sensors": ["distance_front", "distance_left", "distance_right", "battery", "camera"],
            },
            "wheeled_4wd": {
                "motors": {
                    "front_left": (-100, 100),
                    "front_right": (-100, 100),
                    "rear_left": (-100, 100),
                    "rear_right": (-100, 100),
                },
                "sensors": ["distance_front", "distance_left", "distance_right", "battery", "camera", "gyro"],
            },
            "humanoid_upper": {
                "motors": {
                    "head_pan": (-90, 90),
                    "head_tilt": (-45, 45),
                    "left_shoulder_pitch": (-180, 60),
                    "left_shoulder_roll": (-30, 180),
                    "left_elbow": (0, 135),
                    "left_gripper": (0, 1),
                    "right_shoulder_pitch": (-180, 60),
                    "right_shoulder_roll": (-180, 30),
                    "right_elbow": (0, 135),
                    "right_gripper": (0, 1),
                },
                "sensors": ["camera", "microphone", "touch_sensors"],
            },
            "drone": {
                "motors": {
                    "throttle": (0, 100),
                    "pitch": (-45, 45),
                    "roll": (-45, 45),
                    "yaw": (-180, 180),
                },
                "sensors": ["altitude", "gps", "camera", "battery", "gyro", "accelerometer"],
            },
        }
    
    def generate_robot_context(self, robot_type: str) -> str:
        """Generate robot configuration context."""
        if robot_type not in self.robot_configs:
            return f"[ROBOT: type={robot_type}]"
        
        config = self.robot_configs[robot_type]
        motor_strs = [f"{name}({min_v},{max_v})" for name, (min_v, max_v) in config["motors"].items()]
        
        return f"[ROBOT: type={robot_type}, motors={{{', '.join(motor_strs)}}}]"
    
    def format_sensors(self, sensors: dict[str, Any]) -> str:
        """Format sensor readings."""
        parts = [f"{k}={v}" for k, v in sensors.items()]
        return f"[SENSORS: {', '.join(parts)}]"
    
    def format_motors(self, motors: dict[str, float]) -> str:
        """Format motor output values."""
        parts = []
        for motor, value in motors.items():
            if isinstance(value, float) and value != int(value):
                parts.append(f"{motor}={value:.2f}")
            else:
                parts.append(f"{motor}={int(value)}")
        
        return f"[MOTOR: {', '.join(parts)}]"
    
    def generate_arm_examples(self) -> list[RobotTrainingExample]:
        """Generate examples for robot arm control."""
        examples = []
        
        scenarios = [
            {
                "sensors": {"gripper_force": "0N", "object_detected": True},
                "visual": "Red cube on table, 30cm ahead",
                "instruction": "Pick up the red cube",
                "motors": {"shoulder": 30, "elbow": 90, "wrist_pitch": -45, "gripper": 1},
                "narration": "*moves arm down and grabs cube*",
            },
            {
                "sensors": {"gripper_force": "2N", "holding_object": True},
                "visual": "Holding cube, bin visible to the left",
                "instruction": "Put the cube in the bin",
                "motors": {"base": -45, "shoulder": 45, "elbow": 60, "gripper": 0},
                "narration": "*rotates to bin and releases cube*",
            },
            {
                "sensors": {"gripper_force": "0N", "object_detected": False},
                "visual": "Empty table surface",
                "instruction": "Return to home position",
                "motors": {"base": 0, "shoulder": 0, "elbow": 0, "wrist_pitch": 0, "wrist_roll": 0, "gripper": 0},
                "narration": "*returns to neutral position*",
            },
            {
                "sensors": {"gripper_force": "0N", "object_detected": True},
                "visual": "Small screw on table",
                "instruction": "Pick up the screw carefully",
                "motors": {"shoulder": 35, "elbow": 100, "wrist_pitch": -60, "gripper": 0.3},
                "narration": "*carefully grips small screw*",
            },
            {
                "sensors": {"gripper_force": "5N", "holding_object": True},
                "visual": "Holding heavy bottle",
                "instruction": "Place it gently",
                "motors": {"shoulder": 40, "elbow": 80, "gripper": 0},
                "narration": "*slowly lowers and releases bottle*",
            },
            {
                "sensors": {"gripper_force": "0N"},
                "visual": "Object on right side of workspace",
                "instruction": "Reach to the right",
                "motors": {"base": 60, "shoulder": 20, "elbow": 45},
                "narration": "*rotates and extends toward object*",
            },
        ]
        
        for s in scenarios:
            examples.append(RobotTrainingExample(
                robot_type="arm_6dof",
                sensor_state=s["sensors"],
                visual_state=s["visual"],
                instruction=s["instruction"],
                motor_outputs=s["motors"],
                narration=s["narration"],
            ))
        
        return examples
    
    def generate_wheeled_examples(self) -> list[RobotTrainingExample]:
        """Generate examples for wheeled robot control."""
        examples = []
        
        scenarios = [
            {
                "sensors": {"distance_front": "200cm", "battery": "85%"},
                "visual": "Clear hallway ahead",
                "instruction": "Drive forward",
                "motors": {"left_wheel": 50, "right_wheel": 50},
                "narration": "*drives forward*",
            },
            {
                "sensors": {"distance_front": "30cm", "battery": "85%"},
                "visual": "Wall ahead blocking path",
                "instruction": "Stop and turn around",
                "motors": {"left_wheel": -50, "right_wheel": 50},
                "narration": "*stops and rotates in place*",
            },
            {
                "sensors": {"distance_front": "100cm", "distance_left": "20cm", "battery": "85%"},
                "visual": "Wall on left, open space ahead and right",
                "instruction": "Follow the left wall",
                "motors": {"left_wheel": 40, "right_wheel": 50},
                "narration": "*curves slightly left to follow wall*",
            },
            {
                "sensors": {"distance_front": "50cm", "battery": "15%"},
                "visual": "Charging station visible ahead",
                "instruction": "Go to charging station",
                "motors": {"left_wheel": 30, "right_wheel": 30},
                "narration": "*slowly approaches charger*",
            },
            {
                "sensors": {"distance_front": "300cm", "battery": "90%"},
                "visual": "Long empty corridor",
                "instruction": "Drive fast",
                "motors": {"left_wheel": 100, "right_wheel": 100},
                "narration": "*speeds down corridor*",
            },
            {
                "sensors": {"distance_front": "80cm", "distance_right": "25cm"},
                "visual": "Doorway on the right",
                "instruction": "Turn right into doorway",
                "motors": {"left_wheel": 60, "right_wheel": 20},
                "narration": "*turns right through doorway*",
            },
        ]
        
        for s in scenarios:
            examples.append(RobotTrainingExample(
                robot_type="wheeled_2wd",
                sensor_state=s["sensors"],
                visual_state=s["visual"],
                instruction=s["instruction"],
                motor_outputs=s["motors"],
                narration=s["narration"],
            ))
        
        return examples
    
    def generate_drone_examples(self) -> list[RobotTrainingExample]:
        """Generate examples for drone control."""
        examples = []
        
        scenarios = [
            {
                "sensors": {"altitude": "0m", "battery": "100%"},
                "visual": "On ground, clear sky above",
                "instruction": "Take off",
                "motors": {"throttle": 60, "pitch": 0, "roll": 0, "yaw": 0},
                "narration": "*lifts off ground*",
            },
            {
                "sensors": {"altitude": "10m", "battery": "80%"},
                "visual": "Hovering, target building ahead",
                "instruction": "Fly forward",
                "motors": {"throttle": 50, "pitch": 15, "roll": 0, "yaw": 0},
                "narration": "*flies forward*",
            },
            {
                "sensors": {"altitude": "5m", "battery": "20%"},
                "visual": "Low battery warning",
                "instruction": "Land immediately",
                "motors": {"throttle": 20, "pitch": 0, "roll": 0, "yaw": 0},
                "narration": "*descends to land*",
            },
            {
                "sensors": {"altitude": "15m", "battery": "70%"},
                "visual": "Object of interest to the left",
                "instruction": "Turn left and look",
                "motors": {"throttle": 45, "pitch": 0, "roll": -10, "yaw": -30},
                "narration": "*rotates left to view object*",
            },
            {
                "sensors": {"altitude": "20m", "battery": "60%"},
                "visual": "High altitude, good view",
                "instruction": "Hover in place",
                "motors": {"throttle": 45, "pitch": 0, "roll": 0, "yaw": 0},
                "narration": "*hovers steadily*",
            },
            {
                "sensors": {"altitude": "8m", "battery": "50%", "wind": "strong"},
                "visual": "Windy conditions, drifting right",
                "instruction": "Stabilize against wind",
                "motors": {"throttle": 50, "pitch": 0, "roll": -15, "yaw": 0},
                "narration": "*compensates for wind drift*",
            },
        ]
        
        for s in scenarios:
            examples.append(RobotTrainingExample(
                robot_type="drone",
                sensor_state=s["sensors"],
                visual_state=s["visual"],
                instruction=s["instruction"],
                motor_outputs=s["motors"],
                narration=s["narration"],
            ))
        
        return examples
    
    def generate_humanoid_examples(self) -> list[RobotTrainingExample]:
        """Generate examples for humanoid robot control."""
        examples = []
        
        scenarios = [
            {
                "sensors": {"camera": "active", "person_detected": True},
                "visual": "Person waving at robot",
                "instruction": "Wave back",
                "motors": {"right_shoulder_pitch": -90, "right_shoulder_roll": 0, "right_elbow": 45},
                "narration": "*waves at person*",
            },
            {
                "sensors": {"camera": "active", "object_detected": True},
                "visual": "Cup on table in front",
                "instruction": "Pick up the cup",
                "motors": {"right_shoulder_pitch": 30, "right_elbow": 90, "right_gripper": 0.7},
                "narration": "*reaches for and grips cup*",
            },
            {
                "sensors": {"camera": "active"},
                "visual": "Person to the left",
                "instruction": "Look at the person",
                "motors": {"head_pan": -45, "head_tilt": 0},
                "narration": "*turns head to look at person*",
            },
            {
                "sensors": {"camera": "active", "holding_object": True},
                "visual": "Holding cup, person has hand extended",
                "instruction": "Hand over the cup",
                "motors": {"right_shoulder_pitch": -30, "right_elbow": 60, "right_gripper": 0},
                "narration": "*extends arm and releases cup*",
            },
            {
                "sensors": {"touch_sensors": "activated"},
                "visual": "Person shaking robot's hand",
                "instruction": "Shake hand",
                "motors": {"right_shoulder_pitch": 0, "right_elbow": 90, "right_gripper": 0.5},
                "narration": "*shakes hand*",
            },
        ]
        
        for s in scenarios:
            examples.append(RobotTrainingExample(
                robot_type="humanoid_upper",
                sensor_state=s["sensors"],
                visual_state=s["visual"],
                instruction=s["instruction"],
                motor_outputs=s["motors"],
                narration=s["narration"],
            ))
        
        return examples
    
    def generate_all(self, output_file: str = None) -> list[RobotTrainingExample]:
        """Generate all training examples."""
        examples = []
        
        examples.extend(self.generate_arm_examples())
        examples.extend(self.generate_wheeled_examples())
        examples.extend(self.generate_drone_examples())
        examples.extend(self.generate_humanoid_examples())
        
        random.shuffle(examples)
        
        if output_file:
            self.save_examples(examples, output_file)
        
        return examples
    
    def save_examples(self, examples: list[RobotTrainingExample], filename: str):
        """Save examples to file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            for ex in examples:
                robot_ctx = self.generate_robot_context(ex.robot_type)
                sensor_str = self.format_sensors(ex.sensor_state)
                motor_str = self.format_motors(ex.motor_outputs)
                
                data = {
                    "input": f"{robot_ctx}\n{sensor_str}\n[CAMERA: {ex.visual_state}]\nUser: {ex.instruction}",
                    "output": f"Assistant: {motor_str} {ex.narration}",
                    "metadata": {"robot_type": ex.robot_type, **ex.metadata},
                }
                f.write(json.dumps(data) + "\n")
        
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            for ex in examples:
                f.write(f"[ROBOT: {ex.robot_type}]\n")
                f.write(f"[SENSORS: {ex.sensor_state}]\n")
                f.write(f"[CAMERA: {ex.visual_state}]\n")
                f.write(f"User: {ex.instruction}\n")
                f.write(f"Assistant: {self.format_motors(ex.motor_outputs)} {ex.narration}\n\n")
        
        print(f"Saved {len(examples)} examples to {output_path}")
        return output_path


def main():
    """Generate robot training data."""
    generator = RobotTrainingGenerator()
    examples = generator.generate_all("robot_embodied.jsonl")
    print(f"Generated {len(examples)} training examples")
    
    print("\n--- Sample Examples ---")
    for ex in examples[:5]:
        print(f"[ROBOT: {ex.robot_type}]")
        print(f"[CAMERA: {ex.visual_state}]")
        print(f"User: {ex.instruction}")
        print(f"Assistant: {generator.format_motors(ex.motor_outputs)} {ex.narration}")
        print()


if __name__ == "__main__":
    main()
