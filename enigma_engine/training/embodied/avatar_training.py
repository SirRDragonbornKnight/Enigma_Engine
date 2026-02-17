"""
Avatar Embodied Training Data Generator

Generates training data for AI to directly control avatar bones
by outputting angle values instead of calling gesture tools.

Training Format:
    [SKELETON: bone1(-min,max), bone2(-min,max), ...]
    [POSE: bone1=current, bone2=current, ...]
    User: <request>
    Assistant: <response with [MOVE: bone1=target, bone2=target, ...]>

The AI learns to:
1. Understand its current pose from [POSE] context
2. Output specific angle values to achieve desired motion
3. Express nuance (enthusiastic wave vs tired wave)
4. Combine movements naturally
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import finger poses as training examples
# Use lazy import to avoid torch dependency chain
FINGER_POSES = {}
STANDARD_BONE_LIMITS = {}

def _load_pose_data():
    """Lazy load pose data to avoid import chain issues."""
    global FINGER_POSES, STANDARD_BONE_LIMITS
    if not FINGER_POSES:
        try:
            from ...avatar.bone_control import FINGER_POSES as fp, STANDARD_BONE_LIMITS as sbl
            FINGER_POSES.update(fp)
            STANDARD_BONE_LIMITS.update(sbl)
        except ImportError:
            pass  # Will use empty dicts


@dataclass
class TrainingExample:
    """Single training example."""
    user_input: str
    assistant_output: str
    context: str = ""
    metadata: dict = field(default_factory=dict)


class AvatarTrainingGenerator:
    """
    Generates training data for embodied avatar control.
    
    Converts predefined poses + variations into text training pairs
    that teach the model to output bone angles directly.
    """
    
    def __init__(self, output_dir: str = "data/training/avatar"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Movement variations for natural language
        self.intensities = ["", "slightly ", "somewhat ", "very ", "enthusiastically ", "lazily "]
        self.speeds = ["", "quickly ", "slowly ", "smoothly "]
        
    def generate_skeleton_context(self, bones: dict[str, Any] = None) -> str:
        """Generate skeleton description for context."""
        _load_pose_data()  # Lazy load to avoid import chain
        if bones is None:
            bones = STANDARD_BONE_LIMITS
        
        parts = []
        for name, limits in bones.items():
            if hasattr(limits, 'pitch_min'):
                parts.append(f"{name}(p:{limits.pitch_min}/{limits.pitch_max})")
        
        return f"[SKELETON: {', '.join(parts[:20])}...]"  # Truncate for readability
    
    def generate_pose_context(self, pose: dict[str, dict]) -> str:
        """Generate current pose state for context."""
        parts = []
        for bone, values in pose.items():
            angle_parts = []
            for axis, val in values.items():
                if val != 0:
                    angle_parts.append(f"{axis[0]}={val}")
            if angle_parts:
                parts.append(f"{bone}:{','.join(angle_parts)}")
        
        if not parts:
            return "[POSE: neutral]"
        return f"[POSE: {', '.join(parts)}]"
    
    def generate_move_output(self, pose: dict[str, dict]) -> str:
        """Generate movement command from pose."""
        parts = []
        for bone, values in pose.items():
            angle_parts = []
            for axis, val in values.items():
                angle_parts.append(f"{axis[0]}={val}")
            parts.append(f"{bone}:{','.join(angle_parts)}")
        
        return f"[MOVE: {', '.join(parts)}]"
    
    def generate_finger_pose_examples(self) -> list[TrainingExample]:
        """Generate training examples from finger poses."""
        _load_pose_data()  # Lazy load to avoid import chain
        examples = []
        
        # Map poses to natural language requests
        pose_requests = {
            "thumbs_up": [
                "give me a thumbs up",
                "thumbs up!",
                "show approval",
                "that's great!",
            ],
            "thumbs_down": [
                "thumbs down",
                "show disapproval", 
                "that's bad",
                "boo!",
            ],
            "peace_sign": [
                "show me the peace sign",
                "do the V sign",
                "peace!",
                "victory sign",
            ],
            "thumbs_up_extra": [
                "thumbs up please",
                "show me approval",
                "give me the thumbs up sign",
                "do a thumbs up",
            ],
            "pointing": [
                "point at something",
                "point forward",
                "point at me",
                "indicate direction",
            ],
            "fist": [
                "make a fist",
                "close your hand",
                "punch pose",
                "angry fist",
            ],
            "open_hand": [
                "open your hand",
                "show your palm",
                "stop gesture",
                "high five pose",
            ],
            "rock_on": [
                "do rock on",
                "devil horns",
                "metal sign",
                "rock and roll!",
            ],
            "ok_sign": [
                "do the OK sign",
                "everything's fine",
                "perfect gesture",
                "make a circle with your fingers",
            ],
            "finger_gun": [
                "do finger guns",
                "pew pew",
                "shoot finger guns",
                "point like a gun",
            ],
            "wave": [
                "wave at me",
                "say hello",
                "wave goodbye",
                "hi there!",
            ],
        }
        
        # Response templates
        responses = {
            "thumbs_up": ["You got it!", "Great job!", "Awesome!", "Sure thing!"],
            "thumbs_down": ["Boo!", "That's not good.", "Disapproved.", "Nope."],
            "peace_sign": ["Peace!", "✌️", "Got it!", "Peace out!"],
            "thumbs_up_extra": ["You got it!", "Awesome!", "Great job!", "Sure thing!"],
            "pointing": ["Look there!", "Over there!", "Right there.", "See?"],
            "fist": ["*makes fist*", "Ready!", "Pow!", "Got it."],
            "open_hand": ["*opens hand*", "Here.", "Stop!", "Here's my hand."],
            "rock_on": ["Rock on! 🤘", "Metal!", "Yeah!", "Rock and roll!"],
            "ok_sign": ["OK!", "Perfect!", "All good!", "Alright!"],
            "finger_gun": ["Pew pew!", "Gotcha!", "Bang bang!", "*finger guns*"],
            "wave": ["Hi!", "Hello!", "Hey there!", "*waves*"],
        }
        
        skeleton_ctx = self.generate_skeleton_context()
        neutral_pose = {}  # Start from neutral
        
        for pose_name, requests in pose_requests.items():
            if pose_name not in FINGER_POSES:
                continue
                
            pose_data = FINGER_POSES[pose_name]
            move_cmd = self.generate_move_output(pose_data)
            
            for request in requests:
                for intensity in self.intensities[:3]:  # Limit variations
                    response = random.choice(responses.get(pose_name, ["Done!"]))
                    
                    example = TrainingExample(
                        user_input=f"{intensity}{request}",
                        assistant_output=f"{move_cmd} {response}",
                        context=f"{skeleton_ctx}\n{self.generate_pose_context(neutral_pose)}",
                        metadata={"pose": pose_name, "type": "finger_gesture"}
                    )
                    examples.append(example)
        
        return examples
    
    def generate_body_pose_examples(self) -> list[TrainingExample]:
        """Generate training examples for full body poses."""
        examples = []
        
        # Define some body poses
        body_poses = {
            "nod_yes": {
                "request": ["nod yes", "agree", "nod your head", "say yes with your head"],
                "response": ["*nods*", "Yes!", "*nods in agreement*"],
                "bones": {"head": {"pitch": 15}},  # Then back to 0
            },
            "shake_no": {
                "request": ["shake your head no", "disagree", "say no"],
                "response": ["*shakes head*", "No.", "*shakes head no*"],
                "bones": {"head": {"yaw": 20}},  # Then -20, then 0
            },
            "look_left": {
                "request": ["look left", "look to your left", "turn head left"],
                "response": ["*looks left*", "Looking.", "*turns head*"],
                "bones": {"head": {"yaw": -45}},
            },
            "look_right": {
                "request": ["look right", "look to your right", "turn head right"],
                "response": ["*looks right*", "Looking.", "*turns head*"],
                "bones": {"head": {"yaw": 45}},
            },
            "look_up": {
                "request": ["look up", "look at the ceiling", "tilt head up"],
                "response": ["*looks up*", "Hmm...", "*gazes upward*"],
                "bones": {"head": {"pitch": -30}},
            },
            "look_down": {
                "request": ["look down", "look at the floor", "bow your head"],
                "response": ["*looks down*", "*bows head*", "..."],
                "bones": {"head": {"pitch": 30}},
            },
            "shrug": {
                "request": ["shrug", "I don't know gesture", "whatever"],
                "response": ["*shrugs*", "I don't know.", "¯\\_(ツ)_/¯"],
                "bones": {
                    "left_shoulder": {"pitch": -20},
                    "right_shoulder": {"pitch": -20},
                },
            },
            "arms_crossed": {
                "request": ["cross your arms", "look stern", "arms crossed pose"],
                "response": ["*crosses arms*", "Hmph.", "*looks stern*"],
                "bones": {
                    "left_upper_arm": {"pitch": 45, "roll": 30},
                    "right_upper_arm": {"pitch": 45, "roll": -30},
                    "left_forearm": {"pitch": 90},
                    "right_forearm": {"pitch": 90},
                },
            },
            "thinking": {
                "request": ["think about it", "thinking pose", "hmm let me think"],
                "response": ["Hmm...", "*thinking*", "Let me consider..."],
                "bones": {
                    "head": {"pitch": 10, "yaw": 15},
                    "right_upper_arm": {"pitch": 60},
                    "right_forearm": {"pitch": 120},
                },
            },
        }
        
        skeleton_ctx = self.generate_skeleton_context()
        
        for pose_name, pose_info in body_poses.items():
            move_cmd = self.generate_move_output(pose_info["bones"])
            
            for request in pose_info["request"]:
                response = random.choice(pose_info["response"])
                
                example = TrainingExample(
                    user_input=request,
                    assistant_output=f"{move_cmd} {response}",
                    context=skeleton_ctx,
                    metadata={"pose": pose_name, "type": "body_pose"}
                )
                examples.append(example)
        
        return examples
    
    def generate_emotion_examples(self) -> list[TrainingExample]:
        """Generate examples that combine emotion with movement."""
        examples = []
        
        emotions = {
            "happy": {
                "triggers": ["I got a promotion!", "I won the lottery!", "Great news!"],
                "movements": {
                    "head": {"pitch": -10},  # Look up slightly
                    "spine": {"pitch": -5},   # Straighten up
                },
                "response_style": "enthusiastic",
            },
            "sad": {
                "triggers": ["My pet died.", "I failed the test.", "I'm so sad."],
                "movements": {
                    "head": {"pitch": 20},  # Look down
                    "spine": {"pitch": 10},  # Slouch
                    "left_shoulder": {"pitch": 10},
                    "right_shoulder": {"pitch": 10},
                },
                "response_style": "sympathetic",
            },
            "excited": {
                "triggers": ["Let's celebrate!", "This is amazing!", "I can't wait!"],
                "movements": {
                    "head": {"pitch": -15, "roll": 5},
                    "spine": {"pitch": -10},
                },
                "response_style": "hyper",
            },
            "bored": {
                "triggers": ["This is so boring.", "Ugh, another meeting.", "Meh."],
                "movements": {
                    "head": {"pitch": 15, "roll": -10},
                    "spine": {"pitch": 15},
                },
                "response_style": "flat",
            },
        }
        
        for emotion, data in emotions.items():
            move_cmd = self.generate_move_output(data["movements"])
            
            for trigger in data["triggers"]:
                example = TrainingExample(
                    user_input=trigger,
                    assistant_output=f"{move_cmd} *looks {emotion}*",
                    context="",
                    metadata={"emotion": emotion, "type": "emotion_response"}
                )
                examples.append(example)
        
        return examples
    
    def generate_all(self, output_file: str = None) -> list[TrainingExample]:
        """Generate all training examples."""
        examples = []
        
        # Collect from all generators
        examples.extend(self.generate_finger_pose_examples())
        examples.extend(self.generate_body_pose_examples())
        examples.extend(self.generate_emotion_examples())
        
        # Shuffle for better training
        random.shuffle(examples)
        
        # Save if path provided
        if output_file:
            self.save_examples(examples, output_file)
        
        return examples
    
    def save_examples(self, examples: list[TrainingExample], filename: str):
        """Save examples to file in training format."""
        output_path = self.output_dir / filename
        
        # Save as JSONL for easy loading
        with open(output_path, 'w') as f:
            for ex in examples:
                data = {
                    "input": f"{ex.context}\nUser: {ex.user_input}".strip(),
                    "output": f"Assistant: {ex.assistant_output}",
                    "metadata": ex.metadata,
                }
                f.write(json.dumps(data) + "\n")
        
        # Also save as plain text for simple training
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            for ex in examples:
                f.write(f"User: {ex.user_input}\n")
                f.write(f"Assistant: {ex.assistant_output}\n\n")
        
        print(f"Saved {len(examples)} examples to {output_path}")
        return output_path
    
    def format_for_training(self, examples: list[TrainingExample]) -> str:
        """Format examples as a single training text."""
        lines = []
        for ex in examples:
            if ex.context:
                lines.append(ex.context)
            lines.append(f"User: {ex.user_input}")
            lines.append(f"Assistant: {ex.assistant_output}")
            lines.append("")  # Blank line separator
        
        return "\n".join(lines)


def main():
    """Generate avatar training data."""
    generator = AvatarTrainingGenerator()
    examples = generator.generate_all("avatar_embodied.jsonl")
    print(f"Generated {len(examples)} training examples")
    
    # Preview some examples
    print("\n--- Sample Examples ---")
    for ex in examples[:5]:
        print(f"User: {ex.user_input}")
        print(f"Assistant: {ex.assistant_output}")
        print()


if __name__ == "__main__":
    main()
