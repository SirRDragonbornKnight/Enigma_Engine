"""
Game Embodied Training Data Generator

Generates training data for AI to directly control game inputs
by outputting continuous values for keyboard/mouse/controller.

Training Format:
    [INPUTS: W,A,S,D,space,shift,mouse(x,y),click(L,R)]
    [STATE: W=0,A=0,S=0,D=0,mouse=(0,0)]
    [SCREEN: <description or base64>]
    User: <game situation or instruction>
    Assistant: [INPUT: W=1,mouse_dx=+10,L_click=1] <action description>

The AI learns to:
1. Understand game state from screen/description
2. Output appropriate input combinations
3. Time inputs correctly (press, hold, release)
4. React to game events
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GameTrainingExample:
    """Single game training example."""
    game_state: str  # Description of what's on screen
    instruction: str  # What to do
    inputs: dict[str, float]  # Input values to output
    narration: str  # AI's description of action
    game_type: str = "general"
    metadata: dict = field(default_factory=dict)


class GameTrainingGenerator:
    """
    Generates training data for embodied game control.
    
    Creates examples that teach the model to output
    keyboard/mouse/controller values directly.
    """
    
    def __init__(self, output_dir: str = "data/training/game"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Available input types
        self.keyboard_keys = [
            "W", "A", "S", "D",  # Movement
            "space", "shift", "ctrl",  # Modifiers
            "E", "F", "R", "Q",  # Actions
            "1", "2", "3", "4", "5",  # Slots
            "escape", "tab", "enter",  # UI
        ]
        
        self.mouse_inputs = ["mouse_x", "mouse_y", "left_click", "right_click", "scroll"]
        
        self.controller_inputs = [
            "left_stick_x", "left_stick_y",
            "right_stick_x", "right_stick_y",
            "left_trigger", "right_trigger",
            "A", "B", "X", "Y",
            "left_bumper", "right_bumper",
            "dpad_up", "dpad_down", "dpad_left", "dpad_right",
        ]
    
    def generate_input_context(self, input_type: str = "keyboard_mouse") -> str:
        """Generate available inputs context."""
        if input_type == "keyboard_mouse":
            keys = ",".join(self.keyboard_keys[:10])
            return f"[INPUTS: keys({keys}), mouse(x,y,L,R,scroll)]"
        elif input_type == "controller":
            return "[INPUTS: left_stick(x,y), right_stick(x,y), triggers(L,R), buttons(A,B,X,Y,LB,RB)]"
        return "[INPUTS: keyboard, mouse]"
    
    def format_inputs(self, inputs: dict[str, float]) -> str:
        """Format input values as output string."""
        parts = []
        for key, value in inputs.items():
            if value != 0:
                if isinstance(value, float) and value != int(value):
                    parts.append(f"{key}={value:.2f}")
                else:
                    parts.append(f"{key}={int(value)}")
        
        if not parts:
            return "[INPUT: none]"
        return f"[INPUT: {', '.join(parts)}]"
    
    def generate_fps_examples(self) -> list[GameTrainingExample]:
        """Generate examples for FPS games."""
        examples = []
        
        scenarios = [
            {
                "state": "Enemy spotted ahead, crosshair near target",
                "instruction": "Shoot the enemy",
                "inputs": {"left_click": 1},
                "narration": "*fires at enemy*",
            },
            {
                "state": "Low on health, enemy approaching",
                "instruction": "Take cover",
                "inputs": {"S": 1, "ctrl": 1},
                "narration": "*backs up and crouches behind cover*",
            },
            {
                "state": "Empty hallway ahead",
                "instruction": "Move forward carefully",
                "inputs": {"W": 1, "shift": 1},
                "narration": "*walks forward slowly*",
            },
            {
                "state": "Ammo indicator shows 0 bullets",
                "instruction": "Reload",
                "inputs": {"R": 1},
                "narration": "*reloads weapon*",
            },
            {
                "state": "Enemy on the left side of screen",
                "instruction": "Aim at enemy",
                "inputs": {"mouse_x": -50},
                "narration": "*turns to face enemy*",
            },
            {
                "state": "Grenade indicator available, enemies clustered",
                "instruction": "Throw grenade",
                "inputs": {"G": 1},
                "narration": "*throws grenade at group*",
            },
            {
                "state": "Under fire, health critical",
                "instruction": "Sprint to cover",
                "inputs": {"W": 1, "shift": 1, "A": 1},
                "narration": "*sprints diagonally to cover*",
            },
        ]
        
        for scenario in scenarios:
            examples.append(GameTrainingExample(
                game_state=scenario["state"],
                instruction=scenario["instruction"],
                inputs=scenario["inputs"],
                narration=scenario["narration"],
                game_type="fps",
            ))
        
        return examples
    
    def generate_platformer_examples(self) -> list[GameTrainingExample]:
        """Generate examples for platformer games."""
        examples = []
        
        scenarios = [
            {
                "state": "Gap ahead, platform visible across",
                "instruction": "Jump across",
                "inputs": {"W": 1, "space": 1},
                "narration": "*runs and jumps across gap*",
            },
            {
                "state": "Enemy walking toward player",
                "instruction": "Jump on enemy",
                "inputs": {"W": 1, "space": 1},
                "narration": "*jumps onto enemy*",
            },
            {
                "state": "Spikes on floor ahead",
                "instruction": "Avoid spikes",
                "inputs": {"space": 1},
                "narration": "*jumps over spikes*",
            },
            {
                "state": "Collectible coin above",
                "instruction": "Get the coin",
                "inputs": {"space": 1},
                "narration": "*jumps to collect coin*",
            },
            {
                "state": "Wall ahead, can wall-jump",
                "instruction": "Wall jump",
                "inputs": {"W": 1, "space": 1, "D": 1},
                "narration": "*wall jumps up*",
            },
            {
                "state": "Moving platform approaching",
                "instruction": "Land on platform",
                "inputs": {"space": 1},
                "narration": "*times jump onto platform*",
            },
        ]
        
        for scenario in scenarios:
            examples.append(GameTrainingExample(
                game_state=scenario["state"],
                instruction=scenario["instruction"],
                inputs=scenario["inputs"],
                narration=scenario["narration"],
                game_type="platformer",
            ))
        
        return examples
    
    def generate_racing_examples(self) -> list[GameTrainingExample]:
        """Generate examples for racing games."""
        examples = []
        
        scenarios = [
            {
                "state": "Straight road ahead, green light",
                "instruction": "Accelerate",
                "inputs": {"W": 1},
                "narration": "*floors the gas*",
            },
            {
                "state": "Sharp turn coming up on the left",
                "instruction": "Take the turn",
                "inputs": {"A": 1, "S": 0.3},
                "narration": "*brakes slightly and turns left*",
            },
            {
                "state": "Car ahead, space on right to pass",
                "instruction": "Overtake",
                "inputs": {"W": 1, "D": 0.5},
                "narration": "*moves right to overtake*",
            },
            {
                "state": "Approaching finish line, in 1st place",
                "instruction": "Cross finish",
                "inputs": {"W": 1},
                "narration": "*crosses finish line!*",
            },
            {
                "state": "Lost control, car spinning",
                "instruction": "Recover",
                "inputs": {"S": 1, "A": 0.5},
                "narration": "*counter-steers to recover*",
            },
            {
                "state": "Nitro boost available, straight ahead",
                "instruction": "Use boost",
                "inputs": {"W": 1, "shift": 1},
                "narration": "*activates nitro boost*",
            },
        ]
        
        for scenario in scenarios:
            examples.append(GameTrainingExample(
                game_state=scenario["state"],
                instruction=scenario["instruction"],
                inputs=scenario["inputs"],
                narration=scenario["narration"],
                game_type="racing",
            ))
        
        return examples
    
    def generate_rpg_examples(self) -> list[GameTrainingExample]:
        """Generate examples for RPG games."""
        examples = []
        
        scenarios = [
            {
                "state": "NPC with quest marker visible",
                "instruction": "Talk to NPC",
                "inputs": {"E": 1},
                "narration": "*interacts with NPC*",
            },
            {
                "state": "Dialog options shown, option 2 is kind",
                "instruction": "Choose kind option",
                "inputs": {"2": 1},
                "narration": "*selects kind dialog option*",
            },
            {
                "state": "Enemy attacking, health bar showing",
                "instruction": "Attack enemy",
                "inputs": {"left_click": 1},
                "narration": "*attacks the enemy*",
            },
            {
                "state": "Inventory shows health potion, HP low",
                "instruction": "Use health potion",
                "inputs": {"1": 1},
                "narration": "*drinks health potion*",
            },
            {
                "state": "Loot sparkle on ground nearby",
                "instruction": "Pick up loot",
                "inputs": {"F": 1},
                "narration": "*picks up the loot*",
            },
            {
                "state": "Multiple enemies, one is casting spell",
                "instruction": "Interrupt caster",
                "inputs": {"mouse_x": 30, "left_click": 1},
                "narration": "*targets and attacks the caster*",
            },
        ]
        
        for scenario in scenarios:
            examples.append(GameTrainingExample(
                game_state=scenario["state"],
                instruction=scenario["instruction"],
                inputs=scenario["inputs"],
                narration=scenario["narration"],
                game_type="rpg",
            ))
        
        return examples
    
    def generate_controller_examples(self) -> list[GameTrainingExample]:
        """Generate examples for controller-based games."""
        examples = []
        
        scenarios = [
            {
                "state": "Character standing, path ahead",
                "instruction": "Walk forward",
                "inputs": {"left_stick_y": 0.5},
                "narration": "*walks forward*",
            },
            {
                "state": "Enemy in front, weapon equipped",
                "instruction": "Attack",
                "inputs": {"right_trigger": 1},
                "narration": "*attacks enemy*",
            },
            {
                "state": "Ledge ahead, need to jump",
                "instruction": "Jump across",
                "inputs": {"left_stick_y": 1, "A": 1},
                "narration": "*runs and jumps*",
            },
            {
                "state": "Camera facing wrong direction",
                "instruction": "Look around",
                "inputs": {"right_stick_x": 0.8},
                "narration": "*rotates camera*",
            },
            {
                "state": "Aiming at target, scope zoomed",
                "instruction": "Fire",
                "inputs": {"right_trigger": 1},
                "narration": "*fires weapon*",
            },
            {
                "state": "Menu open, cursor on option",
                "instruction": "Select option",
                "inputs": {"A": 1},
                "narration": "*confirms selection*",
            },
        ]
        
        for scenario in scenarios:
            examples.append(GameTrainingExample(
                game_state=scenario["state"],
                instruction=scenario["instruction"],
                inputs=scenario["inputs"],
                narration=scenario["narration"],
                game_type="controller",
            ))
        
        return examples
    
    def generate_all(self, output_file: str = None) -> list[GameTrainingExample]:
        """Generate all training examples."""
        examples = []
        
        examples.extend(self.generate_fps_examples())
        examples.extend(self.generate_platformer_examples())
        examples.extend(self.generate_racing_examples())
        examples.extend(self.generate_rpg_examples())
        examples.extend(self.generate_controller_examples())
        
        random.shuffle(examples)
        
        if output_file:
            self.save_examples(examples, output_file)
        
        return examples
    
    def save_examples(self, examples: list[GameTrainingExample], filename: str):
        """Save examples to file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            for ex in examples:
                context = self.generate_input_context(
                    "controller" if ex.game_type == "controller" else "keyboard_mouse"
                )
                input_str = self.format_inputs(ex.inputs)
                
                data = {
                    "input": f"{context}\n[SCREEN: {ex.game_state}]\nUser: {ex.instruction}",
                    "output": f"Assistant: {input_str} {ex.narration}",
                    "metadata": {"game_type": ex.game_type, **ex.metadata},
                }
                f.write(json.dumps(data) + "\n")
        
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            for ex in examples:
                f.write(f"[SCREEN: {ex.game_state}]\n")
                f.write(f"User: {ex.instruction}\n")
                f.write(f"Assistant: {self.format_inputs(ex.inputs)} {ex.narration}\n\n")
        
        print(f"Saved {len(examples)} examples to {output_path}")
        return output_path


def main():
    """Generate game training data."""
    generator = GameTrainingGenerator()
    examples = generator.generate_all("game_embodied.jsonl")
    print(f"Generated {len(examples)} training examples")
    
    print("\n--- Sample Examples ---")
    for ex in examples[:5]:
        print(f"[SCREEN: {ex.game_state}]")
        print(f"User: {ex.instruction}")
        print(f"Assistant: {generator.format_inputs(ex.inputs)} {ex.narration}")
        print()


if __name__ == "__main__":
    main()
