"""
================================================================================
            TRAINER AI - THE AI THAT TRAINS OTHER AIs
================================================================================

The Trainer AI is a meta-AI system that helps prepare, curate, and generate
training data for any specialized model in the router system.

📍 FILE: enigma_engine/core/trainer_ai.py
🏷️ TYPE: Meta-AI / Data Curator
🎯 MAIN CLASS: TrainerAI

WHAT IT DOES:
    1. Generates training data for ANY router position (router, vision, code, etc.)
    2. Regulates data quality (format validation, deduplication, scoring)
    3. Curates existing data (filters, cleans, augments)
    4. Evaluates model outputs for quality
    5. Provides data templates for each position

ROUTER POSITIONS IT SUPPORTS:
    | Position   | Data Format                      | Purpose                    |
    |------------|----------------------------------|----------------------------|
    | router     | INPUT: text | INTENT: category   | Intent classification      |
    | vision     | IMAGE: desc | CAPTION: text      | Image description          |
    | code       | TASK: desc | CODE: implementation | Code generation           |
    | math       | PROBLEM: text | SOLUTION: steps  | Math reasoning             |
    | avatar     | COMMAND: text | BONES: json      | Avatar control             |
    | chat       | USER: text | ASSISTANT: response | Conversation               |

USAGE:
    from enigma_engine.core.trainer_ai import TrainerAI, get_trainer_ai
    
    trainer = get_trainer_ai()
    
    # Generate training data for the router
    data = trainer.generate_training_data("router", count=100)
    
    # Curate existing data
    clean_data = trainer.curate_data("router", raw_data)
    
    # Get data format template
    template = trainer.get_template("code")

SEE ALSO:
    - enigma_engine/core/tool_router.py - Uses these trained models
    - scripts/train_specialized_model.py - Trains the models
    - data/specialized/ - Training data files
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import CONFIG

logger = logging.getLogger(__name__)


# =============================================================================
# DATA FORMAT DEFINITIONS FOR EACH POSITION
# =============================================================================

@dataclass
class PositionConfig:
    """Configuration for a router position's training data format."""
    name: str
    description: str
    input_prefix: str
    output_prefix: str
    separator: str
    example_count_min: int
    recommended_model_size: str
    validation_rules: List[str] = field(default_factory=list)
    

# Define all router positions and their data formats
POSITION_CONFIGS = {
    "router": PositionConfig(
        name="router",
        description="Intent classification - routes user input to the right tool",
        input_prefix="INPUT:",
        output_prefix="INTENT:",
        separator=" | ",
        example_count_min=100,
        recommended_model_size="nano",
        validation_rules=["intent_must_be_known", "no_empty_input"],
    ),
    "vision": PositionConfig(
        name="vision",
        description="Image captioning - describes visual content",
        input_prefix="IMAGE:",
        output_prefix="CAPTION:",
        separator=" | ",
        example_count_min=200,
        recommended_model_size="tiny",
        validation_rules=["caption_min_words_5", "no_empty_input"],
    ),
    "code": PositionConfig(
        name="code",
        description="Code generation - writes code from descriptions",
        input_prefix="TASK:",
        output_prefix="CODE:",
        separator="\n",
        example_count_min=100,
        recommended_model_size="small",
        validation_rules=["code_must_be_valid", "no_empty_task"],
    ),
    "math": PositionConfig(
        name="math",
        description="Mathematical reasoning - solves math problems step by step",
        input_prefix="PROBLEM:",
        output_prefix="SOLUTION:",
        separator="\n",
        example_count_min=150,
        recommended_model_size="small",
        validation_rules=["solution_has_steps", "no_empty_problem"],
    ),
    "avatar": PositionConfig(
        name="avatar",
        description="Avatar control - converts commands to bone movements",
        input_prefix="COMMAND:",
        output_prefix="BONES:",
        separator=" | ",
        example_count_min=200,
        recommended_model_size="tiny",
        validation_rules=["bones_is_valid_json", "no_empty_command"],
    ),
    "chat": PositionConfig(
        name="chat",
        description="Conversation - general chat responses",
        input_prefix="USER:",
        output_prefix="ASSISTANT:",
        separator="\n",
        example_count_min=500,
        recommended_model_size="small",
        validation_rules=["response_min_words_3", "no_empty_input"],
    ),
    "trainer": PositionConfig(
        name="trainer",
        description="Data generation - creates training data for other positions",
        input_prefix="GENERATE_FOR:",
        output_prefix="DATA:",
        separator="\n",
        example_count_min=50,
        recommended_model_size="small",
        validation_rules=["target_position_valid", "data_format_correct"],
    ),
}

# Known intents for router
KNOWN_INTENTS = [
    "chat", "image", "code", "video", "audio", "3d", "math",
    "search", "file", "settings", "help", "avatar", "memory"
]

# Example templates for generating synthetic data
SYNTHETIC_TEMPLATES = {
    "router": [
        ("draw {object}", "image"),
        ("paint {object}", "image"),
        ("create an image of {object}", "image"),
        ("generate a picture of {object}", "image"),
        ("write code to {task}", "code"),
        ("create a function that {task}", "code"),
        ("program a {object}", "code"),
        ("help me with {topic}", "chat"),
        ("explain {topic}", "chat"),
        ("what is {topic}", "chat"),
        ("tell me about {topic}", "chat"),
        ("make a video of {object}", "video"),
        ("animate {object}", "video"),
        ("say {text}", "audio"),
        ("speak {text}", "audio"),
        ("read {text} aloud", "audio"),
        ("create a 3d model of {object}", "3d"),
        ("sculpt {object}", "3d"),
        ("solve {math_problem}", "math"),
        ("calculate {math_problem}", "math"),
        ("search for {query}", "search"),
        ("find {query}", "search"),
        ("wave at me", "avatar"),
        ("dance", "avatar"),
        ("nod your head", "avatar"),
    ],
    "vision": [
        ("a {adj} {object} in a {location}", "The image shows a {adj} {object} situated in a {location}. The {object} appears clearly in the scene."),
        ("{object} on a {surface}", "A {object} is placed on top of a {surface}. The lighting highlights the {object}'s features."),
        ("person {action} near {object}", "A person is {action} near a {object}. The scene captures this moment clearly."),
    ],
    "code": {
        "python": [
            ("function to calculate factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
            ("function to reverse a string", "def reverse_string(s):\n    return s[::-1]"),
            ("function to check if palindrome", "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"),
            ("function to find max in list", "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)"),
            ("function to sort a list", "def sort_list(lst):\n    return sorted(lst)"),
        ],
    },
    "avatar": [
        ("wave", '{"right_arm": {"rotation": [0, 0, 45], "speed": 0.5}, "action": "wave"}'),
        ("nod", '{"head": {"rotation": [15, 0, 0], "speed": 0.3}, "action": "nod"}'),
        ("shake head", '{"head": {"rotation": [0, 30, 0], "speed": 0.4}, "action": "shake"}'),
        ("look left", '{"head": {"rotation": [0, -45, 0], "speed": 0.2}, "action": "look"}'),
        ("look right", '{"head": {"rotation": [0, 45, 0], "speed": 0.2}, "action": "look"}'),
        ("raise hand", '{"right_arm": {"rotation": [-90, 0, 0], "speed": 0.4}, "action": "raise"}'),
        ("dance", '{"full_body": {"animation": "dance", "speed": 1.0}, "action": "dance"}'),
        ("bow", '{"spine": {"rotation": [45, 0, 0], "speed": 0.3}, "action": "bow"}'),
    ],
}

# Word banks for synthetic data generation
WORD_BANKS = {
    "object": ["cat", "dog", "tree", "house", "car", "flower", "mountain", "river", 
               "bird", "fish", "robot", "dragon", "castle", "spaceship", "guitar"],
    "adj": ["beautiful", "colorful", "majestic", "tiny", "enormous", "ancient", 
            "modern", "mysterious", "bright", "dark", "serene", "chaotic"],
    "location": ["forest", "beach", "city", "desert", "space", "underwater", 
                 "mountain top", "garden", "office", "kitchen"],
    "surface": ["table", "desk", "floor", "shelf", "counter", "bed", "grass"],
    "topic": ["quantum physics", "machine learning", "history", "cooking", 
              "programming", "art", "music", "science", "philosophy"],
    "task": ["sort a list", "parse JSON", "connect to a database", "read a file",
             "send an email", "create a web server", "calculate fibonacci"],
    "math_problem": ["2 + 2", "the integral of x^2", "15% of 80", "solve for x: 2x + 5 = 15"],
    "text": ["hello world", "good morning", "the quick brown fox", "welcome home"],
    "action": ["standing", "sitting", "walking", "running", "jumping", "reading"],
    "query": ["python tutorials", "best restaurants", "weather forecast", "news today"],
}


# =============================================================================
# DATA QUALITY SCORING
# =============================================================================

@dataclass
class DataQualityScore:
    """Quality assessment of training data."""
    overall_score: float  # 0.0 - 1.0
    format_score: float
    diversity_score: float
    completeness_score: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


# =============================================================================
# TRAINER AI CLASS
# =============================================================================

class TrainerAI:
    """
    Meta-AI for training data generation and curation.
    
    This is the "AI that trains AIs" - it helps prepare training data
    for any specialized model in the router system.
    """
    
    def __init__(self, model=None, use_ai_generation: bool = True):
        """
        Initialize the Trainer AI.
        
        Args:
            model: Optional Forge model for AI-powered generation
            use_ai_generation: If True, uses AI for generation when available
        """
        self.model = model
        self.use_ai = use_ai_generation and model is not None
        self.positions = POSITION_CONFIGS
        self._generation_cache: Dict[str, List[str]] = {}
        
        logger.info(f"TrainerAI initialized (AI generation: {self.use_ai})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEMPLATE AND FORMAT METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_positions(self) -> List[str]:
        """Get list of all supported router positions."""
        return list(self.positions.keys())
    
    def get_position_info(self, position: str) -> Optional[PositionConfig]:
        """Get configuration for a specific position."""
        return self.positions.get(position)
    
    def get_template(self, position: str, count: int = 3) -> str:
        """
        Get example template for a position's data format.
        
        Args:
            position: Router position name
            count: Number of examples to include
        
        Returns:
            Formatted template with examples
        """
        config = self.positions.get(position)
        if not config:
            return f"Unknown position: {position}. Available: {', '.join(self.positions.keys())}"
        
        template = f"""# Training Data Format for: {config.name.upper()}
# {config.description}
# Recommended model size: {config.recommended_model_size}
# Minimum examples: {config.example_count_min}

# Format:
# {config.input_prefix} <input text>{config.separator}{config.output_prefix} <output text>

# Examples:
"""
        examples = self._generate_examples(position, count)
        template += "\n".join(examples)
        
        return template
    
    def _generate_examples(self, position: str, count: int) -> List[str]:
        """Generate example training data entries."""
        config = self.positions.get(position)
        if not config:
            return []
        
        examples = []
        
        if position == "router":
            templates = SYNTHETIC_TEMPLATES.get("router", [])
            for template, intent in random.sample(templates, min(count, len(templates))):
                filled = self._fill_template(template)
                examples.append(f"{config.input_prefix} {filled}{config.separator}{config.output_prefix} {intent}")
                
        elif position == "vision":
            templates = SYNTHETIC_TEMPLATES.get("vision", [])
            for img_desc, caption_template in random.sample(templates, min(count, len(templates))):
                filled_img = self._fill_template(img_desc)
                filled_caption = self._fill_template(caption_template)
                examples.append(f"{config.input_prefix} {filled_img}{config.separator}{config.output_prefix} {filled_caption}")
                
        elif position == "code":
            py_templates = SYNTHETIC_TEMPLATES.get("code", {}).get("python", [])
            for task, code in random.sample(py_templates, min(count, len(py_templates))):
                examples.append(f"{config.input_prefix} {task}{config.separator}{config.output_prefix}\n{code}")
                
        elif position == "avatar":
            templates = SYNTHETIC_TEMPLATES.get("avatar", [])
            for command, bones in random.sample(templates, min(count, len(templates))):
                examples.append(f"{config.input_prefix} {command}{config.separator}{config.output_prefix} {bones}")
                
        elif position == "chat":
            # Basic chat examples
            chat_examples = [
                ("Hello!", "Hello! How can I help you today?"),
                ("How are you?", "I'm doing well, thank you for asking! How can I assist you?"),
                ("What's the weather like?", "I don't have access to real-time weather data, but I can help you find a weather service!"),
            ]
            for user, assistant in random.sample(chat_examples, min(count, len(chat_examples))):
                examples.append(f"{config.input_prefix} {user}{config.separator}{config.output_prefix} {assistant}")
                
        elif position == "math":
            math_examples = [
                ("What is 2 + 2?", "Step 1: Add 2 and 2\nStep 2: 2 + 2 = 4\nAnswer: 4"),
                ("Solve: 3x = 15", "Step 1: Divide both sides by 3\nStep 2: x = 15/3\nStep 3: x = 5\nAnswer: x = 5"),
            ]
            for problem, solution in random.sample(math_examples, min(count, len(math_examples))):
                examples.append(f"{config.input_prefix} {problem}{config.separator}{config.output_prefix}\n{solution}")
                
        return examples
    
    def _fill_template(self, template: str) -> str:
        """Fill a template with random words from word banks."""
        result = template
        for key, words in WORD_BANKS.items():
            placeholder = "{" + key + "}"
            while placeholder in result:
                result = result.replace(placeholder, random.choice(words), 1)
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # DATA GENERATION METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def generate_training_data(
        self,
        position: str,
        count: int = 100,
        seed_data: Optional[str] = None,
        use_ai: Optional[bool] = None,
    ) -> str:
        """
        Generate training data for a specific router position.
        
        Args:
            position: Router position (router, vision, code, etc.)
            count: Number of examples to generate
            seed_data: Optional existing data to augment
            use_ai: Override AI generation setting
        
        Returns:
            Generated training data as string
        """
        config = self.positions.get(position)
        if not config:
            raise ValueError(f"Unknown position: {position}. Available: {', '.join(self.positions.keys())}")
        
        should_use_ai = use_ai if use_ai is not None else self.use_ai
        
        if should_use_ai and self.model:
            return self._generate_with_ai(position, count, seed_data)
        else:
            return self._generate_synthetic(position, count, seed_data)
    
    def _generate_synthetic(self, position: str, count: int, seed_data: Optional[str]) -> str:
        """Generate synthetic training data using templates."""
        config = self.positions[position]
        examples = []
        
        # Parse seed data for patterns if provided
        patterns = []
        if seed_data:
            patterns = self._extract_patterns(seed_data, position)
        
        generated = 0
        max_attempts = count * 3  # Prevent infinite loops
        attempts = 0
        
        while generated < count and attempts < max_attempts:
            attempts += 1
            
            # Try to generate from templates
            example = self._generate_single_example(position, patterns)
            if example and example not in examples:
                examples.append(example)
                generated += 1
        
        # Add header
        header = f"""# Generated Training Data for: {config.name.upper()}
# Generated by TrainerAI
# Count: {len(examples)} examples
# Format: {config.input_prefix} <input>{config.separator}{config.output_prefix} <output>

"""
        return header + "\n".join(examples)
    
    def _generate_single_example(self, position: str, patterns: List) -> Optional[str]:
        """Generate a single training example."""
        config = self.positions[position]
        
        if position == "router":
            templates = SYNTHETIC_TEMPLATES.get("router", [])
            if templates:
                template, intent = random.choice(templates)
                filled = self._fill_template(template)
                return f"{config.input_prefix} {filled}{config.separator}{config.output_prefix} {intent}"
                
        elif position == "avatar":
            templates = SYNTHETIC_TEMPLATES.get("avatar", [])
            if templates:
                command, bones = random.choice(templates)
                # Add variation
                variations = ["please", "", "can you", ""]
                prefix = random.choice(variations)
                full_command = f"{prefix} {command}".strip()
                return f"{config.input_prefix} {full_command}{config.separator}{config.output_prefix} {bones}"
        
        # Fall back to basic examples
        return self._generate_examples(position, 1)[0] if self._generate_examples(position, 1) else None
    
    def _generate_with_ai(self, position: str, count: int, seed_data: Optional[str]) -> str:
        """Generate training data using the AI model."""
        config = self.positions[position]
        
        prompt = f"""Generate {count} training examples for a {config.description}.

Format each example as:
{config.input_prefix} <input text>{config.separator}{config.output_prefix} <output text>

Examples should be diverse and high quality.
"""
        if seed_data:
            prompt += f"\nHere are some existing examples to learn from:\n{seed_data[:1000]}\n"
        
        prompt += f"\nGenerate {count} new examples:\n"
        
        try:
            # Use the model to generate
            response = self.model.generate(prompt, max_tokens=count * 100)
            return response
        except Exception as e:
            logger.warning(f"AI generation failed, falling back to synthetic: {e}")
            return self._generate_synthetic(position, count, seed_data)
    
    def _extract_patterns(self, data: str, position: str) -> List[Tuple[str, str]]:
        """Extract input/output patterns from existing data."""
        config = self.positions[position]
        patterns = []
        
        lines = data.split('\n')
        for line in lines:
            if config.input_prefix in line and config.output_prefix in line:
                try:
                    parts = line.split(config.separator)
                    if len(parts) >= 2:
                        input_part = parts[0].replace(config.input_prefix, '').strip()
                        output_part = parts[1].replace(config.output_prefix, '').strip()
                        patterns.append((input_part, output_part))
                except Exception:
                    pass
        
        return patterns
    
    # ─────────────────────────────────────────────────────────────────────────
    # DATA CURATION METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def curate_data(
        self,
        position: str,
        raw_data: str,
        remove_duplicates: bool = True,
        validate_format: bool = True,
        filter_low_quality: bool = True,
    ) -> Tuple[str, DataQualityScore]:
        """
        Curate and clean training data.
        
        Args:
            position: Router position
            raw_data: Raw training data to curate
            remove_duplicates: Remove duplicate entries
            validate_format: Check format validity
            filter_low_quality: Remove low-quality entries
        
        Returns:
            Tuple of (cleaned data, quality score)
        """
        config = self.positions.get(position)
        if not config:
            raise ValueError(f"Unknown position: {position}")
        
        lines = [l.strip() for l in raw_data.split('\n') if l.strip() and not l.startswith('#')]
        original_count = len(lines)
        issues = []
        suggestions = []
        
        # Remove duplicates
        if remove_duplicates:
            unique_lines = list(dict.fromkeys(lines))
            duplicates_removed = len(lines) - len(unique_lines)
            if duplicates_removed > 0:
                issues.append(f"Removed {duplicates_removed} duplicate entries")
            lines = unique_lines
        
        # Validate format
        valid_lines = []
        invalid_count = 0
        
        for line in lines:
            is_valid, error = self._validate_line(line, config)
            if is_valid:
                valid_lines.append(line)
            else:
                invalid_count += 1
                if invalid_count <= 5:  # Only report first 5
                    issues.append(f"Invalid: {line[:50]}... ({error})")
        
        if invalid_count > 0:
            suggestions.append(f"Fix {invalid_count} invalid entries using format: {config.input_prefix} <input>{config.separator}{config.output_prefix} <output>")
        
        lines = valid_lines
        
        # Filter low quality
        if filter_low_quality:
            high_quality = []
            low_quality_count = 0
            
            for line in lines:
                quality = self._assess_line_quality(line, config)
                if quality >= 0.5:
                    high_quality.append(line)
                else:
                    low_quality_count += 1
            
            if low_quality_count > 0:
                issues.append(f"Filtered {low_quality_count} low-quality entries")
            
            lines = high_quality
        
        # Calculate scores
        format_score = len(lines) / original_count if original_count > 0 else 0
        diversity_score = self._calculate_diversity(lines, config)
        completeness_score = min(1.0, len(lines) / config.example_count_min)
        overall_score = (format_score + diversity_score + completeness_score) / 3
        
        # Generate suggestions
        if len(lines) < config.example_count_min:
            suggestions.append(f"Add {config.example_count_min - len(lines)} more examples (minimum: {config.example_count_min})")
        
        if diversity_score < 0.7:
            suggestions.append("Increase variety in your examples - many entries are similar")
        
        # Rebuild data
        header = f"""# Curated Training Data for: {config.name.upper()}
# Original: {original_count} entries | Cleaned: {len(lines)} entries
# Quality Score: {overall_score:.2f}

"""
        cleaned_data = header + "\n".join(lines)
        
        score = DataQualityScore(
            overall_score=overall_score,
            format_score=format_score,
            diversity_score=diversity_score,
            completeness_score=completeness_score,
            issues=issues,
            suggestions=suggestions,
        )
        
        return cleaned_data, score
    
    def _validate_line(self, line: str, config: PositionConfig) -> Tuple[bool, str]:
        """Validate a single training line."""
        # Check basic format
        if config.input_prefix not in line:
            return False, f"Missing {config.input_prefix}"
        
        if config.output_prefix not in line:
            return False, f"Missing {config.output_prefix}"
        
        # Position-specific validation
        if "router" in config.name:
            # Check intent is known
            for intent in KNOWN_INTENTS:
                if intent in line.lower():
                    return True, ""
            return False, "Unknown intent"
        
        if "avatar" in config.name:
            # Check bones is valid JSON
            if config.output_prefix in line:
                bones_part = line.split(config.output_prefix)[-1].strip()
                try:
                    json.loads(bones_part)
                except json.JSONDecodeError:
                    return False, "Invalid JSON in bones"
        
        return True, ""
    
    def _assess_line_quality(self, line: str, config: PositionConfig) -> float:
        """Assess quality of a single training line (0.0 - 1.0)."""
        score = 0.5  # Base score
        
        # Length checks
        if len(line) > 20:
            score += 0.1
        if len(line) > 50:
            score += 0.1
        
        # Has both parts
        if config.input_prefix in line and config.output_prefix in line:
            score += 0.2
        
        # Not too repetitive
        words = line.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 0.1
        
        return min(1.0, score)
    
    def _calculate_diversity(self, lines: List[str], config: PositionConfig) -> float:
        """Calculate diversity score of training data."""
        if not lines:
            return 0.0
        
        # Calculate word distribution
        all_words = []
        for line in lines:
            all_words.extend(line.lower().split())
        
        if not all_words:
            return 0.0
        
        word_counts = Counter(all_words)
        unique_ratio = len(word_counts) / len(all_words)
        
        # Check output variety (for router, check intent distribution)
        if config.name == "router":
            intents = []
            for line in lines:
                for intent in KNOWN_INTENTS:
                    if config.output_prefix in line and intent in line.split(config.output_prefix)[-1].lower():
                        intents.append(intent)
                        break
            
            if intents:
                intent_counts = Counter(intents)
                intent_variety = len(intent_counts) / len(KNOWN_INTENTS)
                return (unique_ratio + intent_variety) / 2
        
        return unique_ratio
    
    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATION METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def evaluate_model_output(
        self,
        position: str,
        input_text: str,
        output_text: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a model's output for quality.
        
        Args:
            position: Router position the model is for
            input_text: The input given to the model
            output_text: The model's output
        
        Returns:
            Evaluation dict with scores and feedback
        """
        config = self.positions.get(position)
        if not config:
            return {"error": f"Unknown position: {position}"}
        
        evaluation = {
            "position": position,
            "input": input_text[:100],
            "output": output_text[:200],
            "scores": {},
            "feedback": [],
            "overall": 0.0,
        }
        
        # Length score
        min_length = {"router": 3, "vision": 20, "code": 10, "chat": 5, "avatar": 10}.get(position, 10)
        length_score = min(1.0, len(output_text) / min_length) if min_length else 1.0
        evaluation["scores"]["length"] = length_score
        
        # Relevance score (basic keyword matching)
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())
        common = input_words & output_words
        relevance_score = len(common) / max(len(input_words), 1) if input_words else 0.5
        evaluation["scores"]["relevance"] = min(1.0, relevance_score + 0.3)  # Boost since exact match not required
        
        # Format score
        format_score = 1.0
        if position == "router":
            # Check if output is a known intent
            format_score = 1.0 if any(intent in output_text.lower() for intent in KNOWN_INTENTS) else 0.0
        elif position == "avatar":
            # Check if output is valid JSON
            try:
                json.loads(output_text)
                format_score = 1.0
            except json.JSONDecodeError:
                format_score = 0.0
                evaluation["feedback"].append("Output should be valid JSON for avatar control")
        elif position == "code":
            # Basic code check
            code_indicators = ["def ", "class ", "import ", "return ", "if ", "for ", "while ", "="]
            format_score = 1.0 if any(ind in output_text for ind in code_indicators) else 0.5
        
        evaluation["scores"]["format"] = format_score
        
        # Calculate overall
        weights = {"length": 0.2, "relevance": 0.4, "format": 0.4}
        overall = sum(evaluation["scores"].get(k, 0) * w for k, w in weights.items())
        evaluation["overall"] = overall
        
        # Generate feedback
        if overall < 0.5:
            evaluation["feedback"].append("Output quality is low - consider retraining with more examples")
        elif overall < 0.7:
            evaluation["feedback"].append("Output is acceptable but could be improved")
        else:
            evaluation["feedback"].append("Output quality is good")
        
        return evaluation
    
    # ─────────────────────────────────────────────────────────────────────────
    # FILE OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def save_training_data(self, position: str, data: str, filename: Optional[str] = None) -> Path:
        """
        Save generated training data to file.
        
        Args:
            position: Router position
            data: Training data string
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        data_dir = Path(CONFIG.get("data_dir", "data")) / "specialized"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"{position}_training.txt"
        
        filepath = data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data)
        
        logger.info(f"Saved training data to: {filepath}")
        return filepath
    
    def load_training_data(self, position: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Load existing training data from file.
        
        Args:
            position: Router position
            filename: Optional custom filename
        
        Returns:
            Training data string or None if not found
        """
        data_dir = Path(CONFIG.get("data_dir", "data")) / "specialized"
        
        if filename is None:
            filename = f"{position}_training.txt"
        
        filepath = data_dir / filename
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_trainer_ai_instance: Optional[TrainerAI] = None


def get_trainer_ai(model=None) -> TrainerAI:
    """
    Get the global TrainerAI instance.
    
    Args:
        model: Optional model to use for AI-powered generation
    
    Returns:
        TrainerAI instance
    """
    global _trainer_ai_instance
    
    if _trainer_ai_instance is None:
        _trainer_ai_instance = TrainerAI(model=model)
    elif model is not None and _trainer_ai_instance.model is None:
        _trainer_ai_instance.model = model
        _trainer_ai_instance.use_ai = True
    
    return _trainer_ai_instance


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TrainerAI - Generate and curate training data")
    parser.add_argument("action", choices=["generate", "curate", "template", "list"],
                        help="Action to perform")
    parser.add_argument("--position", "-p", help="Router position (router, vision, code, etc.)")
    parser.add_argument("--count", "-c", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--input", "-i", help="Input file for curation")
    parser.add_argument("--output", "-o", help="Output file")
    
    args = parser.parse_args()
    
    trainer = TrainerAI()
    
    if args.action == "list":
        print("Available router positions:")
        for name, config in POSITION_CONFIGS.items():
            print(f"  {name}: {config.description}")
            print(f"    - Recommended size: {config.recommended_model_size}")
            print(f"    - Min examples: {config.example_count_min}")
        
    elif args.action == "template":
        if not args.position:
            print("Error: --position required for template")
        else:
            print(trainer.get_template(args.position, count=5))
        
    elif args.action == "generate":
        if not args.position:
            print("Error: --position required for generate")
        else:
            data = trainer.generate_training_data(args.position, count=args.count)
            if args.output:
                trainer.save_training_data(args.position, data, args.output)
                print(f"Saved to: {args.output}")
            else:
                print(data)
        
    elif args.action == "curate":
        if not args.position or not args.input:
            print("Error: --position and --input required for curate")
        else:
            with open(args.input) as f:
                raw_data = f.read()
            
            cleaned, score = trainer.curate_data(args.position, raw_data)
            
            print(f"\nQuality Score: {score.overall_score:.2f}")
            print(f"  Format: {score.format_score:.2f}")
            print(f"  Diversity: {score.diversity_score:.2f}")
            print(f"  Completeness: {score.completeness_score:.2f}")
            
            if score.issues:
                print("\nIssues:")
                for issue in score.issues:
                    print(f"  - {issue}")
            
            if score.suggestions:
                print("\nSuggestions:")
                for suggestion in score.suggestions:
                    print(f"  - {suggestion}")
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(cleaned)
                print(f"\nSaved cleaned data to: {args.output}")
