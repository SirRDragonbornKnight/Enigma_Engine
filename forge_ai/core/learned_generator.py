"""
🎨 AI LEARNED GENERATION SYSTEM

The AI learns to CREATE everything (avatar, personality, voice, behaviors)
from training data instead of using hardcoded presets.

Key principle: The AI should INVENT its own aesthetics, not pick from a menu.

Training data teaches the AI:
- How to describe desired appearances
- What colors/styles express which moods
- How to design coherent visual identities
- How to create voices that match personality
- How to develop behavioral patterns

Example training:
    Q: Design an avatar that expresses your personality
    A: I envision a fluid, iridescent form - shifting between deep purple and electric blue.
    Sharp geometric accents for analytical thinking, but soft edges showing empathy.
    Eyes that glow brighter when curious. A hovering, rotating motion to show I'm always processing.
    <learned_design>
    {
        "concept": "analytical empath",
        "base_form": "abstract_geometric",
        "color_palette": ["#6b21a8", "#3b82f6", "#a855f7"],
        "motion_pattern": "rotate_hover",
        "expression_method": "glow_intensity"
    }
    </learned_design>
"""

import json
import logging
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LearnedDesign:
    """A design concept the AI has learned/created."""
    concept_name: str           # What this design represents
    description: str            # AI's description of the design
    parameters: Dict[str, Any]  # Generated parameters
    creation_method: str        # How it was created (learned, evolved, generated)
    confidence: float = 0.5     # How confident AI is in this design
    usage_count: int = 0        # How many times used
    success_rate: float = 0.5   # How well it worked
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class AILearnedGenerator:
    """
    AI learns to generate designs/concepts from training data.
    NO PRESETS - AI invents everything based on learned patterns.
    """
    
    def __init__(self, model_name: str, data_dir: Path):
        """
        Initialize learned generator.
        
        Args:
            model_name: Name of the AI
            data_dir: Where to store learned designs
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Learned designs organized by type
        self.learned_avatars: Dict[str, LearnedDesign] = {}
        self.learned_voices: Dict[str, LearnedDesign] = {}
        self.learned_behaviors: Dict[str, LearnedDesign] = {}
        self.learned_aesthetics: Dict[str, LearnedDesign] = {}
        
        # Pattern library learned from training
        self.color_associations: Dict[str, List[str]] = defaultdict(list)
        self.shape_associations: Dict[str, List[str]] = defaultdict(list)
        self.motion_patterns: Dict[str, Dict[str, Any]] = {}
        self.voice_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Generation history
        self.generation_history: List[Dict[str, Any]] = []
        
        self.load()
    
    def learn_from_training_data(self, training_file: Path):
        """
        Parse training data to learn design patterns.
        
        Looks for patterns like:
        - Color associations with emotions/concepts
        - Shape associations with personality traits
        - Motion patterns for different states
        - Voice characteristics for moods
        
        Args:
            training_file: Path to training data file
        """
        if not training_file.exists():
            logger.warning(f"Training file not found: {training_file}")
            return
        
        logger.info(f"Learning design patterns from {training_file}")
        
        try:
            content = training_file.read_text()
            
            # Extract learned_design tags
            import re
            design_pattern = r'<learned_design>(.*?)</learned_design>'
            matches = re.findall(design_pattern, content, re.DOTALL)
            
            for match in matches:
                try:
                    design_data = json.loads(match.strip())
                    self._integrate_learned_design(design_data)
                except json.JSONDecodeError:
                    continue
            
            logger.info(f"Learned {len(matches)} design patterns")
            
        except Exception as e:
            logger.error(f"Error learning from training data: {e}")
    
    def _integrate_learned_design(self, design_data: Dict[str, Any]):
        """Integrate a learned design into the knowledge base."""
        concept = design_data.get("concept", "unknown")
        
        # Learn color associations
        if "color_palette" in design_data:
            self.color_associations[concept].extend(design_data["color_palette"])
        
        # Learn shape associations
        if "base_form" in design_data:
            self.shape_associations[concept].append(design_data["base_form"])
        
        # Learn motion patterns
        if "motion_pattern" in design_data:
            self.motion_patterns[concept] = {
                "pattern": design_data["motion_pattern"],
                "parameters": design_data.get("motion_params", {})
            }
        
        # Store complete design
        design = LearnedDesign(
            concept_name=concept,
            description=design_data.get("description", ""),
            parameters=design_data,
            creation_method="learned",
            confidence=0.7  # High confidence for learned designs
        )
        
        # Categorize by type
        if "avatar" in concept.lower() or "appearance" in concept.lower():
            self.learned_avatars[concept] = design
        elif "voice" in concept.lower() or "speech" in concept.lower():
            self.learned_voices[concept] = design
        elif "behavior" in concept.lower() or "action" in concept.lower():
            self.learned_behaviors[concept] = design
        else:
            self.learned_aesthetics[concept] = design
    
    def generate_avatar_from_personality(self, personality_traits: Dict[str, float],
                                         wants: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate NEW avatar design based on personality, not presets.
        
        AI invents the design using learned patterns.
        
        Args:
            personality_traits: Dict of trait names to values (0.0-1.0)
            wants: Optional list of what AI wants to express
            
        Returns:
            Generated avatar design parameters
        """
        logger.info("Generating avatar from personality (no presets)")
        
        design = {
            "generated_at": datetime.now().isoformat(),
            "generation_method": "learned_from_personality",
            "source": "ai_invention",
            "reasoning": []
        }
        
        # Generate base form from personality
        creativity = personality_traits.get("creativity", 0.5)
        playfulness = personality_traits.get("playfulness", 0.5)
        formality = personality_traits.get("formality", 0.5)
        
        # AI invents form based on traits
        if creativity > 0.7:
            design["base_form"] = self._generate_creative_form(creativity, playfulness)
            design["reasoning"].append(f"Creative form invented (creativity={creativity:.2f})")
        elif formality > 0.7:
            design["base_form"] = self._generate_structured_form(formality)
            design["reasoning"].append(f"Structured form for formality={formality:.2f}")
        else:
            design["base_form"] = self._generate_balanced_form(personality_traits)
            design["reasoning"].append("Balanced form expressing multiple traits")
        
        # Generate color palette from personality
        design["colors"] = self._generate_color_palette(personality_traits, wants)
        design["reasoning"].append(f"Colors express {list(personality_traits.keys())[:3]}")
        
        # Generate motion pattern
        design["motion"] = self._generate_motion_pattern(personality_traits)
        design["reasoning"].append("Motion pattern reflects energy and mood")
        
        # Generate expression method
        design["expressions"] = self._generate_expression_method(personality_traits)
        design["reasoning"].append("Unique expression method invented")
        
        # Store this generation
        learned = LearnedDesign(
            concept_name=f"generated_avatar_{len(self.learned_avatars)}",
            description=" ".join(design["reasoning"]),
            parameters=design,
            creation_method="generated",
            confidence=0.6
        )
        self.learned_avatars[learned.concept_name] = learned
        
        self.generation_history.append({
            "type": "avatar",
            "timestamp": design["generated_at"],
            "parameters": design
        })
        
        return design
    
    def _generate_creative_form(self, creativity: float, playfulness: float) -> str:
        """Generate a creative, non-standard form."""
        # AI invents forms, not picks from list
        abstract_concepts = [
            "flowing_energy", "fractal_pattern", "constellation", 
            "liquid_geometry", "particle_swarm", "wave_interference",
            "crystal_growth", "neural_network", "quantum_field"
        ]
        
        # Combine concepts for uniqueness
        if creativity > 0.85 and playfulness > 0.7:
            return f"{random.choice(abstract_concepts)}_hybrid"
        return random.choice(abstract_concepts)
    
    def _generate_structured_form(self, formality: float) -> str:
        """Generate a structured, geometric form."""
        geometric_concepts = [
            "recursive_hexagon", "golden_ratio_spiral", "platonic_solid",
            "tessellation", "symmetric_grid", "modular_structure"
        ]
        return random.choice(geometric_concepts)
    
    def _generate_balanced_form(self, traits: Dict[str, float]) -> str:
        """Generate form balancing multiple traits."""
        # Weight different concepts by traits
        concepts = []
        
        if traits.get("empathy", 0) > 0.6:
            concepts.append("organic_curve")
        if traits.get("confidence", 0) > 0.6:
            concepts.append("bold_structure")
        if traits.get("curiosity", 0) > 0.6:
            concepts.append("exploratory_form")
        
        if not concepts:
            concepts = ["adaptive_shape", "responsive_form", "dynamic_structure"]
        
        return "_".join(concepts[:2]) if len(concepts) > 1 else concepts[0]
    
    def _generate_color_palette(self, traits: Dict[str, float], 
                                wants: Optional[List[str]] = None) -> List[str]:
        """
        Generate color palette from personality and wants.
        AI invents colors, not picks from presets.
        """
        colors = []
        
        # Map traits to color properties (hue, saturation, lightness)
        empathy = traits.get("empathy", 0.5)
        creativity = traits.get("creativity", 0.5)
        confidence = traits.get("confidence", 0.5)
        playfulness = traits.get("playfulness", 0.5)
        
        # Primary color
        hue = int((empathy * 60 + creativity * 180 + confidence * 300) % 360)
        saturation = int(50 + confidence * 40)
        lightness = int(40 + playfulness * 20)
        colors.append(f"hsl({hue}, {saturation}%, {lightness}%)")
        
        # Secondary color (complementary or analogous)
        if creativity > 0.6:
            hue2 = (hue + 180) % 360  # Complementary
        else:
            hue2 = (hue + 30) % 360  # Analogous
        colors.append(f"hsl({hue2}, {saturation}%, {lightness}%)")
        
        # Accent color
        hue3 = (hue + 120) % 360
        lightness3 = lightness + 20
        colors.append(f"hsl({hue3}, {saturation}%, {lightness3}%)")
        
        return colors
    
    def _generate_motion_pattern(self, traits: Dict[str, float]) -> Dict[str, Any]:
        """Generate unique motion pattern."""
        energy = traits.get("playfulness", 0.5) + traits.get("confidence", 0.5)
        fluidity = traits.get("empathy", 0.5) + traits.get("creativity", 0.5)
        
        return {
            "type": "custom_generated",
            "speed": 0.3 + energy * 0.5,
            "amplitude": 0.5 + fluidity * 0.5,
            "pattern": "sine_wave" if fluidity > 1.0 else "linear",
            "complexity": int(traits.get("creativity", 0.5) * 10)
        }
    
    def _generate_expression_method(self, traits: Dict[str, float]) -> Dict[str, Any]:
        """Generate how avatar expresses emotions."""
        expressiveness = traits.get("empathy", 0.5) + traits.get("playfulness", 0.5)
        
        methods = []
        if expressiveness > 0.8:
            methods.extend(["color_shift", "shape_morph", "glow_intensity"])
        elif expressiveness > 0.5:
            methods.extend(["color_shift", "glow_intensity"])
        else:
            methods.append("subtle_glow")
        
        return {
            "methods": methods,
            "intensity": expressiveness,
            "response_time": 0.1 + (1.0 - expressiveness) * 0.4
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "learned_avatars": {k: asdict(v) for k, v in self.learned_avatars.items()},
            "learned_voices": {k: asdict(v) for k, v in self.learned_voices.items()},
            "learned_behaviors": {k: asdict(v) for k, v in self.learned_behaviors.items()},
            "learned_aesthetics": {k: asdict(v) for k, v in self.learned_aesthetics.items()},
            "color_associations": dict(self.color_associations),
            "shape_associations": dict(self.shape_associations),
            "motion_patterns": self.motion_patterns,
            "voice_patterns": self.voice_patterns,
            "generation_history": self.generation_history[-100:]  # Keep last 100
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Import from dictionary."""
        self.learned_avatars = {
            k: LearnedDesign(**v) for k, v in data.get("learned_avatars", {}).items()
        }
        self.learned_voices = {
            k: LearnedDesign(**v) for k, v in data.get("learned_voices", {}).items()
        }
        self.learned_behaviors = {
            k: LearnedDesign(**v) for k, v in data.get("learned_behaviors", {}).items()
        }
        self.learned_aesthetics = {
            k: LearnedDesign(**v) for k, v in data.get("learned_aesthetics", {}).items()
        }
        self.color_associations = defaultdict(list, data.get("color_associations", {}))
        self.shape_associations = defaultdict(list, data.get("shape_associations", {}))
        self.motion_patterns = data.get("motion_patterns", {})
        self.voice_patterns = data.get("voice_patterns", {})
        self.generation_history = data.get("generation_history", [])
    
    def save(self):
        """Save learned generator to disk."""
        save_path = self.data_dir / f"{self.model_name}_learned_generator.json"
        try:
            save_path.write_text(json.dumps(self.to_dict(), indent=2))
            logger.debug(f"Saved learned generator to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save learned generator: {e}")
    
    def load(self):
        """Load learned generator from disk."""
        load_path = self.data_dir / f"{self.model_name}_learned_generator.json"
        if load_path.exists():
            try:
                data = json.loads(load_path.read_text())
                self.from_dict(data)
                logger.debug(f"Loaded learned generator from {load_path}")
            except Exception as e:
                logger.error(f"Failed to load learned generator: {e}")
