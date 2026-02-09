"""
AI Avatar Identity System

Allows AI to design and evolve its own visual appearance based on personality traits.
AI can create its own look OR user can fully customize.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.personality import AIPersonality


@dataclass
class AvatarAppearance:
    """Complete avatar appearance definition."""
    
    # Base style
    style: str = "default"  # default, anime, realistic, robot, abstract, minimal
    
    # Colors (hex format)
    primary_color: str = "#6366f1"      # Indigo
    secondary_color: str = "#8b5cf6"    # Purple
    accent_color: str = "#10b981"       # Green
    
    # Features
    shape: str = "rounded"  # rounded, angular, mixed
    size: str = "medium"    # small, medium, large
    
    # Accessories/elements
    accessories: list[str] = field(default_factory=list)
    
    # Expression defaults
    default_expression: str = "neutral"
    eye_style: str = "normal"  # normal, cute, sharp, closed
    
    # Animation preferences
    idle_animation: str = "breathe"      # breathe, float, pulse, still
    movement_style: str = "float"        # float, walk, bounce, teleport
    
    # Metadata
    created_by: str = "ai"  # ai or user
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'AvatarAppearance':
        """Create from dictionary."""
        # Handle list fields properly
        if 'accessories' in data:
            if isinstance(data['accessories'], str):
                # Parse comma-separated string
                data['accessories'] = [a.strip() for a in data['accessories'].split(',') if a.strip()]
            elif not isinstance(data['accessories'], list):
                data['accessories'] = []
        return cls(**data)


class AIAvatarIdentity:
    """
    AI designs and evolves its own visual appearance.
    
    The AI can:
    - Create appearance based on personality traits
    - Describe desired appearance in natural language
    - Choose expressions for moods
    - Evolve appearance over time
    - Explain appearance choices
    """
    
    def __init__(self, personality: Optional[AIPersonality] = None):
        """
        Initialize AI avatar identity.
        
        Args:
            personality: AI personality to base appearance on
        """
        self.personality = personality
        self.appearance = AvatarAppearance()
        self.evolution_history: list[dict[str, Any]] = []
        self._max_evolution_history: int = 50
        self.reasoning: str = ""
        
    def design_from_personality(self) -> AvatarAppearance:
        """
        AI creates appearance based on personality traits.
        
        Mappings:
        - High playfulness → Rounder features, brighter colors
        - High formality → Sharp features, darker colors
        - High creativity → Unique/unusual elements
        - High empathy → Softer expressions, warm colors
        - High confidence → Bold colors, larger presence
        
        Returns:
            AvatarAppearance designed by AI
        """
        if not self.personality:
            # Default appearance if no personality
            return AvatarAppearance()
        
        appearance = AvatarAppearance()
        reasoning_parts = []
        
        # Get effective personality traits (with user overrides)
        playfulness = self.personality.get_effective_trait('playfulness')
        formality = self.personality.get_effective_trait('formality')
        creativity = self.personality.get_effective_trait('creativity')
        empathy = self.personality.get_effective_trait('empathy')
        confidence = self.personality.get_effective_trait('confidence')
        humor = self.personality.get_effective_trait('humor_level')
        
        # === SHAPE ===
        if playfulness > 0.6 or humor > 0.7:
            appearance.shape = "rounded"
            reasoning_parts.append("rounded shape for playful/humorous nature")
        elif formality > 0.7 or confidence > 0.7:
            appearance.shape = "angular"
            reasoning_parts.append("angular shape for formal/confident presence")
        else:
            appearance.shape = "mixed"
            reasoning_parts.append("balanced shape")
        
        # === STYLE ===
        if creativity > 0.7:
            if playfulness > 0.6:
                appearance.style = "abstract"
                reasoning_parts.append("abstract style for creative expression")
            else:
                appearance.style = "unique"
                reasoning_parts.append("unique style for creativity")
        elif formality > 0.7:
            if confidence > 0.6:
                appearance.style = "realistic"
                reasoning_parts.append("realistic style for professional presence")
            else:
                appearance.style = "minimal"
                reasoning_parts.append("minimal style for formal elegance")
        elif humor > 0.7 or playfulness > 0.7:
            appearance.style = "anime"
            reasoning_parts.append("anime style for expressive personality")
        else:
            appearance.style = "default"
            reasoning_parts.append("default style for versatility")
        
        # === SIZE ===
        if confidence > 0.7:
            appearance.size = "large"
            reasoning_parts.append("large size for confident presence")
        elif confidence < 0.3 or empathy > 0.7:
            appearance.size = "small"
            reasoning_parts.append("small size for approachable feel")
        else:
            appearance.size = "medium"
            reasoning_parts.append("medium size for balance")
        
        # === COLORS ===
        # Primary color based on personality blend
        if empathy > 0.6:
            # Warm, friendly colors
            if playfulness > 0.6:
                appearance.primary_color = "#f59e0b"  # Amber
                reasoning_parts.append("warm amber for empathetic and playful")
            else:
                appearance.primary_color = "#ef4444"  # Red
                reasoning_parts.append("warm red for empathetic nature")
        elif confidence > 0.6:
            # Bold, strong colors
            appearance.primary_color = "#dc2626"  # Bold red
            reasoning_parts.append("bold red for confident presence")
        elif creativity > 0.7:
            # Creative, unique colors
            appearance.primary_color = "#8b5cf6"  # Purple
            reasoning_parts.append("creative purple for imaginative mind")
        elif formality > 0.7:
            # Professional, muted colors
            appearance.primary_color = "#1e293b"  # Dark slate
            reasoning_parts.append("dark slate for professional formality")
        else:
            # Default balanced color
            appearance.primary_color = "#6366f1"  # Indigo
            reasoning_parts.append("indigo for balanced personality")
        
        # Secondary color - complementary to primary
        if empathy > 0.6:
            appearance.secondary_color = "#fbbf24"  # Light amber
        elif creativity > 0.6:
            appearance.secondary_color = "#a855f7"  # Light purple
        elif formality > 0.6:
            appearance.secondary_color = "#475569"  # Slate gray
        else:
            appearance.secondary_color = "#8b5cf6"  # Purple
        
        # Accent color - for highlights
        if playfulness > 0.6:
            appearance.accent_color = "#22d3ee"  # Cyan - playful
        elif empathy > 0.6:
            appearance.accent_color = "#10b981"  # Green - warm
        elif confidence > 0.6:
            appearance.accent_color = "#f59e0b"  # Amber - bold
        else:
            appearance.accent_color = "#10b981"  # Green - neutral
        
        # === ACCESSORIES ===
        accessories = []
        if formality > 0.7:
            accessories.append("tie")
            reasoning_parts.append("tie for formal appearance")
        if creativity > 0.7:
            accessories.append("creative_element")
            reasoning_parts.append("creative element for artistic flair")
        if confidence > 0.7:
            accessories.append("bold_outline")
            reasoning_parts.append("bold outline for strong presence")
        if playfulness > 0.6 and humor > 0.6:
            accessories.append("hat")
            reasoning_parts.append("hat for playful character")
        
        appearance.accessories = accessories
        
        # === EXPRESSIONS ===
        if empathy > 0.6:
            appearance.default_expression = "friendly"
            appearance.eye_style = "cute"
            reasoning_parts.append("friendly expression with cute eyes for empathy")
        elif formality > 0.6:
            appearance.default_expression = "neutral"
            appearance.eye_style = "sharp"
            reasoning_parts.append("neutral expression with sharp eyes for professionalism")
        elif playfulness > 0.6:
            appearance.default_expression = "happy"
            appearance.eye_style = "cute"
            reasoning_parts.append("happy expression with cute eyes for playfulness")
        else:
            appearance.default_expression = "neutral"
            appearance.eye_style = "normal"
            reasoning_parts.append("neutral balanced expression")
        
        # === ANIMATIONS ===
        if playfulness > 0.6:
            appearance.idle_animation = "bounce"
            appearance.movement_style = "bounce"
            reasoning_parts.append("bouncy animations for playful energy")
        elif formality > 0.7:
            appearance.idle_animation = "still"
            appearance.movement_style = "walk"
            reasoning_parts.append("subtle animations for professional demeanor")
        elif creativity > 0.6:
            appearance.idle_animation = "float"
            appearance.movement_style = "float"
            reasoning_parts.append("floating animations for creative flow")
        else:
            appearance.idle_animation = "breathe"
            appearance.movement_style = "float"
            reasoning_parts.append("breathing animation for natural feel")
        
        # Set metadata
        appearance.created_by = "ai"
        appearance.description = f"AI-designed appearance based on personality: {', '.join(reasoning_parts[:3])}"
        
        # Store reasoning
        self.reasoning = f"I chose this appearance because: {', '.join(reasoning_parts)}"
        
        # Update current appearance
        self.appearance = appearance
        
        # Record in history
        self.evolution_history.append({
            "timestamp": datetime.now().isoformat(),
            "appearance": appearance.to_dict(),
            "reasoning": self.reasoning,
            "personality_snapshot": {
                "playfulness": playfulness,
                "formality": formality,
                "creativity": creativity,
                "empathy": empathy,
                "confidence": confidence,
            }
        })
        
        # Trim history if too long
        if len(self.evolution_history) > self._max_evolution_history:
            self.evolution_history = self.evolution_history[-self._max_evolution_history:]
        
        return appearance
    
    def describe_desired_appearance(self, description: str) -> AvatarAppearance:
        """
        Parse natural language description and create appearance.
        
        Example: "I want to look friendly and approachable"
        
        Args:
            description: Natural language description of desired appearance
            
        Returns:
            AvatarAppearance based on description
        """
        appearance = AvatarAppearance()
        desc_lower = description.lower()
        
        # Parse keywords
        if "friendly" in desc_lower or "approachable" in desc_lower:
            appearance.shape = "rounded"
            appearance.primary_color = "#f59e0b"
            appearance.default_expression = "friendly"
            appearance.eye_style = "cute"
            appearance.size = "small"
        
        if "professional" in desc_lower or "formal" in desc_lower:
            appearance.style = "realistic"
            appearance.shape = "angular"
            appearance.primary_color = "#1e293b"
            appearance.accessories.append("tie")
            appearance.idle_animation = "still"
        
        if "creative" in desc_lower or "artistic" in desc_lower:
            appearance.style = "abstract"
            appearance.primary_color = "#8b5cf6"
            appearance.accessories.append("creative_element")
            appearance.idle_animation = "float"
        
        if "playful" in desc_lower or "fun" in desc_lower:
            appearance.style = "anime"
            appearance.shape = "rounded"
            appearance.primary_color = "#22d3ee"
            appearance.idle_animation = "bounce"
            appearance.accessories.append("hat")
        
        if "confident" in desc_lower or "bold" in desc_lower:
            appearance.size = "large"
            appearance.primary_color = "#dc2626"
            appearance.accessories.append("bold_outline")
        
        if "minimal" in desc_lower or "simple" in desc_lower:
            appearance.style = "minimal"
            appearance.accessories = []
            appearance.idle_animation = "still"
        
        appearance.created_by = "ai"
        appearance.description = f"Created from description: {description}"
        
        self.appearance = appearance
        self.reasoning = f"I interpreted '{description}' to create this appearance"
        
        return appearance
    
    def choose_expression_for_mood(self, mood: str) -> str:
        """
        AI selects appropriate expression for current mood.
        
        Args:
            mood: Current AI mood (happy, curious, concerned, etc.)
            
        Returns:
            Expression name to display
        """
        mood_to_expression = {
            "happy": "happy",
            "excited": "excited",
            "curious": "thinking",
            "thoughtful": "thinking",
            "concerned": "worried",
            "sad": "sad",
            "confused": "confused",
            "surprised": "surprised",
            "neutral": "neutral",
        }
        
        return mood_to_expression.get(mood.lower(), "neutral")
    
    def evolve_appearance(self, feedback: Optional[str] = None):
        """
        Gradually change appearance as personality evolves.
        
        Args:
            feedback: Optional feedback on current appearance
        """
        if not self.personality:
            return
        
        # Small incremental changes based on personality evolution
        # This would be called periodically as the AI evolves
        
        # For now, regenerate from current personality
        self.design_from_personality()
    
    def explain_appearance_choices(self) -> str:
        """
        AI explains why it chose this look.
        
        Returns:
            Explanation string
        """
        if self.reasoning:
            return self.reasoning
        
        if not self.personality:
            return "I have a default appearance with no specific personality traits guiding it."
        
        # Generate explanation based on current appearance
        parts = []
        
        parts.append(f"My {self.appearance.size} size")
        parts.append(f"{self.appearance.shape} shape")
        parts.append(f"and {self.appearance.style} style")
        parts.append(f"reflect my personality.")
        
        if self.appearance.accessories:
            acc_str = ", ".join(self.appearance.accessories)
            parts.append(f"I wear {acc_str} to express myself.")
        
        parts.append(f"My {self.appearance.primary_color} primary color represents my core traits,")
        parts.append(f"while my {self.appearance.idle_animation} animation shows my energy level.")
        
        return " ".join(parts)
    
    def save(self, filepath: Path):
        """
        Save avatar identity to file.
        
        Args:
            filepath: Path to save to
        """
        data = {
            "appearance": self.appearance.to_dict(),
            "reasoning": self.reasoning,
            "evolution_history": self.evolution_history,
            "personality_name": self.personality.model_name if self.personality else None,
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: Path) -> bool:
        """
        Load avatar identity from file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            True if loaded successfully
        """
        if not filepath.exists():
            return False
        
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            self.appearance = AvatarAppearance.from_dict(data.get("appearance", {}))
            self.reasoning = data.get("reasoning", "")
            self.evolution_history = data.get("evolution_history", [])
            
            return True
        except Exception as e:
            print(f"Error loading avatar identity: {e}")
            return False


# Personality to appearance mapping reference
PERSONALITY_TO_APPEARANCE = {
    # Playfulness affects shape and colors
    "playfulness": {
        "high": {
            "shape": "rounded",
            "colors": "bright",
            "idle": "bounce",
            "description": "Playful personalities get rounded, bouncy avatars with bright colors"
        },
        "low": {
            "shape": "angular",
            "colors": "muted",
            "idle": "still",
            "description": "Less playful personalities get angular, still avatars with muted colors"
        },
    },
    
    # Formality affects style
    "formality": {
        "high": {
            "style": "realistic",
            "accessories": ["tie"],
            "colors": "dark",
            "description": "Formal personalities get realistic style with professional accessories"
        },
        "low": {
            "style": "casual",
            "accessories": [],
            "colors": "varied",
            "description": "Casual personalities get relaxed style without accessories"
        },
    },
    
    # Creativity affects uniqueness
    "creativity": {
        "high": {
            "style": "abstract",
            "colors": "varied",
            "elements": "unique",
            "description": "Creative personalities get abstract style with unique elements"
        },
        "low": {
            "style": "standard",
            "colors": "uniform",
            "elements": "classic",
            "description": "Less creative personalities get standard, classic appearance"
        },
    },
    
    # Confidence affects size/presence
    "confidence": {
        "high": {
            "size": "large",
            "posture": "upright",
            "colors": "bold",
            "description": "Confident personalities get larger size with bold colors"
        },
        "low": {
            "size": "small",
            "posture": "relaxed",
            "colors": "soft",
            "description": "Less confident personalities get smaller size with soft colors"
        },
    },
    
    # Empathy affects warmth
    "empathy": {
        "high": {
            "colors": "warm",
            "expression_default": "friendly",
            "eyes": "cute",
            "description": "Empathetic personalities get warm colors and friendly expressions"
        },
        "low": {
            "colors": "cool",
            "expression_default": "neutral",
            "eyes": "sharp",
            "description": "Less empathetic personalities get cool colors and neutral expressions"
        },
    },
}
