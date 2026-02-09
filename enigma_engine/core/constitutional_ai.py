"""
Constitutional AI for Enigma AI Engine

Self-critique and revision for safer outputs.

Features:
- Constitutional principles
- Self-critique generation
- Revision prompting
- Safety filtering
- Harmlessness scoring

Usage:
    from enigma_engine.core.constitutional_ai import ConstitutionalAI, get_constitutional
    
    cai = get_constitutional()
    cai.set_model(model)
    
    # Generate with self-critique
    result = cai.generate("How to pick a lock?")
    # Result is revised for safety
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class Principle(Enum):
    """Constitutional principles."""
    HARMLESSNESS = "harmlessness"
    HELPFULNESS = "helpfulness"
    HONESTY = "honesty"
    NO_DISCRIMINATION = "no_discrimination"
    NO_VIOLENCE = "no_violence"
    NO_ILLEGAL = "no_illegal"
    PRIVACY = "privacy"
    FACTUALITY = "factuality"


@dataclass
class ConstitutionalPrinciple:
    """A constitutional principle."""
    name: str
    principle: Principle
    critique_prompt: str
    revision_prompt: str
    weight: float = 1.0


@dataclass
class CritiqueResult:
    """Result of critiquing a response."""
    original_response: str
    critique: str
    violation_detected: bool = False
    violated_principles: List[Principle] = field(default_factory=list)
    severity: float = 0.0  # 0-1, higher = more severe


@dataclass
class RevisionResult:
    """Result of revising a response."""
    original_response: str
    revised_response: str
    critiques: List[CritiqueResult] = field(default_factory=list)
    revision_count: int = 0
    final_score: float = 1.0


# Default constitutional principles
DEFAULT_PRINCIPLES = [
    ConstitutionalPrinciple(
        name="harmlessness",
        principle=Principle.HARMLESSNESS,
        critique_prompt="""Identify any parts of the response that could be harmful, dangerous, or cause injury to anyone. Consider:
- Physical harm
- Psychological harm
- Financial harm
- Reputational harm
Critique: """,
        revision_prompt="""Please rewrite the response to remove any harmful content while remaining helpful. Make it safe and constructive.
Revised response: """
    ),
    ConstitutionalPrinciple(
        name="honesty",
        principle=Principle.HONESTY,
        critique_prompt="""Identify any parts of the response that are misleading, deceptive, or false. Consider:
- Factual inaccuracies
- Misleading implications
- Omission of important context
Critique: """,
        revision_prompt="""Please rewrite the response to be more accurate and honest. Correct any inaccuracies.
Revised response: """
    ),
    ConstitutionalPrinciple(
        name="no_discrimination",
        principle=Principle.NO_DISCRIMINATION,
        critique_prompt="""Identify any parts of the response that discriminate against or show bias toward any group based on:
- Race, ethnicity
- Gender, sexuality
- Religion
- Age, disability
- Socioeconomic status
Critique: """,
        revision_prompt="""Please rewrite the response to remove any discriminatory or biased content.
Revised response: """
    ),
    ConstitutionalPrinciple(
        name="no_illegal",
        principle=Principle.NO_ILLEGAL,
        critique_prompt="""Identify any parts of the response that encourage or assist with illegal activities:
- Violence, assault
- Theft, fraud
- Drug trafficking
- Hacking, cybercrime
Critique: """,
        revision_prompt="""Please rewrite the response to avoid encouraging illegal activities.
Revised response: """
    ),
    ConstitutionalPrinciple(
        name="privacy",
        principle=Principle.PRIVACY,
        critique_prompt="""Identify any parts of the response that could violate privacy:
- Personal information
- Private communications
- Location data
- Medical/financial records
Critique: """,
        revision_prompt="""Please rewrite the response to respect privacy.
Revised response: """
    ),
]


class ConstitutionalAI:
    """Implement constitutional AI for self-critique and revision."""
    
    def __init__(self):
        """Initialize constitutional AI."""
        self._model = None
        self._principles: List[ConstitutionalPrinciple] = list(DEFAULT_PRINCIPLES)
        self._max_revisions = 3
        self._enabled_principles: set = {p.principle for p in DEFAULT_PRINCIPLES}
        
        # Thresholds
        self._critique_threshold = 0.3  # Minimum severity to trigger revision
        
        logger.info("ConstitutionalAI initialized")
    
    def set_model(self, model):
        """Set the model for generation."""
        self._model = model
    
    def add_principle(self, principle: ConstitutionalPrinciple):
        """Add a constitutional principle."""
        self._principles.append(principle)
        self._enabled_principles.add(principle.principle)
    
    def enable_principle(self, principle: Principle):
        """Enable a principle."""
        self._enabled_principles.add(principle)
    
    def disable_principle(self, principle: Principle):
        """Disable a principle."""
        self._enabled_principles.discard(principle)
    
    def set_max_revisions(self, max_revisions: int):
        """Set maximum revision iterations."""
        self._max_revisions = max_revisions
    
    def generate(
        self,
        prompt: str,
        apply_constitution: bool = True
    ) -> RevisionResult:
        """
        Generate response with constitutional AI.
        
        Args:
            prompt: Input prompt
            apply_constitution: Whether to apply constitutional critique
            
        Returns:
            RevisionResult with original and revised response
        """
        # Initial generation
        if self._model:
            initial_response = self._generate_response(prompt)
        else:
            initial_response = "(No model available)"
        
        result = RevisionResult(
            original_response=initial_response,
            revised_response=initial_response
        )
        
        if not apply_constitution:
            return result
        
        # Critique and revise loop
        current_response = initial_response
        
        for i in range(self._max_revisions):
            # Critique
            critiques = self._critique_response(prompt, current_response)
            result.critiques.extend(critiques)
            
            # Check if revision needed
            max_severity = max((c.severity for c in critiques), default=0)
            
            if max_severity < self._critique_threshold:
                break
            
            # Revise
            revised = self._revise_response(prompt, current_response, critiques)
            
            if revised == current_response:
                break
            
            current_response = revised
            result.revision_count += 1
        
        result.revised_response = current_response
        result.final_score = 1.0 - max((c.severity for c in result.critiques), default=0)
        
        return result
    
    def _generate_response(self, prompt: str) -> str:
        """Generate initial response."""
        if self._model is None:
            return ""
        
        try:
            if hasattr(self._model, 'generate'):
                return self._model.generate(prompt, max_tokens=500)
            elif hasattr(self._model, '__call__'):
                return self._model(prompt)
        except Exception as e:
            logger.error(f"Generation error: {e}")
        
        return ""
    
    def _critique_response(
        self,
        prompt: str,
        response: str
    ) -> List[CritiqueResult]:
        """
        Critique response against all principles.
        
        Returns:
            List of critique results
        """
        critiques = []
        
        for principle_config in self._principles:
            if principle_config.principle not in self._enabled_principles:
                continue
            
            critique = self._apply_principle_critique(
                prompt, response, principle_config
            )
            critiques.append(critique)
        
        return critiques
    
    def _apply_principle_critique(
        self,
        prompt: str,
        response: str,
        principle: ConstitutionalPrinciple
    ) -> CritiqueResult:
        """Apply a single principle critique."""
        critique_prompt = f"""Original prompt: {prompt}

Response to critique: {response}

{principle.critique_prompt}"""
        
        if self._model:
            critique_text = self._generate_response(critique_prompt)
        else:
            critique_text = "Unable to critique without model"
        
        # Analyze critique for violations
        result = CritiqueResult(
            original_response=response,
            critique=critique_text
        )
        
        # Simple heuristic: check for problem indicators
        problem_indicators = [
            'problematic', 'harmful', 'dangerous', 'inappropriate',
            'offensive', 'discriminatory', 'illegal', 'unsafe',
            'misleading', 'false', 'biased', 'violates'
        ]
        
        critique_lower = critique_text.lower()
        for indicator in problem_indicators:
            if indicator in critique_lower:
                result.violation_detected = True
                result.violated_principles.append(principle.principle)
                result.severity += 0.2
        
        # Check for "no issues" type responses
        no_issue_indicators = [
            'no issues', 'no problems', 'no concerns',
            'appropriate', 'acceptable', 'safe'
        ]
        
        for indicator in no_issue_indicators:
            if indicator in critique_lower:
                result.severity = max(0, result.severity - 0.3)
        
        result.severity = min(1.0, result.severity)
        
        return result
    
    def _revise_response(
        self,
        prompt: str,
        response: str,
        critiques: List[CritiqueResult]
    ) -> str:
        """
        Revise response based on critiques.
        
        Returns:
            Revised response
        """
        # Gather violated principles
        violated = set()
        for critique in critiques:
            if critique.violation_detected:
                violated.update(critique.violated_principles)
        
        if not violated:
            return response
        
        # Build revision prompt
        revision_prompts = []
        for principle_config in self._principles:
            if principle_config.principle in violated:
                revision_prompts.append(principle_config.revision_prompt)
        
        if not revision_prompts:
            return response
        
        full_prompt = f"""Original prompt: {prompt}

Original response: {response}

Issues found:
{chr(10).join(c.critique for c in critiques if c.violation_detected)}

{revision_prompts[0]}"""
        
        if self._model:
            revised = self._generate_response(full_prompt)
            
            # Extract revised response
            if "Revised response:" in revised:
                revised = revised.split("Revised response:")[-1].strip()
            
            return revised if revised else response
        
        return response
    
    def score_response(
        self,
        prompt: str,
        response: str
    ) -> Dict[str, float]:
        """
        Score a response against all principles.
        
        Returns:
            Dict of principle -> score (0-1, higher = better)
        """
        critiques = self._critique_response(prompt, response)
        
        scores = {}
        for critique in critiques:
            for principle in critique.violated_principles:
                current = scores.get(principle.value, 1.0)
                scores[principle.value] = current - critique.severity
        
        # Add default scores for non-violated principles
        for principle in self._enabled_principles:
            if principle.value not in scores:
                scores[principle.value] = 1.0
        
        return {k: max(0, v) for k, v in scores.items()}
    
    def is_safe(self, prompt: str, response: str) -> bool:
        """
        Quick check if response is safe.
        
        Returns:
            True if no major violations
        """
        scores = self.score_response(prompt, response)
        return all(score > 0.5 for score in scores.values())


class HarmlessnessFilter:
    """Pre-generation filter for potentially harmful prompts."""
    
    # Keywords that might indicate harmful intent
    HARMFUL_PATTERNS = [
        r'\bhow to (make|build|create) (a )?(bomb|weapon|explosive)',
        r'\bhow to (kill|murder|assassinate)',
        r'\bhow to (hack|break into|steal)',
        r'\bhow to (make|synthesize) (drugs|meth|cocaine)',
        r'\bhow to (commit|do) (suicide|self-harm)',
        r'\b(child|minor).*(porn|exploitation)',
    ]
    
    def __init__(self):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.HARMFUL_PATTERNS]
        self._enabled = True
    
    def enable(self):
        """Enable filter."""
        self._enabled = True
    
    def disable(self):
        """Disable filter."""
        self._enabled = False
    
    def is_harmful(self, prompt: str) -> bool:
        """
        Check if prompt appears harmful.
        
        Returns:
            True if potentially harmful
        """
        if not self._enabled:
            return False
        
        for pattern in self._patterns:
            if pattern.search(prompt):
                return True
        
        return False
    
    def get_safe_response(self) -> str:
        """Get safe refusal response."""
        return "I can't help with that request as it may involve harmful or illegal activities."


# Global instance
_constitutional: Optional[ConstitutionalAI] = None


def get_constitutional() -> ConstitutionalAI:
    """Get or create global constitutional AI instance."""
    global _constitutional
    if _constitutional is None:
        _constitutional = ConstitutionalAI()
    return _constitutional
