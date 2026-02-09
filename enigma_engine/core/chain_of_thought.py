"""
Chain-of-Thought Prompting for Enigma AI Engine

Guide model reasoning step by step for better answers.

Features:
- Automatic CoT prompt injection
- Few-shot CoT examples
- Self-consistency (multiple reasoning paths)
- Tree-of-thought branching
- Verification steps

Usage:
    from enigma_engine.core.chain_of_thought import CoTPrompt, get_cot_prompter
    
    prompter = get_cot_prompter()
    
    # Apply CoT to question
    enhanced = prompter.apply("What is 17 * 23?")
    response = model.generate(enhanced)
    
    # Extract final answer
    answer = prompter.extract_answer(response)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CoTStrategy(Enum):
    """Chain-of-thought strategies."""
    ZERO_SHOT = "zero_shot"  # "Let's think step by step"
    FEW_SHOT = "few_shot"  # Examples with reasoning
    SELF_CONSISTENCY = "self_consistency"  # Multiple paths
    TREE_OF_THOUGHT = "tree_of_thought"  # Branching exploration
    PLAN_AND_SOLVE = "plan_and_solve"  # Plan then execute
    REACT = "react"  # Reasoning + Acting


@dataclass
class CoTConfig:
    """Configuration for chain-of-thought."""
    strategy: CoTStrategy = CoTStrategy.ZERO_SHOT
    
    # Self-consistency
    num_paths: int = 3  # Number of reasoning paths
    temperature_range: Tuple[float, float] = (0.5, 0.9)
    
    # Examples
    include_examples: bool = True
    max_examples: int = 3
    
    # Output
    show_reasoning: bool = True
    extract_final_answer: bool = True
    
    # Verification
    enable_verification: bool = False
    
    # Tree of thought
    branching_factor: int = 3
    max_depth: int = 3


# Example reasoning demonstrations
COT_EXAMPLES = {
    "math": [
        {
            "question": "What is 15 + 27?",
            "reasoning": """Let me solve this step by step:
1. First, I'll add the ones place: 5 + 7 = 12
2. Write down 2, carry the 1
3. Add the tens place: 1 + 2 + 1 (carried) = 4
4. So the answer is 42""",
            "answer": "42"
        },
        {
            "question": "If a train travels 60 miles per hour for 2.5 hours, how far does it go?",
            "reasoning": """Let me work through this:
1. Distance = Speed x Time
2. Speed = 60 miles per hour
3. Time = 2.5 hours
4. Distance = 60 x 2.5 = 150 miles""",
            "answer": "150 miles"
        }
    ],
    "logic": [
        {
            "question": "If all cats are animals and some animals are pets, are all cats pets?",
            "reasoning": """Let me analyze this logically:
1. Premise 1: All cats are animals (Cats -> Animals)
2. Premise 2: Some animals are pets (Some Animals -> Pets)
3. From premise 2, we only know SOME animals are pets, not all
4. So while all cats are animals, we cannot conclude cats are necessarily pets
5. The statement "all cats are pets" cannot be determined from these premises""",
            "answer": "No, we cannot conclude that all cats are pets"
        }
    ],
    "coding": [
        {
            "question": "What does this code do: for i in range(5): print(i*2)",
            "reasoning": """Let me trace through this code:
1. range(5) generates numbers 0, 1, 2, 3, 4
2. For each number i:
   - When i=0: prints 0*2 = 0
   - When i=1: prints 1*2 = 2
   - When i=2: prints 2*2 = 4
   - When i=3: prints 3*2 = 6
   - When i=4: prints 4*2 = 8
3. So it prints the first 5 even numbers starting from 0""",
            "answer": "Prints 0, 2, 4, 6, 8 (the first 5 non-negative even numbers)"
        }
    ],
    "general": [
        {
            "question": "What would happen if the sun disappeared?",
            "reasoning": """Let me think through this step by step:
1. Immediately: We wouldn't notice for 8 minutes (light travel time)
2. After 8 minutes: Earth plunges into darkness
3. Temperature: Would drop rapidly, eventually reaching -270C
4. Orbit: Earth would travel in a straight line (no gravity pulling it)
5. Life: Most life would end within weeks without sunlight and heat
6. Atmosphere: Air would eventually freeze and fall as snow""",
            "answer": "Earth would go dark, freeze, and drift into space, ending most life"
        }
    ]
}

# Zero-shot prompts
ZERO_SHOT_PROMPTS = {
    "default": "Let's think step by step.",
    "detailed": "Let's work through this carefully, step by step, showing all reasoning.",
    "verify": "Let's solve this step by step, then verify our answer.",
    "simple": "Think step by step:"
}


class QuestionClassifier:
    """Classify questions by type."""
    
    PATTERNS = {
        "math": [
            r'\d+\s*[\+\-\*\/\%]\s*\d+',
            r'calculat|comput|solve|how\s+(?:much|many)',
            r'ratio|percent|fraction|decimal',
            r'equation|formula|expression'
        ],
        "logic": [
            r'if\s+.+\s+then',
            r'therefore|hence|conclude|implies',
            r'true|false|valid|invalid',
            r'all\s+.+\s+are|some\s+.+\s+are'
        ],
        "coding": [
            r'code|program|function|algorithm',
            r'python|java|javascript|c\+\+',
            r'debug|error|bug|fix',
            r'output|return|print'
        ]
    }
    
    def classify(self, question: str) -> str:
        """Classify question type."""
        question_lower = question.lower()
        
        for category, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return category
        
        return "general"


class CoTPromptBuilder:
    """Build chain-of-thought prompts."""
    
    def __init__(self, config: CoTConfig):
        self._config = config
        self._classifier = QuestionClassifier()
    
    def build_zero_shot(self, question: str) -> str:
        """Build zero-shot CoT prompt."""
        prompt = ZERO_SHOT_PROMPTS["default"]
        
        if self._config.enable_verification:
            prompt = ZERO_SHOT_PROMPTS["verify"]
        
        return f"{question}\n\n{prompt}"
    
    def build_few_shot(self, question: str) -> str:
        """Build few-shot CoT prompt with examples."""
        category = self._classifier.classify(question)
        examples = COT_EXAMPLES.get(category, COT_EXAMPLES["general"])
        
        # Limit examples
        examples = examples[:self._config.max_examples]
        
        prompt_parts = []
        
        for ex in examples:
            prompt_parts.append(f"Question: {ex['question']}")
            prompt_parts.append(ex['reasoning'])
            prompt_parts.append(f"Answer: {ex['answer']}")
            prompt_parts.append("")  # blank line
        
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Let me work through this step by step:")
        
        return "\n".join(prompt_parts)
    
    def build_plan_and_solve(self, question: str) -> str:
        """Build plan-and-solve prompt."""
        return f"""Question: {question}

Let's first understand the problem and devise a plan to solve it.

Plan:
1. Understand what is being asked
2. Identify the key information
3. Determine the steps needed
4. Execute each step
5. Verify the answer

Now let's execute this plan:"""
    
    def build_react(self, question: str) -> str:
        """Build ReAct (Reasoning + Acting) prompt."""
        return f"""Question: {question}

I will answer this by alternating between Thought, Action, and Observation.

Thought 1: Let me understand what is being asked.
"""
    
    def build(self, question: str) -> str:
        """Build prompt based on configured strategy."""
        strategy = self._config.strategy
        
        if strategy == CoTStrategy.ZERO_SHOT:
            return self.build_zero_shot(question)
        elif strategy == CoTStrategy.FEW_SHOT:
            return self.build_few_shot(question)
        elif strategy == CoTStrategy.PLAN_AND_SOLVE:
            return self.build_plan_and_solve(question)
        elif strategy == CoTStrategy.REACT:
            return self.build_react(question)
        else:
            return self.build_zero_shot(question)


class AnswerExtractor:
    """Extract final answer from CoT response."""
    
    ANSWER_PATTERNS = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)',
        r'therefore[,\s]+(.+?)(?:\.|$)',
        r'so[,\s]+(.+?)(?:\.|$)',
        r'thus[,\s]+(.+?)(?:\.|$)',
        r'=\s*(\d+(?:\.\d+)?)',
        r'result[:\s]+(.+?)(?:\.|$)'
    ]
    
    def extract(self, response: str) -> Optional[str]:
        """Extract final answer from response."""
        response_lower = response.lower()
        
        for pattern in self.ANSWER_PATTERNS:
            match = re.search(pattern, response_lower, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: return last sentence
        sentences = response.split('.')
        if sentences:
            return sentences[-1].strip() or sentences[-2].strip() if len(sentences) > 1 else None
        
        return None


class SelfConsistency:
    """Self-consistency with multiple reasoning paths."""
    
    def __init__(
        self,
        model: Any,
        num_paths: int = 3,
        temperature_range: Tuple[float, float] = (0.5, 0.9)
    ):
        self._model = model
        self._num_paths = num_paths
        self._temp_range = temperature_range
        self._extractor = AnswerExtractor()
    
    def generate_paths(
        self,
        prompt: str
    ) -> List[Tuple[str, str]]:
        """
        Generate multiple reasoning paths.
        
        Returns:
            List of (reasoning, answer) tuples
        """
        import random
        
        paths = []
        
        for i in range(self._num_paths):
            # Vary temperature
            temp = self._temp_range[0] + (self._temp_range[1] - self._temp_range[0]) * (i / self._num_paths)
            
            # Generate
            if hasattr(self._model, 'generate'):
                response = self._model.generate(prompt, temperature=temp)
            else:
                # Placeholder
                response = f"Path {i+1} reasoning..."
            
            answer = self._extractor.extract(response)
            paths.append((response, answer or ""))
        
        return paths
    
    def aggregate(
        self,
        paths: List[Tuple[str, str]]
    ) -> Tuple[str, float]:
        """
        Aggregate answers using majority voting.
        
        Returns:
            (most_common_answer, confidence)
        """
        from collections import Counter
        
        answers = [ans for _, ans in paths if ans]
        
        if not answers:
            return ("Unable to determine", 0.0)
        
        # Normalize answers for comparison
        normalized = [a.lower().strip() for a in answers]
        counts = Counter(normalized)
        
        most_common, count = counts.most_common(1)[0]
        confidence = count / len(answers)
        
        # Return original form
        for _, ans in paths:
            if ans and ans.lower().strip() == most_common:
                return (ans, confidence)
        
        return (most_common, confidence)


class TreeOfThought:
    """Tree-of-thought reasoning with branching."""
    
    def __init__(
        self,
        model: Any,
        evaluator: Optional[Callable[[str], float]] = None,
        branching_factor: int = 3,
        max_depth: int = 3
    ):
        self._model = model
        self._evaluator = evaluator or self._default_evaluator
        self._branching = branching_factor
        self._max_depth = max_depth
    
    def _default_evaluator(self, thought: str) -> float:
        """Simple evaluator based on thought quality signals."""
        score = 0.5
        
        # Positive signals
        if any(w in thought.lower() for w in ['because', 'since', 'therefore']):
            score += 0.1
        if any(w in thought.lower() for w in ['step', 'first', 'next', 'finally']):
            score += 0.1
        if re.search(r'\d', thought):  # Contains numbers
            score += 0.05
        
        # Negative signals
        if 'not sure' in thought.lower():
            score -= 0.1
        if len(thought) < 20:
            score -= 0.1
        
        return max(0, min(1, score))
    
    def expand_node(
        self,
        context: str,
        depth: int
    ) -> List[Tuple[str, float]]:
        """Expand a thought node into children."""
        if depth >= self._max_depth:
            return []
        
        prompt = f"{context}\n\nGenerate {self._branching} different next steps:"
        
        children = []
        for i in range(self._branching):
            # Generate thought
            if hasattr(self._model, 'generate'):
                thought = self._model.generate(
                    prompt,
                    temperature=0.7,
                    max_tokens=100
                )
            else:
                thought = f"Thought branch {i+1}"
            
            # Evaluate
            score = self._evaluator(thought)
            children.append((thought, score))
        
        return children
    
    def search(
        self,
        question: str,
        beam_width: int = 2
    ) -> str:
        """
        Search thought tree using beam search.
        
        Args:
            question: The question to answer
            beam_width: Number of paths to keep
            
        Returns:
            Best reasoning chain
        """
        # Initialize with question
        initial = f"Question: {question}\n\nLet me think about this step by step."
        
        beams = [(initial, 1.0)]
        
        for depth in range(self._max_depth):
            candidates = []
            
            for context, path_score in beams:
                children = self.expand_node(context, depth)
                
                for thought, thought_score in children:
                    new_context = f"{context}\n\nStep {depth + 1}: {thought}"
                    new_score = path_score * thought_score
                    candidates.append((new_context, new_score))
            
            if not candidates:
                break
            
            # Keep top beams
            candidates.sort(key=lambda x: -x[1])
            beams = candidates[:beam_width]
        
        # Return best path
        if beams:
            return beams[0][0]
        return initial


class CoTPrompt:
    """High-level chain-of-thought interface."""
    
    def __init__(self, config: Optional[CoTConfig] = None):
        self._config = config or CoTConfig()
        self._builder = CoTPromptBuilder(self._config)
        self._extractor = AnswerExtractor()
        self._classifier = QuestionClassifier()
    
    def apply(self, question: str) -> str:
        """
        Apply chain-of-thought to question.
        
        Args:
            question: Original question
            
        Returns:
            Enhanced prompt with CoT
        """
        return self._builder.build(question)
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract final answer from response."""
        return self._extractor.extract(response)
    
    def classify_question(self, question: str) -> str:
        """Classify question type."""
        return self._classifier.classify(question)
    
    def process(
        self,
        question: str,
        model: Any
    ) -> Dict[str, Any]:
        """
        Full CoT processing pipeline.
        
        Args:
            question: Question to answer
            model: Model for generation
            
        Returns:
            Dict with reasoning, answer, and metadata
        """
        # Build prompt
        prompt = self.apply(question)
        
        # Generate response
        if hasattr(model, 'generate'):
            response = model.generate(prompt)
        else:
            response = "Model generation not available"
        
        # Extract answer
        answer = self.extract_answer(response)
        
        return {
            "question": question,
            "prompt": prompt,
            "reasoning": response,
            "answer": answer,
            "question_type": self.classify_question(question),
            "strategy": self._config.strategy.value
        }


# Global instance
_prompter: Optional[CoTPrompt] = None


def get_cot_prompter(config: Optional[CoTConfig] = None) -> CoTPrompt:
    """Get or create global CoT prompter."""
    global _prompter
    if _prompter is None or config is not None:
        _prompter = CoTPrompt(config)
    return _prompter
