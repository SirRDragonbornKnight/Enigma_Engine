"""
Reasoning Chains for Enigma AI Engine

Chain-of-thought and structured reasoning.

Features:
- Step-by-step reasoning
- Self-verification
- Backtracking on errors
- Multiple reasoning strategies
- Confidence scoring

Usage:
    from enigma_engine.core.reasoning import ReasoningChain, ChainOfThought
    
    # Create reasoning chain
    chain = ChainOfThought(model)
    result = chain.solve("What is 15% of 240?")
    
    # With verification
    result = chain.solve("Complex problem", verify=True)
    
    # Access reasoning steps
    for step in result.steps:
        print(f"{step.number}: {step.thought}")
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ReasoningStrategy(Enum):
    """Reasoning strategies."""
    DIRECT = auto()         # Direct answer
    CHAIN_OF_THOUGHT = auto()  # Step-by-step
    TREE_OF_THOUGHT = auto()   # Branch and explore
    SELF_REFINE = auto()    # Generate then improve
    DEBATE = auto()         # Multiple perspectives
    VERIFICATION = auto()   # Generate and verify


@dataclass
class ThoughtStep:
    """A single reasoning step."""
    number: int
    thought: str
    action: Optional[str] = None
    result: Optional[str] = None
    confidence: float = 1.0
    is_error: bool = False
    branches: List["ThoughtStep"] = field(default_factory=list)


@dataclass
class ReasoningResult:
    """Result of reasoning process."""
    question: str
    answer: str
    steps: List[ThoughtStep]
    final_confidence: float
    strategy: ReasoningStrategy
    verified: bool = False
    verification_passed: bool = True
    reasoning_time: float = 0.0
    tokens_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "steps": [
                {
                    "number": s.number,
                    "thought": s.thought,
                    "action": s.action,
                    "result": s.result,
                    "confidence": s.confidence
                }
                for s in self.steps
            ],
            "confidence": self.final_confidence,
            "strategy": self.strategy.name,
            "verified": self.verified,
            "verification_passed": self.verification_passed
        }
    
    def format_chain(self) -> str:
        """Format as readable chain."""
        lines = [f"Question: {self.question}", ""]
        
        for step in self.steps:
            lines.append(f"Step {step.number}: {step.thought}")
            if step.action:
                lines.append(f"  Action: {step.action}")
            if step.result:
                lines.append(f"  Result: {step.result}")
            lines.append("")
        
        lines.append(f"Answer: {self.answer}")
        lines.append(f"Confidence: {self.final_confidence:.0%}")
        
        return "\n".join(lines)


class ChainOfThought:
    """
    Chain-of-thought reasoning engine.
    """
    
    COT_PROMPT = """Let's solve this step by step.

Question: {question}

Think through this carefully, showing each step of your reasoning.
After each step, state what you've determined.
Finally, give your answer.

Step 1:"""
    
    VERIFY_PROMPT = """Verify this reasoning:

Question: {question}

Proposed answer: {answer}

Reasoning steps:
{steps}

Is this reasoning correct? If not, identify the error and provide the correct answer.

Verification:"""
    
    def __init__(
        self,
        model: Any = None,
        max_steps: int = 10,
        min_confidence: float = 0.7
    ):
        """
        Initialize chain-of-thought reasoner.
        
        Args:
            model: Language model (EnigmaEngine or similar)
            max_steps: Maximum reasoning steps
            min_confidence: Minimum confidence threshold
        """
        self._model = model
        self._max_steps = max_steps
        self._min_confidence = min_confidence
    
    def solve(
        self,
        question: str,
        verify: bool = False,
        strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT
    ) -> ReasoningResult:
        """
        Solve a problem with reasoning.
        
        Args:
            question: The problem to solve
            verify: Whether to verify the answer
            strategy: Reasoning strategy to use
            
        Returns:
            Reasoning result with steps
        """
        start_time = time.time()
        
        if strategy == ReasoningStrategy.DIRECT:
            result = self._solve_direct(question)
        elif strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            result = self._solve_cot(question)
        elif strategy == ReasoningStrategy.TREE_OF_THOUGHT:
            result = self._solve_tot(question)
        elif strategy == ReasoningStrategy.SELF_REFINE:
            result = self._solve_refine(question)
        else:
            result = self._solve_cot(question)
        
        result.reasoning_time = time.time() - start_time
        result.strategy = strategy
        
        # Verify if requested
        if verify:
            result = self._verify(result)
        
        return result
    
    def _generate(self, prompt: str, **kwargs) -> str:
        """Generate text from model."""
        if self._model is None:
            # Fallback: simple mock for testing
            return "I don't have a model loaded."
        
        # Use model's generate method
        if hasattr(self._model, "generate"):
            return self._model.generate(prompt, **kwargs)
        elif hasattr(self._model, "chat"):
            return self._model.chat(prompt, **kwargs)
        else:
            return str(self._model(prompt))
    
    def _solve_direct(self, question: str) -> ReasoningResult:
        """Direct answer without reasoning steps."""
        answer = self._generate(question)
        
        return ReasoningResult(
            question=question,
            answer=answer,
            steps=[ThoughtStep(1, "Direct response")],
            final_confidence=0.8,
            strategy=ReasoningStrategy.DIRECT
        )
    
    def _solve_cot(self, question: str) -> ReasoningResult:
        """Chain-of-thought reasoning."""
        prompt = self.COT_PROMPT.format(question=question)
        
        full_response = self._generate(prompt, max_tokens=1024)
        
        # Parse steps from response
        steps = self._parse_steps(full_response)
        
        # Extract final answer
        answer = self._extract_answer(full_response)
        
        # Estimate confidence
        confidence = self._estimate_confidence(steps, full_response)
        
        return ReasoningResult(
            question=question,
            answer=answer,
            steps=steps,
            final_confidence=confidence,
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        )
    
    def _solve_tot(self, question: str, n_branches: int = 3) -> ReasoningResult:
        """Tree-of-thought reasoning with branching."""
        # Initial step
        initial_prompt = f"What are {n_branches} different approaches to solve: {question}"
        approaches = self._generate(initial_prompt, max_tokens=512)
        
        # Parse approaches
        approach_list = self._parse_list(approaches)[:n_branches]
        
        if not approach_list:
            approach_list = ["Think step by step"]
        
        # Explore each branch
        branches = []
        for i, approach in enumerate(approach_list):
            branch_prompt = f"""Question: {question}

Approach: {approach}

Using this approach, solve the problem step by step:"""
            
            branch_response = self._generate(branch_prompt, max_tokens=512)
            
            branch_step = ThoughtStep(
                number=i + 1,
                thought=approach,
                result=branch_response,
                confidence=0.8
            )
            branches.append(branch_step)
        
        # Select best branch
        best_branch = max(branches, key=lambda b: b.confidence)
        answer = self._extract_answer(best_branch.result or "")
        
        # Create steps with branches
        steps = [
            ThoughtStep(1, "Explored multiple approaches", branches=branches),
            ThoughtStep(2, f"Selected: {best_branch.thought}"),
            ThoughtStep(3, f"Reasoning: {best_branch.result}")
        ]
        
        return ReasoningResult(
            question=question,
            answer=answer,
            steps=steps,
            final_confidence=best_branch.confidence,
            strategy=ReasoningStrategy.TREE_OF_THOUGHT
        )
    
    def _solve_refine(self, question: str, n_iterations: int = 2) -> ReasoningResult:
        """Self-refine: generate then improve."""
        # Initial answer
        initial_answer = self._generate(question, max_tokens=512)
        
        steps = [ThoughtStep(1, "Initial answer", result=initial_answer)]
        current_answer = initial_answer
        
        # Refine iterations
        for i in range(n_iterations):
            refine_prompt = f"""Question: {question}

Current answer: {current_answer}

Review this answer. Identify any errors or areas for improvement.
Then provide an improved answer.

Feedback:"""
            
            refinement = self._generate(refine_prompt, max_tokens=512)
            
            # Extract improved answer
            improved = self._extract_after(refinement, ["improved answer:", "better answer:", "corrected:"])
            if not improved:
                improved = refinement
            
            steps.append(ThoughtStep(
                number=i + 2,
                thought="Refinement",
                action="Self-critique",
                result=improved
            ))
            
            current_answer = improved
        
        return ReasoningResult(
            question=question,
            answer=current_answer,
            steps=steps,
            final_confidence=0.85,
            strategy=ReasoningStrategy.SELF_REFINE
        )
    
    def _verify(self, result: ReasoningResult) -> ReasoningResult:
        """Verify a reasoning result."""
        # Format steps
        steps_text = "\n".join(
            f"Step {s.number}: {s.thought}" + (f" -> {s.result}" if s.result else "")
            for s in result.steps
        )
        
        prompt = self.VERIFY_PROMPT.format(
            question=result.question,
            answer=result.answer,
            steps=steps_text
        )
        
        verification = self._generate(prompt, max_tokens=512)
        
        # Check if verification passed
        verification_lower = verification.lower()
        passed = any(word in verification_lower for word in ["correct", "valid", "yes", "right"])
        failed = any(word in verification_lower for word in ["incorrect", "wrong", "error", "no"])
        
        result.verified = True
        result.verification_passed = passed and not failed
        
        # Add verification step
        result.steps.append(ThoughtStep(
            number=len(result.steps) + 1,
            thought="Verification",
            result=verification,
            confidence=1.0 if result.verification_passed else 0.5
        ))
        
        # Adjust confidence based on verification
        if not result.verification_passed:
            result.final_confidence *= 0.5
        
        return result
    
    def _parse_steps(self, text: str) -> List[ThoughtStep]:
        """Parse reasoning steps from text."""
        steps = []
        
        # Pattern: Step N: ... or numbered list
        step_pattern = r"(?:Step\s*)?(\d+)[:.]\s*(.+?)(?=(?:Step\s*)?\d+[:.]\s*|$)"
        matches = re.findall(step_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for num_str, thought in matches:
            num = int(num_str)
            thought = thought.strip()
            
            if thought:
                steps.append(ThoughtStep(
                    number=num,
                    thought=thought[:500]  # Truncate long thoughts
                ))
        
        # If no steps found, create a single step
        if not steps:
            steps = [ThoughtStep(1, text[:500])]
        
        return steps
    
    def _parse_list(self, text: str) -> List[str]:
        """Parse a list from text."""
        items = []
        
        # Pattern: numbered or bulleted list
        patterns = [
            r"^\d+[.)]\s*(.+)$",
            r"^[-*]\s*(.+)$",
            r"^[a-z][.)]\s*(.+)$"
        ]
        
        for line in text.split("\n"):
            line = line.strip()
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    items.append(match.group(1).strip())
                    break
        
        return items
    
    def _extract_answer(self, text: str) -> str:
        """Extract final answer from text."""
        # Look for explicit answer markers
        markers = ["answer:", "therefore:", "thus:", "result:", "conclusion:"]
        
        text_lower = text.lower()
        for marker in markers:
            idx = text_lower.rfind(marker)
            if idx >= 0:
                answer = text[idx + len(marker):].strip()
                # Take first sentence or line
                answer = answer.split("\n")[0].strip()
                if answer:
                    return answer
        
        # Fall back to last non-empty line
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            return lines[-1]
        
        return text.strip()[:200]
    
    def _extract_after(self, text: str, markers: List[str]) -> str:
        """Extract text after any of the markers."""
        text_lower = text.lower()
        
        for marker in markers:
            idx = text_lower.find(marker)
            if idx >= 0:
                return text[idx + len(marker):].strip()
        
        return ""
    
    def _estimate_confidence(self, steps: List[ThoughtStep], text: str) -> float:
        """Estimate confidence in the reasoning."""
        confidence = 0.7  # Base confidence
        
        # More steps = more thorough
        if len(steps) > 3:
            confidence += 0.1
        
        # Check for uncertainty language
        uncertainty_words = ["maybe", "perhaps", "might", "possibly", "unsure", "not sure"]
        text_lower = text.lower()
        
        for word in uncertainty_words:
            if word in text_lower:
                confidence -= 0.1
        
        # Check for confidence language
        confidence_words = ["therefore", "clearly", "definitely", "certainly"]
        for word in confidence_words:
            if word in text_lower:
                confidence += 0.05
        
        return max(0.1, min(1.0, confidence))


class ReasoningChain:
    """
    Composable reasoning chain for complex problems.
    """
    
    def __init__(self, model: Any = None):
        """Initialize chain."""
        self._model = model
        self._steps: List[Callable] = []
    
    def add_step(self, step: Callable[[str], str]) -> "ReasoningChain":
        """Add a reasoning step."""
        self._steps.append(step)
        return self
    
    def think(self) -> "ReasoningChain":
        """Add think step."""
        def _think(context: str) -> str:
            prompt = f"Think about this: {context}\n\nThoughts:"
            return self._generate(prompt)
        
        self._steps.append(_think)
        return self
    
    def analyze(self) -> "ReasoningChain":
        """Add analysis step."""
        def _analyze(context: str) -> str:
            prompt = f"Analyze: {context}\n\nAnalysis:"
            return self._generate(prompt)
        
        self._steps.append(_analyze)
        return self
    
    def plan(self) -> "ReasoningChain":
        """Add planning step."""
        def _plan(context: str) -> str:
            prompt = f"Make a plan for: {context}\n\nPlan:"
            return self._generate(prompt)
        
        self._steps.append(_plan)
        return self
    
    def execute(self) -> "ReasoningChain":
        """Add execution step."""
        def _execute(context: str) -> str:
            prompt = f"Execute: {context}\n\nExecution:"
            return self._generate(prompt)
        
        self._steps.append(_execute)
        return self
    
    def verify(self) -> "ReasoningChain":
        """Add verification step."""
        def _verify(context: str) -> str:
            prompt = f"Verify this is correct: {context}\n\nVerification:"
            return self._generate(prompt)
        
        self._steps.append(_verify)
        return self
    
    def run(self, input_text: str) -> Tuple[str, List[str]]:
        """
        Run the chain.
        
        Args:
            input_text: Initial input
            
        Returns:
            (final_result, intermediate_steps)
        """
        current = input_text
        history = [current]
        
        for step in self._steps:
            current = step(current)
            history.append(current)
        
        return current, history
    
    def _generate(self, prompt: str) -> str:
        """Generate text."""
        if self._model is None:
            return prompt
        
        if hasattr(self._model, "generate"):
            return self._model.generate(prompt)
        return str(self._model(prompt))


# Convenience function
def solve_with_reasoning(
    question: str,
    model: Any = None,
    strategy: str = "chain_of_thought",
    verify: bool = False
) -> ReasoningResult:
    """
    Solve a question with reasoning.
    
    Args:
        question: Question to solve
        model: Language model
        strategy: Reasoning strategy name
        verify: Whether to verify
        
    Returns:
        ReasoningResult
    """
    strategy_map = {
        "direct": ReasoningStrategy.DIRECT,
        "chain_of_thought": ReasoningStrategy.CHAIN_OF_THOUGHT,
        "cot": ReasoningStrategy.CHAIN_OF_THOUGHT,
        "tree_of_thought": ReasoningStrategy.TREE_OF_THOUGHT,
        "tot": ReasoningStrategy.TREE_OF_THOUGHT,
        "refine": ReasoningStrategy.SELF_REFINE,
        "self_refine": ReasoningStrategy.SELF_REFINE
    }
    
    strat = strategy_map.get(strategy.lower(), ReasoningStrategy.CHAIN_OF_THOUGHT)
    
    solver = ChainOfThought(model)
    return solver.solve(question, verify=verify, strategy=strat)
