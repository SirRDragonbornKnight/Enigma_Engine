"""
Self-Play Training for Enigma AI Engine

Train models through self-debate and self-play.

Features:
- AI debates itself
- Multi-turn self-play
- Quality scoring
- Preference learning
- Iterative improvement

Usage:
    from enigma_engine.core.self_play import SelfPlayTrainer
    
    trainer = SelfPlayTrainer(model)
    
    # Run self-play
    trainer.run_debate(topic="Should AI be regulated?")
    
    # Train on best responses
    trainer.train_on_preferences()
"""

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DebateRole(Enum):
    """Roles in a debate."""
    PROPONENT = "proponent"      # Argues for
    OPPONENT = "opponent"        # Argues against
    JUDGE = "judge"              # Evaluates


class ResponseQuality(Enum):
    """Quality of a response."""
    EXCELLENT = 5
    GOOD = 4
    ADEQUATE = 3
    POOR = 2
    BAD = 1


@dataclass
class DebateTurn:
    """A single turn in a debate."""
    role: DebateRole
    content: str
    quality_score: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class DebateResult:
    """Result of a debate session."""
    topic: str
    turns: List[DebateTurn]
    winner: Optional[DebateRole] = None
    judge_reasoning: str = ""
    training_pairs: List[Tuple[str, str, float]] = field(default_factory=list)


@dataclass
class SelfPlayConfig:
    """Self-play configuration."""
    # Debate settings
    max_turns: int = 6
    min_turns: int = 4
    
    # Model settings
    temperature: float = 0.7
    diverse_sampling: bool = True
    
    # Training settings
    learning_rate: float = 1e-5
    preference_margin: float = 0.1
    
    # Quality thresholds
    min_quality: float = 0.5
    excellent_threshold: float = 0.8


class ResponseGenerator:
    """Generate diverse responses for self-play."""
    
    def __init__(self, model: nn.Module, tokenizer: Any):
        """
        Initialize generator.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self._device = next(model.parameters()).device
    
    def generate(
        self,
        prompt: str,
        role: DebateRole,
        temperature: float = 0.7,
        num_samples: int = 1
    ) -> List[str]:
        """
        Generate response(s) for a role.
        
        Args:
            prompt: Context prompt
            role: Debate role
            temperature: Sampling temperature
            num_samples: Number of samples
            
        Returns:
            List of generated responses
        """
        # Add role instruction
        role_prompts = {
            DebateRole.PROPONENT: "Argue IN FAVOR of the following position:",
            DebateRole.OPPONENT: "Argue AGAINST the following position:",
            DebateRole.JUDGE: "As an impartial judge, evaluate the arguments:"
        }
        
        full_prompt = f"{role_prompts[role]}\n\n{prompt}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        responses = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0, inputs["input_ids"].size(1):],
                skip_special_tokens=True
            )
            responses.append(response.strip())
        
        return responses


class ResponseScorer:
    """Score response quality."""
    
    def __init__(
        self,
        judge_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize scorer.
        
        Args:
            judge_model: Model to use as judge (or None for heuristic)
            tokenizer: Tokenizer
        """
        self.judge_model = judge_model
        self.tokenizer = tokenizer
    
    def score(
        self,
        response: str,
        context: str,
        role: DebateRole
    ) -> float:
        """
        Score a response.
        
        Args:
            response: Response to score
            context: Debate context
            role: Response role
            
        Returns:
            Quality score (0-1)
        """
        if self.judge_model and self.tokenizer:
            return self._model_score(response, context, role)
        else:
            return self._heuristic_score(response, context, role)
    
    def _model_score(
        self,
        response: str,
        context: str,
        role: DebateRole
    ) -> float:
        """Score using judge model."""
        prompt = f"""Rate this argument on a scale of 1-5:

Context: {context}
Argument: {response}

Scoring criteria:
- Logical coherence
- Relevant evidence
- Clear reasoning
- Persuasiveness

Score (1-5):"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.judge_model.generate(
                inputs["input_ids"],
                max_new_tokens=5,
                temperature=0.1
            )
        
        score_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract numeric score
        import re
        match = re.search(r'[1-5]', score_text)
        if match:
            return int(match.group()) / 5.0
        
        return 0.5
    
    def _heuristic_score(
        self,
        response: str,
        context: str,
        role: DebateRole
    ) -> float:
        """Score using heuristics."""
        score = 0.5
        
        # Length check (too short is bad)
        if len(response) < 50:
            score -= 0.2
        elif len(response) > 100:
            score += 0.1
        
        # Argument markers
        argument_words = [
            'because', 'therefore', 'however', 'furthermore',
            'evidence', 'research', 'shows', 'proves'
        ]
        response_lower = response.lower()
        
        for word in argument_words:
            if word in response_lower:
                score += 0.05
        
        # Coherence check (simple)
        sentences = response.split('.')
        if len(sentences) >= 2:
            score += 0.1
        
        # Relevance (mentions topic)
        context_words = set(context.lower().split())
        response_words = set(response_lower.split())
        overlap = len(context_words & response_words)
        score += min(0.2, overlap * 0.02)
        
        return min(1.0, max(0.0, score))


class SelfPlayTrainer:
    """Train model through self-play."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[SelfPlayConfig] = None
    ):
        """
        Initialize self-play trainer.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer
            config: Configuration
        """
        self.config = config or SelfPlayConfig()
        self.model = model
        self.tokenizer = tokenizer
        
        # Components
        self._generator = ResponseGenerator(model, tokenizer)
        self._scorer = ResponseScorer()
        
        # Training data
        self._preference_pairs: List[Tuple[str, str, str, float]] = []
        
        # History
        self._debates: List[DebateResult] = []
        
        # Optimizer
        self._optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate
        )
    
    def run_debate(
        self,
        topic: str,
        num_turns: Optional[int] = None
    ) -> DebateResult:
        """
        Run a debate on a topic.
        
        Args:
            topic: Debate topic
            num_turns: Number of turns
            
        Returns:
            Debate result
        """
        num_turns = num_turns or random.randint(
            self.config.min_turns,
            self.config.max_turns
        )
        
        turns = []
        context = f"Topic: {topic}\n\n"
        
        # Alternate between roles
        roles = [DebateRole.PROPONENT, DebateRole.OPPONENT]
        
        for i in range(num_turns):
            role = roles[i % 2]
            
            # Generate response
            responses = self._generator.generate(
                prompt=context,
                role=role,
                temperature=self.config.temperature,
                num_samples=1
            )
            
            response = responses[0]
            
            # Score response
            score = self._scorer.score(response, context, role)
            
            turn = DebateTurn(
                role=role,
                content=response,
                quality_score=score
            )
            turns.append(turn)
            
            # Update context
            role_name = "Pro" if role == DebateRole.PROPONENT else "Con"
            context += f"\n{role_name}: {response}\n"
        
        # Judge the debate
        winner, reasoning = self._judge_debate(topic, turns)
        
        # Extract training pairs
        training_pairs = self._extract_training_pairs(topic, turns)
        
        result = DebateResult(
            topic=topic,
            turns=turns,
            winner=winner,
            judge_reasoning=reasoning,
            training_pairs=training_pairs
        )
        
        self._debates.append(result)
        
        # Add to preference data
        self._add_preference_data(result)
        
        return result
    
    def run_self_play(
        self,
        prompts: List[str],
        responses_per_prompt: int = 2
    ) -> List[Tuple[str, str, float]]:
        """
        Run self-play on prompts.
        
        Args:
            prompts: List of prompts
            responses_per_prompt: Number of responses per prompt
            
        Returns:
            List of (prompt, response, score) tuples
        """
        results = []
        
        for prompt in prompts:
            # Generate multiple responses
            responses = self._generator.generate(
                prompt=prompt,
                role=DebateRole.PROPONENT,  # Using as general generation
                temperature=self.config.temperature,
                num_samples=responses_per_prompt
            )
            
            # Score each response
            scored = []
            for response in responses:
                score = self._scorer.score(response, prompt, DebateRole.PROPONENT)
                scored.append((response, score))
                results.append((prompt, response, score))
            
            # Create preference pairs from best vs worst
            sorted_responses = sorted(scored, key=lambda x: x[1], reverse=True)
            
            if len(sorted_responses) >= 2:
                best = sorted_responses[0]
                worst = sorted_responses[-1]
                
                if best[1] - worst[1] > self.config.preference_margin:
                    self._preference_pairs.append((
                        prompt,
                        best[0],
                        worst[0],
                        best[1] - worst[1]
                    ))
        
        return results
    
    def train_on_preferences(
        self,
        epochs: int = 1,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Train model on collected preferences.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training statistics
        """
        if not self._preference_pairs:
            logger.warning("No preference pairs collected")
            return {"loss": 0.0}
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            # Shuffle pairs
            random.shuffle(self._preference_pairs)
            
            for i in range(0, len(self._preference_pairs), batch_size):
                batch = self._preference_pairs[i:i + batch_size]
                
                loss = self._train_batch(batch)
                total_loss += loss
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            "loss": avg_loss,
            "num_pairs": len(self._preference_pairs),
            "epochs": epochs
        }
    
    def _train_batch(
        self,
        batch: List[Tuple[str, str, str, float]]
    ) -> float:
        """Train on a batch of preference pairs."""
        self._optimizer.zero_grad()
        
        total_loss = 0.0
        
        for prompt, chosen, rejected, margin in batch:
            # Get log probs for chosen
            chosen_input = self.tokenizer(
                f"{prompt}\n{chosen}",
                return_tensors="pt"
            )
            
            # Get log probs for rejected
            rejected_input = self.tokenizer(
                f"{prompt}\n{rejected}",
                return_tensors="pt"
            )
            
            # Forward pass
            with torch.set_grad_enabled(True):
                chosen_out = self.model(**chosen_input)
                rejected_out = self.model(**rejected_input)
                
                # Get logits
                if hasattr(chosen_out, 'logits'):
                    chosen_logits = chosen_out.logits
                    rejected_logits = rejected_out.logits
                else:
                    chosen_logits = chosen_out
                    rejected_logits = rejected_out
                
                # Simple preference loss: maximize chosen, minimize rejected
                chosen_score = chosen_logits.mean()
                rejected_score = rejected_logits.mean()
                
                loss = -torch.log(
                    torch.sigmoid(chosen_score - rejected_score)
                )
                
                total_loss += loss.item()
                loss.backward()
        
        self._optimizer.step()
        
        return total_loss / len(batch)
    
    def _judge_debate(
        self,
        topic: str,
        turns: List[DebateTurn]
    ) -> Tuple[Optional[DebateRole], str]:
        """Judge a debate."""
        pro_score = sum(
            t.quality_score for t in turns
            if t.role == DebateRole.PROPONENT
        )
        con_score = sum(
            t.quality_score for t in turns
            if t.role == DebateRole.OPPONENT
        )
        
        if pro_score > con_score + 0.1:
            winner = DebateRole.PROPONENT
            reasoning = f"Proponent wins with stronger arguments (score: {pro_score:.2f} vs {con_score:.2f})"
        elif con_score > pro_score + 0.1:
            winner = DebateRole.OPPONENT
            reasoning = f"Opponent wins with stronger arguments (score: {con_score:.2f} vs {pro_score:.2f})"
        else:
            winner = None
            reasoning = f"Debate is a draw (scores: {pro_score:.2f} vs {con_score:.2f})"
        
        return winner, reasoning
    
    def _extract_training_pairs(
        self,
        topic: str,
        turns: List[DebateTurn]
    ) -> List[Tuple[str, str, float]]:
        """Extract training pairs from debate."""
        pairs = []
        
        context = f"Topic: {topic}"
        
        for turn in turns:
            pairs.append((context, turn.content, turn.quality_score))
            context += f"\n{turn.role.value}: {turn.content}"
        
        return pairs
    
    def _add_preference_data(self, result: DebateResult):
        """Add debate result to preference data."""
        # Find best and worst turns
        if len(result.turns) < 2:
            return
        
        sorted_turns = sorted(
            result.turns,
            key=lambda t: t.quality_score,
            reverse=True
        )
        
        best = sorted_turns[0]
        worst = sorted_turns[-1]
        
        margin = best.quality_score - worst.quality_score
        
        if margin > self.config.preference_margin:
            self._preference_pairs.append((
                result.topic,
                best.content,
                worst.content,
                margin
            ))
    
    def get_statistics(self) -> Dict:
        """Get training statistics."""
        if not self._debates:
            return {}
        
        wins = {DebateRole.PROPONENT: 0, DebateRole.OPPONENT: 0, None: 0}
        
        for debate in self._debates:
            wins[debate.winner] += 1
        
        avg_turns = sum(len(d.turns) for d in self._debates) / len(self._debates)
        
        return {
            "total_debates": len(self._debates),
            "proponent_wins": wins[DebateRole.PROPONENT],
            "opponent_wins": wins[DebateRole.OPPONENT],
            "draws": wins[None],
            "average_turns": avg_turns,
            "preference_pairs": len(self._preference_pairs)
        }


# Global trainer
_trainer: Optional[SelfPlayTrainer] = None


def get_self_play_trainer(
    model: Optional[nn.Module] = None,
    tokenizer: Optional[Any] = None
) -> Optional[SelfPlayTrainer]:
    """Get or create global self-play trainer."""
    global _trainer
    
    if _trainer is None and model is not None and tokenizer is not None:
        _trainer = SelfPlayTrainer(model, tokenizer)
    
    return _trainer
